import wandb
import numpy as np
from envs import my_matc_env
from PPO_attention import PPO_Agent
from PPO_attention import Memory
from PPO_attention import compute_gae, discount
from PPO_attention import schedule_beta
import bluesky as bs
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
from envs.scenario.utils import haversine_distance, normalize
import csv
import os
import time
import random
import pickle
import pandas as pd
import argparse

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")

MAX_NUM_AGENTS = 20
NUM_INTRUDERS = 3

STATE_SIZE = 6
INTRUDER_STATE_SIZE = 5
ACTION_SIZE = 3

NUM_EPISODES = 3000
BATCH_SIZE = 512
EPOCHS = 10
ALPHA = 0.2
BETA = 1e-3
INITIAL_BETA = 0.01
FINAL_BETA = 0.001
LR = 1e-4
POLICY_UPDATE_TIME_STEP = 1024
SHUFFLE = True

# Example usage
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="Unique name for this run")
    args = parser.parse_args()


    TRAINING = True
    WANDB = False
    RENDER_MODE = 'sim'

    DATE = args.date
    print(f"Run Date: {DATE}")
    # Model name
    MODEL_NAME_COMPANY1 = f'best_policy_CoA_{DATE}.pt'
    MODEL_NAME_COMPANY2 = f'best_policy_CoB_{DATE}.pt'
    WANDB_NAME = f'Training-{DATE}'

    MAX_STEPS_SO_FAR = 20000

    # Set the random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    # set the random for mps 
    # if torch.backends.mps.is_available():
    #     torch.manual_seed(42)
    #     torch.backends.mps.fallback_random = True

    # Initialize the PPO agent
    state_size = STATE_SIZE 
    context_size = INTRUDER_STATE_SIZE
    action_size = ACTION_SIZE  
    num_intruders = NUM_INTRUDERS  
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    num_agents = MAX_NUM_AGENTS
    numEpisodes = NUM_EPISODES

    # Initializing the environment
    env = my_matc_env.MATCEnv('routes/matc/routes_two_companies.json', render_mode=RENDER_MODE)

    # Company list
    companylist = ['CoA', 'CoB']

    # Initialize the PPO agent
    CoA_agent = PPO_Agent(state_size=state_size, context_size=context_size, action_size=action_size, num_intruders=num_intruders)
    CoB_agent = PPO_Agent(state_size=state_size, context_size=context_size, action_size=action_size, num_intruders=num_intruders)

    device = CoA_agent.device

    if not TRAINING:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Load the best model
        CoA_agent.load(name=f'{script_dir}/trained_models/best_policy_CoA_Aug16.pt')
        CoB_agent.load(name=f'{script_dir}/trained_models/best_policy_CoB_Aug16.pt')

    # Initialize the memory
    memory_CoA = Memory()
    memory_CoB = Memory()

    # Initialize wandb
    if WANDB:
        wandb.init(project="ATC-MARL-CoA+CoB", name=WANDB_NAME, config={
            "state_size": state_size,
            "action_size": action_size,
            "num_intruders": num_intruders,
            "num_agents": num_agents,
            "numEpisodes": numEpisodes,
            "batch_size": batch_size,
            "epochs": epochs
        })

    if TRAINING:  
        best_avg_rewards_CoA = float('-inf')  # Initialize the best average rewards
        best_avg_rewards_CoB = float('-inf')  # Initialize the best average rewards

    # Initialize Dataset
    dataframe_name = f'dataset_{DATE}.csv'
    mydataframe = {
                "N_success": [],
                "Collisions": [],
                "Avg Rewards": [],
                "Avg Distance Rewards": [],
                "Avg Action Rewards": [],
                "Avg Time Rewards": [],
                "Avg Mission Acc Rewards": [],
                "Avg Velocity Rewards": [],
                "Entropy": [],
                "KL Divergence": [],
                "Average Loss": [],
                "Actions": [],
                "Avg Speed": [],
            }

    # Training loop
    for episode in range(numEpisodes):

        # Starting time for the episode
        start_time = time.time()

        rewards = {}
        processed_states, infos = env.reset()
        dones = {agent: 0 for agent in processed_states.keys()}  # Initialize done states for all agents
        episode_actions = {agent: [] for agent in processed_states.keys()}
        # for i in range(episode_length):
        counter = 0
        episodic_losses_CoA = []
        episodic_losses_CoB = []
        entropies_CoA = []
        entropies_CoB = []
        kl_divergences_CoA = []
        kl_divergences_CoB = []
        training_steps = 0
        collided_agents_info = {}

        # update beta 
        beta_CoA = schedule_beta(episode=episode, decay_episode=1000, initial_beta=INITIAL_BETA, final_beta=FINAL_BETA)
        beta_CoB = schedule_beta(episode=episode, decay_episode=1000, initial_beta=INITIAL_BETA, final_beta=FINAL_BETA)

        # print(f"Beta: {beta}")
        END_EPISODE = False
        while not END_EPISODE: 

            CoA_agent.model.to(torch.device('cpu'))
            CoB_agent.model.to(torch.device('cpu'))

            Actions = {}
            LogProbs = {}

            # Prepare batched inputs for all agents
            agent_keys = list(processed_states.keys())

            # if len(agent_keys) == 0:
            #     print("No agents left in the environment. Ending episode.")
            #     break

            # extract company 1 and company 2 agents
            agent_keys_CoA = [agent for agent in agent_keys if agent.startswith('AA') or agent.startswith('AC')]
            agent_keys_CoB = [agent for agent in agent_keys if agent.startswith('AB') or agent.startswith('AD')]

            # Collect all states and contexts as lists (CPU-side)
            all_states_CoA = []
            all_states_CoB = []
            all_contexts_CoA = []
            all_contexts_CoB = []

            for ownship_agent in agent_keys:
                ownship_state = np.array([value for _, value in processed_states[ownship_agent]['ownship_state'].items()], dtype=np.float32)
                intruder_states = processed_states[ownship_agent]['intruder_states'].astype(np.float32)
                # if ownship_agent == 'AD0003':
                #     print(f"Ownship Agent: {ownship_agent}, State: {ownship_state}, Intruder States: {intruder_states}")

                if ownship_agent in agent_keys_CoA:
                    all_states_CoA.append(ownship_state)
                    all_contexts_CoA.append(intruder_states)
                elif ownship_agent in agent_keys_CoB:
                    all_states_CoB.append(ownship_state)
                    all_contexts_CoB.append(intruder_states)

            # Ensure all_states and all_contexts have the correct shape
            # Company #1
            if len(all_states_CoA) == 0:
                batched_states_CoA = []
                batched_contexts_CoA = []
            elif len(all_states_CoA) == 1:
                batched_states_CoA = torch.tensor(all_states_CoA[0], dtype=torch.float32).unsqueeze(0)
                batched_contexts_CoA = torch.tensor(all_contexts_CoA[0], dtype=torch.float32).unsqueeze(0)
            else:
                batched_states_CoA = torch.tensor(np.vstack(all_states_CoA), dtype=torch.float32)
                batched_contexts_CoA = torch.tensor(np.stack(all_contexts_CoA), dtype=torch.float32)

            # Company #2
            if len(all_states_CoB) == 0:
                batched_states_CoB = []
                batched_contexts_CoB = []
            elif len(all_states_CoB) == 1:
                batched_states_CoB = torch.tensor(all_states_CoB[0], dtype=torch.float32).unsqueeze(0)
                batched_contexts_CoB = torch.tensor(all_contexts_CoB[0], dtype=torch.float32).unsqueeze(0)
            else:
                batched_states_CoB = torch.tensor(np.vstack(all_states_CoB), dtype=torch.float32)
                batched_contexts_CoB = torch.tensor(np.stack(all_contexts_CoB), dtype=torch.float32)

            # Get action probabilities for all agents in a single call
            if TRAINING:
                with torch.no_grad():
                    # Company #1
                    if len(batched_states_CoA) > 0:
                        policies_CoA, old_values_CoA = CoA_agent.model(batched_states_CoA, batched_contexts_CoA)
                        dists_CoA = torch.distributions.Categorical(policies_CoA)
                        actions_CoA = dists_CoA.sample()
                        logprobs_CoA = dists_CoA.log_prob(actions_CoA)
                    # Company #2
                    if len(batched_states_CoB) > 0:
                        policies_CoB, old_values_CoB = CoB_agent.model(batched_states_CoB, batched_contexts_CoB)
                        dists_CoB = torch.distributions.Categorical(policies_CoB)
                        actions_CoB = dists_CoB.sample()
                        logprobs_CoB = dists_CoB.log_prob(actions_CoB)
            else: # Take the action with the highest probability
                with torch.no_grad():
                    # Company #1
                    if len(batched_states_CoA) > 0:
                        policies_CoA, old_values_CoA = CoA_agent.model(batched_states_CoA, batched_contexts_CoA)
                        actions_CoA = torch.argmax(policies_CoA, dim=1)
                        logprobs_CoA = torch.zeros_like(actions_CoA, dtype=torch.float32)
                    # Company #2
                    if len(batched_states_CoB) > 0:
                        policies_CoB, old_values_CoB = CoB_agent.model(batched_states_CoB, batched_contexts_CoB)
                        actions_CoB = torch.argmax(policies_CoB, dim=1)
                        logprobs_CoB = torch.zeros_like(actions_CoB, dtype=torch.float32)

            # Assign actions, log probabilities, and old values back to individual agents
            # Company #1
            old_values_dict_CoA = {}
            for idx, ownship_agent in enumerate(agent_keys_CoA):
                Actions[ownship_agent] = actions_CoA[idx].item()
                LogProbs[ownship_agent] = logprobs_CoA[idx].detach()
                old_values_dict_CoA[ownship_agent] = old_values_CoA[idx].detach()
            # Company #2
            old_values_dict_CoB = {}
            for idx, ownship_agent in enumerate(agent_keys_CoB):
                Actions[ownship_agent] = actions_CoB[idx].item()
                LogProbs[ownship_agent] = logprobs_CoB[idx].detach()
                old_values_dict_CoB[ownship_agent] = old_values_CoB[idx].detach()

            for ownship_agent in processed_states.keys():
                if dones.get(ownship_agent, False):
                    # drop the agent from Actions, LogProbs, and old_values_dict
                    Actions.pop(ownship_agent, None)
                    LogProbs.pop(ownship_agent, None)
                    old_values_dict_CoA.pop(ownship_agent, None)
                    old_values_dict_CoB.pop(ownship_agent, None)

            # Step the environment
            next_processed_states, rewards, terminations, truncations, dones = env.step(Actions)
            # print(f"Rewards: {rewards}")
            # print(all(env.dones.values()))

            # Check for collisions
            collided_agents = env.collided_agents
            num_of_collisions = len(env.collided_agents)

            # === [5] Store experience into memory efficiently ===
            # Company #1
            for idx, ownship_agent in enumerate(agent_keys_CoA):
                if ownship_agent in Actions.keys():
                    reward = rewards[ownship_agent]
                    done = dones.get(ownship_agent, 0)
                    # if reward <= -1 or reward > 0:
                    #     done = 1
                    #     dones[ownship_agent] = 1  # Ensure done is set to 1 if the agent has collided

                    # Use precomputed tensors (slices from the batched input)
                    state = batched_states_CoA[idx].unsqueeze(0).detach()
                    context = batched_contexts_CoA[idx].unsqueeze(0).detach()
                    action = Actions[ownship_agent]
                    logprob = LogProbs[ownship_agent].detach()
                    old_value = old_values_dict_CoA[ownship_agent].detach()

                    # Store into memory
                    if reward is not None:
                        memory_CoA.add(ownship_agent, state, context, action, reward, done, logprob, old_value)
            # Company #2
            for idx, ownship_agent in enumerate(agent_keys_CoB):
                if ownship_agent in Actions.keys():
                    reward = rewards[ownship_agent]
                    done = dones.get(ownship_agent, 0)
                    # if reward <= -1 or reward > 0:
                    #     done = 1
                    #     dones[ownship_agent] = 1  # Ensure done is set to 1 if the agent has collided

                    # Use precomputed tensors (slices from the batched input)
                    state = batched_states_CoB[idx].unsqueeze(0).detach()
                    context = batched_contexts_CoB[idx].unsqueeze(0).detach()
                    action = Actions[ownship_agent]
                    logprob = LogProbs[ownship_agent].detach()
                    old_value = old_values_dict_CoB[ownship_agent].detach()

                    # Store into memory
                    if reward is not None:
                        memory_CoB.add(ownship_agent, state, context, action, reward, done, logprob, old_value)

            for agent in processed_states.keys():
                if agent in episode_actions.keys():
                    if agent in Actions.keys():
                        episode_actions[agent].append(Actions[agent])
                else:
                    if agent in Actions.keys():
                        episode_actions[agent] = [Actions[agent]]

            # Update the observations
            processed_states = next_processed_states.copy()

            # Calculate the total length of all states in memory
            # Increment the counter
            counter += 1

            # collided_agent_list = [item for tup in env.collided_agents for item in tup]

            if TRAINING and (all(env.dones.values()) or counter >= MAX_STEPS_SO_FAR):

                # print("Training the PPO agent...")

                # Move the models to the device
                CoA_agent.model.to(device)
                CoB_agent.model.to(device)
                
                # print("Training the PPO agent...")
                # Training the PPO agent
                # Check if all agents are done

                for company in companylist:
                    if company == 'CoA':
                        memory = memory_CoA
                        beta = beta_CoA
                        # agent data
                        agent_data = memory.agent_data
                        all_states, all_contexts, all_actions, all_rewards, all_dones = [], [], [], [], []
                        all_advantages, all_returns, all_logprobs = [], [], []

                        # Company #1 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_states = torch.stack(agent_data[agent]['states']).to(device)
                            agent_contexts = torch.stack(agent_data[agent]['contexts']).to(device)
                            agent_actions = torch.tensor(agent_data[agent]['actions'], dtype=torch.long, device=device)
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_dones = torch.tensor(agent_data[agent]['dones'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_logprobs = torch.tensor(agent_data[agent]['log_probs'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_old_values = torch.stack(agent_data[agent]['old_values']).to(device)  # Use stored old values

                            # Compute GAE and returns for the current agent
                            advantages, returns = compute_gae(agent_rewards, agent_old_values, agent_dones)

                            # Append the current agent's data to the combined lists
                            all_states.append(agent_states)
                            all_contexts.append(agent_contexts)
                            all_actions.append(agent_actions)
                            all_rewards.append(agent_rewards)
                            all_dones.append(agent_dones)
                            all_advantages.append(advantages)
                            all_returns.append(returns)
                            all_logprobs.append(agent_logprobs)                

                        states = torch.cat(all_states, dim=0).to(device)
                        contexts = torch.cat(all_contexts, dim=0).to(device)
                        actions = torch.cat(all_actions, dim=0).to(device)
                        rewards = torch.cat(all_rewards, dim=0).to(device)
                        dones_ = torch.cat(all_dones, dim=0).to(device)
                        advantages = torch.cat(all_advantages, dim=0).to(device)
                        returns = torch.cat(all_returns, dim=0).to(device)
                        logprobs = torch.cat(all_logprobs, dim=0).to(device)

                        # Normalize advantages for the current agent
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # Create dataloader for minibatch training
                        dataset = TensorDataset(states, contexts, actions, logprobs, returns, advantages)
                        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

                        # PPO training loop
                        for _ in range(EPOCHS):
                            for s_b, c_b, a_b, old_lp_b, ret_b, adv_b in loader:
                                CoA_agent.optimizer.zero_grad()
                                s_b = s_b.view(-1, STATE_SIZE)
                                c_b = c_b.view(-1, NUM_INTRUDERS, INTRUDER_STATE_SIZE)

                                new_policies, new_values = CoA_agent.model(s_b, c_b)
                                dist = torch.distributions.Categorical(new_policies)

                                old_lp_b = old_lp_b.view(-1)
                                new_logprobs = dist.log_prob(a_b).view(-1)

                                # PPO ratio and losses
                                ratio = (new_logprobs - old_lp_b).exp()
                                surr1 = ratio * adv_b.view(-1)
                                surr2 = torch.clamp(ratio, 1 - ALPHA, 1 + ALPHA) * adv_b.view(-1)
                                policy_loss = -torch.min(surr1, surr2).mean()

                                value_loss = F.mse_loss(new_values, ret_b)
                                entropy = dist.entropy().mean()

                                loss = policy_loss + 0.5 * value_loss - beta * entropy
                                loss.backward() 
                                CoA_agent.optimizer.step()
                                training_steps += 1
                                episodic_losses_CoA.append(loss.item())

                                entropies_CoA.append(entropy.item())
                                kl_divergences_CoA.append((old_lp_b - new_logprobs).mean().item())

                        # Compute avg rewards for company 1
                        avg_rewards_all_agents_CoA = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                    elif company == 'CoB':
                        memory = memory_CoB
                        beta = beta_CoB
                        # agent data
                        agent_data = memory.agent_data
                        all_states, all_contexts, all_actions, all_rewards, all_dones = [], [], [], [], []
                        all_advantages, all_returns, all_logprobs = [], [], []

                        # Company #2 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_states = torch.stack(agent_data[agent]['states']).to(device)
                            agent_contexts = torch.stack(agent_data[agent]['contexts']).to(device)
                            agent_actions = torch.tensor(agent_data[agent]['actions'], dtype=torch.long, device=device)
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_dones = torch.tensor(agent_data[agent]['dones'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_logprobs = torch.tensor(agent_data[agent]['log_probs'], dtype=torch.float32, device=device).reshape(-1, 1)
                            agent_old_values = torch.stack(agent_data[agent]['old_values']).to(device)  # Use stored old values

                            # Compute GAE and returns for the current agent
                            advantages, returns = compute_gae(agent_rewards, agent_old_values, agent_dones)

                            # Append the current agent's data to the combined lists
                            all_states.append(agent_states)
                            all_contexts.append(agent_contexts)
                            all_actions.append(agent_actions)
                            all_rewards.append(agent_rewards)
                            all_dones.append(agent_dones)
                            all_advantages.append(advantages)
                            all_returns.append(returns)
                            all_logprobs.append(agent_logprobs)                

                        states = torch.cat(all_states, dim=0).to(device)
                        contexts = torch.cat(all_contexts, dim=0).to(device)
                        actions = torch.cat(all_actions, dim=0).to(device)
                        rewards = torch.cat(all_rewards, dim=0).to(device)
                        dones_ = torch.cat(all_dones, dim=0).to(device)
                        advantages = torch.cat(all_advantages, dim=0).to(device)
                        returns = torch.cat(all_returns, dim=0).to(device)
                        logprobs = torch.cat(all_logprobs, dim=0).to(device)

                        # Normalize advantages for the current agent
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # Create dataloader for minibatch training
                        dataset = TensorDataset(states, contexts, actions, logprobs, returns, advantages)
                        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

                        # PPO training loop
                        for _ in range(EPOCHS):
                            for s_b, c_b, a_b, old_lp_b, ret_b, adv_b in loader:
                                CoB_agent.optimizer.zero_grad()
                                s_b = s_b.view(-1, STATE_SIZE)
                                c_b = c_b.view(-1, NUM_INTRUDERS, INTRUDER_STATE_SIZE)

                                new_policies, new_values = CoB_agent.model(s_b, c_b)
                                dist = torch.distributions.Categorical(new_policies)

                                old_lp_b = old_lp_b.view(-1)
                                new_logprobs = dist.log_prob(a_b).view(-1)

                                # PPO ratio and losses
                                ratio = (new_logprobs - old_lp_b).exp()
                                surr1 = ratio * adv_b.view(-1)
                                surr2 = torch.clamp(ratio, 1 - ALPHA, 1 + ALPHA) * adv_b.view(-1)
                                policy_loss = -torch.min(surr1, surr2).mean()

                                value_loss = F.mse_loss(new_values, ret_b)
                                entropy = dist.entropy().mean()

                                loss = policy_loss + 0.5 * value_loss - beta * entropy
                                loss.backward() 
                                CoB_agent.optimizer.step()
                                training_steps += 1
                                episodic_losses_CoB.append(loss.item())

                                entropies_CoB.append(entropy.item())
                                kl_divergences_CoB.append((old_lp_b - new_logprobs).mean().item())

                        # Compute avg rewards for company 2
                        avg_rewards_all_agents_CoB = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                # Reset the environment
                END_EPISODE = True

            if not TRAINING and (all(env.dones.values()) or counter >= MAX_STEPS_SO_FAR):
                
                for company in companylist:
                    if company == 'CoA':
                        memory = memory_CoA
                        beta = beta_CoA
                        # agent data
                        agent_data = memory.agent_data
                        all_rewards = []

                        # Company #1 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'][:-1], dtype=torch.float32).reshape(-1, 1) 
                            all_rewards.append(agent_rewards)              

                        rewards = torch.cat(all_rewards, dim=0)

                        # Compute avg rewards for company 1
                        avg_rewards_all_agents_CoA = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                    elif company == 'CoB':
                        memory = memory_CoB
                        beta = beta_CoB
                        # agent data
                        agent_data = memory.agent_data
                        all_rewards = []

                        # Company #2 Data Preparation
                        for agent in agent_data.keys():
                            # Extract data for the current agent
                            agent_rewards = torch.tensor(agent_data[agent]['rewards'][:-1], dtype=torch.float32).reshape(-1, 1)
                            all_rewards.append(agent_rewards)

                        rewards = torch.cat(all_rewards, dim=0)

                        # Compute avg rewards for company 2
                        avg_rewards_all_agents_CoB = float(sum(rewards)/len(rewards)) * 1000

                        # Reset the memory for the next episode
                        memory.reset()

                END_EPISODE = True
        
        # calculate sum of agent_counters
        total_agent_steps = sum(env.agent_counter.values())
        if episode == 0:
            MAX_STEPS_SO_FAR = total_agent_steps
        else:
            MAX_STEPS_SO_FAR = max(MAX_STEPS_SO_FAR, total_agent_steps)
        # print(f"Total agent steps: {total_agent_steps}")

        # calculate max agent counter
        max_agent_counter = max(env.agent_counter.values())
        # print(f"Max agent time: {float(max_agent_counter*env.dt/60):.2f} min")

        N_success = env.get_num_successful_agents()
        # print(f"Number of successful agents: {N_success}")

        # Update the learning rate
        if TRAINING:
            CoA_agent.update_learning_rate(episode)
            CoB_agent.update_learning_rate(episode)

        # Calculate the time taken for the episode
        end_time = time.time()
        elapsed_time = end_time - start_time     

        # print out the agent counters from env.agent_counter = {agent: count}
        # print(f"Agent counters: {env.agent_counter}")
        # if len(dones) > (MAX_NUM_AGENTS - 1):

        total_steps = sum(env.agent_counter.values())
        if total_steps == 0:
            total_steps = 1
            print("Sth went wrong in the environment. Total steps = 0. Setting it to 1.")

        # collided agents
        # collided_agents_info = env.collided_agents

        # # finding the sum subrewards of each agent during a given episode
        episode_rewards = env.each_agent_rewards
        sum_episode_rewards = env.sum_episode_rewards

        SumRewards = {}
        for agent, rewards in episode_rewards.items():
            sum_distance_rewards = 0
            sum_action_rewards = 0
            sum_time_rewards = 0
            sum_mission_acc_rewards = 0
            sum_velocity_rewards = 0
            for reward in rewards:
                sum_distance_rewards += reward[0]
                sum_action_rewards += reward[1]
                sum_time_rewards += reward[2]
                sum_mission_acc_rewards += reward[3]
                sum_velocity_rewards += reward[4]
            SumRewards[agent] = [sum_distance_rewards, sum_action_rewards, sum_time_rewards, sum_mission_acc_rewards, sum_velocity_rewards]

        # Extract cumulative rewards for each company
        sum_rewards_CoA = 0
        sum_distance_rewards_CoA = 0
        sum_action_rewards_CoA = 0
        sum_time_rewards_CoA = 0
        sum_mission_acc_rewards_CoA = 0
        sum_velocity_rewards_CoA = 0
        sum_rewards_CoB = 0
        sum_distance_rewards_CoB = 0
        sum_velocity_rewards_CoB = 0
        sum_action_rewards_CoB = 0
        sum_time_rewards_CoB = 0
        sum_mission_acc_rewards_CoB = 0
        sum_velocity_rewards_CoB = 0
        for agent in SumRewards.keys():
            # Check if the agent belongs to Company A
            if agent.startswith('AA') or agent.startswith('AC'):
                sum_rewards_CoA += sum(SumRewards[agent])
                sum_distance_rewards_CoA += SumRewards[agent][0]
                sum_action_rewards_CoA += SumRewards[agent][1]
                sum_time_rewards_CoA += SumRewards[agent][2]
                sum_mission_acc_rewards_CoA += SumRewards[agent][3]
                sum_velocity_rewards_CoA += SumRewards[agent][4]
            else:
                sum_rewards_CoB += sum(SumRewards[agent])
                sum_distance_rewards_CoB += SumRewards[agent][0]
                sum_action_rewards_CoB += SumRewards[agent][1]
                sum_time_rewards_CoB += SumRewards[agent][2]
                sum_mission_acc_rewards_CoB += SumRewards[agent][3]
                sum_velocity_rewards_CoB += SumRewards[agent][4]

        # Calculate average rewards for each company
        avg_distance_rewards_CoA = sum_distance_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_action_rewards_CoA = sum_action_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_time_rewards_CoA = sum_time_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_mission_acc_rewards_CoA = sum_mission_acc_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_velocity_rewards_CoA = sum_velocity_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000

        # Company B
        avg_distance_rewards_CoB = sum_distance_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_action_rewards_CoB = sum_action_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_time_rewards_CoB = sum_time_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_mission_acc_rewards_CoB = sum_mission_acc_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_velocity_rewards_CoB = sum_velocity_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000

        # sum_rewards_CoA = sum_episode_rewards_CoA #sum_distance_rewards_CoA + sum_velocity_rewards_CoA + sum_action_rewards_CoA + sum_time_rewards_CoA

        # avg_rewards_all_agents = sum_episode_rewards/total_steps * counter
        # # avg_rewards_all_agents = sum_episode_rewards/counter * 100.0

        # Save the best model after 100 episodes
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if TRAINING and num_of_collisions < 1:
            if avg_rewards_all_agents_CoA >= best_avg_rewards_CoA:
                best_avg_rewards_CoA = avg_rewards_all_agents_CoA
                print(f"New best average rewards: {best_avg_rewards_CoA}. Saving the model...")

                CoA_agent.save(save_path=f'{script_dir}/trained_models/{MODEL_NAME_COMPANY1}')  # Save the best model
            
            if avg_rewards_all_agents_CoB >= best_avg_rewards_CoB:
                best_avg_rewards_CoB = avg_rewards_all_agents_CoB
                print(f"New best average rewards: {best_avg_rewards_CoB}. Saving the model...")
                CoB_agent.save(save_path=f'{script_dir}/trained_models/{MODEL_NAME_COMPANY2}')

        # avg_episode_actions = {agent: float(np.mean(actions)) for agent, actions in episode_actions.items()}
        # # Average of avg_episode_actions
        # avg_action = np.mean(list(avg_episode_actions.values()))

        # Average loss per episode
        avg_loss_CoA = np.mean(episodic_losses_CoA)
        avg_loss_CoB = np.mean(episodic_losses_CoB)

        # Average entropy per episode
        avg_entropy_CoA = np.mean(entropies_CoA)
        avg_entropy_CoB = np.mean(entropies_CoB)

        # Average KL divergence per episode
        avg_kl_divergence_CoA = np.mean(kl_divergences_CoA)
        avg_kl_divergence_CoB = np.mean(kl_divergences_CoB)

        # Average speeds per episode
        avg_spds = env.get_avg_episode_spd()
        avg_spd_CoA = avg_spds['CoA']
        avg_spd_CoB = avg_spds['CoB']
        # print(f"Average speed Co. A: {avg_spd_CoA:.4f}, Average speed Co. B: {avg_spd_CoB:.4f}")

        # Merging points and intersection collisions
        # M1_collisions = env.collision_counter['M1']
        # M2_collisions = env.collision_counter['M2']
        # In_collisions = env.collision_counter['In']
        # All_collisions = env.collision_counter['All']
        # print(f"Merging points collisions: {M1_collisions}, {M2_collisions}, Intersection collisions: {In_collisions}, All collisions: {All_collisions}")

        # # Format `sum_rewards` to 4 decimal places
        # avg_rewards_formatted = f"{avg_rewards_all_agents:.4f}"

        # # Format `SumRewards` dictionary values to 4 decimal places
        # # SumRewards_formatted = {agent: [f"{reward:.4f}" for reward in rewards] for agent, rewards in SumRewards.items()}
        # sum_subrewards_formatted = [f"{sum_distance_rewards_all_agents:.4f}", f"{sum_action_rewards_all_agents:.4f}", f"{sum_time_rewards_all_agents:.4f}"]
        # avg_subrewards_formatted = [f"{avg_distance_rewards_all_agents:.4f}", f"{avg_action_rewards_all_agents:.4f}", f"{avg_time_rewards_all_agents:.4f}"]
        # # Format `avg_episode_actions` dictionary values to 4 decimal places
        # avg_episode_actions_formatted = {agent: f"{avg_action:.4f}" for agent, avg_action in avg_episode_actions.items()}

        # # Foramt avg_loss to 4 decimal places
        # avg_loss = f"{avg_loss:.4f}"

        if num_of_collisions < 1:
            # Save Scenario
            # env.record_scenario()

            # Save speeds
            # At the end of your training loop, save the dictionary
            if episode > 2000:
                file_name = 'episode_speeds_'+DATE+'.pkl'
                with open(file_name, 'wb') as f:
                    pickle.dump(env.episode_spds, f)

        # Print the formatted results
        # print(f"\nEpisode {episode}>> Collisions: {num_of_collisions}, Avg Rewards: {avg_rewards_formatted}, Avg Subrewards: {avg_subrewards_formatted}, Sum Subrewards: {sum_subrewards_formatted}, Avg Loss: {avg_loss}, Elapsed Time: {elapsed_time:.2f}, LR: {ppo_agent.new_lr:.5f}, Beta: {beta:.4f}, Training Steps: {training_steps}.")
        print(f"\nEpisode {episode}>> N_success: {N_success}, "
                f"Collisions: {num_of_collisions}, "
                f"Avg Rewards Co. A: {avg_rewards_all_agents_CoA:.4f}, "
                f"Avg Rewards Co. B: {avg_rewards_all_agents_CoB:.4f}, "
                f"Elapsed Time: {elapsed_time:.2f}, "
                f"Max agent time: {float(max_agent_counter*env.dt/60):.2f} (min).")
        # print(f"Average Actions: {avg_episode_actions_formatted}.")

        # print(f"Episode Counter:{counter}, Collided Agents: {collided_agents_info}, Average Actions: {avg_episode_actions_formatted}.")

        if WANDB:
            wandb.log({
                "N_success": N_success,
                "Collisions": env.collision_counter['All'],
                "Avg Rewards Co. A": avg_rewards_all_agents_CoA,
                "Avg Rewards Co. B": avg_rewards_all_agents_CoB,
                "Avg Distance Rewards Co. A": avg_distance_rewards_CoA,
                "Avg Action Rewards Co. A": avg_action_rewards_CoA,
                "Avg Time Rewards Co. A": avg_time_rewards_CoA,
                "Avg Mission Acc Rewards Co. A": avg_mission_acc_rewards_CoA,
                "Avg Velocity Rewards Co. A": avg_velocity_rewards_CoA,
                "Avg Distance Rewards Co. B": avg_distance_rewards_CoB,
                "Avg Action Rewards Co. B": avg_action_rewards_CoB,
                "Avg Time Rewards Co. B": avg_time_rewards_CoB,
                "Avg Mission Acc Rewards Co. B": avg_mission_acc_rewards_CoB,
                "Avg Velocity Rewards Co. B": avg_velocity_rewards_CoB,
                "Entropy Co. A": avg_entropy_CoA,
                "KL Divergence Co. A": avg_kl_divergence_CoA,
                "Average Loss Co. A": float(avg_loss_CoA),
                "Entropy Co. B": avg_entropy_CoB,
                "KL Divergence Co. B": avg_kl_divergence_CoB,
                "Average Loss Co. B": float(avg_loss_CoB),
                "AA Collisions": float(env.collision_counter['AA']),
                "AB Collisions": float(env.collision_counter['AB']),
                "BB Collisions": float(env.collision_counter['BB']),
            })
            
        # append data to the dataset
        mydataframe['N_success'].append(N_success)
        mydataframe['Collisions'].append([env.collision_counter['All'], env.collision_counter['AA'], env.collision_counter['AB'], env.collision_counter['BB'],
                                          env.collision_counter['M1'], env.collision_counter['M2'], env.collision_counter['In']])
        mydataframe['Avg Rewards'].append([avg_rewards_all_agents_CoA, avg_rewards_all_agents_CoB])
        mydataframe['Avg Distance Rewards'].append([avg_distance_rewards_CoA, avg_distance_rewards_CoB])
        mydataframe['Avg Action Rewards'].append([avg_action_rewards_CoA, avg_action_rewards_CoB])
        mydataframe['Avg Time Rewards'].append([avg_time_rewards_CoA, avg_time_rewards_CoB])
        mydataframe['Avg Mission Acc Rewards'].append([avg_mission_acc_rewards_CoA, avg_mission_acc_rewards_CoB])
        mydataframe['Avg Velocity Rewards'].append([avg_velocity_rewards_CoA, avg_velocity_rewards_CoB])
        mydataframe['Entropy'].append([avg_entropy_CoA, avg_entropy_CoB])
        mydataframe['KL Divergence'].append([avg_kl_divergence_CoA, avg_kl_divergence_CoB])
        mydataframe['Average Loss'].append([float(avg_loss_CoA), float(avg_loss_CoB)])
        mydataframe['Actions'].append([[env.action_counter_A['decelerate'], env.action_counter_A['hold'], env.action_counter_A['accelerate']],
                                        [env.action_counter_B['decelerate'], env.action_counter_B['hold'], env.action_counter_B['accelerate']]])
        mydataframe['Avg Speed'].append([avg_spd_CoA, avg_spd_CoB])
        # Save the dataset to a CSV file after each 500 episodes
        if (episode+1) % 500 == 0:
            df = pd.DataFrame(mydataframe)
            df.to_csv(dataframe_name, index=False)


    # Finish wandb run
    if WANDB:
        wandb.finish()
