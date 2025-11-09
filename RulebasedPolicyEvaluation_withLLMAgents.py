import wandb
import numpy as np
from envs import my_matc_env_rb_strong
# from PPO_attention import PPO_Agent
# from PPO_attention import Memory
# from PPO_attention import compute_gae, discount
# from PPO_attention import schedule_beta
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
import json

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")

MAX_NUM_AGENTS = 20
NUM_INTRUDERS = 3
NUM_EPISODES = 1

# Example usage
if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--date", type=str, required=True, help="Unique name for this run")
    # args = parser.parse_args()

    TRAINING = False
    WANDB = False
    RENDER_MODE = 'sim' 

    DATE = 'Nov6' # args.date
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
    num_agents = MAX_NUM_AGENTS
    numEpisodes = NUM_EPISODES

    # Initializing the environment
    flight_route_file = 'routes/matc/auto_airspace_config.json'
    aircraft_config_file = 'routes/matc/aircraft_configurations_auto.json'
    env = my_matc_env_rb_strong.MATCEnv(flight_route_file, aircraft_config_file, render_mode=RENDER_MODE)

    # Company list
    companylist = ['CoA', 'CoB']

    # Initialize Dataset
    dataframe_name = 'dataset'+DATE+'.csv'
    mydataframe = {
                "N_success": [],
                "Collisions": [],
                "Avg Rewards": [],
                "Avg Distance Rewards": [],
                "Avg Action Rewards": [],
                "Avg Time Rewards": [],
                "Avg Mission Acc Rewards": [],
                "Avg Velocity Rewards": [],
                "Actions": [],
                "Avg Speed": [],
            } 

    # Training loop
    for episode in range(numEpisodes):

        # Starting time for the episode
        start_time = time.time()

        rewards = {}
        observations, processed_states = env.reset()
        dones = {agent: 0 for agent in processed_states.keys()}  # Initialize done states for all agents
        episode_actions = {agent: [] for agent in processed_states.keys()}
        # for i in range(episode_length):
        counter = 0

        # print(f"Beta: {beta}")
        END_EPISODE = False
        while not END_EPISODE: 

            Actions, Reasons = env.get_rb_actions(observations)

            llm_agents = ['AA0001', 'AA0002', 'AB0001', 'AB0002', "AA0003"]
            for agent in llm_agents.copy():
                if agent not in Actions.keys():
                    llm_agents.remove(agent)
            if len(llm_agents) > 0:
                llm_Actions = env.get_llm_actions(observations, agent_list=llm_agents)

            # update Actions with llm_Actions
            for agent in llm_agents:
                Actions[agent] = llm_Actions[agent]
                print(f"LLM Action for {agent}: {llm_Actions[agent]}")

            # Step the environment
            if Actions != {}:
                observations, next_processed_states, rewards, terminations, truncations, dones = env.step(Actions)
            else:
                END_EPISODE = True

            # Check for collisions
            collided_agents = env.collided_agents
            num_of_collisions = len(env.collided_agents)

            # Update the observations
            processed_states = next_processed_states.copy()

            # Calculate the total length of all states in memory
            # Increment the counter
            counter += 1
            print(counter, end='\r')

            # pause simulation for a short while to let the server breathe
            if RENDER_MODE == 'server':
                time.sleep(0.1)

            if (all(env.dones.values()) or counter >= MAX_STEPS_SO_FAR):
                END_EPISODE = True

            # Save data to a json file
                
        
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
        avg_rewards_all_agents_CoA = sum_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_distance_rewards_CoA = sum_distance_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_action_rewards_CoA = sum_action_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_time_rewards_CoA = sum_time_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_mission_acc_rewards_CoA = sum_mission_acc_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_velocity_rewards_CoA = sum_velocity_rewards_CoA / total_steps / (MAX_NUM_AGENTS / 2) * 1000

        # Company B
        avg_rewards_all_agents_CoB = sum_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_distance_rewards_CoB = sum_distance_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_action_rewards_CoB = sum_action_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_time_rewards_CoB = sum_time_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_mission_acc_rewards_CoB = sum_mission_acc_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000
        avg_velocity_rewards_CoB = sum_velocity_rewards_CoB / total_steps / (MAX_NUM_AGENTS / 2) * 1000

        # Average speeds per episode
        avg_spds = env.get_avg_episode_spd()
        avg_spd_CoA = avg_spds['CoA']
        avg_spd_CoB = avg_spds['CoB']

        if num_of_collisions < 1:
            # Save Scenario
            # env.record_scenario()

            # Save speeds
            # At the end of your training loop, save the dictionary
            if episode > 100:
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
        mydataframe['Actions'].append([[env.action_counter_A['decelerate'], env.action_counter_A['hold'], env.action_counter_A['accelerate']],
                                        [env.action_counter_B['decelerate'], env.action_counter_B['hold'], env.action_counter_B['accelerate']]])
        mydataframe['Avg Speed'].append([avg_spd_CoA, avg_spd_CoB])
        # Save the dataset to a CSV file after each 10 episodes
        if (episode+1) % 10 == 0:
            df = pd.DataFrame(mydataframe)
            df.to_csv(dataframe_name, index=False)

    # Save dataset at the end of training
    if len(env.all_state_action_pairs) > 0:
        # whether the file exists. if yes append to it else create a new one and write to it
        file_exists = os.path.isfile("raw_ownship_intruders_data_"+DATE+".jsonl")
        if file_exists:
            print("Appending to existing dataset file.")
            mode = "a"
        else:
            print("Creating new dataset file.")
            mode = "w"
        with open("raw_ownship_intruders_data_"+DATE+".jsonl", mode) as f:
            for record in env.all_state_action_pairs:
                f.write(json.dumps(record) + "\n")
            
    # Finish wandb run
    if WANDB:
        wandb.finish()