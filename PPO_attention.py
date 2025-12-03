import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HIDDEN_SIZE = 64

def schedule_beta(episode, decay_episode=1000, initial_beta=0.01, final_beta=0.001):
    if episode <= decay_episode:
        # Linear decay from initial_beta to final_beta over decay_episode
        beta = initial_beta - (initial_beta - final_beta) * (episode / decay_episode)
    else:
        # Stop decaying after decay_episode
        beta = final_beta
    return beta

def discount(r, discounted_r, cumul_r):
    """ Compute the gamma-discounted rewards over an episode
    """
    for t in range(len(r) - 1, -1, -1):
        cumul_r = r[t] + cumul_r * 0.99
        discounted_r[t] = cumul_r
    return discounted_r

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages.detach(), returns.detach()

class PPONetwork(nn.Module):
    def __init__(self, state_size=6, context_size=5, action_size=3, num_intruders=3):
        super(PPONetwork, self).__init__()
        #ownership observation layers
        self.ownship_fc = nn.Linear(state_size, HIDDEN_SIZE)
        self.ownship_activation = nn.LeakyReLU(0.2)
        # intruder observation layers
        self.intruder_fc = nn.Linear(context_size, HIDDEN_SIZE)
        self.intruder_activation = nn.LeakyReLU(0.2)
        #Attention mechanism
        self.attention_fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.attention_activation = nn.Softmax(dim=1)
        # Combined processing
        self.combined_fc1 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
        self.combined_activation1 = nn.LeakyReLU(0.2)
        self.combined_fc2 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
        self.combined_activation2 = nn.LeakyReLU(0.2)
        self.final_fc = nn.Linear(HIDDEN_SIZE * 2, action_size + 1)

    def forward(self, state, context):
        # Ensure inputs are float32
        state = state.float()
        context = context.float()
        # Process ownship state
        x_ownship = self.ownship_activation(self.ownship_fc(state))
        # Process intruder context
        batch_size, num_intruders, _ = context.size()
        x_intruders = self.intruder_activation(self.intruder_fc(context.reshape(-1, context.size(-1))))
        x_intruders = x_intruders.view(batch_size, num_intruders, HIDDEN_SIZE)
        # Apply attention mechanism
        attention_score = self.attention_fc(x_intruders)
        attention_score = torch.bmm(attention_score, x_ownship.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, num_intruders)
        attention_weights = self.attention_activation(attention_score).unsqueeze(1)  # Shape: (batch_size, 1, num_intruders)
        context_attention = torch.bmm(attention_weights, x_intruders).sum(dim=1)  # Shape: (batch_size, HIDDEN_SIZE)
        # Combine ownship and context
        combined = torch.cat([x_ownship, context_attention], dim=1)  # Shape: (batch_size, 2 * HIDDEN_SIZE)
        # Process through combined layers
        x = self.combined_activation1(self.combined_fc1(combined))
        x = self.combined_activation2(self.combined_fc2(x))
        x = self.final_fc(x)
        action_probs = F.softmax(x[:, :-1], dim=1)  # Policy output
        value = x[:, -1:]  # Value output
        return action_probs, value

class PPO_Agent:
    def __init__(self, state_size=6, context_size=5, action_size=3, num_intruders=3, lr=1e-4, min_lr=1e-4, decay_episodes=1000):
        self.state_size = state_size
        self.context_size = context_size
        self.action_size = action_size
        self.num_intruders = num_intruders
        self.lr = lr
        self.min_lr = min_lr
        self.decay_episodes = decay_episodes
        self.new_lr = lr

        # Initialize model and optimizer
        # Initialize model and optimizer
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")  # Force CPU for compatibility with all environments
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = PPONetwork(state_size=state_size, context_size=context_size, action_size=action_size, num_intruders=num_intruders).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def update_learning_rate(self, current_episode):
        """Update the learning rate based on the current episode."""
        if current_episode <= self.decay_episodes:
            # Linear decay from lr to min_lr over decay_episodes
            self.new_lr = self.lr - (self.lr - self.min_lr) * (current_episode / self.decay_episodes)
        else:
            # Stop decaying after decay_episodes
            self.new_lr = self.min_lr

        # Update the optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.new_lr

        # print(f"Episode {current_episode}: Updated learning rate to {selfnew_lr:.6f}")

    def to_tensor(self, data, dtype=torch.float32):
        """Helper function to convert numpy arrays to torch tensors with proper dtype"""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=dtype, device=self.device)
        return data

    def save(self, save_path='', best=False):
        # save_path = save_path+f'best_model_attention.pt' if best else f'model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load(self, name):
        print('Loading weights...')
        checkpoint = torch.load(name, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Successfully loaded model weights from {name}.')
    
class Memory:
    def __init__(self, max_size=10000):
        self.agent_data = {}
        self.max_size = max_size

    def add(self, agent, state, context, action, reward, done, log_prob, old_value):
        if agent not in self.agent_data:
            self.agent_data[agent] = {
                "states": [],
                "contexts": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "log_probs": [],
                "old_values": []         
            }
        self.agent_data[agent]["states"].append(state)
        self.agent_data[agent]["contexts"].append(context)
        self.agent_data[agent]["actions"].append(action)
        self.agent_data[agent]["rewards"].append(reward)
        self.agent_data[agent]["dones"].append(done)
        self.agent_data[agent]["log_probs"].append(log_prob)
        self.agent_data[agent]["old_values"].append(old_value)

        # Remove oldest experience if size exceeds max_size
        if len(self.agent_data[agent]["states"]) > self.max_size:
            self.agent_data[agent]["states"].pop(0)
            self.agent_data[agent]["contexts"].pop(0)
            self.agent_data[agent]["actions"].pop(0)
            self.agent_data[agent]["rewards"].pop(0)
            self.agent_data[agent]["dones"].pop(0)
            self.agent_data[agent]["log_probs"].pop(0)
            self.agent_data[agent]["old_values"].pop(0)

    def reset(self):
        self.agent_data = {}


