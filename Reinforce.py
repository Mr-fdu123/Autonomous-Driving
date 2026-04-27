import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda",
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.device = device
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim=128).to(device=device)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)

    def select_action(self, state : np.ndarray):
        # state (state_dim)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(state)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_p = dist.log_prob(action)

        return int(action.item()), log_p
    
    def compute_return(self, rewards):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns
    
    def update(self, log_probs: list[torch.Tensor], rewards: list[float]):
        returns = self.compute_return(rewards)
        policy_loss = []

        for log_p, r in zip(log_probs, returns):
            policy_loss.append(-log_p*r)

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
def train_reinforce(env_name: str = "CartPole-v1", episodes: int = 500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, device, lr=1e-3, gamma=0.99)
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        log_probs = []
        rewards = []

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            done = terminated or truncated
            state = next_state
            total_reward += reward

        loss = agent.update(log_probs, rewards)
        episode_rewards.append(total_reward)
    
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(
                f"Episode {episode+1}, "
                f"avg reward (last 10): {avg_reward:.2f}, "
                f"loss: {loss:.4f}"
            )

    env.close()

    return agent, episode_rewards

if __name__ == "__main__":
    agent, rewards = train_reinforce()



