import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
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
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class ActorCriticAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.device = device
        self.gamma = gamma

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, state : np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_p = dist.log_prob(action)
        return int(action.item()), log_p

    def update(
        self,
        state: np.ndarray,
        log_prob: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)

        value = self.critic(state_tensor)               # V(s)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor) # V(s')

        td_target = reward_tensor + self.gamma * (1.0 - done_tensor) * next_value
        td_error = td_target - value

        # actor loss
        actor_loss = -log_prob * td_error.detach().squeeze()

        # critic loss
        critic_loss = td_error.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

def train_actor_critic(env_name: str = "CartPole-v1", episodes: int = 500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
    )

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        last_actor_loss = 0.0
        last_critic_loss = 0.0

        while not done:
            action, log_prob = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actor_loss, critic_loss = agent.update(
                state=state,
                log_prob=log_prob,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state
            total_reward += reward
            last_actor_loss = actor_loss
            last_critic_loss = critic_loss

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(
                f"Episode {episode+1}, "
                f"avg reward (last 10): {avg_reward:.2f}, "
                f"actor loss: {last_actor_loss:.4f}, "
                f"critic loss: {last_critic_loss:.4f}"
            )

    env.close()
    return agent, episode_rewards

if __name__ == "__main__":
    agent, rewards = train_actor_critic()