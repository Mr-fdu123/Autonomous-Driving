import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Qnet(nn.Module):
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
    
@dataclass
class Transition:
    state : np.ndarray
    action : int
    reward : float
    next_state : np.ndarray
    done : bool

class ReplayBuffer:
    def __init__(self, capsity):
        self.buffer = deque(maxlen=capsity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda",
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 200,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_net = Qnet(state_dim, action_dim).to(device)
        self.target_net = Qnet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.train_steps = 0

    def select_action(self, states):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions = self.q_net(states)
            return int(torch.argmax(actions, dim=1).item())
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # update Q
        q_values = self.q_net(states) # (B,a)
        current_q = torch.gather(q_values, dim=1, index=actions) # (B, 1)

        with torch.no_grad():
            q_next_max = torch.max(self.q_net(next_states), dim=1)[0]
            target_q = rewards + self.gamma * (1.0 - dones) * q_next_max

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_steps += 1
        if self.train_steps % self.target_update_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
    
def train_dqn(env_name: str = "CartPole-v1", episodes: int = 300):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=200,
    )

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        loss_value = None

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss_value = agent.train_step()

            state = next_state
            total_reward += reward

        # epsilon 衰减
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(
                f"Episode {episode+1}, "
                f"avg reward (last 10): {avg_reward:.2f}, "
                f"epsilon: {agent.epsilon:.3f}, "
                f"loss: {loss_value}"
            )

    env.close()
    return agent, episode_rewards

if __name__ == "__main__":
    agent, rewards = train_dqn()