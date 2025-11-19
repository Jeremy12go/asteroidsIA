import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- Modelo deep Q-network ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = []
        self.max_size = max_size

    def push(self, exp):
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)


# --- DQN AGENT ---
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.model = DQN(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay = ReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            qvals = self.model(torch.FloatTensor(state))
        return torch.argmax(qvals).item()

    def train_step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return
        
        minibatch = self.replay.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        next_q = self.model(next_states).max(1)[0]
        q_target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_action, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
