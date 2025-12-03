import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from DQN_model import QNetwork   # tu red neuronal
from ReplayBuffer import ReplayBuffer

class DQN_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Redes
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)

        # Copia inicial de pesos (hard update)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

        # Replay memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 128

        # Parámetros DQN
        self.gamma = 0.99
        self.epsilon = 0.995
        self.epsilon_min = 0.35
        self.epsilon_decay = 0.9995

        self.replay = ReplayBuffer()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q(s, a)
        q_vals = self.model(states).gather(1, actions)

        # Q_target = r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * torch.max(self.target_model(next_states), dim=1, keepdim=True)[0]

        # loss
        loss = nn.MSELoss()(q_vals, target_q)

        # step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # actualización suave (soft update)
        tau = 0.01
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        # decaimiento de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
