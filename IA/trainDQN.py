from Environment import Environment
from DQN_agent import DQNAgent
import numpy as np
import torch

EPISODES = 3000
MAX_STEPS = 2000

INPUT_SIZE = 14
ACTION_SIZE = 6
HIDDEN_SIZE = 64

env = Environment()

agent = DQNAgent(INPUT_SIZE, ACTION_SIZE, hidden_size=HIDDEN_SIZE)

# Cargar modelo si existe
try:
    agent.model.load_state_dict(torch.load("best_dqn.pth"))
    print("Modelo cargado correctamente.")
except:
    print("No hay modelo previo. Comenzando desde cero.")

for episode in range(EPISODES):

    state = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):

        action = agent.act(state)
        env.game.current_model = agent.model
        env.game.current_state = state
        next_state, reward, done = env.step(action)

        agent.replay.push((state, action, reward, next_state, done))
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    agent.update_epsilon()

    print(f"Episode {episode}  | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # guardar mejor modelo
    if episode % 20 == 0:
        torch.save(agent.model.state_dict(), "best_dqn.pth")
        print("Modelo guardado.")
