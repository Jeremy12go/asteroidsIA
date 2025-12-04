from Environment import Environment
from DQN_agent import DQN_agent
import numpy as np
import torch

EPISODES = 3000
MAX_STEPS = 2000

INPUT_SIZE = 14
ACTION_SIZE = 5

env = Environment()

agent = DQN_agent(INPUT_SIZE, ACTION_SIZE)

try:
    checkpoint = torch.load("dqn_model.pth")
    agent.model.load_state_dict(checkpoint["model_state"])
    agent.target_model.load_state_dict(checkpoint["target_state"])
    agent.epsilon = checkpoint.get("epsilon", 1.0)
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

    if episode % 20 == 0:
        torch.save({
            "model_state": agent.model.state_dict(),
            "target_state": agent.target_model.state_dict(),
            "epsilon": agent.epsilon
        }, "dqn_model.pth")
        print("Modelo guardado.")
