import numpy as np

# Patch numpy to define bool8 if it's missing
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import os
from q_learning_agent import QLearningAgent  # assuming your agent class is named like this

Q_TABLE_PATH = "q_table.pkl"

def run_reinforcement_learning(agent, episodes=20):
    env = gym.make('CartPole-v1')
    print("[RL] Starting reinforcement learning in Gym environment...")

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)  # Get action from agent
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state, done)  # Learn from experience
            state = next_state
            total_reward += reward

        print(f"Episode {ep + 1}: Total Reward: {total_reward}")

    print(f"[RL] Completed {episodes} episodes.")

def main():
    # CartPole state space is 4-dimensional continuous
    # Discretize each dimension into bins, e.g. (6, 12, 6, 12)
    state_space_shape = (6, 12, 6, 12)
    action_space_size = 2  # CartPole has 2 actions: left, right

    agent = QLearningAgent(state_space_shape, action_space_size)

    if os.path.exists(Q_TABLE_PATH):
        agent.load(Q_TABLE_PATH)
        print("[QAgent] Loaded existing Q-table.")
    else:
        print("[QAgent] No Q-table found, starting fresh.")

    run_reinforcement_learning(agent, episodes=50)

    agent.save(Q_TABLE_PATH)
    print("[QAgent] Q-table saved.")

if __name__ == "__main__":
    main()
