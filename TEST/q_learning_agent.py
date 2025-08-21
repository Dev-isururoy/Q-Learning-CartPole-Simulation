import numpy as np
import random

class QLearningAgent:

    import os

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.q_table = np.load(filepath)
            print("[QAgent] Q-table loaded from", filepath)
        else:
            print("[QAgent] No Q-table found, starting fresh.")


    def __init__(self, state_space_shape, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size
        self.q_table = np.zeros((*state_space_shape, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def _to_int_tuple(self, state):
        # Convert state to a tuple of ints (for indexing)
        if isinstance(state, (list, np.ndarray)):
            return tuple(int(s) for s in state)
        elif isinstance(state, tuple):
            return tuple(int(s) for s in state)
        else:
            return (int(state),)

    def choose_action(self, state):
        state = self._to_int_tuple(state)
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        state = self._to_int_tuple(state)
        next_state = self._to_int_tuple(next_state)

        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        if done:
            self.exploration_rate *= self.exploration_decay

    def get_q_table(self):
        return self.q_table
