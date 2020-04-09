from CNN import CNN
from random import random, randint, randrange
import numpy as np
import os


class Agent:
    def __init__(self, input_shape, action_space, game_name,
                 memory=10000,
                 epsilon=1,
                 min_epsilon=0.1,
                 decay_rate=0.9,
                 batch_size=32,
                 load_weights=True,
                 test=False):
        self.action_set = action_space  # if action_space <= 6 else 6
        self.input_shape = input_shape
        self.memory_size = memory
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.decay = decay_rate
        self.batch_size = batch_size
        self.game = game_name

        filepath = str(self.game + "_weights") if load_weights else None

        self.policy_network = CNN(self.input_shape,
                                  self.action_set,
                                  batch_size=self.batch_size,
                                  weights=filepath if os.path.exists(filepath) else None)
        if not test:
            self.target_network = CNN(self.input_shape, self.action_set, batch_size=self.batch_size)

            self.target_network.model.set_weights(self.policy_network.model.get_weights())

            self.experiences = []

    def action(self, state):
        if random() < self.epsilon:
            return randint(0, self.action_set-1)
        else:
            return self.policy_network.predict(state).argmax()

    def experience_gain(self, current_state, action, reward, next_state, done):
        if self.experiences.__len__() >= self.memory_size:
            self.experiences.pop(0)

        self.experiences.append({'current': current_state,
                                 'action': action,
                                 'reward': reward,
                                 'next_state': next_state,
                                 'done': done})

    def sample_experiences(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.experiences[randrange(0, self.experiences.__len__())])
        return np.asarray(batch)

    def train(self):
        self.policy_network.train(self.sample_experiences(), self.target_network)

    def greedy(self):
        if self.epsilon - self.decay > self.epsilon_min:
            self.epsilon -= self.decay
        else:
            self.epsilon = self.epsilon_min

    def update_target_network(self):
        self.target_network.model.set_weights(self.policy_network.model.get_weights())

    def experience_available(self):
        return True if self.experiences.__len__() >= self.batch_size else False

    def save(self):
        self.policy_network.save(filepath=str(self.game+"_weights"))
