from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import RMSprop
import numpy as np


class CNN:
    def __init__(self, input_dim, action_space,
                 discount_factor=0.99, learning_rate=0.00025, batch_size=32, weights=None):
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.model = Sequential()

        self.model.add(Conv2D(16,
                              8,
                              strides=(4, 4),
                              activation='relu',
                              input_shape=input_dim,
                              data_format='channels_last'))
        self.model.add(Conv2D(32,
                              4,
                              strides=(2, 2),
                              activation='relu',
                              data_format='channels_last'))
        # self.model.add(Conv2D(64, 3, (1, 1), activation='relu', data_format='channels_first'))

        self.model.add(Flatten())

        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(action_space))

        self.model.compile(optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                           loss="mean_squared_error", metrics=["accuracy"])

        if weights is not None:
            self.model.load_weights(weights)

    def predict(self, current_state):
        return self.model.predict(np.moveaxis(current_state.astype(np.float64), [1, 2, 3], [3, 1, 2]), batch_size=1)

    def train(self, batch, target_network):
        x = []
        y = []

        for experience in batch:
            x.append(experience['current'].astype(np.float64))

            next_state = experience['next_state'].astype(np.float64)
            next_state_pred = target_network.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            target = list(self.predict(experience['current'])[0])
            if experience['done']:
                target[experience['action']] = experience['reward']
            else:
                target[experience['action']] = experience['reward'] + self.discount_factor * next_q_value
            y.append(target)

            # print(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]).shape)
            self.model.fit(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]),
                           np.asarray(y),
                           batch_size=self.batch_size,
                           epochs=1,
                           verbose=0)

    def save(self, filepath="model_weights"):
        print("Saving Weights!!")
        self.model.save_weights(filepath)
