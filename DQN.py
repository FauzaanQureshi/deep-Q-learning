from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class DQN:
    def __init__(self, dim):
        self.model = Sequential()

        self.model.add(Dense(24, activation="relu", input_dim=dim))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(2))
        self.model.compile(optimizer="adam", loss='mean_squared_error')

    def forward(self, v):
        #print(v.shape)
        v = self.model.predict(v)
        print(v[0])
        return v
