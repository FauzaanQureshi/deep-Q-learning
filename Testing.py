import torch
import gym
import time
from PIL import Image
import numpy as np
from DQN import DQN


def gpu_test():
    print("CUDA" if torch.cuda.is_available() else "CPU")


def render_test():
    env = gym.make("Breakout-v0")
    env.reset()
    # print(env.action_space.sample())
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.01)
    env.close()


def dqn_test():
    env = gym.make("Breakout-v0")
    env.reset()
    img = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
    print(img.shape)
    dqn = DQN(img.shape[0])

    for _ in range(1000):
        env.render()
        screen = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
        action = dqn.forward(np.array([screen]))[0]
        if action[0] > action[1]:
            step = 2
        else:
            step = 5
        env.step(step)
        time.sleep(0.005)
    env.close()


def processed_screen(screen):
    img = Image.fromarray(screen).convert('L')
    return img


if __name__ == '__main__':
    # render_test()
    # gpu_test()
    dqn_test()
