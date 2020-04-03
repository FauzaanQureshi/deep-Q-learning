import torch
import gym
import time
from PIL import Image
import numpy as np
from DQN import DQN


def gpu_test():
    print("CUDA" if torch.cuda.is_available() else "CPU")


def render_test():
    env = gym.make("Pong-v0")
    env.reset()
    # print(env.action_space.sample())
    r = 0
    _ = 0
    done = False
    #for _ in range(1500):
    while not done:
        _ +=1
        env.render()
        _1, reward, done, _2 = env.step(env.action_space.sample())
        r += reward
        if _ % 100 == 0: print(r, done)
        time.sleep(0.001)
    print(_, r)
    env.close()


def dqn_test():
    env = gym.make("Pong-v0")
    env.reset()
    img = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
    #print(img.shape)
    dqn_policy = DQN(img.shape[0])

    dqn_target = DQN(img.shape[0])
    dqn_target.model.set_weights(dqn_policy.model.get_weights())

    for _ in range(0):
        env.render()
        screen = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
        action = dqn_policy.forward(np.array([screen]))[0]
        if action[0] > action[1]:
            step = 2
        else:
            step = 5
        env.step(step)
        time.sleep(0.005)
        if _ % 10 == 0: dqn_target.model.set_weights(dqn_policy.model.get_weights())
    env.close()


def processed_screen(screen):
    img = Image.fromarray(screen).convert('L')
    return img


if __name__ == '__main__':
    # render_test()
    # gpu_test()
    dqn_test()
