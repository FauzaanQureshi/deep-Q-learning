import random
from QValues import QValues
import torch
import gym
import os.path
from PIL import Image
import numpy as np
from DQN import DQN


def gpu_test():
    print("CUDA" if torch.cuda.is_available() else "CPU")


def render_test():
    env = gym.make("Pong-v0")
    env.reset()
    # action = random.randrange(2)

    r = 0
    _ = 0
    done = False
    # for _ in range(0):
    while not done:
        _ += 1
        env.render()
        action = 2 if random.randrange(2) == 0 else 5
        step = torch.tensor([action]).to("cuda").item()
        _1, reward, done, _2 = env.step(step)   # (env.action_space.sample())
        r += reward
        if _ % 100 == 0:
            print(r, done, step)
        # time.sleep(0.01)
    print(_, r)
    env.close()


def dqn_test():
    env = gym.make("Pong-v0")
    env.reset()
    img = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
    # print(img.shape)
    dqn_policy = DQN(img.shape[0])
    dqn_target = DQN(img.shape[0])
    if os.path.exists("PolicyNet"):
        dqn_policy.model.load_weights("PolicyNet")
        print("WEIGHTS FOUND!!")
    # dqn_target.model.load_weights("PolicyNet")
    dqn_target.model.set_weights(dqn_policy.model.get_weights())
    # screen_ = screen = np.zeros(img.shape[0])

    for episode in range(5):
        episode_reward = 0
        screen_ = screen = np.zeros(img.shape[0])
        done = False
        env.reset()
        _ = 0
        while not done:
            _ += 1
            env.render()
            sc_prev = screen
            # current_q_val = dqn_policy.forward(np.array([screen_-screen]))[0]
            current_q_val = QValues.q_val(dqn_policy, screen_-screen)[0]
            screen = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
            if current_q_val[0] > current_q_val[1]:
                step = 5
            else:
                step = 2
            next_state, reward, done, info = env.step(step)
            episode_reward += reward
            next_screen = np.asanyarray(processed_screen(next_state)).flatten()

            # target_q_val = dqn_target.forward(np.array([next_screen - screen]))[0]
            target_q_val = QValues.q_val(dqn_target, next_screen - screen)
            # print(_, current_q_val, target_q_val)
            screen_ = np.asanyarray(processed_screen(env.render("rgb_array"))).flatten()
            # time.sleep(0.005)
            dqn_policy.model.fit(np.array([screen_-sc_prev]), target_q_val, use_multiprocessing=True, verbose=0)

            if _ % 10 == 0:
                dqn_target.model.set_weights(dqn_policy.model.get_weights())
        print(episode, "\tTotal Reward = ", episode_reward)

    dqn_policy.model.save_weights("PolicyNet")
    env.close()


def processed_screen(screen):
    img = Image.fromarray(screen).convert('L')
    return img


if __name__ == '__main__':
    # render_test()
    # gpu_test()
    dqn_test()
