import gym
from PIL import Image
import numpy as np
from Agent import Agent
import sys
# from time import time


def process_state(observation):
    img = np.asanyarray(Image.fromarray(observation, 'RGB').convert('L').resize((84, 110)))
    img = np.delete(img, np.s_[-13:], 0)
    img = np.delete(img, np.s_[:13], 0)
    # print(img.shape)
    return img

    # Image._show(Image.fromarray(img))
    # Image._show((Image.open(observation).convert('L').resize((84, 110))))
    # Image._showxv(Image.fromarray(observation, 'RGB').convert('L').resize((84, 110)), "BEFORE CROPPING")


def get_next_state(current, observation):
    return np.append(current[1:], [observation], axis=0)


def main(_name_=None):
    game_name = "Seaquest" if not _name_ else _name_
    game_name = game_name + "-v0"
    print(game_name, type(game_name))
    # log = open("log.txt", 'a')
    # log.write("==========Starting Session============\n\n\n")
    no_episodes = int(input("Number of Episodes? : "))
    # timer = time()

    env = gym.make(game_name)

    agent = Agent((84, 84, 4), env.action_space.n, game_name, load_weights=True)

    for episode in range(no_episodes):
        obsv = process_state(env.reset())
        current_state = np.array([obsv, obsv, obsv, obsv])
        score = 0
        done = False
        count = _ = 0

        while not done:
            _ += 1
            # env.render() if no_episodes-episode-1 <= 3 else None

            action = agent.action(np.asarray([current_state]))

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)

            clipped_reward = np.clip(reward, -1, 1)
            agent.experience_gain(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            if agent.experience_available():
                agent.greedy()
                if _ % 4 == 0:
                    # _ = 0
                    count += 1
                    agent.train()
                    # print("Steps : ", _, "\tCount = ", count)
                    if count == 750:
                        count = 0
                        agent.save()
                        agent.update_target_network()

            # if agent.experience_available():
            #    agent.greedy()

            current_state = next_state
            score += reward

        agent.save()
        env.close()
        # timer = time() - timer
        print(episode+1, "\tTotalReward = ", score)
        input("\n\n\nPress any key to exit.")
        # log.write(str(episode+1)+"\tTotalReward = "+str(score))

    # log.close()


def test():
    game_name = "Seaquest-v0"
    env = gym.make(game_name)

    agent = Agent((84, 84, 4), env.action_space.n, game_name, load_weights=True, test=True)

    obsv = process_state(env.reset())
    current_state = np.array([obsv, obsv, obsv, obsv])
    done = False
    score = _ = 0
    while not done:
        _ += 1
        env.render()
        action = agent.action(np.asarray([current_state]))
        agent.greedy() if _ % 5 == 0 else None

        obsv, reward, done, info = env.step(action)
        obsv = process_state(obsv)
        next_state = get_next_state(current_state, obsv)

        current_state = next_state
        score += reward

    print("Total Reward: ", score, "\nSteps: ", _)
    env.close()


def processed_screen():
    env = gym.make("Seaquest-v0")
    env.reset()
    for i in range(120):
        env.step(env.action_space.sample())
    screen = env.render("rgb_array")
    Image.fromarray(screen).save("Seaquest.png")
    Image.fromarray(process_state(screen)).show()


if __name__ == '__main__':
    # process_state("SC0.png")
    try:
        main(sys.argv[1])
    except IndexError:
        main()
    # test()
    # processed_screen()
