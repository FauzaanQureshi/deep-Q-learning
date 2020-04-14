import gym
from PIL import Image
import numpy as np
from Agent import Agent
import sys
from time import time, sleep


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


def main(game):
    # game_name = "Seaquest" if not _name_ else _name_
    env_name = game + "-v0"
    print(env_name, type(env_name))
    log = open("log.txt", 'a')
    no_episodes = int(input("Number of Episodes? : "))
    timer = time()
    log.write("\n\n=================================  Starting Session  ==============================\n")

    env = gym.make(env_name)

    agent = Agent((84, 84, 4), env.action_space.n, env_name, load_weights=True)
    count = 0
    _ = 0
    av_score = 0
    min_experiences = 0 if agent.load_state(True) else 1500  # if loading state, min exp req = 0 else 1500
    for episode in range(no_episodes):
        obsv = process_state(env.reset())
        current_state = np.array([obsv, obsv, obsv, obsv])
        score = 0
        done = False
        # _ = 0
        steps = _
        while not done:
            _ += 1
            # env.render() if no_episodes-episode-1 <= 3 else None

            action = agent.action(np.asarray([current_state]))

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)

            clipped_reward = np.clip(reward, -1, 1)
            agent.experience_gain(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            if agent.experience_available() and _ > min_experiences:
                agent.greedy()
                if _ % 4 == 0:
                    # _ = 0
                    count += 1
                    agent.train()
                    # print("Steps : ", _, "\tCount = ", count)
                    if count == 1000:
                        count = 0
                        agent.save()
                        agent.update_target_network()

            # if agent.experience_available():
            #    agent.greedy()

            current_state = next_state
            score += reward
        steps = _ - steps

        agent.save()
        env.close()
        timer = time() - timer
        av_score = (av_score + score)/2 if episode != 0 else score
        print(episode+1, "\tTotalReward = ", score, "\tSteps: ", steps, "\tMoving Avg: {:.2f}".format(av_score),
              "\tTime: %d" % (timer/60), "\b:{:.0f}".format((timer % 60)))
        log.write(str(episode+1) + "\tTotalReward = " + str(score) + "\tSteps: " + str(steps) + "\tMoving Avg: {:.2f}".format(av_score) + "\tTime: %d" % int(timer/60) + ":{:.0f} \n".format((timer % 60)))
        timer = time()

    print("\n\nTotal Steps = ", _)
    agent.save_state()
    input("\n\n\nPress Return to exit.")
    log.write("=================================   Ending Session   ==============================\n")
    log.close()


def test(game):
    env_name = game+"-v0"
    env = gym.make(env_name)

    agent = Agent((84, 84, 4), env.action_space.n, env_name, epsilon=0, load_weights=True, test=True)

    run = 'y'
    while run == 'y' or run == 'Y':
        obsv = process_state(env.reset())
        current_state = np.array([obsv, obsv, obsv, obsv])
        done = False
        score = _ = 0
        while not done:
            _ += 1
            env.render()
            sleep(0.01)
            action = agent.action(np.asarray([current_state]))
            # Image.fromarray(process_state(env.render("rgb_array"))).show() if _ % 500 == 0 else None

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)

            current_state = next_state
            score += reward

        print("Total Reward: ", score, "\nSteps: ", _)
        run = input("\nRUN TEST AGAIN? (Y/N) : ")
    print("Exiting Environment.")
    env.close()


def processed_screen():
    env = gym.make("Seaquest-v0")
    env.reset()
    for i in range(400):
        env.step(env.action_space.sample())
    screen = env.render("rgb_array")
    Image.fromarray(screen).save("Seaquest.png")
    Image.fromarray(process_state(screen)).show()


if __name__ == '__main__':
    try:
        game_name = sys.argv[1]
    except IndexError:
        game_name = "Seaquest"
    if input("1. Train Agent\n2. Run Test\n\n: ") == '1':
        main(game_name)
    else:
        test(game_name)
        # process_state("SC0.png")
        # processed_screen()
