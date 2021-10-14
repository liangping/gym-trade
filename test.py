import gym
# from gym import envs
# from gym.envs.registration import register
import numpy

from gym_trade import *

if __name__ == '__main__':

    # r = numpy.random.rand()
    # print(r)
    # exit()

    print("hello")
    env = gym.make('trade-v1')
    env.reset()
    for _ in range(28):
        env.render()
        step = env.action_space.sample()
        s, r, d, info = env.step(step)  # take a random action
        print(s, r, d)
        #print(step, r, info)
    env.close()

