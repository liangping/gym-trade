import os

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding


def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))


def load_data(datafile='btc.csv'):

    days = 5

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), datafile), index_col=0, parse_dates=True)
    df = df[["Open", "High", "Close", "Low", 'Volume']]

    # price = np.array((df['High']+df['Low'])/2)[days:]
    # logK1 = np.log(np.array(df['High'])/np.array(df['Low']))
    # logK2 = np.log(np.array(df['Open']/np.average(df['Close'])))
    # volume = np.log(np.log(np.array(df['Volume'][days:])))
    # logRet_1 = np.diff(np.log(df['Close']))[days-1:]
    # logRet_5 = np.log(np.array(df['Close'][days:])/np.array(df['Close'][:-days]))
    # # logRet_5 = np.diff(np.log(df['Close']), n=5)
    # # logVol_5 = np.log(np.array(df['Volume'][5:]/np.array(df['Volume'][:-5])))
    #
    # return np.column_stack([price, logK1[days:], logRet_1, logRet_5, volume])

    return df


class TradeEnv(gym.Env):
    """
        Crypto Environment for RL
    """

    def __init__(self):
        super(TradeEnv, self).__init__()
        self.initials = 500000
        self.np_random = None
        self.step_index = 0
        self.action_space = spaces.Discrete(3)
        self.seed()

        self.last_state = None
        self.state = None  # Start at beginning of the chain
        self.obs_data = load_data()
        self.length = len(self.obs_data)
        self.qty = 0
        self.remained_cash = 0
        self.total_value_on_last_step = 0

        self.reset()  # reset values for business logic

    def reset(self):
        self.step_index = 0
        self.state = self.obs_data[self.step_index]
        self.qty = int(self.initials * 0.5 / self.get_price())
        self.remained_cash = self.initials - self.qty * self.get_price()
        self.total_value_on_last_step = self.initials
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # 1. calculate reward
        # 2. get next states based on action
        # 3. check if episode ends?
        self.step_index = self.step_index + 1
        self.state = self.obs_data[self.step_index]
        reward = self.__reward__(action)
        done = self.__is_done__()
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print("Total Value", self.__stock_value__())

    def close(self):
        pass

    def __stock_value__(self):
        return self.qty * self.get_price() + self.remained_cash

    def __reward__(self, action):
        qty = 1
        price = self.get_price()
        old = self.total_value_on_last_step
        self.total_value_on_last_step = self.__stock_value__()
        r = self.total_value_on_last_step - old
        # self.episode_reward += r

        # print(action, qty, price, self.total_value_on_last_step, old, r)

        if action == 0:
            # buy operation
            self.qty += 1
            self.remained_cash -= qty * price
        elif action == 1:
            pass
        elif action == 2:
            # sell operation
            self.qty -= 1
            self.remained_cash += qty * price

        return r*0.001

    def get_price(self):
        return np.average(self.state)

    def __is_done__(self):
        if self.step_index >= self.length \
                or self.qty < 1 \
                or self.remained_cash < 1\
                or self.__stock_value__() < self.initials * 0.8:
            return True
        else:
            return False

