import os

import pandas
import torch
from hmmlearn.hmm import GaussianHMM
from gym_trade.envs import trade_env
import numpy as np
import mplfinance as mpf
import pickle

# data = trade_env.load_data()
num_of_hidden = 5
days = 5
colors = ['red', 'blue', 'yellow', 'green', 'pink', '#7189aa', '#7189aa', 'black']

data = trade_env.load_data('eth-210704.csv')

# start = np.random.randint(400, size=1)[0]
# data = data[start:start+100]


def markov(show=False, train=True):

    logK1 = np.log(np.array(data['High'])/np.array(data['Low']))
    logK2 = np.log(np.array(data['Open']/np.average(data['Close'])))
    # volume = np.log(np.array(data['Volume']))
    logRet_1 = np.diff(np.log(data['Close']))[days-1:]
    logRet_5 = np.log(np.array(data['Close'][days:])/np.array(data['Close'][:-days]))
    # logRet_5 = np.diff(np.log(data['Close']), n=5)
    # logVol_5 = np.log(np.array(data['Volume'][5:]/np.array(data['Volume'][:-5])))

    print(len(logK1[days:]), len(logRet_1), len(logRet_5))

    # the histogram of the raw observation sequences
    # if show:
    #     n, bins, patches = plt.hist(logDel, 50, facecolor='green', alpha=0.75)
    #
    #     plt.show()
    #
    #     n, bins, patches = plt.hist(logRet_5, 50, facecolor='green', alpha=0.75)
    #
    #     plt.show()
    #
    #     n, bins, patches = plt.hist(logVol_5, 50, facecolor='green', alpha=0.75)
    #
    #     plt.show()

    x = np.column_stack([logK1[days:], logRet_1, logRet_5])

    print(x.shape)
    if train:
        model = GaussianHMM(n_components=num_of_hidden, covariance_type="full", n_iter=2000).fit(x)
        with open("hmm.pkl", "wb") as file:
            pickle.dump(model, file)
    else:
        with open("hmm.pkl", "rb") as file:
            model = pickle.load(file)
    hidden_states = model.predict(x)
    print(hidden_states, hidden_states.shape)

    return hidden_states


def show_hidden_states(states):
    w = len(data) - days
    position = np.array(data['Low'][days:])

    # m = np.zeros([data_length-days, num_of_hidden], dtype=np.float16)
    m = np.array([np.nan]).repeat(w * num_of_hidden).reshape((w, num_of_hidden))
    flag = np.zeros(num_of_hidden)
    for i in range(w):
        flag[states[i]] += 1
        m[i, states[i]] = position[i]-5

    m = m.transpose()
    print(m[4])
    add_plot = []
    for i in range(num_of_hidden):
        if flag[i] > 0:
            add_plot.append(mpf.make_addplot(m[i], type="scatter", scatter=True, markersize=10, color=colors[i], marker='o'))

    mpf.plot(data[days:], addplot=add_plot, type="candle", volume=True, style="binance")
    mpf.show()


s = markov(train=False)
show_hidden_states(s)




