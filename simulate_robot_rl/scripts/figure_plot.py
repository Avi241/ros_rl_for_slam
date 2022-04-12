#!/usr/bin/env python3

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


import numpy as np
import matplotlib.pyplot as plt

reward = np.load('reward.npy')
map = np.load('map.npy')
map = moving_average(map,100)
reward = moving_average(reward,100)

fig = plt.figure()
plt.plot(map)
plt.xlabel("episdoes")
plt.ylabel("map_completeness")
plt.show()

fig = plt.figure()
plt.plot(reward)
plt.xlabel("episdoes")
plt.ylabel("rewards")
plt.show()
