#!/usr/bin/env python3

import tensorflow as tf
from matplotlib.pyplot import step
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import time
import random
from tqdm import tqdm
import gym
import os
from PIL import Image
import cv2

import robot_rl_env
from DQN import DQNAgent

# Environment settings
EPISODES = 10
MAX_STEP_IN_ESISODE = 10


ENV_NAME = "RobotEnv-v0"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = "2x256"
MIN_REWARD = 100  # For model save
MEMORY_FRACTION = 0.20

for i in range(100):
    print("hiiiiiiiiiiiiiiiiiii")

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    print("actions = ")
    print(env.action_space.n)
    print("Observation")
    print(env.reset())
    for i in range(1000):      
        print(env.map_value())
        time.sleep(0.2)
    # print(env.step(2))
    # for i in range(100):
    #     print(env.step(2))
        


