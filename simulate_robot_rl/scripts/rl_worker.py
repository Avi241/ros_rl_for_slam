#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
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
import matplotlib.pyplot as plt
import robot_rl_env
from DQN import DQNAgent

# Environment settings
EPISODES = 100
MAX_STEP_IN_ESISODE = 100


ENV_NAME = "RobotEnv-v0"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = "2x256"
MIN_REWARD = 50  # For model save
MEMORY_FRACTION = 0.20

for i in range(100):
    print("check")

# Exploration settings
epsilon = 0.6  # not a constant, going to be decayed
EPSILON_DECAY = 0.9997
MIN_EPSILON = 0.1

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
pub = rospy.Publisher('episode_done', String, queue_size=10)
# rospy.init_node('rl_worker')
file_path = __file__
dir_path = file_path[: (len(file_path) - len("rl_worker.py"))]
MODELS_PATH = dir_path + "models/"  # model save directory
FIGURES_PATH = dir_path + "figures/"

ep_rewards = [0]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir("models"):
    os.makedirs("models")


def kill_all_nodes() -> None:
    """
    kill all ros node except for roscore
    """
    nodes = os.popen("rosnode list").readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace("\n", "")
    for node in nodes:
        os.system("rosnode kill " + node)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n
    print("actions = ")
    print(nb_actions)
    print("Observation")
    print(env.observation_space.shape[0])
    map_completeness = []
    agent = DQNAgent(env)
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        env.reset()
        print("environment rested")
        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1

        current_state = env.reset()
        done = False

        while not done:

            # for i in range(100):
            #     print(count)
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))

            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)

            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                # env.render()
                pass

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
            current_state = new_state
            step += 1

            if step >= MAX_STEP_IN_ESISODE:
                print("step over")
                break

        map_completeness.append(new_state[13])

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-AGGREGATE_STATS_EVERY:]
            )
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward,
                reward_min=min_reward,
                reward_max=max_reward,
                epsilon=epsilon,
            )

            # Save model, but only when min reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                agent.model.save(
                    f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
                )

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        pub.publish("episode={0}, map_com={1} ,reward={2} ".format(episode,new_state[13],episode_reward))
        # for i in range(10000):
        #     print("done with episode")

agent.model.save(
    f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
)

np.save("map", map_completeness)
np.save("reward", ep_rewards)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

map = moving_average(map_completeness,20)
reward = moving_average(ep_rewards,30)

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
