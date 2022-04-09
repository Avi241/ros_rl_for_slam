import tensorflow as tf
from matplotlib.pyplot import step
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
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

import dqn_for_slam.environment
from dqn_for_slam.custom_policy import CustomEpsGreedy
from dqn_for_slam.DQN import DQNAgent

ENV_NAME = 'RobotEnv-v0'
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 200
for i in range(1000):
    print("hiiiiiiiiiiiiiiiiiii")

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

file_path = __file__
dir_path = file_path[:(len(file_path) - len('rl_worker.py'))]
MODELS_PATH = dir_path + 'models/'   # model save directory
FIGURES_PATH = dir_path + 'figures/'

ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


def kill_all_nodes() -> None:
    """
    kill all ros node except for roscore
    """
    nodes = os.popen('rosnode list').readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace('\n', '')
    for node in nodes:
        os.system('rosnode kill ' + node)    


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n
    print("actions = ")
    print(nb_actions)
    print("Observation")
    print(env.reset())

    agent = DQNAgent(env)
    for episode in tqdm(range(1,EPISODES+1),ascii = True, unit='episodes'):
        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1

        current_state = env.reset()
        done = False

        while not done:

            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            
            else:
                action = np.random.randint(0,env.action_space.n)
            
            new_state,reward,done,_ = env.step(action)

            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                # env.render()
                pass

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        
        while(1):
            print("done")