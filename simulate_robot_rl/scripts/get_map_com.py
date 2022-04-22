import gym
import robot_rl_env

env = gym.make("RobotEnv-v0")
while(True):
    env.map_value()