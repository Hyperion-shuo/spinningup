import  numpy as np
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)
observation_space = env.observation_space
action_space = env.action_space
print("observation_space.shape:", observation_space.shape, " action_space.shape:", action_space.shape)
print("observation_space.shape[0]:", observation_space.shape[0], " action_space.shape.n:", action_space.n)
print("env.observation_space.high: ", env.observation_space.high)
print("env.observation_space.low: ", env.observation_space.low)