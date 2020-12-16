from spinup import trpo_tf1 as trpo
import tensorflow as tf
import gym

env_fn = lambda : gym.make('Ant-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='../data/trpo/bant_4000_750', exp_name='ant_trpo')

trpo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=750, logger_kwargs=logger_kwargs)