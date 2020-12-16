from spinup import vpg_tf1 as vpg
import tensorflow as tf
import gym

env_fn = lambda : gym.make('Ant-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='../data/vpg/ant', exp_name='ant_vpg')

vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=750, logger_kwargs=logger_kwargs)