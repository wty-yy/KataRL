# -*- coding: utf-8 -*-
'''
@File    : dqn.py
@Time    : 2023/08/26 12:00:32
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023.8.26. 优化DQN参数，
加入参数train_frequency, total_timesteps, num_envs
'''

total_timesteps = int(5e5)  # new
num_envs = 4  # new
learning_rate = 2.5e-4  # 1e-3 -> 2.5e-4
gamma = 0.99  # discount rate 0.95 -> 0.99

epsilon_max = 1.0
epsilon_min = 0.05  # 0.01 -> 0.05
exporation_proportion = 0.5  # new

memory_size = int(1e4)  # 1e6 -> 1e4
start_fit_size = int(1e4)

batch_size = 64  # 4 -> 128
train_frequency = 10  # new
write_logs_frequency = 10  # new
# episodes = 1000  # remove
