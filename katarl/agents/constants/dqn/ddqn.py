total_timesteps = int(5e5)
num_envs = 4
learning_rate = 2.5e-4
gamma = 0.99  # discount rate

epsilon_max = 1.0
epsilon_min = 0.05
exporation_proportion = 0.5

memory_size = int(1e4)
start_fit_size = int(1e4)

batch_size = 128
train_frequency = 10
target_model_update_frequency = 500
tau = 1.
write_logs_frequency = 1000