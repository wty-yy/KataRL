total_timesteps = int(5e5)
num_envs = 4
gamma = 0.99
memory_size = int(1e5)
start_fit_size = int(1e4)
tau = 1.0
batch_size = 128
learning_rate_p = 2.5e-4
learning_rate_q = 2.5e-4
target_model_update_frequency = 1000
train_frequency = 10
alpha = 0.2
flag_autotune_alpha = True
learning_rate_alpha = 1e-4
coef_target_entropy = 0.75  # !!!
# max_grad_clip_norm = 0.5
# flag_anneal_lr = True
flag_anneal_lr = True
EPS = 1e-5
write_logs_frequency = 1000