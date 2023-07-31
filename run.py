from agents.DQN import DQN
from envs.gym_env import GymEnv

# DQN-hyper-args
args = {
    "batch_size": [1, 2, 3, 4, 6, 8, 16],
    "memory_size": 1e6,
    "start_fit_size": 1e4,
    "episodes": 1000
}

if  __name__ == '__main__':
    start_idx = 0
    N = 1
    for batch_size in args['batch_size']:
        for idx in range(start_idx, start_idx + N):
            print(f"{idx}/{N}:")
            dqn = DQN(
                agent_name=f'DQN-{batch_size}',
                env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
                verbose=False, agent_id=idx, episodes=1000, load_id=None,
                batch_size=batch_size
            )
            dqn.train()