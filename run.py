from agents.DQN import DQN
from envs.gym_env import GymEnv

if  __name__ == '__main__':
    start_idx = 0
    N = 10
    for idx in range(start_idx, start_idx + N):
        print(f"{idx}/{N}:")
        dqn = DQN(
            agent_name='DQN-16-1e4-1e6-1000',
            env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
            verbose=False, agent_id=idx, episodes=1000, model_id=None
        )
        dqn.train()