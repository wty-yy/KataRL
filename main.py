from run.DQN.cartpole import DQN_cartpole_train, DQN_cartpole_eval
from run.A2C.cartpole import A2C_cartpole_train, A2C_cartpole_eval

if  __name__ == '__main__':
    # DQN_cartpole_train()
    # DQN_cartpole_eval(agent_id=0, load_id=9, batch_size=1, episodes=5)
    A2C_cartpole_train()
    # A2C_cartpole_eval(agent_id=11, load_id=9, episodes=10)