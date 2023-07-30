import numpy as np
from datetime import datetime
import json

def make_onehot(x:np.ndarray, depth=None):
    if depth is None: depth = x.max() + 1
    x = x.squeeze()
    ret = np.zeros((x.size, depth))
    ret[np.arange(x.size), x] = 1
    return ret

def get_time_str():
    return datetime.now().strftime(r"%Y%m%d-%H%M%S")

def save_json(path, value):
    with open(path, 'w') as file:
        json.dump(value, file, indent=2)

def read_json(path):
    with open(path, 'r') as file:
        ret = json.load(file)
    return ret

def save_npy(path, value):
    np.save(path, value)

def read_npy(path) -> dict:
    return np.load(path, allow_pickle=True).item()

if __name__ == '__main__':
    action = np.array([1, 0, 1, 1, 0])
    print(make_onehot(action).astype('bool'))
