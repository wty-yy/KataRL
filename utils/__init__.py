"""
General functions:

-   make_onehot(x, depth): Convert x to one-hot vectors with depth.

-   sample_from_proba(proba): Sample indexs from the 'proba'.

-   get_time_str(): Return current datetime by str.

-   'json', 'npy' file save/read:
    (convert '*' with 'json' or 'npy')
    -   save_*(path, value): Save value to path, use type of *.
    -   read_*(path): Read file from path, use type of *.
"""

import numpy as np
from datetime import datetime
import json

def make_onehot(x:np.ndarray, depth=None):
    if depth is None: depth = x.max() + 1
    x = x.squeeze()
    ret = np.zeros((x.size, depth))
    ret[np.arange(x.size), x] = 1
    return ret

def sample_from_proba(proba:np.ndarray):
    # proba[np.isnan(proba)] = 0
    # print(proba)
    choice = lambda p: np.random.choice(len(p), 1, p=p)[0]
    try:
        action = np.apply_along_axis(choice, axis=1, arr=proba).astype('int32')
    except:
        print("sample proba GG, sample randomly", proba)
        raise
    return action

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
