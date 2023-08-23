import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

class MeanMetric():

    count: int = 0
    mean: float = 0

    def update_state(self, value):
        value = float(value)
        self.count += 1
        self.mean += (value - self.mean) / self.count
    
    def reset_state(self):
        self.count, self.mean = 0, 0.
    
    def result(self):
        return self.mean


class Logs:
    """
    The logs during 'one episode' of the Agent.
    
    -   support_type (list): The type that logs support to update

    Initialize:
    -   init_logs (dict): What logs you want to memory, such like:
            init_logs = {
                'episode': 0,
                'step': 0,
                'q_value': keras.metrics.Mean(name='q_value'),
                'loss': keras.metrics.Mean(name='loss'),
                'frame': []
            }
    
    Function
    -   reset(): reset the logs, use it after one episode.

    -   update(keys:list, values:list):
            update 'zip(keys, values)' one by one in logs.

    -   to_dict(drops:list):
            Cast 'logs' to 'dict' for saving to 'utils/History' class.
            -   drops: The keys in logs that you don't want to return.
    """
    support_type = [
        int,
        float,
        list,
        MeanMetric
    ]

    def __init__(self, init_logs:dict):
        self.start_time = time.time()
        for key, value in init_logs.items():
            flag = False
            for type in self.support_type:
                if isinstance(value, type):
                    flag = True; break
            if not flag:
                raise Exception(
                    f"Error: Don't know {key}'s type '{type(value)}'!"
                )
        self.logs = init_logs
    
    def reset(self):
        self.start_time = time.time()
        # BUGFIX:
        #   Can't use self.logs = self.init_logs.copy()
        #   'keras.metrics.Metric' and 'list' will not be reset
        for key, value in self.logs.items():
            if isinstance(value, list):
                self.logs[key] = []
            elif isinstance(value, MeanMetric):
                value.reset_state()
            else:  # int or float
                self.logs[key] = 0
    
    def update(self, keys:list, values:list):
        for key, value in zip(keys, values):
            if value is not None:
                target = self.logs[key]
                if isinstance(target, MeanMetric):
                    target.update_state(value)
                elif isinstance(target, list):
                    target.append(value)
                else: # int or float
                    self.logs[key] = value

    def to_dict(self, drops:list=None):
        ret = self.logs.copy()
        for key, value in ret.items():
            if value is not None:
                target = self.logs[key]
                if isinstance(target, MeanMetric):
                    ret[key] = target.result() if target.count else None
        if drops is not None:
            for drop in drops:
                ret.pop(drop)
        return ret
    
    def get_time_length(self):
        return time.time() - self.start_time
    