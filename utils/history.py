from agents.constants import PATH
from utils import save_json, save_npy
from utils.logs_manager import LogsManager

class History:

    def __init__(self, agent_name, agent_id, timestr, fname, type=None):
        self.agent_name, self.agent_id, self.timestr, self.fname, self.type = \
            agent_name, agent_id, timestr, fname, type
        self.best = None
        self.logs = {}
        self.path = PATH.HISTORY.joinpath(f"{self.fname}-{self.timestr}.{self.type}")
        self.logs_manager = LogsManager()
    
    def update_item(self, key, value):
        if self.logs.get(key) is None:
            self.logs[key] = []
        self.logs[key].append(value)
    
    def update_dict(self, d:dict):
        for key, value in d.items():
            self.update_item(key, value)
        self.to_file()
    
    def update_best(self, now, logs):
        if self.best is None or now > self.best:
            self.best, self.logs = now, logs
        self.to_file()
    
    def to_file(self):
        if self.type == 'json':
            save_json(self.path, self.logs)
        elif self.type == 'npy':
            save_npy(self.path, self.logs)
    
    def plot(self):
        self.logs_manager.update(PATH.AGENT, self.agent_name)
        self.logs_manager.plot()
