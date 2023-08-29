from agents.constants import PATH
from utils import save_json, save_npy
from utils.logs_manager import LogsManager

class History:
    """
    Save episode logs (utils/logs_manager.py) 'Logs.to_dir()'.
    **
    This class will be instantiated in Agent class(parent),
    to variable 'best_episode' and 'history'
    **

    Initialize:
    -   agent_name: The name of Agent.
    -   agent_id: The id of Agent.
    -   timestr: The timestamp when the Agent was created.
    -   fname: The saving file's prefix.
    -   type: The saving file's suffix.

    Function:
    - update_item(key, value): Update item with key and value.

    - update_dict(d:dict): Add dict of 'd' to history.

    - update_best(now, logs):
        Compare 'now' with 'self.best', if 'now' is bigger,
        then update history with 'logs'.

    - to_file():
        The history will be saved at '/logs/agent_name/history-id',
        with file name 'fname-timestr.type'

    - plot():
        Use `utils/logs_manager` to plot the history.
    """

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
