from pathlib import Path

def mkdir(path:Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

class PathManager:

    def __init__(self):
        self.LOGS = mkdir(Path.cwd().joinpath("logs"))  # logs path
        self.FIGURES = mkdir(self.LOGS.joinpath("figures"))

        # These paths are waiting for init at Agent.__init__
        self.AGENT = Path()
        # self.HISTORY = Path()
        self.CHECKPOINTS = Path()

    def get_logs_path(self, agent_name: str):
        self.AGENT = mkdir(self.LOGS.joinpath(f"{agent_name}"))
        # self.HISTORY = mkdir(self.AGENT.joinpath(f"history-{agent_id:04}"))
        self.CHECKPOINTS = mkdir(self.AGENT.joinpath(f"checkpoints"))
        # print(f"{self.AGENT=}\n{self.HISTORY=}\n{self.CHECKPOINTS=}")

PATH = PathManager()