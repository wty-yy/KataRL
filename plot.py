from agents.constants import PATH
from utils.logs_manager import LogsManager
from utils import get_time_str, read_npy
from utils.generate_gif import save_frames_as_gif
import matplotlib.pyplot as plt
import time
from pathlib import Path
import argparse

class PlotManager:

    def __init__(
            self, load_path:Path, save_path:Path, fname,
            model_name, data_names, metric_names
        ):
        self.load_path, self.save_path, self.fname, \
        self.model_name, self.data_names, self.metric_names = \
            load_path, save_path, fname, \
            model_name, data_names, metric_names
        self.logs_manager = LogsManager()
    
    def get_names(self):
        self.update()
        return self.logs_manager.name_collection.names
    
    def update(self):
        self.logs_manager.update(path=self.load_path, model_name=self.model_name)

    def plot_logs(self, verbose=True):
        self.logs_manager.plot(
            data_names=self.data_names,
            metric_names=['step', 'q_value', 'loss'],
            to_file=self.save_path.joinpath(f"{self.fname}.png")
        )
        plt.close()
        if verbose:
            print(f"update figure at {get_time_str()}")
    
    def plot_cycle(self, cycle_time=10):
        while True:
            self.update()
            self.plot_logs()
            time.sleep(cycle_time)

    def plot_frame(self, file_path):
        path = self.load_path.joinpath(file_path)
        logs = read_npy(path)
        logs_show = logs.copy(); logs_show.pop('frame')
        print(f"Load file '{path.absolute()}'")
        print(logs_show)
        print("start convert to gif...")
        save_frames_as_gif(logs['frame'], self.fname, self.save_path)

def get_plot_manager(model_name, idx=None):
    PATH.get_logs_path(model_name)
    timestamp = get_time_str()
    data_names = None if idx is None else [f"history-{idx:04}"]
    plot_manager = PlotManager(
        load_path=PATH.AGENT,
        save_path=PATH.FIGURES,
        fname=timestamp,
        model_name=model_name,
        data_names=data_names,
        metric_names=['step', 'q_value', 'loss'],
    )
    print(plot_manager.get_names())
    return plot_manager

def load_parse():
    parser = argparse.ArgumentParser(description="Plot figure from PATH.AGENT")
    parser.add_argument('-i', '--id', type=int, help="Id of Data (int)")
    parser.add_argument('-m', '--model', default='DQN', help="The type of model (str)")
    parser.add_argument('-pc', '--plot-cycle', action='store_true', help="Whether plot cyclely (bool)")
    parser.add_argument('-pf', '--plot-frame', action='store_true', help="Plot the frames from file (bool)")
    parser.add_argument('-ff', '--frame-file', type=str, help="The frame file path (str)")
    args = parser.parse_args()
    if args.id is None:
        raise Exception("Args Error: Need add '-i, --id' of data")
    print(args)

    plot_manager = get_plot_manager(args.model, args.id)
    if args.plot_cycle:
        plot_manager.plot_cycle()
    if args.plot_frame:
        if args.frame_file is None:
            raise Exception("Args Error: Need add '-ff, --frame-file' for the frame path")
        plot_manager.plot_frame(args.frame_file)

def plot_merge_data():
    plot_manager = get_plot_manager('DQN')
    plot_manager.logs_manager.plot(data_names=[
        # "history-0000",
        # "history-0001",
        # "history-0002",
        # "history-0003",
        # "history-0004",
        # "history-0005",
        # "history-0006",
        # "history-0007",
        # "history-0008",
        # "history-0009",
        "history-0010",
        "history-0011",
        "history-0012",
        "history-0013",
        "history-0014",
        "history-0015",
        "history-0016",
        "history-0017",
        # "history-0018",
        # "history-0019",
    ], to_file=PATH.FIGURES.joinpath(get_time_str()+'.png'), merge_data=True)
    # plt.show()

if __name__ == '__main__':
    load_parse()
