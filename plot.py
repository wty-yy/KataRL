from pathlib import Path
from agents.constants import PATH
from utils.logs_manager import LogsManager
from utils import get_time_str, read_npy
from utils.generate_gif import save_frames_as_gif
import matplotlib.pyplot as plt
import time
import warnings
import argparse

class PlotManager:

    def __init__(
            self, load_paths:Path, save_path:Path, fname,
            model_names, data_names, metric_names, merge_data=False,
            alpha=0.5
        ):
        self.load_paths, self.save_path, self.fname, \
        self.model_names, self.data_names, self.metric_names, \
        self.merge_data, self.alpha = \
            load_paths, save_path, fname, \
            model_names, data_names, metric_names, merge_data, alpha
        self.logs_manager = LogsManager()
    
    def get_names(self):
        self.update()
        return self.logs_manager.name_collection.names
    
    def update(self):
        for load_path, model_name in zip(self.load_paths, self.model_names):
            self.logs_manager.update(path=load_path, model_name=model_name)

    def plot_logs(self, verbose=True):
        self.logs_manager.plot(
            data_names=self.data_names,
            # metric_names=['step', 'q_value', 'loss'],
            to_file=self.save_path.joinpath(f"{self.fname}.png"),
            merge_data=self.merge_data,
            alpha=self.alpha,
        )
        plt.close()
        if verbose:
            print(f"update figure at {get_time_str()}")
    
    def plot_cycle(self, cycle_time=10):
        while True:
            # try:
            self.update()
            self.plot_logs()
            time.sleep(cycle_time)
            # except KeyboardInterrupt:
            #     break
            # except:
            #     continue

    def plot_frame(self, file_path):
        path = self.load_paths[0].joinpath(file_path)
        logs = read_npy(path)
        logs_show = logs.copy(); logs_show.pop('frame')
        print(f"Load file '{path.absolute()}'")
        print(logs_show)
        print("start convert to gif...")
        save_frames_as_gif(logs['frame'], self.fname, self.save_path)

def get_plot_manager(model_names, idxs=None, merge_data=False, alpha=0.5) -> PlotManager:
    timestamp = get_time_str()
    data_names = None if idxs is None else [f"history-{idx:04}" for idx in idxs]
    load_paths = [PATH.LOGS.joinpath(model_name) for model_name in model_names]
    plot_manager = PlotManager(
        load_paths=load_paths,
        save_path=PATH.FIGURES,
        fname=timestamp,
        model_names=model_names,
        data_names=data_names,
        metric_names=['step', 'q_value', 'loss'],
        merge_data=merge_data,
        alpha=alpha,
    )
    print(plot_manager.get_names())
    return plot_manager

def load_parse():
    parser = argparse.ArgumentParser(description="Plot figure from PATH.AGENT")
    parser.add_argument('-i', '--id', nargs='+', default=None, type=int, help="Id of Data (int)")
    parser.add_argument('-m', '--model', nargs='+', default=['DQN'], type=str, help="The type of model (str)")
    parser.add_argument('-pc', '--plot-cycle', action='store_true', help="Whether plot cyclely (bool)")
    parser.add_argument('--merge', action='store_true', help="Plot the data after merge")
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help="Alpha of figure")
    args = parser.parse_args()
    if args.id is None and args.merge is False:
        warnings.warn("The id is 'None' and merge is False, so you want plot all data?")
    print(args)

    plot_manager = get_plot_manager(args.model, args.id, args.merge, args.alpha)
    if args.plot_cycle:
        plot_manager.plot_cycle()
    else:
        plot_manager.plot_logs()

if __name__ == '__main__':
    load_parse()
