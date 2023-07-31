import re, warnings
from pathlib import Path
from utils import read_json, get_time_str
from agents.constants import PATH
IGNORE_DATANAME = [  # use regex to match which you want ignore
    r"^cp",  # checkpoint data
    r"^best", # best episode data
]

def is_ignore_data(name):
    for s in IGNORE_DATANAME:
        if len(re.findall(s, name)) > 0:
            return True
    return False

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif', 'SimSun']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

class NameCollection:

    def __init__(self):
        self.names = {'model': [], 'data': [], 'metric': []}
    
    def update(self, key, names):
        if len(self.names[key]) < len(names):
            self.names[key] = names

class LogsManager:

    def __init__(self):
        self.painters = {}
        self.name_collection = NameCollection()

    def update(self, path, model_name):
        painter = ModelLogsPainter(path, model_name, self.name_collection)
        self.painters[model_name] = painter
        self.name_collection.update('model', list(self.painters.keys()))
    
    def reset(self):
        del self.painters
        self.painters = {}
    
    def plot(
            self, data_names=None, metric_names=None, 
            model_names=None, 
            to_file=None, 
            merge_data=False,
            dpi=100,
            **kwargs
        ):
        if data_names is None: data_names = self.name_collection.names['data']
        if metric_names is None: metric_names = self.name_collection.names['metric']
        if model_names is None: model_names = self.name_collection.names['model']
        if merge_data:
            metric_names.remove('episode')
        r = len(data_names)
        c = len(metric_names)
        if r == 0 or c == 0:
            raise Exception("Error: Don't find any data!")
        if merge_data: r = 1
        fig, axs = plt.subplots(r, c, figsize=(c*4, r*4))
        axs = axs.reshape(r, c)
        if merge_data:
            self.plot_merge_data(axs, data_names, metric_names, model_names, **kwargs)
        else:
            self.plot_sparse(axs, data_names, metric_names, model_names, **kwargs)
        fig.tight_layout()
        if to_file:
            fig.savefig(to_file, dpi=dpi)
    
    def plot_sparse(self, axs, data_names, metric_names, model_names, **kwargs):
        for i, data_name in enumerate(data_names):
            for j, metric_name in enumerate(metric_names):
                for model_name in model_names:
                    ax = axs[i, j]
                    self.painters[model_name].plot(ax, data_name, metric_name, **kwargs)
                    ax.set_title(f"{data_name} {metric_name}")
                    ax.grid(True, ls='--', alpha=0.5)
                    ax.set_xlim(left=0)
                    ax.legend()
    
    def plot_merge_data(self, axs, data_names, metric_names, model_names, **kwargs):
        # update model data frame first
        for model_name in model_names:
            self.painters[model_name].get_dataframe(data_names)
        for i, metric_name in enumerate(metric_names):
            for model_name in model_names:
                print(f"Seaborn is ploting '{metric_name}'...")
                ax = axs[0, i]
                self.painters[model_name].plot_merge_data(ax, metric_name, **kwargs)
                ax.set_title(f"{metric_name}")
                ax.grid(True, ls='--', alpha=0.5)
                ax.set_xlim(left=0)
                ax.legend(loc='upper left')
    
class ModelLogsPainter:

    def __init__(self, path:Path, name, name_collection):
        self.path, self.name, self.name_collection = path, name, name_collection
        self.painters = {}
        self.epoch_sizes = None
        self.__read_dir()
        self.df = None
    
    def __read_dir(self):
        path = self.path
        path_epoch_sizes = path.joinpath("epoch_sizes.json")
        if path_epoch_sizes.exists():
            self.epoch_sizes = read_json(path_epoch_sizes)
        data_names = []
        for f in path.iterdir():
            if f.is_dir() and not is_ignore_data(f.name):
                data_names.append(f.name)
                painter = DataLogsPainter(f, self.name, f.name, self.epoch_sizes, self.name_collection)
                self.painters[f.name] = painter
        self.name_collection.update('data', data_names)
    
    def plot(self, ax, data_name, metric_name, **kwargs):
        self.painters[data_name].plot(ax, metric_name, **kwargs)
    
    def get_dataframe(self, data_names):
        self.df = None
        for data_name in data_names:
            painter = self.painters[data_name]
            if self.df is None: self.df = painter.to_df()
            else: self.df = pd.concat([self.df, painter.to_df()]).reset_index(drop=True)

    def plot_merge_data(self, ax, metric_name, **kwargs):
        if self.df is None:
            raise Exception("Error: model_logs_painter.df=None, it hasn't init!")
        sns.lineplot(ax=ax, data=self.df, x='episode', y=metric_name, label=self.name, **kwargs)

def get_suffix(path):
    return path.name.split('.')[-1]

class DataLogsPainter:

    suffix_list = ['csv', 'json']

    def __init__(self, path:Path, model_name, data_name, epoch_sizes=None, name_collection=None):
        self.path, self.epoch_sizes, self.name_collection = path, epoch_sizes, name_collection
        self.epoch_counts = []
        self.model_name, self.data_name = model_name, data_name
        self.metric_names = []
        self.logs = {}
        self.__read_dir()
    
    def __read_dir(self):
        if self.epoch_sizes is None:
            path_epoch_sizes = self.path.joinpath("epoch_sizes.json")
            if not path_epoch_sizes.exists():
                # raise Exception(f"Error: {self.model_name}'s {self.data_name} dataset don't have 'epoch_sizes.json' file!")
                warnings.warn("Warning: {self.model_name}'s {self.data_name} dataset don't have 'epoch_sizes.json' file, default by 1")
            else: self.epoch_sizes = read_json(path_epoch_sizes)
        files = []
        for f in self.path.iterdir():
            if is_ignore_data(f.name): continue
            if get_suffix(f) not in self.suffix_list:
                warnings.warn(f"Warning: Could not read file '{f}'!")
            elif f.name != "epoch_sizes.json":
                files.append(f)
        files = sorted(files)
        if self.epoch_sizes is None:
            self.epoch_sizes = [1 for _ in range(len(files))]
        if len(self.epoch_sizes) != len(files):
            raise Exception(f"Error: {self.model_name}'s {self.data_name} epoch_size length {len(self.epoch_sizes)} != files number {len(files)}")
        for idx, f in enumerate(files):
            suffix = get_suffix(f)
            if suffix == 'csv': self.__read_csv(f, self.epoch_sizes[idx])
            elif suffix == 'json': self.__read_json(f, self.epoch_sizes[idx])
    
    def __read_csv(self, path): pass

    def __read_json(self, path, epoch_size):
        logs = read_json(path)
        self.name_collection.update('metric', list(logs.keys()))
        for key, value in logs.items():
            if len(value) % epoch_size != 0:
                raise Exception(f"Error: {self.model_name}-{self.data_name} '{path.name}' {key}'s length {len(value)} % epoch size {epoch_size} != 0")
            if self.logs.get(key) is None:
                self.logs[key] = []
            self.logs[key] += value
        self.epoch_counts.append(len(value) // epoch_size)
    
    def plot(self, ax, metric_name, **kwargs):
        x = []
        now = 0
        for size, count in zip(self.epoch_sizes, self.epoch_counts):
            x = np.concatenate([x, now + np.arange(1, count * size + 1) / size])
            now += count
        ndim = np.array(self.logs[metric_name]).ndim
        if ndim != 1:
            raise Exception(f"Error: Could not plot {self.model_name}-{self.data_name}-{metric_name} since it's ndim={ndim} not 1")
        ax.plot(x, self.logs[metric_name], label=self.model_name, **kwargs)
    
    def to_df(self):
        return pd.DataFrame(self.logs)
        
if __name__ == '__main__':
    logs_manager = LogsManager()
    VGG16_path = Path('/home/wty/Coding/replicate-papers(local)/VGG16/logs/history')
    logs_manager.update(VGG16_path, 'VGG16')
    # logs_manager.plot(data_names=['train'], metric_names=['loss', 'Top1'])
    logs_manager.plot(to_file=True)
    plt.show()
