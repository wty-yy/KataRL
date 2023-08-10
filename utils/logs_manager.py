import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re, warnings, tensorflow
from matplotlib.ticker import ScalarFormatter

def custom_warning_message(message, category, filename, lineno, file=None, line=None):
        return '%s, line %s,\n%s\n\n' % (filename, lineno, message)
warnings.formatwarning = custom_warning_message

from pathlib import Path
from utils import read_json
keras = tensorflow.keras

# Use regex to match which dataname you want ignore in LogsManager
IGNORE_DATANAME = [  
    r"^cp",  # checkpoint data
    r"^best", # best episode data
]
LEGEND_LOC = {
    # "step": "upper left",
    "q_value": "lower right",
    "v_value": "lower right",
    # "loss": "upper right",
}

# x axis metric name, priority by the idx order
X_METRIC_NAMES = [
    'episode', 'frame'
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
        list,
        keras.metrics.Metric,
    ]

    def __init__(self, init_logs:dict):
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
        # BUGFIX:
        #   Can't use self.logs = self.init_logs.copy()
        #   'keras.metrics.Metric' and 'list' will not be reset
        for key, value in self.logs.items():
            if isinstance(value, int):
                self.logs[key] = 0
            elif isinstance(value, list):
                self.logs[key] = []
            elif isinstance(value, keras.metrics.Metric):
                value.reset_state()
    
    def update(self, keys:list, values:list):
        for key, value in zip(keys, values):
            if value is not None:
                target = self.logs[key]
                if isinstance(target, keras.metrics.Metric):
                    target.update_state(value)
                elif isinstance(target, list):
                    target.append(value)
                elif isinstance(target, int):
                    self.logs[key] = value

    def to_dict(self, drops:list=None):
        ret = self.logs.copy()
        for key, value in ret.items():
            if value is not None:
                target = self.logs[key]
                if isinstance(target, keras.metrics.Metric):
                    ret[key] = round(float(target.result()), 5) \
                        if target.count else None
        if drops is not None:
            for drop in drops:
                ret.pop(drop)
        return ret


class NameCollection:
    """
    To memory the names that load in LogsManager.
    """

    def __init__(self):
        self.names = {'model': [], 'data': [], 'metric': []}
    
    def update(self, key, names):
        if self.names.get(key) is None:
            self.names[key] = names
        else:
            for name in names:
                if name not in self.names[key]:
                    self.names[key].append(name)
        self.names[key] = sorted(self.names[key])
    
    def get_x_metric_name(self):
        for x_metric in X_METRIC_NAMES:
            if x_metric in self.names['metric']:
                return x_metric
        raise Exception(f"\
Error: Don't know the X metric name!\n\
Current metrics: {self.names['metric']}")
    
    def get_metric_names(self):
        ret = self.names['metric'].copy()
        ret.remove(self.get_x_metric_name())
        return ret

def reset_xticks(ax:plt.Axes, min=None, max=None):
    xticks = np.array(ax.get_xticks())
    if min is None: min = xticks.min()
    if max is None: max = xticks.max()
    xticks = xticks[(xticks >= min) & (xticks <= max)]
    xticks = list(np.unique(np.r_[min,xticks,max]))
    ax.set_xlim(left=min, right=max)
    tmp = xticks.copy()
    for i in range(len(tmp)-1):
        delta = tmp[i+1] - tmp[i]
        if delta < tmp[i+1] / 10:
            xticks.remove(tmp[i])
    ax.set_xticks(xticks)

class PlotRange:
    def __init__(self, min=None, max=None):
        self.min, self.max = min, max
    
    def reset(self):
        self.min, self.max = None, None
    
    def update(self, min=None, max=None):
        if min is not None and (self.min is None or min < self.min):
            self.min = min
        if max is not None and (self.max is None or max > self.max):
            self.max = max

    # 加上单引号，解决类的循环导入问题，并且不要重写两个一样的函数名
    # 和C++不同，Python只会保留相同函数名中最后一个
    def update_from_range(self, range:'PlotRange'):
        self.update(range.min, range.max)
    
    def set_ax(self, ax:plt.Axes):
        min, max = self.min, self.max
        if min % 10 != 0:
            min -= self.min % 10
        if max % 10 != 0:
            max += 10 - self.max % 10
        reset_xticks(ax, min, max)

    def __repr__(self) -> str:
        return f"<type=PlotRange, value={(self.min, self.max)}>"

class MyScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"

class LogsManager:
    """
    To plot the logs in such log-files tree:
        -----------------------------------
        /model
            /data-0
                /history-0.json
                /history-1.json
                /...
            /data-1
                /history-0.json
                /history-1.json
                /...
            /data-...
        P.S. The file name is flexible.
        -----------------------------------
    Function:
    - plot(
        data_names, metric_names, model_names,
        to_file, merge_data:bool, dpi:int
    ):  Plot with 'data_names', 'metric_names', 'model_names',
        support two version:
        -   Sparse plot (merge_data=False):
            Plot figures in rxc, r=len(data_names), c=len(metric_names)

            let d_i:=data_names[i], m_j:=metric_names[j], plot result:
                    m_1      m_2      m_3      ...
                 -----------------------------------
                 |        |        |        |
            d_1  |  fig11 |  fig12 |  fig13 |  ...
                 |        |        |        |
                 -----------------------------------
                 |        |        |        |
            d_2  |  fig21 |  fig22 |  fig23 |  ...
                 |        |        |        |
                 -----------------------------------
                 |        |        |        |
            ...  |  ...   |  ...   |  ...   |  ...
                 |        |        |        |
                 -----------------------------------
        -   Merge data plot (merge_data=True):
            Plot figures in 1xc, c=len(metric_names)

            let m_j:=metric_names[j], plot result:
                        m_1      m_2      m_3      ...
                    -----------------------------------
                    |        |        |        |
            merge_d |  fig1  |  fig2  |  fig3  |  ...
                    |        |        |        |
                    -----------------------------------
            where, 'merge_d' is the 95% confidence interval from
            'data_names' (plot by sns.lineplot)

    -   update(path, model_name):
        -   path: The model-logs path, need stisfy the log-files tree.
        -   model_name: The model's name.

    -   reset(): Reset the LogsManager.
    """

    def __init__(self):
        self.painters = {}  # ModelLogsManager
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
            merge_data:bool=False,
            dpi:int=100,
            **kwargs
        ):
        if data_names is None: data_names = self.name_collection.names['data']
        if metric_names is None: metric_names = self.name_collection.get_metric_names()
        if model_names is None: model_names = self.name_collection.names['model']
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
    
    def config_ax(self, ax:plt.Axes, range:PlotRange, metric_name:str):
        ax.grid(True, ls='--', alpha=0.5)
        range.set_ax(ax)
        if metric_name in LEGEND_LOC.keys():
            ax.legend(loc=LEGEND_LOC[metric_name])
        else: ax.legend()
        ax.set_xlabel(self.name_collection.get_x_metric_name())
        ax.xaxis.set_major_formatter(MyScalarFormatter())
        ax.ticklabel_format(style='sci', scilimits=[-2,3])
    
    def plot_sparse(self, axs, data_names, metric_names, model_names, **kwargs):
        for i, data_name in enumerate(data_names):
            range = PlotRange()
            for j, metric_name in enumerate(metric_names):
                range.reset()
                ax = axs[i, j]
                for model_name in model_names:
                    self.painters[model_name].plot(ax, data_name, metric_name, range, **kwargs)
                ax.set_title(f"{data_name} {metric_name}")
                self.config_ax(ax, range, metric_name)
    
    def plot_merge_data(self, axs, data_names, metric_names, model_names, **kwargs):
        # update model data frame first
        for model_name in model_names:
            self.painters[model_name].get_dataframe(data_names)
        range = PlotRange()
        for i, metric_name in enumerate(metric_names):
            ax = axs[0, i]
            range.reset()
            for model_name in model_names:
                print(f"Seaborn is ploting '{model_name}' '{metric_name}'...")
                self.painters[model_name].plot_merge_data(ax, metric_name, range, **kwargs)
            ax.set_title(f"{metric_name}")
            self.config_ax(ax, range, metric_name)
    
class ModelLogsPainter:
    """
    Save model level logs.
    """

    def __init__(self, path:Path, name, name_collection:NameCollection):
        self.path, self.name, self.name_collection = path, name, name_collection
        self.painters = {}  # DataLogsPainter
        self.__read_dir()
        self.df = None
    
    def __read_dir(self):
        path = self.path
        data_names = []
        for f in path.iterdir():
            if f.is_dir() and not is_ignore_data(f.name):
                data_names.append(f.name)
                painter = DataLogsPainter(f, self.name, f.name, self.name_collection)
                self.painters[f.name] = painter
        self.name_collection.update('data', data_names)
    
    def plot(self, ax, data_name, metric_name, range, **kwargs):
        if self.painters.get(data_name) is None:
            warnings.warn(f"Warning: \
Model '{self.name}' don't have data '{data_name}', skip plot it.")
            return
        self.painters[data_name].plot(ax, metric_name, range, **kwargs)
    
    def get_dataframe(self, data_names):
        self.df = None
        for data_name in data_names:
            if self.painters.get(data_name) is None:
                warnings.warn(f"Warning: \
Model '{self.name}' don't have data '{data_name}', skip merge it.")
                continue
            painter = self.painters[data_name]
            if self.df is None: self.df = painter.to_df()
            else: self.df = pd.concat([self.df, painter.to_df()]).reset_index(drop=True)
        self.df_notna = ~self.df.groupby(
            self.name_collection.get_x_metric_name()
        ).mean().isna()
    
    def get_notna_range(self, metric_name) -> PlotRange:
        true_index = self.df_notna[self.df_notna[metric_name]].index
        return PlotRange(true_index.min(), true_index.max())

    def plot_merge_data(self, ax, metric_name, range:PlotRange, **kwargs):
        if self.df is None:
            raise Exception("Error: model_logs_painter.df=None, it hasn't init!")
        range.update_from_range(self.get_notna_range(metric_name))
        sns.lineplot(
            ax=ax, data=self.df,
            x=self.name_collection.get_x_metric_name(),
            y=metric_name, label=self.name, **kwargs
        )

def get_suffix(path):
    return path.name.split('.')[-1]

class DataLogsPainter:
    """
    Save data level logs. Can merge *.json files.
    """

    suffix_list = ['csv', 'json']

    def __init__(self, path:Path, model_name, data_name, name_collection=None):
        self.path, self.name_collection = path, name_collection
        self.model_name, self.data_name = model_name, data_name
        self.metric_names = []
        self.logs = {}
        self.__read_dir()
    
    def __read_dir(self):
        files = []
        for f in self.path.iterdir():
            if is_ignore_data(f.name): continue
            if get_suffix(f) not in self.suffix_list:
                warnings.warn(f"Warning: Could not read file '{f}'!")
            else: files.append(f)
        files = sorted(files)
        for idx, f in enumerate(files):
            suffix = get_suffix(f)
            if suffix == 'csv': self.__read_csv(f)
            elif suffix == 'json': self.__read_json(f)
    
    def __read_csv(self, path): pass

    def __read_json(self, path):
        logs = read_json(path)
        self.name_collection.update('metric', list(logs.keys()))
        for key, value in logs.items():
            if self.logs.get(key) is None:
                self.logs[key] = []
            self.logs[key] += value
    
    def plot(self, ax, metric_name, range:PlotRange, **kwargs):
        if self.logs.get(metric_name) is None:
            warnings.warn(f"Warning: \
Model '{self.model_name}' data '{self.data_name}' don't have\
metric '{metric_name}', skip plot it.")
            return
        x = np.array(self.logs.get(self.name_collection.get_x_metric_name()))
        ndim = np.array(self.logs[metric_name]).ndim
        if ndim != 1:
            raise Exception(f"Error: Could not plot {self.model_name}-{self.data_name}-{metric_name} since it's ndim={ndim} not 1")
        ax.plot(x, self.logs[metric_name], label=self.model_name, **kwargs)
        not_nan = ~np.isnan(np.array(self.logs[metric_name], dtype='float32'))
        range.update(min=x[not_nan].min(), max=x[not_nan].max())
    
    def to_df(self):
        return pd.DataFrame(self.logs)
        
if __name__ == '__main__':
    logs_manager = LogsManager()
    VGG16_path = Path('/home/wty/Coding/replicate-papers(local)/VGG16/logs/history')
    logs_manager.update(VGG16_path, 'VGG16')
    # logs_manager.plot(data_names=['train'], metric_names=['loss', 'Top1'])
    logs_manager.plot(to_file=True)
    plt.show()
