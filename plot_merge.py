from pathlib import Path
import argparse, warnings
from utils import get_time_str
from utils.logs_manager import LogsManager
from agents.constants import PATH

def load_parse():
    parser = argparse.ArgumentParser(description="Plot merge model logs.")
    parser.add_argument("-p", "--path", type=str, help="Logs path.")
    parser.add_argument(
        "-m", "--model", nargs="+",
        help="Model names in 'path' to plot."
    )
    parser.add_argument(
        "-a", "--alpha", default=1.0, type=float,
        help="The transparency of the curve, range=(0,1)."
    )
    parser.add_argument(
        "-dpi", default=100, type=int,
        help="The dpi of saving figure."
    )

    args = parser.parse_args()
    if args.path is None:
        raise Exception("Error: Don't have logs path!")
    logs_path = Path.cwd().joinpath(args.path)
    if not logs_path.exists() or not logs_path.is_dir():
        raise Exception(f"Error: Fold '{logs_path.absolute()}' don't exist!")
    model_names = args.model
    if model_names is None:
        model_names = []
        for model_name in logs_path.iterdir():
            if not model_name.is_dir(): continue
            model_names.append(model_name.name)
        warnings.warn("\
Warning: The model_names is None,\
it will plot all the model under the logs path")
    logs_manager = LogsManager()
    for model_name in model_names:
        logs_manager.update(logs_path.joinpath(model_name), model_name)
    print(logs_manager.name_collection.names)
    logs_manager.plot(
        to_file=PATH.FIGURES.joinpath(get_time_str()+'-merge.png'),
        merge_data=True, alpha=args.alpha, dpi=args.dpi
    )
    

if __name__ == '__main__':
    load_parse()