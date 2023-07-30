from agents.models import Model
from agents.models.MLP import MLP
name2model = {
    'MLP': MLP
}

def build_model(model_name, input_shape, load_id) -> Model:
    if name2model.get(model_name) is None:
        raise Exception(f"Not find model '{model_name}'")
    return name2model[model_name](input_shape, load_id)