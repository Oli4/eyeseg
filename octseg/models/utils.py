import importlib
import functools
import yaml


@functools.lru_cache(1)
def load_model(model_id, input_shape):
    resource_module = importlib.import_module(f"octseg.models.weights.{model_id}")

    config_path = importlib.resources.path(resource_module, "config.yaml")
    weights_path = importlib.resources.path(resource_module, "model-best.h5")
    with config_path as path:
        with open(path, "r") as stream:
            model_config = yaml.safe_load(stream)

    my_module = importlib.import_module(model_config["module"])
    layer_model = my_module.model(input_shape=input_shape, **model_config["parameters"])

    with weights_path as path:
        layer_model.load_weights(path)
    return layer_model, model_config
