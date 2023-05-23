import importlib
import functools
import yaml
import numpy as np
from tqdm import tqdm
import skimage


@functools.lru_cache(1)
def load_model(model_id, input_shape):
    resource_module = importlib.import_module(
        f"eyeseg.models.weights.{model_id}")

    config_path = importlib.resources.path(resource_module, "config.yaml")
    weights_path = importlib.resources.path(resource_module, "model-best.h5")
    with config_path as path:
        with open(path, "r") as stream:
            model_config = yaml.safe_load(stream)

    my_module = importlib.import_module(model_config["module"])
    layer_model = my_module.model(input_shape=input_shape,
                                  **model_config["parameters"])

    with weights_path as path:
        layer_model.load_weights(path)
    return layer_model, model_config


def preprocess_spectralis(data, input_width):
    image = np.zeros((512, input_width))
    image[:496, :input_width] = data
    image = image - np.mean(image)
    image = image / np.std(image)
    image = np.reshape(image, (1, 512, input_width, 1))
    return image

def preprocess_bioptigen(data, input_width):
    image = data
    image = image - np.mean(image)
    image = image / np.std(image)
    image = np.reshape(image, (1, 512, input_width, 1))
    return image

def preprocess(data, input_width, device):
    if device == "spectralis":
        return preprocess_spectralis(data, input_width)
    elif device == "bioptigen":
        return preprocess_bioptigen(data, input_width)
    else:
        raise ValueError(f"Unknown device {device}")


def get_layers(data, model_id, device="spectralis"):
    if data.meta["scale_x"] < 0.009 and device=="spectralis":
        factor = 2
    else:
        factor = 1

    width = data[0].shape[1]
    layer_model, model_config = load_model(model_id, (512, width // factor, 1))
    results = []
    for bscan in tqdm(data, desc=f"Predict Volume: ", position=1, leave=False):
        img = skimage.transform.rescale(bscan.data, (1, 1 / factor))
        img = preprocess(img, width // factor, device)
        prediction = layer_model.predict(img, verbose=0)[0]
        results.append(prediction)

    results = np.stack(results, axis=0)
    for index, name in model_config["layer_mapping"].items():
        if factor != 1:
            height_map = skimage.transform.rescale(results[..., index],
                                                   (1, factor))
            # height_map = np.interp(np.arange(width), np.arange(width//factor) * factor, results[..., index])
        else:
            height_map = results[..., index]
        data.add_layer_annotation(height_map, name=name)
    return data
