import click
from pathlib import Path

import tensorflow as tf

from octseg.models.utils import load_model

import importlib
import octseg

from tqdm import tqdm
import numpy as np
import eyepy as ep
import pickle


@click.command()
@click.option(
    "--gpu", type=int, default=0, help="Number of the GPU if more than one is available"
)
@click.option(
    "--output_path",
    type=click.Path(exists=True),
    help="Location to store the results. The default is processed/ in data_path",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Whether to overwrite existing layers.",
)
@click.argument("data_path", type=click.Path(exists=True), default="/home/data")
@click.argument("model_id", type=click.STRING, default="2c41ukad")
def layers(data_path, output_path, model_id, overwrite, gpu):
    """Predict layers for data in data_path using the model with ID model_id

    \b
    MODEL_ID: Specifies the model
        Pretrained models:
            Spectralis:
                2c41ukad: 3 classes (BM, RPE, EZ) - (Default)
            Bioptigen:
                3avqygsi: 3 classes (BM, IBRPE, ILM)

    DATA_PATH: Path to your data. Currently only Spectrals XML and VOL exports are supported.
    \f
    """
    data_path = Path(data_path)
    if output_path is None:
        output_path = Path(data_path) / "processed"
    else:
        output_path = Path(output_path)

    # Check if specified model is available
    if not model_id in list(importlib.resources.contents(octseg.models.weights)):
        raise ValueError("The specified model is not available. Check the --help")

    # Find volumes
    if data_path.is_dir():
        vol_volumes = data_path.glob("**.vol", recursive=True)
        xml_volumes = data_path.glob("**.xml", recursive=True)
        # We do not support multiple XML exports in the same folder.
        xml_volumes = [v.parent for v in xml_volumes]
    elif data_path.is_file():
        if ".vol" == data_path.suffix:
            vol_volumes = [data_path]
            xml_volumes = []
        if ".xml" == data_path.suffix:
            xml_volumes = [data_path]
            vol_volumes = []
    else:
        raise ValueError("Data not found")

    # Check for which volumes layers need to be predicted
    if overwrite == False:
        # Remove path from volumes if layers are found in the output location
        precomputed_layers = [
            p.name for p in output_path.iterdir() if (p / "layers.pkl").exists()
        ]
        vol_volumes = [v for v in vol_volumes if v.name not in precomputed_layers]
        xml_volumes = [v for v in xml_volumes if v.name not in precomputed_layers]

    # Select gpu
    try:
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
    except IndexError:
        print("No GPU found, using the CPU instead.")

    vol_volumes = set(vol_volumes)
    xml_volumes = set(xml_volumes)
    volumes = vol_volumes + xml_volumes
    # Predict layers and save
    for path in tqdm(volumes):
        # Load data
        if path in vol_volumes:
            data = ep.Oct.from_heyex_vol(path)
        if path in xml_volumes:
            data = ep.Oct.from_heyex_xml(path)

        # Predict layers
        data = get_layers(data, model_id)
        # Save predicted layers
        output_dir = output_path / path.stem
        with open(output_dir / ("layers.pkl"), "wb") as myfile:
            pickle.dump(data.layers, myfile)


def get_layers(data, model_id):
    layer_model, model_config = load_model(model_id, data[0].shape)
    for bscan in tqdm(data, desc=f"Predict {data.data_path.name}: "):
        img = preprocess_standard(bscan.scan, bscan.shape[1])
        prediction = layer_model.predict(img)[0]
        for index, name in model_config["mapping"]:
            bscan.layers[name] = prediction[:, index]

    # if (output_dir / (data.data_path.stem + ".pkl")).is_file():
    #    with open(output_dir / (data.data_path.stem + ".pkl"), "rb") as myfile:
    #        layers = pickle.load(myfile)

    #    for key, val in layers.items():
    #        for i, bscan in tqdm(enumerate(data), desc=f"Load {data.data_path.stem}: "):
    #            heights = val[-(i + 1)]
    #            #heights_clean = np.full_like(heights, np.nan)
    #            #heights_clean[100:-100] = heights[100:-100]
    #            bscan.layers[key] = heights#heights_clean
    return data


def preprocess_standard(data, input_width):
    image = np.zeros((512, input_width))
    image[:496, :input_width] = data
    image = image - np.mean(image)
    image = image / np.std(image)
    image = np.reshape(image, (1, 512, input_width, 1))
    return image
