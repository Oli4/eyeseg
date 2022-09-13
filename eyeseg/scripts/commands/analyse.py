import click
import logging

from importlib import resources

import skimage.transform

from eyeseg.models import weights as weights_resources
from eyeseg.models.utils import load_model
from eyeseg.scripts.utils import find_volumes

from tqdm import tqdm
import numpy as np
import eyepy as ep
import pickle

logger = logging.getLogger("eyeseg.layers")


@click.command()
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Whether to overwrite existing layers. Default is --no-overwrite.",
)
@click.option(
    "--drusen_threshold",
    "-t",
    type=click.INT,
    default=2,
    help="Minimum height for drusen to be included",
)
@click.argument("model_id", type=click.STRING, default="2c41ukad")
@click.pass_context
def analyse(ctx: click.Context, model_id, overwrite, drusen_threshold):
    """Predict OCT layers

    \b
    MODEL_ID: Specifies the model
        Pretrained models:
            Spectralis:
                2c41ukad: 3 classes (BM, RPE, EZ) - (Default)
    \f
    """
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    # Check if specified model is available
    if not model_id in list(resources.contents(weights_resources)):
        msg = f"A model with ID {model_id} is not available. Check 'eyeseg layers --help' for available models."
        logger.error(msg)
        raise ValueError(msg)

    # Find volumes
    volumes = find_volumes(input_path)

    # Check for which volumes layers need to be predicted
    if overwrite is False and output_path.is_dir():
        # Remove path from volumes if layers are found in the output location
        precomputed_layers = [
            p.name for p in output_path.iterdir() if (p / "layers.pkl").exists()
        ]
        for datatype in volumes.keys():
            volumes[datatype] = [
                v for v in volumes[datatype] if v.name not in precomputed_layers
            ]

    data_readers = {"vol": ep.import_heyex_vol, "xml": ep.import_heyex_xml}
    # Predict layers and save
    for datatype, volumes in volumes.items():
        for path in tqdm(
            volumes,
            desc="Volumes: ",
        ):
            # Load data
            data = data_readers[datatype](path)

            # Predict layers
            data = get_layers(data, model_id)
            # Save predicted layers
            output_dir = output_path / path.relative_to(input_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            if "RPE" in data.layers and "BM" in data.layers:
                drusen = ep.drusen(
                    data.layers["RPE"],
                    data.layers["BM"],
                    data.shape,
                    minimum_height=drusen_threshold,
                )
                data.add_voxel_annotation(drusen, name="drusen")

            data.save(output_dir / (path.name + ".eye"))

    click.echo(
        "\nComputed layers and drusen are saved. You can now use the 'quantify', 'plot-enface' and 'plot-bscans' commands"
    )


def get_layers(data, model_id):
    if data.meta["scale_x"] < 0.009:
        factor = 2
    else:
        factor = 1

    width = data[0].shape[1]
    layer_model, model_config = load_model(model_id, (512, width // factor, 1))
    results = []
    for bscan in tqdm(data, desc=f"Predict '{data.meta['visit_date']}': "):
        img = skimage.transform.rescale(bscan.data, (1, 1 / factor))
        img = preprocess_standard(img, width // factor)
        prediction = layer_model.predict(img)[0]
        results.append(prediction)

    results = np.flip(np.stack(results, axis=0), axis=0)
    for index, name in model_config["layer_mapping"].items():
        if factor != 1:
            height_map = skimage.transform.rescale(results[..., index], (1, factor))
            # height_map = np.interp(np.arange(width), np.arange(width//factor) * factor, results[..., index])
        else:
            height_map = results[..., index]
        data.add_layer_annotation(height_map, name=name)
    return data


def preprocess_standard(data, input_width):
    image = np.zeros((512, input_width))
    image[:496, :input_width] = data
    image = image - np.mean(image)
    image = image / np.std(image)
    image = np.reshape(image, (1, 512, input_width, 1))
    return image
