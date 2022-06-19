import click
import logging

from importlib import resources

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
@click.argument("model_id", type=click.STRING, default="2c41ukad")
@click.pass_context
def layers(ctx: click.Context, model_id, overwrite):
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
            output_dir = output_path / path.relative_to(input_path).parent / path.name
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / ("layers.pkl"), "wb") as myfile:
                pickle.dump(
                    {name: data.data for name, data in data.layers.items()}, myfile
                )

    click.echo("\nPredicted OCT layers are saved. You can now use the 'drusen' command")


def get_layers(data, model_id):
    layer_model, model_config = load_model(model_id, (512, data[0].shape[1], 1))
    results = []
    for bscan in tqdm(data, desc=f"Predict '{data.meta['visit_date']}': "):
        img = preprocess_standard(bscan.data, bscan.shape[1])
        prediction = layer_model.predict(img)[0]
        results.append(prediction)

    results = np.flip(np.stack(results, axis=0), axis=0)
    for index, name in model_config["layer_mapping"].items():
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
