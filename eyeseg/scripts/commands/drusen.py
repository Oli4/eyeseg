import click
from pathlib import Path
import logging

import eyepy as ep
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

from eyeseg.scripts.utils import find_volumes

logger = logging.getLogger("eyeseg.drusen")


@click.command()
@click.option(
    "--drusen_threshold",
    "-t",
    type=click.INT,
    default=2,
    help="Minimum height for drusen to be included",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Whether to overwrite existing drusen. Default is --no-overwrite.",
)
@click.pass_context
def drusen(ctx: click.Context, drusen_threshold, overwrite):
    """Compute drusen from BM and RPE layer segmentation

    \f
    :param drusen_threshold:
    :return:
    """
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = find_volumes(input_path)

    # Check for which volumes drusen need to be predicted
    if overwrite is False and output_path.is_dir():
        # Remove path from volumes if layers are found in the output location
        precomputed_drusen = [
            p.name for p in output_path.iterdir() if (p / "drusen.pkl").exists()
        ]
        for datatype in volumes.keys():
            volumes[datatype] = [
                v for v in volumes[datatype] if v.name not in precomputed_drusen
            ]

    data_readers = {"vol": ep.import_heyex_vol, "xml": ep.import_heyex_xml}
    # Read data
    no_layers_volumes = []
    results = []
    for datatype, volumes in volumes.items():
        for path in tqdm(volumes):
            # Load data
            data = data_readers[datatype](path)
            # Read layers
            output_dir = output_path / path.relative_to(input_path).parent / path.name
            layers_filepath = output_dir / "layers.pkl"
            try:
                with open(layers_filepath, "rb") as myfile:
                    layers = pickle.load(myfile)
            except FileNotFoundError:
                logger.warning(f"No layers.pkl found for {path.name}")
                no_layers_volumes.append(path)
                continue

            for name, layer in layers.items():
                data.add_layer_annotation(layer, name=name)

            # Compute drusen
            drusen = ep.drusen(
                data.layers["RPE"],
                data.layers["BM"],
                data.shape,
                minimum_height=drusen_threshold,
            )
            output_dir = output_path / path.relative_to(input_path).parent / path.name
            drusen_filepath = output_dir / "drusen.pkl"
            with open(drusen_filepath, "wb") as myfile:
                pickle.dump(drusen, myfile)

    if len(no_layers_volumes) > 0:
        click.echo(
            f"No retinal layers found for {len(no_layers_volumes)} volumes. To predict layers run the 'layers' command."
        )
    else:
        click.echo(
            "\nComputed drusen are saved. You can now use the 'quantify', 'plot-enface' and 'plot-bscans' commands"
        )