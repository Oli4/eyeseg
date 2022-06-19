import click
from pathlib import Path
import logging

import eyepy as ep
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

from eyeseg.scripts.utils import find_volumes

# from eyeseg.grids import grid

logger = logging.getLogger("eyeseg.quantify")


@click.command()
@click.option(
    "--radii",
    "-r",
    type=click.FLOAT,
    multiple=True,
    default=[0.8, 1.8],
    help="Radii for quantification grid in mm",
)
@click.option(
    "--sectors",
    "-s",
    type=click.INT,
    multiple=True,
    default=[1, 4],
    help="Number of Sectors corresponding to radii",
)
@click.option(
    "--offsets",
    "-o",
    type=click.FLOAT,
    multiple=True,
    default=[0.0, 45.0],
    help="Angular offset from the horizontal line for sectors in degree.",
)
@click.pass_context
def quantify(ctx: click.Context, radii, sectors, offsets):
    """Quantify drusen on a sectorized circular grid

    \f
    :param radii:
    :param sectors:
    :param offsets:
    :return:
    """
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = find_volumes(input_path)

    data_readers = {"vol": ep.import_heyex_vol, "xml": ep.import_heyex_xml}
    # Read data
    no_drusen_volumes = []
    results = []
    for datatype, volumes in volumes.items():
        for path in tqdm(volumes):
            # Load data
            data = data_readers[datatype](path)
            # Read layers
            output_dir = output_path / path.relative_to(input_path).parent / path.name
            layers_filepath = output_dir / "layers.pkl"
            drusen_filepath = output_dir / "drusen.pkl"

            try:
                with open(drusen_filepath, "rb") as myfile:
                    drusen = pickle.load(myfile)
            except FileNotFoundError:
                no_drusen_volumes.append(path)
                continue

            data.add_voxel_annotation(
                drusen, name="drusen", radii=radii, n_sectors=sectors, offsets=offsets
            )
            results.append(data.volume_maps["drusen"].quantification)

    # Save quantification results as csv
    if len(results) > 0:
        csv = pd.DataFrame.from_records(results)
        csv = csv.set_index(["Visit", "Laterality"])
        csv = csv.sort_index()
        csv.to_csv(output_path / f"drusen_results.csv")

        click.echo(f"Drusen quantification saved for {len(csv)} volumes.")

    if len(no_drusen_volumes) > 0:
        click.echo(
            f"No drusen found for {len(no_drusen_volumes)} volumes. To compute drusen run the 'drusen' command after having predicted layers with the 'layers' command."
        )
