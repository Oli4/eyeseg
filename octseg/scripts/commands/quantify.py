import click
from pathlib import Path
import logging

from eyepy.core.drusen import drusen2d, filter_by_height
import eyepy as ep
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

from octseg.scripts.utils import find_volumes
from octseg.grids import grid

logger = logging.getLogger("octseg.quantify")


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

    data_readers = {"vol": ep.Oct.from_heyex_vol, "xml": ep.Oct.from_heyex_xml}
    # Read data
    no_drusen_volumes = []
    results = []
    for datatype, volumes in volumes.items():
        for path in tqdm(volumes):
            # Load data
            data = data_readers[datatype](path)
            # Read layers
            layers_filepath = output_path / path.stem / "layers.pkl"
            drusen_filepath = output_path / path.stem / "drusen.pkl"

            try:
                with open(layers_filepath, "rb") as myfile:
                    layers = pickle.load(myfile)
                with open(drusen_filepath, "rb") as myfile:
                    drusen = pickle.load(myfile)
            except FileNotFoundError:
                no_drusen_volumes.append(path)
                continue

            for key, val in layers.items():
                for i, bscan in enumerate(data):
                    heights = val[-(i + 1)]
                    bscan.layers[key] = heights
            data._drusen = drusen

            results.append(quantify_drusen(data, radii, sectors, offsets))

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


def quantify_drusen(oct_obj, radii, n_sectors, offsets):
    # Quantify drusen
    if not oct_obj.Distance:
        distances = []
        for i, b in enumerate(oct_obj[1:]):
            i = i + 1
            distances.append(oct_obj[i - 1].StartY - oct_obj[i].StartY)
        oct_obj.meta["Distance"] = np.mean(distances)

    masks = grid(
        mask_shape=(oct_obj.SizeXSlo, oct_obj.SizeYSlo),
        radii=radii,
        laterality=oct_obj.ScanPosition,
        n_sectors=n_sectors,
        offsets=offsets,
        radii_scale=oct_obj.ScaleXSlo,
    )

    enface_voxel_size_µm3 = (
        oct_obj.ScaleXSlo * 1e3 * oct_obj.ScaleYSlo * 1e3 * oct_obj.ScaleZ * 1e3
    )
    oct_voxel_size_µm3 = (
        oct_obj.ScaleX * 1e3 * oct_obj.Distance * 1e3 * oct_obj.ScaleZ * 1e3
    )

    drusen_enface = oct_obj.drusen_enface

    results = {}
    for name, mask in masks.items():
        results[f"{name} [mm³]"] = (
            (drusen_enface * mask).sum() * enface_voxel_size_µm3 / 1e9
        )

    results["Total [mm³]"] = drusen_enface.sum() * enface_voxel_size_µm3 / 1e9
    results["Total [OCT voxels]"] = oct_obj.drusen_projection.sum()
    results["OCT Voxel Size [µm³]"] = oct_voxel_size_µm3

    results["VisitDate"] = str(oct_obj.VisitDate)
    results["Laterality"] = oct_obj.ScanPosition
    results["Visit"] = oct_obj.data_path.parent.name
    return results