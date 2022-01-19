import click
from pathlib import Path
from eyepy.core.drusen import drusen2d, filter_by_height
import eyepy as ep
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

from octseg.scripts.utils import find_volumes
from octseg.grids import grid


@click.command()
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
@click.option(
    "--drusen_threshold",
    "-t",
    type=click.INT,
    default=2,
    help="Minimum height for drusen to be included",
)
@click.argument("data_path", type=click.Path(exists=True), default="/home/data")
def drusen(
    data_path, output_path, overwrite, drusen_threshold, radii, sectors, offsets
):
    """Compute drusen from BM and RPE layer segmentation

    :param data_path:
    :param output_path:
    :param overwrite:
    :param drusen_threshold:
    :param radii:
    :param sectors:
    :param offsets:
    :return:
    """
    data_path = Path(data_path)
    if output_path is None:
        output_path = Path(data_path) / "processed"
    else:
        output_path = Path(output_path)

    all_volumes = find_volumes(data_path)
    data_readers = {"vol": ep.Oct.from_heyex_vol, "xml": ep.Oct.from_heyex_xml}
    # Read data
    no_layers_volumes = []
    results = []
    for datatype, volumes in all_volumes.items():
        for path in tqdm(volumes):
            # Load data
            data = data_readers[datatype](path)
            # Read layers
            layers_filepath = output_path / path.stem / "layers.pkl"
            if layers_filepath.is_file():
                with open(layers_filepath, "rb") as myfile:
                    layers = pickle.load(myfile)
            else:
                no_layers_volumes.append(path)
                continue

            for key, val in layers.items():
                for i, bscan in enumerate(data):
                    heights = val[-(i + 1)]
                    bscan.layers[key] = heights

            # Compute drusen
            drusen = drusen2d(data.layers["RPE"], data.layers["BM"], data.shape)
            data._drusen = filter_by_height(drusen, minimum_height=drusen_threshold)

            results.append(quantify_drusen(data, radii, sectors, offsets))

    # Save quantification results as csv
    csv = pd.DataFrame.from_records(results)
    csv = csv.set_index(["Visit", "Laterality"])
    csv = csv.sort_index()
    csv.to_csv(output_path / f"results.csv")


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
