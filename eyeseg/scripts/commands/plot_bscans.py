import click
import sys
import logging
import eyepy as ep
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from eyeseg.scripts.utils import find_volumes

logger = logging.getLogger("eyeseg.plot_bscans")


@click.command()
@click.option(
    "--drusen/--no-drusen",
    default=True,
    help="Whether to plot drusen overlay Default is --drusen.",
)
@click.option(
    "--layers",
    "-l",
    type=click.Choice(["BM", "RPE", "iRPE", "EZ"], case_sensitive=False),
    multiple=True,
    default=[],
    help="Layers predictions to overlay on the B-scan",
)
@click.option(
    "--volumes",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[],
    help="Volumes to plot B-scans for. If not specified B-scans are plotted for all volumes.",
)
@click.pass_context
def plot_bscans(ctx: click.Context, drusen, layers, volumes):
    """Plot B-scans

    \f
    :return:
    """
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    available_volumes = find_volumes(input_path)
    if len(volumes) != 0:
        # Select volume path if any folder in the volumes path is in the folders specified by volumes
        new_volumes = {
            k: [
                v
                for v in volums
                if not set(v.relative_to(input_path).parts).isdisjoint(set(volumes))
            ]
            for k, volums in available_volumes.items()
        }
        click.echo(
            f"You specified {len(volumes)} volumes. {sum([len(v) for v in new_volumes.values()])} volumes were found."
        )

        if sum([len(v) for v in new_volumes.values()]) == 0:
            sys.exit()
        volumes = new_volumes
    else:
        volumes = available_volumes

    data_readers = {"vol": ep.import_heyex_vol, "xml": ep.import_heyex_xml}
    # Read data
    for datatype, volumes in volumes.items():
        for path in tqdm(volumes):
            # Load data
            data = data_readers[datatype](path)
            # Load layers and drusen
            output_dir = output_path / path.relative_to(input_path).parent / path.name
            layers_filepath = output_dir / "layers.pkl"
            drusen_filepath = output_dir / "drusen.pkl"

            try:
                with open(layers_filepath, "rb") as myfile:
                    layer_data = pickle.load(myfile)
            except FileNotFoundError:
                logger.warning(f"No layers.pkl found for {path.name}")
                continue

            try:
                with open(drusen_filepath, "rb") as myfile:
                    drusen_data = pickle.load(myfile)
            except FileNotFoundError:
                logger.warning(f"No drusen.pkl found for {path.stem}")
                continue

            for name, layer in layer_data.items():
                data.add_layer_annotation(layer, name=name)
            data.add_voxel_annotation(drusen_data, name="drusen")

            if "iRPE" in layers:
                irpe = ep.quantification._drusen.ideal_rpe(
                    data.layers["RPE"].data, data.layers["BM"].data, data.shape
                )
                data.add_layer_annotation(irpe, name="iRPE")

            save_path = (
                output_path
                / "plots"
                / "bscans"
                / path.relative_to(input_path).parent
                / path.stem
            )
            save_path.mkdir(parents=True, exist_ok=True)
            for bscan in tqdm(data):
                bscan.plot(areas=["drusen"], layers=layers)
                plt.axis("off")
                plt.savefig(
                    save_path / f"{bscan.index}.jpeg",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=200,
                )
                plt.close()

    click.echo("\nB-scan plots are saved.")
