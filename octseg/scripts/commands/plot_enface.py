import click
from pathlib import Path
import logging
import eyepy as ep
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from octseg.scripts.utils import find_volumes

logger = logging.getLogger("octseg.plot_enface")


@click.command()
@click.option(
    "--drusen/--no-drusen",
    default=True,
    help="Whether to plot drusen overlay Default is --drusen.",
)
@click.option(
    "--bscan_area/--no-bscan_area",
    default=False,
    help="Whether to plot a rectangle surrounding the B-scan area. Default is --no-bscan_area.",
)
@click.option(
    "--bscan_positions/--no-bscan_positions",
    default=False,
    help="Whether to plot B-scan positions. Default is --no-bscan_positions.",
)
@click.pass_context
def plot_enface(ctx: click.Context, drusen, bscan_area, bscan_positions):
    """Plot drusen enface projections and B-scans

    \f
    :return:
    """
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = find_volumes(input_path)

    data_readers = {"vol": ep.Oct.from_heyex_vol, "xml": ep.Oct.from_heyex_xml}
    # Read data
    for datatype, volumes in volumes.items():
        for path in tqdm(volumes):
            # Load data
            data = data_readers[datatype](path)
            # Load layers and drusen
            output_dir = output_path / path.relative_to(input_path).parent / path.stem
            layers_filepath = output_dir / "layers.pkl"
            drusen_filepath = output_dir / "drusen.pkl"

            try:
                with open(layers_filepath, "rb") as myfile:
                    layer_data = pickle.load(myfile)
            except FileNotFoundError:
                logger.warning(f"No layers.pkl found for {path.stem}")
                continue

            try:
                with open(drusen_filepath, "rb") as myfile:
                    drusen_data = pickle.load(myfile)
            except FileNotFoundError:
                logger.warning(f"No drusen.pkl found for {path.stem}")
                continue

            for key, val in layer_data.items():
                for i, bscan in enumerate(data):
                    heights = val[-(i + 1)]
                    bscan.layers[key] = heights
            data._drusen = drusen_data

            save_path = (
                output_path
                / "plots"
                / "enface"
                / path.relative_to(input_path).parent
                / path.stem
            )
            save_path.mkdir(parents=True, exist_ok=True)

            if not bscan_positions:
                bscan_positions = None
            data.plot(
                drusen=drusen, bscan_region=bscan_area, bscan_positions=bscan_positions
            )
            plt.savefig(save_path / f"{path.stem}.jpeg", bbox_inches="tight", dpi=200)
            plt.close()

    click.echo("\nDrusen enface plots are saved.")
