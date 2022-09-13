import click
from pathlib import Path
import logging
import eyepy as ep
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from eyeseg.scripts.utils import find_volumes

logger = logging.getLogger("eyeseg.plot_enface")


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

    volumes = [p for p in (input_path / "processed").iterdir() if p.suffix == ".eye"]

    for path in tqdm(volumes):
        # Load data
        data = ep.EyeVolume.load(path)

        save_path = output_path / "plots" / "enface"
        save_path.mkdir(parents=True, exist_ok=True)

        if not bscan_positions:
            bscan_positions = None
        data.plot(
            projections=["drusen"],
            bscan_region=bscan_area,
            bscan_positions=bscan_positions,
        )
        plt.savefig(save_path / f"{path.stem}.jpeg", bbox_inches="tight", dpi=200)
        plt.close()

    click.echo("\nDrusen enface plots are saved.")
