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
@click.pass_context
def plot_bscans(ctx: click.Context, drusen, layers):
    """Plot B-scans

    \f
    :return:
    """
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = [p for p in (input_path / "processed").iterdir() if p.suffix == ".eye"]
    if len(volumes) == 0:
        click.echo("\nNo volumes found.")
        sys.exit()

    if drusen:
        areas = ["drusen"]
    else:
        ares = []

    for path in tqdm(volumes):
        # Load data
        data = ep.EyeVolume.load(path)
        save_path = output_path / "plots" / "bscans" / path.stem
        save_path.mkdir(parents=True, exist_ok=True)
        for bscan in tqdm(data):
            bscan.plot(areas=areas, layers=layers)
            plt.axis("off")
            plt.savefig(
                save_path / f"{bscan.index}.jpeg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=200,
            )
            plt.close()

    click.echo("\nB-scan plots are saved.")
