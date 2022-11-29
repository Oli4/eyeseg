import click
import logging

logger = logging.getLogger("eyeseg.plot_enface")


@click.command()
@click.option(
    "--drusen/--no-drusen",
    default=True,
    help="Whether to plot drusen overlay Default is --drusen.",
)
@click.option(
    "--bscan-area/--no-bscan-area",
    default=False,
    help=
    "Whether to plot a rectangle surrounding the B-scan area. Default is --no-bscan_area.",
)
@click.option(
    "--bscan-position",
    "-p",
    default=[],
    type=click.IntRange(0),
    multiple=True,
    help=
    "Index of B-scan to mark on the enface plot. Multiple values can be passed.",
)
@click.pass_context
def plot_enface(ctx: click.Context, drusen, bscan_area, bscan_position):
    """Plot drusen enface projections and B-scans

    \f
    :return:
    """
    # Delay imports for faster CLI
    import eyepy as ep
    from eyeseg.scripts.utils import find_volumes
    from pathlib import Path
    from tqdm import tqdm
    import pickle
    import matplotlib.pyplot as plt

    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = [
        p for p in (input_path / "processed").iterdir() if p.suffix == ".eye"
    ]
    if len(volumes) == 0:
        logger.error(f"No data found in '{input_path}/processed' folder.")
        raise click.Abort

    for path in tqdm(volumes):
        # Load data
        data = ep.EyeVolume.load(path)

        save_path = output_path / "plots" / "enface"
        save_path.mkdir(parents=True, exist_ok=True)

        #if not bscan_positions:
        #    bscan_positions = None
        data.plot(
            projections=["drusen"],
            bscan_region=bscan_area,
            bscan_positions=bscan_position,
        )
        plt.savefig(save_path / f"{path.stem}.jpeg",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()

    click.echo("\nDrusen enface plots are saved.")
