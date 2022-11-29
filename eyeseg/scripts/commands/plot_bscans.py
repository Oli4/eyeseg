import click
import logging

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
    # Delay imports for faster CLI
    import eyepy as ep
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = [
        p for p in (input_path / "processed").iterdir() if p.suffix == ".eye"
    ]
    if len(volumes) == 0:
        logger.error(f"No data found in '{input_path}/processed' folder.")
        raise click.Abort

    if drusen:
        areas = ["drusen"]
    else:
        ares = []

    for path in tqdm(volumes):
        # Load data
        data = ep.EyeVolume.load(path)
        save_path = output_path / "plots" / "bscans" / path.stem
        save_path.mkdir(parents=True, exist_ok=True)

        #for key in data.layers:
        #heights = data.layers[key].data
        #heights_clean = np.full_like(heights, np.nan)
        #heights_clean[:, 100:-100] = heights[:, 100:-100]
        #data.layers[key].data = heights_clean

        drusen = ep.drusen(
            data.layers["RPE"],
            data.layers["BM"],
            data.shape,
            minimum_height=5,
        )
        data.delete_voxel_annotations("drusen")
        data.add_voxel_annotation(drusen, name="drusen")

        for bscan in tqdm(data):
            bscan.plot(areas=areas, layers=layers)
            plt.axis("off")
            plt.savefig(
                save_path / f"{bscan.index}.jpeg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            plt.close()

    click.echo("\nB-scan plots are saved.")
