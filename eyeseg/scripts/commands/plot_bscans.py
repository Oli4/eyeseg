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
        areas = []

    for path in tqdm(volumes):
        # Load data
        data = ep.EyeVolume.load(path)
        save_path = output_path / "plots" / "bscans" / path.stem
        save_path.mkdir(parents=True, exist_ok=True)

        rpe_names = [l for l in data.layers if "RPE_" in l]
        bm_names = [l for l in data.layers if "BM_" in l]

        if len(rpe_names) > 1 or len(bm_names) > 1:
            logger.warning(
                f"Multiple RPE or BM layers found. Using the first one. \nRPE Annotations: {rpe_names} \nBM Annotations: {bm_names}"
            )

        drusen = ep.drusen(
            data.layers[rpe_names[0]],
            data.layers[bm_names[0]],
            data.shape,
            minimum_height=5,
        )
        try:
            data.remove_pixel_annotations("drusen")
        except:
            # For the case that eyepy corrects the function name
            data.remove_pixel_annotation("drusen")
        data.add_pixel_annotation(drusen, name="drusen")

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
