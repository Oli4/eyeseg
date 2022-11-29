import click
import logging

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
    # Delay imports for faster CLI
    import eyepy as ep
    from tqdm import tqdm
    import pandas as pd

    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    volumes = [
        p for p in (input_path / "processed").iterdir() if p.suffix == ".eye"
    ]
    if len(volumes) == 0:
        logger.error(f"No data found in '{input_path}/processed' folder.")
        raise click.Abort

    # Read data
    results = []

    for path in tqdm(volumes):
        # Load data
        data = ep.EyeVolume.load(path)

        vm = data.volume_maps["drusen"]
        vm.radii = radii
        vm.n_sectors = sectors
        vm.offsets = offsets

        quant = vm.quantification
        quant["Visit"] = path.stem
        results.append(vm.quantification)

    # Save quantification results as csv
    if len(results) > 0:
        csv = pd.DataFrame.from_records(results)
        csv = csv.set_index(["Visit", "Laterality"])
        csv = csv.sort_index()
        csv.to_csv(output_path / f"drusen_results.csv")

        click.echo(f"Drusen quantification saved for {len(csv)} volumes.")
