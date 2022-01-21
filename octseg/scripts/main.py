import click
from pathlib import Path
from octseg.scripts.commands.layers import layers
from octseg.scripts.commands.drusen import drusen
from octseg.scripts.commands.check import check

import logging
import warnings

logger = logging.getLogger("octseg")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setLevel("WARNING")
ch.setFormatter(formatter)
logger.addHandler(ch)


@click.group()
@click.option(
    "--input_path",
    "-i",
    type=click.Path(exists=True),
    default=".",
    help="Path to your input data. Currently only Spectrals XML and VOL exports are supported.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(exists=True),
    help="Location to store the results. The default is processed/ in input_path",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="WARNING",
    help="Set the logging level for the script",
)
@click.pass_context
def main(ctx, input_path, output_path, log_level):
    """
    \f
    :param ctx:
    :param input_path:
    :param output_path:
    :param log_level:
    :return:
    """
    ctx.ensure_object(dict)

    if output_path is None:
        output_path = Path(input_path) / "processed"
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_path / "octseg.log")
    fh.setLevel("DEBUG")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(log_level)

    if log_level != logging.DEBUG:
        warnings.filterwarnings("ignore")

    ctx.obj["input_path"] = Path(input_path)
    ctx.obj["output_path"] = Path(output_path)


main.add_command(drusen)
main.add_command(layers)
main.add_command(check)
