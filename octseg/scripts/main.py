import click
from pathlib import Path
from octseg.scripts.layers import layers
from octseg.scripts.drusen import drusen

import logging


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
    default="INFO",
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

    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    if output_path is None:
        output_path = Path(input_path) / "processed"
    else:
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    ctx.obj["input_path"] = input_path
    ctx.obj["output_path"] = output_path


main.add_command(drusen)
main.add_command(layers)
