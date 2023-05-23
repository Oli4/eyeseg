import click
import logging

import os

from eyeseg.scripts.commands.check import check
from eyeseg.scripts.commands.segment import segment
from eyeseg.scripts.commands.quantify import quantify
from eyeseg.scripts.commands.plot_enface import plot_enface
from eyeseg.scripts.commands.plot_bscans import plot_bscans
#from eyeseg.scripts.commands.train import train
#from eyeseg.scripts.commands.evaluate import evaluate
#from eyeseg.scripts.commands.plot_data import plot_data

logger = logging.getLogger("eyeseg")

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    help=
    "Path to your input data. Currently only Spectrals XML and VOL exports are supported.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(exists=True),
    help=
    "Location to store the results. The default is processed/ in input_path",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="WARNING",
    help="Set the logging level for the script",
)
@click.option(
    "--gpu",
    type=int,
    default=0,
    help="Number of the GPU if more than one is available. Default is 0.",
)
@click.pass_context
def main(ctx, input_path, output_path, log_level, gpu):
    """
    \f
    :param ctx:
    :param input_path:
    :param output_path:
    :param log_level:
    :return:
    """
    # Delay imports for fast CLI

    from pathlib import Path
    log_int = {
        "DEBUG": "0",
        "INFO": "1",
        "WARNING": "2",
        "ERROR": "3",
    }

    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = log_int[log_level]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_int[log_level]
    from tensorflow import get_logger as get_tf_logger
    from tensorflow import config as tf_config

    get_tf_logger().setLevel(log_level)

    ctx.ensure_object(dict)

    if input_path is None:
        raise click.UsageError(
            "You need to provide the input_path to the eyeseg command")

    if output_path is None:
        output_path = Path(input_path) / "processed"
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(output_path)

    fh = logging.FileHandler(output_path / "eyeseg.log")
    fh.setLevel("DEBUG")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(log_level)

    if log_level != logging.DEBUG:
        import warnings
        warnings.filterwarnings("ignore")

    ctx.obj["input_path"] = Path(input_path)
    ctx.obj["output_path"] = Path(output_path)

    # Select gpu
    try:
        gpus = tf_config.list_physical_devices("GPU")
        tf_config.experimental.set_visible_devices(gpus[gpu], "GPU")
    except IndexError:
        msg = "No GPU found, using the CPU instead."
        logger.warning(msg)


main.add_command(check)
main.add_command(segment)
main.add_command(quantify)
main.add_command(plot_enface)
main.add_command(plot_bscans)

#main.add_command(train)
#main.add_command(evaluate)
#main.add_command(plot_data)
