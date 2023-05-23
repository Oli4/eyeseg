import click
import logging

logger = logging.getLogger("eyeseg.check")
logger.setLevel("INFO")


@click.command()
@click.pass_context
def check(
    ctx: click.Context,
):
    """Check XML exports for common problems"""
    from eyeseg.scripts.utils import find_volumes

    input_path = ctx.obj["input_path"]

    volumes = find_volumes(input_path)["xml"]

    error_count_multiple = check_multiple_exports(volumes)
    error_count_white = check_white_background(volumes)

    if error_count_white + error_count_multiple == 0:
        msg = "All XML exports are checked and 0 problems were found."
        logger.info(msg)
    else:
        msg = f"Please fix {error_count_white+error_count_multiple} errors before processing your data."
        logger.info(msg)

    click.echo(msg)


def check_multiple_exports(v_paths):
    # Delayed import for faster CLI
    from tqdm import tqdm
    errors = 0
    for p in tqdm(v_paths, desc="Check for double exports: "):
        if len(list(p.glob("*.xml"))) != 1:
            logger.warning(f"{p.name}: Folder contains more than 1 XML export.")
            errors += 1
    return errors


def check_white_background(v_paths):
    # Delayed import for faster CLI
    from tqdm import tqdm
    import numpy as np
    import eyepy as ep
    errors = 0
    for p in tqdm(v_paths, desc="Check for inverted contrast: "):
        data = ep.import_heyex_xml(p)
        if np.mean(data[0].data) > 128:
            logger.warning(f"{p.name}: B-scans have inverted contrast.")
            errors += 1
    return errors
