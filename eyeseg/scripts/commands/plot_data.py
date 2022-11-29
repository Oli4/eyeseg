import click
import logging

logger = logging.getLogger("eyeseg.evaluate")


@click.command()
@click.option(
    "-c",
    "--model-config",
    default="./config.yaml",
    type=click.Path(exists=True),
    help="Path to to model configuration as yaml file. If not provided a new file is generated from the provided arguments.",
)
@click.option(
    "-n",
    "--number-of-examples",
    type=int,
    default=20,
    help="Number of examples to plot",
)
@click.pass_context
def plot_data(ctx: click.Context, model_config, number_of_examples):
    """Visualize dataset after transformation/augmentation"""
    # Delay imports for faster CLI
    import yaml
    from pathlib import Path
    import matplotlib.pyplot as plt
    from eyeseg.io_utils.input_pipe import get_split

    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    # load config file if provided
    if model_config and Path(model_config).is_file():
        with open(model_config, "r") as stream:
            config = yaml.full_load(stream)

    # Find volumes
    data = get_split(
        input_path,
        config["layer_mapping"],
        config["training"]["input_shape"],
        1,
        1,
        "train",
    )
    data = iter(data)

    path = output_path / "examples"
    path.mkdir(parents=True, exist_ok=True)
    for i in range(number_of_examples):
        sample = next(data)
        image, layerout = sample

        plt.imshow(image[0, ..., 0], cmap="gray")

        for layer in range(9):
            plt.plot(image.shape[1] - layerout["layer_output"][0, ..., layer])

        plt.savefig(path / f"sample_{i}")
        plt.close()
