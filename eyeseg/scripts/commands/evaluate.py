import click
import logging


logger = logging.getLogger("eyeseg.evaluate")


@click.command()
@click.option(
    "-r",
    "--run-path",
    type=click.Path(exists=True),
    help="Path to folder with model configuration and network weights. (model_config.yaml; model-best.h5)",
)
@click.option(
    "-s",
    "--input-shape",
    nargs=2,
    type=int,
    help="Shape of the data.",
)
@click.option("-b", "--batch_size", type=int, help="Batch size used during training")
@click.option(
    "-p", "--plot", type=bool, default=True, help="Plot test dataset with predictions"
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    run_path,
    input_shape,
    batch_size,
    plot,
):
    """Evaluate a model on a test dataset"""
    # Delay imports for faster CLI
    import yaml
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from eyeseg.models.feature_refinement_net_he import model
    from eyeseg.io_utils.losses import MovingMeanFocalSSE
    from eyeseg.io_utils.input_pipe import get_split
    from eyeseg.io_utils.utils import get_metrics

    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    # weights_file = wandb.restore('model-best.h5', run_path=run_path)
    # model_config = wandb.restore('model_config.yaml', run_path=run_path)

    ######## Temporary
    weights_file = str(output_path / "model-best.h5")
    model_config = str(output_path / "model_config.yaml")

    # load config file if provided
    with open(model_config, "r") as myfile:
        config = yaml.full_load(myfile)

    ########

    if input_shape is None:
        input_shape = config["training"]["input_shape"]
    if batch_size is None:
        batch_size = config["training"]["batch_size"]

    # Find volumes
    test_data = get_split(
        input_path,
        config["layer_mapping"],
        input_shape,
        batch_size=1,
        epochs=1,
        split="test",
    )
    metrics = get_metrics(config["layer_mapping"])

    my_model = model(
        input_shape=input_shape + (1,),
        num_classes=len(config["layer_mapping"]),
        **config["parameters"],
    )
    my_model.load_weights(weights_file)

    loss_fn = MovingMeanFocalSSE(
        window_size=config["training"]["boosting_window_size"],
        curv_weight=config["training"]["curv_weight"],
    )
    my_model.compile(
        loss={"layer_output": loss_fn},
        metrics=metrics,
        sample_weight_mode="temporal",
    )

    (output_path / "results").mkdir(parents=True, exist_ok=True)

    results = {}
    for image, data in test_data:
        prediction = my_model.predict(image)
        layerout = data["layer_output"]
        volume, bscan, group = (
            bytes.decode(data["Volume"].numpy()[0]),
            bytes.decode(data["Bscan"].numpy()[0]),
            bytes.decode(data["Group"].numpy()[0]),
        )

        results[(volume, bscan, group)] = np.abs(prediction - layerout)

        if plot:
            plt.imshow(image[0, ..., 0], cmap="gray")
            for layer in range(9):
                plt.plot(image.shape[1] - prediction[0, ..., layer])

            plt.savefig(
                output_path
                / "results"
                / f"{np.mean(results[(volume, bscan, group)]):.2f}_{volume}_{bscan}.jpeg",
                bbox_inches="tight",
            )
            plt.close()

    with open(output_path / "results" / "test_results.pkl", "wb") as myfile:
        pickle.dump(results, myfile)

    # results = my_model.evaluate(test_data, batch_size=batch_size, return_dict=True)
    # print(results)
