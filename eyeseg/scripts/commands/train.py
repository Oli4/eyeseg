import os
import click
import logging
from pathlib import Path
import yaml
import wandb
from wandb.keras import WandbCallback

from eyeseg.models.feature_refinement_net import model
from eyeseg.io_utils.input_pipe import (
    get_augment_function,
    get_parse_function,
    get_transform_func_combined,
    _normalize,
    _prepare_train,
)
from eyeseg.io_utils.losses import MovingMeanFocalSSE
from eyeseg.io_utils.metrics import get_mae, get_curv, get_layer_curv, get_layer_mae

import tensorflow as tf

logger = logging.getLogger("eyeseg.train")


@click.command()
@click.option(
    "-c",
    "--model-config",
    default=None,
    help="Path to to model configuration as yaml file. If not provided a new file is generated from the provided arguments.",
)
@click.option(
    "-s",
    "--input-shape",
    nargs=2,
    type=int,
    help="Shape of the data.",
)
@click.option(
    "-f",
    "--filters",
    type=int,
    help="Number of filters",
)
@click.option(
    "-a",
    "--activation",
    type=click.STRING,
    help="Activation function used by the model",
)
@click.option("-d", "--dropout", type=float, help="Dropout ratio used by the model")
@click.option(
    "-sp",
    "--spatial-dropout",
    type=float,
    help="Spatial dropout ratio used by the model",
)
@click.option(
    "-l",
    "--layer-mapping",
    type=(int, str),
    multiple=True,
    help="Layer Mapping, index, name pair for proper naming of metrics",
)
@click.option(
    "-b", "--batch_size", type=int, default=2, help="Batch size used during training"
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs for the model to train",
)
@click.option("-lr", "--lr", type=float, help="Learning rate")
@click.option("-ld", "--lr-decay", type=float, help="Exponential learning rate decay")
@click.option(
    "-vf", "--validation-frequency", type=int, help="Exponential learning rate decay"
)
@click.option(
    "--train-size",
    type=int,
    help="Number of training samples in the dataset. If not provided, samples are counted",
)
@click.option(
    "--val-size",
    type=int,
    help="Number of validation samples in the dataset. If not provided, samples are counted",
)
@click.option(
    "-bws",
    "--boosting-window-size",
    type=int,
    help="Size of running window used for boosting. Boosting focuses the learning on errors greater than the exponential moving MAE",
)
@click.option(
    "-cw",
    "--curv-weight",
    type=float,
    help="Loss weighting for an additional loss term computed from the second derivative of the layer heights.",
)
@click.pass_context
def train(
    ctx: click.Context,
    model_config,
    input_shape,
    filters,
    activation,
    dropout,
    spatial_dropout,
    layer_mapping,
    batch_size,
    epochs,
    lr,
    lr_decay,
    validation_frequency,
    train_size,
    val_size,
    boosting_window_size,
    curv_weight,
):
    """Train a new layer segmentation model"""
    input_path = ctx.obj["input_path"]
    output_path = ctx.obj["output_path"]

    # load config file if provided
    if model_config and Path(model_config).is_file():
        with open(model_config, "r") as stream:
            config = yaml.full_load(stream)
    else:
        config = {
            "device": "spectralis",
            "module": "eyeseg.models.feature_refinement_net",
            "layer_mapping": {},
            "parameters": {
                "filters": 64,
                "activation": "swish",
                "dropout": 0.1,
                "spatial_dropout": 0.1,
            },
            "training": {
                "lr": 0.001,
                "lr_decay": -0.1,
                "validation_frequency": 1,
                "boosting_window_size": 0,
                "curv_weight": 0,
                "batch_size": 2,
                "epochs": 40,
            },
        }

    # change config file with additional parameters
    if filters:
        config["parameters"]["filters"] = filters
    if activation:
        config["parameters"]["activation"] = activation
    if dropout:
        config["parameters"]["dropout"] = dropout
    if spatial_dropout:
        config["parameters"]["spatial_dropout"] = spatial_dropout
    if layer_mapping:
        config["layer_mapping"] = {x[0]: x[1] for x in layer_mapping}
    elif len(config["layer_mapping"]) == 0:
        raise click.UsageError(
            "You need to specify the layer_mapping either in a config file or using -l option. "
            "The layer_mapping maps the layer indices in the data to their name."
        )
    if lr:
        config["training"]["lr"] = lr
    if lr_decay:
        config["training"]["lr_decay"] = lr_decay
    if validation_frequency:
        config["training"]["validation_frequency"] = validation_frequency
    if boosting_window_size:
        config["training"]["boosting_window_size"] = boosting_window_size
    if curv_weight:
        config["training"]["curv_weight"] = curv_weight
    if input_shape:
        config["training"]["input_shape"] = input_shape
    elif "input_shape" not in config["training"]:
        raise click.UsageError("The input_shape needs to be provided.")

    if train_size:
        config["training"]["train_size"] = train_size
    elif "train_size" in config["training"]:
        pass
    else:
        config["training"]["train_size"] = count_samples(
            input_path, "train", config["layer_mapping"], input_shape
        )

    if val_size:
        config["training"]["val_size"] = val_size
    elif "val_size" in config["training"]:
        pass
    else:
        config["training"]["val_size"] = count_samples(
            input_path, "val", config["layer_mapping"], input_shape
        )

    if batch_size:
        config["training"]["batch_size"] = batch_size
    if epochs:
        config["training"]["epochs"] = epochs

    # save config file
    with open(output_path / "config.yaml", "w") as outfile:
        yaml.dump(config, outfile)

    # Find volumes
    train_data = get_split(
        input_path,
        config["layer_mapping"],
        config["training"]["input_shape"],
        config["training"]["batch_size"],
        config["training"]["epochs"],
        "train",
    )
    val_data = get_split(
        input_path,
        config["layer_mapping"],
        config["training"]["input_shape"],
        config["training"]["batch_size"],
        config["training"]["epochs"],
        "val",
    )

    metrics = get_metrics(config["layer_mapping"])

    my_model = model(
        input_shape=config["training"]["input_shape"] + (1,),
        num_classes=len(config["layer_mapping"]),
        **config["parameters"],
    )

    os.environ["WANDB_API_KEY"] = config["wandb"]["apikey"]
    os.environ["WANDB_DIR"] = str(output_path)
    wandb_config = dict(
        input_shape=config["training"]["input_shape"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        num_classes=len(config["layer_mapping"]),
        filters=config["parameters"]["filters"],
        boosting_window_size=config["training"]["boosting_window_size"],
        learning_rate=config["training"]["lr"],
        lr_decay=config["training"]["lr_decay"],
        curv_weight=config["training"]["curv_weight"],
        order_guarantee=True,
        activation=config["parameters"]["activation"],
        dropout=config["parameters"]["dropout"],
        sp_dropout=config["parameters"]["spatial_dropout"],
        model_parameters=my_model.count_params(),
        model="feature_refinement_net",
    )

    run = wandb.init(
        project=config["wandb"]["project_name"], config=wandb_config, reinit=False
    )

    def scheduler(epoch, lr):
        return lr * tf.math.exp(config["training"]["lr_decay"])

    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    callbacks.append(WandbCallback())

    def internal_metric(f, p, name):
        def get_metric(*args, **kwargs):
            return getattr(f, p)

        get_metric.__name__ = name
        return get_metric

    loss_fn = MovingMeanFocalSSE(
        window_size=config["training"]["boosting_window_size"],
        curv_weight=config["training"]["curv_weight"],
    )
    metrics["layer_output"].append(internal_metric(loss_fn, "ema", name="EMA"))

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["lr"], clipnorm=1.0, clipvalue=0.5
        ),
        loss={"layer_output": loss_fn},
        metrics=metrics,
        sample_weight_mode="temporal",
    )

    my_model.fit(
        train_data,
        # Epochs based on the Drusen Dataset
        steps_per_epoch=int(
            config["training"]["train_size"] / config["training"]["batch_size"]
        ),  # len(train_filepaths) // BATCH_SIZE,
        epochs=config["training"]["epochs"],
        validation_data=val_data,
        validation_freq=config["training"]["validation_frequency"],
        validation_steps=int(
            config["training"]["val_size"] / config["training"]["val_size"]
        ),  # len(val_filepaths) // BATCH_SIZE,
        callbacks=callbacks,
    )


def get_split(path, layer_mapping, input_shape, batch_size, epochs, split, seed=42):
    _transform = get_transform_func_combined(
        translation=[-90, 90],
        scale=[0.95, 1.05],
        rotation=(-0, 0),
        num_classes=len(layer_mapping),
    )
    _augment = get_augment_function(
        contrast_range=(0.9, 1.1),
        brightnesdelta=0.1,
        augment_probability=1,
        occlusion=False,
    )
    _parse_image_function = get_parse_function(layer_mapping, input_shape)

    if split not in ["train", "val", "test"]:
        raise ValueError("The 'split' parameters has to be one of train, test or val.")

    paths = [str(p) for p in path.glob(f"{split}*")]
    if paths is []:
        raise click.UsageError(
            f"There are no files starting with {split} in the given folder"
        )
    raw_data = tf.data.TFRecordDataset(paths, num_parallel_reads=10)
    parsed_data = raw_data.map(_parse_image_function)

    if split == "train":
        dataset = (
            parsed_data.shuffle(
                14000, seed, reshuffle_each_iteration=True
            )  # .map(_augment)
            .map(_transform)
            .map(_normalize)
            .batch(batch_size)
            .map(_prepare_train)
            .repeat(epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        dataset = (
            parsed_data.map(_normalize)
            .batch(batch_size)
            .map(_prepare_train)
            .repeat(epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return dataset


def count_samples(path, split, layer_mapping, input_shape):
    _parse_image_function = get_parse_function(
        mapping=layer_mapping, input_shape=input_shape
    )

    if split not in ["train", "val", "test"]:
        raise ValueError("The 'split' parameters has to be one of train, test or val.")

    raw_data = tf.data.TFRecordDataset(
        [str(p) for p in path.glob(f"{split}*")], num_parallel_reads=10
    )
    parsed_data = raw_data.map(_parse_image_function).batch(1)

    count = sum([1 for _ in parsed_data])
    return count


def get_metrics(layer_mapping):
    mae = [
        get_mae(),
    ] + [get_layer_mae(i, name) for i, name in layer_mapping.items()]
    curv_mae = [
        get_curv(),
    ] + [get_layer_curv(i, name) for i, name in layer_mapping.items()]

    metrics = {"layer_output": mae + curv_mae}
    return metrics
