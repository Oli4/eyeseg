import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

import numpy as np

import click


def get_parse_function(mapping, input_shape, **extra_features):
    @tf.function
    def _parse_function(input_proto):
        # Create a dictionary describing the features.
        layer_features = {
            val.lower(): tf.io.FixedLenFeature([], tf.string)
            for val in mapping.values()
        }
        other_features = {
            "volume": tf.io.FixedLenFeature([], tf.string),
            "bscan": tf.io.FixedLenFeature([], tf.string),
            # "layer_positions": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "group": tf.io.FixedLenFeature([], tf.string),
        }

        image_feature_description = {
            **layer_features,
            **other_features,
            **extra_features,
        }

        # Parse the input_proto using the dictionary above.
        data = tf.io.parse_single_example(input_proto, image_feature_description)

        image = tf.io.parse_tensor(data["image"], tf.uint8)
        image = tf.reshape(image, input_shape + (1,), name="reshape_1")

        # layer_positions = tf.io.parse_tensor(data["layer_positions"], tf.float32)
        # layer_positions = tf.reshape(
        #    layer_positions, input_shape + (len(mapping),), name="reshape_1.5"
        # )

        # Sort mapping for guaranteed order
        layerout = tf.stack(
            [
                tf.io.parse_tensor(data[mapping[i].lower()], tf.float32)
                for i in range(len(mapping))
            ],
            axis=-1,
        )
        layerout = tf.reshape(
            layerout, (input_shape[1], len(mapping)), name="reshape_2"
        )

        volume = data["volume"]
        bscan = data["bscan"]
        group = data["group"]

        return {
            # "layer_positions": layer_positions,
            "image": image,
            "layerout": layerout,
            "Volume": volume,
            "Bscan": bscan,
            "Group": group,
        }

    return _parse_function


def get_augment_function(
    contrast_range=(1.0, 1.0), brightnesdelta=0.0, augment_probability=0.0
):
    @tf.function
    def _augment(in_data):
        image, layerout, = (
            in_data["image"],
            in_data["layerout"],
        )

        image = tf.cond(
            tf.random.uniform(shape=[]) < augment_probability,
            lambda: tf.image.random_contrast(image, *contrast_range),
            lambda: image,
        )
        image = tf.cond(
            tf.random.uniform(shape=[]) < augment_probability,
            lambda: tf.image.random_brightness(image, max_delta=brightnesdelta),
            lambda: image,
        )

        return {
            "image": image,
            "layerout": layerout,
            # "layer_positions": in_data["layer_positions"],
        }

    return _augment


@tf.function
def _normalize(in_data):
    image, layerout, = (
        in_data["image"],
        in_data["layerout"],
    )

    # image = tf_clahe.clahe(image)

    image = tf.cast(image, tf.float32)
    image = image - tf.math.reduce_mean(image)
    image = image / tf.math.reduce_std(image)
    return {
        **in_data,
        **{
            "image": image,
            "layerout": layerout,
            # "layer_positions": in_data["layer_positions"],
        },
    }


@tf.function
def _prepare_train(in_data):
    image, layerout = (
        in_data["image"],
        in_data["layerout"],
        # in_data["layer_positions"],
    )

    return image, {
        "layer_output": layerout,
    }  # "columnwise_softmax": layer_positions}


@tf.function
def _prepare_test(in_data):
    volume, bscan, group, image, layerout = (
        in_data["Volume"],
        in_data["Bscan"],
        in_data["Group"],
        in_data["image"],
        in_data["layerout"],
    )
    return image, {
        "layer_output": layerout,
        "Volume": volume,
        "Bscan": bscan,
        "Group": group,
    }


def _prepare_train_layer(layer):
    mapping = {"BM": 0, "RPE": 1, "ILM": 2}

    @tf.function
    def _prepare_train(in_data):
        image, layerout = (
            in_data["image"],
            in_data["layerout"][..., mapping[layer] : mapping[layer] + 1],
        )
        return image, {
            "layer_output": layerout,
            # "columnwise_softmax": in_data["layer_positions"],
        }

    return _prepare_train


def get_transform_func_combined(
    input_shape,
    rotation=(0, 0),
    translation=[0, 0],
    scale=[1, 1],
    num_classes=3,
):
    @tf.function
    def _transform(in_data):
        image, layerout = in_data["image"], in_data["layerout"]
        height, width = input_shape

        # Rotations may lead to one x positions having multiple y positions, this is not supported by our design
        # Get rotation matrix
        # Compute current layer rotations
        layer = layerout[:, 0]
        layer_positions = tf.ragged.boolean_mask(layer, ~tf.math.is_nan(layer))
        height_delta = layer[0] - layer[-1]
        width_delta = tf.reduce_sum(tf.ones_like(layer_positions))
        angle_rad = tf.math.atan(height_delta / width_delta)

        rotation_f = tf.random.uniform(
            shape=[],
            minval=rotation[0] * (np.pi / 180),
            maxval=rotation[1] * (np.pi / 180),
            dtype=tf.dtypes.float32,
        )

        cos_angle = tf.math.cos(rotation_f - angle_rad)
        sin_angle = tf.math.sin(rotation_f - angle_rad)
        rotation_m = tf.convert_to_tensor(
            [cos_angle, -sin_angle, 0, sin_angle, cos_angle, 0, 0, 0]
        )

        # Get translation matrix
        xtranslation_f = tf.random.uniform(
            shape=[],
            minval=translation[0],
            maxval=translation[1],
            dtype=tf.dtypes.float32,
        )
        ytranslation_f = tf.random.uniform(
            shape=[],
            minval=translation[0],
            maxval=translation[1],
            dtype=tf.dtypes.float32,
        )

        # Shift layer to the center
        layerout = tf.transpose(layerout, [1, 0])
        mask = tf.logical_not(tf.math.is_nan(layerout))
        layerout_clean = tf.ragged.boolean_mask(layerout, mask)
        top_layer_mean = tf.reduce_mean(layerout_clean[0])
        bot_layer_mean = tf.reduce_mean(layerout_clean[-1])
        retina_center = (top_layer_mean + bot_layer_mean) / 2
        center_offset = height / 2 - retina_center

        ytranslation_f = -center_offset + ytranslation_f

        translation_m = tf.convert_to_tensor([1, 0, 0, 0, 1, ytranslation_f, 0, 0])

        # Get scaling matrix
        scale_f = tf.random.uniform(
            shape=[], minval=scale[0], maxval=scale[1], dtype=tf.dtypes.float32
        )
        scale_m = tf.convert_to_tensor([scale_f, 0, 0, 0, scale_f, 0, 0, 0])

        # Get flip matrix
        flip = tf.cast(
            tf.random.uniform(shape=[], minval=0, maxval=1 + 1, dtype=tf.int64) * 2 - 1,
            tf.float32,
        )
        flip_m = tf.convert_to_tensor([flip, 0, 0, 0, 1, 0, 0, 0])

        w_offset = width / 2
        h_offset = height / 2
        offset_m = tf.convert_to_tensor([1, 0, w_offset, 0, 1, h_offset, 0, 0])
        reset_m = tf.convert_to_tensor(
            [1, 0, -w_offset * scale_f, 0, 1, -h_offset * scale_f, 0, 0]
        )

        # Combine matrices
        combined_matrix = tfa.image.compose_transforms(
            [
                offset_m,
                rotation_m,
                scale_m,
                translation_m,
                flip_m,
                reset_m,
            ]
        )

        combined_matrix = tfa.image.transform_ops.flat_transforms_to_matrices(
            combined_matrix
        )

        # Warp 2D data
        transformed_image = tfa.image.transform(
            images=image,
            transforms=tfa.image.transform_ops.matrices_to_flat_transforms(
                tf.linalg.inv(combined_matrix)
            ),
            interpolation="bilinear",
            output_shape=input_shape,
        )

        # Warp 1D data
        x_vals = (
            tf.tile(
                tf.reshape(
                    tf.range(0, width, dtype=tf.float32), (1, width), name="reshape_3"
                ),
                [num_classes, 1],
            )
            + 0.5
        )
        z_vals = tf.ones_like(x_vals, dtype=tf.float32)
        # Create position vectors for the height values (num_classes, width ,xyz) - eg(9, 512, 3)
        layer_vectors = tf.stack([x_vals, height - layerout, z_vals], axis=1)
        # tf.print(x_vals.shape, layerout.shape, z_vals.shape, layer_vectors.shape)

        warped_layers = combined_matrix @ layer_vectors

        # iterate over layers
        transformed_layers = tf.TensorArray(tf.float32, size=num_classes)
        for i in range(num_classes):
            notnan_cols = tf.where(~tf.math.is_nan(warped_layers[i, 0, :]))[:, 0]
            warped_layers_clean = tf.gather(warped_layers[i, ...], notnan_cols, axis=1)
            if not tf.equal(tf.size(notnan_cols), 0):
                x_min = warped_layers_clean[0, 0]
                x_max = warped_layers_clean[0, -1]

                # x_min = tf.reduce_max([x_min, 0.5])
                # x_max = tf.reduce_min([x_max, width - 0.5])
                transformed_layers = transformed_layers.write(
                    i,
                    height
                    - tfp.math.batch_interp_regular_1d_grid(
                        x=x_vals[0, ...],
                        x_ref_min=x_min,
                        x_ref_max=x_max,
                        y_ref=warped_layers_clean[1, :],
                        fill_value=float("NaN"),
                    ),
                )
            else:
                transformed_layers = transformed_layers.write(
                    i, tf.fill((width,), float("NaN"))
                )

        transformed_layers = transformed_layers.stack()

        transformed_layers = tf.transpose(transformed_layers, [1, 0])
        transformed_layers = tf.where(transformed_layers < 0.0, 0.0, transformed_layers)
        transformed_layers = tf.where(
            transformed_layers > height - 1.0, height - 1.0, transformed_layers
        )

        return {"image": transformed_image, "layerout": transformed_layers}

    return _transform


def get_split(
    path,
    layer_mapping,
    input_shape,
    batch_size,
    epochs,
    split,
    transform_parameters=None,
    augment_parameters=None,
    seed=42,
):
    transform_parameters = (
        transform_parameters if not transform_parameters is None else {}
    )
    augment_parameters = augment_parameters if not augment_parameters is None else {}

    _transform = get_transform_func_combined(
        input_shape=input_shape, num_classes=len(layer_mapping), **transform_parameters
    )
    _augment = get_augment_function(**augment_parameters)
    _parse_image_function = get_parse_function(layer_mapping, input_shape)

    if split not in ["train", "val", "test"]:
        raise click.UsageError(
            "The 'split' parameters has to be one of train, test or val."
        )

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
            # .map(_transform)
            .map(_normalize)
            .batch(batch_size)
            .map(_prepare_train)
            .repeat(epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    elif split == "test":
        dataset = (
            parsed_data.map(_normalize)
            .batch(batch_size)
            .map(_prepare_test)
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
