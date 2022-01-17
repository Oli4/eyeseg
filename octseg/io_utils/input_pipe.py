import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp


def get_parse_function(mapping, **extra_features):
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
        image = tf.reshape(image, (512, 512, 1))

        # Sort mapping for guaranteed order
        layerout = tf.stack(
            [
                tf.io.parse_tensor(data[mapping[i].lower()], tf.float32)
                for i in range(len(mapping))
            ],
            axis=-1,
        )
        layerout = tf.reshape(layerout, (512, len(mapping)))

        volume = data["volume"]
        bscan = data["bscan"]
        group = data["group"]

        return {
            "image": image,
            "layerout": layerout,
            "Volume": volume,
            "Bscan": bscan,
            "Group": group,
        }

    return _parse_function


def get_augment_function(
    contrast_range=(0.9, 1.1),
    brightnesdelta=0.1,
    augment_probability=1.0,
    occlusion=False,
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

        mean = tf.random.uniform(shape=[]) * 512
        std = tf.random.uniform(shape=[]) * 128

        dst = tf.range(0 - mean, 512 - mean, 1)
        gaus = tf.exp(-((dst - mean) ** 2 / (2.0 * std ** 2)))
        occlusion = tf.abs(tf.tile(tf.reshape(gaus, (1, 512)), (512, 1)) - 1)

        tf.cond(
            tf.logical_and(occlusion, tf.random.uniform(shape=[]) < 0.5),
            lambda: image * occlusion,
        )
        # image = tf.cond(tf.random.uniform(shape=[]) < augment_probability,
        #                lambda: tf.image.adjust_gamma(image, gamma=,  gain=),
        #                lambda: image)

        return {"image": image, "layerout": layerout}

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
    return {**in_data, **{"image": image, "layerout": layerout}}


@tf.function
def _prepare_train(in_data):
    image, layerout = in_data["image"], in_data["layerout"]
    return image, {
        "layer_output": layerout,
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
        }

    return _prepare_train


def get_transform_func_combined(
    rotation=(0, 0), translation=[0, 0], scale=[1, 1], num_classes=3
):
    @tf.function
    def _transform(in_data):
        image, layerout = in_data["image"], in_data["layerout"]

        # Rotations may lead to one x positions having multiple y positions, this is not supported by our design
        # Get rotation matrix
        # Compute current layer rotations
        # bm = layerout[:, 0]
        # bm_positions = tf.ragged.boolean_mask(bm,  ~tf.math.is_nan(bm))
        # height_delta = bm_positions[0] - bm_positions[-1]
        # width_delta = tf.reduce_sum(tf.ones_like(bm_positions))
        # angle_rad = tf.math.atan(height_delta / width_delta)

        # rotation_f = tf.random.uniform(shape=[], minval=rotation[0]*(np.pi/180), maxval=rotation[1]*(np.pi/180), dtype=tf.dtypes.float32)

        # cos_angle = tf.math.cos(rotation_f)#-angle_rad)
        # sin_angle = tf.math.sin(rotation_f)#-angle_rad)
        # rotation_m = tf.convert_to_tensor([cos_angle, -sin_angle, 0, sin_angle, cos_angle, 0, 0, 0])

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
        retina_center = tf.reduce_mean(layerout_clean[0]) - 75

        ytranslation_f = -(256 - retina_center) + ytranslation_f
        translation_m = tf.convert_to_tensor([1, 0, 0, 0, 1, ytranslation_f, 0, 0])
        # translation_m = tf.convert_to_tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0])

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

        w_offset = (512 - 1) / 2
        h_offset = (512 - 1) / 2
        offset_m = tf.convert_to_tensor([1, 0, w_offset, 0, 1, h_offset, 0, 0])
        reset_m = tf.convert_to_tensor([1, 0, -w_offset, 0, 1, -h_offset, 0, 0])
        # Combine matrices
        combined_matrix = tfa.image.transform_ops.compose_transforms(
            [
                offset_m,  # rotation_m,
                scale_m,
                flip_m,
                translation_m,
                reset_m,
            ]
        )

        # Warp 2D data
        transformed_image = tfa.image.transform(
            images=image, transforms=combined_matrix, interpolation="bilinear"
        )

        # Warp 1D data
        combined_matrix = tfa.image.transform_ops.flat_transforms_to_matrices(
            combined_matrix
        )

        x_vals = (
            tf.tile(
                tf.reshape(tf.range(0, 512, dtype=tf.float32), (1, 512)),
                [num_classes, 1],
            )
            + 0.5
        )
        z_vals = tf.ones_like(x_vals, dtype=tf.float32)
        # Create position vectors for the height values (3, 512 ,3) - (layers, width, xyz)
        layer_vectors = tf.stack([x_vals, layerout, z_vals], axis=1)
        combined_matrix = tf.linalg.inv(combined_matrix)

        warped_layers = combined_matrix @ layer_vectors

        # iterate over layers
        transformed_layers = tf.TensorArray(tf.float32, size=num_classes)
        x_min = 0.0
        x_max = 0.0
        for i in range(num_classes):
            notnan_cols = tf.where(~tf.math.is_nan(warped_layers[i, 0, :]))[:, 0]
            warped_layers_clean = tf.gather(warped_layers[i, ...], notnan_cols, axis=1)
            if not tf.equal(tf.size(notnan_cols), 0):
                x_min = warped_layers_clean[0, 0]
                x_max = warped_layers_clean[0, -1]
                transformed_layers = transformed_layers.write(
                    i,
                    tfp.math.batch_interp_regular_1d_grid(
                        x=x_vals[0, ...],
                        x_ref_min=x_min,
                        x_ref_max=x_max,
                        y_ref=warped_layers_clean[1, :],
                        fill_value=float("NaN"),
                    ),
                )
            else:
                transformed_layers = transformed_layers.write(
                    i, tf.fill((512,), float("NaN"))
                )

        transformed_layers = transformed_layers.stack()

        transformed_layers = tf.transpose(transformed_layers, [1, 0])
        transformed_layers = tf.where(transformed_layers < 0.0, 0.0, transformed_layers)
        transformed_layers = tf.where(
            transformed_layers > 511.0, 511.0, transformed_layers
        )

        return {"image": transformed_image, "layerout": transformed_layers}

    return _transform
