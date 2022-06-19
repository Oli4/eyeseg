from tensorflow.keras import layers
from tensorflow.keras.models import Model

from eyeseg.models.parts import get_dilation_convblock, get_msgb, get_output


def gated_dilation_net(
    input_shape=(256, 256, 1),
    num_classes=2,
    filters=64,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
    dilations=None,
    activation="swish",
):

    if dilations is None:
        dilations = [[1, 1], [2, 2], [4, 4], [8, 8], [4, 4], [2, 2], [1, 1]]
    dilations_flat = [x for sublist in dilations for x in sublist]

    conv_blocks = [
        get_dilation_convblock(
            f,
            kernel_size,
            kernel_initializer,
            dilation_rates=d,
            name=i,
            activation=activation,
        )
        for i, (f, d) in enumerate(
            zip(
                [
                    filters,
                ]
                * len(dilations_flat),
                dilations,
            )
        )
    ]
    msgbs = [
        get_msgb(f, i, kernel_initializer)
        for i, (f, d) in enumerate(
            zip(
                [
                    filters,
                ]
                * len(dilations),
                dilations,
            )
        )
    ]

    inputs = layers.Input(input_shape)
    x = inputs
    gate_outputs = []

    for msgb, conv_block in zip(msgbs, conv_blocks):
        out = conv_block(x)
        gate_outputs.append(msgb(out))
        x = layers.concatenate([x, out])

    outputs = layers.concatenate(gate_outputs)
    output = get_output(outputs, num_classes, input_shape)

    output = layers.Reshape((input_shape[1], num_classes), name="layer_output")(output)

    model = Model(inputs=[inputs], outputs=[output])
    return model


def original_gdn(
    input_shape=(256, 256, 1),
    num_classes=2,
    filters=64,
    kernel_size=(3, 3),
    dilation_factor=2,
    depth=4,
    kernel_initializer="he_uniform",
    activation="relu",
):
    # Depth: 4
    # With kernel_size (3,3)-dilation_factor 2 RF: 89
    # With kernel_size (5,5)-dilation_factor 3 RF: 425
    powers = list(range(depth)) + list(range(depth - 1))[::-1]
    dilations = [[dilation_factor**p, dilation_factor**p] for p in powers]
    return gated_dilation_net(
        input_shape,
        num_classes,
        filters,
        kernel_size,
        kernel_initializer,
        dilations,
        activation,
    )


def gdn_no_gridding(
    input_shape=(256, 256, 1),
    num_classes=2,
    filters=64,
    kernel_size=(5, 5),
    kernel_initializer="he_uniform",
):
    # With kernel_size (3,3) RF:217; With kernel_size (5,5) RF: 433
    dilations = [[1, 1], [3, 5], [7, 11], [23, 29], [11, 7], [5, 3], [1, 1]]
    return gated_dilation_net(
        input_shape, num_classes, filters, kernel_size, kernel_initializer, dilations
    )


def gdn_no_gridding_full_rf(
    input_shape=(256, 256, 1),
    num_classes=2,
    filters=64,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
):
    # With kernel_size (3,3) RF: 509
    dilations = [
        [1, 1, 1],
        [2, 3, 5],
        [7, 11, 13],
        [23, 29, 31],
        [31, 29, 23],
        [13, 11, 7],
        [5, 3, 2],
        [1, 1, 1],
    ]
    return gated_dilation_net(
        input_shape, num_classes, filters, kernel_size, kernel_initializer, dilations
    )
