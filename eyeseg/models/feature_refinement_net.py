from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

from typing import Tuple

from eyeseg.models.parts import get_dilation_convblock, get_output


def model(
    input_shape: Tuple = (256, 256, 1),
    num_classes: int = 2,
    filters: int = 64,
    kernel_size: Tuple = (3, 3),
    kernel_initializer: str = "he_uniform",
    activation: str = "swish",
    normalization: str = "batch_norm",
    dropout: float = 0.1,
    spatial_dropout: float = 0.1,
    guaranteed_order: bool = True,
    self_attention: bool = True,
    residual_learning: bool = True,
    norm_last: bool = True,
    soft_layerhead=False,
) -> Model:
    def clip_norm():
        def clip_norm_func(input):
            return tf.keras.backend.clip(input, -100, 100)

        return clip_norm_func

    # def weight_norm():
    #    def weight_norm_func(input):
    #        return tfa.layers.WeightNormalization(input)
    #    return weight_norm_func

    if normalization == "batch_norm":
        norm_func = layers.BatchNormalization
    elif normalization == "layer_norm":
        norm_func = layers.LayerNormalization
    elif normalization == "clip_norm":
        norm_func = clip_norm
    # elif normalization == "weight_norm":
    #    norm_func = weight_norm
    # elif normalization == "instance_norm":
    #    norm_func = tfa.layers.InstanceNormalization
    # elif normalization == "filter_response_norm":
    #    norm_func = tfa.layers.FilterResponseNormalization
    else:
        raise ValueError("Normalization not known")

    # Receptive field 727
    dilations = [
        [3, 9, 3],
        [9, 27, 9],
        [27, 243, 27],
    ]

    conv_blocks = [
        get_dilation_convblock(
            filters,
            kernel_size,
            kernel_initializer,
            dilation_rates=d,
            name=i,
            sp_dropout=spatial_dropout,
            activation=activation,
            normalization=norm_func,
            norm_last=norm_last,
        )
        for i, d in enumerate(dilations)
    ]
    inputs = layers.Input(input_shape)
    x = inputs
    for i in [filters // 4, filters // 2, filters]:
        conv = layers.Conv2D(
            i,
            (3, 3),
            kernel_initializer=kernel_initializer,
            padding="same",
            name=f"Conv1x1_expand_{i}",
        )
        activation_layer = layers.Activation(activation)
        drop_layer = layers.Dropout(rate=dropout)
        batchnorm = norm_func()
        x = batchnorm(drop_layer(activation_layer(conv(x))))

    for i, conv_block in enumerate(conv_blocks):
        # Comput higher level features
        new_x = conv_block(x)
        # Reorganize higher level features for combination with lower level features
        conv = layers.Conv2D(
            filters,
            (1, 1),
            kernel_initializer=kernel_initializer,
            padding="same",
            name=f"Conv1x1_featurematch_{i}",
        )

        activation_layer = layers.Activation(activation)
        if residual_learning:
            # Combine higher level features with lower level features, then BN+Activation, in contrast to BN+Activaiton
            # on the recombined high level features. This makes it straight forward not to change lower level feature maps
            # by the recombination, weighting all input maps with 0 to produce a 0 output map. Therefore we also avoid an
            # activation after summation here.

            x = norm_func()(layers.Add()([x, activation_layer(conv(new_x))]))
        else:
            x = norm_func()(activation_layer(conv(new_x)))

    if self_attention:
        output_features = layers.Conv2D(
            filters, (1, 1), kernel_initializer=kernel_initializer
        )(x)
        output_features = layers.Activation("swish")(norm_func()(output_features))
        attention_map = layers.Conv2D(
            filters, (1, 1), kernel_initializer=kernel_initializer
        )(x)
        attention_map = layers.Activation("sigmoid")(norm_func()(attention_map))
        outputs = layers.Multiply()([output_features, attention_map])
    else:
        outputs = x

    if soft_layerhead:
        outputs, col_softmax = get_output(
            outputs,
            num_classes,
            input_shape,
            guaranteed_order=guaranteed_order,
            soft=soft_layerhead,
        )
        output = layers.Reshape((input_shape[1], num_classes), name="layer_output")(
            outputs
        )
        model = Model(inputs=[inputs], outputs=[output, col_softmax])
    else:
        outputs, _ = get_output(
            outputs,
            num_classes,
            input_shape,
            guaranteed_order=guaranteed_order,
            soft=soft_layerhead,
        )
        output = layers.Reshape((input_shape[1], num_classes), name="layer_output")(
            outputs
        )
        model = Model(inputs=[inputs], outputs=[output])

    return model
