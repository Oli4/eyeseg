import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf


class FRNLayer(layers.Layer):
    def __init__(self):
        super(FRNLayer, self).__init__()

    def build(self, input_shape):
        self.input_spec = [keras.layers.InputSpec(shape=input_shape)]
        self.beta = K.variable(
            keras.initializers.get("zeros")((1, 1, 1, input_shape[-1]))
        )
        self.gamma = K.variable(
            keras.initializers.get("ones")((1, 1, 1, input_shape[-1]))
        )
        self.eps = K.variable(1e-6)
        self._trainable_weights = [self.beta, self.gamma, self.eps]

    def call(self, x):
        nu2 = K.mean(K.square(x), axis=[1, 2], keepdims=True)
        x = x * (1.0 / K.sqrt(nu2 + K.abs(self.eps)))
        return x * self.gamma + self.beta


class TLU(layers.Layer):
    def __init__(self):
        super(TLU, self).__init__()

    def build(self, input_shape):
        self.input_spec = [keras.layers.InputSpec(shape=input_shape)]
        self.tau = K.variable(
            keras.initializers.get("zeros")((1, 1, 1, input_shape[-1]))
        )
        self._trainable_weights = [self.tau]

    def call(self, x):
        return K.maximum(x, self.tau)


class TSwish(layers.Layer):
    def __init__(self):
        super(TSwish, self).__init__()

    def build(self, input_shape):
        self.input_spec = [keras.layers.InputSpec(shape=input_shape)]
        self.shift = K.variable(
            keras.initializers.get("zeros")((1, 1, 1, input_shape[-1]))
        )
        self._trainable_weights = [self.shift]

    def call(self, x):
        x = x + self.shift
        return x * K.sigmoid(x)


def get_dilation_convblock(
    filters,
    kernel_size=(3, 3),
    kernel_initializer="he_uniform",
    dilation_rates=(1, 2, 3),
    sp_dropout=0.2,
    activation="swish",
    name=0,
    normalization="batch_norm",
    norm_last=False,
):
    if normalization == "batch_norm":
        norm_func = layers.BatchNormalization
    elif normalization == "layer_norm":
        norm_func = layers.LayerNormalization
    elif callable(normalization):
        norm_func = normalization
    else:
        raise ValueError("Normalization not known")

    if type(filters) is int:
        filters = [
            filters,
        ] * len(dilation_rates)

    conv = layers.Conv2D(
        filters[0],
        (1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        name=f"Conv1x1_{name}",
    )

    if activation is None:
        activation_layer = lambda: layers.Lambda(lambda x: x)
    elif callable(activation):
        activation_layer = lambda: activation()
    else:
        activation_layer = lambda: layers.Activation(activation)

    dil_convs = [
        layers.Conv2D(
            f,
            kernel_size,
            kernel_initializer=kernel_initializer,
            padding="same",
            dilation_rate=r,
            name=f"{name}_{i}_DilatedConv_{r}",
        )
        for i, (r, f) in enumerate(zip(dilation_rates, filters))
    ]
    dil_activations = [activation_layer() for _ in dilation_rates]
    dil_normalization = [norm_func() for _ in dilation_rates]

    spatial_dropout = layers.SpatialDropout2D(rate=sp_dropout)

    def dilation_convblock(inputs):
        if norm_last:
            x = norm_func()(activation_layer()(conv(inputs)))
        else:
            x = activation_layer()(norm_func()(conv(inputs)))

        if sp_dropout > 0:
            x = spatial_dropout(x)
        dilation_ready = x

        for c, b, a in zip(dil_convs, dil_normalization, dil_activations):
            if norm_last:
                dilation_ready = b(a(c(dilation_ready)))
            else:
                dilation_ready = a(b(c(dilation_ready)))

        return dilation_ready

    return dilation_convblock


def get_msgb(
    filters,
    name,
    kernel_initializer="he_uniform",
):
    # Multi-Scale Gate Block
    conv = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        kernel_initializer=kernel_initializer,
        name=f"MSGBConv_{name}",
    )

    def msgb(x):
        x = conv(x)
        split = tf.split(x, 2, -1)
        return layers.Multiply()(
            [activations.tanh(split[0]), activations.sigmoid(split[1])]
        )

    return msgb


def get_output(input, num_classes, input_shape, guaranteed_order=True, soft=False):
    if soft:
        output_top_to_bottom = tf.keras.layers.Conv2D(
            num_classes, (1, 1), kernel_initializer="he_uniform"
        )(input)

        col_softmax = layers.Softmax(axis=1, name="columnwise_softmax")(
            output_top_to_bottom
        )

        col_softargmax = tf.reduce_sum(
            col_softmax
            * tf.reshape(tf.cast(tf.range(0, input_shape[0]), tf.float32), (-1, 1, 1)),
            axis=1,
        )

        if guaranteed_order:
            output_list_top_to_bottom = [col_softargmax[..., -1]]
            for i in range(num_classes - 2, -1, -1):
                output_list_top_to_bottom.insert(
                    0,
                    output_list_top_to_bottom[0]
                    + tf.keras.activations.relu(
                        col_softargmax[..., i] - output_list_top_to_bottom[0]
                    ),
                )
            output_top_to_bottom = tf.stack(
                output_list_top_to_bottom, axis=-1, name="layer_output"
            )

        # Return the layer heights for L1 loss and the column-wise softmax for CE Loss as in He et al
        return output_top_to_bottom, col_softmax

    else:
        output_top_to_bottom = layers.Conv2D(
            num_classes, (1, 1), activation="relu", kernel_initializer="he_uniform"
        )(input)
        # Every map becomes the
        if guaranteed_order:
            output_list_top_to_bottom = [output_top_to_bottom[..., -1]]
            for i in range(num_classes - 2, -1, -1):
                output_list_top_to_bottom.insert(
                    0, output_list_top_to_bottom[0] + output_top_to_bottom[..., i]
                )
            output_top_to_bottom = tf.stack(output_list_top_to_bottom, axis=-1)

        output_top_to_bottom = tf.clip_by_value(output_top_to_bottom, 0, 1)
        output_top_to_bottom = layers.AveragePooling2D((input_shape[0], 1))(
            output_top_to_bottom
        )
        output_top_to_bottom = tf.math.multiply(
            output_top_to_bottom, input_shape[0], name="layer_output"
        )

        return output_top_to_bottom, False
