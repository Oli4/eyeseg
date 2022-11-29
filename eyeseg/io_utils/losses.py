import tensorflow as tf


def layer_ce(true, pred):
    # true : batch, columns, channels

    # pred: batch, rows, columns, channels
    all = tf.math.log(tf.reduce_sum(true * pred, axis=1))
    mask = tf.logical_not(tf.math.is_inf(all))
    all_clean = tf.ragged.boolean_mask(all, mask)

    return tf.reduce_sum(-tf.reduce_sum(all_clean, axis=1), axis=-1)

    ctrue = tf.cast(tf.round(true), dtype=tf.int32)

    cols = tf.range(1024, dtype=tf.int32)
    layers = tf.range(9, dtype=tf.int32)

    batch_losses = []
    for batch in tf.range(1, dtype=tf.int32):
        layer_losses = []
        for layer in layers:
            layer_results = []
            for c in cols:
                row = ctrue[batch, c, layer]
                # if True: #not tf.math.is_nan(true[batch, c, layer]):
                value = pred[batch, row, c, layer]
                layer_results.append(value)
            layer_losses.append(
                -tf.reduce_sum(
                    tf.math.log(tf.cast(tf.stack(layer_results), tf.float32))
                )
            )
        batch_losses.append(tf.reduce_mean(layer_losses))

    return tf.stack(batch_losses)


class MovingMeanFocalSSE(tf.keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, window_size, curv_weight=0, focus_layer=False):
        super().__init__()
        self.curv_weight = curv_weight
        self.target_window_size = window_size
        self.window_size = tf.Variable(
            0, dtype=tf.float32, name="Window Size", trainable=False
        )
        self.focus_layer = focus_layer
        self.ema = tf.Variable(0, dtype=tf.float32, name="EMA", trainable=False)

    # compute loss
    def call(self, y_true, y_pred):
        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        y_true_clean = tf.ragged.boolean_mask(y_true, mask)
        y_pred_clean = tf.ragged.boolean_mask(y_pred, mask)

        diff = y_true_clean - y_pred_clean
        squared_error = (diff) ** 2

        if self.focus_layer is True:
            layer_scores = tf.math.softmax(
                tf.math.reduce_sum(diff, axis=(0, 1), keepdims=True), axis=2
            )
            focal_mse = tf.math.reduce_sum(layer_scores * squared_error, axis=(1, 2))
            mse = tf.math.reduce_sum(squared_error, axis=(1, 2))
            mse = 0.2 * focal_mse + 0.8 * mse
        elif not self.target_window_size == 0:
            abs_error = tf.math.abs(diff)
            mae = tf.math.reduce_mean(abs_error)
            new_windowsize = tf.cond(
                self.window_size < self.target_window_size,
                lambda: tf.add(self.window_size, 1),
                lambda: self.window_size,
            )
            self.window_size.assign(new_windowsize)
            weight = 2 / (self.window_size + 1)
            self.ema.assign(mae * weight + self.ema * (1 - weight))
            # Select absolute errors greater than the exponential moving mae
            focal_mask = tf.math.greater(abs_error, self.ema)

            # Further increase the weight of hard to predict regions
            focal_mse = tf.math.reduce_sum(
                tf.ragged.boolean_mask(squared_error, focal_mask), axis=(1, 2)
            )
            mse = tf.math.reduce_sum(squared_error, axis=(1, 2))
            mse = focal_mse + mse
        else:
            mse = tf.math.reduce_sum(squared_error, axis=(1, 2))

        # In case the mask contains no values, the mse becomes nan. Change the loss here to 0
        clean_focal_mse = tf.where(tf.math.is_nan(mse), x=tf.zeros_like(mse), y=mse)
        if self.curv_weight != 0:
            true_1deriv = y_true[:, 1:, :] - y_true[:, :-1, :]
            pred_1deriv = y_pred[:, 1:, :] - y_pred[:, :-1, :]

            true_2deriv = true_1deriv[:, 1:, :] - true_1deriv[:, :-1, :]
            pred_2deriv = pred_1deriv[:, 1:, :] - pred_1deriv[:, :-1, :]

            mask = tf.logical_not(tf.math.is_nan(true_2deriv + pred_2deriv))
            true_2deriv_clean = tf.ragged.boolean_mask(true_2deriv, mask)
            pred_2deriv_clean = tf.ragged.boolean_mask(pred_2deriv, mask)
            mse_2deriv = tf.reduce_sum(
                tf.square(true_2deriv_clean - pred_2deriv_clean), axis=(1, 2)
            )
        else:
            mse_2deriv = 0

        return clean_focal_mse + self.curv_weight * mse_2deriv
