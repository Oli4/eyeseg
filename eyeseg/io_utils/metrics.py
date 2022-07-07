import tensorflow as tf


def get_mae(name="mae"):
    def func(y_true, y_pred):
        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)
        return tf.math.abs(y_true - y_pred)

    return tf.keras.metrics.MeanMetricWrapper(func, name)


def get_layer_mae(index, name):
    def func(y_true, y_pred):
        y_true = y_true[..., index]
        y_pred = y_pred[..., index]

        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)
        return tf.math.abs(y_true - y_pred)

    return tf.keras.metrics.MeanMetricWrapper(func, f"mae_{name}")


def get_layer_curv(index, name):
    def func(y_true, y_pred):
        true_1deriv = y_true[:, 1:, index] - y_true[:, :-1, index]
        pred_1deriv = y_pred[:, 1:, index] - y_pred[:, :-1, index]

        true_2deriv = true_1deriv[:, 1:] - true_1deriv[:, :-1]
        pred_2deriv = pred_1deriv[:, 1:] - pred_1deriv[:, :-1]

        mask = tf.logical_not(tf.math.is_nan(true_2deriv + pred_2deriv))
        true_2deriv_clean = tf.ragged.boolean_mask(true_2deriv, mask)
        pred_2deriv_clean = tf.ragged.boolean_mask(pred_2deriv, mask)
        # mse_2deriv = tf.reduce_mean()
        return tf.square(true_2deriv_clean - pred_2deriv_clean)

    return tf.keras.metrics.MeanMetricWrapper(func, f"curv_{name}")


def get_curv(name="curv"):
    def func(y_true, y_pred):
        true_1deriv = y_true[:, 1:, :] - y_true[:, :-1, :]
        pred_1deriv = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        true_2deriv = true_1deriv[:, 1:, :] - true_1deriv[:, :-1, :]
        pred_2deriv = pred_1deriv[:, 1:, :] - pred_1deriv[:, :-1, :]

        mask = tf.logical_not(tf.math.is_nan(true_2deriv + pred_2deriv))
        true_2deriv_clean = tf.ragged.boolean_mask(true_2deriv, mask)
        pred_2deriv_clean = tf.ragged.boolean_mask(pred_2deriv, mask)
        return tf.square(true_2deriv_clean - pred_2deriv_clean)

    return tf.keras.metrics.MeanMetricWrapper(func, name)
