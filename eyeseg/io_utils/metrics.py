import tensorflow as tf


def get_mae(name="mae"):
    @tf.function
    def mae(y_true, y_pred):
        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)
        # y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        # y_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred)
        return tf.math.abs(y_true - y_pred)

    mae.__name__ = f"{name}"
    return mae


def get_drusenregion_mae(name="mae"):
    @tf.function
    def drusenregion_mae(y_true, y_pred, sample_weight):
        # sample_weight = sample_weight["drusen_region"]
        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        mask = tf.logical_and(mask, tf.math.equal(sample_weight, 1))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)
        # y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        # y_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred)
        return tf.math.abs(y_true - y_pred)

    drusenregion_mae.__name__ = f"drusen_region_{name}"
    return drusenregion_mae


class DrusenregionMAE(tf.keras.metrics.Metric):
    def __init__(self, name="mae", **kwargs):
        super(DrusenregionMAE, self).__init__(name=name, **kwargs)
        self.mae_sum = self.add_weight(name="ms", initializer="zeros")
        self.n_samples = self.add_weight(name="samples", initializer="zeros")
        self.__name__ = "DrusenregionMAE"

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight):

        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        mask = tf.logical_and(mask, tf.math.equal(sample_weight, 1))
        mask.set_shape([None, None, None])
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)

        def add_drusenregion(y_true, y_pred):
            return tf.reduce_mean(tf.math.abs(y_true - y_pred))

        condition = tf.equal(tf.math.reduce_sum(tf.cast(mask, tf.float32)), 0)
        mae = tf.cond(
            condition,
            lambda: tf.constant(0, dtype=tf.float32),
            lambda: add_drusenregion(y_true, y_pred),
        )
        count = tf.cond(
            condition,
            lambda: tf.constant(0, dtype=tf.float32),
            lambda: tf.constant(1, dtype=tf.float32),
        )

        self.mae_sum.assign_add(mae)
        self.n_samples.assign_add(count)

    def result(self):
        return self.mae_sum / self.n_samples

    def reset_states(self):
        self.mae_sum.assign(0)
        self.n_samples.assign(0)


def get_layer_mae(index, name):
    @tf.function
    def layer_mae(y_true, y_pred):
        # y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        # y_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred)
        y_true = y_true[..., index]
        y_pred = y_pred[..., index]

        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)
        return tf.math.abs(y_true - y_pred)

    layer_mae.__name__ = f"mae_{name}"
    return layer_mae


def get_layer_drusenregion_mae(index, name):
    @tf.function
    def layer_mae(y_true, y_pred, sample_weight):
        # sample_weight = sample_weight["drusen_region"]
        # y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        # y_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred)
        y_true = y_true[..., index]
        y_pred = y_pred[..., index]

        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        mask = tf.logical_and(mask, tf.math.equal(sample_weight, 1))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)
        return tf.math.abs(y_true - y_pred)

    layer_mae.__name__ = f"drusenregion_mae_{name}"
    return layer_mae


def get_regularization_metric(name="LayerOrderRegularization"):
    def reg_met(y_true, y_pred):
        reg_bmrpe = y_pred[..., 0] - y_pred[..., 1]
        reg_bmrpe = tf.reduce_sum(
            tf.square(tf.where(reg_bmrpe < 0, reg_bmrpe, tf.zeros_like(reg_bmrpe)))
        )

        reg_rpeez = y_pred[..., 1] - y_pred[..., 2]
        reg_rpeez = tf.reduce_sum(
            tf.square(tf.where(reg_rpeez < 0, reg_rpeez, tf.zeros_like(reg_rpeez)))
        )

        reg_bmez = y_pred[..., 0] - y_pred[..., 2]
        reg_bmez = tf.reduce_sum(
            tf.square(tf.where(reg_bmez < 0, reg_bmez, tf.zeros_like(reg_bmez)))
        )

        reg_loss = reg_bmez + reg_rpeez + reg_bmrpe
        return reg_loss

    reg_met.__name__ = f"{name}"
    return reg_met


class DrusenregionLayerMAE(tf.keras.metrics.Metric):
    def __init__(self, index, name="mae", **kwargs):
        super(DrusenregionLayerMAE, self).__init__(name=f"{name}_mae", **kwargs)
        self.mae_sum = self.add_weight(name="ms", initializer="zeros")
        self.n_samples = self.add_weight(name="samples", initializer="zeros")

        self.index = index
        self.__name__ = "DrusenregionLayerMAE"

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight):
        y_true = y_true[..., self.index]
        y_pred = y_pred[..., self.index]

        mask = tf.logical_not(tf.math.is_nan(y_true + y_pred))
        sample_weight = tf.reshape(sample_weight, [-1, 512])
        mask = tf.logical_and(mask, tf.math.equal(sample_weight, 1))
        y_true = tf.ragged.boolean_mask(y_true, mask)
        y_pred = tf.ragged.boolean_mask(y_pred, mask)

        def add_drusenregion(y_true, y_pred):
            return tf.reduce_mean(tf.math.abs(y_true - y_pred))

        condition = tf.equal(tf.math.reduce_sum(tf.cast(mask, tf.float32)), 0)
        mae = tf.cond(
            condition,
            lambda: tf.constant(0, dtype=tf.float32),
            lambda: add_drusenregion(y_true, y_pred),
        )
        count = tf.cond(
            condition,
            lambda: tf.constant(0, dtype=tf.float32),
            lambda: tf.constant(1, dtype=tf.float32),
        )

        self.mae_sum.assign_add(mae)
        self.n_samples.assign_add(count)

    def result(self):
        return self.mae_sum / self.n_samples

    def reset_states(self):
        self.mae_sum.assign(0)
        self.n_samples.assign(0)


def get_layer_curv(index, name):
    @tf.function
    def curv(y_true, y_pred):
        true_1deriv = y_true[:, 1:, index] - y_true[:, :-1, index]
        pred_1deriv = y_pred[:, 1:, index] - y_pred[:, :-1, index]

        true_2deriv = true_1deriv[:, 1:] - true_1deriv[:, :-1]
        pred_2deriv = pred_1deriv[:, 1:] - pred_1deriv[:, :-1]

        mask = tf.logical_not(tf.math.is_nan(true_2deriv + pred_2deriv))
        true_2deriv_clean = tf.ragged.boolean_mask(true_2deriv, mask)
        pred_2deriv_clean = tf.ragged.boolean_mask(pred_2deriv, mask)
        # mse_2deriv = tf.reduce_mean()
        return tf.square(true_2deriv_clean - pred_2deriv_clean)

    curv.__name__ = f"curv_{name}"
    return curv


def get_curv(name="curv"):
    @tf.function
    def curv(y_true, y_pred):
        true_1deriv = y_true[:, 1:, :] - y_true[:, :-1, :]
        pred_1deriv = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        true_2deriv = true_1deriv[:, 1:, :] - true_1deriv[:, :-1, :]
        pred_2deriv = pred_1deriv[:, 1:, :] - pred_1deriv[:, :-1, :]

        mask = tf.logical_not(tf.math.is_nan(true_2deriv + pred_2deriv))
        true_2deriv_clean = tf.ragged.boolean_mask(true_2deriv, mask)
        pred_2deriv_clean = tf.ragged.boolean_mask(pred_2deriv, mask)
        # mse_2deriv = tf.reduce_mean()
        return tf.square(true_2deriv_clean - pred_2deriv_clean)

    curv.__name__ = f"{name}"
    return curv
