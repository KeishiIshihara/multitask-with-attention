import tensorflow as tf
import numpy as np


cce = tf.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1, reduction=tf.keras.losses.Reduction.AUTO,
    name='categorical_crossentropy'
)
mse = tf.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.AUTO,
    name='mean_squared_error'
)
mae = tf.losses.MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.AUTO,
    name='mean_absolute_error'
)

def MSE(y_true, y_pred):
    losses = tf.math.square(y_true - y_pred) # (batch,)
    mse_loss = tf.reduce_sum(losses) * (1. / len(losses)) # reduce batch axis
    return mse_loss, tf.squeeze(losses), # mse_loss_

def weighted_sequence_mse(y_true, y_pred, weights):
    square = tf.math.square(y_true - y_pred)
    sample_losses = tf.reduce_sum(square, axis=-1) # each point should represent one sample loss, (batch,)
    losses = square * weights
    weighted_mse = tf.reduce_sum(losses) * (1 / len(losses))
    return weighted_mse, sample_losses

def weighted_sequence_mse_steer(y_true, y_pred, weights):
    square = tf.math.square(y_true - y_pred)
    weighted_losses = square * weights # sequencely weighted
    weighted_losses = weighted_losses * (tf.math.square(y_true) + 1)
    weighted_mse = tf.reduce_sum(weighted_losses) * (1 / len(weighted_losses))
    return weighted_mse

# The weighted loss: https://stackoverflow.com/a/44563055
# `logits` meaning: https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits
def weighted_softmax_crossentropy(onehot_labels, logits, weights):
    # Not sure about the shape or ndim of the weights arg
    weights = tf.reduce_sum(weights * onehot_labels, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
    weighted_losses = unweighted_losses * weights
    loss = tf.reduce_mean(weighted_losses)
    return loss

weighted_softmax_crossentropy_with_logits = weighted_softmax_crossentropy

def scale_invariant_error(y_true, y_pred):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    Source:
        https://github.com/lmb-freiburg/demon/blob/master/python/depthmotionnet/evaluation/metrics.py#L128
    """
    # sqrt(Eq. 3)
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = tf.math.log(y_true) - tf.math.log(y_pred)
    num_pixels = log_diff.size

    return tf.math.sqrt(tf.reduce_sum(tf.math.square(log_diff)) / num_pixels - tf.math.square(tf.reduce_sum(log_diff)) / tf.math.square(num_pixels))
