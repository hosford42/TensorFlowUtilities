import tensorflow as tf


@tf.custom_gradient
def sample_multivariate_normal(mean, stddev, batch_shape=(), seed=None) -> tf.Tensor:
    """Produce a sample from a multivariate normal distribution with the given mean and
    lower-triangular stddev matrix, with the appropriate gradients so the correct mean and stddev
    matrix are learned when the loss is computed as the MSE of the model-generated sample and a
    sample randomly drawn from the distribution to be learned. (This is in contrast to the way
    that tfp.distributions.MultivariateNormalTriL samples from the distribution, which results in
    the stddev matrix asymptotically approaching zero once the mean is learned, regardless of the
    target distribution's characteristics.)"""
    mean = tf.convert_to_tensor(mean)
    stddev = tf.convert_to_tensor(stddev)
    batch_shape = tf.convert_to_tensor(batch_shape, dtype=tf.int32)
    batch_size = tf.reduce_prod(batch_shape)

    sample = tf.stop_gradient(sample_mv_normal(mean, stddev, batch_shape, seed))

    def grad(upstream):
        mean_grad = upstream

        average_sample_grad_sq = tf.reduce_mean(upstream[..., :, tf.newaxis] *
                                                upstream[..., tf.newaxis, :],
                                                axis=tf.range(tf.size(batch_shape)))
        model_variance = stddev_to_variance(stddev)
        difference = ((average_sample_grad_sq *
                       tf.cast(tf.square(batch_size), mean.dtype) /
                       tf.constant(4.0)) -
                      model_variance)
        stddev_target = tf.linalg.cholesky(difference)

        valid = tf.reduce_all(tf.math.is_finite(stddev_target), axis=(-2, -1), keepdims=True)
        stddev_grad = tf.where(valid, stddev_target - stddev, 0.0)

        return mean_grad, stddev_grad

    # noinspection PyTypeChecker
    return sample, grad
