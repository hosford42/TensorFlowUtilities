import tensorflow as tf

try:
    import tensorflow_probability as tfp
except ImportError:
    tfp = None


@tf.function
def recompose_cholesky(lower) -> tf.Tensor:
    """Given the lower triangular cholesky decomposition of a matrix, compute the original matrix
    that was factored and return it."""
    lower = tf.convert_to_tensor(lower)
    return tf.matmul(lower, lower, transpose_b=True)


@tf.function
def stddev_to_flat(stddev, multivariate=True) -> tf.Tensor:
    """Convert a lower triangular matrix of shape (N, N) to a vector of shape (N * (N + 1) // 2,)."""
    stddev = tf.convert_to_tensor(stddev)
    if multivariate:
        if tfp is None:
            raise ImportError("This functionality requires tensorflow_probability to be installed.")
        return tfp.math.fill_triangular_inverse(stddev)
    else:
        return stddev  # It's already flat


@tf.function
def flat_to_stddev(flat, multivariate=True, min_stddev=0.000001, stddev_bias=0.0) -> tf.Tensor:
    """Convert a vector of shape (N * (N + 1) / 2,) to a lower triangular matrix of shape (N, N),
    ensuring that the diagonal is strictly positive, as would be expected for the scale parameter of
    tensorflow_probability's MultivariateNormalTriL class, which is the cholesky decomposition of
    the covariance matrix."""
    flat = tf.convert_to_tensor(flat)
    if multivariate:
        if tfp is None:
            raise ImportError("This functionality requires tensorflow_probability to be installed.")
        lower = tfp.math.fill_triangular(flat)
        lower = tf.linalg.set_diag(lower,
                                   tf.abs(tf.linalg.diag_part(lower) + stddev_bias) + min_stddev)
        tf.assert_greater(tf.linalg.diag_part(lower), 0.0)
        return lower
    else:
        return tf.abs(flat + stddev_bias) + min_stddev


@tf.function
def stddev_to_variance(stddev, multivariate=True) -> tf.Tensor:
    """Convert a standard deviation vector or cholesky decomposition of the covariance matrix to a
    variance vector or covariance matrix, respectively."""
    stddev = tf.convert_to_tensor(stddev)
    if multivariate:
        tf.assert_greater(tf.rank(stddev), 1)
        tf.assert_equal(tf.shape(stddev)[-2], tf.shape(stddev)[-1])
        return recompose_cholesky(stddev)
    else:
        tf.assert_equal(multivariate, False)
        return tf.square(stddev)


@tf.function
def variance_to_stddev(variance, multivariate=True) -> tf.Tensor:
    """Convert a variance vector or covariance matrix to a standard deviation vector or cholesky
    decomposition of the covariance matrix, respectively."""
    variance = tf.convert_to_tensor(variance)
    if multivariate:
        return tf.linalg.cholesky(variance)
    else:
        return tf.sqrt(variance)


@tf.function
def condition_covariance_matrix(variance, min_var=0.000001, jitter=0.000001) -> tf.Tensor:
    """Ensure that the covariance matrix has the expected properties of being symmetric and
    positive definite by making minor adjustments to it."""
    # Ensure symmetric
    variance = 0.5 * (variance + transpose_matrix(variance))

    # Ensure positive definite
    diagonal = tf.linalg.diag_part(variance)
    diagonal = diagonal - tf.stop_gradient(tf.minimum(diagonal, 0.0))
    diagonal = diagonal + min_var
    if jitter > 0.0:
        diagonal = diagonal + tf.random.uniform(tf.shape(diagonal), 0.0, jitter)
    variance = tf.linalg.set_diag(variance, diagonal)
    return variance


@tf.function
def dynamically_condition_covariance_matrix(variance, min_var=0.0001, jitter=0.0001, max_iters=10,
                                            min_var_factor=2.0, jitter_factor=1.0) -> tf.Tensor:
    """Ensure that the covariance matrix has the expected properties of being symmetric and
    positive definite by making minor adjustments to it. If minor adjustments fail, try repeatedly
    with increasing adjustment sizes until they succeed or max_iters is exhausted."""
    variance = tf.convert_to_tensor(variance)
    min_var = tf.convert_to_tensor(min_var)
    min_var_factor = tf.convert_to_tensor(min_var_factor)
    jitter_factor = tf.convert_to_tensor(jitter_factor)
    for _ in tf.range(max_iters):
        stddev = variance_to_stddev(variance)
        if tf.reduce_all(tf.math.is_finite(stddev)):
            break
        variance = condition_covariance_matrix(variance, min_var, jitter)
        min_var = min_var * min_var_factor
        jitter = jitter * jitter_factor
    return variance


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


@tf.function
def unit_vector_to_rotation_matrix(orientation, initial_orientation=None) -> tf.Tensor:
    """Assuming the given initial orientation, return the rotation matrix corresponding to the
    given orientation vector. If no initial orientation is specified, the unit vector
    [1, 0, ..., 0] is assumed. Both arguments should be unit-length vectors.

    To apply the rotation matrix to a batch of vectors v, use:
        v_rot = tf.matmul(r, v[..., tf.newaxis])[..., 0]

    NOTE: The computed rotation matrix is its own rotational inverse, so applying it a second time
          will return your rotated vector (or batch) back to its original orientation.
    """
    orientation = tf.convert_to_tensor(orientation)
    channels = tf.shape(orientation)[-1]
    if initial_orientation is None:
        initial_orientation = tf.concat([[1.0], tf.zeros(channels - 1, orientation.dtype)], axis=0)
    else:
        initial_orientation = tf.convert_to_tensor(initial_orientation)
    o_sum = orientation + initial_orientation
    row = o_sum[..., tf.newaxis, :]
    col = o_sum[..., :, tf.newaxis]
    prod = row * col
    prod_magnitude = dot(o_sum, o_sum)[..., tf.newaxis, tf.newaxis]
    return 2.0 * prod / prod_magnitude - tf.eye(channels, dtype=o_sum.dtype)


@tf.function
def uniform_random_rotation_matrix(d, batch_shape=()) -> tf.Tensor:
    """Generate a (batch of) uniformly distributed random rotation matrices of dimension d."""
    # Sample uniformly distributed vectors on the unit hypersphere in d dimensions.
    if tfp is None:
        raise ImportError("This functionality requires tensorflow_probability to be installed.")
    orientations = tfp.random.spherical_uniform(batch_shape, d)
    return unit_vector_to_rotation_matrix(orientations)
