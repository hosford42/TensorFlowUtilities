import math
from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.types.core import TensorLike

try:
    import tensorflow_probability as tfp
except ImportError:
    tfp = None


@tf.function
def dot(x, y, axis=-1) -> tf.Tensor:
    """Vector dot product along the given axis."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    return tf.reduce_sum(x * y, axis=axis)


@tf.function
def transpose_matrix(m) -> tf.Tensor:
    m = tf.convert_to_tensor(m)
    rank = tf.rank(m)
    tf.assert_greater(rank, 1)
    perm = tf.concat([tf.range(rank - 2), [rank - 1, rank - 2]], axis=-1)
    return tf.transpose(m, perm)


@tf.function
def matrix_eye_like(m) -> tf.Tensor:
    m = tf.convert_to_tensor(m)
    rank = tf.rank(m)
    tf.assert_greater(rank, 1)
    shape = tf.shape(m)
    return tf.eye(shape[-2], shape[-1], batch_shape=shape[:-2])


@tf.function
def recompose_cholesky(lower) -> tf.Tensor:
    """Given the lower triangular cholesky decomposition of a matrix, compute the original matrix
    that was factored and return it."""
    lower = tf.convert_to_tensor(lower)
    return tf.matmul(lower, lower, transpose_b=True)


@tf.function
def stddev_to_flat(stddev, multivariate=True) -> tf.Tensor:
    """Convert a lower triangular matrix of shape (N, N) to a vector of shape
    (N * (N + 1) // 2,)."""
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
def sample_stddev_matrix(data, sample_axis=0, keepdims=False, tolerance=0.000001) -> tf.Tensor:
    """Compute the lower-triangular stddev matrix of a set of sample data.
    NOTE: This function replaces tfp.stats.cholesky_covariance, which raises an exception
          if the covariance matrix isn't invertible instead of using nans."""
    if tfp is None:
        raise ImportError("This functionality requires tensorflow_probability to be installed.")
    covariance = tfp.stats.covariance(data, sample_axis=sample_axis, keepdims=keepdims)
    broadcast_eye = tf.reshape(tf.eye(tf.shape(covariance)[-1]),
                               tf.concat([tf.ones_like(tf.shape(covariance)[:sample_axis+1]),
                                          tf.shape(covariance)[sample_axis+1:]],
                                         axis=-1))
    return variance_to_stddev(covariance + tolerance * broadcast_eye)


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


@tf.function
def mahalanobis_sq(points, stddev) -> tf.Tensor:
    """Compute the square of the mahalanobis distance of a batch of points from the mean w.r.t. a 
    zero-mean distribution with the given lower-triangular standard deviation matrix."""
    var_inv = tf.linalg.pinv(stddev_to_variance(stddev))
    result = tf.matmul(points[..., tf.newaxis, :],
                       tf.matmul(var_inv[tf.newaxis],
                                 points[..., :, tf.newaxis]))
    return tf.squeeze(result, axis=(-2, -1))


@tf.function
def _sample_multivariate_normal(mean, stddev, batch_shape=(), seed=None) -> tf.Tensor:
    mean = tf.convert_to_tensor(mean)
    stddev = tf.convert_to_tensor(stddev)
    dims = tf.shape(mean)[-1]
    param_batch_shape = tf.shape(mean)[:-1]
    tf.assert_rank(stddev, tf.rank(mean) + 1)
    tf.assert_equal(tf.shape(stddev)[-1], dims)
    tf.assert_equal(tf.shape(stddev)[-2], dims)
    tf.assert_equal(param_batch_shape, tf.shape(stddev)[:-2])
    batch_shape = tf.convert_to_tensor(batch_shape, dtype=dims.dtype)
    broadcast_mean_shape = tf.concat([tf.ones_like(batch_shape), tf.shape(mean)], axis=-1)
    mean = tf.reshape(mean, broadcast_mean_shape)
    broadcast_stddev_shape = tf.concat([tf.ones_like(batch_shape), tf.shape(stddev)], axis=-1)
    stddev = tf.reshape(stddev, broadcast_stddev_shape)
    sample_shape = tf.concat([batch_shape, param_batch_shape, [dims]], axis=-1)
    standard_normal_sample = tf.random.normal(sample_shape, seed=seed)
    return mean + tf.linalg.LinearOperatorLowerTriangular(stddev).matvec(standard_normal_sample)


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

    sample = tf.stop_gradient(_sample_multivariate_normal(mean, stddev, batch_shape, seed))

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
def generalized_variance(stddev) -> tf.Tensor:
    """Compute the generalized variance (GV) of a distribution, given its lower-triangular
    standard deviation matrix. 
    (See https://faculty.ucr.edu/~ashis/publication/publications/ESS6053.PDF)
    """
    stddev = tf.convert_to_tensor(stddev)
    det = tf.linalg.det(stddev_to_variance(stddev))
    # Insure it's non-negative. Floating point errors can result in it being very slightly negative.
    det = det - tf.minimum(0.0, tf.stop_gradient(det))
    return det


@tf.function
def standardized_generalized_variance(stddev) -> tf.Tensor:
    """Compute the standardized generalized variance (SGV) of a distribution, given its 
    lower-triangular standard deviation matrix. 
    (See https://faculty.ucr.edu/~ashis/publication/publications/ESS6053.PDF)
    """
    stddev = tf.convert_to_tensor(stddev)
    gv = generalized_variance(stddev)
    return tf.pow(gv, 1.0 / tf.cast(tf.shape(stddev)[-1], gv.dtype))


@tf.function
def sum_of_independent_gaussians(mean1, stddev1, mean2, stddev2,
                                 multivariate=True) -> Tuple[tf.Tensor, tf.Tensor]:
    """The sum of two independent gaussian variables follows a gaussian distribution with the mean
    and variance equal to the sums of the means and variances, respectively, of the variables.
    Compute the mean and lower-triangular standard deviation matrix of the gaussian variable that 
    results from adding two gaussian variables with the indicated statistics."""
    mean1 = tf.convert_to_tensor(mean1)
    stddev1 = tf.convert_to_tensor(stddev1)
    mean2 = tf.convert_to_tensor(mean2)
    stddev2 = tf.convert_to_tensor(stddev2)
    variance1 = stddev_to_variance(stddev1, multivariate=multivariate)
    variance2 = stddev_to_variance(stddev2, multivariate=multivariate)
    mean = mean1 + mean2
    variance = variance1 + variance2
    return mean, variance_to_stddev(variance, multivariate=multivariate)


@tf.function
def gaussian_pdf_product(mean1, stddev1, mean2, stddev2,
                         multivariate=True) -> Tuple[tf.Tensor, tf.Tensor]:
    """The product of the PDFs of two gaussian distributions is proportionate to a gaussian PDF
    whose mean and variance are weighted averages of the means and variances, respectively, of the
    original PDFs. Compute the mean and standard deviation of the gaussian PDF that is proportionate
    to the product of the provided gaussian PDFs."""
    mean1 = tf.convert_to_tensor(mean1)
    stddev1 = tf.convert_to_tensor(stddev1)
    mean2 = tf.convert_to_tensor(mean2)
    stddev2 = tf.convert_to_tensor(stddev2)
    variance1 = stddev_to_variance(stddev1, multivariate=multivariate)
    variance2 = stddev_to_variance(stddev2, multivariate=multivariate)
    if multivariate:
        denominator = tf.linalg.pinv(dynamically_condition_covariance_matrix(variance1 + variance2))
        mean = (tf.matmul(variance2, tf.matmul(denominator, mean1[..., tf.newaxis])) +
                tf.matmul(variance1, tf.matmul(denominator, mean2[..., tf.newaxis])))
        mean = tf.squeeze(mean, axis=-1)
        variance = tf.matmul(variance1, tf.matmul(denominator, variance2))
        variance = dynamically_condition_covariance_matrix(variance)
    else:
        denominator = variance1 + variance2
        mean = (variance2 * mean1 + variance1 * mean2) / denominator
        variance = variance1 * variance2 / denominator
        tf.assert_rank(mean, tf.rank(variance))
    return mean, variance_to_stddev(variance, multivariate=multivariate)


@tf.function
def batched_gaussian_pdf_product(term_means, term_stddevs):
    """Given a batch of sequences of multivariate gaussian means and standard deviations, compute
    the gaussian pdf product of each sequence in the batch. The means and standard deviations may be
    stored in a ragged tensor in the case where the sequences have different lengths."""
    if isinstance(term_means, tf.RaggedTensor):
        assert isinstance(term_stddevs, tf.RaggedTensor)
        term_counts = term_means.row_lengths(axis=-2)
        max_terms = term_means.bounding_shape(axis=tf.rank(term_means) - 2)  # Negatives don't work
        tf.assert_equal(term_counts, term_stddevs.row_lengths(axis=-3))
        tf.assert_equal(max_terms, term_stddevs.bounding_shape(axis=tf.rank(term_stddevs) - 3))
        term_means = term_means.to_tensor(math.nan)
        term_stddevs = term_stddevs.to_tensor(math.nan)
        tf.assert_equal(tf.shape(term_counts)[-1], tf.shape(term_means)[-3])
    else:
        assert not isinstance(term_stddevs, tf.RaggedTensor)
        term_means = tf.convert_to_tensor(term_means)
        term_stddevs = tf.convert_to_tensor(term_stddevs)
        max_terms = tf.shape(term_means)[-2]
        term_counts = tf.reshape(max_terms, (1,))
        tf.assert_equal(max_terms, tf.shape(term_stddevs)[-3])
    tf.assert_greater(term_counts, tf.cast(0, term_counts.dtype),
                      "Gaussian product is ill-defined for zero terms.")

    product_mean = term_means[..., 0, :]
    product_stddev = term_stddevs[..., 0, :, :]
    for counter in tf.range(1, max_terms):
        component_mean = term_means[..., counter, :]
        component_stddev = term_stddevs[..., counter, :, :]
        maybe_new_mean, maybe_new_stddev = gaussian_pdf_product(product_mean, product_stddev,
                                                                component_mean, component_stddev)
        new_observation = counter < term_counts
        product_mean = tf.where(new_observation[:, tf.newaxis], maybe_new_mean, product_mean)
        product_stddev = tf.where(new_observation[:, tf.newaxis, tf.newaxis],
                                  maybe_new_stddev, product_stddev)
    return product_mean, product_stddev


@tf.function(experimental_relax_shapes=True)
def leave_one_out_batched_gaussian_pdf_product(term_means, term_stddevs):
    """Given a batch of sequences of gaussian means and standard deviations, compute the N gaussian
    pdf products of each sequence in the batch that result from leaving each entry in the sequence
    out. The means and standard deviations may be stored in a ragged tensor in the case where the
    sequences have different lengths.

    This works like batched_gaussian_pdf_product, except that instead of just one product per
    sequence, N distinct products are computed, each with a different term from the sequence left
    out, which is useful for generating training targets that don't depend on the output of the
    model being trained."""

    # Input shapes:
    #   term_means: batch_shape + (max_terms, channels)
    #   term_stddevs: batch_shape + (max_terms, channels, channels)
    # Output shapes:
    #   product_mean: batch_shape + (max_terms, channels)
    #   product_stddev: batch_shape + (max_terms, channels, channels)

    if isinstance(term_means, tf.RaggedTensor):
        assert isinstance(term_stddevs, tf.RaggedTensor)
        term_counts = term_means.row_lengths(axis=-2)
        max_terms = term_means.bounding_shape(axis=tf.rank(term_means) - 2)  # Negatives don't work
        tf.assert_equal(term_counts, term_stddevs.row_lengths(axis=-3))
        tf.assert_equal(max_terms, term_stddevs.bounding_shape(axis=tf.rank(term_stddevs) - 3))
        term_means = term_means.to_tensor(math.nan)
        term_stddevs = term_stddevs.to_tensor(math.nan)
        tf.assert_equal(tf.shape(term_counts)[-1], tf.shape(term_means)[-3])
    else:
        assert not isinstance(term_stddevs, tf.RaggedTensor)
        term_means = tf.convert_to_tensor(term_means)
        term_stddevs = tf.convert_to_tensor(term_stddevs)
        max_terms = tf.shape(term_means)[-2]
        term_counts = tf.reshape(max_terms, (1,))
        tf.assert_equal(max_terms, tf.shape(term_stddevs)[-3])
    tf.assert_greater(term_counts, tf.cast(1, term_counts.dtype),
                      "Leave-one-out gaussian product is ill-defined for less than 2 terms.")

    batch_shape = tf.shape(term_means)[:-2]
    batch_broadcast_shape = tf.ones_like(batch_shape)
    channels = tf.shape(term_means)[-1]

    tf.assert_rank(term_counts, 1)
    term_counts = tf.reshape(term_counts,
                             tf.concat([batch_broadcast_shape[:-1],
                                        tf.shape(term_counts)[-1:],
                                        [1]],
                                       axis=-1))

    product_defined_shape = tf.concat([batch_shape, [max_terms]], axis=-1)
    product_mean_shape = tf.concat([batch_shape, [max_terms, channels]], axis=-1)
    product_stddev_shape = tf.concat([batch_shape, [max_terms, channels, channels]], axis=-1)

    product_defined = tf.zeros(product_defined_shape, dtype=tf.bool)
    product_mean = tf.zeros(product_mean_shape, dtype=term_means.dtype)
    product_stddev = tf.zeros(product_stddev_shape, dtype=term_stddevs.dtype)

    for counter in tf.range(max_terms):  # For each term in the group
        term_mean = term_means[..., counter:counter + 1, :]
        term_stddev = term_stddevs[..., counter:counter + 1, :, :]

        maybe_new_mean, maybe_new_stddev = gaussian_pdf_product(product_mean, product_stddev,
                                                                term_mean, term_stddev)

        # Is the current term we are looking at within the bounds of the group?
        new_observation = counter < term_counts

        # Is the current term we are looking at the same term as the one we are computing the
        # product for?
        non_self = tf.reshape(counter != tf.range(max_terms),
                              tf.concat([batch_broadcast_shape, [max_terms]], axis=-1))

        keep_product = new_observation & non_self

        product_mean = tf.where(keep_product[..., tf.newaxis], maybe_new_mean, product_mean)
        product_stddev = tf.where(keep_product[..., tf.newaxis, tf.newaxis],
                                  maybe_new_stddev, product_stddev)

        product_mean = tf.where(product_defined[..., tf.newaxis], product_mean, term_mean)
        product_stddev = tf.where(product_defined[..., tf.newaxis, tf.newaxis],
                                  product_stddev, term_stddev)

        product_defined = product_defined | keep_product

    tf.assert_equal(tf.shape(product_mean), tf.shape(term_means))
    tf.assert_equal(tf.shape(product_stddev), tf.shape(term_stddevs))

    return product_mean, product_stddev


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


@tf.function
def randomly_rotate_batch(x) -> tf.Tensor:
    """Apply the same uniformly random rotation about the origin to all the vectors in the batch."""
    x = tf.convert_to_tensor(x)
    channels = tf.shape(x)[-1]
    rotation = uniform_random_rotation_matrix(channels)
    rotation = tf.reshape(rotation, tf.concat([tf.ones(tf.rank(x) - 1, dtype=tf.shape(x).dtype),
                                               [channels, channels]], axis=-1))
    return tf.matmul(rotation, x[..., tf.newaxis])[..., 0]


@tf.function(experimental_relax_shapes=True)
def sliding_window(x, width, axis=-1):
    """Replace the axis dimension of size s with 2 dimensions of shape (s - w + 1, w), where
    each of the s - w + 1 indices of the first new dimension corresponds to a consecutive window of
    with w from the original axis. For example, if x has shape (6, 7) and the window width is 3, the
    returned shape will be (6, 7 - 3 + 1, 3) == (6, 5, 3).

    Let y = sliding_window(x, w). Then for each i in range(y.shape[0]), each j in range(y.shape[1]),
    and each k in range(y.shape[2]), y[i, j, k] == x[i, j + k] will evaluate to True."""
    x = tf.convert_to_tensor(x)
    rank = tf.rank(x)
    axis = axis if axis >= 0 else axis + rank
    tf.assert_greater(axis + 1, 0)
    tf.assert_less(axis, rank)
    leading_shape = tf.shape(x)[:axis]
    trailing_shape = tf.shape(x)[axis + 1:]
    slice_count = tf.shape(x)[axis] - width + 1
    slices = tf.TensorArray(x.dtype, slice_count)
    for offset in tf.range(slice_count):
        start = tf.concat([tf.zeros_like(leading_shape), [offset], tf.zeros_like(trailing_shape)],
                          axis=-1)
        size = tf.concat([leading_shape, [width], trailing_shape], axis=-1)
        window = tf.slice(x, start, size)
        slices = slices.write(offset, window)
    stacked = slices.stack()
    permutation = tf.concat([tf.range(1, axis + 1), [0], tf.range(axis + 1, rank + 1)], axis=-1)
    return tf.transpose(stacked, permutation)


@tf.function
def complex_matrix_pow(m, p) -> tf.Tensor:
    m = tf.convert_to_tensor(m)
    p = tf.convert_to_tensor(p)
    eig_vals, eig_vects = tf.eig(m)
    eig_vects_inv = tf.linalg.inv(eig_vects)
    diagonal = tf.linalg.set_diag(tf.zeros_like(m, dtype=eig_vals.dtype), eig_vals)
    return tf.matmul(eig_vects, tf.matmul(tf.pow(diagonal, tf.cast(p, eig_vals.dtype)),
                                          eig_vects_inv))


@tf.function
def real_matrix_pow(m, p, tolerance=0.0001) -> tf.Tensor:
    complex_result = complex_matrix_pow(m, p)
    return tf.where(tf.reduce_all(tf.abs(tf.math.imag(complex_result)) <= tolerance,
                                  axis=(-2, -1),
                                  keepdims=True),
                    tf.cast(complex_result, tf.float32),
                    math.nan)


@tf.custom_gradient
def invert_gradient(x) -> tf.Tensor:
    """Invert the gradient of the given tensor."""
    x = tf.convert_to_tensor(x)

    def grad(upstream):
        return -upstream

    # noinspection PyTypeChecker
    return x, grad


@tf.custom_gradient
def clip_grad_by_norm(x, clip_norm) -> tf.Tensor:
    """Restrict the magnitude (norm) of a tensor's gradient to an upper bound."""
    def grad(upstream):
        return tf.clip_by_norm(upstream, clip_norm), None

    # noinspection PyTypeChecker
    return x, grad


@tf.custom_gradient
def zero_nan_grad(x) -> tf.Tensor:
    """Replace NaN values in a tensor's gradient with zeros to prevent ruining a partially trained
    model due to undefined gradients."""
    def grad(upstream):
        return tf.where(tf.math.is_nan(upstream), 0.0, upstream)

    # noinspection PyTypeChecker
    return x, grad


@tf.function
def unreported_loss(loss: tf.Tensor) -> tf.Tensor:
    """Hide a regularization loss, so it isn't reported by Keras in the total loss on each epoch."""
    return loss - tf.stop_gradient(loss)


@tf.function
def safe_mse(targets, predictions, name=None) -> tf.Tensor:
    """Mean squared error, ignoring rows where the loss is undefined (nan)."""
    targets = tf.convert_to_tensor(targets)
    predictions = tf.convert_to_tensor(predictions)
    error = tf.square(tf.stop_gradient(targets) - predictions)
    safe_error = tf.where(tf.math.is_finite(error), error, 0.0)
    return tf.reduce_mean(safe_error, name=name)


@tf.function
def find_permutation_cycles(permutation: TensorLike, max_length: int = None) -> tf.Tensor:
    permutation = tf.convert_to_tensor(permutation)
    tf.assert_rank(permutation, 1)
    tf.assert_greater(tf.size(permutation), 0)
    if max_length is None:
        max_length = tf.size(permutation)
    else:
        max_length = tf.convert_to_tensor(max_length, tf.int32)
    cycles = permutation
    for length in range(1, max_length + 1):
        permuted_cycles = tf.gather(cycles, permutation)
        cycles = tf.minimum(cycles, permuted_cycles)
        if tf.reduce_all(cycles == permuted_cycles):
            break
    return cycles


@tf.function
def swap(vector: TensorLike, index1: int, index2: int) -> tf.Tensor:
    vector = tf.convert_to_tensor(vector)
    index1 = tf.convert_to_tensor(index1, tf.int32)
    tf.assert_rank(index1, 0)
    index2 = tf.convert_to_tensor(index2, tf.int32)
    tf.assert_rank(index2, 0)
    if index1 == index2:
        return vector
    if index1 < index2:
        lower = index1
        upper = index2
    else:
        lower = index2
        upper = index1
    return tf.concat([
        vector[..., :lower],
        vector[..., upper:upper + 1],
        vector[..., lower + 1:upper],
        vector[..., lower:lower + 1],
        vector[..., upper + 1:]
    ], axis=-1)


@tf.function
def random_permutation_with_cycles(cycle_lengths: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    """Given a set of cycle lengths, generate a single random permutation that contains interleaved
    cycles of the given lengths. Returns two int32 vectors, both with length equal to the sum of the
    cycle lengths. The first is the generated permutation. The second is a mapping from each index
    to the cycle it belongs to -- an integer index into the original cycle_lengths vector."""
    cycle_lengths = tf.convert_to_tensor(cycle_lengths, dtype=tf.int32)
    tf.assert_rank(cycle_lengths, 1)
    tf.assert_greater(cycle_lengths, 0)

    cycle_count = tf.size(cycle_lengths)
    result_size = tf.reduce_sum(cycle_lengths)
    cycle_starts = tf.cumsum(cycle_lengths, exclusive=True)

    # Maps from each index to the cycle it belongs to, where cycles are numbered according to
    # their positions in the original cycle_lengths vector.
    partition = tf.TensorArray(tf.int32, size=cycle_count, dynamic_size=False,
                               clear_after_read=True, infer_shape=False,
                               element_shape=tf.TensorShape([None]))
    indices = tf.TensorArray(tf.int32, size=cycle_count, dynamic_size=False,
                             clear_after_read=True, infer_shape=False,
                             element_shape=tf.TensorShape([None]))
    for index in tf.range(cycle_count):
        tf.autograph.experimental.set_loop_options(parallel_iterations=True)
        cycle_labels = tf.repeat(index, cycle_lengths[index])
        partition = partition.write(index, cycle_labels)
        cycle_indices = cycle_starts[index] + tf.concat([tf.range(1, cycle_lengths[index]), [0]],
                                                        axis=0)
        indices = indices.write(index, cycle_indices)
    partition = partition.concat()
    tf.assert_equal(tf.shape(partition), (result_size,))
    indices = indices.concat()
    tf.assert_equal(tf.shape(indices), (result_size,))

    shuffle_order = tf.random.shuffle(tf.range(result_size))
    partition = tf.gather(partition, shuffle_order)
    indices = tf.gather(indices, shuffle_order)
    partition_inv = tf.argsort(indices)
    permutation_inv = tf.gather(partition_inv, shuffle_order)
    permutation = tf.argsort(permutation_inv)

    tf.assert_equal(tf.shape(permutation), (result_size,))
    tf.assert_equal(tf.shape(partition), (result_size,))
    tf.assert_equal(partition, tf.gather(partition, permutation))
    return permutation, partition


@tf.function
def partitioned_mean(x: TensorLike, partition: TensorLike) -> tf.Tensor:
    """Compute the mean of the values of x appearing in each partition, and return a tensor with
    the same shape and dtype as x, but with each value replaced by the mean of the partition it
    appears within. The partition must be an integer tensor, and x must be a floating-point
    tensor. Both must have the same shape."""
    x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
    partition = tf.convert_to_tensor(partition, dtype_hint=tf.int32)
    assert x.dtype.is_floating
    assert partition.dtype.is_integer
    tf.assert_equal(tf.shape(x), tf.shape(partition))
    shape = tf.shape(x)
    x = tf.reshape(x, (tf.size(x),))
    partition = tf.reshape(partition, (tf.size(partition),))
    partition_labels = tf.unique(partition).y
    indices = tf.TensorArray(tf.int64, size=tf.size(partition_labels), dynamic_size=False,
                             clear_after_read=True, element_shape=(None,))
    means = tf.TensorArray(x.dtype, size=tf.size(partition_labels), dynamic_size=False,
                           clear_after_read=True, element_shape=(None,))
    for label_counter in tf.range(tf.size(partition_labels)):
        tf.autograph.experimental.set_loop_options(parallel_iterations=True)
        label = partition_labels[label_counter]
        label_indices = tf.squeeze(tf.where(partition == label), axis=1)
        label_values = tf.gather(x, label_indices)
        label_means = tf.repeat(tf.reduce_mean(label_values), tf.size(label_indices))
        indices = indices.write(label_counter, label_indices)
        means = means.write(label_counter, label_means)
    indices = indices.concat()
    means = means.concat()
    result = tf.gather(means, tf.argsort(indices))
    return tf.reshape(result, shape)


@tf.function
def softminus(x: TensorLike) -> Tensor:
    """Like softplus, but negative."""
    x = tf.convert_to_tensor(x)
    return -tf.nn.softplus(-x)
