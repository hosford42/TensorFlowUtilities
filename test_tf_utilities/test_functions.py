from tf_utilities.functions import *


def test_clip_grad_by_norm():
    x = tf.Variable([0.0, 1.0, 2.0])

    with tf.GradientTape() as tape:
        # noinspection PyTypeChecker
        loss = tf.reduce_sum(tf.square(100.0 - x))
    actual_gradient = tape.gradient(loss, x)
    tf.assert_greater(tf.norm(actual_gradient), 2.0)
    with tf.GradientTape() as tape:
        # noinspection PyTypeChecker
        loss = tf.reduce_sum(tf.square(100.0 - clip_grad_by_norm(x, 2.0)))
    clipped_gradient = tape.gradient(loss, x)
    tf.debugging.assert_near(tf.norm(clipped_gradient), 2.0)
    tf.debugging.assert_near(clipped_gradient, 2.0 * actual_gradient / tf.norm(actual_gradient))

    with tf.GradientTape() as tape:
        # noinspection PyTypeChecker
        loss = tf.reduce_sum(tf.square(2.0 - x))
    actual_gradient = tape.gradient(loss, x)
    tf.assert_less(tf.norm(actual_gradient), 100.0)
    with tf.GradientTape() as tape:
        # noinspection PyTypeChecker
        loss = tf.reduce_sum(tf.square(2.0 - clip_grad_by_norm(x, 100.0)))
    clipped_gradient = tape.gradient(loss, x)
    tf.debugging.assert_less(tf.norm(clipped_gradient), 100.0)
    tf.assert_equal(clipped_gradient, actual_gradient)


def test_zero_nan_grad():
    x = tf.Variable([0.0, 1.0, 2.0])

    with tf.GradientTape() as tape:
        # NOTE: Intentionally no reduce_sum or reduce_mean
        loss = tf.square(x / x)
    actual_gradient = tape.gradient(loss, x)
    assert tf.reduce_any(tf.math.is_nan(actual_gradient))
    assert tf.reduce_any(tf.logical_not(tf.math.is_nan(actual_gradient)))

    with tf.GradientTape() as tape:
        # NOTE: Intentionally no reduce_sum or reduce_mean
        loss = tf.square(zero_nan_grad(x) / zero_nan_grad(x))
    repaired_gradient = tape.gradient(loss, x)
    tf.debugging.assert_all_finite(repaired_gradient, "oops")

    nan_mask = tf.math.is_nan(actual_gradient)
    tf.assert_equal(tf.cast(nan_mask, tf.float32) * repaired_gradient, 0.0)
    assert tf.reduce_all(nan_mask | (repaired_gradient == actual_gradient))


def test_unreported_loss():
    x = tf.Variable(0.0)

    with tf.GradientTape() as tape:
        # noinspection PyTypeChecker
        loss = tf.square(100.0 - x)
    tf.assert_greater(loss, 0.0)
    actual_gradient = tape.gradient(loss, x)

    with tf.GradientTape() as tape:
        # noinspection PyTypeChecker
        loss = unreported_loss(tf.square(100.0 - x))
    tf.assert_equal(loss, 0.0)
    injected_gradient = tape.gradient(loss, x)
    tf.assert_equal(actual_gradient, injected_gradient)


def test_dot():
    tf.random.set_seed(0)
    x = tf.random.uniform((3, 10, 5))
    y = tf.random.uniform((3, 10, 5))
    z = dot(x, y, axis=-2)
    tf.assert_rank(z, 2)
    tf.assert_equal(z.shape, (3, 5))
    tf.assert_equal(z, tf.reduce_sum(x * y, axis=-2))


def test_recompose_cholesky():
    tf.random.set_seed(0)
    x = tf.random.uniform((3, 3)) * 2.0 - 1.0 + 2.0 * tf.eye(3)
    x = x + tf.transpose(x)
    c = tf.linalg.cholesky(x)
    tf.debugging.assert_all_finite(c, "oops")
    tf.debugging.assert_near(x, recompose_cholesky(c))


def test_flat_to_stddev():
    tf.random.set_seed(0)

    x = tf.random.uniform((3,)) * 2.0 - 1.0
    s = flat_to_stddev(x, multivariate=False, min_stddev=0.000001)
    tf.assert_equal(s.shape, (3,))
    tf.assert_greater(s, 0.000001)

    x = tf.random.uniform((3 * (3 + 1) // 2,)) * 2.0 - 1.0
    s = flat_to_stddev(x, multivariate=True, min_stddev=0.000001)
    tf.assert_equal(s.shape, (3, 3))
    tf.assert_greater(tf.linalg.diag_part(s), 0.000001)


def test_flat_to_mean_and_stddev():
    tf.random.set_seed(0)

    x = tf.random.uniform((3 + 3,)) * 2.0 - 1.0
    m, s = flat_to_mean_and_stddev(x, 3, multivariate=False, min_stddev=0.000001)
    tf.assert_equal(m.shape, (3,))
    tf.assert_equal(m, x[:3])
    tf.assert_equal(s.shape, (3,))
    tf.assert_greater(s, 0.000001)

    x = tf.random.uniform((3 + 3 * (3 + 1) // 2,)) * 2.0 - 1.0
    m, s = flat_to_mean_and_stddev(x, 3, multivariate=True, min_stddev=0.000001)
    tf.assert_equal(m.shape, (3,))
    tf.assert_equal(m, x[:3])
    tf.assert_equal(s.shape, (3, 3))
    tf.assert_greater(tf.linalg.diag_part(s), 0.000001)


def test_stddev_to_variance():
    tf.random.set_seed(0)

    x = tf.random.uniform((3,)) * 2.0 - 1.0
    s = flat_to_stddev(x, multivariate=False, min_stddev=0.000001)
    v = stddev_to_variance(s, multivariate=False)
    tf.assert_equal(s.shape, v.shape)
    tf.assert_equal(s, tf.sqrt(v))

    x = tf.random.uniform((3 * (3 + 1) // 2,)) * 2.0 - 1.0
    s = flat_to_stddev(x, multivariate=True, min_stddev=0.000001)
    v = stddev_to_variance(s, multivariate=True)
    tf.assert_equal(s.shape, v.shape)
    tf.debugging.assert_near(s, tf.linalg.cholesky(v))


def test_variance_to_stddev():
    tf.random.set_seed(0)

    x = tf.random.uniform((3,)) * 2.0 - 1.0
    s = flat_to_stddev(x, multivariate=False, min_stddev=0.000001)
    v = stddev_to_variance(s, multivariate=False)
    s2 = variance_to_stddev(v, multivariate=False)
    tf.assert_equal(s2.shape, v.shape)
    tf.assert_equal(s2, tf.sqrt(v))

    x = tf.random.uniform((3 * (3 + 1) // 2,)) * 2.0 - 1.0
    s = flat_to_stddev(x, multivariate=True, min_stddev=0.000001)
    v = stddev_to_variance(s, multivariate=True)
    s2 = variance_to_stddev(v, multivariate=True)
    tf.assert_equal(s2.shape, v.shape)
    tf.debugging.assert_near(s2, tf.linalg.cholesky(v))


def test_sum_of_independent_gaussians():
    tf.random.set_seed(0)

    m1 = tf.random.uniform((5,)) * 2.0 - 1.0
    s1 = tf.random.uniform((5,)) + 1.0
    m2 = tf.random.uniform((5,)) * 2.0 - 1.0
    s2 = tf.random.uniform((5,)) + 1.0
    m3, s3 = sum_of_independent_gaussians(m1, s1, m2, s2, multivariate=False)
    tf.assert_equal(m3, m1 + m2)
    tf.debugging.assert_near(tf.square(s3), tf.square(s1) + tf.square(s2))

    m1 = tf.random.uniform((5,)) * 2.0 - 1.0
    s1 = tf.random.uniform((5, 5)) * 2.0 - 1.0
    s1 = s1 + tf.transpose(s1) + 2.0 * tf.eye(5)
    m2 = tf.random.uniform((5,)) * 2.0 - 1.0
    s2 = tf.random.uniform((5, 5)) * 2.0 - 1.0
    s2 = s2 + tf.transpose(s1) + 2.0 * tf.eye(5)
    m3, s3 = sum_of_independent_gaussians(m1, s1, m2, s2, multivariate=True)
    tf.assert_equal(m3, m1 + m2)
    tf.debugging.assert_near(recompose_cholesky(s3),
                             recompose_cholesky(s1) + recompose_cholesky(s2))


def test_gaussian_pdf_product():
    m, s = gaussian_pdf_product(0.0, 1.0 ** 0.5, 3.0, 1.0 ** 0.5, multivariate=False)
    tf.debugging.assert_near(m, (1.0 * 0.0 + 1.0 * 3.0) / (1.0 + 1.0))
    tf.debugging.assert_near(s, (1.0 * 1.0 / (1.0 + 1.0)) ** 0.5)

    m, s = gaussian_pdf_product(0.0, 1.0 ** 0.5, 3.0, 2.0 ** 0.5, multivariate=False)
    tf.debugging.assert_near(m, (2.0 * 0.0 + 1.0 * 3.0) / (1.0 + 2.0))
    tf.debugging.assert_near(s, (1.0 * 2.0 / (1.0 + 2.0)) ** 0.5)


def test_batched_gaussian_pdf_product():
    assert False  # TODO


def test_leave_one_out_batched_gaussian_pdf_product():
    assert False  # TODO


def test_unit_vector_to_rotation_matrix():
    assert False  # TODO


def test_uniform_random_rotation_matrix():
    tf.random.set_seed(0)
    rotation_matrix = uniform_random_rotation_matrix(3, (10, 5))
    tf.assert_equal(rotation_matrix.shape, (10, 5, 3, 3))
    tf.debugging.assert_near(tf.matmul(rotation_matrix, rotation_matrix),
                             tf.eye(3)[tf.newaxis, tf.newaxis])


def test_randomly_rotate_batch():
    tf.random.set_seed(0)
    data = tf.random.uniform((20, 20), -10.0, 10.0)
    magnitudes = tf.norm(data, axis=-1)
    rotated_data = randomly_rotate_batch(data)
    tf.debugging.assert_none_equal(tf.reduce_all(data == rotated_data, axis=-1), True)
    rotated_magnitudes = tf.norm(rotated_data, axis=-1)
    tf.debugging.assert_near(magnitudes, rotated_magnitudes)


def test_sliding_window():
    tf.random.set_seed(0)
    data = tf.random.uniform((11, 7, 5, 3))
    windows = sliding_window(data, 2, axis=-2)
    assert windows.shape == (11, 7, 5 - 2 + 1, 2, 3)
    for i in range(11):
        for j in range(7):
            for k in range(5 - 2 + 1):
                for l in range(2):
                    for m in range(3):
                        assert windows[i, j, k, l, m] == data[i, j, k + l, m]
