import math
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

from tf_utilities.functions import softminus

if TYPE_CHECKING:
    import keras
    from keras import layers
    from keras import optimizers
else:
    keras = tf.keras
    layers = tf.keras.layers
    optimizers = tf.keras.optimizers


Shape = tuple[int, ...]


def standard_activation(minimum: float = None, maximum: float = None,
                        bias: float = None) -> layers.Activation:
    if minimum is None:
        if maximum is None:
            if bias is None:
                bias = 0.0
            return layers.Activation(lambda x: x + bias)
        else:
            if bias is None:
                bias = 0.0
            else:
                assert bias <= maximum
                bias -= maximum
            bias_preimage = -math.log(math.exp(max(-bias, 0.000001)) - 1)
            # noinspection PyTypeChecker
            return layers.Activation(lambda x: softminus(x + bias_preimage) + maximum)
    elif maximum is None:
        if bias is None:
            bias = 0.0
        else:
            assert bias >= minimum
            bias -= minimum
        bias_preimage = math.log(math.exp(max(bias, 0.000001)) - 1)
        return layers.Activation(lambda x: tf.nn.softplus(x + bias_preimage) + minimum)
    else:
        assert maximum > minimum
        reward_scale = 0.5 * (maximum - minimum)
        reward_shift = 0.5 * (maximum + minimum)
        if bias is None:
            bias_preimage = 0.0
        else:
            assert minimum <= bias <= maximum
            bias_preimage = math.atanh(reward_shift / (reward_scale + 0.000001))
        return layers.Activation(lambda x: tf.tanh(x + bias_preimage) * reward_scale + reward_shift)


def standard_variance_activation(value_minimum: float = None, value_maximum: float = None,
                                 variance_minimum: float = 0.0,
                                 variance_bias: float = None) -> layers.Activation:
    assert variance_minimum >= 0.0
    if value_minimum is None or value_maximum is None:
        variance_maximum = None
    else:
        assert value_maximum > value_minimum
        variance_maximum = (value_maximum - value_minimum) ** 2
    return standard_activation(variance_minimum, variance_maximum, variance_bias)


def composite_activation(*sections: tuple[int, Any], name: str = None) -> layers.Activation:
    ends = np.cumsum([channels for channels, activation in sections])
    starts = np.concatenate([[0], ends], axis=0)
    activations = [keras.activations.get(activation) for channels, activation in sections]

    def f(x):
        y_sections = []
        for start, end, activation in zip(starts, ends, activations):
            x_section = x[..., start:end]
            y_sections.append(activation(x_section))
        return tf.concat(y_sections, axis=-1)

    return layers.Activation(f, name=name)


def standard_sequential_model(input_shape: Shape, output_shape: Shape, depth: int = 2,
                              hidden_activation='swish', final_activation=None,
                              name: str = None) -> keras.Sequential:
    # TODO: Support convolution. Assume all dimensions except the last one have a topological
    #       association, meaning that convolution makes sense for them because they will tend to
    #       have consistent relationships based on relative position.
    assert len(input_shape) == 1 and len(output_shape) == 1

    assert depth >= 1

    initial_channels = np.prod(input_shape)
    final_channels = np.prod(output_shape)
    channel_delta = (final_channels - initial_channels) / depth

    layer_list = [keras.Input(input_shape, name='input')]
    for layer_index in range(1, depth):
        channels = initial_channels + channel_delta * layer_index
        layer = layers.Dense(channels, activation=hidden_activation)
        layer_list.append(layer)
    if isinstance(final_activation, layers.Activation):
        layer_list.append(layers.Dense(final_channels))
        layer_list.append(final_activation)
    else:
        layer_list.append(layers.Dense(final_channels, activation=final_activation))

    return keras.Sequential(layer_list, name=name)
