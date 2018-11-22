from abc import ABC
from typing import Tuple

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, \
    seq_to_batch, lstm, conv_bn_lrelu, dconv_bn_relu, dconv
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.input import observation_input



def encode_cnn(scaled_images, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    CNN from Count based with Successor Feature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    # None, 84, 84, 4
    assert scaled_images.shape.as_list() == [None, 84, 84, 4]
    layer_1 = conv_bn_lrelu(scaled_images, 'c1', n_filters=64,
                            filter_size=6, stride=2, init_scale=np.sqrt(2))  # None, 40, 40, 64
    assert layer_1.shape.as_list() == [None, 40, 40, 64], layer_1.shape
    layer_2 = conv_bn_lrelu(layer_1, 'c2', n_filters=64,
                            filter_size=6, stride=2, init_scale=np.sqrt(2), pad='same')  # None, 20, 20, 64
    assert layer_2.shape.as_list() == [None, 20, 20, 64], layer_2.shape
    layer_3 = conv_bn_lrelu(layer_2, 'c3', n_filters=64,
                            filter_size=6, stride=2, init_scale=np.sqrt(2), pad='same')  # None, 10, 10, 64
    assert layer_3.shape.as_list() == [None, 10, 10, 64]
    layer_3 = conv_to_fc(layer_3)  # None, 10 * 10 * 64
    assert layer_3.shape.as_list() == [None, 10 * 10 * 64]
    phi_mu = linear(layer_3, 'z_mu', n_hidden=1024, init_scale=np.sqrt(2))  # None, 1024
    phi_log_sigma_sq = linear(layer_3, 'z_log_sigma_sq', n_hidden=1024, init_scale=np.sqrt(2))  # None, 1024

    return phi_mu, phi_log_sigma_sq


def decode_cnn(hidden: tf.Tensor, **kwargs) -> tf.Tensor:
    assert hidden.shape.as_list() == [None, 1024]
    relu = tf.nn.relu
    layer_1 = relu(linear(hidden, 'fc1', n_hidden=2048, init_scale=np.sqrt(2)))  # None, 2048
    # assert layer_2.shape.as_list() == [None, 20, 20, 64], layer_2.shape
    # <Multiply action>
    layer_2 = relu(linear(layer_1, 'fc2', n_hidden=1024, init_scale=np.sqrt(2)))  # None, 1024
    # assert layer_2.shape.as_list() == [None, 20, 20, 64], layer_2.shape
    layer_3 = relu(linear(layer_2, 'fc3', n_hidden=6400, init_scale=np.sqrt(2)))  # None, 6400
    # assert layer_2.shape.as_list() == [None, 20, 20, 64], layer_2.shape
    layer_3 = tf.reshape(layer_3, (-1, 10, 10, 64))  # None, 10, 10, 64
    # assert layer_2.shape.as_list() == [None, 20, 20, 64], layer_2.shape
    dconv1 = dconv_bn_relu(layer_3, 'dconv1', n_filters=64,
                           filter_size=6, stride=2, pad='same')  # None, 20, 20, 64
    assert dconv1.shape.as_list() == [None, 20, 20, 64], dconv1.shape
    dconv2 = dconv_bn_relu(dconv1, 'dconv2', n_filters=64,
                           filter_size=6, stride=2, pad='same')  # None, 40, 40, 64
    assert dconv2.shape.as_list() == [None, 40, 40, 64], dconv2.shape
    dconv3 = tf.tanh(dconv(dconv2, 'dconv3', n_filters=4,
                           filter_size=6, stride=2, pad='valid'))  # None, 84, 84, 4
    assert dconv3.shape.as_list() == [None, 84, 84, 4]
    return dconv3
