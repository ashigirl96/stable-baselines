from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import functools
from stable_baselines.a2c_sil.utils import conv_to_fc, linear


def encode_cnn(scaled_images, is_training=True, reuse=False, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  CNN from Count based with Successor Feature paper.
  :param scaled_images: (TensorFlow Tensor) Image input placeholder
  :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
  :return: (TensorFlow Tensor) The CNN output layer
  """
  # None, 84, 84, 4
  assert scaled_images.shape.as_list() == [5, 84, 84, 4], scaled_images.shape

  batch_norm_fn = functools.partial(slim.batch_norm,
                                    scale=True,
                                    updates_collections=None,
                                    is_training=is_training)
  weights_initializer = tf.initializers.orthogonal(np.sqrt(2))

  with tf.variable_scope('encode', reuse=reuse):
    x = slim.conv2d(scaled_images, 64, 6, 2,
                    'valid',
                    activation_fn=tf.nn.leaky_relu,
                    normalizer_fn=batch_norm_fn,
                    weights_initializer=weights_initializer,
                    scope='conv1')

    assert x.shape.as_list() == [5, 40, 40, 64], x.shape

    x = slim.conv2d(x, 64, 6, 2,
                    'same',
                    activation_fn=tf.nn.leaky_relu,
                    normalizer_fn=batch_norm_fn,
                    weights_initializer=weights_initializer,
                    scope='conv2')
    assert x.shape.as_list() == [5, 20, 20, 64], x.shape

    x = slim.conv2d(x, 64, 6, 2,
                    'same',
                    activation_fn=tf.nn.leaky_relu,
                    normalizer_fn=batch_norm_fn,
                    weights_initializer=weights_initializer,
                    scope='conv3')
    assert x.shape.as_list() == [5, 10, 10, 64], x.shape

    x = conv_to_fc(x)
    assert x.shape.as_list() == [5, 10 * 10 * 64], x.shape

    phi_mu = tf.layers.dense(x, 1024,
                             kernel_initializer=weights_initializer,
                             name='phi_mu_')
    phi_sigma_sq = tf.layers.dense(x, 1024,
                                   kernel_initializer=weights_initializer,
                                   name='phi_sigma_sq_')
  return phi_mu, phi_sigma_sq


def decode_cnn(hidden: tf.Tensor, is_training=True, reuse=False, *kwargs) -> tf.Tensor:
  batch_norm_fn = functools.partial(slim.batch_norm,
                                    scale=True,
                                    updates_collections=None,
                                    is_training=is_training)
  weights_initializer = tf.initializers.orthogonal(np.sqrt(2))

  assert hidden.shape.as_list() == [5, 1024]
  relu = tf.nn.relu
  with tf.variable_scope('decode', reuse=reuse):
    x = relu(linear(hidden, 'fc1', n_hidden=2048, init_scale=np.sqrt(2)))  # None, 2048
    # assert layer_2.shape.as_list() == [None, 20, 20, 64], layer_2.shape
    # <Multiply action>
    x = relu(linear(x, 'fc2', n_hidden=1024, init_scale=np.sqrt(2)))  # None, 1024
    assert x.shape.as_list() == [5, 1024], x.shape

    x = relu(linear(x, 'fc3', n_hidden=6400, init_scale=np.sqrt(2)))  # 5, 6400
    assert x.shape.as_list() == [5, 6400], x.shape

    x = tf.reshape(x, (-1, 10, 10, 64))  # None, 10, 10, 64
    assert x.shape.as_list() == [5, 10, 10, 64], x.shape

    x = slim.conv2d_transpose(x, 64, 6, 2,
                              'same',
                              activation_fn=tf.nn.relu,
                              normalizer_fn=batch_norm_fn,
                              weights_initializer=weights_initializer,
                              scope='conv_transpose1')
    assert x.shape.as_list() == [5, 20, 20, 64], x.shape

    x = slim.conv2d_transpose(x, 64, 6, 2,
                              'same',
                              activation_fn=tf.nn.relu,
                              normalizer_fn=batch_norm_fn,
                              weights_initializer=weights_initializer,
                              scope='conv_transpose2')
    assert x.shape.as_list() == [5, 40, 40, 64], x.shape

    x = slim.conv2d_transpose(x, 4, 6, 2,
                              'valid',
                              activation_fn=tf.tanh,
                              normalizer_fn=None,
                              weights_initializer=weights_initializer,
                              scope='conv_transpose3')
    assert x.shape.as_list() == [5, 84, 84, 4]
  return x
