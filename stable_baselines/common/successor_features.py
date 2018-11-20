from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import functools
from stable_baselines.a2c_sil.utils import conv_to_fc, linear


def encode_cnn(scaled_images, is_training=True, reuse=False, num_batch=None,
               use_batch_norm=True, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  CNN from Count based with Successor Feature paper.
  :param scaled_images: (TensorFlow Tensor) Image input placeholder
  :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
  :return: (TensorFlow Tensor) The CNN output layer
  """
  # None, 84, 84, 4
  assert scaled_images.shape.as_list() == [num_batch, 84, 84, 4], scaled_images.shape

  batch_norm_fn = None
  if use_batch_norm:
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

    assert x.shape.as_list() == [num_batch, 40, 40, 64], x.shape

    x = slim.conv2d(x, 64, 6, 2,
                    'same',
                    activation_fn=tf.nn.leaky_relu,
                    normalizer_fn=batch_norm_fn,
                    weights_initializer=weights_initializer,
                    scope='conv2')
    assert x.shape.as_list() == [num_batch, 20, 20, 64], x.shape

    x = slim.conv2d(x, 64, 6, 2,
                    'same',
                    activation_fn=tf.nn.leaky_relu,
                    normalizer_fn=batch_norm_fn,
                    weights_initializer=weights_initializer,
                    scope='conv3')
    assert x.shape.as_list() == [num_batch, 10, 10, 64], x.shape

    x = conv_to_fc(x)
    assert x.shape.as_list() == [num_batch, 10 * 10 * 64], x.shape

    phi_mu = tf.layers.dense(x, 1024,
                             kernel_initializer=weights_initializer,
                             name='phi_mu')
    phi_sigma_sq = tf.layers.dense(x, 1024,
                                   kernel_initializer=weights_initializer,
                                   name='phi_sigma_sq')
  return phi_mu, phi_sigma_sq


def decode_cnn(hidden: tf.Tensor, action_ph, is_training=True, reuse=False, num_batch=None,
               use_batch_norm=True, *kwargs) -> tf.Tensor:
  batch_norm_fn = None
  if use_batch_norm:
    batch_norm_fn = functools.partial(slim.batch_norm,
                                      scale=True,
                                      updates_collections=None,
                                      is_training=is_training)
  weights_initializer = tf.initializers.orthogonal(np.sqrt(2))

  assert hidden.shape.as_list() == [num_batch, 1024]
  relu = tf.nn.relu
  with tf.variable_scope('decode', reuse=reuse):
    x = relu(linear(hidden, 'fc1', n_hidden=2048, init_scale=np.sqrt(2)))
    action_embedding = tf.layers.dense(action_ph, 2048,
                                       activation=None,
                                       kernel_initializer=weights_initializer,
                                       bias_initializer=tf.constant_initializer(0.),
                                       name='embedding_action')
    x = tf.multiply(x=action_embedding, y=x, name='action_mul_feature')
    assert x.shape.as_list() == [num_batch, 2048], x.shape

    x = relu(linear(x, 'fc2', n_hidden=1024, init_scale=np.sqrt(2)))
    assert x.shape.as_list() == [num_batch, 1024], x.shape

    x = relu(linear(x, 'fc3', n_hidden=6400, init_scale=np.sqrt(2)))
    assert x.shape.as_list() == [num_batch, 6400], x.shape

    x = tf.reshape(x, (-1, 10, 10, 64))
    assert x.shape.as_list() == [num_batch, 10, 10, 64], x.shape

    x = slim.conv2d_transpose(x, 64, 6, 2,
                              'same',
                              activation_fn=tf.nn.relu,
                              normalizer_fn=batch_norm_fn,
                              weights_initializer=weights_initializer,
                              scope='conv_transpose1')
    assert x.shape.as_list() == [num_batch, 20, 20, 64], x.shape

    x = slim.conv2d_transpose(x, 64, 6, 2,
                              'same',
                              activation_fn=tf.nn.relu,
                              normalizer_fn=batch_norm_fn,
                              weights_initializer=weights_initializer,
                              scope='conv_transpose2')
    assert x.shape.as_list() == [num_batch, 40, 40, 64], x.shape

    x = slim.conv2d_transpose(x, 4, 6, 2,
                              'valid',
                              activation_fn=tf.nn.sigmoid,
                              normalizer_fn=None,
                              weights_initializer=weights_initializer,
                              scope='conv_transpose3')
    assert x.shape.as_list() == [num_batch, 84, 84, 4], x.shape
  return x
