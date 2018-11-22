from stable_baselines.common.policies import ActorCriticPolicy, nature_cnn, linear
import tensorflow as tf
from tensorflow.contrib import slim
from gym.spaces import Discrete, Box
import numpy as np
from typing import Union, Tuple

from stable_baselines.a2c.utils import conv, linear, conv_to_fc

ObservSpace = Union[Discrete, Box]
ActionSpace = Union[Discrete, Box]
FEATURE_SIZE = 1024

def sf_cnn(scaled_images, **kwargs) -> tf.Tensor:
  """
  CNN from Nature paper.

  :param scaled_images: (TensorFlow Tensor) Image input placeholder
  :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
  :return: (TensorFlow Tensor) The CNN output layer
  """
  activ = tf.nn.relu
  layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
  layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
  layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
  layer_3 = conv_to_fc(layer_3)
  return activ(linear(layer_3, 'fc1', n_hidden=FEATURE_SIZE, init_scale=np.sqrt(2)))

def reconstruct(extracted_features: tf.Tensor,
                scope: str,
                action_ph: tf.Tensor,
                num_action_space: int) -> tf.Tensor:
  with tf.variable_scope(scope):
    x = tf.layers.dense(extracted_features, 2048,
                        kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                        bias_initializer=tf.constant_initializer(0.),
                        activation=tf.nn.relu)

    one_hot_actions = tf.one_hot(action_ph, num_action_space)
    action_embedding = tf.layers.dense(one_hot_actions, 2048,
                                       kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                       bias_initializer=tf.constant_initializer(0.),
                                       activation=tf.nn.relu,
                                       name='action_embedding')
    x = tf.multiply(x=action_embedding,
                    y=x)
    x = tf.layers.dense(x, 1024, None,
                        kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                        bias_initializer=tf.constant_initializer(0.))
    assert x.shape.as_list()[1:] == [1024], x.shape

    x = tf.layers.dense(x, 6400, tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.constant_initializer(0.))
    assert x.shape.as_list()[1:] == [6400], x.shape

    x = tf.reshape(x, [-1, 10, 10, 64])
    assert x.shape.as_list()[1:] == [10, 10, 64], x.shape

    x = slim.conv2d_transpose(x, 64, 6, 2,
                              'same',
                              activation_fn=tf.nn.relu,
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              biases_initializer=tf.constant_initializer(0.),
                              scope='conv_transpose1')
    assert x.shape.as_list()[1:] == [20, 20, 64], x.shape

    x = slim.conv2d_transpose(x, 64, 6, 2,
                              'same',
                              activation_fn=tf.nn.relu,
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              biases_initializer=tf.constant_initializer(0.),
                              scope='conv_transpose2')
    assert x.shape.as_list()[1:] == [40, 40, 64], x.shape

    # x = slim.conv2d_transpose(x, 4, 6, 2,
    #                           'valid',
    #                           activation_fn=tf.nn.sigmoid,
    #                           weights_initializer=tf.contrib.layers.xavier_initializer(),
    #                           scope='conv_transpose3')
    # assert x.shape.as_list()[1:] == [84, 84, 4], x.shape
    x = slim.conv2d_transpose(x, 1, 6, 2,
                              'valid',
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              biases_initializer=tf.constant_initializer(0.),
                              scope='conv_transpose3')
    assert x.shape.as_list()[1:] == [84, 84, 1], x.shape

    x = tf.layers.flatten(x)
    assert x.shape.as_list()[1:] == [84 * 84], x.shape
    return x


def sf_estimator(l2_normalized_feature, scope='sf_estimator'):
  with tf.variable_scope(scope):
    block_gradient = tf.stop_gradient(input=l2_normalized_feature)

    fc_sr1 = tf.layers.dense(inputs=block_gradient,
                             units=2 * FEATURE_SIZE,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.0),
                             activation=tf.nn.relu)

    sr_output = tf.layers.dense(inputs=fc_sr1,
                                units=FEATURE_SIZE,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.constant_initializer(0.0),
                                activation=tf.nn.relu)
    return sr_output


class FeedForwardPolicy(ActorCriticPolicy):
  """
  Policy object that implements actor critic, using a feed forward neural network.

  :param sess: (TensorFlow session) The current TensorFlow session
  :param ob_space: (Gym Space) The observation space of the environment
  :param ac_space: (Gym Space) The action space of the environment
  :param n_env: (int) The number of environments to run
  :param n_steps: (int) The number of steps to run for each environment
  :param n_batch: (int) The number of batch to run (n_envs * n_steps)
  :param reuse: (bool) If the policy is reusable or not
  :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
  :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
  :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
  :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
  """

  def __init__(self, sess: tf.Session, ob_space: ObservSpace, ac_space: ActionSpace,
               n_env: int, n_steps: int, n_batch: int, reuse=False, layers=None,
               cnn_extractor=sf_cnn, feature_extraction="cnn", add_action_ph=True, **kwargs):
    super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                            reuse=reuse, scale=(feature_extraction == "cnn"),
                                            add_action_ph=add_action_ph)
    if layers is None:
      layers = [64, 64]

    with tf.variable_scope("model", reuse=reuse):
      value_fn: tf.Tensor = None
      recons_mod: tf.Tensor = None
      successor_feature: tf.Tensor = None
      extracted_features: tf.Tensor = None
      if feature_extraction == "cnn":
        extracted_features = cnn_extractor(self.processed_x, **kwargs)
        # TODO: L2 Normalize extracted features
        assert len(extracted_features.shape) == 2
        extracted_features = tf.nn.l2_normalize(extracted_features, axis=1)
        # TODO: Add machado reconstruction module
        recons_mod = reconstruct(extracted_features, 'reconstruct',
                                 action_ph=self.action_ph,
                                 num_action_space=ac_space.n)
        # TODO: Add machado SF estimator
        successor_feature = sf_estimator(extracted_features)
        value_fn = linear(extracted_features, 'vf', 1)
        pi_latent = extracted_features
        vf_latent = extracted_features
      else:
        raise NotImplementedError('Not implement reconstruction module yet.')
        # activ = tf.tanh
        # processed_x = tf.layers.flatten(self.processed_x)
        # pi_h = processed_x
        # vf_h = processed_x
        # for i, layer_size in enumerate(layers):
        #   pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
        #   vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
        # value_fn = linear(vf_h, 'vf', 1)
        # pi_latent = pi_h
        # vf_latent = vf_h

      self.proba_distribution, self.policy, self.q_value = \
        self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

    self.value_fn = value_fn
    self.recons_mod = recons_mod
    self.successor_feature = successor_feature
    self._feature = extracted_features
    self.initial_state = None
    self._setup_init()

  def step(self, obs, state=None, mask=None, deterministic=False):
    if deterministic:
      action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                             {self.obs_ph: obs})
    else:
      action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                             {self.obs_ph: obs})
    return action, value, self.initial_state, neglogp

  def step_with_sf(self, observ, state=None, mask=None, deterministic=False) \
      -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if deterministic:
      action, value, neglogp, feature = self.sess.run(
        [self.deterministic_action, self._value, self.neglogp, self._feature],
        {self.obs_ph: observ})
    else:
      action, value, neglogp, feature = self.sess.run(
        [self.action, self._value, self.neglogp, self._feature],
        {self.obs_ph: observ})
    return action, value, self.initial_state, neglogp, feature

  def proba_step(self, obs, state=None, mask=None):
    return self.sess.run(self.policy_proba, {self.obs_ph: obs})

  def value(self, obs, state=None, mask=None):
    return self.sess.run(self._value, {self.obs_ph: obs})

  def estimate_recons(self, observ, action):
    return self.sess.run(self.recons_mod, {self.obs_ph: observ,
                                           self.action_ph: action})

  def estimate_sf(self, observ):
    return self.sess.run(self.successor_feature, {self.obs_ph: observ})


class CnnPolicy(FeedForwardPolicy):
  """
  Policy object that implements actor critic, using a CNN (the nature CNN)

  :param sess: (TensorFlow session) The current TensorFlow session
  :param ob_space: (Gym Space) The observation space of the environment
  :param ac_space: (Gym Space) The action space of the environment
  :param n_env: (int) The number of environments to run
  :param n_steps: (int) The number of steps to run for each environment
  :param n_batch: (int) The number of batch to run (n_envs * n_steps)
  :param reuse: (bool) If the policy is reusable or not
  :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
  """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
    super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                    feature_extraction="cnn", **_kwargs)
