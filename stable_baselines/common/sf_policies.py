from stable_baselines.common.policies import ActorCriticPolicy, nature_cnn, linear
import tensorflow as tf
from tensorflow.contrib import slim
from gym.spaces import Discrete, Box
from typing import Union

ObservSpace = Union[Discrete, Box]
ActionSpace = Union[Discrete, Box]


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
               cnn_extractor=nature_cnn, feature_extraction="cnn", add_action_ph=True, **kwargs):
    super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                            reuse=reuse, scale=(feature_extraction == "cnn"),
                                            add_action_ph=add_action_ph)
    if layers is None:
      layers = [64, 64]

    with tf.variable_scope("model", reuse=reuse):
      value_fn: tf.Tensor = None
      recons_mod: tf.Tensor = None
      if feature_extraction == "cnn":
        extracted_features = cnn_extractor(self.processed_x, **kwargs)
        # TODO: L2 Normalize extracted features
        assert len(extracted_features.shape) == 2
        extracted_features = tf.nn.l2_normalize(extracted_features, axis=1)
        # TODO: Add machado reconstruction module
        recons_mod = reconstruct(extracted_features, 'reconstruct',
                                 action_ph=self.action_ph,
                                 num_action_space=ac_space.n)
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

  def proba_step(self, obs, state=None, mask=None):
    return self.sess.run(self.policy_proba, {self.obs_ph: obs})

  def value(self, obs, state=None, mask=None):
    return self.sess.run(self._value, {self.obs_ph: obs})


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
