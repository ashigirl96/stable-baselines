import tensorflow as tf
import numpy as np
from gym import spaces
from typing import Tuple, Union
from pathlib import Path
from tqdm import tqdm

from stable_baselines.common.input import observation_input
from stable_baselines.common import tf_util
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.reconstruction import utility

tfd = tf.distributions
ObservSpace = Union[spaces.Discrete, spaces.Box]
ActionSpace = Union[spaces.Discrete, spaces.Box]
BETA = 3.
PROPERTY = 'Pacman'
LOG_DIR = Path('tf_log/2M') / PROPERTY
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Trainer(object):
  def __init__(self, name='MsPacmanNoFrameskip-v4'):
    self._make_env(name)
    self.sess = tf_util.make_session()
    self.input_x, self.process_x = observation_input(self.observ_space, self.num_batch)
    self.input_next_x, self.process_next_x = observation_input(self.observ_space, self.num_batch)
    self.conv_vae = ConvVAE(self.input_x, 200, self.sess, self.observ_space, self.action_space)
    self._define_loss_fn()
    self._define_optimizer()
    self._define_summary()

  def train(self):
    writer = tf.summary.FileWriter(LOG_DIR.as_posix(), self.sess.graph)
    self.sess.run(tf.global_variables_initializer())
    observs = []
    actions = []
    next_observs = []

    observ = self.env.reset()
    for step in tqdm(range(1, 20_000_000 + 1, self.num_env)):
      action = [self.env.action_space.sample() for _ in range(self.num_env)]
      next_observ, rewards, terminals, _ = self.env.step(action)

      observs.extend(observ)
      actions.extend(action)
      next_observs.extend(next_observ)

      observ = next_observ

      if len(observs) == self.num_batch:
        feed_dict = {self.input_x: observs,
                     self.input_next_x: next_observs}
        if (step // self.num_env + 1) % 1000 == 0:
          summary, _ = self.sess.run([self.summary_op, self.train_op], feed_dict=feed_dict)
          writer.add_summary(summary, step)
        else:
          _ = self.sess.run([self.train_op], feed_dict=feed_dict)

        observs = []
        actions = []
        next_observs = []

  def _make_env(self, name):
    self.num_env = 16
    num_steps = 5
    self.num_batch = self.num_env * num_steps
    seed = 0
    env_args = {'episode_life': False, 'clip_rewards': False, 'scale': True}
    self.env = VecFrameStack(make_atari_env(name, self.num_env, seed, wrapper_kwargs=env_args), 4)
    self.observ_space = self.env.observation_space
    self.action_space = self.env.action_space

  def _define_loss_fn(self):
    self.likelihood = tf.losses.mean_squared_error(self.process_next_x,
                                                   self.conv_vae.reconst)
    kl_divergence = -0.5 * tf.reduce_sum(
      1 + self.conv_vae.logvar - tf.square(self.conv_vae.mu) - tf.exp(self.conv_vae.logvar))
    self.kl_divergence = tf.reduce_sum(kl_divergence / self.num_batch)

    self.loss = self.likelihood + BETA * self.kl_divergence
    print('likelihood', self.likelihood)
    print('kl divergence', self.kl_divergence)

  def _define_optimizer(self):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=1e-5)
    self.train_op = optimizer.minimize(self.loss)

  def _define_summary(self):
    self.summary_op = utility.summary({
      self.loss: 'loss',
      self.kl_divergence: 'kl_divergences',
      self.likelihood: 'likelihood',
      self.process_next_x: 'next_x',
      self.conv_vae.reconst: 'reconst_x',
    }, self.env.observation_space.shape,
      ignore=['recons_x', 'recons_losses', 'process_x'])


class ConvVAE(object):

  def __init__(self, input_x: tf.Tensor, z_dim, sess: tf.Session,
               observ_space: ObservSpace, action_space: ActionSpace):
    self._observ_space = observ_space
    self._action_space = action_space
    self._sess = sess

    input_shape = input_x.shape
    self.image_size = 3 * input_shape[0] * input_shape[1]
    self.input_x = input_x
    self.z_dim = z_dim

    self.conv1 = tf.layers.Conv2D(32, 4, 2, 'same')
    self.conv2 = tf.layers.Conv2D(64, 4, 2, 'same')
    self.conv3 = tf.layers.Conv2D(128, 4, 2, 'same')
    self.conv4 = tf.layers.Conv2D(256, 4, 2, 'same')

    self.eps = tfd.Uniform(0., 1., allow_nan_stats=False)

    self.fc1 = tf.layers.Dense(self.z_dim)
    self.fc2 = tf.layers.Dense(self.z_dim)
    self.fc3 = tf.layers.Dense(256 * 6 * 6)

    self.deconv1 = tf.layers.Conv2DTranspose(128, 3, strides=2, padding='valid')
    self.deconv2 = tf.layers.Conv2DTranspose(64, 4, strides=2, padding='valid')
    self.deconv3 = tf.layers.Conv2DTranspose(32, 4, strides=2, padding='valid')
    self.deconv4 = tf.layers.Conv2DTranspose(16, 6, strides=2, padding='valid')
    self.deconv5 = tf.layers.Conv2DTranspose(4, 6, strides=2, padding='valid')

    self.mu, self.logvar = self._encode(self.input_x)
    self.latent_code = self._reparameterize(self.mu, self.logvar)
    self.reconst = self._decode(self.latent_code)

  def _encode(self, x):
    h = tf.nn.relu(self.conv1(x))
    print(h)
    h = tf.nn.relu(self.conv2(h))
    print(h)
    h = tf.nn.relu(self.conv3(h))
    print(h)
    h = tf.nn.relu(self.conv4(h))
    print(h)
    h = tf.reshape(h, [-1, 256 * 6 * 6])
    print(h)
    return self.fc1(h), self.fc2(h)

  def _reparameterize(self, mu, logvar):
    std = tf.exp(0.5 * logvar)
    eps = self.eps.sample(sample_shape=tf.shape(std))
    return eps * std + mu

  def _decode(self, latent_code):
    h = self.fc3(latent_code)
    h = tf.reshape(h, [-1, 1, 1, 256 * 6 * 6])
    print('reshaped', h)
    h = tf.nn.relu(self.deconv1(h))
    print('deconv1', h)
    h = tf.nn.relu(self.deconv2(h))
    print(h)
    h = tf.nn.relu(self.deconv3(h))
    print(h)
    h = tf.nn.relu(self.deconv4(h))
    print(h)
    h = tf.nn.sigmoid(self.deconv5(h))
    print(h)
    return h

  def __call__(self, x, encode=False, mean=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feed_dict = {self.input_x: x}
    if encode:
      if mean:
        return self._sess.run(self.mu, feed_dict)
      return self._sess.run(self.latent_code, feed_dict)
    return self._sess.run([self.reconst, self.mu, self.logvar],
                          feed_dict)


if __name__ == '__main__':
  trainer = Trainer()
  trainer.train()
