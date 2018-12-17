from stable_baselines.common.successor_features import encode_cnn, decode_cnn
from stable_baselines.reconstruction import utility
from stable_baselines.common import tf_util
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import make_proba_dist_type
import tensorflow as tf
from stable_baselines.a2c_sil.utils import find_trainable_variables
from typing import Tuple, Union
from pathlib import Path

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from tqdm import tqdm
import cloudpickle
from gym import spaces
import numpy as np

ObservSpace = Union[spaces.Discrete, spaces.Box]
ActionSpace = Union[spaces.Discrete, spaces.Box]

PROPERTY = 'pacman_nonscale_noActiv_recons'
LOG_DIR = Path('tf_log/vae') / PROPERTY
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _save_to_file(save_path, data=None, params=None):
  save_path = Path(save_path).with_suffix('.pkl')
  save_path.touch(exist_ok=True)
  with open(save_path, "wb") as file:
    cloudpickle.dump((data, params), file)


def _load_from_file(load_path):
  load_path = Path(load_path).with_suffix('.pkl')
  if not load_path.exists():
    raise ValueError("Error: the file {} could not be found".format(load_path))

  with open(load_path, "rb") as file:
    data, params = cloudpickle.load(file)

  return data, params


def _calculate_encoding_capacity(step,
                                 capacity_limit=25,
                                 capacity_change_duration=100_000) -> float:
  c = capacity_limit
  if step <= capacity_change_duration:
    c *= (step / capacity_change_duration)
  return c


class ReconstructionModule(object):

  def __init__(self, sess: tf.Session,
               observation_space: ObservSpace,
               action_space: ActionSpace,
               num_batch: int):
    self._sess = sess
    self._observ_space = observation_space
    self._action_space = action_space
    self._num_batch = num_batch

    self._setup_input()
    self.mu, self.log_sigma_sq, self.recons_x = self._build_network(self.process_x,
                                                                    self.one_hot_actions)

  def _setup_input(self):
    with tf.variable_scope('input', reuse=False):
      self.input_x, self.process_x = observation_input(self._observ_space, self._num_batch)
      self.next_input_x, self.next_process_x = observation_input(self._observ_space, self._num_batch)
      pdtype = make_proba_dist_type(self._action_space)
      self.actions_ph = pdtype.sample_placeholder([self._num_batch], name="action_ph")
      self.one_hot_actions = tf.one_hot(self.actions_ph, self._action_space.n)

      self.capacity_ph = tf.placeholder(tf.float32, [], name='capacity_ph')

  def _build_network(self, observ: tf.Tensor,
                     onehot_action: tf.Tensor, is_training=True, reuse=False) \
      -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Encoding
    with tf.variable_scope('model', reuse=reuse):
      mu, log_sigma_sq = encode_cnn(observ, is_training=is_training, reuse=reuse,
                                    num_batch=self._num_batch)

      # Sampling
      with tf.name_scope('Sampling'):
        epsilon = tf.random_normal(tf.shape(mu))
        z = mu
        if is_training:
          z = tf.add(z, tf.multiply(tf.exp(tf.multiply(0.5, log_sigma_sq)), epsilon))
        z = tf.nn.l2_normalize(z, axis=1)

      # Decoding
      recons_x = decode_cnn(z, onehot_action, is_training=is_training, reuse=reuse,
                            num_batch=self._num_batch)
      return z, log_sigma_sq, recons_x


def main():
  beta = 0

  env_id = 'MsPacmanNoFrameskip-v4'
  num_env = 16
  num_steps = 5
  num_batch = num_env * num_steps

  seed = 0
  env_args = {'episode_life': False, 'clip_rewards': False, 'scale': False}
  env = VecFrameStack(make_atari_env(env_id, num_env, seed, wrapper_kwargs=env_args), 4)

  graph = tf.Graph()
  with graph.as_default():
    sess = tf_util.make_session(graph=graph)
    policy = ReconstructionModule(sess,
                                  env.observation_space,
                                  env.action_space,
                                  num_batch)

    def save(save_path: Path, params):
      data = {
        'policy': ReconstructionModule,
      }
      params = sess.run(params)
      _save_to_file(save_path, data=data, params=params)

    print(policy.mu)
    print(policy.log_sigma_sq)
    print(policy.recons_x)

    params = find_trainable_variables('model')
    tf.global_variables_initializer().run(session=sess)

    def load(load_path: Path):
      _data, load_params = _load_from_file(LOG_DIR)
      restores = []
      for param, load_param in zip(params, load_params):
        restores.append(param.assign(load_param))
      sess.run(restores)

    with tf.name_scope('losses'):
      recons_losses = tf.squared_difference(policy.next_process_x,
                                            policy.recons_x)
      recons_loss = tf.reduce_mean(recons_losses, name='reconstruction_loss')

    summary = utility.summary({
      policy.capacity_ph: 'capacity',
      recons_losses: 'recons_losses',
      policy.process_x: 'process_x',
      policy.next_process_x: 'next_process_x',
      policy.recons_x: 'recons_x',
    }, env.observation_space.shape,
      ignore=['recons_x', 'recons_losses', 'process_x'])
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=1e-5)
    optimizer = tf.train.AdamOptimizer(5e-4)
    train_op = optimizer.minimize(recons_loss)

    for event_file in LOG_DIR.glob('event*'):
      event_file.unlink()
    writer = tf.summary.FileWriter(LOG_DIR.as_posix(), sess.graph)
    sess.run(tf.global_variables_initializer())

    observs = []
    actions = []
    next_observs = []

    observ = env.reset()
    global_step = 0
    while True:
      if global_step > 100_000:
        break
      print('\rStep Global Step {}/{}'.format(global_step, 100_000 + 1), end='', flush=True)
      action = [env.action_space.sample() for _ in range(num_env)]
      next_observ, rewards, terminals, _ = env.step(action)

      observs.extend(observ)
      actions.extend(action)
      next_observs.extend(next_observ)

      observ = next_observ
      global_step += num_env

      if len(observs) == num_batch:
        feed_dict = {policy.input_x: np.asarray(observs),
                     policy.next_input_x: np.asarray(next_observs),
                     policy.actions_ph: np.asarray(actions),
                     policy.capacity_ph: _calculate_encoding_capacity(step),
                     }
        if global_step % (5 * num_batch) == 0:
          summary_, _ = sess.run([summary, train_op], feed_dict=feed_dict)

          writer.add_summary(summary_, global_step)
        else:
          _ = sess.run([train_op], feed_dict=feed_dict)

        observs = []
        actions = []
        next_observs = []

    save(LOG_DIR, params)


if __name__ == '__main__':
  main()
