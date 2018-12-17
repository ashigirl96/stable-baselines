from stable_baselines.common.successor_features import encode_cnn, decode_cnn
from stable_baselines.reconstruction import utility
from stable_baselines.common import tf_util
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import make_proba_dist_type
import tensorflow as tf
from typing import Tuple
from pathlib import Path

from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack

LOG_DIR = Path('tf_log')
LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_network(img, action_ph, is_training=True, reuse=False) \
    -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  # Encoding
  mu, sigma_sq = encode_cnn(img, is_training=is_training, reuse=reuse)

  # Sampling
  with tf.name_scope('Sampling'):
    epsilon = tf.random_normal(tf.shape(mu))
    if is_training:
      z = tf.add(mu,
                 tf.multiply(
                   tf.exp(tf.multiply(0.5, sigma_sq)),
                   epsilon))
    else:
      z = mu

  # Decoding
  img_reconstruction = decode_cnn(z, action_ph, is_training=is_training, reuse=reuse)
  return mu, sigma_sq, img_reconstruction


def main():
  env_id = 'BreakoutNoFrameskip-v4'
  num_env = 5
  seed = 0
  env_args = {'episode_life': False, 'clip_rewards': False}
  env = VecFrameStack(make_atari_env(env_id, num_env, seed, wrapper_kwargs=env_args), 4)
  graph = tf.Graph()
  with graph.as_default():
    sess = tf_util.make_session(graph=graph)
    with tf.variable_scope('input', reuse=False):
      input_x, process_x = observation_input(env.observation_space, num_env)
      print(env.action_space.shape)
      pdtype = make_proba_dist_type(env.action_space)
      actions_ph = pdtype.sample_placeholder([num_env], name="action_ph")
      one_hot_actions = tf.one_hot(actions_ph, env.action_space.n)
      
    print(input_x, process_x)
    print('action', actions_ph, one_hot_actions)

    beta = 0.1
    mu, sigma_sq, recons_x = build_network(process_x, one_hot_actions)
    print(mu)
    print(sigma_sq)
    print(recons_x)

    with tf.name_scope('losses'):
      recons_loss = tf.losses.mean_squared_error(input_x, recons_x, scope='recons_loss')
      kl_divergence = -tf.reduce_mean(0.5 * (tf.add(1., sigma_sq) - tf.pow(mu, 2) - tf.exp(sigma_sq)),
                                      name='kl_divergence')
      loss = tf.add(recons_loss,
                    tf.multiply(
                      kl_divergence,
                      beta), name='objective')
      print(loss)
    summary = utility.summary({recons_loss: 'recons_loss',
                               kl_divergence: 'kl_divergence',
                               mu: 'phi_mu',
                               sigma_sq: 'sigma_sq',
                               recons_x: 'recons_x',
                               input_x: 'input_x',
                               }, env.observation_space.shape)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    train_op = optimizer.minimize(loss)

    for event_file in LOG_DIR.glob('event*'):
      event_file.unlink()
    writer = tf.summary.FileWriter(LOG_DIR.as_posix(), sess.graph)
    sess.run(tf.global_variables_initializer())

    observ = env.reset()
    actions = [env.action_space.sample() for _ in range(num_env)]
    print(env.observation_space)
    print(observ.shape)

    recons_image, summary_ = sess.run([recons_x, summary],
                                      feed_dict={input_x: observ,
                                                 actions_ph: actions})
    writer.add_summary(summary_, 0)


if __name__ == '__main__':
  main()
