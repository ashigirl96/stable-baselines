import time

import gym
import numpy as np
import tensorflow as tf
from typing import Union, Type
from pathlib import Path

from stable_baselines import logger
from stable_baselines.common.self_imitation import SelfImitation
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy, CnnLnLstmPolicy, CnnLstmPolicy
from stable_baselines.common.sf_policies import CnnPolicy, FeedForwardPolicy, FEATURE_SIZE
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c_sf.utility import discounts_with_dones
from stable_baselines.a2c.utils import Scheduler, find_trainable_variables, mse, \
  total_episode_reward_logger

Policies = Type[Union[CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy]]


class SuccessorFeatureA2C(ActorCriticRLModel):
  """
  The SelfImitationA2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

  :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
  :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
  :param gamma: (float) Discount factor
  :param n_steps: (int) The number of steps to run for each environment per update
      (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
  :param vf_coef: (float) Value function coefficient for the loss calculation
  :param ent_coef: (float) Entropy coefficient for the loss caculation
  :param max_grad_norm: (float) The maximum value for the gradient clipping
  :param learning_rate: (float) The learning rate
  :param alpha: (float)  RMSProp decay parameter (default: 0.99)
  :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
      (default: 1e-5)
  :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                            'double_linear_con', 'middle_drop' or 'double_middle_drop')
  :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
  :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
  :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                            (used only for loading)
  """

  def __init__(self, policy: Policies, env, gamma=0.99, n_steps=5,
               vf_coef=0.25, ent_coef=0.01, recons_coef=1., sf_coef=2.,
               max_grad_norm=0.5, learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=0,
               tensorboard_log=None,
               _init_setup_model=True, sil_update=4, sil_beta=0):
    self.policy: Policies = None
    super(SuccessorFeatureA2C, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                              _init_setup_model=_init_setup_model)

    self.sil_update = sil_update
    self.sil_beta = sil_beta

    self.n_steps = n_steps
    self.gamma = gamma
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef
    self.recons_coef = recons_coef
    self.sf_coef = sf_coef
    self.max_grad_norm = max_grad_norm
    self.alpha = alpha
    self.epsilon = epsilon
    self.lr_schedule = lr_schedule
    self.learning_rate = learning_rate
    self.tensorboard_log = tensorboard_log

    self.graph = None
    self.sess: tf.Session = None
    self.learning_rate_ph = None
    self.n_batch = None
    self.actions_ph = None
    self.advs_ph = None
    self.rewards_ph = None
    self.pg_loss = None
    self.vf_loss = None
    self.entropy = None
    self.params = None
    self.apply_backprop = None
    self.train_model = None
    self.step_model = None
    self.step = None
    self.proba_step = None
    self.value = None
    # TODO: ADD
    self.successor_feature = None
    self.initial_state = None
    self.learning_rate_schedule = None
    self.summary = None
    self.episode_reward = None
    self.save_directory: Path = None

    # if we are loading, it is possible the environment is not known, however the obs and action space are known
    if _init_setup_model:
      self.setup_model()

  def setup_model(self):
    with SetVerbosity(self.verbose):

      assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                         "instance of common.policies.ActorCriticPolicy."
      assert issubclass(self.policy, FeedForwardPolicy), "Error: the input policy for the A2C model must be an " \
                                                         "instance of common.policies.FeedFowardPolicy."

      self.graph = tf.Graph()
      with self.graph.as_default():
        self.sess = tf_util.make_session(graph=self.graph)

        self.n_batch = self.n_envs * self.n_steps

        n_batch_step = None
        n_batch_train = None
        n_batch_sil = None
        if issubclass(self.policy, LstmPolicy):
          n_batch_step = self.n_envs
          n_batch_train = self.n_envs * self.n_steps
          # TODO: Add
          n_batch_sil = 512

        step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                 n_batch_step, reuse=False)

        # TODO: Add
        with tf.variable_scope("train_model", reuse=True,
                               custom_getter=tf_util.outer_scope_getter("train_model")):
          train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                    self.n_steps, n_batch_train, reuse=True)

        with tf.variable_scope("sil_model", reuse=True,
                               custom_getter=tf_util.outer_scope_getter("sil_model")):
          sil_model = self.policy(self.sess, self.observation_space, self.action_space,
                                  self.n_envs, self.n_steps, n_batch_sil, reuse=True)

        with tf.variable_scope("loss", reuse=False):
          # self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
          self.actions_ph = train_model.action_ph
          self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
          self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
          self.successor_feature_ph = tf.placeholder(tf.float32, [None, FEATURE_SIZE], name="successor_feature_ph")
          self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

          neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
          last_frame = tf.reshape(train_model.obs_ph[..., 3], shape=[-1, 84 * 84])
          recons_losses = tf.squared_difference(x=last_frame,
                                                y=train_model.recons_mod)
          self.recons_loss = tf.losses.mean_squared_error(labels=last_frame,
                                                          predictions=train_model.recons_mod)
          self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
          self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
          self.vf_loss = mse(tf.squeeze(train_model.value_fn), self.rewards_ph)
          # TODO: loss of SF
          self.sf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.successor_feature),
                                            self.successor_feature_ph))
          loss = self.pg_loss - \
                 self.entropy * self.ent_coef + \
                 self.vf_loss * self.vf_coef + \
                 self.sf_loss * self.sf_coef + \
                 self.recons_loss * self.recons_coef

          tf.summary.scalar('recons_loss/max', tf.reduce_max(recons_losses))
          tf.summary.scalar('recons_loss/min', tf.reduce_min(recons_losses))
          tf.summary.scalar('recons_loss', self.recons_loss)
          tf.summary.scalar('entropy_loss', self.entropy)
          tf.summary.scalar('policy_gradient_loss', self.pg_loss)
          tf.summary.scalar('value_function_loss', self.vf_loss)
          tf.summary.scalar('successor_feature_loss', self.sf_loss)
          tf.summary.scalar('loss', loss)

          self.params = find_trainable_variables("model")
          grads = tf.gradients(loss, self.params)
          if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
          grads = list(zip(grads, self.params))

        _last_frame = tf.reshape(last_frame, [-1, 84, 84, 1])
        _recons_mod = tf.reshape(train_model.recons_mod, [-1, 84, 84, 1])
        with tf.variable_scope("input_info", reuse=False):
          tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
          tf.summary.histogram('discounted_rewards', self.rewards_ph)
          tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate))
          tf.summary.histogram('learning_rate', self.learning_rate)
          tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
          tf.summary.histogram('advantage', self.advs_ph)
          tf.summary.image('last_frame', _last_frame)
          tf.summary.image('reconstruction', _recons_mod)
          if len(self.observation_space.shape) == 3:
            tf.summary.image('observation', train_model.obs_ph)
          else:
            tf.summary.histogram('observation', train_model.obs_ph)

        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                            epsilon=self.epsilon)
        self.apply_backprop = trainer.apply_gradients(grads)

        # TODO: Add
        self.sil = SelfImitation(
          model_ob=sil_model.obs_ph,
          model_vf=sil_model.value_fn,
          model_entropy=sil_model.proba_distribution.entropy(),
          fn_value=sil_model.value,
          fn_neg_log_prob=sil_model.proba_distribution.neglogp,
          ac_space=self.action_space,
          fn_reward=np.sign,
          n_env=self.n_envs,
          n_update=self.sil_update,
          beta=self.sil_beta)

        self.sil.build_train_op(
          params=self.params,
          optim=trainer,
          lr=self.learning_rate_ph,
          max_grad_norm=self.max_grad_norm)

        self.train_model = train_model
        self.step_model = step_model
        # self.step = step_model.step
        self.step = step_model.step_with_sf
        self.proba_step = step_model.proba_step
        self.value = step_model.value
        # TODO: Add
        self.successor_feature = step_model.estimate_sf
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=self.sess)

        self.summary = tf.summary.merge_all()

  def _train_step(self, obs, states, rewards, masks, actions, values, update,
                  writer: tf.summary.FileWriter = None, features=None):
    """
    applies a training step to the model

    :param obs: ([float]) The input observations
    :param states: ([float]) The states (used for recurrent policies)
    :param rewards: ([float]) The rewards from the environment
    :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
    :param actions: ([float]) The actions taken
    :param values: ([float]) The logits values
    :param update: (int) the current step iteration
    :param writer: (TensorFlow Summary.writer) the writer for tensorboard
    :return: (float, float, float) policy loss, value loss, policy entropy
    """
    advs = rewards - values
    cur_lr = None
    for _ in range(len(obs)):
      cur_lr = self.learning_rate_schedule.value()
    assert cur_lr is not None, "Error: the observation input array cannon be empty"

    td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
              self.rewards_ph: rewards, self.learning_rate_ph: cur_lr,
              self.successor_feature_ph: features}
    if states is not None:
      td_map[self.train_model.states_ph] = states
      td_map[self.train_model.masks_ph] = masks

    if writer is not None:
      # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
      if (1 + update) % 10 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, policy_loss, value_loss, policy_entropy, _, sf_loss = self.sess.run(
          [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop,
           self.sf_loss], td_map, options=run_options, run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'step%d' % (update * (self.n_batch + 1)))
      else:
        summary, policy_loss, value_loss, policy_entropy, _, sf_loss = self.sess.run(
          [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop,
           self.sf_loss], td_map)
      writer.add_summary(summary, update * (self.n_batch + 1))

    else:
      policy_loss, value_loss, policy_entropy, _, sf_loss = self.sess.run(
        [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop, self.sf_loss], td_map)

    return policy_loss, value_loss, policy_entropy, sf_loss

  def _train_sil(self):
    cur_lr = self.learning_rate_schedule.value()
    return self.sil.train(self.sess, cur_lr)

  def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="SIL_A2C"):
    with SetVerbosity(self.verbose), \
         TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:  # type: tf.summary.FileWriter
      self._setup_learn(seed)
      self.save_directory = Path(writer.get_logdir())

      self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                              schedule=self.lr_schedule)

      runner = SuccessorFeatureA2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
      self.episode_reward = np.zeros((self.n_envs,))

      t_start = time.time()
      for update in range(1, total_timesteps // self.n_batch + 1):
        # true_reward is the reward without discount
        obs, states, rewards, masks, actions, values, true_reward, raw_rewards, features = runner.run()
        _, value_loss, policy_entropy, sf_loss = self._train_step(obs, states, rewards, masks, actions, values, update,
                                                                  writer, features=features)
        sil_loss, sil_adv, sil_samples, sil_nlogp = self._train_sil()
        n_seconds = time.time() - t_start
        fps = int((update * self.n_batch) / n_seconds)

        if writer is not None:
          self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                            raw_rewards.reshape((self.n_envs, self.n_steps)),
                                                            masks.reshape((self.n_envs, self.n_steps)),
                                                            writer, update * (self.n_batch + 1))
          summary = tf.Summary(value=[tf.Summary.Value(
            tag="episode_reward/best_reward", simple_value=self.sil.get_best_reward())])
          writer.add_summary(summary, update * (self.n_batch + 1))

        if callback is not None:
          callback(locals(), globals())

        if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
          explained_var = explained_variance(values, rewards)
          logger.record_tabular("nupdates", update)
          logger.record_tabular("total_timesteps", update * self.n_batch)
          logger.record_tabular("fps", fps)
          logger.record_tabular("policy_entropy", float(policy_entropy))
          logger.record_tabular("value_loss", float(value_loss))
          logger.record_tabular('sf_loss', float(sf_loss))
          logger.record_tabular("explained_variance", float(explained_var))
          logger.record_tabular("best_episode_reward", float(self.sil.get_best_reward()))
          if self.sil_update > 0:
            logger.record_tabular("sil_num_episodes", float(self.sil.num_episodes()))
            logger.record_tabular("sil_valid_samples", float(sil_samples))
            logger.record_tabular("sil_steps", float(self.sil.num_steps()))
          logger.dump_tabular()

    return self

  def save(self, save_path):
    data = {
      "gamma": self.gamma,
      "n_steps": self.n_steps,
      "vf_coef": self.vf_coef,
      "ent_coef": self.ent_coef,
      "recons_coef": self.recons_coef,
      "sf_coef": self.sf_coef,
      "max_grad_norm": self.max_grad_norm,
      "learning_rate": self.learning_rate,
      "alpha": self.alpha,
      "epsilon": self.epsilon,
      "lr_schedule": self.lr_schedule,
      "verbose": self.verbose,
      "policy": self.policy,
      "observation_space": self.observation_space,
      "action_space": self.action_space,
      "n_envs": self.n_envs,
      "_vectorize_action": self._vectorize_action
    }

    params = self.sess.run(self.params)

    self._save_to_file(save_path, data=data, params=params)


class SuccessorFeatureA2CRunner(AbstractEnvRunner):
  def __init__(self,
               env,
               model: SuccessorFeatureA2C,
               n_steps=5, gamma=0.99):
    """
    A runner to learn the policy of an environment for an a2c model

    :param env: (Gym environment) The environment to learn from
    :param model: (Model) The model to learn
    :param n_steps: (int) The number of steps to run for each environment
    :param gamma: (float) Discount factor
    """
    super(SuccessorFeatureA2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
    self.model = model
    self.gamma = gamma
    feature_size = model.train_model.successor_feature.shape.as_list()[1]
    self.batch_sf_shape = (env.num_envs * n_steps,) + (feature_size,)

  def run(self):
    """
    Run a learning step of the model

    :return: ([float], [float], [float], [bool], [float], [float])
             observations, states, rewards, masks, actions, values
    """
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
    # TODO: add
    mb_raw_rewards = []
    mb_features = []
    mb_states = self.states
    for _ in range(self.n_steps):
      actions, values, states, _, features = self.model.step(self.obs, self.states, self.dones)
      mb_obs.append(np.copy(self.obs))
      mb_actions.append(actions)
      mb_values.append(values)
      mb_features.append(features)
      mb_dones.append(self.dones)
      clipped_actions = actions
      # Clip the actions to avoid out of bound error
      if isinstance(self.env.action_space, gym.spaces.Box):
        clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
      # obs, rewards, dones, _ = self.env.step(clipped_actions)
      obs, raw_rewards, dones, _ = self.env.step(clipped_actions)
      mb_raw_rewards.append(raw_rewards)
      rewards = np.sign(raw_rewards)
      self.states = states
      self.dones = dones
      if hasattr(self.model, 'sil'):
        self.model.sil.step(self.obs, actions, raw_rewards, dones)
      else:
        raise ValueError('Model doesnot have SIL module.')
      self.obs = obs
      mb_rewards.append(rewards)
    mb_dones.append(self.dones)
    # batch of steps to batch of rollouts
    mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
    mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(0, 1)
    mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
    # TODO: Add MB Features (not Successor Features)
    mb_features = np.asarray(mb_features, dtype=np.float32).swapaxes(1, 0)  # (16, 5, FEATURE_SIZE)
    assert mb_features.shape == (16, 5, FEATURE_SIZE), mb_features.shape
    mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
    mb_raw_rewards = np.asarray(mb_raw_rewards, dtype=np.float32).swapaxes(0, 1)
    mb_masks = mb_dones[:, :-1]
    mb_dones = mb_dones[:, 1:]
    true_rewards = np.copy(mb_rewards)
    last_values = self.model.value(self.obs, self.states, self.dones).tolist()  # (num_envs,)
    # TODO: Last SF
    last_sf = self.model.successor_feature(self.obs)  # (num_envs, FEATURE_SIZE)

    # TODO: Calculate discount Feature Representation
    for n, (rewards, dones, value, features, sf) in enumerate(
        zip(mb_rewards, mb_dones, last_values, mb_features, last_sf)):
      rewards = rewards.tolist()
      dones = dones.tolist()
      features = list(features)
      # TODO: discount_with_dones Features
      if dones[-1] == 0:
        rewards, features = discounts_with_dones(rewards + [value],
                                                 features + [sf],
                                                 dones + [0], self.gamma)
        rewards = rewards[:-1]
        features = features[:-1]
      else:
        rewards, features = discounts_with_dones(rewards,
                                                 features,
                                                 dones, self.gamma)
      mb_rewards[n] = rewards
      mb_features[n] = features  # (16, 5, FEATURE_SIZE)

    # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
    mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
    mb_raw_rewards = mb_raw_rewards.reshape(-1, *mb_raw_rewards.shape[2:])
    mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
    mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
    mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
    mb_features = mb_features.reshape(self.batch_sf_shape)
    return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, true_rewards, \
           mb_raw_rewards, mb_features
