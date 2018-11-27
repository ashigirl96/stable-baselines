#!/usr/bin/env python3

from stable_baselines import logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.a2c_sf import SuccessorFeatureA2C
from stable_baselines.common.policies import CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.sf_policies import CnnPolicy


def train(env_id, num_timesteps, seed, policy, lr_schedule, num_env,
          sil_update, sil_beta,
          tensorboard_log, tb_log_name):
  """
  Train A2C model for atari environment, for testing purposes

  :param env_id: (str) Environment ID
  :param num_timesteps: (int) The total number of samples
  :param seed: (int) The initial seed for training
  :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
  :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                               'double_linear_con', 'middle_drop' or 'double_middle_drop')
  :param num_env: (int) The number of environments
  """
  policy_fn = None
  if policy == 'cnn':
    policy_fn = CnnPolicy
  elif policy == 'lstm':
    policy_fn = CnnLstmPolicy
  elif policy == 'lnlstm':
    policy_fn = CnnLnLstmPolicy
  if policy_fn is None:
    raise ValueError("Error: policy {} not implemented".format(policy))

  env_args = {'episode_life': False, 'clip_rewards': False, 'scale': True}
  env = VecFrameStack(make_atari_env(env_id, num_env, seed, wrapper_kwargs=env_args), 4)

  model = SuccessorFeatureA2C(policy_fn, env, lr_schedule=lr_schedule, tensorboard_log=tensorboard_log,
                              verbose=1, sil_update=sil_update, sil_beta=sil_beta)
  model.learn(total_timesteps=int(num_timesteps * 1.1), seed=seed, tb_log_name=tb_log_name)
  model.save(model.save_directory.as_posix())
  env.close()


def main():
  """
  Runs the test
  """
  parser = atari_arg_parser()
  parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm'], default='cnn', help='Policy architecture')
  parser.add_argument('--lr_schedule', choices=['constant', 'linear'], default='constant',
                      help='Learning rate schedule')
  parser.add_argument('--sil-update', type=int, default=4, help="Number of updates per iteration")
  parser.add_argument('--sil-beta', type=float, default=0.1, help="Beta for weighted IS")
  parser.add_argument('--tensorboard-log', type=str, default='./tf_log/SF')
  parser.add_argument('--tb', type=str, default='SIL_A2C')
  args = parser.parse_args()
  logger.configure()
  train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy, lr_schedule=args.lr_schedule,
        num_env=16, sil_update=args.sil_update, sil_beta=args.sil_beta,
        tensorboard_log=args.tensorboard_log, tb_log_name=args.tb)


if __name__ == '__main__':
  main()
