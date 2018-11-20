"""Play atari with trained model"""

from stable_baselines import logger
from stable_baselines.common.cmd_util import make_video_atari_env, atari_arg_parser
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.a2c_sil import SelfImitationA2C
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
import numpy as np
from pathlib import Path


def play(env_id, num_timesteps, seed, policy, lr_schedule, num_env,
         sil_update, sil_beta, load_path):
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
  env = make_video_atari_env(env_id, num_env, seed, wrapper_kwargs=env_args)
  env = VecFrameStack(env, 4)

  model = SelfImitationA2C.load(load_path, env=env)
  print(model.params)
  return_ = np.zeros((env.num_envs,))
  terminals_ = np.zeros((env.num_envs,), dtype=np.bool)
  print(model.env)
  observ = env.reset()
  while True:
    actions, values, states, _ = model.step(observ, None, None)
    next_observ, rewards, terminals, _ = env.step(actions)
    print(rewards)
    return_ += rewards
    terminals_ |= terminals
    # print('terminals', terminals_)
    done = True
    for terminal in terminals_.tolist():
      done &= terminal
    if done:
      break

  for mp4_file in Path('/tmp/video').glob('*.mp4'):
    if int(mp4_file.stat().st_size) < 100:
      mp4_file.unlink()
  print(return_)


def main():
  """
  Runs the test
  """
  parser = atari_arg_parser()
  parser.add_argument('--load-path', default=None, type=str)
  parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm'], default='cnn', help='Policy architecture')
  parser.add_argument('--lr_schedule', choices=['constant', 'linear'], default='constant',
                      help='Learning rate schedule')
  parser.add_argument('--sil-update', type=int, default=4, help="Number of updates per iteration")
  parser.add_argument('--sil-beta', type=float, default=0.1, help="Beta for weighted IS")
  args = parser.parse_args()
  assert args.load_path != None
  logger.configure()
  play(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy, lr_schedule=args.lr_schedule,
       num_env=16, sil_update=args.sil_update, sil_beta=args.sil_beta, load_path=args.load_path)


if __name__ == '__main__':
  main()
