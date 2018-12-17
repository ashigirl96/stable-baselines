from stable_baselines.reconstruction.worldmodel_recons import ConvVAE
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.atari_wrappers import TransposeImage
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
import numpy as np
from tqdm import tqdm
import torch


def tensor(input_x: np.ndarray) -> torch.Tensor:
  if isinstance(input_x, torch.Tensor):
    return input_x
  input_x = torch.tensor(input_x, dtype=torch.float32)
  return input_x


def main():
  env_id = 'PongNoFrameskip-v4'
  # env_id = 'MsPacmanNoFrameskip-v4'
  # env_id = 'BreakoutNoFrameskip-v4'
  num_env = 16
  num_steps = 5
  num_batch = num_env * num_steps

  seed = 0
  env_args = {'episode_life': False, 'clip_rewards': False, 'scale': False,
              'transpose_image': True}
  env = VecFrameStack(make_atari_env(env_id, num_env, seed, wrapper_kwargs=env_args), 4)

  network = ConvVAE([84, 84], 2048)

  observs = []
  actions = []
  next_observs = []

  observ = env.reset()
  observ = observ.transpose(0, 3, 2, 1)
  observ = tensor(observ)
  print(observ.shape)
  out = network(observ)[0]
  print(out.shape)

  # for step in tqdm(range(1, 2_000_000 + 1, num_env)):
  #   action = [env.action_space.sample() for _ in range(num_env)]
  #   next_observ, rewards, terminals, _ = env.step(action)
  #
  #   observs.extend(observ)
  #   actions.extend(action)
  #   next_observs.extend(next_observ)
  #
  #   observ = next_observ
  #
  #   if len(observs) == num_batch:
  #     if (step // num_env + 1) % 1000 == 0:
  #       pass
  #     else:
  #       pass
  #
  #     observs = []
  #     actions = []
  #     next_observs = []


if __name__ == '__main__':
  main()
