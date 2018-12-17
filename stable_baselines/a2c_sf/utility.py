# https://github.com/ashigirl96/trace_policy/blob/master/baselines/a2c/utils.py

from typing import Tuple
import numpy as np

def discounts_with_dones(rewards, features, dones, gamma) -> Tuple[np.ndarray, np.ndarray]:
  discounted_r = []
  discounted_f = []
  r = 0
  f = 0
  for reward, feature, done in zip(rewards[::-1], features[::-1], dones[::-1]):
    r = reward + gamma * r * (1. - done)  # fixed off by one bug
    f = feature + gamma * f * (1. - done)
    discounted_r.append(r)
    discounted_f.append(f)
  return discounted_r[::-1], discounted_f[::-1]
