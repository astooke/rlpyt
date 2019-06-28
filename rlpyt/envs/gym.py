
import numpy as np
import gym
from gym import Wrapper
from collections import namedtuple

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import SpaceWrapper


EnvInfo = None


class GymWrapper(Wrapper):
    """Converts env_info from dict to namedtuple, and ensures actual
    observation dtype matches that of observation space (e.g. gym mujoco
    environments have float32 obs space but actually output float64."""

    def __init__(self, env):
        super().__init__(env)
        o = self.env.reset()
        assert isinstance(o, np.ndarray)
        o, r, d, info = self.env.step(self.action_space.sample())
        global EnvInfo  # In case pickling, define at module level.
        # (Might break down if wrapping multiple, different envs, if
        # so, make different files.)
        # (To record, need all env_info fields present at every step.)
        EnvInfo = namedtuple("EnvInfo", list(info.keys()))
        # Wrap spaces to allow multiple samples at once.
        self.action_space = SpaceWrapper(self.env.action_space)
        dtype = np.float32 if o.dtype == np.float64 else None
        self.observation_space = SpaceWrapper(self.env.observation_space,
            dtype=dtype)

    def step(self, action):
        o, r, d, info = self.env.step(action)
        o = np.asarray(o, dtype=self.observation_space.dtype)
        # Fields appearing in info must be the same at every step.
        info = EnvInfo(**{k: v for k, v in info.items() if k in EnvInfo._fields})
        return EnvStep(o, r, d, info)

    def reset(self):
        return np.asarray(self.env.reset(), dtype=self.observation_space.dtype)

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )


def make(*args, **kwargs):
    return GymWrapper(gym.make(*args, **kwargs))
