
import numpy as np
import gym
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper_schema import GymSpaceWrapper
from rlpyt.utils.collections import NamedTupleSchema


class GymEnvWrapper(Wrapper):
    """Gym-style wrapper for converting the Openai Gym interface to the
    rlpyt interface.  Action and observation spaces are wrapped by rlpyt's
    ``GymSpaceWrapper``.

    Output `env_info` is automatically converted from a dictionary to a
    corresponding namedtuple, which the rlpyt sampler expects.  For this to
    work, every key that might appear in the gym environments `env_info` at
    any step must appear at the first step after a reset, as the `env_info`
    entries will have sampler memory pre-allocated for them (so they also
    cannot change dtype or shape). (see `EnvInfoWrapper`, `build_info_schemas`,
    and `info_to_nt` in file or more help/details)

    Warning:
        Unrecognized keys in `env_info` appearing later during use will be
        silently ignored.

    This wrapper looks for gym's ``TimeLimit`` env wrapper to
    see whether to add the field ``timeout`` to env info.   
    """

    def __init__(self, env,
            act_null_value=0, obs_null_value=0, force_float32=True):
        super().__init__(env)
        o = self.env.reset()
        o, r, d, info = self.env.step(self.env.action_space.sample())
        env_ = self.env
        time_limit = isinstance(self.env, TimeLimit)
        while not time_limit and hasattr(env_, "env"):
            env_ = env_.env
            time_limit = isinstance(self.env, TimeLimit)
        if time_limit:
            info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
        self._time_limit = time_limit
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        self._info_schemas = {}
        self._build_info_schemas(info)

    def _build_info_schemas(self, info, name="info"):
        ntc = self._info_schemas.get(name)
        info_keys = [str(k).replace(".", "_") for k in info.keys()]
        if ntc is None:
            self._info_schemas[name] = NamedTupleSchema(
                name, list(info_keys))
        elif not (isinstance(ntc, NamedTupleSchema) and
                  sorted(ntc._fields) == sorted(info_keys)):
            raise ValueError(f"Name clash in schema index: {name}.")
        for k, v in info.items():
            if isinstance(v, dict):
                self._build_info_schemas(v, "_".join([name, k]))

    def step(self, action):
        """Reverts the action from rlpyt format to gym format (i.e. if composite-to-
        dictionary spaces), steps the gym environment, converts the observation
        from gym to rlpyt format (i.e. if dict-to-composite), and converts the
        env_info from dictionary into namedtuple."""
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info, self._info_schemas)
        return EnvStep(obs, r, d, info)

    def reset(self):
        """Returns converted observation from gym env reset."""
        return self.observation_space.convert(self.env.reset())

    @property
    def spaces(self):
        """Returns the rlpyt spaces for the wrapped env."""
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )


def info_to_nt(value, schemas, name="info"):
    if not isinstance(value, dict):
        return value
    ntc = schemas[name]
    # Disregard unrecognized keys:
    values = {k: info_to_nt(v, schemas, "_".join([name, k]))
              for k, v in value.items() if k in ntc._fields}
    # Can catch some missing values (doesn't nest):
    values.update({k: 0 for k in ntc._fields if k not in values})
    return ntc(**values)


# To use: return a dict of keys and default values which sometimes appear in
# the wrapped env's env_info, so this env always presents those values (i.e.
# make keys and values keep the same structure and shape at all time steps.)
# Here, a dict of kwargs to be fed to `sometimes_info` should be passed as an
# env_kwarg into the `make` function, which should be used as the EnvCls.
# def sometimes_info(*args, **kwargs):
#     # e.g. Feed the env_id.
#     # Return a dictionary (possibly nested) of keys: default_values
#     # for this env.
#     return {}


class EnvInfoWrapper(Wrapper):
    """Gym-style environment wrapper to infill the `env_info` dict of every
    ``step()`` with a pre-defined set of examples, so that `env_info` has
    those fields at every step and they are made available to the algorithm in
    the sampler's batch of data.
    """

    def __init__(self, env, info_example):
        super().__init__(env)
        # self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._sometimes_info = info_example

    def step(self, action):
        """If need be, put extra fields into the `env_info` dict returned.
        See file for function ``infill_info()`` for details."""
        o, r, d, info = super().step(action)
        # Try to make info dict same key structure at every step.
        return o, r, d, infill_info(info, self._sometimes_info)


def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info


def make(*args, info_example=None, **kwargs):
    """Use as factory function for making instances of gym environment with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``.  If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    if info_example is None:
        return GymEnvWrapper(gym.make(*args, **kwargs))
    else:
        return GymEnvWrapper(EnvInfoWrapper(
            gym.make(*args, **kwargs), info_example))
