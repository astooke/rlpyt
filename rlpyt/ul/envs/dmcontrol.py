
import numpy as np
import gym
from gym.wrappers import TimeLimit
import dmc2gym
from collections import deque

from rlpyt.envs.gym import GymEnvWrapper


class TimeLimitMinusOne(TimeLimit):
    """Like TimeLimit, but checks against max_episode_steps - 1, because
    that's when dcm2gym environments seem to end the episode.
    Subclass TimeLimit because the rlpyt GymEnvWrapper looks for instances of
    this class to change the env_info field name to simply 'timeout'."""

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps - 1:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def make(*args, frame_stack=3, from_pixels=True, height=84, width=84,
        frame_skip=4, **kwargs):
    env = dmc2gym.make(*args,
        frame_skip=frame_skip,
        visualize_reward=False,
        from_pixels=from_pixels,
        height=height,
        width=width,
        **kwargs)
    if isinstance(env, TimeLimit):
        # Strip the gym TimeLimit wrapper and replace with one which
        # outputs TimeLimit.truncated=True at max_episode_steps - 1,
        # because that's when the dmc2gym env seems to end the episode.
        print("WARNING: replacing Gym TimeLimit wrapper by TimeLimitMinusOne")
        env = TimeLimitMinusOne(env.env)
    if from_pixels:
        env = FrameStack(env, k=frame_stack)
    elif frame_stack != 1:
        print("WARNING: dmcontrol.make() requested with frame_stack>1, but not"
            " doing it on state.")
    env = GymEnvWrapper(env)
    env._frame_skip = frame_skip

    return env
