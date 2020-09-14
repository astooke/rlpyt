
import os
import os.path as osp
import shutil
import numpy as np
import deepmind_lab
from collections import deque

from rlpyt.envs.base import Env
from rlpyt.spaces.int_box import IntBox


class DmlabEnv(Env):

    def __init__(
            self,
            level,
            height=72,
            width=96,
            action_repeat=4,
            frame_history=1,
            renderer="hardware",
            fps=None,
            episode_length_seconds=None,
            config_kwargs=None,
            cache_dir="/data/adam/dmlab_cache",
            gpu_device_index="EGL_DEVICE_ID",
            ):
        if level in DMLAB30:
            level = "/contributed/dmlab30/" + level
        level_cache = None if cache_dir is None else LevelCache(cache_dir)
        config = dict(height=str(height), width=str(width))
        if fps is not None:
            config["fps"] = str(fps)
        if episode_length_seconds is not None:
            config["episodeLengthSeconds"] = str(episode_length_seconds)
        if gpu_device_index is not None:
            if isinstance(gpu_device_index, int):
                gpu_device_index = str(gpu_device_index)
            else:
                gpu_device_index = os.environ.get(gpu_device_index, "0")
            config["gpuDeviceIndex"] = gpu_device_index
        if config_kwargs is not None:
            if config.keys() & config_kwargs.keys():
                raise KeyError(f"Had duplicate key(s) in config_kwargs: "
                    f"{config.keys() & config_kwargs.keys()}")
            config.update(config_kwargs)
        self.dmlab_env = deepmind_lab.Lab(
            level=level,
            observations=["RGB"],
            config=config,
            renderer=renderer,
            level_cache=level_cache,
        )
        self._action_map = np.array([ 
            [  0, 0,  0,  1, 0, 0, 0],  # Forward
            [  0, 0,  0, -1, 0, 0, 0],  # Backward
            [  0, 0, -1,  0, 0, 0, 0],  # Move Left
            [  0, 0,  1,  0, 0, 0, 0],  # Move Right
            [-20, 0,  0,  0, 0, 0, 0],  # Look Left
            [ 20, 0,  0,  0, 0, 0, 0],  # Look Right
            [-20, 0,  0,  1, 0, 0, 0],  # Left Forward
            [ 20, 0,  0,  1, 0, 0, 0],  # Right Forward
            [  0, 0,  0,  0, 1, 0, 0],  # Fire
                                     ], dtype=np.int32)
        self._action_space = IntBox(low=0, high=len(self._action_map))
        self._observation_space = IntBox(low=0, high=256,
            shape=(3 * frame_history, height, width), dtype=np.uint8)
        self._zero_obs = np.zeros((3, height, width), dtype=np.uint8)
        if frame_history > 1:
            self._obs_deque = deque(maxlen=frame_history)
        self._frame_history = frame_history
        self._action_repeat = action_repeat

    def reset(self):
        self.dmlab_env.reset()
        for _ in range(self._frame_history - 1):
            self._obs_deque.append(self._zero_obs)
        obs, _ = self.update_obs()
        return obs

    def step(self, action):
        reward = self.dmlab_env.step(self._action_map[action],
            num_steps=self._action_repeat)
        obs, done = self.update_obs()
        return obs, reward, done, ()  # Might need to make dummy namedtuple?

    def update_obs(self):
        done = not self.dmlab_env.is_running()
        obs = self._zero_obs if done else self.dmlab_env.observations()["RGB"]
        if self._frame_history > 1:
            self._obs_deque.append(obs)
            obs = np.concatenate(self._obs_deque)  # OLDEST to NEWEST
        return obs, done

    def close(self):
        self.dmlab_env.close()


class LevelCache(object):
    """Copied from DMLab documentation."""

    def __init__(self, cache_dir):
        self._cache_dir = cache_dir

    def fetch(self, key, pk3_path):
        path = osp.join(self._cache_dir, key)

        if osp.isfile(path):
            # Copy the cached file to the path expected by DeepMind Lab.
            shutil.copyfile(path, pk3_path)
            return True
        return False

    def write(self, key, pk3_path):
        path = osp.join(self._cache_dir, key)

        if not osp.isfile(path):
            # Copy the cached file DeepMind Lab has written to the cache directory.
            shutil.copyfile(pk3_path, path)


DMLAB30 = [
    "rooms_collect_good_objects_train",
    "rooms_collect_good_objects_test",
    "rooms_exploit_deferred_effects_train",
    "rooms_exploit_deferred_effects_test",
    "rooms_select_nonmatching_object",
    "rooms_watermaze",
    "rooms_keys_doors_puzzle",
    "language_select_described_object",
    "language_select_located_object",
    "language_execute_random_task",
    "language_answer_quantitative_question",
    "lasertag_one_opponent_small",
    "lasertag_three_opponents_small",
    "lasertag_one_opponent_large",
    "lasertag_three_opponents_large",
    "natlab_fixed_large_map",
    "natlab_varying_map_regrowth",
    "natlab_varying_map_randomized",
    "skymaze_irreversible_path_hard",
    "skymaze_irreversible_path_varied",
    "psychlab_arbitrary_visuomotor_mapping",
    "psychlab_continuous_recognition",
    "psychlab_sequential_comparison",
    "psychlab_visual_search",
    "explore_object_locations_small",
    "explore_object_locations_large",
    "explore_obstructed_goals_small",
    "explore_obstructed_goals_large",
    "explore_goal_locations_small",
    "explore_goal_locations_large",
    "explore_object_rewards_few",
    "explore_object_rewards_many",
]

ACTION_MEANINGS = [
    "FORWARD",
    "BACKWARD",
    "MOVE LEFT",
    "MOVE RIGHT",
    "LOOK LEFT",
    "LOOK RIGHT",
    "LEFT FORWARD",
    "RIGHT FORWARD",
    "FIRE",
]
