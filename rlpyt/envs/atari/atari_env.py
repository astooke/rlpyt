
import numpy as np
import os
import atari_py
import cv2
from collections import namedtuple

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.discrete import Discrete
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save_args


W, H = (80, 104)  # Fixed size: crop two rows, then downsample by 2x.


EnvInfo = namedtuple("EnvInfo", ["raw_reward", "need_reset"])


class AtariEnv(Env):

    def __init__(self,
                 game="pong",
                 frame_skip=4,
                 num_img_obs=4,
                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 ):
        save_args(locals(), underscore=True)
        # ALE
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game {} but path {} does not "
                " exist".format(game, game_path))
        self.ale = atari_py.ALEInterface()
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.loadROM(game_path)

        # Spaces
        self._action_set = self.ale.getMinimalActionSet()
        self._action_space = Discrete(len(self._action_set),
            null_value=ACTION_INDEX["NOOP"])
        obs_shape = (num_img_obs, H, W)
        self._observation_space = IntBox(shape=obs_shape, dtype="uint8")
        self._max_frame = self.ale.getScreenGrayscale()
        self._raw_frame_1 = self._max_frame.copy()
        self._raw_frame_2 = self._max_frame.copy()
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")

        # Settings
        self._has_fire = "FIRE" in self.get_action_meanings()
        self._has_up = "UP" in self.get_action_meanings()
        self._done = self._done_episodic_lives if episodic_lives else \
            self._done_no_epidosic_lives

        # Get ready
        self.reset()

    def step(self, action):
        a = self._action_set[action]
        raw_reward = np.array(0., dtype="float32")
        for _ in range(self._frame_skip - 1):
            raw_reward += self.ale.act(a)
        self._get_screen(1)
        raw_reward += self.ale.act(a)
        self._update_obs()
        reward = np.sign(raw_reward) if self._clip_reward else raw_reward
        done, need_reset = self._done()
        info = EnvInfo(raw_reward, need_reset)
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10, show_full_obs=False):
        img = self.get_obs()
        if show_full_obs:
            shape = img.shape
            img = img.reshape(shape[0] * shape[1], shape[2])
        else:
            img = img[0]
        cv2.imshow(self._game, img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    def reset(self):
        self.ale.reset_game()
        self._reset_obs()
        self._life_reset()
        for _ in range(np.random.randint(0, self._max_start_noops + 1)):
            self.ale.act(0)
        self._update_obs()  # (don't bother to populate any frame history)
        return self.get_obs()

    ###########################################################################
    # Helpers

    def _get_screen(self, frame=1):
        frame = self._raw_frame_1 if frame == 1 else self._raw_frame_2
        self.ale.getScreenGrayscale(frame)

    def _update_obs(self):
        """Max of last two frames; crop two rows; downsample by 2x."""
        self._get_screen(2)
        np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
        img = cv2.resize(self._max_frame[1:-1], (W, H), cv2.INTER_NEAREST)
        # NOTE: this order--oldest to newest--needed for ReplayFrameBuffer in DQN!!
        self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])

    def _reset_obs(self):
        self._obs[:] = 0
        self._max_frame[:] = 0
        self._raw_frame_1[:] = 0
        self._raw_frame_2[:] = 0

    def _check_life(self):
        lives = self.ale.lives()
        lost_life = (lives < self._lives) and (lives > 0)
        if lost_life:
            self._life_reset()
        return lost_life

    def _life_reset(self):
        self.ale.act(0)  # (advance from lost life state)
        if self._has_fire:
            # TODO: for sticky actions, make sure fire is actually pressed
            self.ale.act(1)  # (e.g. needed in Breakout, not sure what others)
        if self._has_up:
            self.ale.act(2)  # (not sure if this is necessary, saw it somewhere)
        self._lives = self.ale.lives()

    def _done_no_epidosic_lives(self):
        self._check_life()
        done = self.ale.game_over()
        return done, done

    def _done_episodic_lives(self):
        need_reset = self.ale.game_over()
        lost_life = self._check_life()
        if lost_life:
            self._reset_obs()  # (reset here, so sampler does NOT call reset)
            self._update_obs()  # (will have already advanced in check_life)
        return lost_life or need_reset, need_reset

    ###########################################################################
    # Properties

    @property
    def game(self):
        return self._game

    @property
    def frame_skip(self):
        return self._frame_skip

    @property
    def num_img_obs(self):
        return self._num_img_obs

    @property
    def clip_reward(self):
        return self._clip_reward

    @property
    def max_start_noops(self):
        return self._max_start_noops

    @property
    def episodic_lives(self):
        return self._episodic_lives

    @property
    def repeat_action_probability(self):
        return self._repeat_action_probability

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_INDEX = {v: k for k, v in ACTION_MEANING.items()}
