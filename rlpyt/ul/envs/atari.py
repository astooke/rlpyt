
import numpy as np
import cv2
import atari_py
import os

from rlpyt.spaces.int_box import IntBox
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.utils.quick_args import save__init__args


class AtariEnv84(AtariEnv):
    """
    Same as built-in AtariEnv except returns standard 84x84 frames.

    Actually, can resize the image to whatever square size you want.
    """

    def __init__(self,
                 game="pong",
                 frame_skip=4,
                 num_img_obs=4,
                 clip_reward=True,
                 episodic_lives=False,  # !
                 max_start_noops=30,
                 repeat_action_probability=0.25,  # !
                 horizon=27000,
                 obs_size=84,  # square resize
                 fire_on_reset=True,
                 ):
        save__init__args(locals(), underscore=True)
        # ALE
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game {} but path {} does not "
                " exist".format(game, game_path))
        self.ale = atari_py.ALEInterface()
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.loadROM(game_path)

        # Spaces
        self._obs_size = obs_size
        self._action_set = self.ale.getMinimalActionSet()
        self._action_space = IntBox(low=0, high=len(self._action_set))
        obs_shape = (num_img_obs, self._obs_size, self._obs_size)
        self._observation_space = IntBox(low=0, high=256, shape=obs_shape,
            dtype="uint8")
        self._max_frame = self.ale.getScreenGrayscale()
        self._raw_frame_1 = self._max_frame.copy()
        self._raw_frame_2 = self._max_frame.copy()
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")
        
        # Settings
        self._has_fire = "FIRE" in self.get_action_meanings()
        self._has_up = "UP" in self.get_action_meanings()
        self._horizon = int(horizon)
        self.reset()

    def reset(self):
        """Performs hard reset of ALE game."""
        self.ale.reset_game()
        self._reset_obs()
        self._life_reset()
        for _ in range(np.random.randint(0, self._max_start_noops + 1)):
            self.ale.act(0)
        if self._fire_on_reset:
            self.fire_and_up()
        self._update_obs()  # (don't bother to populate any frame history)
        self._step_counter = 0
        return self.get_obs()

    def _update_obs(self):
        """Max of last two frames; resize to standard 84x84."""
        self._get_screen(2)
        np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
        img = cv2.resize(self._max_frame, (self._obs_size, self._obs_size), cv2.INTER_AREA)
        # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
        self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])

    def _life_reset(self):
        self.ale.act(0)  # (advance from lost life state)
        self._lives = self.ale.lives()

    def fire_and_up(self):
        if self._has_fire:
            # TODO: for sticky actions, make sure fire is actually pressed
            self.ale.act(1)  # (e.g. needed in Breakout, not sure what others)
        if self._has_up:
            self.ale.act(2)  # (not sure if this is necessary, saw it somewhere)
