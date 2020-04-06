import unittest

import gym
import numpy as np

from rlpyt.spaces.gym_wrapper import GymSpaceWrapper


class TestGymWrapper(unittest.TestCase):
    def test_seed(self):
        space = GymSpaceWrapper(gym.spaces.Box(low=np.zeros(1), high=np.ones(1)))
        space.seed(0)
        sample_1 = space.sample()
        space.seed(0)
        sample_2 = space.sample()
        self.assertEqual(sample_1, sample_2)

        sample_3 = space.sample()
        self.assertNotEqual(sample_1, sample_3)


if __name__ == "__main__":
    unittest.main()
