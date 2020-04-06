import unittest

from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.envs.gym import make as gym_make
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.seed import set_seed


class TestGymWrapper(unittest.TestCase):
    def test_seed(self):
        sampler = SerialSampler(
            EnvCls=gym_make,
            env_kwargs={"id": "MountainCarContinuous-v0"},
            batch_T=1,
            batch_B=1,
        )

        agent = SacAgent(pretrain_std=0.0)
        agent.give_min_itr_learn(10000)

        set_seed(0)
        sampler.initialize(agent, seed=0)
        samples_1 = sampler.obtain_samples(0)

        set_seed(0)
        sampler.initialize(agent, seed=0)
        samples_2 = sampler.obtain_samples(0)

        # Dirty hack to compare objects containing tensors.
        self.assertEqual(str(samples_1), str(samples_2))

        samples_3 = sampler.obtain_samples(0)
        self.assertNotEqual(samples_1, samples_3)


if __name__ == "__main__":
    unittest.main()
