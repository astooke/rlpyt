
import numpy as np

from rlpyt.samplers.base import BaseCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example
from rlpyt.utils.logging import logger


class DecorrelatingStartCollector(BaseCollector):

    def start_envs(self, max_decorrelation_steps=0):
        """Calls reset() on every env and returns agent_inputs buffer."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, obs in enumerate(observations):
            observation[b] = obs  # numpy array or namedarraytuple
        prev_action = self.envs[0].action_space.sample(len(self.envs), null=True)
        prev_reward = np.zeros(len(self.envs), dtype="float32")
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}")
        if max_decorrelation_steps == 0:
            return AgentInputs(observation, prev_action, prev_reward), traj_infos
        for b, env in enumerate(self.envs):
            n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
            env_actions = env.action_space.sample(n_steps)
            for a in env_actions:
                o, r, d, info = env.step(a)
                traj_infos[b].step(o, a, r, None, info)
                if getattr(info, "need_reset", d):  # For episodic lives.
                    o = env.reset()
                    a = env.action_space.sample(null=True)
                    r = 0
                    traj_infos[b] = self.TrajInfoCls()
            observation[b] = o
            prev_action[b] = a
            prev_reward[b] = r
        return AgentInputs(observation, prev_action, prev_reward), traj_infos
