
import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args


class BaseCollector:
    """Class that steps through environments, possibly in worker process."""

    def __init__(
            self,
            rank,
            envs,
            samples_np,
            batch_T,
            TrajInfoCls,
            agent=None,  # Present or not, depending on collector class.
            sync=None,
            step_buffer_np=None,
            global_B=1,
            env_ranks=None,
            ):
        save__init__args(locals())

    def start_envs(self):
        """Calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        if getattr(self, "agent", None) is not None:  # Not in GPU collectors.
            self.agent.collector_initialize(
                global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
                env_ranks=self.env_ranks,
            )
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, agent_inputs, traj_infos):
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class BaseEvalCollector:
    """Does not record intermediate data."""

    def __init__(
            self,
            rank,
            envs,
            TrajInfoCls,
            traj_infos_queue,
            max_T,
            agent=None,
            sync=None,
            step_buffer_np=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self):
        raise NotImplementedError


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
        prev_action = np.stack([env.action_space.null_value()
            for env in self.envs])
        prev_reward = np.zeros(len(self.envs), dtype="float32")
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}")
        if max_decorrelation_steps == 0:
            return AgentInputs(observation, prev_action, prev_reward), traj_infos
        for b, env in enumerate(self.envs):
            n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
            for _ in range(n_steps):
                a = env.action_space.sample()
                o, r, d, info = env.step(a)
                traj_infos[b].step(o, a, r, d, None, info)
                if getattr(info, "traj_done", d):
                    o = env.reset()
                    traj_infos[b] = self.TrajInfoCls()
                if d:
                    a = env.action_space.null_value()
                    r = 0
            observation[b] = o
            prev_action[b] = a
            prev_reward[b] = r
        return AgentInputs(observation, prev_action, prev_reward), traj_infos
