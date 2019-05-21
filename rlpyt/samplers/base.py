

from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.utils.quick_args import save__init__args


class BaseSampler(object):
    """Class which interfaces with the Runner, in master process only."""

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_T,
            batch_B,
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            CollectorCls=None,
            ):
        save__init__args(locals())
        self.batch_spec = BatchSpec(batch_T, batch_B)
        self.mid_batch_reset = CollectorCls.mid_batch_reset

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        return NotImplementedError  # type: Samples

    def shutdown(self):
        pass


class BaseCollector(object):
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
            ):
        save__init__args(locals())

    def start_envs(self):
        """Calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        if getattr(self, "agent", None) is not None:
            self.agent.reset()
            self.agent.model.eval()  # Do once inside worker.

    def collect_batch(self, agent_inputs, traj_infos):
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        return agent_inputs

