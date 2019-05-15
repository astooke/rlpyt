

from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.utils.quick_args import save_args


class BaseSampler(object):
    """Class which interfaces with the Runner, in master process only."""

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_T,
            batch_B,
            max_path_length=int(1e6),
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            CollectorCls=None,
            ):
        save_args(locals())
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
            max_path_length,
            TrajInfoCls,
            agent=None,  # Present depending on collector class.
            sync=None,
            step_buf=None,
            ):
        save_args(locals())
        self.horizon = len(samples_np.env.reward)  # Time major.

    def start_envs(self):
        """Calls reset() on every env."""
        raise NotImplementedError

    def collect_batch(self, agent_input, traj_infos):
        raise NotImplementedError

    def reset_if_needed(self, agent_input):
        """Reset agent and or env as needed, between batches."""
        pass

