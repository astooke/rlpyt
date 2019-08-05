

from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.utils.quick_args import save__init__args


class BaseSampler:
    """Class which interfaces with the Runner, in master process only."""

    alternating = False

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_T,
            batch_B,
            CollectorCls,
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            eval_n_envs=0,  # 0 for no eval setup.
            eval_CollectorCls=None,  # Must supply if doing eval.
            eval_env_kwargs=None,
            eval_max_steps=None,  # int if using evaluation.
            eval_max_trajectories=None,  # Optional earlier cutoff.
            ):
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        eval_max_trajectories = (None if eval_max_trajectories is None else
            int(eval_max_trajectories))
        save__init__args(locals())
        self.batch_spec = BatchSpec(batch_T, batch_B)
        self.mid_batch_reset = CollectorCls.mid_batch_reset

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        raise NotImplementedError  # type: Samples

    def evaluate_agent(self, itr):
        raise NotImplementedError

    def shutdown(self):
        pass

    @property
    def batch_size(self):
        return self.batch_spec.size  # For logging at least.
