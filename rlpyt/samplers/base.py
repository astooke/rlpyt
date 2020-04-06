

from rlpyt.samplers.collections import BatchSpec, TrajInfo, Samples
from rlpyt.utils.quick_args import save__init__args


class BaseSampler:
    """Class which interfaces with the Runner, in master process only.

    Args:
        EnvCls: class (or factory function) callable to instantiate an environment object
        env_kwargs (dict): keyword arguments passed to ``EnvCls()`` to instantiate
        batch_T (int): number of time-steps per sample batch
        batch_B (int): number of environment instances to run (in parallel), becomes second batch dimension
        CollectorCls: callable to instantiate the collector, which manages agent-environment interaction loop
        max_decorrelation_steps (int): if taking random number of steps before start of training, to decorrelate batch states
        TrajInfoCls: callable to instantiate object for tracking trajectory-wise info
        eval_n_envs (int): number of environment instances for agent evaluation (0 for no separate evaluation)
        eval_CollectorCls: callable to instantiate evaluation collector
        eval_env_kwargs: keyword arguments passed to ``EnvCls()`` for eval envs
        eval_max_steps: max total number of steps (time * n_envs) per evaluation call
        eval_max_trajectories: optional earlier cutoff for evaluation phase
    """

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
        """Should instantiate all components, including setup of parallel
        process if applicable."""
        raise NotImplementedError

    def obtain_samples(self, itr):
        """Execute agent-environment interactions and return data batch."""
        raise NotImplementedError  # type: Samples

    def evaluate_agent(self, itr):
        """Run offline agent evaluation, if applicable."""
        raise NotImplementedError

    def shutdown(self):
        pass

    @property
    def batch_size(self):
        return self.batch_spec.size  # For logging at least.
