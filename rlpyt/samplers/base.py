

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
            CollectorCls=None,  # Not auto-populated.
            eval_n_envs=0,  # 0 for no eval setup.
            eval_CollectorCls=None,  # Maybe auto-populated.
            eval_env_kwargs=None,  # Must supply if doing eval.
            eval_max_steps=None,  # int if using evaluation.
            eval_max_trajectories=None,  # Optional earlier cutoff.
            eval_min_envs_reset=1,
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


class BaseEvalCollector(object):
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
