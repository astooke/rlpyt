
from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import build_samples_buffer
from rlpyt.utils.logging import logger
from rlpyt.samplers.collectors import SerialEvalCollector


class SerialSampler(BaseSampler):
    """Uses same functionality as ParallelSampler but does not fork worker
    processes; can be easier for debugging (e.g. breakpoint() in master).  Use
    with collectors which sample actions themselves (e.g. under cpu
    category)."""

    def initialize(self, agent, affinity=None, seed=None,
            bootstrap_value=False, traj_info_kwargs=None):
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(self.batch_spec.B)]
        agent.initialize(envs[0].spec, share_memory=False)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in self.eval_n_envs]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Serial Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        self.samples_np[:] = 0
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        agent_inputs = self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        return self.eval_collector.collect_evaluation(itr)
