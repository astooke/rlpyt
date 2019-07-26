
from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import build_samples_buffer
from rlpyt.samplers.collectors import SerialEvalCollector
from rlpyt.utils.seed import make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict


class AsyncSerialSampler(BaseSampler):

    ###########################################################################
    # Master runner methods.
    ###########################################################################

    def master_runner_initialize(self, agent, bootstrap_value=False,
            traj_info_kwargs=None, seed=None):
        self.seed = make_seed() if seed is None else seed
        env = self.EnvCls(**self.env_kwargs)
        agent.initialize(env.spaces, share_memory=True,
            global_B=self.batch_spec.B, env_ranks=list(range(self.batch_spec.B)))
        _, samples_np, examples = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True, env_shared=True,
            subprocess=False)  # Would like subprocess=True, but might hang?
        _, samples_np2, _ = build_samples_buffer(agent, env, self.batch_spec,
            bootstrap_value, agent_shared=True, env_shared=True, subprocess=False)
        env.close()
        del env
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)
        self.double_buffer = double_buffer = (samples_np, samples_np2)
        self.examples = examples
        self.agent = agent
        return double_buffer, examples

    ###########################################################################
    # Sampler runner methods (forked).
    ###########################################################################

    def sampler_process_initialize(self, affinity):
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        sync = AttrDict(j=AttrDict(value=0))  # Mimic the mp.RawValue format.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=self.double_buffer,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=self.agent,
            sync=sync
        )
        if self.eval_n_envs > 0:
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=self.agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )
        self.agent.initialize_device(cuda_idx=affinity.get("cuda_idx", None),
            ddp=False)

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        self.sync = sync
        logger.log("Serial sampler initialized.")

    def obtain_samples(self, itr, j):
        self.sync.j.value = j  # Tell the collector which buffer.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return traj_infos

    def evaluate_agent(self, itr):
        return self.eval_collector.collect_evaluation(itr)
