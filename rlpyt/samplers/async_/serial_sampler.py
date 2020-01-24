
import psutil
import torch

from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.async_.base import AsyncSamplerMixin
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.samplers.async_.collectors import DbCpuResetCollector
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict


class AsyncSerialSampler(AsyncSamplerMixin, BaseSampler):
    """Sampler which runs asynchronously in a python process forked from the
    master (training) process, but with no further parallelism.
    """

    def __init__(self, *args, CollectorCls=DbCpuResetCollector,
            eval_CollectorCls=SerialEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    ###########################################################################
    # Sampler runner methods (forked).
    ###########################################################################

    def initialize(self, affinity):
        """Initialization inside the main sampler process.  Sets process hardware
        affinities, creates specified number of environment instances and instantiates
        the collector with them.  If applicable, does the same for evaluation
        environment instances.  Moves the agent to device (could be GPU), and 
        calls on ``agent.async_cpu()`` initialization.  Starts up collector.
        """
        p = psutil.Process()
        if affinity.get("set_affinity", True):
            p.cpu_affinity(affinity["master_cpus"])
        # torch.set_num_threads(affinity["master_torch_threads"])
        torch.set_num_threads(1)  # Needed to prevent MKL hang :( .
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        sync = AttrDict(db_idx=AttrDict(value=0))  # Mimic the mp.RawValue format.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=self.double_buffer,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=self.agent,
            sync=sync,
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
        self.agent.to_device(cuda_idx=affinity.get("cuda_idx", None))
        self.agent.async_cpu(share_memory=False)

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        self.sync = sync
        logger.log("Serial sampler initialized.")

    def obtain_samples(self, itr, db_idx):
        """First calls the agent to retrieve new parameter values from the
        training process's agent.  Then passes the double-buffer index to the
        collector and collects training sample batch.  Returns list of
        completed trajectory-info objects.
        """
        self.agent.recv_shared_memory()
        self.sync.db_idx.value = db_idx  # Tell the collector which buffer.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return completed_infos

    def evaluate_agent(self, itr):
        """First calls the agent to retrieve new parameter values from
        the training process's agent.
        """
        self.agent.recv_shared_memory()
        return self.eval_collector.collect_evaluation(itr)
