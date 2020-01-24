
import torch
import multiprocessing as mp
import ctypes
import psutil

from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.async_.base import AsyncParallelSamplerMixin
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.gpu.sampler import GpuSamplerBase, build_step_buffer
from rlpyt.samplers.async_.collectors import DbGpuResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuEvalCollector
from rlpyt.samplers.async_.action_server import AsyncActionServer
from rlpyt.samplers.parallel.worker import sampling_process
from rlpyt.utils.logging import logger
from rlpyt.utils.seed import make_seed
from rlpyt.utils.collections import AttrDict


class AsyncGpuSamplerBase(AsyncParallelSamplerMixin, ParallelSamplerBase):
    """Main definitions for asynchronous parallel sampler using GPU(s) for
    action selection.  The main sampler process (forked from the overall
    master), forks action-server processes, one per GPU to be used, and
    the action-server process(es) fork their own parallel CPU workers.
    This same sampler object is used in the main sampler process and in
    the action server process(es), but for different methods, labeled by
    comments in the code (easier way to pass arguments along).
    """

    def __init__(self, *args, CollectorCls=DbGpuResetCollector,
            eval_CollectorCls=GpuEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    ##########################################
    # In forked sampler runner process.
    ##########################################

    def initialize(self, affinity):
        """Initialization inside the main sampler process.  Builds one level
        of parallel synchronization objects, and forks action-server processes,
        one per GPU to be used.
        """
        torch.set_num_threads(1)  # Needed to avoid MKL hang :( .
        self.world_size = n_server = len(affinity)
        n_envs_lists = self._get_n_envs_lists(affinity)
        n_server = len(n_envs_lists)
        n_worker = sum([len(n_envs_list) for n_envs_list in n_envs_lists])
        self.n_worker = n_worker

        if self.eval_n_envs > 0:
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = eval_max_T = int(self.eval_max_steps // eval_n_envs)

        self._build_parallel_ctrl(n_server, n_worker)

        servers_kwargs = self._assemble_servers_kwargs(affinity, self.seed,
            n_envs_lists)
        servers = [mp.Process(target=self.action_server_process,
            kwargs=s_kwargs)
            for s_kwargs in servers_kwargs]
        for s in servers:
            s.start()
        self.servers = servers
        self.ctrl.barrier_out.wait()  # Wait for workers to decorrelate envs.

    # obtain_samples() and evaluate_agent() remain the same.

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for s in self.servers:
            s.join()

    def _get_n_envs_lists(self, affinity):
        B = self.batch_spec.B
        n_server = len(affinity)
        n_workers = [len(aff["workers_cpus"]) for aff in affinity]
        if B < n_server:
            raise ValueError(f"Request fewer envs ({B}) than action servers "
                f"({n_server}).")
        server_Bs = [B // n_server] * n_server
        if n_workers.count(n_workers[0]) != len(n_workers):
            logger.log("WARNING: affinity requested different number of "
                "environment workers per action server, but environments "
                "will be assigned equally across action servers anyway.")
        if B % n_server > 0:
            for s in range(B % n_server):
                server_Bs[s] += 1  # Spread across action servers.

        n_envs_lists = list()
        for s_worker, s_B in zip(n_workers, server_Bs):
            n_envs_lists.append(self._get_n_envs_list(n_worker=s_worker, B=s_B))

        return n_envs_lists

    def _build_parallel_ctrl(self, n_server, n_worker):
        super()._build_parallel_ctrl(n_worker + n_server)

    def _assemble_servers_kwargs(self, affinity, seed, n_envs_lists):
        servers_kwargs = list()
        i_env = 0
        i_worker = 0
        for rank in range(len(affinity)):
            n_worker = len(affinity[rank]["workers_cpus"])
            n_env = sum(n_envs_lists[rank])
            slice_B = slice(i_env, i_env + n_env)
            server_kwargs = dict(
                rank=rank,
                env_ranks=list(range(i_env, i_env + n_env)),
                double_buffer_slice=tuple(buf[:, slice_B]
                    for buf in self.double_buffer),
                affinity=affinity[rank],
                n_envs_list=n_envs_lists[rank],
                seed=seed + i_worker,
            )
            servers_kwargs.append(server_kwargs)
            i_worker += n_worker
            i_env += n_env
        return servers_kwargs

    ############################################
    # In action server processes (forked again).
    ############################################

    def action_server_process(self, rank, env_ranks, double_buffer_slice,
            affinity, seed, n_envs_list):
        """Target method used for forking action-server process(es) from the
        main sampler process.  By inheriting the sampler object from the
        sampler process, can more easily pass args to the environment worker
        processes, which are forked from here.

        Assigns hardware affinity, and then forks parallel worker processes
        and moves agent model to device.  Then enters infinite loop: waits for
        signals from main sampler process to collect training samples or
        perform evaluation, and then serves actions during collection.  At
        every loop, calls agent to retrieve new parameter values from the
        training process, which are communicated through shared CPU memory.
        """
        self.rank = rank
        p = psutil.Process()
        if affinity.get("set_affinity", True):
            p.cpu_affinity(affinity["master_cpus"])
        # torch.set_num_threads(affinity["master_torch_threads"])
        torch.set_num_threads(1)  # Possibly needed to avoid MKL hang.
        self.launch_workers(double_buffer_slice, affinity, seed, n_envs_list)
        self.agent.to_device(cuda_idx=affinity["cuda_idx"])
        self.agent.collector_initialize(global_B=self.batch_spec.B,  # Not updated.
            env_ranks=env_ranks)  # For vector eps-greedy.
        self.ctrl.barrier_out.wait()  # Wait for workers to decorrelate envs.
        
        while True:
            self.sync.stop_eval.value = False  # Reset.
            self.ctrl.barrier_in.wait()
            if self.ctrl.quit.value:
                break
            self.agent.recv_shared_memory()
            if self.ctrl.do_eval.value:
                self.agent.eval_mode(self.ctrl.itr.value)
                self.serve_actions_evaluation(self.ctrl.itr.value)
            else:
                self.agent.sample_mode(self.ctrl.itr.value)
                # Only for bootstrap_value:
                self.samples_np = self.double_buffer[self.ctrl.db_idx.value]
                if hasattr(self, "double_bootstrap_value_pair"):  # Alternating sampler.
                    self.bootstrap_value_pair = \
                        self.double_bootstrap_value_pair[self.ctrl.db_idx.value]
                self.serve_actions(self.ctrl.itr.value)
            self.ctrl.barrier_out.wait()
        self.shutdown_workers()

    def launch_workers(self, double_buffer_slice, affinity, seed, n_envs_list):
        self.n_worker = n_worker = len(n_envs_list)
        # A little slight-of-hand to make 2-level signal:
        self.ctrl.stop_eval = self.sync.stop_eval
        self.sync = AttrDict(
            obs_ready=[mp.Semaphore(0) for _ in range(n_worker)],
            act_ready=[mp.Semaphore(0) for _ in range(n_worker)],
            stop_eval=mp.RawValue(ctypes.c_bool, False),  # Overwrite.
            # stop_eval=self.ctrl.stop_eval,  # No, make 2-level signal.
            db_idx=self.ctrl.db_idx,  # Copy into sync which passes to Collector.
        )
        self.step_buffer_pyt, self.step_buffer_np = build_step_buffer(
            self.examples, sum(n_envs_list))
        self.agent_inputs = AgentInputs(self.step_buffer_pyt.observation,
            self.step_buffer_pyt.action, self.step_buffer_pyt.reward)

        if self.eval_n_envs > 0:
            eval_n_envs = self.eval_n_envs_per * n_worker
            eval_step_buffer_pyt, eval_step_buffer_np = build_step_buffer(
                self.examples, eval_n_envs)
            self.eval_step_buffer_pyt = eval_step_buffer_pyt
            self.eval_step_buffer_np = eval_step_buffer_np
            self.eval_agent_inputs = AgentInputs(
                self.eval_step_buffer_pyt.observation,
                self.eval_step_buffer_pyt.action,
                self.eval_step_buffer_pyt.reward,
            )
            # eval_max_T already made in earlier initialize.

        self.double_buffer = double_buffer_slice  # Now only see my part.
        common_kwargs = self._assemble_common_kwargs(affinity)
        common_kwargs["agent"] = None  # Remove.
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed,
            n_envs_list)

        # Yes, fork again.
        self.workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in self.workers:
            w.start()

    def shutdown_workers(self):
        for w in self.workers:
            w.join()  # Already signaled to quit by central master.

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = super()._assemble_workers_kwargs(affinity, seed,
            n_envs_list)
        i_env = 0
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["sync"] = AttrDict(
                stop_eval=self.sync.stop_eval,
                obs_ready=self.sync.obs_ready[rank],
                act_ready=self.sync.act_ready[rank],
                db_idx=self.sync.db_idx,
            )
            w_kwargs["step_buffer_np"] = self.step_buffer_np[slice_B]
            if self.eval_n_envs > 0:
                eval_slice_B = slice(self.eval_n_envs_per * rank,
                    self.eval_n_envs_per * (rank + 1))
                w_kwargs["eval_step_buffer_np"] = \
                    self.eval_step_buffer_np[eval_slice_B]
            i_env += n_envs
        return workers_kwargs


class AsyncGpuSampler(AsyncActionServer, AsyncGpuSamplerBase):
    pass
