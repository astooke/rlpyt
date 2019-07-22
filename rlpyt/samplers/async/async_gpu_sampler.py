
import time
import numpy as np
import multiprocessing as mp
import ctypes

from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import build_samples_buffer, build_step_buffer
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.samplers.gpu.collectors import EvalCollector
from rlpyt.utils.logging import logger
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.collections import AttrDict


EVAL_TRAJ_CHECK = 0.1  # Seconds.


class AsyncGpuSampler(BaseSampler):

    ###########################################################################
    # Master runner methods.
    ###########################################################################

    def master_runner_initialize(self, agent, bootstrap_value=False,
            traj_info_kwargs=None):
        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        agent.initialize(env.spaces, share_memory=True)  # Actual agent initialization, keep.
        samples_pyt, samples_np, examples = build_samples_buffer(agent, env,
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
        return double_buffer, examples

    ###########################################################################
    # Sampler runner methods (forked).
    ###########################################################################

    def sampler_process_initialize(self, affinity):
        n_server = len(affinity)
        n_worker = sum(len(aff["workers_cpus"]) for aff in affinity)
        n_envs_list = [self.batch_spec.B // n_worker] * n_worker
        if not self.batch_spec.B % n_worker == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                f"batch_B {self.batch_spec.B} and n_parallel {n_worker} "
                "(possible suboptimal speed).")
            for b in range(self.batch_spec.B % n_worker):
                n_envs_list[b] += 1

        if self.eval_n_envs > 0:
            eval_n_envs_per = max(1, self.eval_n_envs // len(n_envs_list))
            eval_n_envs = eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = 1 + int(self.eval_max_steps // eval_n_envs)
            self.eval_n_envs_per = eval_n_envs_per
        else:
            self.eval_n_envs_per = 0
            self.eval_max_T = 0

        ctrl = AttrDict(
            quit=mp.RawValue(ctypes.c_bool, False),
            barrier_in=mp.Barrier(n_server + n_worker + 1),
            barrier_out=mp.Barrier(n_server + n_worker + 1),
            do_eval=mp.RawValue(ctypes.c_bool, False),
            itr=mp.RawValue(ctypes.c_long, 0),
            j=mp.RawValue("i", 0),  # Double buffer index.
        )
        traj_infos_queue = mp.Queue()

        common_kwargs = dict(
            ctrl=ctrl,
            traj_infos_queue=traj_infos_queue,
            world_size=n_server,
        )
        servers_kwargs = assemble_servers_kwargs(affinity, n_envs_list,
            self.seed, self.double_buffer)
        servers = [mp.Process(target=self.action_server_process,
            kwargs=s_kwargs.update(**common_kwargs))
            for s_kwargs in servers_kwargs]
        for s in servers:
            s.start()
        self.servers = servers
        self.ctrl = ctrl
        self.traj_infos_queue = traj_infos_queue

    def obtain_samples(self, itr, j):
        self.ctrl.itr.value = itr
        self.ctrl.j.value = j  # Tell collectors which buffer to use.
        self.ctrl.barrier_in.wait()
        # Sampling in sub-processes here.
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        return traj_infos

    def evaluate_agent(self, itr):
        self.ctrl.do_eval = True
        self.sync.stop_eval.value = False
        self.ctrl.barrier_in.wait()
        traj_infos = list()
        if self.eval_max_trajectories is not None:
            while True:
                time.sleep(EVAL_TRAJ_CHECK)
                while self.traj_infos_queue.qsize():
                    traj_infos.append(self.traj_infos_queue.get())
                if len(traj_infos) >= self.eval_max_trajectories:
                    self.sync.stop_eval.value = True
                    logger.log("Evaluation reached max num trajectories "
                        f"({self.eval_max_trajectories}).")
                    break  # Stop possibly before workers reach max_T.
                if self.ctrl.barrier_out.parties - self.ctrl.barrier_out.n_waiting == 1:
                    logger.log("Evaluation reached max num time steps "
                        f"({self.eval_max_T}).")
                    break  # Workers reached max_T.
        self.ctrl.barrier_out.wait()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        self.ctrl.do_eval.value = False
        return traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for s in self.servers:
            s.join()

    ###########################################################################
    # Methods in forked action server process.
    ###########################################################################

    def action_server_process(self, rank, env_ranks, double_buffer_slice, ctrl,
            traj_infos_queue, affinity, seed, n_envs_list):
        """Runs in forked process, inherits from original process, so can easily
        pass args to env worker processes, forked from here."""
        self.rank = rank
        self.ctrl = ctrl
        self.launch_workers(double_buffer_slice, traj_infos_queue, affinity,
            seed, n_envs_list)
        self.agent.initialize_device(cuda_idx=affinity["cuda_idx"], ddp=False)
        self.agent.collector_initialize(global_B=self.batch_spec.B,  # Not updated.
            env_ranks=env_ranks)  # For vector eps-greedy.
        while True:
            ctrl.barrier_in.wait()
            if self.ctrl.quit.value:
                break
            self.agent.recv_shared_memory()
            if ctrl.do_eval.value:
                self.agent.eval_mode(self.ctrl.itr.value)
                self.serve_actions_evaluation()
            else:
                self.agent.sample_mode(self.ctrl.itr.value)
                self.serve_actions()
            ctrl.barrier_out.wait()
        self.shutdown_workers()

    def serve_actions(self):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_np, step_pyt = self.step_buffer_np, self.step_buffer_pyt

        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.batch_spec.T):
            for b in step_blockers:
                b.acquire()  # Workers written obs and rew, first prev_act.
            if self.mid_batch_reset and np.any(step_np.done):
                for b_reset in np.where(step_np.done)[0]:
                    step_np.action[b_reset] = 0  # Null prev_action into agent.
                    step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                    self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action  # Worker applies to env.
            step_np.agent_info[:] = agent_info  # Worker sends to traj_info.
            for w in act_waiters:
                w.release()  # Signal to worker.

        for b in step_blockers:
            b.acquire()
        if "bootstrap_value" in self.samples_np.agent:
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(
                *agent_inputs)
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                self.agent.reset_one(idx=b_reset)
            # step_np.done[:] = False  # DON'T DO. Worker resets later.

    def serve_actions_evaluation(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_np, step_pyt = self.eval_step_buffer_np, self.eval_step_buffer_pyt
        self.agent.reset()
        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.eval_max_T):
            for b in step_blockers:
                b.acquire()
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action.
                step_np.reward[b_reset] = 0  # Null prev_reward.
                self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action
            step_np.agent_info[:] = agent_info
            for w in act_waiters:
                w.release()
            if self.sync.stop_eval.value:
                break  # TODO: Double-check where this goes relative to semas.
        for b in step_blockers:
            b.acquire()  # Workers always do extra release; drain it.

    def launch_workers(self, double_buffer, traj_infos_queue, affinity,
            seed, n_envs_list, eval_n_envs_per):
        n_worker = len(affinity["workers_cpus"])
        sync = AttrDict(
            step_blockers=[mp.Semaphore(0) for _ in range(n_worker)],
            act_waiters=[mp.Semaphore(0) for _ in range(n_worker)],
            stop_eval=mp.RawValue(ctypes.c_bool, False),
            j=self.ctrl.j,  # Copy into sync which passes to Collector.
        )
        step_buffer_pyt, step_buffer_np = build_step_buffer(self.examples,
            sum(n_envs_list))

        if self.eval_n_envs_per > 0:
            eval_n_envs = self.eval_n_envs_per * n_worker
            eval_step_buffer_pyt, eval_step_buffer_np = build_step_buffer(
                self.examples, eval_n_envs)
            self.eval_step_buffer_pyt = eval_step_buffer_pyt
            self.eval_step_buffer_np = eval_step_buffer_np
        else:
            eval_step_buffer_np = None

        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            agent=None,
            batch_T=self.batch_spec.T,
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=traj_infos_queue,
            ctrl=self.ctrl,
            max_decorrelation_steps=self.max_decorrelation_steps,
            eval_n_envs=self.eval_n_envs_per,
            eval_CollectorCls=self.eval_CollectorCls or EvalCollector,
            eval_env_kwargs=self.eval_env_kwargs,
            eval_max_T=self.eval_max_T,
            )
        workers_kwargs = assemble_workers_kwargs(affinity, seed, double_buffer,
            n_envs_list, step_buffer_np, sync, self.eval_n_envs_per,
            eval_step_buffer_np)

        workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.workers = workers
        self.step_buffer_pyt = step_buffer_pyt
        self.step_buffer_np = step_buffer_np
        self.sync = sync
        self.mid_batch_reset = self.CollectorCls.mid_batch_reset

    def shutdown_workers(self):
        for w in self.workers:
            w.join()  # Already signaled to quit by central master.


def assemble_servers_kwargs(double_buffer, affinity, n_envs_list, seed):
    servers_kwargs = list()
    i_env = 0
    i_worker = 0
    for rank in range(len(affinity)):
        n_worker = len(affinity["workers_cpus"])
        n_env = sum(n_envs_list[i_worker:i_worker + n_worker])
        slice_B = slice(i_env, i_env + n_env)
        server_kwargs = dict(
            rank=rank,
            env_ranks=list(range(i_env, i_env + n_env)),
            double_buffer_slice=tuple(buf[:, slice_B] for buf in double_buffer),
            affinity=affinity[rank],
            n_envs_list=n_envs_list[i_worker:i_worker + n_worker],
            seed=seed + i_worker,
        )
        servers_kwargs.append(server_kwargs)
        i_worker += n_worker
        i_env += n_env
    return servers_kwargs


def assemble_workers_kwargs(affinity, seed, double_buffer, n_envs_list,
        step_buffer_np, sync, eval_n_envs, eval_step_buffer_np):
    workers_kwargs = list()
    i_env = 0
    for rank in range(len(affinity["workers_cpus"])):
        n_envs = n_envs_list[rank]
        slice_B = slice(i_env, i_env + n_envs)
        w_sync = AttrDict(
            step_blocker=sync.step_blockers[rank],
            act_waiter=sync.act_waiters[rank],
            stop_eval=sync.stop_eval,
            j=sync.j,
        )
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinity["workers_cpus"][rank],
            n_envs=n_envs,
            samples_np=tuple(buf[:, slice_B] for buf in double_buffer),
            step_buffer_np=step_buffer_np[slice_B],
            sync=w_sync,
        )
        i_env += n_envs
        if eval_n_envs > 0:
            eval_slice_B = slice(rank * eval_n_envs, (rank + 1) * eval_n_envs)
            worker_kwargs["eval_step_buffer_np"] = eval_step_buffer_np[eval_slice_B]
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
