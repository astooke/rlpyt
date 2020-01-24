
import multiprocessing as mp
import ctypes
import time

from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.worker import sampling_process
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 0.1  # seconds.


class ParallelSamplerBase(BaseSampler):
    """Base class for samplers which use worker processes to run environment
    steps in parallel, across CPU resources.
    """

    gpu = False

    ######################################
    # API
    ######################################

    def initialize(
            self,
            agent,
            affinity,
            seed,
            bootstrap_value=False,
            traj_info_kwargs=None,
            world_size=1,
            rank=0,
            worker_process=None,
            ):
        """
        Creates an example instance of the environment for agent initialization
        (which may differ by sub-class) and to pre-allocate batch buffers, then deletes
        the environment instance.  Batch buffers are allocated on shared memory, so
        that worker processes can read/write directly to them.

        Computes the number of parallel processes based on the ``affinity``
        argument.  Forks worker processes, which instantiate their own environment
        and collector objects.  Waits for the worker process to complete all initialization
        (such as decorrelating environment states) before returning.  Barriers and other
        parallel indicators are constructed to manage worker processes.
        
        .. warning::
            If doing offline agent evaluation, will use at least one evaluation environment
            instance per parallel worker, which might increase the total
            number of evaluation instances over what was requested.  This may
            result in bias towards shorter episodes if the episode length is
            variable, and if the max number of evalution steps divided over the
            number of eval environments (`eval_max_steps /
            actual_eval_n_envs`), is not large relative to the max episode
            length.
        """
        n_envs_list = self._get_n_envs_list(affinity=affinity)
        self.n_worker = n_worker = len(n_envs_list)
        B = self.batch_spec.B
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        self.world_size = world_size
        self.rank = rank

        if self.eval_n_envs > 0:
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = eval_max_T = int(self.eval_max_steps // eval_n_envs)

        env = self.EnvCls(**self.env_kwargs)
        self._agent_init(agent, env, global_B=global_B,
            env_ranks=env_ranks)
        examples = self._build_buffers(env, bootstrap_value)
        env.close()
        del env

        self._build_parallel_ctrl(n_worker)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing every init.

        common_kwargs = self._assemble_common_kwargs(affinity, global_B)
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed, n_envs_list)

        target = sampling_process if worker_process is None else worker_process
        self.workers = [mp.Process(target=target,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in self.workers:
            w.start()

        self.ctrl.barrier_out.wait()  # Wait for workers ready (e.g. decorrelate).
        return examples  # e.g. In case useful to build replay buffer.

    def obtain_samples(self, itr):
        """Signal worker processes to collect samples, and wait until they
        finish. Workers will write directly to the pre-allocated samples
        buffer, which this method returns.  Trajectory-info objects from
        completed trajectories are retrieved from workers through a parallel
        queue object and are also returned.
        """
        self.ctrl.itr.value = itr
        self.ctrl.barrier_in.wait()
        # Workers step environments and sample actions here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        """Signal worker processes to perform agent evaluation.  If a max
        number of evaluation trajectories was specified, keep watch over the
        number of trajectories finished and signal an early end if the limit
        is reached.  Return a list of trajectory-info objects from the
        completed episodes.
        """
        self.ctrl.itr.value = itr
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.ctrl.barrier_in.wait()
        traj_infos = list()
        if self.eval_max_trajectories is not None:
            while True:
                time.sleep(EVAL_TRAJ_CHECK)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                    guard_sentinel=True))
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
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
            n_sentinel=self.n_worker))
        self.ctrl.do_eval.value = False
        return traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()

    ######################################
    # Helpers
    ######################################

    def _get_n_envs_list(self, affinity=None, n_worker=None, B=None):
        B = self.batch_spec.B if B is None else B
        n_worker = len(affinity["workers_cpus"]) if n_worker is None else n_worker
        if B < n_worker:
            logger.log(f"WARNING: requested fewer envs ({B}) than available worker "
                f"processes ({n_worker}). Using fewer workers (but maybe better to "
                "increase sampler's `batch_B`.")
            n_worker = B
        n_envs_list = [B // n_worker] * n_worker
        if not B % n_worker == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                f"batch_B {self.batch_spec.B} and n_worker {n_worker} "
                "(possible suboptimal speed).")
            for b in range(B % n_worker):
                n_envs_list[b] += 1
        return n_envs_list

    def _agent_init(self, agent, env, global_B=1, env_ranks=None):
        agent.initialize(env.spaces, share_memory=True,
            global_B=global_B, env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, env, bootstrap_value):
        self.samples_pyt, self.samples_np, examples = build_samples_buffer(
            self.agent, env, self.batch_spec, bootstrap_value,
            agent_shared=True, env_shared=True, subprocess=True)
        return examples

    def _build_parallel_ctrl(self, n_worker):
        self.ctrl = AttrDict(
            quit=mp.RawValue(ctypes.c_bool, False),
            barrier_in=mp.Barrier(n_worker + 1),
            barrier_out=mp.Barrier(n_worker + 1),
            do_eval=mp.RawValue(ctypes.c_bool, False),
            itr=mp.RawValue(ctypes.c_long, 0),
        )
        self.traj_infos_queue = mp.Queue()
        self.eval_traj_infos_queue = mp.Queue()
        self.sync = AttrDict(stop_eval=mp.RawValue(ctypes.c_bool, False))

    def _assemble_common_kwargs(self, affinity, global_B=1):
        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            agent=self.agent,
            batch_T=self.batch_spec.T,
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=self.traj_infos_queue,
            ctrl=self.ctrl,
            max_decorrelation_steps=self.max_decorrelation_steps,
            torch_threads=affinity.get("worker_torch_threads", 1),
            global_B=global_B,
        )
        if self.eval_n_envs > 0:
            common_kwargs.update(dict(
                eval_n_envs=self.eval_n_envs_per,
                eval_CollectorCls=self.eval_CollectorCls,
                eval_env_kwargs=self.eval_env_kwargs,
                eval_max_T=self.eval_max_T,
                eval_traj_infos_queue=self.eval_traj_infos_queue,
                )
            )
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = list()
        i_env = 0
        g_env = sum(n_envs_list) * self.rank
        for rank in range(len(n_envs_list)):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            env_ranks = list(range(g_env, g_env + n_envs))
            worker_kwargs = dict(
                rank=rank,
                env_ranks=env_ranks,
                seed=seed + rank,
                cpus=(affinity["workers_cpus"][rank]
                    if affinity.get("set_affinity", True) else None),
                n_envs=n_envs,
                samples_np=self.samples_np[:, slice_B],
                sync=self.sync,  # Only for eval, on CPU.
            )
            i_env += n_envs
            g_env += n_envs
            workers_kwargs.append(worker_kwargs)
        return workers_kwargs
