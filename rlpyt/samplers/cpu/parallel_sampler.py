
import multiprocessing as mp
import time


from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import build_samples_buffer, build_par_objs
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.samplers.cpu.collectors import EvalCollector
from rlpyt.utils.logging import logger
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 1  # Seconds.


class CpuParallelSampler(BaseSampler):

    def initialize(
            self,
            agent,
            affinity,
            seed,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
            ):
        B = self.batch_spec.B
        n_parallel = len(affinity["workers_cpus"])
        n_envs_list = [B // n_parallel] * n_parallel
        if not B % n_parallel == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                f"batch_B {self.batch_spec.B} and n_parallel {n_parallel} "
                "(possibly suboptimal speed).")
            for b in range(B % n_parallel):
                n_envs_list[b] += 1

        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(env.spaces, share_memory=True,  # Actual agent initialization.
            global_B=global_B, env_ranks=env_ranks)  # Maybe overridden in worker.
        samples_pyt, samples_np, examples = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True, env_shared=True,
            subprocess=True)
        env.close()
        del env

        ctrl, traj_infos_queue, sync = build_par_objs(n_parallel)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.

        if self.eval_n_envs > 0:
            # assert self.eval_n_envs % n_parallel == 0
            eval_n_envs_per = max(1, self.eval_n_envs // n_parallel)
            eval_n_envs = eval_n_envs_per * n_parallel
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}")
            self.eval_max_T = eval_max_T = self.eval_max_steps // eval_n_envs
        else:
            eval_n_envs_per = 0
            eval_max_T = None

        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            agent=agent,
            batch_T=self.batch_spec.T,
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=traj_infos_queue,
            ctrl=ctrl,
            max_decorrelation_steps=self.max_decorrelation_steps,
            torch_threads=affinity.get("worker_torch_threads", None),
            eval_n_envs=eval_n_envs_per,
            eval_CollectorCls=self.eval_CollectorCls or EvalCollector,
            eval_env_kwargs=self.eval_env_kwargs,
            eval_max_T=eval_max_T,
            global_B=global_B,
        )

        workers_kwargs = assemble_workers_kwargs(affinity, seed, samples_np,
            n_envs_list, sync)

        workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.workers = workers
        self.ctrl = ctrl
        self.traj_infos_queue = traj_infos_queue
        self.sync = sync
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.agent = agent

        self.ctrl.barrier_out.wait()  # Wait for workers to decorrelate envs.
        return examples  # e.g. In case useful to build replay buffer.

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # New weights in workers, if needed.
        # self.samples_np[:] = 0  # Reset all batch sample values (optional?).
        self.ctrl.itr.value = itr
        self.ctrl.barrier_in.wait()
        # Workers step environments and sample actions here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        self.agent.sync_shared_memory()
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.ctrl.itr.value = itr
        self.ctrl.barrier_in.wait()
        traj_infos = list()
        # Workers step environments and sample actions here.
        if self.eval_max_trajectories is not None:
            while True:
                time.sleep(EVAL_TRAJ_CHECK)
                traj_infos.extend(drain_queue(self.traj_infos_queue))
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
        traj_infos.extend(drain_queue(self.traj_infos_queue))
        self.ctrl.do_eval.value = False
        return traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()


def assemble_workers_kwargs(affinity, seed, samples_np, n_envs_list, sync, rank):
    workers_kwargs = list()
    i_env = 0
    g_env = sum(n_envs_list) * rank
    for w_rank in range(len(affinity["workers_cpus"])):
        n_envs = n_envs_list[w_rank]
        slice_B = slice(i_env, i_env + n_envs)
        env_ranks = list(range(g_env, g_env + n_envs))
        worker_kwargs = dict(
            rank=w_rank,
            env_ranks=env_ranks,
            seed=seed + w_rank,
            cpus=affinity["workers_cpus"][w_rank],
            n_envs=n_envs,
            samples_np=samples_np[:, slice_B],
            sync=sync,  # (only for eval, on cpu.)
        )
        i_env += n_envs
        g_env += n_envs
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
