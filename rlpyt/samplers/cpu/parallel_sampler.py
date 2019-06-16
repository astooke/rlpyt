
import multiprocessing as mp
import time


from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import build_samples_buffer, build_par_objs
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.samplers.cpu.collectors import EvalCollector
from rlpyt.utils.logging import logger


EVAL_TRAJ_CHECK = 1  # Seconds.


class CpuParallelSampler(BaseSampler):

    def initialize(self, agent, affinity, seed,
            bootstrap_value=False, traj_info_kwargs=None):
        n_parallel = len(affinity["workers_cpus"])
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        n_envs = self.batch_spec.B // n_parallel  # Per worker.

        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        agent.initialize(env.spec, share_memory=True)  # Actual agent initialization.
        samples_pyt, samples_np, examples = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True, env_shared=True,
            subprocess=False)  # TODO: subprocess=True fix!!
        env.terminate()
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
            eval_max_T = self.eval_max_steps // eval_n_envs
        else:
            eval_n_envs_per = 0
            eval_max_T = None

        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            n_envs=n_envs,
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
        )

        workers_kwargs = assemble_workers_kwargs(affinity, seed, samples_np,
            n_envs, sync)

        workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.workers = workers
        self.ctrl = ctrl
        self.traj_infos_queue = traj_infos_queue
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.agent = agent

        self.ctrl.barrier_out.wait()  # Wait for workers to decorrelate envs.
        return examples  # e.g. In case useful to build replay buffer.

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # New weights in workers, if needed.
        self.samples_np[:] = 0  # Reset all batch sample values (optional?).
        self.ctrl.itr.value = itr
        self.ctrl.barrier_in.wait()
        # Workers step environments and sample actions here.
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        self.agent.sync_shared_memory()
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.ctrl.itr.value = itr
        self.ctrl.barrier_in.wait()
        traj_infos = list()
        # Workers step environments and sample actions here.
        if self.max_eval_trajectories is not None:
            while True:
                time.sleep(EVAL_TRAJ_CHECK)
                while self.traj_infos_queue.qsize():
                    traj_infos.append(self.traj_infos_queue.get())
                if len(traj_infos) >= self.max_eval_trajectories:
                    self.sync.stop_eval.value = True
                    logger.log("Evaluation reached max num trajectories "
                        f"({self.eval__trajectories}).")
                    break  # Stop before workers reach max_T.
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
        for w in self.workers:
            w.join()


def assemble_workers_kwargs(affinity, seed, samples_np, n_envs, sync):
    workers_kwargs = list()
    for rank in range(len(affinity["workers_cpus"])):
        slice_B = slice(rank * n_envs, (rank + 1) * n_envs)
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinity["workers_cpus"][rank],
            samples_np=samples_np[:, slice_B],
            sync=sync,  # (only for eval, on cpu.)
        )
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
