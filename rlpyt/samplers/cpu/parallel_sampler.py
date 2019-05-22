
import multiprocessing as mp


from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import build_samples_buffer, build_par_objs
from rlpyt.samplers.parallel_worker import sampling_process


class CpuParallelSampler(BaseSampler):

    def initialize(self, agent, affinity, seed,
            bootstrap_value=False, traj_info_kwargs=None):
        n_parallel = len(affinity["workers_cpus"])
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        n_envs = self.batch_spec.B // n_parallel  # Per worker.

        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        agent.initialize(env.spec, share_memory=True)  # Actual agent initialization.
        samples_pyt, samples_np = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value,
            agent_shared=True, env_shared=True, build_step_buffer=False)
        env.terminate()
        del env

        ctrl, traj_infos_queue = build_par_objs(n_parallel, sync=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.

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
        )

        workers_kwargs = assemble_workers_kwargs(affinity, seed, samples_np, n_envs)

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

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # New weights in workers, if needed.
        self.samples_np[:] = 0  # Reset all batch sample values (optional?).
        self.ctrl.barrier_in.wait()
        # Workers step environments and sample actions here.
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        return self.samples_pyt, traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()


def assemble_workers_kwargs(affinity, seed, samples_np, n_envs):
    workers_kwargs = list()
    for rank in range(len(affinity["workers_cpus"])):
        slice_B = slice(rank * n_envs, (rank + 1) * n_envs)
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinity["workers_cpus"][rank],
            samples_np=samples_np[:, slice_B],
        )
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
