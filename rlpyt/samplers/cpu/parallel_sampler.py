
import multiprocessing as mp


from rlpyt.samplers.parallel_sampler import ParallelSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.utils import build_par_objs
from rlpyt.samplers.parallel_worker import sampling_process


class CpuParallelSampler(ParallelSampler):

    def initialize(self, agent, affinities, seed,
            bootstrap_value=False, traj_info_kwargs=None):
        n_parallel = len(affinities.worker_cpus)
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        n_envs = self.batch_spec.B // n_parallel  # Per worker.

        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_args)
        agent.initialize(env.spec, share_memory=True)
        samples_pyt, samples_np = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True,
            env_shared=True, build_step_buffer=False)
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
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=traj_infos_queue,
            ctrl=ctrl,
            max_path_length=self.max_path_length,
            max_decorrelation_steps=self.max_decorrelation_steps,
        )

        workers_kwargs = assemble_workers_kwargs(affinities, seed,
            samples_pyt, samples_np, n_envs)

        workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.ctrl = ctrl
        self.traj_infos_queue = traj_infos_queue
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.agent = agent

        self.ctrl.barrier.wait()  # Wait for workers to decorrelate envs.


def assemble_workers_kwargs(affinities, seed, samples_np, n_envs):
    workers_kwargs = list()
    for rank in range(len(affinities.workers_cpus)):
        slice_B = slice(rank * n_envs, (rank + 1) * n_envs)
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinities.workers_cpus[rank],
            samples_np=samples_np[:, slice_B],
        )
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
