
import multiprocessing as mp


from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.utils import build_par_objs, assemble_workers_kwargs
from rlpyt.samplers.cpu.worker import sampling_process


class CpuSampler(BaseSampler):

    def initialize(self, agent, affinities, seed,
            bootstrap_value=False, traj_info_kwargs=None):
        n_parallel = len(affinities.worker_cpus)
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        n_envs = self.batch_spec.B // n_parallel
        env = self.EnvCls(**self.env_args)
        agent.initialize(env.spec, share_memory=True)
        samples_pyt, samples_np = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True,
            env_shared=True)
        del env
        ctrl, traj_infos_queue = build_par_objs(n_parallel)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.

        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            n_envs=n_envs,  # Per worker.
            agent=agent,
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

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # Weight values appear in workers.
        self.ctrl.barrier_in.wait()
        # Workers collect samples here.
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.apppend(self.traj_infos_queue.get())
        return self.samples_pyt, traj_infos

