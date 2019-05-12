
import multiprocessing as mp


from rlpyt.samplers.base import BaseSampler, Samples
from rlpyt.samplers.utils import build_samples_buffers, build_par_objs, assemble_workers_kwargs
from rlpyt.samplers.cpu.worker import sampling_process


class CpuSampler(BaseSampler):

    def initialize(self, agent, affinities, seed, traj_info_kwargs=None):
        n_parallel = len(affinities.worker_cpus)
        assert self.batch_spec.E % n_parallel == 0  # Same num envs per worker.
        example_env = self.EnvCls(**self.env_args)
        agent.initialize(example_env.spec, share_memory=True)
        agent_buf, env_buf = build_samples_buffers(agent, example_env, self.batch_spec)
        del example_env
        ctrl, traj_infos_queue = build_par_objs(n_parallel)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)

        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            n_envs=self.batch_spec.E // n_parallel,  # Per worker.
            agent=agent,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=traj_infos_queue,
            ctrl=ctrl,
            max_path_length=self.max_path_length,
            max_decorrelation_steps=self.max_decorrelation_steps,
        )

        workers_kwargs = assemble_workers_kwargs(affinities, seed, env_buf, agent_buf)

        workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.ctrl = ctrl
        self.traj_infos_queue = traj_infos_queue
        self.agent_buf = agent_buf
        self.env_buf = env_buf
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
        return Samples(self.agent_buf, self.env_buf), traj_infos

