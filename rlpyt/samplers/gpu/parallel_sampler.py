
import multiprocessing as mp


from rlpyt.samplers.parallel_sampler import ParallelSampler
from rlpyt.samplers.utils import build_samples_buffer, build_par_objs
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.utils.collections import AttrDict
from rlpyt.agents.base import AgentInputs


class GpuParallelSampler(ParallelSampler):

    def initialize(self, agent, affinity, seed,
            bootstrap_value=False, traj_info_kwargs=None):
        n_parallel = len(affinity["workers_cpus"])
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        n_envs = self.batch_spec.B // n_parallel  # Per worker.

        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        agent.initialize(env.spec)  # Actual agent initialization, keep.
        buffers = build_samples_buffer(agent, env, self.batch_spec,
            bootstrap_value, agent_shared=True, env_shared=True,
            build_step_buffer=True)
        samples_pyt, samples_np, step_buffer_pyt, step_buffer_np = buffers
        env.terminate()
        del env

        ctrl, traj_infos_queue, sync = build_par_objs(n_parallel, sync=True)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.

        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            n_envs=n_envs,
            agent=None,
            batch_T=self.batch_spec.T,
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=traj_infos_queue,
            ctrl=ctrl,
            max_decorrelation_steps=self.max_decorrelation_steps,
            torch_threads=affinity.get("worker_torch_threads", None),
        )

        workers_kwargs = assemble_workers_kwargs(affinity, seed, samples_np,
            n_envs, step_buffer_np, sync)

        workers = [mp.Process(target=sampling_process,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.agent = agent
        self.workers = workers
        self.ctrl = ctrl
        self.traj_infos_queue = traj_infos_queue
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.step_buffer_pyt = step_buffer_pyt
        self.step_buffer_np = step_buffer_np
        self.sync = sync

        self.ctrl.barrier_out.wait()  # Wait for workers to decorrelate envs.

    def serve_actions(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_np = self.step_buffer_np
        step_pyt = self.step_buffer_pyt

        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.batch_spec.T):
            for b in step_blockers:
                b.acquire()  # Workers written obs and rew, first prev_act.
            action, agent_info = self.agent.sample_action(*agent_inputs)
            step_np.action[:] = action  # Worker applies to env.
            step_np.agent_info[:] = agent_info  # Worker sends to traj_info.
            for w in act_waiters:
                w.release()  # Signal to worker.

        for b in step_blockers:
            b.acquire()
        if "bootstrap_value" in self.samples_np.agent:
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(
                *agent_inputs)

        if any(step_np.done):  # Reset at end of batch; ready for next.
            for i, d in enumerate(step_np.done):
                if d:
                    self.agent.reset_one(idx=i)
            step_np.done[:] = 0


def assemble_workers_kwargs(affinity, seed, samples_np, n_envs, step_buffer_np,
        sync):
    workers_kwargs = list()
    for rank in range(len(affinity["workers_cpus"])):
        slice_B = slice(rank * n_envs, (rank + 1) * n_envs)
        w_sync = AttrDict(
            step_blocker=sync.step_blockers[rank],
            act_waiter=sync.act_waiters[rank],
        )
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinity["workers_cpus"][rank],
            samples_np=samples_np[:, slice_B],
            step_buffer_np=step_buffer_np[slice_B],
            sync=w_sync,
        )
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
