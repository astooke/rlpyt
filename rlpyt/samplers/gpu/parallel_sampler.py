
import multiprocessing as mp


from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.utils import (build_samples_buffer, build_par_objs,
    build_step_buffer)
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.samplers.gpu.collectors import EvalCollector
from rlpyt.utils.collections import AttrDict
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.logging import logger


EVAL_TRAJ_CHECK = 20  # Time steps.


class GpuParallelSampler(BaseSampler):

    def initialize(self, agent, affinity, seed,
            bootstrap_value=False, traj_info_kwargs=None):
        n_parallel = len(affinity["workers_cpus"])
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        n_envs = self.batch_spec.B // n_parallel  # Per worker.

        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        agent.initialize(env.spec, share_memory=False)  # Actual agent initialization, keep.
        samples_pyt, samples_np, examples = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True, env_shared=True,
            subprocess=False)
        env.terminate()
        del env
        step_buffer_pyt, step_buffer_np = build_step_buffer(examples, self.batch_spec.B)

        if self.eval_n_envs > 0:
            # assert self.eval_n_envs % n_parallel == 0
            eval_n_envs_per = max(1, self.eval_n_envs // n_parallel)
            eval_n_envs = eval_n_envs_per * n_parallel
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}")
            self.eval_max_T = eval_max_T = self.eval_max_steps // eval_n_envs
            eval_step_buffer_pyt, eval_step_buffer_np = build_step_buffer(examples,
                eval_n_envs)
            self.eval_step_buffer_pyt = eval_step_buffer_pyt
            self.eval_step_buffer_np = eval_step_buffer_np
            assert self.eval_min_envs_reset <= eval_n_envs
        else:
            eval_n_envs_per = 0
            eval_step_buffer_np = None
            eval_max_T = None

        ctrl, traj_infos_queue, sync = build_par_objs(n_parallel)
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
            torch_threads=None,
            eval_n_envs=eval_n_envs_per,
            eval_CollectorCls=self.eval_CollectorCls or EvalCollector,
            eval_env_kwargs=self.eval_env_kwargs,
            eval_max_T=eval_max_T,
        )

        workers_kwargs = assemble_workers_kwargs(affinity, seed, samples_np,
            n_envs, step_buffer_np, sync, eval_n_envs_per, eval_step_buffer_np)

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
        return examples  # e.g. In case useful to build replay buffer

    def obtain_samples(self, itr):
        self.samples_np[:] = 0  # Reset all batch sample values (optional?).
        self.agent.sample_mode(itr)
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.agent.eval_mode(itr)
        self.ctrl.barrier_in.wait()
        traj_infos = self.serve_actions_evaluation(itr)
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

    def serve_actions(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_np, step_pyt = self.step_buffer_np, self.step_buffer_pyt

        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.batch_spec.T):
            for b in step_blockers:
                b.acquire()  # Workers written obs and rew, first prev_act.
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

        if any(step_np.done):  # Reset at end of batch; ready for next.
            for b, d in enumerate(step_np.done):
                if d:
                    self.agent.reset_one(idx=b)
            step_np.done[:] = 0

    def serve_actions_evaluation(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_np, step_pyt = self.eval_step_buffer_np, self.eval_step_buffer_pyt
        traj_infos = list()
        self.agent.reset()
        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.eval_max_T):
            if t % EVAL_TRAJ_CHECK == 0:  # (While workers stepping.)
                while self.traj_infos_queue.qsize():
                    traj_infos.append(self.traj_infos_queue.get())
            for b in step_blockers:
                b.acquire()
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action
            step_np.agent_info[:] = agent_info

            self.sync.do_reset.value = sum(step_np.need_reset) >= self.eval_min_envs_reset
            if self.sync.do_reset.value:
                for b, need in enumerate(step_np.need_reset):
                    if need:
                        self.agent.reset_one(idx=b)
                        step_np.action[b] = 0  # Prev_action for next t.
                    # Do not set need_reset[b] = False; worker needs it and will set False.
            if self.eval_max_trajectories is not None and t % EVAL_TRAJ_CHECK == 0:
                self.sync.stop_eval.value = len(traj_infos) >= self.eval_max_trajectories
            for w in act_waiters:
                w.release()
            if self.sync.stop_eval.value:
                logger.log("Evaluation reach max num trajectories "
                    f"({self.eval_max_trajectories}).")
                break
        if t == self.eval_max_T - 1 and self.eval_max_trajectories is not None:
            logger.log("Evaluation reached max num time steps "
                f"({self.eval_max_T}).")

        return traj_infos


def assemble_workers_kwargs(affinity, seed, samples_np, n_envs, step_buffer_np,
        sync, eval_n_envs, eval_step_buffer_np):
    workers_kwargs = list()
    for rank in range(len(affinity["workers_cpus"])):
        slice_B = slice(rank * n_envs, (rank + 1) * n_envs)
        w_sync = AttrDict(
            step_blocker=sync.step_blockers[rank],
            act_waiter=sync.act_waiters[rank],
            do_reset=sync.do_reset,
            stop_eval=sync.stop_eval,
        )
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinity["workers_cpus"][rank],
            samples_np=samples_np[:, slice_B],
            step_buffer_np=step_buffer_np[slice_B],
            sync=w_sync,
        )
        if eval_n_envs > 0:
            eval_slice_B = slice(rank * eval_n_envs, (rank + 1) * eval_n_envs)
            worker_kwargs["eval_step_buffer_np"] = eval_step_buffer_np[eval_slice_B]
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
