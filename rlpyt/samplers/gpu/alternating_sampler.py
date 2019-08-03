
import multiprocessing as mp
import numpy as np


from rlpyt.samplers.gpu.parallel_sampler import GpuParallelSampler
from rlpyt.samplers.utils import (build_samples_buffer, build_par_objs,
    build_step_buffer)
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.samplers.gpu.collectors import EvalCollector
from rlpyt.utils.collections import AttrDict
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.logging import logger
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 20  # Time steps.


class GpuAlternatingSampler(GpuParallelSampler):
    """Environment instances divided in two; while one set steps, the GPU gets the action
    for the other set.  That set waits for actions to be returned and for the opposite set
    to finish execution before starting (as managed by semaphores here)."""

    def initialize(self, agent, *args, **kwargs):
        if agent.recurrent and not agent.alternating:
            raise TypeError("If agent is recurrent, must be 'alternating' to use here.")
        elif not agent.recurrent:
            agent.alternating = True  # FF agent doesn't need special class, but tell it so.
        examples = super().initialize(agent, *args, **kwargs)
        half_w = self.n_worker // 2  # Half of workers.
        self.half_B = half_B = self.batch_spec.B // 2  # Half of envs.
        self.step_blockers_pair = (step_blockers[:half_w], step_blockers[half_w:])
        self.act_waiters_pair = (act_waiters[:half_w], act_waiter[half_w:])
        self.step_buffer_np_pair = (step_buffer_np[:half_B], step_buffer_np[half_B:])
        self.agent_inputs_pair = (self.agent_inputs[:half_B], self.agent_inputs[half_B:])
        return examples

    def get_n_envs_list(self, affinity):
        assert self.batch_spec.B % 2 == 0, "Need even number of envs."
        assert len(affinity["workers_cpus"]) % 2 == 0, "Need even number of workers."
        B = self.batch_spec.B
        n_worker = len(affinity["workers_cpus"])
        if B < n_worker:
            logger.log(f"WARNING: requested fewer envs ({B}) than available worker "
                f"processes ({n_worker}). Using fewer workers (but maybe better to "
                "increase sampler's `batch_B`.")
            n_worker = B
        self.n_worker = n_worker
        n_envs_list = [B // n_worker] * n_worker
        if not B % n_worker == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                f"batch_B {B} and n_worker {n_worker} "
                "(possible suboptimal speed).")
            for b in range((B % n_worker) // 2):
                n_envs_list[b] += 1
                n_envs_list[b + half_B] += 1  # Paired worker.
        return n_envs_list

    def serve_actions(self, itr):
        step_blockers = self.sync.step_blockers
        step_blockers_pair = self.step_blockers_pair
        act_waiters_pair = self.act_waiters_pair
        step_np, step_np_pair = self.step_buffer_np, self.step_buffer_np_pair
        agent_inputs, agent_inputs_pair = self.agent_inputs, self.agent_inputs_pair

        # Can easily write overlap and no overlap of workers versions.
        for t in range(self.batch_spec.T):
            for alt in range(2):
                step_np_ = step_np_pair[alt]
                for b in step_blockers_pair[alt]:
                    b.acquire()  # Workers written obs and rew, first prev_act.
                    # assert not b.acquire(block=False)  # Debug check.
                if self.mid_batch_reset and np.any(step_np_.done):
                    for b_reset in np.where(step_np_.done)[0]:
                        step_np_.action[b_reset] = 0  # Null prev_action into agent.
                        step_np_.reward[b_reset] = 0  # Null prev_reward into agent.
                        self.agent.reset_one(idx=b_reset, alt=alt)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_np_.action[:] = action  # Worker applies to env.
                step_np_.agent_info[:] = agent_info  # Worker sends to traj_info.
                for w in act_waiters_pair[alt]:  # Final release.
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()  # Signal to worker.

        for b in step_blockers:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        if "bootstrap_value" in self.samples_np.agent:
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(
                *agent_inputs)  # All in one call: or maybe alternate these too.
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                self.agent.reset_one(idx=b_reset)  # TODO: figure out alt.
            # step_np.done[:] = False  # Worker resets at start of next.

    def serve_actions_evaluation(self, itr):
        step_blockers = self.sync.step_blockers
        step_blockers_pair = self.step_blockers_pair
        act_waiters_pair = self.act_waiters_pair
        step_np, step_np_pair = self.eval_step_buffer_np, self.eval_step_buffer_np_pair
        agent_inputs = self.eval_agent_inputs
        agent_inputs_pair = self.eval_agent_inputs_pair
        traj_infos = list()
        self.agent.reset()

        for t in range(self.eval_max_T):
            if t % EVAL_TRAJ_CHECK == 0:  # (While workers stepping.)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                    guard_sentinel=True))
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in step_blockers_pair[alt]:
                    b.acquire()
                    # assert not b.acquire(block=False)  # Debug check.
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0  # Null prev_action.
                    step_h.reward[b_reset] = 0  # Null prev_reward.
                    self.agent.reset_one(idx=b_reset, alt=alt)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action
                step_h.agent_info[:] = agent_info
                if (self.eval_max_trajectories is not None and
                        t % EVAL_TRAJ_CHECK == 0 and alt == 0):
                    self.sync.stop_eval.value = len(traj_infos) >= self.eval_max_trajectories
                for w in act_waiters_pair[alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                if self.sync.stop_eval.value:
                    for w in act_waiters_pair[1 - alt]:
                        # assert not w.acquire(block=False)  # Debug check.
                        w.release()
                    logger.log("Evaluation reached max num trajectories "
                        f"({self.eval_max_trajectories}).")
                    break

        # TODO: check exit logic for/while ..?
        if t == self.eval_max_T - 1 and self.eval_max_trajectories is not None:
            logger.log("Evaluation reached max num time steps "
                f"({self.eval_max_T}).")

        for b in step_blockers:
            b.acquire()  # Workers always do extra release; drain it.
            # assert not b.acquire(block=False)  # Debug check.

        return traj_infos


class GpuAltNoOverlapSampler(GpuAlternatingSampler):
    """Two environment instance groups may overlap in time of execution."""

    def serve_actions(self, itr):
        step_blockers = self.sync.step_blockers
        step_blockers_pair = self.step_blockers_pair
        act_waiters_pair = self.act_waiters_pair
        step_np, step_np_pair = self.step_buffer_np, self.step_buffer_np_pair
        agent_inputs, agent_inputs_pair = self.agent_inputs, self.agent_inputs_pair

        for t in range(self.batch_spec.T):
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in step_blockers_pair[alt]:
                    b.acquire()  # Workers written obs and rew, first prev_act.
                    # assert not b.acquire(block=False)  # Debug check.
                if t > 0 or alt > 0:  # Just don't do the very first one.
                    # Only let `alt` workers go after `1-alt` workers done stepping.
                    for w in act_waiters_pair[1 - alt]:
                        # assert not w.acquire(block=False)  # Debug check.
                        w.release()

                if self.mid_batch_reset and np.any(step_h.done):
                    for b_reset in np.where(step_h.done)[0]:
                        step_h.action[b_reset] = 0  # Null prev_action into agent.
                        step_h.reward[b_reset] = 0  # Null prev_reward into agent.
                        self.agent.reset_one(idx=b_reset + alt * self.half_B)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action  # Worker applies to env.
                step_h.agent_info[:] = agent_info  # Worker sends to traj_info.

        for b in step_blockers_pair[0]:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        for w in act_waiters_pair[1]:  # Final release.
            # assert not w.acquire(block=False)  # Debug check.
            w.release()  # Signal to worker.
        for b in step_blockers_pair[1]:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        if "bootstrap_value" in self.samples_np.agent:
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(
                *agent_inputs)  # TODO: make into two calls.
        step_np = self.step_buffer_np
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                self.agent.reset_one(idx=b_reset)
            # step_np.done[:] = False  # Worker resets at start of next.    

    def serve_actions_evaluation(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_blockers_pair = self.step_blockers_pair
        act_waiters_pair = self.act_waiters_pair
        step_np, step_np_pair = self.eval_step_buffer_np, self.eval_step_buffer_np_pair
        agent_inputs = self.eval_agent_inputs
        agent_inputs_pair = self.eval_agent_inputs_pair
        traj_infos = list()
        self.agent.reset()
        step_np.action[:] = 0  # Null prev_action.
        step_np.reward[:] = 0  # Null prev_reward.

        # First step of both.
        alt = 0
        step_h = step_np_pair[alt]
        for b in step_blockers_pair[alt]:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        action, agent_info = self.agent.step(*agent_inputs_pair[alt])
        step_h.action[:] = action
        step_h.agent_info[:] = agent_info
        alt = 1
        step_h = step_np_pair[alt]
        for b in step_blockers_pair[alt]:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        for w in act_waiters_pair[1 - alt]:
            # assert not w.acquire(block=False)  # Debug check.
            w.release()
        action, agent_info = self.agent.step(*agent_inputs_pair[alt])
        step_h.action[:] = action
        step_h.agent_info[:] = agent_info

        for t in range(1, self.eval_max_T):
            if t % EVAL_TRAJ_CHECK == 0:  # (While workers stepping.)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                    guard_sentinel=True))
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in step_blockers_pair[alt]:
                    b.acquire()
                    # assert not b.acquire(block=False)  # Debug check.
                for w in act_waiters_pair[1 - alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0  # Null prev_action.
                    step_h.reward[b_reset] = 0  # Null prev_reward.
                    self.agent.reset_one(idx=b_reset, alt=alt)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action
                step_h.agent_info[:] = agent_info
            if self.eval_max_trajectories is not None and t % EVAL_TRAJ_CHECK == 0:
                self.sync.stop_eval.value = len(traj_infos) >= self.eval_max_trajectories
            if self.sync.stop_eval.value:
                for w in act_waiters_pair[1 - alt]:  # Other released past loop.
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                logger.log("Evaluation reached max num trajectories "
                    f"({self.eval_max_trajectories}).")
                break

        # TODO: check logic when traj limit hits at natural end of loop?
        for w in act_waiters_pair[alt]:
            # assert not w.acquire(block=False)  # Debug check.
            w.release()
        if t == self.eval_max_T - 1 and self.eval_max_trajectories is not None:
            logger.log("Evaluation reached max num time steps "
                f"({self.eval_max_T}).")

        for b in step_blockers:
            b.acquire()  # Workers always do extra release; drain it.
            # assert not b.acquire(block=False)  # Debug check.

        return traj_infos
