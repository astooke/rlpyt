
import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.utils.synchronize import drain_queue
from rlpyt.utils.logging import logger


EVAL_TRAJ_CHECK = 20  # [steps].


class ActionServer:
    """Mixin class with methods for serving actions to worker processes which execute
    environment steps.
    """

    def serve_actions(self, itr):
        """Called in master process during ``obtain_samples()``.

        Performs agent action- selection loop in concert with workers
        executing environment steps.  Uses shared memory buffers to
        communicate agent/environment data at each time step.  Uses semaphores
        for synchronization: one per worker to acquire when they finish
        writing the next step of observations, one per worker to release when
        master has written the next actions.  Resets the agent one B-index at a time when the
        corresponding environment resets (i.e. agent's recurrent state, with
        leading dimension ``batch_B``).

        Also communicates ``agent_info`` to workers, which are responsible
        for recording all data into the batch buffer.

        If requested, collects additional agent value estimation of final
        observation for bootstrapping (the one thing written to the batch
        buffer here).


        .. warning::
            If trying to modify, must be careful to keep correct logic of the semaphores,
            to make sure they drain properly.  If a semaphore ends up with an extra release,
            synchronization can be lost silently, leading to wrong and confusing results.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        step_np, agent_inputs = self.step_buffer_np, self.agent_inputs

        for t in range(self.batch_spec.T):
            for b in obs_ready:
                b.acquire()  # Workers written obs and rew, first prev_act.
                # assert not b.acquire(block=False)  # Debug check.
            if self.mid_batch_reset and np.any(step_np.done):
                for b_reset in np.where(step_np.done)[0]:
                    step_np.action[b_reset] = 0  # Null prev_action into agent.
                    step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                    self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action  # Worker applies to env.
            step_np.agent_info[:] = agent_info  # Worker sends to traj_info.
            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()  # Signal to worker.

        for b in obs_ready:
            b.acquire()
            assert not b.acquire(block=False)  # Debug check.
        if "bootstrap_value" in self.samples_np.agent:
            self.samples_np.agent.bootstrap_value[:] = self.agent.value(
                *agent_inputs)
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                self.agent.reset_one(idx=b_reset)
            # step_np.done[:] = False  # Worker resets at start of next.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.

    def serve_actions_evaluation(self, itr):
        """Similar to ``serve_actions()``.  If a maximum number of eval trajectories
        was specified, keeps track of the number completed and terminates evaluation
        if the max is reached.  Returns a list of completed trajectory-info objects.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        step_np, step_pyt = self.eval_step_buffer_np, self.eval_step_buffer_pyt
        traj_infos = list()
        self.agent.reset()
        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.eval_max_T):
            if t % EVAL_TRAJ_CHECK == 0:  # (While workers stepping.)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                    guard_sentinel=True))
            for b in obs_ready:
                b.acquire()
                # assert not b.acquire(block=False)  # Debug check.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action.
                step_np.reward[b_reset] = 0  # Null prev_reward.
                self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action
            step_np.agent_info[:] = agent_info
            if self.eval_max_trajectories is not None and t % EVAL_TRAJ_CHECK == 0:
                self.sync.stop_eval.value = len(traj_infos) >= self.eval_max_trajectories
            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()
            if self.sync.stop_eval.value:
                logger.log("Evaluation reach max num trajectories "
                    f"({self.eval_max_trajectories}).")
                break
        if t == self.eval_max_T - 1 and self.eval_max_trajectories is not None:
            logger.log("Evaluation reached max num time steps "
                f"({self.eval_max_T}).")
        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            assert not b.acquire(block=False)  # Debug check.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.

        return traj_infos


class AlternatingActionServer:
    """Mixin class for serving actions in the alternating GPU sampler.  The
    synchronization format in this class allows the two worker groups to
    execute partially simultaneously; workers wait to step for their new
    action to be ready but do not wait for the other set of workers to be done
    stepping.
    """

    def serve_actions(self, itr):
        obs_ready_pair = self.obs_ready_pair
        act_ready_pair = self.act_ready_pair
        step_np_pair = self.step_buffer_np_pair
        agent_inputs_pair = self.agent_inputs_pair

        # Can easily write overlap and no overlap of workers versions.
        for t in range(self.batch_spec.T):
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in obs_ready_pair[alt]:
                    b.acquire()  # Workers written obs and rew, first prev_act.
                    # assert not b.acquire(block=False)  # Debug check.
                if self.mid_batch_reset and np.any(step_h.done):
                    for b_reset in np.where(step_h.done)[0]:
                        step_h.action[b_reset] = 0  # Null prev_action into agent.
                        step_h.reward[b_reset] = 0  # Null prev_reward into agent.
                        self.agent.reset_one(idx=b_reset)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action  # Worker applies to env.
                step_h.agent_info[:] = agent_info  # Worker sends to traj_info.
                for w in act_ready_pair[alt]:  # Final release.
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()  # Signal to worker.

        for alt in range(2):
            step_h = step_np_pair[alt]
            for b in obs_ready_pair[alt]:
                b.acquire()
                # assert not b.acquire(block=False)  # Debug check.
            if "bootstrap_value" in self.samples_np.agent:
                self.bootstrap_value_pair[alt][:] = self.agent.value(*agent_inputs_pair[alt])
            if np.any(step_h.done):
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0
                    step_h.reward[b_reset] = 0
                    self.agent.reset_one(idx=b_reset)
            self.agent.toggle_alt()  # Value and reset method do not advance rnn state.

        for b in self.sync.obs_ready:
            assert not b.acquire(block=False)  # Debug check.
        for w in self.sync.act_ready:
            assert not w.acquire(block=False)  # Debug check.

    def serve_actions_evaluation(self, itr):
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        obs_ready_pair = self.obs_ready_pair
        act_ready_pair = self.act_ready_pair
        step_np_pair = self.eval_step_buffer_np_pair
        agent_inputs_pair = self.eval_agent_inputs_pair
        traj_infos = list()
        self.agent.reset()
        stop = False

        for t in range(self.eval_max_T):
            if t % EVAL_TRAJ_CHECK == 0:  # (While workers stepping.)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                    guard_sentinel=True))
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in obs_ready_pair[alt]:
                    b.acquire()
                    # assert not b.acquire(block=False)  # Debug check.
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0  # Null prev_action.
                    step_h.reward[b_reset] = 0  # Null prev_reward.
                    self.agent.reset_one(idx=b_reset)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action
                step_h.agent_info[:] = agent_info
                if (self.eval_max_trajectories is not None and
                        t % EVAL_TRAJ_CHECK == 0 and alt == 0):
                    if len(traj_infos) >= self.eval_max_trajectories:
                        for b in obs_ready_pair[1 - alt]:
                            b.acquire()  # Now all workers waiting.
                        self.sync.stop_eval.value = stop = True
                        for w in act_ready[alt]:
                            w.release()
                        break
                for w in act_ready_pair[alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
            if stop:
                logger.log("Evaluation reached max num trajectories "
                    f"({self.eval_max_trajectories}).")
                break

        # TODO: check exit logic for/while ..?
        if not stop:
            logger.log("Evaluation reached max num time steps "
                f"({self.eval_max_T}).")

        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            assert not b.acquire(block=False)  # Debug check.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.

        return traj_infos


class NoOverlapAlternatingActionServer:
    """Mixin class for serving actions in the alternating GPU sampler.  The
    synchronization format in this class disallows the two worker groups from
    executing simultaneously; workers wait to step for their new action to be
    ready and also wait for the other set of workers to be done stepping.

    .. warning::
        Not sure the logic around semaphores is correct for all cases at the end of
        ``serve_actions_evaluation()`` (see TODO comment).
    """

    def serve_actions(self, itr):
        obs_ready = self.sync.obs_ready
        obs_ready_pair = self.obs_ready_pair
        act_ready_pair = self.act_ready_pair
        step_np, step_np_pair = self.step_buffer_np, self.step_buffer_np_pair
        agent_inputs, agent_inputs_pair = self.agent_inputs, self.agent_inputs_pair

        for t in range(self.batch_spec.T):
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in obs_ready_pair[alt]:
                    b.acquire()  # Workers written obs and rew, first prev_act.
                    # assert not b.acquire(block=False)  # Debug check.
                if t > 0 or alt > 0:  # Just don't do the very first one.
                    # Only let `alt` workers go after `1-alt` workers done stepping.
                    for w in act_ready_pair[1 - alt]:
                        # assert not w.acquire(block=False)  # Debug check.
                        w.release()

                if self.mid_batch_reset and np.any(step_h.done):
                    for b_reset in np.where(step_h.done)[0]:
                        step_h.action[b_reset] = 0  # Null prev_action into agent.
                        step_h.reward[b_reset] = 0  # Null prev_reward into agent.
                        self.agent.reset_one(idx=b_reset)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action  # Worker applies to env.
                step_h.agent_info[:] = agent_info  # Worker sends to traj_info.

        for alt in range(2):
            step_h = step_np_pair[alt]
            for b in obs_ready_pair[alt]:
                b.acquire()
                # assert not b.acquire(block=False)  # Debug check.
            if alt == 0:
                for w in act_ready_pair[1]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
            if "bootstrap_value" in self.samples_np.agent:
                self.bootstrap_value_pair[alt][:] = self.agent.value(*agent_inputs_pair[alt])
            if np.any(step_h.done):
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0
                    step_h.reward[b_reset] = 0
                    self.agent.reset_one(idx=b_reset)
            self.agent.toggle_alt()  # Value and reset method do not advance rnn state.

    def serve_actions_evaluation(self, itr):
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        obs_ready_pair = self.obs_ready_pair
        act_ready_pair = self.act_ready_pair
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
        for b in obs_ready_pair[alt]:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        action, agent_info = self.agent.step(*agent_inputs_pair[alt])
        step_h.action[:] = action
        step_h.agent_info[:] = agent_info
        alt = 1
        step_h = step_np_pair[alt]
        for b in obs_ready_pair[alt]:
            b.acquire()
            # assert not b.acquire(block=False)  # Debug check.
        for w in act_ready_pair[1 - alt]:
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
                for b in obs_ready_pair[alt]:
                    b.acquire()
                    # assert not b.acquire(block=False)  # Debug check.
                for w in act_ready_pair[1 - alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0  # Null prev_action.
                    step_h.reward[b_reset] = 0  # Null prev_reward.
                    self.agent.reset_one(idx=b_reset)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action
                step_h.agent_info[:] = agent_info
            if self.eval_max_trajectories is not None and t % EVAL_TRAJ_CHECK == 0:
                self.sync.stop_eval.value = len(traj_infos) >= self.eval_max_trajectories
            if self.sync.stop_eval.value:
                for w in act_ready_pair[1 - alt]:  # Other released past loop.
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                logger.log("Evaluation reached max num trajectories "
                    f"({self.eval_max_trajectories}).")
                break

        # TODO: check logic when traj limit hits at natural end of loop?
        for w in act_ready_pair[alt]:
            # assert not w.acquire(block=False)  # Debug check.
            w.release()
        if t == self.eval_max_T - 1 and self.eval_max_trajectories is not None:
            logger.log("Evaluation reached max num time steps "
                f"({self.eval_max_T}).")

        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            # assert not b.acquire(block=False)  # Debug check.

        return traj_infos
