
import numpy as np

from rlpyt.samplers.gpu.action_server import (ActionServer,
    AlternatingActionServer, NoOverlapAlternatingActionServer)
from rlpyt.agents.base import AgentInputs


class AsyncActionServer(ActionServer):

    def serve_actions_evaluation(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_np, step_pyt = self.eval_step_buffer_np, self.eval_step_buffer_pyt
        self.agent.reset()
        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.eval_max_T):
            for b in step_blockers:
                b.acquire()
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action.
                step_np.reward[b_reset] = 0  # Null prev_reward.
                self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action
            step_np.agent_info[:] = agent_info
            if self.ctrl.stop_eval.value:  # From overall master process.
                self.sync.stop_eval.value = True  # Give to my workers.
            for w in act_waiters:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()
            if self.sync.stop_eval.value:
                break
        for b in step_blockers:
            b.acquire()  # Workers always do extra release; drain it.
            # assert not b.acquire(block=False)  # Debug check.   


class AsyncAlternatingActionServer(AlternatingActionServer):

    def serve_actions_evaluation(self, itr):
        step_blockers = self.sync.step_blockers
        step_blockers_pair = self.step_blockers_pair
        act_waiters_pair = self.act_waiters_pair
        step_np, step_np_pair = self.eval_step_buffer_np, self.eval_step_buffer_np_pair
        agent_inputs = self.eval_agent_inputs
        agent_inputs_pair = self.eval_agent_inputs_pair
        self.agent.reset()

        for t in range(self.eval_max_T):
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
                if self.ctrl.stop_eval.value:  # From overall master.
                    self.sync.stop_eval.value = True  # To my workers.
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


class AsyncNoOverlapAlternatingActionServer(NoOverlapAlternatingActionServer):

    def serve_actions_evaluation(self, itr):
        step_blockers, act_waiters = self.sync.step_blockers, self.sync.act_waiters
        step_blockers_pair = self.step_blockers_pair
        act_waiters_pair = self.act_waiters_pair
        step_np, step_np_pair = self.eval_step_buffer_np, self.eval_step_buffer_np_pair
        agent_inputs = self.eval_agent_inputs
        agent_inputs_pair = self.eval_agent_inputs_pair
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
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in step_blockers_pair[alt]:
                    b.acquire()
                    # assert not b.acquire(block=False)  # Debug check.
                if self.ctrl.stop_eval.value:
                    self.sync.stop_eval.value = True
                for w in act_waiters_pair[1 - alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                if self.sync.stop_eval.value:
                    break
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0  # Null prev_action.
                    step_h.reward[b_reset] = 0  # Null prev_reward.
                    self.agent.reset_one(idx=b_reset, alt=alt)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action
                step_h.agent_info[:] = agent_info

        # TODO: check logic when traj limit hits at natural end of loop?
        for w in act_waiters_pair[alt]:
            # assert not w.acquire(block=False)  # Debug check.
            w.release()

        for b in step_blockers:
            b.acquire()  # Workers always do extra release; drain it.
            # assert not b.acquire(block=False)  # Debug check.
