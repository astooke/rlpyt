
import numpy as np

from rlpyt.samplers.parallel.gpu.action_server import (ActionServer,
    AlternatingActionServer, NoOverlapAlternatingActionServer)
from rlpyt.agents.base import AgentInputs


class AsyncActionServer(ActionServer):

    def serve_actions_evaluation(self, itr):
        """Similar to normal action-server, but with different signaling logic
        for ending evaluation early; receive signal from main sampler process
        and pass it along to my workers.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        step_np, step_pyt = self.eval_step_buffer_np, self.eval_step_buffer_pyt
        self.agent.reset()
        agent_inputs = AgentInputs(step_pyt.observation, step_pyt.action,
            step_pyt.reward)  # Fixed buffer objects.

        for t in range(self.eval_max_T):
            for b in obs_ready:
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
            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()
            if self.sync.stop_eval.value:  # Signal from sampler runner.
                break
        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            assert not b.acquire(block=False)  # Debug check.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.


class AsyncAlternatingActionServer(AlternatingActionServer):

    def serve_actions_evaluation(self, itr):
        """Similar to normal action-server, but with different signaling logic
        for ending evaluation early; receive signal from main sampler process
        and pass it along to my workers.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        obs_ready_pair = self.obs_ready_pair
        act_ready_pair = self.act_ready_pair
        step_np, step_np_pair = self.eval_step_buffer_np, self.eval_step_buffer_np_pair
        agent_inputs_pair = self.eval_agent_inputs_pair
        self.agent.reset()
        step_np.action[:] = 0  # Null prev_action.
        step_np.reward[:] = 0  # Null prev_reward.
        stop = False

        for t in range(self.eval_max_T):
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
                if self.ctrl.stop_eval.value:  # From overall master.
                    for b in obs_ready_pair[1 - alt]:
                        b.acquire()  # Wait until all workers are waiting.
                        # assert not b.acquire(block=False)
                    self.sync.stop_eval.value = stop = True  # To my workers.
                    for w in act_ready:
                        w.release()
                    break
                for w in act_ready_pair[alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
            if stop:
                break

        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            assert not b.acquire(block=False)  # Debug check.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.



class AsyncNoOverlapAlternatingActionServer(NoOverlapAlternatingActionServer):
    """Not tested, possibly faulty corner cases for synchronization."""

    def serve_actions_evaluation(self, itr):
        """Similar to normal action-server, but with different signaling logic
        for ending evaluation early; receive signal from main sampler process
        and pass it along to my workers.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        obs_ready_pair = self.obs_ready_pair
        act_ready_pair = self.act_ready_pair
        step_np, step_np_pair = self.eval_step_buffer_np, self.eval_step_buffer_np_pair
        agent_inputs_pair = self.eval_agent_inputs_pair
        self.agent.reset()
        step_np.action[:] = 0  # Null prev_action.
        step_np.reward[:] = 0  # Null prev_reward.
        stop = False

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
            for alt in range(2):
                step_h = step_np_pair[alt]
                for b in obs_ready_pair[alt]:
                    b.acquire()
                    # assert not b.acquire(block=False)  # Debug check.
                if self.ctrl.stop_eval.value:  # Signal from sampler runner.
                    self.sync.stop_eval.value = stop = True  # Signal to my workers.
                for w in act_ready_pair[1 - alt]:
                    # assert not w.acquire(block=False)  # Debug check.
                    w.release()
                if stop:
                    break
                for b_reset in np.where(step_h.done)[0]:
                    step_h.action[b_reset] = 0  # Null prev_action.
                    step_h.reward[b_reset] = 0  # Null prev_reward.
                    self.agent.reset_one(idx=b_reset)
                action, agent_info = self.agent.step(*agent_inputs_pair[alt])
                step_h.action[:] = action
                step_h.agent_info[:] = agent_info
            if stop:
                break

        # TODO: check logic when traj limit hits at natural end of loop?
        for w in act_ready_pair[alt]:
            # assert not w.acquire(block=False)  # Debug check.
            w.release()

        for b in obs_ready:
            b.acquire()  # Workers always do extra release; drain it.
            assert not b.acquire(block=False)  # Debug check.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.
