
import numpy as np

from rlpyt.samplers.base import BaseEvalCollector
from rlpyt.samplers.collectors import DecorrelatingStartCollector
from rlpyt.utils.buffer import buffer_method


class GpuResetCollector(DecorrelatingStartCollector):
    """Valid to run episodic lives."""

    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """Params agent_inputs and itr unused."""
        act_waiter, step_blocker = self.sync.act_waiter, self.sync.step_blocker
        step = self.step_buffer_np
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        env_buf.prev_reward[0] = step.reward
        step_blocker.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_waiter.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r, d,
                    step.agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = step.action  # OPTIONAL BY SERVER
            env_buf.reward[t] = step.reward
            env_buf.done[t] = step.done
            if step.agent_info:
                agent_buf.agent_info[t] = step.agent_info  # OPTIONAL BY SERVER
            step_blocker.release()  # Ready for server to use/write step buffer.

        return None, traj_infos, completed_infos


class GpuWaitResetCollector(DecorrelatingStartCollector):
    """Valid to run episodic lives."""

    mid_batch_reset = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_reset = np.zeros(len(self.envs), dtype=np.bool)
        # e.g. For episodic lives, hold the observation output when done, record
        # blanks for the rest of the batch, but reinstate the observation to start
        # next batch.
        self.temp_observation = buffer_method(self.step_buffer_np.observation, "copy")

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """Params agent_inputs and itr unused."""
        act_waiter, step_blocker = self.sync.act_waiter, self.sync.step_blocker
        step = self.step_buffer_np
        b = np.where(step.done)[0]
        step.observation[b] = self.temp_observation[b]
        step.done[:] = False  # Did resets in between batches.
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        env_buf.prev_reward[0] = step.reward
        step_blocker.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_waiter.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                if step.done[b]:
                    step.action[b] = 0  # Record blank.
                    step.reward[b] = 0
                    if step.agent_info:
                        step.agent_info[b] = 0
                    # Leave step.done[b] = True, record that.
                    continue
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r, d,
                    step.agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    self.need_reset[b] = True
                if d:
                    self.temp_observation[b] = o  # Store until start of next batch.
                    o = 0  # Record blank.
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = step.action  # OPTIONAL BY SERVER
            env_buf.reward[t] = step.reward
            env_buf.done[t] = step.done
            if step.agent_info:
                agent_buf.agent_info[t] = step.agent_info  # OPTIONAL BY SERVER
            step_blocker.release()  # Ready for server to use/write step buffer.

        return None, traj_infos, completed_infos

    def reset_if_needed(self, agent_inputs):
        """Param agent_inputs unused, using step_buffer instead."""
        # step.done[:] = 0  # No, turn off in master after it resets agent.
        if np.any(self.need_reset):
            step = self.step_buffer_np
            for b in np.where(self.need_reset)[0]:
                step.observation[b] = self.envs[b].reset()
                step.action[b] = 0  # Prev_action to agent.
                step.reward[b] = 0
            self.need_reset[:] = False


class EvalCollector(BaseEvalCollector):

    def collect_evaluation(self, itr):
        """Param itr unused."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        act_waiter, step_blocker = self.sync.act_waiter, self.sync.step_blocker
        step = self.step_buffer_np
        for b, env in enumerate(self.envs):
            step.observation[b] = env.reset()
        step.done[:] = False
        step_blocker.release()

        for t in range(self.max_T):
            act_waiter.acquire()
            if self.sync.stop_eval.value:
                step_blocker.release()  # Always release at end of loop.
                break
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r, d,
                    step.agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    self.traj_infos_queue.put(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
            step_blocker.release()
        self.traj_infos_queue.put(None)  # End sentinel.
