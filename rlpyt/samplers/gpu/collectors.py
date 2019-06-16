
from rlpyt.samplers.base import BaseEvalCollector
from rlpyt.samplers.collectors import DecorrelatingStartCollector


class ResetCollector(DecorrelatingStartCollector):
    """Valid to run episodic lives."""

    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """Params agent_inputs and itr unused."""
        act_waiter, step_blocker = self.sync.act_waiter, self.sync.step_blocker
        step = self.step_buffer_np
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        step_blocker.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        env_buf.prev_reward[0] = step.reward
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_waiter.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r,
                    step.agent_info[b], env_info)
                if getattr(env_info, "need_reset", d):
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


class WaitResetCollector(DecorrelatingStartCollector):
    """Valid to run episodic lives."""

    mid_batch_reset = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_reset = [False] * len(self.envs)

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """Params agent_inputs and itr unused."""
        act_waiter, step_blocker = self.sync.act_waiter, self.sync.step_blocker
        step = self.step_buffer_np
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        step_blocker.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        env_buf.prev_reward[0] = step.reward
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_waiter.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                if self.need_reset[b]:
                    continue
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r,
                    step.agent_info[b], env_info)
                if getattr(env_info, "need_reset", d):
                    self.need_reset[b] = True
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                else:
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
        """Param agent_inputs unused."""
        step = self.step_buffer_np
        for b, need in enumerate(self.need_reset):
            if need:
                step.observation[b] = self.envs[b].reset()
                step.action[b] = 0
                step.reward[b] = 0
                self.need_reset[b] = False
        return None


class EvalCollector(BaseEvalCollector):

    def collect_evaluation(self, itr):
        """Param itr unused."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        act_waiter, step_blocker = self.sync.act_waiter, self.sync.step_blocker
        step = self.step_buffer_np
        for b, env in enumerate(self.envs):
            step.observation[b] = env.reset()
        step_blocker.release()

        for t in range(self.max_T):
            act_waiter.acquire()
            if self.sync.stop_eval.value:
                break
            for b, env in enumerate(self.envs):
                if step.need_reset[b]:
                    if self.sync.do_reset.value:
                        step.observation[b] = env.reset()
                        step.reward[b] = 0  # Prev_reward for next t.
                        step.need_reset[b] = False
                        traj_infos[b] = self.TrajInfoCls()
                    continue  # Wait for new action, next t.
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r,
                    step.agent_info[b], env_info)
                if getattr(env_info, "need_reset", d):
                    self.traj_infos_queue.put(traj_infos[b].terminate(o))
                    step.need_reset[b] = True
                else:
                    step.observation[b] = o
                    step.reward[b] = r
            step_blocker.release()
