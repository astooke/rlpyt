

from rlpyt.utils.quick_args import save_args
from rlpyt.samplers.base import BaseCollector, AgentInput
from rlpyt.samplers.buffer import torchify_buffer, numpify_buffer


class EpisodicLivesCollector(BaseCollector):
    """Allows the learning agent to receive "done" signal but only reset the
    env with additional signal "need_reset" in env_info.  Still pause and
    reset the agent+env, so that recurrent agents reintialize at the next
    batch. So can track trajectory info online."""

    def __init__(self, agent, **kwargs):
        save_args(locals())
        super().__init__(**kwargs)
        self.need_env_reset = [False] * len(self.envs)
        self.need_agent_reset = [False] * len(self.envs)

    def collect_batch(self, agent_input, traj_infos):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_input
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_input)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        for s in range(self.horizon):
            env_buf.observation[s] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for i, env in enumerate(self.envs):
                if self.need_agent_reset[i]:
                    continue
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[i])
                traj_infos[i].step(observation[i], action[i], r, env_info)
                env_d = getattr(env_info, "need_reset", False)
                env_d |= traj_infos[i].Length >= self.max_path_length
                d |= env_d
                if env_d:
                    self.need_env_reset[i] = True
                    completed_infos.append(traj_infos[i].terminate(o))
                    traj_infos[i] = self.TrajInfoCls()
                else:
                    observation[i] = o
                self.need_agent_resest[i] = d
                reward[i] = r
                env_buf.dones[s, i] = d
                if env_info:
                    env_buf.env_info[s, i] = env_info
            agent_buf.action[s] = action
            env_buf.reward[s] = reward
            if agent_info:
                agent_buf.agent_info[s] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInput(observation, action, reward), traj_infos, completed_infos

    def reset_if_needed(self, agent_input):
        for i, (need_agent, need_env) in enumerate(zip(
                self.need_agent_reset, self.need_env_reset)):
            if need_agent:
                agent_input.prev_action[i] = 0.
                agent_input.prev_reward[i] = 0.
                self.agent.reset_one(idx=i)
            if need_env:
                agent_input.observation[i] = self.envs[i].reset()
            self.need_agent_reset[i] = False
            self.need_env_reset[i] = False
        return agent_input

