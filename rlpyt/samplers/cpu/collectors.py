
import numpy as np

from rlpyt.samplers.collectors import DecorrelatingStartCollector
from rlpyt.samplers.base import BaseEvalCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer, buffer_from_example


class ResetCollector(DecorrelatingStartCollector):

    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, agent_info[b],
                    env_info)
                if d:
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.dones[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos


class WaitResetCollector(DecorrelatingStartCollector):

    mid_batch_reset = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_reset = [False] * len(self.envs)

    def collect_batch(self, agent_inputs, traj_infos):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                if self.need_reset[b]:
                    continue
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, agent_info[b],
                    env_info)
                if d:
                    self.need_reset[b] = True
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                else:
                    observation[b] = o
                reward[b] = r
                env_buf.dones[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos

    def reset_if_needed(self, agent_inputs):
        for b, need in enumerate(self.need_reset):
            if need:
                agent_inputs[b] = 0.
                agent_inputs.observation[b] = self.envs[b].reset()
                self.agent.reset_one(idx=b)
                self.need_reset[b] = False
        return agent_inputs


class EvalCollector(BaseEvalCollector):

    def collect_evaluation(self):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in enumerate(self.envs):
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        action = buffer_from_example(self.envs[0].action_space.sample(
            len(self.envs), null=True))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        completed_infos = list()
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r,
                    agent_info[b], env_info)
                if getattr(env_info, "need_reset", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                    self.agent.reset_one(b)
            observation[b] = o
            reward[b] = o
            if self.ctrl.stop_eval.value:
                break
        return completed_infos
