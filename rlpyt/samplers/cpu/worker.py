

from rlpyt.utils.struct import Struct
from rlpyt.utils.quick_args import save_args
from rlpyt.samplers.base import BaseCollector, AgentInput
from rlpyt.samplers.utils import initialize_worker
from rlpyt.samplers.buffer import torchify_buffer, numpify_buffer


class Collector(BaseCollector):

    def __init__(self, agent, **kwargs):
        save_args(locals())
        super().__init__(**kwargs)
        self.need_reset = [False] * len(self.envs)

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
                if self.need_reset[i]:
                    continue
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[i])
                traj_infos[i].step(observation[i], action[i], r, env_info)
                d |= traj_infos[i].Length >= self.max_path_length
                if d:
                    self.need_reset[i] = True
                    completed_infos.append(traj_infos[i].terminate(o))
                    traj_infos[i] = self.TrajInfoCls()
                else:
                    observation[i] = o
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
        for i, need in enumerate(self.need_reset):
            if need:
                agent_input[i] = 0.
                agent_input.observation[i] = self.envs[i].reset()
                self.agent.reset_one(idx=i)
                self.need_reset[i] = False
        return agent_input


def sampling_process(common_kwargs, worker_kwargs):
    c, w = Struct(**common_kwargs), Struct(**worker_kwargs)

    initialize_worker(w.rank, w.seed, w.cpus)
    envs = [c.EnvCls(**c.env_kwargs) for _ in range(c.n_envs)]

    collector = Collector(
        envs=envs,
        agent=c.agent,
        samples_np=w.samples_np,
        max_path_length=c.max_path_length,
        TrajInfoCls=c.TrajInfoCls,
    )

    agent_input, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    c.agent.reset()
    ctrl = c.ctrl
    ctrl.barrier_out.wait()

    while True:
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        agent_input = collector.reset_if_needed(agent_input)
        agent_input, traj_infos, completed_infos = collector.collect_batch(
            agent_input, traj_infos)
        for info in completed_infos:
            c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()
