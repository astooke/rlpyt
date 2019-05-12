

from rlpyt.util.struct import Struct
from rlpyt.utils.quick_args import save_args
from rlpyt.sampler.base import BaseCollector, AgentInputs
from rlpyt.sampler.utils import initialize_worker


class Collector(BaseCollector):

    def __init__(self, agent, agent_buf, **kwargs):
        save_args(locals())
        super().__init__(**kwargs)
        self.need_reset = [False] * len(self.envs)

    def collect_batch(self, agent_inputs, traj_infos):
        env_buf, agent_buf = (self.env_buf, self.agent_buf)
        completed_infos = list()
        observations, actions, rewards = agent_inputs
        agent_buf.prev_actions[0] = actions  # Record leading prev_action and prev_reward.
        env_buf.prev_rewards[0] = rewards
        for s in range(self.horizon):
            env_buf.observations[s] = observations
            actions, agent_infos = self.agent.get_actions(observations, actions, rewards)
            for i, env in enumerate(self.envs):
                if self.need_reset[i]:
                    continue
                o, r, d, env_info = env.step(actions[i])
                traj_infos[i].step(observations[i], actions[i], r, env_info)
                d |= traj_infos[i].Length >= self.max_path_length
                if d:
                    self.need_reset[i] = True
                    completed_infos.append(traj_infos[i].terminate(o))
                    traj_infos[i] = self.TrajInfoCls()
                else:
                    observations[i] = o
                rewards[i] = r
                env_buf.dones[s, i] = d
                if env_info:
                    env_buf.env_infos[s, i] = env_info
            agent_buf.actions[s] = actions
            env_buf.rewards[s] = rewards
            agent_buf.agent_infos[s] = agent_infos

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(observations, actions, rewards)

        return AgentInputs(observations, actions, rewards), traj_infos, completed_infos

    def reset_if_needed(self, agent_inputs):
        for i, need in enumerate(self.need_reset):
            if need:
                agent_inputs[i] = 0.
                agent_inputs.observations[i] = self.envs[i].reset()
                self.agent.reset_one(idx=i)
                self.need_reset[i] = False
        return agent_inputs


def sampling_process(common_kwargs, worker_kwargs):
    c, w = Struct(**common_kwargs), Struct(**worker_kwargs)

    initialize_worker(w.rank, w.seed, w.cpus)
    envs = [c.EnvCls(**c.env_kwargs) for _ in range(c.n_envs)]

    collector = Collector(
        envs=envs,
        agent=c.agent,
        agent_buf=w.agent_buf,
        env_buf=w.env_buf,
        max_path_length=c.max_path_length,
        TrajInfoCls=c.TrajInfoCls,
    )

    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    c.agent.reset()
    ctrl = c.ctrl
    ctrl.barrier_out.wait()

    while True:
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        agent_inputs = collector.reset_if_needed(agent_inputs)
        agent_inputs, traj_infos, completed_infos = collector.collect_batch(
            agent_inputs, traj_infos)
        for info in completed_infos:
            c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()
