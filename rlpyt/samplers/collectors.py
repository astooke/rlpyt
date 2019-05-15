

from rlpyt.samplers.base import BaseCollector
from rlpyt.agents.base import AgentInput


class DecorrelatingStartCollector(BaseCollector):

    def start_envs(self, max_decorrelation_steps=0):
        """calls reset() on every env"""
        observation, prev_action, prev_reward = list(), list(), list()
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        for env in self.envs:
            observation.append(env.reset())
            prev_action.append(env.action_space.sample(null=True))
            prev_reward.append(0.)
        observation = np.array(observation)
        prev_action = np.array(prev_action)
        prev_reward = np.array(prev_reward)
        if self.rank == 0:
            logger.log(
                "Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}"
            )
        if max_decorrelation_steps == 0:
            return AgentInput(observation, prev_action, prev_reward), traj_infos
        for i, env in enumerate(self.envs):
            n_steps = int(get_random_fraction() * max_decorrelation_steps)
            env_actions = env.action_space.sample(n_steps)
            for a in env_actions:
                o, r, d, info = env.step(a)
                traj_infos[i].step(r, info)
                d = getattr(info, "need_reset", d)  # For episodic lives.
                d |= traj_infos[i].Length >= self.max_path_length
                if d:
                    o = env.reset()
                    a = env.action_space.sample(null=True)
                    r = 0.
                    traj_infos[i] = self.TrajInfoCls()
            observation[i] = o
            prev_action[i] = a
            prev_reward[i] = r
        return AgentInput(observation, prev_action, prev_reward), traj_infos
