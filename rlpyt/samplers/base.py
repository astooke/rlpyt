

from collections import namedtuple
from rlpyt.utils.quick_args import save_args
from rlpyt.utils.struct import Struct


class TrajInfo(Struct):
    """
    Because it inits as a Struct, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()

    Intent: all attributes not starting with underscore "_" will be logged.

    (Can subclass for more fields.)
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for Struct behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._cur_discount = 1

    def step(self, _observation, _action, reward, _env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount

    def terminate(self, _final_observation):
        return self


class BaseSampler(object):

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_spec,
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            ):
        save_args(locals())

    def initialize(self, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        return NotImplementedError

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()


Samples = namedtuple("samples", ["agent", "env"])


AgentInputs = namedtuple("agent_inputs",
    ["observations", "actions", "rewards"])


class BaseCollector(object):

    def __init__(
            self,
            rank,
            envs,
            env_buf,
            max_path_length,
            TrajInfoCls,
            ):
        save_args(locals())
        self.horizon = len(env_buf.rewards)  # Time major.

    def start_envs(self, max_decorrelation_steps=0):
        """calls reset() on every env"""
        observations, actions, rewards = list(), list(), list()
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        for env in self.envs:
            observations.append(env.reset())
            actions.append(env.action_space.sample().fill(0))
            rewards.append(0.)
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        if self.rank == 0:
            logger.log(
                f"Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}"
            )
        if max_decorrelation_steps == 0:
            return (observations, actions, rewards), traj_infos
        for i, env in enumerate(self.envs):
            n_steps = int(get_random_fraction() * max_decorrelation_steps)
            if hasattr(env.action_space, "sample_n"):
                env_actions = env.action_space.sample_n(n_steps)
            else:
                env_actions = [env.action_space.sample() for _ in range(n_steps)]
            for a in env_actions:
                o, r, d, info = env.step(a)
                traj_infos[i].step(r, info)
                d |= traj_infos[i].Length >= self.max_path_length
                if d:
                    o = env.reset()
                    traj_infos[i] = self.TrajInfoCls()
            observations[i] = o
            actions[i] = a
            rewards[i] = r
        return AgentInputs(observations, actions, rewards), traj_infos

