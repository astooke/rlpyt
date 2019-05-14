

from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save_args
from rlpyt.utils.struct import Struct


Samples = namedarraytuple("Samples", ["agent", "env"])
AgentSamples = namedarraytuple("AgentSamples",
    ["action", "prev_action", "agent_info"])
AgentSamplesBs = namedarraytuple("AgentSamples",
    ["action", "prev_action", "agent_info", "boostrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
    ["observation", "reward", "prev_reward", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """
    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


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
            max_path_length=int(1e6),
            max_decorrelation_steps=100,
            TrajInfoCls=TrajInfo,
            ):
        assert isinstance(batch_spec, BatchSpec)
        save_args(locals())

    def initialize(self, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        return NotImplementedError  # type: Samples

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()


AgentInput = namedarraytuple("AgentInput",
    ["observation", "prev_action", "prev_reward"])


class BaseCollector(object):

    def __init__(
            self,
            rank,
            envs,
            samples_buffer,
            max_path_length,
            TrajInfoCls,
            ):
        save_args(locals())
        self.horizon = len(samples_buffer.env.reward)  # Time major.

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

