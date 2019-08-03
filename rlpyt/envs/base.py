
from collections import namedtuple


EnvStep = namedtuple("EnvStep",
    ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", [])  # Define in env file.
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class Env:

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a namedtuple containing other diagnostic information from the previous action
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    def horizon(self):
        """Horizon of the environment, if it has one."""
        raise NotImplementedError

    def close(self):
        """Clean up operation."""
        pass

