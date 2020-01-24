
Environments
============

The Atari Environment and a Gym Env Wrapper are included in rlpyt.

Atari
-----

.. autoclass:: rlpyt.envs.atari.atari_env.AtariTrajInfo
    :show-inheritance:

.. autoclass:: rlpyt.envs.atari.atari_env.AtariEnv
    :members: reset
    :show-inheritance:


Gym Wrappers
------------

.. autoclass:: rlpyt.envs.gym.GymEnvWrapper
    :members: step, reset, spaces

.. autoclass:: rlpyt.envs.gym.EnvInfoWrapper
    :members: step

.. autofunction:: rlpyt.envs.gym.make
