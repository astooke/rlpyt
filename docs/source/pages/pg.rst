
Policy Gradient Implementations
===============================

This page documents the implemented policy gradient / actor-critic algorithms, agents, and models.


Algorithms
----------

.. autoclass:: rlpyt.algos.pg.base.PolicyGradientAlgo
    :members: initialize, process_returns
    :show-inheritance:

.. autoclass:: rlpyt.algos.pg.a2c.A2C
    :members: __init__, optimize_agent, loss
    :show-inheritance:

.. autoclass::  rlpyt.algos.pg.ppo.PPO
    :members: __init__, initialize, optimize_agent, loss
    :show-inheritance:


Agents
------

Continuous Actions
^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.agents.pg.gaussian.GaussianPgAgent
    :members: __call__, initialize, step, value
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.gaussian.RecurrentGaussianPgAgentBase
    :members: __call__, step, value
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.gaussian.RecurrentGaussianPgAgent
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.gaussian.AlternatingRecurrentGaussianPgAgent
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.mujoco.MujocoMixin
    :members: make_env_to_model_kwargs

.. autoclass:: rlpyt.agents.pg.mujoco.MujocoFfAgent
    :members: __init__
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.mujoco.MujocoLstmAgent
    :members: __init__
    :show-inheritance:


Discrete Actions
^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.agents.pg.categorical.CategoricalPgAgent
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.atari.AtariMixin
    :members: make_env_to_model_kwargs

.. autoclass:: rlpyt.agents.pg.atari.AtariFfAgent
    :members: __init__
    :show-inheritance:

.. autoclass:: rlpyt.agents.pg.atari.AtariLstmAgent
    :members: __init__
    :show-inheritance:


Models
------

.. autoclass:: rlpyt.models.pg.mujoco_ff_model.MujocoFfModel
    :members: __init__, forward
    :show-inheritance:

.. autoclass:: rlpyt.models.pg.mujoco_lstm_model.MujocoLstmModel
    :members: __init__, forward
    :show-inheritance:

.. autoclass:: rlpyt.models.pg.atari_ff_model.AtariFfModel
    :members: __init__, forward
    :show-inheritance:

.. autoclass:: rlpyt.models.pg.atari_lstm_model.AtariLstmModel
    :members: __init__, forward
    :show-inheritance:
