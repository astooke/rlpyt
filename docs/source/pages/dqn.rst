
Deep Q-Learning Implementations
===============================

This page documents the implemented deep Q-learning algorithms, agents, and models.  Up to Rainbow, minus noisy nets, can be run using Categorical-DQN with the options for double-DQN, dueling heads, n-step returns, and prioritized replay.

DQN
---

.. autoclass:: rlpyt.algos.dqn.dqn.DQN
    :members: __init__, initialize, async_initialize, optim_initialize, initialize_replay_buffer, optimize_agent, samples_to_buffer, loss
    :show-inheritance:

.. autoclass:: rlpyt.agents.dqn.epsilon_greedy.EpsilonGreedyAgentMixin
    :members: __init__, collector_initialize, make_vec_eps, sample_mode, eval_mode

.. autoclass:: rlpyt.agents.dqn.dqn_agent.DqnAgent
    :members: __call__, initialize, step, target, update_target
    :show-inheritance:

.. autoclass:: rlpyt.models.dqn.atari_dqn_model.AtariDqnModel
    :members: __init__, forward
    :show-inheritance:


Categorical-DQN
---------------

.. autoclass:: rlpyt.algos.dqn.cat_dqn.CategoricalDQN
    :members: __init__, loss
    :show-inheritance:

.. autoclass:: rlpyt.agents.dqn.catdqn_agent.CatDqnAgent
    :members: __init__, step
    :show-inheritance:

.. autoclass:: rlpyt.models.dqn.atari_catdqn_model.AtariCatDqnModel
    :members: __init__, forward
    :show-inheritance:


Recurrent DQN (R2D1)
--------------------

.. autoclass:: rlpyt.algos.dqn.r2d1.R2D1
    :members:  __init__, initialize_replay_buffer, optimize_agent, compute_input_priorities, loss, value_scale, inv_value_scale
    :show-inheritance:

.. autoclass:: rlpyt.agents.dqn.r2d1_agent.R2d1AgentBase
    :members: step
    :show-inheritance:

.. autoclass:: rlpyt.agents.dqn.r2d1_agent.R2d1Agent
    :show-inheritance:

.. autoclass:: rlpyt.models.dqn.atari_r2d1_model.AtariR2d1Model
    :members: __init__
    :show-inheritance:


Miscellaneous
-------------

.. autoclass:: rlpyt.models.dqn.dueling.DuelingHeadModel
    :members: forward, advantage
    :show-inheritance:

.. autoclass:: rlpyt.models.dqn.dueling.DistributionalDuelingHeadModel
    :show-inheritance:

.. autoclass:: rlpyt.models.dqn.atari_catdqn_model.DistributionalHeadModel
    :show-inheritance:



