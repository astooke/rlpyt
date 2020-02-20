
Q-Value Policy Gradient Implementations
=======================================

This page documents algorithms, agents, and models implemented for Q-value policy gradient methods.  (Much of the functionality around training and replay buffers looks similar to DQN.)

Deep Deterministc Policy Gradient (DDPG)
----------------------------------------

.. autoclass:: rlpyt.algos.qpg.ddpg.DDPG
    :members: __init__, initialize, async_initialize, optim_initialize, initialize_replay_buffer, optimize_agent, samples_to_buffer, mu_loss, q_loss
    :show-inheritance:

.. autoclass:: rlpyt.agents.qpg.ddpg_agent.DdpgAgent
    :members: __init__, initialize, q, q_at_mu, target_q_at_mu, step
    :show-inheritance:

.. autoclass:: rlpyt.models.qpg.mlp.MuMlpModel
    :members: __init__
    :show-inheritance:

.. autoclass:: rlpyt.models.qpg.mlp.QofMuMlpModel
    :members: __init__
    :show-inheritance:


Twin Delayed Deep Deterministic Policy Gradient (TD3)
-----------------------------------------------------

.. autoclass:: rlpyt.algos.qpg.td3.TD3
    :members: __init__, q_loss
    :show-inheritance:

.. autoclass:: rlpyt.agents.qpg.td3_agent.Td3Agent
    :members: __init__, q, target_q_at_mu
    :show-inheritance:

Soft Actor Critic (SAC)
-----------------------

.. autoclass:: rlpyt.algos.qpg.sac.SAC
    :members: __init__, initialize, optim_initialize, initialize_replay_buffer, optimize_agent, samples_to_buffer, loss
    :show-inheritance:

.. autoclass:: rlpyt.agents.qpg.sac_agent.SacAgent
    :members: __init__, q, target_q, pi
    :show-inheritance:

.. autoclass:: rlpyt.models.qpg.mlp.PiMlpModel




