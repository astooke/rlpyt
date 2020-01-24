
Distributions
=============

Distributions are used to select randomized actions during sampling, and for some algorithms to compute likelihood and related values for training.  Typically, the distribution is owned by the agent.  This page documents the implemented distributions and some methods--see the code for details.


.. autoclass:: rlpyt.distributions.base.Distribution
    :members: sample, kl, mean_kl, log_likelihood, likelihood_ratio, entropy, mean_entropy, perplexity, mean_perplexity

.. autoclass:: rlpyt.distributions.discrete.DiscreteMixin
    :members: to_onehot, from_onehot

.. autoclass:: rlpyt.distributions.categorical.Categorical
    :members: sample
    :show-inheritance:

.. autoclass:: rlpyt.distributions.epsilon_greedy.EpsilonGreedy
    :members: sample, set_epsilon
    :show-inheritance:

.. autoclass:: rlpyt.distributions.epsilon_greedy.CategoricalEpsilonGreedy
    :members: sample, set_z
    :show-inheritance:

.. autoclass:: rlpyt.distributions.gaussian.Gaussian
    :members: entropy, log_likelihood, sample_loglikelihood, sample, set_clip, set_squash, set_noise_clip, set_std

