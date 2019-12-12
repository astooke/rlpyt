
Base Classes and Interfaces
===========================

This page describes the base classes for several main components: runner, algorithm, agent, and environment, including how they are intended to interact.  More details on specific instances of these components appear in following pages.

Commonly, these classes will simply store their keyword arguments when instantiated, and actual initialization occurs in methods to be called later.


Runner
------
The runner orchestrates all the other components to run the training loop.  During startup it initializes the sampler, algorithm, and agent.  It exposes a ``train()`` method which conducts an entire RL training run.  The implemented runners all alternate between gathering experience using the ``sampler.obtain_samples()`` method and training the agent using the ``algo.optimize_agent()`` method.  The runner also manages logging to record agent performance during training.  Different runner classes may be used depending on hardware configuration (e.g. multi-GPU) and agent evaluation mode (i.e. offline vs online).


Algorithm
---------
The algorithm performs all processing of gathered samples for the purpose of training the agent, through the ``optimize_agent()`` method.  The ``optim_state_dict()`` and ``load_optim_state_dict()`` methods should return/load optimizer state, e.g. RMSProp or Adam parameters.


Agent
-----
The agent class performs many functions, including: action-selection during sampling, returning policy-relevant values to use in training (e.g. action probabilities), storing recurrent state during sampling, managing model device, and performing model parameter communication between processes.  The agent is the interface between neural network and sampler and between neural network and algorithm.  Typically, each algorithm and environment combination will require its own agent functionality.

