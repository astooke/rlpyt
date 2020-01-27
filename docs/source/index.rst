.. rlpyt documentation master file, created by
   sphinx-quickstart on Thu Oct 31 11:51:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rlpyt's documentation!
=================================


rlpyt includes modular, optimized implementations of common deep RL algorithms in PyTorch, with unified infrastructure supporting all three major families of model-free algorithms: policy gradient, deep-q learning, and q-function policy gradient. It is intended to be a high-throughput code-base for small- to medium-scale research (large-scale meaning like OpenAI Dota with 100's GPUs). A conceptual overview is provided in the `white paper <https://arxiv.org/abs/1909.01500>`_, and the code (with examples) in the `github repository <https://github.com/astooke/rlpyt>`_.

This documentation aims to explain the intent of the code structure, to make it easier to use and modify (it might not detail every keyword argument as in a fixed library).  See the github README for installation instructions and other introductory notes.  Please share any questions or comments to do with documenantation on the github issues.

The sections are organized as follows.  First, several of the base classes are introduced.  Then, each algorithm family and associated agents and models are grouped together.  Infrastructure code such as the runner classes and sampler classes are covered next.  All the remaining components are covered thereafter, in no particular order.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pages/base.rst
   pages/pg.rst
   pages/dqn.rst
   pages/qpg.rst
   pages/runner.rst
   pages/sampler.rst
   pages/async_sampler.rst
   pages/collector.rst
   pages/distribution.rst
   pages/space.rst
   pages/model.rst
   pages/env.rst
   pages/replay.rst
   pages/nat.rst
   pages/util.rst
   pages/log.rst
   pages/launch.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
