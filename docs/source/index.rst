.. rlpyt documentation master file, created by
   sphinx-quickstart on Thu Oct 31 11:51:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rlpyt's documentation!
=================================

rlpyt includes modular, optimized implementations of common deep RL algorithms in PyTorch, with unified infrastructure supporting all three major families of model-free algorithms: policy gradient, deep-q learning, and q-function policy gradient. It is intended to be a high-throughput code-base for small- to medium-scale research (large-scale meaning like OpenAI Dota with 100's GPUs). A conceptual overview is provided in the white paper (link), and the code (with examples) in the github repository (link).

This documentation aims to explain the intent of the code structure, for the purpose of making it easier to use and modify (it might not detail every keyword argument as in a fixed library).  See the github README for installation instructions and other introductory notes.  

The sections are organized as...


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pages/base.rst
   pages/runner.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
