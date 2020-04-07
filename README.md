<img align="right" width="205" height="109" src="/images/bair_logo.png">

# rlpyt
[![tests](https://github.com/astooke/rlpyt/workflows/tests/badge.svg)](https://github.com/astooke/rlpyt/actions)
[![codecov](https://codecov.io/gh/astooke/rlpyt/graph/badge.svg)](https://codecov.io/gh/astooke/rlpyt)
[![Docs](https://readthedocs.org/projects/rlpyt/badge/?version=latest&style=flat)](https://rlpyt.readthedocs.io/en/latest/)
[![GitHub license](https://img.shields.io/github/license/astooke/rlpyt)](https://github.com/astooke/rlpyt/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/astooke/rlpyt)](https://github.com/astooke/rlpyt/issues)
[![GitHub stars](https://img.shields.io/github/stars/astooke/rlpyt)](https://github.com/astooke/rlpyt/stargazers)

## Deep Reinforcement Learning in PyTorch

*NEW: extended documentation available at [https://rlpyt.readthedocs.io](https://rlpyt.readthedocs.io)  (as of 27 Jan 2020)*


Modular, optimized implementations of common deep RL algorithms in PyTorch, with unified infrastructure supporting all three major families of model-free algorithms: policy gradient, deep-q learning, and q-function policy gradient.  Intended to be a high-throughput code-base for small- to medium-scale research (large-scale meaning like OpenAI Dota with 100's GPUs).  Key capabilities/features include:

* Run experiments in serial mode (helpful for debugging during development, or maybe sufficient for experiments).
* Run experiments fully parallelized, with options for parallel sampling and/or multi-GPU optimization.
  * Multi-GPU optimization uses PyTorch's DistributedDataParallel, which supports gradient reduction concurrent with backprop.
* Use CPU or GPU for training and/or batched action selection during environment sampling.
* Sampling and optimization synchronous or asynchronous (via replay buffer).
* Full support for recurrent agents.
  * All agents receive `observation, prev_action, prev_reward`.
  * Training data always organized with leading indexes as `[Time, Batch]`.
* Online or offline evaluation of agent diagnostics during training.
* Launching utilities for stacking/queueing sets of experiments in parallel on given **local** hardware resources (e.g. run 40 experiments on an 8-GPU machine with 1 experiment per GPU at a time).
* Compatible with the OpenAI Gym environment interface.<sup>1</sup>
* Modularity for easy modification / re-use of existing components.

### Implemented Algorithms
**Policy Gradient** A2C, PPO.

**Replay Buffers** (supporting both DQN + QPG) non-sequence and sequence (for recurrent) replay, n-step returns, uniform or prioritized replay, full-observation or frame-based buffer (e.g. for Atari, stores only unique frames to save memory, reconstructs multi-frame observations).

**Deep Q-Learning** DQN + variants: Double, Dueling, Categorical (up to Rainbow minus Noisy Nets), Recurrent (R2D2-style).  *Coming soon*: Implicit Quantile Networks?

**Q-Function Policy Gradient** DDPG, TD3, SAC.  *Coming soon*: Distributional DDPG?


### Getting Started
Follow the installation instructions below, and then get started in the examples folder.  Example scripts are ordered by increasing complexity.

For newcomers to deep RL, it may be better to get familiar with the algorithms using a different resource, such as the excellent OpenAI Spinning Up: [docs](https://spinningup.openai.com/en/latest/), [code](https://github.com/openai/spinningup).

### New data structure: `namedarraytuple`
Rlpyt introduces new object classes `namedarraytuple` for easier organization of collections of numpy arrays / torch tensors. (see `rlpyt/utils/collections.py`).  A `namedarraytuple` is essentially a `namedtuple` which exposes indexed or sliced read/writes into the structure.  For example, consider writing into a (possibly nested) dictionary of arrays:
```python
for k, v in src.items():
  if isinstance(dest[k], dict):
    ..recurse..
  dest[k][slice_or_indexes] = v
 ```
 This code is replaced by the following:
 ```python
 dest[slice_or_indexes] = src
 ```
 Importantly, this syntax looks the same whether `dest` and `src` are indiviual numpy arrays or arbitrarily-structured collections of arrays (the structures of `dest` and `src` must match, or `src` can be a single value, or `None` is an empty placeholder).  Rlpyt uses this data structure extensively--different elements of training data are organized with the same leading dimensions, making it easy to interact with desired time- or batch-dimensions.

This is also intended to support environments with multi-modal observations or actions.  For example, rather than flattening joint-angle and camera-image observations into one observation vector, the environment can store them as-is into a `namedarraytuple` for the observation, and in the forward method of the model, `observation.joint` and `observation.image` can be fed into the desired layers.  Intermediate infrastructure code doesn’t change.

## Future Developments.

Overall the code is stable, but might still develop, changes may occur.  Open to suggestions/contributions for other established algorithms to add or other developments to support more use cases--please see our simple [contribution guidelines](CONTRIBUTING.md).



## Visualization

This package does not include its own visualization, as the logged data is compatible with previous editions (see below). For more features, use [https://github.com/vitchyr/viskit](https://github.com/vitchyr/viskit).


## Installation

1.  Clone this repository to the local machine.

2. Install the anaconda environment appropriate for the machine.
```
conda env create -f linux_[cpu|cuda9|cuda10].yml
source activate rlpyt
```

3. Either A) Edit the PYTHONPATH to include the rlpyt directory, or
          B) Install as editable python package
```
#A
export PYTHONPATH=path_to_rlpyt:$PYTHONPATH

#B
pip install -e .
```

4. Install any packages / files pertaining to desired environments (e.g. gym, mujoco).  Atari is included.

Hint: for easy access, add the following to your `~/.bashrc` (might substitute `conda` for `source`).
```
alias rlpyt="source activate rlpyt; cd path_to_rlpyt"
```

## Extended Notes

For more discussion, please see the [white paper on Arxiv](https://arxiv.org/abs/1909.01500).  If you use this repository in your work or otherwise wish to cite it, please make reference to the white paper.



### Code Organization

The class types perform the following roles:

* **Runner** - Connects the `sampler`, `agent`, and `algorithm`; manages the training loop and logging of diagnostics.
  * **Sampler** - Manages `agent` / `environment` interaction to collect training data, can initialize parallel workers.
    * **Collector** - Steps `environments` (and maybe operates `agent`) and records samples, attached to `sampler`.
      * **Environment** - The task to be learned.
        * **Observation Space/Action Space** - Interface specifications from `environment` to `agent`.
      * **TrajectoryInfo** - Diagnostics logged on a per-trajectory basis.
  * **Agent** - Chooses control action to the `environment` in `sampler`; trained by the `algorithm`.  Interface to `model`.
    * **Model** - Torch neural network module, attached to the `agent`.
    * **Distribution** - Samples actions for stochastic `agents` and defines related formulas for use in loss function, attached to the `agent`.
  * **Algorithm** - Uses gathered samples to train the `agent` (e.g. defines a loss function and performs gradient descent).
    * **Optimizer** - Training update rule (e.g. Adam), attached to the `algorithm`.
    * **OptimizationInfo** - Diagnostics logged on a per-training batch basis.

### Historical, Scaling, Interfaces

This code is a revision and extension of [accel_rl](https://github.com/astooke/accel_rl), which explored scaling RL in the Atari domain using Theano.  Scaling results were recorded here: [A. Stooke & P. Abbeel, "Accelerated Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1803.02811).  For an insightful study of batch-size scaling across deep learning including RL, see [S. McCandlish, et. al "An Empirical Model of Large-Batch Training"](https://arxiv.org/abs/1812.06162).

Accel_rl was inspired by [rllab](https://github.com/rll/rllab) (the `logger` here is nearly a direct copy).  Rlpyt follows the rllab interfaces: agents output `action, agent_info`, environments output `observation, reward, done, env_info`.  In general in rlpyt, agent inputs/outputs are torch tensors, and environment inputs/ouputs are numpy arrays, with conversions handled automatically.

1.  Regarding OpenAI Gym compatibility, rlpyt uses a `namedtuple` for `env_info` rather than a `dict`.  This makes for easier data recording but does require the same fields to be output at every environment step.  An environment wrapper is provided.  Wrappers are also provided for Gym spaces to convert to rlpyt spaces (notably `Dict` to `composite`).

### Acknowledgements
Thanks for support / mentoring from Pieter Abbeel, the Fannie & John Hertz Foundation, NVIDIA, Max Jaderberg, OpenAI, and the BAIR community.  And thanks in advance to any contributors!

Happy reinforcement learning!
