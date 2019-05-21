# rlpyt
## Deep Reinforcement Learning in PyTorch

Runs reinforcement learning algorithms with parallel sampling and GPU training, if available.  Highly modular (modifiable) and optimized codebase with functionality for launching large sets of parallel experiments locally on multi-GPU or many-core machines.

Based on [accel_rl](https://github.com/astooke/accel_rl), which in turn was based on [rllab](https://github.com/rll/rllab).

Follows the rllab interfaces: agents output `action, agent_info`, environments output `observation, reward, done, env_info`, but introduces new object classes `namedarraytuple` for easier organization.  This permits each output to be be either an individual numpy array [torch tensor] or an arbitrary collection of numpy arrays [torch tensors], without changing interfaces.  In general, agent inputs/outputs are torch tensors, and environment inputs/ouputs are numpy arrays, with conversions handled automatically.

Recurrent agents are supported, as training batches are organized with leading indexes as `[Time, Batch]`, and agents receive previous action and previous reward as input, in addition to the observation.

Start from `rlpyt/experiments/scripts/atari/pg/launch/launch_atari_ff_a2c_cpu.py` as a complete example, and follow the code backwards from there.  :)


## Current Status

Multi-GPU training within one learning run is not implemented (see [accel_rl](https://github.com/astooke/accel_rl) for hint of how it might be done, or maybe easier with PyTorch's data parallel functionality).  Stacking multiple experiments per machine is more effective for multiple runs / variations.

A2C is the first algorithm in place.  See [accel_rl](https://github.com/astooke/accel_rl) for similar implementations of other algorithms, including DQN+variants, which could be ported.


## Visualization

This package does not include its own visualization, as the logged data is compatible with previous editions (see above). For more features, use [https://github.com/vitchyr/viskit](https://github.com/vitchyr/viskit).


## Installation

1. Install the anaconda environment appropriate for the machine.
```
conda env create -f linux_[cpu|cuda9|cuda10].yml
source activate rlpyt
```

2. Either A) Edit the PYTHONPATH to include the rlpyt directory, or
          B) Install as editable python package
```
#A
export PYTHONPATH=path_to_rlpyt:$PYTHONPATH

#B
pip install -e .
```

3. Install any packages / files pertaining to desired environments.  Atari is included.


## Code Organization

The class types perform the following roles:

* `Runner` - Connects the sampler, agent, and algorithm; manages the training loop and logging of diagnostics.
  * `Sampler` - Manages agent / environment interaction to collect training data, can initialize parallel workers.
    * `Collector` - Steps environments (and maybe operates agent) and records samples, attached to sampler.
      * `Environment` - The task to be learned.
        * `Space` - Interface specifications from environment to agent.
  * `Agent` - Chooses control action to the environment; trained by the algorithm.  Interface to model.
    * `Model` - Neural network module, attached to the agent.
    * `Distribution` - Samples actions for stochastic agents and defines related formulas for use in loss function, attached to the agent.
  * `Algorithm` - Uses gathered samples to train the agent (e.g. defines a loss function and calls gradient descent).
    * `Optimizer` - Training update rule (e.g. Adam), attached to the algorithm.
