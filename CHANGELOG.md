# Change Log

Versioning updates and major additional components will be listed here.  Lots of little additions might not make this list.


## 07 July 2020

### Safe RL project
Added code for "Responsive Safety in RL by PID Lagrangian Methods" (arxiv link pending) (See README within project folder.)

### "projects" folder
Added a new "projects" folder ``rlpyt/rlpyt/projects`` to host code from research projects.

### rlpyt version 0.1.2
* Previous rlpyt version 0.1.1dev
* Updates for PyTorch 1.5.1 (from 1.2)
  - ``grad_norm`` logging (it's now a tensor)
  - ``DistributedDataParallelCPU`` is deprecated, uses ``DistributedDataParallel``
  - conda environment yml files refer to new PyTorch version