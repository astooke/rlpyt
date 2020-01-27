
Logger
======

The logger is nearly a direct copy from `rllab`, which implemented it as a module.  It provides convenient recording of diagnostics to the terminal, which is also saved to `debug.log`, tabular diagnostics to comma-separated file, `progress.csv`, and training snapshot files (e.g. agent parameters), `params.pkl`.  The logger is not extensively documented here; its usage is mostly exposed in the examples.

.. autofunction:: rlpyt.utils.logging.context.logger_context

.. autofunction:: rlpyt.utils.logging.context.add_exp_param
