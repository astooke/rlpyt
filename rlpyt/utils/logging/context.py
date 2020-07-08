import datetime
import json
import os
import os.path as osp
from contextlib import contextmanager
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    print("Unable to import tensorboard SummaryWriter, proceeding without.")

from rlpyt.utils.logging import logger

LOG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../data'))


def get_log_dir(experiment_name, root_log_dir=None, date=True):
    yyyymmdd_hhmmss = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    yyyymmdd, hhmmss = yyyymmdd_hhmmss.split("-")
    root_log_dir = LOG_DIR if root_log_dir is None else root_log_dir
    log_dir = osp.join(root_log_dir, "local", yyyymmdd, hhmmss, experiment_name)
    return log_dir


@contextmanager
def logger_context(
    log_dir, run_ID, name, log_params=None, snapshot_mode="none", override_prefix=False,
    use_summary_writer=False,
):
    """Use as context manager around calls to the runner's ``train()`` method.
    Sets up the logger directory and filenames.  Unless override_prefix is
    True, this function automatically prepends ``log_dir`` with the rlpyt
    logging directory and the date: `path-to-rlpyt/data/yyyymmdd/hhmmss`
    (`data/` is in the gitignore), and appends with `/run_{run_ID}` to
    separate multiple runs of the same settings. Saves hyperparameters
    provided in ``log_params`` to `params.json`, along with experiment `name`
    and `run_ID`.

    Input ``snapshot_mode`` refers to how often the logger actually saves the
    snapshot (e.g. may include agent parameters).  The runner calls on the
    logger to save the snapshot at every iteration, but the input
    ``snapshot_mode`` sets how often the logger actually saves (e.g. snapshot
    may include agent parameters). Possible modes include (but check inside
    the logger itself):
        * "none": don't save at all
        * "last": always save and overwrite the previous
        * "all": always save and keep each iteration
        * "gap": save periodically and keep each (will also need to set the gap, not done here) 

    The cleanup operations after the ``yield`` close files but might not be
    strictly necessary if not launching another training session in the same
    python process.
    """
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_log_tabular_only(False)
    log_dir = osp.join(log_dir, f"run_{run_ID}")
    exp_dir = osp.abspath(log_dir)
    if LOG_DIR != osp.commonpath([exp_dir, LOG_DIR]) and not override_prefix:
        print(f"logger_context received log_dir outside of {LOG_DIR}: "
            f"prepending by {LOG_DIR}/local/<yyyymmdd>/<hhmmss>/")
        exp_dir = get_log_dir(log_dir)
    tabular_log_file = osp.join(exp_dir, "progress.csv")
    text_log_file = osp.join(exp_dir, "debug.log")
    params_log_file = osp.join(exp_dir, "params.json")

    logger.set_snapshot_dir(exp_dir)
    if use_summary_writer:
        logger.set_tf_summary_writer(SummaryWriter(exp_dir))
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.push_prefix(f"{name}_{run_ID} ")

    if log_params is None:
        log_params = dict()
    log_params["name"] = name
    log_params["run_ID"] = run_ID
    with open(params_log_file, "w") as f:
        json.dump(log_params, f, default=lambda o: type(o).__name__)

    yield

    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()


def add_exp_param(param_name, param_val, exp_dir=None, overwrite=False):
    """Puts a param in all experiments in immediate subdirectories.
    So you can write a new distinguising param after the fact, perhaps
    reflecting a combination of settings."""
    if exp_dir is None:
        exp_dir = os.getcwd()
    for sub_dir in os.walk(exp_dir):
        if "params.json" in sub_dir[2]:
            update_param = True
            params_f = osp.join(sub_dir[0], "params.json")
            with open(params_f, "r") as f:
                params = json.load(f)
                if param_name in params:
                    if overwrite:
                        print("Overwriting param: {}, old val: {}, new val: {}".format(
                            param_name, params[param_name], param_val))
                    else:
                        print("Param {} already found & overwrite set to False; "
                            "leaving old val: {}.".format(param_name, params[param_name]))
                        update_param = False
            if update_param:
                os.remove(params_f)
                if param_name in params and isinstance(params[param_name], dict) and isinstance(param_val, dict):
                    print(f"Param {param_name} is a dict and so is val, just updating.")
                    params[param_name].update(param_val)
                else:
                    params[param_name] = param_val
                with open(params_f, "w") as f:
                    json.dump(params, f, default=lambda o: type(o).__name__)


def check_progress(exp_dir=None):
    """Print to stdout the number of lines in all ``progress.csv`` files in
    the directory.  Call like:
     ``python -c 'from rlpyt.util.logging.context import check_progress;
     check_progress('path_to_dir')``
    """
    if exp_dir is None:
        exp_dir = os.getcwd()
    for sub_dir in os.walk(exp_dir):
        if "progress.csv" in sub_dir[2]:
            progress_f = osp.join(sub_dir[0], "progress.csv")
            os.system(f"wc -l {progress_f}")
