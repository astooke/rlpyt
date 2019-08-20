
from contextlib import contextmanager
import datetime
import os
import os.path as osp
import json

from rlpyt.utils.logging import logger

LOG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../data'))


def get_log_dir(experiment_name):
    yyyymmdd = datetime.datetime.today().strftime("%Y%m%d")
    log_dir = osp.join(LOG_DIR, "local", yyyymmdd, experiment_name)
    return log_dir


@contextmanager
def logger_context(log_dir, run_ID, name, log_params=None, snapshot_mode="none"):
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_log_tabular_only(False)
    log_dir = osp.join(log_dir, f"run_{run_ID}")
    exp_dir = osp.abspath(log_dir)
    if LOG_DIR != osp.commonpath([exp_dir, LOG_DIR]):
        print(f"logger_context received log_dir outside of {LOG_DIR}: "
            f"prepending by {LOG_DIR}/local/<yyyymmdd>/")
        exp_dir = get_log_dir(log_dir)
    tabular_log_file = osp.join(exp_dir, "progress.csv")
    text_log_file = osp.join(exp_dir, "debug.log")
    params_log_file = osp.join(exp_dir, "params.json")

    logger.set_snapshot_dir(exp_dir)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.push_prefix(f"{name}_{run_ID} ")

    if log_params is None:
        log_params = dict()
    log_params["name"] = name
    log_params["run_ID"] = run_ID
    with open(params_log_file, "w") as f:
        json.dump(log_params, f)

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
    # exp_folders = get_immediate_subdirectories(exp_dir)
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
                params[param_name] = param_val
                with open(params_f, "w") as f:
                    json.dump(params, f)


# def get_immediate_subdirectories(a_dir):
#     return [osp.join(a_dir, name) for name in os.listdir(a_dir)
#             if osp.isdir(osp.join(a_dir, name))]
