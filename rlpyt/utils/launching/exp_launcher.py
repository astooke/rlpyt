
import subprocess
import time
import os
import os.path as osp
import sys

from rlpyt.utils.launching.affinity import get_n_run_slots, prepend_run_slot, affinity_from_code
from rlpyt.utils.logging.context import get_log_dir
from rlpyt.utils.launching.variant import save_variant


def log_exps_tree(exp_dir, log_dirs, runs_per_setting):
    os.makedirs(exp_dir, exist_ok=True)
    with open(osp.join(exp_dir, "experiments_tree.txt"), "w") as f:
        f.write(f"Experiment manager process ID: {os.getpid()}.\n")
        f.write("Number of settings (experiments) to run: "
            f"{len(log_dirs)}  ({runs_per_setting * len(log_dirs)}).\n\n")
        [f.write(log_dir + "\n") for log_dir in log_dirs]


def log_num_launched(exp_dir, n, total):
    with open(osp.join(exp_dir, "num_launched.txt"), "w") as f:
        f.write(f"Experiments launched so far: {n} out of {total}.\n")


def launch_experiment(
    script,
    run_slot,
    affinity_code,
    log_dir,
    variant,
    run_ID,
    args,
    python_executable=None,
):
    """Launches one learning run using ``subprocess.Popen()`` to call the
    python script.  Calls the script as:
    ``python {script} {slot_affinity_code} {log_dir} {run_ID} {*args}``
    If ``affinity_code["all_cpus"]`` is provided, then the call is prepended
    with ``tasket -c ..`` and the listed cpus (this is the most sure way to
    keep the run limited to these CPU cores).  Also saves the `variant` file.
    Returns the process handle, which can be monitored.
    """
    slot_affinity_code = prepend_run_slot(run_slot, affinity_code)
    affinity = affinity_from_code(slot_affinity_code)
    call_list = list()
    if isinstance(affinity, dict) and affinity.get("all_cpus", False):
        cpus = ",".join(str(c) for c in affinity["all_cpus"])
    elif isinstance(affinity, list) and affinity[0].get("all_cpus", False):
        cpus = ",".join(str(c) for aff in affinity for c in aff["all_cpus"])
    else:
        cpus = ()
    if cpus:
        call_list += ["taskset", "-c", cpus]  # PyTorch obeys better than just psutil.
    py = python_executable if python_executable else sys.executable or "python"
    call_list += [py, script, slot_affinity_code, log_dir, str(run_ID)]
    call_list += [str(a) for a in args]
    save_variant(variant, log_dir)
    print("\ncall string:\n", " ".join(call_list))
    p = subprocess.Popen(call_list)
    return p


def run_experiments(script, affinity_code, experiment_title, runs_per_setting,
        variants, log_dirs, common_args=None, runs_args=None):
    """Call in a script to run a set of experiments locally on a machine.  Uses
    the ``launch_experiment()`` function for each individual run, which is a 
    call to the ``script`` file.  The number of experiments to run at the same
    time is determined from the ``affinity_code``, which expresses the hardware
    resources of the machine and how much resource each run gets (e.g. 4 GPU
    machine, 2 GPUs per run).  Experiments are queued and run in sequence, with
    the intention to avoid hardware overlap.  Inputs ``variants`` and ``log_dirs``
    should be lists of the same length, containing each experiment configuration
    and where to save its log files (which have the same name, so can't exist
    in the same folder).

    Hint:
        To monitor progress, view the `num_launched.txt` file and `experiments_tree.txt`
        file in the experiment root directory, and also check the length of each
        `progress.csv` file, e.g. ``wc -l experiment-directory/.../run_*/progress.csv``.
    """
    n_run_slots = get_n_run_slots(affinity_code)
    exp_dir = get_log_dir(experiment_title)
    procs = [None] * n_run_slots
    common_args = () if common_args is None else common_args
    assert len(variants) == len(log_dirs)
    if runs_args is None:
        runs_args = [()] * len(variants)
    assert len(runs_args) == len(variants)
    log_exps_tree(exp_dir, log_dirs, runs_per_setting)
    num_launched, total = 0, runs_per_setting * len(variants)
    for run_ID in range(runs_per_setting):
        for variant, log_dir, run_args in zip(variants, log_dirs, runs_args):
            launched = False
            log_dir = osp.join(exp_dir, log_dir)
            os.makedirs(log_dir, exist_ok=True)
            while not launched:
                for run_slot, p in enumerate(procs):
                    if p is None or p.poll() is not None:
                        procs[run_slot] = launch_experiment(
                            script=script,
                            run_slot=run_slot,
                            affinity_code=affinity_code,
                            log_dir=log_dir,
                            variant=variant,
                            run_ID=run_ID,
                            args=common_args + run_args,
                        )
                        launched = True
                        num_launched += 1
                        log_num_launched(exp_dir, num_launched, total)
                        break
                if not launched:
                    time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # Don't return until they are all done.
