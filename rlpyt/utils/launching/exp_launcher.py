
import subprocess
import time
import os
import os.path as osp

from rlpyt.utils.launching.affinity import get_n_run_slots, prepend_run_slot, get_affinity
from rlpyt.utils.logging.context import get_log_dir
from rlpyt.utils.launching.variant import save_variant


def log_exps_tree(exp_dir, log_dirs):
    os.makedirs(exp_dir, exist_ok=True)
    with open(osp.join(exp_dir, "experiments_tree.txt"), "w") as f:
        [f.write(log_dir + "\n") for log_dir in log_dirs]


def launch_experiment(script, run_slot, affinity_code, log_dir, variant, run_ID, args):
    slot_affinity_code = prepend_run_slot(run_slot, affinity_code)
    affinity = get_affinity(slot_affinity_code)
    call_list = []
    if affinity["all_cpus"]:
        cpus = ",".join(str(c) for c in affinity["all_cpus"])
        call_list += ["taskset", "-c", cpus]  # PyTorch obeys better than just psutil.
    call_list += ["python", script, slot_affinity_code, log_dir, str(run_ID)]
    call_list += [str(a) for a in args]
    save_variant(variant, log_dir)
    print("\ncall string:\n", " ".join(call_list))
    p = subprocess.Popen(call_list)
    return p


def run_experiments(script, affinity_code, experiment_title, runs_per_setting,
        variants, log_dirs, common_args=None, runs_args=None):
    n_run_slots = get_n_run_slots(affinity_code)
    exp_dir = get_log_dir(experiment_title)
    procs = [None] * n_run_slots
    common_args = () if common_args is None else common_args
    assert len(variants) == len(log_dirs)
    if runs_args is None:
        runs_args = [()] * len(variants)
    assert len(runs_args) == len(variants)
    log_exps_tree(exp_dir, log_dirs)
    n, total = 0, runs_per_setting * len(variants)
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
                        n += 1
                        with open(osp.join(exp_dir, "num_launched.txt"), "w") as f:
                            f.write(f"Experiments launched so far: {n} out of {total}.")
                        break
                if not launched:
                    time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # Don't return until they are all done.
