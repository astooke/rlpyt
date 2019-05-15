
import subprocess
import time

from rlpyt.utils.launching.affinities import get_run_slots, prepend_run_slot


def launch_exp(script, run_slot_affinities_code, common_args, run_args):
    call_list = ["python", script, run_slot_affinities_code]
    call_list += [str(c) for c in common_args] + [str(r) for r in run_args]
    print("\ncall string:\n", " ".join(call_list))
    p = subprocess.Popen(call_list)
    return p


def run_exps(script, affinities_code, common_args, runs_args):
    n_run_slots = get_run_slots(affinities_code)
    procs = [None] * n_run_slots
    for run_args in runs_args:
        launched = False
        while not launched:
            for run_slot, p in enumerate(procs):
                if p is None or p.poll() is not None:
                    run_slot_affinities_code = prepend_run_slot(run_slot,
                        affinities_code)
                    procs[run_slot] = launch_exp(script,
                        run_slot_affinities_code, common_args, run_args)
                    launched = True
                    break
            if not launched:
                time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # Don't return until they are all done.
