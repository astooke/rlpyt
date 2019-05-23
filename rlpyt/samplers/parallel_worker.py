
import psutil
import time
import torch

from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import logger
from rlpyt.utils.seed import set_seed


def initialize_worker(rank, seed=None, cpu=None, torch_threads=None, group=None):
    log_str = f"Sampler rank {rank} initialized"
    p = psutil.Process()
    if cpu is not None:
        p.cpu_affinity([cpu] if isinstance(cpu, int) else cpu)
    log_str += f", CPU affinity {p.cpu_affinity()}"
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    log_str += f", Torch threads {torch.get_num_threads()}"
    if seed is not None:
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed {seed}"
    logger.log(log_str)


def sampling_process(common_kwargs, worker_kwargs):
    """Arguments fed from the Sampler class in master process."""
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads,
        w.get("group", None))
    envs = [c.EnvCls(**c.env_kwargs) for _ in range(c.n_envs)]
    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agent", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    collector.start_agent()
    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        agent_inputs = collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        agent_inputs, traj_infos, completed_infos = collector.collect_batch(
            agent_inputs, traj_infos)
        for info in completed_infos:
            c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs:
        env.terminate()
