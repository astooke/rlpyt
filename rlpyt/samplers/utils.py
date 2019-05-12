
import psutil
import time
import multiprocessing as mp

from rlpyt.utils.buffers import build_buffer_from_examples


def initialize_worker(rank, seed, cpu, group=None):
    log_str = f"Sampler rank: {rank} initialized"
    try:
        p = psutil.Process()
        cpus = [cpu] if isinstance(cpu, int) else cpu  # list or tuple
        p.cpu_affinity(cpus)
        log_str += f", CPU Affinity: {p.cpu_affinity()}"
    except AttributeError:
        pass
    if seed is not None:
        ext.set_seed(seed)  # TODO.
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed: {seed}"
    logger.log(log_str)  # TODO.


def build_samples_buffers(agent, env, batch_spec, agent_shared=True, env_shared=True):
    o = env.reset()
    a = env.action_space.sample()
    a, agent_info = agent.get_action(o, a, 0.)
    o, r, d, env_info = env.step(a)
    agent_examples = dict(  # One example: no T or B dimension.
        actions=a,
        agent_infos=agent_info,
    )
    env_examples = dict(
        observations=o,
        rewards=r,
        dones=d,
        env_infos=env_info,
    )
    agent_buf = build_buffer_from_examples(agent_examples, batch_spec,
        shared=agent_shared)
    env_buf = build_buffer_from_examples(env_examples, batch_spec,
        shared=env_shared)
    return agent_buf, env_buf


def build_par_objs(n_parallel):
    ctrl = Struct(
        quit=mp.RawValue(ctypes.c_bool, False),
        barrier_in=mp.Barrier(n_parallel + 1),
        barrier_out=mp.Barrier(n_parallel + 1),
    )
    traj_infos_queue = mp.Queue()
    return ctrl, traj_infos_queue


def view_worker_buf(master_buf, n, rank):
    """Returns Struct with sliced view along batch dimension, for all fields."""
    w_buf = Struct()
    for k, v in master_buf.items():
        if isinstance(v, dict):  # Recurse nested Struct.
            w_buf[k] = view_worker_buf(v, n, rank)
        else:
            B = v.shape[1]  # Total batch dimension.
            assert B % n == 0
            w_B = B // n  # Worker batch dimension.
            w_buf[k] = v[:, w_B * rank: w_B * (rank + 1)]  # Slice batch dim.
    return w_buf


def assemble_workers_kwargs(affinities, seed, env_buf, agent_buf=None):
    n = len(affinities.workers_cpus)
    workers_kwargs = list()
    for rank in range(n):
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinities.workers_cpus[rank],
            env_buf=view_worker_buf(env_buf, n, rank),
        )
        if agent_buf is not None:
            worker_kwargs["agent_buf"] = view_worker_buf(agent_buf, n, rank)
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
