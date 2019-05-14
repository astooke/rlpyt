
import psutil
import time
import multiprocessing as mp

from rlpyt.utils.seed import set_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer
from rlpyt.samplers.base import (Samples, AgentSamples, AgentSamplesBs,
    EnvSamples)


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
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed: {seed}"
    logger.log(log_str)


def build_samples_buffer(agent, env, batch_spec, bootstrap_value=False,
        agent_shared=True, env_shared=True):
    # One example: no T or B dimension.
    o = env.reset()
    a = env.action_space.sample()
    a, agent_info = agent.step(o, a, np.array(0, dtype="float32"))
    o, r, d, env_info = env.step(a)
    T, B = batch_spec

    all_action = buffer_from_example(a, (T + 1, B), agent_shared)
    action = all_action[1:]
    prev_action = all_action[:-1]  # Writing to action will populate prev_action.
    agent_info = buffer_from_example(agent_info, (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        prev_action=prev_action,
        agent_info=agent_info,
    )
    if bootstrap_value:
        bv = buffer_from_example(agent_info.value, (1, B), agent_shared)
        agent_buffer = AgentSamplesBs(*agent_buffer, boostrap_value=bv)

    observation = buffer_from_example(o, (T, B), env_shared)
    all_reward = buffer_from_example(r, (T + 1, B), env_shared)
    reward = all_reward[1:]
    prev_reward = all_reward[:-1]  # Writing to reward will populate prev_reward.
    done = buffer_from_example(d, (T, B), env_shared)
    env_info = buffer_from_example(env_info, (T, B), env_shared)
    env_buffer = EnvSamples(
        observation=observation,
        reward=reward,
        prev_reward=prev_reward,
        done=done,
        env_info=env_info,
    )
    samples_np = Samples(agent_buffer, env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    return samples_pyt, samples_np


def build_par_objs(n_parallel):
    ctrl = Struct(
        quit=mp.RawValue(ctypes.c_bool, False),
        barrier_in=mp.Barrier(n_parallel + 1),
        barrier_out=mp.Barrier(n_parallel + 1),
    )
    traj_infos_queue = mp.Queue()
    return ctrl, traj_infos_queue


def assemble_workers_kwargs(affinities, seed, samples_pyt, samples_np, n_envs):
    workers_kwargs = list()
    for rank in range(len(affinities.workers_cpus)):
        slice_B = slice(rank * n_envs, (rank + 1) * n_envs)
        worker_kwargs = dict(
            rank=rank,
            seed=seed + rank,
            cpus=affinities.workers_cpus[rank],
            samples_np=samples_np[:, slice_B],
        )
        workers_kwargs.append(worker_kwargs)
    return workers_kwargs
