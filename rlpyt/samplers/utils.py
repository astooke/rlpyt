
import multiprocessing as mp
import ctypes
import numpy as np

from rlpyt.utils.buffer import buffer_from_example, torchify_buffer
from rlpyt.utils.collections import AttrDict
from rlpyt.agents.base import AgentInput
from rlpyt.samplers.collections import (Samples, AgentSamples, AgentSamplesBs,
    EnvSamples)


def build_samples_buffer(agent, env, batch_spec, bootstrap_value=False,
        agent_shared=True, env_shared=True, build_step_buffer=False):
    """Step/reset agent and env in subprocess, so it doesn't affect settings
    in master before forking workers (e.g. torch num_threads (MKL) may be set
    at first forward computation.)"""
    mgr = mp.Manager()
    examples = mgr.dict()
    w = mp.Process(target=get_example_outputs, args=(agent, env, examples))
    w.start()
    w.join()

    T, B = batch_spec
    all_action = buffer_from_example(examples["action"], (T + 1, B), agent_shared)
    action = all_action[1:]
    prev_action = all_action[:-1]  # Writing to action will populate prev_action.
    agent_info = buffer_from_example(examples["agent_info"], (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        prev_action=prev_action,
        agent_info=agent_info,
    )
    if bootstrap_value:
        bv = buffer_from_example(examples["agent_info"].value, (1, B), agent_shared)
        agent_buffer = AgentSamplesBs(*agent_buffer, bootstrap_value=bv)

    observation = buffer_from_example(examples["observation"], (T, B), env_shared)
    all_reward = buffer_from_example(examples["reward"], (T + 1, B), env_shared)
    reward = all_reward[1:]
    prev_reward = all_reward[:-1]  # Writing to reward will populate prev_reward.
    done = buffer_from_example(examples["done"], (T, B), env_shared)
    env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
    env_buffer = EnvSamples(
        observation=observation,
        reward=reward,
        prev_reward=prev_reward,
        done=done,
        env_info=env_info,
    )
    samples_np = Samples(agent_buffer, env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    if build_step_buffer:
        raise NotImplementedError
        # return samples_pyt, samples_np, step_pyt, step_np
    return samples_pyt, samples_np


def build_par_objs(n, sync=False, groups=1):
    ctrl = AttrDict(
        quit=mp.RawValue(ctypes.c_bool, False),
        barrier_in=mp.Barrier(n * groups + 1),
        barrier_out=mp.Barrier(n * groups + 1),
    )
    traj_infos_queue = mp.Queue()
    if sync:
        step_blockers = [[mp.Semaphore(0) for _ in range(n)] for _ in range(groups)]
        act_waiters = [[mp.Semaphore(0) for _ in range(n)] for _ in range(groups)]
        if groups == 1:
            step_blockers = step_blockers[0]
            act_waiters = act_waiters[0]
        sync = AttrDict(step_blockers=step_blockers, act_waiters=act_waiters)
        return ctrl, traj_infos_queue, sync
    return ctrl, traj_infos_queue


def get_example_outputs(agent, env, examples):
    """Do this in a sub-process to avoid setup conflict in master/workers."""
    o = env.reset()
    a = env.action_space.sample()
    print("get_example_outputs -- sampled action: ", a)
    o, r, d, env_info = env.step(a)
    r = np.asarray(r, dtype="float32")  # Important to match dtype here.
    agent.reset()
    agent_input = torchify_buffer(AgentInput(o, a, r))
    a, agent_info = agent.step(*agent_input)
    examples["observation"] = o  # Copy to master in mp managed dictionary.
    examples["reward"] = r
    examples["done"] = d
    examples["env_info"] = env_info
    examples["action"] = a  # Probably OK to put torch tensor here, could numpify.
    examples["agent_info"] = agent_info
