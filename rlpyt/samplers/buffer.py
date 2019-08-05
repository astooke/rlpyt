
import multiprocessing as mp
import numpy as np

from rlpyt.utils.buffer import buffer_from_example, torchify_buffer
from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.collections import (Samples, AgentSamples, AgentSamplesBsv,
    EnvSamples)


def build_samples_buffer(agent, env, batch_spec, bootstrap_value=False,
        agent_shared=True, env_shared=True, subprocess=True, examples=None):
    """Recommended to step/reset agent and env in subprocess, so it doesn't
    affect settings in master before forking workers (e.g. torch num_threads
    (MKL) may be set at first forward computation.)"""
    if examples is None:
        if subprocess:
            mgr = mp.Manager()
            examples = mgr.dict()  # Examples pickled back to master.
            w = mp.Process(target=get_example_outputs,
                args=(agent, env, examples, subprocess))
            w.start()
            w.join()
        else:
            examples = dict()
            get_example_outputs(agent, env, examples)

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
        agent_buffer = AgentSamplesBsv(*agent_buffer, bootstrap_value=bv)

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
    samples_np = Samples(agent=agent_buffer, env=env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    return samples_pyt, samples_np, examples


def get_example_outputs(agent, env, examples, subprocess=False):
    """Do this in a sub-process to avoid setup conflict in master/workers (e.g.
    MKL)."""
    if subprocess:  # i.e. in subprocess.
        import torch
        torch.set_num_threads(1)  # Some fix to prevent MKL hang.
    o = env.reset()
    a = env.action_space.sample()
    o, r, d, env_info = env.step(a)
    r = np.asarray(r, dtype="float32")  # Must match torch float dtype here.
    agent.reset()
    agent_inputs = torchify_buffer(AgentInputs(o, a, r))
    a, agent_info = agent.step(*agent_inputs)
    if "prev_rnn_state" in agent_info:
        # Agent leaves B dimension in, strip it: [B,N,H] --> [N,H]
        agent_info = agent_info._replace(prev_rnn_state=agent_info.prev_rnn_state[0])
    examples["observation"] = o
    examples["reward"] = r
    examples["done"] = d
    examples["env_info"] = env_info
    examples["action"] = a  # OK to put torch tensor here, could numpify.
    examples["agent_info"] = agent_info
