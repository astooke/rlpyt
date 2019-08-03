
import numpy as np

from rlpyt.samplers.gpu.base import GpuParallelSamplerBase
from rlpyt.samplers.gpu.action_server import (AlternatingActionServer,
    NoOverlapAlternatingActionServer)
from rlpyt.utils.logging import logger
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 20  # Time steps.


class AlternatingSamplerBase(GpuParallelSamplerBase):
    """Environment instances divided in two; while one set steps, the GPU gets the action
    for the other set.  That set waits for actions to be returned and for the opposite set
    to finish execution before starting (as managed by semaphores here)."""

    alternating = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.batch_spec.B % 2 == 0, "Need even number for sampler batch_B."

    def initialize(self, agent, *args, **kwargs):
        if agent.recurrent and not agent.alternating:
            raise TypeError("If agent is recurrent, must be 'alternating' to use here.")
        elif not agent.recurrent:
            agent.alternating = True  # FF agent doesn't need special class, but tell it so.
        examples = super().initialize(agent, *args, **kwargs)
        half_w = self.n_worker // 2  # Half of workers.
        self.half_B = half_B = self.batch_spec.B // 2  # Half of envs.
        self.step_blockers_pair = (self.sync.step_blockers[:half_w], self.sync.step_blockers[half_w:])
        self.act_waiters_pair = (self.sync.act_waiters[:half_w], self.sync.act_waiter[half_w:])
        self.step_buffer_np_pair = (self.step_buffer_np[:half_B], self.step_buffer_np[half_B:])
        self.agent_inputs_pair = (self.agent_inputs[:half_B], self.agent_inputs[half_B:])
        if self.eval_n_envs > 0:
            eval_half_B = self.eval_n_envs // 2
            self.eval_step_buffer_np_pair = (self.eval_step_buffer_np[:eval_half_B],
                self.eval_step_buffer_np[eval_half_B:])
            self.eval_agent_inputs_pair = (self.eval_agent_inputs[:eval_half_B],
                self.eval_agent_inputs[eval_half_B:])
        if "bootstrap_value" in self.samples_np.agent:
            self.bootstrap_value_pair = (self.samples_np.agent[:half_B],
                self.samples_np.agent[half_B:])
        return examples

    def get_n_envs_list(self, affinity):
        assert len(affinity["workers_cpus"]) % 2 == 0, "Need even number of workers."
        B = self.batch_spec.B
        n_worker = len(affinity["workers_cpus"])
        if B < n_worker:
            logger.log(f"WARNING: requested fewer envs ({B}) than available worker "
                f"processes ({n_worker}). Using fewer workers (but maybe better to "
                "increase sampler's `batch_B`.")
            n_worker = B
        self.n_worker = n_worker
        n_envs_list = [B // n_worker] * n_worker
        if not B % n_worker == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                f"batch_B {B} and n_worker {n_worker} "
                "(possible suboptimal speed).")
            for b in range((B % n_worker) // 2):
                n_envs_list[b] += 1
                n_envs_list[b + half_B] += 1  # Paired worker.
        return n_envs_list


class AlternatingSampler(AlternatingActionServer, AlternatingSamplerBase):
    pass


class NoOverlapAlternatingSampler(NoOverlapAlternatingActionServer,
        AlternatingSamplerBase):
    pass
