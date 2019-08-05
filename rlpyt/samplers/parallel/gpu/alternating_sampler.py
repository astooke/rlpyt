
from rlpyt.samplers.parallel.gpu.sampler import GpuSamplerBase
from rlpyt.samplers.parallel.gpu.action_server import (AlternatingActionServer,
    NoOverlapAlternatingActionServer)


class AlternatingSamplerBase(GpuSamplerBase):
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
        self._make_alternating_pairs()
        return examples

    def _make_alternating_pairs(self):
        half_w = self.n_worker // 2  # Half of workers.
        self.half_B = half_B = self.batch_spec.B // 2  # Half of envs.
        self.obs_ready_pair = (self.sync.obs_ready[:half_w], self.sync.obs_ready[half_w:])
        self.act_ready_pair = (self.sync.act_ready[:half_w], self.sync.act_ready[half_w:])
        self.step_buffer_np_pair = (self.step_buffer_np[:half_B], self.step_buffer_np[half_B:])
        self.agent_inputs_pair = (self.agent_inputs[:half_B], self.agent_inputs[half_B:])
        if self.eval_n_envs > 0:
            assert self.eval_n_envs % 2 == 0
            eval_half_B = self.eval_n_envs // 2
            self.eval_step_buffer_np_pair = (self.eval_step_buffer_np[:eval_half_B],
                self.eval_step_buffer_np[eval_half_B:])
            self.eval_agent_inputs_pair = (self.eval_agent_inputs[:eval_half_B],
                self.eval_agent_inputs[eval_half_B:])
        if "bootstrap_value" in self.samples_np.agent:
            self.bootstrap_value_pair = (self.samples_np.agent.bootstrap_value[:half_B],
                self.samples_np.agent.bootstrap_value[half_B:])

    def _get_n_envs_list(self, affinity=None, n_worker=None, B=None):
        if affinity is not None:
            assert affinity.get("alternating", False), "Need alternating affinity."
        n_worker = len(affinity["workers_cpus"]) if n_worker is None else n_worker
        assert n_worker % 2 == 0, "Need even number workers."
        B = self.batch_spec.B if B is None else B
        assert B % 2 == 0
        # To log warnings:
        n_envs_list = super()._get_n_envs_list(n_worker=n_worker, B=B)
        if B % n_worker > 0:
            # Redistribute extra envs.
            n_envs_list = [B // n_worker] * n_worker
            for w in range((B % n_worker) // 2):
                n_envs_list[w] += 1
                n_envs_list[w + n_worker // 2] += 1  # Paired worker.
        return n_envs_list


class AlternatingSampler(AlternatingActionServer, AlternatingSamplerBase):
    pass  # These use the same Gpu collectors.


class NoOverlapAlternatingSampler(NoOverlapAlternatingActionServer,
        AlternatingSamplerBase):
    pass
