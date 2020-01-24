
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSamplerBase
from rlpyt.samplers.async_.action_server import (AsyncAlternatingActionServer,
    AsyncNoOverlapAlternatingActionServer)
from rlpyt.utils.logging import logger


class AsyncAlternatingSamplerBase(AsyncGpuSamplerBase):
    """Defines several methods to extend the asynchronous GPU sampler to use
    two alternating sets of environment workers.  
    """

    alternating = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.batch_spec.B % 2 == 0, "Need even number for sampler batch_B."

    def async_initialize(self, agent, *args, **kwargs):
        if agent.recurrent and not agent.alternating:
            raise TypeError("If agent is recurrent, must be 'alternating' to use here.")
        elif not agent.recurrent:
            agent.alternating = True  # FF agent doesn't need special class, but tell it so.
        return super().async_initialize(agent, *args, **kwargs)

    def launch_workers(self, double_buffer_slice, affinity, seed, n_envs_list):
        super().launch_workers(double_buffer_slice, affinity, seed, n_envs_list)
        self._make_alternating_pairs(n_envs_list)

    def _make_alternating_pairs(self, n_envs_list):
        assert len(n_envs_list) % 2 == 0
        assert sum(n_envs_list) % 2 == 0
        half_w = len(n_envs_list) // 2  # Half of workers.
        self.half_B = half_B = sum(n_envs_list) // 2  # Half of envs.
        self.obs_ready_pair = (self.sync.obs_ready[:half_w], self.sync.obs_ready[half_w:])
        self.act_ready_pair = (self.sync.act_ready[:half_w], self.sync.act_ready[half_w:])
        self.step_buffer_np_pair = (self.step_buffer_np[:half_B], self.step_buffer_np[half_B:])
        self.agent_inputs_pair = (self.agent_inputs[:half_B], self.agent_inputs[half_B:])
        if self.eval_n_envs > 0:
            assert self.eval_n_envs_per * len(n_envs_list) % 2 == 0
            eval_half_B = self.eval_n_envs_per * len(n_envs_list) // 2
            self.eval_step_buffer_np_pair = (self.eval_step_buffer_np[:eval_half_B],
                self.eval_step_buffer_np[eval_half_B:])
            self.eval_agent_inputs_pair = (self.eval_agent_inputs[:eval_half_B],
                self.eval_agent_inputs[eval_half_B:])
        if "bootstrap_value" in self.samples_np.agent:
            self.double_bootstrap_value_pair = tuple(
                (buf.agent.bootstrap_value[:half_B],
                    buf.agent.bootstrap_value[half_B:])
                for buf in self.double_buffer)

    def _get_n_envs_lists(self, affinity):
        for aff in affinity:
            assert aff.get("alternating", False), "Need alternating affinity."
        B = self.batch_spec.B
        n_server = len(affinity)
        n_workers = [len(aff["workers_cpus"]) for aff in affinity]
        if B < n_server:
            raise ValueError(f"Request fewer envs ({B}) than action servers "
                f"({n_server}).")
        server_Bs = [B // n_server] * n_server
        if n_workers.count(n_workers[0]) != len(n_workers):
            logger.log("WARNING: affinity requested different number of "
                "environment workers per action server, but environments "
                "will be assigned equally across action servers anyway.")
        if B % n_server > 0:
            assert (B % n_server) % 2 == 0, "Need even num extra envs per server."
            for s in range((B % n_server) // 2):
                server_Bs[s] += 2  # Spread across action servers in pairs.

        n_envs_lists = list()
        for s_worker, s_B in zip(n_workers, server_Bs):
            n_envs_lists.append(self._get_n_envs_list(n_worker=s_worker, B=s_B))

        return n_envs_lists

    def _get_n_envs_list(self, affinity=None, n_worker=None, B=None):
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


class AsyncAlternatingSampler(AsyncAlternatingActionServer,
        AsyncAlternatingSamplerBase):
    pass


class AsyncNoOverlapAlternatingSampler(AsyncNoOverlapAlternatingActionServer,
        AsyncAlternatingSamplerBase):
    pass
