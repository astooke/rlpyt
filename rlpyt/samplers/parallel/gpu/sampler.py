
import multiprocessing as mp

from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.gpu.action_server import ActionServer
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector,
    GpuEvalCollector)
from rlpyt.utils.collections import namedarraytuple, AttrDict
from rlpyt.utils.synchronize import drain_queue
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer

StepBuffer = namedarraytuple("StepBuffer",
    ["observation", "action", "reward", "done", "agent_info"])


class GpuSamplerBase(ParallelSamplerBase):

    gpu = True

    def __init__(self, *args, CollectorCls=GpuResetCollector,
            eval_CollectorCls=GpuEvalCollector, **kwargs):
        # e.g. or use GpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        # self.samples_np[:] = 0  # Reset all batch sample values (optional).
        self.agent.sample_mode(itr)
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.agent.eval_mode(itr)
        self.ctrl.barrier_in.wait()
        traj_infos = self.serve_actions_evaluation(itr)
        self.ctrl.barrier_out.wait()
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
            n_sentinel=self.n_worker))  # Block until all finish submitting.
        self.ctrl.do_eval.value = False
        return traj_infos

    def _agent_init(self, agent, env, global_B=1, env_ranks=None):
        agent.initialize(env.spaces, share_memory=False,  # No share memory.
            global_B=global_B, env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, *args, **kwargs):
        examples = super()._build_buffers(*args, **kwargs)
        self.step_buffer_pyt, self.step_buffer_np = build_step_buffer(
            examples, self.batch_spec.B)
        self.agent_inputs = AgentInputs(self.step_buffer_pyt.observation,
            self.step_buffer_pyt.action, self.step_buffer_pyt.reward)
        if self.eval_n_envs > 0:
            self.eval_step_buffer_pyt, self.eval_step_buffer_np = \
                build_step_buffer(examples, self.eval_n_envs)
            self.eval_agent_inputs = AgentInputs(
                self.eval_step_buffer_pyt.observation,
                self.eval_step_buffer_pyt.action,
                self.eval_step_buffer_pyt.reward,
            )
        return examples

    def _build_parallel_ctrl(self, n_worker):
        super()._build_parallel_ctrl(n_worker)
        self.sync.obs_ready = [mp.Semaphore(0) for _ in range(n_worker)]
        self.sync.act_ready = [mp.Semaphore(0) for _ in range(n_worker)]

    def _assemble_common_kwargs(self, *args, **kwargs):
        common_kwargs = super()._assemble_common_kwargs(*args, **kwargs)
        common_kwargs["agent"] = None  # Remove.
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = super()._assemble_workers_kwargs(affinity, seed,
            n_envs_list)
        i_env = 0
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["sync"] = AttrDict(
                stop_eval=self.sync.stop_eval,
                obs_ready=self.sync.obs_ready[rank],
                act_ready=self.sync.act_ready[rank],
            )
            w_kwargs["step_buffer_np"] = self.step_buffer_np[slice_B]
            if self.eval_n_envs > 0:
                eval_slice_B = slice(self.eval_n_envs_per * rank,
                    self.eval_n_envs_per * (rank + 1))
                w_kwargs["eval_step_buffer_np"] = \
                    self.eval_step_buffer_np[eval_slice_B]
            i_env += n_envs
        return workers_kwargs


class GpuSampler(ActionServer, GpuSamplerBase):
    pass


def build_step_buffer(examples, B):
    step_bufs = {k: buffer_from_example(examples[k], B, share_memory=True)
        for k in ["observation", "action", "reward", "done", "agent_info"]}
    step_buffer_np = StepBuffer(**step_bufs)
    step_buffer_pyt = torchify_buffer(step_buffer_np)
    return step_buffer_pyt, step_buffer_np
