
import multiprocessing as mp

from rlpyt.utils.seed import make_seed
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.utils.synchronize import drain_queue


class AsyncSamplerMixin:
    """
    Mixin class defining master runner initialization method for all
    asynchronous samplers.
    """

    ###########################################################################
    # Master runner methods.
    ###########################################################################

    def async_initialize(self, agent, bootstrap_value=False,
            traj_info_kwargs=None, seed=None):
        """Instantiate an example environment and use it to initialize the
        agent (on shared memory).  Pre-allocate a double-buffer for sample
        batches, and return that buffer along with example data (e.g.
        `observation`, `action`, etc.)
        """
        self.seed = make_seed() if seed is None else seed
        # Construct an example of each kind of data that needs to be stored.
        env = self.EnvCls(**self.env_kwargs)
        # Sampler always receives new params through shared memory:
        agent.initialize(env.spaces, share_memory=True,
            global_B=self.batch_spec.B, env_ranks=list(range(self.batch_spec.B)))
        _, samples_np, examples = build_samples_buffer(agent, env,
            self.batch_spec, bootstrap_value, agent_shared=True, env_shared=True,
            subprocess=True)  # Would like subprocess=True, but might hang?
        _, samples_np2, _ = build_samples_buffer(agent, env, self.batch_spec,
            bootstrap_value, agent_shared=True, env_shared=True, subprocess=True)
        env.close()
        del env
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)
        self.double_buffer = double_buffer = (samples_np, samples_np2)
        self.samples_np = samples_np  # In case leftover use during worker init.
        self.examples = examples
        self.agent = agent
        return double_buffer, examples


class AsyncParallelSamplerMixin(AsyncSamplerMixin):
    """
    Mixin class defining methods for the asynchronous sampler main process
    (which is forked from the overall master process).
    """

    ###########################################################################
    # Sampler runner methods (forked once).
    ###########################################################################

    def obtain_samples(self, itr, db_idx):
        """Communicates to workers which batch buffer to use, and signals them
        to start collection.  Waits until workers finish, and then retrieves
        completed trajectory-info objects from the workers and returns them in
        a list.
        """
        self.ctrl.itr.value = itr
        self.sync.db_idx.value = db_idx  # Double buffer index.
        self.ctrl.barrier_in.wait()
        # Workers step environments and sample actions here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return traj_infos

    def _agent_init(self, *args, **kwargs):
        pass  # Already init'ed in master process.

    def _build_buffers(self, *args, **kwargs):
        pass  # Already built in master process.

    def _build_parallel_ctrl(self, *args, **kwargs):
        super()._build_parallel_ctrl(*args, **kwargs)
        self.ctrl.db_idx = mp.RawValue("i", 0)
        self.sync.db_idx = self.ctrl.db_idx  # Pass along to collectors.
        # CPU maybe only needs it in sync?

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = super()._assemble_workers_kwargs(affinity, seed,
            n_envs_list)
        i_env = 0  # OK for GPU, because will already hold slice from global.
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["samples_np"] = tuple(buf[:, slice_B]
                for buf in self.double_buffer)  # Replace.
            i_env += n_envs
        return workers_kwargs
