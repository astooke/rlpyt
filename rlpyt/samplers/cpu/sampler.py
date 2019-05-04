
from rlpyt.utils.quick_Args import save_args
from rlpyt.samplers.base import BaseSampler

from collections import namedtuple

Samples = namedtuple("samples", ["agent", "env"])



class CpuSampler(BaseSampler):

    def __init__(
            self,
            EnvCls,
            env_kwargs,
            batch_spec,
            max_path_length=int(1e6),
            max_decorrelation_steps=100,
            ):
        save_args(locals())

    def initialize(self, agent, affinities, seed, discount=1):
        n_parallel = len(affinities.cores)
        assert self.batch_spec.B % n_parallel == 0  # Same num envs per worker.
        example_env = self.EnvCls(**self.env_args)
        agent.initialize(example_env.spec)
        agent.share_memory()  # TODO: maybe combine into agent?
        agent_buf, env_buf = build_buffers(agent, example_env, self.batch_spec)
        del example_env
        par_objs = build_par_objs(n_parallel)
        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            agent=agent,
            discount=discount,
            n_envs=self.batch_spec.B // n_parallel,
            include_bootstrap=self.batch_spec.include_bootstrap,
            )
        workers_kwargs = assemble_workers_kwargs(n_parallel, agent_buf, env_buf,
            par_objs, common_kwargs, affinities, seed)

        workers = [mp.Process(target=sampling_process, kwargs=w_kwargs)
            for w_kwargs in workers_kwargs]
        for w in workers:
            w.start()

        self.ctrl, self.traj_infos_queue = par_objs
        self.agent_buf = agent_buf
        self.env_buf = env_buf
        self.agent = agent

        self.ctrl.barrier.wait()  # Wait for workers to decorrelate envs.

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # Appears in workers.
        self.ctrl.barrier_in.wait()
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.apppend(self.traj_infos_queue.get())
        return Samples(self.agent_buf, self.env_buf), traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()


def build_buffers(agent, env, batch_spec):
    o = env.reset()
    a = env.action_space.sample()
    r = np.array(0., dtype=np.float32).reshape(1, 1)  # [B=1, d=1] but no T.
    a, agent_info = agent.get_actions(o[None], a[None], r)
    o, r, d, env_info = env.step(a)
    examples = dict(
        observations=o,
        rewards=r,
        dones=d,
        env_infos=dict(**env_info),
        actions=a)


