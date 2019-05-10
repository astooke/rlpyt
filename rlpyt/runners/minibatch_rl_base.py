
import numpy as np
import psutil

from rlpyt.utils.quick_args import save_args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils import logger
from rlpyt.utils.prog_bar import ProgBarCounter

from rlpyt.runners.base import BaseRunner


class MinibatchRlBase(BaseRunner):

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            seed=None,
            affinities=None,
            cuda=None,
            ):
        n_steps = int(n_steps)
        save_args(locals())

    def startup(self):
        p = psutil.Process()
        p.cpu_affinity(self.affinities.get("master_cpus"), p.cpu_affinity())
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)  # TODO
        self.sampler.initialize(
            agent=self.agent,
            affinities=self.affinities,
            seed=self.seed + 1,
            traj_info_kwargs=self.get_traj_info_kwargs(),
        )
        # Agent initialized in sampler?


        self.algo.initialize(self.agent)
        n_itr = self.get_n_itr(self.sampler.batch_spec)
        self.algo.set_n_itr(n_itr)
        self.initialize_logger()
        return n_itr

    def get_traj_info_kwargs(self):
        return dict(discount=self.algo.get("discount", 1))



