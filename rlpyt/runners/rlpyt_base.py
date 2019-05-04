
import numpy as np

from rlpyt.utils.quick_args import save_args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils import logger
from rlpyt.utils.prog_bar import ProgBarCounter

from rlpyt.runners.base import BaseRunner


class RlPytBase(BaseRunner):

    def __init__(
            self,
            algo,
            policy,
            sampler,
            n_steps,
            seed=None,
            affinities=None,
            use_gpu=True,
            ):
        n_steps = int(n_steps)
        save_args(locals())

    def startup(self):
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        env_spec, batch_spec = self.sampler.initialize(
            seed=self.seed + 1,
            affinities=self.affinities,
            discount=getattr(self.algo, "discount", None),
        )
        self.initialize_policy(env_spec)
        self.algo.initialize(
            policy=self.policy,
            env_spec=env_spec,
            batch_spec=batch_spec,
        )
        self.sampler.initialize_policy(self.policy)
        n_itr = self.get_n_itr(batch_spec)
        self.algo.set_n_itr(n_itr)
        self.initialize_logger()
        return n_itr



