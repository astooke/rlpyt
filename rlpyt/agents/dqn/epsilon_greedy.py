
# import multiprocessing as mp
# import numpy as np
# import ctypes
import torch

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.buffer import np_mp_array


class EpsilonGreedyAgentMixin:
    """
    Mixin class to operate all epsilon-greedy agents.  Includes epsilon
    annealing, switching between sampling and evaluation epsilons, and
    vector-valued epsilons.  The agent subclass must use a compatible
    epsilon-greedy distribution.
    """

    def __init__(
            self,
            eps_init=1,
            eps_final=0.01,
            eps_final_min=None,  # Give < eps_final for vector epsilon.
            eps_itr_min=50,  # Algo may overwrite.
            eps_itr_max=1000,
            eps_eval=0.001,
            *args,
            **kwargs
            ):
        """Saves input arguments.  ``eps_final_min`` other than ``None`` will use 
        vector-valued epsilon, log-spaced."""
        super().__init__(*args, **kwargs)
        save__init__args(locals())
        self._eps_final_scalar = eps_final  # In case multiple vec_eps calls.
        self._eps_init_scalar = eps_init
        self._eps_itr_min_max = np_mp_array(2, "int")  # Shared memory for CpuSampler
        self._eps_itr_min_max[0] = eps_itr_min
        self._eps_itr_min_max[1] = eps_itr_max

    def collector_initialize(self, global_B=1, env_ranks=None):
        """For vector-valued epsilon, the agent inside the sampler worker process
        must initialize with its own epsilon values."""
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def make_vec_eps(self, global_B, env_ranks):
        """Construct log-spaced epsilon values and select local assignments
        from the global number of sampler environment instances (for SyncRl
        and AsyncRl)."""
        if (self.eps_final_min is not None and
                self.eps_final_min != self._eps_final_scalar):  # vector epsilon.
            if self.alternating:  # In FF case, sampler sets agent.alternating.
                assert global_B % 2 == 0
                global_B = global_B // 2  # Env pairs will share epsilon.
                env_ranks = list(set([i // 2 for i in env_ranks]))
            self.eps_init = self._eps_init_scalar * torch.ones(len(env_ranks))
            global_eps_final = torch.logspace(
                torch.log10(torch.tensor(self.eps_final_min)),
                torch.log10(torch.tensor(self._eps_final_scalar)),
                global_B)
            self.eps_final = global_eps_final[env_ranks]
        self.eps_sample = self.eps_init

    def set_epsilon_itr_min_max(self, eps_itr_min, eps_itr_max):
        # Beginning and end of linear ramp down of epsilon.
        logger.log(f"Agent setting min/max epsilon itrs: {eps_itr_min}, "
            f"{eps_itr_max}")
        self.eps_itr_min = eps_itr_min
        self.eps_itr_max = eps_itr_max
        self._eps_itr_min_max[0] = eps_itr_min  # Shared memory for CpuSampler
        self._eps_itr_min_max[1] = eps_itr_max

    # def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
    #     if share_memory:
    #         self.eval_epsilon = mp.RawValue(ctypes.c_float, 1)
    #         self.sample_epsilon = np_mp_array(batch_B, np.float32)
    #     else:
    #         self.eval_epsilon = 1
    #         self.sample_epsilon = 1

    def set_sample_epsilon_greedy(self, epsilon):
        self.distribution.set_epsilon(epsilon)

    # def set_sample_epsilon_greedy(self, epsilon):
    #     if self.share_memory:
    #         self.sample_epsilon[:] = epsilon
    #         if hasattr(self, "env_ranks"):
    #             epsilon = epsilon[self.env_ranks]
    #     else:
    #         self.sample_epsilon = epsilon
    #     self.distribution.set_epsilon(epsilon)

    # def give_eval_epsilon_greedy(self, epsilon):
    #     if self.share_memory:
    #         self.eval_epsilon.value = epsilon
    #     else:
    #         self.eval_epsilon = epsilon

    def sample_mode(self, itr):
        """Extend method to set epsilon for sampling (including annealing)."""
        super().sample_mode(itr)
        itr_min = self._eps_itr_min_max[0]  # Shared memory for CpuSampler
        itr_max = self._eps_itr_min_max[1]
        if itr <= itr_max:
            prog = min(1, max(0, itr - itr_min) / (itr_max - itr_min))
            self.eps_sample = prog * self.eps_final + (1 - prog) * self.eps_init
            if itr % (itr_max // 10) == 0 or itr == itr_max:
                logger.log(f"Agent at itr {itr}, sample eps {self.eps_sample}"
                    f" (min itr: {itr_min}, max_itr: {itr_max})")
        self.distribution.set_epsilon(self.eps_sample)

    # def sample_mode(self, itr):
    #     super().sample_mode(itr)
    #     sample_epsilon = self.sample_epsilon
    #     if self.share_memory and hasattr(self, "env_ranks"):
    #         sample_epsilon = sample_epsilon[self.env_ranks]
    #     self.distribution.set_epsilon(sample_epsilon)

    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        super().eval_mode(itr)
        logger.log(f"Agent at itr {itr}, eval eps "
            f"{self.eps_eval if itr > 0 else 1.}")
        self.distribution.set_epsilon(self.eps_eval if itr > 0 else 1.)

    # def eval_mode(self, itr):
    #     super().eval_mode(itr)
    #     eval_epsilon = self.eval_epsilon.value if self.share_memory else self.eval_epsilon
    #     self.distribution.set_epsilon(eval_epsilon if itr > 0 else 1.)
