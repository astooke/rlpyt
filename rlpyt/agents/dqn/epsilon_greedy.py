
# import multiprocessing as mp
# import numpy as np
# import ctypes
import torch

from rlpyt.utils.quick_args import save__init__args
# from rlpyt.utils.buffer import np_mp_array


class EpsilonGreedyAgentMixin(object):

    def __init__(
            self,
            eps_init=1,
            eps_final=0.01,
            eps_final_min=None,
            eps_itr_min=50,
            eps_itr_max=1000,
            eps_eval=0.001,
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        save__init__args(locals())

    def initialize(self, env_space, share_memory=False, global_B=1,
            env_ranks=None):
        self._make_eps_init_final(global_B, env_ranks)

    def collector_initialize(self, global_B=1, env_ranks=None):
        self._make_eps_init_final(global_B, env_ranks)

    def _make_eps_init_final(self, global_B, env_ranks):
        if env_ranks is None:
            print("OOPS HAD NONE ENV_RANKS")
            return
        if (self.eps_final_min is not None and
                self.eps_final_min != self.eps_final):  # vector epsilon.
            self.eps_init = self.eps_init * torch.ones(len(env_ranks))
            global_eps_final = torch.logspace(
                torch.log10(torch.tensor(self.eps_final_min)),
                torch.log10(torch.tensor(self.eps_final)),
                global_B)
            self.eps_final = global_eps_final[env_ranks]
        self.eps_sample = self.eps_init

    def set_epsilon_itr_min_max(self, eps_itr_min, eps_itr_max):
        # Beginning and end of linear ramp down of epsilon.
        self.eps_itr_min = eps_itr_min
        self.eps_itr_max = eps_itr_max

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
        super().sample_mode(itr)
        if itr <= self.eps_itr:
            prog = min(1, max(0, itr - self.eps_itr_min) / self.eps_itr_max)
            self.eps_sample = prog * self.eps_final + (1 - prog) * self.eps_init
        self.distribution.set_epsilon(self.eps_sample)


    # def sample_mode(self, itr):
    #     super().sample_mode(itr)
    #     sample_epsilon = self.sample_epsilon
    #     if self.share_memory and hasattr(self, "env_ranks"):
    #         sample_epsilon = sample_epsilon[self.env_ranks]
    #     self.distribution.set_epsilon(sample_epsilon)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.distribution.set_epsilon(self.eps_eval if itr > 0 else 1.)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        eval_epsilon = self.eval_epsilon.value if self.share_memory else self.eval_epsilon
        self.distribution.set_epsilon(eval_epsilon if itr > 0 else 1.)
