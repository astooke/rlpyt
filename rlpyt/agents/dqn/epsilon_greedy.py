
import multiprocessing as mp
import ctypes


class EpsilonGreedyAgentMixin(object):

    def initialize(self, env_spec, share_memory=False):
        print("IN EPS GREEDY AGNET INITIALIXE, share_memory: ", share_memory)
        if share_memory:
            self.eval_epsilon = mp.RawValue(ctypes.c_float, 0)
            # Does not support vector-valued epsilon.
            print("MADE SAMPLE EPSILON MP RAWVALE")
            self.sample_epsilon = mp.RawValue(ctypes.c_float, 1)

    def set_sample_epsilon_greedy(self, epsilon):
        if self.share_memory:
            self.sample_epsilon.value = epsilon
        else:
            self.sample_epsilon = epsilon
        self.distribution.set_epsilon(epsilon)

    def give_eval_epsilon_greedy(self, epsilon):
        if self.share_memory:
            self.eval_epsilon.value = epsilon
        else:
            self.eval_epsilon = epsilon

    def sample_mode(self, itr):
        super().sample_mode(itr)
        sample_epsilon = self.sample_epsilon.value if self.share_memory else self.sample_epsilon
        self.distribution.set_epsilon(sample_epsilon)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        eval_epsilon = self.eval_epsilon.value if self.share_memory else self.eval_epsilon
        self.distribution.set_epsilon(eval_epsilon if itr > 0 else 1.)
