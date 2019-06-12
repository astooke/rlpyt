

class EpsilonGreedyAgentMixin(object):

    def set_epsilon_greedy(self, epsilon):
        self.sample_epsilon = epsilon
        self.distribution.set_epsilon(epsilon)

    def give_eval_epsilon_greedy(self, epsilon):
        self.eval_epsilon = epsilon

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.distribution.set_epsilon(self.sample_epsilon)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.distribution.set_epsilon(self.eval_epsilon)
