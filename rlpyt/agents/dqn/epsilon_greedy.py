

class EpsilonGreedyAgentMixin(object):

    def set_epsilon_greedy(self, epsilon):
        self.sample_epsilon = epsilon
        self.distribution.set_epsilon(epsilon)

    def give_eval_epsilon_greedy(self, epsilon):
        self.eval_epsilon = epsilon

    def sample_mode(self):
        super().sample_mode()
        self.distribution.set_epsilon(self.sample_epsilon)

    def eval_mode(self):
        super().eval_mode()
        self.distribution.set_epsilon(self.eval_epsilon)
