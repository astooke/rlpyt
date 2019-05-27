
from rlpyt.agents.base import BaseAgent
from rlpyt.distributions import EpsilonGreedy, CategoricalEpsilonGreedy


class EpsilonGreedyAgent(BaseAgent):

    def initialize(self, env_spec, share_memory=False):
        self.distribution = EpsilonGreedy()

    def set_epsilon_greedy(self, epsilon):
        self.sample_epsilon = epsilon
        self.distribution.set_epsilon(epsilon)

    def give_eval_epsilon_greedy(self, epsilon):
        self.eval_epsilon = epsilon

    def train_mode(self):
        self.model.train()

    def sample_mode(self):
        self.model.eval()
        self.distribution.set_epsilon(self.sample_epsilon)

    def eval_mode(self):
        self.model.eval()
        self.distribution.set_epsilon(self.eval_epsilon)


class CategoricalEpsilonGreedyAgent(EpsilonGreedyAgent):

    def intitialize(self, env_spec, share_memory=False):
        self.distribution = CategoricalEpsilonGreedy()
