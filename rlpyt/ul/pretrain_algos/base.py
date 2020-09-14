

class UlAlgorithm:

    opt_info_fields = ()

    def initialize(self):
        raise NotImplementedError

    def load_replay(self):
        raise NotImplementedError

    def optimize(self, itr):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def eval(self):
        """Call this on NN modules."""
        raise NotImplementedError

    def train(self):
        """Call this on NN modules."""
        raise NotImplementedError