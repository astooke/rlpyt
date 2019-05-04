

class BaseSampler(object):

    def initialize(self, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError
