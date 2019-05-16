

from collections import namedtuple
from rlpyt.utils.collections import namedarraytuple

OptData = namedarraytuple("OptData", [])
OptInfo = namedtuple("OptInfo", [])


class RlAlgorithm(object):

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr):
        raise NotImplementedError

    def optimize_agent(self, samples, itr):
        raise NotImplementedError
