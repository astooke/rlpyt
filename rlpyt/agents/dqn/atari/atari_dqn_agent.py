
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin


class AtariDqnAgent(AtariMixin, DqnAgent):

    def __init__(self, ModelCls=AtariDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
