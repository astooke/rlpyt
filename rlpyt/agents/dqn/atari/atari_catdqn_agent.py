from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin


class AtariCatDqnAgent(AtariMixin, CatDqnAgent):

    def __init__(self, ModelCls=AtariCatDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
