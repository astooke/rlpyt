from rlpyt.agents.dqn.r2d1_agent import R2d1Agent, R2d1AlternatingAgent
from rlpyt.models.dqn.atari_r2d1_model import AtariR2d1Model
from rlpyt.agents.dqn.atari.mixin import AtariMixin


class AtariR2d1Agent(AtariMixin, R2d1Agent):

    def __init__(self, ModelCls=AtariR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AtariR2d1AlternatingAgent(AtariMixin, R2d1AlternatingAgent):

    def __init__(self, ModelCls=AtariR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
    
