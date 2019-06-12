
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel
from rlpyt.models.dqn.atari_r2d1_model import AtariR2d1Model


class AtariMixin(object):

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)


class AtariDqnAgent(AtariMixin, DqnAgent):

    def __init__(self, ModelCls=AtariDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AtariCatDqnAgent(AtariMixin, CatDqnAgent):

    def __init__(self, ModelCls=AtariCatDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AtariR2d1Agent(AtariMixin, R2d1Agent):

    def __init__(self, ModelCls=AtariR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
