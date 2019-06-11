

from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel


class AtariCatDqnAgent(CatDqnAgent):

    def __init__(self, ModelCls=AtariCatDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)
