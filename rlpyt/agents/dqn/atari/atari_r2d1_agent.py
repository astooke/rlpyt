

from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.atari_r2d1_model import AtariR2d1Model


class AtariR2d1Agent(R2d1Agent):

    def __init__(self, ModelCls=AtariR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)
