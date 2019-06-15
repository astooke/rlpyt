

from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent)
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.models.pg.atari_lstm_model import AtariLstmModel


class AtariMixin(object):

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_size=env_spec.action_space.n)


class AtariFfAgent(AtariMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=AtariFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AtariLstmAgent(AtariMixin, RecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=AtariLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
