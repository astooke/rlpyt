

from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent)
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.models.pg.atari_lstm_model import AtariLstmModel


class AtariMixin(object):

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class AtariFfAgent(AtariMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=AtariFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AtariLstmAgent(AtariMixin, RecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=AtariLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
