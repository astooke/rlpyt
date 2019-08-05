

from rlpyt.agents.pg.gaussian import (GaussianPgAgent,
    RecurrentGaussianPgAgent, AlternatingRecurrentGaussianPgAgent)
from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
from rlpyt.models.pg.mujoco_lstm_model import MujocoLstmModel


class MujocoMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.shape[0])


class MujocoFfAgent(MujocoMixin, GaussianPgAgent):

    def __init__(self, ModelCls=MujocoFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class MujocoLstmAgent(MujocoMixin, RecurrentGaussianPgAgent):

    def __init__(self, ModelCls=MujocoLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingMujocoLstmAgent(MujocoMixin,
        AlternatingRecurrentGaussianPgAgent):

    def __init__(self, ModelCls=MujocoLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
