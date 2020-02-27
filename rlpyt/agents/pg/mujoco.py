

from rlpyt.agents.pg.gaussian import (GaussianPgAgent,
    RecurrentGaussianPgAgent, AlternatingRecurrentGaussianPgAgent)
from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
from rlpyt.models.pg.mujoco_lstm_model import MujocoLstmModel
from rlpyt.utils.buffer import buffer_to


class MujocoMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    Now supports observation normalization, including multi-GPU.
    """
    _ddp = False  # Sets True if data parallel, for normalized obs

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract observation_shape and action_size."""
        assert len(env_spaces.action.shape) == 1
        return dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.shape[0])

    def update_obs_rms(self, observation):
        observation = buffer_to(observation, device=self.device)
        if self._ddp:
            self.model.module.update_obs_rms(observation)
        else:
            self.model.update_obs_rms(observation)

    def data_parallel(self, *args, **kwargs):
        super().data_parallel(*args, **kwargs)
        self._ddp = True


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
