
from rlpyt.utils.quick_args import save_args


class BasePolicy(object):

    def __init__(self, NetworkCls, network_kwargs, initial_state_dict=None):
        save_args(locals(), underscore=True)

    def initialize(self, env_spec):
        self._env_spec = env_spec
        self.network = self._NetworkCls(env_spec, **self._network_kwargs)
        if self._initial_state_dict is not None:
            self.load_state_dict(self._initial_state_dict)

    def get_actions(self, observations, prev_actions, prev_rewards):
        raise NotImplementedError

    def reset(self):
        pass

    def reset_one(self, idx):
        pass

    @property
    def recurrent(self):
        return False

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)

    def state_dict(self):
        return self.network.state_dict()

    def share_memory(self):
        """Call after initialize, by sampler master."""
        if self._device is not None and self._device.type == "cuda":
            self.network_shared = self._NetworkCls(self._env_spec,
                **self._network_kwargs).share_memory()
        else:
            self.network_shared = self.network.share_memory()  # same object
        self.sync_shared_memory()

    def sync_shared_memory(self):
        """Call in sampler master, must call share_memory() before fork."""
        if self.network_shared is not self.network:
            self.network_shared.load_state_dict(self.network.state_dict())

    def swap_worker_network(self):
        """Call in sampler worker, to make shared network the main one."""
        self.network = self.network_shared  # Drops any GPU network.


class BaseRecurrentPolicy(BasePolicy):

    def initialize(self, env_spec):
        super().initialize(env_spec)
        self._prev_rnn_state = None
        self._prev_action = None

    @property
    def recurrent(self):
        return True

    def reset(self):
        self._prev_rnn_state = None  # Gets passed as None; module makes zeros.

    def reset_one(self, idx):
        self._reset_one(idx, self._prev_rnn_state)

    def _reset_one(self, idx, prev_rnn_state):
        # Assume each state is of shape: [B, H], but can be nested list/tuple.
        if isinstance(prev_rnn_state, (list, tuple)):
            for prev_state in prev_rnn_state:
                self._reset_one(prev_rnn_state)
        elif prev_rnn_state is not None:
            prev_rnn_state[idx] = 0.

    def advance_rnn_state(self, new_rnn_state):
        self._prev_rnn_state = new_rnn_state

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state

    def advance_action(self, new_action):
        self._prev_action = new_action

    @property
    def prev_action(self):
        return self._prev_action
