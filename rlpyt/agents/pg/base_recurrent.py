
from rlpyt.agents.base import BaseRecurrentAgent
from rlpyt.agents.policy_gradient.base import BasePgAgent
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo",  # Must define uniquely in file.
    ["dist_info", "value", "prev_rnn_state"])


class BaseRecurrentPgAgent(BasePgAgent, BaseRecurrentAgent):
    pass
