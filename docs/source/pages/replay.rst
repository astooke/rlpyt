
Replay Buffers
==============

Several variants of replay buffers are included in rlpyt.  Options include: n-step returns (computed by the replay buffer), prioritized replay (sum-tree), frame-based observation storage (for memory savings), and replay of sequences.

All buffers are based on pre-allocated a size of memory with leading dimensions [T,B], where B is the expected (and required) corresponding dimension in the input sample batches (which will be the number of parallel environmnets in the sampler), and T is chosen to attain the total requested buffer size.  A universal time cursor tracks the position of latest inputs along the T dimension of the buffer, and it wraps automatically.  Use of namedarraytuples makes it straightforward to write data of arbitrary structure to the buffer's next indexes.  Further benefits are that pre-allocated storage doesn't grow and is more easily shared across processes (async mode).  But this format does require accounting for which samples are currently invalid due to partial memory overwrite, based on n-step returns or needing to replay sequences.  If memory and performance optimization are less of a concern, it might be preferable to write a simpler buffer which, for example, stores a rotating list of complete sequences to replay.

.. hint::
    The implemented replay buffers share a lot of components, and sub-classing with multiple inheritances is used to prevent redundant code.  If modifying a replay buffer, it might be easier to first copy all desired components into one monolithic class, and then work from there.


Replay Buffer Components
------------------------

Base Buffers
^^^^^^^^^^^^

.. autoclass:: rlpyt.replays.base.BaseReplayBuffer
    :members: append_samples, sample_batch

.. autoclass:: rlpyt.replays.n_step.BaseNStepReturnBuffer
    :members: append_samples, compute_returns
    :show-inheritance:

.. autoclass:: rlpyt.replays.frame.FrameBufferMixin
    :members: append_samples

.. autoclass:: rlpyt.replays.async_.AsyncReplayBufferMixin


Non-Sequence Replays
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.replays.non_sequence.n_step.NStepReturnBuffer
    :members: extract_batch, extract_observation
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.frame.NStepFrameBuffer
    :members: extract_observation
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.uniform.UniformReplay
    :members: sample_batch, sample_idxs

.. autoclass:: rlpyt.replays.non_sequence.prioritized.PrioritizedReplay
    :members: append_samples, sample_batch, update_batch_priorities

.. autoclass:: rlpyt.replays.non_sequence.time_limit.NStepTimeLimitBuffer
    :show-inheritance:


Sequence Replays
^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.replays.sequence.n_step.SequenceNStepReturnBuffer
    :members: append_samples, extract_batch
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.frame.SequenceNStepFrameBuffer
    :members: extract_observation
    :show-inheritance:


.. autoclass:: rlpyt.replays.sequence.uniform.UniformSequenceReplay
    :members: sample_batch, sample_idxs, set_batch_T

.. autoclass:: rlpyt.replays.sequence.prioritized.PrioritizedSequenceReplay
    :members: append_samples, sample_batch


Priority Tree
^^^^^^^^^^^^^

.. autoclass:: rlpyt.replays.sum_tree.SumTree
    :members: reset, advance, sample, update_batch_priorities, print_tree

.. autoclass:: rlpyt.replays.sum_tree.AsyncSumTree
    :show-inheritance:


Full Replay Buffer Classes
--------------------------

These are all defined purely as sub-classes with above components.

Non-Sequence Replay
^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.replays.non_sequence.uniform.UniformReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.uniform.AsyncUniformReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.prioritized.PrioritizedReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.prioritized.AsyncPrioritizedReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.frame.UniformReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.frame.PrioritizedReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.frame.AsyncUniformReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.frame.AsyncPrioritizedReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.time_limit.TlUniformReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.time_limit.TlPrioritizedReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.time_limit.AsyncTlUniformReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.non_sequence.time_limit.AsyncTlPrioritizedReplayBuffer
    :show-inheritance:

Sequence Replay
^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.replays.sequence.uniform.UniformSequenceReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.uniform.AsyncUniformSequenceReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.prioritized.PrioritizedSequenceReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.prioritized.AsyncPrioritizedSequenceReplayBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.frame.UniformSequenceReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.frame.PrioritizedSequenceReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.frame.AsyncUniformSequenceReplayFrameBuffer
    :show-inheritance:

.. autoclass:: rlpyt.replays.sequence.frame.AsyncPrioritizedSequenceReplayFrameBuffer
    :show-inheritance:


