
import numpy as np
import multiprocessing as mp

from rlpyt.utils.buffer import np_mp_array


class SumTree:
    """
    Sum tree for matrix of values stored as [T,B], updated in chunks along T
    dimension, applying to the full B dimension at each update.  Priorities
    represented as first T*B leaves of binary tree. Turns on/off entries in
    vicinity of cursor position according to "off_backward" (e.g.
    n_step_return) and "off_forward" (e.g. 1 for
    prev_action or max(1, frames-1) for frame-wise buffer).
    Provides efficient sampling from non-uniform probability masses.

    NOTE: 
        Tried single precision (float32) tree, and it sometimes returned
        samples with priority 0.0, because subtraction during tree cascade
        left random value larger than the remaining sum; suggest keeping
        float64.
    """

    async_ = False

    def __init__(self, T, B, off_backward, off_forward,
            default_value=1,
            enable_input_priorities=False,
            input_priority_shift=0,  # Does not apply to update_batch_pri.
            ):
        self.T = T
        self.B = B
        self.size = T * B
        self.off_backward = off_backward
        self.off_forward = off_forward
        self.default_value = default_value
        self.input_priority_shift = input_priority_shift  # (See self.sample()).
        self.tree_levels = int(np.ceil(np.log2(self.size + 1)) + 1)
        self._allocate_tree()
        self.low_idx = 2 ** (self.tree_levels - 1) - 1  # pri_idx + low_idx -> tree_idx
        self.high_idx = self.size + self.low_idx
        self.priorities = self.tree[self.low_idx:self.high_idx].reshape(T, B)
        if enable_input_priorities:
            self.input_priorities = default_value * np.ones((T, B))
        else:
            self.input_priorities = None  # Save memory.
        self.reset()

    def _allocate_tree(self):
        self.tree = np.zeros(2 ** self.tree_levels - 1)  # Double precision.

    def reset(self):
        self.tree.fill(0)
        self.t = 0
        self._initial_wrap_guard = True
        if self.input_priorities is not None:
            self.input_priorities[:] = self.default_value

    def advance(self, T, priorities=None):
        """Cursor advances by T: set priorities to zero in vicinity of new
        cursor position and turn priorities on for new samples since previous
        cursor position.
        Optional param ``priorities`` can be None for default, or of
        dimensions [T, B], or [B] or scalar will broadcast. (Must have enabled
        ``input_priorities=True`` when instantiating the tree.)  These will be
        stored at the current cursor position, meaning these priorities
        correspond to the current values being added to the buffer, even
        though their priority might temporarily be set to zero until future
        advances.
        """
        if T == 0:
            return
        t, b, f = self.t, self.off_backward, self.off_forward
        low_on_t = (t - b) % self.T  # inclusive range: [0, self.T-1]
        high_on_t = ((t + T - b - 1) % self.T) + 1  # inclusive: [1, self.T]
        low_off_t = (t + T - b) % self.T
        high_off_t = ((t + T + f - 1) % self.T) + 1
        if self._initial_wrap_guard:
            low_on_t = max(f, t - b)  # Don't wrap back to end, and off_forward.
            high_on_t = low_off_t = max(low_on_t, t + T - b)
            if t + T - b >= f:  # Next low_on_t >= f.
                self._initial_wrap_guard = False
        if priorities is not None:
            assert self.input_priorities is not None, "Must enable input priorities."
            # e.g. Use input_priority_shift = warmup_T // rnn_state_interval
            # to make the fresh priority at t be the one input with the later
            # samples at t + shift, which would be the start of training
            # (priorities are aligned with start of warmup sequence).
            input_t = t - self.input_priority_shift
            if input_t < 0 or input_t + T > self.T:  # Wrap (even at very first).
                idxs = np.arange(input_t, input_t + T) % self.T
            else:
                idxs = slice(input_t, input_t + T)
            self.input_priorities[idxs] = priorities
            if self._initial_wrap_guard and input_t < 0:
                self.input_priorities[input_t:] = self.default_value  # Restore.
        self.reconstruct_advance(low_on_t, high_on_t, low_off_t, high_off_t)
        self.t = (t + T) % self.T

    def sample(self, n, unique=False):
        """Get `n` samples, with replacement (default) or without.  Use 
        ``np.random.rand()`` to generate random values with which to descend
        the tree to each sampled leaf node. Returns `T_idxs` and `B_idxs`, and sample
        priorities."""
        self._sampled_unique = unique
        random_values = np.random.rand(int(n*1 if unique else n))
        tree_idxes, scaled_random_values = self.find(random_values)
        if unique:
            i = 0
            while i < 100:
                tree_idxes, unique_idx = np.unique(tree_idxes, return_index=True)
                scaled_random_values = scaled_random_values[unique_idx]
                if len(tree_idxes) < n:
                    new_idxes, new_values = self.find(np.random.rand(2 * (n - len(tree_idxes))))
                    tree_idxes = np.concatenate([tree_idxes, new_idxes])
                    scaled_random_values = np.concatenate([scaled_random_values, new_values])
                else:
                    break
                i += 1
            if len(tree_idxes) < n:
                raise RuntimeError("After 100 tries, unable to get unique indexes.")
            tree_idxes = tree_idxes[:n]

        priorities = self.tree[tree_idxes]
        self.prev_tree_idxs = tree_idxes
        T_idxs, B_idxs = np.divmod(tree_idxes - self.low_idx, self.B)
        return (T_idxs, B_idxs), priorities

    def update_batch_priorities(self, priorities):
        """Apply new priorities to tree at the leaf positions where the last
        batch was returned from the ``sample()`` method.
        """
        if not self._sampled_unique:  # Must remove duplicates
            self.prev_tree_idxs, unique_idxs = np.unique(self.prev_tree_idxs,
                return_index=True)
            priorities = priorities[unique_idxs]
        self.reconstruct(self.prev_tree_idxs, priorities)

    def print_tree(self, level=None):
        """Print values for whole tree or at specified level."""
        levels = range(self.tree_levels) if level is None else [level]
        for k in levels:
            for j in range(2 ** k - 1, 2 ** (k + 1) - 1):
                print(self.tree[j], end=' ')
            print()

    # Helpers.

    def reconstruct(self, tree_idxs, values):
        diffs = values - self.tree[tree_idxs]  # Numpy upcasts to float64.
        self.tree[tree_idxs] = values
        self.propagate_diffs(tree_idxs, diffs, min_level=1)

    def reconstruct_advance(self, low_on_t, high_on_t, low_off_t, high_off_t):
        """Efficiently write new values / zeros into tree."""
        low_on_idx = low_on_t * self.B + self.low_idx
        high_on_idx = high_on_t * self.B + self.low_idx
        low_off_idx = low_off_t * self.B + self.low_idx
        high_off_idx = high_off_t * self.B + self.low_idx
        idxs, diffs = list(), list()
        if high_on_t > low_on_t:
            if self.input_priorities is None:
                input_priorities = self.default_value
            else:
                input_priorities = self.input_priorities[low_on_t:high_on_t]
            diffs.append(input_priorities - self.priorities[low_on_t:high_on_t])
            self.priorities[low_on_t:high_on_t] = input_priorities
            idxs.append(np.arange(low_on_idx, high_on_idx))
        elif high_on_t < low_on_t:  # Wrap
            if self.input_priorities is None:
                diffs.append(self.default_value - np.concatenate([
                    self.priorities[low_on_t:], self.priorities[:high_on_t]],
                    axis=0))
                self.priorities[low_on_t:] = self.default_value
                self.priorities[:high_on_t] = self.default_value
            else:
                diffs.append(
                    np.concatenate(
                        [self.input_priorities[low_on_t:],
                        self.input_priorities[:high_on_t]], axis=0) -
                    np.concatenate(
                        [self.priorities[low_on_t:],
                        self.priorities[:high_on_t]], axis=0)
                )
                self.priorities[low_on_t:] = self.input_priorities[low_on_t:]
                self.priorities[:high_on_t] = self.input_priorities[:high_on_t]
            idxs.extend([np.arange(low_on_idx, self.high_idx),
                np.arange(self.low_idx, high_on_idx)])
        if high_off_t > low_off_t:
            diffs.append(-self.priorities[low_off_t:high_off_t])
            self.priorities[low_off_t:high_off_t] = 0
            idxs.append(np.arange(low_off_idx, high_off_idx))
        else:  # Wrap.
            diffs.extend([-self.priorities[low_off_t:],
                -self.priorities[:high_off_t]])
            self.priorities[low_off_t:] = 0
            self.priorities[:high_off_t] = 0
            idxs.extend([np.arange(low_off_idx, self.high_idx),
                np.arange(self.low_idx, high_off_idx)])
        if diffs:
            diffs = np.concatenate(diffs).reshape(-1)
            idxs = np.concatenate(idxs)
            self.propagate_diffs(idxs, diffs, min_level=1)

    def propagate_diffs(self, tree_idxs, diffs, min_level=1):
        for _ in range(min_level, self.tree_levels):
            tree_idxs = (tree_idxs - 1) // 2  # Rise a level
            np.add.at(self.tree, tree_idxs, diffs)

    def find(self, random_values):
        """Param random_values: numpy array of floats in range [0, 1] """
        random_values = self.tree[0] * random_values  # Double precision.
        scaled_random_values = random_values.copy()
        tree_idxs = np.zeros(len(random_values), dtype=np.int64)
        for _ in range(self.tree_levels - 1):
            tree_idxs = 2 * tree_idxs + 1
            left_values = self.tree[tree_idxs]
            where_right = np.where(random_values > left_values)[0]
            tree_idxs[where_right] += 1
            random_values[where_right] -= left_values[where_right]
        return tree_idxs, scaled_random_values


class AsyncSumTree(SumTree):
    """Allocates the tree into shared memory, and manages asynchronous cursor
    position, for different read and write processes. Assumes that writing to
    tree values is lock protected elsewhere, i.e. by the replay buffer.
    """

    async_ = True

    def __init__(self, *args, **kwargs):
        self.async_t = mp.RawValue("l", 0)
        super().__init__(*args, **kwargs)
        # Wrap guard behavior should be fine without async--each will catch it.

    def _allocate_tree(self):
        self.tree = np_mp_array(2 ** self.tree_levels - 1, np.float64)  # Shared memory.
        self.tree.fill(0)  # Just in case.

    def reset(self):
        super().reset()
        self.async_t.value = 0

    def advance(self, *args, **kwargs):
        self.t = self.async_t.value
        super().advance(*args, **kwargs)
        self.async_t.value = self.t
