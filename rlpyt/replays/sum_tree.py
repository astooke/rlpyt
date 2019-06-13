
import numpy as np


class SumTree(object):
    """
    Sum tree for matrix of values stored as [T,B], updated in chunks along T
    dimension.  Priorities represented as first T*B leaves of binary tree.
    Turns on/off entries in vicinity of cursor position according to
    "off_backward" (e.g. n_step_return) and "off_forward" (e.g. 1 for
    prev_action or max(1, frames-1) for frame-wise buffer).
    Provides efficient sampling from non-uniform probability masses.

    NOTE: Tried single precision (float32) tree, and it sometimes returned
    samples with priority 0.0, because subtraction during tree cascade left
    random value larger than the remaining sum; suggest keeping float64.
    """

    def __init__(self, T, B, off_backward, off_forward,
            default_value=1):
        self.T = T
        self.B = B
        self.size = T * B
        self.off_backward = off_backward
        self.off_forward = off_forward
        self.default_value = default_value
        self.tree_levels = int(np.ceil(np.log2(self.size + 1)) + 1)
        self.tree = np.zeros(2 ** self.tree_levels - 1)  # Double precision.
        self.low_idx = 2 ** (self.tree_levels - 1) - 1  # pri_idx + low_idx -> tree_idx
        self.high_idx = self.size + self.low_idx
        self.priorities = self.tree[self.low_idx:self.high_idx].reshape(T, B)
        self.reset()

    def reset(self):
        self.tree[:] = 0
        self.t = 0
        self._initial_wrap_guard = True

    def advance(self, T, priorities=None):
        """Cursor advances by T; set priorities to zero in vicinity of new
        cursor position and turn priorities on for new samples since previous
        cursor position.  Optional priorities can be None for default,
        scalar, or with dimensions [T] or [T, B]."""
        t, b, f = self.t, self.off_backward, self.off_forward
        low_on_t = (t - b) % self.T
        high_on_t = low_off_t = (t + T - b) % self.T
        high_off_t = (t + T + f) % self.T
        if self._initial_wrap_guard:
            low_on_t = max(f, t - b)  # Don't wrap back to end, and off forward.
            high_on_t = low_off_t = max(low_on_t, t + T - b)
            if priorities is not None:
                if hasattr(priorities, "shape") and priorities.shape[0] == T:
                    priorities = priorities[-(high_on_t - low_on_t):]
            if t + T - b >= f:  # Next low_on_t >= f.
                self._initial_wrap_guard = False
        on_value = self.default_value if priorities is None else priorities
        self.reconstruct_advance(low_on_t, high_on_t, low_off_t, high_off_t,
            on_value)
        self.t = (t + T) % self.T

    def sample(self, n, unique=False):
        """Get n samples, with (default) or without replacement."""
        self._sampled_unique = unique
        if unique:
            tree_idxs = np.unique(self.find(np.random.rand(int(n * 1.05))))
            i = 0
            while len(tree_idxs) < n:
                if i >= 100:
                    raise RuntimeError("After 100 tries, unable to get unique indexes.")
                new_idxs = self.find(np.random.rand(2 * (n - len(tree_idxs))))
                tree_idxs = np.unique(np.concatenate([tree_idxs, new_idxs]))
                i += 1
            tree_idxs = tree_idxs[:n]
        else:
            random_values = np.random.rand(n)
            tree_idxs, scaled_random_values = self.find(random_values)
        priorities = self.tree[tree_idxs]
        self.prev_tree_idxs = tree_idxs
        T_idxs, B_idxs = np.divmod(tree_idxs - self.low_idx, self.B)
        return (T_idxs, B_idxs), priorities

    def update_batch_priorities(self, priorities):
        if not self._sampled_unique:  # Must remove duplicates
            self.prev_tree_idxs, unique_idxs = np.unique(self.prev_tree_idxs,
                return_index=True)
            priorities = priorities[unique_idxs]
        self.reconstruct(self.prev_tree_idxs, priorities)

    def print_tree(self, level=None):
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

    def reconstruct_advance(self, low_on_t, high_on_t, low_off_t, high_off_t,
            on_value):
        """Efficiently write new values / zeros into tree."""
        low_on_idx = low_on_t * self.B + self.low_idx
        low_off_idx = high_on_idx = high_on_t * self.B + self.low_idx
        high_off_idx = high_off_t * self.B + self.low_idx
        if high_on_t >= low_on_t:  # Equal for initial_wrap_guard -> empty arrays.
            on_diffs = on_value - self.priorities[low_on_t:high_on_t]
            self.priorities[low_on_t:high_on_t] = on_value
            on_idxs = np.arange(low_on_idx, high_on_idx)
        else:  # Wrap.
            on_diffs = on_value - np.concatenate([self.priorities[low_on_t:],
                self.priorities[:high_on_t]])
            self.priorities[low_on_t:] += on_diffs[:-high_on_t]
            self.priorities[:high_on_t] += on_diffs[-high_on_t:]
            on_idxs = np.concatenate([np.arange(low_on_idx, self.high_idx),
                np.arange(self.low_idx, high_on_idx)])
        if high_off_t >= low_off_t:  # Equal for no off -> empty arrays.
            off_diffs = -self.priorities[low_off_t:high_off_t]
            self.priorities[low_off_t:high_off_t] = 0
            off_idxs = np.arange(low_off_idx, high_off_idx)
        else:  # Wrap.
            off_diffs = -np.concatenate([self.priorities[low_off_t:],
                self.priorities[:high_on_t]])
            self.priorities[low_off_t:] = 0
            self.priorities[:high_off_t] = 0
            off_idxs = np.concatenate([np.arange(low_off_idx, self.high_idx),
                np.arange(self.low_idx, high_off_idx)])
        diffs = np.concatenate([on_diffs, off_diffs]).reshape(-1)
        tree_idxs = np.concatenate([on_idxs, off_idxs])
        self.propagate_diffs(tree_idxs, diffs, min_level=1)

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
