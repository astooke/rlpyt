
import numpy as np


class SumTree(object):
    """
    Sum tree for matrix of values stored as [T,B], updated in chunks along T
    dimension.  Priorities represented as first T*B leaves of binary tree.
    Turns on/off entries in vicinity of cursor position according to
    "off_backward" (e.g. n_step_return) and "off_forward" (e.g. 1 for
    prev_action or max(1, frames-1) for frame-wise buffer).
    Provides efficient sampling from non-uniform probability masses.
    """

    def __init__(self, size, B, off_backward, off_forward,
            default_value=1):
        self.T = T = (size + B - 1) // B  # Ceiling div.
        self.B = B
        self.off_backward = off_backward
        self.off_forward = off_forward
        self.default_value = default_value
        self.tree_levels = int(np.ceil(np.log2(T * B + 1)) + 1)
        self.tree = np.zeros(2 ** self.tree_levels - 1)
        self.low_idx = 2 ** (self.tree_levels - 1) - 1  # leaf_idx + low_idx -> tree_idx
        self.high_idx = T * B + self.low_idx
        self.reset()

    def advance(self, T):
        t, b, f = self.t, self.off_backward, self.off_forward
        low_on_t = (t - b) % self.T
        low_off_t = (t + T - b) % self.T  # = high_on_t
        max_off_t = (t + T + f) % self.T
        low_on_idx = low_on_t * self.B + self.low_idx
        low_off_idx = low_off_t * self.B + self.low_idx
        high_off_idx = (max_off_t + 1) * self.B + self.low_idx
        if t + T + f < self.T and t - b >= 0:  # No wrap.
            change_idxs = np.arange(low_on_idx, high_off_idx)
            off_diffs = -self.tree[low_off_idx:high_off_idx]
        else:
            change_idxs = np.concatenate([np.arange(low_on_idx, self.high_idx),
                np.arange(self.low_idx, high_off_idx)])
            if t - b < 0 or t + T - b >= self.T:  # Wrap in "on".
                off_diffs = -self.tree[low_off_idx:high_off_idx]
            else:  # Wrap in "off".
                off_diffs = -np.concatenate([self.tree[low_off_idx:self.high_idx],
                    self.tree[self.low_idx:high_off_idx]])
        change_diffs = np.concatenate([self.default_value * np.ones(T * self.B),
            off_diffs])
        self.reconstruct(change_idxs, change_diffs)
        self.t = (t + T) % self.T

    def sample(self, n, unique=False):
        """Get n samples, with (default) or without replacement."""
        if unique:
            tree_idxs = np.unique(self.find(np.random.rand(int(n * 1.05))))
            i = 0
            while len(tree_idxs) < n:
                if i >= 100:
                    raise RuntimeError("After 100 tries, unable to get unique indexes.")
                new_idxs = self.find(np.random.rand(2 * (n - len(tree_idxs))))
                tree_idxs = np.unique(np.concatenate([tree_idxs, new_idxs]))
                i += 1
        else:
            tree_idxs = self.find(np.random.rand(n))
        self.last_tree_idxs = tree_idxs
        self.last_priorities = priorities = self.tree[tree_idxs]
        T_idxs, B_idxs = np.divmod(tree_idxs - self.low_idx, self.B)
        return (T_idxs, B_idxs), priorities

    def update_batch_priorities(self, priorities):
        self.reconstruct(self.last_tree_idxs, priorities - self.last_priorities)

    def reset(self):
        """For wrapped turn-on during first advance, prep negative values."""
        self.tree[:] = 0
        low_t = self.T - self.n_step
        low_idx = low_t * self.B + self.low_idx
        tree_idxs = np.arange(low_idx, self.high_idx)
        diffs = -self.default_value * np.ones(self.n_step * self.B)
        self.reconstruct(tree_idxs, diffs)
        self.t = 0

    def print_tree(self):
        for k in range(self.tree_levels):
            for j in range(2 ** k - 1, 2 ** (k + 1) - 1):
                print(self.tree[j], end=' ')
            print()

    # Helpers.

    def reconstruct(self, tree_idxs, diffs):
        for _ in range(self.tree_levels):
            np.add.at(self.tree, tree_idxs, diffs)
            tree_idxs = (tree_idxs - 1) // 2  # Rise a level.

    def find(self, random_values):
        """Param random_values: numpy array of floats in range [0, 1] """
        random_values *= self.tree[0]
        tree_idxs = np.zeros(len(random_values), dtype=np.int32)
        for _ in range(self.tree_levels - 1):
            tree_idxs = 2 * tree_idxs + 1
            left_values = self.tree[tree_idxs]
            where_right = np.where(random_values > left_values)[0]
            tree_idxs[where_right] += 1
            random_values[where_right] -= left_values[where_right]
        return tree_idxs
