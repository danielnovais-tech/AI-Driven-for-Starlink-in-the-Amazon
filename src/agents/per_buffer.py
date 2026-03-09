"""
Prioritized Experience Replay (PER) buffer for online DRL fine-tuning.

Implements a sum-tree-based priority buffer where transitions with higher
TD-error are sampled more frequently.  The sampling distribution is:

    P(i) = p_i^alpha / sum_j(p_j^alpha)

where p_i is the priority of transition i and alpha controls the degree of
prioritisation (0 = uniform, 1 = full prioritisation).

Importance-sampling weights correct for the bias introduced by non-uniform
sampling:

    w_i = (1 / (N * P(i)))^beta

where beta is annealed towards 1.0 over training.

References:
    Schaul et al. (2016) – "Prioritized Experience Replay."
    https://arxiv.org/abs/1511.05952
"""

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Sum-tree
# ---------------------------------------------------------------------------

class _SumTree:
    """
    Binary sum-tree for O(log N) priority updates and sampling.

    Leaves store individual transition priorities; each internal node stores
    the sum of its children.

    Args:
        capacity: Maximum number of transitions (must be > 0).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data_idx: int = 0   # next write position in the leaves
        self._size: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_priority(self) -> float:
        """Sum of all leaf priorities."""
        return float(self._tree[0])

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, priority: float) -> int:
        """
        Add a new leaf with the given priority.

        Args:
            priority: Priority value (must be > 0).

        Returns:
            Leaf index of the added element.
        """
        leaf_idx = self._data_idx + self.capacity - 1
        self._update(leaf_idx, priority)
        self._data_idx = (self._data_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return leaf_idx

    def update(self, leaf_idx: int, priority: float) -> None:
        """
        Update the priority of an existing leaf.

        Args:
            leaf_idx: Leaf index (as returned by :meth:`add`).
            priority: New priority value.
        """
        self._update(leaf_idx, priority)

    def sample(self, value: float) -> Tuple[int, int, float]:
        """
        Retrieve the leaf containing ``value`` in the cumulative priority sum.

        Args:
            value: A scalar in [0, total_priority).

        Returns:
            Tuple of (leaf_index, data_index, priority).
        """
        node_idx = 0
        while node_idx < self.capacity - 1:
            left = 2 * node_idx + 1
            right = left + 1
            if value <= self._tree[left]:
                node_idx = left
            else:
                value -= self._tree[left]
                node_idx = right
        data_idx = node_idx - (self.capacity - 1)
        return node_idx, data_idx, float(self._tree[node_idx])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        # Propagate delta up to the root
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self._tree[idx] += delta


# ---------------------------------------------------------------------------
# PER Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Stores (state, action, reward, next_state, done) transitions and
    supports priority-weighted sampling.

    Args:
        capacity:     Maximum number of stored transitions.
        alpha:        Prioritisation exponent.  0 = uniform, 1 = full PER.
        beta_start:   Initial value for the IS-weight exponent.
        beta_end:     Final value (annealed to 1.0 by ``beta_steps``).
        beta_steps:   Number of :meth:`sample` calls over which beta is
                      annealed from ``beta_start`` to ``beta_end``.
        epsilon:      Small constant added to priorities to ensure no
                      transition has zero probability.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._tree = _SumTree(capacity)
        self._data: List[Optional[tuple]] = [None] * capacity
        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta_start
        self._beta_end = beta_end
        self._beta_increment = (beta_end - beta_start) / max(1, beta_steps)
        self._epsilon = epsilon
        self._max_priority: float = 1.0
        self._write_ptr: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._tree.size

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
    ) -> None:
        """
        Add a transition to the buffer.

        New transitions receive the maximum priority seen so far (or the
        provided ``td_error``) to ensure they are sampled at least once.

        Args:
            state:      State at time t.
            action:     Action taken at time t.
            reward:     Reward received.
            next_state: State at time t+1.
            done:       Terminal flag.
            td_error:   Optional initial TD-error; if None, the maximum
                        priority seen so far is used.
        """
        if td_error is not None:
            priority = self._compute_priority(float(abs(td_error)))
        else:
            priority = self._max_priority

        leaf_idx = self._tree.add(priority)
        data_idx = leaf_idx - (self._capacity - 1)
        self._data[data_idx] = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[List[tuple], List[int], np.ndarray]:
        """
        Sample a mini-batch of transitions using priority-weighted sampling.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of:
            - batch:        List of (s, a, r, s', done) transition tuples.
            - leaf_indices: Leaf indices for subsequent priority updates.
            - is_weights:   Importance-sampling weights (normalised), shape (B,).

        Raises:
            RuntimeError: If the buffer does not contain enough transitions.
        """
        if self._tree.size < batch_size:
            raise RuntimeError(
                f"Buffer has {self._tree.size} transitions; "
                f"need {batch_size} to sample."
            )

        batch: List[tuple] = []
        leaf_indices: List[int] = []
        priorities: List[float] = []

        segment = self._tree.total_priority / batch_size
        self._beta = min(self._beta_end, self._beta + self._beta_increment)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx, data_idx, priority = self._tree.sample(value)

            transition = self._data[data_idx]
            if transition is None:
                # Fallback: sample from a filled slot
                transition = self._data[0]  # type: ignore[assignment]
            batch.append(transition)
            leaf_indices.append(leaf_idx)
            priorities.append(max(priority, self._epsilon))

        # IS weights
        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / (self._tree.total_priority + 1e-12)
        is_weights = (1.0 / (self._tree.size * probs + 1e-12)) ** self._beta
        is_weights /= is_weights.max()  # normalise to [0, 1]
        return batch, leaf_indices, is_weights.astype(np.float32)

    def update_priorities(
        self, leaf_indices: List[int], td_errors: np.ndarray
    ) -> None:
        """
        Update priorities for a set of sampled transitions after a gradient
        step.

        Args:
            leaf_indices: Leaf indices returned by :meth:`sample`.
            td_errors:    Absolute TD-errors from the latest forward pass.
        """
        for leaf_idx, td_err in zip(leaf_indices, td_errors):
            priority = self._compute_priority(float(abs(td_err)))
            self._tree.update(leaf_idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def get_leaf_priority(self, leaf_idx: int) -> float:
        """
        Return the raw priority stored at ``leaf_idx``.

        This is a public helper mainly intended for testing; production code
        should rely on :meth:`sample` and :meth:`update_priorities`.

        Args:
            leaf_idx: Leaf index as returned by :meth:`sample`.

        Returns:
            Priority value (float ≥ 0).
        """
        return float(self._tree._tree[leaf_idx])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_priority(self, td_error: float) -> float:
        return (abs(td_error) + self._epsilon) ** self._alpha
