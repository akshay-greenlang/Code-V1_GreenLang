# -*- coding: utf-8 -*-
"""
Experience Replay Buffer Module

This module provides sophisticated experience replay mechanisms for
GreenLang Process Heat agents, including prioritized replay, reservoir
sampling, and importance sampling for rare events critical in
industrial process monitoring.

Experience replay is essential for stabilizing online learning,
preventing catastrophic forgetting, and ensuring that important
historical patterns (like rare failure modes) remain in the training
distribution.

Example:
    >>> from greenlang.ml.self_learning import ExperienceReplayBuffer
    >>> buffer = ExperienceReplayBuffer(capacity=10000)
    >>> buffer.add(observation, action, reward, next_obs, done)
    >>> batch = buffer.sample(batch_size=32)
    >>> # Use batch for training
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple, NamedTuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import deque
import heapq
import random

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Sampling strategies for experience replay."""
    UNIFORM = "uniform"  # Random sampling
    PRIORITIZED = "prioritized"  # Priority-based (PER)
    RESERVOIR = "reservoir"  # Reservoir sampling
    STRATIFIED = "stratified"  # Class-balanced sampling
    IMPORTANCE = "importance"  # Importance sampling for rare events


class Experience(NamedTuple):
    """Single experience tuple."""
    state: np.ndarray
    action: Any
    reward: float
    next_state: np.ndarray
    done: bool
    info: Optional[Dict[str, Any]] = None


class ExperienceReplayConfig(BaseModel):
    """Configuration for experience replay buffer."""

    capacity: int = Field(
        default=10000,
        ge=100,
        description="Maximum buffer capacity"
    )
    sampling_strategy: SamplingStrategy = Field(
        default=SamplingStrategy.UNIFORM,
        description="Strategy for sampling experiences"
    )
    alpha: float = Field(
        default=0.6,
        ge=0,
        le=1.0,
        description="Priority exponent for PER"
    )
    beta: float = Field(
        default=0.4,
        ge=0,
        le=1.0,
        description="Importance sampling exponent"
    )
    beta_annealing_steps: int = Field(
        default=100000,
        ge=1,
        description="Steps to anneal beta to 1.0"
    )
    epsilon: float = Field(
        default=1e-6,
        gt=0,
        description="Small constant for priority calculation"
    )
    rare_event_threshold: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Threshold for identifying rare events"
    )
    rare_event_boost: float = Field(
        default=2.0,
        ge=1.0,
        description="Sampling boost factor for rare events"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class BufferStatistics(BaseModel):
    """Statistics about the replay buffer."""

    current_size: int = Field(
        ...,
        description="Current number of experiences"
    )
    capacity: int = Field(
        ...,
        description="Maximum capacity"
    )
    total_added: int = Field(
        ...,
        description="Total experiences added"
    )
    total_sampled: int = Field(
        ...,
        description="Total experiences sampled"
    )
    rare_events_count: int = Field(
        default=0,
        description="Number of rare events in buffer"
    )
    avg_priority: float = Field(
        default=0.0,
        description="Average priority (for PER)"
    )
    max_priority: float = Field(
        default=0.0,
        description="Maximum priority"
    )
    class_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of classes/labels"
    )


class SampleBatch(BaseModel):
    """A batch of sampled experiences."""

    states: Any = Field(
        ...,
        description="Batch of states"
    )
    actions: Any = Field(
        ...,
        description="Batch of actions"
    )
    rewards: Any = Field(
        ...,
        description="Batch of rewards"
    )
    next_states: Any = Field(
        ...,
        description="Batch of next states"
    )
    dones: Any = Field(
        ...,
        description="Batch of done flags"
    )
    indices: List[int] = Field(
        ...,
        description="Indices of sampled experiences"
    )
    weights: Optional[Any] = Field(
        default=None,
        description="Importance sampling weights"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )

    class Config:
        arbitrary_types_allowed = True


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.

    A binary tree where each node contains the sum of its children.
    Enables O(log n) sampling proportional to priorities.
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree.

        Args:
            capacity: Maximum number of leaves
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for given value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get sum of all priorities."""
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """Add experience with priority."""
        idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get experience for cumulative priority value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]

    def min_priority(self) -> float:
        """Get minimum priority."""
        return np.min(self.tree[-self.capacity:][self.tree[-self.capacity:] > 0])


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for GreenLang Process Heat agents.

    This class provides sophisticated experience replay capabilities
    including prioritized replay, reservoir sampling, and importance
    sampling for rare industrial events.

    Key capabilities:
    - Prioritized Experience Replay (PER)
    - Reservoir sampling for bounded memory
    - Importance sampling for rare events
    - Configurable buffer size and sampling strategies
    - Integration with online_learner.py
    - Provenance tracking

    Attributes:
        config: Buffer configuration
        _buffer: Internal storage
        _sum_tree: Sum tree for prioritized sampling
        _statistics: Buffer statistics

    Example:
        >>> buffer = ExperienceReplayBuffer(
        ...     config=ExperienceReplayConfig(
        ...         capacity=10000,
        ...         sampling_strategy=SamplingStrategy.PRIORITIZED,
        ...         alpha=0.6,
        ...         beta=0.4
        ...     )
        ... )
        >>> # Add experiences
        >>> for transition in transitions:
        ...     buffer.add(*transition)
        >>> # Sample batch for training
        >>> batch = buffer.sample(batch_size=32)
        >>> # Update priorities based on TD errors
        >>> buffer.update_priorities(batch.indices, td_errors)
    """

    def __init__(
        self,
        config: Optional[ExperienceReplayConfig] = None,
        capacity: Optional[int] = None
    ):
        """
        Initialize experience replay buffer.

        Args:
            config: Buffer configuration
            capacity: Override capacity
        """
        self.config = config or ExperienceReplayConfig()

        if capacity is not None:
            self.config.capacity = capacity

        # Initialize storage based on strategy
        if self.config.sampling_strategy == SamplingStrategy.PRIORITIZED:
            self._sum_tree = SumTree(self.config.capacity)
            self._max_priority = 1.0
        else:
            self._buffer: deque = deque(maxlen=self.config.capacity)
            self._priorities: List[float] = []
            self._sum_tree = None

        # For importance sampling of rare events
        self._rare_events: List[int] = []
        self._event_labels: List[Any] = []

        # Statistics tracking
        self._total_added = 0
        self._total_sampled = 0
        self._current_beta = self.config.beta
        self._step_count = 0

        # For reservoir sampling
        self._reservoir_count = 0

        np.random.seed(self.config.random_state)
        random.seed(self.config.random_state)

        logger.info(
            f"ExperienceReplayBuffer initialized: "
            f"capacity={self.config.capacity}, "
            f"strategy={self.config.sampling_strategy}"
        )

    def __len__(self) -> int:
        """Get current buffer size."""
        if self._sum_tree is not None:
            return self._sum_tree.n_entries
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
        is_rare_event: bool = False
    ) -> None:
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional info
            priority: Priority for PER (uses max if not provided)
            is_rare_event: Mark as rare event for importance sampling

        Example:
            >>> buffer.add(obs, action, reward, next_obs, done)
        """
        experience = Experience(
            state=np.asarray(state),
            action=action,
            reward=reward,
            next_state=np.asarray(next_state),
            done=done,
            info=info
        )

        if self.config.sampling_strategy == SamplingStrategy.PRIORITIZED:
            # Use max priority for new experiences
            p = priority if priority is not None else self._max_priority
            self._sum_tree.add(p ** self.config.alpha, experience)

        elif self.config.sampling_strategy == SamplingStrategy.RESERVOIR:
            self._add_reservoir(experience)

        else:
            self._buffer.append(experience)

            if is_rare_event:
                self._rare_events.append(len(self._buffer) - 1)

        self._total_added += 1

        # Track for importance sampling
        if is_rare_event:
            logger.debug(f"Rare event added at index {self._total_added}")

    def _add_reservoir(self, experience: Experience) -> None:
        """Add experience using reservoir sampling."""
        self._reservoir_count += 1

        if len(self._buffer) < self.config.capacity:
            self._buffer.append(experience)
        else:
            # Reservoir sampling: replace with probability capacity/count
            replace_prob = self.config.capacity / self._reservoir_count
            if random.random() < replace_prob:
                replace_idx = random.randint(0, self.config.capacity - 1)
                self._buffer[replace_idx] = experience

    def sample(
        self,
        batch_size: int = 32,
        return_indices: bool = True
    ) -> SampleBatch:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            return_indices: Include indices in result

        Returns:
            SampleBatch with experiences and metadata

        Example:
            >>> batch = buffer.sample(batch_size=64)
            >>> states, actions, rewards = batch.states, batch.actions, batch.rewards
        """
        if len(self) < batch_size:
            batch_size = len(self)

        if len(self) == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Update beta for importance sampling annealing
        self._step_count += 1
        self._current_beta = min(
            1.0,
            self.config.beta + (1.0 - self.config.beta) *
            (self._step_count / self.config.beta_annealing_steps)
        )

        # Sample based on strategy
        if self.config.sampling_strategy == SamplingStrategy.PRIORITIZED:
            indices, experiences, weights = self._sample_prioritized(batch_size)

        elif self.config.sampling_strategy == SamplingStrategy.IMPORTANCE:
            indices, experiences, weights = self._sample_importance(batch_size)

        elif self.config.sampling_strategy == SamplingStrategy.STRATIFIED:
            indices, experiences, weights = self._sample_stratified(batch_size)

        else:
            indices, experiences, weights = self._sample_uniform(batch_size)

        # Convert to arrays
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        self._total_sampled += batch_size

        # Calculate provenance
        provenance_hash = self._calculate_provenance(indices, batch_size)

        return SampleBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            indices=indices,
            weights=weights,
            provenance_hash=provenance_hash
        )

    def _sample_uniform(
        self,
        batch_size: int
    ) -> Tuple[List[int], List[Experience], Optional[np.ndarray]]:
        """Sample uniformly at random."""
        indices = random.sample(range(len(self._buffer)), batch_size)
        experiences = [self._buffer[i] for i in indices]
        return indices, experiences, None

    def _sample_prioritized(
        self,
        batch_size: int
    ) -> Tuple[List[int], List[Experience], np.ndarray]:
        """Sample based on priorities (PER)."""
        indices = []
        experiences = []
        priorities = []

        total = self._sum_tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            # Sample from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, experience = self._sum_tree.get(s)
            indices.append(idx)
            experiences.append(experience)
            priorities.append(priority)

        # Calculate importance sampling weights
        min_priority = self._sum_tree.min_priority()
        max_weight = (min_priority / total * len(self)) ** (-self._current_beta)

        weights = []
        for priority in priorities:
            weight = (priority / total * len(self)) ** (-self._current_beta)
            weights.append(weight / max_weight)

        return indices, experiences, np.array(weights)

    def _sample_importance(
        self,
        batch_size: int
    ) -> Tuple[List[int], List[Experience], np.ndarray]:
        """Sample with importance weighting for rare events."""
        buffer_list = list(self._buffer)
        n = len(buffer_list)

        # Calculate sampling probabilities
        probs = np.ones(n) / n

        # Boost probability of rare events
        for idx in self._rare_events:
            if idx < n:
                probs[idx] *= self.config.rare_event_boost

        # Normalize
        probs /= probs.sum()

        # Sample
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        experiences = [buffer_list[i] for i in indices]

        # Calculate importance weights (inverse of sampling probability)
        weights = 1.0 / (n * probs[indices])
        weights /= weights.max()

        return list(indices), experiences, weights

    def _sample_stratified(
        self,
        batch_size: int
    ) -> Tuple[List[int], List[Experience], Optional[np.ndarray]]:
        """Sample with stratification by labels/classes."""
        buffer_list = list(self._buffer)

        # Group by label if available
        label_indices: Dict[Any, List[int]] = {}
        for i, exp in enumerate(buffer_list):
            label = exp.info.get('label', 'default') if exp.info else 'default'
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(i)

        # Sample equally from each group
        indices = []
        samples_per_group = max(1, batch_size // len(label_indices))

        for label, group_indices in label_indices.items():
            n_samples = min(samples_per_group, len(group_indices))
            selected = random.sample(group_indices, n_samples)
            indices.extend(selected)

        # Fill remaining with uniform sampling
        while len(indices) < batch_size:
            idx = random.randint(0, len(buffer_list) - 1)
            if idx not in indices:
                indices.append(idx)

        indices = indices[:batch_size]
        experiences = [buffer_list[i] for i in indices]

        return indices, experiences, None

    def update_priorities(
        self,
        indices: List[int],
        priorities: Union[List[float], np.ndarray]
    ) -> None:
        """
        Update priorities for PER.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically TD errors)

        Example:
            >>> # After computing TD errors
            >>> buffer.update_priorities(batch.indices, td_errors)
        """
        if self.config.sampling_strategy != SamplingStrategy.PRIORITIZED:
            logger.warning("update_priorities only applicable for PER")
            return

        for idx, priority in zip(indices, priorities):
            # Add epsilon for stability
            priority = abs(priority) + self.config.epsilon
            self._max_priority = max(self._max_priority, priority)
            self._sum_tree.update(idx, priority ** self.config.alpha)

    def mark_rare_event(
        self,
        index: int,
        boost_factor: Optional[float] = None
    ) -> None:
        """
        Mark an experience as a rare event for importance sampling.

        Args:
            index: Index of experience
            boost_factor: Override boost factor

        Example:
            >>> # Mark failure mode as rare event
            >>> buffer.mark_rare_event(idx, boost_factor=3.0)
        """
        if index not in self._rare_events:
            self._rare_events.append(index)

        if boost_factor is not None and self.config.sampling_strategy == SamplingStrategy.PRIORITIZED:
            current_priority = self._sum_tree.tree[index]
            self._sum_tree.update(index, current_priority * boost_factor)

    def _calculate_provenance(
        self,
        indices: List[int],
        batch_size: int
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = f"{sorted(indices)}|{batch_size}|{self._total_sampled}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_statistics(self) -> BufferStatistics:
        """Get buffer statistics."""
        if self._sum_tree is not None:
            priorities = self._sum_tree.tree[-self._sum_tree.capacity:]
            valid_priorities = priorities[priorities > 0]
            avg_priority = float(np.mean(valid_priorities)) if len(valid_priorities) > 0 else 0.0
            max_priority = float(np.max(valid_priorities)) if len(valid_priorities) > 0 else 0.0
        else:
            avg_priority = 0.0
            max_priority = 0.0

        # Calculate class distribution
        class_dist: Dict[str, int] = {}
        buffer_list = self._buffer if hasattr(self, '_buffer') else []
        for exp in buffer_list:
            if exp is not None and hasattr(exp, 'info') and exp.info:
                label = str(exp.info.get('label', 'unknown'))
                class_dist[label] = class_dist.get(label, 0) + 1

        return BufferStatistics(
            current_size=len(self),
            capacity=self.config.capacity,
            total_added=self._total_added,
            total_sampled=self._total_sampled,
            rare_events_count=len(self._rare_events),
            avg_priority=avg_priority,
            max_priority=max_priority,
            class_distribution=class_dist
        )

    def clear(self) -> None:
        """Clear the buffer."""
        if self._sum_tree is not None:
            self._sum_tree = SumTree(self.config.capacity)
            self._max_priority = 1.0
        else:
            self._buffer.clear()
            self._priorities.clear()

        self._rare_events.clear()
        self._total_added = 0
        self._total_sampled = 0

        logger.info("ExperienceReplayBuffer cleared")

    def save(self, path: str) -> None:
        """
        Save buffer to disk.

        Args:
            path: Path to save buffer
        """
        import pickle

        data = {
            'buffer': list(self._buffer) if hasattr(self, '_buffer') else None,
            'sum_tree': self._sum_tree,
            'config': self.config.dict(),
            'statistics': {
                'total_added': self._total_added,
                'total_sampled': self._total_sampled,
                'rare_events': self._rare_events
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Buffer saved to {path}")

    def load(self, path: str) -> None:
        """
        Load buffer from disk.

        Args:
            path: Path to load buffer from
        """
        import pickle

        with open(path, 'rb') as f:
            data = pickle.load(f)

        if data['buffer'] is not None:
            self._buffer = deque(data['buffer'], maxlen=self.config.capacity)
        self._sum_tree = data['sum_tree']
        self.config = ExperienceReplayConfig(**data['config'])
        self._total_added = data['statistics']['total_added']
        self._total_sampled = data['statistics']['total_sampled']
        self._rare_events = data['statistics']['rare_events']

        logger.info(f"Buffer loaded from {path}")

    # Integration with online_learner.py
    def to_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert buffer to training data for online_learner.

        Returns:
            Tuple of (features, labels) arrays

        Example:
            >>> X, y = buffer.to_training_data()
            >>> online_learner.learn_many(X, y)
        """
        buffer_list = list(self._buffer) if hasattr(self, '_buffer') else []

        if len(buffer_list) == 0:
            return np.array([]), np.array([])

        states = np.array([e.state for e in buffer_list if e is not None])
        rewards = np.array([e.reward for e in buffer_list if e is not None])

        return states, rewards

    def sample_for_online_learner(
        self,
        batch_size: int = 32
    ) -> Tuple[List[Dict[str, Any]], List[Union[int, float]]]:
        """
        Sample batch in format compatible with online_learner.py.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (feature_dicts, labels)
        """
        batch = self.sample(batch_size)

        # Convert states to feature dicts
        feature_dicts = []
        for state in batch.states:
            if isinstance(state, dict):
                feature_dicts.append(state)
            else:
                # Convert array to dict
                feature_dict = {f"f_{i}": float(v) for i, v in enumerate(state.flatten())}
                feature_dicts.append(feature_dict)

        labels = batch.rewards.tolist()

        return feature_dicts, labels


# Specialized buffers for Process Heat applications
class ProcessHeatReplayBuffer(ExperienceReplayBuffer):
    """
    Specialized replay buffer for Process Heat applications.

    Includes domain-specific features like:
    - Automatic rare event detection for equipment failures
    - Temperature/pressure anomaly tracking
    - Efficiency degradation patterns
    """

    def __init__(
        self,
        config: Optional[ExperienceReplayConfig] = None,
        failure_reward_threshold: float = -10.0,
        anomaly_z_threshold: float = 3.0
    ):
        """
        Initialize Process Heat replay buffer.

        Args:
            config: Buffer configuration
            failure_reward_threshold: Reward threshold for failure detection
            anomaly_z_threshold: Z-score threshold for anomaly detection
        """
        super().__init__(config)
        self.failure_reward_threshold = failure_reward_threshold
        self.anomaly_z_threshold = anomaly_z_threshold

        # Track running statistics for anomaly detection
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._reward_count = 0

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
        is_rare_event: bool = False
    ) -> None:
        """Add experience with automatic rare event detection."""
        # Update running statistics
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        if self._reward_count > 1:
            self._reward_std = np.sqrt(
                ((self._reward_count - 2) * self._reward_std ** 2 + delta ** 2) /
                (self._reward_count - 1)
            )

        # Detect rare events
        if reward <= self.failure_reward_threshold:
            is_rare_event = True
            logger.warning(f"Failure event detected: reward={reward}")

        # Detect anomalies using z-score
        if self._reward_std > 0:
            z_score = abs(reward - self._reward_mean) / self._reward_std
            if z_score > self.anomaly_z_threshold:
                is_rare_event = True
                logger.info(f"Anomaly detected: z-score={z_score:.2f}")

        # Add to buffer
        super().add(
            state, action, reward, next_state, done,
            info, priority, is_rare_event
        )


# Factory functions
def create_prioritized_buffer(
    capacity: int = 10000,
    alpha: float = 0.6,
    beta: float = 0.4
) -> ExperienceReplayBuffer:
    """Create a prioritized experience replay buffer."""
    config = ExperienceReplayConfig(
        capacity=capacity,
        sampling_strategy=SamplingStrategy.PRIORITIZED,
        alpha=alpha,
        beta=beta
    )
    return ExperienceReplayBuffer(config)


def create_reservoir_buffer(
    capacity: int = 10000
) -> ExperienceReplayBuffer:
    """Create a reservoir sampling buffer."""
    config = ExperienceReplayConfig(
        capacity=capacity,
        sampling_strategy=SamplingStrategy.RESERVOIR
    )
    return ExperienceReplayBuffer(config)


def create_importance_sampling_buffer(
    capacity: int = 10000,
    rare_event_boost: float = 2.0
) -> ExperienceReplayBuffer:
    """Create an importance sampling buffer for rare events."""
    config = ExperienceReplayConfig(
        capacity=capacity,
        sampling_strategy=SamplingStrategy.IMPORTANCE,
        rare_event_boost=rare_event_boost
    )
    return ExperienceReplayBuffer(config)


# Unit test stubs
class TestExperienceReplayBuffer:
    """Unit tests for ExperienceReplayBuffer."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        buffer = ExperienceReplayBuffer()
        assert buffer.config.capacity == 10000
        assert buffer.config.sampling_strategy == SamplingStrategy.UNIFORM

    def test_add_and_sample(self):
        """Test adding and sampling experiences."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Add experiences
        for i in range(50):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, i % 3, float(i), next_state, False)

        assert len(buffer) == 50

        # Sample
        batch = buffer.sample(batch_size=10)
        assert batch.states.shape[0] == 10

    def test_prioritized_replay(self):
        """Test prioritized experience replay."""
        buffer = create_prioritized_buffer(capacity=100)

        # Add experiences
        for i in range(50):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, float(i), next_state, False)

        # Sample
        batch = buffer.sample(batch_size=10)
        assert batch.weights is not None

        # Update priorities
        buffer.update_priorities(batch.indices, np.random.rand(10))

    def test_reservoir_sampling(self):
        """Test reservoir sampling."""
        buffer = create_reservoir_buffer(capacity=100)

        # Add more than capacity
        for i in range(200):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, float(i), next_state, False)

        assert len(buffer) == 100

    def test_rare_event_tracking(self):
        """Test rare event tracking."""
        buffer = create_importance_sampling_buffer(capacity=100)

        # Add normal experiences
        for i in range(40):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, float(i), next_state, False)

        # Add rare events
        for i in range(10):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, -100.0, next_state, True, is_rare_event=True)

        stats = buffer.get_statistics()
        assert stats.rare_events_count == 10

    def test_statistics(self):
        """Test buffer statistics."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(50):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, float(i), next_state, False)

        buffer.sample(batch_size=10)

        stats = buffer.get_statistics()
        assert stats.current_size == 50
        assert stats.total_added == 50
        assert stats.total_sampled == 10

    def test_provenance(self):
        """Test provenance hash generation."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(50):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, float(i), next_state, False)

        batch = buffer.sample(batch_size=10)
        assert len(batch.provenance_hash) == 64

    def test_online_learner_integration(self):
        """Test integration with online_learner format."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(50):
            state = np.random.randn(4)
            next_state = np.random.randn(4)
            buffer.add(state, 0, float(i), next_state, False)

        features, labels = buffer.sample_for_online_learner(batch_size=10)
        assert len(features) == 10
        assert len(labels) == 10
        assert isinstance(features[0], dict)
