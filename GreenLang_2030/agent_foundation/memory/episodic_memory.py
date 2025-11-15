"""
Episodic Memory implementation for GreenLang Agent Foundation.

This module implements experience and event-based memory with learning
mechanisms, pattern extraction, and case-based reasoning capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Emotion/sentiment types for episodes."""
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    NEUTRAL = "neutral"


class ActionType(Enum):
    """Types of actions in episodes."""
    CALCULATION = "calculation"
    VALIDATION = "validation"
    RETRIEVAL = "retrieval"
    STORAGE = "storage"
    ANALYSIS = "analysis"
    DECISION = "decision"
    COMMUNICATION = "communication"


class Episode(BaseModel):
    """Individual episode in episodic memory."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique episode ID")
    agent_id: str = Field(..., description="Agent identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Episode timestamp")
    context: Dict[str, Any] = Field(..., description="Environmental state during episode")
    actions: List[Dict[str, Any]] = Field(..., description="List of actions taken")
    outcomes: Dict[str, Any] = Field(..., description="Results achieved")
    rewards: Dict[str, float] = Field(default_factory=dict, description="Success metrics")
    emotions: Dict[EmotionType, float] = Field(default_factory=dict, description="Emotional state")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Episode importance")
    embeddings: Optional[List[float]] = Field(None, description="Vector representation")
    provenance_hash: str = Field("", description="SHA-256 hash for audit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('provenance_hash', always=True)
    def calculate_provenance(cls, v, values):
        """Calculate SHA-256 hash if not provided."""
        if not v and 'episode_id' in values:
            episode_str = json.dumps({
                'episode_id': values['episode_id'],
                'agent_id': values.get('agent_id', ''),
                'timestamp': values.get('timestamp', datetime.now()).isoformat(),
                'actions': values.get('actions', [])
            }, sort_keys=True, default=str)
            return hashlib.sha256(episode_str.encode()).hexdigest()
        return v

    def calculate_importance_score(self) -> float:
        """
        Calculate importance based on rewards, emotions, and outcomes.

        Returns:
            Importance score 0-1
        """
        # Reward component (40% weight)
        reward_score = np.mean(list(self.rewards.values())) if self.rewards else 0.5

        # Emotion component (30% weight) - confidence and satisfaction are positive
        emotion_score = 0.5
        if self.emotions:
            positive_emotions = [
                self.emotions.get(EmotionType.CONFIDENT, 0),
                self.emotions.get(EmotionType.SATISFIED, 0)
            ]
            negative_emotions = [
                self.emotions.get(EmotionType.UNCERTAIN, 0),
                self.emotions.get(EmotionType.FRUSTRATED, 0)
            ]
            emotion_score = np.mean(positive_emotions) - np.mean(negative_emotions) * 0.5
            emotion_score = max(0, min(1, (emotion_score + 1) / 2))  # Normalize to 0-1

        # Outcome success component (30% weight)
        outcome_score = self.outcomes.get('success_rate', 0.5) if self.outcomes else 0.5

        # Weighted combination
        self.importance = (
            0.4 * reward_score +
            0.3 * emotion_score +
            0.3 * outcome_score
        )

        return self.importance


@dataclass
class Pattern:
    """Extracted pattern from episodes."""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    action_sequence: List[str]
    typical_context: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    episodes: List[str]  # Episode IDs


class LearningMechanism:
    """Learning mechanisms for episodic memory."""

    def __init__(self):
        """Initialize learning mechanism."""
        self.patterns: Dict[str, Pattern] = {}
        self.case_library: Dict[str, Episode] = {}
        self.learning_stats = {
            "patterns_extracted": 0,
            "cases_stored": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0
        }

    def experience_replay(
        self,
        episodes: List[Episode],
        selection_method: str = "importance",
        sample_size: int = 100
    ) -> List[Episode]:
        """
        Replay successful episodes for learning.

        Args:
            episodes: Available episodes
            selection_method: Method for selecting episodes
            sample_size: Number of episodes to replay

        Returns:
            Selected episodes for replay
        """
        if not episodes:
            return []

        if selection_method == "importance":
            # Importance-weighted sampling
            weights = np.array([e.importance for e in episodes])
            if weights.sum() == 0:
                weights = np.ones(len(episodes))
            probabilities = weights / weights.sum()

            indices = np.random.choice(
                len(episodes),
                size=min(sample_size, len(episodes)),
                replace=False,
                p=probabilities
            )
            selected = [episodes[i] for i in indices]

        elif selection_method == "recent":
            # Select most recent episodes
            sorted_episodes = sorted(episodes, key=lambda x: x.timestamp, reverse=True)
            selected = sorted_episodes[:sample_size]

        elif selection_method == "successful":
            # Select episodes with high rewards
            sorted_episodes = sorted(
                episodes,
                key=lambda x: np.mean(list(x.rewards.values())) if x.rewards else 0,
                reverse=True
            )
            selected = sorted_episodes[:sample_size]

        else:
            # Random selection
            indices = np.random.choice(
                len(episodes),
                size=min(sample_size, len(episodes)),
                replace=False
            )
            selected = [episodes[i] for i in indices]

        logger.info(f"Selected {len(selected)} episodes for experience replay")
        return selected

    def extract_patterns(
        self,
        episodes: List[Episode],
        min_support: float = 0.05,
        min_confidence: float = 0.7
    ) -> List[Pattern]:
        """
        Identify recurring patterns in episodes.

        Args:
            episodes: Episodes to analyze
            min_support: Minimum frequency threshold
            min_confidence: Minimum confidence threshold

        Returns:
            List of extracted patterns
        """
        if len(episodes) < 10:
            return []

        # Extract action sequences
        action_sequences = []
        for episode in episodes:
            sequence = [action.get('type', 'unknown') for action in episode.actions]
            action_sequences.append(sequence)

        # Find frequent subsequences
        patterns = self._sequential_pattern_mining(
            action_sequences,
            min_support,
            episodes
        )

        # Filter by confidence
        valid_patterns = []
        for pattern in patterns:
            # Calculate confidence based on outcome consistency
            episode_outcomes = [
                episodes[i].outcomes
                for i, seq in enumerate(action_sequences)
                if self._contains_sequence(seq, pattern['sequence'])
            ]

            if episode_outcomes:
                # Check outcome consistency
                success_rates = [
                    o.get('success_rate', 0) for o in episode_outcomes
                ]
                confidence = np.mean(success_rates) if success_rates else 0

                if confidence >= min_confidence:
                    pattern_obj = Pattern(
                        pattern_id=hashlib.sha256(
                            str(pattern['sequence']).encode()
                        ).hexdigest()[:16],
                        pattern_type="action_sequence",
                        frequency=pattern['count'],
                        confidence=confidence,
                        action_sequence=pattern['sequence'],
                        typical_context=self._extract_typical_context(
                            episodes, pattern['sequence']
                        ),
                        expected_outcomes=self._extract_expected_outcomes(
                            episodes, pattern['sequence']
                        ),
                        episodes=pattern['episodes']
                    )
                    valid_patterns.append(pattern_obj)
                    self.patterns[pattern_obj.pattern_id] = pattern_obj

        self.learning_stats["patterns_extracted"] += len(valid_patterns)
        logger.info(f"Extracted {len(valid_patterns)} patterns from {len(episodes)} episodes")
        return valid_patterns

    def _sequential_pattern_mining(
        self,
        sequences: List[List[str]],
        min_support: float,
        episodes: List[Episode]
    ) -> List[Dict]:
        """
        Mine sequential patterns from action sequences.

        Args:
            sequences: List of action sequences
            min_support: Minimum support threshold
            episodes: Original episodes

        Returns:
            List of patterns
        """
        patterns = []
        min_count = int(len(sequences) * min_support)

        # Find frequent 2-grams, 3-grams, etc.
        for n in range(2, 6):  # Look for patterns of length 2-5
            ngram_counts = Counter()
            ngram_episodes = defaultdict(list)

            for i, seq in enumerate(sequences):
                if len(seq) >= n:
                    for j in range(len(seq) - n + 1):
                        ngram = tuple(seq[j:j+n])
                        ngram_counts[ngram] += 1
                        ngram_episodes[ngram].append(episodes[i].episode_id)

            # Filter by minimum support
            for ngram, count in ngram_counts.items():
                if count >= min_count:
                    patterns.append({
                        'sequence': list(ngram),
                        'count': count,
                        'support': count / len(sequences),
                        'episodes': ngram_episodes[ngram]
                    })

        return patterns

    def _contains_sequence(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains pattern as subsequence."""
        if len(pattern) > len(sequence):
            return False

        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False

    def _extract_typical_context(
        self,
        episodes: List[Episode],
        pattern: List[str]
    ) -> Dict[str, Any]:
        """Extract typical context for a pattern."""
        matching_contexts = []

        for episode in episodes:
            sequence = [action.get('type', 'unknown') for action in episode.actions]
            if self._contains_sequence(sequence, pattern):
                matching_contexts.append(episode.context)

        if not matching_contexts:
            return {}

        # Find common context elements
        typical_context = {}
        all_keys = set()
        for ctx in matching_contexts:
            all_keys.update(ctx.keys())

        for key in all_keys:
            values = [ctx.get(key) for ctx in matching_contexts if key in ctx]
            if values:
                # Use most common value for categorical, mean for numeric
                if isinstance(values[0], (int, float)):
                    typical_context[key] = np.mean(values)
                else:
                    typical_context[key] = Counter(values).most_common(1)[0][0]

        return typical_context

    def _extract_expected_outcomes(
        self,
        episodes: List[Episode],
        pattern: List[str]
    ) -> Dict[str, Any]:
        """Extract expected outcomes for a pattern."""
        matching_outcomes = []

        for episode in episodes:
            sequence = [action.get('type', 'unknown') for action in episode.actions]
            if self._contains_sequence(sequence, pattern):
                matching_outcomes.append(episode.outcomes)

        if not matching_outcomes:
            return {}

        # Aggregate outcomes
        expected_outcomes = {}
        all_keys = set()
        for outcome in matching_outcomes:
            all_keys.update(outcome.keys())

        for key in all_keys:
            values = [o.get(key) for o in matching_outcomes if key in o]
            if values:
                if isinstance(values[0], (int, float)):
                    expected_outcomes[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values)
                    }
                else:
                    counts = Counter(values)
                    expected_outcomes[key] = dict(counts)

        return expected_outcomes

    def case_based_reasoning(
        self,
        current_situation: Dict[str, Any],
        episodes: List[Episode],
        similarity_threshold: float = 0.85
    ) -> Optional[Dict[str, Any]]:
        """
        Find and adapt solutions from similar past situations.

        Args:
            current_situation: Current context/situation
            episodes: Available episodes
            similarity_threshold: Minimum similarity required

        Returns:
            Adapted solution or None
        """
        # Find most similar episode
        best_match = None
        best_similarity = 0

        for episode in episodes:
            similarity = self._calculate_similarity(
                current_situation,
                episode.context
            )

            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = episode

        if not best_match:
            logger.debug(f"No similar case found (threshold: {similarity_threshold})")
            return None

        # Store in case library
        self.case_library[best_match.episode_id] = best_match
        self.learning_stats["cases_stored"] += 1

        # Adapt the solution
        adapted_solution = self._adapt_solution(
            current_situation,
            best_match,
            best_similarity
        )

        logger.info(f"Found similar case with {best_similarity:.2%} similarity")
        return adapted_solution

    def _calculate_similarity(
        self,
        situation1: Dict[str, Any],
        situation2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two situations.

        Args:
            situation1: First situation
            situation2: Second situation

        Returns:
            Similarity score 0-1
        """
        if not situation1 or not situation2:
            return 0

        # Get common keys
        keys1 = set(situation1.keys())
        keys2 = set(situation2.keys())
        common_keys = keys1.intersection(keys2)

        if not common_keys:
            return 0

        similarities = []

        for key in common_keys:
            val1 = situation1[key]
            val2 = situation2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                if val1 == val2:
                    sim = 1.0
                else:
                    # Normalize difference
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        sim = 1 - abs(val1 - val2) / max_val
                    else:
                        sim = 1.0
                similarities.append(sim)

            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                sim = 1.0 if val1 == val2 else 0.0
                similarities.append(sim)

            elif isinstance(val1, list) and isinstance(val2, list):
                # List similarity (Jaccard index)
                set1 = set(val1)
                set2 = set(val2)
                if set1 or set2:
                    sim = len(set1.intersection(set2)) / len(set1.union(set2))
                else:
                    sim = 1.0
                similarities.append(sim)

        # Weight by key coverage
        coverage = len(common_keys) / max(len(keys1), len(keys2))
        similarity = np.mean(similarities) * coverage if similarities else 0

        return similarity

    def _adapt_solution(
        self,
        current_situation: Dict[str, Any],
        similar_episode: Episode,
        similarity: float
    ) -> Dict[str, Any]:
        """
        Adapt solution from similar episode to current situation.

        Args:
            current_situation: Current context
            similar_episode: Most similar past episode
            similarity: Similarity score

        Returns:
            Adapted solution
        """
        # Start with the successful actions from similar episode
        adapted_actions = similar_episode.actions.copy()

        # Adjust based on context differences
        context_diff = self._get_context_differences(
            current_situation,
            similar_episode.context
        )

        # Modify actions based on differences
        for i, action in enumerate(adapted_actions):
            # Adjust parameters based on context differences
            if 'parameters' in action:
                for key, diff in context_diff.items():
                    if key in action['parameters']:
                        # Simple linear adjustment
                        if isinstance(diff, (int, float)):
                            action['parameters'][key] = (
                                action['parameters'][key] + diff * (1 - similarity)
                            )

        solution = {
            'actions': adapted_actions,
            'expected_outcomes': similar_episode.outcomes,
            'confidence': similarity,
            'source_episode': similar_episode.episode_id,
            'adaptations': context_diff
        }

        return solution

    def _get_context_differences(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get differences between two contexts."""
        differences = {}

        all_keys = set(context1.keys()).union(set(context2.keys()))

        for key in all_keys:
            val1 = context1.get(key)
            val2 = context2.get(key)

            if val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    differences[key] = val1 - val2
                else:
                    differences[key] = f"{val2} -> {val1}"

        return differences


class EpisodicMemory:
    """
    Episodic memory system for experience-based learning.

    Manages episode storage, pattern extraction, and case-based reasoning
    with hierarchical consolidation to semantic memory.
    """

    def __init__(self, agent_id: str = "default"):
        """
        Initialize episodic memory.

        Args:
            agent_id: Agent identifier
        """
        self.agent_id = agent_id
        self.episodes: List[Episode] = []
        self.learning = LearningMechanism()

        # Consolidation parameters
        self.consolidation_interval_hours = 6
        self.compression_method = "hierarchical_summarization"
        self.transfer_importance_threshold = 0.8
        self.transfer_frequency_threshold = 5

        # Episode buffer
        self.max_episodes = 10000
        self.replay_frequency = 100  # Replay every N episodes

        # Statistics
        self.stats = {
            "total_episodes": 0,
            "consolidated_episodes": 0,
            "patterns_found": 0,
            "successful_adaptations": 0
        }

    async def record_episode(
        self,
        context: Dict[str, Any],
        actions: List[Dict[str, Any]],
        outcomes: Dict[str, Any],
        rewards: Optional[Dict[str, float]] = None,
        emotions: Optional[Dict[EmotionType, float]] = None,
        embeddings: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a new episode.

        Args:
            context: Environmental state
            actions: List of actions taken
            outcomes: Results achieved
            rewards: Optional success metrics
            emotions: Optional emotional state
            embeddings: Optional vector representation
            metadata: Optional additional metadata

        Returns:
            Episode ID
        """
        start_time = datetime.now()

        episode = Episode(
            agent_id=self.agent_id,
            context=context,
            actions=actions,
            outcomes=outcomes,
            rewards=rewards or {},
            emotions=emotions or {},
            embeddings=embeddings,
            metadata=metadata or {}
        )

        # Calculate importance
        episode.calculate_importance_score()

        # Add to episodes
        self.episodes.append(episode)
        self.stats["total_episodes"] += 1

        # Manage buffer size
        if len(self.episodes) > self.max_episodes:
            # Remove oldest low-importance episodes
            self.episodes.sort(key=lambda x: x.importance)
            self.episodes = self.episodes[-self.max_episodes:]

        # Trigger experience replay periodically
        if self.stats["total_episodes"] % self.replay_frequency == 0:
            await self._trigger_experience_replay()

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Recorded episode {episode.episode_id[:8]} "
            f"with importance {episode.importance:.2f} ({processing_time:.2f}ms)"
        )

        return episode.episode_id

    async def retrieve_similar_episodes(
        self,
        context: Dict[str, Any],
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Episode]:
        """
        Retrieve episodes similar to given context.

        Args:
            context: Context to match
            limit: Maximum episodes to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar episodes
        """
        similarities = []

        for episode in self.episodes:
            similarity = self.learning._calculate_similarity(
                context,
                episode.context
            )
            if similarity >= min_similarity:
                similarities.append((similarity, episode))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top episodes
        return [episode for _, episode in similarities[:limit]]

    async def find_solution(
        self,
        current_situation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find solution for current situation using case-based reasoning.

        Args:
            current_situation: Current context

        Returns:
            Adapted solution or None
        """
        solution = self.learning.case_based_reasoning(
            current_situation,
            self.episodes,
            similarity_threshold=0.85
        )

        if solution:
            self.stats["successful_adaptations"] += 1
            logger.info(f"Found solution with {solution['confidence']:.2%} confidence")

        return solution

    async def extract_patterns(self) -> List[Pattern]:
        """
        Extract patterns from recent episodes.

        Returns:
            List of extracted patterns
        """
        patterns = self.learning.extract_patterns(
            self.episodes,
            min_support=0.05,
            min_confidence=0.7
        )

        self.stats["patterns_found"] += len(patterns)
        return patterns

    async def _trigger_experience_replay(self) -> None:
        """Trigger experience replay for learning."""
        selected_episodes = self.learning.experience_replay(
            self.episodes,
            selection_method="importance",
            sample_size=min(100, len(self.episodes))
        )

        # Extract patterns from replayed episodes
        if len(selected_episodes) >= 10:
            await self.extract_patterns()

    async def consolidate_to_semantic(
        self,
        importance_threshold: Optional[float] = None,
        frequency_threshold: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Consolidate important episodes to semantic memory.

        Args:
            importance_threshold: Override importance threshold
            frequency_threshold: Override frequency threshold

        Returns:
            List of consolidated memories
        """
        importance_threshold = importance_threshold or self.transfer_importance_threshold
        frequency_threshold = frequency_threshold or self.transfer_frequency_threshold

        consolidated = []

        # Group episodes by pattern
        pattern_episodes = defaultdict(list)
        for episode in self.episodes:
            # Check if episode matches any known pattern
            for pattern in self.learning.patterns.values():
                if episode.episode_id in pattern.episodes:
                    pattern_episodes[pattern.pattern_id].append(episode)
                    break

        # Consolidate pattern-based episodes
        for pattern_id, episodes in pattern_episodes.items():
            if len(episodes) >= frequency_threshold:
                pattern = self.learning.patterns[pattern_id]

                # Create semantic memory from pattern
                semantic_memory = {
                    'type': 'procedure',
                    'content': {
                        'action_sequence': pattern.action_sequence,
                        'typical_context': pattern.typical_context,
                        'expected_outcomes': pattern.expected_outcomes
                    },
                    'confidence': pattern.confidence,
                    'frequency': len(episodes),
                    'source_episodes': [e.episode_id for e in episodes],
                    'provenance_hash': hashlib.sha256(
                        f"{pattern_id}{datetime.now().isoformat()}".encode()
                    ).hexdigest()
                }
                consolidated.append(semantic_memory)

        # Consolidate high-importance individual episodes
        for episode in self.episodes:
            if episode.importance >= importance_threshold:
                # Create semantic memory from important episode
                semantic_memory = {
                    'type': 'fact',
                    'content': {
                        'context': episode.context,
                        'successful_actions': episode.actions,
                        'outcomes': episode.outcomes
                    },
                    'importance': episode.importance,
                    'source_episode': episode.episode_id,
                    'provenance_hash': episode.provenance_hash
                }
                consolidated.append(semantic_memory)

        self.stats["consolidated_episodes"] += len(consolidated)
        logger.info(f"Consolidated {len(consolidated)} episodes to semantic memory")

        return consolidated

    async def compress_episodes(self) -> Dict[str, Any]:
        """
        Compress episodes using hierarchical summarization.

        Returns:
            Compression statistics
        """
        original_count = len(self.episodes)

        if self.compression_method == "hierarchical_summarization":
            # Group similar episodes
            clusters = self._cluster_episodes()

            # Keep representative episodes from each cluster
            compressed_episodes = []
            for cluster in clusters:
                # Keep most important episode from cluster
                representative = max(cluster, key=lambda x: x.importance)
                compressed_episodes.append(representative)

                # Create summary episode for cluster if large
                if len(cluster) > 5:
                    summary = self._create_summary_episode(cluster)
                    compressed_episodes.append(summary)

            self.episodes = compressed_episodes

        compression_ratio = len(self.episodes) / original_count if original_count > 0 else 1

        logger.info(
            f"Compressed episodes from {original_count} to {len(self.episodes)} "
            f"(ratio: {compression_ratio:.2%})"
        )

        return {
            'original_count': original_count,
            'compressed_count': len(self.episodes),
            'compression_ratio': compression_ratio,
            'method': self.compression_method
        }

    def _cluster_episodes(self) -> List[List[Episode]]:
        """Cluster similar episodes."""
        clusters = []
        clustered = set()

        for episode in self.episodes:
            if episode.episode_id in clustered:
                continue

            cluster = [episode]
            clustered.add(episode.episode_id)

            # Find similar episodes
            for other in self.episodes:
                if other.episode_id not in clustered:
                    similarity = self.learning._calculate_similarity(
                        episode.context,
                        other.context
                    )
                    if similarity > 0.8:
                        cluster.append(other)
                        clustered.add(other.episode_id)

            clusters.append(cluster)

        return clusters

    def _create_summary_episode(self, cluster: List[Episode]) -> Episode:
        """Create summary episode from cluster."""
        # Aggregate context
        typical_context = {}
        all_keys = set()
        for episode in cluster:
            all_keys.update(episode.context.keys())

        for key in all_keys:
            values = [
                episode.context.get(key)
                for episode in cluster
                if key in episode.context
            ]
            if values:
                if isinstance(values[0], (int, float)):
                    typical_context[key] = np.mean(values)
                else:
                    typical_context[key] = Counter(values).most_common(1)[0][0]

        # Aggregate actions (most common sequence)
        action_sequences = [episode.actions for episode in cluster]
        # Simplified: take most common length sequence
        typical_actions = max(action_sequences, key=lambda x: action_sequences.count(x))

        # Aggregate outcomes
        typical_outcomes = {}
        for episode in cluster:
            for key, value in episode.outcomes.items():
                if key not in typical_outcomes:
                    typical_outcomes[key] = []
                typical_outcomes[key].append(value)

        for key in typical_outcomes:
            values = typical_outcomes[key]
            if isinstance(values[0], (int, float)):
                typical_outcomes[key] = np.mean(values)

        # Create summary episode
        summary = Episode(
            agent_id=self.agent_id,
            context=typical_context,
            actions=typical_actions,
            outcomes=typical_outcomes,
            importance=np.mean([e.importance for e in cluster]),
            metadata={
                'type': 'summary',
                'cluster_size': len(cluster),
                'episode_ids': [e.episode_id for e in cluster]
            }
        )

        return summary

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get episodic memory statistics.

        Returns:
            Memory statistics
        """
        return {
            **self.stats,
            'current_episodes': len(self.episodes),
            'patterns_active': len(self.learning.patterns),
            'cases_stored': len(self.learning.case_library),
            'avg_importance': np.mean([e.importance for e in self.episodes]) if self.episodes else 0,
            'learning_stats': self.learning.learning_stats
        }