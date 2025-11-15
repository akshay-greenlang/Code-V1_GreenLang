"""
Memory Manager implementation for GreenLang Agent Foundation.

This module orchestrates memory consolidation, pruning, compression,
and quality management across all memory systems.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from collections import Counter
import numpy as np

# Import memory systems
from .short_term_memory import ShortTermMemory, MemoryItem, Priority
from .long_term_memory import LongTermMemory, StorageTier, RetrievalStrategy
from .episodic_memory import EpisodicMemory, Episode, Pattern
from .semantic_memory import SemanticMemory, KnowledgeType

# Import infrastructure managers (NEW)
from ..database.postgres_manager import PostgresManager
from ..cache.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies."""
    COMPRESSION = "compression"          # Summarization and abstraction
    INTEGRATION = "integration"          # Merge related memories
    GENERALIZATION = "generalization"    # Extract patterns from instances


class PruningPolicy(Enum):
    """Memory pruning policies."""
    AGE_BASED = "age_based"              # Based on age
    IMPORTANCE_BASED = "importance_based"  # Based on importance
    REDUNDANCY_BASED = "redundancy_based"  # Remove duplicates
    CAPACITY_BASED = "capacity_based"     # Based on storage limits


class QualityMetric(Enum):
    """Memory quality metrics."""
    ACCURACY = "accuracy"          # Factual correctness
    RELEVANCE = "relevance"        # Contextual applicability
    COMPLETENESS = "completeness"  # Information coverage
    RECENCY = "recency"           # Time since last access
    FREQUENCY = "frequency"        # Access count


class MemoryMetrics(BaseModel):
    """Metrics for memory system monitoring."""

    total_memories: int = Field(0, description="Total memories across all systems")
    short_term_count: int = Field(0, description="Short-term memory items")
    long_term_count: int = Field(0, description="Long-term memory items")
    episodic_count: int = Field(0, description="Episodic memory items")
    semantic_count: int = Field(0, description="Semantic memory items")
    compression_ratio: float = Field(1.0, description="Overall compression ratio")
    quality_score: float = Field(0.0, description="Average quality score")
    storage_distribution: Dict[str, int] = Field(default_factory=dict, description="Storage tier distribution")
    access_patterns: Dict[str, float] = Field(default_factory=dict, description="Access frequency patterns")
    last_consolidation: Optional[datetime] = Field(None, description="Last consolidation time")
    last_pruning: Optional[datetime] = Field(None, description="Last pruning time")


class ConsolidationConfig(BaseModel):
    """Configuration for memory consolidation."""

    compression_ratio: float = Field(0.2, description="Target compression ratio")
    similarity_threshold: float = Field(0.9, description="Similarity threshold for merging")
    min_instances: int = Field(5, description="Minimum instances for generalization")
    confidence_threshold: float = Field(0.85, description="Minimum confidence for patterns")
    batch_size: int = Field(1000, description="Batch size for processing")


class PruningConfig(BaseModel):
    """Configuration for memory pruning."""

    hot_to_warm_days: int = Field(1, description="Days before hot to warm migration")
    warm_to_cold_days: int = Field(30, description="Days before warm to cold migration")
    cold_to_archive_days: int = Field(90, description="Days before cold to archive migration")
    delete_after_years: int = Field(7, description="Years before deletion")
    capacity_threshold: float = Field(0.8, description="Capacity threshold for pruning")
    duplicate_threshold: float = Field(0.95, description="Similarity threshold for duplicates")


class MemoryManager:
    """
    Central memory management system for GreenLang Agent Foundation.

    Orchestrates consolidation, pruning, compression, and quality management
    across short-term, long-term, episodic, and semantic memory systems.
    """

    def __init__(
        self,
        agent_id: str = "default",
        redis_manager: Optional[RedisManager] = None,
        postgres_manager: Optional[PostgresManager] = None,
        s3_bucket: str = "greenlang-ltm",
        # Backward compatibility (deprecated)
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None
    ):
        """
        Initialize memory manager.

        Args:
            agent_id: Agent identifier
            redis_manager: Centralized Redis manager (recommended)
            postgres_manager: Centralized PostgreSQL manager (recommended)
            s3_bucket: S3 bucket name
            redis_url: Redis connection URL (deprecated, use redis_manager)
            postgres_url: PostgreSQL connection URL (deprecated, use postgres_manager)
        """
        self.agent_id = agent_id
        self.redis_manager = redis_manager
        self.postgres_manager = postgres_manager

        # Initialize memory systems
        # Note: Pass managers to memory systems (they need to be updated to accept them)
        # For now, maintain backward compatibility with URL-based initialization
        if redis_url:
            logger.warning(
                "Using deprecated redis_url parameter. "
                "Please pass RedisManager instance via redis_manager parameter."
            )
            self.short_term = ShortTermMemory(redis_url, agent_id)
            self.long_term = LongTermMemory(redis_url, postgres_url or "", s3_bucket)
        else:
            # Use centralized managers (NEW PATH)
            self.short_term = ShortTermMemory(
                redis_manager=redis_manager,
                agent_id=agent_id
            )
            self.long_term = LongTermMemory(
                redis_manager=redis_manager,
                postgres_manager=postgres_manager,
                s3_bucket=s3_bucket
            )

        self.episodic = EpisodicMemory(agent_id)
        self.semantic = SemanticMemory(agent_id)

        # Configurations
        self.consolidation_config = ConsolidationConfig()
        self.pruning_config = PruningConfig()

        # Metrics
        self.metrics = MemoryMetrics()

        # Quality tracking
        self.quality_scores: Dict[str, float] = {}

        # Consolidation and pruning tasks
        self.consolidation_task: Optional[asyncio.Task] = None
        self.pruning_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "consolidations_performed": 0,
            "memories_compressed": 0,
            "memories_integrated": 0,
            "patterns_generalized": 0,
            "memories_pruned": 0,
            "quality_checks": 0
        }

    async def initialize(self) -> None:
        """Initialize all memory systems and start background tasks."""
        await self.short_term.initialize()
        await self.long_term.initialize()

        # Start background tasks
        self.consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.pruning_task = asyncio.create_task(self._pruning_loop())

        logger.info(f"Memory manager initialized for agent: {self.agent_id}")

    async def store_memory(
        self,
        content: Any,
        memory_type: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store memory in appropriate system.

        Args:
            content: Memory content
            memory_type: Type of memory (short, long, episodic, semantic)
            importance: Importance score 0-1
            metadata: Optional metadata

        Returns:
            Memory ID or provenance hash
        """
        start_time = datetime.now()
        memory_id = ""

        try:
            if memory_type == "short":
                memory_id = await self.short_term.add_memory(
                    content, importance, metadata=metadata
                )

            elif memory_type == "long":
                memory_id = await self.long_term.store(
                    content, self.agent_id, memory_type,
                    importance=importance, metadata=metadata
                )

            elif memory_type == "episodic":
                if isinstance(content, dict):
                    memory_id = await self.episodic.record_episode(
                        content.get('context', {}),
                        content.get('actions', []),
                        content.get('outcomes', {}),
                        metadata=metadata
                    )

            elif memory_type == "semantic":
                if isinstance(content, dict):
                    if content.get('type') == 'fact':
                        memory_id = await self.semantic.store_fact(
                            content.get('content'), metadata
                        )
                    elif content.get('type') == 'concept':
                        memory_id = await self.semantic.store_concept(
                            content.get('name', ''),
                            content.get('definition', ''),
                            content.get('properties')
                        )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Stored {memory_type} memory: {memory_id[:8]} ({processing_time:.2f}ms)")

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

        await self._update_metrics()
        return memory_id

    async def retrieve_memory(
        self,
        query: Dict[str, Any],
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories across systems.

        Args:
            query: Query parameters
            memory_types: Types to search (default: all)
            limit: Maximum results

        Returns:
            List of memory results
        """
        start_time = datetime.now()
        results = []

        memory_types = memory_types or ["short", "long", "episodic", "semantic"]

        try:
            # Search short-term memory
            if "short" in memory_types:
                short_results = await self.short_term.retrieve_recent(limit)
                for item in short_results:
                    results.append({
                        'type': 'short_term',
                        'content': item.content,
                        'timestamp': item.timestamp,
                        'importance': item.priority,
                        'metadata': item.metadata
                    })

            # Search long-term memory
            if "long" in memory_types:
                strategy = RetrievalStrategy(query.get('strategy', 'temporal'))
                long_results = await self.long_term.retrieve(
                    strategy=strategy,
                    criteria=query.get('criteria', {}),
                    limit=limit
                )
                for record in long_results:
                    results.append({
                        'type': 'long_term',
                        'content': record.content,
                        'timestamp': record.timestamp,
                        'importance': record.importance,
                        'tier': record.tier.value,
                        'metadata': record.metadata
                    })

            # Search episodic memory
            if "episodic" in memory_types and 'context' in query:
                episodic_results = await self.episodic.retrieve_similar_episodes(
                    query['context'], limit
                )
                for episode in episodic_results:
                    results.append({
                        'type': 'episodic',
                        'episode_id': episode.episode_id,
                        'context': episode.context,
                        'actions': episode.actions,
                        'outcomes': episode.outcomes,
                        'importance': episode.importance
                    })

            # Search semantic memory
            if "semantic" in memory_types:
                knowledge_type = KnowledgeType(query.get('knowledge_type', 'fact'))
                semantic_results = await self.semantic.query(
                    knowledge_type,
                    pattern=query.get('pattern'),
                    limit=limit
                )
                for item in semantic_results:
                    results.append({
                        'type': 'semantic',
                        'knowledge_type': item.knowledge_type.value,
                        'content': item.content,
                        'confidence': item.confidence,
                        'metadata': item.metadata
                    })

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Retrieved {len(results)} memories in {processing_time:.2f}ms")

        return results[:limit]

    async def consolidate_memories(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.COMPRESSION
    ) -> Dict[str, Any]:
        """
        Consolidate memories using specified strategy.

        Args:
            strategy: Consolidation strategy

        Returns:
            Consolidation statistics
        """
        start_time = datetime.now()
        stats = {
            'strategy': strategy.value,
            'memories_processed': 0,
            'memories_consolidated': 0,
            'compression_achieved': 0.0
        }

        try:
            if strategy == ConsolidationStrategy.COMPRESSION:
                stats.update(await self._consolidate_by_compression())

            elif strategy == ConsolidationStrategy.INTEGRATION:
                stats.update(await self._consolidate_by_integration())

            elif strategy == ConsolidationStrategy.GENERALIZATION:
                stats.update(await self._consolidate_by_generalization())

            self.metrics.last_consolidation = datetime.now()
            self.stats["consolidations_performed"] += 1

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Consolidation complete: {stats} ({processing_time:.1f}s)")

        return stats

    async def _consolidate_by_compression(self) -> Dict[str, Any]:
        """Consolidate by compression and summarization."""
        stats = {'memories_processed': 0, 'memories_consolidated': 0}

        # Compress short-term memory
        stm_compression = await self.short_term.compress()
        stats['stm_compression'] = stm_compression

        # Compress episodic memory
        episodic_compression = await self.episodic.compress_episodes()
        stats['episodic_compression'] = episodic_compression

        # Move compressed memories to long-term
        consolidated = await self.short_term.consolidate()
        for item in consolidated:
            await self.long_term.store(
                item.content,
                self.agent_id,
                "consolidated",
                importance=item.priority,
                metadata=item.metadata
            )
            stats['memories_consolidated'] += 1

        self.stats["memories_compressed"] += stats['memories_consolidated']
        return stats

    async def _consolidate_by_integration(self) -> Dict[str, Any]:
        """Consolidate by merging related memories."""
        stats = {'memories_processed': 0, 'memories_merged': 0}

        # Find similar memories in long-term storage
        all_memories = await self.long_term.retrieve(limit=1000)

        # Group by similarity
        clusters = []
        processed = set()

        for i, mem1 in enumerate(all_memories):
            if mem1.memory_id in processed:
                continue

            cluster = [mem1]
            processed.add(mem1.memory_id)

            for mem2 in all_memories[i + 1:]:
                if mem2.memory_id not in processed:
                    similarity = self._calculate_memory_similarity(mem1, mem2)
                    if similarity >= self.consolidation_config.similarity_threshold:
                        cluster.append(mem2)
                        processed.add(mem2.memory_id)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Merge clusters
        for cluster in clusters:
            merged = await self._merge_memories(cluster)
            if merged:
                stats['memories_merged'] += len(cluster) - 1

        stats['memories_processed'] = len(all_memories)
        self.stats["memories_integrated"] += stats['memories_merged']
        return stats

    async def _consolidate_by_generalization(self) -> Dict[str, Any]:
        """Consolidate by extracting patterns and generalizing."""
        stats = {'patterns_extracted': 0, 'instances_generalized': 0}

        # Extract patterns from episodic memory
        patterns = await self.episodic.extract_patterns()
        stats['patterns_extracted'] = len(patterns)

        # Convert patterns to semantic knowledge
        for pattern in patterns:
            if pattern.frequency >= self.consolidation_config.min_instances:
                # Store as procedure in semantic memory
                await self.semantic.store_procedure(
                    name=f"pattern_{pattern.pattern_id[:8]}",
                    steps=[{'action': action} for action in pattern.action_sequence],
                    expected_outcomes=pattern.expected_outcomes,
                    metadata={
                        'frequency': pattern.frequency,
                        'confidence': pattern.confidence,
                        'source': 'episodic_generalization'
                    }
                )
                stats['instances_generalized'] += pattern.frequency

        # Consolidate episodic to semantic
        semantic_memories = await self.episodic.consolidate_to_semantic()
        for memory in semantic_memories:
            if memory['type'] == 'fact':
                await self.semantic.store_fact(memory['content'])
            elif memory['type'] == 'procedure':
                await self.semantic.store_procedure(
                    name=memory['content'].get('name', 'unnamed'),
                    steps=memory['content'].get('action_sequence', [])
                )

        self.stats["patterns_generalized"] += stats['patterns_extracted']
        return stats

    def _calculate_memory_similarity(self, mem1: Any, mem2: Any) -> float:
        """Calculate similarity between two memories."""
        # Simplified similarity based on content comparison
        if hasattr(mem1, 'content') and hasattr(mem2, 'content'):
            content1 = json.dumps(mem1.content, sort_keys=True, default=str)
            content2 = json.dumps(mem2.content, sort_keys=True, default=str)

            # Simple character-based similarity
            if content1 == content2:
                return 1.0

            # Calculate Jaccard similarity on tokens
            tokens1 = set(content1.split())
            tokens2 = set(content2.split())

            if tokens1 or tokens2:
                return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

        return 0.0

    async def _merge_memories(self, memories: List[Any]) -> Optional[Any]:
        """Merge similar memories into one."""
        if not memories:
            return None

        # Use the most important memory as base
        base = max(memories, key=lambda x: getattr(x, 'importance', 0))

        # Merge metadata
        merged_metadata = {}
        for mem in memories:
            if hasattr(mem, 'metadata'):
                merged_metadata.update(mem.metadata)

        # Update base memory
        if hasattr(base, 'metadata'):
            base.metadata = merged_metadata
            base.access_count = sum(getattr(m, 'access_count', 0) for m in memories)

        # Delete other memories
        for mem in memories:
            if mem != base and hasattr(mem, 'memory_id'):
                await self.long_term.migrate_tier(mem.memory_id, StorageTier.ARCHIVE)

        return base

    async def prune_memories(
        self,
        policy: PruningPolicy = PruningPolicy.AGE_BASED
    ) -> Dict[str, int]:
        """
        Prune memories based on policy.

        Args:
            policy: Pruning policy

        Returns:
            Pruning statistics
        """
        start_time = datetime.now()
        stats = {
            'policy': policy.value,
            'pruned': 0,
            'migrated': 0
        }

        try:
            if policy == PruningPolicy.AGE_BASED:
                stats.update(await self._prune_by_age())

            elif policy == PruningPolicy.IMPORTANCE_BASED:
                stats.update(await self._prune_by_importance())

            elif policy == PruningPolicy.REDUNDANCY_BASED:
                stats.update(await self._prune_by_redundancy())

            elif policy == PruningPolicy.CAPACITY_BASED:
                stats.update(await self._prune_by_capacity())

            self.metrics.last_pruning = datetime.now()
            self.stats["memories_pruned"] += stats['pruned']

        except Exception as e:
            logger.error(f"Pruning failed: {e}")

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pruning complete: {stats} ({processing_time:.1f}s)")

        return stats

    async def _prune_by_age(self) -> Dict[str, int]:
        """Prune memories based on age."""
        stats = await self.long_term.age_out_memories()

        # Forget old short-term memories
        forgotten = await self.short_term.forget(threshold=0.3)
        stats['short_term_forgotten'] = forgotten

        return stats

    async def _prune_by_importance(self) -> Dict[str, int]:
        """Prune memories based on importance."""
        stats = {'pruned': 0}

        # Define importance thresholds by tier
        thresholds = {
            'critical': None,  # Never prune
            'high': 365,       # Keep 1 year minimum
            'medium': 90,      # Keep 90 days minimum
            'low': 7          # Prune after 7 days
        }

        memories = await self.long_term.retrieve(limit=10000)
        now = datetime.now()

        for memory in memories:
            age_days = (now - memory.timestamp).days

            if memory.importance < 0.3:  # Low importance
                if age_days > thresholds['low']:
                    await self.long_term.migrate_tier(
                        memory.memory_id,
                        StorageTier.ARCHIVE
                    )
                    stats['pruned'] += 1

            elif memory.importance < 0.7:  # Medium importance
                if age_days > thresholds['medium']:
                    await self.long_term.migrate_tier(
                        memory.memory_id,
                        StorageTier.COLD
                    )
                    stats['pruned'] += 1

        return stats

    async def _prune_by_redundancy(self) -> Dict[str, int]:
        """Prune redundant/duplicate memories."""
        stats = {'duplicates_removed': 0}

        # Consolidate semantic knowledge to remove duplicates
        semantic_stats = await self.semantic.consolidate_knowledge()
        stats.update(semantic_stats)

        return stats

    async def _prune_by_capacity(self) -> Dict[str, int]:
        """Prune when approaching capacity limits."""
        stats = {'pruned': 0}

        # Check current capacity (simplified - would check actual storage)
        metrics = await self._update_metrics()

        if metrics.total_memories > 100000:  # Example threshold
            # Remove lowest importance memories
            memories = await self.long_term.retrieve(limit=10000)
            memories.sort(key=lambda x: x.importance)

            # Prune bottom 20%
            to_prune = int(len(memories) * 0.2)
            for memory in memories[:to_prune]:
                await self.long_term.migrate_tier(
                    memory.memory_id,
                    StorageTier.ARCHIVE
                )
                stats['pruned'] += 1

        return stats

    async def assess_quality(self) -> Dict[str, float]:
        """
        Assess memory quality across all systems.

        Returns:
            Quality scores by metric
        """
        quality_scores = {
            QualityMetric.ACCURACY.value: 0.0,
            QualityMetric.RELEVANCE.value: 0.0,
            QualityMetric.COMPLETENESS.value: 0.0,
            QualityMetric.RECENCY.value: 0.0,
            QualityMetric.FREQUENCY.value: 0.0
        }

        # Sample memories for quality assessment
        samples = await self.long_term.retrieve(limit=100)

        if not samples:
            return quality_scores

        now = datetime.now()

        for memory in samples:
            # Accuracy (based on confidence if available)
            if hasattr(memory, 'confidence'):
                quality_scores[QualityMetric.ACCURACY.value] += memory.confidence

            # Relevance (based on importance)
            quality_scores[QualityMetric.RELEVANCE.value] += memory.importance

            # Completeness (based on metadata richness)
            metadata_fields = len(memory.metadata) if memory.metadata else 0
            quality_scores[QualityMetric.COMPLETENESS.value] += min(metadata_fields / 10, 1.0)

            # Recency (based on last access)
            age_days = (now - memory.last_accessed).days
            recency_score = max(0, 1 - (age_days / 365))
            quality_scores[QualityMetric.RECENCY.value] += recency_score

            # Frequency (based on access count)
            freq_score = min(memory.access_count / 100, 1.0)
            quality_scores[QualityMetric.FREQUENCY.value] += freq_score

        # Normalize scores
        for metric in quality_scores:
            quality_scores[metric] /= len(samples)

        # Update metrics
        self.quality_scores = quality_scores
        self.metrics.quality_score = np.mean(list(quality_scores.values()))
        self.stats["quality_checks"] += 1

        logger.info(f"Quality assessment: {quality_scores}")
        return quality_scores

    async def _update_metrics(self) -> MemoryMetrics:
        """Update system metrics."""
        # Count memories
        stm_count = len(self.short_term.working_memory.items)
        ltm_stats = await self.long_term.get_statistics()
        episodic_stats = self.episodic.get_statistics()
        semantic_stats = self.semantic.get_statistics()

        self.metrics.short_term_count = stm_count
        self.metrics.long_term_count = ltm_stats.get('total_memories', 0)
        self.metrics.episodic_count = episodic_stats.get('current_episodes', 0)
        self.metrics.semantic_count = (
            semantic_stats.get('total_facts', 0) +
            semantic_stats.get('total_concepts', 0) +
            semantic_stats.get('total_procedures', 0) +
            semantic_stats.get('total_relationships', 0)
        )

        self.metrics.total_memories = (
            self.metrics.short_term_count +
            self.metrics.long_term_count +
            self.metrics.episodic_count +
            self.metrics.semantic_count
        )

        # Storage distribution
        self.metrics.storage_distribution = ltm_stats.get('tier_distribution', {})

        return self.metrics

    async def _consolidation_loop(self) -> None:
        """Background task for periodic consolidation."""
        while True:
            try:
                # Wait 6 hours
                await asyncio.sleep(6 * 3600)

                # Perform consolidation
                await self.consolidate_memories(ConsolidationStrategy.COMPRESSION)
                await self.consolidate_memories(ConsolidationStrategy.INTEGRATION)

                # Extract patterns periodically
                if self.stats["consolidations_performed"] % 4 == 0:
                    await self.consolidate_memories(ConsolidationStrategy.GENERALIZATION)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation loop error: {e}")

    async def _pruning_loop(self) -> None:
        """Background task for periodic pruning."""
        while True:
            try:
                # Wait 24 hours
                await asyncio.sleep(24 * 3600)

                # Perform age-based pruning daily
                await self.prune_memories(PruningPolicy.AGE_BASED)

                # Check capacity weekly
                if datetime.now().weekday() == 0:
                    await self.prune_memories(PruningPolicy.CAPACITY_BASED)

                # Remove redundancy monthly
                if datetime.now().day == 1:
                    await self.prune_memories(PruningPolicy.REDUNDANCY_BASED)

                # Assess quality
                await self.assess_quality()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pruning loop error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system statistics.

        Returns:
            System statistics
        """
        return {
            'metrics': self.metrics.dict(),
            'quality_scores': self.quality_scores,
            'stats': self.stats,
            'short_term': {
                'items': self.metrics.short_term_count,
                'compression_ratio': self.short_term.compression_ratio
            },
            'long_term': self.long_term.get_statistics() if hasattr(self.long_term, 'get_statistics') else {},
            'episodic': self.episodic.get_statistics(),
            'semantic': self.semantic.get_statistics()
        }

    async def close(self) -> None:
        """Shutdown memory manager and cleanup resources."""
        # Cancel background tasks
        if self.consolidation_task:
            self.consolidation_task.cancel()
        if self.pruning_task:
            self.pruning_task.cancel()

        # Close connections
        await self.short_term.close()
        await self.long_term.close()

        logger.info("Memory manager shutdown complete")