# -*- coding: utf-8 -*-
"""
Short-Term Memory implementation for GreenLang Agent Foundation.

This module implements working memory, attention buffer, and context window
for immediate agent tasks with FIFO structure and priority override capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime, timedelta
from collections import deque
import asyncio
import redis.asyncio as redis
import json
from enum import Enum
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for memory items."""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    MINIMAL = 0.1


class MemoryItem(BaseModel):
    """Individual memory item in short-term memory."""

    content: Any = Field(..., description="Memory content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation time")
    priority: float = Field(0.5, ge=0.0, le=1.0, description="Priority level 0-1")
    attention_weight: float = Field(1.0, ge=0.0, description="Attention weight")
    access_count: int = Field(0, ge=0, description="Number of times accessed")
    provenance_hash: str = Field("", description="SHA-256 hash for audit trail")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('provenance_hash', always=True)
    def calculate_provenance(cls, v, values):
        """Calculate SHA-256 hash if not provided."""
        if not v and 'content' in values:
            content_str = json.dumps(values['content'], sort_keys=True, default=str)
            timestamp_str = values.get('timestamp', DeterministicClock.now()).isoformat()
            provenance_str = f"{content_str}{timestamp_str}"
            return hashlib.sha256(provenance_str.encode()).hexdigest()
        return v


class WorkingMemory(BaseModel):
    """Working memory for immediate agent tasks."""

    capacity: int = Field(2048, description="Maximum tokens capacity")
    duration_minutes: int = Field(5, description="Retention duration in minutes")
    items: List[MemoryItem] = Field(default_factory=list, description="Memory items")
    structure: str = Field("FIFO with priority override", description="Memory structure type")

    class Config:
        arbitrary_types_allowed = True

    def add(self, content: Any, priority: float = 0.5, metadata: Optional[Dict] = None) -> str:
        """
        Add new item to working memory.

        Args:
            content: Content to store
            priority: Priority level 0-1
            metadata: Optional metadata

        Returns:
            Provenance hash of added item
        """
        item = MemoryItem(
            content=content,
            priority=priority,
            metadata=metadata or {}
        )

        # Remove expired items
        self._cleanup_expired()

        # Check capacity and remove low-priority if needed
        if len(self.items) >= self.capacity:
            if priority > min(i.priority for i in self.items):
                # Remove lowest priority item
                min_item = min(self.items, key=lambda x: x.priority)
                self.items.remove(min_item)
                logger.info(f"Evicted low-priority item: {min_item.provenance_hash[:8]}")
            else:
                logger.warning(f"Working memory full, item rejected: priority {priority}")
                return ""

        self.items.append(item)
        logger.debug(f"Added to working memory: {item.provenance_hash[:8]}")
        return item.provenance_hash

    def retrieve_recent(self, n: int = 10) -> List[MemoryItem]:
        """
        Get n most recent memories.

        Args:
            n: Number of items to retrieve

        Returns:
            List of recent memory items
        """
        self._cleanup_expired()
        sorted_items = sorted(self.items, key=lambda x: x.timestamp, reverse=True)
        for item in sorted_items[:n]:
            item.access_count += 1
        return sorted_items[:n]

    def retrieve_by_priority(self, threshold: float = 0.7) -> List[MemoryItem]:
        """
        Get memories above priority threshold.

        Args:
            threshold: Minimum priority threshold

        Returns:
            List of high-priority items
        """
        self._cleanup_expired()
        filtered = [i for i in self.items if i.priority >= threshold]
        for item in filtered:
            item.access_count += 1
        return filtered

    def _cleanup_expired(self) -> None:
        """Remove expired items based on duration."""
        cutoff = DeterministicClock.now() - timedelta(minutes=self.duration_minutes)
        original_count = len(self.items)
        self.items = [i for i in self.items if i.timestamp > cutoff]
        removed = original_count - len(self.items)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired items from working memory")


class AttentionBuffer(BaseModel):
    """Attention buffer for immediate focus."""

    capacity: int = Field(512, description="Maximum tokens capacity")
    duration_seconds: int = Field(30, description="Retention duration in seconds")
    queue: deque = Field(default_factory=lambda: deque(maxlen=512), description="Attention queue")
    weights: Dict[str, float] = Field(default_factory=dict, description="Attention weights")

    class Config:
        arbitrary_types_allowed = True

    def focus(self, content: Any, weight: float = 1.0) -> str:
        """
        Add content to attention focus.

        Args:
            content: Content to focus on
            weight: Attention weight

        Returns:
            Provenance hash
        """
        item = MemoryItem(
            content=content,
            attention_weight=weight,
            timestamp=DeterministicClock.now()
        )

        # Add to queue (auto-removes oldest if at capacity)
        self.queue.append(item)
        self.weights[item.provenance_hash] = weight

        # Cleanup old weights
        self._cleanup_weights()

        logger.debug(f"Focused attention on: {item.provenance_hash[:8]}")
        return item.provenance_hash

    def get_focused(self, top_k: int = 5) -> List[MemoryItem]:
        """
        Get top-k items by attention weight.

        Args:
            top_k: Number of top items

        Returns:
            List of top attention items
        """
        self._cleanup_expired()
        sorted_items = sorted(
            self.queue,
            key=lambda x: self.weights.get(x.provenance_hash, 0),
            reverse=True
        )
        return list(sorted_items[:top_k])

    def update_attention(self, provenance_hash: str, weight_delta: float) -> None:
        """
        Update attention weight for an item.

        Args:
            provenance_hash: Item identifier
            weight_delta: Change in attention weight
        """
        if provenance_hash in self.weights:
            self.weights[provenance_hash] = max(0, self.weights[provenance_hash] + weight_delta)
            logger.debug(f"Updated attention weight: {provenance_hash[:8]} -> {self.weights[provenance_hash]:.2f}")

    def _cleanup_expired(self) -> None:
        """Remove expired items from queue."""
        cutoff = DeterministicClock.now() - timedelta(seconds=self.duration_seconds)
        self.queue = deque(
            (i for i in self.queue if i.timestamp > cutoff),
            maxlen=self.capacity
        )

    def _cleanup_weights(self) -> None:
        """Remove weights for items no longer in queue."""
        current_hashes = {item.provenance_hash for item in self.queue}
        self.weights = {k: v for k, v in self.weights.items() if k in current_hashes}


class ShortTermMemory:
    """
    Short-term memory system combining working memory, attention buffer, and context window.

    This class orchestrates immediate memory needs for agents with Redis backing
    for persistence across process restarts.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", agent_id: str = "default"):
        """
        Initialize short-term memory.

        Args:
            redis_url: Redis connection URL
            agent_id: Agent identifier for namespacing
        """
        self.redis_url = redis_url
        self.agent_id = agent_id
        self.working_memory = WorkingMemory()
        self.attention_buffer = AttentionBuffer()
        self.context_window: List[MemoryItem] = []  # Model-dependent sliding window
        self.redis_client: Optional[redis.Redis] = None

        # Memory optimization parameters
        self.compression_ratio = 0.3
        self.priority_threshold = 0.7
        self.recency_weight = 0.4
        self.relevance_weight = 0.6

    async def initialize(self) -> None:
        """Initialize Redis connection and restore state."""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self._restore_from_redis()
            logger.info(f"Short-term memory initialized for agent: {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            # Continue without Redis (in-memory only)

    async def add_memory(
        self,
        content: Any,
        priority: float = 0.5,
        attention_weight: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add new information to short-term memory.

        Args:
            content: Memory content
            priority: Priority level 0-1
            attention_weight: Initial attention weight
            metadata: Optional metadata

        Returns:
            Provenance hash
        """
        start_time = DeterministicClock.now()

        # Add to working memory
        provenance_hash = self.working_memory.add(content, priority, metadata)

        if provenance_hash:
            # Add to attention buffer if high priority
            if priority >= self.priority_threshold:
                self.attention_buffer.focus(content, attention_weight)

            # Update context window
            item = MemoryItem(
                content=content,
                priority=priority,
                attention_weight=attention_weight,
                metadata=metadata or {},
                provenance_hash=provenance_hash
            )
            self.context_window.append(item)

            # Persist to Redis
            if self.redis_client:
                await self._persist_to_redis(item)

            processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
            logger.info(f"Added memory {provenance_hash[:8]} in {processing_time:.2f}ms")

        return provenance_hash

    async def retrieve_recent(self, n: int = 10) -> List[MemoryItem]:
        """
        Get recent memories with recency and relevance weighting.

        Args:
            n: Number of items

        Returns:
            List of weighted recent memories
        """
        items = self.working_memory.retrieve_recent(n)

        # Apply recency and relevance weighting
        now = DeterministicClock.now()
        for item in items:
            age_minutes = (now - item.timestamp).total_seconds() / 60
            recency_score = max(0, 1 - (age_minutes / self.working_memory.duration_minutes))

            relevance_score = item.priority * item.attention_weight

            item.metadata['weighted_score'] = (
                self.recency_weight * recency_score +
                self.relevance_weight * relevance_score
            )

        # Sort by weighted score
        items.sort(key=lambda x: x.metadata.get('weighted_score', 0), reverse=True)
        return items[:n]

    async def update_attention(self, provenance_hash: str, weight_delta: float) -> None:
        """
        Adjust attention weights for a memory.

        Args:
            provenance_hash: Memory identifier
            weight_delta: Change in attention weight
        """
        self.attention_buffer.update_attention(provenance_hash, weight_delta)

        # Update in working memory if present
        for item in self.working_memory.items:
            if item.provenance_hash == provenance_hash:
                item.attention_weight = max(0, item.attention_weight + weight_delta)
                break

    async def consolidate(self) -> List[MemoryItem]:
        """
        Transfer important memories to long-term storage.

        Returns:
            List of memories ready for long-term storage
        """
        candidates = []

        # Select high-priority items
        high_priority = self.working_memory.retrieve_by_priority(self.priority_threshold)

        # Select frequently accessed items
        frequent = [i for i in self.working_memory.items if i.access_count > 5]

        # Combine and deduplicate
        seen_hashes = set()
        for item in high_priority + frequent:
            if item.provenance_hash not in seen_hashes:
                candidates.append(item)
                seen_hashes.add(item.provenance_hash)

        logger.info(f"Consolidated {len(candidates)} memories for long-term storage")
        return candidates

    async def forget(self, threshold: float = 0.3) -> int:
        """
        Remove low-priority items below threshold.

        Args:
            threshold: Priority threshold for removal

        Returns:
            Number of items forgotten
        """
        original_count = len(self.working_memory.items)
        self.working_memory.items = [
            i for i in self.working_memory.items
            if i.priority >= threshold
        ]

        forgotten = original_count - len(self.working_memory.items)
        if forgotten > 0:
            logger.info(f"Forgot {forgotten} low-priority memories")

            # Clean Redis
            if self.redis_client:
                await self._cleanup_redis()

        return forgotten

    async def compress(self) -> Dict[str, Any]:
        """
        Compress memory to target ratio.

        Returns:
            Compression statistics
        """
        original_size = len(self.working_memory.items)
        target_size = int(original_size * self.compression_ratio)

        if original_size <= target_size:
            return {"compressed": False, "original": original_size, "final": original_size}

        # Sort by weighted importance
        scored_items = []
        for item in self.working_memory.items:
            score = (
                item.priority * 0.4 +
                item.attention_weight * 0.3 +
                min(item.access_count / 10, 1.0) * 0.3
            )
            scored_items.append((score, item))

        # Keep top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        self.working_memory.items = [item for _, item in scored_items[:target_size]]

        logger.info(f"Compressed memory from {original_size} to {len(self.working_memory.items)} items")

        return {
            "compressed": True,
            "original": original_size,
            "final": len(self.working_memory.items),
            "ratio": len(self.working_memory.items) / original_size
        }

    async def _persist_to_redis(self, item: MemoryItem) -> None:
        """Persist memory item to Redis."""
        if not self.redis_client:
            return

        try:
            key = f"stm:{self.agent_id}:{item.provenance_hash}"
            value = item.json()
            ttl = self.working_memory.duration_minutes * 60

            await self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Failed to persist to Redis: {e}")

    async def _restore_from_redis(self) -> None:
        """Restore memory state from Redis."""
        if not self.redis_client:
            return

        try:
            pattern = f"stm:{self.agent_id}:*"
            keys = await self.redis_client.keys(pattern)

            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    item = MemoryItem.parse_raw(value)
                    # Only restore if not expired
                    if item.timestamp > DeterministicClock.now() - timedelta(minutes=self.working_memory.duration_minutes):
                        self.working_memory.items.append(item)

            logger.info(f"Restored {len(self.working_memory.items)} items from Redis")
        except Exception as e:
            logger.error(f"Failed to restore from Redis: {e}")

    async def _cleanup_redis(self) -> None:
        """Clean up Redis entries for forgotten items."""
        if not self.redis_client:
            return

        try:
            current_hashes = {i.provenance_hash for i in self.working_memory.items}
            pattern = f"stm:{self.agent_id}:*"
            keys = await self.redis_client.keys(pattern)

            for key in keys:
                hash_part = key.decode().split(":")[-1]
                if hash_part not in current_hashes:
                    await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to cleanup Redis: {e}")

    async def close(self) -> None:
        """Close connections and cleanup."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Short-term memory connections closed")