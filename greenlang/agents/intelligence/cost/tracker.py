# -*- coding: utf-8 -*-
"""
Cost Tracker Implementation

Per-request cost tracking with attempt-level granularity:
- Track cost/tokens for each request by request_id
- Record attempt counts (including JSON repair retries)
- Aggregate costs across multiple requests
- Query cost breakdown by request_id

Usage:
    tracker = CostTracker()

    # Record cost for a request
    tracker.record(
        request_id="req_123",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0015,
        attempt=1
    )

    # Get cost breakdown
    cost = tracker.get("req_123")
    print(f"Total cost: ${cost.total_cost_usd:.4f}")
    print(f"Attempts: {cost.attempt_count}")
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class AttemptCost:
    """
    Cost for a single attempt

    Attributes:
        attempt_number: Attempt number (0-indexed)
        input_tokens: Input token count
        output_tokens: Output token count
        cost_usd: Cost in USD
        embedding_tokens: Embedding tokens generated (RAG)
        embedding_cost_usd: Embedding generation cost (RAG)
        vector_db_queries: Number of vector DB queries (RAG)
        vector_db_cost_usd: Vector DB operation cost (RAG)
        timestamp: When this attempt occurred
    """
    attempt_number: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    embedding_tokens: int = 0
    embedding_cost_usd: float = 0.0
    vector_db_queries: int = 0
    vector_db_cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        """Total tokens for this attempt (LLM + embeddings)"""
        return self.input_tokens + self.output_tokens + self.embedding_tokens

    @property
    def total_cost_usd(self) -> float:
        """Total cost for this attempt (LLM + embeddings + vector DB)"""
        return self.cost_usd + self.embedding_cost_usd + self.vector_db_cost_usd


@dataclass
class RequestCost:
    """
    Cost breakdown for a request

    Aggregates costs across all attempts (including retries).

    Attributes:
        request_id: Request identifier
        attempts: List of attempt costs
        created_at: When request started
    """
    request_id: str
    attempts: List[AttemptCost] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def attempt_count(self) -> int:
        """Number of attempts (including retries)"""
        return len(self.attempts)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all attempts"""
        return sum(a.input_tokens for a in self.attempts)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all attempts"""
        return sum(a.output_tokens for a in self.attempts)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all attempts"""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD across all attempts (LLM + embeddings + vector DB)"""
        return sum(a.total_cost_usd for a in self.attempts)

    @property
    def total_embedding_tokens(self) -> int:
        """Total embedding tokens across all attempts"""
        return sum(a.embedding_tokens for a in self.attempts)

    @property
    def total_embedding_cost_usd(self) -> float:
        """Total embedding cost in USD across all attempts"""
        return sum(a.embedding_cost_usd for a in self.attempts)

    @property
    def total_vector_db_queries(self) -> int:
        """Total vector DB queries across all attempts"""
        return sum(a.vector_db_queries for a in self.attempts)

    @property
    def total_vector_db_cost_usd(self) -> float:
        """Total vector DB cost in USD across all attempts"""
        return sum(a.vector_db_cost_usd for a in self.attempts)

    def add_attempt(
        self,
        attempt_number: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        embedding_tokens: int = 0,
        embedding_cost_usd: float = 0.0,
        vector_db_queries: int = 0,
        vector_db_cost_usd: float = 0.0
    ) -> None:
        """
        Add an attempt to this request

        Args:
            attempt_number: Attempt number (0-indexed)
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Cost in USD
            embedding_tokens: Embedding tokens generated (RAG)
            embedding_cost_usd: Embedding generation cost (RAG)
            vector_db_queries: Number of vector DB queries (RAG)
            vector_db_cost_usd: Vector DB operation cost (RAG)
        """
        self.attempts.append(AttemptCost(
            attempt_number=attempt_number,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            embedding_tokens=embedding_tokens,
            embedding_cost_usd=embedding_cost_usd,
            vector_db_queries=vector_db_queries,
            vector_db_cost_usd=vector_db_cost_usd
        ))

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "request_id": self.request_id,
            "attempt_count": self.attempt_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_embedding_tokens": self.total_embedding_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_embedding_cost_usd": self.total_embedding_cost_usd,
            "total_vector_db_queries": self.total_vector_db_queries,
            "total_vector_db_cost_usd": self.total_vector_db_cost_usd,
            "created_at": self.created_at.isoformat(),
            "attempts": [
                {
                    "attempt": a.attempt_number,
                    "input_tokens": a.input_tokens,
                    "output_tokens": a.output_tokens,
                    "embedding_tokens": a.embedding_tokens,
                    "total_tokens": a.total_tokens,
                    "cost_usd": a.cost_usd,
                    "embedding_cost_usd": a.embedding_cost_usd,
                    "vector_db_queries": a.vector_db_queries,
                    "vector_db_cost_usd": a.vector_db_cost_usd,
                    "total_cost_usd": a.total_cost_usd,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in self.attempts
            ]
        }


class CostTracker:
    """
    Global cost tracker for all requests

    Thread-safe tracker for recording and querying costs.

    Usage:
        # Global instance
        tracker = CostTracker()

        # Record cost
        tracker.record(
            request_id="req_123",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0015,
            attempt=0
        )

        # Get breakdown
        cost = tracker.get("req_123")
        print(f"Cost: ${cost.total_cost_usd:.4f}")
        print(f"Attempts: {cost.attempt_count}")

        # Get all requests
        all_costs = tracker.get_all()
        total = sum(c.total_cost_usd for c in all_costs)
    """

    def __init__(self):
        """Initialize cost tracker"""
        self._costs: Dict[str, RequestCost] = {}
        self._lock = threading.Lock()

    def record(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        attempt: int = 0,
        embedding_tokens: int = 0,
        embedding_cost_usd: float = 0.0,
        vector_db_queries: int = 0,
        vector_db_cost_usd: float = 0.0
    ) -> None:
        """
        Record cost for a request attempt

        Args:
            request_id: Request identifier
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Cost in USD
            attempt: Attempt number (0-indexed)
            embedding_tokens: Embedding tokens generated (RAG)
            embedding_cost_usd: Embedding generation cost (RAG)
            vector_db_queries: Number of vector DB queries (RAG)
            vector_db_cost_usd: Vector DB operation cost (RAG)

        Example:
            # LLM-only request
            tracker.record("req_1", 100, 50, 0.0015, attempt=0)

            # RAG-augmented request
            tracker.record(
                "req_2",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0015,
                embedding_tokens=256,
                embedding_cost_usd=0.0001,
                vector_db_queries=5,
                vector_db_cost_usd=0.00005,
                attempt=0
            )
        """
        with self._lock:
            if request_id not in self._costs:
                self._costs[request_id] = RequestCost(request_id=request_id)

            self._costs[request_id].add_attempt(
                attempt_number=attempt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                embedding_tokens=embedding_tokens,
                embedding_cost_usd=embedding_cost_usd,
                vector_db_queries=vector_db_queries,
                vector_db_cost_usd=vector_db_cost_usd
            )

    def get(self, request_id: str) -> Optional[RequestCost]:
        """
        Get cost breakdown for a request

        Args:
            request_id: Request identifier

        Returns:
            RequestCost with breakdown, or None if not found

        Example:
            cost = tracker.get("req_123")
            if cost:
                print(f"Total: ${cost.total_cost_usd:.4f}")
                print(f"Attempts: {cost.attempt_count}")
                for attempt in cost.attempts:
                    print(f"  Attempt {attempt.attempt_number}: ${attempt.cost_usd:.6f}")
        """
        with self._lock:
            return self._costs.get(request_id)

    def get_all(self) -> List[RequestCost]:
        """
        Get all request costs

        Returns:
            List of all RequestCost objects

        Example:
            all_costs = tracker.get_all()
            total = sum(c.total_cost_usd for c in all_costs)
            print(f"Total spent: ${total:.2f}")
        """
        with self._lock:
            return list(self._costs.values())

    def clear(self) -> None:
        """
        Clear all tracked costs

        Useful for testing or resetting between sessions.
        """
        with self._lock:
            self._costs.clear()

    def total_cost(self) -> float:
        """
        Get total cost across all requests

        Returns:
            Total cost in USD

        Example:
            total = tracker.total_cost()
            print(f"Total spent: ${total:.2f}")
        """
        with self._lock:
            return sum(c.total_cost_usd for c in self._costs.values())

    def total_tokens(self) -> int:
        """
        Get total tokens across all requests

        Returns:
            Total token count

        Example:
            tokens = tracker.total_tokens()
            print(f"Total tokens: {tokens:,}")
        """
        with self._lock:
            return sum(c.total_tokens for c in self._costs.values())

    def request_count(self) -> int:
        """
        Get number of tracked requests

        Returns:
            Number of unique requests
        """
        with self._lock:
            return len(self._costs)

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"CostTracker("
            f"requests={self.request_count()}, "
            f"total_cost=${self.total_cost():.4f}, "
            f"total_tokens={self.total_tokens():,})"
        )


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_global_tracker() -> CostTracker:
    """
    Get global cost tracker instance

    Returns:
        Singleton CostTracker instance

    Example:
        from greenlang.agents.intelligence.cost.tracker import get_global_tracker

        tracker = get_global_tracker()
        tracker.record("req_1", 100, 50, 0.0015)
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker
