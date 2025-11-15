"""
Routing - Dynamic routing and scatter-gather patterns for agent communication.

This module implements intelligent message routing strategies including
content-based routing, scatter-gather, and dynamic load balancing for
10,000+ concurrent agents.

Example:
    >>> router = MessageRouter(message_bus)
    >>> await router.initialize()
    >>>
    >>> # Scatter-gather pattern
    >>> scatter = ScatterGather(router)
    >>> responses = await scatter.execute(
    ...     request=request_message,
    ...     target_agents=["agent-001", "agent-002", "agent-003"],
    ...     aggregation_strategy="majority_vote"
    ... )
"""

from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
from simpleeval import simple_eval
import logging
from datetime import datetime, timezone
import uuid
import hashlib
import random
from collections import defaultdict
import re

from prometheus_client import Counter, Histogram, Gauge

from .message_bus import MessageBus, Message, MessageType, Priority

logger = logging.getLogger(__name__)

# Metrics
routing_counter = Counter('routing_messages_total', 'Total messages routed', ['strategy', 'status'])
routing_latency_histogram = Histogram('routing_latency_ms', 'Routing decision latency')
scatter_gather_histogram = Histogram('scatter_gather_duration_ms', 'Scatter-gather operation duration')
route_cache_hits = Counter('routing_cache_hits_total', 'Routing cache hits')


class RoutingStrategy(str, Enum):
    """Message routing strategies."""
    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_LOADED = "LEAST_LOADED"
    WEIGHTED = "WEIGHTED"
    CONTENT_BASED = "CONTENT_BASED"
    PRIORITY_BASED = "PRIORITY_BASED"
    AFFINITY = "AFFINITY"
    BROADCAST = "BROADCAST"
    FAILOVER = "FAILOVER"
    CONSISTENT_HASH = "CONSISTENT_HASH"


class AggregationStrategy(str, Enum):
    """Response aggregation strategies for scatter-gather."""
    FIRST = "FIRST"                   # Return first response
    ALL = "ALL"                       # Return all responses
    MAJORITY_VOTE = "MAJORITY_VOTE"   # Return most common response
    AVERAGE = "AVERAGE"               # Average numeric responses
    MERGE = "MERGE"                   # Merge dict/list responses
    CUSTOM = "CUSTOM"                 # Custom aggregation function


class RouteRule(BaseModel):
    """Routing rule definition."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Rule name")
    priority: int = Field(default=100, ge=0, le=1000)
    condition: str = Field(..., description="Routing condition expression")
    targets: List[str] = Field(..., description="Target agent IDs")
    strategy: RoutingStrategy = Field(default=RoutingStrategy.ROUND_ROBIN)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)

    def evaluate(self, message: Message) -> bool:
        """Evaluate if rule applies to message."""
        try:
            # Create evaluation context
            context = {
                "message_type": message.message_type,
                "priority": message.priority.value,
                "sender": message.sender_id,
                "recipient": message.recipient_id,
                "payload": message.payload
            }

            # Safe evaluation
            return simple_eval(self.condition, names=context)  # SECURITY FIX
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return False


class RouteTable(BaseModel):
    """Routing table for managing routes."""

    routes: Dict[str, List[str]] = Field(default_factory=dict, description="Static routes")
    rules: List[RouteRule] = Field(default_factory=list, description="Dynamic routing rules")
    default_route: Optional[str] = Field(None, description="Default route if no match")
    cache_ttl_seconds: int = Field(default=60, ge=0)
    max_cache_size: int = Field(default=10000, ge=0)


class LoadInfo(BaseModel):
    """Agent load information for routing decisions."""

    agent_id: str = Field(...)
    message_queue_size: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    capacity: int = Field(default=100, ge=1)
    weight: float = Field(default=1.0, ge=0.0)
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def load_factor(self) -> float:
        """Calculate load factor (0.0 = idle, 1.0 = full capacity)."""
        queue_load = min(self.message_queue_size / self.capacity, 1.0)
        error_penalty = self.error_rate * 0.5
        return min(queue_load + error_penalty, 1.0)


class MessageRouter:
    """
    Intelligent message routing system.

    Provides dynamic routing, load balancing, and fault tolerance
    for agent communication networks.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        route_table: Optional[RouteTable] = None
    ):
        """Initialize message router."""
        self.message_bus = message_bus
        self.route_table = route_table or RouteTable()

        # Agent tracking
        self.agent_loads: Dict[str, LoadInfo] = {}
        self.agent_affinities: Dict[str, str] = {}  # sender -> preferred target

        # Routing state
        self.round_robin_indices: Dict[str, int] = defaultdict(int)
        self.route_cache: Dict[str, Tuple[List[str], float]] = {}
        self.failover_chains: Dict[str, List[str]] = {}

        # Consistent hashing
        self.hash_ring: Dict[int, str] = {}
        self._build_hash_ring()

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize router and start background tasks."""
        logger.info("Initializing MessageRouter")

        self._running = True

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._load_monitor())
        )
        self._tasks.append(
            asyncio.create_task(self._cache_cleaner())
        )

        logger.info("MessageRouter initialized")

    async def shutdown(self) -> None:
        """Shutdown router."""
        logger.info("Shutting down MessageRouter")
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("MessageRouter shutdown complete")

    async def route(
        self,
        message: Message,
        strategy: Optional[RoutingStrategy] = None
    ) -> List[str]:
        """
        Route message to target agents.

        Args:
            message: Message to route
            strategy: Override routing strategy

        Returns:
            List of target agent IDs
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Check cache
            cache_key = self._get_cache_key(message)
            if cache_key in self.route_cache:
                targets, timestamp = self.route_cache[cache_key]
                if (datetime.now(timezone.utc).timestamp() - timestamp) < self.route_table.cache_ttl_seconds:
                    route_cache_hits.inc()
                    return targets

            # Determine routing strategy
            if not strategy:
                strategy = self._determine_strategy(message)

            # Get candidate targets
            candidates = self._get_candidates(message)

            if not candidates:
                logger.warning(f"No routing candidates for message {message.message_id}")
                routing_counter.labels(strategy=strategy, status="no_targets").inc()
                return []

            # Apply routing strategy
            targets = await self._apply_strategy(message, candidates, strategy)

            # Update cache
            if len(self.route_cache) < self.route_table.max_cache_size:
                self.route_cache[cache_key] = (targets, datetime.now(timezone.utc).timestamp())

            # Update metrics
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            routing_latency_histogram.observe(latency_ms)
            routing_counter.labels(strategy=strategy, status="success").inc()

            logger.debug(f"Routed message {message.message_id} to {len(targets)} targets using {strategy}")
            return targets

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            routing_counter.labels(strategy=strategy or "unknown", status="error").inc()
            return []

    async def _apply_strategy(
        self,
        message: Message,
        candidates: List[str],
        strategy: RoutingStrategy
    ) -> List[str]:
        """Apply routing strategy to select targets."""
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin(candidates, message.sender_id)

        elif strategy == RoutingStrategy.LEAST_LOADED:
            return await self._least_loaded(candidates)

        elif strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_random(candidates)

        elif strategy == RoutingStrategy.CONTENT_BASED:
            return self._content_based(message, candidates)

        elif strategy == RoutingStrategy.PRIORITY_BASED:
            return self._priority_based(message, candidates)

        elif strategy == RoutingStrategy.AFFINITY:
            return self._affinity_based(message, candidates)

        elif strategy == RoutingStrategy.BROADCAST:
            return candidates

        elif strategy == RoutingStrategy.FAILOVER:
            return await self._failover(candidates)

        elif strategy == RoutingStrategy.CONSISTENT_HASH:
            return self._consistent_hash(message, candidates)

        else:
            return [random.choice(candidates)]

    def _round_robin(self, candidates: List[str], key: str) -> List[str]:
        """Round-robin routing."""
        index = self.round_robin_indices[key] % len(candidates)
        self.round_robin_indices[key] = (index + 1) % len(candidates)
        return [candidates[index]]

    async def _least_loaded(self, candidates: List[str]) -> List[str]:
        """Route to least loaded agent."""
        loads = []
        for agent_id in candidates:
            load = self.agent_loads.get(agent_id, LoadInfo(agent_id=agent_id))
            loads.append((agent_id, load.load_factor))

        # Sort by load factor
        loads.sort(key=lambda x: x[1])
        return [loads[0][0]] if loads else []

    def _weighted_random(self, candidates: List[str]) -> List[str]:
        """Weighted random selection."""
        weights = []
        for agent_id in candidates:
            load = self.agent_loads.get(agent_id, LoadInfo(agent_id=agent_id))
            # Inverse weight based on load (less loaded = higher weight)
            weight = load.weight * (1.0 - load.load_factor)
            weights.append(weight)

        if not weights or sum(weights) == 0:
            return [random.choice(candidates)]

        # Weighted random choice
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            if upto + w >= r:
                return [candidates[i]]
            upto += w

        return [candidates[-1]]

    def _content_based(self, message: Message, candidates: List[str]) -> List[str]:
        """Content-based routing using message payload."""
        # Example: Route based on task type in payload
        task_type = message.payload.get("task_type", "default")

        # Map task types to specialized agents
        task_mapping = {
            "calculation": lambda c: [a for a in c if "calculator" in a.lower()],
            "validation": lambda c: [a for a in c if "validator" in a.lower()],
            "reporting": lambda c: [a for a in c if "reporter" in a.lower()]
        }

        if task_type in task_mapping:
            specialized = task_mapping[task_type](candidates)
            if specialized:
                return [specialized[0]]

        return [candidates[0]] if candidates else []

    def _priority_based(self, message: Message, candidates: List[str]) -> List[str]:
        """Priority-based routing."""
        if message.priority == Priority.CRITICAL:
            # Route to most reliable agents
            reliable = []
            for agent_id in candidates:
                load = self.agent_loads.get(agent_id, LoadInfo(agent_id=agent_id))
                if load.error_rate < 0.05:  # Less than 5% error rate
                    reliable.append(agent_id)
            return [reliable[0]] if reliable else [candidates[0]]

        elif message.priority == Priority.LOW:
            # Route to any available agent
            return [random.choice(candidates)]

        else:
            # Normal priority - use least loaded
            return self._round_robin(candidates, "normal")

    def _affinity_based(self, message: Message, candidates: List[str]) -> List[str]:
        """Affinity-based routing for session continuity."""
        # Check if sender has affinity
        if message.sender_id in self.agent_affinities:
            preferred = self.agent_affinities[message.sender_id]
            if preferred in candidates:
                return [preferred]

        # Establish new affinity
        target = random.choice(candidates)
        self.agent_affinities[message.sender_id] = target
        return [target]

    async def _failover(self, candidates: List[str]) -> List[str]:
        """Failover routing with health checks."""
        for agent_id in candidates:
            # Check agent health
            load = self.agent_loads.get(agent_id, LoadInfo(agent_id=agent_id))
            if load.error_rate < 0.1 and load.load_factor < 0.9:
                return [agent_id]

        # All unhealthy, return least bad option
        return [candidates[0]] if candidates else []

    def _consistent_hash(self, message: Message, candidates: List[str]) -> List[str]:
        """Consistent hashing for stable routing."""
        # Hash message key
        key = f"{message.sender_id}:{message.payload.get('key', '')}"
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)

        # Find position on ring
        positions = sorted(self.hash_ring.keys())
        for pos in positions:
            if hash_value <= pos:
                agent = self.hash_ring[pos]
                if agent in candidates:
                    return [agent]

        # Wrap around
        return [self.hash_ring[positions[0]]] if positions else []

    def _build_hash_ring(self, virtual_nodes: int = 150) -> None:
        """Build consistent hash ring."""
        self.hash_ring.clear()

        # Add virtual nodes for better distribution
        for i in range(virtual_nodes):
            key = f"node-{i}"
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            self.hash_ring[hash_value] = f"agent-{i % 100:03d}"  # Distribute across 100 agents

    def _determine_strategy(self, message: Message) -> RoutingStrategy:
        """Determine routing strategy based on message and rules."""
        # Check routing rules
        for rule in sorted(self.route_table.rules, key=lambda r: r.priority, reverse=True):
            if rule.enabled and rule.evaluate(message):
                return rule.strategy

        # Default strategies based on message type
        if message.recipient_id == "broadcast":
            return RoutingStrategy.BROADCAST
        elif message.priority == Priority.CRITICAL:
            return RoutingStrategy.PRIORITY_BASED
        elif message.message_type == MessageType.REQUEST:
            return RoutingStrategy.LEAST_LOADED
        else:
            return RoutingStrategy.ROUND_ROBIN

    def _get_candidates(self, message: Message) -> List[str]:
        """Get candidate agents for routing."""
        # Check static routes
        if message.recipient_id in self.route_table.routes:
            return self.route_table.routes[message.recipient_id]

        # Check routing rules
        candidates = set()
        for rule in self.route_table.rules:
            if rule.enabled and rule.evaluate(message):
                candidates.update(rule.targets)

        if candidates:
            return list(candidates)

        # Default route
        if self.route_table.default_route:
            return [self.route_table.default_route]

        return []

    def _get_cache_key(self, message: Message) -> str:
        """Generate cache key for message."""
        return f"{message.sender_id}:{message.recipient_id}:{message.message_type}"

    async def _load_monitor(self) -> None:
        """Monitor agent loads."""
        while self._running:
            try:
                # Collect load metrics via message bus
                # This would typically query agents for their load
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Load monitor error: {e}")

    async def _cache_cleaner(self) -> None:
        """Clean expired cache entries."""
        while self._running:
            try:
                now = datetime.now(timezone.utc).timestamp()
                expired = []

                for key, (_, timestamp) in self.route_cache.items():
                    if (now - timestamp) > self.route_table.cache_ttl_seconds:
                        expired.append(key)

                for key in expired:
                    del self.route_cache[key]

                await asyncio.sleep(60)  # Clean every minute

            except Exception as e:
                logger.error(f"Cache cleaner error: {e}")

    def update_agent_load(self, agent_id: str, load: LoadInfo) -> None:
        """Update agent load information."""
        self.agent_loads[agent_id] = load

    def add_route_rule(self, rule: RouteRule) -> None:
        """Add routing rule."""
        self.route_table.rules.append(rule)
        # Clear cache as rules changed
        self.route_cache.clear()

    def remove_route_rule(self, rule_id: str) -> bool:
        """Remove routing rule."""
        for i, rule in enumerate(self.route_table.rules):
            if rule.rule_id == rule_id:
                self.route_table.rules.pop(i)
                self.route_cache.clear()
                return True
        return False


class ScatterGather:
    """
    Scatter-gather pattern implementation.

    Distributes requests to multiple agents and aggregates responses.
    """

    def __init__(self, router: MessageRouter):
        """Initialize scatter-gather."""
        self.router = router
        self.message_bus = router.message_bus

    async def execute(
        self,
        request: Message,
        target_agents: List[str],
        aggregation_strategy: AggregationStrategy = AggregationStrategy.ALL,
        timeout_ms: int = 5000,
        min_responses: int = 1,
        custom_aggregator: Optional[Callable] = None
    ) -> Any:
        """
        Execute scatter-gather operation.

        Args:
            request: Request message to scatter
            target_agents: List of target agents
            aggregation_strategy: How to aggregate responses
            timeout_ms: Timeout for gathering responses
            min_responses: Minimum responses required
            custom_aggregator: Custom aggregation function

        Returns:
            Aggregated response
        """
        start_time = datetime.now(timezone.utc)
        request_id = str(uuid.uuid4())
        responses = []

        try:
            logger.info(f"Starting scatter-gather {request_id} to {len(target_agents)} agents")

            # Scatter phase - send to all targets
            tasks = []
            for agent_id in target_agents:
                msg = Message(
                    sender_id=request.sender_id,
                    recipient_id=agent_id,
                    message_type=MessageType.REQUEST,
                    priority=request.priority,
                    payload=request.payload,
                    metadata=request.metadata
                )
                msg.metadata.correlation_id = request_id

                tasks.append(
                    self.message_bus.request_response(msg, timeout_ms)
                )

            # Gather phase - collect responses
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Message):
                    responses.append(result.payload)
                elif not isinstance(result, Exception):
                    responses.append(result)

            # Check minimum responses
            if len(responses) < min_responses:
                raise ValueError(f"Insufficient responses: {len(responses)}/{min_responses}")

            # Aggregate responses
            aggregated = self._aggregate(
                responses,
                aggregation_strategy,
                custom_aggregator
            )

            # Update metrics
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            scatter_gather_histogram.observe(duration_ms)

            logger.info(f"Scatter-gather {request_id} completed with {len(responses)} responses")
            return aggregated

        except Exception as e:
            logger.error(f"Scatter-gather {request_id} failed: {e}")
            raise

    def _aggregate(
        self,
        responses: List[Any],
        strategy: AggregationStrategy,
        custom_aggregator: Optional[Callable]
    ) -> Any:
        """Aggregate responses based on strategy."""
        if not responses:
            return None

        if strategy == AggregationStrategy.FIRST:
            return responses[0]

        elif strategy == AggregationStrategy.ALL:
            return responses

        elif strategy == AggregationStrategy.MAJORITY_VOTE:
            # Count occurrences
            from collections import Counter
            counts = Counter(str(r) for r in responses)
            return counts.most_common(1)[0][0] if counts else None

        elif strategy == AggregationStrategy.AVERAGE:
            # Average numeric responses
            numeric = [r for r in responses if isinstance(r, (int, float))]
            return sum(numeric) / len(numeric) if numeric else None

        elif strategy == AggregationStrategy.MERGE:
            # Merge dict/list responses
            if all(isinstance(r, dict) for r in responses):
                merged = {}
                for r in responses:
                    merged.update(r)
                return merged
            elif all(isinstance(r, list) for r in responses):
                merged = []
                for r in responses:
                    merged.extend(r)
                return merged
            else:
                return responses

        elif strategy == AggregationStrategy.CUSTOM and custom_aggregator:
            return custom_aggregator(responses)

        else:
            return responses

