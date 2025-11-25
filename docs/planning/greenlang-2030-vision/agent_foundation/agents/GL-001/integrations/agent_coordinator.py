# -*- coding: utf-8 -*-
"""
Multi-Agent Coordination System for GL-001 ProcessHeatOrchestrator

Orchestrates communication and coordination with 99 process heat agents (GL-002 to GL-100).
Implements message bus integration, command broadcasting, and response aggregation.

Features:
- Asynchronous message passing via internal agent_foundation
- Pub/sub patterns for event broadcasting
- Command/query/event message types
- Response aggregation with timeout handling
- Agent registry and health monitoring
- Load balancing across agents
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the agent system."""
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentMessage:
    """
    Standard message format for agent communication.

    Follows agent_foundation message specification.
    """
    message_id: str
    source_agent: str
    target_agents: List[str]
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timeout_seconds: int = 30
    requires_ack: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'source_agent': self.source_agent,
            'target_agents': self.target_agents,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'timeout_seconds': self.timeout_seconds,
            'requires_ack': self.requires_ack
        }


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_name: str
    agent_type: str
    capabilities: List[str]
    status: str  # online, offline, busy
    last_heartbeat: datetime
    performance_score: float = 100.0  # 0-100
    message_count: int = 0
    error_count: int = 0


@dataclass
class CoordinationStrategy:
    """Strategy for coordinating multiple agents."""
    strategy_type: str  # broadcast, round_robin, load_balanced, capability_based
    max_parallel: int = 10
    timeout_seconds: int = 30
    retry_count: int = 2
    aggregation_method: str = "all"  # all, majority, first, consensus


class MessageBus:
    """
    Internal message bus for agent communication.

    Implements pub/sub messaging patterns with topic-based routing.
    """

    def __init__(self):
        """Initialize message bus."""
        self.topics: Dict[str, Set[str]] = defaultdict(set)  # topic -> subscribers
        self.message_queues: Dict[str, asyncio.Queue] = {}  # agent_id -> queue
        self.pending_messages: Dict[str, AgentMessage] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, agent_id: str, topics: List[str]):
        """
        Subscribe agent to topics.

        Args:
            agent_id: Agent identifier
            topics: List of topics to subscribe
        """
        async with self._lock:
            # Create queue if not exists
            if agent_id not in self.message_queues:
                self.message_queues[agent_id] = asyncio.Queue(maxsize=1000)

            # Subscribe to topics
            for topic in topics:
                self.topics[topic].add(agent_id)
                logger.info(f"Agent {agent_id} subscribed to topic {topic}")

    async def unsubscribe(self, agent_id: str, topics: List[str]):
        """
        Unsubscribe agent from topics.

        Args:
            agent_id: Agent identifier
            topics: List of topics to unsubscribe
        """
        async with self._lock:
            for topic in topics:
                if topic in self.topics:
                    self.topics[topic].discard(agent_id)

    async def publish(self, topic: str, message: AgentMessage):
        """
        Publish message to topic.

        Args:
            topic: Topic to publish to
            message: Message to publish
        """
        async with self._lock:
            subscribers = self.topics.get(topic, set())

            for subscriber_id in subscribers:
                if subscriber_id in self.message_queues:
                    try:
                        await self.message_queues[subscriber_id].put(message)
                        logger.debug(f"Published message {message.message_id} to {subscriber_id}")
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for agent {subscriber_id}")

    async def send_direct(self, agent_id: str, message: AgentMessage):
        """
        Send message directly to agent.

        Args:
            agent_id: Target agent ID
            message: Message to send
        """
        async with self._lock:
            if agent_id not in self.message_queues:
                self.message_queues[agent_id] = asyncio.Queue(maxsize=1000)

            try:
                await self.message_queues[agent_id].put(message)
                self.pending_messages[message.message_id] = message
                logger.debug(f"Sent direct message {message.message_id} to {agent_id}")
            except asyncio.QueueFull:
                logger.error(f"Queue full for agent {agent_id}")
                raise

    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """
        Receive message for agent.

        Args:
            agent_id: Agent identifier
            timeout: Timeout in seconds

        Returns:
            Message or None if timeout
        """
        if agent_id not in self.message_queues:
            return None

        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                message = await self.message_queues[agent_id].get()

            return message

        except asyncio.TimeoutError:
            return None


class AgentRegistry:
    """
    Registry of all process heat agents (GL-002 to GL-100).

    Maintains agent information, capabilities, and health status.
    """

    def __init__(self):
        """Initialize agent registry."""
        self.agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

        # Initialize with known process heat agents
        self._initialize_process_heat_agents()

    def _initialize_process_heat_agents(self):
        """Initialize registry with process heat agents."""
        # Agent definitions from Agent_Process_Heat.csv
        process_heat_agents = [
            ("GL-002", "BoilerEfficiencyOptimizer", ["boiler", "efficiency", "optimization"]),
            ("GL-003", "SteamDistributionController", ["steam", "distribution", "control"]),
            ("GL-004", "CombustionOptimizer", ["combustion", "optimization", "emissions"]),
            ("GL-005", "HeatRecoveryManager", ["heat_recovery", "waste_heat", "efficiency"]),
            ("GL-006", "ThermalStorageController", ["thermal_storage", "energy_storage"]),
            ("GL-007", "FurnaceOptimizer", ["furnace", "temperature", "optimization"]),
            ("GL-008", "HeatExchangerMonitor", ["heat_exchanger", "monitoring", "maintenance"]),
            ("GL-009", "BurnerControlSystem", ["burner", "control", "safety"]),
            ("GL-010", "InsulationAnalyzer", ["insulation", "heat_loss", "analysis"]),
            # ... Continue for all 99 agents
        ]

        for agent_id, agent_name, capabilities in process_heat_agents:
            self.agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_type="process_heat",
                capabilities=capabilities,
                status="offline",
                last_heartbeat=DeterministicClock.utcnow() - timedelta(hours=1)
            )

    async def register_agent(self, agent_info: AgentInfo):
        """
        Register or update agent in registry.

        Args:
            agent_info: Agent information
        """
        async with self._lock:
            self.agents[agent_info.agent_id] = agent_info
            logger.info(f"Registered agent {agent_info.agent_id}: {agent_info.agent_name}")

    async def update_heartbeat(self, agent_id: str):
        """
        Update agent heartbeat timestamp.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].last_heartbeat = DeterministicClock.utcnow()
                self.agents[agent_id].status = "online"

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent information.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent info or None
        """
        async with self._lock:
            return self.agents.get(agent_id)

    async def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """
        Get agents with specific capability.

        Args:
            capability: Required capability

        Returns:
            List of agents with capability
        """
        async with self._lock:
            return [
                agent for agent in self.agents.values()
                if capability in agent.capabilities and agent.status == "online"
            ]

    async def get_online_agents(self) -> List[AgentInfo]:
        """Get all online agents."""
        async with self._lock:
            cutoff = DeterministicClock.utcnow() - timedelta(minutes=5)
            return [
                agent for agent in self.agents.values()
                if agent.last_heartbeat > cutoff
            ]


class CommandBroadcaster:
    """
    Broadcasts commands to multiple agents with various strategies.

    Implements different broadcasting patterns for agent coordination.
    """

    def __init__(self, message_bus: MessageBus, registry: AgentRegistry):
        """Initialize command broadcaster."""
        self.message_bus = message_bus
        self.registry = registry

    async def broadcast_command(
        self,
        command: str,
        agent_ids: List[str],
        parameters: Dict[str, Any],
        strategy: CoordinationStrategy
    ) -> Dict[str, Any]:
        """
        Broadcast command to multiple agents.

        Args:
            command: Command to execute
            agent_ids: Target agent IDs
            parameters: Command parameters
            strategy: Coordination strategy

        Returns:
            Command execution results
        """
        # Create command message
        message = AgentMessage(
            message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            source_agent="GL-001",
            target_agents=agent_ids,
            message_type=MessageType.COMMAND,
            priority=MessagePriority.NORMAL,
            payload={
                'command': command,
                'parameters': parameters
            },
            timestamp=DeterministicClock.utcnow(),
            timeout_seconds=strategy.timeout_seconds,
            requires_ack=True
        )

        # Apply broadcasting strategy
        if strategy.strategy_type == "broadcast":
            return await self._broadcast_all(message, agent_ids, strategy)
        elif strategy.strategy_type == "round_robin":
            return await self._broadcast_round_robin(message, agent_ids, strategy)
        elif strategy.strategy_type == "load_balanced":
            return await self._broadcast_load_balanced(message, agent_ids, strategy)
        elif strategy.strategy_type == "capability_based":
            return await self._broadcast_capability_based(message, parameters.get('required_capability'), strategy)
        else:
            raise ValueError(f"Unknown strategy type: {strategy.strategy_type}")

    async def _broadcast_all(
        self,
        message: AgentMessage,
        agent_ids: List[str],
        strategy: CoordinationStrategy
    ) -> Dict[str, Any]:
        """Broadcast to all agents simultaneously."""
        tasks = []

        # Create tasks for parallel execution
        for i in range(0, len(agent_ids), strategy.max_parallel):
            batch = agent_ids[i:i + strategy.max_parallel]
            for agent_id in batch:
                task = asyncio.create_task(
                    self._send_and_wait(agent_id, message, strategy.timeout_seconds)
                )
                tasks.append((agent_id, task))

            # Wait for batch completion if not last batch
            if i + strategy.max_parallel < len(agent_ids):
                await asyncio.sleep(0.1)  # Small delay between batches

        # Collect responses
        responses = {}
        for agent_id, task in tasks:
            try:
                response = await task
                responses[agent_id] = response
            except Exception as e:
                logger.error(f"Error broadcasting to {agent_id}: {e}")
                responses[agent_id] = {'error': str(e)}

        return responses

    async def _broadcast_round_robin(
        self,
        message: AgentMessage,
        agent_ids: List[str],
        strategy: CoordinationStrategy
    ) -> Dict[str, Any]:
        """Broadcast using round-robin pattern."""
        responses = {}

        for agent_id in agent_ids:
            try:
                response = await self._send_and_wait(agent_id, message, strategy.timeout_seconds)
                responses[agent_id] = response

                # Stop if we have enough responses
                if strategy.aggregation_method == "first" and responses:
                    break

            except Exception as e:
                logger.error(f"Error sending to {agent_id}: {e}")
                responses[agent_id] = {'error': str(e)}

        return responses

    async def _broadcast_load_balanced(
        self,
        message: AgentMessage,
        agent_ids: List[str],
        strategy: CoordinationStrategy
    ) -> Dict[str, Any]:
        """Broadcast based on agent load."""
        # Get agent performance scores
        agent_scores = {}
        for agent_id in agent_ids:
            agent = await self.registry.get_agent(agent_id)
            if agent:
                agent_scores[agent_id] = agent.performance_score

        # Sort by performance (higher score = better)
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        # Send to top performers
        responses = {}
        for agent_id, score in sorted_agents[:strategy.max_parallel]:
            try:
                response = await self._send_and_wait(agent_id, message, strategy.timeout_seconds)
                responses[agent_id] = response
            except Exception as e:
                logger.error(f"Error sending to {agent_id}: {e}")
                responses[agent_id] = {'error': str(e)}

        return responses

    async def _broadcast_capability_based(
        self,
        message: AgentMessage,
        capability: str,
        strategy: CoordinationStrategy
    ) -> Dict[str, Any]:
        """Broadcast based on agent capabilities."""
        # Find agents with required capability
        capable_agents = await self.registry.get_agents_by_capability(capability)

        if not capable_agents:
            logger.warning(f"No agents found with capability: {capability}")
            return {}

        # Send to capable agents
        agent_ids = [agent.agent_id for agent in capable_agents]
        return await self._broadcast_all(message, agent_ids, strategy)

    async def _send_and_wait(
        self,
        agent_id: str,
        message: AgentMessage,
        timeout: float
    ) -> Dict[str, Any]:
        """Send message and wait for response."""
        # Send message
        await self.message_bus.send_direct(agent_id, message)

        # Wait for response
        response = await self.message_bus.receive(f"GL-001_response", timeout)

        if response and response.correlation_id == message.message_id:
            return response.payload
        else:
            raise TimeoutError(f"No response from {agent_id} within {timeout} seconds")


class ResponseAggregator:
    """
    Aggregates responses from multiple agents.

    Implements various aggregation strategies for multi-agent responses.
    """

    def __init__(self):
        """Initialize response aggregator."""
        self.aggregation_methods = {
            'all': self._aggregate_all,
            'majority': self._aggregate_majority,
            'average': self._aggregate_average,
            'consensus': self._aggregate_consensus,
            'first': self._aggregate_first,
            'best': self._aggregate_best
        }

    async def aggregate_responses(
        self,
        responses: Dict[str, Any],
        method: str = 'all'
    ) -> Dict[str, Any]:
        """
        Aggregate multiple agent responses.

        Args:
            responses: Dictionary of agent responses
            method: Aggregation method

        Returns:
            Aggregated result
        """
        if method not in self.aggregation_methods:
            raise ValueError(f"Unknown aggregation method: {method}")

        return await self.aggregation_methods[method](responses)

    async def _aggregate_all(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Return all responses."""
        return {
            'method': 'all',
            'responses': responses,
            'total_agents': len(responses),
            'successful': sum(1 for r in responses.values() if 'error' not in r)
        }

    async def _aggregate_majority(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Return majority consensus."""
        # Count votes for each unique response
        votes = defaultdict(list)
        for agent_id, response in responses.items():
            if 'error' not in response:
                # Create hash of response for grouping
                response_hash = hashlib.md5(
                    json.dumps(response, sort_keys=True).encode()
                ).hexdigest()
                votes[response_hash].append((agent_id, response))

        # Find majority
        if votes:
            majority_hash = max(votes.keys(), key=lambda k: len(votes[k]))
            majority_agents, majority_response = zip(*votes[majority_hash])

            return {
                'method': 'majority',
                'result': majority_response[0],
                'agents': list(majority_agents),
                'confidence': len(majority_agents) / len(responses)
            }

        return {
            'method': 'majority',
            'result': None,
            'confidence': 0
        }

    async def _aggregate_average(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate average of numeric responses."""
        numeric_values = []
        for response in responses.values():
            if 'error' not in response and 'value' in response:
                try:
                    numeric_values.append(float(response['value']))
                except (TypeError, ValueError):
                    continue

        if numeric_values:
            return {
                'method': 'average',
                'result': sum(numeric_values) / len(numeric_values),
                'count': len(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values)
            }

        return {
            'method': 'average',
            'result': None,
            'count': 0
        }

    async def _aggregate_consensus(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Require consensus among all agents."""
        valid_responses = [
            r for r in responses.values()
            if 'error' not in r
        ]

        if not valid_responses:
            return {
                'method': 'consensus',
                'consensus': False,
                'result': None
            }

        # Check if all responses are identical
        first_response = json.dumps(valid_responses[0], sort_keys=True)
        consensus = all(
            json.dumps(r, sort_keys=True) == first_response
            for r in valid_responses
        )

        return {
            'method': 'consensus',
            'consensus': consensus,
            'result': valid_responses[0] if consensus else None,
            'agreement_rate': 1.0 if consensus else 0.0
        }

    async def _aggregate_first(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Return first valid response."""
        for agent_id, response in responses.items():
            if 'error' not in response:
                return {
                    'method': 'first',
                    'result': response,
                    'agent': agent_id
                }

        return {
            'method': 'first',
            'result': None,
            'agent': None
        }

    async def _aggregate_best(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Return best response based on quality score."""
        best_response = None
        best_score = -1
        best_agent = None

        for agent_id, response in responses.items():
            if 'error' not in response and 'quality_score' in response:
                score = response.get('quality_score', 0)
                if score > best_score:
                    best_score = score
                    best_response = response
                    best_agent = agent_id

        return {
            'method': 'best',
            'result': best_response,
            'agent': best_agent,
            'quality_score': best_score
        }


class AgentCoordinator:
    """
    Main coordinator for multi-agent system.

    Orchestrates all agent communication, coordination, and management for GL-001.
    """

    def __init__(self):
        """Initialize agent coordinator."""
        self.message_bus = MessageBus()
        self.registry = AgentRegistry()
        self.broadcaster = CommandBroadcaster(self.message_bus, self.registry)
        self.aggregator = ResponseAggregator()
        self.monitoring_task = None

    async def initialize(self):
        """Initialize agent coordination system."""
        logger.info("Initializing agent coordinator for GL-001")

        # Subscribe to orchestrator topics
        await self.message_bus.subscribe("GL-001", [
            "heat_optimization",
            "energy_efficiency",
            "system_status",
            "emergency"
        ])

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_agents())

        logger.info("Agent coordinator initialized")

    async def execute_command(
        self,
        command: str,
        target_agents: List[str],
        parameters: Dict[str, Any],
        strategy: Optional[CoordinationStrategy] = None
    ) -> Dict[str, Any]:
        """
        Execute command across multiple agents.

        Args:
            command: Command to execute
            target_agents: List of target agent IDs
            parameters: Command parameters
            strategy: Coordination strategy (optional)

        Returns:
            Aggregated command results
        """
        if not strategy:
            strategy = CoordinationStrategy(
                strategy_type="broadcast",
                max_parallel=10,
                timeout_seconds=30,
                aggregation_method="all"
            )

        # Broadcast command
        responses = await self.broadcaster.broadcast_command(
            command, target_agents, parameters, strategy
        )

        # Aggregate responses
        aggregated = await self.aggregator.aggregate_responses(
            responses, strategy.aggregation_method
        )

        logger.info(f"Executed command '{command}' across {len(target_agents)} agents")
        return aggregated

    async def query_agents(
        self,
        query: str,
        capability: Optional[str] = None,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Query agents for information.

        Args:
            query: Query string
            capability: Required capability (optional)
            timeout: Query timeout in seconds

        Returns:
            Query results from agents
        """
        # Find target agents
        if capability:
            agents = await self.registry.get_agents_by_capability(capability)
            target_agents = [agent.agent_id for agent in agents]
        else:
            agents = await self.registry.get_online_agents()
            target_agents = [agent.agent_id for agent in agents]

        if not target_agents:
            return {'error': 'No suitable agents found'}

        # Create query message
        message = AgentMessage(
            message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            source_agent="GL-001",
            target_agents=target_agents,
            message_type=MessageType.QUERY,
            priority=MessagePriority.NORMAL,
            payload={'query': query},
            timestamp=DeterministicClock.utcnow(),
            timeout_seconds=timeout
        )

        # Send query
        responses = {}
        for agent_id in target_agents:
            await self.message_bus.send_direct(agent_id, message)

        # Collect responses
        end_time = DeterministicClock.utcnow() + timedelta(seconds=timeout)
        while DeterministicClock.utcnow() < end_time and len(responses) < len(target_agents):
            response = await self.message_bus.receive("GL-001", timeout=1)
            if response and response.correlation_id == message.message_id:
                responses[response.source_agent] = response.payload

        return responses

    async def broadcast_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ):
        """
        Broadcast event to all relevant agents.

        Args:
            event_type: Type of event
            event_data: Event data
            priority: Message priority
        """
        # Create event message
        message = AgentMessage(
            message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            source_agent="GL-001",
            target_agents=["*"],  # Broadcast to all
            message_type=MessageType.EVENT,
            priority=priority,
            payload={
                'event_type': event_type,
                'data': event_data
            },
            timestamp=DeterministicClock.utcnow(),
            requires_ack=False
        )

        # Publish to event topic
        await self.message_bus.publish(f"event.{event_type}", message)

        logger.info(f"Broadcasted event '{event_type}' to all agents")

    async def _monitor_agents(self):
        """Monitor agent health and status."""
        while True:
            try:
                # Send heartbeat to all agents
                online_agents = await self.registry.get_online_agents()

                for agent in online_agents:
                    heartbeat = AgentMessage(
                        message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                        source_agent="GL-001",
                        target_agents=[agent.agent_id],
                        message_type=MessageType.HEARTBEAT,
                        priority=MessagePriority.LOW,
                        payload={},
                        timestamp=DeterministicClock.utcnow(),
                        timeout_seconds=5
                    )

                    await self.message_bus.send_direct(agent.agent_id, heartbeat)

                # Update agent status based on responses
                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Agent monitoring error: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        """Shutdown agent coordinator."""
        logger.info("Shutting down agent coordinator")

        if self.monitoring_task:
            self.monitoring_task.cancel()

        logger.info("Agent coordinator shutdown complete")


# Example usage
async def main():
    """Example agent coordinator usage."""

    # Create coordinator
    coordinator = AgentCoordinator()
    await coordinator.initialize()

    # Execute optimization command across boiler agents
    result = await coordinator.execute_command(
        command="optimize_efficiency",
        target_agents=["GL-002", "GL-003", "GL-004"],
        parameters={
            'target_efficiency': 0.95,
            'max_temperature': 200
        },
        strategy=CoordinationStrategy(
            strategy_type="broadcast",
            max_parallel=3,
            timeout_seconds=10,
            aggregation_method="average"
        )
    )

    print(f"Optimization result: {result}")

    # Query agents with heat recovery capability
    heat_recovery_data = await coordinator.query_agents(
        query="get_recovery_potential",
        capability="heat_recovery",
        timeout=5
    )

    print(f"Heat recovery data: {heat_recovery_data}")

    # Broadcast system event
    await coordinator.broadcast_event(
        event_type="temperature_alert",
        event_data={
            'location': 'Boiler_01',
            'temperature': 250,
            'threshold': 200
        },
        priority=MessagePriority.HIGH
    )

    # Shutdown
    await coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())