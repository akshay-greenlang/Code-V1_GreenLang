# -*- coding: utf-8 -*-
"""
GreenLang Agent Foundation - Orchestration and Coordination Systems

This package provides enterprise-grade orchestration capabilities for managing
10,000+ concurrent AI agents with production-ready communication, coordination,
and distributed intelligence patterns.

Components:
    - message_bus: Kafka-based event bus for agent communication
    - agent_coordinator: Central coordinator for managing multiple agents
    - pipeline: Agent pipeline orchestration (sequential, parallel, conditional)
    - swarm: Agent swarm implementation for distributed intelligence
    - routing: Dynamic routing and scatter-gather patterns
    - saga: Long-running distributed transaction coordination
    - agent_registry: Agent discovery and registry service

Example:
    >>> from orchestration import AgentCoordinator, MessageBus, Pipeline
    >>>
    >>> # Initialize orchestration infrastructure
    >>> bus = MessageBus(kafka_config)
    >>> coordinator = AgentCoordinator(bus)
    >>>
    >>> # Create and execute agent pipeline
    >>> pipeline = Pipeline([agent1, agent2, agent3])
    >>> result = await pipeline.execute(input_data)
"""

from .message_bus import MessageBus, Message, MessageType, Priority
from .agent_coordinator import AgentCoordinator, CoordinatorConfig
from .pipeline import Pipeline, PipelineStage, ExecutionMode
from .swarm import SwarmOrchestrator, SwarmConfig, SwarmAgent
from .routing import MessageRouter, RoutingStrategy, ScatterGather
from .saga import SagaOrchestrator, SagaTransaction, CompensationAction
from .agent_registry import AgentRegistry, AgentDescriptor, ServiceDiscovery

__version__ = "1.0.0"
__author__ = "GreenLang AI Architecture Team"

__all__ = [
    # Message Bus
    "MessageBus",
    "Message",
    "MessageType",
    "Priority",

    # Coordinator
    "AgentCoordinator",
    "CoordinatorConfig",

    # Pipeline
    "Pipeline",
    "PipelineStage",
    "ExecutionMode",

    # Swarm
    "SwarmOrchestrator",
    "SwarmConfig",
    "SwarmAgent",

    # Routing
    "MessageRouter",
    "RoutingStrategy",
    "ScatterGather",

    # Saga
    "SagaOrchestrator",
    "SagaTransaction",
    "CompensationAction",

    # Registry
    "AgentRegistry",
    "AgentDescriptor",
    "ServiceDiscovery",
]

# Performance targets validation
PERFORMANCE_TARGETS = {
    "message_latency_ms": 10,
    "concurrent_agents": 10000,
    "throughput_per_second": 100000,
    "availability_target": 0.9999,
    "kafka_partitions": 100,
    "replication_factor": 3
}

def validate_configuration():
    """Validate orchestration configuration meets performance requirements."""
    return PERFORMANCE_TARGETS