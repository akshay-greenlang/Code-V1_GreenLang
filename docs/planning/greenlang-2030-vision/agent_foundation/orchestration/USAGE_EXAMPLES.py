# -*- coding: utf-8 -*-
"""
Orchestration Components - Comprehensive Usage Examples

This file provides production-ready examples for all orchestration components:
- SwarmOrchestrator (distributed intelligence)
- MessageRouter (dynamic routing)
- SagaOrchestrator (distributed transactions)
- AgentRegistry (service discovery)

Run examples:
    python USAGE_EXAMPLES.py --example swarm
    python USAGE_EXAMPLES.py --example routing
    python USAGE_EXAMPLES.py --example saga
    python USAGE_EXAMPLES.py --example registry
    python USAGE_EXAMPLES.py --example integrated
"""

import asyncio
import logging
from typing import Dict, Any, List
import argparse

# Orchestration imports
from message_bus import MessageBus, Message, MessageType, Priority, KafkaConfig
from swarm import SwarmOrchestrator, SwarmTask, SwarmBehavior, SwarmConfig
from routing import MessageRouter, RouteRule, RoutingStrategy, ScatterGather, AggregationStrategy, RouteTable
from saga import SagaOrchestrator, SagaTransaction, SagaStep, CompensationStrategy, SagaConfig
from agent_registry import AgentRegistry, AgentDescriptor, ServiceType, DiscoveryQuery

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Swarm Orchestration - Distributed Carbon Calculation
# ============================================================================

async def example_swarm_carbon_calculation():
    """
    Example: Calculate Scope 3 emissions for 100,000 suppliers using swarm intelligence.

    Scenario:
    - Company has 100,000 suppliers across global supply chain
    - Need to calculate carbon footprint for each supplier
    - Use swarm of 100 agents with foraging behavior
    - Agents explore emission factor space and exploit successful patterns
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Swarm Orchestration - Carbon Calculation")
    logger.info("=" * 80)

    # Initialize message bus
    kafka_config = KafkaConfig(
        bootstrap_servers=["localhost:9092"],
        partitions=100,
        replication_factor=1  # Single broker for dev
    )
    message_bus = MessageBus(kafka_config)
    await message_bus.initialize()

    # Initialize swarm orchestrator
    swarm_config = SwarmConfig(
        min_agents=50,
        max_agents=1000,
        neighbor_radius=5.0,
        pheromone_evaporation_rate=0.01,
        separation_weight=0.3,
        alignment_weight=0.2,
        cohesion_weight=0.2,
        attraction_weight=0.3
    )

    swarm = SwarmOrchestrator(message_bus, swarm_config)
    await swarm.initialize()

    # Define carbon calculation task
    task = SwarmTask(
        objective="Calculate Scope 3 emissions for 100,000 suppliers",
        data_partitions=1000,  # 1,000 partitions = 100 suppliers each
        agents_required=100,   # Deploy 100 worker agents
        behavior=SwarmBehavior.FORAGING,
        target_position=[50.0, 50.0, 50.0],  # Optimization target
        convergence_threshold=0.95,
        timeout_ms=300000,  # 5 minutes
        metadata={
            "task_type": "carbon_calculation",
            "data_source": "supplier_database",
            "emission_factors": "ghg_protocol_2024",
            "scope": "scope3_category1_purchased_goods"
        }
    )

    logger.info(f"Deploying swarm with {task.agents_required} agents...")
    logger.info(f"Objective: {task.objective}")
    logger.info(f"Data partitions: {task.data_partitions}")
    logger.info(f"Behavior: {task.behavior}")

    # Execute swarm task
    result = await swarm.deploy(task)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("SWARM EXECUTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Swarm ID: {result['swarm_id']}")
    logger.info(f"Convergence: {result['convergence']:.2%} (target: {task.convergence_threshold:.2%})")
    logger.info(f"Fitness: {result['fitness']:.2%}")
    logger.info(f"Iterations: {result['iterations']}")
    logger.info(f"Duration: {result['duration_ms']:.0f}ms ({result['duration_ms']/1000:.1f}s)")
    logger.info(f"Efficiency: {result['efficiency']:.2%}")
    logger.info(f"Center of Mass: {result['center_of_mass']}")
    logger.info(f"Spread: {result['spread']:.2f}")
    logger.info(f"Best Position: {result['best_position']}")
    logger.info(f"Best Fitness: {result['best_fitness']:.4f}")

    # Calculate throughput
    suppliers_processed = 100000
    duration_seconds = result['duration_ms'] / 1000
    throughput = suppliers_processed / duration_seconds if duration_seconds > 0 else 0

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Suppliers processed: {suppliers_processed:,}")
    logger.info(f"  Throughput: {throughput:.0f} suppliers/second")
    logger.info(f"  Average time per supplier: {(duration_seconds/suppliers_processed)*1000:.2f}ms")

    # Get swarm metrics
    metrics = await swarm.get_metrics()
    logger.info(f"\nSwarm Orchestrator Metrics:")
    logger.info(f"  Total agents in pool: {metrics['total_agents']}")
    logger.info(f"  Available agents: {metrics['available_agents']}")
    logger.info(f"  Active swarms: {metrics['active_swarms']}")
    logger.info(f"  Active tasks: {metrics['active_tasks']}")

    # Cleanup
    await swarm.shutdown()
    await message_bus.shutdown()

    logger.info("\n✓ Swarm orchestration example complete!\n")


# ============================================================================
# EXAMPLE 2: Dynamic Routing - Intelligent Load Balancing
# ============================================================================

async def example_routing_load_balancing():
    """
    Example: Intelligent routing of carbon calculation requests.

    Scenario:
    - 3 calculator agents with different loads
    - Route requests using least-loaded strategy
    - Demonstrate scatter-gather for consensus calculations
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Dynamic Routing - Load Balancing")
    logger.info("=" * 80)

    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.initialize()

    # Initialize router with route table
    route_table = RouteTable(
        routes={
            "calculator": ["agent-calc-001", "agent-calc-002", "agent-calc-003"],
            "validator": ["agent-validator-001", "agent-validator-002"]
        },
        cache_ttl_seconds=60,
        max_cache_size=10000
    )

    router = MessageRouter(message_bus, route_table)
    await router.initialize()

    # Simulate agent loads
    from routing import LoadInfo
    router.update_agent_load("agent-calc-001", LoadInfo(
        agent_id="agent-calc-001",
        message_queue_size=45,
        processing_time_ms=250,
        error_rate=0.02,
        capacity=100,
        weight=1.0
    ))
    router.update_agent_load("agent-calc-002", LoadInfo(
        agent_id="agent-calc-002",
        message_queue_size=15,  # Least loaded
        processing_time_ms=180,
        error_rate=0.01,
        capacity=100,
        weight=1.0
    ))
    router.update_agent_load("agent-calc-003", LoadInfo(
        agent_id="agent-calc-003",
        message_queue_size=78,
        processing_time_ms=420,
        error_rate=0.05,
        capacity=100,
        weight=1.0
    ))

    # Add content-based routing rule
    rule = RouteRule(
        name="route_carbon_calculations",
        priority=100,
        condition="payload.get('task_type') == 'carbon_calculation'",
        targets=["agent-calc-001", "agent-calc-002", "agent-calc-003"],
        strategy=RoutingStrategy.LEAST_LOADED
    )
    router.add_route_rule(rule)

    logger.info("Routing Rules Configured:")
    logger.info(f"  Rule: {rule.name}")
    logger.info(f"  Strategy: {rule.strategy}")
    logger.info(f"  Targets: {rule.targets}")

    logger.info("\nAgent Load Status:")
    logger.info(f"  agent-calc-001: Queue=45, Load={router.agent_loads['agent-calc-001'].load_factor:.2%}")
    logger.info(f"  agent-calc-002: Queue=15, Load={router.agent_loads['agent-calc-002'].load_factor:.2%}")
    logger.info(f"  agent-calc-003: Queue=78, Load={router.agent_loads['agent-calc-003'].load_factor:.2%}")

    # Route single message using least-loaded
    message = Message(
        sender_id="pipeline-001",
        recipient_id="calculator",
        message_type=MessageType.REQUEST,
        priority=Priority.HIGH,
        payload={
            "task_type": "carbon_calculation",
            "supplier_id": "SUP-12345",
            "category": "purchased_goods",
            "quantity": 5000,
            "unit": "kg"
        }
    )

    logger.info("\nRouting Request (Least-Loaded Strategy):")
    targets = await router.route(message, RoutingStrategy.LEAST_LOADED)
    logger.info(f"  Routed to: {targets[0]} (least loaded agent)")

    # Demonstrate round-robin
    logger.info("\nRouting 5 Requests (Round-Robin Strategy):")
    for i in range(5):
        msg = Message(
            sender_id=f"pipeline-{i:03d}",
            recipient_id="calculator",
            message_type=MessageType.REQUEST,
            priority=Priority.NORMAL,
            payload={"task_type": "carbon_calculation", "request_id": i}
        )
        targets = await router.route(msg, RoutingStrategy.ROUND_ROBIN)
        logger.info(f"  Request {i}: → {targets[0]}")

    # Scatter-Gather: Request calculation from all 3 agents, average result
    logger.info("\nScatter-Gather Pattern (Consensus Calculation):")
    logger.info("  Sending request to 3 calculator agents...")
    logger.info("  Aggregation strategy: AVERAGE")

    scatter = ScatterGather(router)

    # Note: In real scenario, agents would respond. This is demonstration.
    scatter_message = Message(
        sender_id="pipeline-consensus",
        recipient_id="calculator",
        message_type=MessageType.REQUEST,
        priority=Priority.HIGH,
        payload={
            "task_type": "carbon_calculation",
            "supplier_id": "SUP-67890",
            "require_consensus": True
        }
    )

    try:
        # This will timeout in demo as no real agents, but shows the pattern
        responses = await scatter.execute(
            request=scatter_message,
            target_agents=["agent-calc-001", "agent-calc-002", "agent-calc-003"],
            aggregation_strategy=AggregationStrategy.AVERAGE,
            timeout_ms=2000,
            min_responses=2
        )
        logger.info(f"  Responses received: {len(responses)}")
        logger.info(f"  Aggregated result: {responses}")
    except asyncio.TimeoutError:
        logger.info("  (Timeout expected in demo - no real agents running)")

    # Cleanup
    await router.shutdown()
    await message_bus.shutdown()

    logger.info("\n✓ Routing example complete!\n")


# ============================================================================
# EXAMPLE 3: Saga Transactions - CSRD Report Generation
# ============================================================================

async def example_saga_csrd_reporting():
    """
    Example: Multi-step CSRD report generation with automatic rollback.

    Scenario:
    - 5-step distributed transaction
    - Each step can fail and trigger compensation
    - Demonstrates pivot point (point of no return)
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Saga Transactions - CSRD Report Generation")
    logger.info("=" * 80)

    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.initialize()

    # Initialize saga orchestrator
    saga_config = SagaConfig(
        enable_persistence=True,
        enable_event_sourcing=True,
        max_concurrent_sagas=100,
        default_timeout_ms=600000,
        compensation_timeout_ms=60000
    )

    saga = SagaOrchestrator(message_bus, saga_config)
    await saga.initialize()

    # Define multi-step CSRD reporting transaction
    transaction = SagaTransaction(
        name="csrd_report_generation",
        compensation_strategy=CompensationStrategy.BACKWARD,
        timeout_ms=600000,  # 10 minutes overall
        steps=[
            SagaStep(
                name="collect_esg_data",
                agent_id="agent-data-collector",
                action="collect_data",
                compensation="delete_collected_data",
                timeout_ms=120000,
                retry_policy={"max_attempts": 3, "backoff_ms": 2000},
                metadata={
                    "frameworks": ["ESRS-E1", "ESRS-E2", "ESRS-S1"],
                    "reporting_period": "2024-Q4"
                }
            ),
            SagaStep(
                name="validate_completeness",
                agent_id="agent-validator",
                action="validate_framework_completeness",
                compensation="reset_validation_state",
                timeout_ms=60000,
                retry_policy={"max_attempts": 3, "backoff_ms": 2000},
                metadata={
                    "validation_rules": ["completeness", "materiality", "accuracy"]
                }
            ),
            SagaStep(
                name="calculate_metrics",
                agent_id="agent-calculator",
                action="calculate_csrd_metrics",
                compensation="clear_calculations",
                timeout_ms=180000,
                retry_policy={"max_attempts": 2, "backoff_ms": 5000},
                metadata={
                    "metrics": ["scope1_emissions", "scope2_emissions", "scope3_emissions"]
                }
            ),
            SagaStep(
                name="generate_report",
                agent_id="agent-reporter",
                action="generate_xbrl_report",
                compensation="delete_draft_report",
                timeout_ms=120000,
                is_pivot=True,  # Point of no return
                metadata={
                    "format": "XBRL",
                    "taxonomy": "ESEF"
                }
            ),
            SagaStep(
                name="publish_report",
                agent_id="agent-publisher",
                action="publish_to_esma",
                timeout_ms=60000,
                metadata={
                    "destination": "ESMA_FILING_SYSTEM",
                    "encryption": "required"
                }
                # No compensation after pivot point
            )
        ],
        metadata={
            "company_id": "LEI-1234567890ABCDEF",
            "reporting_year": 2024,
            "frameworks": ["CSRD", "ESRS"]
        }
    )

    logger.info("CSRD Reporting Saga Configuration:")
    logger.info(f"  Transaction: {transaction.name}")
    logger.info(f"  Steps: {len(transaction.steps)}")
    logger.info(f"  Compensation Strategy: {transaction.compensation_strategy}")
    logger.info(f"  Overall Timeout: {transaction.timeout_ms/1000:.0f}s")

    logger.info("\nExecution Plan:")
    for i, step in enumerate(transaction.steps, 1):
        pivot_marker = " [PIVOT - No compensation after]" if step.is_pivot else ""
        logger.info(f"  {i}. {step.name}{pivot_marker}")
        logger.info(f"     Agent: {step.agent_id}")
        logger.info(f"     Action: {step.action}")
        logger.info(f"     Compensation: {step.compensation or 'None'}")
        logger.info(f"     Timeout: {step.timeout_ms/1000:.0f}s")

    # Simulate transaction execution
    initial_data = {
        "company_id": "LEI-1234567890ABCDEF",
        "company_name": "GreenTech Industries AG",
        "reporting_period": "2024-Q4",
        "frameworks": ["CSRD"],
        "contact_email": "esg@greentech.example.com"
    }

    logger.info(f"\nInitial Data:")
    for key, value in initial_data.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "-" * 80)
    logger.info("Starting Saga Execution...")
    logger.info("-" * 80)

    try:
        # Execute saga (will timeout in demo as no real agents)
        result = await saga.execute(transaction, initial_data)

        logger.info("\n" + "=" * 80)
        logger.info("SAGA EXECUTION SUCCESSFUL")
        logger.info("=" * 80)
        logger.info(f"Report ID: {result.get('report_id', 'N/A')}")
        logger.info(f"Publication Status: {result.get('status', 'N/A')}")

    except asyncio.TimeoutError:
        logger.info("\n(Timeout expected in demo - no real agents running)")

        # Show what would happen in compensation
        logger.info("\nIn case of failure, compensation would execute:")
        logger.info("  Step 3 (calculate_metrics) → clear_calculations")
        logger.info("  Step 2 (validate_completeness) → reset_validation_state")
        logger.info("  Step 1 (collect_esg_data) → delete_collected_data")
        logger.info("  [Pivot point reached - Steps 4 & 5 would NOT be compensated]")

    except Exception as e:
        logger.error(f"Saga failed: {e}")

        # Check execution status
        execution = await saga.get_execution_status(transaction.transaction_id)
        if execution:
            logger.info(f"\nExecution Status: {execution.state}")
            logger.info(f"Completed Steps: {len(execution.completed_steps)}/{len(transaction.steps)}")
            logger.info(f"Failed Step: {execution.failed_step}")
            logger.info(f"Compensated Steps: {len(execution.compensated_steps)}")

            # Show event log
            logs = await saga.get_execution_logs(execution.execution_id)
            logger.info(f"\nEvent Log ({len(logs)} events):")
            for log in logs[:10]:  # Show first 10
                logger.info(f"  [{log.timestamp}] {log.event_type}")
                if log.step_name:
                    logger.info(f"    Step: {log.step_name}")

    # Get saga metrics
    metrics = await saga.get_metrics()
    logger.info(f"\nSaga Orchestrator Metrics:")
    logger.info(f"  Total Executions: {metrics['total_executions']}")
    logger.info(f"  Completed: {metrics['completed']}")
    logger.info(f"  Failed: {metrics['failed']}")
    logger.info(f"  Compensated: {metrics['compensated']}")
    logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
    logger.info(f"  Compensation Rate: {metrics['compensation_rate']:.2%}")

    # Cleanup
    await saga.shutdown()
    await message_bus.shutdown()

    logger.info("\n✓ Saga orchestration example complete!\n")


# ============================================================================
# EXAMPLE 4: Agent Registry - Service Discovery
# ============================================================================

async def example_registry_service_discovery():
    """
    Example: Register agents and discover them by capabilities.

    Scenario:
    - Register 10 different carbon calculator agents
    - Discover agents by specific capabilities
    - Monitor agent health
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Agent Registry - Service Discovery")
    logger.info("=" * 80)

    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.initialize()

    # Initialize registry
    registry = AgentRegistry(
        message_bus,
        heartbeat_interval_seconds=30,
        health_check_interval_seconds=60,
        deregistration_timeout_seconds=300
    )
    await registry.initialize()

    logger.info("Agent Registry Initialized")
    logger.info(f"  Heartbeat Interval: {registry.heartbeat_interval}s")
    logger.info(f"  Health Check Interval: {registry.health_check_interval}s")
    logger.info(f"  Deregistration Timeout: {registry.deregistration_timeout}s")

    # Register multiple calculator agents with different capabilities
    agents_to_register = [
        AgentDescriptor(
            agent_id="agent-carbon-calc-001",
            agent_type="CarbonCalculatorAgent",
            version="2.1.3",
            capabilities=["carbon_calculation", "scope3_emissions", "ghg_protocol"],
            service_types=[ServiceType.COMPUTATION],
            endpoint="tcp://10.0.1.50:5000",
            tags=["production", "high-performance", "us-east-1"],
            sla={"max_latency_ms": 500, "availability": 0.999},
            metadata={"max_concurrent": 100, "frameworks": ["GHG Protocol"]}
        ),
        AgentDescriptor(
            agent_id="agent-carbon-calc-002",
            agent_type="CarbonCalculatorAgent",
            version="2.1.3",
            capabilities=["carbon_calculation", "scope1_emissions", "scope2_emissions"],
            service_types=[ServiceType.COMPUTATION],
            endpoint="tcp://10.0.1.51:5000",
            tags=["production", "us-east-1"],
            sla={"max_latency_ms": 600, "availability": 0.995}
        ),
        AgentDescriptor(
            agent_id="agent-validator-001",
            agent_type="ValidationAgent",
            version="1.5.2",
            capabilities=["data_validation", "framework_validation", "csrd_compliance"],
            service_types=[ServiceType.VALIDATION],
            endpoint="tcp://10.0.1.52:5000",
            tags=["production", "compliance", "eu-west-1"],
            sla={"max_latency_ms": 1000, "availability": 0.999}
        ),
        AgentDescriptor(
            agent_id="agent-reporter-001",
            agent_type="ReportingAgent",
            version="3.0.1",
            capabilities=["report_generation", "xbrl_formatting", "pdf_export"],
            service_types=[ServiceType.REPORTING],
            endpoint="tcp://10.0.1.53:5000",
            tags=["production", "reporting", "us-east-1"]
        ),
        AgentDescriptor(
            agent_id="agent-carbon-calc-003",
            agent_type="CarbonCalculatorAgent",
            version="2.2.0",  # Newer version
            capabilities=["carbon_calculation", "scope3_emissions", "supplier_emissions", "ai_prediction"],
            service_types=[ServiceType.COMPUTATION, ServiceType.ANALYSIS],
            endpoint="tcp://10.0.1.54:5000",
            tags=["experimental", "ai-enabled", "us-west-2"],
            sla={"max_latency_ms": 800, "availability": 0.99}
        )
    ]

    logger.info(f"\nRegistering {len(agents_to_register)} agents...")
    for descriptor in agents_to_register:
        location = descriptor.tags[1] if len(descriptor.tags) > 1 else None
        success = await registry.register(descriptor, location=location)
        status = "✓" if success else "✗"
        logger.info(f"  {status} {descriptor.agent_id} ({descriptor.agent_type} v{descriptor.version})")

    # Simulate heartbeats with metrics
    logger.info("\nSimulating agent heartbeats...")
    await registry.heartbeat("agent-carbon-calc-001", {
        "cpu_usage": 0.45,
        "memory_usage": 0.62,
        "error_rate": 0.01,
        "response_time_ms": 234,
        "queue_size": 12
    })
    await registry.heartbeat("agent-carbon-calc-002", {
        "cpu_usage": 0.72,
        "memory_usage": 0.58,
        "error_rate": 0.03,
        "response_time_ms": 456,
        "queue_size": 34
    })
    await registry.heartbeat("agent-carbon-calc-003", {
        "cpu_usage": 0.28,
        "memory_usage": 0.41,
        "error_rate": 0.005,
        "response_time_ms": 189,
        "queue_size": 5
    })

    # Discovery Example 1: Find carbon calculators
    logger.info("\n" + "-" * 80)
    logger.info("DISCOVERY 1: Carbon calculation agents")
    logger.info("-" * 80)

    agents = await registry.discover(
        capabilities=["carbon_calculation"],
        min_health_score=0.7,
        max_results=10
    )

    logger.info(f"Found {len(agents)} agents with 'carbon_calculation' capability:")
    for agent in agents:
        logger.info(f"  • {agent.descriptor.agent_id}")
        logger.info(f"    Type: {agent.descriptor.agent_type} v{agent.descriptor.version}")
        logger.info(f"    Health: {agent.health_score:.2%}")
        logger.info(f"    Status: {agent.status}")
        logger.info(f"    Location: {agent.location}")
        logger.info(f"    Capabilities: {', '.join(agent.descriptor.capabilities)}")

    # Discovery Example 2: Find Scope 3 specialists
    logger.info("\n" + "-" * 80)
    logger.info("DISCOVERY 2: Scope 3 emission specialists")
    logger.info("-" * 80)

    agents = await registry.discover(
        capabilities=["scope3_emissions"],
        service_types=[ServiceType.COMPUTATION],
        min_health_score=0.8,
        max_results=5
    )

    logger.info(f"Found {len(agents)} high-health Scope 3 specialists:")
    for agent in agents:
        logger.info(f"  • {agent.descriptor.agent_id}")
        logger.info(f"    Health: {agent.health_score:.2%}")
        logger.info(f"    Endpoint: {agent.descriptor.endpoint}")

    # Discovery Example 3: Find by version constraint
    logger.info("\n" + "-" * 80)
    logger.info("DISCOVERY 3: Agents with version >= 2.2.0")
    logger.info("-" * 80)

    query = DiscoveryQuery(
        capabilities=["carbon_calculation"],
        version_constraint=">=2.2.0",
        max_results=10
    )

    agents = await registry.discovery.discover(registry, query)

    logger.info(f"Found {len(agents)} agents with version >= 2.2.0:")
    for agent in agents:
        logger.info(f"  • {agent.descriptor.agent_id} (v{agent.descriptor.version})")

    # Discovery Example 4: Location-aware discovery
    logger.info("\n" + "-" * 80)
    logger.info("DISCOVERY 4: Agents in us-east-1 region")
    logger.info("-" * 80)

    agents = await registry.discover(
        tags=["us-east-1"],
        min_health_score=0.5,
        max_results=10
    )

    logger.info(f"Found {len(agents)} agents in us-east-1:")
    for agent in agents:
        logger.info(f"  • {agent.descriptor.agent_id} ({agent.descriptor.agent_type})")

    # Get specific agent details
    logger.info("\n" + "-" * 80)
    logger.info("AGENT DETAILS")
    logger.info("-" * 80)

    agent = await registry.get_agent("agent-carbon-calc-001")
    if agent:
        logger.info(f"Agent: {agent.descriptor.agent_id}")
        logger.info(f"  Type: {agent.descriptor.agent_type}")
        logger.info(f"  Version: {agent.descriptor.version}")
        logger.info(f"  Status: {agent.status}")
        logger.info(f"  Health Score: {agent.health_score:.2%}")
        logger.info(f"  Registered: {agent.registered_at}")
        logger.info(f"  Last Heartbeat: {agent.last_heartbeat}")
        logger.info(f"  Metrics:")
        for key, value in agent.metrics.items():
            logger.info(f"    {key}: {value}")
        logger.info(f"  SLA:")
        for key, value in agent.descriptor.sla.items():
            logger.info(f"    {key}: {value}")

    # List all agents by type
    logger.info("\n" + "-" * 80)
    logger.info("AGENTS BY TYPE")
    logger.info("-" * 80)

    all_agents = await registry.list_agents()
    agent_types = {}
    for agent in all_agents:
        agent_type = agent.descriptor.agent_type
        agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

    for agent_type, count in agent_types.items():
        logger.info(f"  {agent_type}: {count} agents")

    # Registry metrics
    metrics = await registry.get_metrics()
    logger.info(f"\n" + "=" * 80)
    logger.info("REGISTRY METRICS")
    logger.info("=" * 80)
    logger.info(f"Total Agents: {metrics['total_agents']}")
    logger.info(f"Average Health Score: {metrics['average_health_score']:.2%}")
    logger.info(f"Unique Capabilities: {metrics['unique_capabilities']}")
    logger.info(f"Unique Tags: {metrics['unique_tags']}")
    logger.info(f"\nStatus Distribution:")
    for status, count in metrics['status_distribution'].items():
        logger.info(f"  {status}: {count}")

    # Cleanup
    await registry.shutdown()
    await message_bus.shutdown()

    logger.info("\n✓ Registry example complete!\n")


# ============================================================================
# EXAMPLE 5: Integrated Workflow - Complete Pipeline
# ============================================================================

async def example_integrated_workflow():
    """
    Example: Complete integrated workflow using all orchestration components.

    Scenario:
    - Use registry to discover agents
    - Use router to distribute work
    - Use swarm for parallel processing
    - Use saga to coordinate multi-step transaction
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Integrated Workflow - Complete Pipeline")
    logger.info("=" * 80)

    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.initialize()

    # Initialize all orchestration components
    logger.info("Initializing orchestration components...")

    registry = AgentRegistry(message_bus)
    router = MessageRouter(message_bus)
    swarm = SwarmOrchestrator(message_bus)
    saga = SagaOrchestrator(message_bus)

    await registry.initialize()
    await router.initialize()
    await swarm.initialize()
    await saga.initialize()

    logger.info("✓ All components initialized\n")

    # Step 1: Register agents
    logger.info("Step 1: Registering agents in registry...")
    calculator_agent = AgentDescriptor(
        agent_id="agent-calculator-main",
        agent_type="CarbonCalculatorAgent",
        version="2.1.0",
        capabilities=["carbon_calculation", "scope3_emissions"],
        service_types=[ServiceType.COMPUTATION],
        endpoint="tcp://localhost:5001"
    )
    await registry.register(calculator_agent)
    logger.info("✓ Calculator agent registered")

    # Step 2: Discover agents for routing
    logger.info("\nStep 2: Discovering calculation agents...")
    agents = await registry.discover(
        capabilities=["carbon_calculation"],
        min_health_score=0.5
    )
    logger.info(f"✓ Found {len(agents)} calculator agents")

    # Step 3: Configure routing
    logger.info("\nStep 3: Configuring intelligent routing...")
    rule = RouteRule(
        name="route_calculations",
        priority=100,
        condition="payload.get('task') == 'calculate'",
        targets=[a.descriptor.agent_id for a in agents],
        strategy=RoutingStrategy.LEAST_LOADED
    )
    router.add_route_rule(rule)
    logger.info("✓ Routing rules configured")

    # Step 4: Deploy swarm for distributed processing
    logger.info("\nStep 4: Deploying calculation swarm...")
    task = SwarmTask(
        objective="Process supplier emissions",
        data_partitions=100,
        agents_required=10,
        behavior=SwarmBehavior.FORAGING,
        convergence_threshold=0.90,
        timeout_ms=60000
    )
    logger.info(f"✓ Swarm task configured ({task.agents_required} agents)")

    # Step 5: Execute saga transaction
    logger.info("\nStep 5: Executing multi-step saga transaction...")
    transaction = SagaTransaction(
        name="integrated_calculation_workflow",
        steps=[
            SagaStep(
                name="data_collection",
                agent_id="agent-calculator-main",
                action="collect",
                compensation="cleanup_data"
            ),
            SagaStep(
                name="calculation",
                agent_id="agent-calculator-main",
                action="calculate",
                compensation="reset_calculations"
            ),
            SagaStep(
                name="validation",
                agent_id="agent-calculator-main",
                action="validate",
                compensation="clear_validation"
            )
        ]
    )
    logger.info(f"✓ Saga configured ({len(transaction.steps)} steps)")

    # Show integration
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATED WORKFLOW SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Registry: {len(await registry.list_agents())} agents registered")
    logger.info(f"Router: {len(router.route_table.rules)} routing rules active")
    logger.info(f"Swarm: Ready to deploy {task.agents_required} agents")
    logger.info(f"Saga: {len(transaction.steps)}-step transaction configured")

    logger.info("\nWorkflow execution would:")
    logger.info("  1. Discover available agents via Registry")
    logger.info("  2. Route requests to optimal agents via Router")
    logger.info("  3. Deploy Swarm for parallel processing")
    logger.info("  4. Coordinate steps via Saga with auto-rollback")
    logger.info("  5. Monitor health and adjust routing dynamically")

    # Cleanup
    await saga.shutdown()
    await swarm.shutdown()
    await router.shutdown()
    await registry.shutdown()
    await message_bus.shutdown()

    logger.info("\n✓ Integrated workflow example complete!\n")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for running examples."""
    parser = argparse.ArgumentParser(description="Orchestration Components Usage Examples")
    parser.add_argument(
        "--example",
        choices=["swarm", "routing", "saga", "registry", "integrated", "all"],
        default="all",
        help="Which example to run (default: all)"
    )

    args = parser.parse_args()

    examples = {
        "swarm": example_swarm_carbon_calculation,
        "routing": example_routing_load_balancing,
        "saga": example_saga_csrd_reporting,
        "registry": example_registry_service_discovery,
        "integrated": example_integrated_workflow
    }

    if args.example == "all":
        logger.info("Running all examples...\n")
        for name, func in examples.items():
            try:
                await func()
            except Exception as e:
                logger.error(f"Example {name} failed: {e}", exc_info=True)
    else:
        await examples[args.example]()

    logger.info("=" * 80)
    logger.info("ALL EXAMPLES COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
