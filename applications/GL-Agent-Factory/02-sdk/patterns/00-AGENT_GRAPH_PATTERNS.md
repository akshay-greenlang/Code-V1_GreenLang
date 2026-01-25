# Agent Graph Patterns - Standard Composition Patterns

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Specification

## Executive Summary

This document defines standard agent graph patterns for composing multi-agent workflows in the GreenLang Agent SDK. These patterns, extracted from production implementations (GL-001 through GL-007), provide reusable architectures for building complex agent systems with well-defined data flow, error handling, and orchestration strategies.

---

## Table of Contents

1. [Pattern Overview](#pattern-overview)
2. [Linear Pipeline Pattern](#linear-pipeline-pattern)
3. [Parallel Processing Pattern](#parallel-processing-pattern)
4. [Conditional Routing Pattern](#conditional-routing-pattern)
5. [Hierarchical Orchestration Pattern](#hierarchical-orchestration-pattern)
6. [Event-Driven Pattern](#event-driven-pattern)
7. [Feedback Loop Pattern](#feedback-loop-pattern)
8. [Real-World Examples](#real-world-examples)

---

## Pattern Overview

### Why Agent Graphs?

Agent graphs enable:
- **Separation of Concerns**: Each agent has single responsibility
- **Reusability**: Agents can be reused in multiple workflows
- **Scalability**: Agents can run in parallel or distributed
- **Testability**: Each agent can be tested independently
- **Maintainability**: Changes to one agent don't affect others

### Pattern Categories

| Pattern | Use Case | Complexity | Examples |
|---------|----------|------------|----------|
| **Linear Pipeline** | Sequential processing | Low | Intake → Validate → Calculate → Report |
| **Parallel Processing** | Independent operations | Medium | Multi-scope emissions calculation |
| **Conditional Routing** | Dynamic agent selection | Medium | Route by fuel type, region, or framework |
| **Hierarchical Orchestration** | Master-slave coordination | High | GL-001 ProcessHeatOrchestrator |
| **Event-Driven** | Reactive workflows | High | Real-time monitoring and alerting |
| **Feedback Loop** | Iterative optimization | High | Boiler efficiency optimization |

---

## Linear Pipeline Pattern

### Description

Sequential execution of agents where output of one agent becomes input to the next. This is the most common pattern for data processing workflows.

### Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  IntakeAgent │ -> │ValidationAgent│ -> │CalculationAgent│ -> │ReportingAgent│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     Input              Validated           Calculated           Report
     Data                 Data                 Results            Output
```

### Implementation

```python
from greenlang_sdk.patterns import LinearPipeline
from greenlang_sdk.base import SDKAgentBase

class IntakeAgent(SDKAgentBase[RawInput, CleanedInput]):
    """Intake and clean raw data."""

    def execute_impl(self, input: RawInput, context) -> CleanedInput:
        # Clean and normalize data
        cleaned = self.use_tool("data_normalizer", {"data": input.raw_data})
        return CleanedInput(**cleaned.data)


class ValidationAgent(SDKAgentBase[CleanedInput, ValidatedInput]):
    """Validate cleaned data against schema."""

    def execute_impl(self, input: CleanedInput, context) -> ValidatedInput:
        # Validate schema and constraints
        result = self.use_tool("data_validator", {"data": input.dict()})

        if not result.data["is_valid"]:
            raise ValidationError(result.data["errors"])

        return ValidatedInput(**input.dict())


class CalculationAgent(SDKAgentBase[ValidatedInput, CalculationResult]):
    """Calculate emissions from validated data."""

    def execute_impl(self, input: ValidatedInput, context) -> CalculationResult:
        # Get emission factor
        ef = self.get_emission_factor(input.fuel_type, region=input.region)

        # Calculate emissions
        result = self.use_tool(
            "emissions_calculator",
            {
                "activity_data": input.amount,
                "emission_factor": ef["value"],
                "unit": input.unit
            }
        )

        return CalculationResult(
            emissions_kg_co2e=result.data["co2e_emissions_kg"],
            provenance_hash=result.provenance["output_hash"]
        )


class ReportingAgent(SDKAgentBase[CalculationResult, FinalReport]):
    """Generate final report."""

    def execute_impl(self, input: CalculationResult, context) -> FinalReport:
        # Map to framework
        framework_data = self.map_to_framework(
            input.dict(),
            framework="GRI_305"
        )

        return FinalReport(
            emissions=input.emissions_kg_co2e,
            framework_data=framework_data,
            provenance_hash=input.provenance_hash
        )


# Compose pipeline
pipeline = LinearPipeline([
    IntakeAgent(),
    ValidationAgent(),
    CalculationAgent(),
    ReportingAgent()
])

# Execute
result = pipeline.run(RawInput(raw_data={"amount": 1000, "fuel": "natural_gas"}))
```

### Pipeline Implementation

```python
class LinearPipeline:
    """Linear agent pipeline with error handling."""

    def __init__(self, agents: List[SDKAgentBase]):
        self.agents = agents

    def run(self, initial_input: Any) -> AgentResult:
        """Execute pipeline sequentially."""
        current_input = initial_input
        results = []

        for i, agent in enumerate(self.agents):
            try:
                logger.info(f"Executing agent {i+1}/{len(self.agents)}: {agent.agent_id}")

                # Execute agent
                result = agent.run(current_input)

                if not result.success:
                    logger.error(f"Agent {agent.agent_id} failed: {result.error}")
                    return result

                # Store result
                results.append(result)

                # Use output as input for next agent
                current_input = result.data

            except Exception as e:
                logger.error(f"Pipeline failed at agent {agent.agent_id}: {e}", exc_info=True)
                return AgentResult(
                    success=False,
                    error=f"Pipeline failed at {agent.agent_id}: {str(e)}",
                    metadata={"failed_at_agent": i, "partial_results": results}
                )

        # Return final result
        return results[-1]
```

### Use Cases

- **Data Processing**: Intake → Validate → Transform → Output
- **Emissions Calculation**: Collect → Validate → Calculate → Report
- **Compliance Reporting**: Gather → Validate → Map to Framework → Generate Report

---

## Parallel Processing Pattern

### Description

Execute multiple agents in parallel for independent operations, then aggregate results. This pattern is ideal for batch processing and multi-scope calculations.

### Architecture

```
                     ┌──────────────────┐
                     │  OrchestorAgent   │
                     └────────┬─────────┘
                              │ Fan-out
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼─────────┐ ┌──▼──────────┐ ┌──▼──────────┐
    │  Scope1Calculator │ │Scope2Calculator│ │Scope3Calculator│
    └─────────┬─────────┘ └──┬──────────┘ └──┬──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │ Fan-in
                     ┌────────▼─────────┐
                     │ AggregatorAgent  │
                     └──────────────────┘
```

### Implementation

```python
from greenlang_sdk.patterns import ParallelProcessor
from typing import List
import asyncio

class Scope1Calculator(SDKAgentBase[ScopeInput, ScopeResult]):
    """Calculate Scope 1 emissions."""

    def execute_impl(self, input: ScopeInput, context) -> ScopeResult:
        # Calculate direct emissions
        emissions = self.calculate_direct_emissions(input)
        return ScopeResult(scope="scope1", emissions=emissions)


class Scope2Calculator(SDKAgentBase[ScopeInput, ScopeResult]):
    """Calculate Scope 2 emissions."""

    def execute_impl(self, input: ScopeInput, context) -> ScopeResult:
        # Calculate indirect emissions from electricity
        emissions = self.calculate_indirect_emissions(input)
        return ScopeResult(scope="scope2", emissions=emissions)


class Scope3Calculator(SDKAgentBase[ScopeInput, ScopeResult]):
    """Calculate Scope 3 emissions."""

    def execute_impl(self, input: ScopeInput, context) -> ScopeResult:
        # Calculate value chain emissions
        emissions = self.calculate_value_chain_emissions(input)
        return ScopeResult(scope="scope3", emissions=emissions)


class AggregatorAgent(SDKAgentBase[List[ScopeResult], TotalEmissions]):
    """Aggregate emissions from all scopes."""

    def execute_impl(self, input: List[ScopeResult], context) -> TotalEmissions:
        total = sum(result.emissions for result in input)

        return TotalEmissions(
            scope1=next(r.emissions for r in input if r.scope == "scope1"),
            scope2=next(r.emissions for r in input if r.scope == "scope2"),
            scope3=next(r.emissions for r in input if r.scope == "scope3"),
            total=total
        )


# Compose parallel processor
processor = ParallelProcessor(
    parallel_agents=[
        Scope1Calculator(),
        Scope2Calculator(),
        Scope3Calculator()
    ],
    aggregator=AggregatorAgent()
)

# Execute
result = processor.run(ScopeInput(data=emissions_data))
```

### Parallel Processor Implementation

```python
class ParallelProcessor:
    """Execute agents in parallel and aggregate results."""

    def __init__(
        self,
        parallel_agents: List[SDKAgentBase],
        aggregator: Optional[SDKAgentBase] = None
    ):
        self.parallel_agents = parallel_agents
        self.aggregator = aggregator

    async def run_async(self, input: Any) -> AgentResult:
        """Execute agents in parallel asynchronously."""
        # Create tasks for all agents
        tasks = [
            asyncio.create_task(self._run_agent_async(agent, input))
            for agent in self.parallel_agents
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.error(f"Parallel execution had {len(failures)} failures")
            return AgentResult(
                success=False,
                error=f"{len(failures)} agents failed",
                metadata={"failures": [str(f) for f in failures]}
            )

        # Aggregate results if aggregator provided
        if self.aggregator:
            aggregated = self.aggregator.run([r.data for r in results])
            return aggregated

        # Return all results
        return AgentResult(
            success=True,
            data={"results": [r.data for r in results]},
            metadata={"agent_count": len(results)}
        )

    def run(self, input: Any) -> AgentResult:
        """Synchronous wrapper for async execution."""
        return asyncio.run(self.run_async(input))

    async def _run_agent_async(self, agent: SDKAgentBase, input: Any) -> AgentResult:
        """Run single agent asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.run, input)
```

### Use Cases

- **Multi-Scope Emissions**: Calculate Scope 1/2/3 in parallel
- **Multi-Region Reporting**: Generate reports for multiple regions simultaneously
- **Batch Processing**: Process multiple facilities/sites in parallel
- **Multi-Framework Mapping**: Map to GRI/SASB/TCFD/CDP in parallel

---

## Conditional Routing Pattern

### Description

Route input to different agents based on conditions. This pattern enables dynamic workflow execution based on input characteristics.

### Architecture

```
                     ┌──────────────┐
                     │ RouterAgent  │
                     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
     If fuel_type=gas   If fuel_type=coal  If fuel_type=oil
              │             │             │
    ┌─────────▼─────┐ ┌─────▼──────┐ ┌────▼───────┐
    │  GasCalculator│ │CoalCalculator│ │OilCalculator│
    └───────────────┘ └────────────┘ └────────────┘
```

### Implementation

```python
from greenlang_sdk.patterns import ConditionalRouter
from typing import Callable

class GasCalculator(SDKAgentBase[FuelInput, EmissionsResult]):
    """Calculate emissions for natural gas."""
    def execute_impl(self, input: FuelInput, context) -> EmissionsResult:
        # Gas-specific calculation
        pass

class CoalCalculator(SDKAgentBase[FuelInput, EmissionsResult]):
    """Calculate emissions for coal."""
    def execute_impl(self, input: FuelInput, context) -> EmissionsResult:
        # Coal-specific calculation
        pass

class OilCalculator(SDKAgentBase[FuelInput, EmissionsResult]):
    """Calculate emissions for oil."""
    def execute_impl(self, input: FuelInput, context) -> EmissionsResult:
        # Oil-specific calculation
        pass


# Define routing function
def route_by_fuel_type(input: FuelInput) -> str:
    """Route based on fuel type."""
    fuel_map = {
        "natural_gas": "gas",
        "coal": "coal",
        "diesel": "oil",
        "gasoline": "oil",
    }
    return fuel_map.get(input.fuel_type, "default")


# Compose router
router = ConditionalRouter(
    agents={
        "gas": GasCalculator(),
        "coal": CoalCalculator(),
        "oil": OilCalculator(),
    },
    router_func=route_by_fuel_type,
    default_agent=GasCalculator()  # Fallback
)

# Execute
result = router.run(FuelInput(fuel_type="natural_gas", amount=1000))
```

### Conditional Router Implementation

```python
class ConditionalRouter:
    """Route input to appropriate agent based on condition."""

    def __init__(
        self,
        agents: Dict[str, SDKAgentBase],
        router_func: Callable[[Any], str],
        default_agent: Optional[SDKAgentBase] = None
    ):
        self.agents = agents
        self.router_func = router_func
        self.default_agent = default_agent

    def run(self, input: Any) -> AgentResult:
        """Route input to appropriate agent."""
        try:
            # Determine route
            route_key = self.router_func(input)
            logger.info(f"Routing to: {route_key}")

            # Get agent
            agent = self.agents.get(route_key, self.default_agent)

            if agent is None:
                return AgentResult(
                    success=False,
                    error=f"No agent found for route '{route_key}' and no default agent"
                )

            # Execute agent
            return agent.run(input)

        except Exception as e:
            logger.error(f"Routing failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=f"Routing failed: {str(e)}"
            )
```

### Use Cases

- **Fuel-Type Routing**: Different calculations for gas/coal/oil
- **Region Routing**: Different EF databases for different regions
- **Framework Routing**: Route to GRI/SASB/TCFD based on report type
- **Complexity Routing**: Simple vs. detailed calculation methods

---

## Hierarchical Orchestration Pattern

### Description

Master orchestrator coordinates multiple sub-agents with complex dependencies. This is the pattern used in GL-001 ProcessHeatOrchestrator.

### Architecture

```
                    ┌────────────────────────┐
                    │   Master Orchestrator  │
                    │      (GL-001)          │
                    └────────────┬───────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
  ┌─────▼──────┐         ┌──────▼──────┐        ┌───────▼────────┐
  │  Boiler    │         │ Combustion  │        │  Heat Recovery │
  │Coordinator │         │ Coordinator │        │  Coordinator   │
  └─────┬──────┘         └──────┬──────┘        └───────┬────────┘
        │                       │                        │
  ┌─────┼─────┐          ┌──────┼──────┐         ┌──────┼──────┐
  │     │     │          │      │      │         │      │      │
GL-002 GL-003 GL-004  GL-021 GL-022 GL-023   GL-041 GL-042 GL-043
```

### Implementation

```python
from greenlang_sdk.patterns import HierarchicalOrchestrator

class BoilerCoordinator(SDKAgentBase[BoilerInput, BoilerResult]):
    """Coordinate boiler-related agents."""

    def __init__(self):
        super().__init__()
        self.sub_agents = [
            BoilerEfficiencyAgent(),      # GL-002
            SteamQualityAgent(),          # GL-003
            FeedwaterTreatmentAgent(),    # GL-004
        ]

    def execute_impl(self, input: BoilerInput, context) -> BoilerResult:
        results = []

        # Execute sub-agents in sequence
        for agent in self.sub_agents:
            result = agent.run(input)
            if not result.success:
                logger.error(f"Sub-agent {agent.agent_id} failed")
                # Continue with other agents or fail fast based on criticality
            results.append(result)

        # Aggregate results
        return BoilerResult(
            efficiency=results[0].data["efficiency"],
            steam_quality=results[1].data["quality"],
            feedwater_quality=results[2].data["quality"]
        )


class MasterOrchestrator(SDKAgentBase[OrchestratorInput, OrchestratorResult]):
    """Master orchestrator (GL-001)."""

    def __init__(self):
        super().__init__()
        self.coordinators = {
            "boiler": BoilerCoordinator(),
            "combustion": CombustionCoordinator(),
            "recovery": HeatRecoveryCoordinator(),
        }

    def execute_impl(self, input: OrchestratorInput, context) -> OrchestratorResult:
        # Determine which coordinators to run based on input
        active_coordinators = self._determine_active_coordinators(input)

        # Execute coordinators in parallel or sequence
        results = {}
        for name in active_coordinators:
            coordinator = self.coordinators[name]
            result = coordinator.run(input)
            results[name] = result

        # Perform system-level optimization
        optimized = self._optimize_system(results)

        return OrchestratorResult(
            coordinator_results=results,
            system_optimization=optimized,
            total_savings=self._calculate_savings(results, optimized)
        )
```

### Orchestrator Implementation

```python
class HierarchicalOrchestrator:
    """Hierarchical orchestration with dependency management."""

    def __init__(
        self,
        master: SDKAgentBase,
        coordinators: Dict[str, SDKAgentBase],
        dependencies: Optional[Dict[str, List[str]]] = None
    ):
        self.master = master
        self.coordinators = coordinators
        self.dependencies = dependencies or {}

    def run(self, input: Any) -> AgentResult:
        """Execute hierarchical orchestration."""
        # Build execution graph
        execution_graph = self._build_execution_graph()

        # Execute in topological order
        results = {}
        for node in execution_graph:
            agent = self.coordinators.get(node, self.master)

            # Wait for dependencies
            dep_results = {
                dep: results[dep]
                for dep in self.dependencies.get(node, [])
            }

            # Execute with dependency results
            result = agent.run({**input, "dependencies": dep_results})
            results[node] = result

        # Execute master orchestrator
        final_result = self.master.run({**input, "coordinator_results": results})
        return final_result

    def _build_execution_graph(self) -> List[str]:
        """Build topologically sorted execution graph."""
        # Topological sort implementation
        pass
```

### Use Cases

- **Process Heat Orchestration**: GL-001 coordinating 99 sub-agents
- **Multi-Facility Management**: Corporate-level orchestration of sites
- **Supply Chain Emissions**: Orchestrate upstream/downstream calculations
- **Integrated Reporting**: Coordinate multiple framework reports

---

## Event-Driven Pattern

### Description

Agents react to events published to a message bus. This pattern enables loose coupling and real-time processing.

### Architecture

```
                      ┌─────────────────┐
                      │   Message Bus   │
                      └────────┬────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────▼──────┐      ┌──────▼─────┐      ┌──────▼─────┐
    │  Subscriber│      │ Subscriber │      │ Subscriber │
    │  Agent 1   │      │  Agent 2   │      │  Agent 3   │
    └────────────┘      └────────────┘      └────────────┘
     (Monitoring)        (Alerting)         (Logging)
```

### Implementation

```python
from greenlang_sdk.patterns import EventBus, EventSubscriber

class EmissionsMonitoringAgent(EventSubscriber):
    """Monitor emissions and publish alerts."""

    def on_event(self, event: Event) -> None:
        if event.type == "emissions_reading":
            # Check threshold
            if event.data["value"] > self.threshold:
                self.publish_event(Event(
                    type="emissions_alert",
                    data={"value": event.data["value"], "threshold": self.threshold}
                ))


class AlertingAgent(EventSubscriber):
    """Send alerts based on events."""

    def on_event(self, event: Event) -> None:
        if event.type == "emissions_alert":
            self._send_alert(event.data)


# Create event bus
event_bus = EventBus()

# Register subscribers
event_bus.subscribe("emissions_reading", EmissionsMonitoringAgent())
event_bus.subscribe("emissions_alert", AlertingAgent())

# Publish event
event_bus.publish(Event(
    type="emissions_reading",
    data={"value": 150, "unit": "kgCO2e"}
))
```

### Use Cases

- **Real-Time Monitoring**: Monitor emissions, energy, safety metrics
- **Alert Management**: Threshold-based alerting and escalation
- **Audit Logging**: Log all agent executions for audit trail
- **Integration Events**: React to external system events (SCADA, ERP)

---

## Feedback Loop Pattern

### Description

Iterative optimization where agent output feeds back as input for next iteration. This pattern is used in optimization and control applications.

### Architecture

```
    ┌──────────────────────────────────────────┐
    │                                          │
    │  ┌──────────┐    ┌──────────┐    ┌────▼─────┐
    └─►│ Optimizer│ -> │ Validator│ -> │Convergence│
       │  Agent   │    │  Agent   │    │  Check    │
       └──────────┘    └──────────┘    └──────────┘
```

### Implementation

```python
class OptimizerAgent(SDKAgentBase[OptimizerInput, OptimizerOutput]):
    """Optimize boiler parameters."""

    def execute_impl(self, input: OptimizerInput, context) -> OptimizerOutput:
        # Optimize parameters using tool
        result = self.use_tool(
            "optimizer",
            {
                "current_params": input.parameters,
                "objective": "minimize_fuel_consumption",
                "constraints": input.constraints
            }
        )

        return OptimizerOutput(
            optimized_params=result.data["parameters"],
            predicted_savings=result.data["savings"]
        )


class FeedbackLoop:
    """Feedback loop for iterative optimization."""

    def __init__(
        self,
        optimizer: SDKAgentBase,
        validator: SDKAgentBase,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01
    ):
        self.optimizer = optimizer
        self.validator = validator
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def run(self, initial_input: Any) -> AgentResult:
        """Run feedback loop until convergence."""
        current_input = initial_input
        iteration = 0
        history = []

        while iteration < self.max_iterations:
            # Optimize
            opt_result = self.optimizer.run(current_input)

            # Validate
            val_result = self.validator.run(opt_result.data)

            # Check convergence
            improvement = self._calculate_improvement(history, opt_result)
            history.append(opt_result)

            if improvement < self.convergence_threshold:
                logger.info(f"Converged after {iteration+1} iterations")
                break

            # Prepare next iteration
            current_input = opt_result.data
            iteration += 1

        return AgentResult(
            success=True,
            data=history[-1].data,
            metadata={
                "iterations": iteration + 1,
                "convergence_history": [h.data for h in history]
            }
        )
```

### Use Cases

- **Boiler Optimization**: Iteratively optimize combustion parameters
- **Control Systems**: Model predictive control
- **Machine Learning**: Iterative model training and validation
- **Multi-Objective Optimization**: Balance efficiency, emissions, cost

---

## Real-World Examples

### Example 1: GL-001 ProcessHeatOrchestrator

**Pattern**: Hierarchical Orchestration

```python
class GL001ProcessHeatOrchestrator(SDKAgentBase):
    """Master orchestrator for process heat (99 sub-agents)."""

    def __init__(self):
        super().__init__()

        # Domain coordinators
        self.coordinators = {
            "boiler_steam": [GL002(), GL003(), GL004(), ...],  # 9 agents
            "combustion": [GL021(), GL022(), GL023(), ...],     # 8 agents
            "heat_recovery": [GL041(), GL042(), GL043(), ...],  # 8 agents
            "maintenance": [GL061(), GL062(), GL063(), ...],    # 7 agents
            "analytics": [GL071(), GL072(), GL073(), ...],      # 8 agents
            "decarbonization": [GL081(), GL082(), GL083(), ...] # 8 agents
        }

    def execute_impl(self, input: OrchInput, context) -> OrchResult:
        # Phase 1: Data collection (parallel)
        data_collection_results = self._run_parallel(
            [coord[0] for coord in self.coordinators.values()],  # Intake agents
            input
        )

        # Phase 2: Analysis (parallel by domain)
        analysis_results = {}
        for domain, agents in self.coordinators.items():
            results = self._run_domain_analysis(domain, agents, data_collection_results)
            analysis_results[domain] = results

        # Phase 3: System optimization
        optimization = self.use_tool(
            "multi_agent_coordinator",
            {
                "domain_results": analysis_results,
                "objective": "minimize_total_cost"
            }
        )

        # Phase 4: Reporting
        report = self._generate_master_report(analysis_results, optimization)

        return OrchResult(
            domain_results=analysis_results,
            optimization=optimization.data,
            report=report
        )
```

### Example 2: Multi-Framework Reporting Pipeline

**Pattern**: Linear Pipeline + Parallel Processing

```python
# Stage 1: Intake (Linear)
intake_pipeline = LinearPipeline([
    DataCollectionAgent(),
    DataValidationAgent(),
    DataNormalizationAgent()
])

# Stage 2: Calculations (Parallel)
calculation_processor = ParallelProcessor([
    Scope1Calculator(),
    Scope2Calculator(),
    Scope3Calculator(),
    EnergyMetricsCalculator(),
    WaterMetricsCalculator()
])

# Stage 3: Framework Mapping (Parallel)
framework_processor = ParallelProcessor([
    GRIMapper(),
    SASBMapper(),
    TCFDMapper(),
    CDPMapper()
])

# Stage 4: Report Generation (Linear)
reporting_pipeline = LinearPipeline([
    ReportAggregatorAgent(),
    ReportFormatterAgent(),
    ReportValidatorAgent()
])

# Compose complete workflow
workflow = LinearPipeline([
    intake_pipeline,
    calculation_processor,
    framework_processor,
    reporting_pipeline
])

# Execute
result = workflow.run(RawReportingData(year=2025))
```

### Example 3: Real-Time Emissions Monitoring

**Pattern**: Event-Driven

```python
# Create event bus
event_bus = EventBus()

# Data ingestion agent (publishes events)
class DataIngestionAgent(EventSubscriber):
    def run(self):
        while True:
            # Read from SCADA
            data = self.scada_connector.read_tags(["emissions", "flow_rate"])

            # Publish event
            event_bus.publish(Event(
                type="emissions_reading",
                data=data,
                timestamp=datetime.now()
            ))

            time.sleep(1)  # 1 Hz sampling

# Monitoring agents (subscribe to events)
event_bus.subscribe("emissions_reading", EmissionsCalculatorAgent())
event_bus.subscribe("emissions_reading", ThresholdMonitorAgent())
event_bus.subscribe("emissions_reading", DataLoggerAgent())

# Alerting agents (subscribe to alerts)
event_bus.subscribe("emissions_alert", EmailAlertAgent())
event_bus.subscribe("emissions_alert", SMSAlertAgent())
event_bus.subscribe("emissions_alert", DashboardUpdateAgent())

# Start monitoring
data_ingestion = DataIngestionAgent()
data_ingestion.start()
```

---

## Pattern Selection Guide

### Decision Matrix

| Requirement | Recommended Pattern |
|-------------|-------------------|
| Sequential data processing | Linear Pipeline |
| Independent parallel tasks | Parallel Processing |
| Dynamic workflow routing | Conditional Routing |
| Complex multi-agent coordination | Hierarchical Orchestration |
| Real-time event processing | Event-Driven |
| Iterative optimization | Feedback Loop |

### Pattern Combination

Patterns can be combined for complex workflows:

```python
# Example: Hierarchical + Parallel + Event-Driven
orchestrator = HierarchicalOrchestrator(
    master=MasterAgent(),
    coordinators={
        "domain1": ParallelProcessor([Agent1(), Agent2(), Agent3()]),
        "domain2": LinearPipeline([Agent4(), Agent5()]),
        "domain3": EventDrivenProcessor(event_bus)
    }
)
```

---

## Best Practices

### Design Guidelines

1. **Single Responsibility**: Each agent should have one clear purpose
2. **Loose Coupling**: Agents should not depend on implementation details of other agents
3. **Error Isolation**: Errors in one agent should not crash the entire workflow
4. **Observable**: All agent executions should be logged and traceable
5. **Testable**: Each agent and pattern should be independently testable

### Performance Considerations

- **Parallel Processing**: Use for I/O-bound operations, be careful with CPU-bound
- **Event-Driven**: Consider message queue capacity and backpressure
- **Feedback Loop**: Set maximum iterations to prevent infinite loops
- **Hierarchical**: Monitor memory usage with many sub-agents

### Error Handling

```python
# Fail fast vs. continue
if critical_agent:
    if not result.success:
        raise AgentExecutionError(f"Critical agent failed: {agent.agent_id}")
else:
    if not result.success:
        logger.warning(f"Non-critical agent failed: {agent.agent_id}")
        # Continue with other agents
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-03
**Author**: GL-BackendDeveloper
**Status**: Specification - Ready for Implementation
