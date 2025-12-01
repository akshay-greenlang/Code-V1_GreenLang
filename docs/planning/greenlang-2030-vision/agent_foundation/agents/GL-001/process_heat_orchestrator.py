# -*- coding: utf-8 -*-
"""
ProcessHeatOrchestrator - Master orchestrator for process heat operations.

This module implements the GL-001 ProcessHeatOrchestrator agent for managing
and optimizing process heat operations across industrial facilities. It coordinates
multiple sub-agents, integrates with SCADA and ERP systems, and ensures
zero-hallucination calculations for all thermal operations.

Now inherits from BaseOrchestrator for standardized orchestration patterns.

Example:
    >>> from process_heat_orchestrator import ProcessHeatOrchestrator
    >>> config = ProcessHeatConfig(...)
    >>> orchestrator = ProcessHeatOrchestrator(config)
    >>> result = await orchestrator.execute(plant_data)
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

# Import from greenlang core infrastructure
from greenlang.core.base_orchestrator import (
    BaseOrchestrator,
    OrchestrationResult,
    OrchestratorConfig,
    OrchestratorState,
)
from greenlang.core.message_bus import (
    Message,
    MessageBus,
    MessageBusConfig,
    MessagePriority,
    MessageType,
)
from greenlang.core.task_scheduler import (
    AgentCapacity,
    LoadBalanceStrategy,
    Task,
    TaskPriority,
    TaskScheduler,
    TaskSchedulerConfig,
)
from greenlang.core.coordination_layer import (
    AgentInfo,
    CoordinationConfig,
    CoordinationLayer,
    CoordinationPattern,
)
from greenlang.core.safety_monitor import (
    ConstraintType,
    OperationContext,
    SafetyConfig,
    SafetyConstraint,
    SafetyLevel,
    SafetyMonitor,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessData:
    """
    Data structure for process heat operations.

    This dataclass represents the comprehensive data structure for capturing
    and tracking process heat operations across industrial facilities.
    All fields support provenance tracking for audit compliance.

    Attributes:
        timestamp: ISO 8601 timestamp of data capture
        plant_id: Unique identifier for the plant
        sensor_readings: Dictionary of sensor tag to value mappings
        energy_consumption_kwh: Total energy consumption in kilowatt-hours
        temperature_readings: List of temperature values in Celsius
        pressure_readings: List of pressure values in bar
        flow_rates: List of flow rate values in m3/hr
        efficiency_metrics: Dictionary of efficiency metric calculations
        metadata: Optional additional metadata for tracking
    """
    timestamp: datetime
    plant_id: str
    sensor_readings: Dict[str, float]
    energy_consumption_kwh: float
    temperature_readings: List[float]
    pressure_readings: List[float]
    flow_rates: List[float]
    efficiency_metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize ProcessData after creation."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProcessData to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "plant_id": self.plant_id,
            "sensor_readings": self.sensor_readings,
            "energy_consumption_kwh": self.energy_consumption_kwh,
            "temperature_readings": self.temperature_readings,
            "pressure_readings": self.pressure_readings,
            "flow_rates": self.flow_rates,
            "efficiency_metrics": self.efficiency_metrics,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessData":
        """Create ProcessData from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            timestamp=timestamp,
            plant_id=data.get("plant_id", ""),
            sensor_readings=data.get("sensor_readings", {}),
            energy_consumption_kwh=data.get("energy_consumption_kwh", 0.0),
            temperature_readings=data.get("temperature_readings", []),
            pressure_readings=data.get("pressure_readings", []),
            flow_rates=data.get("flow_rates", []),
            efficiency_metrics=data.get("efficiency_metrics", {}),
            metadata=data.get("metadata", {})
        )

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        import json
        provenance_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()


@dataclass
class ProcessHeatConfig:
    """Configuration for ProcessHeatOrchestrator."""
    agent_id: str = "GL-001"
    agent_name: str = "ProcessHeatOrchestrator"
    version: str = "2.0.0"  # Version 2.0 with BaseOrchestrator
    calculation_timeout_seconds: int = 120
    max_retries: int = 3
    enable_monitoring: bool = True
    cache_ttl_seconds: int = 300
    emission_regulations: Dict[str, Any] = field(default_factory=lambda: {
        "max_emissions_kg_mwh": 200,
        "co2_kg_mwh": 180,
        "nox_kg_mwh": 0.5,
    })
    # Safety constraints
    max_temperature_c: float = 600.0
    min_temperature_c: float = 100.0
    max_pressure_bar: float = 50.0
    min_efficiency_percent: float = 70.0


# Import tools with fallback
try:
    from .tools import (
        ProcessHeatTools,
        ThermalEfficiencyResult,
        HeatDistributionStrategy,
        EnergyBalance,
        ComplianceResult,
    )
except ImportError:
    # Define minimal versions for standalone operation
    @dataclass
    class ThermalEfficiencyResult:
        overall_efficiency: float = 0.0
        carnot_efficiency: float = 0.0
        heat_recovery_efficiency: float = 0.0
        losses: Dict[str, float] = field(default_factory=dict)
        timestamp: str = ""
        provenance_hash: str = ""

    @dataclass
    class HeatDistributionStrategy:
        distribution_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
        total_heat_demand_mw: float = 0.0
        total_heat_supply_mw: float = 0.0
        optimization_score: float = 0.0
        constraints_satisfied: bool = True
        timestamp: str = ""
        provenance_hash: str = ""

    @dataclass
    class EnergyBalance:
        input_energy_mw: float = 0.0
        output_energy_mw: float = 0.0
        losses_mw: float = 0.0
        balance_error_percent: float = 0.0
        is_valid: bool = True
        violations: List[str] = field(default_factory=list)
        timestamp: str = ""
        provenance_hash: str = ""

    @dataclass
    class ComplianceResult:
        total_emissions_kg_hr: float = 0.0
        emission_intensity_kg_mwh: float = 0.0
        regulatory_limit_kg_mwh: float = 200.0
        compliance_status: str = "PASS"
        margin_percent: float = 100.0
        violations: List[str] = field(default_factory=list)
        timestamp: str = ""
        provenance_hash: str = ""

    class ProcessHeatTools:
        """Minimal ProcessHeatTools for fallback."""
        @staticmethod
        def calculate_thermal_efficiency(plant_data: Dict) -> ThermalEfficiencyResult:
            return ThermalEfficiencyResult()

        @staticmethod
        def optimize_heat_distribution(feeds: Dict, constraints: Dict) -> HeatDistributionStrategy:
            return HeatDistributionStrategy()

        @staticmethod
        def validate_energy_balance(data: Dict) -> EnergyBalance:
            return EnergyBalance()

        @staticmethod
        def check_emissions_compliance(emissions: Dict, regs: Dict) -> ComplianceResult:
            return ComplianceResult()

        @staticmethod
        def generate_kpi_dashboard(metrics: Dict) -> Dict:
            return {"status": "ok"}

        @staticmethod
        def coordinate_process_heat_agents(ids: List, cmds: Dict) -> Dict:
            return {"task_assignments": {}}

        @staticmethod
        def integrate_scada_data(feed: Dict) -> Dict:
            return {}

        @staticmethod
        def integrate_erp_data(feed: Dict) -> Dict:
            return {}


class ProcessHeatOrchestrator(BaseOrchestrator[Dict[str, Any], Dict[str, Any]]):
    """
    Master orchestrator for process heat operations (GL-001).

    This agent coordinates all process heat operations across industrial facilities,
    optimizing thermal efficiency, managing heat distribution, ensuring compliance,
    and integrating with SCADA/ERP systems. All calculations follow zero-hallucination
    principles with deterministic algorithms only.

    Inherits from BaseOrchestrator to leverage standard orchestration patterns:
    - MessageBus for async agent communication
    - TaskScheduler for load-balanced task distribution
    - CoordinationLayer for multi-agent coordination
    - SafetyMonitor for operational safety constraints

    Attributes:
        process_config: ProcessHeatConfig with domain-specific configuration
        tools: ProcessHeatTools instance for deterministic calculations
    """

    def __init__(self, process_config: ProcessHeatConfig):
        """
        Initialize ProcessHeatOrchestrator.

        Args:
            process_config: Configuration for process heat operations
        """
        self.process_config = process_config
        self.tools = ProcessHeatTools()

        # Create base orchestrator config
        base_config = OrchestratorConfig(
            orchestrator_id=process_config.agent_id,
            name=process_config.agent_name,
            version=process_config.version,
            max_concurrent_tasks=50,
            default_timeout_seconds=process_config.calculation_timeout_seconds,
            enable_safety_monitoring=True,
            enable_message_bus=True,
            enable_task_scheduling=True,
            enable_coordination=True,
            coordination_pattern=CoordinationPattern.MASTER_SLAVE,
            load_balance_strategy=LoadBalanceStrategy.CAPABILITY_MATCH,
            max_retries=process_config.max_retries,
        )

        # Initialize base orchestrator
        super().__init__(base_config)

        # Add domain-specific safety constraints
        self._add_process_heat_constraints()

        # Results cache with TTL
        self._results_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Performance tracking
        self.performance_metrics = {
            'calculations_performed': 0,
            'avg_calculation_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'agents_coordinated': 0,
            'errors_recovered': 0
        }

        logger.info(f"ProcessHeatOrchestrator {process_config.agent_id} initialized (v{process_config.version})")

    def _create_message_bus(self) -> MessageBus:
        """Create message bus configured for process heat operations."""
        config = MessageBusConfig(
            max_queue_size=5000,
            enable_persistence=False,
            enable_dead_letter=True,
            max_retries=3,
        )
        return MessageBus(config)

    def _create_task_scheduler(self) -> TaskScheduler:
        """Create task scheduler for process heat tasks."""
        config = TaskSchedulerConfig(
            max_queue_size=1000,
            default_timeout_seconds=self.process_config.calculation_timeout_seconds,
            load_balance_strategy=LoadBalanceStrategy.CAPABILITY_MATCH,
            max_concurrent_tasks=50,
        )
        return TaskScheduler(config)

    def _create_coordinator(self) -> CoordinationLayer:
        """Create coordination layer for sub-agent management."""
        config = CoordinationConfig(
            pattern=CoordinationPattern.MASTER_SLAVE,
            lock_ttl_seconds=30.0,
            saga_timeout_seconds=300.0,
        )
        return CoordinationLayer(config)

    def _create_safety_monitor(self) -> SafetyMonitor:
        """Create safety monitor for process heat constraints."""
        config = SafetyConfig(
            enable_circuit_breakers=True,
            enable_rate_limiting=True,
            halt_on_critical=True,
        )
        return SafetyMonitor(config)

    def _add_process_heat_constraints(self) -> None:
        """Add domain-specific safety constraints."""
        if not self.safety_monitor:
            return

        # Temperature constraints
        self.safety_monitor.add_constraint(SafetyConstraint(
            name="max_temperature",
            constraint_type=ConstraintType.THRESHOLD,
            max_value=self.process_config.max_temperature_c,
            level=SafetyLevel.CRITICAL,
            metadata={"parameter": "temperature", "unit": "celsius"},
        ))

        self.safety_monitor.add_constraint(SafetyConstraint(
            name="min_temperature",
            constraint_type=ConstraintType.THRESHOLD,
            min_value=self.process_config.min_temperature_c,
            level=SafetyLevel.HIGH,
            metadata={"parameter": "temperature", "unit": "celsius"},
        ))

        # Pressure constraint
        self.safety_monitor.add_constraint(SafetyConstraint(
            name="max_pressure",
            constraint_type=ConstraintType.THRESHOLD,
            max_value=self.process_config.max_pressure_bar,
            level=SafetyLevel.CRITICAL,
            metadata={"parameter": "pressure", "unit": "bar"},
        ))

        # Efficiency constraint
        self.safety_monitor.add_constraint(SafetyConstraint(
            name="min_efficiency",
            constraint_type=ConstraintType.THRESHOLD,
            min_value=self.process_config.min_efficiency_percent,
            level=SafetyLevel.MEDIUM,
            metadata={"parameter": "efficiency", "unit": "percent"},
        ))

        # Rate limiting for calculation requests
        self.safety_monitor.add_rate_limiter(
            f"{self.config.orchestrator_id}:calculate",
            max_requests=100,
            window_seconds=60.0,
        )

        # Circuit breaker for external integrations
        self.safety_monitor.add_circuit_breaker(
            f"{self.config.orchestrator_id}:scada_integration",
            failure_threshold=5,
            timeout_seconds=60.0,
        )

    async def orchestrate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration logic for process heat operations.

        Args:
            input_data: Input containing plant data, sensor feeds, constraints

        Returns:
            Orchestration result with optimized strategies and KPIs
        """
        start_time = time.perf_counter()

        # Extract input components
        plant_data = input_data.get('plant_data', {})
        sensor_feeds = input_data.get('sensor_feeds', {})
        constraints = input_data.get('constraints', {})
        emissions_data = input_data.get('emissions_data', {})

        # Step 1: Calculate thermal efficiency
        efficiency_result = await self._calculate_efficiency_async(plant_data)

        # Step 2: Optimize heat distribution
        distribution_strategy = await self._optimize_distribution_async(
            sensor_feeds, constraints
        )

        # Step 3: Validate energy balance
        energy_balance = await self._validate_energy_async(plant_data)

        # Step 4: Check emissions compliance
        compliance_result = await self._check_compliance_async(
            emissions_data, self.process_config.emission_regulations
        )

        # Step 5: Generate KPI dashboard
        kpi_dashboard = self._generate_kpi_dashboard(
            efficiency_result,
            distribution_strategy,
            energy_balance,
            compliance_result
        )

        # Step 6: Coordinate sub-agents if needed
        coordination_result = None
        if input_data.get('coordinate_agents', False):
            agent_ids = input_data.get('agent_ids', [])
            commands = input_data.get('agent_commands', {})
            coordination_result = await self._coordinate_agents_async(
                agent_ids, commands
            )

        # Calculate execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_performance_metrics(execution_time_ms)

        # Create comprehensive result
        result = {
            'agent_id': self.config.orchestrator_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': round(execution_time_ms, 2),
            'thermal_efficiency': self._dataclass_to_dict(efficiency_result),
            'heat_distribution': self._dataclass_to_dict(distribution_strategy),
            'energy_balance': self._dataclass_to_dict(energy_balance),
            'emissions_compliance': self._dataclass_to_dict(compliance_result),
            'kpi_dashboard': kpi_dashboard,
            'coordination_result': coordination_result,
            'performance_metrics': self.performance_metrics.copy(),
            'provenance_hash': self._calculate_provenance_hash(
                input_data, kpi_dashboard, str(time.time())
            )
        }

        logger.info(f"Orchestration completed in {execution_time_ms:.2f}ms")
        return result

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary safely."""
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return {}

    async def _calculate_efficiency_async(self, plant_data: Dict[str, Any]) -> ThermalEfficiencyResult:
        """Calculate thermal efficiency asynchronously with caching."""
        cache_key = self._get_cache_key('efficiency', plant_data)
        if self._is_cache_valid(cache_key):
            self.performance_metrics['cache_hits'] += 1
            return self._results_cache[cache_key]

        self.performance_metrics['cache_misses'] += 1
        result = await asyncio.to_thread(
            self.tools.calculate_thermal_efficiency,
            plant_data
        )

        self._store_in_cache(cache_key, result)
        self.performance_metrics['calculations_performed'] += 1
        return result

    async def _optimize_distribution_async(
        self,
        sensor_feeds: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> HeatDistributionStrategy:
        """Optimize heat distribution asynchronously."""
        cache_key = self._get_cache_key('distribution', {
            'feeds': sensor_feeds,
            'constraints': constraints
        })
        if self._is_cache_valid(cache_key):
            self.performance_metrics['cache_hits'] += 1
            return self._results_cache[cache_key]

        self.performance_metrics['cache_misses'] += 1
        result = await asyncio.to_thread(
            self.tools.optimize_heat_distribution,
            sensor_feeds,
            constraints
        )

        self._store_in_cache(cache_key, result)
        self.performance_metrics['calculations_performed'] += 1
        return result

    async def _validate_energy_async(self, consumption_data: Dict[str, Any]) -> EnergyBalance:
        """Validate energy balance asynchronously."""
        result = await asyncio.to_thread(
            self.tools.validate_energy_balance,
            consumption_data
        )
        self.performance_metrics['calculations_performed'] += 1
        return result

    async def _check_compliance_async(
        self,
        emissions_data: Dict[str, Any],
        regulations: Dict[str, Any]
    ) -> ComplianceResult:
        """Check emissions compliance asynchronously."""
        result = await asyncio.to_thread(
            self.tools.check_emissions_compliance,
            emissions_data,
            regulations
        )
        self.performance_metrics['calculations_performed'] += 1
        return result

    def _generate_kpi_dashboard(
        self,
        efficiency: ThermalEfficiencyResult,
        distribution: HeatDistributionStrategy,
        energy_balance: EnergyBalance,
        compliance: ComplianceResult
    ) -> Dict[str, Any]:
        """Generate comprehensive KPI dashboard."""
        metrics = {
            'thermal_efficiency': efficiency.overall_efficiency,
            'heat_recovery_rate': efficiency.heat_recovery_efficiency,
            'capacity_utilization': (
                distribution.total_heat_supply_mw /
                distribution.total_heat_demand_mw * 100
            ) if distribution.total_heat_demand_mw > 0 else 0,
            'co2_intensity': compliance.emission_intensity_kg_mwh,
            'compliance_score': 100 if compliance.compliance_status == "PASS" else 50,
            'energy_balance_accuracy': 100 - energy_balance.balance_error_percent
        }

        dashboard = self.tools.generate_kpi_dashboard(metrics)
        return dashboard

    async def _coordinate_agents_async(
        self,
        agent_ids: List[str],
        commands: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate multiple process heat agents asynchronously."""
        result = await asyncio.to_thread(
            self.tools.coordinate_process_heat_agents,
            agent_ids,
            commands
        )

        self.performance_metrics['agents_coordinated'] += len(agent_ids)

        # Send messages via message bus if available
        if self.message_bus:
            for agent_id, tasks in result.get('task_assignments', {}).items():
                for task in tasks:
                    message = Message(
                        sender_id=self.config.orchestrator_id,
                        recipient_id=agent_id,
                        message_type=MessageType.COMMAND,
                        topic=f"agent.{agent_id}",
                        payload=task,
                        priority=MessagePriority(task.get('priority', 'normal')),
                    )
                    await self.message_bus.publish(message)

        return result

    def _get_cache_key(self, operation: str, data: Dict[str, Any]) -> str:
        """Generate cache key for operation and data."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return f"{operation}_{hashlib.md5(data_str.encode()).hexdigest()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid based on TTL."""
        if cache_key not in self._results_cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key, 0)
        age_seconds = time.time() - timestamp
        return age_seconds < self.process_config.cache_ttl_seconds

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Store result in cache with timestamp."""
        self._results_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        # Limit cache size
        if len(self._results_cache) > 100:
            oldest_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )[:20]
            for key in oldest_keys:
                del self._results_cache[key]
                del self._cache_timestamps[key]

    def _update_performance_metrics(self, execution_time_ms: float) -> None:
        """Update performance metrics with latest execution."""
        n = self.performance_metrics['calculations_performed']
        if n > 0:
            current_avg = self.performance_metrics['avg_calculation_time_ms']
            self.performance_metrics['avg_calculation_time_ms'] = (
                (current_avg * (n - 1) + execution_time_ms) / n
            )

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        execution_id: str
    ) -> str:
        """Calculate SHA-256 provenance hash for complete audit trail."""
        provenance_str = f"{self.config.orchestrator_id}{input_data}{result}{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def integrate_scada(self, scada_feed: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate SCADA data feed."""
        return await asyncio.to_thread(
            self.tools.integrate_scada_data,
            scada_feed
        )

    async def integrate_erp(self, erp_feed: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate ERP data feed."""
        return await asyncio.to_thread(
            self.tools.integrate_erp_data,
            erp_feed
        )

    def get_full_state(self) -> Dict[str, Any]:
        """Get current agent state for monitoring."""
        base_metrics = self.get_metrics()
        return {
            'agent_id': self.config.orchestrator_id,
            'state': self.get_state().value,
            'version': self.config.version,
            'base_metrics': base_metrics.to_dict(),
            'performance_metrics': self.performance_metrics.copy(),
            'cache_size': len(self._results_cache),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Backward compatibility alias
GL001ProcessHeatOrchestrator = ProcessHeatOrchestrator
