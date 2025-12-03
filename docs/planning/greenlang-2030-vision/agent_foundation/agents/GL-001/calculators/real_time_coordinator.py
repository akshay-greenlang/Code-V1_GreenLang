# -*- coding: utf-8 -*-
"""
Real-Time Coordinator - Zero Hallucination Guarantee

Implements multi-agent coordination for real-time process heat operations,
including load shedding, emergency response, demand response, grid interaction,
and predictive load balancing with complete provenance tracking.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: IEC 61131-3, ISA-95, IEEE 2030.5
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

from .provenance import ProvenanceTracker, ProvenanceRecord


class AgentPriority(IntEnum):
    """Agent priority levels for load shedding (higher = more critical)."""
    CRITICAL = 100      # Safety systems, cannot be shed
    HIGH = 75           # Core production, shed only in emergency
    MEDIUM = 50         # Support processes, can be reduced
    LOW = 25            # Non-essential, first to shed
    DEFERRABLE = 10     # Can be completely deferred


class EmergencyLevel(IntEnum):
    """Emergency severity levels."""
    NORMAL = 0
    ALERT = 1           # Elevated monitoring
    WARNING = 2         # Prepare for load shedding
    EMERGENCY = 3       # Active load shedding
    CRITICAL = 4        # Maximum load shedding, safety-only mode


class DemandResponseType(Enum):
    """Types of demand response events."""
    ECONOMIC = "economic"           # Price-based
    EMERGENCY = "emergency"         # Grid stability
    CAPACITY = "capacity"           # Peak reduction
    ANCILLARY = "ancillary"         # Grid services
    REGULATION = "regulation"       # Frequency regulation


class GridInteractionMode(Enum):
    """Grid interaction modes."""
    IMPORT = "import"               # Drawing from grid
    EXPORT = "export"               # Supplying to grid
    ISLAND = "island"               # Disconnected operation
    PEAK_SHAVE = "peak_shave"       # Reducing peak demand
    LOAD_SHIFT = "load_shift"       # Shifting load in time


class StorageMode(Enum):
    """Energy storage operation modes."""
    CHARGING = "charging"
    DISCHARGING = "discharging"
    STANDBY = "standby"
    PEAK_SHAVE = "peak_shave"
    EMERGENCY_RESERVE = "emergency_reserve"


@dataclass(frozen=True)
class AgentState:
    """Immutable state of a coordinated agent."""
    agent_id: str
    agent_type: str  # 'GL-001', 'GL-002', etc.
    priority: AgentPriority
    current_load_kw: float
    max_load_kw: float
    min_load_kw: float
    ramp_rate_kw_min: float
    can_shed: bool
    shed_amount_kw: float = 0.0
    response_time_seconds: float = 60.0
    status: str = "online"


@dataclass(frozen=True)
class LoadSheddingTier:
    """Immutable load shedding tier configuration."""
    tier_level: int
    priority_threshold: AgentPriority
    max_shed_percent: float
    activation_delay_seconds: float
    description: str


@dataclass(frozen=True)
class EmergencyProtocol:
    """Immutable emergency response protocol."""
    protocol_id: str
    emergency_level: EmergencyLevel
    actions: Tuple[str, ...]
    target_reduction_percent: float
    max_response_time_seconds: float
    escalation_threshold_seconds: float


@dataclass(frozen=True)
class DemandResponseEvent:
    """Immutable demand response event."""
    event_id: str
    event_type: DemandResponseType
    start_time: datetime
    end_time: datetime
    target_reduction_kw: float
    price_signal_usd_kwh: float
    penalty_usd_kwh: float = 0.0
    is_mandatory: bool = False


@dataclass(frozen=True)
class GridSignal:
    """Immutable grid signal for coordination."""
    timestamp: datetime
    frequency_hz: float
    voltage_pu: float  # Per unit
    import_price_usd_kwh: float
    export_price_usd_kwh: float
    carbon_intensity_kg_kwh: float
    congestion_level: float  # 0-1
    renewable_percent: float


@dataclass(frozen=True)
class EnergyStorageState:
    """Immutable energy storage system state."""
    storage_id: str
    storage_type: str  # 'thermal', 'battery', 'compressed_air'
    capacity_kwh: float
    current_soc_percent: float  # State of charge
    max_charge_rate_kw: float
    max_discharge_rate_kw: float
    efficiency_percent: float
    mode: StorageMode
    min_soc_percent: float = 10.0
    max_soc_percent: float = 90.0


@dataclass(frozen=True)
class CoordinationContext:
    """Immutable context for real-time coordination."""
    timestamp: datetime
    site_id: str
    agents: Tuple[AgentState, ...]
    total_load_kw: float
    max_capacity_kw: float
    grid_signal: Optional[GridSignal] = None
    demand_response_event: Optional[DemandResponseEvent] = None
    emergency_level: EmergencyLevel = EmergencyLevel.NORMAL
    storage_systems: Tuple[EnergyStorageState, ...] = ()
    load_forecast_kw: Tuple[float, ...] = ()  # Next 24 hours
    ambient_temp_c: float = 20.0


@dataclass
class CoordinationResult:
    """Result of real-time coordination."""
    # Agent commands
    agent_setpoints: Dict[str, Dict[str, float]]
    load_shedding_commands: Dict[str, float]

    # Load balancing
    total_load_before_kw: float
    total_load_after_kw: float
    load_reduction_kw: float
    load_reduction_percent: float

    # Emergency response
    emergency_level: EmergencyLevel
    emergency_actions: List[str]

    # Demand response
    demand_response_status: Dict[str, Any]
    demand_response_achievement_percent: float

    # Grid interaction
    grid_interaction_mode: GridInteractionMode
    grid_exchange_kw: float  # Positive = export, negative = import
    grid_cost_usd_hr: float

    # Storage dispatch
    storage_dispatch: Dict[str, Dict[str, float]]
    storage_contribution_kw: float

    # Optimization metrics
    optimization_score: float
    response_time_ms: float
    constraints_satisfied: bool

    # Provenance
    provenance_hash: str
    calculation_timestamp: str
    version: str = "1.0.0"


class RealTimeCoordinator:
    """
    Real-time multi-agent coordinator for process heat operations.

    Zero Hallucination Guarantee:
    - Pure mathematical calculations using control theory principles
    - No LLM inference in calculation path
    - Bit-perfect reproducibility with frozen dataclasses
    - Complete provenance tracking with SHA-256 hashing
    - Thread-safe caching for performance optimization

    Implements:
    - Priority-based load shedding with multiple tiers
    - Emergency response protocols with escalation
    - Demand response activation and tracking
    - Grid interaction optimization
    - Energy storage dispatch algorithms
    - Predictive load balancing
    - Setpoint optimization across units

    Coordination with other GL agents:
    - GL-002: Waste heat recovery coordination
    - GL-003: Emissions monitoring integration
    - GL-004: Energy storage dispatch
    - GL-005: Grid interaction signals
    """

    # Physical constants
    FREQUENCY_NOMINAL_HZ = Decimal('50.0')  # Or 60.0 for North America
    VOLTAGE_NOMINAL_PU = Decimal('1.0')

    # Control parameters
    FREQUENCY_DEADBAND_HZ = Decimal('0.1')
    VOLTAGE_DEADBAND_PU = Decimal('0.02')

    # Thread-safe cache
    _cache_lock = threading.Lock()
    _setpoint_cache: Dict[str, Tuple[datetime, Dict[str, float]]] = {}
    _cache_ttl_seconds = 5.0  # Cache validity

    # Load shedding tiers (default configuration)
    DEFAULT_SHEDDING_TIERS = (
        LoadSheddingTier(1, AgentPriority.DEFERRABLE, 100.0, 0.0, "Defer non-essential loads"),
        LoadSheddingTier(2, AgentPriority.LOW, 80.0, 5.0, "Reduce low-priority loads"),
        LoadSheddingTier(3, AgentPriority.MEDIUM, 50.0, 15.0, "Reduce medium-priority loads"),
        LoadSheddingTier(4, AgentPriority.HIGH, 30.0, 30.0, "Reduce high-priority loads"),
        LoadSheddingTier(5, AgentPriority.CRITICAL, 0.0, 60.0, "Safety systems only"),
    )

    # Emergency protocols
    DEFAULT_PROTOCOLS = (
        EmergencyProtocol(
            "EP-001", EmergencyLevel.ALERT,
            ("increase_monitoring", "prepare_shedding_tier_1"),
            10.0, 300.0, 600.0
        ),
        EmergencyProtocol(
            "EP-002", EmergencyLevel.WARNING,
            ("activate_tier_1", "notify_operators", "start_backup_systems"),
            25.0, 120.0, 300.0
        ),
        EmergencyProtocol(
            "EP-003", EmergencyLevel.EMERGENCY,
            ("activate_tier_2", "activate_tier_3", "island_from_grid"),
            50.0, 60.0, 120.0
        ),
        EmergencyProtocol(
            "EP-004", EmergencyLevel.CRITICAL,
            ("activate_all_tiers", "emergency_shutdown_non_critical", "safety_mode"),
            80.0, 10.0, 30.0
        ),
    )

    def __init__(
        self,
        version: str = "1.0.0",
        shedding_tiers: Optional[Tuple[LoadSheddingTier, ...]] = None,
        emergency_protocols: Optional[Tuple[EmergencyProtocol, ...]] = None
    ):
        """Initialize coordinator with configuration."""
        self.version = version
        self.shedding_tiers = shedding_tiers or self.DEFAULT_SHEDDING_TIERS
        self.emergency_protocols = emergency_protocols or self.DEFAULT_PROTOCOLS

    def coordinate(self, context: CoordinationContext) -> CoordinationResult:
        """
        Perform real-time coordination across all agents.

        Args:
            context: Current coordination context with agent states

        Returns:
            CoordinationResult with commands and optimization results
        """
        start_time = time.perf_counter()

        tracker = ProvenanceTracker(
            calculation_id=f"coordination_{context.site_id}_{int(context.timestamp.timestamp())}",
            calculation_type="real_time_coordination",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs(self._context_to_dict(context))

        # Step 1: Assess emergency level
        assessed_level = self._assess_emergency_level(context, tracker)

        # Step 2: Execute emergency response if needed
        emergency_actions = self._execute_emergency_response(
            context, assessed_level, tracker
        )

        # Step 3: Calculate load shedding requirements
        shedding_commands = self._calculate_load_shedding(
            context, assessed_level, tracker
        )

        # Step 4: Process demand response
        dr_status, dr_achievement = self._process_demand_response(
            context, tracker
        )

        # Step 5: Optimize grid interaction
        grid_mode, grid_exchange, grid_cost = self._optimize_grid_interaction(
            context, shedding_commands, tracker
        )

        # Step 6: Dispatch energy storage
        storage_dispatch, storage_contribution = self._dispatch_storage(
            context, grid_exchange, tracker
        )

        # Step 7: Calculate optimal setpoints
        agent_setpoints = self._optimize_setpoints(
            context, shedding_commands, storage_contribution, tracker
        )

        # Step 8: Verify constraints
        constraints_ok = self._verify_constraints(
            context, agent_setpoints, shedding_commands, tracker
        )

        # Calculate load changes
        total_before = Decimal(str(context.total_load_kw))
        total_shedding = sum(Decimal(str(v)) for v in shedding_commands.values())
        total_after = total_before - total_shedding + Decimal(str(storage_contribution))

        if total_before > 0:
            reduction_percent = (total_shedding / total_before) * Decimal('100')
        else:
            reduction_percent = Decimal('0')

        # Calculate optimization score
        opt_score = self._calculate_optimization_score(
            context, agent_setpoints, shedding_commands, dr_achievement, tracker
        )

        response_time = (time.perf_counter() - start_time) * 1000

        # Generate provenance
        result_data = {
            'total_shedding': float(total_shedding),
            'grid_exchange': grid_exchange,
            'storage_contribution': storage_contribution,
            'optimization_score': opt_score
        }
        provenance_record = tracker.get_provenance_record(result_data)

        return CoordinationResult(
            agent_setpoints=agent_setpoints,
            load_shedding_commands=shedding_commands,
            total_load_before_kw=float(total_before),
            total_load_after_kw=float(total_after),
            load_reduction_kw=float(total_shedding),
            load_reduction_percent=float(reduction_percent),
            emergency_level=assessed_level,
            emergency_actions=emergency_actions,
            demand_response_status=dr_status,
            demand_response_achievement_percent=dr_achievement,
            grid_interaction_mode=grid_mode,
            grid_exchange_kw=grid_exchange,
            grid_cost_usd_hr=grid_cost,
            storage_dispatch=storage_dispatch,
            storage_contribution_kw=storage_contribution,
            optimization_score=opt_score,
            response_time_ms=response_time,
            constraints_satisfied=constraints_ok,
            provenance_hash=provenance_record.provenance_hash,
            calculation_timestamp=provenance_record.timestamp,
            version=self.version
        )

    def _assess_emergency_level(
        self,
        context: CoordinationContext,
        tracker: ProvenanceTracker
    ) -> EmergencyLevel:
        """Assess current emergency level based on system state."""
        # Start with current level
        assessed_level = context.emergency_level

        # Check capacity utilization
        if context.max_capacity_kw > 0:
            utilization = Decimal(str(context.total_load_kw)) / Decimal(str(context.max_capacity_kw))
        else:
            utilization = Decimal('0')

        # Escalate based on utilization thresholds
        if utilization > Decimal('0.95'):
            assessed_level = max(assessed_level, EmergencyLevel.CRITICAL)
        elif utilization > Decimal('0.90'):
            assessed_level = max(assessed_level, EmergencyLevel.EMERGENCY)
        elif utilization > Decimal('0.85'):
            assessed_level = max(assessed_level, EmergencyLevel.WARNING)
        elif utilization > Decimal('0.80'):
            assessed_level = max(assessed_level, EmergencyLevel.ALERT)

        # Check grid signal if available
        if context.grid_signal:
            freq = Decimal(str(context.grid_signal.frequency_hz))
            freq_deviation = abs(freq - self.FREQUENCY_NOMINAL_HZ)

            if freq_deviation > Decimal('0.5'):
                assessed_level = max(assessed_level, EmergencyLevel.CRITICAL)
            elif freq_deviation > Decimal('0.3'):
                assessed_level = max(assessed_level, EmergencyLevel.EMERGENCY)
            elif freq_deviation > Decimal('0.2'):
                assessed_level = max(assessed_level, EmergencyLevel.WARNING)

            # Check voltage
            voltage = Decimal(str(context.grid_signal.voltage_pu))
            voltage_deviation = abs(voltage - self.VOLTAGE_NOMINAL_PU)

            if voltage_deviation > Decimal('0.1'):
                assessed_level = max(assessed_level, EmergencyLevel.EMERGENCY)
            elif voltage_deviation > Decimal('0.05'):
                assessed_level = max(assessed_level, EmergencyLevel.WARNING)

        tracker.record_step(
            operation="emergency_assessment",
            description="Assess emergency level from system state",
            inputs={
                'utilization_percent': float(utilization * Decimal('100')),
                'current_level': context.emergency_level.value,
                'has_grid_signal': context.grid_signal is not None
            },
            output_value=assessed_level.value,
            output_name="assessed_emergency_level",
            formula="Level = max(utilization_level, frequency_level, voltage_level)",
            units="level"
        )

        return assessed_level

    def _execute_emergency_response(
        self,
        context: CoordinationContext,
        emergency_level: EmergencyLevel,
        tracker: ProvenanceTracker
    ) -> List[str]:
        """Execute emergency response protocol."""
        actions = []

        if emergency_level == EmergencyLevel.NORMAL:
            return actions

        # Find applicable protocol
        protocol = None
        for p in self.emergency_protocols:
            if p.emergency_level == emergency_level:
                protocol = p
                break

        if protocol:
            actions = list(protocol.actions)

            # Add level-specific actions
            if emergency_level >= EmergencyLevel.WARNING:
                actions.append("notify_control_room")

            if emergency_level >= EmergencyLevel.EMERGENCY:
                actions.append("activate_backup_power")

            if emergency_level >= EmergencyLevel.CRITICAL:
                actions.append("initiate_controlled_shutdown")
                actions.append("notify_grid_operator")

        tracker.record_step(
            operation="emergency_response",
            description="Execute emergency response protocol",
            inputs={
                'emergency_level': emergency_level.value,
                'protocol_id': protocol.protocol_id if protocol else None
            },
            output_value=len(actions),
            output_name="emergency_actions_count",
            formula="Protocol-based action selection",
            units="count"
        )

        return actions

    def _calculate_load_shedding(
        self,
        context: CoordinationContext,
        emergency_level: EmergencyLevel,
        tracker: ProvenanceTracker
    ) -> Dict[str, float]:
        """Calculate load shedding commands by priority."""
        shedding_commands: Dict[str, float] = {}

        if emergency_level == EmergencyLevel.NORMAL:
            # No shedding in normal operation
            for agent in context.agents:
                shedding_commands[agent.agent_id] = 0.0
            return shedding_commands

        # Determine target reduction based on emergency level
        target_reduction_percent = Decimal('0')
        for protocol in self.emergency_protocols:
            if protocol.emergency_level == emergency_level:
                target_reduction_percent = Decimal(str(protocol.target_reduction_percent))
                break

        # Calculate required reduction in kW
        total_load = Decimal(str(context.total_load_kw))
        target_reduction_kw = (total_load * target_reduction_percent) / Decimal('100')

        # Sort agents by priority (lowest priority first for shedding)
        sorted_agents = sorted(context.agents, key=lambda a: a.priority)

        # Apply load shedding by tier
        remaining_reduction = target_reduction_kw

        for tier in self.shedding_tiers:
            if remaining_reduction <= 0:
                break

            for agent in sorted_agents:
                if remaining_reduction <= 0:
                    break

                # Skip if agent priority is above tier threshold
                if agent.priority > tier.priority_threshold:
                    continue

                # Skip if agent cannot shed
                if not agent.can_shed:
                    shedding_commands[agent.agent_id] = 0.0
                    continue

                # Calculate max shed for this tier
                max_shed_tier = (Decimal(str(agent.current_load_kw)) *
                                Decimal(str(tier.max_shed_percent)) / Decimal('100'))

                # Constrain to agent's sheddable amount
                max_shed_agent = Decimal(str(agent.shed_amount_kw))
                max_shed = min(max_shed_tier, max_shed_agent)

                # Shed what we can
                shed_amount = min(max_shed, remaining_reduction)
                shedding_commands[agent.agent_id] = float(shed_amount)
                remaining_reduction -= shed_amount

        # Ensure all agents have an entry
        for agent in context.agents:
            if agent.agent_id not in shedding_commands:
                shedding_commands[agent.agent_id] = 0.0

        total_shed = sum(Decimal(str(v)) for v in shedding_commands.values())

        tracker.record_step(
            operation="load_shedding_calculation",
            description="Calculate priority-based load shedding",
            inputs={
                'emergency_level': emergency_level.value,
                'target_reduction_percent': float(target_reduction_percent),
                'total_load_kw': float(total_load),
                'num_agents': len(context.agents)
            },
            output_value=float(total_shed),
            output_name="total_shed_kw",
            formula="Tier-based shedding by priority",
            units="kW"
        )

        return shedding_commands

    def _process_demand_response(
        self,
        context: CoordinationContext,
        tracker: ProvenanceTracker
    ) -> Tuple[Dict[str, Any], float]:
        """Process active demand response event."""
        status: Dict[str, Any] = {
            'active': False,
            'event_id': None,
            'event_type': None,
            'target_kw': 0.0,
            'achieved_kw': 0.0,
            'penalty_risk_usd': 0.0
        }

        if not context.demand_response_event:
            return status, 0.0

        event = context.demand_response_event
        now = context.timestamp

        # Check if event is active
        if not (event.start_time <= now <= event.end_time):
            status['message'] = 'Event not active at current time'
            return status, 0.0

        status['active'] = True
        status['event_id'] = event.event_id
        status['event_type'] = event.event_type.value
        status['target_kw'] = event.target_reduction_kw

        # Calculate available reduction capacity
        available_reduction = Decimal('0')
        for agent in context.agents:
            if agent.can_shed:
                available_reduction += Decimal(str(agent.shed_amount_kw))

        target = Decimal(str(event.target_reduction_kw))
        achieved = min(available_reduction, target)

        status['achieved_kw'] = float(achieved)
        status['available_kw'] = float(available_reduction)

        # Calculate achievement percentage
        if target > 0:
            achievement_percent = (achieved / target) * Decimal('100')
        else:
            achievement_percent = Decimal('100')

        # Calculate penalty risk if under-performing
        if achieved < target and event.is_mandatory:
            shortfall = target - achieved
            hours_remaining = (event.end_time - now).total_seconds() / 3600
            penalty_risk = shortfall * Decimal(str(event.penalty_usd_kwh)) * Decimal(str(hours_remaining))
            status['penalty_risk_usd'] = float(penalty_risk)
            status['shortfall_kw'] = float(shortfall)

        # Economic value
        if event.price_signal_usd_kwh > 0:
            economic_value = achieved * Decimal(str(event.price_signal_usd_kwh))
            status['economic_value_usd_hr'] = float(economic_value)

        tracker.record_step(
            operation="demand_response_processing",
            description="Process demand response event",
            inputs={
                'event_type': event.event_type.value,
                'target_kw': event.target_reduction_kw,
                'available_reduction_kw': float(available_reduction)
            },
            output_value=float(achievement_percent),
            output_name="achievement_percent",
            formula="Achievement = min(Available, Target) / Target * 100",
            units="%"
        )

        return status, float(achievement_percent)

    def _optimize_grid_interaction(
        self,
        context: CoordinationContext,
        shedding_commands: Dict[str, float],
        tracker: ProvenanceTracker
    ) -> Tuple[GridInteractionMode, float, float]:
        """Optimize grid interaction based on signals and load."""
        if not context.grid_signal:
            return GridInteractionMode.IMPORT, 0.0, 0.0

        signal = context.grid_signal

        # Net load after shedding
        net_load = Decimal(str(context.total_load_kw))
        net_load -= sum(Decimal(str(v)) for v in shedding_commands.values())

        # Generation capacity from CHP/on-site generation (simplified)
        on_site_generation = Decimal('0')
        for agent in context.agents:
            if agent.agent_type in ['CHP', 'generator']:
                on_site_generation += Decimal(str(agent.current_load_kw))

        # Net grid exchange (positive = export, negative = import)
        grid_exchange = on_site_generation - net_load

        # Determine mode
        if context.emergency_level >= EmergencyLevel.EMERGENCY:
            # Island if possible in emergency
            mode = GridInteractionMode.ISLAND
            grid_exchange = Decimal('0')
        elif signal.import_price_usd_kwh > signal.export_price_usd_kwh * 1.2:
            # High import price - maximize self-consumption
            mode = GridInteractionMode.PEAK_SHAVE
        elif grid_exchange > 0:
            mode = GridInteractionMode.EXPORT
        else:
            mode = GridInteractionMode.IMPORT

        # Calculate cost/revenue
        if grid_exchange > 0:
            # Exporting - revenue
            cost = -float(grid_exchange) * signal.export_price_usd_kwh
        else:
            # Importing - cost
            cost = -float(grid_exchange) * signal.import_price_usd_kwh

        tracker.record_step(
            operation="grid_interaction_optimization",
            description="Optimize grid interaction mode and exchange",
            inputs={
                'net_load_kw': float(net_load),
                'on_site_generation_kw': float(on_site_generation),
                'import_price': signal.import_price_usd_kwh,
                'export_price': signal.export_price_usd_kwh
            },
            output_value=float(grid_exchange),
            output_name="grid_exchange_kw",
            formula="Exchange = On-site Generation - Net Load",
            units="kW"
        )

        return mode, float(grid_exchange), cost

    def _dispatch_storage(
        self,
        context: CoordinationContext,
        grid_exchange: float,
        tracker: ProvenanceTracker
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """Dispatch energy storage systems optimally."""
        dispatch: Dict[str, Dict[str, float]] = {}
        total_contribution = Decimal('0')

        if not context.storage_systems:
            return dispatch, 0.0

        # Determine storage strategy based on conditions
        should_discharge = False
        should_charge = False

        if context.emergency_level >= EmergencyLevel.WARNING:
            # Discharge in emergencies
            should_discharge = True
        elif context.grid_signal:
            # Price-based strategy
            price_threshold = Decimal('0.08')  # USD/kWh
            if Decimal(str(context.grid_signal.import_price_usd_kwh)) > price_threshold:
                should_discharge = True
            elif Decimal(str(context.grid_signal.import_price_usd_kwh)) < price_threshold * Decimal('0.5'):
                should_charge = True

        # Check if we have excess load to absorb
        if grid_exchange < -100:  # Importing >100 kW
            should_discharge = True
        elif grid_exchange > 100:  # Exporting >100 kW
            should_charge = True

        for storage in context.storage_systems:
            soc = Decimal(str(storage.current_soc_percent))
            capacity = Decimal(str(storage.capacity_kwh))
            efficiency = Decimal(str(storage.efficiency_percent)) / Decimal('100')

            storage_command = {
                'storage_id': storage.storage_id,
                'mode': StorageMode.STANDBY.value,
                'power_kw': 0.0,
                'soc_percent': float(soc),
                'available_energy_kwh': float(capacity * soc / Decimal('100'))
            }

            if should_discharge and soc > Decimal(str(storage.min_soc_percent)):
                # Discharge
                available_energy = capacity * (soc - Decimal(str(storage.min_soc_percent))) / Decimal('100')
                max_power = min(
                    Decimal(str(storage.max_discharge_rate_kw)),
                    available_energy  # 1-hour equivalent
                )
                power = max_power * efficiency

                storage_command['mode'] = StorageMode.DISCHARGING.value
                storage_command['power_kw'] = float(power)
                total_contribution += power

            elif should_charge and soc < Decimal(str(storage.max_soc_percent)):
                # Charge
                available_capacity = capacity * (Decimal(str(storage.max_soc_percent)) - soc) / Decimal('100')
                max_power = min(
                    Decimal(str(storage.max_charge_rate_kw)),
                    available_capacity
                )
                power = max_power / efficiency  # Power drawn from system

                storage_command['mode'] = StorageMode.CHARGING.value
                storage_command['power_kw'] = -float(power)  # Negative = consuming
                total_contribution -= power

            dispatch[storage.storage_id] = storage_command

        tracker.record_step(
            operation="storage_dispatch",
            description="Dispatch energy storage systems",
            inputs={
                'num_storage_systems': len(context.storage_systems),
                'should_discharge': should_discharge,
                'should_charge': should_charge,
                'grid_exchange_kw': grid_exchange
            },
            output_value=float(total_contribution),
            output_name="storage_contribution_kw",
            formula="Sum of discharge power - Sum of charge power",
            units="kW"
        )

        return dispatch, float(total_contribution)

    def _optimize_setpoints(
        self,
        context: CoordinationContext,
        shedding_commands: Dict[str, float],
        storage_contribution: float,
        tracker: ProvenanceTracker
    ) -> Dict[str, Dict[str, float]]:
        """Calculate optimal setpoints for all agents."""
        setpoints: Dict[str, Dict[str, float]] = {}

        # Check cache first
        cache_key = f"{context.site_id}_{context.timestamp.isoformat()}"
        with self._cache_lock:
            if cache_key in self._setpoint_cache:
                cached_time, cached_setpoints = self._setpoint_cache[cache_key]
                if (context.timestamp - cached_time).total_seconds() < self._cache_ttl_seconds:
                    return cached_setpoints

        for agent in context.agents:
            current_load = Decimal(str(agent.current_load_kw))
            shed_amount = Decimal(str(shedding_commands.get(agent.agent_id, 0)))
            target_load = current_load - shed_amount

            # Ensure within bounds
            target_load = max(
                Decimal(str(agent.min_load_kw)),
                min(target_load, Decimal(str(agent.max_load_kw)))
            )

            # Calculate ramp rate constraint
            ramp_rate = Decimal(str(agent.ramp_rate_kw_min))
            max_change = ramp_rate  # Per minute

            # Smooth setpoint change
            if abs(target_load - current_load) > max_change:
                if target_load > current_load:
                    target_load = current_load + max_change
                else:
                    target_load = current_load - max_change

            setpoints[agent.agent_id] = {
                'load_setpoint_kw': float(target_load),
                'previous_load_kw': float(current_load),
                'shed_amount_kw': float(shed_amount),
                'ramp_limited': abs(float(target_load) - agent.current_load_kw) >= agent.ramp_rate_kw_min * 0.99,
                'at_minimum': float(target_load) <= agent.min_load_kw * 1.01,
                'at_maximum': float(target_load) >= agent.max_load_kw * 0.99
            }

        # Store in cache
        with self._cache_lock:
            self._setpoint_cache[cache_key] = (context.timestamp, setpoints)

        tracker.record_step(
            operation="setpoint_optimization",
            description="Calculate optimal setpoints for agents",
            inputs={
                'num_agents': len(context.agents),
                'total_shedding_kw': sum(shedding_commands.values()),
                'storage_contribution_kw': storage_contribution
            },
            output_value=len(setpoints),
            output_name="setpoint_count",
            formula="Target = Current - Shed, constrained by min/max and ramp rate",
            units="count"
        )

        return setpoints

    def _verify_constraints(
        self,
        context: CoordinationContext,
        setpoints: Dict[str, Dict[str, float]],
        shedding_commands: Dict[str, float],
        tracker: ProvenanceTracker
    ) -> bool:
        """Verify all constraints are satisfied."""
        violations = []

        # Check load bounds
        for agent in context.agents:
            sp = setpoints.get(agent.agent_id, {})
            load = sp.get('load_setpoint_kw', 0)

            if load < agent.min_load_kw * 0.99:  # 1% tolerance
                violations.append(f"{agent.agent_id}: load {load} below min {agent.min_load_kw}")

            if load > agent.max_load_kw * 1.01:
                violations.append(f"{agent.agent_id}: load {load} above max {agent.max_load_kw}")

        # Check total capacity constraint
        total_new_load = sum(
            sp.get('load_setpoint_kw', 0) for sp in setpoints.values()
        )
        if total_new_load > context.max_capacity_kw * 1.05:
            violations.append(f"Total load {total_new_load} exceeds capacity {context.max_capacity_kw}")

        # Check ramp rate constraints
        for agent in context.agents:
            sp = setpoints.get(agent.agent_id, {})
            change = abs(sp.get('load_setpoint_kw', 0) - agent.current_load_kw)
            if change > agent.ramp_rate_kw_min * 1.1:  # 10% tolerance
                violations.append(
                    f"{agent.agent_id}: change {change} exceeds ramp rate {agent.ramp_rate_kw_min}"
                )

        constraints_ok = len(violations) == 0

        tracker.record_step(
            operation="constraint_verification",
            description="Verify all operational constraints",
            inputs={
                'num_agents': len(context.agents),
                'num_constraints_checked': 3 * len(context.agents) + 1
            },
            output_value=len(violations),
            output_name="violation_count",
            formula="Check bounds, capacity, and ramp rates",
            units="count"
        )

        return constraints_ok

    def _calculate_optimization_score(
        self,
        context: CoordinationContext,
        setpoints: Dict[str, Dict[str, float]],
        shedding_commands: Dict[str, float],
        dr_achievement: float,
        tracker: ProvenanceTracker
    ) -> float:
        """Calculate overall optimization score (0-100)."""
        scores = []

        # Load balance score (how well distributed)
        loads = [sp.get('load_setpoint_kw', 0) for sp in setpoints.values()]
        if loads and max(loads) > 0:
            load_variance = sum((l - sum(loads)/len(loads))**2 for l in loads) / len(loads)
            max_variance = max(loads)**2
            balance_score = 100 * (1 - min(load_variance / max_variance, 1)) if max_variance > 0 else 100
        else:
            balance_score = 100

        scores.append(('load_balance', balance_score, 0.2))

        # Capacity utilization score
        total_load = sum(loads)
        if context.max_capacity_kw > 0:
            utilization = total_load / context.max_capacity_kw
            # Optimal around 75-85%
            if 0.75 <= utilization <= 0.85:
                util_score = 100
            elif utilization < 0.75:
                util_score = 100 * (utilization / 0.75)
            else:
                util_score = 100 * max(0, (1 - (utilization - 0.85) / 0.15))
        else:
            util_score = 50

        scores.append(('utilization', util_score, 0.2))

        # Demand response achievement
        scores.append(('demand_response', dr_achievement, 0.25))

        # Cost efficiency score
        if context.grid_signal:
            # Lower price = better
            price = context.grid_signal.import_price_usd_kwh
            if price <= 0.05:
                cost_score = 100
            elif price >= 0.20:
                cost_score = 0
            else:
                cost_score = 100 * (1 - (price - 0.05) / 0.15)
        else:
            cost_score = 50

        scores.append(('cost_efficiency', cost_score, 0.2))

        # Emergency response score
        if context.emergency_level == EmergencyLevel.NORMAL:
            emergency_score = 100
        else:
            total_shed = sum(shedding_commands.values())
            target_for_level = {
                EmergencyLevel.ALERT: 0.1,
                EmergencyLevel.WARNING: 0.25,
                EmergencyLevel.EMERGENCY: 0.5,
                EmergencyLevel.CRITICAL: 0.8
            }
            target = target_for_level.get(context.emergency_level, 0.5) * context.total_load_kw
            if target > 0:
                emergency_score = min(100, 100 * total_shed / target)
            else:
                emergency_score = 100

        scores.append(('emergency_response', emergency_score, 0.15))

        # Calculate weighted average
        total_weight = sum(w for _, _, w in scores)
        weighted_sum = sum(s * w for _, s, w in scores)
        final_score = weighted_sum / total_weight if total_weight > 0 else 0

        tracker.record_step(
            operation="optimization_scoring",
            description="Calculate overall optimization score",
            inputs={name: score for name, score, _ in scores},
            output_value=final_score,
            output_name="optimization_score",
            formula="Weighted average of component scores",
            units="score"
        )

        return round(final_score, 2)

    def _context_to_dict(self, context: CoordinationContext) -> Dict[str, Any]:
        """Convert context to dictionary for provenance."""
        return {
            'site_id': context.site_id,
            'timestamp': context.timestamp.isoformat(),
            'num_agents': len(context.agents),
            'total_load_kw': context.total_load_kw,
            'max_capacity_kw': context.max_capacity_kw,
            'emergency_level': context.emergency_level.value,
            'num_storage_systems': len(context.storage_systems),
            'has_grid_signal': context.grid_signal is not None,
            'has_dr_event': context.demand_response_event is not None
        }

    def clear_cache(self) -> None:
        """Clear the setpoint cache (thread-safe)."""
        with self._cache_lock:
            self._setpoint_cache.clear()

    def get_shedding_tiers(self) -> Tuple[LoadSheddingTier, ...]:
        """Get current load shedding tier configuration."""
        return self.shedding_tiers

    def get_emergency_protocols(self) -> Tuple[EmergencyProtocol, ...]:
        """Get current emergency protocol configuration."""
        return self.emergency_protocols
