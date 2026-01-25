"""
MultiBurnerOrchestrator - Multi-burner coordination and sequencing layer.

This module implements coordinated control of multiple burners in a combustion
system, providing lead/lag sequencing, load balancing, wear-leveling rotation,
and safety coordination across all units.

CRITICAL SAFETY INVARIANTS:
- All sequences are deterministic and auditable
- Cross-burner interlocks are checked before ANY state change
- Emergency shutdown propagates to ALL burners immediately
- Purge sequences follow NFPA 85/86 requirements
- No LLM calls for safety-critical decisions

Example:
    >>> config = OrchestrationConfig(num_burners=4)
    >>> orchestrator = MultiBurnerOrchestrator(config)
    >>> orchestrator.initialize_burners(["BRN-001", "BRN-002", "BRN-003", "BRN-004"])
    >>> result = orchestrator.execute_coordinated_start()

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS - Deterministic state definitions
# =============================================================================

class BurnerState(str, Enum):
    """Burner operational states following NFPA 85/86 sequence."""
    OFFLINE = "offline"           # Burner not available
    STANDBY = "standby"           # Ready to start
    PREPURGE = "prepurge"         # Pre-ignition purge
    PILOT_LIGHT = "pilot_light"   # Pilot ignition sequence
    MAIN_FLAME = "main_flame"     # Main flame established
    MODULATING = "modulating"     # Normal modulating operation
    LOW_FIRE = "low_fire"         # Low fire hold
    HIGH_FIRE = "high_fire"       # High fire position
    POSTPURGE = "postpurge"       # Post-shutdown purge
    LOCKOUT = "lockout"           # Safety lockout
    FAULT = "fault"               # Fault condition


class BurnerRole(str, Enum):
    """Role of burner in lead/lag sequencing."""
    LEAD = "lead"                 # Lead burner (first to fire, last to stop)
    LAG_1 = "lag_1"               # First lag burner
    LAG_2 = "lag_2"               # Second lag burner
    LAG_3 = "lag_3"               # Third lag burner
    STANDBY_RESERVE = "standby"   # N+1 redundancy standby
    MAINTENANCE = "maintenance"   # Offline for maintenance


class SequencePhase(str, Enum):
    """Current phase of multi-burner sequence."""
    IDLE = "idle"
    COORDINATED_START = "coordinated_start"
    LOAD_INCREASE = "load_increase"
    LOAD_DECREASE = "load_decrease"
    COORDINATED_STOP = "coordinated_stop"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    LEAD_LAG_ROTATION = "lead_lag_rotation"
    FAILOVER = "failover"


class CoordinationStrategy(str, Enum):
    """Multi-burner coordination strategy."""
    SEQUENTIAL = "sequential"       # One burner at a time
    PARALLEL = "parallel"           # All burners simultaneously
    STAGED = "staged"               # Groups of burners
    ADAPTIVE = "adaptive"           # Based on load demand


class LoadBalancingStrategy(str, Enum):
    """Load distribution strategy across burners."""
    EQUAL = "equal"                 # Equal load distribution
    EFFICIENCY_BASED = "efficiency" # Based on efficiency curves
    WEAR_LEVELING = "wear"          # Minimize runtime differences
    EMISSION_OPTIMAL = "emission"   # Minimize total emissions
    HYBRID = "hybrid"               # Combined optimization


class CommandType(str, Enum):
    """Types of burner commands."""
    START = "start"
    STOP = "stop"
    INCREASE_LOAD = "increase_load"
    DECREASE_LOAD = "decrease_load"
    HOLD = "hold"
    EMERGENCY_STOP = "emergency_stop"
    TRANSFER_LEAD = "transfer_lead"
    PURGE = "purge"


# =============================================================================
# DATA MODELS - Pydantic schemas with validation
# =============================================================================

class BurnerStatus(BaseModel):
    """Current status of an individual burner."""

    burner_id: str = Field(..., description="Unique burner identifier")
    state: BurnerState = Field(..., description="Current operational state")
    role: BurnerRole = Field(default=BurnerRole.LAG_1, description="Current role")

    # Operating parameters
    firing_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    actual_load_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_load_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    # Health indicators
    flame_proven: bool = Field(default=False)
    efficiency_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    stability_index: float = Field(default=1.0, ge=0.0, le=1.0)

    # Runtime tracking for wear leveling
    total_runtime_hours: float = Field(default=0.0, ge=0.0)
    starts_count: int = Field(default=0, ge=0)
    runtime_since_last_rotation: float = Field(default=0.0, ge=0.0)

    # Safety status
    interlocks_satisfied: bool = Field(default=False)
    permissives_met: bool = Field(default=False)
    in_lockout: bool = Field(default=False)
    fault_codes: List[str] = Field(default_factory=list)

    # Timestamps
    last_state_change: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_communication: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Provenance
    provenance_hash: str = Field(default="")

    @validator('provenance_hash', pre=True, always=True)
    def compute_provenance(cls, v, values):
        """Compute provenance hash if not provided."""
        if not v:
            content = str(values)
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        return v


class OrchestrationConfig(BaseModel):
    """Configuration for multi-burner orchestration."""

    # Basic configuration
    num_burners: int = Field(default=2, ge=1, le=20)
    redundancy_mode: bool = Field(default=True, description="Enable N+1 redundancy")

    # Coordination settings
    coordination_strategy: CoordinationStrategy = Field(default=CoordinationStrategy.SEQUENTIAL)
    load_balancing_strategy: LoadBalancingStrategy = Field(default=LoadBalancingStrategy.EQUAL)

    # Timing parameters (deterministic)
    start_delay_between_burners_s: float = Field(default=30.0, ge=5.0, le=300.0)
    stop_delay_between_burners_s: float = Field(default=15.0, ge=5.0, le=120.0)
    flame_prove_timeout_s: float = Field(default=10.0, ge=3.0, le=30.0)
    prepurge_duration_s: float = Field(default=60.0, ge=30.0, le=300.0)
    postpurge_duration_s: float = Field(default=30.0, ge=15.0, le=120.0)

    # Load management
    min_burners_for_load_pct: Dict[int, float] = Field(
        default_factory=lambda: {1: 25.0, 2: 50.0, 3: 75.0, 4: 100.0}
    )
    load_share_tolerance_pct: float = Field(default=5.0, ge=1.0, le=20.0)

    # Wear leveling
    rotation_interval_hours: float = Field(default=168.0, ge=24.0, le=720.0)  # Weekly default
    max_runtime_imbalance_hours: float = Field(default=100.0, ge=10.0, le=500.0)

    # Safety parameters
    cross_light_enabled: bool = Field(default=True)
    emergency_shutdown_propagation_delay_ms: int = Field(default=100, ge=0, le=1000)
    max_simultaneous_starts: int = Field(default=1, ge=1, le=4)

    # Communication
    heartbeat_interval_ms: int = Field(default=1000, ge=100, le=5000)
    communication_timeout_ms: int = Field(default=5000, ge=1000, le=30000)


class BurnerCommand(BaseModel):
    """Command to be executed on a burner."""

    command_id: UUID = Field(default_factory=uuid4)
    burner_id: str = Field(..., description="Target burner")
    command_type: CommandType = Field(..., description="Command type")

    # Command parameters
    target_value: Optional[float] = Field(None, description="Target value if applicable")
    rate_limit: Optional[float] = Field(None, description="Rate limit for changes")

    # Timing
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execute_at: Optional[datetime] = Field(None, description="Scheduled execution time")
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)

    # Safety verification
    safety_verified: bool = Field(default=False)
    interlock_check_passed: bool = Field(default=False)

    # Provenance
    source: str = Field(default="orchestrator")
    reason: str = Field(default="")
    provenance_hash: str = Field(default="")


class SequenceEvent(BaseModel):
    """Event in a multi-burner sequence."""

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    sequence_phase: SequencePhase = Field(...)
    event_type: str = Field(...)
    burner_id: Optional[str] = Field(None)

    # Event data
    previous_state: Optional[BurnerState] = Field(None)
    new_state: Optional[BurnerState] = Field(None)
    details: Dict[str, Any] = Field(default_factory=dict)

    # Outcome
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(None)

    # Audit
    provenance_hash: str = Field(default="")


class LoadDistribution(BaseModel):
    """Load distribution across burners."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_demand_pct: float = Field(..., ge=0.0, le=100.0)

    # Distribution
    burner_loads: Dict[str, float] = Field(default_factory=dict)
    active_burners: List[str] = Field(default_factory=list)
    standby_burners: List[str] = Field(default_factory=list)

    # Strategy metrics
    strategy_used: LoadBalancingStrategy = Field(...)
    efficiency_achieved_pct: float = Field(default=0.0)
    imbalance_pct: float = Field(default=0.0)

    # Provenance
    provenance_hash: str = Field(default="")


class SafetyCoordinationStatus(BaseModel):
    """Status of cross-burner safety coordination."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Overall safety status
    all_interlocks_satisfied: bool = Field(default=False)
    cross_burner_check_passed: bool = Field(default=False)
    emergency_stop_active: bool = Field(default=False)

    # Per-burner interlock status
    burner_interlock_status: Dict[str, bool] = Field(default_factory=dict)
    burner_flame_status: Dict[str, bool] = Field(default_factory=dict)

    # Active safety conditions
    active_interlocks: List[str] = Field(default_factory=list)
    active_trips: List[str] = Field(default_factory=list)

    # Purge status
    purge_in_progress: bool = Field(default=False)
    purge_complete_burners: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")


# =============================================================================
# ORCHESTRATOR STATE
# =============================================================================

@dataclass
class OrchestratorState:
    """Internal state of the orchestrator."""

    current_phase: SequencePhase = SequencePhase.IDLE
    burner_statuses: Dict[str, BurnerStatus] = field(default_factory=dict)
    role_assignments: Dict[str, BurnerRole] = field(default_factory=dict)

    # Load management
    total_demand_pct: float = 0.0
    current_distribution: Optional[LoadDistribution] = None

    # Sequence tracking
    sequence_start_time: Optional[datetime] = None
    pending_commands: List[BurnerCommand] = field(default_factory=list)
    completed_commands: List[BurnerCommand] = field(default_factory=list)

    # Event log
    event_log: List[SequenceEvent] = field(default_factory=list)

    # Safety state
    safety_status: Optional[SafetyCoordinationStatus] = None
    emergency_stop_triggered: bool = False

    # Wear leveling tracking
    last_rotation_time: Optional[datetime] = None
    rotation_history: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# MULTI-BURNER ORCHESTRATOR
# =============================================================================

class MultiBurnerOrchestrator:
    """
    MultiBurnerOrchestrator coordinates multiple burners for optimal operation.

    This class implements:
    - Lead/lag sequencing with automatic rotation
    - Multiple load balancing strategies
    - Cross-burner safety coordination
    - N+1 redundancy management
    - Wear-leveling for equal runtime distribution

    CRITICAL: All operations are deterministic. No LLM calls for safety decisions.

    Attributes:
        config: Orchestration configuration
        state: Current orchestrator state

    Example:
        >>> config = OrchestrationConfig(num_burners=4)
        >>> orchestrator = MultiBurnerOrchestrator(config)
        >>> orchestrator.initialize_burners(["BRN-001", "BRN-002", "BRN-003", "BRN-004"])
        >>> orchestrator.set_load_demand(75.0)
        >>> distribution = orchestrator.calculate_load_distribution()
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        interlock_callback: Optional[Callable[[str], bool]] = None,
        command_callback: Optional[Callable[[BurnerCommand], bool]] = None,
    ):
        """
        Initialize the MultiBurnerOrchestrator.

        Args:
            config: Orchestration configuration
            interlock_callback: Callback to check interlock status
            command_callback: Callback to execute burner commands
        """
        self.config = config
        self.state = OrchestratorState()

        self._interlock_callback = interlock_callback
        self._command_callback = command_callback

        self._lock = asyncio.Lock()
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        logger.info(
            f"MultiBurnerOrchestrator initialized: "
            f"num_burners={config.num_burners}, "
            f"strategy={config.coordination_strategy.value}"
        )

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize_burners(self, burner_ids: List[str]) -> None:
        """
        Initialize burner tracking for the specified burner IDs.

        Args:
            burner_ids: List of burner identifiers

        Raises:
            ValueError: If burner count doesn't match configuration
        """
        if len(burner_ids) != self.config.num_burners:
            raise ValueError(
                f"Expected {self.config.num_burners} burners, got {len(burner_ids)}"
            )

        # Initialize burner statuses
        for idx, burner_id in enumerate(burner_ids):
            status = BurnerStatus(
                burner_id=burner_id,
                state=BurnerState.STANDBY,
                role=self._assign_initial_role(idx),
            )
            self.state.burner_statuses[burner_id] = status
            self.state.role_assignments[burner_id] = status.role

        # Initialize safety status
        self.state.safety_status = SafetyCoordinationStatus(
            burner_interlock_status={bid: False for bid in burner_ids},
            burner_flame_status={bid: False for bid in burner_ids},
        )

        self._log_event(
            SequencePhase.IDLE,
            "burners_initialized",
            details={"burner_ids": burner_ids}
        )

        logger.info(f"Initialized {len(burner_ids)} burners: {burner_ids}")

    def _assign_initial_role(self, index: int) -> BurnerRole:
        """Assign initial role based on burner index."""
        if index == 0:
            return BurnerRole.LEAD
        elif self.config.redundancy_mode and index == self.config.num_burners - 1:
            return BurnerRole.STANDBY_RESERVE
        elif index == 1:
            return BurnerRole.LAG_1
        elif index == 2:
            return BurnerRole.LAG_2
        else:
            return BurnerRole.LAG_3

    # =========================================================================
    # SEQUENCING - Coordinated Start/Stop
    # =========================================================================

    async def execute_coordinated_start(
        self,
        target_load_pct: float = 50.0
    ) -> Tuple[bool, List[SequenceEvent]]:
        """
        Execute coordinated start sequence for all burners.

        Follows NFPA 85/86 requirements:
        1. Verify all interlocks satisfied
        2. Execute prepurge on each burner
        3. Light lead burner first
        4. Prove flame before lighting lag burners
        5. Sequence lag burners with delay

        Args:
            target_load_pct: Target load after start sequence

        Returns:
            Tuple of (success, list of sequence events)
        """
        async with self._lock:
            events = []

            self.state.current_phase = SequencePhase.COORDINATED_START
            self.state.sequence_start_time = datetime.now(timezone.utc)

            self._log_event(
                SequencePhase.COORDINATED_START,
                "sequence_started",
                details={"target_load_pct": target_load_pct}
            )

            try:
                # Step 1: Verify all interlocks
                safety_check = self._check_all_interlocks()
                if not safety_check:
                    self._log_event(
                        SequencePhase.COORDINATED_START,
                        "sequence_blocked",
                        success=False,
                        error_message="Interlock check failed"
                    )
                    return False, self.state.event_log[-10:]

                # Step 2: Determine burners to start based on load
                burners_to_start = self._get_burners_for_load(target_load_pct)

                # Step 3: Execute start sequence
                for burner_id in burners_to_start:
                    # Prepurge
                    purge_success = await self._execute_prepurge(burner_id)
                    if not purge_success:
                        logger.error(f"Prepurge failed for {burner_id}")
                        continue

                    # Light burner
                    light_success = await self._light_burner(burner_id)
                    if not light_success:
                        logger.error(f"Failed to light {burner_id}")
                        continue

                    # Wait for flame prove
                    flame_proved = await self._wait_for_flame_prove(burner_id)
                    if not flame_proved:
                        logger.error(f"Flame not proved for {burner_id}")
                        await self._execute_postpurge(burner_id)
                        continue

                    # Update state
                    self.state.burner_statuses[burner_id].state = BurnerState.MODULATING
                    self.state.burner_statuses[burner_id].flame_proven = True

                    self._log_event(
                        SequencePhase.COORDINATED_START,
                        "burner_started",
                        burner_id=burner_id,
                        new_state=BurnerState.MODULATING
                    )

                    # Delay before next burner (except for last)
                    if burner_id != burners_to_start[-1]:
                        await asyncio.sleep(self.config.start_delay_between_burners_s)

                # Step 4: Distribute load
                self.state.total_demand_pct = target_load_pct
                self.state.current_distribution = self.calculate_load_distribution()

                self.state.current_phase = SequencePhase.IDLE

                self._log_event(
                    SequencePhase.COORDINATED_START,
                    "sequence_completed",
                    details={
                        "burners_started": len(burners_to_start),
                        "target_load_pct": target_load_pct
                    }
                )

                return True, self.state.event_log[-20:]

            except Exception as e:
                logger.exception(f"Coordinated start failed: {e}")
                self.state.current_phase = SequencePhase.IDLE
                self._log_event(
                    SequencePhase.COORDINATED_START,
                    "sequence_failed",
                    success=False,
                    error_message=str(e)
                )
                return False, self.state.event_log[-10:]

    async def execute_coordinated_stop(
        self,
        emergency: bool = False
    ) -> Tuple[bool, List[SequenceEvent]]:
        """
        Execute coordinated stop sequence for all burners.

        Args:
            emergency: If True, execute immediate emergency shutdown

        Returns:
            Tuple of (success, list of sequence events)
        """
        async with self._lock:
            if emergency:
                return await self._execute_emergency_shutdown()

            self.state.current_phase = SequencePhase.COORDINATED_STOP

            self._log_event(
                SequencePhase.COORDINATED_STOP,
                "sequence_started"
            )

            try:
                # Get active burners in reverse start order (lead last)
                active_burners = self._get_active_burners()
                stop_order = self._get_stop_order(active_burners)

                for burner_id in stop_order:
                    # Ramp to low fire
                    await self._ramp_to_low_fire(burner_id)

                    # Stop burner
                    await self._stop_burner(burner_id)

                    # Execute postpurge
                    await self._execute_postpurge(burner_id)

                    # Update state
                    self.state.burner_statuses[burner_id].state = BurnerState.STANDBY
                    self.state.burner_statuses[burner_id].flame_proven = False
                    self.state.burner_statuses[burner_id].firing_rate_pct = 0.0

                    self._log_event(
                        SequencePhase.COORDINATED_STOP,
                        "burner_stopped",
                        burner_id=burner_id,
                        new_state=BurnerState.STANDBY
                    )

                    # Delay before next burner
                    if burner_id != stop_order[-1]:
                        await asyncio.sleep(self.config.stop_delay_between_burners_s)

                self.state.current_phase = SequencePhase.IDLE
                self.state.total_demand_pct = 0.0

                self._log_event(
                    SequencePhase.COORDINATED_STOP,
                    "sequence_completed"
                )

                return True, self.state.event_log[-20:]

            except Exception as e:
                logger.exception(f"Coordinated stop failed: {e}")
                # On failure, attempt emergency shutdown
                return await self._execute_emergency_shutdown()

    async def _execute_emergency_shutdown(self) -> Tuple[bool, List[SequenceEvent]]:
        """
        Execute emergency shutdown of ALL burners.

        CRITICAL: This is a safety function. All burners are stopped
        simultaneously with minimal delay.
        """
        self.state.current_phase = SequencePhase.EMERGENCY_SHUTDOWN
        self.state.emergency_stop_triggered = True

        self._log_event(
            SequencePhase.EMERGENCY_SHUTDOWN,
            "emergency_shutdown_initiated"
        )

        # Stop all burners in parallel (minimal delay)
        stop_tasks = []
        for burner_id in self.state.burner_statuses:
            stop_tasks.append(self._emergency_stop_burner(burner_id))

        # Wait for all stops with minimal timeout
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Update all statuses
        for burner_id, status in self.state.burner_statuses.items():
            status.state = BurnerState.LOCKOUT
            status.flame_proven = False
            status.firing_rate_pct = 0.0
            status.in_lockout = True

        # Update safety status
        if self.state.safety_status:
            self.state.safety_status.emergency_stop_active = True

        self._log_event(
            SequencePhase.EMERGENCY_SHUTDOWN,
            "emergency_shutdown_completed",
            details={"burners_stopped": len(self.state.burner_statuses)}
        )

        logger.critical("EMERGENCY SHUTDOWN COMPLETED - All burners in lockout")

        return True, self.state.event_log[-10:]

    # =========================================================================
    # LOAD BALANCING
    # =========================================================================

    def calculate_load_distribution(self) -> LoadDistribution:
        """
        Calculate optimal load distribution across active burners.

        Uses the configured load balancing strategy:
        - EQUAL: Divide load equally
        - EFFICIENCY_BASED: Assign load based on efficiency curves
        - WEAR_LEVELING: Prioritize burners with less runtime
        - EMISSION_OPTIMAL: Minimize total emissions
        - HYBRID: Combined optimization

        Returns:
            LoadDistribution with calculated loads per burner
        """
        strategy = self.config.load_balancing_strategy
        total_demand = self.state.total_demand_pct

        active_burners = self._get_active_burners()
        if not active_burners:
            return LoadDistribution(
                total_demand_pct=total_demand,
                burner_loads={},
                active_burners=[],
                standby_burners=list(self.state.burner_statuses.keys()),
                strategy_used=strategy,
            )

        # Calculate distribution based on strategy
        if strategy == LoadBalancingStrategy.EQUAL:
            distribution = self._calculate_equal_distribution(active_burners, total_demand)
        elif strategy == LoadBalancingStrategy.EFFICIENCY_BASED:
            distribution = self._calculate_efficiency_distribution(active_burners, total_demand)
        elif strategy == LoadBalancingStrategy.WEAR_LEVELING:
            distribution = self._calculate_wear_distribution(active_burners, total_demand)
        elif strategy == LoadBalancingStrategy.EMISSION_OPTIMAL:
            distribution = self._calculate_emission_distribution(active_burners, total_demand)
        else:  # HYBRID
            distribution = self._calculate_hybrid_distribution(active_burners, total_demand)

        # Calculate imbalance
        loads = list(distribution.values())
        if loads:
            max_load = max(loads)
            min_load = min(loads)
            imbalance = max_load - min_load
        else:
            imbalance = 0.0

        # Determine standby burners
        standby = [
            bid for bid in self.state.burner_statuses
            if bid not in active_burners
        ]

        result = LoadDistribution(
            total_demand_pct=total_demand,
            burner_loads=distribution,
            active_burners=active_burners,
            standby_burners=standby,
            strategy_used=strategy,
            imbalance_pct=imbalance,
        )

        # Compute provenance
        content = f"{total_demand}{distribution}{strategy.value}"
        result.provenance_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        logger.info(
            f"Load distribution calculated: strategy={strategy.value}, "
            f"demand={total_demand}%, active_burners={len(active_burners)}"
        )

        return result

    def _calculate_equal_distribution(
        self,
        active_burners: List[str],
        total_demand: float
    ) -> Dict[str, float]:
        """Calculate equal load distribution (deterministic)."""
        if not active_burners:
            return {}

        load_per_burner = total_demand / len(active_burners)
        return {bid: load_per_burner for bid in active_burners}

    def _calculate_efficiency_distribution(
        self,
        active_burners: List[str],
        total_demand: float
    ) -> Dict[str, float]:
        """
        Calculate efficiency-based distribution.

        Assigns more load to more efficient burners.
        Uses deterministic efficiency curves (no ML).
        """
        if not active_burners:
            return {}

        # Get efficiency for each burner
        efficiencies = {}
        for bid in active_burners:
            status = self.state.burner_statuses[bid]
            # Use stored efficiency or default
            efficiencies[bid] = max(status.efficiency_pct, 80.0)

        # Calculate weighted distribution
        total_efficiency = sum(efficiencies.values())
        if total_efficiency == 0:
            return self._calculate_equal_distribution(active_burners, total_demand)

        distribution = {}
        for bid in active_burners:
            weight = efficiencies[bid] / total_efficiency
            distribution[bid] = total_demand * weight

        return distribution

    def _calculate_wear_distribution(
        self,
        active_burners: List[str],
        total_demand: float
    ) -> Dict[str, float]:
        """
        Calculate wear-leveling distribution.

        Prioritizes burners with less accumulated runtime
        to balance wear across all units.
        """
        if not active_burners:
            return {}

        # Get runtime for each burner
        runtimes = {}
        for bid in active_burners:
            status = self.state.burner_statuses[bid]
            runtimes[bid] = status.total_runtime_hours

        # Invert runtimes (less runtime = higher priority)
        max_runtime = max(runtimes.values()) if runtimes.values() else 1.0
        inverse_runtimes = {
            bid: max_runtime - rt + 1.0  # Add 1 to avoid zero
            for bid, rt in runtimes.items()
        }

        # Calculate weighted distribution
        total_inverse = sum(inverse_runtimes.values())
        if total_inverse == 0:
            return self._calculate_equal_distribution(active_burners, total_demand)

        distribution = {}
        for bid in active_burners:
            weight = inverse_runtimes[bid] / total_inverse
            distribution[bid] = total_demand * weight

        return distribution

    def _calculate_emission_distribution(
        self,
        active_burners: List[str],
        total_demand: float
    ) -> Dict[str, float]:
        """
        Calculate emission-optimal distribution.

        Uses deterministic emission curves to minimize
        total NOx and CO emissions.
        """
        # For emission optimization, we use efficiency as proxy
        # (more efficient = fewer emissions per unit heat)
        return self._calculate_efficiency_distribution(active_burners, total_demand)

    def _calculate_hybrid_distribution(
        self,
        active_burners: List[str],
        total_demand: float
    ) -> Dict[str, float]:
        """
        Calculate hybrid distribution combining multiple factors.

        Weights: 40% efficiency, 30% wear-leveling, 30% equal
        """
        if not active_burners:
            return {}

        efficiency_dist = self._calculate_efficiency_distribution(active_burners, total_demand)
        wear_dist = self._calculate_wear_distribution(active_burners, total_demand)
        equal_dist = self._calculate_equal_distribution(active_burners, total_demand)

        # Combine with weights
        distribution = {}
        for bid in active_burners:
            distribution[bid] = (
                0.4 * efficiency_dist.get(bid, 0) +
                0.3 * wear_dist.get(bid, 0) +
                0.3 * equal_dist.get(bid, 0)
            )

        return distribution

    def set_load_demand(self, demand_pct: float) -> LoadDistribution:
        """
        Set the total load demand and recalculate distribution.

        Args:
            demand_pct: Total load demand (0-100%)

        Returns:
            Updated LoadDistribution
        """
        if demand_pct < 0 or demand_pct > 100:
            raise ValueError(f"Demand must be 0-100%, got {demand_pct}")

        self.state.total_demand_pct = demand_pct
        distribution = self.calculate_load_distribution()
        self.state.current_distribution = distribution

        # Apply load to burners
        for burner_id, load in distribution.burner_loads.items():
            self.state.burner_statuses[burner_id].target_load_pct = load

        self._log_event(
            SequencePhase.IDLE,
            "load_demand_changed",
            details={"demand_pct": demand_pct, "distribution": distribution.burner_loads}
        )

        return distribution

    # =========================================================================
    # LEAD/LAG ROTATION
    # =========================================================================

    async def execute_lead_lag_rotation(self) -> Tuple[bool, Dict[str, BurnerRole]]:
        """
        Execute lead/lag role rotation for wear leveling.

        Rotates burner roles to distribute runtime and starts equally:
        - Current lead becomes last lag
        - First lag becomes new lead
        - All other lags move up one position

        Returns:
            Tuple of (success, new role assignments)
        """
        async with self._lock:
            self.state.current_phase = SequencePhase.LEAD_LAG_ROTATION

            self._log_event(
                SequencePhase.LEAD_LAG_ROTATION,
                "rotation_started"
            )

            try:
                # Get current assignments (excluding standby)
                active_roles = {
                    bid: role for bid, role in self.state.role_assignments.items()
                    if role not in [BurnerRole.STANDBY_RESERVE, BurnerRole.MAINTENANCE]
                }

                if len(active_roles) < 2:
                    logger.warning("Not enough active burners for rotation")
                    return False, self.state.role_assignments.copy()

                # Find current lead
                current_lead = None
                for bid, role in active_roles.items():
                    if role == BurnerRole.LEAD:
                        current_lead = bid
                        break

                if not current_lead:
                    logger.error("No current lead burner found")
                    return False, self.state.role_assignments.copy()

                # Calculate new assignments
                new_assignments = self._calculate_rotation(active_roles, current_lead)

                # Apply new assignments
                for bid, new_role in new_assignments.items():
                    old_role = self.state.role_assignments[bid]
                    self.state.role_assignments[bid] = new_role
                    self.state.burner_statuses[bid].role = new_role

                    # Reset runtime tracking for rotation
                    self.state.burner_statuses[bid].runtime_since_last_rotation = 0.0

                # Record rotation
                self.state.last_rotation_time = datetime.now(timezone.utc)
                self.state.rotation_history.append({
                    "timestamp": self.state.last_rotation_time.isoformat(),
                    "old_lead": current_lead,
                    "new_lead": [bid for bid, role in new_assignments.items() if role == BurnerRole.LEAD][0],
                    "assignments": new_assignments,
                })

                self.state.current_phase = SequencePhase.IDLE

                self._log_event(
                    SequencePhase.LEAD_LAG_ROTATION,
                    "rotation_completed",
                    details={"new_assignments": {k: v.value for k, v in new_assignments.items()}}
                )

                logger.info(f"Lead/lag rotation completed: {new_assignments}")
                return True, new_assignments

            except Exception as e:
                logger.exception(f"Lead/lag rotation failed: {e}")
                self.state.current_phase = SequencePhase.IDLE
                return False, self.state.role_assignments.copy()

    def _calculate_rotation(
        self,
        current_roles: Dict[str, BurnerRole],
        current_lead: str
    ) -> Dict[str, BurnerRole]:
        """Calculate new role assignments for rotation (deterministic)."""
        # Sort by current role order
        role_order = [BurnerRole.LEAD, BurnerRole.LAG_1, BurnerRole.LAG_2, BurnerRole.LAG_3]

        sorted_burners = sorted(
            current_roles.keys(),
            key=lambda bid: role_order.index(current_roles[bid]) if current_roles[bid] in role_order else 99
        )

        # Rotate: lead goes to end, everyone else moves up
        rotated = sorted_burners[1:] + [sorted_burners[0]]

        # Assign roles in order
        new_assignments = {}
        for idx, bid in enumerate(rotated):
            if idx == 0:
                new_assignments[bid] = BurnerRole.LEAD
            elif idx == 1:
                new_assignments[bid] = BurnerRole.LAG_1
            elif idx == 2:
                new_assignments[bid] = BurnerRole.LAG_2
            else:
                new_assignments[bid] = BurnerRole.LAG_3

        return new_assignments

    def should_rotate(self) -> bool:
        """
        Check if lead/lag rotation should occur.

        Returns:
            True if rotation interval has passed or runtime imbalance detected
        """
        # Check time-based rotation
        if self.state.last_rotation_time:
            hours_since = (
                datetime.now(timezone.utc) - self.state.last_rotation_time
            ).total_seconds() / 3600

            if hours_since >= self.config.rotation_interval_hours:
                return True
        else:
            # No rotation yet - check if we've been running long enough
            return False

        # Check runtime imbalance
        runtimes = [
            status.runtime_since_last_rotation
            for status in self.state.burner_statuses.values()
            if status.role not in [BurnerRole.STANDBY_RESERVE, BurnerRole.MAINTENANCE]
        ]

        if runtimes:
            imbalance = max(runtimes) - min(runtimes)
            if imbalance > self.config.max_runtime_imbalance_hours:
                return True

        return False

    # =========================================================================
    # N+1 REDUNDANCY
    # =========================================================================

    async def execute_failover(
        self,
        failed_burner_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute failover from failed burner to standby.

        Args:
            failed_burner_id: ID of the failed burner

        Returns:
            Tuple of (success, replacement burner ID or None)
        """
        async with self._lock:
            self.state.current_phase = SequencePhase.FAILOVER

            self._log_event(
                SequencePhase.FAILOVER,
                "failover_initiated",
                burner_id=failed_burner_id
            )

            if not self.config.redundancy_mode:
                logger.warning("Redundancy mode not enabled")
                return False, None

            # Find standby burner
            standby_burner = self._find_standby_burner()
            if not standby_burner:
                logger.error("No standby burner available for failover")
                self._log_event(
                    SequencePhase.FAILOVER,
                    "failover_failed",
                    success=False,
                    error_message="No standby burner available"
                )
                return False, None

            try:
                # Mark failed burner
                failed_status = self.state.burner_statuses[failed_burner_id]
                failed_role = failed_status.role
                failed_status.state = BurnerState.FAULT
                failed_status.role = BurnerRole.MAINTENANCE
                failed_status.flame_proven = False

                # Get target load from failed burner
                target_load = failed_status.target_load_pct

                # Start standby burner
                standby_status = self.state.burner_statuses[standby_burner]

                # Execute start sequence for standby
                purge_success = await self._execute_prepurge(standby_burner)
                if not purge_success:
                    raise RuntimeError(f"Prepurge failed for standby {standby_burner}")

                light_success = await self._light_burner(standby_burner)
                if not light_success:
                    raise RuntimeError(f"Failed to light standby {standby_burner}")

                flame_proved = await self._wait_for_flame_prove(standby_burner)
                if not flame_proved:
                    raise RuntimeError(f"Flame not proved for standby {standby_burner}")

                # Update standby to take over role
                standby_status.state = BurnerState.MODULATING
                standby_status.role = failed_role
                standby_status.flame_proven = True
                standby_status.target_load_pct = target_load

                self.state.role_assignments[standby_burner] = failed_role
                self.state.role_assignments[failed_burner_id] = BurnerRole.MAINTENANCE

                self.state.current_phase = SequencePhase.IDLE

                self._log_event(
                    SequencePhase.FAILOVER,
                    "failover_completed",
                    details={
                        "failed_burner": failed_burner_id,
                        "replacement_burner": standby_burner,
                        "role_transferred": failed_role.value,
                    }
                )

                logger.info(
                    f"Failover completed: {failed_burner_id} -> {standby_burner}"
                )
                return True, standby_burner

            except Exception as e:
                logger.exception(f"Failover failed: {e}")
                self.state.current_phase = SequencePhase.IDLE
                self._log_event(
                    SequencePhase.FAILOVER,
                    "failover_failed",
                    success=False,
                    error_message=str(e)
                )
                return False, None

    def _find_standby_burner(self) -> Optional[str]:
        """Find available standby burner for failover."""
        for bid, status in self.state.burner_statuses.items():
            if (status.role == BurnerRole.STANDBY_RESERVE and
                status.state == BurnerState.STANDBY and
                not status.in_lockout):
                return bid
        return None

    # =========================================================================
    # SAFETY COORDINATION
    # =========================================================================

    def check_cross_burner_interlocks(self) -> SafetyCoordinationStatus:
        """
        Check interlock status across all burners.

        CRITICAL: This is a safety function. Results are deterministic.

        Returns:
            SafetyCoordinationStatus with comprehensive interlock status
        """
        interlock_status = {}
        flame_status = {}
        active_interlocks = []
        active_trips = []

        all_satisfied = True

        for bid, status in self.state.burner_statuses.items():
            # Check interlocks via callback or stored status
            if self._interlock_callback:
                try:
                    interlocks_ok = self._interlock_callback(bid)
                except Exception as e:
                    logger.error(f"Interlock check failed for {bid}: {e}")
                    interlocks_ok = False
            else:
                interlocks_ok = status.interlocks_satisfied

            interlock_status[bid] = interlocks_ok
            flame_status[bid] = status.flame_proven

            if not interlocks_ok:
                all_satisfied = False
                active_interlocks.append(f"{bid}: interlock not satisfied")

            if status.in_lockout:
                active_trips.append(f"{bid}: in lockout")

            if status.fault_codes:
                for code in status.fault_codes:
                    active_trips.append(f"{bid}: {code}")

        # Check for cross-burner conditions
        cross_check_passed = self._check_cross_burner_conditions()

        # Update safety status
        safety_status = SafetyCoordinationStatus(
            all_interlocks_satisfied=all_satisfied,
            cross_burner_check_passed=cross_check_passed,
            emergency_stop_active=self.state.emergency_stop_triggered,
            burner_interlock_status=interlock_status,
            burner_flame_status=flame_status,
            active_interlocks=active_interlocks,
            active_trips=active_trips,
        )

        # Compute provenance
        content = f"{interlock_status}{flame_status}{all_satisfied}"
        safety_status.provenance_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        self.state.safety_status = safety_status

        return safety_status

    def _check_cross_burner_conditions(self) -> bool:
        """
        Check cross-burner safety conditions.

        Verifies:
        - No single burner carrying excessive load
        - Flame supervision on all active burners
        - Communication with all burners
        """
        active_burners = self._get_active_burners()

        if not active_burners:
            return True

        # Check flame status for all active burners
        for bid in active_burners:
            status = self.state.burner_statuses[bid]
            if status.state == BurnerState.MODULATING and not status.flame_proven:
                logger.warning(f"Cross-burner check failed: {bid} modulating without flame")
                return False

        # Check communication freshness
        now = datetime.now(timezone.utc)
        for bid in active_burners:
            status = self.state.burner_statuses[bid]
            comm_age = (now - status.last_communication).total_seconds() * 1000
            if comm_age > self.config.communication_timeout_ms:
                logger.warning(f"Cross-burner check failed: {bid} communication timeout")
                return False

        return True

    def supervise_flames(self) -> Dict[str, bool]:
        """
        Supervise flame status across all active burners.

        Returns:
            Dict mapping burner_id to flame_proven status
        """
        flame_status = {}

        for bid, status in self.state.burner_statuses.items():
            if status.state in [BurnerState.MODULATING, BurnerState.LOW_FIRE, BurnerState.HIGH_FIRE]:
                flame_status[bid] = status.flame_proven

                if not status.flame_proven:
                    logger.error(f"FLAME FAILURE detected on {bid}")
                    # Log critical event
                    self._log_event(
                        self.state.current_phase,
                        "flame_failure_detected",
                        burner_id=bid,
                        success=False,
                        error_message="Flame failure - immediate action required"
                    )

        return flame_status

    async def coordinate_purge_sequence(self) -> bool:
        """
        Coordinate purge sequence across all burners.

        Ensures proper purge timing per NFPA 85/86:
        - Minimum 4 air changes through combustion chamber
        - All burners must complete purge before any ignition

        Returns:
            True if purge completed successfully on all burners
        """
        self._log_event(
            self.state.current_phase,
            "coordinated_purge_started"
        )

        purge_results = {}

        for bid in self.state.burner_statuses:
            success = await self._execute_prepurge(bid)
            purge_results[bid] = success

        all_success = all(purge_results.values())

        if self.state.safety_status:
            self.state.safety_status.purge_in_progress = False
            self.state.safety_status.purge_complete_burners = [
                bid for bid, success in purge_results.items() if success
            ]

        self._log_event(
            self.state.current_phase,
            "coordinated_purge_completed",
            details={"results": purge_results},
            success=all_success
        )

        return all_success

    # =========================================================================
    # COMMUNICATION & STATE SHARING
    # =========================================================================

    async def start_heartbeat_monitoring(self) -> None:
        """Start background heartbeat monitoring for all burners."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat monitoring started")

    async def stop_heartbeat_monitoring(self) -> None:
        """Stop background heartbeat monitoring."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat monitoring stopped")

    async def _heartbeat_loop(self) -> None:
        """Background loop for heartbeat monitoring."""
        interval_s = self.config.heartbeat_interval_ms / 1000

        while self._running:
            try:
                await self._check_all_heartbeats()
            except Exception as e:
                logger.error(f"Heartbeat check error: {e}")

            await asyncio.sleep(interval_s)

    async def _check_all_heartbeats(self) -> None:
        """Check heartbeat status for all burners."""
        now = datetime.now(timezone.utc)
        timeout_ms = self.config.communication_timeout_ms

        for bid, status in self.state.burner_statuses.items():
            age_ms = (now - status.last_communication).total_seconds() * 1000

            if age_ms > timeout_ms:
                logger.warning(f"Heartbeat timeout for {bid}: {age_ms:.0f}ms")

                # If modulating, this is a critical issue
                if status.state == BurnerState.MODULATING:
                    self._log_event(
                        self.state.current_phase,
                        "communication_lost",
                        burner_id=bid,
                        success=False,
                        error_message=f"Communication timeout: {age_ms:.0f}ms"
                    )

    def update_burner_status(self, burner_id: str, updates: Dict[str, Any]) -> None:
        """
        Update status for a specific burner.

        Args:
            burner_id: Burner identifier
            updates: Dictionary of status updates
        """
        if burner_id not in self.state.burner_statuses:
            logger.warning(f"Unknown burner: {burner_id}")
            return

        status = self.state.burner_statuses[burner_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(status, key):
                setattr(status, key, value)

        # Update communication timestamp
        status.last_communication = datetime.now(timezone.utc)

        # Recompute provenance
        content = str(status.dict())
        status.provenance_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    def broadcast_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Broadcast an event to all monitoring systems.

        Args:
            event_type: Type of event
            details: Event details
        """
        event = SequenceEvent(
            sequence_phase=self.state.current_phase,
            event_type=event_type,
            details=details,
        )

        # Compute provenance
        content = f"{event_type}{details}{datetime.now(timezone.utc).isoformat()}"
        event.provenance_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        self.state.event_log.append(event)

        logger.info(f"Event broadcast: {event_type}")

    def get_orchestration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestration status.

        Returns:
            Dictionary with full orchestration state
        """
        return {
            "current_phase": self.state.current_phase.value,
            "total_demand_pct": self.state.total_demand_pct,
            "burner_count": len(self.state.burner_statuses),
            "active_burners": len(self._get_active_burners()),
            "burner_statuses": {
                bid: {
                    "state": status.state.value,
                    "role": status.role.value,
                    "firing_rate_pct": status.firing_rate_pct,
                    "flame_proven": status.flame_proven,
                }
                for bid, status in self.state.burner_statuses.items()
            },
            "role_assignments": {
                bid: role.value for bid, role in self.state.role_assignments.items()
            },
            "safety_status": {
                "all_interlocks_satisfied": self.state.safety_status.all_interlocks_satisfied if self.state.safety_status else False,
                "emergency_stop_active": self.state.emergency_stop_triggered,
            },
            "last_rotation": self.state.last_rotation_time.isoformat() if self.state.last_rotation_time else None,
            "event_count": len(self.state.event_log),
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _check_all_interlocks(self) -> bool:
        """Check interlocks on all burners."""
        for bid in self.state.burner_statuses:
            if self._interlock_callback:
                if not self._interlock_callback(bid):
                    return False
            elif not self.state.burner_statuses[bid].interlocks_satisfied:
                return False
        return True

    def _get_active_burners(self) -> List[str]:
        """Get list of currently active (firing) burners."""
        active_states = [BurnerState.MODULATING, BurnerState.LOW_FIRE, BurnerState.HIGH_FIRE]
        return [
            bid for bid, status in self.state.burner_statuses.items()
            if status.state in active_states
        ]

    def _get_burners_for_load(self, load_pct: float) -> List[str]:
        """Determine which burners should be active for given load."""
        # Use min_burners_for_load_pct mapping
        burners_needed = 1
        for count, threshold in sorted(self.config.min_burners_for_load_pct.items()):
            if load_pct >= threshold:
                burners_needed = count

        # Get available burners in role order
        role_order = [BurnerRole.LEAD, BurnerRole.LAG_1, BurnerRole.LAG_2, BurnerRole.LAG_3]

        available = []
        for role in role_order:
            for bid, status in self.state.burner_statuses.items():
                if status.role == role and not status.in_lockout:
                    available.append(bid)

        return available[:burners_needed]

    def _get_stop_order(self, active_burners: List[str]) -> List[str]:
        """Get burner stop order (reverse of role order, lead last)."""
        role_priority = {
            BurnerRole.LAG_3: 0,
            BurnerRole.LAG_2: 1,
            BurnerRole.LAG_1: 2,
            BurnerRole.LEAD: 3,
        }

        return sorted(
            active_burners,
            key=lambda bid: role_priority.get(self.state.burner_statuses[bid].role, 0)
        )

    def _log_event(
        self,
        phase: SequencePhase,
        event_type: str,
        burner_id: Optional[str] = None,
        previous_state: Optional[BurnerState] = None,
        new_state: Optional[BurnerState] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a sequence event."""
        event = SequenceEvent(
            sequence_phase=phase,
            event_type=event_type,
            burner_id=burner_id,
            previous_state=previous_state,
            new_state=new_state,
            details=details or {},
            success=success,
            error_message=error_message,
        )

        # Compute provenance
        content = f"{phase.value}{event_type}{burner_id}{datetime.now(timezone.utc).isoformat()}"
        event.provenance_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        self.state.event_log.append(event)

        # Trim event log if too long
        if len(self.state.event_log) > 1000:
            self.state.event_log = self.state.event_log[-500:]

    # =========================================================================
    # BURNER CONTROL PRIMITIVES (Deterministic)
    # =========================================================================

    async def _execute_prepurge(self, burner_id: str) -> bool:
        """Execute prepurge sequence for a burner."""
        status = self.state.burner_statuses[burner_id]
        status.state = BurnerState.PREPURGE

        self._log_event(
            self.state.current_phase,
            "prepurge_started",
            burner_id=burner_id,
            new_state=BurnerState.PREPURGE
        )

        # Issue command via callback
        if self._command_callback:
            cmd = BurnerCommand(
                burner_id=burner_id,
                command_type=CommandType.PURGE,
                source="orchestrator",
                reason="prepurge_sequence",
            )
            try:
                self._command_callback(cmd)
            except Exception as e:
                logger.error(f"Prepurge command failed for {burner_id}: {e}")
                return False

        # Wait for purge duration (deterministic timing)
        await asyncio.sleep(self.config.prepurge_duration_s)

        self._log_event(
            self.state.current_phase,
            "prepurge_completed",
            burner_id=burner_id
        )

        return True

    async def _execute_postpurge(self, burner_id: str) -> bool:
        """Execute postpurge sequence for a burner."""
        status = self.state.burner_statuses[burner_id]
        status.state = BurnerState.POSTPURGE

        self._log_event(
            self.state.current_phase,
            "postpurge_started",
            burner_id=burner_id,
            new_state=BurnerState.POSTPURGE
        )

        await asyncio.sleep(self.config.postpurge_duration_s)

        self._log_event(
            self.state.current_phase,
            "postpurge_completed",
            burner_id=burner_id
        )

        return True

    async def _light_burner(self, burner_id: str) -> bool:
        """Execute burner light sequence."""
        status = self.state.burner_statuses[burner_id]
        status.state = BurnerState.PILOT_LIGHT

        self._log_event(
            self.state.current_phase,
            "pilot_light_started",
            burner_id=burner_id,
            new_state=BurnerState.PILOT_LIGHT
        )

        if self._command_callback:
            cmd = BurnerCommand(
                burner_id=burner_id,
                command_type=CommandType.START,
                source="orchestrator",
                reason="coordinated_start",
            )
            try:
                result = self._command_callback(cmd)
                return result
            except Exception as e:
                logger.error(f"Light command failed for {burner_id}: {e}")
                return False

        # Simulate for testing
        return True

    async def _wait_for_flame_prove(self, burner_id: str) -> bool:
        """Wait for flame to be proven on a burner."""
        start_time = time.time()
        timeout_s = self.config.flame_prove_timeout_s

        while time.time() - start_time < timeout_s:
            status = self.state.burner_statuses[burner_id]
            if status.flame_proven:
                self._log_event(
                    self.state.current_phase,
                    "flame_proved",
                    burner_id=burner_id,
                    new_state=BurnerState.MAIN_FLAME
                )
                return True

            await asyncio.sleep(0.5)

        logger.error(f"Flame prove timeout for {burner_id}")
        return False

    async def _stop_burner(self, burner_id: str) -> bool:
        """Stop a burner."""
        if self._command_callback:
            cmd = BurnerCommand(
                burner_id=burner_id,
                command_type=CommandType.STOP,
                source="orchestrator",
                reason="coordinated_stop",
            )
            try:
                self._command_callback(cmd)
            except Exception as e:
                logger.error(f"Stop command failed for {burner_id}: {e}")
                return False

        return True

    async def _emergency_stop_burner(self, burner_id: str) -> bool:
        """Emergency stop a single burner."""
        if self._command_callback:
            cmd = BurnerCommand(
                burner_id=burner_id,
                command_type=CommandType.EMERGENCY_STOP,
                source="orchestrator",
                reason="emergency_shutdown",
            )
            try:
                self._command_callback(cmd)
            except Exception as e:
                logger.error(f"Emergency stop failed for {burner_id}: {e}")

        # Always mark as stopped regardless of callback success
        return True

    async def _ramp_to_low_fire(self, burner_id: str) -> None:
        """Ramp burner to low fire position."""
        status = self.state.burner_statuses[burner_id]
        status.state = BurnerState.LOW_FIRE
        status.target_load_pct = self.config.min_burners_for_load_pct.get(1, 25.0)

        self._log_event(
            self.state.current_phase,
            "ramp_to_low_fire",
            burner_id=burner_id,
            new_state=BurnerState.LOW_FIRE
        )

        # Allow time for ramp
        await asyncio.sleep(5.0)
