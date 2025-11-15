"""
BoilerEfficiencyOptimizer - Master orchestrator for boiler efficiency operations.

This module implements the GL-002 BoilerEfficiencyOptimizer agent for managing
and optimizing boiler operations across industrial facilities. It optimizes
combustion parameters, fuel efficiency, emissions control, and steam generation
following zero-hallucination principles with deterministic algorithms only.

Example:
    >>> from boiler_efficiency_orchestrator import BoilerEfficiencyOptimizer
    >>> config = BoilerEfficiencyConfig(...)
    >>> orchestrator = BoilerEfficiencyOptimizer(config)
    >>> result = await orchestrator.execute(boiler_data)
"""

import asyncio
import hashlib
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

# Import from agent_foundation
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from agent_foundation.base_agent import BaseAgent, AgentState, AgentConfig
from agent_foundation.agent_intelligence import (
    AgentIntelligence,
    ChatSession,
    ModelProvider,
    PromptTemplate
)
from agent_foundation.orchestration.message_bus import MessageBus, Message
from agent_foundation.orchestration.saga import SagaOrchestrator, SagaStep
from agent_foundation.memory.short_term_memory import ShortTermMemory
from agent_foundation.memory.long_term_memory import LongTermMemory

# Import local modules
from .config import BoilerEfficiencyConfig, BoilerConfiguration
from .tools import BoilerEfficiencyTools, CombustionOptimizationResult, SteamGenerationStrategy

logger = logging.getLogger(__name__)


# ============================================================================
# THREAD-SAFE CACHE IMPLEMENTATION
# ============================================================================

class ThreadSafeCache:
    """
    Thread-safe cache implementation for concurrent access.

    Provides LRU caching with automatic TTL management and thread safety
    using threading.Lock to prevent race conditions.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0):
        """
        Initialize thread-safe cache.

        Args:
            max_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if valid.

        Args:
            key: Cache key

        Returns:
            Cached value if exists and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check if entry has expired
            age_seconds = time.time() - self._timestamps[key]
            if age_seconds >= self._ttl_seconds:
                # Remove expired entry
                del self._cache[key]
                del self._timestamps[key]
                return None

            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with thread safety.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(
                    self._timestamps.keys(),
                    key=lambda k: self._timestamps[k]
                )
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            # Store new value
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class OperationMode(str, Enum):
    """Boiler operation modes."""
    STARTUP = "startup"
    NORMAL = "normal"
    HIGH_EFFICIENCY = "high_efficiency"
    LOW_LOAD = "low_load"
    SHUTDOWN = "shutdown"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class OptimizationStrategy(str, Enum):
    """Optimization strategies for boiler operations."""
    FUEL_EFFICIENCY = "fuel_efficiency"
    EMISSIONS_REDUCTION = "emissions_reduction"
    STEAM_QUALITY = "steam_quality"
    BALANCED = "balanced"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class BoilerOperationalState:
    """Current operational state of the boiler."""
    mode: OperationMode
    efficiency_percent: float
    fuel_flow_rate_kg_hr: float
    steam_flow_rate_kg_hr: float
    combustion_temperature_c: float
    excess_air_percent: float
    co2_emissions_kg_hr: float
    nox_emissions_ppm: float
    timestamp: datetime


class BoilerEfficiencyOptimizer(BaseAgent):
    """
    Master orchestrator for boiler efficiency operations (GL-002).

    This agent coordinates all boiler optimization operations across industrial
    facilities, maximizing fuel efficiency, minimizing emissions, optimizing
    steam generation, and ensuring safe operation. All calculations follow
    zero-hallucination principles with deterministic algorithms only.

    Attributes:
        config: BoilerEfficiencyConfig with complete configuration
        tools: BoilerEfficiencyTools instance for deterministic calculations
        intelligence: AgentIntelligence for LLM integration (classification only)
        message_bus: MessageBus for multi-agent coordination
        performance_metrics: Real-time performance tracking
    """

    def __init__(self, config: BoilerEfficiencyConfig):
        """
        Initialize BoilerEfficiencyOptimizer.

        Args:
            config: Configuration for boiler efficiency operations
        """
        # Convert to BaseAgent config
        base_config = AgentConfig(
            name=config.agent_name,
            version=config.version,
            agent_id=config.agent_id,
            timeout_seconds=config.calculation_timeout_seconds,
            enable_metrics=config.enable_monitoring,
            checkpoint_enabled=True,
            checkpoint_interval_seconds=300
        )

        super().__init__(base_config)

        self.boiler_config = config
        self.tools = BoilerEfficiencyTools()

        # Initialize intelligence with deterministic settings
        self._init_intelligence()

        # Initialize memory systems
        self.short_term_memory = ShortTermMemory(capacity=2000)
        self.long_term_memory = LongTermMemory(
            storage_path=Path("./gl002_memory") if base_config.state_directory is None
            else base_config.state_directory / "memory"
        )

        # Initialize message bus for agent coordination
        self.message_bus = MessageBus()

        # Performance tracking
        self.performance_metrics = {
            'optimizations_performed': 0,
            'avg_optimization_time_ms': 0,
            'fuel_savings_kg': 0,
            'emissions_reduced_kg': 0,
            'efficiency_improvements': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'agents_coordinated': 0,
            'errors_recovered': 0,
            'total_steam_generated_tons': 0
        }

        # Thread-safe results cache with TTL for performance optimization
        self._results_cache = ThreadSafeCache(max_size=200, ttl_seconds=60)

        # Operational state tracking
        self.current_state = None
        self.state_history = []
        self.optimization_history = []

        logger.info(f"BoilerEfficiencyOptimizer {config.agent_id} initialized successfully")

    def _init_intelligence(self):
        """Initialize AgentIntelligence with deterministic configuration."""
        try:
            # Create deterministic ChatSession for classification tasks only
            self.chat_session = ChatSession(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-3-haiku",  # Fast model for classification
                temperature=0.0,  # Deterministic
                seed=42,  # Fixed seed for reproducibility
                max_tokens=500  # Limited output for classification
            )

            # Initialize prompt templates for classification tasks
            self.anomaly_classification_prompt = PromptTemplate(
                template="""
                Classify the following boiler operational data for anomalies.
                Return one of: normal, efficiency_degradation, combustion_issue,
                emissions_exceedance, maintenance_required, critical_failure

                Operational Data:
                - Efficiency: {efficiency}%
                - Fuel Flow: {fuel_flow} kg/hr
                - Excess Air: {excess_air}%
                - Stack Temperature: {stack_temp}°C
                - NOx: {nox} ppm
                - CO: {co} ppm

                Return only the classification category, nothing else.
                """,
                variables=['efficiency', 'fuel_flow', 'excess_air', 'stack_temp', 'nox', 'co']
            )

            self.optimization_strategy_prompt = PromptTemplate(
                template="""
                Select optimal strategy for boiler based on constraints.
                Constraints: {constraints}
                Current State: {state}

                Return one of: fuel_efficiency, emissions_reduction, steam_quality, balanced
                """,
                variables=['constraints', 'state']
            )

            logger.info("AgentIntelligence initialized with deterministic settings")

        except Exception as e:
            logger.warning(f"AgentIntelligence initialization failed, continuing without LLM: {e}")
            self.chat_session = None

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for boiler efficiency orchestration.

        Args:
            input_data: Input containing boiler data, sensor feeds, constraints

        Returns:
            Orchestration result with optimized parameters and KPIs
        """
        start_time = time.perf_counter()
        self.state = AgentState.EXECUTING

        try:
            # Extract input components
            boiler_data = input_data.get('boiler_data', {})
            sensor_feeds = input_data.get('sensor_feeds', {})
            constraints = input_data.get('constraints', {})
            fuel_data = input_data.get('fuel_data', {})
            steam_demand = input_data.get('steam_demand', {})

            # Step 1: Analyze current operational state
            operational_state = await self._analyze_operational_state_async(
                boiler_data, sensor_feeds
            )

            # Step 2: Optimize combustion parameters
            combustion_result = await self._optimize_combustion_async(
                operational_state, fuel_data, constraints
            )

            # Step 3: Optimize steam generation
            steam_strategy = await self._optimize_steam_generation_async(
                steam_demand, operational_state, constraints
            )

            # Step 4: Minimize emissions
            emissions_optimization = await self._minimize_emissions_async(
                combustion_result, constraints.get('emission_limits', {})
            )

            # Step 5: Real-time parameter adjustments
            parameter_adjustments = await self._calculate_parameter_adjustments_async(
                combustion_result, steam_strategy, emissions_optimization
            )

            # Step 6: Generate efficiency KPI dashboard
            kpi_dashboard = self._generate_efficiency_dashboard(
                operational_state,
                combustion_result,
                steam_strategy,
                emissions_optimization,
                parameter_adjustments
            )

            # Step 7: Coordinate sub-agents if needed
            coordination_result = None
            if input_data.get('coordinate_agents', False):
                agent_ids = input_data.get('agent_ids', [])
                commands = input_data.get('agent_commands', {})
                coordination_result = await self._coordinate_agents_async(
                    agent_ids, commands, kpi_dashboard
                )

            # Store in memory for learning and pattern recognition
            self._store_optimization_memory(input_data, kpi_dashboard, parameter_adjustments)

            # Calculate execution metrics
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(execution_time_ms, combustion_result, emissions_optimization)

            # Create comprehensive result
            result = {
                'agent_id': self.config.agent_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'execution_time_ms': round(execution_time_ms, 2),
                'operational_state': self._serialize_operational_state(operational_state),
                'combustion_optimization': combustion_result.__dict__,
                'steam_generation': steam_strategy.__dict__,
                'emissions_optimization': emissions_optimization.__dict__,
                'parameter_adjustments': parameter_adjustments,
                'kpi_dashboard': kpi_dashboard,
                'coordination_result': coordination_result,
                'performance_metrics': self.performance_metrics.copy(),
                'optimization_success': True,
                'provenance_hash': self._calculate_provenance_hash(
                    input_data, kpi_dashboard
                )
            }

            self.state = AgentState.READY
            logger.info(f"Boiler optimization completed in {execution_time_ms:.2f}ms")

            return result

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Boiler optimization failed: {str(e)}", exc_info=True)

            # Attempt recovery
            if self.config.max_retries > 0:
                return await self._handle_error_recovery(e, input_data)
            else:
                raise

    async def _analyze_operational_state_async(
        self,
        boiler_data: Dict[str, Any],
        sensor_feeds: Dict[str, Any]
    ) -> BoilerOperationalState:
        """
        Analyze current boiler operational state asynchronously.

        Args:
            boiler_data: Boiler configuration and status data
            sensor_feeds: Real-time sensor measurements

        Returns:
            Current operational state analysis
        """
        # Check cache
        cache_key = self._get_cache_key('state_analysis', {
            'boiler': boiler_data,
            'sensors': sensor_feeds
        })

        cached_result = self._results_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result

        # Analyze state
        self.performance_metrics['cache_misses'] += 1

        # Determine operation mode based on load and conditions
        load_percent = sensor_feeds.get('load_percent', 50)
        if load_percent < 20:
            mode = OperationMode.LOW_LOAD
        elif load_percent > 85:
            mode = OperationMode.HIGH_EFFICIENCY
        else:
            mode = OperationMode.NORMAL

        # Calculate current efficiency using ASME PTC 4.1 methodology
        result = await asyncio.to_thread(
            self.tools.calculate_boiler_efficiency,
            boiler_data,
            sensor_feeds
        )

        operational_state = BoilerOperationalState(
            mode=mode,
            efficiency_percent=result.thermal_efficiency,
            fuel_flow_rate_kg_hr=sensor_feeds.get('fuel_flow_kg_hr', 0),
            steam_flow_rate_kg_hr=sensor_feeds.get('steam_flow_kg_hr', 0),
            combustion_temperature_c=sensor_feeds.get('combustion_temp_c', 0),
            excess_air_percent=result.excess_air_percent,
            co2_emissions_kg_hr=result.co2_emissions_kg_hr,
            nox_emissions_ppm=sensor_feeds.get('nox_ppm', 0),
            timestamp=datetime.now(timezone.utc)
        )

        # Store in cache
        self._store_in_cache(cache_key, operational_state)
        self.current_state = operational_state
        self.state_history.append(operational_state)

        # Keep history limited
        if len(self.state_history) > 100:
            self.state_history.pop(0)

        self.performance_metrics['optimizations_performed'] += 1

        return operational_state

    async def _optimize_combustion_async(
        self,
        state: BoilerOperationalState,
        fuel_data: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> CombustionOptimizationResult:
        """
        Optimize combustion parameters for maximum efficiency.

        Args:
            state: Current operational state
            fuel_data: Fuel composition and properties
            constraints: Operational constraints

        Returns:
            Optimized combustion parameters
        """
        # Check cache
        cache_key = self._get_cache_key('combustion_opt', {
            'state': state.__dict__,
            'fuel': fuel_data,
            'constraints': constraints
        })

        cached_result = self._results_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result

        # Optimize combustion
        self.performance_metrics['cache_misses'] += 1

        result = await asyncio.to_thread(
            self.tools.optimize_combustion_parameters,
            state.__dict__,
            fuel_data,
            constraints
        )

        # Store in cache
        self._store_in_cache(cache_key, result)
        self.performance_metrics['optimizations_performed'] += 1

        return result

    async def _optimize_steam_generation_async(
        self,
        steam_demand: Dict[str, Any],
        state: BoilerOperationalState,
        constraints: Dict[str, Any]
    ) -> SteamGenerationStrategy:
        """
        Optimize steam generation strategy.

        Args:
            steam_demand: Steam demand requirements
            state: Current operational state
            constraints: Operational constraints

        Returns:
            Optimized steam generation strategy
        """
        result = await asyncio.to_thread(
            self.tools.optimize_steam_generation,
            steam_demand,
            state.__dict__,
            constraints
        )

        self.performance_metrics['total_steam_generated_tons'] += (
            result.target_steam_flow_kg_hr * 0.001  # Convert to tons
        )

        return result

    async def _minimize_emissions_async(
        self,
        combustion_result: CombustionOptimizationResult,
        emission_limits: Dict[str, Any]
    ) -> Any:
        """
        Minimize emissions while maintaining efficiency.

        Args:
            combustion_result: Current combustion optimization
            emission_limits: Regulatory emission limits

        Returns:
            Emissions optimization result
        """
        result = await asyncio.to_thread(
            self.tools.minimize_emissions,
            combustion_result,
            emission_limits
        )

        # Track emissions reduction
        if hasattr(result, 'co2_reduction_kg'):
            self.performance_metrics['emissions_reduced_kg'] += result.co2_reduction_kg

        return result

    async def _calculate_parameter_adjustments_async(
        self,
        combustion_result: CombustionOptimizationResult,
        steam_strategy: SteamGenerationStrategy,
        emissions_optimization: Any
    ) -> Dict[str, Any]:
        """
        Calculate real-time parameter adjustments.

        Args:
            combustion_result: Combustion optimization result
            steam_strategy: Steam generation strategy
            emissions_optimization: Emissions optimization result

        Returns:
            Parameter adjustment recommendations
        """
        adjustments = await asyncio.to_thread(
            self.tools.calculate_control_adjustments,
            combustion_result,
            steam_strategy,
            emissions_optimization
        )

        # Apply safety constraints
        adjustments = self._apply_safety_constraints(adjustments)

        return adjustments

    def _apply_safety_constraints(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply safety constraints to parameter adjustments.

        This method enforces safe rate-of-change limits to prevent equipment damage.

        Args:
            adjustments: Raw parameter adjustments

        Returns:
            Safety-constrained adjustments
        """
        # Rate of change limits
        max_change_rates: Dict[str, float] = {
            'fuel_flow_change_percent': 5.0,  # Max 5% per adjustment
            'air_flow_change_percent': 3.0,   # Max 3% per adjustment
            'steam_pressure_change_bar': 0.5,  # Max 0.5 bar per adjustment
            'temperature_change_c': 10.0       # Max 10°C per adjustment
        }

        constrained: Dict[str, Any] = {}
        for param, value in adjustments.items():
            if param in max_change_rates:
                max_change = max_change_rates[param]
                # Clamp value between -max_change and +max_change
                constrained[param] = max(min(float(value), max_change), -max_change)
            else:
                constrained[param] = value

        return constrained

    def _generate_efficiency_dashboard(
        self,
        state: BoilerOperationalState,
        combustion: CombustionOptimizationResult,
        steam: SteamGenerationStrategy,
        emissions: Any,
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive efficiency KPI dashboard.

        Args:
            state: Current operational state
            combustion: Combustion optimization result
            steam: Steam generation strategy
            emissions: Emissions optimization result
            adjustments: Parameter adjustments

        Returns:
            KPI dashboard dictionary
        """
        # Calculate efficiency improvements
        baseline_efficiency = self.boiler_config.baseline_efficiency_percent
        current_efficiency = state.efficiency_percent
        improvement = current_efficiency - baseline_efficiency

        # Track improvement history
        self.performance_metrics['efficiency_improvements'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'improvement_percent': improvement
        })

        # Limit history size
        if len(self.performance_metrics['efficiency_improvements']) > 100:
            self.performance_metrics['efficiency_improvements'].pop(0)

        dashboard = {
            'operational_kpis': {
                'thermal_efficiency': current_efficiency,
                'efficiency_improvement': improvement,
                'fuel_utilization': combustion.fuel_efficiency_percent,
                'steam_quality_index': steam.steam_quality_index,
                'capacity_utilization': (state.steam_flow_rate_kg_hr /
                                        self.boiler_config.max_steam_capacity_kg_hr * 100)
            },
            'combustion_kpis': {
                'excess_air_percent': state.excess_air_percent,
                'optimal_excess_air': combustion.optimal_excess_air_percent,
                'combustion_efficiency': combustion.combustion_efficiency_percent,
                'flame_stability_index': combustion.flame_stability_index
            },
            'emissions_kpis': {
                'co2_intensity_kg_mwh': emissions.co2_intensity_kg_mwh,
                'nox_emissions_ppm': state.nox_emissions_ppm,
                'compliance_status': emissions.compliance_status,
                'emissions_reduction_percent': emissions.reduction_percent
            },
            'economic_kpis': {
                'fuel_cost_savings_usd_hr': combustion.fuel_savings_usd_hr,
                'efficiency_value_usd_hr': improvement * self.boiler_config.efficiency_value_usd_per_percent,
                'carbon_credit_value_usd_hr': emissions.carbon_credits_usd_hr,
                'total_savings_usd_hr': (combustion.fuel_savings_usd_hr +
                                         improvement * self.boiler_config.efficiency_value_usd_per_percent +
                                         emissions.carbon_credits_usd_hr)
            },
            'control_actions': adjustments,
            'alerts': self._generate_alerts(state, combustion, emissions),
            'recommendations': self._generate_recommendations(state, combustion, steam, emissions)
        }

        return dashboard

    def _generate_alerts(
        self,
        state: BoilerOperationalState,
        combustion: CombustionOptimizationResult,
        emissions: Any
    ) -> List[Dict[str, Any]]:
        """
        Generate operational alerts based on current conditions.

        Args:
            state: Current operational state
            combustion: Combustion result
            emissions: Emissions result

        Returns:
            List of alerts
        """
        alerts = []

        # Efficiency alert
        if state.efficiency_percent < self.boiler_config.min_acceptable_efficiency:
            alerts.append({
                'level': 'warning',
                'category': 'efficiency',
                'message': f'Efficiency {state.efficiency_percent:.1f}% below minimum {self.boiler_config.min_acceptable_efficiency}%',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Emissions alert
        if emissions.compliance_status != "COMPLIANT":
            alerts.append({
                'level': 'critical',
                'category': 'emissions',
                'message': f'Emissions non-compliant: {emissions.violation_details}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Combustion alert
        if combustion.flame_stability_index < 0.7:
            alerts.append({
                'level': 'warning',
                'category': 'combustion',
                'message': f'Flame stability index low: {combustion.flame_stability_index:.2f}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        return alerts

    def _generate_recommendations(
        self,
        state: BoilerOperationalState,
        combustion: CombustionOptimizationResult,
        steam: SteamGenerationStrategy,
        emissions: Any
    ) -> List[str]:
        """
        Generate operational recommendations.

        Args:
            state: Current operational state
            combustion: Combustion result
            steam: Steam strategy
            emissions: Emissions result

        Returns:
            List of recommendations
        """
        recommendations = []

        # Efficiency recommendations
        if state.efficiency_percent < combustion.theoretical_max_efficiency - 5:
            recommendations.append(
                f"Efficiency gap of {combustion.theoretical_max_efficiency - state.efficiency_percent:.1f}% "
                f"detected. Consider heat recovery upgrades or combustion tuning."
            )

        # Excess air recommendations
        if abs(state.excess_air_percent - combustion.optimal_excess_air_percent) > 2:
            recommendations.append(
                f"Adjust excess air from {state.excess_air_percent:.1f}% to "
                f"{combustion.optimal_excess_air_percent:.1f}% for optimal combustion."
            )

        # Steam quality recommendations
        if steam.steam_quality_index < 0.95:
            recommendations.append(
                "Steam quality below optimal. Check water treatment and separator efficiency."
            )

        # Maintenance recommendations
        if self.performance_metrics['optimizations_performed'] > 1000:
            recommendations.append(
                "Schedule preventive maintenance based on optimization cycle count."
            )

        return recommendations

    async def _coordinate_agents_async(
        self,
        agent_ids: List[str],
        commands: Dict[str, Any],
        dashboard: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate multiple boiler optimization agents asynchronously.

        Args:
            agent_ids: List of agent IDs to coordinate
            commands: Commands to distribute
            dashboard: Current KPI dashboard

        Returns:
            Coordination result
        """
        result = await asyncio.to_thread(
            self.tools.coordinate_boiler_agents,
            agent_ids,
            commands,
            dashboard
        )

        self.performance_metrics['agents_coordinated'] += len(agent_ids)

        # Send messages via message bus
        for agent_id, tasks in result['task_assignments'].items():
            for task in tasks:
                message = Message(
                    sender_id=self.config.agent_id,
                    recipient_id=agent_id,
                    message_type='optimization_command',
                    payload={
                        'task': task,
                        'dashboard': dashboard,
                        'priority': task.get('priority', 'normal')
                    },
                    priority=self._map_priority(task.get('priority', 'normal'))
                )
                await self.message_bus.publish(f"agent.{agent_id}", message)

        return result

    def _map_priority(self, priority_str: str) -> int:
        """
        Map string priority to numeric value.

        Args:
            priority_str: Priority level as string

        Returns:
            Numeric priority (1=critical, 4=low)
        """
        priority_map: Dict[str, int] = {
            'critical': 1,
            'high': 2,
            'normal': 3,
            'low': 4
        }
        return priority_map.get(priority_str.lower(), 3)

    def _store_optimization_memory(
        self,
        input_data: Dict[str, Any],
        dashboard: Dict[str, Any],
        adjustments: Dict[str, Any]
    ) -> None:
        """
        Store optimization in memory for learning and pattern recognition.

        Args:
            input_data: Input data for optimization
            dashboard: KPI dashboard result
            adjustments: Parameter adjustments made
        """
        memory_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_summary': self._summarize_input(input_data),
            'result_summary': self._summarize_result(dashboard),
            'adjustments': adjustments,
            'performance': self.performance_metrics.copy()
        }

        # Store in short-term memory
        self.short_term_memory.store(memory_entry)

        # Store optimization in history
        self.optimization_history.append({
            'timestamp': memory_entry['timestamp'],
            'efficiency': dashboard['operational_kpis']['thermal_efficiency'],
            'fuel_savings': dashboard['economic_kpis']['fuel_cost_savings_usd_hr'],
            'emissions_reduction': dashboard['emissions_kpis']['emissions_reduction_percent']
        })

        # Limit history size
        if len(self.optimization_history) > 500:
            self.optimization_history.pop(0)

        # Periodically persist to long-term memory
        if self.performance_metrics['optimizations_performed'] % 50 == 0:
            asyncio.create_task(self._persist_to_long_term_memory())

    async def _persist_to_long_term_memory(self):
        """Persist short-term memories to long-term storage."""
        try:
            recent_memories = self.short_term_memory.retrieve(limit=50)
            for memory in recent_memories:
                await self.long_term_memory.store(
                    key=f"optimization_{memory['timestamp']}",
                    value=memory,
                    category='optimizations'
                )
            logger.debug("Persisted optimization memories to long-term storage")
        except Exception as e:
            logger.error(f"Failed to persist memories: {e}")

    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of input data for memory storage.

        Args:
            input_data: Original input data

        Returns:
            Summarized input data
        """
        return {
            'has_boiler_data': 'boiler_data' in input_data,
            'has_sensor_feeds': 'sensor_feeds' in input_data,
            'has_constraints': 'constraints' in input_data,
            'has_fuel_data': 'fuel_data' in input_data,
            'steam_demand_kg_hr': input_data.get('steam_demand', {}).get('required_flow_kg_hr', 0),
            'coordinate_agents': input_data.get('coordinate_agents', False),
            'data_points': len(input_data.get('sensor_feeds', {}).get('tags', {}))
        }

    def _summarize_result(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of optimization result for memory storage.

        Args:
            dashboard: Optimization result dashboard

        Returns:
            Summarized result data
        """
        return {
            'efficiency': dashboard.get('operational_kpis', {}).get('thermal_efficiency', 0),
            'improvement': dashboard.get('operational_kpis', {}).get('efficiency_improvement', 0),
            'fuel_savings': dashboard.get('economic_kpis', {}).get('fuel_cost_savings_usd_hr', 0),
            'emissions_reduction': dashboard.get('emissions_kpis', {}).get('emissions_reduction_percent', 0),
            'compliance': dashboard.get('emissions_kpis', {}).get('compliance_status', 'UNKNOWN'),
            'alerts': len(dashboard.get('alerts', [])),
            'optimization_status': 'success' if dashboard else 'failure'
        }

    def _serialize_operational_state(self, state: BoilerOperationalState) -> Dict[str, Any]:
        """
        Serialize operational state for JSON output.

        Args:
            state: Current operational state

        Returns:
            Dictionary representation of operational state
        """
        return {
            'mode': state.mode.value,
            'efficiency_percent': state.efficiency_percent,
            'fuel_flow_rate_kg_hr': state.fuel_flow_rate_kg_hr,
            'steam_flow_rate_kg_hr': state.steam_flow_rate_kg_hr,
            'combustion_temperature_c': state.combustion_temperature_c,
            'excess_air_percent': state.excess_air_percent,
            'co2_emissions_kg_hr': state.co2_emissions_kg_hr,
            'nox_emissions_ppm': state.nox_emissions_ppm,
            'timestamp': state.timestamp.isoformat()
        }

    @lru_cache(maxsize=1000)
    def _get_cache_key(self, operation: str, data: Dict[str, Any]) -> str:
        """
        Generate cache key for operation and data.

        Args:
            operation: Operation identifier
            data: Input data

        Returns:
            Cache key string
        """
        # Convert dict to hashable format
        data_str = json.dumps(data, sort_keys=True, default=str)
        return f"{operation}_{hashlib.md5(data_str.encode()).hexdigest()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached result is still valid based on TTL.

        Args:
            cache_key: Cache key to check

        Returns:
            True if cache is valid (not None means valid and not expired)
        """
        # Thread-safe cache handles TTL internally
        return self._results_cache.get(cache_key) is not None

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """
        Store result in thread-safe cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Thread-safe cache handles size limits internally
        self._results_cache.set(cache_key, result)

    def _update_performance_metrics(
        self,
        execution_time_ms: float,
        combustion_result: CombustionOptimizationResult,
        emissions_result: Any
    ):
        """
        Update performance metrics with latest execution.

        Args:
            execution_time_ms: Execution time in milliseconds
            combustion_result: Combustion optimization result
            emissions_result: Emissions optimization result
        """
        # Update average optimization time
        n = self.performance_metrics['optimizations_performed']
        if n > 0:
            current_avg = self.performance_metrics['avg_optimization_time_ms']
            self.performance_metrics['avg_optimization_time_ms'] = (
                (current_avg * (n - 1) + execution_time_ms) / n
            )

        # Update fuel savings
        if hasattr(combustion_result, 'fuel_saved_kg'):
            self.performance_metrics['fuel_savings_kg'] += combustion_result.fuel_saved_kg

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for complete audit trail.

        Args:
            input_data: Input data
            result: Execution result

        Returns:
            SHA-256 hash string
        """
        provenance_str = f"{self.config.agent_id}{input_data}{result}{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def _handle_error_recovery(
        self,
        error: Exception,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle error recovery with retry logic.

        Args:
            error: Exception that occurred
            input_data: Original input data

        Returns:
            Recovery result or error response
        """
        self.state = AgentState.RECOVERING
        self.performance_metrics['errors_recovered'] += 1

        logger.warning(f"Attempting error recovery: {str(error)}")

        # Simplified recovery - return partial results with safe defaults
        return {
            'agent_id': self.config.agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'partial_success',
            'error': str(error),
            'recovered_data': {
                'operational_state': {
                    'mode': 'safe_mode',
                    'efficiency_percent': 0,
                    'status': 'error_recovery'
                },
                'combustion_optimization': {
                    'fuel_efficiency_percent': 0,
                    'status': 'error'
                },
                'steam_generation': {
                    'optimization_score': 0,
                    'status': 'error'
                },
                'kpi_dashboard': {
                    'status': 'limited_data',
                    'message': 'Operating in safe mode due to error recovery'
                }
            },
            'provenance_hash': self._calculate_provenance_hash(input_data, {})
        }

    async def integrate_scada(self, scada_feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate SCADA data feed for real-time monitoring.

        Args:
            scada_feed: Raw SCADA data

        Returns:
            Processed SCADA data
        """
        return await asyncio.to_thread(
            self.tools.process_scada_data,
            scada_feed
        )

    async def integrate_dcs(self, dcs_feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate DCS (Distributed Control System) data feed.

        Args:
            dcs_feed: Raw DCS data

        Returns:
            Processed DCS data
        """
        return await asyncio.to_thread(
            self.tools.process_dcs_data,
            dcs_feed
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state for monitoring.

        Returns:
            Current state dictionary
        """
        return {
            'agent_id': self.config.agent_id,
            'state': self.state.value,
            'version': self.config.version,
            'current_operational_state': (
                self._serialize_operational_state(self.current_state)
                if self.current_state else None
            ),
            'performance_metrics': self.performance_metrics.copy(),
            'cache_size': self._results_cache.size(),
            'memory_entries': self.short_term_memory.size(),
            'optimization_history_size': len(self.optimization_history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def shutdown(self):
        """Graceful shutdown of the orchestrator."""
        logger.info(f"Shutting down BoilerEfficiencyOptimizer {self.config.agent_id}")

        # Persist remaining memories
        await self._persist_to_long_term_memory()

        # Close connections
        if hasattr(self, 'message_bus'):
            await self.message_bus.close()

        self.state = AgentState.TERMINATED
        logger.info(f"BoilerEfficiencyOptimizer {self.config.agent_id} shutdown complete")

    # Required abstract method implementations from BaseAgent

    async def _initialize_core(self) -> None:
        """Initialize agent-specific resources."""
        logger.info("Initializing BoilerEfficiencyOptimizer core components")

        # Initialize tools if not already done
        if not hasattr(self, 'tools'):
            self.tools = BoilerEfficiencyTools()

        # Initialize state tracking
        self.current_state = None
        self.state_history = []
        self.optimization_history = []

        logger.info("BoilerEfficiencyOptimizer core initialization complete")

    async def _execute_core(self, input_data: Any, context: Any) -> Any:
        """
        Core execution logic for the agent.

        Args:
            input_data: Input data to process
            context: Execution context

        Returns:
            Processed output data
        """
        # Delegate to main execute method
        return await self.execute(input_data)

    async def _terminate_core(self) -> None:
        """Perform agent-specific cleanup."""
        logger.info("Terminating BoilerEfficiencyOptimizer core components")

        # Save final state
        if self.current_state:
            final_state = {
                'final_state': self._serialize_operational_state(self.current_state),
                'total_optimizations': self.performance_metrics['optimizations_performed'],
                'total_fuel_saved_kg': self.performance_metrics['fuel_savings_kg'],
                'total_emissions_reduced_kg': self.performance_metrics['emissions_reduced_kg']
            }
            logger.info(f"Final optimization summary: {final_state}")

        # Cleanup resources
        if hasattr(self, 'tools'):
            await asyncio.to_thread(self.tools.cleanup)

        logger.info("BoilerEfficiencyOptimizer core termination complete")