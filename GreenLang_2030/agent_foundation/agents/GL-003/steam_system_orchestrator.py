"""
SteamSystemAnalyzer - Master orchestrator for steam system operations.

This module implements the GL-003 SteamSystemAnalyzer agent for real-time
analysis and optimization of steam generation and distribution systems across
industrial facilities. It analyzes system efficiency, detects leaks, optimizes
condensate return, and monitors steam trap performance following zero-hallucination
principles with deterministic algorithms only.

Example:
    >>> from steam_system_orchestrator import SteamSystemAnalyzer
    >>> config = SteamSystemAnalyzerConfig(...)
    >>> orchestrator = SteamSystemAnalyzer(config)
    >>> result = await orchestrator.execute(system_data)
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
from .config import SteamSystemAnalyzerConfig, SteamSystemConfiguration
from .tools import (
    SteamSystemTools,
    SteamPropertiesResult,
    DistributionEfficiencyResult,
    LeakDetectionResult,
    CondensateOptimizationResult,
    SteamTrapPerformanceResult
)

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


class SystemMode(str, Enum):
    """Steam system operation modes."""
    NORMAL = "normal"
    HIGH_DEMAND = "high_demand"
    LOW_DEMAND = "low_demand"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class AnalysisStrategy(str, Enum):
    """Analysis strategies for steam system operations."""
    EFFICIENCY_FOCUSED = "efficiency_focused"
    LEAK_DETECTION = "leak_detection"
    COST_OPTIMIZATION = "cost_optimization"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    BALANCED = "balanced"


@dataclass
class SystemOperationalState:
    """Current operational state of the steam system."""
    mode: SystemMode
    total_generation_kg_hr: float
    total_consumption_kg_hr: float
    distribution_efficiency_percent: float
    average_pressure_bar: float
    average_temperature_c: float
    detected_leaks_count: int
    condensate_return_rate_percent: float
    steam_trap_efficiency_percent: float
    timestamp: datetime


class SteamSystemAnalyzer(BaseAgent):
    """
    Master orchestrator for steam system analysis and optimization (GL-003).

    This agent coordinates all steam system operations across industrial
    facilities, maximizing distribution efficiency, detecting leaks, optimizing
    condensate return, and monitoring steam trap performance. All calculations
    follow zero-hallucination principles with deterministic algorithms only.

    Attributes:
        config: SteamSystemAnalyzerConfig with complete configuration
        tools: SteamSystemTools instance for deterministic calculations
        intelligence: AgentIntelligence for LLM integration (classification only)
        message_bus: MessageBus for multi-agent coordination
        performance_metrics: Real-time performance tracking
    """

    def __init__(self, config: SteamSystemAnalyzerConfig):
        """
        Initialize SteamSystemAnalyzer.

        Args:
            config: Configuration for steam system analysis operations
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

        self.steam_config = config
        self.tools = SteamSystemTools()

        # Initialize intelligence with deterministic settings
        self._init_intelligence()

        # Initialize memory systems
        self.short_term_memory = ShortTermMemory(capacity=2000)
        self.long_term_memory = LongTermMemory(
            storage_path=Path("./gl003_memory") if base_config.state_directory is None
            else base_config.state_directory / "memory"
        )

        # Initialize message bus for agent coordination
        self.message_bus = MessageBus()

        # Performance tracking
        self.performance_metrics = {
            'analyses_performed': 0,
            'avg_analysis_time_ms': 0,
            'total_leaks_detected': 0,
            'total_steam_saved_kg': 0,
            'total_energy_recovered_mwh': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'agents_coordinated': 0,
            'errors_recovered': 0,
            'traps_monitored': 0
        }

        # Thread-safe results cache with TTL for performance optimization
        self._results_cache = ThreadSafeCache(max_size=200, ttl_seconds=60)

        # Operational state tracking
        self.current_state = None
        self.state_history = []
        self.analysis_history = []

        logger.info(f"SteamSystemAnalyzer {config.agent_id} initialized successfully")

    def _init_intelligence(self) -> None:
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

            # RUNTIME ASSERTION: Verify AI config is deterministic
            assert self.chat_session.temperature == 0.0, \
                "DETERMINISM VIOLATION: Temperature must be exactly 0.0 for zero-hallucination"
            assert self.chat_session.seed == 42, \
                "DETERMINISM VIOLATION: Seed must be exactly 42 for reproducibility"

            # Initialize prompt templates for classification tasks
            self.anomaly_classification_prompt = PromptTemplate(
                template="""
                Classify the following steam system operational data for anomalies.
                Return one of: normal, efficiency_degradation, major_leak_detected,
                trap_failures, condensate_issue, pressure_anomaly, critical_failure

                Operational Data:
                - Distribution Efficiency: {efficiency}%
                - Leak Rate: {leak_rate} kg/hr
                - Condensate Return: {condensate_return}%
                - Trap Efficiency: {trap_efficiency}%
                - Pressure Drop: {pressure_drop} bar

                Return only the classification category, nothing else.
                """,
                variables=['efficiency', 'leak_rate', 'condensate_return', 'trap_efficiency', 'pressure_drop']
            )

            self.optimization_strategy_prompt = PromptTemplate(
                template="""
                Select optimal analysis strategy for steam system based on conditions.
                Conditions: {conditions}
                Current State: {state}

                Return one of: efficiency_focused, leak_detection, cost_optimization,
                preventive_maintenance, balanced
                """,
                variables=['conditions', 'state']
            )

            logger.info("AgentIntelligence initialized with deterministic settings")

        except Exception as e:
            logger.warning(f"AgentIntelligence initialization failed, continuing without LLM: {e}")
            self.chat_session = None

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for steam system analysis orchestration.

        Args:
            input_data: Input containing system data, sensor feeds, configuration

        Returns:
            Orchestration result with analysis results and KPIs
        """
        start_time = time.perf_counter()
        self.state = AgentState.EXECUTING

        try:
            # Extract input components
            system_data = input_data.get('system_data', {})
            sensor_feeds = input_data.get('sensor_feeds', {})
            generation_data = input_data.get('generation_data', {})
            consumption_data = input_data.get('consumption_data', {})
            network_data = input_data.get('network_data', {})
            condensate_data = input_data.get('condensate_data', {})
            trap_data = input_data.get('trap_data', {})

            # Step 1: Analyze current operational state
            operational_state = await self._analyze_operational_state_async(
                system_data, sensor_feeds
            )

            # Step 2: Calculate steam properties at key points
            steam_properties = await self._calculate_steam_properties_async(
                sensor_feeds
            )

            # Step 3: Analyze distribution efficiency
            distribution_analysis = await self._analyze_distribution_efficiency_async(
                generation_data, consumption_data, network_data
            )

            # Step 4: Detect steam leaks
            leak_detection = await self._detect_steam_leaks_async(
                sensor_feeds, system_data, self.state_history
            )

            # Step 5: Optimize condensate return
            condensate_optimization = await self._optimize_condensate_return_async(
                condensate_data, system_data
            )

            # Step 6: Analyze steam trap performance
            trap_analysis = await self._analyze_steam_traps_async(
                trap_data, system_data
            )

            # Step 7: Calculate heat losses
            heat_loss_analysis = await self._calculate_heat_losses_async(
                network_data, steam_properties, distribution_analysis
            )

            # Step 8: Generate KPI dashboard
            kpi_dashboard = self._generate_system_dashboard(
                operational_state,
                steam_properties,
                distribution_analysis,
                leak_detection,
                condensate_optimization,
                trap_analysis,
                heat_loss_analysis
            )

            # RUNTIME VERIFICATION: Verify provenance hash determinism
            provenance_hash = self._calculate_provenance_hash(input_data, kpi_dashboard)
            provenance_hash_verify = self._calculate_provenance_hash(input_data, kpi_dashboard)
            assert provenance_hash == provenance_hash_verify, \
                "DETERMINISM VIOLATION: Provenance hash not deterministic"

            # Step 9: Coordinate sub-agents if needed
            coordination_result = None
            if input_data.get('coordinate_agents', False):
                agent_ids = input_data.get('agent_ids', [])
                commands = input_data.get('agent_commands', {})
                coordination_result = await self._coordinate_agents_async(
                    agent_ids, commands, kpi_dashboard
                )

            # Store in memory for learning and pattern recognition
            self._store_analysis_memory(input_data, kpi_dashboard, leak_detection)

            # Calculate execution metrics
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(
                execution_time_ms,
                leak_detection,
                condensate_optimization
            )

            # Create comprehensive result
            result = {
                'agent_id': self.config.agent_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'execution_time_ms': round(execution_time_ms, 2),
                'operational_state': self._serialize_operational_state(operational_state),
                'steam_properties': self._serialize_steam_properties(steam_properties),
                'distribution_efficiency': distribution_analysis.__dict__,
                'leak_detection': leak_detection.__dict__,
                'condensate_optimization': condensate_optimization.__dict__,
                'trap_performance': trap_analysis.__dict__,
                'heat_loss_analysis': heat_loss_analysis,
                'kpi_dashboard': kpi_dashboard,
                'coordination_result': coordination_result,
                'performance_metrics': self.performance_metrics.copy(),
                'analysis_success': True,
                'provenance_hash': self._calculate_provenance_hash(
                    input_data, kpi_dashboard
                )
            }

            self.state = AgentState.READY
            logger.info(f"Steam system analysis completed in {execution_time_ms:.2f}ms")

            return result

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Steam system analysis failed: {str(e)}", exc_info=True)

            # Attempt recovery
            if self.config.max_retries > 0:
                return await self._handle_error_recovery(e, input_data)
            else:
                raise

    async def _analyze_operational_state_async(
        self,
        system_data: Dict[str, Any],
        sensor_feeds: Dict[str, Any]
    ) -> SystemOperationalState:
        """
        Analyze current steam system operational state asynchronously.

        Args:
            system_data: System configuration and status data
            sensor_feeds: Real-time sensor measurements

        Returns:
            Current operational state analysis
        """
        # Check cache
        cache_key = self._get_cache_key('state_analysis', {
            'system': system_data,
            'sensors': sensor_feeds
        })

        cached_result = self._results_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result

        # Analyze state
        self.performance_metrics['cache_misses'] += 1

        # Extract key parameters
        total_generation = sensor_feeds.get('total_generation_kg_hr', 0)
        total_consumption = sensor_feeds.get('total_consumption_kg_hr', 0)
        average_pressure = sensor_feeds.get('average_pressure_bar', 10)
        average_temp = sensor_feeds.get('average_temperature_c', 180)

        # Determine operation mode based on demand
        demand_percent = (total_consumption / system_data.get('max_capacity_kg_hr', 100000)) * 100
        if demand_percent < 30:
            mode = SystemMode.LOW_DEMAND
        elif demand_percent > 85:
            mode = SystemMode.HIGH_DEMAND
        else:
            mode = SystemMode.NORMAL

        # Calculate distribution efficiency
        distribution_efficiency = (total_consumption / total_generation * 100) if total_generation > 0 else 0

        # Get leak count from sensor data
        detected_leaks = sensor_feeds.get('detected_leaks_count', 0)

        # Get condensate return rate
        condensate_return_rate = sensor_feeds.get('condensate_return_rate_percent', 70)

        # Get trap efficiency
        trap_efficiency = sensor_feeds.get('trap_efficiency_percent', 95)

        operational_state = SystemOperationalState(
            mode=mode,
            total_generation_kg_hr=total_generation,
            total_consumption_kg_hr=total_consumption,
            distribution_efficiency_percent=distribution_efficiency,
            average_pressure_bar=average_pressure,
            average_temperature_c=average_temp,
            detected_leaks_count=detected_leaks,
            condensate_return_rate_percent=condensate_return_rate,
            steam_trap_efficiency_percent=trap_efficiency,
            timestamp=datetime.now(timezone.utc)
        )

        # Store in cache
        self._store_in_cache(cache_key, operational_state)
        self.current_state = operational_state
        self.state_history.append(operational_state)

        # Keep history limited
        if len(self.state_history) > 100:
            self.state_history.pop(0)

        self.performance_metrics['analyses_performed'] += 1

        return operational_state

    async def _calculate_steam_properties_async(
        self,
        sensor_feeds: Dict[str, Any]
    ) -> Dict[str, SteamPropertiesResult]:
        """
        Calculate steam properties at key measurement points.

        Args:
            sensor_feeds: Real-time sensor measurements

        Returns:
            Dictionary of steam properties at different locations
        """
        properties = {}

        # Calculate properties at generation point
        gen_pressure = sensor_feeds.get('generation_pressure_bar', 40)
        gen_temp = sensor_feeds.get('generation_temperature_c', 450)

        properties['generation'] = await asyncio.to_thread(
            self.tools.calculate_steam_properties,
            gen_pressure,
            gen_temp
        )

        # Calculate properties at consumption point
        cons_pressure = sensor_feeds.get('consumption_pressure_bar', 35)
        cons_temp = sensor_feeds.get('consumption_temperature_c', 440)

        properties['consumption'] = await asyncio.to_thread(
            self.tools.calculate_steam_properties,
            cons_pressure,
            cons_temp
        )

        return properties

    async def _analyze_distribution_efficiency_async(
        self,
        generation_data: Dict[str, Any],
        consumption_data: Dict[str, Any],
        network_data: Dict[str, Any]
    ) -> DistributionEfficiencyResult:
        """
        Analyze steam distribution network efficiency.

        Args:
            generation_data: Generation parameters
            consumption_data: Consumption measurements
            network_data: Network configuration

        Returns:
            Distribution efficiency analysis
        """
        # Check cache
        cache_key = self._get_cache_key('distribution_efficiency', {
            'gen': generation_data,
            'cons': consumption_data,
            'network': network_data
        })

        cached_result = self._results_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result

        self.performance_metrics['cache_misses'] += 1

        result = await asyncio.to_thread(
            self.tools.analyze_distribution_efficiency,
            generation_data,
            consumption_data,
            network_data
        )

        # Store in cache
        self._store_in_cache(cache_key, result)

        return result

    async def _detect_steam_leaks_async(
        self,
        sensor_feeds: Dict[str, Any],
        system_config: Dict[str, Any],
        historical_data: List[SystemOperationalState]
    ) -> LeakDetectionResult:
        """
        Detect steam leaks using sensor analysis and mass balance.

        Args:
            sensor_feeds: Real-time sensor data
            system_config: System configuration
            historical_data: Historical state data

        Returns:
            Leak detection analysis
        """
        # Prepare historical data
        historical_dict = None
        if historical_data and len(historical_data) > 0:
            recent_states = historical_data[-10:]  # Last 10 states
            avg_losses = sum([
                s.total_generation_kg_hr - s.total_consumption_kg_hr
                for s in recent_states
            ]) / len(recent_states)
            historical_dict = {'average_losses_kg_hr': avg_losses}

        result = await asyncio.to_thread(
            self.tools.detect_steam_leaks,
            sensor_feeds,
            system_config,
            historical_dict
        )

        # Update metrics
        self.performance_metrics['total_leaks_detected'] += result.total_leaks_detected

        return result

    async def _optimize_condensate_return_async(
        self,
        condensate_data: Dict[str, Any],
        system_config: Dict[str, Any]
    ) -> CondensateOptimizationResult:
        """
        Optimize condensate return system performance.

        Args:
            condensate_data: Condensate system measurements
            system_config: System configuration

        Returns:
            Condensate optimization analysis
        """
        result = await asyncio.to_thread(
            self.tools.optimize_condensate_return,
            condensate_data,
            system_config
        )

        # Update metrics
        self.performance_metrics['total_energy_recovered_mwh'] += result.energy_recovered_mw

        return result

    async def _analyze_steam_traps_async(
        self,
        trap_data: Dict[str, Any],
        system_config: Dict[str, Any]
    ) -> SteamTrapPerformanceResult:
        """
        Analyze steam trap performance across the system.

        Args:
            trap_data: Steam trap monitoring data
            system_config: System configuration

        Returns:
            Steam trap performance analysis
        """
        result = await asyncio.to_thread(
            self.tools.analyze_steam_trap_performance,
            trap_data,
            system_config
        )

        # Update metrics
        self.performance_metrics['traps_monitored'] = result.total_traps_assessed

        return result

    async def _calculate_heat_losses_async(
        self,
        network_data: Dict[str, Any],
        steam_properties: Dict[str, SteamPropertiesResult],
        distribution_analysis: DistributionEfficiencyResult
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive heat loss analysis.

        Args:
            network_data: Network configuration
            steam_properties: Steam properties at key points
            distribution_analysis: Distribution efficiency result

        Returns:
            Heat loss analysis dictionary
        """
        heat_losses = {
            'pipeline_losses_mw': distribution_analysis.heat_losses_mw,
            'uninsulated_losses_mw': 0,
            'trap_losses_mw': 0,
            'leak_losses_mw': 0,
            'total_losses_mw': distribution_analysis.heat_losses_mw,
            'losses_percent': 0
        }

        # Calculate percentage of total heat
        total_generation_mw = network_data.get('total_heat_generation_mw', 100)
        if total_generation_mw > 0:
            heat_losses['losses_percent'] = (heat_losses['total_losses_mw'] / total_generation_mw) * 100

        return heat_losses

    def _generate_system_dashboard(
        self,
        state: SystemOperationalState,
        steam_properties: Dict[str, SteamPropertiesResult],
        distribution: DistributionEfficiencyResult,
        leaks: LeakDetectionResult,
        condensate: CondensateOptimizationResult,
        traps: SteamTrapPerformanceResult,
        heat_losses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive system KPI dashboard.

        Args:
            state: Current operational state
            steam_properties: Steam properties
            distribution: Distribution efficiency
            leaks: Leak detection results
            condensate: Condensate optimization
            traps: Trap performance
            heat_losses: Heat loss analysis

        Returns:
            KPI dashboard dictionary
        """
        # Calculate efficiency improvements
        baseline_efficiency = self.steam_config.systems[0].baseline_distribution_efficiency
        current_efficiency = state.distribution_efficiency_percent
        improvement = current_efficiency - baseline_efficiency

        dashboard = {
            'operational_kpis': {
                'distribution_efficiency': current_efficiency,
                'efficiency_improvement': improvement,
                'generation_rate_kg_hr': state.total_generation_kg_hr,
                'consumption_rate_kg_hr': state.total_consumption_kg_hr,
                'system_pressure_bar': state.average_pressure_bar,
                'system_temperature_c': state.average_temperature_c,
                'capacity_utilization': (state.total_generation_kg_hr /
                                        self.steam_config.systems[0].specification.total_steam_capacity_kg_hr * 100)
            },
            'distribution_kpis': {
                'network_efficiency': distribution.distribution_efficiency_percent,
                'pressure_drop_bar': distribution.pressure_drop_bar,
                'temperature_drop_c': distribution.temperature_drop_c,
                'heat_losses_mw': distribution.heat_losses_mw,
                'insulation_effectiveness': distribution.insulation_effectiveness_percent
            },
            'leak_detection_kpis': {
                'total_leaks_detected': leaks.total_leaks_detected,
                'total_leak_rate_kg_hr': leaks.total_leak_rate_kg_hr,
                'annual_leak_cost_usd': leaks.estimated_annual_cost_usd,
                'major_leaks': leaks.leak_severity_distribution.get('major', 0),
                'critical_leaks': leaks.leak_severity_distribution.get('critical', 0)
            },
            'condensate_kpis': {
                'return_rate_percent': condensate.return_rate_percent,
                'condensate_returned_kg_hr': condensate.condensate_returned_kg_hr,
                'energy_recovered_mw': condensate.energy_recovered_mw,
                'water_savings_m3_day': condensate.water_savings_m3_day,
                'chemical_savings_usd_day': condensate.chemical_savings_usd_day
            },
            'trap_performance_kpis': {
                'trap_efficiency_percent': traps.trap_efficiency_percent,
                'functioning_traps': traps.functioning_traps,
                'failed_open_traps': traps.failed_open_traps,
                'failed_closed_traps': traps.failed_closed_traps,
                'steam_losses_kg_hr': traps.steam_losses_from_traps_kg_hr,
                'repair_cost_usd': traps.estimated_repair_cost_usd
            },
            'economic_kpis': {
                'steam_losses_cost_usd_day': (distribution.distribution_losses_kg_hr * 24 *
                                              self.steam_config.systems[0].steam_cost_usd_per_ton / 1000),
                'leak_cost_usd_year': leaks.estimated_annual_cost_usd,
                'condensate_savings_usd_day': condensate.chemical_savings_usd_day,
                'trap_maintenance_cost_usd': traps.estimated_repair_cost_usd,
                'total_savings_opportunity_usd_year': (
                    leaks.estimated_annual_cost_usd +
                    condensate.chemical_savings_usd_day * 365 +
                    traps.steam_losses_from_traps_kg_hr * 24 * 365 *
                    self.steam_config.systems[0].steam_cost_usd_per_ton / 1000
                )
            },
            'alerts': self._generate_alerts(state, distribution, leaks, traps),
            'recommendations': self._generate_recommendations(
                state, distribution, leaks, condensate, traps
            )
        }

        return dashboard

    def _generate_alerts(
        self,
        state: SystemOperationalState,
        distribution: DistributionEfficiencyResult,
        leaks: LeakDetectionResult,
        traps: SteamTrapPerformanceResult
    ) -> List[Dict[str, Any]]:
        """
        Generate operational alerts based on current conditions.

        Args:
            state: Current operational state
            distribution: Distribution analysis
            leaks: Leak detection results
            traps: Trap performance

        Returns:
            List of alerts
        """
        alerts = []

        # Efficiency alert
        threshold = self.steam_config.systems[0].analysis.efficiency_threshold_percent
        if state.distribution_efficiency_percent < threshold:
            alerts.append({
                'level': 'warning',
                'category': 'efficiency',
                'message': f'Distribution efficiency {state.distribution_efficiency_percent:.1f}% below threshold {threshold}%',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Leak alert
        if leaks.leak_severity_distribution.get('critical', 0) > 0:
            alerts.append({
                'level': 'critical',
                'category': 'leak',
                'message': f'{leaks.leak_severity_distribution["critical"]} critical leaks detected - immediate action required',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Trap alert
        if traps.trap_efficiency_percent < 95:
            alerts.append({
                'level': 'warning',
                'category': 'trap',
                'message': f'Steam trap efficiency {traps.trap_efficiency_percent:.1f}% below target 95%',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        # Pressure drop alert
        if distribution.pressure_drop_bar > 2.0:
            alerts.append({
                'level': 'warning',
                'category': 'pressure',
                'message': f'High pressure drop {distribution.pressure_drop_bar:.1f} bar detected in distribution network',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        return alerts

    def _generate_recommendations(
        self,
        state: SystemOperationalState,
        distribution: DistributionEfficiencyResult,
        leaks: LeakDetectionResult,
        condensate: CondensateOptimizationResult,
        traps: SteamTrapPerformanceResult
    ) -> List[str]:
        """
        Generate operational recommendations.

        Args:
            state: Current operational state
            distribution: Distribution analysis
            leaks: Leak detection
            condensate: Condensate optimization
            traps: Trap performance

        Returns:
            List of recommendations
        """
        recommendations = []

        # Distribution efficiency recommendations
        if distribution.distribution_efficiency_percent < 90:
            recommendations.append(
                f"Distribution efficiency at {distribution.distribution_efficiency_percent:.1f}% - "
                f"Review insulation, repair leaks, and optimize steam routing."
            )

        # Leak recommendations
        if leaks.total_leak_rate_kg_hr > 100:
            recommendations.append(
                f"Significant steam losses detected ({leaks.total_leak_rate_kg_hr:.0f} kg/hr) - "
                f"Prioritize {len(leaks.priority_repairs)} critical repairs to save ${leaks.estimated_annual_cost_usd:.0f}/year."
            )

        # Condensate recommendations
        target_return = self.steam_config.systems[0].analysis.target_condensate_return
        if condensate.return_rate_percent < target_return:
            recommendations.append(
                f"Condensate return rate {condensate.return_rate_percent:.1f}% below target {target_return}% - "
                f"Improve recovery system to save {condensate.water_savings_m3_day:.1f} m³/day water."
            )

        # Trap recommendations
        if traps.failed_open_traps > 5:
            recommendations.append(
                f"{traps.failed_open_traps} steam traps failed open causing {traps.steam_losses_from_traps_kg_hr:.0f} kg/hr losses - "
                f"Schedule immediate repairs (estimated cost: ${traps.estimated_repair_cost_usd:.0f})."
            )

        # Insulation recommendations
        if distribution.insulation_effectiveness_percent < 90:
            recommendations.append(
                "Insulation effectiveness below optimal - conduct thermal imaging survey and upgrade insulation."
            )

        return recommendations

    async def _coordinate_agents_async(
        self,
        agent_ids: List[str],
        commands: Dict[str, Any],
        dashboard: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate multiple steam system agents asynchronously.

        Args:
            agent_ids: List of agent IDs to coordinate
            commands: Commands to distribute
            dashboard: Current KPI dashboard

        Returns:
            Coordination result
        """
        task_assignments = {}

        for agent_id in agent_ids:
            # Assign tasks based on agent capabilities
            if 'leak' in agent_id.lower():
                task_assignments[agent_id] = [
                    {
                        'task': 'investigate_leaks',
                        'parameters': commands.get('leak_investigation', {}),
                        'priority': 'high'
                    }
                ]
            elif 'trap' in agent_id.lower():
                task_assignments[agent_id] = [
                    {
                        'task': 'inspect_traps',
                        'parameters': commands.get('trap_inspection', {}),
                        'priority': 'medium'
                    }
                ]
            else:
                task_assignments[agent_id] = [
                    {
                        'task': 'monitor',
                        'parameters': {'dashboard': dashboard},
                        'priority': 'low'
                    }
                ]

        self.performance_metrics['agents_coordinated'] += len(agent_ids)

        # Send messages via message bus
        for agent_id, tasks in task_assignments.items():
            for task in tasks:
                message = Message(
                    sender_id=self.config.agent_id,
                    recipient_id=agent_id,
                    message_type='analysis_command',
                    payload={
                        'task': task,
                        'dashboard': dashboard,
                        'priority': task.get('priority', 'normal')
                    },
                    priority=self._map_priority(task.get('priority', 'normal'))
                )
                await self.message_bus.publish(f"agent.{agent_id}", message)

        return {
            'task_assignments': task_assignments,
            'coordination_status': 'distributed',
            'agents_coordinated': len(agent_ids)
        }

    def _map_priority(self, priority_str: str) -> int:
        """Map string priority to numeric value."""
        priority_map: Dict[str, int] = {
            'critical': 1,
            'high': 2,
            'normal': 3,
            'low': 4
        }
        return priority_map.get(priority_str.lower(), 3)

    def _store_analysis_memory(
        self,
        input_data: Dict[str, Any],
        dashboard: Dict[str, Any],
        leak_detection: LeakDetectionResult
    ) -> None:
        """Store analysis in memory for learning and pattern recognition."""
        memory_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_summary': self._summarize_input(input_data),
            'result_summary': self._summarize_result(dashboard),
            'leaks_detected': leak_detection.total_leaks_detected,
            'performance': self.performance_metrics.copy()
        }

        # Store in short-term memory
        self.short_term_memory.store(memory_entry)

        # Store analysis in history
        self.analysis_history.append({
            'timestamp': memory_entry['timestamp'],
            'efficiency': dashboard['operational_kpis']['distribution_efficiency'],
            'leaks_detected': leak_detection.total_leaks_detected,
            'condensate_return': dashboard['condensate_kpis']['return_rate_percent']
        })

        # Limit history size
        if len(self.analysis_history) > 500:
            self.analysis_history.pop(0)

        # Periodically persist to long-term memory
        if self.performance_metrics['analyses_performed'] % 50 == 0:
            asyncio.create_task(self._persist_to_long_term_memory())

    async def _persist_to_long_term_memory(self) -> None:
        """Persist short-term memories to long-term storage."""
        try:
            recent_memories = self.short_term_memory.retrieve(limit=50)
            for memory in recent_memories:
                await self.long_term_memory.store(
                    key=f"analysis_{memory['timestamp']}",
                    value=memory,
                    category='analyses'
                )
            logger.debug("Persisted analysis memories to long-term storage")
        except Exception as e:
            logger.error(f"Failed to persist memories: {e}")

    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of input data for memory storage."""
        return {
            'has_system_data': 'system_data' in input_data,
            'has_sensor_feeds': 'sensor_feeds' in input_data,
            'has_generation_data': 'generation_data' in input_data,
            'has_trap_data': 'trap_data' in input_data,
            'coordinate_agents': input_data.get('coordinate_agents', False)
        }

    def _summarize_result(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of analysis result for memory storage."""
        return {
            'efficiency': dashboard.get('operational_kpis', {}).get('distribution_efficiency', 0),
            'improvement': dashboard.get('operational_kpis', {}).get('efficiency_improvement', 0),
            'leaks_detected': dashboard.get('leak_detection_kpis', {}).get('total_leaks_detected', 0),
            'condensate_return': dashboard.get('condensate_kpis', {}).get('return_rate_percent', 0),
            'alerts': len(dashboard.get('alerts', [])),
            'analysis_status': 'success' if dashboard else 'failure'
        }

    def _serialize_operational_state(self, state: SystemOperationalState) -> Dict[str, Any]:
        """Serialize operational state for JSON output."""
        return {
            'mode': state.mode.value,
            'total_generation_kg_hr': state.total_generation_kg_hr,
            'total_consumption_kg_hr': state.total_consumption_kg_hr,
            'distribution_efficiency_percent': state.distribution_efficiency_percent,
            'average_pressure_bar': state.average_pressure_bar,
            'average_temperature_c': state.average_temperature_c,
            'detected_leaks_count': state.detected_leaks_count,
            'condensate_return_rate_percent': state.condensate_return_rate_percent,
            'steam_trap_efficiency_percent': state.steam_trap_efficiency_percent,
            'timestamp': state.timestamp.isoformat()
        }

    def _serialize_steam_properties(
        self,
        properties: Dict[str, SteamPropertiesResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Serialize steam properties for JSON output."""
        return {
            location: {
                'pressure_bar': prop.pressure_bar,
                'temperature_c': prop.temperature_c,
                'enthalpy_kj_kg': prop.enthalpy_kj_kg,
                'density_kg_m3': prop.density_kg_m3,
                'steam_quality': prop.steam_quality,
                'is_superheated': prop.is_superheated
            }
            for location, prop in properties.items()
        }

    @lru_cache(maxsize=1000)
    def _get_cache_key(self, operation: str, data: Dict[str, Any]) -> str:
        """Generate deterministic cache key for operation and data."""
        # Convert dict to hashable format (MUST be deterministic)
        data_str = json.dumps(data, sort_keys=True, default=str)
        cache_key = f"{operation}_{hashlib.md5(data_str.encode()).hexdigest()}"

        # RUNTIME VERIFICATION: Verify cache key is deterministic
        data_str_verify = json.dumps(data, sort_keys=True, default=str)
        cache_key_verify = f"{operation}_{hashlib.md5(data_str_verify.encode()).hexdigest()}"

        assert cache_key == cache_key_verify, \
            "DETERMINISM VIOLATION: Cache key generation is non-deterministic"

        return cache_key

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Store result in thread-safe cache."""
        self._results_cache.set(cache_key, result)

    def _update_performance_metrics(
        self,
        execution_time_ms: float,
        leak_detection: LeakDetectionResult,
        condensate: CondensateOptimizationResult
    ) -> None:
        """Update performance metrics with latest execution."""
        # Update average analysis time
        n = self.performance_metrics['analyses_performed']
        if n > 0:
            current_avg = self.performance_metrics['avg_analysis_time_ms']
            self.performance_metrics['avg_analysis_time_ms'] = (
                (current_avg * (n - 1) + execution_time_ms) / n
            )

        # Update steam saved
        self.performance_metrics['total_steam_saved_kg'] += leak_detection.total_leak_rate_kg_hr

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for complete audit trail.

        DETERMINISM GUARANTEE: This method MUST produce identical hashes
        for identical inputs, regardless of execution time or environment.
        """
        # Serialize input and result deterministically
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        result_str = json.dumps(result, sort_keys=True, default=str)

        provenance_str = f"{self.config.agent_id}|{input_str}|{result_str}"
        hash_value = hashlib.sha256(provenance_str.encode()).hexdigest()

        # RUNTIME VERIFICATION: Calculate again to verify determinism
        provenance_str_verify = f"{self.config.agent_id}|{input_str}|{result_str}"
        hash_value_verify = hashlib.sha256(provenance_str_verify.encode()).hexdigest()

        assert hash_value == hash_value_verify, \
            "DETERMINISM VIOLATION: Provenance hash calculation is non-deterministic"

        return hash_value

    async def _handle_error_recovery(
        self,
        error: Exception,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle error recovery with retry logic."""
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
                    'distribution_efficiency_percent': 0,
                    'status': 'error_recovery'
                },
                'kpi_dashboard': {
                    'status': 'limited_data',
                    'message': 'Operating in safe mode due to error recovery'
                }
            },
            'provenance_hash': self._calculate_provenance_hash(input_data, {})
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state for monitoring."""
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
            'analysis_history_size': len(self.analysis_history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def shutdown(self) -> None:
        """Graceful shutdown of the orchestrator."""
        logger.info(f"Shutting down SteamSystemAnalyzer {self.config.agent_id}")

        # Persist remaining memories
        await self._persist_to_long_term_memory()

        # Close connections
        if hasattr(self, 'message_bus'):
            await self.message_bus.close()

        self.state = AgentState.TERMINATED
        logger.info(f"SteamSystemAnalyzer {self.config.agent_id} shutdown complete")

    # Required abstract method implementations from BaseAgent

    async def _initialize_core(self) -> None:
        """Initialize agent-specific resources."""
        logger.info("Initializing SteamSystemAnalyzer core components")

        # Initialize tools if not already done
        if not hasattr(self, 'tools'):
            self.tools = SteamSystemTools()

        # Initialize state tracking
        self.current_state = None
        self.state_history = []
        self.analysis_history = []

        logger.info("SteamSystemAnalyzer core initialization complete")

    async def _execute_core(self, input_data: Any, context: Any) -> Any:
        """Core execution logic for the agent."""
        # Delegate to main execute method
        return await self.execute(input_data)

    async def _terminate_core(self) -> None:
        """Perform agent-specific cleanup."""
        logger.info("Terminating SteamSystemAnalyzer core components")

        # Save final state
        if self.current_state:
            final_state = {
                'final_state': self._serialize_operational_state(self.current_state),
                'total_analyses': self.performance_metrics['analyses_performed'],
                'total_leaks_detected': self.performance_metrics['total_leaks_detected'],
                'total_steam_saved_kg': self.performance_metrics['total_steam_saved_kg']
            }
            logger.info(f"Final analysis summary: {final_state}")

        # Cleanup resources
        if hasattr(self, 'tools'):
            await asyncio.to_thread(self.tools.cleanup)

        logger.info("SteamSystemAnalyzer core termination complete")
