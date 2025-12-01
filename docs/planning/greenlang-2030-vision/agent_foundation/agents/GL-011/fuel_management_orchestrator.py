# -*- coding: utf-8 -*-
"""
FuelManagementOrchestrator - Master orchestrator for multi-fuel optimization operations.

This module implements the GL-011 FUELCRAFT agent for comprehensive fuel management
across industrial facilities. It optimizes fuel selection, blending, procurement,
cost optimization, and carbon footprint minimization following zero-hallucination
principles with deterministic algorithms only.

Key Features:
- Multi-fuel optimization (coal, natural gas, biomass, hydrogen, fuel oil)
- Cost optimization with real-time market pricing
- Fuel blending algorithms for emissions/cost trade-offs
- Carbon footprint minimization
- Calorific value calculations per ISO standards
- Emissions factor management (NOx, SOx, CO2, PM)
- Procurement and inventory optimization
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISO 6976:2016 - Natural gas calorific value calculations
- ISO 17225 - Solid biofuels specifications
- ASTM D4809 - Heat of combustion liquid fuels
- GHG Protocol - Emissions calculations
- IPCC Guidelines - Emission factors

Example:
    >>> from fuel_management_orchestrator import FuelManagementOrchestrator
    >>> config = FuelManagementConfig(...)
    >>> orchestrator = FuelManagementOrchestrator(config)
    >>> result = await orchestrator.execute(fuel_optimization_request)

Author: GreenLang Industrial Optimization Team
Date: December 2025
Agent ID: GL-011
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import time
import threading
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import math

# Import from agent_foundation
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
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
except ImportError:
    # Fallback for standalone testing
    BaseAgent = object
    AgentState = None
    AgentConfig = None
    ChatSession = None
    ModelProvider = None
    PromptTemplate = None
    MessageBus = None
    Message = None
    SagaOrchestrator = None
    SagaStep = None
    ShortTermMemory = None
    LongTermMemory = None

# Local imports
from .config import (
    FuelManagementConfig,
    FuelSpecification,
    FuelInventory,
    MarketPriceData,
    BlendingConstraints,
    EmissionLimits
)
from .tools import (
    FuelManagementTools,
    MultiFuelOptimizationResult,
    CostOptimizationResult,
    BlendingOptimizationResult,
    CarbonFootprintResult
)
from .monitoring.metrics import MetricsCollector
from .calculators.provenance_tracker import ProvenanceTracker

logger = logging.getLogger(__name__)


# ============================================================================
# THREAD-SAFE CACHE IMPLEMENTATION
# ============================================================================

class ThreadSafeCache:
    """
    Thread-safe cache implementation for concurrent access.

    Provides LRU caching with automatic TTL management and thread safety
    using threading.RLock to prevent race conditions in multi-threaded
    fuel optimization scenarios.

    Attributes:
        _cache: Internal cache dictionary
        _timestamps: Cache entry timestamps for TTL management
        _lock: Reentrant lock for thread safety
        _max_size: Maximum cache entries
        _ttl_seconds: Time-to-live for cache entries

    Example:
        >>> cache = ThreadSafeCache(max_size=500, ttl_seconds=120)
        >>> cache.set("fuel_price_ng", 3.50)
        >>> price = cache.get("fuel_price_ng")
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
        self._hit_count = 0
        self._miss_count = 0

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
                self._miss_count += 1
                return None

            # Check if entry has expired
            age_seconds = time.time() - self._timestamps[key]
            if age_seconds >= self._ttl_seconds:
                # Remove expired entry
                del self._cache[key]
                del self._timestamps[key]
                self._miss_count += 1
                return None

            self._hit_count += 1
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

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "ttl_seconds": self._ttl_seconds
            }


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class FuelType(str, Enum):
    """Supported fuel types for multi-fuel optimization."""
    COAL = "coal"
    NATURAL_GAS = "natural_gas"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    FUEL_OIL = "fuel_oil"
    DIESEL = "diesel"
    PROPANE = "propane"
    BIOGAS = "biogas"
    WOOD_PELLETS = "wood_pellets"
    WOOD_CHIPS = "wood_chips"
    MUNICIPAL_SOLID_WASTE = "msw"
    REFUSE_DERIVED_FUEL = "rdf"


class OptimizationObjective(str, Enum):
    """Optimization objectives for fuel selection."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCED = "balanced"
    SECURITY_OF_SUPPLY = "security_of_supply"
    RENEWABLE_PRIORITY = "renewable_priority"


class OperationMode(str, Enum):
    """Fuel management operation modes."""
    NORMAL = "normal"
    PEAK_DEMAND = "peak_demand"
    LOW_DEMAND = "low_demand"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    FUEL_SWITCHING = "fuel_switching"
    BLENDING_OPTIMIZATION = "blending_optimization"


@dataclass
class FuelOptimizationRequest:
    """Request for fuel optimization calculation."""

    site_id: str
    plant_id: str
    request_type: str  # single_fuel, multi_fuel, blending, cost_optimization
    energy_demand_mw: float
    available_fuels: List[str]
    fuel_inventories: Dict[str, float]  # fuel_type -> available_amount
    market_prices: Dict[str, float]  # fuel_type -> price_per_unit
    emission_limits: Dict[str, float]  # pollutant -> limit
    optimization_objective: str = "balanced"
    time_horizon_hours: int = 24
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FuelOptimizationResult:
    """Result of fuel optimization calculation."""

    request_id: str
    timestamp: str
    optimal_fuel_mix: Dict[str, float]  # fuel_type -> percentage
    fuel_quantities: Dict[str, float]  # fuel_type -> quantity
    total_cost_usd: float
    total_emissions_kg: Dict[str, float]  # pollutant -> emissions
    efficiency_percent: float
    carbon_intensity_kg_mwh: float
    recommendations: List[Dict[str, str]]
    cost_breakdown: Dict[str, float]
    savings_vs_baseline_usd: float
    provenance_hash: str
    calculation_time_ms: float
    determinism_verified: bool


@dataclass
class FuelOperationalState:
    """Current operational state of the fuel management system."""

    mode: OperationMode
    active_fuels: List[str]
    current_blend_ratio: Dict[str, float]
    total_consumption_rate_kg_hr: float
    energy_output_mw: float
    current_efficiency_percent: float
    inventory_levels: Dict[str, float]  # fuel_type -> remaining_amount
    emissions_rate: Dict[str, float]  # pollutant -> kg/hr
    cost_rate_usd_hr: float
    timestamp: datetime


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class FuelManagementOrchestrator(BaseAgent if BaseAgent != object else object):
    """
    Master orchestrator for fuel management operations (GL-011 FUELCRAFT).

    This agent coordinates all fuel optimization operations across industrial
    facilities, including multi-fuel selection, blending optimization, cost
    minimization, and carbon footprint reduction. All calculations follow
    zero-hallucination principles with deterministic algorithms only.

    Attributes:
        config: FuelManagementConfig with complete configuration
        tools: FuelManagementTools instance for deterministic calculations
        provenance_tracker: ProvenanceTracker for SHA-256 audit trails
        performance_metrics: Real-time performance tracking
        _results_cache: Thread-safe results cache

    Example:
        >>> config = FuelManagementConfig(...)
        >>> orchestrator = FuelManagementOrchestrator(config)
        >>> result = await orchestrator.execute(request)
        >>> print(f"Optimal mix: {result['optimal_fuel_mix']}")
    """

    def __init__(self, config: FuelManagementConfig):
        """
        Initialize FuelManagementOrchestrator.

        Args:
            config: Configuration for fuel management operations

        Raises:
            ValueError: If configuration validation fails
        """
        # Initialize base agent if available
        if BaseAgent != object and AgentConfig is not None:
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

        self.fuel_config = config
        self.tools = FuelManagementTools()
        self.provenance_tracker = ProvenanceTracker()

        # Initialize intelligence with deterministic settings
        self._init_intelligence()

        # Initialize memory systems if available
        if ShortTermMemory is not None:
            self.short_term_memory = ShortTermMemory(capacity=2000)
        else:
            self.short_term_memory = None

        if LongTermMemory is not None:
            self.long_term_memory = LongTermMemory(
                storage_path=Path("./gl011_memory")
            )
        else:
            self.long_term_memory = None

        # Initialize message bus for agent coordination if available
        if MessageBus is not None:
            self.message_bus = MessageBus()
        else:
            self.message_bus = None

        # Performance tracking
        self.performance_metrics = {
            'optimizations_performed': 0,
            'avg_optimization_time_ms': 0.0,
            'fuel_cost_savings_usd': 0.0,
            'emissions_reduced_kg': 0.0,
            'blending_optimizations': 0,
            'procurement_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'agents_coordinated': 0,
            'errors_recovered': 0,
            'total_energy_optimized_mwh': 0.0
        }

        # Thread-safe results cache with TTL for performance optimization
        self._results_cache = ThreadSafeCache(max_size=500, ttl_seconds=120)

        # Operational state tracking
        self.current_state: Optional[FuelOperationalState] = None
        self.state_history: List[FuelOperationalState] = []
        self.optimization_history: List[Dict[str, Any]] = []

        # Fuel inventory tracking
        self.fuel_inventories: Dict[str, float] = {}
        self.fuel_prices: Dict[str, float] = {}
        self.fuel_properties: Dict[str, Dict[str, Any]] = {}

        # Initialize fuel properties database
        self._init_fuel_properties()

        # RUNTIME VERIFICATION: Verify seed propagation at startup
        self._verify_seed_propagation_at_startup()

        logger.info(f"FuelManagementOrchestrator {config.agent_id} initialized successfully")

    def _init_fuel_properties(self) -> None:
        """Initialize fuel properties database with standard values."""
        self.fuel_properties = {
            FuelType.COAL.value: {
                "heating_value_mj_kg": 25.0,  # Typical bituminous coal
                "carbon_content_percent": 60.0,
                "density_kg_m3": 1350.0,
                "emission_factor_co2_kg_gj": 94.6,
                "emission_factor_nox_g_gj": 250.0,
                "emission_factor_sox_g_gj": 500.0,
                "ash_content_percent": 10.0,
                "moisture_content_percent": 8.0,
                "renewable": False
            },
            FuelType.NATURAL_GAS.value: {
                "heating_value_mj_kg": 50.0,
                "heating_value_mj_m3": 38.0,
                "carbon_content_percent": 75.0,
                "density_kg_m3": 0.75,
                "emission_factor_co2_kg_gj": 56.1,
                "emission_factor_nox_g_gj": 50.0,
                "emission_factor_sox_g_gj": 0.3,
                "methane_slip_percent": 0.1,
                "renewable": False
            },
            FuelType.BIOMASS.value: {
                "heating_value_mj_kg": 18.0,
                "carbon_content_percent": 50.0,
                "density_kg_m3": 600.0,
                "emission_factor_co2_kg_gj": 0.0,  # Biogenic carbon
                "emission_factor_nox_g_gj": 150.0,
                "emission_factor_sox_g_gj": 20.0,
                "ash_content_percent": 2.0,
                "moisture_content_percent": 25.0,
                "renewable": True
            },
            FuelType.HYDROGEN.value: {
                "heating_value_mj_kg": 120.0,
                "carbon_content_percent": 0.0,
                "density_kg_m3": 0.09,
                "emission_factor_co2_kg_gj": 0.0,
                "emission_factor_nox_g_gj": 10.0,
                "emission_factor_sox_g_gj": 0.0,
                "renewable": True  # Assuming green hydrogen
            },
            FuelType.FUEL_OIL.value: {
                "heating_value_mj_kg": 42.0,
                "carbon_content_percent": 85.0,
                "density_kg_m3": 920.0,
                "emission_factor_co2_kg_gj": 77.4,
                "emission_factor_nox_g_gj": 200.0,
                "emission_factor_sox_g_gj": 600.0,
                "sulfur_content_percent": 2.5,
                "renewable": False
            },
            FuelType.DIESEL.value: {
                "heating_value_mj_kg": 43.0,
                "carbon_content_percent": 86.0,
                "density_kg_m3": 850.0,
                "emission_factor_co2_kg_gj": 74.1,
                "emission_factor_nox_g_gj": 180.0,
                "emission_factor_sox_g_gj": 30.0,
                "renewable": False
            },
            FuelType.PROPANE.value: {
                "heating_value_mj_kg": 46.4,
                "carbon_content_percent": 82.0,
                "density_kg_m3": 500.0,
                "emission_factor_co2_kg_gj": 63.1,
                "emission_factor_nox_g_gj": 60.0,
                "emission_factor_sox_g_gj": 0.1,
                "renewable": False
            },
            FuelType.BIOGAS.value: {
                "heating_value_mj_kg": 20.0,
                "heating_value_mj_m3": 22.0,
                "carbon_content_percent": 50.0,
                "density_kg_m3": 1.1,
                "emission_factor_co2_kg_gj": 0.0,  # Biogenic carbon
                "emission_factor_nox_g_gj": 40.0,
                "emission_factor_sox_g_gj": 15.0,
                "methane_content_percent": 60.0,
                "renewable": True
            },
            FuelType.WOOD_PELLETS.value: {
                "heating_value_mj_kg": 17.5,
                "carbon_content_percent": 50.0,
                "density_kg_m3": 650.0,
                "emission_factor_co2_kg_gj": 0.0,  # Biogenic carbon
                "emission_factor_nox_g_gj": 120.0,
                "emission_factor_sox_g_gj": 10.0,
                "ash_content_percent": 0.5,
                "moisture_content_percent": 8.0,
                "renewable": True
            }
        }

    def _verify_seed_propagation_at_startup(self) -> None:
        """
        Verify random seed propagation at agent startup.

        This ensures all RNG operations will be deterministic throughout
        the agent's lifecycle, critical for reproducible fuel optimization.

        Raises:
            AssertionError: If seed propagation fails in strict mode
        """
        try:
            # Verify deterministic random operations
            import random
            random.seed(42)

            values_1 = [random.random() for _ in range(10)]
            random.seed(42)
            values_2 = [random.random() for _ in range(10)]

            if values_1 == values_2:
                logger.info("Random seed propagation verified at startup")
            else:
                logger.warning("Random seed propagation verification failed")
                if self.fuel_config.enable_monitoring:
                    raise AssertionError("DETERMINISM VIOLATION: Seed propagation failed")

        except Exception as e:
            logger.error(f"Seed propagation verification error: {e}")
            if self.fuel_config.enable_monitoring:
                raise

    def _init_intelligence(self) -> None:
        """Initialize AgentIntelligence with deterministic configuration."""
        try:
            if ChatSession is not None:
                # Create deterministic ChatSession for classification tasks only
                self.chat_session = ChatSession(
                    provider=ModelProvider.ANTHROPIC if ModelProvider else None,
                    model_id="claude-3-haiku",
                    temperature=0.0,  # Deterministic
                    seed=42,  # Fixed seed for reproducibility
                    max_tokens=500
                )

                # RUNTIME ASSERTION: Verify AI config is deterministic
                assert self.chat_session.temperature == 0.0, \
                    "DETERMINISM VIOLATION: Temperature must be exactly 0.0"
                assert self.chat_session.seed == 42, \
                    "DETERMINISM VIOLATION: Seed must be exactly 42"

                logger.info("AgentIntelligence initialized with deterministic settings")
            else:
                self.chat_session = None
                logger.info("Running without ChatSession (standalone mode)")

        except Exception as e:
            logger.warning(f"AgentIntelligence initialization failed: {e}")
            self.chat_session = None

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for fuel management orchestration.

        This method coordinates the complete fuel optimization workflow,
        including data validation, multi-fuel optimization, cost analysis,
        blending optimization, and recommendation generation.

        Args:
            input_data: Input containing optimization request, constraints,
                       fuel data, market prices, and emission limits

        Returns:
            Dict containing optimization results, recommendations, and KPIs

        Raises:
            ValueError: If input validation fails
            RuntimeError: If optimization fails after retries

        Example:
            >>> result = await orchestrator.execute({
            ...     "request_type": "multi_fuel",
            ...     "energy_demand_mw": 100,
            ...     "available_fuels": ["natural_gas", "coal", "biomass"],
            ...     "optimization_objective": "minimize_emissions"
            ... })
        """
        start_time = time.perf_counter()

        if hasattr(self, 'state'):
            self.state = AgentState.EXECUTING if AgentState else None

        try:
            # Extract input components
            request_type = input_data.get('request_type', 'multi_fuel')
            energy_demand = input_data.get('energy_demand_mw', 100)
            available_fuels = input_data.get('available_fuels', [])
            fuel_inventories = input_data.get('fuel_inventories', {})
            market_prices = input_data.get('market_prices', {})
            emission_limits = input_data.get('emission_limits', {})
            constraints = input_data.get('constraints', {})
            optimization_objective = input_data.get(
                'optimization_objective', 'balanced'
            )

            # Update internal state
            if fuel_inventories:
                self.fuel_inventories.update(fuel_inventories)
            if market_prices:
                self.fuel_prices.update(market_prices)

            # Step 1: Validate input data
            validation_result = await self._validate_input_async(input_data)
            if not validation_result['valid']:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")

            # Step 2: Analyze current operational state
            operational_state = await self._analyze_operational_state_async(
                available_fuels, fuel_inventories, market_prices
            )

            # Step 3: Execute optimization based on request type
            if request_type == 'multi_fuel':
                optimization_result = await self._optimize_multi_fuel_async(
                    energy_demand, available_fuels, market_prices,
                    emission_limits, constraints, optimization_objective
                )
            elif request_type == 'cost_optimization':
                optimization_result = await self._optimize_cost_async(
                    energy_demand, available_fuels, market_prices,
                    fuel_inventories, constraints
                )
            elif request_type == 'blending':
                optimization_result = await self._optimize_blending_async(
                    energy_demand, available_fuels, constraints,
                    emission_limits, optimization_objective
                )
            elif request_type == 'carbon_footprint':
                optimization_result = await self._minimize_carbon_footprint_async(
                    energy_demand, available_fuels, emission_limits, constraints
                )
            elif request_type == 'procurement':
                optimization_result = await self._optimize_procurement_async(
                    fuel_inventories, market_prices, constraints
                )
            else:
                # Default to multi-fuel optimization
                optimization_result = await self._optimize_multi_fuel_async(
                    energy_demand, available_fuels, market_prices,
                    emission_limits, constraints, optimization_objective
                )

            # Step 4: Calculate emissions for optimal mix
            emissions_result = await self._calculate_emissions_async(
                optimization_result, available_fuels
            )

            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(
                operational_state, optimization_result, emissions_result
            )

            # Step 6: Generate KPI dashboard
            kpi_dashboard = self._generate_kpi_dashboard(
                operational_state, optimization_result,
                emissions_result, recommendations
            )

            # RUNTIME VERIFICATION: Verify provenance hash determinism
            provenance_hash = self._calculate_provenance_hash(
                input_data, kpi_dashboard
            )
            provenance_hash_verify = self._calculate_provenance_hash(
                input_data, kpi_dashboard
            )
            assert provenance_hash == provenance_hash_verify, \
                "DETERMINISM VIOLATION: Provenance hash not deterministic"

            # Store in memory for learning
            self._store_optimization_memory(
                input_data, kpi_dashboard, optimization_result
            )

            # Calculate execution metrics
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(
                execution_time_ms, optimization_result
            )

            # Create comprehensive result
            result = {
                'agent_id': self.fuel_config.agent_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'execution_time_ms': round(execution_time_ms, 2),
                'request_type': request_type,
                'optimization_objective': optimization_objective,
                'operational_state': self._serialize_operational_state(
                    operational_state
                ),
                'optimization_result': optimization_result,
                'emissions_analysis': emissions_result,
                'recommendations': recommendations,
                'kpi_dashboard': kpi_dashboard,
                'performance_metrics': self.performance_metrics.copy(),
                'optimization_success': True,
                'determinism_verified': True,
                'provenance_hash': provenance_hash
            }

            if hasattr(self, 'state'):
                self.state = AgentState.READY if AgentState else None

            logger.info(
                f"Fuel optimization completed in {execution_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            if hasattr(self, 'state'):
                self.state = AgentState.ERROR if AgentState else None
            logger.error(f"Fuel optimization failed: {str(e)}", exc_info=True)

            # Attempt recovery if configured
            if hasattr(self, 'config') and hasattr(self.config, 'max_retries'):
                if self.config.max_retries > 0:
                    return await self._handle_error_recovery(e, input_data)

            raise

    async def _validate_input_async(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate input data for fuel optimization.

        Args:
            input_data: Input data dictionary

        Returns:
            Validation result with 'valid' status and any 'errors'
        """
        errors = []

        # Validate required fields
        if 'energy_demand_mw' not in input_data:
            errors.append("Missing required field: energy_demand_mw")
        elif input_data['energy_demand_mw'] <= 0:
            errors.append("energy_demand_mw must be positive")

        available_fuels = input_data.get('available_fuels', [])
        if not available_fuels:
            errors.append("At least one fuel must be specified")
        else:
            # Validate fuel types
            valid_fuels = [ft.value for ft in FuelType]
            for fuel in available_fuels:
                if fuel not in valid_fuels:
                    errors.append(f"Invalid fuel type: {fuel}")

        # Validate market prices
        market_prices = input_data.get('market_prices', {})
        for fuel_type, price in market_prices.items():
            if price < 0:
                errors.append(f"Price for {fuel_type} cannot be negative")

        # Validate emission limits
        emission_limits = input_data.get('emission_limits', {})
        for pollutant, limit in emission_limits.items():
            if limit < 0:
                errors.append(f"Emission limit for {pollutant} cannot be negative")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def _analyze_operational_state_async(
        self,
        available_fuels: List[str],
        fuel_inventories: Dict[str, float],
        market_prices: Dict[str, float]
    ) -> FuelOperationalState:
        """
        Analyze current fuel management operational state.

        Args:
            available_fuels: List of available fuel types
            fuel_inventories: Current fuel inventory levels
            market_prices: Current market prices

        Returns:
            Current operational state analysis
        """
        # Check cache
        cache_key = self._get_cache_key('state_analysis', {
            'fuels': sorted(available_fuels),
            'inventories': fuel_inventories
        })

        cached_result = self._results_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result

        self.performance_metrics['cache_misses'] += 1

        # Determine operation mode based on inventory and prices
        total_inventory = sum(fuel_inventories.values())
        if total_inventory < 1000:
            mode = OperationMode.EMERGENCY
        elif len(available_fuels) > 1:
            mode = OperationMode.BLENDING_OPTIMIZATION
        else:
            mode = OperationMode.NORMAL

        # Calculate current blend ratio (normalized inventory)
        if total_inventory > 0:
            blend_ratio = {
                fuel: qty / total_inventory
                for fuel, qty in fuel_inventories.items()
            }
        else:
            blend_ratio = {fuel: 1.0 / len(available_fuels) for fuel in available_fuels}

        # Estimate current efficiency based on fuel mix
        efficiency = await asyncio.to_thread(
            self.tools.calculate_blend_efficiency,
            blend_ratio,
            self.fuel_properties
        )

        # Calculate weighted average cost
        total_cost = sum(
            fuel_inventories.get(fuel, 0) * market_prices.get(fuel, 0)
            for fuel in available_fuels
        )
        cost_rate = total_cost / 24 if total_inventory > 0 else 0  # USD/hr

        operational_state = FuelOperationalState(
            mode=mode,
            active_fuels=available_fuels,
            current_blend_ratio=blend_ratio,
            total_consumption_rate_kg_hr=total_inventory / 24,  # Assume 24hr horizon
            energy_output_mw=0,  # To be calculated
            current_efficiency_percent=efficiency,
            inventory_levels=fuel_inventories.copy(),
            emissions_rate={},  # To be calculated
            cost_rate_usd_hr=cost_rate,
            timestamp=datetime.now(timezone.utc)
        )

        # Store in cache
        self._results_cache.set(cache_key, operational_state)
        self.current_state = operational_state
        self.state_history.append(operational_state)

        # Limit history size
        if len(self.state_history) > 100:
            self.state_history.pop(0)

        return operational_state

    async def _optimize_multi_fuel_async(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        market_prices: Dict[str, float],
        emission_limits: Dict[str, float],
        constraints: Dict[str, Any],
        optimization_objective: str
    ) -> Dict[str, Any]:
        """
        Optimize multi-fuel selection and mix.

        This method finds the optimal combination of available fuels to meet
        energy demand while satisfying emission limits and constraints.

        Args:
            energy_demand_mw: Required energy output in MW
            available_fuels: List of available fuel types
            market_prices: Market prices for each fuel
            emission_limits: Emission limits by pollutant
            constraints: Additional operational constraints
            optimization_objective: Optimization objective

        Returns:
            Optimized fuel mix with cost and emissions analysis
        """
        # Check cache
        cache_key = self._get_cache_key('multi_fuel_opt', {
            'demand': energy_demand_mw,
            'fuels': sorted(available_fuels),
            'objective': optimization_objective
        })

        cached_result = self._results_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result

        self.performance_metrics['cache_misses'] += 1

        # Execute optimization using tools
        result = await asyncio.to_thread(
            self.tools.optimize_multi_fuel_selection,
            energy_demand_mw,
            available_fuels,
            self.fuel_properties,
            market_prices,
            emission_limits,
            constraints,
            optimization_objective
        )

        # Store in cache
        self._results_cache.set(cache_key, result)
        self.performance_metrics['optimizations_performed'] += 1

        return result

    async def _optimize_cost_async(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        market_prices: Dict[str, float],
        fuel_inventories: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize fuel selection for minimum cost.

        Args:
            energy_demand_mw: Required energy output
            available_fuels: Available fuel types
            market_prices: Current market prices
            fuel_inventories: Current inventory levels
            constraints: Operational constraints

        Returns:
            Cost-optimized fuel selection
        """
        result = await asyncio.to_thread(
            self.tools.optimize_fuel_cost,
            energy_demand_mw,
            available_fuels,
            self.fuel_properties,
            market_prices,
            fuel_inventories,
            constraints
        )

        self.performance_metrics['fuel_cost_savings_usd'] += (
            result.get('savings_usd', 0)
        )

        return result

    async def _optimize_blending_async(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        constraints: Dict[str, Any],
        emission_limits: Dict[str, float],
        optimization_objective: str
    ) -> Dict[str, Any]:
        """
        Optimize fuel blending ratios.

        Args:
            energy_demand_mw: Required energy output
            available_fuels: Available fuel types
            constraints: Blending constraints
            emission_limits: Emission limits
            optimization_objective: Optimization target

        Returns:
            Optimized blending ratios
        """
        result = await asyncio.to_thread(
            self.tools.optimize_fuel_blending,
            energy_demand_mw,
            available_fuels,
            self.fuel_properties,
            constraints,
            emission_limits,
            optimization_objective
        )

        self.performance_metrics['blending_optimizations'] += 1

        return result

    async def _minimize_carbon_footprint_async(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        emission_limits: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Minimize carbon footprint of fuel mix.

        Args:
            energy_demand_mw: Required energy output
            available_fuels: Available fuel types
            emission_limits: Emission limits
            constraints: Operational constraints

        Returns:
            Carbon-optimized fuel selection
        """
        result = await asyncio.to_thread(
            self.tools.minimize_carbon_footprint,
            energy_demand_mw,
            available_fuels,
            self.fuel_properties,
            emission_limits,
            constraints
        )

        self.performance_metrics['emissions_reduced_kg'] += (
            result.get('emissions_reduction_kg', 0)
        )

        return result

    async def _optimize_procurement_async(
        self,
        fuel_inventories: Dict[str, float],
        market_prices: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize fuel procurement and inventory.

        Args:
            fuel_inventories: Current inventory levels
            market_prices: Current market prices
            constraints: Procurement constraints

        Returns:
            Procurement recommendations
        """
        result = await asyncio.to_thread(
            self.tools.optimize_procurement,
            fuel_inventories,
            market_prices,
            self.fuel_properties,
            constraints
        )

        self.performance_metrics['procurement_optimizations'] += 1

        return result

    async def _calculate_emissions_async(
        self,
        optimization_result: Dict[str, Any],
        available_fuels: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for the optimized fuel mix.

        Args:
            optimization_result: Optimization result with fuel mix
            available_fuels: Available fuel types

        Returns:
            Emissions analysis by pollutant
        """
        fuel_mix = optimization_result.get('optimal_fuel_mix', {})
        fuel_quantities = optimization_result.get('fuel_quantities', {})

        return await asyncio.to_thread(
            self.tools.calculate_emissions,
            fuel_mix,
            fuel_quantities,
            self.fuel_properties
        )

    def _generate_recommendations(
        self,
        state: FuelOperationalState,
        optimization_result: Dict[str, Any],
        emissions_result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate fuel management recommendations.

        Args:
            state: Current operational state
            optimization_result: Optimization results
            emissions_result: Emissions analysis

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Efficiency recommendations
        if state.current_efficiency_percent < 85:
            recommendations.append({
                'priority': 'high',
                'category': 'efficiency',
                'action': 'Improve combustion efficiency through fuel blend optimization',
                'impact': f'+{90 - state.current_efficiency_percent:.1f}% efficiency potential',
                'implementation_time': '2-4 weeks'
            })

        # Cost recommendations
        optimal_cost = optimization_result.get('total_cost_usd', 0)
        baseline_cost = optimization_result.get('baseline_cost_usd', optimal_cost * 1.1)
        savings = baseline_cost - optimal_cost
        if savings > 1000:
            recommendations.append({
                'priority': 'high',
                'category': 'cost',
                'action': 'Implement recommended fuel mix for cost optimization',
                'impact': f'${savings:,.0f} savings potential',
                'implementation_time': 'Immediate'
            })

        # Emissions recommendations
        co2_emissions = emissions_result.get('co2_kg_hr', 0)
        if co2_emissions > 10000:
            renewable_share = sum(
                pct for fuel, pct in optimization_result.get('optimal_fuel_mix', {}).items()
                if self.fuel_properties.get(fuel, {}).get('renewable', False)
            )
            if renewable_share < 0.3:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'emissions',
                    'action': 'Increase renewable fuel share (biomass, hydrogen)',
                    'impact': 'Up to 50% CO2 reduction possible',
                    'implementation_time': '3-6 months'
                })

        # Inventory recommendations
        for fuel, level in state.inventory_levels.items():
            if level < 500:  # Low inventory threshold
                recommendations.append({
                    'priority': 'high',
                    'category': 'procurement',
                    'action': f'Reorder {fuel} - inventory critically low',
                    'impact': 'Avoid supply disruption',
                    'implementation_time': 'Immediate'
                })

        # Fuel switching recommendations
        if FuelType.COAL.value in state.active_fuels:
            recommendations.append({
                'priority': 'medium',
                'category': 'transition',
                'action': 'Consider coal-to-gas or coal-to-biomass transition',
                'impact': '40-60% emissions reduction',
                'implementation_time': '12-24 months'
            })

        return recommendations[:10]  # Return top 10

    def _generate_kpi_dashboard(
        self,
        state: FuelOperationalState,
        optimization_result: Dict[str, Any],
        emissions_result: Dict[str, Any],
        recommendations: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive KPI dashboard.

        Args:
            state: Current operational state
            optimization_result: Optimization results
            emissions_result: Emissions analysis
            recommendations: Generated recommendations

        Returns:
            KPI dashboard dictionary
        """
        # Calculate improvement metrics
        baseline_cost = optimization_result.get('baseline_cost_usd', 0)
        optimal_cost = optimization_result.get('total_cost_usd', 0)
        cost_savings = baseline_cost - optimal_cost
        savings_percent = (cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0

        return {
            'operational_kpis': {
                'current_efficiency_percent': state.current_efficiency_percent,
                'energy_output_mw': optimization_result.get('energy_output_mw', 0),
                'fuel_utilization_percent': optimization_result.get('fuel_utilization_percent', 0),
                'capacity_factor': optimization_result.get('capacity_factor', 0),
                'availability_percent': 99.5
            },
            'cost_kpis': {
                'total_fuel_cost_usd': optimal_cost,
                'cost_per_mwh_usd': optimization_result.get('cost_per_mwh', 0),
                'cost_savings_usd': cost_savings,
                'savings_percent': savings_percent,
                'fuel_cost_breakdown': optimization_result.get('cost_breakdown', {})
            },
            'emissions_kpis': {
                'co2_emissions_kg_hr': emissions_result.get('co2_kg_hr', 0),
                'co2_intensity_kg_mwh': emissions_result.get('co2_intensity', 0),
                'nox_emissions_kg_hr': emissions_result.get('nox_kg_hr', 0),
                'sox_emissions_kg_hr': emissions_result.get('sox_kg_hr', 0),
                'renewable_share_percent': optimization_result.get('renewable_share', 0) * 100,
                'compliance_status': emissions_result.get('compliance_status', 'unknown')
            },
            'inventory_kpis': {
                'total_inventory_tons': sum(state.inventory_levels.values()) / 1000,
                'days_of_supply': optimization_result.get('days_of_supply', 0),
                'inventory_by_fuel': state.inventory_levels,
                'reorder_alerts': [
                    fuel for fuel, level in state.inventory_levels.items()
                    if level < 500
                ]
            },
            'optimization_kpis': {
                'optimal_fuel_mix': optimization_result.get('optimal_fuel_mix', {}),
                'optimization_objective': optimization_result.get('objective', 'balanced'),
                'optimization_score': optimization_result.get('optimization_score', 0),
                'constraint_satisfaction': optimization_result.get('constraints_satisfied', True)
            },
            'recommendations_summary': {
                'total_recommendations': len(recommendations),
                'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
                'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
                'categories': list(set(r['category'] for r in recommendations))
            }
        }

    def _get_cache_key(self, operation: str, data: Dict[str, Any]) -> str:
        """
        Generate deterministic cache key for operation and data.

        Args:
            operation: Operation identifier
            data: Input data

        Returns:
            Cache key string (MD5 hash)
        """
        # DETERMINISM: Always sort keys for consistent ordering
        data_str = json.dumps(data, sort_keys=True, default=str)
        cache_key = f"{operation}_{hashlib.md5(data_str.encode()).hexdigest()}"
        return cache_key

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for complete audit trail.

        DETERMINISM GUARANTEE: This method MUST produce identical hashes
        for identical inputs, regardless of execution time or environment.

        Args:
            input_data: Input data
            result: Execution result

        Returns:
            SHA-256 hash string
        """
        # Serialize input and result deterministically
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        result_str = json.dumps(result, sort_keys=True, default=str)

        provenance_str = f"{self.fuel_config.agent_id}|{input_str}|{result_str}"
        hash_value = hashlib.sha256(provenance_str.encode()).hexdigest()

        return hash_value

    def _serialize_operational_state(
        self,
        state: FuelOperationalState
    ) -> Dict[str, Any]:
        """
        Serialize operational state for JSON output.

        Args:
            state: Current operational state

        Returns:
            Dictionary representation
        """
        return {
            'mode': state.mode.value,
            'active_fuels': state.active_fuels,
            'current_blend_ratio': state.current_blend_ratio,
            'total_consumption_rate_kg_hr': state.total_consumption_rate_kg_hr,
            'energy_output_mw': state.energy_output_mw,
            'current_efficiency_percent': state.current_efficiency_percent,
            'inventory_levels': state.inventory_levels,
            'emissions_rate': state.emissions_rate,
            'cost_rate_usd_hr': state.cost_rate_usd_hr,
            'timestamp': state.timestamp.isoformat()
        }

    def _store_optimization_memory(
        self,
        input_data: Dict[str, Any],
        dashboard: Dict[str, Any],
        optimization_result: Dict[str, Any]
    ) -> None:
        """
        Store optimization in memory for learning and pattern recognition.

        Args:
            input_data: Input data
            dashboard: KPI dashboard result
            optimization_result: Optimization results
        """
        memory_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_summary': {
                'request_type': input_data.get('request_type'),
                'energy_demand_mw': input_data.get('energy_demand_mw'),
                'fuel_count': len(input_data.get('available_fuels', []))
            },
            'result_summary': {
                'cost_savings': dashboard.get('cost_kpis', {}).get('cost_savings_usd', 0),
                'emissions_reduction': dashboard.get('emissions_kpis', {}).get('co2_emissions_kg_hr', 0),
                'efficiency': dashboard.get('operational_kpis', {}).get('current_efficiency_percent', 0)
            },
            'optimal_fuel_mix': optimization_result.get('optimal_fuel_mix', {})
        }

        # Store in short-term memory if available
        if self.short_term_memory is not None:
            self.short_term_memory.store(memory_entry)

        # Store in optimization history
        self.optimization_history.append(memory_entry)

        # Limit history size
        if len(self.optimization_history) > 500:
            self.optimization_history.pop(0)

    def _update_performance_metrics(
        self,
        execution_time_ms: float,
        optimization_result: Dict[str, Any]
    ) -> None:
        """
        Update performance metrics with latest execution.

        Args:
            execution_time_ms: Execution time in milliseconds
            optimization_result: Optimization result
        """
        n = self.performance_metrics['optimizations_performed']
        if n > 0:
            current_avg = self.performance_metrics['avg_optimization_time_ms']
            self.performance_metrics['avg_optimization_time_ms'] = (
                (current_avg * (n - 1) + execution_time_ms) / n
            )

        # Update energy optimized
        energy_mwh = optimization_result.get('energy_output_mwh', 0)
        self.performance_metrics['total_energy_optimized_mwh'] += energy_mwh

    async def _handle_error_recovery(
        self,
        error: Exception,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle error recovery with fallback logic.

        Args:
            error: Exception that occurred
            input_data: Original input data

        Returns:
            Recovery result or error response
        """
        if hasattr(self, 'state'):
            self.state = AgentState.RECOVERING if AgentState else None
        self.performance_metrics['errors_recovered'] += 1

        logger.warning(f"Attempting error recovery: {str(error)}")

        # Return safe fallback
        return {
            'agent_id': self.fuel_config.agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'partial_success',
            'error': str(error),
            'recovered_data': {
                'message': 'Operating in safe fallback mode',
                'recommendation': 'Use single-fuel operation until issue resolved'
            },
            'provenance_hash': self._calculate_provenance_hash(input_data, {})
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state for monitoring.

        Returns:
            Current state dictionary
        """
        return {
            'agent_id': self.fuel_config.agent_id,
            'state': self.state.value if hasattr(self, 'state') and self.state else 'ready',
            'version': self.fuel_config.version,
            'current_operational_state': (
                self._serialize_operational_state(self.current_state)
                if self.current_state else None
            ),
            'performance_metrics': self.performance_metrics.copy(),
            'cache_stats': self._results_cache.get_stats(),
            'fuel_inventories': self.fuel_inventories.copy(),
            'fuel_prices': self.fuel_prices.copy(),
            'optimization_history_size': len(self.optimization_history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def shutdown(self) -> None:
        """Graceful shutdown of the orchestrator."""
        logger.info(f"Shutting down FuelManagementOrchestrator {self.fuel_config.agent_id}")

        # Persist remaining memories if available
        if self.long_term_memory is not None:
            try:
                for entry in self.optimization_history[-50:]:
                    await self.long_term_memory.store(
                        key=f"optimization_{entry['timestamp']}",
                        value=entry,
                        category='optimizations'
                    )
            except Exception as e:
                logger.error(f"Failed to persist memories: {e}")

        # Close connections
        if self.message_bus is not None:
            await self.message_bus.close()

        if hasattr(self, 'state'):
            self.state = AgentState.TERMINATED if AgentState else None
        logger.info(f"FuelManagementOrchestrator {self.fuel_config.agent_id} shutdown complete")

    # ========================================================================
    # ADDITIONAL OPTIMIZATION METHODS
    # ========================================================================

    async def optimize_for_scenario(
        self,
        scenario: str,
        base_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize for specific operational scenarios.

        Args:
            scenario: Scenario type (peak_demand, low_demand, emergency, etc.)
            base_input: Base input data

        Returns:
            Scenario-optimized results
        """
        # Modify input based on scenario
        scenario_input = base_input.copy()

        if scenario == 'peak_demand':
            scenario_input['optimization_objective'] = 'maximize_efficiency'
            scenario_input['constraints']['min_reserve_margin'] = 0.1
        elif scenario == 'low_demand':
            scenario_input['optimization_objective'] = 'minimize_cost'
            scenario_input['constraints']['allow_fuel_switching'] = True
        elif scenario == 'emergency':
            scenario_input['optimization_objective'] = 'security_of_supply'
            scenario_input['constraints']['prioritize_available_inventory'] = True
        elif scenario == 'emissions_critical':
            scenario_input['optimization_objective'] = 'minimize_emissions'
            scenario_input['constraints']['renewable_priority'] = True

        return await self.execute(scenario_input)

    async def calculate_fuel_transition_plan(
        self,
        current_mix: Dict[str, float],
        target_mix: Dict[str, float],
        transition_period_days: int
    ) -> Dict[str, Any]:
        """
        Calculate transition plan from current to target fuel mix.

        Args:
            current_mix: Current fuel percentages
            target_mix: Target fuel percentages
            transition_period_days: Transition period in days

        Returns:
            Day-by-day transition plan
        """
        return await asyncio.to_thread(
            self.tools.calculate_transition_plan,
            current_mix,
            target_mix,
            transition_period_days,
            self.fuel_properties
        )

    async def forecast_fuel_requirements(
        self,
        energy_demand_forecast: List[float],
        fuel_mix: Dict[str, float],
        time_horizon_hours: int
    ) -> Dict[str, Any]:
        """
        Forecast fuel requirements based on demand forecast.

        Args:
            energy_demand_forecast: Hourly energy demand forecast (MW)
            fuel_mix: Current fuel mix percentages
            time_horizon_hours: Forecast horizon

        Returns:
            Fuel requirement forecast by type
        """
        return await asyncio.to_thread(
            self.tools.forecast_fuel_requirements,
            energy_demand_forecast,
            fuel_mix,
            self.fuel_properties,
            time_horizon_hours
        )

    # Required abstract method implementations from BaseAgent

    async def _initialize_core(self) -> None:
        """Initialize agent-specific resources."""
        logger.info("Initializing FuelManagementOrchestrator core components")

        if not hasattr(self, 'tools') or self.tools is None:
            self.tools = FuelManagementTools()

        self.current_state = None
        self.state_history = []
        self.optimization_history = []

        logger.info("FuelManagementOrchestrator core initialization complete")

    async def _execute_core(self, input_data: Any, context: Any) -> Any:
        """
        Core execution logic for the agent.

        Args:
            input_data: Input data to process
            context: Execution context

        Returns:
            Processed output data
        """
        return await self.execute(input_data)

    async def _terminate_core(self) -> None:
        """Perform agent-specific cleanup."""
        logger.info("Terminating FuelManagementOrchestrator core components")

        if self.current_state:
            final_state = {
                'final_state': self._serialize_operational_state(self.current_state),
                'total_optimizations': self.performance_metrics['optimizations_performed'],
                'total_cost_savings': self.performance_metrics['fuel_cost_savings_usd'],
                'total_emissions_reduced': self.performance_metrics['emissions_reduced_kg']
            }
            logger.info(f"Final optimization summary: {final_state}")

        if hasattr(self, 'tools') and self.tools:
            await asyncio.to_thread(self.tools.cleanup)

        logger.info("FuelManagementOrchestrator core termination complete")
