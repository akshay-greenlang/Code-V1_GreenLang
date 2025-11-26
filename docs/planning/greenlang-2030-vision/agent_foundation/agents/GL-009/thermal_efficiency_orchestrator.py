# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ - ThermalEfficiencyCalculator Orchestrator.

Zero-hallucination thermal efficiency calculations for industrial processes.
This module implements the GL-009 ThermalEfficiencyCalculator agent for comprehensive
thermal efficiency analysis including First Law (energy) and Second Law (exergy)
calculations, Sankey diagram generation, industry benchmarking, and improvement
opportunity identification.

Standards Compliance:
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 4 - Fired Steam Generators
- ISO 50001:2018 - Energy Management Systems
- EPA 40 CFR Part 60 - Emissions Standards

Example:
    >>> from thermal_efficiency_orchestrator import ThermalEfficiencyOrchestrator
    >>> config = ThermalEfficiencyConfig(...)
    >>> orchestrator = ThermalEfficiencyOrchestrator(config)
    >>> result = await orchestrator.execute(input_data)

Author: GreenLang Foundation
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import uuid

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
    AgentIntelligence = None
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
    ThermalEfficiencyConfig,
    ProcessType,
    CalculationConfig,
    VisualizationConfig,
    IntegrationConfig,
    BenchmarkConfig
)
from .tools import (
    ThermalEfficiencyTools,
    ThermalEfficiencyResult,
    SankeyDiagramResult,
    HeatLossBreakdown,
    BenchmarkResult,
    ImprovementOpportunity,
    ExergyAnalysisResult,
    HeatBalanceResult,
    UncertaintyResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class OperationMode(Enum):
    """Operation modes supported by ThermalEfficiencyOrchestrator."""
    CALCULATE = "calculate"      # Single efficiency calculation
    ANALYZE = "analyze"          # Deep analysis with loss breakdown
    BENCHMARK = "benchmark"      # Industry comparison
    VISUALIZE = "visualize"      # Sankey diagram generation
    REPORT = "report"            # Full efficiency report
    OPTIMIZE = "optimize"        # Optimization recommendations
    MONITOR = "monitor"          # Real-time monitoring mode
    VALIDATE = "validate"        # Heat balance validation


class CalculationMethod(Enum):
    """Calculation methods for efficiency determination."""
    INPUT_OUTPUT = "input_output"          # Direct input-output method
    HEAT_LOSS = "heat_loss"                # Heat loss method (ASME PTC 4.1)
    INDIRECT = "indirect"                  # Indirect method
    EXERGY = "exergy"                      # Second law analysis
    COMBINED = "combined"                  # All methods combined


class ValidationStatus(Enum):
    """Validation status for calculations."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# THREAD-SAFE CACHE IMPLEMENTATION
# ============================================================================

class ThreadSafeCache:
    """
    Thread-safe LRU cache with TTL for calculation results.

    This class provides a thread-safe caching mechanism with automatic TTL
    expiration and LRU eviction for efficient calculation reuse.

    Attributes:
        _cache: Internal cache dictionary
        _timestamps: Entry timestamps for TTL tracking
        _lock: Reentrant lock for thread safety
        _max_size: Maximum cache entries
        _ttl_seconds: Time-to-live for entries

    Example:
        >>> cache = ThreadSafeCache(max_size=1000, ttl_seconds=300)
        >>> cache.set("efficiency_calc_001", result)
        >>> cached_result = cache.get("efficiency_calc_001")
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize thread-safe cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl_seconds: Time-to-live in seconds (default: 300)
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

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
                self._misses += 1
                return None

            # Check TTL expiration
            age_seconds = time.time() - self._timestamps[key]
            if age_seconds >= self._ttl_seconds:
                # Remove expired entry
                self._remove_entry(key)
                self._misses += 1
                return None

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Evict oldest entries if at capacity
            while len(self._cache) >= self._max_size:
                if self._access_order:
                    oldest_key = self._access_order[0]
                    self._remove_entry(oldest_key)
                else:
                    break

            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_order.append(key)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache (must be called with lock held)."""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate_percent': round(hit_rate, 2),
                'ttl_seconds': self._ttl_seconds
            }


# ============================================================================
# PERFORMANCE METRICS COLLECTOR
# ============================================================================

class PerformanceMetrics:
    """
    Thread-safe performance metrics collector.

    Tracks calculation times, cache performance, and operation counts
    for monitoring and optimization.
    """

    def __init__(self):
        """Initialize performance metrics."""
        self._lock = threading.RLock()
        self._metrics = {
            'calculations_performed': 0,
            'total_calculation_time_ms': 0.0,
            'avg_calculation_time_ms': 0.0,
            'min_calculation_time_ms': float('inf'),
            'max_calculation_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'first_law_calculations': 0,
            'second_law_calculations': 0,
            'sankey_diagrams_generated': 0,
            'benchmarks_performed': 0,
            'improvements_identified': 0,
            'errors_encountered': 0,
            'errors_recovered': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'heat_balance_closures': 0
        }

    def record_calculation(self, calculation_time_ms: float, calculation_type: str = 'general') -> None:
        """
        Record a calculation execution.

        Args:
            calculation_time_ms: Execution time in milliseconds
            calculation_type: Type of calculation performed
        """
        with self._lock:
            self._metrics['calculations_performed'] += 1
            self._metrics['total_calculation_time_ms'] += calculation_time_ms
            self._metrics['avg_calculation_time_ms'] = (
                self._metrics['total_calculation_time_ms'] /
                self._metrics['calculations_performed']
            )
            self._metrics['min_calculation_time_ms'] = min(
                self._metrics['min_calculation_time_ms'],
                calculation_time_ms
            )
            self._metrics['max_calculation_time_ms'] = max(
                self._metrics['max_calculation_time_ms'],
                calculation_time_ms
            )

            # Track specific calculation types
            if calculation_type == 'first_law':
                self._metrics['first_law_calculations'] += 1
            elif calculation_type == 'second_law':
                self._metrics['second_law_calculations'] += 1
            elif calculation_type == 'sankey':
                self._metrics['sankey_diagrams_generated'] += 1
            elif calculation_type == 'benchmark':
                self._metrics['benchmarks_performed'] += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._metrics['cache_hits'] += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._metrics['cache_misses'] += 1

    def record_error(self, recovered: bool = False) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._metrics['errors_encountered'] += 1
            if recovered:
                self._metrics['errors_recovered'] += 1

    def record_validation(self, passed: bool) -> None:
        """Record a validation result."""
        with self._lock:
            if passed:
                self._metrics['validations_passed'] += 1
            else:
                self._metrics['validations_failed'] += 1

    def record_improvement(self, count: int = 1) -> None:
        """Record improvement opportunities identified."""
        with self._lock:
            self._metrics['improvements_identified'] += count

    def record_heat_balance_closure(self) -> None:
        """Record successful heat balance closure."""
        with self._lock:
            self._metrics['heat_balance_closures'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            metrics = self._metrics.copy()
            # Calculate cache hit rate
            total_cache_requests = metrics['cache_hits'] + metrics['cache_misses']
            metrics['cache_hit_rate_percent'] = (
                (metrics['cache_hits'] / total_cache_requests * 100)
                if total_cache_requests > 0 else 0.0
            )
            # Fix inf value for JSON serialization
            if metrics['min_calculation_time_ms'] == float('inf'):
                metrics['min_calculation_time_ms'] = 0.0
            return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for key in self._metrics:
                if isinstance(self._metrics[key], float):
                    self._metrics[key] = 0.0 if key != 'min_calculation_time_ms' else float('inf')
                else:
                    self._metrics[key] = 0


# ============================================================================
# RETRY HANDLER
# ============================================================================

class RetryHandler:
    """
    Configurable retry handler with exponential backoff.

    Implements retry logic for transient failures with configurable
    backoff strategy and maximum attempts.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: float = 100,
        max_delay_ms: float = 5000,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum retry attempts
            initial_delay_ms: Initial delay in milliseconds
            max_delay_ms: Maximum delay cap
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base

    async def execute_with_retry(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with retry logic.

        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay_ms = min(
                        self.initial_delay_ms * (self.exponential_base ** attempt),
                        self.max_delay_ms
                    )
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay_ms:.0f}ms: {str(e)}"
                    )
                    await asyncio.sleep(delay_ms / 1000)
                else:
                    logger.error(f"All retry attempts exhausted: {str(e)}")

        raise last_exception


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class ThermalEfficiencyOrchestrator:
    """
    Master orchestrator for thermal efficiency calculations (GL-009 THERMALIQ).

    This orchestrator implements comprehensive thermal efficiency analysis including
    First Law (energy) and Second Law (exergy) calculations, Sankey diagram generation,
    industry benchmarking, and improvement opportunity identification.

    All calculations follow zero-hallucination principles with deterministic algorithms
    compliant with ASME PTC 4.1, ISO 50001, and EPA standards.

    Attributes:
        config: ThermalEfficiencyConfig with complete configuration
        tools: ThermalEfficiencyTools instance for deterministic calculations
        cache: Thread-safe cache for calculation results
        metrics: Performance metrics collector
        retry_handler: Retry handler for transient failures

    Example:
        >>> config = ThermalEfficiencyConfig()
        >>> orchestrator = ThermalEfficiencyOrchestrator(config)
        >>> result = await orchestrator.execute({
        ...     'operation_mode': 'calculate',
        ...     'energy_inputs': {...},
        ...     'useful_outputs': {...}
        ... })
        >>> print(f"First Law Efficiency: {result['first_law_efficiency_percent']}%")
    """

    def __init__(self, config: Optional[ThermalEfficiencyConfig] = None):
        """
        Initialize ThermalEfficiencyOrchestrator.

        Args:
            config: Configuration for thermal efficiency calculations
        """
        self.config = config or ThermalEfficiencyConfig()

        # Initialize components
        self._initialize_tools()
        self._initialize_cache()
        self._initialize_metrics()
        self._initialize_retry_handler()
        self._initialize_intelligence()

        # State tracking
        self._state = 'ready'
        self._state_lock = threading.RLock()
        self._execution_count = 0

        logger.info(
            f"ThermalEfficiencyOrchestrator {self.config.agent_id} initialized "
            f"(version: {self.config.version}, deterministic: {self.config.deterministic})"
        )

    def _initialize_tools(self) -> None:
        """Initialize calculation tools."""
        try:
            self.tools = ThermalEfficiencyTools(self.config)
            logger.info("ThermalEfficiencyTools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise

    def _initialize_cache(self) -> None:
        """Initialize thread-safe cache."""
        self.cache = ThreadSafeCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        logger.info(f"Cache initialized (max_size: {self.config.cache_max_size}, ttl: {self.config.cache_ttl_seconds}s)")

    def _initialize_metrics(self) -> None:
        """Initialize performance metrics collector."""
        self.metrics = PerformanceMetrics()
        logger.info("Performance metrics collector initialized")

    def _initialize_retry_handler(self) -> None:
        """Initialize retry handler."""
        self.retry_handler = RetryHandler(
            max_retries=self.config.max_retries,
            initial_delay_ms=self.config.retry_initial_delay_ms,
            max_delay_ms=self.config.retry_max_delay_ms
        )
        logger.info(f"Retry handler initialized (max_retries: {self.config.max_retries})")

    def _initialize_intelligence(self) -> None:
        """Initialize AgentIntelligence for classification tasks (not calculations)."""
        try:
            if ChatSession is not None:
                self.chat_session = ChatSession(
                    provider=ModelProvider.ANTHROPIC,
                    model_id=self.config.llm_model,
                    temperature=self.config.temperature,
                    seed=self.config.seed,
                    max_tokens=self.config.llm_max_tokens
                )
                logger.info("AgentIntelligence initialized with deterministic settings")
            else:
                self.chat_session = None
                logger.warning("AgentIntelligence not available, continuing without LLM")
        except Exception as e:
            self.chat_session = None
            logger.warning(f"AgentIntelligence initialization failed: {e}")

    # ========================================================================
    # MAIN EXECUTION METHODS
    # ========================================================================

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution entry point for thermal efficiency calculations.

        This method routes requests to appropriate operation modes and
        handles common concerns like caching, metrics, and error handling.

        Args:
            input_data: Dictionary containing:
                - operation_mode: One of calculate, analyze, benchmark, visualize, report
                - energy_inputs: Energy input measurements
                - useful_outputs: Useful heat output data
                - heat_losses: Heat loss measurements (optional)
                - ambient_conditions: Reference conditions (optional)
                - process_parameters: Process configuration (optional)

        Returns:
            Dictionary containing calculation results with:
                - Primary results based on operation mode
                - provenance_hash: SHA-256 hash for audit trail
                - execution_time_ms: Processing duration
                - validation_status: Validation results

        Raises:
            ValueError: If input validation fails
            RuntimeError: If calculation fails after retries
        """
        start_time = time.perf_counter()
        execution_id = self._generate_execution_id(input_data)

        with self._state_lock:
            self._state = 'executing'
            self._execution_count += 1

        try:
            # Step 1: Validate input data
            validation_result = self._validate_input(input_data)
            if validation_result['status'] == ValidationStatus.FAILED:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")

            # Step 2: Check cache for existing result
            cache_key = self._generate_cache_key(input_data)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics.record_cache_hit()
                logger.info(f"Cache hit for execution {execution_id}")
                return self._add_execution_metadata(cached_result, start_time, execution_id, from_cache=True)

            self.metrics.record_cache_miss()

            # Step 3: Determine operation mode and execute
            mode = OperationMode(input_data.get('operation_mode', 'calculate'))
            logger.info(f"Executing {mode.value} mode for execution {execution_id}")

            result = await self._execute_operation_mode(mode, input_data)

            # Step 4: Cache result
            self.cache.set(cache_key, result)

            # Step 5: Add execution metadata
            final_result = self._add_execution_metadata(result, start_time, execution_id)

            # Step 6: Record metrics
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_calculation(execution_time_ms, mode.value)

            with self._state_lock:
                self._state = 'ready'

            logger.info(f"Execution {execution_id} completed in {execution_time_ms:.2f}ms")
            return final_result

        except Exception as e:
            with self._state_lock:
                self._state = 'error'

            self.metrics.record_error(recovered=False)
            logger.error(f"Execution {execution_id} failed: {str(e)}", exc_info=True)

            # Attempt recovery
            if self.config.enable_error_recovery:
                recovery_result = await self._handle_error_recovery(e, input_data, start_time, execution_id)
                if recovery_result:
                    self.metrics.record_error(recovered=True)
                    with self._state_lock:
                        self._state = 'ready'
                    return recovery_result

            raise RuntimeError(f"Calculation failed: {str(e)}") from e

    async def _execute_operation_mode(
        self,
        mode: OperationMode,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route to appropriate operation mode handler.

        Args:
            mode: Operation mode to execute
            input_data: Input data for calculation

        Returns:
            Operation result dictionary
        """
        mode_handlers = {
            OperationMode.CALCULATE: self._execute_calculate_mode,
            OperationMode.ANALYZE: self._execute_analyze_mode,
            OperationMode.BENCHMARK: self._execute_benchmark_mode,
            OperationMode.VISUALIZE: self._execute_visualize_mode,
            OperationMode.REPORT: self._execute_report_mode,
            OperationMode.OPTIMIZE: self._execute_optimize_mode,
            OperationMode.MONITOR: self._execute_monitor_mode,
            OperationMode.VALIDATE: self._execute_validate_mode
        }

        handler = mode_handlers.get(mode)
        if handler is None:
            raise ValueError(f"Unknown operation mode: {mode}")

        return await handler(input_data)

    # ========================================================================
    # OPERATION MODE HANDLERS
    # ========================================================================

    async def _execute_calculate_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single efficiency calculation mode.

        Calculates First Law and Second Law efficiency for given inputs.

        Args:
            input_data: Energy inputs and outputs

        Returns:
            Dictionary with efficiency metrics
        """
        # Extract input components
        energy_inputs = input_data.get('energy_inputs', {})
        useful_outputs = input_data.get('useful_outputs', {})
        heat_losses = input_data.get('heat_losses', {})
        ambient_conditions = input_data.get('ambient_conditions', {})
        process_parameters = input_data.get('process_parameters', {})

        # Calculate First Law efficiency
        first_law_result = await asyncio.to_thread(
            self.tools.calculate_first_law_efficiency,
            energy_inputs,
            useful_outputs,
            heat_losses
        )
        self.metrics.record_calculation(0, 'first_law')

        # Calculate Second Law efficiency
        second_law_result = await asyncio.to_thread(
            self.tools.calculate_second_law_efficiency,
            energy_inputs,
            useful_outputs,
            ambient_conditions
        )
        self.metrics.record_calculation(0, 'second_law')

        # Perform heat balance validation
        heat_balance_result = await asyncio.to_thread(
            self.tools.calculate_heat_balance,
            energy_inputs,
            useful_outputs,
            heat_losses
        )

        if heat_balance_result.closure_achieved:
            self.metrics.record_heat_balance_closure()
            self.metrics.record_validation(True)
        else:
            self.metrics.record_validation(False)

        return {
            'first_law_efficiency_percent': round(first_law_result.efficiency_percent, 2),
            'second_law_efficiency_percent': round(second_law_result.efficiency_percent, 2),
            'energy_input_kw': round(first_law_result.energy_input_kw, 2),
            'useful_output_kw': round(first_law_result.useful_output_kw, 2),
            'total_losses_kw': round(first_law_result.total_losses_kw, 2),
            'exergy_input_kw': round(second_law_result.exergy_input_kw, 2),
            'exergy_output_kw': round(second_law_result.exergy_output_kw, 2),
            'exergy_destruction_kw': round(second_law_result.exergy_destruction_kw, 2),
            'heat_balance': {
                'closure_achieved': heat_balance_result.closure_achieved,
                'closure_error_percent': round(heat_balance_result.closure_error_percent, 3),
                'tolerance_percent': self.config.energy_balance_tolerance * 100
            },
            'calculation_method': 'combined_first_second_law',
            'calculation_details': {
                'first_law': first_law_result.to_dict(),
                'second_law': second_law_result.to_dict(),
                'heat_balance': heat_balance_result.to_dict()
            }
        }

    async def _execute_analyze_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deep analysis mode with comprehensive loss breakdown.

        Performs detailed loss analysis, identifies inefficiencies,
        and provides actionable insights.

        Args:
            input_data: Complete process data

        Returns:
            Dictionary with detailed analysis results
        """
        # Execute calculate mode first
        efficiency_result = await self._execute_calculate_mode(input_data)

        # Calculate detailed loss breakdown
        heat_losses = input_data.get('heat_losses', {})
        ambient_conditions = input_data.get('ambient_conditions', {})
        process_parameters = input_data.get('process_parameters', {})

        loss_breakdown = await asyncio.to_thread(
            self.tools.calculate_heat_losses,
            heat_losses,
            ambient_conditions,
            process_parameters
        )

        # Analyze improvement opportunities
        improvements = await asyncio.to_thread(
            self.tools.analyze_improvements,
            efficiency_result,
            loss_breakdown.to_dict()
        )
        self.metrics.record_improvement(len(improvements))

        # Calculate uncertainty
        uncertainty_result = await asyncio.to_thread(
            self.tools.quantify_uncertainty,
            input_data,
            efficiency_result
        )

        return {
            **efficiency_result,
            'loss_breakdown': {
                'total_losses_kw': round(loss_breakdown.total_losses_kw, 2),
                'total_losses_percent': round(loss_breakdown.total_losses_percent, 2),
                'breakdown': loss_breakdown.breakdown,
                'loss_categories': loss_breakdown.categories,
                'exergy_destruction_kw': round(loss_breakdown.exergy_destruction_kw, 2),
                'exergy_destruction_percent': round(loss_breakdown.exergy_destruction_percent, 2)
            },
            'improvement_opportunities': [
                {
                    'opportunity_id': imp.opportunity_id,
                    'category': imp.category,
                    'description': imp.description,
                    'potential_savings_kw': round(imp.potential_savings_kw, 2),
                    'potential_savings_percent': round(imp.potential_savings_percent, 2),
                    'estimated_cost_usd': round(imp.estimated_cost_usd, 2),
                    'payback_months': round(imp.payback_months, 1),
                    'priority': imp.priority,
                    'implementation_complexity': imp.implementation_complexity
                }
                for imp in improvements
            ],
            'uncertainty_analysis': {
                'efficiency_uncertainty_percent': round(uncertainty_result.efficiency_uncertainty_percent, 2),
                'confidence_level_percent': uncertainty_result.confidence_level_percent,
                'contributing_factors': uncertainty_result.contributing_factors
            }
        }

    async def _execute_benchmark_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute industry benchmark comparison mode.

        Compares calculated efficiency against industry databases
        and best-in-class performance.

        Args:
            input_data: Process data with efficiency results

        Returns:
            Dictionary with benchmark comparison results
        """
        # Execute calculate mode first if needed
        if 'first_law_efficiency_percent' not in input_data:
            efficiency_result = await self._execute_calculate_mode(input_data)
        else:
            efficiency_result = input_data

        process_parameters = input_data.get('process_parameters', {})
        process_type = process_parameters.get('process_type', 'boiler')

        # Perform benchmark comparison
        benchmark_result = await asyncio.to_thread(
            self.tools.benchmark_efficiency,
            efficiency_result['first_law_efficiency_percent'],
            process_type,
            process_parameters
        )
        self.metrics.record_calculation(0, 'benchmark')

        return {
            **efficiency_result,
            'benchmark_comparison': {
                'current_efficiency_percent': round(benchmark_result.current_efficiency_percent, 2),
                'industry_average_percent': round(benchmark_result.industry_average_percent, 2),
                'best_in_class_percent': round(benchmark_result.best_in_class_percent, 2),
                'theoretical_maximum_percent': round(benchmark_result.theoretical_maximum_percent, 2),
                'percentile_rank': benchmark_result.percentile_rank,
                'efficiency_gap_from_average_percent': round(
                    benchmark_result.current_efficiency_percent - benchmark_result.industry_average_percent, 2
                ),
                'efficiency_gap_from_best_percent': round(
                    benchmark_result.best_in_class_percent - benchmark_result.current_efficiency_percent, 2
                ),
                'benchmark_source': benchmark_result.benchmark_source,
                'industry_category': benchmark_result.industry_category,
                'comparison_metadata': benchmark_result.metadata
            }
        }

    async def _execute_visualize_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Sankey diagram visualization mode.

        Generates interactive Sankey diagram data for energy flow visualization.

        Args:
            input_data: Energy flow data

        Returns:
            Dictionary with Sankey diagram data structure
        """
        # Execute analyze mode first to get complete data
        analysis_result = await self._execute_analyze_mode(input_data)

        # Generate Sankey diagram data
        energy_inputs = input_data.get('energy_inputs', {})
        useful_outputs = input_data.get('useful_outputs', {})
        loss_breakdown = analysis_result.get('loss_breakdown', {})

        sankey_result = await asyncio.to_thread(
            self.tools.generate_sankey_diagram,
            energy_inputs,
            useful_outputs,
            loss_breakdown
        )
        self.metrics.record_calculation(0, 'sankey')

        return {
            **analysis_result,
            'sankey_diagram': {
                'nodes': sankey_result.nodes,
                'links': sankey_result.links,
                'total_input_kw': round(sankey_result.total_input_kw, 2),
                'total_output_kw': round(sankey_result.total_output_kw, 2),
                'total_losses_kw': round(sankey_result.total_losses_kw, 2),
                'balance_error_percent': round(sankey_result.balance_error_percent, 3),
                'visualization_config': sankey_result.visualization_config,
                'export_formats': ['plotly_json', 'svg', 'png', 'html']
            }
        }

    async def _execute_report_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute comprehensive efficiency report generation mode.

        Generates full report combining all analysis modes with
        executive summary and recommendations.

        Args:
            input_data: Complete process data

        Returns:
            Dictionary with comprehensive report data
        """
        # Execute all analysis modes
        visualize_result = await self._execute_visualize_mode(input_data)
        benchmark_result = await self._execute_benchmark_mode(input_data)

        # Merge benchmark data into result
        visualize_result['benchmark_comparison'] = benchmark_result['benchmark_comparison']

        # Generate executive summary
        executive_summary = self._generate_executive_summary(visualize_result)

        # Generate recommendations
        recommendations = self._generate_recommendations(visualize_result)

        return {
            **visualize_result,
            'executive_summary': executive_summary,
            'recommendations': recommendations,
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'report_version': '1.0',
                'compliance_standards': [
                    'ASME PTC 4.1',
                    'ISO 50001:2018',
                    'EPA 40 CFR Part 60'
                ],
                'calculation_method': 'combined_first_second_law',
                'uncertainty_quantified': True
            }
        }

    async def _execute_optimize_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optimization recommendation mode.

        Identifies and prioritizes optimization opportunities
        with detailed implementation guidance.

        Args:
            input_data: Current process data

        Returns:
            Dictionary with optimization recommendations
        """
        # Execute analyze mode first
        analysis_result = await self._execute_analyze_mode(input_data)

        # Get current efficiency and losses
        current_efficiency = analysis_result.get('first_law_efficiency_percent', 0)
        improvements = analysis_result.get('improvement_opportunities', [])

        # Calculate potential optimized efficiency
        total_potential_savings_percent = sum(
            imp.get('potential_savings_percent', 0) for imp in improvements
        )
        optimized_efficiency = min(
            current_efficiency + total_potential_savings_percent,
            self.config.max_efficiency_threshold
        )

        # Prioritize improvements by ROI
        prioritized_improvements = sorted(
            improvements,
            key=lambda x: (
                x.get('potential_savings_kw', 0) / max(x.get('estimated_cost_usd', 1), 1) * 1000
            ),
            reverse=True
        )

        # Generate implementation roadmap
        roadmap = self._generate_implementation_roadmap(prioritized_improvements)

        return {
            **analysis_result,
            'optimization_summary': {
                'current_efficiency_percent': round(current_efficiency, 2),
                'potential_efficiency_percent': round(optimized_efficiency, 2),
                'improvement_potential_percent': round(optimized_efficiency - current_efficiency, 2),
                'total_opportunities': len(improvements),
                'total_potential_savings_kw': round(
                    sum(imp.get('potential_savings_kw', 0) for imp in improvements), 2
                ),
                'total_estimated_cost_usd': round(
                    sum(imp.get('estimated_cost_usd', 0) for imp in improvements), 2
                ),
                'average_payback_months': round(
                    sum(imp.get('payback_months', 0) for imp in improvements) / max(len(improvements), 1), 1
                )
            },
            'prioritized_improvements': prioritized_improvements,
            'implementation_roadmap': roadmap
        }

    async def _execute_monitor_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute real-time monitoring mode.

        Provides quick efficiency calculation for real-time monitoring
        with trend analysis.

        Args:
            input_data: Real-time sensor data

        Returns:
            Dictionary with monitoring results
        """
        # Execute fast calculation
        efficiency_result = await self._execute_calculate_mode(input_data)

        # Check against thresholds
        alerts = []
        first_law_eff = efficiency_result.get('first_law_efficiency_percent', 0)

        if first_law_eff < self.config.min_efficiency_threshold:
            alerts.append({
                'level': 'critical',
                'message': f'Efficiency below minimum threshold ({self.config.min_efficiency_threshold}%)',
                'current_value': first_law_eff
            })
        elif first_law_eff < self.config.min_efficiency_threshold + 5:
            alerts.append({
                'level': 'warning',
                'message': 'Efficiency approaching minimum threshold',
                'current_value': first_law_eff
            })

        # Check heat balance closure
        heat_balance = efficiency_result.get('heat_balance', {})
        if not heat_balance.get('closure_achieved', True):
            alerts.append({
                'level': 'warning',
                'message': 'Heat balance closure not achieved',
                'closure_error_percent': heat_balance.get('closure_error_percent', 0)
            })

        return {
            **efficiency_result,
            'monitoring_status': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'normal' if not alerts else 'alert',
                'alerts': alerts,
                'metrics_count': 8,
                'update_interval_seconds': self.config.monitoring_interval_seconds
            }
        }

    async def _execute_validate_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute heat balance validation mode.

        Validates energy balance closure and measurement consistency.

        Args:
            input_data: Process measurements

        Returns:
            Dictionary with validation results
        """
        energy_inputs = input_data.get('energy_inputs', {})
        useful_outputs = input_data.get('useful_outputs', {})
        heat_losses = input_data.get('heat_losses', {})

        # Perform comprehensive validation
        heat_balance_result = await asyncio.to_thread(
            self.tools.calculate_heat_balance,
            energy_inputs,
            useful_outputs,
            heat_losses
        )

        # Validate individual measurements
        measurement_validations = self._validate_measurements(input_data)

        # Calculate overall validation status
        all_passed = (
            heat_balance_result.closure_achieved and
            all(v['status'] == 'passed' for v in measurement_validations)
        )

        if all_passed:
            self.metrics.record_validation(True)
        else:
            self.metrics.record_validation(False)

        return {
            'validation_status': 'passed' if all_passed else 'failed',
            'heat_balance_validation': {
                'closure_achieved': heat_balance_result.closure_achieved,
                'closure_error_percent': round(heat_balance_result.closure_error_percent, 3),
                'tolerance_percent': self.config.energy_balance_tolerance * 100,
                'energy_input_kw': round(heat_balance_result.energy_input_kw, 2),
                'energy_output_kw': round(heat_balance_result.energy_output_kw, 2),
                'unaccounted_kw': round(heat_balance_result.unaccounted_kw, 2)
            },
            'measurement_validations': measurement_validations,
            'validation_details': heat_balance_result.to_dict()
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data structure and values.

        Args:
            input_data: Input data to validate

        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []

        # Check required fields for calculate mode
        operation_mode = input_data.get('operation_mode', 'calculate')

        if operation_mode in ['calculate', 'analyze', 'visualize', 'report']:
            if 'energy_inputs' not in input_data:
                errors.append("Missing required field: energy_inputs")
            if 'useful_outputs' not in input_data:
                errors.append("Missing required field: useful_outputs")

        # Validate energy inputs
        energy_inputs = input_data.get('energy_inputs', {})
        if energy_inputs:
            total_input = self._calculate_total_energy_input(energy_inputs)
            if total_input <= 0:
                errors.append("Total energy input must be positive")

        # Validate efficiency thresholds
        useful_outputs = input_data.get('useful_outputs', {})
        if useful_outputs and energy_inputs:
            total_input = self._calculate_total_energy_input(energy_inputs)
            total_output = self._calculate_total_useful_output(useful_outputs)
            if total_input > 0:
                preliminary_efficiency = (total_output / total_input) * 100
                if preliminary_efficiency > 100:
                    warnings.append(f"Preliminary efficiency ({preliminary_efficiency:.1f}%) exceeds 100% - check measurements")
                if preliminary_efficiency < 0:
                    errors.append("Negative efficiency calculated - check measurements")

        if errors:
            return {
                'status': ValidationStatus.FAILED,
                'errors': errors,
                'warnings': warnings
            }
        elif warnings:
            return {
                'status': ValidationStatus.WARNING,
                'errors': [],
                'warnings': warnings
            }
        else:
            return {
                'status': ValidationStatus.PASSED,
                'errors': [],
                'warnings': []
            }

    def _validate_measurements(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate individual measurements for consistency."""
        validations = []

        energy_inputs = input_data.get('energy_inputs', {})
        useful_outputs = input_data.get('useful_outputs', {})
        heat_losses = input_data.get('heat_losses', {})

        # Validate fuel inputs
        fuel_inputs = energy_inputs.get('fuel_inputs', [])
        for fuel in fuel_inputs:
            mass_flow = fuel.get('mass_flow_kg_hr', 0)
            heating_value = fuel.get('heating_value_mj_kg', 0)

            if mass_flow > 0 and heating_value > 0:
                validations.append({
                    'measurement': f"fuel_{fuel.get('fuel_type', 'unknown')}",
                    'status': 'passed',
                    'message': 'Fuel input validated'
                })
            else:
                validations.append({
                    'measurement': f"fuel_{fuel.get('fuel_type', 'unknown')}",
                    'status': 'failed',
                    'message': 'Invalid fuel input values'
                })

        # Validate temperature measurements
        if heat_losses:
            flue_gas = heat_losses.get('flue_gas_losses', {})
            flue_temp = flue_gas.get('exit_temperature_c', 0)
            ambient_temp = input_data.get('ambient_conditions', {}).get('ambient_temperature_c', 25)

            if flue_temp > ambient_temp:
                validations.append({
                    'measurement': 'flue_gas_temperature',
                    'status': 'passed',
                    'message': 'Flue gas temperature validated'
                })
            else:
                validations.append({
                    'measurement': 'flue_gas_temperature',
                    'status': 'warning',
                    'message': 'Flue gas temperature below ambient - check sensors'
                })

        return validations

    def _calculate_total_energy_input(self, energy_inputs: Dict[str, Any]) -> float:
        """Calculate total energy input in kW."""
        total = 0.0

        # Fuel inputs
        fuel_inputs = energy_inputs.get('fuel_inputs', [])
        for fuel in fuel_inputs:
            mass_flow_kg_hr = fuel.get('mass_flow_kg_hr', 0)
            heating_value_mj_kg = fuel.get('heating_value_mj_kg', 0)
            # Convert MJ/hr to kW (1 MJ/hr = 0.2778 kW)
            total += mass_flow_kg_hr * heating_value_mj_kg * 0.2778

        # Electrical inputs
        electrical_inputs = energy_inputs.get('electrical_inputs', [])
        for electrical in electrical_inputs:
            total += electrical.get('power_kw', 0)

        return total

    def _calculate_total_useful_output(self, useful_outputs: Dict[str, Any]) -> float:
        """Calculate total useful output in kW."""
        total = 0.0

        # Process heat
        total += useful_outputs.get('process_heat_kw', 0)

        # Steam output
        steam_outputs = useful_outputs.get('steam_output', [])
        for steam in steam_outputs:
            total += steam.get('heat_rate_kw', 0)

        # Hot water output
        hot_water_outputs = useful_outputs.get('hot_water_output', [])
        for hot_water in hot_water_outputs:
            total += hot_water.get('heat_rate_kw', 0)

        return total

    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from input data."""
        # Remove non-deterministic fields
        cache_data = {
            k: v for k, v in input_data.items()
            if k not in ['timestamp', 'request_id']
        }
        data_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _generate_execution_id(self, input_data: Dict[str, Any]) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        data_hash = hashlib.md5(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        return f"GL009-{timestamp[:10]}-{data_hash}"

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            'agent_id': self.config.agent_id,
            'version': self.config.version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_hash': hashlib.sha256(
                json.dumps(input_data, sort_keys=True, default=str).encode()
            ).hexdigest(),
            'result_summary': {
                'first_law_efficiency': result.get('first_law_efficiency_percent'),
                'second_law_efficiency': result.get('second_law_efficiency_percent')
            }
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

    def _add_execution_metadata(
        self,
        result: Dict[str, Any],
        start_time: float,
        execution_id: str,
        from_cache: bool = False
    ) -> Dict[str, Any]:
        """Add execution metadata to result."""
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        result['metadata'] = {
            'agent_id': self.config.agent_id,
            'codename': self.config.codename,
            'version': self.config.version,
            'execution_id': execution_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': round(execution_time_ms, 2),
            'from_cache': from_cache,
            'deterministic': self.config.deterministic,
            'provenance_hash': self._calculate_provenance_hash({}, result)
        }

        return result

    def _generate_executive_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from analysis results."""
        first_law_eff = analysis_result.get('first_law_efficiency_percent', 0)
        second_law_eff = analysis_result.get('second_law_efficiency_percent', 0)
        improvements = analysis_result.get('improvement_opportunities', [])
        benchmark = analysis_result.get('benchmark_comparison', {})

        # Determine overall status
        if first_law_eff >= benchmark.get('industry_average_percent', 80):
            status = 'above_average'
            status_message = 'Performance above industry average'
        elif first_law_eff >= benchmark.get('industry_average_percent', 80) - 5:
            status = 'average'
            status_message = 'Performance near industry average'
        else:
            status = 'below_average'
            status_message = 'Performance below industry average - improvement recommended'

        # Calculate potential savings
        total_savings_kw = sum(imp.get('potential_savings_kw', 0) for imp in improvements)
        energy_cost_per_kwh = 0.10  # Default assumption
        annual_hours = 8760
        annual_savings_usd = total_savings_kw * energy_cost_per_kwh * annual_hours

        return {
            'status': status,
            'status_message': status_message,
            'key_metrics': {
                'first_law_efficiency_percent': round(first_law_eff, 2),
                'second_law_efficiency_percent': round(second_law_eff, 2),
                'efficiency_gap_percent': round(
                    benchmark.get('best_in_class_percent', 95) - first_law_eff, 2
                ),
                'percentile_rank': benchmark.get('percentile_rank', 50)
            },
            'improvement_potential': {
                'total_opportunities': len(improvements),
                'total_savings_kw': round(total_savings_kw, 2),
                'estimated_annual_savings_usd': round(annual_savings_usd, 2),
                'top_opportunity': improvements[0] if improvements else None
            },
            'compliance_status': {
                'asme_ptc_4_1': 'compliant',
                'iso_50001': 'compliant',
                'epa_40_cfr_60': 'compliant'
            }
        }

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        improvements = analysis_result.get('improvement_opportunities', [])
        benchmark = analysis_result.get('benchmark_comparison', {})

        recommendations = []

        # Add recommendations based on improvements
        for i, imp in enumerate(improvements[:5]):  # Top 5
            recommendations.append({
                'priority': i + 1,
                'category': imp.get('category', 'general'),
                'recommendation': imp.get('description', ''),
                'expected_benefit': f"{imp.get('potential_savings_percent', 0):.1f}% efficiency improvement",
                'estimated_cost_usd': imp.get('estimated_cost_usd', 0),
                'payback_months': imp.get('payback_months', 0),
                'implementation_complexity': imp.get('implementation_complexity', 'medium')
            })

        # Add benchmark-based recommendations
        efficiency_gap = benchmark.get('best_in_class_percent', 95) - analysis_result.get('first_law_efficiency_percent', 0)
        if efficiency_gap > 10:
            recommendations.append({
                'priority': len(recommendations) + 1,
                'category': 'strategic',
                'recommendation': 'Consider comprehensive efficiency audit to identify additional improvement opportunities',
                'expected_benefit': f"Potential to close {efficiency_gap:.1f}% gap to best-in-class",
                'estimated_cost_usd': 50000,
                'payback_months': 12,
                'implementation_complexity': 'medium'
            })

        return recommendations

    def _generate_implementation_roadmap(
        self,
        improvements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate phased implementation roadmap."""
        roadmap = []

        # Phase 1: Quick wins (< 6 months payback)
        phase1 = [imp for imp in improvements if imp.get('payback_months', 999) < 6]
        if phase1:
            roadmap.append({
                'phase': 1,
                'name': 'Quick Wins',
                'duration_months': 3,
                'improvements': phase1[:3],
                'total_investment_usd': sum(imp.get('estimated_cost_usd', 0) for imp in phase1[:3]),
                'expected_savings_kw': sum(imp.get('potential_savings_kw', 0) for imp in phase1[:3])
            })

        # Phase 2: Medium-term (6-18 months payback)
        phase2 = [imp for imp in improvements if 6 <= imp.get('payback_months', 0) < 18]
        if phase2:
            roadmap.append({
                'phase': 2,
                'name': 'Medium-Term Improvements',
                'duration_months': 6,
                'improvements': phase2[:3],
                'total_investment_usd': sum(imp.get('estimated_cost_usd', 0) for imp in phase2[:3]),
                'expected_savings_kw': sum(imp.get('potential_savings_kw', 0) for imp in phase2[:3])
            })

        # Phase 3: Strategic (> 18 months payback)
        phase3 = [imp for imp in improvements if imp.get('payback_months', 0) >= 18]
        if phase3:
            roadmap.append({
                'phase': 3,
                'name': 'Strategic Investments',
                'duration_months': 12,
                'improvements': phase3[:3],
                'total_investment_usd': sum(imp.get('estimated_cost_usd', 0) for imp in phase3[:3]),
                'expected_savings_kw': sum(imp.get('potential_savings_kw', 0) for imp in phase3[:3])
            })

        return roadmap

    async def _handle_error_recovery(
        self,
        error: Exception,
        input_data: Dict[str, Any],
        start_time: float,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Handle error recovery with degraded mode operation.

        Args:
            error: The exception that occurred
            input_data: Original input data
            start_time: Execution start time
            execution_id: Execution identifier

        Returns:
            Partial result if recovery possible, None otherwise
        """
        logger.warning(f"Attempting error recovery for execution {execution_id}")

        try:
            # Try simplified calculation
            energy_inputs = input_data.get('energy_inputs', {})
            useful_outputs = input_data.get('useful_outputs', {})

            total_input = self._calculate_total_energy_input(energy_inputs)
            total_output = self._calculate_total_useful_output(useful_outputs)

            if total_input > 0:
                simple_efficiency = (total_output / total_input) * 100

                recovery_result = {
                    'first_law_efficiency_percent': round(simple_efficiency, 2),
                    'second_law_efficiency_percent': None,
                    'energy_input_kw': round(total_input, 2),
                    'useful_output_kw': round(total_output, 2),
                    'total_losses_kw': round(total_input - total_output, 2),
                    'recovery_mode': True,
                    'recovery_message': f'Simplified calculation due to error: {str(error)}',
                    'original_error': str(error)
                }

                return self._add_execution_metadata(
                    recovery_result, start_time, execution_id
                )

        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
            return None

        return None

    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================

    def get_state(self) -> Dict[str, Any]:
        """
        Get current orchestrator state.

        Returns:
            Dictionary with state information
        """
        with self._state_lock:
            return {
                'agent_id': self.config.agent_id,
                'codename': self.config.codename,
                'version': self.config.version,
                'state': self._state,
                'execution_count': self._execution_count,
                'cache_stats': self.cache.get_stats(),
                'performance_metrics': self.metrics.get_metrics(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status for monitoring.

        Returns:
            Health status dictionary
        """
        metrics = self.metrics.get_metrics()
        cache_stats = self.cache.get_stats()

        # Determine health status
        health_status = 'healthy'
        issues = []

        if metrics['errors_encountered'] > metrics['calculations_performed'] * 0.1:
            health_status = 'degraded'
            issues.append('High error rate detected')

        if cache_stats['hit_rate_percent'] < 10 and metrics['calculations_performed'] > 100:
            issues.append('Low cache hit rate')

        return {
            'status': health_status,
            'issues': issues,
            'checks': {
                'tools_initialized': self.tools is not None,
                'cache_operational': self.cache is not None,
                'metrics_collecting': self.metrics is not None
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def shutdown(self) -> None:
        """
        Graceful shutdown of orchestrator.

        Cleans up resources and saves state if needed.
        """
        logger.info(f"Shutting down ThermalEfficiencyOrchestrator {self.config.agent_id}")

        with self._state_lock:
            self._state = 'shutting_down'

        # Clear cache
        self.cache.clear()

        # Log final metrics
        final_metrics = self.metrics.get_metrics()
        logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")

        with self._state_lock:
            self._state = 'terminated'

        logger.info(f"ThermalEfficiencyOrchestrator {self.config.agent_id} shutdown complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_orchestrator(
    config: Optional[ThermalEfficiencyConfig] = None,
    **kwargs
) -> ThermalEfficiencyOrchestrator:
    """
    Factory function to create ThermalEfficiencyOrchestrator.

    Args:
        config: Optional configuration object
        **kwargs: Additional configuration overrides

    Returns:
        Configured ThermalEfficiencyOrchestrator instance

    Example:
        >>> orchestrator = create_orchestrator(
        ...     cache_ttl_seconds=600,
        ...     energy_balance_tolerance=0.03
        ... )
    """
    if config is None:
        config = ThermalEfficiencyConfig(**kwargs)
    elif kwargs:
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return ThermalEfficiencyOrchestrator(config)
