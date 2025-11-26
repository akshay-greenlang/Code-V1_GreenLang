# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH - EmissionsComplianceAgent Orchestrator.

Zero-hallucination emissions compliance monitoring for industrial processes.
This module implements the GL-010 EmissionsComplianceAgent for comprehensive
emissions monitoring, regulatory compliance checking, violation detection,
and multi-jurisdiction reporting (EPA, EU IED, China MEE).

Emissions Coverage:
- NOx (Nitrogen Oxides): Thermal + Fuel + Prompt NOx
- SOx (Sulfur Oxides): Fuel sulfur-based calculations
- CO2 (Carbon Dioxide): Combustion stoichiometry
- PM (Particulate Matter): PM10/PM2.5

Standards Compliance:
- EPA 40 CFR Parts 60, 75 - Continuous Emissions Monitoring
- EU Industrial Emissions Directive 2010/75/EU
- China MEE Emission Standards (GB 13223-2011)
- ASME PTC 19.10 - Flue Gas Analysis

Example:
    >>> from emissions_compliance_orchestrator import EmissionsComplianceOrchestrator
    >>> config = EmissionsComplianceConfig(...)
    >>> orchestrator = EmissionsComplianceOrchestrator(config)
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
    EmissionsComplianceConfig,
    Jurisdiction,
    PollutantType,
    AlertSeverity,
    ReportFormat,
    CEMSConfig,
    AlertConfig,
    ReportingConfig
)
from .tools import (
    EmissionsComplianceTools,
    NOxEmissionsResult,
    SOxEmissionsResult,
    CO2EmissionsResult,
    PMEmissionsResult,
    ComplianceCheckResult,
    ViolationResult,
    RegulatoryReportResult,
    ExceedancePredictionResult,
    EmissionFactorResult,
    DispersionResult,
    AuditTrailResult,
    EMISSIONS_TOOL_SCHEMAS
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class OperationMode(Enum):
    """Operation modes supported by EmissionsComplianceOrchestrator."""
    MONITOR = "monitor"      # Real-time CEMS data monitoring
    REPORT = "report"        # Generate regulatory reports (EPA, EU, China)
    ALERT = "alert"          # Violation detection and notification
    ANALYZE = "analyze"      # Emissions trend analysis
    PREDICT = "predict"      # Exceedance prediction
    AUDIT = "audit"          # Compliance audit trail
    BENCHMARK = "benchmark"  # Compare against permit limits
    VALIDATE = "validate"    # Data quality validation


class ComplianceStatus(Enum):
    """Compliance status codes."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"
    PENDING_REVIEW = "pending_review"


class ValidationStatus(Enum):
    """Validation status for data quality."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class DataQualityCode(Enum):
    """CEMS data quality codes per EPA Part 75."""
    VALID = "valid"
    SUBSTITUTE = "substitute"
    MISSING = "missing"
    OUT_OF_CONTROL = "out_of_control"
    MAINTENANCE = "maintenance"


# ============================================================================
# THREAD-SAFE CACHE IMPLEMENTATION
# ============================================================================

class ThreadSafeCache:
    """
    Thread-safe LRU cache with TTL for emissions calculation results.

    This class provides a thread-safe caching mechanism with automatic TTL
    expiration and LRU eviction for efficient emissions data reuse.

    Attributes:
        _cache: Internal cache dictionary
        _timestamps: Entry timestamps for TTL tracking
        _lock: Reentrant lock for thread safety
        _max_size: Maximum cache entries
        _ttl_seconds: Time-to-live for entries

    Example:
        >>> cache = ThreadSafeCache(max_size=1000, ttl_seconds=300)
        >>> cache.set("nox_calc_001", result)
        >>> cached_result = cache.get("nox_calc_001")
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
    Thread-safe performance metrics collector for emissions monitoring.

    Tracks calculation times, cache performance, operation counts,
    and emissions-specific metrics for monitoring and optimization.
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
            'nox_calculations': 0,
            'sox_calculations': 0,
            'co2_calculations': 0,
            'pm_calculations': 0,
            'compliance_checks': 0,
            'violations_detected': 0,
            'reports_generated': 0,
            'alerts_sent': 0,
            'predictions_made': 0,
            'audits_performed': 0,
            'errors_encountered': 0,
            'errors_recovered': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'data_quality_issues': 0
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
            if calculation_type == 'nox':
                self._metrics['nox_calculations'] += 1
            elif calculation_type == 'sox':
                self._metrics['sox_calculations'] += 1
            elif calculation_type == 'co2':
                self._metrics['co2_calculations'] += 1
            elif calculation_type == 'pm':
                self._metrics['pm_calculations'] += 1
            elif calculation_type == 'compliance':
                self._metrics['compliance_checks'] += 1
            elif calculation_type == 'report':
                self._metrics['reports_generated'] += 1
            elif calculation_type == 'prediction':
                self._metrics['predictions_made'] += 1
            elif calculation_type == 'audit':
                self._metrics['audits_performed'] += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._metrics['cache_hits'] += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._metrics['cache_misses'] += 1

    def record_violation(self, count: int = 1) -> None:
        """Record violation detection."""
        with self._lock:
            self._metrics['violations_detected'] += count

    def record_alert(self, count: int = 1) -> None:
        """Record alert sent."""
        with self._lock:
            self._metrics['alerts_sent'] += count

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

    def record_data_quality_issue(self, count: int = 1) -> None:
        """Record data quality issues detected."""
        with self._lock:
            self._metrics['data_quality_issues'] += count

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

class EmissionsComplianceOrchestrator:
    """
    Master orchestrator for emissions compliance monitoring (GL-010 EMISSIONWATCH).

    This orchestrator implements comprehensive emissions compliance monitoring including
    NOx, SOx, CO2, and PM calculations, multi-jurisdiction regulatory compliance
    (EPA, EU IED, China MEE), violation detection, predictive analytics, and
    audit trail generation.

    All calculations follow zero-hallucination principles with deterministic algorithms
    compliant with EPA Methods 19, 2, 3A, 5, and equivalent international standards.

    Attributes:
        config: EmissionsComplianceConfig with complete configuration
        tools: EmissionsComplianceTools instance for deterministic calculations
        cache: Thread-safe cache for calculation results
        metrics: Performance metrics collector
        retry_handler: Retry handler for transient failures

    Example:
        >>> config = EmissionsComplianceConfig()
        >>> orchestrator = EmissionsComplianceOrchestrator(config)
        >>> result = await orchestrator.execute({
        ...     'operation_mode': 'monitor',
        ...     'cems_data': {...},
        ...     'fuel_data': {...}
        ... })
        >>> print(f"NOx: {result['nox_ppm']} ppm, Compliance: {result['compliance_status']}")
    """

    def __init__(self, config: Optional[EmissionsComplianceConfig] = None):
        """
        Initialize EmissionsComplianceOrchestrator.

        Args:
            config: Configuration for emissions compliance monitoring
        """
        self.config = config or EmissionsComplianceConfig()

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

        # Alert state
        self._active_alerts: List[Dict[str, Any]] = []
        self._alert_history: List[Dict[str, Any]] = []

        logger.info(
            f"EmissionsComplianceOrchestrator {self.config.agent_id} initialized "
            f"(version: {self.config.version}, jurisdiction: {self.config.jurisdiction})"
        )

    def _initialize_tools(self) -> None:
        """Initialize emissions calculation tools."""
        try:
            self.tools = EmissionsComplianceTools(self.config)
            logger.info("EmissionsComplianceTools initialized successfully")
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
                    temperature=self.config.llm_temperature,
                    seed=self.config.llm_seed,
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
        Main execution entry point for emissions compliance monitoring.

        This method routes requests to appropriate operation modes and
        handles common concerns like caching, metrics, and error handling.

        Args:
            input_data: Dictionary containing:
                - operation_mode: One of monitor, report, alert, analyze, predict, audit, benchmark, validate
                - cems_data: Continuous emissions monitoring data
                - fuel_data: Fuel consumption and analysis data
                - process_parameters: Process operating conditions
                - jurisdiction: Regulatory jurisdiction (optional)

        Returns:
            Dictionary containing calculation results with:
                - Primary results based on operation mode
                - provenance_hash: SHA-256 hash for audit trail
                - execution_time_ms: Processing duration
                - validation_status: Data quality validation results
                - compliance_status: Regulatory compliance status

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
            mode = OperationMode(input_data.get('operation_mode', 'monitor'))
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
            OperationMode.MONITOR: self._execute_monitoring_mode,
            OperationMode.REPORT: self._execute_reporting_mode,
            OperationMode.ALERT: self._execute_alert_mode,
            OperationMode.ANALYZE: self._execute_analysis_mode,
            OperationMode.PREDICT: self._execute_prediction_mode,
            OperationMode.AUDIT: self._execute_audit_mode,
            OperationMode.BENCHMARK: self._execute_benchmark_mode,
            OperationMode.VALIDATE: self._execute_validate_mode
        }

        handler = mode_handlers.get(mode)
        if handler is None:
            raise ValueError(f"Unknown operation mode: {mode}")

        return await handler(input_data)

    # ========================================================================
    # OPERATION MODE HANDLERS
    # ========================================================================

    async def _execute_monitoring_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute real-time CEMS data monitoring mode.

        Processes continuous emissions monitoring data to calculate
        current emissions levels and check compliance.

        Args:
            input_data: CEMS data with fuel and process parameters

        Returns:
            Dictionary with real-time emissions and compliance status
        """
        cems_data = input_data.get('cems_data', {})
        fuel_data = input_data.get('fuel_data', {})
        process_parameters = input_data.get('process_parameters', {})

        # Calculate all emissions
        emissions_result = await self._calculate_all_emissions(cems_data, fuel_data, process_parameters)

        # Check compliance against limits
        jurisdiction = input_data.get('jurisdiction', self.config.jurisdiction)
        compliance_result = await self._check_multi_jurisdiction_compliance(
            emissions_result,
            jurisdiction,
            process_parameters
        )

        # Check for violations and generate alerts if needed
        violations = []
        alerts = []
        if compliance_result['overall_status'] == ComplianceStatus.NON_COMPLIANT.value:
            violations = compliance_result.get('violations', [])
            alerts = self._generate_violation_alerts(violations)
            self.metrics.record_violation(len(violations))
            self.metrics.record_alert(len(alerts))

        return {
            'monitoring_status': 'active',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'emissions': {
                'nox_ppm': round(emissions_result['nox']['concentration_ppm'], 2),
                'nox_lb_mmbtu': round(emissions_result['nox']['emission_rate_lb_mmbtu'], 4),
                'nox_kg_hr': round(emissions_result['nox']['mass_rate_kg_hr'], 2),
                'sox_ppm': round(emissions_result['sox']['concentration_ppm'], 2),
                'sox_lb_mmbtu': round(emissions_result['sox']['emission_rate_lb_mmbtu'], 4),
                'sox_kg_hr': round(emissions_result['sox']['mass_rate_kg_hr'], 2),
                'co2_percent': round(emissions_result['co2']['concentration_percent'], 2),
                'co2_tons_hr': round(emissions_result['co2']['mass_rate_tons_hr'], 2),
                'pm_mg_m3': round(emissions_result['pm']['concentration_mg_m3'], 2),
                'pm_kg_hr': round(emissions_result['pm']['mass_rate_kg_hr'], 2)
            },
            'compliance_status': compliance_result['overall_status'],
            'compliance_details': {
                'nox_compliant': compliance_result['nox_status'] == 'compliant',
                'sox_compliant': compliance_result['sox_status'] == 'compliant',
                'co2_compliant': compliance_result['co2_status'] == 'compliant',
                'pm_compliant': compliance_result['pm_status'] == 'compliant'
            },
            'violations': violations,
            'alerts': alerts,
            'data_quality': {
                'cems_status': cems_data.get('quality_code', 'valid'),
                'validation_passed': len(violations) == 0
            },
            'calculation_details': emissions_result
        }

    async def _execute_reporting_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regulatory report generation mode.

        Generates compliance reports in EPA, EU, or China formats
        with complete data summaries and attestations.

        Args:
            input_data: Reporting period data and parameters

        Returns:
            Dictionary with regulatory report data
        """
        report_format = input_data.get('report_format', self.config.default_report_format)
        reporting_period = input_data.get('reporting_period', {})
        facility_data = input_data.get('facility_data', {})
        emissions_data = input_data.get('emissions_data', [])

        # Generate regulatory report
        report_result = await asyncio.to_thread(
            self.tools.generate_regulatory_report,
            report_format,
            reporting_period,
            facility_data,
            emissions_data
        )
        self.metrics.record_calculation(0, 'report')

        return {
            'report_status': 'generated',
            'report_format': report_format,
            'report_id': report_result.report_id,
            'reporting_period': reporting_period,
            'facility_id': facility_data.get('facility_id', 'unknown'),
            'summary': {
                'total_operating_hours': report_result.total_operating_hours,
                'avg_nox_lb_mmbtu': round(report_result.avg_nox_lb_mmbtu, 4),
                'avg_sox_lb_mmbtu': round(report_result.avg_sox_lb_mmbtu, 4),
                'total_co2_tons': round(report_result.total_co2_tons, 2),
                'avg_pm_lb_mmbtu': round(report_result.avg_pm_lb_mmbtu, 4),
                'compliance_rate_percent': round(report_result.compliance_rate_percent, 2),
                'exceedance_count': report_result.exceedance_count,
                'data_availability_percent': round(report_result.data_availability_percent, 2)
            },
            'compliance_certification': {
                'certified': report_result.compliance_rate_percent >= 95.0,
                'certifier': report_result.certifier,
                'certification_date': report_result.certification_date
            },
            'regulatory_submission': {
                'jurisdiction': report_result.jurisdiction,
                'submission_deadline': report_result.submission_deadline,
                'format_version': report_result.format_version
            },
            'report_sections': report_result.sections,
            'attachments': report_result.attachments,
            'provenance_hash': report_result.provenance_hash
        }

    async def _execute_alert_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute violation detection and notification mode.

        Detects emissions violations and generates appropriate alerts
        based on severity and regulatory requirements.

        Args:
            input_data: Current emissions data for violation checking

        Returns:
            Dictionary with detected violations and alerts
        """
        cems_data = input_data.get('cems_data', {})
        fuel_data = input_data.get('fuel_data', {})
        process_parameters = input_data.get('process_parameters', {})
        permit_limits = input_data.get('permit_limits', {})

        # Calculate current emissions
        emissions_result = await self._calculate_all_emissions(cems_data, fuel_data, process_parameters)

        # Detect violations
        violations = await asyncio.to_thread(
            self.tools.detect_violations,
            emissions_result,
            permit_limits or self._get_default_permit_limits()
        )

        # Generate alerts for violations
        alerts = []
        for violation in violations:
            alert = self._create_alert(violation)
            alerts.append(alert)
            self._active_alerts.append(alert)
            self._alert_history.append(alert)

        # Record metrics
        if violations:
            self.metrics.record_violation(len(violations))
            self.metrics.record_alert(len(alerts))

        return {
            'alert_status': 'processed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'violations_detected': len(violations),
            'violations': [
                {
                    'violation_id': v.violation_id,
                    'pollutant': v.pollutant,
                    'measured_value': round(v.measured_value, 2),
                    'limit_value': round(v.limit_value, 2),
                    'exceedance_percent': round(v.exceedance_percent, 2),
                    'severity': v.severity,
                    'duration_minutes': v.duration_minutes,
                    'regulatory_reference': v.regulatory_reference
                }
                for v in violations
            ],
            'alerts': alerts,
            'active_alerts_count': len(self._active_alerts),
            'notification_status': {
                'email_sent': len(alerts) > 0 and self.config.enable_email_alerts,
                'sms_sent': len(alerts) > 0 and self.config.enable_sms_alerts,
                'webhook_triggered': len(alerts) > 0 and self.config.enable_webhook_alerts
            }
        }

    async def _execute_analysis_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute emissions trend analysis mode.

        Analyzes historical emissions data to identify trends,
        patterns, and correlations with operating conditions.

        Args:
            input_data: Historical emissions data for analysis

        Returns:
            Dictionary with trend analysis results
        """
        historical_data = input_data.get('historical_data', [])
        analysis_period = input_data.get('analysis_period', {})
        process_parameters = input_data.get('process_parameters', {})

        if not historical_data:
            return {
                'analysis_status': 'no_data',
                'message': 'No historical data provided for analysis'
            }

        # Calculate statistics for each pollutant
        nox_values = [d.get('nox_ppm', 0) for d in historical_data]
        sox_values = [d.get('sox_ppm', 0) for d in historical_data]
        co2_values = [d.get('co2_percent', 0) for d in historical_data]
        pm_values = [d.get('pm_mg_m3', 0) for d in historical_data]

        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {'min': 0, 'max': 0, 'avg': 0, 'std_dev': 0}
            n = len(values)
            avg = sum(values) / n
            variance = sum((x - avg) ** 2 for x in values) / n if n > 1 else 0
            std_dev = variance ** 0.5
            return {
                'min': round(min(values), 2),
                'max': round(max(values), 2),
                'avg': round(avg, 2),
                'std_dev': round(std_dev, 2)
            }

        # Trend calculation (simple linear regression)
        def calculate_trend(values: List[float]) -> Dict[str, Any]:
            n = len(values)
            if n < 2:
                return {'slope': 0, 'direction': 'stable', 'r_squared': 0}

            x_values = list(range(n))
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)

            if denominator == 0:
                return {'slope': 0, 'direction': 'stable', 'r_squared': 0}

            slope = numerator / denominator

            # Calculate R-squared
            y_pred = [slope * x + (y_mean - slope * x_mean) for x in x_values]
            ss_res = sum((y - yp) ** 2 for y, yp in zip(values, y_pred))
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            direction = 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')

            return {
                'slope': round(slope, 4),
                'direction': direction,
                'r_squared': round(r_squared, 4)
            }

        return {
            'analysis_status': 'completed',
            'analysis_period': analysis_period,
            'data_points': len(historical_data),
            'statistics': {
                'nox': calculate_stats(nox_values),
                'sox': calculate_stats(sox_values),
                'co2': calculate_stats(co2_values),
                'pm': calculate_stats(pm_values)
            },
            'trends': {
                'nox': calculate_trend(nox_values),
                'sox': calculate_trend(sox_values),
                'co2': calculate_trend(co2_values),
                'pm': calculate_trend(pm_values)
            },
            'correlations': {
                'nox_vs_load': self._calculate_correlation(
                    nox_values,
                    [d.get('load_percent', 100) for d in historical_data]
                ),
                'sox_vs_fuel_sulfur': self._calculate_correlation(
                    sox_values,
                    [d.get('fuel_sulfur_percent', 0.5) for d in historical_data]
                )
            },
            'recommendations': self._generate_analysis_recommendations(
                calculate_stats(nox_values),
                calculate_stats(sox_values),
                calculate_trend(nox_values),
                calculate_trend(sox_values)
            )
        }

    async def _execute_prediction_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute exceedance prediction mode.

        Predicts potential emissions exceedances based on current
        trends and operating conditions.

        Args:
            input_data: Current and historical data for prediction

        Returns:
            Dictionary with exceedance predictions
        """
        current_data = input_data.get('current_data', {})
        historical_data = input_data.get('historical_data', [])
        forecast_horizon_hours = input_data.get('forecast_horizon_hours', 24)
        permit_limits = input_data.get('permit_limits', self._get_default_permit_limits())

        # Get current emissions
        current_nox = current_data.get('nox_ppm', 0)
        current_sox = current_data.get('sox_ppm', 0)
        current_co2 = current_data.get('co2_percent', 0)
        current_pm = current_data.get('pm_mg_m3', 0)

        # Calculate trends from historical data
        predictions = {}
        for pollutant, current_value in [
            ('nox', current_nox),
            ('sox', current_sox),
            ('co2', current_co2),
            ('pm', current_pm)
        ]:
            historical_values = [d.get(f'{pollutant}_ppm', d.get(f'{pollutant}_percent', d.get(f'{pollutant}_mg_m3', 0)))
                               for d in historical_data[-24:]]  # Last 24 data points

            if historical_values:
                # Simple linear extrapolation
                trend = (historical_values[-1] - historical_values[0]) / max(len(historical_values), 1)
                predicted_value = current_value + (trend * forecast_horizon_hours)
            else:
                predicted_value = current_value

            limit_key = f'{pollutant}_limit'
            limit = permit_limits.get(limit_key, 100)

            exceedance_probability = min(100, max(0, (predicted_value / limit) * 100 - 50))
            if predicted_value > limit:
                exceedance_probability = min(100, 50 + (predicted_value - limit) / limit * 50)

            predictions[pollutant] = {
                'current_value': round(current_value, 2),
                'predicted_value': round(predicted_value, 2),
                'limit_value': round(limit, 2),
                'exceedance_probability_percent': round(exceedance_probability, 1),
                'trend_direction': 'increasing' if predicted_value > current_value else 'decreasing',
                'time_to_exceedance_hours': self._calculate_time_to_exceedance(
                    current_value, predicted_value, limit, forecast_horizon_hours
                )
            }

        self.metrics.record_calculation(0, 'prediction')

        # Generate warnings for high probability exceedances
        warnings = []
        for pollutant, pred in predictions.items():
            if pred['exceedance_probability_percent'] > 70:
                warnings.append({
                    'pollutant': pollutant.upper(),
                    'severity': 'high' if pred['exceedance_probability_percent'] > 90 else 'medium',
                    'message': f"{pollutant.upper()} exceedance likely within {pred['time_to_exceedance_hours']} hours",
                    'recommended_action': self._get_mitigation_recommendation(pollutant)
                })

        return {
            'prediction_status': 'completed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'forecast_horizon_hours': forecast_horizon_hours,
            'predictions': predictions,
            'warnings': warnings,
            'confidence_level': 'medium' if len(historical_data) > 12 else 'low',
            'model_type': 'linear_extrapolation',
            'recommendations': [w['recommended_action'] for w in warnings] if warnings else [
                'Continue monitoring - no exceedances predicted'
            ]
        }

    async def _execute_audit_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute compliance audit trail generation mode.

        Generates complete audit trail with SHA-256 provenance
        hashes for regulatory compliance verification.

        Args:
            input_data: Data to include in audit trail

        Returns:
            Dictionary with audit trail information
        """
        audit_period = input_data.get('audit_period', {})
        facility_data = input_data.get('facility_data', {})
        emissions_records = input_data.get('emissions_records', [])
        compliance_events = input_data.get('compliance_events', [])

        # Generate audit trail
        audit_result = await asyncio.to_thread(
            self.tools.generate_audit_trail,
            audit_period,
            facility_data,
            emissions_records,
            compliance_events
        )
        self.metrics.record_calculation(0, 'audit')

        return {
            'audit_status': 'completed',
            'audit_id': audit_result.audit_id,
            'audit_period': audit_period,
            'facility_id': facility_data.get('facility_id', 'unknown'),
            'summary': {
                'total_records': audit_result.total_records,
                'compliant_records': audit_result.compliant_records,
                'non_compliant_records': audit_result.non_compliant_records,
                'data_quality_score': round(audit_result.data_quality_score, 2),
                'completeness_percent': round(audit_result.completeness_percent, 2)
            },
            'provenance': {
                'hash_algorithm': 'SHA-256',
                'root_hash': audit_result.root_hash,
                'record_hashes': audit_result.record_hashes[:10],  # First 10 for brevity
                'chain_valid': audit_result.chain_valid
            },
            'compliance_events': audit_result.compliance_events,
            'data_corrections': audit_result.data_corrections,
            'certification': {
                'certifier': audit_result.certifier,
                'certification_date': audit_result.certification_date,
                'certification_statement': audit_result.certification_statement
            },
            'regulatory_requirements': {
                'epa_part_75_compliant': audit_result.epa_part_75_compliant,
                'data_retention_met': audit_result.data_retention_met,
                'qapp_requirements_met': audit_result.qapp_requirements_met
            }
        }

    async def _execute_benchmark_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute permit limits benchmark comparison mode.

        Compares current and historical emissions against permit
        limits and industry benchmarks.

        Args:
            input_data: Emissions data for benchmarking

        Returns:
            Dictionary with benchmark comparison results
        """
        current_emissions = input_data.get('current_emissions', {})
        permit_limits = input_data.get('permit_limits', self._get_default_permit_limits())
        industry_benchmarks = input_data.get('industry_benchmarks', {})
        process_type = input_data.get('process_type', 'boiler')

        # Get default industry benchmarks if not provided
        if not industry_benchmarks:
            industry_benchmarks = self._get_industry_benchmarks(process_type)

        # Calculate benchmark comparisons
        comparisons = {}
        for pollutant in ['nox', 'sox', 'co2', 'pm']:
            current_value = current_emissions.get(f'{pollutant}_value', 0)
            permit_limit = permit_limits.get(f'{pollutant}_limit', 100)
            industry_avg = industry_benchmarks.get(f'{pollutant}_avg', permit_limit * 0.7)
            industry_best = industry_benchmarks.get(f'{pollutant}_best', permit_limit * 0.5)

            # Calculate percentages
            permit_usage = (current_value / permit_limit * 100) if permit_limit > 0 else 0
            vs_industry_avg = ((current_value - industry_avg) / industry_avg * 100) if industry_avg > 0 else 0
            vs_industry_best = ((current_value - industry_best) / industry_best * 100) if industry_best > 0 else 0

            # Determine rating
            if permit_usage < 50:
                rating = 'excellent'
            elif permit_usage < 75:
                rating = 'good'
            elif permit_usage < 90:
                rating = 'acceptable'
            elif permit_usage < 100:
                rating = 'marginal'
            else:
                rating = 'non_compliant'

            comparisons[pollutant] = {
                'current_value': round(current_value, 2),
                'permit_limit': round(permit_limit, 2),
                'permit_usage_percent': round(permit_usage, 1),
                'industry_average': round(industry_avg, 2),
                'industry_best': round(industry_best, 2),
                'vs_industry_avg_percent': round(vs_industry_avg, 1),
                'vs_industry_best_percent': round(vs_industry_best, 1),
                'rating': rating,
                'headroom_to_limit': round(permit_limit - current_value, 2)
            }

        # Overall assessment
        ratings = [c['rating'] for c in comparisons.values()]
        if 'non_compliant' in ratings:
            overall_rating = 'non_compliant'
        elif 'marginal' in ratings:
            overall_rating = 'needs_improvement'
        elif all(r in ['excellent', 'good'] for r in ratings):
            overall_rating = 'excellent'
        else:
            overall_rating = 'acceptable'

        return {
            'benchmark_status': 'completed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'process_type': process_type,
            'comparisons': comparisons,
            'overall_rating': overall_rating,
            'permit_compliance': {
                'all_within_limits': all(c['permit_usage_percent'] < 100 for c in comparisons.values()),
                'closest_to_limit': max(comparisons.items(), key=lambda x: x[1]['permit_usage_percent'])[0],
                'highest_margin': min(comparisons.items(), key=lambda x: x[1]['permit_usage_percent'])[0]
            },
            'industry_comparison': {
                'below_average_count': sum(1 for c in comparisons.values() if c['vs_industry_avg_percent'] < 0),
                'above_average_count': sum(1 for c in comparisons.values() if c['vs_industry_avg_percent'] >= 0),
                'best_in_class_count': sum(1 for c in comparisons.values() if c['vs_industry_best_percent'] <= 0)
            },
            'recommendations': self._generate_benchmark_recommendations(comparisons)
        }

    async def _execute_validate_mode(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute CEMS data quality validation mode.

        Validates CEMS data quality according to EPA Part 75
        requirements and identifies data issues.

        Args:
            input_data: CEMS data for validation

        Returns:
            Dictionary with data quality validation results
        """
        cems_data = input_data.get('cems_data', {})
        validation_period = input_data.get('validation_period', {})
        qapp_requirements = input_data.get('qapp_requirements', {})

        # Perform validations
        validations = []

        # 1. Range validation
        range_validation = self._validate_data_ranges(cems_data)
        validations.append(range_validation)

        # 2. Completeness validation
        completeness_validation = self._validate_data_completeness(cems_data)
        validations.append(completeness_validation)

        # 3. Consistency validation
        consistency_validation = self._validate_data_consistency(cems_data)
        validations.append(consistency_validation)

        # 4. Calibration validation
        calibration_validation = self._validate_calibration_status(cems_data)
        validations.append(calibration_validation)

        # Calculate overall status
        all_passed = all(v['status'] == 'passed' for v in validations)
        any_failed = any(v['status'] == 'failed' for v in validations)

        if all_passed:
            overall_status = 'passed'
            self.metrics.record_validation(True)
        elif any_failed:
            overall_status = 'failed'
            self.metrics.record_validation(False)
        else:
            overall_status = 'warning'
            self.metrics.record_validation(True)

        # Count data quality issues
        issues = []
        for v in validations:
            issues.extend(v.get('issues', []))
        if issues:
            self.metrics.record_data_quality_issue(len(issues))

        return {
            'validation_status': overall_status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'validation_period': validation_period,
            'validations': validations,
            'summary': {
                'total_checks': len(validations),
                'passed': sum(1 for v in validations if v['status'] == 'passed'),
                'warnings': sum(1 for v in validations if v['status'] == 'warning'),
                'failed': sum(1 for v in validations if v['status'] == 'failed'),
                'data_quality_score': self._calculate_data_quality_score(validations)
            },
            'issues': issues,
            'data_availability': {
                'total_expected_hours': completeness_validation.get('expected_hours', 0),
                'valid_hours': completeness_validation.get('valid_hours', 0),
                'availability_percent': completeness_validation.get('availability_percent', 0)
            },
            'epa_part_75_status': {
                'compliant': overall_status != 'failed',
                'substitute_data_required': any_failed,
                'rata_status': calibration_validation.get('rata_status', 'unknown')
            },
            'recommendations': self._generate_validation_recommendations(validations, issues)
        }

    # ========================================================================
    # CORE CALCULATION METHODS
    # ========================================================================

    async def _calculate_all_emissions(
        self,
        cems_data: Dict[str, Any],
        fuel_data: Dict[str, Any],
        process_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate all emissions (NOx + SOx + CO2 + PM).

        Uses deterministic formulas compliant with EPA methods.
        No LLM involvement in calculations.

        Args:
            cems_data: CEMS measurements
            fuel_data: Fuel composition and flow
            process_parameters: Operating conditions

        Returns:
            Dictionary with all emissions calculations
        """
        # Calculate NOx emissions
        nox_result = await asyncio.to_thread(
            self.tools.calculate_nox_emissions,
            cems_data,
            fuel_data,
            process_parameters
        )
        self.metrics.record_calculation(0, 'nox')

        # Calculate SOx emissions
        sox_result = await asyncio.to_thread(
            self.tools.calculate_sox_emissions,
            fuel_data,
            process_parameters
        )
        self.metrics.record_calculation(0, 'sox')

        # Calculate CO2 emissions
        co2_result = await asyncio.to_thread(
            self.tools.calculate_co2_emissions,
            fuel_data,
            process_parameters
        )
        self.metrics.record_calculation(0, 'co2')

        # Calculate PM emissions
        pm_result = await asyncio.to_thread(
            self.tools.calculate_particulate_matter,
            cems_data,
            fuel_data,
            process_parameters
        )
        self.metrics.record_calculation(0, 'pm')

        return {
            'nox': nox_result.to_dict(),
            'sox': sox_result.to_dict(),
            'co2': co2_result.to_dict(),
            'pm': pm_result.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'calculation_method': 'EPA_Methods_19_2_3A_5'
        }

    async def _check_multi_jurisdiction_compliance(
        self,
        emissions_result: Dict[str, Any],
        jurisdiction: str,
        process_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check compliance against multi-jurisdiction rules (EPA + EU + China).

        Args:
            emissions_result: Calculated emissions
            jurisdiction: Primary jurisdiction
            process_parameters: Process parameters

        Returns:
            Dictionary with compliance status for each pollutant
        """
        compliance_result = await asyncio.to_thread(
            self.tools.check_compliance_status,
            emissions_result,
            jurisdiction,
            process_parameters
        )
        self.metrics.record_calculation(0, 'compliance')

        return compliance_result.to_dict()

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

        # Check required fields based on operation mode
        operation_mode = input_data.get('operation_mode', 'monitor')

        if operation_mode in ['monitor', 'alert', 'predict']:
            if 'cems_data' not in input_data and 'current_data' not in input_data:
                errors.append("Missing required field: cems_data or current_data")

        if operation_mode == 'report':
            if 'reporting_period' not in input_data:
                errors.append("Missing required field: reporting_period")
            if 'facility_data' not in input_data:
                warnings.append("Missing facility_data - using defaults")

        if operation_mode == 'audit':
            if 'audit_period' not in input_data:
                errors.append("Missing required field: audit_period")

        # Validate CEMS data if present
        cems_data = input_data.get('cems_data', {})
        if cems_data:
            # Check for negative values
            for key in ['nox_ppm', 'sox_ppm', 'co2_percent', 'pm_mg_m3']:
                value = cems_data.get(key, 0)
                if value < 0:
                    errors.append(f"Invalid negative value for {key}: {value}")

            # Check for unreasonable values
            if cems_data.get('nox_ppm', 0) > 5000:
                warnings.append(f"NOx value {cems_data['nox_ppm']} ppm seems high - verify measurement")
            if cems_data.get('co2_percent', 0) > 25:
                warnings.append(f"CO2 value {cems_data['co2_percent']}% seems high - verify measurement")

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

    def _validate_data_ranges(self, cems_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CEMS data ranges."""
        issues = []
        status = 'passed'

        range_limits = {
            'nox_ppm': (0, 2000),
            'sox_ppm': (0, 3000),
            'co2_percent': (0, 20),
            'pm_mg_m3': (0, 500),
            'o2_percent': (0, 21),
            'flow_rate_scfm': (0, 1000000),
            'temperature_f': (100, 1000)
        }

        for key, (min_val, max_val) in range_limits.items():
            value = cems_data.get(key)
            if value is not None:
                if value < min_val:
                    issues.append(f"{key} value {value} below minimum {min_val}")
                    status = 'failed'
                elif value > max_val:
                    issues.append(f"{key} value {value} above maximum {max_val}")
                    status = 'warning'

        return {
            'validation_type': 'range_check',
            'status': status,
            'issues': issues
        }

    def _validate_data_completeness(self, cems_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CEMS data completeness."""
        required_fields = ['nox_ppm', 'o2_percent', 'flow_rate_scfm']
        missing_fields = [f for f in required_fields if f not in cems_data]

        expected_hours = cems_data.get('expected_hours', 720)  # Default 30 days
        valid_hours = cems_data.get('valid_hours', expected_hours)
        availability_percent = (valid_hours / expected_hours * 100) if expected_hours > 0 else 0

        status = 'passed'
        issues = []

        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
            status = 'failed'

        if availability_percent < 90:
            issues.append(f"Data availability {availability_percent:.1f}% below 90% requirement")
            status = 'failed' if availability_percent < 75 else 'warning'

        return {
            'validation_type': 'completeness_check',
            'status': status,
            'issues': issues,
            'expected_hours': expected_hours,
            'valid_hours': valid_hours,
            'availability_percent': round(availability_percent, 2)
        }

    def _validate_data_consistency(self, cems_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CEMS data consistency."""
        issues = []
        status = 'passed'

        # Check O2 + CO2 consistency (should sum to approximately 21%)
        o2_percent = cems_data.get('o2_percent', 0)
        co2_percent = cems_data.get('co2_percent', 0)

        if o2_percent > 0 and co2_percent > 0:
            # Rough check - sum should be between 15% and 21%
            total = o2_percent + co2_percent
            if total < 12 or total > 23:
                issues.append(f"O2 + CO2 sum ({total:.1f}%) outside expected range (12-23%)")
                status = 'warning'

        # Check NOx vs temperature correlation
        temp_f = cems_data.get('temperature_f', 300)
        nox_ppm = cems_data.get('nox_ppm', 0)

        if temp_f < 250 and nox_ppm > 100:
            issues.append(f"High NOx ({nox_ppm} ppm) at low temperature ({temp_f}F) - verify readings")
            status = 'warning'

        return {
            'validation_type': 'consistency_check',
            'status': status,
            'issues': issues
        }

    def _validate_calibration_status(self, cems_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CEMS calibration status."""
        issues = []
        status = 'passed'

        last_calibration = cems_data.get('last_calibration_date')
        rata_date = cems_data.get('last_rata_date')

        if last_calibration:
            try:
                cal_date = datetime.fromisoformat(last_calibration.replace('Z', '+00:00'))
                days_since_cal = (datetime.now(timezone.utc) - cal_date).days
                if days_since_cal > 7:
                    issues.append(f"Daily calibration overdue by {days_since_cal - 7} days")
                    status = 'warning'
            except ValueError:
                issues.append("Invalid calibration date format")
                status = 'warning'

        rata_status = 'unknown'
        if rata_date:
            try:
                rata_dt = datetime.fromisoformat(rata_date.replace('Z', '+00:00'))
                days_since_rata = (datetime.now(timezone.utc) - rata_dt).days
                if days_since_rata > 365:
                    issues.append(f"RATA overdue - last performed {days_since_rata} days ago")
                    status = 'failed'
                    rata_status = 'overdue'
                else:
                    rata_status = 'current'
            except ValueError:
                issues.append("Invalid RATA date format")
                rata_status = 'unknown'

        return {
            'validation_type': 'calibration_check',
            'status': status,
            'issues': issues,
            'rata_status': rata_status
        }

    def _calculate_data_quality_score(self, validations: List[Dict[str, Any]]) -> float:
        """Calculate overall data quality score (0-100)."""
        if not validations:
            return 0.0

        scores = []
        for v in validations:
            if v['status'] == 'passed':
                scores.append(100)
            elif v['status'] == 'warning':
                scores.append(75)
            else:
                scores.append(0)

        return round(sum(scores) / len(scores), 2)

    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from input data."""
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
        return f"GL010-{timestamp[:10]}-{data_hash}"

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
                'compliance_status': result.get('compliance_status'),
                'violations_count': len(result.get('violations', []))
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
            'deterministic': True,
            'provenance_hash': self._calculate_provenance_hash({}, result)
        }

        return result

    def _get_default_permit_limits(self) -> Dict[str, float]:
        """Get default permit limits based on jurisdiction."""
        jurisdiction = self.config.jurisdiction

        if jurisdiction == 'EPA':
            return {
                'nox_limit': self.config.nox_limit_ppm,
                'sox_limit': self.config.sox_limit_ppm,
                'co2_limit': self.config.co2_limit_tons_hr,
                'pm_limit': self.config.pm_limit_mg_m3
            }
        elif jurisdiction == 'EU_IED':
            return {
                'nox_limit': 100.0,  # mg/Nm3
                'sox_limit': 150.0,  # mg/Nm3
                'co2_limit': 100.0,  # tons/hr
                'pm_limit': 10.0     # mg/Nm3
            }
        else:  # CHINA_MEE
            return {
                'nox_limit': 50.0,
                'sox_limit': 35.0,
                'co2_limit': 80.0,
                'pm_limit': 10.0
            }

    def _get_industry_benchmarks(self, process_type: str) -> Dict[str, float]:
        """Get industry benchmarks for process type."""
        benchmarks = {
            'boiler': {
                'nox_avg': 35.0, 'nox_best': 15.0,
                'sox_avg': 70.0, 'sox_best': 30.0,
                'co2_avg': 35.0, 'co2_best': 25.0,
                'pm_avg': 20.0, 'pm_best': 10.0
            },
            'turbine': {
                'nox_avg': 25.0, 'nox_best': 9.0,
                'sox_avg': 20.0, 'sox_best': 5.0,
                'co2_avg': 30.0, 'co2_best': 20.0,
                'pm_avg': 15.0, 'pm_best': 5.0
            },
            'incinerator': {
                'nox_avg': 100.0, 'nox_best': 50.0,
                'sox_avg': 80.0, 'sox_best': 40.0,
                'co2_avg': 50.0, 'co2_best': 30.0,
                'pm_avg': 25.0, 'pm_best': 10.0
            }
        }
        return benchmarks.get(process_type, benchmarks['boiler'])

    def _generate_violation_alerts(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts for detected violations."""
        alerts = []
        for violation in violations:
            alert = self._create_alert(violation)
            alerts.append(alert)
        return alerts

    def _create_alert(self, violation: Any) -> Dict[str, Any]:
        """Create alert from violation."""
        if isinstance(violation, dict):
            severity = violation.get('severity', 'medium')
            pollutant = violation.get('pollutant', 'unknown')
            exceedance = violation.get('exceedance_percent', 0)
        else:
            severity = getattr(violation, 'severity', 'medium')
            pollutant = getattr(violation, 'pollutant', 'unknown')
            exceedance = getattr(violation, 'exceedance_percent', 0)

        return {
            'alert_id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'severity': severity,
            'pollutant': pollutant,
            'message': f"{pollutant.upper()} exceedance detected: {exceedance:.1f}% over limit",
            'action_required': severity in ['high', 'critical'],
            'notification_channels': ['email', 'dashboard'] if severity == 'high' else ['dashboard']
        }

    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x_values)
        if n < 2 or n != len(y_values):
            return 0.0

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_ss = sum((x - x_mean) ** 2 for x in x_values)
        y_ss = sum((y - y_mean) ** 2 for y in y_values)

        denominator = (x_ss * y_ss) ** 0.5
        if denominator == 0:
            return 0.0

        return round(numerator / denominator, 4)

    def _calculate_time_to_exceedance(
        self,
        current_value: float,
        predicted_value: float,
        limit: float,
        horizon_hours: float
    ) -> Optional[float]:
        """Calculate estimated time to exceedance."""
        if current_value >= limit:
            return 0.0
        if predicted_value <= limit:
            return None

        # Linear interpolation
        rate = (predicted_value - current_value) / horizon_hours
        if rate <= 0:
            return None

        time_to_limit = (limit - current_value) / rate
        return round(time_to_limit, 1)

    def _get_mitigation_recommendation(self, pollutant: str) -> str:
        """Get mitigation recommendation for pollutant."""
        recommendations = {
            'nox': 'Reduce combustion temperature or increase SCR/SNCR injection rate',
            'sox': 'Switch to lower sulfur fuel or increase FGD scrubber efficiency',
            'co2': 'Improve combustion efficiency or reduce load',
            'pm': 'Check baghouse/ESP operation and increase cleaning frequency'
        }
        return recommendations.get(pollutant, 'Review process parameters')

    def _generate_analysis_recommendations(
        self,
        nox_stats: Dict[str, float],
        sox_stats: Dict[str, float],
        nox_trend: Dict[str, Any],
        sox_trend: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if nox_trend['direction'] == 'increasing':
            recommendations.append('NOx trending upward - consider combustion optimization')

        if sox_stats['avg'] > 50:
            recommendations.append('SOx average high - evaluate fuel sulfur content')

        if nox_stats['std_dev'] > nox_stats['avg'] * 0.3:
            recommendations.append('High NOx variability - review process control stability')

        if not recommendations:
            recommendations.append('Emissions within normal operating parameters')

        return recommendations

    def _generate_benchmark_recommendations(
        self,
        comparisons: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on benchmark comparison."""
        recommendations = []

        for pollutant, data in comparisons.items():
            if data['rating'] == 'non_compliant':
                recommendations.append(
                    f"CRITICAL: {pollutant.upper()} exceeds permit limit - immediate action required"
                )
            elif data['rating'] == 'marginal':
                recommendations.append(
                    f"{pollutant.upper()} approaching permit limit ({data['permit_usage_percent']:.0f}% usage) - "
                    f"implement controls to increase margin"
                )
            elif data['vs_industry_avg_percent'] > 20:
                recommendations.append(
                    f"{pollutant.upper()} above industry average by {data['vs_industry_avg_percent']:.0f}% - "
                    f"evaluate improvement opportunities"
                )

        if not recommendations:
            recommendations.append('All emissions within acceptable ranges and comparable to industry benchmarks')

        return recommendations

    def _generate_validation_recommendations(
        self,
        validations: List[Dict[str, Any]],
        issues: List[str]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        for v in validations:
            if v['status'] == 'failed':
                if v['validation_type'] == 'calibration_check':
                    recommendations.append('Perform RATA or daily calibration immediately')
                elif v['validation_type'] == 'completeness_check':
                    recommendations.append('Review CEMS uptime and implement substitute data procedures')
                elif v['validation_type'] == 'range_check':
                    recommendations.append('Verify analyzer calibration and span settings')

        if not recommendations:
            recommendations.append('Data quality meets EPA Part 75 requirements')

        return recommendations

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
            # Return minimal status response
            recovery_result = {
                'status': 'error_recovery',
                'original_error': str(error),
                'message': 'Partial result returned due to error',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            return self._add_execution_metadata(
                recovery_result, start_time, execution_id
            )

        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
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
                'jurisdiction': self.config.jurisdiction,
                'active_alerts': len(self._active_alerts),
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

        if metrics['violations_detected'] > 0:
            issues.append(f"{metrics['violations_detected']} violations detected")

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

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return list(self._active_alerts)

    def clear_alert(self, alert_id: str) -> bool:
        """Clear an active alert by ID."""
        for alert in self._active_alerts:
            if alert.get('alert_id') == alert_id:
                self._active_alerts.remove(alert)
                return True
        return False

    async def shutdown(self) -> None:
        """
        Graceful shutdown of orchestrator.

        Cleans up resources and saves state if needed.
        """
        logger.info(f"Shutting down EmissionsComplianceOrchestrator {self.config.agent_id}")

        with self._state_lock:
            self._state = 'shutting_down'

        # Clear cache
        self.cache.clear()

        # Log final metrics
        final_metrics = self.metrics.get_metrics()
        logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")

        with self._state_lock:
            self._state = 'terminated'

        logger.info(f"EmissionsComplianceOrchestrator {self.config.agent_id} shutdown complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_orchestrator(
    config: Optional[EmissionsComplianceConfig] = None,
    **kwargs
) -> EmissionsComplianceOrchestrator:
    """
    Factory function to create EmissionsComplianceOrchestrator.

    Args:
        config: Optional configuration object
        **kwargs: Additional configuration overrides

    Returns:
        Configured EmissionsComplianceOrchestrator instance

    Example:
        >>> orchestrator = create_orchestrator(
        ...     jurisdiction='EU_IED',
        ...     nox_limit_ppm=100.0
        ... )
    """
    if config is None:
        config = EmissionsComplianceConfig(**kwargs)
    elif kwargs:
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return EmissionsComplianceOrchestrator(config)
