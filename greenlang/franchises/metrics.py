# -*- coding: utf-8 -*-
"""
Franchises Prometheus Metrics - AGENT-MRV-027

12 Prometheus metrics with gl_frn_ prefix for monitoring the
GL-MRV-S3-014 Franchises Agent.

This module provides Prometheus metrics tracking for franchise emissions
calculations (Scope 3, Category 14) including franchise-specific,
average-data, spend-based, and hybrid calculation methods across all
10 franchise types and 7 emission sources.

Thread-safe singleton pattern with graceful fallback if Prometheus
unavailable.

Metrics prefix: gl_frn_

12 Prometheus Metrics:
    1.  gl_frn_calculations_total              - Counter: total calculations
    2.  gl_frn_emissions_kg_co2e_total         - Counter: total emissions kgCO2e
    3.  gl_frn_units_processed_total           - Counter: franchise units processed
    4.  gl_frn_excluded_units_total            - Counter: units excluded (DC-FRN-001)
    5.  gl_frn_factor_selections_total         - Counter: EF selections
    6.  gl_frn_compliance_checks_total         - Counter: compliance checks
    7.  gl_frn_batch_jobs_total                - Counter: batch jobs
    8.  gl_frn_network_calculations_total      - Counter: network-level calculations
    9.  gl_frn_calculation_duration_seconds    - Histogram: calculation duration
    10. gl_frn_batch_size                      - Histogram: batch sizes
    11. gl_frn_active_calculations             - Gauge: active calculations
    12. gl_frn_data_coverage_ratio             - Gauge: data coverage ratio

Example:
    >>> metrics = FranchisesMetrics()
    >>> metrics.record_calculation(
    ...     method="franchise_specific",
    ...     franchise_type="qsr_restaurant",
    ...     status="success",
    ...     duration=0.045,
    ...     co2e=185000.0,
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful Prometheus import
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will be no-ops")
    PROMETHEUS_AVAILABLE = False

    class _NoOpMetric:
        """No-op metric stub for environments without prometheus_client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float = 0) -> None:
            pass

        def observe(self, amount: float = 0) -> None:
            pass

        def info(self, data: Optional[Dict[str, str]] = None) -> None:
            pass

    Counter = _NoOpMetric  # type: ignore[misc,assignment]
    Histogram = _NoOpMetric  # type: ignore[misc,assignment]
    Gauge = _NoOpMetric  # type: ignore[misc,assignment]
    Info = _NoOpMetric  # type: ignore[misc,assignment]


# ===========================================================================
# Enumerations -- Franchise domain-specific label value sets
# ===========================================================================


class CalculationMethodLabel(str, Enum):
    """Calculation methods for franchise emissions."""
    FRANCHISE_SPECIFIC = "franchise_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class FranchiseTypeLabel(str, Enum):
    """Franchise business type labels."""
    QSR_RESTAURANT = "qsr_restaurant"
    FULL_SERVICE_RESTAURANT = "full_service_restaurant"
    HOTEL = "hotel"
    CONVENIENCE_STORE = "convenience_store"
    RETAIL_STORE = "retail_store"
    FITNESS_CENTER = "fitness_center"
    AUTOMOTIVE_SERVICE = "automotive_service"
    HEALTHCARE_CLINIC = "healthcare_clinic"
    EDUCATION_CENTER = "education_center"
    OTHER_SERVICE = "other_service"


class CalculationStatusLabel(str, Enum):
    """Calculation operation status."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"
    EXCLUDED = "excluded"


class EmissionSourceLabel(str, Enum):
    """Emission source categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    PURCHASED_ELECTRICITY = "purchased_electricity"
    PURCHASED_HEATING = "purchased_heating"
    PURCHASED_COOLING = "purchased_cooling"
    PROCESS_EMISSIONS = "process_emissions"


class EFSourceLabel(str, Enum):
    """Emission factor sources."""
    DEFRA_2024 = "defra_2024"
    EPA_2024 = "epa_2024"
    IEA_2024 = "iea_2024"
    EGRID_2024 = "egrid_2024"
    IPCC_AR6 = "ipcc_ar6"
    CUSTOM = "custom"


class FrameworkLabel(str, Enum):
    """Compliance frameworks."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class ComplianceStatusLabel(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class BatchStatusLabel(str, Enum):
    """Batch processing job status."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ExclusionReasonLabel(str, Enum):
    """Reason for unit exclusion."""
    COMPANY_OWNED = "company_owned"
    INACTIVE = "inactive"
    INSUFFICIENT_DATA = "insufficient_data"
    DC_RULE = "dc_rule"


# ===========================================================================
# FranchisesMetrics -- Thread-safe Singleton
# ===========================================================================


class FranchisesMetrics:
    """
    Thread-safe singleton metrics collector for Franchises Agent (MRV-027).

    Provides 12 Prometheus metrics for tracking Scope 3 Category 14
    franchise emissions calculations.

    All metrics use the ``gl_frn_`` prefix for namespace isolation.

    Example:
        >>> metrics = FranchisesMetrics()
        >>> metrics.record_calculation(
        ...     method="franchise_specific",
        ...     franchise_type="qsr_restaurant",
        ...     status="success",
        ...     duration=0.045,
        ...     co2e=185000.0,
        ... )
    """

    _instance: Optional["FranchisesMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FranchisesMetrics":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            "calculations": 0,
            "emissions_kg_co2e": 0.0,
            "units_processed": 0,
            "excluded_units": 0,
            "factor_selections": 0,
            "compliance_checks": 0,
            "batch_jobs": 0,
            "network_calculations": 0,
            "errors": 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "FranchisesMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 12 Prometheus metrics with gl_frn_ prefix.
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(
                metric_cls: type, name: str, *args: Any, **kwargs: Any
            ) -> Any:
                """Create a metric, unregistering prior collector on conflict."""
                try:
                    return metric_cls(name, *args, **kwargs)
                except ValueError:
                    try:
                        REGISTRY.unregister(
                            REGISTRY._names_to_collectors.get(name)
                        )
                    except Exception:
                        for collector in list(
                            REGISTRY._names_to_collectors.values()
                        ):
                            try:
                                REGISTRY.unregister(collector)
                            except Exception:
                                pass
                    return metric_cls(name, *args, **kwargs)
        else:
            def _safe_create(
                metric_cls: type, name: str, *args: Any, **kwargs: Any
            ) -> Any:
                return metric_cls(name, *args, **kwargs)

        # 1. gl_frn_calculations_total
        self.calculations_total = _safe_create(
            Counter,
            "gl_frn_calculations_total",
            "Total franchise emission calculations performed",
            ["method", "franchise_type", "status"],
        )

        # 2. gl_frn_emissions_kg_co2e_total
        self.emissions_kg_co2e_total = _safe_create(
            Counter,
            "gl_frn_emissions_kg_co2e_total",
            "Total franchise emissions calculated in kgCO2e",
            ["franchise_type", "emission_source"],
        )

        # 3. gl_frn_units_processed_total
        self.units_processed_total = _safe_create(
            Counter,
            "gl_frn_units_processed_total",
            "Total franchise units processed",
            ["franchise_type", "method"],
        )

        # 4. gl_frn_excluded_units_total
        self.excluded_units_total = _safe_create(
            Counter,
            "gl_frn_excluded_units_total",
            "Total franchise units excluded (DC-FRN-001, inactive, etc.)",
            ["reason"],
        )

        # 5. gl_frn_factor_selections_total
        self.factor_selections_total = _safe_create(
            Counter,
            "gl_frn_factor_selections_total",
            "Total emission factor selections for franchises",
            ["source", "franchise_type"],
        )

        # 6. gl_frn_compliance_checks_total
        self.compliance_checks_total = _safe_create(
            Counter,
            "gl_frn_compliance_checks_total",
            "Total franchise compliance checks performed",
            ["framework", "status"],
        )

        # 7. gl_frn_batch_jobs_total
        self.batch_jobs_total = _safe_create(
            Counter,
            "gl_frn_batch_jobs_total",
            "Total franchise batch processing jobs",
            ["status"],
        )

        # 8. gl_frn_network_calculations_total
        self.network_calculations_total = _safe_create(
            Counter,
            "gl_frn_network_calculations_total",
            "Total network-level franchise calculations",
            ["method"],
        )

        # 9. gl_frn_calculation_duration_seconds
        self.calculation_duration_seconds = _safe_create(
            Histogram,
            "gl_frn_calculation_duration_seconds",
            "Duration of franchise calculation operations",
            ["method", "franchise_type"],
            buckets=(
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                0.5, 1.0, 2.5, 5.0, 10.0,
            ),
        )

        # 10. gl_frn_batch_size
        self.batch_size = _safe_create(
            Histogram,
            "gl_frn_batch_size",
            "Batch calculation size for franchise operations",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
        )

        # 11. gl_frn_active_calculations
        self.active_calculations = _safe_create(
            Gauge,
            "gl_frn_active_calculations",
            "Number of currently active franchise calculations",
        )

        # 12. gl_frn_data_coverage_ratio
        self.data_coverage_ratio = _safe_create(
            Gauge,
            "gl_frn_data_coverage_ratio",
            "Current franchise network data coverage ratio (0-1)",
        )

        # Agent info
        self.agent_info = _safe_create(
            Info,
            "gl_frn_agent",
            "Franchises Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-S3-014",
                    "version": "1.0.0",
                    "scope": "scope_3_category_14",
                    "description": "Franchises emissions calculator",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        franchise_type: str,
        status: str,
        duration: float,
        co2e: float,
        emission_source: str = "purchased_electricity",
    ) -> None:
        """
        Record a franchise emission calculation operation.

        Args:
            method: Calculation method
            franchise_type: Franchise business type
            status: Calculation status
            duration: Operation duration in seconds
            co2e: Emissions calculated in kgCO2e
            emission_source: Primary emission source
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel,
                CalculationMethodLabel.FRANCHISE_SPECIFIC.value,
            )
            franchise_type = self._validate_enum_value(
                franchise_type, FranchiseTypeLabel,
                FranchiseTypeLabel.OTHER_SERVICE.value,
            )
            status = self._validate_enum_value(
                status, CalculationStatusLabel,
                CalculationStatusLabel.ERROR.value,
            )
            emission_source = self._validate_enum_value(
                emission_source, EmissionSourceLabel,
                EmissionSourceLabel.PURCHASED_ELECTRICITY.value,
            )

            self.calculations_total.labels(
                method=method, franchise_type=franchise_type, status=status
            ).inc()

            if duration is not None and duration > 0:
                self.calculation_duration_seconds.labels(
                    method=method, franchise_type=franchise_type
                ).observe(duration)

            if co2e is not None and co2e > 0:
                self.emissions_kg_co2e_total.labels(
                    franchise_type=franchise_type,
                    emission_source=emission_source,
                ).inc(co2e)

                with self._stats_lock:
                    self._in_memory_stats["emissions_kg_co2e"] += co2e

            with self._stats_lock:
                self._in_memory_stats["calculations"] += 1
                if status == CalculationStatusLabel.ERROR.value:
                    self._in_memory_stats["errors"] += 1

            logger.debug(
                "Recorded calculation: method=%s, type=%s, status=%s, "
                "duration=%.3fs, co2e=%.2f kgCO2e",
                method, franchise_type, status,
                duration if duration else 0.0,
                co2e if co2e else 0.0,
            )

        except Exception as e:
            logger.error(
                "Failed to record calculation metrics: %s", e, exc_info=True
            )

    def record_unit_processed(
        self,
        franchise_type: str,
        method: str,
    ) -> None:
        """
        Record a franchise unit being processed.

        Args:
            franchise_type: Franchise business type
            method: Calculation method used
        """
        try:
            franchise_type = self._validate_enum_value(
                franchise_type, FranchiseTypeLabel,
                FranchiseTypeLabel.OTHER_SERVICE.value,
            )
            method = self._validate_enum_value(
                method, CalculationMethodLabel,
                CalculationMethodLabel.FRANCHISE_SPECIFIC.value,
            )

            self.units_processed_total.labels(
                franchise_type=franchise_type, method=method
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["units_processed"] += 1

        except Exception as e:
            logger.error(
                "Failed to record unit processed: %s", e, exc_info=True
            )

    def record_unit_excluded(self, reason: str) -> None:
        """
        Record a franchise unit being excluded.

        Args:
            reason: Exclusion reason
        """
        try:
            reason = self._validate_enum_value(
                reason, ExclusionReasonLabel,
                ExclusionReasonLabel.COMPANY_OWNED.value,
            )

            self.excluded_units_total.labels(reason=reason).inc()

            with self._stats_lock:
                self._in_memory_stats["excluded_units"] += 1

        except Exception as e:
            logger.error(
                "Failed to record unit excluded: %s", e, exc_info=True
            )

    def record_factor_selection(
        self,
        source: str,
        franchise_type: str,
    ) -> None:
        """
        Record an emission factor selection/lookup.

        Args:
            source: EF source
            franchise_type: Franchise type for which EF was selected
        """
        try:
            source = self._validate_enum_value(
                source, EFSourceLabel, EFSourceLabel.DEFRA_2024.value,
            )
            franchise_type = self._validate_enum_value(
                franchise_type, FranchiseTypeLabel,
                FranchiseTypeLabel.OTHER_SERVICE.value,
            )

            self.factor_selections_total.labels(
                source=source, franchise_type=franchise_type
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["factor_selections"] += 1

        except Exception as e:
            logger.error(
                "Failed to record factor selection: %s", e, exc_info=True
            )

    def record_compliance_check(
        self,
        framework: str,
        status: str,
    ) -> None:
        """
        Record a compliance check operation.

        Args:
            framework: Compliance framework
            status: Compliance status
        """
        try:
            framework = self._validate_enum_value(
                framework, FrameworkLabel,
                FrameworkLabel.GHG_PROTOCOL.value,
            )
            status = self._validate_enum_value(
                status, ComplianceStatusLabel,
                ComplianceStatusLabel.NON_COMPLIANT.value,
            )

            self.compliance_checks_total.labels(
                framework=framework, status=status
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["compliance_checks"] += 1

        except Exception as e:
            logger.error(
                "Failed to record compliance check: %s", e, exc_info=True
            )

    def record_batch_job(
        self,
        status: str,
        size: int,
    ) -> None:
        """
        Record a batch processing job.

        Args:
            status: Batch status
            size: Number of items in batch
        """
        try:
            status = self._validate_enum_value(
                status, BatchStatusLabel, BatchStatusLabel.FAILED.value,
            )

            self.batch_jobs_total.labels(status=status).inc()

            if size is not None and size > 0:
                self.batch_size.observe(size)

            with self._stats_lock:
                self._in_memory_stats["batch_jobs"] += 1

        except Exception as e:
            logger.error(
                "Failed to record batch job: %s", e, exc_info=True
            )

    def record_network_calculation(self, method: str) -> None:
        """
        Record a network-level calculation.

        Args:
            method: Calculation method used
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethodLabel,
                CalculationMethodLabel.HYBRID.value,
            )

            self.network_calculations_total.labels(method=method).inc()

            with self._stats_lock:
                self._in_memory_stats["network_calculations"] += 1

        except Exception as e:
            logger.error(
                "Failed to record network calculation: %s", e, exc_info=True
            )

    def update_data_coverage(self, ratio: float) -> None:
        """
        Update the data coverage ratio gauge.

        Args:
            ratio: Coverage ratio (0.0 to 1.0)
        """
        try:
            if ratio is not None and 0 <= ratio <= 1:
                self.data_coverage_ratio.set(ratio)
        except Exception as e:
            logger.error(
                "Failed to update data coverage: %s", e, exc_info=True
            )

    # ======================================================================
    # Stats and reset
    # ======================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns:
            Dictionary with metrics summary.
        """
        with self._stats_lock:
            stats = self._in_memory_stats.copy()

        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        stats["uptime_seconds"] = uptime
        stats["prometheus_available"] = PROMETHEUS_AVAILABLE
        stats["agent_id"] = "GL-MRV-S3-014"
        stats["version"] = "1.0.0"

        if uptime > 0:
            hours = uptime / 3600
            stats["rates"] = {
                "calculations_per_hour": (
                    stats["calculations"] / hours if hours > 0 else 0
                ),
                "units_per_hour": (
                    stats["units_processed"] / hours if hours > 0 else 0
                ),
            }

        return stats

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        WARNING: Not safe for concurrent use.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    "calculations": 0,
                    "emissions_kg_co2e": 0.0,
                    "units_processed": 0,
                    "excluded_units": 0,
                    "factor_selections": 0,
                    "compliance_checks": 0,
                    "batch_jobs": 0,
                    "network_calculations": 0,
                    "errors": 0,
                }
                self._start_time = datetime.utcnow()
        except Exception as e:
            logger.error("Failed to reset stats: %s", e, exc_info=True)

    # ======================================================================
    # Private helpers
    # ======================================================================

    @staticmethod
    def _validate_enum_value(
        value: Optional[str],
        enum_class: type,
        default: str,
    ) -> str:
        """
        Validate a string value against an Enum class.

        Args:
            value: The string value to validate
            enum_class: The Enum class to validate against
            default: The default value if validation fails

        Returns:
            Validated string value or default
        """
        if value is None:
            return default

        valid_values = [m.value for m in enum_class]
        if value not in valid_values:
            logger.warning(
                "Invalid %s value '%s', using default '%s'",
                enum_class.__name__, value, default,
            )
            return default

        return value


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def get_metrics() -> FranchisesMetrics:
    """
    Get the singleton FranchisesMetrics instance.

    Returns:
        FranchisesMetrics singleton instance
    """
    return FranchisesMetrics()


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.
    """
    FranchisesMetrics.reset()


@contextmanager
def track_calculation(
    method: str = "franchise_specific",
    franchise_type: str = "qsr_restaurant",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation gauge,
    measures wall-clock duration, and records the calculation when the
    context exits.

    Args:
        method: Calculation method
        franchise_type: Franchise type

    Yields:
        Mutable context dict.  Set ``context['co2e']`` inside the block.

    Example:
        >>> with track_calculation("franchise_specific", "hotel") as ctx:
        ...     result = calculate(unit_data)
        ...     ctx['co2e'] = float(result.total_emissions_kgco2e)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        "method": method,
        "franchise_type": franchise_type,
        "co2e": 0.0,
        "status": "success",
    }
    metrics.active_calculations.inc()
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        metrics.active_calculations.dec()
        metrics.record_calculation(
            method=context.get("method", method),
            franchise_type=context.get("franchise_type", franchise_type),
            status=context.get("status", "success"),
            duration=duration,
            co2e=context.get("co2e", 0.0),
        )


@contextmanager
def track_batch() -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch job's lifecycle.

    Yields:
        Mutable context dict.  Set ``context['size']`` to record batch size.

    Example:
        >>> with track_batch() as ctx:
        ...     results = process_batch(units)
        ...     ctx['size'] = len(units)
    """
    context: Dict[str, Any] = {
        "status": "completed",
        "size": 0,
    }
    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "failed"
        raise
    finally:
        metrics = get_metrics()
        metrics.record_batch_job(
            status=context.get("status", "completed"),
            size=context.get("size", 0),
        )


# ===========================================================================
# MODULE EXPORTS
# ===========================================================================

__all__ = [
    # Availability flag
    "PROMETHEUS_AVAILABLE",
    # Enums
    "CalculationMethodLabel",
    "FranchiseTypeLabel",
    "CalculationStatusLabel",
    "EmissionSourceLabel",
    "EFSourceLabel",
    "FrameworkLabel",
    "ComplianceStatusLabel",
    "BatchStatusLabel",
    "ExclusionReasonLabel",
    # Singleton class
    "FranchisesMetrics",
    # Convenience functions
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_calculation",
    "track_batch",
]
