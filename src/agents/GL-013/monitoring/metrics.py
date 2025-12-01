"""
GL-013 PREDICTMAINT - Prometheus Metrics Collection

Comprehensive observability for predictive maintenance operations.
Provides real-time metrics for equipment health, failure predictions,
maintenance scheduling, and operational performance.

Key Features:
    - Equipment health index monitoring (0-100 scale)
    - Remaining Useful Life (RUL) tracking in hours
    - Failure probability gauges per failure mode
    - Vibration analysis metrics (ISO 10816 compliant)
    - Temperature monitoring with thermal life tracking
    - Anomaly detection counters by severity
    - Maintenance scheduling metrics
    - Cache performance monitoring
    - Integration connector status

Standards Compliance:
    - ISO 10816: Vibration zone classification
    - ISO 13373: Condition monitoring metrics
    - ISO 13381: Prognostics metrics
    - OpenMetrics: Prometheus exposition format

Example:
    >>> from gl_013.monitoring.metrics import MetricsCollector
    >>> collector = MetricsCollector()
    >>> collector.record_equipment_health("PUMP-001", "pump_centrifugal", 85.5)
    >>> collector.record_rul("PUMP-001", "pump_centrifugal", 2400.0)
    >>> summary = collector.get_metrics_summary()

Author: GL-MonitoringEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from prometheus_client import Counter, Gauge, Histogram, Info, Summary, REGISTRY
from prometheus_client.core import CollectorRegistry
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
import threading
import time
import logging
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Equipment Health Metrics
# -----------------------------------------------------------------------------

equipment_health_index = Gauge(
    'gl013_equipment_health_index',
    'Current equipment health index on 0-100 scale. Higher values indicate better health.',
    ['equipment_id', 'equipment_type']
)

equipment_rul_hours = Gauge(
    'gl013_equipment_rul_hours',
    'Estimated remaining useful life in hours based on reliability models.',
    ['equipment_id', 'equipment_type']
)

equipment_rul_days = Gauge(
    'gl013_equipment_rul_days',
    'Estimated remaining useful life in days for dashboard display.',
    ['equipment_id', 'equipment_type']
)

equipment_reliability = Gauge(
    'gl013_equipment_reliability',
    'Current reliability R(t) value between 0 and 1.',
    ['equipment_id', 'equipment_type', 'model']
)

failure_probability = Gauge(
    'gl013_failure_probability',
    'Current failure probability between 0 and 1 for specific failure modes.',
    ['equipment_id', 'failure_mode']
)

failure_probability_30d = Gauge(
    'gl013_failure_probability_30d',
    'Probability of failure within next 30 days.',
    ['equipment_id', 'equipment_type']
)

failure_probability_90d = Gauge(
    'gl013_failure_probability_90d',
    'Probability of failure within next 90 days.',
    ['equipment_id', 'equipment_type']
)

equipment_operating_hours = Gauge(
    'gl013_equipment_operating_hours',
    'Total operating hours for equipment.',
    ['equipment_id', 'equipment_type']
)

equipment_age_days = Gauge(
    'gl013_equipment_age_days',
    'Equipment age in days since installation.',
    ['equipment_id', 'equipment_type']
)

# -----------------------------------------------------------------------------
# Vibration Metrics (ISO 10816 Compliant)
# -----------------------------------------------------------------------------

vibration_velocity_mm_s = Gauge(
    'gl013_vibration_velocity_mm_s',
    'Vibration velocity in mm/s RMS per ISO 10816.',
    ['equipment_id', 'measurement_point', 'axis']
)

vibration_zone = Gauge(
    'gl013_vibration_zone',
    'ISO 10816 vibration zone: 1=A (Good), 2=B (Acceptable), 3=C (Alert), 4=D (Danger).',
    ['equipment_id', 'machine_class']
)

vibration_zone_margin = Gauge(
    'gl013_vibration_zone_margin_mm_s',
    'Margin to next vibration zone threshold in mm/s.',
    ['equipment_id', 'machine_class']
)

vibration_acceleration_g = Gauge(
    'gl013_vibration_acceleration_g',
    'Vibration acceleration in g RMS for high-frequency analysis.',
    ['equipment_id', 'measurement_point', 'axis']
)

vibration_displacement_um = Gauge(
    'gl013_vibration_displacement_um',
    'Vibration displacement in micrometers peak-to-peak.',
    ['equipment_id', 'measurement_point', 'axis']
)

bearing_fault_frequency_energy = Gauge(
    'gl013_bearing_fault_frequency_energy',
    'Energy at bearing fault frequencies (BPFO, BPFI, BSF, FTF).',
    ['equipment_id', 'bearing_id', 'fault_type']
)

vibration_spectrum_dominant_freq = Gauge(
    'gl013_vibration_spectrum_dominant_freq_hz',
    'Dominant frequency in vibration spectrum.',
    ['equipment_id', 'measurement_point']
)

vibration_trend_rate = Gauge(
    'gl013_vibration_trend_rate',
    'Rate of vibration change in mm/s per day.',
    ['equipment_id', 'measurement_point']
)

# -----------------------------------------------------------------------------
# Temperature Metrics
# -----------------------------------------------------------------------------

temperature_celsius = Gauge(
    'gl013_temperature_celsius',
    'Equipment temperature in degrees Celsius.',
    ['equipment_id', 'sensor_location']
)

temperature_delta_ambient = Gauge(
    'gl013_temperature_delta_ambient_celsius',
    'Temperature rise above ambient in Celsius.',
    ['equipment_id', 'sensor_location']
)

thermal_life_consumed_percent = Gauge(
    'gl013_thermal_life_consumed_percent',
    'Percentage of thermal life consumed based on Arrhenius model.',
    ['equipment_id', 'insulation_class']
)

thermal_life_remaining_hours = Gauge(
    'gl013_thermal_life_remaining_hours',
    'Estimated remaining thermal life in hours.',
    ['equipment_id', 'insulation_class']
)

temperature_exceedance_count = Counter(
    'gl013_temperature_exceedance_total',
    'Count of temperature threshold exceedances.',
    ['equipment_id', 'threshold_type']
)

# -----------------------------------------------------------------------------
# Operation Latency Metrics
# -----------------------------------------------------------------------------

operation_latency_seconds = Histogram(
    'gl013_operation_latency_seconds',
    'Latency distribution for predictive maintenance operations.',
    ['operation_type', 'equipment_type'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

calculation_latency_seconds = Histogram(
    'gl013_calculation_latency_seconds',
    'Latency distribution for specific calculations.',
    ['calculation_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]
)

operations_total = Counter(
    'gl013_operations_total',
    'Total number of predictive maintenance operations executed.',
    ['operation_type', 'status']
)

predictions_total = Counter(
    'gl013_predictions_total',
    'Total number of predictions made.',
    ['prediction_type', 'equipment_type']
)

# -----------------------------------------------------------------------------
# Maintenance Metrics
# -----------------------------------------------------------------------------

maintenance_tasks_scheduled = Counter(
    'gl013_maintenance_tasks_scheduled_total',
    'Total maintenance tasks scheduled by the system.',
    ['maintenance_type', 'urgency', 'equipment_type']
)

maintenance_tasks_active = Gauge(
    'gl013_maintenance_tasks_active',
    'Currently active maintenance tasks.',
    ['maintenance_type', 'urgency']
)

maintenance_tasks_overdue = Gauge(
    'gl013_maintenance_tasks_overdue',
    'Number of overdue maintenance tasks.',
    ['maintenance_type']
)

maintenance_cost_savings_usd = Counter(
    'gl013_maintenance_cost_savings_usd_total',
    'Estimated cumulative cost savings from predictive maintenance in USD.',
    ['equipment_type', 'savings_type']
)

maintenance_downtime_prevented_hours = Counter(
    'gl013_maintenance_downtime_prevented_hours_total',
    'Total downtime hours prevented through predictive maintenance.',
    ['equipment_type']
)

maintenance_lead_time_days = Gauge(
    'gl013_maintenance_lead_time_days',
    'Days until next scheduled maintenance.',
    ['equipment_id', 'maintenance_type']
)

spare_parts_required = Gauge(
    'gl013_spare_parts_required',
    'Number of spare parts required for upcoming maintenance.',
    ['part_category', 'criticality']
)

spare_parts_available = Gauge(
    'gl013_spare_parts_available',
    'Number of spare parts available in inventory.',
    ['part_category']
)

# -----------------------------------------------------------------------------
# Anomaly Detection Metrics
# -----------------------------------------------------------------------------

anomalies_detected_total = Counter(
    'gl013_anomalies_detected_total',
    'Total anomalies detected by the system.',
    ['equipment_id', 'anomaly_type', 'severity']
)

anomalies_active = Gauge(
    'gl013_anomalies_active',
    'Currently active unresolved anomalies.',
    ['anomaly_type', 'severity']
)

anomaly_score = Gauge(
    'gl013_anomaly_score',
    'Current anomaly score (0-1) for equipment.',
    ['equipment_id', 'detection_method']
)

anomaly_detection_latency_seconds = Histogram(
    'gl013_anomaly_detection_latency_seconds',
    'Latency for anomaly detection operations.',
    ['detection_method'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# -----------------------------------------------------------------------------
# Model Performance Metrics
# -----------------------------------------------------------------------------

model_accuracy = Gauge(
    'gl013_model_accuracy',
    'Model accuracy metric (0-1).',
    ['model_type', 'equipment_type']
)

model_precision = Gauge(
    'gl013_model_precision',
    'Model precision metric (0-1).',
    ['model_type', 'equipment_type']
)

model_recall = Gauge(
    'gl013_model_recall',
    'Model recall metric (0-1).',
    ['model_type', 'equipment_type']
)

model_f1_score = Gauge(
    'gl013_model_f1_score',
    'Model F1 score (0-1).',
    ['model_type', 'equipment_type']
)

model_prediction_confidence = Histogram(
    'gl013_model_prediction_confidence',
    'Distribution of prediction confidence scores.',
    ['model_type'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
)

model_inference_count = Counter(
    'gl013_model_inference_total',
    'Total model inference count.',
    ['model_type', 'equipment_type']
)

# -----------------------------------------------------------------------------
# Cache Metrics
# -----------------------------------------------------------------------------

cache_hits_total = Counter(
    'gl013_cache_hits_total',
    'Total cache hits.',
    ['cache_name']
)

cache_misses_total = Counter(
    'gl013_cache_misses_total',
    'Total cache misses.',
    ['cache_name']
)

cache_hit_rate = Gauge(
    'gl013_cache_hit_rate',
    'Current cache hit rate percentage.',
    ['cache_name']
)

cache_size_bytes = Gauge(
    'gl013_cache_size_bytes',
    'Current cache size in bytes.',
    ['cache_name']
)

cache_entries_count = Gauge(
    'gl013_cache_entries_count',
    'Current number of entries in cache.',
    ['cache_name']
)

cache_evictions_total = Counter(
    'gl013_cache_evictions_total',
    'Total cache evictions.',
    ['cache_name', 'reason']
)

# -----------------------------------------------------------------------------
# Integration Metrics
# -----------------------------------------------------------------------------

connector_status = Gauge(
    'gl013_connector_status',
    'Connector connection status: 0=disconnected, 1=connected, 2=degraded.',
    ['connector_type', 'endpoint']
)

connector_latency_seconds = Histogram(
    'gl013_connector_latency_seconds',
    'Connector operation latency distribution.',
    ['connector_type', 'operation'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

connector_requests_total = Counter(
    'gl013_connector_requests_total',
    'Total connector requests.',
    ['connector_type', 'status']
)

connector_errors_total = Counter(
    'gl013_connector_errors_total',
    'Total connector errors.',
    ['connector_type', 'error_type']
)

data_sync_lag_seconds = Gauge(
    'gl013_data_sync_lag_seconds',
    'Data synchronization lag in seconds.',
    ['connector_type', 'data_source']
)

# -----------------------------------------------------------------------------
# Provenance Metrics
# -----------------------------------------------------------------------------

provenance_records_total = Counter(
    'gl013_provenance_records_total',
    'Total provenance records created.',
    ['calculation_type']
)

provenance_storage_bytes = Gauge(
    'gl013_provenance_storage_bytes',
    'Total provenance storage usage in bytes.'
)

provenance_validation_failures = Counter(
    'gl013_provenance_validation_failures_total',
    'Total provenance validation failures.',
    ['failure_type']
)

# -----------------------------------------------------------------------------
# System Info Metric
# -----------------------------------------------------------------------------

system_info = Info(
    'gl013_system',
    'GL-013 PREDICTMAINT system information'
)


# =============================================================================
# ENUMS
# =============================================================================

class OperationType(str, Enum):
    """Operation types for metrics tracking."""
    RUL_CALCULATION = "rul_calculation"
    FAILURE_PREDICTION = "failure_prediction"
    VIBRATION_ANALYSIS = "vibration_analysis"
    THERMAL_ANALYSIS = "thermal_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    HEALTH_INDEX_CALCULATION = "health_index_calculation"
    SPARE_PARTS_FORECAST = "spare_parts_forecast"
    DATA_INGESTION = "data_ingestion"
    REPORT_GENERATION = "report_generation"


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MaintenanceUrgency(str, Enum):
    """Maintenance task urgency levels."""
    ROUTINE = "routine"
    PLANNED = "planned"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class ConnectorStatus(int, Enum):
    """Connector status codes."""
    DISCONNECTED = 0
    CONNECTED = 1
    DEGRADED = 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetricSnapshot:
    """Snapshot of key metrics at a point in time."""
    timestamp: datetime
    equipment_count: int
    healthy_equipment_count: int
    critical_equipment_count: int
    active_anomalies: int
    pending_maintenance_tasks: int
    total_rul_hours: float
    average_health_index: float
    cache_hit_rate: float
    total_predictions: int
    total_cost_savings: float


@dataclass
class EquipmentMetrics:
    """Comprehensive metrics for a single equipment."""
    equipment_id: str
    equipment_type: str
    health_index: float
    rul_hours: float
    reliability: float
    failure_probability_30d: float
    vibration_velocity: Optional[float] = None
    vibration_zone: Optional[int] = None
    temperature: Optional[float] = None
    anomaly_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


# =============================================================================
# METRICS COLLECTOR CLASS
# =============================================================================

class MetricsCollector:
    """
    Centralized metrics collection for GL-013 PREDICTMAINT.

    This class provides a unified interface for recording all predictive
    maintenance metrics. It handles metric aggregation, rate limiting,
    and provides helper methods for common metric patterns.

    Thread-safe implementation for concurrent access.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_equipment_health("PUMP-001", "pump_centrifugal", 85.5)
        >>> with collector.timed_operation(OperationType.RUL_CALCULATION):
        ...     # perform RUL calculation
        ...     pass
        >>> summary = collector.get_metrics_summary()
    """

    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> 'MetricsCollector':
        """Singleton pattern for metrics collector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        enable_system_info: bool = True,
        rate_limit_seconds: float = 0.1
    ):
        """
        Initialize MetricsCollector.

        Args:
            enable_system_info: Whether to set system info metrics
            rate_limit_seconds: Minimum interval between identical metric updates
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self._rate_limit = rate_limit_seconds
        self._last_updates: Dict[str, datetime] = {}
        self._update_lock = threading.Lock()
        self._cache_stats: Dict[str, Dict[str, int]] = {}
        self._equipment_cache: Dict[str, EquipmentMetrics] = {}

        if enable_system_info:
            self._set_system_info()

        logger.info("MetricsCollector initialized")

    def _set_system_info(self) -> None:
        """Set system information metric."""
        system_info.info({
            'version': '1.0.0',
            'agent_id': 'GL-013',
            'codename': 'PREDICTMAINT',
            'environment': 'production',
            'standards': 'ISO_10816,ISO_13373,ISO_13381',
        })

    def _should_update(self, metric_key: str) -> bool:
        """Check if metric should be updated based on rate limiting."""
        now = datetime.now()
        with self._update_lock:
            last_update = self._last_updates.get(metric_key)
            if last_update is None or (now - last_update).total_seconds() >= self._rate_limit:
                self._last_updates[metric_key] = now
                return True
            return False

    # -------------------------------------------------------------------------
    # Equipment Health Methods
    # -------------------------------------------------------------------------

    def record_equipment_health(
        self,
        equipment_id: str,
        equipment_type: str,
        health_index: float
    ) -> None:
        """
        Record equipment health index.

        Args:
            equipment_id: Unique equipment identifier
            equipment_type: Equipment type classification
            health_index: Health index value (0-100)
        """
        if not 0 <= health_index <= 100:
            logger.warning(f"Health index {health_index} out of range [0-100] for {equipment_id}")
            health_index = max(0, min(100, health_index))

        equipment_health_index.labels(
            equipment_id=equipment_id,
            equipment_type=equipment_type
        ).set(health_index)

        logger.debug(f"Recorded health index {health_index} for {equipment_id}")

    def record_rul(
        self,
        equipment_id: str,
        equipment_type: str,
        rul_hours: float,
        reliability: Optional[float] = None,
        model: str = "weibull"
    ) -> None:
        """
        Record Remaining Useful Life metrics.

        Args:
            equipment_id: Unique equipment identifier
            equipment_type: Equipment type classification
            rul_hours: RUL in hours
            reliability: Current reliability R(t) value
            model: Reliability model used
        """
        equipment_rul_hours.labels(
            equipment_id=equipment_id,
            equipment_type=equipment_type
        ).set(rul_hours)

        equipment_rul_days.labels(
            equipment_id=equipment_id,
            equipment_type=equipment_type
        ).set(rul_hours / 24.0)

        if reliability is not None:
            equipment_reliability.labels(
                equipment_id=equipment_id,
                equipment_type=equipment_type,
                model=model
            ).set(reliability)

        logger.debug(f"Recorded RUL {rul_hours:.1f}h for {equipment_id}")

    def record_failure_probability(
        self,
        equipment_id: str,
        equipment_type: str,
        probability_30d: float,
        probability_90d: Optional[float] = None,
        failure_mode: Optional[str] = None,
        mode_probability: Optional[float] = None
    ) -> None:
        """
        Record failure probability metrics.

        Args:
            equipment_id: Unique equipment identifier
            equipment_type: Equipment type classification
            probability_30d: 30-day failure probability
            probability_90d: 90-day failure probability
            failure_mode: Specific failure mode
            mode_probability: Probability for specific failure mode
        """
        failure_probability_30d.labels(
            equipment_id=equipment_id,
            equipment_type=equipment_type
        ).set(probability_30d)

        if probability_90d is not None:
            failure_probability_90d.labels(
                equipment_id=equipment_id,
                equipment_type=equipment_type
            ).set(probability_90d)

        if failure_mode and mode_probability is not None:
            failure_probability.labels(
                equipment_id=equipment_id,
                failure_mode=failure_mode
            ).set(mode_probability)

    def record_operating_hours(
        self,
        equipment_id: str,
        equipment_type: str,
        operating_hours: float,
        age_days: Optional[float] = None
    ) -> None:
        """
        Record equipment operating hours and age.

        Args:
            equipment_id: Unique equipment identifier
            equipment_type: Equipment type classification
            operating_hours: Total operating hours
            age_days: Equipment age in days
        """
        equipment_operating_hours.labels(
            equipment_id=equipment_id,
            equipment_type=equipment_type
        ).set(operating_hours)

        if age_days is not None:
            equipment_age_days.labels(
                equipment_id=equipment_id,
                equipment_type=equipment_type
            ).set(age_days)

    # -------------------------------------------------------------------------
    # Vibration Analysis Methods
    # -------------------------------------------------------------------------

    def record_vibration_velocity(
        self,
        equipment_id: str,
        measurement_point: str,
        velocity_mm_s: float,
        axis: str = "radial"
    ) -> None:
        """
        Record vibration velocity measurement.

        Args:
            equipment_id: Unique equipment identifier
            measurement_point: Sensor location (e.g., "DE", "NDE")
            velocity_mm_s: Velocity in mm/s RMS
            axis: Measurement axis (radial, axial, tangential)
        """
        vibration_velocity_mm_s.labels(
            equipment_id=equipment_id,
            measurement_point=measurement_point,
            axis=axis
        ).set(velocity_mm_s)

    def record_vibration_zone(
        self,
        equipment_id: str,
        machine_class: str,
        zone: int,
        margin_mm_s: float
    ) -> None:
        """
        Record ISO 10816 vibration zone.

        Args:
            equipment_id: Unique equipment identifier
            machine_class: ISO 10816 machine class (I, II, III, IV)
            zone: Zone number (1=A, 2=B, 3=C, 4=D)
            margin_mm_s: Margin to next zone in mm/s
        """
        vibration_zone.labels(
            equipment_id=equipment_id,
            machine_class=machine_class
        ).set(zone)

        vibration_zone_margin.labels(
            equipment_id=equipment_id,
            machine_class=machine_class
        ).set(margin_mm_s)

    def record_bearing_fault_energy(
        self,
        equipment_id: str,
        bearing_id: str,
        bpfo_energy: float,
        bpfi_energy: float,
        bsf_energy: float,
        ftf_energy: float
    ) -> None:
        """
        Record bearing fault frequency energies.

        Args:
            equipment_id: Unique equipment identifier
            bearing_id: Bearing identifier
            bpfo_energy: Ball Pass Frequency Outer energy
            bpfi_energy: Ball Pass Frequency Inner energy
            bsf_energy: Ball Spin Frequency energy
            ftf_energy: Fundamental Train Frequency energy
        """
        bearing_fault_frequency_energy.labels(
            equipment_id=equipment_id,
            bearing_id=bearing_id,
            fault_type="BPFO"
        ).set(bpfo_energy)

        bearing_fault_frequency_energy.labels(
            equipment_id=equipment_id,
            bearing_id=bearing_id,
            fault_type="BPFI"
        ).set(bpfi_energy)

        bearing_fault_frequency_energy.labels(
            equipment_id=equipment_id,
            bearing_id=bearing_id,
            fault_type="BSF"
        ).set(bsf_energy)

        bearing_fault_frequency_energy.labels(
            equipment_id=equipment_id,
            bearing_id=bearing_id,
            fault_type="FTF"
        ).set(ftf_energy)

    def record_vibration_spectrum(
        self,
        equipment_id: str,
        measurement_point: str,
        dominant_freq_hz: float,
        trend_rate: float
    ) -> None:
        """
        Record vibration spectrum analysis results.

        Args:
            equipment_id: Unique equipment identifier
            measurement_point: Sensor location
            dominant_freq_hz: Dominant frequency in Hz
            trend_rate: Trend rate in mm/s per day
        """
        vibration_spectrum_dominant_freq.labels(
            equipment_id=equipment_id,
            measurement_point=measurement_point
        ).set(dominant_freq_hz)

        vibration_trend_rate.labels(
            equipment_id=equipment_id,
            measurement_point=measurement_point
        ).set(trend_rate)

    # -------------------------------------------------------------------------
    # Temperature Methods
    # -------------------------------------------------------------------------

    def record_temperature(
        self,
        equipment_id: str,
        sensor_location: str,
        temperature_c: float,
        ambient_temperature_c: Optional[float] = None
    ) -> None:
        """
        Record temperature measurement.

        Args:
            equipment_id: Unique equipment identifier
            sensor_location: Sensor location (e.g., "winding", "bearing")
            temperature_c: Temperature in Celsius
            ambient_temperature_c: Ambient temperature for delta calculation
        """
        temperature_celsius.labels(
            equipment_id=equipment_id,
            sensor_location=sensor_location
        ).set(temperature_c)

        if ambient_temperature_c is not None:
            delta = temperature_c - ambient_temperature_c
            temperature_delta_ambient.labels(
                equipment_id=equipment_id,
                sensor_location=sensor_location
            ).set(delta)

    def record_thermal_life(
        self,
        equipment_id: str,
        insulation_class: str,
        life_consumed_percent: float,
        life_remaining_hours: float
    ) -> None:
        """
        Record thermal life consumption.

        Args:
            equipment_id: Unique equipment identifier
            insulation_class: Insulation class (A, B, F, H)
            life_consumed_percent: Percentage of thermal life consumed
            life_remaining_hours: Remaining thermal life in hours
        """
        thermal_life_consumed_percent.labels(
            equipment_id=equipment_id,
            insulation_class=insulation_class
        ).set(life_consumed_percent)

        thermal_life_remaining_hours.labels(
            equipment_id=equipment_id,
            insulation_class=insulation_class
        ).set(life_remaining_hours)

    def record_temperature_exceedance(
        self,
        equipment_id: str,
        threshold_type: str
    ) -> None:
        """
        Record temperature threshold exceedance.

        Args:
            equipment_id: Unique equipment identifier
            threshold_type: Type of threshold exceeded (warning, alarm, trip)
        """
        temperature_exceedance_count.labels(
            equipment_id=equipment_id,
            threshold_type=threshold_type
        ).inc()

    # -------------------------------------------------------------------------
    # Operation Timing Methods
    # -------------------------------------------------------------------------

    def record_operation(
        self,
        operation_type: Union[OperationType, str],
        duration_seconds: float,
        success: bool,
        equipment_type: str = "all"
    ) -> None:
        """
        Record operation execution.

        Args:
            operation_type: Type of operation
            duration_seconds: Operation duration in seconds
            success: Whether operation succeeded
            equipment_type: Equipment type involved
        """
        op_type = operation_type.value if isinstance(operation_type, OperationType) else operation_type
        status = "success" if success else "failure"

        operation_latency_seconds.labels(
            operation_type=op_type,
            equipment_type=equipment_type
        ).observe(duration_seconds)

        operations_total.labels(
            operation_type=op_type,
            status=status
        ).inc()

    def record_calculation_latency(
        self,
        calculation_type: str,
        duration_seconds: float
    ) -> None:
        """
        Record calculation latency.

        Args:
            calculation_type: Type of calculation
            duration_seconds: Calculation duration in seconds
        """
        calculation_latency_seconds.labels(
            calculation_type=calculation_type
        ).observe(duration_seconds)

    def record_prediction(
        self,
        prediction_type: str,
        equipment_type: str
    ) -> None:
        """
        Record prediction made.

        Args:
            prediction_type: Type of prediction
            equipment_type: Equipment type
        """
        predictions_total.labels(
            prediction_type=prediction_type,
            equipment_type=equipment_type
        ).inc()

    class TimedOperation:
        """Context manager for timing operations."""

        def __init__(
            self,
            collector: 'MetricsCollector',
            operation_type: Union[OperationType, str],
            equipment_type: str = "all"
        ):
            self.collector = collector
            self.operation_type = operation_type
            self.equipment_type = equipment_type
            self.start_time: Optional[float] = None
            self.success = True

        def __enter__(self) -> 'MetricsCollector.TimedOperation':
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            duration = time.perf_counter() - self.start_time
            self.success = exc_type is None
            self.collector.record_operation(
                self.operation_type,
                duration,
                self.success,
                self.equipment_type
            )
            return False

        def set_failure(self) -> None:
            """Mark operation as failed."""
            self.success = False

    def timed_operation(
        self,
        operation_type: Union[OperationType, str],
        equipment_type: str = "all"
    ) -> TimedOperation:
        """
        Create timed operation context manager.

        Args:
            operation_type: Type of operation to time
            equipment_type: Equipment type involved

        Returns:
            TimedOperation context manager

        Example:
            >>> with collector.timed_operation(OperationType.RUL_CALCULATION):
            ...     result = calculate_rul(equipment_id)
        """
        return self.TimedOperation(self, operation_type, equipment_type)

    # -------------------------------------------------------------------------
    # Decorator for timing functions
    # -------------------------------------------------------------------------

    F = TypeVar('F', bound=Callable[..., Any])

    def timed(
        self,
        operation_type: Union[OperationType, str],
        equipment_type: str = "all"
    ) -> Callable[[F], F]:
        """
        Decorator to time function execution.

        Args:
            operation_type: Type of operation
            equipment_type: Equipment type

        Returns:
            Decorated function

        Example:
            >>> @collector.timed(OperationType.RUL_CALCULATION)
            ... def calculate_rul(equipment_id):
            ...     pass
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.timed_operation(operation_type, equipment_type):
                    return func(*args, **kwargs)
            return wrapper  # type: ignore
        return decorator

    # -------------------------------------------------------------------------
    # Anomaly Methods
    # -------------------------------------------------------------------------

    def record_anomaly(
        self,
        equipment_id: str,
        anomaly_type: str,
        severity: Union[AnomalySeverity, str]
    ) -> None:
        """
        Record detected anomaly.

        Args:
            equipment_id: Unique equipment identifier
            anomaly_type: Type of anomaly detected
            severity: Anomaly severity level
        """
        sev = severity.value if isinstance(severity, AnomalySeverity) else severity

        anomalies_detected_total.labels(
            equipment_id=equipment_id,
            anomaly_type=anomaly_type,
            severity=sev
        ).inc()

        logger.info(f"Anomaly recorded: {equipment_id} - {anomaly_type} ({sev})")

    def record_anomaly_score(
        self,
        equipment_id: str,
        score: float,
        detection_method: str = "isolation_forest"
    ) -> None:
        """
        Record anomaly score.

        Args:
            equipment_id: Unique equipment identifier
            score: Anomaly score (0-1)
            detection_method: Detection method used
        """
        anomaly_score.labels(
            equipment_id=equipment_id,
            detection_method=detection_method
        ).set(score)

    def record_active_anomalies(
        self,
        anomaly_type: str,
        severity: Union[AnomalySeverity, str],
        count: int
    ) -> None:
        """
        Record count of active anomalies.

        Args:
            anomaly_type: Type of anomaly
            severity: Severity level
            count: Number of active anomalies
        """
        sev = severity.value if isinstance(severity, AnomalySeverity) else severity
        anomalies_active.labels(
            anomaly_type=anomaly_type,
            severity=sev
        ).set(count)

    # -------------------------------------------------------------------------
    # Maintenance Methods
    # -------------------------------------------------------------------------

    def record_maintenance_scheduled(
        self,
        maintenance_type: str,
        urgency: Union[MaintenanceUrgency, str],
        equipment_type: str,
        cost_savings_usd: float = 0.0
    ) -> None:
        """
        Record scheduled maintenance task.

        Args:
            maintenance_type: Type of maintenance
            urgency: Task urgency level
            equipment_type: Equipment type
            cost_savings_usd: Estimated cost savings
        """
        urg = urgency.value if isinstance(urgency, MaintenanceUrgency) else urgency

        maintenance_tasks_scheduled.labels(
            maintenance_type=maintenance_type,
            urgency=urg,
            equipment_type=equipment_type
        ).inc()

        if cost_savings_usd > 0:
            maintenance_cost_savings_usd.labels(
                equipment_type=equipment_type,
                savings_type="preventive"
            ).inc(cost_savings_usd)

    def record_maintenance_lead_time(
        self,
        equipment_id: str,
        maintenance_type: str,
        lead_time_days: float
    ) -> None:
        """
        Record maintenance lead time.

        Args:
            equipment_id: Unique equipment identifier
            maintenance_type: Type of maintenance
            lead_time_days: Days until maintenance
        """
        maintenance_lead_time_days.labels(
            equipment_id=equipment_id,
            maintenance_type=maintenance_type
        ).set(lead_time_days)

    def record_downtime_prevented(
        self,
        equipment_type: str,
        hours: float
    ) -> None:
        """
        Record downtime prevented.

        Args:
            equipment_type: Equipment type
            hours: Hours of downtime prevented
        """
        maintenance_downtime_prevented_hours.labels(
            equipment_type=equipment_type
        ).inc(hours)

    def record_spare_parts(
        self,
        part_category: str,
        required: int,
        available: int,
        criticality: str = "normal"
    ) -> None:
        """
        Record spare parts metrics.

        Args:
            part_category: Part category
            required: Number of parts required
            available: Number of parts available
            criticality: Part criticality level
        """
        spare_parts_required.labels(
            part_category=part_category,
            criticality=criticality
        ).set(required)

        spare_parts_available.labels(
            part_category=part_category
        ).set(available)

    # -------------------------------------------------------------------------
    # Model Performance Methods
    # -------------------------------------------------------------------------

    def record_model_metrics(
        self,
        model_type: str,
        equipment_type: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float
    ) -> None:
        """
        Record model performance metrics.

        Args:
            model_type: Type of model
            equipment_type: Equipment type
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1: F1 score
        """
        model_accuracy.labels(model_type=model_type, equipment_type=equipment_type).set(accuracy)
        model_precision.labels(model_type=model_type, equipment_type=equipment_type).set(precision)
        model_recall.labels(model_type=model_type, equipment_type=equipment_type).set(recall)
        model_f1_score.labels(model_type=model_type, equipment_type=equipment_type).set(f1)

    def record_model_confidence(
        self,
        model_type: str,
        confidence: float
    ) -> None:
        """
        Record model prediction confidence.

        Args:
            model_type: Type of model
            confidence: Confidence score (0-1)
        """
        model_prediction_confidence.labels(model_type=model_type).observe(confidence)

    def record_model_inference(
        self,
        model_type: str,
        equipment_type: str
    ) -> None:
        """
        Record model inference.

        Args:
            model_type: Type of model
            equipment_type: Equipment type
        """
        model_inference_count.labels(
            model_type=model_type,
            equipment_type=equipment_type
        ).inc()

    # -------------------------------------------------------------------------
    # Cache Methods
    # -------------------------------------------------------------------------

    def record_cache_hit(self, cache_name: str) -> None:
        """Record cache hit."""
        cache_hits_total.labels(cache_name=cache_name).inc()
        self._update_cache_hit_rate(cache_name, hit=True)

    def record_cache_miss(self, cache_name: str) -> None:
        """Record cache miss."""
        cache_misses_total.labels(cache_name=cache_name).inc()
        self._update_cache_hit_rate(cache_name, hit=False)

    def _update_cache_hit_rate(self, cache_name: str, hit: bool) -> None:
        """Update cache hit rate calculation."""
        if cache_name not in self._cache_stats:
            self._cache_stats[cache_name] = {"hits": 0, "total": 0}

        self._cache_stats[cache_name]["total"] += 1
        if hit:
            self._cache_stats[cache_name]["hits"] += 1

        stats = self._cache_stats[cache_name]
        rate = (stats["hits"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        cache_hit_rate.labels(cache_name=cache_name).set(rate)

    def record_cache_stats(
        self,
        cache_name: str,
        size_bytes: int,
        entries_count: int
    ) -> None:
        """
        Record cache statistics.

        Args:
            cache_name: Cache name
            size_bytes: Cache size in bytes
            entries_count: Number of entries
        """
        cache_size_bytes.labels(cache_name=cache_name).set(size_bytes)
        cache_entries_count.labels(cache_name=cache_name).set(entries_count)

    def record_cache_eviction(
        self,
        cache_name: str,
        reason: str = "lru"
    ) -> None:
        """
        Record cache eviction.

        Args:
            cache_name: Cache name
            reason: Eviction reason (lru, ttl, memory)
        """
        cache_evictions_total.labels(
            cache_name=cache_name,
            reason=reason
        ).inc()

    # -------------------------------------------------------------------------
    # Connector Methods
    # -------------------------------------------------------------------------

    def record_connector_status(
        self,
        connector_type: str,
        endpoint: str,
        status: Union[ConnectorStatus, int]
    ) -> None:
        """
        Record connector status.

        Args:
            connector_type: Type of connector
            endpoint: Endpoint address
            status: Connection status
        """
        status_val = status.value if isinstance(status, ConnectorStatus) else status
        connector_status.labels(
            connector_type=connector_type,
            endpoint=endpoint
        ).set(status_val)

    def record_connector_request(
        self,
        connector_type: str,
        operation: str,
        duration_seconds: float,
        success: bool
    ) -> None:
        """
        Record connector request.

        Args:
            connector_type: Type of connector
            operation: Operation performed
            duration_seconds: Request duration
            success: Whether request succeeded
        """
        connector_latency_seconds.labels(
            connector_type=connector_type,
            operation=operation
        ).observe(duration_seconds)

        connector_requests_total.labels(
            connector_type=connector_type,
            status="success" if success else "failure"
        ).inc()

    def record_connector_error(
        self,
        connector_type: str,
        error_type: str
    ) -> None:
        """
        Record connector error.

        Args:
            connector_type: Type of connector
            error_type: Error type
        """
        connector_errors_total.labels(
            connector_type=connector_type,
            error_type=error_type
        ).inc()

    def record_data_sync_lag(
        self,
        connector_type: str,
        data_source: str,
        lag_seconds: float
    ) -> None:
        """
        Record data synchronization lag.

        Args:
            connector_type: Type of connector
            data_source: Data source identifier
            lag_seconds: Lag in seconds
        """
        data_sync_lag_seconds.labels(
            connector_type=connector_type,
            data_source=data_source
        ).set(lag_seconds)

    # -------------------------------------------------------------------------
    # Provenance Methods
    # -------------------------------------------------------------------------

    def record_provenance(
        self,
        calculation_type: str
    ) -> None:
        """
        Record provenance record creation.

        Args:
            calculation_type: Type of calculation
        """
        provenance_records_total.labels(
            calculation_type=calculation_type
        ).inc()

    def record_provenance_storage(self, bytes_used: int) -> None:
        """
        Record provenance storage usage.

        Args:
            bytes_used: Bytes used for provenance storage
        """
        provenance_storage_bytes.set(bytes_used)

    def record_provenance_validation_failure(
        self,
        failure_type: str
    ) -> None:
        """
        Record provenance validation failure.

        Args:
            failure_type: Type of validation failure
        """
        provenance_validation_failures.labels(
            failure_type=failure_type
        ).inc()

    # -------------------------------------------------------------------------
    # Summary Methods
    # -------------------------------------------------------------------------

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with metric summary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "version": "1.0.0",
                "agent_id": "GL-013",
                "codename": "PREDICTMAINT"
            },
            "cache_stats": dict(self._cache_stats),
            "collection_active": True
        }

        return summary

    def create_snapshot(self) -> MetricSnapshot:
        """
        Create a snapshot of key metrics.

        Returns:
            MetricSnapshot with current metric values
        """
        return MetricSnapshot(
            timestamp=datetime.now(),
            equipment_count=len(self._equipment_cache),
            healthy_equipment_count=sum(
                1 for e in self._equipment_cache.values()
                if e.health_index >= 70
            ),
            critical_equipment_count=sum(
                1 for e in self._equipment_cache.values()
                if e.health_index < 30
            ),
            active_anomalies=0,  # Would need to track separately
            pending_maintenance_tasks=0,  # Would need to track separately
            total_rul_hours=sum(
                e.rul_hours for e in self._equipment_cache.values()
            ),
            average_health_index=sum(
                e.health_index for e in self._equipment_cache.values()
            ) / max(1, len(self._equipment_cache)),
            cache_hit_rate=self._get_overall_cache_hit_rate(),
            total_predictions=0,  # Would need to track separately
            total_cost_savings=0.0  # Would need to track separately
        )

    def _get_overall_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total_hits = sum(s["hits"] for s in self._cache_stats.values())
        total_requests = sum(s["total"] for s in self._cache_stats.values())
        return (total_hits / total_requests * 100) if total_requests > 0 else 0.0

    def update_equipment_cache(
        self,
        equipment_id: str,
        metrics: EquipmentMetrics
    ) -> None:
        """
        Update equipment metrics cache.

        Args:
            equipment_id: Equipment identifier
            metrics: Equipment metrics
        """
        self._equipment_cache[equipment_id] = metrics

    def get_equipment_metrics(
        self,
        equipment_id: str
    ) -> Optional[EquipmentMetrics]:
        """
        Get cached equipment metrics.

        Args:
            equipment_id: Equipment identifier

        Returns:
            EquipmentMetrics or None if not found
        """
        return self._equipment_cache.get(equipment_id)

    def reset_metrics(self) -> None:
        """Reset all internal tracking state."""
        with self._update_lock:
            self._last_updates.clear()
            self._cache_stats.clear()
            self._equipment_cache.clear()
        logger.info("Metrics collector state reset")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Metrics
    "equipment_health_index",
    "equipment_rul_hours",
    "equipment_rul_days",
    "equipment_reliability",
    "failure_probability",
    "failure_probability_30d",
    "failure_probability_90d",
    "equipment_operating_hours",
    "equipment_age_days",
    "vibration_velocity_mm_s",
    "vibration_zone",
    "vibration_zone_margin",
    "vibration_acceleration_g",
    "vibration_displacement_um",
    "bearing_fault_frequency_energy",
    "vibration_spectrum_dominant_freq",
    "vibration_trend_rate",
    "temperature_celsius",
    "temperature_delta_ambient",
    "thermal_life_consumed_percent",
    "thermal_life_remaining_hours",
    "temperature_exceedance_count",
    "operation_latency_seconds",
    "calculation_latency_seconds",
    "operations_total",
    "predictions_total",
    "maintenance_tasks_scheduled",
    "maintenance_tasks_active",
    "maintenance_tasks_overdue",
    "maintenance_cost_savings_usd",
    "maintenance_downtime_prevented_hours",
    "maintenance_lead_time_days",
    "spare_parts_required",
    "spare_parts_available",
    "anomalies_detected_total",
    "anomalies_active",
    "anomaly_score",
    "anomaly_detection_latency_seconds",
    "model_accuracy",
    "model_precision",
    "model_recall",
    "model_f1_score",
    "model_prediction_confidence",
    "model_inference_count",
    "cache_hits_total",
    "cache_misses_total",
    "cache_hit_rate",
    "cache_size_bytes",
    "cache_entries_count",
    "cache_evictions_total",
    "connector_status",
    "connector_latency_seconds",
    "connector_requests_total",
    "connector_errors_total",
    "data_sync_lag_seconds",
    "provenance_records_total",
    "provenance_storage_bytes",
    "provenance_validation_failures",
    "system_info",
    # Enums
    "OperationType",
    "AnomalySeverity",
    "MaintenanceUrgency",
    "ConnectorStatus",
    # Data classes
    "MetricSnapshot",
    "EquipmentMetrics",
    # Main class
    "MetricsCollector",
]
