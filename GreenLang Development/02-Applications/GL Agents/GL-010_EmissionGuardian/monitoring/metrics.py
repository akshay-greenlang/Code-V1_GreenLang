# -*- coding: utf-8 -*-
"""GL-010 EmissionsGuardian - Prometheus Metrics Module"""

from __future__ import annotations
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

METRICS_PREFIX = 'greenlang_emissionsguardian'
METRICS_PORT = 9010
DEFAULT_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))

class SeverityLevel(str, Enum):
    INFORMATIONAL = 'informational'
    WARNING = 'warning'
    MINOR = 'minor'
    MODERATE = 'moderate'
    MAJOR = 'major'
    CRITICAL = 'critical'

class ConfidenceLevel(str, Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    VERY_HIGH = 'very_high'

class Pollutant(str, Enum):
    NOX = 'nox'
    SO2 = 'so2'
    CO2 = 'co2'
    CO = 'co'
    PM = 'pm'
    VOC = 'voc'
    NH3 = 'nh3'

class ProcessingStage(str, Enum):
    INTAKE = 'intake'
    VALIDATION = 'validation'
    CALCULATION = 'calculation'
    COMPLIANCE_CHECK = 'compliance_check'
    REPORTING = 'reporting'
    AUDIT = 'audit'

@dataclass
class MetricLabel:
    key: str
    value: str
    def __str__(self) -> str:
        escaped_value = self.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{self.key}="{escaped_value}"'


class BaseMetric:
    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        self.name = f'{METRICS_PREFIX}_{name}'
        self.help_text = help_text
        self.label_names = label_names or []
        self._lock = threading.Lock()
        self._values: Dict[Tuple[str, ...], float] = {}
        self._created_at = datetime.utcnow()

    def _validate_labels(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        if set(labels.keys()) != set(self.label_names):
            raise ValueError(f'Label mismatch. Expected {self.label_names}, got {list(labels.keys())}')
        return tuple(labels.get(name, '') for name in self.label_names)

    def _format_labels(self, label_values: Tuple[str, ...]) -> str:
        if not label_values:
            return ''
        label_pairs = [MetricLabel(name, value) for name, value in zip(self.label_names, label_values)]
        return '{' + ','.join(str(lp) for lp in label_pairs) + '}'

    def _get_type_name(self) -> str:
        raise NotImplementedError

    def expose(self) -> str:
        raise NotImplementedError


class Counter(BaseMetric):
    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        super().__init__(name, help_text, label_names)
        if not label_names:
            self._values[()] = 0.0

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        if value < 0:
            raise ValueError('Counter can only be incremented')
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self._values[label_key] = current + value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            return self._values.get(label_key, 0.0)

    def _get_type_name(self) -> str:
        return 'counter'

    def expose(self) -> str:
        lines = [f'# HELP {self.name} {self.help_text}', f'# TYPE {self.name} {self._get_type_name()}']
        with self._lock:
            for label_values, value in sorted(self._values.items()):
                labels_str = self._format_labels(label_values)
                lines.append(f'{self.name}{labels_str} {value}')
        return '
'.join(lines)


class Gauge(BaseMetric):
    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        super().__init__(name, help_text, label_names)
        if not label_names:
            self._values[()] = 0.0

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self._values[label_key] = current + value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self._values[label_key] = current - value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            return self._values.get(label_key, 0.0)

    def set_to_current_time(self, labels: Optional[Dict[str, str]] = None) -> None:
        self.set(time.time(), labels)

    def _get_type_name(self) -> str:
        return 'gauge'

    def expose(self) -> str:
        lines = [f'# HELP {self.name} {self.help_text}', f'# TYPE {self.name} {self._get_type_name()}']
        with self._lock:
            for label_values, value in sorted(self._values.items()):
                labels_str = self._format_labels(label_values)
                lines.append(f'{self.name}{labels_str} {value}')
        return '
'.join(lines)



class Histogram(BaseMetric):
    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None, buckets: Optional[Tuple[float, ...]] = None):
        super().__init__(name, help_text, label_names)
        self.buckets = buckets or DEFAULT_HISTOGRAM_BUCKETS
        self.buckets = tuple(sorted(set(self.buckets) | {float('inf')}))
        self._bucket_counts: Dict[Tuple[str, ...], Dict[float, int]] = {}
        self._sums: Dict[Tuple[str, ...], float] = {}
        self._counts: Dict[Tuple[str, ...], int] = {}
        if not label_names:
            self._init_buckets(())

    def _init_buckets(self, label_key: Tuple[str, ...]) -> None:
        self._bucket_counts[label_key] = {b: 0 for b in self.buckets}
        self._sums[label_key] = 0.0
        self._counts[label_key] = 0

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        label_key = self._validate_labels(labels or {}) if self.label_names else ()
        with self._lock:
            if label_key not in self._bucket_counts:
                self._init_buckets(label_key)
            self._sums[label_key] += value
            self._counts[label_key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1

    def time(self, labels: Optional[Dict[str, str]] = None) -> 'HistogramTimer':
        return HistogramTimer(self, labels)

    def _get_type_name(self) -> str:
        return 'histogram'

    def expose(self) -> str:
        lines = [f'# HELP {self.name} {self.help_text}', f'# TYPE {self.name} {self._get_type_name()}']
        with self._lock:
            for label_values in sorted(self._bucket_counts.keys()):
                base_labels = self._format_labels(label_values)
                for bucket in sorted(self.buckets):
                    bucket_value = '+Inf' if bucket == float('inf') else str(bucket)
                    if base_labels:
                        labels_str = base_labels[:-1] + f',le="{bucket_value}"' + '}'
                    else:
                        labels_str = '{le="' + bucket_value + '"}'
                    actual_cumulative = sum(self._bucket_counts[label_values].get(b, 0) for b in sorted(self.buckets) if b <= bucket)
                    lines.append(f'{self.name}_bucket{labels_str} {actual_cumulative}')
                lines.append(f'{self.name}_sum{base_labels} {self._sums[label_values]}')
                lines.append(f'{self.name}_count{base_labels} {self._counts[label_values]}')
        return '
'.join(lines)


class HistogramTimer:
    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        self.histogram = histogram
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self) -> 'HistogramTimer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration, self.labels)



class MetricsRegistry:
    _instance: Optional['MetricsRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'MetricsRegistry':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._metrics: Dict[str, BaseMetric] = {}
        self._init_metrics()
        logger.info('MetricsRegistry initialized with all EmissionsGuardian metrics')

    def _init_metrics(self) -> None:
        # Counters
        self.emissions_readings_processed = Counter(name='emissions_readings_processed_total', help_text='Total number of emissions readings processed', label_names=['pollutant', 'unit_id', 'facility_id'])
        self._metrics['emissions_readings_processed_total'] = self.emissions_readings_processed

        self.compliance_exceedances_total = Counter(name='compliance_exceedances_total', help_text='Total number of compliance exceedances detected', label_names=['severity', 'pollutant', 'unit_id'])
        self._metrics['compliance_exceedances_total'] = self.compliance_exceedances_total

        self.fugitive_detections_total = Counter(name='fugitive_detections_total', help_text='Total number of fugitive emission detections', label_names=['confidence', 'source_type', 'facility_id'])
        self._metrics['fugitive_detections_total'] = self.fugitive_detections_total

        self.validation_errors_total = Counter(name='validation_errors_total', help_text='Total number of data validation errors', label_names=['error_type', 'stage'])
        self._metrics['validation_errors_total'] = self.validation_errors_total

        self.alerts_generated = Counter(name='alerts_generated_total', help_text='Total number of alerts generated', label_names=['alert_type', 'severity', 'channel'])
        self._metrics['alerts_generated_total'] = self.alerts_generated

        # Gauges
        self.cems_data_availability_percent = Gauge(name='cems_data_availability_percent', help_text='CEMS data availability percentage (0-100)', label_names=['unit_id', 'pollutant'])
        self._metrics['cems_data_availability_percent'] = self.cems_data_availability_percent

        self.compliance_score = Gauge(name='compliance_score', help_text='Current compliance score (0-100)', label_names=['facility_id', 'unit_id'])
        self._metrics['compliance_score'] = self.compliance_score

        self.allowance_position_mtco2e = Gauge(name='allowance_position_mtco2e', help_text='Current carbon allowance position in metric tons CO2 equivalent', label_names=['market', 'facility_id'])
        self._metrics['allowance_position_mtco2e'] = self.allowance_position_mtco2e

        self.active_exceedances = Gauge(name='active_exceedances', help_text='Number of currently active exceedances', label_names=['severity', 'facility_id'])
        self._metrics['active_exceedances'] = self.active_exceedances

        self.pipeline_queue_size = Gauge(name='pipeline_queue_size', help_text='Number of items in processing pipeline queue', label_names=['stage'])
        self._metrics['pipeline_queue_size'] = self.pipeline_queue_size

        # Histograms
        self.processing_latency_seconds = Histogram(name='processing_latency_seconds', help_text='Processing latency in seconds', label_names=['stage', 'operation'], buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
        self._metrics['processing_latency_seconds'] = self.processing_latency_seconds

        self.calculation_duration_seconds = Histogram(name='calculation_duration_seconds', help_text='Duration of emission calculations in seconds', label_names=['calculation_type', 'pollutant'], buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
        self._metrics['calculation_duration_seconds'] = self.calculation_duration_seconds

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        return self._metrics.get(name)

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    def expose(self) -> str:
        sections = []
        for metric in self._metrics.values():
            sections.append(metric.expose())
        return '

'.join(sections) + '
'
'

    def reset(self) -> None:
        self._init_metrics()



class MetricsHandler(BaseHTTPRequestHandler):
    registry: Optional[MetricsRegistry] = None

    def do_GET(self) -> None:
        if self.path == '/metrics':
            self._serve_metrics()
        elif self.path == '/':
            self._serve_index()
        else:
            self.send_error(404, 'Not Found')

    def _serve_metrics(self) -> None:
        if self.registry is None:
            self.registry = MetricsRegistry()
        try:
            content = self.registry.expose()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            logger.error(f'Error serving metrics: {e}')
            self.send_error(500, str(e))

    def _serve_index(self) -> None:
        content = '<html><body><h1>GL-010 EmissionsGuardian Metrics</h1><a href="/metrics">Metrics</a></body></html>'
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(f'Metrics HTTP: {format % args}')


class MetricsServer:
    def __init__(self, host: str = '0.0.0.0', port: int = METRICS_PORT, registry: Optional[MetricsRegistry] = None):
        self.host = host
        self.port = port
        self.registry = registry or MetricsRegistry()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        MetricsHandler.registry = self.registry
        self._server = HTTPServer((self.host, self.port), MetricsHandler)
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info(f'Metrics server started on http://{self.host}:{self.port}/metrics')

    def _serve(self) -> None:
        if self._server:
            self._server.serve_forever()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info('Metrics server stopped')

    def is_running(self) -> bool:
        return self._running


F = TypeVar('F', bound=Callable[..., Any])


def track_processing_time(histogram: Histogram, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with histogram.time(labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def count_calls(counter: Counter, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counter.inc(1.0, labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def calculate_metrics_provenance(registry: MetricsRegistry) -> str:
    content = registry.expose()
    timestamp = datetime.utcnow().isoformat()
    provenance_data = f'{timestamp}|{content}'
    return hashlib.sha256(provenance_data.encode()).hexdigest()


_default_registry: Optional[MetricsRegistry] = None
_default_server: Optional[MetricsServer] = None


def get_registry() -> MetricsRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry()
    return _default_registry


def start_metrics_server(host: str = '0.0.0.0', port: int = METRICS_PORT) -> MetricsServer:
    global _default_server
    if _default_server is None or not _default_server.is_running():
        _default_server = MetricsServer(host=host, port=port)
        _default_server.start()
    return _default_server


def stop_metrics_server() -> None:
    global _default_server
    if _default_server is not None:
        _default_server.stop()
        _default_server = None
