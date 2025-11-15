"""
Distributed Tracing with OpenTelemetry
======================================

Production-grade distributed tracing with support for:
- Jaeger backend
- DataDog APM
- New Relic integration
- Adaptive sampling
- Agent lifecycle tracing
- LLM call tracing

Author: GL-DevOpsEngineer
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from functools import wraps
import random

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Jaeger exporter
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False

# DataDog exporter
try:
    from opentelemetry.exporter.datadog import DatadogExportSpanProcessor, DatadogSpanExporter
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False

# New Relic exporter
try:
    import newrelic.agent
    NEW_RELIC_AVAILABLE = True
except ImportError:
    NEW_RELIC_AVAILABLE = False


class TracePoint(Enum):
    """Standard trace points in agent lifecycle"""
    AGENT_INIT = "agent.init"
    AGENT_STATE_CHANGE = "agent.state_change"
    MESSAGE_SEND = "message.send"
    MESSAGE_RECEIVE = "message.receive"
    LLM_CALL = "llm.call"
    TOOL_EXECUTION = "tool.execute"
    DATABASE_QUERY = "database.query"
    CACHE_OPERATION = "cache.operation"
    API_CALL = "api.call"
    TASK_EXECUTION = "task.execute"


class SamplingStrategy(Enum):
    """Sampling strategies for trace collection"""
    ALWAYS = "always"
    NEVER = "never"
    PROBABILISTIC = "probabilistic"
    ADAPTIVE = "adaptive"
    ERROR_BASED = "error_based"


class SpanContext:
    """Context holder for span attributes"""

    def __init__(self):
        self.attributes = {}
        self.events = []
        self.links = []

    def add_attribute(self, key: str, value: Any):
        """Add span attribute"""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add span event"""
        self.events.append({
            'name': name,
            'attributes': attributes or {},
            'timestamp': time.time()
        })

    def add_link(self, trace_id: str, span_id: str, attributes: Optional[Dict[str, Any]] = None):
        """Add span link"""
        self.links.append({
            'trace_id': trace_id,
            'span_id': span_id,
            'attributes': attributes or {}
        })


class AdaptiveSampler:
    """Adaptive sampling based on system load and error rates"""

    def __init__(
        self,
        base_rate: float = 0.01,
        error_rate: float = 1.0,
        slow_rate: float = 0.1,
        slow_threshold_ms: float = 1000
    ):
        """
        Initialize adaptive sampler

        Args:
            base_rate: Base sampling rate (0-1)
            error_rate: Sampling rate for errors
            slow_rate: Sampling rate for slow requests
            slow_threshold_ms: Threshold for slow requests in ms
        """
        self.base_rate = base_rate
        self.error_rate = error_rate
        self.slow_rate = slow_rate
        self.slow_threshold_ms = slow_threshold_ms

        # Adaptive parameters
        self.current_rate = base_rate
        self.request_count = 0
        self.error_count = 0
        self.window_start = time.time()

    def should_sample(
        self,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None
    ) -> bool:
        """Determine if span should be sampled"""
        # Always sample if parent is sampled
        if parent_span and parent_span.is_recording():
            return True

        # Check for error
        if attributes and attributes.get('error', False):
            return random.random() < self.error_rate

        # Check for slow request
        if attributes and attributes.get('duration_ms', 0) > self.slow_threshold_ms:
            return random.random() < self.slow_rate

        # Update adaptive rate
        self._update_adaptive_rate()

        # Probabilistic sampling
        return random.random() < self.current_rate

    def _update_adaptive_rate(self):
        """Update sampling rate based on system metrics"""
        self.request_count += 1

        # Adjust rate every 1000 requests or 60 seconds
        window_duration = time.time() - self.window_start
        if self.request_count >= 1000 or window_duration >= 60:
            error_ratio = self.error_count / max(self.request_count, 1)

            # Increase sampling if error rate is high
            if error_ratio > 0.05:  # >5% errors
                self.current_rate = min(self.base_rate * 2, 0.1)
            elif error_ratio < 0.01:  # <1% errors
                self.current_rate = self.base_rate
            else:
                self.current_rate = self.base_rate * (1 + error_ratio * 10)

            # Reset counters
            self.request_count = 0
            self.error_count = 0
            self.window_start = time.time()


class TracingManager:
    """
    Centralized distributed tracing management
    """

    def __init__(
        self,
        service_name: str = "greenlang-agents",
        backends: Optional[List[str]] = None,
        sampler: Optional[AdaptiveSampler] = None,
        jaeger_config: Optional[Dict[str, Any]] = None,
        datadog_config: Optional[Dict[str, Any]] = None,
        newrelic_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tracing manager

        Args:
            service_name: Service name for traces
            backends: List of backends ['jaeger', 'datadog', 'newrelic', 'console']
            sampler: Sampling strategy
            jaeger_config: Jaeger configuration
            datadog_config: DataDog configuration
            newrelic_config: New Relic configuration
        """
        self.service_name = service_name
        self.backends = backends or ['console']
        self.sampler = sampler or AdaptiveSampler()

        # Create resource
        self.resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry"
        })

        # Setup tracer provider
        self.tracer_provider = TracerProvider(resource=self.resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Setup exporters
        self._setup_exporters(jaeger_config, datadog_config, newrelic_config)

        # Get tracer
        self.tracer = trace.get_tracer(service_name)

        # Setup propagation
        set_global_textmap(TraceContextTextMapPropagator())

        # Instrument libraries
        RequestsInstrumentor().instrument()

        # Metrics
        self.span_count = 0
        self.error_count = 0

    def _setup_exporters(
        self,
        jaeger_config: Optional[Dict[str, Any]] = None,
        datadog_config: Optional[Dict[str, Any]] = None,
        newrelic_config: Optional[Dict[str, Any]] = None
    ):
        """Setup trace exporters for configured backends"""

        # Console exporter
        if 'console' in self.backends:
            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )

        # Jaeger exporter
        if 'jaeger' in self.backends and JAEGER_AVAILABLE:
            config = jaeger_config or {}
            jaeger_exporter = JaegerExporter(
                agent_host_name=config.get('host', 'localhost'),
                agent_port=config.get('port', 6831),
                collector_endpoint=config.get('collector_endpoint'),
                username=config.get('username'),
                password=config.get('password')
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )

        # DataDog exporter
        if 'datadog' in self.backends and DATADOG_AVAILABLE:
            config = datadog_config or {}
            datadog_exporter = DatadogSpanExporter(
                agent_url=config.get('agent_url', 'http://localhost:8126'),
                service=self.service_name,
                env=config.get('env', 'production')
            )
            self.tracer_provider.add_span_processor(
                DatadogExportSpanProcessor(datadog_exporter)
            )

        # New Relic setup
        if 'newrelic' in self.backends and NEW_RELIC_AVAILABLE:
            config = newrelic_config or {}
            if config.get('license_key'):
                newrelic.agent.initialize(
                    config_file=config.get('config_file'),
                    environment=config.get('environment', 'production'),
                    license_key=config['license_key']
                )

    @contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Create a traced span context

        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            links: Related spans

        Yields:
            Current span
        """
        # Check sampling
        if not self.sampler.should_sample(name, attributes):
            with self.tracer.start_span(
                name,
                kind=kind,
                attributes=attributes or {},
                record_exception=False,
                set_status_on_exception=False
            ) as span:
                yield span
            return

        # Create span with full recording
        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {},
            links=links or []
        ) as span:
            self.span_count += 1
            start_time = time.time()

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                self.error_count += 1
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Add duration
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", duration_ms)

    def trace_agent(
        self,
        agent_id: str,
        operation: str,
        state: Optional[str] = None
    ):
        """
        Trace agent operations

        Args:
            agent_id: Agent identifier
            operation: Operation being performed
            state: Current agent state
        """
        attributes = {
            "agent.id": agent_id,
            "agent.operation": operation
        }
        if state:
            attributes["agent.state"] = state

        return self.trace(
            f"agent.{operation}",
            kind=SpanKind.INTERNAL,
            attributes=attributes
        )

    def trace_llm_call(
        self,
        model: str,
        provider: str,
        tokens_input: int,
        tokens_output: int,
        cost: float
    ):
        """
        Trace LLM API calls

        Args:
            model: Model name
            provider: Provider name
            tokens_input: Input token count
            tokens_output: Output token count
            cost: Estimated cost
        """
        return self.trace(
            "llm.call",
            kind=SpanKind.CLIENT,
            attributes={
                "llm.model": model,
                "llm.provider": provider,
                "llm.tokens.input": tokens_input,
                "llm.tokens.output": tokens_output,
                "llm.cost": cost
            }
        )

    def trace_database(
        self,
        operation: str,
        table: str,
        database: str = "postgresql"
    ):
        """
        Trace database operations

        Args:
            operation: Database operation (SELECT, INSERT, etc.)
            table: Table name
            database: Database type
        """
        return self.trace(
            f"db.{operation.lower()}",
            kind=SpanKind.CLIENT,
            attributes={
                "db.system": database,
                "db.operation": operation,
                "db.table": table
            }
        )

    def trace_api(
        self,
        method: str,
        url: str,
        status_code: Optional[int] = None
    ):
        """
        Trace API calls

        Args:
            method: HTTP method
            url: Request URL
            status_code: Response status code
        """
        attributes = {
            "http.method": method,
            "http.url": url
        }
        if status_code:
            attributes["http.status_code"] = status_code

        return self.trace(
            f"http.{method.lower()}",
            kind=SpanKind.CLIENT,
            attributes=attributes
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        return {
            'service_name': self.service_name,
            'span_count': self.span_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.span_count, 1),
            'sampling_rate': self.sampler.current_rate,
            'backends': self.backends
        }


def trace_agent(agent_id: str, operation: str):
    """Decorator for tracing agent methods"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"agent.{operation}",
                attributes={
                    "agent.id": agent_id,
                    "agent.operation": operation
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator


def trace_llm_call(model: str, provider: str):
    """Decorator for tracing LLM calls"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "llm.call",
                kind=SpanKind.CLIENT,
                attributes={
                    "llm.model": model,
                    "llm.provider": provider
                }
            ) as span:
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    span.set_attribute("llm.duration_seconds", duration)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator


def setup_tracing(
    service_name: str = "greenlang-agents",
    backends: List[str] = None,
    jaeger_config: Optional[Dict[str, Any]] = None,
    datadog_config: Optional[Dict[str, Any]] = None,
    newrelic_config: Optional[Dict[str, Any]] = None
) -> TracingManager:
    """
    Setup global tracing configuration

    Args:
        service_name: Service name
        backends: List of backends
        jaeger_config: Jaeger config
        datadog_config: DataDog config
        newrelic_config: New Relic config

    Returns:
        Configured TracingManager instance
    """
    return TracingManager(
        service_name=service_name,
        backends=backends or ['console'],
        jaeger_config=jaeger_config,
        datadog_config=datadog_config,
        newrelic_config=newrelic_config
    )


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracer() -> TracingManager:
    """Get global tracing manager"""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = setup_tracing()
    return _tracing_manager