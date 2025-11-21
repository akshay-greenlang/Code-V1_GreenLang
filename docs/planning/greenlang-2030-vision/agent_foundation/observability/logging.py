# -*- coding: utf-8 -*-
"""
Structured JSON Logging with OpenTelemetry Integration
======================================================

Enterprise-grade logging infrastructure with support for:
- JSON structured logging
- OpenTelemetry integration
- Elasticsearch sink
- CloudWatch integration
- Correlation ID tracking
- Performance metrics in logs

Author: GL-DevOpsEngineer
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import uuid
import threading
from pathlib import Path

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Elasticsearch integration
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# CloudWatch integration
try:
    import boto3
    CLOUDWATCH_AVAILABLE = True
except ImportError:
    CLOUDWATCH_AVAILABLE = False


class LogLevel(Enum):
    """Log levels with numeric values for comparison"""
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50


class LogContext:
    """Thread-local context for correlation and tracking"""
    _local = threading.local()

    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for request tracking"""
        cls._local.correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID"""
        return getattr(cls._local, 'correlation_id', None)

    @classmethod
    def set_agent_id(cls, agent_id: str):
        """Set current agent ID"""
        cls._local.agent_id = agent_id

    @classmethod
    def get_agent_id(cls) -> Optional[str]:
        """Get current agent ID"""
        return getattr(cls._local, 'agent_id', None)

    @classmethod
    def set_context(cls, **kwargs):
        """Set multiple context values"""
        for key, value in kwargs.items():
            setattr(cls._local, key, value)

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get all context values"""
        return {
            key: value
            for key, value in cls._local.__dict__.items()
            if not key.startswith('_')
        }

    @classmethod
    def clear(cls):
        """Clear all context"""
        cls._local.__dict__.clear()


class StructuredLogger:
    """
    Production-grade structured JSON logger with multiple outputs
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        outputs: Optional[List[str]] = None,
        elasticsearch_config: Optional[Dict[str, Any]] = None,
        cloudwatch_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize structured logger

        Args:
            name: Logger name
            level: Minimum log level
            outputs: List of outputs ['console', 'file', 'elasticsearch', 'cloudwatch']
            elasticsearch_config: ES connection config
            cloudwatch_config: CloudWatch config
        """
        self.name = name
        self.level = level
        self.outputs = outputs or ['console']

        # Python logger setup
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_python_level(level))
        self.logger.handlers.clear()

        # Initialize outputs
        self._setup_console_handler()

        if 'file' in self.outputs:
            self._setup_file_handler()

        if 'elasticsearch' in self.outputs and ELASTICSEARCH_AVAILABLE:
            self._setup_elasticsearch(elasticsearch_config or {})

        if 'cloudwatch' in self.outputs and CLOUDWATCH_AVAILABLE:
            self._setup_cloudwatch(cloudwatch_config or {})

        # OpenTelemetry tracer
        self.tracer = trace.get_tracer(name)

        # Performance tracking
        self.log_count = 0
        self.error_count = 0
        self.start_time = DeterministicClock.utcnow()

    def _get_python_level(self, level: LogLevel) -> int:
        """Convert LogLevel to Python logging level"""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.FATAL: logging.CRITICAL
        }
        return mapping[level]

    def _setup_console_handler(self):
        """Setup console handler with JSON formatting"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def _setup_file_handler(self):
        """Setup file handler with rotation"""
        from logging.handlers import RotatingFileHandler

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        handler = RotatingFileHandler(
            log_dir / f"{self.name}.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def _setup_elasticsearch(self, config: Dict[str, Any]):
        """Setup Elasticsearch handler"""
        self.es_client = Elasticsearch(
            hosts=config.get('hosts', ['localhost:9200']),
            basic_auth=(
                config.get('username'),
                config.get('password')
            ) if config.get('username') else None,
            verify_certs=config.get('verify_certs', True),
            ssl_show_warn=config.get('ssl_show_warn', False)
        )
        self.es_index = config.get('index', 'greenlang-logs')
        self.es_buffer = []
        self.es_buffer_size = config.get('buffer_size', 100)

    def _setup_cloudwatch(self, config: Dict[str, Any]):
        """Setup CloudWatch handler"""
        self.cloudwatch_client = boto3.client(
            'logs',
            region_name=config.get('region', 'us-east-1')
        )
        self.cloudwatch_group = config.get('log_group', '/aws/greenlang')
        self.cloudwatch_stream = config.get(
            'log_stream',
            f"{self.name}-{DeterministicClock.utcnow().strftime('%Y%m%d')}"
        )

        # Create log group and stream if not exists
        try:
            self.cloudwatch_client.create_log_group(logGroupName=self.cloudwatch_group)
        except:
            pass

        try:
            self.cloudwatch_client.create_log_stream(
                logGroupName=self.cloudwatch_group,
                logStreamName=self.cloudwatch_stream
            )
        except:
            pass

    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Create structured log entry"""
        entry = {
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'level': level.name,
            'logger': self.name,
            'message': message,
            'correlation_id': LogContext.get_correlation_id() or str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            'agent_id': LogContext.get_agent_id(),
            'context': {
                **LogContext.get_context(),
                **(context or {})
            }
        }

        # Add performance metrics
        if performance:
            entry['performance'] = performance

        # Add error details
        if error:
            entry['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'stack_trace': traceback.format_exc()
            }

        # Add OpenTelemetry trace context
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            entry['trace'] = {
                'trace_id': format(span_context.trace_id, '032x'),
                'span_id': format(span_context.span_id, '016x')
            }

        return entry

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        if self.level.value <= LogLevel.DEBUG.value:
            self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        if self.level.value <= LogLevel.INFO.value:
            self._log(LogLevel.INFO, message, **kwargs)

    def warn(self, message: str, **kwargs):
        """Log warning message"""
        if self.level.value <= LogLevel.WARN.value:
            self._log(LogLevel.WARN, message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if self.level.value <= LogLevel.ERROR.value:
            self.error_count += 1
            self._log(LogLevel.ERROR, message, error=error, **kwargs)

    def fatal(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log fatal message"""
        if self.level.value <= LogLevel.FATAL.value:
            self.error_count += 1
            self._log(LogLevel.FATAL, message, error=error, **kwargs)

    def _log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        performance: Optional[Dict[str, float]] = None
    ):
        """Internal log method"""
        self.log_count += 1

        # Create log entry
        entry = self._create_log_entry(level, message, context, error, performance)

        # Python logger
        python_level = self._get_python_level(level)
        self.logger.log(python_level, json.dumps(entry))

        # Elasticsearch
        if 'elasticsearch' in self.outputs and ELASTICSEARCH_AVAILABLE:
            self._log_to_elasticsearch(entry)

        # CloudWatch
        if 'cloudwatch' in self.outputs and CLOUDWATCH_AVAILABLE:
            self._log_to_cloudwatch(entry)

        # Update OpenTelemetry span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(message, attributes=entry.get('context', {}))
            if error:
                span.set_status(Status(StatusCode.ERROR, str(error)))

    def _log_to_elasticsearch(self, entry: Dict[str, Any]):
        """Send log to Elasticsearch"""
        self.es_buffer.append({
            '_index': f"{self.es_index}-{DeterministicClock.utcnow().strftime('%Y.%m.%d')}",
            '_source': entry
        })

        if len(self.es_buffer) >= self.es_buffer_size:
            self._flush_elasticsearch()

    def _flush_elasticsearch(self):
        """Flush Elasticsearch buffer"""
        if self.es_buffer and ELASTICSEARCH_AVAILABLE:
            try:
                bulk(self.es_client, self.es_buffer)
                self.es_buffer.clear()
            except Exception as e:
                print(f"Failed to write to Elasticsearch: {e}")

    def _log_to_cloudwatch(self, entry: Dict[str, Any]):
        """Send log to CloudWatch"""
        if CLOUDWATCH_AVAILABLE:
            try:
                self.cloudwatch_client.put_log_events(
                    logGroupName=self.cloudwatch_group,
                    logStreamName=self.cloudwatch_stream,
                    logEvents=[
                        {
                            'timestamp': int(DeterministicClock.utcnow().timestamp() * 1000),
                            'message': json.dumps(entry)
                        }
                    ]
                )
            except Exception as e:
                print(f"Failed to write to CloudWatch: {e}")

    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations"""
        start = DeterministicClock.utcnow()
        try:
            yield
        finally:
            duration = (DeterministicClock.utcnow() - start).total_seconds()
            self.info(
                f"Operation completed: {operation}",
                performance={'duration_seconds': duration}
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        uptime = (DeterministicClock.utcnow() - self.start_time).total_seconds()
        return {
            'name': self.name,
            'log_count': self.log_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.log_count, 1),
            'uptime_seconds': uptime,
            'logs_per_second': self.log_count / max(uptime, 1)
        }


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for Python logging"""

    def format(self, record):
        """Format log record as JSON"""
        # Parse JSON if already formatted
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            try:
                return record.msg if record.msg.startswith('{') else json.dumps({'message': record.msg})
            except:
                return json.dumps({'message': str(record.msg)})
        return json.dumps({'message': str(record.getMessage())})


def setup_logging(
    app_name: str = "greenlang",
    level: str = "INFO",
    outputs: List[str] = None,
    elasticsearch_config: Optional[Dict[str, Any]] = None,
    cloudwatch_config: Optional[Dict[str, Any]] = None
) -> StructuredLogger:
    """
    Setup global logging configuration

    Args:
        app_name: Application name
        level: Log level (DEBUG, INFO, WARN, ERROR, FATAL)
        outputs: List of outputs
        elasticsearch_config: ES config
        cloudwatch_config: CloudWatch config

    Returns:
        Configured StructuredLogger instance
    """
    # Set OpenTelemetry resource
    resource = Resource.create({
        "service.name": app_name,
        "service.version": "1.0.0",
        "deployment.environment": "production"
    })

    # Configure tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))

    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)

    # Create and return logger
    return StructuredLogger(
        name=app_name,
        level=LogLevel[level],
        outputs=outputs or ['console'],
        elasticsearch_config=elasticsearch_config,
        cloudwatch_config=cloudwatch_config
    )


# Convenience functions for module-level logging
_default_logger = None


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """Get or create a logger instance"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()

    if name:
        return StructuredLogger(name=name)
    return _default_logger


def log_debug(message: str, **kwargs):
    """Log debug message"""
    get_logger().debug(message, **kwargs)


def log_info(message: str, **kwargs):
    """Log info message"""
    get_logger().info(message, **kwargs)


def log_warn(message: str, **kwargs):
    """Log warning message"""
    get_logger().warn(message, **kwargs)


def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """Log error message"""
    get_logger().error(message, error=error, **kwargs)


def log_fatal(message: str, error: Optional[Exception] = None, **kwargs):
    """Log fatal message"""
    get_logger().fatal(message, error=error, **kwargs)