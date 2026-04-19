# -*- coding: utf-8 -*-
"""
GL-VCCI Scope 3 Platform - Structured Logging Configuration

Production-grade logging with:
- JSON structured logging for machine parsing
- Correlation IDs for request tracing across distributed systems
- Carbon-specific context injection (category, tier, supplier)
- Multiple log levels and handlers
- Performance metrics in logs
- Security-conscious (no sensitive data in logs)

Features:
- Structured JSON format for log aggregation tools (ELK, Splunk, CloudWatch)
- Correlation IDs for distributed tracing across 5 agents
- Automatic context injection (user, org, category, tier, supplier)
- Performance timing decorators
- Log sanitization (removes sensitive data)
- Carbon calculation audit trails

Version: 1.0.0
Author: GreenLang VCCI Team (Monitoring & Observability)
Date: 2025-11-08
"""

import json
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import threading
import uuid
from greenlang.determinism import deterministic_uuid, DeterministicClock


# ============================================================================
# CORRELATION ID CONTEXT
# ============================================================================

class CorrelationContext:
    """Thread-safe correlation ID context for request tracing."""

    _local = threading.local()

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID."""
        return getattr(cls._local, 'correlation_id', None)

    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id

    @classmethod
    def clear_correlation_id(cls):
        """Clear correlation ID for current thread."""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')

    @classmethod
    def new_correlation_id(cls) -> str:
        """Generate and set new correlation ID."""
        correlation_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        cls.set_correlation_id(correlation_id)
        return correlation_id


# ============================================================================
# CARBON CONTEXT (VCCI-Specific)
# ============================================================================

class CarbonContext:
    """Thread-safe carbon calculation context for VCCI-specific tracking."""

    _local = threading.local()

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get current carbon context."""
        return {
            'category': getattr(cls._local, 'category', None),
            'tier': getattr(cls._local, 'tier', None),
            'supplier_id': getattr(cls._local, 'supplier_id', None),
            'entity_id': getattr(cls._local, 'entity_id', None),
            'organization_id': getattr(cls._local, 'organization_id', None),
        }

    @classmethod
    def set_context(
        cls,
        category: Optional[str] = None,
        tier: Optional[str] = None,
        supplier_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ):
        """Set carbon calculation context."""
        if category:
            cls._local.category = category
        if tier:
            cls._local.tier = tier
        if supplier_id:
            cls._local.supplier_id = supplier_id
        if entity_id:
            cls._local.entity_id = entity_id
        if organization_id:
            cls._local.organization_id = organization_id

    @classmethod
    def clear_context(cls):
        """Clear carbon context."""
        for attr in ['category', 'tier', 'supplier_id', 'entity_id', 'organization_id']:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)


# ============================================================================
# JSON FORMATTER
# ============================================================================

class VCCIJsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging with VCCI carbon context.

    Outputs logs in JSON format with:
    - Timestamp (ISO 8601)
    - Log level
    - Logger name
    - Message
    - Correlation ID
    - Carbon context (category, tier, supplier)
    - Additional context fields
    - Exception information (if present)
    """

    SENSITIVE_FIELDS = {
        'password', 'token', 'api_key', 'secret', 'authorization',
        'credit_card', 'ssn', 'api_secret', 'private_key'
    }

    def __init__(self, service_name: str = "vcci-scope3-platform"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_entry = {
            "timestamp": DeterministicClock.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name
        }

        # Add correlation ID if present
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add carbon context if present
        carbon_context = CarbonContext.get_context()
        if any(carbon_context.values()):
            log_entry["carbon_context"] = {k: v for k, v in carbon_context.items() if v}

        # Add thread and process information
        log_entry["thread"] = {
            "id": record.thread,
            "name": record.threadName
        }
        log_entry["process"] = {
            "id": record.process,
            "name": record.processName
        }

        # Add source location
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName
        }

        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            extra = self._sanitize_dict(record.extra_fields)
            log_entry["context"] = extra

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add performance metrics if present
        if hasattr(record, 'duration_ms'):
            log_entry["performance"] = {
                "duration_ms": round(record.duration_ms, 2)
            }

        # Add emissions if present (VCCI-specific)
        if hasattr(record, 'emissions_tco2'):
            log_entry["emissions"] = {
                "tco2": round(record.emissions_tco2, 4)
            }

        return json.dumps(log_entry)

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from dictionary."""
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            # Check if field name suggests sensitive data
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LoggingConfig:
    """
    Centralized logging configuration for VCCI application.

    Provides:
    - Multiple handlers (console, file, JSON)
    - Structured JSON logging
    - Log rotation
    - Different configurations for dev/prod
    """

    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        log_dir: Optional[str] = None,
        enable_json: bool = True,
        enable_console: bool = True,
        enable_file: bool = False,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Configure application logging.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (if file logging enabled)
            enable_json: Enable JSON formatted logging
            enable_console: Enable console logging
            enable_file: Enable file logging
            max_bytes: Max size of log file before rotation
            backup_count: Number of backup files to keep

        Returns:
            Configured root logger
        """
        # Convert level string to logging constant
        log_level = getattr(logging, level.upper(), logging.INFO)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers
        root_logger.handlers = []

        handlers = []

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)

            if enable_json:
                console_handler.setFormatter(VCCIJsonFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(cls.DEFAULT_FORMAT, cls.DEFAULT_DATE_FORMAT)
                )

            handlers.append(console_handler)

        # File handler with rotation
        if enable_file and log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            # Standard log file
            log_file = log_dir_path / "vcci-scope3.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                logging.Formatter(cls.DEFAULT_FORMAT, cls.DEFAULT_DATE_FORMAT)
            )
            handlers.append(file_handler)

            # JSON log file for machine parsing
            if enable_json:
                json_log_file = log_dir_path / "vcci-scope3.json.log"
                json_file_handler = logging.handlers.RotatingFileHandler(
                    json_log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                json_file_handler.setLevel(log_level)
                json_file_handler.setFormatter(VCCIJsonFormatter())
                handlers.append(json_file_handler)

            # Error-only log file
            error_log_file = log_dir_path / "vcci-scope3-errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(
                logging.Formatter(cls.DEFAULT_FORMAT, cls.DEFAULT_DATE_FORMAT)
            )
            handlers.append(error_handler)

        # Add all handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)

        # Configure third-party library logging
        cls._configure_third_party_logging(log_level)

        return root_logger

    @staticmethod
    def _configure_third_party_logging(level: int):
        """Configure logging for third-party libraries."""
        # Reduce noise from verbose libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('anthropic').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)

    @classmethod
    def get_logger(cls, name: str, extra_fields: Dict[str, Any] = None) -> logging.Logger:
        """
        Get a logger with optional extra fields.

        Args:
            name: Logger name (usually __name__)
            extra_fields: Additional context fields to include in all logs

        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)

        if extra_fields:
            # Create adapter that adds extra fields to all log records
            logger = ContextLogger(logger, extra_fields)

        return logger


# ============================================================================
# CONTEXT LOGGER ADAPTER
# ============================================================================

class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context fields to all log records."""

    def process(self, msg, kwargs):
        """Add extra fields to log record."""
        # Merge extra fields
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        kwargs['extra']['extra_fields'] = {**self.extra, **kwargs['extra'].get('extra_fields', {})}

        return msg, kwargs


# ============================================================================
# PERFORMANCE LOGGING DECORATOR
# ============================================================================

def log_performance(logger: logging.Logger = None):
    """
    Decorator to log function execution time.

    Usage:
        @log_performance()
        def calculate_emissions():
            pass

        @log_performance(logger=custom_logger)
        def calculate_emissions():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Create log record with performance data
                log_record = _logger.makeRecord(
                    _logger.name,
                    logging.INFO,
                    func.__code__.co_filename,
                    func.__code__.co_firstlineno,
                    f"Function {func.__name__} completed",
                    (),
                    None,
                    func.__name__
                )
                log_record.duration_ms = duration_ms

                _logger.handle(log_record)

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Log error with performance data
                log_record = _logger.makeRecord(
                    _logger.name,
                    logging.ERROR,
                    func.__code__.co_filename,
                    func.__code__.co_firstlineno,
                    f"Function {func.__name__} failed: {str(e)}",
                    (),
                    sys.exc_info(),
                    func.__name__
                )
                log_record.duration_ms = duration_ms

                _logger.handle(log_record)

                raise

        return wrapper
    return decorator


# ============================================================================
# STRUCTURED LOGGING HELPERS
# ============================================================================

class StructuredLogger:
    """
    Helper class for structured logging with common patterns.

    Example:
        logger = StructuredLogger("vcci.calculator")

        logger.info("Calculating emissions", category="1", tier="tier1", supplier_id="SUP-001")
        logger.error("Calculation failed", errors=calculation_errors)

        with logger.operation("Calculate Category 1 emissions"):
            # ... do work ...
            pass
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **context):
        """Log with structured context."""
        extra = {'extra_fields': context}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **context):
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context):
        """Log info message with context."""
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context):
        """Log warning message with context."""
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, exception: Exception = None, **context):
        """Log error message with context and optional exception."""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_message'] = str(exception)

        extra = {'extra_fields': context}
        self.logger.error(message, exc_info=exception is not None, extra=extra)

    def critical(self, message: str, exception: Exception = None, **context):
        """Log critical message with context and optional exception."""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_message'] = str(exception)

        extra = {'extra_fields': context}
        self.logger.critical(message, exc_info=exception is not None, extra=extra)

    class operation:
        """Context manager for logging operations with timing."""

        def __init__(self, logger: 'StructuredLogger', operation_name: str, **context):
            self.logger = logger
            self.operation_name = operation_name
            self.context = context
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            self.logger.info(
                f"Starting operation: {self.operation_name}",
                operation=self.operation_name,
                **self.context
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.time() - self.start_time) * 1000

            if exc_type is None:
                self.logger.info(
                    f"Completed operation: {self.operation_name}",
                    operation=self.operation_name,
                    duration_ms=round(duration_ms, 2),
                    status="success",
                    **self.context
                )
            else:
                self.logger.error(
                    f"Failed operation: {self.operation_name}",
                    exception=exc_val,
                    operation=self.operation_name,
                    duration_ms=round(duration_ms, 2),
                    status="failed",
                    **self.context
                )

            return False  # Don't suppress exceptions


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def configure_development_logging():
    """Configure logging for development environment."""
    return LoggingConfig.configure(
        level="DEBUG",
        enable_json=False,  # Human-readable for dev
        enable_console=True,
        enable_file=False
    )


def configure_production_logging(log_dir: str = "/var/log/vcci"):
    """Configure logging for production environment."""
    return LoggingConfig.configure(
        level="INFO",
        log_dir=log_dir,
        enable_json=True,  # JSON for log aggregation
        enable_console=True,
        enable_file=True
    )


def configure_testing_logging():
    """Configure logging for testing environment."""
    return LoggingConfig.configure(
        level="WARNING",  # Less verbose for tests
        enable_json=False,
        enable_console=True,
        enable_file=False
    )


def setup_logging():
    """
    Setup logging based on environment.
    Called from backend/main.py during startup.
    """
    import os
    env = os.getenv('APP_ENV', 'development')

    if env == 'production':
        return configure_production_logging()
    elif env == 'testing':
        return configure_testing_logging()
    else:
        return configure_development_logging()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    configure_development_logging()

    # Create structured logger
    logger = StructuredLogger("vcci.example")

    # Example: Simple log with carbon context
    CarbonContext.set_context(
        category="1",
        tier="tier1",
        supplier_id="SUP-12345"
    )

    logger.info(
        "Calculating purchased goods emissions",
        quantity=1000,
        unit="kg",
        emission_factor=2.5
    )

    # Example: Operation with timing
    with logger.operation("calculate_category_1_emissions", supplier_count=150):
        time.sleep(0.1)  # Simulate work

    # Example: Error logging
    try:
        raise ValueError("Invalid emission factor")
    except Exception as e:
        logger.error("Calculation failed", exception=e, category="1")

    # Example: Correlation ID
    correlation_id = CorrelationContext.new_correlation_id()
    logger.info("Request started", user_id="user-123")

    # Nested operations maintain correlation ID
    logger.info("Nested operation", step="validation")

    CorrelationContext.clear_correlation_id()
    CarbonContext.clear_context()
