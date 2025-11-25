# -*- coding: utf-8 -*-
"""
Structured logging configuration for GL-007 FurnacePerformanceMonitor.

Provides:
- JSON-formatted structured logging
- Correlation ID tracking
- Context propagation
- Log levels configuration
- Log rotation
- ELK/Loki integration
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
import traceback
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Context variables for request tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
furnace_id_var: ContextVar[Optional[str]] = ContextVar('furnace_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs logs in JSON format compatible with ELK stack and Loki.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_data = {
            'timestamp': DeterministicClock.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'agent_id': 'GL-007',
            'agent_name': 'FurnacePerformanceMonitor',
        }

        # Add correlation ID if available
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_data['correlation_id'] = correlation_id

        # Add furnace ID if available
        furnace_id = furnace_id_var.get()
        if furnace_id:
            log_data['furnace_id'] = furnace_id

        # Add user ID if available
        user_id = user_id_var.get()
        if user_id:
            log_data['user_id'] = user_id

        # Add source location
        log_data['source'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName,
            'module': record.module,
        }

        # Add process/thread info
        log_data['process'] = {
            'pid': record.process,
            'process_name': record.processName,
            'thread_id': record.thread,
            'thread_name': record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info),
            }

        # Add extra fields from record
        # Skip standard LogRecord attributes
        skip_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'message', 'pathname', 'process', 'processName', 'relativeCreated',
            'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
            'asctime'
        }

        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in skip_fields and not key.startswith('_'):
                extra_fields[key] = value

        if extra_fields:
            log_data['extra'] = extra_fields

        # Add environment info
        log_data['environment'] = os.getenv('ENVIRONMENT', 'production')

        return json.dumps(log_data, default=str)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record."""
        record.correlation_id = correlation_id_var.get()
        record.furnace_id = furnace_id_var.get()
        record.user_id = user_id_var.get()
        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10,
) -> None:
    """
    Configure structured logging for GL-007.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for no file logging)
        enable_console: Enable console output
        enable_json: Use JSON formatting (False for human-readable format)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if enable_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            'log_level': log_level,
            'json_format': enable_json,
            'console_enabled': enable_console,
            'file_enabled': log_file is not None,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Unique request/correlation identifier
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """
    Get correlation ID for current context.

    Returns:
        Current correlation ID or None
    """
    return correlation_id_var.get()


def set_furnace_id(furnace_id: str) -> None:
    """
    Set furnace ID for current context.

    Args:
        furnace_id: Furnace identifier
    """
    furnace_id_var.set(furnace_id)


def get_furnace_id() -> Optional[str]:
    """
    Get furnace ID for current context.

    Returns:
        Current furnace ID or None
    """
    return furnace_id_var.get()


def set_user_id(user_id: str) -> None:
    """
    Set user ID for current context.

    Args:
        user_id: User identifier
    """
    user_id_var.set(user_id)


def get_user_id() -> Optional[str]:
    """
    Get user ID for current context.

    Returns:
        Current user ID or None
    """
    return user_id_var.get()


def clear_context() -> None:
    """Clear all context variables."""
    correlation_id_var.set(None)
    furnace_id_var.set(None)
    user_id_var.set(None)


# Logging middleware for FastAPI/Starlette
class LoggingMiddleware:
    """
    Middleware to add correlation ID to all requests.

    Usage:
        app.add_middleware(LoggingMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Process request with correlation ID."""
        if scope['type'] == 'http':
            # Generate or extract correlation ID
            import uuid
            headers = dict(scope.get('headers', []))
            correlation_id = headers.get(b'x-correlation-id', None)

            if correlation_id:
                correlation_id = correlation_id.decode('utf-8')
            else:
                correlation_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

            # Set correlation ID in context
            set_correlation_id(correlation_id)

            # Log request
            logger = get_logger(__name__)
            logger.info(
                "HTTP request received",
                extra={
                    'method': scope.get('method'),
                    'path': scope.get('path'),
                    'query_string': scope.get('query_string', b'').decode('utf-8'),
                    'client': scope.get('client'),
                }
            )

        await self.app(scope, receive, send)

        # Clear context after request
        clear_context()


# Example usage and configuration presets
LOGGING_CONFIGS = {
    'development': {
        'log_level': 'DEBUG',
        'enable_console': True,
        'enable_json': False,
        'log_file': None,
    },
    'staging': {
        'log_level': 'INFO',
        'enable_console': True,
        'enable_json': True,
        'log_file': '/var/log/greenlang/gl-007/app.log',
    },
    'production': {
        'log_level': 'INFO',
        'enable_console': True,
        'enable_json': True,
        'log_file': '/var/log/greenlang/gl-007/app.log',
    },
}


def setup_logging_for_environment(environment: str = 'production') -> None:
    """
    Setup logging based on environment.

    Args:
        environment: Environment name (development, staging, production)
    """
    config = LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS['production'])
    setup_logging(**config)


# Context manager for temporary context
class LogContext:
    """
    Context manager for temporary logging context.

    Usage:
        with LogContext(correlation_id='abc-123', furnace_id='F-001'):
            logger.info("Processing furnace data")
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        furnace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.correlation_id = correlation_id
        self.furnace_id = furnace_id
        self.user_id = user_id
        self.prev_correlation_id = None
        self.prev_furnace_id = None
        self.prev_user_id = None

    def __enter__(self):
        """Enter context."""
        self.prev_correlation_id = get_correlation_id()
        self.prev_furnace_id = get_furnace_id()
        self.prev_user_id = get_user_id()

        if self.correlation_id:
            set_correlation_id(self.correlation_id)
        if self.furnace_id:
            set_furnace_id(self.furnace_id)
        if self.user_id:
            set_user_id(self.user_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Restore previous context
        if self.prev_correlation_id:
            set_correlation_id(self.prev_correlation_id)
        else:
            correlation_id_var.set(None)

        if self.prev_furnace_id:
            set_furnace_id(self.prev_furnace_id)
        else:
            furnace_id_var.set(None)

        if self.prev_user_id:
            set_user_id(self.prev_user_id)
        else:
            user_id_var.set(None)


# Example usage
if __name__ == '__main__':
    # Setup logging for development
    setup_logging_for_environment('development')

    logger = get_logger(__name__)

    # Simple logging
    logger.info("Application started")

    # Logging with context
    with LogContext(correlation_id='req-123', furnace_id='F-001'):
        logger.info("Processing furnace data")
        logger.warning("Temperature approaching threshold", extra={'temperature': 1250.5})

    # Logging with exception
    try:
        raise ValueError("Example error")
    except Exception as e:
        logger.error("Operation failed", exc_info=True)
