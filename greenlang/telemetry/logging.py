# -*- coding: utf-8 -*-
"""
Structured logging and aggregation for GreenLang
"""

import json
import logging
import sys
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import threading
from collections import deque, defaultdict
import re
from contextlib import contextmanager
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Logging context"""

    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    environment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: datetime
    level: LogLevel
    message: str
    context: LogContext
    data: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        entry = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            **self.context.to_dict(),
        }

        if self.data:
            entry["data"] = self.data

        if self.exception:
            entry["exception"] = self.exception

        return entry

    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict())


class StructuredLogger:
    """Structured logging with context"""

    def __init__(self, name: str, context: Optional[LogContext] = None):
        """
        Initialize structured logger

        Args:
            name: Logger name
            context: Default context
        """
        self.name = name
        self.context = context or LogContext()
        self.logger = logging.getLogger(name)
        self._context_stack = []

    @contextmanager
    def with_context(self, **kwargs):
        """
        Temporarily add context

        Args:
            **kwargs: Context fields to add
        """
        # Save current context
        old_context = LogContext(**asdict(self.context))

        # Update context
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

        try:
            yield self
        finally:
            # Restore context
            self.context = old_context

    def debug(self, message: str, **data):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, data)

    def info(self, message: str, **data):
        """Log info message"""
        self._log(LogLevel.INFO, message, data)

    def warning(self, message: str, **data):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, data)

    def error(self, message: str, exception: Optional[Exception] = None, **data):
        """Log error message"""
        self._log(LogLevel.ERROR, message, data, exception)

    def critical(self, message: str, exception: Optional[Exception] = None, **data):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, data, exception)

    def _log(
        self,
        level: LogLevel,
        message: str,
        data: Dict[str, Any],
        exception: Optional[Exception] = None,
    ):
        """Internal logging method"""
        # Create log entry
        entry = LogEntry(
            timestamp=DeterministicClock.utcnow(),
            level=level,
            message=message,
            context=self.context,
            data=data,
        )

        # Add exception info if present
        if exception:
            entry.exception = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }

        # Log to standard logger
        log_level = getattr(logging, level.value)
        self.logger.log(log_level, entry.to_json())

        # Send to aggregator
        get_log_aggregator().add_log(entry)


class LogFormatter(logging.Formatter):
    """Custom JSON log formatter"""

    def format(self, record):
        """Format log record as JSON"""
        # Check if message is already JSON
        try:
            log_data = json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            # Create structured log from regular message
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            if record.exc_info:
                log_data["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": self.formatException(record.exc_info),
                }

        return json.dumps(log_data)


class LogAggregator:
    """Aggregate and analyze logs"""

    def __init__(self, max_logs: int = 10000):
        """
        Initialize log aggregator

        Args:
            max_logs: Maximum logs to keep in memory
        """
        self.max_logs = max_logs
        self.logs: deque = deque(maxlen=max_logs)
        self.log_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

        # Patterns for error detection
        self.error_regex_patterns = [
            (r"database.*error", "database_error"),
            (r"connection.*failed", "connection_error"),
            (r"timeout", "timeout_error"),
            (r"memory.*error", "memory_error"),
            (r"permission.*denied", "permission_error"),
            (r"not found", "not_found_error"),
        ]

    def add_log(self, entry: LogEntry):
        """
        Add log entry

        Args:
            entry: Log entry
        """
        with self._lock:
            self.logs.append(entry)

            # Update counts
            self.log_counts[entry.level.value][
                entry.context.component or "unknown"
            ] += 1

            # Analyze errors
            if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self._analyze_error(entry)

    def _analyze_error(self, entry: LogEntry):
        """Analyze error patterns"""
        message_lower = entry.message.lower()

        for pattern, error_type in self.error_regex_patterns:
            if re.search(pattern, message_lower):
                self.error_patterns[error_type] += 1

    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        tenant_id: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Get filtered logs

        Args:
            level: Filter by log level
            tenant_id: Filter by tenant
            component: Filter by component
            start_time: Filter by time
            limit: Maximum logs to return

        Returns:
            Filtered logs
        """
        with self._lock:
            filtered_logs = []

            for log in reversed(self.logs):  # Most recent first
                # Apply filters
                if level and log.level != level:
                    continue
                if tenant_id and log.context.tenant_id != tenant_id:
                    continue
                if component and log.context.component != component:
                    continue
                if start_time and log.timestamp < start_time:
                    continue

                filtered_logs.append(log)

                if len(filtered_logs) >= limit:
                    break

            return filtered_logs

    def get_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        with self._lock:
            total_logs = len(self.logs)

            # Calculate time range
            if self.logs:
                oldest = self.logs[0].timestamp
                newest = self.logs[-1].timestamp
                time_range = {
                    "oldest": oldest.isoformat(),
                    "newest": newest.isoformat(),
                    "span_hours": (newest - oldest).total_seconds() / 3600,
                }
            else:
                time_range = None

            return {
                "total_logs": total_logs,
                "log_counts": dict(self.log_counts),
                "error_patterns": dict(self.error_patterns),
                "time_range": time_range,
            }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        with self._lock:
            errors = [
                log
                for log in self.logs
                if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
            ]

            # Group errors by type
            error_groups = defaultdict(list)
            for error in errors:
                if error.exception:
                    error_type = error.exception.get("type", "Unknown")
                else:
                    error_type = "General"
                error_groups[error_type].append(error)

            # Create summary
            summary = {}
            for error_type, error_list in error_groups.items():
                summary[error_type] = {
                    "count": len(error_list),
                    "latest": (
                        error_list[-1].timestamp.isoformat() if error_list else None
                    ),
                    "components": list(
                        set(
                            e.context.component
                            for e in error_list
                            if e.context.component
                        )
                    ),
                }

            return {
                "total_errors": len(errors),
                "error_types": summary,
                "error_patterns": dict(self.error_patterns),
            }


class LogShipper:
    """Ship logs to external systems"""

    def __init__(self):
        """Initialize log shipper"""
        self.destinations: List[Callable[[LogEntry], None]] = []
        self._shipping_thread = None
        self._stop_shipping = threading.Event()
        self.ship_interval = 5  # seconds
        self.batch_size = 100
        self.pending_logs: deque = deque()
        self._lock = threading.Lock()

    def add_destination(self, destination: Callable[[LogEntry], None]):
        """
        Add log destination

        Args:
            destination: Function to ship logs to
        """
        self.destinations.append(destination)

    def ship_log(self, entry: LogEntry):
        """
        Queue log for shipping

        Args:
            entry: Log entry
        """
        with self._lock:
            self.pending_logs.append(entry)

    def start_shipping(self):
        """Start background log shipping"""
        self._shipping_thread = threading.Thread(target=self._ship_loop, daemon=True)
        self._shipping_thread.start()
        logger.info("Started log shipping")

    def stop_shipping(self):
        """Stop background log shipping"""
        self._stop_shipping.set()
        if self._shipping_thread:
            self._shipping_thread.join(timeout=5)

        # Ship remaining logs
        self._ship_batch()
        logger.info("Stopped log shipping")

    def _ship_loop(self):
        """Background shipping loop"""
        while not self._stop_shipping.is_set():
            try:
                self._ship_batch()
            except Exception as e:
                logger.error(f"Error shipping logs: {e}")

            self._stop_shipping.wait(self.ship_interval)

    def _ship_batch(self):
        """Ship a batch of logs"""
        with self._lock:
            if not self.pending_logs:
                return

            # Get batch
            batch = []
            for _ in range(min(self.batch_size, len(self.pending_logs))):
                batch.append(self.pending_logs.popleft())

        # Ship to all destinations
        for destination in self.destinations:
            try:
                for log in batch:
                    destination(log)
            except Exception as e:
                logger.error(f"Error shipping to destination: {e}")


def configure_logging(
    level: str = "INFO", format_json: bool = True, log_file: Optional[str] = None
):
    """
    Configure logging system

    Args:
        level: Log level
        format_json: Use JSON formatting
        log_file: Optional log file path
    """
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, level))

    # Remove existing handlers
    logging.getLogger().handlers = []

    # Create formatter
    if format_json:
        formatter = LogFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    logger.info(f"Logging configured: level={level}, json={format_json}")


def add_log_context(**kwargs):
    """
    Add context to current thread's logs

    Args:
        **kwargs: Context fields
    """
    # This would use thread-local storage in production


# Global instances
_log_aggregator: Optional[LogAggregator] = None
_log_shipper: Optional[LogShipper] = None
_loggers: Dict[str, StructuredLogger] = {}


def get_log_aggregator() -> LogAggregator:
    """Get global log aggregator"""
    global _log_aggregator
    if not _log_aggregator:
        _log_aggregator = LogAggregator()
    return _log_aggregator


def get_log_shipper() -> LogShipper:
    """Get global log shipper"""
    global _log_shipper
    if not _log_shipper:
        _log_shipper = LogShipper()
    return _log_shipper


def get_logger(name: str, context: Optional[LogContext] = None) -> StructuredLogger:
    """
    Get structured logger

    Args:
        name: Logger name
        context: Logger context

    Returns:
        Structured logger
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, context)
    return _loggers[name]
