# -*- coding: utf-8 -*-
"""
CSRD Reporting Platform - Structured Logging Configuration
===========================================================

Production-grade structured logging with JSON format and ESRS/CSRD context.
Integrates with ELK stack, CloudWatch, and other log aggregation systems.

Features:
- JSON structured logging
- ESRS context injection
- Request/response logging
- Performance tracking
- Compliance audit logging

Author: GreenLang Operations Team (Team B3)
Date: 2025-11-08
"""

import logging
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from contextvars import ContextVar
import structlog
from pythonjsonlogger import jsonlogger
from greenlang.determinism import deterministic_uuid, DeterministicClock


# ============================================================================
# CONTEXT VARIABLES
# ============================================================================

# Thread-safe context variables for request tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
company_id_ctx: ContextVar[Optional[str]] = ContextVar('company_id', default=None)
esrs_standard_ctx: ContextVar[Optional[str]] = ContextVar('esrs_standard', default=None)
agent_name_ctx: ContextVar[Optional[str]] = ContextVar('agent_name', default=None)


# ============================================================================
# CUSTOM JSON FORMATTER
# ============================================================================

class ESRSJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with ESRS/CSRD context.

    Adds standard fields to all log records:
    - timestamp (ISO 8601)
    - level
    - logger_name
    - message
    - request_id (if available)
    - user_id (if available)
    - company_id (if available)
    - esrs_standard (if available)
    - agent_name (if available)
    """

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO 8601 format
        log_record['timestamp'] = DeterministicClock.utcnow().isoformat() + 'Z'

        # Add log level
        log_record['level'] = record.levelname

        # Add logger name
        log_record['logger'] = record.name

        # Add source location
        log_record['source'] = {
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName
        }

        # Add context from ContextVars
        if request_id := request_id_ctx.get():
            log_record['request_id'] = request_id

        if user_id := user_id_ctx.get():
            log_record['user_id'] = user_id

        if company_id := company_id_ctx.get():
            log_record['company_id'] = company_id

        if esrs_standard := esrs_standard_ctx.get():
            log_record['esrs_standard'] = esrs_standard

        if agent_name := agent_name_ctx.get():
            log_record['agent_name'] = agent_name

        # Add environment and service info
        log_record['service'] = 'csrd-reporting-platform'
        log_record['environment'] = 'production'

        # Add process and thread info for debugging
        log_record['process'] = {
            'pid': record.process,
            'name': record.processName
        }

        log_record['thread'] = {
            'id': record.thread,
            'name': record.threadName
        }


# ============================================================================
# STRUCTURED LOGGING SETUP
# ============================================================================

def setup_structured_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    enable_json: bool = True,
    enable_console: bool = True
) -> logging.Logger:
    """
    Configure structured logging for CSRD platform.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path for log output
        enable_json: Enable JSON formatting (recommended for production)
        enable_console: Enable console output

    Returns:
        Configured root logger
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # JSON formatter for structured logs
    if enable_json:
        json_formatter = ESRSJsonFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        )
    else:
        # Human-readable formatter for development
        standard_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        if enable_json:
            console_handler.setFormatter(json_formatter)
        else:
            console_handler.setFormatter(standard_formatter)

        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed in file

        if enable_json:
            file_handler.setFormatter(json_formatter)
        else:
            # More detailed format for file
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    logger.info(
        "Structured logging configured",
        extra={
            'log_level': log_level,
            'log_file': log_file,
            'json_enabled': enable_json
        }
    )

    return logger


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class LogContext:
    """
    Context manager for adding ESRS/CSRD context to logs.

    Usage:
        with LogContext(request_id="req-123", company_id="comp-456", esrs_standard="E1"):
            logger.info("Processing ESRS E1 data")
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        company_id: Optional[str] = None,
        esrs_standard: Optional[str] = None,
        agent_name: Optional[str] = None
    ):
        """Initialize log context."""
        self.request_id = request_id
        self.user_id = user_id
        self.company_id = company_id
        self.esrs_standard = esrs_standard
        self.agent_name = agent_name

        # Store previous values for restoration
        self.prev_request_id = None
        self.prev_user_id = None
        self.prev_company_id = None
        self.prev_esrs_standard = None
        self.prev_agent_name = None

    def __enter__(self):
        """Set context variables."""
        if self.request_id:
            self.prev_request_id = request_id_ctx.get()
            request_id_ctx.set(self.request_id)

        if self.user_id:
            self.prev_user_id = user_id_ctx.get()
            user_id_ctx.set(self.user_id)

        if self.company_id:
            self.prev_company_id = company_id_ctx.get()
            company_id_ctx.set(self.company_id)

        if self.esrs_standard:
            self.prev_esrs_standard = esrs_standard_ctx.get()
            esrs_standard_ctx.set(self.esrs_standard)

        if self.agent_name:
            self.prev_agent_name = agent_name_ctx.get()
            agent_name_ctx.set(self.agent_name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context variables."""
        if self.request_id:
            request_id_ctx.set(self.prev_request_id)

        if self.user_id:
            user_id_ctx.set(self.prev_user_id)

        if self.company_id:
            company_id_ctx.set(self.prev_company_id)

        if self.esrs_standard:
            esrs_standard_ctx.set(self.prev_esrs_standard)

        if self.agent_name:
            agent_name_ctx.set(self.prev_agent_name)


class AuditLogger:
    """
    Compliance audit logger for CSRD/ESRS operations.

    Logs all compliance-relevant operations for audit trails.
    """

    def __init__(self, logger_name: str = 'csrd.audit'):
        """Initialize audit logger."""
        self.logger = logging.getLogger(logger_name)

    def log_data_access(
        self,
        user_id: str,
        company_id: str,
        esrs_standard: str,
        action: str,
        data_points: Optional[int] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log data access for compliance audit.

        Args:
            user_id: User accessing data
            company_id: Company ID
            esrs_standard: ESRS standard accessed
            action: Action performed (read, write, update, delete)
            data_points: Number of data points accessed
            success: Whether operation succeeded
            details: Additional details
        """
        audit_entry = {
            'audit_type': 'data_access',
            'user_id': user_id,
            'company_id': company_id,
            'esrs_standard': esrs_standard,
            'action': action,
            'data_points': data_points,
            'success': success,
            'timestamp': DeterministicClock.utcnow().isoformat() + 'Z'
        }

        if details:
            audit_entry['details'] = details

        self.logger.info(
            f"Data access: {action} on {esrs_standard}",
            extra=audit_entry
        )

    def log_report_generation(
        self,
        user_id: str,
        company_id: str,
        report_type: str,
        reporting_period: str,
        esrs_standards: list,
        success: bool = True,
        duration_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log report generation for compliance audit.

        Args:
            user_id: User generating report
            company_id: Company ID
            report_type: Type of report (annual, interim, etc.)
            reporting_period: Reporting period (e.g., "2024")
            esrs_standards: List of ESRS standards in report
            success: Whether generation succeeded
            duration_seconds: Time taken to generate
            details: Additional details
        """
        audit_entry = {
            'audit_type': 'report_generation',
            'user_id': user_id,
            'company_id': company_id,
            'report_type': report_type,
            'reporting_period': reporting_period,
            'esrs_standards': esrs_standards,
            'success': success,
            'duration_seconds': duration_seconds,
            'timestamp': DeterministicClock.utcnow().isoformat() + 'Z'
        }

        if details:
            audit_entry['details'] = details

        self.logger.info(
            f"Report generation: {report_type} for {reporting_period}",
            extra=audit_entry
        )

    def log_validation(
        self,
        company_id: str,
        esrs_standard: str,
        validation_type: str,
        passed: bool,
        errors_count: int = 0,
        warnings_count: int = 0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log validation operations for compliance audit.

        Args:
            company_id: Company ID
            esrs_standard: ESRS standard validated
            validation_type: Type of validation
            passed: Whether validation passed
            errors_count: Number of errors found
            warnings_count: Number of warnings found
            details: Additional details
        """
        audit_entry = {
            'audit_type': 'validation',
            'company_id': company_id,
            'esrs_standard': esrs_standard,
            'validation_type': validation_type,
            'passed': passed,
            'errors_count': errors_count,
            'warnings_count': warnings_count,
            'timestamp': DeterministicClock.utcnow().isoformat() + 'Z'
        }

        if details:
            audit_entry['details'] = details

        level = logging.WARNING if not passed else logging.INFO

        self.logger.log(
            level,
            f"Validation: {validation_type} on {esrs_standard} - {'PASSED' if passed else 'FAILED'}",
            extra=audit_entry
        )


# ============================================================================
# LOGGER FACTORY
# ============================================================================

def get_logger(
    name: str,
    log_level: Optional[str] = None,
    agent_name: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger for a specific module or agent.

    Args:
        name: Logger name (typically __name__)
        log_level: Optional specific log level for this logger
        agent_name: Optional agent name for context

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if log_level:
        logger.setLevel(getattr(logging, log_level.upper()))

    # Add agent context if provided
    if agent_name:
        agent_name_ctx.set(agent_name)

    return logger


def get_audit_logger() -> AuditLogger:
    """
    Get the audit logger instance.

    Returns:
        AuditLogger instance
    """
    return AuditLogger()


# ============================================================================
# MIDDLEWARE INTEGRATION
# ============================================================================

async def logging_middleware(request, call_next):
    """
    FastAPI middleware for request/response logging.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        app.middleware("http")(logging_middleware)
    """
    import uuid
    import time

    # Generate request ID
    request_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

    # Set context
    request_id_ctx.set(request_id)

    # Extract user/company from auth headers if available
    if user_id := request.headers.get('X-User-ID'):
        user_id_ctx.set(user_id)

    if company_id := request.headers.get('X-Company-ID'):
        company_id_ctx.set(company_id)

    # Log request
    logger = logging.getLogger('csrd.http')
    start_time = time.time()

    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            'request': {
                'method': request.method,
                'path': request.url.path,
                'query_params': str(request.query_params),
                'client_ip': request.client.host if request.client else None
            }
        }
    )

    # Process request
    try:
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time

        logger.info(
            f"Response: {response.status_code} in {duration:.3f}s",
            extra={
                'response': {
                    'status_code': response.status_code,
                    'duration_seconds': duration
                }
            }
        )

        # Add request ID to response headers
        response.headers['X-Request-ID'] = request_id

        return response

    except Exception as e:
        duration = time.time() - start_time

        logger.error(
            f"Request failed: {str(e)}",
            extra={
                'error': {
                    'type': type(e).__name__,
                    'message': str(e)
                },
                'duration_seconds': duration
            },
            exc_info=True
        )
        raise


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Setup structured logging
    setup_structured_logging(
        log_level='INFO',
        log_file='logs/csrd-app.log',
        enable_json=True
    )

    # Get logger
    logger = get_logger(__name__)

    # Basic logging
    logger.info("Application started")

    # Logging with context
    with LogContext(
        request_id="req-12345",
        company_id="comp-acme",
        esrs_standard="E1"
    ):
        logger.info("Processing ESRS E1 climate data")
        logger.warning("Missing data points detected", extra={'missing_count': 5})

    # Audit logging
    audit = get_audit_logger()

    audit.log_data_access(
        user_id="user-123",
        company_id="comp-acme",
        esrs_standard="E1",
        action="read",
        data_points=150,
        success=True
    )

    audit.log_report_generation(
        user_id="user-123",
        company_id="comp-acme",
        report_type="annual",
        reporting_period="2024",
        esrs_standards=["E1", "E2", "S1"],
        success=True,
        duration_seconds=45.2
    )

    audit.log_validation(
        company_id="comp-acme",
        esrs_standard="E1",
        validation_type="data_completeness",
        passed=True,
        errors_count=0,
        warnings_count=3
    )

    print("\nLog files created. Check logs/csrd-app.log for structured JSON logs.")
