# -*- coding: utf-8 -*-
# SAP Audit Logger
# Comprehensive audit logging for compliance and data lineage

"""
Audit Logger for SAP Integration
=================================

Provides comprehensive audit logging for SAP API calls, authentication events,
and data lineage tracking for compliance and troubleshooting.

Features:
---------
- Structured JSON logging
- API call logging (timestamp, endpoint, method, status, duration)
- Authentication event logging
- Error event logging (rate limits, auth failures, timeouts)
- Data lineage tracking (SAP transaction ID → internal calculation ID)
- Database storage for audit trail
- Support for compliance audits (SOC 2, ESRS)

Usage:
------
```python
from connectors.sap.utils.audit_logger import AuditLogger
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Initialize logger
audit_logger = AuditLogger()

# Log API call
audit_logger.log_api_call(
    endpoint="/api/purchaseorders",
    method="GET",
    status_code=200,
    duration=0.5,
    request_id="req-123"
)

# Log authentication event
audit_logger.log_auth_event(
    event_type="login",
    success=True,
    user="sap_service_account"
)

# Log data lineage
audit_logger.log_lineage(
    sap_transaction_id="PO-12345",
    internal_id="calc-67890",
    entity_type="purchase_order"
)
```
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import structlog
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# SQLAlchemy models for audit trail database
Base = declarative_base()


class AuditLog(Base):
    """Audit log entry model."""

    __tablename__ = "sap_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)
    endpoint = Column(String(500), nullable=True)
    method = Column(String(10), nullable=True)
    status_code = Column(Integer, nullable=True)
    duration = Column(String(20), nullable=True)
    request_id = Column(String(100), nullable=True, index=True)
    user = Column(String(100), nullable=True)
    success = Column(String(10), nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(Text, nullable=True)  # JSON


class LineageLog(Base):
    """Data lineage tracking model."""

    __tablename__ = "sap_lineage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    sap_transaction_id = Column(String(100), nullable=False, index=True)
    internal_id = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    sap_module = Column(String(10), nullable=True)
    metadata = Column(Text, nullable=True)  # JSON


class AuditLogger:
    """
    Comprehensive audit logger for SAP integration.

    Logs all SAP API interactions, authentication events, and data lineage
    for compliance and troubleshooting.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        enable_console: bool = True,
        enable_database: bool = True,
    ):
        """
        Initialize the audit logger.

        Args:
            db_url: Database URL for audit trail storage (optional)
            enable_console: Enable console logging (default: True)
            enable_database: Enable database logging (default: True)
        """
        self.enable_console = enable_console
        self.enable_database = enable_database

        # Initialize database connection if enabled
        if enable_database and db_url:
            try:
                self.engine = create_engine(db_url, pool_pre_ping=True)
                Base.metadata.create_all(self.engine)
                self.SessionLocal = sessionmaker(bind=self.engine)
                logger.info("Audit logger database initialized", db_url=db_url)
            except Exception as e:
                logger.error("Failed to initialize audit database", error=str(e))
                self.enable_database = False
        else:
            self.enable_database = False

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_id: Optional[str] = None,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an SAP API call.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            status_code: HTTP status code
            duration: Request duration in seconds
            request_id: Unique request identifier (optional)
            user: User or service account (optional)
            metadata: Additional metadata (optional)
        """
        log_data = {
            "event_type": "api_call",
            "event_category": "sap_integration",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration": f"{duration:.3f}s",
            "request_id": request_id or str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            "user": user,
            "success": "true" if 200 <= status_code < 300 else "false",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            log_data["metadata"] = json.dumps(metadata)

        # Console logging
        if self.enable_console:
            logger.info(
                "SAP API call",
                **{k: v for k, v in log_data.items() if v is not None},
            )

        # Database logging
        if self.enable_database:
            self._write_to_database(AuditLog, log_data)

    def log_auth_event(
        self,
        event_type: str,
        success: bool,
        user: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an authentication event.

        Args:
            event_type: Type of auth event (login, token_refresh, logout)
            success: Whether the event succeeded
            user: User or service account
            error_message: Error message if failed (optional)
            metadata: Additional metadata (optional)
        """
        log_data = {
            "event_type": event_type,
            "event_category": "authentication",
            "success": "true" if success else "false",
            "user": user,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            log_data["metadata"] = json.dumps(metadata)

        # Console logging
        if self.enable_console:
            if success:
                logger.info(
                    "SAP authentication event",
                    **{k: v for k, v in log_data.items() if v is not None},
                )
            else:
                logger.warning(
                    "SAP authentication failed",
                    **{k: v for k, v in log_data.items() if v is not None},
                )

        # Database logging
        if self.enable_database:
            self._write_to_database(AuditLog, log_data)

    def log_error_event(
        self,
        error_type: str,
        endpoint: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an error event.

        Args:
            error_type: Type of error (rate_limit, timeout, connection_error, etc.)
            endpoint: API endpoint (optional)
            error_message: Error message
            metadata: Additional metadata (optional)
        """
        log_data = {
            "event_type": error_type,
            "event_category": "error",
            "endpoint": endpoint,
            "success": "false",
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            log_data["metadata"] = json.dumps(metadata)

        # Console logging
        if self.enable_console:
            logger.error(
                "SAP error event",
                **{k: v for k, v in log_data.items() if v is not None},
            )

        # Database logging
        if self.enable_database:
            self._write_to_database(AuditLog, log_data)

    def log_lineage(
        self,
        sap_transaction_id: str,
        internal_id: str,
        entity_type: str,
        sap_module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log data lineage mapping.

        Args:
            sap_transaction_id: SAP transaction/document ID
            internal_id: Internal calculation/entity ID
            entity_type: Type of entity (purchase_order, delivery, etc.)
            sap_module: SAP module (MM, SD, FI)
            metadata: Additional metadata (optional)
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc),
            "sap_transaction_id": sap_transaction_id,
            "internal_id": internal_id,
            "entity_type": entity_type,
            "sap_module": sap_module,
        }

        if metadata:
            log_data["metadata"] = json.dumps(metadata)

        # Console logging
        if self.enable_console:
            logger.info(
                "SAP data lineage",
                sap_id=sap_transaction_id,
                internal_id=internal_id,
                entity_type=entity_type,
                module=sap_module,
            )

        # Database logging
        if self.enable_database:
            self._write_to_database(LineageLog, log_data)

    def _write_to_database(
        self, model_class: type, log_data: Dict[str, Any]
    ) -> None:
        """
        Write log entry to database.

        Args:
            model_class: SQLAlchemy model class
            log_data: Log data dictionary
        """
        try:
            session: Session = self.SessionLocal()
            try:
                # Convert timestamp string to datetime if needed
                if "timestamp" in log_data and isinstance(log_data["timestamp"], str):
                    log_data["timestamp"] = datetime.fromisoformat(
                        log_data["timestamp"]
                    )

                log_entry = model_class(**log_data)
                session.add(log_entry)
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.error("Failed to write audit log to database", error=str(e))
