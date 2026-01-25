# -*- coding: utf-8 -*-
"""
GreenLang Tool Audit Logging
=============================

Production-grade audit logging for tool execution with privacy protection.

Features:
- Comprehensive execution logging
- Privacy-safe hashing (no raw sensitive data)
- Thread-safe operation
- Log rotation and retention management
- Query interface for log analysis
- JSON-based log format
- Optional database backend

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import json
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Audit Log Entry
# ==============================================================================

@dataclass
class AuditLogEntry:
    """
    Single audit log entry for tool execution.

    Uses privacy-safe hashing for inputs/outputs to avoid logging sensitive data.
    """

    timestamp: str  # ISO format timestamp
    tool_name: str
    user_id: Optional[str]
    session_id: Optional[str]
    input_hash: str  # SHA256 of inputs (privacy-safe)
    output_hash: str  # SHA256 of outputs (privacy-safe)
    execution_time_ms: float
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditLogEntry:
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> AuditLogEntry:
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ==============================================================================
# Audit Logger
# ==============================================================================

class AuditLogger:
    """
    Thread-safe audit logger for tool execution.

    Features:
    - Privacy-safe hashing of inputs/outputs
    - Automatic log rotation
    - Retention policy enforcement
    - Query interface
    - Statistics tracking

    Example:
        >>> logger = AuditLogger(log_file=Path("logs/audit/tool_audit.jsonl"))
        >>> logger.log_execution(
        ...     tool_name="calculate_emissions",
        ...     inputs={"amount": 100, "factor": 0.5},
        ...     result=ToolResult(success=True, data={"emissions": 50}),
        ...     execution_time_ms=12.5
        ... )
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        log_to_db: bool = False,
        retention_days: int = 90,
        auto_rotate: bool = True,
        max_log_size_mb: int = 100
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to log file (None = logs/audit/tool_audit.jsonl)
            log_to_db: Enable database logging (not yet implemented)
            retention_days: Days to retain logs before deletion
            auto_rotate: Automatically rotate logs when size exceeded
            max_log_size_mb: Maximum log file size before rotation
        """
        # Set default log file path
        if log_file is None:
            log_file = Path("logs/audit/tool_audit.jsonl")

        self.log_file = log_file
        self.log_to_db = log_to_db
        self.retention_days = retention_days
        self.auto_rotate = auto_rotate
        self.max_log_size_mb = max_log_size_mb

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # In-memory cache for recent logs (for queries)
        self._cache: List[AuditLogEntry] = []
        self._cache_max_size = 1000

        # Statistics
        self._total_logged = 0
        self._success_count = 0
        self._failure_count = 0
        self._execution_time_total_ms = 0.0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized audit logger: {self.log_file}")

    def _hash_data(self, data: Any) -> str:
        """
        Create privacy-safe hash of data.

        Args:
            data: Data to hash (will be JSON-serialized)

        Returns:
            SHA256 hash string
        """
        try:
            # Convert to JSON for consistent hashing
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to hash data: {e}")
            return "HASH_ERROR"

    def log_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        result: Any,  # ToolResult or similar
        execution_time_ms: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log tool execution with privacy-safe hashing.

        Args:
            tool_name: Name of executed tool
            inputs: Tool input parameters
            result: Tool execution result (should have success, data, error attrs)
            execution_time_ms: Execution time in milliseconds
            user_id: User ID (optional)
            session_id: Session ID (optional)
            metadata: Additional metadata (optional)
        """
        with self._lock:
            try:
                # Extract result attributes
                if hasattr(result, 'success'):
                    success = result.success
                    error_message = getattr(result, 'error', None)
                    output_data = getattr(result, 'data', {})
                else:
                    # Fallback for non-ToolResult objects
                    success = True
                    error_message = None
                    output_data = result

                # Create privacy-safe hashes
                input_hash = self._hash_data(inputs)
                output_hash = self._hash_data(output_data)

                # Create log entry
                entry = AuditLogEntry(
                    timestamp=DeterministicClock.utcnow().isoformat() + "Z",
                    tool_name=tool_name,
                    user_id=user_id,
                    session_id=session_id,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    execution_time_ms=round(execution_time_ms, 3),
                    success=success,
                    error_message=error_message,
                    metadata=metadata or {}
                )

                # Write to log file
                self._write_log_entry(entry)

                # Add to cache
                self._add_to_cache(entry)

                # Update statistics
                self._total_logged += 1
                if success:
                    self._success_count += 1
                else:
                    self._failure_count += 1
                self._execution_time_total_ms += execution_time_ms

                # Check if rotation needed
                if self.auto_rotate:
                    self._check_rotation()

            except Exception as e:
                self.logger.error(f"Failed to log execution: {e}", exc_info=True)

    def _write_log_entry(self, entry: AuditLogEntry) -> None:
        """
        Write log entry to file.

        Args:
            entry: Log entry to write
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write log entry: {e}", exc_info=True)

    def _add_to_cache(self, entry: AuditLogEntry) -> None:
        """
        Add entry to in-memory cache.

        Args:
            entry: Log entry to cache
        """
        self._cache.append(entry)

        # Trim cache if too large
        if len(self._cache) > self._cache_max_size:
            self._cache = self._cache[-self._cache_max_size:]

    def _check_rotation(self) -> None:
        """Check if log rotation is needed and perform if necessary."""
        try:
            if not self.log_file.exists():
                return

            # Check file size
            size_mb = self.log_file.stat().st_size / (1024 * 1024)

            if size_mb >= self.max_log_size_mb:
                self.logger.info(f"Rotating log file (size: {size_mb:.2f}MB)")
                self.rotate_logs()

        except Exception as e:
            self.logger.error(f"Failed to check rotation: {e}", exc_info=True)

    def rotate_logs(self) -> None:
        """
        Rotate log files.

        Renames current log file with timestamp and starts new file.
        """
        with self._lock:
            try:
                if not self.log_file.exists():
                    return

                # Create rotated file name with timestamp
                timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
                rotated_file = self.log_file.parent / f"{self.log_file.stem}_{timestamp}.jsonl"

                # Rename current log file
                self.log_file.rename(rotated_file)

                self.logger.info(f"Rotated log file to: {rotated_file}")

                # Clean up old logs based on retention policy
                self._cleanup_old_logs()

            except Exception as e:
                self.logger.error(f"Failed to rotate logs: {e}", exc_info=True)

    def _cleanup_old_logs(self) -> None:
        """Delete log files older than retention period."""
        try:
            cutoff_date = DeterministicClock.now() - timedelta(days=self.retention_days)

            # Find all rotated log files
            pattern = f"{self.log_file.stem}_*.jsonl"
            for log_file in self.log_file.parent.glob(pattern):
                try:
                    # Parse timestamp from filename
                    parts = log_file.stem.split('_')
                    if len(parts) >= 3:
                        date_str = parts[-2]  # YYYYMMDD
                        time_str = parts[-1]  # HHMMSS

                        file_date = datetime.strptime(
                            f"{date_str}_{time_str}",
                            "%Y%m%d_%H%M%S"
                        )

                        if file_date < cutoff_date:
                            log_file.unlink()
                            self.logger.info(f"Deleted old log file: {log_file}")

                except Exception as e:
                    self.logger.warning(f"Failed to process log file {log_file}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}", exc_info=True)

    def query_logs(
        self,
        tool_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: bool = False,
        failure_only: bool = False,
        limit: Optional[int] = None
    ) -> List[AuditLogEntry]:
        """
        Query audit logs with filters.

        Args:
            tool_name: Filter by tool name
            user_id: Filter by user ID
            session_id: Filter by session ID
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (inclusive)
            success_only: Only return successful executions
            failure_only: Only return failed executions
            limit: Maximum number of results

        Returns:
            List of matching AuditLogEntry objects
        """
        with self._lock:
            results = []

            # First check cache (faster for recent logs)
            for entry in reversed(self._cache):
                if self._matches_filters(
                    entry,
                    tool_name,
                    user_id,
                    session_id,
                    start_time,
                    end_time,
                    success_only,
                    failure_only
                ):
                    results.append(entry)

                    if limit and len(results) >= limit:
                        return results

            # If we need more results, read from file
            if limit is None or len(results) < limit:
                file_results = self._query_log_file(
                    tool_name,
                    user_id,
                    session_id,
                    start_time,
                    end_time,
                    success_only,
                    failure_only,
                    limit - len(results) if limit else None
                )
                results.extend(file_results)

            return results

    def _matches_filters(
        self,
        entry: AuditLogEntry,
        tool_name: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        success_only: bool,
        failure_only: bool
    ) -> bool:
        """Check if entry matches query filters."""
        if tool_name and entry.tool_name != tool_name:
            return False

        if user_id and entry.user_id != user_id:
            return False

        if session_id and entry.session_id != session_id:
            return False

        if success_only and not entry.success:
            return False

        if failure_only and entry.success:
            return False

        # Parse timestamp for time filtering
        entry_time = datetime.fromisoformat(entry.timestamp.rstrip('Z'))

        if start_time and entry_time < start_time:
            return False

        if end_time and entry_time > end_time:
            return False

        return True

    def _query_log_file(
        self,
        tool_name: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        success_only: bool,
        failure_only: bool,
        limit: Optional[int]
    ) -> List[AuditLogEntry]:
        """Query log file directly."""
        results = []

        try:
            if not self.log_file.exists():
                return results

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = AuditLogEntry.from_json(line.strip())

                        if self._matches_filters(
                            entry,
                            tool_name,
                            user_id,
                            session_id,
                            start_time,
                            end_time,
                            success_only,
                            failure_only
                        ):
                            results.append(entry)

                            if limit and len(results) >= limit:
                                break

                    except Exception as e:
                        self.logger.warning(f"Failed to parse log entry: {e}")

        except Exception as e:
            self.logger.error(f"Failed to query log file: {e}", exc_info=True)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get audit logging statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            avg_execution_time = (
                self._execution_time_total_ms / self._total_logged
                if self._total_logged > 0
                else 0.0
            )

            success_rate = (
                self._success_count / self._total_logged * 100
                if self._total_logged > 0
                else 0.0
            )

            return {
                "total_logged": self._total_logged,
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "success_rate_percentage": round(success_rate, 2),
                "avg_execution_time_ms": round(avg_execution_time, 2),
                "cache_size": len(self._cache),
                "log_file": str(self.log_file),
                "log_file_exists": self.log_file.exists(),
                "log_file_size_mb": (
                    round(self.log_file.stat().st_size / (1024 * 1024), 2)
                    if self.log_file.exists()
                    else 0.0
                ),
                "retention_days": self.retention_days,
            }

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific tool.

        Args:
            tool_name: Tool name

        Returns:
            Dictionary with tool-specific statistics
        """
        entries = self.query_logs(tool_name=tool_name)

        if not entries:
            return {
                "tool_name": tool_name,
                "total_executions": 0,
                "success_count": 0,
                "failure_count": 0,
            }

        success_count = sum(1 for e in entries if e.success)
        failure_count = len(entries) - success_count

        execution_times = [e.execution_time_ms for e in entries]
        avg_time = sum(execution_times) / len(execution_times)

        return {
            "tool_name": tool_name,
            "total_executions": len(entries),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate_percentage": round(success_count / len(entries) * 100, 2),
            "avg_execution_time_ms": round(avg_time, 2),
            "min_execution_time_ms": round(min(execution_times), 2),
            "max_execution_time_ms": round(max(execution_times), 2),
        }

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        with self._lock:
            self._cache.clear()
            self.logger.info("Cleared audit log cache")

    def __repr__(self) -> str:
        return (
            f"AuditLogger(log_file={self.log_file}, "
            f"logged={self._total_logged}, "
            f"retention_days={self.retention_days})"
        )


# ==============================================================================
# Global Audit Logger Instance
# ==============================================================================

_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get global audit logger instance.

    Returns:
        Global AuditLogger instance
    """
    global _global_audit_logger

    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()

    return _global_audit_logger


def configure_audit_logger(
    log_file: Optional[Path] = None,
    retention_days: int = 90,
    auto_rotate: bool = True,
    max_log_size_mb: int = 100
) -> AuditLogger:
    """
    Configure global audit logger.

    Args:
        log_file: Path to log file
        retention_days: Days to retain logs
        auto_rotate: Enable automatic rotation
        max_log_size_mb: Maximum log file size

    Returns:
        Configured AuditLogger instance
    """
    global _global_audit_logger

    _global_audit_logger = AuditLogger(
        log_file=log_file,
        retention_days=retention_days,
        auto_rotate=auto_rotate,
        max_log_size_mb=max_log_size_mb
    )

    logger.info(f"Configured global audit logger: {_global_audit_logger}")
    return _global_audit_logger
