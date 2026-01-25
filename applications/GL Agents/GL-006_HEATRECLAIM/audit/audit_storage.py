"""
GL-006 HEATRECLAIM - Audit Storage Backends

Multiple storage backends for audit records with query support.
Implements file-based, database, and cloud storage options.

Storage backends:
- FileAuditStorage: JSON/JSONL file storage for development
- DatabaseAuditStorage: PostgreSQL/SQLite for production
- S3AuditStorage: AWS S3 for cloud deployments (future)
"""

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit_logger import AuditRecord

logger = logging.getLogger(__name__)


@dataclass
class AuditQuery:
    """Query parameters for audit records."""

    event_type: Optional[str] = None
    action: Optional[str] = None
    severity: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    design_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class AuditQueryResult:
    """Result of audit query."""

    records: List[AuditRecord]
    total_count: int
    query: AuditQuery
    execution_time_ms: float = 0.0

    @property
    def has_more(self) -> bool:
        """Check if there are more records."""
        return self.total_count > self.query.offset + len(self.records)


class AuditStorage(ABC):
    """Abstract base class for audit storage backends."""

    @abstractmethod
    def store(self, record: AuditRecord) -> None:
        """Store a single audit record."""
        pass

    @abstractmethod
    def store_batch(self, records: List[AuditRecord]) -> None:
        """Store multiple audit records."""
        pass

    @abstractmethod
    def query(self, query: AuditQuery) -> AuditQueryResult:
        """Query audit records."""
        pass

    @abstractmethod
    def get_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """Get a single record by ID."""
        pass

    @abstractmethod
    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        pass


class FileAuditStorage(AuditStorage):
    """
    File-based audit storage using JSON Lines format.

    Suitable for development and small deployments.
    Each record is stored as a single JSON line for append-only writes.
    """

    def __init__(
        self,
        directory: str = "audit_logs",
        rotate_size_mb: float = 100.0,
        compress_old: bool = True,
    ):
        """
        Initialize file storage.

        Args:
            directory: Directory for audit files
            rotate_size_mb: Rotate file when size exceeds this
            compress_old: Compress rotated files
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rotate_size_bytes = int(rotate_size_mb * 1024 * 1024)
        self.compress_old = compress_old
        self._current_file: Optional[Path] = None

        logger.info(f"FileAuditStorage initialized at {self.directory}")

    def _get_current_file(self) -> Path:
        """Get current audit log file."""
        if self._current_file is None:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            self._current_file = self.directory / f"audit_{date_str}.jsonl"
        return self._current_file

    def _check_rotation(self) -> None:
        """Check if file rotation is needed."""
        current_file = self._get_current_file()
        if current_file.exists():
            size = current_file.stat().st_size
            if size >= self.rotate_size_bytes:
                self._rotate_file(current_file)

    def _rotate_file(self, file_path: Path) -> None:
        """Rotate the current audit file."""
        timestamp = datetime.utcnow().strftime("%H%M%S")
        new_name = file_path.with_suffix(f".{timestamp}.jsonl")
        file_path.rename(new_name)
        self._current_file = None

        if self.compress_old:
            import gzip
            with open(new_name, "rb") as f_in:
                with gzip.open(f"{new_name}.gz", "wb") as f_out:
                    f_out.write(f_in.read())
            new_name.unlink()

        logger.info(f"Rotated audit log: {file_path} -> {new_name}")

    def store(self, record: AuditRecord) -> None:
        """Store a single audit record."""
        self._check_rotation()
        file_path = self._get_current_file()

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def store_batch(self, records: List[AuditRecord]) -> None:
        """Store multiple audit records."""
        self._check_rotation()
        file_path = self._get_current_file()

        with open(file_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_json() + "\n")

    def query(self, query: AuditQuery) -> AuditQueryResult:
        """Query audit records from files."""
        import time
        start_time = time.perf_counter()

        records = []
        total_count = 0

        # Get all audit files sorted by date
        files = sorted(self.directory.glob("audit_*.jsonl"), reverse=True)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        record = AuditRecord.from_dict(data)

                        if self._matches_query(record, query):
                            total_count += 1
                            if len(records) < query.limit:
                                if total_count > query.offset:
                                    records.append(record)
                    except json.JSONDecodeError:
                        continue

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return AuditQueryResult(
            records=records,
            total_count=total_count,
            query=query,
            execution_time_ms=execution_time_ms,
        )

    def _matches_query(self, record: AuditRecord, query: AuditQuery) -> bool:
        """Check if record matches query filters."""
        if query.event_type and record.event_type != query.event_type:
            return False
        if query.action and record.action != query.action:
            return False
        if query.severity and record.severity != query.severity:
            return False
        if query.correlation_id:
            if record.context.get("correlation_id") != query.correlation_id:
                return False
        if query.user_id:
            if record.context.get("user_id") != query.user_id:
                return False

        if query.start_time:
            record_time = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            if record_time < query.start_time:
                return False

        if query.end_time:
            record_time = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            if record_time > query.end_time:
                return False

        return True

    def get_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """Get a single record by ID."""
        files = sorted(self.directory.glob("audit_*.jsonl"), reverse=True)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("record_id") == record_id:
                            return AuditRecord.from_dict(data)
                    except json.JSONDecodeError:
                        continue

        return None

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity of all records."""
        import hashlib

        files = sorted(self.directory.glob("audit_*.jsonl"))
        previous_hash = None

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        record = AuditRecord.from_dict(data)

                        # Verify hash chain
                        stored_previous = record.provenance.get("previous_hash")
                        if previous_hash is not None:
                            if stored_previous != previous_hash:
                                logger.error(
                                    f"Hash chain broken at {record.record_id}: "
                                    f"expected {previous_hash}, got {stored_previous}"
                                )
                                return False

                        previous_hash = record.provenance.get("record_hash")

                    except json.JSONDecodeError:
                        continue

        logger.info("Hash chain integrity verified")
        return True


class DatabaseAuditStorage(AuditStorage):
    """
    Database-backed audit storage using SQLite or PostgreSQL.

    Suitable for production deployments with query requirements.
    """

    def __init__(
        self,
        connection_string: str = "sqlite:///audit.db",
    ):
        """
        Initialize database storage.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self._connection: Optional[sqlite3.Connection] = None

        self._init_database()
        logger.info(f"DatabaseAuditStorage initialized: {connection_string}")

    def _init_database(self) -> None:
        """Initialize database schema."""
        if self.connection_string.startswith("sqlite:///"):
            db_path = self.connection_string.replace("sqlite:///", "")
            self._connection = sqlite3.connect(db_path, check_same_thread=False)

            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    message TEXT,
                    details TEXT,
                    metadata TEXT,
                    context TEXT,
                    provenance TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON audit_records(event_type)
            """)
            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp)
            """)
            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_severity ON audit_records(severity)
            """)

            self._connection.commit()

    def store(self, record: AuditRecord) -> None:
        """Store a single audit record."""
        self._connection.execute(
            """
            INSERT INTO audit_records
            (record_id, timestamp, event_type, action, severity, outcome,
             message, details, metadata, context, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.timestamp,
                record.event_type,
                record.action,
                record.severity,
                record.outcome,
                record.message,
                json.dumps(record.details),
                json.dumps(record.metadata),
                json.dumps(record.context),
                json.dumps(record.provenance),
            ),
        )
        self._connection.commit()

    def store_batch(self, records: List[AuditRecord]) -> None:
        """Store multiple audit records."""
        for record in records:
            self.store(record)

    def query(self, query: AuditQuery) -> AuditQueryResult:
        """Query audit records from database."""
        import time
        start_time = time.perf_counter()

        sql = "SELECT * FROM audit_records WHERE 1=1"
        params = []

        if query.event_type:
            sql += " AND event_type = ?"
            params.append(query.event_type)

        if query.action:
            sql += " AND action = ?"
            params.append(query.action)

        if query.severity:
            sql += " AND severity = ?"
            params.append(query.severity)

        if query.start_time:
            sql += " AND timestamp >= ?"
            params.append(query.start_time.isoformat())

        if query.end_time:
            sql += " AND timestamp <= ?"
            params.append(query.end_time.isoformat())

        if query.correlation_id:
            sql += " AND json_extract(context, '$.correlation_id') = ?"
            params.append(query.correlation_id)

        # Count total
        count_sql = sql.replace("SELECT *", "SELECT COUNT(*)")
        cursor = self._connection.execute(count_sql, params)
        total_count = cursor.fetchone()[0]

        # Get records with pagination
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])

        cursor = self._connection.execute(sql, params)
        rows = cursor.fetchall()

        records = []
        for row in rows:
            records.append(AuditRecord(
                record_id=row[0],
                timestamp=row[1],
                event_type=row[2],
                action=row[3],
                severity=row[4],
                outcome=row[5],
                message=row[6],
                details=json.loads(row[7]),
                metadata=json.loads(row[8]),
                context=json.loads(row[9]),
                provenance=json.loads(row[10]),
            ))

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return AuditQueryResult(
            records=records,
            total_count=total_count,
            query=query,
            execution_time_ms=execution_time_ms,
        )

    def get_by_id(self, record_id: str) -> Optional[AuditRecord]:
        """Get a single record by ID."""
        cursor = self._connection.execute(
            "SELECT * FROM audit_records WHERE record_id = ?",
            (record_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return AuditRecord(
            record_id=row[0],
            timestamp=row[1],
            event_type=row[2],
            action=row[3],
            severity=row[4],
            outcome=row[5],
            message=row[6],
            details=json.loads(row[7]),
            metadata=json.loads(row[8]),
            context=json.loads(row[9]),
            provenance=json.loads(row[10]),
        )

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        cursor = self._connection.execute(
            "SELECT provenance FROM audit_records ORDER BY timestamp"
        )

        previous_hash = None
        for row in cursor:
            provenance = json.loads(row[0])
            stored_previous = provenance.get("previous_hash")

            if previous_hash is not None:
                if stored_previous != previous_hash:
                    logger.error("Hash chain integrity violation detected")
                    return False

            previous_hash = provenance.get("record_hash")

        return True

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
