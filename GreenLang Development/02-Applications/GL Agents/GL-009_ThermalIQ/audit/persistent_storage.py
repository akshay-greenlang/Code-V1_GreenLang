# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ - Persistent Audit Storage

Provides immutable, tamper-evident audit storage for thermal efficiency
calculations and recommendations. Supports 7-year retention per EPA/FDA.

Reference: EPA 40 CFR Part 75, FDA 21 CFR Part 11, SOC 2 Type II
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
import gzip, hashlib, json, logging, os, sqlite3, threading, uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Supported storage backends."""
    SQLITE = "sqlite"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"


class RetentionPolicy(str, Enum):
    """Data retention policies per regulation."""
    EPA_7_YEAR = "epa_7_year"
    FDA_LIFETIME = "fda_lifetime"
    SOC2_1_YEAR = "soc2_1_year"
    CUSTOM = "custom"


class IntegrityStatus(str, Enum):
    """Audit record integrity status."""
    VALID = "valid"
    CORRUPTED = "corrupted"
    MISSING_CHAIN = "missing_chain"
    TAMPERED = "tampered"


@dataclass(frozen=True)
class AuditRecord:
    """Immutable audit record with cryptographic seal."""
    record_id: str
    timestamp: datetime
    record_type: str
    payload: Dict[str, Any]
    input_hash: str
    output_hash: str
    provenance_chain: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of record contents."""
        content = json.dumps({
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "record_type": self.record_type,
            "payload": self.payload,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "provenance_chain": list(self.provenance_chain),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "record_type": self.record_type,
            "payload": self.payload,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "provenance_chain": list(self.provenance_chain),
            "metadata": dict(self.metadata),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        return cls(
            record_id=data["record_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            record_type=data["record_type"],
            payload=data["payload"],
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            provenance_chain=tuple(data.get("provenance_chain", [])),
            metadata=data.get("metadata", {}),
            signature=data.get("signature", ""),
        )


@dataclass
class IntegrityCheckResult:
    """Result of integrity verification."""
    status: IntegrityStatus
    records_checked: int
    valid_records: int
    corrupted_records: List[str]
    chain_breaks: List[Tuple[str, str]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_valid(self) -> bool:
        return self.status == IntegrityStatus.VALID


class AuditStorageBackend(ABC):
    """Abstract base class for audit storage backends."""

    @abstractmethod
    def store(self, record: AuditRecord) -> bool:
        """Store an audit record."""
        pass

    @abstractmethod
    def retrieve(self, record_id: str) -> Optional[AuditRecord]:
        """Retrieve a specific record by ID."""
        pass

    @abstractmethod
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        record_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditRecord]:
        """Query records with filters."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total record count."""
        pass

    @abstractmethod
    def get_chain_head(self) -> Optional[str]:
        """Get the hash of the latest record (chain head)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""
        pass


class SQLiteAuditStorage(AuditStorageBackend):
    """SQLite-based audit storage with write-ahead logging."""

    def __init__(self, db_path: Union[str, Path], retention_days: int = 2555):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_records (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                record_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                provenance_chain TEXT,
                metadata TEXT,
                signature TEXT,
                record_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_record_type ON audit_records(record_type)
        """)
        self._conn.commit()
        logger.info(f"SQLite audit storage initialized at {self.db_path}")

    def store(self, record: AuditRecord) -> bool:
        """Store an audit record with integrity hash."""
        with self._lock:
            try:
                record_hash = record.compute_hash()
                self._conn.execute(
                    """INSERT INTO audit_records
                       (record_id, timestamp, record_type, payload, input_hash,
                        output_hash, provenance_chain, metadata, signature, record_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.record_id,
                        record.timestamp.isoformat(),
                        record.record_type,
                        json.dumps(record.payload),
                        record.input_hash,
                        record.output_hash,
                        json.dumps(list(record.provenance_chain)),
                        json.dumps(dict(record.metadata)),
                        record.signature,
                        record_hash,
                    )
                )
                self._conn.commit()
                logger.debug(f"Stored audit record {record.record_id}")
                return True
            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate record ID: {record.record_id}")
                return False
            except Exception as e:
                logger.error(f"Failed to store audit record: {e}")
                return False

    def retrieve(self, record_id: str) -> Optional[AuditRecord]:
        """Retrieve a specific record by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM audit_records WHERE record_id = ?",
                (record_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
            return None

    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        record_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditRecord]:
        """Query records with filters."""
        with self._lock:
            conditions = []
            params = []

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
            if record_type:
                conditions.append("record_type = ?")
                params.append(record_type)

            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            query = f"""
                SELECT * FROM audit_records {where}
                ORDER BY timestamp DESC LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            cursor = self._conn.execute(query, params)
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Get total record count."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM audit_records")
            return cursor.fetchone()[0]

    def get_chain_head(self) -> Optional[str]:
        """Get the hash of the latest record."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT record_hash FROM audit_records ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def verify_integrity(self) -> IntegrityCheckResult:
        """Verify integrity of all stored records."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM audit_records ORDER BY timestamp ASC"
            )
            rows = cursor.fetchall()

            records_checked = 0
            valid_records = 0
            corrupted = []
            chain_breaks = []
            prev_hash = None

            for row in rows:
                records_checked += 1
                record = self._row_to_record(row)
                stored_hash = row[9]  # record_hash column
                computed_hash = record.compute_hash()

                if stored_hash != computed_hash:
                    corrupted.append(record.record_id)
                else:
                    valid_records += 1

                # Check chain continuity
                if prev_hash and record.provenance_chain:
                    if prev_hash not in record.provenance_chain:
                        chain_breaks.append((prev_hash, record.record_id))

                prev_hash = stored_hash

            if corrupted:
                status = IntegrityStatus.CORRUPTED
            elif chain_breaks:
                status = IntegrityStatus.MISSING_CHAIN
            else:
                status = IntegrityStatus.VALID

            return IntegrityCheckResult(
                status=status,
                records_checked=records_checked,
                valid_records=valid_records,
                corrupted_records=corrupted,
                chain_breaks=chain_breaks,
            )

    def purge_expired(self) -> int:
        """Remove records past retention period."""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            cursor = self._conn.execute(
                "DELETE FROM audit_records WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            self._conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Purged {count} expired audit records")
            return count

    def export_records(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        compress: bool = True
    ) -> int:
        """Export records to JSON file."""
        records = self.query(start_time=start_time, end_time=end_time, limit=1000000)
        data = [r.to_dict() for r in records]
        content = json.dumps(data, indent=2)

        if compress:
            output_path = output_path.with_suffix(".json.gz")
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        logger.info(f"Exported {len(records)} records to {output_path}")
        return len(records)

    def _row_to_record(self, row: Tuple) -> AuditRecord:
        """Convert database row to AuditRecord."""
        return AuditRecord(
            record_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            record_type=row[2],
            payload=json.loads(row[3]),
            input_hash=row[4],
            output_hash=row[5],
            provenance_chain=tuple(json.loads(row[6] or "[]")),
            metadata=json.loads(row[7] or "{}"),
            signature=row[8] or "",
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("SQLite audit storage closed")


class ThermalIQAuditPersistence:
    """
    High-level audit persistence manager for GL-009 ThermalIQ.

    Provides immutable audit trail for:
    - Thermal efficiency calculations
    - Fluid property lookups
    - Optimization recommendations
    - Configuration changes

    Features:
    - 7-year retention per EPA requirements
    - Cryptographic integrity verification
    - Chain-of-custody provenance
    - Export for regulatory submission
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        retention_policy: RetentionPolicy = RetentionPolicy.EPA_7_YEAR,
        agent_id: str = "GL-009"
    ):
        self.agent_id = agent_id
        self.retention_policy = retention_policy

        # Set retention days based on policy
        retention_map = {
            RetentionPolicy.EPA_7_YEAR: 2555,
            RetentionPolicy.FDA_LIFETIME: 36500,
            RetentionPolicy.SOC2_1_YEAR: 365,
            RetentionPolicy.CUSTOM: 2555,
        }
        retention_days = retention_map.get(retention_policy, 2555)

        # Initialize storage
        if storage_path is None:
            storage_path = Path.home() / ".thermaliq" / "audit.db"
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._storage = SQLiteAuditStorage(storage_path, retention_days)
        self._chain_head: Optional[str] = self._storage.get_chain_head()

        logger.info(
            f"ThermalIQAuditPersistence initialized: "
            f"policy={retention_policy.value}, path={storage_path}"
        )

    def log_calculation(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a thermal calculation to audit trail."""
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Compute input/output hashes
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()

        # Build provenance chain
        provenance = (self._chain_head,) if self._chain_head else ()

        record = AuditRecord(
            record_id=record_id,
            timestamp=timestamp,
            record_type=f"CALCULATION:{calculation_type}",
            payload={"inputs": inputs, "outputs": outputs},
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_chain=provenance,
            metadata=metadata or {},
        )

        if self._storage.store(record):
            self._chain_head = record.compute_hash()
            logger.debug(f"Logged calculation {calculation_type}: {record_id}")
            return record_id
        else:
            raise RuntimeError(f"Failed to store audit record: {record_id}")

    def log_recommendation(
        self,
        recommendation_type: str,
        recommendation: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Log an optimization recommendation."""
        return self._log_record(
            record_type=f"RECOMMENDATION:{recommendation_type}",
            payload={"recommendation": recommendation, "context": context}
        )

    def log_configuration_change(
        self,
        component: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        changed_by: str = "system"
    ) -> str:
        """Log a configuration change."""
        return self._log_record(
            record_type="CONFIG_CHANGE",
            payload={
                "component": component,
                "old_config": old_config,
                "new_config": new_config,
                "changed_by": changed_by,
            }
        )

    def log_fluid_lookup(
        self,
        fluid_name: str,
        conditions: Dict[str, Any],
        properties: Dict[str, Any],
        source: str = "IAPWS-IF97"
    ) -> str:
        """Log a fluid property lookup."""
        return self._log_record(
            record_type="FLUID_LOOKUP",
            payload={
                "fluid": fluid_name,
                "conditions": conditions,
                "properties": properties,
                "source": source,
            }
        )

    def _log_record(
        self,
        record_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Internal method to log a generic record."""
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        payload_str = json.dumps(payload, sort_keys=True)
        input_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        output_hash = input_hash  # Same for non-calculation records

        provenance = (self._chain_head,) if self._chain_head else ()

        record = AuditRecord(
            record_id=record_id,
            timestamp=timestamp,
            record_type=record_type,
            payload=payload,
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_chain=provenance,
            metadata=metadata or {"agent_id": self.agent_id},
        )

        if self._storage.store(record):
            self._chain_head = record.compute_hash()
            return record_id
        else:
            raise RuntimeError(f"Failed to store audit record: {record_id}")

    def verify_chain(self) -> IntegrityCheckResult:
        """Verify integrity of the entire audit chain."""
        return self._storage.verify_integrity()

    def get_record(self, record_id: str) -> Optional[AuditRecord]:
        """Retrieve a specific audit record."""
        return self._storage.retrieve(record_id)

    def query_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        record_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """Query audit records with filters."""
        return self._storage.query(
            start_time=start_time,
            end_time=end_time,
            record_type=record_type,
            limit=limit
        )

    def export_for_compliance(
        self,
        output_dir: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Path:
        """Export audit records for regulatory compliance submission."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"thermaliq_audit_{timestamp}.json"
        output_path = output_dir / filename

        self._storage.export_records(output_path, start_time, end_time, compress=True)
        return output_path.with_suffix(".json.gz")

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit storage statistics."""
        integrity = self._storage.verify_integrity()
        return {
            "total_records": self._storage.count(),
            "chain_head": self._chain_head,
            "integrity_status": integrity.status.value,
            "valid_records": integrity.valid_records,
            "retention_policy": self.retention_policy.value,
            "agent_id": self.agent_id,
        }

    def close(self) -> None:
        """Close audit storage."""
        self._storage.close()


# Factory function
def create_audit_persistence(
    storage_path: Optional[Path] = None,
    retention_policy: RetentionPolicy = RetentionPolicy.EPA_7_YEAR
) -> ThermalIQAuditPersistence:
    """Create a configured audit persistence instance."""
    return ThermalIQAuditPersistence(
        storage_path=storage_path,
        retention_policy=retention_policy
    )


__all__ = [
    "StorageBackend",
    "RetentionPolicy",
    "IntegrityStatus",
    "AuditRecord",
    "IntegrityCheckResult",
    "AuditStorageBackend",
    "SQLiteAuditStorage",
    "ThermalIQAuditPersistence",
    "create_audit_persistence",
]
