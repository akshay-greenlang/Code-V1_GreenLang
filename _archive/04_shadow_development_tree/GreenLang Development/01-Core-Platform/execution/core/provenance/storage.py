# -*- coding: utf-8 -*-
"""
GreenLang Provenance Storage - Persistent Storage for Audit Trails

This module provides persistent storage for calculation provenance records,
enabling audit trail queries and compliance reporting.

Features:
- SQLite storage for lightweight deployments
- PostgreSQL support for enterprise deployments
- Query interface for audit trail retrieval
- Automatic indexing for performance
- Compliance report generation

Author: GreenLang Team
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional, Protocol
from pathlib import Path
import sqlite3
import json
from datetime import datetime
from abc import ABC, abstractmethod

from .calculation_provenance import CalculationProvenance


class ProvenanceStorage(Protocol):
    """
    Protocol for provenance storage implementations.

    All storage backends must implement this interface.
    """

    def store(self, provenance: CalculationProvenance) -> str:
        """
        Store a provenance record.

        Args:
            provenance: Provenance record to store

        Returns:
            Storage ID for the record
        """
        ...

    def retrieve(self, calculation_id: str) -> Optional[CalculationProvenance]:
        """
        Retrieve a provenance record by ID.

        Args:
            calculation_id: ID of the calculation

        Returns:
            CalculationProvenance if found, None otherwise
        """
        ...

    def query(
        self,
        agent_name: Optional[str] = None,
        calculation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CalculationProvenance]:
        """
        Query provenance records.

        Args:
            agent_name: Filter by agent name
            calculation_type: Filter by calculation type
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (inclusive)
            limit: Maximum number of records to return

        Returns:
            List of matching provenance records
        """
        ...

    def delete(self, calculation_id: str) -> bool:
        """
        Delete a provenance record.

        Args:
            calculation_id: ID of the calculation

        Returns:
            True if deleted, False if not found
        """
        ...


class SQLiteProvenanceStorage:
    """
    SQLite implementation of provenance storage.

    This implementation provides lightweight, file-based storage suitable
    for development, testing, and small-scale deployments.

    Attributes:
        db_path: Path to SQLite database file

    Example:
        >>> storage = SQLiteProvenanceStorage("provenance.db")
        >>> storage.store(provenance)
        '5f3a8b9c1d2e4f6a'
        >>>
        >>> # Later, retrieve the record
        >>> retrieved = storage.retrieve('5f3a8b9c1d2e4f6a')
        >>> assert retrieved.calculation_id == provenance.calculation_id
    """

    def __init__(self, db_path: str = "provenance.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create provenance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provenance (
                calculation_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                agent_version TEXT NOT NULL,
                calculation_type TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT,
                timestamp_start TEXT NOT NULL,
                timestamp_end TEXT,
                duration_ms REAL,
                steps_count INTEGER,
                warnings_count INTEGER,
                errors_count INTEGER,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_name
            ON provenance(agent_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_calculation_type
            ON provenance(calculation_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp_start
            ON provenance(timestamp_start)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_input_hash
            ON provenance(input_hash)
        """)

        # Create steps table for detailed querying
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calculation_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calculation_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                operation TEXT NOT NULL,
                description TEXT,
                data_source TEXT,
                standard_reference TEXT,
                step_json TEXT NOT NULL,
                FOREIGN KEY (calculation_id) REFERENCES provenance(calculation_id)
                    ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_calculation_id
            ON calculation_steps(calculation_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_source
            ON calculation_steps(data_source)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_standard_reference
            ON calculation_steps(standard_reference)
        """)

        conn.commit()
        conn.close()

    def store(self, provenance: CalculationProvenance) -> str:
        """
        Store a provenance record.

        Args:
            provenance: Provenance record to store

        Returns:
            Calculation ID

        Example:
            >>> calc_id = storage.store(provenance)
            >>> print(f"Stored calculation: {calc_id}")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            now = datetime.utcnow().isoformat()
            record_json = json.dumps(provenance.to_dict(), indent=2)

            # Insert main record
            cursor.execute("""
                INSERT OR REPLACE INTO provenance (
                    calculation_id, agent_name, agent_version, calculation_type,
                    input_hash, output_hash, timestamp_start, timestamp_end,
                    duration_ms, steps_count, warnings_count, errors_count,
                    record_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provenance.calculation_id,
                provenance.metadata.agent_name,
                provenance.metadata.agent_version,
                provenance.metadata.calculation_type,
                provenance.input_hash,
                provenance.output_hash,
                provenance.timestamp_start,
                provenance.timestamp_end,
                provenance.duration_ms,
                len(provenance.steps),
                len(provenance.metadata.warnings),
                len(provenance.metadata.errors),
                record_json,
                now,
                now,
            ))

            # Delete existing steps (for updates)
            cursor.execute("""
                DELETE FROM calculation_steps WHERE calculation_id = ?
            """, (provenance.calculation_id,))

            # Insert steps
            for step in provenance.steps:
                step_json = json.dumps(step.to_dict())
                cursor.execute("""
                    INSERT INTO calculation_steps (
                        calculation_id, step_number, operation, description,
                        data_source, standard_reference, step_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    provenance.calculation_id,
                    step.step_number,
                    step.operation,
                    step.description,
                    step.data_source,
                    step.standard_reference,
                    step_json,
                ))

            conn.commit()
            return provenance.calculation_id

        finally:
            conn.close()

    def retrieve(self, calculation_id: str) -> Optional[CalculationProvenance]:
        """
        Retrieve a provenance record by ID.

        Args:
            calculation_id: ID of the calculation

        Returns:
            CalculationProvenance if found, None otherwise

        Example:
            >>> provenance = storage.retrieve("5f3a8b9c1d2e4f6a")
            >>> if provenance:
            ...     print(provenance.get_audit_summary())
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT record_json FROM provenance WHERE calculation_id = ?
            """, (calculation_id,))

            row = cursor.fetchone()
            if row is None:
                return None

            record_data = json.loads(row[0])
            return CalculationProvenance.from_dict(record_data)

        finally:
            conn.close()

    def query(
        self,
        agent_name: Optional[str] = None,
        calculation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        has_errors: Optional[bool] = None,
        has_warnings: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CalculationProvenance]:
        """
        Query provenance records.

        Args:
            agent_name: Filter by agent name
            calculation_type: Filter by calculation type
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (inclusive)
            has_errors: Filter by presence of errors
            has_warnings: Filter by presence of warnings
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of matching provenance records

        Example:
            >>> # Get all emissions calculations from last week
            >>> from datetime import datetime, timedelta
            >>> week_ago = datetime.utcnow() - timedelta(days=7)
            >>> records = storage.query(
            ...     calculation_type="scope1_emissions",
            ...     start_time=week_ago,
            ...     limit=50
            ... )
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Build query
            query = "SELECT record_json FROM provenance WHERE 1=1"
            params = []

            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)

            if calculation_type:
                query += " AND calculation_type = ?"
                params.append(calculation_type)

            if start_time:
                query += " AND timestamp_start >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp_start <= ?"
                params.append(end_time.isoformat())

            if has_errors is not None:
                if has_errors:
                    query += " AND errors_count > 0"
                else:
                    query += " AND errors_count = 0"

            if has_warnings is not None:
                if has_warnings:
                    query += " AND warnings_count > 0"
                else:
                    query += " AND warnings_count = 0"

            # Order by most recent first
            query += " ORDER BY timestamp_start DESC"

            # Add pagination
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                CalculationProvenance.from_dict(json.loads(row[0]))
                for row in rows
            ]

        finally:
            conn.close()

    def delete(self, calculation_id: str) -> bool:
        """
        Delete a provenance record.

        Args:
            calculation_id: ID of the calculation

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = storage.delete("5f3a8b9c1d2e4f6a")
            >>> assert deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM provenance WHERE calculation_id = ?
            """, (calculation_id,))

            conn.commit()
            return cursor.rowcount > 0

        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics

        Example:
            >>> stats = storage.get_statistics()
            >>> print(f"Total calculations: {stats['total_calculations']}")
            >>> print(f"Agents: {stats['unique_agents']}")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Total calculations
            cursor.execute("SELECT COUNT(*) FROM provenance")
            total = cursor.fetchone()[0]

            # Unique agents
            cursor.execute("SELECT COUNT(DISTINCT agent_name) FROM provenance")
            unique_agents = cursor.fetchone()[0]

            # Calculation types
            cursor.execute("""
                SELECT calculation_type, COUNT(*) as count
                FROM provenance
                GROUP BY calculation_type
                ORDER BY count DESC
            """)
            calc_types = {row[0]: row[1] for row in cursor.fetchall()}

            # Records with errors
            cursor.execute("SELECT COUNT(*) FROM provenance WHERE errors_count > 0")
            with_errors = cursor.fetchone()[0]

            # Records with warnings
            cursor.execute("SELECT COUNT(*) FROM provenance WHERE warnings_count > 0")
            with_warnings = cursor.fetchone()[0]

            # Average duration
            cursor.execute("SELECT AVG(duration_ms) FROM provenance WHERE duration_ms > 0")
            avg_duration = cursor.fetchone()[0] or 0.0

            return {
                "total_calculations": total,
                "unique_agents": unique_agents,
                "calculation_types": calc_types,
                "records_with_errors": with_errors,
                "records_with_warnings": with_warnings,
                "average_duration_ms": round(avg_duration, 2),
            }

        finally:
            conn.close()

    def find_by_input_hash(self, input_hash: str) -> List[CalculationProvenance]:
        """
        Find calculations with matching input hash.

        This enables deduplication - finding calculations that used
        the same inputs.

        Args:
            input_hash: SHA-256 hash of input data

        Returns:
            List of calculations with matching input hash

        Example:
            >>> # Find duplicate calculations
            >>> duplicates = storage.find_by_input_hash(provenance.input_hash)
            >>> if len(duplicates) > 1:
            ...     print(f"Found {len(duplicates)} calculations with same inputs")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT record_json FROM provenance WHERE input_hash = ?
                ORDER BY timestamp_start DESC
            """, (input_hash,))

            rows = cursor.fetchall()
            return [
                CalculationProvenance.from_dict(json.loads(row[0]))
                for row in rows
            ]

        finally:
            conn.close()

    def find_by_data_source(self, data_source: str) -> List[CalculationProvenance]:
        """
        Find calculations using a specific data source.

        Args:
            data_source: Data source identifier (e.g., "EPA eGRID 2023")

        Returns:
            List of calculations using this data source

        Example:
            >>> # Find all calculations using DEFRA 2024 factors
            >>> calcs = storage.find_by_data_source("DEFRA 2024")
            >>> print(f"Found {len(calcs)} calculations using DEFRA 2024")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT DISTINCT p.record_json
                FROM provenance p
                JOIN calculation_steps s ON p.calculation_id = s.calculation_id
                WHERE s.data_source = ?
                ORDER BY p.timestamp_start DESC
            """, (data_source,))

            rows = cursor.fetchall()
            return [
                CalculationProvenance.from_dict(json.loads(row[0]))
                for row in rows
            ]

        finally:
            conn.close()

    def export_audit_report(
        self,
        output_path: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> str:
        """
        Export audit report as JSON.

        Args:
            output_path: Path to output file
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)

        Returns:
            Path to exported file

        Example:
            >>> # Export audit report for last month
            >>> from datetime import datetime, timedelta
            >>> month_ago = datetime.utcnow() - timedelta(days=30)
            >>> report_path = storage.export_audit_report(
            ...     "audit_report.json",
            ...     start_time=month_ago
            ... )
        """
        records = self.query(
            start_time=start_time,
            end_time=end_time,
            limit=10000,  # Large limit for export
        )

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "filter": {
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
            },
            "statistics": self.get_statistics(),
            "records": [record.to_dict() for record in records],
        }

        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return str(output_file)


# Convenience function for creating storage
def create_storage(
    storage_type: str = "sqlite",
    db_path: str = "provenance.db",
    **kwargs
) -> ProvenanceStorage:
    """
    Create a provenance storage instance.

    Args:
        storage_type: Type of storage ("sqlite" or "postgresql")
        db_path: Database path/connection string
        **kwargs: Additional storage-specific arguments

    Returns:
        ProvenanceStorage instance

    Example:
        >>> storage = create_storage(storage_type="sqlite", db_path="my_provenance.db")
    """
    if storage_type == "sqlite":
        return SQLiteProvenanceStorage(db_path=db_path)
    elif storage_type == "postgresql":
        # Future implementation
        raise NotImplementedError("PostgreSQL storage not yet implemented")
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
