"""
Provenance Tracker
==================

Data lineage and audit trail tracking for GreenLang.

Author: Infrastructure Team
Created: 2025-11-21
"""

import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

from greenlang.determinism import deterministic_uuid
from greenlang.infrastructure.base import BaseInfrastructureComponent, InfrastructureConfig

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """A single provenance record."""
    id: str
    timestamp: datetime
    operation: str
    input_hash: str
    output_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    agent_name: Optional[str] = None
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DataLineage:
    """Complete lineage for a data item."""
    data_id: str
    created_at: datetime
    records: List[ProvenanceRecord] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    agents_involved: List[str] = field(default_factory=list)

    def add_record(self, record: ProvenanceRecord) -> None:
        """Add a provenance record to lineage."""
        self.records.append(record)
        if record.operation not in self.transformations:
            self.transformations.append(record.operation)
        if record.agent_name and record.agent_name not in self.agents_involved:
            self.agents_involved.append(record.agent_name)

    def get_chain(self) -> List[str]:
        """Get chain of transformations."""
        return [r.operation for r in sorted(self.records, key=lambda x: x.timestamp)]


class ProvenanceTracker(BaseInfrastructureComponent):
    """
    Tracks data lineage and maintains audit trails.

    Provides SHA-256 hashing for data integrity and complete transformation history.
    """

    def __init__(self, config: Optional[InfrastructureConfig] = None):
        """Initialize provenance tracker."""
        super().__init__(config or InfrastructureConfig(component_name="ProvenanceTracker"))
        self.records: Dict[str, ProvenanceRecord] = {}
        self.lineages: Dict[str, DataLineage] = {}
        self.record_count = 0

    def _initialize(self) -> None:
        """Initialize provenance resources."""
        logger.info("ProvenanceTracker initialized")

    def start(self) -> None:
        """Start the provenance tracker."""
        self.status = self.status.RUNNING
        logger.info("ProvenanceTracker started")

    def stop(self) -> None:
        """Stop the provenance tracker."""
        self.status = self.status.STOPPED
        logger.info("ProvenanceTracker stopped")

    def record_transformation(
        self,
        input_data: Any,
        output_data: Any,
        transformation: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        parent_id: Optional[str] = None
    ) -> ProvenanceRecord:
        """
        Record a data transformation.

        Args:
            input_data: Input data
            output_data: Output data
            transformation: Name/type of transformation
            metadata: Optional metadata about the transformation
            agent_name: Name of agent performing transformation
            parent_id: ID of parent provenance record

        Returns:
            ProvenanceRecord created for this transformation
        """
        self.update_activity()
        self.record_count += 1

        # Generate hashes
        input_hash = self.calculate_hash(input_data)
        output_hash = self.calculate_hash(output_data)

        # Create record with deterministic ID
        # Generate deterministic ID from content
        record_content = f"{transformation}:{input_hash}:{output_hash}:{parent_id}"
        record_id = deterministic_uuid(record_content)

        record = ProvenanceRecord(
            id=record_id,
            timestamp=datetime.now(),
            operation=transformation,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata or {},
            parent_id=parent_id,
            agent_name=agent_name
        )

        # Store record
        self.records[record.id] = record

        # Update lineage
        data_id = output_hash
        if data_id not in self.lineages:
            self.lineages[data_id] = DataLineage(
                data_id=data_id,
                created_at=datetime.now()
            )
        self.lineages[data_id].add_record(record)

        logger.debug(f"Recorded transformation: {transformation} (ID: {record.id})")

        self._update_metrics()
        return record

    def get_lineage(self, data_id: str) -> Optional[DataLineage]:
        """
        Get complete lineage for a data item.

        Args:
            data_id: ID or hash of data item

        Returns:
            DataLineage or None if not found
        """
        return self.lineages.get(data_id)

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """
        Get a specific provenance record.

        Args:
            record_id: Record ID

        Returns:
            ProvenanceRecord or None if not found
        """
        return self.records.get(record_id)

    def calculate_hash(self, data: Any) -> str:
        """
        Calculate SHA-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hash as hex string
        """
        # Convert data to JSON string for consistent hashing
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify_integrity(self, data: Any, expected_hash: str) -> bool:
        """
        Verify data integrity against expected hash.

        Args:
            data: Data to verify
            expected_hash: Expected SHA-256 hash

        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = self.calculate_hash(data)
        is_valid = actual_hash == expected_hash

        if not is_valid:
            logger.warning(f"Integrity check failed. Expected: {expected_hash}, Got: {actual_hash}")

        return is_valid

    def get_transformation_chain(self, start_hash: str, end_hash: str) -> List[ProvenanceRecord]:
        """
        Get chain of transformations between two data states.

        Args:
            start_hash: Starting data hash
            end_hash: Ending data hash

        Returns:
            List of ProvenanceRecords in transformation chain
        """
        chain = []

        # Find all records that match the path
        current_hash = start_hash
        visited = set()

        while current_hash != end_hash and current_hash not in visited:
            visited.add(current_hash)

            # Find record with input_hash = current_hash
            for record in self.records.values():
                if record.input_hash == current_hash:
                    chain.append(record)
                    current_hash = record.output_hash
                    break
            else:
                # No more transformations found
                break

        return chain

    def get_audit_trail(self, agent_name: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[ProvenanceRecord]:
        """
        Get audit trail of transformations.

        Args:
            agent_name: Filter by agent name
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of ProvenanceRecords matching filters
        """
        records = list(self.records.values())

        if agent_name:
            records = [r for r in records if r.agent_name == agent_name]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        return sorted(records, key=lambda r: r.timestamp)

    def export_lineage(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Export lineage as JSON-serializable dictionary.

        Args:
            data_id: ID of data to export lineage for

        Returns:
            Dictionary representation of lineage
        """
        lineage = self.get_lineage(data_id)
        if not lineage:
            return None

        return {
            "data_id": lineage.data_id,
            "created_at": lineage.created_at.isoformat(),
            "transformations": lineage.transformations,
            "agents_involved": lineage.agents_involved,
            "records": [r.to_dict() for r in lineage.records],
            "chain": lineage.get_chain()
        }

    def _update_metrics(self) -> None:
        """Update provenance metrics."""
        self._metrics.update({
            "record_count": self.record_count,
            "lineage_count": len(self.lineages),
            "unique_transformations": len(set(r.operation for r in self.records.values())),
            "unique_agents": len(set(r.agent_name for r in self.records.values() if r.agent_name))
        })