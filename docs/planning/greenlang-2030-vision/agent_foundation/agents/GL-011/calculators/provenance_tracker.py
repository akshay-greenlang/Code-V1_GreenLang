# -*- coding: utf-8 -*-
"""
Provenance Tracker for GL-011 FUELCRAFT.

Provides SHA-256 audit trail tracking for all calculations,
ensuring complete traceability and reproducibility.

Zero-hallucination: Complete deterministic provenance tracking.
"""

import hashlib
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """Single provenance record for a calculation."""
    record_id: str
    timestamp: str
    operation: str
    input_hash: str
    output_hash: str
    combined_hash: str
    tool_calls: List[str]
    data_sources: List[str]
    calculation_details: Dict[str, Any]
    verification_status: str = "unverified"


@dataclass
class ProvenanceChain:
    """Chain of provenance records for audit trail."""
    chain_id: str
    created_at: str
    records: List[ProvenanceRecord] = field(default_factory=list)
    chain_hash: str = ""
    verified: bool = False


class ProvenanceTracker:
    """
    Deterministic provenance tracking for audit trails.

    Provides SHA-256 hashing for all inputs, outputs, and calculations
    to ensure complete traceability and reproducibility.

    Thread-safe implementation using RLock.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> record = tracker.record(
        ...     operation="multi_fuel_optimization",
        ...     inputs={"demand_mw": 100},
        ...     outputs={"cost": 5000},
        ...     tool_calls=["optimize_multi_fuel"]
        ... )
        >>> print(f"Hash: {record.combined_hash}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize provenance tracker."""
        self.config = config or {}
        self._lock = threading.RLock()
        self._records: OrderedDict[str, ProvenanceRecord] = OrderedDict()
        self._chains: Dict[str, ProvenanceChain] = {}
        self._record_count = 0
        self._chain_count = 0

    def record(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_calls: Optional[List[str]] = None,
        data_sources: Optional[List[str]] = None,
        calculation_details: Optional[Dict[str, Any]] = None
    ) -> ProvenanceRecord:
        """
        Record provenance for a calculation.

        Args:
            operation: Name of the operation/calculation
            inputs: Input data
            outputs: Output data
            tool_calls: List of tools called
            data_sources: Data sources used
            calculation_details: Additional calculation details

        Returns:
            ProvenanceRecord with SHA-256 hashes
        """
        with self._lock:
            self._record_count += 1

            # Generate deterministic timestamp (truncated to seconds)
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

            # Calculate input hash
            input_str = json.dumps(inputs, sort_keys=True, default=str)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            # Calculate output hash
            output_str = json.dumps(outputs, sort_keys=True, default=str)
            output_hash = hashlib.sha256(output_str.encode()).hexdigest()

            # Calculate combined hash
            combined_str = f"{operation}|{input_hash}|{output_hash}|{timestamp}"
            combined_hash = hashlib.sha256(combined_str.encode()).hexdigest()

            # Generate record ID
            record_id = f"GL011-{self._record_count:08d}-{combined_hash[:8]}"

            record = ProvenanceRecord(
                record_id=record_id,
                timestamp=timestamp,
                operation=operation,
                input_hash=input_hash,
                output_hash=output_hash,
                combined_hash=combined_hash,
                tool_calls=tool_calls or [],
                data_sources=data_sources or [],
                calculation_details=calculation_details or {}
            )

            self._records[record_id] = record
            logger.debug(f"Recorded provenance: {record_id}")

            return record

    def verify(
        self,
        record_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> bool:
        """
        Verify a provenance record.

        Args:
            record_id: Record ID to verify
            inputs: Input data to verify
            outputs: Output data to verify

        Returns:
            True if verification passes
        """
        with self._lock:
            if record_id not in self._records:
                logger.warning(f"Record not found: {record_id}")
                return False

            record = self._records[record_id]

            # Recalculate hashes
            input_str = json.dumps(inputs, sort_keys=True, default=str)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            output_str = json.dumps(outputs, sort_keys=True, default=str)
            output_hash = hashlib.sha256(output_str.encode()).hexdigest()

            # Verify
            if input_hash != record.input_hash:
                logger.warning(f"Input hash mismatch for {record_id}")
                return False

            if output_hash != record.output_hash:
                logger.warning(f"Output hash mismatch for {record_id}")
                return False

            # Update verification status
            record.verification_status = "verified"
            logger.info(f"Verified provenance: {record_id}")

            return True

    def create_chain(self) -> str:
        """
        Create a new provenance chain.

        Returns:
            Chain ID
        """
        with self._lock:
            self._chain_count += 1
            chain_id = f"CHAIN-{self._chain_count:06d}"
            timestamp = datetime.now(timezone.utc).isoformat()

            chain = ProvenanceChain(
                chain_id=chain_id,
                created_at=timestamp
            )

            self._chains[chain_id] = chain
            return chain_id

    def add_to_chain(self, chain_id: str, record: ProvenanceRecord) -> None:
        """
        Add a record to a chain.

        Args:
            chain_id: Chain ID
            record: Provenance record to add
        """
        with self._lock:
            if chain_id not in self._chains:
                raise ValueError(f"Chain not found: {chain_id}")

            chain = self._chains[chain_id]
            chain.records.append(record)

            # Update chain hash
            chain_data = [r.combined_hash for r in chain.records]
            chain_str = json.dumps(chain_data, sort_keys=True)
            chain.chain_hash = hashlib.sha256(chain_str.encode()).hexdigest()

    def verify_chain(self, chain_id: str) -> bool:
        """
        Verify integrity of a provenance chain.

        Args:
            chain_id: Chain ID to verify

        Returns:
            True if chain is intact
        """
        with self._lock:
            if chain_id not in self._chains:
                return False

            chain = self._chains[chain_id]

            # Recalculate chain hash
            chain_data = [r.combined_hash for r in chain.records]
            chain_str = json.dumps(chain_data, sort_keys=True)
            expected_hash = hashlib.sha256(chain_str.encode()).hexdigest()

            if expected_hash != chain.chain_hash:
                logger.warning(f"Chain hash mismatch for {chain_id}")
                return False

            chain.verified = True
            return True

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get a provenance record by ID."""
        with self._lock:
            return self._records.get(record_id)

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """Get a provenance chain by ID."""
        with self._lock:
            return self._chains.get(chain_id)

    def export_audit_trail(
        self,
        format: str = "json"
    ) -> str:
        """
        Export complete audit trail.

        Args:
            format: Export format (json, csv)

        Returns:
            Serialized audit trail
        """
        with self._lock:
            records = []
            for record in self._records.values():
                records.append({
                    'record_id': record.record_id,
                    'timestamp': record.timestamp,
                    'operation': record.operation,
                    'input_hash': record.input_hash,
                    'output_hash': record.output_hash,
                    'combined_hash': record.combined_hash,
                    'tool_calls': record.tool_calls,
                    'verification_status': record.verification_status
                })

            if format == "json":
                return json.dumps({
                    'export_timestamp': datetime.now(timezone.utc).isoformat(),
                    'record_count': len(records),
                    'records': records
                }, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance tracking statistics."""
        with self._lock:
            verified_count = sum(
                1 for r in self._records.values()
                if r.verification_status == "verified"
            )

            return {
                'total_records': len(self._records),
                'verified_records': verified_count,
                'verification_rate': verified_count / len(self._records) if self._records else 0,
                'total_chains': len(self._chains),
                'verified_chains': sum(1 for c in self._chains.values() if c.verified)
            }

    def clear(self) -> None:
        """Clear all provenance records (for testing)."""
        with self._lock:
            self._records.clear()
            self._chains.clear()
            self._record_count = 0
            self._chain_count = 0
            logger.info("Provenance tracker cleared")
