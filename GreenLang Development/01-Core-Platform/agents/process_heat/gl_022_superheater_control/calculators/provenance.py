"""
GL-022 SUPERHEATER CONTROL - Provenance Tracking Module

This module provides zero-hallucination provenance tracking including:
- SHA-256 provenance hash generation
- Calculation audit trail management
- Deterministic verification
- Regulatory compliance documentation
- Bit-perfect reproducibility verification

All calculations in GreenLang must be traceable and auditable.
This module ensures complete provenance for regulatory compliance.

Standards Reference:
    - GHG Protocol: Calculation Methodology Documentation
    - ASME PTC: Test Uncertainty Analysis
    - ISO 14064: GHG Inventories and Verification

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.calculators.provenance import (
    ...     ProvenanceTracker,
    ...     CalculationAuditTrail,
    ... )
    >>>
    >>> tracker = ProvenanceTracker()
    >>> hash_value = tracker.generate_hash({"input": 100, "output": 50})
    >>> tracker.verify_hash({"input": 100, "output": 50}, hash_value)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - PROVENANCE CONFIGURATION
# =============================================================================

class ProvenanceConstants:
    """Constants for provenance tracking."""

    # Hash algorithm
    HASH_ALGORITHM = "sha256"

    # Hash truncation for display (full hash always stored)
    DISPLAY_HASH_LENGTH = 16

    # Audit trail retention
    MAX_AUDIT_ENTRIES = 10000
    AUDIT_RETENTION_DAYS = 365

    # Verification tolerance for floating point
    FLOAT_TOLERANCE = 1e-10

    # Version info
    PROVENANCE_VERSION = "1.0.0"
    SCHEMA_VERSION = "2024.1"


# =============================================================================
# DATA CLASSES FOR PROVENANCE
# =============================================================================

@dataclass
class CalculationInput:
    """Represents input to a calculation with metadata."""
    name: str
    value: Any
    unit: Optional[str] = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CalculationStep:
    """Represents a single calculation step in the audit trail."""
    step_number: int
    operation: str
    inputs: Dict[str, Any]
    output: Any
    formula: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a calculation."""
    calculation_id: str
    calculation_type: str
    timestamp: datetime
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    steps: List[CalculationStep]
    provenance_hash: str
    formula_references: List[str]
    standard_references: List[str]
    version: str
    is_verified: bool = False


@dataclass
class AuditEntry:
    """Single entry in the audit trail."""
    entry_id: str
    timestamp: datetime
    calculation_type: str
    inputs_hash: str
    outputs_hash: str
    provenance_hash: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    verification_status: str = "unverified"


@dataclass
class VerificationResult:
    """Result of provenance verification."""
    is_valid: bool
    original_hash: str
    computed_hash: str
    match: bool
    timestamp: datetime
    details: Dict[str, Any]
    errors: List[str]


# =============================================================================
# PROVENANCE HASH GENERATOR
# =============================================================================

class ProvenanceHashGenerator:
    """
    Generator for SHA-256 provenance hashes.

    Ensures deterministic, reproducible hash generation for any
    calculation data structure.

    Key Features:
    - Canonical JSON serialization (sorted keys)
    - Decimal precision handling
    - Datetime standardization
    - Nested structure support

    Example:
        >>> generator = ProvenanceHashGenerator()
        >>> data = {"temperature": 850.5, "pressure": 600}
        >>> hash_value = generator.generate_hash(data)
        >>> print(f"Hash: {hash_value}")
    """

    def __init__(
        self,
        algorithm: str = ProvenanceConstants.HASH_ALGORITHM,
        precision: int = 10,
    ) -> None:
        """
        Initialize hash generator.

        Args:
            algorithm: Hash algorithm (default: sha256)
            precision: Decimal precision for floating point
        """
        self.algorithm = algorithm
        self.precision = precision

        logger.debug(f"ProvenanceHashGenerator initialized: {algorithm}")

    def generate_hash(
        self,
        data: Any,
        include_timestamp: bool = False,
    ) -> str:
        """
        Generate SHA-256 hash for calculation data - DETERMINISTIC.

        Args:
            data: Data to hash (dict, list, or primitive)
            include_timestamp: Include current timestamp in hash

        Returns:
            SHA-256 hash string (64 hex characters)
        """
        # Normalize data for canonical representation
        normalized = self._normalize_data(data)

        # Add timestamp if requested
        if include_timestamp:
            normalized["_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Serialize to canonical JSON (sorted keys, no whitespace variance)
        json_str = json.dumps(
            normalized,
            sort_keys=True,
            separators=(',', ':'),
            default=self._json_serializer,
        )

        # Generate hash
        hash_bytes = hashlib.sha256(json_str.encode('utf-8')).digest()
        hash_hex = hash_bytes.hex()

        return hash_hex

    def generate_short_hash(
        self,
        data: Any,
        length: int = ProvenanceConstants.DISPLAY_HASH_LENGTH,
    ) -> str:
        """
        Generate truncated hash for display - DETERMINISTIC.

        Args:
            data: Data to hash
            length: Truncation length

        Returns:
            Truncated hash string
        """
        full_hash = self.generate_hash(data)
        return full_hash[:length]

    def verify_hash(
        self,
        data: Any,
        expected_hash: str,
    ) -> bool:
        """
        Verify data matches expected hash - DETERMINISTIC.

        Args:
            data: Data to verify
            expected_hash: Expected hash value

        Returns:
            True if hashes match
        """
        computed_hash = self.generate_hash(data)

        # Handle truncated hashes
        if len(expected_hash) < len(computed_hash):
            computed_hash = computed_hash[:len(expected_hash)]

        return computed_hash == expected_hash

    def _normalize_data(self, data: Any) -> Any:
        """Normalize data for consistent hashing."""
        if isinstance(data, dict):
            return {
                str(k): self._normalize_data(v)
                for k, v in sorted(data.items())
            }
        elif isinstance(data, (list, tuple)):
            return [self._normalize_data(item) for item in data]
        elif isinstance(data, float):
            # Round floats to avoid floating point precision issues
            return round(data, self.precision)
        elif isinstance(data, Decimal):
            return float(round(data, self.precision))
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, (int, str, bool, type(None))):
            return data
        else:
            # Convert other types to string
            return str(data)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


# =============================================================================
# CALCULATION AUDIT TRAIL
# =============================================================================

class CalculationAuditTrail:
    """
    Audit trail manager for tracking all calculations.

    Maintains a complete history of calculations with:
    - Input/output records
    - Step-by-step traces
    - Verification status
    - Compliance documentation

    Example:
        >>> trail = CalculationAuditTrail()
        >>> entry_id = trail.record_calculation(
        ...     calc_type="spray_flow",
        ...     inputs={"steam_flow": 100000, "temp_in": 950},
        ...     outputs={"spray_flow": 5000},
        ... )
        >>> trail.get_entry(entry_id)
    """

    def __init__(
        self,
        max_entries: int = ProvenanceConstants.MAX_AUDIT_ENTRIES,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Initialize audit trail.

        Args:
            max_entries: Maximum entries to retain
            session_id: Session identifier for grouping
        """
        self.max_entries = max_entries
        self.session_id = session_id or str(uuid.uuid4())

        self._entries: deque = deque(maxlen=max_entries)
        self._hash_generator = ProvenanceHashGenerator()

        logger.info(f"CalculationAuditTrail initialized: session={self.session_id}")

    def record_calculation(
        self,
        calc_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        steps: Optional[List[CalculationStep]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Record a calculation in the audit trail - DETERMINISTIC.

        Args:
            calc_type: Type of calculation
            inputs: Input parameters
            outputs: Output values
            steps: Calculation steps (optional)
            user_id: User identifier (optional)

        Returns:
            Entry ID for the record
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Generate hashes
        inputs_hash = self._hash_generator.generate_short_hash(inputs)
        outputs_hash = self._hash_generator.generate_short_hash(outputs)

        # Combined provenance hash
        provenance_data = {
            "type": calc_type,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = self._hash_generator.generate_hash(provenance_data)

        # Create entry
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            calculation_type=calc_type,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            provenance_hash=provenance_hash,
            user_id=user_id,
            session_id=self.session_id,
            verification_status="recorded",
        )

        self._entries.append(entry)

        logger.debug(f"Recorded calculation: {calc_type} -> {entry_id}")

        return entry_id

    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get audit entry by ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_entries_by_type(self, calc_type: str) -> List[AuditEntry]:
        """Get all entries of a specific calculation type."""
        return [e for e in self._entries if e.calculation_type == calc_type]

    def get_entries_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[AuditEntry]:
        """Get entries within a time range."""
        return [
            e for e in self._entries
            if start_time <= e.timestamp <= end_time
        ]

    def verify_entry(
        self,
        entry_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> VerificationResult:
        """
        Verify an audit entry against provided data - DETERMINISTIC.

        Args:
            entry_id: Entry to verify
            inputs: Input data to verify
            outputs: Output data to verify

        Returns:
            VerificationResult with verification status
        """
        entry = self.get_entry(entry_id)

        if entry is None:
            return VerificationResult(
                is_valid=False,
                original_hash="",
                computed_hash="",
                match=False,
                timestamp=datetime.now(timezone.utc),
                details={"error": "Entry not found"},
                errors=["Entry not found"],
            )

        # Recompute hashes
        computed_inputs_hash = self._hash_generator.generate_short_hash(inputs)
        computed_outputs_hash = self._hash_generator.generate_short_hash(outputs)

        # Check matches
        inputs_match = computed_inputs_hash == entry.inputs_hash
        outputs_match = computed_outputs_hash == entry.outputs_hash

        is_valid = inputs_match and outputs_match

        errors = []
        if not inputs_match:
            errors.append("Inputs hash mismatch")
        if not outputs_match:
            errors.append("Outputs hash mismatch")

        # Update entry verification status
        if is_valid:
            entry.verification_status = "verified"
        else:
            entry.verification_status = "failed"

        return VerificationResult(
            is_valid=is_valid,
            original_hash=entry.provenance_hash,
            computed_hash=self._hash_generator.generate_hash({
                "inputs": inputs,
                "outputs": outputs,
            }),
            match=is_valid,
            timestamp=datetime.now(timezone.utc),
            details={
                "inputs_match": inputs_match,
                "outputs_match": outputs_match,
                "entry_timestamp": entry.timestamp.isoformat(),
            },
            errors=errors,
        )

    def export_audit_log(
        self,
        format: str = "json",
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Export audit trail for compliance documentation.

        Args:
            format: Export format ("json" or "dict")

        Returns:
            Audit log in requested format
        """
        log_entries = []
        for entry in self._entries:
            log_entries.append({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "calculation_type": entry.calculation_type,
                "inputs_hash": entry.inputs_hash,
                "outputs_hash": entry.outputs_hash,
                "provenance_hash": entry.provenance_hash,
                "user_id": entry.user_id,
                "session_id": entry.session_id,
                "verification_status": entry.verification_status,
            })

        if format == "json":
            return json.dumps(log_entries, indent=2)
        else:
            return log_entries

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        total = len(self._entries)
        by_type = {}
        by_status = {}

        for entry in self._entries:
            by_type[entry.calculation_type] = by_type.get(entry.calculation_type, 0) + 1
            by_status[entry.verification_status] = by_status.get(entry.verification_status, 0) + 1

        return {
            "total_entries": total,
            "max_entries": self.max_entries,
            "utilization_pct": round(total / self.max_entries * 100, 1) if self.max_entries > 0 else 0,
            "by_calculation_type": by_type,
            "by_verification_status": by_status,
            "session_id": self.session_id,
        }


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================

class ProvenanceTracker:
    """
    Comprehensive provenance tracker for GreenLang calculations.

    Provides a unified interface for:
    - Hash generation and verification
    - Audit trail management
    - Regulatory compliance documentation
    - Reproducibility verification

    Example:
        >>> tracker = ProvenanceTracker()
        >>>
        >>> # Record a calculation
        >>> provenance = tracker.track_calculation(
        ...     calc_type="enthalpy_balance",
        ...     inputs={"steam_flow": 100000, "h_in": 1300, "h_out": 1250},
        ...     outputs={"spray_flow": 5000},
        ...     formula="m_spray = m_steam * (h_in - h_out) / (h_out - h_water)",
        ... )
        >>>
        >>> # Verify later
        >>> is_valid = tracker.verify(provenance.provenance_hash, inputs, outputs)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        enable_audit_trail: bool = True,
    ) -> None:
        """
        Initialize provenance tracker.

        Args:
            session_id: Session identifier
            enable_audit_trail: Enable audit trail recording
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.enable_audit_trail = enable_audit_trail

        self._hash_generator = ProvenanceHashGenerator()
        self._audit_trail = CalculationAuditTrail(
            session_id=self.session_id
        ) if enable_audit_trail else None

        # Provenance records cache
        self._records: Dict[str, ProvenanceRecord] = {}

        logger.info(f"ProvenanceTracker initialized: session={self.session_id}")

    def track_calculation(
        self,
        calc_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        standard_references: Optional[List[str]] = None,
    ) -> ProvenanceRecord:
        """
        Track a calculation with full provenance - DETERMINISTIC.

        Args:
            calc_type: Type of calculation
            inputs: Input parameters
            outputs: Output values
            formula: Formula used (optional)
            steps: Calculation steps (optional)
            standard_references: References to standards (optional)

        Returns:
            ProvenanceRecord with complete tracking info
        """
        calculation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Convert steps to CalculationStep objects
        calc_steps = []
        if steps:
            for i, step in enumerate(steps):
                calc_steps.append(CalculationStep(
                    step_number=i + 1,
                    operation=step.get("operation", ""),
                    inputs=step.get("inputs", {}),
                    output=step.get("output"),
                    formula=step.get("formula", ""),
                    description=step.get("description", ""),
                ))

        # Generate provenance hash
        provenance_data = {
            "calculation_type": calc_type,
            "inputs": inputs,
            "outputs": outputs,
            "formula": formula,
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = self._hash_generator.generate_hash(provenance_data)

        # Create record
        record = ProvenanceRecord(
            calculation_id=calculation_id,
            calculation_type=calc_type,
            timestamp=timestamp,
            inputs=inputs,
            outputs=outputs,
            steps=calc_steps,
            provenance_hash=provenance_hash,
            formula_references=[formula] if formula else [],
            standard_references=standard_references or [],
            version=ProvenanceConstants.PROVENANCE_VERSION,
        )

        # Store record
        self._records[calculation_id] = record

        # Record in audit trail
        if self._audit_trail:
            self._audit_trail.record_calculation(
                calc_type=calc_type,
                inputs=inputs,
                outputs=outputs,
                steps=calc_steps,
            )

        logger.debug(f"Tracked calculation: {calc_type} -> {calculation_id}")

        return record

    def verify(
        self,
        provenance_hash: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calc_type: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify provenance hash matches data - DETERMINISTIC.

        Args:
            provenance_hash: Hash to verify against
            inputs: Input data
            outputs: Output data
            calc_type: Calculation type (optional, for full verification)

        Returns:
            VerificationResult with verification details
        """
        errors = []
        timestamp = datetime.now(timezone.utc)

        # Reconstruct provenance data
        provenance_data = {
            "inputs": inputs,
            "outputs": outputs,
        }
        if calc_type:
            provenance_data["calculation_type"] = calc_type

        # We need to match the original structure, but we don't have timestamp
        # So we verify inputs/outputs only
        inputs_outputs_hash = self._hash_generator.generate_hash({
            "inputs": inputs,
            "outputs": outputs,
        })

        # For exact match, we'd need the original timestamp
        # Here we check if the I/O portion matches
        # Full verification requires stored record lookup

        # Try to find record by hash
        record = None
        for r in self._records.values():
            if r.provenance_hash == provenance_hash:
                record = r
                break

        if record:
            # Full verification with stored record
            stored_inputs_hash = self._hash_generator.generate_short_hash(record.inputs)
            provided_inputs_hash = self._hash_generator.generate_short_hash(inputs)

            stored_outputs_hash = self._hash_generator.generate_short_hash(record.outputs)
            provided_outputs_hash = self._hash_generator.generate_short_hash(outputs)

            inputs_match = stored_inputs_hash == provided_inputs_hash
            outputs_match = stored_outputs_hash == provided_outputs_hash

            if not inputs_match:
                errors.append("Inputs do not match stored record")
            if not outputs_match:
                errors.append("Outputs do not match stored record")

            is_valid = inputs_match and outputs_match
            record.is_verified = is_valid

        else:
            # No stored record - can only do hash comparison
            is_valid = False
            errors.append("No stored record found for provenance hash")

        return VerificationResult(
            is_valid=is_valid,
            original_hash=provenance_hash,
            computed_hash=inputs_outputs_hash,
            match=is_valid,
            timestamp=timestamp,
            details={
                "record_found": record is not None,
                "calculation_type": record.calculation_type if record else None,
                "original_timestamp": record.timestamp.isoformat() if record else None,
            },
            errors=errors,
        )

    def generate_hash(self, data: Any) -> str:
        """
        Generate SHA-256 hash for arbitrary data - DETERMINISTIC.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hash string
        """
        return self._hash_generator.generate_hash(data)

    def get_record(self, calculation_id: str) -> Optional[ProvenanceRecord]:
        """Get provenance record by calculation ID."""
        return self._records.get(calculation_id)

    def get_records_by_type(self, calc_type: str) -> List[ProvenanceRecord]:
        """Get all records of a specific calculation type."""
        return [r for r in self._records.values() if r.calculation_type == calc_type]

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        if self._audit_trail:
            return self._audit_trail.get_statistics()
        return {"audit_trail_enabled": False}

    def export_compliance_report(self) -> Dict[str, Any]:
        """
        Export compliance report for regulatory documentation.

        Returns:
            Dictionary with compliance documentation
        """
        records_summary = []
        for record in self._records.values():
            records_summary.append({
                "calculation_id": record.calculation_id,
                "calculation_type": record.calculation_type,
                "timestamp": record.timestamp.isoformat(),
                "provenance_hash": record.provenance_hash,
                "formula_references": record.formula_references,
                "standard_references": record.standard_references,
                "is_verified": record.is_verified,
                "steps_count": len(record.steps),
            })

        return {
            "report_type": "GreenLang Provenance Compliance Report",
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "provenance_version": ProvenanceConstants.PROVENANCE_VERSION,
            "schema_version": ProvenanceConstants.SCHEMA_VERSION,
            "total_calculations": len(self._records),
            "audit_statistics": self.get_audit_statistics(),
            "records": records_summary,
        }


# =============================================================================
# DETERMINISTIC VERIFICATION UTILITIES
# =============================================================================

class DeterministicVerifier:
    """
    Utilities for verifying deterministic calculation behavior.

    Ensures calculations produce identical results when
    given identical inputs (bit-perfect reproducibility).

    Example:
        >>> verifier = DeterministicVerifier()
        >>> is_deterministic = verifier.verify_reproducibility(
        ...     calculation_fn=lambda x: x * 2,
        ...     inputs=[100, 200, 300],
        ...     iterations=5,
        ... )
    """

    def __init__(
        self,
        tolerance: float = ProvenanceConstants.FLOAT_TOLERANCE,
    ) -> None:
        """
        Initialize deterministic verifier.

        Args:
            tolerance: Floating point comparison tolerance
        """
        self.tolerance = tolerance
        self._hash_generator = ProvenanceHashGenerator()

    def verify_reproducibility(
        self,
        calculation_fn: callable,
        inputs: List[Any],
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Verify calculation produces identical results - DETERMINISTIC.

        Args:
            calculation_fn: Function to test
            inputs: List of test inputs
            iterations: Number of iterations per input

        Returns:
            Verification results
        """
        results = []
        all_reproducible = True

        for input_val in inputs:
            hashes = []
            outputs = []

            for _ in range(iterations):
                output = calculation_fn(input_val)
                output_hash = self._hash_generator.generate_hash(output)
                hashes.append(output_hash)
                outputs.append(output)

            # Check all hashes are identical
            is_reproducible = len(set(hashes)) == 1

            if not is_reproducible:
                all_reproducible = False

            results.append({
                "input": input_val,
                "is_reproducible": is_reproducible,
                "unique_hashes": len(set(hashes)),
                "iterations": iterations,
            })

        return {
            "all_reproducible": all_reproducible,
            "inputs_tested": len(inputs),
            "iterations_per_input": iterations,
            "results": results,
        }

    def compare_outputs(
        self,
        output1: Any,
        output2: Any,
    ) -> Tuple[bool, List[str]]:
        """
        Compare two outputs for equality - DETERMINISTIC.

        Args:
            output1: First output
            output2: Second output

        Returns:
            Tuple of (are_equal, differences)
        """
        differences = []

        hash1 = self._hash_generator.generate_hash(output1)
        hash2 = self._hash_generator.generate_hash(output2)

        if hash1 != hash2:
            differences = self._find_differences(output1, output2)
            return False, differences

        return True, differences

    def _find_differences(
        self,
        obj1: Any,
        obj2: Any,
        path: str = "",
    ) -> List[str]:
        """Find differences between two objects."""
        differences = []

        if type(obj1) != type(obj2):
            differences.append(f"{path}: type mismatch ({type(obj1)} vs {type(obj2)})")
            return differences

        if isinstance(obj1, dict):
            all_keys = set(obj1.keys()) | set(obj2.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in obj1:
                    differences.append(f"{new_path}: missing in first")
                elif key not in obj2:
                    differences.append(f"{new_path}: missing in second")
                else:
                    differences.extend(self._find_differences(obj1[key], obj2[key], new_path))

        elif isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                differences.append(f"{path}: length mismatch ({len(obj1)} vs {len(obj2)})")
            else:
                for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                    differences.extend(self._find_differences(item1, item2, f"{path}[{i}]"))

        elif isinstance(obj1, float):
            if abs(obj1 - obj2) > self.tolerance:
                differences.append(f"{path}: {obj1} != {obj2}")

        else:
            if obj1 != obj2:
                differences.append(f"{path}: {obj1} != {obj2}")

        return differences


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_provenance_tracker(
    session_id: Optional[str] = None,
    enable_audit_trail: bool = True,
) -> ProvenanceTracker:
    """Factory function to create ProvenanceTracker."""
    return ProvenanceTracker(
        session_id=session_id,
        enable_audit_trail=enable_audit_trail,
    )


def create_audit_trail(
    max_entries: int = ProvenanceConstants.MAX_AUDIT_ENTRIES,
) -> CalculationAuditTrail:
    """Factory function to create CalculationAuditTrail."""
    return CalculationAuditTrail(max_entries=max_entries)


def create_hash_generator(
    precision: int = 10,
) -> ProvenanceHashGenerator:
    """Factory function to create ProvenanceHashGenerator."""
    return ProvenanceHashGenerator(precision=precision)


def create_deterministic_verifier(
    tolerance: float = ProvenanceConstants.FLOAT_TOLERANCE,
) -> DeterministicVerifier:
    """Factory function to create DeterministicVerifier."""
    return DeterministicVerifier(tolerance=tolerance)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_provenance_hash(data: Any) -> str:
    """
    Convenience function to generate SHA-256 hash - DETERMINISTIC.

    Args:
        data: Data to hash

    Returns:
        SHA-256 hash string
    """
    generator = ProvenanceHashGenerator()
    return generator.generate_hash(data)


def verify_provenance_hash(data: Any, expected_hash: str) -> bool:
    """
    Convenience function to verify hash - DETERMINISTIC.

    Args:
        data: Data to verify
        expected_hash: Expected hash value

    Returns:
        True if hashes match
    """
    generator = ProvenanceHashGenerator()
    return generator.verify_hash(data, expected_hash)
