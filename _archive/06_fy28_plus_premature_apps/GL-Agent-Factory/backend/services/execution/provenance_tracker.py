"""
Provenance Tracker

This module provides SHA-256 based provenance tracking for all calculations
and data transformations, ensuring complete audit trails for regulatory
compliance.

The ProvenanceTracker implements:
- Input/output hashing with SHA-256
- Calculation chain tracking
- Provenance verification
- Audit trail export

Example:
    >>> tracker = ProvenanceTracker()
    >>> input_hash = tracker.track_input({"fuel_type": "natural_gas", "quantity": 1000})
    >>> output_hash = tracker.track_output({"emissions": 2500.5})
    >>> chain_hash = tracker.build_provenance_chain([input_hash, output_hash])
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProvenanceStep(BaseModel):
    """A single step in the provenance chain."""

    step_type: str = Field(..., description="Type of step (input, calculation, output)")
    step_id: str = Field(..., description="Unique step identifier")
    input_hash: Optional[str] = Field(None, description="Hash of step input")
    output_hash: Optional[str] = Field(None, description="Hash of step output")
    formula_id: Optional[str] = Field(None, description="Formula used (if calculation)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    """Complete provenance record for an execution."""

    execution_id: str = Field(..., description="Execution this provenance belongs to")
    chain_hash: str = Field(..., description="Final hash of provenance chain")
    steps: List[ProvenanceStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def verify(self) -> bool:
        """Verify the integrity of the provenance chain."""
        if not self.steps:
            return False

        # Rebuild chain hash and compare
        step_data = [step.dict() for step in self.steps]
        recalculated = ProvenanceTracker._compute_hash(step_data)
        return recalculated == self.chain_hash


class ProvenanceTracker:
    """
    Provenance Tracker for zero-hallucination audit trails.

    This class provides comprehensive provenance tracking with SHA-256
    hashing for all inputs, outputs, and calculations. It enables:
    - Complete reproducibility of calculations
    - Regulatory audit compliance
    - Data integrity verification

    Attributes:
        steps: List of tracked provenance steps
        storage: Optional persistence layer

    Example:
        >>> tracker = ProvenanceTracker()
        >>> input_hash = tracker.track_input({"value": 100})
        >>> calc_hash = tracker.track_calculation("emissions_calc", {"value": 100}, 250.0)
        >>> output_hash = tracker.track_output({"emissions": 250.0})
        >>> chain = tracker.build_provenance_chain([input_hash, calc_hash, output_hash])
    """

    def __init__(self, storage: Optional[Any] = None):
        """
        Initialize the ProvenanceTracker.

        Args:
            storage: Optional storage backend for persisting provenance records
        """
        self.storage = storage
        self._current_steps: List[ProvenanceStep] = []
        self._step_counter = 0

    def track_input(self, input_data: Any) -> str:
        """
        Track input data and return its hash.

        Args:
            input_data: Input data to track (any JSON-serializable value)

        Returns:
            SHA-256 hash of the input data

        Example:
            >>> hash = tracker.track_input({"fuel_type": "gas", "quantity": 1000})
            >>> len(hash) == 64  # SHA-256 produces 64 hex characters
            True
        """
        input_hash = self._compute_hash(input_data)

        step = ProvenanceStep(
            step_type="input",
            step_id=self._generate_step_id(),
            input_hash=None,
            output_hash=input_hash,
            metadata={"data_type": type(input_data).__name__},
        )
        self._current_steps.append(step)

        logger.debug(f"Tracked input with hash: {input_hash[:16]}...")
        return input_hash

    def track_output(self, output_data: Any) -> str:
        """
        Track output data and return its hash.

        Args:
            output_data: Output data to track (any JSON-serializable value)

        Returns:
            SHA-256 hash of the output data

        Example:
            >>> hash = tracker.track_output({"emissions_kgco2e": 2500.5})
        """
        output_hash = self._compute_hash(output_data)

        step = ProvenanceStep(
            step_type="output",
            step_id=self._generate_step_id(),
            input_hash=None,
            output_hash=output_hash,
            metadata={"data_type": type(output_data).__name__},
        )
        self._current_steps.append(step)

        logger.debug(f"Tracked output with hash: {output_hash[:16]}...")
        return output_hash

    def track_calculation(
        self,
        formula_id: str,
        inputs: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Track a calculation step with its inputs, formula, and result.

        This method creates a provenance record for deterministic calculations,
        enabling complete reproducibility and audit.

        Args:
            formula_id: Identifier of the formula used
            inputs: Input parameters for the calculation
            result: Calculation result
            metadata: Optional additional metadata

        Returns:
            SHA-256 hash of the calculation record

        Example:
            >>> hash = tracker.track_calculation(
            ...     formula_id="emissions.scope1.stationary",
            ...     inputs={"quantity": 1000, "ef": 2.5},
            ...     result=2500.0
            ... )
        """
        calc_record = {
            "formula_id": formula_id,
            "inputs": inputs,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

        input_hash = self._compute_hash(inputs)
        output_hash = self._compute_hash(result)
        calc_hash = self._compute_hash(calc_record)

        step = ProvenanceStep(
            step_type="calculation",
            step_id=self._generate_step_id(),
            input_hash=input_hash,
            output_hash=output_hash,
            formula_id=formula_id,
            metadata=metadata or {},
        )
        self._current_steps.append(step)

        logger.debug(
            f"Tracked calculation {formula_id} with hash: {calc_hash[:16]}..."
        )
        return calc_hash

    def track_emission_factor_lookup(
        self,
        material_id: str,
        region: str,
        year: int,
        factor_value: float,
        source: str,
    ) -> str:
        """
        Track an emission factor lookup.

        Args:
            material_id: Material identifier
            region: Geographic region
            year: Reference year
            factor_value: Retrieved emission factor value
            source: Source database (EPA, IPCC, etc.)

        Returns:
            SHA-256 hash of the lookup record
        """
        lookup_record = {
            "lookup_type": "emission_factor",
            "material_id": material_id,
            "region": region,
            "year": year,
            "factor_value": factor_value,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }

        lookup_hash = self._compute_hash(lookup_record)

        step = ProvenanceStep(
            step_type="lookup",
            step_id=self._generate_step_id(),
            output_hash=lookup_hash,
            metadata={
                "material_id": material_id,
                "region": region,
                "source": source,
            },
        )
        self._current_steps.append(step)

        logger.debug(f"Tracked EF lookup for {material_id}: {factor_value}")
        return lookup_hash

    def build_provenance_chain(
        self,
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a provenance chain hash from all tracked steps.

        This creates a single hash that represents the entire execution
        history, enabling verification of the complete audit trail.

        Args:
            steps: Optional list of step dictionaries. If None, uses internal steps.

        Returns:
            SHA-256 hash of the complete provenance chain

        Example:
            >>> chain_hash = tracker.build_provenance_chain()
            >>> len(chain_hash) == 64
            True
        """
        if steps is None:
            step_data = [step.dict() for step in self._current_steps]
        else:
            step_data = steps

        chain_hash = self._compute_hash(step_data)

        logger.info(f"Built provenance chain with {len(step_data)} steps: {chain_hash[:16]}...")
        return chain_hash

    def verify_provenance(self, record: ProvenanceRecord) -> bool:
        """
        Verify a provenance record's integrity.

        Args:
            record: Provenance record to verify

        Returns:
            True if the record is valid, False otherwise
        """
        return record.verify()

    def get_current_record(self, execution_id: str) -> ProvenanceRecord:
        """
        Get the current provenance record.

        Args:
            execution_id: Execution ID for this record

        Returns:
            Complete provenance record
        """
        chain_hash = self.build_provenance_chain()

        return ProvenanceRecord(
            execution_id=execution_id,
            chain_hash=chain_hash,
            steps=self._current_steps.copy(),
        )

    def reset(self) -> None:
        """Reset the tracker for a new execution."""
        self._current_steps = []
        self._step_counter = 0

    def export_audit_trail(self, format: str = "json") -> Union[str, Dict]:
        """
        Export the provenance chain as an audit trail.

        Args:
            format: Export format ("json" or "dict")

        Returns:
            Audit trail in requested format
        """
        audit_data = {
            "provenance_chain": self.build_provenance_chain(),
            "step_count": len(self._current_steps),
            "steps": [step.dict() for step in self._current_steps],
            "exported_at": datetime.utcnow().isoformat(),
        }

        if format == "json":
            return json.dumps(audit_data, indent=2, default=str)
        return audit_data

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """
        Compute SHA-256 hash of any JSON-serializable data.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hex digest
        """
        # Serialize to canonical JSON (sorted keys for consistency)
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def _generate_step_id(self) -> str:
        """Generate a unique step identifier."""
        self._step_counter += 1
        return f"step-{self._step_counter:04d}"
