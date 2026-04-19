"""
GL-002 FLAMEGUARD - Provenance Tracking

Zero-hallucination calculation provenance with cryptographic verification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    """Single provenance record for audit trail."""
    record_id: str
    timestamp: datetime
    calculation_type: str
    boiler_id: str

    # Input data
    inputs: Dict[str, Any]
    inputs_hash: str

    # Output data
    outputs: Dict[str, Any]
    outputs_hash: str

    # Calculation metadata
    formula_version: str
    standard_reference: str
    emission_factors_version: Optional[str] = None

    # Verification
    combined_hash: str = ""
    verified: bool = False

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "calculation_type": self.calculation_type,
            "boiler_id": self.boiler_id,
            "inputs": self.inputs,
            "inputs_hash": self.inputs_hash,
            "outputs": self.outputs,
            "outputs_hash": self.outputs_hash,
            "formula_version": self.formula_version,
            "standard_reference": self.standard_reference,
            "combined_hash": self.combined_hash,
            "verified": self.verified,
        }


@dataclass
class CalculationProvenance:
    """Detailed calculation provenance for regulatory compliance."""
    calculation_id: str
    calculation_type: str
    timestamp: datetime
    boiler_id: str

    # Input values
    input_values: Dict[str, float] = field(default_factory=dict)

    # Intermediate calculations
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)

    # Output values
    output_values: Dict[str, float] = field(default_factory=dict)

    # Formula metadata
    formulas_used: List[str] = field(default_factory=list)
    emission_factors: Dict[str, float] = field(default_factory=dict)
    emission_factors_source: str = ""

    # Standards
    standard: str = ""
    standard_section: str = ""

    # Hashes
    inputs_hash: str = ""
    outputs_hash: str = ""
    formula_hash: str = ""
    provenance_hash: str = ""

    def add_step(
        self,
        step_name: str,
        formula: str,
        inputs: Dict[str, float],
        result: float,
    ) -> None:
        """Add intermediate calculation step."""
        self.intermediate_steps.append({
            "step_number": len(self.intermediate_steps) + 1,
            "step_name": step_name,
            "formula": formula,
            "inputs": inputs,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def compute_hashes(self) -> None:
        """Compute all provenance hashes."""
        # Input hash
        input_data = json.dumps(self.input_values, sort_keys=True)
        self.inputs_hash = hashlib.sha256(input_data.encode()).hexdigest()

        # Output hash
        output_data = json.dumps(self.output_values, sort_keys=True)
        self.outputs_hash = hashlib.sha256(output_data.encode()).hexdigest()

        # Formula hash
        formula_data = json.dumps(self.formulas_used, sort_keys=True)
        self.formula_hash = hashlib.sha256(formula_data.encode()).hexdigest()

        # Combined provenance hash
        combined = f"{self.inputs_hash}:{self.outputs_hash}:{self.formula_hash}"
        self.provenance_hash = hashlib.sha256(combined.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
            "timestamp": self.timestamp.isoformat(),
            "boiler_id": self.boiler_id,
            "input_values": self.input_values,
            "intermediate_steps": self.intermediate_steps,
            "output_values": self.output_values,
            "formulas_used": self.formulas_used,
            "emission_factors": self.emission_factors,
            "standard": self.standard,
            "standard_section": self.standard_section,
            "hashes": {
                "inputs": self.inputs_hash,
                "outputs": self.outputs_hash,
                "formula": self.formula_hash,
                "provenance": self.provenance_hash,
            },
        }


class ProvenanceTracker:
    """
    Tracks calculation provenance for zero-hallucination compliance.

    Features:
    - Input/output hashing
    - Calculation step tracking
    - Formula versioning
    - Regulatory standard references
    - Cryptographic verification
    """

    def __init__(
        self,
        agent_id: str = "GL-002",
        agent_version: str = "1.0.0",
    ) -> None:
        self.agent_id = agent_id
        self.agent_version = agent_version

        # Records storage
        self._records: Dict[str, ProvenanceRecord] = {}
        self._calculations: Dict[str, CalculationProvenance] = {}

        # Formula versions
        self._formula_versions = {
            "efficiency_indirect": "ASME-PTC-4.1-2023-v1",
            "efficiency_direct": "ASME-PTC-4.1-2023-v1",
            "emissions_nox": "EPA-40CFR60-2024-v1",
            "emissions_co2": "EPA-40CFR60-2024-v1",
            "heat_balance": "ASME-PTC-4.1-2023-v1",
            "o2_correction": "EPA-METHOD-19-v1",
        }

        # Emission factor versions
        self._ef_versions = {
            "natural_gas": "EPA-AP42-CH1.4-2024",
            "fuel_oil": "EPA-AP42-CH1.3-2024",
            "coal": "EPA-AP42-CH1.1-2024",
        }

        logger.info(f"ProvenanceTracker initialized: {agent_id} v{agent_version}")

    def start_calculation(
        self,
        calculation_type: str,
        boiler_id: str,
        inputs: Dict[str, float],
        standard: str = "ASME PTC 4.1",
        standard_section: str = "",
    ) -> CalculationProvenance:
        """Start tracking a calculation."""
        calc_id = str(uuid.uuid4())

        provenance = CalculationProvenance(
            calculation_id=calc_id,
            calculation_type=calculation_type,
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            input_values=dict(inputs),
            standard=standard,
            standard_section=standard_section,
        )

        self._calculations[calc_id] = provenance
        logger.debug(f"Started calculation tracking: {calc_id}")
        return provenance

    def add_calculation_step(
        self,
        calc_id: str,
        step_name: str,
        formula: str,
        inputs: Dict[str, float],
        result: float,
    ) -> None:
        """Add intermediate calculation step."""
        if calc_id not in self._calculations:
            return

        provenance = self._calculations[calc_id]
        provenance.add_step(step_name, formula, inputs, result)
        provenance.formulas_used.append(formula)

    def complete_calculation(
        self,
        calc_id: str,
        outputs: Dict[str, float],
        emission_factors: Optional[Dict[str, float]] = None,
        emission_factors_source: str = "",
    ) -> ProvenanceRecord:
        """Complete calculation and create provenance record."""
        if calc_id not in self._calculations:
            raise ValueError(f"Unknown calculation: {calc_id}")

        provenance = self._calculations[calc_id]
        provenance.output_values = outputs

        if emission_factors:
            provenance.emission_factors = emission_factors
            provenance.emission_factors_source = emission_factors_source

        # Compute hashes
        provenance.compute_hashes()

        # Create provenance record
        record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=provenance.timestamp,
            calculation_type=provenance.calculation_type,
            boiler_id=provenance.boiler_id,
            inputs=provenance.input_values,
            inputs_hash=provenance.inputs_hash,
            outputs=provenance.output_values,
            outputs_hash=provenance.outputs_hash,
            formula_version=self._get_formula_version(provenance.calculation_type),
            standard_reference=f"{provenance.standard} {provenance.standard_section}",
            emission_factors_version=emission_factors_source,
            combined_hash=provenance.provenance_hash,
            verified=True,
        )

        self._records[record.record_id] = record
        logger.info(f"Calculation completed: {calc_id} -> {record.record_id}")

        return record

    def _get_formula_version(self, calculation_type: str) -> str:
        """Get formula version for calculation type."""
        return self._formula_versions.get(calculation_type, "unknown")

    def verify_calculation(
        self,
        record_id: str,
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> bool:
        """Verify a calculation against its provenance record."""
        if record_id not in self._records:
            logger.warning(f"Unknown record: {record_id}")
            return False

        record = self._records[record_id]

        # Recompute input hash
        input_data = json.dumps(inputs, sort_keys=True)
        computed_inputs_hash = hashlib.sha256(input_data.encode()).hexdigest()

        # Recompute output hash
        output_data = json.dumps(outputs, sort_keys=True)
        computed_outputs_hash = hashlib.sha256(output_data.encode()).hexdigest()

        # Verify
        inputs_match = computed_inputs_hash == record.inputs_hash
        outputs_match = computed_outputs_hash == record.outputs_hash

        if inputs_match and outputs_match:
            logger.info(f"Calculation verified: {record_id}")
            return True
        else:
            logger.warning(f"Calculation verification failed: {record_id}")
            if not inputs_match:
                logger.warning("Input hash mismatch")
            if not outputs_match:
                logger.warning("Output hash mismatch")
            return False

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get provenance record by ID."""
        return self._records.get(record_id)

    def get_calculation(self, calc_id: str) -> Optional[CalculationProvenance]:
        """Get calculation provenance by ID."""
        return self._calculations.get(calc_id)

    def get_records_for_boiler(
        self,
        boiler_id: str,
        calculation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProvenanceRecord]:
        """Get provenance records for a boiler."""
        records = [
            r for r in self._records.values()
            if r.boiler_id == boiler_id
        ]

        if calculation_type:
            records = [r for r in records if r.calculation_type == calculation_type]

        # Sort by timestamp descending
        records.sort(key=lambda r: r.timestamp, reverse=True)

        return records[:limit]

    def export_audit_trail(
        self,
        boiler_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """Export audit trail for compliance reporting."""
        records = self.get_records_for_boiler(boiler_id)

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        return [r.to_dict() for r in records]

    def get_statistics(self) -> Dict:
        """Get provenance tracking statistics."""
        calc_types = {}
        for r in self._records.values():
            calc_types[r.calculation_type] = calc_types.get(r.calculation_type, 0) + 1

        return {
            "total_records": len(self._records),
            "total_calculations": len(self._calculations),
            "records_by_type": calc_types,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
        }
