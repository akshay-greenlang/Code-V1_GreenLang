# -*- coding: utf-8 -*-
"""
Provenance Tracking Utilities for GL-016 Waterguard Thermal Calculations

This module provides cryptographic provenance tracking for all thermal calculations,
ensuring complete audit trails and bit-perfect reproducibility.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standard: SHA-256 cryptographic hashing
Agent: GL-016_Waterguard
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
from dataclasses import dataclass, field, asdict


def _get_utc_now() -> datetime:
    """Get current UTC time in a deterministic manner."""
    return datetime.now(timezone.utc)


@dataclass
class CalculationStep:
    """Represents a single step in a calculation chain."""
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    timestamp: str = field(default_factory=lambda: _get_utc_now().isoformat())
    formula: Optional[str] = None
    units: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        for key, value in data['inputs'].items():
            if isinstance(value, Decimal):
                data['inputs'][key] = str(value)
        if isinstance(data['output_value'], Decimal):
            data['output_value'] = str(data['output_value'])
        return data


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a calculation."""
    calculation_id: str
    calculation_type: str
    version: str
    input_parameters: Dict[str, Any]
    calculation_steps: List[CalculationStep]
    final_result: Any
    provenance_hash: str
    timestamp: str = field(default_factory=lambda: _get_utc_now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    standard: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = {
            'calculation_id': self.calculation_id,
            'calculation_type': self.calculation_type,
            'version': self.version,
            'input_parameters': {},
            'calculation_steps': [step.to_dict() for step in self.calculation_steps],
            'final_result': str(self.final_result) if isinstance(self.final_result, Decimal) else self.final_result,
            'provenance_hash': self.provenance_hash,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'standard': self.standard
        }
        for key, value in self.input_parameters.items():
            if isinstance(value, Decimal):
                data['input_parameters'][key] = str(value)
            else:
                data['input_parameters'][key] = value
        return data


class ProvenanceTracker:
    """Tracks calculation provenance with SHA-256 hashing."""

    def __init__(
        self,
        calculation_id: str,
        calculation_type: str,
        version: str = "1.0.0",
        standard: Optional[str] = None
    ):
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.version = version
        self.standard = standard
        self.steps: List[CalculationStep] = []
        self.input_parameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def record_inputs(self, parameters: Dict[str, Any]) -> None:
        self.input_parameters = self._normalize_values(parameters.copy())

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def record_step(
        self,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_value: Any,
        output_name: str,
        formula: Optional[str] = None,
        units: Optional[str] = None,
        source: Optional[str] = None
    ) -> CalculationStep:
        step = CalculationStep(
            step_number=len(self.steps) + 1,
            operation=operation,
            description=description,
            inputs=self._normalize_values(inputs),
            output_value=self._normalize_value(output_value),
            output_name=output_name,
            formula=formula,
            units=units,
            source=source
        )
        self.steps.append(step)
        return step

    def generate_hash(self, final_result: Any) -> str:
        canonical_data = {
            'calculation_id': self.calculation_id,
            'calculation_type': self.calculation_type,
            'version': self.version,
            'standard': self.standard or '',
            'input_parameters': self._serialize_for_hash(self.input_parameters),
            'steps': [self._serialize_step_for_hash(step) for step in self.steps],
            'final_result': self._serialize_for_hash(final_result)
        }
        canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def get_provenance_record(self, final_result: Any) -> ProvenanceRecord:
        provenance_hash = self.generate_hash(final_result)
        return ProvenanceRecord(
            calculation_id=self.calculation_id,
            calculation_type=self.calculation_type,
            version=self.version,
            input_parameters=self.input_parameters,
            calculation_steps=self.steps,
            final_result=self._normalize_value(final_result),
            provenance_hash=provenance_hash,
            metadata=self.metadata,
            standard=self.standard
        )

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return Decimal(str(value))
        elif isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        return value

    def _normalize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._normalize_value(v) for k, v in values.items()}

    def _serialize_for_hash(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(Decimal(str(value)))
        elif isinstance(value, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            return [self._serialize_for_hash(v) for v in value]
        elif isinstance(value, bool):
            return value
        return str(value) if value is not None else ''

    def _serialize_step_for_hash(self, step: CalculationStep) -> Dict:
        return {
            'step_number': step.step_number,
            'operation': step.operation,
            'description': step.description,
            'inputs': self._serialize_for_hash(step.inputs),
            'output_value': self._serialize_for_hash(step.output_value),
            'output_name': step.output_name,
            'formula': step.formula or '',
            'units': step.units or '',
            'source': step.source or ''
        }


class ProvenanceValidator:
    """Validates calculation provenance for integrity."""

    @staticmethod
    def validate_hash(record: ProvenanceRecord) -> bool:
        tracker = ProvenanceTracker(
            record.calculation_id,
            record.calculation_type,
            record.version,
            record.standard
        )
        tracker.input_parameters = record.input_parameters
        tracker.steps = record.calculation_steps
        tracker.metadata = record.metadata
        expected_hash = tracker.generate_hash(record.final_result)
        return expected_hash == record.provenance_hash

    @staticmethod
    def validate_reproducibility(
        record1: ProvenanceRecord,
        record2: ProvenanceRecord
    ) -> bool:
        if record1.input_parameters != record2.input_parameters:
            return False
        if record1.calculation_type != record2.calculation_type:
            return False
        if str(record1.final_result) != str(record2.final_result):
            return False
        return record1.provenance_hash == record2.provenance_hash


def create_calculation_hash(data: Dict[str, Any]) -> str:
    def serialize(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        elif isinstance(value, dict):
            return {k: serialize(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            return [serialize(v) for v in value]
        return value

    canonical_json = json.dumps(serialize(data), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
