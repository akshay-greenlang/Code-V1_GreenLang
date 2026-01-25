# -*- coding: utf-8 -*-
"""
Provenance - Data lineage and provenance tracking for GL-011 FuelCraft.

This module implements comprehensive provenance tracking including input snapshots,
unit conversion audits, model versions, solver configurations, and output lineage.
All provenance records use SHA-256 hashing for integrity verification.

Key Features:
- Complete data lineage from input to output
- Version tracking for all master data and models
- Unit conversion audit trail
- Solver configuration capture
- Zero-hallucination traceability

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal
import hashlib
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Type of data source."""
    ERP = "erp"
    SCADA = "scada"
    HISTORIAN = "historian"
    MANUAL_ENTRY = "manual"
    API = "api"
    FILE_IMPORT = "file"
    CALCULATED = "calculated"


class VersionType(str, Enum):
    """Type of versioned artifact."""
    MASTER_DATA = "master_data"
    MODEL = "model"
    CONFIGURATION = "configuration"
    FORMULA = "formula"
    CONSTRAINT = "constraint"


class InputSnapshot(BaseModel):
    """
    Snapshot of input data with version tracking.

    Captures the exact state of input data at a point in time
    with cryptographic hash for verification.
    """
    snapshot_id: str = Field(default_factory=lambda: f"SNAP-{uuid.uuid4().hex[:12].upper()}")
    snapshot_name: str = Field(...)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Source information
    source_type: DataSourceType = Field(...)
    source_system: str = Field(...)
    source_query: Optional[str] = Field(None)

    # Version information
    data_version: Optional[str] = Field(None)
    data_effective_date: Optional[datetime] = Field(None)

    # Content
    record_count: int = Field(..., ge=0)
    schema_hash: str = Field(...)
    content_hash: str = Field(...)

    # Data quality
    completeness_pct: float = Field(100.0, ge=0, le=100)
    quality_issues: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_hash(self, data: Any) -> str:
        """Calculate content hash for data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class ConversionRecord(BaseModel):
    """Record of a single unit conversion."""
    conversion_id: str = Field(default_factory=lambda: f"CONV-{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Values
    input_value: float = Field(...)
    input_unit: str = Field(...)
    output_value: float = Field(...)
    output_unit: str = Field(...)

    # Conversion details
    conversion_factor: float = Field(...)
    conversion_formula: str = Field(...)
    formula_version: str = Field(...)

    # Context
    field_name: str = Field(...)
    record_id: Optional[str] = Field(None)

    # Provenance
    provenance_hash: str = Field(...)


class ConversionAuditTrail(BaseModel):
    """
    Complete audit trail of unit conversions for a run.

    Captures all unit conversions performed during processing
    for complete traceability.
    """
    trail_id: str = Field(default_factory=lambda: f"TRAIL-{uuid.uuid4().hex[:12].upper()}")
    run_id: str = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Conversions
    conversions: List[ConversionRecord] = Field(default_factory=list)
    total_conversions: int = Field(0)

    # Statistics
    unique_units: List[str] = Field(default_factory=list)
    conversion_types: Dict[str, int] = Field(default_factory=dict)

    # Integrity
    trail_hash: Optional[str] = Field(None)

    def add_conversion(
        self,
        input_value: float,
        input_unit: str,
        output_value: float,
        output_unit: str,
        conversion_factor: float,
        conversion_formula: str,
        formula_version: str,
        field_name: str,
        record_id: Optional[str] = None
    ) -> ConversionRecord:
        """Add a conversion record to the trail."""
        provenance_hash = hashlib.sha256(
            json.dumps({
                "input": input_value, "input_unit": input_unit,
                "output": output_value, "output_unit": output_unit,
                "factor": conversion_factor
            }, sort_keys=True).encode()
        ).hexdigest()

        record = ConversionRecord(
            input_value=input_value,
            input_unit=input_unit,
            output_value=output_value,
            output_unit=output_unit,
            conversion_factor=conversion_factor,
            conversion_formula=conversion_formula,
            formula_version=formula_version,
            field_name=field_name,
            record_id=record_id,
            provenance_hash=provenance_hash
        )

        self.conversions.append(record)
        self.total_conversions = len(self.conversions)

        # Track unique units
        for unit in [input_unit, output_unit]:
            if unit not in self.unique_units:
                self.unique_units.append(unit)

        # Track conversion types
        conv_type = f"{input_unit}_to_{output_unit}"
        self.conversion_types[conv_type] = self.conversion_types.get(conv_type, 0) + 1

        return record

    def seal(self) -> str:
        """Seal the trail and calculate final hash."""
        conv_hashes = [c.provenance_hash for c in self.conversions]
        self.trail_hash = hashlib.sha256(
            "|".join(sorted(conv_hashes)).encode()
        ).hexdigest()
        return self.trail_hash


class ModelVersionRecord(BaseModel):
    """
    Record of a model version used in processing.

    Captures the exact version and configuration of any model
    (ML, optimization, rules engine) used during a run.
    """
    record_id: str = Field(default_factory=lambda: f"MODEL-{uuid.uuid4().hex[:12].upper()}")
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Model identification
    model_name: str = Field(...)
    model_type: str = Field(...)  # ml_classifier, optimizer, rules_engine, etc.
    model_version: str = Field(...)
    model_artifact_hash: str = Field(...)

    # Training/Configuration
    trained_at: Optional[datetime] = Field(None)
    training_data_hash: Optional[str] = Field(None)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    # Performance metrics at time of use
    validation_metrics: Dict[str, float] = Field(default_factory=dict)

    # Environment
    framework: str = Field(...)  # scikit-learn, pytorch, pulp, etc.
    framework_version: str = Field(...)
    python_version: str = Field(...)

    # Provenance
    provenance_hash: str = Field(...)

    @classmethod
    def create(
        cls,
        model_name: str,
        model_type: str,
        model_version: str,
        model_artifact_hash: str,
        framework: str,
        framework_version: str,
        python_version: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> "ModelVersionRecord":
        """Factory method to create a model version record."""
        provenance_hash = hashlib.sha256(
            json.dumps({
                "name": model_name,
                "version": model_version,
                "artifact_hash": model_artifact_hash,
                "framework": framework
            }, sort_keys=True).encode()
        ).hexdigest()

        return cls(
            model_name=model_name,
            model_type=model_type,
            model_version=model_version,
            model_artifact_hash=model_artifact_hash,
            framework=framework,
            framework_version=framework_version,
            python_version=python_version,
            hyperparameters=hyperparameters or {},
            validation_metrics=validation_metrics or {},
            provenance_hash=provenance_hash
        )


class SolverConfigRecord(BaseModel):
    """
    Record of solver configuration used in optimization.

    Captures complete solver settings for reproducibility.
    """
    record_id: str = Field(default_factory=lambda: f"SOLVER-{uuid.uuid4().hex[:12].upper()}")
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Solver identification
    solver_name: str = Field(...)
    solver_version: str = Field(...)
    solver_type: str = Field(...)  # LP, MIP, QP, NLP, etc.

    # Configuration
    objective: str = Field(...)  # minimize_cost, maximize_efficiency, etc.
    time_limit_seconds: Optional[float] = Field(None)
    gap_tolerance: Optional[float] = Field(None)
    iteration_limit: Optional[int] = Field(None)

    # Tolerances
    optimality_tolerance: float = Field(1e-6)
    feasibility_tolerance: float = Field(1e-6)
    integer_tolerance: float = Field(1e-5)

    # Solver-specific options
    solver_options: Dict[str, Any] = Field(default_factory=dict)

    # Problem statistics (captured after solve)
    num_variables: Optional[int] = Field(None)
    num_constraints: Optional[int] = Field(None)
    num_integers: Optional[int] = Field(None)

    # Provenance
    config_hash: str = Field(...)

    @classmethod
    def create(
        cls,
        solver_name: str,
        solver_version: str,
        solver_type: str,
        objective: str,
        time_limit_seconds: Optional[float] = None,
        gap_tolerance: Optional[float] = None,
        solver_options: Optional[Dict[str, Any]] = None
    ) -> "SolverConfigRecord":
        """Factory method to create a solver config record."""
        config_data = {
            "solver": solver_name,
            "version": solver_version,
            "type": solver_type,
            "objective": objective,
            "options": solver_options or {}
        }
        config_hash = hashlib.sha256(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()

        return cls(
            solver_name=solver_name,
            solver_version=solver_version,
            solver_type=solver_type,
            objective=objective,
            time_limit_seconds=time_limit_seconds,
            gap_tolerance=gap_tolerance,
            solver_options=solver_options or {},
            config_hash=config_hash
        )


class OutputLineage(BaseModel):
    """
    Complete lineage for an output value.

    Traces an output back to its input sources, transformations,
    and calculations for complete auditability.
    """
    lineage_id: str = Field(default_factory=lambda: f"LINEAGE-{uuid.uuid4().hex[:12].upper()}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Output identification
    output_name: str = Field(...)
    output_value: Any = Field(...)
    output_type: str = Field(...)
    output_hash: str = Field(...)

    # Input sources
    input_snapshots: List[str] = Field(default_factory=list)  # Snapshot IDs
    input_hashes: Dict[str, str] = Field(default_factory=dict)

    # Transformations
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    conversion_trail_id: Optional[str] = Field(None)

    # Calculations
    calculation_steps: List[Dict[str, Any]] = Field(default_factory=list)
    formulas_used: List[str] = Field(default_factory=list)

    # Models and solvers
    model_versions: List[str] = Field(default_factory=list)  # Model record IDs
    solver_config_id: Optional[str] = Field(None)

    # Provenance
    lineage_hash: str = Field(...)

    def add_input(self, snapshot_id: str, content_hash: str) -> None:
        """Add an input source to the lineage."""
        self.input_snapshots.append(snapshot_id)
        self.input_hashes[snapshot_id] = content_hash

    def add_transformation(
        self,
        transformation_type: str,
        description: str,
        input_fields: List[str],
        output_fields: List[str]
    ) -> None:
        """Add a transformation step."""
        self.transformations.append({
            "type": transformation_type,
            "description": description,
            "inputs": input_fields,
            "outputs": output_fields,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def add_calculation(
        self,
        formula_id: str,
        formula_expression: str,
        inputs: Dict[str, Any],
        result: Any
    ) -> None:
        """Add a calculation step."""
        self.calculation_steps.append({
            "formula_id": formula_id,
            "expression": formula_expression,
            "inputs": inputs,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if formula_id not in self.formulas_used:
            self.formulas_used.append(formula_id)

    def finalize(self) -> str:
        """Finalize lineage and calculate hash."""
        lineage_data = {
            "output_hash": self.output_hash,
            "input_hashes": sorted(self.input_hashes.values()),
            "transformations": len(self.transformations),
            "calculations": len(self.calculation_steps)
        }
        self.lineage_hash = hashlib.sha256(
            json.dumps(lineage_data, sort_keys=True).encode()
        ).hexdigest()
        return self.lineage_hash


class ProvenanceTracker:
    """
    Central provenance tracker for a processing run.

    Coordinates all provenance tracking components to provide
    complete data lineage from inputs to outputs.
    """

    def __init__(self, run_id: str):
        """Initialize provenance tracker."""
        self._run_id = run_id
        self._snapshots: Dict[str, InputSnapshot] = {}
        self._conversion_trail = ConversionAuditTrail(run_id=run_id)
        self._model_versions: Dict[str, ModelVersionRecord] = {}
        self._solver_configs: Dict[str, SolverConfigRecord] = {}
        self._lineages: Dict[str, OutputLineage] = {}

        logger.info(f"ProvenanceTracker initialized for run {run_id}")

    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self._run_id

    def capture_input(
        self,
        name: str,
        data: Any,
        source_type: DataSourceType,
        source_system: str,
        data_version: Optional[str] = None,
        source_query: Optional[str] = None
    ) -> InputSnapshot:
        """
        Capture an input snapshot.

        Args:
            name: Snapshot name
            data: Input data
            source_type: Type of data source
            source_system: Name of source system
            data_version: Version of the data
            source_query: Query used to retrieve data

        Returns:
            InputSnapshot record
        """
        # Calculate hashes
        json_data = json.dumps(data, sort_keys=True, default=str)
        content_hash = hashlib.sha256(json_data.encode()).hexdigest()

        # Calculate schema hash (simplified - just keys for dict/list of dicts)
        if isinstance(data, dict):
            schema_hash = hashlib.sha256(
                json.dumps(sorted(data.keys())).encode()
            ).hexdigest()
            record_count = 1
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                schema_hash = hashlib.sha256(
                    json.dumps(sorted(data[0].keys())).encode()
                ).hexdigest()
            else:
                schema_hash = hashlib.sha256(b"list").hexdigest()
            record_count = len(data)
        else:
            schema_hash = hashlib.sha256(str(type(data)).encode()).hexdigest()
            record_count = 1

        snapshot = InputSnapshot(
            snapshot_name=name,
            source_type=source_type,
            source_system=source_system,
            source_query=source_query,
            data_version=data_version,
            record_count=record_count,
            schema_hash=schema_hash,
            content_hash=content_hash
        )

        self._snapshots[snapshot.snapshot_id] = snapshot

        logger.info(f"Captured input snapshot: {name} ({record_count} records)")
        return snapshot

    def record_conversion(
        self,
        input_value: float,
        input_unit: str,
        output_value: float,
        output_unit: str,
        conversion_factor: float,
        conversion_formula: str,
        formula_version: str,
        field_name: str,
        record_id: Optional[str] = None
    ) -> ConversionRecord:
        """Record a unit conversion."""
        return self._conversion_trail.add_conversion(
            input_value=input_value,
            input_unit=input_unit,
            output_value=output_value,
            output_unit=output_unit,
            conversion_factor=conversion_factor,
            conversion_formula=conversion_formula,
            formula_version=formula_version,
            field_name=field_name,
            record_id=record_id
        )

    def register_model(
        self,
        model_name: str,
        model_type: str,
        model_version: str,
        model_artifact_hash: str,
        framework: str,
        framework_version: str,
        python_version: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> ModelVersionRecord:
        """Register a model version."""
        record = ModelVersionRecord.create(
            model_name=model_name,
            model_type=model_type,
            model_version=model_version,
            model_artifact_hash=model_artifact_hash,
            framework=framework,
            framework_version=framework_version,
            python_version=python_version,
            hyperparameters=hyperparameters
        )

        self._model_versions[record.record_id] = record

        logger.info(f"Registered model: {model_name} v{model_version}")
        return record

    def register_solver(
        self,
        solver_name: str,
        solver_version: str,
        solver_type: str,
        objective: str,
        time_limit_seconds: Optional[float] = None,
        gap_tolerance: Optional[float] = None,
        solver_options: Optional[Dict[str, Any]] = None
    ) -> SolverConfigRecord:
        """Register a solver configuration."""
        record = SolverConfigRecord.create(
            solver_name=solver_name,
            solver_version=solver_version,
            solver_type=solver_type,
            objective=objective,
            time_limit_seconds=time_limit_seconds,
            gap_tolerance=gap_tolerance,
            solver_options=solver_options
        )

        self._solver_configs[record.record_id] = record

        logger.info(f"Registered solver: {solver_name} v{solver_version}")
        return record

    def create_output_lineage(
        self,
        output_name: str,
        output_value: Any,
        output_type: str
    ) -> OutputLineage:
        """Create an output lineage record."""
        output_hash = hashlib.sha256(
            json.dumps(output_value, sort_keys=True, default=str).encode()
        ).hexdigest()

        lineage = OutputLineage(
            output_name=output_name,
            output_value=output_value,
            output_type=output_type,
            output_hash=output_hash,
            lineage_hash=""
        )

        self._lineages[lineage.lineage_id] = lineage

        return lineage

    def get_full_provenance(self) -> Dict[str, Any]:
        """Get complete provenance record for the run."""
        self._conversion_trail.seal()

        return {
            "run_id": self._run_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "snapshots": {k: v.model_dump() for k, v in self._snapshots.items()},
            "conversions": self._conversion_trail.model_dump(),
            "models": {k: v.model_dump() for k, v in self._model_versions.items()},
            "solvers": {k: v.model_dump() for k, v in self._solver_configs.items()},
            "lineages": {k: v.model_dump() for k, v in self._lineages.items()},
            "provenance_hash": self._calculate_provenance_hash()
        }

    def _calculate_provenance_hash(self) -> str:
        """Calculate overall provenance hash."""
        components = [
            self._run_id,
            "|".join(sorted(s.content_hash for s in self._snapshots.values())),
            self._conversion_trail.trail_hash or "",
            "|".join(sorted(m.provenance_hash for m in self._model_versions.values())),
            "|".join(sorted(s.config_hash for s in self._solver_configs.values()))
        ]
        return hashlib.sha256("|".join(components).encode()).hexdigest()
