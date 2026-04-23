"""
Evidence Pack Generator for GL-001 ThermalCommand

This module implements per-decision evidence pack generation as specified in
Appendix B. Evidence packs provide comprehensive documentation of every
optimization decision for regulatory compliance and audit purposes.

Evidence Pack Contents:
    - Correlation ID and timestamps (ingestion, decision, actuation)
    - Input dataset references (hashes/IDs), schema versions
    - Unit conversion version
    - Constraint set and safety boundary policy version
    - ML model versions (demand, health, anomaly)
    - SHAP/LIME artifacts and UQ intervals
    - Solver status (optimal/feasible/infeasible)
    - Objective breakdown, binding constraints summary
    - Recommended actions (tag, value, bounds), ramps
    - Expected impact (cost/emissions/risk)
    - Operator action taken

Example:
    >>> generator = EvidencePackGenerator(storage_path="/audit/evidence")
    >>> pack = generator.generate(
    ...     decision_event=decision_event,
    ...     action_events=action_events,
    ...     lineage_graph=lineage_graph
    ... )
    >>> pack_id = generator.store(pack)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from datetime import datetime, timezone, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .audit_events import (
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    RecommendedAction,
    ExpectedImpact,
    ConstraintInfo,
    ExplainabilityArtifact,
    UncertaintyQuantification,
    ModelVersionInfo,
    InputDatasetReference,
    SolverStatus,
    ActionStatus,
)
from .provenance_enhanced import (
    LineageGraph,
    ProvenanceNode,
    ModelVersionRecord,
    EnhancedProvenanceTracker,
)

logger = logging.getLogger(__name__)


class EvidencePackStatus(str, Enum):
    """Status of evidence pack."""

    DRAFT = "DRAFT"
    COMPLETE = "COMPLETE"
    SEALED = "SEALED"
    ARCHIVED = "ARCHIVED"


class EvidencePackFormat(str, Enum):
    """Output format for evidence pack."""

    JSON = "JSON"
    ZIP = "ZIP"
    PDF = "PDF"


class TimestampRecord(BaseModel):
    """Record of key timestamps in the decision lifecycle."""

    ingestion_timestamp: datetime = Field(..., description="When data was ingested")
    preprocessing_timestamp: Optional[datetime] = Field(
        None, description="When preprocessing completed"
    )
    inference_timestamp: Optional[datetime] = Field(
        None, description="When ML inference completed"
    )
    optimization_timestamp: Optional[datetime] = Field(
        None, description="When optimization completed"
    )
    decision_timestamp: datetime = Field(..., description="When decision was made")
    recommendation_timestamp: datetime = Field(..., description="When recommendation sent")
    operator_review_timestamp: Optional[datetime] = Field(
        None, description="When operator reviewed"
    )
    approval_timestamp: Optional[datetime] = Field(None, description="When action approved")
    actuation_timestamp: Optional[datetime] = Field(None, description="When action executed")
    verification_timestamp: Optional[datetime] = Field(
        None, description="When action verified"
    )

    @property
    def total_latency_ms(self) -> float:
        """Calculate total latency from ingestion to decision."""
        return (self.decision_timestamp - self.ingestion_timestamp).total_seconds() * 1000

    @property
    def actuation_latency_ms(self) -> Optional[float]:
        """Calculate latency from decision to actuation."""
        if self.actuation_timestamp:
            return (self.actuation_timestamp - self.decision_timestamp).total_seconds() * 1000
        return None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DatasetSummary(BaseModel):
    """Summary of input datasets used in decision."""

    datasets: List[InputDatasetReference] = Field(
        default_factory=list, description="Input datasets"
    )
    total_records: int = Field(0, ge=0, description="Total records across datasets")
    time_range_start: Optional[datetime] = Field(None, description="Earliest data timestamp")
    time_range_end: Optional[datetime] = Field(None, description="Latest data timestamp")
    combined_hash: str = Field(..., description="Combined hash of all datasets")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ModelVersionSummary(BaseModel):
    """Summary of ML models used in decision."""

    demand_forecast_model: Optional[ModelVersionInfo] = Field(
        None, description="Demand forecast model"
    )
    equipment_health_model: Optional[ModelVersionInfo] = Field(
        None, description="Equipment health model"
    )
    anomaly_detection_model: Optional[ModelVersionInfo] = Field(
        None, description="Anomaly detection model"
    )
    price_forecast_model: Optional[ModelVersionInfo] = Field(
        None, description="Price forecast model"
    )
    additional_models: Dict[str, ModelVersionInfo] = Field(
        default_factory=dict, description="Any additional models"
    )


class ConstraintSummary(BaseModel):
    """Summary of optimization constraints."""

    constraint_set_id: str = Field(..., description="Constraint set identifier")
    constraint_set_version: str = Field(..., description="Constraint set version")
    total_constraints: int = Field(0, ge=0, description="Total constraints")
    binding_constraints: List[ConstraintInfo] = Field(
        default_factory=list, description="Active constraints"
    )
    violated_constraints: List[ConstraintInfo] = Field(
        default_factory=list, description="Violated constraints"
    )
    safety_boundary_policy_version: str = Field(
        ..., description="Safety boundary version"
    )


class SolverSummary(BaseModel):
    """Summary of optimization solver results."""

    solver_name: str = Field(..., description="Solver used")
    solver_version: str = Field(..., description="Solver version")
    solver_status: SolverStatus = Field(..., description="Termination status")
    solve_time_ms: float = Field(..., ge=0, description="Solve time in ms")
    mip_gap: Optional[float] = Field(None, ge=0, description="MIP optimality gap")
    iterations: Optional[int] = Field(None, ge=0, description="Solver iterations")
    objective_value: float = Field(..., description="Optimal objective value")
    objective_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Objective components"
    )


class ExplainabilitySummary(BaseModel):
    """Summary of explainability artifacts."""

    shap_available: bool = Field(False, description="SHAP explanations available")
    lime_available: bool = Field(False, description="LIME explanations available")
    shap_artifacts: List[ExplainabilityArtifact] = Field(
        default_factory=list, description="SHAP artifacts"
    )
    lime_artifacts: List[ExplainabilityArtifact] = Field(
        default_factory=list, description="LIME artifacts"
    )
    uncertainty_quantification: Dict[str, UncertaintyQuantification] = Field(
        default_factory=dict, description="UQ intervals"
    )
    top_features: Dict[str, float] = Field(
        default_factory=dict, description="Top influential features"
    )


class ActionSummary(BaseModel):
    """Summary of recommended and executed actions."""

    total_actions: int = Field(0, ge=0, description="Total recommended actions")
    actions: List[RecommendedAction] = Field(
        default_factory=list, description="Recommended actions"
    )
    approved_count: int = Field(0, ge=0, description="Approved actions")
    rejected_count: int = Field(0, ge=0, description="Rejected actions")
    executed_count: int = Field(0, ge=0, description="Executed actions")
    modified_count: int = Field(0, ge=0, description="Modified actions")
    pending_count: int = Field(0, ge=0, description="Pending actions")


class OperatorActionRecord(BaseModel):
    """Record of operator actions taken."""

    operator_id: str = Field(..., description="Operator identifier")
    action_type: str = Field(..., description="Type of action taken")
    timestamp: datetime = Field(..., description="When action was taken")
    original_recommendation: Optional[Dict[str, Any]] = Field(
        None, description="Original recommendation"
    )
    modified_values: Optional[Dict[str, Any]] = Field(None, description="Modified values")
    notes: Optional[str] = Field(None, description="Operator notes")
    authorization_level: str = Field(..., description="Authorization level")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ImpactSummary(BaseModel):
    """Summary of expected and actual impacts."""

    expected_impact: Optional[ExpectedImpact] = Field(
        None, description="Expected impact"
    )
    actual_impact: Optional[ExpectedImpact] = Field(
        None, description="Actual measured impact"
    )
    variance_cost_pct: Optional[float] = Field(None, description="Cost variance %")
    variance_emissions_pct: Optional[float] = Field(None, description="Emissions variance %")


class EvidencePack(BaseModel):
    """
    Complete evidence pack for a decision.

    This is the main output of the EvidencePackGenerator, containing
    all information required for regulatory compliance and audit.
    """

    # Identification
    pack_id: UUID = Field(default_factory=uuid4, description="Unique pack identifier")
    correlation_id: str = Field(..., description="Decision correlation ID")
    decision_event_id: str = Field(..., description="Decision event ID")
    pack_version: str = Field(default="1.0.0", description="Evidence pack schema version")
    status: EvidencePackStatus = Field(
        default=EvidencePackStatus.DRAFT, description="Pack status"
    )

    # Generation metadata
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Pack generation timestamp"
    )
    generated_by: str = Field(default="GL-001", description="Agent that generated pack")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Asset information
    asset_id: str = Field(..., description="Asset identifier")
    facility_id: Optional[str] = Field(None, description="Facility identifier")

    # Timestamps
    timestamps: TimestampRecord = Field(..., description="Key timestamps")

    # Input data
    dataset_summary: DatasetSummary = Field(..., description="Input datasets summary")
    unit_conversion_version: str = Field(..., description="Unit conversion version")

    # Configuration versions
    constraint_summary: ConstraintSummary = Field(..., description="Constraints summary")

    # Model versions
    model_summary: ModelVersionSummary = Field(..., description="ML models summary")

    # Explainability
    explainability_summary: ExplainabilitySummary = Field(
        ..., description="Explainability artifacts"
    )

    # Optimization results
    solver_summary: SolverSummary = Field(..., description="Solver results")

    # Actions
    action_summary: ActionSummary = Field(..., description="Actions summary")

    # Operator actions
    operator_actions: List[OperatorActionRecord] = Field(
        default_factory=list, description="Operator actions taken"
    )

    # Impact
    impact_summary: ImpactSummary = Field(..., description="Impact summary")

    # Provenance
    lineage_graph_id: Optional[str] = Field(None, description="Lineage graph ID")
    provenance_merkle_root: Optional[str] = Field(None, description="Merkle root hash")

    # Related events
    related_safety_events: List[str] = Field(
        default_factory=list, description="Related safety event IDs"
    )
    related_compliance_events: List[str] = Field(
        default_factory=list, description="Related compliance event IDs"
    )

    # Hash for integrity
    pack_hash: Optional[str] = Field(None, description="SHA-256 hash of pack contents")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of pack contents."""
        data = self.dict(exclude={"pack_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        """Seal the evidence pack with hash and status update."""
        self.pack_hash = self.calculate_hash()
        self.status = EvidencePackStatus.SEALED

    def verify(self) -> bool:
        """Verify pack integrity."""
        if not self.pack_hash:
            return False
        return self.calculate_hash() == self.pack_hash


class EvidencePackGenerator:
    """
    Generator for per-decision evidence packs.

    This class creates comprehensive evidence packs from decision events,
    action events, and provenance data for regulatory compliance.

    Attributes:
        storage_path: Path for storing evidence packs
        provenance_tracker: Provenance tracker instance

    Example:
        >>> generator = EvidencePackGenerator(storage_path="/audit/evidence")
        >>> pack = generator.generate(
        ...     decision_event=decision_event,
        ...     action_events=action_events,
        ...     lineage_graph=lineage_graph
        ... )
        >>> generator.store(pack)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        provenance_tracker: Optional[EnhancedProvenanceTracker] = None,
    ):
        """
        Initialize evidence pack generator.

        Args:
            storage_path: Path for storing evidence packs
            provenance_tracker: Optional provenance tracker instance
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.provenance_tracker = provenance_tracker

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EvidencePackGenerator initialized",
            extra={"storage_path": str(self.storage_path)}
        )

    def generate(
        self,
        decision_event: DecisionAuditEvent,
        action_events: Optional[List[ActionAuditEvent]] = None,
        safety_events: Optional[List[SafetyAuditEvent]] = None,
        compliance_events: Optional[List[ComplianceAuditEvent]] = None,
        lineage_graph: Optional[LineageGraph] = None,
        operator_actions: Optional[List[OperatorActionRecord]] = None,
        actual_impact: Optional[ExpectedImpact] = None,
    ) -> EvidencePack:
        """
        Generate a complete evidence pack for a decision.

        Args:
            decision_event: The decision audit event
            action_events: Related action events
            safety_events: Related safety events
            compliance_events: Related compliance events
            lineage_graph: Provenance lineage graph
            operator_actions: Operator actions taken
            actual_impact: Actual measured impact

        Returns:
            Complete EvidencePack

        Raises:
            ValueError: If decision_event is invalid
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            f"Generating evidence pack for decision {decision_event.event_id}",
            extra={"correlation_id": decision_event.correlation_id}
        )

        action_events = action_events or []
        safety_events = safety_events or []
        compliance_events = compliance_events or []
        operator_actions = operator_actions or []

        # Build timestamps record
        timestamps = self._build_timestamps(decision_event, action_events)

        # Build dataset summary
        dataset_summary = self._build_dataset_summary(decision_event)

        # Build constraint summary
        constraint_summary = self._build_constraint_summary(decision_event)

        # Build model summary
        model_summary = self._build_model_summary(decision_event)

        # Build explainability summary
        explainability_summary = self._build_explainability_summary(decision_event)

        # Build solver summary
        solver_summary = self._build_solver_summary(decision_event)

        # Build action summary
        action_summary = self._build_action_summary(decision_event, action_events)

        # Build impact summary
        impact_summary = self._build_impact_summary(decision_event, actual_impact)

        # Create evidence pack
        pack = EvidencePack(
            correlation_id=decision_event.correlation_id,
            decision_event_id=str(decision_event.event_id),
            asset_id=decision_event.asset_id,
            facility_id=decision_event.facility_id,
            timestamps=timestamps,
            dataset_summary=dataset_summary,
            unit_conversion_version=decision_event.unit_conversion_version,
            constraint_summary=constraint_summary,
            model_summary=model_summary,
            explainability_summary=explainability_summary,
            solver_summary=solver_summary,
            action_summary=action_summary,
            operator_actions=operator_actions,
            impact_summary=impact_summary,
            lineage_graph_id=str(lineage_graph.graph_id) if lineage_graph else None,
            provenance_merkle_root=lineage_graph.merkle_root if lineage_graph else None,
            related_safety_events=[str(e.event_id) for e in safety_events],
            related_compliance_events=[str(e.event_id) for e in compliance_events],
        )

        # Seal the pack
        pack.seal()

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Evidence pack generated: {pack.pack_id}",
            extra={
                "correlation_id": decision_event.correlation_id,
                "processing_time_ms": processing_time,
                "pack_hash": pack.pack_hash[:16] + "...",
            }
        )

        return pack

    def _build_timestamps(
        self,
        decision_event: DecisionAuditEvent,
        action_events: List[ActionAuditEvent],
    ) -> TimestampRecord:
        """Build timestamp record from events."""
        # Find actuation and verification timestamps from action events
        actuation_ts = None
        verification_ts = None
        operator_review_ts = None
        approval_ts = None

        for action in action_events:
            if action.actuation_timestamp:
                if actuation_ts is None or action.actuation_timestamp < actuation_ts:
                    actuation_ts = action.actuation_timestamp

            if action.verification_timestamp:
                if verification_ts is None or action.verification_timestamp > verification_ts:
                    verification_ts = action.verification_timestamp

            if action.action_status in [ActionStatus.APPROVED, ActionStatus.EXECUTED]:
                if action.recommended_timestamp:
                    if approval_ts is None or action.recommended_timestamp < approval_ts:
                        approval_ts = action.recommended_timestamp

        return TimestampRecord(
            ingestion_timestamp=decision_event.ingestion_timestamp,
            decision_timestamp=decision_event.decision_timestamp,
            recommendation_timestamp=decision_event.timestamp,
            operator_review_timestamp=operator_review_ts,
            approval_timestamp=approval_ts,
            actuation_timestamp=actuation_ts,
            verification_timestamp=verification_ts,
        )

    def _build_dataset_summary(
        self,
        decision_event: DecisionAuditEvent,
    ) -> DatasetSummary:
        """Build dataset summary from decision event."""
        datasets = decision_event.input_datasets
        total_records = sum(d.record_count for d in datasets)

        # Find time range
        time_start = None
        time_end = None
        for d in datasets:
            if d.time_range_start:
                if time_start is None or d.time_range_start < time_start:
                    time_start = d.time_range_start
            if d.time_range_end:
                if time_end is None or d.time_range_end > time_end:
                    time_end = d.time_range_end

        # Calculate combined hash
        hash_inputs = sorted([d.data_hash for d in datasets])
        combined = hashlib.sha256("".join(hash_inputs).encode()).hexdigest()

        return DatasetSummary(
            datasets=datasets,
            total_records=total_records,
            time_range_start=time_start,
            time_range_end=time_end,
            combined_hash=combined,
        )

    def _build_constraint_summary(
        self,
        decision_event: DecisionAuditEvent,
    ) -> ConstraintSummary:
        """Build constraint summary from decision event."""
        return ConstraintSummary(
            constraint_set_id=decision_event.constraint_set_id,
            constraint_set_version=decision_event.constraint_set_version,
            total_constraints=len(decision_event.binding_constraints) + len(decision_event.constraint_violations),
            binding_constraints=decision_event.binding_constraints,
            violated_constraints=decision_event.constraint_violations,
            safety_boundary_policy_version=decision_event.safety_boundary_policy_version,
        )

    def _build_model_summary(
        self,
        decision_event: DecisionAuditEvent,
    ) -> ModelVersionSummary:
        """Build model version summary from decision event."""
        return ModelVersionSummary(
            demand_forecast_model=decision_event.demand_model,
            equipment_health_model=decision_event.health_model,
            anomaly_detection_model=decision_event.anomaly_model,
        )

    def _build_explainability_summary(
        self,
        decision_event: DecisionAuditEvent,
    ) -> ExplainabilitySummary:
        """Build explainability summary from decision event."""
        shap_artifacts = decision_event.shap_artifacts or []
        lime_artifacts = decision_event.lime_artifacts or []
        uq = decision_event.uncertainty_quantification or {}

        # Extract top features from SHAP
        top_features: Dict[str, float] = {}
        for artifact in shap_artifacts:
            for feature, importance in artifact.feature_importances.items():
                if feature not in top_features or abs(importance) > abs(top_features[feature]):
                    top_features[feature] = importance

        # Sort and keep top 10
        sorted_features = sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = dict(sorted_features[:10])

        return ExplainabilitySummary(
            shap_available=len(shap_artifacts) > 0,
            lime_available=len(lime_artifacts) > 0,
            shap_artifacts=shap_artifacts,
            lime_artifacts=lime_artifacts,
            uncertainty_quantification=uq,
            top_features=top_features,
        )

    def _build_solver_summary(
        self,
        decision_event: DecisionAuditEvent,
    ) -> SolverSummary:
        """Build solver summary from decision event."""
        return SolverSummary(
            solver_name=decision_event.solver_name,
            solver_version=decision_event.solver_version,
            solver_status=decision_event.solver_status,
            solve_time_ms=decision_event.solve_time_ms,
            mip_gap=decision_event.mip_gap,
            objective_value=decision_event.objective_value,
            objective_breakdown=decision_event.objective_breakdown,
        )

    def _build_action_summary(
        self,
        decision_event: DecisionAuditEvent,
        action_events: List[ActionAuditEvent],
    ) -> ActionSummary:
        """Build action summary from events."""
        actions = decision_event.recommended_actions

        # Count action statuses
        approved = 0
        rejected = 0
        executed = 0
        modified = 0
        pending = 0

        for action_event in action_events:
            status = action_event.action_status
            if status == ActionStatus.APPROVED:
                approved += 1
            elif status == ActionStatus.REJECTED:
                rejected += 1
            elif status == ActionStatus.EXECUTED:
                executed += 1
            elif status == ActionStatus.OVERRIDDEN:
                modified += 1
            elif status == ActionStatus.PENDING:
                pending += 1

        return ActionSummary(
            total_actions=len(actions),
            actions=actions,
            approved_count=approved,
            rejected_count=rejected,
            executed_count=executed,
            modified_count=modified,
            pending_count=pending,
        )

    def _build_impact_summary(
        self,
        decision_event: DecisionAuditEvent,
        actual_impact: Optional[ExpectedImpact],
    ) -> ImpactSummary:
        """Build impact summary."""
        expected = decision_event.expected_impact

        variance_cost = None
        variance_emissions = None

        if expected and actual_impact:
            if expected.cost_delta_usd != 0:
                variance_cost = (
                    (actual_impact.cost_delta_usd - expected.cost_delta_usd)
                    / abs(expected.cost_delta_usd)
                    * 100
                )
            if expected.emissions_delta_kg_co2e != 0:
                variance_emissions = (
                    (actual_impact.emissions_delta_kg_co2e - expected.emissions_delta_kg_co2e)
                    / abs(expected.emissions_delta_kg_co2e)
                    * 100
                )

        return ImpactSummary(
            expected_impact=expected,
            actual_impact=actual_impact,
            variance_cost_pct=variance_cost,
            variance_emissions_pct=variance_emissions,
        )

    def store(
        self,
        pack: EvidencePack,
        format: EvidencePackFormat = EvidencePackFormat.JSON,
    ) -> str:
        """
        Store evidence pack to configured storage.

        Args:
            pack: Evidence pack to store
            format: Output format

        Returns:
            Storage path/key

        Raises:
            ValueError: If storage path not configured
        """
        if not self.storage_path:
            raise ValueError("Storage path not configured")

        # Create date-based directory structure
        date_path = pack.generated_at.strftime("%Y/%m/%d")
        full_path = self.storage_path / date_path
        full_path.mkdir(parents=True, exist_ok=True)

        filename = f"evidence_pack_{pack.pack_id}"

        if format == EvidencePackFormat.JSON:
            file_path = full_path / f"{filename}.json"
            with open(file_path, "w") as f:
                json.dump(pack.dict(), f, indent=2, default=str)

        elif format == EvidencePackFormat.ZIP:
            file_path = full_path / f"{filename}.zip"
            self._write_zip(pack, file_path)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(
            f"Evidence pack stored: {file_path}",
            extra={"pack_id": str(pack.pack_id)}
        )

        return str(file_path)

    def _write_zip(self, pack: EvidencePack, path: Path) -> None:
        """Write evidence pack as ZIP archive."""
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Main pack JSON
            pack_json = json.dumps(pack.dict(), indent=2, default=str)
            zf.writestr("evidence_pack.json", pack_json)

            # Separate files for key components
            zf.writestr(
                "timestamps.json",
                json.dumps(pack.timestamps.dict(), indent=2, default=str)
            )
            zf.writestr(
                "datasets.json",
                json.dumps(pack.dataset_summary.dict(), indent=2, default=str)
            )
            zf.writestr(
                "constraints.json",
                json.dumps(pack.constraint_summary.dict(), indent=2, default=str)
            )
            zf.writestr(
                "models.json",
                json.dumps(pack.model_summary.dict(), indent=2, default=str)
            )
            zf.writestr(
                "explainability.json",
                json.dumps(pack.explainability_summary.dict(), indent=2, default=str)
            )
            zf.writestr(
                "solver.json",
                json.dumps(pack.solver_summary.dict(), indent=2, default=str)
            )
            zf.writestr(
                "actions.json",
                json.dumps(pack.action_summary.dict(), indent=2, default=str)
            )
            zf.writestr(
                "impact.json",
                json.dumps(pack.impact_summary.dict(), indent=2, default=str)
            )

            # Manifest
            manifest = {
                "pack_id": str(pack.pack_id),
                "correlation_id": pack.correlation_id,
                "generated_at": pack.generated_at.isoformat(),
                "pack_hash": pack.pack_hash,
                "files": [
                    "evidence_pack.json",
                    "timestamps.json",
                    "datasets.json",
                    "constraints.json",
                    "models.json",
                    "explainability.json",
                    "solver.json",
                    "actions.json",
                    "impact.json",
                ],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    def load(self, path: str) -> EvidencePack:
        """
        Load evidence pack from storage.

        Args:
            path: Path to evidence pack file

        Returns:
            Loaded EvidencePack

        Raises:
            FileNotFoundError: If file not found
            ValueError: If format not supported
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Evidence pack not found: {path}")

        if file_path.suffix == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            return EvidencePack(**data)

        elif file_path.suffix == ".zip":
            with zipfile.ZipFile(file_path, "r") as zf:
                with zf.open("evidence_pack.json") as f:
                    data = json.load(f)
            return EvidencePack(**data)

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def verify_pack(self, pack: EvidencePack) -> bool:
        """
        Verify evidence pack integrity.

        Args:
            pack: Evidence pack to verify

        Returns:
            True if pack is valid
        """
        return pack.verify()

    def list_packs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        asset_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List stored evidence packs.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            asset_id: Filter by asset ID

        Returns:
            List of pack metadata
        """
        if not self.storage_path:
            return []

        packs = []
        for json_file in self.storage_path.rglob("*.json"):
            if json_file.name == "manifest.json":
                continue

            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                pack_info = {
                    "pack_id": data.get("pack_id"),
                    "correlation_id": data.get("correlation_id"),
                    "asset_id": data.get("asset_id"),
                    "generated_at": data.get("generated_at"),
                    "status": data.get("status"),
                    "path": str(json_file),
                }

                # Apply filters
                if asset_id and data.get("asset_id") != asset_id:
                    continue

                if start_date or end_date:
                    gen_time = datetime.fromisoformat(data.get("generated_at", ""))
                    if start_date and gen_time < start_date:
                        continue
                    if end_date and gen_time > end_date:
                        continue

                packs.append(pack_info)

            except Exception as e:
                logger.warning(f"Failed to read pack {json_file}: {e}")

        return sorted(packs, key=lambda x: x.get("generated_at", ""), reverse=True)

    def archive_packs(
        self,
        older_than: datetime,
        archive_path: str,
    ) -> int:
        """
        Archive old evidence packs.

        Args:
            older_than: Archive packs older than this date
            archive_path: Path to archive location

        Returns:
            Number of packs archived
        """
        if not self.storage_path:
            return 0

        archive_dir = Path(archive_path)
        archive_dir.mkdir(parents=True, exist_ok=True)

        archived = 0
        for json_file in self.storage_path.rglob("*.json"):
            if json_file.name == "manifest.json":
                continue

            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                gen_time = datetime.fromisoformat(data.get("generated_at", ""))
                if gen_time < older_than:
                    # Move to archive
                    archive_file = archive_dir / json_file.name
                    json_file.rename(archive_file)
                    archived += 1
                    logger.info(f"Archived evidence pack: {json_file.name}")

            except Exception as e:
                logger.warning(f"Failed to archive pack {json_file}: {e}")

        return archived
