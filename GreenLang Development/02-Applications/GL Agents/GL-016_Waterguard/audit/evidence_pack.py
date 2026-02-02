"""
Evidence Pack Models for GL-016 Waterguard

This module defines data models for regulatory evidence packages used
in compliance documentation and audits. Evidence packs contain all
information needed to prove the validity of water chemistry decisions.

Evidence Pack Contents:
    - Chemistry calculation summaries
    - Constraint compliance status
    - Operator decisions and approvals
    - ASME/ABMA alignment documentation
    - Provenance information

Example:
    >>> pack = EvidencePack(
    ...     pack_id="evp-12345",
    ...     correlation_id="corr-12345",
    ...     asset_id="boiler-001",
    ...     chemistry_summary=chemistry_summary,
    ...     constraint_summary=constraint_summary
    ... )
    >>> pack.seal()
    >>> pack.verify()
    True

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .audit_events import (
    ChemistryParameter,
    ConstraintType,
    RecommendationType,
    SeverityLevel,
    OperatorActionType,
)


class EvidencePackStatus(str, Enum):
    """Status of evidence pack."""

    DRAFT = "DRAFT"
    COMPLETE = "COMPLETE"
    SEALED = "SEALED"
    ARCHIVED = "ARCHIVED"
    SUPERSEDED = "SUPERSEDED"


class ComplianceStandard(str, Enum):
    """Compliance standards for water chemistry."""

    ASME = "ASME"
    ABMA = "ABMA"
    EPRI = "EPRI"
    VGB = "VGB"
    INTERNAL = "INTERNAL"


class ChemistryParameterSummary(BaseModel):
    """Summary of a chemistry parameter over a period."""

    parameter: ChemistryParameter = Field(..., description="Chemistry parameter")
    unit: str = Field(..., description="Engineering unit")
    reading_count: int = Field(0, ge=0, description="Number of readings")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    avg_value: Optional[float] = Field(None, description="Average value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")
    limit_min: Optional[float] = Field(None, description="Minimum limit")
    limit_max: Optional[float] = Field(None, description="Maximum limit")
    out_of_spec_count: int = Field(0, ge=0, description="Out of spec readings")
    out_of_spec_pct: float = Field(0.0, ge=0, le=100, description="Out of spec percentage")
    last_value: Optional[float] = Field(None, description="Most recent value")
    last_reading_time: Optional[datetime] = Field(None, description="Time of last reading")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChemistryCalculationSummary(BaseModel):
    """Summary of chemistry calculations for evidence pack."""

    calculation_period_start: datetime = Field(..., description="Period start")
    calculation_period_end: datetime = Field(..., description="Period end")
    total_calculations: int = Field(0, ge=0, description="Total calculations performed")
    asset_id: str = Field(..., description="Asset ID")

    # Parameter summaries
    parameters: Dict[str, ChemistryParameterSummary] = Field(
        default_factory=dict, description="Parameter summaries"
    )

    # Operating conditions
    avg_drum_pressure_psig: Optional[float] = Field(None, description="Average drum pressure")
    avg_steam_flow_klb_hr: Optional[float] = Field(None, description="Average steam flow")
    avg_cycles_of_concentration: Optional[float] = Field(None, description="Average cycles")
    avg_blowdown_rate_pct: Optional[float] = Field(None, description="Average blowdown rate")

    # Calculation quality
    data_completeness_pct: float = Field(100.0, ge=0, le=100, description="Data completeness")
    calculation_engine_version: str = Field(..., description="Calculation engine version")
    formula_version: str = Field(..., description="Formula version")
    input_data_hash: str = Field(..., description="Hash of input data")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConstraintComplianceRecord(BaseModel):
    """Record of compliance with a single constraint."""

    constraint_id: str = Field(..., description="Constraint ID")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    parameter: ChemistryParameter = Field(..., description="Constrained parameter")
    limit_type: str = Field(..., description="MIN, MAX, or RANGE")
    limit_value: float = Field(..., description="Limit value")
    unit: str = Field(..., description="Engineering unit")

    # Compliance metrics
    compliant_readings: int = Field(0, ge=0, description="Compliant readings")
    total_readings: int = Field(0, ge=0, description="Total readings")
    compliance_pct: float = Field(100.0, ge=0, le=100, description="Compliance percentage")
    violations: int = Field(0, ge=0, description="Number of violations")
    max_deviation_pct: float = Field(0.0, description="Maximum deviation from limit")

    # Standard reference
    standard: ComplianceStandard = Field(..., description="Compliance standard")
    standard_reference: Optional[str] = Field(None, description="Standard section reference")
    pressure_range: Optional[str] = Field(None, description="Applicable pressure range")

    class Config:
        frozen = True


class ConstraintComplianceSummary(BaseModel):
    """Summary of constraint compliance for evidence pack."""

    compliance_period_start: datetime = Field(..., description="Period start")
    compliance_period_end: datetime = Field(..., description="Period end")
    asset_id: str = Field(..., description="Asset ID")

    # Overall compliance
    total_constraints: int = Field(0, ge=0, description="Total constraints evaluated")
    compliant_constraints: int = Field(0, ge=0, description="Fully compliant constraints")
    overall_compliance_pct: float = Field(100.0, ge=0, le=100, description="Overall compliance")

    # Constraint details
    constraints: List[ConstraintComplianceRecord] = Field(
        default_factory=list, description="Individual constraint records"
    )

    # Violation summary
    total_violations: int = Field(0, ge=0, description="Total violations")
    critical_violations: int = Field(0, ge=0, description="Critical violations")
    warning_violations: int = Field(0, ge=0, description="Warning violations")

    # Configuration
    constraint_set_version: str = Field(..., description="Constraint set version")
    constraint_set_hash: str = Field(..., description="Constraint set hash")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RecommendationRecord(BaseModel):
    """Record of a single recommendation."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    recommendation_type: RecommendationType = Field(..., description="Type")
    timestamp: datetime = Field(..., description="Recommendation timestamp")

    # Values
    current_value: float = Field(..., description="Current value")
    recommended_value: float = Field(..., description="Recommended value")
    actual_value: Optional[float] = Field(None, description="Actual achieved value")
    unit: str = Field(..., description="Engineering unit")

    # Rationale
    explanation: str = Field(..., description="Recommendation explanation")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence 0-1")

    # Outcome
    was_implemented: bool = Field(False, description="Whether implemented")
    implementation_timestamp: Optional[datetime] = Field(None, description="When implemented")
    operator_action: Optional[str] = Field(None, description="Operator action taken")

    # Impact
    expected_savings: Optional[Dict[str, float]] = Field(None, description="Expected savings")
    actual_savings: Optional[Dict[str, float]] = Field(None, description="Actual savings")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RecommendationSummary(BaseModel):
    """Summary of recommendations for evidence pack."""

    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    asset_id: str = Field(..., description="Asset ID")

    # Counts
    total_recommendations: int = Field(0, ge=0, description="Total recommendations")
    implemented_recommendations: int = Field(0, ge=0, description="Implemented count")
    rejected_recommendations: int = Field(0, ge=0, description="Rejected count")
    modified_recommendations: int = Field(0, ge=0, description="Modified count")

    # By type
    recommendations_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Count by type"
    )

    # Savings
    total_expected_water_savings_gal: float = Field(0.0, description="Expected water savings")
    total_expected_energy_savings_mmbtu: float = Field(0.0, description="Expected energy savings")
    total_expected_chemical_savings_usd: float = Field(0.0, description="Expected chemical savings")

    total_actual_water_savings_gal: Optional[float] = Field(None, description="Actual water savings")
    total_actual_energy_savings_mmbtu: Optional[float] = Field(None, description="Actual energy savings")
    total_actual_chemical_savings_usd: Optional[float] = Field(None, description="Actual chemical savings")

    # Individual records (limited for pack size)
    recommendations: List[RecommendationRecord] = Field(
        default_factory=list, description="Individual recommendations (limited)"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class OperatorDecisionRecord(BaseModel):
    """Record of an operator decision."""

    action_id: str = Field(..., description="Action ID")
    action_type: OperatorActionType = Field(..., description="Action type")
    timestamp: datetime = Field(..., description="Action timestamp")

    # Operator info
    operator_id: str = Field(..., description="Operator ID")
    operator_name: Optional[str] = Field(None, description="Operator name")
    authorization_level: str = Field(..., description="Authorization level")

    # Related entity
    related_entity_id: str = Field(..., description="Related recommendation/command ID")
    related_entity_type: str = Field(..., description="Type of related entity")

    # Decision details
    original_value: Optional[Any] = Field(None, description="Original value")
    modified_value: Optional[Any] = Field(None, description="Modified value if changed")
    justification: str = Field(..., description="Justification for decision")

    # Override details (if applicable)
    is_override: bool = Field(False, description="Whether this is an override")
    override_duration_hours: Optional[float] = Field(None, description="Override duration")
    override_expiry: Optional[datetime] = Field(None, description="Override expiry")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class OperatorDecisionSummary(BaseModel):
    """Summary of operator decisions for evidence pack."""

    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    asset_id: str = Field(..., description="Asset ID")

    # Counts
    total_decisions: int = Field(0, ge=0, description="Total operator decisions")
    approvals: int = Field(0, ge=0, description="Approvals")
    rejections: int = Field(0, ge=0, description="Rejections")
    modifications: int = Field(0, ge=0, description="Modifications")
    overrides: int = Field(0, ge=0, description="Overrides")

    # Active overrides
    active_overrides: int = Field(0, ge=0, description="Currently active overrides")

    # Individual records
    decisions: List[OperatorDecisionRecord] = Field(
        default_factory=list, description="Individual decisions"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ASMEAlignment(BaseModel):
    """ASME standard alignment documentation."""

    standard_id: str = Field(default="ASME PTC 4.2", description="Standard identifier")
    standard_version: str = Field(..., description="Standard version")
    applicable_sections: List[str] = Field(
        default_factory=list, description="Applicable sections"
    )

    # Alignment status
    is_aligned: bool = Field(True, description="Overall alignment status")
    alignment_pct: float = Field(100.0, ge=0, le=100, description="Alignment percentage")

    # Specific alignments
    pressure_limits_aligned: bool = Field(True, description="Pressure limits aligned")
    conductivity_limits_aligned: bool = Field(True, description="Conductivity limits aligned")
    silica_limits_aligned: bool = Field(True, description="Silica limits aligned")
    ph_limits_aligned: bool = Field(True, description="pH limits aligned")

    # Deviations
    deviations: List[str] = Field(default_factory=list, description="Documented deviations")
    deviation_justifications: Dict[str, str] = Field(
        default_factory=dict, description="Justifications for deviations"
    )


class ABMAAlignment(BaseModel):
    """ABMA standard alignment documentation."""

    standard_id: str = Field(default="ABMA 402", description="Standard identifier")
    standard_version: str = Field(..., description="Standard version")
    applicable_sections: List[str] = Field(
        default_factory=list, description="Applicable sections"
    )

    # Alignment status
    is_aligned: bool = Field(True, description="Overall alignment status")
    alignment_pct: float = Field(100.0, ge=0, le=100, description="Alignment percentage")

    # Specific alignments
    feedwater_limits_aligned: bool = Field(True, description="Feedwater limits aligned")
    boiler_water_limits_aligned: bool = Field(True, description="Boiler water limits aligned")
    blowdown_guidance_aligned: bool = Field(True, description="Blowdown guidance aligned")

    # Deviations
    deviations: List[str] = Field(default_factory=list, description="Documented deviations")
    deviation_justifications: Dict[str, str] = Field(
        default_factory=dict, description="Justifications for deviations"
    )


class ProvenanceSummary(BaseModel):
    """Provenance summary for evidence pack."""

    correlation_id: str = Field(..., description="Correlation ID")
    lineage_graph_id: Optional[str] = Field(None, description="Lineage graph ID")
    merkle_root: str = Field(..., description="Merkle root hash")

    # Version tracking
    config_version: str = Field(..., description="Configuration version")
    code_version: str = Field(..., description="Agent code version")
    formula_version: str = Field(..., description="Formula version")
    constraint_version: str = Field(..., description="Constraint set version")

    # Input traceability
    input_event_count: int = Field(0, ge=0, description="Number of input events")
    input_data_hash: str = Field(..., description="Combined hash of inputs")

    # Chain verification
    chain_verified: bool = Field(False, description="Whether chain was verified")
    chain_verification_time: Optional[datetime] = Field(None, description="Verification time")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class EvidencePack(BaseModel):
    """
    Complete evidence pack for regulatory compliance.

    Contains all documentation needed to prove the validity of
    water chemistry decisions and demonstrate regulatory compliance.
    """

    # Identification
    pack_id: UUID = Field(default_factory=uuid4, description="Unique pack identifier")
    correlation_id: str = Field(..., description="Primary correlation ID")
    related_correlation_ids: List[str] = Field(
        default_factory=list, description="Related correlation IDs"
    )
    pack_version: str = Field(default="1.0.0", description="Evidence pack schema version")
    status: EvidencePackStatus = Field(
        default=EvidencePackStatus.DRAFT, description="Pack status"
    )

    # Generation metadata
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Pack generation timestamp"
    )
    generated_by: str = Field(default="GL-016", description="Agent that generated pack")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Asset information
    asset_id: str = Field(..., description="Asset identifier")
    asset_name: Optional[str] = Field(None, description="Asset name")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    facility_name: Optional[str] = Field(None, description="Facility name")

    # Time period
    period_start: datetime = Field(..., description="Evidence period start")
    period_end: datetime = Field(..., description="Evidence period end")

    # Chemistry summary
    chemistry_summary: ChemistryCalculationSummary = Field(
        ..., description="Chemistry calculation summary"
    )

    # Constraint compliance
    constraint_summary: ConstraintComplianceSummary = Field(
        ..., description="Constraint compliance summary"
    )

    # Recommendations
    recommendation_summary: Optional[RecommendationSummary] = Field(
        None, description="Recommendation summary"
    )

    # Operator decisions
    operator_summary: Optional[OperatorDecisionSummary] = Field(
        None, description="Operator decision summary"
    )

    # Standards alignment
    asme_alignment: Optional[ASMEAlignment] = Field(None, description="ASME alignment")
    abma_alignment: Optional[ABMAAlignment] = Field(None, description="ABMA alignment")

    # Provenance
    provenance_summary: ProvenanceSummary = Field(..., description="Provenance summary")

    # Attachments (references)
    attachment_ids: List[str] = Field(
        default_factory=list, description="Attached document IDs"
    )

    # Integrity
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

    def is_compliant(self) -> bool:
        """Check if evidence pack shows overall compliance."""
        return (
            self.constraint_summary.overall_compliance_pct >= 95.0
            and self.constraint_summary.critical_violations == 0
        )

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics from the evidence pack."""
        return {
            "asset_id": self.asset_id,
            "period": f"{self.period_start.date()} to {self.period_end.date()}",
            "total_calculations": self.chemistry_summary.total_calculations,
            "data_completeness_pct": self.chemistry_summary.data_completeness_pct,
            "overall_compliance_pct": self.constraint_summary.overall_compliance_pct,
            "total_violations": self.constraint_summary.total_violations,
            "critical_violations": self.constraint_summary.critical_violations,
            "total_recommendations": (
                self.recommendation_summary.total_recommendations
                if self.recommendation_summary else 0
            ),
            "is_compliant": self.is_compliant(),
            "is_sealed": self.status == EvidencePackStatus.SEALED,
            "is_verified": self.verify() if self.pack_hash else False,
        }
