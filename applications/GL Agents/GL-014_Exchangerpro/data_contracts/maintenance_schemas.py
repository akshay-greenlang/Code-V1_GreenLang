# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro: Maintenance Schemas - Version 1.0

Provides validated data schemas for maintenance and cleaning history,
inspection records, and work order integration for heat exchangers.

This module defines Pydantic v2 models for:
- CleaningEvent: Complete cleaning event record with effectiveness metrics
- InspectionRecord: Inspection findings including deposit characterization
- WorkOrder: CMMS work order linkage and status tracking

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CleaningType(str, Enum):
    """Type of cleaning performed on exchanger."""
    CHEMICAL_ONLINE = "chemical_online"  # CIP - Clean In Place
    CHEMICAL_OFFLINE = "chemical_offline"  # Chemical cleaning during shutdown
    MECHANICAL_HYDROBLAST = "mechanical_hydroblast"
    MECHANICAL_PIGGING = "mechanical_pigging"
    MECHANICAL_BRUSHING = "mechanical_brushing"
    MECHANICAL_RODDING = "mechanical_rodding"
    MECHANICAL_SANDBLAST = "mechanical_sandblast"
    THERMAL = "thermal"  # Thermal shock cleaning
    COMBINED = "combined"  # Multiple methods
    OTHER = "other"


class CleaningMethod(str, Enum):
    """Specific cleaning method/technique."""
    CIP_ALKALINE = "cip_alkaline"
    CIP_ACID = "cip_acid"
    CIP_CHELANT = "cip_chelant"
    CIP_SOLVENT = "cip_solvent"
    HYDROBLAST_HIGH_PRESSURE = "hydroblast_high_pressure"
    HYDROBLAST_ULTRA_HIGH = "hydroblast_ultra_high"
    PIG_SOFT = "pig_soft"
    PIG_HARD = "pig_hard"
    PIG_BRUSH = "pig_brush"
    BRUSH_MANUAL = "brush_manual"
    BRUSH_POWERED = "brush_powered"
    THERMAL_BAKEOUT = "thermal_bakeout"
    THERMAL_STEAM = "thermal_steam"
    OTHER = "other"


class CleaningSide(str, Enum):
    """Which side of exchanger was cleaned."""
    SHELL = "shell"
    TUBE = "tube"
    BOTH = "both"


class DepositType(str, Enum):
    """Classification of deposit/fouling type."""
    BIOLOGICAL = "biological"  # Biofilm, algae
    SCALE_CALCIUM = "scale_calcium"  # CaCO3, CaSO4
    SCALE_SILICA = "scale_silica"
    SCALE_MIXED = "scale_mixed"
    CORROSION_PRODUCTS = "corrosion_products"
    PARTICULATE = "particulate"  # Silt, sand
    ORGANIC_HYDROCARBON = "organic_hydrocarbon"
    ORGANIC_POLYMER = "organic_polymer"
    COKE = "coke"  # Refinery applications
    WAX = "wax"  # Petroleum applications
    ASPHALTENE = "asphaltene"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class InspectionType(str, Enum):
    """Type of inspection performed."""
    VISUAL_EXTERNAL = "visual_external"
    VISUAL_INTERNAL = "visual_internal"
    EDDY_CURRENT = "eddy_current"
    ULTRASONIC_THICKNESS = "ultrasonic_thickness"
    ULTRASONIC_PHASED_ARRAY = "ultrasonic_phased_array"
    RADIOGRAPHIC = "radiographic"
    HYDROSTATIC_TEST = "hydrostatic_test"
    PNEUMATIC_TEST = "pneumatic_test"
    BORESCOPE = "borescope"
    THERMOGRAPHIC = "thermographic"
    DYE_PENETRANT = "dye_penetrant"
    MAGNETIC_PARTICLE = "magnetic_particle"
    ACOUSTIC_EMISSION = "acoustic_emission"


class WorkOrderStatus(str, Enum):
    """Work order lifecycle status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class WorkOrderPriority(str, Enum):
    """Work order priority level."""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ROUTINE = "routine"


class MaintenanceCategory(str, Enum):
    """Category of maintenance activity."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    BREAKDOWN = "breakdown"
    TURNAROUND = "turnaround"
    PROJECT = "project"


# =============================================================================
# CHEMICAL USAGE
# =============================================================================

class ChemicalUsage(BaseModel):
    """Details of chemical used in cleaning process."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "chemical_name": "Hydrochloric Acid",
                    "chemical_id": "HCL-10PCT",
                    "concentration_percent": 10.0,
                    "volume_liters": 500.0,
                    "contact_time_minutes": 120,
                    "temperature_c": 60.0
                }
            ]
        }
    )

    chemical_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of chemical used"
    )
    chemical_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Chemical inventory ID or product code"
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=100,
        description="Chemical manufacturer"
    )
    concentration_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Concentration in percent"
    )
    volume_liters: Optional[float] = Field(
        None,
        gt=0,
        description="Volume used in liters"
    )
    mass_kg: Optional[float] = Field(
        None,
        gt=0,
        description="Mass used in kg"
    )
    contact_time_minutes: Optional[float] = Field(
        None,
        gt=0,
        description="Contact time in minutes"
    )
    temperature_c: Optional[float] = Field(
        None,
        ge=-50,
        le=200,
        description="Application temperature in Celsius"
    )
    circulation_rate_m3_h: Optional[float] = Field(
        None,
        gt=0,
        description="Circulation rate in m^3/h during cleaning"
    )
    ph_initial: Optional[float] = Field(
        None,
        ge=0,
        le=14,
        description="Initial pH of cleaning solution"
    )
    ph_final: Optional[float] = Field(
        None,
        ge=0,
        le=14,
        description="Final pH after cleaning"
    )
    inhibitor_used: Optional[str] = Field(
        None,
        max_length=200,
        description="Corrosion inhibitor used (if applicable)"
    )


# =============================================================================
# CLEANING EVENT
# =============================================================================

class CleaningEvent(BaseModel):
    """
    Complete cleaning event record for heat exchanger.

    Captures all aspects of a cleaning intervention including timing,
    methods, chemicals, costs, and effectiveness metrics.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event_id": "CLN-2024-0015",
                    "exchanger_id": "HX-1001",
                    "cleaning_type": "chemical_offline",
                    "cleaning_side": "tube",
                    "start_time": "2024-01-15T08:00:00Z",
                    "end_time": "2024-01-15T16:00:00Z",
                    "method": "cip_acid",
                    "labor_hours": 24.0,
                    "total_cost_usd": 15000.0,
                    "effectiveness_rating": 4,
                    "pre_clean_ua_w_k": 28000.0,
                    "post_clean_ua_w_k": 42000.0
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    event_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique cleaning event identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )

    # Timing
    start_time: datetime = Field(
        ...,
        description="Cleaning start time"
    )
    end_time: datetime = Field(
        ...,
        description="Cleaning end time"
    )

    # Cleaning specification
    cleaning_type: CleaningType = Field(
        ...,
        description="Type of cleaning performed"
    )
    cleaning_side: CleaningSide = Field(
        default=CleaningSide.BOTH,
        description="Which side of exchanger was cleaned"
    )
    method: CleaningMethod = Field(
        ...,
        description="Specific cleaning method used"
    )
    method_details: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional method details or procedure notes"
    )

    # Chemicals used
    chemicals_used: List[ChemicalUsage] = Field(
        default_factory=list,
        description="List of chemicals used in cleaning"
    )

    # Labor and costs
    labor_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Total labor hours for cleaning"
    )
    contractor_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Cleaning contractor name (if outsourced)"
    )
    chemical_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Cost of chemicals in USD"
    )
    labor_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Labor cost in USD"
    )
    equipment_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Equipment rental/usage cost in USD"
    )
    disposal_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Waste disposal cost in USD"
    )
    total_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Total cleaning cost in USD"
    )

    # Production impact
    production_loss_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Hours of production loss during cleaning"
    )
    production_loss_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Cost of production loss in USD"
    )

    # Effectiveness metrics
    effectiveness_rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Cleaning effectiveness rating (1=poor, 5=excellent)"
    )
    pre_clean_ua_w_k: Optional[float] = Field(
        None,
        gt=0,
        description="UA value before cleaning in W/K"
    )
    post_clean_ua_w_k: Optional[float] = Field(
        None,
        gt=0,
        description="UA value after cleaning in W/K"
    )
    pre_clean_dp_shell_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Shell-side pressure drop before cleaning in bar"
    )
    post_clean_dp_shell_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Shell-side pressure drop after cleaning in bar"
    )
    pre_clean_dp_tube_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Tube-side pressure drop before cleaning in bar"
    )
    post_clean_dp_tube_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Tube-side pressure drop after cleaning in bar"
    )

    # Fouling removed
    deposit_type_removed: Optional[DepositType] = Field(
        None,
        description="Primary type of deposit removed"
    )
    deposit_mass_kg: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated mass of deposit removed in kg"
    )
    tubes_plugged_before: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tubes plugged before cleaning"
    )
    tubes_plugged_after: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tubes plugged after cleaning"
    )

    # Documentation
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional notes on cleaning event"
    )
    photos_ref: List[str] = Field(
        default_factory=list,
        description="References to before/after photos"
    )
    procedure_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to cleaning procedure document"
    )

    # Work order linkage
    work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Associated CMMS work order ID"
    )

    # Personnel
    performed_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Name or ID of person/team who performed cleaning"
    )
    approved_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Name of person who approved cleaning completion"
    )

    # Provenance
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record last update timestamp"
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def validate_timing(self) -> "CleaningEvent":
        """Validate time consistency."""
        if self.end_time < self.start_time:
            raise ValueError("End time must be after start time")
        return self

    @property
    def duration_hours(self) -> float:
        """Calculate cleaning duration in hours."""
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600

    @property
    def ua_recovery_percent(self) -> Optional[float]:
        """Calculate UA recovery percentage."""
        if self.pre_clean_ua_w_k and self.post_clean_ua_w_k:
            return (
                (self.post_clean_ua_w_k - self.pre_clean_ua_w_k) /
                self.pre_clean_ua_w_k * 100
            )
        return None

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = (
            f"{self.event_id}"
            f"{self.exchanger_id}"
            f"{self.start_time.isoformat()}"
            f"{self.end_time.isoformat()}"
            f"{self.cleaning_type.value}"
            f"{self.method.value}"
        )
        if self.pre_clean_ua_w_k:
            content += f"{self.pre_clean_ua_w_k:.2f}"
        if self.post_clean_ua_w_k:
            content += f"{self.post_clean_ua_w_k:.2f}"

        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# INSPECTION RECORD
# =============================================================================

class InspectionFinding(BaseModel):
    """Individual finding from inspection."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "finding_id": "FND-001",
                    "location": "Tube bundle, rows 1-10",
                    "finding_type": "fouling",
                    "severity": "moderate",
                    "description": "Calcium scale deposits on tube OD",
                    "deposit_type": "scale_calcium",
                    "deposit_thickness_mm": 2.5
                }
            ]
        }
    )

    finding_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique finding identifier"
    )
    location: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Location of finding within exchanger"
    )
    finding_type: Literal[
        "fouling", "corrosion", "erosion", "crack",
        "leak", "deformation", "blockage", "damage", "wear", "other"
    ] = Field(..., description="Type of finding")
    severity: Literal[
        "minor", "moderate", "major", "critical"
    ] = Field(..., description="Severity of finding")
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Detailed description of finding"
    )

    # Deposit-specific fields
    deposit_type: Optional[DepositType] = Field(
        None,
        description="Type of deposit (if fouling finding)"
    )
    deposit_thickness_mm: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Deposit thickness in mm"
    )
    deposit_coverage_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage of surface covered by deposit"
    )

    # Corrosion-specific fields
    wall_loss_mm: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Wall thickness loss in mm (if corrosion)"
    )
    remaining_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        description="Remaining wall thickness in mm"
    )
    corrosion_rate_mm_yr: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated corrosion rate in mm/year"
    )

    # Tube-specific fields
    tubes_affected: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tubes affected"
    )
    tube_numbers: Optional[List[str]] = Field(
        None,
        description="List of affected tube numbers/IDs"
    )

    # Recommendations
    recommended_action: Optional[str] = Field(
        None,
        max_length=500,
        description="Recommended action for this finding"
    )
    action_deadline: Optional[datetime] = Field(
        None,
        description="Deadline for recommended action"
    )

    # Photos
    photo_refs: List[str] = Field(
        default_factory=list,
        description="References to photos of finding"
    )


class InspectionRecord(BaseModel):
    """
    Complete inspection record for heat exchanger.

    Captures inspection details, findings, and recommendations
    for condition assessment and maintenance planning.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "inspection_id": "INS-2024-0042",
                    "exchanger_id": "HX-1001",
                    "inspection_type": "visual_internal",
                    "inspection_date": "2024-01-20T10:00:00Z",
                    "inspector_name": "John Smith",
                    "overall_condition": "fair",
                    "findings": [
                        {
                            "finding_id": "FND-001",
                            "location": "Tube bundle",
                            "finding_type": "fouling",
                            "severity": "moderate",
                            "description": "Moderate fouling observed"
                        }
                    ]
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    inspection_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique inspection identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )

    # Inspection details
    inspection_type: InspectionType = Field(
        ...,
        description="Type of inspection performed"
    )
    inspection_date: datetime = Field(
        ...,
        description="Date and time of inspection"
    )
    inspection_scope: Optional[str] = Field(
        None,
        max_length=500,
        description="Scope of inspection"
    )

    # Personnel
    inspector_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of inspector"
    )
    inspector_company: Optional[str] = Field(
        None,
        max_length=200,
        description="Inspector's company (if third-party)"
    )
    inspector_certification: Optional[str] = Field(
        None,
        max_length=200,
        description="Inspector certification level"
    )

    # Equipment used
    inspection_equipment: Optional[List[str]] = Field(
        None,
        description="List of inspection equipment used"
    )

    # Overall assessment
    overall_condition: Literal[
        "excellent", "good", "fair", "poor", "critical"
    ] = Field(
        ...,
        description="Overall condition assessment"
    )
    fitness_for_service: Literal[
        "fit", "fit_with_restrictions", "unfit"
    ] = Field(
        default="fit",
        description="Fitness for service determination"
    )

    # Findings
    findings: List[InspectionFinding] = Field(
        default_factory=list,
        description="List of inspection findings"
    )

    # Deposit characterization (summary)
    primary_deposit_type: Optional[DepositType] = Field(
        None,
        description="Primary type of deposit observed"
    )
    average_deposit_thickness_mm: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Average deposit thickness in mm"
    )
    max_deposit_thickness_mm: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Maximum deposit thickness in mm"
    )
    deposit_sample_taken: bool = Field(
        default=False,
        description="Whether deposit sample was collected for analysis"
    )
    deposit_analysis_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to deposit analysis report"
    )

    # Tube condition
    total_tubes: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of tubes inspected"
    )
    tubes_with_defects: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tubes with defects"
    )
    tubes_plugged: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tubes currently plugged"
    )
    tubes_recommended_for_plugging: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tubes recommended for plugging"
    )

    # Documentation
    photos_ref: List[str] = Field(
        default_factory=list,
        description="References to inspection photos"
    )
    report_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to full inspection report"
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional inspection notes"
    )

    # Recommendations
    recommendations: Optional[str] = Field(
        None,
        max_length=2000,
        description="Overall recommendations from inspection"
    )
    next_inspection_due: Optional[datetime] = Field(
        None,
        description="Recommended date for next inspection"
    )
    cleaning_recommended: bool = Field(
        default=False,
        description="Whether cleaning is recommended"
    )
    repair_required: bool = Field(
        default=False,
        description="Whether repair is required"
    )

    # Work order linkage
    work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Associated CMMS work order ID"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record last update timestamp"
    )

    @property
    def finding_count(self) -> int:
        """Return number of findings."""
        return len(self.findings)

    @property
    def critical_findings(self) -> List[InspectionFinding]:
        """Return list of critical findings."""
        return [f for f in self.findings if f.severity == "critical"]


# =============================================================================
# WORK ORDER
# =============================================================================

class WorkOrderCost(BaseModel):
    """Cost breakdown for work order."""

    model_config = ConfigDict(frozen=True)

    labor_cost_usd: float = Field(
        default=0,
        ge=0,
        description="Labor cost in USD"
    )
    material_cost_usd: float = Field(
        default=0,
        ge=0,
        description="Material/parts cost in USD"
    )
    contractor_cost_usd: float = Field(
        default=0,
        ge=0,
        description="Contractor cost in USD"
    )
    equipment_cost_usd: float = Field(
        default=0,
        ge=0,
        description="Equipment rental cost in USD"
    )
    other_cost_usd: float = Field(
        default=0,
        ge=0,
        description="Other costs in USD"
    )

    @property
    def total_cost_usd(self) -> float:
        """Calculate total cost."""
        return (
            self.labor_cost_usd +
            self.material_cost_usd +
            self.contractor_cost_usd +
            self.equipment_cost_usd +
            self.other_cost_usd
        )


class WorkOrder(BaseModel):
    """
    CMMS work order for heat exchanger maintenance.

    Provides linkage to external CMMS systems and tracks
    work order lifecycle for maintenance activities.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "work_order_id": "WO-2024-00156",
                    "exchanger_id": "HX-1001",
                    "cmms_system": "SAP PM",
                    "cmms_work_order_id": "4000012345",
                    "title": "HX-1001 Tube Bundle Cleaning",
                    "category": "predictive",
                    "priority": "high",
                    "status": "scheduled",
                    "scheduled_start": "2024-02-01T08:00:00Z"
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    work_order_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Internal work order identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )

    # CMMS linkage
    cmms_system: Optional[str] = Field(
        None,
        max_length=100,
        description="CMMS system name (e.g., SAP PM, Maximo)"
    )
    cmms_work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Work order ID in CMMS system"
    )
    cmms_notification_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Notification/request ID in CMMS"
    )

    # Work order details
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Work order title"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed work order description"
    )
    category: MaintenanceCategory = Field(
        ...,
        description="Maintenance category"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Work order priority"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Current work order status"
    )

    # Timing
    created_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Work order creation date"
    )
    scheduled_start: Optional[datetime] = Field(
        None,
        description="Scheduled start date/time"
    )
    scheduled_end: Optional[datetime] = Field(
        None,
        description="Scheduled end date/time"
    )
    actual_start: Optional[datetime] = Field(
        None,
        description="Actual start date/time"
    )
    actual_end: Optional[datetime] = Field(
        None,
        description="Actual end date/time"
    )

    # Resource allocation
    assigned_to: Optional[str] = Field(
        None,
        max_length=200,
        description="Person or team assigned"
    )
    craft_required: Optional[List[str]] = Field(
        None,
        description="Crafts/skills required"
    )
    estimated_labor_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated labor hours"
    )
    actual_labor_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Actual labor hours spent"
    )

    # Parts and materials
    parts_required: List[str] = Field(
        default_factory=list,
        description="List of required parts/materials"
    )
    parts_ordered: bool = Field(
        default=False,
        description="Whether required parts have been ordered"
    )
    parts_received: bool = Field(
        default=False,
        description="Whether required parts have been received"
    )

    # Costs
    estimated_cost: Optional[WorkOrderCost] = Field(
        None,
        description="Estimated costs"
    )
    actual_cost: Optional[WorkOrderCost] = Field(
        None,
        description="Actual costs"
    )

    # Completion
    completion_notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Notes on work completion"
    )
    failure_found: bool = Field(
        default=False,
        description="Whether a failure was found/confirmed"
    )
    failure_description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Description of failure found"
    )

    # Related records
    related_cleaning_event_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Related cleaning event ID"
    )
    related_inspection_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Related inspection record ID"
    )
    parent_work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Parent work order ID (if child WO)"
    )

    # Approval workflow
    requested_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person who requested the work"
    )
    approved_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person who approved the work order"
    )
    closed_by: Optional[str] = Field(
        None,
        max_length=200,
        description="Person who closed the work order"
    )

    # Timestamps
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record last update timestamp"
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "WorkOrder":
        """Validate date consistency."""
        if self.scheduled_end and self.scheduled_start:
            if self.scheduled_end < self.scheduled_start:
                raise ValueError("Scheduled end must be after scheduled start")

        if self.actual_end and self.actual_start:
            if self.actual_end < self.actual_start:
                raise ValueError("Actual end must be after actual start")

        return self

    @property
    def is_overdue(self) -> bool:
        """Check if work order is overdue."""
        if self.scheduled_end and self.status not in [
            WorkOrderStatus.COMPLETED,
            WorkOrderStatus.CANCELLED,
            WorkOrderStatus.CLOSED,
        ]:
            return datetime.now(timezone.utc) > self.scheduled_end
        return False


# =============================================================================
# EXPORTS
# =============================================================================

MAINTENANCE_SCHEMAS = {
    "CleaningType": CleaningType,
    "CleaningMethod": CleaningMethod,
    "CleaningSide": CleaningSide,
    "DepositType": DepositType,
    "InspectionType": InspectionType,
    "WorkOrderStatus": WorkOrderStatus,
    "WorkOrderPriority": WorkOrderPriority,
    "MaintenanceCategory": MaintenanceCategory,
    "ChemicalUsage": ChemicalUsage,
    "CleaningEvent": CleaningEvent,
    "InspectionFinding": InspectionFinding,
    "InspectionRecord": InspectionRecord,
    "WorkOrderCost": WorkOrderCost,
    "WorkOrder": WorkOrder,
}

__all__ = [
    # Enumerations
    "CleaningType",
    "CleaningMethod",
    "CleaningSide",
    "DepositType",
    "InspectionType",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "MaintenanceCategory",
    # Supporting models
    "ChemicalUsage",
    "InspectionFinding",
    "WorkOrderCost",
    # Main schemas
    "CleaningEvent",
    "InspectionRecord",
    "WorkOrder",
    # Export dictionary
    "MAINTENANCE_SCHEMAS",
]
