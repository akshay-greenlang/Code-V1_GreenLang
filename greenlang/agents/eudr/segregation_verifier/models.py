# -*- coding: utf-8 -*-
"""
Segregation Verifier Data Models - AGENT-EUDR-010

Pydantic v2 data models for the Segregation Verifier Agent covering
physical segregation verification of EUDR-compliant vs non-compliant
material across the supply chain: storage segregation, transport
segregation, processing line verification, cross-contamination
detection, labeling compliance, and facility assessment.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all segregation verification operations per
EU 2023/1115 Article 10(2)(f) and ISO 22095:2020.

Enumerations (15):
    - SCPType, SCPStatus, SegregationMethod, StorageType,
      TransportType, ProcessingLineType, ContaminationPathway,
      ContaminationSeverity, LabelType, LabelStatus,
      FacilityCapabilityLevel, ReportFormat, RiskClassification,
      ComplianceStatus, CleaningMethod

Core Models (12):
    - SegregationControlPoint, StorageZone, StorageEvent,
      TransportVehicle, TransportVerification, ProcessingLine,
      ChangeoverRecord, ContaminationEvent, LabelRecord,
      FacilityAssessment, ContaminationImpact, SegregationReport

Request Models (15):
    - RegisterSCPRequest, ValidateSCPRequest,
      RegisterStorageZoneRequest, RecordStorageEventRequest,
      RegisterVehicleRequest, VerifyTransportRequest,
      RegisterProcessingLineRequest, RecordChangeoverRequest,
      DetectContaminationRequest, RecordContaminationRequest,
      VerifyLabelsRequest, RunAssessmentRequest,
      GenerateReportRequest, SearchSCPRequest,
      BatchImportSCPRequest

Response Models (11):
    - SCPResponse, StorageAuditResponse,
      TransportVerificationResponse, ProcessingVerificationResponse,
      ContaminationDetectionResponse, ContaminationImpactResponse,
      LabelAuditResponse, AssessmentResponse, ReportResponse,
      BatchJobResponse, HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, and AGENT-EUDR-009
    Chain of Custody Agent.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier (GL-EUDR-SGV-010)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Maximum number of segregation control points in a single facility.
MAX_SEGREGATION_POINTS: int = 10_000

#: Maximum number of storage zones in a single facility.
MAX_STORAGE_ZONES: int = 5_000

#: Maximum number of transport vehicles tracked per operator.
MAX_TRANSPORT_VEHICLES: int = 50_000

#: Default temporal proximity threshold in hours for contamination detection.
DEFAULT_TEMPORAL_PROXIMITY_HOURS: float = 4.0

#: Default spatial proximity threshold in meters for contamination detection.
DEFAULT_SPATIAL_PROXIMITY_METERS: float = 5.0

#: Default reverification interval in days for segregation control points.
DEFAULT_REVERIFICATION_DAYS: int = 90

#: Default minimum changeover time in minutes between compliant/non-compliant.
DEFAULT_MIN_CHANGEOVER_MINUTES: int = 60

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Maximum number of previous cargoes tracked for transport verification.
MAX_PREVIOUS_CARGOES: int = 5

#: Maximum contamination event depth for impact tracing.
MAX_CONTAMINATION_DEPTH: int = 10

#: Facility assessment weight for layout component (0-1).
ASSESSMENT_LAYOUT_WEIGHT: float = 0.30

#: Facility assessment weight for protocols component (0-1).
ASSESSMENT_PROTOCOL_WEIGHT: float = 0.25

#: Facility assessment weight for history component (0-1).
ASSESSMENT_HISTORY_WEIGHT: float = 0.20

#: Facility assessment weight for labeling component (0-1).
ASSESSMENT_LABELING_WEIGHT: float = 0.15

#: Facility assessment weight for documentation component (0-1).
ASSESSMENT_DOCUMENTATION_WEIGHT: float = 0.10

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 100_000

#: EUDR-regulated primary commodities (Annex I).
PRIMARY_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

#: Facility assessment capability levels with description.
FACILITY_ASSESSMENT_LEVELS: Dict[str, str] = {
    "level_0": "No segregation capability - material freely mixed",
    "level_1": "Basic segregation - administrative separation only",
    "level_2": "Intermediate - physical barriers with shared equipment",
    "level_3": "Advanced - dedicated zones with some shared handling",
    "level_4": "High - fully dedicated zones, equipment, and personnel",
    "level_5": "Maximum - separate buildings/facilities per material type",
}


# =============================================================================
# Enumerations
# =============================================================================


class SCPType(str, Enum):
    """Type of segregation control point in the supply chain.

    Identifies the specific stage in the supply chain where
    segregation of EUDR-compliant vs non-compliant material
    must be verified. Each SCP type has distinct verification
    requirements and risk profiles.

    STORAGE: Fixed location where material is held (warehouses,
        silos, yards). Requires physical barriers or dedicated zones.
    TRANSPORT: Vehicle-based movement between facilities. Requires
        cleaning verification and cargo history tracking.
    PROCESSING: Production or processing line where material is
        transformed. Requires changeover protocols and flush procedures.
    HANDLING: Loading, unloading, and manual handling operations.
        Requires personnel training and labeling compliance.
    LOADING_UNLOADING: Specific loading/unloading points at docks,
        berths, or truck bays. Requires temporal separation controls.
    """

    STORAGE = "storage"
    TRANSPORT = "transport"
    PROCESSING = "processing"
    HANDLING = "handling"
    LOADING_UNLOADING = "loading_unloading"


class SCPStatus(str, Enum):
    """Current status of a segregation control point.

    Tracks the verification lifecycle of each control point to
    ensure continuous compliance monitoring per EUDR Article 10(2)(f).

    VERIFIED: SCP has passed verification and is within the
        reverification interval. Segregation is confirmed.
    UNVERIFIED: SCP has not yet been verified. Default state
        for newly registered control points.
    FAILED: SCP has failed verification. Segregation is not
        assured and corrective action is required.
    EXPIRED: SCP verification has expired beyond the
        reverification interval. Re-verification is required.
    PENDING_INSPECTION: SCP is awaiting on-site inspection to
        confirm or renew verification status.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    EXPIRED = "expired"
    PENDING_INSPECTION = "pending_inspection"


class SegregationMethod(str, Enum):
    """Physical method used to segregate compliant material.

    Defines the approach used at a segregation control point
    to ensure EUDR-compliant material is kept separate from
    non-compliant material. Methods range from complete facility
    separation to temporal controls.

    DEDICATED_FACILITY: Entirely separate facility exclusively
        for compliant material. Highest assurance level.
    PHYSICAL_BARRIER: Physical wall, fence, or partition within
        a shared facility. Prevents accidental mixing.
    SEALED_CONTAINER: Material stored in sealed, tamper-evident
        containers (e.g., locked IBCs, sealed bulk bags).
    TEMPORAL_SEPARATION: Same equipment/space used but with
        mandatory time gaps and cleaning between materials.
    DEDICATED_LINE: Dedicated processing line within a shared
        facility. No shared equipment with non-compliant lines.
    COLOR_CODED_ZONE: Zones marked with color codes to visually
        distinguish compliant from non-compliant areas.
    LOCKED_AREA: Restricted access area with key/card access
        control for compliant material only.
    SEPARATE_BUILDING: Separate building within the same site
        complex. High assurance with shared site services.
    """

    DEDICATED_FACILITY = "dedicated_facility"
    PHYSICAL_BARRIER = "physical_barrier"
    SEALED_CONTAINER = "sealed_container"
    TEMPORAL_SEPARATION = "temporal_separation"
    DEDICATED_LINE = "dedicated_line"
    COLOR_CODED_ZONE = "color_coded_zone"
    LOCKED_AREA = "locked_area"
    SEPARATE_BUILDING = "separate_building"


class StorageType(str, Enum):
    """Type of storage facility or unit for commodity segregation.

    Categorizes storage infrastructure based on physical
    characteristics that affect segregation capability and
    contamination risk profiles.

    SILO: Vertical cylindrical storage for bulk grain, coffee,
        cocoa beans. Requires dedicated fill/discharge points.
    WAREHOUSE_BAY: Partitioned bay within a larger warehouse.
        Separated by walls, curtains, or floor markings.
    TANK: Liquid storage tank for palm oil, soybean oil, or
        rubber latex. Requires dedicated piping.
    CONTAINER_YARD: Outdoor area for intermodal containers.
        Segregation via container placement and labeling.
    COLD_ROOM: Temperature-controlled storage. Common for
        cattle products (beef, leather).
    DRY_STORE: Climate-controlled dry storage for cocoa, coffee,
        rubber. Humidity and pest control.
    BONDED_AREA: Customs-bonded storage area with restricted
        access and regulatory oversight.
    OPEN_YARD: Uncovered outdoor storage area for timber,
        rubber bales. Weather exposure risk.
    COVERED_SHED: Semi-enclosed structure with roof cover.
        Used for intermediate storage and sorting.
    SEALED_UNIT: Self-contained sealed storage unit (e.g.,
        refrigerated container, sealed IBC).
    LOCKED_CAGE: Locked wire mesh or solid cage within a
        shared warehouse. Visual and physical separation.
    SEGREGATED_FLOOR: Entire floor of a multi-story warehouse
        dedicated to compliant material.
    """

    SILO = "silo"
    WAREHOUSE_BAY = "warehouse_bay"
    TANK = "tank"
    CONTAINER_YARD = "container_yard"
    COLD_ROOM = "cold_room"
    DRY_STORE = "dry_store"
    BONDED_AREA = "bonded_area"
    OPEN_YARD = "open_yard"
    COVERED_SHED = "covered_shed"
    SEALED_UNIT = "sealed_unit"
    LOCKED_CAGE = "locked_cage"
    SEGREGATED_FLOOR = "segregated_floor"


class TransportType(str, Enum):
    """Type of transport vehicle for commodity movement.

    Categorizes vehicles used to move EUDR commodities between
    supply chain points. Each type has distinct cleaning
    requirements and contamination risk profiles.

    BULK_TRUCK: Open or enclosed truck for loose bulk cargo.
        Requires thorough cleaning between loads.
    CONTAINER_TRUCK: Truck carrying intermodal containers.
        Containers provide sealed segregation.
    TANKER: Liquid tanker for palm oil, soybean oil, latex.
        Requires tank wash and heel verification.
    DRY_BULK_VESSEL: Bulk cargo vessel with holds for grain,
        cocoa, coffee, soya. Requires hold cleaning.
    CONTAINER_VESSEL: Container ship. Segregation via container
        placement and stowage planning.
    TANKER_VESSEL: Marine tanker for liquid commodities.
        Tank cleaning and grade-change protocols apply.
    RAIL_HOPPER: Rail hopper car for bulk grain or soya.
        Requires interior cleaning between loads.
    RAIL_CONTAINER: Rail flatcar carrying intermodal containers.
        Containers provide sealed segregation.
    BARGE: Inland waterway vessel for bulk or containerized
        cargo. Common for palm oil and timber transport.
    AIR_FREIGHT: Air cargo for high-value commodities.
        Containerized in ULDs with segregation by position.
    """

    BULK_TRUCK = "bulk_truck"
    CONTAINER_TRUCK = "container_truck"
    TANKER = "tanker"
    DRY_BULK_VESSEL = "dry_bulk_vessel"
    CONTAINER_VESSEL = "container_vessel"
    TANKER_VESSEL = "tanker_vessel"
    RAIL_HOPPER = "rail_hopper"
    RAIL_CONTAINER = "rail_container"
    BARGE = "barge"
    AIR_FREIGHT = "air_freight"


class ProcessingLineType(str, Enum):
    """Type of processing line for commodity transformation.

    Identifies the industrial processing operation that
    converts raw commodities into derived products. Each
    line type has specific changeover requirements to prevent
    cross-contamination between compliant and non-compliant runs.

    EXTRACTION: Oil extraction (palm oil, soybean oil).
    PRESSING: Mechanical pressing (cocoa butter, palm kernel oil).
    MILLING: Grain milling (soya, coffee, cocoa nibs).
    REFINING: Oil or product refining (palm oil, soybean oil).
    ROASTING: Heat-based roasting (cocoa beans, coffee beans).
    FERMENTING: Biological fermentation (cocoa, coffee, rubber).
    DRYING: Thermal or air drying (cocoa, coffee, rubber, wood).
    CUTTING: Physical cutting (timber, cattle hides).
    TANNING: Leather tanning process for cattle hides.
    SPINNING: Fiber spinning (rubber thread).
    SMELTING: Metal smelting (for derivative products).
    FRACTIONATION: Oil fractionation (palm oil derivatives).
    BLENDING_LINE: Blending or mixing line for multiple inputs.
    PACKAGING: Final packaging and containerization.
    GRADING: Quality grading and sorting line.
    """

    EXTRACTION = "extraction"
    PRESSING = "pressing"
    MILLING = "milling"
    REFINING = "refining"
    ROASTING = "roasting"
    FERMENTING = "fermenting"
    DRYING = "drying"
    CUTTING = "cutting"
    TANNING = "tanning"
    SPINNING = "spinning"
    SMELTING = "smelting"
    FRACTIONATION = "fractionation"
    BLENDING_LINE = "blending_line"
    PACKAGING = "packaging"
    GRADING = "grading"


class ContaminationPathway(str, Enum):
    """Pathway through which cross-contamination can occur.

    Identifies the mechanism by which non-compliant material
    may contaminate EUDR-compliant material. Each pathway has
    a distinct detection strategy and corrective action profile.

    SHARED_STORAGE: Compliant and non-compliant material stored
        in the same zone without adequate barriers.
    SHARED_TRANSPORT: Vehicle used for both compliant and
        non-compliant material without adequate cleaning.
    SHARED_PROCESSING: Processing line used for both material
        types without adequate changeover/flush.
    SHARED_EQUIPMENT: Common equipment (conveyors, forklifts,
        hoppers) used for both material types.
    TEMPORAL_OVERLAP: Compliant and non-compliant operations
        occurring too close together in time.
    ADJACENT_STORAGE: Compliant material stored adjacent to
        non-compliant without minimum separation distance.
    RESIDUAL_MATERIAL: Material residue from previous
        non-compliant processing remaining in equipment.
    HANDLING_ERROR: Human error during manual handling causing
        accidental mixing or mislabeling.
    LABELING_ERROR: Incorrect or missing labeling leading to
        material being placed in the wrong zone or stream.
    DOCUMENTATION_ERROR: Administrative error in documentation
        causing traceability loss or misallocation.
    """

    SHARED_STORAGE = "shared_storage"
    SHARED_TRANSPORT = "shared_transport"
    SHARED_PROCESSING = "shared_processing"
    SHARED_EQUIPMENT = "shared_equipment"
    TEMPORAL_OVERLAP = "temporal_overlap"
    ADJACENT_STORAGE = "adjacent_storage"
    RESIDUAL_MATERIAL = "residual_material"
    HANDLING_ERROR = "handling_error"
    LABELING_ERROR = "labeling_error"
    DOCUMENTATION_ERROR = "documentation_error"


class ContaminationSeverity(str, Enum):
    """Severity classification for contamination events.

    Determines the urgency of response and the regulatory
    impact of a contamination event on EUDR compliance status.

    CRITICAL: Total compliance loss. Affected material must be
        immediately quarantined and downgraded to non-compliant.
        Triggers mandatory incident report to competent authority.
    MAJOR: Significant compliance risk. Material must be
        quarantined pending investigation. Partial downgrade
        may be required based on contamination extent.
    MINOR: Limited compliance impact. Material can remain in
        compliant stream pending investigation, but must be
        flagged for enhanced monitoring.
    OBSERVATION: Informational finding that does not affect
        compliance status but indicates a process improvement
        opportunity. No material impact.
    """

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


class LabelType(str, Enum):
    """Type of physical label or sign used for segregation marking.

    Categorizes the visual identification markers that help
    personnel distinguish between compliant and non-compliant
    material, zones, and equipment.

    COMPLIANCE_TAG: Tag attached to individual items, bags,
        or containers indicating compliance status.
    ZONE_SIGN: Fixed sign posted at zone boundaries to
        identify compliant vs non-compliant areas.
    VEHICLE_PLACARD: Placard displayed on transport vehicles
        indicating current cargo compliance status.
    CONTAINER_SEAL_LABEL: Label applied to container seals
        with compliance status and batch reference.
    BATCH_STICKER: Sticker applied to batch packaging with
        batch ID and compliance status.
    PALLET_MARKER: Color-coded marker on pallets identifying
        compliance status of the load.
    SILO_SIGN: Sign posted on silo access points indicating
        compliance status of contents.
    PROCESSING_LINE_MARKER: Marker on processing line
        indicating current compliance run status.
    """

    COMPLIANCE_TAG = "compliance_tag"
    ZONE_SIGN = "zone_sign"
    VEHICLE_PLACARD = "vehicle_placard"
    CONTAINER_SEAL_LABEL = "container_seal_label"
    BATCH_STICKER = "batch_sticker"
    PALLET_MARKER = "pallet_marker"
    SILO_SIGN = "silo_sign"
    PROCESSING_LINE_MARKER = "processing_line_marker"


class LabelStatus(str, Enum):
    """Current status of a physical label or sign.

    Tracks whether labels are present, readable, and valid
    for segregation compliance purposes.

    APPLIED: Label has been applied and is in good condition.
    READABLE: Label is present and can be read but may show
        wear or partial damage.
    DAMAGED: Label is present but damaged to the extent that
        compliance status cannot be reliably determined.
    MISSING: Label is expected but not found at the designated
        location. Requires immediate corrective action.
    EXPIRED: Label validity period has passed. Requires
        replacement with a current label.
    """

    APPLIED = "applied"
    READABLE = "readable"
    DAMAGED = "damaged"
    MISSING = "missing"
    EXPIRED = "expired"


class FacilityCapabilityLevel(str, Enum):
    """Capability level of a facility for segregation operations.

    Six-tier classification system for assessing a facility's
    ability to maintain physical segregation of EUDR-compliant
    material from non-compliant material.

    LEVEL_0: No segregation capability. Material is freely mixed
        with no controls. Facility cannot handle compliant material.
    LEVEL_1: Basic administrative separation. Paper-based tracking
        but no physical barriers. High contamination risk.
    LEVEL_2: Intermediate capability. Physical barriers present
        but some equipment is shared between streams.
    LEVEL_3: Advanced capability. Dedicated zones with some
        shared handling at transfer points. Changeover protocols.
    LEVEL_4: High capability. Fully dedicated zones, equipment,
        and trained personnel. Minimal shared infrastructure.
    LEVEL_5: Maximum capability. Separate buildings or facilities
        for each material type. Zero shared infrastructure.
    """

    LEVEL_0 = "level_0"
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    LEVEL_5 = "level_5"


class ReportFormat(str, Enum):
    """Output format for segregation verification reports.

    JSON: Machine-readable JSON format for API integration.
    PDF: Human-readable PDF format for regulatory submission.
    CSV: Tabular CSV format for spreadsheet analysis.
    EUDR_XML: EU Information System XML schema for DDS submission.
    """

    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EUDR_XML = "eudr_xml"


class RiskClassification(str, Enum):
    """Risk classification for segregation control points.

    LOW: Minimal risk of cross-contamination. Strong physical
        controls and dedicated infrastructure.
    MEDIUM: Moderate risk requiring regular monitoring.
        Adequate controls with some shared elements.
    HIGH: Elevated risk requiring enhanced monitoring and
        frequent re-verification.
    CRITICAL: Unacceptable risk level. Immediate corrective
        action required to prevent compliance loss.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance status for storage zones and facilities.

    COMPLIANT: Zone or facility is verified for handling
        EUDR-compliant material only.
    NON_COMPLIANT: Zone or facility is used for non-compliant
        material only.
    PENDING: Compliance status is being evaluated and has
        not yet been determined.
    UNKNOWN: Compliance status cannot be determined due to
        insufficient data or documentation.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNKNOWN = "unknown"


class CleaningMethod(str, Enum):
    """Method used to clean transport vehicles or equipment.

    Defines the cleaning technique applied between loads or
    processing runs to remove residual material and prevent
    cross-contamination.

    POWER_WASH: High-pressure water wash for bulk containers,
        truck bodies, and warehouse floors.
    STEAM_CLEAN: Steam-based cleaning for tanks, vessels, and
        processing equipment.
    FUMIGATION: Chemical fumigation for pest control and
        decontamination of storage facilities.
    FLUSH: Liquid flush through pipelines, tanks, and
        processing equipment (oil, water, or solvent).
    SWEEP_WASH: Mechanical sweeping followed by water wash.
        Common for bulk trucks and rail hoppers.
    COMPRESSED_AIR: Compressed air blow-out for dry cargo
        holds, hoppers, and conveyor systems.
    TANK_WASH: Specialized tank washing for liquid tankers
        with CIP (clean-in-place) systems.
    """

    POWER_WASH = "power_wash"
    STEAM_CLEAN = "steam_clean"
    FUMIGATION = "fumigation"
    FLUSH = "flush"
    SWEEP_WASH = "sweep_wash"
    COMPRESSED_AIR = "compressed_air"
    TANK_WASH = "tank_wash"


# =============================================================================
# Core Models
# =============================================================================


class SegregationControlPoint(BaseModel):
    """A segregation control point in the supply chain.

    Represents a specific location or stage where physical
    segregation of EUDR-compliant vs non-compliant material
    must be verified and maintained. Each SCP has a type,
    method, status, and risk classification.

    Attributes:
        scp_id: Unique identifier for this control point.
        facility_id: Identifier of the facility hosting this SCP.
        location_lat: Latitude of the SCP location.
        location_lon: Longitude of the SCP location.
        scp_type: Type of segregation control point.
        commodity: EUDR commodity handled at this SCP.
        capacity_kg: Maximum capacity in kilograms.
        segregation_method: Physical method used for segregation.
        status: Current verification status.
        risk_classification: Risk classification level.
        compliance_score: Compliance score (0.0-100.0).
        verification_date: Date of last verification.
        next_verification_date: Date when re-verification is due.
        notes: Free-text notes or observations.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash for audit trail.
        created_at: UTC timestamp when the record was created.
        updated_at: UTC timestamp when the record was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    scp_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this control point",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the facility hosting this SCP",
    )
    location_lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Latitude of the SCP location",
    )
    location_lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Longitude of the SCP location",
    )
    scp_type: SCPType = Field(
        ...,
        description="Type of segregation control point",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="EUDR commodity handled at this SCP",
    )
    capacity_kg: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Maximum capacity in kilograms",
    )
    segregation_method: SegregationMethod = Field(
        ...,
        description="Physical method used for segregation",
    )
    status: SCPStatus = Field(
        default=SCPStatus.UNVERIFIED,
        description="Current verification status",
    )
    risk_classification: RiskClassification = Field(
        default=RiskClassification.MEDIUM,
        description="Risk classification level",
    )
    compliance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Compliance score (0.0-100.0)",
    )
    verification_date: Optional[datetime] = Field(
        None,
        description="Date of last verification",
    )
    next_verification_date: Optional[datetime] = Field(
        None,
        description="Date when re-verification is due",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes or observations",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was last updated",
    )


class StorageZone(BaseModel):
    """A storage zone within a facility for segregated material.

    Represents a physically delineated area within a storage
    facility that is designated for either compliant or
    non-compliant material. Tracks capacity, occupancy, barrier
    type, and adjacent zones for contamination risk assessment.

    Attributes:
        zone_id: Unique identifier for this storage zone.
        facility_id: Identifier of the facility containing this zone.
        zone_name: Human-readable name of the storage zone.
        storage_type: Type of storage infrastructure.
        compliance_status: Compliance designation of this zone.
        barrier_type: Type of physical barrier separating this zone.
        capacity_kg: Maximum storage capacity in kilograms.
        current_occupancy_kg: Current quantity stored in kilograms.
        adjacent_zones: List of zone IDs that are physically adjacent.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the record was created.
        updated_at: UTC timestamp when the record was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    zone_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this storage zone",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the facility containing this zone",
    )
    zone_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name of the storage zone",
    )
    storage_type: StorageType = Field(
        ...,
        description="Type of storage infrastructure",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
        description="Compliance designation of this zone",
    )
    barrier_type: Optional[str] = Field(
        None,
        description="Type of physical barrier separating this zone",
    )
    capacity_kg: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Maximum storage capacity in kilograms",
    )
    current_occupancy_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Current quantity stored in kilograms",
    )
    adjacent_zones: List[str] = Field(
        default_factory=list,
        description="List of zone IDs that are physically adjacent",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was last updated",
    )


class StorageEvent(BaseModel):
    """An event recording material movement in a storage zone.

    Tracks inbound and outbound material movements for a
    specific storage zone, enabling audit trail reconstruction
    and occupancy monitoring.

    Attributes:
        event_id: Unique identifier for this storage event.
        zone_id: Identifier of the storage zone.
        event_type: Type of event (inbound, outbound, transfer, audit).
        batch_id: Identifier of the material batch.
        quantity_kg: Quantity moved in kilograms.
        timestamp: UTC timestamp when the event occurred.
        operator_id: Identifier of the operator performing the action.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this storage event",
    )
    zone_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the storage zone",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Type of event (inbound, outbound, transfer, audit)",
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the material batch",
    )
    quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Quantity moved in kilograms",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the event occurred",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the operator performing the action",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )


class TransportVehicle(BaseModel):
    """A transport vehicle tracked for segregation verification.

    Represents a vehicle (truck, vessel, railcar, etc.) that
    transports EUDR commodities. Tracks cleaning status, cargo
    history, and dedication status for contamination risk assessment.

    Attributes:
        vehicle_id: Unique identifier for this vehicle.
        vehicle_type: Type of transport vehicle.
        owner_operator_id: Identifier of the vehicle owner/operator.
        dedicated_status: Whether vehicle is dedicated to compliant cargo.
        last_cargo_type: Compliance status of the last cargo carried.
        last_cleaning_date: Date of last cleaning operation.
        cleaning_method: Method used for last cleaning.
        cargo_history: List of recent cargo compliance statuses.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the record was created.
        updated_at: UTC timestamp when the record was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    vehicle_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this vehicle",
    )
    vehicle_type: TransportType = Field(
        ...,
        description="Type of transport vehicle",
    )
    owner_operator_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the vehicle owner/operator",
    )
    dedicated_status: bool = Field(
        default=False,
        description="Whether vehicle is dedicated to compliant cargo",
    )
    last_cargo_type: Optional[str] = Field(
        None,
        description="Compliance status of the last cargo carried",
    )
    last_cleaning_date: Optional[datetime] = Field(
        None,
        description="Date of last cleaning operation",
    )
    cleaning_method: Optional[CleaningMethod] = Field(
        None,
        description="Method used for last cleaning",
    )
    cargo_history: List[str] = Field(
        default_factory=list,
        description="List of recent cargo compliance statuses",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was last updated",
    )


class TransportVerification(BaseModel):
    """Verification record for a transport shipment.

    Records the outcome of verifying that a transport vehicle
    is suitable for carrying EUDR-compliant material, including
    cleaning status, seal integrity, and cargo history review.

    Attributes:
        verification_id: Unique identifier for this verification.
        vehicle_id: Identifier of the verified vehicle.
        batch_id: Identifier of the batch being transported.
        cleaning_verified: Whether cleaning has been verified.
        seal_number: Container or cargo seal number.
        seal_intact: Whether the seal is intact.
        previous_cargoes: List of previous cargo types.
        score: Verification score (0.0-100.0).
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        timestamp: UTC timestamp of the verification.
    """

    model_config = ConfigDict(from_attributes=True)

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this verification",
    )
    vehicle_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the verified vehicle",
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the batch being transported",
    )
    cleaning_verified: bool = Field(
        default=False,
        description="Whether cleaning has been verified",
    )
    seal_number: Optional[str] = Field(
        None,
        description="Container or cargo seal number",
    )
    seal_intact: Optional[bool] = Field(
        None,
        description="Whether the seal is intact",
    )
    previous_cargoes: List[str] = Field(
        default_factory=list,
        description="List of previous cargo types",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Verification score (0.0-100.0)",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the verification",
    )


class ProcessingLine(BaseModel):
    """A processing line tracked for segregation verification.

    Represents a production or processing line within a facility
    that handles EUDR commodities. Tracks dedication status,
    changeover history, and capacity for contamination risk
    assessment.

    Attributes:
        line_id: Unique identifier for this processing line.
        facility_id: Identifier of the facility hosting this line.
        line_type: Type of processing line.
        commodity: Primary commodity processed on this line.
        capacity_kg_per_hour: Line throughput capacity in kg/hour.
        dedicated_status: Whether line is dedicated to compliant material.
        last_changeover_date: Date of last changeover operation.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the record was created.
        updated_at: UTC timestamp when the record was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    line_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this processing line",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the facility hosting this line",
    )
    line_type: ProcessingLineType = Field(
        ...,
        description="Type of processing line",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Primary commodity processed on this line",
    )
    capacity_kg_per_hour: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Line throughput capacity in kg/hour",
    )
    dedicated_status: bool = Field(
        default=False,
        description="Whether line is dedicated to compliant material",
    )
    last_changeover_date: Optional[datetime] = Field(
        None,
        description="Date of last changeover operation",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was last updated",
    )


class ChangeoverRecord(BaseModel):
    """Record of a processing line changeover operation.

    Captures the details of switching a processing line between
    compliant and non-compliant material runs, including flush
    procedures, cleaning method, and verification.

    Attributes:
        changeover_id: Unique identifier for this changeover record.
        line_id: Identifier of the processing line.
        previous_batch_type: Compliance status of the previous batch.
        next_batch_type: Compliance status of the next batch.
        flush_volume_liters: Volume of flush material used in liters.
        flush_duration_minutes: Duration of the flush in minutes.
        cleaning_method: Cleaning method used during changeover.
        verified_by: Identifier of the person who verified changeover.
        timestamp: UTC timestamp when the changeover occurred.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(from_attributes=True)

    changeover_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this changeover record",
    )
    line_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the processing line",
    )
    previous_batch_type: str = Field(
        ...,
        min_length=1,
        description="Compliance status of the previous batch",
    )
    next_batch_type: str = Field(
        ...,
        min_length=1,
        description="Compliance status of the next batch",
    )
    flush_volume_liters: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Volume of flush material used in liters",
    )
    flush_duration_minutes: Optional[int] = Field(
        None,
        ge=0,
        description="Duration of the flush in minutes",
    )
    cleaning_method: Optional[CleaningMethod] = Field(
        None,
        description="Cleaning method used during changeover",
    )
    verified_by: Optional[str] = Field(
        None,
        description="Identifier of the person who verified changeover",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the changeover occurred",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )


class ContaminationEvent(BaseModel):
    """Record of a cross-contamination event.

    Captures the details of an event where EUDR-compliant material
    may have been contaminated by non-compliant material, including
    pathway, severity, affected batches, root cause, and corrective
    action taken.

    Attributes:
        event_id: Unique identifier for this contamination event.
        facility_id: Identifier of the facility where event occurred.
        pathway_type: Contamination pathway identified.
        severity: Severity classification of the event.
        affected_batch_ids: List of batch IDs affected by contamination.
        affected_quantity_kg: Total quantity affected in kilograms.
        timestamp: UTC timestamp when the event was detected.
        root_cause: Root cause description.
        corrective_action: Corrective action taken or planned.
        resolved: Whether the event has been resolved.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the record was created.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this contamination event",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the facility where event occurred",
    )
    pathway_type: ContaminationPathway = Field(
        ...,
        description="Contamination pathway identified",
    )
    severity: ContaminationSeverity = Field(
        ...,
        description="Severity classification of the event",
    )
    affected_batch_ids: List[str] = Field(
        default_factory=list,
        description="List of batch IDs affected by contamination",
    )
    affected_quantity_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total quantity affected in kilograms",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the event was detected",
    )
    root_cause: Optional[str] = Field(
        None,
        description="Root cause description",
    )
    corrective_action: Optional[str] = Field(
        None,
        description="Corrective action taken or planned",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the event has been resolved",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was created",
    )


class LabelRecord(BaseModel):
    """Record of a physical label or sign for segregation marking.

    Tracks the lifecycle of labeling elements used to visually
    distinguish compliant from non-compliant material, zones,
    and equipment.

    Attributes:
        label_id: Unique identifier for this label record.
        scp_id: Identifier of the associated segregation control point.
        label_type: Type of physical label or sign.
        status: Current status of the label.
        content_fields: Key-value pairs of label content.
        placement_verified: Whether label placement has been verified.
        applied_date: Date when the label was applied.
        verified_date: Date when the label was last verified.
        expiry_date: Date when the label expires.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(from_attributes=True)

    label_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this label record",
    )
    scp_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the associated segregation control point",
    )
    label_type: LabelType = Field(
        ...,
        description="Type of physical label or sign",
    )
    status: LabelStatus = Field(
        default=LabelStatus.APPLIED,
        description="Current status of the label",
    )
    content_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs of label content",
    )
    placement_verified: bool = Field(
        default=False,
        description="Whether label placement has been verified",
    )
    applied_date: Optional[datetime] = Field(
        None,
        description="Date when the label was applied",
    )
    verified_date: Optional[datetime] = Field(
        None,
        description="Date when the label was last verified",
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="Date when the label expires",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )


class FacilityAssessment(BaseModel):
    """Assessment of a facility's segregation capability.

    Captures a comprehensive evaluation of a facility's ability
    to maintain physical segregation of EUDR-compliant material.
    Uses a weighted scoring system across five dimensions:
    layout, protocols, history, labeling, and documentation.

    Attributes:
        assessment_id: Unique identifier for this assessment.
        facility_id: Identifier of the assessed facility.
        capability_level: Overall capability level assigned.
        layout_score: Score for physical layout (0.0-100.0).
        protocol_score: Score for segregation protocols (0.0-100.0).
        history_score: Score for compliance history (0.0-100.0).
        labeling_score: Score for labeling compliance (0.0-100.0).
        documentation_score: Score for documentation (0.0-100.0).
        overall_score: Weighted overall score (0.0-100.0).
        recommendations: List of improvement recommendations.
        assessment_date: Date when the assessment was conducted.
        assessor_id: Identifier of the assessor.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the record was created.
    """

    model_config = ConfigDict(from_attributes=True)

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this assessment",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the assessed facility",
    )
    capability_level: FacilityCapabilityLevel = Field(
        ...,
        description="Overall capability level assigned",
    )
    layout_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score for physical layout (0.0-100.0)",
    )
    protocol_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score for segregation protocols (0.0-100.0)",
    )
    history_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score for compliance history (0.0-100.0)",
    )
    labeling_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score for labeling compliance (0.0-100.0)",
    )
    documentation_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score for documentation (0.0-100.0)",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Weighted overall score (0.0-100.0)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of improvement recommendations",
    )
    assessment_date: datetime = Field(
        default_factory=_utcnow,
        description="Date when the assessment was conducted",
    )
    assessor_id: Optional[str] = Field(
        None,
        description="Identifier of the assessor",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was created",
    )


class ContaminationImpact(BaseModel):
    """Impact assessment of a contamination event.

    Traces the downstream impact of a contamination event on
    subsequent batches, tracking affected quantities and status
    downgrades cascading through the supply chain.

    Attributes:
        impact_id: Unique identifier for this impact record.
        contamination_event_id: Identifier of the originating event.
        downstream_batch_ids: Batches affected downstream.
        total_affected_quantity_kg: Total downstream quantity affected.
        status_downgrades: List of status downgrade descriptions.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        timestamp: UTC timestamp of the impact assessment.
    """

    model_config = ConfigDict(from_attributes=True)

    impact_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this impact record",
    )
    contamination_event_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the originating contamination event",
    )
    downstream_batch_ids: List[str] = Field(
        default_factory=list,
        description="Batches affected downstream",
    )
    total_affected_quantity_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total downstream quantity affected in kilograms",
    )
    status_downgrades: List[str] = Field(
        default_factory=list,
        description="List of status downgrade descriptions",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the impact assessment",
    )


class SegregationReport(BaseModel):
    """A segregation verification report.

    Represents a generated report capturing segregation
    verification results, facility assessments, contamination
    events, and compliance status across the supply chain.

    Attributes:
        report_id: Unique identifier for this report.
        report_type: Type of report (facility, scp, contamination, etc.).
        facility_id: Identifier of the facility (if facility-specific).
        format: Output format of the report.
        generated_at: UTC timestamp when the report was generated.
        data: Report data payload.
        file_path: File path or storage reference for the report.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    report_type: str = Field(
        ...,
        min_length=1,
        description="Type of report (facility, scp, contamination, etc.)",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Identifier of the facility (if facility-specific)",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format of the report",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the report was generated",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report data payload",
    )
    file_path: Optional[str] = Field(
        None,
        description="File path or storage reference for the report",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )


# =============================================================================
# Request Models
# =============================================================================


class RegisterSCPRequest(BaseModel):
    """Request to register a new segregation control point.

    Attributes:
        facility_id: Facility hosting the SCP.
        scp_type: Type of segregation control point.
        commodity: EUDR commodity handled.
        segregation_method: Physical segregation method.
        location_lat: Latitude of the SCP.
        location_lon: Longitude of the SCP.
        capacity_kg: Maximum capacity in kilograms.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    scp_type: SCPType = Field(...)
    commodity: str = Field(..., min_length=1)
    segregation_method: SegregationMethod = Field(...)
    location_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    location_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    capacity_kg: Optional[Decimal] = Field(None, gt=0)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidateSCPRequest(BaseModel):
    """Request to validate a segregation control point.

    Attributes:
        scp_id: Identifier of the SCP to validate.
        verified_by: Identifier of the verifier.
        inspection_notes: Notes from the inspection.
        compliance_score: Score assigned by verifier (0-100).
        risk_classification: Risk classification assigned.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    scp_id: str = Field(..., min_length=1)
    verified_by: str = Field(..., min_length=1)
    inspection_notes: Optional[str] = Field(None)
    compliance_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    risk_classification: Optional[RiskClassification] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegisterStorageZoneRequest(BaseModel):
    """Request to register a new storage zone.

    Attributes:
        facility_id: Facility containing the zone.
        zone_name: Human-readable name of the zone.
        storage_type: Type of storage infrastructure.
        compliance_status: Compliance designation.
        barrier_type: Type of physical barrier.
        capacity_kg: Maximum capacity in kilograms.
        adjacent_zones: List of adjacent zone IDs.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    zone_name: str = Field(..., min_length=1)
    storage_type: StorageType = Field(...)
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
    )
    barrier_type: Optional[str] = Field(None)
    capacity_kg: Optional[Decimal] = Field(None, gt=0)
    adjacent_zones: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecordStorageEventRequest(BaseModel):
    """Request to record a storage event.

    Attributes:
        zone_id: Identifier of the storage zone.
        event_type: Type of event (inbound, outbound, transfer, audit).
        batch_id: Identifier of the material batch.
        quantity_kg: Quantity moved in kilograms.
        operator_id: Identifier of the operator.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    zone_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    batch_id: str = Field(..., min_length=1)
    quantity_kg: Decimal = Field(..., gt=0)
    operator_id: str = Field(..., min_length=1)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegisterVehicleRequest(BaseModel):
    """Request to register a transport vehicle.

    Attributes:
        vehicle_type: Type of transport vehicle.
        owner_operator_id: Identifier of the owner/operator.
        dedicated_status: Whether vehicle is dedicated to compliant cargo.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    vehicle_type: TransportType = Field(...)
    owner_operator_id: str = Field(..., min_length=1)
    dedicated_status: bool = Field(default=False)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerifyTransportRequest(BaseModel):
    """Request to verify transport vehicle segregation.

    Attributes:
        vehicle_id: Identifier of the vehicle to verify.
        batch_id: Identifier of the batch being transported.
        cleaning_verified: Whether cleaning has been verified.
        seal_number: Container or cargo seal number.
        seal_intact: Whether the seal is intact.
        cleaning_method: Cleaning method used.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    vehicle_id: str = Field(..., min_length=1)
    batch_id: str = Field(..., min_length=1)
    cleaning_verified: bool = Field(default=False)
    seal_number: Optional[str] = Field(None)
    seal_intact: Optional[bool] = Field(None)
    cleaning_method: Optional[CleaningMethod] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegisterProcessingLineRequest(BaseModel):
    """Request to register a processing line.

    Attributes:
        facility_id: Facility hosting the line.
        line_type: Type of processing line.
        commodity: Primary commodity processed.
        capacity_kg_per_hour: Throughput capacity in kg/hour.
        dedicated_status: Whether line is dedicated to compliant material.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    line_type: ProcessingLineType = Field(...)
    commodity: str = Field(..., min_length=1)
    capacity_kg_per_hour: Optional[Decimal] = Field(None, gt=0)
    dedicated_status: bool = Field(default=False)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecordChangeoverRequest(BaseModel):
    """Request to record a processing line changeover.

    Attributes:
        line_id: Identifier of the processing line.
        previous_batch_type: Compliance status of previous batch.
        next_batch_type: Compliance status of next batch.
        flush_volume_liters: Volume of flush material in liters.
        flush_duration_minutes: Duration of flush in minutes.
        cleaning_method: Cleaning method used.
        verified_by: Identifier of the verifier.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    line_id: str = Field(..., min_length=1)
    previous_batch_type: str = Field(..., min_length=1)
    next_batch_type: str = Field(..., min_length=1)
    flush_volume_liters: Optional[Decimal] = Field(None, ge=0)
    flush_duration_minutes: Optional[int] = Field(None, ge=0)
    cleaning_method: Optional[CleaningMethod] = Field(None)
    verified_by: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DetectContaminationRequest(BaseModel):
    """Request to run contamination detection on a facility.

    Attributes:
        facility_id: Identifier of the facility to scan.
        temporal_proximity_hours: Time proximity threshold for detection.
        spatial_proximity_meters: Distance proximity threshold.
        include_storage: Whether to scan storage zones.
        include_transport: Whether to scan transport records.
        include_processing: Whether to scan processing lines.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    temporal_proximity_hours: float = Field(
        default=DEFAULT_TEMPORAL_PROXIMITY_HOURS,
        gt=0.0,
    )
    spatial_proximity_meters: float = Field(
        default=DEFAULT_SPATIAL_PROXIMITY_METERS,
        gt=0.0,
    )
    include_storage: bool = Field(default=True)
    include_transport: bool = Field(default=True)
    include_processing: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecordContaminationRequest(BaseModel):
    """Request to record a contamination event.

    Attributes:
        facility_id: Facility where contamination occurred.
        pathway_type: Contamination pathway.
        severity: Severity classification.
        affected_batch_ids: Batches affected.
        affected_quantity_kg: Quantity affected in kilograms.
        root_cause: Root cause description.
        corrective_action: Corrective action taken or planned.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    pathway_type: ContaminationPathway = Field(...)
    severity: ContaminationSeverity = Field(...)
    affected_batch_ids: List[str] = Field(default_factory=list)
    affected_quantity_kg: Decimal = Field(default=Decimal("0"), ge=0)
    root_cause: Optional[str] = Field(None)
    corrective_action: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerifyLabelsRequest(BaseModel):
    """Request to verify labels at a segregation control point.

    Attributes:
        scp_id: Identifier of the SCP to verify labels for.
        label_types: Specific label types to check (empty = all).
        verified_by: Identifier of the verifier.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    scp_id: str = Field(..., min_length=1)
    label_types: List[LabelType] = Field(default_factory=list)
    verified_by: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunAssessmentRequest(BaseModel):
    """Request to run a facility segregation assessment.

    Attributes:
        facility_id: Identifier of the facility to assess.
        assessor_id: Identifier of the assessor.
        layout_score: Score for physical layout (0-100).
        protocol_score: Score for segregation protocols (0-100).
        history_score: Score for compliance history (0-100).
        labeling_score: Score for labeling compliance (0-100).
        documentation_score: Score for documentation (0-100).
        recommendations: List of recommendations.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    assessor_id: Optional[str] = Field(None)
    layout_score: float = Field(..., ge=0.0, le=100.0)
    protocol_score: float = Field(..., ge=0.0, le=100.0)
    history_score: float = Field(..., ge=0.0, le=100.0)
    labeling_score: float = Field(..., ge=0.0, le=100.0)
    documentation_score: float = Field(..., ge=0.0, le=100.0)
    recommendations: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerateReportRequest(BaseModel):
    """Request to generate a segregation verification report.

    Attributes:
        facility_id: Facility to report on (optional).
        report_type: Type of report (facility, scp, contamination).
        report_format: Desired output format.
        date_from: Report start date.
        date_to: Report end date.
        include_assessments: Include facility assessments.
        include_contamination: Include contamination events.
        include_labels: Include labeling audit results.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: Optional[str] = Field(None)
    report_type: str = Field(default="facility")
    report_format: ReportFormat = Field(default=ReportFormat.JSON)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    include_assessments: bool = Field(default=True)
    include_contamination: bool = Field(default=True)
    include_labels: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchSCPRequest(BaseModel):
    """Request to search for segregation control points.

    Attributes:
        facility_id: Filter by facility.
        scp_type: Filter by SCP type.
        commodity: Filter by commodity.
        status: Filter by status.
        risk_classification: Filter by risk classification.
        limit: Maximum results to return.
        offset: Offset for pagination.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: Optional[str] = Field(None)
    scp_type: Optional[SCPType] = Field(None)
    commodity: Optional[str] = Field(None)
    status: Optional[SCPStatus] = Field(None)
    risk_classification: Optional[RiskClassification] = Field(None)
    limit: int = Field(default=100, ge=1, le=10_000)
    offset: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchImportSCPRequest(BaseModel):
    """Request to batch import segregation control points.

    Attributes:
        items: List of SCP registration requests to import.
        skip_validation: Whether to skip validation for speed.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    items: List[RegisterSCPRequest] = Field(..., min_length=1)
    skip_validation: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("items")
    @classmethod
    def validate_batch_size(
        cls, v: List[RegisterSCPRequest],
    ) -> List[RegisterSCPRequest]:
        """Ensure batch size does not exceed maximum."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        return v


# =============================================================================
# Response Models
# =============================================================================


class SCPResponse(BaseModel):
    """Response after SCP registration or validation.

    Attributes:
        scp_id: Identifier of the SCP.
        facility_id: Facility hosting the SCP.
        scp_type: Type of SCP.
        status: Current status.
        compliance_score: Compliance score (0-100).
        risk_classification: Risk level.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    scp_id: str = Field(...)
    facility_id: str = Field(...)
    scp_type: SCPType = Field(...)
    status: SCPStatus = Field(...)
    compliance_score: float = Field(default=0.0)
    risk_classification: RiskClassification = Field(
        default=RiskClassification.MEDIUM,
    )
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class StorageAuditResponse(BaseModel):
    """Response after a storage zone audit.

    Attributes:
        zone_id: Identifier of the audited zone.
        facility_id: Facility containing the zone.
        compliance_status: Compliance status of the zone.
        occupancy_pct: Current occupancy percentage.
        adjacent_risk: Whether adjacent zones pose contamination risk.
        findings: List of audit findings.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the audit.
    """

    model_config = ConfigDict(from_attributes=True)

    zone_id: str = Field(...)
    facility_id: str = Field(...)
    compliance_status: ComplianceStatus = Field(...)
    occupancy_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    adjacent_risk: bool = Field(default=False)
    findings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class TransportVerificationResponse(BaseModel):
    """Response after transport vehicle verification.

    Attributes:
        verification_id: Identifier of the verification.
        vehicle_id: Identifier of the verified vehicle.
        batch_id: Identifier of the batch.
        cleaning_verified: Whether cleaning was verified.
        seal_intact: Whether seal is intact.
        score: Verification score (0-100).
        risk_factors: List of identified risk factors.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the verification.
    """

    model_config = ConfigDict(from_attributes=True)

    verification_id: str = Field(...)
    vehicle_id: str = Field(...)
    batch_id: str = Field(...)
    cleaning_verified: bool = Field(...)
    seal_intact: Optional[bool] = Field(None)
    score: float = Field(default=0.0)
    risk_factors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class ProcessingVerificationResponse(BaseModel):
    """Response after processing line verification.

    Attributes:
        line_id: Identifier of the processing line.
        facility_id: Facility hosting the line.
        dedicated_status: Whether line is dedicated.
        changeover_compliant: Whether changeover met requirements.
        changeover_duration_minutes: Duration of changeover.
        flush_adequate: Whether flush volume was adequate.
        findings: List of verification findings.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the verification.
    """

    model_config = ConfigDict(from_attributes=True)

    line_id: str = Field(...)
    facility_id: str = Field(...)
    dedicated_status: bool = Field(...)
    changeover_compliant: bool = Field(default=False)
    changeover_duration_minutes: Optional[int] = Field(None)
    flush_adequate: Optional[bool] = Field(None)
    findings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class ContaminationDetectionResponse(BaseModel):
    """Response after contamination detection scan.

    Attributes:
        facility_id: Identifier of the scanned facility.
        events_detected: Number of contamination events detected.
        critical_count: Number of critical severity events.
        major_count: Number of major severity events.
        minor_count: Number of minor severity events.
        observation_count: Number of observations.
        events: List of detected contamination events.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the scan.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(...)
    events_detected: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    major_count: int = Field(default=0, ge=0)
    minor_count: int = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    events: List[ContaminationEvent] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class ContaminationImpactResponse(BaseModel):
    """Response after contamination impact assessment.

    Attributes:
        impact_id: Identifier of the impact assessment.
        contamination_event_id: Originating contamination event.
        downstream_batches_affected: Number of downstream batches affected.
        total_affected_quantity_kg: Total affected quantity.
        downgrades_applied: Number of status downgrades applied.
        impact: The full impact assessment record.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the assessment.
    """

    model_config = ConfigDict(from_attributes=True)

    impact_id: str = Field(...)
    contamination_event_id: str = Field(...)
    downstream_batches_affected: int = Field(default=0, ge=0)
    total_affected_quantity_kg: Decimal = Field(default=Decimal("0"), ge=0)
    downgrades_applied: int = Field(default=0, ge=0)
    impact: Optional[ContaminationImpact] = Field(None)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class LabelAuditResponse(BaseModel):
    """Response after label verification audit.

    Attributes:
        scp_id: Identifier of the audited SCP.
        labels_checked: Total labels checked.
        labels_compliant: Number of compliant labels.
        labels_missing: Number of missing labels.
        labels_damaged: Number of damaged labels.
        labels_expired: Number of expired labels.
        compliance_pct: Label compliance percentage.
        findings: List of label audit findings.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the audit.
    """

    model_config = ConfigDict(from_attributes=True)

    scp_id: str = Field(...)
    labels_checked: int = Field(default=0, ge=0)
    labels_compliant: int = Field(default=0, ge=0)
    labels_missing: int = Field(default=0, ge=0)
    labels_damaged: int = Field(default=0, ge=0)
    labels_expired: int = Field(default=0, ge=0)
    compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    findings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class AssessmentResponse(BaseModel):
    """Response after facility assessment.

    Attributes:
        assessment_id: Identifier of the assessment.
        facility_id: Identifier of the assessed facility.
        capability_level: Assigned capability level.
        overall_score: Weighted overall score (0-100).
        layout_score: Layout component score.
        protocol_score: Protocol component score.
        history_score: History component score.
        labeling_score: Labeling component score.
        documentation_score: Documentation component score.
        recommendations: List of recommendations.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the assessment.
    """

    model_config = ConfigDict(from_attributes=True)

    assessment_id: str = Field(...)
    facility_id: str = Field(...)
    capability_level: FacilityCapabilityLevel = Field(...)
    overall_score: float = Field(...)
    layout_score: float = Field(...)
    protocol_score: float = Field(...)
    history_score: float = Field(...)
    labeling_score: float = Field(...)
    documentation_score: float = Field(...)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class ReportResponse(BaseModel):
    """Response after report generation.

    Attributes:
        report_id: Identifier of the generated report.
        report_type: Type of report.
        report_format: Output format.
        file_reference: Storage reference for the report file.
        file_size_bytes: Size of the generated report.
        record_count: Number of records included.
        date_from: Report period start.
        date_to: Report period end.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the generation.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    report_type: str = Field(...)
    report_format: ReportFormat = Field(...)
    file_reference: Optional[str] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    record_count: int = Field(default=0, ge=0)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=_utcnow)


class BatchJobResponse(BaseModel):
    """Response for batch processing jobs.

    Attributes:
        total_records: Total records in the batch.
        successful: Number of successfully processed records.
        failed: Number of failed records.
        errors: List of error descriptions.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash of the batch job.
        timestamp: UTC timestamp of the batch completion.
    """

    model_config = ConfigDict(from_attributes=True)

    total_records: int = Field(..., ge=0)
    successful: int = Field(..., ge=0)
    failed: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(...)
    provenance_hash: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def validate_counts(self) -> BatchJobResponse:
        """Ensure successful + failed equals total_records."""
        if self.successful + self.failed != self.total_records:
            raise ValueError(
                f"successful ({self.successful}) + failed ({self.failed}) "
                f"must equal total_records ({self.total_records})"
            )
        return self


class HealthResponse(BaseModel):
    """Health check response for the segregation verifier service.

    Attributes:
        status: Service health status (healthy, degraded, unhealthy).
        version: Service version string.
        agent_id: Agent identifier.
        uptime_seconds: Service uptime in seconds.
        active_scps: Number of active segregation control points.
        database_connected: Whether database is connected.
        redis_connected: Whether Redis cache is connected.
        timestamp: UTC timestamp of the health check.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="healthy")
    version: str = Field(default=VERSION)
    agent_id: str = Field(default="GL-EUDR-SGV-010")
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    active_scps: int = Field(default=0, ge=0)
    database_connected: bool = Field(default=False)
    redis_connected: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_SEGREGATION_POINTS",
    "MAX_STORAGE_ZONES",
    "MAX_TRANSPORT_VEHICLES",
    "DEFAULT_TEMPORAL_PROXIMITY_HOURS",
    "DEFAULT_SPATIAL_PROXIMITY_METERS",
    "DEFAULT_REVERIFICATION_DAYS",
    "DEFAULT_MIN_CHANGEOVER_MINUTES",
    "EUDR_RETENTION_YEARS",
    "MAX_PREVIOUS_CARGOES",
    "MAX_CONTAMINATION_DEPTH",
    "ASSESSMENT_LAYOUT_WEIGHT",
    "ASSESSMENT_PROTOCOL_WEIGHT",
    "ASSESSMENT_HISTORY_WEIGHT",
    "ASSESSMENT_LABELING_WEIGHT",
    "ASSESSMENT_DOCUMENTATION_WEIGHT",
    "MAX_BATCH_SIZE",
    "PRIMARY_COMMODITIES",
    "FACILITY_ASSESSMENT_LEVELS",
    # Enumerations
    "SCPType",
    "SCPStatus",
    "SegregationMethod",
    "StorageType",
    "TransportType",
    "ProcessingLineType",
    "ContaminationPathway",
    "ContaminationSeverity",
    "LabelType",
    "LabelStatus",
    "FacilityCapabilityLevel",
    "ReportFormat",
    "RiskClassification",
    "ComplianceStatus",
    "CleaningMethod",
    # Core Models
    "SegregationControlPoint",
    "StorageZone",
    "StorageEvent",
    "TransportVehicle",
    "TransportVerification",
    "ProcessingLine",
    "ChangeoverRecord",
    "ContaminationEvent",
    "LabelRecord",
    "FacilityAssessment",
    "ContaminationImpact",
    "SegregationReport",
    # Request Models
    "RegisterSCPRequest",
    "ValidateSCPRequest",
    "RegisterStorageZoneRequest",
    "RecordStorageEventRequest",
    "RegisterVehicleRequest",
    "VerifyTransportRequest",
    "RegisterProcessingLineRequest",
    "RecordChangeoverRequest",
    "DetectContaminationRequest",
    "RecordContaminationRequest",
    "VerifyLabelsRequest",
    "RunAssessmentRequest",
    "GenerateReportRequest",
    "SearchSCPRequest",
    "BatchImportSCPRequest",
    # Response Models
    "SCPResponse",
    "StorageAuditResponse",
    "TransportVerificationResponse",
    "ProcessingVerificationResponse",
    "ContaminationDetectionResponse",
    "ContaminationImpactResponse",
    "LabelAuditResponse",
    "AssessmentResponse",
    "ReportResponse",
    "BatchJobResponse",
    "HealthResponse",
]
