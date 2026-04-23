"""
Evidence Packager - GL-007_FurnacePulse Safety Module

This module generates immutable evidence packages for safety incidents,
compliance audits, and regulatory investigations. Packages include
comprehensive data with SHA-256 integrity hashing and retention policy compliance.

Evidence Package Contents:
    - Event summary and affected assets
    - Trend snapshots (fuel, temperatures, draft, flame signals)
    - IR snapshots with hotspot annotations
    - Model outputs with SHAP/LIME explanations and confidence scores
    - User acknowledgements and actions taken
    - CMMS work order links
    - HAZOP/LOPA references

Example:
    >>> packager = EvidencePackager(config)
    >>> package = packager.create_package(
    ...     event_id="EVT-001",
    ...     furnace_id="FRN-001",
    ...     package_type=PackageType.INCIDENT_INVESTIGATION
    ... )
    >>> print(f"Package hash: {package.integrity_hash}")
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from pathlib import Path
import hashlib
import logging
import json
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class PackageType(str, Enum):
    """Types of evidence packages."""
    INCIDENT_INVESTIGATION = "incident_investigation"
    COMPLIANCE_AUDIT = "compliance_audit"
    REGULATORY_SUBMISSION = "regulatory_submission"
    INSURANCE_CLAIM = "insurance_claim"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    SAFETY_REVIEW = "safety_review"
    PERFORMANCE_BASELINE = "performance_baseline"


class RetentionClass(str, Enum):
    """Retention classes for evidence packages."""
    STANDARD = "standard"  # 7 years
    EXTENDED = "extended"  # 15 years
    PERMANENT = "permanent"  # Indefinite
    REGULATORY = "regulatory"  # Per regulation requirements


class DataClassification(str, Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class SnapshotType(str, Enum):
    """Types of trend snapshots."""
    FUEL_FLOW = "fuel_flow"
    TEMPERATURE = "temperature"
    DRAFT_PRESSURE = "draft_pressure"
    FLAME_SIGNAL = "flame_signal"
    O2_LEVEL = "o2_level"
    CO_LEVEL = "co_level"
    EFFICIENCY = "efficiency"
    CUSTOM = "custom"


# =============================================================================
# Retention Policy
# =============================================================================

RETENTION_POLICY: Dict[RetentionClass, Dict[str, Any]] = {
    RetentionClass.STANDARD: {
        "duration_years": 7,
        "description": "Standard business records retention",
        "legal_basis": "Corporate policy",
    },
    RetentionClass.EXTENDED: {
        "duration_years": 15,
        "description": "Extended retention for significant events",
        "legal_basis": "Safety regulations",
    },
    RetentionClass.PERMANENT: {
        "duration_years": 100,  # Effectively permanent
        "description": "Permanent retention for critical records",
        "legal_basis": "Regulatory requirement",
    },
    RetentionClass.REGULATORY: {
        "duration_years": 25,
        "description": "Regulatory mandated retention period",
        "legal_basis": "OSHA/EPA/State regulations",
    },
}


# =============================================================================
# Pydantic Models
# =============================================================================

class EvidencePackagerConfig(BaseModel):
    """Configuration for Evidence Packager."""

    site_id: str = Field(..., description="Site identifier")
    organization_name: str = Field(..., description="Organization name")
    default_retention_class: RetentionClass = Field(
        default=RetentionClass.STANDARD,
        description="Default retention class"
    )
    default_classification: DataClassification = Field(
        default=DataClassification.CONFIDENTIAL,
        description="Default data classification"
    )
    storage_path: Optional[str] = Field(
        None, description="Base path for package storage"
    )
    enable_compression: bool = Field(
        default=True, description="Enable package compression"
    )
    enable_encryption: bool = Field(
        default=False, description="Enable package encryption"
    )
    audit_retention_years: int = Field(
        default=10, ge=1, le=100,
        description="Audit record retention period"
    )


class EventSummary(BaseModel):
    """Summary of the event being documented."""

    event_id: str = Field(..., description="Unique event ID")
    event_type: str = Field(..., description="Type of event")
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Detailed description")
    occurred_at: datetime = Field(..., description="Event occurrence time")
    discovered_at: datetime = Field(..., description="Event discovery time")
    severity: str = Field(..., description="Event severity level")
    status: str = Field(..., description="Current event status")
    root_cause: Optional[str] = Field(None, description="Root cause if determined")
    contributing_factors: List[str] = Field(
        default_factory=list, description="Contributing factors"
    )
    immediate_actions: List[str] = Field(
        default_factory=list, description="Immediate actions taken"
    )
    long_term_actions: List[str] = Field(
        default_factory=list, description="Long-term corrective actions"
    )


class AffectedAsset(BaseModel):
    """Information about an affected asset."""

    asset_id: str = Field(..., description="Asset identifier")
    asset_name: str = Field(..., description="Asset name")
    asset_type: str = Field(..., description="Asset type")
    location: str = Field(..., description="Physical location")
    damage_assessment: Optional[str] = Field(
        None, description="Damage assessment if applicable"
    )
    repair_status: Optional[str] = Field(None, description="Repair status")
    downtime_hours: Optional[float] = Field(None, description="Downtime in hours")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost impact")


class TrendSnapshot(BaseModel):
    """Trend data snapshot for a specific parameter."""

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot ID"
    )
    snapshot_type: SnapshotType = Field(..., description="Type of snapshot")
    parameter_name: str = Field(..., description="Parameter name")
    parameter_tag: str = Field(..., description="Historian tag")
    unit: str = Field(..., description="Engineering unit")
    start_time: datetime = Field(..., description="Snapshot start time")
    end_time: datetime = Field(..., description="Snapshot end time")
    sample_interval_seconds: int = Field(..., description="Sample interval")
    data_points: List[Dict[str, Any]] = Field(
        ..., description="List of {timestamp, value} points"
    )
    statistics: Dict[str, float] = Field(
        ..., description="Statistics (min, max, avg, std)"
    )
    annotations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Annotations on data"
    )
    data_hash: str = Field(..., description="SHA-256 hash of data points")


class IRSnapshot(BaseModel):
    """Infrared camera snapshot with annotations."""

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot ID"
    )
    timestamp: datetime = Field(..., description="Capture timestamp")
    camera_id: str = Field(..., description="Camera identifier")
    camera_location: str = Field(..., description="Camera location/zone")
    image_reference: str = Field(..., description="Image file reference or base64")
    image_format: str = Field(
        default="png", description="Image format (png, jpeg, tiff)"
    )
    resolution: Dict[str, int] = Field(..., description="Image resolution {width, height}")
    temperature_range: Dict[str, float] = Field(
        ..., description="Temperature range {min, max} in image"
    )
    color_palette: str = Field(default="iron", description="IR color palette used")
    hotspot_annotations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Hotspot annotations with coordinates and temps"
    )
    analysis_notes: Optional[str] = Field(None, description="Analysis notes")
    image_hash: str = Field(..., description="SHA-256 hash of image data")


class ModelOutput(BaseModel):
    """Machine learning model output with explainability."""

    output_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique output ID"
    )
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    inference_timestamp: datetime = Field(..., description="Inference timestamp")
    input_features: Dict[str, Any] = Field(..., description="Input features")
    prediction: Any = Field(..., description="Model prediction")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence"
    )
    prediction_type: str = Field(
        ..., description="Type: classification, regression, anomaly"
    )

    # Explainability
    shap_values: Optional[Dict[str, float]] = Field(
        None, description="SHAP values for features"
    )
    lime_explanation: Optional[Dict[str, Any]] = Field(
        None, description="LIME explanation"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores"
    )
    explanation_narrative: Optional[str] = Field(
        None, description="Human-readable explanation"
    )

    # Validation
    model_hash: str = Field(..., description="Hash of model artifact")
    input_hash: str = Field(..., description="Hash of input features")


class UserAction(BaseModel):
    """User acknowledgement or action record."""

    action_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique action ID"
    )
    action_type: str = Field(
        ..., description="Type: acknowledgement, override, intervention, etc."
    )
    timestamp: datetime = Field(..., description="Action timestamp")
    user_id: str = Field(..., description="User identifier")
    user_name: str = Field(..., description="User display name")
    user_role: str = Field(..., description="User role at time of action")
    action_description: str = Field(..., description="Description of action")
    justification: Optional[str] = Field(None, description="Justification if required")
    approval_chain: List[str] = Field(
        default_factory=list, description="Approval chain if applicable"
    )
    related_alert_id: Optional[str] = Field(None, description="Related alert ID")
    digital_signature: Optional[str] = Field(
        None, description="Digital signature if available"
    )


class CMMSWorkOrder(BaseModel):
    """CMMS work order reference."""

    work_order_id: str = Field(..., description="Work order ID")
    work_order_type: str = Field(
        ..., description="Type: corrective, preventive, emergency"
    )
    title: str = Field(..., description="Work order title")
    description: str = Field(..., description="Work order description")
    status: str = Field(..., description="Current status")
    priority: str = Field(..., description="Priority level")
    created_at: datetime = Field(..., description="Creation timestamp")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled date")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    assigned_to: str = Field(..., description="Assigned technician/team")
    estimated_hours: Optional[float] = Field(None, description="Estimated hours")
    actual_hours: Optional[float] = Field(None, description="Actual hours")
    parts_used: List[Dict[str, Any]] = Field(
        default_factory=list, description="Parts used"
    )
    labor_cost: Optional[float] = Field(None, description="Labor cost")
    parts_cost: Optional[float] = Field(None, description="Parts cost")
    findings: Optional[str] = Field(None, description="Findings during work")


class HAZOPLOPAReference(BaseModel):
    """Reference to HAZOP and LOPA records."""

    reference_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique reference ID"
    )
    reference_type: str = Field(
        ..., pattern="^(hazop|lopa|ipl|moc)$",
        description="Type of reference"
    )
    external_id: str = Field(..., description="External system ID")
    title: str = Field(..., description="Reference title")
    description: str = Field(..., description="Reference description")
    status: str = Field(..., description="Current status")
    relevance: str = Field(..., description="Relevance to this event")
    linked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When linked to this package"
    )
    linked_by: str = Field(..., description="User who created link")


class PackageMetadata(BaseModel):
    """Metadata for evidence package."""

    package_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique package ID"
    )
    package_version: str = Field(default="1.0", description="Package format version")
    package_type: PackageType = Field(..., description="Type of package")
    title: str = Field(..., description="Package title")
    description: str = Field(..., description="Package description")

    # Classification and Retention
    classification: DataClassification = Field(..., description="Data classification")
    retention_class: RetentionClass = Field(..., description="Retention class")
    retention_until: datetime = Field(..., description="Retention expiry date")

    # Scope
    furnace_id: str = Field(..., description="Primary furnace ID")
    additional_furnaces: List[str] = Field(
        default_factory=list, description="Additional furnaces if applicable"
    )
    time_period_start: datetime = Field(..., description="Evidence period start")
    time_period_end: datetime = Field(..., description="Evidence period end")

    # Creation
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Package creation timestamp"
    )
    created_by: str = Field(..., description="Package creator")
    organization: str = Field(..., description="Organization name")
    site_id: str = Field(..., description="Site identifier")

    # Integrity
    integrity_hash: str = Field(..., description="SHA-256 hash of entire package")
    component_hashes: Dict[str, str] = Field(
        default_factory=dict, description="Hashes of individual components"
    )


class EvidencePackage(BaseModel):
    """Complete evidence package."""

    metadata: PackageMetadata = Field(..., description="Package metadata")
    event_summary: EventSummary = Field(..., description="Event summary")
    affected_assets: List[AffectedAsset] = Field(
        default_factory=list, description="Affected assets"
    )
    trend_snapshots: List[TrendSnapshot] = Field(
        default_factory=list, description="Trend data snapshots"
    )
    ir_snapshots: List[IRSnapshot] = Field(
        default_factory=list, description="IR camera snapshots"
    )
    model_outputs: List[ModelOutput] = Field(
        default_factory=list, description="ML model outputs"
    )
    user_actions: List[UserAction] = Field(
        default_factory=list, description="User actions and acknowledgements"
    )
    cmms_work_orders: List[CMMSWorkOrder] = Field(
        default_factory=list, description="CMMS work orders"
    )
    hazop_lopa_references: List[HAZOPLOPAReference] = Field(
        default_factory=list, description="HAZOP/LOPA references"
    )
    attachments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Additional attachments"
    )
    chain_of_custody: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chain of custody records"
    )


class PackageAuditEntry(BaseModel):
    """Audit log entry for package activities."""

    audit_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique audit ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    package_id: str = Field(..., description="Associated package ID")
    action: str = Field(..., description="Action performed")
    actor_id: str = Field(..., description="Actor user or system ID")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    data_hash: str = Field(..., description="SHA-256 hash of change data")


# =============================================================================
# Evidence Packager
# =============================================================================

class EvidencePackager:
    """
    Evidence Packager for creating immutable audit packages.

    This packager creates comprehensive evidence packages for safety incidents,
    compliance audits, and regulatory investigations. Packages include data
    integrity verification via SHA-256 hashing and retention policy compliance.

    Attributes:
        config: Packager configuration
        packages: Created packages
        audit_log: Audit trail for package activities

    Example:
        >>> config = EvidencePackagerConfig(
        ...     site_id="SITE-001",
        ...     organization_name="Industrial Corp"
        ... )
        >>> packager = EvidencePackager(config)
        >>> package = packager.create_package(
        ...     event_id="EVT-001",
        ...     furnace_id="FRN-001",
        ...     package_type=PackageType.INCIDENT_INVESTIGATION
        ... )
    """

    def __init__(self, config: EvidencePackagerConfig):
        """
        Initialize EvidencePackager.

        Args:
            config: Packager configuration
        """
        self.config = config
        self.packages: Dict[str, EvidencePackage] = {}
        self.audit_log: List[PackageAuditEntry] = []

        logger.info(
            f"EvidencePackager initialized for {config.organization_name} "
            f"site {config.site_id}"
        )

    def create_package(
        self,
        event_id: str,
        furnace_id: str,
        package_type: PackageType,
        title: str,
        description: str,
        event_summary: EventSummary,
        time_period_start: datetime,
        time_period_end: datetime,
        created_by: str,
        retention_class: Optional[RetentionClass] = None,
        classification: Optional[DataClassification] = None
    ) -> EvidencePackage:
        """
        Create a new evidence package.

        Args:
            event_id: Event identifier
            furnace_id: Primary furnace ID
            package_type: Type of package
            title: Package title
            description: Package description
            event_summary: Event summary data
            time_period_start: Evidence period start
            time_period_end: Evidence period end
            created_by: Creator user ID
            retention_class: Optional retention class override
            classification: Optional classification override

        Returns:
            Created EvidencePackage (initially empty, populate with add_* methods)
        """
        start_time = datetime.now(timezone.utc)

        # Use defaults if not specified
        retention = retention_class or self.config.default_retention_class
        classif = classification or self.config.default_classification

        # Calculate retention date
        retention_policy = RETENTION_POLICY[retention]
        retention_until = datetime.now(timezone.utc).replace(
            year=datetime.now(timezone.utc).year + retention_policy["duration_years"]
        )

        # Create metadata (hash will be computed later)
        metadata = PackageMetadata(
            package_type=package_type,
            title=title,
            description=description,
            classification=classif,
            retention_class=retention,
            retention_until=retention_until,
            furnace_id=furnace_id,
            time_period_start=time_period_start,
            time_period_end=time_period_end,
            created_by=created_by,
            organization=self.config.organization_name,
            site_id=self.config.site_id,
            integrity_hash="",  # Will be computed
            component_hashes={},
        )

        # Create package
        package = EvidencePackage(
            metadata=metadata,
            event_summary=event_summary,
        )

        # Compute initial hash
        self._update_integrity_hash(package)

        self.packages[metadata.package_id] = package

        # Add to chain of custody
        package.chain_of_custody.append({
            "action": "PACKAGE_CREATED",
            "timestamp": start_time.isoformat(),
            "actor": created_by,
            "hash_before": None,
            "hash_after": package.metadata.integrity_hash,
        })

        # Log audit
        self._log_audit(
            package_id=metadata.package_id,
            action="PACKAGE_CREATED",
            actor_id=created_by,
            details={
                "event_id": event_id,
                "package_type": package_type.value,
                "retention_class": retention.value,
                "retention_until": retention_until.isoformat(),
            }
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Evidence package created: {metadata.package_id} "
            f"(Type: {package_type.value}) in {processing_time:.2f}ms"
        )

        return package

    def add_affected_asset(
        self,
        package_id: str,
        asset: AffectedAsset,
        added_by: str
    ) -> EvidencePackage:
        """
        Add an affected asset to the package.

        Args:
            package_id: Package identifier
            asset: Affected asset data
            added_by: User adding the asset

        Returns:
            Updated EvidencePackage

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        package.affected_assets.append(asset)
        self._update_integrity_hash(package)

        self._add_custody_record(package, "ASSET_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="ASSET_ADDED",
            actor_id=added_by,
            details={"asset_id": asset.asset_id, "asset_name": asset.asset_name}
        )

        logger.info(f"Asset added to package {package_id}: {asset.asset_id}")
        return package

    def add_trend_snapshot(
        self,
        package_id: str,
        snapshot_type: SnapshotType,
        parameter_name: str,
        parameter_tag: str,
        unit: str,
        start_time: datetime,
        end_time: datetime,
        data_points: List[Dict[str, Any]],
        sample_interval_seconds: int,
        added_by: str,
        annotations: Optional[List[Dict[str, Any]]] = None
    ) -> TrendSnapshot:
        """
        Add a trend data snapshot to the package.

        Args:
            package_id: Package identifier
            snapshot_type: Type of trend snapshot
            parameter_name: Parameter name
            parameter_tag: Historian tag
            unit: Engineering unit
            start_time: Snapshot start time
            end_time: Snapshot end time
            data_points: List of {timestamp, value} data points
            sample_interval_seconds: Sample interval
            added_by: User adding snapshot
            annotations: Optional annotations

        Returns:
            Created TrendSnapshot

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        # Calculate statistics
        values = [p["value"] for p in data_points if p.get("value") is not None]
        statistics = self._calculate_statistics(values)

        # Hash data points
        data_str = json.dumps(data_points, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        snapshot = TrendSnapshot(
            snapshot_type=snapshot_type,
            parameter_name=parameter_name,
            parameter_tag=parameter_tag,
            unit=unit,
            start_time=start_time,
            end_time=end_time,
            sample_interval_seconds=sample_interval_seconds,
            data_points=data_points,
            statistics=statistics,
            annotations=annotations or [],
            data_hash=data_hash,
        )

        package.trend_snapshots.append(snapshot)
        package.metadata.component_hashes[f"trend_{snapshot.snapshot_id}"] = data_hash
        self._update_integrity_hash(package)

        self._add_custody_record(package, "TREND_SNAPSHOT_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="TREND_SNAPSHOT_ADDED",
            actor_id=added_by,
            details={
                "snapshot_id": snapshot.snapshot_id,
                "parameter_name": parameter_name,
                "data_points_count": len(data_points),
            }
        )

        logger.info(
            f"Trend snapshot added to package {package_id}: "
            f"{parameter_name} ({len(data_points)} points)"
        )
        return snapshot

    def add_ir_snapshot(
        self,
        package_id: str,
        timestamp: datetime,
        camera_id: str,
        camera_location: str,
        image_data: Union[str, bytes],
        image_format: str,
        resolution: Dict[str, int],
        temperature_range: Dict[str, float],
        added_by: str,
        hotspot_annotations: Optional[List[Dict[str, Any]]] = None,
        color_palette: str = "iron",
        analysis_notes: Optional[str] = None
    ) -> IRSnapshot:
        """
        Add an IR camera snapshot to the package.

        Args:
            package_id: Package identifier
            timestamp: Capture timestamp
            camera_id: Camera identifier
            camera_location: Camera location/zone
            image_data: Image as base64 string or bytes
            image_format: Image format
            resolution: {width, height}
            temperature_range: {min, max} in image
            added_by: User adding snapshot
            hotspot_annotations: Hotspot annotations
            color_palette: IR color palette
            analysis_notes: Analysis notes

        Returns:
            Created IRSnapshot

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        # Handle image data
        if isinstance(image_data, bytes):
            image_reference = base64.b64encode(image_data).decode()
            image_bytes = image_data
        else:
            image_reference = image_data
            image_bytes = base64.b64decode(image_data)

        # Hash image
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        snapshot = IRSnapshot(
            timestamp=timestamp,
            camera_id=camera_id,
            camera_location=camera_location,
            image_reference=image_reference,
            image_format=image_format,
            resolution=resolution,
            temperature_range=temperature_range,
            color_palette=color_palette,
            hotspot_annotations=hotspot_annotations or [],
            analysis_notes=analysis_notes,
            image_hash=image_hash,
        )

        package.ir_snapshots.append(snapshot)
        package.metadata.component_hashes[f"ir_{snapshot.snapshot_id}"] = image_hash
        self._update_integrity_hash(package)

        self._add_custody_record(package, "IR_SNAPSHOT_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="IR_SNAPSHOT_ADDED",
            actor_id=added_by,
            details={
                "snapshot_id": snapshot.snapshot_id,
                "camera_id": camera_id,
                "hotspot_count": len(hotspot_annotations or []),
            }
        )

        logger.info(
            f"IR snapshot added to package {package_id}: "
            f"Camera {camera_id} at {camera_location}"
        )
        return snapshot

    def add_model_output(
        self,
        package_id: str,
        model_name: str,
        model_version: str,
        model_hash: str,
        input_features: Dict[str, Any],
        prediction: Any,
        confidence: float,
        prediction_type: str,
        added_by: str,
        shap_values: Optional[Dict[str, float]] = None,
        lime_explanation: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        explanation_narrative: Optional[str] = None
    ) -> ModelOutput:
        """
        Add ML model output with explainability to the package.

        Args:
            package_id: Package identifier
            model_name: Model name
            model_version: Model version
            model_hash: Hash of model artifact
            input_features: Input features
            prediction: Model prediction
            confidence: Prediction confidence
            prediction_type: Type of prediction
            added_by: User adding output
            shap_values: SHAP values for features
            lime_explanation: LIME explanation
            feature_importance: Feature importance scores
            explanation_narrative: Human-readable explanation

        Returns:
            Created ModelOutput

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        # Hash input features
        input_str = json.dumps(input_features, sort_keys=True, default=str)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()

        output = ModelOutput(
            model_name=model_name,
            model_version=model_version,
            inference_timestamp=datetime.now(timezone.utc),
            input_features=input_features,
            prediction=prediction,
            confidence=confidence,
            prediction_type=prediction_type,
            shap_values=shap_values,
            lime_explanation=lime_explanation,
            feature_importance=feature_importance,
            explanation_narrative=explanation_narrative,
            model_hash=model_hash,
            input_hash=input_hash,
        )

        package.model_outputs.append(output)
        package.metadata.component_hashes[f"model_{output.output_id}"] = input_hash
        self._update_integrity_hash(package)

        self._add_custody_record(package, "MODEL_OUTPUT_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="MODEL_OUTPUT_ADDED",
            actor_id=added_by,
            details={
                "output_id": output.output_id,
                "model_name": model_name,
                "model_version": model_version,
                "confidence": confidence,
            }
        )

        logger.info(
            f"Model output added to package {package_id}: "
            f"{model_name} v{model_version} (confidence: {confidence:.2%})"
        )
        return output

    def add_user_action(
        self,
        package_id: str,
        action_type: str,
        user_id: str,
        user_name: str,
        user_role: str,
        action_description: str,
        action_timestamp: Optional[datetime] = None,
        justification: Optional[str] = None,
        approval_chain: Optional[List[str]] = None,
        related_alert_id: Optional[str] = None,
        added_by: str = "SYSTEM"
    ) -> UserAction:
        """
        Add user action or acknowledgement to the package.

        Args:
            package_id: Package identifier
            action_type: Type of action
            user_id: User identifier
            user_name: User display name
            user_role: User role
            action_description: Description of action
            action_timestamp: When action occurred (default: now)
            justification: Justification if required
            approval_chain: Approval chain if applicable
            related_alert_id: Related alert ID
            added_by: User adding this record

        Returns:
            Created UserAction

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        action = UserAction(
            action_type=action_type,
            timestamp=action_timestamp or datetime.now(timezone.utc),
            user_id=user_id,
            user_name=user_name,
            user_role=user_role,
            action_description=action_description,
            justification=justification,
            approval_chain=approval_chain or [],
            related_alert_id=related_alert_id,
        )

        package.user_actions.append(action)
        self._update_integrity_hash(package)

        self._add_custody_record(package, "USER_ACTION_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="USER_ACTION_ADDED",
            actor_id=added_by,
            details={
                "action_id": action.action_id,
                "action_type": action_type,
                "user_id": user_id,
            }
        )

        logger.info(
            f"User action added to package {package_id}: "
            f"{action_type} by {user_name}"
        )
        return action

    def add_cmms_work_order(
        self,
        package_id: str,
        work_order: CMMSWorkOrder,
        added_by: str
    ) -> EvidencePackage:
        """
        Add CMMS work order reference to the package.

        Args:
            package_id: Package identifier
            work_order: CMMS work order data
            added_by: User adding reference

        Returns:
            Updated EvidencePackage

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        package.cmms_work_orders.append(work_order)
        self._update_integrity_hash(package)

        self._add_custody_record(package, "WORK_ORDER_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="WORK_ORDER_ADDED",
            actor_id=added_by,
            details={
                "work_order_id": work_order.work_order_id,
                "work_order_type": work_order.work_order_type,
            }
        )

        logger.info(
            f"Work order added to package {package_id}: {work_order.work_order_id}"
        )
        return package

    def add_hazop_lopa_reference(
        self,
        package_id: str,
        reference_type: str,
        external_id: str,
        title: str,
        description: str,
        status: str,
        relevance: str,
        linked_by: str
    ) -> HAZOPLOPAReference:
        """
        Add HAZOP/LOPA reference to the package.

        Args:
            package_id: Package identifier
            reference_type: Type (hazop, lopa, ipl, moc)
            external_id: External system ID
            title: Reference title
            description: Reference description
            status: Current status
            relevance: Relevance to event
            linked_by: User creating link

        Returns:
            Created HAZOPLOPAReference

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        reference = HAZOPLOPAReference(
            reference_type=reference_type,
            external_id=external_id,
            title=title,
            description=description,
            status=status,
            relevance=relevance,
            linked_by=linked_by,
        )

        package.hazop_lopa_references.append(reference)
        self._update_integrity_hash(package)

        self._add_custody_record(package, "HAZOP_LOPA_REFERENCE_ADDED", linked_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="HAZOP_LOPA_REFERENCE_ADDED",
            actor_id=linked_by,
            details={
                "reference_id": reference.reference_id,
                "reference_type": reference_type,
                "external_id": external_id,
            }
        )

        logger.info(
            f"HAZOP/LOPA reference added to package {package_id}: "
            f"{reference_type} - {external_id}"
        )
        return reference

    def add_attachment(
        self,
        package_id: str,
        attachment_name: str,
        attachment_type: str,
        content: Union[str, bytes],
        description: str,
        added_by: str
    ) -> Dict[str, Any]:
        """
        Add an attachment to the package.

        Args:
            package_id: Package identifier
            attachment_name: Attachment file name
            attachment_type: MIME type
            content: Content as base64 string or bytes
            description: Attachment description
            added_by: User adding attachment

        Returns:
            Attachment metadata dict

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        # Handle content
        if isinstance(content, bytes):
            content_b64 = base64.b64encode(content).decode()
            content_bytes = content
        else:
            content_b64 = content
            content_bytes = base64.b64decode(content)

        # Hash content
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        attachment_id = str(uuid4())

        attachment = {
            "attachment_id": attachment_id,
            "name": attachment_name,
            "type": attachment_type,
            "content_reference": content_b64,
            "size_bytes": len(content_bytes),
            "content_hash": content_hash,
            "description": description,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "added_by": added_by,
        }

        package.attachments.append(attachment)
        package.metadata.component_hashes[f"attachment_{attachment_id}"] = content_hash
        self._update_integrity_hash(package)

        self._add_custody_record(package, "ATTACHMENT_ADDED", added_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="ATTACHMENT_ADDED",
            actor_id=added_by,
            details={
                "attachment_id": attachment_id,
                "name": attachment_name,
                "size_bytes": len(content_bytes),
            }
        )

        logger.info(
            f"Attachment added to package {package_id}: "
            f"{attachment_name} ({len(content_bytes)} bytes)"
        )
        return attachment

    def finalize_package(
        self,
        package_id: str,
        finalized_by: str
    ) -> EvidencePackage:
        """
        Finalize package (mark as complete and compute final hash).

        Args:
            package_id: Package identifier
            finalized_by: User finalizing package

        Returns:
            Finalized EvidencePackage

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        old_hash = package.metadata.integrity_hash

        # Ensure final hash is current
        self._update_integrity_hash(package)

        self._add_custody_record(package, "PACKAGE_FINALIZED", finalized_by, old_hash)
        self._log_audit(
            package_id=package_id,
            action="PACKAGE_FINALIZED",
            actor_id=finalized_by,
            details={
                "final_hash": package.metadata.integrity_hash,
                "component_count": {
                    "assets": len(package.affected_assets),
                    "trend_snapshots": len(package.trend_snapshots),
                    "ir_snapshots": len(package.ir_snapshots),
                    "model_outputs": len(package.model_outputs),
                    "user_actions": len(package.user_actions),
                    "work_orders": len(package.cmms_work_orders),
                    "hazop_lopa_refs": len(package.hazop_lopa_references),
                    "attachments": len(package.attachments),
                },
            }
        )

        logger.info(
            f"Evidence package finalized: {package_id} "
            f"(hash: {package.metadata.integrity_hash[:16]}...)"
        )
        return package

    def verify_package_integrity(self, package_id: str) -> Dict[str, Any]:
        """
        Verify package integrity by recomputing hash.

        Args:
            package_id: Package identifier

        Returns:
            Verification result with details

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        stored_hash = package.metadata.integrity_hash

        # Temporarily clear hash and recompute
        package.metadata.integrity_hash = ""
        package_dict = package.dict(exclude={"metadata": {"integrity_hash"}})
        package_str = json.dumps(package_dict, sort_keys=True, default=str)
        computed_hash = hashlib.sha256(package_str.encode()).hexdigest()

        # Restore hash
        package.metadata.integrity_hash = stored_hash

        is_valid = stored_hash == computed_hash

        result = {
            "package_id": package_id,
            "stored_hash": stored_hash,
            "computed_hash": computed_hash,
            "integrity_valid": is_valid,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

        self._log_audit(
            package_id=package_id,
            action="INTEGRITY_VERIFIED",
            actor_id="SYSTEM",
            details=result
        )

        if not is_valid:
            logger.error(
                f"Package integrity FAILED: {package_id} "
                f"(stored: {stored_hash[:16]}... computed: {computed_hash[:16]}...)"
            )
        else:
            logger.info(f"Package integrity verified: {package_id}")

        return result

    def get_package(self, package_id: str) -> Optional[EvidencePackage]:
        """Get package by ID."""
        return self.packages.get(package_id)

    def get_retention_status(self, package_id: str) -> Dict[str, Any]:
        """
        Get retention status for a package.

        Args:
            package_id: Package identifier

        Returns:
            Retention status information

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)
        now = datetime.now(timezone.utc)

        days_until_expiry = (package.metadata.retention_until - now).days
        retention_policy = RETENTION_POLICY[package.metadata.retention_class]

        return {
            "package_id": package_id,
            "retention_class": package.metadata.retention_class.value,
            "retention_until": package.metadata.retention_until.isoformat(),
            "days_until_expiry": days_until_expiry,
            "is_expired": days_until_expiry < 0,
            "policy_description": retention_policy["description"],
            "legal_basis": retention_policy["legal_basis"],
        }

    def export_package(
        self,
        package_id: str,
        format: str = "json",
        include_binary: bool = True
    ) -> Dict[str, Any]:
        """
        Export package for external use.

        Args:
            package_id: Package identifier
            format: Export format (json only currently)
            include_binary: Include binary attachments

        Returns:
            Export result with data or path

        Raises:
            ValueError: If package not found
        """
        package = self._get_package(package_id)

        # Verify integrity before export
        integrity = self.verify_package_integrity(package_id)
        if not integrity["integrity_valid"]:
            raise ValueError("Package integrity verification failed - cannot export")

        package_dict = package.dict()

        if not include_binary:
            # Remove binary data, keep references
            for ir in package_dict.get("ir_snapshots", []):
                ir["image_reference"] = f"[BINARY_EXCLUDED:{ir.get('image_hash', '')}]"
            for att in package_dict.get("attachments", []):
                att["content_reference"] = f"[BINARY_EXCLUDED:{att.get('content_hash', '')}]"

        self._log_audit(
            package_id=package_id,
            action="PACKAGE_EXPORTED",
            actor_id="SYSTEM",
            details={
                "format": format,
                "include_binary": include_binary,
            }
        )

        return {
            "package_id": package_id,
            "format": format,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data": package_dict,
        }

    def _get_package(self, package_id: str) -> EvidencePackage:
        """Get package or raise ValueError."""
        if package_id not in self.packages:
            raise ValueError(f"Package {package_id} not found")
        return self.packages[package_id]

    def _update_integrity_hash(self, package: EvidencePackage) -> str:
        """Compute and update package integrity hash."""
        # Clear current hash before computing
        package.metadata.integrity_hash = ""

        # Compute hash of entire package
        package_dict = package.dict(exclude={"metadata": {"integrity_hash"}})
        package_str = json.dumps(package_dict, sort_keys=True, default=str)
        integrity_hash = hashlib.sha256(package_str.encode()).hexdigest()

        package.metadata.integrity_hash = integrity_hash
        return integrity_hash

    def _add_custody_record(
        self,
        package: EvidencePackage,
        action: str,
        actor: str,
        hash_before: str
    ) -> None:
        """Add record to chain of custody."""
        package.chain_of_custody.append({
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "hash_before": hash_before,
            "hash_after": package.metadata.integrity_hash,
        })

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "std": 0, "count": 0}

        import statistics

        return {
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "count": len(values),
        }

    def _log_audit(
        self,
        package_id: str,
        action: str,
        actor_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Add entry to audit log with integrity hash."""
        details_str = json.dumps(details, sort_keys=True, default=str)
        data_hash = hashlib.sha256(details_str.encode()).hexdigest()

        entry = PackageAuditEntry(
            package_id=package_id,
            action=action,
            actor_id=actor_id,
            details=details,
            data_hash=data_hash,
        )
        self.audit_log.append(entry)

    def get_audit_log(
        self,
        package_id: Optional[str] = None,
        action_filter: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PackageAuditEntry]:
        """
        Retrieve audit log with optional filters.

        Args:
            package_id: Optional package ID filter
            action_filter: Optional action type filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching PackageAuditEntry records
        """
        filtered = self.audit_log

        if package_id:
            filtered = [e for e in filtered if e.package_id == package_id]
        if action_filter:
            filtered = [e for e in filtered if e.action == action_filter]
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        return filtered
