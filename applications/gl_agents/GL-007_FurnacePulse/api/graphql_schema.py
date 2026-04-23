"""
GL-007 FurnacePulse - GraphQL Schema

Strawberry-based GraphQL schema for flexible queries on furnace
monitoring data, predictions, and compliance status.

Types:
- Furnace
- TMTReading
- HotspotAlert
- EfficiencyKPI
- RULPrediction
- ComplianceStatus
- EvidencePackage

Queries:
- furnace(id) - Single furnace details
- furnaces(filter) - List furnaces with filtering
- alerts(filter) - List alerts with filtering
- predictions(filter) - List predictions
- compliance(furnace_id) - Compliance status

Mutations:
- acknowledgeAlert - Acknowledge an alert
- generateEvidence - Generate evidence package
- createWorkOrder - Create maintenance work order

Subscriptions:
- telemetryUpdates - Real-time telemetry stream
- alertStream - Real-time alert notifications

Author: GreenLang API Team
Version: 1.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
import logging
import uuid
import hashlib

try:
    import strawberry
    from strawberry.types import Info
    from strawberry.scalars import JSON
    HAS_STRAWBERRY = True
except ImportError:
    HAS_STRAWBERRY = False

    # Create dummy decorators for import compatibility
    class strawberry:
        """Dummy strawberry module for when strawberry is not installed."""

        @staticmethod
        def type(cls=None, **kwargs):
            def decorator(c):
                return c
            return decorator(cls) if cls else decorator

        @staticmethod
        def input(cls=None, **kwargs):
            def decorator(c):
                return c
            return decorator(cls) if cls else decorator

        @staticmethod
        def enum(cls):
            return cls

        @staticmethod
        def field(*args, **kwargs):
            def decorator(func):
                return func
            if args and callable(args[0]):
                return args[0]
            return decorator

        @staticmethod
        def mutation(*args, **kwargs):
            def decorator(func):
                return func
            if args and callable(args[0]):
                return args[0]
            return decorator

        @staticmethod
        def subscription(*args, **kwargs):
            def decorator(func):
                return func
            if args and callable(args[0]):
                return args[0]
            return decorator

        class Schema:
            def __init__(self, *args, **kwargs):
                pass

    JSON = Dict[str, Any]

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

@strawberry.enum
class AlertSeverityEnum(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@strawberry.enum
class AlertStatusEnum(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@strawberry.enum
class ComponentTypeEnum(Enum):
    """Furnace component types."""
    RADIANT_TUBE = "radiant_tube"
    BURNER = "burner"
    REFRACTORY = "refractory"
    THERMOCOUPLE = "thermocouple"
    DAMPER = "damper"
    FAN = "fan"
    HEAT_EXCHANGER = "heat_exchanger"
    CONTROL_VALVE = "control_valve"


@strawberry.enum
class ComplianceStatusEnum(Enum):
    """NFPA 86 compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    WAIVER_GRANTED = "waiver_granted"


@strawberry.enum
class ExplanationMethodEnum(Enum):
    """XAI explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    COUNTERFACTUAL = "counterfactual"


# =============================================================================
# GraphQL Types
# =============================================================================

@strawberry.type
class Position3D:
    """3D position in furnace space."""
    x: float
    y: float
    z: float


@strawberry.type
class TMTReadingType:
    """Tube Metal Temperature reading from a sensor."""
    sensor_id: str
    position: Position3D
    temperature_c: float
    timestamp: datetime
    quality_score: float
    is_valid: bool


@strawberry.type
class TMTSummaryType:
    """TMT readings summary statistics."""
    furnace_id: str
    timestamp: datetime
    avg_temperature_c: float
    max_temperature_c: float
    min_temperature_c: float
    gradient_max_c_m: float
    sensor_count: int
    valid_sensor_count: int
    readings: List[TMTReadingType]


@strawberry.type
class HotspotAlertType:
    """Hotspot alert detected in furnace."""
    alert_id: str
    furnace_id: str
    cluster_id: str
    severity: AlertSeverityEnum
    center: Position3D
    radius_m: float
    peak_temperature_c: float
    avg_temperature_c: float
    temperature_delta_c: float
    affected_sensors: List[str]
    detected_at: datetime
    status: AlertStatusEnum
    recommended_action: str


@strawberry.type
class EfficiencyKPIType:
    """Efficiency KPI measurement."""
    kpi_id: str
    name: str
    value: float
    unit: str
    target: Optional[float]
    threshold_low: Optional[float]
    threshold_high: Optional[float]
    status: str
    trend: str
    timestamp: datetime


@strawberry.type
class EfficiencySummaryType:
    """Efficiency KPIs summary for a furnace."""
    furnace_id: str
    timestamp: datetime
    overall_efficiency_pct: float
    fuel_consumption_kg_h: float
    excess_air_pct: float
    stack_loss_pct: float
    co2_emissions_kg_h: float
    kpis: List[EfficiencyKPIType]


@strawberry.type
class RULPredictionType:
    """Remaining Useful Life prediction for a component."""
    prediction_id: str
    component_id: str
    component_type: ComponentTypeEnum
    component_name: str
    rul_days: float
    rul_hours: float
    confidence_lower_days: float
    confidence_upper_days: float
    confidence_level: float
    failure_probability_30d: float
    failure_mode: str
    health_score: float
    degradation_rate: float
    last_maintenance: Optional[datetime]
    recommended_action: str
    predicted_at: datetime


@strawberry.type
class RULSummaryType:
    """RUL predictions summary for a furnace."""
    furnace_id: str
    timestamp: datetime
    components_at_risk: int
    avg_health_score: float
    next_maintenance_due: Optional[datetime]
    maintenance_window_start: Optional[datetime]
    maintenance_window_end: Optional[datetime]
    predictions: List[RULPredictionType]


@strawberry.type
class NFPA86RequirementType:
    """NFPA 86 compliance requirement."""
    requirement_id: str
    section: str
    description: str
    status: ComplianceStatusEnum
    last_verified_at: Optional[datetime]
    evidence_ids: List[str]
    notes: Optional[str]


@strawberry.type
class ComplianceStatusType:
    """NFPA 86 compliance status for a furnace."""
    furnace_id: str
    timestamp: datetime
    overall_status: ComplianceStatusEnum
    compliance_score_pct: float
    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    pending_count: int
    next_audit_due: Optional[datetime]
    requirements: List[NFPA86RequirementType]


@strawberry.type
class AlertType:
    """General alert."""
    alert_id: str
    furnace_id: str
    alert_type: str
    severity: AlertSeverityEnum
    status: AlertStatusEnum
    title: str
    description: str
    source: str
    created_at: datetime
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]
    metadata: Optional[JSON]


@strawberry.type
class EvidencePackageType:
    """Evidence package for compliance/audit."""
    package_id: str
    furnace_id: str
    package_type: str
    status: str
    created_at: datetime
    download_url: Optional[str]
    sha256_hash: Optional[str]
    size_bytes: Optional[int]
    expires_at: Optional[datetime]


@strawberry.type
class WorkOrderType:
    """Maintenance work order."""
    work_order_id: str
    furnace_id: str
    component_id: str
    priority: str
    title: str
    description: str
    assigned_to: Optional[str]
    due_date: Optional[datetime]
    created_at: datetime
    status: str


@strawberry.type
class FeatureImportanceType:
    """Feature importance for explainability."""
    feature_name: str
    importance_value: float
    direction: str
    contribution_pct: float


@strawberry.type
class ExplanationType:
    """ML prediction explanation."""
    prediction_id: str
    prediction_type: str
    prediction_value: float
    explanation_method: ExplanationMethodEnum
    feature_importance: List[FeatureImportanceType]
    summary: str
    key_drivers: List[str]
    confidence_score: float
    counterfactuals: Optional[JSON]
    generated_at: datetime
    computation_hash: str


@strawberry.type
class FurnaceType:
    """Furnace with all monitoring data."""
    furnace_id: str
    name: str
    location: str
    furnace_type: str
    capacity_mw: float
    status: str
    commissioned_at: datetime
    last_maintenance: Optional[datetime]

    @strawberry.field
    def tmt_readings(self) -> TMTSummaryType:
        """Get current TMT readings."""
        return _get_mock_tmt_readings(self.furnace_id)

    @strawberry.field
    def hotspots(
        self,
        severity: Optional[AlertSeverityEnum] = None,
    ) -> List[HotspotAlertType]:
        """Get active hotspot alerts."""
        return _get_mock_hotspots(self.furnace_id, severity)

    @strawberry.field
    def efficiency(self) -> EfficiencySummaryType:
        """Get efficiency KPIs."""
        return _get_mock_efficiency(self.furnace_id)

    @strawberry.field
    def rul_predictions(
        self,
        component_type: Optional[ComponentTypeEnum] = None,
        at_risk_only: bool = False,
    ) -> RULSummaryType:
        """Get RUL predictions for components."""
        return _get_mock_rul_predictions(self.furnace_id, component_type, at_risk_only)

    @strawberry.field
    def compliance(self) -> ComplianceStatusType:
        """Get NFPA 86 compliance status."""
        return _get_mock_compliance(self.furnace_id)

    @strawberry.field
    def alerts(
        self,
        severity: Optional[AlertSeverityEnum] = None,
        status: Optional[AlertStatusEnum] = None,
        limit: int = 50,
    ) -> List[AlertType]:
        """Get alerts for this furnace."""
        return _get_mock_alerts(self.furnace_id, severity, status, limit)


@strawberry.type
class TelemetryUpdateType:
    """Real-time telemetry update."""
    furnace_id: str
    timestamp: datetime
    sensor_id: str
    measurement_type: str
    value: float
    unit: str
    quality_score: float


@strawberry.type
class AlertNotificationType:
    """Real-time alert notification."""
    alert: AlertType
    event_type: str  # created, updated, acknowledged, resolved


# =============================================================================
# Input Types
# =============================================================================

@strawberry.input
class FurnaceFilterInput:
    """Filter input for furnace queries."""
    furnace_ids: Optional[List[str]] = None
    location: Optional[str] = None
    status: Optional[str] = None
    min_efficiency_pct: Optional[float] = None


@strawberry.input
class AlertFilterInput:
    """Filter input for alert queries."""
    furnace_id: Optional[str] = None
    severity: Optional[AlertSeverityEnum] = None
    status: Optional[AlertStatusEnum] = None
    alert_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@strawberry.input
class PredictionFilterInput:
    """Filter input for prediction queries."""
    furnace_id: Optional[str] = None
    component_type: Optional[ComponentTypeEnum] = None
    prediction_type: Optional[str] = None
    min_confidence: Optional[float] = None


@strawberry.input
class EvidenceGenerateInput:
    """Input for evidence package generation."""
    furnace_id: str
    package_type: str = "compliance"
    start_date: datetime
    end_date: datetime
    include_telemetry: bool = True
    include_alerts: bool = True
    include_predictions: bool = True
    include_maintenance: bool = True
    requirement_ids: Optional[List[str]] = None
    format: str = "json"


@strawberry.input
class WorkOrderCreateInput:
    """Input for work order creation."""
    furnace_id: str
    component_id: str
    priority: str = "medium"
    title: str
    description: str
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None


@strawberry.input
class AlertAcknowledgeInput:
    """Input for alert acknowledgment."""
    alert_id: str
    notes: Optional[str] = None
    assign_to: Optional[str] = None


# =============================================================================
# Mock Data Functions
# =============================================================================

def _get_mock_tmt_readings(furnace_id: str) -> TMTSummaryType:
    """Generate mock TMT readings."""
    import random
    now = datetime.now(timezone.utc)

    readings = []
    temps = []
    for i in range(48):
        temp = 850.0 + random.uniform(-30, 50)
        is_valid = random.random() > 0.05
        temps.append(temp)

        readings.append(TMTReadingType(
            sensor_id=f"TMT-{i+1:03d}",
            position=Position3D(
                x=(i % 8) * 0.5,
                y=(i // 8) * 0.5,
                z=0.0 if i < 24 else 1.5
            ),
            temperature_c=temp,
            timestamp=now,
            quality_score=0.98 if is_valid else 0.45,
            is_valid=is_valid,
        ))

    return TMTSummaryType(
        furnace_id=furnace_id,
        timestamp=now,
        avg_temperature_c=sum(temps) / len(temps),
        max_temperature_c=max(temps),
        min_temperature_c=min(temps),
        gradient_max_c_m=25.5,
        sensor_count=48,
        valid_sensor_count=sum(1 for r in readings if r.is_valid),
        readings=readings,
    )


def _get_mock_hotspots(
    furnace_id: str,
    severity: Optional[AlertSeverityEnum] = None,
) -> List[HotspotAlertType]:
    """Generate mock hotspot alerts."""
    now = datetime.now(timezone.utc)

    hotspots = [
        HotspotAlertType(
            alert_id="hs-001",
            furnace_id=furnace_id,
            cluster_id="cluster-001",
            severity=AlertSeverityEnum.HIGH,
            center=Position3D(x=2.5, y=1.2, z=0.8),
            radius_m=0.35,
            peak_temperature_c=925.0,
            avg_temperature_c=895.0,
            temperature_delta_c=45.0,
            affected_sensors=["TMT-023", "TMT-024", "TMT-025"],
            detected_at=now - timedelta(minutes=15),
            status=AlertStatusEnum.ACTIVE,
            recommended_action="Inspect radiant tube section B2",
        ),
        HotspotAlertType(
            alert_id="hs-002",
            furnace_id=furnace_id,
            cluster_id="cluster-002",
            severity=AlertSeverityEnum.MEDIUM,
            center=Position3D(x=4.1, y=0.8, z=1.5),
            radius_m=0.25,
            peak_temperature_c=875.0,
            avg_temperature_c=855.0,
            temperature_delta_c=25.0,
            affected_sensors=["TMT-045", "TMT-046"],
            detected_at=now - timedelta(hours=2),
            status=AlertStatusEnum.ACKNOWLEDGED,
            recommended_action="Monitor zone C1",
        ),
    ]

    if severity:
        hotspots = [h for h in hotspots if h.severity == severity]

    return hotspots


def _get_mock_efficiency(furnace_id: str) -> EfficiencySummaryType:
    """Generate mock efficiency data."""
    now = datetime.now(timezone.utc)

    kpis = [
        EfficiencyKPIType(
            kpi_id="kpi-001",
            name="Thermal Efficiency",
            value=87.5,
            unit="%",
            target=90.0,
            threshold_low=80.0,
            threshold_high=95.0,
            status="normal",
            trend="stable",
            timestamp=now,
        ),
        EfficiencyKPIType(
            kpi_id="kpi-002",
            name="Fuel Consumption Rate",
            value=125.3,
            unit="kg/h",
            target=120.0,
            threshold_low=100.0,
            threshold_high=150.0,
            status="warning",
            trend="up",
            timestamp=now,
        ),
        EfficiencyKPIType(
            kpi_id="kpi-003",
            name="Excess Air Ratio",
            value=12.5,
            unit="%",
            target=10.0,
            threshold_low=5.0,
            threshold_high=20.0,
            status="normal",
            trend="down",
            timestamp=now,
        ),
    ]

    return EfficiencySummaryType(
        furnace_id=furnace_id,
        timestamp=now,
        overall_efficiency_pct=87.5,
        fuel_consumption_kg_h=125.3,
        excess_air_pct=12.5,
        stack_loss_pct=8.2,
        co2_emissions_kg_h=385.7,
        kpis=kpis,
    )


def _get_mock_rul_predictions(
    furnace_id: str,
    component_type: Optional[ComponentTypeEnum] = None,
    at_risk_only: bool = False,
) -> RULSummaryType:
    """Generate mock RUL predictions."""
    now = datetime.now(timezone.utc)

    predictions = [
        RULPredictionType(
            prediction_id="rul-001",
            component_id="RT-001",
            component_type=ComponentTypeEnum.RADIANT_TUBE,
            component_name="Radiant Tube Zone A1",
            rul_days=145.5,
            rul_hours=3492.0,
            confidence_lower_days=120.0,
            confidence_upper_days=175.0,
            confidence_level=0.95,
            failure_probability_30d=0.02,
            failure_mode="Wall thinning due to oxidation",
            health_score=82.5,
            degradation_rate=0.12,
            last_maintenance=now - timedelta(days=180),
            recommended_action="Schedule inspection during next planned outage",
            predicted_at=now,
        ),
        RULPredictionType(
            prediction_id="rul-002",
            component_id="BRN-003",
            component_type=ComponentTypeEnum.BURNER,
            component_name="Burner Unit 3",
            rul_days=25.0,
            rul_hours=600.0,
            confidence_lower_days=18.0,
            confidence_upper_days=35.0,
            confidence_level=0.95,
            failure_probability_30d=0.65,
            failure_mode="Igniter electrode erosion",
            health_score=45.0,
            degradation_rate=0.85,
            last_maintenance=now - timedelta(days=365),
            recommended_action="URGENT: Schedule burner replacement within 2 weeks",
            predicted_at=now,
        ),
        RULPredictionType(
            prediction_id="rul-003",
            component_id="TC-012",
            component_type=ComponentTypeEnum.THERMOCOUPLE,
            component_name="Thermocouple Zone B2",
            rul_days=210.0,
            rul_hours=5040.0,
            confidence_lower_days=180.0,
            confidence_upper_days=250.0,
            confidence_level=0.95,
            failure_probability_30d=0.01,
            failure_mode="Drift due to contamination",
            health_score=92.0,
            degradation_rate=0.05,
            last_maintenance=now - timedelta(days=90),
            recommended_action="No immediate action required",
            predicted_at=now,
        ),
    ]

    if component_type:
        predictions = [p for p in predictions if p.component_type == component_type]

    if at_risk_only:
        predictions = [p for p in predictions if p.rul_days < 30]

    at_risk_count = sum(1 for p in predictions if p.rul_days < 30)
    avg_health = sum(p.health_score for p in predictions) / len(predictions) if predictions else 0

    return RULSummaryType(
        furnace_id=furnace_id,
        timestamp=now,
        components_at_risk=at_risk_count,
        avg_health_score=avg_health,
        next_maintenance_due=now + timedelta(days=18) if at_risk_count > 0 else None,
        maintenance_window_start=now + timedelta(days=14) if at_risk_count > 0 else None,
        maintenance_window_end=now + timedelta(days=21) if at_risk_count > 0 else None,
        predictions=predictions,
    )


def _get_mock_compliance(furnace_id: str) -> ComplianceStatusType:
    """Generate mock compliance data."""
    now = datetime.now(timezone.utc)

    requirements = [
        NFPA86RequirementType(
            requirement_id="8.4.1",
            section="8.4 Safety Controls",
            description="Temperature limiting devices shall be provided",
            status=ComplianceStatusEnum.COMPLIANT,
            last_verified_at=now - timedelta(days=15),
            evidence_ids=["EVD-001", "EVD-002"],
            notes="Verified during Q4 2024 audit",
        ),
        NFPA86RequirementType(
            requirement_id="8.5.2",
            section="8.5 Combustion Safeguards",
            description="Flame detection system required for Class A furnaces",
            status=ComplianceStatusEnum.COMPLIANT,
            last_verified_at=now - timedelta(days=15),
            evidence_ids=["EVD-003"],
            notes=None,
        ),
        NFPA86RequirementType(
            requirement_id="8.6.1",
            section="8.6 Purge Requirements",
            description="Minimum 4 volume changes purge before ignition",
            status=ComplianceStatusEnum.COMPLIANT,
            last_verified_at=now - timedelta(days=30),
            evidence_ids=["EVD-004", "EVD-005"],
            notes=None,
        ),
        NFPA86RequirementType(
            requirement_id="9.2.1",
            section="9.2 Ventilation",
            description="Adequate ventilation for combustion air supply",
            status=ComplianceStatusEnum.PENDING_REVIEW,
            last_verified_at=now - timedelta(days=90),
            evidence_ids=[],
            notes="Scheduled for review in next audit cycle",
        ),
    ]

    compliant = sum(1 for r in requirements if r.status == ComplianceStatusEnum.COMPLIANT)
    non_compliant = sum(1 for r in requirements if r.status == ComplianceStatusEnum.NON_COMPLIANT)
    pending = sum(1 for r in requirements if r.status == ComplianceStatusEnum.PENDING_REVIEW)

    return ComplianceStatusType(
        furnace_id=furnace_id,
        timestamp=now,
        overall_status=ComplianceStatusEnum.COMPLIANT if non_compliant == 0 else ComplianceStatusEnum.NON_COMPLIANT,
        compliance_score_pct=100.0 * compliant / len(requirements),
        total_requirements=len(requirements),
        compliant_count=compliant,
        non_compliant_count=non_compliant,
        pending_count=pending,
        next_audit_due=now + timedelta(days=90),
        requirements=requirements,
    )


def _get_mock_alerts(
    furnace_id: Optional[str],
    severity: Optional[AlertSeverityEnum],
    status: Optional[AlertStatusEnum],
    limit: int,
) -> List[AlertType]:
    """Generate mock alerts."""
    now = datetime.now(timezone.utc)

    alerts = [
        AlertType(
            alert_id="ALT-001",
            furnace_id="FRN-001",
            alert_type="HOTSPOT_DETECTED",
            severity=AlertSeverityEnum.HIGH,
            status=AlertStatusEnum.ACTIVE,
            title="High temperature hotspot detected in Zone B2",
            description="TMT readings show sustained temperature 45C above normal",
            source="HotspotDetector",
            created_at=now - timedelta(minutes=15),
            acknowledged_at=None,
            acknowledged_by=None,
            resolved_at=None,
            resolved_by=None,
            metadata={"cluster_id": "cluster-001", "peak_temp_c": 925.0},
        ),
        AlertType(
            alert_id="ALT-002",
            furnace_id="FRN-001",
            alert_type="RUL_WARNING",
            severity=AlertSeverityEnum.CRITICAL,
            status=AlertStatusEnum.ACTIVE,
            title="Burner Unit 3 requires urgent maintenance",
            description="RUL prediction indicates 65% failure probability within 30 days",
            source="RULPredictor",
            created_at=now - timedelta(hours=2),
            acknowledged_at=None,
            acknowledged_by=None,
            resolved_at=None,
            resolved_by=None,
            metadata={"component_id": "BRN-003", "rul_days": 25.0},
        ),
        AlertType(
            alert_id="ALT-003",
            furnace_id="FRN-002",
            alert_type="EFFICIENCY_DROP",
            severity=AlertSeverityEnum.MEDIUM,
            status=AlertStatusEnum.ACKNOWLEDGED,
            title="Thermal efficiency below target",
            description="Efficiency dropped to 82.5% (target: 90%)",
            source="EfficiencyCalculator",
            created_at=now - timedelta(hours=6),
            acknowledged_at=now - timedelta(hours=4),
            acknowledged_by="operator@example.com",
            resolved_at=None,
            resolved_by=None,
            metadata={"efficiency_pct": 82.5, "target_pct": 90.0},
        ),
    ]

    # Apply filters
    if furnace_id:
        alerts = [a for a in alerts if a.furnace_id == furnace_id]
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    if status:
        alerts = [a for a in alerts if a.status == status]

    return alerts[:limit]


def _get_mock_furnaces(filter_input: Optional[FurnaceFilterInput] = None) -> List[FurnaceType]:
    """Generate mock furnace list."""
    now = datetime.now(timezone.utc)

    furnaces = [
        FurnaceType(
            furnace_id="FRN-001",
            name="Process Heater A",
            location="Plant 1 - North",
            furnace_type="Reformer",
            capacity_mw=25.0,
            status="operating",
            commissioned_at=now - timedelta(days=3650),
            last_maintenance=now - timedelta(days=45),
        ),
        FurnaceType(
            furnace_id="FRN-002",
            name="Process Heater B",
            location="Plant 1 - South",
            furnace_type="Reformer",
            capacity_mw=25.0,
            status="operating",
            commissioned_at=now - timedelta(days=3600),
            last_maintenance=now - timedelta(days=30),
        ),
        FurnaceType(
            furnace_id="FRN-003",
            name="Crude Heater 1",
            location="Plant 2 - CDU",
            furnace_type="Crude",
            capacity_mw=45.0,
            status="operating",
            commissioned_at=now - timedelta(days=5000),
            last_maintenance=now - timedelta(days=60),
        ),
    ]

    if filter_input:
        if filter_input.furnace_ids:
            furnaces = [f for f in furnaces if f.furnace_id in filter_input.furnace_ids]
        if filter_input.location:
            furnaces = [f for f in furnaces if filter_input.location in f.location]
        if filter_input.status:
            furnaces = [f for f in furnaces if f.status == filter_input.status]

    return furnaces


# =============================================================================
# Query Resolvers
# =============================================================================

@strawberry.type
class FurnacePulseQuery:
    """GraphQL queries for FurnacePulse API."""

    @strawberry.field
    def health(self) -> str:
        """Health check."""
        return "GL-007 FurnacePulse GraphQL API is healthy"

    @strawberry.field
    def furnace(self, furnace_id: str) -> Optional[FurnaceType]:
        """
        Get a single furnace by ID.

        Args:
            furnace_id: Furnace identifier.

        Returns:
            Furnace details or None if not found.
        """
        furnaces = _get_mock_furnaces()
        for f in furnaces:
            if f.furnace_id == furnace_id:
                return f
        return None

    @strawberry.field
    def furnaces(
        self,
        filter: Optional[FurnaceFilterInput] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[FurnaceType]:
        """
        List furnaces with optional filtering.

        Args:
            filter: Optional filter criteria.
            limit: Maximum number of results.
            offset: Result offset for pagination.

        Returns:
            List of furnaces.
        """
        furnaces = _get_mock_furnaces(filter)
        return furnaces[offset:offset + limit]

    @strawberry.field
    def alerts(
        self,
        filter: Optional[AlertFilterInput] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AlertType]:
        """
        List alerts with optional filtering.

        Args:
            filter: Optional filter criteria.
            limit: Maximum number of results.
            offset: Result offset for pagination.

        Returns:
            List of alerts.
        """
        furnace_id = filter.furnace_id if filter else None
        severity = filter.severity if filter else None
        status = filter.status if filter else None

        alerts = _get_mock_alerts(furnace_id, severity, status, limit + offset)
        return alerts[offset:offset + limit]

    @strawberry.field
    def predictions(
        self,
        filter: Optional[PredictionFilterInput] = None,
        limit: int = 50,
    ) -> List[RULPredictionType]:
        """
        List RUL predictions with optional filtering.

        Args:
            filter: Optional filter criteria.
            limit: Maximum number of results.

        Returns:
            List of RUL predictions.
        """
        furnace_id = filter.furnace_id if filter else "FRN-001"
        component_type = filter.component_type if filter else None

        rul_summary = _get_mock_rul_predictions(furnace_id, component_type, False)
        return rul_summary.predictions[:limit]

    @strawberry.field
    def compliance(self, furnace_id: str) -> ComplianceStatusType:
        """
        Get NFPA 86 compliance status for a furnace.

        Args:
            furnace_id: Furnace identifier.

        Returns:
            Compliance status with requirements.
        """
        return _get_mock_compliance(furnace_id)

    @strawberry.field
    def explanation(
        self,
        prediction_id: str,
        method: ExplanationMethodEnum = ExplanationMethodEnum.SHAP,
    ) -> ExplanationType:
        """
        Get explanation for a ML prediction.

        Args:
            prediction_id: Prediction identifier.
            method: Explanation method to use.

        Returns:
            Feature importances and explanation.
        """
        now = datetime.now(timezone.utc)

        feature_importance = [
            FeatureImportanceType(
                feature_name="operating_hours_since_maintenance",
                importance_value=0.35,
                direction="positive",
                contribution_pct=35.0,
            ),
            FeatureImportanceType(
                feature_name="avg_temperature_delta",
                importance_value=0.25,
                direction="positive",
                contribution_pct=25.0,
            ),
            FeatureImportanceType(
                feature_name="vibration_rms",
                importance_value=0.18,
                direction="positive",
                contribution_pct=18.0,
            ),
        ]

        hash_input = f"{prediction_id}:{method.value}:{now.isoformat()}"
        computation_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        return ExplanationType(
            prediction_id=prediction_id,
            prediction_type="rul_prediction",
            prediction_value=25.0,
            explanation_method=method,
            feature_importance=feature_importance,
            summary="RUL prediction driven by operating hours and temperature delta.",
            key_drivers=[
                "Operating hours 46% above service interval",
                "Temperature delta trending upward",
            ],
            confidence_score=0.87,
            counterfactuals=None,
            generated_at=now,
            computation_hash=computation_hash,
        )


# =============================================================================
# Mutation Resolvers
# =============================================================================

@strawberry.type
class FurnacePulseMutation:
    """GraphQL mutations for FurnacePulse API."""

    @strawberry.mutation
    def acknowledge_alert(
        self,
        input: AlertAcknowledgeInput,
        info: Info,
    ) -> AlertType:
        """
        Acknowledge an alert.

        Args:
            input: Acknowledgment input.
            info: GraphQL context info.

        Returns:
            Updated alert.
        """
        now = datetime.now(timezone.utc)

        # In production, update in database
        return AlertType(
            alert_id=input.alert_id,
            furnace_id="FRN-001",
            alert_type="HOTSPOT_DETECTED",
            severity=AlertSeverityEnum.HIGH,
            status=AlertStatusEnum.ACKNOWLEDGED,
            title="High temperature hotspot detected in Zone B2",
            description="TMT readings show sustained temperature 45C above normal",
            source="HotspotDetector",
            created_at=now - timedelta(minutes=15),
            acknowledged_at=now,
            acknowledged_by="user@example.com",
            resolved_at=None,
            resolved_by=None,
            metadata={"notes": input.notes} if input.notes else None,
        )

    @strawberry.mutation
    def generate_evidence(
        self,
        input: EvidenceGenerateInput,
    ) -> EvidencePackageType:
        """
        Generate evidence package for compliance/audit.

        Args:
            input: Evidence package specification.

        Returns:
            Evidence package metadata.
        """
        now = datetime.now(timezone.utc)
        package_id = f"EVP-{uuid.uuid4().hex[:12].upper()}"

        return EvidencePackageType(
            package_id=package_id,
            furnace_id=input.furnace_id,
            package_type=input.package_type,
            status="processing",
            created_at=now,
            download_url=None,
            sha256_hash=None,
            size_bytes=None,
            expires_at=now + timedelta(days=7),
        )

    @strawberry.mutation
    def create_work_order(
        self,
        input: WorkOrderCreateInput,
    ) -> WorkOrderType:
        """
        Create a maintenance work order.

        Args:
            input: Work order details.

        Returns:
            Created work order.
        """
        now = datetime.now(timezone.utc)
        work_order_id = f"WO-{uuid.uuid4().hex[:8].upper()}"

        return WorkOrderType(
            work_order_id=work_order_id,
            furnace_id=input.furnace_id,
            component_id=input.component_id,
            priority=input.priority,
            title=input.title,
            description=input.description,
            assigned_to=input.assigned_to,
            due_date=input.due_date,
            created_at=now,
            status="open",
        )


# =============================================================================
# Subscription Resolvers
# =============================================================================

@strawberry.type
class FurnacePulseSubscription:
    """GraphQL subscriptions for real-time updates."""

    @strawberry.subscription
    async def telemetry_updates(
        self,
        furnace_id: str,
        sensor_type: Optional[str] = None,
    ) -> AsyncGenerator[TelemetryUpdateType, None]:
        """
        Subscribe to real-time telemetry updates.

        Args:
            furnace_id: Furnace to monitor.
            sensor_type: Optional sensor type filter.

        Yields:
            Telemetry updates as they occur.
        """
        import asyncio
        import random

        while True:
            await asyncio.sleep(1.0)  # Simulate 1 Hz telemetry

            yield TelemetryUpdateType(
                furnace_id=furnace_id,
                timestamp=datetime.now(timezone.utc),
                sensor_id=f"TMT-{random.randint(1, 48):03d}",
                measurement_type="temperature",
                value=850.0 + random.uniform(-30, 50),
                unit="C",
                quality_score=0.98,
            )

    @strawberry.subscription
    async def alert_stream(
        self,
        furnace_id: Optional[str] = None,
        min_severity: Optional[AlertSeverityEnum] = None,
    ) -> AsyncGenerator[AlertNotificationType, None]:
        """
        Subscribe to real-time alert notifications.

        Args:
            furnace_id: Optional furnace filter.
            min_severity: Minimum severity to receive.

        Yields:
            Alert notifications as they occur.
        """
        import asyncio

        while True:
            await asyncio.sleep(30.0)  # Simulate alert every 30 seconds

            alert = AlertType(
                alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                furnace_id=furnace_id or "FRN-001",
                alert_type="HOTSPOT_DETECTED",
                severity=AlertSeverityEnum.HIGH,
                status=AlertStatusEnum.ACTIVE,
                title="New hotspot detected",
                description="TMT readings show elevated temperature",
                source="HotspotDetector",
                created_at=datetime.now(timezone.utc),
                acknowledged_at=None,
                acknowledged_by=None,
                resolved_at=None,
                resolved_by=None,
                metadata=None,
            )

            yield AlertNotificationType(
                alert=alert,
                event_type="created",
            )


# =============================================================================
# Create Schema
# =============================================================================

if HAS_STRAWBERRY:
    schema = strawberry.Schema(
        query=FurnacePulseQuery,
        mutation=FurnacePulseMutation,
        subscription=FurnacePulseSubscription,
    )
else:
    schema = None
    logger.warning("Strawberry not installed - GraphQL unavailable")
