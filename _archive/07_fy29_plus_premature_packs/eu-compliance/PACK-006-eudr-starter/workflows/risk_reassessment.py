# -*- coding: utf-8 -*-
"""
Risk Reassessment Workflow
============================

Three-phase periodic risk reassessment workflow for EUDR compliance. Gathers
updated data sources, recalculates risk scores, and generates alerts for
material risk changes.

Regulatory Context:
    Per EU Regulation 2023/1115 (EUDR):
    - Article 11: Operators must review and update due diligence as necessary
    - Article 13: Monitoring obligations include periodic reassessment of
      risk factors as new information becomes available
    - Article 29: Country risk benchmarking is updated periodically by the
      Commission; operators must track changes
    - Article 10(2)(c): Risk mitigation must be proportionate to the risk
      level; reassessment ensures mitigation remains adequate

    Triggers for reassessment include:
    - New deforestation data (satellite monitoring alerts)
    - Country risk reclassification by the Commission
    - Supplier certification expiry or revocation
    - New market intelligence on commodity risk
    - Periodic schedule (quarterly minimum recommended)

Phases:
    1. Data collection - Gather updated data sources
    2. Risk recalculation - Recalculate and compare risk scores
    3. Alert generation - Generate alerts for material changes

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RiskLevel(str, Enum):
    """Risk classification level."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class AlertType(str, Enum):
    """Risk change alert type."""
    RISK_INCREASE = "risk_increase"
    RISK_DECREASE = "risk_decrease"
    THRESHOLD_CROSSED = "threshold_crossed"
    CERTIFICATION_EXPIRED = "certification_expired"
    COUNTRY_RECLASSIFIED = "country_reclassified"
    DEFORESTATION_ALERT = "deforestation_alert"
    DD_TYPE_CHANGE = "dd_type_change"


class DDType(str, Enum):
    """Due diligence type."""
    STANDARD = "standard"
    SIMPLIFIED = "simplified"


# Country risk benchmarking
HIGH_RISK_COUNTRIES = {
    "BR", "CD", "CM", "CO", "CI", "EC", "GA", "GH", "GT", "GN",
    "HN", "ID", "KH", "LA", "LR", "MG", "MM", "MY", "MZ", "NG",
    "PA", "PE", "PG", "PH", "SL", "TZ", "TH", "UG", "VN",
}

LOW_RISK_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "NO", "IS", "CH", "LI", "GB", "AU", "NZ", "JP", "KR", "CA",
}

# Material change threshold (points)
DEFAULT_MATERIAL_CHANGE_THRESHOLD = 10.0


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    checkpoints: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = Field(None)
    last_checkpoint_at: Optional[datetime] = Field(None)

    class Config:
        arbitrary_types_allowed = True


class RiskReassessmentInput(BaseModel):
    """Input data for the risk reassessment workflow."""
    supplier_ids: List[str] = Field(
        default_factory=list, description="Supplier IDs to reassess (empty = all)"
    )
    material_change_threshold: float = Field(
        default=10.0, ge=0.0,
        description="Score change (points) to flag as material"
    )
    include_deforestation_data: bool = Field(
        default=True, description="Include satellite deforestation alerts"
    )
    include_market_data: bool = Field(
        default=True, description="Include commodity market risk updates"
    )
    assessment_date: Optional[str] = Field(
        None, description="Assessment date YYYY-MM-DD (default: today)"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class RiskAlert(BaseModel):
    """Risk change alert."""
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity")
    supplier_id: str = Field(default="", description="Affected supplier")
    supplier_name: str = Field(default="", description="Supplier name")
    description: str = Field(..., description="Alert description")
    previous_score: float = Field(default=0.0, description="Previous risk score")
    new_score: float = Field(default=0.0, description="New risk score")
    change_magnitude: float = Field(default=0.0, description="Score change")
    recommended_action: str = Field(default="", description="Recommended action")
    created_at: str = Field(default="", description="Alert creation timestamp")


class RiskReassessmentResult(BaseModel):
    """Complete result from the risk reassessment workflow."""
    workflow_name: str = Field(default="risk_reassessment")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    suppliers_assessed: int = Field(default=0, ge=0)
    material_changes: int = Field(default=0, ge=0)
    alerts_generated: int = Field(default=0, ge=0)
    critical_alerts: int = Field(default=0, ge=0)
    threshold_crossings: int = Field(default=0, ge=0)
    suppliers_upgraded: int = Field(default=0, ge=0)
    suppliers_downgraded: int = Field(default=0, ge=0)
    previous_scores_archived: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# RISK REASSESSMENT WORKFLOW
# =============================================================================


class RiskReassessmentWorkflow:
    """
    Three-phase periodic risk reassessment workflow.

    Gathers updated data sources, recalculates all risk scores,
    compares with previous assessments, and generates alerts for
    material changes.

    Attributes:
        config: Workflow configuration.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.
        _checkpoint_store: Checkpoint data for resume.

    Example:
        >>> wf = RiskReassessmentWorkflow()
        >>> result = await wf.run(RiskReassessmentInput(
        ...     material_change_threshold=10.0,
        ... ))
        >>> assert result.status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the RiskReassessmentWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.RiskReassessmentWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []
        self._checkpoint_store: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(
        self, input_data: RiskReassessmentInput
    ) -> RiskReassessmentResult:
        """
        Execute the full 3-phase risk reassessment workflow.

        Args:
            input_data: Reassessment parameters and configuration.

        Returns:
            RiskReassessmentResult with alerts and score changes.
        """
        started_at = datetime.utcnow()

        self.logger.info(
            "Starting risk reassessment workflow execution_id=%s threshold=%.1f",
            self._execution_id, input_data.material_change_threshold,
        )

        context = WorkflowContext(
            execution_id=self._execution_id,
            config={**self.config, **input_data.config},
            started_at=started_at,
            state={
                "supplier_ids": input_data.supplier_ids,
                "material_change_threshold": input_data.material_change_threshold,
                "include_deforestation_data": input_data.include_deforestation_data,
                "include_market_data": input_data.include_market_data,
                "assessment_date": (
                    input_data.assessment_date
                    or datetime.utcnow().strftime("%Y-%m-%d")
                ),
            },
        )

        phase_handlers = [
            ("data_collection", self._phase_1_data_collection),
            ("risk_recalculation", self._phase_2_risk_recalculation),
            ("alert_generation", self._phase_3_alert_generation),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase_name, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase_name)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (
                    datetime.utcnow() - phase_start
                ).total_seconds()
            except Exception as exc:
                self.logger.error(
                    "Phase '%s' failed: %s", phase_name, exc, exc_info=True,
                )
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc)}),
                )

            self._phase_results.append(phase_result)
            context.phase_results = list(self._phase_results)

            self._checkpoint_store[phase_name] = {
                "result": phase_result.model_dump(),
                "saved_at": datetime.utcnow().isoformat(),
            }

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "data_collection":
                    self.logger.error("Data collection failed; halting.")
                    break

        completed_at = datetime.utcnow()

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
        })

        alerts = context.state.get("alerts", [])

        self.logger.info(
            "Risk reassessment finished execution_id=%s status=%s alerts=%d",
            self._execution_id, overall_status.value, len(alerts),
        )

        return RiskReassessmentResult(
            status=overall_status,
            phases=self._phase_results,
            suppliers_assessed=context.state.get("suppliers_assessed", 0),
            material_changes=context.state.get("material_changes", 0),
            alerts_generated=len(alerts),
            critical_alerts=sum(
                1 for a in alerts if a.get("severity") == "critical"
            ),
            threshold_crossings=context.state.get("threshold_crossings", 0),
            suppliers_upgraded=context.state.get("suppliers_upgraded", 0),
            suppliers_downgraded=context.state.get("suppliers_downgraded", 0),
            previous_scores_archived=context.state.get("scores_archived", False),
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_1_data_collection(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Gather updated data sources: new country deforestation data,
        supplier certification updates, commodity market changes, and
        document submissions.
        """
        phase_name = "data_collection"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        supplier_ids = context.state.get("supplier_ids", [])
        include_deforestation = context.state.get("include_deforestation_data", True)
        include_market = context.state.get("include_market_data", True)

        # Fetch current supplier data with previous scores
        suppliers = await self._fetch_suppliers_with_scores(supplier_ids)
        if not suppliers:
            suppliers = self._generate_sample_data()
            warnings.append(
                "No supplier data found. Using sample data for demonstration."
            )

        self.logger.info("Collected data for %d supplier(s)", len(suppliers))

        # Fetch deforestation data
        deforestation_alerts: List[Dict[str, Any]] = []
        if include_deforestation:
            deforestation_alerts = await self._fetch_deforestation_alerts()
            if deforestation_alerts:
                warnings.append(
                    f"{len(deforestation_alerts)} deforestation alert(s) "
                    "detected in supplier regions."
                )

        # Fetch certification updates
        cert_updates = await self._fetch_certification_updates(supplier_ids)

        # Fetch commodity market risk changes
        market_changes: Dict[str, float] = {}
        if include_market:
            market_changes = await self._fetch_market_risk_changes()

        # Fetch country risk reclassifications
        country_reclassifications = await self._fetch_country_reclassifications()
        if country_reclassifications:
            for reclass in country_reclassifications:
                warnings.append(
                    f"Country {reclass.get('country', '')} reclassified: "
                    f"{reclass.get('old_level', '')} -> {reclass.get('new_level', '')}"
                )

        context.state["suppliers"] = suppliers
        context.state["deforestation_alerts"] = deforestation_alerts
        context.state["cert_updates"] = cert_updates
        context.state["market_changes"] = market_changes
        context.state["country_reclassifications"] = country_reclassifications

        outputs["suppliers_loaded"] = len(suppliers)
        outputs["deforestation_alerts"] = len(deforestation_alerts)
        outputs["certification_updates"] = len(cert_updates)
        outputs["market_risk_changes"] = len(market_changes)
        outputs["country_reclassifications"] = len(country_reclassifications)

        self.logger.info(
            "Phase 1 complete: %d suppliers, %d deforestation alerts, "
            "%d cert updates, %d market changes",
            len(suppliers), len(deforestation_alerts),
            len(cert_updates), len(market_changes),
        )

        provenance = self._hash({
            "phase": phase_name,
            "suppliers": len(suppliers),
            "data_sources": {
                "deforestation": len(deforestation_alerts),
                "certs": len(cert_updates),
                "market": len(market_changes),
            },
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Risk Recalculation
    # -------------------------------------------------------------------------

    async def _phase_2_risk_recalculation(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Recalculate all risk scores, compare with previous assessment,
        identify material changes (> threshold point change), and flag
        for attention.
        """
        phase_name = "risk_recalculation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        suppliers = context.state.get("suppliers", [])
        threshold = context.state.get("material_change_threshold", 10.0)
        deforestation_alerts = context.state.get("deforestation_alerts", [])
        cert_updates = context.state.get("cert_updates", [])
        market_changes = context.state.get("market_changes", {})
        country_reclassifications = context.state.get("country_reclassifications", [])

        if not suppliers:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"scored": 0},
                warnings=["No suppliers for recalculation"],
                provenance_hash=self._hash({"phase": phase_name, "scored": 0}),
            )

        # Build lookup maps
        reclassified_countries = {
            r["country"]: r["new_level"]
            for r in country_reclassifications
        }
        deforestation_countries = {
            a.get("country", "") for a in deforestation_alerts
        }
        cert_update_map: Dict[str, str] = {}
        for cu in cert_updates:
            sid = cu.get("supplier_id", "")
            status = cu.get("new_status", "")
            cert_update_map[sid] = status

        commodity_risk_map: Dict[str, float] = {
            "oil_palm": 85.0, "soya": 75.0, "cattle": 70.0,
            "cocoa": 65.0, "rubber": 60.0, "coffee": 55.0, "wood": 50.0,
        }

        scored_suppliers: List[Dict[str, Any]] = []
        material_changes: List[Dict[str, Any]] = []
        threshold_crossings = 0
        suppliers_upgraded = 0
        suppliers_downgraded = 0

        for supplier in suppliers:
            sid = supplier.get("supplier_id", "")
            country = supplier.get("country_code", "")
            commodity = supplier.get("commodity", "")
            previous_score = supplier.get("previous_composite_score", 50.0)
            previous_level = supplier.get("previous_risk_level", "standard")

            # Country risk (35%)
            if country in reclassified_countries:
                new_level = reclassified_countries[country]
                country_score = {"high": 80.0, "standard": 50.0, "low": 15.0}.get(
                    new_level, 50.0
                )
            elif country in HIGH_RISK_COUNTRIES:
                country_score = 80.0
            elif country in LOW_RISK_COUNTRIES:
                country_score = 15.0
            else:
                country_score = 50.0

            # Deforestation adjustment
            if country in deforestation_countries:
                country_score = min(100.0, country_score + 15.0)

            # Commodity risk (25%)
            commodity_base = commodity_risk_map.get(commodity, 50.0)
            market_adjustment = market_changes.get(commodity, 0.0)
            commodity_score = min(100.0, max(0.0, commodity_base + market_adjustment))

            # Supplier-specific risk (25%)
            cert_status = cert_update_map.get(sid, "valid")
            if cert_status == "expired":
                supplier_specific = 70.0
            elif cert_status == "revoked":
                supplier_specific = 90.0
            else:
                valid_certs = supplier.get("valid_certifications", 0)
                supplier_specific = max(10.0, 60.0 - (valid_certs * 10.0))

            # Document completeness risk (15%)
            doc_completeness = supplier.get("document_completeness", 0.5)
            doc_risk = (1.0 - doc_completeness) * 100.0

            # Composite (deterministic)
            composite = (
                country_score * 0.35
                + commodity_score * 0.25
                + supplier_specific * 0.25
                + doc_risk * 0.15
            )
            composite = round(min(100.0, max(0.0, composite)), 2)

            # Determine risk level
            if composite >= 70.0:
                risk_level = RiskLevel.HIGH
            elif composite >= 30.0:
                risk_level = RiskLevel.STANDARD
            else:
                risk_level = RiskLevel.LOW

            # DD type
            if country in LOW_RISK_COUNTRIES and composite < 30.0:
                dd_type = DDType.SIMPLIFIED
            else:
                dd_type = DDType.STANDARD

            # Detect material change
            change = composite - previous_score
            is_material = abs(change) >= threshold

            if is_material:
                material_changes.append({
                    "supplier_id": sid,
                    "supplier_name": supplier.get("supplier_name", ""),
                    "previous_score": previous_score,
                    "new_score": composite,
                    "change": round(change, 2),
                    "direction": "increased" if change > 0 else "decreased",
                    "previous_level": previous_level,
                    "new_level": risk_level.value,
                })

            # Detect threshold crossing (level change)
            if risk_level.value != previous_level:
                threshold_crossings += 1
                if self._risk_level_to_int(risk_level.value) > self._risk_level_to_int(previous_level):
                    suppliers_downgraded += 1
                else:
                    suppliers_upgraded += 1

            supplier["new_risk"] = {
                "composite_score": composite,
                "risk_level": risk_level.value,
                "dd_type": dd_type.value,
                "country_score": round(country_score, 2),
                "commodity_score": round(commodity_score, 2),
                "supplier_score": round(supplier_specific, 2),
                "doc_risk": round(doc_risk, 2),
                "change_from_previous": round(change, 2),
                "is_material_change": is_material,
                "assessed_at": datetime.utcnow().isoformat(),
            }

            scored_suppliers.append(supplier)

        context.state["scored_suppliers"] = scored_suppliers
        context.state["material_changes_list"] = material_changes
        context.state["suppliers_assessed"] = len(scored_suppliers)
        context.state["material_changes"] = len(material_changes)
        context.state["threshold_crossings"] = threshold_crossings
        context.state["suppliers_upgraded"] = suppliers_upgraded
        context.state["suppliers_downgraded"] = suppliers_downgraded

        outputs["suppliers_assessed"] = len(scored_suppliers)
        outputs["material_changes"] = len(material_changes)
        outputs["threshold_crossings"] = threshold_crossings
        outputs["suppliers_upgraded"] = suppliers_upgraded
        outputs["suppliers_downgraded"] = suppliers_downgraded
        outputs["risk_distribution"] = {
            "high": sum(
                1 for s in scored_suppliers
                if s.get("new_risk", {}).get("risk_level") == "high"
            ),
            "standard": sum(
                1 for s in scored_suppliers
                if s.get("new_risk", {}).get("risk_level") == "standard"
            ),
            "low": sum(
                1 for s in scored_suppliers
                if s.get("new_risk", {}).get("risk_level") == "low"
            ),
        }

        if len(material_changes) > 0:
            warnings.append(
                f"{len(material_changes)} material risk change(s) detected "
                f"(threshold: {threshold:.1f} points)."
            )

        self.logger.info(
            "Phase 2 complete: %d assessed, %d material changes, "
            "%d threshold crossings",
            len(scored_suppliers), len(material_changes), threshold_crossings,
        )

        provenance = self._hash({
            "phase": phase_name,
            "assessed": len(scored_suppliers),
            "material": len(material_changes),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Alert Generation
    # -------------------------------------------------------------------------

    async def _phase_3_alert_generation(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Generate risk change alerts, update risk dashboards, trigger
        re-assessment workflows for suppliers crossing risk thresholds,
        and archive previous scores for trend analysis.
        """
        phase_name = "alert_generation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        material_changes = context.state.get("material_changes_list", [])
        scored_suppliers = context.state.get("scored_suppliers", [])
        deforestation_alerts = context.state.get("deforestation_alerts", [])
        country_reclassifications = context.state.get("country_reclassifications", [])
        cert_updates = context.state.get("cert_updates", [])

        alerts: List[Dict[str, Any]] = []
        now = datetime.utcnow().isoformat()

        # Generate alerts for material risk changes
        for change in material_changes:
            alert_id = f"ALERT-{uuid.uuid4().hex[:8]}"
            magnitude = abs(change["change"])

            if magnitude >= 20.0:
                severity = AlertSeverity.CRITICAL
            elif magnitude >= 15.0:
                severity = AlertSeverity.HIGH
            elif magnitude >= 10.0:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

            alert_type = (
                AlertType.RISK_INCREASE
                if change["direction"] == "increased"
                else AlertType.RISK_DECREASE
            )

            # Check if threshold was crossed
            if change["previous_level"] != change["new_level"]:
                alert_type = AlertType.THRESHOLD_CROSSED
                if change["new_level"] == "high":
                    severity = AlertSeverity.CRITICAL

            recommended_action = self._determine_action(
                alert_type, severity, change,
            )

            alerts.append({
                "alert_id": alert_id,
                "alert_type": alert_type.value,
                "severity": severity.value,
                "supplier_id": change["supplier_id"],
                "supplier_name": change["supplier_name"],
                "description": (
                    f"Risk score {change['direction']} by "
                    f"{magnitude:.1f} points: "
                    f"{change['previous_score']:.1f} -> {change['new_score']:.1f} "
                    f"(level: {change['previous_level']} -> {change['new_level']})"
                ),
                "previous_score": change["previous_score"],
                "new_score": change["new_score"],
                "change_magnitude": round(magnitude, 2),
                "recommended_action": recommended_action,
                "created_at": now,
            })

        # Generate alerts for deforestation detections
        for df_alert in deforestation_alerts:
            alert_id = f"ALERT-{uuid.uuid4().hex[:8]}"
            alerts.append({
                "alert_id": alert_id,
                "alert_type": AlertType.DEFORESTATION_ALERT.value,
                "severity": AlertSeverity.CRITICAL.value,
                "supplier_id": df_alert.get("supplier_id", ""),
                "supplier_name": df_alert.get("supplier_name", ""),
                "description": (
                    f"Deforestation detected in {df_alert.get('country', '')} "
                    f"region: {df_alert.get('description', 'satellite alert')}"
                ),
                "previous_score": 0.0,
                "new_score": 0.0,
                "change_magnitude": 0.0,
                "recommended_action": (
                    "Immediately investigate deforestation alert. "
                    "Suspend imports from affected plots until verified."
                ),
                "created_at": now,
            })

        # Generate alerts for country reclassifications
        for reclass in country_reclassifications:
            alert_id = f"ALERT-{uuid.uuid4().hex[:8]}"
            alerts.append({
                "alert_id": alert_id,
                "alert_type": AlertType.COUNTRY_RECLASSIFIED.value,
                "severity": AlertSeverity.HIGH.value,
                "supplier_id": "",
                "supplier_name": "",
                "description": (
                    f"Country {reclass.get('country', '')} reclassified: "
                    f"{reclass.get('old_level', '')} -> {reclass.get('new_level', '')}. "
                    "All suppliers in this country require reassessment."
                ),
                "previous_score": 0.0,
                "new_score": 0.0,
                "change_magnitude": 0.0,
                "recommended_action": (
                    "Review all suppliers sourcing from this country. "
                    "Update DD type (standard/simplified) accordingly."
                ),
                "created_at": now,
            })

        # Generate alerts for certification expirations
        for cu in cert_updates:
            if cu.get("new_status") in ("expired", "revoked"):
                alert_id = f"ALERT-{uuid.uuid4().hex[:8]}"
                alerts.append({
                    "alert_id": alert_id,
                    "alert_type": AlertType.CERTIFICATION_EXPIRED.value,
                    "severity": AlertSeverity.HIGH.value,
                    "supplier_id": cu.get("supplier_id", ""),
                    "supplier_name": cu.get("supplier_name", ""),
                    "description": (
                        f"Certification {cu.get('cert_id', '')} "
                        f"({cu.get('cert_type', '')}) is now "
                        f"{cu.get('new_status', 'expired')}."
                    ),
                    "previous_score": 0.0,
                    "new_score": 0.0,
                    "change_magnitude": 0.0,
                    "recommended_action": (
                        "Request certification renewal from supplier. "
                        "Update risk score to reflect loss of certification."
                    ),
                    "created_at": now,
                })

        # Archive previous scores for trend analysis
        archive_result = await self._archive_scores(scored_suppliers)
        context.state["scores_archived"] = archive_result.get("archived", False)

        # Update risk dashboard
        dashboard_updated = await self._update_risk_dashboard(alerts, scored_suppliers)

        context.state["alerts"] = alerts

        outputs["alerts_generated"] = len(alerts)
        outputs["alert_severity_breakdown"] = {
            "critical": sum(1 for a in alerts if a["severity"] == "critical"),
            "high": sum(1 for a in alerts if a["severity"] == "high"),
            "medium": sum(1 for a in alerts if a["severity"] == "medium"),
            "low": sum(1 for a in alerts if a["severity"] == "low"),
            "info": sum(1 for a in alerts if a["severity"] == "info"),
        }
        outputs["alert_type_breakdown"] = {}
        for a in alerts:
            at = a["alert_type"]
            outputs["alert_type_breakdown"][at] = (
                outputs["alert_type_breakdown"].get(at, 0) + 1
            )
        outputs["scores_archived"] = archive_result.get("archived", False)
        outputs["dashboard_updated"] = dashboard_updated

        critical_count = sum(1 for a in alerts if a["severity"] == "critical")
        if critical_count > 0:
            warnings.append(
                f"{critical_count} CRITICAL alert(s) generated. "
                "Immediate action required."
            )

        self.logger.info(
            "Phase 3 complete: %d alerts generated (critical=%d)",
            len(alerts), critical_count,
        )

        provenance = self._hash({
            "phase": phase_name,
            "alerts": len(alerts),
            "critical": critical_count,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _risk_level_to_int(self, level: str) -> int:
        """Convert risk level to integer for comparison."""
        return {"low": 0, "standard": 1, "high": 2}.get(level, 1)

    def _determine_action(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        change: Dict[str, Any],
    ) -> str:
        """Determine recommended action based on alert details."""
        if alert_type == AlertType.THRESHOLD_CROSSED:
            new_level = change.get("new_level", "standard")
            if new_level == "high":
                return (
                    "Upgrade to enhanced due diligence. Review all active "
                    "DDS for this supplier and implement additional "
                    "mitigation measures."
                )
            elif new_level == "low":
                return (
                    "Supplier now eligible for simplified due diligence. "
                    "Review and update DD type in active DDS."
                )
            return "Review supplier risk profile and update DD type."

        if severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
            return (
                "Review supplier risk profile immediately. Consider "
                "additional verification or import suspension."
            )

        if alert_type == AlertType.RISK_DECREASE:
            return (
                "Monitor supplier for sustained improvement. "
                "Consider upgrading DD type if sustained."
            )

        return "Review and acknowledge risk change."

    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample supplier data for demonstration."""
        return [
            {
                "supplier_id": "SUP-SAMPLE-001",
                "supplier_name": "Brazil Coffee Co",
                "country_code": "BR",
                "commodity": "coffee",
                "previous_composite_score": 55.0,
                "previous_risk_level": "standard",
                "valid_certifications": 1,
                "document_completeness": 0.7,
            },
            {
                "supplier_id": "SUP-SAMPLE-002",
                "supplier_name": "Indonesia Palm Oil Ltd",
                "country_code": "ID",
                "commodity": "oil_palm",
                "previous_composite_score": 72.0,
                "previous_risk_level": "high",
                "valid_certifications": 0,
                "document_completeness": 0.4,
            },
        ]

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _fetch_suppliers_with_scores(
        self, supplier_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch suppliers with their previous risk scores."""
        await asyncio.sleep(0)
        return []

    async def _fetch_deforestation_alerts(self) -> List[Dict[str, Any]]:
        """Fetch satellite deforestation alerts."""
        await asyncio.sleep(0)
        return []

    async def _fetch_certification_updates(
        self, supplier_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch certification status updates."""
        await asyncio.sleep(0)
        return []

    async def _fetch_market_risk_changes(self) -> Dict[str, float]:
        """Fetch commodity market risk score adjustments."""
        await asyncio.sleep(0)
        return {}

    async def _fetch_country_reclassifications(self) -> List[Dict[str, Any]]:
        """Fetch country risk reclassification updates."""
        await asyncio.sleep(0)
        return []

    async def _archive_scores(
        self, suppliers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Archive previous scores for trend analysis."""
        await asyncio.sleep(0)
        return {"archived": True, "count": len(suppliers)}

    async def _update_risk_dashboard(
        self,
        alerts: List[Dict[str, Any]],
        suppliers: List[Dict[str, Any]],
    ) -> bool:
        """Update the risk monitoring dashboard."""
        await asyncio.sleep(0)
        return True

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
