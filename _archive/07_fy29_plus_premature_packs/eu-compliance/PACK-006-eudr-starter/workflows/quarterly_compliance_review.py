# -*- coding: utf-8 -*-
"""
Quarterly Compliance Review Workflow
======================================

Three-phase quarterly compliance review workflow for EUDR. Refreshes supplier
data, recalculates risk scores, and generates compliance reporting to maintain
ongoing EUDR compliance posture.

Regulatory Context:
    Per EU Regulation 2023/1115 (EUDR):
    - Article 11: Operators shall review and update due diligence as
      appropriate to ensure ongoing compliance
    - Article 13: Monitoring obligations require periodic reassessment
    - Article 29: Country risk benchmarking may change; operators must
      track updates to the EU Commission's benchmarking list
    - Article 30: Competent authorities conduct checks on operators;
      maintaining current compliance documentation reduces audit risk

    Quarterly reviews ensure that:
    - Certification statuses remain current (not expired)
    - Country risk changes are captured promptly
    - New high-risk suppliers are identified and flagged
    - Simplified DD eligibility is re-evaluated
    - Upcoming DDS deadlines are tracked

Phases:
    1. Data refresh - Update supplier data, re-validate certifications
    2. Risk recalculation - Recalculate all risk scores with updated data
    3. Compliance reporting - Generate quarterly report and dashboard updates

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class DDType(str, Enum):
    """Due diligence type."""
    STANDARD = "standard"
    SIMPLIFIED = "simplified"


class CertificationStatus(str, Enum):
    """Certification validity status."""
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"  # Within 90 days
    EXPIRED = "expired"
    NOT_VERIFIED = "not_verified"


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


class QuarterlyReviewInput(BaseModel):
    """Input data for the quarterly compliance review workflow."""
    quarter: str = Field(..., description="Quarter label e.g. '2026-Q1'")
    year: int = Field(..., ge=2024, description="Review year")
    organization_id: Optional[str] = Field(None, description="Organization identifier")
    supplier_ids: List[str] = Field(default_factory=list, description="Specific supplier IDs to review")
    include_new_transactions: bool = Field(default=True, description="Import new transactions")
    risk_change_threshold: float = Field(default=10.0, ge=0.0, description="Score change to flag")
    config: Dict[str, Any] = Field(default_factory=dict)


class QuarterlyReviewResult(BaseModel):
    """Complete result from the quarterly compliance review workflow."""
    workflow_name: str = Field(default="quarterly_compliance_review")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    quarter: str = Field(default="")
    year: int = Field(default=0)
    suppliers_reviewed: int = Field(default=0, ge=0)
    certifications_checked: int = Field(default=0, ge=0)
    certifications_expiring: int = Field(default=0, ge=0)
    certifications_expired: int = Field(default=0, ge=0)
    risk_changes_flagged: int = Field(default=0, ge=0)
    new_high_risk_suppliers: int = Field(default=0, ge=0)
    simplified_dd_eligible: int = Field(default=0, ge=0)
    compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    upcoming_dds_deadlines: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# QUARTERLY COMPLIANCE REVIEW WORKFLOW
# =============================================================================


class QuarterlyComplianceReviewWorkflow:
    """
    Three-phase quarterly compliance review workflow.

    Orchestrates periodic review of EUDR compliance posture by refreshing
    data, recalculating risks, and generating compliance reports.

    Agent Dependencies:
        - DATA-001 (PDF Extractor)
        - DATA-002 (Excel/CSV Normalizer)
        - EUDR-016 (Country Risk Classifier)
        - EUDR-017 (Supplier Risk Scorer)
        - EUDR-018 (Commodity Risk Analyzer)
        - EUDR-030 (Documentation Generator)
        - GL-EUDR-APP Dashboard API

    Attributes:
        config: Workflow configuration.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.
        _checkpoint_store: Checkpoint data for resume.

    Example:
        >>> wf = QuarterlyComplianceReviewWorkflow()
        >>> result = await wf.run(QuarterlyReviewInput(
        ...     quarter="2026-Q1",
        ...     year=2026,
        ... ))
        >>> assert result.status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the QuarterlyComplianceReviewWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(
            f"{__name__}.QuarterlyComplianceReviewWorkflow"
        )
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []
        self._checkpoint_store: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(
        self, input_data: QuarterlyReviewInput
    ) -> QuarterlyReviewResult:
        """
        Execute the full 3-phase quarterly compliance review workflow.

        Args:
            input_data: Review parameters including quarter, year, and config.

        Returns:
            QuarterlyReviewResult with compliance metrics and recommendations.
        """
        started_at = datetime.utcnow()

        self.logger.info(
            "Starting quarterly compliance review execution_id=%s quarter=%s",
            self._execution_id, input_data.quarter,
        )

        context = WorkflowContext(
            execution_id=self._execution_id,
            config={**self.config, **input_data.config},
            started_at=started_at,
            state={
                "quarter": input_data.quarter,
                "year": input_data.year,
                "organization_id": input_data.organization_id,
                "supplier_ids": input_data.supplier_ids,
                "include_new_transactions": input_data.include_new_transactions,
                "risk_change_threshold": input_data.risk_change_threshold,
            },
        )

        phase_handlers = [
            ("data_refresh", self._phase_1_data_refresh),
            ("risk_recalculation", self._phase_2_risk_recalculation),
            ("compliance_reporting", self._phase_3_compliance_reporting),
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

            # Save checkpoint
            self._checkpoint_store[phase_name] = {
                "result": phase_result.model_dump(),
                "state_snapshot": dict(context.state),
                "saved_at": datetime.utcnow().isoformat(),
            }

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "data_refresh":
                    self.logger.error("Data refresh failed; halting review.")
                    break

        completed_at = datetime.utcnow()

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "quarter": input_data.quarter,
        })

        self.logger.info(
            "Quarterly compliance review finished execution_id=%s status=%s",
            self._execution_id, overall_status.value,
        )

        return QuarterlyReviewResult(
            status=overall_status,
            phases=self._phase_results,
            quarter=input_data.quarter,
            year=input_data.year,
            suppliers_reviewed=context.state.get("suppliers_reviewed", 0),
            certifications_checked=context.state.get("certifications_checked", 0),
            certifications_expiring=context.state.get("certifications_expiring", 0),
            certifications_expired=context.state.get("certifications_expired", 0),
            risk_changes_flagged=context.state.get("risk_changes_flagged", 0),
            new_high_risk_suppliers=context.state.get("new_high_risk_suppliers", 0),
            simplified_dd_eligible=context.state.get("simplified_dd_eligible", 0),
            compliance_score=context.state.get("compliance_score", 0.0),
            upcoming_dds_deadlines=context.state.get("upcoming_dds_deadlines", 0),
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Data Refresh
    # -------------------------------------------------------------------------

    async def _phase_1_data_refresh(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Update supplier data, re-validate certifications (check expiry),
        refresh country risk data, and import new transactions.

        Uses:
            - DATA-001 (PDF Extractor)
            - DATA-002 (Excel/CSV Normalizer)

        Steps:
            1. Fetch current supplier data from the system
            2. Check all certification expiry dates
            3. Identify expired and soon-to-expire certifications
            4. Refresh country risk benchmarking data
            5. Import new transactions if configured
        """
        phase_name = "data_refresh"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        supplier_ids = context.state.get("supplier_ids", [])
        now = datetime.utcnow()
        now_str = now.strftime("%Y-%m-%d")
        expiry_threshold = (now + timedelta(days=90)).strftime("%Y-%m-%d")

        # Fetch supplier data
        suppliers = await self._fetch_supplier_data(supplier_ids)
        if not suppliers:
            suppliers = self._generate_sample_supplier_data()
            warnings.append(
                "No supplier data found in system. Using sample data "
                "for demonstration purposes."
            )

        self.logger.info("Refreshing data for %d supplier(s)", len(suppliers))

        # Check certification expiry
        certifications_checked = 0
        certifications_expiring = 0
        certifications_expired = 0
        cert_details: List[Dict[str, Any]] = []

        for supplier in suppliers:
            certs = supplier.get("certifications", [])
            for cert in certs:
                certifications_checked += 1
                expiry = cert.get("expiry_date", "")

                if expiry and expiry < now_str:
                    cert["status"] = CertificationStatus.EXPIRED.value
                    certifications_expired += 1
                    warnings.append(
                        f"Supplier '{supplier.get('supplier_name', '')}': "
                        f"Certificate {cert.get('cert_id', '')} "
                        f"({cert.get('cert_type', '')}) expired on {expiry}"
                    )
                elif expiry and expiry <= expiry_threshold:
                    cert["status"] = CertificationStatus.EXPIRING_SOON.value
                    certifications_expiring += 1
                    warnings.append(
                        f"Supplier '{supplier.get('supplier_name', '')}': "
                        f"Certificate {cert.get('cert_id', '')} "
                        f"({cert.get('cert_type', '')}) expiring on {expiry}"
                    )
                else:
                    cert["status"] = CertificationStatus.VALID.value

                cert_details.append({
                    "supplier_id": supplier.get("supplier_id", ""),
                    "cert_id": cert.get("cert_id", ""),
                    "cert_type": cert.get("cert_type", ""),
                    "status": cert["status"],
                    "expiry_date": expiry,
                })

        # Refresh country risk data
        country_risk_updates = await self._fetch_country_risk_updates()
        risk_updates_count = len(country_risk_updates) if country_risk_updates else 0

        if risk_updates_count > 0:
            warnings.append(
                f"{risk_updates_count} country risk classification update(s) detected."
            )

        # Import new transactions
        new_transactions = 0
        if context.state.get("include_new_transactions", True):
            new_transactions = await self._import_new_transactions(
                context.state.get("quarter", ""),
            )

        context.state["suppliers"] = suppliers
        context.state["cert_details"] = cert_details
        context.state["certifications_checked"] = certifications_checked
        context.state["certifications_expiring"] = certifications_expiring
        context.state["certifications_expired"] = certifications_expired
        context.state["country_risk_updates"] = country_risk_updates or []

        outputs["suppliers_loaded"] = len(suppliers)
        outputs["certifications_checked"] = certifications_checked
        outputs["certifications_expiring"] = certifications_expiring
        outputs["certifications_expired"] = certifications_expired
        outputs["country_risk_updates"] = risk_updates_count
        outputs["new_transactions_imported"] = new_transactions

        self.logger.info(
            "Phase 1 complete: %d suppliers, %d certs checked, "
            "%d expiring, %d expired",
            len(suppliers), certifications_checked,
            certifications_expiring, certifications_expired,
        )

        provenance = self._hash({
            "phase": phase_name,
            "suppliers": len(suppliers),
            "certs_checked": certifications_checked,
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
        Recalculate all risk scores with updated data, identify risk changes
        exceeding thresholds, flag new high-risk suppliers, and update
        simplified DD eligibility.

        Uses:
            - EUDR-016 (Country Risk Classifier)
            - EUDR-017 (Supplier Risk Scorer)
            - EUDR-018 (Commodity Risk Analyzer)
        """
        phase_name = "risk_recalculation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        suppliers = context.state.get("suppliers", [])
        risk_threshold = context.state.get("risk_change_threshold", 10.0)
        cert_details = context.state.get("cert_details", [])

        if not suppliers:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"scored": 0},
                warnings=["No suppliers for risk recalculation"],
                provenance_hash=self._hash({"phase": phase_name, "scored": 0}),
            )

        # Build certification validity map
        valid_cert_map: Dict[str, int] = {}
        for cd in cert_details:
            if cd.get("status") == CertificationStatus.VALID.value:
                sid = cd["supplier_id"]
                valid_cert_map[sid] = valid_cert_map.get(sid, 0) + 1

        risk_changes: List[Dict[str, Any]] = []
        new_high_risk: List[str] = []
        simplified_eligible: List[str] = []
        scored_suppliers: List[Dict[str, Any]] = []

        commodity_risk_map: Dict[str, float] = {
            "oil_palm": 85.0, "soya": 75.0, "cattle": 70.0,
            "cocoa": 65.0, "rubber": 60.0, "coffee": 55.0, "wood": 50.0,
        }

        for supplier in suppliers:
            sid = supplier.get("supplier_id", "")
            country = supplier.get("country_code", "")
            commodity = supplier.get("commodity", "")
            previous_score = supplier.get("previous_risk_score", 50.0)

            # Country risk
            if country in HIGH_RISK_COUNTRIES:
                country_score = 80.0
            elif country in LOW_RISK_COUNTRIES:
                country_score = 15.0
            else:
                country_score = 50.0

            # Commodity risk
            commodity_score = commodity_risk_map.get(commodity, 50.0)

            # Certification adjustment
            valid_certs = valid_cert_map.get(sid, 0)
            cert_adjustment = min(20.0, valid_certs * 5.0)

            # Expired cert penalty
            expired_certs = sum(
                1 for cd in cert_details
                if cd["supplier_id"] == sid
                and cd.get("status") == CertificationStatus.EXPIRED.value
            )
            expired_penalty = expired_certs * 10.0

            # Composite score (deterministic calculation)
            composite = (
                country_score * 0.35
                + commodity_score * 0.30
                + (50.0 - cert_adjustment + expired_penalty) * 0.35
            )
            composite = round(min(100.0, max(0.0, composite)), 2)

            # Risk level
            if composite >= 70.0:
                risk_level = RiskLevel.HIGH
            elif composite >= 30.0:
                risk_level = RiskLevel.STANDARD
            else:
                risk_level = RiskLevel.LOW

            # DD type eligibility
            if country in LOW_RISK_COUNTRIES and composite < 30.0:
                dd_type = DDType.SIMPLIFIED
                simplified_eligible.append(sid)
            else:
                dd_type = DDType.STANDARD

            # Detect risk change
            score_change = composite - previous_score
            if abs(score_change) >= risk_threshold:
                direction = "increased" if score_change > 0 else "decreased"
                risk_changes.append({
                    "supplier_id": sid,
                    "supplier_name": supplier.get("supplier_name", ""),
                    "previous_score": previous_score,
                    "new_score": composite,
                    "change": round(score_change, 2),
                    "direction": direction,
                    "new_risk_level": risk_level.value,
                })
                warnings.append(
                    f"Supplier '{supplier.get('supplier_name', sid)}': "
                    f"risk {direction} by {abs(score_change):.1f} points "
                    f"({previous_score:.1f} -> {composite:.1f})"
                )

            # Track newly high-risk suppliers
            previous_level = supplier.get("previous_risk_level", "standard")
            if risk_level == RiskLevel.HIGH and previous_level != "high":
                new_high_risk.append(sid)

            supplier["current_risk"] = {
                "composite_score": composite,
                "risk_level": risk_level.value,
                "dd_type": dd_type.value,
                "country_score": country_score,
                "commodity_score": commodity_score,
            }

            scored_suppliers.append(supplier)

        context.state["scored_suppliers"] = scored_suppliers
        context.state["suppliers_reviewed"] = len(scored_suppliers)
        context.state["risk_changes_flagged"] = len(risk_changes)
        context.state["new_high_risk_suppliers"] = len(new_high_risk)
        context.state["simplified_dd_eligible"] = len(simplified_eligible)

        outputs["suppliers_scored"] = len(scored_suppliers)
        outputs["risk_changes_detected"] = len(risk_changes)
        outputs["risk_changes"] = risk_changes
        outputs["new_high_risk_suppliers"] = new_high_risk
        outputs["simplified_dd_eligible"] = len(simplified_eligible)
        outputs["risk_distribution"] = {
            "high": sum(
                1 for s in scored_suppliers
                if s.get("current_risk", {}).get("risk_level") == "high"
            ),
            "standard": sum(
                1 for s in scored_suppliers
                if s.get("current_risk", {}).get("risk_level") == "standard"
            ),
            "low": sum(
                1 for s in scored_suppliers
                if s.get("current_risk", {}).get("risk_level") == "low"
            ),
        }

        if new_high_risk:
            warnings.append(
                f"{len(new_high_risk)} supplier(s) newly classified as HIGH risk. "
                "Immediate attention required."
            )

        self.logger.info(
            "Phase 2 complete: %d scored, %d risk changes, %d new high-risk",
            len(scored_suppliers), len(risk_changes), len(new_high_risk),
        )

        provenance = self._hash({
            "phase": phase_name,
            "scored": len(scored_suppliers),
            "changes": len(risk_changes),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Compliance Reporting
    # -------------------------------------------------------------------------

    async def _phase_3_compliance_reporting(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Generate quarterly compliance report, update compliance dashboard,
        identify upcoming DDS deadlines, and produce executive summary.

        Uses:
            - EUDR-030 (Documentation Generator)
            - GL-EUDR-APP Dashboard API
        """
        phase_name = "compliance_reporting"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        quarter = context.state.get("quarter", "")
        year = context.state.get("year", 0)
        scored_suppliers = context.state.get("scored_suppliers", [])
        risk_changes = context.state.get("risk_changes_flagged", 0)
        certs_expired = context.state.get("certifications_expired", 0)
        certs_expiring = context.state.get("certifications_expiring", 0)
        new_high_risk = context.state.get("new_high_risk_suppliers", 0)

        # Calculate overall compliance score
        compliance_score = self._calculate_compliance_score(
            scored_suppliers, certs_expired, certs_expiring,
        )

        context.state["compliance_score"] = compliance_score

        # Identify upcoming DDS deadlines
        upcoming_deadlines = await self._fetch_upcoming_deadlines(quarter)
        context.state["upcoming_dds_deadlines"] = len(upcoming_deadlines)

        # Generate executive summary
        summary = {
            "quarter": quarter,
            "year": year,
            "generated_at": datetime.utcnow().isoformat(),
            "compliance_score": compliance_score,
            "total_suppliers": len(scored_suppliers),
            "certifications_status": {
                "total_checked": context.state.get("certifications_checked", 0),
                "valid": (
                    context.state.get("certifications_checked", 0)
                    - certs_expired - certs_expiring
                ),
                "expiring_soon": certs_expiring,
                "expired": certs_expired,
            },
            "risk_summary": {
                "changes_flagged": risk_changes,
                "new_high_risk": new_high_risk,
                "simplified_dd_eligible": context.state.get(
                    "simplified_dd_eligible", 0
                ),
            },
            "upcoming_deadlines": len(upcoming_deadlines),
            "recommendations": [],
        }

        # Generate recommendations
        recommendations: List[str] = []
        if certs_expired > 0:
            recommendations.append(
                f"Renew {certs_expired} expired certification(s) immediately "
                "to maintain compliance posture."
            )
        if certs_expiring > 0:
            recommendations.append(
                f"Plan renewal for {certs_expiring} certification(s) expiring "
                "within the next 90 days."
            )
        if new_high_risk > 0:
            recommendations.append(
                f"Review {new_high_risk} newly high-risk supplier(s) and "
                "implement enhanced due diligence measures."
            )
        if compliance_score < 70.0:
            recommendations.append(
                f"Compliance score ({compliance_score:.1f}%) is below target. "
                "Prioritize data quality improvements and certification renewals."
            )
        if len(upcoming_deadlines) > 0:
            recommendations.append(
                f"{len(upcoming_deadlines)} DDS deadline(s) approaching. "
                "Ensure all due diligence statements are prepared in time."
            )

        summary["recommendations"] = recommendations

        # Generate dashboard update payload
        dashboard_update = {
            "quarter": quarter,
            "compliance_score": compliance_score,
            "suppliers_reviewed": len(scored_suppliers),
            "risk_changes": risk_changes,
            "cert_alerts": certs_expired + certs_expiring,
            "updated_at": datetime.utcnow().isoformat(),
        }

        outputs["compliance_score"] = compliance_score
        outputs["executive_summary"] = summary
        outputs["recommendations_count"] = len(recommendations)
        outputs["recommendations"] = recommendations
        outputs["upcoming_deadlines"] = len(upcoming_deadlines)
        outputs["dashboard_updated"] = True
        outputs["report_generated_at"] = datetime.utcnow().isoformat()

        for rec in recommendations:
            warnings.append(f"Recommendation: {rec}")

        self.logger.info(
            "Phase 3 complete: compliance_score=%.1f, %d recommendations",
            compliance_score, len(recommendations),
        )

        provenance = self._hash({
            "phase": phase_name,
            "compliance_score": compliance_score,
            "quarter": quarter,
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

    def _calculate_compliance_score(
        self,
        suppliers: List[Dict[str, Any]],
        expired_certs: int,
        expiring_certs: int,
    ) -> float:
        """
        Calculate overall compliance score (0-100).

        Scoring:
            - Start at 100
            - Subtract 3 per expired certification
            - Subtract 1 per expiring certification
            - Subtract 2 per high-risk supplier without mitigation
            - Minimum 0
        """
        score = 100.0

        score -= expired_certs * 3.0
        score -= expiring_certs * 1.0

        high_risk = sum(
            1 for s in suppliers
            if s.get("current_risk", {}).get("risk_level") == "high"
        )
        score -= high_risk * 2.0

        return round(max(0.0, min(100.0, score)), 2)

    def _generate_sample_supplier_data(self) -> List[Dict[str, Any]]:
        """Generate sample supplier data for demonstration."""
        return [
            {
                "supplier_id": "SUP-SAMPLE-001",
                "supplier_name": "Sample Supplier",
                "country_code": "BR",
                "commodity": "coffee",
                "previous_risk_score": 50.0,
                "previous_risk_level": "standard",
                "certifications": [],
            },
        ]

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _fetch_supplier_data(
        self, supplier_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch current supplier data from the system."""
        await asyncio.sleep(0)
        return []

    async def _fetch_country_risk_updates(self) -> List[Dict[str, Any]]:
        """Fetch updated country risk benchmarking data."""
        await asyncio.sleep(0)
        return []

    async def _import_new_transactions(self, quarter: str) -> int:
        """Import new transactions for the quarter. Returns count."""
        await asyncio.sleep(0)
        return 0

    async def _fetch_upcoming_deadlines(
        self, quarter: str
    ) -> List[Dict[str, Any]]:
        """Fetch upcoming DDS deadlines."""
        await asyncio.sleep(0)
        return []

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
