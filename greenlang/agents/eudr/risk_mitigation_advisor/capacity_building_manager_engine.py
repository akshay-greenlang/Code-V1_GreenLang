# -*- coding: utf-8 -*-
"""
Capacity Building Manager Engine - AGENT-EUDR-025

Designs and manages supplier development programs including training
curricula, technical assistance packages, resource allocation, progress
tracking, and competency assessments. Supports 4 capacity building tiers
(Awareness, Basic, Advanced, Leadership) with commodity-specific and
region-specific content modules across all 7 EUDR commodities.

Core capabilities:
    - 4-tier progressive capacity building framework
    - 7 commodity-specific curricula (22 modules each = 154 total)
    - Individual supplier progress tracking through tiers
    - Competency assessment at each tier gate
    - Technical assistance resource allocation
    - Capacity building scorecards with risk score correlation
    - Mobile-friendly content for low-connectivity environments
    - Integration with EUDR-017 Supplier Risk Scorer
    - Batch enrollment for portfolio-level programs
    - Progress dashboards with tier distribution charts

Tier Framework:
    Tier 1 - Awareness (4 modules per commodity):
        EUDR overview, deforestation-free basics, GPS introduction, DDS requirements
    Tier 2 - Basic (8 modules per commodity):
        Data collection, GPS capture, record keeping, self-assessment,
        environmental monitoring, traceability, risk self-assessment, CAP
    Tier 3 - Advanced (6 modules per commodity):
        Advanced practices, agroforestry, biodiversity, climate-smart,
        community engagement, certification readiness
    Tier 4 - Leadership (4 modules per commodity):
        Peer mentoring, CBNRM, landscape sustainability, certification maintenance

PRD: PRD-AGENT-EUDR-025, Feature 3: Supplier Capacity Building Manager
Agent ID: GL-EUDR-RMA-025
Status: Production Ready

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from greenlang.agents.eudr.risk_mitigation_advisor.config import (
    RiskMitigationAdvisorConfig,
    get_config,
)
from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    CapacityTier,
    EnrollmentStatus,
    EUDRCommodity,
    CapacityBuildingEnrollment,
    EnrollSupplierRequest,
    EnrollSupplierResponse,
)
from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
    ProvenanceTracker,
    get_tracker,
)

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.metrics import (
        record_capacity_enrollment,
    )
except ImportError:
    record_capacity_enrollment = None


# ---------------------------------------------------------------------------
# Training module definitions per tier
# ---------------------------------------------------------------------------

TIER_MODULES: Dict[str, Dict[str, List[str]]] = {
    "tier_1_awareness": {
        "cattle": [
            "EUDR Overview for Cattle Producers",
            "Deforestation-Free Cattle Production Basics",
            "GPS Plot Mapping Introduction",
            "Due Diligence Statement Requirements",
        ],
        "cocoa": [
            "EUDR Overview for Cocoa Farmers",
            "Deforestation-Free Cocoa Fundamentals",
            "Farm Boundary GPS Recording",
            "EUDR Documentation Requirements",
        ],
        "coffee": [
            "EUDR Overview for Coffee Producers",
            "Sustainable Coffee Production Basics",
            "Plot Geolocation Mapping",
            "Compliance Documentation Introduction",
        ],
        "palm_oil": [
            "EUDR Overview for Palm Oil Producers",
            "Zero-Deforestation Palm Oil Basics",
            "Plantation GPS Boundary Recording",
            "RSPO and EUDR Alignment",
        ],
        "rubber": [
            "EUDR Overview for Rubber Producers",
            "Deforestation-Free Rubber Basics",
            "Plot GPS Mapping Fundamentals",
            "Compliance Record Keeping",
        ],
        "soya": [
            "EUDR Overview for Soya Producers",
            "Sustainable Soya Production Basics",
            "Field Boundary GPS Recording",
            "Traceability Documentation",
        ],
        "wood": [
            "EUDR Overview for Timber Operators",
            "Legal Timber Harvesting Basics",
            "Forest Concession GPS Mapping",
            "Chain of Custody Introduction",
        ],
    },
    "tier_2_basic": {
        commodity: [
            f"Data Collection Methods for {commodity.replace('_', ' ').title()}",
            f"GPS Coordinate Capture for {commodity.replace('_', ' ').title()} Plots",
            f"Record Keeping and Documentation for {commodity.replace('_', ' ').title()}",
            f"Self-Assessment Checklist for {commodity.replace('_', ' ').title()}",
            f"Environmental Monitoring Basics for {commodity.replace('_', ' ').title()}",
            f"Supply Chain Traceability for {commodity.replace('_', ' ').title()}",
            f"Risk Self-Assessment for {commodity.replace('_', ' ').title()}",
            f"Corrective Action Planning for {commodity.replace('_', ' ').title()}",
        ] for commodity in ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"]
    },
    "tier_3_advanced": {
        commodity: [
            f"Advanced Sustainable {commodity.replace('_', ' ').title()} Practices",
            f"Agroforestry Integration for {commodity.replace('_', ' ').title()}",
            f"Biodiversity Conservation in {commodity.replace('_', ' ').title()} Production",
            f"Climate-Smart {commodity.replace('_', ' ').title()} Agriculture",
            f"Community Engagement for {commodity.replace('_', ' ').title()} Producers",
            f"Certification Readiness for {commodity.replace('_', ' ').title()}",
        ] for commodity in ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"]
    },
    "tier_4_leadership": {
        commodity: [
            f"Peer Mentoring Program for {commodity.replace('_', ' ').title()} Leaders",
            f"Community-Based Natural Resource Management for {commodity.replace('_', ' ').title()}",
            f"Landscape-Level {commodity.replace('_', ' ').title()} Sustainability",
            f"Certification Achievement and Maintenance for {commodity.replace('_', ' ').title()}",
        ] for commodity in ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"]
    },
}


# Competency assessment criteria per tier gate
TIER_GATE_CRITERIA: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Awareness Gate",
        "passing_score": Decimal("60"),
        "criteria": [
            "Demonstrates understanding of EUDR requirements",
            "Can identify deforestation-free production principles",
            "Understands GPS data collection purpose",
            "Knows DDS documentation requirements",
        ],
        "assessment_type": "quiz",
        "max_attempts": 3,
    },
    2: {
        "name": "Basic Competency Gate",
        "passing_score": Decimal("70"),
        "criteria": [
            "Can independently collect GPS coordinates",
            "Maintains accurate records per requirements",
            "Completes self-assessment checklists",
            "Understands supply chain traceability",
            "Can develop corrective action plans",
        ],
        "assessment_type": "practical_assessment",
        "max_attempts": 2,
    },
    3: {
        "name": "Advanced Competency Gate",
        "passing_score": Decimal("75"),
        "criteria": [
            "Implements advanced sustainable practices",
            "Integrates agroforestry or conservation measures",
            "Actively engages community stakeholders",
            "Demonstrates climate-smart approaches",
            "Meets pre-certification requirements",
        ],
        "assessment_type": "portfolio_review",
        "max_attempts": 2,
    },
    4: {
        "name": "Leadership Gate",
        "passing_score": Decimal("80"),
        "criteria": [
            "Successfully mentors peer producers",
            "Manages community-based resource programs",
            "Contributes to landscape-level sustainability",
            "Achieves or maintains third-party certification",
        ],
        "assessment_type": "panel_evaluation",
        "max_attempts": 1,
    },
}


class CapacityBuildingManagerEngine:
    """Supplier capacity building management engine.

    Manages 4-tier progressive capacity building programs with
    commodity-specific training modules, competency assessments,
    and risk score correlation tracking.

    Attributes:
        config: Agent configuration.
        provenance: Provenance tracker.
        _db_pool: PostgreSQL connection pool.
        _redis_client: Redis client.
        _enrollments: In-memory enrollment store.
        _assessments: Assessment results cache.

    Example:
        >>> engine = CapacityBuildingManagerEngine(config=get_config())
        >>> response = await engine.enroll_supplier(request)
        >>> assert response.enrollment.status == EnrollmentStatus.ACTIVE
    """

    def __init__(
        self,
        config: Optional[RiskMitigationAdvisorConfig] = None,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize CapacityBuildingManagerEngine."""
        self.config = config or get_config()
        self.provenance = provenance or get_tracker()
        self._db_pool = db_pool
        self._redis_client = redis_client
        self._enrollments: Dict[str, CapacityBuildingEnrollment] = {}
        self._assessments: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            f"CapacityBuildingManagerEngine initialized: "
            f"tiers={self.config.capacity_tiers}, "
            f"modules_per_commodity={self.config.modules_per_commodity}"
        )

    async def enroll_supplier(
        self, request: EnrollSupplierRequest,
    ) -> EnrollSupplierResponse:
        """Enroll a supplier in a capacity building program.

        Args:
            request: Enrollment request with supplier, commodity, tier.

        Returns:
            EnrollSupplierResponse with enrollment details.
        """
        start = time.monotonic()
        enrollment_id = str(uuid.uuid4())
        program_id = f"cbp-{request.commodity}-{str(uuid.uuid4())[:8]}"

        tier_key = self._get_tier_key(request.initial_tier)
        commodity_modules = TIER_MODULES.get(tier_key, {}).get(
            request.commodity, []
        )
        total_modules = self.config.modules_per_commodity

        enrollment = CapacityBuildingEnrollment(
            enrollment_id=enrollment_id,
            supplier_id=request.supplier_id,
            program_id=program_id,
            commodity=request.commodity,
            current_tier=request.initial_tier,
            modules_completed=0,
            modules_total=total_modules,
            competency_scores={},
            enrolled_date=date.today(),
            target_completion_date=date.today() + timedelta(
                weeks=request.target_completion_weeks
            ),
            status=EnrollmentStatus.ACTIVE,
            risk_score_at_enrollment=Decimal("0"),
            current_risk_score=Decimal("0"),
        )

        # Store enrollment
        self._enrollments[enrollment_id] = enrollment

        self.provenance.record(
            entity_type="capacity_enrollment",
            action="create",
            entity_id=enrollment_id,
            actor="capacity_building_manager_engine",
            metadata={
                "supplier_id": request.supplier_id,
                "commodity": request.commodity,
                "tier": request.initial_tier,
                "modules": len(commodity_modules),
                "target_weeks": request.target_completion_weeks,
            },
        )

        elapsed_ms = Decimal(str(round(
            (time.monotonic() - start) * 1000, 2
        )))

        if record_capacity_enrollment is not None:
            record_capacity_enrollment(request.commodity, str(request.initial_tier))

        logger.info(
            f"Supplier enrolled: id={enrollment_id}, "
            f"supplier={request.supplier_id}, "
            f"commodity={request.commodity}, "
            f"tier={request.initial_tier}"
        )

        return EnrollSupplierResponse(
            enrollment=enrollment,
            modules_assigned=len(commodity_modules),
            processing_time_ms=elapsed_ms,
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "id": enrollment_id,
                    "supplier": request.supplier_id,
                    "commodity": request.commodity,
                    "tier": request.initial_tier,
                }, sort_keys=True).encode()
            ).hexdigest(),
        )

    def _get_tier_key(self, tier: int) -> str:
        """Get the TIER_MODULES key for a tier number.

        Args:
            tier: Tier number (1-4).

        Returns:
            Tier key string.
        """
        tier_map = {
            1: "tier_1_awareness",
            2: "tier_2_basic",
            3: "tier_3_advanced",
            4: "tier_4_leadership",
        }
        return tier_map.get(tier, "tier_1_awareness")

    def get_modules_for_tier(
        self, commodity: str, tier: int,
    ) -> List[str]:
        """Get training modules for a specific commodity and tier.

        Args:
            commodity: EUDR commodity.
            tier: Capacity building tier (1-4).

        Returns:
            List of module names.
        """
        tier_key = self._get_tier_key(tier)
        return TIER_MODULES.get(tier_key, {}).get(commodity, [])

    def get_all_modules_for_commodity(
        self, commodity: str,
    ) -> Dict[str, List[str]]:
        """Get all modules across all tiers for a commodity.

        Args:
            commodity: EUDR commodity.

        Returns:
            Dictionary of tier to module list.
        """
        result: Dict[str, List[str]] = {}
        for tier in range(1, 5):
            tier_key = self._get_tier_key(tier)
            modules = TIER_MODULES.get(tier_key, {}).get(commodity, [])
            result[f"tier_{tier}"] = modules
        return result

    def record_module_completion(
        self,
        enrollment_id: str,
        module_name: str,
        score: Optional[Decimal] = None,
        actor: str = "system",
    ) -> bool:
        """Record completion of a training module.

        Args:
            enrollment_id: Enrollment identifier.
            module_name: Name of completed module.
            score: Optional assessment score (0-100).
            actor: User recording the completion.

        Returns:
            True if recorded successfully.
        """
        enrollment = self._enrollments.get(enrollment_id)
        if enrollment is None:
            logger.warning(f"Enrollment '{enrollment_id}' not found")
            return False

        self.provenance.record(
            entity_type="capacity_enrollment",
            action="module_complete",
            entity_id=enrollment_id,
            actor=actor,
            metadata={
                "module_name": module_name,
                "score": str(score) if score else None,
                "modules_completed": enrollment.modules_completed + 1,
            },
        )

        logger.info(
            f"Module completed: enrollment={enrollment_id}, "
            f"module={module_name}, score={score}"
        )

        return True

    def assess_tier_gate(
        self,
        enrollment_id: str,
        tier: int,
        scores: Dict[str, Decimal],
        assessor: str = "system",
    ) -> Dict[str, Any]:
        """Assess a supplier at a tier gate.

        Evaluates whether the supplier has achieved the competency
        required to advance to the next tier.

        Args:
            enrollment_id: Enrollment identifier.
            tier: Tier being assessed (1-4).
            scores: Scores per criterion.
            assessor: Name of the assessor.

        Returns:
            Assessment result with pass/fail and recommendation.
        """
        enrollment = self._enrollments.get(enrollment_id)
        if enrollment is None:
            return {"error": f"Enrollment '{enrollment_id}' not found"}

        gate = TIER_GATE_CRITERIA.get(tier)
        if gate is None:
            return {"error": f"No gate criteria defined for tier {tier}"}

        passing_score = gate["passing_score"]

        # Calculate average score
        if not scores:
            avg_score = Decimal("0")
        else:
            avg_score = (
                sum(scores.values()) / Decimal(str(len(scores)))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        passed = avg_score >= passing_score

        # Store assessment
        assessment = {
            "assessment_id": str(uuid.uuid4()),
            "enrollment_id": enrollment_id,
            "tier": tier,
            "gate_name": gate["name"],
            "scores": {k: str(v) for k, v in scores.items()},
            "average_score": str(avg_score),
            "passing_score": str(passing_score),
            "passed": passed,
            "assessor": assessor,
            "assessment_type": gate["assessment_type"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if enrollment_id not in self._assessments:
            self._assessments[enrollment_id] = []
        self._assessments[enrollment_id].append(assessment)

        self.provenance.record(
            entity_type="capacity_enrollment",
            action="tier_gate_assessment",
            entity_id=enrollment_id,
            actor=assessor,
            metadata={
                "tier": tier,
                "average_score": str(avg_score),
                "passed": passed,
            },
        )

        recommendation = ""
        if passed:
            if tier < 4:
                recommendation = f"Advance to Tier {tier + 1}"
            else:
                recommendation = "Supplier achieved Leadership tier. Consider as peer mentor."
        else:
            deficit = passing_score - avg_score
            recommendation = (
                f"Additional training required. Score deficit: {deficit} points. "
                f"Focus on lowest-scoring criteria."
            )

        return {
            "enrollment_id": enrollment_id,
            "tier_assessed": tier,
            "gate_name": gate["name"],
            "average_score": str(avg_score),
            "passing_score": str(passing_score),
            "result": "PASS" if passed else "FAIL",
            "recommendation": recommendation,
            "criteria_count": len(gate["criteria"]),
            "assessment_type": gate["assessment_type"],
        }

    def get_enrollment_scorecard(
        self, enrollment_id: str,
    ) -> Dict[str, Any]:
        """Generate capacity building scorecard for an enrollment.

        Args:
            enrollment_id: Enrollment identifier.

        Returns:
            Scorecard with progress, assessments, and risk correlation.
        """
        enrollment = self._enrollments.get(enrollment_id)
        if enrollment is None:
            return {"error": f"Enrollment '{enrollment_id}' not found"}

        assessments = self._assessments.get(enrollment_id, [])

        # Calculate progress
        progress_pct = Decimal("0")
        if enrollment.modules_total > 0:
            progress_pct = (
                Decimal(str(enrollment.modules_completed))
                / Decimal(str(enrollment.modules_total))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Risk score correlation
        risk_change = Decimal("0")
        if enrollment.risk_score_at_enrollment > Decimal("0"):
            risk_change = (
                (enrollment.risk_score_at_enrollment - enrollment.current_risk_score)
                / enrollment.risk_score_at_enrollment
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Days in program
        days_enrolled = (date.today() - enrollment.enrolled_date).days
        days_remaining = max(
            0, (enrollment.target_completion_date - date.today()).days
        )

        return {
            "enrollment_id": enrollment_id,
            "supplier_id": enrollment.supplier_id,
            "commodity": enrollment.commodity,
            "current_tier": enrollment.current_tier,
            "status": enrollment.status.value,
            "progress_pct": str(progress_pct),
            "modules_completed": enrollment.modules_completed,
            "modules_total": enrollment.modules_total,
            "assessments_completed": len(assessments),
            "assessments_passed": sum(
                1 for a in assessments if a.get("passed", False)
            ),
            "risk_score_at_enrollment": str(enrollment.risk_score_at_enrollment),
            "current_risk_score": str(enrollment.current_risk_score),
            "risk_reduction_pct": str(risk_change),
            "days_enrolled": days_enrolled,
            "days_remaining": days_remaining,
            "on_track": days_remaining > 0 and progress_pct >= Decimal(str(
                max(0, (days_enrolled * 100) // max(1, days_enrolled + days_remaining))
            )),
        }

    def get_program_dashboard(self) -> Dict[str, Any]:
        """Generate program-level dashboard across all enrollments.

        Returns:
            Dashboard with tier distribution, commodity breakdown, and KPIs.
        """
        if not self._enrollments:
            return {
                "total_enrollments": 0,
                "status": "no_enrollments",
            }

        tier_distribution: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        commodity_distribution: Dict[str, int] = {}
        status_distribution: Dict[str, int] = {}
        total_modules_completed = 0

        for enrollment in self._enrollments.values():
            tier_distribution[enrollment.current_tier] = (
                tier_distribution.get(enrollment.current_tier, 0) + 1
            )
            commodity_distribution[enrollment.commodity] = (
                commodity_distribution.get(enrollment.commodity, 0) + 1
            )
            status_distribution[enrollment.status.value] = (
                status_distribution.get(enrollment.status.value, 0) + 1
            )
            total_modules_completed += enrollment.modules_completed

        return {
            "total_enrollments": len(self._enrollments),
            "active_enrollments": status_distribution.get("active", 0),
            "completed_enrollments": status_distribution.get("completed", 0),
            "tier_distribution": tier_distribution,
            "commodity_distribution": commodity_distribution,
            "status_distribution": status_distribution,
            "total_modules_completed": total_modules_completed,
            "avg_modules_per_supplier": round(
                total_modules_completed / max(1, len(self._enrollments)), 1
            ),
        }

    async def enroll_batch(
        self,
        requests: List[EnrollSupplierRequest],
    ) -> Dict[str, Any]:
        """Enroll multiple suppliers in capacity building programs.

        Args:
            requests: List of enrollment requests.

        Returns:
            Batch enrollment summary.
        """
        start = time.monotonic()
        results: List[EnrollSupplierResponse] = []
        errors: List[str] = []

        for req in requests:
            try:
                result = await self.enroll_supplier(req)
                results.append(result)
            except Exception as e:
                errors.append(f"{req.supplier_id}: {str(e)}")

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        return {
            "total_enrolled": len(results),
            "total_errors": len(errors),
            "errors": errors[:10],  # Limit error output
            "processing_time_ms": elapsed_ms,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        total_modules = sum(
            len(modules)
            for tier_data in TIER_MODULES.values()
            for modules in tier_data.values()
        )
        return {
            "status": "available",
            "tiers": self.config.capacity_tiers,
            "total_modules": total_modules,
            "commodities": 7,
            "enrollments": len(self._enrollments),
            "assessments": sum(
                len(v) for v in self._assessments.values()
            ),
            "gate_criteria": len(TIER_GATE_CRITERIA),
        }

    async def shutdown(self) -> None:
        """Shutdown engine."""
        self._enrollments.clear()
        self._assessments.clear()
        logger.info("CapacityBuildingManagerEngine shut down")
