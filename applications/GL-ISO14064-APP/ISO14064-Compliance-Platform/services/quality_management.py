"""
Quality Management -- ISO 14064-1:2018 Clause 6/7 Implementation

Implements the quality management system for GHG inventory data quality
assurance and control:

  - Quality management plan CRUD with procedures
  - Data quality assessment per source (activity data + emission factor)
  - Composite quality scoring with tier inference
  - Corrective action lifecycle tracking (finding -> action -> resolution)
  - Internal audit scheduling
  - Document version control (EFs, procedures)
  - Calibration records for measurement equipment
  - Overall inventory quality score

Reference: ISO 14064-1:2018 Clauses 6 and 7.

Example:
    >>> mgr = QualityManager(config)
    >>> plan = mgr.create_plan("org-1")
    >>> mgr.add_procedure(plan.id, "Data Collection Procedure")
    >>> score = mgr.calculate_quality_score(plan.id)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ActionCategory,
    ActionStatus,
    DataQualityTier,
    ISO14064AppConfig,
    ISOCategory,
)
from .models import (
    CorrectiveAction,
    DataQualityScore,
    QualityManagementPlan,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


class QualityManager:
    """
    ISO 14064-1 Clauses 6/7 quality management system.

    Manages quality plans, procedures, data quality assessments,
    corrective actions, and document control for GHG inventories.
    """

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
    ) -> None:
        """
        Initialize QualityManager.

        Args:
            config: Application configuration.
        """
        self.config = config or ISO14064AppConfig()
        self._plans: Dict[str, QualityManagementPlan] = {}
        # Track per-source assessments separately (keyed by plan_id)
        self._source_assessments: Dict[str, List[Dict[str, Any]]] = {}
        # Calibration records (keyed by plan_id)
        self._calibrations: Dict[str, List[Dict[str, Any]]] = {}
        # Document versions (keyed by plan_id)
        self._documents: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("QualityManager initialized")

    # ------------------------------------------------------------------
    # Plan CRUD
    # ------------------------------------------------------------------

    def create_plan(
        self,
        org_id: str,
        review_frequency: str = "annual",
        data_quality_objectives: Optional[Dict[str, str]] = None,
    ) -> QualityManagementPlan:
        """
        Create a new quality management plan.

        Args:
            org_id: Organization ID.
            review_frequency: Frequency of management review.
            data_quality_objectives: Quality objectives per ISO category.

        Returns:
            Created QualityManagementPlan.
        """
        plan = QualityManagementPlan(
            org_id=org_id,
            review_frequency=review_frequency,
            data_quality_objectives=data_quality_objectives or {},
        )
        self._plans[plan.id] = plan

        logger.info("Created quality plan for org %s (id=%s)", org_id, plan.id)
        return plan

    def get_plan(self, plan_id: str) -> Optional[QualityManagementPlan]:
        """Retrieve a quality plan by ID."""
        return self._plans.get(plan_id)

    def get_plans_for_org(self, org_id: str) -> List[QualityManagementPlan]:
        """Get all quality plans for an organization."""
        return [p for p in self._plans.values() if p.org_id == org_id]

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a quality plan."""
        if plan_id in self._plans:
            del self._plans[plan_id]
            logger.info("Deleted quality plan %s", plan_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Procedures
    # ------------------------------------------------------------------

    def add_procedure(
        self,
        plan_id: str,
        procedure_text: str,
    ) -> QualityManagementPlan:
        """
        Add a procedure to a quality plan.

        Args:
            plan_id: Quality plan ID.
            procedure_text: Description of the procedure.

        Returns:
            Updated QualityManagementPlan.
        """
        plan = self._get_plan_or_raise(plan_id)
        plan.procedures.append(procedure_text)
        plan.updated_at = _now()

        logger.info("Added procedure to plan %s: %s", plan_id, procedure_text[:60])
        return plan

    def get_procedures(self, plan_id: str) -> List[str]:
        """Get all procedures from a plan."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return []
        return plan.procedures

    # ------------------------------------------------------------------
    # Responsibilities
    # ------------------------------------------------------------------

    def set_responsibility(
        self,
        plan_id: str,
        role: str,
        responsibility: str,
    ) -> QualityManagementPlan:
        """
        Set a role-responsibility mapping.

        Args:
            plan_id: Quality plan ID.
            role: Role name (e.g., "Data Manager").
            responsibility: Responsibility description.

        Returns:
            Updated QualityManagementPlan.
        """
        plan = self._get_plan_or_raise(plan_id)
        plan.responsibilities[role] = responsibility
        plan.updated_at = _now()

        logger.info("Set responsibility for '%s' in plan %s", role, plan_id)
        return plan

    # ------------------------------------------------------------------
    # Audit Schedule
    # ------------------------------------------------------------------

    def add_audit_entry(
        self,
        plan_id: str,
        audit_type: str,
        scheduled_date: str,
        scope: str = "",
        assigned_to: str = "",
    ) -> QualityManagementPlan:
        """
        Add an internal audit schedule entry.

        Args:
            plan_id: Quality plan ID.
            audit_type: Type of audit (e.g., "data_review", "process_audit").
            scheduled_date: ISO format date.
            scope: Audit scope description.
            assigned_to: Person responsible.

        Returns:
            Updated QualityManagementPlan.
        """
        plan = self._get_plan_or_raise(plan_id)

        entry: Dict[str, Any] = {
            "id": _new_id(),
            "audit_type": audit_type,
            "scheduled_date": scheduled_date,
            "scope": scope,
            "assigned_to": assigned_to,
            "status": "scheduled",
            "completed_date": None,
            "findings_count": 0,
        }
        plan.audit_schedule.append(entry)
        plan.updated_at = _now()

        logger.info("Added audit entry to plan %s: %s on %s", plan_id, audit_type, scheduled_date)
        return plan

    def get_audit_schedule(self, plan_id: str) -> List[Dict[str, Any]]:
        """Get all audit schedule entries for a plan."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return []
        return plan.audit_schedule

    # ------------------------------------------------------------------
    # Data Quality Assessment per Source
    # ------------------------------------------------------------------

    def assess_data_quality(
        self,
        plan_id: str,
        source_name: str,
        activity_data_quality: int,
        emission_factor_quality: int,
        category: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Assess data quality for a single emission source.

        Composite quality = 0.60 * activity_data + 0.40 * emission_factor.

        Args:
            plan_id: Quality plan ID.
            source_name: Data source name.
            activity_data_quality: Score 0-100.
            emission_factor_quality: Score 0-100.
            category: Optional ISO category value.
            notes: Quality notes.

        Returns:
            Dict with composite score and tier.
        """
        self._get_plan_or_raise(plan_id)

        composite = (
            Decimal("0.60") * Decimal(str(activity_data_quality))
            + Decimal("0.40") * Decimal(str(emission_factor_quality))
        ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        tier = self._infer_tier(composite)

        assessment = {
            "id": _new_id(),
            "plan_id": plan_id,
            "source_name": source_name,
            "activity_data_quality": activity_data_quality,
            "emission_factor_quality": emission_factor_quality,
            "composite_quality": str(composite),
            "tier": tier.value,
            "category": category,
            "notes": notes,
            "assessed_at": _now().isoformat(),
        }

        if plan_id not in self._source_assessments:
            self._source_assessments[plan_id] = []
        self._source_assessments[plan_id].append(assessment)

        logger.info(
            "Assessed quality for '%s': AD=%d, EF=%d, composite=%.1f, tier=%s",
            source_name, activity_data_quality, emission_factor_quality,
            composite, tier.value,
        )
        return assessment

    def get_assessments(
        self,
        plan_id: str,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all quality assessments, optionally filtered."""
        assessments = self._source_assessments.get(plan_id, [])
        if category:
            return [a for a in assessments if a.get("category") == category]
        return assessments

    def get_overall_quality_score(self, plan_id: str) -> DataQualityScore:
        """Compute overall quality score across all assessed sources."""
        assessments = self._source_assessments.get(plan_id, [])
        if not assessments:
            return DataQualityScore()

        ad_scores = [a["activity_data_quality"] for a in assessments]
        ef_scores = [a["emission_factor_quality"] for a in assessments]
        composites = [Decimal(a["composite_quality"]) for a in assessments]

        avg_ad = Decimal(str(round(sum(ad_scores) / len(ad_scores), 1)))
        avg_ef = Decimal(str(round(sum(ef_scores) / len(ef_scores), 1)))
        avg_composite = (sum(composites) / Decimal(str(len(composites)))).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP,
        )

        tier = self._infer_tier(avg_composite)

        # By category breakdown
        by_category: Dict[str, Decimal] = {}
        cat_groups: Dict[str, List[Decimal]] = {}
        for a in assessments:
            cat = a.get("category")
            if cat:
                if cat not in cat_groups:
                    cat_groups[cat] = []
                cat_groups[cat].append(Decimal(a["composite_quality"]))
        for cat, scores in cat_groups.items():
            by_category[cat] = (sum(scores) / Decimal(str(len(scores)))).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP,
            )

        return DataQualityScore(
            activity_data_score=avg_ad,
            emission_factor_score=avg_ef,
            composite_score=avg_composite,
            tier=tier,
            completeness_pct=Decimal("100") if assessments else Decimal("0"),
            by_category=by_category,
        )

    # ------------------------------------------------------------------
    # Corrective Actions
    # ------------------------------------------------------------------

    def add_corrective_action(
        self,
        plan_id: str,
        description: str,
        action_type: ActionCategory = ActionCategory.DATA_IMPROVEMENT,
        finding_id: Optional[str] = None,
        assigned_to: Optional[str] = None,
        deadline: Optional[date] = None,
    ) -> CorrectiveAction:
        """
        Add a corrective action to a quality plan.

        Args:
            plan_id: Quality plan ID.
            description: Action description.
            action_type: Category of corrective action.
            finding_id: Related finding ID.
            assigned_to: Person responsible.
            deadline: Target completion date.

        Returns:
            Created CorrectiveAction.
        """
        plan = self._get_plan_or_raise(plan_id)

        ca = CorrectiveAction(
            description=description,
            action_type=action_type,
            finding_id=finding_id,
            assigned_to=assigned_to,
            deadline=deadline,
            status=ActionStatus.PLANNED,
        )

        plan.corrective_actions.append(ca)
        plan.updated_at = _now()

        logger.info("Added corrective action to plan %s: '%s'", plan_id, description[:50])
        return ca

    def resolve_corrective_action(
        self,
        plan_id: str,
        action_id: str,
        resolution_notes: str,
    ) -> CorrectiveAction:
        """Resolve a corrective action."""
        plan = self._get_plan_or_raise(plan_id)

        for ca in plan.corrective_actions:
            if ca.id == action_id:
                ca.status = ActionStatus.COMPLETED
                ca.resolution_notes = resolution_notes
                ca.completed_at = _now()
                plan.updated_at = _now()
                logger.info("Resolved corrective action %s", action_id)
                return ca

        raise ValueError(f"Corrective action not found: {action_id}")

    def get_corrective_actions(
        self,
        plan_id: str,
        status: Optional[ActionStatus] = None,
    ) -> List[CorrectiveAction]:
        """Get corrective actions, optionally filtered by status."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return []

        actions = plan.corrective_actions
        if status:
            return [a for a in actions if a.status == status]
        return actions

    # ------------------------------------------------------------------
    # Calibration Records
    # ------------------------------------------------------------------

    def add_calibration_record(
        self,
        plan_id: str,
        equipment_name: str,
        calibration_date: str,
        next_calibration_date: Optional[str] = None,
        certificate_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a calibration record for measurement equipment."""
        self._get_plan_or_raise(plan_id)

        record = {
            "id": _new_id(),
            "plan_id": plan_id,
            "equipment_name": equipment_name,
            "calibration_date": calibration_date,
            "next_calibration_date": next_calibration_date,
            "certificate_ref": certificate_ref,
            "status": "valid",
        }

        if plan_id not in self._calibrations:
            self._calibrations[plan_id] = []
        self._calibrations[plan_id].append(record)

        logger.info("Added calibration record for '%s'", equipment_name)
        return record

    def get_calibration_records(self, plan_id: str) -> List[Dict[str, Any]]:
        """Get all calibration records for a plan."""
        return self._calibrations.get(plan_id, [])

    # ------------------------------------------------------------------
    # Document Version Control
    # ------------------------------------------------------------------

    def add_document_version(
        self,
        plan_id: str,
        document_name: str,
        document_type: str,
        version: str,
        effective_date: str,
        change_description: str = "",
    ) -> Dict[str, Any]:
        """Register a new document version."""
        self._get_plan_or_raise(plan_id)

        # Supersede previous
        for doc in self._documents.get(plan_id, []):
            if doc["document_name"] == document_name and doc.get("superseded_date") is None:
                doc["superseded_date"] = effective_date

        doc_version = {
            "id": _new_id(),
            "plan_id": plan_id,
            "document_name": document_name,
            "document_type": document_type,
            "version": version,
            "effective_date": effective_date,
            "superseded_date": None,
            "change_description": change_description,
        }

        if plan_id not in self._documents:
            self._documents[plan_id] = []
        self._documents[plan_id].append(doc_version)

        logger.info("Added document version: %s v%s", document_name, version)
        return doc_version

    def get_document_versions(
        self,
        plan_id: str,
        current_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get document versions."""
        docs = self._documents.get(plan_id, [])
        if current_only:
            return [d for d in docs if d.get("superseded_date") is None]
        return docs

    # ------------------------------------------------------------------
    # Quality Summary
    # ------------------------------------------------------------------

    def generate_quality_summary(
        self,
        plan_id: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive quality summary."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"message": "No quality plan found", "plan_id": plan_id}

        assessments = self._source_assessments.get(plan_id, [])
        open_actions = [
            a for a in plan.corrective_actions
            if a.status in (ActionStatus.PLANNED, ActionStatus.IN_PROGRESS)
        ]
        resolved_actions = [
            a for a in plan.corrective_actions
            if a.status == ActionStatus.COMPLETED
        ]
        calibrations = self._calibrations.get(plan_id, [])
        documents = self._documents.get(plan_id, [])

        overall_score = self.get_overall_quality_score(plan_id)

        return {
            "plan_id": plan_id,
            "org_id": plan.org_id,
            "overall_quality_score": str(overall_score.composite_score),
            "overall_tier": overall_score.tier.value,
            "procedures_count": len(plan.procedures),
            "quality_assessments": len(assessments),
            "corrective_actions": {
                "total": len(plan.corrective_actions),
                "open": len(open_actions),
                "resolved": len(resolved_actions),
            },
            "calibration_records": len(calibrations),
            "document_versions": {
                "total": len(documents),
                "current": len([d for d in documents if d.get("superseded_date") is None]),
            },
            "audit_schedule_entries": len(plan.audit_schedule),
            "review_frequency": plan.review_frequency,
            "last_review_date": plan.last_review_date.isoformat() if plan.last_review_date else None,
            "next_review_date": plan.next_review_date.isoformat() if plan.next_review_date else None,
        }

    # ------------------------------------------------------------------
    # Review Management
    # ------------------------------------------------------------------

    def set_review_dates(
        self,
        plan_id: str,
        last_review_date: Optional[date] = None,
        next_review_date: Optional[date] = None,
    ) -> QualityManagementPlan:
        """Set review dates for a quality plan."""
        plan = self._get_plan_or_raise(plan_id)
        if last_review_date is not None:
            plan.last_review_date = last_review_date
        if next_review_date is not None:
            plan.next_review_date = next_review_date
        plan.updated_at = _now()
        logger.info("Updated review dates for plan %s", plan_id)
        return plan

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_plan_or_raise(self, plan_id: str) -> QualityManagementPlan:
        """Retrieve plan or raise ValueError."""
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Quality plan not found: {plan_id}")
        return plan

    @staticmethod
    def _infer_tier(composite_score: Decimal) -> DataQualityTier:
        """Infer data quality tier from composite score."""
        if composite_score >= Decimal("90"):
            return DataQualityTier.TIER_4
        if composite_score >= Decimal("75"):
            return DataQualityTier.TIER_3
        if composite_score >= Decimal("50"):
            return DataQualityTier.TIER_2
        return DataQualityTier.TIER_1
