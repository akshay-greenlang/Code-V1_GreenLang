# -*- coding: utf-8 -*-
"""
Action Generator Engine - AGENT-EUDR-035: Improvement Plan Creator

Generates SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
improvement actions from identified compliance gaps. Maps gap types to
action templates, estimates effort and cost, validates SMART criteria,
and produces structured action records ready for prioritization.

Zero-Hallucination:
    - Action templates are static lookup tables
    - Effort and cost estimates use deterministic multipliers
    - SMART validation is rule-based (no LLM)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import ImprovementPlanCreatorConfig, get_config
from .models import (
    AGENT_ID,
    ActionStatus,
    ActionType,
    ComplianceGap,
    GapSeverity,
    ImprovementAction,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)

# Action templates mapped to gap risk dimensions
_ACTION_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "risk_assessment": [
        {
            "title": "Enhance risk assessment methodology",
            "type": ActionType.PROCESS_CHANGE,
            "specific": "Update risk assessment to cover identified gaps",
            "measurable": "Risk assessment completeness score >= 90%",
            "achievable": "Internal risk team with methodology support",
            "relevant": "EUDR Article 10 requires comprehensive risk assessment",
            "effort_hours": Decimal("40"),
            "cost": Decimal("5000"),
        },
        {
            "title": "Implement automated risk scoring",
            "type": ActionType.TECHNOLOGY_UPGRADE,
            "specific": "Deploy automated risk scoring for flagged areas",
            "measurable": "Automated coverage >= 80% of risk dimensions",
            "achievable": "Technology team with vendor support",
            "relevant": "Ensures consistent and timely risk evaluation",
            "effort_hours": Decimal("80"),
            "cost": Decimal("15000"),
        },
    ],
    "country_risk": [
        {
            "title": "Strengthen country risk monitoring",
            "type": ActionType.MONITORING_ENHANCEMENT,
            "specific": "Add real-time country risk data feeds",
            "measurable": "Country risk data refresh interval <= 24 hours",
            "achievable": "Data integration team with vendor APIs",
            "relevant": "EUDR Article 10 requires country-level risk assessment",
            "effort_hours": Decimal("24"),
            "cost": Decimal("3000"),
        },
    ],
    "supplier_risk": [
        {
            "title": "Enhance supplier engagement program",
            "type": ActionType.SUPPLIER_ENGAGEMENT,
            "specific": "Implement supplier questionnaire and audit program",
            "measurable": "Supplier audit coverage >= 95%",
            "achievable": "Procurement team with compliance support",
            "relevant": "EUDR requires supply chain due diligence",
            "effort_hours": Decimal("60"),
            "cost": Decimal("10000"),
        },
    ],
    "deforestation_alert": [
        {
            "title": "Implement deforestation response protocol",
            "type": ActionType.CORRECTIVE,
            "specific": "Define and deploy deforestation alert response SOP",
            "measurable": "Alert response time <= 48 hours",
            "achievable": "Sustainability team with satellite data access",
            "relevant": "EUDR Article 3 prohibition on deforestation-linked products",
            "effort_hours": Decimal("30"),
            "cost": Decimal("4000"),
        },
    ],
    "legal_compliance": [
        {
            "title": "Update legal compliance framework",
            "type": ActionType.POLICY_UPDATE,
            "specific": "Review and update internal compliance policies for EUDR",
            "measurable": "All policies reviewed and updated",
            "achievable": "Legal team with regulatory support",
            "relevant": "EUDR Article 4 due diligence obligations",
            "effort_hours": Decimal("50"),
            "cost": Decimal("8000"),
        },
    ],
    "document_authentication": [
        {
            "title": "Strengthen document verification process",
            "type": ActionType.PROCESS_CHANGE,
            "specific": "Implement multi-factor document authentication",
            "measurable": "Document verification accuracy >= 99%",
            "achievable": "Compliance team with technology support",
            "relevant": "EUDR Article 14 DDS requirements",
            "effort_hours": Decimal("35"),
            "cost": Decimal("6000"),
        },
    ],
    "satellite_monitoring": [
        {
            "title": "Expand satellite monitoring coverage",
            "type": ActionType.MONITORING_ENHANCEMENT,
            "specific": "Increase satellite monitoring to all source regions",
            "measurable": "Monitoring coverage >= 95% of supply chain plots",
            "achievable": "GIS team with satellite data provider",
            "relevant": "EUDR Article 12 monitoring and review obligation",
            "effort_hours": Decimal("45"),
            "cost": Decimal("12000"),
        },
    ],
    "mitigation_measure": [
        {
            "title": "Implement additional mitigation measures",
            "type": ActionType.CORRECTIVE,
            "specific": "Deploy targeted mitigation for identified risk areas",
            "measurable": "Residual risk score reduced by >= 30%",
            "achievable": "Cross-functional mitigation task force",
            "relevant": "EUDR Article 11 risk mitigation requirements",
            "effort_hours": Decimal("40"),
            "cost": Decimal("7000"),
        },
    ],
    "audit_manager": [
        {
            "title": "Enhance audit program",
            "type": ActionType.AUDIT_ENHANCEMENT,
            "specific": "Expand audit scope and frequency for identified gaps",
            "measurable": "Audit findings closure rate >= 90% within SLA",
            "achievable": "Internal audit team with external auditor",
            "relevant": "EUDR Article 29 cooperation with competent authorities",
            "effort_hours": Decimal("60"),
            "cost": Decimal("20000"),
        },
    ],
    "commodity_risk": [
        {
            "title": "Implement commodity-specific risk controls",
            "type": ActionType.PROCESS_CHANGE,
            "specific": "Develop commodity-specific due diligence procedures",
            "measurable": "Commodity-specific controls documented and tested",
            "achievable": "Commodity managers with compliance team",
            "relevant": "EUDR commodity-specific risk assessment requirements",
            "effort_hours": Decimal("30"),
            "cost": Decimal("5000"),
        },
    ],
    "default": [
        {
            "title": "Address identified compliance gap",
            "type": ActionType.CORRECTIVE,
            "specific": "Investigate and remediate identified non-conformity",
            "measurable": "Gap closed and verified by compliance review",
            "achievable": "Compliance team with department support",
            "relevant": "EUDR compliance improvement requirement",
            "effort_hours": Decimal("20"),
            "cost": Decimal("3000"),
        },
    ],
}

# Severity multipliers for effort/cost estimation
_SEVERITY_MULTIPLIERS: Dict[GapSeverity, Decimal] = {
    GapSeverity.CRITICAL: Decimal("2.0"),
    GapSeverity.HIGH: Decimal("1.5"),
    GapSeverity.MEDIUM: Decimal("1.0"),
    GapSeverity.LOW: Decimal("0.7"),
    GapSeverity.INFORMATIONAL: Decimal("0.5"),
}

# Deadline days by severity
_DEADLINE_DAYS: Dict[GapSeverity, int] = {
    GapSeverity.CRITICAL: 14,
    GapSeverity.HIGH: 30,
    GapSeverity.MEDIUM: 60,
    GapSeverity.LOW: 90,
    GapSeverity.INFORMATIONAL: 120,
}


class ActionGenerator:
    """Generates SMART improvement actions from compliance gaps.

    Maps each gap to appropriate action templates, applies severity-based
    scaling for effort and cost estimates, validates SMART criteria, and
    produces structured ImprovementAction records.

    Example:
        >>> engine = ActionGenerator()
        >>> actions = await engine.generate_actions(gaps, "PLAN-001")
        >>> assert all(a.specific_outcome for a in actions)
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize ActionGenerator.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._store: Dict[str, List[ImprovementAction]] = {}
        logger.info("ActionGenerator initialized")

    async def generate_actions(
        self,
        gaps: List[ComplianceGap],
        plan_id: str,
    ) -> List[ImprovementAction]:
        """Generate SMART actions from compliance gaps.

        Args:
            gaps: Compliance gaps to address.
            plan_id: Parent plan identifier.

        Returns:
            List of generated ImprovementAction records.
        """
        start = time.monotonic()
        actions: List[ImprovementAction] = []

        for gap in gaps:
            gap_actions = self._generate_for_gap(gap, plan_id)
            actions.extend(gap_actions)

        # Enforce max actions
        max_actions = self.config.max_actions_per_plan
        actions = actions[:max_actions]

        # Validate SMART criteria
        if self.config.smart_validation_enabled:
            for action in actions:
                is_smart = self._validate_smart(action)
                result_label = "pass" if is_smart else "fail"
                m.record_smart_validation(result_label)

        # Store
        self._store[plan_id] = actions

        # Provenance
        provenance_data = {
            "plan_id": plan_id,
            "actions_generated": len(actions),
            "gaps_addressed": len(gaps),
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)
        for action in actions:
            action.provenance_hash = provenance_hash

        self._provenance.record(
            "action_generation", "generate", plan_id, AGENT_ID,
            metadata={"count": len(actions)},
        )

        elapsed = time.monotonic() - start
        m.observe_action_generation_duration(elapsed)

        logger.info(
            "Generated %d actions from %d gaps for plan %s in %.1fms",
            len(actions), len(gaps), plan_id, elapsed * 1000,
        )
        return actions

    def _generate_for_gap(
        self, gap: ComplianceGap, plan_id: str
    ) -> List[ImprovementAction]:
        """Generate actions for a single gap.

        Args:
            gap: Compliance gap to address.
            plan_id: Parent plan identifier.

        Returns:
            List of actions for this gap.
        """
        dimension = gap.risk_dimension or "default"
        templates = _ACTION_TEMPLATES.get(dimension, _ACTION_TEMPLATES["default"])

        # For critical gaps, ensure minimum action count
        if gap.severity == GapSeverity.CRITICAL:
            min_actions = self.config.min_actions_per_critical_gap
            while len(templates) < min_actions:
                templates = templates + _ACTION_TEMPLATES["default"]

        severity_mult = _SEVERITY_MULTIPLIERS.get(gap.severity, Decimal("1.0"))
        deadline_days = _DEADLINE_DAYS.get(
            gap.severity, self.config.default_action_deadline_days
        )
        deadline = datetime.now(timezone.utc) + timedelta(days=deadline_days)

        actions: List[ImprovementAction] = []
        for tpl in templates:
            action_id = f"ACT-{uuid.uuid4().hex[:12]}"
            effort = (tpl["effort_hours"] * severity_mult).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            cost = (tpl["cost"] * severity_mult).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            action = ImprovementAction(
                action_id=action_id,
                plan_id=plan_id,
                gap_id=gap.gap_id,
                title=tpl["title"],
                description=f"Action to address: {gap.title}",
                action_type=tpl["type"],
                status=ActionStatus.DRAFT,
                specific_outcome=tpl["specific"],
                measurable_kpi=tpl["measurable"],
                achievable_resources=tpl["achievable"],
                relevant_justification=tpl["relevant"],
                time_bound_deadline=deadline,
                estimated_effort_hours=effort,
                estimated_cost=cost,
            )
            actions.append(action)
            m.record_action_generated(tpl["type"].value)

        return actions

    def _validate_smart(self, action: ImprovementAction) -> bool:
        """Validate that an action meets SMART criteria.

        Args:
            action: Action to validate.

        Returns:
            True if all SMART fields are populated.
        """
        return all([
            bool(action.specific_outcome),
            bool(action.measurable_kpi),
            bool(action.achievable_resources),
            bool(action.relevant_justification),
            action.time_bound_deadline is not None,
        ])

    async def get_actions(self, plan_id: str) -> List[ImprovementAction]:
        """Retrieve stored actions for a plan.

        Args:
            plan_id: Plan identifier.

        Returns:
            List of ImprovementAction.
        """
        return self._store.get(plan_id, [])

    async def update_action_status(
        self,
        plan_id: str,
        action_id: str,
        new_status: ActionStatus,
    ) -> Optional[ImprovementAction]:
        """Update an action's status.

        Args:
            plan_id: Plan identifier.
            action_id: Action identifier.
            new_status: New status to set.

        Returns:
            Updated action or None if not found.
        """
        actions = self._store.get(plan_id, [])
        for action in actions:
            if action.action_id == action_id:
                action.status = new_status
                if new_status == ActionStatus.IN_PROGRESS and not action.started_at:
                    action.started_at = datetime.now(timezone.utc)
                elif new_status == ActionStatus.COMPLETED:
                    action.completed_at = datetime.now(timezone.utc)
                    m.record_action_completed()
                elif new_status == ActionStatus.VERIFIED:
                    action.verified_at = datetime.now(timezone.utc)
                    m.record_action_verified()
                self._provenance.record(
                    "action", "status_update", action_id, AGENT_ID,
                    metadata={"new_status": new_status.value},
                )
                return action
        return None

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "ActionGenerator",
            "status": "healthy",
            "plans_with_actions": len(self._store),
        }
