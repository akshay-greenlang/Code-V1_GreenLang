# -*- coding: utf-8 -*-
"""
Facility Assessment Engine - AGENT-EUDR-010: Segregation Verifier (Feature 7)

Conducts comprehensive facility segregation capability assessments using a
6-level (level_0 through level_5) classification system with weighted scoring
across 5 dimensions: layout (0.30), protocols (0.25), history (0.20), labeling
(0.15), and documentation (0.10). Supports certification readiness assessment
for FSC, RSPO, and ISCC, peer benchmarking by commodity, improvement trajectory
tracking, and reassessment scheduling.

Zero-Hallucination Guarantees:
    - All scores are computed via deterministic weighted arithmetic formulas
    - Capability level determination uses static score-range mapping
    - Certification readiness uses static minimum-score requirements
    - Peer comparison uses static reference benchmark averages
    - Improvement actions use static priority/impact templates
    - SHA-256 provenance hashes on all assessment results
    - No ML/LLM used for any scoring, classification, or recommendation

Performance Targets:
    - Single facility assessment: <100ms
    - Certification readiness check: <20ms
    - Peer comparison: <10ms

Regulatory References:
    - EUDR Article 10(2)(f): Segregation capability requirements
    - EUDR Article 14: Competent authority inspection readiness
    - ISO 22095:2020: Chain of Custody - Facility requirements
    - FSC-STD-40-004: Chain of Custody certification requirements
    - RSPO SCC Standard: Segregation system requirements
    - ISCC PLUS: Chain of Custody system requirements

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 7: Facility Segregation Assessment)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "asmt") -> str:
    """Generate a unique identifier with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"

# ---------------------------------------------------------------------------
# Constants: Assessment Configuration
# ---------------------------------------------------------------------------

#: Assessment dimension weights (must sum to 1.0).
ASSESSMENT_WEIGHTS: Dict[str, float] = {
    "layout": 0.30,
    "protocols": 0.25,
    "history": 0.20,
    "labeling": 0.15,
    "documentation": 0.10,
}

#: Capability level definitions with score ranges.
CAPABILITY_LEVELS: Dict[str, Tuple[float, float]] = {
    "level_0": (0.0, 19.99),
    "level_1": (20.0, 39.99),
    "level_2": (40.0, 59.99),
    "level_3": (60.0, 74.99),
    "level_4": (75.0, 89.99),
    "level_5": (90.0, 100.0),
}

#: Capability level descriptions.
CAPABILITY_LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "level_0": "No segregation capability - material freely mixed",
    "level_1": "Basic segregation - administrative separation only",
    "level_2": "Intermediate - physical barriers with shared equipment",
    "level_3": "Advanced - dedicated zones with some shared handling",
    "level_4": "High - fully dedicated zones, equipment, and personnel",
    "level_5": "Maximum - separate buildings/facilities per material type",
}

#: Certification requirements: minimum scores per dimension.
CERTIFICATION_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    "FSC": {
        "layout": 70.0,
        "protocols": 75.0,
        "history": 60.0,
        "labeling": 80.0,
        "documentation": 85.0,
        "overall": 75.0,
    },
    "RSPO": {
        "layout": 65.0,
        "protocols": 70.0,
        "history": 55.0,
        "labeling": 75.0,
        "documentation": 80.0,
        "overall": 70.0,
    },
    "ISCC": {
        "layout": 60.0,
        "protocols": 65.0,
        "history": 50.0,
        "labeling": 70.0,
        "documentation": 75.0,
        "overall": 65.0,
    },
}

#: Peer average benchmark scores by commodity.
PEER_AVERAGES: Dict[str, Dict[str, float]] = {
    "cattle": {
        "layout": 55.0,
        "protocols": 50.0,
        "history": 45.0,
        "labeling": 40.0,
        "documentation": 50.0,
        "overall": 50.0,
    },
    "cocoa": {
        "layout": 65.0,
        "protocols": 60.0,
        "history": 55.0,
        "labeling": 55.0,
        "documentation": 60.0,
        "overall": 60.0,
    },
    "coffee": {
        "layout": 70.0,
        "protocols": 65.0,
        "history": 60.0,
        "labeling": 60.0,
        "documentation": 65.0,
        "overall": 65.0,
    },
    "oil_palm": {
        "layout": 60.0,
        "protocols": 55.0,
        "history": 50.0,
        "labeling": 50.0,
        "documentation": 55.0,
        "overall": 55.0,
    },
    "rubber": {
        "layout": 55.0,
        "protocols": 50.0,
        "history": 45.0,
        "labeling": 45.0,
        "documentation": 50.0,
        "overall": 50.0,
    },
    "soya": {
        "layout": 60.0,
        "protocols": 55.0,
        "history": 50.0,
        "labeling": 50.0,
        "documentation": 55.0,
        "overall": 55.0,
    },
    "wood": {
        "layout": 75.0,
        "protocols": 70.0,
        "history": 65.0,
        "labeling": 65.0,
        "documentation": 70.0,
        "overall": 70.0,
    },
}

#: Priority impact scores for improvement action categories.
IMPROVEMENT_PRIORITIES: Dict[str, float] = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

#: Improvement action templates by dimension and score range.
IMPROVEMENT_ACTION_TEMPLATES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "layout": {
        "critical": [
            {
                "category": "infrastructure",
                "priority": "high",
                "description": "Install physical barriers between all material streams",
                "estimated_impact_score": 25.0,
                "estimated_effort": "6-12 weeks",
            },
            {
                "category": "infrastructure",
                "priority": "high",
                "description": "Designate separate entry/exit points for each material stream",
                "estimated_impact_score": 15.0,
                "estimated_effort": "4-8 weeks",
            },
        ],
        "moderate": [
            {
                "category": "infrastructure",
                "priority": "medium",
                "description": "Upgrade barriers to full-height separation where partial exists",
                "estimated_impact_score": 10.0,
                "estimated_effort": "2-4 weeks",
            },
            {
                "category": "access_control",
                "priority": "medium",
                "description": "Install access control systems at zone boundaries",
                "estimated_impact_score": 8.0,
                "estimated_effort": "2-3 weeks",
            },
        ],
        "minor": [
            {
                "category": "maintenance",
                "priority": "low",
                "description": "Enhance floor markings and zone boundary indicators",
                "estimated_impact_score": 5.0,
                "estimated_effort": "1-2 weeks",
            },
        ],
    },
    "protocols": {
        "critical": [
            {
                "category": "sop",
                "priority": "high",
                "description": "Develop and implement comprehensive segregation SOPs",
                "estimated_impact_score": 20.0,
                "estimated_effort": "4-6 weeks",
            },
            {
                "category": "training",
                "priority": "high",
                "description": "Conduct mandatory segregation training for all personnel",
                "estimated_impact_score": 15.0,
                "estimated_effort": "2-4 weeks",
            },
        ],
        "moderate": [
            {
                "category": "cleaning",
                "priority": "medium",
                "description": "Implement validated cleaning schedules between material changeovers",
                "estimated_impact_score": 12.0,
                "estimated_effort": "2-3 weeks",
            },
            {
                "category": "inspection",
                "priority": "medium",
                "description": "Increase inspection frequency to daily zone checks",
                "estimated_impact_score": 8.0,
                "estimated_effort": "1-2 weeks",
            },
        ],
        "minor": [
            {
                "category": "training",
                "priority": "low",
                "description": "Schedule annual refresher training for experienced personnel",
                "estimated_impact_score": 5.0,
                "estimated_effort": "1 week",
            },
        ],
    },
    "history": {
        "critical": [
            {
                "category": "investigation",
                "priority": "high",
                "description": "Conduct root cause analysis for all unresolved contamination events",
                "estimated_impact_score": 15.0,
                "estimated_effort": "2-4 weeks",
            },
            {
                "category": "corrective",
                "priority": "high",
                "description": "Close all overdue corrective actions from previous audits",
                "estimated_impact_score": 12.0,
                "estimated_effort": "2-6 weeks",
            },
        ],
        "moderate": [
            {
                "category": "monitoring",
                "priority": "medium",
                "description": "Implement continuous monitoring for high-risk zones",
                "estimated_impact_score": 10.0,
                "estimated_effort": "2-3 weeks",
            },
        ],
        "minor": [
            {
                "category": "review",
                "priority": "low",
                "description": "Review historical event patterns and update risk register",
                "estimated_impact_score": 5.0,
                "estimated_effort": "1-2 weeks",
            },
        ],
    },
    "labeling": {
        "critical": [
            {
                "category": "labeling",
                "priority": "high",
                "description": "Apply compliance labels to all unlabeled SCPs immediately",
                "estimated_impact_score": 20.0,
                "estimated_effort": "1-2 weeks",
            },
            {
                "category": "color_coding",
                "priority": "high",
                "description": "Implement consistent color coding across all zones",
                "estimated_impact_score": 10.0,
                "estimated_effort": "1-2 weeks",
            },
        ],
        "moderate": [
            {
                "category": "labeling",
                "priority": "medium",
                "description": "Replace damaged and expired labels in identified areas",
                "estimated_impact_score": 8.0,
                "estimated_effort": "1 week",
            },
        ],
        "minor": [
            {
                "category": "labeling",
                "priority": "low",
                "description": "Upgrade label materials for better durability and readability",
                "estimated_impact_score": 5.0,
                "estimated_effort": "2-4 weeks",
            },
        ],
    },
    "documentation": {
        "critical": [
            {
                "category": "records",
                "priority": "high",
                "description": "Implement comprehensive record-keeping system for all segregation operations",
                "estimated_impact_score": 20.0,
                "estimated_effort": "4-8 weeks",
            },
            {
                "category": "chain_of_custody",
                "priority": "high",
                "description": "Establish complete chain of custody documentation for all material flows",
                "estimated_impact_score": 15.0,
                "estimated_effort": "3-6 weeks",
            },
        ],
        "moderate": [
            {
                "category": "accessibility",
                "priority": "medium",
                "description": "Digitize paper-based records for improved accessibility and searchability",
                "estimated_impact_score": 10.0,
                "estimated_effort": "4-8 weeks",
            },
        ],
        "minor": [
            {
                "category": "timeliness",
                "priority": "low",
                "description": "Reduce record entry lag to within 24 hours of event occurrence",
                "estimated_impact_score": 5.0,
                "estimated_effort": "1-2 weeks",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Internal Dataclass Result Types
# ---------------------------------------------------------------------------

@dataclass
class FacilityProfile:
    """Profile of a facility for segregation assessment.

    Attributes:
        facility_id: Unique facility identifier.
        facility_type: Type of facility (warehouse/processing/port/farm).
        commodities_handled: List of commodities processed at facility.
        coc_models_used: Chain of custody models (segregation/mass_balance).
        capability_level: Current capability level (level_0..level_5).
        scps: Number of segregation control points.
        zones: Number of storage/processing zones.
        lines: Number of processing lines.
        metadata: Additional facility data.
        provenance_hash: SHA-256 hash for audit trail.
    """

    facility_id: str
    facility_type: str
    commodities_handled: List[str]
    coc_models_used: List[str]
    capability_level: str
    scps: int
    zones: int
    lines: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

@dataclass
class AssessmentResult:
    """Result of a facility segregation capability assessment.

    Attributes:
        assessment_id: Unique assessment identifier.
        facility_id: Assessed facility identifier.
        capability_level: Determined capability level.
        layout_score: Layout dimension score (0-100).
        protocol_score: Protocol dimension score (0-100).
        history_score: History dimension score (0-100).
        labeling_score: Labeling dimension score (0-100).
        documentation_score: Documentation dimension score (0-100).
        overall_score: Weighted composite score (0-100).
        sub_scores: Detailed sub-score breakdown.
        recommendations: Prioritized improvement recommendations.
        improvement_actions: Structured improvement action list.
        certification_readiness: Readiness status per standard.
        peer_comparison: Benchmark comparison data.
        assessment_date: ISO timestamp of assessment.
        provenance_hash: SHA-256 hash for audit trail.
    """

    assessment_id: str
    facility_id: str
    capability_level: str
    layout_score: float
    protocol_score: float
    history_score: float
    labeling_score: float
    documentation_score: float
    overall_score: float
    sub_scores: Dict[str, Any]
    recommendations: List[str]
    improvement_actions: List[Dict[str, Any]]
    certification_readiness: Dict[str, Any]
    peer_comparison: Dict[str, Any]
    assessment_date: str
    provenance_hash: str = ""

@dataclass
class ImprovementAction:
    """A structured improvement action for a facility.

    Attributes:
        action_id: Unique action identifier.
        category: Action category (infrastructure/sop/training/etc).
        priority: Priority level (high/medium/low).
        description: Description of the improvement action.
        estimated_impact_score: Expected score improvement (0-100).
        estimated_effort: Effort estimate description.
        target_date: Target completion date (ISO format).
    """

    action_id: str
    category: str
    priority: str
    description: str
    estimated_impact_score: float
    estimated_effort: str
    target_date: str

@dataclass
class CertificationReadiness:
    """Readiness assessment for a specific certification standard.

    Attributes:
        standard: Certification standard (FSC/RSPO/ISCC).
        current_score: Current facility overall score.
        target_score: Required score for certification.
        gaps: List of gap descriptions.
        estimated_readiness_date: Estimated date to reach target.
    """

    standard: str
    current_score: float
    target_score: float
    gaps: List[str]
    estimated_readiness_date: str

# ---------------------------------------------------------------------------
# FacilityAssessmentEngine
# ---------------------------------------------------------------------------

class FacilityAssessmentEngine:
    """Conducts facility segregation capability assessments.

    Implements a 6-level capability classification (level_0 through
    level_5) using weighted scoring across 5 dimensions: layout (0.30),
    protocols (0.25), history (0.20), labeling (0.15), and documentation
    (0.10). Supports FSC/RSPO/ISCC certification readiness assessment,
    peer benchmarking by commodity, and improvement trajectory tracking.

    All scoring and classification logic is deterministic (no ML/LLM).
    Recommendations use static template dictionaries indexed by
    dimension and score range.

    Attributes:
        _profiles: In-memory store of facility profiles keyed by facility_id.
        _assessments: In-memory store of assessments keyed by assessment_id.
        _facility_assessments: Mapping of facility_id to assessment_id list.
        _reassessment_schedule: Mapping of facility_id to scheduled dates.

    Example:
        >>> engine = FacilityAssessmentEngine()
        >>> profile = engine.register_facility_profile(
        ...     facility_id="fac-001",
        ...     facility_type="warehouse",
        ...     commodities=["cocoa"],
        ...     coc_models=["segregation"],
        ... )
        >>> result = engine.run_assessment(
        ...     facility_id="fac-001",
        ...     layout_data={...},
        ...     protocol_data={...},
        ...     history_data={...},
        ...     labeling_data={...},
        ...     documentation_data={...},
        ... )
        >>> assert result.capability_level in CAPABILITY_LEVELS
    """

    def __init__(self) -> None:
        """Initialize FacilityAssessmentEngine."""
        self._profiles: Dict[str, FacilityProfile] = {}
        self._assessments: Dict[str, AssessmentResult] = {}
        self._facility_assessments: Dict[str, List[str]] = {}
        self._reassessment_schedule: Dict[str, str] = {}
        logger.info(
            "FacilityAssessmentEngine initialized: "
            "levels=%d, dimensions=%d, module_version=%s",
            len(CAPABILITY_LEVELS),
            len(ASSESSMENT_WEIGHTS),
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Facility Profile Management
    # ------------------------------------------------------------------

    def register_facility_profile(
        self,
        facility_id: str,
        facility_type: str,
        commodities: List[str],
        coc_models: List[str],
    ) -> FacilityProfile:
        """Register a facility profile for assessment.

        Args:
            facility_id: Unique facility identifier.
            facility_type: Type of facility (warehouse/processing/port/farm).
            commodities: List of EUDR commodities handled.
            coc_models: Chain of custody models used.

        Returns:
            FacilityProfile with provenance hash.

        Raises:
            ValueError: If facility_id is empty.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")

        profile = FacilityProfile(
            facility_id=facility_id,
            facility_type=facility_type,
            commodities_handled=list(commodities),
            coc_models_used=list(coc_models),
            capability_level="level_0",
            scps=0,
            zones=0,
            lines=0,
            metadata={
                "module_version": _MODULE_VERSION,
                "registered_at": utcnow().isoformat(),
            },
        )
        profile.provenance_hash = _compute_hash({
            "facility_id": facility_id,
            "facility_type": facility_type,
            "commodities": commodities,
            "coc_models": coc_models,
        })

        self._profiles[facility_id] = profile
        self._facility_assessments.setdefault(facility_id, [])

        logger.info(
            "Facility profile registered: facility=%s, type=%s, "
            "commodities=%s",
            facility_id,
            facility_type,
            commodities,
        )
        return profile

    def get_facility_profile(
        self,
        facility_id: str,
    ) -> Optional[FacilityProfile]:
        """Retrieve a facility profile.

        Args:
            facility_id: Facility identifier.

        Returns:
            FacilityProfile if found, None otherwise.
        """
        return self._profiles.get(facility_id)

    def update_facility_profile(
        self,
        facility_id: str,
        updates: Dict[str, Any],
    ) -> FacilityProfile:
        """Update a facility profile with new data.

        Args:
            facility_id: Facility identifier.
            updates: Dictionary of fields to update. Supported keys:
                facility_type, commodities_handled, coc_models_used,
                scps, zones, lines.

        Returns:
            Updated FacilityProfile.

        Raises:
            ValueError: If facility_id is not found.
        """
        if facility_id not in self._profiles:
            raise ValueError(f"Facility profile not found: {facility_id}")

        profile = self._profiles[facility_id]

        # Apply updates to mutable fields
        if "facility_type" in updates:
            profile.facility_type = str(updates["facility_type"])
        if "commodities_handled" in updates:
            profile.commodities_handled = list(updates["commodities_handled"])
        if "coc_models_used" in updates:
            profile.coc_models_used = list(updates["coc_models_used"])
        if "scps" in updates:
            profile.scps = int(updates["scps"])
        if "zones" in updates:
            profile.zones = int(updates["zones"])
        if "lines" in updates:
            profile.lines = int(updates["lines"])

        profile.metadata["updated_at"] = utcnow().isoformat()
        profile.provenance_hash = _compute_hash({
            "facility_id": facility_id,
            "facility_type": profile.facility_type,
            "commodities": profile.commodities_handled,
            "updated_at": profile.metadata["updated_at"],
        })

        logger.info(
            "Facility profile updated: facility=%s, fields=%s",
            facility_id,
            list(updates.keys()),
        )
        return profile

    # ------------------------------------------------------------------
    # Public API: Assessment Execution
    # ------------------------------------------------------------------

    def run_assessment(
        self,
        facility_id: str,
        layout_data: Dict[str, Any],
        protocol_data: Dict[str, Any],
        history_data: Dict[str, Any],
        labeling_data: Dict[str, Any],
        documentation_data: Dict[str, Any],
    ) -> AssessmentResult:
        """Run a comprehensive facility segregation assessment.

        Evaluates 5 dimensions with weighted scoring to determine
        the facility's capability level and generate improvement
        recommendations.

        Args:
            facility_id: Facility identifier.
            layout_data: Layout assessment data with keys: zones,
                barriers, access_controls, separation_distances.
            protocol_data: Protocol assessment data with keys:
                sops, training_records, cleaning_schedules,
                inspection_frequency.
            history_data: Historical performance data with keys:
                contamination_events, audit_findings,
                corrective_actions.
            labeling_data: Labeling assessment data with keys:
                label_audit_result or individual label scores.
            documentation_data: Documentation assessment data with keys:
                record_completeness, timeliness, accessibility.

        Returns:
            AssessmentResult with all scores and recommendations.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")

        assessment_id = _generate_id("asmt")
        now = utcnow()

        # Score each dimension
        layout_score = self.assess_layout(
            facility_id,
            layout_data.get("zones", []),
            layout_data.get("barriers", []),
            layout_data.get("access_controls", []),
        )
        protocol_score = self.assess_protocols(
            facility_id,
            protocol_data.get("sops", []),
            protocol_data.get("training_records", []),
            protocol_data.get("cleaning_schedules", []),
            protocol_data.get("inspection_frequency", 0),
        )
        history_score = self.assess_history(
            facility_id,
            history_data.get("contamination_events", []),
            history_data.get("audit_findings", []),
            history_data.get("corrective_actions", []),
        )
        labeling_score = self.assess_labeling(
            facility_id,
            labeling_data,
        )
        documentation_score = self.assess_documentation(
            facility_id,
            documentation_data.get("record_completeness", 0.0),
            documentation_data.get("timeliness", 0.0),
            documentation_data.get("accessibility", 0.0),
        )

        # Compute weighted overall score
        overall_score = (
            layout_score * ASSESSMENT_WEIGHTS["layout"]
            + protocol_score * ASSESSMENT_WEIGHTS["protocols"]
            + history_score * ASSESSMENT_WEIGHTS["history"]
            + labeling_score * ASSESSMENT_WEIGHTS["labeling"]
            + documentation_score * ASSESSMENT_WEIGHTS["documentation"]
        )
        overall_score = round(overall_score, 2)

        # Determine capability level
        capability_level = self.determine_capability_level(overall_score)

        # Build sub-scores dict
        sub_scores = {
            "layout": {
                "score": round(layout_score, 2),
                "weight": ASSESSMENT_WEIGHTS["layout"],
                "weighted_contribution": round(
                    layout_score * ASSESSMENT_WEIGHTS["layout"], 2,
                ),
            },
            "protocols": {
                "score": round(protocol_score, 2),
                "weight": ASSESSMENT_WEIGHTS["protocols"],
                "weighted_contribution": round(
                    protocol_score * ASSESSMENT_WEIGHTS["protocols"], 2,
                ),
            },
            "history": {
                "score": round(history_score, 2),
                "weight": ASSESSMENT_WEIGHTS["history"],
                "weighted_contribution": round(
                    history_score * ASSESSMENT_WEIGHTS["history"], 2,
                ),
            },
            "labeling": {
                "score": round(labeling_score, 2),
                "weight": ASSESSMENT_WEIGHTS["labeling"],
                "weighted_contribution": round(
                    labeling_score * ASSESSMENT_WEIGHTS["labeling"], 2,
                ),
            },
            "documentation": {
                "score": round(documentation_score, 2),
                "weight": ASSESSMENT_WEIGHTS["documentation"],
                "weighted_contribution": round(
                    documentation_score * ASSESSMENT_WEIGHTS["documentation"], 2,
                ),
            },
        }

        # Generate improvement actions
        improvement_actions = self.generate_recommendations(
            AssessmentResult(
                assessment_id=assessment_id,
                facility_id=facility_id,
                capability_level=capability_level,
                layout_score=round(layout_score, 2),
                protocol_score=round(protocol_score, 2),
                history_score=round(history_score, 2),
                labeling_score=round(labeling_score, 2),
                documentation_score=round(documentation_score, 2),
                overall_score=overall_score,
                sub_scores=sub_scores,
                recommendations=[],
                improvement_actions=[],
                certification_readiness={},
                peer_comparison={},
                assessment_date=now.isoformat(),
            ),
        )

        # Text recommendations
        recommendations = [
            f"[{act.get('category', '')}] ({act.get('priority', '')}): "
            f"{act.get('description', '')}"
            for act in improvement_actions[:10]
        ]

        # Certification readiness
        cert_readiness: Dict[str, Any] = {}
        for standard in ("FSC", "RSPO", "ISCC"):
            readiness = self.assess_certification_readiness(
                facility_id, standard,
                layout_score=layout_score,
                protocol_score=protocol_score,
                history_score=history_score,
                labeling_score=labeling_score,
                documentation_score=documentation_score,
                overall_score=overall_score,
            )
            cert_readiness[standard] = {
                "standard": readiness.standard,
                "current_score": readiness.current_score,
                "target_score": readiness.target_score,
                "gaps": readiness.gaps,
                "estimated_readiness_date": readiness.estimated_readiness_date,
                "ready": readiness.current_score >= readiness.target_score,
            }

        # Peer comparison
        profile = self._profiles.get(facility_id)
        primary_commodity = (
            profile.commodities_handled[0]
            if profile and profile.commodities_handled
            else "cocoa"
        )
        peer_comparison = self.compare_with_peers(
            facility_id, primary_commodity,
            layout_score=layout_score,
            protocol_score=protocol_score,
            history_score=history_score,
            labeling_score=labeling_score,
            documentation_score=documentation_score,
            overall_score=overall_score,
        )

        result = AssessmentResult(
            assessment_id=assessment_id,
            facility_id=facility_id,
            capability_level=capability_level,
            layout_score=round(layout_score, 2),
            protocol_score=round(protocol_score, 2),
            history_score=round(history_score, 2),
            labeling_score=round(labeling_score, 2),
            documentation_score=round(documentation_score, 2),
            overall_score=overall_score,
            sub_scores=sub_scores,
            recommendations=recommendations,
            improvement_actions=improvement_actions,
            certification_readiness=cert_readiness,
            peer_comparison=peer_comparison,
            assessment_date=now.isoformat(),
        )
        result.provenance_hash = _compute_hash({
            "assessment_id": assessment_id,
            "facility_id": facility_id,
            "overall_score": overall_score,
            "capability_level": capability_level,
            "assessment_date": result.assessment_date,
            "module_version": _MODULE_VERSION,
        })

        # Store assessment
        self._assessments[assessment_id] = result
        self._facility_assessments.setdefault(facility_id, []).append(
            assessment_id,
        )

        # Update facility profile capability level
        if facility_id in self._profiles:
            self._profiles[facility_id].capability_level = capability_level

        logger.info(
            "Assessment completed: id=%s, facility=%s, "
            "score=%.1f, level=%s",
            assessment_id,
            facility_id,
            overall_score,
            capability_level,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Dimension Scoring
    # ------------------------------------------------------------------

    def assess_layout(
        self,
        facility_id: str,
        zone_data: List[Dict[str, Any]],
        barrier_data: List[Dict[str, Any]],
        access_control_data: List[Dict[str, Any]],
    ) -> float:
        """Assess facility layout for segregation capability.

        Evaluates physical layout including zone separation, barrier
        quality, access control coverage, and dedicated infrastructure.

        Args:
            facility_id: Facility identifier.
            zone_data: List of zone dicts with keys: zone_id,
                has_barrier, barrier_type, separation_distance_m,
                dedicated.
            barrier_data: List of barrier dicts with keys: barrier_id,
                barrier_type, full_height, condition (good/fair/poor).
            access_control_data: List of access control dicts with keys:
                zone_id, access_controlled, control_type.

        Returns:
            Layout score (0.0 to 100.0).
        """
        if not zone_data:
            return 0.0

        scores: List[float] = []

        # Zone barrier coverage
        zones_with_barriers = sum(
            1 for z in zone_data if z.get("has_barrier", False)
        )
        barrier_coverage = (zones_with_barriers / len(zone_data)) * 100.0
        scores.append(barrier_coverage)

        # Barrier quality
        if barrier_data:
            quality_scores = {
                "good": 100.0,
                "fair": 60.0,
                "poor": 20.0,
            }
            barrier_quality = sum(
                quality_scores.get(
                    str(b.get("condition", "poor")).lower(), 20.0,
                )
                for b in barrier_data
            ) / len(barrier_data)
            scores.append(barrier_quality)

            # Full-height barriers
            full_height_count = sum(
                1 for b in barrier_data if b.get("full_height", False)
            )
            full_height_pct = (full_height_count / len(barrier_data)) * 100.0
            scores.append(full_height_pct)
        else:
            scores.extend([0.0, 0.0])

        # Separation distances
        min_sep = 5.0  # default minimum meters
        adequate_sep = sum(
            1 for z in zone_data
            if float(z.get("separation_distance_m", 0)) >= min_sep
        )
        sep_score = (adequate_sep / max(len(zone_data), 1)) * 100.0
        scores.append(sep_score)

        # Access control
        if access_control_data:
            controlled = sum(
                1 for ac in access_control_data
                if ac.get("access_controlled", False)
            )
            access_score = (controlled / len(access_control_data)) * 100.0
            scores.append(access_score)
        else:
            scores.append(0.0)

        # Dedicated zones
        dedicated = sum(
            1 for z in zone_data if z.get("dedicated", False)
        )
        dedicated_score = (dedicated / max(len(zone_data), 1)) * 100.0
        scores.append(dedicated_score)

        layout_score = sum(scores) / max(len(scores), 1)
        return round(min(100.0, max(0.0, layout_score)), 2)

    def assess_protocols(
        self,
        facility_id: str,
        sop_data: List[Dict[str, Any]],
        training_data: List[Dict[str, Any]],
        cleaning_schedules: List[Dict[str, Any]],
        inspection_frequency: int,
    ) -> float:
        """Assess facility protocols for segregation operations.

        Evaluates SOPs, training records, cleaning schedules, and
        inspection frequency.

        Args:
            facility_id: Facility identifier.
            sop_data: List of SOP dicts with keys: sop_id,
                topic, last_reviewed, approved.
            training_data: List of training dicts with keys:
                employee_id, training_type, completion_date, passed.
            cleaning_schedules: List of cleaning schedule dicts with
                keys: zone_or_line_id, frequency, last_cleaned, verified.
            inspection_frequency: Number of inspections per month.

        Returns:
            Protocol score (0.0 to 100.0).
        """
        scores: List[float] = []

        # SOP coverage and quality
        if sop_data:
            approved_count = sum(
                1 for s in sop_data if s.get("approved", False)
            )
            sop_score = (approved_count / len(sop_data)) * 100.0
            scores.append(sop_score)
        else:
            scores.append(0.0)

        # Training coverage
        if training_data:
            passed_count = sum(
                1 for t in training_data if t.get("passed", False)
            )
            training_score = (passed_count / len(training_data)) * 100.0
            scores.append(training_score)
        else:
            scores.append(0.0)

        # Cleaning schedule adherence
        if cleaning_schedules:
            verified_count = sum(
                1 for c in cleaning_schedules if c.get("verified", False)
            )
            cleaning_score = (verified_count / len(cleaning_schedules)) * 100.0
            scores.append(cleaning_score)
        else:
            scores.append(0.0)

        # Inspection frequency score (target: 20+ per month = 100%)
        insp_score = min(100.0, (inspection_frequency / 20.0) * 100.0)
        scores.append(insp_score)

        protocol_score = sum(scores) / max(len(scores), 1)
        return round(min(100.0, max(0.0, protocol_score)), 2)

    def assess_history(
        self,
        facility_id: str,
        contamination_events: List[Dict[str, Any]],
        audit_findings: List[Dict[str, Any]],
        corrective_actions: List[Dict[str, Any]],
    ) -> float:
        """Assess historical performance for segregation.

        Evaluates contamination event history, audit findings, and
        corrective action closure rate. Fewer and less severe events
        yield higher scores.

        Args:
            facility_id: Facility identifier.
            contamination_events: List of event dicts with keys:
                severity, resolved.
            audit_findings: List of finding dicts with keys:
                severity, closed.
            corrective_actions: List of action dicts with keys:
                status (open/closed/overdue).

        Returns:
            History score (0.0 to 100.0).
        """
        # Start from 100 and deduct for issues
        score = 100.0

        # Contamination event deductions
        severity_deductions = {
            "critical": 15.0,
            "major": 8.0,
            "minor": 3.0,
            "observation": 1.0,
        }
        for event in contamination_events:
            severity = str(event.get("severity", "minor")).lower()
            resolved = bool(event.get("resolved", False))
            deduction = severity_deductions.get(severity, 3.0)
            if not resolved:
                deduction *= 1.5  # unresolved events penalized more
            score -= deduction

        # Audit finding deductions
        for finding in audit_findings:
            severity = str(finding.get("severity", "minor")).lower()
            closed = bool(finding.get("closed", False))
            deduction = severity_deductions.get(severity, 3.0) * 0.5
            if not closed:
                deduction *= 1.5
            score -= deduction

        # Corrective action closure bonus
        if corrective_actions:
            closed_count = sum(
                1 for ca in corrective_actions
                if str(ca.get("status", "")).lower() == "closed"
            )
            closure_rate = closed_count / len(corrective_actions)
            # Bonus up to 10 points for high closure rate
            score += closure_rate * 10.0

        return round(min(100.0, max(0.0, score)), 2)

    def assess_labeling(
        self,
        facility_id: str,
        label_audit_result: Dict[str, Any],
    ) -> float:
        """Assess labeling compliance from audit data.

        Args:
            facility_id: Facility identifier.
            label_audit_result: Label audit data dict with keys:
                overall_compliance_score (float 0-100), or individual
                coverage_score, readability_score, accuracy_score,
                timeliness_score.

        Returns:
            Labeling score (0.0 to 100.0).
        """
        # Direct overall score
        if "overall_compliance_score" in label_audit_result:
            score = float(label_audit_result["overall_compliance_score"])
            return round(min(100.0, max(0.0, score)), 2)

        # Compute from component scores
        coverage = float(label_audit_result.get("coverage_score", 0.0))
        readability = float(label_audit_result.get("readability_score", 0.0))
        accuracy = float(label_audit_result.get("accuracy_score", 0.0))
        timeliness = float(label_audit_result.get("timeliness_score", 0.0))

        # Use labeling score weights
        score = (
            coverage * 0.30
            + readability * 0.25
            + accuracy * 0.25
            + timeliness * 0.20
        )
        return round(min(100.0, max(0.0, score)), 2)

    def assess_documentation(
        self,
        facility_id: str,
        record_completeness: float,
        timeliness: float,
        accessibility: float,
    ) -> float:
        """Assess documentation quality for segregation records.

        Args:
            facility_id: Facility identifier.
            record_completeness: Completeness score (0-100).
            timeliness: Record timeliness score (0-100).
            accessibility: Record accessibility score (0-100).

        Returns:
            Documentation score (0.0 to 100.0).
        """
        # Weighted average of documentation components
        score = (
            record_completeness * 0.40
            + timeliness * 0.35
            + accessibility * 0.25
        )
        return round(min(100.0, max(0.0, score)), 2)

    # ------------------------------------------------------------------
    # Public API: Level Determination
    # ------------------------------------------------------------------

    def determine_capability_level(
        self,
        overall_score: float,
    ) -> str:
        """Map an overall score to a capability level.

        Uses static score-range mapping from CAPABILITY_LEVELS.

        Args:
            overall_score: Overall assessment score (0-100).

        Returns:
            Capability level string (level_0 through level_5).
        """
        clamped = max(0.0, min(100.0, overall_score))
        for level, (low, high) in CAPABILITY_LEVELS.items():
            if low <= clamped <= high:
                return level
        return "level_0"

    # ------------------------------------------------------------------
    # Public API: Recommendations and Certification
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        assessment_result: AssessmentResult,
    ) -> List[Dict[str, Any]]:
        """Generate prioritized improvement actions from assessment.

        Uses static template dictionaries indexed by dimension and
        score severity category.

        Args:
            assessment_result: AssessmentResult to analyze.

        Returns:
            List of improvement action dictionaries sorted by priority.
        """
        actions: List[Dict[str, Any]] = []
        dimension_scores = {
            "layout": assessment_result.layout_score,
            "protocols": assessment_result.protocol_score,
            "history": assessment_result.history_score,
            "labeling": assessment_result.labeling_score,
            "documentation": assessment_result.documentation_score,
        }

        for dimension, score in dimension_scores.items():
            # Determine severity category
            if score < 40.0:
                category = "critical"
            elif score < 70.0:
                category = "moderate"
            else:
                category = "minor"

            templates = IMPROVEMENT_ACTION_TEMPLATES.get(dimension, {})
            dimension_actions = templates.get(category, [])

            for tmpl in dimension_actions:
                action_id = _generate_id("ia")
                target_date = (
                    utcnow() + timedelta(weeks=12)
                ).isoformat()

                actions.append({
                    "action_id": action_id,
                    "dimension": dimension,
                    "category": tmpl.get("category", "general"),
                    "priority": tmpl.get("priority", "medium"),
                    "description": tmpl.get("description", ""),
                    "estimated_impact_score": tmpl.get(
                        "estimated_impact_score", 5.0,
                    ),
                    "estimated_effort": tmpl.get(
                        "estimated_effort", "TBD",
                    ),
                    "target_date": target_date,
                    "current_dimension_score": round(score, 2),
                })

        # Sort by priority (high first), then by impact
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(
            key=lambda a: (
                priority_order.get(a.get("priority", "low"), 3),
                -a.get("estimated_impact_score", 0),
            ),
        )

        return actions

    def assess_certification_readiness(
        self,
        facility_id: str,
        standard: str,
        layout_score: Optional[float] = None,
        protocol_score: Optional[float] = None,
        history_score: Optional[float] = None,
        labeling_score: Optional[float] = None,
        documentation_score: Optional[float] = None,
        overall_score: Optional[float] = None,
    ) -> CertificationReadiness:
        """Assess readiness for a specific certification standard.

        Args:
            facility_id: Facility identifier.
            standard: Certification standard (FSC/RSPO/ISCC).
            layout_score: Optional override layout score.
            protocol_score: Optional override protocol score.
            history_score: Optional override history score.
            labeling_score: Optional override labeling score.
            documentation_score: Optional override documentation score.
            overall_score: Optional override overall score.

        Returns:
            CertificationReadiness with gaps and estimated date.
        """
        requirements = CERTIFICATION_REQUIREMENTS.get(standard)
        if requirements is None:
            return CertificationReadiness(
                standard=standard,
                current_score=0.0,
                target_score=0.0,
                gaps=[f"Unknown certification standard: {standard}"],
                estimated_readiness_date="unknown",
            )

        # Use provided scores or look up from latest assessment
        scores = {
            "layout": layout_score,
            "protocols": protocol_score,
            "history": history_score,
            "labeling": labeling_score,
            "documentation": documentation_score,
            "overall": overall_score,
        }

        # Fall back to latest assessment if scores not provided
        if any(v is None for v in scores.values()):
            latest = self._get_latest_assessment(facility_id)
            if latest:
                if scores["layout"] is None:
                    scores["layout"] = latest.layout_score
                if scores["protocols"] is None:
                    scores["protocols"] = latest.protocol_score
                if scores["history"] is None:
                    scores["history"] = latest.history_score
                if scores["labeling"] is None:
                    scores["labeling"] = latest.labeling_score
                if scores["documentation"] is None:
                    scores["documentation"] = latest.documentation_score
                if scores["overall"] is None:
                    scores["overall"] = latest.overall_score

        # Default to 0 for any still-missing scores
        for key in scores:
            if scores[key] is None:
                scores[key] = 0.0

        # Identify gaps
        gaps: List[str] = []
        max_gap = 0.0
        for dimension, required in requirements.items():
            current = scores.get(dimension, 0.0)
            if current < required:
                gap = required - current
                max_gap = max(max_gap, gap)
                gaps.append(
                    f"{dimension}: current {current:.0f} < "
                    f"required {required:.0f} (gap: {gap:.0f} points)"
                )

        # Estimate readiness date (assume 5 points improvement per month)
        if max_gap > 0:
            months = max(1, int(max_gap / 5.0) + 1)
            est_date = (
                utcnow() + timedelta(days=months * 30)
            ).strftime("%Y-%m-%d")
        else:
            est_date = utcnow().strftime("%Y-%m-%d")

        return CertificationReadiness(
            standard=standard,
            current_score=round(scores.get("overall", 0.0), 2),
            target_score=requirements.get("overall", 0.0),
            gaps=gaps,
            estimated_readiness_date=est_date,
        )

    def compare_with_peers(
        self,
        facility_id: str,
        commodity: str,
        layout_score: Optional[float] = None,
        protocol_score: Optional[float] = None,
        history_score: Optional[float] = None,
        labeling_score: Optional[float] = None,
        documentation_score: Optional[float] = None,
        overall_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare facility scores against commodity peer averages.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity for peer group selection.
            layout_score: Optional layout score override.
            protocol_score: Optional protocol score override.
            history_score: Optional history score override.
            labeling_score: Optional labeling score override.
            documentation_score: Optional documentation score override.
            overall_score: Optional overall score override.

        Returns:
            Dictionary with per-dimension comparison data.
        """
        peer_avg = PEER_AVERAGES.get(commodity, PEER_AVERAGES.get("cocoa", {}))

        # Build facility scores dict
        scores = {
            "layout": layout_score,
            "protocols": protocol_score,
            "history": history_score,
            "labeling": labeling_score,
            "documentation": documentation_score,
            "overall": overall_score,
        }

        # Fall back to latest assessment
        if any(v is None for v in scores.values()):
            latest = self._get_latest_assessment(facility_id)
            if latest:
                if scores["layout"] is None:
                    scores["layout"] = latest.layout_score
                if scores["protocols"] is None:
                    scores["protocols"] = latest.protocol_score
                if scores["history"] is None:
                    scores["history"] = latest.history_score
                if scores["labeling"] is None:
                    scores["labeling"] = latest.labeling_score
                if scores["documentation"] is None:
                    scores["documentation"] = latest.documentation_score
                if scores["overall"] is None:
                    scores["overall"] = latest.overall_score

        for key in scores:
            if scores[key] is None:
                scores[key] = 0.0

        comparison: Dict[str, Any] = {
            "commodity": commodity,
            "facility_id": facility_id,
            "dimensions": {},
        }

        for dimension in ("layout", "protocols", "history", "labeling",
                          "documentation", "overall"):
            facility_val = scores.get(dimension, 0.0)
            peer_val = peer_avg.get(dimension, 50.0)
            diff = facility_val - peer_val

            comparison["dimensions"][dimension] = {
                "facility_score": round(facility_val, 2),
                "peer_average": round(peer_val, 2),
                "difference": round(diff, 2),
                "above_average": diff >= 0,
                "percentile_estimate": self._estimate_percentile(
                    facility_val, peer_val,
                ),
            }

        return comparison

    # ------------------------------------------------------------------
    # Public API: Assessment History and Scheduling
    # ------------------------------------------------------------------

    def get_assessment_history(
        self,
        facility_id: str,
    ) -> List[AssessmentResult]:
        """Retrieve all past assessments for a facility.

        Args:
            facility_id: Facility identifier.

        Returns:
            List of AssessmentResult objects, newest first.
        """
        assessment_ids = self._facility_assessments.get(facility_id, [])
        results = [
            self._assessments[aid] for aid in assessment_ids
            if aid in self._assessments
        ]
        results.sort(key=lambda r: r.assessment_date, reverse=True)
        return results

    def schedule_reassessment(
        self,
        facility_id: str,
        trigger_reason: str,
    ) -> Dict[str, Any]:
        """Schedule a reassessment for a facility.

        Args:
            facility_id: Facility identifier.
            trigger_reason: Reason for reassessment (contamination_event,
                corrective_action_completed, periodic, capability_upgrade).

        Returns:
            Dictionary with scheduled_date and trigger details.
        """
        # Determine reassessment timing based on trigger
        delay_days = {
            "contamination_event": 7,
            "corrective_action_completed": 14,
            "periodic": 90,
            "capability_upgrade": 30,
        }
        days = delay_days.get(trigger_reason, 90)
        scheduled_date = (
            utcnow() + timedelta(days=days)
        ).isoformat()

        self._reassessment_schedule[facility_id] = scheduled_date

        logger.info(
            "Reassessment scheduled: facility=%s, reason=%s, date=%s",
            facility_id,
            trigger_reason,
            scheduled_date,
        )

        return {
            "facility_id": facility_id,
            "trigger_reason": trigger_reason,
            "scheduled_date": scheduled_date,
            "delay_days": days,
        }

    def get_improvement_trajectory(
        self,
        facility_id: str,
    ) -> Dict[str, Any]:
        """Compute score improvement trajectory over assessment history.

        Args:
            facility_id: Facility identifier.

        Returns:
            Dictionary with score trends and improvement rate.
        """
        history = self.get_assessment_history(facility_id)

        if not history:
            return {
                "facility_id": facility_id,
                "assessments_count": 0,
                "trend": "no_data",
                "score_history": [],
                "improvement_rate": 0.0,
            }

        # Build score timeline (oldest first)
        history_sorted = sorted(history, key=lambda r: r.assessment_date)
        score_history = [
            {
                "assessment_id": r.assessment_id,
                "date": r.assessment_date,
                "overall_score": r.overall_score,
                "capability_level": r.capability_level,
            }
            for r in history_sorted
        ]

        # Compute improvement rate
        if len(history_sorted) >= 2:
            first_score = history_sorted[0].overall_score
            last_score = history_sorted[-1].overall_score
            score_change = last_score - first_score

            try:
                first_date = datetime.fromisoformat(
                    history_sorted[0].assessment_date,
                )
                last_date = datetime.fromisoformat(
                    history_sorted[-1].assessment_date,
                )
                days = max(1, (last_date - first_date).days)
                monthly_rate = (score_change / days) * 30.0
            except (ValueError, TypeError):
                monthly_rate = 0.0

            trend = (
                "improving" if score_change > 2.0
                else "declining" if score_change < -2.0
                else "stable"
            )
        else:
            monthly_rate = 0.0
            trend = "insufficient_data"

        return {
            "facility_id": facility_id,
            "assessments_count": len(history_sorted),
            "trend": trend,
            "score_history": score_history,
            "improvement_rate": round(monthly_rate, 2),
            "current_score": history_sorted[-1].overall_score,
            "current_level": history_sorted[-1].capability_level,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_latest_assessment(
        self,
        facility_id: str,
    ) -> Optional[AssessmentResult]:
        """Get the most recent assessment for a facility.

        Args:
            facility_id: Facility identifier.

        Returns:
            Most recent AssessmentResult or None.
        """
        assessment_ids = self._facility_assessments.get(facility_id, [])
        if not assessment_ids:
            return None

        latest_id = assessment_ids[-1]
        return self._assessments.get(latest_id)

    def _estimate_percentile(
        self,
        facility_score: float,
        peer_average: float,
    ) -> int:
        """Estimate percentile rank relative to peer average.

        Uses a simplified normal distribution approximation
        centered on the peer average with standard deviation of 15.

        Args:
            facility_score: Facility's dimension score.
            peer_average: Peer group average score.

        Returns:
            Estimated percentile (1-99).
        """
        std_dev = 15.0
        if std_dev <= 0.0:
            return 50

        z_score = (facility_score - peer_average) / std_dev

        # Simplified percentile from z-score (no scipy dependency)
        # Using logistic approximation: P(Z <= z) ~ 1 / (1 + exp(-1.7*z))
        try:
            import math

            prob = 1.0 / (1.0 + math.exp(-1.7 * z_score))
            percentile = int(round(prob * 100.0))
        except (OverflowError, ValueError):
            percentile = 99 if z_score > 0 else 1

        return max(1, min(99, percentile))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "ASSESSMENT_WEIGHTS",
    "CAPABILITY_LEVELS",
    "CAPABILITY_LEVEL_DESCRIPTIONS",
    "CERTIFICATION_REQUIREMENTS",
    "PEER_AVERAGES",
    "IMPROVEMENT_ACTION_TEMPLATES",
    # Result types
    "FacilityProfile",
    "AssessmentResult",
    "ImprovementAction",
    "CertificationReadiness",
    # Engine
    "FacilityAssessmentEngine",
]
