# -*- coding: utf-8 -*-
"""
Compliance Gap Workflow - PACK-018 EU Green Claims Prep
========================================================

4-phase workflow that maps applicable EU Green Claims Directive articles,
evaluates an organisation's existing environmental claims practices against
those requirements, identifies gaps with severity classification, and
generates a phased remediation roadmap.

Phases:
    1. RegulatoryMapping       -- Map applicable Directive articles
    2. CurrentStateAssessment  -- Evaluate existing practices vs requirements
    3. GapAnalysis             -- Identify gaps with severity classification
    4. RemediationRoadmap      -- Generate phased remediation plan

Reference:
    EU Green Claims Directive (COM/2023/166)
    Empowering Consumers Directive (EU) 2024/825
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ComplianceStatus

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class GapPhase(str, Enum):
    """Compliance gap workflow phase identifiers."""
    REGULATORY_MAPPING = "RegulatoryMapping"
    CURRENT_STATE_ASSESSMENT = "CurrentStateAssessment"
    GAP_ANALYSIS = "GapAnalysis"
    REMEDIATION_ROADMAP = "RemediationRoadmap"

class GapSeverity(str, Enum):
    """Severity classification for identified compliance gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADVISORY = "advisory"

class RoadmapPriority(str, Enum):
    """Remediation roadmap phase priority."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

# =============================================================================
# DATA MODELS
# =============================================================================

class GapConfig(BaseModel):
    """Configuration for ComplianceGapWorkflow."""
    include_sme_provisions: bool = Field(
        default=False, description="Include SME/micro-enterprise simplified provisions",
    )
    critical_articles: List[str] = Field(
        default_factory=lambda: ["article_3", "article_10"],
        description="Articles treated as critical for severity classification",
    )
    high_articles: List[str] = Field(
        default_factory=lambda: ["article_4", "article_8"],
        description="Articles treated as high-priority for severity classification",
    )

class GapResult(BaseModel):
    """Individual gap result with severity and remediation detail."""
    gap_id: str = Field(..., description="Unique gap identifier")
    article_id: str = Field(..., description="Related Directive article")
    requirement_text: str = Field(..., description="Requirement description")
    current_status: str = Field(..., description="Current compliance status")
    severity: str = Field(..., description="Gap severity classification")
    impact: str = Field(default="", description="Impact description")
    remediation_steps: List[str] = Field(default_factory=list)
    estimated_effort_days: int = Field(default=0, ge=0)

class WorkflowInput(BaseModel):
    """Input model for ComplianceGapWorkflow."""
    current_practices: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of current organisational practices for green claims",
    )
    applicable_articles: List[str] = Field(
        default_factory=list,
        description="Optional pre-selected Directive articles to assess",
    )
    entity_name: str = Field(default="", description="Reporting entity name")
    industry_sector: str = Field(
        default="general", description="Industry sector for relevance filtering",
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    config: Dict[str, Any] = Field(default_factory=dict)

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)

class WorkflowResult(BaseModel):
    """Complete result from ComplianceGapWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="compliance_gap")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ComplianceGapWorkflow:
    """
    4-phase compliance gap analysis workflow for EU Green Claims Directive.

    Maps applicable Directive articles, evaluates current organisational
    practices against requirements, identifies compliance gaps with severity
    scoring, and produces a phased remediation roadmap.

    Zero-hallucination: all scoring and gap classification uses deterministic
    rule-based logic. No LLM calls in calculation paths.

    Example:
        >>> wf = ComplianceGapWorkflow()
        >>> result = wf.execute(
        ...     current_practices=[{"area": "substantiation", "status": "partial"}],
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "compliance_gap"

    # Directive articles and their requirements
    DIRECTIVE_ARTICLES: Dict[str, Dict[str, Any]] = {
        "article_3": {
            "title": "Substantiation of explicit environmental claims",
            "requirements": [
                "Claims based on recognised scientific evidence",
                "Life cycle perspective considered",
                "Significant environmental aspects identified",
                "Performance beyond legal requirements demonstrated",
            ],
            "applies_to": {"all"},
        },
        "article_4": {
            "title": "Communication of environmental claims",
            "requirements": [
                "Claims accompanied by substantiation information",
                "Claims not misleading about environmental benefits",
                "Claims do not omit relevant environmental information",
                "Comparison claims use equivalent methodologies",
            ],
            "applies_to": {"all"},
        },
        "article_5": {
            "title": "Environmental claims about future performance",
            "requirements": [
                "Detailed implementation plan with measurable targets",
                "Independent third-party monitoring commitment",
                "Regular public progress reporting",
            ],
            "applies_to": {"all"},
        },
        "article_6": {
            "title": "Comparison claims",
            "requirements": [
                "Equivalent data and methodology used for comparison",
                "Fair and verifiable comparison basis",
                "Same functional unit applied",
            ],
            "applies_to": {"all"},
        },
        "article_8": {
            "title": "Environmental labelling requirements",
            "requirements": [
                "Labels based on certification schemes",
                "Certification body independence verified",
                "Publicly accessible scheme criteria",
                "Transparent governance and complaint mechanisms",
            ],
            "applies_to": {"all"},
        },
        "article_10": {
            "title": "Verification of environmental claims",
            "requirements": [
                "Third-party verification before publication",
                "Verifier accredited under Regulation (EC) 765/2008",
                "Verification certificate publicly available",
            ],
            "applies_to": {"all"},
        },
        "article_11": {
            "title": "Small and micro enterprise provisions",
            "requirements": [
                "Simplified substantiation for micro enterprises",
                "Proportionate verification requirements",
            ],
            "applies_to": {"sme", "micro"},
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ComplianceGapWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.gap_config = GapConfig(**self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 4-phase compliance gap analysis pipeline.

        Keyword Args:
            current_practices: List of practice dictionaries.
            applicable_articles: Optional list of article identifiers.
            industry_sector: Sector for relevance filtering.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            current_practices=kwargs.get("current_practices", []),
            applicable_articles=kwargs.get("applicable_articles", []),
            entity_name=kwargs.get("entity_name", ""),
            industry_sector=kwargs.get("industry_sector", "general"),
            reporting_year=kwargs.get("reporting_year", 2025),
            config=kwargs.get("config", {}),
        )

        started_at = utcnow()
        self.logger.info("Starting %s workflow %s", self.WORKFLOW_NAME, self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Regulatory Mapping
            phase_results.append(self._run_regulatory_mapping(input_data))

            # Phase 2 -- Current State Assessment
            mapping_data = phase_results[0].result_data
            phase_results.append(
                self._run_current_state_assessment(input_data, mapping_data)
            )

            # Phase 3 -- Gap Analysis
            state_data = phase_results[1].result_data
            phase_results.append(
                self._run_gap_analysis(mapping_data, state_data)
            )

            # Phase 4 -- Remediation Roadmap
            gap_data = phase_results[2].result_data
            phase_results.append(
                self._run_remediation_roadmap(input_data, gap_data)
            )

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=utcnow(),
                completed_at=utcnow(),
                error_message=str(exc),
            ))

        completed_at = utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "total_practices_assessed": len(input_data.current_practices),
            "phases_completed": len(completed_phases),
            "phases_total": 4,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Workflow %s %s in %.1fs -- %d practices assessed",
            self.workflow_id,
            overall_status.value,
            (completed_at - started_at).total_seconds(),
            len(input_data.current_practices),
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # run_phase dispatcher
    # ------------------------------------------------------------------

    def run_phase(self, phase: GapPhase, **kwargs: Any) -> PhaseResult:
        """
        Run a single named phase independently.

        Args:
            phase: The GapPhase to execute.
            **kwargs: Phase-specific keyword arguments.

        Returns:
            PhaseResult for the executed phase.
        """
        dispatch: Dict[GapPhase, Any] = {
            GapPhase.REGULATORY_MAPPING: lambda: self._run_regulatory_mapping(
                WorkflowInput(
                    current_practices=kwargs.get("current_practices", []),
                    applicable_articles=kwargs.get("applicable_articles", []),
                    industry_sector=kwargs.get("industry_sector", "general"),
                )
            ),
            GapPhase.CURRENT_STATE_ASSESSMENT: lambda: self._run_current_state_assessment(
                WorkflowInput(current_practices=kwargs.get("current_practices", [])),
                kwargs.get("mapping_data", {}),
            ),
            GapPhase.GAP_ANALYSIS: lambda: self._run_gap_analysis(
                kwargs.get("mapping_data", {}),
                kwargs.get("state_data", {}),
            ),
            GapPhase.REMEDIATION_ROADMAP: lambda: self._run_remediation_roadmap(
                WorkflowInput(),
                kwargs.get("gap_data", {}),
            ),
        }
        handler = dispatch.get(phase)
        if handler is None:
            return PhaseResult(
                phase_name=phase.value,
                status=PhaseStatus.FAILED,
                error_message=f"Unknown phase: {phase.value}",
            )
        return handler()

    # ------------------------------------------------------------------
    # Phase 1: Regulatory Mapping
    # ------------------------------------------------------------------

    def _run_regulatory_mapping(self, input_data: WorkflowInput) -> PhaseResult:
        """Map applicable Directive articles to the organisation."""
        started = utcnow()
        self.logger.info("Phase 1/4 RegulatoryMapping -- identifying applicable articles")

        applicable: List[Dict[str, Any]] = []
        sector = input_data.industry_sector.lower()

        for article_id, article_info in self.DIRECTIVE_ARTICLES.items():
            # Filter by user pre-selection if provided
            if input_data.applicable_articles:
                if article_id not in input_data.applicable_articles:
                    continue

            # Filter SME provisions unless configured
            if article_id == "article_11" and not self.gap_config.include_sme_provisions:
                continue

            # Check sector applicability
            applies_to = article_info["applies_to"]
            if "all" in applies_to or sector in applies_to:
                applicable.append({
                    "article_id": article_id,
                    "title": article_info["title"],
                    "requirements_count": len(article_info["requirements"]),
                    "requirements": article_info["requirements"],
                })

        result_data: Dict[str, Any] = {
            "applicable_articles": applicable,
            "total_articles": len(applicable),
            "total_requirements": sum(a["requirements_count"] for a in applicable),
            "sector": sector,
        }

        return PhaseResult(
            phase_name=GapPhase.REGULATORY_MAPPING.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Current State Assessment
    # ------------------------------------------------------------------

    def _run_current_state_assessment(
        self, input_data: WorkflowInput, mapping_data: Dict[str, Any],
    ) -> PhaseResult:
        """Evaluate existing practices against mapped requirements."""
        started = utcnow()
        self.logger.info("Phase 2/4 CurrentStateAssessment -- evaluating %d practices",
                         len(input_data.current_practices))

        practices_by_area: Dict[str, Dict[str, Any]] = {}
        for practice in input_data.current_practices:
            area = practice.get("area", "unknown")
            practices_by_area[area] = practice

        assessments: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {s.value: 0 for s in ComplianceStatus}

        for article in mapping_data.get("applicable_articles", []):
            article_id = article["article_id"]
            for req_idx, requirement in enumerate(article.get("requirements", [])):
                req_id = f"{article_id}_req_{req_idx}"
                compliance = self._assess_requirement(requirement, practices_by_area)
                status_counts[compliance.value] += 1

                assessments.append({
                    "requirement_id": req_id,
                    "article_id": article_id,
                    "requirement_text": requirement,
                    "compliance_status": compliance.value,
                    "matched_practice": self._find_matching_practice(
                        requirement, practices_by_area,
                    ),
                    "evidence_available": compliance != ComplianceStatus.NOT_ASSESSED,
                })

        total = len(assessments)
        compliant_count = status_counts.get(ComplianceStatus.COMPLIANT.value, 0)

        result_data: Dict[str, Any] = {
            "assessments": assessments,
            "status_distribution": status_counts,
            "total_requirements_assessed": total,
            "compliance_rate_pct": round(
                (compliant_count / total * 100) if total else 0.0, 1
            ),
        }

        return PhaseResult(
            phase_name=GapPhase.CURRENT_STATE_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # ------------------------------------------------------------------

    def _run_gap_analysis(
        self, mapping_data: Dict[str, Any], state_data: Dict[str, Any],
    ) -> PhaseResult:
        """Identify compliance gaps with severity classification."""
        started = utcnow()
        self.logger.info("Phase 3/4 GapAnalysis -- identifying gaps")

        gaps: List[Dict[str, Any]] = []
        severity_counts: Dict[str, int] = {s.value: 0 for s in GapSeverity}

        for assessment in state_data.get("assessments", []):
            status = assessment.get("compliance_status", "")
            if status == ComplianceStatus.COMPLIANT.value:
                continue

            severity = self._classify_gap_severity(
                assessment["article_id"], status,
            )
            severity_counts[severity.value] += 1

            gap_result = GapResult(
                gap_id=_new_uuid(),
                article_id=assessment["article_id"],
                requirement_text=assessment["requirement_text"],
                current_status=status,
                severity=severity.value,
                impact=self._describe_gap_impact(assessment["article_id"], severity),
                remediation_steps=self._get_remediation_steps(assessment["article_id"]),
                estimated_effort_days=self._estimate_gap_effort(severity),
            )
            gaps.append(gap_result.model_dump())

        total_gaps = len(gaps)
        critical_high = (
            severity_counts.get(GapSeverity.CRITICAL.value, 0)
            + severity_counts.get(GapSeverity.HIGH.value, 0)
        )

        total_assessed = len(state_data.get("assessments", []))
        result_data: Dict[str, Any] = {
            "gaps": gaps,
            "total_gaps": total_gaps,
            "severity_distribution": severity_counts,
            "critical_high_count": critical_high,
            "gap_rate_pct": round(
                (total_gaps / total_assessed * 100) if total_assessed else 0.0, 1
            ),
        }

        return PhaseResult(
            phase_name=GapPhase.GAP_ANALYSIS.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Remediation Roadmap
    # ------------------------------------------------------------------

    def _run_remediation_roadmap(
        self, input_data: WorkflowInput, gap_data: Dict[str, Any],
    ) -> PhaseResult:
        """Generate a phased remediation roadmap from identified gaps."""
        started = utcnow()
        self.logger.info("Phase 4/4 RemediationRoadmap -- building plan")

        roadmap: Dict[str, List[Dict[str, Any]]] = {
            p.value: [] for p in RoadmapPriority
        }

        for gap in gap_data.get("gaps", []):
            priority = self._assign_priority(gap["severity"])
            roadmap[priority.value].append({
                "gap_id": gap["gap_id"],
                "article_id": gap["article_id"],
                "requirement_text": gap["requirement_text"],
                "severity": gap["severity"],
                "priority": priority.value,
                "remediation_steps": gap.get("remediation_steps", []),
                "estimated_effort_days": gap.get("estimated_effort_days", 10),
                "target_completion": self._target_completion(priority),
            })

        total_effort = sum(
            item["estimated_effort_days"]
            for items in roadmap.values()
            for item in items
        )

        result_data: Dict[str, Any] = {
            "roadmap": roadmap,
            "total_gaps_addressed": gap_data.get("total_gaps", 0),
            "immediate_actions": len(roadmap.get(RoadmapPriority.IMMEDIATE.value, [])),
            "short_term_actions": len(roadmap.get(RoadmapPriority.SHORT_TERM.value, [])),
            "medium_term_actions": len(roadmap.get(RoadmapPriority.MEDIUM_TERM.value, [])),
            "long_term_actions": len(roadmap.get(RoadmapPriority.LONG_TERM.value, [])),
            "estimated_total_effort_days": total_effort,
            "readiness_assessment": self._assess_readiness(gap_data),
        }

        return PhaseResult(
            phase_name=GapPhase.REMEDIATION_ROADMAP.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assess_requirement(
        self, requirement: str, practices: Dict[str, Dict[str, Any]],
    ) -> ComplianceStatus:
        """Assess compliance status for a single requirement."""
        req_lower = requirement.lower()
        keyword_area_map: Dict[str, str] = {
            "substantiation": "substantiation",
            "scientific evidence": "substantiation",
            "life cycle": "lifecycle",
            "lifecycle": "lifecycle",
            "verification": "verification",
            "third-party": "verification",
            "labelling": "labelling",
            "label": "labelling",
            "certification": "certification",
            "comparison": "comparison",
            "communication": "communication",
            "implementation plan": "planning",
            "progress reporting": "reporting",
        }

        for keyword, area in keyword_area_map.items():
            if keyword in req_lower and area in practices:
                practice_status = practices[area].get("status", "").lower()
                if practice_status in ("complete", "compliant", "implemented"):
                    return ComplianceStatus.COMPLIANT
                if practice_status in ("partial", "in_progress", "partially_compliant"):
                    return ComplianceStatus.PARTIALLY_COMPLIANT
                return ComplianceStatus.NON_COMPLIANT

        return ComplianceStatus.NOT_ASSESSED

    def _find_matching_practice(
        self, requirement: str, practices: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        """Find the practice area that best matches a requirement."""
        req_lower = requirement.lower()
        for area in practices:
            if area.lower() in req_lower:
                return area
        return None

    def _classify_gap_severity(self, article_id: str, status: str) -> GapSeverity:
        """Classify gap severity based on article importance and compliance status."""
        critical_set = set(self.gap_config.critical_articles)
        high_set = set(self.gap_config.high_articles)

        if status == ComplianceStatus.NON_COMPLIANT.value:
            if article_id in critical_set:
                return GapSeverity.CRITICAL
            if article_id in high_set:
                return GapSeverity.HIGH
            return GapSeverity.MEDIUM

        if status == ComplianceStatus.PARTIALLY_COMPLIANT.value:
            if article_id in critical_set:
                return GapSeverity.HIGH
            if article_id in high_set:
                return GapSeverity.MEDIUM
            return GapSeverity.LOW

        return GapSeverity.ADVISORY

    def _describe_gap_impact(self, article_id: str, severity: GapSeverity) -> str:
        """Describe the impact of a compliance gap."""
        impact_map: Dict[str, str] = {
            "article_3": "Inability to substantiate claims risks enforcement action under Article 17",
            "article_4": "Misleading communication may trigger consumer complaints and penalties",
            "article_5": "Future claims without implementation plans violate Directive requirements",
            "article_6": "Unfair comparison claims risk competitor legal challenges",
            "article_8": "Non-compliant labels must be withdrawn from market",
            "article_10": "Unverified claims cannot be published under the Directive",
            "article_11": "Failure to leverage SME provisions may result in unnecessary compliance costs",
        }
        return impact_map.get(article_id, "Potential non-compliance with Directive requirements")

    def _estimate_gap_effort(self, severity: GapSeverity) -> int:
        """Estimate effort in working days to close a gap."""
        effort_map: Dict[GapSeverity, int] = {
            GapSeverity.CRITICAL: 30,
            GapSeverity.HIGH: 20,
            GapSeverity.MEDIUM: 10,
            GapSeverity.LOW: 5,
            GapSeverity.ADVISORY: 2,
        }
        return effort_map.get(severity, 10)

    def _assign_priority(self, severity: str) -> RoadmapPriority:
        """Map gap severity to roadmap priority."""
        priority_map: Dict[str, RoadmapPriority] = {
            GapSeverity.CRITICAL.value: RoadmapPriority.IMMEDIATE,
            GapSeverity.HIGH.value: RoadmapPriority.SHORT_TERM,
            GapSeverity.MEDIUM.value: RoadmapPriority.MEDIUM_TERM,
            GapSeverity.LOW.value: RoadmapPriority.LONG_TERM,
            GapSeverity.ADVISORY.value: RoadmapPriority.LONG_TERM,
        }
        return priority_map.get(severity, RoadmapPriority.MEDIUM_TERM)

    def _target_completion(self, priority: RoadmapPriority) -> str:
        """Return target completion timeframe for a priority level."""
        timeframe_map: Dict[RoadmapPriority, str] = {
            RoadmapPriority.IMMEDIATE: "Within 30 days",
            RoadmapPriority.SHORT_TERM: "Within 90 days",
            RoadmapPriority.MEDIUM_TERM: "Within 6 months",
            RoadmapPriority.LONG_TERM: "Within 12 months",
        }
        return timeframe_map.get(priority, "Within 6 months")

    def _get_remediation_steps(self, article_id: str) -> List[str]:
        """Return remediation steps for a given article gap."""
        steps_map: Dict[str, List[str]] = {
            "article_3": [
                "Commission lifecycle assessment or scientific study",
                "Document significant environmental aspects",
                "Compile evidence demonstrating performance beyond legal minimum",
            ],
            "article_4": [
                "Review all claim communications for misleading language",
                "Prepare substantiation information annexes",
                "Ensure no material omissions in environmental claims",
            ],
            "article_5": [
                "Develop detailed implementation plan with interim milestones",
                "Appoint independent third-party monitor",
                "Establish public progress reporting cadence",
            ],
            "article_6": [
                "Align comparison methodology across claim subjects",
                "Document functional unit and system boundaries",
                "Ensure data equivalence and recency",
            ],
            "article_8": [
                "Audit current labels against certification scheme requirements",
                "Verify certifying body independence and accreditation",
                "Publish scheme criteria and complaint procedures",
            ],
            "article_10": [
                "Engage accredited verifier per Regulation (EC) 765/2008",
                "Submit claims for pre-publication verification",
                "Publish verification certificates",
            ],
            "article_11": [
                "Assess eligibility for SME simplified requirements",
                "Document use of proportionate verification approach",
            ],
        }
        return steps_map.get(article_id, [
            "Review requirement against current organisational practices",
            "Develop action plan to address identified gap",
        ])

    def _assess_readiness(self, gap_data: Dict[str, Any]) -> str:
        """Determine overall Directive readiness based on gap analysis."""
        total = gap_data.get("total_gaps", 0)
        critical_high = gap_data.get("critical_high_count", 0)

        if total == 0:
            return "READY -- No compliance gaps identified"
        if critical_high == 0 and total <= 3:
            return "MOSTLY READY -- Minor gaps require attention before enforcement"
        if critical_high <= 2:
            return "PARTIALLY READY -- Address high-priority gaps within 90 days"
        return "NOT READY -- Significant compliance gaps require immediate action"
