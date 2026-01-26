# -*- coding: utf-8 -*-
"""
GL-ADAPT-X-011: TCFD Alignment Agent
=====================================

Aligns climate risk and adaptation assessments with TCFD (Task Force on
Climate-related Financial Disclosures) recommendations.

Capabilities:
    - TCFD pillar alignment assessment
    - Governance evaluation
    - Strategy alignment analysis
    - Risk management process review
    - Metrics and targets assessment
    - Disclosure gap identification
    - Reporting template generation

Zero-Hallucination Guarantees:
    - All assessments against TCFD framework requirements
    - Deterministic alignment scoring
    - Complete provenance tracking
    - No LLM-based compliance determinations

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TCFDPillar(str, Enum):
    """TCFD core pillars."""
    GOVERNANCE = "governance"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    METRICS_TARGETS = "metrics_targets"


class AlignmentLevel(str, Enum):
    """TCFD alignment levels."""
    FULLY_ALIGNED = "fully_aligned"
    SUBSTANTIALLY_ALIGNED = "substantially_aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    MINIMALLY_ALIGNED = "minimally_aligned"
    NOT_ALIGNED = "not_aligned"


class DisclosureStatus(str, Enum):
    """Status of specific disclosures."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    PLANNED = "planned"
    NOT_STARTED = "not_started"


# TCFD recommended disclosures by pillar
TCFD_DISCLOSURES = {
    TCFDPillar.GOVERNANCE: [
        "board_oversight",
        "management_role"
    ],
    TCFDPillar.STRATEGY: [
        "climate_risks_opportunities",
        "impact_on_business",
        "resilience_of_strategy"
    ],
    TCFDPillar.RISK_MANAGEMENT: [
        "risk_identification_process",
        "risk_management_process",
        "integration_overall_risk"
    ],
    TCFDPillar.METRICS_TARGETS: [
        "climate_metrics",
        "scope_1_2_3_emissions",
        "climate_targets"
    ]
}


# =============================================================================
# Pydantic Models
# =============================================================================

class DisclosureItem(BaseModel):
    """Single TCFD disclosure item assessment."""
    disclosure_id: str = Field(...)
    pillar: TCFDPillar = Field(...)
    name: str = Field(...)
    description: str = Field(default="")
    status: DisclosureStatus = Field(...)
    score: float = Field(..., ge=0, le=1)
    evidence: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class PillarAssessment(BaseModel):
    """Assessment of a TCFD pillar."""
    pillar: TCFDPillar = Field(...)
    alignment_level: AlignmentLevel = Field(...)
    overall_score: float = Field(..., ge=0, le=1)
    disclosures: List[DisclosureItem] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)


class ScenarioAnalysisAssessment(BaseModel):
    """Assessment of scenario analysis practices."""
    scenarios_used: List[str] = Field(default_factory=list)
    time_horizons_covered: List[str] = Field(default_factory=list)
    physical_risks_assessed: bool = Field(default=False)
    transition_risks_assessed: bool = Field(default=False)
    quantitative_analysis: bool = Field(default=False)
    alignment_score: float = Field(default=0.0, ge=0, le=1)
    recommendations: List[str] = Field(default_factory=list)


class TCFDAlignmentInput(BaseModel):
    """Input model for TCFD Alignment Agent."""
    assessment_id: str = Field(...)
    organization_name: str = Field(...)

    # Governance inputs
    board_climate_oversight: bool = Field(default=False)
    board_climate_frequency: Optional[str] = Field(None)
    management_climate_responsibility: bool = Field(default=False)
    climate_committee_exists: bool = Field(default=False)

    # Strategy inputs
    climate_risks_identified: List[str] = Field(default_factory=list)
    climate_opportunities_identified: List[str] = Field(default_factory=list)
    scenario_analysis_conducted: bool = Field(default=False)
    scenarios_used: List[str] = Field(default_factory=list)
    strategy_time_horizons: List[str] = Field(default_factory=list)

    # Risk management inputs
    risk_identification_process_documented: bool = Field(default=False)
    risk_management_process_documented: bool = Field(default=False)
    climate_integrated_erm: bool = Field(default=False)

    # Metrics inputs
    scope_1_emissions_reported: bool = Field(default=False)
    scope_2_emissions_reported: bool = Field(default=False)
    scope_3_emissions_reported: bool = Field(default=False)
    climate_targets_set: bool = Field(default=False)
    targets_science_based: bool = Field(default=False)
    other_climate_metrics: List[str] = Field(default_factory=list)

    # Existing disclosures
    existing_disclosures: Dict[str, str] = Field(default_factory=dict)


class TCFDAlignmentOutput(BaseModel):
    """Output model for TCFD Alignment Agent."""
    assessment_id: str = Field(...)
    organization_name: str = Field(...)
    completed_at: datetime = Field(default_factory=DeterministicClock.now)

    # Overall alignment
    overall_alignment_level: AlignmentLevel = Field(...)
    overall_alignment_score: float = Field(..., ge=0, le=1)

    # Pillar assessments
    pillar_assessments: List[PillarAssessment] = Field(default_factory=list)

    # Scenario analysis
    scenario_analysis_assessment: Optional[ScenarioAnalysisAssessment] = Field(None)

    # Summary
    total_disclosures_required: int = Field(default=11)
    disclosures_complete: int = Field(default=0)
    disclosures_partial: int = Field(default=0)
    disclosures_not_started: int = Field(default=0)

    # Gaps and recommendations
    critical_gaps: List[str] = Field(default_factory=list)
    priority_recommendations: List[str] = Field(default_factory=list)

    # Roadmap
    improvement_roadmap: Dict[str, List[str]] = Field(default_factory=dict)

    # Processing info
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# TCFD Alignment Agent Implementation
# =============================================================================

class TCFDAlignmentAgent(BaseAgent):
    """
    GL-ADAPT-X-011: TCFD Alignment Agent

    Aligns climate risk assessments with TCFD recommendations and
    identifies disclosure gaps.

    Zero-Hallucination Implementation:
        - All assessments against TCFD framework
        - Deterministic scoring
        - No LLM-based compliance determinations
        - Complete audit trail

    Example:
        >>> agent = TCFDAlignmentAgent()
        >>> result = agent.run({
        ...     "assessment_id": "TCFD001",
        ...     "organization_name": "Example Corp",
        ...     "board_climate_oversight": True,
        ...     ...
        ... })
    """

    AGENT_ID = "GL-ADAPT-X-011"
    AGENT_NAME = "TCFD Alignment Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the TCFD Alignment Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Aligns assessments with TCFD recommendations",
                version=self.VERSION,
                parameters={}
            )

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def initialize(self):
        """Initialize agent resources."""
        logger.info("TCFD Alignment Agent initialized")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute TCFD alignment assessment."""
        start_time = time.time()

        try:
            tcfd_input = TCFDAlignmentInput(**input_data)
            self.logger.info(
                f"Starting TCFD alignment assessment: {tcfd_input.assessment_id}"
            )

            # Assess each pillar
            pillar_assessments = []
            pillar_scores = []

            # Governance
            gov_assessment = self._assess_governance(tcfd_input)
            pillar_assessments.append(gov_assessment)
            pillar_scores.append(gov_assessment.overall_score)

            # Strategy
            strategy_assessment = self._assess_strategy(tcfd_input)
            pillar_assessments.append(strategy_assessment)
            pillar_scores.append(strategy_assessment.overall_score)

            # Risk Management
            rm_assessment = self._assess_risk_management(tcfd_input)
            pillar_assessments.append(rm_assessment)
            pillar_scores.append(rm_assessment.overall_score)

            # Metrics & Targets
            metrics_assessment = self._assess_metrics_targets(tcfd_input)
            pillar_assessments.append(metrics_assessment)
            pillar_scores.append(metrics_assessment.overall_score)

            # Scenario analysis assessment
            scenario_assessment = self._assess_scenario_analysis(tcfd_input)

            # Overall alignment
            overall_score = sum(pillar_scores) / len(pillar_scores)
            overall_level = self._determine_alignment_level(overall_score)

            # Count disclosures
            all_disclosures = []
            for pa in pillar_assessments:
                all_disclosures.extend(pa.disclosures)

            complete = sum(1 for d in all_disclosures if d.status == DisclosureStatus.COMPLETE)
            partial = sum(1 for d in all_disclosures if d.status == DisclosureStatus.PARTIAL)
            not_started = sum(1 for d in all_disclosures if d.status == DisclosureStatus.NOT_STARTED)

            # Identify critical gaps
            critical_gaps = self._identify_critical_gaps(pillar_assessments)

            # Priority recommendations
            priority_recs = self._generate_priority_recommendations(pillar_assessments)

            # Improvement roadmap
            roadmap = self._create_improvement_roadmap(pillar_assessments)

            processing_time = (time.time() - start_time) * 1000

            output = TCFDAlignmentOutput(
                assessment_id=tcfd_input.assessment_id,
                organization_name=tcfd_input.organization_name,
                overall_alignment_level=overall_level,
                overall_alignment_score=overall_score,
                pillar_assessments=pillar_assessments,
                scenario_analysis_assessment=scenario_assessment,
                total_disclosures_required=11,
                disclosures_complete=complete,
                disclosures_partial=partial,
                disclosures_not_started=not_started,
                critical_gaps=critical_gaps,
                priority_recommendations=priority_recs,
                improvement_roadmap=roadmap,
                processing_time_ms=processing_time,
            )

            output.provenance_hash = self._calculate_provenance_hash(tcfd_input, output)

            self.logger.info(
                f"TCFD alignment assessment complete: {overall_level.value}, "
                f"score: {overall_score:.2f}"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "version": self.VERSION,
                    "alignment_level": overall_level.value
                }
            )

        except Exception as e:
            self.logger.error(f"TCFD alignment assessment failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

    def _assess_governance(self, input_data: TCFDAlignmentInput) -> PillarAssessment:
        """Assess Governance pillar."""
        disclosures = []

        # Board oversight
        board_score = 0.0
        board_gaps = []
        board_evidence = []
        if input_data.board_climate_oversight:
            board_score = 0.7
            board_evidence.append("Board has climate oversight responsibility")
        else:
            board_gaps.append("Board climate oversight not established")
        if input_data.board_climate_frequency:
            board_score = min(1.0, board_score + 0.3)
            board_evidence.append(f"Board reviews climate {input_data.board_climate_frequency}")

        disclosures.append(DisclosureItem(
            disclosure_id="GOV-A",
            pillar=TCFDPillar.GOVERNANCE,
            name="Board Oversight",
            description="Board's oversight of climate-related risks and opportunities",
            status=self._determine_disclosure_status(board_score),
            score=board_score,
            evidence=board_evidence,
            gaps=board_gaps,
            recommendations=["Establish regular board climate reviews"] if board_score < 0.7 else []
        ))

        # Management role
        mgmt_score = 0.0
        mgmt_gaps = []
        mgmt_evidence = []
        if input_data.management_climate_responsibility:
            mgmt_score = 0.6
            mgmt_evidence.append("Management has climate responsibility")
        else:
            mgmt_gaps.append("Management climate responsibility not defined")
        if input_data.climate_committee_exists:
            mgmt_score = min(1.0, mgmt_score + 0.4)
            mgmt_evidence.append("Climate committee/function exists")

        disclosures.append(DisclosureItem(
            disclosure_id="GOV-B",
            pillar=TCFDPillar.GOVERNANCE,
            name="Management Role",
            description="Management's role in assessing and managing climate risks",
            status=self._determine_disclosure_status(mgmt_score),
            score=mgmt_score,
            evidence=mgmt_evidence,
            gaps=mgmt_gaps,
            recommendations=["Define management climate responsibilities"] if mgmt_score < 0.7 else []
        ))

        overall_score = (board_score + mgmt_score) / 2
        strengths = [e for d in disclosures for e in d.evidence]
        weaknesses = [g for d in disclosures for g in d.gaps]

        return PillarAssessment(
            pillar=TCFDPillar.GOVERNANCE,
            alignment_level=self._determine_alignment_level(overall_score),
            overall_score=overall_score,
            disclosures=disclosures,
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            priority_actions=[r for d in disclosures for r in d.recommendations][:3]
        )

    def _assess_strategy(self, input_data: TCFDAlignmentInput) -> PillarAssessment:
        """Assess Strategy pillar."""
        disclosures = []

        # Risks and opportunities
        ro_score = 0.0
        ro_evidence = []
        ro_gaps = []
        if input_data.climate_risks_identified:
            ro_score = 0.4
            ro_evidence.append(f"{len(input_data.climate_risks_identified)} climate risks identified")
        else:
            ro_gaps.append("Climate risks not systematically identified")
        if input_data.climate_opportunities_identified:
            ro_score = min(1.0, ro_score + 0.4)
            ro_evidence.append(f"{len(input_data.climate_opportunities_identified)} opportunities identified")
        else:
            ro_gaps.append("Climate opportunities not identified")

        disclosures.append(DisclosureItem(
            disclosure_id="STR-A",
            pillar=TCFDPillar.STRATEGY,
            name="Climate Risks and Opportunities",
            description="Climate-related risks and opportunities identified",
            status=self._determine_disclosure_status(ro_score),
            score=ro_score,
            evidence=ro_evidence,
            gaps=ro_gaps,
            recommendations=[]
        ))

        # Impact on business
        impact_score = 0.5 if input_data.climate_risks_identified else 0.0
        disclosures.append(DisclosureItem(
            disclosure_id="STR-B",
            pillar=TCFDPillar.STRATEGY,
            name="Impact on Business",
            description="Impact on organization's businesses, strategy, and financial planning",
            status=self._determine_disclosure_status(impact_score),
            score=impact_score,
            evidence=[],
            gaps=["Business impact analysis needed"] if impact_score < 0.5 else [],
            recommendations=[]
        ))

        # Resilience
        resilience_score = 0.0
        if input_data.scenario_analysis_conducted:
            resilience_score = 0.7
        if len(input_data.strategy_time_horizons) >= 2:
            resilience_score = min(1.0, resilience_score + 0.3)

        disclosures.append(DisclosureItem(
            disclosure_id="STR-C",
            pillar=TCFDPillar.STRATEGY,
            name="Resilience of Strategy",
            description="Resilience of strategy under different climate scenarios",
            status=self._determine_disclosure_status(resilience_score),
            score=resilience_score,
            evidence=["Scenario analysis conducted"] if input_data.scenario_analysis_conducted else [],
            gaps=["Scenario analysis not conducted"] if not input_data.scenario_analysis_conducted else [],
            recommendations=[]
        ))

        overall_score = sum(d.score for d in disclosures) / len(disclosures)
        return PillarAssessment(
            pillar=TCFDPillar.STRATEGY,
            alignment_level=self._determine_alignment_level(overall_score),
            overall_score=overall_score,
            disclosures=disclosures,
            strengths=[e for d in disclosures for e in d.evidence][:3],
            weaknesses=[g for d in disclosures for g in d.gaps][:3],
            priority_actions=["Conduct comprehensive scenario analysis"] if not input_data.scenario_analysis_conducted else []
        )

    def _assess_risk_management(self, input_data: TCFDAlignmentInput) -> PillarAssessment:
        """Assess Risk Management pillar."""
        disclosures = []

        # Risk identification
        id_score = 0.7 if input_data.risk_identification_process_documented else 0.2
        disclosures.append(DisclosureItem(
            disclosure_id="RM-A",
            pillar=TCFDPillar.RISK_MANAGEMENT,
            name="Risk Identification Process",
            description="Process for identifying climate-related risks",
            status=self._determine_disclosure_status(id_score),
            score=id_score,
            evidence=["Risk identification process documented"] if input_data.risk_identification_process_documented else [],
            gaps=[] if input_data.risk_identification_process_documented else ["No documented risk identification process"],
            recommendations=[]
        ))

        # Risk management
        mgmt_score = 0.7 if input_data.risk_management_process_documented else 0.2
        disclosures.append(DisclosureItem(
            disclosure_id="RM-B",
            pillar=TCFDPillar.RISK_MANAGEMENT,
            name="Risk Management Process",
            description="Process for managing climate-related risks",
            status=self._determine_disclosure_status(mgmt_score),
            score=mgmt_score,
            evidence=["Risk management process documented"] if input_data.risk_management_process_documented else [],
            gaps=[] if input_data.risk_management_process_documented else ["No documented risk management process"],
            recommendations=[]
        ))

        # Integration
        int_score = 0.8 if input_data.climate_integrated_erm else 0.2
        disclosures.append(DisclosureItem(
            disclosure_id="RM-C",
            pillar=TCFDPillar.RISK_MANAGEMENT,
            name="Integration with Overall Risk",
            description="Integration into overall risk management",
            status=self._determine_disclosure_status(int_score),
            score=int_score,
            evidence=["Climate integrated into ERM"] if input_data.climate_integrated_erm else [],
            gaps=[] if input_data.climate_integrated_erm else ["Climate not integrated into ERM"],
            recommendations=[]
        ))

        overall_score = sum(d.score for d in disclosures) / len(disclosures)
        return PillarAssessment(
            pillar=TCFDPillar.RISK_MANAGEMENT,
            alignment_level=self._determine_alignment_level(overall_score),
            overall_score=overall_score,
            disclosures=disclosures,
            strengths=[e for d in disclosures for e in d.evidence][:3],
            weaknesses=[g for d in disclosures for g in d.gaps][:3],
            priority_actions=["Integrate climate into enterprise risk management"] if not input_data.climate_integrated_erm else []
        )

    def _assess_metrics_targets(self, input_data: TCFDAlignmentInput) -> PillarAssessment:
        """Assess Metrics & Targets pillar."""
        disclosures = []

        # Climate metrics
        metrics_score = 0.2
        metrics_evidence = []
        if input_data.other_climate_metrics:
            metrics_score += 0.3 * min(1, len(input_data.other_climate_metrics) / 3)
            metrics_evidence.append(f"{len(input_data.other_climate_metrics)} climate metrics tracked")

        disclosures.append(DisclosureItem(
            disclosure_id="MT-A",
            pillar=TCFDPillar.METRICS_TARGETS,
            name="Climate Metrics",
            description="Metrics used to assess climate-related risks and opportunities",
            status=self._determine_disclosure_status(metrics_score),
            score=metrics_score,
            evidence=metrics_evidence,
            gaps=["Limited climate metrics"] if metrics_score < 0.5 else [],
            recommendations=[]
        ))

        # GHG emissions
        ghg_score = 0.0
        ghg_evidence = []
        if input_data.scope_1_emissions_reported:
            ghg_score += 0.35
            ghg_evidence.append("Scope 1 emissions reported")
        if input_data.scope_2_emissions_reported:
            ghg_score += 0.35
            ghg_evidence.append("Scope 2 emissions reported")
        if input_data.scope_3_emissions_reported:
            ghg_score += 0.30
            ghg_evidence.append("Scope 3 emissions reported")

        disclosures.append(DisclosureItem(
            disclosure_id="MT-B",
            pillar=TCFDPillar.METRICS_TARGETS,
            name="Scope 1, 2, 3 Emissions",
            description="Scope 1, Scope 2, and Scope 3 GHG emissions",
            status=self._determine_disclosure_status(ghg_score),
            score=ghg_score,
            evidence=ghg_evidence,
            gaps=["Complete GHG inventory needed"] if ghg_score < 0.7 else [],
            recommendations=[]
        ))

        # Targets
        targets_score = 0.0
        targets_evidence = []
        if input_data.climate_targets_set:
            targets_score = 0.6
            targets_evidence.append("Climate targets established")
        if input_data.targets_science_based:
            targets_score = 1.0
            targets_evidence.append("Science-based targets adopted")

        disclosures.append(DisclosureItem(
            disclosure_id="MT-C",
            pillar=TCFDPillar.METRICS_TARGETS,
            name="Climate Targets",
            description="Targets used to manage climate-related risks and performance",
            status=self._determine_disclosure_status(targets_score),
            score=targets_score,
            evidence=targets_evidence,
            gaps=["Climate targets not established"] if not input_data.climate_targets_set else [],
            recommendations=[]
        ))

        overall_score = sum(d.score for d in disclosures) / len(disclosures)
        return PillarAssessment(
            pillar=TCFDPillar.METRICS_TARGETS,
            alignment_level=self._determine_alignment_level(overall_score),
            overall_score=overall_score,
            disclosures=disclosures,
            strengths=[e for d in disclosures for e in d.evidence][:3],
            weaknesses=[g for d in disclosures for g in d.gaps][:3],
            priority_actions=["Set science-based climate targets"] if not input_data.targets_science_based else []
        )

    def _assess_scenario_analysis(
        self,
        input_data: TCFDAlignmentInput
    ) -> ScenarioAnalysisAssessment:
        """Assess scenario analysis practices."""
        score = 0.0
        recommendations = []

        if input_data.scenario_analysis_conducted:
            score = 0.4
        else:
            recommendations.append("Conduct climate scenario analysis")

        if input_data.scenarios_used:
            score += 0.2 * min(1, len(input_data.scenarios_used) / 2)

        if len(input_data.strategy_time_horizons) >= 2:
            score += 0.2

        # Assess physical vs transition
        physical_assessed = any("physical" in r.lower() for r in input_data.climate_risks_identified)
        transition_assessed = any("transition" in r.lower() for r in input_data.climate_risks_identified)

        if physical_assessed:
            score += 0.1
        else:
            recommendations.append("Assess physical climate risks in scenarios")

        if transition_assessed:
            score += 0.1
        else:
            recommendations.append("Assess transition risks in scenarios")

        return ScenarioAnalysisAssessment(
            scenarios_used=input_data.scenarios_used,
            time_horizons_covered=input_data.strategy_time_horizons,
            physical_risks_assessed=physical_assessed,
            transition_risks_assessed=transition_assessed,
            quantitative_analysis=input_data.scenario_analysis_conducted,
            alignment_score=min(1.0, score),
            recommendations=recommendations
        )

    def _determine_alignment_level(self, score: float) -> AlignmentLevel:
        """Determine alignment level from score."""
        if score >= 0.9:
            return AlignmentLevel.FULLY_ALIGNED
        elif score >= 0.7:
            return AlignmentLevel.SUBSTANTIALLY_ALIGNED
        elif score >= 0.5:
            return AlignmentLevel.PARTIALLY_ALIGNED
        elif score >= 0.25:
            return AlignmentLevel.MINIMALLY_ALIGNED
        else:
            return AlignmentLevel.NOT_ALIGNED

    def _determine_disclosure_status(self, score: float) -> DisclosureStatus:
        """Determine disclosure status from score."""
        if score >= 0.8:
            return DisclosureStatus.COMPLETE
        elif score >= 0.4:
            return DisclosureStatus.PARTIAL
        elif score > 0:
            return DisclosureStatus.PLANNED
        else:
            return DisclosureStatus.NOT_STARTED

    def _identify_critical_gaps(
        self,
        pillar_assessments: List[PillarAssessment]
    ) -> List[str]:
        """Identify critical gaps across all pillars."""
        gaps = []
        for pa in pillar_assessments:
            if pa.alignment_level in [AlignmentLevel.NOT_ALIGNED, AlignmentLevel.MINIMALLY_ALIGNED]:
                gaps.append(f"{pa.pillar.value}: Critical alignment gap")
            gaps.extend(pa.weaknesses)
        return gaps[:5]

    def _generate_priority_recommendations(
        self,
        pillar_assessments: List[PillarAssessment]
    ) -> List[str]:
        """Generate priority recommendations."""
        recs = []
        for pa in pillar_assessments:
            recs.extend(pa.priority_actions)
        return recs[:5]

    def _create_improvement_roadmap(
        self,
        pillar_assessments: List[PillarAssessment]
    ) -> Dict[str, List[str]]:
        """Create improvement roadmap."""
        roadmap = {
            "immediate": [],
            "short_term": [],
            "medium_term": []
        }

        for pa in pillar_assessments:
            if pa.alignment_level in [AlignmentLevel.NOT_ALIGNED, AlignmentLevel.MINIMALLY_ALIGNED]:
                roadmap["immediate"].extend(pa.priority_actions[:2])
            elif pa.alignment_level == AlignmentLevel.PARTIALLY_ALIGNED:
                roadmap["short_term"].extend(pa.priority_actions[:2])
            else:
                roadmap["medium_term"].extend(pa.priority_actions[:1])

        return roadmap

    def _calculate_provenance_hash(
        self,
        input_data: TCFDAlignmentInput,
        output: TCFDAlignmentOutput
    ) -> str:
        """Calculate SHA-256 hash for provenance."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "assessment_id": input_data.assessment_id,
            "overall_score": output.overall_alignment_score,
            "timestamp": output.completed_at.isoformat(),
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TCFDAlignmentAgent",
    "TCFDPillar",
    "AlignmentLevel",
    "DisclosureStatus",
    "DisclosureItem",
    "PillarAssessment",
    "ScenarioAnalysisAssessment",
    "TCFDAlignmentInput",
    "TCFDAlignmentOutput",
    "TCFD_DISCLOSURES",
]
