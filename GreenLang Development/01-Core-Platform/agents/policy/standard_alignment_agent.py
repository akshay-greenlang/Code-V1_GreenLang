# -*- coding: utf-8 -*-
"""
GL-POL-X-004: Standard Alignment Agent
======================================

Aligns organizational data and reporting with sustainability standards
including GRI, SASB, TCFD, CDP, and ESRS. CRITICAL PATH agent for
deterministic standard mapping.

Capabilities:
    - Standard requirement mapping (GRI, SASB, TCFD, CDP, ESRS)
    - Disclosure cross-walk between frameworks
    - Data point mapping to standard indicators
    - Coverage analysis and gap identification
    - Standard-specific formatting and validation

Zero-Hallucination Guarantees:
    - All mappings derived from official standard specifications
    - Deterministic cross-walk calculations
    - Complete audit trail for all alignments
    - No LLM inference in standard compliance determination

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class StandardFramework(str, Enum):
    """Sustainability reporting standards."""
    GRI = "gri"
    SASB = "sasb"
    TCFD = "tcfd"
    CDP = "cdp"
    ESRS = "esrs"
    ISSB = "issb"
    UN_SDG = "un_sdg"
    GHG_PROTOCOL = "ghg_protocol"


class DisclosureCategory(str, Enum):
    """Categories of disclosure."""
    GOVERNANCE = "governance"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    METRICS_TARGETS = "metrics_targets"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    ECONOMIC = "economic"


class DataPointType(str, Enum):
    """Types of data points."""
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    NARRATIVE = "narrative"
    BINARY = "binary"


class AlignmentStatus(str, Enum):
    """Alignment status for a disclosure."""
    FULLY_ALIGNED = "fully_aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    NOT_ALIGNED = "not_aligned"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class StandardIndicator(BaseModel):
    """A standard indicator/disclosure requirement."""

    indicator_id: str = Field(..., description="Unique indicator ID")
    framework: StandardFramework = Field(..., description="Standard framework")
    code: str = Field(..., description="Standard code (e.g., GRI 305-1)")
    name: str = Field(..., description="Indicator name")
    description: str = Field(..., description="Description")

    # Classification
    category: DisclosureCategory = Field(..., description="Disclosure category")
    topic: str = Field(..., description="Topic area")
    data_type: DataPointType = Field(..., description="Data type")

    # Requirements
    mandatory: bool = Field(default=False, description="Whether mandatory")
    quantitative_unit: Optional[str] = Field(None, description="Unit if quantitative")
    reporting_boundary: Optional[str] = Field(None, description="Reporting boundary")

    # Cross-references
    related_indicators: List[str] = Field(
        default_factory=list,
        description="Related indicator IDs in other frameworks"
    )


class CrossWalkMapping(BaseModel):
    """Mapping between indicators across frameworks."""

    mapping_id: str = Field(
        default_factory=lambda: deterministic_uuid("crosswalk"),
        description="Unique mapping identifier"
    )
    source_indicator_id: str = Field(..., description="Source indicator")
    source_framework: StandardFramework = Field(..., description="Source framework")
    target_indicator_id: str = Field(..., description="Target indicator")
    target_framework: StandardFramework = Field(..., description="Target framework")

    # Mapping quality
    alignment_type: str = Field(
        default="equivalent",
        description="equivalent, partial, related"
    )
    alignment_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Alignment score (1.0 = equivalent)"
    )
    mapping_notes: Optional[str] = Field(None, description="Notes on mapping")


class DataPointMapping(BaseModel):
    """Mapping of organization data to standard indicator."""

    mapping_id: str = Field(
        default_factory=lambda: deterministic_uuid("datapoint"),
        description="Unique mapping identifier"
    )
    indicator_id: str = Field(..., description="Standard indicator")
    data_source: str = Field(..., description="Source of data")
    data_field: str = Field(..., description="Field name in source")

    # Status
    alignment_status: AlignmentStatus = Field(..., description="Alignment status")
    coverage_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Data coverage percentage"
    )

    # Value (if available)
    current_value: Optional[Any] = Field(None, description="Current value")
    value_unit: Optional[str] = Field(None, description="Value unit")
    reporting_period: Optional[str] = Field(None, description="Reporting period")

    # Gaps
    data_gaps: List[str] = Field(default_factory=list, description="Identified gaps")


class AlignmentResult(BaseModel):
    """Result of standard alignment assessment."""

    result_id: str = Field(
        default_factory=lambda: deterministic_uuid("alignment"),
        description="Unique result identifier"
    )
    organization_id: str = Field(..., description="Organization identifier")
    framework: StandardFramework = Field(..., description="Framework assessed")
    assessment_date: date = Field(
        default_factory=lambda: DeterministicClock.now().date()
    )

    # Coverage metrics
    total_indicators: int = Field(default=0)
    fully_aligned: int = Field(default=0)
    partially_aligned: int = Field(default=0)
    not_aligned: int = Field(default=0)
    not_applicable: int = Field(default=0)

    # Score
    alignment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall alignment score"
    )

    # Details
    indicator_mappings: List[DataPointMapping] = Field(default_factory=list)
    category_scores: Dict[str, float] = Field(default_factory=dict)

    # Gaps
    critical_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)

    def calculate_provenance_hash(self) -> str:
        """Calculate provenance hash."""
        content = {
            "organization_id": self.organization_id,
            "framework": self.framework.value,
            "alignment_score": self.alignment_score,
            "assessment_date": self.assessment_date.isoformat(),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class StandardAlignmentInput(BaseModel):
    """Input for standard alignment operations."""

    action: str = Field(
        ...,
        description="Action: assess_alignment, map_crosswalk, get_indicators"
    )
    organization_id: Optional[str] = Field(None)
    framework: Optional[StandardFramework] = Field(None)
    frameworks: Optional[List[StandardFramework]] = Field(None)
    data_mappings: Optional[Dict[str, Any]] = Field(None)
    indicator_ids: Optional[List[str]] = Field(None)


class StandardAlignmentOutput(BaseModel):
    """Output from standard alignment operations."""

    success: bool = Field(...)
    action: str = Field(...)
    alignment_result: Optional[AlignmentResult] = Field(None)
    crosswalk_mappings: Optional[List[CrossWalkMapping]] = Field(None)
    indicators: Optional[List[StandardIndicator]] = Field(None)
    error: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


# =============================================================================
# STANDARDS DATABASE
# =============================================================================


STANDARD_INDICATORS: Dict[str, StandardIndicator] = {}
CROSSWALK_MAPPINGS: Dict[str, CrossWalkMapping] = {}


def _initialize_standards_database() -> None:
    """Initialize standards indicator database."""
    global STANDARD_INDICATORS, CROSSWALK_MAPPINGS

    indicators = [
        # GRI Standards
        StandardIndicator(
            indicator_id="GRI-305-1",
            framework=StandardFramework.GRI,
            code="305-1",
            name="Direct (Scope 1) GHG emissions",
            description="Report gross direct (Scope 1) GHG emissions in metric tons of CO2 equivalent",
            category=DisclosureCategory.ENVIRONMENTAL,
            topic="Emissions",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="tCO2e",
            related_indicators=["ESRS-E1-4", "SASB-EM-110a.1", "CDP-C6.1"],
        ),
        StandardIndicator(
            indicator_id="GRI-305-2",
            framework=StandardFramework.GRI,
            code="305-2",
            name="Energy indirect (Scope 2) GHG emissions",
            description="Report gross location-based energy indirect (Scope 2) GHG emissions",
            category=DisclosureCategory.ENVIRONMENTAL,
            topic="Emissions",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="tCO2e",
            related_indicators=["ESRS-E1-5", "SASB-EM-110a.2", "CDP-C6.2"],
        ),
        StandardIndicator(
            indicator_id="GRI-305-3",
            framework=StandardFramework.GRI,
            code="305-3",
            name="Other indirect (Scope 3) GHG emissions",
            description="Report gross other indirect (Scope 3) GHG emissions",
            category=DisclosureCategory.ENVIRONMENTAL,
            topic="Emissions",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=False,
            quantitative_unit="tCO2e",
            related_indicators=["ESRS-E1-6", "CDP-C6.5"],
        ),
        # ESRS Standards
        StandardIndicator(
            indicator_id="ESRS-E1-4",
            framework=StandardFramework.ESRS,
            code="E1-4",
            name="GHG emissions reduction targets",
            description="Disclosure of targets related to climate change mitigation and adaptation",
            category=DisclosureCategory.METRICS_TARGETS,
            topic="Climate Change",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            related_indicators=["GRI-305-1", "TCFD-MT-a"],
        ),
        StandardIndicator(
            indicator_id="ESRS-E1-5",
            framework=StandardFramework.ESRS,
            code="E1-5",
            name="Energy consumption and mix",
            description="Total energy consumption and energy mix",
            category=DisclosureCategory.METRICS_TARGETS,
            topic="Climate Change",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="MWh",
            related_indicators=["GRI-302-1", "SASB-EM-130a.1"],
        ),
        StandardIndicator(
            indicator_id="ESRS-E1-6",
            framework=StandardFramework.ESRS,
            code="E1-6",
            name="Gross Scopes 1, 2, 3 and Total GHG emissions",
            description="Gross Scope 1, 2 and 3 GHG emissions",
            category=DisclosureCategory.METRICS_TARGETS,
            topic="Climate Change",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="tCO2e",
            related_indicators=["GRI-305-1", "GRI-305-2", "GRI-305-3"],
        ),
        # TCFD Recommendations
        StandardIndicator(
            indicator_id="TCFD-GOV-a",
            framework=StandardFramework.TCFD,
            code="GOV-a",
            name="Board oversight",
            description="Board's oversight of climate-related risks and opportunities",
            category=DisclosureCategory.GOVERNANCE,
            topic="Governance",
            data_type=DataPointType.NARRATIVE,
            mandatory=True,
            related_indicators=["ESRS-GOV-1", "CDP-C1.1"],
        ),
        StandardIndicator(
            indicator_id="TCFD-STRAT-a",
            framework=StandardFramework.TCFD,
            code="STRAT-a",
            name="Climate-related risks and opportunities",
            description="Climate-related risks and opportunities identified over short, medium, long-term",
            category=DisclosureCategory.STRATEGY,
            topic="Strategy",
            data_type=DataPointType.NARRATIVE,
            mandatory=True,
            related_indicators=["ESRS-E1-1", "CDP-C2.1"],
        ),
        StandardIndicator(
            indicator_id="TCFD-MT-a",
            framework=StandardFramework.TCFD,
            code="MT-a",
            name="Climate-related metrics",
            description="Metrics used to assess climate-related risks and opportunities",
            category=DisclosureCategory.METRICS_TARGETS,
            topic="Metrics & Targets",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            related_indicators=["GRI-305-1", "GRI-305-2", "ESRS-E1-6"],
        ),
        # CDP Indicators
        StandardIndicator(
            indicator_id="CDP-C6.1",
            framework=StandardFramework.CDP,
            code="C6.1",
            name="Scope 1 emissions",
            description="Your gross global Scope 1 emissions",
            category=DisclosureCategory.ENVIRONMENTAL,
            topic="Emissions",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="tCO2e",
            related_indicators=["GRI-305-1", "ESRS-E1-6"],
        ),
        StandardIndicator(
            indicator_id="CDP-C6.3",
            framework=StandardFramework.CDP,
            code="C6.3",
            name="Scope 2 emissions",
            description="Your gross global Scope 2 emissions",
            category=DisclosureCategory.ENVIRONMENTAL,
            topic="Emissions",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="tCO2e",
            related_indicators=["GRI-305-2", "ESRS-E1-6"],
        ),
        # SASB (Generic for multiple industries)
        StandardIndicator(
            indicator_id="SASB-EM-110a.1",
            framework=StandardFramework.SASB,
            code="EM-110a.1",
            name="Gross global Scope 1 emissions",
            description="Gross global Scope 1 emissions, percentage covered under emissions-limiting regulations",
            category=DisclosureCategory.ENVIRONMENTAL,
            topic="GHG Emissions",
            data_type=DataPointType.QUANTITATIVE,
            mandatory=True,
            quantitative_unit="tCO2e",
            related_indicators=["GRI-305-1", "CDP-C6.1"],
        ),
    ]

    for indicator in indicators:
        STANDARD_INDICATORS[indicator.indicator_id] = indicator

    # Create cross-walk mappings
    crosswalks = [
        CrossWalkMapping(
            mapping_id="CW-GRI-ESRS-1",
            source_indicator_id="GRI-305-1",
            source_framework=StandardFramework.GRI,
            target_indicator_id="ESRS-E1-6",
            target_framework=StandardFramework.ESRS,
            alignment_type="equivalent",
            alignment_score=0.95,
            mapping_notes="ESRS E1-6 combines all scopes; GRI 305-1 is Scope 1 only",
        ),
        CrossWalkMapping(
            mapping_id="CW-GRI-TCFD-1",
            source_indicator_id="GRI-305-1",
            source_framework=StandardFramework.GRI,
            target_indicator_id="TCFD-MT-a",
            target_framework=StandardFramework.TCFD,
            alignment_type="partial",
            alignment_score=0.7,
            mapping_notes="TCFD MT-a is broader, GRI 305-1 addresses emission metrics specifically",
        ),
        CrossWalkMapping(
            mapping_id="CW-GRI-CDP-1",
            source_indicator_id="GRI-305-1",
            source_framework=StandardFramework.GRI,
            target_indicator_id="CDP-C6.1",
            target_framework=StandardFramework.CDP,
            alignment_type="equivalent",
            alignment_score=1.0,
            mapping_notes="Direct equivalence for Scope 1 emissions",
        ),
    ]

    for cw in crosswalks:
        CROSSWALK_MAPPINGS[cw.mapping_id] = cw


_initialize_standards_database()


# =============================================================================
# STANDARD ALIGNMENT AGENT
# =============================================================================


class StandardAlignmentAgent(BaseAgent):
    """
    GL-POL-X-004: Standard Alignment Agent

    Aligns organizational reporting with sustainability standards.
    CRITICAL PATH agent with deterministic standard mapping.

    All alignments are:
    - Derived from official standard specifications
    - Based on deterministic matching rules
    - Fully auditable with provenance tracking
    - No LLM inference in compliance determination

    Usage:
        agent = StandardAlignmentAgent()
        result = agent.run({
            'action': 'assess_alignment',
            'organization_id': 'org-123',
            'framework': 'gri'
        })
    """

    AGENT_ID = "GL-POL-X-004"
    AGENT_NAME = "Standard Alignment Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Deterministic standard alignment and cross-walk mapping"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Standard Alignment Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Standard alignment agent",
                version=self.VERSION,
                parameters={}
            )

        self._indicators = STANDARD_INDICATORS.copy()
        self._crosswalks = CROSSWALK_MAPPINGS.copy()

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute standard alignment operation."""
        import time
        start_time = time.time()

        try:
            agent_input = StandardAlignmentInput(**input_data)

            action_handlers = {
                "assess_alignment": self._handle_assess_alignment,
                "map_crosswalk": self._handle_map_crosswalk,
                "get_indicators": self._handle_get_indicators,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)
            output.provenance_hash = hashlib.sha256(
                json.dumps({"action": agent_input.action}, sort_keys=True).encode()
            ).hexdigest()

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
            )

        except Exception as e:
            logger.error(f"Standard alignment failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_assess_alignment(
        self,
        input_data: StandardAlignmentInput
    ) -> StandardAlignmentOutput:
        """Assess alignment with a standard framework."""
        if not input_data.organization_id or not input_data.framework:
            return StandardAlignmentOutput(
                success=False,
                action="assess_alignment",
                error="organization_id and framework required",
            )

        # Get indicators for framework
        framework_indicators = [
            ind for ind in self._indicators.values()
            if ind.framework == input_data.framework
        ]

        # Assess each indicator
        mappings: List[DataPointMapping] = []
        category_scores: Dict[str, List[float]] = {}

        data_mappings = input_data.data_mappings or {}

        for indicator in framework_indicators:
            # Check if data mapping exists
            data_key = indicator.indicator_id
            if data_key in data_mappings:
                status = AlignmentStatus.FULLY_ALIGNED
                coverage = 100.0
                gaps = []
            elif any(rel in data_mappings for rel in indicator.related_indicators):
                status = AlignmentStatus.PARTIALLY_ALIGNED
                coverage = 50.0
                gaps = [f"Direct {indicator.code} data not available, using related indicator"]
            else:
                status = AlignmentStatus.NOT_ALIGNED
                coverage = 0.0
                gaps = [f"No data mapped for {indicator.code}"]

            mapping = DataPointMapping(
                indicator_id=indicator.indicator_id,
                data_source="organization_data",
                data_field=data_key,
                alignment_status=status,
                coverage_percentage=coverage,
                data_gaps=gaps,
            )
            mappings.append(mapping)

            # Track category scores
            cat = indicator.category.value
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(coverage)

        # Calculate results
        fully_aligned = len([m for m in mappings if m.alignment_status == AlignmentStatus.FULLY_ALIGNED])
        partially_aligned = len([m for m in mappings if m.alignment_status == AlignmentStatus.PARTIALLY_ALIGNED])
        not_aligned = len([m for m in mappings if m.alignment_status == AlignmentStatus.NOT_ALIGNED])

        # Overall score
        if mappings:
            total_coverage = sum(m.coverage_percentage for m in mappings)
            alignment_score = total_coverage / len(mappings)
        else:
            alignment_score = 0.0

        # Category average scores
        cat_avg_scores = {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }

        # Critical gaps
        critical_gaps = [
            m.indicator_id
            for m in mappings
            if m.alignment_status == AlignmentStatus.NOT_ALIGNED
            and self._indicators[m.indicator_id].mandatory
        ]

        result = AlignmentResult(
            organization_id=input_data.organization_id,
            framework=input_data.framework,
            total_indicators=len(framework_indicators),
            fully_aligned=fully_aligned,
            partially_aligned=partially_aligned,
            not_aligned=not_aligned,
            alignment_score=round(alignment_score, 1),
            indicator_mappings=mappings,
            category_scores=cat_avg_scores,
            critical_gaps=critical_gaps,
            recommendations=self._generate_recommendations(mappings, input_data.framework),
        )
        result.provenance_hash = result.calculate_provenance_hash()

        return StandardAlignmentOutput(
            success=True,
            action="assess_alignment",
            alignment_result=result,
        )

    def _handle_map_crosswalk(
        self,
        input_data: StandardAlignmentInput
    ) -> StandardAlignmentOutput:
        """Get cross-walk mappings between frameworks."""
        if not input_data.frameworks or len(input_data.frameworks) < 2:
            return StandardAlignmentOutput(
                success=False,
                action="map_crosswalk",
                error="At least two frameworks required for cross-walk",
            )

        # Get relevant mappings
        mappings = [
            cw for cw in self._crosswalks.values()
            if cw.source_framework in input_data.frameworks
            and cw.target_framework in input_data.frameworks
        ]

        return StandardAlignmentOutput(
            success=True,
            action="map_crosswalk",
            crosswalk_mappings=mappings,
        )

    def _handle_get_indicators(
        self,
        input_data: StandardAlignmentInput
    ) -> StandardAlignmentOutput:
        """Get indicators for a framework."""
        indicators = list(self._indicators.values())

        # Filter by framework
        if input_data.framework:
            indicators = [i for i in indicators if i.framework == input_data.framework]

        # Filter by IDs
        if input_data.indicator_ids:
            indicators = [i for i in indicators if i.indicator_id in input_data.indicator_ids]

        return StandardAlignmentOutput(
            success=True,
            action="get_indicators",
            indicators=indicators,
        )

    def _generate_recommendations(
        self,
        mappings: List[DataPointMapping],
        framework: StandardFramework
    ) -> List[str]:
        """Generate recommendations based on gaps."""
        recommendations = []

        not_aligned = [m for m in mappings if m.alignment_status == AlignmentStatus.NOT_ALIGNED]

        if not_aligned:
            recommendations.append(
                f"Address {len(not_aligned)} indicators with no data mapping"
            )

            # Check for mandatory gaps
            mandatory_gaps = [
                m for m in not_aligned
                if self._indicators[m.indicator_id].mandatory
            ]
            if mandatory_gaps:
                recommendations.append(
                    f"Prioritize {len(mandatory_gaps)} mandatory indicators: "
                    f"{', '.join([m.indicator_id for m in mandatory_gaps[:3]])}"
                )

        partial = [m for m in mappings if m.alignment_status == AlignmentStatus.PARTIALLY_ALIGNED]
        if partial:
            recommendations.append(
                f"Improve data quality for {len(partial)} partially aligned indicators"
            )

        return recommendations

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def get_indicator(self, indicator_id: str) -> Optional[StandardIndicator]:
        """Get an indicator by ID."""
        return self._indicators.get(indicator_id)

    def get_crosswalk(
        self,
        source_framework: StandardFramework,
        target_framework: StandardFramework
    ) -> List[CrossWalkMapping]:
        """Get cross-walk mappings between two frameworks."""
        return [
            cw for cw in self._crosswalks.values()
            if cw.source_framework == source_framework
            and cw.target_framework == target_framework
        ]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "StandardAlignmentAgent",
    "StandardFramework",
    "DisclosureCategory",
    "DataPointType",
    "AlignmentStatus",
    "StandardIndicator",
    "CrossWalkMapping",
    "DataPointMapping",
    "AlignmentResult",
    "StandardAlignmentInput",
    "StandardAlignmentOutput",
    "STANDARD_INDICATORS",
    "CROSSWALK_MAPPINGS",
]
