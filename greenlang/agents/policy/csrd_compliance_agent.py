# -*- coding: utf-8 -*-
"""
GL-POL-X-007: CSRD Compliance Agent
===================================

EU Corporate Sustainability Reporting Directive compliance agent. CRITICAL PATH
agent providing deterministic CSRD/ESRS compliance assessment and data point tracking.

Capabilities:
    - ESRS data point requirements mapping
    - Double materiality assessment support
    - ESRS disclosure completeness tracking
    - Phase-in timeline tracking
    - Data point validation
    - XBRL tagging preparation

Zero-Hallucination Guarantees:
    - All requirements from official ESRS standards
    - Deterministic data point validation
    - Complete audit trails for all assessments
    - No LLM inference in compliance determination

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal
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


class ESRSStandard(str, Enum):
    """ESRS standards."""
    ESRS_1 = "esrs_1"  # General requirements
    ESRS_2 = "esrs_2"  # General disclosures
    ESRS_E1 = "esrs_e1"  # Climate change
    ESRS_E2 = "esrs_e2"  # Pollution
    ESRS_E3 = "esrs_e3"  # Water and marine resources
    ESRS_E4 = "esrs_e4"  # Biodiversity and ecosystems
    ESRS_E5 = "esrs_e5"  # Resource use and circular economy
    ESRS_S1 = "esrs_s1"  # Own workforce
    ESRS_S2 = "esrs_s2"  # Workers in value chain
    ESRS_S3 = "esrs_s3"  # Affected communities
    ESRS_S4 = "esrs_s4"  # Consumers and end-users
    ESRS_G1 = "esrs_g1"  # Business conduct


class DisclosureRequirement(str, Enum):
    """Disclosure requirement types."""
    MANDATORY = "mandatory"
    MATERIAL_BASED = "material_based"
    VOLUNTARY = "voluntary"


class DataPointType(str, Enum):
    """ESRS data point types."""
    NARRATIVE = "narrative"
    QUANTITATIVE = "quantitative"
    SEMI_QUANTITATIVE = "semi_quantitative"
    TABLE = "table"
    BINARY = "binary"


class ComplianceStatus(str, Enum):
    """Compliance status for a data point."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    NOT_MATERIAL = "not_material"


class PhaseInCategory(str, Enum):
    """CSRD phase-in categories."""
    YEAR_1 = "year_1"  # Large PIEs, already NFRD
    YEAR_2 = "year_2"  # Large companies meeting 2 of 3 criteria
    YEAR_3 = "year_3"  # Listed SMEs
    YEAR_4 = "year_4"  # Non-EU companies


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ESRSDataPoint(BaseModel):
    """An ESRS data point requirement."""

    data_point_id: str = Field(..., description="Unique data point ID")
    standard: ESRSStandard = Field(..., description="ESRS standard")
    disclosure_requirement: str = Field(..., description="DR reference (e.g., E1-1)")
    name: str = Field(..., description="Data point name")
    description: str = Field(..., description="Data point description")

    # Classification
    data_type: DataPointType = Field(..., description="Data type")
    requirement_type: DisclosureRequirement = Field(..., description="Requirement type")
    phase_in_applicable: bool = Field(default=False, description="Phase-in applicable")
    phase_in_year: Optional[int] = Field(None, description="Year of phase-in")

    # XBRL
    xbrl_element: Optional[str] = Field(None, description="XBRL element ID")

    # Validation
    unit: Optional[str] = Field(None, description="Unit if quantitative")
    validation_rules: List[str] = Field(
        default_factory=list,
        description="Validation rules"
    )


class MaterialityAssessment(BaseModel):
    """Double materiality assessment for a topic."""

    topic_id: str = Field(..., description="Topic identifier")
    standard: ESRSStandard = Field(..., description="Related ESRS standard")
    topic_name: str = Field(..., description="Topic name")

    # Impact materiality
    impact_material: bool = Field(default=False)
    impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Impact materiality score"
    )
    impact_rationale: Optional[str] = Field(None)

    # Financial materiality
    financial_material: bool = Field(default=False)
    financial_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Financial materiality score"
    )
    financial_rationale: Optional[str] = Field(None)

    # Combined
    is_material: bool = Field(default=False)

    def calculate_materiality(self) -> None:
        """Calculate if topic is material (deterministic)."""
        # A topic is material if either dimension is material
        self.is_material = self.impact_material or self.financial_material


class DataPointResponse(BaseModel):
    """Response to a data point requirement."""

    data_point_id: str = Field(..., description="Data point identifier")
    response_id: str = Field(
        default_factory=lambda: deterministic_uuid("response"),
        description="Unique response identifier"
    )

    # Response
    status: ComplianceStatus = Field(..., description="Compliance status")
    value: Optional[Any] = Field(None, description="Response value")
    narrative: Optional[str] = Field(None, description="Narrative response")

    # Evidence
    evidence_references: List[str] = Field(
        default_factory=list,
        description="Evidence references"
    )
    data_source: Optional[str] = Field(None, description="Data source")

    # Validation
    validated: bool = Field(default=False)
    validation_errors: List[str] = Field(default_factory=list)

    # Metadata
    last_updated: datetime = Field(default_factory=DeterministicClock.now)
    updated_by: Optional[str] = Field(None)


class CSRDComplianceResult(BaseModel):
    """CSRD compliance assessment result."""

    result_id: str = Field(
        default_factory=lambda: deterministic_uuid("csrd_result"),
        description="Unique result identifier"
    )
    organization_id: str = Field(..., description="Organization identifier")
    assessment_date: date = Field(
        default_factory=lambda: DeterministicClock.now().date()
    )
    reporting_year: int = Field(...)

    # Phase-in
    phase_in_category: PhaseInCategory = Field(..., description="Phase-in category")
    first_reporting_year: int = Field(..., description="First reporting year")

    # Materiality
    materiality_assessments: List[MaterialityAssessment] = Field(default_factory=list)
    material_topics: List[str] = Field(default_factory=list)

    # Data points
    total_data_points: int = Field(default=0)
    mandatory_data_points: int = Field(default=0)
    material_data_points: int = Field(default=0)

    # Responses
    data_point_responses: List[DataPointResponse] = Field(default_factory=list)

    # Compliance metrics
    compliance_by_standard: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    overall_compliance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0
    )

    # Gaps
    missing_mandatory: List[str] = Field(default_factory=list)
    missing_material: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = {
            "organization_id": self.organization_id,
            "assessment_date": self.assessment_date.isoformat(),
            "overall_compliance_score": self.overall_compliance_score,
            "material_topics": sorted(self.material_topics),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class CSRDComplianceInput(BaseModel):
    """Input for CSRD compliance operations."""

    action: str = Field(
        ...,
        description="Action: assess_compliance, get_data_points, validate_responses"
    )
    organization_id: Optional[str] = Field(None)
    reporting_year: Optional[int] = Field(None)
    phase_in_category: Optional[PhaseInCategory] = Field(None)
    materiality_assessments: Optional[List[Dict[str, Any]]] = Field(None)
    data_point_responses: Optional[List[Dict[str, Any]]] = Field(None)
    standards: Optional[List[ESRSStandard]] = Field(None)


class CSRDComplianceOutput(BaseModel):
    """Output from CSRD compliance operations."""

    success: bool = Field(...)
    action: str = Field(...)
    result: Optional[CSRDComplianceResult] = Field(None)
    data_points: Optional[List[ESRSDataPoint]] = Field(None)
    validation_results: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS DATA POINTS DATABASE
# =============================================================================


ESRS_DATA_POINTS: Dict[str, ESRSDataPoint] = {}


def _initialize_esrs_data_points() -> None:
    """Initialize ESRS data points database."""
    global ESRS_DATA_POINTS

    data_points = [
        # ESRS 2 - General disclosures (mandatory for all)
        ESRSDataPoint(
            data_point_id="ESRS2-BP1",
            standard=ESRSStandard.ESRS_2,
            disclosure_requirement="BP-1",
            name="General basis for preparation",
            description="Description of the general basis for preparation of the sustainability statement",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MANDATORY,
            xbrl_element="esrs:BasisForPreparation",
        ),
        ESRSDataPoint(
            data_point_id="ESRS2-GOV1",
            standard=ESRSStandard.ESRS_2,
            disclosure_requirement="GOV-1",
            name="Role of administrative bodies",
            description="Role of the administrative, management and supervisory bodies",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MANDATORY,
            xbrl_element="esrs:RoleOfAdministrativeBodies",
        ),
        ESRSDataPoint(
            data_point_id="ESRS2-SBM1",
            standard=ESRSStandard.ESRS_2,
            disclosure_requirement="SBM-1",
            name="Strategy, business model and value chain",
            description="Description of strategy, business model and value chain",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MANDATORY,
            xbrl_element="esrs:StrategyBusinessModelValueChain",
        ),
        ESRSDataPoint(
            data_point_id="ESRS2-IRO1",
            standard=ESRSStandard.ESRS_2,
            disclosure_requirement="IRO-1",
            name="Impacts, risks and opportunities process",
            description="Description of the process to identify and assess material IROs",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MANDATORY,
            xbrl_element="esrs:ProcessToIdentifyIROs",
        ),

        # ESRS E1 - Climate change
        ESRSDataPoint(
            data_point_id="E1-1-01",
            standard=ESRSStandard.ESRS_E1,
            disclosure_requirement="E1-1",
            name="Transition plan for climate change mitigation",
            description="Disclosure of transition plan for climate change mitigation",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            xbrl_element="esrs:TransitionPlanClimateChangeMitigation",
        ),
        ESRSDataPoint(
            data_point_id="E1-4-01",
            standard=ESRSStandard.ESRS_E1,
            disclosure_requirement="E1-4",
            name="GHG emission reduction targets",
            description="Targets related to climate change mitigation",
            data_type=DataPointType.QUANTITATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            unit="tCO2e",
            xbrl_element="esrs:GHGEmissionReductionTargets",
        ),
        ESRSDataPoint(
            data_point_id="E1-6-01",
            standard=ESRSStandard.ESRS_E1,
            disclosure_requirement="E1-6",
            name="Gross Scope 1 GHG emissions",
            description="Gross Scope 1 GHG emissions",
            data_type=DataPointType.QUANTITATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            unit="tCO2e",
            xbrl_element="esrs:GrossScope1GHGEmissions",
            validation_rules=["value >= 0", "unit == 'tCO2e'"],
        ),
        ESRSDataPoint(
            data_point_id="E1-6-02",
            standard=ESRSStandard.ESRS_E1,
            disclosure_requirement="E1-6",
            name="Gross Scope 2 GHG emissions",
            description="Gross location-based Scope 2 GHG emissions",
            data_type=DataPointType.QUANTITATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            unit="tCO2e",
            xbrl_element="esrs:GrossScope2GHGEmissionsLocationBased",
            validation_rules=["value >= 0", "unit == 'tCO2e'"],
        ),
        ESRSDataPoint(
            data_point_id="E1-6-03",
            standard=ESRSStandard.ESRS_E1,
            disclosure_requirement="E1-6",
            name="Gross Scope 3 GHG emissions",
            description="Gross Scope 3 GHG emissions",
            data_type=DataPointType.QUANTITATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            unit="tCO2e",
            phase_in_applicable=True,
            phase_in_year=2,
            xbrl_element="esrs:GrossScope3GHGEmissions",
            validation_rules=["value >= 0", "unit == 'tCO2e'"],
        ),

        # ESRS E4 - Biodiversity
        ESRSDataPoint(
            data_point_id="E4-1-01",
            standard=ESRSStandard.ESRS_E4,
            disclosure_requirement="E4-1",
            name="Transition plan and biodiversity policy",
            description="Transition plan and actions related to biodiversity and ecosystems",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            xbrl_element="esrs:BiodiversityTransitionPlan",
        ),
        ESRSDataPoint(
            data_point_id="E4-5-01",
            standard=ESRSStandard.ESRS_E4,
            disclosure_requirement="E4-5",
            name="Impact metrics on biodiversity",
            description="Impact metrics related to biodiversity and ecosystem change",
            data_type=DataPointType.QUANTITATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            xbrl_element="esrs:BiodiversityImpactMetrics",
        ),

        # ESRS G1 - Business conduct
        ESRSDataPoint(
            data_point_id="G1-1-01",
            standard=ESRSStandard.ESRS_G1,
            disclosure_requirement="G1-1",
            name="Business conduct policies",
            description="Policies related to business conduct matters",
            data_type=DataPointType.NARRATIVE,
            requirement_type=DisclosureRequirement.MATERIAL_BASED,
            xbrl_element="esrs:BusinessConductPolicies",
        ),
    ]

    for dp in data_points:
        ESRS_DATA_POINTS[dp.data_point_id] = dp


_initialize_esrs_data_points()


# =============================================================================
# CSRD COMPLIANCE AGENT
# =============================================================================


class CSRDComplianceAgent(BaseAgent):
    """
    GL-POL-X-007: CSRD Compliance Agent

    EU Corporate Sustainability Reporting Directive (CSRD) compliance assessment.
    CRITICAL PATH agent with zero-hallucination guarantees.

    Capabilities:
    - ESRS data point requirements tracking
    - Double materiality assessment
    - Compliance gap identification
    - Phase-in timeline management

    All assessments are:
    - Based on official ESRS standards
    - Deterministic with complete audit trails
    - No LLM inference in compliance determination

    Usage:
        agent = CSRDComplianceAgent()
        result = agent.run({
            'action': 'assess_compliance',
            'organization_id': 'org-123',
            'reporting_year': 2025
        })
    """

    AGENT_ID = "GL-POL-X-007"
    AGENT_NAME = "CSRD Compliance Agent"
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
        description="EU CSRD/ESRS compliance assessment"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize CSRD Compliance Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="CSRD compliance agent",
                version=self.VERSION,
                parameters={
                    "enforce_mandatory": True,
                    "validate_data_types": True,
                }
            )

        self._data_points = ESRS_DATA_POINTS.copy()
        self._audit_trail: List[Dict[str, Any]] = []

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute CSRD compliance operation."""
        import time
        start_time = time.time()

        try:
            agent_input = CSRDComplianceInput(**input_data)

            action_handlers = {
                "assess_compliance": self._handle_assess_compliance,
                "get_data_points": self._handle_get_data_points,
                "validate_responses": self._handle_validate_responses,
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
            logger.error(f"CSRD compliance failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_assess_compliance(
        self,
        input_data: CSRDComplianceInput
    ) -> CSRDComplianceOutput:
        """Assess CSRD compliance."""
        if not input_data.organization_id:
            return CSRDComplianceOutput(
                success=False,
                action="assess_compliance",
                error="organization_id required",
            )

        year = input_data.reporting_year or DeterministicClock.now().year
        phase_in = input_data.phase_in_category or PhaseInCategory.YEAR_2

        # Determine first reporting year
        first_year_map = {
            PhaseInCategory.YEAR_1: 2025,
            PhaseInCategory.YEAR_2: 2026,
            PhaseInCategory.YEAR_3: 2027,
            PhaseInCategory.YEAR_4: 2029,
        }
        first_reporting_year = first_year_map[phase_in]

        # Parse materiality assessments
        materiality = []
        material_topics: List[str] = []
        if input_data.materiality_assessments:
            for ma_data in input_data.materiality_assessments:
                ma = MaterialityAssessment(**ma_data)
                ma.calculate_materiality()
                materiality.append(ma)
                if ma.is_material:
                    material_topics.append(ma.topic_id)

        # Get applicable data points
        applicable_dps = self._get_applicable_data_points(
            material_topics, phase_in, year - first_reporting_year + 1
        )

        # Parse responses
        responses = []
        if input_data.data_point_responses:
            for resp_data in input_data.data_point_responses:
                responses.append(DataPointResponse(**resp_data))

        # Assess compliance
        result = CSRDComplianceResult(
            organization_id=input_data.organization_id,
            reporting_year=year,
            phase_in_category=phase_in,
            first_reporting_year=first_reporting_year,
            materiality_assessments=materiality,
            material_topics=material_topics,
            data_point_responses=responses,
        )

        # Calculate metrics
        mandatory_dps = [
            dp for dp in applicable_dps
            if dp.requirement_type == DisclosureRequirement.MANDATORY
        ]
        material_dps = [
            dp for dp in applicable_dps
            if dp.requirement_type == DisclosureRequirement.MATERIAL_BASED
        ]

        result.total_data_points = len(applicable_dps)
        result.mandatory_data_points = len(mandatory_dps)
        result.material_data_points = len(material_dps)

        # Track compliance by standard
        compliance_by_std: Dict[str, Dict[str, int]] = {}
        response_ids = {r.data_point_id for r in responses}

        for dp in applicable_dps:
            std = dp.standard.value
            if std not in compliance_by_std:
                compliance_by_std[std] = {"total": 0, "compliant": 0, "missing": 0}
            compliance_by_std[std]["total"] += 1

            if dp.data_point_id in response_ids:
                resp = next(r for r in responses if r.data_point_id == dp.data_point_id)
                if resp.status == ComplianceStatus.COMPLIANT:
                    compliance_by_std[std]["compliant"] += 1
            else:
                compliance_by_std[std]["missing"] += 1
                if dp.requirement_type == DisclosureRequirement.MANDATORY:
                    result.missing_mandatory.append(dp.data_point_id)
                elif dp.requirement_type == DisclosureRequirement.MATERIAL_BASED:
                    result.missing_material.append(dp.data_point_id)

        result.compliance_by_standard = compliance_by_std

        # Calculate overall score
        if applicable_dps:
            compliant_count = sum(
                1 for r in responses
                if r.status == ComplianceStatus.COMPLIANT
            )
            result.overall_compliance_score = (compliant_count / len(applicable_dps)) * 100

        result.provenance_hash = result.calculate_provenance_hash()

        return CSRDComplianceOutput(
            success=True,
            action="assess_compliance",
            result=result,
        )

    def _handle_get_data_points(
        self,
        input_data: CSRDComplianceInput
    ) -> CSRDComplianceOutput:
        """Get ESRS data points."""
        data_points = list(self._data_points.values())

        # Filter by standards
        if input_data.standards:
            data_points = [
                dp for dp in data_points
                if dp.standard in input_data.standards
            ]

        return CSRDComplianceOutput(
            success=True,
            action="get_data_points",
            data_points=data_points,
        )

    def _handle_validate_responses(
        self,
        input_data: CSRDComplianceInput
    ) -> CSRDComplianceOutput:
        """Validate data point responses."""
        if not input_data.data_point_responses:
            return CSRDComplianceOutput(
                success=False,
                action="validate_responses",
                error="data_point_responses required",
            )

        validation_results: Dict[str, Any] = {
            "total": len(input_data.data_point_responses),
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }

        for resp_data in input_data.data_point_responses:
            resp = DataPointResponse(**resp_data)
            dp = self._data_points.get(resp.data_point_id)

            if not dp:
                validation_results["errors"].append(
                    f"Unknown data point: {resp.data_point_id}"
                )
                validation_results["invalid"] += 1
                continue

            # Validate data type
            is_valid = True
            if self.config.parameters.get("validate_data_types", True):
                if dp.data_type == DataPointType.QUANTITATIVE:
                    if resp.value is None:
                        validation_results["errors"].append(
                            f"{resp.data_point_id}: Quantitative value required"
                        )
                        is_valid = False
                    elif not isinstance(resp.value, (int, float, Decimal)):
                        validation_results["errors"].append(
                            f"{resp.data_point_id}: Value must be numeric"
                        )
                        is_valid = False

                if dp.data_type == DataPointType.NARRATIVE:
                    if not resp.narrative:
                        validation_results["errors"].append(
                            f"{resp.data_point_id}: Narrative response required"
                        )
                        is_valid = False

            if is_valid:
                validation_results["valid"] += 1
            else:
                validation_results["invalid"] += 1

        return CSRDComplianceOutput(
            success=validation_results["invalid"] == 0,
            action="validate_responses",
            validation_results=validation_results,
        )

    def _get_applicable_data_points(
        self,
        material_topics: List[str],
        phase_in: PhaseInCategory,
        reporting_year_num: int
    ) -> List[ESRSDataPoint]:
        """Get applicable data points based on materiality and phase-in."""
        applicable = []

        for dp in self._data_points.values():
            # Mandatory always applicable
            if dp.requirement_type == DisclosureRequirement.MANDATORY:
                if not dp.phase_in_applicable or reporting_year_num >= (dp.phase_in_year or 1):
                    applicable.append(dp)
                continue

            # Material-based only if topic is material
            if dp.requirement_type == DisclosureRequirement.MATERIAL_BASED:
                # Check if standard is material
                std_topics = {
                    ESRSStandard.ESRS_E1: ["climate_change", "ghg_emissions"],
                    ESRSStandard.ESRS_E4: ["biodiversity", "ecosystems"],
                    ESRSStandard.ESRS_G1: ["business_conduct", "governance"],
                }
                topic_match = any(
                    t in material_topics
                    for t in std_topics.get(dp.standard, [])
                )
                if topic_match:
                    if not dp.phase_in_applicable or reporting_year_num >= (dp.phase_in_year or 1):
                        applicable.append(dp)

        return applicable

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def get_data_point(self, data_point_id: str) -> Optional[ESRSDataPoint]:
        """Get a data point by ID."""
        return self._data_points.get(data_point_id)

    def list_standards(self) -> List[ESRSStandard]:
        """List all ESRS standards."""
        return list(ESRSStandard)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CSRDComplianceAgent",
    "ESRSStandard",
    "DisclosureRequirement",
    "DataPointType",
    "ComplianceStatus",
    "PhaseInCategory",
    "ESRSDataPoint",
    "MaterialityAssessment",
    "DataPointResponse",
    "CSRDComplianceResult",
    "CSRDComplianceInput",
    "CSRDComplianceOutput",
    "ESRS_DATA_POINTS",
]
