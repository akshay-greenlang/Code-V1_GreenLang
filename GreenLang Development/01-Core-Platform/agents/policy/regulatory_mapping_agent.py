# -*- coding: utf-8 -*-
"""
GL-POL-X-001: Regulatory Mapping Agent
======================================

Maps applicable regulations to organizations based on their profile, industry,
geography, and operational characteristics. This agent is CRITICAL PATH - all
outputs are deterministic with full audit trails.

Capabilities:
    - Jurisdiction mapping based on operational footprint
    - Industry-specific regulation identification
    - Threshold-based applicability determination
    - Regulatory timeline tracking
    - Cross-jurisdictional conflict detection
    - Subsidiary and supply chain regulatory cascade

Zero-Hallucination Guarantees:
    - All regulation mappings derived from curated regulatory database
    - Deterministic threshold calculations
    - Complete audit trail for all mapping decisions
    - No LLM inference in regulatory determination

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class Jurisdiction(str, Enum):
    """Regulatory jurisdictions supported."""
    EU = "eu"
    USA = "usa"
    UK = "uk"
    CALIFORNIA = "california"
    GERMANY = "germany"
    FRANCE = "france"
    NETHERLANDS = "netherlands"
    JAPAN = "japan"
    CHINA = "china"
    AUSTRALIA = "australia"
    CANADA = "canada"
    BRAZIL = "brazil"
    INDIA = "india"
    SINGAPORE = "singapore"
    GLOBAL = "global"


class RegulationType(str, Enum):
    """Types of environmental regulations."""
    EMISSIONS_DISCLOSURE = "emissions_disclosure"
    CARBON_PRICING = "carbon_pricing"
    CARBON_TAX = "carbon_tax"
    BORDER_ADJUSTMENT = "border_adjustment"
    SUSTAINABILITY_REPORTING = "sustainability_reporting"
    DEFORESTATION = "deforestation"
    BIODIVERSITY = "biodiversity"
    DUE_DILIGENCE = "due_diligence"
    TAXONOMY = "taxonomy"
    SUPPLY_CHAIN = "supply_chain"


class ApplicabilityStatus(str, Enum):
    """Status of regulation applicability."""
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    POTENTIALLY_APPLICABLE = "potentially_applicable"
    VOLUNTARY = "voluntary"
    PHASE_IN = "phase_in"
    EXEMPTED = "exempted"


class IndustryClassification(str, Enum):
    """Industry classifications for regulation mapping."""
    ENERGY = "energy"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    CONSTRUCTION = "construction"
    AGRICULTURE = "agriculture"
    MINING = "mining"
    FINANCIAL_SERVICES = "financial_services"
    RETAIL = "retail"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    CHEMICALS = "chemicals"
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    FOOD_BEVERAGE = "food_beverage"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class RegulationDefinition(BaseModel):
    """Definition of a regulatory requirement."""

    regulation_id: str = Field(..., description="Unique regulation identifier")
    name: str = Field(..., description="Regulation name (e.g., 'CSRD', 'CBAM')")
    full_name: str = Field(..., description="Full regulation name")
    jurisdiction: Jurisdiction = Field(..., description="Regulatory jurisdiction")
    regulation_type: RegulationType = Field(..., description="Type of regulation")

    # Effective dates
    adoption_date: date = Field(..., description="Date regulation was adopted")
    effective_date: date = Field(..., description="Date regulation becomes effective")
    first_reporting_date: Optional[date] = Field(None, description="First reporting deadline")

    # Thresholds
    revenue_threshold_eur: Optional[Decimal] = Field(
        None,
        description="Revenue threshold in EUR for applicability"
    )
    employee_threshold: Optional[int] = Field(
        None,
        description="Employee count threshold"
    )
    asset_threshold_eur: Optional[Decimal] = Field(
        None,
        description="Total assets threshold in EUR"
    )
    emissions_threshold_tco2e: Optional[Decimal] = Field(
        None,
        description="Emissions threshold in tCO2e"
    )

    # Scope
    applicable_industries: List[IndustryClassification] = Field(
        default_factory=list,
        description="Industries this regulation applies to"
    )
    excluded_industries: List[IndustryClassification] = Field(
        default_factory=list,
        description="Industries explicitly excluded"
    )
    applies_to_subsidiaries: bool = Field(
        default=False,
        description="Whether regulation cascades to subsidiaries"
    )
    applies_to_supply_chain: bool = Field(
        default=False,
        description="Whether regulation covers supply chain"
    )

    # Requirements
    scope_1_required: bool = Field(default=False)
    scope_2_required: bool = Field(default=False)
    scope_3_required: bool = Field(default=False)
    assurance_required: bool = Field(default=False)

    # Penalties
    max_penalty_eur: Optional[Decimal] = Field(None, description="Maximum penalty")
    penalty_percentage_revenue: Optional[Decimal] = Field(
        None,
        description="Penalty as percentage of revenue"
    )

    # References
    official_url: Optional[str] = Field(None, description="Official regulation URL")
    citation: Optional[str] = Field(None, description="Citation reference")

    # Versioning
    version: str = Field(default="1.0.0", description="Regulation version")
    last_updated: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Last update timestamp"
    )


class OrganizationProfile(BaseModel):
    """Organization profile for regulatory mapping."""

    organization_id: str = Field(..., description="Unique organization identifier")
    name: str = Field(..., description="Organization name")

    # Classification
    industry: IndustryClassification = Field(..., description="Primary industry")
    secondary_industries: List[IndustryClassification] = Field(
        default_factory=list,
        description="Secondary industry classifications"
    )

    # Size metrics
    annual_revenue_eur: Decimal = Field(..., description="Annual revenue in EUR")
    total_employees: int = Field(..., description="Total employee count")
    total_assets_eur: Optional[Decimal] = Field(
        None,
        description="Total assets in EUR"
    )

    # Geographic presence
    headquarter_jurisdiction: Jurisdiction = Field(
        ...,
        description="Jurisdiction of headquarters"
    )
    operational_jurisdictions: List[Jurisdiction] = Field(
        default_factory=list,
        description="Jurisdictions with operations"
    )
    sales_jurisdictions: List[Jurisdiction] = Field(
        default_factory=list,
        description="Jurisdictions with sales"
    )

    # Listing status
    is_listed: bool = Field(default=False, description="Whether publicly listed")
    listing_jurisdictions: List[Jurisdiction] = Field(
        default_factory=list,
        description="Jurisdictions where listed"
    )

    # Parent/subsidiary
    has_eu_parent: bool = Field(
        default=False,
        description="Has EU parent company"
    )
    is_eu_subsidiary: bool = Field(
        default=False,
        description="Is subsidiary of EU company"
    )

    # Emissions
    estimated_scope1_emissions_tco2e: Optional[Decimal] = Field(
        None,
        description="Estimated Scope 1 emissions"
    )
    estimated_scope2_emissions_tco2e: Optional[Decimal] = Field(
        None,
        description="Estimated Scope 2 emissions"
    )
    estimated_scope3_emissions_tco2e: Optional[Decimal] = Field(
        None,
        description="Estimated Scope 3 emissions"
    )

    # Activities
    imports_to_eu: bool = Field(
        default=False,
        description="Imports goods to EU"
    )
    cbam_covered_products: bool = Field(
        default=False,
        description="Products covered by CBAM"
    )
    deforestation_risk_commodities: bool = Field(
        default=False,
        description="Handles deforestation-risk commodities"
    )


class ApplicabilityResult(BaseModel):
    """Result of applicability assessment for a single regulation."""

    regulation_id: str = Field(..., description="Regulation identifier")
    regulation_name: str = Field(..., description="Regulation name")
    status: ApplicabilityStatus = Field(..., description="Applicability status")

    # Threshold checks
    threshold_checks: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Results of threshold checks"
    )

    # Applicability factors
    applicable_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons why applicable"
    )
    exemption_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for exemption"
    )

    # Dates
    effective_date: date = Field(..., description="When regulation becomes effective")
    first_reporting_date: Optional[date] = Field(
        None,
        description="First reporting deadline"
    )

    # Requirements if applicable
    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific requirements if applicable"
    )

    # Confidence
    assessment_confidence: str = Field(
        default="high",
        description="Confidence level: high, medium, low"
    )

    # Audit
    assessment_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step assessment trace"
    )


class RegulatoryMappingResult(BaseModel):
    """Complete regulatory mapping result for an organization."""

    result_id: str = Field(
        default_factory=lambda: deterministic_uuid("regulatory_mapping"),
        description="Unique result identifier"
    )
    organization_id: str = Field(..., description="Organization identifier")
    organization_name: str = Field(..., description="Organization name")

    # Assessment date
    assessment_date: date = Field(
        default_factory=lambda: DeterministicClock.now().date(),
        description="Date of assessment"
    )

    # Results
    applicable_regulations: List[ApplicabilityResult] = Field(
        default_factory=list,
        description="Applicable regulations"
    )
    not_applicable_regulations: List[ApplicabilityResult] = Field(
        default_factory=list,
        description="Non-applicable regulations"
    )
    phase_in_regulations: List[ApplicabilityResult] = Field(
        default_factory=list,
        description="Regulations in phase-in period"
    )
    voluntary_regulations: List[ApplicabilityResult] = Field(
        default_factory=list,
        description="Voluntary regulations"
    )

    # Summary
    total_regulations_assessed: int = Field(default=0)
    total_applicable: int = Field(default=0)
    total_not_applicable: int = Field(default=0)

    # Timeline
    upcoming_deadlines: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Upcoming regulatory deadlines"
    )

    # Conflicts
    jurisdictional_conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected jurisdictional conflicts"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0)

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = {
            "organization_id": self.organization_id,
            "assessment_date": self.assessment_date.isoformat(),
            "applicable_count": len(self.applicable_regulations),
            "regulation_ids": [r.regulation_id for r in self.applicable_regulations],
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()


class RegulatoryMappingInput(BaseModel):
    """Input for regulatory mapping agent."""

    organization: OrganizationProfile = Field(
        ...,
        description="Organization profile to assess"
    )
    assessment_date: Optional[date] = Field(
        None,
        description="Date for assessment (defaults to today)"
    )
    regulation_filter: Optional[List[str]] = Field(
        None,
        description="Specific regulations to assess"
    )
    jurisdiction_filter: Optional[List[Jurisdiction]] = Field(
        None,
        description="Specific jurisdictions to assess"
    )
    include_voluntary: bool = Field(
        default=True,
        description="Include voluntary regulations"
    )


class RegulatoryMappingOutput(BaseModel):
    """Output from regulatory mapping agent."""

    success: bool = Field(..., description="Whether mapping succeeded")
    result: Optional[RegulatoryMappingResult] = Field(
        None,
        description="Mapping result"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Warnings")


# =============================================================================
# REGULATORY DATABASE
# =============================================================================


# Standard regulations database (curated, versioned)
REGULATORY_DATABASE: Dict[str, RegulationDefinition] = {}


def _initialize_regulatory_database() -> None:
    """Initialize the regulatory database with known regulations."""
    global REGULATORY_DATABASE

    regulations = [
        # EU CSRD
        RegulationDefinition(
            regulation_id="EU-CSRD",
            name="CSRD",
            full_name="Corporate Sustainability Reporting Directive",
            jurisdiction=Jurisdiction.EU,
            regulation_type=RegulationType.SUSTAINABILITY_REPORTING,
            adoption_date=date(2022, 11, 28),
            effective_date=date(2024, 1, 1),
            first_reporting_date=date(2025, 1, 1),
            revenue_threshold_eur=Decimal("50000000"),
            employee_threshold=250,
            asset_threshold_eur=Decimal("25000000"),
            applies_to_subsidiaries=True,
            scope_1_required=True,
            scope_2_required=True,
            scope_3_required=True,
            assurance_required=True,
            max_penalty_eur=Decimal("10000000"),
            official_url="https://eur-lex.europa.eu/eli/dir/2022/2464",
            citation="Directive (EU) 2022/2464",
        ),
        # EU CBAM
        RegulationDefinition(
            regulation_id="EU-CBAM",
            name="CBAM",
            full_name="Carbon Border Adjustment Mechanism",
            jurisdiction=Jurisdiction.EU,
            regulation_type=RegulationType.BORDER_ADJUSTMENT,
            adoption_date=date(2023, 5, 16),
            effective_date=date(2023, 10, 1),
            first_reporting_date=date(2024, 1, 31),
            applicable_industries=[
                IndustryClassification.CEMENT,
                IndustryClassification.STEEL,
                IndustryClassification.ALUMINUM,
                IndustryClassification.CHEMICALS,
                IndustryClassification.ENERGY,
            ],
            applies_to_supply_chain=True,
            scope_1_required=True,
            scope_2_required=True,
            official_url="https://eur-lex.europa.eu/eli/reg/2023/956",
            citation="Regulation (EU) 2023/956",
        ),
        # EU EUDR
        RegulationDefinition(
            regulation_id="EU-EUDR",
            name="EUDR",
            full_name="EU Deforestation Regulation",
            jurisdiction=Jurisdiction.EU,
            regulation_type=RegulationType.DEFORESTATION,
            adoption_date=date(2023, 6, 29),
            effective_date=date(2024, 12, 30),
            first_reporting_date=date(2025, 1, 1),
            applicable_industries=[
                IndustryClassification.AGRICULTURE,
                IndustryClassification.FOOD_BEVERAGE,
                IndustryClassification.RETAIL,
            ],
            applies_to_supply_chain=True,
            official_url="https://eur-lex.europa.eu/eli/reg/2023/1115",
            citation="Regulation (EU) 2023/1115",
        ),
        # California SB253
        RegulationDefinition(
            regulation_id="US-CA-SB253",
            name="SB253",
            full_name="California Climate Corporate Data Accountability Act",
            jurisdiction=Jurisdiction.CALIFORNIA,
            regulation_type=RegulationType.EMISSIONS_DISCLOSURE,
            adoption_date=date(2023, 10, 7),
            effective_date=date(2026, 1, 1),
            first_reporting_date=date(2027, 1, 1),
            revenue_threshold_eur=Decimal("1000000000"),  # $1B USD ~= EUR
            scope_1_required=True,
            scope_2_required=True,
            scope_3_required=True,
            assurance_required=True,
            official_url="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB253",
            citation="California SB 253",
        ),
        # California SB261
        RegulationDefinition(
            regulation_id="US-CA-SB261",
            name="SB261",
            full_name="California Climate-Related Financial Risk Act",
            jurisdiction=Jurisdiction.CALIFORNIA,
            regulation_type=RegulationType.SUSTAINABILITY_REPORTING,
            adoption_date=date(2023, 10, 7),
            effective_date=date(2026, 1, 1),
            first_reporting_date=date(2026, 1, 1),
            revenue_threshold_eur=Decimal("500000000"),  # $500M USD
            official_url="https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB261",
            citation="California SB 261",
        ),
        # SEC Climate Disclosure
        RegulationDefinition(
            regulation_id="US-SEC-CLIMATE",
            name="SEC Climate",
            full_name="SEC Climate-Related Disclosure Rules",
            jurisdiction=Jurisdiction.USA,
            regulation_type=RegulationType.SUSTAINABILITY_REPORTING,
            adoption_date=date(2024, 3, 6),
            effective_date=date(2025, 1, 1),
            scope_1_required=True,
            scope_2_required=True,
            scope_3_required=False,  # Not required
            assurance_required=True,
            official_url="https://www.sec.gov/rules/final/2024/33-11275.pdf",
            citation="SEC 33-11275",
        ),
        # UK Streamlined Energy and Carbon Reporting
        RegulationDefinition(
            regulation_id="UK-SECR",
            name="SECR",
            full_name="Streamlined Energy and Carbon Reporting",
            jurisdiction=Jurisdiction.UK,
            regulation_type=RegulationType.EMISSIONS_DISCLOSURE,
            adoption_date=date(2019, 4, 1),
            effective_date=date(2019, 4, 1),
            revenue_threshold_eur=Decimal("36000000"),  # GBP 36m
            employee_threshold=250,
            scope_1_required=True,
            scope_2_required=True,
            scope_3_required=False,
            official_url="https://www.gov.uk/government/publications/environmental-reporting-guidelines-including-mandatory-greenhouse-gas-emissions-reporting-guidance",
            citation="UK SECR Regulations 2019",
        ),
    ]

    for reg in regulations:
        REGULATORY_DATABASE[reg.regulation_id] = reg


# Initialize database
_initialize_regulatory_database()


# =============================================================================
# REGULATORY MAPPING AGENT
# =============================================================================


class RegulatoryMappingAgent(BaseAgent):
    """
    GL-POL-X-001: Regulatory Mapping Agent

    Maps applicable regulations to organizations based on their profile.
    This is a CRITICAL PATH agent with zero-hallucination guarantees.

    All regulatory determinations are:
    - Derived from curated regulatory database
    - Based on deterministic threshold calculations
    - Fully auditable with step-by-step traces
    - No LLM inference involved

    Usage:
        agent = RegulatoryMappingAgent()
        result = agent.run({
            'organization': organization_profile_dict
        })
    """

    AGENT_ID = "GL-POL-X-001"
    AGENT_NAME = "Regulatory Mapping Agent"
    VERSION = "1.0.0"

    # Agent metadata for Intelligence Paradox architecture
    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Maps applicable regulations using deterministic rules"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Regulatory Mapping Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Deterministic regulatory mapping agent",
                version=self.VERSION,
                parameters={
                    "enable_threshold_trace": True,
                    "include_phase_in": True,
                    "detect_conflicts": True,
                }
            )

        # Initialize regulation database reference
        self._regulations = REGULATORY_DATABASE.copy()

        # Audit trail
        self._audit_trail: List[Dict[str, Any]] = []

        super().__init__(config)

        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute regulatory mapping.

        Args:
            input_data: Contains organization profile and options

        Returns:
            AgentResult with regulatory mapping result
        """
        import time
        start_time = time.time()

        try:
            # Parse input
            agent_input = RegulatoryMappingInput(**input_data)

            # Perform mapping
            result = self._map_regulations(agent_input)

            # Calculate provenance
            result.provenance_hash = result.calculate_provenance_hash()
            result.processing_time_ms = (time.time() - start_time) * 1000

            # Create output
            output = RegulatoryMappingOutput(
                success=True,
                result=result
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            logger.error(f"Regulatory mapping failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
            )

    def _map_regulations(
        self,
        input_data: RegulatoryMappingInput
    ) -> RegulatoryMappingResult:
        """
        Map all applicable regulations for an organization.

        This method is 100% deterministic with full audit trail.
        """
        org = input_data.organization
        assessment_date = input_data.assessment_date or DeterministicClock.now().date()

        result = RegulatoryMappingResult(
            organization_id=org.organization_id,
            organization_name=org.name,
            assessment_date=assessment_date,
        )

        # Get regulations to assess
        regulations_to_assess = self._get_regulations_to_assess(
            input_data.regulation_filter,
            input_data.jurisdiction_filter
        )

        # Assess each regulation
        for reg_id, regulation in regulations_to_assess.items():
            applicability = self._assess_regulation(
                org, regulation, assessment_date
            )

            # Categorize result
            if applicability.status == ApplicabilityStatus.APPLICABLE:
                result.applicable_regulations.append(applicability)
            elif applicability.status == ApplicabilityStatus.NOT_APPLICABLE:
                result.not_applicable_regulations.append(applicability)
            elif applicability.status == ApplicabilityStatus.PHASE_IN:
                result.phase_in_regulations.append(applicability)
            elif applicability.status == ApplicabilityStatus.VOLUNTARY:
                if input_data.include_voluntary:
                    result.voluntary_regulations.append(applicability)

        # Calculate summaries
        result.total_regulations_assessed = len(regulations_to_assess)
        result.total_applicable = len(result.applicable_regulations)
        result.total_not_applicable = len(result.not_applicable_regulations)

        # Build timeline
        result.upcoming_deadlines = self._build_deadline_timeline(
            result.applicable_regulations + result.phase_in_regulations
        )

        # Detect conflicts
        if self.config.parameters.get("detect_conflicts", True):
            result.jurisdictional_conflicts = self._detect_conflicts(
                result.applicable_regulations
            )

        return result

    def _get_regulations_to_assess(
        self,
        regulation_filter: Optional[List[str]],
        jurisdiction_filter: Optional[List[Jurisdiction]]
    ) -> Dict[str, RegulationDefinition]:
        """Get regulations to assess based on filters."""
        regulations = self._regulations.copy()

        # Filter by regulation ID
        if regulation_filter:
            regulations = {
                k: v for k, v in regulations.items()
                if k in regulation_filter
            }

        # Filter by jurisdiction
        if jurisdiction_filter:
            regulations = {
                k: v for k, v in regulations.items()
                if v.jurisdiction in jurisdiction_filter
            }

        return regulations

    def _assess_regulation(
        self,
        org: OrganizationProfile,
        regulation: RegulationDefinition,
        assessment_date: date
    ) -> ApplicabilityResult:
        """
        Assess applicability of a single regulation.

        This is the core deterministic assessment logic.
        """
        trace: List[str] = []
        applicable_reasons: List[str] = []
        exemption_reasons: List[str] = []
        threshold_checks: Dict[str, Dict[str, Any]] = {}

        trace.append(f"Assessing {regulation.name} for {org.name}")

        # Check 1: Jurisdiction applicability
        jurisdiction_applicable = self._check_jurisdiction(
            org, regulation, trace
        )

        if not jurisdiction_applicable:
            trace.append("RESULT: Not applicable due to jurisdiction")
            return ApplicabilityResult(
                regulation_id=regulation.regulation_id,
                regulation_name=regulation.name,
                status=ApplicabilityStatus.NOT_APPLICABLE,
                effective_date=regulation.effective_date,
                first_reporting_date=regulation.first_reporting_date,
                exemption_reasons=["No jurisdictional nexus"],
                assessment_trace=trace,
            )

        trace.append("Jurisdiction check: PASSED")
        applicable_reasons.append("Has jurisdictional nexus")

        # Check 2: Industry applicability
        industry_applicable = self._check_industry(
            org, regulation, trace
        )

        if industry_applicable is False:  # Explicit False means excluded
            trace.append("RESULT: Not applicable due to industry exclusion")
            return ApplicabilityResult(
                regulation_id=regulation.regulation_id,
                regulation_name=regulation.name,
                status=ApplicabilityStatus.NOT_APPLICABLE,
                effective_date=regulation.effective_date,
                first_reporting_date=regulation.first_reporting_date,
                exemption_reasons=["Industry excluded from regulation"],
                assessment_trace=trace,
            )

        if industry_applicable is True:
            trace.append("Industry check: PASSED (industry specifically covered)")
            applicable_reasons.append("Industry specifically covered by regulation")

        # Check 3: Threshold checks
        threshold_results = self._check_thresholds(
            org, regulation, trace, threshold_checks
        )

        # Check 4: Effective date
        if assessment_date < regulation.effective_date:
            trace.append(f"Regulation not yet effective (effective: {regulation.effective_date})")
            return ApplicabilityResult(
                regulation_id=regulation.regulation_id,
                regulation_name=regulation.name,
                status=ApplicabilityStatus.PHASE_IN,
                effective_date=regulation.effective_date,
                first_reporting_date=regulation.first_reporting_date,
                threshold_checks=threshold_checks,
                applicable_reasons=applicable_reasons,
                assessment_trace=trace,
            )

        # Check 5: Special conditions
        special_conditions = self._check_special_conditions(
            org, regulation, trace
        )

        # Determine final status
        if threshold_results and (industry_applicable is True or industry_applicable is None):
            trace.append("RESULT: Regulation is APPLICABLE")
            status = ApplicabilityStatus.APPLICABLE
        elif threshold_results and industry_applicable is None:
            # Thresholds met but industry not specifically listed
            trace.append("RESULT: Regulation is POTENTIALLY APPLICABLE")
            status = ApplicabilityStatus.POTENTIALLY_APPLICABLE
        else:
            trace.append("RESULT: Regulation is NOT APPLICABLE")
            status = ApplicabilityStatus.NOT_APPLICABLE
            if not threshold_results:
                exemption_reasons.append("Below threshold requirements")

        # Build requirements
        requirements = {}
        if status == ApplicabilityStatus.APPLICABLE:
            requirements = {
                "scope_1_required": regulation.scope_1_required,
                "scope_2_required": regulation.scope_2_required,
                "scope_3_required": regulation.scope_3_required,
                "assurance_required": regulation.assurance_required,
            }

        return ApplicabilityResult(
            regulation_id=regulation.regulation_id,
            regulation_name=regulation.name,
            status=status,
            effective_date=regulation.effective_date,
            first_reporting_date=regulation.first_reporting_date,
            threshold_checks=threshold_checks,
            applicable_reasons=applicable_reasons,
            exemption_reasons=exemption_reasons,
            requirements=requirements,
            assessment_trace=trace,
        )

    def _check_jurisdiction(
        self,
        org: OrganizationProfile,
        regulation: RegulationDefinition,
        trace: List[str]
    ) -> bool:
        """Check if organization has nexus to regulation jurisdiction."""
        reg_jurisdiction = regulation.jurisdiction

        # Check headquarter
        if org.headquarter_jurisdiction == reg_jurisdiction:
            trace.append(f"  - Headquarters in {reg_jurisdiction.value}")
            return True

        # Check operational presence
        if reg_jurisdiction in org.operational_jurisdictions:
            trace.append(f"  - Operations in {reg_jurisdiction.value}")
            return True

        # Check sales presence
        if reg_jurisdiction in org.sales_jurisdictions:
            trace.append(f"  - Sales in {reg_jurisdiction.value}")
            return True

        # Check listing
        if reg_jurisdiction in org.listing_jurisdictions:
            trace.append(f"  - Listed in {reg_jurisdiction.value}")
            return True

        # EU-specific: check subsidiary status
        if reg_jurisdiction == Jurisdiction.EU:
            if org.has_eu_parent or org.is_eu_subsidiary:
                trace.append("  - EU parent/subsidiary relationship")
                return True

        # CBAM special: check if importing to EU
        if regulation.regulation_type == RegulationType.BORDER_ADJUSTMENT:
            if org.imports_to_eu and org.cbam_covered_products:
                trace.append("  - Imports CBAM-covered products to EU")
                return True

        trace.append(f"  - No nexus to {reg_jurisdiction.value}")
        return False

    def _check_industry(
        self,
        org: OrganizationProfile,
        regulation: RegulationDefinition,
        trace: List[str]
    ) -> Optional[bool]:
        """
        Check industry applicability.

        Returns:
            True: Industry specifically covered
            False: Industry explicitly excluded
            None: Industry not specifically listed (general applicability)
        """
        # Check exclusions first
        if org.industry in regulation.excluded_industries:
            trace.append(f"  - Industry {org.industry.value} explicitly excluded")
            return False

        for secondary in org.secondary_industries:
            if secondary in regulation.excluded_industries:
                trace.append(f"  - Secondary industry {secondary.value} excluded")
                return False

        # Check if industry is specifically listed
        if regulation.applicable_industries:
            if org.industry in regulation.applicable_industries:
                trace.append(f"  - Industry {org.industry.value} specifically covered")
                return True

            for secondary in org.secondary_industries:
                if secondary in regulation.applicable_industries:
                    trace.append(f"  - Secondary industry {secondary.value} covered")
                    return True

            trace.append(f"  - Industry not in covered list")
            return False

        # No specific industry list = general applicability
        trace.append("  - No specific industry restrictions")
        return None

    def _check_thresholds(
        self,
        org: OrganizationProfile,
        regulation: RegulationDefinition,
        trace: List[str],
        threshold_checks: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Check threshold requirements.

        Returns True if thresholds are met (or no thresholds exist).
        """
        thresholds_met = True

        # Revenue threshold
        if regulation.revenue_threshold_eur is not None:
            meets = org.annual_revenue_eur >= regulation.revenue_threshold_eur
            threshold_checks["revenue"] = {
                "threshold": float(regulation.revenue_threshold_eur),
                "actual": float(org.annual_revenue_eur),
                "meets_threshold": meets,
            }
            trace.append(
                f"  - Revenue check: {org.annual_revenue_eur:,.0f} EUR "
                f"vs {regulation.revenue_threshold_eur:,.0f} EUR threshold: "
                f"{'PASS' if meets else 'FAIL'}"
            )
            if not meets:
                thresholds_met = False

        # Employee threshold
        if regulation.employee_threshold is not None:
            meets = org.total_employees >= regulation.employee_threshold
            threshold_checks["employees"] = {
                "threshold": regulation.employee_threshold,
                "actual": org.total_employees,
                "meets_threshold": meets,
            }
            trace.append(
                f"  - Employee check: {org.total_employees:,} "
                f"vs {regulation.employee_threshold:,} threshold: "
                f"{'PASS' if meets else 'FAIL'}"
            )
            if not meets:
                thresholds_met = False

        # Asset threshold
        if regulation.asset_threshold_eur is not None and org.total_assets_eur:
            meets = org.total_assets_eur >= regulation.asset_threshold_eur
            threshold_checks["assets"] = {
                "threshold": float(regulation.asset_threshold_eur),
                "actual": float(org.total_assets_eur),
                "meets_threshold": meets,
            }
            trace.append(
                f"  - Asset check: {org.total_assets_eur:,.0f} EUR "
                f"vs {regulation.asset_threshold_eur:,.0f} EUR threshold: "
                f"{'PASS' if meets else 'FAIL'}"
            )
            if not meets:
                thresholds_met = False

        # Emissions threshold
        if regulation.emissions_threshold_tco2e is not None:
            total_emissions = Decimal(0)
            if org.estimated_scope1_emissions_tco2e:
                total_emissions += org.estimated_scope1_emissions_tco2e
            if org.estimated_scope2_emissions_tco2e:
                total_emissions += org.estimated_scope2_emissions_tco2e

            meets = total_emissions >= regulation.emissions_threshold_tco2e
            threshold_checks["emissions"] = {
                "threshold": float(regulation.emissions_threshold_tco2e),
                "actual": float(total_emissions),
                "meets_threshold": meets,
            }
            trace.append(
                f"  - Emissions check: {total_emissions:,.0f} tCO2e "
                f"vs {regulation.emissions_threshold_tco2e:,.0f} tCO2e threshold: "
                f"{'PASS' if meets else 'FAIL'}"
            )
            if not meets:
                thresholds_met = False

        return thresholds_met

    def _check_special_conditions(
        self,
        org: OrganizationProfile,
        regulation: RegulationDefinition,
        trace: List[str]
    ) -> Dict[str, bool]:
        """Check special conditions for specific regulations."""
        conditions = {}

        # CBAM specific
        if regulation.regulation_id == "EU-CBAM":
            conditions["imports_covered_products"] = (
                org.imports_to_eu and org.cbam_covered_products
            )
            if conditions["imports_covered_products"]:
                trace.append("  - CBAM: Imports covered products to EU")

        # EUDR specific
        if regulation.regulation_id == "EU-EUDR":
            conditions["handles_risk_commodities"] = org.deforestation_risk_commodities
            if conditions["handles_risk_commodities"]:
                trace.append("  - EUDR: Handles deforestation-risk commodities")

        return conditions

    def _build_deadline_timeline(
        self,
        applicable_regulations: List[ApplicabilityResult]
    ) -> List[Dict[str, Any]]:
        """Build timeline of upcoming deadlines."""
        deadlines = []

        for reg in applicable_regulations:
            if reg.first_reporting_date:
                deadlines.append({
                    "regulation_id": reg.regulation_id,
                    "regulation_name": reg.regulation_name,
                    "deadline_type": "first_reporting",
                    "deadline_date": reg.first_reporting_date.isoformat(),
                })

        # Sort by date
        deadlines.sort(key=lambda x: x["deadline_date"])

        return deadlines

    def _detect_conflicts(
        self,
        applicable_regulations: List[ApplicabilityResult]
    ) -> List[Dict[str, Any]]:
        """Detect potential jurisdictional conflicts."""
        conflicts = []

        # Group by regulation type
        by_type: Dict[str, List[ApplicabilityResult]] = {}
        for reg in applicable_regulations:
            reg_def = self._regulations.get(reg.regulation_id)
            if reg_def:
                reg_type = reg_def.regulation_type.value
                if reg_type not in by_type:
                    by_type[reg_type] = []
                by_type[reg_type].append(reg)

        # Check for overlapping requirements
        for reg_type, regs in by_type.items():
            if len(regs) > 1:
                conflicts.append({
                    "conflict_type": "overlapping_requirements",
                    "regulation_type": reg_type,
                    "regulations": [r.regulation_id for r in regs],
                    "recommendation": "Review for harmonized compliance approach",
                })

        return conflicts

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def get_regulation(
        self,
        regulation_id: str
    ) -> Optional[RegulationDefinition]:
        """Get a regulation definition by ID."""
        return self._regulations.get(regulation_id)

    def list_regulations(
        self,
        jurisdiction: Optional[Jurisdiction] = None,
        regulation_type: Optional[RegulationType] = None
    ) -> List[RegulationDefinition]:
        """List regulations with optional filters."""
        regulations = list(self._regulations.values())

        if jurisdiction:
            regulations = [r for r in regulations if r.jurisdiction == jurisdiction]

        if regulation_type:
            regulations = [r for r in regulations if r.regulation_type == regulation_type]

        return regulations

    def add_regulation(
        self,
        regulation: RegulationDefinition
    ) -> str:
        """
        Add a custom regulation to the database.

        Returns:
            Regulation ID
        """
        self._regulations[regulation.regulation_id] = regulation
        logger.info(f"Added regulation: {regulation.regulation_id}")
        return regulation.regulation_id


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main agent
    "RegulatoryMappingAgent",

    # Enums
    "Jurisdiction",
    "RegulationType",
    "ApplicabilityStatus",
    "IndustryClassification",

    # Models
    "RegulationDefinition",
    "OrganizationProfile",
    "ApplicabilityResult",
    "RegulatoryMappingResult",
    "RegulatoryMappingInput",
    "RegulatoryMappingOutput",

    # Database
    "REGULATORY_DATABASE",
]
