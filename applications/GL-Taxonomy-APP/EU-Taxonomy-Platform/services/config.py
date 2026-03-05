"""
GL-Taxonomy-APP v1.0 -- EU Taxonomy Alignment & Green Investment Ratio Configuration

Enumerations, constants, activity libraries, threshold databases, DNSH criteria
matrices, minimum safeguard checks, GAR exposure definitions, reporting template
specifications, and application settings for implementing the EU Taxonomy
Regulation (2020/852), associated Delegated Acts, and EBA Pillar 3 GAR/BTAR
requirements.

The EU Taxonomy establishes six environmental objectives and a four-step
alignment test: (1) eligibility screening, (2) substantial contribution to at
least one objective, (3) Do No Significant Harm to remaining objectives, and
(4) compliance with minimum safeguards.  Financial institutions additionally
compute the Green Asset Ratio (GAR) and Banking-Book Taxonomy Alignment Ratio
(BTAR) per EBA ITS.

All settings use the TAXONOMY_APP_ prefix for environment variable overrides.

Reference:
    - EU Taxonomy Regulation 2020/852 (June 2020)
    - Climate Delegated Act 2021/2139 (June 2021)
    - Complementary Climate Delegated Act 2022/1214 (March 2022)
    - Environmental Delegated Act 2023/2486 (June 2023)
    - Taxonomy Simplification Delegated Act 2025 (proposed)
    - EBA ITS on Pillar 3 ESG Disclosures (January 2022)
    - Article 8 Disclosures Delegated Act 2021/2178 (July 2021)
    - Platform on Sustainable Finance Technical Screening Criteria
    - OECD Guidelines for Multinational Enterprises (2023 update)
    - UN Guiding Principles on Business and Human Rights (2011)
    - ILO Declaration on Fundamental Principles and Rights at Work

Example:
    >>> config = TaxonomyAppConfig()
    >>> config.app_name
    'GL-Taxonomy-APP'
    >>> EnvironmentalObjective.CLIMATE_MITIGATION.value
    'climate_mitigation'
    >>> TAXONOMY_SECTORS[Sector.ENERGY]["nace_codes"]
    ['D35']
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EnvironmentalObjective(str, Enum):
    """
    Six EU Taxonomy environmental objectives per Article 9 of Regulation 2020/852.

    An economic activity must substantially contribute to at least one objective
    while doing no significant harm to the remaining five.
    """

    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER_MARINE = "water_marine"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY_ECOSYSTEMS = "biodiversity_ecosystems"


class ActivityType(str, Enum):
    """
    Classification of taxonomy-eligible economic activities.

    Own-performance activities directly contribute to an objective.
    Enabling activities enable other activities to contribute.
    Transitional activities are for sectors with no low-carbon alternative.
    """

    OWN_PERFORMANCE = "own_performance"
    ENABLING = "enabling"
    TRANSITIONAL = "transitional"


class AlignmentStatus(str, Enum):
    """Taxonomy alignment status for an economic activity or portfolio."""

    NOT_SCREENED = "not_screened"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    ALIGNED = "aligned"
    NOT_ELIGIBLE = "not_eligible"


class DNSHStatus(str, Enum):
    """Do No Significant Harm assessment outcome per objective."""

    NOT_ASSESSED = "not_assessed"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class SafeguardTopic(str, Enum):
    """
    Minimum safeguard topics per Article 18 of Regulation 2020/852.

    Based on OECD Guidelines, UN Guiding Principles, ILO Declaration,
    and the International Bill of Human Rights.
    """

    HUMAN_RIGHTS = "human_rights"
    ANTI_CORRUPTION = "anti_corruption"
    TAXATION = "taxation"
    FAIR_COMPETITION = "fair_competition"


class SafeguardTestType(str, Enum):
    """Type of minimum safeguard compliance test."""

    PROCEDURAL = "procedural"
    OUTCOME = "outcome"


class KPIType(str, Enum):
    """
    Non-financial undertaking KPI types per Article 8 Delegated Act.

    Turnover: share of revenue from taxonomy-aligned activities.
    CapEx: share of capital expenditure for taxonomy-aligned activities.
    OpEx: share of operating expenditure for taxonomy-aligned activities.
    """

    TURNOVER = "turnover"
    CAPEX = "capex"
    OPEX = "opex"


class GARType(str, Enum):
    """
    Green Asset Ratio type per EBA ITS.

    Stock: ratio based on outstanding portfolio at reference date.
    Flow: ratio based on new origination during reporting period.
    """

    STOCK = "stock"
    FLOW = "flow"


class ExposureType(str, Enum):
    """Counterparty exposure types for financial institution GAR calculation."""

    CORPORATE_LOAN = "corporate_loan"
    DEBT_SECURITY = "debt_security"
    EQUITY = "equity"
    RETAIL_MORTGAGE = "retail_mortgage"
    AUTO_LOAN = "auto_loan"
    PROJECT_FINANCE = "project_finance"
    GREEN_BOND = "green_bond"


class AssetClass(str, Enum):
    """Balance-sheet classification of financial institution exposures."""

    ON_BALANCE_SHEET = "on_balance_sheet"
    OFF_BALANCE_SHEET = "off_balance_sheet"
    TRADING_BOOK = "trading_book"


class EPCRating(str, Enum):
    """Energy Performance Certificate ratings for real estate assets."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class DelegatedAct(str, Enum):
    """
    EU Taxonomy delegated acts containing technical screening criteria.

    Climate DA covers mitigation/adaptation (effective Jan 2022).
    Environmental DA covers remaining four objectives (effective Jan 2024).
    Complementary DA adds nuclear/gas under conditions (effective Jan 2023).
    Simplification DA proposes streamlined criteria (proposed 2025).
    """

    CLIMATE_DA_2021 = "climate_da_2021"
    ENVIRONMENTAL_DA_2023 = "environmental_da_2023"
    COMPLEMENTARY_DA_2022 = "complementary_da_2022"
    SIMPLIFICATION_DA_2025 = "simplification_da_2025"


class Sector(str, Enum):
    """
    Thirteen NACE macro-sectors covered by the EU Taxonomy.

    Each sector contains multiple economic activities with specific
    technical screening criteria.
    """

    FORESTRY = "forestry"
    ENVIRONMENTAL_PROTECTION = "environmental_protection"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    WATER_WASTE = "water_waste"
    TRANSPORT = "transport"
    CONSTRUCTION_REAL_ESTATE = "construction_real_estate"
    ICT = "ict"
    PROFESSIONAL_SERVICES = "professional_services"
    FINANCIAL_INSURANCE = "financial_insurance"
    EDUCATION = "education"
    HEALTH_SOCIAL = "health_social"
    ARTS_RECREATION = "arts_recreation"


class ReportTemplate(str, Enum):
    """
    Reporting templates for Article 8 and EBA Pillar 3 disclosures.

    Article 8 templates apply to non-financial undertakings.
    EBA templates 6-10 apply to credit institutions.
    """

    ARTICLE_8_TURNOVER = "article_8_turnover"
    ARTICLE_8_CAPEX = "article_8_capex"
    ARTICLE_8_OPEX = "article_8_opex"
    EBA_TEMPLATE_6 = "eba_template_6"
    EBA_TEMPLATE_7 = "eba_template_7"
    EBA_TEMPLATE_8 = "eba_template_8"
    EBA_TEMPLATE_9 = "eba_template_9"
    EBA_TEMPLATE_10 = "eba_template_10"


class DataQualityDimension(str, Enum):
    """Dimensions for assessing taxonomy alignment data quality."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    COVERAGE = "coverage"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"


class AssessmentStatus(str, Enum):
    """Lifecycle status of a taxonomy alignment assessment."""

    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class EntityType(str, Enum):
    """Legal entity type for taxonomy reporting obligations."""

    FINANCIAL = "financial"
    NON_FINANCIAL = "non_financial"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"
    XBRL = "xbrl"


class DataQualityGrade(str, Enum):
    """Overall data quality grade for taxonomy assessments."""

    HIGH = "high"
    ADEQUATE = "adequate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class GapPriority(str, Enum):
    """Priority level for gap analysis items."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapCategory(str, Enum):
    """Category of taxonomy alignment gap."""

    ELIGIBILITY = "eligibility"
    SUBSTANTIAL_CONTRIBUTION = "substantial_contribution"
    DNSH = "dnsh"
    MINIMUM_SAFEGUARDS = "minimum_safeguards"
    DATA_AVAILABILITY = "data_availability"
    REPORTING = "reporting"


# ---------------------------------------------------------------------------
# Environmental Objectives Library
# ---------------------------------------------------------------------------

ENVIRONMENTAL_OBJECTIVES: Dict[EnvironmentalObjective, Dict[str, Any]] = {
    EnvironmentalObjective.CLIMATE_MITIGATION: {
        "description": (
            "Stabilising GHG concentrations consistent with the long-term "
            "temperature goal of the Paris Agreement (well below 2C, pursue 1.5C)."
        ),
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
        "activity_count": 88,
        "application_date": "2022-01-01",
        "article_ref": "Article 10",
    },
    EnvironmentalObjective.CLIMATE_ADAPTATION: {
        "description": (
            "Reducing or preventing the adverse impact of current or expected "
            "future climate, or the risks of such adverse impact, on the activity "
            "itself, people, nature, or assets."
        ),
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
        "activity_count": 102,
        "application_date": "2022-01-01",
        "article_ref": "Article 11",
    },
    EnvironmentalObjective.WATER_MARINE: {
        "description": (
            "Sustainable use and protection of water and marine resources, "
            "including achieving good status under Water Framework Directive "
            "and Marine Strategy Framework Directive."
        ),
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
        "activity_count": 18,
        "application_date": "2024-01-01",
        "article_ref": "Article 12",
    },
    EnvironmentalObjective.CIRCULAR_ECONOMY: {
        "description": (
            "Transitioning to a circular economy, including waste prevention, "
            "re-use, and recycling consistent with EU waste hierarchy."
        ),
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
        "activity_count": 29,
        "application_date": "2024-01-01",
        "article_ref": "Article 13",
    },
    EnvironmentalObjective.POLLUTION_PREVENTION: {
        "description": (
            "Prevention and control of pollution to air, water, and land, "
            "including reducing the use and release of substances of concern."
        ),
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
        "activity_count": 12,
        "application_date": "2024-01-01",
        "article_ref": "Article 14",
    },
    EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS: {
        "description": (
            "Protection and restoration of biodiversity and ecosystems, "
            "including achieving good condition and halting biodiversity loss."
        ),
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
        "activity_count": 14,
        "application_date": "2024-01-01",
        "article_ref": "Article 15",
    },
}


# ---------------------------------------------------------------------------
# Taxonomy Sectors (13 NACE Macro-Sectors)
# ---------------------------------------------------------------------------

TAXONOMY_SECTORS: Dict[Sector, Dict[str, Any]] = {
    Sector.FORESTRY: {
        "name": "Forestry",
        "nace_codes": ["A1", "A2"],
        "description": "Afforestation, reforestation, forest management, conservation forestry",
        "activity_count": 4,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.ENVIRONMENTAL_PROTECTION: {
        "name": "Environmental protection and restoration",
        "nace_codes": ["E37", "E38", "E39"],
        "description": "Wetland restoration, environmental remediation, ecosystem conservation",
        "activity_count": 4,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.MANUFACTURING: {
        "name": "Manufacturing",
        "nace_codes": ["C10", "C11", "C17", "C19", "C20", "C22", "C23", "C24", "C25", "C27"],
        "description": (
            "Manufacture of low-carbon technologies, cement, aluminium, iron/steel, "
            "hydrogen, carbon black, soda ash, chlorine, organic/inorganic chemicals, "
            "plastics, batteries, energy-efficiency equipment"
        ),
        "activity_count": 24,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.ENERGY: {
        "name": "Electricity, gas, steam and air conditioning supply",
        "nace_codes": ["D35"],
        "description": (
            "Solar PV, concentrated solar, wind, ocean energy, hydropower, "
            "geothermal, bioenergy, hydrogen, storage, transmission/distribution, "
            "cogeneration, nuclear (complementary DA)"
        ),
        "activity_count": 18,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.WATER_WASTE: {
        "name": "Water supply, sewerage, waste management and remediation",
        "nace_codes": ["E36", "E37", "E38", "E39"],
        "description": (
            "Water collection/treatment/supply, centralised wastewater treatment, "
            "anaerobic digestion, composting, material recovery, landfill gas capture"
        ),
        "activity_count": 8,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.TRANSPORT: {
        "name": "Transport",
        "nace_codes": ["H49", "H50", "H51", "H52"],
        "description": (
            "Rail passenger/freight, public transport, zero-emission vehicles, "
            "cycling infrastructure, inland waterways, sea/coastal transport, "
            "infrastructure for low-carbon transport"
        ),
        "activity_count": 13,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.CONSTRUCTION_REAL_ESTATE: {
        "name": "Construction and real estate",
        "nace_codes": ["F41", "F42", "F43", "L68"],
        "description": (
            "Construction of new buildings, renovation, installation of EE/RE "
            "equipment, acquisition/ownership of buildings, professional energy services"
        ),
        "activity_count": 7,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.ICT: {
        "name": "Information and communication",
        "nace_codes": ["J58", "J61", "J62", "J63"],
        "description": (
            "Data processing/hosting, data-driven solutions for GHG reduction, "
            "computer programming and consultancy"
        ),
        "activity_count": 3,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.PROFESSIONAL_SERVICES: {
        "name": "Professional, scientific and technical activities",
        "nace_codes": ["M71", "M72"],
        "description": "Engineering, research on climate solutions, close-to-market R&D",
        "activity_count": 2,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.FINANCIAL_INSURANCE: {
        "name": "Financial and insurance activities",
        "nace_codes": ["K64", "K65", "K66"],
        "description": "Non-life insurance underwriting for climate perils, reinsurance",
        "activity_count": 2,
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    Sector.EDUCATION: {
        "name": "Education",
        "nace_codes": ["P85"],
        "description": "Education and training services for climate adaptation/mitigation",
        "activity_count": 1,
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
    },
    Sector.HEALTH_SOCIAL: {
        "name": "Human health and social work activities",
        "nace_codes": ["Q86", "Q87", "Q88"],
        "description": "Residential care, hospital activities contributing to adaptation",
        "activity_count": 1,
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
    },
    Sector.ARTS_RECREATION: {
        "name": "Arts, entertainment and recreation",
        "nace_codes": ["R90", "R91", "R93"],
        "description": "Creative, cultural and nature-based tourism activities",
        "activity_count": 1,
        "delegated_act": DelegatedAct.ENVIRONMENTAL_DA_2023.value,
    },
}


# ---------------------------------------------------------------------------
# Taxonomy Activities (Representative Subset of ~50 Key Activities)
# ---------------------------------------------------------------------------

TAXONOMY_ACTIVITIES: Dict[str, Dict[str, Any]] = {
    # --- Forestry (Section 1) ---
    "1.1": {
        "activity_code": "1.1",
        "name": "Afforestation",
        "nace_codes": ["A2"],
        "sector": Sector.FORESTRY.value,
        "description": "Establishment of forest through planting/seeding on land not recently forested.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 1.1",
        "dnsh_criteria_ref": "Annex I, Section 1.1, Appendix A",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "1.3": {
        "activity_code": "1.3",
        "name": "Forest management",
        "nace_codes": ["A2"],
        "sector": Sector.FORESTRY.value,
        "description": "Forest management including silviculture, harvesting and regeneration.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 1.3",
        "dnsh_criteria_ref": "Annex I, Section 1.3, Appendix A",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "1.4": {
        "activity_code": "1.4",
        "name": "Conservation forestry",
        "nace_codes": ["A2"],
        "sector": Sector.FORESTRY.value,
        "description": "Conservation of existing forest including maintenance and protection.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 1.4",
        "dnsh_criteria_ref": "Annex I, Section 1.4, Appendix A",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Environmental Protection (Section 2) ---
    "2.1": {
        "activity_code": "2.1",
        "name": "Restoration of wetlands",
        "nace_codes": ["E39.00"],
        "sector": Sector.ENVIRONMENTAL_PROTECTION.value,
        "description": "Restoration and rewetting of wetlands for carbon sequestration.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 2.1",
        "dnsh_criteria_ref": "Annex I, Section 2.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Manufacturing (Section 3) ---
    "3.1": {
        "activity_code": "3.1",
        "name": "Manufacture of renewable energy technologies",
        "nace_codes": ["C25", "C27", "C28"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of equipment for renewable energy generation.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 3.1",
        "dnsh_criteria_ref": "Annex I, Section 3.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.3": {
        "activity_code": "3.3",
        "name": "Manufacture of low carbon technologies for transport",
        "nace_codes": ["C29", "C30"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of zero direct emission vehicles, rolling stock, vessels.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 3.3",
        "dnsh_criteria_ref": "Annex I, Section 3.3",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.4": {
        "activity_code": "3.4",
        "name": "Manufacture of batteries",
        "nace_codes": ["C27.20"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of rechargeable batteries, battery packs, and accumulators.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 3.4",
        "dnsh_criteria_ref": "Annex I, Section 3.4",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.5": {
        "activity_code": "3.5",
        "name": "Manufacture of energy efficiency equipment for buildings",
        "nace_codes": ["C23", "C25", "C27", "C28"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of insulation, windows, doors, heat pumps, HVAC controls.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 3.5",
        "dnsh_criteria_ref": "Annex I, Section 3.5",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.6": {
        "activity_code": "3.6",
        "name": "Manufacture of other low carbon technologies",
        "nace_codes": ["C22", "C25", "C26", "C27", "C28"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of technologies aimed at substantial GHG reductions.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 3.6",
        "dnsh_criteria_ref": "Annex I, Section 3.6",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.7": {
        "activity_code": "3.7",
        "name": "Manufacture of cement",
        "nace_codes": ["C23.51"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of cement clinker, cement, or alternative binder.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.TRANSITIONAL.value,
        "sc_criteria_ref": "Annex I, Section 3.7",
        "dnsh_criteria_ref": "Annex I, Section 3.7",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.8": {
        "activity_code": "3.8",
        "name": "Manufacture of aluminium",
        "nace_codes": ["C24.42", "C24.53"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of aluminium through primary smelting or secondary recycling.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.TRANSITIONAL.value,
        "sc_criteria_ref": "Annex I, Section 3.8",
        "dnsh_criteria_ref": "Annex I, Section 3.8",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.9": {
        "activity_code": "3.9",
        "name": "Manufacture of iron and steel",
        "nace_codes": ["C24.10", "C24.20", "C24.51"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of iron and steel including basic iron/steel and ferro-alloys.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.TRANSITIONAL.value,
        "sc_criteria_ref": "Annex I, Section 3.9",
        "dnsh_criteria_ref": "Annex I, Section 3.9",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.10": {
        "activity_code": "3.10",
        "name": "Manufacture of hydrogen",
        "nace_codes": ["C20.11"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of hydrogen via electrolysis, reforming with CCS, or bio-based.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 3.10",
        "dnsh_criteria_ref": "Annex I, Section 3.10",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.14": {
        "activity_code": "3.14",
        "name": "Manufacture of organic basic chemicals",
        "nace_codes": ["C20.14"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of HVC, aromatics, vinyl chloride, styrene, ethylene oxide.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.TRANSITIONAL.value,
        "sc_criteria_ref": "Annex I, Section 3.14",
        "dnsh_criteria_ref": "Annex I, Section 3.14",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "3.17": {
        "activity_code": "3.17",
        "name": "Manufacture of plastics in primary form",
        "nace_codes": ["C20.16"],
        "sector": Sector.MANUFACTURING.value,
        "description": "Manufacture of plastics from bio-based or recycled feedstock.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.TRANSITIONAL.value,
        "sc_criteria_ref": "Annex I, Section 3.17",
        "dnsh_criteria_ref": "Annex I, Section 3.17",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Energy (Section 4) ---
    "4.1": {
        "activity_code": "4.1",
        "name": "Electricity generation using solar photovoltaic technology",
        "nace_codes": ["D35.11", "F42.22"],
        "sector": Sector.ENERGY.value,
        "description": "Generation of electricity using solar PV technology.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 4.1",
        "dnsh_criteria_ref": "Annex I, Section 4.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.3": {
        "activity_code": "4.3",
        "name": "Electricity generation from wind power",
        "nace_codes": ["D35.11", "F42.22"],
        "sector": Sector.ENERGY.value,
        "description": "Generation of electricity from onshore and offshore wind.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 4.3",
        "dnsh_criteria_ref": "Annex I, Section 4.3",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.5": {
        "activity_code": "4.5",
        "name": "Electricity generation from hydropower",
        "nace_codes": ["D35.11", "F42.22"],
        "sector": Sector.ENERGY.value,
        "description": "Generation of electricity from hydropower (run-of-river and reservoir).",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 4.5",
        "dnsh_criteria_ref": "Annex I, Section 4.5",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.6": {
        "activity_code": "4.6",
        "name": "Electricity generation from geothermal energy",
        "nace_codes": ["D35.11", "F42.22"],
        "sector": Sector.ENERGY.value,
        "description": "Electricity from geothermal steam or binary cycle plants.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 4.6",
        "dnsh_criteria_ref": "Annex I, Section 4.6",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.7": {
        "activity_code": "4.7",
        "name": "Electricity generation from renewable non-fossil gaseous and liquid fuels",
        "nace_codes": ["D35.11"],
        "sector": Sector.ENERGY.value,
        "description": "Electricity from biofuels, biogas, biomethane, synthetic fuels.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 4.7",
        "dnsh_criteria_ref": "Annex I, Section 4.7",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.9": {
        "activity_code": "4.9",
        "name": "Transmission and distribution of electricity",
        "nace_codes": ["D35.12", "D35.13"],
        "sector": Sector.ENERGY.value,
        "description": "Construction/operation of transmission/distribution systems.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 4.9",
        "dnsh_criteria_ref": "Annex I, Section 4.9",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.10": {
        "activity_code": "4.10",
        "name": "Storage of electricity",
        "nace_codes": ["D35.11"],
        "sector": Sector.ENERGY.value,
        "description": "Electricity storage: batteries, pumped hydro, compressed air, hydrogen.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 4.10",
        "dnsh_criteria_ref": "Annex I, Section 4.10",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.13": {
        "activity_code": "4.13",
        "name": "Manufacture of biogas and biofuels for use in transport and of bioliquids",
        "nace_codes": ["D35.21"],
        "sector": Sector.ENERGY.value,
        "description": "Production of biogas, biofuels, bioliquids from sustainable feedstocks.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 4.13",
        "dnsh_criteria_ref": "Annex I, Section 4.13",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.15": {
        "activity_code": "4.15",
        "name": "District heating/cooling distribution",
        "nace_codes": ["D35.30"],
        "sector": Sector.ENERGY.value,
        "description": "Construction/operation of district heating/cooling distribution networks.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 4.15",
        "dnsh_criteria_ref": "Annex I, Section 4.15",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "4.29": {
        "activity_code": "4.29",
        "name": "Electricity generation from fossil gaseous fuels",
        "nace_codes": ["D35.11"],
        "sector": Sector.ENERGY.value,
        "description": "Electricity generation from natural gas meeting lifecycle threshold.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.TRANSITIONAL.value,
        "sc_criteria_ref": "Complementary DA, Section 4.29",
        "dnsh_criteria_ref": "Complementary DA, Section 4.29",
        "delegated_act": DelegatedAct.COMPLEMENTARY_DA_2022.value,
    },
    # --- Water/Waste (Section 5) ---
    "5.1": {
        "activity_code": "5.1",
        "name": "Construction, extension and operation of water collection, treatment and supply",
        "nace_codes": ["E36.00", "F42.99"],
        "sector": Sector.WATER_WASTE.value,
        "description": "Water infrastructure meeting energy efficiency and leakage thresholds.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 5.1",
        "dnsh_criteria_ref": "Annex I, Section 5.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "5.3": {
        "activity_code": "5.3",
        "name": "Construction, extension and operation of wastewater collection and treatment",
        "nace_codes": ["E37.00", "F42.99"],
        "sector": Sector.WATER_WASTE.value,
        "description": "Centralised wastewater treatment with energy efficiency requirements.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 5.3",
        "dnsh_criteria_ref": "Annex I, Section 5.3",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "5.9": {
        "activity_code": "5.9",
        "name": "Material recovery from non-hazardous waste",
        "nace_codes": ["E38.32"],
        "sector": Sector.WATER_WASTE.value,
        "description": "Sorting and processing of non-hazardous waste into secondary raw materials.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 5.9",
        "dnsh_criteria_ref": "Annex I, Section 5.9",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Transport (Section 6) ---
    "6.1": {
        "activity_code": "6.1",
        "name": "Passenger interurban rail transport",
        "nace_codes": ["H49.10"],
        "sector": Sector.TRANSPORT.value,
        "description": "Interurban passenger rail (zero direct emissions or <50g CO2/pkm).",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 6.1",
        "dnsh_criteria_ref": "Annex I, Section 6.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "6.3": {
        "activity_code": "6.3",
        "name": "Urban and suburban transport, road passenger transport",
        "nace_codes": ["H49.31", "H49.39", "N77.39"],
        "sector": Sector.TRANSPORT.value,
        "description": "Public transport and shared mobility with zero direct emissions.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 6.3",
        "dnsh_criteria_ref": "Annex I, Section 6.3",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "6.5": {
        "activity_code": "6.5",
        "name": "Transport by motorbikes, passenger cars and light commercial vehicles",
        "nace_codes": ["H49.32", "H49.39", "N77.11"],
        "sector": Sector.TRANSPORT.value,
        "description": "Zero tailpipe emission vehicles (BEV, FCEV).",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 6.5",
        "dnsh_criteria_ref": "Annex I, Section 6.5",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "6.6": {
        "activity_code": "6.6",
        "name": "Freight transport services by road",
        "nace_codes": ["H49.41", "H53.10", "H53.20"],
        "sector": Sector.TRANSPORT.value,
        "description": "Zero-emission freight vehicles (heavy-duty BEV, FCEV, catenary trucks).",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 6.6",
        "dnsh_criteria_ref": "Annex I, Section 6.6",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "6.14": {
        "activity_code": "6.14",
        "name": "Infrastructure for rail transport",
        "nace_codes": ["F42.12"],
        "sector": Sector.TRANSPORT.value,
        "description": "Construction/maintenance/operation of electrified rail infrastructure.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 6.14",
        "dnsh_criteria_ref": "Annex I, Section 6.14",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "6.15": {
        "activity_code": "6.15",
        "name": "Infrastructure enabling low-carbon road transport and public transport",
        "nace_codes": ["F42.11", "F42.13"],
        "sector": Sector.TRANSPORT.value,
        "description": "EV charging infrastructure, hydrogen refuelling, electric road systems.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 6.15",
        "dnsh_criteria_ref": "Annex I, Section 6.15",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Construction & Real Estate (Section 7) ---
    "7.1": {
        "activity_code": "7.1",
        "name": "Construction of new buildings",
        "nace_codes": ["F41.1", "F41.2"],
        "sector": Sector.CONSTRUCTION_REAL_ESTATE.value,
        "description": "New building with PED at least 10% below NZEB threshold.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 7.1",
        "dnsh_criteria_ref": "Annex I, Section 7.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "7.2": {
        "activity_code": "7.2",
        "name": "Renovation of existing buildings",
        "nace_codes": ["F41", "F43"],
        "sector": Sector.CONSTRUCTION_REAL_ESTATE.value,
        "description": "Major renovation achieving at least 30% PED reduction.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 7.2",
        "dnsh_criteria_ref": "Annex I, Section 7.2",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "7.3": {
        "activity_code": "7.3",
        "name": "Installation, maintenance and repair of energy efficiency equipment",
        "nace_codes": ["F43"],
        "sector": Sector.CONSTRUCTION_REAL_ESTATE.value,
        "description": "Installation of insulation, energy-efficient windows, lighting, HVAC.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 7.3",
        "dnsh_criteria_ref": "Annex I, Section 7.3",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "7.6": {
        "activity_code": "7.6",
        "name": "Installation, maintenance and repair of renewable energy technologies",
        "nace_codes": ["F43"],
        "sector": Sector.CONSTRUCTION_REAL_ESTATE.value,
        "description": "Rooftop solar, heat pumps, solar thermal, small-scale wind.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 7.6",
        "dnsh_criteria_ref": "Annex I, Section 7.6",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "7.7": {
        "activity_code": "7.7",
        "name": "Acquisition and ownership of buildings",
        "nace_codes": ["L68"],
        "sector": Sector.CONSTRUCTION_REAL_ESTATE.value,
        "description": "EPC class A or top 15% national/regional building stock.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 7.7",
        "dnsh_criteria_ref": "Annex I, Section 7.7",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- ICT (Section 8) ---
    "8.1": {
        "activity_code": "8.1",
        "name": "Data processing, hosting and related activities",
        "nace_codes": ["J63.11"],
        "sector": Sector.ICT.value,
        "description": "Data centres meeting European Code of Conduct for Energy Efficiency.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.OWN_PERFORMANCE.value,
        "sc_criteria_ref": "Annex I, Section 8.1",
        "dnsh_criteria_ref": "Annex I, Section 8.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "8.2": {
        "activity_code": "8.2",
        "name": "Data-driven solutions for GHG emissions reductions",
        "nace_codes": ["J61", "J62", "J63.11"],
        "sector": Sector.ICT.value,
        "description": "ICT solutions enabling GHG reductions in other sectors.",
        "objectives": [EnvironmentalObjective.CLIMATE_MITIGATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 8.2",
        "dnsh_criteria_ref": "Annex I, Section 8.2",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Professional Services (Section 9) ---
    "9.1": {
        "activity_code": "9.1",
        "name": "Close to market research, development and innovation",
        "nace_codes": ["M71.12", "M72.1"],
        "sector": Sector.PROFESSIONAL_SERVICES.value,
        "description": "R&D for solutions enabling GHG reductions or climate adaptation.",
        "objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION.value,
            EnvironmentalObjective.CLIMATE_ADAPTATION.value,
        ],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex I, Section 9.1",
        "dnsh_criteria_ref": "Annex I, Section 9.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    # --- Financial/Insurance (Section 10) ---
    "10.1": {
        "activity_code": "10.1",
        "name": "Non-life insurance and reinsurance underwriting of climate-related perils",
        "nace_codes": ["K65.11", "K65.12"],
        "sector": Sector.FINANCIAL_INSURANCE.value,
        "description": "Insurance/reinsurance of natural catastrophe and weather perils.",
        "objectives": [EnvironmentalObjective.CLIMATE_ADAPTATION.value],
        "activity_type": ActivityType.ENABLING.value,
        "sc_criteria_ref": "Annex II, Section 10.1",
        "dnsh_criteria_ref": "Annex II, Section 10.1",
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
}


# ---------------------------------------------------------------------------
# Substantial Contribution Quantitative Thresholds
# ---------------------------------------------------------------------------

SC_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "electricity_generation": {
        "metric": "lifecycle_ghg_emissions",
        "threshold": Decimal("100"),
        "unit": "gCO2e/kWh",
        "description": "Electricity generation must be below 100 gCO2e/kWh lifecycle.",
        "activity_codes": ["4.1", "4.3", "4.5", "4.7"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "cement": {
        "metric": "specific_ghg_emissions",
        "threshold": Decimal("0.722"),
        "unit": "tCO2e/t_clinker",
        "description": "Grey cement clinker below 0.722 tCO2e per tonne.",
        "activity_codes": ["3.7"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "steel_eaf": {
        "metric": "specific_ghg_emissions",
        "threshold": Decimal("0.283"),
        "unit": "tCO2e/t_steel",
        "description": "EAF steel production below 0.283 tCO2e per tonne.",
        "activity_codes": ["3.9"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "steel_integrated": {
        "metric": "specific_ghg_emissions",
        "threshold": Decimal("1.331"),
        "unit": "tCO2e/t_steel",
        "description": "Integrated steel (BF-BOF) below 1.331 tCO2e per tonne.",
        "activity_codes": ["3.9"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "aluminium_primary": {
        "metric": "specific_ghg_emissions",
        "threshold": Decimal("1.514"),
        "unit": "tCO2e/t_aluminium",
        "description": "Primary aluminium smelting below 1.514 tCO2e per tonne.",
        "activity_codes": ["3.8"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "hydrogen": {
        "metric": "lifecycle_ghg_emissions",
        "threshold": Decimal("3.0"),
        "unit": "tCO2e/tH2",
        "description": "Hydrogen production below 3.0 tCO2e per tonne H2.",
        "activity_codes": ["3.10"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "new_buildings": {
        "metric": "primary_energy_demand",
        "threshold": Decimal("10"),
        "unit": "percent_below_nzeb",
        "description": "New buildings PED at least 10% below NZEB threshold.",
        "activity_codes": ["7.1"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "building_renovation": {
        "metric": "primary_energy_reduction",
        "threshold": Decimal("30"),
        "unit": "percent_reduction",
        "description": "Major renovation at least 30% PED reduction.",
        "activity_codes": ["7.2"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "building_acquisition": {
        "metric": "epc_rating",
        "threshold": "A",
        "unit": "epc_class",
        "description": "EPC class A or top 15% national/regional building stock.",
        "activity_codes": ["7.7"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "vehicles_zero_emission": {
        "metric": "tailpipe_emissions",
        "threshold": Decimal("0"),
        "unit": "gCO2/km",
        "description": "Zero tailpipe emissions for passenger cars/light commercial vehicles.",
        "activity_codes": ["6.5"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "rail_passenger": {
        "metric": "direct_co2_per_pkm",
        "threshold": Decimal("50"),
        "unit": "gCO2/pkm",
        "description": "Interurban rail zero direct emissions or below 50 gCO2/pkm.",
        "activity_codes": ["6.1"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "data_centres": {
        "metric": "code_of_conduct_compliance",
        "threshold": "compliant",
        "unit": "boolean_compliance",
        "description": "European Code of Conduct for Energy Efficiency compliance.",
        "activity_codes": ["8.1"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
    "natural_gas_electricity": {
        "metric": "lifecycle_ghg_emissions",
        "threshold": Decimal("270"),
        "unit": "gCO2e/kWh",
        "description": "Natural gas electricity below 270 gCO2e/kWh lifecycle.",
        "activity_codes": ["4.29"],
        "delegated_act": DelegatedAct.COMPLEMENTARY_DA_2022.value,
    },
    "organic_chemicals_hvc": {
        "metric": "specific_ghg_emissions",
        "threshold": Decimal("0.693"),
        "unit": "tCO2e/t_hvc",
        "description": "HVC production below 0.693 tCO2e per tonne.",
        "activity_codes": ["3.14"],
        "delegated_act": DelegatedAct.CLIMATE_DA_2021.value,
    },
}


# ---------------------------------------------------------------------------
# DNSH Criteria Matrix (activity_code -> objective -> DNSH description)
# ---------------------------------------------------------------------------

DNSH_MATRIX: Dict[str, Dict[str, str]] = {
    "4.1": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A, adaptation solutions "
            "within 5 years for material risks."
        ),
        EnvironmentalObjective.WATER_MARINE.value: "Not applicable.",
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "Equipment reuse and recyclability, waste management plan for "
            "end-of-life PV panels per Directive 2012/19/EU (WEEE)."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "No SVHC above 0.1% w/w. Compliance with REACH and RoHS."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU. No harm to Natura 2000 sites."
        ),
    },
    "4.3": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A."
        ),
        EnvironmentalObjective.WATER_MARINE.value: "Not applicable.",
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "Waste management plan for turbine blades; equipment recyclability."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "No SVHC above 0.1% w/w; compliance with REACH."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU; avoidance of bird/bat mortality "
            "hotspots; radar-assisted shutdown on demand."
        ),
    },
    "3.9": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A."
        ),
        EnvironmentalObjective.WATER_MARINE.value: (
            "Water use and protection per Water Framework Directive."
        ),
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "Scrap steel recycling rates; by-product management (slag, dust)."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "BAT conclusions for iron and steel under IED 2010/75/EU."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU."
        ),
    },
    "7.1": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A; resilient design."
        ),
        EnvironmentalObjective.WATER_MARINE.value: (
            "Water-efficient appliances: taps <=6L/min, showers <=8L/min, "
            "toilets <=6L/flush."
        ),
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "At least 70% C&D waste for reuse/recycling; design for adaptability."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "No SVHC; REACH compliance; indoor air quality per EN 16798-1."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "Not on arable/greenfield of high biodiversity value; EIA required."
        ),
    },
    "7.2": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A."
        ),
        EnvironmentalObjective.WATER_MARINE.value: (
            "Same flow rate requirements as 7.1 where appliances installed."
        ),
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "At least 70% C&D waste for reuse/recycling."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "No SVHC; asbestos management; indoor air quality testing."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU where required."
        ),
    },
    "6.5": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A."
        ),
        EnvironmentalObjective.WATER_MARINE.value: "Not applicable for BEV/FCEV.",
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "Battery reuse/recycling per Directive 2006/66/EC; "
            "ELV per Directive 2000/53/EC."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "Tyres comply with rolling noise/resistance limits; RoHS compliance."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU for manufacturing facilities."
        ),
    },
    "3.10": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A."
        ),
        EnvironmentalObjective.WATER_MARINE.value: (
            "Water use per local permits; no water body degradation."
        ),
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "Equipment recyclability; catalyst recovery plans."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "Emissions within BAT-AEL per IED; REACH compliance."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU where required."
        ),
    },
    "8.1": {
        EnvironmentalObjective.CLIMATE_ADAPTATION.value: (
            "Physical climate risk assessment per Appendix A."
        ),
        EnvironmentalObjective.WATER_MARINE.value: (
            "Water usage assessment; local discharge permit compliance."
        ),
        EnvironmentalObjective.CIRCULAR_ECONOMY.value: (
            "Equipment reuse/recycling plan; WEEE compliance."
        ),
        EnvironmentalObjective.POLLUTION_PREVENTION.value: (
            "REACH compliance; no ozone-depleting substances in cooling."
        ),
        EnvironmentalObjective.BIODIVERSITY_ECOSYSTEMS.value: (
            "EIA per Directive 2011/92/EU where applicable."
        ),
    },
}


# ---------------------------------------------------------------------------
# Minimum Safeguard Topics and Checks
# ---------------------------------------------------------------------------

MINIMUM_SAFEGUARD_TOPICS: Dict[SafeguardTopic, Dict[str, Any]] = {
    SafeguardTopic.HUMAN_RIGHTS: {
        "description": (
            "Alignment with UN Guiding Principles on Business and Human Rights "
            "and OECD Guidelines for MNEs (Chapter IV)."
        ),
        "legal_basis": "Article 18(1) Regulation 2020/852",
        "procedural_checks": [
            "Has adopted a human rights due diligence policy",
            "Has publicly available human rights policy statement",
            "Conducts human rights impact assessments",
            "Has grievance mechanism accessible to stakeholders",
            "Tracks and reports on remediation of adverse impacts",
            "Monitors supply chain for forced/child labour risks",
        ],
        "outcome_checks": [
            "No adverse NCP findings in past 3 years",
            "No unresolved court judgments for human rights violations",
            "No credible NGO reports of severe abuses unaddressed",
            "No inclusion on exclusion lists for human rights violations",
        ],
    },
    SafeguardTopic.ANTI_CORRUPTION: {
        "description": (
            "Alignment with UN Convention against Corruption and "
            "OECD Anti-Bribery Convention."
        ),
        "legal_basis": "Article 18(1) Regulation 2020/852",
        "procedural_checks": [
            "Has anti-corruption and anti-bribery policy",
            "Conducts anti-corruption risk assessments",
            "Provides anti-corruption training",
            "Has whistleblower protection mechanisms",
            "Due diligence on business partners and intermediaries",
        ],
        "outcome_checks": [
            "No bribery/corruption convictions in past 5 years",
            "No pending corruption proceedings",
            "No debarment from public procurement for corruption",
        ],
    },
    SafeguardTopic.TAXATION: {
        "description": (
            "Compliance with tax obligations and OECD Guidelines "
            "for MNEs (Chapter XI)."
        ),
        "legal_basis": "Article 18(1) Regulation 2020/852",
        "procedural_checks": [
            "Has tax governance framework with board oversight",
            "Publishes CbCR or tax transparency report",
            "Transfer pricing policy aligned with OECD",
            "No aggressive tax planning structures",
            "Cooperates with tax authorities",
        ],
        "outcome_checks": [
            "No findings on EU non-cooperative tax jurisdictions list",
            "No material tax evasion convictions in past 5 years",
            "No adverse state aid rulings on tax arrangements",
        ],
    },
    SafeguardTopic.FAIR_COMPETITION: {
        "description": (
            "Compliance with competition law and OECD Guidelines "
            "for MNEs (Chapter X)."
        ),
        "legal_basis": "Article 18(1) Regulation 2020/852",
        "procedural_checks": [
            "Has antitrust/competition compliance policy",
            "Provides competition law training",
            "Procedures to prevent anti-competitive behaviour",
            "Monitors and audits competition compliance",
        ],
        "outcome_checks": [
            "No material antitrust fines in past 5 years",
            "No pending cartel investigations",
            "No unresolved anti-competitive practice rulings",
        ],
    },
}


# ---------------------------------------------------------------------------
# GAR Exposure Type Definitions
# ---------------------------------------------------------------------------

GAR_EXPOSURE_TYPES: Dict[ExposureType, Dict[str, Any]] = {
    ExposureType.CORPORATE_LOAN: {
        "name": "Loans and advances to corporates",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "Counterparty NFRD/CSRD, allocation per taxonomy KPIs.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_7.value,
        "counterparty_scope": "non_financial_corporates_csrd",
    },
    ExposureType.DEBT_SECURITY: {
        "name": "Debt securities (corporate bonds)",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "Issuer NFRD/CSRD; allocation per issuer taxonomy KPIs.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_7.value,
        "counterparty_scope": "non_financial_corporates_csrd",
    },
    ExposureType.EQUITY: {
        "name": "Equity holdings (not held for trading)",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "Investee NFRD/CSRD; use reported Taxonomy KPIs.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_7.value,
        "counterparty_scope": "non_financial_corporates_csrd",
    },
    ExposureType.RETAIL_MORTGAGE: {
        "name": "Loans collateralised by residential immovable property",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "EPC A or top 15% national stock = aligned.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_8.value,
        "counterparty_scope": "retail_households",
    },
    ExposureType.AUTO_LOAN: {
        "name": "Loans for motor vehicles",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "Zero tailpipe = aligned.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_8.value,
        "counterparty_scope": "retail_households",
    },
    ExposureType.PROJECT_FINANCE: {
        "name": "Project finance exposures",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "Use-of-proceeds: aligned project = full exposure counted.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_7.value,
        "counterparty_scope": "special_purpose_vehicles",
    },
    ExposureType.GREEN_BOND: {
        "name": "EU Green Bond Standard compliant bonds",
        "asset_class": AssetClass.ON_BALANCE_SHEET.value,
        "eligibility_rule": "EU GBS use-of-proceeds to taxonomy-aligned activities.",
        "eba_template": ReportTemplate.EBA_TEMPLATE_7.value,
        "counterparty_scope": "all_issuers",
    },
}


# ---------------------------------------------------------------------------
# Reporting Template Definitions
# ---------------------------------------------------------------------------

REPORTING_TEMPLATES: Dict[ReportTemplate, Dict[str, Any]] = {
    ReportTemplate.ARTICLE_8_TURNOVER: {
        "name": "Article 8 - Turnover KPI Template",
        "entity_scope": EntityType.NON_FINANCIAL.value,
        "description": "Turnover proportion from taxonomy-eligible and aligned activities.",
        "legal_basis": "Article 8, Delegated Act 2021/2178, Annex II",
        "columns": [
            "economic_activity", "nace_code", "absolute_turnover", "proportion_turnover",
            "climate_mitigation_sc", "climate_adaptation_sc", "water_sc",
            "circular_economy_sc", "pollution_sc", "biodiversity_sc",
            "climate_mitigation_dnsh", "climate_adaptation_dnsh", "water_dnsh",
            "circular_economy_dnsh", "pollution_dnsh", "biodiversity_dnsh",
            "minimum_safeguards", "proportion_aligned", "enabling_transitional",
        ],
    },
    ReportTemplate.ARTICLE_8_CAPEX: {
        "name": "Article 8 - CapEx KPI Template",
        "entity_scope": EntityType.NON_FINANCIAL.value,
        "description": "CapEx proportion including CapEx plans for eligible-not-aligned.",
        "legal_basis": "Article 8, Delegated Act 2021/2178, Annex II",
        "columns": [
            "economic_activity", "nace_code", "absolute_capex", "proportion_capex",
            "climate_mitigation_sc", "climate_adaptation_sc", "water_sc",
            "circular_economy_sc", "pollution_sc", "biodiversity_sc",
            "climate_mitigation_dnsh", "climate_adaptation_dnsh", "water_dnsh",
            "circular_economy_dnsh", "pollution_dnsh", "biodiversity_dnsh",
            "minimum_safeguards", "proportion_aligned", "enabling_transitional",
            "capex_plan_eligible",
        ],
    },
    ReportTemplate.ARTICLE_8_OPEX: {
        "name": "Article 8 - OpEx KPI Template",
        "entity_scope": EntityType.NON_FINANCIAL.value,
        "description": "OpEx proportion for taxonomy-eligible and aligned activities.",
        "legal_basis": "Article 8, Delegated Act 2021/2178, Annex II",
        "columns": [
            "economic_activity", "nace_code", "absolute_opex", "proportion_opex",
            "climate_mitigation_sc", "climate_adaptation_sc", "water_sc",
            "circular_economy_sc", "pollution_sc", "biodiversity_sc",
            "climate_mitigation_dnsh", "climate_adaptation_dnsh", "water_dnsh",
            "circular_economy_dnsh", "pollution_dnsh", "biodiversity_dnsh",
            "minimum_safeguards", "proportion_aligned", "enabling_transitional",
        ],
    },
    ReportTemplate.EBA_TEMPLATE_6: {
        "name": "EBA Template 6 - Summary of GAR KPIs",
        "entity_scope": EntityType.FINANCIAL.value,
        "description": "Summary table of Green Asset Ratio for credit institutions.",
        "legal_basis": "EBA ITS on Pillar 3 ESG Disclosures, Template 6",
        "columns": [
            "total_covered_assets", "taxonomy_relevant_sectors_exposures",
            "gar_stock_turnover", "gar_stock_capex", "gar_stock_opex",
            "gar_flow_turnover", "gar_flow_capex", "gar_flow_opex",
        ],
    },
    ReportTemplate.EBA_TEMPLATE_7: {
        "name": "EBA Template 7 - GAR Sector Information",
        "entity_scope": EntityType.FINANCIAL.value,
        "description": "GAR by counterparty sector and environmental objective.",
        "legal_basis": "EBA ITS on Pillar 3 ESG Disclosures, Template 7",
        "columns": [
            "counterparty_sector", "nace_code", "gross_carrying_amount",
            "eligible_amount", "aligned_amount_mitigation",
            "aligned_amount_adaptation", "aligned_amount_water",
            "aligned_amount_circular", "aligned_amount_pollution",
            "aligned_amount_biodiversity", "enabling_share", "transitional_share",
        ],
    },
    ReportTemplate.EBA_TEMPLATE_8: {
        "name": "EBA Template 8 - GAR by Counterparty Sector (Households)",
        "entity_scope": EntityType.FINANCIAL.value,
        "description": "GAR for retail exposures (mortgages, auto loans).",
        "legal_basis": "EBA ITS on Pillar 3 ESG Disclosures, Template 8",
        "columns": [
            "exposure_type", "gross_carrying_amount", "eligible_amount",
            "aligned_amount", "epc_based_aligned", "vehicle_based_aligned",
        ],
    },
    ReportTemplate.EBA_TEMPLATE_9: {
        "name": "EBA Template 9 - BTAR Sector Information",
        "entity_scope": EntityType.FINANCIAL.value,
        "description": "Banking-Book Taxonomy Alignment Ratio (extended counterparties).",
        "legal_basis": "EBA ITS on Pillar 3 ESG Disclosures, Template 9",
        "columns": [
            "counterparty_sector", "gross_carrying_amount",
            "extended_eligible_amount", "extended_aligned_amount",
            "btar_percentage",
        ],
    },
    ReportTemplate.EBA_TEMPLATE_10: {
        "name": "EBA Template 10 - Other Climate-related Mitigating Actions",
        "entity_scope": EntityType.FINANCIAL.value,
        "description": "Non-taxonomy activities contributing to climate mitigation.",
        "legal_basis": "EBA ITS on Pillar 3 ESG Disclosures, Template 10",
        "columns": [
            "exposure_type", "gross_carrying_amount", "of_which_stage_2",
            "of_which_non_performing", "mitigating_action_type",
        ],
    },
}


# ---------------------------------------------------------------------------
# De Minimis Threshold
# ---------------------------------------------------------------------------

DE_MINIMIS_THRESHOLD: Decimal = Decimal("0.10")
"""Activities below 10% of total may be excluded under Article 8 simplification."""


# ---------------------------------------------------------------------------
# Data Quality Dimension Weights
# ---------------------------------------------------------------------------

DATA_QUALITY_WEIGHTS: Dict[DataQualityDimension, Decimal] = {
    DataQualityDimension.COMPLETENESS: Decimal("0.25"),
    DataQualityDimension.ACCURACY: Decimal("0.25"),
    DataQualityDimension.COVERAGE: Decimal("0.20"),
    DataQualityDimension.CONSISTENCY: Decimal("0.15"),
    DataQualityDimension.TIMELINESS: Decimal("0.15"),
}


# ---------------------------------------------------------------------------
# Data Quality Grade Thresholds
# ---------------------------------------------------------------------------

DATA_QUALITY_GRADE_THRESHOLDS: Dict[DataQualityGrade, Dict[str, Decimal]] = {
    DataQualityGrade.HIGH: {"min": Decimal("0.80"), "max": Decimal("1.00")},
    DataQualityGrade.ADEQUATE: {"min": Decimal("0.60"), "max": Decimal("0.80")},
    DataQualityGrade.LOW: {"min": Decimal("0.40"), "max": Decimal("0.60")},
    DataQualityGrade.INSUFFICIENT: {"min": Decimal("0.00"), "max": Decimal("0.40")},
}


# ---------------------------------------------------------------------------
# EPC Rating Numeric Scores (for GAR real-estate alignment)
# ---------------------------------------------------------------------------

EPC_RATING_SCORES: Dict[EPCRating, int] = {
    EPCRating.A: 7,
    EPCRating.B: 6,
    EPCRating.C: 5,
    EPCRating.D: 4,
    EPCRating.E: 3,
    EPCRating.F: 2,
    EPCRating.G: 1,
}


# ---------------------------------------------------------------------------
# Alignment Test Step Names
# ---------------------------------------------------------------------------

ALIGNMENT_STEPS: List[Dict[str, str]] = [
    {
        "step": "1",
        "name": "Eligibility Screening",
        "description": "Determine if the activity is covered by the EU Taxonomy.",
    },
    {
        "step": "2",
        "name": "Substantial Contribution",
        "description": "Assess if the activity meets quantitative SC thresholds.",
    },
    {
        "step": "3",
        "name": "Do No Significant Harm",
        "description": "Verify no significant harm to other five objectives.",
    },
    {
        "step": "4",
        "name": "Minimum Safeguards",
        "description": "Confirm OECD/UN/ILO compliance on human rights, corruption, tax, competition.",
    },
]


# ---------------------------------------------------------------------------
# MRV Agent to Taxonomy Sector Mapping
# ---------------------------------------------------------------------------

MRV_AGENT_TO_TAXONOMY_SECTOR: Dict[str, Dict[str, str]] = {
    "MRV-001": {"sector": Sector.ENERGY.value, "name": "Stationary Combustion"},
    "MRV-002": {"sector": Sector.MANUFACTURING.value, "name": "Refrigerants & F-Gas"},
    "MRV-003": {"sector": Sector.TRANSPORT.value, "name": "Mobile Combustion"},
    "MRV-004": {"sector": Sector.MANUFACTURING.value, "name": "Process Emissions"},
    "MRV-005": {"sector": Sector.ENERGY.value, "name": "Fugitive Emissions"},
    "MRV-006": {"sector": Sector.FORESTRY.value, "name": "Land Use Emissions"},
    "MRV-007": {"sector": Sector.WATER_WASTE.value, "name": "Waste Treatment"},
    "MRV-008": {"sector": Sector.FORESTRY.value, "name": "Agricultural Emissions"},
    "MRV-009": {"sector": Sector.ENERGY.value, "name": "Scope 2 Location-Based"},
    "MRV-010": {"sector": Sector.ENERGY.value, "name": "Scope 2 Market-Based"},
    "MRV-014": {"sector": Sector.MANUFACTURING.value, "name": "Purchased Goods"},
    "MRV-017": {"sector": Sector.TRANSPORT.value, "name": "Upstream Transport"},
    "MRV-019": {"sector": Sector.TRANSPORT.value, "name": "Business Travel"},
    "MRV-022": {"sector": Sector.TRANSPORT.value, "name": "Downstream Transport"},
    "MRV-028": {"sector": Sector.FINANCIAL_INSURANCE.value, "name": "Investments"},
}


# ---------------------------------------------------------------------------
# Regulatory Jurisdictions Supporting Article 8 / GAR
# ---------------------------------------------------------------------------

REGULATORY_JURISDICTIONS: List[str] = [
    "EU", "UK", "CH", "NO", "IS", "LI",
]


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class TaxonomyAppConfig(BaseSettings):
    """
    GL-Taxonomy-APP v1.0 platform configuration.

    All settings can be overridden via environment variables prefixed
    with ``TAXONOMY_APP_``.

    Example:
        >>> config = TaxonomyAppConfig()
        >>> config.app_name
        'GL-Taxonomy-APP'
        >>> config.de_minimis_threshold
        Decimal('0.10')
    """

    model_config = {"env_prefix": "TAXONOMY_APP_"}

    # -- Application Metadata -----------------------------------------------
    app_name: str = Field(default="GL-Taxonomy-APP", description="Application display name")
    version: str = Field(default="1.0.0", description="Semantic version")
    debug: bool = Field(default=False, description="Enable debug mode")

    # -- Reporting Period ---------------------------------------------------
    reporting_year: int = Field(
        default=2025, ge=2022, le=2100,
        description="Current reporting year (EU Taxonomy from 2022)",
    )
    reporting_period_start_month: int = Field(
        default=1, ge=1, le=12, description="Start month for fiscal year",
    )

    # -- Taxonomy Defaults --------------------------------------------------
    default_delegated_act: DelegatedAct = Field(
        default=DelegatedAct.CLIMATE_DA_2021, description="Default delegated act",
    )
    default_kpi_type: KPIType = Field(
        default=KPIType.TURNOVER, description="Default KPI type",
    )
    default_gar_type: GARType = Field(
        default=GARType.STOCK, description="Default GAR type",
    )

    # -- De Minimis ---------------------------------------------------------
    de_minimis_threshold: Decimal = Field(
        default=Decimal("0.10"), ge=Decimal("0.0"), le=Decimal("1.0"),
        description="De minimis threshold (10%)",
    )

    # -- Alignment Confidence -----------------------------------------------
    minimum_confidence_threshold: Decimal = Field(
        default=Decimal("0.70"), ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Minimum confidence for auto-approve",
    )

    # -- Data Quality -------------------------------------------------------
    minimum_data_quality_score: Decimal = Field(
        default=Decimal("0.60"), ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Minimum DQ score for approval",
    )

    # -- EPC Configuration --------------------------------------------------
    epc_alignment_threshold: EPCRating = Field(
        default=EPCRating.A, description="Min EPC rating for building alignment",
    )
    top_pct_building_stock: Decimal = Field(
        default=Decimal("15"), ge=Decimal("1"), le=Decimal("50"),
        description="Top % of national building stock for alternative",
    )

    # -- MRV Agent Integration ----------------------------------------------
    mrv_agent_base_url: str = Field(
        default="http://localhost:8000/api/v1/mrv", description="MRV agent base URL",
    )
    mrv_agent_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="MRV agent timeout (seconds)",
    )
    mrv_agent_retry_count: int = Field(
        default=3, ge=0, le=10, description="MRV agent retries",
    )

    # -- Financial Configuration --------------------------------------------
    currency: str = Field(default="EUR", description="Default currency")
    supported_currencies: List[str] = Field(
        default=["EUR", "USD", "GBP", "CHF", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF"],
        description="Supported currencies",
    )

    # -- Report Generation --------------------------------------------------
    default_report_format: ReportFormat = Field(
        default=ReportFormat.EXCEL, description="Default report format",
    )
    report_storage_path: str = Field(
        default="reports/taxonomy/", description="Report storage path prefix",
    )

    # -- GAR Computation Defaults -------------------------------------------
    exclude_sovereign_exposures: bool = Field(
        default=True, description="Exclude sovereign/central-bank from GAR denominator",
    )
    exclude_trading_book: bool = Field(
        default=True, description="Exclude trading book from GAR denominator",
    )
    include_derivatives_in_gar: bool = Field(
        default=False, description="Include derivatives in GAR",
    )

    # -- Logging ------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level")
