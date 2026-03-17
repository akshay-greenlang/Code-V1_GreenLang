"""
PACK-017 ESRS Full Coverage Pack - Configuration Manager

This module implements the ESRSFullCoverageConfig and PackConfig classes that
load, merge, and validate all configuration for the ESRS Full Coverage Pack.
It provides comprehensive Pydantic v2 models for every aspect of the complete
ESRS disclosure process across all 12 topical and cross-cutting standards:

Cross-Cutting Standards:
    - ESRS 1: General Requirements (methodology, materiality assessment)
    - ESRS 2: General Disclosures (GOV-1 to GOV-5, SBM-1 to SBM-3, IRO-1, IRO-2)

Environmental Standards:
    - E1: Climate Change (GHG emissions, energy, transition plan, targets)
    - E2: Pollution (air, water, soil pollutants, substances of concern)
    - E3: Water and Marine Resources (consumption, stress, marine)
    - E4: Biodiversity and Ecosystems (land use, species, ecosystem services)
    - E5: Resource Use and Circular Economy (inflows, outflows, circularity)

Social Standards:
    - S1: Own Workforce (working conditions, equal treatment, health & safety)
    - S2: Workers in the Value Chain (engagement, risk, grievance)
    - S3: Affected Communities (rights, FPIC, indigenous peoples)
    - S4: Consumers and End-Users (safety, privacy, inclusion)

Governance Standards:
    - G1: Business Conduct (anti-corruption, political engagement, payments)

Total Disclosure Requirements: 82 DRs across all standards
    - ESRS 2:  10 DRs (GOV-1 to GOV-5, SBM-1 to SBM-3, IRO-1, IRO-2)
    - E1:       9 DRs (E1-1 to E1-9)
    - E2:       6 DRs (E2-1 to E2-6)
    - E3:       5 DRs (E3-1 to E3-5)
    - E4:       6 DRs (E4-1 to E4-6)
    - E5:       6 DRs (E5-1 to E5-6)
    - S1:      17 DRs (S1-1 to S1-17)
    - S2:       5 DRs (S2-1 to S2-5)
    - S3:       5 DRs (S3-1 to S3-5)
    - S4:       5 DRs (S4-1 to S4-5)
    - G1:       6 DRs (G1-1 to G1-6)
    - ESRS 1:   2 DRs (ESRS1-BP1, ESRS1-BP2 - basis for preparation)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (manufacturing / financial_services / energy /
       retail / technology / multi_sector)
    3. Environment overrides (ESRS_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - ESRS Set 1: Delegated Regulation (EU) 2023/2772 (31 July 2023)
    - ESRS 1: General Requirements (EFRAG, November 2022)
    - ESRS 2: General Disclosures (mandatory for all undertakings)
    - CSRD: Directive (EU) 2022/2464 (14 December 2022)
    - Omnibus I Simplification Proposal (COM/2025/80, 26 February 2025)
    - EFRAG XBRL Taxonomy for ESRS (December 2023, updated 2024)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - GRI Standards 2021 (cross-referenced in ESRS)
    - EU Taxonomy Regulation 2020/852

Phase-In Schedule (per Delegated Regulation 2023/2772):
    - Year 1 (FY2024): ESRS 2 mandatory; E1 mandatory if >750 employees
    - Year 2 (FY2025): All standards subject to materiality; E1-6/E1-9 full
    - Year 3 (FY2026): S1 own workforce; E1-9 quantitative
    - Omnibus: Possible deferral/voluntary for <1000 employees

Example:
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.esrs2.governance_body_count)
    1
    >>> material = get_material_standards(config.pack)
    >>> print(len(material))
    5
    >>> print(get_total_disclosure_count())
    82
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - ESRS Full Coverage enumeration types (19 enums)
# =============================================================================


class ESRSStandard(str, Enum):
    """All ESRS standards in Set 1 per Delegated Regulation (EU) 2023/2772."""

    ESRS_1 = "ESRS_1"    # General Requirements
    ESRS_2 = "ESRS_2"    # General Disclosures
    E1 = "E1"            # Climate Change
    E2 = "E2"            # Pollution
    E3 = "E3"            # Water and Marine Resources
    E4 = "E4"            # Biodiversity and Ecosystems
    E5 = "E5"            # Resource Use and Circular Economy
    S1 = "S1"            # Own Workforce
    S2 = "S2"            # Workers in the Value Chain
    S3 = "S3"            # Affected Communities
    S4 = "S4"            # Consumers and End-Users
    G1 = "G1"            # Business Conduct


class MaterialityStatus(str, Enum):
    """Double materiality assessment outcome per ESRS 1 sect. 3.4-3.5."""

    MATERIAL = "MATERIAL"            # Impact and/or financial materiality confirmed
    NOT_MATERIAL = "NOT_MATERIAL"    # Neither impact nor financial materiality
    PENDING = "PENDING"              # Materiality assessment not yet completed


class DisclosureStatus(str, Enum):
    """Disclosure completion status for individual DRs."""

    COMPLETE = "COMPLETE"              # All datapoints populated and validated
    PARTIAL = "PARTIAL"                # Some datapoints populated
    NOT_STARTED = "NOT_STARTED"        # No datapoints populated
    NOT_APPLICABLE = "NOT_APPLICABLE"  # DR not applicable (not material)
    OMITTED = "OMITTED"                # Omitted with explanation per ESRS 1 para 35


class ComplianceLevel(str, Enum):
    """Compliance level for reporting obligations."""

    FULL = "FULL"                      # Full ESRS compliance
    PHASE_IN = "PHASE_IN"              # Using phase-in provisions
    VOLUNTARY = "VOLUNTARY"            # Voluntary early adoption
    OMNIBUS_REDUCED = "OMNIBUS_REDUCED"  # Reduced under Omnibus simplification


class ReportingBoundary(str, Enum):
    """Reporting boundary per ESRS 1 para 62-67."""

    CONSOLIDATED = "CONSOLIDATED"    # Group consolidated report
    INDIVIDUAL = "INDIVIDUAL"        # Individual entity report
    COMBINED = "COMBINED"            # Combined report (parent + subsidiaries)


class TimeHorizon(str, Enum):
    """Time horizon classification per ESRS 1 sect. 6.4 para 77."""

    SHORT_TERM = "SHORT_TERM"        # Up to 1 year
    MEDIUM_TERM = "MEDIUM_TERM"      # 1 to 5 years
    LONG_TERM = "LONG_TERM"          # More than 5 years


class ValueChainScope(str, Enum):
    """Value chain scope for disclosure boundaries per ESRS 1 para 63."""

    UPSTREAM = "UPSTREAM"              # Suppliers, raw material sourcing
    OWN_OPERATIONS = "OWN_OPERATIONS"  # Own facilities and employees
    DOWNSTREAM = "DOWNSTREAM"          # Customers, end-of-life
    FULL = "FULL"                      # Entire value chain


class AssuranceLevel(str, Enum):
    """Assurance level per CSRD Article 34 requirements."""

    LIMITED = "LIMITED"        # Limited assurance (initial CSRD requirement)
    REASONABLE = "REASONABLE"  # Reasonable assurance (future CSRD requirement)
    NONE = "NONE"              # No external assurance


class PollutantMedium(str, Enum):
    """Environmental medium for pollutant emissions per ESRS E2."""

    AIR = "AIR"        # Emissions to air (E2-4 para 28)
    WATER = "WATER"    # Emissions to water (E2-4 para 29)
    SOIL = "SOIL"      # Emissions to soil (E2-4 para 30)


class WaterStressLevel(str, Enum):
    """Water stress classification per WRI Aqueduct, referenced in ESRS E3."""

    LOW = "LOW"                    # <10% baseline water stress
    MEDIUM_LOW = "MEDIUM_LOW"      # 10-20% baseline water stress
    MEDIUM_HIGH = "MEDIUM_HIGH"    # 20-40% baseline water stress
    HIGH = "HIGH"                  # 40-80% baseline water stress
    EXTREMELY_HIGH = "EXTREMELY_HIGH"  # >80% baseline water stress


class BiodiversitySensitivity(str, Enum):
    """Biodiversity sensitivity classification per ESRS E4."""

    LOW = "LOW"            # Low sensitivity area
    MEDIUM = "MEDIUM"      # Medium sensitivity area
    HIGH = "HIGH"          # High sensitivity area (near protected areas)
    CRITICAL = "CRITICAL"  # Critical biodiversity area (KBA, World Heritage)


class CircularityStrategy(str, Enum):
    """Circular economy strategy per ESRS E5 waste hierarchy."""

    REDUCE = "REDUCE"        # Source reduction / prevention
    REUSE = "REUSE"          # Reuse of products / components
    RECYCLE = "RECYCLE"      # Material recycling
    RECOVER = "RECOVER"      # Energy recovery
    REDESIGN = "REDESIGN"    # Redesign for circularity


class WorkforceCategory(str, Enum):
    """Workforce employment category per ESRS S1-6 para 50."""

    PERMANENT = "PERMANENT"                      # Permanent / indefinite contracts
    TEMPORARY = "TEMPORARY"                      # Fixed-term / temporary contracts
    NON_GUARANTEED_HOURS = "NON_GUARANTEED_HOURS"  # Zero-hour / on-call contracts
    FULL_TIME = "FULL_TIME"                      # Full-time equivalent
    PART_TIME = "PART_TIME"                      # Part-time employees


class GenderCategory(str, Enum):
    """Gender categories for S1 workforce disclosures per ESRS S1-6 para 50a."""

    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    NOT_DISCLOSED = "NOT_DISCLOSED"


class IncidentSeverity(str, Enum):
    """Incident severity classification for S1 H&S and human rights."""

    LOW = "LOW"            # Minor impact, easily remediated
    MEDIUM = "MEDIUM"      # Moderate impact, remediation ongoing
    HIGH = "HIGH"          # Significant impact, systemic action required
    CRITICAL = "CRITICAL"  # Severe impact, immediate escalation required


class GovernanceBodyType(str, Enum):
    """Governance body types per ESRS 2 GOV-1."""

    BOARD = "BOARD"              # Board of directors / supervisory board
    COMMITTEE = "COMMITTEE"      # Board sub-committee (audit, sustainability)
    EXECUTIVE = "EXECUTIVE"      # Executive / management board
    MANAGEMENT = "MANAGEMENT"    # Senior management team


class CorruptionRiskLevel(str, Enum):
    """Corruption risk level classification per ESRS G1-4."""

    LOW = "LOW"            # Low corruption risk (Transparency Intl. CPI >60)
    MEDIUM = "MEDIUM"      # Medium corruption risk (CPI 40-60)
    HIGH = "HIGH"          # High corruption risk (CPI 20-40)
    VERY_HIGH = "VERY_HIGH"  # Very high corruption risk (CPI <20)


class PaymentPracticeType(str, Enum):
    """Payment practice classification per ESRS G1-6."""

    STANDARD = "STANDARD"            # Payment within agreed terms
    LATE = "LATE"                    # Payment beyond agreed terms
    EARLY_DISCOUNT = "EARLY_DISCOUNT"  # Early payment with discount


class SectorPreset(str, Enum):
    """Available sector presets for pack configuration."""

    MANUFACTURING = "MANUFACTURING"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    ENERGY = "ENERGY"
    RETAIL = "RETAIL"
    TECHNOLOGY = "TECHNOLOGY"
    MULTI_SECTOR = "MULTI_SECTOR"


# =============================================================================
# ESRS Disclosure Requirements Constants
# =============================================================================
# Each entry: {name, paragraphs, application_requirements, mandatory,
#              quantitative, phase_in_year}
# phase_in_year: the first reporting year the DR becomes mandatory
#   (None = mandatory from Year 1; 2025/2026 = phase-in)
# =============================================================================


ESRS_2_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "GOV-1": {
        "name": "The Role of the Administrative, Management and Supervisory Bodies",
        "paragraphs": "21-27",
        "application_requirements": "AR 1 through AR 7",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "GOV-2": {
        "name": "Information Provided to and Sustainability Matters Addressed by the Undertaking's Administrative, Management and Supervisory Bodies",
        "paragraphs": "28-30",
        "application_requirements": "AR 8 through AR 10",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "GOV-3": {
        "name": "Integration of Sustainability-Related Performance in Incentive Schemes",
        "paragraphs": "31-33",
        "application_requirements": "AR 11 through AR 13",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "GOV-4": {
        "name": "Statement on Due Diligence",
        "paragraphs": "34-36",
        "application_requirements": "AR 14 through AR 16",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "GOV-5": {
        "name": "Risk Management and Internal Controls Over Sustainability Reporting",
        "paragraphs": "37-40",
        "application_requirements": "AR 17 through AR 19",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "SBM-1": {
        "name": "Strategy, Business Model and Value Chain",
        "paragraphs": "40-47",
        "application_requirements": "AR 20 through AR 27",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "SBM-2": {
        "name": "Interests and Views of Stakeholders",
        "paragraphs": "48-51",
        "application_requirements": "AR 28 through AR 30",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "SBM-3": {
        "name": "Material Impacts, Risks and Opportunities and Their Interaction with Strategy and Business Model",
        "paragraphs": "52-56",
        "application_requirements": "AR 31 through AR 35",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "IRO-1": {
        "name": "Description of the Processes to Identify and Assess Material Impacts, Risks and Opportunities",
        "paragraphs": "57-62",
        "application_requirements": "AR 36 through AR 42",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
    "IRO-2": {
        "name": "Disclosure Requirements in ESRS Covered by the Undertaking's Sustainability Statement",
        "paragraphs": "63-66",
        "application_requirements": "AR 43 through AR 45",
        "mandatory": True,
        "quantitative": False,
        "phase_in_year": None,
    },
}

E1_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "E1-1": {
        "name": "Transition Plan for Climate Change Mitigation",
        "paragraphs": "14-16",
        "application_requirements": "AR 1 through AR 12",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E1-2": {
        "name": "Policies Related to Climate Change Mitigation and Adaptation",
        "paragraphs": "22-24",
        "application_requirements": "AR 13 through AR 17",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E1-3": {
        "name": "Actions and Resources Related to Climate Change Policies",
        "paragraphs": "26-28",
        "application_requirements": "AR 18 through AR 23",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E1-4": {
        "name": "Targets Related to Climate Change Mitigation and Adaptation",
        "paragraphs": "30-33",
        "application_requirements": "AR 24 through AR 37",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E1-5": {
        "name": "Energy Consumption and Mix",
        "paragraphs": "35-39",
        "application_requirements": "AR 38 through AR 45",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E1-6": {
        "name": "Gross Scopes 1, 2, 3 and Total GHG Emissions",
        "paragraphs": "44-55",
        "application_requirements": "AR 46 through AR 62",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E1-7": {
        "name": "GHG Removals and GHG Mitigation Projects Financed Through Carbon Credits",
        "paragraphs": "56-58",
        "application_requirements": "AR 63 through AR 68",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E1-8": {
        "name": "Internal Carbon Pricing",
        "paragraphs": "59-61",
        "application_requirements": "AR 69 through AR 73",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E1-9": {
        "name": "Anticipated Financial Effects from Material Physical and Transition Risks and Potential Climate-Related Opportunities",
        "paragraphs": "64-68",
        "application_requirements": "AR 74 through AR 81",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": 2026,
    },
}

E2_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "E2-1": {
        "name": "Policies Related to Pollution",
        "paragraphs": "10-12",
        "application_requirements": "AR 1 through AR 5",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E2-2": {
        "name": "Actions and Resources Related to Pollution",
        "paragraphs": "14-16",
        "application_requirements": "AR 6 through AR 10",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E2-3": {
        "name": "Targets Related to Pollution",
        "paragraphs": "18-22",
        "application_requirements": "AR 11 through AR 16",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E2-4": {
        "name": "Pollution of Air, Water and Soil",
        "paragraphs": "24-34",
        "application_requirements": "AR 17 through AR 30",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E2-5": {
        "name": "Substances of Concern and Substances of Very High Concern",
        "paragraphs": "36-40",
        "application_requirements": "AR 31 through AR 36",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E2-6": {
        "name": "Anticipated Financial Effects from Material Pollution-Related Risks and Opportunities",
        "paragraphs": "42-44",
        "application_requirements": "AR 37 through AR 41",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": 2026,
    },
}

E3_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "E3-1": {
        "name": "Policies Related to Water and Marine Resources",
        "paragraphs": "9-12",
        "application_requirements": "AR 1 through AR 6",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E3-2": {
        "name": "Actions and Resources Related to Water and Marine Resources",
        "paragraphs": "14-16",
        "application_requirements": "AR 7 through AR 12",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E3-3": {
        "name": "Targets Related to Water and Marine Resources",
        "paragraphs": "18-22",
        "application_requirements": "AR 13 through AR 18",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E3-4": {
        "name": "Water Consumption",
        "paragraphs": "24-30",
        "application_requirements": "AR 19 through AR 28",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E3-5": {
        "name": "Anticipated Financial Effects from Material Water and Marine Resources-Related Risks and Opportunities",
        "paragraphs": "32-34",
        "application_requirements": "AR 29 through AR 33",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": 2026,
    },
}

E4_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "E4-1": {
        "name": "Transition Plan on Biodiversity and Ecosystems",
        "paragraphs": "11-14",
        "application_requirements": "AR 1 through AR 4",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E4-2": {
        "name": "Policies Related to Biodiversity and Ecosystems",
        "paragraphs": "16-20",
        "application_requirements": "AR 5 through AR 12",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E4-3": {
        "name": "Actions and Resources Related to Biodiversity and Ecosystems",
        "paragraphs": "22-26",
        "application_requirements": "AR 13 through AR 20",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E4-4": {
        "name": "Targets Related to Biodiversity and Ecosystems",
        "paragraphs": "28-34",
        "application_requirements": "AR 21 through AR 28",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E4-5": {
        "name": "Impact Metrics Related to Biodiversity and Ecosystems Change",
        "paragraphs": "36-46",
        "application_requirements": "AR 29 through AR 40",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E4-6": {
        "name": "Anticipated Financial Effects from Material Biodiversity and Ecosystem-Related Risks and Opportunities",
        "paragraphs": "48-52",
        "application_requirements": "AR 41 through AR 46",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": 2026,
    },
}

E5_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "E5-1": {
        "name": "Policies Related to Resource Use and Circular Economy",
        "paragraphs": "11-14",
        "application_requirements": "AR 1 through AR 6",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "E5-2": {
        "name": "Actions and Resources Related to Resource Use and Circular Economy",
        "paragraphs": "16-20",
        "application_requirements": "AR 7 through AR 12",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E5-3": {
        "name": "Targets Related to Resource Use and Circular Economy",
        "paragraphs": "22-26",
        "application_requirements": "AR 13 through AR 18",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E5-4": {
        "name": "Resource Inflows",
        "paragraphs": "28-32",
        "application_requirements": "AR 19 through AR 26",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E5-5": {
        "name": "Resource Outflows",
        "paragraphs": "34-40",
        "application_requirements": "AR 27 through AR 34",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "E5-6": {
        "name": "Anticipated Financial Effects from Material Resource Use and Circular Economy-Related Risks and Opportunities",
        "paragraphs": "42-46",
        "application_requirements": "AR 35 through AR 39",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": 2026,
    },
}

S1_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "S1-1": {
        "name": "Policies Related to Own Workforce",
        "paragraphs": "19-24",
        "application_requirements": "AR 1 through AR 8",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S1-2": {
        "name": "Processes for Engaging with Own Workforce and Workers' Representatives About Impacts",
        "paragraphs": "26-30",
        "application_requirements": "AR 9 through AR 13",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S1-3": {
        "name": "Processes to Remediate Negative Impacts and Channels for Own Workforce to Raise Concerns",
        "paragraphs": "32-36",
        "application_requirements": "AR 14 through AR 18",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S1-4": {
        "name": "Taking Action on Material Impacts on Own Workforce, and Approaches to Managing Material Risks and Pursuing Material Opportunities Related to Own Workforce, and Effectiveness of Those Actions",
        "paragraphs": "38-42",
        "application_requirements": "AR 19 through AR 25",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-5": {
        "name": "Targets Related to Managing Material Negative Impacts, Advancing Positive Impacts, and Managing Material Risks and Opportunities",
        "paragraphs": "44-48",
        "application_requirements": "AR 26 through AR 30",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-6": {
        "name": "Characteristics of the Undertaking's Employees",
        "paragraphs": "50-54",
        "application_requirements": "AR 31 through AR 42",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-7": {
        "name": "Characteristics of the Undertaking's Non-Employee Workers",
        "paragraphs": "56-58",
        "application_requirements": "AR 43 through AR 48",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-8": {
        "name": "Collective Bargaining Coverage and Social Dialogue",
        "paragraphs": "60-64",
        "application_requirements": "AR 49 through AR 55",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-9": {
        "name": "Diversity Metrics",
        "paragraphs": "66-70",
        "application_requirements": "AR 56 through AR 62",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-10": {
        "name": "Adequate Wages",
        "paragraphs": "72-74",
        "application_requirements": "AR 63 through AR 68",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-11": {
        "name": "Social Protection",
        "paragraphs": "76-78",
        "application_requirements": "AR 69 through AR 72",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-12": {
        "name": "Persons with Disabilities",
        "paragraphs": "80-82",
        "application_requirements": "AR 73 through AR 76",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-13": {
        "name": "Training and Skills Development Metrics",
        "paragraphs": "84-86",
        "application_requirements": "AR 77 through AR 82",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-14": {
        "name": "Health and Safety Metrics",
        "paragraphs": "88-92",
        "application_requirements": "AR 83 through AR 92",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-15": {
        "name": "Work-Life Balance Metrics",
        "paragraphs": "94-96",
        "application_requirements": "AR 93 through AR 96",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S1-16": {
        "name": "Remuneration Metrics (Pay Gap and Total Remuneration)",
        "paragraphs": "98-102",
        "application_requirements": "AR 97 through AR 104",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": 2025,
    },
    "S1-17": {
        "name": "Incidents, Complaints and Severe Human Rights Impacts",
        "paragraphs": "104-106",
        "application_requirements": "AR 105 through AR 109",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
}

S2_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "S2-1": {
        "name": "Policies Related to Value Chain Workers",
        "paragraphs": "14-18",
        "application_requirements": "AR 1 through AR 8",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S2-2": {
        "name": "Processes for Engaging with Value Chain Workers About Impacts",
        "paragraphs": "20-24",
        "application_requirements": "AR 9 through AR 15",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S2-3": {
        "name": "Processes to Remediate Negative Impacts and Channels for Value Chain Workers to Raise Concerns",
        "paragraphs": "26-30",
        "application_requirements": "AR 16 through AR 21",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S2-4": {
        "name": "Taking Action on Material Impacts on Value Chain Workers, and Approaches to Managing Material Risks and Pursuing Material Opportunities Related to Value Chain Workers, and Effectiveness of Those Actions",
        "paragraphs": "32-36",
        "application_requirements": "AR 22 through AR 28",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S2-5": {
        "name": "Targets Related to Managing Material Negative Impacts, Advancing Positive Impacts, and Managing Material Risks and Opportunities",
        "paragraphs": "38-42",
        "application_requirements": "AR 29 through AR 33",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
}

S3_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "S3-1": {
        "name": "Policies Related to Affected Communities",
        "paragraphs": "14-18",
        "application_requirements": "AR 1 through AR 8",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S3-2": {
        "name": "Processes for Engaging with Affected Communities About Impacts",
        "paragraphs": "20-24",
        "application_requirements": "AR 9 through AR 15",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S3-3": {
        "name": "Processes to Remediate Negative Impacts and Channels for Affected Communities to Raise Concerns",
        "paragraphs": "26-30",
        "application_requirements": "AR 16 through AR 21",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S3-4": {
        "name": "Taking Action on Material Impacts on Affected Communities, and Approaches to Managing Material Risks and Pursuing Material Opportunities Related to Affected Communities, and Effectiveness of Those Actions",
        "paragraphs": "32-36",
        "application_requirements": "AR 22 through AR 28",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S3-5": {
        "name": "Targets Related to Managing Material Negative Impacts, Advancing Positive Impacts, and Managing Material Risks and Opportunities",
        "paragraphs": "38-42",
        "application_requirements": "AR 29 through AR 33",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
}

S4_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "S4-1": {
        "name": "Policies Related to Consumers and End-Users",
        "paragraphs": "14-18",
        "application_requirements": "AR 1 through AR 8",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S4-2": {
        "name": "Processes for Engaging with Consumers and End-Users About Impacts",
        "paragraphs": "20-24",
        "application_requirements": "AR 9 through AR 15",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S4-3": {
        "name": "Processes to Remediate Negative Impacts and Channels for Consumers and End-Users to Raise Concerns",
        "paragraphs": "26-30",
        "application_requirements": "AR 16 through AR 21",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "S4-4": {
        "name": "Taking Action on Material Impacts on Consumers and End-Users, and Approaches to Managing Material Risks and Pursuing Material Opportunities Related to Consumers and End-Users, and Effectiveness of Those Actions",
        "paragraphs": "32-36",
        "application_requirements": "AR 22 through AR 28",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "S4-5": {
        "name": "Targets Related to Managing Material Negative Impacts, Advancing Positive Impacts, and Managing Material Risks and Opportunities",
        "paragraphs": "38-42",
        "application_requirements": "AR 29 through AR 33",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
}

G1_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "G1-1": {
        "name": "Business Conduct Policies and Corporate Culture",
        "paragraphs": "10-14",
        "application_requirements": "AR 1 through AR 7",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "G1-2": {
        "name": "Management of Relationships with Suppliers",
        "paragraphs": "16-18",
        "application_requirements": "AR 8 through AR 12",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    },
    "G1-3": {
        "name": "Prevention and Detection of Corruption and Bribery",
        "paragraphs": "20-24",
        "application_requirements": "AR 13 through AR 19",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "G1-4": {
        "name": "Incidents of Corruption or Bribery",
        "paragraphs": "26-28",
        "application_requirements": "AR 20 through AR 24",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "G1-5": {
        "name": "Political Influence and Lobbying Activities",
        "paragraphs": "30-34",
        "application_requirements": "AR 25 through AR 30",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
    "G1-6": {
        "name": "Payment Practices",
        "paragraphs": "36-40",
        "application_requirements": "AR 31 through AR 36",
        "mandatory": False,
        "quantitative": True,
        "phase_in_year": None,
    },
}

# Consolidated map of all DR dicts by standard for lookup
_ALL_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "ESRS_2": ESRS_2_DISCLOSURE_REQUIREMENTS,
    "E1": E1_DISCLOSURE_REQUIREMENTS,
    "E2": E2_DISCLOSURE_REQUIREMENTS,
    "E3": E3_DISCLOSURE_REQUIREMENTS,
    "E4": E4_DISCLOSURE_REQUIREMENTS,
    "E5": E5_DISCLOSURE_REQUIREMENTS,
    "S1": S1_DISCLOSURE_REQUIREMENTS,
    "S2": S2_DISCLOSURE_REQUIREMENTS,
    "S3": S3_DISCLOSURE_REQUIREMENTS,
    "S4": S4_DISCLOSURE_REQUIREMENTS,
    "G1": G1_DISCLOSURE_REQUIREMENTS,
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing": "Industrial manufacturing with E1/E2/E5/S1/G1 focus and process emissions",
    "financial_services": "Financial services with E1/S1/S2/G1 focus and financed emissions",
    "energy": "Energy sector with E1/E2/E3/E4/S1 focus and transition plan emphasis",
    "retail": "Retail sector with E1/E5/S1/S2/S4/G1 focus and supply chain emphasis",
    "technology": "Technology sector with E1/S1/S4/G1 focus and data privacy emphasis",
    "multi_sector": "Multi-sector conglomerate with comprehensive all-standard coverage",
}


# =============================================================================
# Pydantic Sub-Config Models (12 sub-config models)
# =============================================================================


class ESRS2Config(BaseModel):
    """Configuration for ESRS 2 General Disclosures engine.

    Covers governance, strategy, impact/risk/opportunity identification, and
    metrics/targets per ESRS 2 GOV-1 through GOV-5, SBM-1 through SBM-3,
    and IRO-1 through IRO-2.
    """

    enabled: bool = Field(
        True,
        description="Enable ESRS 2 General Disclosures processing",
    )
    governance_body_count: int = Field(
        1,
        ge=1,
        le=20,
        description="Number of administrative, management and supervisory bodies to disclose (GOV-1)",
    )
    governance_body_types: List[GovernanceBodyType] = Field(
        default_factory=lambda: [
            GovernanceBodyType.BOARD,
            GovernanceBodyType.COMMITTEE,
            GovernanceBodyType.EXECUTIVE,
        ],
        description="Types of governance bodies to include in GOV-1",
    )
    sustainability_committee_exists: bool = Field(
        True,
        description="Whether a dedicated sustainability committee exists (GOV-1 para 22)",
    )
    board_sustainability_expertise: bool = Field(
        False,
        description="Whether the board has sustainability expertise or access to it (GOV-1 para 23)",
    )
    due_diligence_processes: bool = Field(
        True,
        description="Whether the undertaking has due diligence processes per GOV-4 para 34",
    )
    due_diligence_frameworks: List[str] = Field(
        default_factory=lambda: [
            "UN Guiding Principles on Business and Human Rights",
            "OECD Guidelines for Multinational Enterprises",
        ],
        description="Due diligence frameworks referenced in GOV-4",
    )
    risk_management_integration: bool = Field(
        True,
        description="Whether sustainability risks are integrated into enterprise risk management (GOV-5)",
    )
    internal_controls_sustainability: bool = Field(
        True,
        description="Whether internal controls cover sustainability reporting (GOV-5 para 38)",
    )
    incentive_schemes_linked: bool = Field(
        False,
        description="Whether incentive schemes are linked to sustainability targets (GOV-3 para 31)",
    )
    incentive_scheme_details: Optional[str] = Field(
        None,
        description="Description of sustainability-linked incentive schemes if applicable",
    )
    strategy_time_horizons: Dict[str, int] = Field(
        default_factory=lambda: {
            "short_term_years": 1,
            "medium_term_years": 5,
            "long_term_years": 10,
        },
        description="Time horizon definitions per ESRS 1 sect. 6.4 para 77",
    )
    stakeholder_engagement_approach: str = Field(
        "structured",
        description="Approach to stakeholder engagement: structured, ad_hoc, continuous (SBM-2)",
    )
    value_chain_description_scope: ValueChainScope = Field(
        ValueChainScope.FULL,
        description="Scope of value chain description in SBM-1",
    )
    double_materiality_methodology: str = Field(
        "inside_out_and_outside_in",
        description="Double materiality assessment methodology: inside_out_and_outside_in, integrated",
    )
    materiality_threshold_impact: str = Field(
        "severity_scale",
        description="Impact materiality threshold method: severity_scale, likelihood_matrix",
    )
    materiality_threshold_financial: str = Field(
        "quantitative_threshold",
        description="Financial materiality threshold method: quantitative_threshold, qualitative_judgment",
    )
    iro_process_frequency: str = Field(
        "annual",
        description="Frequency of IRO identification and assessment: annual, continuous, event_driven",
    )

    @field_validator("strategy_time_horizons")
    @classmethod
    def validate_time_horizons(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate time horizon definitions are logically ordered."""
        short = v.get("short_term_years", 1)
        medium = v.get("medium_term_years", 5)
        long_ = v.get("long_term_years", 10)
        if not (short < medium < long_):
            raise ValueError(
                f"Time horizons must be ascending: short({short}) < "
                f"medium({medium}) < long({long_})"
            )
        return v


class E2PollutionConfig(BaseModel):
    """Configuration for ESRS E2 Pollution engine.

    Covers pollutant emissions to air, water and soil, substances of concern
    and substances of very high concern, and anticipated financial effects
    per E2-1 through E2-6.
    """

    enabled: bool = Field(
        True,
        description="Enable E2 Pollution disclosures",
    )
    pollutant_types: List[PollutantMedium] = Field(
        default_factory=lambda: [
            PollutantMedium.AIR,
            PollutantMedium.WATER,
            PollutantMedium.SOIL,
        ],
        description="Pollutant mediums to track per E2-4",
    )
    emission_to_air_tracked: bool = Field(
        True,
        description="Track emissions to air: NOx, SOx, PM, VOCs, HAPs (E2-4 para 28)",
    )
    emission_to_air_substances: List[str] = Field(
        default_factory=lambda: [
            "NOx", "SOx", "PM2.5", "PM10",
            "NMVOC", "NH3", "heavy_metals",
        ],
        description="Specific air pollutant substances to track",
    )
    emission_to_water_tracked: bool = Field(
        True,
        description="Track emissions to water: priority substances, nutrients, heavy metals (E2-4 para 29)",
    )
    emission_to_water_substances: List[str] = Field(
        default_factory=lambda: [
            "total_nitrogen", "total_phosphorus", "heavy_metals",
            "BOD", "COD", "priority_substances",
        ],
        description="Specific water pollutant substances to track",
    )
    emission_to_soil_tracked: bool = Field(
        True,
        description="Track emissions to soil: heavy metals, pesticides, hydrocarbons (E2-4 para 30)",
    )
    substances_of_concern: bool = Field(
        True,
        description="Track substances of concern per E2-5 and REACH Regulation",
    )
    substances_of_very_high_concern: bool = Field(
        True,
        description="Track SVHCs per ECHA Candidate List (E2-5 para 37)",
    )
    microplastics_tracked: bool = Field(
        False,
        description="Track microplastic emissions per E2-4 para 31",
    )
    pollutant_reporting_unit: str = Field(
        "tonnes",
        description="Reporting unit for pollutant quantities: tonnes, kg, mg",
    )
    e_prtr_alignment: bool = Field(
        True,
        description="Align pollutant reporting with European Pollutant Release and Transfer Register",
    )

    @field_validator("pollutant_reporting_unit")
    @classmethod
    def validate_reporting_unit(cls, v: str) -> str:
        """Validate pollutant reporting unit."""
        valid_units = {"tonnes", "kg", "mg"}
        if v not in valid_units:
            raise ValueError(
                f"Invalid pollutant reporting unit: {v}. Valid: {sorted(valid_units)}"
            )
        return v


class E3WaterConfig(BaseModel):
    """Configuration for ESRS E3 Water and Marine Resources engine.

    Covers water consumption, withdrawal, discharge, water stress area
    identification, and marine resource impacts per E3-1 through E3-5.
    """

    enabled: bool = Field(
        True,
        description="Enable E3 Water and Marine Resources disclosures",
    )
    water_consumption_tracked: bool = Field(
        True,
        description="Track total water consumption (E3-4 para 26)",
    )
    water_withdrawal_tracked: bool = Field(
        True,
        description="Track water withdrawal by source: surface, ground, seawater, produced, third-party",
    )
    water_discharge_tracked: bool = Field(
        True,
        description="Track water discharge by destination: surface, ground, seawater, third-party",
    )
    water_stress_areas: bool = Field(
        True,
        description="Identify and disclose operations in water-stressed areas (E3-4 para 28)",
    )
    water_stress_tool: str = Field(
        "WRI_AQUEDUCT",
        description="Water stress assessment tool: WRI_AQUEDUCT, WWF_WATER_RISK_FILTER",
    )
    water_stress_threshold: WaterStressLevel = Field(
        WaterStressLevel.HIGH,
        description="Minimum water stress level triggering detailed disclosure",
    )
    marine_resources_relevant: bool = Field(
        False,
        description="Whether marine resources are relevant to the undertaking (E3 para 7)",
    )
    ocean_acidification_tracked: bool = Field(
        False,
        description="Track impacts on ocean acidification where relevant",
    )
    water_recycling_rate_target: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Target water recycling/reuse rate as percentage (E3-3)",
    )
    water_intensity_denominators: List[str] = Field(
        default_factory=lambda: ["net_revenue_eur_million"],
        description="Denominators for water intensity metrics",
    )
    water_reporting_unit: str = Field(
        "cubic_meters",
        description="Reporting unit for water volumes: cubic_meters, megaliters",
    )

    @field_validator("water_stress_tool")
    @classmethod
    def validate_water_stress_tool(cls, v: str) -> str:
        """Validate water stress assessment tool is recognized."""
        valid_tools = {"WRI_AQUEDUCT", "WWF_WATER_RISK_FILTER", "CUSTOM"}
        if v not in valid_tools:
            raise ValueError(
                f"Unknown water stress tool: {v}. Valid: {sorted(valid_tools)}"
            )
        return v


class E4BiodiversityConfig(BaseModel):
    """Configuration for ESRS E4 Biodiversity and Ecosystems engine.

    Covers land use change, species at risk, ecosystem services dependencies,
    site-level biodiversity assessment, and deforestation commitments
    per E4-1 through E4-6.
    """

    enabled: bool = Field(
        True,
        description="Enable E4 Biodiversity and Ecosystems disclosures",
    )
    sites_near_sensitive_areas: bool = Field(
        True,
        description="Identify sites near biodiversity-sensitive areas (E4-5 para 38)",
    )
    sensitive_area_buffer_km: float = Field(
        5.0,
        ge=0.0,
        le=50.0,
        description="Buffer zone distance (km) for identifying proximity to sensitive areas",
    )
    sensitive_area_databases: List[str] = Field(
        default_factory=lambda: [
            "IUCN_KBA",
            "NATURA_2000",
            "RAMSAR",
            "UNESCO_WHS",
            "CDDA",
        ],
        description="Biodiversity-sensitive area databases for site screening",
    )
    land_use_change_tracked: bool = Field(
        True,
        description="Track land use and land use change impacts (E4-5 para 37)",
    )
    land_use_categories: List[str] = Field(
        default_factory=lambda: [
            "artificial_surfaces",
            "cropland",
            "grassland",
            "forest",
            "wetland",
            "water_bodies",
        ],
        description="Land use categories per IPCC land use classification",
    )
    species_at_risk: bool = Field(
        True,
        description="Identify species at risk in areas of operation (E4-5 para 40)",
    )
    species_risk_source: str = Field(
        "IUCN_RED_LIST",
        description="Species risk classification source: IUCN_RED_LIST, NATIONAL_RED_LIST",
    )
    ecosystem_services_dependencies: bool = Field(
        True,
        description="Assess dependencies on ecosystem services (E4-5 para 42)",
    )
    ecosystem_services_framework: str = Field(
        "TNFD_LEAP",
        description="Framework for ecosystem services assessment: TNFD_LEAP, ENCORE, CUSTOM",
    )
    deforestation_free_commitment: bool = Field(
        False,
        description="Whether the undertaking has a deforestation-free commitment",
    )
    deforestation_cutoff_date: Optional[str] = Field(
        None,
        description="Cut-off date for deforestation-free commitment (YYYY-MM-DD)",
    )
    biodiversity_offset_tracking: bool = Field(
        False,
        description="Track biodiversity offsets and net-gain commitments",
    )
    invasive_species_tracking: bool = Field(
        False,
        description="Track introduction and management of invasive alien species",
    )

    @field_validator("ecosystem_services_framework")
    @classmethod
    def validate_ecosystem_framework(cls, v: str) -> str:
        """Validate ecosystem services framework is recognized."""
        valid = {"TNFD_LEAP", "ENCORE", "SEEA_EA", "CUSTOM"}
        if v not in valid:
            raise ValueError(
                f"Unknown ecosystem services framework: {v}. Valid: {sorted(valid)}"
            )
        return v


class E5CircularConfig(BaseModel):
    """Configuration for ESRS E5 Resource Use and Circular Economy engine.

    Covers resource inflows, outflows, waste generation, recycled content,
    circular material use rate, and product durability per E5-1 through E5-6.
    """

    enabled: bool = Field(
        True,
        description="Enable E5 Resource Use and Circular Economy disclosures",
    )
    resource_inflows_tracked: bool = Field(
        True,
        description="Track resource inflows by weight and type (E5-4 para 28)",
    )
    resource_inflow_categories: List[str] = Field(
        default_factory=lambda: [
            "virgin_renewable",
            "virgin_non_renewable",
            "secondary_recycled",
            "secondary_reused",
        ],
        description="Resource inflow categories per E5-4",
    )
    recycled_content_target: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Target recycled content percentage in products/materials (E5-3)",
    )
    waste_generation_tracked: bool = Field(
        True,
        description="Track waste generation by type and disposal method (E5-5 para 36)",
    )
    waste_categories: List[str] = Field(
        default_factory=lambda: [
            "hazardous",
            "non_hazardous",
            "radioactive",
        ],
        description="Waste categories per E5-5 and European Waste Catalogue",
    )
    waste_disposal_methods: List[str] = Field(
        default_factory=lambda: [
            "preparation_for_reuse",
            "recycling",
            "other_recovery",
            "incineration",
            "landfill",
            "other_disposal",
        ],
        description="Waste disposal method categories per waste hierarchy",
    )
    circular_material_use_rate: bool = Field(
        True,
        description="Calculate circular material use rate (E5-4 para 30)",
    )
    product_durability_tracked: bool = Field(
        False,
        description="Track product durability and lifespan extension measures (E5-4 para 32)",
    )
    product_end_of_life_tracked: bool = Field(
        True,
        description="Track product end-of-life collection and recycling rates",
    )
    circularity_strategies: List[CircularityStrategy] = Field(
        default_factory=lambda: [
            CircularityStrategy.REDUCE,
            CircularityStrategy.REUSE,
            CircularityStrategy.RECYCLE,
            CircularityStrategy.RECOVER,
        ],
        description="Circular economy strategies implemented by the undertaking",
    )
    resource_reporting_unit: str = Field(
        "tonnes",
        description="Reporting unit for resource and waste quantities",
    )
    extended_producer_responsibility: bool = Field(
        False,
        description="Whether the undertaking is subject to EPR schemes",
    )

    @field_validator("resource_reporting_unit")
    @classmethod
    def validate_resource_unit(cls, v: str) -> str:
        """Validate resource reporting unit."""
        valid_units = {"tonnes", "kg", "cubic_meters"}
        if v not in valid_units:
            raise ValueError(
                f"Invalid resource reporting unit: {v}. Valid: {sorted(valid_units)}"
            )
        return v


class S1WorkforceConfig(BaseModel):
    """Configuration for ESRS S1 Own Workforce engine.

    Covers working conditions, equal treatment, health and safety, training,
    collective bargaining, diversity, remuneration, and human rights incidents
    per S1-1 through S1-17.
    """

    enabled: bool = Field(
        True,
        description="Enable S1 Own Workforce disclosures",
    )
    headcount_threshold: int = Field(
        0,
        ge=0,
        description="Minimum headcount threshold for detailed S1 disclosures (0 = always disclose)",
    )
    total_employees: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of employees for materiality and phase-in assessment",
    )
    employee_categories: List[WorkforceCategory] = Field(
        default_factory=lambda: [
            WorkforceCategory.PERMANENT,
            WorkforceCategory.TEMPORARY,
            WorkforceCategory.FULL_TIME,
            WorkforceCategory.PART_TIME,
        ],
        description="Employee categories to break down per S1-6 para 50",
    )
    non_employee_workers_tracked: bool = Field(
        True,
        description="Track non-employee workers (contractors, agency) per S1-7",
    )
    gender_breakdown: bool = Field(
        True,
        description="Break down workforce metrics by gender per S1-6 para 50a",
    )
    gender_categories: List[GenderCategory] = Field(
        default_factory=lambda: [
            GenderCategory.MALE,
            GenderCategory.FEMALE,
            GenderCategory.OTHER,
        ],
        description="Gender categories for workforce breakdowns",
    )
    country_breakdown: bool = Field(
        True,
        description="Break down workforce metrics by country per S1-6 para 51",
    )
    gender_pay_gap_tracked: bool = Field(
        True,
        description="Track and disclose unadjusted gender pay gap (S1-16 para 98)",
    )
    pay_gap_methodology: str = Field(
        "mean_and_median",
        description="Pay gap calculation methodology: mean, median, mean_and_median",
    )
    adequate_wages_assessment: bool = Field(
        True,
        description="Assess whether employees receive adequate wages (S1-10 para 72)",
    )
    adequate_wage_benchmark: str = Field(
        "applicable_benchmarks",
        description="Adequate wage benchmark: legal_minimum, living_wage, applicable_benchmarks",
    )
    training_hours_tracked: bool = Field(
        True,
        description="Track average training hours per employee (S1-13 para 84)",
    )
    training_breakdown: str = Field(
        "gender_and_category",
        description="Training hours breakdown: total, gender, category, gender_and_category",
    )
    h_and_s_incidents_tracked: bool = Field(
        True,
        description="Track health and safety incidents, fatalities, and lost-day rates (S1-14 para 88)",
    )
    h_and_s_metrics: List[str] = Field(
        default_factory=lambda: [
            "fatalities",
            "recordable_work_related_accidents",
            "days_lost_to_injury",
            "occupational_diseases",
        ],
        description="Health and safety metrics per S1-14",
    )
    collective_bargaining_coverage: bool = Field(
        True,
        description="Disclose collective bargaining coverage rate (S1-8 para 60)",
    )
    social_dialogue_tracked: bool = Field(
        True,
        description="Track social dialogue at European Works Council or equivalent level (S1-8)",
    )
    diversity_targets: bool = Field(
        True,
        description="Track diversity metrics and targets (S1-9 para 66)",
    )
    diversity_dimensions: List[str] = Field(
        default_factory=lambda: [
            "gender",
            "age_group",
            "disability",
            "ethnicity",
        ],
        description="Diversity dimensions to track per S1-9",
    )
    work_life_balance_tracked: bool = Field(
        True,
        description="Track work-life balance indicators: family leave, flexible working (S1-15)",
    )
    social_protection_tracked: bool = Field(
        True,
        description="Track social protection coverage for employees (S1-11 para 76)",
    )
    persons_with_disabilities_tracked: bool = Field(
        True,
        description="Track employment of persons with disabilities (S1-12 para 80)",
    )
    human_rights_incidents_tracked: bool = Field(
        True,
        description="Track severe human rights incidents and complaints (S1-17 para 104)",
    )

    @field_validator("adequate_wage_benchmark")
    @classmethod
    def validate_wage_benchmark(cls, v: str) -> str:
        """Validate adequate wage benchmark option."""
        valid = {"legal_minimum", "living_wage", "applicable_benchmarks"}
        if v not in valid:
            raise ValueError(
                f"Invalid wage benchmark: {v}. Valid: {sorted(valid)}"
            )
        return v


class S2ValueChainConfig(BaseModel):
    """Configuration for ESRS S2 Workers in the Value Chain engine.

    Covers policies, engagement, remediation, actions, and targets related
    to workers in the upstream and downstream value chain per S2-1 through S2-5.
    """

    enabled: bool = Field(
        True,
        description="Enable S2 Value Chain Workers disclosures",
    )
    value_chain_workers_mapped: bool = Field(
        True,
        description="Whether the undertaking has mapped workers in its value chain (S2-1 para 15)",
    )
    value_chain_tiers: List[str] = Field(
        default_factory=lambda: ["tier_1", "tier_2", "beyond_tier_2"],
        description="Value chain tiers covered in worker assessment",
    )
    high_risk_sectors: List[str] = Field(
        default_factory=lambda: [
            "agriculture",
            "mining",
            "textiles",
            "electronics",
            "construction",
        ],
        description="High-risk sectors for value chain worker impacts",
    )
    high_risk_geographies: List[str] = Field(
        default_factory=lambda: [],
        description="High-risk geographies for value chain worker impacts",
    )
    engagement_mechanisms: List[str] = Field(
        default_factory=lambda: [
            "supplier_audits",
            "worker_surveys",
            "trade_union_engagement",
            "multi_stakeholder_initiatives",
        ],
        description="Mechanisms for engaging with value chain workers (S2-2)",
    )
    grievance_channels: List[str] = Field(
        default_factory=lambda: [
            "supplier_hotline",
            "whistleblower_mechanism",
            "third_party_mechanism",
        ],
        description="Grievance channels for value chain workers (S2-3)",
    )
    due_diligence_scope: ValueChainScope = Field(
        ValueChainScope.UPSTREAM,
        description="Scope of human rights due diligence for value chain workers",
    )
    child_labor_risk_assessed: bool = Field(
        True,
        description="Whether child labor risk is specifically assessed",
    )
    forced_labor_risk_assessed: bool = Field(
        True,
        description="Whether forced labor risk is specifically assessed",
    )


class S3CommunitiesConfig(BaseModel):
    """Configuration for ESRS S3 Affected Communities engine.

    Covers policies, engagement, remediation, actions, and targets related
    to affected communities including indigenous peoples per S3-1 through S3-5.
    """

    enabled: bool = Field(
        True,
        description="Enable S3 Affected Communities disclosures",
    )
    affected_communities_identified: bool = Field(
        True,
        description="Whether affected communities have been identified (S3-1 para 15)",
    )
    community_categories: List[str] = Field(
        default_factory=lambda: [
            "local_communities",
            "indigenous_peoples",
            "environmental_defenders",
            "land_rights_holders",
        ],
        description="Categories of affected communities to assess",
    )
    indigenous_peoples_relevant: bool = Field(
        False,
        description="Whether indigenous peoples are affected by operations (S3 para 7)",
    )
    fpic_processes: bool = Field(
        False,
        description="Whether Free Prior and Informed Consent (FPIC) processes are in place",
    )
    fpic_standard: str = Field(
        "ILO_169",
        description="FPIC standard applied: ILO_169, UNDRIP, IFC_PS7",
    )
    land_acquisition_policies: bool = Field(
        False,
        description="Whether policies exist for land acquisition and resettlement",
    )
    community_investment_tracked: bool = Field(
        True,
        description="Track community investment and development programs",
    )
    human_rights_impact_assessment: bool = Field(
        True,
        description="Whether human rights impact assessments are conducted for communities",
    )
    community_grievance_mechanism: bool = Field(
        True,
        description="Whether a community-level grievance mechanism exists (S3-3)",
    )

    @field_validator("fpic_standard")
    @classmethod
    def validate_fpic_standard(cls, v: str) -> str:
        """Validate FPIC standard reference."""
        valid = {"ILO_169", "UNDRIP", "IFC_PS7", "CUSTOM"}
        if v not in valid:
            raise ValueError(
                f"Unknown FPIC standard: {v}. Valid: {sorted(valid)}"
            )
        return v


class S4ConsumersConfig(BaseModel):
    """Configuration for ESRS S4 Consumers and End-Users engine.

    Covers policies, engagement, remediation, actions, and targets related
    to consumer and end-user impacts including product safety, data privacy,
    and inclusion of vulnerable consumers per S4-1 through S4-5.
    """

    enabled: bool = Field(
        True,
        description="Enable S4 Consumers and End-Users disclosures",
    )
    product_safety_tracked: bool = Field(
        True,
        description="Track product and service safety incidents and recalls (S4-4 para 33)",
    )
    product_safety_metrics: List[str] = Field(
        default_factory=lambda: [
            "recalls",
            "safety_incidents",
            "complaints",
            "regulatory_actions",
        ],
        description="Product safety metrics to track",
    )
    data_privacy_assessed: bool = Field(
        True,
        description="Assess data privacy and protection practices (S4-1 para 16)",
    )
    data_privacy_framework: str = Field(
        "GDPR",
        description="Data privacy framework: GDPR, CCPA, LGPD, CUSTOM",
    )
    data_breach_tracking: bool = Field(
        True,
        description="Track data breach incidents and notification compliance",
    )
    vulnerable_consumers_identified: bool = Field(
        True,
        description="Identify and assess impacts on vulnerable consumers (S4-1 para 17)",
    )
    vulnerable_consumer_categories: List[str] = Field(
        default_factory=lambda: [
            "children",
            "elderly",
            "persons_with_disabilities",
            "low_income",
        ],
        description="Categories of vulnerable consumers assessed",
    )
    responsible_marketing_policy: bool = Field(
        True,
        description="Whether responsible marketing/advertising policies exist",
    )
    accessibility_tracking: bool = Field(
        False,
        description="Track product/service accessibility for persons with disabilities",
    )
    consumer_grievance_mechanism: bool = Field(
        True,
        description="Whether a consumer grievance/complaint mechanism exists (S4-3)",
    )

    @field_validator("data_privacy_framework")
    @classmethod
    def validate_privacy_framework(cls, v: str) -> str:
        """Validate data privacy framework reference."""
        valid = {"GDPR", "CCPA", "LGPD", "PIPL", "POPIA", "CUSTOM"}
        if v not in valid:
            raise ValueError(
                f"Unknown privacy framework: {v}. Valid: {sorted(valid)}"
            )
        return v


class G1GovernanceConfig(BaseModel):
    """Configuration for ESRS G1 Business Conduct engine.

    Covers business conduct policies, corporate culture, supplier management,
    anti-corruption, political engagement, lobbying, and payment practices
    per G1-1 through G1-6.
    """

    enabled: bool = Field(
        True,
        description="Enable G1 Business Conduct disclosures",
    )
    code_of_conduct_exists: bool = Field(
        True,
        description="Whether a code of conduct/ethics exists (G1-1 para 11)",
    )
    code_of_conduct_coverage: str = Field(
        "all_employees_and_contractors",
        description="Coverage of code of conduct: all_employees, all_employees_and_contractors, full_value_chain",
    )
    anti_corruption_training: bool = Field(
        True,
        description="Whether anti-corruption/bribery training is provided (G1-3 para 21)",
    )
    anti_corruption_training_coverage_pct: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of relevant employees trained on anti-corruption",
    )
    anti_corruption_risk_assessment: bool = Field(
        True,
        description="Whether corruption risk assessments are conducted (G1-3 para 22)",
    )
    corruption_risk_level: CorruptionRiskLevel = Field(
        CorruptionRiskLevel.LOW,
        description="Overall assessed corruption risk level for operations",
    )
    corruption_incidents_tracked: bool = Field(
        True,
        description="Track confirmed incidents of corruption or bribery (G1-4 para 26)",
    )
    political_engagement_tracked: bool = Field(
        True,
        description="Track political engagement and lobbying activities (G1-5 para 30)",
    )
    political_contributions_policy: str = Field(
        "prohibited",
        description="Political contributions policy: prohibited, allowed_with_disclosure, no_policy",
    )
    lobbying_expenditure_tracked: bool = Field(
        True,
        description="Track lobbying expenditure and memberships in trade associations (G1-5 para 32)",
    )
    eu_transparency_register: bool = Field(
        False,
        description="Whether registered in the EU Transparency Register",
    )
    payment_practices_tracked: bool = Field(
        True,
        description="Track payment practices including average payment terms (G1-6 para 36)",
    )
    payment_practice_metrics: List[str] = Field(
        default_factory=lambda: [
            "average_payment_term_days",
            "standard_payment_terms",
            "late_payment_interest_rate",
            "invoices_paid_within_terms_pct",
        ],
        description="Payment practice metrics per G1-6",
    )
    sme_payment_focus: bool = Field(
        True,
        description="Specific focus on payment practices toward SMEs (G1-6 para 38)",
    )
    whistleblower_channel: bool = Field(
        True,
        description="Whether a whistleblower/reporting channel exists (G1-1 para 13)",
    )
    whistleblower_protection_policy: bool = Field(
        True,
        description="Whether whistleblower protection policy complies with EU Directive 2019/1937",
    )

    @field_validator("political_contributions_policy")
    @classmethod
    def validate_political_policy(cls, v: str) -> str:
        """Validate political contributions policy option."""
        valid = {"prohibited", "allowed_with_disclosure", "no_policy"}
        if v not in valid:
            raise ValueError(
                f"Invalid political contributions policy: {v}. Valid: {sorted(valid)}"
            )
        return v


class OrchestratorConfig(BaseModel):
    """Configuration for the ESRS Full Coverage pipeline orchestrator.

    Controls execution behavior, concurrency, timeouts, and retry logic
    for the multi-standard processing pipeline.
    """

    parallel_execution: bool = Field(
        True,
        description="Enable parallel execution of independent standard engines",
    )
    max_concurrent_engines: int = Field(
        4,
        ge=1,
        le=12,
        description="Maximum number of standard engines running concurrently",
    )
    timeout_seconds: int = Field(
        600,
        ge=60,
        le=3600,
        description="Maximum execution time per engine in seconds",
    )
    retry_count: int = Field(
        3,
        ge=0,
        le=10,
        description="Number of retries on engine failure",
    )
    retry_backoff_seconds: int = Field(
        5,
        ge=1,
        le=60,
        description="Base backoff delay between retries in seconds",
    )
    retry_backoff_multiplier: float = Field(
        2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier for retries",
    )
    fail_fast: bool = Field(
        False,
        description="Stop pipeline on first engine failure (False = continue remaining)",
    )
    execution_order: List[str] = Field(
        default_factory=lambda: [
            "ESRS_2", "E1", "E2", "E3", "E4", "E5",
            "S1", "S2", "S3", "S4", "G1",
        ],
        description="Execution order of standard engines (ESRS 2 always first)",
    )
    checkpoint_enabled: bool = Field(
        True,
        description="Enable checkpoint saving for pipeline resumption",
    )
    checkpoint_interval_seconds: int = Field(
        60,
        ge=10,
        le=300,
        description="Interval for saving pipeline state checkpoints",
    )
    memory_limit_mb: int = Field(
        4096,
        ge=512,
        le=32768,
        description="Memory limit per engine in megabytes",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=50000,
        description="Batch size for bulk data processing within engines",
    )

    @field_validator("execution_order")
    @classmethod
    def validate_execution_order(cls, v: List[str]) -> List[str]:
        """Validate execution order contains valid standard identifiers."""
        valid_standards = {
            "ESRS_2", "E1", "E2", "E3", "E4", "E5",
            "S1", "S2", "S3", "S4", "G1",
        }
        invalid = [s for s in v if s not in valid_standards]
        if invalid:
            raise ValueError(
                f"Invalid standards in execution_order: {invalid}. "
                f"Valid: {sorted(valid_standards)}"
            )
        if "ESRS_2" in v and v[0] != "ESRS_2":
            logger.warning(
                "ESRS 2 should be first in execution order as it provides "
                "cross-cutting context for all topical standards."
            )
        return v


class ReportingConfig(BaseModel):
    """Configuration for the consolidated ESRS report generation engine.

    Controls output formats, assurance, XBRL tagging, and provenance
    tracking for the full sustainability statement.
    """

    enabled: bool = Field(
        True,
        description="Enable consolidated ESRS report generation",
    )
    reporting_period_start: str = Field(
        "2025-01-01",
        description="Reporting period start date (YYYY-MM-DD)",
    )
    reporting_period_end: str = Field(
        "2025-12-31",
        description="Reporting period end date (YYYY-MM-DD)",
    )
    base_year: int = Field(
        2024,
        ge=2020,
        le=2030,
        description="Comparative base year for year-over-year analysis",
    )
    currency: str = Field(
        "EUR",
        description="Reporting currency for all financial values",
    )
    xbrl_tagging_enabled: bool = Field(
        True,
        description="Tag all quantitative datapoints with EFRAG XBRL taxonomy identifiers",
    )
    xbrl_taxonomy_version: str = Field(
        "EFRAG_ESRS_2024",
        description="XBRL taxonomy version to apply",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Level of external assurance for sustainability statement",
    )
    assurance_provider: Optional[str] = Field(
        None,
        description="Name of the external assurance provider",
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ["PDF", "XBRL", "HTML", "JSON"],
        description="Output formats for the sustainability statement",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all calculated values",
    )
    multi_language: bool = Field(
        False,
        description="Enable multi-language report generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages for reports (ISO 639-1 codes)",
    )
    review_workflow: bool = Field(
        True,
        description="Enable review and approval workflow for disclosure documents",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved disclosure documents",
    )
    audit_trail_report: bool = Field(
        True,
        description="Generate separate audit trail report for assurance readiness",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary across all material standards",
    )
    year_over_year_comparison: bool = Field(
        True,
        description="Include year-over-year comparison for all quantitative DRs",
    )
    data_quality_disclosure: bool = Field(
        True,
        description="Disclose data quality scores per data source",
    )
    assumption_disclosure: bool = Field(
        True,
        description="Disclose key assumptions, estimates, and judgments",
    )
    connectivity_with_financial: bool = Field(
        True,
        description="Include connectivity with financial statements per ESRS 1 sect. 9",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Report and audit trail retention period in years",
    )

    @field_validator("reporting_period_start", "reporting_period_end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid date format: {v}. Expected YYYY-MM-DD."
            )
        return v

    @model_validator(mode="after")
    def validate_period_order(self) -> "ReportingConfig":
        """Validate reporting period end is after start."""
        start = datetime.strptime(self.reporting_period_start, "%Y-%m-%d")
        end = datetime.strptime(self.reporting_period_end, "%Y-%m-%d")
        if end <= start:
            raise ValueError(
                f"Reporting period end ({self.reporting_period_end}) must be "
                f"after start ({self.reporting_period_start})."
            )
        return self


# =============================================================================
# Main Configuration Model
# =============================================================================


class ESRSFullCoverageConfig(BaseModel):
    """Main configuration for PACK-017 ESRS Full Coverage Pack.

    This is the root configuration model that composes all 12 sub-configurations
    covering every ESRS standard. The materiality_results dictionary drives
    which topical standards are activated in the disclosure pipeline.

    Standards subject to materiality assessment (per ESRS 1 sect. 3.5):
        - E1 through E5 (environmental)
        - S1 through S4 (social)
        - G1 (governance)

    Standards always mandatory:
        - ESRS 2 (general disclosures)

    Standards providing methodology (no separate disclosures):
        - ESRS 1 (general requirements)
    """

    # Company identification
    company_name: str = Field(
        "",
        description="Legal entity name of the undertaking",
    )
    company_lei: Optional[str] = Field(
        None,
        description="Legal Entity Identifier (LEI) for the undertaking",
    )
    reporting_year: int = Field(
        2025,
        ge=2024,
        le=2035,
        description="Reporting year for CSRD sustainability statement",
    )
    sector: str = Field(
        "GENERAL",
        description="Primary NACE sector code or label",
    )
    nace_codes: List[str] = Field(
        default_factory=list,
        description="NACE Rev. 2 codes for the undertaking's activities",
    )
    fiscal_year_end: str = Field(
        "12-31",
        description="Fiscal year end date (MM-DD)",
    )
    currency: str = Field(
        "EUR",
        description="Reporting currency for all financial values",
    )
    reporting_boundary: ReportingBoundary = Field(
        ReportingBoundary.CONSOLIDATED,
        description="Reporting boundary per ESRS 1 para 62",
    )
    compliance_level: ComplianceLevel = Field(
        ComplianceLevel.FULL,
        description="ESRS compliance level: FULL, PHASE_IN, VOLUNTARY, OMNIBUS_REDUCED",
    )
    employee_count: Optional[int] = Field(
        None,
        ge=0,
        description="Total employee count for phase-in and Omnibus threshold determination",
    )
    net_turnover_eur: Optional[float] = Field(
        None,
        ge=0.0,
        description="Net turnover in EUR for size classification",
    )
    is_listed: bool = Field(
        True,
        description="Whether the undertaking is listed on an EU-regulated market",
    )

    # Materiality results - drives which standards are active
    materiality_results: Dict[str, MaterialityStatus] = Field(
        default_factory=lambda: {
            ESRSStandard.ESRS_2.value: MaterialityStatus.MATERIAL,
            ESRSStandard.E1.value: MaterialityStatus.MATERIAL,
            ESRSStandard.E2.value: MaterialityStatus.PENDING,
            ESRSStandard.E3.value: MaterialityStatus.PENDING,
            ESRSStandard.E4.value: MaterialityStatus.PENDING,
            ESRSStandard.E5.value: MaterialityStatus.PENDING,
            ESRSStandard.S1.value: MaterialityStatus.MATERIAL,
            ESRSStandard.S2.value: MaterialityStatus.PENDING,
            ESRSStandard.S3.value: MaterialityStatus.PENDING,
            ESRSStandard.S4.value: MaterialityStatus.PENDING,
            ESRSStandard.G1.value: MaterialityStatus.PENDING,
        },
        description="Double materiality assessment results per standard",
    )

    # Engine sub-configurations (all 12)
    esrs2: ESRS2Config = Field(
        default_factory=ESRS2Config,
        description="ESRS 2 General Disclosures configuration",
    )
    e2_pollution: E2PollutionConfig = Field(
        default_factory=E2PollutionConfig,
        description="E2 Pollution configuration",
    )
    e3_water: E3WaterConfig = Field(
        default_factory=E3WaterConfig,
        description="E3 Water and Marine Resources configuration",
    )
    e4_biodiversity: E4BiodiversityConfig = Field(
        default_factory=E4BiodiversityConfig,
        description="E4 Biodiversity and Ecosystems configuration",
    )
    e5_circular: E5CircularConfig = Field(
        default_factory=E5CircularConfig,
        description="E5 Resource Use and Circular Economy configuration",
    )
    s1_workforce: S1WorkforceConfig = Field(
        default_factory=S1WorkforceConfig,
        description="S1 Own Workforce configuration",
    )
    s2_value_chain: S2ValueChainConfig = Field(
        default_factory=S2ValueChainConfig,
        description="S2 Workers in the Value Chain configuration",
    )
    s3_communities: S3CommunitiesConfig = Field(
        default_factory=S3CommunitiesConfig,
        description="S3 Affected Communities configuration",
    )
    s4_consumers: S4ConsumersConfig = Field(
        default_factory=S4ConsumersConfig,
        description="S4 Consumers and End-Users configuration",
    )
    g1_governance: G1GovernanceConfig = Field(
        default_factory=G1GovernanceConfig,
        description="G1 Business Conduct configuration",
    )
    orchestrator: OrchestratorConfig = Field(
        default_factory=OrchestratorConfig,
        description="Pipeline orchestrator configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Consolidated report generation configuration",
    )

    @model_validator(mode="after")
    def validate_esrs2_always_material(self) -> "ESRSFullCoverageConfig":
        """Ensure ESRS 2 is always marked as material (mandatory for all)."""
        esrs2_key = ESRSStandard.ESRS_2.value
        if esrs2_key in self.materiality_results:
            if self.materiality_results[esrs2_key] != MaterialityStatus.MATERIAL:
                logger.warning(
                    "ESRS 2 General Disclosures is mandatory for all undertakings. "
                    "Overriding materiality status to MATERIAL."
                )
                self.materiality_results[esrs2_key] = MaterialityStatus.MATERIAL
        else:
            self.materiality_results[esrs2_key] = MaterialityStatus.MATERIAL
        return self

    @model_validator(mode="after")
    def validate_e1_large_undertaking(self) -> "ESRSFullCoverageConfig":
        """Warn if E1 is not material for large undertakings (>750 employees)."""
        e1_key = ESRSStandard.E1.value
        if (
            self.employee_count is not None
            and self.employee_count > 750
            and self.materiality_results.get(e1_key) == MaterialityStatus.NOT_MATERIAL
        ):
            logger.warning(
                "E1 Climate Change is not material but the undertaking has >750 "
                "employees. Per ESRS 1 para 32, E1 materiality should be carefully "
                "justified for large undertakings."
            )
        return self

    @model_validator(mode="after")
    def validate_disabled_engines_match_materiality(self) -> "ESRSFullCoverageConfig":
        """Warn if an engine is disabled but its standard is marked material."""
        engine_map: Dict[str, BaseModel] = {
            ESRSStandard.ESRS_2.value: self.esrs2,
            ESRSStandard.E2.value: self.e2_pollution,
            ESRSStandard.E3.value: self.e3_water,
            ESRSStandard.E4.value: self.e4_biodiversity,
            ESRSStandard.E5.value: self.e5_circular,
            ESRSStandard.S1.value: self.s1_workforce,
            ESRSStandard.S2.value: self.s2_value_chain,
            ESRSStandard.S3.value: self.s3_communities,
            ESRSStandard.S4.value: self.s4_consumers,
            ESRSStandard.G1.value: self.g1_governance,
        }
        for std_key, engine in engine_map.items():
            materiality = self.materiality_results.get(std_key)
            engine_enabled = getattr(engine, "enabled", True)
            if materiality == MaterialityStatus.MATERIAL and not engine_enabled:
                logger.warning(
                    "Standard %s is marked MATERIAL but its engine is disabled. "
                    "Enable the engine to produce required disclosures.",
                    std_key,
                )
            if materiality == MaterialityStatus.NOT_MATERIAL and engine_enabled:
                logger.info(
                    "Standard %s is NOT_MATERIAL but its engine is enabled. "
                    "Engine output will be marked as voluntary disclosure.",
                    std_key,
                )
        return self

    @model_validator(mode="after")
    def validate_omnibus_thresholds(self) -> "ESRSFullCoverageConfig":
        """Apply Omnibus simplification logic for smaller undertakings."""
        if self.compliance_level == ComplianceLevel.OMNIBUS_REDUCED:
            if self.employee_count is not None and self.employee_count >= 1000:
                logger.warning(
                    "OMNIBUS_REDUCED compliance level selected but employee count "
                    "is %d (>=1000). Omnibus simplification applies primarily to "
                    "undertakings with <1000 employees.",
                    self.employee_count,
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-017.

    Handles preset loading, environment variable overrides, YAML loading,
    configuration merging, and validation.
    """

    pack: ESRSFullCoverageConfig = Field(
        default_factory=ESRSFullCoverageConfig,
        description="Main ESRS Full Coverage configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-017-esrs-full-coverage",
        description="Pack identifier",
    )
    created_at: Optional[str] = Field(
        None,
        description="Configuration creation timestamp (ISO 8601)",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (manufacturing, financial_services, etc.)
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in AVAILABLE_PRESETS.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)

        pack_config = ESRSFullCoverageConfig(**preset_data)
        return cls(
            pack=pack_config,
            preset_name=preset_name,
            created_at=datetime.utcnow().isoformat(),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = ESRSFullCoverageConfig(**config_data)
        return cls(
            pack=pack_config,
            created_at=datetime.utcnow().isoformat(),
        )

    def merge(self, overrides: Dict[str, Any]) -> "PackConfig":
        """Merge additional overrides into the current configuration.

        Args:
            overrides: Dictionary of configuration overrides.

        Returns:
            New PackConfig instance with overrides applied.
        """
        current_data = self.pack.model_dump()
        merged_data = self._deep_merge(current_data, overrides)
        new_pack = ESRSFullCoverageConfig(**merged_data)
        return PackConfig(
            pack=new_pack,
            preset_name=self.preset_name,
            config_version=self.config_version,
            pack_id=self.pack_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def validate(self) -> List[str]:
        """Run comprehensive validation on the current configuration.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with ESRS_PACK_ are loaded and mapped
        to configuration keys. Nested keys use double underscore.

        Example: ESRS_PACK_S1_WORKFORCE__GENDER_PAY_GAP_TRACKED=true
        """
        overrides: Dict[str, Any] = {}
        prefix = "ESRS_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            SHA-256 hex digest string.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def get_material_standard_count(self) -> int:
        """Return the count of standards marked as MATERIAL.

        Returns:
            Integer count of material standards.
        """
        return sum(
            1
            for status in self.pack.materiality_results.values()
            if status == MaterialityStatus.MATERIAL
            or status == MaterialityStatus.MATERIAL.value
        )

    def get_active_engines(self) -> List[str]:
        """Return list of standard keys whose engines are enabled and material.

        Returns:
            List of standard key strings (e.g., ["ESRS_2", "E1", "S1"]).
        """
        engine_map: Dict[str, BaseModel] = {
            ESRSStandard.ESRS_2.value: self.pack.esrs2,
            ESRSStandard.E2.value: self.pack.e2_pollution,
            ESRSStandard.E3.value: self.pack.e3_water,
            ESRSStandard.E4.value: self.pack.e4_biodiversity,
            ESRSStandard.E5.value: self.pack.e5_circular,
            ESRSStandard.S1.value: self.pack.s1_workforce,
            ESRSStandard.S2.value: self.pack.s2_value_chain,
            ESRSStandard.S3.value: self.pack.s3_communities,
            ESRSStandard.S4.value: self.pack.s4_consumers,
            ESRSStandard.G1.value: self.pack.g1_governance,
        }
        active = []
        for std_key, engine in engine_map.items():
            materiality = self.pack.materiality_results.get(std_key)
            engine_enabled = getattr(engine, "enabled", True)
            is_material = (
                materiality == MaterialityStatus.MATERIAL
                or materiality == MaterialityStatus.MATERIAL.value
            )
            if is_material and engine_enabled:
                active.append(std_key)
        return active


# =============================================================================
# Utility Functions
# =============================================================================


def get_disclosure_requirements(standard: ESRSStandard) -> Dict[str, Dict[str, Any]]:
    """Get all disclosure requirements for a given ESRS standard.

    Args:
        standard: ESRSStandard enum value.

    Returns:
        Dictionary mapping DR identifiers to their metadata.
        Returns empty dict if the standard has no DR definitions
        (e.g., ESRS_1 which is a methodology standard).
    """
    key = standard.value
    return _ALL_DISCLOSURE_REQUIREMENTS.get(key, {})


def get_material_standards(
    config: ESRSFullCoverageConfig,
) -> List[ESRSStandard]:
    """Get list of standards marked as MATERIAL in the configuration.

    Args:
        config: ESRSFullCoverageConfig instance.

    Returns:
        List of ESRSStandard enum values that are material.
    """
    material: List[ESRSStandard] = []
    for std_key, status in config.materiality_results.items():
        if status == MaterialityStatus.MATERIAL or status == MaterialityStatus.MATERIAL.value:
            try:
                material.append(ESRSStandard(std_key))
            except ValueError:
                logger.warning(
                    "Unknown standard key in materiality_results: %s", std_key
                )
    return material


def get_total_disclosure_count() -> int:
    """Get the total number of disclosure requirements across all ESRS standards.

    The ESRS Set 1 defines 82 disclosure requirements in total:
        - ESRS 2:  10 DRs (GOV-1 to GOV-5, SBM-1 to SBM-3, IRO-1, IRO-2)
        - E1:       9 DRs (E1-1 to E1-9)
        - E2:       6 DRs (E2-1 to E2-6)
        - E3:       5 DRs (E3-1 to E3-5)
        - E4:       6 DRs (E4-1 to E4-6)
        - E5:       6 DRs (E5-1 to E5-6)
        - S1:      17 DRs (S1-1 to S1-17)
        - S2:       5 DRs (S2-1 to S2-5)
        - S3:       5 DRs (S3-1 to S3-5)
        - S4:       5 DRs (S4-1 to S4-5)
        - G1:       6 DRs (G1-1 to G1-6)
        - ESRS 1:   2 DRs (basis for preparation, not counted in topical)

    Returns:
        Total count of disclosure requirements (82).
    """
    total = 0
    for dr_dict in _ALL_DISCLOSURE_REQUIREMENTS.values():
        total += len(dr_dict)
    # ESRS 1 has 2 basis-for-preparation DRs not tracked in topical DR dicts
    total += 2
    return total


def get_mandatory_disclosures() -> List[str]:
    """Get list of all mandatory disclosure requirements.

    Mandatory disclosures are those that must be reported regardless of
    materiality assessment. In ESRS Set 1, all ESRS 2 DRs are mandatory.

    Returns:
        List of DR identifier strings (e.g., ["GOV-1", "GOV-2", ...]).
    """
    mandatory: List[str] = []
    for dr_dict in _ALL_DISCLOSURE_REQUIREMENTS.values():
        for dr_id, dr_info in dr_dict.items():
            if dr_info.get("mandatory", False):
                mandatory.append(dr_id)
    return sorted(mandatory)


def get_phase_in_disclosures(year: int) -> List[str]:
    """Get list of disclosure requirements that phase in for a given year.

    Phase-in provisions allow certain DRs to be deferred to later reporting
    years per Delegated Regulation (EU) 2023/2772 Appendix C.

    Args:
        year: Reporting year to check (e.g., 2025, 2026).

    Returns:
        List of DR identifier strings that phase in for the given year.
    """
    phase_in: List[str] = []
    for dr_dict in _ALL_DISCLOSURE_REQUIREMENTS.values():
        for dr_id, dr_info in dr_dict.items():
            phase_in_year = dr_info.get("phase_in_year")
            if phase_in_year is not None and phase_in_year == year:
                phase_in.append(dr_id)
    return sorted(phase_in)


def get_all_disclosure_ids() -> List[str]:
    """Get a flat list of all disclosure requirement identifiers.

    Returns:
        Sorted list of all DR identifier strings across all standards.
    """
    all_ids: List[str] = []
    for dr_dict in _ALL_DISCLOSURE_REQUIREMENTS.values():
        all_ids.extend(dr_dict.keys())
    return sorted(all_ids)


def get_disclosure_info(dr_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific disclosure requirement.

    Args:
        dr_id: Disclosure requirement identifier (e.g., "E1-6", "GOV-1", "S1-14").

    Returns:
        Dictionary with DR name, paragraphs, application requirements,
        mandatory flag, quantitative flag, and phase_in_year.
        Returns default values if DR is not found.
    """
    for dr_dict in _ALL_DISCLOSURE_REQUIREMENTS.values():
        if dr_id in dr_dict:
            return dr_dict[dr_id]
    return {
        "name": dr_id,
        "paragraphs": "Unknown",
        "application_requirements": "Unknown",
        "mandatory": False,
        "quantitative": False,
        "phase_in_year": None,
    }


def get_quantitative_disclosures() -> List[str]:
    """Get list of all quantitative disclosure requirements.

    Quantitative DRs require numeric datapoints and are typically subject
    to XBRL tagging per EFRAG taxonomy.

    Returns:
        Sorted list of DR identifiers that are quantitative.
    """
    quantitative: List[str] = []
    for dr_dict in _ALL_DISCLOSURE_REQUIREMENTS.values():
        for dr_id, dr_info in dr_dict.items():
            if dr_info.get("quantitative", False):
                quantitative.append(dr_id)
    return sorted(quantitative)


def get_standard_dr_count(standard: ESRSStandard) -> int:
    """Get the number of disclosure requirements for a specific standard.

    Args:
        standard: ESRSStandard enum value.

    Returns:
        Number of DRs defined for the standard.
    """
    drs = get_disclosure_requirements(standard)
    return len(drs)


def validate_config(config: ESRSFullCoverageConfig) -> List[str]:
    """Validate an ESRS Full Coverage configuration and return any warnings.

    Performs comprehensive cross-field validation checks across all
    sub-configurations and materiality results.

    Args:
        config: ESRSFullCoverageConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check ESRS 2 is material (always mandatory)
    esrs2_status = config.materiality_results.get(ESRSStandard.ESRS_2.value)
    if esrs2_status != MaterialityStatus.MATERIAL and esrs2_status != MaterialityStatus.MATERIAL.value:
        warnings.append(
            "ESRS 2 General Disclosures must always be MATERIAL. "
            "It is mandatory for all undertakings under CSRD."
        )

    # Check no standards are still PENDING
    pending_count = sum(
        1
        for status in config.materiality_results.values()
        if status == MaterialityStatus.PENDING or status == MaterialityStatus.PENDING.value
    )
    if pending_count > 0:
        warnings.append(
            f"{pending_count} standard(s) have PENDING materiality status. "
            f"Complete double materiality assessment before disclosure."
        )

    # Check E1 materiality for large undertakings
    if (
        config.employee_count is not None
        and config.employee_count > 750
    ):
        e1_status = config.materiality_results.get(ESRSStandard.E1.value)
        if e1_status == MaterialityStatus.NOT_MATERIAL or e1_status == MaterialityStatus.NOT_MATERIAL.value:
            warnings.append(
                "E1 Climate Change marked NOT_MATERIAL for undertaking with "
                f"{config.employee_count} employees (>750). E1 requires "
                "detailed justification for non-materiality in large undertakings."
            )

    # Check S1 workforce config consistency
    if (
        config.materiality_results.get(ESRSStandard.S1.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.S1.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.s1_workforce.enabled:
            if not config.s1_workforce.h_and_s_incidents_tracked:
                warnings.append(
                    "S1 is material but health and safety incident tracking is "
                    "disabled. S1-14 requires disclosure of H&S metrics."
                )
            if not config.s1_workforce.gender_pay_gap_tracked:
                warnings.append(
                    "S1 is material but gender pay gap tracking is disabled. "
                    "S1-16 requires disclosure of remuneration metrics."
                )
            if not config.s1_workforce.collective_bargaining_coverage:
                warnings.append(
                    "S1 is material but collective bargaining coverage tracking "
                    "is disabled. S1-8 requires this disclosure."
                )

    # Check E2 pollution config consistency
    if (
        config.materiality_results.get(ESRSStandard.E2.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.E2.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.e2_pollution.enabled:
            if not any([
                config.e2_pollution.emission_to_air_tracked,
                config.e2_pollution.emission_to_water_tracked,
                config.e2_pollution.emission_to_soil_tracked,
            ]):
                warnings.append(
                    "E2 is material but no pollutant medium is tracked. "
                    "Enable at least one of air, water, or soil emission tracking."
                )

    # Check E3 water config consistency
    if (
        config.materiality_results.get(ESRSStandard.E3.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.E3.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.e3_water.enabled:
            if not config.e3_water.water_consumption_tracked:
                warnings.append(
                    "E3 is material but water consumption tracking is disabled. "
                    "E3-4 requires disclosure of water consumption."
                )

    # Check E4 biodiversity config consistency
    if (
        config.materiality_results.get(ESRSStandard.E4.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.E4.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.e4_biodiversity.enabled:
            if not config.e4_biodiversity.sites_near_sensitive_areas:
                warnings.append(
                    "E4 is material but proximity screening for biodiversity-sensitive "
                    "areas is disabled. E4-5 requires site-level assessment."
                )

    # Check E5 circular economy config consistency
    if (
        config.materiality_results.get(ESRSStandard.E5.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.E5.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.e5_circular.enabled:
            if not config.e5_circular.resource_inflows_tracked:
                warnings.append(
                    "E5 is material but resource inflow tracking is disabled. "
                    "E5-4 requires disclosure of resource inflows."
                )
            if not config.e5_circular.waste_generation_tracked:
                warnings.append(
                    "E5 is material but waste generation tracking is disabled. "
                    "E5-5 requires disclosure of resource outflows."
                )

    # Check G1 governance config consistency
    if (
        config.materiality_results.get(ESRSStandard.G1.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.G1.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.g1_governance.enabled:
            if not config.g1_governance.anti_corruption_training:
                warnings.append(
                    "G1 is material but anti-corruption training is disabled. "
                    "G1-3 requires disclosure of corruption prevention measures."
                )
            if not config.g1_governance.whistleblower_channel:
                warnings.append(
                    "G1 is material but no whistleblower channel is configured. "
                    "G1-1 para 13 requires disclosure of reporting channels."
                )

    # Check ESRS 2 governance body configuration
    if config.esrs2.enabled:
        if not config.esrs2.due_diligence_processes:
            warnings.append(
                "ESRS 2 GOV-4 requires disclosure of due diligence processes. "
                "due_diligence_processes is set to False."
            )
        if not config.esrs2.risk_management_integration:
            warnings.append(
                "ESRS 2 GOV-5 requires disclosure of risk management and internal "
                "controls. risk_management_integration is disabled."
            )

    # Check reporting configuration
    if config.reporting.enabled:
        if not config.reporting.sha256_provenance:
            warnings.append(
                "SHA-256 provenance tracking is disabled. Consider enabling "
                "for audit trail integrity and assurance readiness."
            )
        if not config.reporting.xbrl_tagging_enabled:
            warnings.append(
                "XBRL tagging is disabled. ESEF requirements mandate inline XBRL "
                "tagging for listed undertakings."
            )
        if (
            config.reporting.assurance_level == AssuranceLevel.NONE
            and config.is_listed
        ):
            warnings.append(
                "No assurance level configured for a listed undertaking. "
                "CSRD requires at least limited assurance."
            )

    # Check Omnibus compliance level
    if config.compliance_level == ComplianceLevel.OMNIBUS_REDUCED:
        if config.employee_count is not None and config.employee_count >= 1000:
            warnings.append(
                f"OMNIBUS_REDUCED compliance level but employee count is "
                f"{config.employee_count} (>=1000). Omnibus simplification "
                f"primarily targets undertakings with <1000 employees."
            )

    # Check S3 communities config for indigenous peoples
    if (
        config.materiality_results.get(ESRSStandard.S3.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.S3.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.s3_communities.enabled:
            if (
                config.s3_communities.indigenous_peoples_relevant
                and not config.s3_communities.fpic_processes
            ):
                warnings.append(
                    "S3: Indigenous peoples are marked as relevant but FPIC "
                    "processes are not in place. UNDRIP and ILO 169 require FPIC."
                )

    # Check S4 consumers data privacy
    if (
        config.materiality_results.get(ESRSStandard.S4.value) == MaterialityStatus.MATERIAL
        or config.materiality_results.get(ESRSStandard.S4.value) == MaterialityStatus.MATERIAL.value
    ):
        if config.s4_consumers.enabled:
            if not config.s4_consumers.data_privacy_assessed:
                warnings.append(
                    "S4 is material but data privacy assessment is disabled. "
                    "S4-1 requires assessment of data privacy practices."
                )

    return warnings


def get_default_config(sector: str = "GENERAL") -> ESRSFullCoverageConfig:
    """Get default ESRS Full Coverage configuration for a given sector.

    Args:
        sector: Sector identifier (e.g., "MANUFACTURING", "GENERAL").

    Returns:
        ESRSFullCoverageConfig instance with sector-appropriate defaults.
    """
    return ESRSFullCoverageConfig(sector=sector)


def list_available_presets() -> Dict[str, str]:
    """List all available ESRS Full Coverage configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def get_standard_label(standard: ESRSStandard) -> str:
    """Get human-readable label for an ESRS standard.

    Args:
        standard: ESRSStandard enum value.

    Returns:
        Human-readable label string.
    """
    labels: Dict[ESRSStandard, str] = {
        ESRSStandard.ESRS_1: "ESRS 1 - General Requirements",
        ESRSStandard.ESRS_2: "ESRS 2 - General Disclosures",
        ESRSStandard.E1: "ESRS E1 - Climate Change",
        ESRSStandard.E2: "ESRS E2 - Pollution",
        ESRSStandard.E3: "ESRS E3 - Water and Marine Resources",
        ESRSStandard.E4: "ESRS E4 - Biodiversity and Ecosystems",
        ESRSStandard.E5: "ESRS E5 - Resource Use and Circular Economy",
        ESRSStandard.S1: "ESRS S1 - Own Workforce",
        ESRSStandard.S2: "ESRS S2 - Workers in the Value Chain",
        ESRSStandard.S3: "ESRS S3 - Affected Communities",
        ESRSStandard.S4: "ESRS S4 - Consumers and End-Users",
        ESRSStandard.G1: "ESRS G1 - Business Conduct",
    }
    return labels.get(standard, standard.value)
