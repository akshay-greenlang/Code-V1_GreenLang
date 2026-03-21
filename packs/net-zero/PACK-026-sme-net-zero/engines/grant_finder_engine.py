# -*- coding: utf-8 -*-
"""
GrantFinderEngine - PACK-026 SME Net Zero Pack Engine 7
=========================================================

Grant and incentive matching engine for SME decarbonization projects.
Database of 50+ major grant programs (UK, EU, US) with matching
algorithm based on industry code, company size, location, emissions
profile, and project type.

The engine scores each grant for eligibility (0-100) and returns
the top 3-5 matched programs with deadlines, funding ranges, and
application requirements.

Calculation Methodology:
    Eligibility Score = weighted_sum(
        industry_match * 30%,
        size_match * 20%,
        location_match * 20%,
        project_type_match * 20%,
        emissions_profile_match * 10%
    )

    Each dimension is scored 0-100:
        0 = Not eligible
        50 = Partial match
        100 = Full match

Regulatory References:
    - UK Industrial Energy Transformation Fund (IETF)
    - UK Energy Savings Opportunity Scheme (ESOS)
    - UK Salix Finance public sector decarbonization
    - EU Cohesion Fund / Just Transition Mechanism
    - EU LIFE Programme
    - EU Innovation Fund
    - US DOE Office of Energy Efficiency and Renewable Energy
    - US IRA (Inflation Reduction Act) provisions

Zero-Hallucination:
    - Grant database uses published program parameters
    - Scoring is deterministic (no ML/LLM)
    - All calculations use Decimal arithmetic
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-026 SME Net Zero Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GrantRegion(str, Enum):
    """Geographic region for grant eligibility."""
    UK = "uk"
    EU = "eu"
    US = "us"
    AUSTRALIA = "australia"
    CANADA = "canada"
    GLOBAL = "global"


class CompanySize(str, Enum):
    """Company size classification."""
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    ANY = "any"


class ProjectType(str, Enum):
    """Type of decarbonization project."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    EV_FLEET = "ev_fleet"
    BUILDING_RETROFIT = "building_retrofit"
    PROCESS_IMPROVEMENT = "process_improvement"
    WASTE_REDUCTION = "waste_reduction"
    CIRCULAR_ECONOMY = "circular_economy"
    RD_INNOVATION = "rd_innovation"
    CARBON_CAPTURE = "carbon_capture"
    HEAT_PUMP = "heat_pump"
    HYDROGEN = "hydrogen"
    GENERAL_DECARBONIZATION = "general_decarbonization"


class IndustryCode(str, Enum):
    """Simplified NACE/NAICS industry code."""
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    CONSTRUCTION = "construction"
    RETAIL = "retail"
    TRANSPORT = "transport"
    HOSPITALITY = "hospitality"
    IT_SERVICES = "it_services"
    FINANCIAL = "financial"
    PROFESSIONAL = "professional"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ANY = "any"


class GrantStatus(str, Enum):
    """Current status of the grant program."""
    OPEN = "open"
    UPCOMING = "upcoming"
    CLOSING_SOON = "closing_soon"
    CLOSED = "closed"
    ROLLING = "rolling"


# ---------------------------------------------------------------------------
# Grant Database (50+ programs)
# ---------------------------------------------------------------------------


class GrantProgram:
    """Internal definition of a grant program."""
    __slots__ = (
        "id", "name", "region", "provider", "description",
        "min_funding_usd", "max_funding_usd", "coverage_pct",
        "eligible_sizes", "eligible_industries", "eligible_projects",
        "eligible_countries", "deadline", "status",
        "application_url", "requirements", "notes",
    )

    def __init__(
        self, *, id: str, name: str, region: GrantRegion, provider: str,
        description: str, min_funding_usd: str, max_funding_usd: str,
        coverage_pct: str, eligible_sizes: List[CompanySize],
        eligible_industries: List[IndustryCode],
        eligible_projects: List[ProjectType],
        eligible_countries: List[str], deadline: str, status: GrantStatus,
        application_url: str, requirements: List[str], notes: str,
    ):
        self.id = id
        self.name = name
        self.region = region
        self.provider = provider
        self.description = description
        self.min_funding_usd = Decimal(min_funding_usd)
        self.max_funding_usd = Decimal(max_funding_usd)
        self.coverage_pct = Decimal(coverage_pct)
        self.eligible_sizes = eligible_sizes
        self.eligible_industries = eligible_industries
        self.eligible_projects = eligible_projects
        self.eligible_countries = eligible_countries
        self.deadline = deadline
        self.status = status
        self.application_url = application_url
        self.requirements = requirements
        self.notes = notes


# Source: Government program websites, Green Finance Institute, DOE, EC.
GRANTS_DB: List[GrantProgram] = [
    # ---- UK GRANTS (15) ----
    GrantProgram(
        id="GR-UK-001", name="Industrial Energy Transformation Fund (IETF)",
        region=GrantRegion.UK, provider="UK BEIS/DESNZ",
        description="Supports high energy-use businesses to cut energy bills and carbon emissions through energy efficiency and low carbon technologies.",
        min_funding_usd="125000", max_funding_usd="18750000",
        coverage_pct="50", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.MANUFACTURING, IndustryCode.CONSTRUCTION],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.PROCESS_IMPROVEMENT, ProjectType.HEAT_PUMP],
        eligible_countries=["GB"], deadline="Rolling applications", status=GrantStatus.ROLLING,
        application_url="https://www.gov.uk/government/collections/industrial-energy-transformation-fund",
        requirements=["UK registered business", "Energy-intensive operations", "Feasibility study"],
        notes="Phase 3 now open; covers feasibility studies and deployment.",
    ),
    GrantProgram(
        id="GR-UK-002", name="Boiler Upgrade Scheme (BUS)",
        region=GrantRegion.UK, provider="UK DESNZ / Ofgem",
        description="Grants for replacing fossil fuel heating with heat pumps or biomass boilers.",
        min_funding_usd="6250", max_funding_usd="9375",
        coverage_pct="40", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.HEAT_PUMP],
        eligible_countries=["GB-ENG", "GB-WLS"], deadline="2028-03-31", status=GrantStatus.OPEN,
        application_url="https://www.gov.uk/apply-boiler-upgrade-scheme",
        requirements=["Valid EPC", "Property in England or Wales", "Replacing fossil fuel system"],
        notes="Up to 7500 GBP for ASHP; property must have EPC.",
    ),
    GrantProgram(
        id="GR-UK-003", name="Salix Finance - Public Sector Decarbonisation",
        region=GrantRegion.UK, provider="Salix Finance",
        description="Interest-free loans and grants for public sector energy efficiency.",
        min_funding_usd="6250", max_funding_usd="12500000",
        coverage_pct="100", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.EDUCATION, IndustryCode.HEALTHCARE],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY, ProjectType.HEAT_PUMP, ProjectType.BUILDING_RETROFIT],
        eligible_countries=["GB"], deadline="Rolling", status=GrantStatus.ROLLING,
        application_url="https://www.salixfinance.co.uk/",
        requirements=["Public sector organization", "Energy-saving project"],
        notes="Interest-free finance; some grant elements available.",
    ),
    GrantProgram(
        id="GR-UK-004", name="Green Business Grant Scheme",
        region=GrantRegion.UK, provider="Local Enterprise Partnerships",
        description="Small grants for SME energy efficiency improvements.",
        min_funding_usd="1250", max_funding_usd="12500",
        coverage_pct="50", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["GB"], deadline="Varies by LEP", status=GrantStatus.ROLLING,
        application_url="https://www.lepnetwork.net/",
        requirements=["SME registered in LEP area", "Energy audit"],
        notes="Availability varies by region; check local LEP.",
    ),
    GrantProgram(
        id="GR-UK-005", name="Workplace Charging Scheme (WCS)",
        region=GrantRegion.UK, provider="UK OZEV",
        description="Voucher for EV chargepoint installation at workplace.",
        min_funding_usd="450", max_funding_usd="18750",
        coverage_pct="75", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.EV_FLEET],
        eligible_countries=["GB"], deadline="2025-03-31", status=GrantStatus.OPEN,
        application_url="https://www.gov.uk/government/collections/government-grants-for-low-emission-vehicles",
        requirements=["Off-street parking", "Registered business"],
        notes="Up to 350 GBP per socket, max 40 sockets.",
    ),
    GrantProgram(
        id="GR-UK-006", name="Smart Export Guarantee (SEG)",
        region=GrantRegion.UK, provider="Energy suppliers",
        description="Payment for excess solar/renewable electricity exported to grid.",
        min_funding_usd="0", max_funding_usd="5000",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["GB"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.ofgem.gov.uk/environmental-and-social-schemes/smart-export-guarantee-seg",
        requirements=["MCS-certified installation", "Smart meter", "< 5MW capacity"],
        notes="Revenue stream not grant; supports solar PV business case.",
    ),
    GrantProgram(
        id="GR-UK-007", name="UK Research & Innovation (UKRI) Net Zero Grants",
        region=GrantRegion.UK, provider="UKRI / Innovate UK",
        description="Innovation grants for net-zero technologies and processes.",
        min_funding_usd="31250", max_funding_usd="1250000",
        coverage_pct="70", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RD_INNOVATION, ProjectType.PROCESS_IMPROVEMENT],
        eligible_countries=["GB"], deadline="Competition-based", status=GrantStatus.ROLLING,
        application_url="https://www.ukri.org/",
        requirements=["Innovation element", "Collaborative project preferred"],
        notes="Check Innovate UK for current competitions.",
    ),
    GrantProgram(
        id="GR-UK-008", name="Carbon Trust Green Business Fund",
        region=GrantRegion.UK, provider="Carbon Trust",
        description="Fully funded energy assessments and capital grants for SMEs.",
        min_funding_usd="0", max_funding_usd="6250",
        coverage_pct="30", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["GB"], deadline="Rolling", status=GrantStatus.ROLLING,
        application_url="https://www.carbontrust.com/",
        requirements=["UK SME", "< 250 employees"],
        notes="Free energy survey plus capital contribution.",
    ),
    GrantProgram(
        id="GR-UK-009", name="British Business Bank - Start Up Loans (Green)",
        region=GrantRegion.UK, provider="British Business Bank",
        description="Government-backed loans for green business start-ups.",
        min_funding_usd="625", max_funding_usd="31250",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["GB"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.startuploans.co.uk/",
        requirements=["UK resident", "Business plan"],
        notes="Loan at 6% fixed; mentoring included.",
    ),
    GrantProgram(
        id="GR-UK-010", name="Energy Entrepreneurs Fund",
        region=GrantRegion.UK, provider="UK DESNZ",
        description="Grants for innovative clean energy solutions.",
        min_funding_usd="31250", max_funding_usd="625000",
        coverage_pct="60", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RD_INNOVATION, ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["GB"], deadline="Phase-based", status=GrantStatus.ROLLING,
        application_url="https://www.gov.uk/government/collections/energy-entrepreneurs-fund",
        requirements=["Innovative energy technology", "Prototype stage+"],
        notes="Supports demonstration and early commercialization.",
    ),
    GrantProgram(
        id="GR-UK-011", name="Clean Heat Grant (Scotland)",
        region=GrantRegion.UK, provider="Scottish Government",
        description="Grants for renewable heating systems in Scotland.",
        min_funding_usd="9375", max_funding_usd="9375",
        coverage_pct="40", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.HEAT_PUMP],
        eligible_countries=["GB-SCT"], deadline="2025-03-31", status=GrantStatus.OPEN,
        application_url="https://www.homeenergyscotland.org/",
        requirements=["Property in Scotland", "Replacing fossil fuel heating"],
        notes="Up to 7500 GBP per property.",
    ),
    GrantProgram(
        id="GR-UK-012", name="SME Climate Hub Commitment",
        region=GrantRegion.UK, provider="SME Climate Hub",
        description="Free tools and resources for SMEs committing to net zero (not a grant - zero cost to join).",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["GB"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://smeclimatehub.org/",
        requirements=["Commit to halve emissions by 2030, net zero by 2050"],
        notes="Free tools, training, and Race to Zero recognition.",
    ),
    GrantProgram(
        id="GR-UK-013", name="Low Carbon Workspaces",
        region=GrantRegion.UK, provider="ERDF / Local Authorities",
        description="Grants for SME energy efficiency and low-carbon measures.",
        min_funding_usd="1250", max_funding_usd="6250",
        coverage_pct="50", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["GB"], deadline="Varies", status=GrantStatus.ROLLING,
        application_url="https://www.lowcarbonworkspaces.co.uk/",
        requirements=["SME in eligible area", "Match funding"],
        notes="Regional availability; check postcodes.",
    ),
    GrantProgram(
        id="GR-UK-014", name="Enhanced Capital Allowances (ECA)",
        region=GrantRegion.UK, provider="UK HMRC",
        description="100% first-year capital allowance on energy-saving equipment.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["GB"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.gov.uk/guidance/energy-technology-list",
        requirements=["Equipment on Energy Technology List"],
        notes="Tax relief, not grant; improves payback on qualifying equipment.",
    ),
    GrantProgram(
        id="GR-UK-015", name="EV Infrastructure Grant (EVIG)",
        region=GrantRegion.UK, provider="UK OZEV",
        description="Grants for EV charging infrastructure at business premises.",
        min_funding_usd="1875", max_funding_usd="18750",
        coverage_pct="75", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.EV_FLEET],
        eligible_countries=["GB"], deadline="2025-03-31", status=GrantStatus.OPEN,
        application_url="https://www.gov.uk/government/collections/government-grants-for-low-emission-vehicles",
        requirements=["Off-street parking", "Staff or visitor use"],
        notes="Covers up to 75% of installation costs.",
    ),
    # ---- EU GRANTS (12) ----
    GrantProgram(
        id="GR-EU-001", name="EU LIFE Programme - Clean Energy Transition",
        region=GrantRegion.EU, provider="European Commission",
        description="Co-funding for clean energy transition projects.",
        min_funding_usd="54000", max_funding_usd="2160000",
        coverage_pct="60", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY, ProjectType.CIRCULAR_ECONOMY],
        eligible_countries=["EU"], deadline="Annual calls", status=GrantStatus.ROLLING,
        application_url="https://cinea.ec.europa.eu/programmes/life_en",
        requirements=["EU-based organization", "Cross-border element preferred"],
        notes="Annual work programme published by CINEA.",
    ),
    GrantProgram(
        id="GR-EU-002", name="EU Innovation Fund - Small Scale",
        region=GrantRegion.EU, provider="European Commission",
        description="Grants for innovative low-carbon technologies (small-scale: < EUR 7.5M capex).",
        min_funding_usd="540000", max_funding_usd="8100000",
        coverage_pct="60", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.MANUFACTURING, IndustryCode.CONSTRUCTION],
        eligible_projects=[ProjectType.RD_INNOVATION, ProjectType.CARBON_CAPTURE, ProjectType.HYDROGEN],
        eligible_countries=["EU", "NO", "IS"], deadline="Annual calls", status=GrantStatus.ROLLING,
        application_url="https://ec.europa.eu/clima/eu-action/funding-climate-action/innovation-fund_en",
        requirements=["Innovative technology", "GHG reduction demonstration"],
        notes="Funded from EU ETS revenues.",
    ),
    GrantProgram(
        id="GR-EU-003", name="Just Transition Mechanism",
        region=GrantRegion.EU, provider="European Commission",
        description="Support for regions most affected by the transition to climate neutrality.",
        min_funding_usd="27000", max_funding_usd="2700000",
        coverage_pct="70", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION, ProjectType.PROCESS_IMPROVEMENT],
        eligible_countries=["EU"], deadline="Regional calls", status=GrantStatus.ROLLING,
        application_url="https://ec.europa.eu/info/strategy/priorities-2019-2024/european-green-deal/finance-and-green-deal/just-transition-mechanism_en",
        requirements=["Located in eligible Just Transition region"],
        notes="Focus on coal/carbon-intensive regions.",
    ),
    GrantProgram(
        id="GR-EU-004", name="Horizon Europe - Clean Energy Cluster",
        region=GrantRegion.EU, provider="European Commission",
        description="R&D grants for clean energy innovation.",
        min_funding_usd="270000", max_funding_usd="5400000",
        coverage_pct="70", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RD_INNOVATION],
        eligible_countries=["EU", "Associated countries"], deadline="Annual calls", status=GrantStatus.ROLLING,
        application_url="https://ec.europa.eu/info/horizon-europe_en",
        requirements=["Consortium required", "TRL advancement"],
        notes="SME instrument available for single applicants.",
    ),
    GrantProgram(
        id="GR-EU-005", name="EU Cohesion Fund - Green Investment",
        region=GrantRegion.EU, provider="European Commission / Member States",
        description="Infrastructure investment for environmental sustainability.",
        min_funding_usd="54000", max_funding_usd="5400000",
        coverage_pct="80", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.BUILDING_RETROFIT, ProjectType.WASTE_REDUCTION],
        eligible_countries=["EU"], deadline="Regional calls", status=GrantStatus.ROLLING,
        application_url="https://ec.europa.eu/regional_policy/funding/cohesion-fund_en",
        requirements=["Member state with GNI/capita < 90% EU average"],
        notes="Managed through national operational programmes.",
    ),
    GrantProgram(
        id="GR-EU-006", name="EIC Accelerator (SME Instrument)",
        region=GrantRegion.EU, provider="European Innovation Council",
        description="Grants and equity for breakthrough innovations by SMEs.",
        min_funding_usd="540000", max_funding_usd="2970000",
        coverage_pct="70", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RD_INNOVATION],
        eligible_countries=["EU", "Associated"], deadline="Cut-off dates", status=GrantStatus.ROLLING,
        application_url="https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en",
        requirements=["SME with high-impact innovation", "Market-creating potential"],
        notes="Grant + optional equity investment up to EUR 15M.",
    ),
    GrantProgram(
        id="GR-EU-007", name="ERDF SME Competitiveness - Green",
        region=GrantRegion.EU, provider="Member State Managing Authorities",
        description="ERDF funding for SME green competitiveness and resource efficiency.",
        min_funding_usd="5400", max_funding_usd="270000",
        coverage_pct="50", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.CIRCULAR_ECONOMY, ProjectType.WASTE_REDUCTION],
        eligible_countries=["EU"], deadline="Regional", status=GrantStatus.ROLLING,
        application_url="https://ec.europa.eu/regional_policy/funding/erdf_en",
        requirements=["SME in eligible NUTS-2 region"],
        notes="Check national/regional operational programme for details.",
    ),
    GrantProgram(
        id="GR-EU-008", name="InvestEU - Green Guarantee",
        region=GrantRegion.EU, provider="European Investment Bank",
        description="EU guarantee to unlock green investment for SMEs via financial intermediaries.",
        min_funding_usd="27000", max_funding_usd="5400000",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION, ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["EU"], deadline="Through intermediaries", status=GrantStatus.ROLLING,
        application_url="https://investeu.europa.eu/",
        requirements=["Apply through local bank/intermediary"],
        notes="Not a direct grant; provides favorable loan terms.",
    ),
    GrantProgram(
        id="GR-EU-009", name="Modernisation Fund",
        region=GrantRegion.EU, provider="European Commission",
        description="Support for energy system modernization in lower-income EU member states.",
        min_funding_usd="108000", max_funding_usd="10800000",
        coverage_pct="60", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.MANUFACTURING, IndustryCode.CONSTRUCTION],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY, ProjectType.PROCESS_IMPROVEMENT],
        eligible_countries=["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "PL", "RO", "SK"],
        deadline="Annual", status=GrantStatus.ROLLING,
        application_url="https://ec.europa.eu/clima/eu-action/funding-climate-action/modernisation-fund_en",
        requirements=["Located in eligible member state"],
        notes="Funded from EU ETS revenues.",
    ),
    GrantProgram(
        id="GR-EU-010", name="CEF Energy - Small Projects",
        region=GrantRegion.EU, provider="CINEA",
        description="Connecting Europe Facility grants for cross-border energy projects.",
        min_funding_usd="270000", max_funding_usd="2700000",
        coverage_pct="50", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.RD_INNOVATION],
        eligible_countries=["EU"], deadline="Annual calls", status=GrantStatus.ROLLING,
        application_url="https://cinea.ec.europa.eu/programmes/connecting-europe-facility/cef-energy_en",
        requirements=["Cross-border impact", "Energy infrastructure focus"],
        notes="Focus on energy grid and interconnection.",
    ),
    GrantProgram(
        id="GR-EU-011", name="Interreg Europe - Low Carbon",
        region=GrantRegion.EU, provider="Interreg",
        description="Interregional cooperation for low-carbon economy transition.",
        min_funding_usd="27000", max_funding_usd="1350000",
        coverage_pct="70", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["EU"], deadline="Call-based", status=GrantStatus.ROLLING,
        application_url="https://www.interregeurope.eu/",
        requirements=["Partnership with regions from different EU countries"],
        notes="Policy learning and knowledge exchange focus.",
    ),
    GrantProgram(
        id="GR-EU-012", name="European Local Energy Assistance (ELENA)",
        region=GrantRegion.EU, provider="EIB",
        description="Technical assistance for local authorities and SMEs preparing energy investment.",
        min_funding_usd="270000", max_funding_usd="2700000",
        coverage_pct="90", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.BUILDING_RETROFIT, ProjectType.RENEWABLE_ENERGY, ProjectType.EV_FLEET],
        eligible_countries=["EU"], deadline="Rolling", status=GrantStatus.ROLLING,
        application_url="https://www.eib.org/en/products/advising/elena/index.htm",
        requirements=["Minimum investment programme size EUR 30M"],
        notes="Covers project development costs, not investment itself.",
    ),
    # ---- US GRANTS (13) ----
    GrantProgram(
        id="GR-US-001", name="DOE Small Business Innovation Research (SBIR)",
        region=GrantRegion.US, provider="US DOE",
        description="Federal R&D grants for small businesses in clean energy.",
        min_funding_usd="200000", max_funding_usd="1500000",
        coverage_pct="100", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RD_INNOVATION],
        eligible_countries=["US"], deadline="Phase-based", status=GrantStatus.ROLLING,
        application_url="https://science.osti.gov/sbir",
        requirements=["US-owned small business", "< 500 employees", "R&D capability"],
        notes="Phase I: $200K feasibility; Phase II: $1.5M development.",
    ),
    GrantProgram(
        id="GR-US-002", name="USDA REAP - Renewable Energy for Rural SMEs",
        region=GrantRegion.US, provider="USDA",
        description="Grants for rural SMEs to install renewable energy or energy efficiency.",
        min_funding_usd="2500", max_funding_usd="1000000",
        coverage_pct="50", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.AGRICULTURE, IndustryCode.MANUFACTURING, IndustryCode.RETAIL],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["US"], deadline="Quarterly", status=GrantStatus.ROLLING,
        application_url="https://www.rd.usda.gov/programs-services/energy-programs/rural-energy-america-program-renewable-energy-systems-energy-efficiency-improvement-guaranteed-loans",
        requirements=["Rural location", "Small business or farm"],
        notes="Grants up to 50% of project cost; loan guarantees also available.",
    ),
    GrantProgram(
        id="GR-US-003", name="IRA Section 48 Investment Tax Credit (ITC)",
        region=GrantRegion.US, provider="US IRS",
        description="Tax credit for solar, wind, and other clean energy investments.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="30", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.energy.gov/eere/solar/federal-tax-credits-solar-manufacturers",
        requirements=["US taxpayer", "Qualifying clean energy property"],
        notes="30% ITC base; up to 50% with domestic content + energy community bonuses.",
    ),
    GrantProgram(
        id="GR-US-004", name="IRA Section 45L Energy Efficient Home Credit",
        region=GrantRegion.US, provider="US IRS",
        description="Tax credit for energy-efficient commercial building improvements.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.BUILDING_RETROFIT, ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.energy.gov/eere/buildings/179d-commercial-buildings-energy-efficiency-tax-deduction",
        requirements=["Commercial building improvement", "Meet efficiency thresholds"],
        notes="Up to $5/sqft deduction for qualifying improvements.",
    ),
    GrantProgram(
        id="GR-US-005", name="EPA Environmental Justice Grants",
        region=GrantRegion.US, provider="US EPA",
        description="Grants for communities disproportionately affected by pollution.",
        min_funding_usd="50000", max_funding_usd="500000",
        coverage_pct="100", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION, ProjectType.WASTE_REDUCTION],
        eligible_countries=["US"], deadline="Annual", status=GrantStatus.ROLLING,
        application_url="https://www.epa.gov/environmentaljustice/environmental-justice-grants-funding-and-technical-assistance",
        requirements=["EJ community location", "Community engagement"],
        notes="Priority for disadvantaged communities.",
    ),
    GrantProgram(
        id="GR-US-006", name="SBA 504 Green Loan Program",
        region=GrantRegion.US, provider="US SBA",
        description="Favorable loan terms for green building and energy-efficient projects.",
        min_funding_usd="125000", max_funding_usd="5500000",
        coverage_pct="0", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.BUILDING_RETROFIT, ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.sba.gov/funding-programs/loans/504-loans",
        requirements=["US small business", "Green project component"],
        notes="Below-market rate long-term financing; higher limits for green projects.",
    ),
    GrantProgram(
        id="GR-US-007", name="DOE State Energy Program (SEP)",
        region=GrantRegion.US, provider="US DOE",
        description="Federal funding to states for energy efficiency and renewable energy programs.",
        min_funding_usd="5000", max_funding_usd="250000",
        coverage_pct="50", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY],
        eligible_countries=["US"], deadline="State-dependent", status=GrantStatus.ROLLING,
        application_url="https://www.energy.gov/scep/state-energy-program",
        requirements=["Apply through state energy office"],
        notes="Programs vary significantly by state.",
    ),
    GrantProgram(
        id="GR-US-008", name="IRA Clean Vehicle Tax Credit (30D)",
        region=GrantRegion.US, provider="US IRS",
        description="Tax credit for purchase of qualifying electric vehicles.",
        min_funding_usd="3750", max_funding_usd="7500",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.EV_FLEET],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://fueleconomy.gov/feg/tax2023.shtml",
        requirements=["Qualifying new EV", "Income limits for personal; no limit for business"],
        notes="Up to $7,500 per vehicle; commercial vehicles up to $40K credit.",
    ),
    GrantProgram(
        id="GR-US-009", name="DSIRE - Database of State Incentives",
        region=GrantRegion.US, provider="NC Clean Energy Technology Center",
        description="Comprehensive database of state-level incentives and policies.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["US"], deadline="N/A", status=GrantStatus.ROLLING,
        application_url="https://www.dsireusa.org/",
        requirements=["Search by state and technology"],
        notes="Not a grant; comprehensive search tool for state programs.",
    ),
    GrantProgram(
        id="GR-US-010", name="EPA Greenhouse Gas Reduction Fund (GHGRF)",
        region=GrantRegion.US, provider="US EPA",
        description="National clean financing network for clean energy and climate projects.",
        min_funding_usd="50000", max_funding_usd="5000000",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY, ProjectType.EV_FLEET],
        eligible_countries=["US"], deadline="Through intermediaries", status=GrantStatus.ROLLING,
        application_url="https://www.epa.gov/greenhouse-gas-reduction-fund",
        requirements=["Apply through eligible community lender"],
        notes="$27B total; deployed through green banks and CDFIs.",
    ),
    GrantProgram(
        id="GR-US-011", name="DOE Industrial Assessment Centers (IAC)",
        region=GrantRegion.US, provider="US DOE",
        description="Free energy assessments for small and medium manufacturers.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="100", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.MANUFACTURING],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.PROCESS_IMPROVEMENT],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://iac.university/",
        requirements=["Manufacturing SIC code", "Gross annual sales < $100M", "< 500 employees"],
        notes="Free assessment by university teams; typically identifies 15-30% energy savings.",
    ),
    GrantProgram(
        id="GR-US-012", name="DOE Better Buildings Challenge",
        region=GrantRegion.US, provider="US DOE",
        description="Technical assistance and recognition for building energy improvement.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.BUILDING_RETROFIT, ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://betterbuildingssolutioncenter.energy.gov/challenge",
        requirements=["Commit to 20% energy improvement over 10 years"],
        notes="Free tools and peer networking; no direct funding.",
    ),
    GrantProgram(
        id="GR-US-013", name="State PACE (Property Assessed Clean Energy)",
        region=GrantRegion.US, provider="State/Local Governments",
        description="Financing for energy improvements repaid through property tax assessment.",
        min_funding_usd="10000", max_funding_usd="5000000",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.BUILDING_RETROFIT, ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["US"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.energy.gov/scep/property-assessed-clean-energy-programs",
        requirements=["Property in PACE-enabled jurisdiction"],
        notes="Attached to property, not borrower; 15-25 year repayment.",
    ),
    # ---- AUSTRALIA (5) ----
    GrantProgram(
        id="GR-AU-001", name="Climate Active Certification",
        region=GrantRegion.AUSTRALIA, provider="Australian Government",
        description="Certification and tools for organizations achieving carbon neutrality.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["AU"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.climateactive.org.au/",
        requirements=["Measure, reduce, offset, report emissions"],
        notes="Not a grant; provides certification and marketing rights.",
    ),
    GrantProgram(
        id="GR-AU-002", name="ARENA (Australian Renewable Energy Agency)",
        region=GrantRegion.AUSTRALIA, provider="ARENA",
        description="Funding for innovative renewable energy projects.",
        min_funding_usd="65000", max_funding_usd="6500000",
        coverage_pct="50", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.RD_INNOVATION],
        eligible_countries=["AU"], deadline="Rolling", status=GrantStatus.ROLLING,
        application_url="https://arena.gov.au/",
        requirements=["Australian entity", "Innovative renewable energy element"],
        notes="Focus on pre-commercial and first-of-kind projects.",
    ),
    GrantProgram(
        id="GR-AU-003", name="CEFC (Clean Energy Finance Corporation)",
        region=GrantRegion.AUSTRALIA, provider="CEFC",
        description="Concessional finance for clean energy and energy efficiency.",
        min_funding_usd="65000", max_funding_usd="32500000",
        coverage_pct="0", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY, ProjectType.EV_FLEET],
        eligible_countries=["AU"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.cefc.com.au/",
        requirements=["Australian business", "Clean energy project"],
        notes="Below-market finance, not grants.",
    ),
    GrantProgram(
        id="GR-AU-004", name="Small Business Energy Incentive",
        region=GrantRegion.AUSTRALIA, provider="Australian Government",
        description="Bonus tax deduction for small business electrification and energy efficiency.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="20", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.ENERGY_EFFICIENCY, ProjectType.RENEWABLE_ENERGY, ProjectType.EV_FLEET],
        eligible_countries=["AU"], deadline="2025-06-30", status=GrantStatus.OPEN,
        application_url="https://www.ato.gov.au/",
        requirements=["Australian small business", "Turnover < AUD $50M"],
        notes="20% bonus deduction on qualifying assets up to AUD $100K.",
    ),
    GrantProgram(
        id="GR-AU-005", name="Sustainability Advantage Program (NSW)",
        region=GrantRegion.AUSTRALIA, provider="NSW Government",
        description="Free sustainability advisory service for businesses in NSW.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["AU-NSW"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.environment.nsw.gov.au/topics/sustainability/sustainability-advantage",
        requirements=["NSW-based business"],
        notes="Free advisory support, not direct funding.",
    ),
    # ---- GLOBAL / CANADA (5) ----
    GrantProgram(
        id="GR-CA-001", name="Canada Greener Homes Grant",
        region=GrantRegion.CANADA, provider="Government of Canada / NRCan",
        description="Grants for home/small business energy retrofits.",
        min_funding_usd="370", max_funding_usd="3700",
        coverage_pct="30", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.BUILDING_RETROFIT, ProjectType.HEAT_PUMP],
        eligible_countries=["CA"], deadline="2027-03-31", status=GrantStatus.OPEN,
        application_url="https://natural-resources.canada.ca/energy-efficiency/homes/canada-greener-homes-initiative/canada-greener-homes-grant/24831",
        requirements=["Canadian homeowner/small business", "EnerGuide audit"],
        notes="Up to CAD $5,000 per property.",
    ),
    GrantProgram(
        id="GR-CA-002", name="IRAP (Industrial Research Assistance Program)",
        region=GrantRegion.CANADA, provider="NRC-IRAP",
        description="Advisory and funding support for Canadian SME innovation.",
        min_funding_usd="7400", max_funding_usd="740000",
        coverage_pct="75", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RD_INNOVATION],
        eligible_countries=["CA"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://nrc.canada.ca/en/support-technology-innovation",
        requirements=["Canadian SME", "< 500 employees", "R&D project"],
        notes="Free advisory + project funding; one of Canada's best SME programs.",
    ),
    GrantProgram(
        id="GR-CA-003", name="SRECan (Sustainable and Renewable Energy Canada)",
        region=GrantRegion.CANADA, provider="Various provincial utilities",
        description="Provincial incentive programs for renewable energy and efficiency.",
        min_funding_usd="370", max_funding_usd="37000",
        coverage_pct="25", eligible_sizes=[CompanySize.MICRO, CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY],
        eligible_countries=["CA"], deadline="Provincial", status=GrantStatus.ROLLING,
        application_url="https://www.nrcan.gc.ca/energy/efficiency/11702",
        requirements=["Province-specific; check local utility"],
        notes="Programs vary by province (e.g., Save on Energy in ON, Efficiency NS).",
    ),
    GrantProgram(
        id="GR-GL-001", name="UNEP Finance Initiative (Global)",
        region=GrantRegion.GLOBAL, provider="UNEP",
        description="Technical assistance for aligning finance with sustainability goals.",
        min_funding_usd="0", max_funding_usd="0",
        coverage_pct="0", eligible_sizes=[CompanySize.ANY],
        eligible_industries=[IndustryCode.FINANCIAL],
        eligible_projects=[ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["GLOBAL"], deadline="Ongoing", status=GrantStatus.ROLLING,
        application_url="https://www.unepfi.org/",
        requirements=["Financial institution"],
        notes="Frameworks and tools rather than direct funding.",
    ),
    GrantProgram(
        id="GR-GL-002", name="IFC Climate-Smart SME Fund (Global)",
        region=GrantRegion.GLOBAL, provider="International Finance Corporation",
        description="Concessional finance for SMEs in developing countries pursuing climate-smart investments.",
        min_funding_usd="50000", max_funding_usd="5000000",
        coverage_pct="0", eligible_sizes=[CompanySize.SMALL, CompanySize.MEDIUM],
        eligible_industries=[IndustryCode.ANY],
        eligible_projects=[ProjectType.RENEWABLE_ENERGY, ProjectType.ENERGY_EFFICIENCY, ProjectType.GENERAL_DECARBONIZATION],
        eligible_countries=["Developing countries"], deadline="Through intermediaries", status=GrantStatus.ROLLING,
        application_url="https://www.ifc.org/",
        requirements=["SME in IFC client country", "Climate-smart project"],
        notes="Deployed through local financial intermediaries.",
    ),
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class GrantFinderInput(BaseModel):
    """Input for grant matching.

    Attributes:
        entity_name: Company name.
        industry: Industry classification.
        company_size: Company size.
        country: Country code (ISO 3166-1 alpha-2 or name).
        region_code: Sub-national region code (optional).
        project_types: Types of projects planned.
        total_emissions_tco2e: Current total emissions.
        scope1_pct: Scope 1 as % of total.
        scope2_pct: Scope 2 as % of total.
        scope3_pct: Scope 3 as % of total.
        project_budget_usd: Planned project budget.
        top_n: Number of grants to return.
        include_tax_incentives: Whether to include tax credits/deductions.
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    industry: IndustryCode = Field(default=IndustryCode.ANY)
    company_size: CompanySize = Field(default=CompanySize.SMALL)
    country: str = Field(..., min_length=2, max_length=50)
    region_code: Optional[str] = Field(None, max_length=20)
    project_types: List[ProjectType] = Field(
        default_factory=lambda: [ProjectType.ENERGY_EFFICIENCY]
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0")
    )
    scope1_pct: Decimal = Field(default=Decimal("33"), ge=Decimal("0"), le=Decimal("100"))
    scope2_pct: Decimal = Field(default=Decimal("33"), ge=Decimal("0"), le=Decimal("100"))
    scope3_pct: Decimal = Field(default=Decimal("34"), ge=Decimal("0"), le=Decimal("100"))
    project_budget_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    top_n: int = Field(default=5, ge=1, le=20)
    include_tax_incentives: bool = Field(default=True)

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        return v.upper() if len(v) <= 3 else v


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class GrantMatch(BaseModel):
    """A single matched grant program.

    Attributes:
        grant_id: Grant identifier.
        name: Program name.
        provider: Granting body.
        region: Geographic region.
        eligibility_score: Overall match score (0-100).
        score_breakdown: Score by dimension.
        funding_range_usd: Min-max funding range.
        coverage_pct: Maximum coverage percentage.
        deadline: Application deadline.
        status: Current program status.
        requirements: Application requirements.
        application_url: URL to apply.
        relevance_notes: Why this grant was matched.
    """
    grant_id: str = Field(default="")
    name: str = Field(default="")
    provider: str = Field(default="")
    region: str = Field(default="")
    eligibility_score: Decimal = Field(default=Decimal("0"))
    score_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    funding_range_usd: str = Field(default="")
    coverage_pct: Decimal = Field(default=Decimal("0"))
    deadline: str = Field(default="")
    status: str = Field(default="")
    requirements: List[str] = Field(default_factory=list)
    application_url: str = Field(default="")
    relevance_notes: List[str] = Field(default_factory=list)


class GrantFinderResult(BaseModel):
    """Complete grant finder result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Company name.
        matches: Ranked list of matched grants.
        total_grants_evaluated: Total database entries.
        total_grants_eligible: Number that passed minimum threshold.
        max_potential_funding_usd: Sum of max funding across all matches.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")

    matches: List[GrantMatch] = Field(default_factory=list)
    total_grants_evaluated: int = Field(default=0)
    total_grants_eligible: int = Field(default=0)
    max_potential_funding_usd: Decimal = Field(default=Decimal("0"))

    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GrantFinderEngine:
    """Grant and incentive matching engine for SME decarbonization.

    Scores 50+ grant programs against the SME's profile and returns
    the top matches with eligibility scores, funding ranges, and
    application details.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any scoring path.

    Usage::

        engine = GrantFinderEngine()
        result = engine.calculate(grant_input)
        for grant in result.matches:
            print(f"{grant.name}: score={grant.eligibility_score}, range={grant.funding_range_usd}")
    """

    engine_version: str = _MODULE_VERSION

    def calculate(self, data: GrantFinderInput) -> GrantFinderResult:
        """Find and score matching grants.

        Args:
            data: Validated grant finder input.

        Returns:
            GrantFinderResult with ranked matches.
        """
        t0 = time.perf_counter()
        logger.info(
            "Grant Finder: entity=%s, country=%s, industry=%s, projects=%s",
            data.entity_name, data.country, data.industry.value,
            [p.value for p in data.project_types],
        )

        scored: List[GrantMatch] = []
        country_upper = data.country.upper()

        for grant in GRANTS_DB:
            # Skip closed grants
            if grant.status == GrantStatus.CLOSED:
                continue

            # Skip tax incentives if not requested
            if not data.include_tax_incentives and grant.max_funding_usd == Decimal("0"):
                continue

            # Score each dimension
            industry_score = self._score_industry(data.industry, grant.eligible_industries)
            size_score = self._score_size(data.company_size, grant.eligible_sizes)
            location_score = self._score_location(country_upper, data.region_code, grant.eligible_countries, grant.region)
            project_score = self._score_project_types(data.project_types, grant.eligible_projects)
            emissions_score = self._score_emissions(data.total_emissions_tco2e, data.scope1_pct)

            # Weighted eligibility score
            overall = _round_val(
                industry_score * Decimal("0.30")
                + size_score * Decimal("0.20")
                + location_score * Decimal("0.20")
                + project_score * Decimal("0.20")
                + emissions_score * Decimal("0.10"),
                1,
            )

            # Minimum threshold: 30
            if overall < Decimal("30"):
                continue

            # Relevance notes
            notes: List[str] = []
            if project_score >= Decimal("80"):
                matching_projects = [
                    p.value for p in data.project_types
                    if p in grant.eligible_projects
                ]
                if matching_projects:
                    notes.append(f"Strong project match: {', '.join(matching_projects)}")
            if location_score >= Decimal("80"):
                notes.append(f"Available in your region ({data.country})")
            if grant.status == GrantStatus.CLOSING_SOON:
                notes.append("Deadline approaching - apply soon")

            funding_range = f"${float(grant.min_funding_usd):,.0f} - ${float(grant.max_funding_usd):,.0f}"
            if grant.max_funding_usd == Decimal("0"):
                funding_range = "Tax incentive / Free resource"

            scored.append(GrantMatch(
                grant_id=grant.id,
                name=grant.name,
                provider=grant.provider,
                region=grant.region.value,
                eligibility_score=overall,
                score_breakdown={
                    "industry": industry_score,
                    "size": size_score,
                    "location": location_score,
                    "project_type": project_score,
                    "emissions_profile": emissions_score,
                },
                funding_range_usd=funding_range,
                coverage_pct=grant.coverage_pct,
                deadline=grant.deadline,
                status=grant.status.value,
                requirements=grant.requirements,
                application_url=grant.application_url,
                relevance_notes=notes,
            ))

        # Sort by eligibility score descending
        scored.sort(key=lambda x: x.eligibility_score, reverse=True)

        total_eligible = len(scored)
        top_matches = scored[:data.top_n]

        max_funding = sum(
            GRANTS_DB[i].max_funding_usd
            for i, grant in enumerate(GRANTS_DB)
            if grant.id in {m.grant_id for m in top_matches}
        )
        # Recalculate from matches
        max_funding = Decimal("0")
        for m in top_matches:
            for g in GRANTS_DB:
                if g.id == m.grant_id:
                    max_funding += g.max_funding_usd
                    break

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = GrantFinderResult(
            entity_name=data.entity_name,
            matches=top_matches,
            total_grants_evaluated=len(GRANTS_DB),
            total_grants_eligible=total_eligible,
            max_potential_funding_usd=_round_val(max_funding, 2),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Grant Finder complete: %d/%d eligible, top-%d returned, hash=%s",
            total_eligible, len(GRANTS_DB), len(top_matches),
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Scoring Methods                                                      #
    # ------------------------------------------------------------------ #

    def _score_industry(
        self, sme_industry: IndustryCode,
        eligible: List[IndustryCode],
    ) -> Decimal:
        if IndustryCode.ANY in eligible:
            return Decimal("80")
        if sme_industry in eligible:
            return Decimal("100")
        if sme_industry == IndustryCode.ANY:
            return Decimal("50")
        return Decimal("0")

    def _score_size(
        self, sme_size: CompanySize,
        eligible: List[CompanySize],
    ) -> Decimal:
        if CompanySize.ANY in eligible:
            return Decimal("80")
        if sme_size in eligible:
            return Decimal("100")
        return Decimal("0")

    def _score_location(
        self,
        country: str,
        region_code: Optional[str],
        eligible_countries: List[str],
        grant_region: GrantRegion,
    ) -> Decimal:
        # Direct country match
        for ec in eligible_countries:
            ec_upper = ec.upper()
            if ec_upper == country:
                return Decimal("100")
            # Regional match (e.g., GB-ENG for GB)
            if "-" in ec_upper and ec_upper.split("-")[0] == country:
                if region_code and region_code.upper() == ec_upper:
                    return Decimal("100")
                return Decimal("70")

        # EU member check
        eu_countries = {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        }
        if "EU" in [e.upper() for e in eligible_countries] and country in eu_countries:
            return Decimal("90")

        if "GLOBAL" in [e.upper() for e in eligible_countries]:
            return Decimal("60")

        # Region-level match
        region_countries = {
            GrantRegion.UK: {"GB", "UK"},
            GrantRegion.EU: eu_countries,
            GrantRegion.US: {"US"},
            GrantRegion.AUSTRALIA: {"AU"},
            GrantRegion.CANADA: {"CA"},
        }
        region_set = region_countries.get(grant_region, set())
        if country in region_set:
            return Decimal("80")

        return Decimal("0")

    def _score_project_types(
        self,
        sme_projects: List[ProjectType],
        eligible: List[ProjectType],
    ) -> Decimal:
        if not sme_projects:
            return Decimal("50")

        matches = sum(1 for p in sme_projects if p in eligible)
        if ProjectType.GENERAL_DECARBONIZATION in eligible:
            matches = max(matches, 1)

        if matches == 0:
            return Decimal("0")

        ratio = Decimal(str(matches)) / Decimal(str(len(sme_projects)))
        return _round_val(ratio * Decimal("100"), 0)

    def _score_emissions(
        self,
        total_tco2e: Decimal,
        scope1_pct: Decimal,
    ) -> Decimal:
        # Higher emissions = more relevant for grants
        if total_tco2e == Decimal("0"):
            return Decimal("50")  # unknown
        if total_tco2e > Decimal("1000"):
            return Decimal("90")
        if total_tco2e > Decimal("100"):
            return Decimal("70")
        return Decimal("50")
