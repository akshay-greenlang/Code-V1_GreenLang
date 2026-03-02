"""
Investments Agent Models (AGENT-MRV-028)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 15
(Investments) emissions calculations using PCAF methodology.

Supports:
- 8 asset classes (listed equity, corporate bond, private equity, project finance,
  commercial real estate, mortgage, motor vehicle loan, sovereign bond)
- 5 calculation methods (reported, physical activity, revenue EEIO, sector avg, asset-specific)
- PCAF data quality scoring (1-5 scale, 1=best)
- EVIC-based attribution for equity and corporate bonds
- GDP-PPP attribution for sovereign bonds
- EUI-based calculations for real estate
- Per-vehicle emission factors for motor vehicle loans
- WACI (Weighted Average Carbon Intensity) portfolio metrics
- Portfolio alignment (1.5C, 2C) and SBTi-FI tracking
- 9 compliance frameworks (GHG Protocol, PCAF, ISO 14064, CSRD, CDP, SBTi-FI,
  SB 253, TCFD, NZBA)
- Double-counting prevention (DC-INV-001 through DC-INV-008)
- Multi-currency conversion with CPI deflation
- SHA-256 provenance chain with 10-stage pipeline
- Uncertainty quantification (Monte Carlo, analytical, PCAF data quality)

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.investments.models import EquityInvestmentInput, AssetClass
    >>> equity = EquityInvestmentInput(
    ...     company_name="Acme Corp",
    ...     isin="US0378331005",
    ...     outstanding_amount=Decimal("10000000"),
    ...     evic=Decimal("500000000"),
    ...     company_emissions_scope1=Decimal("50000"),
    ...     company_emissions_scope2=Decimal("25000"),
    ...     sector=SectorClassification.INDUSTRIALS,
    ...     country="US",
    ...     data_quality_score=PCAFDataQuality.SCORE_1,
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict
import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"

# Decimal quantization constants
_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")
_QUANT_8DP = Decimal("0.00000001")
_QUANT_10DP = Decimal("0.0000000001")

# ==============================================================================
# ENUMERATIONS (22)
# ==============================================================================


class AssetClass(str, Enum):
    """PCAF asset classes for financed emissions attribution."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BOND = "corporate_bond"
    PRIVATE_EQUITY = "private_equity"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE_LOAN = "motor_vehicle_loan"
    SOVEREIGN_BOND = "sovereign_bond"


class InvestmentType(str, Enum):
    """High-level investment type classification."""

    EQUITY = "equity"
    DEBT = "debt"
    FUND = "fund"
    DERIVATIVE = "derivative"


class SectorClassification(str, Enum):
    """GICS sector classification for investee companies."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    OTHER = "other"


class PCAFDataQuality(str, Enum):
    """PCAF data quality score (1=best, 5=worst).

    Score 1: Reported emissions -- audited, verified investee data.
    Score 2: Physical activity -- energy use or production data.
    Score 3: Revenue-based -- company revenue x sector EEIO factor.
    Score 4: Estimated -- estimated asset-level data.
    Score 5: Sector average -- asset class average with no company data.
    """

    SCORE_1 = "score_1"
    SCORE_2 = "score_2"
    SCORE_3 = "score_3"
    SCORE_4 = "score_4"
    SCORE_5 = "score_5"


class CalculationMethod(str, Enum):
    """Calculation method for financed emissions."""

    REPORTED_EMISSIONS = "reported_emissions"
    PHYSICAL_ACTIVITY = "physical_activity"
    REVENUE_EEIO = "revenue_eeio"
    SECTOR_AVERAGE = "sector_average"
    ASSET_SPECIFIC = "asset_specific"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approach for organizational boundary."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class EmissionScope(str, Enum):
    """Emission scope coverage for investee data."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"


class PropertyType(str, Enum):
    """Commercial real estate / mortgage property types."""

    OFFICE = "office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    RESIDENTIAL = "residential"
    MIXED_USE = "mixed_use"
    HOSPITALITY = "hospitality"


class VehicleCategory(str, Enum):
    """Motor vehicle loan vehicle categories."""

    PASSENGER_CAR = "passenger_car"
    LIGHT_COMMERCIAL = "light_commercial"
    HEAVY_COMMERCIAL = "heavy_commercial"
    MOTORCYCLE = "motorcycle"
    ELECTRIC_VEHICLE = "electric_vehicle"


class FuelStandard(str, Enum):
    """Vehicle fuel/emission standard classification."""

    EURO_6 = "euro_6"
    US_TIER_3 = "us_tier_3"
    CHINA_6 = "china_6"
    OTHER = "other"


class EFSource(str, Enum):
    """Emission factor data source."""

    CDP = "cdp"
    DEFRA_2024 = "defra_2024"
    EPA_2024 = "epa_2024"
    IEA_2024 = "iea_2024"
    PCAF_2024 = "pcaf_2024"
    EXIOBASE = "exiobase"
    CUSTOM = "custom"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for financial calculations."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    CNY = "CNY"
    KRW = "KRW"
    SGD = "SGD"
    HKD = "HKD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    BRL = "BRL"


class DataQualityTier(str, Enum):
    """Data quality tier for uncertainty estimation."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol / PCAF."""

    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    TECHNOLOGICAL = "technological"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"


class ComplianceFramework(str, Enum):
    """Regulatory/reporting frameworks for compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"
    PCAF = "pcaf"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI_FI = "sbti_fi"
    SB_253 = "sb_253"
    TCFD = "tcfd"
    NZBA = "nzba"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class PipelineStage(str, Enum):
    """Processing pipeline stages for provenance tracking."""

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method."""

    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    PCAF_DATA_QUALITY = "pcaf_data_quality"


class BatchStatus(str, Enum):
    """Batch calculation processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GWPSource(str, Enum):
    """IPCC Global Warming Potential assessment report version."""

    AR5 = "ar5"
    AR6 = "ar6"


class PortfolioAlignment(str, Enum):
    """Portfolio temperature alignment classification."""

    ALIGNED_1_5C = "aligned_1_5c"
    ALIGNED_2C = "aligned_2c"
    NOT_ALIGNED = "not_aligned"
    UNKNOWN = "unknown"


class SBTiTarget(str, Enum):
    """SBTi target type for financial institutions."""

    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"


# ==============================================================================
# CONSTANT TABLES (16)
# ==============================================================================

# 1. PCAF Attribution Rules -- 8 asset classes
# For each asset class: formula description, denominator description
PCAF_ATTRIBUTION_RULES: Dict[str, Dict[str, str]] = {
    AssetClass.LISTED_EQUITY.value: {
        "formula": "outstanding_amount / evic",
        "denominator": "Enterprise Value Including Cash (EVIC)",
        "description": (
            "Attribution factor = outstanding equity investment / EVIC of investee. "
            "EVIC = market cap + total debt (book value) + non-controlling interests."
        ),
    },
    AssetClass.CORPORATE_BOND.value: {
        "formula": "outstanding_amount / (total_equity + total_debt)",
        "denominator": "Total Equity + Total Debt (book value)",
        "description": (
            "Attribution factor = outstanding bond amount / (total equity + total debt). "
            "Uses book value of debt. For unlisted bonds, use total balance sheet."
        ),
    },
    AssetClass.PRIVATE_EQUITY.value: {
        "formula": "outstanding_amount / evic",
        "denominator": "Enterprise Value Including Cash (EVIC) or total equity",
        "description": (
            "Attribution factor = equity investment / EVIC. "
            "If EVIC unavailable, use total equity (book value). "
            "For VC, use post-money valuation at last funding round."
        ),
    },
    AssetClass.PROJECT_FINANCE.value: {
        "formula": "outstanding_amount / total_project_cost",
        "denominator": "Total project cost (equity + debt)",
        "description": (
            "Attribution factor = outstanding project finance exposure / "
            "total project cost. Project emissions allocated pro-rata."
        ),
    },
    AssetClass.COMMERCIAL_REAL_ESTATE.value: {
        "formula": "outstanding_amount / property_value",
        "denominator": "Property value at origination (or latest appraisal)",
        "description": (
            "Attribution factor = outstanding CRE loan or equity / property value. "
            "Emissions based on building energy use intensity (EUI) and grid EFs."
        ),
    },
    AssetClass.MORTGAGE.value: {
        "formula": "outstanding_loan / property_value",
        "denominator": "Property value at origination",
        "description": (
            "Attribution factor = outstanding mortgage balance / property value. "
            "Emissions based on property EPC rating or EUI benchmarks."
        ),
    },
    AssetClass.MOTOR_VEHICLE_LOAN.value: {
        "formula": "outstanding_loan / vehicle_value",
        "denominator": "Vehicle value at origination",
        "description": (
            "Attribution factor = outstanding auto loan / vehicle value. "
            "Emissions from annual vehicle use (distance x per-km EF)."
        ),
    },
    AssetClass.SOVEREIGN_BOND.value: {
        "formula": "outstanding_amount / gdp_ppp",
        "denominator": "GDP (PPP) of the sovereign in billion USD",
        "description": (
            "Attribution factor = outstanding sovereign bond / GDP (PPP). "
            "Attributed emissions = country total GHG * attribution factor. "
            "Excludes LULUCF by default per PCAF."
        ),
    },
}

# 2. Sector Emission Factors -- 12 GICS sectors (tCO2e per $M revenue)
SECTOR_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    SectorClassification.ENERGY.value: {
        "ef_tco2e_per_m_revenue": Decimal("450.00"),
        "description_sector": Decimal("0"),  # placeholder for type consistency
    },
    SectorClassification.MATERIALS.value: {
        "ef_tco2e_per_m_revenue": Decimal("320.00"),
    },
    SectorClassification.INDUSTRIALS.value: {
        "ef_tco2e_per_m_revenue": Decimal("180.00"),
    },
    SectorClassification.CONSUMER_DISCRETIONARY.value: {
        "ef_tco2e_per_m_revenue": Decimal("85.00"),
    },
    SectorClassification.CONSUMER_STAPLES.value: {
        "ef_tco2e_per_m_revenue": Decimal("110.00"),
    },
    SectorClassification.HEALTHCARE.value: {
        "ef_tco2e_per_m_revenue": Decimal("45.00"),
    },
    SectorClassification.FINANCIALS.value: {
        "ef_tco2e_per_m_revenue": Decimal("12.00"),
    },
    SectorClassification.INFORMATION_TECHNOLOGY.value: {
        "ef_tco2e_per_m_revenue": Decimal("32.00"),
    },
    SectorClassification.COMMUNICATION_SERVICES.value: {
        "ef_tco2e_per_m_revenue": Decimal("28.00"),
    },
    SectorClassification.UTILITIES.value: {
        "ef_tco2e_per_m_revenue": Decimal("680.00"),
    },
    SectorClassification.REAL_ESTATE.value: {
        "ef_tco2e_per_m_revenue": Decimal("120.00"),
    },
    SectorClassification.OTHER.value: {
        "ef_tco2e_per_m_revenue": Decimal("95.00"),
    },
}

# 3. Country Emission Factors -- 50+ countries
# total_ghg_mt: total GHG emissions in Mt CO2e (excl. LULUCF)
# gdp_ppp_billion_usd: GDP PPP in billion USD
# per_capita_tco2e: per-capita emissions in tCO2e
COUNTRY_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "US": {"total_ghg_mt": Decimal("5222"), "gdp_ppp_billion_usd": Decimal("25460"), "per_capita_tco2e": Decimal("15.52")},
    "CN": {"total_ghg_mt": Decimal("12668"), "gdp_ppp_billion_usd": Decimal("30180"), "per_capita_tco2e": Decimal("8.99")},
    "IN": {"total_ghg_mt": Decimal("3400"), "gdp_ppp_billion_usd": Decimal("13030"), "per_capita_tco2e": Decimal("2.39")},
    "RU": {"total_ghg_mt": Decimal("2180"), "gdp_ppp_billion_usd": Decimal("4770"), "per_capita_tco2e": Decimal("15.12")},
    "JP": {"total_ghg_mt": Decimal("1080"), "gdp_ppp_billion_usd": Decimal("6080"), "per_capita_tco2e": Decimal("8.57")},
    "DE": {"total_ghg_mt": Decimal("674"), "gdp_ppp_billion_usd": Decimal("5310"), "per_capita_tco2e": Decimal("8.09")},
    "KR": {"total_ghg_mt": Decimal("616"), "gdp_ppp_billion_usd": Decimal("2730"), "per_capita_tco2e": Decimal("11.89")},
    "CA": {"total_ghg_mt": Decimal("672"), "gdp_ppp_billion_usd": Decimal("2240"), "per_capita_tco2e": Decimal("17.27")},
    "GB": {"total_ghg_mt": Decimal("384"), "gdp_ppp_billion_usd": Decimal("3750"), "per_capita_tco2e": Decimal("5.55")},
    "FR": {"total_ghg_mt": Decimal("306"), "gdp_ppp_billion_usd": Decimal("3680"), "per_capita_tco2e": Decimal("4.52")},
    "AU": {"total_ghg_mt": Decimal("488"), "gdp_ppp_billion_usd": Decimal("1690"), "per_capita_tco2e": Decimal("18.32")},
    "BR": {"total_ghg_mt": Decimal("1280"), "gdp_ppp_billion_usd": Decimal("3840"), "per_capita_tco2e": Decimal("5.99")},
    "MX": {"total_ghg_mt": Decimal("490"), "gdp_ppp_billion_usd": Decimal("2890"), "per_capita_tco2e": Decimal("3.80")},
    "ID": {"total_ghg_mt": Decimal("1030"), "gdp_ppp_billion_usd": Decimal("4020"), "per_capita_tco2e": Decimal("3.72")},
    "SA": {"total_ghg_mt": Decimal("588"), "gdp_ppp_billion_usd": Decimal("2110"), "per_capita_tco2e": Decimal("16.16")},
    "ZA": {"total_ghg_mt": Decimal("468"), "gdp_ppp_billion_usd": Decimal("960"), "per_capita_tco2e": Decimal("7.65")},
    "TR": {"total_ghg_mt": Decimal("506"), "gdp_ppp_billion_usd": Decimal("3320"), "per_capita_tco2e": Decimal("5.94")},
    "IT": {"total_ghg_mt": Decimal("326"), "gdp_ppp_billion_usd": Decimal("3040"), "per_capita_tco2e": Decimal("5.55")},
    "ES": {"total_ghg_mt": Decimal("262"), "gdp_ppp_billion_usd": Decimal("2210"), "per_capita_tco2e": Decimal("5.52")},
    "PL": {"total_ghg_mt": Decimal("332"), "gdp_ppp_billion_usd": Decimal("1600"), "per_capita_tco2e": Decimal("8.76")},
    "NL": {"total_ghg_mt": Decimal("150"), "gdp_ppp_billion_usd": Decimal("1160"), "per_capita_tco2e": Decimal("8.49")},
    "SE": {"total_ghg_mt": Decimal("44"), "gdp_ppp_billion_usd": Decimal("660"), "per_capita_tco2e": Decimal("4.21")},
    "NO": {"total_ghg_mt": Decimal("49"), "gdp_ppp_billion_usd": Decimal("440"), "per_capita_tco2e": Decimal("8.93")},
    "DK": {"total_ghg_mt": Decimal("32"), "gdp_ppp_billion_usd": Decimal("420"), "per_capita_tco2e": Decimal("5.41")},
    "CH": {"total_ghg_mt": Decimal("40"), "gdp_ppp_billion_usd": Decimal("710"), "per_capita_tco2e": Decimal("4.55")},
    "AT": {"total_ghg_mt": Decimal("72"), "gdp_ppp_billion_usd": Decimal("590"), "per_capita_tco2e": Decimal("7.99")},
    "BE": {"total_ghg_mt": Decimal("105"), "gdp_ppp_billion_usd": Decimal("680"), "per_capita_tco2e": Decimal("9.06")},
    "FI": {"total_ghg_mt": Decimal("42"), "gdp_ppp_billion_usd": Decimal("310"), "per_capita_tco2e": Decimal("7.57")},
    "IE": {"total_ghg_mt": Decimal("62"), "gdp_ppp_billion_usd": Decimal("620"), "per_capita_tco2e": Decimal("12.26")},
    "PT": {"total_ghg_mt": Decimal("52"), "gdp_ppp_billion_usd": Decimal("430"), "per_capita_tco2e": Decimal("5.05")},
    "GR": {"total_ghg_mt": Decimal("62"), "gdp_ppp_billion_usd": Decimal("380"), "per_capita_tco2e": Decimal("5.95")},
    "CZ": {"total_ghg_mt": Decimal("112"), "gdp_ppp_billion_usd": Decimal("500"), "per_capita_tco2e": Decimal("10.47")},
    "HU": {"total_ghg_mt": Decimal("54"), "gdp_ppp_billion_usd": Decimal("370"), "per_capita_tco2e": Decimal("5.56")},
    "RO": {"total_ghg_mt": Decimal("82"), "gdp_ppp_billion_usd": Decimal("710"), "per_capita_tco2e": Decimal("4.27")},
    "TH": {"total_ghg_mt": Decimal("324"), "gdp_ppp_billion_usd": Decimal("1470"), "per_capita_tco2e": Decimal("4.62")},
    "MY": {"total_ghg_mt": Decimal("262"), "gdp_ppp_billion_usd": Decimal("1090"), "per_capita_tco2e": Decimal("7.85")},
    "SG": {"total_ghg_mt": Decimal("48"), "gdp_ppp_billion_usd": Decimal("680"), "per_capita_tco2e": Decimal("8.56")},
    "HK": {"total_ghg_mt": Decimal("36"), "gdp_ppp_billion_usd": Decimal("520"), "per_capita_tco2e": Decimal("4.79")},
    "PH": {"total_ghg_mt": Decimal("186"), "gdp_ppp_billion_usd": Decimal("1180"), "per_capita_tco2e": Decimal("1.67")},
    "VN": {"total_ghg_mt": Decimal("382"), "gdp_ppp_billion_usd": Decimal("1280"), "per_capita_tco2e": Decimal("3.88")},
    "PK": {"total_ghg_mt": Decimal("416"), "gdp_ppp_billion_usd": Decimal("1520"), "per_capita_tco2e": Decimal("1.81")},
    "BD": {"total_ghg_mt": Decimal("234"), "gdp_ppp_billion_usd": Decimal("1280"), "per_capita_tco2e": Decimal("1.39")},
    "EG": {"total_ghg_mt": Decimal("340"), "gdp_ppp_billion_usd": Decimal("1560"), "per_capita_tco2e": Decimal("3.20")},
    "NG": {"total_ghg_mt": Decimal("282"), "gdp_ppp_billion_usd": Decimal("1270"), "per_capita_tco2e": Decimal("1.31")},
    "AR": {"total_ghg_mt": Decimal("366"), "gdp_ppp_billion_usd": Decimal("1170"), "per_capita_tco2e": Decimal("7.97")},
    "CL": {"total_ghg_mt": Decimal("102"), "gdp_ppp_billion_usd": Decimal("560"), "per_capita_tco2e": Decimal("5.22")},
    "CO": {"total_ghg_mt": Decimal("188"), "gdp_ppp_billion_usd": Decimal("940"), "per_capita_tco2e": Decimal("3.63")},
    "PE": {"total_ghg_mt": Decimal("124"), "gdp_ppp_billion_usd": Decimal("510"), "per_capita_tco2e": Decimal("3.70")},
    "NZ": {"total_ghg_mt": Decimal("38"), "gdp_ppp_billion_usd": Decimal("260"), "per_capita_tco2e": Decimal("7.36")},
    "IL": {"total_ghg_mt": Decimal("74"), "gdp_ppp_billion_usd": Decimal("530"), "per_capita_tco2e": Decimal("7.88")},
    "AE": {"total_ghg_mt": Decimal("204"), "gdp_ppp_billion_usd": Decimal("760"), "per_capita_tco2e": Decimal("20.67")},
    "QA": {"total_ghg_mt": Decimal("108"), "gdp_ppp_billion_usd": Decimal("270"), "per_capita_tco2e": Decimal("37.29")},
    "KW": {"total_ghg_mt": Decimal("94"), "gdp_ppp_billion_usd": Decimal("230"), "per_capita_tco2e": Decimal("21.64")},
}

# 4. Grid Emission Factors -- 12 countries + 26 eGRID subregions (kgCO2e/kWh)
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # Countries
    "US": Decimal("0.3937"),
    "CN": Decimal("0.5810"),
    "IN": Decimal("0.7080"),
    "JP": Decimal("0.4570"),
    "DE": Decimal("0.3380"),
    "GB": Decimal("0.2070"),
    "FR": Decimal("0.0520"),
    "KR": Decimal("0.4590"),
    "CA": Decimal("0.1200"),
    "AU": Decimal("0.6560"),
    "BR": Decimal("0.0740"),
    "ZA": Decimal("0.9280"),
    # eGRID subregions (US)
    "AKGD": Decimal("0.4349"),
    "AKMS": Decimal("0.2308"),
    "AZNM": Decimal("0.4051"),
    "CAMX": Decimal("0.2256"),
    "ERCT": Decimal("0.3924"),
    "FRCC": Decimal("0.3954"),
    "HIMS": Decimal("0.5424"),
    "HIOA": Decimal("0.6593"),
    "MROE": Decimal("0.5482"),
    "MROW": Decimal("0.4521"),
    "NEWE": Decimal("0.2135"),
    "NWPP": Decimal("0.2796"),
    "NYCW": Decimal("0.2451"),
    "NYLI": Decimal("0.3687"),
    "NYUP": Decimal("0.1217"),
    "PRMS": Decimal("0.0000"),
    "RFCE": Decimal("0.3231"),
    "RFCM": Decimal("0.5248"),
    "RFCW": Decimal("0.4683"),
    "RMPA": Decimal("0.5589"),
    "SPNO": Decimal("0.4725"),
    "SPSO": Decimal("0.4329"),
    "SRMV": Decimal("0.3565"),
    "SRMW": Decimal("0.6548"),
    "SRSO": Decimal("0.4032"),
    "SRTV": Decimal("0.4279"),
}

# 5. Building EUI Benchmarks -- 6 property types x 5 climate zones (kWh/m2/yr)
# Climate zones: cold, cool, moderate, warm, hot
BUILDING_EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    PropertyType.OFFICE.value: {
        "cold": Decimal("280"), "cool": Decimal("240"),
        "moderate": Decimal("200"), "warm": Decimal("220"),
        "hot": Decimal("260"),
    },
    PropertyType.RETAIL.value: {
        "cold": Decimal("310"), "cool": Decimal("270"),
        "moderate": Decimal("230"), "warm": Decimal("250"),
        "hot": Decimal("290"),
    },
    PropertyType.INDUSTRIAL.value: {
        "cold": Decimal("220"), "cool": Decimal("190"),
        "moderate": Decimal("160"), "warm": Decimal("170"),
        "hot": Decimal("200"),
    },
    PropertyType.RESIDENTIAL.value: {
        "cold": Decimal("200"), "cool": Decimal("170"),
        "moderate": Decimal("140"), "warm": Decimal("150"),
        "hot": Decimal("180"),
    },
    PropertyType.MIXED_USE.value: {
        "cold": Decimal("260"), "cool": Decimal("225"),
        "moderate": Decimal("190"), "warm": Decimal("205"),
        "hot": Decimal("245"),
    },
    PropertyType.HOSPITALITY.value: {
        "cold": Decimal("340"), "cool": Decimal("300"),
        "moderate": Decimal("260"), "warm": Decimal("280"),
        "hot": Decimal("320"),
    },
}

# 6. Vehicle Emission Factors -- 5 categories: annual kgCO2e, avg distance km
VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    VehicleCategory.PASSENGER_CAR.value: {
        "annual_emissions_kgco2e": Decimal("2400"),
        "avg_annual_distance_km": Decimal("15000"),
        "ef_kgco2e_per_km": Decimal("0.1600"),
    },
    VehicleCategory.LIGHT_COMMERCIAL.value: {
        "annual_emissions_kgco2e": Decimal("3600"),
        "avg_annual_distance_km": Decimal("20000"),
        "ef_kgco2e_per_km": Decimal("0.1800"),
    },
    VehicleCategory.HEAVY_COMMERCIAL.value: {
        "annual_emissions_kgco2e": Decimal("18000"),
        "avg_annual_distance_km": Decimal("60000"),
        "ef_kgco2e_per_km": Decimal("0.3000"),
    },
    VehicleCategory.MOTORCYCLE.value: {
        "annual_emissions_kgco2e": Decimal("900"),
        "avg_annual_distance_km": Decimal("8000"),
        "ef_kgco2e_per_km": Decimal("0.1125"),
    },
    VehicleCategory.ELECTRIC_VEHICLE.value: {
        "annual_emissions_kgco2e": Decimal("600"),
        "avg_annual_distance_km": Decimal("15000"),
        "ef_kgco2e_per_km": Decimal("0.0400"),
    },
}

# 7. EEIO Sector Factors -- 12 sectors (kgCO2e per $ revenue) from EXIOBASE
EEIO_SECTOR_FACTORS: Dict[str, Dict[str, Decimal]] = {
    SectorClassification.ENERGY.value: {
        "ef_kgco2e_per_usd": Decimal("0.4500"),
        "uncertainty_pct": Decimal("40"),
    },
    SectorClassification.MATERIALS.value: {
        "ef_kgco2e_per_usd": Decimal("0.3200"),
        "uncertainty_pct": Decimal("40"),
    },
    SectorClassification.INDUSTRIALS.value: {
        "ef_kgco2e_per_usd": Decimal("0.1800"),
        "uncertainty_pct": Decimal("40"),
    },
    SectorClassification.CONSUMER_DISCRETIONARY.value: {
        "ef_kgco2e_per_usd": Decimal("0.0850"),
        "uncertainty_pct": Decimal("45"),
    },
    SectorClassification.CONSUMER_STAPLES.value: {
        "ef_kgco2e_per_usd": Decimal("0.1100"),
        "uncertainty_pct": Decimal("40"),
    },
    SectorClassification.HEALTHCARE.value: {
        "ef_kgco2e_per_usd": Decimal("0.0450"),
        "uncertainty_pct": Decimal("45"),
    },
    SectorClassification.FINANCIALS.value: {
        "ef_kgco2e_per_usd": Decimal("0.0120"),
        "uncertainty_pct": Decimal("50"),
    },
    SectorClassification.INFORMATION_TECHNOLOGY.value: {
        "ef_kgco2e_per_usd": Decimal("0.0320"),
        "uncertainty_pct": Decimal("45"),
    },
    SectorClassification.COMMUNICATION_SERVICES.value: {
        "ef_kgco2e_per_usd": Decimal("0.0280"),
        "uncertainty_pct": Decimal("45"),
    },
    SectorClassification.UTILITIES.value: {
        "ef_kgco2e_per_usd": Decimal("0.6800"),
        "uncertainty_pct": Decimal("35"),
    },
    SectorClassification.REAL_ESTATE.value: {
        "ef_kgco2e_per_usd": Decimal("0.1200"),
        "uncertainty_pct": Decimal("40"),
    },
    SectorClassification.OTHER.value: {
        "ef_kgco2e_per_usd": Decimal("0.0950"),
        "uncertainty_pct": Decimal("50"),
    },
}

# 8. PCAF Data Quality Matrix -- 8 asset classes x 5 quality scores
# criteria: description of data requirement; uncertainty_pct: associated uncertainty
PCAF_DATA_QUALITY_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
    AssetClass.LISTED_EQUITY.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Verified Scope 1+2 reported by investee", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Non-verified Scope 1+2 reported by investee", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "Emissions from physical activity data (energy use)", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Emissions from revenue-based EEIO model", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "Emissions from sector average per asset class", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.CORPORATE_BOND.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Verified Scope 1+2 reported by investee", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Non-verified Scope 1+2 reported by investee", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "Emissions from physical activity data (energy use)", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Emissions from revenue-based EEIO model", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "Emissions from sector average per asset class", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.PRIVATE_EQUITY.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Verified Scope 1+2 reported by portfolio company", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Non-verified Scope 1+2 reported by portfolio company", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "Emissions from physical activity data", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Emissions estimated from revenue-based model", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "Sector average per asset class", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.PROJECT_FINANCE.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Verified project emissions (monitored)", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Non-verified project emissions reported", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "Emissions from physical activity data (energy, output)", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Estimated from project type EFs", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "Sector average per project type", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.COMMERCIAL_REAL_ESTATE.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Actual building energy data (metered) + supplier EFs", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Actual building energy data + grid-average EFs", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "EPC label-based estimated consumption", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Estimated from floor area + EUI benchmarks", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "Sector average per building type", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.MORTGAGE.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Actual building energy data (metered) + supplier EFs", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Actual building energy data + grid-average EFs", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "EPC label-based estimated consumption", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Estimated from floor area + EUI benchmarks", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "National average per property type", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.MOTOR_VEHICLE_LOAN.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Actual fuel use or telematic distance + per-km EF", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Make/model-specific per-km EF + estimated distance", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "Vehicle category-specific per-km EF + avg distance", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Estimated from vehicle type average", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "National vehicle fleet average per loan", "uncertainty_pct": Decimal("60")},
    },
    AssetClass.SOVEREIGN_BOND.value: {
        PCAFDataQuality.SCORE_1.value: {"criteria": "Country official GHG inventory (verified, <2yr old)", "uncertainty_pct": Decimal("10")},
        PCAFDataQuality.SCORE_2.value: {"criteria": "Country official GHG inventory (>2yr old)", "uncertainty_pct": Decimal("20")},
        PCAFDataQuality.SCORE_3.value: {"criteria": "Third-party estimate (IEA, WRI CAIT)", "uncertainty_pct": Decimal("35")},
        PCAFDataQuality.SCORE_4.value: {"criteria": "Estimated from GDP + regional EF", "uncertainty_pct": Decimal("50")},
        PCAFDataQuality.SCORE_5.value: {"criteria": "Global average per capita * population", "uncertainty_pct": Decimal("60")},
    },
}

# 9. Currency Conversion Rates -- 15 currencies to USD (base year 2024)
CURRENCY_CONVERSION_RATES: Dict[str, Decimal] = {
    CurrencyCode.USD.value: Decimal("1.0000"),
    CurrencyCode.EUR.value: Decimal("1.0850"),
    CurrencyCode.GBP.value: Decimal("1.2650"),
    CurrencyCode.JPY.value: Decimal("0.00667"),
    CurrencyCode.CHF.value: Decimal("1.1380"),
    CurrencyCode.CAD.value: Decimal("0.7420"),
    CurrencyCode.AUD.value: Decimal("0.6530"),
    CurrencyCode.CNY.value: Decimal("0.1380"),
    CurrencyCode.KRW.value: Decimal("0.000750"),
    CurrencyCode.SGD.value: Decimal("0.7440"),
    CurrencyCode.HKD.value: Decimal("0.1282"),
    CurrencyCode.SEK.value: Decimal("0.0942"),
    CurrencyCode.NOK.value: Decimal("0.0928"),
    CurrencyCode.DKK.value: Decimal("0.1456"),
    CurrencyCode.BRL.value: Decimal("0.2010"),
}

# CPI Deflators relative to base year 2021 = 1.0000
CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.8710"),
    2016: Decimal("0.8820"),
    2017: Decimal("0.9010"),
    2018: Decimal("0.9230"),
    2019: Decimal("0.9410"),
    2020: Decimal("0.9530"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1130"),
    2024: Decimal("1.1490"),
    2025: Decimal("1.1780"),
    2026: Decimal("1.2080"),
}

# 10. Sovereign Country Data -- 50 countries with LULUCF adjustment
# lulucf_mt: LULUCF net emissions/removals in Mt CO2e
# total_excl_lulucf = total_ghg_mt (from table 3, already excl. LULUCF)
SOVEREIGN_COUNTRY_DATA: Dict[str, Dict[str, Decimal]] = {
    "US": {"lulucf_mt": Decimal("-812"), "population_millions": Decimal("336.5")},
    "CN": {"lulucf_mt": Decimal("-580"), "population_millions": Decimal("1409.7")},
    "IN": {"lulucf_mt": Decimal("-316"), "population_millions": Decimal("1422.0")},
    "RU": {"lulucf_mt": Decimal("-540"), "population_millions": Decimal("144.1")},
    "JP": {"lulucf_mt": Decimal("-48"), "population_millions": Decimal("126.0")},
    "DE": {"lulucf_mt": Decimal("-24"), "population_millions": Decimal("83.3")},
    "KR": {"lulucf_mt": Decimal("-42"), "population_millions": Decimal("51.8")},
    "CA": {"lulucf_mt": Decimal("26"), "population_millions": Decimal("38.9")},
    "GB": {"lulucf_mt": Decimal("-12"), "population_millions": Decimal("69.2")},
    "FR": {"lulucf_mt": Decimal("-28"), "population_millions": Decimal("67.7")},
    "AU": {"lulucf_mt": Decimal("-22"), "population_millions": Decimal("26.6")},
    "BR": {"lulucf_mt": Decimal("870"), "population_millions": Decimal("213.8")},
    "MX": {"lulucf_mt": Decimal("-148"), "population_millions": Decimal("128.9")},
    "ID": {"lulucf_mt": Decimal("520"), "population_millions": Decimal("276.9")},
    "SA": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("36.4")},
    "ZA": {"lulucf_mt": Decimal("-8"), "population_millions": Decimal("61.2")},
    "TR": {"lulucf_mt": Decimal("-64"), "population_millions": Decimal("85.2")},
    "IT": {"lulucf_mt": Decimal("-32"), "population_millions": Decimal("58.8")},
    "ES": {"lulucf_mt": Decimal("-38"), "population_millions": Decimal("47.4")},
    "PL": {"lulucf_mt": Decimal("-36"), "population_millions": Decimal("37.9")},
    "NL": {"lulucf_mt": Decimal("4"), "population_millions": Decimal("17.7")},
    "SE": {"lulucf_mt": Decimal("-44"), "population_millions": Decimal("10.5")},
    "NO": {"lulucf_mt": Decimal("-24"), "population_millions": Decimal("5.5")},
    "DK": {"lulucf_mt": Decimal("5"), "population_millions": Decimal("5.9")},
    "CH": {"lulucf_mt": Decimal("-2"), "population_millions": Decimal("8.8")},
    "AT": {"lulucf_mt": Decimal("-5"), "population_millions": Decimal("9.0")},
    "BE": {"lulucf_mt": Decimal("-1"), "population_millions": Decimal("11.6")},
    "FI": {"lulucf_mt": Decimal("-14"), "population_millions": Decimal("5.5")},
    "IE": {"lulucf_mt": Decimal("5"), "population_millions": Decimal("5.1")},
    "PT": {"lulucf_mt": Decimal("-10"), "population_millions": Decimal("10.3")},
    "GR": {"lulucf_mt": Decimal("-4"), "population_millions": Decimal("10.4")},
    "CZ": {"lulucf_mt": Decimal("-4"), "population_millions": Decimal("10.7")},
    "HU": {"lulucf_mt": Decimal("-5"), "population_millions": Decimal("9.7")},
    "RO": {"lulucf_mt": Decimal("-24"), "population_millions": Decimal("19.2")},
    "TH": {"lulucf_mt": Decimal("-72"), "population_millions": Decimal("70.1")},
    "MY": {"lulucf_mt": Decimal("-54"), "population_millions": Decimal("33.4")},
    "SG": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("5.6")},
    "HK": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("7.5")},
    "PH": {"lulucf_mt": Decimal("-96"), "population_millions": Decimal("111.5")},
    "VN": {"lulucf_mt": Decimal("-36"), "population_millions": Decimal("98.5")},
    "PK": {"lulucf_mt": Decimal("-14"), "population_millions": Decimal("229.5")},
    "BD": {"lulucf_mt": Decimal("-18"), "population_millions": Decimal("168.4")},
    "EG": {"lulucf_mt": Decimal("-2"), "population_millions": Decimal("106.2")},
    "NG": {"lulucf_mt": Decimal("42"), "population_millions": Decimal("215.0")},
    "AR": {"lulucf_mt": Decimal("54"), "population_millions": Decimal("45.9")},
    "CL": {"lulucf_mt": Decimal("-62"), "population_millions": Decimal("19.5")},
    "CO": {"lulucf_mt": Decimal("38"), "population_millions": Decimal("51.8")},
    "PE": {"lulucf_mt": Decimal("-42"), "population_millions": Decimal("33.5")},
    "NZ": {"lulucf_mt": Decimal("-30"), "population_millions": Decimal("5.2")},
    "IL": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("9.4")},
    "AE": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("9.9")},
    "QA": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("2.9")},
    "KW": {"lulucf_mt": Decimal("0"), "population_millions": Decimal("4.3")},
}

# 11. Carbon Intensity Benchmarks -- by sector: tCO2e/$M revenue
# aligned_1_5c_target: target intensity for 1.5C alignment by 2030
CARBON_INTENSITY_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    SectorClassification.ENERGY.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("450.00"),
        "aligned_1_5c_target": Decimal("180.00"),
        "aligned_2c_target": Decimal("270.00"),
    },
    SectorClassification.MATERIALS.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("320.00"),
        "aligned_1_5c_target": Decimal("128.00"),
        "aligned_2c_target": Decimal("192.00"),
    },
    SectorClassification.INDUSTRIALS.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("180.00"),
        "aligned_1_5c_target": Decimal("72.00"),
        "aligned_2c_target": Decimal("108.00"),
    },
    SectorClassification.CONSUMER_DISCRETIONARY.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("85.00"),
        "aligned_1_5c_target": Decimal("34.00"),
        "aligned_2c_target": Decimal("51.00"),
    },
    SectorClassification.CONSUMER_STAPLES.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("110.00"),
        "aligned_1_5c_target": Decimal("44.00"),
        "aligned_2c_target": Decimal("66.00"),
    },
    SectorClassification.HEALTHCARE.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("45.00"),
        "aligned_1_5c_target": Decimal("18.00"),
        "aligned_2c_target": Decimal("27.00"),
    },
    SectorClassification.FINANCIALS.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("12.00"),
        "aligned_1_5c_target": Decimal("4.80"),
        "aligned_2c_target": Decimal("7.20"),
    },
    SectorClassification.INFORMATION_TECHNOLOGY.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("32.00"),
        "aligned_1_5c_target": Decimal("12.80"),
        "aligned_2c_target": Decimal("19.20"),
    },
    SectorClassification.COMMUNICATION_SERVICES.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("28.00"),
        "aligned_1_5c_target": Decimal("11.20"),
        "aligned_2c_target": Decimal("16.80"),
    },
    SectorClassification.UTILITIES.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("680.00"),
        "aligned_1_5c_target": Decimal("272.00"),
        "aligned_2c_target": Decimal("408.00"),
    },
    SectorClassification.REAL_ESTATE.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("120.00"),
        "aligned_1_5c_target": Decimal("48.00"),
        "aligned_2c_target": Decimal("72.00"),
    },
    SectorClassification.OTHER.value: {
        "benchmark_tco2e_per_m_revenue": Decimal("95.00"),
        "aligned_1_5c_target": Decimal("38.00"),
        "aligned_2c_target": Decimal("57.00"),
    },
}

# 12. Double-Counting Rules (DC-INV-001 through DC-INV-008)
DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-INV-001": {
        "rule": "Investments consolidated in Scope 1/2 must NOT be counted in Cat 15",
        "description": (
            "If the investee is consolidated in the reporting company's organizational "
            "boundary (operational control, financial control, or equity share > 50%), "
            "its emissions are already in Scope 1 and Scope 2. Do not double-count."
        ),
        "action": "EXCLUDE",
    },
    "DC-INV-002": {
        "rule": "Equity method investments above equity share threshold",
        "description": (
            "If equity share >= 20% and < 50%, the investee may be accounted under "
            "equity method. Ensure emissions are reported under either Scope 1/2 "
            "(equity share approach) OR Cat 15, not both."
        ),
        "action": "CHECK",
    },
    "DC-INV-003": {
        "rule": "Sovereign bond emissions vs corporate bond/equity investee country",
        "description": (
            "Sovereign bond attributed emissions (country total / GDP PPP) may overlap "
            "with corporate bond/equity investee emissions in same country. PCAF "
            "acknowledges this overlap but does not require adjustment."
        ),
        "action": "DISCLOSE",
    },
    "DC-INV-004": {
        "rule": "Fund-of-funds double counting through layered attribution",
        "description": (
            "When investing in funds that in turn invest in other funds, ensure "
            "look-through attribution does not double-count the same underlying "
            "investee across multiple fund layers."
        ),
        "action": "DEDUP",
    },
    "DC-INV-005": {
        "rule": "Multiple financing instruments for same investee",
        "description": (
            "If the FI holds both equity and debt in the same investee, each "
            "instrument has its own attribution factor. The sum of attribution "
            "factors for equity + debt should not exceed 1.0 per investee."
        ),
        "action": "CHECK",
    },
    "DC-INV-006": {
        "rule": "Project finance vs corporate bond for same entity",
        "description": (
            "If a project is on-balance-sheet of a corporate where the FI also "
            "holds corporate bonds, avoid counting project emissions under both "
            "project finance and corporate bond attribution."
        ),
        "action": "DEDUP",
    },
    "DC-INV-007": {
        "rule": "Real estate equity vs mortgage for same property",
        "description": (
            "If the FI holds both a CRE equity position and a mortgage on the same "
            "property, total attribution should not exceed 1.0 for that property."
        ),
        "action": "CHECK",
    },
    "DC-INV-008": {
        "rule": "Scope 3 Cat 15 vs Cat 1 (purchased financial services)",
        "description": (
            "Emissions from purchased financial services (Cat 1) should not overlap "
            "with financed emissions (Cat 15). Cat 1 covers operational emissions "
            "of financial service providers; Cat 15 covers financed emissions."
        ),
        "action": "DISCLOSE",
    },
}

# 13. Compliance Framework Rules -- 9 frameworks with requirements
COMPLIANCE_FRAMEWORK_RULES: Dict[str, Dict[str, Any]] = {
    ComplianceFramework.GHG_PROTOCOL.value: {
        "name": "GHG Protocol Scope 3 Standard",
        "required_fields": [
            "total_financed_emissions", "asset_class_breakdown",
            "calculation_method", "emission_factors_source",
            "data_quality_assessment",
        ],
        "mandatory": True,
        "version": "2011 (with 2013 amendments)",
    },
    ComplianceFramework.PCAF.value: {
        "name": "PCAF Global GHG Accounting Standard",
        "required_fields": [
            "total_financed_emissions", "by_asset_class", "attribution_factors",
            "pcaf_data_quality_score", "coverage_ratio", "emission_scopes",
            "methodology_description",
        ],
        "mandatory": True,
        "version": "3rd Edition (2024)",
    },
    ComplianceFramework.ISO_14064.value: {
        "name": "ISO 14064-1:2018",
        "required_fields": [
            "total_co2e", "gases_included", "gwp_source",
            "base_year", "consolidation_approach", "uncertainty",
        ],
        "mandatory": False,
        "version": "2018",
    },
    ComplianceFramework.CSRD_ESRS.value: {
        "name": "CSRD ESRS E1 Climate Change",
        "required_fields": [
            "scope_3_category_15", "asset_class_breakdown",
            "pcaf_methodology", "data_quality", "financed_emissions_intensity",
            "transition_risks",
        ],
        "mandatory": True,
        "version": "ESRS E1 (2024)",
    },
    ComplianceFramework.CDP.value: {
        "name": "CDP Climate Change Questionnaire",
        "required_fields": [
            "category_15_emissions", "methodology",
            "data_quality", "engagement_strategy",
            "portfolio_alignment",
        ],
        "mandatory": False,
        "version": "2024",
    },
    ComplianceFramework.SBTI_FI.value: {
        "name": "SBTi Financial Institutions",
        "required_fields": [
            "financed_emissions_baseline", "target_year_emissions",
            "sectoral_decarbonization_approach", "portfolio_coverage",
            "temperature_alignment",
        ],
        "mandatory": False,
        "version": "SBTi-FI v1.1 (2024)",
    },
    ComplianceFramework.SB_253.value: {
        "name": "California SB 253",
        "required_fields": [
            "total_co2e", "scope_3_category_detail",
            "methodology", "assurance_opinion",
        ],
        "mandatory": False,
        "version": "2023",
    },
    ComplianceFramework.TCFD.value: {
        "name": "TCFD Recommendations",
        "required_fields": [
            "financed_emissions", "waci", "portfolio_alignment",
            "climate_risk_assessment", "scenario_analysis",
        ],
        "mandatory": False,
        "version": "2017 (final 2023)",
    },
    ComplianceFramework.NZBA.value: {
        "name": "Net-Zero Banking Alliance",
        "required_fields": [
            "financed_emissions_by_sector", "interim_targets",
            "decarbonization_strategy", "portfolio_alignment_1_5c",
        ],
        "mandatory": False,
        "version": "2021",
    },
}

# 14. DQI Scoring -- PCAF score to DQI dimension mapping
DQI_SCORING: Dict[str, Dict[str, Decimal]] = {
    PCAFDataQuality.SCORE_1.value: {
        DQIDimension.TEMPORAL.value: Decimal("5.0"),
        DQIDimension.GEOGRAPHICAL.value: Decimal("5.0"),
        DQIDimension.TECHNOLOGICAL.value: Decimal("5.0"),
        DQIDimension.COMPLETENESS.value: Decimal("5.0"),
        DQIDimension.RELIABILITY.value: Decimal("5.0"),
    },
    PCAFDataQuality.SCORE_2.value: {
        DQIDimension.TEMPORAL.value: Decimal("4.0"),
        DQIDimension.GEOGRAPHICAL.value: Decimal("4.0"),
        DQIDimension.TECHNOLOGICAL.value: Decimal("4.0"),
        DQIDimension.COMPLETENESS.value: Decimal("4.0"),
        DQIDimension.RELIABILITY.value: Decimal("3.5"),
    },
    PCAFDataQuality.SCORE_3.value: {
        DQIDimension.TEMPORAL.value: Decimal("3.0"),
        DQIDimension.GEOGRAPHICAL.value: Decimal("3.0"),
        DQIDimension.TECHNOLOGICAL.value: Decimal("3.0"),
        DQIDimension.COMPLETENESS.value: Decimal("3.0"),
        DQIDimension.RELIABILITY.value: Decimal("3.0"),
    },
    PCAFDataQuality.SCORE_4.value: {
        DQIDimension.TEMPORAL.value: Decimal("2.0"),
        DQIDimension.GEOGRAPHICAL.value: Decimal("2.5"),
        DQIDimension.TECHNOLOGICAL.value: Decimal("2.0"),
        DQIDimension.COMPLETENESS.value: Decimal("2.0"),
        DQIDimension.RELIABILITY.value: Decimal("2.0"),
    },
    PCAFDataQuality.SCORE_5.value: {
        DQIDimension.TEMPORAL.value: Decimal("1.0"),
        DQIDimension.GEOGRAPHICAL.value: Decimal("1.5"),
        DQIDimension.TECHNOLOGICAL.value: Decimal("1.0"),
        DQIDimension.COMPLETENESS.value: Decimal("1.0"),
        DQIDimension.RELIABILITY.value: Decimal("1.0"),
    },
}

# 15. Uncertainty Ranges -- PCAF score 1-5 -> uncertainty percentage
UNCERTAINTY_RANGES: Dict[str, Decimal] = {
    PCAFDataQuality.SCORE_1.value: Decimal("10"),
    PCAFDataQuality.SCORE_2.value: Decimal("20"),
    PCAFDataQuality.SCORE_3.value: Decimal("35"),
    PCAFDataQuality.SCORE_4.value: Decimal("50"),
    PCAFDataQuality.SCORE_5.value: Decimal("60"),
}

# 16. Portfolio Alignment Thresholds -- temperature thresholds
PORTFOLIO_ALIGNMENT_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    PortfolioAlignment.ALIGNED_1_5C.value: {
        "max_temperature_c": Decimal("1.50"),
        "annual_reduction_pct": Decimal("7.0"),
        "description_target": Decimal("0"),  # placeholder
    },
    PortfolioAlignment.ALIGNED_2C.value: {
        "max_temperature_c": Decimal("2.00"),
        "annual_reduction_pct": Decimal("4.2"),
    },
    PortfolioAlignment.NOT_ALIGNED.value: {
        "max_temperature_c": Decimal("99.99"),
        "annual_reduction_pct": Decimal("0.0"),
    },
}


# ==============================================================================
# PYDANTIC MODELS (16) -- all frozen
# ==============================================================================


class EquityInvestmentInput(BaseModel):
    """Input for listed equity or private equity financed emissions calculation.

    PCAF attribution: outstanding_amount / EVIC.

    Example:
        >>> inp = EquityInvestmentInput(
        ...     company_name="Acme Corp",
        ...     isin="US0378331005",
        ...     outstanding_amount=Decimal("10000000"),
        ...     evic=Decimal("500000000"),
        ...     company_emissions_scope1=Decimal("50000"),
        ...     company_emissions_scope2=Decimal("25000"),
        ...     sector=SectorClassification.INDUSTRIALS,
        ...     country="US",
        ...     data_quality_score=PCAFDataQuality.SCORE_1,
        ... )
    """

    company_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Investee company name",
    )
    isin: Optional[str] = Field(
        default=None, min_length=12, max_length=12,
        description="ISIN identifier (12-char)",
    )
    ticker: Optional[str] = Field(
        default=None, max_length=20,
        description="Stock ticker symbol",
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0,
        description="Outstanding investment amount in reporting currency",
    )
    evic: Decimal = Field(
        ..., gt=0,
        description="Enterprise Value Including Cash of investee",
    )
    company_emissions_scope1: Decimal = Field(
        ..., ge=0,
        description="Investee Scope 1 emissions in tCO2e",
    )
    company_emissions_scope2: Decimal = Field(
        ..., ge=0,
        description="Investee Scope 2 emissions in tCO2e",
    )
    company_emissions_scope3: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Investee Scope 3 emissions in tCO2e (optional)",
    )
    company_revenue: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Investee annual revenue for intensity calculation",
    )
    sector: SectorClassification = Field(
        ..., description="GICS sector classification",
    )
    country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code of investee",
    )
    data_quality_score: PCAFDataQuality = Field(
        ..., description="PCAF data quality score (1-5, 1=best)",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    reporting_year: int = Field(
        default=2024, ge=2015, le=2030,
        description="Reporting year",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("outstanding_amount")
    def validate_outstanding_not_exceed_evic(cls, v: Decimal, values: dict) -> Decimal:
        """Validate outstanding does not exceed EVIC (warn but allow)."""
        return v

    @validator("country")
    def validate_country_uppercase(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()


class DebtInvestmentInput(BaseModel):
    """Input for corporate bond / debt financed emissions calculation.

    PCAF attribution: outstanding_amount / (total_equity + total_debt).

    Example:
        >>> inp = DebtInvestmentInput(
        ...     company_name="Beta Inc",
        ...     outstanding_amount=Decimal("5000000"),
        ...     total_equity_plus_debt=Decimal("200000000"),
        ...     company_emissions=Decimal("30000"),
        ...     sector=SectorClassification.MATERIALS,
        ... )
    """

    company_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Investee company name",
    )
    isin: Optional[str] = Field(
        default=None, min_length=12, max_length=12,
        description="Bond ISIN identifier (12-char)",
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0,
        description="Outstanding bond/debt amount in reporting currency",
    )
    total_equity_plus_debt: Decimal = Field(
        ..., gt=0,
        description="Total equity + total debt (book value) of investee",
    )
    company_emissions: Decimal = Field(
        ..., ge=0,
        description="Investee total emissions (Scope 1+2) in tCO2e",
    )
    company_emissions_scope3: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Investee Scope 3 emissions in tCO2e (optional)",
    )
    maturity_date: Optional[str] = Field(
        default=None, description="Bond maturity date (YYYY-MM-DD)",
    )
    coupon_rate: Optional[Decimal] = Field(
        default=None, ge=0, le=100,
        description="Annual coupon rate as percentage",
    )
    sector: SectorClassification = Field(
        ..., description="GICS sector classification",
    )
    country: str = Field(
        default="US", min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    data_quality_score: PCAFDataQuality = Field(
        default=PCAFDataQuality.SCORE_3,
        description="PCAF data quality score",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("country")
    def validate_country_uppercase(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()


class ProjectFinanceInput(BaseModel):
    """Input for project finance financed emissions calculation.

    PCAF attribution: outstanding_amount / total_project_cost.

    Example:
        >>> inp = ProjectFinanceInput(
        ...     project_name="Wind Farm Alpha",
        ...     outstanding_amount=Decimal("20000000"),
        ...     total_project_cost=Decimal("100000000"),
        ...     project_type="renewable_energy",
        ...     project_emissions=Decimal("500"),
        ...     project_lifetime_years=25,
        ... )
    """

    project_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Project name",
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0,
        description="Outstanding project finance exposure",
    )
    total_project_cost: Decimal = Field(
        ..., gt=0,
        description="Total project cost (equity + debt)",
    )
    project_type: str = Field(
        ..., min_length=1,
        description="Project type (e.g., renewable_energy, fossil_fuel, infrastructure)",
    )
    project_emissions: Decimal = Field(
        ..., ge=0,
        description="Annual project emissions in tCO2e",
    )
    project_lifetime_years: int = Field(
        ..., gt=0, le=100,
        description="Project expected lifetime in years",
    )
    sector: SectorClassification = Field(
        default=SectorClassification.ENERGY,
        description="GICS sector classification",
    )
    country: str = Field(
        default="US", min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    data_quality_score: PCAFDataQuality = Field(
        default=PCAFDataQuality.SCORE_2,
        description="PCAF data quality score",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("country")
    def validate_country_uppercase(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()


class CREInvestmentInput(BaseModel):
    """Input for commercial real estate financed emissions calculation.

    PCAF attribution: outstanding_amount / property_value.
    Emissions from building energy use intensity (EUI) and grid EFs.

    Example:
        >>> inp = CREInvestmentInput(
        ...     property_name="Downtown Office Tower",
        ...     property_type=PropertyType.OFFICE,
        ...     outstanding_amount=Decimal("15000000"),
        ...     property_value=Decimal("50000000"),
        ...     floor_area_m2=Decimal("10000"),
        ...     energy_kwh=Decimal("2000000"),
        ...     location_country="US",
        ... )
    """

    property_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Property name or identifier",
    )
    property_type: PropertyType = Field(
        ..., description="Property type classification",
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0,
        description="Outstanding CRE loan or equity amount",
    )
    property_value: Decimal = Field(
        ..., gt=0,
        description="Property value at origination or latest appraisal",
    )
    floor_area_m2: Decimal = Field(
        ..., gt=0,
        description="Gross floor area in square metres",
    )
    energy_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual energy consumption in kWh (if metered)",
    )
    energy_cert_rating: Optional[str] = Field(
        default=None, max_length=10,
        description="Energy Performance Certificate rating (A-G)",
    )
    location_country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    location_region: Optional[str] = Field(
        default=None, max_length=10,
        description="eGRID subregion code (US) or regional identifier",
    )
    climate_zone: Optional[str] = Field(
        default=None,
        description="Climate zone: cold, cool, moderate, warm, hot",
    )
    data_quality_score: PCAFDataQuality = Field(
        default=PCAFDataQuality.SCORE_3,
        description="PCAF data quality score",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("location_country")
    def validate_country_uppercase(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()


class MortgageInput(BaseModel):
    """Input for residential mortgage financed emissions calculation.

    PCAF attribution: outstanding_loan / property_value.

    Example:
        >>> inp = MortgageInput(
        ...     outstanding_loan=Decimal("250000"),
        ...     property_value=Decimal("400000"),
        ...     property_type=PropertyType.RESIDENTIAL,
        ...     floor_area_m2=Decimal("150"),
        ...     energy_cert_rating="C",
        ...     location_country="GB",
        ... )
    """

    outstanding_loan: Decimal = Field(
        ..., gt=0,
        description="Outstanding mortgage loan balance",
    )
    property_value: Decimal = Field(
        ..., gt=0,
        description="Property value at origination",
    )
    property_type: PropertyType = Field(
        default=PropertyType.RESIDENTIAL,
        description="Property type classification",
    )
    floor_area_m2: Decimal = Field(
        ..., gt=0,
        description="Gross floor area in square metres",
    )
    energy_cert_rating: Optional[str] = Field(
        default=None, max_length=10,
        description="Energy Performance Certificate rating (A-G)",
    )
    location_country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    location_region: Optional[str] = Field(
        default=None, max_length=10,
        description="eGRID subregion code (US) or regional identifier",
    )
    data_quality_score: PCAFDataQuality = Field(
        default=PCAFDataQuality.SCORE_4,
        description="PCAF data quality score",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("location_country")
    def validate_country_uppercase(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()


class MotorVehicleLoanInput(BaseModel):
    """Input for motor vehicle loan financed emissions calculation.

    PCAF attribution: outstanding_loan / vehicle_value.

    Example:
        >>> inp = MotorVehicleLoanInput(
        ...     outstanding_loan=Decimal("25000"),
        ...     vehicle_value=Decimal("35000"),
        ...     vehicle_category=VehicleCategory.PASSENGER_CAR,
        ...     make="Toyota",
        ...     model="Camry",
        ...     year=2023,
        ...     annual_distance_km=Decimal("15000"),
        ...     fuel_type="petrol",
        ... )
    """

    outstanding_loan: Decimal = Field(
        ..., gt=0,
        description="Outstanding auto loan balance",
    )
    vehicle_value: Decimal = Field(
        ..., gt=0,
        description="Vehicle value at origination",
    )
    vehicle_category: VehicleCategory = Field(
        ..., description="Vehicle category classification",
    )
    make: Optional[str] = Field(
        default=None, max_length=100,
        description="Vehicle manufacturer",
    )
    model: Optional[str] = Field(
        default=None, max_length=100,
        description="Vehicle model name",
    )
    year: Optional[int] = Field(
        default=None, ge=1990, le=2030,
        description="Vehicle model year",
    )
    annual_distance_km: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Estimated or actual annual distance in km",
    )
    fuel_type: Optional[str] = Field(
        default=None, max_length=50,
        description="Fuel type (petrol, diesel, electric, hybrid, etc.)",
    )
    fuel_standard: FuelStandard = Field(
        default=FuelStandard.OTHER,
        description="Vehicle emissions standard",
    )
    data_quality_score: PCAFDataQuality = Field(
        default=PCAFDataQuality.SCORE_3,
        description="PCAF data quality score",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


class SovereignBondInput(BaseModel):
    """Input for sovereign bond financed emissions calculation.

    PCAF attribution: outstanding_amount / GDP_PPP.
    Attributed emissions = country_total_GHG * attribution_factor.

    Example:
        >>> inp = SovereignBondInput(
        ...     country_code="US",
        ...     outstanding_amount=Decimal("100000000"),
        ... )
    """

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    outstanding_amount: Decimal = Field(
        ..., gt=0,
        description="Outstanding sovereign bond amount in reporting currency",
    )
    gdp_ppp_billion_usd: Optional[Decimal] = Field(
        default=None, gt=0,
        description="GDP PPP in billion USD (looked up from table if not provided)",
    )
    include_lulucf: bool = Field(
        default=False,
        description="Include LULUCF in country emissions (PCAF default: exclude)",
    )
    data_quality_score: PCAFDataQuality = Field(
        default=PCAFDataQuality.SCORE_2,
        description="PCAF data quality score",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency of financial amounts",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("country_code")
    def validate_country_uppercase(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()


class PortfolioInput(BaseModel):
    """Input for portfolio-level financed emissions calculation.

    Wraps a list of investments across multiple asset classes for
    aggregated portfolio metrics (WACI, alignment, coverage).

    Example:
        >>> portfolio = PortfolioInput(
        ...     investments=[...],
        ...     reporting_period_start="2024-01-01",
        ...     reporting_period_end="2024-12-31",
        ...     total_aum=Decimal("5000000000"),
        ... )
    """

    investments: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of investment dicts (parsed into asset-class-specific models)",
    )
    reporting_period_start: str = Field(
        ..., description="Reporting period start date (YYYY-MM-DD)",
    )
    reporting_period_end: str = Field(
        ..., description="Reporting period end date (YYYY-MM-DD)",
    )
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.FINANCIAL_CONTROL,
        description="Organizational boundary consolidation approach",
    )
    total_aum: Decimal = Field(
        ..., gt=0,
        description="Total Assets Under Management in reporting currency",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Reporting currency",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


class InvestmentCalculationResult(BaseModel):
    """Result from a single investment financed emissions calculation."""

    investment_id: str = Field(
        ..., description="Unique identifier for this investment calculation",
    )
    asset_class: AssetClass = Field(
        ..., description="PCAF asset class",
    )
    financed_emissions_kgco2e: Decimal = Field(
        ..., ge=0,
        description="Financed (attributed) emissions in kgCO2e",
    )
    financed_emissions_tco2e: Decimal = Field(
        ..., ge=0,
        description="Financed (attributed) emissions in tCO2e",
    )
    attribution_factor: Decimal = Field(
        ..., ge=0, le=Decimal("1.0"),
        description="PCAF attribution factor (0 to 1)",
    )
    pcaf_data_quality: PCAFDataQuality = Field(
        ..., description="PCAF data quality score assigned",
    )
    carbon_intensity_per_m_invested: Optional[Decimal] = Field(
        default=None, ge=0,
        description="tCO2e per million USD invested",
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used",
    )
    ef_source: EFSource = Field(
        default=EFSource.PCAF_2024,
        description="Emission factor source",
    )
    emission_scope_coverage: EmissionScope = Field(
        default=EmissionScope.SCOPE_1_2,
        description="Emission scopes included in calculation",
    )
    uncertainty_pct: Decimal = Field(
        ..., ge=0,
        description="Uncertainty percentage from PCAF data quality",
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash",
    )

    model_config = ConfigDict(frozen=True)


class PortfolioAggregationResult(BaseModel):
    """Aggregated portfolio-level financed emissions result."""

    total_financed_emissions_tco2e: Decimal = Field(
        ..., ge=0,
        description="Total financed emissions across all investments (tCO2e)",
    )
    total_financed_emissions_kgco2e: Decimal = Field(
        ..., ge=0,
        description="Total financed emissions across all investments (kgCO2e)",
    )
    by_asset_class: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financed emissions breakdown by asset class (tCO2e)",
    )
    by_sector: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financed emissions breakdown by sector (tCO2e)",
    )
    by_country: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financed emissions breakdown by country (tCO2e)",
    )
    waci: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Weighted Average Carbon Intensity (tCO2e/$M revenue)",
    )
    total_aum: Decimal = Field(
        ..., gt=0,
        description="Total AUM in reporting currency",
    )
    coverage_ratio: Decimal = Field(
        ..., ge=0, le=Decimal("1.0"),
        description="Fraction of portfolio covered by financed emissions data",
    )
    weighted_data_quality: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="AUM-weighted average PCAF data quality score",
    )
    reporting_period: str = Field(
        ..., description="Reporting period (start -- end)",
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash for aggregated result",
    )

    model_config = ConfigDict(frozen=True)


class ComplianceResult(BaseModel):
    """Result from compliance check against a specific framework."""

    framework: ComplianceFramework = Field(
        ..., description="Framework checked",
    )
    status: ComplianceStatus = Field(
        ..., description="Compliance status",
    )
    score: Decimal = Field(
        ..., ge=0, le=Decimal("100"),
        description="Compliance score (0-100)",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Specific findings (gaps, issues)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement",
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """Provenance record for audit trail."""

    record_id: str = Field(
        ..., description="Unique provenance record identifier",
    )
    sha256_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the record data",
    )
    parent_hash: Optional[str] = Field(
        default=None, min_length=64, max_length=64,
        description="SHA-256 hash of the parent record (chain link)",
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of record creation",
    )
    operation: str = Field(
        ..., description="Operation that produced this record",
    )
    stage: PipelineStage = Field(
        ..., description="Pipeline stage",
    )
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent that produced this record",
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """Detailed data quality score with PCAF and DQI dimensions."""

    pcaf_score: PCAFDataQuality = Field(
        ..., description="PCAF data quality score (1-5, 1=best)",
    )
    temporal: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Temporal representativeness (1-5)",
    )
    geographical: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Geographical representativeness (1-5)",
    )
    technological: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Technological representativeness (1-5)",
    )
    completeness: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Data completeness (1-5)",
    )
    reliability: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Data reliability (1-5)",
    )
    composite: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Composite DQI score (weighted average)",
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """Result from uncertainty quantification."""

    method: UncertaintyMethod = Field(
        ..., description="Uncertainty method used",
    )
    lower_bound: Decimal = Field(
        ..., description="Lower bound of confidence interval (tCO2e)",
    )
    upper_bound: Decimal = Field(
        ..., description="Upper bound of confidence interval (tCO2e)",
    )
    confidence_level: Decimal = Field(
        ..., ge=0, le=Decimal("1.0"),
        description="Confidence level (e.g., 0.95 for 95%)",
    )
    cv: Decimal = Field(
        ..., ge=0,
        description="Coefficient of variation",
    )
    pcaf_uncertainty_pct: Optional[Decimal] = Field(
        default=None, ge=0,
        description="PCAF score-implied uncertainty percentage",
    )

    model_config = ConfigDict(frozen=True)


class CarbonIntensityResult(BaseModel):
    """Carbon intensity metrics for portfolio or investment."""

    waci: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Weighted Average Carbon Intensity (tCO2e/$M revenue)",
    )
    revenue_intensity: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Revenue-based carbon intensity (tCO2e/$M revenue)",
    )
    physical_intensity: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Physical carbon intensity (tCO2e per unit output)",
    )
    financed_intensity_per_m: Decimal = Field(
        ..., ge=0,
        description="Financed emissions per million USD invested (tCO2e/$M)",
    )

    model_config = ConfigDict(frozen=True)


class PortfolioAlignmentResult(BaseModel):
    """Portfolio alignment assessment result."""

    alignment_status: PortfolioAlignment = Field(
        ..., description="Portfolio temperature alignment classification",
    )
    implied_temperature_rise: Optional[Decimal] = Field(
        default=None, ge=Decimal("0.0"),
        description="Implied temperature rise in degrees Celsius",
    )
    sbti_coverage_pct: Decimal = Field(
        ..., ge=0, le=Decimal("100.0"),
        description="Percentage of portfolio with SBTi-validated targets",
    )
    aligned_pct: Decimal = Field(
        ..., ge=0, le=Decimal("100.0"),
        description="Percentage of portfolio aligned with temperature goal",
    )
    sbti_target_type: Optional[SBTiTarget] = Field(
        default=None,
        description="SBTi target type if applicable",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS (18)
# ==============================================================================


def validate_asset_class(asset_class: str) -> bool:
    """Validate that the asset class string is a valid PCAF asset class.

    Args:
        asset_class: Asset class string to validate.

    Returns:
        True if valid, False otherwise.

    Example:
        >>> validate_asset_class("listed_equity")
        True
        >>> validate_asset_class("crypto")
        False
    """
    valid_values = {ac.value for ac in AssetClass}
    return asset_class in valid_values


def validate_pcaf_score(score: str) -> bool:
    """Validate that the PCAF score string is valid (score_1 through score_5).

    Args:
        score: PCAF data quality score string.

    Returns:
        True if valid, False otherwise.

    Example:
        >>> validate_pcaf_score("score_1")
        True
        >>> validate_pcaf_score("score_6")
        False
    """
    valid_values = {s.value for s in PCAFDataQuality}
    return score in valid_values


def validate_evic(outstanding_amount: Decimal, evic: Decimal) -> bool:
    """Validate that outstanding amount does not exceed EVIC.

    Per PCAF, attribution factor should be <= 1.0.
    outstanding_amount / EVIC > 1.0 is a data quality issue.

    Args:
        outstanding_amount: Investment outstanding amount.
        evic: Enterprise Value Including Cash.

    Returns:
        True if valid (attribution <= 1.0), False otherwise.

    Example:
        >>> validate_evic(Decimal("10000000"), Decimal("500000000"))
        True
        >>> validate_evic(Decimal("600000000"), Decimal("500000000"))
        False
    """
    if evic <= 0:
        return False
    return outstanding_amount <= evic


def calculate_attribution_factor(
    outstanding_amount: Decimal,
    denominator: Decimal,
) -> Decimal:
    """Calculate PCAF attribution factor.

    Attribution factor = outstanding_amount / denominator.
    Capped at 1.0 to prevent over-attribution.

    Args:
        outstanding_amount: Numerator (investment amount).
        denominator: Denominator (EVIC, equity+debt, property value, GDP PPP, etc.).

    Returns:
        Attribution factor as Decimal, capped at 1.0.

    Raises:
        ValueError: If denominator is zero or negative.

    Example:
        >>> calculate_attribution_factor(Decimal("10000000"), Decimal("500000000"))
        Decimal('0.02000000')
    """
    if denominator <= 0:
        raise ValueError(
            f"Denominator must be positive, got {denominator}"
        )
    factor = (outstanding_amount / denominator).quantize(
        _QUANT_8DP, rounding=ROUND_HALF_UP
    )
    # Cap at 1.0 per PCAF guidance
    if factor > Decimal("1.0"):
        factor = Decimal("1.0")
    return factor


def calculate_waci(
    investments: List[Dict[str, Decimal]],
    total_portfolio_value: Decimal,
) -> Decimal:
    """Calculate Weighted Average Carbon Intensity (WACI).

    WACI = sum(weight_i * intensity_i) where:
      weight_i = investment_value_i / total_portfolio_value
      intensity_i = investee_emissions_i / investee_revenue_i (tCO2e/$M)

    Args:
        investments: List of dicts with keys 'value', 'emissions_tco2e', 'revenue'.
        total_portfolio_value: Total portfolio value for weight calculation.

    Returns:
        WACI in tCO2e per million USD revenue.

    Raises:
        ValueError: If total_portfolio_value is zero or negative.

    Example:
        >>> invs = [
        ...     {"value": Decimal("100"), "emissions_tco2e": Decimal("50"), "revenue": Decimal("200")},
        ...     {"value": Decimal("200"), "emissions_tco2e": Decimal("30"), "revenue": Decimal("400")},
        ... ]
        >>> calculate_waci(invs, Decimal("300"))
    """
    if total_portfolio_value <= 0:
        raise ValueError(
            f"Total portfolio value must be positive, got {total_portfolio_value}"
        )
    waci = Decimal("0")
    for inv in investments:
        weight = inv["value"] / total_portfolio_value
        revenue = inv.get("revenue", Decimal("0"))
        if revenue > 0:
            intensity = (inv["emissions_tco2e"] / revenue) * Decimal("1000000")
        else:
            intensity = Decimal("0")
        waci += weight * intensity
    return waci.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


def calculate_carbon_intensity(
    financed_emissions_tco2e: Decimal,
    investment_amount: Decimal,
) -> Decimal:
    """Calculate financed carbon intensity per million invested.

    Intensity = financed_emissions / (investment_amount / 1,000,000).

    Args:
        financed_emissions_tco2e: Financed emissions in tCO2e.
        investment_amount: Investment amount in reporting currency.

    Returns:
        Carbon intensity in tCO2e per million invested.

    Raises:
        ValueError: If investment_amount is zero or negative.

    Example:
        >>> calculate_carbon_intensity(Decimal("500"), Decimal("10000000"))
        Decimal('50.0000')
    """
    if investment_amount <= 0:
        raise ValueError(
            f"Investment amount must be positive, got {investment_amount}"
        )
    millions = investment_amount / Decimal("1000000")
    if millions == 0:
        return Decimal("0")
    intensity = (financed_emissions_tco2e / millions).quantize(
        _QUANT_4DP, rounding=ROUND_HALF_UP
    )
    return intensity


def convert_currency(
    amount: Decimal,
    from_currency: CurrencyCode,
    to_currency: CurrencyCode = CurrencyCode.USD,
) -> Decimal:
    """Convert amount between currencies using CURRENCY_CONVERSION_RATES.

    Args:
        amount: Amount in source currency.
        from_currency: Source currency code.
        to_currency: Target currency code (default USD).

    Returns:
        Converted amount in target currency.

    Raises:
        ValueError: If currency not found in conversion rates.

    Example:
        >>> convert_currency(Decimal("1000"), CurrencyCode.EUR, CurrencyCode.USD)
        Decimal('1085.00000000')
    """
    from_rate = CURRENCY_CONVERSION_RATES.get(from_currency.value)
    to_rate = CURRENCY_CONVERSION_RATES.get(to_currency.value)
    if from_rate is None:
        raise ValueError(
            f"Currency '{from_currency.value}' not found in CURRENCY_CONVERSION_RATES"
        )
    if to_rate is None:
        raise ValueError(
            f"Currency '{to_currency.value}' not found in CURRENCY_CONVERSION_RATES"
        )
    # Convert: amount_in_from * from_rate_to_usd / to_rate_to_usd
    usd_amount = amount * from_rate
    if to_rate == Decimal("0"):
        raise ValueError("Target currency rate is zero")
    result = (usd_amount / to_rate).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
    return result


def apply_cpi_deflation(
    amount: Decimal,
    year: int,
    base_year: int = 2021,
) -> Decimal:
    """Apply CPI deflation to convert nominal spend to base-year USD.

    real_usd = nominal_usd * (base_deflator / year_deflator).

    Args:
        amount: Nominal amount.
        year: Year of the amount.
        base_year: Base year for deflation (default 2021).

    Returns:
        Deflated (real) amount.

    Raises:
        ValueError: If year or base_year not in CPI_DEFLATORS.

    Example:
        >>> apply_cpi_deflation(Decimal("10000"), 2024, 2021)
    """
    year_deflator = CPI_DEFLATORS.get(year)
    base_deflator = CPI_DEFLATORS.get(base_year)
    if year_deflator is None:
        raise ValueError(
            f"CPI deflator not available for year {year}. "
            f"Available: {sorted(CPI_DEFLATORS.keys())}"
        )
    if base_deflator is None:
        raise ValueError(
            f"CPI deflator not available for base year {base_year}"
        )
    if year_deflator == 0:
        raise ValueError("Year deflator is zero, cannot deflate")
    result = (amount * base_deflator / year_deflator).quantize(
        _QUANT_8DP, rounding=ROUND_HALF_UP
    )
    return result


def get_sector_ef(sector: SectorClassification) -> Decimal:
    """Get sector emission factor in tCO2e per million USD revenue.

    Args:
        sector: GICS sector classification.

    Returns:
        Emission factor (tCO2e/$M revenue).

    Raises:
        ValueError: If sector not found.

    Example:
        >>> get_sector_ef(SectorClassification.ENERGY)
        Decimal('450.00')
    """
    entry = SECTOR_EMISSION_FACTORS.get(sector.value)
    if entry is None:
        raise ValueError(f"Sector '{sector.value}' not found in SECTOR_EMISSION_FACTORS")
    return entry["ef_tco2e_per_m_revenue"]


def get_country_emissions(country_code: str) -> Dict[str, Decimal]:
    """Get country emission data (total GHG, GDP PPP, per capita).

    Args:
        country_code: ISO 3166-1 alpha-2 country code (uppercase).

    Returns:
        Dictionary with total_ghg_mt, gdp_ppp_billion_usd, per_capita_tco2e.

    Raises:
        ValueError: If country not found.

    Example:
        >>> data = get_country_emissions("US")
        >>> data["total_ghg_mt"]
        Decimal('5222')
    """
    code = country_code.upper()
    entry = COUNTRY_EMISSION_FACTORS.get(code)
    if entry is None:
        raise ValueError(
            f"Country '{code}' not found in COUNTRY_EMISSION_FACTORS. "
            f"Available: {sorted(COUNTRY_EMISSION_FACTORS.keys())}"
        )
    return entry


def get_grid_ef(region_or_country: str) -> Decimal:
    """Get grid emission factor in kgCO2e per kWh.

    Supports country codes and eGRID subregion codes.

    Args:
        region_or_country: Country code or eGRID subregion code.

    Returns:
        Grid emission factor (kgCO2e/kWh).

    Raises:
        ValueError: If region not found.

    Example:
        >>> get_grid_ef("US")
        Decimal('0.3937')
        >>> get_grid_ef("CAMX")
        Decimal('0.2256')
    """
    code = region_or_country.upper()
    ef = GRID_EMISSION_FACTORS.get(code)
    if ef is None:
        raise ValueError(
            f"Grid region '{code}' not found in GRID_EMISSION_FACTORS"
        )
    return ef


def is_consolidated_in_scope1_2(
    equity_share_pct: Decimal,
    consolidation_approach: ConsolidationApproach,
    has_operational_control: bool = False,
    has_financial_control: bool = False,
) -> bool:
    """Check if an investment is consolidated in Scope 1/2 (DC-INV-001).

    If consolidated, the investment must NOT be counted in Cat 15.

    Args:
        equity_share_pct: Equity share percentage (0-100).
        consolidation_approach: Consolidation approach used.
        has_operational_control: Whether reporting company has operational control.
        has_financial_control: Whether reporting company has financial control.

    Returns:
        True if investment is consolidated in Scope 1/2.

    Example:
        >>> is_consolidated_in_scope1_2(
        ...     Decimal("60"), ConsolidationApproach.EQUITY_SHARE
        ... )
        True
    """
    if consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL:
        return has_operational_control
    elif consolidation_approach == ConsolidationApproach.FINANCIAL_CONTROL:
        return has_financial_control
    elif consolidation_approach == ConsolidationApproach.EQUITY_SHARE:
        return equity_share_pct > Decimal("50")
    return False


def should_exclude_investment(
    equity_share_pct: Decimal,
    consolidation_approach: ConsolidationApproach,
    has_operational_control: bool = False,
    has_financial_control: bool = False,
) -> bool:
    """Determine if an investment should be excluded from Cat 15.

    Wraps is_consolidated_in_scope1_2 for pipeline use.

    Args:
        equity_share_pct: Equity share percentage (0-100).
        consolidation_approach: Consolidation approach.
        has_operational_control: Operational control flag.
        has_financial_control: Financial control flag.

    Returns:
        True if investment should be excluded from Cat 15.
    """
    return is_consolidated_in_scope1_2(
        equity_share_pct=equity_share_pct,
        consolidation_approach=consolidation_approach,
        has_operational_control=has_operational_control,
        has_financial_control=has_financial_control,
    )


def format_financed_emissions(emissions_kgco2e: Decimal) -> str:
    """Format financed emissions for display.

    Displays in tCO2e if >= 1000 kgCO2e, otherwise kgCO2e.

    Args:
        emissions_kgco2e: Emissions in kgCO2e.

    Returns:
        Formatted string with appropriate unit.

    Example:
        >>> format_financed_emissions(Decimal("50000"))
        '50.00 tCO2e'
        >>> format_financed_emissions(Decimal("500"))
        '500.00 kgCO2e'
    """
    if emissions_kgco2e >= Decimal("1000"):
        tco2e = (emissions_kgco2e / Decimal("1000")).quantize(
            _QUANT_2DP, rounding=ROUND_HALF_UP
        )
        return f"{tco2e} tCO2e"
    return f"{emissions_kgco2e.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)} kgCO2e"


def format_intensity(intensity: Decimal, unit: str = "tCO2e/$M") -> str:
    """Format carbon intensity value for display.

    Args:
        intensity: Intensity value.
        unit: Unit string (default 'tCO2e/$M').

    Returns:
        Formatted string.

    Example:
        >>> format_intensity(Decimal("125.4567"))
        '125.46 tCO2e/$M'
    """
    return f"{intensity.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)} {unit}"


def get_pcaf_uncertainty(data_quality_score: PCAFDataQuality) -> Decimal:
    """Get uncertainty percentage for a PCAF data quality score.

    Args:
        data_quality_score: PCAF score (score_1 through score_5).

    Returns:
        Uncertainty percentage (e.g., 10 for Score 1, 60 for Score 5).

    Raises:
        ValueError: If score not found.

    Example:
        >>> get_pcaf_uncertainty(PCAFDataQuality.SCORE_1)
        Decimal('10')
        >>> get_pcaf_uncertainty(PCAFDataQuality.SCORE_5)
        Decimal('60')
    """
    pct = UNCERTAINTY_RANGES.get(data_quality_score.value)
    if pct is None:
        raise ValueError(
            f"PCAF score '{data_quality_score.value}' not found in UNCERTAINTY_RANGES"
        )
    return pct


def get_data_quality_tier(data_quality_score: PCAFDataQuality) -> DataQualityTier:
    """Map PCAF data quality score to a data quality tier.

    Tier 1: Score 1-2 (primary / verified data).
    Tier 2: Score 3 (secondary / activity-based data).
    Tier 3: Score 4-5 (estimated / sector average data).

    Args:
        data_quality_score: PCAF score.

    Returns:
        DataQualityTier classification.

    Example:
        >>> get_data_quality_tier(PCAFDataQuality.SCORE_1)
        <DataQualityTier.TIER_1: 'tier_1'>
        >>> get_data_quality_tier(PCAFDataQuality.SCORE_3)
        <DataQualityTier.TIER_2: 'tier_2'>
        >>> get_data_quality_tier(PCAFDataQuality.SCORE_5)
        <DataQualityTier.TIER_3: 'tier_3'>
    """
    if data_quality_score in (PCAFDataQuality.SCORE_1, PCAFDataQuality.SCORE_2):
        return DataQualityTier.TIER_1
    elif data_quality_score == PCAFDataQuality.SCORE_3:
        return DataQualityTier.TIER_2
    else:
        return DataQualityTier.TIER_3


def calculate_provenance_hash(
    data: Any,
    parent_hash: Optional[str] = None,
) -> str:
    """Calculate SHA-256 provenance hash for audit trail.

    Creates a deterministic hash from input data and optional parent hash
    to form a provenance chain.

    Args:
        data: Data to hash (will be serialized to JSON string).
        parent_hash: Optional parent hash for chain linking.

    Returns:
        SHA-256 hex digest string (64 characters).

    Example:
        >>> h = calculate_provenance_hash({"emissions": "50000"})
        >>> len(h)
        64
    """
    if isinstance(data, BaseModel):
        data_str = data.model_dump_json(indent=None)
    elif isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)

    if parent_hash:
        data_str = f"{parent_hash}:{data_str}"

    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enums (22)
    "AssetClass",
    "InvestmentType",
    "SectorClassification",
    "PCAFDataQuality",
    "CalculationMethod",
    "ConsolidationApproach",
    "EmissionScope",
    "PropertyType",
    "VehicleCategory",
    "FuelStandard",
    "EFSource",
    "CurrencyCode",
    "DataQualityTier",
    "DQIDimension",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "UncertaintyMethod",
    "BatchStatus",
    "GWPSource",
    "PortfolioAlignment",
    "SBTiTarget",

    # Constants (16)
    "PCAF_ATTRIBUTION_RULES",
    "SECTOR_EMISSION_FACTORS",
    "COUNTRY_EMISSION_FACTORS",
    "GRID_EMISSION_FACTORS",
    "BUILDING_EUI_BENCHMARKS",
    "VEHICLE_EMISSION_FACTORS",
    "EEIO_SECTOR_FACTORS",
    "PCAF_DATA_QUALITY_MATRIX",
    "CURRENCY_CONVERSION_RATES",
    "CPI_DEFLATORS",
    "SOVEREIGN_COUNTRY_DATA",
    "CARBON_INTENSITY_BENCHMARKS",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",
    "PORTFOLIO_ALIGNMENT_THRESHOLDS",

    # Input models
    "EquityInvestmentInput",
    "DebtInvestmentInput",
    "ProjectFinanceInput",
    "CREInvestmentInput",
    "MortgageInput",
    "MotorVehicleLoanInput",
    "SovereignBondInput",
    "PortfolioInput",

    # Result models
    "InvestmentCalculationResult",
    "PortfolioAggregationResult",
    "ComplianceResult",
    "ProvenanceRecord",
    "DataQualityScore",
    "UncertaintyResult",
    "CarbonIntensityResult",
    "PortfolioAlignmentResult",

    # Helper functions (18)
    "validate_asset_class",
    "validate_pcaf_score",
    "validate_evic",
    "calculate_attribution_factor",
    "calculate_waci",
    "calculate_carbon_intensity",
    "convert_currency",
    "apply_cpi_deflation",
    "get_sector_ef",
    "get_country_emissions",
    "get_grid_ef",
    "is_consolidated_in_scope1_2",
    "should_exclude_investment",
    "format_financed_emissions",
    "format_intensity",
    "get_pcaf_uncertainty",
    "get_data_quality_tier",
    "calculate_provenance_hash",
]
