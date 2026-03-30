"""
PACK-005 CBAM Complete Pack - Configuration Manager

This module implements the CBAMCompleteConfig class and all supporting Pydantic
models for configuring the CBAM Complete Pack. It extends PACK-004's
CBAMPackConfig with enterprise-grade features: certificate trading, multi-entity
group management, registry API integration, advanced analytics, customs
automation, cross-regulation alignment, audit management, and deep precursor
chain analysis.

Configuration Merge Order (later overrides earlier):
    1. PACK-004 base defaults (inherited Pydantic models)
    2. PACK-005 extended defaults defined here
    3. Pack preset (enterprise_importer / customs_broker / steel_group / etc.)
    4. Sector preset (automotive_oem / construction / chemical_manufacturing)
    5. Environment overrides (CBAM_COMPLETE_* environment variables)
    6. Explicit runtime overrides

Regulatory References:
    - CBAM Regulation (EU) 2023/956 (Articles 1-35)
    - CBAM Implementing Regulation (EU) 2023/1773
    - EU ETS Directive 2003/87/EC
    - EU Customs Code (EU) No 952/2013
    - Anti-circumvention: Article 27

Example:
    >>> config = CBAMCompleteConfig.from_preset("enterprise_importer")
    >>> print(config.trading.buying_strategy)
    <BuyingStrategy.DCA: 'dca'>
    >>> print(config.entity_group.cost_allocation_method)
    <CostAllocationMethod.VOLUME: 'volume'>
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"


# =============================================================================
# Enumerations - Inherited from PACK-004
# =============================================================================


class CBAMGoodsCategory(str, Enum):
    """CBAM goods categories as defined in Annex I of Regulation (EU) 2023/956."""

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class CalculationMethod(str, Enum):
    """Methods for calculating embedded emissions per CBAM methodology."""

    ACTUAL = "actual"
    DEFAULT = "default"
    COUNTRY_DEFAULT = "country_default"


class ReportingPeriod(str, Enum):
    """CBAM reporting period phase."""

    TRANSITIONAL = "transitional"
    DEFINITIVE = "definitive"


class CostScenario(str, Enum):
    """Cost projection scenarios for CBAM certificate price forecasting."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


class VerificationFrequency(str, Enum):
    """Frequency of third-party verification of embedded emissions."""

    ANNUAL = "annual"
    BIENNIAL = "biennial"


class ETSPriceSource(str, Enum):
    """Source for EU ETS / CBAM certificate pricing."""

    AUCTION = "auction"
    SPOT = "spot"
    MANUAL = "manual"


class EmissionFactorSource(str, Enum):
    """Source database for emission factors."""

    EU_DEFAULT = "eu_default"
    IPCC = "ipcc"
    INDUSTRY = "industry"


class DataSubmissionFormat(str, Enum):
    """Supported formats for supplier data submission."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    EXCEL = "excel"


class ReportLanguage(str, Enum):
    """Supported languages for CBAM reports."""

    EN = "EN"
    DE = "DE"
    FR = "FR"
    IT = "IT"
    ES = "ES"
    NL = "NL"
    PL = "PL"
    PT = "PT"


class EUMemberState(str, Enum):
    """EU member states for importer registration."""

    AT = "AT"
    BE = "BE"
    BG = "BG"
    HR = "HR"
    CY = "CY"
    CZ = "CZ"
    DK = "DK"
    EE = "EE"
    FI = "FI"
    FR = "FR"
    DE = "DE"
    GR = "GR"
    HU = "HU"
    IE = "IE"
    IT = "IT"
    LV = "LV"
    LT = "LT"
    LU = "LU"
    MT = "MT"
    NL = "NL"
    PL = "PL"
    PT = "PT"
    RO = "RO"
    SK = "SK"
    SI = "SI"
    ES = "ES"
    SE = "SE"


# =============================================================================
# Enumerations - NEW in PACK-005
# =============================================================================


class BuyingStrategy(str, Enum):
    """
    Certificate buying strategies for optimizing purchase timing and cost.

    - BUDGET_PACED: Spreads purchases evenly across the budget period.
    - PRICE_TRIGGERED: Purchases when price drops below a threshold.
    - BULK_QUARTERLY: Single bulk purchase at each quarter end.
    - DCA: Dollar-cost averaging with regular fixed-amount purchases.
    - OPPORTUNISTIC: Algorithmic buying based on price trend analysis.
    """

    BUDGET_PACED = "budget_paced"
    PRICE_TRIGGERED = "price_triggered"
    BULK_QUARTERLY = "bulk_quarterly"
    DCA = "dca"
    OPPORTUNISTIC = "opportunistic"


class ValuationMethod(str, Enum):
    """
    Valuation methods for certificate portfolio accounting.

    - FIFO: First-in, first-out cost basis.
    - WEIGHTED_AVERAGE: Weighted average cost basis.
    - MARK_TO_MARKET: Current market value.
    """

    FIFO = "fifo"
    WEIGHTED_AVERAGE = "weighted_average"
    MARK_TO_MARKET = "mark_to_market"


class CostAllocationMethod(str, Enum):
    """
    Cost allocation methods for distributing CBAM costs across entities.

    - VOLUME: Proportional to import volume (tonnes).
    - REVENUE: Proportional to import revenue (EUR).
    - PROFIT_CENTER: Based on profit center budgets.
    - EQUAL: Equal split across entities.
    - CUSTOM: Custom allocation percentages.
    """

    VOLUME = "volume"
    REVENUE = "revenue"
    PROFIT_CENTER = "profit_center"
    EQUAL = "equal"
    CUSTOM = "custom"


class AllocationMethod(str, Enum):
    """
    Emission allocation methods for precursor chain analysis.

    - MASS: Allocation proportional to mass of precursor in final product.
    - ECONOMIC: Allocation proportional to economic value.
    - ENERGY: Allocation proportional to energy content.
    """

    MASS = "mass"
    ECONOMIC = "economic"
    ENERGY = "energy"


class EntityRole(str, Enum):
    """
    Role of an entity within a corporate group for CBAM purposes.

    - PARENT: Parent company, typically the consolidated declarant.
    - SUBSIDIARY: Subsidiary entity with own import operations.
    - CUSTOMS_REPRESENTATIVE: Acts on behalf of importers under delegation.
    - BRANCH: Branch office (not a separate legal entity).
    """

    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    CUSTOMS_REPRESENTATIVE = "customs_representative"
    BRANCH = "branch"


class DeclarantStatus(str, Enum):
    """
    Status of the authorized CBAM declarant application.

    Tracks the lifecycle of declarant authorization with the competent
    authority per Articles 5-6 of the CBAM Regulation.
    """

    NOT_APPLIED = "not_applied"
    APPLICATION_SUBMITTED = "application_submitted"
    UNDER_REVIEW = "under_review"
    GRANTED = "granted"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class RegulationTarget(str, Enum):
    """
    Target regulatory frameworks for cross-regulation data mapping.

    CBAM data can feed into multiple frameworks for single-entry compliance.
    """

    CSRD = "csrd"
    CDP = "cdp"
    SBTI = "sbti"
    EU_TAXONOMY = "eu_taxonomy"
    EU_ETS = "eu_ets"
    EUDR = "eudr"


class AntiCircumventionRule(str, Enum):
    """
    Anti-circumvention rule types per Article 27 of the CBAM Regulation.

    Each rule type detects a specific circumvention practice that importers
    must monitor and defend against.
    """

    ORIGIN_CHANGE = "origin_change"
    CN_RECLASSIFICATION = "cn_reclassification"
    SCRAP_RATIO = "scrap_ratio"
    RESTRUCTURING = "restructuring"
    MINOR_PROCESSING = "minor_processing"


# =============================================================================
# Reference Data Constants
# =============================================================================


# EU ETS free allocation phase-out schedule for CBAM sectors (2026-2034).
# Source: Article 31, Regulation (EU) 2023/956
# Values represent percentage of free allocation remaining.
FREE_ALLOCATION_PHASEOUT: Dict[int, float] = {
    2026: 97.5,
    2027: 95.0,
    2028: 90.0,
    2029: 82.5,
    2030: 75.0,
    2031: 60.0,
    2032: 45.0,
    2033: 30.0,
    2034: 0.0,
}


# Expanded CN code mapping (167 codes) organized by goods category.
# Source: Annex I of Regulation (EU) 2023/956
EXPANDED_CN_CODES: Dict[str, List[Dict[str, str]]] = {
    "cement": [
        {"code": "2523 10 00", "description": "Cement clinkers"},
        {"code": "2523 21 00", "description": "White Portland cement"},
        {"code": "2523 29 00", "description": "Other Portland cement"},
        {"code": "2523 30 00", "description": "Aluminous cement"},
        {"code": "2523 90 00", "description": "Other hydraulic cements"},
    ],
    "iron_steel": [
        {"code": "7201 10 11", "description": "Non-alloy pig iron, P <=0.5%, in pigs"},
        {"code": "7201 10 19", "description": "Non-alloy pig iron, P <=0.5%, other"},
        {"code": "7201 10 30", "description": "Non-alloy pig iron, P >0.5%"},
        {"code": "7201 20 00", "description": "Alloy pig iron; spiegeleisen"},
        {"code": "7202 11 20", "description": "Ferro-manganese >2% C, >4% Si"},
        {"code": "7202 11 80", "description": "Ferro-manganese >2% C, other"},
        {"code": "7202 19 00", "description": "Ferro-manganese <=2% C"},
        {"code": "7202 21 00", "description": "Ferro-silicon >55% Si"},
        {"code": "7202 29 10", "description": "Ferro-silicon 4-55% Si, >=10% Mg"},
        {"code": "7202 29 90", "description": "Ferro-silicon 4-55% Si, other"},
        {"code": "7202 30 00", "description": "Ferro-silico-manganese"},
        {"code": "7202 41 10", "description": "Ferro-chromium >4% C, >6% C"},
        {"code": "7202 41 90", "description": "Ferro-chromium >4% C, <=6% C"},
        {"code": "7202 49 10", "description": "Ferro-chromium <=0.05% C"},
        {"code": "7202 49 50", "description": "Ferro-chromium 0.05-0.5% C"},
        {"code": "7202 49 90", "description": "Ferro-chromium 0.5-4% C"},
        {"code": "7202 50 00", "description": "Ferro-silico-chromium"},
        {"code": "7202 60 00", "description": "Ferro-nickel"},
        {"code": "7202 70 00", "description": "Ferro-molybdenum"},
        {"code": "7202 80 00", "description": "Ferro-tungsten and ferro-silico-tungsten"},
        {"code": "7202 91 00", "description": "Ferro-titanium and ferro-silico-titanium"},
        {"code": "7202 92 00", "description": "Ferro-vanadium"},
        {"code": "7202 93 00", "description": "Ferro-niobium"},
        {"code": "7202 99 10", "description": "Ferro-phosphorus"},
        {"code": "7202 99 30", "description": "Ferro-silico-magnesium"},
        {"code": "7202 99 80", "description": "Other ferro-alloys"},
        {"code": "7203 10 00", "description": "DRI from iron ore"},
        {"code": "7203 90 00", "description": "Other spongy ferrous products"},
        {"code": "7204 10 00", "description": "Waste and scrap of cast iron"},
        {"code": "7204 21 10", "description": "Stainless waste/scrap, >=8% Ni"},
        {"code": "7204 21 90", "description": "Other stainless waste/scrap"},
        {"code": "7204 29 00", "description": "Waste/scrap of other alloy steel"},
        {"code": "7204 30 00", "description": "Waste/scrap of tinned iron/steel"},
        {"code": "7204 41 10", "description": "Turnings, shavings, chips, filings"},
        {"code": "7204 41 91", "description": "Bundles of ferrous waste/scrap"},
        {"code": "7204 41 99", "description": "Other ferrous waste/scrap"},
        {"code": "7204 49 10", "description": "Fragmentised (shredded) waste/scrap"},
        {"code": "7204 49 30", "description": "Other ferrous waste/scrap, bundles"},
        {"code": "7204 49 90", "description": "Other ferrous waste/scrap"},
        {"code": "7204 50 00", "description": "Remelting scrap ingots"},
        {"code": "7205 10 00", "description": "Granules of pig iron/steel"},
        {"code": "7205 21 00", "description": "Powders of alloy steel"},
        {"code": "7205 29 00", "description": "Powders of iron/non-alloy steel"},
        {"code": "7206 10 00", "description": "Iron/non-alloy steel ingots"},
        {"code": "7206 90 00", "description": "Other primary forms"},
        {"code": "7207 11 14", "description": "Semi-finished, rect, C<0.25%, w>=2t"},
        {"code": "7207 11 16", "description": "Semi-finished, rect, C<0.25%, other"},
        {"code": "7207 12 10", "description": "Semi-finished, rect, C 0.25-0.6%"},
        {"code": "7207 19 12", "description": "Semi-finished, circular/polygonal"},
        {"code": "7207 19 80", "description": "Other semi-finished non-alloy"},
        {"code": "7207 20 17", "description": "Semi-finished stainless, >=8% Ni"},
        {"code": "7207 20 32", "description": "Semi-finished other alloy, rect"},
        {"code": "7207 20 52", "description": "Semi-finished other alloy, circular"},
        {"code": "7207 20 80", "description": "Other semi-finished alloy steel"},
        {"code": "7208", "description": "Hot-rolled flat, non-alloy, >=600mm"},
        {"code": "7209", "description": "Cold-rolled flat, non-alloy, >=600mm"},
        {"code": "7210", "description": "Clad/plated/coated flat, >=600mm"},
        {"code": "7211", "description": "Flat-rolled, non-alloy, <600mm"},
        {"code": "7212", "description": "Clad/plated/coated flat, <600mm"},
        {"code": "7213", "description": "Hot-rolled bars and rods (coils)"},
        {"code": "7214", "description": "Forged/cold-formed bars and rods"},
        {"code": "7215", "description": "Other bars and rods"},
        {"code": "7216", "description": "Angles, shapes and sections"},
        {"code": "7217", "description": "Wire of iron or non-alloy steel"},
        {"code": "7218", "description": "Stainless steel semi-finished"},
        {"code": "7219", "description": "Stainless flat, >=600mm"},
        {"code": "7220", "description": "Stainless flat, <600mm"},
        {"code": "7221", "description": "Stainless bars, hot-rolled"},
        {"code": "7222", "description": "Stainless bars, other"},
        {"code": "7223", "description": "Stainless wire"},
        {"code": "7224", "description": "Other alloy semi-finished"},
        {"code": "7225", "description": "Other alloy flat, >=600mm"},
        {"code": "7226", "description": "Other alloy flat, <600mm"},
        {"code": "7227", "description": "Other alloy bars, hot-rolled"},
        {"code": "7228", "description": "Other alloy bars, other"},
        {"code": "7229", "description": "Other alloy wire"},
        {"code": "7301", "description": "Sheet piling"},
        {"code": "7302", "description": "Railway track material"},
        {"code": "7303", "description": "Cast iron tubes and pipes"},
        {"code": "7304", "description": "Seamless tubes and pipes"},
        {"code": "7305", "description": "Welded tubes, >406.4mm"},
        {"code": "7306", "description": "Other welded tubes and pipes"},
        {"code": "7307", "description": "Tube/pipe fittings"},
        {"code": "7308", "description": "Structures and parts"},
        {"code": "7309", "description": "Reservoirs/tanks, >300L"},
        {"code": "7310", "description": "Tanks/casks/drums, <=300L"},
        {"code": "7311", "description": "Compressed gas containers"},
        {"code": "7312", "description": "Stranded wire/ropes/cables"},
        {"code": "7313", "description": "Barbed wire"},
        {"code": "7318", "description": "Screws, bolts, nuts, rivets"},
        {"code": "7326", "description": "Other articles of iron/steel"},
    ],
    "aluminium": [
        {"code": "7601 10 00", "description": "Unwrought aluminium, not alloyed"},
        {"code": "7601 20 20", "description": "Unwrought aluminium alloy, slabs/billets"},
        {"code": "7601 20 80", "description": "Unwrought aluminium alloy, other"},
        {"code": "7602 00 11", "description": "Al waste, turnings/shavings/chips"},
        {"code": "7602 00 19", "description": "Other aluminium waste"},
        {"code": "7602 00 90", "description": "Aluminium scrap"},
        {"code": "7603 10 00", "description": "Al powders, non-lamellar"},
        {"code": "7603 20 00", "description": "Al powders, lamellar; Al flakes"},
        {"code": "7604 10 10", "description": "Al bars/rods/profiles, not alloyed, hollow"},
        {"code": "7604 10 90", "description": "Al bars/rods/profiles, not alloyed, other"},
        {"code": "7604 21 00", "description": "Al alloy hollow profiles"},
        {"code": "7604 29 10", "description": "Al alloy bars and rods"},
        {"code": "7604 29 90", "description": "Al alloy profiles, other"},
        {"code": "7605 11 00", "description": "Al wire, not alloyed, >7mm"},
        {"code": "7605 19 00", "description": "Al wire, not alloyed, <=7mm"},
        {"code": "7605 21 00", "description": "Al alloy wire, >7mm"},
        {"code": "7605 29 00", "description": "Al alloy wire, <=7mm"},
        {"code": "7606", "description": "Al plates/sheets/strip, >0.2mm"},
        {"code": "7607", "description": "Al foil, <=0.2mm"},
        {"code": "7608 10 00", "description": "Al tubes/pipes, not alloyed"},
        {"code": "7608 20", "description": "Al alloy tubes/pipes"},
        {"code": "7609 00 00", "description": "Al tube/pipe fittings"},
        {"code": "7610 10 00", "description": "Al doors, windows, frames"},
        {"code": "7610 90 10", "description": "Al bridges, towers, masts"},
        {"code": "7610 90 90", "description": "Other Al structures"},
        {"code": "7611 00 00", "description": "Al reservoirs/tanks, >300L"},
        {"code": "7612 10 00", "description": "Al collapsible tubular, <=300L"},
        {"code": "7612 90 20", "description": "Al rigid tubular, <=300L"},
        {"code": "7612 90 80", "description": "Other Al containers, <=300L"},
        {"code": "7613 00 00", "description": "Al compressed gas containers"},
        {"code": "7614 10 00", "description": "Al stranded wire/cables, steel core"},
        {"code": "7614 90 00", "description": "Al stranded wire/cables, other"},
        {"code": "7616 10 00", "description": "Al nails, tacks, screws, bolts"},
        {"code": "7616 91 00", "description": "Al cloth, grill, netting, fencing"},
        {"code": "7616 99 10", "description": "Al castings"},
        {"code": "7616 99 90", "description": "Other articles of aluminium"},
    ],
    "fertilizers": [
        {"code": "2808 00 00", "description": "Nitric acid; sulphonitric acids"},
        {"code": "2814 10 00", "description": "Anhydrous ammonia"},
        {"code": "2814 20 00", "description": "Ammonia in aqueous solution"},
        {"code": "2834 10 00", "description": "Nitrites"},
        {"code": "3102 10 10", "description": "Urea, >45% N (dry anhydrous)"},
        {"code": "3102 10 90", "description": "Other urea"},
        {"code": "3102 21 00", "description": "Ammonium sulphate"},
        {"code": "3102 29 00", "description": "Double salts ammonium sulphate/nitrate"},
        {"code": "3102 30 10", "description": "Ammonium nitrate in aqueous solution"},
        {"code": "3102 30 90", "description": "Other ammonium nitrate"},
        {"code": "3102 40 10", "description": "AN + CaCO3 mixtures, N <=28%"},
        {"code": "3102 40 90", "description": "AN + CaCO3 mixtures, N >28%"},
        {"code": "3102 50 00", "description": "Sodium nitrate"},
        {"code": "3102 60 00", "description": "Calcium nitrate + AN double salts"},
        {"code": "3102 80 00", "description": "UAN solutions"},
        {"code": "3102 90 00", "description": "Other nitrogenous fertilizers"},
        {"code": "3103 11 00", "description": "Triple superphosphate (>=35% P2O5)"},
        {"code": "3103 19 00", "description": "Other superphosphates"},
        {"code": "3103 90 00", "description": "Other phosphatic fertilizers"},
        {"code": "3104 20 10", "description": "Potassium chloride, K2O <=60%"},
        {"code": "3104 20 50", "description": "Potassium chloride, K2O >60%"},
        {"code": "3104 30 00", "description": "Potassium sulphate"},
        {"code": "3104 90 00", "description": "Other potassic fertilizers"},
        {"code": "3105 10 00", "description": "Fertilizers in tablets/packages <=10kg"},
        {"code": "3105 20 10", "description": "NPK fertilizers, N >10%"},
        {"code": "3105 20 90", "description": "Other NPK fertilizers"},
        {"code": "3105 30 00", "description": "Diammonium phosphate (DAP)"},
        {"code": "3105 40 00", "description": "Monoammonium phosphate (MAP)"},
        {"code": "3105 51 00", "description": "NP fertilizers (nitrates+phosphates)"},
        {"code": "3105 59 00", "description": "Other NP fertilizers"},
        {"code": "3105 60 00", "description": "PK fertilizers"},
        {"code": "3105 90 20", "description": "Other fertilizers, N >10%"},
        {"code": "3105 90 80", "description": "Other fertilizers, N <=10%"},
    ],
    "electricity": [
        {"code": "2716 00 00", "description": "Electrical energy"},
    ],
    "hydrogen": [
        {"code": "2804 10 00", "description": "Hydrogen"},
    ],
}


# CBAM penalty rate per tCO2e of non-surrendered certificates.
# Source: Article 26, Regulation (EU) 2023/956.
# Base rate EUR 100/tCO2e, indexed to Eurozone inflation (HICP).
CBAM_PENALTIES: Dict[str, Any] = {
    "base_rate_eur_per_tco2e": 100.0,
    "inflation_indexed": True,
    "inflation_index": "HICP_Eurozone",
    "projected_rates": {
        2026: 100.0,
        2027: 103.0,
        2028: 106.1,
        2029: 109.3,
        2030: 112.6,
        2031: 115.9,
        2032: 119.4,
        2033: 123.0,
        2034: 126.7,
        2035: 130.5,
    },
    "late_surrender_additional_pct": 50.0,
    "repeated_offense_multiplier": 2.0,
    "non_compliance_reporting_penalty_eur": 50000.0,
}


# Third-country carbon pricing schemes recognized for CBAM deductions.
# Source: Article 9, Regulation (EU) 2023/956.
# Carbon price must be "effectively paid" and "not subject to rebate or refund".
THIRD_COUNTRY_CARBON_PRICING: Dict[str, Dict[str, Any]] = {
    "CN": {"name": "China National ETS", "type": "ets", "coverage": "power_sector", "price_eur_2024": 8.50, "currency": "CNY", "recognized": True},
    "KR": {"name": "Korea ETS", "type": "ets", "coverage": "industrial", "price_eur_2024": 12.00, "currency": "KRW", "recognized": True},
    "GB": {"name": "UK ETS", "type": "ets", "coverage": "industrial", "price_eur_2024": 45.00, "currency": "GBP", "recognized": True},
    "NZ": {"name": "New Zealand ETS", "type": "ets", "coverage": "economy_wide", "price_eur_2024": 28.00, "currency": "NZD", "recognized": True},
    "CA": {"name": "Canadian Federal Carbon Price", "type": "carbon_tax", "coverage": "economy_wide", "price_eur_2024": 50.00, "currency": "CAD", "recognized": True},
    "CA-QC": {"name": "Quebec Cap-and-Trade", "type": "ets", "coverage": "industrial", "price_eur_2024": 25.00, "currency": "CAD", "recognized": True},
    "CA-AB": {"name": "Alberta TIER", "type": "ets", "coverage": "industrial", "price_eur_2024": 50.00, "currency": "CAD", "recognized": True},
    "JP": {"name": "Japan Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 2.50, "currency": "JPY", "recognized": True},
    "SG": {"name": "Singapore Carbon Tax", "type": "carbon_tax", "coverage": "industrial", "price_eur_2024": 18.00, "currency": "SGD", "recognized": True},
    "ZA": {"name": "South Africa Carbon Tax", "type": "carbon_tax", "coverage": "industrial", "price_eur_2024": 7.50, "currency": "ZAR", "recognized": True},
    "MX": {"name": "Mexico Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 3.00, "currency": "MXN", "recognized": True},
    "CL": {"name": "Chile Carbon Tax", "type": "carbon_tax", "coverage": "power_sector", "price_eur_2024": 5.00, "currency": "CLP", "recognized": True},
    "CO": {"name": "Colombia Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 4.50, "currency": "COP", "recognized": True},
    "UA": {"name": "Ukraine Carbon Tax", "type": "carbon_tax", "coverage": "industrial", "price_eur_2024": 1.00, "currency": "UAH", "recognized": True},
    "ID": {"name": "Indonesia Carbon Tax", "type": "carbon_tax", "coverage": "power_sector", "price_eur_2024": 2.00, "currency": "IDR", "recognized": True},
    "TW": {"name": "Taiwan Carbon Fee", "type": "carbon_tax", "coverage": "industrial", "price_eur_2024": 9.00, "currency": "TWD", "recognized": True},
    "CH": {"name": "Switzerland CO2 Levy", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 110.00, "currency": "CHF", "recognized": True},
    "CH-ETS": {"name": "Switzerland ETS", "type": "ets", "coverage": "industrial", "price_eur_2024": 72.00, "currency": "CHF", "recognized": True},
    "NO": {"name": "Norway Carbon Tax", "type": "carbon_tax", "coverage": "petroleum", "price_eur_2024": 85.00, "currency": "NOK", "recognized": True},
    "SE": {"name": "Sweden Carbon Tax", "type": "carbon_tax", "coverage": "heating_fuels", "price_eur_2024": 115.00, "currency": "SEK", "recognized": True},
    "FI": {"name": "Finland Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 77.00, "currency": "EUR", "recognized": True},
    "DK": {"name": "Denmark Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 25.00, "currency": "DKK", "recognized": True},
    "IS": {"name": "Iceland Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 33.00, "currency": "ISK", "recognized": True},
    "LI": {"name": "Liechtenstein CO2 Levy", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 110.00, "currency": "CHF", "recognized": True},
    "AR": {"name": "Argentina Carbon Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 5.00, "currency": "ARS", "recognized": True},
    "UY": {"name": "Uruguay Carbon Tax", "type": "carbon_tax", "coverage": "transport", "price_eur_2024": 100.00, "currency": "USD", "recognized": True},
    "PT": {"name": "Portugal Carbon Tax", "type": "carbon_tax", "coverage": "non_ets_sectors", "price_eur_2024": 56.00, "currency": "EUR", "recognized": True},
    "IE": {"name": "Ireland Carbon Tax", "type": "carbon_tax", "coverage": "non_ets_sectors", "price_eur_2024": 56.00, "currency": "EUR", "recognized": True},
    "EE": {"name": "Estonia CO2 Charge", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 2.00, "currency": "EUR", "recognized": True},
    "LV": {"name": "Latvia Natural Resource Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 15.00, "currency": "EUR", "recognized": True},
    "PL": {"name": "Poland Environmental Fee", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 0.07, "currency": "PLN", "recognized": True},
    "SI": {"name": "Slovenia CO2 Tax", "type": "carbon_tax", "coverage": "fossil_fuels", "price_eur_2024": 17.30, "currency": "EUR", "recognized": True},
    "KZ": {"name": "Kazakhstan ETS", "type": "ets", "coverage": "industrial", "price_eur_2024": 1.50, "currency": "KZT", "recognized": True},
    "TR": {"name": "Turkey (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "TRY", "recognized": False},
    "IN": {"name": "India (PAT scheme only)", "type": "other", "coverage": "industrial_efficiency", "price_eur_2024": 0.0, "currency": "INR", "recognized": False},
    "RU": {"name": "Russia (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "RUB", "recognized": False},
    "BR": {"name": "Brazil (ETS in development)", "type": "ets_planned", "coverage": "pending", "price_eur_2024": 0.0, "currency": "BRL", "recognized": False},
    "EG": {"name": "Egypt (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "EGP", "recognized": False},
    "VN": {"name": "Vietnam (ETS in development)", "type": "ets_planned", "coverage": "pending", "price_eur_2024": 0.0, "currency": "VND", "recognized": False},
    "TH": {"name": "Thailand (ETS in development)", "type": "ets_planned", "coverage": "pending", "price_eur_2024": 0.0, "currency": "THB", "recognized": False},
    "MY": {"name": "Malaysia (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "MYR", "recognized": False},
    "PH": {"name": "Philippines (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "PHP", "recognized": False},
    "BD": {"name": "Bangladesh (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "BDT", "recognized": False},
    "PK": {"name": "Pakistan (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "PKR", "recognized": False},
    "NG": {"name": "Nigeria (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "NGN", "recognized": False},
    "GH": {"name": "Ghana (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "GHS", "recognized": False},
    "KE": {"name": "Kenya (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "KES", "recognized": False},
    "AE": {"name": "UAE (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "AED", "recognized": False},
    "SA": {"name": "Saudi Arabia (no carbon pricing)", "type": "none", "coverage": "none", "price_eur_2024": 0.0, "currency": "SAR", "recognized": False},
    "US": {"name": "US (no federal carbon price)", "type": "other", "coverage": "state_level", "price_eur_2024": 0.0, "currency": "USD", "recognized": False},
    "US-CA": {"name": "California Cap-and-Trade", "type": "ets", "coverage": "economy_wide", "price_eur_2024": 30.00, "currency": "USD", "recognized": True},
    "US-RGGI": {"name": "RGGI (Northeast US)", "type": "ets", "coverage": "power_sector", "price_eur_2024": 13.00, "currency": "USD", "recognized": True},
}


# Anti-circumvention detection thresholds per rule type.
# Source: Article 27 guidance and Commission methodology.
ANTI_CIRCUMVENTION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "origin_change": {
        "description": "Detects goods re-routed through non-CBAM countries to avoid origin-based pricing",
        "volume_shift_pct_trigger": 30.0,
        "lookback_months": 12,
        "min_volume_tonnes": 100.0,
        "new_supplier_country_alert": True,
        "severity": "critical",
    },
    "cn_reclassification": {
        "description": "Flags suspicious CN code changes for same physical goods",
        "reclassification_frequency_trigger": 3,
        "lookback_months": 6,
        "same_supplier_different_cn_alert": True,
        "cn_heading_change_alert": True,
        "severity": "major",
    },
    "scrap_ratio": {
        "description": "Monitors unusual scrap-to-primary ratios indicating misclassification",
        "max_scrap_ratio_eaf": 1.05,
        "min_scrap_ratio_eaf": 0.70,
        "sudden_ratio_change_pct": 25.0,
        "lookback_months": 6,
        "severity": "major",
    },
    "restructuring": {
        "description": "Detects supply chain restructuring to circumvent CBAM",
        "new_intermediary_alert": True,
        "intermediary_country_whitelist": [],
        "processing_country_change_alert": True,
        "lookback_months": 12,
        "severity": "critical",
    },
    "minor_processing": {
        "description": "Identifies minimal transformation to change tariff classification",
        "value_added_threshold_pct": 15.0,
        "processing_country_different_from_origin": True,
        "assembly_only_alert": True,
        "lookback_months": 6,
        "severity": "major",
    },
}


# EU default emission factors by goods category and product type.
# Source: Commission Implementing Regulation (EU) 2023/1773, Annex III
EU_DEFAULT_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "cement": {
        "clinker": 0.84,
        "portland_cement": 0.68,
        "aluminous_cement": 0.72,
        "other_hydraulic_cement": 0.60,
    },
    "iron_steel": {
        "pig_iron_bf_bof": 1.85,
        "pig_iron_bf_bof_pellets": 1.92,
        "crude_steel_bof": 1.85,
        "crude_steel_eaf": 0.45,
        "crude_steel_dri_eaf": 1.10,
        "hot_rolled_flat": 1.98,
        "cold_rolled_flat": 2.15,
        "coated_flat": 2.25,
        "long_products": 1.95,
        "tubes_pipes": 2.10,
        "stainless_steel": 2.80,
        "alloy_steel": 2.50,
        "ferro_alloys": 3.50,
    },
    "aluminium": {
        "unwrought_primary": 8.00,
        "unwrought_secondary": 0.50,
        "bars_rods_profiles": 8.50,
        "plates_sheets": 8.80,
        "foil": 9.20,
        "tubes_pipes": 8.60,
        "structures": 9.00,
    },
    "fertilizers": {
        "ammonia_anhydrous": 2.10,
        "ammonia_aqueous": 1.80,
        "urea": 1.60,
        "ammonium_nitrate": 3.10,
        "ammonium_sulphate": 1.50,
        "nitric_acid": 1.70,
        "npk_fertilizer": 2.40,
        "dap": 1.90,
        "map": 1.70,
        "uan_solution": 2.00,
    },
    "electricity": {
        "eu_average": 0.23,
    },
    "hydrogen": {
        "grey_smr": 10.00,
        "blue_smr_ccs": 3.00,
        "green_electrolysis": 0.50,
        "turquoise_pyrolysis": 4.00,
    },
}


# Country-specific default emission factors for major exporting countries.
COUNTRY_DEFAULT_FACTORS: Dict[str, Dict[str, float]] = {
    "TR": {"iron_steel_bf_bof": 1.95, "iron_steel_eaf": 0.52, "cement_clinker": 0.88, "aluminium_primary": 9.50, "electricity_grid": 0.48},
    "CN": {"iron_steel_bf_bof": 2.15, "iron_steel_eaf": 0.65, "cement_clinker": 0.92, "aluminium_primary": 12.50, "fertilizer_ammonia": 2.80, "electricity_grid": 0.58, "hydrogen_grey": 12.00},
    "RU": {"iron_steel_bf_bof": 2.05, "iron_steel_eaf": 0.55, "cement_clinker": 0.85, "aluminium_primary": 5.50, "fertilizer_ammonia": 2.20, "electricity_grid": 0.42, "hydrogen_grey": 10.50},
    "IN": {"iron_steel_bf_bof": 2.50, "iron_steel_eaf": 0.70, "cement_clinker": 0.95, "aluminium_primary": 14.00, "fertilizer_ammonia": 2.90, "electricity_grid": 0.72},
    "UA": {"iron_steel_bf_bof": 2.20, "iron_steel_eaf": 0.58, "cement_clinker": 0.90, "electricity_grid": 0.45},
    "EG": {"iron_steel_eaf": 0.60, "fertilizer_ammonia": 2.40, "electricity_grid": 0.50},
    "BR": {"iron_steel_bf_bof": 1.90, "iron_steel_eaf": 0.48, "aluminium_primary": 6.00, "electricity_grid": 0.08},
    "ZA": {"iron_steel_bf_bof": 2.30, "cement_clinker": 0.92, "aluminium_primary": 13.00, "electricity_grid": 0.95},
    "KR": {"iron_steel_bf_bof": 1.88, "iron_steel_eaf": 0.50, "cement_clinker": 0.80, "aluminium_primary": 8.50, "electricity_grid": 0.42},
    "NO": {"aluminium_primary": 2.50, "electricity_grid": 0.01},
    "IS": {"aluminium_primary": 2.00, "electricity_grid": 0.00},
    "VN": {"iron_steel_bf_bof": 2.35, "iron_steel_eaf": 0.62, "cement_clinker": 0.93, "electricity_grid": 0.55},
    "ID": {"iron_steel_eaf": 0.58, "cement_clinker": 0.91, "aluminium_primary": 11.00, "electricity_grid": 0.65},
    "TH": {"iron_steel_eaf": 0.55, "cement_clinker": 0.88, "electricity_grid": 0.52},
    "PK": {"iron_steel_eaf": 0.68, "cement_clinker": 0.94, "electricity_grid": 0.48},
}


# Flat CN code to category lookup
CN_CODE_TO_CATEGORY: Dict[str, CBAMGoodsCategory] = {}


def _build_cn_code_lookup() -> None:
    """Build the flat CN code to category lookup table from EXPANDED_CN_CODES."""
    category_enum_map = {
        "cement": CBAMGoodsCategory.CEMENT,
        "iron_steel": CBAMGoodsCategory.IRON_STEEL,
        "aluminium": CBAMGoodsCategory.ALUMINIUM,
        "fertilizers": CBAMGoodsCategory.FERTILIZERS,
        "electricity": CBAMGoodsCategory.ELECTRICITY,
        "hydrogen": CBAMGoodsCategory.HYDROGEN,
    }
    for cat_key, cn_list in EXPANDED_CN_CODES.items():
        category = category_enum_map[cat_key]
        for entry in cn_list:
            code = entry["code"].replace(" ", "")
            CN_CODE_TO_CATEGORY[code] = category
            prefix_4 = code[:4]
            if prefix_4 not in CN_CODE_TO_CATEGORY:
                CN_CODE_TO_CATEGORY[prefix_4] = category


_build_cn_code_lookup()

# Default CN codes grouped by category
DEFAULT_CN_CODES_BY_CATEGORY: Dict[str, List[str]] = {
    cat: [entry["code"] for entry in entries]
    for cat, entries in EXPANDED_CN_CODES.items()
}


# =============================================================================
# PACK-004 Inherited Pydantic Models
# =============================================================================


class ImporterConfig(BaseModel):
    """Configuration for the EU importer / authorized CBAM declarant."""

    company_name: str = Field("", description="Legal name of the importing company")
    eori_number: str = Field("", description="EORI number (CC + up to 15 alphanumeric)")
    authorized_declarant: str = Field("", description="Authorized CBAM declarant name")
    eu_member_state: Optional[EUMemberState] = Field(None, description="EU member state of establishment")
    cbam_registry_id: str = Field("", description="CBAM registry account identifier")
    contact_email: str = Field("", description="Primary contact email")
    contact_phone: str = Field("", description="Primary contact phone number")
    customs_office_code: str = Field("", description="Customs office code of import")

    @field_validator("eori_number")
    @classmethod
    def validate_eori_format(cls, v: str) -> str:
        """Validate EORI number format."""
        if v and len(v) < 3:
            raise ValueError("EORI must be >= 3 chars (2-letter country code + identifier)")
        if v and not v[:2].isalpha():
            raise ValueError("EORI must start with 2-letter country code")
        return v.upper() if v else v


class GoodsCategoryConfig(BaseModel):
    """Configuration for CBAM goods categories and CN code mapping."""

    enabled_categories: List[CBAMGoodsCategory] = Field(
        default_factory=lambda: list(CBAMGoodsCategory),
        description="List of enabled CBAM goods categories",
    )
    cn_codes_per_category: Dict[str, List[str]] = Field(
        default_factory=lambda: dict(DEFAULT_CN_CODES_BY_CATEGORY),
        description="Mapping of goods category to CN codes",
    )
    custom_cn_codes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Additional CN codes beyond Annex I defaults",
    )
    precursor_tracking: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "cement": ["clinker"],
            "iron_steel": ["pig_iron", "direct_reduced_iron", "crude_steel", "ferro_alloys"],
            "aluminium": ["alumina", "unwrought_aluminium"],
            "fertilizers": ["ammonia", "nitric_acid", "urea"],
            "electricity": [],
            "hydrogen": [],
        },
        description="Precursor products tracked per goods category",
    )

    def get_cn_codes_for_category(self, category: CBAMGoodsCategory) -> List[str]:
        """Return all CN codes for a given goods category."""
        cat_key = category.value
        base_codes = self.cn_codes_per_category.get(cat_key, [])
        custom_codes = self.custom_cn_codes.get(cat_key, [])
        return base_codes + custom_codes

    def get_all_enabled_cn_codes(self) -> List[str]:
        """Return all CN codes across all enabled categories."""
        codes: List[str] = []
        for category in self.enabled_categories:
            codes.extend(self.get_cn_codes_for_category(category))
        return codes


class EmissionConfig(BaseModel):
    """Configuration for embedded emission calculation methodology."""

    calculation_method: CalculationMethod = Field(CalculationMethod.ACTUAL, description="Preferred calculation method")
    fallback_method: CalculationMethod = Field(CalculationMethod.COUNTRY_DEFAULT, description="Fallback method")
    default_markup_percentage: float = Field(25.0, ge=0.0, le=100.0, description="Markup on country defaults")
    indirect_emissions_included: bool = Field(True, description="Include indirect emissions from electricity")
    precursor_tracking_enabled: bool = Field(True, description="Track precursor emissions")
    emission_factor_source: EmissionFactorSource = Field(EmissionFactorSource.EU_DEFAULT, description="EF source")
    emission_factor_vintage_year: int = Field(2024, ge=2020, le=2030, description="EF reference year")
    carbon_price_deduction_enabled: bool = Field(True, description="Enable carbon price deductions")
    country_carbon_price_sources: List[str] = Field(
        default_factory=lambda: ["eu_ets", "uk_ets", "china_ets", "korea_ets", "carbon_tax_registry"],
        description="Recognized carbon pricing mechanisms",
    )


class CertificateConfig(BaseModel):
    """Configuration for CBAM certificate management."""

    ets_price_source: ETSPriceSource = Field(ETSPriceSource.AUCTION, description="Certificate pricing source")
    manual_price_eur_per_tco2e: Optional[float] = Field(None, ge=0.0, description="Manual price override")
    free_allocation_enabled: bool = Field(True, description="Apply free allocation phase-out")
    carbon_deduction_enabled: bool = Field(True, description="Deduct country carbon price")
    cost_scenario: CostScenario = Field(CostScenario.MID, description="Default cost scenario")
    cost_scenario_prices: Dict[str, Dict[int, float]] = Field(
        default_factory=lambda: {
            "low": {2026: 55.0, 2027: 58.0, 2028: 60.0, 2029: 62.0, 2030: 65.0, 2031: 68.0, 2032: 72.0, 2033: 75.0, 2034: 78.0, 2035: 80.0},
            "mid": {2026: 75.0, 2027: 80.0, 2028: 85.0, 2029: 90.0, 2030: 95.0, 2031: 100.0, 2032: 108.0, 2033: 115.0, 2034: 120.0, 2035: 125.0},
            "high": {2026: 100.0, 2027: 110.0, 2028: 120.0, 2029: 130.0, 2030: 140.0, 2031: 155.0, 2032: 170.0, 2033: 185.0, 2034: 200.0, 2035: 220.0},
        },
        description="Price projections per scenario per year",
    )
    quarterly_holding_target_pct: float = Field(50.0, ge=0.0, le=100.0, description="Quarterly holding target %")
    surrender_deadline_month: int = Field(5, ge=1, le=12, description="Surrender deadline month (May=5)")
    surrender_deadline_day: int = Field(31, ge=1, le=31, description="Surrender deadline day")
    repurchase_enabled: bool = Field(True, description="Enable certificate repurchase")
    repurchase_max_pct: float = Field(33.33, ge=0.0, le=100.0, description="Max repurchase % (one-third)")

    @model_validator(mode="after")
    def validate_manual_price(self) -> "CertificateConfig":
        """Ensure manual price is provided when source is MANUAL."""
        if self.ets_price_source == ETSPriceSource.MANUAL and self.manual_price_eur_per_tco2e is None:
            raise ValueError("manual_price_eur_per_tco2e required when ets_price_source is MANUAL")
        return self


class QuarterlyConfig(BaseModel):
    """Configuration for quarterly CBAM report generation and submission."""

    auto_schedule: bool = Field(True, description="Auto-schedule quarterly reports")
    submission_deadline_buffer_days: int = Field(7, ge=0, le=30, description="Days buffer before deadline")
    amendment_window_days: int = Field(60, ge=0, le=120, description="Amendment window in days")
    xml_validation_enabled: bool = Field(True, description="Enable XML schema validation")
    report_language: ReportLanguage = Field(ReportLanguage.EN, description="Primary report language")
    additional_languages: List[ReportLanguage] = Field(default_factory=list, description="Additional languages")
    include_supporting_documents: bool = Field(True, description="Attach supporting docs")
    quarterly_deadlines: Dict[str, str] = Field(
        default_factory=lambda: {"Q1": "April 30", "Q2": "July 31", "Q3": "October 31", "Q4": "January 31"},
        description="Deadlines per quarter",
    )
    archive_reports: bool = Field(True, description="Archive submitted reports with provenance")
    max_retries_on_submission_failure: int = Field(3, ge=0, le=10, description="Max submission retries")


class SupplierConfig(BaseModel):
    """Configuration for supplier emission data management."""

    auto_request_frequency_months: int = Field(3, ge=1, le=12, description="Questionnaire dispatch frequency")
    quality_threshold: float = Field(70.0, ge=0.0, le=100.0, description="Min data quality score")
    max_installations_per_supplier: int = Field(20, ge=1, le=100, description="Max installations/supplier")
    eori_validation_enabled: bool = Field(True, description="Validate supplier EORI")
    data_submission_format: DataSubmissionFormat = Field(DataSubmissionFormat.JSON, description="Preferred format")
    accepted_formats: List[DataSubmissionFormat] = Field(
        default_factory=lambda: list(DataSubmissionFormat), description="Accepted formats"
    )
    questionnaire_template_version: str = Field("1.0.0", description="Questionnaire template version")
    reminder_days_before_deadline: List[int] = Field(
        default_factory=lambda: [30, 14, 7, 3, 1], description="Reminder schedule"
    )
    auto_fallback_to_defaults: bool = Field(True, description="Fallback to defaults if no actual data")
    supplier_portal_enabled: bool = Field(True, description="Enable supplier portal")
    data_retention_years: int = Field(10, ge=5, le=20, description="Data retention period")

    @field_validator("quality_threshold")
    @classmethod
    def validate_quality_threshold(cls, v: float) -> float:
        """Warn if quality threshold is very low."""
        if v < 30.0:
            logger.warning("Quality threshold %.1f%% is very low; consider >=70%%", v)
        return v


class DeMinimisConfig(BaseModel):
    """Configuration for de minimis threshold monitoring."""

    monitoring_enabled: bool = Field(True, description="Enable threshold monitoring")
    threshold_tonnes: float = Field(50.0, ge=0.0, description="Annual tonnage threshold")
    threshold_value_eur: float = Field(150.0, ge=0.0, description="Per-consignment value threshold EUR")
    alert_thresholds: List[float] = Field(
        default_factory=lambda: [80.0, 90.0, 95.0, 100.0], description="Alert percentage thresholds"
    )
    auto_exemption: bool = Field(False, description="Auto-generate exemption docs")
    sector_grouping: bool = Field(True, description="Assess per goods category")
    monitoring_frequency: str = Field("per_import", description="Check frequency")
    cumulative_tracking_start: str = Field("january_1", description="Cumulative tracking start")
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "dashboard", "webhook"], description="Alert channels"
    )

    @field_validator("alert_thresholds")
    @classmethod
    def validate_alert_thresholds(cls, v: List[float]) -> List[float]:
        """Ensure alert thresholds are sorted and in valid range."""
        for threshold in v:
            if threshold < 0.0 or threshold > 200.0:
                raise ValueError(f"Alert threshold {threshold} must be between 0 and 200")
        return sorted(v)


class VerificationConfig(BaseModel):
    """Configuration for third-party verification of embedded emissions."""

    frequency: VerificationFrequency = Field(VerificationFrequency.ANNUAL, description="Verification frequency")
    materiality_threshold_pct: float = Field(5.0, ge=0.0, le=25.0, description="Materiality threshold %")
    verifier_accreditation_required: bool = Field(True, description="Require accredited verifier")
    accepted_accreditation_bodies: List[str] = Field(
        default_factory=lambda: ["DAkkS", "UKAS", "COFRAC", "ACCREDIA", "ENAC", "RvA", "SAS", "JAS-ANZ"],
        description="Accepted accreditation bodies",
    )
    evidence_retention_years: int = Field(10, ge=5, le=20, description="Evidence retention period")
    verification_standards: List[str] = Field(
        default_factory=lambda: ["ISO 14064-3", "ISO 14065", "CBAM Delegated Regulation"],
        description="Applicable standards",
    )
    site_visit_required: bool = Field(True, description="Require site visits")
    remote_verification_allowed: bool = Field(False, description="Allow remote verification")
    sampling_methodology: str = Field("risk_based", description="Sampling methodology")
    max_findings_before_rejection: int = Field(5, ge=1, le=20, description="Max major findings")
    corrective_action_deadline_days: int = Field(30, ge=7, le=90, description="Corrective action deadline")


# =============================================================================
# PACK-005 NEW Pydantic Models (8 sub-configs)
# =============================================================================


class CertificateTradingConfig(BaseModel):
    """
    Configuration for CBAM certificate trading and portfolio optimization.

    Controls buying strategy, price alert triggers, valuation methodology,
    budget limits, auto-purchase settings, and quarterly rebalancing targets.
    """

    buying_strategy: BuyingStrategy = Field(
        BuyingStrategy.DCA,
        description="Certificate buying strategy for cost optimization",
    )
    price_alert_below_eur: Optional[float] = Field(
        None, ge=0.0,
        description="Alert when certificate price drops below this EUR/tCO2e level",
    )
    price_alert_above_eur: Optional[float] = Field(
        None, ge=0.0,
        description="Alert when certificate price rises above this EUR/tCO2e level",
    )
    repurchase_threshold_pct: float = Field(
        10.0, ge=0.0, le=50.0,
        description="Price drop % trigger for certificate repurchase consideration",
    )
    valuation_method: ValuationMethod = Field(
        ValuationMethod.WEIGHTED_AVERAGE,
        description="Portfolio valuation method for accounting and reporting",
    )
    budget_limit_eur: Optional[Decimal] = Field(
        None, ge=0,
        description="Maximum annual budget for certificate purchases (EUR)",
    )
    auto_purchase_enabled: bool = Field(
        False,
        description="Enable automatic certificate purchases via registry API",
    )
    auto_purchase_max_quantity: Optional[int] = Field(
        None, ge=1,
        description="Maximum certificates per automatic purchase order",
    )
    quarterly_holding_target_pct: float = Field(
        50.0, ge=0.0, le=100.0,
        description="Target holding as % of estimated quarterly obligation",
    )
    dca_frequency_weeks: int = Field(
        4, ge=1, le=52,
        description="Frequency of DCA purchases in weeks (if DCA strategy)",
    )
    dca_fixed_amount_eur: Optional[Decimal] = Field(
        None, ge=0,
        description="Fixed EUR amount per DCA purchase",
    )
    bulk_quarterly_month_offset: int = Field(
        1, ge=0, le=2,
        description="Months after quarter end to execute bulk purchase (0=immediately)",
    )
    opportunistic_price_window_pct: float = Field(
        5.0, ge=1.0, le=20.0,
        description="Price window % for opportunistic buying (buy when price is in bottom N%)",
    )
    mark_to_market_frequency: str = Field(
        "weekly",
        description="Frequency of mark-to-market valuation: daily, weekly, monthly",
    )

    @model_validator(mode="after")
    def validate_dca_config(self) -> "CertificateTradingConfig":
        """Validate DCA-specific fields when DCA strategy is selected."""
        if self.buying_strategy == BuyingStrategy.DCA and self.dca_fixed_amount_eur is None:
            logger.warning("DCA strategy selected but dca_fixed_amount_eur not set; will use equal budget splits")
        return self


class EntityConfig(BaseModel):
    """Configuration for a single entity within a corporate group."""

    entity_id: str = Field(..., description="Unique entity identifier")
    entity_name: str = Field(..., description="Legal name of the entity")
    eori_number: str = Field("", description="Entity EORI number")
    role: EntityRole = Field(EntityRole.SUBSIDIARY, description="Role within group")
    eu_member_state: Optional[EUMemberState] = Field(None, description="EU member state")
    declarant_status: DeclarantStatus = Field(
        DeclarantStatus.NOT_APPLIED, description="CBAM declarant authorization status"
    )
    enabled_categories: List[CBAMGoodsCategory] = Field(
        default_factory=list, description="CBAM categories this entity imports"
    )
    cost_allocation_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Custom cost allocation % (used with CUSTOM allocation method)",
    )
    delegated_to: Optional[str] = Field(
        None, description="Entity ID of the customs representative handling compliance"
    )


class EntityGroupConfig(BaseModel):
    """
    Configuration for multi-entity corporate group CBAM compliance.

    Manages the group hierarchy, cost allocation, consolidated reporting,
    and delegated compliance mode where a customs representative handles
    CBAM obligations on behalf of multiple importers.
    """

    entities: List[EntityConfig] = Field(
        default_factory=list, description="List of entities in the group"
    )
    hierarchy_enabled: bool = Field(
        True, description="Enable parent-subsidiary hierarchy"
    )
    cost_allocation_method: CostAllocationMethod = Field(
        CostAllocationMethod.VOLUME,
        description="Method for allocating shared CBAM costs across entities",
    )
    consolidation_currency: str = Field(
        "EUR", description="Currency for consolidated group reporting"
    )
    delegated_compliance_mode: bool = Field(
        False,
        description="Enable delegated compliance (customs representative mode)",
    )
    financial_guarantee_eur: Optional[Decimal] = Field(
        None, ge=0,
        description="Financial guarantee amount required by competent authority (EUR)",
    )
    inter_entity_transfer_tracking: bool = Field(
        True, description="Track inter-entity transfers of CBAM goods"
    )
    consolidated_declaration: bool = Field(
        False,
        description="Submit single consolidated declaration for the group",
    )

    @model_validator(mode="after")
    def validate_group_structure(self) -> "EntityGroupConfig":
        """Validate group hierarchy consistency."""
        if self.entities:
            parent_count = sum(1 for e in self.entities if e.role == EntityRole.PARENT)
            if self.hierarchy_enabled and parent_count == 0:
                logger.warning("Group hierarchy enabled but no PARENT entity defined")
            if parent_count > 1:
                logger.warning("Multiple PARENT entities found; only one recommended")
        return self


class RegistryAPIConfig(BaseModel):
    """
    Configuration for CBAM Transitional/Definitive Registry API integration.

    Manages connection credentials, polling intervals, retry logic, and
    sandbox mode for testing without affecting production registry data.
    """

    base_url: str = Field(
        "https://cbam-registry.ec.europa.eu/api/v1",
        description="Production CBAM registry API base URL",
    )
    sandbox_url: str = Field(
        "https://cbam-registry-sandbox.ec.europa.eu/api/v1",
        description="Sandbox CBAM registry API URL for testing",
    )
    use_sandbox: bool = Field(
        True, description="Use sandbox API (True) or production (False)"
    )
    auth_type: str = Field(
        "mtls", description="Authentication type: mtls, oauth2, api_key"
    )
    cert_path: str = Field(
        "", description="Path to mTLS client certificate (.pem)"
    )
    key_path: str = Field(
        "", description="Path to mTLS client private key (.pem)"
    )
    ca_bundle_path: str = Field(
        "", description="Path to CA bundle for registry TLS verification"
    )
    polling_interval_seconds: int = Field(
        60, ge=10, le=3600,
        description="Interval for polling registry submission status",
    )
    max_retries: int = Field(
        5, ge=0, le=20,
        description="Maximum retry attempts for failed API calls",
    )
    retry_backoff_factor: float = Field(
        2.0, ge=1.0, le=10.0,
        description="Exponential backoff factor for retries",
    )
    timeout_seconds: int = Field(
        30, ge=5, le=120,
        description="HTTP request timeout in seconds",
    )
    rate_limit_requests_per_minute: int = Field(
        60, ge=1, le=600,
        description="Maximum API requests per minute (rate limiting)",
    )


class AdvancedAnalyticsConfig(BaseModel):
    """
    Configuration for Monte Carlo simulation and advanced analytics.

    Controls simulation parameters, optimization solver, benchmarking,
    and forecasting horizons for strategic CBAM cost planning.
    """

    monte_carlo_iterations: int = Field(
        10000, ge=100, le=1000000,
        description="Number of Monte Carlo simulation iterations",
    )
    optimization_solver: str = Field(
        "scipy_optimize",
        description="Optimization solver: scipy_optimize, cvxpy, or_tools",
    )
    benchmark_source: str = Field(
        "eu_bat",
        description="Benchmark data source: eu_bat, worldsteel, iea",
    )
    forecast_horizon_years: int = Field(
        5, ge=1, le=15,
        description="Number of years for cost/emission forecasting",
    )
    confidence_level: float = Field(
        0.95, ge=0.5, le=0.99,
        description="Confidence level for Monte Carlo confidence intervals",
    )
    scenario_count: int = Field(
        3, ge=1, le=10,
        description="Number of price scenarios to model (low/mid/high by default)",
    )
    sensitivity_variables: List[str] = Field(
        default_factory=lambda: [
            "ets_price", "import_volume", "emission_intensity",
            "scrap_ratio", "free_allocation_pct", "exchange_rate",
        ],
        description="Variables for sensitivity analysis",
    )
    price_volatility_annual_pct: float = Field(
        20.0, ge=1.0, le=100.0,
        description="Assumed annual price volatility for Monte Carlo (% std dev)",
    )


class CustomsAutomationConfig(BaseModel):
    """
    Configuration for customs automation and anti-circumvention monitoring.

    Integrates with TARIC for CN code validation, SAD document parsing,
    AEO status checks, and anti-circumvention pattern detection.
    """

    taric_api_url: str = Field(
        "https://ec.europa.eu/taxation_customs/dds2/taric/consultation",
        description="TARIC consultation API URL",
    )
    sad_format: str = Field(
        "xml", description="Single Administrative Document format: xml, edi"
    )
    aeo_check_enabled: bool = Field(
        True, description="Verify Authorized Economic Operator status"
    )
    anti_circumvention_rules: List[AntiCircumventionRule] = Field(
        default_factory=lambda: list(AntiCircumventionRule),
        description="Active anti-circumvention detection rules",
    )
    downstream_monitoring: bool = Field(
        True,
        description="Monitor downstream processing/re-export of CBAM goods",
    )
    cn_cache_ttl_hours: int = Field(
        24, ge=1, le=168,
        description="TTL for cached TARIC CN code lookups (hours)",
    )
    origin_verification_enabled: bool = Field(
        True,
        description="Enable country-of-origin verification against customs data",
    )
    preferential_origin_check: bool = Field(
        True,
        description="Check for preferential origin claims that may affect CBAM",
    )
    suspicious_pattern_alert_threshold: int = Field(
        3, ge=1, le=10,
        description="Number of suspicious patterns before generating alert",
    )


class CrossRegulationConfig(BaseModel):
    """
    Configuration for cross-regulation data mapping and synchronization.

    Maps CBAM embedded emissions data to CSRD, CDP, SBTi, EU Taxonomy,
    EU ETS, and EUDR frameworks for single-entry compliance.
    """

    csrd_enabled: bool = Field(True, description="Map CBAM data to CSRD E1 disclosures")
    cdp_enabled: bool = Field(True, description="Map CBAM data to CDP Climate questionnaire")
    sbti_enabled: bool = Field(True, description="Map CBAM data to SBTi target tracking")
    taxonomy_enabled: bool = Field(True, description="Map CBAM data to EU Taxonomy screening")
    ets_enabled: bool = Field(True, description="Map CBAM data to EU ETS monitoring")
    eudr_enabled: bool = Field(False, description="Map CBAM supply chain data to EUDR records")
    sync_frequency_hours: int = Field(
        24, ge=1, le=168,
        description="Frequency of cross-regulation data sync (hours)",
    )
    mapping_rules: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "csrd": ["E1-1", "E1-2", "E1-3", "E1-4", "E1-5", "E1-6", "E1-7", "E1-8", "E1-9"],
            "cdp": ["C1.1", "C1.2", "C2.1", "C4.1", "C6.1", "C6.3", "C7.1"],
            "sbti": ["scope_1", "scope_2_location", "scope_2_market", "scope_3_cat1"],
            "taxonomy": ["ccm_1", "ccm_2", "ccm_3"],
            "ets": ["installation_emissions", "verified_emissions"],
            "eudr": ["supply_chain_origin", "geolocation"],
        },
        description="Mapping rules per target regulation",
    )
    auto_populate_enabled: bool = Field(
        True,
        description="Automatically populate target framework data from CBAM",
    )
    conflict_resolution: str = Field(
        "cbam_primary",
        description="Conflict resolution when data differs: cbam_primary, manual_review",
    )


class AuditManagementConfig(BaseModel):
    """
    Configuration for NCA audit readiness and evidence management.

    Manages evidence retention, encrypted data rooms, auto-remediation
    alerts, NCA response deadlines, and penalty risk tracking.
    """

    retention_years: int = Field(
        10, ge=5, le=20,
        description="Evidence retention period in years (CBAM requires minimum 10)",
    )
    data_room_enabled: bool = Field(
        True,
        description="Enable secure data room for NCA audit evidence sharing",
    )
    auto_remediation_alerts: bool = Field(
        True,
        description="Generate automatic alerts for compliance gaps requiring remediation",
    )
    nca_response_deadline_days: int = Field(
        30, ge=7, le=90,
        description="Deadline in days for responding to NCA inquiries",
    )
    penalty_tracking: bool = Field(
        True,
        description="Track potential penalty exposure from non-compliance",
    )
    evidence_encryption: bool = Field(
        True,
        description="Encrypt evidence at rest using AES-256-GCM",
    )
    evidence_encryption_key_rotation_days: int = Field(
        90, ge=30, le=365,
        description="Key rotation period for evidence encryption keys",
    )
    audit_trail_export_formats: List[str] = Field(
        default_factory=lambda: ["json", "csv", "pdf"],
        description="Supported export formats for audit trail",
    )
    data_room_provider: str = Field(
        "internal",
        description="Data room provider: internal, intralinks, ansarada, firmex",
    )
    max_evidence_size_mb: int = Field(
        500, ge=50, le=10000,
        description="Maximum total evidence package size (MB)",
    )


class PrecursorChainConfig(BaseModel):
    """
    Configuration for deep precursor chain analysis.

    Controls chain depth, allocation methods, fallback logic, scrap
    classification, and mass balance validation for multi-tier precursor
    chains (e.g., iron ore -> pig iron -> crude steel -> HRC -> coated flat).
    """

    max_chain_depth: int = Field(
        5, ge=1, le=10,
        description="Maximum precursor chain depth (tiers) to analyze",
    )
    allocation_method: AllocationMethod = Field(
        AllocationMethod.MASS,
        description="Default emission allocation method across precursor tiers",
    )
    default_fallback_waterfall: List[str] = Field(
        default_factory=lambda: [
            "installation_actual",
            "supplier_actual",
            "country_default",
            "eu_default",
            "conservative_estimate",
        ],
        description="Fallback waterfall when actual precursor data is unavailable",
    )
    scrap_classification_enabled: bool = Field(
        True,
        description="Enable scrap vs. primary material classification for EAF routes",
    )
    mass_balance_validation: bool = Field(
        True,
        description="Validate mass balance across precursor chain tiers",
    )
    mass_balance_tolerance_pct: float = Field(
        5.0, ge=0.0, le=20.0,
        description="Tolerance for mass balance validation (% deviation)",
    )
    economic_allocation_price_source: str = Field(
        "lme_average",
        description="Price source for economic allocation: lme_average, contract, spot",
    )
    energy_allocation_unit: str = Field(
        "gj",
        description="Energy unit for energy-based allocation: gj, mwh, kwh",
    )
    precursor_chain_visualization: bool = Field(
        True,
        description="Generate visual precursor chain diagrams",
    )
    chain_completeness_threshold_pct: float = Field(
        80.0, ge=50.0, le=100.0,
        description="Minimum % of chain tiers with actual data to be considered complete",
    )


# =============================================================================
# Main Pack Configuration
# =============================================================================


class CBAMCompleteConfig(BaseModel):
    """
    Root configuration for PACK-005 CBAM Complete Pack.

    Extends PACK-004 CBAMPackConfig with 8 new sub-configuration blocks for
    enterprise CBAM compliance: certificate trading, multi-entity group
    management, registry API, advanced analytics, customs automation,
    cross-regulation, audit management, and precursor chain analysis.

    Example:
        >>> config = CBAMCompleteConfig.from_preset("enterprise_importer")
        >>> print(config.trading.buying_strategy)
        <BuyingStrategy.DCA: 'dca'>

        >>> config = CBAMCompleteConfig.from_yaml("config/presets/steel_group.yaml")
        >>> issues = config.validate_config()
        >>> assert len(issues) == 0
    """

    # Pack identification
    pack_id: str = Field("PACK-005-cbam-complete", description="Pack identifier")
    version: str = Field("1.0.0", description="Pack version")
    extends: str = Field("PACK-004-cbam-readiness", description="Base pack extended by this pack")

    # PACK-004 inherited sub-configurations
    importer: ImporterConfig = Field(default_factory=ImporterConfig, description="Importer identification")
    goods: GoodsCategoryConfig = Field(default_factory=GoodsCategoryConfig, description="Goods categories and CN codes")
    emission: EmissionConfig = Field(default_factory=EmissionConfig, description="Emission calculation methodology")
    certificate: CertificateConfig = Field(default_factory=CertificateConfig, description="Certificate management")
    quarterly: QuarterlyConfig = Field(default_factory=QuarterlyConfig, description="Quarterly reporting")
    supplier: SupplierConfig = Field(default_factory=SupplierConfig, description="Supplier data management")
    deminimis: DeMinimisConfig = Field(default_factory=DeMinimisConfig, description="De minimis monitoring")
    verification: VerificationConfig = Field(default_factory=VerificationConfig, description="Third-party verification")

    # PACK-005 NEW sub-configurations
    trading: CertificateTradingConfig = Field(
        default_factory=CertificateTradingConfig,
        description="Certificate trading and portfolio optimization",
    )
    entity_group: EntityGroupConfig = Field(
        default_factory=EntityGroupConfig,
        description="Multi-entity corporate group management",
    )
    registry: RegistryAPIConfig = Field(
        default_factory=RegistryAPIConfig,
        description="CBAM registry API integration",
    )
    analytics: AdvancedAnalyticsConfig = Field(
        default_factory=AdvancedAnalyticsConfig,
        description="Monte Carlo and advanced analytics",
    )
    customs: CustomsAutomationConfig = Field(
        default_factory=CustomsAutomationConfig,
        description="Customs automation and anti-circumvention",
    )
    cross_regulation: CrossRegulationConfig = Field(
        default_factory=CrossRegulationConfig,
        description="Cross-regulation alignment",
    )
    audit: AuditManagementConfig = Field(
        default_factory=AuditManagementConfig,
        description="NCA audit management",
    )
    precursor: PrecursorChainConfig = Field(
        default_factory=PrecursorChainConfig,
        description="Deep precursor chain analysis",
    )

    # Pack-level settings
    reporting_year: int = Field(2026, ge=2023, le=2040, description="Calendar year for CBAM reporting")
    reporting_period: ReportingPeriod = Field(ReportingPeriod.DEFINITIVE, description="Reporting period phase")
    transitional_mode: bool = Field(False, description="Legacy transitional mode flag")
    demo_mode: bool = Field(False, description="Enable demo mode with sample data")
    log_level: str = Field("INFO", description="Logging level")
    provenance_enabled: bool = Field(True, description="Enable SHA-256 provenance hashing")

    # Class-level constants
    AVAILABLE_PRESETS: ClassVar[List[str]] = [
        "enterprise_importer",
        "customs_broker",
        "steel_group",
        "multi_commodity_group",
    ]

    AVAILABLE_SECTORS: ClassVar[List[str]] = [
        "automotive_oem",
        "construction",
        "chemical_manufacturing",
    ]

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "CBAMCompleteConfig":
        """
        Load CBAMCompleteConfig from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully validated CBAMCompleteConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content fails validation.
        """
        path = Path(yaml_path)
        if not path.is_absolute():
            path = CONFIG_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info("Loading CBAM Complete config from: %s", path)

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Empty configuration file: {path}")

        config = cls.model_validate(raw)
        logger.info(
            "Loaded CBAM Complete config: %d categories, year=%d, %d entities",
            len(config.goods.enabled_categories),
            config.reporting_year,
            len(config.entity_group.entities),
        )
        return config

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        sector_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "CBAMCompleteConfig":
        """
        Load CBAMCompleteConfig from a named preset with optional sector overlay.

        Args:
            preset_name: Name of the pack preset.
            sector_name: Optional sector preset to merge.
            overrides: Optional dict of field overrides applied last.

        Returns:
            Fully validated CBAMCompleteConfig instance.

        Raises:
            ValueError: If the preset name is not recognized.
            FileNotFoundError: If the preset file does not exist.
        """
        if preset_name not in cls.AVAILABLE_PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {cls.AVAILABLE_PRESETS}")

        preset_path = PRESETS_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")

        logger.info("Loading CBAM Complete preset: %s", preset_name)
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        if sector_name:
            if sector_name not in cls.AVAILABLE_SECTORS:
                raise ValueError(f"Unknown sector '{sector_name}'. Available: {cls.AVAILABLE_SECTORS}")
            sector_path = SECTORS_DIR / f"{sector_name}.yaml"
            if not sector_path.exists():
                raise FileNotFoundError(f"Sector file not found: {sector_path}")

            logger.info("Merging sector preset: %s", sector_name)
            with open(sector_path, "r", encoding="utf-8") as f:
                sector_data = yaml.safe_load(f) or {}
            preset_data = cls._deep_merge(preset_data, sector_data)

        if overrides:
            logger.info("Applying %d runtime overrides", len(overrides))
            preset_data = cls._deep_merge(preset_data, overrides)

        env_overrides = cls._load_env_overrides()
        if env_overrides:
            logger.info("Applying %d environment variable overrides", len(env_overrides))
            preset_data = cls._deep_merge(preset_data, env_overrides)

        config = cls.model_validate(preset_data)
        logger.info(
            "Loaded CBAM Complete config from preset '%s': %d categories, year=%d",
            preset_name, len(config.goods.enabled_categories), config.reporting_year,
        )
        return config

    @classmethod
    def from_demo(cls) -> "CBAMCompleteConfig":
        """
        Load the demo configuration (EuroSteel Group).

        Returns:
            CBAMCompleteConfig configured for the demo scenario.
        """
        demo_path = DEMO_DIR / "demo_config.yaml"
        config = cls.from_yaml(demo_path)
        config.demo_mode = True
        return config

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> List[str]:
        """
        Run comprehensive validation on the complete configuration.

        Returns:
            List of validation warning/error messages. Empty list means valid.
        """
        issues: List[str] = []

        # PACK-004 validations
        if self.reporting_year <= 2025 and self.reporting_period == ReportingPeriod.DEFINITIVE:
            issues.append(
                f"reporting_year={self.reporting_year} in transitional period "
                f"but reporting_period=DEFINITIVE"
            )

        if self.reporting_year >= 2026 and self.reporting_period == ReportingPeriod.TRANSITIONAL:
            issues.append(
                f"reporting_year={self.reporting_year} in definitive period "
                f"but reporting_period=TRANSITIONAL"
            )

        if not self.demo_mode:
            if not self.importer.company_name:
                issues.append("importer.company_name is not set")
            if not self.importer.eori_number:
                issues.append("importer.eori_number is not set (required for registry)")

        if not self.goods.enabled_categories:
            issues.append("No goods categories enabled")

        for category in self.goods.enabled_categories:
            cn_codes = self.goods.get_cn_codes_for_category(category)
            if not cn_codes:
                issues.append(f"No CN codes for enabled category: {category.value}")

        if self.supplier.quality_threshold < 50.0:
            issues.append(
                f"supplier.quality_threshold={self.supplier.quality_threshold}% is low; "
                f"recommended >=70%"
            )

        if self.reporting_period == ReportingPeriod.DEFINITIVE:
            if not self.verification.verifier_accreditation_required:
                issues.append("Verifier accreditation not required but definitive period mandates it")

        # PACK-005 validations
        if self.trading.auto_purchase_enabled and not self.registry.cert_path:
            issues.append("Auto-purchase enabled but registry cert_path not configured")

        if self.entity_group.entities:
            total_alloc = sum(
                e.cost_allocation_pct or 0.0
                for e in self.entity_group.entities
            )
            if (
                self.entity_group.cost_allocation_method == CostAllocationMethod.CUSTOM
                and abs(total_alloc - 100.0) > 0.01
            ):
                issues.append(
                    f"CUSTOM cost allocation percentages sum to {total_alloc}%, expected 100%"
                )

        if self.entity_group.delegated_compliance_mode and not self.entity_group.entities:
            issues.append("Delegated compliance mode enabled but no entities configured")

        if not self.registry.use_sandbox and not self.registry.cert_path:
            issues.append("Production registry mode requires cert_path for mTLS")

        if self.precursor.max_chain_depth > 5:
            issues.append(
                f"precursor.max_chain_depth={self.precursor.max_chain_depth} "
                f"exceeds recommended maximum of 5"
            )

        if self.customs.anti_circumvention_rules and not self.customs.origin_verification_enabled:
            issues.append(
                "Anti-circumvention rules active but origin verification disabled"
            )

        if issues:
            for issue in issues:
                logger.warning("Configuration issue: %s", issue)

        return issues

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------

    @property
    def total_cn_codes(self) -> int:
        """Return total number of enabled CN codes."""
        return len(self.goods.get_all_enabled_cn_codes())

    @property
    def entity_count(self) -> int:
        """Return number of entities in the group."""
        return len(self.entity_group.entities)

    @property
    def is_multi_entity(self) -> bool:
        """Return whether this is a multi-entity group deployment."""
        return len(self.entity_group.entities) > 1

    @property
    def active_anti_circumvention_rules(self) -> List[AntiCircumventionRule]:
        """Return list of active anti-circumvention rules."""
        return self.customs.anti_circumvention_rules

    @property
    def cross_regulation_targets(self) -> List[str]:
        """Return list of enabled cross-regulation frameworks."""
        targets: List[str] = []
        if self.cross_regulation.csrd_enabled:
            targets.append("CSRD")
        if self.cross_regulation.cdp_enabled:
            targets.append("CDP")
        if self.cross_regulation.sbti_enabled:
            targets.append("SBTi")
        if self.cross_regulation.taxonomy_enabled:
            targets.append("EU_TAXONOMY")
        if self.cross_regulation.ets_enabled:
            targets.append("EU_ETS")
        if self.cross_regulation.eudr_enabled:
            targets.append("EUDR")
        return targets

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_free_allocation_factor(self, year: Optional[int] = None) -> float:
        """
        Get the EU ETS free allocation percentage for a given year.

        Args:
            year: Calendar year. Defaults to self.reporting_year.

        Returns:
            Free allocation percentage (0.0 to 100.0).
        """
        target_year = year or self.reporting_year
        return FREE_ALLOCATION_PHASEOUT.get(
            target_year, 0.0 if target_year > 2034 else 100.0
        )

    def get_cbam_coverage_factor(self, year: Optional[int] = None) -> float:
        """Get CBAM coverage factor (inverse of free allocation)."""
        return 100.0 - self.get_free_allocation_factor(year)

    def get_eu_default_factor(
        self, category: CBAMGoodsCategory, product_type: str
    ) -> Optional[float]:
        """Look up EU default emission factor."""
        cat_factors = EU_DEFAULT_EMISSION_FACTORS.get(category.value, {})
        return cat_factors.get(product_type)

    def get_country_default_factor(
        self, country_code: str, product_type: str
    ) -> Optional[float]:
        """Look up country-specific default emission factor."""
        country_factors = COUNTRY_DEFAULT_FACTORS.get(country_code, {})
        return country_factors.get(product_type)

    def get_country_carbon_price(self, country_code: str) -> Optional[float]:
        """
        Look up the carbon price for a third country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Carbon price in EUR/tCO2e, or None if country has no recognized scheme.
        """
        scheme = THIRD_COUNTRY_CARBON_PRICING.get(country_code)
        if scheme and scheme.get("recognized"):
            return scheme.get("price_eur_2024")
        return None

    def get_penalty_rate(self, year: Optional[int] = None) -> float:
        """
        Get the CBAM penalty rate for non-surrendered certificates.

        Args:
            year: Calendar year. Defaults to self.reporting_year.

        Returns:
            Penalty rate in EUR/tCO2e.
        """
        target_year = year or self.reporting_year
        rates = CBAM_PENALTIES.get("projected_rates", {})
        return rates.get(target_year, CBAM_PENALTIES.get("base_rate_eur_per_tco2e", 100.0))

    def classify_cn_code(self, cn_code: str) -> Optional[CBAMGoodsCategory]:
        """
        Classify a CN code to its CBAM goods category.

        Args:
            cn_code: Combined Nomenclature code.

        Returns:
            Matching CBAMGoodsCategory, or None if not a CBAM good.
        """
        clean_code = cn_code.replace(" ", "").replace(".", "")
        if clean_code in CN_CODE_TO_CATEGORY:
            return CN_CODE_TO_CATEGORY[clean_code]
        for length in [8, 6, 4]:
            prefix = clean_code[:length]
            if prefix in CN_CODE_TO_CATEGORY:
                return CN_CODE_TO_CATEGORY[prefix]
        return None

    def estimate_certificate_cost(
        self,
        embedded_emissions_tco2e: float,
        year: Optional[int] = None,
        scenario: Optional[CostScenario] = None,
        country_carbon_price_eur: float = 0.0,
    ) -> Dict[str, float]:
        """
        Estimate CBAM certificate cost for a given volume of embedded emissions.

        Args:
            embedded_emissions_tco2e: Total embedded emissions in tCO2e.
            year: Calendar year for pricing.
            scenario: Cost scenario.
            country_carbon_price_eur: Carbon price already paid (EUR/tCO2e).

        Returns:
            Dict with cost breakdown including net cost, deductions, and metadata.
        """
        target_year = year or self.reporting_year
        target_scenario = scenario or self.certificate.cost_scenario
        cbam_coverage = self.get_cbam_coverage_factor(target_year) / 100.0
        gross_obligation = embedded_emissions_tco2e * cbam_coverage
        carbon_deduction_per_tonne = (
            country_carbon_price_eur if self.certificate.carbon_deduction_enabled else 0.0
        )
        scenario_prices = self.certificate.cost_scenario_prices.get(target_scenario.value, {})
        price_per_tco2e = scenario_prices.get(target_year, 80.0)
        net_price = max(0.0, price_per_tco2e - carbon_deduction_per_tonne)
        gross_cost = gross_obligation * price_per_tco2e
        carbon_deduction_total = gross_obligation * carbon_deduction_per_tonne
        net_cost = gross_obligation * net_price

        return {
            "gross_obligation_tco2e": round(gross_obligation, 4),
            "net_obligation_tco2e": round(gross_obligation, 4),
            "price_per_tco2e_eur": round(price_per_tco2e, 2),
            "carbon_deduction_per_tco2e_eur": round(carbon_deduction_per_tonne, 2),
            "gross_cost_eur": round(gross_cost, 2),
            "carbon_deduction_total_eur": round(carbon_deduction_total, 2),
            "net_cost_eur": round(net_cost, 2),
            "free_allocation_pct": round(self.get_free_allocation_factor(target_year), 2),
            "cbam_coverage_pct": round(cbam_coverage * 100, 2),
            "penalty_rate_eur": round(self.get_penalty_rate(target_year), 2),
            "year": target_year,
            "scenario": target_scenario.value,
        }

    def compute_provenance_hash(self) -> str:
        """
        Compute SHA-256 hash of the entire configuration for audit provenance.

        Returns:
            Hex-encoded SHA-256 hash of the JSON-serialized config.
        """
        config_data = self.model_dump(mode="json")
        config_json = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Write the current configuration to a YAML file.

        Args:
            output_path: Path where the YAML file will be written.
        """
        path = Path(output_path)
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info("Configuration written to: %s", path)

    def summary(self) -> Dict[str, Any]:
        """
        Generate a concise summary of the configuration.

        Returns:
            Dict with key configuration parameters for display.
        """
        return {
            "pack_id": self.pack_id,
            "version": self.version,
            "extends": self.extends,
            "reporting_year": self.reporting_year,
            "reporting_period": self.reporting_period.value,
            "importer": self.importer.company_name or "(not configured)",
            "eori": self.importer.eori_number or "(not configured)",
            "member_state": self.importer.eu_member_state.value if self.importer.eu_member_state else "(not set)",
            "enabled_categories": [c.value for c in self.goods.enabled_categories],
            "total_cn_codes": self.total_cn_codes,
            "calculation_method": self.emission.calculation_method.value,
            "ets_price_source": self.certificate.ets_price_source.value,
            "cost_scenario": self.certificate.cost_scenario.value,
            "buying_strategy": self.trading.buying_strategy.value,
            "valuation_method": self.trading.valuation_method.value,
            "entity_count": self.entity_count,
            "is_multi_entity": self.is_multi_entity,
            "cost_allocation": self.entity_group.cost_allocation_method.value,
            "registry_sandbox": self.registry.use_sandbox,
            "monte_carlo_iterations": self.analytics.monte_carlo_iterations,
            "anti_circumvention_rules": len(self.customs.anti_circumvention_rules),
            "cross_regulation_targets": self.cross_regulation_targets,
            "precursor_max_depth": self.precursor.max_chain_depth,
            "audit_retention_years": self.audit.retention_years,
            "demo_mode": self.demo_mode,
            "provenance_hash": self.compute_provenance_hash()[:16] + "...",
        }

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep-merge two dictionaries. Values from overlay take precedence.

        Args:
            base: Base dictionary.
            overlay: Dictionary to merge on top.

        Returns:
            New merged dictionary.
        """
        result = dict(base)
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = CBAMCompleteConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from CBAM_COMPLETE_* environment variables.

        Returns:
            Dict of overrides parsed from environment variables.
        """
        overrides: Dict[str, Any] = {}
        prefix = "CBAM_COMPLETE_"

        env_mapping: Dict[str, Tuple[str, type]] = {
            "REPORTING_YEAR": ("reporting_year", int),
            "DEMO_MODE": ("demo_mode", bool),
            "LOG_LEVEL": ("log_level", str),
            "IMPORTER_COMPANY_NAME": ("importer.company_name", str),
            "IMPORTER_EORI": ("importer.eori_number", str),
            "EMISSION_METHOD": ("emission.calculation_method", str),
            "BUYING_STRATEGY": ("trading.buying_strategy", str),
            "VALUATION_METHOD": ("trading.valuation_method", str),
            "REGISTRY_SANDBOX": ("registry.use_sandbox", bool),
            "MONTE_CARLO_ITERATIONS": ("analytics.monte_carlo_iterations", int),
            "PRECURSOR_MAX_DEPTH": ("precursor.max_chain_depth", int),
            "AUDIT_RETENTION_YEARS": ("audit.retention_years", int),
        }

        for env_suffix, (config_path, value_type) in env_mapping.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        parsed: Any = env_value.lower() in ("true", "1", "yes")
                    elif value_type == int:
                        parsed = int(env_value)
                    elif value_type == float:
                        parsed = float(env_value)
                    else:
                        parsed = env_value

                    parts = config_path.split(".")
                    current = overrides
                    for part in parts[:-1]:
                        current = current.setdefault(part, {})
                    current[parts[-1]] = parsed
                    logger.info("Applied env override: %s = %s", env_var, parsed)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid env override %s=%s: %s", env_var, env_value, e)

        return overrides


# =============================================================================
# Preset Loader Utility
# =============================================================================


def load_preset(
    preset_name: str,
    sector_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CBAMCompleteConfig:
    """
    Convenience function to load a CBAM Complete Pack preset.

    Args:
        preset_name: Name of the pack preset.
        sector_name: Optional sector overlay.
        overrides: Optional runtime overrides.

    Returns:
        Fully validated CBAMCompleteConfig.
    """
    return CBAMCompleteConfig.from_preset(preset_name, sector_name, overrides)


def config_factory(
    source: str = "default",
    yaml_path: Optional[str] = None,
    preset_name: Optional[str] = None,
    sector_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CBAMCompleteConfig:
    """
    Factory function for creating CBAMCompleteConfig from various sources.

    Args:
        source: Config source - "default", "yaml", "preset", or "demo".
        yaml_path: Path to YAML file (when source="yaml").
        preset_name: Preset name (when source="preset").
        sector_name: Optional sector overlay (when source="preset").
        overrides: Optional runtime overrides.

    Returns:
        Fully validated CBAMCompleteConfig.

    Raises:
        ValueError: If source is invalid or required args are missing.
    """
    if source == "default":
        config = CBAMCompleteConfig()
    elif source == "yaml":
        if yaml_path is None:
            raise ValueError("yaml_path required when source='yaml'")
        config = CBAMCompleteConfig.from_yaml(yaml_path)
    elif source == "preset":
        if preset_name is None:
            raise ValueError("preset_name required when source='preset'")
        config = CBAMCompleteConfig.from_preset(preset_name, sector_name)
    elif source == "demo":
        config = CBAMCompleteConfig.from_demo()
    else:
        raise ValueError(f"Unknown config source: {source}. Use: default, yaml, preset, demo")

    if overrides and source != "preset":
        merged = CBAMCompleteConfig._deep_merge(config.model_dump(mode="json"), overrides)
        config = CBAMCompleteConfig.model_validate(merged)

    return config
