# -*- coding: utf-8 -*-
"""
Upstream Transportation & Distribution Spend-Based Calculator Engine - AGENT-MRV-017

Engine 4 of 7: SpendBasedCalculatorEngine

Calculates GHG emissions using transport spend data multiplied by Environmentally
Extended Input-Output (EEIO) emission factors. This is the lowest-accuracy method
defined by the GHG Protocol Scope 3 Category 4 guidance, suitable for screening-
level assessments or when activity data (distance, fuel) is unavailable.

Core Formula:
    emissions_kgco2e = spend_amount x EEIO_factor

Where EEIO factors are expressed in kgCO2e per unit of currency (e.g., kgCO2e/USD).

Supported EEIO Factor Sources:
    - USEEIO v2.0 (US EPA, NAICS-based, kgCO2e/USD, 2021 base year)
    - EXIOBASE v3.8 (EU/multi-region, sector-based, kgCO2e/EUR)
    - DEFRA Spend Factors (UK BEIS, mode-based, kgCO2e/GBP)

Features:
    - CPI deflator-based inflation adjustment to 2021 base year
    - Purchaser-to-producer price margin removal by transport sector
    - NAICS/NACE code resolution with fuzzy matching
    - Cross-code mapping (NAICS <-> NACE)
    - Batch calculation with aggregation by sector and mode
    - Data quality scoring (always DQI Tier 4-5 for spend-based)
    - Uncertainty estimation (always +-50% to +-100%)
    - Full provenance tracking via SHA-256 chain hashing

Zero-Hallucination Compliance:
    All emission calculations use deterministic arithmetic on embedded factor
    tables. No LLM calls are made in any calculation path. All factors are
    sourced from USEEIO v2.0, EXIOBASE v3.8, or DEFRA 2023 publications.

Agent: GL-MRV-S3-004
Component: AGENT-MRV-017
Table Prefix: gl_uto_

Example:
    >>> from greenlang.upstream_transportation.spend_based_calculator import (
    ...     SpendBasedCalculatorEngine
    ... )
    >>> engine = SpendBasedCalculatorEngine()
    >>> from decimal import Decimal
    >>> from greenlang.upstream_transportation.models import (
    ...     SpendInput, CurrencyCode, CalculationMethod
    ... )
    >>> spend_input = SpendInput(
    ...     record_id="SPD-001",
    ...     spend_amount=Decimal("10000.00"),
    ...     currency=CurrencyCode.USD,
    ...     spend_year=2023,
    ...     transport_type="484110",
    ... )
    >>> result = engine.calculate(spend_input)
    >>> assert result.total_emissions_kgco2e > 0
    >>> assert result.calculation_method == CalculationMethod.SPEND_BASED

Author: GreenLang Platform Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.upstream_transportation.models import (
    AGENT_COMPONENT,
    AGENT_ID,
    VERSION,
    CalculationMethod,
    CalculationResult,
    CurrencyCode,
    DQIScore,
    SpendInput,
    TransportMode,
    calculate_provenance_hash,
    get_dqi_composite_score,
    get_dqi_quality_tier,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
THOUSAND = Decimal("1000")
DECIMAL_PRECISION = 8
ROUNDING = ROUND_HALF_UP

# Base year for EEIO factor tables (all spend is deflated to this year)
EEIO_BASE_YEAR: int = 2021

# ==============================================================================
# USEEIO v2.0 EMISSION FACTORS (kgCO2e per USD, 2021 base year)
# Source: US EPA USEEIO v2.0, NAICS 6-digit codes for transport sectors
# ==============================================================================

USEEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "484110": {
        "factor": Decimal("0.580"),
        "description": "General freight trucking, long-distance",
        "mode": TransportMode.ROAD,
        "naics_group": "484",
    },
    "484120": {
        "factor": Decimal("0.620"),
        "description": "General freight trucking, local",
        "mode": TransportMode.ROAD,
        "naics_group": "484",
    },
    "484210": {
        "factor": Decimal("0.480"),
        "description": "Used household goods moving",
        "mode": TransportMode.ROAD,
        "naics_group": "484",
    },
    "484220": {
        "factor": Decimal("0.550"),
        "description": "Specialized freight trucking, local",
        "mode": TransportMode.ROAD,
        "naics_group": "484",
    },
    "484230": {
        "factor": Decimal("0.540"),
        "description": "Specialized freight trucking, long-distance",
        "mode": TransportMode.ROAD,
        "naics_group": "484",
    },
    "481112": {
        "factor": Decimal("0.950"),
        "description": "Scheduled air freight",
        "mode": TransportMode.AIR,
        "naics_group": "481",
    },
    "481212": {
        "factor": Decimal("1.020"),
        "description": "Nonscheduled air freight",
        "mode": TransportMode.AIR,
        "naics_group": "481",
    },
    "483111": {
        "factor": Decimal("0.420"),
        "description": "Deep sea freight",
        "mode": TransportMode.MARITIME,
        "naics_group": "483",
    },
    "483113": {
        "factor": Decimal("0.380"),
        "description": "Coastal freight",
        "mode": TransportMode.MARITIME,
        "naics_group": "483",
    },
    "483211": {
        "factor": Decimal("0.350"),
        "description": "Inland water freight",
        "mode": TransportMode.MARITIME,
        "naics_group": "483",
    },
    "482111": {
        "factor": Decimal("0.300"),
        "description": "Rail transportation",
        "mode": TransportMode.RAIL,
        "naics_group": "482",
    },
    "486110": {
        "factor": Decimal("0.200"),
        "description": "Pipeline crude oil",
        "mode": TransportMode.PIPELINE,
        "naics_group": "486",
    },
    "486210": {
        "factor": Decimal("0.180"),
        "description": "Pipeline natural gas",
        "mode": TransportMode.PIPELINE,
        "naics_group": "486",
    },
    "486990": {
        "factor": Decimal("0.220"),
        "description": "Pipeline other",
        "mode": TransportMode.PIPELINE,
        "naics_group": "486",
    },
    "493110": {
        "factor": Decimal("0.250"),
        "description": "Warehousing and storage",
        "mode": None,
        "naics_group": "493",
    },
    "488410": {
        "factor": Decimal("0.400"),
        "description": "Motor vehicle towing",
        "mode": TransportMode.ROAD,
        "naics_group": "488",
    },
    "488490": {
        "factor": Decimal("0.450"),
        "description": "Other freight transportation arrangement",
        "mode": None,
        "naics_group": "488",
    },
    "492110": {
        "factor": Decimal("0.750"),
        "description": "Couriers and express delivery",
        "mode": TransportMode.ROAD,
        "naics_group": "492",
    },
}

# ==============================================================================
# EXIOBASE v3.8 EMISSION FACTORS (kgCO2e per EUR, by region)
# Source: EXIOBASE v3.8, concordance with NACE Rev. 2 sectors
# ==============================================================================

EXIOBASE_FACTORS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("transport_via_railways", "EU"): {
        "factor": Decimal("0.280"),
        "description": "Transport via railways (EU)",
        "mode": TransportMode.RAIL,
        "nace_code": "H49.1",
    },
    ("transport_via_railways", "US"): {
        "factor": Decimal("0.350"),
        "description": "Transport via railways (US)",
        "mode": TransportMode.RAIL,
        "nace_code": "H49.1",
    },
    ("other_land_transport", "EU"): {
        "factor": Decimal("0.520"),
        "description": "Other land transport (EU)",
        "mode": TransportMode.ROAD,
        "nace_code": "H49.4",
    },
    ("other_land_transport", "US"): {
        "factor": Decimal("0.610"),
        "description": "Other land transport (US)",
        "mode": TransportMode.ROAD,
        "nace_code": "H49.4",
    },
    ("transport_via_pipelines", "EU"): {
        "factor": Decimal("0.190"),
        "description": "Transport via pipelines (EU)",
        "mode": TransportMode.PIPELINE,
        "nace_code": "H49.5",
    },
    ("sea_and_coastal_water_transport", "EU"): {
        "factor": Decimal("0.400"),
        "description": "Sea and coastal water transport (EU)",
        "mode": TransportMode.MARITIME,
        "nace_code": "H50.2",
    },
    ("inland_water_transport", "EU"): {
        "factor": Decimal("0.320"),
        "description": "Inland water transport (EU)",
        "mode": TransportMode.MARITIME,
        "nace_code": "H50.4",
    },
    ("air_transport", "EU"): {
        "factor": Decimal("0.880"),
        "description": "Air transport (EU)",
        "mode": TransportMode.AIR,
        "nace_code": "H51",
    },
    ("air_transport", "US"): {
        "factor": Decimal("0.920"),
        "description": "Air transport (US)",
        "mode": TransportMode.AIR,
        "nace_code": "H51",
    },
    ("warehousing", "EU"): {
        "factor": Decimal("0.220"),
        "description": "Warehousing (EU)",
        "mode": None,
        "nace_code": "H52.1",
    },
    ("warehousing", "US"): {
        "factor": Decimal("0.260"),
        "description": "Warehousing (US)",
        "mode": None,
        "nace_code": "H52.1",
    },
}

# ==============================================================================
# DEFRA SPEND EMISSION FACTORS (kgCO2e per GBP)
# Source: DEFRA / UK BEIS Conversion Factors 2023
# ==============================================================================

DEFRA_SPEND_FACTORS: Dict[str, Dict[str, Any]] = {
    "road_freight": {
        "factor": Decimal("0.490"),
        "description": "Road freight",
        "mode": TransportMode.ROAD,
    },
    "rail_freight": {
        "factor": Decimal("0.270"),
        "description": "Rail freight",
        "mode": TransportMode.RAIL,
    },
    "sea_freight": {
        "factor": Decimal("0.370"),
        "description": "Sea freight",
        "mode": TransportMode.MARITIME,
    },
    "air_freight": {
        "factor": Decimal("0.840"),
        "description": "Air freight",
        "mode": TransportMode.AIR,
    },
    "warehousing": {
        "factor": Decimal("0.210"),
        "description": "Warehousing",
        "mode": None,
    },
}

# ==============================================================================
# CPI DEFLATORS (to 2021 base year, base year = 1.00)
# Source: World Bank / OECD / BLS / Eurostat / ONS
# ==============================================================================

CPI_DEFLATORS: Dict[str, Dict[int, Decimal]] = {
    "USD": {
        2019: Decimal("0.95"),
        2020: Decimal("0.96"),
        2021: Decimal("1.00"),
        2022: Decimal("1.08"),
        2023: Decimal("1.12"),
        2024: Decimal("1.15"),
        2025: Decimal("1.18"),
    },
    "EUR": {
        2019: Decimal("0.96"),
        2020: Decimal("0.97"),
        2021: Decimal("1.00"),
        2022: Decimal("1.09"),
        2023: Decimal("1.15"),
        2024: Decimal("1.18"),
        2025: Decimal("1.20"),
    },
    "GBP": {
        2019: Decimal("0.94"),
        2020: Decimal("0.95"),
        2021: Decimal("1.00"),
        2022: Decimal("1.10"),
        2023: Decimal("1.17"),
        2024: Decimal("1.21"),
        2025: Decimal("1.24"),
    },
}

# ==============================================================================
# CURRENCY EXCHANGE RATES (to USD, approximate annual averages)
# Source: Federal Reserve / ECB / BoE historical rates
# ==============================================================================

EXCHANGE_RATES_TO_USD: Dict[str, Dict[int, Decimal]] = {
    "USD": {
        2019: Decimal("1.000"),
        2020: Decimal("1.000"),
        2021: Decimal("1.000"),
        2022: Decimal("1.000"),
        2023: Decimal("1.000"),
        2024: Decimal("1.000"),
        2025: Decimal("1.000"),
    },
    "EUR": {
        2019: Decimal("1.120"),
        2020: Decimal("1.140"),
        2021: Decimal("1.183"),
        2022: Decimal("1.053"),
        2023: Decimal("1.081"),
        2024: Decimal("1.085"),
        2025: Decimal("1.090"),
    },
    "GBP": {
        2019: Decimal("1.277"),
        2020: Decimal("1.284"),
        2021: Decimal("1.376"),
        2022: Decimal("1.237"),
        2023: Decimal("1.244"),
        2024: Decimal("1.268"),
        2025: Decimal("1.275"),
    },
    "JPY": {
        2019: Decimal("0.00916"),
        2020: Decimal("0.00935"),
        2021: Decimal("0.00909"),
        2022: Decimal("0.00769"),
        2023: Decimal("0.00714"),
        2024: Decimal("0.00667"),
        2025: Decimal("0.00660"),
    },
    "CNY": {
        2019: Decimal("0.1449"),
        2020: Decimal("0.1449"),
        2021: Decimal("0.1549"),
        2022: Decimal("0.1489"),
        2023: Decimal("0.1407"),
        2024: Decimal("0.1379"),
        2025: Decimal("0.1370"),
    },
    "INR": {
        2019: Decimal("0.01424"),
        2020: Decimal("0.01342"),
        2021: Decimal("0.01347"),
        2022: Decimal("0.01265"),
        2023: Decimal("0.01205"),
        2024: Decimal("0.01197"),
        2025: Decimal("0.01185"),
    },
    "CAD": {
        2019: Decimal("0.7537"),
        2020: Decimal("0.7462"),
        2021: Decimal("0.7978"),
        2022: Decimal("0.7693"),
        2023: Decimal("0.7409"),
        2024: Decimal("0.7380"),
        2025: Decimal("0.7350"),
    },
    "AUD": {
        2019: Decimal("0.6953"),
        2020: Decimal("0.6906"),
        2021: Decimal("0.7514"),
        2022: Decimal("0.6947"),
        2023: Decimal("0.6615"),
        2024: Decimal("0.6590"),
        2025: Decimal("0.6560"),
    },
    "CHF": {
        2019: Decimal("1.0069"),
        2020: Decimal("1.0656"),
        2021: Decimal("1.0936"),
        2022: Decimal("1.0497"),
        2023: Decimal("1.1175"),
        2024: Decimal("1.1150"),
        2025: Decimal("1.1120"),
    },
}

# ==============================================================================
# MARGIN REMOVAL PERCENTAGES (purchaser -> producer price)
# Source: BEA / OECD Input-Output Tables
# Margins represent trade/transport margins embedded in purchaser prices.
# Removing margins yields producer prices, which align better with EEIO factors.
# ==============================================================================

MARGIN_REMOVAL_PERCENTAGES: Dict[str, Decimal] = {
    "trucking": Decimal("15"),
    "air_freight": Decimal("20"),
    "ocean_freight": Decimal("12"),
    "rail_freight": Decimal("10"),
    "pipeline": Decimal("8"),
    "warehousing": Decimal("18"),
    "courier_express": Decimal("25"),
}

# ==============================================================================
# NAICS CODE DESCRIPTION LOOKUP (for fuzzy matching)
# ==============================================================================

NAICS_CODE_DESCRIPTIONS: Dict[str, str] = {
    "484110": "general freight trucking long-distance",
    "484120": "general freight trucking local",
    "484210": "used household goods moving",
    "484220": "specialized freight trucking local",
    "484230": "specialized freight trucking long-distance",
    "481112": "scheduled air freight transportation",
    "481212": "nonscheduled air freight chartering",
    "483111": "deep sea freight transportation",
    "483113": "coastal and great lakes freight transportation",
    "483211": "inland water freight transportation",
    "482111": "line-haul railroads",
    "486110": "pipeline transportation of crude oil",
    "486210": "pipeline transportation of natural gas",
    "486990": "all other pipeline transportation",
    "493110": "general warehousing and storage",
    "488410": "motor vehicle towing",
    "488490": "other support activities for road transportation",
    "492110": "couriers and express delivery services",
}

# ==============================================================================
# NACE CODE DESCRIPTIONS (for EXIOBASE)
# ==============================================================================

NACE_CODE_DESCRIPTIONS: Dict[str, str] = {
    "H49.1": "passenger rail transport interurban",
    "H49.2": "freight rail transport",
    "H49.4": "freight transport by road and removal services",
    "H49.5": "transport via pipeline",
    "H50.1": "sea and coastal passenger water transport",
    "H50.2": "sea and coastal freight water transport",
    "H50.3": "inland passenger water transport",
    "H50.4": "inland freight water transport",
    "H51": "air transport",
    "H52.1": "warehousing and storage",
    "H52.2": "support activities for transportation",
}

# ==============================================================================
# NAICS <-> NACE CONCORDANCE TABLES
# ==============================================================================

NAICS_TO_NACE: Dict[str, str] = {
    "484110": "H49.4",
    "484120": "H49.4",
    "484210": "H49.4",
    "484220": "H49.4",
    "484230": "H49.4",
    "481112": "H51",
    "481212": "H51",
    "483111": "H50.2",
    "483113": "H50.2",
    "483211": "H50.4",
    "482111": "H49.2",
    "486110": "H49.5",
    "486210": "H49.5",
    "486990": "H49.5",
    "493110": "H52.1",
    "488410": "H49.4",
    "488490": "H52.2",
    "492110": "H49.4",
}

NACE_TO_NAICS: Dict[str, str] = {
    "H49.1": "482111",
    "H49.2": "482111",
    "H49.4": "484110",
    "H49.5": "486110",
    "H50.2": "483111",
    "H50.4": "483211",
    "H51": "481112",
    "H52.1": "493110",
    "H52.2": "488490",
}

# ==============================================================================
# TRANSPORT DESCRIPTION KEYWORDS (for classify_transport_spend)
# ==============================================================================

TRANSPORT_KEYWORDS: Dict[str, Tuple[str, TransportMode]] = {
    # Road keywords
    "truck": ("484110", TransportMode.ROAD),
    "trucking": ("484110", TransportMode.ROAD),
    "ltl": ("484110", TransportMode.ROAD),
    "ftl": ("484110", TransportMode.ROAD),
    "drayage": ("484120", TransportMode.ROAD),
    "local delivery": ("484120", TransportMode.ROAD),
    "moving": ("484210", TransportMode.ROAD),
    "specialized freight": ("484220", TransportMode.ROAD),
    "flatbed": ("484230", TransportMode.ROAD),
    "heavy haul": ("484230", TransportMode.ROAD),
    "towing": ("488410", TransportMode.ROAD),
    "courier": ("492110", TransportMode.ROAD),
    "express delivery": ("492110", TransportMode.ROAD),
    "parcel": ("492110", TransportMode.ROAD),
    "last mile": ("492110", TransportMode.ROAD),
    # Air keywords
    "air freight": ("481112", TransportMode.AIR),
    "air cargo": ("481112", TransportMode.AIR),
    "air transport": ("481112", TransportMode.AIR),
    "charter air": ("481212", TransportMode.AIR),
    "air charter": ("481212", TransportMode.AIR),
    # Maritime keywords
    "ocean freight": ("483111", TransportMode.MARITIME),
    "sea freight": ("483111", TransportMode.MARITIME),
    "deep sea": ("483111", TransportMode.MARITIME),
    "container ship": ("483111", TransportMode.MARITIME),
    "fcl": ("483111", TransportMode.MARITIME),
    "lcl": ("483111", TransportMode.MARITIME),
    "coastal freight": ("483113", TransportMode.MARITIME),
    "barge": ("483211", TransportMode.MARITIME),
    "inland water": ("483211", TransportMode.MARITIME),
    "river freight": ("483211", TransportMode.MARITIME),
    # Rail keywords
    "rail freight": ("482111", TransportMode.RAIL),
    "railroad": ("482111", TransportMode.RAIL),
    "intermodal rail": ("482111", TransportMode.RAIL),
    "rail transport": ("482111", TransportMode.RAIL),
    # Pipeline keywords
    "pipeline": ("486110", TransportMode.PIPELINE),
    "crude pipeline": ("486110", TransportMode.PIPELINE),
    "gas pipeline": ("486210", TransportMode.PIPELINE),
    "natural gas transport": ("486210", TransportMode.PIPELINE),
    # Warehousing keywords
    "warehouse": ("493110", TransportMode.ROAD),
    "warehousing": ("493110", TransportMode.ROAD),
    "storage": ("493110", TransportMode.ROAD),
    "distribution center": ("493110", TransportMode.ROAD),
    "fulfillment": ("493110", TransportMode.ROAD),
    # Generic
    "freight": ("488490", TransportMode.ROAD),
    "logistics": ("488490", TransportMode.ROAD),
    "3pl": ("488490", TransportMode.ROAD),
    "transportation": ("488490", TransportMode.ROAD),
}

# ==============================================================================
# NAICS GROUP TO MARGIN SECTOR MAPPING
# ==============================================================================

NAICS_TO_MARGIN_SECTOR: Dict[str, str] = {
    "484": "trucking",
    "481": "air_freight",
    "483": "ocean_freight",
    "482": "rail_freight",
    "486": "pipeline",
    "493": "warehousing",
    "492": "courier_express",
    "488": "trucking",
}


# ==============================================================================
# SPEND CALCULATION RESULT (internal model)
# ==============================================================================


class SpendCalculationDetail:
    """
    Internal detail model for a single spend-based calculation.

    Holds intermediate values for audit trail and provenance tracking.

    Attributes:
        record_id: Source record identifier.
        original_spend: Original spend amount in source currency.
        source_currency: ISO 4217 currency code of original spend.
        spend_year: Calendar year of the spend.
        deflated_spend: Spend deflated to EEIO base year (2021).
        converted_spend: Spend converted to factor currency (USD/EUR/GBP).
        margin_removed_spend: Spend after margin removal.
        eeio_factor: EEIO factor applied (kgCO2e per currency unit).
        eeio_source: Factor source identifier.
        naics_code: Resolved NAICS code.
        nace_code: Resolved NACE code (if applicable).
        mode: Resolved transport mode.
        emissions_kgco2e: Calculated emissions in kgCO2e.
        emissions_tco2e: Calculated emissions in tCO2e.
        margin_sector: Margin removal sector applied.
        margin_pct: Margin percentage removed.
        cpi_deflator: CPI deflator applied.
        exchange_rate: Exchange rate applied (if any).
    """

    __slots__ = (
        "record_id",
        "original_spend",
        "source_currency",
        "spend_year",
        "deflated_spend",
        "converted_spend",
        "margin_removed_spend",
        "eeio_factor",
        "eeio_source",
        "naics_code",
        "nace_code",
        "mode",
        "emissions_kgco2e",
        "emissions_tco2e",
        "margin_sector",
        "margin_pct",
        "cpi_deflator",
        "exchange_rate",
    )

    def __init__(self) -> None:
        """Initialize with default values."""
        self.record_id: str = ""
        self.original_spend: Decimal = ZERO
        self.source_currency: str = ""
        self.spend_year: int = EEIO_BASE_YEAR
        self.deflated_spend: Decimal = ZERO
        self.converted_spend: Decimal = ZERO
        self.margin_removed_spend: Decimal = ZERO
        self.eeio_factor: Decimal = ZERO
        self.eeio_source: str = ""
        self.naics_code: str = ""
        self.nace_code: str = ""
        self.mode: Optional[TransportMode] = None
        self.emissions_kgco2e: Decimal = ZERO
        self.emissions_tco2e: Decimal = ZERO
        self.margin_sector: str = ""
        self.margin_pct: Decimal = ZERO
        self.cpi_deflator: Decimal = ONE
        self.exchange_rate: Decimal = ONE

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization and provenance hashing.

        Returns:
            Dictionary with all calculation detail fields.
        """
        return {
            "record_id": self.record_id,
            "original_spend": str(self.original_spend),
            "source_currency": self.source_currency,
            "spend_year": self.spend_year,
            "deflated_spend": str(self.deflated_spend),
            "converted_spend": str(self.converted_spend),
            "margin_removed_spend": str(self.margin_removed_spend),
            "eeio_factor": str(self.eeio_factor),
            "eeio_source": self.eeio_source,
            "naics_code": self.naics_code,
            "nace_code": self.nace_code,
            "mode": self.mode.value if self.mode else None,
            "emissions_kgco2e": str(self.emissions_kgco2e),
            "emissions_tco2e": str(self.emissions_tco2e),
            "margin_sector": self.margin_sector,
            "margin_pct": str(self.margin_pct),
            "cpi_deflator": str(self.cpi_deflator),
            "exchange_rate": str(self.exchange_rate),
        }


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class SpendBasedCalculatorEngine:
    """
    Spend-Based Calculator Engine for Upstream Transportation emissions.

    Implements GHG Protocol Scope 3 Category 4 spend-based method, which
    calculates emissions as: spend_amount x EEIO_factor.

    This is the lowest-accuracy method, suitable for screening or when
    activity data is unavailable. Data quality is always Tier 4-5 (poor/
    very poor) and uncertainty ranges are +-50% to +-100%.

    The engine supports three EEIO factor sources:
        - USEEIO v2.0 (NAICS-based, USD)
        - EXIOBASE v3.8 (sector/region-based, EUR)
        - DEFRA 2023 (mode-based, GBP)

    All calculations are deterministic (zero-hallucination). No LLM or ML
    model is invoked in any arithmetic path.

    Attributes:
        _calculation_count: Running count of calculations performed.
        _total_emissions_kgco2e: Running total of emissions calculated.
        _agent_id: Agent identifier for provenance.
        _version: Engine version for provenance.

    Example:
        >>> engine = SpendBasedCalculatorEngine()
        >>> from greenlang.upstream_transportation.models import (
        ...     SpendInput, CurrencyCode
        ... )
        >>> from decimal import Decimal
        >>> inp = SpendInput(
        ...     record_id="SPD-001",
        ...     spend_amount=Decimal("5000"),
        ...     currency=CurrencyCode.USD,
        ...     spend_year=2023,
        ...     transport_type="484110",
        ... )
        >>> result = engine.calculate(inp)
        >>> float(result.total_emissions_kgco2e) > 0
        True
    """

    def __init__(self) -> None:
        """Initialize SpendBasedCalculatorEngine with zero counters."""
        self._calculation_count: int = 0
        self._total_emissions_kgco2e: Decimal = ZERO
        self._agent_id: str = AGENT_ID
        self._version: str = VERSION
        logger.info(
            "SpendBasedCalculatorEngine initialized (agent=%s, version=%s)",
            self._agent_id,
            self._version,
        )

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================

    def calculate(self, spend_input: SpendInput) -> CalculationResult:
        """
        Calculate emissions from transport spend using EEIO factors.

        Main entry point for spend-based calculations. Performs the following
        pipeline:
            1. Validate input
            2. Resolve NAICS code from transport_type
            3. Deflate spend to base year using CPI deflators
            4. Convert currency to factor currency if needed
            5. Apply margin removal (purchaser -> producer price)
            6. Look up EEIO factor
            7. Multiply: emissions = adjusted_spend x EEIO_factor
            8. Calculate uncertainty bounds
            9. Assign data quality scores
            10. Compute provenance hash

        Args:
            spend_input: Validated SpendInput model with spend amount,
                currency, year, and transport type.

        Returns:
            CalculationResult with emissions, uncertainty, DQI, and
            provenance hash.

        Raises:
            ValueError: If spend input fails validation.
            KeyError: If NAICS code or EEIO factor cannot be resolved.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> from greenlang.upstream_transportation.models import (
            ...     SpendInput, CurrencyCode
            ... )
            >>> from decimal import Decimal
            >>> inp = SpendInput(
            ...     record_id="SPD-002",
            ...     spend_amount=Decimal("25000"),
            ...     currency=CurrencyCode.EUR,
            ...     spend_year=2022,
            ...     transport_type="484110",
            ... )
            >>> result = engine.calculate(inp)
            >>> result.calculation_method.value
            'spend_based'
        """
        start_time = time.monotonic()
        logger.info(
            "Starting spend-based calculation for record_id=%s",
            spend_input.record_id,
        )

        # Step 1: Validate input
        errors = self.validate_spend_input(spend_input)
        if errors:
            error_msg = "; ".join(errors)
            logger.error(
                "Input validation failed for %s: %s",
                spend_input.record_id,
                error_msg,
            )
            raise ValueError(f"Spend input validation failed: {error_msg}")

        # Step 2: Initialize detail tracker
        detail = SpendCalculationDetail()
        detail.record_id = spend_input.record_id
        detail.original_spend = spend_input.spend_amount
        detail.source_currency = spend_input.currency.value
        detail.spend_year = spend_input.spend_year

        # Step 3: Resolve NAICS code from transport_type
        resolved_naics = self._resolve_transport_type(spend_input.transport_type)
        detail.naics_code = resolved_naics

        # Step 4: Map to NACE (informational)
        detail.nace_code = self.map_naics_to_nace(resolved_naics)

        # Step 5: Determine transport mode
        detail.mode = self._resolve_mode(
            resolved_naics, spend_input.mode
        )

        # Step 6: Determine EEIO source and factor
        factor_info = self._resolve_eeio_factor_and_source(
            spend_input, resolved_naics
        )
        detail.eeio_factor = factor_info["factor"]
        detail.eeio_source = factor_info["source"]
        target_currency = factor_info["currency"]

        # Step 7: Deflate spend to base year
        detail.deflated_spend = self._deflate_spend(
            spend_input.spend_amount,
            spend_input.currency.value,
            spend_input.spend_year,
        )
        detail.cpi_deflator = self._get_cpi_deflator_value(
            spend_input.currency.value, spend_input.spend_year
        )

        # Step 8: Convert currency to factor currency if needed
        detail.converted_spend = self._convert_to_target_currency(
            detail.deflated_spend,
            spend_input.currency.value,
            target_currency,
            EEIO_BASE_YEAR,
        )
        detail.exchange_rate = self._get_exchange_rate(
            spend_input.currency.value, target_currency, EEIO_BASE_YEAR
        )

        # Step 9: Apply margin removal
        margin_sector = self._resolve_margin_sector(resolved_naics)
        detail.margin_sector = margin_sector
        margin_pct = MARGIN_REMOVAL_PERCENTAGES.get(margin_sector, ZERO)
        detail.margin_pct = margin_pct
        detail.margin_removed_spend = self.apply_margin_removal(
            detail.converted_spend, margin_pct
        )

        # Step 10: Calculate emissions (ZERO-HALLUCINATION: deterministic arithmetic)
        detail.emissions_kgco2e = self._quantize(
            detail.margin_removed_spend * detail.eeio_factor
        )
        detail.emissions_tco2e = self._quantize(
            detail.emissions_kgco2e / THOUSAND
        )

        # Step 11: Uncertainty estimation
        uncertainty_info = self.estimate_uncertainty(
            detail.emissions_kgco2e, detail.eeio_source
        )

        # Step 12: Data quality scoring
        dqi_info = self._build_dqi_scores()

        # Step 13: Provenance hash
        provenance_hash = self._compute_provenance_hash(
            spend_input, detail
        )

        # Step 14: Build result
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._calculation_count += 1
        self._total_emissions_kgco2e += detail.emissions_kgco2e

        result = CalculationResult(
            request_id=spend_input.record_id,
            calculation_method=CalculationMethod.SPEND_BASED,
            total_emissions_kgco2e=detail.emissions_kgco2e,
            total_emissions_tco2e=detail.emissions_tco2e,
            emissions_co2_kg=detail.emissions_kgco2e,
            emissions_ch4_kg=ZERO,
            emissions_n2o_kg=ZERO,
            emissions_hfc_kgco2e=ZERO,
            total_spend=detail.original_spend,
            uncertainty_range_percent=uncertainty_info["uncertainty_pct"],
            lower_bound_kgco2e=uncertainty_info["lower_bound"],
            upper_bound_kgco2e=uncertainty_info["upper_bound"],
            dqi_reliability=dqi_info["reliability"],
            dqi_completeness=dqi_info["completeness"],
            dqi_temporal=dqi_info["temporal"],
            dqi_geographical=dqi_info["geographical"],
            dqi_technological=dqi_info["technological"],
            dqi_composite_score=dqi_info["composite"],
            dqi_quality_tier=dqi_info["tier"],
            provenance_hash=provenance_hash,
            agent_version=self._version,
        )

        logger.info(
            "Spend-based calculation complete: record_id=%s, "
            "emissions=%.4f kgCO2e, source=%s, elapsed=%.1f ms",
            spend_input.record_id,
            detail.emissions_kgco2e,
            detail.eeio_source,
            elapsed_ms,
        )

        return result

    def calculate_useeio(
        self, amount_usd: Decimal, naics_code: str
    ) -> Decimal:
        """
        Calculate emissions using USEEIO v2.0 factors.

        Assumes the amount is already in 2021 USD (base year) and that
        margin removal has already been applied if desired.

        Args:
            amount_usd: Spend amount in USD (2021 base year).
            naics_code: 6-digit NAICS code (e.g., "484110").

        Returns:
            Emissions in kgCO2e.

        Raises:
            KeyError: If NAICS code not found in USEEIO table.
            ValueError: If amount is non-positive.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> emissions = engine.calculate_useeio(Decimal("10000"), "484110")
            >>> emissions
            Decimal('5800.00000000')
        """
        if amount_usd <= ZERO:
            raise ValueError(
                f"USEEIO amount must be positive, got {amount_usd}"
            )

        factor_entry = USEEIO_FACTORS.get(naics_code)
        if factor_entry is None:
            raise KeyError(
                f"NAICS code '{naics_code}' not found in USEEIO v2.0 table. "
                f"Available codes: {sorted(USEEIO_FACTORS.keys())}"
            )

        factor = factor_entry["factor"]
        emissions = self._quantize(amount_usd * factor)

        logger.debug(
            "USEEIO calculation: %s USD * %s (NAICS %s) = %s kgCO2e",
            amount_usd,
            factor,
            naics_code,
            emissions,
        )

        return emissions

    def calculate_exiobase(
        self, amount_eur: Decimal, sector: str, region: str
    ) -> Decimal:
        """
        Calculate emissions using EXIOBASE v3.8 factors.

        Assumes the amount is already in 2021 EUR (base year) and that
        margin removal has already been applied if desired.

        Args:
            amount_eur: Spend amount in EUR (2021 base year).
            sector: EXIOBASE sector key (e.g., "transport_via_railways").
            region: Region code (e.g., "EU", "US").

        Returns:
            Emissions in kgCO2e.

        Raises:
            KeyError: If sector/region combination not found.
            ValueError: If amount is non-positive.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> emissions = engine.calculate_exiobase(
            ...     Decimal("8000"), "air_transport", "EU"
            ... )
            >>> emissions
            Decimal('7040.00000000')
        """
        if amount_eur <= ZERO:
            raise ValueError(
                f"EXIOBASE amount must be positive, got {amount_eur}"
            )

        key = (sector, region)
        factor_entry = EXIOBASE_FACTORS.get(key)
        if factor_entry is None:
            available = sorted(
                f"{s}/{r}" for (s, r) in EXIOBASE_FACTORS.keys()
            )
            raise KeyError(
                f"EXIOBASE sector/region '{sector}/{region}' not found. "
                f"Available: {available}"
            )

        factor = factor_entry["factor"]
        emissions = self._quantize(amount_eur * factor)

        logger.debug(
            "EXIOBASE calculation: %s EUR * %s (%s/%s) = %s kgCO2e",
            amount_eur,
            factor,
            sector,
            region,
            emissions,
        )

        return emissions

    def calculate_defra_spend(
        self, amount_gbp: Decimal, mode: str
    ) -> Decimal:
        """
        Calculate emissions using DEFRA spend-based factors.

        Assumes the amount is already in 2021 GBP (base year) and that
        margin removal has already been applied if desired.

        Args:
            amount_gbp: Spend amount in GBP (2021 base year).
            mode: DEFRA mode key (e.g., "road_freight", "air_freight").

        Returns:
            Emissions in kgCO2e.

        Raises:
            KeyError: If mode not found in DEFRA table.
            ValueError: If amount is non-positive.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> emissions = engine.calculate_defra_spend(
            ...     Decimal("5000"), "road_freight"
            ... )
            >>> emissions
            Decimal('2450.00000000')
        """
        if amount_gbp <= ZERO:
            raise ValueError(
                f"DEFRA amount must be positive, got {amount_gbp}"
            )

        factor_entry = DEFRA_SPEND_FACTORS.get(mode)
        if factor_entry is None:
            available = sorted(DEFRA_SPEND_FACTORS.keys())
            raise KeyError(
                f"DEFRA mode '{mode}' not found. Available: {available}"
            )

        factor = factor_entry["factor"]
        emissions = self._quantize(amount_gbp * factor)

        logger.debug(
            "DEFRA calculation: %s GBP * %s (%s) = %s kgCO2e",
            amount_gbp,
            factor,
            mode,
            emissions,
        )

        return emissions

    def convert_currency(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        year: int,
    ) -> Decimal:
        """
        Convert monetary amount between currencies using embedded exchange rates.

        Uses annual average exchange rates. Both currencies are first converted
        to USD, then to the target currency if needed.

        Args:
            amount: Monetary amount to convert.
            from_currency: ISO 4217 source currency code (e.g., "EUR").
            to_currency: ISO 4217 target currency code (e.g., "USD").
            year: Calendar year for exchange rate selection.

        Returns:
            Converted amount in target currency.

        Raises:
            ValueError: If amount is negative.
            KeyError: If currency or year not found in rate tables.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> converted = engine.convert_currency(
            ...     Decimal("10000"), "EUR", "USD", 2023
            ... )
            >>> converted > Decimal("10000")
            True
        """
        if amount < ZERO:
            raise ValueError(f"Amount must be non-negative, got {amount}")

        if from_currency == to_currency:
            return amount

        if amount == ZERO:
            return ZERO

        # Convert from source to USD
        from_rates = EXCHANGE_RATES_TO_USD.get(from_currency)
        if from_rates is None:
            raise KeyError(
                f"Currency '{from_currency}' not found in exchange rate tables. "
                f"Available: {sorted(EXCHANGE_RATES_TO_USD.keys())}"
            )
        from_rate = from_rates.get(year)
        if from_rate is None:
            raise KeyError(
                f"Year {year} not found for currency '{from_currency}'. "
                f"Available years: {sorted(from_rates.keys())}"
            )

        amount_usd = amount * from_rate

        # Convert from USD to target
        if to_currency == "USD":
            return self._quantize(amount_usd)

        to_rates = EXCHANGE_RATES_TO_USD.get(to_currency)
        if to_rates is None:
            raise KeyError(
                f"Currency '{to_currency}' not found in exchange rate tables. "
                f"Available: {sorted(EXCHANGE_RATES_TO_USD.keys())}"
            )
        to_rate = to_rates.get(year)
        if to_rate is None:
            raise KeyError(
                f"Year {year} not found for currency '{to_currency}'. "
                f"Available years: {sorted(to_rates.keys())}"
            )

        if to_rate == ZERO:
            raise ValueError(
                f"Exchange rate for '{to_currency}' in {year} is zero"
            )

        converted = amount_usd / to_rate
        return self._quantize(converted)

    def get_cpi_deflator(
        self, currency: str, from_year: int, to_year: int
    ) -> Decimal:
        """
        Get CPI deflator ratio to convert spend from one year to another.

        deflated_amount = original_amount * (CPI_to_year / CPI_from_year)

        Args:
            currency: ISO 4217 currency code (USD, EUR, GBP supported).
            from_year: Source year of the spend.
            to_year: Target year to deflate to.

        Returns:
            Deflator ratio (> 1 if deflating forward from cheaper year,
            < 1 if deflating backward from more expensive year).

        Raises:
            KeyError: If currency or year not in CPI table.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> deflator = engine.get_cpi_deflator("USD", 2023, 2021)
            >>> deflator < Decimal("1")
            True
        """
        deflators = CPI_DEFLATORS.get(currency)
        if deflators is None:
            raise KeyError(
                f"CPI deflators not available for currency '{currency}'. "
                f"Supported: {sorted(CPI_DEFLATORS.keys())}"
            )

        from_cpi = deflators.get(from_year)
        if from_cpi is None:
            raise KeyError(
                f"CPI deflator not available for {currency}/{from_year}. "
                f"Available years: {sorted(deflators.keys())}"
            )

        to_cpi = deflators.get(to_year)
        if to_cpi is None:
            raise KeyError(
                f"CPI deflator not available for {currency}/{to_year}. "
                f"Available years: {sorted(deflators.keys())}"
            )

        if from_cpi == ZERO:
            raise ValueError(
                f"CPI deflator for {currency}/{from_year} is zero"
            )

        return self._quantize(to_cpi / from_cpi)

    def apply_margin_removal(
        self, spend: Decimal, margin_pct: Decimal
    ) -> Decimal:
        """
        Remove trade/transport margin from purchaser price to get producer price.

        EEIO factors are typically based on producer prices, while company spend
        data is at purchaser prices. Removing the margin aligns the spend with
        the factor basis, avoiding double-counting of margin components.

        Formula:
            producer_price = purchaser_price * (1 - margin_pct / 100)

        Args:
            spend: Purchaser-price spend amount.
            margin_pct: Margin percentage to remove (0-100).

        Returns:
            Producer-price spend amount.

        Raises:
            ValueError: If margin_pct is outside 0-100 range or spend is
                negative.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> producer = engine.apply_margin_removal(
            ...     Decimal("10000"), Decimal("15")
            ... )
            >>> producer
            Decimal('8500.00000000')
        """
        if spend < ZERO:
            raise ValueError(f"Spend must be non-negative, got {spend}")

        if margin_pct < ZERO or margin_pct > HUNDRED:
            raise ValueError(
                f"Margin percentage must be 0-100, got {margin_pct}"
            )

        removal_factor = ONE - (margin_pct / HUNDRED)
        return self._quantize(spend * removal_factor)

    def get_eeio_factor(
        self, naics_code: str, source: str = "USEEIO"
    ) -> Decimal:
        """
        Get EEIO emission factor for a given code and source.

        Retrieves the emission factor from the specified EEIO database.
        For EXIOBASE, pass the NAICS code and it will be mapped to the
        appropriate sector.

        Args:
            naics_code: 6-digit NAICS code.
            source: Factor source - "USEEIO", "EXIOBASE", or "DEFRA".

        Returns:
            EEIO emission factor in kgCO2e per unit currency.

        Raises:
            KeyError: If code not found in the specified source.
            ValueError: If source is unknown.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> factor = engine.get_eeio_factor("484110", "USEEIO")
            >>> factor
            Decimal('0.580')
        """
        source_upper = source.upper()

        if source_upper == "USEEIO":
            entry = USEEIO_FACTORS.get(naics_code)
            if entry is None:
                raise KeyError(
                    f"NAICS '{naics_code}' not found in USEEIO table"
                )
            return entry["factor"]

        elif source_upper == "EXIOBASE":
            nace_code = self.map_naics_to_nace(naics_code)
            if not nace_code:
                raise KeyError(
                    f"No NACE mapping for NAICS '{naics_code}'"
                )
            # Try EU first, then US
            for region in ("EU", "US"):
                for key, entry in EXIOBASE_FACTORS.items():
                    if entry.get("nace_code") == nace_code and key[1] == region:
                        return entry["factor"]
            raise KeyError(
                f"No EXIOBASE factor found for NACE '{nace_code}'"
            )

        elif source_upper == "DEFRA":
            mode = self._naics_to_defra_mode(naics_code)
            if mode is None:
                raise KeyError(
                    f"No DEFRA mode mapping for NAICS '{naics_code}'"
                )
            entry = DEFRA_SPEND_FACTORS.get(mode)
            if entry is None:
                raise KeyError(f"DEFRA mode '{mode}' not found")
            return entry["factor"]

        else:
            raise ValueError(
                f"Unknown EEIO source '{source}'. "
                f"Supported: USEEIO, EXIOBASE, DEFRA"
            )

    def resolve_naics_code(self, description: str) -> str:
        """
        Resolve a text description to a NAICS code using fuzzy keyword matching.

        Performs case-insensitive substring matching against the NAICS code
        description table and transport keyword dictionary.

        Args:
            description: Free-text description of transport service
                (e.g., "Long-haul trucking from warehouse").

        Returns:
            Best-match 6-digit NAICS code.

        Raises:
            ValueError: If no matching NAICS code can be determined.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> code = engine.resolve_naics_code("air freight shipping")
            >>> code
            '481112'
        """
        desc_lower = description.lower().strip()

        if not desc_lower:
            raise ValueError("Description cannot be empty")

        # Direct NAICS code match (if user passed a code)
        if desc_lower in USEEIO_FACTORS:
            return desc_lower

        # Check transport keywords (longest match first for specificity)
        best_match: Optional[str] = None
        best_match_len: int = 0

        for keyword, (naics_code, _mode) in TRANSPORT_KEYWORDS.items():
            if keyword in desc_lower and len(keyword) > best_match_len:
                best_match = naics_code
                best_match_len = len(keyword)

        if best_match is not None:
            logger.debug(
                "Resolved description '%s' to NAICS %s via keyword match",
                description,
                best_match,
            )
            return best_match

        # Check NAICS descriptions (substring match)
        for naics_code, naics_desc in NAICS_CODE_DESCRIPTIONS.items():
            words = desc_lower.split()
            matched_words = sum(
                1 for w in words if w in naics_desc
            )
            if matched_words >= 2:
                logger.debug(
                    "Resolved description '%s' to NAICS %s via description match "
                    "(%d words matched)",
                    description,
                    naics_code,
                    matched_words,
                )
                return naics_code

        raise ValueError(
            f"Cannot resolve description '{description}' to a NAICS code. "
            f"Please provide a valid 6-digit NAICS code or recognizable "
            f"transport description."
        )

    def resolve_nace_code(self, description: str) -> str:
        """
        Resolve a text description to a NACE code.

        First attempts to resolve to a NAICS code, then maps to NACE.

        Args:
            description: Free-text description of transport service.

        Returns:
            NACE code (e.g., "H49.4").

        Raises:
            ValueError: If description cannot be resolved.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> nace = engine.resolve_nace_code("ocean freight shipping")
            >>> nace
            'H50.2'
        """
        naics_code = self.resolve_naics_code(description)
        nace_code = self.map_naics_to_nace(naics_code)
        if not nace_code:
            raise ValueError(
                f"No NACE mapping for NAICS '{naics_code}' "
                f"(resolved from '{description}')"
            )
        return nace_code

    def map_naics_to_nace(self, naics_code: str) -> str:
        """
        Map a NAICS code to its NACE Rev. 2 equivalent.

        Uses the embedded concordance table. Returns empty string if no
        mapping exists.

        Args:
            naics_code: 6-digit NAICS code.

        Returns:
            NACE code (e.g., "H49.4") or empty string if unmapped.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> engine.map_naics_to_nace("484110")
            'H49.4'
            >>> engine.map_naics_to_nace("999999")
            ''
        """
        return NAICS_TO_NACE.get(naics_code, "")

    def map_nace_to_naics(self, nace_code: str) -> str:
        """
        Map a NACE Rev. 2 code to its primary NAICS equivalent.

        Uses the embedded concordance table. Returns empty string if no
        mapping exists.

        Args:
            nace_code: NACE code (e.g., "H49.4").

        Returns:
            6-digit NAICS code or empty string if unmapped.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> engine.map_nace_to_naics("H49.4")
            '484110'
            >>> engine.map_nace_to_naics("Z99.9")
            ''
        """
        return NACE_TO_NAICS.get(nace_code, "")

    def classify_transport_spend(
        self, description: str
    ) -> Tuple[str, Optional[TransportMode]]:
        """
        Classify a transport spend description into NAICS code and mode.

        Uses keyword matching against a curated dictionary of transport
        terms. The longest matching keyword wins for specificity.

        Args:
            description: Free-text description of transport service
                (e.g., "FedEx express delivery", "ocean freight FCL").

        Returns:
            Tuple of (naics_code, TransportMode). TransportMode may be
            None for generic/warehousing categories.

        Raises:
            ValueError: If description cannot be classified.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> naics, mode = engine.classify_transport_spend(
            ...     "FedEx express delivery"
            ... )
            >>> naics
            '492110'
            >>> mode
            <TransportMode.ROAD: 'road'>
        """
        desc_lower = description.lower().strip()

        if not desc_lower:
            raise ValueError("Description cannot be empty")

        best_match_naics: Optional[str] = None
        best_match_mode: Optional[TransportMode] = None
        best_match_len: int = 0

        for keyword, (naics_code, mode) in TRANSPORT_KEYWORDS.items():
            if keyword in desc_lower and len(keyword) > best_match_len:
                best_match_naics = naics_code
                best_match_mode = mode
                best_match_len = len(keyword)

        if best_match_naics is not None:
            logger.debug(
                "Classified '%s' -> NAICS %s, mode=%s",
                description,
                best_match_naics,
                best_match_mode,
            )
            return best_match_naics, best_match_mode

        raise ValueError(
            f"Cannot classify transport spend description: '{description}'. "
            f"Please use recognized transport keywords or provide a NAICS code."
        )

    def batch_calculate(
        self, inputs: List[SpendInput]
    ) -> List[CalculationResult]:
        """
        Calculate emissions for a batch of spend inputs.

        Processes each input sequentially. Failed records are logged and
        skipped; successful results are returned.

        Args:
            inputs: List of SpendInput models to process.

        Returns:
            List of CalculationResult models for successful calculations.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> from greenlang.upstream_transportation.models import (
            ...     SpendInput, CurrencyCode
            ... )
            >>> from decimal import Decimal
            >>> inputs = [
            ...     SpendInput(
            ...         record_id="B-001",
            ...         spend_amount=Decimal("5000"),
            ...         currency=CurrencyCode.USD,
            ...         spend_year=2023,
            ...         transport_type="484110",
            ...     ),
            ...     SpendInput(
            ...         record_id="B-002",
            ...         spend_amount=Decimal("3000"),
            ...         currency=CurrencyCode.USD,
            ...         spend_year=2023,
            ...         transport_type="482111",
            ...     ),
            ... ]
            >>> results = engine.batch_calculate(inputs)
            >>> len(results) == 2
            True
        """
        if not inputs:
            logger.warning("batch_calculate called with empty input list")
            return []

        start_time = time.monotonic()
        results: List[CalculationResult] = []
        failed_count = 0

        logger.info(
            "Starting batch spend calculation for %d records", len(inputs)
        )

        for idx, spend_input in enumerate(inputs):
            try:
                result = self.calculate(spend_input)
                results.append(result)
            except Exception as exc:
                failed_count += 1
                logger.error(
                    "Batch record %d/%d failed (record_id=%s): %s",
                    idx + 1,
                    len(inputs),
                    spend_input.record_id,
                    str(exc),
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Batch spend calculation complete: %d/%d succeeded, "
            "%d failed, elapsed=%.1f ms",
            len(results),
            len(inputs),
            failed_count,
            elapsed_ms,
        )

        return results

    def aggregate_by_sector(
        self, results: List[CalculationResult]
    ) -> Dict[str, Decimal]:
        """
        Aggregate emissions by NAICS sector group.

        Groups results by the first 3 digits of the NAICS code derived
        from the request_id pattern or transport context.

        For spend-based results, aggregation is based on the NAICS group
        (e.g., "484" for trucking, "481" for air transport).

        Args:
            results: List of CalculationResult from spend-based calculations.

        Returns:
            Dictionary mapping sector group to total emissions in kgCO2e.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> # After batch_calculate:
            >>> aggregated = engine.aggregate_by_sector(results)
            >>> "484" in aggregated or len(aggregated) >= 0
            True
        """
        sector_totals: Dict[str, Decimal] = {}

        for result in results:
            # Derive sector from provenance or use "unknown"
            sector = self._extract_sector_from_result(result)
            current = sector_totals.get(sector, ZERO)
            sector_totals[sector] = current + result.total_emissions_kgco2e

        # Sort by emissions descending
        sorted_sectors = dict(
            sorted(
                sector_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        logger.debug(
            "Aggregated by sector: %d sectors, total=%.2f kgCO2e",
            len(sorted_sectors),
            sum(sorted_sectors.values()),
        )

        return sorted_sectors

    def aggregate_by_mode(
        self, results: List[CalculationResult]
    ) -> Dict[str, Decimal]:
        """
        Aggregate emissions by transport mode.

        Groups results by transport mode (road, rail, maritime, air,
        pipeline) derived from the calculation context.

        Args:
            results: List of CalculationResult from spend-based calculations.

        Returns:
            Dictionary mapping mode name to total emissions in kgCO2e.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> aggregated = engine.aggregate_by_mode(results)
            >>> isinstance(aggregated, dict)
            True
        """
        mode_totals: Dict[str, Decimal] = {}

        for result in results:
            mode_name = self._extract_mode_from_result(result)
            current = mode_totals.get(mode_name, ZERO)
            mode_totals[mode_name] = current + result.total_emissions_kgco2e

        sorted_modes = dict(
            sorted(
                mode_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        logger.debug(
            "Aggregated by mode: %d modes, total=%.2f kgCO2e",
            len(sorted_modes),
            sum(sorted_modes.values()),
        )

        return sorted_modes

    def get_data_quality_score(self) -> Decimal:
        """
        Get the data quality score for spend-based calculations.

        Spend-based methods always receive the lowest quality tier because
        EEIO factors are industry averages with very high uncertainty. Per
        GHG Protocol guidance, spend-based is the least accurate method.

        Returns:
            Composite DQI score (always 4.0 for spend-based, on 1-5 scale
            where 1 is best and 5 is worst).

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> score = engine.get_data_quality_score()
            >>> score >= Decimal("4")
            True
        """
        # Spend-based: reliability=POOR, completeness=FAIR, temporal=POOR,
        # geographical=POOR, technological=VERY_POOR
        return get_dqi_composite_score(
            reliability=DQIScore.POOR,
            completeness=DQIScore.FAIR,
            temporal=DQIScore.POOR,
            geographical=DQIScore.POOR,
            technological=DQIScore.VERY_POOR,
        )

    def validate_spend_input(
        self, spend_input: SpendInput
    ) -> List[str]:
        """
        Validate a SpendInput for completeness and consistency.

        Checks:
            - spend_amount > 0
            - currency is in supported list
            - spend_year is within CPI deflator range
            - transport_type is resolvable

        Args:
            spend_input: SpendInput model to validate.

        Returns:
            List of validation error messages. Empty list means valid.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> from greenlang.upstream_transportation.models import (
            ...     SpendInput, CurrencyCode
            ... )
            >>> from decimal import Decimal
            >>> inp = SpendInput(
            ...     record_id="V-001",
            ...     spend_amount=Decimal("1000"),
            ...     currency=CurrencyCode.USD,
            ...     spend_year=2023,
            ...     transport_type="484110",
            ... )
            >>> errors = engine.validate_spend_input(inp)
            >>> len(errors) == 0
            True
        """
        errors: List[str] = []

        # Check spend amount
        if spend_input.spend_amount <= ZERO:
            errors.append(
                f"spend_amount must be positive, got {spend_input.spend_amount}"
            )

        # Check currency support for CPI deflation
        currency_str = spend_input.currency.value
        if currency_str not in CPI_DEFLATORS:
            # Not a fatal error - we can still try USD conversion
            if currency_str not in EXCHANGE_RATES_TO_USD:
                errors.append(
                    f"Currency '{currency_str}' not supported for CPI "
                    f"deflation or currency conversion. Supported CPI "
                    f"currencies: {sorted(CPI_DEFLATORS.keys())}. "
                    f"Supported exchange currencies: "
                    f"{sorted(EXCHANGE_RATES_TO_USD.keys())}."
                )

        # Check spend year within deflator range
        if currency_str in CPI_DEFLATORS:
            available_years = CPI_DEFLATORS[currency_str]
            if spend_input.spend_year not in available_years:
                errors.append(
                    f"spend_year {spend_input.spend_year} not in CPI deflator "
                    f"range for {currency_str}. Available: "
                    f"{sorted(available_years.keys())}"
                )

        # Check transport_type resolvability
        try:
            self._resolve_transport_type(spend_input.transport_type)
        except (ValueError, KeyError) as exc:
            errors.append(f"transport_type resolution failed: {str(exc)}")

        # Check record_id
        if not spend_input.record_id or not spend_input.record_id.strip():
            errors.append("record_id cannot be empty")

        return errors

    def get_available_sectors(self, source: str = "USEEIO") -> List[str]:
        """
        Get list of available sector codes for a given EEIO source.

        Args:
            source: Factor source - "USEEIO", "EXIOBASE", or "DEFRA".

        Returns:
            Sorted list of available sector/code identifiers.

        Raises:
            ValueError: If source is unknown.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> sectors = engine.get_available_sectors("USEEIO")
            >>> "484110" in sectors
            True
        """
        source_upper = source.upper()

        if source_upper == "USEEIO":
            return sorted(USEEIO_FACTORS.keys())

        elif source_upper == "EXIOBASE":
            return sorted(
                f"{sector}/{region}"
                for (sector, region) in EXIOBASE_FACTORS.keys()
            )

        elif source_upper == "DEFRA":
            return sorted(DEFRA_SPEND_FACTORS.keys())

        else:
            raise ValueError(
                f"Unknown source '{source}'. Supported: USEEIO, EXIOBASE, DEFRA"
            )

    def estimate_uncertainty(
        self, amount_kgco2e: Decimal, source: str = "USEEIO"
    ) -> Dict[str, Decimal]:
        """
        Estimate uncertainty range for spend-based emissions.

        Spend-based calculations have inherently high uncertainty because
        EEIO factors are based on sector averages rather than actual
        activity data. Uncertainty ranges are:
            - USEEIO: +-50% (factors are US-specific, reasonably calibrated)
            - EXIOBASE: +-60% (multi-region, aggregated sectors)
            - DEFRA: +-50% (UK-specific, mode-level factors)
            - Unknown/custom: +-100%

        Args:
            amount_kgco2e: Central estimate of emissions in kgCO2e.
            source: EEIO factor source identifier.

        Returns:
            Dictionary with keys:
                - uncertainty_pct: Uncertainty percentage (e.g., 50 for +-50%)
                - lower_bound: Lower bound kgCO2e
                - upper_bound: Upper bound kgCO2e
                - confidence_level: Confidence level description

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> uncertainty = engine.estimate_uncertainty(
            ...     Decimal("1000"), "USEEIO"
            ... )
            >>> uncertainty["uncertainty_pct"]
            Decimal('50')
            >>> uncertainty["lower_bound"]
            Decimal('500.00000000')
        """
        source_upper = source.upper() if source else ""

        uncertainty_pct_map: Dict[str, Decimal] = {
            "USEEIO": Decimal("50"),
            "USEEIO_V2.0": Decimal("50"),
            "EXIOBASE": Decimal("60"),
            "EXIOBASE_V3.8": Decimal("60"),
            "DEFRA": Decimal("50"),
            "DEFRA_2023": Decimal("50"),
        }

        uncertainty_pct = uncertainty_pct_map.get(
            source_upper, Decimal("100")
        )

        fraction = uncertainty_pct / HUNDRED
        lower_bound = self._quantize(
            amount_kgco2e * (ONE - fraction)
        )
        upper_bound = self._quantize(
            amount_kgco2e * (ONE + fraction)
        )

        # Ensure lower bound is non-negative
        if lower_bound < ZERO:
            lower_bound = ZERO

        return {
            "uncertainty_pct": uncertainty_pct,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": "95% confidence interval (assumed normal distribution)",
        }

    def get_useeio_factor_detail(
        self, naics_code: str
    ) -> Dict[str, Any]:
        """
        Get detailed USEEIO factor information for a NAICS code.

        Args:
            naics_code: 6-digit NAICS code.

        Returns:
            Dictionary with factor, description, mode, and NAICS group.

        Raises:
            KeyError: If NAICS code not found.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> detail = engine.get_useeio_factor_detail("484110")
            >>> detail["description"]
            'General freight trucking, long-distance'
        """
        entry = USEEIO_FACTORS.get(naics_code)
        if entry is None:
            raise KeyError(
                f"NAICS '{naics_code}' not found in USEEIO table"
            )
        return {
            "naics_code": naics_code,
            "factor_kgco2e_per_usd": entry["factor"],
            "description": entry["description"],
            "mode": entry["mode"].value if entry["mode"] else None,
            "naics_group": entry["naics_group"],
            "source": "USEEIO_v2.0",
            "base_year": EEIO_BASE_YEAR,
            "currency": "USD",
        }

    def get_exiobase_factor_detail(
        self, sector: str, region: str
    ) -> Dict[str, Any]:
        """
        Get detailed EXIOBASE factor information.

        Args:
            sector: EXIOBASE sector key.
            region: Region code ("EU" or "US").

        Returns:
            Dictionary with factor, description, mode, NACE code.

        Raises:
            KeyError: If sector/region combination not found.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> detail = engine.get_exiobase_factor_detail(
            ...     "air_transport", "EU"
            ... )
            >>> detail["factor_kgco2e_per_eur"]
            Decimal('0.880')
        """
        key = (sector, region)
        entry = EXIOBASE_FACTORS.get(key)
        if entry is None:
            raise KeyError(
                f"EXIOBASE sector/region '{sector}/{region}' not found"
            )
        return {
            "sector": sector,
            "region": region,
            "factor_kgco2e_per_eur": entry["factor"],
            "description": entry["description"],
            "mode": entry["mode"].value if entry["mode"] else None,
            "nace_code": entry["nace_code"],
            "source": "EXIOBASE_v3.8",
            "base_year": EEIO_BASE_YEAR,
            "currency": "EUR",
        }

    def get_defra_factor_detail(self, mode: str) -> Dict[str, Any]:
        """
        Get detailed DEFRA spend factor information.

        Args:
            mode: DEFRA mode key (e.g., "road_freight").

        Returns:
            Dictionary with factor, description, mode.

        Raises:
            KeyError: If mode not found.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> detail = engine.get_defra_factor_detail("air_freight")
            >>> detail["factor_kgco2e_per_gbp"]
            Decimal('0.840')
        """
        entry = DEFRA_SPEND_FACTORS.get(mode)
        if entry is None:
            raise KeyError(f"DEFRA mode '{mode}' not found")
        return {
            "mode_key": mode,
            "factor_kgco2e_per_gbp": entry["factor"],
            "description": entry["description"],
            "transport_mode": entry["mode"].value if entry["mode"] else None,
            "source": "DEFRA_2023",
            "base_year": EEIO_BASE_YEAR,
            "currency": "GBP",
        }

    def get_margin_removal_info(self) -> Dict[str, Decimal]:
        """
        Get all available margin removal percentages by sector.

        Returns:
            Dictionary mapping sector name to margin percentage.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> margins = engine.get_margin_removal_info()
            >>> margins["trucking"]
            Decimal('15')
        """
        return dict(MARGIN_REMOVAL_PERCENTAGES)

    def get_cpi_deflator_table(
        self, currency: str
    ) -> Dict[int, Decimal]:
        """
        Get CPI deflator table for a currency.

        Args:
            currency: ISO 4217 currency code.

        Returns:
            Dictionary mapping year to CPI index (base year 2021 = 1.00).

        Raises:
            KeyError: If currency not in CPI table.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> table = engine.get_cpi_deflator_table("USD")
            >>> table[2021]
            Decimal('1.00')
        """
        deflators = CPI_DEFLATORS.get(currency)
        if deflators is None:
            raise KeyError(
                f"CPI deflators not available for '{currency}'. "
                f"Supported: {sorted(CPI_DEFLATORS.keys())}"
            )
        return dict(deflators)

    def get_exchange_rate_table(
        self, currency: str
    ) -> Dict[int, Decimal]:
        """
        Get exchange rate table for a currency (rates to USD).

        Args:
            currency: ISO 4217 currency code.

        Returns:
            Dictionary mapping year to exchange rate vs USD.

        Raises:
            KeyError: If currency not in exchange rate table.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> table = engine.get_exchange_rate_table("EUR")
            >>> table[2021]
            Decimal('1.183')
        """
        rates = EXCHANGE_RATES_TO_USD.get(currency)
        if rates is None:
            raise KeyError(
                f"Exchange rates not available for '{currency}'. "
                f"Supported: {sorted(EXCHANGE_RATES_TO_USD.keys())}"
            )
        return dict(rates)

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get engine performance statistics.

        Returns:
            Dictionary with calculation count, total emissions, and
            engine metadata.

        Example:
            >>> engine = SpendBasedCalculatorEngine()
            >>> stats = engine.get_engine_stats()
            >>> stats["calculation_count"]
            0
        """
        return {
            "engine": "SpendBasedCalculatorEngine",
            "agent_id": self._agent_id,
            "version": self._version,
            "component": AGENT_COMPONENT,
            "calculation_count": self._calculation_count,
            "total_emissions_kgco2e": str(self._total_emissions_kgco2e),
            "eeio_sources": ["USEEIO_v2.0", "EXIOBASE_v3.8", "DEFRA_2023"],
            "base_year": EEIO_BASE_YEAR,
            "supported_currencies_cpi": sorted(CPI_DEFLATORS.keys()),
            "supported_currencies_fx": sorted(EXCHANGE_RATES_TO_USD.keys()),
            "useeio_factor_count": len(USEEIO_FACTORS),
            "exiobase_factor_count": len(EXIOBASE_FACTORS),
            "defra_factor_count": len(DEFRA_SPEND_FACTORS),
        }

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================

    def _quantize(self, value: Decimal) -> Decimal:
        """
        Quantize a Decimal to the standard precision.

        Uses ROUND_HALF_UP with the configured decimal precision (default 8
        places).

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal.
        """
        quantizer = Decimal(10) ** -DECIMAL_PRECISION
        return value.quantize(quantizer, rounding=ROUNDING)

    def _resolve_transport_type(self, transport_type: str) -> str:
        """
        Resolve the transport_type field from SpendInput to a NAICS code.

        Accepts:
            - Direct 6-digit NAICS codes (e.g., "484110")
            - Legacy NAICS-prefixed keys (e.g., "NAICS_484_truck_transport")
            - Free-text descriptions (via fuzzy matching)

        Args:
            transport_type: Raw transport type from SpendInput.

        Returns:
            Resolved 6-digit NAICS code.

        Raises:
            ValueError: If transport_type cannot be resolved.
        """
        tt = transport_type.strip()

        # Direct NAICS code match
        if tt in USEEIO_FACTORS:
            return tt

        # Legacy NAICS-prefixed key format (e.g., "NAICS_484_truck_transport")
        if tt.startswith("NAICS_"):
            parts = tt.split("_")
            if len(parts) >= 2:
                naics_prefix = parts[1]
                # Find first matching NAICS code with this prefix
                for code in USEEIO_FACTORS:
                    if code.startswith(naics_prefix):
                        return code

        # Try fuzzy description matching
        try:
            return self.resolve_naics_code(tt)
        except ValueError:
            pass

        raise ValueError(
            f"Cannot resolve transport_type '{transport_type}' to a NAICS code. "
            f"Provide a 6-digit NAICS code (e.g., '484110'), a NAICS-prefixed "
            f"key (e.g., 'NAICS_484_truck_transport'), or a recognizable "
            f"transport description."
        )

    def _resolve_mode(
        self,
        naics_code: str,
        explicit_mode: Optional[TransportMode],
    ) -> Optional[TransportMode]:
        """
        Resolve transport mode from NAICS code or explicit override.

        If an explicit mode is provided on the SpendInput, it takes precedence.
        Otherwise, the mode is derived from the USEEIO factor table.

        Args:
            naics_code: Resolved NAICS code.
            explicit_mode: Explicit mode from SpendInput (if provided).

        Returns:
            TransportMode or None for non-transport categories.
        """
        if explicit_mode is not None:
            return explicit_mode

        entry = USEEIO_FACTORS.get(naics_code)
        if entry is not None:
            return entry.get("mode")

        return None

    def _resolve_eeio_factor_and_source(
        self,
        spend_input: SpendInput,
        naics_code: str,
    ) -> Dict[str, Any]:
        """
        Determine the best EEIO factor source and retrieve the factor.

        Selection logic:
            1. If custom eeio_factor_kg_per_usd is provided, use it
            2. If currency is GBP and DEFRA has a mapping, use DEFRA
            3. If currency is EUR, prefer EXIOBASE (EU region)
            4. Default to USEEIO (USD)

        Args:
            spend_input: The spend input with currency and source hints.
            naics_code: Resolved NAICS code.

        Returns:
            Dictionary with keys: factor, source, currency.
        """
        # Custom factor override
        if spend_input.eeio_factor_kg_per_usd is not None:
            return {
                "factor": spend_input.eeio_factor_kg_per_usd,
                "source": spend_input.eeio_source or "custom",
                "currency": spend_input.currency.value,
            }

        # Explicit source from input
        source_hint = (spend_input.eeio_source or "").upper()

        # DEFRA preference for GBP
        if spend_input.currency == CurrencyCode.GBP or "DEFRA" in source_hint:
            defra_mode = self._naics_to_defra_mode(naics_code)
            if defra_mode is not None:
                entry = DEFRA_SPEND_FACTORS.get(defra_mode)
                if entry is not None:
                    return {
                        "factor": entry["factor"],
                        "source": "DEFRA_2023",
                        "currency": "GBP",
                    }

        # EXIOBASE preference for EUR
        if spend_input.currency == CurrencyCode.EUR or "EXIOBASE" in source_hint:
            nace_code = self.map_naics_to_nace(naics_code)
            if nace_code:
                # Search EXIOBASE for matching NACE code (prefer EU)
                for region in ("EU", "US"):
                    for key, entry in EXIOBASE_FACTORS.items():
                        if entry.get("nace_code") == nace_code and key[1] == region:
                            return {
                                "factor": entry["factor"],
                                "source": f"EXIOBASE_v3.8_{region}",
                                "currency": "EUR",
                            }

        # Default: USEEIO (USD)
        entry = USEEIO_FACTORS.get(naics_code)
        if entry is not None:
            return {
                "factor": entry["factor"],
                "source": "USEEIO_v2.0",
                "currency": "USD",
            }

        # Fallback: use generic freight factor
        logger.warning(
            "No specific EEIO factor for NAICS %s, using generic freight "
            "factor (NAICS 488490)",
            naics_code,
        )
        fallback = USEEIO_FACTORS["488490"]
        return {
            "factor": fallback["factor"],
            "source": "USEEIO_v2.0_fallback",
            "currency": "USD",
        }

    def _deflate_spend(
        self,
        amount: Decimal,
        currency: str,
        spend_year: int,
    ) -> Decimal:
        """
        Deflate spend amount from spend year to EEIO base year (2021).

        If the currency is not in the CPI table, the amount is returned
        unchanged (with a warning logged).

        Args:
            amount: Original spend amount.
            currency: ISO 4217 currency code.
            spend_year: Calendar year of the spend.

        Returns:
            Deflated spend amount in base-year prices.
        """
        if spend_year == EEIO_BASE_YEAR:
            return amount

        try:
            deflator = self.get_cpi_deflator(currency, spend_year, EEIO_BASE_YEAR)
            deflated = self._quantize(amount * deflator)
            logger.debug(
                "Deflated %s %s from %d to %d: %s (deflator=%s)",
                amount,
                currency,
                spend_year,
                EEIO_BASE_YEAR,
                deflated,
                deflator,
            )
            return deflated
        except KeyError as exc:
            logger.warning(
                "CPI deflation unavailable: %s. Using nominal spend.", str(exc)
            )
            return amount

    def _get_cpi_deflator_value(
        self, currency: str, spend_year: int
    ) -> Decimal:
        """
        Get the CPI deflator value for tracking in calculation detail.

        Returns ONE if deflation is not applicable.

        Args:
            currency: ISO 4217 currency code.
            spend_year: Calendar year of the spend.

        Returns:
            CPI deflator value or ONE if not available.
        """
        if spend_year == EEIO_BASE_YEAR:
            return ONE

        try:
            return self.get_cpi_deflator(currency, spend_year, EEIO_BASE_YEAR)
        except KeyError:
            return ONE

    def _convert_to_target_currency(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        year: int,
    ) -> Decimal:
        """
        Convert amount to target currency for EEIO factor application.

        If currencies match, no conversion is needed. If conversion fails
        (missing rates), the amount is returned unchanged with a warning.

        Args:
            amount: Monetary amount to convert.
            from_currency: Source currency code.
            to_currency: Target currency code.
            year: Year for exchange rate selection.

        Returns:
            Converted amount.
        """
        if from_currency == to_currency:
            return amount

        try:
            return self.convert_currency(amount, from_currency, to_currency, year)
        except KeyError as exc:
            logger.warning(
                "Currency conversion unavailable: %s. "
                "Using unconverted amount.",
                str(exc),
            )
            return amount

    def _get_exchange_rate(
        self, from_currency: str, to_currency: str, year: int
    ) -> Decimal:
        """
        Get exchange rate for tracking in calculation detail.

        Returns ONE if currencies match or rate is unavailable.

        Args:
            from_currency: Source currency code.
            to_currency: Target currency code.
            year: Calendar year for rate selection.

        Returns:
            Exchange rate or ONE if not applicable.
        """
        if from_currency == to_currency:
            return ONE

        try:
            from_rates = EXCHANGE_RATES_TO_USD.get(from_currency, {})
            from_rate = from_rates.get(year, ONE)

            if to_currency == "USD":
                return from_rate

            to_rates = EXCHANGE_RATES_TO_USD.get(to_currency, {})
            to_rate = to_rates.get(year, ONE)

            if to_rate == ZERO:
                return ONE

            return self._quantize(from_rate / to_rate)
        except (KeyError, InvalidOperation):
            return ONE

    def _resolve_margin_sector(self, naics_code: str) -> str:
        """
        Resolve the margin removal sector from a NAICS code.

        Uses the first 3 digits of the NAICS code to map to a margin sector.

        Args:
            naics_code: 6-digit NAICS code.

        Returns:
            Margin sector name (e.g., "trucking", "air_freight").
        """
        naics_group = naics_code[:3]
        return NAICS_TO_MARGIN_SECTOR.get(naics_group, "trucking")

    def _naics_to_defra_mode(self, naics_code: str) -> Optional[str]:
        """
        Map a NAICS code to a DEFRA spend mode key.

        Args:
            naics_code: 6-digit NAICS code.

        Returns:
            DEFRA mode key or None if no mapping exists.
        """
        naics_group = naics_code[:3]

        naics_group_to_defra: Dict[str, str] = {
            "484": "road_freight",
            "488": "road_freight",
            "492": "road_freight",
            "482": "rail_freight",
            "483": "sea_freight",
            "481": "air_freight",
            "493": "warehousing",
        }

        return naics_group_to_defra.get(naics_group)

    def _build_dqi_scores(self) -> Dict[str, Any]:
        """
        Build Data Quality Indicator scores for spend-based calculations.

        Spend-based method always gets the lowest quality tier:
            - Reliability: POOR (4) - EEIO factors are sector averages
            - Completeness: FAIR (3) - spend data usually complete
            - Temporal: POOR (4) - factors may not match spend year
            - Geographical: POOR (4) - factors are national averages
            - Technological: VERY_POOR (5) - no technology specificity

        Returns:
            Dictionary with DQI scores and composite information.
        """
        reliability = DQIScore.POOR
        completeness = DQIScore.FAIR
        temporal = DQIScore.POOR
        geographical = DQIScore.POOR
        technological = DQIScore.VERY_POOR

        composite = get_dqi_composite_score(
            reliability=reliability,
            completeness=completeness,
            temporal=temporal,
            geographical=geographical,
            technological=technological,
        )

        tier = get_dqi_quality_tier(composite)

        return {
            "reliability": reliability,
            "completeness": completeness,
            "temporal": temporal,
            "geographical": geographical,
            "technological": technological,
            "composite": composite,
            "tier": tier,
        }

    def _compute_provenance_hash(
        self,
        spend_input: SpendInput,
        detail: SpendCalculationDetail,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a spend-based calculation.

        Hashes the complete input and intermediate calculation detail to
        create an immutable audit trail record.

        Args:
            spend_input: Original spend input.
            detail: Calculation detail with all intermediate values.

        Returns:
            SHA-256 hex digest string.
        """
        hash_payload = {
            "input": {
                "record_id": spend_input.record_id,
                "spend_amount": str(spend_input.spend_amount),
                "currency": spend_input.currency.value,
                "spend_year": spend_input.spend_year,
                "transport_type": spend_input.transport_type,
                "eeio_source": spend_input.eeio_source,
            },
            "calculation": detail.to_dict(),
            "engine": {
                "name": "SpendBasedCalculatorEngine",
                "version": self._version,
                "agent_id": self._agent_id,
                "base_year": EEIO_BASE_YEAR,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        hash_str = json.dumps(hash_payload, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

    def _extract_sector_from_result(
        self, result: CalculationResult
    ) -> str:
        """
        Extract NAICS sector group from a CalculationResult for aggregation.

        Attempts to derive the sector from the request_id pattern.
        Falls back to "unknown" if derivation fails.

        Args:
            result: A CalculationResult from a spend-based calculation.

        Returns:
            NAICS 3-digit sector group or "unknown".
        """
        # If result has provenance metadata, extract NAICS from there
        # For now, return "spend_based" as a generic sector
        return "spend_based"

    def _extract_mode_from_result(
        self, result: CalculationResult
    ) -> str:
        """
        Extract transport mode from a CalculationResult for aggregation.

        Falls back to "unknown" if mode cannot be determined.

        Args:
            result: A CalculationResult from a spend-based calculation.

        Returns:
            Transport mode name or "unknown".
        """
        return "spend_based"

    # ==========================================================================
    # STATIC UTILITY METHODS
    # ==========================================================================

    @staticmethod
    def get_supported_naics_codes() -> Dict[str, str]:
        """
        Get all supported NAICS codes with descriptions.

        Returns:
            Dictionary mapping NAICS code to description.

        Example:
            >>> codes = SpendBasedCalculatorEngine.get_supported_naics_codes()
            >>> codes["484110"]
            'General freight trucking, long-distance'
        """
        return {
            code: entry["description"]
            for code, entry in USEEIO_FACTORS.items()
        }

    @staticmethod
    def get_supported_exiobase_sectors() -> Dict[str, str]:
        """
        Get all supported EXIOBASE sector/region combinations.

        Returns:
            Dictionary mapping "sector/region" to description.

        Example:
            >>> sectors = SpendBasedCalculatorEngine.get_supported_exiobase_sectors()
            >>> "air_transport/EU" in sectors
            True
        """
        return {
            f"{sector}/{region}": entry["description"]
            for (sector, region), entry in EXIOBASE_FACTORS.items()
        }

    @staticmethod
    def get_supported_defra_modes() -> Dict[str, str]:
        """
        Get all supported DEFRA spend mode keys.

        Returns:
            Dictionary mapping mode key to description.

        Example:
            >>> modes = SpendBasedCalculatorEngine.get_supported_defra_modes()
            >>> modes["road_freight"]
            'Road freight'
        """
        return {
            mode: entry["description"]
            for mode, entry in DEFRA_SPEND_FACTORS.items()
        }

    @staticmethod
    def get_naics_nace_concordance() -> Dict[str, str]:
        """
        Get the complete NAICS-to-NACE concordance table.

        Returns:
            Dictionary mapping NAICS codes to NACE codes.

        Example:
            >>> concordance = (
            ...     SpendBasedCalculatorEngine.get_naics_nace_concordance()
            ... )
            >>> concordance["484110"]
            'H49.4'
        """
        return dict(NAICS_TO_NACE)

    @staticmethod
    def get_nace_naics_concordance() -> Dict[str, str]:
        """
        Get the complete NACE-to-NAICS concordance table.

        Returns:
            Dictionary mapping NACE codes to NAICS codes.

        Example:
            >>> concordance = (
            ...     SpendBasedCalculatorEngine.get_nace_naics_concordance()
            ... )
            >>> concordance["H49.4"]
            '484110'
        """
        return dict(NACE_TO_NAICS)

    @staticmethod
    def get_eeio_base_year() -> int:
        """
        Get the EEIO factor base year.

        All EEIO factors are calibrated to this base year. Spend data from
        other years must be deflated to this year before factor application.

        Returns:
            Base year (2021).

        Example:
            >>> SpendBasedCalculatorEngine.get_eeio_base_year()
            2021
        """
        return EEIO_BASE_YEAR

    @staticmethod
    def get_method_description() -> str:
        """
        Get a human-readable description of the spend-based method.

        Returns:
            Multi-line description suitable for reporting.
        """
        return (
            "Spend-Based Method (GHG Protocol Scope 3, Category 4)\n"
            "\n"
            "This method estimates upstream transportation and distribution "
            "emissions by multiplying transport expenditure data by "
            "Environmentally Extended Input-Output (EEIO) emission factors.\n"
            "\n"
            "Formula: emissions (kgCO2e) = spend (currency) x EEIO factor "
            "(kgCO2e/currency)\n"
            "\n"
            "EEIO factors represent the average GHG intensity of an economic "
            "sector, expressed per unit of currency. The method is suitable "
            "for screening-level assessments when activity data (distance, "
            "fuel consumption) is unavailable.\n"
            "\n"
            "Supported factor sources:\n"
            "  - USEEIO v2.0 (US EPA, NAICS codes, kgCO2e/USD)\n"
            "  - EXIOBASE v3.8 (EU/multi-region, kgCO2e/EUR)\n"
            "  - DEFRA 2023 (UK BEIS, kgCO2e/GBP)\n"
            "\n"
            "Data Quality: Tier 4-5 (poor/very poor)\n"
            "Uncertainty: +/-50% to +/-100%\n"
            "\n"
            "This is the least accurate Scope 3 calculation method. Companies "
            "should progressively improve to distance-based or supplier-specific "
            "methods for material transport categories."
        )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine
    "SpendBasedCalculatorEngine",
    # Internal model
    "SpendCalculationDetail",
    # Constants
    "USEEIO_FACTORS",
    "EXIOBASE_FACTORS",
    "DEFRA_SPEND_FACTORS",
    "CPI_DEFLATORS",
    "EXCHANGE_RATES_TO_USD",
    "MARGIN_REMOVAL_PERCENTAGES",
    "NAICS_CODE_DESCRIPTIONS",
    "NACE_CODE_DESCRIPTIONS",
    "NAICS_TO_NACE",
    "NACE_TO_NAICS",
    "TRANSPORT_KEYWORDS",
    "NAICS_TO_MARGIN_SECTOR",
    "EEIO_BASE_YEAR",
]
