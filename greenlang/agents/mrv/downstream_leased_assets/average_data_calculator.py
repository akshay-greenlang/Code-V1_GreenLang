# -*- coding: utf-8 -*-
"""
AverageDataCalculatorEngine - AGENT-MRV-026 Engine 3

GHG Protocol Scope 3 Category 13 average-data (Tier 2) emissions calculator
using benchmark Energy Use Intensity (EUI) and activity-data intensities for
downstream leased assets.

This engine calculates emissions when the lessor (reporting company) does NOT
have metered tenant energy data, and must rely on building-type benchmarks,
fleet-average vehicle distances, equipment load factors, or IT power defaults.

Supported Asset Categories:
    1. **Buildings**: 8 building types x 5 climate zones, EUI benchmark lookup,
       EPC/NABERS/Energy Star rating proxies, vacancy + base-load adjustments.
    2. **Vehicles**: 8 vehicle types with default annual distances and
       type/fuel-specific emission factors.
    3. **Equipment**: 6 equipment types with rated-power x load-factor x
       annual-hours x fuel-EF calculations.
    4. **IT Assets**: 7 IT asset types with default power x PUE x annual-hours
       x grid-EF calculations.

Core Formulas:
    Building:
        E = floor_area_sqm x EUI(type, climate) x grid_EF(region)
            x lease_share x (1 - vacancy_fraction x (1 - base_load_fraction))

    Vehicle:
        E = annual_distance x vehicle_EF(type, fuel) x lease_share x fleet_count

    Equipment:
        E = rated_power x load_factor x annual_hours x fuel_EF x lease_share

    IT Asset:
        E = default_power x default_PUE x hours_per_year x grid_EF
            x lease_share x quantity

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern for concurrent pipeline use.

Data Quality:
    Average-data is Tier 2 accuracy. Default uncertainty is +/-30%.
    Organizations should prioritize asset-specific (metered) data when
    tenant cooperation allows.

References:
    - GHG Protocol Technical Guidance for Scope 3 Category 13
    - ASHRAE 90.1 EUI Benchmarks by Building Type
    - ENERGY STAR Portfolio Manager Technical Reference
    - NABERS Energy Rating System
    - EPA eGRID / IEA Grid Emission Factors

Example:
    >>> engine = get_average_data_calculator()
    >>> result = engine.calculate({
    ...     "asset_category": "building",
    ...     "building_type": "office",
    ...     "climate_zone": "temperate",
    ...     "floor_area_sqm": Decimal("5000"),
    ...     "region": "US_AVERAGE",
    ...     "lease_share": Decimal("1.0"),
    ... })
    >>> result["co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import hashlib
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "average_data_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-013"

# Decimal precision
PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_2DP: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")

# Default uncertainty for Tier 2 average-data method (+/-30%)
TIER_2_UNCERTAINTY: Decimal = Decimal("0.30")

# Hours in a year for annualized calculations
HOURS_PER_YEAR: Decimal = Decimal("8760")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class BuildingType(str, Enum):
    """Building types for EUI benchmark lookup."""

    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    DATA_CENTER = "data_center"
    MIXED_USE = "mixed_use"


class ClimateZone(str, Enum):
    """ASHRAE climate zones for EUI adjustment."""

    HOT_HUMID = "hot_humid"              # ASHRAE 1A-2A
    HOT_DRY = "hot_dry"                  # ASHRAE 2B-3B
    TEMPERATE = "temperate"              # ASHRAE 3A-4A
    COLD = "cold"                        # ASHRAE 5A-6A
    VERY_COLD = "very_cold"              # ASHRAE 7-8


class VehicleType(str, Enum):
    """Vehicle types for fleet benchmark calculations."""

    SMALL_CAR = "small_car"
    MEDIUM_CAR = "medium_car"
    LARGE_CAR = "large_car"
    SUV = "suv"
    LIGHT_VAN = "light_van"
    HEAVY_TRUCK = "heavy_truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class VehicleFuelType(str, Enum):
    """Fuel types for leased vehicles."""

    PETROL = "petrol"
    DIESEL = "diesel"
    HYBRID = "hybrid"
    BEV = "bev"
    LPG = "lpg"
    CNG = "cng"
    HYDROGEN = "hydrogen"


class EquipmentType(str, Enum):
    """Equipment types for leased industrial assets."""

    MANUFACTURING = "manufacturing"
    CONSTRUCTION = "construction"
    GENERATOR = "generator"
    COMPRESSOR = "compressor"
    FORKLIFT = "forklift"
    AGRICULTURAL = "agricultural"


class ITAssetType(str, Enum):
    """IT asset types for leased technology."""

    SERVER = "server"
    STORAGE_ARRAY = "storage_array"
    NETWORK_SWITCH = "network_switch"
    UPS = "ups"
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    PRINTER = "printer"


class EPCRating(str, Enum):
    """Energy Performance Certificate rating (EU standard)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class AssetCategory(str, Enum):
    """Top-level asset categories."""

    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_ASSET = "it_asset"


# ==============================================================================
# BENCHMARK DATA TABLES
# ==============================================================================

# Building EUI benchmarks (kWh/sqm/year) by building type and climate zone
# Source: ASHRAE 90.1 / ENERGY STAR Portfolio Manager / CBECS
BUILDING_EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    BuildingType.OFFICE.value: {
        ClimateZone.HOT_HUMID.value: Decimal("250"),
        ClimateZone.HOT_DRY.value: Decimal("230"),
        ClimateZone.TEMPERATE.value: Decimal("200"),
        ClimateZone.COLD.value: Decimal("220"),
        ClimateZone.VERY_COLD.value: Decimal("260"),
    },
    BuildingType.RETAIL.value: {
        ClimateZone.HOT_HUMID.value: Decimal("300"),
        ClimateZone.HOT_DRY.value: Decimal("280"),
        ClimateZone.TEMPERATE.value: Decimal("240"),
        ClimateZone.COLD.value: Decimal("260"),
        ClimateZone.VERY_COLD.value: Decimal("310"),
    },
    BuildingType.WAREHOUSE.value: {
        ClimateZone.HOT_HUMID.value: Decimal("120"),
        ClimateZone.HOT_DRY.value: Decimal("110"),
        ClimateZone.TEMPERATE.value: Decimal("100"),
        ClimateZone.COLD.value: Decimal("130"),
        ClimateZone.VERY_COLD.value: Decimal("150"),
    },
    BuildingType.HOTEL.value: {
        ClimateZone.HOT_HUMID.value: Decimal("340"),
        ClimateZone.HOT_DRY.value: Decimal("310"),
        ClimateZone.TEMPERATE.value: Decimal("280"),
        ClimateZone.COLD.value: Decimal("300"),
        ClimateZone.VERY_COLD.value: Decimal("360"),
    },
    BuildingType.HOSPITAL.value: {
        ClimateZone.HOT_HUMID.value: Decimal("450"),
        ClimateZone.HOT_DRY.value: Decimal("420"),
        ClimateZone.TEMPERATE.value: Decimal("380"),
        ClimateZone.COLD.value: Decimal("410"),
        ClimateZone.VERY_COLD.value: Decimal("470"),
    },
    BuildingType.SCHOOL.value: {
        ClimateZone.HOT_HUMID.value: Decimal("200"),
        ClimateZone.HOT_DRY.value: Decimal("185"),
        ClimateZone.TEMPERATE.value: Decimal("160"),
        ClimateZone.COLD.value: Decimal("180"),
        ClimateZone.VERY_COLD.value: Decimal("220"),
    },
    BuildingType.DATA_CENTER.value: {
        ClimateZone.HOT_HUMID.value: Decimal("1200"),
        ClimateZone.HOT_DRY.value: Decimal("1100"),
        ClimateZone.TEMPERATE.value: Decimal("1000"),
        ClimateZone.COLD.value: Decimal("950"),
        ClimateZone.VERY_COLD.value: Decimal("900"),
    },
    BuildingType.MIXED_USE.value: {
        ClimateZone.HOT_HUMID.value: Decimal("270"),
        ClimateZone.HOT_DRY.value: Decimal("250"),
        ClimateZone.TEMPERATE.value: Decimal("220"),
        ClimateZone.COLD.value: Decimal("240"),
        ClimateZone.VERY_COLD.value: Decimal("280"),
    },
}

# EPC rating multipliers relative to a C-rated building (baseline = 1.0)
# A-rated buildings use ~50% of C-rated, G-rated use ~200%
EPC_RATING_MULTIPLIERS: Dict[str, Decimal] = {
    EPCRating.A.value: Decimal("0.50"),
    EPCRating.B.value: Decimal("0.70"),
    EPCRating.C.value: Decimal("1.00"),
    EPCRating.D.value: Decimal("1.20"),
    EPCRating.E.value: Decimal("1.40"),
    EPCRating.F.value: Decimal("1.70"),
    EPCRating.G.value: Decimal("2.00"),
}

# Default base-load fractions by building type
# Base load = fraction of energy consumed even when vacant (lighting, HVAC standby)
BASE_LOAD_FRACTIONS: Dict[str, Decimal] = {
    BuildingType.OFFICE.value: Decimal("0.30"),
    BuildingType.RETAIL.value: Decimal("0.25"),
    BuildingType.WAREHOUSE.value: Decimal("0.15"),
    BuildingType.HOTEL.value: Decimal("0.35"),
    BuildingType.HOSPITAL.value: Decimal("0.50"),
    BuildingType.SCHOOL.value: Decimal("0.20"),
    BuildingType.DATA_CENTER.value: Decimal("0.80"),
    BuildingType.MIXED_USE.value: Decimal("0.30"),
}

# Default occupancy rates by building type
DEFAULT_OCCUPANCY_RATES: Dict[str, Decimal] = {
    BuildingType.OFFICE.value: Decimal("0.85"),
    BuildingType.RETAIL.value: Decimal("0.90"),
    BuildingType.WAREHOUSE.value: Decimal("0.95"),
    BuildingType.HOTEL.value: Decimal("0.65"),
    BuildingType.HOSPITAL.value: Decimal("0.75"),
    BuildingType.SCHOOL.value: Decimal("0.70"),
    BuildingType.DATA_CENTER.value: Decimal("0.95"),
    BuildingType.MIXED_USE.value: Decimal("0.80"),
}

# Grid emission factors (kgCO2e/kWh) by region
# Source: EPA eGRID 2022 / IEA 2023
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US_AVERAGE": Decimal("0.3937"),
    "US_NORTHEAST": Decimal("0.2780"),
    "US_SOUTHEAST": Decimal("0.4200"),
    "US_MIDWEST": Decimal("0.5100"),
    "US_WEST": Decimal("0.2850"),
    "US_TEXAS": Decimal("0.3950"),
    "EU_AVERAGE": Decimal("0.2560"),
    "EU_NORDIC": Decimal("0.0400"),
    "EU_CENTRAL": Decimal("0.3200"),
    "EU_SOUTHERN": Decimal("0.2700"),
    "UK": Decimal("0.2070"),
    "CANADA": Decimal("0.1200"),
    "AUSTRALIA": Decimal("0.6200"),
    "JAPAN": Decimal("0.4570"),
    "CHINA": Decimal("0.5810"),
    "INDIA": Decimal("0.7080"),
    "BRAZIL": Decimal("0.0740"),
    "SOUTH_AFRICA": Decimal("0.9280"),
    "SOUTH_KOREA": Decimal("0.4590"),
    "SINGAPORE": Decimal("0.4085"),
    "GLOBAL_AVERAGE": Decimal("0.4360"),
}

# Default annual distances (km/year) by vehicle type
DEFAULT_ANNUAL_DISTANCES: Dict[str, Decimal] = {
    VehicleType.SMALL_CAR.value: Decimal("15000"),
    VehicleType.MEDIUM_CAR.value: Decimal("20000"),
    VehicleType.LARGE_CAR.value: Decimal("20000"),
    VehicleType.SUV.value: Decimal("18000"),
    VehicleType.LIGHT_VAN.value: Decimal("25000"),
    VehicleType.HEAVY_TRUCK.value: Decimal("80000"),
    VehicleType.BUS.value: Decimal("60000"),
    VehicleType.MOTORCYCLE.value: Decimal("8000"),
}

# Vehicle emission factors (kgCO2e/km) by vehicle type and fuel type
# Source: DEFRA 2024 / EPA SmartWay
VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    VehicleType.SMALL_CAR.value: {
        VehicleFuelType.PETROL.value: Decimal("0.14960"),
        VehicleFuelType.DIESEL.value: Decimal("0.13860"),
        VehicleFuelType.HYBRID.value: Decimal("0.10250"),
        VehicleFuelType.BEV.value: Decimal("0.04406"),
        VehicleFuelType.LPG.value: Decimal("0.16200"),
        VehicleFuelType.CNG.value: Decimal("0.15500"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.MEDIUM_CAR.value: {
        VehicleFuelType.PETROL.value: Decimal("0.19240"),
        VehicleFuelType.DIESEL.value: Decimal("0.17170"),
        VehicleFuelType.HYBRID.value: Decimal("0.12710"),
        VehicleFuelType.BEV.value: Decimal("0.05280"),
        VehicleFuelType.LPG.value: Decimal("0.20800"),
        VehicleFuelType.CNG.value: Decimal("0.19900"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.LARGE_CAR.value: {
        VehicleFuelType.PETROL.value: Decimal("0.28350"),
        VehicleFuelType.DIESEL.value: Decimal("0.23280"),
        VehicleFuelType.HYBRID.value: Decimal("0.17830"),
        VehicleFuelType.BEV.value: Decimal("0.07005"),
        VehicleFuelType.LPG.value: Decimal("0.29500"),
        VehicleFuelType.CNG.value: Decimal("0.27600"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.SUV.value: {
        VehicleFuelType.PETROL.value: Decimal("0.24100"),
        VehicleFuelType.DIESEL.value: Decimal("0.21500"),
        VehicleFuelType.HYBRID.value: Decimal("0.15200"),
        VehicleFuelType.BEV.value: Decimal("0.06500"),
        VehicleFuelType.LPG.value: Decimal("0.25800"),
        VehicleFuelType.CNG.value: Decimal("0.24500"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.LIGHT_VAN.value: {
        VehicleFuelType.PETROL.value: Decimal("0.24520"),
        VehicleFuelType.DIESEL.value: Decimal("0.24940"),
        VehicleFuelType.HYBRID.value: Decimal("0.18200"),
        VehicleFuelType.BEV.value: Decimal("0.07200"),
        VehicleFuelType.LPG.value: Decimal("0.26800"),
        VehicleFuelType.CNG.value: Decimal("0.25200"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.HEAVY_TRUCK.value: {
        VehicleFuelType.PETROL.value: Decimal("0.84480"),
        VehicleFuelType.DIESEL.value: Decimal("0.85310"),
        VehicleFuelType.HYBRID.value: Decimal("0.68200"),
        VehicleFuelType.BEV.value: Decimal("0.24000"),
        VehicleFuelType.LPG.value: Decimal("0.88000"),
        VehicleFuelType.CNG.value: Decimal("0.78500"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.BUS.value: {
        VehicleFuelType.PETROL.value: Decimal("1.02500"),
        VehicleFuelType.DIESEL.value: Decimal("0.89180"),
        VehicleFuelType.HYBRID.value: Decimal("0.62500"),
        VehicleFuelType.BEV.value: Decimal("0.32000"),
        VehicleFuelType.LPG.value: Decimal("1.05000"),
        VehicleFuelType.CNG.value: Decimal("0.82000"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
    VehicleType.MOTORCYCLE.value: {
        VehicleFuelType.PETROL.value: Decimal("0.11337"),
        VehicleFuelType.DIESEL.value: Decimal("0.10500"),
        VehicleFuelType.HYBRID.value: Decimal("0.07500"),
        VehicleFuelType.BEV.value: Decimal("0.02500"),
        VehicleFuelType.LPG.value: Decimal("0.12000"),
        VehicleFuelType.CNG.value: Decimal("0.11500"),
        VehicleFuelType.HYDROGEN.value: Decimal("0.00000"),
    },
}

# Default annual operating hours by equipment type
DEFAULT_ANNUAL_HOURS: Dict[str, Decimal] = {
    EquipmentType.MANUFACTURING.value: Decimal("4000"),
    EquipmentType.CONSTRUCTION.value: Decimal("2000"),
    EquipmentType.GENERATOR.value: Decimal("1500"),
    EquipmentType.COMPRESSOR.value: Decimal("3500"),
    EquipmentType.FORKLIFT.value: Decimal("2500"),
    EquipmentType.AGRICULTURAL.value: Decimal("1200"),
}

# Default load factors by equipment type (fraction of rated power)
DEFAULT_LOAD_FACTORS: Dict[str, Decimal] = {
    EquipmentType.MANUFACTURING.value: Decimal("0.65"),
    EquipmentType.CONSTRUCTION.value: Decimal("0.50"),
    EquipmentType.GENERATOR.value: Decimal("0.75"),
    EquipmentType.COMPRESSOR.value: Decimal("0.60"),
    EquipmentType.FORKLIFT.value: Decimal("0.45"),
    EquipmentType.AGRICULTURAL.value: Decimal("0.55"),
}

# Fuel emission factors for equipment (kgCO2e/kWh) by fuel type
# Source: DEFRA / EPA / IPCC
EQUIPMENT_FUEL_EFS: Dict[str, Decimal] = {
    "diesel": Decimal("0.25301"),
    "petrol": Decimal("0.22975"),
    "natural_gas": Decimal("0.18316"),
    "lpg": Decimal("0.21445"),
    "electricity": Decimal("0.39370"),   # US average grid
    "biodiesel_b20": Decimal("0.20240"),
}

# Default power consumption for IT assets (kW)
IT_ASSET_DEFAULT_POWER: Dict[str, Decimal] = {
    ITAssetType.SERVER.value: Decimal("0.500"),
    ITAssetType.STORAGE_ARRAY.value: Decimal("1.200"),
    ITAssetType.NETWORK_SWITCH.value: Decimal("0.150"),
    ITAssetType.UPS.value: Decimal("0.800"),
    ITAssetType.DESKTOP.value: Decimal("0.120"),
    ITAssetType.LAPTOP.value: Decimal("0.045"),
    ITAssetType.PRINTER.value: Decimal("0.060"),
}

# Default PUE (Power Usage Effectiveness) by facility type
# PUE = Total facility power / IT equipment power
DEFAULT_PUE: Dict[str, Decimal] = {
    "enterprise_data_center": Decimal("1.60"),
    "colocation": Decimal("1.50"),
    "hyperscale": Decimal("1.20"),
    "server_room": Decimal("2.00"),
    "edge": Decimal("1.80"),
    "default": Decimal("1.58"),
}

# Default annual operating hours for IT assets
IT_ASSET_DEFAULT_HOURS: Dict[str, Decimal] = {
    ITAssetType.SERVER.value: Decimal("8760"),         # 24/7
    ITAssetType.STORAGE_ARRAY.value: Decimal("8760"),  # 24/7
    ITAssetType.NETWORK_SWITCH.value: Decimal("8760"), # 24/7
    ITAssetType.UPS.value: Decimal("8760"),            # 24/7
    ITAssetType.DESKTOP.value: Decimal("2500"),        # ~10h/day weekdays
    ITAssetType.LAPTOP.value: Decimal("2000"),         # ~8h/day weekdays
    ITAssetType.PRINTER.value: Decimal("1500"),        # ~6h/day weekdays
}

# NABERS Star Rating EUI multipliers (Australian)
NABERS_MULTIPLIERS: Dict[str, Decimal] = {
    "1_star": Decimal("1.80"),
    "2_star": Decimal("1.50"),
    "3_star": Decimal("1.20"),
    "4_star": Decimal("1.00"),
    "5_star": Decimal("0.75"),
    "6_star": Decimal("0.50"),
}

# Energy Star score EUI multipliers (US)
ENERGY_STAR_MULTIPLIERS: Dict[str, Decimal] = {
    "1_25": Decimal("1.60"),    # Score 1-25
    "26_50": Decimal("1.20"),   # Score 26-50
    "51_75": Decimal("0.90"),   # Score 51-75 (better than median)
    "76_100": Decimal("0.65"),  # Score 76-100 (top quartile)
}

# DQI dimension weights for Tier 2 scoring
DQI_WEIGHTS_TIER2: Dict[str, Decimal] = {
    "representativeness": Decimal("0.30"),
    "completeness": Decimal("0.25"),
    "temporal": Decimal("0.15"),
    "geographical": Decimal("0.15"),
    "technological": Decimal("0.15"),
}

# Default DQI scores for Tier 2 average-data method
DEFAULT_DQI_SCORES_TIER2: Dict[str, Decimal] = {
    "representativeness": Decimal("3"),   # Industry average
    "completeness": Decimal("3"),         # Partial coverage
    "temporal": Decimal("3"),             # Within 3 years
    "geographical": Decimal("3"),         # Country/region level
    "technological": Decimal("3"),        # Average technology
}


# ==============================================================================
# PROVENANCE HELPER
# ==============================================================================


def _calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Supports Decimal, dict, list, and any stringifiable objects.
    Ensures deterministic output via sorted JSON serialization.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, dict):
            hash_input += json.dumps(inp, sort_keys=True, default=str)
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUNDING))
        elif isinstance(inp, (list, tuple)):
            hash_input += json.dumps(
                [str(x) if isinstance(x, Decimal) else x for x in inp],
                sort_keys=True,
                default=str,
            )
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# AverageDataCalculatorEngine
# ==============================================================================


class AverageDataCalculatorEngine:
    """
    Average-data (Tier 2) emissions calculator for downstream leased assets.

    Implements benchmark-based emissions estimation for GHG Protocol Scope 3
    Category 13 (Downstream Leased Assets) when tenant-specific metered data
    is unavailable. Uses building EUI benchmarks, fleet vehicle averages,
    equipment load factors, and IT power defaults.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.
        All mutable state is protected by the lock.

    Data Quality:
        Average-data estimates are Tier 2. Default uncertainty is +/-30%.
        GHG Protocol recommends progressing to asset-specific (Tier 1) data
        as tenant relationships mature.

    Lessor-Specific Considerations:
        - EPC/NABERS/Energy Star ratings as EUI proxy when available
        - Fleet telemetry data as distance estimate for leased vehicles
        - Default occupancy assumptions by building type
        - Vacancy period handling with base-load fraction

    Attributes:
        _calculation_count: Running count of calculations performed
        _batch_count: Running count of batch calculations performed

    Example:
        >>> engine = AverageDataCalculatorEngine.get_instance()
        >>> result = engine.calculate({
        ...     "asset_category": "building",
        ...     "building_type": "office",
        ...     "climate_zone": "temperate",
        ...     "floor_area_sqm": Decimal("5000"),
        ...     "region": "US_AVERAGE",
        ...     "lease_share": Decimal("1.0"),
        ... })
        >>> result["co2e_kg"] > Decimal("0")
        True
    """

    _instance: Optional["AverageDataCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize AverageDataCalculatorEngine."""
        self._calculation_count: int = 0
        self._batch_count: int = 0
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "AverageDataCalculatorEngine initialized: version=%s, agent=%s",
            ENGINE_VERSION,
            AGENT_ID,
        )

    @classmethod
    def get_instance(cls) -> "AverageDataCalculatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            AverageDataCalculatorEngine singleton instance.

        Example:
            >>> engine = AverageDataCalculatorEngine.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level lock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("AverageDataCalculatorEngine singleton reset")

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def calculate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate average-data emissions for a downstream leased asset.

        Routes to the appropriate benchmark calculator based on asset_category.

        Args:
            input_data: Dictionary containing:
                - asset_category (str): "building", "vehicle", "equipment",
                  or "it_asset"
                - Plus category-specific fields (see individual methods)

        Returns:
            Dictionary with co2e_kg, method, tier, uncertainty, DQI,
            provenance_hash, and category-specific detail fields.

        Raises:
            ValueError: If asset_category is missing or unsupported.
            ValueError: If required fields are missing or invalid.

        Example:
            >>> result = engine.calculate({
            ...     "asset_category": "building",
            ...     "building_type": "office",
            ...     "climate_zone": "temperate",
            ...     "floor_area_sqm": Decimal("5000"),
            ...     "region": "US_AVERAGE",
            ...     "lease_share": Decimal("1.0"),
            ... })
        """
        start_time = time.monotonic()

        # Validate asset_category
        category = input_data.get("asset_category", "").lower()
        if not category:
            raise ValueError("asset_category is required")

        # Route to appropriate calculator
        if category == AssetCategory.BUILDING.value:
            result = self.calculate_building_benchmark(input_data)
        elif category == AssetCategory.VEHICLE.value:
            result = self.calculate_vehicle_benchmark(input_data)
        elif category == AssetCategory.EQUIPMENT.value:
            result = self.calculate_equipment_benchmark(input_data)
        elif category == AssetCategory.IT_ASSET.value:
            result = self.calculate_it_benchmark(input_data)
        else:
            raise ValueError(
                f"Unsupported asset_category '{category}'. "
                f"Valid categories: {[c.value for c in AssetCategory]}"
            )

        # Add common metadata
        duration = time.monotonic() - start_time
        result["engine_id"] = ENGINE_ID
        result["engine_version"] = ENGINE_VERSION
        result["calculation_method"] = "average_data"
        result["data_quality_tier"] = "tier_2"
        result["processing_time_ms"] = round(duration * 1000, 4)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        self._calculation_count += 1

        logger.info(
            "Average-data calculation complete: category=%s, co2e=%s kgCO2e, "
            "duration=%.4fs",
            category,
            result.get("co2e_kg", _ZERO),
            duration,
        )

        return result

    def calculate_building_benchmark(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate building emissions using EUI benchmark data.

        Formula:
            E = floor_area_sqm x EUI(type, climate) x grid_EF(region)
                x lease_share
                x (1 - vacancy_fraction x (1 - base_load_fraction))

        If an EPC/NABERS/Energy Star rating is provided, the EUI is adjusted
        by the corresponding multiplier.

        Args:
            input_data: Dictionary containing:
                - building_type (str): One of BuildingType values
                - climate_zone (str): One of ClimateZone values
                - floor_area_sqm (Decimal): Total leasable floor area
                - region (str): Grid emission factor region key
                - lease_share (Decimal): Fraction of asset leased (0-1)
                - vacancy_fraction (Decimal, optional): Vacancy rate (0-1)
                - base_load_fraction (Decimal, optional): Override base load
                - epc_rating (str, optional): EPC rating A-G
                - nabers_rating (str, optional): NABERS star rating
                - energy_star_band (str, optional): Energy Star score band

        Returns:
            Dictionary with co2e_kg, eui_kwh_sqm_yr, grid_ef, floor_area_sqm,
            vacancy_adjustment, provenance_hash, dqi_score, uncertainty.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        start_time = time.monotonic()

        # Validate required fields
        self._validate_building_inputs(input_data)

        building_type = input_data["building_type"].lower()
        climate_zone = input_data["climate_zone"].lower()
        floor_area_sqm = Decimal(str(input_data["floor_area_sqm"]))
        region = input_data.get("region", "GLOBAL_AVERAGE")
        lease_share = Decimal(str(input_data.get("lease_share", "1.0")))

        # Look up EUI benchmark
        eui = self.estimate_building_eui(building_type, climate_zone)

        # Apply EPC/NABERS/Energy Star adjustment if provided
        eui = self._apply_rating_adjustment(eui, input_data)

        # Look up grid emission factor
        grid_ef = self.get_benchmark_ef(region)

        # Vacancy adjustment
        vacancy_fraction = Decimal(
            str(input_data.get("vacancy_fraction", "0"))
        )
        base_load_fraction = Decimal(
            str(
                input_data.get(
                    "base_load_fraction",
                    str(BASE_LOAD_FRACTIONS.get(building_type, Decimal("0.30"))),
                )
            )
        )
        vacancy_adjustment = _ONE - (
            vacancy_fraction * (_ONE - base_load_fraction)
        )

        # Core formula
        energy_kwh = (floor_area_sqm * eui).quantize(_QUANT_8DP, rounding=ROUNDING)
        co2e_kg = (
            energy_kwh * grid_ef * lease_share * vacancy_adjustment
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(input_data)
        uncertainty = self.compute_uncertainty(co2e_kg)

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            building_type, climate_zone, str(floor_area_sqm),
            str(eui), str(grid_ef), str(lease_share),
            str(vacancy_adjustment), str(co2e_kg),
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "Building benchmark: type=%s, zone=%s, area=%s sqm, "
            "EUI=%s kWh/sqm/yr, grid_EF=%s, co2e=%s kgCO2e",
            building_type, climate_zone, floor_area_sqm,
            eui, grid_ef, co2e_kg,
        )

        return {
            "asset_category": AssetCategory.BUILDING.value,
            "building_type": building_type,
            "climate_zone": climate_zone,
            "floor_area_sqm": str(floor_area_sqm),
            "eui_kwh_sqm_yr": str(eui),
            "energy_kwh": str(energy_kwh),
            "grid_ef_kg_per_kwh": str(grid_ef),
            "region": region,
            "lease_share": str(lease_share),
            "vacancy_fraction": str(vacancy_fraction),
            "base_load_fraction": str(base_load_fraction),
            "vacancy_adjustment": str(vacancy_adjustment),
            "co2e_kg": co2e_kg,
            "dqi_score": dqi_score,
            "uncertainty_pct": str(uncertainty),
            "uncertainty_lower_kg": str(
                (co2e_kg * (_ONE - uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "uncertainty_upper_kg": str(
                (co2e_kg * (_ONE + uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "provenance_hash": provenance_hash,
        }

    def calculate_vehicle_benchmark(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate vehicle emissions using benchmark annual distances.

        Formula:
            E = annual_distance x vehicle_EF(type, fuel) x lease_share
                x fleet_count

        If annual_distance is not provided, uses DEFAULT_ANNUAL_DISTANCES
        for the given vehicle type.

        Args:
            input_data: Dictionary containing:
                - vehicle_type (str): One of VehicleType values
                - fuel_type (str, optional): One of VehicleFuelType values
                  (default: "petrol")
                - annual_distance_km (Decimal, optional): Override distance
                - lease_share (Decimal, optional): Fraction leased (default 1.0)
                - fleet_count (int, optional): Number of vehicles (default 1)

        Returns:
            Dictionary with co2e_kg, vehicle_ef, annual_distance_km,
            fleet_count, provenance_hash, dqi_score, uncertainty.

        Raises:
            ValueError: If vehicle_type is missing or unsupported.
        """
        start_time = time.monotonic()

        self._validate_vehicle_inputs(input_data)

        vehicle_type = input_data["vehicle_type"].lower()
        fuel_type = input_data.get("fuel_type", VehicleFuelType.PETROL.value).lower()
        lease_share = Decimal(str(input_data.get("lease_share", "1.0")))
        fleet_count = int(input_data.get("fleet_count", 1))

        # Resolve annual distance
        annual_distance_km = self.estimate_vehicle_distance(
            vehicle_type,
            override_km=input_data.get("annual_distance_km"),
        )

        # Look up emission factor
        vehicle_ef = self._lookup_vehicle_ef(vehicle_type, fuel_type)

        # Core formula
        co2e_kg = (
            annual_distance_km * vehicle_ef * lease_share * Decimal(str(fleet_count))
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(input_data)
        uncertainty = self.compute_uncertainty(co2e_kg)

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            vehicle_type, fuel_type, str(annual_distance_km),
            str(vehicle_ef), str(lease_share), str(fleet_count),
            str(co2e_kg),
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "Vehicle benchmark: type=%s, fuel=%s, dist=%s km, "
            "EF=%s, fleet=%d, co2e=%s kgCO2e",
            vehicle_type, fuel_type, annual_distance_km,
            vehicle_ef, fleet_count, co2e_kg,
        )

        return {
            "asset_category": AssetCategory.VEHICLE.value,
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "annual_distance_km": str(annual_distance_km),
            "vehicle_ef_kg_per_km": str(vehicle_ef),
            "lease_share": str(lease_share),
            "fleet_count": fleet_count,
            "co2e_kg": co2e_kg,
            "dqi_score": dqi_score,
            "uncertainty_pct": str(uncertainty),
            "uncertainty_lower_kg": str(
                (co2e_kg * (_ONE - uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "uncertainty_upper_kg": str(
                (co2e_kg * (_ONE + uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "provenance_hash": provenance_hash,
        }

    def calculate_equipment_benchmark(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate equipment emissions using rated power and load factor.

        Formula:
            E = rated_power_kw x load_factor x annual_hours x fuel_EF
                x lease_share

        Uses defaults for load_factor and annual_hours when not provided.

        Args:
            input_data: Dictionary containing:
                - equipment_type (str): One of EquipmentType values
                - rated_power_kw (Decimal): Rated power in kilowatts
                - fuel_type (str, optional): Fuel type (default "diesel")
                - load_factor (Decimal, optional): Override load factor (0-1)
                - annual_hours (Decimal, optional): Override annual hours
                - lease_share (Decimal, optional): Fraction leased (default 1.0)
                - quantity (int, optional): Number of units (default 1)

        Returns:
            Dictionary with co2e_kg, energy_kwh, rated_power_kw, load_factor,
            annual_hours, fuel_ef, provenance_hash, dqi_score, uncertainty.

        Raises:
            ValueError: If equipment_type or rated_power_kw is missing.
        """
        start_time = time.monotonic()

        self._validate_equipment_inputs(input_data)

        equipment_type = input_data["equipment_type"].lower()
        rated_power_kw = Decimal(str(input_data["rated_power_kw"]))
        fuel_type = input_data.get("fuel_type", "diesel").lower()
        lease_share = Decimal(str(input_data.get("lease_share", "1.0")))
        quantity = int(input_data.get("quantity", 1))

        # Resolve load factor
        load_factor = Decimal(
            str(
                input_data.get(
                    "load_factor",
                    str(DEFAULT_LOAD_FACTORS.get(equipment_type, Decimal("0.50"))),
                )
            )
        )

        # Resolve annual hours
        annual_hours = Decimal(
            str(
                input_data.get(
                    "annual_hours",
                    str(DEFAULT_ANNUAL_HOURS.get(equipment_type, Decimal("2000"))),
                )
            )
        )

        # Look up fuel emission factor
        fuel_ef = EQUIPMENT_FUEL_EFS.get(fuel_type)
        if fuel_ef is None:
            raise ValueError(
                f"Unsupported equipment fuel type '{fuel_type}'. "
                f"Available: {sorted(EQUIPMENT_FUEL_EFS.keys())}"
            )

        # Core formula: energy_kwh = rated_power x load_factor x hours
        energy_kwh = (rated_power_kw * load_factor * annual_hours).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_per_unit = (energy_kwh * fuel_ef * lease_share).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_kg = (co2e_per_unit * Decimal(str(quantity))).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(input_data)
        uncertainty = self.compute_uncertainty(co2e_kg)

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            equipment_type, str(rated_power_kw), fuel_type,
            str(load_factor), str(annual_hours), str(fuel_ef),
            str(lease_share), str(quantity), str(co2e_kg),
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "Equipment benchmark: type=%s, power=%s kW, load=%s, "
            "hours=%s, fuel=%s, qty=%d, co2e=%s kgCO2e",
            equipment_type, rated_power_kw, load_factor,
            annual_hours, fuel_type, quantity, co2e_kg,
        )

        return {
            "asset_category": AssetCategory.EQUIPMENT.value,
            "equipment_type": equipment_type,
            "rated_power_kw": str(rated_power_kw),
            "load_factor": str(load_factor),
            "annual_hours": str(annual_hours),
            "energy_kwh": str(energy_kwh),
            "fuel_type": fuel_type,
            "fuel_ef_kg_per_kwh": str(fuel_ef),
            "lease_share": str(lease_share),
            "quantity": quantity,
            "co2e_per_unit_kg": str(co2e_per_unit),
            "co2e_kg": co2e_kg,
            "dqi_score": dqi_score,
            "uncertainty_pct": str(uncertainty),
            "uncertainty_lower_kg": str(
                (co2e_kg * (_ONE - uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "uncertainty_upper_kg": str(
                (co2e_kg * (_ONE + uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "provenance_hash": provenance_hash,
        }

    def calculate_it_benchmark(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate IT asset emissions using default power and PUE.

        Formula:
            E = default_power_kw x PUE x hours_per_year x grid_EF
                x lease_share x quantity

        Args:
            input_data: Dictionary containing:
                - it_asset_type (str): One of ITAssetType values
                - region (str, optional): Grid EF region (default GLOBAL_AVERAGE)
                - pue (Decimal, optional): Override PUE (default by facility)
                - facility_type (str, optional): For default PUE lookup
                - power_kw (Decimal, optional): Override default power
                - annual_hours (Decimal, optional): Override default hours
                - lease_share (Decimal, optional): Fraction leased (default 1.0)
                - quantity (int, optional): Number of units (default 1)

        Returns:
            Dictionary with co2e_kg, power_kw, pue, annual_hours, grid_ef,
            energy_kwh, provenance_hash, dqi_score, uncertainty.

        Raises:
            ValueError: If it_asset_type is missing or unsupported.
        """
        start_time = time.monotonic()

        self._validate_it_inputs(input_data)

        it_asset_type = input_data["it_asset_type"].lower()
        region = input_data.get("region", "GLOBAL_AVERAGE")
        lease_share = Decimal(str(input_data.get("lease_share", "1.0")))
        quantity = int(input_data.get("quantity", 1))

        # Resolve power
        power_kw = Decimal(
            str(
                input_data.get(
                    "power_kw",
                    str(IT_ASSET_DEFAULT_POWER.get(it_asset_type, Decimal("0.100"))),
                )
            )
        )

        # Resolve PUE
        facility_type = input_data.get("facility_type", "default")
        pue = Decimal(
            str(
                input_data.get(
                    "pue",
                    str(DEFAULT_PUE.get(facility_type, DEFAULT_PUE["default"])),
                )
            )
        )

        # Resolve annual hours
        annual_hours = Decimal(
            str(
                input_data.get(
                    "annual_hours",
                    str(IT_ASSET_DEFAULT_HOURS.get(it_asset_type, HOURS_PER_YEAR)),
                )
            )
        )

        # Look up grid EF
        grid_ef = self.get_benchmark_ef(region)

        # Core formula
        energy_kwh = (power_kw * pue * annual_hours).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_per_unit = (energy_kwh * grid_ef * lease_share).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_kg = (co2e_per_unit * Decimal(str(quantity))).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(input_data)
        uncertainty = self.compute_uncertainty(co2e_kg)

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            it_asset_type, str(power_kw), str(pue),
            str(annual_hours), str(grid_ef), region,
            str(lease_share), str(quantity), str(co2e_kg),
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "IT benchmark: type=%s, power=%s kW, PUE=%s, hours=%s, "
            "grid_EF=%s, qty=%d, co2e=%s kgCO2e",
            it_asset_type, power_kw, pue, annual_hours,
            grid_ef, quantity, co2e_kg,
        )

        return {
            "asset_category": AssetCategory.IT_ASSET.value,
            "it_asset_type": it_asset_type,
            "power_kw": str(power_kw),
            "pue": str(pue),
            "facility_type": facility_type,
            "annual_hours": str(annual_hours),
            "energy_kwh": str(energy_kwh),
            "grid_ef_kg_per_kwh": str(grid_ef),
            "region": region,
            "lease_share": str(lease_share),
            "quantity": quantity,
            "co2e_per_unit_kg": str(co2e_per_unit),
            "co2e_kg": co2e_kg,
            "dqi_score": dqi_score,
            "uncertainty_pct": str(uncertainty),
            "uncertainty_lower_kg": str(
                (co2e_kg * (_ONE - uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "uncertainty_upper_kg": str(
                (co2e_kg * (_ONE + uncertainty)).quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "provenance_hash": provenance_hash,
        }

    def calculate_batch(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate average-data emissions for a batch of assets.

        Processes each input sequentially, collecting results and logging
        per-record errors without aborting the entire batch.

        Args:
            inputs: List of input dictionaries (each with asset_category
                and category-specific fields).

        Returns:
            List of result dictionaries. Failed records are excluded
            from results and logged at ERROR level.

        Raises:
            ValueError: If inputs list is empty.

        Example:
            >>> results = engine.calculate_batch([
            ...     {"asset_category": "building", "building_type": "office", ...},
            ...     {"asset_category": "vehicle", "vehicle_type": "medium_car", ...},
            ... ])
        """
        if not inputs:
            raise ValueError("Batch inputs list cannot be empty")

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        error_count = 0

        logger.info(
            "Starting average-data batch calculation: %d records", len(inputs)
        )

        for idx, input_data in enumerate(inputs):
            try:
                result = self.calculate(input_data)
                results.append(result)
            except (ValueError, InvalidOperation, KeyError) as e:
                error_count += 1
                logger.error(
                    "Batch record %d failed: %s (category=%s)",
                    idx,
                    str(e),
                    input_data.get("asset_category", "unknown"),
                )

        duration = time.monotonic() - start_time
        self._batch_count += 1

        logger.info(
            "Average-data batch complete: %d/%d succeeded, %d failed, "
            "duration=%.4fs",
            len(results),
            len(inputs),
            error_count,
            duration,
        )

        return results

    def estimate_building_eui(
        self, building_type: str, climate_zone: str
    ) -> Decimal:
        """
        Estimate building EUI from benchmark tables.

        Args:
            building_type: Building type key (e.g., "office", "retail").
            climate_zone: Climate zone key (e.g., "temperate", "cold").

        Returns:
            EUI in kWh/sqm/year as Decimal.

        Raises:
            ValueError: If building_type or climate_zone is not found.

        Example:
            >>> engine.estimate_building_eui("office", "temperate")
            Decimal('200')
        """
        bt_lower = building_type.lower()
        cz_lower = climate_zone.lower()

        bt_data = BUILDING_EUI_BENCHMARKS.get(bt_lower)
        if bt_data is None:
            raise ValueError(
                f"Building type '{building_type}' not found. "
                f"Available: {sorted(BUILDING_EUI_BENCHMARKS.keys())}"
            )

        eui = bt_data.get(cz_lower)
        if eui is None:
            raise ValueError(
                f"Climate zone '{climate_zone}' not found for building type "
                f"'{building_type}'. Available: {sorted(bt_data.keys())}"
            )

        return eui

    def estimate_vehicle_distance(
        self,
        vehicle_type: str,
        override_km: Optional[Any] = None,
    ) -> Decimal:
        """
        Estimate annual vehicle distance from defaults or override.

        Args:
            vehicle_type: Vehicle type key.
            override_km: Optional distance override (Decimal or numeric).

        Returns:
            Annual distance in km as Decimal.

        Raises:
            ValueError: If vehicle_type is not found and no override given.

        Example:
            >>> engine.estimate_vehicle_distance("medium_car")
            Decimal('20000')
        """
        if override_km is not None:
            dist = Decimal(str(override_km))
            if dist <= _ZERO:
                raise ValueError(
                    f"annual_distance_km must be positive, got {dist}"
                )
            return dist

        vt_lower = vehicle_type.lower()
        default_dist = DEFAULT_ANNUAL_DISTANCES.get(vt_lower)
        if default_dist is None:
            raise ValueError(
                f"Vehicle type '{vehicle_type}' not found in defaults. "
                f"Available: {sorted(DEFAULT_ANNUAL_DISTANCES.keys())}. "
                f"Provide annual_distance_km explicitly."
            )

        return default_dist

    def get_benchmark_ef(self, region: str) -> Decimal:
        """
        Get grid emission factor for a region.

        Falls back to GLOBAL_AVERAGE if region is not found.

        Args:
            region: Grid emission factor region key.

        Returns:
            Grid EF in kgCO2e/kWh as Decimal.

        Example:
            >>> engine.get_benchmark_ef("US_AVERAGE")
            Decimal('0.3937')
            >>> engine.get_benchmark_ef("UNKNOWN_REGION")
            Decimal('0.4360')
        """
        ef = GRID_EMISSION_FACTORS.get(region)
        if ef is None:
            logger.warning(
                "Region '%s' not found in grid EFs, falling back to "
                "GLOBAL_AVERAGE (%s kgCO2e/kWh)",
                region,
                GRID_EMISSION_FACTORS["GLOBAL_AVERAGE"],
            )
            ef = GRID_EMISSION_FACTORS["GLOBAL_AVERAGE"]
        return ef

    def compute_dqi_score(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute Data Quality Indicator score for Tier 2 method.

        Uses default Tier 2 DQI scores unless overridden in input_data
        via a 'dqi_overrides' dictionary.

        Args:
            input_data: Optional dict with 'dqi_overrides' key containing
                dimension: score mappings (Decimal or numeric).

        Returns:
            Dictionary with dimension scores, weights, weighted_score,
            composite_score, classification, and tier.

        Example:
            >>> dqi = engine.compute_dqi_score()
            >>> dqi["composite_score"]
            Decimal('3.00000000')
            >>> dqi["classification"]
            'Fair'
        """
        # Start with defaults
        scores = dict(DEFAULT_DQI_SCORES_TIER2)

        # Apply overrides
        if input_data and "dqi_overrides" in input_data:
            overrides = input_data["dqi_overrides"]
            for dim, score in overrides.items():
                if dim in scores:
                    scores[dim] = Decimal(str(score))

        # Compute weighted composite
        weighted_sum = _ZERO
        total_weight = _ZERO
        dimension_results: Dict[str, str] = {}

        for dim, weight in DQI_WEIGHTS_TIER2.items():
            score = scores.get(dim, Decimal("3"))
            weighted_sum += score * weight
            total_weight += weight
            dimension_results[dim] = str(score)

        composite = (weighted_sum / total_weight).quantize(
            _QUANT_8DP, rounding=ROUNDING
        ) if total_weight > _ZERO else Decimal("3.00000000")

        # Classification
        if composite >= Decimal("4.5"):
            classification = "Excellent"
        elif composite >= Decimal("3.5"):
            classification = "Good"
        elif composite >= Decimal("2.5"):
            classification = "Fair"
        elif composite >= Decimal("1.5"):
            classification = "Poor"
        else:
            classification = "Very Poor"

        return {
            "dimensions": dimension_results,
            "weights": {k: str(v) for k, v in DQI_WEIGHTS_TIER2.items()},
            "composite_score": composite,
            "classification": classification,
            "tier": "tier_2",
        }

    def compute_uncertainty(
        self,
        co2e_kg: Decimal,
        custom_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Compute uncertainty range for Tier 2 average-data method.

        Default uncertainty is +/-30% for average-data calculations.

        Args:
            co2e_kg: Calculated emissions in kgCO2e.
            custom_pct: Optional custom uncertainty percentage (Decimal, 0-1).

        Returns:
            Uncertainty as a fraction (e.g., Decimal("0.30") for +/-30%).

        Example:
            >>> engine.compute_uncertainty(Decimal("1000"))
            Decimal('0.30')
        """
        if custom_pct is not None:
            pct = Decimal(str(custom_pct))
            if pct < _ZERO or pct > _ONE:
                raise ValueError(
                    f"custom_pct must be between 0 and 1, got {pct}"
                )
            return pct

        return TIER_2_UNCERTAINTY

    def validate_inputs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data and return validation result.

        Checks all required fields, value ranges, and supported enum values.
        Does NOT raise exceptions; instead returns a structured result.

        Args:
            input_data: Input dictionary to validate.

        Returns:
            Dictionary with is_valid (bool), errors (list of str),
            warnings (list of str).

        Example:
            >>> result = engine.validate_inputs({"asset_category": "building"})
            >>> result["is_valid"]
            False
        """
        errors: List[str] = []
        warnings: List[str] = []

        category = input_data.get("asset_category", "").lower()
        if not category:
            errors.append("asset_category is required")
        elif category not in [c.value for c in AssetCategory]:
            errors.append(
                f"asset_category '{category}' not supported. "
                f"Valid: {[c.value for c in AssetCategory]}"
            )

        if category == AssetCategory.BUILDING.value:
            errors.extend(self._validate_building_fields(input_data))
        elif category == AssetCategory.VEHICLE.value:
            errors.extend(self._validate_vehicle_fields(input_data))
        elif category == AssetCategory.EQUIPMENT.value:
            errors.extend(self._validate_equipment_fields(input_data))
        elif category == AssetCategory.IT_ASSET.value:
            errors.extend(self._validate_it_fields(input_data))

        # Common validations
        lease_share = input_data.get("lease_share")
        if lease_share is not None:
            try:
                ls = Decimal(str(lease_share))
                if ls < _ZERO or ls > _ONE:
                    errors.append(
                        f"lease_share must be between 0 and 1, got {ls}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(f"lease_share must be numeric, got '{lease_share}'")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Return engine health status and statistics.

        Returns:
            Dictionary with engine_id, engine_version, status, stats,
            data_tables_loaded, and initialized_at.

        Example:
            >>> health = engine.health_check()
            >>> health["status"]
            'healthy'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "status": "healthy",
            "stats": {
                "calculation_count": self._calculation_count,
                "batch_count": self._batch_count,
            },
            "data_tables_loaded": {
                "building_eui_benchmarks": len(BUILDING_EUI_BENCHMARKS),
                "climate_zones": len(ClimateZone),
                "grid_emission_factors": len(GRID_EMISSION_FACTORS),
                "vehicle_types": len(VehicleType),
                "vehicle_fuel_types": len(VehicleFuelType),
                "equipment_types": len(EquipmentType),
                "it_asset_types": len(ITAssetType),
                "epc_ratings": len(EPC_RATING_MULTIPLIERS),
                "nabers_ratings": len(NABERS_MULTIPLIERS),
                "energy_star_bands": len(ENERGY_STAR_MULTIPLIERS),
            },
            "initialized_at": self._initialized_at,
        }

    def get_available_building_types(self) -> List[Dict[str, Any]]:
        """
        Return all available building types with climate zone EUI values.

        Returns:
            List of dictionaries with building_type, climate_zones,
            base_load_fraction, and default_occupancy_rate.
        """
        result = []
        for bt in BuildingType:
            eui_data = BUILDING_EUI_BENCHMARKS.get(bt.value, {})
            result.append({
                "building_type": bt.value,
                "climate_zones": {
                    cz: str(eui) for cz, eui in eui_data.items()
                },
                "base_load_fraction": str(
                    BASE_LOAD_FRACTIONS.get(bt.value, Decimal("0.30"))
                ),
                "default_occupancy_rate": str(
                    DEFAULT_OCCUPANCY_RATES.get(bt.value, Decimal("0.80"))
                ),
            })
        return result

    def get_available_vehicle_types(self) -> List[Dict[str, Any]]:
        """
        Return all available vehicle types with EF data.

        Returns:
            List of dictionaries with vehicle_type, default_annual_distance_km,
            and fuel_type emission factors.
        """
        result = []
        for vt in VehicleType:
            ef_data = VEHICLE_EMISSION_FACTORS.get(vt.value, {})
            result.append({
                "vehicle_type": vt.value,
                "default_annual_distance_km": str(
                    DEFAULT_ANNUAL_DISTANCES.get(vt.value, Decimal("0"))
                ),
                "emission_factors_kg_per_km": {
                    ft: str(ef) for ft, ef in ef_data.items()
                },
            })
        return result

    def get_available_equipment_types(self) -> List[Dict[str, Any]]:
        """
        Return all available equipment types with default parameters.

        Returns:
            List of dictionaries with equipment_type, default_annual_hours,
            and default_load_factor.
        """
        result = []
        for et in EquipmentType:
            result.append({
                "equipment_type": et.value,
                "default_annual_hours": str(
                    DEFAULT_ANNUAL_HOURS.get(et.value, Decimal("2000"))
                ),
                "default_load_factor": str(
                    DEFAULT_LOAD_FACTORS.get(et.value, Decimal("0.50"))
                ),
            })
        return result

    def get_available_it_types(self) -> List[Dict[str, Any]]:
        """
        Return all available IT asset types with default parameters.

        Returns:
            List of dictionaries with it_asset_type, default_power_kw,
            and default_annual_hours.
        """
        result = []
        for it in ITAssetType:
            result.append({
                "it_asset_type": it.value,
                "default_power_kw": str(
                    IT_ASSET_DEFAULT_POWER.get(it.value, Decimal("0.100"))
                ),
                "default_annual_hours": str(
                    IT_ASSET_DEFAULT_HOURS.get(it.value, HOURS_PER_YEAR)
                ),
            })
        return result

    def get_available_regions(self) -> List[Dict[str, str]]:
        """
        Return all available grid emission factor regions.

        Returns:
            List of dictionaries with region and ef_kg_per_kwh.
        """
        return [
            {"region": region, "ef_kg_per_kwh": str(ef)}
            for region, ef in sorted(GRID_EMISSION_FACTORS.items())
        ]

    def get_calculation_stats(self) -> Dict[str, Any]:
        """
        Return engine calculation statistics.

        Returns:
            Dictionary with engine_id, engine_version, calculation_count,
            and batch_count.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
        }

    # ==========================================================================
    # Internal Helpers - Building
    # ==========================================================================

    def _validate_building_inputs(self, input_data: Dict[str, Any]) -> None:
        """
        Validate building benchmark inputs (raises on failure).

        Args:
            input_data: Input dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        errors = self._validate_building_fields(input_data)
        if errors:
            raise ValueError(
                f"Building input validation failed: {'; '.join(errors)}"
            )

    def _validate_building_fields(
        self, input_data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate building fields, returning list of error strings.

        Args:
            input_data: Input dictionary.

        Returns:
            List of error strings (empty if valid).
        """
        errors: List[str] = []

        bt = input_data.get("building_type", "").lower()
        if not bt:
            errors.append("building_type is required")
        elif bt not in [b.value for b in BuildingType]:
            errors.append(
                f"building_type '{bt}' not supported. "
                f"Valid: {[b.value for b in BuildingType]}"
            )

        cz = input_data.get("climate_zone", "").lower()
        if not cz:
            errors.append("climate_zone is required")
        elif cz not in [c.value for c in ClimateZone]:
            errors.append(
                f"climate_zone '{cz}' not supported. "
                f"Valid: {[c.value for c in ClimateZone]}"
            )

        fa = input_data.get("floor_area_sqm")
        if fa is None:
            errors.append("floor_area_sqm is required")
        else:
            try:
                fa_dec = Decimal(str(fa))
                if fa_dec <= _ZERO:
                    errors.append(
                        f"floor_area_sqm must be positive, got {fa_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(f"floor_area_sqm must be numeric, got '{fa}'")

        # Optional: vacancy_fraction range check
        vf = input_data.get("vacancy_fraction")
        if vf is not None:
            try:
                vf_dec = Decimal(str(vf))
                if vf_dec < _ZERO or vf_dec > _ONE:
                    errors.append(
                        f"vacancy_fraction must be between 0 and 1, got {vf_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(
                    f"vacancy_fraction must be numeric, got '{vf}'"
                )

        return errors

    def _apply_rating_adjustment(
        self, eui: Decimal, input_data: Dict[str, Any]
    ) -> Decimal:
        """
        Adjust EUI based on EPC, NABERS, or Energy Star rating if provided.

        Only the first available rating is applied (priority: EPC > NABERS >
        Energy Star).

        Args:
            eui: Base EUI from benchmark table.
            input_data: Input dictionary with optional rating fields.

        Returns:
            Adjusted EUI.
        """
        # EPC rating (highest priority)
        epc_rating = input_data.get("epc_rating", "").upper()
        if epc_rating and epc_rating in EPC_RATING_MULTIPLIERS:
            multiplier = EPC_RATING_MULTIPLIERS[epc_rating]
            adjusted = (eui * multiplier).quantize(_QUANT_8DP, rounding=ROUNDING)
            logger.debug(
                "EUI adjusted by EPC rating %s: %s -> %s kWh/sqm/yr",
                epc_rating, eui, adjusted,
            )
            return adjusted

        # NABERS rating
        nabers_rating = input_data.get("nabers_rating", "").lower()
        if nabers_rating and nabers_rating in NABERS_MULTIPLIERS:
            multiplier = NABERS_MULTIPLIERS[nabers_rating]
            adjusted = (eui * multiplier).quantize(_QUANT_8DP, rounding=ROUNDING)
            logger.debug(
                "EUI adjusted by NABERS rating %s: %s -> %s kWh/sqm/yr",
                nabers_rating, eui, adjusted,
            )
            return adjusted

        # Energy Star band
        es_band = input_data.get("energy_star_band", "").lower()
        if es_band and es_band in ENERGY_STAR_MULTIPLIERS:
            multiplier = ENERGY_STAR_MULTIPLIERS[es_band]
            adjusted = (eui * multiplier).quantize(_QUANT_8DP, rounding=ROUNDING)
            logger.debug(
                "EUI adjusted by Energy Star band %s: %s -> %s kWh/sqm/yr",
                es_band, eui, adjusted,
            )
            return adjusted

        return eui

    # ==========================================================================
    # Internal Helpers - Vehicle
    # ==========================================================================

    def _validate_vehicle_inputs(self, input_data: Dict[str, Any]) -> None:
        """
        Validate vehicle benchmark inputs (raises on failure).

        Args:
            input_data: Input dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        errors = self._validate_vehicle_fields(input_data)
        if errors:
            raise ValueError(
                f"Vehicle input validation failed: {'; '.join(errors)}"
            )

    def _validate_vehicle_fields(
        self, input_data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate vehicle fields, returning list of error strings.

        Args:
            input_data: Input dictionary.

        Returns:
            List of error strings.
        """
        errors: List[str] = []

        vt = input_data.get("vehicle_type", "").lower()
        if not vt:
            errors.append("vehicle_type is required")
        elif vt not in [v.value for v in VehicleType]:
            errors.append(
                f"vehicle_type '{vt}' not supported. "
                f"Valid: {[v.value for v in VehicleType]}"
            )

        ft = input_data.get("fuel_type", VehicleFuelType.PETROL.value).lower()
        if ft not in [f.value for f in VehicleFuelType]:
            errors.append(
                f"fuel_type '{ft}' not supported. "
                f"Valid: {[f.value for f in VehicleFuelType]}"
            )

        ad = input_data.get("annual_distance_km")
        if ad is not None:
            try:
                ad_dec = Decimal(str(ad))
                if ad_dec <= _ZERO:
                    errors.append(
                        f"annual_distance_km must be positive, got {ad_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(
                    f"annual_distance_km must be numeric, got '{ad}'"
                )

        fc = input_data.get("fleet_count")
        if fc is not None:
            try:
                fc_int = int(fc)
                if fc_int < 1:
                    errors.append(
                        f"fleet_count must be >= 1, got {fc_int}"
                    )
            except (ValueError, TypeError):
                errors.append(f"fleet_count must be integer, got '{fc}'")

        return errors

    def _lookup_vehicle_ef(
        self, vehicle_type: str, fuel_type: str
    ) -> Decimal:
        """
        Look up vehicle emission factor by type and fuel.

        Args:
            vehicle_type: Vehicle type key.
            fuel_type: Fuel type key.

        Returns:
            Emission factor in kgCO2e/km.

        Raises:
            ValueError: If vehicle_type or fuel_type is not found.
        """
        vt_data = VEHICLE_EMISSION_FACTORS.get(vehicle_type)
        if vt_data is None:
            raise ValueError(
                f"Vehicle type '{vehicle_type}' not found in EF table. "
                f"Available: {sorted(VEHICLE_EMISSION_FACTORS.keys())}"
            )

        ef = vt_data.get(fuel_type)
        if ef is None:
            raise ValueError(
                f"Fuel type '{fuel_type}' not found for vehicle "
                f"'{vehicle_type}'. Available: {sorted(vt_data.keys())}"
            )

        return ef

    # ==========================================================================
    # Internal Helpers - Equipment
    # ==========================================================================

    def _validate_equipment_inputs(self, input_data: Dict[str, Any]) -> None:
        """
        Validate equipment benchmark inputs (raises on failure).

        Args:
            input_data: Input dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        errors = self._validate_equipment_fields(input_data)
        if errors:
            raise ValueError(
                f"Equipment input validation failed: {'; '.join(errors)}"
            )

    def _validate_equipment_fields(
        self, input_data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate equipment fields, returning list of error strings.

        Args:
            input_data: Input dictionary.

        Returns:
            List of error strings.
        """
        errors: List[str] = []

        et = input_data.get("equipment_type", "").lower()
        if not et:
            errors.append("equipment_type is required")
        elif et not in [e.value for e in EquipmentType]:
            errors.append(
                f"equipment_type '{et}' not supported. "
                f"Valid: {[e.value for e in EquipmentType]}"
            )

        rp = input_data.get("rated_power_kw")
        if rp is None:
            errors.append("rated_power_kw is required")
        else:
            try:
                rp_dec = Decimal(str(rp))
                if rp_dec <= _ZERO:
                    errors.append(
                        f"rated_power_kw must be positive, got {rp_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(
                    f"rated_power_kw must be numeric, got '{rp}'"
                )

        ft = input_data.get("fuel_type", "diesel").lower()
        if ft not in EQUIPMENT_FUEL_EFS:
            errors.append(
                f"fuel_type '{ft}' not supported for equipment. "
                f"Available: {sorted(EQUIPMENT_FUEL_EFS.keys())}"
            )

        return errors

    # ==========================================================================
    # Internal Helpers - IT
    # ==========================================================================

    def _validate_it_inputs(self, input_data: Dict[str, Any]) -> None:
        """
        Validate IT asset benchmark inputs (raises on failure).

        Args:
            input_data: Input dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        errors = self._validate_it_fields(input_data)
        if errors:
            raise ValueError(
                f"IT asset input validation failed: {'; '.join(errors)}"
            )

    def _validate_it_fields(
        self, input_data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate IT asset fields, returning list of error strings.

        Args:
            input_data: Input dictionary.

        Returns:
            List of error strings.
        """
        errors: List[str] = []

        it = input_data.get("it_asset_type", "").lower()
        if not it:
            errors.append("it_asset_type is required")
        elif it not in [i.value for i in ITAssetType]:
            errors.append(
                f"it_asset_type '{it}' not supported. "
                f"Valid: {[i.value for i in ITAssetType]}"
            )

        pk = input_data.get("power_kw")
        if pk is not None:
            try:
                pk_dec = Decimal(str(pk))
                if pk_dec <= _ZERO:
                    errors.append(
                        f"power_kw must be positive, got {pk_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(f"power_kw must be numeric, got '{pk}'")

        pue_val = input_data.get("pue")
        if pue_val is not None:
            try:
                pue_dec = Decimal(str(pue_val))
                if pue_dec < _ONE:
                    errors.append(
                        f"pue must be >= 1.0, got {pue_dec}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(f"pue must be numeric, got '{pue_val}'")

        qty = input_data.get("quantity")
        if qty is not None:
            try:
                qty_int = int(qty)
                if qty_int < 1:
                    errors.append(f"quantity must be >= 1, got {qty_int}")
            except (ValueError, TypeError):
                errors.append(f"quantity must be integer, got '{qty}'")

        return errors


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_average_data_calculator() -> AverageDataCalculatorEngine:
    """
    Get the AverageDataCalculatorEngine singleton instance.

    Convenience function that delegates to the class-level get_instance().

    Returns:
        AverageDataCalculatorEngine singleton.

    Example:
        >>> engine = get_average_data_calculator()
        >>> engine.health_check()["status"]
        'healthy'
    """
    return AverageDataCalculatorEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "TIER_2_UNCERTAINTY",
    # Enums
    "BuildingType",
    "ClimateZone",
    "VehicleType",
    "VehicleFuelType",
    "EquipmentType",
    "ITAssetType",
    "EPCRating",
    "AssetCategory",
    # Benchmark data
    "BUILDING_EUI_BENCHMARKS",
    "EPC_RATING_MULTIPLIERS",
    "BASE_LOAD_FRACTIONS",
    "DEFAULT_OCCUPANCY_RATES",
    "GRID_EMISSION_FACTORS",
    "DEFAULT_ANNUAL_DISTANCES",
    "VEHICLE_EMISSION_FACTORS",
    "DEFAULT_ANNUAL_HOURS",
    "DEFAULT_LOAD_FACTORS",
    "EQUIPMENT_FUEL_EFS",
    "IT_ASSET_DEFAULT_POWER",
    "DEFAULT_PUE",
    "IT_ASSET_DEFAULT_HOURS",
    "NABERS_MULTIPLIERS",
    "ENERGY_STAR_MULTIPLIERS",
    # Engine
    "AverageDataCalculatorEngine",
    "get_average_data_calculator",
]
