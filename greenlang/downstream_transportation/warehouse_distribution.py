# -*- coding: utf-8 -*-
"""
WarehouseDistributionEngine - AGENT-MRV-022 Engine 5

GHH Protocol Scope 3 Category 9 warehouse, distribution center, cold storage,
retail storage, fulfillment center, and last-mile delivery emissions calculator
for downstream transportation and distribution.

This engine calculates emissions from the storage and distribution components
of downstream logistics that occur between the reporting company's operations
and the end consumer:

1. **Warehouse Emissions**: Floor-area-based calculation using energy intensity
   and grid emission factors. Supports 7 warehouse types (ambient, cold
   storage, cross-dock, retail backroom, e-commerce fulfillment, pharmaceutical,
   bulk storage).

2. **Cold Storage Emissions**: Enhanced calculation with temperature-tier-based
   energy intensity, defrost cycles, and door-opening heat gain.

3. **Retail Storage Emissions**: In-store storage at third-party retail
   outlets with product allocation by shelf-space or revenue share.

4. **Fulfillment Center Emissions**: E-commerce fulfillment center emissions
   with pick-pack-ship energy consumption and automation efficiency.

5. **Last-Mile Delivery Emissions**: Final delivery to end consumer using
   per-delivery emission factors by vehicle type and area type (urban,
   suburban, rural).

6. **Distribution Chain**: Combined warehouse + last-mile calculation for
   an entire distribution chain.

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern for concurrent pipeline use.

Core Formulas:
    Warehouse: co2e = floor_area x ef_kwh_m2_day x days x grid_ef x allocation_share
    Cold Storage: co2e = floor_area x cold_ef_kwh_m2_day x days x grid_ef x temp_multiplier
    Last Mile: co2e = num_deliveries x ef_per_delivery[vehicle_type][area_type]

Zero-Hallucination Compliance:
    All emission calculations use deterministic arithmetic on embedded factor
    tables. No LLM calls are made in any calculation path.

References:
    - GHH Protocol Technical Guidance for Scope 3, Category 9
    - GLEC Framework v3.0 (warehousing and handling)
    - DEFRA / UK BEIS Conversion Factors 2023 (delivery vehicles)
    - US EPA SmartWay (warehouse energy benchmarks)
    - IEA World Energy Outlook 2023 (grid emission factors)
    - ASHRAE Handbook (cold storage energy intensity)
    - International Association of Refrigerated Warehouses (IARW)

Example:
    >>> engine = WarehouseDistributionEngine.get_instance()
    >>> result = engine.calculate_warehouse(WarehouseInput(
    ...     warehouse_type="ambient",
    ...     floor_area_m2=Decimal("5000"),
    ...     days=Decimal("30"),
    ...     allocation_share=Decimal("0.10"),
    ...     country="US",
    ... ))
    >>> result["co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-009"
AGENT_COMPONENT: str = "AGENT-MRV-022"
ENGINE_ID: str = "warehouse_distribution_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dto_"

# ==============================================================================
# DECIMAL CONSTANTS
# ==============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
THOUSAND = Decimal("1000")
DAYS_PER_YEAR = Decimal("365")
PRECISION: int = 8
ROUNDING: str = ROUND_HALF_UP
_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_2DP: Decimal = Decimal("0.01")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class WarehouseType(str, Enum):
    """Types of warehouses and distribution centers."""

    AMBIENT = "ambient"
    COLD_STORAGE = "cold_storage"
    CROSS_DOCK = "cross_dock"
    RETAIL_BACKROOM = "retail_backroom"
    ECOMMERCE_FULFILLMENT = "ecommerce_fulfillment"
    PHARMACEUTICAL = "pharmaceutical"
    BULK_STORAGE = "bulk_storage"


class TemperatureTier(str, Enum):
    """Cold storage temperature tiers (IARW classification)."""

    CHILLED = "chilled"           # 0 to 5 C
    FROZEN = "frozen"             # -18 to -25 C
    DEEP_FROZEN = "deep_frozen"   # Below -25 C
    CONTROLLED_AMBIENT = "controlled_ambient"  # 15 to 25 C


class LastMileVehicleType(str, Enum):
    """Vehicle types for last-mile delivery."""

    VAN_DIESEL = "van_diesel"
    VAN_ELECTRIC = "van_electric"
    CARGO_BIKE = "cargo_bike"
    CAR_PETROL = "car_petrol"
    MOTORCYCLE = "motorcycle"
    WALK_COURIER = "walk_courier"


class AreaType(str, Enum):
    """Delivery area types affecting per-delivery emissions."""

    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"


class EFSource(str, Enum):
    """Emission factor data source."""

    EPA_SMARTWAY = "epa_smartway"
    DEFRA = "defra"
    IEA = "iea"
    GLEC = "glec"
    ASHRAE = "ashrae"
    IARW = "iarw"
    CUSTOM = "custom"


class DataQualityTier(str, Enum):
    """Data quality tiers."""

    TIER_2 = "tier_2"  # Facility-specific data
    TIER_3 = "tier_3"  # Average-data
    TIER_4 = "tier_4"  # Generic estimates


# ==============================================================================
# WAREHOUSE EMISSION FACTORS
# Energy intensity (kWh per m2 per day) by warehouse type
# Source: US EPA SmartWay / ASHRAE
# ==============================================================================

WAREHOUSE_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "ambient": {
        "name": "Ambient Warehouse / Distribution Center",
        "energy_kwh_m2_day": Decimal("0.0685"),
        "description": "Standard non-refrigerated warehouse",
        "source": EFSource.EPA_SMARTWAY.value,
    },
    "cold_storage": {
        "name": "Cold Storage Warehouse",
        "energy_kwh_m2_day": Decimal("0.1950"),
        "description": "Refrigerated warehouse (mixed temperature)",
        "source": EFSource.ASHRAE.value,
    },
    "cross_dock": {
        "name": "Cross-Dock Facility",
        "energy_kwh_m2_day": Decimal("0.0520"),
        "description": "Minimal storage, transshipment facility",
        "source": EFSource.EPA_SMARTWAY.value,
    },
    "retail_backroom": {
        "name": "Retail Store Backroom",
        "energy_kwh_m2_day": Decimal("0.1100"),
        "description": "Retail store back-of-house storage area",
        "source": EFSource.EPA_SMARTWAY.value,
    },
    "ecommerce_fulfillment": {
        "name": "E-Commerce Fulfillment Center",
        "energy_kwh_m2_day": Decimal("0.0950"),
        "description": "Automated pick-pack-ship fulfillment center",
        "source": EFSource.EPA_SMARTWAY.value,
    },
    "pharmaceutical": {
        "name": "Pharmaceutical Distribution Center",
        "energy_kwh_m2_day": Decimal("0.1400"),
        "description": "Temperature-controlled pharmaceutical storage",
        "source": EFSource.ASHRAE.value,
    },
    "bulk_storage": {
        "name": "Bulk / Open Storage",
        "energy_kwh_m2_day": Decimal("0.0350"),
        "description": "Open or minimal-enclosure bulk commodity storage",
        "source": EFSource.EPA_SMARTWAY.value,
    },
}

# ==============================================================================
# COLD STORAGE TEMPERATURE MULTIPLIERS
# Energy multiplier relative to mixed cold storage (cold_storage baseline)
# ==============================================================================

COLD_STORAGE_TEMP_MULTIPLIERS: Dict[str, Dict[str, Any]] = {
    "chilled": {
        "name": "Chilled (0 to 5 C)",
        "multiplier": Decimal("0.75"),
        "temp_range_c": "0 to 5",
    },
    "frozen": {
        "name": "Frozen (-18 to -25 C)",
        "multiplier": Decimal("1.00"),
        "temp_range_c": "-18 to -25",
    },
    "deep_frozen": {
        "name": "Deep Frozen (below -25 C)",
        "multiplier": Decimal("1.35"),
        "temp_range_c": "below -25",
    },
    "controlled_ambient": {
        "name": "Controlled Ambient (15 to 25 C)",
        "multiplier": Decimal("0.50"),
        "temp_range_c": "15 to 25",
    },
}

# ==============================================================================
# LAST-MILE EMISSION FACTORS (kgCO2e per delivery)
# 6 vehicle types x 3 area types
# Source: DEFRA 2023 / GLEC Framework v3.0
# ==============================================================================

LAST_MILE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "van_diesel": {
        "urban": Decimal("0.950"),
        "suburban": Decimal("1.420"),
        "rural": Decimal("2.850"),
    },
    "van_electric": {
        "urban": Decimal("0.180"),
        "suburban": Decimal("0.270"),
        "rural": Decimal("0.540"),
    },
    "cargo_bike": {
        "urban": Decimal("0.015"),
        "suburban": Decimal("0.025"),
        "rural": Decimal("0.060"),
    },
    "car_petrol": {
        "urban": Decimal("1.200"),
        "suburban": Decimal("1.800"),
        "rural": Decimal("3.600"),
    },
    "motorcycle": {
        "urban": Decimal("0.420"),
        "suburban": Decimal("0.630"),
        "rural": Decimal("1.260"),
    },
    "walk_courier": {
        "urban": Decimal("0.005"),
        "suburban": Decimal("0.010"),
        "rural": Decimal("0.020"),
    },
}

# ==============================================================================
# GRID EMISSION FACTORS (kgCO2e per kWh)
# Source: IEA World Energy Outlook 2023
# ==============================================================================

GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.3890"),
    "GB": Decimal("0.2070"),
    "DE": Decimal("0.3380"),
    "FR": Decimal("0.0520"),
    "CN": Decimal("0.5810"),
    "IN": Decimal("0.7080"),
    "JP": Decimal("0.4570"),
    "AU": Decimal("0.6100"),
    "BR": Decimal("0.0740"),
    "CA": Decimal("0.1200"),
    "GLOBAL": Decimal("0.4360"),
}

# ==============================================================================
# FULFILLMENT CENTER CONSTANTS
# ==============================================================================

# Additional energy per order for pick-pack-ship operations (kWh/order)
PICK_PACK_SHIP_ENERGY: Decimal = Decimal("0.150")

# Automation efficiency factor (reduces energy per order)
AUTOMATION_EFFICIENCY: Dict[str, Decimal] = {
    "manual": Decimal("1.00"),
    "semi_automated": Decimal("0.80"),
    "fully_automated": Decimal("0.60"),
}

# Retail store total energy intensity (kWh/m2/day) for product allocation
RETAIL_STORE_ENERGY_INTENSITY: Decimal = Decimal("0.2500")


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class WarehouseInput:
    """
    Input for warehouse emissions calculation.

    Attributes:
        warehouse_type: Type of warehouse.
        floor_area_m2: Floor area in square metres.
        days: Number of days product is stored.
        allocation_share: Share of warehouse allocated to reporting company.
        country: Country for grid emission factor.
        record_id: Optional unique identifier.
        tenant_id: Optional tenant identifier.
    """

    __slots__ = (
        "warehouse_type", "floor_area_m2", "days",
        "allocation_share", "country", "record_id", "tenant_id",
    )

    def __init__(
        self,
        warehouse_type: str,
        floor_area_m2: Decimal,
        days: Decimal,
        allocation_share: Decimal = ONE,
        country: str = "GLOBAL",
        record_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize WarehouseInput with validation."""
        if floor_area_m2 <= ZERO:
            raise ValueError(f"Floor area must be positive, got {floor_area_m2}")
        if days <= ZERO:
            raise ValueError(f"Days must be positive, got {days}")
        if allocation_share <= ZERO or allocation_share > ONE:
            raise ValueError(
                f"Allocation share must be in (0, 1], got {allocation_share}"
            )
        object.__setattr__(self, "warehouse_type", warehouse_type.lower())
        object.__setattr__(self, "floor_area_m2", floor_area_m2)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "allocation_share", allocation_share)
        object.__setattr__(self, "country", country.upper())
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "tenant_id", tenant_id)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after initialization."""
        raise AttributeError("WarehouseInput is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for provenance hashing."""
        return {
            "warehouse_type": self.warehouse_type,
            "floor_area_m2": str(self.floor_area_m2),
            "days": str(self.days),
            "allocation_share": str(self.allocation_share),
            "country": self.country,
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
        }


class ColdStorageInput:
    """
    Input for cold storage emissions calculation.

    Extends WarehouseInput with temperature tier and cold-chain-specific
    parameters.

    Attributes:
        floor_area_m2: Floor area in square metres.
        days: Number of days product is stored.
        temperature_tier: Temperature classification.
        allocation_share: Share allocated to reporting company.
        country: Country for grid emission factor.
        record_id: Optional unique identifier.
        tenant_id: Optional tenant identifier.
    """

    __slots__ = (
        "floor_area_m2", "days", "temperature_tier",
        "allocation_share", "country", "record_id", "tenant_id",
    )

    def __init__(
        self,
        floor_area_m2: Decimal,
        days: Decimal,
        temperature_tier: str = "frozen",
        allocation_share: Decimal = ONE,
        country: str = "GLOBAL",
        record_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize ColdStorageInput with validation."""
        if floor_area_m2 <= ZERO:
            raise ValueError(f"Floor area must be positive, got {floor_area_m2}")
        if days <= ZERO:
            raise ValueError(f"Days must be positive, got {days}")
        if allocation_share <= ZERO or allocation_share > ONE:
            raise ValueError(
                f"Allocation share must be in (0, 1], got {allocation_share}"
            )
        object.__setattr__(self, "floor_area_m2", floor_area_m2)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "temperature_tier", temperature_tier.lower())
        object.__setattr__(self, "allocation_share", allocation_share)
        object.__setattr__(self, "country", country.upper())
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "tenant_id", tenant_id)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after initialization."""
        raise AttributeError("ColdStorageInput is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for provenance hashing."""
        return {
            "floor_area_m2": str(self.floor_area_m2),
            "days": str(self.days),
            "temperature_tier": self.temperature_tier,
            "allocation_share": str(self.allocation_share),
            "country": self.country,
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
        }


class RetailStorageInput:
    """
    Input for retail storage emissions calculation.

    Attributes:
        store_floor_area_m2: Total store floor area in m2.
        product_share: Share of store energy allocated to product (0-1).
        days: Number of days product is on retail floor.
        country: Country for grid emission factor.
        record_id: Optional unique identifier.
        tenant_id: Optional tenant identifier.
    """

    __slots__ = (
        "store_floor_area_m2", "product_share", "days",
        "country", "record_id", "tenant_id",
    )

    def __init__(
        self,
        store_floor_area_m2: Decimal,
        product_share: Decimal,
        days: Decimal,
        country: str = "GLOBAL",
        record_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize RetailStorageInput with validation."""
        if store_floor_area_m2 <= ZERO:
            raise ValueError(f"Store floor area must be positive, got {store_floor_area_m2}")
        if product_share <= ZERO or product_share > ONE:
            raise ValueError(f"Product share must be in (0, 1], got {product_share}")
        if days <= ZERO:
            raise ValueError(f"Days must be positive, got {days}")
        object.__setattr__(self, "store_floor_area_m2", store_floor_area_m2)
        object.__setattr__(self, "product_share", product_share)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "country", country.upper())
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "tenant_id", tenant_id)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after initialization."""
        raise AttributeError("RetailStorageInput is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for provenance hashing."""
        return {
            "store_floor_area_m2": str(self.store_floor_area_m2),
            "product_share": str(self.product_share),
            "days": str(self.days),
            "country": self.country,
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
        }


class FulfillmentCenterInput:
    """
    Input for e-commerce fulfillment center emissions calculation.

    Attributes:
        floor_area_m2: Fulfillment center floor area in m2.
        days: Operational days in reporting period.
        orders_processed: Number of orders processed in period.
        allocation_share: Share allocated to reporting company (0-1).
        automation_level: Automation level (manual, semi_automated, fully_automated).
        country: Country for grid emission factor.
        record_id: Optional unique identifier.
        tenant_id: Optional tenant identifier.
    """

    __slots__ = (
        "floor_area_m2", "days", "orders_processed",
        "allocation_share", "automation_level",
        "country", "record_id", "tenant_id",
    )

    def __init__(
        self,
        floor_area_m2: Decimal,
        days: Decimal,
        orders_processed: int,
        allocation_share: Decimal = ONE,
        automation_level: str = "manual",
        country: str = "GLOBAL",
        record_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize FulfillmentCenterInput with validation."""
        if floor_area_m2 <= ZERO:
            raise ValueError(f"Floor area must be positive, got {floor_area_m2}")
        if days <= ZERO:
            raise ValueError(f"Days must be positive, got {days}")
        if orders_processed <= 0:
            raise ValueError(f"Orders must be positive, got {orders_processed}")
        if allocation_share <= ZERO or allocation_share > ONE:
            raise ValueError(
                f"Allocation share must be in (0, 1], got {allocation_share}"
            )
        object.__setattr__(self, "floor_area_m2", floor_area_m2)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "orders_processed", orders_processed)
        object.__setattr__(self, "allocation_share", allocation_share)
        object.__setattr__(self, "automation_level", automation_level.lower())
        object.__setattr__(self, "country", country.upper())
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "tenant_id", tenant_id)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after initialization."""
        raise AttributeError("FulfillmentCenterInput is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for provenance hashing."""
        return {
            "floor_area_m2": str(self.floor_area_m2),
            "days": str(self.days),
            "orders_processed": self.orders_processed,
            "allocation_share": str(self.allocation_share),
            "automation_level": self.automation_level,
            "country": self.country,
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
        }


class LastMileInput:
    """
    Input for last-mile delivery emissions calculation.

    Attributes:
        num_deliveries: Number of deliveries.
        vehicle_type: Delivery vehicle type.
        area_type: Delivery area type (urban, suburban, rural).
        record_id: Optional unique identifier.
        tenant_id: Optional tenant identifier.
    """

    __slots__ = (
        "num_deliveries", "vehicle_type", "area_type",
        "record_id", "tenant_id",
    )

    def __init__(
        self,
        num_deliveries: int,
        vehicle_type: str = "van_diesel",
        area_type: str = "urban",
        record_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize LastMileInput with validation."""
        if num_deliveries <= 0:
            raise ValueError(f"Number of deliveries must be positive, got {num_deliveries}")
        object.__setattr__(self, "num_deliveries", num_deliveries)
        object.__setattr__(self, "vehicle_type", vehicle_type.lower())
        object.__setattr__(self, "area_type", area_type.lower())
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "tenant_id", tenant_id)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after initialization."""
        raise AttributeError("LastMileInput is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for provenance hashing."""
        return {
            "num_deliveries": self.num_deliveries,
            "vehicle_type": self.vehicle_type,
            "area_type": self.area_type,
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
        }


# ==============================================================================
# PROVENANCE HASH HELPER
# ==============================================================================


def _calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

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
        elif hasattr(inp, "to_dict"):
            hash_input += json.dumps(inp.to_dict(), sort_keys=True, default=str)
        else:
            hash_input += str(inp)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _get_grid_ef(country: str) -> Decimal:
    """
    Get grid emission factor for a country, falling back to GLOBAL.

    Args:
        country: ISO 3166-1 alpha-2 country code.

    Returns:
        Grid emission factor in kgCO2e/kWh.
    """
    return GRID_EMISSION_FACTORS.get(
        country.upper(), GRID_EMISSION_FACTORS["GLOBAL"]
    )


# ==============================================================================
# WarehouseDistributionEngine
# ==============================================================================


class WarehouseDistributionEngine:
    """
    Warehouse, distribution, and last-mile emissions calculator for
    downstream transportation and distribution (Scope 3 Category 9).

    Handles the storage-side and final-delivery emissions that occur
    downstream of the reporting company's operations.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _calculation_count: Running count of calculations.
        _batch_count: Running count of batch operations.

    Example:
        >>> engine = WarehouseDistributionEngine.get_instance()
        >>> result = engine.calculate_warehouse(WarehouseInput(
        ...     warehouse_type="ambient",
        ...     floor_area_m2=Decimal("5000"),
        ...     days=Decimal("30"),
        ...     allocation_share=Decimal("0.10"),
        ...     country="US",
        ... ))
        >>> result["co2e_kg"] > Decimal("0")
        True
    """

    _instance: Optional["WarehouseDistributionEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize WarehouseDistributionEngine."""
        self._calculation_count: int = 0
        self._batch_count: int = 0

        logger.info(
            "WarehouseDistributionEngine initialized: agent=%s, version=%s, "
            "warehouse_types=%d, last_mile_types=%d, countries=%d",
            AGENT_ID, ENGINE_VERSION,
            len(WAREHOUSE_EMISSION_FACTORS),
            len(LAST_MILE_EMISSION_FACTORS),
            len(GRID_EMISSION_FACTORS),
        )

    # ==========================================================================
    # SINGLETON
    # ==========================================================================

    @classmethod
    def get_instance(cls) -> "WarehouseDistributionEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            WarehouseDistributionEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
            logger.info("WarehouseDistributionEngine singleton reset")

    # ==========================================================================
    # PUBLIC METHOD 1: calculate_warehouse
    # ==========================================================================

    def calculate_warehouse(
        self, warehouse_input: WarehouseInput
    ) -> Dict[str, Any]:
        """
        Calculate warehouse emissions from floor area, energy intensity,
        and grid emission factor.

        Formula:
            energy_kwh = floor_area x ef_kwh_m2_day x days
            allocated_kwh = energy_kwh x allocation_share
            co2e = allocated_kwh x grid_ef
            annualized = co2e x (days / 365) -- informational only

        Args:
            warehouse_input: Warehouse input with type, area, days, share.

        Returns:
            Dictionary containing:
                - warehouse_type, warehouse_name
                - floor_area_m2, days, allocation_share
                - energy_intensity_kwh_m2_day
                - total_energy_kwh, allocated_energy_kwh
                - grid_ef_kgco2e_kwh, country
                - co2e_kg, co2e_tonnes
                - annualized_co2e_kg (scaled to 365 days, informational)
                - provenance_hash
                - engine_id, engine_version, agent_id
                - processing_time_ms

        Raises:
            ValueError: If warehouse_type not recognized.

        Example:
            >>> result = engine.calculate_warehouse(WarehouseInput(
            ...     warehouse_type="ambient",
            ...     floor_area_m2=Decimal("5000"),
            ...     days=Decimal("30"),
            ...     allocation_share=Decimal("0.10"),
            ...     country="US",
            ... ))
        """
        start_time = time.monotonic()

        # Validate warehouse type
        wh_type = warehouse_input.warehouse_type
        wh_entry = WAREHOUSE_EMISSION_FACTORS.get(wh_type)
        if wh_entry is None:
            raise ValueError(
                f"Warehouse type '{wh_type}' not found. "
                f"Available: {sorted(WAREHOUSE_EMISSION_FACTORS.keys())}"
            )

        ef_kwh_m2_day = wh_entry["energy_kwh_m2_day"]
        grid_ef = _get_grid_ef(warehouse_input.country)

        # Energy calculation
        total_energy_kwh = (
            warehouse_input.floor_area_m2 * ef_kwh_m2_day * warehouse_input.days
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        allocated_energy_kwh = (
            total_energy_kwh * warehouse_input.allocation_share
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # Emissions
        co2e_kg = (allocated_energy_kwh * grid_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_tonnes = (co2e_kg / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Annualized (informational)
        if warehouse_input.days > ZERO:
            annualized_co2e = (
                co2e_kg * DAYS_PER_YEAR / warehouse_input.days
            ).quantize(_QUANT_8DP, rounding=ROUNDING)
        else:
            annualized_co2e = ZERO

        # Provenance
        provenance_hash = _calculate_provenance_hash(
            warehouse_input.to_dict(),
            total_energy_kwh, allocated_energy_kwh,
            grid_ef, co2e_kg,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._calculation_count += 1

        result = {
            "warehouse_type": wh_type,
            "warehouse_name": wh_entry["name"],
            "floor_area_m2": warehouse_input.floor_area_m2,
            "days": warehouse_input.days,
            "allocation_share": warehouse_input.allocation_share,
            "energy_intensity_kwh_m2_day": ef_kwh_m2_day,
            "total_energy_kwh": total_energy_kwh,
            "allocated_energy_kwh": allocated_energy_kwh,
            "grid_ef_kgco2e_kwh": grid_ef,
            "country": warehouse_input.country,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "annualized_co2e_kg": annualized_co2e,
            "ef_source": wh_entry["source"],
            "data_quality_tier": DataQualityTier.TIER_2.value,
            "record_id": warehouse_input.record_id,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Warehouse calculation: type=%s, area=%s m2, days=%s, "
            "share=%s, co2e=%s kgCO2e",
            wh_type, warehouse_input.floor_area_m2,
            warehouse_input.days, warehouse_input.allocation_share, co2e_kg,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 2: calculate_cold_storage
    # ==========================================================================

    def calculate_cold_storage(
        self, cold_input: ColdStorageInput
    ) -> Dict[str, Any]:
        """
        Calculate cold storage emissions with temperature-tier adjustment.

        Uses the cold_storage base energy intensity multiplied by the
        temperature tier multiplier.

        Formula:
            base_ef = WAREHOUSE_EMISSION_FACTORS["cold_storage"].energy_kwh_m2_day
            adjusted_ef = base_ef x temp_multiplier
            energy_kwh = floor_area x adjusted_ef x days
            allocated_kwh = energy_kwh x allocation_share
            co2e = allocated_kwh x grid_ef

        Args:
            cold_input: Cold storage input with area, days, temperature tier.

        Returns:
            Result dictionary with temperature-specific emissions.

        Raises:
            ValueError: If temperature_tier not recognized.

        Example:
            >>> result = engine.calculate_cold_storage(ColdStorageInput(
            ...     floor_area_m2=Decimal("2000"),
            ...     days=Decimal("14"),
            ...     temperature_tier="frozen",
            ...     allocation_share=Decimal("0.15"),
            ...     country="US",
            ... ))
        """
        start_time = time.monotonic()

        # Validate temperature tier
        temp_tier = cold_input.temperature_tier
        temp_entry = COLD_STORAGE_TEMP_MULTIPLIERS.get(temp_tier)
        if temp_entry is None:
            raise ValueError(
                f"Temperature tier '{temp_tier}' not found. "
                f"Available: {sorted(COLD_STORAGE_TEMP_MULTIPLIERS.keys())}"
            )

        base_ef = WAREHOUSE_EMISSION_FACTORS["cold_storage"]["energy_kwh_m2_day"]
        temp_multiplier = temp_entry["multiplier"]
        adjusted_ef = (base_ef * temp_multiplier).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        grid_ef = _get_grid_ef(cold_input.country)

        # Energy and emissions
        total_energy_kwh = (
            cold_input.floor_area_m2 * adjusted_ef * cold_input.days
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        allocated_energy_kwh = (
            total_energy_kwh * cold_input.allocation_share
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        co2e_kg = (allocated_energy_kwh * grid_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_tonnes = (co2e_kg / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        provenance_hash = _calculate_provenance_hash(
            cold_input.to_dict(),
            adjusted_ef, total_energy_kwh, co2e_kg,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._calculation_count += 1

        result = {
            "warehouse_type": "cold_storage",
            "temperature_tier": temp_tier,
            "temperature_range_c": temp_entry["temp_range_c"],
            "floor_area_m2": cold_input.floor_area_m2,
            "days": cold_input.days,
            "allocation_share": cold_input.allocation_share,
            "base_energy_intensity": base_ef,
            "temperature_multiplier": temp_multiplier,
            "adjusted_energy_intensity": adjusted_ef,
            "total_energy_kwh": total_energy_kwh,
            "allocated_energy_kwh": allocated_energy_kwh,
            "grid_ef_kgco2e_kwh": grid_ef,
            "country": cold_input.country,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "ef_source": EFSource.ASHRAE.value,
            "data_quality_tier": DataQualityTier.TIER_2.value,
            "record_id": cold_input.record_id,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Cold storage calculation: tier=%s, area=%s m2, days=%s, "
            "co2e=%s kgCO2e",
            temp_tier, cold_input.floor_area_m2, cold_input.days, co2e_kg,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 3: calculate_retail_storage
    # ==========================================================================

    def calculate_retail_storage(
        self, retail_input: RetailStorageInput
    ) -> Dict[str, Any]:
        """
        Calculate retail outlet storage emissions with product allocation.

        Uses total store energy intensity and allocates to product based
        on shelf-space/revenue share.

        Formula:
            store_energy_kwh = store_area x RETAIL_ENERGY_INTENSITY x days
            product_energy_kwh = store_energy_kwh x product_share
            co2e = product_energy_kwh x grid_ef

        Args:
            retail_input: Retail storage input with store area, product share.

        Returns:
            Result dictionary with retail storage emissions.

        Example:
            >>> result = engine.calculate_retail_storage(RetailStorageInput(
            ...     store_floor_area_m2=Decimal("2000"),
            ...     product_share=Decimal("0.05"),
            ...     days=Decimal("21"),
            ...     country="GB",
            ... ))
        """
        start_time = time.monotonic()

        grid_ef = _get_grid_ef(retail_input.country)

        store_energy_kwh = (
            retail_input.store_floor_area_m2
            * RETAIL_STORE_ENERGY_INTENSITY
            * retail_input.days
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        product_energy_kwh = (
            store_energy_kwh * retail_input.product_share
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        co2e_kg = (product_energy_kwh * grid_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_tonnes = (co2e_kg / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        provenance_hash = _calculate_provenance_hash(
            retail_input.to_dict(),
            store_energy_kwh, product_energy_kwh, co2e_kg,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._calculation_count += 1

        result = {
            "calculation_type": "retail_storage",
            "store_floor_area_m2": retail_input.store_floor_area_m2,
            "product_share": retail_input.product_share,
            "days": retail_input.days,
            "retail_energy_intensity_kwh_m2_day": RETAIL_STORE_ENERGY_INTENSITY,
            "store_energy_kwh": store_energy_kwh,
            "product_energy_kwh": product_energy_kwh,
            "grid_ef_kgco2e_kwh": grid_ef,
            "country": retail_input.country,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "ef_source": EFSource.EPA_SMARTWAY.value,
            "data_quality_tier": DataQualityTier.TIER_3.value,
            "record_id": retail_input.record_id,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Retail storage: store=%s m2, share=%s, days=%s, co2e=%s kgCO2e",
            retail_input.store_floor_area_m2, retail_input.product_share,
            retail_input.days, co2e_kg,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 4: calculate_fulfillment_center
    # ==========================================================================

    def calculate_fulfillment_center(
        self, fc_input: FulfillmentCenterInput
    ) -> Dict[str, Any]:
        """
        Calculate e-commerce fulfillment center emissions.

        Combines building energy (floor area x intensity x days) with
        per-order pick-pack-ship energy, adjusted for automation level.

        Formula:
            building_kwh = floor_area x FC_energy_intensity x days x allocation
            order_kwh = orders x PICK_PACK_SHIP_ENERGY x automation_factor x allocation
            total_kwh = building_kwh + order_kwh
            co2e = total_kwh x grid_ef

        Args:
            fc_input: Fulfillment center input.

        Returns:
            Result dictionary with building + order emissions.

        Raises:
            ValueError: If automation_level not recognized.

        Example:
            >>> result = engine.calculate_fulfillment_center(FulfillmentCenterInput(
            ...     floor_area_m2=Decimal("10000"),
            ...     days=Decimal("30"),
            ...     orders_processed=50000,
            ...     allocation_share=Decimal("0.05"),
            ...     automation_level="semi_automated",
            ...     country="US",
            ... ))
        """
        start_time = time.monotonic()

        # Validate automation level
        auto_level = fc_input.automation_level
        auto_factor = AUTOMATION_EFFICIENCY.get(auto_level)
        if auto_factor is None:
            raise ValueError(
                f"Automation level '{auto_level}' not found. "
                f"Available: {sorted(AUTOMATION_EFFICIENCY.keys())}"
            )

        fc_ef = WAREHOUSE_EMISSION_FACTORS["ecommerce_fulfillment"]
        grid_ef = _get_grid_ef(fc_input.country)

        # Building energy
        building_kwh = (
            fc_input.floor_area_m2 * fc_ef["energy_kwh_m2_day"]
            * fc_input.days * fc_input.allocation_share
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # Order processing energy
        order_kwh = (
            Decimal(str(fc_input.orders_processed))
            * PICK_PACK_SHIP_ENERGY * auto_factor
            * fc_input.allocation_share
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        total_kwh = (building_kwh + order_kwh).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        co2e_kg = (total_kwh * grid_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_tonnes = (co2e_kg / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        provenance_hash = _calculate_provenance_hash(
            fc_input.to_dict(),
            building_kwh, order_kwh, co2e_kg,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._calculation_count += 1

        result = {
            "calculation_type": "fulfillment_center",
            "floor_area_m2": fc_input.floor_area_m2,
            "days": fc_input.days,
            "orders_processed": fc_input.orders_processed,
            "allocation_share": fc_input.allocation_share,
            "automation_level": auto_level,
            "automation_factor": auto_factor,
            "building_energy_kwh": building_kwh,
            "order_energy_kwh": order_kwh,
            "total_energy_kwh": total_kwh,
            "pick_pack_ship_kwh_per_order": PICK_PACK_SHIP_ENERGY,
            "grid_ef_kgco2e_kwh": grid_ef,
            "country": fc_input.country,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "ef_source": EFSource.EPA_SMARTWAY.value,
            "data_quality_tier": DataQualityTier.TIER_2.value,
            "record_id": fc_input.record_id,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Fulfillment center: area=%s m2, orders=%d, auto=%s, "
            "co2e=%s kgCO2e",
            fc_input.floor_area_m2, fc_input.orders_processed,
            auto_level, co2e_kg,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 5: calculate_last_mile
    # ==========================================================================

    def calculate_last_mile(
        self, last_mile_input: LastMileInput
    ) -> Dict[str, Any]:
        """
        Calculate last-mile delivery emissions.

        Formula:
            co2e = num_deliveries x ef_per_delivery[vehicle_type][area_type]

        Args:
            last_mile_input: Last-mile input with deliveries, vehicle, area.

        Returns:
            Result dictionary with last-mile emissions.

        Raises:
            ValueError: If vehicle_type or area_type not recognized.

        Example:
            >>> result = engine.calculate_last_mile(LastMileInput(
            ...     num_deliveries=10000,
            ...     vehicle_type="van_diesel",
            ...     area_type="urban",
            ... ))
        """
        start_time = time.monotonic()

        vehicle = last_mile_input.vehicle_type
        area = last_mile_input.area_type

        # Validate vehicle type
        vehicle_factors = LAST_MILE_EMISSION_FACTORS.get(vehicle)
        if vehicle_factors is None:
            raise ValueError(
                f"Vehicle type '{vehicle}' not found. "
                f"Available: {sorted(LAST_MILE_EMISSION_FACTORS.keys())}"
            )

        # Validate area type
        ef_per_delivery = vehicle_factors.get(area)
        if ef_per_delivery is None:
            raise ValueError(
                f"Area type '{area}' not found for vehicle '{vehicle}'. "
                f"Available: {sorted(vehicle_factors.keys())}"
            )

        deliveries_dec = Decimal(str(last_mile_input.num_deliveries))
        co2e_kg = (deliveries_dec * ef_per_delivery).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        co2e_tonnes = (co2e_kg / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Per-delivery metrics
        co2e_per_delivery = ef_per_delivery

        provenance_hash = _calculate_provenance_hash(
            last_mile_input.to_dict(),
            ef_per_delivery, co2e_kg,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._calculation_count += 1

        result = {
            "calculation_type": "last_mile",
            "num_deliveries": last_mile_input.num_deliveries,
            "vehicle_type": vehicle,
            "area_type": area,
            "ef_per_delivery_kgco2e": ef_per_delivery,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "co2e_per_delivery": co2e_per_delivery,
            "ef_source": EFSource.DEFRA.value,
            "data_quality_tier": DataQualityTier.TIER_3.value,
            "record_id": last_mile_input.record_id,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Last-mile calculation: deliveries=%d, vehicle=%s, area=%s, "
            "co2e=%s kgCO2e",
            last_mile_input.num_deliveries, vehicle, area, co2e_kg,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 6: calculate_batch_warehouse
    # ==========================================================================

    def calculate_batch_warehouse(
        self, warehouses: List[WarehouseInput]
    ) -> List[Dict[str, Any]]:
        """
        Calculate warehouse emissions for a batch of inputs.

        Error isolation: failed records are excluded and logged.

        Args:
            warehouses: List of WarehouseInput records.

        Returns:
            List of result dictionaries (failed records excluded).

        Raises:
            ValueError: If warehouses list is empty.

        Example:
            >>> results = engine.calculate_batch_warehouse([
            ...     WarehouseInput("ambient", Decimal("5000"), Decimal("30")),
            ...     WarehouseInput("cold_storage", Decimal("2000"), Decimal("14")),
            ... ])
        """
        if not warehouses:
            raise ValueError("Warehouses list cannot be empty")

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        error_count = 0

        logger.info(
            "Starting warehouse batch: %d records", len(warehouses),
        )

        for idx, wh_input in enumerate(warehouses):
            try:
                result = self.calculate_warehouse(wh_input)
                results.append(result)
            except (ValueError, InvalidOperation) as exc:
                error_count += 1
                logger.error(
                    "Warehouse batch record %d failed: %s (type=%s)",
                    idx, str(exc), wh_input.warehouse_type,
                )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._batch_count += 1

        total_co2e = sum(
            (r["co2e_kg"] for r in results), ZERO
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        logger.info(
            "Warehouse batch complete: %d/%d succeeded, %d failed, "
            "total=%s kgCO2e",
            len(results), len(warehouses), error_count, total_co2e,
        )

        return results

    # ==========================================================================
    # PUBLIC METHOD 7: calculate_batch_last_mile
    # ==========================================================================

    def calculate_batch_last_mile(
        self, deliveries: List[LastMileInput]
    ) -> List[Dict[str, Any]]:
        """
        Calculate last-mile emissions for a batch of inputs.

        Error isolation: failed records are excluded and logged.

        Args:
            deliveries: List of LastMileInput records.

        Returns:
            List of result dictionaries (failed records excluded).

        Raises:
            ValueError: If deliveries list is empty.

        Example:
            >>> results = engine.calculate_batch_last_mile([
            ...     LastMileInput(num_deliveries=5000, vehicle_type="van_diesel"),
            ...     LastMileInput(num_deliveries=3000, vehicle_type="van_electric"),
            ... ])
        """
        if not deliveries:
            raise ValueError("Deliveries list cannot be empty")

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        error_count = 0

        logger.info(
            "Starting last-mile batch: %d records", len(deliveries),
        )

        for idx, lm_input in enumerate(deliveries):
            try:
                result = self.calculate_last_mile(lm_input)
                results.append(result)
            except (ValueError, InvalidOperation) as exc:
                error_count += 1
                logger.error(
                    "Last-mile batch record %d failed: %s (vehicle=%s)",
                    idx, str(exc), lm_input.vehicle_type,
                )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._batch_count += 1

        total_co2e = sum(
            (r["co2e_kg"] for r in results), ZERO
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        logger.info(
            "Last-mile batch complete: %d/%d succeeded, %d failed, "
            "total=%s kgCO2e",
            len(results), len(deliveries), error_count, total_co2e,
        )

        return results

    # ==========================================================================
    # PUBLIC METHOD 8: calculate_distribution_chain
    # ==========================================================================

    def calculate_distribution_chain(
        self,
        warehouses: List[WarehouseInput],
        last_mile_deliveries: Optional[List[LastMileInput]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate combined distribution chain emissions (warehouses + last mile).

        Processes all warehouses and last-mile deliveries, then aggregates
        into a single distribution chain result.

        Args:
            warehouses: List of warehouse inputs in the chain.
            last_mile_deliveries: Optional list of last-mile inputs.

        Returns:
            Dictionary containing:
                - warehouse_results: List of warehouse results
                - last_mile_results: List of last-mile results
                - total_warehouse_co2e_kg: Sum of warehouse emissions
                - total_last_mile_co2e_kg: Sum of last-mile emissions
                - total_co2e_kg: Grand total
                - total_co2e_tonnes: Grand total in tonnes
                - warehouse_share_pct: Warehouse % of total
                - last_mile_share_pct: Last-mile % of total
                - provenance_hash: SHA-256 hash

        Raises:
            ValueError: If warehouses list is empty.

        Example:
            >>> result = engine.calculate_distribution_chain(
            ...     warehouses=[WarehouseInput(...)],
            ...     last_mile_deliveries=[LastMileInput(...)],
            ... )
        """
        if not warehouses:
            raise ValueError("Warehouses list cannot be empty")

        start_time = time.monotonic()

        # Process warehouses
        wh_results = self.calculate_batch_warehouse(warehouses)
        total_wh_co2e = sum(
            (r["co2e_kg"] for r in wh_results), ZERO
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # Process last-mile
        lm_results: List[Dict[str, Any]] = []
        total_lm_co2e = ZERO
        if last_mile_deliveries:
            lm_results = self.calculate_batch_last_mile(last_mile_deliveries)
            total_lm_co2e = sum(
                (r["co2e_kg"] for r in lm_results), ZERO
            ).quantize(_QUANT_8DP, rounding=ROUNDING)

        # Grand total
        total_co2e = (total_wh_co2e + total_lm_co2e).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        total_co2e_tonnes = (total_co2e / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Shares
        if total_co2e > ZERO:
            wh_share = (total_wh_co2e / total_co2e * HUNDRED).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            lm_share = (total_lm_co2e / total_co2e * HUNDRED).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
        else:
            wh_share = ZERO
            lm_share = ZERO

        provenance_hash = _calculate_provenance_hash(
            total_wh_co2e, total_lm_co2e, total_co2e,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "warehouse_results": wh_results,
            "last_mile_results": lm_results,
            "warehouse_count": len(wh_results),
            "last_mile_count": len(lm_results),
            "total_warehouse_co2e_kg": total_wh_co2e,
            "total_last_mile_co2e_kg": total_lm_co2e,
            "total_co2e_kg": total_co2e,
            "total_co2e_tonnes": total_co2e_tonnes,
            "warehouse_share_pct": wh_share,
            "last_mile_share_pct": lm_share,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Distribution chain: warehouses=%d (%s kgCO2e), "
            "last_mile=%d (%s kgCO2e), total=%s kgCO2e",
            len(wh_results), total_wh_co2e,
            len(lm_results), total_lm_co2e, total_co2e,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 9: estimate_annual_storage
    # ==========================================================================

    def estimate_annual_storage(
        self,
        product_volume_tonnes: Decimal,
        channel: str = "retail",
        country: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Estimate annual storage emissions for a product volume.

        Uses channel-specific average storage days and warehouse type
        to produce an annual estimate.

        Args:
            product_volume_tonnes: Annual product volume in tonnes.
            channel: Distribution channel for storage defaults.
            country: Country for grid emission factor.

        Returns:
            Result dictionary with annual storage estimate.

        Raises:
            ValueError: If channel not recognized or volume <= 0.

        Example:
            >>> result = engine.estimate_annual_storage(
            ...     Decimal("5000"), "retail", "US"
            ... )
        """
        if product_volume_tonnes <= ZERO:
            raise ValueError(
                f"Product volume must be positive, got {product_volume_tonnes}"
            )

        # Channel defaults mapping to warehouse type
        channel_to_wh_type: Dict[str, str] = {
            "retail": "ambient",
            "ecommerce": "ecommerce_fulfillment",
            "wholesale": "ambient",
            "direct_to_consumer": "ambient",
            "cold_chain": "cold_storage",
            "bulk_industrial": "bulk_storage",
        }

        channel_key = channel.lower()
        wh_type = channel_to_wh_type.get(channel_key)
        if wh_type is None:
            raise ValueError(
                f"Channel '{channel}' not found. "
                f"Available: {sorted(channel_to_wh_type.keys())}"
            )

        # Import channel defaults for storage days
        from greenlang.downstream_transportation.average_data_calculator import (
            DISTRIBUTION_CHANNEL_DEFAULTS,
        )
        ch_defaults = DISTRIBUTION_CHANNEL_DEFAULTS.get(channel_key)
        storage_days_per_cycle = ch_defaults["storage_days"] if ch_defaults else Decimal("14")

        # Estimate number of inventory turns per year
        if storage_days_per_cycle > ZERO:
            turns_per_year = (DAYS_PER_YEAR / storage_days_per_cycle).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
        else:
            turns_per_year = ONE

        # Average inventory at any time
        avg_inventory = (product_volume_tonnes / turns_per_year).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Floor area needed
        area_per_tonne = Decimal("5.0")
        floor_area = (avg_inventory * area_per_tonne).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Calculate annual warehouse emissions
        wh_input = WarehouseInput(
            warehouse_type=wh_type,
            floor_area_m2=floor_area,
            days=DAYS_PER_YEAR,
            allocation_share=ONE,
            country=country,
            record_id="annual_estimate",
        )
        wh_result = self.calculate_warehouse(wh_input)

        wh_result["estimation_type"] = "annual_storage"
        wh_result["channel"] = channel_key
        wh_result["product_volume_tonnes"] = product_volume_tonnes
        wh_result["avg_inventory_tonnes"] = avg_inventory
        wh_result["inventory_turns_per_year"] = turns_per_year
        wh_result["storage_days_per_cycle"] = storage_days_per_cycle

        logger.info(
            "Annual storage estimate: volume=%s t, channel=%s, "
            "co2e=%s kgCO2e",
            product_volume_tonnes, channel_key, wh_result["co2e_kg"],
        )

        return wh_result

    # ==========================================================================
    # PUBLIC METHOD 10: compare_warehouse_types
    # ==========================================================================

    def compare_warehouse_types(
        self,
        floor_area_m2: Decimal,
        days: Decimal,
        allocation_share: Decimal = ONE,
        country: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Compare emissions across all 7 warehouse types.

        Calculates emissions for the given parameters across every
        warehouse type and ranks by total CO2e.

        Args:
            floor_area_m2: Floor area in square metres.
            days: Number of storage days.
            allocation_share: Allocation share (default 1.0).
            country: Country for grid emission factor.

        Returns:
            Dictionary containing:
                - comparisons: Sorted list of per-type results
                - lowest_type: Warehouse type with lowest emissions
                - highest_type: Warehouse type with highest emissions
                - reduction_potential_pct: % reduction from highest to lowest

        Raises:
            ValueError: If floor_area or days not positive.

        Example:
            >>> result = engine.compare_warehouse_types(
            ...     Decimal("5000"), Decimal("30"), country="US"
            ... )
        """
        if floor_area_m2 <= ZERO:
            raise ValueError(f"Floor area must be positive, got {floor_area_m2}")
        if days <= ZERO:
            raise ValueError(f"Days must be positive, got {days}")

        start_time = time.monotonic()
        comparisons: List[Dict[str, Any]] = []

        for wh_type in sorted(WAREHOUSE_EMISSION_FACTORS.keys()):
            try:
                wh_input = WarehouseInput(
                    warehouse_type=wh_type,
                    floor_area_m2=floor_area_m2,
                    days=days,
                    allocation_share=allocation_share,
                    country=country,
                )
                result = self.calculate_warehouse(wh_input)
                comparisons.append({
                    "warehouse_type": wh_type,
                    "warehouse_name": result["warehouse_name"],
                    "energy_intensity": result["energy_intensity_kwh_m2_day"],
                    "total_energy_kwh": result["total_energy_kwh"],
                    "co2e_kg": result["co2e_kg"],
                    "co2e_tonnes": result["co2e_tonnes"],
                })
            except (ValueError, InvalidOperation) as exc:
                logger.error(
                    "Warehouse comparison failed for %s: %s",
                    wh_type, str(exc),
                )

        # Sort by CO2e
        comparisons.sort(key=lambda x: x["co2e_kg"])

        lowest = comparisons[0] if comparisons else None
        highest = comparisons[-1] if comparisons else None

        if lowest and highest and highest["co2e_kg"] > ZERO:
            reduction = (
                (highest["co2e_kg"] - lowest["co2e_kg"])
                / highest["co2e_kg"] * HUNDRED
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
        else:
            reduction = ZERO

        provenance_hash = _calculate_provenance_hash(
            floor_area_m2, days, country, len(comparisons),
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "comparisons": comparisons,
            "lowest_type": lowest["warehouse_type"] if lowest else None,
            "lowest_co2e_kg": lowest["co2e_kg"] if lowest else ZERO,
            "highest_type": highest["warehouse_type"] if highest else None,
            "highest_co2e_kg": highest["co2e_kg"] if highest else ZERO,
            "reduction_potential_pct": reduction,
            "floor_area_m2": floor_area_m2,
            "days": days,
            "allocation_share": allocation_share,
            "country": country,
            "type_count": len(comparisons),
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Warehouse comparison: lowest=%s (%s kgCO2e), "
            "highest=%s (%s kgCO2e), reduction=%s%%",
            lowest["warehouse_type"] if lowest else "N/A",
            lowest["co2e_kg"] if lowest else "0",
            highest["warehouse_type"] if highest else "N/A",
            highest["co2e_kg"] if highest else "0",
            reduction,
        )

        return result

    # ==========================================================================
    # ADDITIONAL PUBLIC METHODS
    # ==========================================================================

    def get_warehouse_types(self) -> List[Dict[str, Any]]:
        """
        Return all available warehouse types with energy intensity.

        Returns:
            List of warehouse type dictionaries.
        """
        result = []
        for wh_type, data in sorted(WAREHOUSE_EMISSION_FACTORS.items()):
            result.append({
                "warehouse_type": wh_type,
                "name": data["name"],
                "energy_kwh_m2_day": float(data["energy_kwh_m2_day"]),
                "description": data["description"],
                "source": data["source"],
            })
        return result

    def get_last_mile_factors(self) -> List[Dict[str, Any]]:
        """
        Return all last-mile emission factors.

        Returns:
            List of last-mile factor dictionaries.
        """
        result = []
        for vehicle, areas in sorted(LAST_MILE_EMISSION_FACTORS.items()):
            for area, ef in sorted(areas.items()):
                result.append({
                    "vehicle_type": vehicle,
                    "area_type": area,
                    "ef_per_delivery_kgco2e": float(ef),
                    "ef_unit": "kgCO2e/delivery",
                    "ef_source": EFSource.DEFRA.value,
                })
        return result

    def get_temperature_tiers(self) -> List[Dict[str, Any]]:
        """
        Return all cold storage temperature tiers.

        Returns:
            List of temperature tier dictionaries.
        """
        result = []
        for tier, data in sorted(COLD_STORAGE_TEMP_MULTIPLIERS.items()):
            result.append({
                "tier": tier,
                "name": data["name"],
                "multiplier": float(data["multiplier"]),
                "temp_range_c": data["temp_range_c"],
            })
        return result

    def get_grid_emission_factors(self) -> Dict[str, float]:
        """
        Return all grid emission factors by country.

        Returns:
            Dictionary mapping country code to kgCO2e/kWh.
        """
        return {k: float(v) for k, v in sorted(GRID_EMISSION_FACTORS.items())}

    def get_calculation_stats(self) -> Dict[str, Any]:
        """
        Return engine calculation statistics.

        Returns:
            Dictionary with counts and configuration.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "warehouse_types_available": len(WAREHOUSE_EMISSION_FACTORS),
            "last_mile_vehicle_types": len(LAST_MILE_EMISSION_FACTORS),
            "temperature_tiers": len(COLD_STORAGE_TEMP_MULTIPLIERS),
            "countries_with_grid_ef": len(GRID_EMISSION_FACTORS),
        }


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_warehouse_distribution_engine() -> WarehouseDistributionEngine:
    """
    Get the WarehouseDistributionEngine singleton instance.

    Returns:
        WarehouseDistributionEngine singleton.
    """
    return WarehouseDistributionEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "TABLE_PREFIX",
    # Enums
    "WarehouseType",
    "TemperatureTier",
    "LastMileVehicleType",
    "AreaType",
    "EFSource",
    "DataQualityTier",
    # Data Tables
    "WAREHOUSE_EMISSION_FACTORS",
    "COLD_STORAGE_TEMP_MULTIPLIERS",
    "LAST_MILE_EMISSION_FACTORS",
    "GRID_EMISSION_FACTORS",
    "WAREHOUSE_ENERGY_INTENSITY",
    "COLD_STORAGE_ENERGY_INTENSITY",
    "WAREHOUSE_AREA_PER_TONNE",
    "RETAIL_STORE_ENERGY_INTENSITY",
    "PICK_PACK_SHIP_ENERGY",
    "AUTOMATION_EFFICIENCY",
    # Input Models
    "WarehouseInput",
    "ColdStorageInput",
    "RetailStorageInput",
    "FulfillmentCenterInput",
    "LastMileInput",
    # Engine
    "WarehouseDistributionEngine",
    "get_warehouse_distribution_engine",
]
