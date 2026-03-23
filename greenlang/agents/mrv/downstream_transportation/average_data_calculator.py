# -*- coding: utf-8 -*-
"""
AverageDataCalculatorEngine - AGENT-MRV-022 Engine 4

GHH Protocol Scope 3 Category 9 average-data emissions calculator for
downstream transportation and distribution.

This engine provides average-data emissions estimation using industry-
standard channel defaults when detailed distance, fuel, or supplier-
specific data is unavailable. It combines transport and storage
components with optional cold-chain uplift.

Core Approach:
    1. **Channel Defaults Table**: Six standard distribution channels
       (retail, e-commerce, wholesale, direct-to-consumer, cold chain,
       bulk/industrial) with embedded average distance, mode split,
       number of transport legs, and storage days.

    2. **Transport Component**: Calculates emissions from average distance
       and mode using embedded emission factors per tonne-km.

    3. **Storage Component**: Calculates warehouse/storage emissions from
       average storage days using country-specific grid emission factors
       and warehouse energy intensity.

    4. **Cold Chain Uplift**: Applies a multiplier for refrigerated /
       temperature-controlled distribution channels.

    5. **Product Category Adjustment**: Adjusts channel defaults based on
       product category (e.g., perishable goods, electronics, bulk
       commodities) to improve estimate accuracy.

    6. **Channel Comparison**: Compares emissions across all six channels
       for a given weight to inform channel optimization.

All calculations use Decimal arithmetic with ROUND_HALF_UP for regulatory
precision. Thread-safe singleton pattern for concurrent pipeline use.

Zero-Hallucination Compliance:
    All emission calculations use deterministic arithmetic on embedded
    factor tables. No LLM calls are made in any calculation path.

References:
    - GHH Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 9
    - GLEC Framework v3.0 (Global Logistics Emissions Council)
    - DEFRA / UK BEIS Conversion Factors 2023
    - US EPA SmartWay Transport Partnership
    - IEA World Energy Outlook (grid emission factors)

Example:
    >>> engine = AverageDataCalculatorEngine.get_instance()
    >>> result = engine.calculate_average_data(AverageDataInput(
    ...     channel="retail",
    ...     weight_tonnes=Decimal("10.0"),
    ...     country="US",
    ... ))
    >>> result["total_co2e_kg"] > Decimal("0")
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
ENGINE_ID: str = "average_data_calculator_engine"
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


class DistributionChannel(str, Enum):
    """Standard downstream distribution channels."""

    RETAIL = "retail"
    ECOMMERCE = "ecommerce"
    WHOLESALE = "wholesale"
    DIRECT_TO_CONSUMER = "direct_to_consumer"
    COLD_CHAIN = "cold_chain"
    BULK_INDUSTRIAL = "bulk_industrial"


class TransportMode(str, Enum):
    """Transport modes for average-data calculations."""

    ROAD = "road"
    RAIL = "rail"
    MARITIME = "maritime"
    AIR = "air"
    INTERMODAL = "intermodal"


class ProductCategory(str, Enum):
    """Product categories affecting distribution defaults."""

    PERISHABLE = "perishable"
    ELECTRONICS = "electronics"
    APPAREL = "apparel"
    FMCG = "fmcg"                       # Fast-moving consumer goods
    BULK_COMMODITY = "bulk_commodity"
    PHARMACEUTICAL = "pharmaceutical"
    FURNITURE = "furniture"
    GENERAL_MERCHANDISE = "general_merchandise"


class EFSource(str, Enum):
    """Emission factor data source."""

    GLEC = "glec"
    DEFRA = "defra"
    EPA_SMARTWAY = "epa_smartway"
    IEA = "iea"
    CUSTOM = "custom"


class DataQualityTier(str, Enum):
    """Data quality tiers for average-data calculations."""

    TIER_3 = "tier_3"  # Average-data (moderate quality)
    TIER_4 = "tier_4"  # Generic average (lower quality)


# ==============================================================================
# DISTRIBUTION CHANNEL DEFAULTS TABLE
# ==============================================================================

DISTRIBUTION_CHANNEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "retail": {
        "name": "Retail Distribution",
        "description": "Products shipped to retail stores via regional DCs",
        "avg_distance_km": Decimal("800"),
        "avg_mode": TransportMode.ROAD,
        "avg_legs": 2,
        "storage_days": Decimal("14"),
        "cold_chain": False,
        "cold_chain_uplift": Decimal("1.0"),
        "avg_load_factor": Decimal("0.70"),
    },
    "ecommerce": {
        "name": "E-Commerce Distribution",
        "description": "Products shipped from fulfillment centers to end consumer",
        "avg_distance_km": Decimal("600"),
        "avg_mode": TransportMode.ROAD,
        "avg_legs": 3,
        "storage_days": Decimal("7"),
        "cold_chain": False,
        "cold_chain_uplift": Decimal("1.0"),
        "avg_load_factor": Decimal("0.55"),
    },
    "wholesale": {
        "name": "Wholesale Distribution",
        "description": "Products shipped to wholesale warehouses in bulk",
        "avg_distance_km": Decimal("1200"),
        "avg_mode": TransportMode.ROAD,
        "avg_legs": 1,
        "storage_days": Decimal("30"),
        "cold_chain": False,
        "cold_chain_uplift": Decimal("1.0"),
        "avg_load_factor": Decimal("0.85"),
    },
    "direct_to_consumer": {
        "name": "Direct-to-Consumer",
        "description": "Products shipped directly from manufacturer to consumer",
        "avg_distance_km": Decimal("500"),
        "avg_mode": TransportMode.ROAD,
        "avg_legs": 2,
        "storage_days": Decimal("3"),
        "cold_chain": False,
        "cold_chain_uplift": Decimal("1.0"),
        "avg_load_factor": Decimal("0.45"),
    },
    "cold_chain": {
        "name": "Cold Chain Distribution",
        "description": "Temperature-controlled distribution for perishables",
        "avg_distance_km": Decimal("500"),
        "avg_mode": TransportMode.ROAD,
        "avg_legs": 2,
        "storage_days": Decimal("5"),
        "cold_chain": True,
        "cold_chain_uplift": Decimal("1.40"),
        "avg_load_factor": Decimal("0.65"),
    },
    "bulk_industrial": {
        "name": "Bulk / Industrial Distribution",
        "description": "Heavy/bulk goods via rail or maritime with long storage",
        "avg_distance_km": Decimal("2000"),
        "avg_mode": TransportMode.RAIL,
        "avg_legs": 1,
        "storage_days": Decimal("45"),
        "cold_chain": False,
        "cold_chain_uplift": Decimal("1.0"),
        "avg_load_factor": Decimal("0.90"),
    },
}

# ==============================================================================
# TRANSPORT EMISSION FACTORS (kgCO2e per tonne-km)
# Source: GLEC Framework v3.0 / DEFRA 2023
# ==============================================================================

TRANSPORT_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "road": {
        "ef_per_tkm": Decimal("0.06200"),
        "name": "Road freight (average truck)",
        "source": EFSource.GLEC.value,
        "wtt_uplift": Decimal("1.14"),  # Well-to-tank uplift factor
    },
    "rail": {
        "ef_per_tkm": Decimal("0.02800"),
        "name": "Rail freight (average)",
        "source": EFSource.GLEC.value,
        "wtt_uplift": Decimal("1.09"),
    },
    "maritime": {
        "ef_per_tkm": Decimal("0.01600"),
        "name": "Maritime freight (container ship average)",
        "source": EFSource.GLEC.value,
        "wtt_uplift": Decimal("1.12"),
    },
    "air": {
        "ef_per_tkm": Decimal("0.60200"),
        "name": "Air freight (belly/freighter average)",
        "source": EFSource.GLEC.value,
        "wtt_uplift": Decimal("1.16"),
    },
    "intermodal": {
        "ef_per_tkm": Decimal("0.04500"),
        "name": "Intermodal freight (road + rail average)",
        "source": EFSource.GLEC.value,
        "wtt_uplift": Decimal("1.12"),
    },
}

# ==============================================================================
# WAREHOUSE ENERGY INTENSITY (kWh per m2 per day)
# Basis for storage emission calculation
# ==============================================================================

WAREHOUSE_ENERGY_INTENSITY: Decimal = Decimal("0.0685")  # kWh/m2/day (ambient)
COLD_STORAGE_ENERGY_INTENSITY: Decimal = Decimal("0.1950")  # kWh/m2/day (refrigerated)

# Average warehouse footprint per tonne of product stored
WAREHOUSE_AREA_PER_TONNE: Decimal = Decimal("5.0")  # m2 per tonne

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
# PRODUCT CATEGORY ADJUSTMENTS
# Multipliers to channel defaults based on product type
# ==============================================================================

PRODUCT_CATEGORY_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    "perishable": {
        "distance_multiplier": Decimal("0.80"),
        "storage_multiplier": Decimal("0.50"),
        "cold_chain_override": True,
        "cold_chain_uplift": Decimal("1.40"),
    },
    "electronics": {
        "distance_multiplier": Decimal("1.20"),
        "storage_multiplier": Decimal("0.70"),
        "cold_chain_override": False,
        "cold_chain_uplift": Decimal("1.0"),
    },
    "apparel": {
        "distance_multiplier": Decimal("1.10"),
        "storage_multiplier": Decimal("1.20"),
        "cold_chain_override": False,
        "cold_chain_uplift": Decimal("1.0"),
    },
    "fmcg": {
        "distance_multiplier": Decimal("0.90"),
        "storage_multiplier": Decimal("0.80"),
        "cold_chain_override": False,
        "cold_chain_uplift": Decimal("1.0"),
    },
    "bulk_commodity": {
        "distance_multiplier": Decimal("1.50"),
        "storage_multiplier": Decimal("2.00"),
        "cold_chain_override": False,
        "cold_chain_uplift": Decimal("1.0"),
    },
    "pharmaceutical": {
        "distance_multiplier": Decimal("0.70"),
        "storage_multiplier": Decimal("0.40"),
        "cold_chain_override": True,
        "cold_chain_uplift": Decimal("1.60"),
    },
    "furniture": {
        "distance_multiplier": Decimal("1.00"),
        "storage_multiplier": Decimal("1.50"),
        "cold_chain_override": False,
        "cold_chain_uplift": Decimal("1.0"),
    },
    "general_merchandise": {
        "distance_multiplier": Decimal("1.00"),
        "storage_multiplier": Decimal("1.00"),
        "cold_chain_override": False,
        "cold_chain_uplift": Decimal("1.0"),
    },
}


# ==============================================================================
# INPUT MODELS
# ==============================================================================


class AverageDataInput:
    """
    Input for average-data emissions calculation.

    Immutable value object for a single average-data calculation request.

    Attributes:
        channel: Distribution channel name.
        weight_tonnes: Weight of goods in metric tonnes.
        country: ISO 3166-1 alpha-2 country code for grid EF.
        product_category: Optional product category for adjustments.
        custom_distance_km: Optional override for average distance.
        custom_storage_days: Optional override for storage days.
        custom_mode: Optional override for transport mode.
        record_id: Optional unique identifier.
        tenant_id: Optional tenant identifier.

    Raises:
        ValueError: If weight_tonnes <= 0.
    """

    __slots__ = (
        "channel", "weight_tonnes", "country", "product_category",
        "custom_distance_km", "custom_storage_days", "custom_mode",
        "record_id", "tenant_id",
    )

    def __init__(
        self,
        channel: str,
        weight_tonnes: Decimal,
        country: str = "GLOBAL",
        product_category: Optional[str] = None,
        custom_distance_km: Optional[Decimal] = None,
        custom_storage_days: Optional[Decimal] = None,
        custom_mode: Optional[str] = None,
        record_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize AverageDataInput with validation."""
        if weight_tonnes <= ZERO:
            raise ValueError(
                f"Weight must be positive, got {weight_tonnes}"
            )
        object.__setattr__(self, "channel", channel.lower())
        object.__setattr__(self, "weight_tonnes", weight_tonnes)
        object.__setattr__(self, "country", country.upper())
        object.__setattr__(self, "product_category", product_category)
        object.__setattr__(self, "custom_distance_km", custom_distance_km)
        object.__setattr__(self, "custom_storage_days", custom_storage_days)
        object.__setattr__(self, "custom_mode", custom_mode)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "tenant_id", tenant_id)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation after initialization."""
        raise AttributeError("AverageDataInput is immutable")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AverageDataInput(channel={self.channel!r}, "
            f"weight_tonnes={self.weight_tonnes}, country={self.country!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for provenance hashing."""
        return {
            "channel": self.channel,
            "weight_tonnes": str(self.weight_tonnes),
            "country": self.country,
            "product_category": self.product_category,
            "custom_distance_km": str(self.custom_distance_km) if self.custom_distance_km else None,
            "custom_storage_days": str(self.custom_storage_days) if self.custom_storage_days else None,
            "custom_mode": self.custom_mode,
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
# INTERNAL CALCULATION HELPERS
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


def _calculate_transport_emissions(
    weight_tonnes: Decimal,
    distance_km: Decimal,
    mode: str,
    cold_chain_uplift: Decimal = ONE,
    load_factor: Decimal = ONE,
) -> Dict[str, Any]:
    """
    Calculate transport emissions for a single leg.

    Formula:
        tonne_km = weight_tonnes x distance_km / load_factor
        base_co2e = tonne_km x ef_per_tkm
        wtt_co2e = base_co2e x (wtt_uplift - 1)
        total_co2e = (base_co2e + wtt_co2e) x cold_chain_uplift

    Args:
        weight_tonnes: Cargo weight in metric tonnes.
        distance_km: Transport distance in kilometres.
        mode: Transport mode key (road, rail, maritime, air, intermodal).
        cold_chain_uplift: Cold chain multiplier (default 1.0).
        load_factor: Average load factor (default 1.0, no adjustment).

    Returns:
        Dictionary with tonne_km, base_co2e, wtt_co2e, total_co2e.

    Raises:
        ValueError: If mode is not in TRANSPORT_EMISSION_FACTORS.
    """
    mode_lower = mode.lower() if isinstance(mode, str) else mode.value.lower()
    ef_entry = TRANSPORT_EMISSION_FACTORS.get(mode_lower)
    if ef_entry is None:
        raise ValueError(
            f"Transport mode '{mode}' not found. "
            f"Available: {sorted(TRANSPORT_EMISSION_FACTORS.keys())}"
        )

    ef_per_tkm = ef_entry["ef_per_tkm"]
    wtt_uplift = ef_entry["wtt_uplift"]

    # Adjusted tonne-km (accounting for load factor)
    if load_factor > ZERO and load_factor != ONE:
        effective_tkm = (weight_tonnes * distance_km / load_factor).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
    else:
        effective_tkm = (weight_tonnes * distance_km).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

    base_co2e = (effective_tkm * ef_per_tkm).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )
    wtt_co2e = (base_co2e * (wtt_uplift - ONE)).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )
    subtotal = (base_co2e + wtt_co2e).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )
    total_co2e = (subtotal * cold_chain_uplift).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )

    return {
        "mode": mode_lower,
        "distance_km": distance_km,
        "weight_tonnes": weight_tonnes,
        "load_factor": load_factor,
        "effective_tonne_km": effective_tkm,
        "ef_per_tkm": ef_per_tkm,
        "base_co2e_kg": base_co2e,
        "wtt_uplift": wtt_uplift,
        "wtt_co2e_kg": wtt_co2e,
        "cold_chain_uplift": cold_chain_uplift,
        "total_co2e_kg": total_co2e,
        "ef_source": ef_entry["source"],
    }


def _calculate_storage_emissions(
    weight_tonnes: Decimal,
    storage_days: Decimal,
    country: str = "GLOBAL",
    is_cold_chain: bool = False,
) -> Dict[str, Any]:
    """
    Calculate storage/warehouse emissions.

    Formula:
        floor_area = weight_tonnes x WAREHOUSE_AREA_PER_TONNE
        energy_kwh = floor_area x energy_intensity x storage_days
        co2e = energy_kwh x grid_ef

    Args:
        weight_tonnes: Weight of goods stored (tonnes).
        storage_days: Number of days in storage.
        country: Country for grid emission factor.
        is_cold_chain: Whether to use cold-storage energy intensity.

    Returns:
        Dictionary with floor_area, energy_kwh, grid_ef, co2e.
    """
    floor_area = (weight_tonnes * WAREHOUSE_AREA_PER_TONNE).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )

    energy_intensity = (
        COLD_STORAGE_ENERGY_INTENSITY if is_cold_chain
        else WAREHOUSE_ENERGY_INTENSITY
    )

    energy_kwh = (floor_area * energy_intensity * storage_days).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )

    grid_ef = _get_grid_ef(country)
    co2e_kg = (energy_kwh * grid_ef).quantize(
        _QUANT_8DP, rounding=ROUNDING
    )

    return {
        "floor_area_m2": floor_area,
        "energy_intensity_kwh_m2_day": energy_intensity,
        "storage_days": storage_days,
        "energy_kwh": energy_kwh,
        "grid_ef_kgco2e_kwh": grid_ef,
        "country": country,
        "is_cold_chain": is_cold_chain,
        "co2e_kg": co2e_kg,
    }


# ==============================================================================
# AverageDataCalculatorEngine
# ==============================================================================


class AverageDataCalculatorEngine:
    """
    Average-data emissions calculator for downstream transportation
    and distribution (Scope 3 Category 9).

    Uses industry-standard channel defaults to estimate emissions
    when detailed activity data is unavailable. Combines transport
    and storage components with optional cold-chain uplift.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Data Quality:
        Average-data estimates are Tier 3-4 quality. Organizations
        should prioritize supplier-specific > distance-based > average-data
        > spend-based whenever more granular data is available.

    Attributes:
        _calculation_count: Running count of calculations.
        _batch_count: Running count of batch operations.

    Example:
        >>> engine = AverageDataCalculatorEngine.get_instance()
        >>> result = engine.calculate_average_data(AverageDataInput(
        ...     channel="retail",
        ...     weight_tonnes=Decimal("10"),
        ...     country="US",
        ... ))
        >>> result["total_co2e_kg"] > Decimal("0")
        True
    """

    _instance: Optional["AverageDataCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize AverageDataCalculatorEngine."""
        self._calculation_count: int = 0
        self._batch_count: int = 0

        logger.info(
            "AverageDataCalculatorEngine initialized: agent=%s, version=%s, "
            "channels=%d, modes=%d",
            AGENT_ID, ENGINE_VERSION,
            len(DISTRIBUTION_CHANNEL_DEFAULTS),
            len(TRANSPORT_EMISSION_FACTORS),
        )

    # ==========================================================================
    # SINGLETON
    # ==========================================================================

    @classmethod
    def get_instance(cls) -> "AverageDataCalculatorEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            AverageDataCalculatorEngine singleton instance.
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
            logger.info("AverageDataCalculatorEngine singleton reset")

    # ==========================================================================
    # PUBLIC METHOD 1: calculate_average_data
    # ==========================================================================

    def calculate_average_data(
        self, avg_input: AverageDataInput
    ) -> Dict[str, Any]:
        """
        Calculate average-data emissions for a distribution channel.

        Combines transport and storage components using channel defaults.
        Applies product category adjustments and cold chain uplift if
        applicable.

        Args:
            avg_input: Average data input with channel, weight, country.

        Returns:
            Dictionary containing:
                - channel: Distribution channel used
                - channel_name: Human-readable channel name
                - weight_tonnes: Weight of goods
                - transport_result: Transport emissions breakdown
                - storage_result: Storage emissions breakdown
                - transport_co2e_kg: Transport component
                - storage_co2e_kg: Storage component
                - total_co2e_kg: Total emissions (transport + storage)
                - total_co2e_tonnes: Total in tonnes CO2e
                - cold_chain_applied: Whether cold chain uplift applied
                - product_adjustments: Product category adjustments applied
                - data_quality_tier: Data quality tier
                - provenance_hash: SHA-256 provenance hash
                - engine_id, engine_version, agent_id
                - calculation_timestamp, processing_time_ms

        Raises:
            ValueError: If channel is not recognized.

        Example:
            >>> result = engine.calculate_average_data(AverageDataInput(
            ...     channel="retail",
            ...     weight_tonnes=Decimal("10"),
            ...     country="US",
            ... ))
        """
        start_time = time.monotonic()

        # Validate channel
        channel_key = avg_input.channel.lower()
        channel_defaults = DISTRIBUTION_CHANNEL_DEFAULTS.get(channel_key)
        if channel_defaults is None:
            raise ValueError(
                f"Distribution channel '{avg_input.channel}' not found. "
                f"Available: {sorted(DISTRIBUTION_CHANNEL_DEFAULTS.keys())}"
            )

        # Resolve effective parameters (custom overrides + product adjustments)
        effective = self._resolve_effective_params(
            channel_defaults, avg_input
        )

        # Calculate transport component
        transport_result = _calculate_transport_emissions(
            weight_tonnes=avg_input.weight_tonnes,
            distance_km=effective["distance_km"],
            mode=effective["mode"],
            cold_chain_uplift=effective["cold_chain_uplift"],
            load_factor=effective["load_factor"],
        )

        # Multiply by number of legs
        legs = effective["legs"]
        transport_co2e = (transport_result["total_co2e_kg"] * Decimal(str(legs))).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Calculate storage component
        storage_result = _calculate_storage_emissions(
            weight_tonnes=avg_input.weight_tonnes,
            storage_days=effective["storage_days"],
            country=avg_input.country,
            is_cold_chain=effective["is_cold_chain"],
        )
        storage_co2e = storage_result["co2e_kg"]

        # Total
        total_co2e = (transport_co2e + storage_co2e).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        total_co2e_tonnes = (total_co2e / THOUSAND).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        # Provenance hash
        provenance_hash = _calculate_provenance_hash(
            avg_input.to_dict(),
            transport_co2e, storage_co2e, total_co2e,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._calculation_count += 1

        result = {
            "channel": channel_key,
            "channel_name": channel_defaults["name"],
            "weight_tonnes": avg_input.weight_tonnes,
            "country": avg_input.country,
            "effective_distance_km": effective["distance_km"],
            "effective_mode": effective["mode"],
            "effective_legs": legs,
            "effective_storage_days": effective["storage_days"],
            "transport_result": transport_result,
            "transport_legs": legs,
            "transport_co2e_kg": transport_co2e,
            "storage_result": storage_result,
            "storage_co2e_kg": storage_co2e,
            "total_co2e_kg": total_co2e,
            "total_co2e_tonnes": total_co2e_tonnes,
            "cold_chain_applied": effective["is_cold_chain"],
            "cold_chain_uplift": effective["cold_chain_uplift"],
            "product_category": avg_input.product_category,
            "product_adjustments_applied": avg_input.product_category is not None,
            "data_quality_tier": DataQualityTier.TIER_3.value,
            "record_id": avg_input.record_id,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Average-data calculation: channel=%s, weight=%s t, "
            "transport=%s kgCO2e, storage=%s kgCO2e, total=%s kgCO2e",
            channel_key, avg_input.weight_tonnes,
            transport_co2e, storage_co2e, total_co2e,
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 2: calculate_by_channel
    # ==========================================================================

    def calculate_by_channel(
        self, channel: str, weight_tonnes: Decimal, country: str = "GLOBAL"
    ) -> Dict[str, Any]:
        """
        Calculate emissions using channel defaults (convenience wrapper).

        Simplified interface for a single channel calculation without
        product category adjustments.

        Args:
            channel: Distribution channel name.
            weight_tonnes: Weight of goods in metric tonnes.
            country: Country code for grid EF (default GLOBAL).

        Returns:
            Full result dictionary (same as calculate_average_data).

        Raises:
            ValueError: If channel not recognized or weight <= 0.

        Example:
            >>> result = engine.calculate_by_channel("retail", Decimal("5"))
        """
        avg_input = AverageDataInput(
            channel=channel,
            weight_tonnes=weight_tonnes,
            country=country,
        )
        return self.calculate_average_data(avg_input)

    # ==========================================================================
    # PUBLIC METHOD 3: calculate_by_product_category
    # ==========================================================================

    def calculate_by_product_category(
        self,
        product_category: str,
        weight_tonnes: Decimal,
        channel: str = "retail",
        country: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Calculate emissions with product-category adjustments.

        Applies product-specific multipliers to the channel defaults
        to improve estimate accuracy for different product types.

        Args:
            product_category: Product category key.
            weight_tonnes: Weight of goods in metric tonnes.
            channel: Distribution channel (default retail).
            country: Country code for grid EF.

        Returns:
            Full result dictionary with product adjustments applied.

        Raises:
            ValueError: If product_category or channel not recognized.

        Example:
            >>> result = engine.calculate_by_product_category(
            ...     "perishable", Decimal("5"), "cold_chain", "US"
            ... )
        """
        cat_lower = product_category.lower()
        if cat_lower not in PRODUCT_CATEGORY_ADJUSTMENTS:
            raise ValueError(
                f"Product category '{product_category}' not found. "
                f"Available: {sorted(PRODUCT_CATEGORY_ADJUSTMENTS.keys())}"
            )

        avg_input = AverageDataInput(
            channel=channel,
            weight_tonnes=weight_tonnes,
            country=country,
            product_category=cat_lower,
        )
        return self.calculate_average_data(avg_input)

    # ==========================================================================
    # PUBLIC METHOD 4: calculate_batch
    # ==========================================================================

    def calculate_batch(
        self, inputs: List[AverageDataInput]
    ) -> List[Dict[str, Any]]:
        """
        Calculate emissions for a batch of average-data inputs.

        Processes each input sequentially with error isolation.
        Failed records are excluded and logged at ERROR level.

        Args:
            inputs: List of AverageDataInput records.

        Returns:
            List of result dictionaries (failed records excluded).

        Raises:
            ValueError: If inputs list is empty.

        Example:
            >>> results = engine.calculate_batch([
            ...     AverageDataInput(channel="retail", weight_tonnes=Decimal("10")),
            ...     AverageDataInput(channel="ecommerce", weight_tonnes=Decimal("5")),
            ... ])
            >>> len(results) == 2
            True
        """
        if not inputs:
            raise ValueError("Batch inputs list cannot be empty")

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        error_count = 0

        logger.info(
            "Starting average-data batch: %d records", len(inputs),
        )

        for idx, avg_input in enumerate(inputs):
            try:
                result = self.calculate_average_data(avg_input)
                results.append(result)
            except (ValueError, InvalidOperation) as exc:
                error_count += 1
                logger.error(
                    "Batch record %d failed: %s (channel=%s, weight=%s)",
                    idx, str(exc), avg_input.channel, avg_input.weight_tonnes,
                )

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self._batch_count += 1

        total_co2e = sum(
            (r["total_co2e_kg"] for r in results), ZERO
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        logger.info(
            "Average-data batch complete: %d/%d succeeded, %d failed, "
            "total=%s kgCO2e, duration=%.4fms",
            len(results), len(inputs), error_count,
            total_co2e, duration_ms,
        )

        return results

    # ==========================================================================
    # PUBLIC METHOD 5: calculate_screening
    # ==========================================================================

    def calculate_screening(
        self,
        total_weight_tonnes: Decimal,
        primary_channel: str = "retail",
        country: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Quick screening estimate for total product weight.

        Uses a single channel as representative proxy for the entire
        downstream distribution portfolio. Suitable for rapid screening
        when channel-specific breakdowns are unavailable.

        Args:
            total_weight_tonnes: Total weight of products distributed.
            primary_channel: Representative channel (default retail).
            country: Country code for grid EF.

        Returns:
            Result dictionary with screening flag.

        Raises:
            ValueError: If weight not positive or channel not recognized.

        Example:
            >>> result = engine.calculate_screening(
            ...     Decimal("1000"), "retail", "US"
            ... )
        """
        if total_weight_tonnes <= ZERO:
            raise ValueError(
                f"Total weight must be positive, got {total_weight_tonnes}"
            )

        result = self.calculate_by_channel(
            primary_channel, total_weight_tonnes, country
        )
        result["estimation_type"] = "screening"
        result["note"] = (
            f"Screening estimate using {primary_channel} channel defaults. "
            "For accuracy, provide per-channel weight breakdown."
        )

        logger.info(
            "Screening estimate: weight=%s t, channel=%s, co2e=%s kgCO2e",
            total_weight_tonnes, primary_channel, result["total_co2e_kg"],
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 6: estimate_storage_emissions
    # ==========================================================================

    def estimate_storage_emissions(
        self,
        weight_tonnes: Decimal,
        channel: str = "retail",
        country: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Estimate only the storage/warehouse component of emissions.

        Useful when transport emissions are calculated separately but
        storage emissions need to be estimated using average data.

        Args:
            weight_tonnes: Weight of goods stored in tonnes.
            channel: Distribution channel for storage day defaults.
            country: Country code for grid EF.

        Returns:
            Storage emissions result dictionary.

        Raises:
            ValueError: If channel not recognized or weight <= 0.

        Example:
            >>> result = engine.estimate_storage_emissions(
            ...     Decimal("50"), "wholesale", "DE"
            ... )
        """
        if weight_tonnes <= ZERO:
            raise ValueError(
                f"Weight must be positive, got {weight_tonnes}"
            )

        channel_key = channel.lower()
        channel_defaults = DISTRIBUTION_CHANNEL_DEFAULTS.get(channel_key)
        if channel_defaults is None:
            raise ValueError(
                f"Channel '{channel}' not found. "
                f"Available: {sorted(DISTRIBUTION_CHANNEL_DEFAULTS.keys())}"
            )

        start_time = time.monotonic()

        storage_days = channel_defaults["storage_days"]
        is_cold = channel_defaults["cold_chain"]

        storage_result = _calculate_storage_emissions(
            weight_tonnes=weight_tonnes,
            storage_days=storage_days,
            country=country,
            is_cold_chain=is_cold,
        )

        provenance_hash = _calculate_provenance_hash(
            weight_tonnes, channel_key, country, storage_result["co2e_kg"],
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "channel": channel_key,
            "weight_tonnes": weight_tonnes,
            "country": country,
            "storage_days": storage_days,
            "is_cold_chain": is_cold,
            "storage_result": storage_result,
            "co2e_kg": storage_result["co2e_kg"],
            "co2e_tonnes": (storage_result["co2e_kg"] / THOUSAND).quantize(
                _QUANT_8DP, rounding=ROUNDING
            ),
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Storage estimate: weight=%s t, channel=%s, country=%s, "
            "co2e=%s kgCO2e",
            weight_tonnes, channel_key, country, storage_result["co2e_kg"],
        )

        return result

    # ==========================================================================
    # PUBLIC METHOD 7: compare_channels
    # ==========================================================================

    def compare_channels(
        self,
        weight_tonnes: Decimal,
        country: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Compare emissions across all six distribution channels.

        Calculates emissions for the given weight across every channel
        and ranks them by total CO2e to inform channel optimization.

        Args:
            weight_tonnes: Weight of goods in metric tonnes.
            country: Country code for grid EF.

        Returns:
            Dictionary containing:
                - comparisons: List of per-channel results (sorted by CO2e)
                - lowest_channel: Channel with lowest emissions
                - highest_channel: Channel with highest emissions
                - reduction_potential_pct: % reduction from highest to lowest
                - weight_tonnes: Input weight
                - country: Country used
                - provenance_hash: SHA-256 hash

        Raises:
            ValueError: If weight not positive.

        Example:
            >>> result = engine.compare_channels(Decimal("10"), "US")
            >>> result["lowest_channel"]
            ...
        """
        if weight_tonnes <= ZERO:
            raise ValueError(
                f"Weight must be positive, got {weight_tonnes}"
            )

        start_time = time.monotonic()
        comparisons: List[Dict[str, Any]] = []

        for channel_key in sorted(DISTRIBUTION_CHANNEL_DEFAULTS.keys()):
            try:
                result = self.calculate_by_channel(
                    channel_key, weight_tonnes, country
                )
                comparisons.append({
                    "channel": channel_key,
                    "channel_name": result["channel_name"],
                    "transport_co2e_kg": result["transport_co2e_kg"],
                    "storage_co2e_kg": result["storage_co2e_kg"],
                    "total_co2e_kg": result["total_co2e_kg"],
                    "total_co2e_tonnes": result["total_co2e_tonnes"],
                    "distance_km": result["effective_distance_km"],
                    "storage_days": result["effective_storage_days"],
                    "cold_chain": result["cold_chain_applied"],
                })
            except (ValueError, InvalidOperation) as exc:
                logger.error(
                    "Channel comparison failed for %s: %s",
                    channel_key, str(exc),
                )

        # Sort by total CO2e
        comparisons.sort(key=lambda x: x["total_co2e_kg"])

        lowest = comparisons[0] if comparisons else None
        highest = comparisons[-1] if comparisons else None

        if lowest and highest and highest["total_co2e_kg"] > ZERO:
            reduction = (
                (highest["total_co2e_kg"] - lowest["total_co2e_kg"])
                / highest["total_co2e_kg"] * HUNDRED
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
        else:
            reduction = ZERO

        provenance_hash = _calculate_provenance_hash(
            weight_tonnes, country, len(comparisons),
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0

        comparison_result = {
            "comparisons": comparisons,
            "lowest_channel": lowest["channel"] if lowest else None,
            "lowest_co2e_kg": lowest["total_co2e_kg"] if lowest else ZERO,
            "highest_channel": highest["channel"] if highest else None,
            "highest_co2e_kg": highest["total_co2e_kg"] if highest else ZERO,
            "reduction_potential_pct": reduction,
            "weight_tonnes": weight_tonnes,
            "country": country,
            "channel_count": len(comparisons),
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "processing_time_ms": round(duration_ms, 4),
        }

        logger.info(
            "Channel comparison: weight=%s t, lowest=%s (%s kgCO2e), "
            "highest=%s (%s kgCO2e), reduction=%s%%",
            weight_tonnes,
            lowest["channel"] if lowest else "N/A",
            lowest["total_co2e_kg"] if lowest else "0",
            highest["channel"] if highest else "N/A",
            highest["total_co2e_kg"] if highest else "0",
            reduction,
        )

        return comparison_result

    # ==========================================================================
    # ADDITIONAL PUBLIC METHODS
    # ==========================================================================

    def get_channel_defaults(self) -> List[Dict[str, Any]]:
        """
        Return all distribution channel defaults.

        Returns:
            List of channel default dictionaries.
        """
        result = []
        for key, data in sorted(DISTRIBUTION_CHANNEL_DEFAULTS.items()):
            result.append({
                "channel": key,
                "name": data["name"],
                "description": data["description"],
                "avg_distance_km": float(data["avg_distance_km"]),
                "avg_mode": data["avg_mode"].value if isinstance(data["avg_mode"], Enum) else data["avg_mode"],
                "avg_legs": data["avg_legs"],
                "storage_days": float(data["storage_days"]),
                "cold_chain": data["cold_chain"],
                "avg_load_factor": float(data["avg_load_factor"]),
            })
        return result

    def get_product_categories(self) -> List[Dict[str, Any]]:
        """
        Return all available product category adjustments.

        Returns:
            List of product category adjustment dictionaries.
        """
        result = []
        for key, data in sorted(PRODUCT_CATEGORY_ADJUSTMENTS.items()):
            result.append({
                "category": key,
                "distance_multiplier": float(data["distance_multiplier"]),
                "storage_multiplier": float(data["storage_multiplier"]),
                "cold_chain_override": data["cold_chain_override"],
            })
        return result

    def get_transport_emission_factors(self) -> List[Dict[str, Any]]:
        """
        Return all transport emission factors.

        Returns:
            List of transport EF dictionaries.
        """
        result = []
        for mode, data in sorted(TRANSPORT_EMISSION_FACTORS.items()):
            result.append({
                "mode": mode,
                "name": data["name"],
                "ef_per_tkm": float(data["ef_per_tkm"]),
                "ef_unit": "kgCO2e/tonne-km",
                "wtt_uplift": float(data["wtt_uplift"]),
                "source": data["source"],
            })
        return result

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
            "channels_available": len(DISTRIBUTION_CHANNEL_DEFAULTS),
            "modes_available": len(TRANSPORT_EMISSION_FACTORS),
            "product_categories_available": len(PRODUCT_CATEGORY_ADJUSTMENTS),
            "countries_with_grid_ef": len(GRID_EMISSION_FACTORS),
        }

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _resolve_effective_params(
        self,
        channel_defaults: Dict[str, Any],
        avg_input: AverageDataInput,
    ) -> Dict[str, Any]:
        """
        Resolve effective calculation parameters from defaults, product
        adjustments, and custom overrides.

        Priority: custom overrides > product adjustments > channel defaults.

        Args:
            channel_defaults: Base channel default values.
            avg_input: User input with optional overrides.

        Returns:
            Dictionary of effective parameters.
        """
        # Start with channel defaults
        distance_km = channel_defaults["avg_distance_km"]
        mode = channel_defaults["avg_mode"]
        if isinstance(mode, Enum):
            mode = mode.value
        legs = channel_defaults["avg_legs"]
        storage_days = channel_defaults["storage_days"]
        is_cold = channel_defaults["cold_chain"]
        cold_uplift = channel_defaults["cold_chain_uplift"]
        load_factor = channel_defaults["avg_load_factor"]

        # Apply product category adjustments
        if avg_input.product_category:
            cat_adj = PRODUCT_CATEGORY_ADJUSTMENTS.get(
                avg_input.product_category.lower()
            )
            if cat_adj:
                distance_km = (
                    distance_km * cat_adj["distance_multiplier"]
                ).quantize(_QUANT_8DP, rounding=ROUNDING)
                storage_days = (
                    storage_days * cat_adj["storage_multiplier"]
                ).quantize(_QUANT_8DP, rounding=ROUNDING)
                if cat_adj["cold_chain_override"]:
                    is_cold = True
                    cold_uplift = cat_adj["cold_chain_uplift"]

        # Apply custom overrides (highest priority)
        if avg_input.custom_distance_km is not None:
            distance_km = avg_input.custom_distance_km
        if avg_input.custom_storage_days is not None:
            storage_days = avg_input.custom_storage_days
        if avg_input.custom_mode is not None:
            mode = avg_input.custom_mode.lower()

        return {
            "distance_km": distance_km,
            "mode": mode,
            "legs": legs,
            "storage_days": storage_days,
            "is_cold_chain": is_cold,
            "cold_chain_uplift": cold_uplift,
            "load_factor": load_factor,
        }


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_average_data_calculator() -> AverageDataCalculatorEngine:
    """
    Get the AverageDataCalculatorEngine singleton instance.

    Returns:
        AverageDataCalculatorEngine singleton.
    """
    return AverageDataCalculatorEngine.get_instance()


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
    "DistributionChannel",
    "TransportMode",
    "ProductCategory",
    "EFSource",
    "DataQualityTier",
    # Data Tables
    "DISTRIBUTION_CHANNEL_DEFAULTS",
    "TRANSPORT_EMISSION_FACTORS",
    "GRID_EMISSION_FACTORS",
    "PRODUCT_CATEGORY_ADJUSTMENTS",
    "WAREHOUSE_ENERGY_INTENSITY",
    "COLD_STORAGE_ENERGY_INTENSITY",
    "WAREHOUSE_AREA_PER_TONNE",
    # Input Model
    "AverageDataInput",
    # Engine
    "AverageDataCalculatorEngine",
    "get_average_data_calculator",
]
