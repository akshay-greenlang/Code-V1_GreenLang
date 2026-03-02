# -*- coding: utf-8 -*-
"""
LifetimeModelingEngine - Engine 5: Use of Sold Products (AGENT-MRV-024)

This module implements the LifetimeModelingEngine for AGENT-MRV-024
(Use of Sold Products, GHG Protocol Scope 3 Category 11). It provides
thread-safe singleton lifetime estimation, usage patterns, degradation
modeling, and fleet survival curves for products sold by the reporting company.

Key Capabilities:
    - Default lifetime tables by product category and type (years)
    - Lifetime adjustment factors (standard, heavy, light, industrial, seasonal)
    - Energy degradation curves (year-by-year efficiency loss)
    - Weibull distribution survival curves for fleet-level modeling
    - Fleet emissions with survival probability weighting
    - Discounted emissions with present-value reduction (optional)
    - Repair/replacement impact on lifetime extension
    - Annual usage profiles (hours/year, km/year, cycles/year)
    - Effective lifetime computation with adjustments and repair

Weibull Shape Parameters:
    - Vehicles:              2.5 (moderate infant mortality, gradual wear-out)
    - Appliances:            3.0 (symmetric, bell-shaped failure)
    - HVAC:                  2.0 (earlier failures from compressor wear)
    - Electronics:           4.0 (tight distribution, late sudden failure)
    - Industrial Equipment:  1.5 (mixed failure modes)

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal with ROUND_HALF_UP to 8 decimal places
    - No LLM calls anywhere in the numeric calculation path
    - Every intermediate value is deterministic and reproducible
    - Weibull CDF computed via Python math.exp for float precision
    - Results immediately converted back to Decimal

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation. All mutable state is protected by dedicated locks.

Example:
    >>> from greenlang.use_of_sold_products.lifetime_modeling import (
    ...     LifetimeModelingEngine,
    ... )
    >>> engine = LifetimeModelingEngine()
    >>> lifetime = engine.get_default_lifetime("APPLIANCES", "refrigerator")
    >>> assert lifetime == 15

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-011
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
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_usp_"
ENGINE_NAME: str = "LifetimeModelingEngine"
ENGINE_NUMBER: int = 5

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUANT_8DP = _PRECISION


def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class ProductCategory(str, Enum):
    """Product categories for lifetime modeling."""

    VEHICLES = "VEHICLES"
    APPLIANCES = "APPLIANCES"
    HVAC = "HVAC"
    LIGHTING = "LIGHTING"
    IT_EQUIPMENT = "IT_EQUIPMENT"
    INDUSTRIAL_EQUIPMENT = "INDUSTRIAL_EQUIPMENT"
    BUILDING_PRODUCTS = "BUILDING_PRODUCTS"
    CONSUMER_PRODUCTS = "CONSUMER_PRODUCTS"
    MEDICAL_DEVICES = "MEDICAL_DEVICES"
    FUELS_FEEDSTOCKS = "FUELS_FEEDSTOCKS"


class LifetimeAdjustment(str, Enum):
    """Product lifetime adjustment factors based on usage intensity."""

    STANDARD = "STANDARD"       # 1.00 -- default usage assumption
    HEAVY = "HEAVY"             # 0.80 -- commercial/fleet, high utilization
    LIGHT = "LIGHT"             # 1.20 -- light residential, low utilization
    INDUSTRIAL = "INDUSTRIAL"   # 0.60 -- continuous 24/7 industrial use
    SEASONAL = "SEASONAL"       # 0.50 -- seasonal use only (AC in temperate)


class UsageUnit(str, Enum):
    """Units for annual usage profiles."""

    HOURS_PER_YEAR = "hours/year"
    KM_PER_YEAR = "km/year"
    CYCLES_PER_YEAR = "cycles/year"
    KWH_PER_YEAR = "kWh/year"
    LITERS_PER_YEAR = "liters/year"
    M3_PER_YEAR = "m3/year"


# ==============================================================================
# REFERENCE DATA TABLES
# ==============================================================================

# Default product lifetimes (years) by category and product type
# Source: GHG Protocol Scope 3 guidance, ENERGY STAR, EU Ecodesign
DEFAULT_LIFETIMES: Dict[str, Dict[str, int]] = {
    "VEHICLES": {
        "passenger_car_gasoline": 15,
        "passenger_car_diesel": 15,
        "passenger_car_ev": 15,
        "light_truck": 15,
        "heavy_truck": 10,
        "motorcycle": 12,
        "bus": 12,
        "aircraft_engine": 20,
    },
    "APPLIANCES": {
        "refrigerator": 15,
        "washing_machine": 12,
        "dishwasher": 12,
        "dryer": 12,
        "oven": 15,
        "microwave": 10,
        "vacuum_cleaner": 8,
        "air_purifier": 10,
    },
    "HVAC": {
        "room_ac": 12,
        "central_ac": 15,
        "heat_pump": 15,
        "gas_furnace": 20,
        "electric_heater": 15,
        "chiller": 20,
        "cooling_tower": 25,
    },
    "LIGHTING": {
        "led_bulb": 15,
        "cfl_bulb": 8,
        "fluorescent_tube": 10,
        "halogen": 3,
        "smart_lighting": 10,
        "industrial_led": 15,
    },
    "IT_EQUIPMENT": {
        "laptop": 5,
        "desktop": 6,
        "server": 5,
        "monitor": 7,
        "printer": 7,
        "network_switch": 8,
        "storage_array": 5,
        "ups": 10,
    },
    "INDUSTRIAL_EQUIPMENT": {
        "diesel_generator": 15,
        "gas_boiler": 20,
        "compressor": 15,
        "electric_motor": 20,
        "pump": 15,
        "turbine": 25,
        "transformer": 30,
    },
    "BUILDING_PRODUCTS": {
        "window": 25,
        "insulation": 50,
        "hvac_duct": 30,
        "roofing": 25,
        "water_heater": 12,
        "solar_panel": 25,
    },
    "CONSUMER_PRODUCTS": {
        "aerosol": 1,
        "cleaning_agent": 1,
        "fertilizer": 1,
        "paint": 1,
        "adhesive": 1,
    },
    "MEDICAL_DEVICES": {
        "imaging_equipment": 10,
        "ventilator": 8,
        "lab_equipment": 10,
        "diagnostic_device": 7,
        "patient_monitor": 8,
    },
    "FUELS_FEEDSTOCKS": {
        "gasoline": 0,
        "diesel": 0,
        "natural_gas": 0,
        "coal": 0,
        "hydrogen": 0,
    },
}

# Lifetime adjustment factors
ADJUSTMENT_FACTORS: Dict[str, Decimal] = {
    "STANDARD": Decimal("1.00"),
    "HEAVY": Decimal("0.80"),
    "LIGHT": Decimal("1.20"),
    "INDUSTRIAL": Decimal("0.60"),
    "SEASONAL": Decimal("0.50"),
}

# Energy degradation rates by product category (annual fraction)
DEGRADATION_RATES: Dict[str, Decimal] = {
    "VEHICLES": Decimal("0.015"),
    "APPLIANCES": Decimal("0.005"),
    "HVAC": Decimal("0.010"),
    "LIGHTING": Decimal("0.020"),
    "IT_EQUIPMENT": Decimal("0.000"),
    "INDUSTRIAL_EQUIPMENT": Decimal("0.010"),
    "BUILDING_PRODUCTS": Decimal("0.003"),
    "CONSUMER_PRODUCTS": Decimal("0.000"),
    "MEDICAL_DEVICES": Decimal("0.002"),
    "FUELS_FEEDSTOCKS": Decimal("0.000"),
}

# Weibull shape parameters by product category
# Higher shape = tighter distribution around characteristic life
# Shape=1.0 -> exponential (constant failure rate)
# Shape=2.0 -> Rayleigh (linearly increasing failure rate)
# Shape>3.0 -> tight bell shape (most failures near characteristic life)
WEIBULL_SHAPE_PARAMETERS: Dict[str, Decimal] = {
    "VEHICLES": Decimal("2.5"),
    "APPLIANCES": Decimal("3.0"),
    "HVAC": Decimal("2.0"),
    "LIGHTING": Decimal("3.0"),
    "IT_EQUIPMENT": Decimal("4.0"),
    "INDUSTRIAL_EQUIPMENT": Decimal("1.5"),
    "BUILDING_PRODUCTS": Decimal("2.5"),
    "CONSUMER_PRODUCTS": Decimal("3.0"),
    "MEDICAL_DEVICES": Decimal("3.0"),
    "FUELS_FEEDSTOCKS": Decimal("1.0"),
}

# Annual usage profiles by category and product type
# Provides typical annual usage in the most relevant unit
USAGE_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "VEHICLES": {
        "passenger_car_gasoline": {
            "value": Decimal("15000"),
            "unit": "km/year",
            "description": "Average annual driving distance",
        },
        "passenger_car_diesel": {
            "value": Decimal("20000"),
            "unit": "km/year",
            "description": "Average annual driving distance (diesel, higher mileage)",
        },
        "passenger_car_ev": {
            "value": Decimal("15000"),
            "unit": "km/year",
            "description": "Average annual driving distance (EV)",
        },
        "light_truck": {
            "value": Decimal("20000"),
            "unit": "km/year",
            "description": "Average annual driving distance",
        },
        "heavy_truck": {
            "value": Decimal("80000"),
            "unit": "km/year",
            "description": "Average annual driving distance (commercial)",
        },
        "motorcycle": {
            "value": Decimal("5000"),
            "unit": "km/year",
            "description": "Average annual riding distance",
        },
    },
    "APPLIANCES": {
        "refrigerator": {
            "value": Decimal("8760"),
            "unit": "hours/year",
            "description": "Continuous operation (24/7)",
        },
        "washing_machine": {
            "value": Decimal("300"),
            "unit": "cycles/year",
            "description": "Average wash cycles per year",
        },
        "dishwasher": {
            "value": Decimal("250"),
            "unit": "cycles/year",
            "description": "Average wash cycles per year",
        },
        "dryer": {
            "value": Decimal("250"),
            "unit": "cycles/year",
            "description": "Average drying cycles per year",
        },
        "oven": {
            "value": Decimal("500"),
            "unit": "hours/year",
            "description": "Average cooking hours per year",
        },
    },
    "HVAC": {
        "room_ac": {
            "value": Decimal("1500"),
            "unit": "hours/year",
            "description": "Cooling season hours",
        },
        "central_ac": {
            "value": Decimal("2000"),
            "unit": "hours/year",
            "description": "Cooling season hours (larger system)",
        },
        "heat_pump": {
            "value": Decimal("3000"),
            "unit": "hours/year",
            "description": "Combined heating and cooling hours",
        },
        "gas_furnace": {
            "value": Decimal("2000"),
            "unit": "hours/year",
            "description": "Heating season hours",
        },
    },
    "IT_EQUIPMENT": {
        "laptop": {
            "value": Decimal("2000"),
            "unit": "hours/year",
            "description": "Average usage hours (office worker)",
        },
        "desktop": {
            "value": Decimal("2500"),
            "unit": "hours/year",
            "description": "Average usage hours",
        },
        "server": {
            "value": Decimal("8760"),
            "unit": "hours/year",
            "description": "Continuous operation (24/7)",
        },
        "monitor": {
            "value": Decimal("2500"),
            "unit": "hours/year",
            "description": "Average display hours",
        },
    },
    "INDUSTRIAL_EQUIPMENT": {
        "diesel_generator": {
            "value": Decimal("4000"),
            "unit": "hours/year",
            "description": "Average operational hours",
        },
        "gas_boiler": {
            "value": Decimal("3000"),
            "unit": "hours/year",
            "description": "Average operational hours",
        },
        "compressor": {
            "value": Decimal("5000"),
            "unit": "hours/year",
            "description": "Average operational hours",
        },
    },
    "LIGHTING": {
        "led_bulb": {
            "value": Decimal("3000"),
            "unit": "hours/year",
            "description": "Average illumination hours (residential)",
        },
        "cfl_bulb": {
            "value": Decimal("3000"),
            "unit": "hours/year",
            "description": "Average illumination hours",
        },
        "industrial_led": {
            "value": Decimal("6000"),
            "unit": "hours/year",
            "description": "Industrial/commercial illumination hours",
        },
    },
}

# Lifetime bounds for validation (min, max years)
LIFETIME_BOUNDS: Dict[str, Tuple[int, int]] = {
    "VEHICLES": (3, 30),
    "APPLIANCES": (3, 25),
    "HVAC": (5, 35),
    "LIGHTING": (1, 25),
    "IT_EQUIPMENT": (2, 15),
    "INDUSTRIAL_EQUIPMENT": (5, 40),
    "BUILDING_PRODUCTS": (5, 60),
    "CONSUMER_PRODUCTS": (1, 5),
    "MEDICAL_DEVICES": (3, 20),
    "FUELS_FEEDSTOCKS": (0, 1),
}


# ==============================================================================
# HASH UTILITY
# ==============================================================================


def _compute_provenance_hash(*parts: Any) -> str:
    """
    Compute SHA-256 provenance hash from variable inputs.

    Args:
        *parts: Variable number of inputs to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for part in parts:
        if isinstance(part, Decimal):
            hash_input += str(part.quantize(_PRECISION, rounding=ROUND_HALF_UP))
        elif isinstance(part, dict):
            hash_input += json.dumps(part, sort_keys=True, default=str)
        elif isinstance(part, (list, tuple)):
            hash_input += json.dumps(
                [str(x) if isinstance(x, Decimal) else x for x in part],
                sort_keys=True,
                default=str,
            )
        else:
            hash_input += str(part)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class LifetimeModelingEngine:
    """
    Thread-safe singleton engine for product lifetime modeling.

    Provides lifetime estimation, usage patterns, degradation curves,
    Weibull survival curves, fleet-level emissions, and repair/replacement
    impact modeling for GHG Protocol Scope 3 Category 11 calculations.

    All arithmetic uses Python Decimal with ROUND_HALF_UP quantization to
    8 decimal places for regulatory precision. Trigonometric and exponential
    functions use Python math module (float), with results immediately
    converted back to Decimal.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The
        _query_count is protected by a dedicated lock.

    Attributes:
        _query_count: Total number of queries/calculations performed.
        _count_lock: Lock protecting the query counter.

    Example:
        >>> engine = LifetimeModelingEngine()
        >>> lifetime = engine.get_default_lifetime("APPLIANCES", "refrigerator")
        >>> assert lifetime == 15
        >>> curve = engine.compute_survival_curve(15, shape=Decimal("3.0"))
        >>> len(curve) == 15
        True
    """

    _instance: Optional["LifetimeModelingEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "LifetimeModelingEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the lifetime modeling engine (once)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._query_count: int = 0
        self._count_lock: threading.Lock = threading.Lock()

        # Count total product types across all categories
        total_types = sum(len(v) for v in DEFAULT_LIFETIMES.values())

        logger.info(
            "%s initialized: agent=%s, version=%s, "
            "categories=%d, product_types=%d, adjustments=%d, "
            "weibull_shapes=%d",
            ENGINE_NAME,
            AGENT_ID,
            VERSION,
            len(DEFAULT_LIFETIMES),
            total_types,
            len(ADJUSTMENT_FACTORS),
            len(WEIBULL_SHAPE_PARAMETERS),
        )

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _increment_count(self) -> int:
        """
        Increment and return the query counter thread-safely.

        Returns:
            Updated query count.
        """
        with self._count_lock:
            self._query_count += 1
            return self._query_count

    # ==========================================================================
    # DEFAULT LIFETIME LOOKUP
    # ==========================================================================

    def get_default_lifetime(
        self,
        category: str,
        product_type: str,
    ) -> int:
        """
        Get the default lifetime in years for a product category and type.

        Looks up the default lifetime from the reference table. If the
        product_type is not found, returns the median lifetime for the
        category. If the category is not found, returns 10 years as
        a conservative global default.

        Args:
            category: Product category code (e.g., "APPLIANCES", "VEHICLES").
            product_type: Specific product type (e.g., "refrigerator", "laptop").

        Returns:
            Default lifetime in years (integer).

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> engine.get_default_lifetime("APPLIANCES", "refrigerator")
            15
            >>> engine.get_default_lifetime("IT_EQUIPMENT", "laptop")
            5
            >>> engine.get_default_lifetime("UNKNOWN", "widget")
            10
        """
        self._increment_count()
        category_upper = category.upper()
        product_lower = product_type.lower()

        category_lifetimes = DEFAULT_LIFETIMES.get(category_upper)
        if category_lifetimes is None:
            logger.warning(
                "Category '%s' not found in DEFAULT_LIFETIMES, "
                "returning global default 10 years",
                category_upper,
            )
            return 10

        lifetime = category_lifetimes.get(product_lower)
        if lifetime is not None:
            logger.debug(
                "Default lifetime for %s/%s: %d years",
                category_upper, product_lower, lifetime,
            )
            return lifetime

        # Fallback: median of category lifetimes
        all_lifetimes = sorted(category_lifetimes.values())
        if not all_lifetimes:
            return 10

        median_idx = len(all_lifetimes) // 2
        median_lifetime = all_lifetimes[median_idx]

        logger.warning(
            "Product type '%s' not found in category '%s', "
            "using category median: %d years",
            product_lower, category_upper, median_lifetime,
        )

        return median_lifetime

    # ==========================================================================
    # LIFETIME ADJUSTMENT
    # ==========================================================================

    def get_adjustment_factor(
        self,
        adjustment: str,
    ) -> Decimal:
        """
        Get the lifetime adjustment multiplier for a usage intensity code.

        Args:
            adjustment: Adjustment code (STANDARD, HEAVY, LIGHT,
                       INDUSTRIAL, SEASONAL).

        Returns:
            Adjustment multiplier as Decimal.

        Raises:
            ValueError: If adjustment code is not recognized.

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> engine.get_adjustment_factor("HEAVY")
            Decimal('0.80')
            >>> engine.get_adjustment_factor("LIGHT")
            Decimal('1.20')
        """
        adj_upper = adjustment.upper()
        factor = ADJUSTMENT_FACTORS.get(adj_upper)
        if factor is None:
            available = sorted(ADJUSTMENT_FACTORS.keys())
            raise ValueError(
                f"Unknown adjustment '{adj_upper}'. Available: {available}"
            )
        return factor

    def apply_adjustment(
        self,
        lifetime_years: int,
        adjustment: str,
    ) -> int:
        """
        Apply a lifetime adjustment factor to a base lifetime.

        Multiplies the base lifetime by the adjustment factor and rounds
        to the nearest integer. The result is clamped to a minimum of 1 year.

        Args:
            lifetime_years: Base lifetime in years.
            adjustment: Adjustment code (STANDARD, HEAVY, LIGHT,
                       INDUSTRIAL, SEASONAL).

        Returns:
            Adjusted lifetime in years (integer, minimum 1).

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> engine.apply_adjustment(15, "STANDARD")
            15
            >>> engine.apply_adjustment(15, "HEAVY")
            12
            >>> engine.apply_adjustment(15, "LIGHT")
            18
            >>> engine.apply_adjustment(15, "INDUSTRIAL")
            9
            >>> engine.apply_adjustment(15, "SEASONAL")
            8
        """
        factor = self.get_adjustment_factor(adjustment)
        adjusted_dec = _q(Decimal(str(lifetime_years)) * factor)
        adjusted_int = int(adjusted_dec.quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ))

        # Minimum 1 year
        if adjusted_int < 1:
            adjusted_int = 1

        logger.debug(
            "Lifetime adjustment: %d years x %s (%s) = %d years",
            lifetime_years, factor, adjustment, adjusted_int,
        )

        return adjusted_int

    # ==========================================================================
    # DEGRADATION MODELING
    # ==========================================================================

    def get_degradation_rate(
        self,
        category: str,
    ) -> Decimal:
        """
        Get the annual energy degradation rate for a product category.

        Args:
            category: Product category code.

        Returns:
            Degradation rate as Decimal fraction (e.g., 0.015 = 1.5%).

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> engine.get_degradation_rate("VEHICLES")
            Decimal('0.015')
            >>> engine.get_degradation_rate("IT_EQUIPMENT")
            Decimal('0.000')
        """
        category_upper = category.upper()
        rate = DEGRADATION_RATES.get(category_upper)
        if rate is None:
            logger.warning(
                "Category '%s' not in DEGRADATION_RATES, using default 0.5%%",
                category_upper,
            )
            rate = Decimal("0.005")
        return rate

    def compute_degradation_curve(
        self,
        base_annual_consumption: Decimal,
        lifetime_years: int,
        degradation_rate: Decimal,
    ) -> List[Decimal]:
        """
        Compute year-by-year energy consumption with degradation.

        Each year's consumption is reduced by the cumulative degradation:
            consumption_year_t = base x (1 - rate) ^ t

        Args:
            base_annual_consumption: Base annual consumption (year 0).
            lifetime_years: Product lifetime in years.
            degradation_rate: Annual degradation rate fraction.

        Returns:
            List of Decimal values, one per year (year 0 to year L-1).

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> curve = engine.compute_degradation_curve(
            ...     Decimal("1000"), 5, Decimal("0.01")
            ... )
            >>> len(curve) == 5
            True
            >>> curve[0] == Decimal("1000.00000000")
            True
            >>> curve[4] < curve[0]
            True
        """
        curve: List[Decimal] = []
        decay_factor = _ONE - degradation_rate

        for year in range(lifetime_years):
            if degradation_rate == _ZERO:
                value = _q(base_annual_consumption)
            else:
                value = _q(base_annual_consumption * _q(decay_factor ** year))
            curve.append(value)

        return curve

    def compute_total_lifetime_consumption(
        self,
        base_annual_consumption: Decimal,
        lifetime_years: int,
        degradation_rate: Decimal,
    ) -> Decimal:
        """
        Compute total lifetime consumption with year-by-year degradation.

        Sums the degradation curve over all years of the product lifetime.

        Args:
            base_annual_consumption: Base annual consumption (year 0).
            lifetime_years: Product lifetime in years.
            degradation_rate: Annual degradation rate.

        Returns:
            Total lifetime consumption, quantized to 8 dp.

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> total = engine.compute_total_lifetime_consumption(
            ...     Decimal("400"), 15, Decimal("0.005")
            ... )
            >>> total > Decimal("0")
            True
            >>> # Without degradation, total = 400 x 15 = 6000
            >>> total_no_deg = engine.compute_total_lifetime_consumption(
            ...     Decimal("400"), 15, Decimal("0.000")
            ... )
            >>> total_no_deg == Decimal("6000.00000000")
            True
        """
        if degradation_rate == _ZERO:
            return _q(base_annual_consumption * Decimal(str(lifetime_years)))

        total = _ZERO
        decay_factor = _ONE - degradation_rate

        for year in range(lifetime_years):
            year_value = _q(base_annual_consumption * _q(decay_factor ** year))
            total = total + year_value

        return _q(total)

    # ==========================================================================
    # WEIBULL SURVIVAL CURVES
    # ==========================================================================

    def compute_survival_curve(
        self,
        lifetime_years: int,
        shape: Decimal = Decimal("2.0"),
    ) -> List[Decimal]:
        """
        Compute Weibull survival probability for each year.

        The Weibull survival function is:
            S(t) = exp(-(t/eta)^beta)
        where:
            beta = shape parameter
            eta  = scale parameter = lifetime / Gamma(1 + 1/beta)

        For simplicity, we set eta = lifetime (characteristic life),
        which means approximately 63.2% of units have failed by year = lifetime.

        The survival probability gives the fraction of units still in
        service at the beginning of each year.

        Args:
            lifetime_years: Characteristic lifetime in years (eta).
            shape: Weibull shape parameter (beta).

        Returns:
            List of Decimal survival probabilities, one per year
            (year 1 through year = lifetime_years). The probability at
            year 0 is implicitly 1.0.

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> curve = engine.compute_survival_curve(15, Decimal("3.0"))
            >>> len(curve) == 15
            True
            >>> curve[0] > Decimal("0.99")  # Almost all survive year 1
            True
            >>> curve[-1] < Decimal("0.5")  # Fewer survive the final year
            True
        """
        if lifetime_years <= 0:
            return []

        eta_float = float(lifetime_years)
        beta_float = float(shape)
        curve: List[Decimal] = []

        for year in range(1, lifetime_years + 1):
            t = float(year)
            # S(t) = exp(-(t/eta)^beta)
            exponent = -((t / eta_float) ** beta_float)
            survival_prob = math.exp(exponent)
            # Convert back to Decimal immediately
            curve.append(_q(Decimal(str(survival_prob))))

        return curve

    def get_weibull_shape(
        self,
        category: str,
    ) -> Decimal:
        """
        Get the Weibull shape parameter for a product category.

        Args:
            category: Product category code.

        Returns:
            Weibull shape parameter (beta) as Decimal.

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> engine.get_weibull_shape("VEHICLES")
            Decimal('2.5')
            >>> engine.get_weibull_shape("IT_EQUIPMENT")
            Decimal('4.0')
        """
        category_upper = category.upper()
        shape = WEIBULL_SHAPE_PARAMETERS.get(category_upper)
        if shape is None:
            logger.warning(
                "Category '%s' not in WEIBULL_SHAPE_PARAMETERS, "
                "using default shape=2.0",
                category_upper,
            )
            shape = Decimal("2.0")
        return shape

    # ==========================================================================
    # FLEET-LEVEL EMISSIONS
    # ==========================================================================

    def compute_fleet_emissions(
        self,
        units_sold: int,
        annual_emission_per_unit: Decimal,
        lifetime_years: int,
        survival_shape: Decimal = Decimal("2.0"),
    ) -> Dict[str, Any]:
        """
        Compute fleet-level emissions weighted by Weibull survival probability.

        For each year of the product lifetime, the fleet emissions are:
            E_year_t = units_sold x annual_emission x S(t)
        where S(t) is the Weibull survival probability at year t.

        Total fleet emissions are the sum over all years.

        This approach accounts for the fact that not all units survive
        to the end of their expected lifetime.

        Args:
            units_sold: Number of units sold in the reporting period.
            annual_emission_per_unit: Annual emissions per unit (kgCO2e).
            lifetime_years: Characteristic lifetime in years.
            survival_shape: Weibull shape parameter.

        Returns:
            Dictionary with:
            - total_fleet_co2e: Total fleet emissions (kgCO2e)
            - annual_fleet_emissions: List of annual fleet emissions
            - survival_curve: List of survival probabilities
            - naive_total: Total without survival weighting (for comparison)
            - survival_reduction_pct: Reduction due to survival weighting

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> result = engine.compute_fleet_emissions(
            ...     units_sold=10000,
            ...     annual_emission_per_unit=Decimal("100"),
            ...     lifetime_years=15,
            ...     survival_shape=Decimal("3.0"),
            ... )
            >>> result["total_fleet_co2e"] > Decimal("0")
            True
            >>> result["total_fleet_co2e"] < result["naive_total"]
            True
        """
        self._increment_count()
        units_dec = Decimal(str(units_sold))

        # Compute survival curve
        survival_curve = self.compute_survival_curve(lifetime_years, survival_shape)

        # Year 0: all units operational (survival=1.0)
        year_0_emissions = _q(units_dec * annual_emission_per_unit)
        annual_emissions: List[Decimal] = [year_0_emissions]
        total_fleet = year_0_emissions

        # Years 1 through lifetime-1: weighted by survival
        for i, survival_prob in enumerate(survival_curve[:-1]):
            year_emission = _q(units_dec * annual_emission_per_unit * survival_prob)
            annual_emissions.append(year_emission)
            total_fleet = _q(total_fleet + year_emission)

        # Naive total (no survival weighting)
        naive_total = _q(
            units_dec * annual_emission_per_unit * Decimal(str(lifetime_years))
        )

        # Reduction percentage
        if naive_total > _ZERO:
            reduction_pct = _q(
                (_ONE - total_fleet / naive_total) * Decimal("100")
            )
        else:
            reduction_pct = _ZERO

        logger.debug(
            "Fleet emissions: %d units x %s kgCO2e/yr x %d yrs: "
            "naive=%s, fleet=%s (-%s%%)",
            units_sold,
            annual_emission_per_unit,
            lifetime_years,
            naive_total,
            total_fleet,
            reduction_pct,
        )

        return {
            "total_fleet_co2e": total_fleet,
            "annual_fleet_emissions": annual_emissions,
            "survival_curve": [_ONE] + survival_curve,
            "naive_total": naive_total,
            "survival_reduction_pct": reduction_pct,
            "units_sold": units_sold,
            "lifetime_years": lifetime_years,
            "survival_shape": survival_shape,
        }

    # ==========================================================================
    # DISCOUNTED EMISSIONS
    # ==========================================================================

    def compute_discounted_emissions(
        self,
        annual_emission: Decimal,
        lifetime_years: int,
        discount_rate: Decimal = Decimal("0.0"),
    ) -> Dict[str, Decimal]:
        """
        Compute total discounted emissions over a product lifetime.

        Applies a time-value discount to future emissions:
            E_discounted = SUM_t( annual_emission / (1 + rate)^t )
        for t = 0 to lifetime-1.

        If discount_rate is 0, returns undiscounted total.

        Args:
            annual_emission: Annual emission value (kgCO2e).
            lifetime_years: Product lifetime in years.
            discount_rate: Annual discount rate (e.g., 0.03 for 3%).

        Returns:
            Dictionary with:
            - discounted_total: Present-value weighted total
            - undiscounted_total: Simple sum (annual x lifetime)
            - discount_factor_applied: Whether discounting was applied

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> result = engine.compute_discounted_emissions(
            ...     Decimal("100"), 15, Decimal("0.03")
            ... )
            >>> result["discounted_total"] < result["undiscounted_total"]
            True
        """
        self._increment_count()

        undiscounted_total = _q(
            annual_emission * Decimal(str(lifetime_years))
        )

        if discount_rate == _ZERO:
            return {
                "discounted_total": undiscounted_total,
                "undiscounted_total": undiscounted_total,
                "discount_factor_applied": False,
            }

        discounted_total = _ZERO
        for year in range(lifetime_years):
            discount_factor = _q(
                _ONE / _q((_ONE + discount_rate) ** year)
            )
            year_emission = _q(annual_emission * discount_factor)
            discounted_total = _q(discounted_total + year_emission)

        return {
            "discounted_total": discounted_total,
            "undiscounted_total": undiscounted_total,
            "discount_factor_applied": True,
        }

    # ==========================================================================
    # REPAIR / REPLACEMENT IMPACT
    # ==========================================================================

    def estimate_replacement_impact(
        self,
        base_lifetime_years: int,
        repair_rate: Decimal,
        extension_years: int,
    ) -> Dict[str, Any]:
        """
        Estimate the impact of repair/maintenance on product lifetime extension.

        Models how regular repair and maintenance can extend the effective
        product lifetime, reducing the number of replacement units needed
        and potentially changing total fleet emissions.

        The effective lifetime is:
            effective = base + extension * repair_rate
        where repair_rate is the fraction of units that receive repair
        (0 to 1), and extension_years is the maximum additional life
        achievable through repair.

        Args:
            base_lifetime_years: Base product lifetime in years.
            repair_rate: Fraction of units receiving repair (0-1).
            extension_years: Maximum additional years from repair.

        Returns:
            Dictionary with:
            - effective_lifetime: Effective lifetime after repair (years, int)
            - base_lifetime: Original base lifetime
            - extension_achieved: Additional years achieved
            - repair_rate: Fraction of units repaired
            - replacement_reduction_pct: Reduction in replacement frequency

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> result = engine.estimate_replacement_impact(
            ...     base_lifetime_years=15,
            ...     repair_rate=Decimal("0.5"),
            ...     extension_years=5,
            ... )
            >>> result["effective_lifetime"]
            18
        """
        self._increment_count()

        # Extension achieved = extension_years x repair_rate
        extension_dec = _q(
            Decimal(str(extension_years)) * repair_rate
        )
        extension_int = int(extension_dec.quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ))

        effective_lifetime = base_lifetime_years + extension_int

        # Replacement reduction percentage
        if base_lifetime_years > 0:
            reduction_pct = _q(
                Decimal(str(extension_int))
                / Decimal(str(base_lifetime_years))
                * Decimal("100")
            )
        else:
            reduction_pct = _ZERO

        logger.debug(
            "Replacement impact: base=%d yrs, repair_rate=%s, "
            "extension=%d yrs -> effective=%d yrs (-%s%% replacement)",
            base_lifetime_years,
            repair_rate,
            extension_int,
            effective_lifetime,
            reduction_pct,
        )

        return {
            "effective_lifetime": effective_lifetime,
            "base_lifetime": base_lifetime_years,
            "extension_achieved": extension_int,
            "repair_rate": repair_rate,
            "replacement_reduction_pct": reduction_pct,
        }

    # ==========================================================================
    # USAGE PROFILES
    # ==========================================================================

    def get_usage_profile(
        self,
        category: str,
        product_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the annual usage profile for a product category and type.

        Returns the typical annual usage (hours/year, km/year, cycles/year)
        with a description of what the value represents.

        Args:
            category: Product category code.
            product_type: Specific product type.

        Returns:
            Usage profile dict with value, unit, description; or None if
            not found.

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> profile = engine.get_usage_profile("VEHICLES", "passenger_car_gasoline")
            >>> profile["value"]
            Decimal('15000')
            >>> profile["unit"]
            'km/year'
        """
        self._increment_count()
        category_upper = category.upper()
        product_lower = product_type.lower()

        category_profiles = USAGE_PROFILES.get(category_upper)
        if category_profiles is None:
            logger.debug(
                "No usage profiles for category '%s'", category_upper
            )
            return None

        profile = category_profiles.get(product_lower)
        if profile is None:
            logger.debug(
                "No usage profile for %s/%s", category_upper, product_lower
            )
            return None

        return {
            "value": profile["value"],
            "unit": profile["unit"],
            "description": profile["description"],
            "category": category_upper,
            "product_type": product_lower,
        }

    # ==========================================================================
    # EFFECTIVE LIFETIME
    # ==========================================================================

    def compute_effective_lifetime(
        self,
        base_lifetime_years: int,
        adjustment: str = "STANDARD",
        repair_rate: Decimal = _ZERO,
        extension_years: int = 0,
    ) -> int:
        """
        Compute the effective product lifetime with adjustments and repair.

        Combines the base lifetime with a usage intensity adjustment and
        optional repair-based lifetime extension:
            adjusted = base x adjustment_factor
            effective = adjusted + extension_years x repair_rate

        Args:
            base_lifetime_years: Base lifetime in years.
            adjustment: Usage intensity adjustment code.
            repair_rate: Fraction of units receiving repair (0-1).
            extension_years: Maximum additional years from repair.

        Returns:
            Effective lifetime in years (integer, minimum 1).

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> engine.compute_effective_lifetime(15, "HEAVY", Decimal("0.5"), 4)
            14
        """
        self._increment_count()

        # Step 1: Apply usage adjustment
        adjusted = self.apply_adjustment(base_lifetime_years, adjustment)

        # Step 2: Apply repair extension
        if repair_rate > _ZERO and extension_years > 0:
            extension_dec = _q(Decimal(str(extension_years)) * repair_rate)
            extension_int = int(extension_dec.quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            ))
            effective = adjusted + extension_int
        else:
            effective = adjusted

        # Minimum 1 year
        if effective < 1:
            effective = 1

        logger.debug(
            "Effective lifetime: base=%d, adj=%s->%d, repair=%s/%d->%d",
            base_lifetime_years,
            adjustment,
            adjusted,
            repair_rate,
            extension_years,
            effective,
        )

        return effective

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def validate_lifetime_assumptions(
        self,
        lifetime_years: int,
        category: str,
    ) -> Dict[str, Any]:
        """
        Validate that a lifetime assumption falls within reasonable bounds.

        Checks the lifetime against category-specific bounds derived from
        industry data and regulatory guidance.

        Args:
            lifetime_years: Proposed lifetime in years.
            category: Product category code.

        Returns:
            Dictionary with:
            - is_valid: Whether lifetime is within bounds
            - min_years: Minimum acceptable lifetime
            - max_years: Maximum acceptable lifetime
            - proposed_years: The proposed lifetime
            - message: Validation message

        Example:
            >>> engine = LifetimeModelingEngine()
            >>> result = engine.validate_lifetime_assumptions(15, "APPLIANCES")
            >>> result["is_valid"]
            True
            >>> result = engine.validate_lifetime_assumptions(100, "IT_EQUIPMENT")
            >>> result["is_valid"]
            False
        """
        self._increment_count()
        category_upper = category.upper()

        bounds = LIFETIME_BOUNDS.get(category_upper)
        if bounds is None:
            return {
                "is_valid": True,
                "min_years": 0,
                "max_years": 50,
                "proposed_years": lifetime_years,
                "message": (
                    f"Category '{category_upper}' has no specific bounds. "
                    "Using global range [0, 50]."
                ),
            }

        min_years, max_years = bounds
        is_valid = min_years <= lifetime_years <= max_years

        if is_valid:
            message = (
                f"Lifetime of {lifetime_years} years is within acceptable "
                f"bounds [{min_years}, {max_years}] for category '{category_upper}'."
            )
        else:
            message = (
                f"Lifetime of {lifetime_years} years is OUTSIDE acceptable "
                f"bounds [{min_years}, {max_years}] for category '{category_upper}'. "
                "Consider adjusting or providing justification."
            )

        return {
            "is_valid": is_valid,
            "min_years": min_years,
            "max_years": max_years,
            "proposed_years": lifetime_years,
            "message": message,
            "category": category_upper,
        }

    # ==========================================================================
    # SUMMARY & STATE
    # ==========================================================================

    def get_query_count(self) -> int:
        """
        Get the total number of queries/calculations performed.

        Returns:
            Integer count.
        """
        with self._count_lock:
            return self._query_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the engine state and configuration.

        Returns:
            Dictionary with engine metadata and statistics.
        """
        total_types = sum(len(v) for v in DEFAULT_LIFETIMES.values())
        total_profiles = sum(len(v) for v in USAGE_PROFILES.values())

        return {
            "engine": ENGINE_NAME,
            "engine_number": ENGINE_NUMBER,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "query_count": self.get_query_count(),
            "categories": len(DEFAULT_LIFETIMES),
            "product_types": total_types,
            "usage_profiles": total_profiles,
            "adjustment_factors": len(ADJUSTMENT_FACTORS),
            "weibull_shapes": len(WEIBULL_SHAPE_PARAMETERS),
            "degradation_rates": len(DEGRADATION_RATES),
        }

    def get_supported_categories(self) -> List[str]:
        """
        Get list of supported product categories.

        Returns:
            Sorted list of category codes.
        """
        return sorted(DEFAULT_LIFETIMES.keys())

    def get_product_types_for_category(
        self,
        category: str,
    ) -> List[str]:
        """
        Get list of product types for a specific category.

        Args:
            category: Product category code.

        Returns:
            Sorted list of product type strings. Empty if category not found.
        """
        category_upper = category.upper()
        types = DEFAULT_LIFETIMES.get(category_upper)
        if types is None:
            return []
        return sorted(types.keys())

    def get_all_lifetimes(self) -> Dict[str, Dict[str, int]]:
        """
        Get the complete default lifetime reference table.

        Returns:
            Nested dictionary of category -> product_type -> lifetime (years).
        """
        return DEFAULT_LIFETIMES.copy()

    def get_all_adjustment_factors(self) -> Dict[str, Decimal]:
        """
        Get all lifetime adjustment factors.

        Returns:
            Dictionary of adjustment_code -> multiplier.
        """
        return ADJUSTMENT_FACTORS.copy()

    def get_all_weibull_shapes(self) -> Dict[str, Decimal]:
        """
        Get all Weibull shape parameters.

        Returns:
            Dictionary of category -> shape_parameter.
        """
        return WEIBULL_SHAPE_PARAMETERS.copy()

    def get_all_degradation_rates(self) -> Dict[str, Decimal]:
        """
        Get all energy degradation rates.

        Returns:
            Dictionary of category -> annual_rate.
        """
        return DEGRADATION_RATES.copy()

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: For use in test fixtures only. Do not call in production.
        """
        with cls._lock:
            cls._instance = None


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================

_engine_instance: Optional[LifetimeModelingEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_lifetime_modeling_engine() -> LifetimeModelingEngine:
    """
    Get the singleton LifetimeModelingEngine instance.

    Thread-safe accessor.

    Returns:
        LifetimeModelingEngine singleton instance.
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = LifetimeModelingEngine()
        return _engine_instance


def reset_lifetime_modeling_engine() -> None:
    """
    Reset the module-level engine instance (for testing only).

    Warning: For use in test fixtures only. Do not call in production.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    LifetimeModelingEngine.reset()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "ENGINE_NAME",
    "ENGINE_NUMBER",
    # Enums
    "ProductCategory",
    "LifetimeAdjustment",
    "UsageUnit",
    # Reference Tables
    "DEFAULT_LIFETIMES",
    "ADJUSTMENT_FACTORS",
    "DEGRADATION_RATES",
    "WEIBULL_SHAPE_PARAMETERS",
    "USAGE_PROFILES",
    "LIFETIME_BOUNDS",
    # Engine
    "LifetimeModelingEngine",
    "get_lifetime_modeling_engine",
    "reset_lifetime_modeling_engine",
]
