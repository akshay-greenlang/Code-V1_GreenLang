# -*- coding: utf-8 -*-
"""
ElectricChillerCalculatorEngine - Engine 2: Cooling Purchase Agent (AGENT-MRV-012)

Core calculation engine for Scope 2 emissions from electric vapour-compression
chillers (centrifugal, screw, reciprocating, scroll). Implements full-load COP
and IPLV/NPLV part-load weighted calculations per AHRI 550/590, with auxiliary
energy accounting, condenser type COP adjustment, gas decomposition, and
multi-chiller plant aggregation.

Core Formulas (all Decimal arithmetic, zero-hallucination):

  Full-load COP:
    Electrical_Input (kWh)  = Cooling_Output (kWh_th) / COP
    Emissions (kgCO2e)      = Electrical_Input (kWh) x Grid_EF (kgCO2e/kWh)

  IPLV part-load weighted (AHRI 550/590):
    IPLV = 0.01 x COP_100% + 0.42 x COP_75% + 0.45 x COP_50% + 0.12 x COP_25%
    Electrical_Input (kWh)  = Cooling_Output (kWh_th) / IPLV
    Emissions (kgCO2e)      = Electrical_Input (kWh) x Grid_EF

  Auxiliary energy:
    Auxiliary_Energy (kWh) = Cooling_Output (kWh_th) x Auxiliary_Pct
    Total_Electricity      = Electrical_Input + Auxiliary_Energy
    Total_Emissions        = Total_Electricity x Grid_EF

  Gas decomposition (grid electricity CO2/CH4/N2O):
    CO2_share = 0.990, CH4_share = 0.005, N2O_share = 0.005 (typical grid)

  Multi-chiller plant:
    Weighted_COP = sum(cooling_i) / sum(cooling_i / cop_i)
    Total_Emissions = sum(emissions_i)

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic formula evaluation only

Sources:
    - GHG Protocol Scope 2 Guidance (2015)
    - AHRI Standard 550/590 (2023)
    - ASHRAE Standard 90.1-2022
    - ASHRAE Handbook - HVAC Systems and Equipment (2024)
    - IPCC AR5/AR6 GWP Tables
    - ISO 14064-1:2018

Example:
    >>> from greenlang.agents.mrv.cooling_purchase.electric_chiller_calculator import (
    ...     ElectricChillerCalculatorEngine,
    ... )
    >>> engine = ElectricChillerCalculatorEngine()
    >>> from greenlang.agents.mrv.cooling_purchase.models import (
    ...     ElectricChillerRequest, CoolingTechnology, GWPSource,
    ... )
    >>> from decimal import Decimal
    >>> request = ElectricChillerRequest(
    ...     cooling_output_kwh_th=Decimal("500000"),
    ...     technology=CoolingTechnology.WATER_COOLED_CENTRIFUGAL,
    ...     grid_ef_kgco2e_per_kwh=Decimal("0.45"),
    ...     use_iplv=True,
    ... )
    >>> result = engine.calculate_electric_chiller(request)
    >>> assert result.emissions_kgco2e > 0
    >>> assert len(result.provenance_hash) == 64

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.mrv.cooling_purchase.config import CoolingPurchaseConfig
from greenlang.agents.mrv.cooling_purchase.metrics import CoolingPurchaseMetrics
from greenlang.agents.mrv.cooling_purchase.models import (
    AHRI_PART_LOAD_WEIGHTS,
    COOLING_TECHNOLOGY_SPECS,
    EFFICIENCY_CONVERSIONS,
    GWP_VALUES,
    CalculationResult,
    CompressorType,
    CondenserType,
    CoolingTechnology,
    CoolingTechnologySpec,
    DataQualityTier,
    EfficiencyMetric,
    ElectricChillerRequest,
    EmissionGas,
    GasEmissionDetail,
    GWPSource,
)
from greenlang.agents.mrv.cooling_purchase.provenance import (
    CoolingPurchaseProvenance,
    get_provenance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places

_ZERO = Decimal("0")
_ONE = Decimal("1")

# ---------------------------------------------------------------------------
# Part-load COP estimation multipliers from full-load COP
# ---------------------------------------------------------------------------
# Typical part-load performance multipliers for electric chillers.
# At part load, centrifugal and screw chillers become more efficient
# due to lower lift ratio and reduced compressor work. These multipliers
# are empirically derived from manufacturer catalogue data and AHRI
# certified performance curves.
#
# Sources:
#   ASHRAE Handbook - HVAC Systems and Equipment (2024), Ch. 43.
#   AHRI Standard 550/590 (2023) certified performance data analysis.
#   US DOE FEMP chiller efficiency benchmarks.

_PART_LOAD_MULTIPLIER_75 = Decimal("1.15")
_PART_LOAD_MULTIPLIER_50 = Decimal("1.30")
_PART_LOAD_MULTIPLIER_25 = Decimal("1.10")

# ---------------------------------------------------------------------------
# Condenser type COP adjustment factors
# ---------------------------------------------------------------------------
# Water-cooled condensers achieve lower condensing temperatures than
# air-cooled, yielding 20-40% higher COP. When a technology's default
# condenser type does not match the actual installation, apply a
# correction factor to the COP.
#
# Sources:
#   ASHRAE Handbook - Fundamentals (2021), Chapter 2.
#   AHRI Standard 550/590 (2023), rating conditions.
#   Carrier/Trane/York/Johnson Controls catalogue data.

_CONDENSER_ADJUSTMENT: Dict[Tuple[str, str], Decimal] = {
    # (technology_condenser, actual_condenser) -> multiplier
    ("water_cooled", "air_cooled"): Decimal("0.72"),
    ("air_cooled", "water_cooled"): Decimal("1.35"),
    ("water_cooled", "water_cooled"): _ONE,
    ("air_cooled", "air_cooled"): _ONE,
}

# ---------------------------------------------------------------------------
# Grid emission gas shares (typical grid decomposition)
# ---------------------------------------------------------------------------
# When only total CO2e is known from the grid emission factor, decompose
# into constituent gases using typical grid generation mix shares.
# CO2 dominates (~99%), with trace CH4 from upstream natural gas
# leakage and N2O from coal/biomass combustion.
#
# Sources:
#   IEA World Energy Outlook 2024, electricity generation fuel mix.
#   EPA eGRID 2022, subregion gas-specific emission factors.
#   IPCC 2006 Guidelines, Vol. 2, Ch. 2, Table 2.2.

_DEFAULT_CO2_SHARE = Decimal("0.990")
_DEFAULT_CH4_SHARE = Decimal("0.005")
_DEFAULT_N2O_SHARE = Decimal("0.005")

# ---------------------------------------------------------------------------
# Cooling tower energy fraction (for water-cooled systems)
# ---------------------------------------------------------------------------
# Cooling tower fans and condenser water pumps consume approximately
# 10-15% of chiller electricity for water-cooled systems. Air-cooled
# condensers have no separate cooling tower energy but higher fan power
# is already captured in lower COP values.
#
# Sources:
#   ASHRAE Handbook - HVAC Applications (2023), Ch. 42.
#   ASHRAE Standard 90.1-2022 Appendix G.

_COOLING_TOWER_ENERGY_FRACTION = Decimal("0.12")

# ---------------------------------------------------------------------------
# Default pump efficiency
# ---------------------------------------------------------------------------
# Chilled water and condenser water pump efficiency. Wire-to-water
# efficiency including motor, VFD, and pump hydraulic efficiency.
#
# Sources:
#   ASHRAE 90.1-2022 Table 6.5.3.

_DEFAULT_PUMP_EFFICIENCY = Decimal("0.65")

# ---------------------------------------------------------------------------
# Pump specific power
# ---------------------------------------------------------------------------
# Watts per kW_th of cooling capacity for chilled water distribution
# pumps. Typical for variable-primary or primary-secondary systems.
#
# Sources:
#   ASHRAE Standard 90.1-2022.
#   Eurovent certification programme pump data.

_PUMP_SPECIFIC_POWER_W_PER_KW = Decimal("25")


# =============================================================================
# ElectricChillerCalculatorEngine
# =============================================================================


class ElectricChillerCalculatorEngine:
    """Scope 2 electric chiller emission calculation engine.

    Calculates GHG emissions from purchased cooling produced by electric
    vapour-compression chillers. Supports centrifugal, screw, reciprocating,
    and scroll compressor types with water-cooled and air-cooled condensers.

    Implements two primary calculation paths:

    1. **Full-load COP**: Uses a single COP value (measured or default) to
       calculate electrical input and emissions. Suitable when only
       nameplate or design COP is available.

    2. **IPLV/NPLV part-load weighted**: Uses four part-load COP values
       weighted per AHRI 550/590 standard (1% at 100%, 42% at 75%,
       45% at 50%, 12% at 25%) for more representative annual
       performance. Default and recommended calculation method.

    This engine follows GreenLang's zero-hallucination principle: all
    calculations use deterministic Python Decimal arithmetic with no
    LLM calls in the calculation path. Every intermediate step is
    recorded in a trace list and a SHA-256 provenance chain for
    complete audit trail.

    Thread Safety:
        Uses a reentrant lock (``threading.RLock``) around the singleton
        creation to prevent race conditions during first access in
        concurrent environments. Individual calculation methods are
        thread-safe because they operate on local state only and use
        the thread-safe CoolingPurchaseProvenance singleton for
        provenance tracking.

    Attributes:
        _config: CoolingPurchaseConfig singleton.
        _metrics: CoolingPurchaseMetrics singleton.
        _provenance: CoolingPurchaseProvenance singleton.

    Example:
        >>> engine = ElectricChillerCalculatorEngine()
        >>> assert engine is ElectricChillerCalculatorEngine()  # singleton
    """

    _instance: Optional[ElectricChillerCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(cls) -> ElectricChillerCalculatorEngine:
        """Return the singleton ElectricChillerCalculatorEngine instance.

        Uses double-checked locking with an ``RLock`` to ensure exactly
        one instance is created even under concurrent first-access.

        Returns:
            The singleton ElectricChillerCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine with config, metrics, and provenance.

        Guarded by the ``_initialized`` flag to prevent re-initialization
        on subsequent singleton access.
        """
        if self._initialized:
            return
        self._config = CoolingPurchaseConfig()
        self._metrics = CoolingPurchaseMetrics()
        self._provenance = get_provenance()
        self._precision = Decimal(
            "0." + "0" * (self._config.decimal_places - 1) + "1"
        ) if self._config.decimal_places > 0 else Decimal("1")
        self._initialized = True
        logger.info(
            "ElectricChillerCalculatorEngine initialized: "
            "decimal_places=%d, use_iplv=%s, min_cop=%s, max_cop=%s",
            self._config.decimal_places,
            self._config.use_iplv_default,
            self._config.min_cop,
            self._config.max_cop,
        )

    # ------------------------------------------------------------------
    # Singleton reset (for testing)
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton instance for testing purposes.

        After calling this method, the next instantiation will create a
        fresh instance with re-read configuration.
        """
        with cls._lock:
            cls._instance = None
        logger.debug("ElectricChillerCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize a Decimal to the configured precision.

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal rounded to configured decimal places
            using ROUND_HALF_UP (banker's rounding).
        """
        return value.quantize(self._precision, rounding=ROUND_HALF_UP)

    def _make_trace_step(self, label: str, detail: str) -> str:
        """Format a trace step string.

        Args:
            label: Short step label (e.g. 'COP_LOOKUP').
            detail: Descriptive detail of what happened.

        Returns:
            Formatted trace step string.
        """
        return f"[{label}] {detail}"

    def _compute_provenance_hash(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 hash of a data dictionary.

        Uses canonical JSON serialization (sorted keys, default=str)
        for deterministic hashing across Python versions and platforms.

        Args:
            data: Dictionary to hash. All values must be JSON-serializable
                or convertible via str().

        Returns:
            64-character hexadecimal SHA-256 digest string.
        """
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _validate_cop(self, cop: Decimal, context: str) -> None:
        """Validate that a COP value is within configured bounds.

        Args:
            cop: COP value to validate.
            context: Description for error messages.

        Raises:
            ValueError: If COP is outside [min_cop, max_cop].
        """
        if cop < self._config.min_cop:
            raise ValueError(
                f"{context}: COP {cop} is below minimum "
                f"{self._config.min_cop}"
            )
        if cop > self._config.max_cop:
            raise ValueError(
                f"{context}: COP {cop} exceeds maximum "
                f"{self._config.max_cop}"
            )

    def _validate_positive(
        self, value: Decimal, name: str,
    ) -> None:
        """Validate that a value is strictly positive.

        Args:
            value: Decimal value to check.
            name: Field name for error messages.

        Raises:
            ValueError: If value is zero or negative.
        """
        if value <= _ZERO:
            raise ValueError(f"{name} must be > 0, got {value}")

    def _validate_non_negative(
        self, value: Decimal, name: str,
    ) -> None:
        """Validate that a value is non-negative.

        Args:
            value: Decimal value to check.
            name: Field name for error messages.

        Raises:
            ValueError: If value is negative.
        """
        if value < _ZERO:
            raise ValueError(f"{name} must be >= 0, got {value}")

    def _validate_fraction(
        self, value: Decimal, name: str,
    ) -> None:
        """Validate that a value is in [0, 1].

        Args:
            value: Decimal value to check.
            name: Field name for error messages.

        Raises:
            ValueError: If value is outside [0, 1].
        """
        if value < _ZERO or value > _ONE:
            raise ValueError(
                f"{name} must be in [0, 1], got {value}"
            )

    def _get_technology_spec(
        self, technology: CoolingTechnology,
    ) -> CoolingTechnologySpec:
        """Look up the technology specification from the constant table.

        Args:
            technology: CoolingTechnology enum value.

        Returns:
            CoolingTechnologySpec for the requested technology.

        Raises:
            ValueError: If the technology is not in the specs table.
        """
        spec = COOLING_TECHNOLOGY_SPECS.get(technology.value)
        if spec is None:
            raise ValueError(
                f"No technology specification found for "
                f"'{technology.value}' in COOLING_TECHNOLOGY_SPECS"
            )
        return spec

    def _is_electric_chiller(
        self, technology: CoolingTechnology,
    ) -> bool:
        """Check if a technology is an electric chiller type.

        Electric chillers are the six vapour-compression types:
        water-cooled centrifugal, air-cooled centrifugal, water-cooled
        screw, air-cooled screw, water-cooled reciprocating, and
        air-cooled scroll.

        Args:
            technology: CoolingTechnology enum value.

        Returns:
            True if the technology is an electric chiller.
        """
        electric_types = frozenset({
            CoolingTechnology.WATER_COOLED_CENTRIFUGAL,
            CoolingTechnology.AIR_COOLED_CENTRIFUGAL,
            CoolingTechnology.WATER_COOLED_SCREW,
            CoolingTechnology.AIR_COOLED_SCREW,
            CoolingTechnology.WATER_COOLED_RECIPROCATING,
            CoolingTechnology.AIR_COOLED_SCROLL,
        })
        return technology in electric_types

    def _get_condenser_type_str(
        self, technology: CoolingTechnology,
    ) -> str:
        """Extract the condenser type string from a technology enum.

        Returns 'water_cooled' or 'air_cooled' based on the technology
        name prefix. Falls back to configured default if unable to
        determine from the name.

        Args:
            technology: CoolingTechnology enum value.

        Returns:
            Condenser type string ('water_cooled' or 'air_cooled').
        """
        tech_val = technology.value
        if tech_val.startswith("water_cooled"):
            return "water_cooled"
        if tech_val.startswith("air_cooled"):
            return "air_cooled"
        return self._config.default_condenser_type

    # ==================================================================
    # IPLV / NPLV Calculations
    # ==================================================================

    def calculate_iplv(
        self,
        cop_100: Decimal,
        cop_75: Decimal,
        cop_50: Decimal,
        cop_25: Decimal,
    ) -> Decimal:
        """Calculate Integrated Part-Load Value per AHRI 550/590.

        IPLV weights four part-load COP measurements to approximate
        annual average performance:
            IPLV = 0.01 * COP_100% + 0.42 * COP_75%
                 + 0.45 * COP_50% + 0.12 * COP_25%

        Args:
            cop_100: COP at 100% load.
            cop_75: COP at 75% load.
            cop_50: COP at 50% load.
            cop_25: COP at 25% load.

        Returns:
            IPLV as a Decimal quantized to configured precision.

        Raises:
            ValueError: If any COP value is not positive.
        """
        self._validate_positive(cop_100, "cop_100")
        self._validate_positive(cop_75, "cop_75")
        self._validate_positive(cop_50, "cop_50")
        self._validate_positive(cop_25, "cop_25")

        w_100 = self._config.ahri_100_weight
        w_75 = self._config.ahri_75_weight
        w_50 = self._config.ahri_50_weight
        w_25 = self._config.ahri_25_weight

        iplv = (
            w_100 * cop_100
            + w_75 * cop_75
            + w_50 * cop_50
            + w_25 * cop_25
        )

        result = self._quantize(iplv)
        logger.debug(
            "IPLV calculated: cop_100=%s, cop_75=%s, cop_50=%s, "
            "cop_25=%s, weights=[%s,%s,%s,%s], iplv=%s",
            cop_100, cop_75, cop_50, cop_25,
            w_100, w_75, w_50, w_25, result,
        )
        return result

    def calculate_nplv(
        self,
        cop_100: Decimal,
        cop_75: Decimal,
        cop_50: Decimal,
        cop_25: Decimal,
        weights: Optional[Dict[str, Decimal]] = None,
    ) -> Decimal:
        """Calculate Non-standard Part-Load Value with custom weights.

        NPLV allows custom weighting for facilities whose actual load
        profile differs from the AHRI standard distribution. If no
        custom weights are provided, falls back to AHRI 550/590 weights
        (equivalent to IPLV).

        The weights must sum to 1.0 and each be non-negative.

        Args:
            cop_100: COP at 100% load.
            cop_75: COP at 75% load.
            cop_50: COP at 50% load.
            cop_25: COP at 25% load.
            weights: Optional dictionary with keys '100%', '75%', '50%',
                '25%' mapping to Decimal weights that sum to 1.0.

        Returns:
            NPLV as a Decimal quantized to configured precision.

        Raises:
            ValueError: If any COP is not positive or weights do not
                sum to 1.0.
        """
        self._validate_positive(cop_100, "cop_100")
        self._validate_positive(cop_75, "cop_75")
        self._validate_positive(cop_50, "cop_50")
        self._validate_positive(cop_25, "cop_25")

        if weights is None:
            return self.calculate_iplv(cop_100, cop_75, cop_50, cop_25)

        w_100 = weights.get("100%", _ZERO)
        w_75 = weights.get("75%", _ZERO)
        w_50 = weights.get("50%", _ZERO)
        w_25 = weights.get("25%", _ZERO)

        self._validate_non_negative(w_100, "weight_100%")
        self._validate_non_negative(w_75, "weight_75%")
        self._validate_non_negative(w_50, "weight_50%")
        self._validate_non_negative(w_25, "weight_25%")

        total_weight = w_100 + w_75 + w_50 + w_25
        if abs(total_weight - _ONE) > Decimal("0.001"):
            raise ValueError(
                f"NPLV weights must sum to 1.0, got {total_weight} "
                f"(100%={w_100}, 75%={w_75}, 50%={w_50}, 25%={w_25})"
            )

        nplv = (
            w_100 * cop_100
            + w_75 * cop_75
            + w_50 * cop_50
            + w_25 * cop_25
        )

        result = self._quantize(nplv)
        logger.debug(
            "NPLV calculated: cop_100=%s, cop_75=%s, cop_50=%s, "
            "cop_25=%s, weights=[%s,%s,%s,%s], nplv=%s",
            cop_100, cop_75, cop_50, cop_25,
            w_100, w_75, w_50, w_25, result,
        )
        return result

    def estimate_part_load_cops(
        self,
        cop_full: Decimal,
    ) -> Dict[str, Decimal]:
        """Estimate part-load COPs from full-load COP.

        When only the full-load (100%) COP is known, estimate the
        75%, 50%, and 25% load COPs using typical part-load performance
        multipliers derived from AHRI certified data and ASHRAE
        Handbook equipment curves.

        Typical multipliers:
            COP_75% = COP_full x 1.15 (peak efficiency near 75% load)
            COP_50% = COP_full x 1.30 (VSD/slide-valve benefit)
            COP_25% = COP_full x 1.10 (reduced but positive benefit)

        Args:
            cop_full: Full-load COP value.

        Returns:
            Dictionary with keys 'cop_100', 'cop_75', 'cop_50', 'cop_25'
            mapped to Decimal values quantized to configured precision.

        Raises:
            ValueError: If cop_full is not positive.
        """
        self._validate_positive(cop_full, "cop_full")

        cops = {
            "cop_100": self._quantize(cop_full),
            "cop_75": self._quantize(
                cop_full * _PART_LOAD_MULTIPLIER_75
            ),
            "cop_50": self._quantize(
                cop_full * _PART_LOAD_MULTIPLIER_50
            ),
            "cop_25": self._quantize(
                cop_full * _PART_LOAD_MULTIPLIER_25
            ),
        }

        logger.debug(
            "Part-load COPs estimated from full-load COP %s: %s",
            cop_full, cops,
        )
        return cops

    # ==================================================================
    # Efficiency Conversions
    # ==================================================================

    def convert_to_cop(
        self,
        value: Decimal,
        metric: str,
    ) -> Decimal:
        """Convert an efficiency metric value to COP.

        Supports conversion from EER, kW/ton, and SEER to COP using
        deterministic Decimal conversion factors from ASHRAE Handbook
        and AHRI standards.

        Conversion formulas:
            COP = EER / 3.412
            COP = 3.517 / kW_per_ton
            COP = SEER / 3.412

        Args:
            value: Efficiency metric value to convert.
            metric: Source metric type. One of 'eer', 'kw_per_ton',
                'seer'. Case-insensitive.

        Returns:
            COP as a Decimal quantized to configured precision.

        Raises:
            ValueError: If value is not positive or metric is unknown.
        """
        self._validate_positive(value, f"{metric}_value")

        metric_lower = metric.strip().lower()

        if metric_lower == "eer":
            cop = value * EFFICIENCY_CONVERSIONS["eer_to_cop"]
        elif metric_lower == "kw_per_ton":
            cop = EFFICIENCY_CONVERSIONS["cop_to_kw_per_ton"] / value
        elif metric_lower == "seer":
            cop = value * EFFICIENCY_CONVERSIONS["seer_to_cop"]
        elif metric_lower == "cop":
            cop = value
        else:
            raise ValueError(
                f"Unknown efficiency metric '{metric}'. "
                f"Supported: eer, kw_per_ton, seer, cop"
            )

        result = self._quantize(cop)
        logger.debug(
            "Efficiency conversion: %s %s -> COP %s",
            value, metric_lower, result,
        )
        return result

    # ==================================================================
    # COP Resolution and Adjustment
    # ==================================================================

    def get_effective_cop(
        self,
        technology: CoolingTechnology,
        cop_override: Optional[Decimal] = None,
        use_iplv: Optional[bool] = None,
    ) -> Decimal:
        """Resolve the effective COP for an electric chiller.

        Resolution priority:
        1. cop_override (measured/site-specific) if provided
        2. IPLV from technology spec if use_iplv is True and IPLV exists
        3. Default COP from technology spec

        Args:
            technology: CoolingTechnology enum value.
            cop_override: Optional measured COP override.
            use_iplv: Whether to prefer IPLV over full-load COP.
                If None, uses the configured default.

        Returns:
            Effective COP as a quantized Decimal.

        Raises:
            ValueError: If resolved COP is outside configured bounds or
                technology not found.
        """
        spec = self._get_technology_spec(technology)

        if use_iplv is None:
            use_iplv = self._config.use_iplv_default

        if cop_override is not None:
            self._validate_cop(cop_override, "cop_override")
            logger.debug(
                "Using COP override %s for %s",
                cop_override, technology.value,
            )
            return self._quantize(cop_override)

        if use_iplv and spec.iplv is not None:
            self._validate_cop(spec.iplv, "technology_iplv")
            logger.debug(
                "Using IPLV %s for %s",
                spec.iplv, technology.value,
            )
            return self._quantize(spec.iplv)

        self._validate_cop(spec.cop_default, "technology_default_cop")
        logger.debug(
            "Using default COP %s for %s",
            spec.cop_default, technology.value,
        )
        return self._quantize(spec.cop_default)

    def adjust_cop_for_condenser(
        self,
        cop: Decimal,
        technology_condenser: str,
        actual_condenser: str,
    ) -> Decimal:
        """Adjust COP for condenser type mismatch.

        Water-cooled condensers achieve lower condensing temperatures
        (29-35 deg C) than air-cooled (40-55 deg C), resulting in
        significantly higher COP. When the actual condenser type
        differs from the technology's rated condenser type, apply a
        correction factor.

        Adjustment factors:
            water_cooled -> air_cooled:  0.72 (COP decreases ~28%)
            air_cooled -> water_cooled:  1.35 (COP increases ~35%)
            same -> same:               1.00 (no adjustment)

        Args:
            cop: Original COP value.
            technology_condenser: Condenser type the COP was rated for.
                One of 'water_cooled', 'air_cooled'.
            actual_condenser: Actual condenser type installed.
                One of 'water_cooled', 'air_cooled'.

        Returns:
            Adjusted COP as a quantized Decimal.

        Raises:
            ValueError: If cop is not positive.
        """
        self._validate_positive(cop, "cop")

        tech_cond = technology_condenser.strip().lower()
        actual_cond = actual_condenser.strip().lower()

        key = (tech_cond, actual_cond)
        factor = _CONDENSER_ADJUSTMENT.get(key, _ONE)

        adjusted = self._quantize(cop * factor)
        if factor != _ONE:
            logger.info(
                "COP adjusted for condenser mismatch: "
                "%s (rated %s) -> %s (actual %s), factor=%s",
                cop, tech_cond, adjusted, actual_cond, factor,
            )
        return adjusted

    # ==================================================================
    # Auxiliary Energy Calculations
    # ==================================================================

    def calculate_auxiliary_energy(
        self,
        cooling_kwh_th: Decimal,
        auxiliary_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate auxiliary electricity consumption.

        Auxiliary energy includes cooling tower fans, condenser water
        pumps, chilled water pumps, and controls that support the
        chiller but are not part of the chiller's rated COP.

        Formula:
            Auxiliary_kWh = Cooling_Output_kWh_th x Auxiliary_Pct

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            auxiliary_pct: Auxiliary energy as a fraction of cooling
                output (0-1). If None, uses configured default.

        Returns:
            Auxiliary energy in kWh as a quantized Decimal.

        Raises:
            ValueError: If cooling_kwh_th is not positive or
                auxiliary_pct is outside [0, 1].
        """
        self._validate_positive(cooling_kwh_th, "cooling_kwh_th")

        if auxiliary_pct is None:
            auxiliary_pct = self._config.default_auxiliary_pct

        self._validate_fraction(auxiliary_pct, "auxiliary_pct")

        result = self._quantize(cooling_kwh_th * auxiliary_pct)
        logger.debug(
            "Auxiliary energy: %s kWh_th x %s = %s kWh",
            cooling_kwh_th, auxiliary_pct, result,
        )
        return result

    def calculate_cooling_tower_energy(
        self,
        cooling_kwh_th: Decimal,
        condenser_type: Optional[str] = None,
    ) -> Decimal:
        """Calculate cooling tower electricity consumption.

        For water-cooled systems, cooling towers consume additional
        electricity for fans and condenser water pumps. This energy
        is typically 10-15% of chiller electrical input. For air-cooled
        systems, returns zero as condenser fan power is included in the
        chiller COP rating.

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            condenser_type: Condenser type ('water_cooled' or
                'air_cooled'). If None, uses configured default.

        Returns:
            Cooling tower energy in kWh as a quantized Decimal.
            Returns 0 for air-cooled systems.

        Raises:
            ValueError: If cooling_kwh_th is not positive.
        """
        self._validate_positive(cooling_kwh_th, "cooling_kwh_th")

        if condenser_type is None:
            condenser_type = self._config.default_condenser_type

        condenser_lower = condenser_type.strip().lower()

        if condenser_lower == "air_cooled":
            logger.debug(
                "No cooling tower energy for air-cooled condenser"
            )
            return _ZERO

        ct_energy = self._quantize(
            cooling_kwh_th * _COOLING_TOWER_ENERGY_FRACTION
        )
        logger.debug(
            "Cooling tower energy: %s kWh_th x %s = %s kWh",
            cooling_kwh_th, _COOLING_TOWER_ENERGY_FRACTION, ct_energy,
        )
        return ct_energy

    def calculate_pump_energy(
        self,
        cooling_kwh_th: Decimal,
        pump_efficiency: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate chilled water distribution pump energy.

        Estimates the electricity consumed by chilled water distribution
        pumps based on the cooling capacity and pump efficiency. Uses
        a specific power factor (W per kW_th) and the number of
        operating hours implied by the energy quantity.

        Formula:
            Pump_kWh = Cooling_kWh_th x (Pump_W_per_kW / 1000)
                       / Pump_Efficiency

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            pump_efficiency: Wire-to-water pump efficiency (0-1).
                If None, uses default 0.65.

        Returns:
            Pump energy in kWh as a quantized Decimal.

        Raises:
            ValueError: If cooling_kwh_th is not positive or
                pump_efficiency is outside (0, 1].
        """
        self._validate_positive(cooling_kwh_th, "cooling_kwh_th")

        if pump_efficiency is None:
            pump_efficiency = _DEFAULT_PUMP_EFFICIENCY

        if pump_efficiency <= _ZERO or pump_efficiency > _ONE:
            raise ValueError(
                f"pump_efficiency must be in (0, 1], got {pump_efficiency}"
            )

        specific_power_kw = _PUMP_SPECIFIC_POWER_W_PER_KW / Decimal("1000")
        pump_kwh = self._quantize(
            cooling_kwh_th * specific_power_kw / pump_efficiency
        )
        logger.debug(
            "Pump energy: %s kWh_th x %s kW/kW / %s = %s kWh",
            cooling_kwh_th, specific_power_kw, pump_efficiency, pump_kwh,
        )
        return pump_kwh

    # ==================================================================
    # Gas Decomposition
    # ==================================================================

    def get_grid_gas_shares(self) -> Dict[str, Decimal]:
        """Return the default grid electricity gas emission shares.

        Returns the typical fraction of total CO2e attributed to each
        greenhouse gas species for grid-consumed electricity. Used when
        only the aggregate CO2e emission factor is available and a
        per-gas breakdown is required for reporting.

        Returns:
            Dictionary with keys 'CO2', 'CH4', 'N2O' mapped to
            Decimal share fractions summing to 1.0.
        """
        return {
            "CO2": _DEFAULT_CO2_SHARE,
            "CH4": _DEFAULT_CH4_SHARE,
            "N2O": _DEFAULT_N2O_SHARE,
        }

    def decompose_grid_emissions(
        self,
        total_co2e: Decimal,
        gwp_source: Optional[str] = None,
    ) -> List[GasEmissionDetail]:
        """Decompose total CO2e into constituent greenhouse gases.

        Splits the aggregate CO2-equivalent emission into individual
        CO2, CH4, and N2O quantities using typical grid gas shares
        and GWP values from the specified IPCC Assessment Report.

        Process:
        1. Allocate total_co2e by gas shares (CO2=99%, CH4=0.5%, N2O=0.5%)
        2. Convert each share back to gas mass using GWP:
           gas_mass_kg = co2e_share_kg / gwp_factor

        Args:
            total_co2e: Total CO2-equivalent emissions in kg.
            gwp_source: IPCC Assessment Report edition (e.g. 'AR5',
                'AR6'). If None, defaults to 'AR5'.

        Returns:
            List of 3 GasEmissionDetail objects (CO2, CH4, N2O), plus
            a 4th CO2e summary entry.

        Raises:
            ValueError: If total_co2e is negative or gwp_source is
                unknown.
        """
        self._validate_non_negative(total_co2e, "total_co2e")

        if gwp_source is None:
            gwp_source = "AR5"

        gwp_key = gwp_source.upper()
        if gwp_key not in GWP_VALUES:
            raise ValueError(
                f"Unknown GWP source '{gwp_source}'. "
                f"Supported: {sorted(GWP_VALUES.keys())}"
            )

        gwp_table = GWP_VALUES[gwp_key]
        shares = self.get_grid_gas_shares()
        details: List[GasEmissionDetail] = []

        # CO2 component
        co2_co2e = self._quantize(total_co2e * shares["CO2"])
        co2_gwp = gwp_table["CO2"]
        co2_mass = self._quantize(co2_co2e / co2_gwp) if co2_gwp > _ZERO else _ZERO
        details.append(GasEmissionDetail(
            gas=EmissionGas.CO2,
            quantity_kg=co2_mass,
            gwp_factor=co2_gwp,
            co2e_kg=co2_co2e,
        ))

        # CH4 component
        ch4_co2e = self._quantize(total_co2e * shares["CH4"])
        ch4_gwp = gwp_table["CH4"]
        ch4_mass = self._quantize(ch4_co2e / ch4_gwp) if ch4_gwp > _ZERO else _ZERO
        details.append(GasEmissionDetail(
            gas=EmissionGas.CH4,
            quantity_kg=ch4_mass,
            gwp_factor=ch4_gwp,
            co2e_kg=ch4_co2e,
        ))

        # N2O component
        n2o_co2e = self._quantize(total_co2e * shares["N2O"])
        n2o_gwp = gwp_table["N2O"]
        n2o_mass = self._quantize(n2o_co2e / n2o_gwp) if n2o_gwp > _ZERO else _ZERO
        details.append(GasEmissionDetail(
            gas=EmissionGas.N2O,
            quantity_kg=n2o_mass,
            gwp_factor=n2o_gwp,
            co2e_kg=n2o_co2e,
        ))

        # CO2e summary
        details.append(GasEmissionDetail(
            gas=EmissionGas.CO2E,
            quantity_kg=total_co2e,
            gwp_factor=_ONE,
            co2e_kg=total_co2e,
        ))

        logger.debug(
            "Gas decomposition (%s): CO2=%s kg, CH4=%s kg, N2O=%s kg, "
            "total_co2e=%s kg",
            gwp_key, co2_mass, ch4_mass, n2o_mass, total_co2e,
        )
        return details

    # ==================================================================
    # Core Calculation: Full-Load COP
    # ==================================================================

    def calculate_full_load(
        self,
        cooling_kwh_th: Decimal,
        cop: Decimal,
        grid_ef: Decimal,
        auxiliary_pct: Optional[Decimal] = None,
        technology: Optional[CoolingTechnology] = None,
    ) -> CalculationResult:
        """Calculate emissions using full-load COP method.

        Implements the basic Scope 2 emission calculation for an
        electric chiller operating at full-load COP:

            Electrical_Input = Cooling_Output / COP
            Auxiliary_Energy = Cooling_Output x Auxiliary_Pct
            Total_Electricity = Electrical_Input + Auxiliary_Energy
            Total_Emissions = Total_Electricity x Grid_EF

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop: Full-load coefficient of performance.
            grid_ef: Grid electricity emission factor (kgCO2e/kWh).
            auxiliary_pct: Auxiliary energy fraction (0-1). If None,
                uses configured default.
            technology: Optional technology for metadata and condenser
                type resolution.

        Returns:
            CalculationResult with emissions, gas breakdown, and
            provenance hash.

        Raises:
            ValueError: If any input fails validation.
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []
        tech_str = technology.value if technology else "unknown"

        try:
            # -- Input validation --
            self._validate_positive(cooling_kwh_th, "cooling_kwh_th")
            self._validate_positive(cop, "cop")
            self._validate_cop(cop, "full_load_cop")
            self._validate_non_negative(grid_ef, "grid_ef")
            trace.append(self._make_trace_step(
                "INPUT_VALIDATION",
                f"cooling_kwh_th={cooling_kwh_th}, cop={cop}, "
                f"grid_ef={grid_ef}, technology={tech_str}",
            ))

            # -- Provenance chain --
            chain_id = self._provenance.create_chain(calc_id)
            self._provenance.add_stage(chain_id, "INPUT_VALIDATION", {
                "calculation_id": calc_id,
                "method": "full_load",
                "cooling_kwh_th": str(cooling_kwh_th),
                "cop": str(cop),
                "grid_ef": str(grid_ef),
                "technology": tech_str,
            })

            # -- Electrical input --
            electrical_input = self._quantize(cooling_kwh_th / cop)
            trace.append(self._make_trace_step(
                "ENERGY_INPUT",
                f"{cooling_kwh_th} kWh_th / {cop} COP = "
                f"{electrical_input} kWh",
            ))
            self._provenance.add_stage(chain_id, "ENERGY_INPUT_CALCULATION", {
                "cooling_kwh_th": str(cooling_kwh_th),
                "cop": str(cop),
                "electrical_input_kwh": str(electrical_input),
            })

            # -- Auxiliary energy --
            if auxiliary_pct is None:
                auxiliary_pct = self._config.default_auxiliary_pct
            self._validate_fraction(auxiliary_pct, "auxiliary_pct")
            aux_energy = self.calculate_auxiliary_energy(
                cooling_kwh_th, auxiliary_pct,
            )
            trace.append(self._make_trace_step(
                "AUXILIARY_ENERGY",
                f"{cooling_kwh_th} kWh_th x {auxiliary_pct} = "
                f"{aux_energy} kWh auxiliary",
            ))
            self._provenance.add_stage(chain_id, "AUXILIARY_ENERGY", {
                "cooling_kwh_th": str(cooling_kwh_th),
                "auxiliary_pct": str(auxiliary_pct),
                "auxiliary_kwh": str(aux_energy),
            })

            # -- Total electricity --
            total_electricity = self._quantize(
                electrical_input + aux_energy
            )
            trace.append(self._make_trace_step(
                "TOTAL_ELECTRICITY",
                f"{electrical_input} + {aux_energy} = "
                f"{total_electricity} kWh total",
            ))

            # -- Grid emission factor application --
            total_emissions = self._quantize(total_electricity * grid_ef)
            trace.append(self._make_trace_step(
                "GRID_EMISSIONS",
                f"{total_electricity} kWh x {grid_ef} kgCO2e/kWh = "
                f"{total_emissions} kgCO2e",
            ))
            self._provenance.add_stage(
                chain_id, "GRID_FACTOR_APPLICATION", {
                    "total_electricity_kwh": str(total_electricity),
                    "grid_ef_kgco2e_per_kwh": str(grid_ef),
                    "total_emissions_kgco2e": str(total_emissions),
                },
            )

            # -- Gas decomposition --
            gas_breakdown = self.decompose_grid_emissions(
                total_emissions, "AR5",
            )
            trace.append(self._make_trace_step(
                "GAS_DECOMPOSITION",
                f"Decomposed {total_emissions} kgCO2e into "
                f"{len(gas_breakdown)} gas species",
            ))
            self._provenance.add_stage(chain_id, "GAS_DECOMPOSITION", {
                "total_co2e_kg": str(total_emissions),
                "gas_count": str(len(gas_breakdown)),
            })

            # -- Result finalization --
            provenance_hash = self._provenance.seal_chain(chain_id)
            trace.append(self._make_trace_step(
                "RESULT_FINALIZED",
                f"provenance_hash={provenance_hash[:16]}...",
            ))

            duration_s = time.monotonic() - start_time
            condenser_str = self._get_condenser_type_str(
                technology
            ) if technology else "unknown"

            # -- Metrics --
            self._metrics.record_calculation(
                technology=tech_str,
                calculation_type="electric",
                tier="tier_1",
                tenant_id="default",
                status="success",
                duration_s=duration_s,
                emissions_kgco2e=float(total_emissions),
                cooling_kwh_th=float(cooling_kwh_th),
                cop_used=float(cop),
                condenser_type=condenser_str,
            )

            result = CalculationResult(
                calculation_id=calc_id,
                calculation_type="electric_chiller_full_load",
                cooling_output_kwh_th=cooling_kwh_th,
                energy_input_kwh=total_electricity,
                cop_used=cop,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=DataQualityTier.TIER_1,
                provenance_hash=provenance_hash,
                trace_steps=trace,
                metadata={
                    "method": "full_load",
                    "technology": tech_str,
                    "grid_ef": str(grid_ef),
                    "auxiliary_pct": str(auxiliary_pct),
                    "electrical_input_kwh": str(electrical_input),
                    "auxiliary_kwh": str(aux_energy),
                },
            )

            logger.info(
                "Full-load calculation complete: calc_id=%s, "
                "emissions=%s kgCO2e, duration=%.4fs",
                calc_id, total_emissions, duration_s,
            )
            return result

        except Exception as exc:
            duration_s = time.monotonic() - start_time
            self._metrics.record_error("calculation", "calculation")
            logger.error(
                "Full-load calculation failed: calc_id=%s, error=%s, "
                "duration=%.4fs",
                calc_id, str(exc), duration_s,
                exc_info=True,
            )
            raise

    # ==================================================================
    # Core Calculation: IPLV Weighted
    # ==================================================================

    def calculate_iplv_weighted(
        self,
        cooling_kwh_th: Decimal,
        iplv: Decimal,
        grid_ef: Decimal,
        auxiliary_pct: Optional[Decimal] = None,
        technology: Optional[CoolingTechnology] = None,
    ) -> CalculationResult:
        """Calculate emissions using IPLV-weighted part-load method.

        Uses the Integrated Part-Load Value (IPLV) per AHRI 550/590
        instead of full-load COP for more representative annual
        performance estimation.

            Electrical_Input = Cooling_Output / IPLV
            Auxiliary_Energy = Cooling_Output x Auxiliary_Pct
            Total_Electricity = Electrical_Input + Auxiliary_Energy
            Total_Emissions = Total_Electricity x Grid_EF

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            iplv: Integrated Part-Load Value.
            grid_ef: Grid electricity emission factor (kgCO2e/kWh).
            auxiliary_pct: Auxiliary energy fraction (0-1). If None,
                uses configured default.
            technology: Optional technology for metadata.

        Returns:
            CalculationResult with emissions, gas breakdown, and
            provenance hash.

        Raises:
            ValueError: If any input fails validation.
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []
        tech_str = technology.value if technology else "unknown"

        try:
            # -- Input validation --
            self._validate_positive(cooling_kwh_th, "cooling_kwh_th")
            self._validate_positive(iplv, "iplv")
            self._validate_cop(iplv, "iplv")
            self._validate_non_negative(grid_ef, "grid_ef")
            trace.append(self._make_trace_step(
                "INPUT_VALIDATION",
                f"cooling_kwh_th={cooling_kwh_th}, iplv={iplv}, "
                f"grid_ef={grid_ef}, technology={tech_str}",
            ))

            # -- Provenance chain --
            chain_id = self._provenance.create_chain(calc_id)
            self._provenance.add_stage(chain_id, "INPUT_VALIDATION", {
                "calculation_id": calc_id,
                "method": "iplv_weighted",
                "cooling_kwh_th": str(cooling_kwh_th),
                "iplv": str(iplv),
                "grid_ef": str(grid_ef),
                "technology": tech_str,
            })

            # -- IPLV provenance --
            self._provenance.add_stage(
                chain_id, "PART_LOAD_CALCULATION", {
                    "iplv": str(iplv),
                    "method": "pre_computed_iplv",
                },
            )
            trace.append(self._make_trace_step(
                "IPLV_APPLIED",
                f"Using pre-computed IPLV={iplv}",
            ))

            # -- Electrical input --
            electrical_input = self._quantize(cooling_kwh_th / iplv)
            trace.append(self._make_trace_step(
                "ENERGY_INPUT",
                f"{cooling_kwh_th} kWh_th / {iplv} IPLV = "
                f"{electrical_input} kWh",
            ))
            self._provenance.add_stage(
                chain_id, "ENERGY_INPUT_CALCULATION", {
                    "cooling_kwh_th": str(cooling_kwh_th),
                    "iplv": str(iplv),
                    "electrical_input_kwh": str(electrical_input),
                },
            )

            # -- Auxiliary energy --
            if auxiliary_pct is None:
                auxiliary_pct = self._config.default_auxiliary_pct
            self._validate_fraction(auxiliary_pct, "auxiliary_pct")
            aux_energy = self.calculate_auxiliary_energy(
                cooling_kwh_th, auxiliary_pct,
            )
            trace.append(self._make_trace_step(
                "AUXILIARY_ENERGY",
                f"{cooling_kwh_th} kWh_th x {auxiliary_pct} = "
                f"{aux_energy} kWh auxiliary",
            ))
            self._provenance.add_stage(chain_id, "AUXILIARY_ENERGY", {
                "cooling_kwh_th": str(cooling_kwh_th),
                "auxiliary_pct": str(auxiliary_pct),
                "auxiliary_kwh": str(aux_energy),
            })

            # -- Total electricity --
            total_electricity = self._quantize(
                electrical_input + aux_energy
            )
            trace.append(self._make_trace_step(
                "TOTAL_ELECTRICITY",
                f"{electrical_input} + {aux_energy} = "
                f"{total_electricity} kWh total",
            ))

            # -- Grid emission factor application --
            total_emissions = self._quantize(total_electricity * grid_ef)
            trace.append(self._make_trace_step(
                "GRID_EMISSIONS",
                f"{total_electricity} kWh x {grid_ef} kgCO2e/kWh = "
                f"{total_emissions} kgCO2e",
            ))
            self._provenance.add_stage(
                chain_id, "GRID_FACTOR_APPLICATION", {
                    "total_electricity_kwh": str(total_electricity),
                    "grid_ef_kgco2e_per_kwh": str(grid_ef),
                    "total_emissions_kgco2e": str(total_emissions),
                },
            )

            # -- Gas decomposition --
            gas_breakdown = self.decompose_grid_emissions(
                total_emissions, "AR5",
            )
            trace.append(self._make_trace_step(
                "GAS_DECOMPOSITION",
                f"Decomposed {total_emissions} kgCO2e into "
                f"{len(gas_breakdown)} gas species",
            ))
            self._provenance.add_stage(chain_id, "GAS_DECOMPOSITION", {
                "total_co2e_kg": str(total_emissions),
                "gas_count": str(len(gas_breakdown)),
            })

            # -- Seal provenance --
            provenance_hash = self._provenance.seal_chain(chain_id)
            trace.append(self._make_trace_step(
                "RESULT_FINALIZED",
                f"provenance_hash={provenance_hash[:16]}...",
            ))

            duration_s = time.monotonic() - start_time
            condenser_str = self._get_condenser_type_str(
                technology
            ) if technology else "unknown"

            # -- Metrics --
            self._metrics.record_calculation(
                technology=tech_str,
                calculation_type="electric",
                tier="tier_1",
                tenant_id="default",
                status="success",
                duration_s=duration_s,
                emissions_kgco2e=float(total_emissions),
                cooling_kwh_th=float(cooling_kwh_th),
                cop_used=float(iplv),
                condenser_type=condenser_str,
            )

            result = CalculationResult(
                calculation_id=calc_id,
                calculation_type="electric_chiller_iplv",
                cooling_output_kwh_th=cooling_kwh_th,
                energy_input_kwh=total_electricity,
                cop_used=iplv,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=DataQualityTier.TIER_1,
                provenance_hash=provenance_hash,
                trace_steps=trace,
                metadata={
                    "method": "iplv_weighted",
                    "technology": tech_str,
                    "grid_ef": str(grid_ef),
                    "auxiliary_pct": str(auxiliary_pct),
                    "electrical_input_kwh": str(electrical_input),
                    "auxiliary_kwh": str(aux_energy),
                },
            )

            logger.info(
                "IPLV calculation complete: calc_id=%s, "
                "emissions=%s kgCO2e, duration=%.4fs",
                calc_id, total_emissions, duration_s,
            )
            return result

        except Exception as exc:
            duration_s = time.monotonic() - start_time
            self._metrics.record_error("calculation", "calculation")
            logger.error(
                "IPLV calculation failed: calc_id=%s, error=%s, "
                "duration=%.4fs",
                calc_id, str(exc), duration_s,
                exc_info=True,
            )
            raise

    # ==================================================================
    # Core Calculation: Custom Part-Load
    # ==================================================================

    def calculate_custom_part_load(
        self,
        cooling_kwh_th: Decimal,
        cop_100: Decimal,
        cop_75: Decimal,
        cop_50: Decimal,
        cop_25: Decimal,
        grid_ef: Decimal,
        auxiliary_pct: Optional[Decimal] = None,
        weights: Optional[Dict[str, Decimal]] = None,
        technology: Optional[CoolingTechnology] = None,
    ) -> CalculationResult:
        """Calculate emissions with custom part-load COP values.

        Computes IPLV (or NPLV with custom weights) from four explicit
        part-load COP measurements, then calculates emissions using
        the weighted COP. This is the most accurate method when
        part-load test data is available.

        Process:
        1. Calculate IPLV/NPLV from the four COP values
        2. Compute electrical input using IPLV/NPLV
        3. Add auxiliary energy
        4. Apply grid emission factor

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop_100: COP at 100% load.
            cop_75: COP at 75% load.
            cop_50: COP at 50% load.
            cop_25: COP at 25% load.
            grid_ef: Grid electricity emission factor (kgCO2e/kWh).
            auxiliary_pct: Auxiliary energy fraction (0-1). If None,
                uses configured default.
            weights: Optional custom NPLV weights. If None, uses
                standard AHRI 550/590 IPLV weights.
            technology: Optional technology for metadata.

        Returns:
            CalculationResult with emissions, gas breakdown, and
            provenance hash.

        Raises:
            ValueError: If any input fails validation.
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []
        tech_str = technology.value if technology else "unknown"
        method_name = "nplv" if weights else "iplv"

        try:
            # -- Input validation --
            self._validate_positive(cooling_kwh_th, "cooling_kwh_th")
            self._validate_positive(cop_100, "cop_100")
            self._validate_positive(cop_75, "cop_75")
            self._validate_positive(cop_50, "cop_50")
            self._validate_positive(cop_25, "cop_25")
            self._validate_non_negative(grid_ef, "grid_ef")
            trace.append(self._make_trace_step(
                "INPUT_VALIDATION",
                f"cooling_kwh_th={cooling_kwh_th}, "
                f"cop_100={cop_100}, cop_75={cop_75}, "
                f"cop_50={cop_50}, cop_25={cop_25}, "
                f"grid_ef={grid_ef}",
            ))

            # -- Provenance chain --
            chain_id = self._provenance.create_chain(calc_id)
            self._provenance.add_stage(chain_id, "INPUT_VALIDATION", {
                "calculation_id": calc_id,
                "method": f"custom_part_load_{method_name}",
                "cooling_kwh_th": str(cooling_kwh_th),
                "cop_100": str(cop_100),
                "cop_75": str(cop_75),
                "cop_50": str(cop_50),
                "cop_25": str(cop_25),
                "grid_ef": str(grid_ef),
            })

            # -- Calculate IPLV/NPLV --
            if weights:
                weighted_cop = self.calculate_nplv(
                    cop_100, cop_75, cop_50, cop_25, weights,
                )
            else:
                weighted_cop = self.calculate_iplv(
                    cop_100, cop_75, cop_50, cop_25,
                )
            trace.append(self._make_trace_step(
                "PART_LOAD_CALCULATION",
                f"{method_name.upper()}={weighted_cop} from "
                f"COP[100%={cop_100}, 75%={cop_75}, "
                f"50%={cop_50}, 25%={cop_25}]",
            ))
            self._provenance.add_stage(
                chain_id, "PART_LOAD_CALCULATION", {
                    "method": method_name,
                    "cop_100": str(cop_100),
                    "cop_75": str(cop_75),
                    "cop_50": str(cop_50),
                    "cop_25": str(cop_25),
                    "weighted_cop": str(weighted_cop),
                },
            )

            # -- Electrical input --
            electrical_input = self._quantize(
                cooling_kwh_th / weighted_cop
            )
            trace.append(self._make_trace_step(
                "ENERGY_INPUT",
                f"{cooling_kwh_th} kWh_th / {weighted_cop} = "
                f"{electrical_input} kWh",
            ))
            self._provenance.add_stage(
                chain_id, "ENERGY_INPUT_CALCULATION", {
                    "cooling_kwh_th": str(cooling_kwh_th),
                    "weighted_cop": str(weighted_cop),
                    "electrical_input_kwh": str(electrical_input),
                },
            )

            # -- Auxiliary energy --
            if auxiliary_pct is None:
                auxiliary_pct = self._config.default_auxiliary_pct
            self._validate_fraction(auxiliary_pct, "auxiliary_pct")
            aux_energy = self.calculate_auxiliary_energy(
                cooling_kwh_th, auxiliary_pct,
            )
            trace.append(self._make_trace_step(
                "AUXILIARY_ENERGY",
                f"{cooling_kwh_th} kWh_th x {auxiliary_pct} = "
                f"{aux_energy} kWh auxiliary",
            ))
            self._provenance.add_stage(chain_id, "AUXILIARY_ENERGY", {
                "auxiliary_pct": str(auxiliary_pct),
                "auxiliary_kwh": str(aux_energy),
            })

            # -- Total electricity --
            total_electricity = self._quantize(
                electrical_input + aux_energy
            )
            trace.append(self._make_trace_step(
                "TOTAL_ELECTRICITY",
                f"{electrical_input} + {aux_energy} = "
                f"{total_electricity} kWh total",
            ))

            # -- Grid emission factor application --
            total_emissions = self._quantize(total_electricity * grid_ef)
            trace.append(self._make_trace_step(
                "GRID_EMISSIONS",
                f"{total_electricity} kWh x {grid_ef} kgCO2e/kWh = "
                f"{total_emissions} kgCO2e",
            ))
            self._provenance.add_stage(
                chain_id, "GRID_FACTOR_APPLICATION", {
                    "total_electricity_kwh": str(total_electricity),
                    "grid_ef": str(grid_ef),
                    "total_emissions_kgco2e": str(total_emissions),
                },
            )

            # -- Gas decomposition --
            gas_breakdown = self.decompose_grid_emissions(
                total_emissions, "AR5",
            )
            trace.append(self._make_trace_step(
                "GAS_DECOMPOSITION",
                f"Decomposed {total_emissions} kgCO2e into "
                f"{len(gas_breakdown)} gas species",
            ))
            self._provenance.add_stage(chain_id, "GAS_DECOMPOSITION", {
                "total_co2e_kg": str(total_emissions),
            })

            # -- Seal provenance --
            provenance_hash = self._provenance.seal_chain(chain_id)
            trace.append(self._make_trace_step(
                "RESULT_FINALIZED",
                f"provenance_hash={provenance_hash[:16]}...",
            ))

            duration_s = time.monotonic() - start_time
            condenser_str = self._get_condenser_type_str(
                technology
            ) if technology else "unknown"

            self._metrics.record_calculation(
                technology=tech_str,
                calculation_type="electric",
                tier="tier_1",
                tenant_id="default",
                status="success",
                duration_s=duration_s,
                emissions_kgco2e=float(total_emissions),
                cooling_kwh_th=float(cooling_kwh_th),
                cop_used=float(weighted_cop),
                condenser_type=condenser_str,
            )

            result = CalculationResult(
                calculation_id=calc_id,
                calculation_type=f"electric_chiller_{method_name}",
                cooling_output_kwh_th=cooling_kwh_th,
                energy_input_kwh=total_electricity,
                cop_used=weighted_cop,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=DataQualityTier.TIER_1,
                provenance_hash=provenance_hash,
                trace_steps=trace,
                metadata={
                    "method": f"custom_part_load_{method_name}",
                    "technology": tech_str,
                    "grid_ef": str(grid_ef),
                    "auxiliary_pct": str(auxiliary_pct),
                    "cop_100": str(cop_100),
                    "cop_75": str(cop_75),
                    "cop_50": str(cop_50),
                    "cop_25": str(cop_25),
                    "weighted_cop": str(weighted_cop),
                    "electrical_input_kwh": str(electrical_input),
                    "auxiliary_kwh": str(aux_energy),
                },
            )

            logger.info(
                "Custom part-load calculation complete: calc_id=%s, "
                "%s=%s, emissions=%s kgCO2e, duration=%.4fs",
                calc_id, method_name.upper(), weighted_cop,
                total_emissions, duration_s,
            )
            return result

        except Exception as exc:
            duration_s = time.monotonic() - start_time
            self._metrics.record_error("calculation", "calculation")
            logger.error(
                "Custom part-load calculation failed: calc_id=%s, "
                "error=%s, duration=%.4fs",
                calc_id, str(exc), duration_s,
                exc_info=True,
            )
            raise

    # ==================================================================
    # Primary Entry Point: calculate_electric_chiller
    # ==================================================================

    def calculate_electric_chiller(
        self,
        request: ElectricChillerRequest,
    ) -> CalculationResult:
        """Calculate Scope 2 emissions from an electric chiller.

        This is the primary public entry point for electric chiller
        emission calculations. Routes to the appropriate calculation
        method based on the request parameters:

        1. If all four part-load COPs are provided (cop_100, cop_75,
           cop_50, cop_25), uses custom part-load calculation
        2. If use_iplv is True and IPLV is available for the technology,
           uses IPLV-weighted calculation
        3. Otherwise uses full-load COP calculation

        The method handles COP resolution, condenser type adjustment,
        part-load estimation, and delegates to the appropriate core
        calculation method.

        Args:
            request: ElectricChillerRequest containing all parameters
                for the calculation.

        Returns:
            CalculationResult with complete emission details,
            gas breakdown, and provenance hash.

        Raises:
            ValueError: If the request contains invalid parameters.

        Example:
            >>> engine = ElectricChillerCalculatorEngine()
            >>> from greenlang.agents.mrv.cooling_purchase.models import (
            ...     ElectricChillerRequest, CoolingTechnology,
            ... )
            >>> from decimal import Decimal
            >>> req = ElectricChillerRequest(
            ...     cooling_output_kwh_th=Decimal("500000"),
            ...     technology=CoolingTechnology.WATER_COOLED_CENTRIFUGAL,
            ...     grid_ef_kgco2e_per_kwh=Decimal("0.45"),
            ... )
            >>> result = engine.calculate_electric_chiller(req)
            >>> assert result.emissions_kgco2e > 0
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []

        try:
            # -- Validate electric chiller technology --
            if not self._is_electric_chiller(request.technology):
                raise ValueError(
                    f"Technology '{request.technology.value}' is not an "
                    f"electric chiller type. Use one of: "
                    f"water_cooled_centrifugal, air_cooled_centrifugal, "
                    f"water_cooled_screw, air_cooled_screw, "
                    f"water_cooled_reciprocating, air_cooled_scroll"
                )

            trace.append(self._make_trace_step(
                "REQUEST_RECEIVED",
                f"technology={request.technology.value}, "
                f"cooling={request.cooling_output_kwh_th} kWh_th, "
                f"use_iplv={request.use_iplv}",
            ))

            # -- Route: custom part-load if all four COPs provided --
            if (
                request.cop_100 is not None
                and request.cop_75 is not None
                and request.cop_50 is not None
                and request.cop_25 is not None
            ):
                logger.info(
                    "Routing to custom part-load: all four COPs provided"
                )
                return self.calculate_custom_part_load(
                    cooling_kwh_th=request.cooling_output_kwh_th,
                    cop_100=request.cop_100,
                    cop_75=request.cop_75,
                    cop_50=request.cop_50,
                    cop_25=request.cop_25,
                    grid_ef=request.grid_ef_kgco2e_per_kwh,
                    auxiliary_pct=request.auxiliary_pct,
                    technology=request.technology,
                )

            # -- Resolve effective COP --
            effective_cop = self.get_effective_cop(
                technology=request.technology,
                cop_override=request.cop_override,
                use_iplv=request.use_iplv,
            )
            trace.append(self._make_trace_step(
                "COP_RESOLVED",
                f"effective_cop={effective_cop} "
                f"(override={request.cop_override}, "
                f"use_iplv={request.use_iplv})",
            ))

            # -- Route: IPLV or full-load --
            spec = self._get_technology_spec(request.technology)
            use_iplv = request.use_iplv

            if use_iplv and spec.iplv is not None and request.cop_override is None:
                # Use IPLV path
                logger.info(
                    "Routing to IPLV-weighted: use_iplv=True, "
                    "iplv=%s for %s",
                    effective_cop, request.technology.value,
                )
                return self.calculate_iplv_weighted(
                    cooling_kwh_th=request.cooling_output_kwh_th,
                    iplv=effective_cop,
                    grid_ef=request.grid_ef_kgco2e_per_kwh,
                    auxiliary_pct=request.auxiliary_pct,
                    technology=request.technology,
                )
            else:
                # Use full-load COP path
                logger.info(
                    "Routing to full-load: cop=%s for %s",
                    effective_cop, request.technology.value,
                )
                return self.calculate_full_load(
                    cooling_kwh_th=request.cooling_output_kwh_th,
                    cop=effective_cop,
                    grid_ef=request.grid_ef_kgco2e_per_kwh,
                    auxiliary_pct=request.auxiliary_pct,
                    technology=request.technology,
                )

        except Exception as exc:
            duration_s = time.monotonic() - start_time
            self._metrics.record_error("calculation", "calculation")
            logger.error(
                "calculate_electric_chiller failed: "
                "technology=%s, error=%s, duration=%.4fs",
                request.technology.value, str(exc), duration_s,
                exc_info=True,
            )
            raise

    # ==================================================================
    # Multi-Chiller Plant Calculations
    # ==================================================================

    def calculate_weighted_cop(
        self,
        chillers: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate capacity-weighted average COP for a chiller plant.

        For a plant with multiple chillers, the weighted average COP
        is computed as:
            Weighted_COP = sum(cooling_i) / sum(cooling_i / cop_i)

        This is the harmonic mean weighted by cooling output, which
        correctly represents the combined efficiency of multiple
        chillers operating at different loads and efficiencies.

        Args:
            chillers: List of chiller dictionaries, each containing:
                - 'cooling_kwh_th': Decimal cooling output
                - 'cop': Decimal COP value

        Returns:
            Capacity-weighted average COP as a quantized Decimal.

        Raises:
            ValueError: If chillers list is empty, any cooling output
                is not positive, or any COP is not positive.
        """
        if not chillers:
            raise ValueError("chillers list must not be empty")

        total_cooling = _ZERO
        total_electrical = _ZERO

        for idx, chiller in enumerate(chillers):
            cooling = Decimal(str(chiller.get("cooling_kwh_th", "0")))
            cop = Decimal(str(chiller.get("cop", "0")))

            self._validate_positive(
                cooling, f"chillers[{idx}].cooling_kwh_th",
            )
            self._validate_positive(cop, f"chillers[{idx}].cop")

            total_cooling += cooling
            total_electrical += self._quantize(cooling / cop)

        if total_electrical <= _ZERO:
            raise ValueError(
                "Total electrical input is zero; cannot compute "
                "weighted COP"
            )

        weighted_cop = self._quantize(total_cooling / total_electrical)
        logger.debug(
            "Weighted COP for %d chillers: total_cooling=%s, "
            "total_electrical=%s, weighted_cop=%s",
            len(chillers), total_cooling, total_electrical, weighted_cop,
        )
        return weighted_cop

    def calculate_multi_chiller(
        self,
        chillers: List[Dict[str, Any]],
        grid_ef: Decimal,
    ) -> CalculationResult:
        """Calculate emissions for a multi-chiller plant.

        Computes individual chiller emissions and aggregates them into
        a single CalculationResult with the capacity-weighted average
        COP and total emissions.

        Each chiller dictionary must contain:
            - 'cooling_kwh_th': Decimal cooling output in kWh thermal
            - 'cop': Decimal COP value
            - 'technology': Optional CoolingTechnology string value
            - 'auxiliary_pct': Optional Decimal auxiliary fraction

        Args:
            chillers: List of chiller specifications.
            grid_ef: Grid electricity emission factor (kgCO2e/kWh).

        Returns:
            Aggregated CalculationResult for the entire plant.

        Raises:
            ValueError: If chillers list is empty or any chiller
                has invalid parameters.
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []

        try:
            if not chillers:
                raise ValueError("chillers list must not be empty")
            self._validate_non_negative(grid_ef, "grid_ef")

            trace.append(self._make_trace_step(
                "MULTI_CHILLER_START",
                f"Processing {len(chillers)} chillers, "
                f"grid_ef={grid_ef}",
            ))

            # -- Provenance chain --
            chain_id = self._provenance.create_chain(calc_id)
            self._provenance.add_stage(chain_id, "INPUT_VALIDATION", {
                "calculation_id": calc_id,
                "method": "multi_chiller",
                "chiller_count": str(len(chillers)),
                "grid_ef": str(grid_ef),
            })

            # -- Process each chiller --
            total_cooling = _ZERO
            total_electricity = _ZERO
            total_emissions = _ZERO
            chiller_details: List[Dict[str, str]] = []

            for idx, chiller in enumerate(chillers):
                cooling = Decimal(str(
                    chiller.get("cooling_kwh_th", "0")
                ))
                cop = Decimal(str(chiller.get("cop", "0")))
                aux_pct_raw = chiller.get("auxiliary_pct")
                aux_pct = (
                    Decimal(str(aux_pct_raw))
                    if aux_pct_raw is not None
                    else self._config.default_auxiliary_pct
                )

                self._validate_positive(
                    cooling, f"chillers[{idx}].cooling_kwh_th",
                )
                self._validate_positive(cop, f"chillers[{idx}].cop")
                self._validate_cop(cop, f"chillers[{idx}].cop")
                self._validate_fraction(
                    aux_pct, f"chillers[{idx}].auxiliary_pct",
                )

                # Electrical input
                elec_input = self._quantize(cooling / cop)
                # Auxiliary
                aux_kwh = self._quantize(cooling * aux_pct)
                # Total for this chiller
                chiller_total_elec = self._quantize(elec_input + aux_kwh)
                chiller_emissions = self._quantize(
                    chiller_total_elec * grid_ef
                )

                total_cooling += cooling
                total_electricity += chiller_total_elec
                total_emissions += chiller_emissions

                tech_val = chiller.get("technology", "unknown")
                chiller_details.append({
                    "index": str(idx),
                    "technology": str(tech_val),
                    "cooling_kwh_th": str(cooling),
                    "cop": str(cop),
                    "electrical_input_kwh": str(elec_input),
                    "auxiliary_kwh": str(aux_kwh),
                    "total_electricity_kwh": str(chiller_total_elec),
                    "emissions_kgco2e": str(chiller_emissions),
                })

                trace.append(self._make_trace_step(
                    f"CHILLER_{idx}",
                    f"cooling={cooling}, cop={cop}, "
                    f"elec={chiller_total_elec}, "
                    f"emissions={chiller_emissions} kgCO2e",
                ))

            # -- Provenance for aggregation --
            self._provenance.add_stage(chain_id, "AGGREGATION", {
                "chiller_count": str(len(chillers)),
                "total_cooling_kwh_th": str(total_cooling),
                "total_electricity_kwh": str(total_electricity),
                "total_emissions_kgco2e": str(total_emissions),
            })

            # -- Weighted COP --
            total_emissions = self._quantize(total_emissions)
            total_electricity = self._quantize(total_electricity)
            total_cooling = self._quantize(total_cooling)

            weighted_cop = self.calculate_weighted_cop(chillers)
            trace.append(self._make_trace_step(
                "WEIGHTED_COP",
                f"weighted_cop={weighted_cop} across "
                f"{len(chillers)} chillers",
            ))

            # -- Gas decomposition --
            gas_breakdown = self.decompose_grid_emissions(
                total_emissions, "AR5",
            )
            trace.append(self._make_trace_step(
                "GAS_DECOMPOSITION",
                f"Decomposed {total_emissions} kgCO2e into "
                f"{len(gas_breakdown)} gas species",
            ))
            self._provenance.add_stage(chain_id, "GAS_DECOMPOSITION", {
                "total_co2e_kg": str(total_emissions),
            })

            # -- Seal provenance --
            provenance_hash = self._provenance.seal_chain(chain_id)
            trace.append(self._make_trace_step(
                "RESULT_FINALIZED",
                f"provenance_hash={provenance_hash[:16]}...",
            ))

            duration_s = time.monotonic() - start_time

            self._metrics.record_calculation(
                technology="multi_chiller",
                calculation_type="electric",
                tier="tier_1",
                tenant_id="default",
                status="success",
                duration_s=duration_s,
                emissions_kgco2e=float(total_emissions),
                cooling_kwh_th=float(total_cooling),
                cop_used=float(weighted_cop),
                condenser_type="mixed",
            )

            result = CalculationResult(
                calculation_id=calc_id,
                calculation_type="electric_chiller_multi",
                cooling_output_kwh_th=total_cooling,
                energy_input_kwh=total_electricity,
                cop_used=weighted_cop,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=DataQualityTier.TIER_1,
                provenance_hash=provenance_hash,
                trace_steps=trace,
                metadata={
                    "method": "multi_chiller",
                    "chiller_count": str(len(chillers)),
                    "grid_ef": str(grid_ef),
                    "weighted_cop": str(weighted_cop),
                },
            )

            logger.info(
                "Multi-chiller calculation complete: calc_id=%s, "
                "%d chillers, total_emissions=%s kgCO2e, "
                "weighted_cop=%s, duration=%.4fs",
                calc_id, len(chillers), total_emissions,
                weighted_cop, duration_s,
            )
            return result

        except Exception as exc:
            duration_s = time.monotonic() - start_time
            self._metrics.record_error("calculation", "calculation")
            logger.error(
                "Multi-chiller calculation failed: calc_id=%s, "
                "error=%s, duration=%.4fs",
                calc_id, str(exc), duration_s,
                exc_info=True,
            )
            raise

    # ==================================================================
    # Advanced: Estimated IPLV from Full-Load COP
    # ==================================================================

    def calculate_estimated_iplv(
        self,
        cop_full: Decimal,
    ) -> Decimal:
        """Estimate IPLV from full-load COP using part-load multipliers.

        When only the full-load COP is known and no part-load test data
        is available, estimate the IPLV by applying typical part-load
        performance multipliers and then computing the AHRI 550/590
        weighted average.

        This is a Tier 1 estimation method with higher uncertainty
        than measured part-load data.

        Args:
            cop_full: Full-load COP value.

        Returns:
            Estimated IPLV as a quantized Decimal.

        Raises:
            ValueError: If cop_full is not positive.
        """
        cops = self.estimate_part_load_cops(cop_full)
        iplv = self.calculate_iplv(
            cops["cop_100"],
            cops["cop_75"],
            cops["cop_50"],
            cops["cop_25"],
        )
        logger.debug(
            "Estimated IPLV from full-load COP %s: %s",
            cop_full, iplv,
        )
        return iplv

    # ==================================================================
    # Advanced: COP from EER/kW_per_ton with Condenser Adjustment
    # ==================================================================

    def resolve_cop_from_efficiency(
        self,
        value: Decimal,
        metric: str,
        technology: CoolingTechnology,
        actual_condenser: Optional[str] = None,
    ) -> Decimal:
        """Resolve COP from any efficiency metric with condenser adjustment.

        Converts the provided efficiency metric (EER, kW/ton, SEER, or
        COP) to COP and optionally adjusts for condenser type mismatch
        between the technology's rated condenser and the actual
        installation.

        Args:
            value: Efficiency metric value.
            metric: Metric type ('eer', 'kw_per_ton', 'seer', 'cop').
            technology: CoolingTechnology to determine rated condenser.
            actual_condenser: Actual condenser type. If None, assumes
                it matches the technology's rated condenser.

        Returns:
            Final adjusted COP as a quantized Decimal.

        Raises:
            ValueError: If value is not positive, metric is unknown,
                or technology is not found.
        """
        cop = self.convert_to_cop(value, metric)
        self._validate_cop(cop, f"converted_cop_from_{metric}")

        if actual_condenser is not None:
            tech_condenser = self._get_condenser_type_str(technology)
            cop = self.adjust_cop_for_condenser(
                cop, tech_condenser, actual_condenser,
            )
            self._validate_cop(cop, "condenser_adjusted_cop")

        logger.debug(
            "Resolved COP: %s %s -> %s (technology=%s, "
            "condenser=%s)",
            value, metric, cop, technology.value,
            actual_condenser or "default",
        )
        return cop

    # ==================================================================
    # Advanced: Batch Electric Chiller Calculations
    # ==================================================================

    def calculate_batch(
        self,
        requests: List[ElectricChillerRequest],
    ) -> List[CalculationResult]:
        """Calculate emissions for a batch of electric chiller requests.

        Processes multiple ElectricChillerRequest objects sequentially,
        collecting results and errors. Each request is independent and
        failure of one does not affect others.

        Args:
            requests: List of ElectricChillerRequest objects.

        Returns:
            List of CalculationResult objects in the same order as
            the input requests. Failed calculations are included with
            zero emissions and error metadata.

        Raises:
            ValueError: If the requests list exceeds the configured
                maximum batch size.
        """
        start_time = time.monotonic()

        if len(requests) > self._config.max_batch_size:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum "
                f"{self._config.max_batch_size}"
            )

        logger.info(
            "Processing batch of %d electric chiller requests",
            len(requests),
        )

        results: List[CalculationResult] = []
        success_count = 0
        failure_count = 0

        for idx, request in enumerate(requests):
            try:
                result = self.calculate_electric_chiller(request)
                results.append(result)
                success_count += 1
            except Exception as exc:
                logger.warning(
                    "Batch item %d failed: %s", idx, str(exc),
                )
                failure_count += 1
                # Create error result
                error_result = CalculationResult(
                    calculation_id=str(uuid.uuid4()),
                    calculation_type="electric_chiller_error",
                    cooling_output_kwh_th=request.cooling_output_kwh_th,
                    energy_input_kwh=_ZERO,
                    cop_used=Decimal("1"),
                    emissions_kgco2e=_ZERO,
                    gas_breakdown=[],
                    calculation_tier=request.calculation_tier,
                    provenance_hash="",
                    trace_steps=[
                        self._make_trace_step(
                            "ERROR", str(exc),
                        ),
                    ],
                    metadata={
                        "error": str(exc),
                        "batch_index": str(idx),
                        "technology": request.technology.value,
                    },
                )
                results.append(error_result)

        duration_s = time.monotonic() - start_time
        logger.info(
            "Batch complete: %d requests, %d success, %d failed, "
            "duration=%.4fs",
            len(requests), success_count, failure_count, duration_s,
        )
        return results

    # ==================================================================
    # Calculation with Full Detail (Tier 2/3 support)
    # ==================================================================

    def calculate_with_tier(
        self,
        request: ElectricChillerRequest,
        measured_cop: Optional[Decimal] = None,
        part_load_cops: Optional[Dict[str, Decimal]] = None,
        actual_condenser: Optional[str] = None,
    ) -> CalculationResult:
        """Calculate with explicit tier support and advanced parameters.

        Supports Tier 2 (measured COP) and Tier 3 (measured part-load
        COP with condenser adjustment) calculations with additional
        parameters not available through the standard request model.

        Args:
            request: Base ElectricChillerRequest.
            measured_cop: Tier 2/3 measured full-load COP.
            part_load_cops: Tier 3 measured part-load COPs with keys
                'cop_100', 'cop_75', 'cop_50', 'cop_25'.
            actual_condenser: Actual condenser type if different from
                the technology's rated condenser.

        Returns:
            CalculationResult with appropriate tier metadata.

        Raises:
            ValueError: If parameters are invalid.
        """
        start_time = time.monotonic()

        # Determine effective tier
        if part_load_cops is not None:
            effective_tier = DataQualityTier.TIER_3
        elif measured_cop is not None:
            effective_tier = DataQualityTier.TIER_2
        else:
            effective_tier = request.calculation_tier

        logger.info(
            "Tier-aware calculation: technology=%s, "
            "effective_tier=%s",
            request.technology.value, effective_tier.value,
        )

        # Tier 3: measured part-load COPs
        if part_load_cops is not None:
            cop_100 = Decimal(str(part_load_cops.get("cop_100", "0")))
            cop_75 = Decimal(str(part_load_cops.get("cop_75", "0")))
            cop_50 = Decimal(str(part_load_cops.get("cop_50", "0")))
            cop_25 = Decimal(str(part_load_cops.get("cop_25", "0")))

            # Apply condenser adjustment if needed
            if actual_condenser is not None:
                tech_cond = self._get_condenser_type_str(
                    request.technology
                )
                cop_100 = self.adjust_cop_for_condenser(
                    cop_100, tech_cond, actual_condenser,
                )
                cop_75 = self.adjust_cop_for_condenser(
                    cop_75, tech_cond, actual_condenser,
                )
                cop_50 = self.adjust_cop_for_condenser(
                    cop_50, tech_cond, actual_condenser,
                )
                cop_25 = self.adjust_cop_for_condenser(
                    cop_25, tech_cond, actual_condenser,
                )

            return self.calculate_custom_part_load(
                cooling_kwh_th=request.cooling_output_kwh_th,
                cop_100=cop_100,
                cop_75=cop_75,
                cop_50=cop_50,
                cop_25=cop_25,
                grid_ef=request.grid_ef_kgco2e_per_kwh,
                auxiliary_pct=request.auxiliary_pct,
                technology=request.technology,
            )

        # Tier 2: measured full-load COP
        if measured_cop is not None:
            cop = measured_cop
            if actual_condenser is not None:
                tech_cond = self._get_condenser_type_str(
                    request.technology
                )
                cop = self.adjust_cop_for_condenser(
                    cop, tech_cond, actual_condenser,
                )

            if request.use_iplv:
                estimated_iplv = self.calculate_estimated_iplv(cop)
                return self.calculate_iplv_weighted(
                    cooling_kwh_th=request.cooling_output_kwh_th,
                    iplv=estimated_iplv,
                    grid_ef=request.grid_ef_kgco2e_per_kwh,
                    auxiliary_pct=request.auxiliary_pct,
                    technology=request.technology,
                )
            else:
                return self.calculate_full_load(
                    cooling_kwh_th=request.cooling_output_kwh_th,
                    cop=cop,
                    grid_ef=request.grid_ef_kgco2e_per_kwh,
                    auxiliary_pct=request.auxiliary_pct,
                    technology=request.technology,
                )

        # Tier 1: default COP from technology spec
        return self.calculate_electric_chiller(request)

    # ==================================================================
    # Summary and Reporting Helpers
    # ==================================================================

    def calculate_emission_intensity(
        self,
        emissions_kgco2e: Decimal,
        cooling_kwh_th: Decimal,
    ) -> Decimal:
        """Calculate emission intensity (kgCO2e per kWh_th cooling).

        Args:
            emissions_kgco2e: Total CO2e emissions in kg.
            cooling_kwh_th: Total cooling output in kWh thermal.

        Returns:
            Emission intensity as kgCO2e per kWh_th, quantized.

        Raises:
            ValueError: If cooling_kwh_th is not positive.
        """
        self._validate_non_negative(emissions_kgco2e, "emissions_kgco2e")
        self._validate_positive(cooling_kwh_th, "cooling_kwh_th")

        intensity = self._quantize(emissions_kgco2e / cooling_kwh_th)
        logger.debug(
            "Emission intensity: %s kgCO2e / %s kWh_th = %s",
            emissions_kgco2e, cooling_kwh_th, intensity,
        )
        return intensity

    def calculate_electricity_intensity(
        self,
        energy_input_kwh: Decimal,
        cooling_kwh_th: Decimal,
    ) -> Decimal:
        """Calculate electricity intensity (kWh_e per kWh_th cooling).

        This is the reciprocal of the effective system COP including
        auxiliary energy.

        Args:
            energy_input_kwh: Total electricity input in kWh.
            cooling_kwh_th: Total cooling output in kWh thermal.

        Returns:
            Electricity intensity as kWh_e per kWh_th, quantized.

        Raises:
            ValueError: If cooling_kwh_th is not positive.
        """
        self._validate_non_negative(energy_input_kwh, "energy_input_kwh")
        self._validate_positive(cooling_kwh_th, "cooling_kwh_th")

        intensity = self._quantize(energy_input_kwh / cooling_kwh_th)
        logger.debug(
            "Electricity intensity: %s kWh / %s kWh_th = %s",
            energy_input_kwh, cooling_kwh_th, intensity,
        )
        return intensity

    def compare_full_load_vs_iplv(
        self,
        cooling_kwh_th: Decimal,
        cop_full: Decimal,
        grid_ef: Decimal,
        auxiliary_pct: Optional[Decimal] = None,
        technology: Optional[CoolingTechnology] = None,
    ) -> Dict[str, Any]:
        """Compare full-load COP vs estimated IPLV emissions.

        Runs both calculation methods and returns a comparison showing
        the emission difference, useful for reporting the impact of
        part-load operation.

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop_full: Full-load COP value.
            grid_ef: Grid electricity emission factor (kgCO2e/kWh).
            auxiliary_pct: Auxiliary energy fraction (0-1).
            technology: Optional technology for metadata.

        Returns:
            Dictionary containing:
                - full_load_result: CalculationResult from full-load
                - iplv_result: CalculationResult from estimated IPLV
                - emission_difference_kgco2e: Decimal difference
                - emission_reduction_pct: Decimal percentage reduction
                - full_load_cop: Decimal
                - estimated_iplv: Decimal
        """
        self._validate_positive(cooling_kwh_th, "cooling_kwh_th")
        self._validate_positive(cop_full, "cop_full")
        self._validate_non_negative(grid_ef, "grid_ef")

        # Full-load calculation
        full_load_result = self.calculate_full_load(
            cooling_kwh_th=cooling_kwh_th,
            cop=cop_full,
            grid_ef=grid_ef,
            auxiliary_pct=auxiliary_pct,
            technology=technology,
        )

        # Estimated IPLV calculation
        estimated_iplv = self.calculate_estimated_iplv(cop_full)
        iplv_result = self.calculate_iplv_weighted(
            cooling_kwh_th=cooling_kwh_th,
            iplv=estimated_iplv,
            grid_ef=grid_ef,
            auxiliary_pct=auxiliary_pct,
            technology=technology,
        )

        # Comparison
        diff = self._quantize(
            full_load_result.emissions_kgco2e
            - iplv_result.emissions_kgco2e
        )
        pct = _ZERO
        if full_load_result.emissions_kgco2e > _ZERO:
            pct = self._quantize(
                (diff / full_load_result.emissions_kgco2e)
                * Decimal("100")
            )

        comparison = {
            "full_load_result": full_load_result,
            "iplv_result": iplv_result,
            "emission_difference_kgco2e": diff,
            "emission_reduction_pct": pct,
            "full_load_cop": cop_full,
            "estimated_iplv": estimated_iplv,
        }

        logger.info(
            "Full-load vs IPLV comparison: full=%s kgCO2e, "
            "iplv=%s kgCO2e, saving=%s kgCO2e (%s%%)",
            full_load_result.emissions_kgco2e,
            iplv_result.emissions_kgco2e,
            diff, pct,
        )
        return comparison

    # ==================================================================
    # Technology Information Helpers
    # ==================================================================

    def get_technology_cop_range(
        self,
        technology: CoolingTechnology,
    ) -> Dict[str, Decimal]:
        """Get the COP range for a technology.

        Args:
            technology: CoolingTechnology enum value.

        Returns:
            Dictionary with keys 'cop_min', 'cop_max', 'cop_default',
            'iplv' (or None).
        """
        spec = self._get_technology_spec(technology)
        return {
            "cop_min": spec.cop_min,
            "cop_max": spec.cop_max,
            "cop_default": spec.cop_default,
            "iplv": spec.iplv if spec.iplv is not None else _ZERO,
        }

    def get_supported_technologies(self) -> List[str]:
        """Return list of supported electric chiller technology names.

        Returns:
            Sorted list of CoolingTechnology enum values that are
            electric chiller types.
        """
        electric_techs = [
            CoolingTechnology.WATER_COOLED_CENTRIFUGAL,
            CoolingTechnology.AIR_COOLED_CENTRIFUGAL,
            CoolingTechnology.WATER_COOLED_SCREW,
            CoolingTechnology.AIR_COOLED_SCREW,
            CoolingTechnology.WATER_COOLED_RECIPROCATING,
            CoolingTechnology.AIR_COOLED_SCROLL,
        ]
        return sorted([t.value for t in electric_techs])

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata and configuration summary.

        Returns:
            Dictionary with engine name, version, configuration
            settings, and supported technology count.
        """
        return {
            "engine_name": "ElectricChillerCalculatorEngine",
            "agent": "AGENT-MRV-012",
            "module": "cooling_purchase",
            "decimal_places": self._config.decimal_places,
            "use_iplv_default": self._config.use_iplv_default,
            "default_auxiliary_pct": str(
                self._config.default_auxiliary_pct
            ),
            "min_cop": str(self._config.min_cop),
            "max_cop": str(self._config.max_cop),
            "default_condenser_type": self._config.default_condenser_type,
            "ahri_weights": {
                "100%": str(self._config.ahri_100_weight),
                "75%": str(self._config.ahri_75_weight),
                "50%": str(self._config.ahri_50_weight),
                "25%": str(self._config.ahri_25_weight),
            },
            "supported_technologies": self.get_supported_technologies(),
            "part_load_multipliers": {
                "75%": str(_PART_LOAD_MULTIPLIER_75),
                "50%": str(_PART_LOAD_MULTIPLIER_50),
                "25%": str(_PART_LOAD_MULTIPLIER_25),
            },
            "grid_gas_shares": {
                k: str(v) for k, v in self.get_grid_gas_shares().items()
            },
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================


def get_electric_chiller_calculator() -> ElectricChillerCalculatorEngine:
    """Return the process-wide singleton ElectricChillerCalculatorEngine.

    This is the recommended entry point for obtaining the engine
    in production code. The singleton is created lazily on first call
    and reused for all subsequent calls.

    Returns:
        The singleton ElectricChillerCalculatorEngine instance.

    Example:
        >>> engine_a = get_electric_chiller_calculator()
        >>> engine_b = get_electric_chiller_calculator()
        >>> assert engine_a is engine_b
    """
    return ElectricChillerCalculatorEngine()


def reset_electric_chiller_calculator() -> None:
    """Destroy the singleton ElectricChillerCalculatorEngine.

    Useful for testing to reset internal state and re-read
    configuration on the next instantiation.
    """
    ElectricChillerCalculatorEngine.reset()


def calculate_electric_chiller(
    request: ElectricChillerRequest,
) -> CalculationResult:
    """Calculate Scope 2 emissions from an electric chiller.

    Module-level convenience function that delegates to the singleton
    engine's ``calculate_electric_chiller`` method.

    Args:
        request: ElectricChillerRequest with calculation parameters.

    Returns:
        CalculationResult with complete emission details.

    Example:
        >>> from greenlang.agents.mrv.cooling_purchase.models import (
        ...     ElectricChillerRequest, CoolingTechnology,
        ... )
        >>> from decimal import Decimal
        >>> req = ElectricChillerRequest(
        ...     cooling_output_kwh_th=Decimal("100000"),
        ...     technology=CoolingTechnology.WATER_COOLED_SCREW,
        ...     grid_ef_kgco2e_per_kwh=Decimal("0.50"),
        ... )
        >>> result = calculate_electric_chiller(req)
        >>> assert result.emissions_kgco2e > 0
    """
    return get_electric_chiller_calculator().calculate_electric_chiller(
        request
    )


def calculate_iplv(
    cop_100: Decimal,
    cop_75: Decimal,
    cop_50: Decimal,
    cop_25: Decimal,
) -> Decimal:
    """Calculate IPLV per AHRI 550/590.

    Module-level convenience function.

    Args:
        cop_100: COP at 100% load.
        cop_75: COP at 75% load.
        cop_50: COP at 50% load.
        cop_25: COP at 25% load.

    Returns:
        IPLV as a quantized Decimal.
    """
    return get_electric_chiller_calculator().calculate_iplv(
        cop_100, cop_75, cop_50, cop_25,
    )


def estimate_part_load_cops(
    cop_full: Decimal,
) -> Dict[str, Decimal]:
    """Estimate part-load COPs from full-load COP.

    Module-level convenience function.

    Args:
        cop_full: Full-load COP value.

    Returns:
        Dictionary with keys 'cop_100', 'cop_75', 'cop_50', 'cop_25'.
    """
    return get_electric_chiller_calculator().estimate_part_load_cops(
        cop_full,
    )


def convert_to_cop(
    value: Decimal,
    metric: str,
) -> Decimal:
    """Convert an efficiency metric value to COP.

    Module-level convenience function.

    Args:
        value: Efficiency metric value.
        metric: Source metric type ('eer', 'kw_per_ton', 'seer', 'cop').

    Returns:
        COP as a quantized Decimal.
    """
    return get_electric_chiller_calculator().convert_to_cop(
        value, metric,
    )


def decompose_grid_emissions(
    total_co2e: Decimal,
    gwp_source: Optional[str] = None,
) -> List[GasEmissionDetail]:
    """Decompose total CO2e into constituent greenhouse gases.

    Module-level convenience function.

    Args:
        total_co2e: Total CO2-equivalent emissions in kg.
        gwp_source: IPCC Assessment Report edition.

    Returns:
        List of GasEmissionDetail objects.
    """
    return get_electric_chiller_calculator().decompose_grid_emissions(
        total_co2e, gwp_source,
    )


def calculate_multi_chiller(
    chillers: List[Dict[str, Any]],
    grid_ef: Decimal,
) -> CalculationResult:
    """Calculate emissions for a multi-chiller plant.

    Module-level convenience function.

    Args:
        chillers: List of chiller specifications.
        grid_ef: Grid electricity emission factor (kgCO2e/kWh).

    Returns:
        Aggregated CalculationResult for the entire plant.
    """
    return get_electric_chiller_calculator().calculate_multi_chiller(
        chillers, grid_ef,
    )
