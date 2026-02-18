# -*- coding: utf-8 -*-
"""
CombustionEfficiencyEngine - Engine 3: Flaring Agent (AGENT-MRV-006)

Models flare combustion efficiency (CE) and Destruction and Removal
Efficiency (DRE) accounting for environmental and operational conditions
including wind speed, tip velocity, gas heating value, steam/air assist
ratios, and flare type characteristics.

Combustion efficiency determines the fraction of hydrocarbons that are
fully oxidized to CO2 and H2O. The remainder (1 - CE) escapes as
uncombusted hydrocarbons (primarily CH4), which has significantly higher
GWP than CO2.

Adjustment Model:
    CE_final = CE_base * wind_factor * velocity_factor * lhv_factor
               * assist_factor

Where each factor is a multiplicative modifier in the range [0, 1].

Key Parameters:
    - CE_base: Default efficiency for the flare type (0.93-0.995)
    - wind_factor: Wind speed degradation above 5 m/s
    - velocity_factor: Tip velocity effect (Mach-based)
    - lhv_factor: Lower heating value stability threshold
    - assist_factor: Steam or air assist ratio optimization

Destruction and Removal Efficiency (DRE):
    DRE = 1 - (mass_out / mass_in) for specific compounds
    Typical: 98% general, 99.5% enclosed, 99.9% with post-combustion

Data Sources:
    - EPA 40 CFR Part 98 Subpart W (Section W.23)
    - TCEQ Flare Study (2010)
    - API Standard 521 (Pressure-relieving and Depressuring Systems)
    - John Zink Hamworthy Combustion Handbook
    - University of Alberta Flare Research Project

Example:
    >>> from greenlang.flaring.combustion_efficiency import CombustionEfficiencyEngine
    >>> from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
    >>> from decimal import Decimal
    >>> db = FlareSystemDatabaseEngine()
    >>> ce_engine = CombustionEfficiencyEngine(flare_database=db)
    >>> result = ce_engine.get_effective_ce(
    ...     flare_type="ELEVATED_STEAM_ASSISTED",
    ...     wind_speed_m_s=Decimal("8.0"),
    ...     tip_velocity_mach=Decimal("0.3"),
    ...     lhv_btu_scf=Decimal("800"),
    ...     steam_to_gas_ratio=Decimal("0.4"),
    ... )
    >>> print(result["effective_ce"])  # ~0.95

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["CombustionEfficiencyEngine"]

# ---------------------------------------------------------------------------
# Conditional imports for GreenLang infrastructure
# ---------------------------------------------------------------------------

try:
    from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
    _DATABASE_AVAILABLE = True
except ImportError:
    _DATABASE_AVAILABLE = False
    FlareSystemDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.flaring.metrics import (
        record_calculation as _record_calculation,
        observe_calculation_duration as _observe_calculation_duration,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places


# ===========================================================================
# Default Combustion Efficiency by Flare Type
# ===========================================================================

_DEFAULT_CE_BY_FLARE_TYPE: Dict[str, Decimal] = {
    "ELEVATED_STEAM_ASSISTED": Decimal("0.98"),
    "ELEVATED_AIR_ASSISTED":   Decimal("0.98"),
    "ELEVATED_UNASSISTED":     Decimal("0.965"),
    "ENCLOSED_GROUND":         Decimal("0.995"),
    "MULTI_POINT_GROUND":      Decimal("0.99"),
    "OFFSHORE_MARINE":         Decimal("0.95"),
    "CANDLESTICK":             Decimal("0.95"),
    "LOW_PRESSURE":            Decimal("0.93"),
}

# Display names for reporting
_FLARE_TYPE_DISPLAY: Dict[str, str] = {
    "ELEVATED_STEAM_ASSISTED": "Elevated Steam-Assisted",
    "ELEVATED_AIR_ASSISTED":   "Elevated Air-Assisted",
    "ELEVATED_UNASSISTED":     "Elevated Unassisted",
    "ENCLOSED_GROUND":         "Enclosed Ground",
    "MULTI_POINT_GROUND":      "Multi-Point Ground (MPGF)",
    "OFFSHORE_MARINE":         "Offshore Marine",
    "CANDLESTICK":             "Candlestick",
    "LOW_PRESSURE":            "Low-Pressure",
}


# ===========================================================================
# Wind Speed Adjustment Parameters
# ===========================================================================

#: Wind speed threshold below which no CE penalty applies (m/s)
_WIND_THRESHOLD_LOW = Decimal("5.0")

#: Wind speed above which the maximum penalty floor applies (m/s)
_WIND_THRESHOLD_HIGH = Decimal("15.0")

#: CE degradation rate per m/s above threshold (fraction per m/s)
_WIND_CE_DEGRADATION_RATE = Decimal("0.01")

#: Minimum wind factor floor (maximum 10% reduction)
_WIND_FACTOR_FLOOR = Decimal("0.90")


# ===========================================================================
# Tip Velocity (Mach Number) Parameters
# ===========================================================================

#: Optimal Mach range lower bound
_MACH_OPTIMAL_LOW = Decimal("0.2")

#: Optimal Mach range upper bound
_MACH_OPTIMAL_HIGH = Decimal("0.5")

#: CE penalty for low velocity (Mach < 0.2): stability concern
_MACH_LOW_PENALTY = Decimal("0.01")

#: CE penalty per 0.1 Mach above optimal (flame lift-off risk)
_MACH_HIGH_PENALTY_PER_01 = Decimal("0.02")

#: Maximum Mach penalty (even at very high velocities)
_MACH_PENALTY_CAP = Decimal("0.20")


# ===========================================================================
# LHV Stability Threshold Parameters
# ===========================================================================

#: Minimum LHV for stable combustion (BTU/scf)
_LHV_STABLE_THRESHOLD = Decimal("200")

#: LHV below which flare may not sustain flame (BTU/scf)
_LHV_CRITICAL_THRESHOLD = Decimal("100")

#: CE reduction for LHV below stable threshold but above critical
_LHV_BELOW_STABLE_PENALTY = Decimal("0.05")

#: CE reduction for LHV below critical threshold
_LHV_BELOW_CRITICAL_PENALTY = Decimal("0.15")


# ===========================================================================
# Steam Assist Parameters
# ===========================================================================

#: Optimal steam-to-gas ratio lower bound (lb steam/lb gas)
_STEAM_RATIO_OPTIMAL_LOW = Decimal("0.3")

#: Optimal steam-to-gas ratio upper bound
_STEAM_RATIO_OPTIMAL_HIGH = Decimal("0.5")

#: Steam-to-gas ratio above which over-steaming penalty applies
_STEAM_RATIO_OVER_THRESHOLD = Decimal("0.7")

#: CE penalty for over-steaming (quenching effect)
_STEAM_OVER_PENALTY = Decimal("0.03")

#: Steam-to-gas ratio below which under-steaming is flagged
_STEAM_RATIO_UNDER_THRESHOLD = Decimal("0.2")

# Under-steaming does not penalize CE, but produces visible smoke


# ===========================================================================
# Air Assist Parameters
# ===========================================================================

#: Minimum excess air fraction for optimal operation
_AIR_EXCESS_OPTIMAL_LOW = Decimal("0.20")

#: Maximum excess air fraction for optimal operation
_AIR_EXCESS_OPTIMAL_HIGH = Decimal("0.50")

#: Excess air threshold above which cooling penalty applies
_AIR_EXCESS_OVER_THRESHOLD = Decimal("1.00")

#: CE penalty for excessive air (cooling effect)
_AIR_EXCESS_OVER_PENALTY = Decimal("0.02")


# ===========================================================================
# DRE Parameters
# ===========================================================================

#: Default DRE by flare type category
_DEFAULT_DRE: Dict[str, Decimal] = {
    "GENERAL":         Decimal("0.98"),
    "ENCLOSED":        Decimal("0.995"),
    "WITH_POST_COMB":  Decimal("0.999"),
    "ELEVATED_STEAM_ASSISTED": Decimal("0.98"),
    "ELEVATED_AIR_ASSISTED":   Decimal("0.98"),
    "ELEVATED_UNASSISTED":     Decimal("0.96"),
    "ENCLOSED_GROUND":         Decimal("0.995"),
    "MULTI_POINT_GROUND":      Decimal("0.99"),
    "OFFSHORE_MARINE":         Decimal("0.95"),
    "CANDLESTICK":             Decimal("0.95"),
    "LOW_PRESSURE":            Decimal("0.93"),
}

#: Speed of sound approximation at standard conditions (m/s)
#: Used for Mach number calculations when not provided directly
_SPEED_OF_SOUND_M_S = Decimal("343.0")


# ===========================================================================
# CombustionEfficiencyEngine
# ===========================================================================


class CombustionEfficiencyEngine:
    """Models flare combustion efficiency and DRE with environmental
    and operational condition adjustments.

    The engine computes effective combustion efficiency by applying
    multiplicative adjustment factors to a base CE value:

        CE_final = CE_base * wind_factor * velocity_factor
                   * lhv_factor * assist_factor

    Each factor is independently calculated based on measured or
    estimated operating conditions. All calculations use Decimal
    arithmetic for precision.

    Thread-safe: all mutable state is guarded by ``threading.Lock()``.

    Attributes:
        _flare_db: Reference to FlareSystemDatabaseEngine.
        _config: Optional configuration dictionary.
        _lock: Thread lock for shared mutable state.
        _provenance: Reference to the provenance tracker.
        _precision_places: Number of Decimal places.

    Example:
        >>> db = FlareSystemDatabaseEngine()
        >>> ce = CombustionEfficiencyEngine(flare_database=db)
        >>> result = ce.get_effective_ce(
        ...     flare_type="ELEVATED_STEAM_ASSISTED",
        ...     wind_speed_m_s=Decimal("12.0"),
        ... )
        >>> assert result["effective_ce"] < Decimal("0.98")
    """

    def __init__(
        self,
        flare_database: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize CombustionEfficiencyEngine.

        Args:
            flare_database: FlareSystemDatabaseEngine instance for
                flare type lookups. If None and module is available,
                a default instance is created.
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance
                  tracking. Defaults to True.
                - ``decimal_precision`` (int): Decimal places.
                  Default 8.
                - ``minimum_ce_floor`` (str): Absolute minimum CE.
                  Default ``"0.50"``.
                - ``enable_wind_adjustment`` (bool): Default True.
                - ``enable_velocity_adjustment`` (bool): Default True.
                - ``enable_lhv_adjustment`` (bool): Default True.
                - ``enable_assist_adjustment`` (bool): Default True.
        """
        if flare_database is not None:
            self._flare_db = flare_database
        elif _DATABASE_AVAILABLE:
            self._flare_db = FlareSystemDatabaseEngine(config=config)
        else:
            self._flare_db = None

        self._config = config or {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True
        )
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places
        self._minimum_ce_floor: Decimal = Decimal(
            str(self._config.get("minimum_ce_floor", "0.50"))
        )
        self._enable_wind: bool = self._config.get(
            "enable_wind_adjustment", True
        )
        self._enable_velocity: bool = self._config.get(
            "enable_velocity_adjustment", True
        )
        self._enable_lhv: bool = self._config.get(
            "enable_lhv_adjustment", True
        )
        self._enable_assist: bool = self._config.get(
            "enable_assist_adjustment", True
        )

        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "CombustionEfficiencyEngine initialized "
            "(precision=%d, min_floor=%s, wind=%s, velocity=%s, "
            "lhv=%s, assist=%s)",
            self._precision_places,
            self._minimum_ce_floor,
            self._enable_wind,
            self._enable_velocity,
            self._enable_lhv,
            self._enable_assist,
        )

    # ==================================================================
    # PUBLIC API: Get Effective CE (Main Entry Point)
    # ==================================================================

    def get_effective_ce(
        self,
        flare_type: Optional[str] = None,
        base_ce: Optional[Decimal] = None,
        wind_speed_m_s: Optional[Decimal] = None,
        tip_velocity_mach: Optional[Decimal] = None,
        tip_velocity_m_s: Optional[Decimal] = None,
        lhv_btu_scf: Optional[Decimal] = None,
        gas_composition: Optional[Dict[str, Decimal]] = None,
        steam_to_gas_ratio: Optional[Decimal] = None,
        air_excess_fraction: Optional[Decimal] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate the effective combustion efficiency with all
        applicable adjustments.

        This is the main entry point. It resolves the base CE, then
        applies multiplicative adjustment factors for wind, tip velocity,
        LHV stability, and steam/air assist.

        Priority for base CE:
        1. Explicit ``base_ce`` parameter
        2. Flare type default from database
        3. 0.98 (EPA default)

        Args:
            flare_type: Flare type identifier for base CE lookup.
            base_ce: Explicit base CE override (0-1).
            wind_speed_m_s: Ambient wind speed in m/s.
            tip_velocity_mach: Flare tip exit velocity as Mach number.
                Mutually exclusive with tip_velocity_m_s.
            tip_velocity_m_s: Flare tip exit velocity in m/s. Converted
                to Mach internally.
            lhv_btu_scf: Lower heating value of flare gas in BTU/scf.
                If gas_composition is provided and lhv_btu_scf is None,
                LHV is calculated from composition.
            gas_composition: Gas composition for LHV calculation.
            steam_to_gas_ratio: Steam-to-gas mass ratio (lb steam/lb gas).
                Only applicable for steam-assisted flares.
            air_excess_fraction: Excess air fraction above stoichiometric.
                Only applicable for air-assisted flares.
            calculation_id: Optional external calculation ID.

        Returns:
            Dictionary with keys:
                - calculation_id (str)
                - flare_type (str or None)
                - base_ce (Decimal)
                - wind_factor (Decimal)
                - velocity_factor (Decimal)
                - lhv_factor (Decimal)
                - assist_factor (Decimal)
                - effective_ce (Decimal)
                - adjustments_applied (List[str])
                - warnings (List[str])
                - provenance_hash (str)
                - processing_time_ms (float)

        Example:
            >>> result = ce_engine.get_effective_ce(
            ...     flare_type="ELEVATED_STEAM_ASSISTED",
            ...     wind_speed_m_s=Decimal("10.0"),
            ...     steam_to_gas_ratio=Decimal("0.4"),
            ... )
            >>> result["effective_ce"]
            Decimal('0.93100000')
        """
        start_time = time.monotonic()
        calc_id = calculation_id or f"fl_ce_{uuid.uuid4().hex[:12]}"
        adjustments: List[str] = []
        warnings: List[str] = []

        # Step 1: Resolve base CE
        resolved_ce = self._resolve_base_ce(
            base_ce, flare_type, adjustments,
        )

        # Step 2: Wind speed adjustment
        wind_factor = Decimal("1.0")
        if self._enable_wind and wind_speed_m_s is not None:
            wind_result = self.adjust_for_wind(wind_speed_m_s)
            wind_factor = wind_result["wind_factor"]
            adjustments.append(
                f"wind: speed={wind_speed_m_s} m/s, factor={wind_factor}"
            )
            if wind_result.get("warnings"):
                warnings.extend(wind_result["warnings"])

        # Step 3: Tip velocity adjustment
        velocity_factor = Decimal("1.0")
        if self._enable_velocity:
            mach = self._resolve_mach(tip_velocity_mach, tip_velocity_m_s)
            if mach is not None:
                velocity_result = self.adjust_for_velocity(mach)
                velocity_factor = velocity_result["velocity_factor"]
                adjustments.append(
                    f"velocity: Mach={mach}, factor={velocity_factor}"
                )
                if velocity_result.get("warnings"):
                    warnings.extend(velocity_result["warnings"])

        # Step 4: LHV stability adjustment
        lhv_factor = Decimal("1.0")
        if self._enable_lhv:
            lhv = self._resolve_lhv(lhv_btu_scf, gas_composition)
            if lhv is not None:
                lhv_result = self.check_lhv_stability(lhv)
                lhv_factor = lhv_result["lhv_factor"]
                adjustments.append(
                    f"lhv: value={lhv} BTU/scf, factor={lhv_factor}"
                )
                if lhv_result.get("warnings"):
                    warnings.extend(lhv_result["warnings"])

        # Step 5: Assist medium adjustment
        assist_factor = Decimal("1.0")
        if self._enable_assist:
            assist_type = self._get_assist_type(flare_type)

            if assist_type == "STEAM" and steam_to_gas_ratio is not None:
                assist_result = self._adjust_for_steam(steam_to_gas_ratio)
                assist_factor = assist_result["assist_factor"]
                adjustments.append(
                    f"steam: ratio={steam_to_gas_ratio}, "
                    f"factor={assist_factor}"
                )
                if assist_result.get("warnings"):
                    warnings.extend(assist_result["warnings"])

            elif assist_type == "AIR" and air_excess_fraction is not None:
                assist_result = self._adjust_for_air(air_excess_fraction)
                assist_factor = assist_result["assist_factor"]
                adjustments.append(
                    f"air: excess={air_excess_fraction}, "
                    f"factor={assist_factor}"
                )
                if assist_result.get("warnings"):
                    warnings.extend(assist_result["warnings"])

        # Step 6: Compose final CE
        effective_ce = self._quantize(
            resolved_ce * wind_factor * velocity_factor
            * lhv_factor * assist_factor
        )

        # Apply absolute floor
        if effective_ce < self._minimum_ce_floor:
            warnings.append(
                f"Effective CE {effective_ce} below minimum floor "
                f"{self._minimum_ce_floor}, clamped to floor"
            )
            effective_ce = self._minimum_ce_floor

        # Ensure CE does not exceed 1.0
        if effective_ce > Decimal("1.0"):
            effective_ce = Decimal("1.0")

        # Provenance
        elapsed_ms = (time.monotonic() - start_time) * 1000
        provenance_hash = self._compute_provenance_hash({
            "calculation_id": calc_id,
            "base_ce": str(resolved_ce),
            "effective_ce": str(effective_ce),
            "wind_factor": str(wind_factor),
            "velocity_factor": str(velocity_factor),
            "lhv_factor": str(lhv_factor),
            "assist_factor": str(assist_factor),
        })

        # Metrics
        if _METRICS_AVAILABLE and _record_calculation is not None:
            _record_calculation(
                flare_type or "UNKNOWN", "ce_calculation", "completed",
            )
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            _observe_calculation_duration(
                "ce_calculation", elapsed_ms / 1000,
            )

        self._record_provenance(
            "calculate_effective_ce", calc_id,
            {
                "flare_type": flare_type,
                "base_ce": str(resolved_ce),
                "effective_ce": str(effective_ce),
                "hash": provenance_hash,
            },
        )

        logger.info(
            "CE calculation %s: base=%s, wind=%s, vel=%s, lhv=%s, "
            "assist=%s -> effective=%s (%.1f ms)",
            calc_id, resolved_ce, wind_factor, velocity_factor,
            lhv_factor, assist_factor, effective_ce, elapsed_ms,
        )

        return {
            "calculation_id": calc_id,
            "flare_type": flare_type,
            "base_ce": resolved_ce,
            "wind_factor": wind_factor,
            "velocity_factor": velocity_factor,
            "lhv_factor": lhv_factor,
            "assist_factor": assist_factor,
            "effective_ce": effective_ce,
            "adjustments_applied": adjustments,
            "warnings": warnings,
            "provenance_hash": provenance_hash,
            "processing_time_ms": elapsed_ms,
        }

    # ==================================================================
    # PUBLIC API: Individual Adjustment Methods
    # ==================================================================

    def calculate_ce(
        self,
        flare_type: str,
    ) -> Decimal:
        """Return the default base combustion efficiency for a flare type.

        This returns the baseline CE without any environmental or
        operational adjustments.

        Args:
            flare_type: Flare type identifier.

        Returns:
            Default CE as Decimal (0-1).

        Raises:
            KeyError: If the flare type is not found.

        Example:
            >>> ce_engine.calculate_ce("ENCLOSED_GROUND")
            Decimal('0.995')
        """
        ft_key = flare_type.upper()

        # Try database first
        if self._flare_db is not None:
            try:
                specs = self._flare_db.get_flare_type_specs(ft_key)
                return specs["typical_ce"]
            except (KeyError, AttributeError):
                pass

        # Fallback to built-in
        if ft_key not in _DEFAULT_CE_BY_FLARE_TYPE:
            raise KeyError(
                f"Unknown flare type: {flare_type}. "
                f"Valid types: {sorted(_DEFAULT_CE_BY_FLARE_TYPE.keys())}"
            )

        return _DEFAULT_CE_BY_FLARE_TYPE[ft_key]

    def adjust_for_wind(
        self,
        wind_speed_m_s: Decimal,
    ) -> Dict[str, Any]:
        """Calculate the wind speed adjustment factor for CE.

        Wind degradation model:
        - wind < 5 m/s: factor = 1.0 (no effect)
        - 5 <= wind <= 15 m/s: factor = 1.0 - 0.01 * (wind - 5)
        - wind > 15 m/s: factor = 0.90 (minimum floor)

        Higher wind speeds cause flame instability and lift-off,
        reducing combustion completeness.

        Args:
            wind_speed_m_s: Ambient wind speed in meters per second.

        Returns:
            Dictionary with:
                - wind_speed_m_s (Decimal)
                - wind_factor (Decimal)
                - category (str): "calm", "moderate", "high", "extreme"
                - warnings (List[str])

        Example:
            >>> result = ce_engine.adjust_for_wind(Decimal("8.0"))
            >>> result["wind_factor"]
            Decimal('0.97000000')
        """
        wind = Decimal(str(wind_speed_m_s))
        warnings: List[str] = []

        if wind < Decimal("0"):
            raise ValueError(
                f"Wind speed cannot be negative: {wind_speed_m_s}"
            )

        if wind < _WIND_THRESHOLD_LOW:
            # Calm conditions: no degradation
            factor = Decimal("1.0")
            category = "calm"
        elif wind <= _WIND_THRESHOLD_HIGH:
            # Moderate: linear degradation
            degradation = _WIND_CE_DEGRADATION_RATE * (
                wind - _WIND_THRESHOLD_LOW
            )
            factor = self._quantize(Decimal("1.0") - degradation)
            category = "moderate" if wind <= Decimal("10") else "high"
        else:
            # Extreme: floor
            factor = _WIND_FACTOR_FLOOR
            category = "extreme"
            warnings.append(
                f"Wind speed {wind} m/s exceeds {_WIND_THRESHOLD_HIGH} m/s. "
                f"CE significantly degraded. Consider operational shutdown."
            )

        if wind > Decimal("10") and wind <= _WIND_THRESHOLD_HIGH:
            warnings.append(
                f"Wind speed {wind} m/s is high. "
                f"Monitor flare stability closely."
            )

        self._record_provenance(
            "adjust_wind", "ce_wind",
            {"wind_speed": str(wind), "factor": str(factor),
             "category": category},
        )

        return {
            "wind_speed_m_s": wind,
            "wind_factor": factor,
            "category": category,
            "warnings": warnings,
        }

    def adjust_for_velocity(
        self,
        tip_velocity_mach: Decimal,
    ) -> Dict[str, Any]:
        """Calculate the tip velocity adjustment factor for CE.

        Velocity effect model (Mach-based):
        - Mach < 0.2: stability concern, -1% CE (low velocity)
        - Mach 0.2-0.5: optimal range, no penalty
        - Mach > 0.5: flame lift-off risk, -2% per 0.1 Mach above 0.5

        Excessively high tip velocities cause the flame to lift off
        the flare tip, allowing uncombusted gas to escape.

        Args:
            tip_velocity_mach: Flare tip exit velocity as Mach number.

        Returns:
            Dictionary with:
                - tip_velocity_mach (Decimal)
                - velocity_factor (Decimal)
                - category (str): "low", "optimal", "high", "excessive"
                - warnings (List[str])

        Example:
            >>> result = ce_engine.adjust_for_velocity(Decimal("0.3"))
            >>> result["velocity_factor"]
            Decimal('1.00000000')
        """
        mach = Decimal(str(tip_velocity_mach))
        warnings: List[str] = []

        if mach < Decimal("0"):
            raise ValueError(
                f"Tip velocity Mach number cannot be negative: {mach}"
            )

        if mach < _MACH_OPTIMAL_LOW:
            # Low velocity: stability concern
            penalty = _MACH_LOW_PENALTY
            factor = self._quantize(Decimal("1.0") - penalty)
            category = "low"
            warnings.append(
                f"Tip velocity Mach {mach} is below optimal range "
                f"({_MACH_OPTIMAL_LOW}-{_MACH_OPTIMAL_HIGH}). "
                f"Flame stability may be compromised."
            )
        elif mach <= _MACH_OPTIMAL_HIGH:
            # Optimal range: no penalty
            factor = Decimal("1.0")
            category = "optimal"
        else:
            # High velocity: flame lift-off risk
            excess_mach = mach - _MACH_OPTIMAL_HIGH
            # Penalty: 2% per 0.1 Mach above optimal
            penalty = self._quantize(
                excess_mach / Decimal("0.1") * _MACH_HIGH_PENALTY_PER_01
            )
            # Cap the penalty
            if penalty > _MACH_PENALTY_CAP:
                penalty = _MACH_PENALTY_CAP
            factor = self._quantize(Decimal("1.0") - penalty)
            # Ensure factor does not go below a reasonable floor
            if factor < Decimal("0.80"):
                factor = Decimal("0.80")

            if mach > Decimal("1.0"):
                category = "excessive"
                warnings.append(
                    f"Tip velocity Mach {mach} exceeds sonic. "
                    f"Severe flame lift-off and noise issues."
                )
            else:
                category = "high"
                warnings.append(
                    f"Tip velocity Mach {mach} exceeds optimal range. "
                    f"Risk of flame lift-off."
                )

        self._record_provenance(
            "adjust_velocity", "ce_velocity",
            {"mach": str(mach), "factor": str(factor),
             "category": category},
        )

        return {
            "tip_velocity_mach": mach,
            "velocity_factor": factor,
            "category": category,
            "warnings": warnings,
        }

    def check_lhv_stability(
        self,
        lhv_btu_scf: Decimal,
    ) -> Dict[str, Any]:
        """Check gas LHV against stability thresholds and compute
        the LHV adjustment factor.

        LHV stability model:
        - LHV >= 200 BTU/scf: stable combustion, factor = 1.0
        - 100 <= LHV < 200: CE drops by 5%
        - LHV < 100: flare may not sustain flame, CE drops by 15%

        Low heating value gases require supplemental fuel or assist
        medium to maintain stable combustion.

        Args:
            lhv_btu_scf: Lower heating value in BTU per standard
                cubic foot.

        Returns:
            Dictionary with:
                - lhv_btu_scf (Decimal)
                - lhv_factor (Decimal)
                - is_stable (bool)
                - category (str): "stable", "marginal", "unstable"
                - warnings (List[str])

        Example:
            >>> result = ce_engine.check_lhv_stability(Decimal("150"))
            >>> result["lhv_factor"]
            Decimal('0.95000000')
        """
        lhv = Decimal(str(lhv_btu_scf))
        warnings: List[str] = []

        if lhv < Decimal("0"):
            raise ValueError(
                f"LHV cannot be negative: {lhv_btu_scf}"
            )

        if lhv >= _LHV_STABLE_THRESHOLD:
            # Stable combustion
            factor = Decimal("1.0")
            is_stable = True
            category = "stable"
        elif lhv >= _LHV_CRITICAL_THRESHOLD:
            # Marginal: CE drops by 5%
            factor = self._quantize(
                Decimal("1.0") - _LHV_BELOW_STABLE_PENALTY
            )
            is_stable = True
            category = "marginal"
            warnings.append(
                f"Gas LHV {lhv} BTU/scf is below stable threshold "
                f"({_LHV_STABLE_THRESHOLD} BTU/scf). Consider "
                f"supplemental fuel or steam/air assist."
            )
        else:
            # Unstable: CE drops by 15%
            factor = self._quantize(
                Decimal("1.0") - _LHV_BELOW_CRITICAL_PENALTY
            )
            is_stable = False
            category = "unstable"
            warnings.append(
                f"Gas LHV {lhv} BTU/scf is below critical threshold "
                f"({_LHV_CRITICAL_THRESHOLD} BTU/scf). Flare may not "
                f"sustain combustion. Supplemental fuel required."
            )

        self._record_provenance(
            "check_lhv_stability", "ce_lhv",
            {"lhv": str(lhv), "factor": str(factor),
             "category": category, "stable": is_stable},
        )

        return {
            "lhv_btu_scf": lhv,
            "lhv_factor": factor,
            "is_stable": is_stable,
            "category": category,
            "warnings": warnings,
        }

    def calculate_dre(
        self,
        flare_type: Optional[str] = None,
        mass_in_kg: Optional[Decimal] = None,
        mass_out_kg: Optional[Decimal] = None,
        has_post_combustion: bool = False,
    ) -> Dict[str, Any]:
        """Calculate or look up Destruction and Removal Efficiency.

        DRE = 1 - (mass_out / mass_in) for specific compounds.

        If mass_in and mass_out are provided, DRE is calculated from
        actual measurements. Otherwise, a default DRE is returned
        based on flare type.

        Typical DRE values:
        - 98% (general elevated flares)
        - 99.5% (enclosed flares)
        - 99.9% (with post-combustion equipment)

        Args:
            flare_type: Flare type for default DRE lookup.
            mass_in_kg: Mass of compound entering the flare (kg).
            mass_out_kg: Mass of compound in flare exhaust (kg).
            has_post_combustion: Whether post-combustion equipment
                is installed.

        Returns:
            Dictionary with:
                - dre (Decimal): DRE as fraction (0-1)
                - dre_percent (Decimal): DRE as percentage
                - source (str): "measured" or "default"
                - flare_type (str or None)
                - warnings (List[str])

        Example:
            >>> result = ce_engine.calculate_dre(
            ...     flare_type="ENCLOSED_GROUND",
            ... )
            >>> result["dre"]
            Decimal('0.99500000')
        """
        warnings: List[str] = []

        if mass_in_kg is not None and mass_out_kg is not None:
            # Measured DRE
            m_in = Decimal(str(mass_in_kg))
            m_out = Decimal(str(mass_out_kg))

            if m_in <= Decimal("0"):
                raise ValueError(
                    f"Mass in must be positive: {mass_in_kg}"
                )
            if m_out < Decimal("0"):
                raise ValueError(
                    f"Mass out cannot be negative: {mass_out_kg}"
                )
            if m_out > m_in:
                warnings.append(
                    f"Mass out ({m_out}) exceeds mass in ({m_in}). "
                    f"Check measurement data."
                )
                m_out = m_in  # clamp

            dre = self._quantize(
                Decimal("1") - (m_out / m_in)
            )
            source = "measured"
        else:
            # Default DRE
            if has_post_combustion:
                dre_key = "WITH_POST_COMB"
            elif flare_type:
                dre_key = flare_type.upper()
            else:
                dre_key = "GENERAL"

            dre = _DEFAULT_DRE.get(dre_key, _DEFAULT_DRE["GENERAL"])
            source = "default"

        dre_percent = self._quantize(dre * Decimal("100"))

        self._record_provenance(
            "calculate_dre", "ce_dre",
            {"dre": str(dre), "source": source,
             "flare_type": flare_type},
        )

        return {
            "dre": dre,
            "dre_percent": dre_percent,
            "source": source,
            "flare_type": flare_type,
            "has_post_combustion": has_post_combustion,
            "warnings": warnings,
        }

    # ==================================================================
    # PUBLIC API: Default CE Lookup
    # ==================================================================

    def get_default_ce(self, flare_type: str) -> Decimal:
        """Return the default base CE for a flare type.

        Convenience method that calls ``calculate_ce()`` internally.

        Args:
            flare_type: Flare type identifier.

        Returns:
            Default CE as Decimal.

        Raises:
            KeyError: If the flare type is unknown.
        """
        return self.calculate_ce(flare_type)

    def list_default_ce_values(self) -> Dict[str, Decimal]:
        """Return all default CE values by flare type.

        Returns:
            Dictionary mapping flare type to default CE.

        Example:
            >>> ce_engine.list_default_ce_values()
            {'CANDLESTICK': Decimal('0.95'), ...}
        """
        return dict(_DEFAULT_CE_BY_FLARE_TYPE)

    # ==================================================================
    # PUBLIC API: Batch CE Calculation
    # ==================================================================

    def calculate_batch_ce(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate effective CE for multiple scenarios in batch.

        Each scenario is a dictionary of keyword arguments for
        ``get_effective_ce()``.

        Args:
            scenarios: List of CE calculation parameter dicts.

        Returns:
            Dictionary with:
                - batch_id (str)
                - results (List[Dict])
                - min_ce (Decimal)
                - max_ce (Decimal)
                - avg_ce (Decimal)
                - processing_time_ms (float)

        Example:
            >>> batch = ce_engine.calculate_batch_ce([
            ...     {"flare_type": "ELEVATED_STEAM_ASSISTED",
            ...      "wind_speed_m_s": Decimal("5")},
            ...     {"flare_type": "CANDLESTICK",
            ...      "wind_speed_m_s": Decimal("12")},
            ... ])
        """
        start_time = time.monotonic()
        batch_id = f"fl_ce_batch_{uuid.uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        ce_values: List[Decimal] = []

        for scenario in scenarios:
            result = self.get_effective_ce(**scenario)
            results.append(result)
            ce_values.append(result["effective_ce"])

        elapsed_ms = (time.monotonic() - start_time) * 1000

        min_ce = min(ce_values) if ce_values else Decimal("0")
        max_ce = max(ce_values) if ce_values else Decimal("0")
        avg_ce = Decimal("0")
        if ce_values:
            avg_ce = self._quantize(
                sum(ce_values) / Decimal(str(len(ce_values)))
            )

        return {
            "batch_id": batch_id,
            "results": results,
            "count": len(scenarios),
            "min_ce": min_ce,
            "max_ce": max_ce,
            "avg_ce": avg_ce,
            "processing_time_ms": elapsed_ms,
        }

    # ==================================================================
    # PUBLIC API: Sensitivity Analysis
    # ==================================================================

    def sensitivity_analysis(
        self,
        flare_type: str,
        parameter: str,
        values: List[Decimal],
    ) -> Dict[str, Any]:
        """Run a sensitivity analysis on a single CE parameter.

        Varies one parameter across the provided range while holding
        all others at their default values.

        Args:
            flare_type: Flare type for base CE.
            parameter: Parameter to vary. One of ``"wind_speed_m_s"``,
                ``"tip_velocity_mach"``, ``"lhv_btu_scf"``,
                ``"steam_to_gas_ratio"``, ``"air_excess_fraction"``.
            values: List of Decimal values to test.

        Returns:
            Dictionary with:
                - parameter (str)
                - flare_type (str)
                - data_points (List[Dict]): each with 'value' and
                  'effective_ce'
                - min_ce (Decimal)
                - max_ce (Decimal)

        Example:
            >>> result = ce_engine.sensitivity_analysis(
            ...     "ELEVATED_STEAM_ASSISTED",
            ...     "wind_speed_m_s",
            ...     [Decimal("0"), Decimal("5"), Decimal("10"),
            ...      Decimal("15"), Decimal("20")],
            ... )
        """
        valid_params = {
            "wind_speed_m_s", "tip_velocity_mach", "lhv_btu_scf",
            "steam_to_gas_ratio", "air_excess_fraction",
        }
        if parameter not in valid_params:
            raise ValueError(
                f"Invalid parameter: {parameter}. "
                f"Valid: {sorted(valid_params)}"
            )

        data_points: List[Dict[str, Any]] = []
        ce_values: List[Decimal] = []

        for val in values:
            kwargs: Dict[str, Any] = {"flare_type": flare_type}
            kwargs[parameter] = val

            result = self.get_effective_ce(**kwargs)
            data_points.append({
                "value": val,
                "effective_ce": result["effective_ce"],
                "adjustments": result["adjustments_applied"],
            })
            ce_values.append(result["effective_ce"])

        min_ce = min(ce_values) if ce_values else Decimal("0")
        max_ce = max(ce_values) if ce_values else Decimal("0")

        return {
            "parameter": parameter,
            "flare_type": flare_type,
            "data_points": data_points,
            "count": len(values),
            "min_ce": min_ce,
            "max_ce": max_ce,
            "ce_range": self._quantize(max_ce - min_ce),
        }

    # ==================================================================
    # PRIVATE: Steam Assist Adjustment
    # ==================================================================

    def _adjust_for_steam(
        self,
        steam_to_gas_ratio: Decimal,
    ) -> Dict[str, Any]:
        """Calculate the steam assist adjustment factor.

        Steam-to-gas ratio model (lb steam / lb gas):
        - < 0.2: under-steaming (smoking but CE maintained)
        - 0.3-0.5: optimal (no penalty)
        - 0.2-0.3 or 0.5-0.7: near-optimal (no penalty)
        - > 0.7: over-steaming, CE drops by 3% (quenching)

        Args:
            steam_to_gas_ratio: Mass ratio of steam to gas.

        Returns:
            Dictionary with assist_factor, category, warnings.
        """
        ratio = Decimal(str(steam_to_gas_ratio))
        warnings: List[str] = []

        if ratio < Decimal("0"):
            raise ValueError(
                f"Steam-to-gas ratio cannot be negative: {ratio}"
            )

        if ratio < _STEAM_RATIO_UNDER_THRESHOLD:
            # Under-steaming: visible smoke but CE maintained
            factor = Decimal("1.0")
            category = "under_steaming"
            warnings.append(
                f"Steam-to-gas ratio {ratio} is below {_STEAM_RATIO_UNDER_THRESHOLD}. "
                f"Visible smoke likely. CE not penalized."
            )
        elif ratio <= _STEAM_RATIO_OVER_THRESHOLD:
            # Optimal to near-optimal range
            factor = Decimal("1.0")
            if ratio < _STEAM_RATIO_OPTIMAL_LOW:
                category = "low_optimal"
            elif ratio <= _STEAM_RATIO_OPTIMAL_HIGH:
                category = "optimal"
            else:
                category = "high_optimal"
        else:
            # Over-steaming: quenching effect
            factor = self._quantize(
                Decimal("1.0") - _STEAM_OVER_PENALTY
            )
            category = "over_steaming"
            warnings.append(
                f"Steam-to-gas ratio {ratio} exceeds {_STEAM_RATIO_OVER_THRESHOLD}. "
                f"Over-steaming causes flame quenching. Reduce steam rate."
            )

        self._record_provenance(
            "adjust_steam", "ce_steam",
            {"ratio": str(ratio), "factor": str(factor),
             "category": category},
        )

        return {
            "steam_to_gas_ratio": ratio,
            "assist_factor": factor,
            "category": category,
            "warnings": warnings,
        }

    # ==================================================================
    # PRIVATE: Air Assist Adjustment
    # ==================================================================

    def _adjust_for_air(
        self,
        air_excess_fraction: Decimal,
    ) -> Dict[str, Any]:
        """Calculate the air assist adjustment factor.

        Air excess model (fraction above stoichiometric):
        - 0.20-0.50: optimal, stoichiometric + 20-50% excess
        - > 1.00: cooling effect, -2% CE
        - < 0.20: slightly lean, but no penalty

        Args:
            air_excess_fraction: Excess air fraction above
                stoichiometric requirements.

        Returns:
            Dictionary with assist_factor, category, warnings.
        """
        excess = Decimal(str(air_excess_fraction))
        warnings: List[str] = []

        if excess < Decimal("0"):
            raise ValueError(
                f"Air excess fraction cannot be negative: {excess}"
            )

        if excess <= _AIR_EXCESS_OPTIMAL_HIGH:
            # Within optimal range
            factor = Decimal("1.0")
            if excess < _AIR_EXCESS_OPTIMAL_LOW:
                category = "lean"
                warnings.append(
                    f"Excess air {excess} is below optimal "
                    f"({_AIR_EXCESS_OPTIMAL_LOW}). "
                    f"May have incomplete combustion due to O2 deficiency."
                )
            else:
                category = "optimal"
        elif excess <= _AIR_EXCESS_OVER_THRESHOLD:
            # Slightly above optimal but acceptable
            factor = Decimal("1.0")
            category = "above_optimal"
        else:
            # Excessive air: cooling effect
            factor = self._quantize(
                Decimal("1.0") - _AIR_EXCESS_OVER_PENALTY
            )
            category = "excessive"
            warnings.append(
                f"Excess air fraction {excess} exceeds "
                f"{_AIR_EXCESS_OVER_THRESHOLD}. "
                f"Cooling effect reduces combustion efficiency."
            )

        self._record_provenance(
            "adjust_air", "ce_air",
            {"excess": str(excess), "factor": str(factor),
             "category": category},
        )

        return {
            "air_excess_fraction": excess,
            "assist_factor": factor,
            "category": category,
            "warnings": warnings,
        }

    # ==================================================================
    # PRIVATE: Resolution Helpers
    # ==================================================================

    def _resolve_base_ce(
        self,
        base_ce: Optional[Decimal],
        flare_type: Optional[str],
        adjustments: List[str],
    ) -> Decimal:
        """Resolve the base combustion efficiency.

        Priority:
        1. Explicit base_ce parameter
        2. Flare type default from database
        3. Built-in default table
        4. EPA default (0.98)

        Args:
            base_ce: Explicit CE value.
            flare_type: Flare type for lookup.
            adjustments: Adjustment log list.

        Returns:
            Base CE as Decimal.
        """
        if base_ce is not None:
            ce = Decimal(str(base_ce))
            if ce < Decimal("0") or ce > Decimal("1"):
                raise ValueError(
                    f"Base CE must be 0-1, got {ce}"
                )
            adjustments.append(f"base_ce: explicit={ce}")
            return ce

        if flare_type:
            ft_key = flare_type.upper()

            # Try database
            if self._flare_db is not None:
                try:
                    specs = self._flare_db.get_flare_type_specs(ft_key)
                    ce = specs["typical_ce"]
                    adjustments.append(
                        f"base_ce: from flare_type={ft_key}, ce={ce}"
                    )
                    return ce
                except (KeyError, AttributeError):
                    pass

            # Try built-in
            if ft_key in _DEFAULT_CE_BY_FLARE_TYPE:
                ce = _DEFAULT_CE_BY_FLARE_TYPE[ft_key]
                adjustments.append(
                    f"base_ce: from built-in {ft_key}, ce={ce}"
                )
                return ce

        # EPA default
        ce = Decimal("0.98")
        adjustments.append(f"base_ce: EPA default={ce}")
        return ce

    def _resolve_mach(
        self,
        mach: Optional[Decimal],
        velocity_m_s: Optional[Decimal],
    ) -> Optional[Decimal]:
        """Resolve Mach number from explicit Mach or velocity in m/s.

        Args:
            mach: Explicit Mach number.
            velocity_m_s: Velocity in m/s (converted to Mach using
                speed of sound approximation).

        Returns:
            Mach number or None if neither is provided.
        """
        if mach is not None:
            return Decimal(str(mach))

        if velocity_m_s is not None:
            vel = Decimal(str(velocity_m_s))
            return self._quantize(vel / _SPEED_OF_SOUND_M_S)

        return None

    def _resolve_lhv(
        self,
        lhv_btu_scf: Optional[Decimal],
        gas_composition: Optional[Dict[str, Decimal]],
    ) -> Optional[Decimal]:
        """Resolve LHV from explicit value or gas composition.

        If lhv_btu_scf is provided, use it directly.
        If gas_composition is provided, calculate LHV from database.

        Args:
            lhv_btu_scf: Explicit LHV.
            gas_composition: Gas composition for LHV calculation.

        Returns:
            LHV in BTU/scf or None.
        """
        if lhv_btu_scf is not None:
            return Decimal(str(lhv_btu_scf))

        if gas_composition and self._flare_db is not None:
            try:
                return self._flare_db.calculate_lhv(
                    gas_composition, "EPA"
                )
            except (KeyError, AttributeError):
                pass

        return None

    def _get_assist_type(self, flare_type: Optional[str]) -> Optional[str]:
        """Get the assist type for a flare type.

        Args:
            flare_type: Flare type identifier.

        Returns:
            Assist type string or None.
        """
        if not flare_type:
            return None

        ft_key = flare_type.upper()

        # Try database
        if self._flare_db is not None:
            try:
                specs = self._flare_db.get_flare_type_specs(ft_key)
                return specs.get("assist_type")
            except (KeyError, AttributeError):
                pass

        # Fallback mapping
        _ASSIST_TYPES: Dict[str, str] = {
            "ELEVATED_STEAM_ASSISTED": "STEAM",
            "ELEVATED_AIR_ASSISTED":   "AIR",
            "ELEVATED_UNASSISTED":     "NONE",
            "ENCLOSED_GROUND":         "ENCLOSED",
            "MULTI_POINT_GROUND":      "STAGED",
            "OFFSHORE_MARINE":         "NONE",
            "CANDLESTICK":             "NONE",
            "LOW_PRESSURE":            "NONE",
        }
        return _ASSIST_TYPES.get(ft_key)

    # ==================================================================
    # PRIVATE: Utility Methods
    # ==================================================================

    def _quantize(self, value: Decimal) -> Decimal:
        """Round a Decimal to the configured precision.

        Args:
            value: Raw Decimal value.

        Returns:
            Rounded Decimal.
        """
        try:
            return value.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        except InvalidOperation:
            logger.warning("Failed to quantize value: %s", value)
            return value

    def _compute_provenance_hash(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            data: Dictionary of calculation identifiers and results.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action name.
            entity_id: Entity identifier.
            data: Optional data dictionary.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="combustion_efficiency",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"CombustionEfficiencyEngine("
            f"precision={self._precision_places}, "
            f"min_floor={self._minimum_ce_floor}, "
            f"wind={self._enable_wind}, "
            f"velocity={self._enable_velocity}, "
            f"lhv={self._enable_lhv}, "
            f"assist={self._enable_assist}, "
            f"db={'yes' if self._flare_db else 'no'})"
        )
