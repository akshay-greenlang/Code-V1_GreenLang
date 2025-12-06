"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - PRV Optimization Module

This module provides Pressure Reducing Valve (PRV) sizing and optimization
per ASME B31.1 Power Piping Code. Includes Cv calculations, opening
percentage targets (50-70%), and desuperheating control.

Features:
    - PRV Cv sizing per ASME B31.1
    - Valve opening optimization (50-70% target range)
    - Critical flow detection
    - Desuperheater spray calculations
    - Performance monitoring
    - Multi-PRV coordination

ASME B31.1 Reference:
    - Chapter II: Design Conditions
    - Chapter VI: Systems

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.prv_optimization import (
    ...     PRVOptimizer,
    ... )
    >>>
    >>> optimizer = PRVOptimizer(config)
    >>> sizing = optimizer.size_prv(inlet_p=600, outlet_p=150, flow=30000)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .config import PRVConfig, DesuperheaterConfig, PRVSizingMethod, DesuperheaterType
from .schemas import PRVOperatingPoint, PRVSizingInput, PRVSizingOutput

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class PRVConstants:
    """Constants for PRV calculations per ASME B31.1."""

    # Critical pressure ratio for steam
    # P2/P1 = 0.577 for steam (k=1.3)
    CRITICAL_PRESSURE_RATIO = 0.577

    # Specific heat ratio for steam
    GAMMA_STEAM = 1.3

    # Steam sizing factor (for Cv calculation)
    # Cv = W / (63.3 * sqrt(dP * (P1 + P2) / (2 * T1)))
    STEAM_CV_FACTOR = 63.3

    # Target opening range per ASME B31.1 guidance
    TARGET_OPENING_MIN = 50.0  # %
    TARGET_OPENING_MAX = 70.0  # %

    # Margin for Cv selection
    CV_SAFETY_MARGIN = 1.15  # 15% margin

    # Valve characteristic curves
    # Equal percentage: flow = Cv_rated * R^(opening - 1)
    # where R is rangeability (typically 50:1)
    EQUAL_PERCENT_R = 50.0

    # Minimum pressure drop for PRV control
    MIN_PRESSURE_DROP_PSI = 10.0


class DesuperheaterConstants:
    """Constants for desuperheater calculations."""

    # Minimum approach to saturation (F)
    MIN_APPROACH = 20.0

    # Specific heat of water (BTU/lb-F)
    CP_WATER = 1.0

    # Spray water atomization efficiency
    SPRAY_EFFICIENCY = 0.98

    # Control deadband (F)
    CONTROL_DEADBAND = 5.0


# =============================================================================
# STEAM PROPERTY HELPERS
# =============================================================================

class PRVSteamProperties:
    """Steam property calculations for PRV sizing."""

    # Saturation data: psig -> (T_sat_F, h_g)
    SATURATION_DATA = {
        0: (212.0, 1150.5),
        15: (250.3, 1164.3),
        50: (298.0, 1178.6),
        100: (337.9, 1188.5),
        150: (365.9, 1196.0),
        200: (387.9, 1199.6),
        250: (406.1, 1201.3),
        300: (421.7, 1201.3),
        400: (448.0, 1198.6),
        500: (470.0, 1194.8),
        600: (489.0, 1189.8),
    }

    @classmethod
    def get_saturation_temp(cls, pressure_psig: float) -> float:
        """Get saturation temperature at pressure."""
        pressures = sorted(cls.SATURATION_DATA.keys())
        pressure_psig = max(0, min(600, pressure_psig))

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1, t2 = cls.SATURATION_DATA[p1][0], cls.SATURATION_DATA[p2][0]
                factor = (pressure_psig - p1) / (p2 - p1) if p2 > p1 else 0
                return t1 + factor * (t2 - t1)

        return cls.SATURATION_DATA[0][0]

    @classmethod
    def get_enthalpy(
        cls,
        pressure_psig: float,
        temperature_f: Optional[float] = None,
    ) -> float:
        """Get steam enthalpy at conditions."""
        pressures = sorted(cls.SATURATION_DATA.keys())
        pressure_psig = max(0, min(600, pressure_psig))

        # Get saturation enthalpy
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                h1, h2 = cls.SATURATION_DATA[p1][1], cls.SATURATION_DATA[p2][1]
                factor = (pressure_psig - p1) / (p2 - p1) if p2 > p1 else 0
                h_sat = h1 + factor * (h2 - h1)
                break
        else:
            h_sat = cls.SATURATION_DATA[0][1]

        # Add superheat if applicable
        if temperature_f is not None:
            t_sat = cls.get_saturation_temp(pressure_psig)
            if temperature_f > t_sat:
                superheat = temperature_f - t_sat
                h_sat += 0.48 * superheat  # Cp for superheated steam

        return h_sat


# =============================================================================
# CV CALCULATOR
# =============================================================================

class CvCalculator:
    """
    Flow coefficient (Cv) calculator for steam PRVs.

    Based on ANSI/ISA-75.01 and ASME B31.1 methods.
    """

    def __init__(self) -> None:
        """Initialize Cv calculator."""
        logger.debug("CvCalculator initialized")

    def calculate_cv_steam(
        self,
        flow_lb_hr: float,
        inlet_pressure_psig: float,
        outlet_pressure_psig: float,
        inlet_temperature_f: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate required Cv for steam flow.

        Uses the steam sizing equation:
        Cv = W / (63.3 * sqrt(dP * (P1 + P2) / (2 * T1_abs)))

        Args:
            flow_lb_hr: Steam flow rate (lb/hr)
            inlet_pressure_psig: Inlet pressure (psig)
            outlet_pressure_psig: Outlet pressure (psig)
            inlet_temperature_f: Inlet temperature (F)

        Returns:
            Tuple of (Cv_required, calculation_details)
        """
        # Convert to absolute
        p1_psia = inlet_pressure_psig + 14.696
        p2_psia = outlet_pressure_psig + 14.696

        # Pressure drop
        dp = p1_psia - p2_psia

        if dp <= 0:
            raise ValueError("Inlet pressure must be greater than outlet pressure")

        # Check for critical flow
        pressure_ratio = p2_psia / p1_psia
        is_critical = pressure_ratio < PRVConstants.CRITICAL_PRESSURE_RATIO

        # Temperature
        if inlet_temperature_f is None:
            inlet_temperature_f = PRVSteamProperties.get_saturation_temp(
                inlet_pressure_psig
            )
        t1_abs = inlet_temperature_f + 459.67  # Rankine

        # Calculate Cv using steam formula
        if is_critical:
            # Critical flow - use inlet conditions only
            # Cv = W / (C * P1 * sqrt(k / (T1 * (2/(k+1))^((k+1)/(k-1)))))
            k = PRVConstants.GAMMA_STEAM
            critical_factor = math.sqrt(
                k / (t1_abs * ((2 / (k + 1)) ** ((k + 1) / (k - 1))))
            )
            cv = flow_lb_hr / (27.3 * p1_psia * critical_factor)
        else:
            # Subcritical flow
            # Standard steam equation
            mean_pressure = (p1_psia + p2_psia) / 2
            cv = flow_lb_hr / (
                PRVConstants.STEAM_CV_FACTOR *
                math.sqrt(dp * mean_pressure / t1_abs)
            )

        details = {
            "flow_lb_hr": flow_lb_hr,
            "inlet_pressure_psia": p1_psia,
            "outlet_pressure_psia": p2_psia,
            "pressure_drop_psi": dp,
            "pressure_ratio": pressure_ratio,
            "is_critical_flow": is_critical,
            "inlet_temperature_f": inlet_temperature_f,
            "inlet_temperature_r": t1_abs,
            "calculation_method": "critical" if is_critical else "subcritical",
        }

        return cv, details

    def calculate_flow_from_cv(
        self,
        cv: float,
        inlet_pressure_psig: float,
        outlet_pressure_psig: float,
        inlet_temperature_f: Optional[float] = None,
    ) -> float:
        """
        Calculate flow rate from Cv.

        Args:
            cv: Flow coefficient
            inlet_pressure_psig: Inlet pressure (psig)
            outlet_pressure_psig: Outlet pressure (psig)
            inlet_temperature_f: Inlet temperature (F)

        Returns:
            Flow rate (lb/hr)
        """
        p1_psia = inlet_pressure_psig + 14.696
        p2_psia = outlet_pressure_psig + 14.696
        dp = p1_psia - p2_psia

        if inlet_temperature_f is None:
            inlet_temperature_f = PRVSteamProperties.get_saturation_temp(
                inlet_pressure_psig
            )
        t1_abs = inlet_temperature_f + 459.67

        pressure_ratio = p2_psia / p1_psia
        is_critical = pressure_ratio < PRVConstants.CRITICAL_PRESSURE_RATIO

        if is_critical:
            k = PRVConstants.GAMMA_STEAM
            critical_factor = math.sqrt(
                k / (t1_abs * ((2 / (k + 1)) ** ((k + 1) / (k - 1))))
            )
            flow = cv * 27.3 * p1_psia * critical_factor
        else:
            mean_pressure = (p1_psia + p2_psia) / 2
            flow = cv * PRVConstants.STEAM_CV_FACTOR * math.sqrt(
                dp * mean_pressure / t1_abs
            )

        return flow

    def calculate_opening_percentage(
        self,
        actual_cv: float,
        rated_cv: float,
        characteristic: str = "equal_percent",
    ) -> float:
        """
        Calculate valve opening percentage.

        Args:
            actual_cv: Required Cv at operating conditions
            rated_cv: Valve rated Cv (at 100% open)
            characteristic: Valve characteristic (equal_percent, linear)

        Returns:
            Opening percentage (0-100)
        """
        if rated_cv <= 0:
            return 0.0

        cv_ratio = actual_cv / rated_cv

        if cv_ratio >= 1.0:
            return 100.0
        if cv_ratio <= 0:
            return 0.0

        if characteristic == "equal_percent":
            # Equal percentage: Cv/Cv_rated = R^(opening - 1)
            # opening = 1 + log(Cv/Cv_rated) / log(R)
            r = PRVConstants.EQUAL_PERCENT_R
            opening = 1 + math.log(cv_ratio) / math.log(r)
            opening_pct = opening * 100
        else:
            # Linear characteristic
            opening_pct = cv_ratio * 100

        return max(0, min(100, opening_pct))


# =============================================================================
# DESUPERHEATER CALCULATOR
# =============================================================================

class DesuperheaterCalculator:
    """
    Calculator for desuperheater spray water requirements.

    Calculates spray water needed to reduce steam temperature
    to target superheat level.
    """

    def __init__(self) -> None:
        """Initialize desuperheater calculator."""
        logger.debug("DesuperheaterCalculator initialized")

    def calculate_spray_rate(
        self,
        steam_flow_lb_hr: float,
        inlet_temperature_f: float,
        outlet_temperature_f: float,
        inlet_pressure_psig: float,
        spray_water_temp_f: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Calculate required spray water rate.

        Energy balance:
        m_steam * h_in + m_spray * h_spray = (m_steam + m_spray) * h_out

        Args:
            steam_flow_lb_hr: Steam flow (lb/hr)
            inlet_temperature_f: Inlet steam temperature (F)
            outlet_temperature_f: Target outlet temperature (F)
            inlet_pressure_psig: Operating pressure (psig)
            spray_water_temp_f: Spray water temperature (F)

        Returns:
            Dictionary with spray calculation results
        """
        # Get saturation temperature
        t_sat = PRVSteamProperties.get_saturation_temp(inlet_pressure_psig)

        # Validate outlet temperature
        min_outlet = t_sat + DesuperheaterConstants.MIN_APPROACH
        if outlet_temperature_f < min_outlet:
            return {
                "error": (
                    f"Outlet temperature {outlet_temperature_f}F is below "
                    f"minimum {min_outlet}F (saturation + approach)"
                ),
                "minimum_outlet_temperature_f": min_outlet,
            }

        # Check if desuperheating is needed
        if inlet_temperature_f <= outlet_temperature_f:
            return {
                "spray_required": False,
                "spray_rate_lb_hr": 0,
                "reason": "Inlet temperature already at or below target",
            }

        # Calculate enthalpies
        h_inlet = PRVSteamProperties.get_enthalpy(
            inlet_pressure_psig,
            inlet_temperature_f
        )
        h_outlet = PRVSteamProperties.get_enthalpy(
            inlet_pressure_psig,
            outlet_temperature_f
        )
        h_spray = DesuperheaterConstants.CP_WATER * (spray_water_temp_f - 32)

        # Solve energy balance for spray rate
        # m_s * h_in + m_w * h_w = (m_s + m_w) * h_out
        # m_w = m_s * (h_in - h_out) / (h_out - h_w)
        denominator = h_outlet - h_spray
        if denominator <= 0:
            return {
                "error": "Cannot calculate - spray water too hot",
                "spray_required": True,
            }

        spray_rate = steam_flow_lb_hr * (h_inlet - h_outlet) / denominator

        # Apply efficiency factor
        spray_rate /= DesuperheaterConstants.SPRAY_EFFICIENCY

        # Calculate superheat reduction
        superheat_reduction = inlet_temperature_f - outlet_temperature_f

        return {
            "spray_required": True,
            "spray_rate_lb_hr": spray_rate,
            "spray_rate_pct": (spray_rate / steam_flow_lb_hr * 100),
            "inlet_temperature_f": inlet_temperature_f,
            "outlet_temperature_f": outlet_temperature_f,
            "superheat_reduction_f": superheat_reduction,
            "saturation_temperature_f": t_sat,
            "outlet_superheat_f": outlet_temperature_f - t_sat,
            "enthalpy_inlet_btu_lb": h_inlet,
            "enthalpy_outlet_btu_lb": h_outlet,
            "enthalpy_spray_btu_lb": h_spray,
        }


# =============================================================================
# PRV OPTIMIZER
# =============================================================================

class PRVOptimizer:
    """
    PRV sizing and optimization per ASME B31.1.

    Provides comprehensive PRV analysis including:
    - Cv sizing with safety margin
    - Opening percentage optimization (50-70% target)
    - Critical flow detection
    - Desuperheater integration
    - Performance monitoring

    Example:
        >>> config = PRVConfig(
        ...     prv_id="PRV-001",
        ...     inlet_pressure_psig=600,
        ...     outlet_pressure_psig=150,
        ...     design_flow_lb_hr=30000,
        ...     max_flow_lb_hr=40000,
        ...     cv_rated=150,
        ... )
        >>> optimizer = PRVOptimizer(config)
        >>> sizing = optimizer.size_prv()
    """

    def __init__(
        self,
        config: PRVConfig,
    ) -> None:
        """
        Initialize PRV optimizer.

        Args:
            config: PRV configuration
        """
        self.config = config
        self.cv_calc = CvCalculator()
        self.desuper_calc = DesuperheaterCalculator()

        logger.info(
            f"PRVOptimizer initialized for {config.prv_id}: "
            f"P1={config.inlet_pressure_psig} psig, "
            f"P2={config.outlet_pressure_psig} psig"
        )

    def size_prv(
        self,
        inlet_temperature_f: Optional[float] = None,
    ) -> PRVSizingOutput:
        """
        Size PRV for design conditions.

        Args:
            inlet_temperature_f: Inlet steam temperature (F)

        Returns:
            PRVSizingOutput with sizing results
        """
        start_time = datetime.now(timezone.utc)
        recommendations = []
        warnings = []

        # Calculate Cv at design flow
        cv_design, design_details = self.cv_calc.calculate_cv_steam(
            flow_lb_hr=self.config.design_flow_lb_hr,
            inlet_pressure_psig=self.config.inlet_pressure_psig,
            outlet_pressure_psig=self.config.outlet_pressure_psig,
            inlet_temperature_f=inlet_temperature_f,
        )

        # Calculate Cv at min and max flows
        cv_min = 0.0
        cv_max = 0.0

        if self.config.min_flow_lb_hr > 0:
            cv_min, _ = self.cv_calc.calculate_cv_steam(
                flow_lb_hr=self.config.min_flow_lb_hr,
                inlet_pressure_psig=self.config.inlet_pressure_psig,
                outlet_pressure_psig=self.config.outlet_pressure_psig,
                inlet_temperature_f=inlet_temperature_f,
            )

        cv_max, _ = self.cv_calc.calculate_cv_steam(
            flow_lb_hr=self.config.max_flow_lb_hr,
            inlet_pressure_psig=self.config.inlet_pressure_psig,
            outlet_pressure_psig=self.config.outlet_pressure_psig,
            inlet_temperature_f=inlet_temperature_f,
        )

        # Calculate recommended Cv with margin
        cv_recommended = cv_max * PRVConstants.CV_SAFETY_MARGIN

        # Check against rated Cv if provided
        cv_rated = self.config.cv_rated
        if cv_rated > 0:
            # Calculate opening percentages
            opening_design = self.cv_calc.calculate_opening_percentage(
                cv_design, cv_rated
            )
            opening_min = self.cv_calc.calculate_opening_percentage(
                cv_min, cv_rated
            )
            opening_max = self.cv_calc.calculate_opening_percentage(
                cv_max, cv_rated
            )

            # Check against targets
            target_min = self.config.target_opening_min_pct
            target_max = self.config.target_opening_max_pct

            meets_targets = (
                target_min <= opening_design <= target_max
            )

            if not meets_targets:
                if opening_design < target_min:
                    warnings.append(
                        f"Valve oversized: design opening {opening_design:.1f}% "
                        f"below target {target_min}%"
                    )
                    recommendations.append(
                        f"Consider smaller valve with Cv ~{cv_design / (target_min/100 * 50):.0f}"
                    )
                elif opening_design > target_max:
                    warnings.append(
                        f"Valve undersized: design opening {opening_design:.1f}% "
                        f"above target {target_max}%"
                    )
                    recommendations.append(
                        f"Consider larger valve with Cv ~{cv_max * 1.3:.0f}"
                    )

            # Check for excessive range
            if opening_max > 90:
                warnings.append(
                    f"Valve at {opening_max:.1f}% at max flow - limited control range"
                )

            # Calculate max flow capacity
            max_capacity = self.cv_calc.calculate_flow_from_cv(
                cv_rated,
                self.config.inlet_pressure_psig,
                self.config.outlet_pressure_psig,
                inlet_temperature_f,
            )

            # Calculate rangeability
            if cv_min > 0:
                rangeability = cv_max / cv_min
            else:
                rangeability = 50.0  # Assume standard

        else:
            # No rated Cv provided - just report requirements
            opening_design = 60.0  # Assume target
            opening_min = (cv_min / cv_recommended * 60) if cv_recommended > 0 else 0
            opening_max = (cv_max / cv_recommended * 60) if cv_recommended > 0 else 100
            meets_targets = True
            max_capacity = self.config.max_flow_lb_hr * 1.15
            rangeability = 50.0

            cv_rated = cv_recommended
            recommendations.append(
                f"Select valve with rated Cv >= {cv_recommended:.0f}"
            )

        # Opening target status
        if meets_targets:
            opening_status = (
                f"Design opening {opening_design:.1f}% within "
                f"{target_min:.0f}-{target_max:.0f}% target range"
            )
        else:
            opening_status = (
                f"Design opening {opening_design:.1f}% outside "
                f"{target_min:.0f}-{target_max:.0f}% target range"
            )

        # ASME B31.1 recommendations
        if design_details["is_critical_flow"]:
            recommendations.append(
                "Critical flow condition - verify noise/vibration ratings"
            )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(design_details)

        return PRVSizingOutput(
            prv_id=self.config.prv_id,
            timestamp=datetime.now(timezone.utc),
            cv_required=cv_design,
            cv_recommended=cv_recommended,
            cv_margin_pct=PRVConstants.CV_SAFETY_MARGIN * 100 - 100,
            opening_at_design_pct=opening_design,
            opening_at_min_pct=opening_min,
            opening_at_max_pct=opening_max,
            meets_opening_targets=meets_targets,
            opening_target_status=opening_status,
            max_flow_capacity_lb_hr=max_capacity,
            rangeability=rangeability,
            is_critical_flow=design_details["is_critical_flow"],
            critical_pressure_ratio=design_details["pressure_ratio"],
            recommendations=recommendations,
            warnings=warnings,
            provenance_hash=provenance_hash,
            formula_reference="ASME B31.1-2020, ANSI/ISA-75.01",
        )

    def analyze_operating_point(
        self,
        operating_point: PRVOperatingPoint,
    ) -> Dict[str, Any]:
        """
        Analyze PRV at current operating point.

        Args:
            operating_point: Current operating data

        Returns:
            Dictionary with operating analysis
        """
        warnings = []
        recommendations = []

        # Calculate required Cv at current conditions
        cv_current, details = self.cv_calc.calculate_cv_steam(
            flow_lb_hr=operating_point.flow_rate_lb_hr,
            inlet_pressure_psig=operating_point.inlet_pressure_psig,
            outlet_pressure_psig=operating_point.outlet_pressure_psig,
            inlet_temperature_f=operating_point.inlet_temperature_f,
        )

        # Compare to rated Cv
        if self.config.cv_rated > 0:
            expected_opening = self.cv_calc.calculate_opening_percentage(
                cv_current, self.config.cv_rated
            )
        else:
            expected_opening = operating_point.opening_pct

        # Compare to actual opening
        actual_opening = operating_point.opening_pct
        opening_deviation = actual_opening - expected_opening

        if abs(opening_deviation) > 5:
            warnings.append(
                f"Opening deviation: actual {actual_opening:.1f}% vs "
                f"expected {expected_opening:.1f}%"
            )
            if opening_deviation > 10:
                recommendations.append(
                    "Check valve trim/seat condition - possible wear"
                )
            elif opening_deviation < -10:
                recommendations.append(
                    "Check actuator/positioner calibration"
                )

        # Check opening vs targets
        target_min = self.config.target_opening_min_pct
        target_max = self.config.target_opening_max_pct

        if actual_opening < target_min:
            recommendations.append(
                f"Valve operating below target range ({actual_opening:.1f}% < {target_min}%)"
            )
        elif actual_opening > target_max:
            recommendations.append(
                f"Valve operating above target range ({actual_opening:.1f}% > {target_max}%)"
            )

        # Check pressure drop
        dp = operating_point.pressure_drop_psi
        if dp and dp < PRVConstants.MIN_PRESSURE_DROP_PSI:
            warnings.append(
                f"Low pressure drop ({dp:.1f} psi) - control may be unstable"
            )

        return {
            "cv_current": cv_current,
            "expected_opening_pct": expected_opening,
            "actual_opening_pct": actual_opening,
            "opening_deviation_pct": opening_deviation,
            "is_critical_flow": details["is_critical_flow"],
            "within_target_range": target_min <= actual_opening <= target_max,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def calculate_desuperheating(
        self,
        steam_flow_lb_hr: float,
        inlet_temperature_f: float,
        target_superheat_f: Optional[float] = None,
        spray_water_temp_f: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Calculate desuperheater requirements.

        Args:
            steam_flow_lb_hr: Steam flow (lb/hr)
            inlet_temperature_f: Inlet steam temperature (F)
            target_superheat_f: Target superheat above saturation (F)
            spray_water_temp_f: Spray water temperature (F)

        Returns:
            Dictionary with desuperheater analysis
        """
        if not self.config.desuperheater_enabled:
            return {
                "enabled": False,
                "message": "Desuperheater not configured for this PRV",
            }

        # Get saturation temperature at outlet
        t_sat = PRVSteamProperties.get_saturation_temp(
            self.config.outlet_pressure_psig
        )

        # Determine target temperature
        if target_superheat_f is None:
            target_superheat_f = self.config.target_superheat_f or 50.0

        target_temp = t_sat + target_superheat_f

        # Calculate spray requirements
        result = self.desuper_calc.calculate_spray_rate(
            steam_flow_lb_hr=steam_flow_lb_hr,
            inlet_temperature_f=inlet_temperature_f,
            outlet_temperature_f=target_temp,
            inlet_pressure_psig=self.config.outlet_pressure_psig,
            spray_water_temp_f=spray_water_temp_f,
        )

        return result

    def _calculate_provenance_hash(self, details: Dict[str, Any]) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "prv_id": self.config.prv_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inlet_pressure_psia": details["inlet_pressure_psia"],
            "outlet_pressure_psia": details["outlet_pressure_psia"],
            "flow_lb_hr": details["flow_lb_hr"],
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# MULTI-PRV COORDINATOR
# =============================================================================

class MultiPRVCoordinator:
    """
    Coordinator for multiple PRVs feeding common headers.

    Optimizes load distribution across multiple PRVs for
    efficient header pressure control.
    """

    def __init__(
        self,
        prvs: List[PRVConfig],
    ) -> None:
        """
        Initialize multi-PRV coordinator.

        Args:
            prvs: List of PRV configurations
        """
        self.prvs = {prv.prv_id: prv for prv in prvs}
        self.optimizers = {
            prv.prv_id: PRVOptimizer(prv) for prv in prvs
        }

        logger.info(f"MultiPRVCoordinator initialized with {len(prvs)} PRVs")

    def optimize_load_distribution(
        self,
        total_flow_required_lb_hr: float,
        header_pressure_psig: float,
    ) -> Dict[str, Any]:
        """
        Optimize flow distribution across PRVs.

        Args:
            total_flow_required_lb_hr: Total flow needed (lb/hr)
            header_pressure_psig: Target header pressure (psig)

        Returns:
            Dictionary with optimized distribution
        """
        allocations = []
        remaining_flow = total_flow_required_lb_hr

        # Sort PRVs by efficiency (larger opening = more efficient)
        sorted_prvs = sorted(
            self.prvs.values(),
            key=lambda p: p.design_flow_lb_hr,
            reverse=True
        )

        for prv in sorted_prvs:
            if remaining_flow <= 0:
                allocations.append({
                    "prv_id": prv.prv_id,
                    "allocated_flow_lb_hr": 0,
                    "opening_pct": 0,
                    "status": "standby",
                })
                continue

            # Calculate how much this PRV can handle
            available_capacity = prv.max_flow_lb_hr
            allocated = min(remaining_flow, available_capacity)

            # Target opening in optimal range
            if allocated > 0:
                # Calculate expected opening
                cv_calc = self.optimizers[prv.prv_id].cv_calc
                cv_needed, _ = cv_calc.calculate_cv_steam(
                    flow_lb_hr=allocated,
                    inlet_pressure_psig=prv.inlet_pressure_psig,
                    outlet_pressure_psig=header_pressure_psig,
                )
                opening = cv_calc.calculate_opening_percentage(
                    cv_needed, prv.cv_rated
                )

                status = "active"
                if opening < 50:
                    status = "low_opening"
                elif opening > 70:
                    status = "high_opening"

            else:
                opening = 0
                status = "standby"

            allocations.append({
                "prv_id": prv.prv_id,
                "allocated_flow_lb_hr": allocated,
                "opening_pct": opening,
                "status": status,
            })

            remaining_flow -= allocated

        # Check if all demand is met
        total_allocated = sum(a["allocated_flow_lb_hr"] for a in allocations)
        shortfall = total_flow_required_lb_hr - total_allocated

        return {
            "total_flow_required_lb_hr": total_flow_required_lb_hr,
            "total_allocated_lb_hr": total_allocated,
            "shortfall_lb_hr": max(0, shortfall),
            "allocations": allocations,
            "all_demand_met": shortfall <= 0,
        }
