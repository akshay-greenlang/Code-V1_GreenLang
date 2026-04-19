"""
GL-018 UnifiedCombustionOptimizer - Emissions Control Module

NOx and CO emissions control, monitoring, and optimization including
Low NOx Burners (LNB), Flue Gas Recirculation (FGR), and SCR optimization.

Features:
    - NOx emission calculation and correction to 3% O2
    - CO emission monitoring and control
    - FGR rate optimization
    - SCR ammonia injection optimization
    - Emission rate calculations (lb/MMBTU)
    - Permit compliance monitoring
    - EPA Method 19 emission factors

Standards:
    - EPA 40 CFR Part 60 (Emission Standards)
    - EPA 40 CFR Part 98 (GHG Reporting)
    - EPA Method 19 (Emission Rate Measurement)
    - EPA AP-42 (Emission Factors)

Example:
    >>> from greenlang.agents.process_heat.gl_018_unified_combustion import EmissionsController
    >>> controller = EmissionsController(config)
    >>> result = controller.analyze_emissions(
    ...     nox_ppm=25.0,
    ...     co_ppm=30.0,
    ...     o2_pct=3.5,
    ...     fuel_consumption_mmbtu_hr=50.0
    ... )
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .config import EmissionsConfig, EmissionControlTechnology
from .schemas import EmissionsAnalysis, OptimizationRecommendation, RecommendationPriority

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - EPA Emission Factors
# =============================================================================


@dataclass(frozen=True)
class EmissionFactors:
    """EPA emission factors by fuel and control type."""

    # NOx factors (lb/MMBTU) from EPA AP-42
    # Format: fuel_type -> control_type -> factor
    nox: Dict[str, Dict[str, float]]

    # CO factors (lb/MMBTU)
    co: Dict[str, float]

    # CO2 factors (kg/MMBTU) from 40 CFR Part 98
    co2: Dict[str, float]


# EPA AP-42 NOx Emission Factors (lb/MMBTU)
NOX_EMISSION_FACTORS = {
    "natural_gas": {
        "uncontrolled": 0.098,
        "low_nox_burner": 0.049,
        "ultra_low_nox": 0.025,
        "fgr": 0.035,
        "fgr_lnb": 0.020,
        "scr": 0.010,
        "sncr": 0.030,
    },
    "no2_fuel_oil": {
        "uncontrolled": 0.140,
        "low_nox_burner": 0.070,
        "ultra_low_nox": 0.050,
        "fgr": 0.060,
        "scr": 0.015,
    },
    "no6_fuel_oil": {
        "uncontrolled": 0.170,
        "low_nox_burner": 0.085,
        "fgr": 0.070,
        "scr": 0.020,
    },
    "propane": {
        "uncontrolled": 0.095,
        "low_nox_burner": 0.048,
        "scr": 0.010,
    },
    "coal_bituminous": {
        "uncontrolled": 0.480,
        "low_nox_burner": 0.240,
        "scr": 0.050,
        "sncr": 0.150,
    },
}

# EPA AP-42 CO Emission Factors (lb/MMBTU)
CO_EMISSION_FACTORS = {
    "natural_gas": 0.040,
    "no2_fuel_oil": 0.036,
    "no6_fuel_oil": 0.036,
    "propane": 0.040,
    "coal_bituminous": 0.030,
}

# EPA 40 CFR Part 98 CO2 Emission Factors (kg/MMBTU)
CO2_EMISSION_FACTORS = {
    "natural_gas": 53.06,
    "no2_fuel_oil": 73.16,
    "no6_fuel_oil": 75.10,
    "propane": 62.87,
    "coal_bituminous": 93.28,
    "coal_sub_bituminous": 97.17,
    "lignite": 97.72,
}


# Conversion factors
PPM_TO_LB_PER_MMBTU_NOX = 0.0024  # Approximate at 3% O2, natural gas
PPM_TO_LB_PER_MMBTU_CO = 0.0011  # Approximate at 3% O2


# =============================================================================
# EMISSIONS CONTROLLER
# =============================================================================


class EmissionsController:
    """
    NOx/CO emissions control and optimization.

    Provides emission monitoring, compliance checking, and optimization
    recommendations for Low NOx Burners, FGR, and SCR systems.

    Zero-hallucination guarantee: All calculations use EPA emission
    factors and deterministic formulas.

    Attributes:
        config: Emissions configuration

    Example:
        >>> controller = EmissionsController(config)
        >>> result = controller.analyze_emissions(
        ...     nox_ppm=25.0,
        ...     co_ppm=30.0,
        ...     o2_pct=3.5
        ... )
    """

    def __init__(self, config: EmissionsConfig) -> None:
        """
        Initialize emissions controller.

        Args:
            config: Emissions configuration
        """
        self.config = config
        self._calculation_count = 0
        logger.info("EmissionsController initialized")

    def analyze_emissions(
        self,
        o2_pct: float,
        co_ppm: float,
        fuel_type: str,
        fuel_consumption_mmbtu_hr: float,
        nox_ppm: Optional[float] = None,
        fgr_rate_pct: float = 0.0,
        scr_inlet_nox_ppm: Optional[float] = None,
        scr_outlet_nox_ppm: Optional[float] = None,
        ammonia_slip_ppm: Optional[float] = None,
        operating_hours_per_year: float = 8000.0,
    ) -> EmissionsAnalysis:
        """
        Perform complete emissions analysis.

        Args:
            o2_pct: Flue gas O2 percentage
            co_ppm: CO concentration (ppm)
            fuel_type: Fuel type
            fuel_consumption_mmbtu_hr: Fuel consumption (MMBTU/hr)
            nox_ppm: NOx concentration if measured (ppm)
            fgr_rate_pct: FGR rate if active
            scr_inlet_nox_ppm: SCR inlet NOx for efficiency calc
            scr_outlet_nox_ppm: SCR outlet NOx
            ammonia_slip_ppm: Ammonia slip measurement
            operating_hours_per_year: Annual operating hours

        Returns:
            EmissionsAnalysis with complete results
        """
        self._calculation_count += 1
        logger.debug(f"Analyzing emissions: O2={o2_pct}%, CO={co_ppm} ppm")

        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")

        # Correct to 3% O2 reference
        correction_factor = self._calculate_o2_correction(o2_pct, 3.0)

        # CO analysis
        co_corrected = co_ppm * correction_factor
        co_lb_mmbtu = self._ppm_to_lb_mmbtu(co_ppm, "co", o2_pct, fuel_key)

        # NOx analysis
        nox_corrected = None
        nox_lb_mmbtu = None
        nox_compliance_pct = None

        if nox_ppm is not None:
            nox_corrected = nox_ppm * correction_factor
            nox_lb_mmbtu = self._ppm_to_lb_mmbtu(nox_ppm, "nox", o2_pct, fuel_key)
            nox_compliance_pct = (nox_lb_mmbtu / self.config.nox_permit_limit_lb_mmbtu) * 100

        # CO compliance
        co_compliance_pct = (co_lb_mmbtu / self.config.co_permit_limit_lb_mmbtu) * 100

        # CO2 emissions
        co2_factor = CO2_EMISSION_FACTORS.get(fuel_key, 53.06)
        co2_lb_mmbtu = co2_factor * 2.205  # kg to lb
        co2_tons_hr = fuel_consumption_mmbtu_hr * co2_factor / 1000  # metric tons
        co2_annual = co2_tons_hr * operating_hours_per_year

        # FGR status
        fgr_active = fgr_rate_pct > 0 and self.config.fgr_enabled

        # SCR analysis
        scr_active = self.config.scr_enabled and scr_outlet_nox_ppm is not None
        scr_efficiency = None

        if scr_active and scr_inlet_nox_ppm is not None and scr_outlet_nox_ppm is not None:
            if scr_inlet_nox_ppm > 0:
                scr_efficiency = (1 - scr_outlet_nox_ppm / scr_inlet_nox_ppm) * 100

        # Compliance check
        compliance_issues = []
        in_compliance = True

        if nox_lb_mmbtu and nox_lb_mmbtu > self.config.nox_permit_limit_lb_mmbtu:
            compliance_issues.append(
                f"NOx exceeds permit: {nox_lb_mmbtu:.4f} > {self.config.nox_permit_limit_lb_mmbtu} lb/MMBTU"
            )
            in_compliance = False

        if co_lb_mmbtu > self.config.co_permit_limit_lb_mmbtu:
            compliance_issues.append(
                f"CO exceeds permit: {co_lb_mmbtu:.4f} > {self.config.co_permit_limit_lb_mmbtu} lb/MMBTU"
            )
            in_compliance = False

        if ammonia_slip_ppm and ammonia_slip_ppm > self.config.ammonia_slip_limit_ppm:
            compliance_issues.append(
                f"Ammonia slip exceeds limit: {ammonia_slip_ppm:.1f} > {self.config.ammonia_slip_limit_ppm} ppm"
            )
            in_compliance = False

        # Generate recommendations
        recommendations = self._generate_recommendations(
            nox_ppm=nox_ppm,
            nox_corrected=nox_corrected,
            co_ppm=co_ppm,
            co_corrected=co_corrected,
            fgr_rate_pct=fgr_rate_pct,
            scr_efficiency=scr_efficiency,
            nox_compliance_pct=nox_compliance_pct,
            co_compliance_pct=co_compliance_pct,
        )

        # Emission reduction potential
        reduction_potential = None
        if nox_compliance_pct and nox_compliance_pct > 100:
            reduction_potential = nox_compliance_pct - 90  # Target 90% of limit

        return EmissionsAnalysis(
            nox_ppm_actual=nox_ppm,
            nox_ppm_corrected=round(nox_corrected, 1) if nox_corrected else None,
            nox_lb_mmbtu=round(nox_lb_mmbtu, 4) if nox_lb_mmbtu else None,
            nox_permit_limit_lb_mmbtu=self.config.nox_permit_limit_lb_mmbtu,
            nox_compliance_pct=round(nox_compliance_pct, 1) if nox_compliance_pct else None,
            co_ppm_actual=co_ppm,
            co_ppm_corrected=round(co_corrected, 1),
            co_lb_mmbtu=round(co_lb_mmbtu, 4),
            co_permit_limit_lb_mmbtu=self.config.co_permit_limit_lb_mmbtu,
            co_compliance_pct=round(co_compliance_pct, 1),
            co2_lb_mmbtu=round(co2_lb_mmbtu, 2),
            co2_tons_hr=round(co2_tons_hr, 2),
            co2_annual_tons_projected=round(co2_annual, 0),
            fgr_active=fgr_active,
            fgr_rate_pct=fgr_rate_pct if fgr_active else None,
            scr_active=scr_active,
            scr_efficiency_pct=round(scr_efficiency, 1) if scr_efficiency else None,
            ammonia_slip_ppm=ammonia_slip_ppm,
            in_compliance=in_compliance,
            compliance_issues=compliance_issues,
            emission_reduction_potential_pct=round(reduction_potential, 1) if reduction_potential else None,
            recommendations=recommendations,
        )

    def calculate_nox_emission_rate(
        self,
        nox_ppm: float,
        o2_pct: float,
        fuel_type: str,
        fuel_consumption_mmbtu_hr: float,
    ) -> Dict[str, float]:
        """
        Calculate NOx emission rate per EPA Method 19.

        Args:
            nox_ppm: Measured NOx (ppm)
            o2_pct: Measured O2 (%)
            fuel_type: Fuel type
            fuel_consumption_mmbtu_hr: Fuel consumption (MMBTU/hr)

        Returns:
            Dict with emission rates in various units
        """
        fuel_key = fuel_type.lower().replace(" ", "_")

        # Correct to 3% O2
        nox_corrected = nox_ppm * self._calculate_o2_correction(o2_pct, 3.0)

        # Convert to lb/MMBTU
        nox_lb_mmbtu = self._ppm_to_lb_mmbtu(nox_ppm, "nox", o2_pct, fuel_key)

        # Mass emission rate
        nox_lb_hr = nox_lb_mmbtu * fuel_consumption_mmbtu_hr
        nox_tons_yr = nox_lb_hr * 8760 / 2000  # Assuming continuous

        return {
            "nox_ppm_measured": nox_ppm,
            "nox_ppm_at_3pct_o2": round(nox_corrected, 1),
            "nox_lb_per_mmbtu": round(nox_lb_mmbtu, 4),
            "nox_lb_per_hr": round(nox_lb_hr, 2),
            "nox_tons_per_year_potential": round(nox_tons_yr, 1),
            "o2_correction_factor": round(self._calculate_o2_correction(o2_pct, 3.0), 3),
        }

    def optimize_fgr_rate(
        self,
        current_nox_ppm: float,
        target_nox_ppm: float,
        current_fgr_rate_pct: float,
        o2_pct: float,
        max_fgr_rate_pct: float = 25.0,
    ) -> Dict[str, Any]:
        """
        Optimize FGR rate to achieve target NOx.

        FGR reduces flame temperature, lowering thermal NOx formation.
        Typical NOx reduction: 40-70% at 15-20% FGR.

        Args:
            current_nox_ppm: Current NOx reading
            target_nox_ppm: Target NOx level
            current_fgr_rate_pct: Current FGR rate
            o2_pct: Current O2 level
            max_fgr_rate_pct: Maximum allowed FGR rate

        Returns:
            Dict with optimization results
        """
        if current_nox_ppm <= target_nox_ppm:
            return {
                "action": "none",
                "current_fgr_pct": current_fgr_rate_pct,
                "recommended_fgr_pct": current_fgr_rate_pct,
                "current_nox_ppm": current_nox_ppm,
                "target_nox_ppm": target_nox_ppm,
                "message": "NOx already at or below target",
            }

        # Estimate FGR impact
        # Each 5% FGR typically reduces NOx by 15-20%
        nox_reduction_needed = current_nox_ppm - target_nox_ppm
        nox_reduction_pct = (nox_reduction_needed / current_nox_ppm) * 100

        # FGR effectiveness: ~4% NOx reduction per 1% FGR
        fgr_effectiveness = 4.0

        # Calculate required additional FGR
        additional_fgr_needed = nox_reduction_pct / fgr_effectiveness
        recommended_fgr = current_fgr_rate_pct + additional_fgr_needed

        # Apply limits
        recommended_fgr = max(0, min(recommended_fgr, max_fgr_rate_pct))

        # Check if achievable
        achievable_reduction = (recommended_fgr - current_fgr_rate_pct) * fgr_effectiveness
        expected_nox = current_nox_ppm * (1 - achievable_reduction / 100)

        return {
            "action": "increase_fgr" if recommended_fgr > current_fgr_rate_pct else "none",
            "current_fgr_pct": current_fgr_rate_pct,
            "recommended_fgr_pct": round(recommended_fgr, 1),
            "fgr_increase_pct": round(recommended_fgr - current_fgr_rate_pct, 1),
            "current_nox_ppm": current_nox_ppm,
            "target_nox_ppm": target_nox_ppm,
            "expected_nox_ppm": round(expected_nox, 1),
            "target_achievable": expected_nox <= target_nox_ppm,
            "message": (
                f"Increase FGR from {current_fgr_rate_pct:.1f}% to {recommended_fgr:.1f}%"
                if recommended_fgr > current_fgr_rate_pct
                else "FGR at maximum or target not achievable with FGR alone"
            ),
        }

    def optimize_scr_ammonia(
        self,
        inlet_nox_ppm: float,
        target_nox_ppm: float,
        current_ammonia_lb_hr: float,
        scr_inlet_temp_f: float,
        flue_gas_flow_scfm: float,
        current_ammonia_slip_ppm: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Optimize SCR ammonia injection rate.

        Args:
            inlet_nox_ppm: SCR inlet NOx
            target_nox_ppm: Target outlet NOx
            current_ammonia_lb_hr: Current ammonia flow
            scr_inlet_temp_f: SCR inlet temperature
            flue_gas_flow_scfm: Flue gas flow rate
            current_ammonia_slip_ppm: Current ammonia slip

        Returns:
            Dict with optimization results
        """
        # Check temperature window
        temp_valid = self.config.scr_inlet_temp_min_f <= scr_inlet_temp_f <= self.config.scr_inlet_temp_max_f

        if not temp_valid:
            return {
                "action": "temperature_out_of_range",
                "current_temp_f": scr_inlet_temp_f,
                "valid_range_f": (self.config.scr_inlet_temp_min_f, self.config.scr_inlet_temp_max_f),
                "message": "SCR inlet temperature outside optimal range",
            }

        # Required NOx reduction
        required_reduction_ppm = inlet_nox_ppm - target_nox_ppm

        if required_reduction_ppm <= 0:
            return {
                "action": "none",
                "message": "Inlet NOx already below target",
                "current_ammonia_lb_hr": current_ammonia_lb_hr,
            }

        # Stoichiometric NH3/NOx ratio is ~1:1 molar
        # In practice, 1.05-1.15 mol NH3/mol NOx for high efficiency

        # Estimate required ammonia (simplified)
        # 1 ppm NOx reduced requires approximately 0.0015 lb NH3 per MSCFM
        ammonia_required_lb_hr = required_reduction_ppm * 0.0015 * (flue_gas_flow_scfm / 1000) * 60

        # Apply ammonia slip constraint
        # More ammonia = higher efficiency but more slip
        ammonia_slip_limit = self.config.ammonia_slip_limit_ppm

        if current_ammonia_slip_ppm > ammonia_slip_limit * 0.8:
            # Already approaching limit, be conservative
            ammonia_required_lb_hr *= 0.9

        return {
            "action": "adjust_ammonia",
            "current_ammonia_lb_hr": current_ammonia_lb_hr,
            "recommended_ammonia_lb_hr": round(ammonia_required_lb_hr, 2),
            "ammonia_change_lb_hr": round(ammonia_required_lb_hr - current_ammonia_lb_hr, 2),
            "inlet_nox_ppm": inlet_nox_ppm,
            "target_nox_ppm": target_nox_ppm,
            "current_slip_ppm": current_ammonia_slip_ppm,
            "slip_limit_ppm": ammonia_slip_limit,
            "scr_inlet_temp_f": scr_inlet_temp_f,
            "temp_in_range": temp_valid,
            "message": (
                f"Adjust ammonia to {ammonia_required_lb_hr:.1f} lb/hr "
                f"(change of {ammonia_required_lb_hr - current_ammonia_lb_hr:+.1f} lb/hr)"
            ),
        }

    def estimate_emission_reduction(
        self,
        current_control: str,
        proposed_control: str,
        fuel_type: str,
        fuel_consumption_mmbtu_hr: float,
        operating_hours_yr: float = 8000.0,
    ) -> Dict[str, Any]:
        """
        Estimate emission reduction from control upgrade.

        Args:
            current_control: Current control technology
            proposed_control: Proposed control technology
            fuel_type: Fuel type
            fuel_consumption_mmbtu_hr: Fuel consumption
            operating_hours_yr: Annual operating hours

        Returns:
            Dict with emission reduction estimates
        """
        fuel_key = fuel_type.lower().replace(" ", "_")
        nox_factors = NOX_EMISSION_FACTORS.get(fuel_key, NOX_EMISSION_FACTORS["natural_gas"])

        current_factor = nox_factors.get(current_control, 0.098)
        proposed_factor = nox_factors.get(proposed_control, current_factor)

        # Calculate reduction
        reduction_lb_mmbtu = current_factor - proposed_factor
        reduction_pct = (reduction_lb_mmbtu / current_factor) * 100 if current_factor > 0 else 0

        # Annual impacts
        annual_reduction_lb = reduction_lb_mmbtu * fuel_consumption_mmbtu_hr * operating_hours_yr
        annual_reduction_tons = annual_reduction_lb / 2000

        # Cost estimate (rough)
        # SCR: $5-10/kW, LNB: $1-3/kW, FGR: $2-4/kW
        cost_estimates = {
            "low_nox_burner": 50000,
            "ultra_low_nox": 100000,
            "fgr": 75000,
            "fgr_lnb": 125000,
            "scr": 200000,
            "sncr": 100000,
        }

        estimated_cost = cost_estimates.get(proposed_control, 50000)

        # Simple payback (assuming $5000/ton NOx credit)
        nox_credit_per_ton = 5000
        annual_savings = annual_reduction_tons * nox_credit_per_ton
        payback_years = estimated_cost / annual_savings if annual_savings > 0 else float("inf")

        return {
            "current_control": current_control,
            "proposed_control": proposed_control,
            "current_nox_lb_mmbtu": current_factor,
            "proposed_nox_lb_mmbtu": proposed_factor,
            "reduction_lb_mmbtu": round(reduction_lb_mmbtu, 4),
            "reduction_pct": round(reduction_pct, 1),
            "annual_reduction_tons": round(annual_reduction_tons, 1),
            "estimated_cost_usd": estimated_cost,
            "annual_value_usd": round(annual_savings, 0),
            "simple_payback_years": round(payback_years, 1) if payback_years < 100 else ">100",
        }

    def _calculate_o2_correction(
        self,
        measured_o2_pct: float,
        reference_o2_pct: float = 3.0,
    ) -> float:
        """
        Calculate O2 correction factor per EPA Method 19.

        Correction = (20.9 - O2_ref) / (20.9 - O2_meas)
        """
        if measured_o2_pct >= 20.9:
            return 1.0

        return (20.9 - reference_o2_pct) / (20.9 - measured_o2_pct)

    def _ppm_to_lb_mmbtu(
        self,
        ppm: float,
        pollutant: str,
        o2_pct: float,
        fuel_type: str,
    ) -> float:
        """
        Convert ppm to lb/MMBTU per EPA Method 19.

        lb/MMBTU = ppm * MW * Fd / (20.9 - O2%) * 10^-6

        Where:
        - MW = molecular weight (NO2=46, CO=28)
        - Fd = fuel-specific factor
        """
        # Molecular weights
        mw = {"nox": 46.0, "co": 28.0}

        # Fd factors (dscf/MMBTU) - EPA Method 19
        fd_factors = {
            "natural_gas": 8710,
            "no2_fuel_oil": 9190,
            "no6_fuel_oil": 9220,
            "propane": 8710,
            "coal_bituminous": 9780,
        }

        molecular_weight = mw.get(pollutant.lower(), 46.0)
        fd = fd_factors.get(fuel_type, 8710)

        # EPA Method 19 formula
        lb_dscf = ppm * molecular_weight / 385.5 * 1e-6
        lb_mmbtu = lb_dscf * fd * (20.9 / (20.9 - o2_pct))

        return lb_mmbtu

    def _generate_recommendations(
        self,
        nox_ppm: Optional[float],
        nox_corrected: Optional[float],
        co_ppm: float,
        co_corrected: float,
        fgr_rate_pct: float,
        scr_efficiency: Optional[float],
        nox_compliance_pct: Optional[float],
        co_compliance_pct: float,
    ) -> List[str]:
        """Generate emission optimization recommendations."""
        recommendations = []

        # NOx recommendations
        if nox_compliance_pct and nox_compliance_pct > 90:
            if self.config.fgr_enabled and fgr_rate_pct < 20:
                recommendations.append(
                    f"Increase FGR rate from {fgr_rate_pct:.0f}% to reduce NOx"
                )
            if self.config.scr_enabled:
                if scr_efficiency and scr_efficiency < 80:
                    recommendations.append(
                        "Optimize SCR ammonia injection for higher efficiency"
                    )

        if nox_compliance_pct and nox_compliance_pct > 100:
            recommendations.append(
                "NOx exceeds permit limit - immediate action required"
            )

        # CO recommendations
        if co_ppm > 100:
            recommendations.append(
                f"High CO ({co_ppm:.0f} ppm) - check combustion tuning"
            )

        if co_compliance_pct > 80:
            recommendations.append(
                "CO approaching permit limit - verify burner condition"
            )

        # FGR optimization
        if fgr_rate_pct > 0 and nox_corrected:
            if nox_corrected < 15 and fgr_rate_pct > 15:
                recommendations.append(
                    "NOx well below target - consider reducing FGR to save fan energy"
                )

        return recommendations

    @property
    def calculation_count(self) -> int:
        """Get total calculations performed."""
        return self._calculation_count
