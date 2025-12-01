# -*- coding: utf-8 -*-
"""
SteamLossCostCalculator for GL-008 TRAPCATCHER

Provides deterministic calculations for steam loss quantification,
energy waste calculation, and ROI analysis for steam trap maintenance.

Standards:
- ISO 7841: Automatic steam traps - Determination of steam loss
- ASME PTC 39: Steam Traps
- DOE Steam System Assessment Protocol

Key Features:
- Steam loss rate calculation (Napier equation)
- Energy waste quantification
- Annual cost calculation
- CO2 emissions estimation
- ROI and payback analysis
- Trap replacement economics

Zero-Hallucination Guarantee:
All calculations use deterministic engineering formulas.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Example:
    >>> calculator = SteamLossCostCalculator()
    >>> result = calculator.calculate_steam_loss_cost(
    ...     steam_loss_kg_hr=25.0, pressure_bar=10.0
    ... )
    >>> print(f"Annual cost: ${result.annual_cost_usd:,.2f}")

Author: GL-CalculatorEngineer
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CostCalculatorConfig:
    """
    Configuration for cost calculations.

    Attributes:
        steam_cost_usd_per_1000kg: Cost of steam per 1000 kg
        electricity_cost_usd_per_kwh: Electricity cost per kWh
        natural_gas_cost_usd_per_mmbtu: Natural gas cost per MMBtu
        boiler_efficiency: Boiler efficiency (0-1)
        operating_hours_per_year: Annual operating hours
        labor_rate_usd_per_hour: Maintenance labor rate
        co2_factor_kg_per_kwh: CO2 emissions factor
        steam_generation_cost_usd_per_kg: Direct steam generation cost
        condensate_return_value_percent: Value of returned condensate
    """
    steam_cost_usd_per_1000kg: float = 15.0
    electricity_cost_usd_per_kwh: float = 0.10
    natural_gas_cost_usd_per_mmbtu: float = 4.0
    boiler_efficiency: float = 0.82
    operating_hours_per_year: int = 8760
    labor_rate_usd_per_hour: float = 75.0
    co2_factor_kg_per_kwh: float = 0.4
    steam_generation_cost_usd_per_kg: float = 0.015
    condensate_return_value_percent: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "steam_cost_usd_per_1000kg": self.steam_cost_usd_per_1000kg,
            "electricity_cost_usd_per_kwh": self.electricity_cost_usd_per_kwh,
            "natural_gas_cost_usd_per_mmbtu": self.natural_gas_cost_usd_per_mmbtu,
            "boiler_efficiency": self.boiler_efficiency,
            "operating_hours_per_year": self.operating_hours_per_year,
            "labor_rate_usd_per_hour": self.labor_rate_usd_per_hour,
            "co2_factor_kg_per_kwh": self.co2_factor_kg_per_kwh,
        }


@dataclass
class SteamProperties:
    """
    Steam thermodynamic properties at given pressure.

    Used for energy calculations.
    """
    pressure_bar: float
    saturation_temp_c: float
    enthalpy_steam_kj_kg: float
    enthalpy_condensate_kj_kg: float
    enthalpy_vaporization_kj_kg: float
    specific_volume_m3_kg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pressure_bar": self.pressure_bar,
            "saturation_temp_c": self.saturation_temp_c,
            "enthalpy_steam_kj_kg": self.enthalpy_steam_kj_kg,
            "enthalpy_condensate_kj_kg": self.enthalpy_condensate_kj_kg,
            "enthalpy_vaporization_kj_kg": self.enthalpy_vaporization_kj_kg,
            "specific_volume_m3_kg": self.specific_volume_m3_kg,
        }


@dataclass
class EnergyMetrics:
    """
    Energy loss metrics from steam loss.

    Attributes:
        steam_loss_kg_hr: Steam loss rate in kg/hr
        energy_loss_kw: Energy loss rate in kW
        energy_loss_mmbtu_hr: Energy loss rate in MMBtu/hr
        fuel_consumption_m3_hr: Equivalent fuel consumption
        daily_energy_loss_kwh: Daily energy loss
        monthly_energy_loss_kwh: Monthly energy loss
        annual_energy_loss_kwh: Annual energy loss
    """
    steam_loss_kg_hr: float
    energy_loss_kw: float
    energy_loss_mmbtu_hr: float
    fuel_consumption_m3_hr: float
    daily_energy_loss_kwh: float
    monthly_energy_loss_kwh: float
    annual_energy_loss_kwh: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "steam_loss_kg_hr": self.steam_loss_kg_hr,
            "energy_loss_kw": self.energy_loss_kw,
            "energy_loss_mmbtu_hr": self.energy_loss_mmbtu_hr,
            "fuel_consumption_m3_hr": self.fuel_consumption_m3_hr,
            "daily_energy_loss_kwh": self.daily_energy_loss_kwh,
            "monthly_energy_loss_kwh": self.monthly_energy_loss_kwh,
            "annual_energy_loss_kwh": self.annual_energy_loss_kwh,
        }


@dataclass
class ROIAnalysis:
    """
    Return on Investment analysis for trap repair/replacement.

    Attributes:
        repair_cost_usd: Total repair/replacement cost
        annual_savings_usd: Annual savings from repair
        simple_payback_days: Simple payback period in days
        simple_payback_months: Simple payback period in months
        roi_percent: Return on investment percentage
        npv_5yr_usd: Net present value over 5 years
        irr_percent: Internal rate of return
        break_even_date: Estimated break-even date
    """
    repair_cost_usd: float
    annual_savings_usd: float
    simple_payback_days: float
    simple_payback_months: float
    roi_percent: float
    npv_5yr_usd: float
    irr_percent: float
    break_even_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repair_cost_usd": self.repair_cost_usd,
            "annual_savings_usd": self.annual_savings_usd,
            "simple_payback_days": self.simple_payback_days,
            "simple_payback_months": self.simple_payback_months,
            "roi_percent": self.roi_percent,
            "npv_5yr_usd": self.npv_5yr_usd,
            "irr_percent": self.irr_percent,
            "break_even_date": self.break_even_date,
        }


@dataclass
class CostAnalysisResult:
    """
    Complete cost analysis result.

    Attributes:
        trap_id: Steam trap identifier
        timestamp: Analysis timestamp
        steam_loss_kg_hr: Steam loss rate
        energy_metrics: Energy loss metrics
        hourly_cost_usd: Hourly cost of loss
        daily_cost_usd: Daily cost of loss
        monthly_cost_usd: Monthly cost of loss
        annual_cost_usd: Annual cost of loss
        co2_emissions_kg_yr: Annual CO2 emissions
        steam_properties: Steam properties at operating pressure
        roi_analysis: ROI analysis (if repair cost provided)
        provenance_hash: SHA-256 hash for audit trail
        calculation_method: Method description
    """
    trap_id: str
    timestamp: datetime
    steam_loss_kg_hr: float
    energy_metrics: EnergyMetrics
    hourly_cost_usd: float
    daily_cost_usd: float
    monthly_cost_usd: float
    annual_cost_usd: float
    co2_emissions_kg_yr: float
    steam_properties: SteamProperties
    roi_analysis: Optional[ROIAnalysis] = None
    provenance_hash: str = ""
    calculation_method: str = "deterministic"

    def __post_init__(self):
        """Generate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "steam_loss_kg_hr": self.steam_loss_kg_hr,
            "annual_cost_usd": self.annual_cost_usd,
            "co2_emissions_kg_yr": self.co2_emissions_kg_yr,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "steam_loss_kg_hr": self.steam_loss_kg_hr,
            "energy_metrics": self.energy_metrics.to_dict(),
            "hourly_cost_usd": self.hourly_cost_usd,
            "daily_cost_usd": self.daily_cost_usd,
            "monthly_cost_usd": self.monthly_cost_usd,
            "annual_cost_usd": self.annual_cost_usd,
            "co2_emissions_kg_yr": self.co2_emissions_kg_yr,
            "steam_properties": self.steam_properties.to_dict(),
            "roi_analysis": self.roi_analysis.to_dict() if self.roi_analysis else None,
            "provenance_hash": self.provenance_hash,
            "calculation_method": self.calculation_method,
        }


# ============================================================================
# STEAM LOSS COST CALCULATOR
# ============================================================================

class SteamLossCostCalculator:
    """
    Deterministic calculator for steam loss costs and ROI analysis.

    Calculates economic impact of steam trap failures including
    energy waste, operating costs, emissions, and repair payback.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic engineering formulas
    - No LLM or ML inference in calculation path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Key Formulas:
    - Steam loss: Napier equation W = C * A * P / 366
    - Energy loss: Q = W * h_fg / 3600 (kW)
    - Annual cost: W * hours * cost_per_kg
    - CO2 emissions: Q * hours * emission_factor

    Example:
        >>> calculator = SteamLossCostCalculator()
        >>> result = calculator.calculate_steam_loss_cost(
        ...     trap_id="T001",
        ...     steam_loss_kg_hr=25.0,
        ...     pressure_bar=10.0,
        ...     repair_cost_usd=250.0
        ... )
        >>> print(f"Annual savings potential: ${result.annual_cost_usd:,.2f}")
    """

    # Steam properties table: pressure (bar gauge) -> properties
    # (T_sat_C, h_f, h_fg, h_g, v_g)
    STEAM_PROPERTIES_TABLE = {
        0: (100.0, 419.0, 2257.0, 2676.0, 1.673),
        2: (133.5, 561.4, 2163.5, 2724.9, 0.606),
        4: (151.8, 640.1, 2108.0, 2748.1, 0.375),
        6: (165.0, 697.0, 2065.6, 2762.6, 0.273),
        8: (175.4, 742.6, 2030.7, 2773.3, 0.215),
        10: (184.1, 781.1, 2000.4, 2781.5, 0.177),
        12: (191.6, 814.7, 1973.0, 2787.7, 0.151),
        14: (198.3, 844.6, 1947.5, 2792.1, 0.131),
        16: (204.3, 871.8, 1923.4, 2795.2, 0.116),
        18: (209.8, 897.0, 1900.3, 2797.3, 0.104),
        20: (214.9, 920.6, 1878.0, 2798.6, 0.094),
        25: (226.0, 971.7, 1825.8, 2797.5, 0.076),
        30: (235.8, 1017.5, 1777.2, 2794.7, 0.064),
    }

    # Orifice discharge coefficients by trap type
    DISCHARGE_COEFFICIENTS = {
        "disc": 0.85,
        "inverted_bucket": 0.70,
        "float": 0.75,
        "thermostatic": 0.80,
        "piston": 0.85,
        "default": 0.75,
    }

    def __init__(self, config: Optional[CostCalculatorConfig] = None):
        """
        Initialize cost calculator.

        Args:
            config: Calculator configuration (uses defaults if not provided)
        """
        self.config = config or CostCalculatorConfig()
        self.calculation_count = 0
        logger.info(
            f"SteamLossCostCalculator initialized "
            f"(steam_cost=${self.config.steam_cost_usd_per_1000kg}/1000kg)"
        )

    def calculate_steam_loss_rate(
        self,
        orifice_diameter_mm: float,
        pressure_bar: float,
        trap_type: str = "default",
        leak_fraction: float = 1.0
    ) -> float:
        """
        Calculate steam loss rate using Napier equation.

        FORMULA (Napier equation for steam flow through orifice):
        W = C_d * A * P_abs / 366

        Where:
        - W = steam flow rate (kg/hr)
        - C_d = discharge coefficient (0.7-0.95)
        - A = orifice area (mm^2)
        - P_abs = absolute pressure (bar)
        - 366 = empirical constant for saturated steam

        ZERO-HALLUCINATION GUARANTEE:
        Deterministic calculation using established engineering formula.

        Args:
            orifice_diameter_mm: Effective orifice diameter in mm
            pressure_bar: Gauge pressure in bar
            trap_type: Type of trap for discharge coefficient
            leak_fraction: Fraction of full leak (0-1)

        Returns:
            Steam loss rate in kg/hr

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if orifice_diameter_mm <= 0:
            raise ValueError(f"Invalid orifice diameter: {orifice_diameter_mm}")
        if pressure_bar < 0:
            raise ValueError(f"Invalid pressure: {pressure_bar}")

        # Get discharge coefficient
        c_d = self.DISCHARGE_COEFFICIENTS.get(
            trap_type.lower(),
            self.DISCHARGE_COEFFICIENTS["default"]
        )

        # Calculate orifice area (mm^2)
        area_mm2 = math.pi * (orifice_diameter_mm / 2) ** 2

        # Absolute pressure (add atmospheric)
        p_abs = pressure_bar + 1.013

        # Napier equation: W = C_d * A * P_abs / 366
        steam_flow = c_d * area_mm2 * p_abs / 366.0

        # Apply leak fraction
        steam_loss = steam_flow * leak_fraction

        return round(steam_loss, 3)

    def calculate_steam_loss_cost(
        self,
        trap_id: str,
        steam_loss_kg_hr: float,
        pressure_bar: float,
        repair_cost_usd: Optional[float] = None,
    ) -> CostAnalysisResult:
        """
        Calculate comprehensive cost analysis for steam loss.

        ZERO-HALLUCINATION: Uses deterministic thermodynamic and economic formulas.

        Args:
            trap_id: Steam trap identifier
            steam_loss_kg_hr: Steam loss rate in kg/hr
            pressure_bar: Operating pressure (bar gauge)
            repair_cost_usd: Optional repair cost for ROI analysis

        Returns:
            CostAnalysisResult with complete cost analysis

        Raises:
            ValueError: If inputs are invalid
        """
        self.calculation_count += 1
        timestamp = datetime.now(timezone.utc)

        # Validate inputs
        if steam_loss_kg_hr < 0:
            raise ValueError(f"Invalid steam loss: {steam_loss_kg_hr}")
        if pressure_bar < 0 or pressure_bar > 100:
            raise ValueError(f"Invalid pressure: {pressure_bar}")

        # Get steam properties
        steam_props = self._get_steam_properties(pressure_bar)

        # Calculate energy metrics
        energy_metrics = self._calculate_energy_metrics(
            steam_loss_kg_hr, steam_props
        )

        # Calculate costs
        cost_per_kg = self.config.steam_cost_usd_per_1000kg / 1000.0
        hours_per_year = self.config.operating_hours_per_year

        hourly_cost = steam_loss_kg_hr * cost_per_kg
        daily_cost = hourly_cost * 24
        monthly_cost = daily_cost * 30
        annual_cost = steam_loss_kg_hr * hours_per_year * cost_per_kg

        # Calculate CO2 emissions
        co2_emissions = self._calculate_co2_emissions(
            energy_metrics.energy_loss_kw, hours_per_year
        )

        # Calculate ROI if repair cost provided
        roi_analysis = None
        if repair_cost_usd is not None and repair_cost_usd > 0:
            roi_analysis = self._calculate_roi(repair_cost_usd, annual_cost)

        return CostAnalysisResult(
            trap_id=trap_id,
            timestamp=timestamp,
            steam_loss_kg_hr=round(steam_loss_kg_hr, 3),
            energy_metrics=energy_metrics,
            hourly_cost_usd=round(hourly_cost, 4),
            daily_cost_usd=round(daily_cost, 2),
            monthly_cost_usd=round(monthly_cost, 2),
            annual_cost_usd=round(annual_cost, 2),
            co2_emissions_kg_yr=round(co2_emissions, 2),
            steam_properties=steam_props,
            roi_analysis=roi_analysis,
        )

    def _get_steam_properties(self, pressure_bar: float) -> SteamProperties:
        """
        Get steam properties at given pressure.

        Uses interpolation from property table.

        Args:
            pressure_bar: Gauge pressure in bar

        Returns:
            SteamProperties at given pressure
        """
        pressures = sorted(self.STEAM_PROPERTIES_TABLE.keys())

        # Handle edge cases
        if pressure_bar <= pressures[0]:
            props = self.STEAM_PROPERTIES_TABLE[pressures[0]]
        elif pressure_bar >= pressures[-1]:
            props = self.STEAM_PROPERTIES_TABLE[pressures[-1]]
        else:
            # Interpolate between two pressures
            for i in range(len(pressures) - 1):
                p_low = pressures[i]
                p_high = pressures[i + 1]

                if p_low <= pressure_bar <= p_high:
                    props_low = self.STEAM_PROPERTIES_TABLE[p_low]
                    props_high = self.STEAM_PROPERTIES_TABLE[p_high]

                    # Linear interpolation
                    f = (pressure_bar - p_low) / (p_high - p_low)
                    props = tuple(
                        low + f * (high - low)
                        for low, high in zip(props_low, props_high)
                    )
                    break
            else:
                props = self.STEAM_PROPERTIES_TABLE[pressures[0]]

        t_sat, h_f, h_fg, h_g, v_g = props

        return SteamProperties(
            pressure_bar=pressure_bar,
            saturation_temp_c=round(t_sat, 2),
            enthalpy_steam_kj_kg=round(h_g, 2),
            enthalpy_condensate_kj_kg=round(h_f, 2),
            enthalpy_vaporization_kj_kg=round(h_fg, 2),
            specific_volume_m3_kg=round(v_g, 4),
        )

    def _calculate_energy_metrics(
        self,
        steam_loss_kg_hr: float,
        steam_props: SteamProperties
    ) -> EnergyMetrics:
        """
        Calculate energy loss metrics.

        FORMULAS:
        - Energy loss (kW) = W * h_fg / 3600
        - Energy loss (MMBtu/hr) = kW * 0.003412

        Args:
            steam_loss_kg_hr: Steam loss rate in kg/hr
            steam_props: Steam properties

        Returns:
            EnergyMetrics with all energy calculations
        """
        h_fg = steam_props.enthalpy_vaporization_kj_kg

        # Energy loss rate (kW)
        energy_kw = steam_loss_kg_hr * h_fg / 3600.0

        # Convert to MMBtu/hr (1 kW = 0.003412 MMBtu/hr)
        energy_mmbtu = energy_kw * 0.003412

        # Equivalent fuel consumption (natural gas, m3/hr)
        # 1 m3 natural gas ~ 36 MJ = 10 kWh
        fuel_m3_hr = energy_kw / 10.0 / self.config.boiler_efficiency

        # Daily, monthly, annual energy
        daily_kwh = energy_kw * 24
        monthly_kwh = daily_kwh * 30
        annual_kwh = energy_kw * self.config.operating_hours_per_year

        return EnergyMetrics(
            steam_loss_kg_hr=round(steam_loss_kg_hr, 3),
            energy_loss_kw=round(energy_kw, 3),
            energy_loss_mmbtu_hr=round(energy_mmbtu, 6),
            fuel_consumption_m3_hr=round(fuel_m3_hr, 4),
            daily_energy_loss_kwh=round(daily_kwh, 2),
            monthly_energy_loss_kwh=round(monthly_kwh, 2),
            annual_energy_loss_kwh=round(annual_kwh, 2),
        )

    def _calculate_co2_emissions(
        self,
        energy_loss_kw: float,
        hours_per_year: int
    ) -> float:
        """
        Calculate annual CO2 emissions from energy loss.

        FORMULA:
        CO2 (kg/yr) = energy_kw * hours * emission_factor

        Args:
            energy_loss_kw: Energy loss rate in kW
            hours_per_year: Annual operating hours

        Returns:
            Annual CO2 emissions in kg
        """
        # Adjust for boiler efficiency (more fuel burned = more emissions)
        adjusted_energy = energy_loss_kw / self.config.boiler_efficiency

        # Calculate emissions
        co2_kg_yr = adjusted_energy * hours_per_year * self.config.co2_factor_kg_per_kwh

        return co2_kg_yr

    def _calculate_roi(
        self,
        repair_cost_usd: float,
        annual_savings_usd: float,
        discount_rate: float = 0.10
    ) -> ROIAnalysis:
        """
        Calculate ROI metrics for trap repair.

        FORMULAS:
        - Simple payback = repair_cost / annual_savings
        - ROI = (annual_savings / repair_cost) * 100
        - NPV = sum(savings / (1 + r)^n) - cost

        Args:
            repair_cost_usd: Total repair/replacement cost
            annual_savings_usd: Annual savings from repair
            discount_rate: Discount rate for NPV (default 10%)

        Returns:
            ROIAnalysis with complete financial analysis
        """
        # Handle zero/negative cases
        if annual_savings_usd <= 0:
            return ROIAnalysis(
                repair_cost_usd=repair_cost_usd,
                annual_savings_usd=0.0,
                simple_payback_days=999999.0,
                simple_payback_months=999999.0,
                roi_percent=0.0,
                npv_5yr_usd=-repair_cost_usd,
                irr_percent=0.0,
            )

        # Simple payback
        payback_years = repair_cost_usd / annual_savings_usd
        payback_days = payback_years * 365
        payback_months = payback_years * 12

        # ROI (first year)
        roi_percent = (annual_savings_usd / repair_cost_usd) * 100

        # NPV over 5 years
        npv = -repair_cost_usd
        for year in range(1, 6):
            npv += annual_savings_usd / ((1 + discount_rate) ** year)

        # Simplified IRR (iterative approximation)
        irr = self._estimate_irr(repair_cost_usd, annual_savings_usd, years=5)

        # Calculate break-even date
        if payback_years < 10:
            from datetime import timedelta
            break_even = datetime.now() + timedelta(days=payback_days)
            break_even_str = break_even.strftime("%Y-%m-%d")
        else:
            break_even_str = None

        return ROIAnalysis(
            repair_cost_usd=round(repair_cost_usd, 2),
            annual_savings_usd=round(annual_savings_usd, 2),
            simple_payback_days=round(payback_days, 1),
            simple_payback_months=round(payback_months, 2),
            roi_percent=round(roi_percent, 1),
            npv_5yr_usd=round(npv, 2),
            irr_percent=round(irr * 100, 1),
            break_even_date=break_even_str,
        )

    def _estimate_irr(
        self,
        initial_cost: float,
        annual_cashflow: float,
        years: int = 5
    ) -> float:
        """
        Estimate Internal Rate of Return using bisection method.

        IRR is the rate r where NPV = 0:
        -cost + sum(cashflow / (1+r)^n) = 0

        Args:
            initial_cost: Initial investment
            annual_cashflow: Annual cash flow
            years: Project duration in years

        Returns:
            Estimated IRR as decimal (0.15 = 15%)
        """
        if annual_cashflow <= 0:
            return 0.0

        def npv_at_rate(rate: float) -> float:
            if rate <= -1:
                return float('inf')
            npv = -initial_cost
            for year in range(1, years + 1):
                npv += annual_cashflow / ((1 + rate) ** year)
            return npv

        # Bisection search for IRR
        low_rate = 0.0
        high_rate = 5.0  # 500% max

        # Check if solution exists in range
        if npv_at_rate(low_rate) < 0:
            return 0.0
        if npv_at_rate(high_rate) > 0:
            return high_rate

        # Bisection iteration
        for _ in range(50):
            mid_rate = (low_rate + high_rate) / 2
            npv = npv_at_rate(mid_rate)

            if abs(npv) < 0.01:
                return mid_rate
            elif npv > 0:
                low_rate = mid_rate
            else:
                high_rate = mid_rate

        return (low_rate + high_rate) / 2

    def calculate_fleet_summary(
        self,
        trap_analyses: List[CostAnalysisResult]
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics for a fleet of traps.

        Args:
            trap_analyses: List of individual trap cost analyses

        Returns:
            Dictionary with fleet summary statistics
        """
        if not trap_analyses:
            return {
                "total_traps": 0,
                "total_steam_loss_kg_hr": 0,
                "total_annual_cost_usd": 0,
                "total_co2_emissions_kg_yr": 0,
            }

        total_steam_loss = sum(t.steam_loss_kg_hr for t in trap_analyses)
        total_annual_cost = sum(t.annual_cost_usd for t in trap_analyses)
        total_co2 = sum(t.co2_emissions_kg_yr for t in trap_analyses)

        # Average energy loss
        avg_energy_kw = sum(
            t.energy_metrics.energy_loss_kw for t in trap_analyses
        ) / len(trap_analyses)

        # Find worst offenders
        sorted_by_cost = sorted(
            trap_analyses,
            key=lambda x: x.annual_cost_usd,
            reverse=True
        )
        top_5_losers = sorted_by_cost[:5]

        return {
            "total_traps": len(trap_analyses),
            "total_steam_loss_kg_hr": round(total_steam_loss, 2),
            "total_annual_cost_usd": round(total_annual_cost, 2),
            "total_co2_emissions_kg_yr": round(total_co2, 2),
            "average_steam_loss_kg_hr": round(
                total_steam_loss / len(trap_analyses), 2
            ),
            "average_energy_loss_kw": round(avg_energy_kw, 2),
            "daily_cost_usd": round(total_annual_cost / 365, 2),
            "monthly_cost_usd": round(total_annual_cost / 12, 2),
            "top_5_losers": [
                {
                    "trap_id": t.trap_id,
                    "annual_cost_usd": t.annual_cost_usd,
                    "steam_loss_kg_hr": t.steam_loss_kg_hr,
                }
                for t in top_5_losers
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator statistics."""
        return {
            "calculation_count": self.calculation_count,
            "config": self.config.to_dict(),
            "supported_pressures_bar": list(self.STEAM_PROPERTIES_TABLE.keys()),
            "trap_types": list(self.DISCHARGE_COEFFICIENTS.keys()),
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "SteamLossCostCalculator",
    "CostCalculatorConfig",
    "CostAnalysisResult",
    "ROIAnalysis",
    "EnergyMetrics",
    "SteamProperties",
]
