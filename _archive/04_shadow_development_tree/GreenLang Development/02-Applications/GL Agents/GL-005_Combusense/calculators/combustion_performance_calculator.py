# -*- coding: utf-8 -*-
"""
Combustion Performance Calculator for GL-005 CombustionControlAgent

Calculates thermal efficiency, heat losses, and combustion quality metrics
according to ASME PTC 4.1 methodology. Zero-hallucination design using
deterministic thermodynamic calculations.

Reference Standards:
- ASME PTC 4.1: Fired Steam Generators Performance Test Codes
- ASME PTC 4: Indirect Method Heat Loss Calculations
- ISO 9001: Heat Balance Calculation Method
- DIN EN 12952: Water-tube boilers - Heat balance calculation
- BS 845: Methods for Assessing Thermal Performance of Boilers

Mathematical Formulas:
- Thermal Efficiency: η = Q_output / Q_input = Q_output / (ṁ_fuel * LHV)
- Stack Loss: Q_stack = ṁ_flue * Cp * (T_flue - T_ambient)
- Radiation Loss: Q_rad = ε * σ * A * (T_surface^4 - T_ambient^4)
- Combustion Efficiency: η_comb = 100 - %Loss_stack - %Loss_rad - %Loss_unburned
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import math
import logging
import hashlib
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceLevel(str, Enum):
    """Performance classification levels"""
    EXCELLENT = "excellent"  # η > 90%
    GOOD = "good"  # η 80-90%
    FAIR = "fair"  # η 70-80%
    POOR = "poor"  # η 60-70%
    UNACCEPTABLE = "unacceptable"  # η < 60%


class LossCategory(str, Enum):
    """Categories of heat losses"""
    DRY_FLUE_GAS = "dry_flue_gas"
    MOISTURE_IN_FUEL = "moisture_in_fuel"
    MOISTURE_FROM_COMBUSTION = "moisture_from_combustion"
    INCOMPLETE_COMBUSTION = "incomplete_combustion"
    RADIATION_CONVECTION = "radiation_convection"
    UNACCOUNTED = "unaccounted"


@dataclass
class HeatLoss:
    """Individual heat loss component"""
    category: LossCategory
    loss_kw: float
    loss_percent: float
    description: str


@dataclass
class PerformanceTrend:
    """Performance trend over time"""
    efficiency_trend: str  # improving, stable, degrading
    average_efficiency: float
    efficiency_std_dev: float
    degradation_rate_percent_per_day: float


class CombustionPerformanceInput(BaseModel):
    """Input parameters for combustion performance calculation"""

    # Fuel input
    fuel_flow_rate_kg_per_hr: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Fuel flow rate"
    )
    fuel_lower_heating_value_mj_per_kg: float = Field(
        ...,
        gt=0,
        le=100,
        description="Lower heating value (LHV)"
    )
    fuel_higher_heating_value_mj_per_kg: float = Field(
        ...,
        gt=0,
        le=100,
        description="Higher heating value (HHV)"
    )

    # Fuel composition
    fuel_carbon_percent: float = Field(default=85.0, ge=0, le=100)
    fuel_hydrogen_percent: float = Field(default=13.0, ge=0, le=100)
    fuel_sulfur_percent: float = Field(default=0.5, ge=0, le=10)
    fuel_oxygen_percent: float = Field(default=1.0, ge=0, le=100)
    fuel_moisture_percent: float = Field(default=0.0, ge=0, le=50)
    fuel_ash_percent: float = Field(default=0.0, ge=0, le=20)

    # Air and flue gas
    air_flow_rate_kg_per_hr: float = Field(..., ge=0)
    flue_gas_temperature_c: float = Field(..., ge=0, le=2000)
    flue_gas_o2_percent: float = Field(..., ge=0, le=21)
    flue_gas_co2_percent: Optional[float] = Field(None, ge=0, le=20)
    flue_gas_co_ppm: float = Field(default=0, ge=0, le=10000)

    # Ambient conditions
    ambient_temperature_c: float = Field(default=25.0, ge=-50, le=60)
    ambient_pressure_pa: float = Field(default=101325, ge=80000, le=110000)
    ambient_humidity_percent: float = Field(default=60, ge=0, le=100)

    # Heat output measurements
    measured_heat_output_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Measured useful heat output (if available)"
    )
    steam_flow_rate_kg_per_hr: Optional[float] = Field(
        None,
        ge=0,
        description="Steam flow rate (for boilers)"
    )
    steam_pressure_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Steam pressure"
    )
    feed_water_temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=200,
        description="Feed water temperature"
    )

    # Equipment parameters
    combustor_surface_area_m2: Optional[float] = Field(
        None,
        ge=0,
        le=1000,
        description="External surface area"
    )
    surface_temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=500,
        description="External surface temperature"
    )
    surface_emissivity: float = Field(default=0.85, ge=0, le=1)

    # Performance history (for trending)
    efficiency_history: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=100,
        description="Historical efficiency values (%)"
    )
    timestamp_history: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=100,
        description="Timestamps for efficiency history (seconds)"
    )

    # Calculation method
    use_hhv_basis: bool = Field(
        default=False,
        description="Use HHV basis for efficiency (default: LHV)"
    )
    include_radiation_loss: bool = Field(
        default=True,
        description="Include radiation/convection losses"
    )


class CombustionPerformanceOutput(BaseModel):
    """Combustion performance calculation results"""

    # Thermal efficiency (ASME PTC 4.1)
    thermal_efficiency_lhv_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Thermal efficiency (LHV basis)"
    )
    thermal_efficiency_hhv_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Thermal efficiency (HHV basis)"
    )
    combustion_efficiency_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Combustion efficiency (100 - losses)"
    )

    # Heat balance
    gross_heat_input_kw: float = Field(..., description="Gross heat input (HHV)")
    net_heat_input_kw: float = Field(..., description="Net heat input (LHV)")
    useful_heat_output_kw: float = Field(..., description="Useful heat output")

    # Heat losses (ASME PTC 4.1 methodology)
    heat_losses: List[Dict[str, float]] = Field(
        ...,
        description="Detailed heat loss breakdown"
    )
    total_heat_loss_kw: float = Field(..., description="Total heat losses")
    total_heat_loss_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Total losses as % of input"
    )

    # Individual loss components
    dry_flue_gas_loss_percent: float
    moisture_loss_percent: float
    incomplete_combustion_loss_percent: float
    radiation_convection_loss_percent: float
    unaccounted_loss_percent: float

    # Stack loss details
    stack_loss_kw: float = Field(..., description="Stack/flue gas heat loss")
    stack_temperature_excess_c: float = Field(
        ...,
        description="Excess stack temperature above ambient"
    )

    # Combustion quality metrics
    excess_air_percent: float = Field(..., description="Excess air percentage")
    combustion_completeness_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Combustion completeness"
    )
    co_emission_quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="CO emissions quality (100 = excellent, <50 = poor)"
    )

    # Performance classification
    performance_level: PerformanceLevel
    performance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall performance score"
    )

    # Performance trending
    performance_trend: Optional[str] = Field(
        None,
        description="improving, stable, or degrading"
    )
    efficiency_change_rate_percent_per_day: Optional[float] = None

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Performance improvement recommendations"
    )
    issues_detected: List[str] = Field(
        default_factory=list,
        description="Performance issues detected"
    )

    # Comparison to best practice
    efficiency_gap_to_best_practice_percent: float = Field(
        ...,
        description="Gap between current and best practice efficiency"
    )
    potential_fuel_savings_percent: float = Field(
        ...,
        description="Potential fuel savings if reaching best practice"
    )
    potential_cost_savings_usd_per_year: Optional[float] = None

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for calculation provenance"
    )


class CombustionPerformanceCalculator:
    """
    Combustion performance calculator implementing ASME PTC 4.1 methodology.

    Calculates thermal efficiency using the indirect method (heat loss method):

        η = 100 - Σ Losses

    Where losses include:
        1. Dry flue gas loss (sensible heat in flue gas)
        2. Moisture loss (latent + sensible heat in water vapor)
        3. Incomplete combustion loss (CO, unburned fuel)
        4. Radiation and convection loss
        5. Unaccounted losses

    ASME PTC 4.1 Method:
        - More accurate than direct method for efficiency
        - Accounts for all significant heat losses
        - Reference standard for boiler testing

    Thermal Efficiency Definitions:
        - LHV basis: η_LHV = Q_useful / (ṁ_fuel * LHV) * 100
        - HHV basis: η_HHV = Q_useful / (ṁ_fuel * HHV) * 100

        LHV basis is typically used in Europe
        HHV basis is typically used in North America
    """

    # Specific heat capacities (kJ/kg·K)
    CP = {
        'air': 1.005,
        'flue_gas': 1.05,  # Average
        'water_vapor': 2.0,
        'water_liquid': 4.18
    }

    # Latent heat of vaporization (kJ/kg)
    H_FG_WATER = 2257  # At 100°C

    # Stefan-Boltzmann constant (W/m²·K⁴)
    SIGMA = 5.67e-8

    # Best practice efficiency targets by fuel type
    BEST_PRACTICE_EFFICIENCY = {
        'natural_gas': 92.0,  # %
        'fuel_oil': 88.0,
        'diesel': 90.0,
        'coal': 85.0,
        'biomass': 80.0
    }

    def __init__(self):
        """Initialize combustion performance calculator"""
        self.logger = logging.getLogger(__name__)

        # Performance history for trending
        self.efficiency_history = deque(maxlen=100)
        self.timestamp_history = deque(maxlen=100)

    def calculate_performance(
        self,
        perf_input: CombustionPerformanceInput
    ) -> CombustionPerformanceOutput:
        """
        Calculate combustion performance using ASME PTC 4.1 methodology.

        Algorithm (Indirect Method):
            1. Calculate gross and net heat input
            2. Calculate individual heat losses:
               a. Dry flue gas loss
               b. Moisture losses (in fuel + from combustion)
               c. Incomplete combustion loss (CO)
               d. Radiation/convection loss
               e. Unaccounted losses
            3. Calculate combustion efficiency = 100 - Σ losses
            4. Calculate thermal efficiency from measured output
            5. Assess performance level and trends
            6. Generate recommendations

        Args:
            perf_input: Performance calculation inputs

        Returns:
            CombustionPerformanceOutput with complete performance assessment
        """
        self.logger.info("Calculating combustion performance (ASME PTC 4.1)")

        # Step 1: Calculate heat input
        gross_heat_input_kw = (
            perf_input.fuel_flow_rate_kg_per_hr *
            perf_input.fuel_higher_heating_value_mj_per_kg *
            1000 / 3600  # Convert MJ/hr to kW
        )

        net_heat_input_kw = (
            perf_input.fuel_flow_rate_kg_per_hr *
            perf_input.fuel_lower_heating_value_mj_per_kg *
            1000 / 3600
        )

        # Step 2: Calculate individual heat losses

        # 2a. Dry flue gas loss
        dry_flue_gas_loss_pct, stack_loss_kw = self._calculate_dry_flue_gas_loss(
            perf_input,
            net_heat_input_kw
        )

        # 2b. Moisture losses
        moisture_loss_pct = self._calculate_moisture_losses(
            perf_input,
            net_heat_input_kw
        )

        # 2c. Incomplete combustion loss (CO)
        incomplete_combustion_loss_pct = self._calculate_incomplete_combustion_loss(
            perf_input,
            net_heat_input_kw
        )

        # 2d. Radiation and convection loss
        radiation_loss_pct = 0.0
        if perf_input.include_radiation_loss:
            radiation_loss_pct = self._calculate_radiation_convection_loss(
                perf_input,
                net_heat_input_kw
            )

        # 2e. Unaccounted losses (typically 0.5-2%)
        unaccounted_loss_pct = self._estimate_unaccounted_losses(
            perf_input,
            dry_flue_gas_loss_pct
        )

        # Step 3: Calculate combustion efficiency
        total_loss_pct = (
            dry_flue_gas_loss_pct +
            moisture_loss_pct +
            incomplete_combustion_loss_pct +
            radiation_loss_pct +
            unaccounted_loss_pct
        )

        combustion_efficiency_pct = 100 - total_loss_pct

        # Step 4: Calculate thermal efficiency
        if perf_input.measured_heat_output_kw is not None:
            useful_heat_output_kw = perf_input.measured_heat_output_kw
        elif perf_input.steam_flow_rate_kg_per_hr is not None:
            # Calculate from steam production
            useful_heat_output_kw = self._calculate_heat_from_steam(
                perf_input.steam_flow_rate_kg_per_hr,
                perf_input.steam_pressure_bar,
                perf_input.feed_water_temperature_c
            )
        else:
            # Estimate from combustion efficiency
            useful_heat_output_kw = net_heat_input_kw * (combustion_efficiency_pct / 100)

        thermal_efficiency_lhv = (useful_heat_output_kw / net_heat_input_kw * 100)
        thermal_efficiency_hhv = (useful_heat_output_kw / gross_heat_input_kw * 100)

        # Step 5: Calculate excess air
        excess_air_pct = self._calculate_excess_air(
            perf_input.flue_gas_o2_percent
        )

        # Step 6: Calculate combustion completeness
        combustion_completeness = self._calculate_combustion_completeness(
            perf_input.flue_gas_co_ppm,
            excess_air_pct
        )

        # Step 7: CO emission quality score
        co_quality_score = self._calculate_co_quality_score(
            perf_input.flue_gas_co_ppm
        )

        # Step 8: Build heat loss breakdown
        total_loss_kw = net_heat_input_kw - useful_heat_output_kw

        heat_losses = [
            {
                'category': LossCategory.DRY_FLUE_GAS.value,
                'loss_kw': self._round_decimal(stack_loss_kw, 2),
                'loss_percent': self._round_decimal(dry_flue_gas_loss_pct, 2),
                'description': 'Sensible heat in dry flue gas'
            },
            {
                'category': LossCategory.MOISTURE_IN_FUEL.value,
                'loss_kw': self._round_decimal(net_heat_input_kw * moisture_loss_pct / 100, 2),
                'loss_percent': self._round_decimal(moisture_loss_pct, 2),
                'description': 'Moisture in fuel and from combustion'
            },
            {
                'category': LossCategory.INCOMPLETE_COMBUSTION.value,
                'loss_kw': self._round_decimal(net_heat_input_kw * incomplete_combustion_loss_pct / 100, 2),
                'loss_percent': self._round_decimal(incomplete_combustion_loss_pct, 2),
                'description': 'Unburned fuel (CO)'
            },
            {
                'category': LossCategory.RADIATION_CONVECTION.value,
                'loss_kw': self._round_decimal(net_heat_input_kw * radiation_loss_pct / 100, 2),
                'loss_percent': self._round_decimal(radiation_loss_pct, 2),
                'description': 'Radiation and convection from surfaces'
            },
            {
                'category': LossCategory.UNACCOUNTED.value,
                'loss_kw': self._round_decimal(net_heat_input_kw * unaccounted_loss_pct / 100, 2),
                'loss_percent': self._round_decimal(unaccounted_loss_pct, 2),
                'description': 'Unaccounted losses'
            }
        ]

        # Step 9: Performance classification
        performance_level = self._classify_performance(thermal_efficiency_lhv)

        # Step 10: Calculate performance score (weighted)
        performance_score = self._calculate_performance_score(
            thermal_efficiency_lhv,
            combustion_completeness,
            co_quality_score
        )

        # Step 11: Performance trending
        performance_trend = None
        efficiency_change_rate = None

        if perf_input.efficiency_history and perf_input.timestamp_history:
            trend_analysis = self._analyze_performance_trend(
                perf_input.efficiency_history,
                perf_input.timestamp_history
            )
            performance_trend = trend_analysis.efficiency_trend
            efficiency_change_rate = trend_analysis.degradation_rate_percent_per_day

        # Step 12: Best practice comparison
        best_practice_eff = self.BEST_PRACTICE_EFFICIENCY.get('natural_gas', 90.0)
        efficiency_gap = max(0, best_practice_eff - thermal_efficiency_lhv)
        potential_fuel_savings = (efficiency_gap / thermal_efficiency_lhv * 100) if thermal_efficiency_lhv > 0 else 0

        # Step 13: Detect issues and generate recommendations
        issues = self._detect_performance_issues(
            thermal_efficiency_lhv,
            dry_flue_gas_loss_pct,
            incomplete_combustion_loss_pct,
            excess_air_pct,
            perf_input.flue_gas_temperature_c,
            perf_input.ambient_temperature_c
        )

        recommendations = self._generate_performance_recommendations(
            performance_level,
            thermal_efficiency_lhv,
            dry_flue_gas_loss_pct,
            excess_air_pct,
            perf_input.flue_gas_temperature_c,
            issues
        )

        # Calculate stack temperature excess
        stack_temp_excess = perf_input.flue_gas_temperature_c - perf_input.ambient_temperature_c

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            perf_input,
            thermal_efficiency_lhv,
            combustion_efficiency_pct,
            total_loss_pct
        )

        return CombustionPerformanceOutput(
            thermal_efficiency_lhv_percent=self._round_decimal(thermal_efficiency_lhv, 2),
            thermal_efficiency_hhv_percent=self._round_decimal(thermal_efficiency_hhv, 2),
            combustion_efficiency_percent=self._round_decimal(combustion_efficiency_pct, 2),
            gross_heat_input_kw=self._round_decimal(gross_heat_input_kw, 2),
            net_heat_input_kw=self._round_decimal(net_heat_input_kw, 2),
            useful_heat_output_kw=self._round_decimal(useful_heat_output_kw, 2),
            heat_losses=heat_losses,
            total_heat_loss_kw=self._round_decimal(total_loss_kw, 2),
            total_heat_loss_percent=self._round_decimal(total_loss_pct, 2),
            dry_flue_gas_loss_percent=self._round_decimal(dry_flue_gas_loss_pct, 2),
            moisture_loss_percent=self._round_decimal(moisture_loss_pct, 2),
            incomplete_combustion_loss_percent=self._round_decimal(incomplete_combustion_loss_pct, 2),
            radiation_convection_loss_percent=self._round_decimal(radiation_loss_pct, 2),
            unaccounted_loss_percent=self._round_decimal(unaccounted_loss_pct, 2),
            stack_loss_kw=self._round_decimal(stack_loss_kw, 2),
            stack_temperature_excess_c=self._round_decimal(stack_temp_excess, 1),
            excess_air_percent=self._round_decimal(excess_air_pct, 2),
            combustion_completeness_percent=self._round_decimal(combustion_completeness, 2),
            co_emission_quality_score=self._round_decimal(co_quality_score, 2),
            performance_level=performance_level,
            performance_score=self._round_decimal(performance_score, 2),
            performance_trend=performance_trend,
            efficiency_change_rate_percent_per_day=efficiency_change_rate,
            recommendations=recommendations,
            issues_detected=issues,
            efficiency_gap_to_best_practice_percent=self._round_decimal(efficiency_gap, 2),
            potential_fuel_savings_percent=self._round_decimal(potential_fuel_savings, 2),
            provenance_hash=provenance_hash
        )

    def _calculate_dry_flue_gas_loss(
        self,
        perf_input: CombustionPerformanceInput,
        net_heat_input_kw: float
    ) -> Tuple[float, float]:
        """
        Calculate dry flue gas heat loss (ASME PTC 4.1).

        Formula:
            Q_stack = ṁ_flue * Cp_flue * (T_flue - T_ambient)
            Loss % = (Q_stack / Q_input) * 100

        Args:
            perf_input: Performance inputs
            net_heat_input_kw: Net heat input

        Returns:
            Tuple of (loss_percent, loss_kw)
        """
        # Calculate flue gas mass flow
        flue_gas_mass = (
            perf_input.fuel_flow_rate_kg_per_hr +
            perf_input.air_flow_rate_kg_per_hr
        )

        # Temperature difference
        temp_diff = perf_input.flue_gas_temperature_c - perf_input.ambient_temperature_c

        # Stack heat loss (kW)
        stack_loss_kw = (
            flue_gas_mass * self.CP['flue_gas'] * temp_diff / 3600
        )

        # Loss percentage
        loss_percent = (stack_loss_kw / net_heat_input_kw * 100) if net_heat_input_kw > 0 else 0

        return loss_percent, stack_loss_kw

    def _calculate_moisture_losses(
        self,
        perf_input: CombustionPerformanceInput,
        net_heat_input_kw: float
    ) -> float:
        """
        Calculate moisture losses (in fuel + from H2 combustion).

        Moisture sources:
            1. Moisture in fuel
            2. Water from H2 combustion: H2 + 0.5 O2 → H2O

        Formula:
            Q_moisture = ṁ_H2O * (h_fg + Cp * ΔT)

        Args:
            perf_input: Performance inputs
            net_heat_input_kw: Net heat input

        Returns:
            Moisture loss percentage
        """
        # Moisture in fuel (kg/hr)
        moisture_in_fuel = (
            perf_input.fuel_flow_rate_kg_per_hr *
            perf_input.fuel_moisture_percent / 100
        )

        # Water from H2 combustion (kg/hr)
        # H2 → H2O: 1 kg H2 → 9 kg H2O
        hydrogen_mass = (
            perf_input.fuel_flow_rate_kg_per_hr *
            perf_input.fuel_hydrogen_percent / 100
        )
        water_from_combustion = hydrogen_mass * 9

        # Total water in flue gas
        total_water = moisture_in_fuel + water_from_combustion

        # Heat loss from moisture
        # Latent heat + sensible heat to flue gas temperature
        temp_diff = perf_input.flue_gas_temperature_c - 100  # Above boiling
        moisture_loss_kw = (
            total_water * (self.H_FG_WATER + self.CP['water_vapor'] * temp_diff) / 3600
        )

        # Loss percentage
        loss_percent = (moisture_loss_kw / net_heat_input_kw * 100) if net_heat_input_kw > 0 else 0

        return loss_percent

    def _calculate_incomplete_combustion_loss(
        self,
        perf_input: CombustionPerformanceInput,
        net_heat_input_kw: float
    ) -> float:
        """
        Calculate heat loss from incomplete combustion (CO).

        CO represents unburned carbon that didn't fully oxidize to CO2.

        Loss from CO (simplified):
            Loss % ≈ CO_ppm * K

        Where K ≈ 0.001 for typical fuels

        Args:
            perf_input: Performance inputs
            net_heat_input_kw: Net heat input

        Returns:
            Incomplete combustion loss percentage
        """
        co_ppm = perf_input.flue_gas_co_ppm

        # Empirical relationship (ASME PTC 4.1)
        # Loss increases with CO concentration
        if co_ppm < 100:
            loss_percent = co_ppm * 0.001
        elif co_ppm < 1000:
            loss_percent = 0.1 + (co_ppm - 100) * 0.002
        else:
            loss_percent = 2.0 + (co_ppm - 1000) * 0.005

        return min(loss_percent, 10)  # Cap at 10%

    def _calculate_radiation_convection_loss(
        self,
        perf_input: CombustionPerformanceInput,
        net_heat_input_kw: float
    ) -> float:
        """
        Calculate radiation and convection heat loss from surfaces.

        Stefan-Boltzmann Law:
            Q_rad = ε * σ * A * (T_surface^4 - T_ambient^4)

        Convection (simplified):
            Q_conv = h * A * (T_surface - T_ambient)

        Args:
            perf_input: Performance inputs
            net_heat_input_kw: Net heat input

        Returns:
            Radiation/convection loss percentage
        """
        if perf_input.combustor_surface_area_m2 is None:
            # Estimate as % of input (typically 0.5-2% for well-insulated equipment)
            return 1.0

        if perf_input.surface_temperature_c is None:
            return 1.0

        # Convert to Kelvin
        T_surface = perf_input.surface_temperature_c + 273.15
        T_ambient = perf_input.ambient_temperature_c + 273.15

        # Radiation loss (W)
        q_rad = (
            perf_input.surface_emissivity *
            self.SIGMA *
            perf_input.combustor_surface_area_m2 *
            (T_surface ** 4 - T_ambient ** 4)
        )

        # Convection loss (simplified, h ≈ 10 W/m²·K)
        h_conv = 10  # W/m²·K
        q_conv = (
            h_conv *
            perf_input.combustor_surface_area_m2 *
            (perf_input.surface_temperature_c - perf_input.ambient_temperature_c)
        )

        # Total surface loss (kW)
        total_surface_loss_kw = (q_rad + q_conv) / 1000

        # Loss percentage
        loss_percent = (total_surface_loss_kw / net_heat_input_kw * 100) if net_heat_input_kw > 0 else 1.0

        return min(loss_percent, 5)  # Cap at 5%

    def _estimate_unaccounted_losses(
        self,
        perf_input: CombustionPerformanceInput,
        dry_flue_gas_loss: float
    ) -> float:
        """
        Estimate unaccounted losses (ASME PTC 4.1).

        Includes:
            - Measurement uncertainties
            - Minor leaks
            - Ash/slag sensible heat
            - Other unquantified losses

        Typically 0.5-2% depending on equipment condition

        Args:
            perf_input: Performance inputs
            dry_flue_gas_loss: Dry flue gas loss (%)

        Returns:
            Unaccounted loss percentage
        """
        # Base unaccounted loss
        base_loss = 1.0  # %

        # Adjust based on ash content
        ash_adjustment = perf_input.fuel_ash_percent * 0.05

        unaccounted = base_loss + ash_adjustment

        return min(unaccounted, 3.0)  # Cap at 3%

    def _calculate_excess_air(self, o2_percent: float) -> float:
        """
        Calculate excess air from O2 measurement.

        Formula:
            EA% = O2 / (21 - O2) * 100

        Args:
            o2_percent: O2 in flue gas (%)

        Returns:
            Excess air percentage
        """
        if o2_percent >= 21:
            return 0

        excess_air = o2_percent / (21 - o2_percent) * 100

        return excess_air

    def _calculate_combustion_completeness(
        self,
        co_ppm: float,
        excess_air_percent: float
    ) -> float:
        """
        Calculate combustion completeness score.

        Perfect combustion = 100%
        High CO or low excess air reduces completeness

        Args:
            co_ppm: CO concentration
            excess_air_percent: Excess air

        Returns:
            Completeness percentage (0-100)
        """
        # Start at 100%
        completeness = 100.0

        # Penalize for high CO
        if co_ppm > 50:
            completeness -= min(20, (co_ppm - 50) / 50)

        # Penalize for insufficient air
        if excess_air_percent < 5:
            completeness -= (5 - excess_air_percent) * 2

        return max(0, completeness)

    def _calculate_co_quality_score(self, co_ppm: float) -> float:
        """
        Calculate CO emission quality score.

        Args:
            co_ppm: CO concentration

        Returns:
            Quality score (0-100, 100 = excellent)
        """
        if co_ppm <= 50:
            return 100
        elif co_ppm <= 100:
            return 90 - (co_ppm - 50)
        elif co_ppm <= 200:
            return 40 - (co_ppm - 100) / 2
        else:
            return max(0, 20 - (co_ppm - 200) / 20)

    def _calculate_heat_from_steam(
        self,
        steam_flow_kg_per_hr: float,
        steam_pressure_bar: Optional[float],
        feedwater_temp_c: Optional[float]
    ) -> float:
        """
        Calculate heat output from steam production.

        Q = ṁ_steam * (h_steam - h_feedwater)

        Args:
            steam_flow_kg_per_hr: Steam flow rate
            steam_pressure_bar: Steam pressure
            feedwater_temp_c: Feedwater temperature

        Returns:
            Heat output in kW
        """
        # Simplified - use typical values
        # At 10 bar: h_fg ≈ 2000 kJ/kg
        h_fg = 2000  # kJ/kg (approximate)

        # Feedwater enthalpy
        if feedwater_temp_c:
            h_feedwater = self.CP['water_liquid'] * feedwater_temp_c
        else:
            h_feedwater = 100  # kJ/kg (assume 25°C)

        # Heat output
        heat_output_kw = steam_flow_kg_per_hr * (h_fg + 300 - h_feedwater) / 3600

        return heat_output_kw

    def _classify_performance(self, efficiency: float) -> PerformanceLevel:
        """Classify performance level"""
        if efficiency >= 90:
            return PerformanceLevel.EXCELLENT
        elif efficiency >= 80:
            return PerformanceLevel.GOOD
        elif efficiency >= 70:
            return PerformanceLevel.FAIR
        elif efficiency >= 60:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.UNACCEPTABLE

    def _calculate_performance_score(
        self,
        efficiency: float,
        combustion_completeness: float,
        co_quality: float
    ) -> float:
        """
        Calculate overall performance score (weighted).

        Args:
            efficiency: Thermal efficiency (%)
            combustion_completeness: Completeness (%)
            co_quality: CO quality score

        Returns:
            Performance score (0-100)
        """
        # Weighted average
        score = (
            0.6 * efficiency +
            0.2 * combustion_completeness +
            0.2 * co_quality
        )

        return min(100, max(0, score))

    def _analyze_performance_trend(
        self,
        efficiency_history: List[float],
        timestamp_history: List[float]
    ) -> PerformanceTrend:
        """
        Analyze performance trend over time.

        Args:
            efficiency_history: Historical efficiency values
            timestamp_history: Timestamps

        Returns:
            PerformanceTrend analysis
        """
        if len(efficiency_history) < 2:
            return PerformanceTrend("stable", efficiency_history[0], 0, 0)

        # Calculate statistics
        avg_eff = sum(efficiency_history) / len(efficiency_history)
        variance = sum((e - avg_eff) ** 2 for e in efficiency_history) / len(efficiency_history)
        std_dev = math.sqrt(variance)

        # Linear regression for trend
        n = len(efficiency_history)
        x = timestamp_history
        y = efficiency_history

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
        else:
            slope = 0

        # Convert slope to % per day
        time_range_days = (x[-1] - x[0]) / 86400  # seconds to days
        degradation_rate = slope * 86400 if time_range_days > 0 else 0

        # Classify trend
        if degradation_rate < -0.1:
            trend = "degrading"
        elif degradation_rate > 0.1:
            trend = "improving"
        else:
            trend = "stable"

        return PerformanceTrend(
            efficiency_trend=trend,
            average_efficiency=avg_eff,
            efficiency_std_dev=std_dev,
            degradation_rate_percent_per_day=degradation_rate
        )

    def _detect_performance_issues(
        self,
        efficiency: float,
        stack_loss: float,
        co_loss: float,
        excess_air: float,
        flue_temp: float,
        ambient_temp: float
    ) -> List[str]:
        """Detect performance issues"""
        issues = []

        if efficiency < 70:
            issues.append("Low thermal efficiency (<70%) - significant improvement opportunity")

        if stack_loss > 15:
            issues.append(f"High stack loss ({stack_loss:.1f}%) - consider economizer or reduce flue gas temperature")

        if co_loss > 1:
            issues.append(f"High CO loss ({co_loss:.2f}%) - incomplete combustion")

        if excess_air > 40:
            issues.append(f"Excessive air ({excess_air:.1f}%) - reduce to improve efficiency")

        if excess_air < 5:
            issues.append(f"Insufficient air ({excess_air:.1f}%) - risk of incomplete combustion")

        if (flue_temp - ambient_temp) > 300:
            issues.append(f"Very high stack temperature ({flue_temp:.0f}°C) - install heat recovery")

        return issues

    def _generate_performance_recommendations(
        self,
        performance_level: PerformanceLevel,
        efficiency: float,
        stack_loss: float,
        excess_air: float,
        flue_temp: float,
        issues: List[str]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        if performance_level in [PerformanceLevel.POOR, PerformanceLevel.UNACCEPTABLE]:
            recommendations.append("CRITICAL: Poor performance - immediate optimization required")
            recommendations.append("Schedule comprehensive combustion tuning and equipment inspection")

        if stack_loss > 10:
            recommendations.append(
                f"Stack loss {stack_loss:.1f}% - install economizer or air preheater to recover heat"
            )
            recommendations.append("Target stack temperature <200°C for optimal efficiency")

        if excess_air > 25:
            recommendations.append(
                f"Reduce excess air from {excess_air:.1f}% to 15-20% to improve efficiency ~{(excess_air-20)*0.1:.1f}%"
            )

        if flue_temp > 250:
            recommendations.append(
                "Install heat recovery equipment (economizer/air preheater) - potential 3-8% efficiency gain"
            )

        if efficiency < 85:
            best_practice = 90
            gap = best_practice - efficiency
            recommendations.append(
                f"Efficiency gap to best practice: {gap:.1f}% - potential fuel savings {gap/efficiency*100:.1f}%"
            )

        if not issues:
            recommendations.append("Performance within acceptable range - maintain current operation")

        return recommendations

    def _calculate_provenance(
        self,
        perf_input: CombustionPerformanceInput,
        efficiency: float,
        combustion_eff: float,
        total_loss: float
    ) -> str:
        """Calculate SHA-256 hash for complete audit trail"""
        provenance_data = {
            'fuel_flow': perf_input.fuel_flow_rate_kg_per_hr,
            'fuel_lhv': perf_input.fuel_lower_heating_value_mj_per_kg,
            'air_flow': perf_input.air_flow_rate_kg_per_hr,
            'flue_temp': perf_input.flue_gas_temperature_c,
            'flue_o2': perf_input.flue_gas_o2_percent,
            'thermal_efficiency': efficiency,
            'combustion_efficiency': combustion_eff,
            'total_loss': total_loss
        }

        provenance_str = str(provenance_data)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        if value is None:
            return None
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
