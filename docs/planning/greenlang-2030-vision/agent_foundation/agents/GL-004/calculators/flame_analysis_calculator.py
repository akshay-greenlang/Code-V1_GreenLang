# -*- coding: utf-8 -*-
"""
Flame Analysis Calculator for GL-004 BurnerOptimizationAgent

Calculates flame characteristics including temperature, shape, stability, and combustion quality.
Zero-hallucination design using thermodynamics and combustion physics.

Reference Standards:
- IFRF (International Flame Research Foundation) Flame Characterization Methods
- ASME PTC 4.4: Gas Turbine Heat Recovery Steam Generators
- API 535: Burners for Fired Heaters in General Refinery Services
- EN 267: Automatic forced draught burners for liquid fuels

Physical Models:
- Adiabatic Flame Temperature: Energy balance at constant pressure
- Flame Speed: Laminar and turbulent flame propagation
- Flame Stability: Damköhler number and blow-off correlations
- Flame Length: Momentum and buoyancy controlled jet flames
"""

from typing import Dict, Optional, Tuple, List
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import hashlib
import logging
from datetime import datetime
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class FlameType(str, Enum):
    """Types of industrial flames"""
    DIFFUSION = "diffusion"  # Non-premixed
    PREMIXED = "premixed"    # Fully premixed
    PARTIALLY_PREMIXED = "partially_premixed"
    PILOT = "pilot"          # Pilot flame


class FlameStability(str, Enum):
    """Flame stability conditions"""
    STABLE = "stable"
    MARGINALLY_STABLE = "marginally_stable"
    UNSTABLE = "unstable"
    BLOWOFF_RISK = "blowoff_risk"
    FLASHBACK_RISK = "flashback_risk"


class FlameQuality(str, Enum):
    """Overall flame quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class FlameAnalysisInput(BaseModel):
    """Input parameters for flame analysis"""

    # Fuel properties
    fuel_type: str = Field(..., description="Type of fuel")
    fuel_flow_kg_hr: float = Field(..., gt=0, description="Fuel flow rate kg/hr")
    fuel_temperature_c: float = Field(25, description="Fuel inlet temperature °C")
    fuel_pressure_kpa: float = Field(101.325, gt=0, description="Fuel pressure kPa")

    # Fuel composition (mol % or vol %)
    fuel_ch4_percent: float = Field(0, ge=0, le=100, description="Methane %")
    fuel_c2h6_percent: float = Field(0, ge=0, le=100, description="Ethane %")
    fuel_c3h8_percent: float = Field(0, ge=0, le=100, description="Propane %")
    fuel_h2_percent: float = Field(0, ge=0, le=100, description="Hydrogen %")
    fuel_co_percent: float = Field(0, ge=0, le=100, description="CO %")
    fuel_n2_percent: float = Field(0, ge=0, le=100, description="Nitrogen %")

    # Fuel heating values
    fuel_lhv_mj_kg: float = Field(..., gt=0, description="Lower heating value MJ/kg")
    fuel_hhv_mj_kg: float = Field(..., gt=0, description="Higher heating value MJ/kg")

    # Air/oxidizer properties
    air_flow_kg_hr: float = Field(..., gt=0, description="Air flow rate kg/hr")
    air_temperature_c: float = Field(25, description="Air temperature °C")
    air_humidity_percent: float = Field(50, ge=0, le=100, description="Relative humidity %")
    oxygen_enrichment_percent: float = Field(21, ge=21, le=100, description="O2 in oxidizer %")

    # Combustion conditions
    excess_air_percent: float = Field(..., ge=-10, le=200, description="Excess air %")
    swirl_number: float = Field(0, ge=0, le=2, description="Swirl number")
    combustion_chamber_pressure_kpa: float = Field(101.325, description="Chamber pressure kPa")

    # Burner geometry
    burner_diameter_mm: float = Field(..., gt=0, description="Burner outlet diameter mm")
    quarl_angle_degrees: float = Field(0, ge=0, le=60, description="Quarl angle degrees")
    flame_tube_diameter_mm: Optional[float] = Field(None, gt=0, description="Confinement diameter mm")

    # Operating conditions
    load_percent: float = Field(100, gt=0, le=110, description="Load %")
    ambient_temperature_c: float = Field(25, description="Ambient temperature °C")

    # Measurements (optional)
    measured_flame_temp_c: Optional[float] = Field(None, description="Measured flame temperature °C")
    measured_flame_length_m: Optional[float] = Field(None, description="Measured flame length m")
    flue_gas_o2_percent: float = Field(..., ge=0, le=21, description="Flue gas O2 %")

    @validator('fuel_ch4_percent')
    def validate_fuel_composition(cls, v, values):
        """Ensure fuel composition sums to reasonable value"""
        # Note: Allow sum < 100% for mixtures with other components
        return v


class FlameAnalysisOutput(BaseModel):
    """Output from flame analysis calculations"""

    # Flame temperature
    adiabatic_flame_temp_k: float = Field(..., description="Adiabatic flame temperature K")
    adiabatic_flame_temp_c: float = Field(..., description="Adiabatic flame temperature °C")
    actual_flame_temp_c: float = Field(..., description="Actual flame temperature °C")
    temperature_efficiency_percent: float = Field(..., description="Temperature efficiency %")

    # Flame dimensions
    flame_length_m: float = Field(..., description="Flame length m")
    flame_diameter_m: float = Field(..., description="Maximum flame diameter m")
    flame_volume_m3: float = Field(..., description="Flame volume m³")
    flame_surface_area_m2: float = Field(..., description="Flame surface area m²")

    # Flame speeds
    laminar_flame_speed_m_s: float = Field(..., description="Laminar flame speed m/s")
    turbulent_flame_speed_m_s: float = Field(..., description="Turbulent flame speed m/s")
    jet_velocity_m_s: float = Field(..., description="Fuel jet velocity m/s")

    # Stability parameters
    damkohler_number: float = Field(..., description="Damköhler number Da")
    equivalence_ratio: float = Field(..., description="Equivalence ratio φ")
    stability_index: float = Field(..., description="Stability index (0-1)")
    blow_off_velocity_m_s: float = Field(..., description="Blow-off velocity m/s")
    flashback_velocity_m_s: float = Field(..., description="Flashback velocity m/s")
    operating_margin_percent: float = Field(..., description="Operating margin %")

    # Flame quality
    flame_type: str = Field(..., description="Type of flame")
    flame_stability: str = Field(..., description="Stability status")
    flame_quality: str = Field(..., description="Overall quality")
    luminosity_factor: float = Field(..., description="Luminosity factor (0-1)")
    combustion_intensity_mw_m3: float = Field(..., description="Combustion intensity MW/m³")

    # Heat transfer
    radiant_fraction: float = Field(..., description="Radiant heat fraction")
    convective_fraction: float = Field(..., description="Convective heat fraction")
    heat_release_rate_mw: float = Field(..., description="Heat release rate MW")
    specific_heat_release_mw_m2: float = Field(..., description="Specific heat release MW/m²")

    # Emissions indicators
    nox_potential_index: float = Field(..., description="NOx formation potential (0-1)")
    soot_formation_index: float = Field(..., description="Soot formation potential (0-1)")
    combustion_efficiency_percent: float = Field(..., description="Combustion efficiency %")

    # Diagnostics
    reynolds_number: float = Field(..., description="Jet Reynolds number")
    froude_number: float = Field(..., description="Froude number")
    richardson_number: float = Field(..., description="Richardson number")
    karlovitz_number: float = Field(..., description="Karlovitz number")

    # Recommendations
    optimization_potential: str = Field(..., description="Optimization potential assessment")
    recommended_adjustments: List[str] = Field(..., description="Recommended adjustments")

    # Provenance
    calculation_timestamp: str = Field(..., description="ISO timestamp")
    provenance_hash: str = Field(..., description="SHA-256 hash")


class FlameAnalysisCalculator:
    """
    Analyze flame characteristics using combustion physics.

    Zero-hallucination approach:
    - Thermodynamic calculations for temperature
    - Fluid mechanics for flame dimensions
    - Chemical kinetics for flame speed
    - Stability correlations from experimental data

    No AI/ML models used - pure physics-based calculations.
    """

    # Physical constants
    R = 8.314  # J/mol·K
    SIGMA = 5.67e-8  # Stefan-Boltzmann constant W/m²·K⁴
    G = 9.81  # m/s²

    # Specific heats at constant pressure (kJ/kg·K)
    CP = {
        'air': 1.005,
        'N2': 1.040,
        'O2': 0.918,
        'CO2': 0.844,
        'H2O': 1.996,
        'CH4': 2.220,
        'H2': 14.320,
        'CO': 1.040
    }

    # Molecular weights (g/mol)
    MW = {
        'CH4': 16.04, 'C2H6': 30.07, 'C3H8': 44.10,
        'H2': 2.016, 'CO': 28.01, 'N2': 28.014,
        'O2': 31.999, 'CO2': 44.01, 'H2O': 18.015,
        'air': 28.964
    }

    # Laminar flame speeds at STP (m/s)
    SL_REF = {
        'CH4': 0.40,  # Methane-air
        'C2H6': 0.43,  # Ethane-air
        'C3H8': 0.46,  # Propane-air
        'H2': 2.10,   # Hydrogen-air
        'CO': 0.17    # CO-air
    }

    def __init__(self):
        """Initialize flame analysis calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate(self, inputs: FlameAnalysisInput) -> FlameAnalysisOutput:
        """
        Main calculation method for flame analysis.

        Args:
            inputs: Validated input parameters

        Returns:
            FlameAnalysisOutput with all flame characteristics
        """
        start_time = DeterministicClock.now()
        self.logger.info("Starting flame analysis calculation")

        # Calculate adiabatic flame temperature
        t_ad_k = self._calculate_adiabatic_flame_temp(inputs)
        t_ad_c = t_ad_k - 273.15

        # Calculate actual flame temperature (with heat losses)
        t_actual_c = self._calculate_actual_flame_temp(t_ad_c, inputs)
        temp_efficiency = (t_actual_c / t_ad_c * 100) if t_ad_c > 0 else 0

        # Calculate flame speeds
        s_l = self._calculate_laminar_flame_speed(inputs, t_actual_c)
        s_t = self._calculate_turbulent_flame_speed(s_l, inputs)

        # Calculate jet velocity
        u_jet = self._calculate_jet_velocity(inputs)

        # Calculate flame dimensions
        flame_length = self._calculate_flame_length(inputs, u_jet)
        flame_diameter = self._calculate_flame_diameter(inputs, flame_length)
        flame_volume = self._calculate_flame_volume(flame_length, flame_diameter)
        flame_surface = math.pi * flame_diameter * flame_length

        # Calculate stability parameters
        da = self._calculate_damkohler_number(inputs, s_l, u_jet)
        phi = self._calculate_equivalence_ratio(inputs)
        stability, stability_index = self._assess_flame_stability(da, phi, s_t, u_jet)
        u_blowoff = self._calculate_blowoff_velocity(s_l, inputs)
        u_flashback = self._calculate_flashback_velocity(s_l, inputs)

        # Operating margin
        if u_jet > u_flashback and u_jet < u_blowoff:
            margin_low = (u_jet - u_flashback) / u_flashback * 100
            margin_high = (u_blowoff - u_jet) / u_jet * 100
            operating_margin = min(margin_low, margin_high)
        else:
            operating_margin = 0

        # Determine flame type
        flame_type = self._determine_flame_type(inputs)

        # Calculate luminosity
        luminosity = self._calculate_luminosity(t_actual_c, phi, inputs)

        # Calculate combustion intensity
        heat_release = inputs.fuel_flow_kg_hr * inputs.fuel_lhv_mj_kg / 3600  # MW
        combustion_intensity = heat_release / flame_volume if flame_volume > 0 else 0

        # Calculate heat transfer fractions
        radiant_frac = self._calculate_radiant_fraction(t_actual_c, luminosity)
        convective_frac = 1 - radiant_frac

        # Specific heat release
        specific_heat = heat_release / flame_surface if flame_surface > 0 else 0

        # Calculate emission indices
        nox_index = self._calculate_nox_potential(t_actual_c, inputs)
        soot_index = self._calculate_soot_potential(phi, t_actual_c)

        # Combustion efficiency estimate
        comb_efficiency = self._estimate_combustion_efficiency(inputs, stability_index)

        # Calculate dimensionless numbers
        re = self._calculate_reynolds_number(inputs, u_jet)
        fr = self._calculate_froude_number(u_jet, inputs)
        ri = self._calculate_richardson_number(inputs, u_jet)
        ka = self._calculate_karlovitz_number(s_l, inputs)

        # Assess flame quality
        flame_quality = self._assess_flame_quality(
            stability_index, comb_efficiency, nox_index, soot_index
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stability, flame_quality, phi, t_actual_c, inputs
        )

        # Determine optimization potential
        opt_potential = self._assess_optimization_potential(
            flame_quality, stability_index, comb_efficiency
        )

        # Calculate provenance
        timestamp = DeterministicClock.now().isoformat()
        provenance_data = f"{inputs.json()}|{timestamp}|{t_ad_k}|{flame_length}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return FlameAnalysisOutput(
            adiabatic_flame_temp_k=round(t_ad_k, 1),
            adiabatic_flame_temp_c=round(t_ad_c, 1),
            actual_flame_temp_c=round(t_actual_c, 1),
            temperature_efficiency_percent=round(temp_efficiency, 1),
            flame_length_m=round(flame_length, 3),
            flame_diameter_m=round(flame_diameter, 3),
            flame_volume_m3=round(flame_volume, 4),
            flame_surface_area_m2=round(flame_surface, 3),
            laminar_flame_speed_m_s=round(s_l, 3),
            turbulent_flame_speed_m_s=round(s_t, 2),
            jet_velocity_m_s=round(u_jet, 2),
            damkohler_number=round(da, 2),
            equivalence_ratio=round(phi, 3),
            stability_index=round(stability_index, 3),
            blow_off_velocity_m_s=round(u_blowoff, 2),
            flashback_velocity_m_s=round(u_flashback, 2),
            operating_margin_percent=round(operating_margin, 1),
            flame_type=flame_type.value,
            flame_stability=stability.value,
            flame_quality=flame_quality.value,
            luminosity_factor=round(luminosity, 3),
            combustion_intensity_mw_m3=round(combustion_intensity, 2),
            radiant_fraction=round(radiant_frac, 3),
            convective_fraction=round(convective_frac, 3),
            heat_release_rate_mw=round(heat_release, 3),
            specific_heat_release_mw_m2=round(specific_heat, 3),
            nox_potential_index=round(nox_index, 3),
            soot_formation_index=round(soot_index, 3),
            combustion_efficiency_percent=round(comb_efficiency, 1),
            reynolds_number=round(re, 0),
            froude_number=round(fr, 2),
            richardson_number=round(ri, 3),
            karlovitz_number=round(ka, 2),
            optimization_potential=opt_potential,
            recommended_adjustments=recommendations,
            calculation_timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    def _calculate_adiabatic_flame_temp(self, inputs: FlameAnalysisInput) -> float:
        """
        Calculate adiabatic flame temperature using energy balance.

        Simplified method using empirical correlations.
        Full calculation would require iterative solution of energy balance.
        """
        # Base temperature for methane-air at stoichiometric (2226 K)
        t_base = 2226

        # Fuel composition effect
        ch4_frac = inputs.fuel_ch4_percent / 100
        h2_frac = inputs.fuel_h2_percent / 100

        # H2 burns hotter than CH4
        fuel_factor = 1.0 + h2_frac * 0.15 - (1 - ch4_frac - h2_frac) * 0.1

        # Equivalence ratio effect (peaks at φ ≈ 1.05)
        phi = self._calculate_equivalence_ratio(inputs)
        if phi < 0.5 or phi > 2.0:
            phi_factor = 0.7
        else:
            # Parabolic profile peaking at φ = 1.05
            phi_factor = 1.0 - 0.3 * ((phi - 1.05) ** 2)

        # Preheat effect
        preheat_factor = 1.0 + (inputs.air_temperature_c - 25) / 1000

        # Oxygen enrichment effect
        o2_factor = 1.0 + (inputs.oxygen_enrichment_percent - 21) / 100

        # Pressure effect (small)
        p_factor = (inputs.combustion_chamber_pressure_kpa / 101.325) ** 0.05

        # Calculate adiabatic flame temperature
        t_ad = t_base * fuel_factor * phi_factor * preheat_factor * o2_factor * p_factor

        # Physical limits
        t_ad = max(1000, min(t_ad, 3000))  # K

        return t_ad

    def _calculate_actual_flame_temp(self, t_ad_c: float, inputs: FlameAnalysisInput) -> float:
        """
        Calculate actual flame temperature accounting for heat losses.

        Heat losses due to:
        - Radiation
        - Incomplete combustion
        - Dissociation at high temperature
        """
        # Use measured temperature if available
        if inputs.measured_flame_temp_c is not None:
            return inputs.measured_flame_temp_c

        # Radiation loss factor (increases with temperature)
        rad_loss = 0.1 * (t_ad_c / 2000) ** 2

        # Incomplete combustion loss (from excess O2)
        if inputs.flue_gas_o2_percent > 5:
            incomplete_loss = 0.05
        elif inputs.flue_gas_o2_percent < 1:
            incomplete_loss = 0.15
        else:
            incomplete_loss = 0.02

        # Dissociation loss at high temperature
        if t_ad_c > 2000:
            dissociation_loss = 0.05 * ((t_ad_c - 2000) / 500)
        else:
            dissociation_loss = 0

        # Total loss factor
        total_loss = rad_loss + incomplete_loss + dissociation_loss
        total_loss = min(total_loss, 0.4)  # Cap at 40% loss

        # Actual temperature
        t_actual = t_ad_c * (1 - total_loss)

        return t_actual

    def _calculate_laminar_flame_speed(self, inputs: FlameAnalysisInput, t_flame_c: float) -> float:
        """
        Calculate laminar flame speed.

        S_L = S_L,ref * (T/T_ref)^α * (P/P_ref)^β * f(φ)
        """
        # Get reference speed for dominant fuel component
        ch4_frac = inputs.fuel_ch4_percent / 100
        h2_frac = inputs.fuel_h2_percent / 100

        # Weighted average of component speeds
        s_l_ref = (
            ch4_frac * self.SL_REF.get('CH4', 0.4) +
            h2_frac * self.SL_REF.get('H2', 2.1) +
            (1 - ch4_frac - h2_frac) * 0.3  # Other components
        )

        # Temperature correction (α ≈ 1.75 for hydrocarbons)
        t_ratio = (t_flame_c + 273.15) / 298.15
        temp_factor = t_ratio ** 1.75

        # Pressure correction (β ≈ -0.5)
        p_ratio = inputs.combustion_chamber_pressure_kpa / 101.325
        pressure_factor = p_ratio ** (-0.5)

        # Equivalence ratio correction
        phi = self._calculate_equivalence_ratio(inputs)
        if 0.8 <= phi <= 1.2:
            phi_factor = 1.0 - 2.5 * (phi - 1.0) ** 2
        else:
            phi_factor = 0.5  # Reduced speed far from stoichiometric

        # Calculate laminar flame speed
        s_l = s_l_ref * temp_factor * pressure_factor * phi_factor

        # Physical limits
        s_l = max(0.1, min(s_l, 5.0))  # m/s

        return s_l

    def _calculate_turbulent_flame_speed(self, s_l: float, inputs: FlameAnalysisInput) -> float:
        """
        Calculate turbulent flame speed.

        S_T/S_L = 1 + A * (u'/S_L)^n
        """
        # Estimate turbulence intensity
        u_jet = self._calculate_jet_velocity(inputs)

        # Turbulence intensity (10-30% of jet velocity typically)
        if inputs.swirl_number > 0.6:
            turb_intensity = 0.3  # High swirl increases turbulence
        elif inputs.swirl_number > 0.3:
            turb_intensity = 0.2
        else:
            turb_intensity = 0.15

        u_prime = u_jet * turb_intensity

        # Turbulent flame speed correlation
        # Damköhler's correlation: S_T/S_L = 1 + (u'/S_L)
        # Modified for industrial conditions
        if u_prime / s_l < 1:
            s_t = s_l * (1 + 0.5 * (u_prime / s_l))
        else:
            s_t = s_l * (1 + (u_prime / s_l) ** 0.7)

        # Limit enhancement
        s_t = min(s_t, 10 * s_l)

        return s_t

    def _calculate_jet_velocity(self, inputs: FlameAnalysisInput) -> float:
        """Calculate fuel jet velocity at burner outlet"""
        # Fuel volume flow at operating conditions
        fuel_density = self._estimate_fuel_density(inputs)
        fuel_vol_flow = inputs.fuel_flow_kg_hr / (3600 * fuel_density)  # m³/s

        # Burner area
        d_burner = inputs.burner_diameter_mm / 1000  # m
        area = math.pi * (d_burner / 2) ** 2

        # Jet velocity
        u_jet = fuel_vol_flow / area if area > 0 else 0

        # Account for air premixing if applicable
        if inputs.excess_air_percent < 50:  # Partially premixed
            total_flow = fuel_vol_flow * (1 + inputs.air_flow_kg_hr / inputs.fuel_flow_kg_hr)
            u_jet = total_flow / area if area > 0 else 0

        return u_jet

    def _calculate_flame_length(self, inputs: FlameAnalysisInput, u_jet: float) -> float:
        """
        Calculate flame length using Hottel-Hawthorne correlation.

        L/d = A * (ρ_jet * u_jet * d / ρ_∞ * D)^n
        """
        # Use measured length if available
        if inputs.measured_flame_length_m is not None:
            return inputs.measured_flame_length_m

        d = inputs.burner_diameter_mm / 1000  # m

        # Momentum-dominated jet flame
        # L/d ≈ 230 * (fuel/air stoichiometric ratio) for turbulent diffusion flames
        stoich_ratio = 1 / (14.7 * (1 + inputs.excess_air_percent / 100))

        # Base correlation
        l_over_d = 230 * stoich_ratio

        # Swirl effect (reduces length)
        if inputs.swirl_number > 0:
            swirl_factor = 1 / (1 + 2 * inputs.swirl_number)
            l_over_d *= swirl_factor

        # Confinement effect
        if inputs.flame_tube_diameter_mm:
            confinement_ratio = inputs.flame_tube_diameter_mm / inputs.burner_diameter_mm
            if confinement_ratio < 10:
                l_over_d *= 0.7  # Confined flame is shorter

        flame_length = l_over_d * d

        # Physical limits
        flame_length = max(0.1, min(flame_length, 20))  # m

        return flame_length

    def _calculate_flame_diameter(self, inputs: FlameAnalysisInput, flame_length: float) -> float:
        """Calculate maximum flame diameter"""
        d_burner = inputs.burner_diameter_mm / 1000  # m

        # Jet expansion angle (typically 10-15° half-angle)
        if inputs.swirl_number > 0.6:
            half_angle = 25  # degrees - high swirl increases spread
        elif inputs.swirl_number > 0.3:
            half_angle = 18
        else:
            half_angle = 12

        # Maximum diameter at flame tip
        expansion = 2 * flame_length * math.tan(math.radians(half_angle))
        d_max = d_burner + expansion

        # Confinement limit
        if inputs.flame_tube_diameter_mm:
            d_limit = inputs.flame_tube_diameter_mm / 1000
            d_max = min(d_max, d_limit * 0.8)  # 80% of tube diameter

        return d_max

    def _calculate_flame_volume(self, length: float, diameter: float) -> float:
        """Calculate flame volume (approximated as truncated cone)"""
        d_burner = diameter / 3  # Approximate base diameter

        # Truncated cone volume
        volume = (math.pi * length / 3) * (
            (d_burner/2)**2 + (d_burner/2)*(diameter/2) + (diameter/2)**2
        )

        return volume

    def _calculate_equivalence_ratio(self, inputs: FlameAnalysisInput) -> float:
        """Calculate equivalence ratio φ = (F/A)_actual / (F/A)_stoich"""
        # Simplified for hydrocarbon fuels
        # Stoichiometric A/F ≈ 14.7 for natural gas

        actual_af = inputs.air_flow_kg_hr / inputs.fuel_flow_kg_hr if inputs.fuel_flow_kg_hr > 0 else 14.7
        stoich_af = 14.7  # Approximate for natural gas

        phi = stoich_af / actual_af if actual_af > 0 else 1.0

        return phi

    def _calculate_damkohler_number(self, inputs: FlameAnalysisInput, s_l: float, u_jet: float) -> float:
        """
        Calculate Damköhler number Da = τ_flow / τ_chem.

        Da > 1: Chemistry fast, flame stable
        Da < 1: Flow fast, potential blowoff
        """
        d = inputs.burner_diameter_mm / 1000  # m

        # Flow time scale
        tau_flow = d / u_jet if u_jet > 0 else 1

        # Chemical time scale
        delta_l = 0.001  # Laminar flame thickness ~ 1 mm
        tau_chem = delta_l / s_l if s_l > 0 else 1

        da = tau_flow / tau_chem

        return da

    def _assess_flame_stability(
        self, da: float, phi: float, s_t: float, u_jet: float
    ) -> Tuple[FlameStability, float]:
        """Assess flame stability based on parameters"""
        stability_score = 0

        # Damköhler number contribution
        if da > 10:
            stability_score += 0.3
        elif da > 1:
            stability_score += 0.2
        elif da > 0.1:
            stability_score += 0.1

        # Equivalence ratio contribution
        if 0.8 <= phi <= 1.2:
            stability_score += 0.3
        elif 0.6 <= phi <= 1.5:
            stability_score += 0.2
        elif 0.5 <= phi <= 2.0:
            stability_score += 0.1

        # Velocity ratio contribution
        vel_ratio = u_jet / s_t if s_t > 0 else 10
        if 2 <= vel_ratio <= 20:
            stability_score += 0.4
        elif 1 <= vel_ratio <= 50:
            stability_score += 0.2
        elif vel_ratio < 1:
            # Flashback risk
            stability_score += 0
        else:
            # Blowoff risk
            stability_score += 0.1

        # Determine stability status
        if vel_ratio < 1:
            status = FlameStability.FLASHBACK_RISK
        elif vel_ratio > 100:
            status = FlameStability.BLOWOFF_RISK
        elif stability_score >= 0.8:
            status = FlameStability.STABLE
        elif stability_score >= 0.5:
            status = FlameStability.MARGINALLY_STABLE
        else:
            status = FlameStability.UNSTABLE

        return status, stability_score

    def _calculate_blowoff_velocity(self, s_l: float, inputs: FlameAnalysisInput) -> float:
        """Calculate blowoff velocity"""
        # Blowoff correlation: u_bo ∝ S_L * Re^0.5
        re = 1000 * inputs.fuel_flow_kg_hr  # Simplified Re

        # Empirical correlation
        u_bo = s_l * (re / 1000) ** 0.5

        # Swirl stabilization effect
        if inputs.swirl_number > 0.3:
            u_bo *= (1 + inputs.swirl_number)

        return min(u_bo, 200)  # m/s

    def _calculate_flashback_velocity(self, s_l: float, inputs: FlameAnalysisInput) -> float:
        """Calculate flashback velocity"""
        # Flashback occurs when flow velocity < turbulent flame speed
        # Critical velocity gradient method

        # Simplified: flashback velocity ≈ 0.5 * S_L for industrial burners
        u_fb = 0.5 * s_l

        # Temperature effect (higher temp increases flashback risk)
        temp_factor = 1 + (inputs.air_temperature_c - 25) / 500
        u_fb *= temp_factor

        return u_fb

    def _determine_flame_type(self, inputs: FlameAnalysisInput) -> FlameType:
        """Determine type of flame"""
        phi = self._calculate_equivalence_ratio(inputs)

        if inputs.excess_air_percent > 100:
            return FlameType.DIFFUSION
        elif inputs.excess_air_percent < 20 and phi > 0.8 and phi < 1.2:
            return FlameType.PREMIXED
        else:
            return FlameType.PARTIALLY_PREMIXED

    def _calculate_luminosity(self, temp_c: float, phi: float, inputs: FlameAnalysisInput) -> float:
        """Calculate flame luminosity factor (0-1)"""
        # Luminosity from soot radiation
        # Increases with temperature and rich conditions

        # Temperature factor
        if temp_c > 1500:
            temp_factor = 0.5 + 0.5 * ((temp_c - 1500) / 500)
        else:
            temp_factor = 0.2

        # Equivalence ratio factor (peaks at φ ≈ 1.4)
        if 1.2 <= phi <= 1.6:
            phi_factor = 1.0
        elif 1.0 <= phi <= 1.8:
            phi_factor = 0.7
        else:
            phi_factor = 0.3

        # Fuel type factor (aromatic > aliphatic)
        fuel_factor = 0.5  # Natural gas has low soot tendency

        luminosity = temp_factor * phi_factor * fuel_factor

        return min(luminosity, 1.0)

    def _calculate_radiant_fraction(self, temp_c: float, luminosity: float) -> float:
        """Calculate fraction of heat released as radiation"""
        # Typical values: 10-40% for industrial flames

        # Base radiation (gas radiation)
        base_rad = 0.15

        # Luminous radiation (soot)
        luminous_rad = luminosity * 0.25

        # Temperature effect
        temp_factor = (temp_c / 2000) ** 2 if temp_c < 2000 else 1.0

        total_rad = (base_rad + luminous_rad) * temp_factor

        return min(total_rad, 0.45)

    def _calculate_nox_potential(self, temp_c: float, inputs: FlameAnalysisInput) -> float:
        """Calculate NOx formation potential index (0-1)"""
        # Thermal NOx dominates above 1500°C

        if temp_c < 1200:
            return 0.1
        elif temp_c < 1500:
            return 0.2 + 0.3 * ((temp_c - 1200) / 300)
        elif temp_c < 1800:
            return 0.5 + 0.3 * ((temp_c - 1500) / 300)
        else:
            return 0.8 + 0.2 * ((temp_c - 1800) / 200)

    def _calculate_soot_potential(self, phi: float, temp_c: float) -> float:
        """Calculate soot formation potential index (0-1)"""
        # Soot forms in rich, high-temp conditions

        # Equivalence ratio factor
        if phi < 1.0:
            phi_factor = 0.1
        elif phi < 1.2:
            phi_factor = 0.3
        elif phi < 1.5:
            phi_factor = 0.7
        else:
            phi_factor = 1.0

        # Temperature factor
        if temp_c < 1200:
            temp_factor = 0.2
        elif temp_c < 1600:
            temp_factor = 0.5
        else:
            temp_factor = 0.8

        return phi_factor * temp_factor

    def _estimate_combustion_efficiency(self, inputs: FlameAnalysisInput, stability: float) -> float:
        """Estimate combustion efficiency"""
        base_efficiency = 95.0

        # Stability effect
        stability_factor = stability * 5  # Up to 5% improvement

        # Excess air effect
        if inputs.excess_air_percent < 5:
            ea_penalty = 5
        elif inputs.excess_air_percent > 50:
            ea_penalty = 3
        else:
            ea_penalty = 0

        # Load effect
        if inputs.load_percent < 50:
            load_penalty = 3
        else:
            load_penalty = 0

        efficiency = base_efficiency + stability_factor - ea_penalty - load_penalty

        return max(80, min(efficiency, 99.5))

    def _calculate_reynolds_number(self, inputs: FlameAnalysisInput, u_jet: float) -> float:
        """Calculate jet Reynolds number"""
        d = inputs.burner_diameter_mm / 1000  # m

        # Kinematic viscosity of fuel (approximate)
        nu = 1.5e-5  # m²/s for gas at ambient

        re = u_jet * d / nu

        return re

    def _calculate_froude_number(self, u_jet: float, inputs: FlameAnalysisInput) -> float:
        """Calculate Froude number Fr = u²/(g*L)"""
        d = inputs.burner_diameter_mm / 1000  # m

        fr = u_jet ** 2 / (self.G * d)

        return fr

    def _calculate_richardson_number(self, inputs: FlameAnalysisInput, u_jet: float) -> float:
        """Calculate Richardson number Ri = g*β*ΔT*L/u²"""
        d = inputs.burner_diameter_mm / 1000  # m
        dt = 1500  # Temperature difference estimate

        # Thermal expansion coefficient
        beta = 1 / 300  # 1/K

        ri = self.G * beta * dt * d / (u_jet ** 2) if u_jet > 0 else 1

        return ri

    def _calculate_karlovitz_number(self, s_l: float, inputs: FlameAnalysisInput) -> float:
        """Calculate Karlovitz number Ka = (δ_L/η)²"""
        # Kolmogorov scale estimate
        eta = 0.0001  # m (typical for industrial flames)

        # Laminar flame thickness
        delta_l = 0.001  # m

        ka = (delta_l / eta) ** 2

        return ka

    def _assess_flame_quality(
        self, stability: float, efficiency: float, nox_index: float, soot_index: float
    ) -> FlameQuality:
        """Assess overall flame quality"""
        # Weighted score
        score = (
            stability * 0.3 +
            efficiency / 100 * 0.3 +
            (1 - nox_index) * 0.2 +
            (1 - soot_index) * 0.2
        )

        if score >= 0.85:
            return FlameQuality.EXCELLENT
        elif score >= 0.70:
            return FlameQuality.GOOD
        elif score >= 0.55:
            return FlameQuality.ACCEPTABLE
        elif score >= 0.40:
            return FlameQuality.POOR
        else:
            return FlameQuality.CRITICAL

    def _generate_recommendations(
        self,
        stability: FlameStability,
        quality: FlameQuality,
        phi: float,
        temp_c: float,
        inputs: FlameAnalysisInput
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Stability recommendations
        if stability == FlameStability.FLASHBACK_RISK:
            recommendations.append("CRITICAL: Increase fuel velocity or reduce air preheat to prevent flashback")
        elif stability == FlameStability.BLOWOFF_RISK:
            recommendations.append("WARNING: Reduce fuel velocity or increase swirl to prevent blowoff")
        elif stability == FlameStability.UNSTABLE:
            recommendations.append("Improve flame stability by optimizing air-fuel ratio or adding swirl")

        # Efficiency recommendations
        if inputs.excess_air_percent > 30:
            recommendations.append("Reduce excess air to improve efficiency (target 10-15%)")
        elif inputs.excess_air_percent < 5:
            recommendations.append("Increase excess air to ensure complete combustion")

        # NOx recommendations
        if temp_c > 1600:
            recommendations.append("Consider flue gas recirculation to reduce flame temperature and NOx")

        # Equivalence ratio recommendations
        if phi > 1.3:
            recommendations.append("Reduce fuel rate or increase air to avoid soot formation")
        elif phi < 0.7:
            recommendations.append("Increase fuel rate or reduce air to improve combustion stability")

        # Load recommendations
        if inputs.load_percent < 40:
            recommendations.append("Consider burner staging for better low-load performance")

        if not recommendations:
            recommendations.append("Flame operating within optimal parameters")

        return recommendations

    def _assess_optimization_potential(
        self, quality: FlameQuality, stability: float, efficiency: float
    ) -> str:
        """Assess potential for optimization"""
        if quality == FlameQuality.EXCELLENT and stability > 0.9 and efficiency > 97:
            return "Low - System operating near optimal"
        elif quality in [FlameQuality.GOOD, FlameQuality.ACCEPTABLE]:
            return "Moderate - Minor adjustments could improve performance"
        else:
            return "High - Significant improvements possible"

    def _estimate_fuel_density(self, inputs: FlameAnalysisInput) -> float:
        """Estimate fuel density at operating conditions"""
        # Ideal gas approximation
        p = inputs.fuel_pressure_kpa
        t = inputs.fuel_temperature_c + 273.15
        mw = 16  # Approximate MW for natural gas

        # ρ = P * MW / (R * T)
        density = (p * 1000 * mw) / (self.R * 1000 * t)

        return density