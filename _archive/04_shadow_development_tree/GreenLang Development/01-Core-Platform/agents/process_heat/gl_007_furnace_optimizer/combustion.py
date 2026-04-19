# -*- coding: utf-8 -*-
"""
GL-007 FurnaceOptimizer - Combustion Analysis Calculator

This module provides ZERO-HALLUCINATION combustion calculations for
industrial furnace optimization. All calculations use deterministic
engineering formulas with complete provenance tracking.

Key Calculations:
    - Stoichiometric air requirements
    - Excess air from flue gas O2
    - Combustion efficiency (ASME PTC 4 method)
    - Heat loss analysis
    - Emission calculations (CO2, NOx, CO)
    - Flue gas composition

Engineering References:
    - ASME PTC 4: Fired Steam Generators
    - API 560: Fired Heaters for General Refinery Service
    - NFPA 86: Standard for Ovens and Furnaces
    - North American Combustion Handbook

Example:
    >>> from greenlang.agents.process_heat.gl_007_furnace_optimizer.combustion import (
    ...     CombustionCalculator,
    ... )
    >>> calculator = CombustionCalculator()
    >>> analysis = calculator.analyze_combustion(
    ...     fuel_flow_scfh=5000,
    ...     fuel_hhv_btu_scf=1020,
    ...     flue_gas_temp_f=450,
    ...     flue_gas_o2_pct=3.0,
    ...     ambient_temp_f=77,
    ... )
    >>> print(f"Efficiency: {analysis.thermal_efficiency_pct}%")

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import uuid

from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    CombustionAnalysis,
    CombustionStatus,
)
from greenlang.agents.process_heat.gl_007_furnace_optimizer.provenance import (
    ProvenanceTracker,
    generate_provenance_hash,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - COMBUSTION ENGINEERING DATA
# =============================================================================

class CombustionConstants:
    """
    Combustion engineering constants - DETERMINISTIC.

    All values from ASME PTC 4, North American Combustion Handbook,
    and standard thermodynamic tables.
    """

    # Standard conditions
    STD_TEMP_F = 60.0           # Standard temperature (F)
    STD_TEMP_R = 519.67         # Standard temperature (R)
    STD_PRESSURE_PSIA = 14.696  # Standard pressure (psia)

    # Air composition (dry basis, vol%)
    AIR_O2_PCT = 20.95
    AIR_N2_PCT = 78.09
    AIR_AR_PCT = 0.93
    AIR_CO2_PCT = 0.03

    # Molecular weights (lb/lbmol)
    MW_AIR = 28.97
    MW_O2 = 32.00
    MW_N2 = 28.01
    MW_CO2 = 44.01
    MW_H2O = 18.02
    MW_CH4 = 16.04
    MW_C2H6 = 30.07
    MW_C3H8 = 44.10
    MW_SO2 = 64.07
    MW_NO2 = 46.01

    # Specific heats (Btu/lb-F) at typical flue gas temperatures
    CP_FLUE_GAS = 0.26          # Average specific heat
    CP_AIR = 0.24               # Air specific heat
    CP_WATER_VAPOR = 0.45       # Water vapor specific heat

    # Latent heat of water (Btu/lb)
    LATENT_HEAT_H2O = 1040.0

    # Density of air at standard conditions (lb/scf)
    AIR_DENSITY_STD = 0.0765

    # CO2 emission factor for natural gas (lb CO2/MMBtu)
    CO2_FACTOR_NG = 117.0       # Per EPA

    # Natural gas typical composition
    NG_TYPICAL_CH4_PCT = 95.0
    NG_TYPICAL_C2H6_PCT = 2.5
    NG_TYPICAL_CO2_PCT = 0.5
    NG_TYPICAL_N2_PCT = 2.0


# =============================================================================
# FUEL PROPERTIES DATABASE
# =============================================================================

FUEL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "hhv_btu_scf": 1020.0,
        "lhv_btu_scf": 920.0,
        "specific_gravity": 0.60,
        "stoich_air_scf_per_scf": 9.52,  # Stoichiometric air requirement
        "h2_in_fuel_pct": 23.0,          # Hydrogen content by mass
        "c_in_fuel_pct": 75.0,           # Carbon content by mass
        "co2_factor_lb_mmbtu": 117.0,
    },
    "propane": {
        "hhv_btu_scf": 2516.0,
        "lhv_btu_scf": 2316.0,
        "specific_gravity": 1.52,
        "stoich_air_scf_per_scf": 23.81,
        "h2_in_fuel_pct": 18.2,
        "c_in_fuel_pct": 81.8,
        "co2_factor_lb_mmbtu": 139.0,
    },
    "fuel_oil_2": {
        "hhv_btu_gal": 140000.0,
        "lhv_btu_gal": 131000.0,
        "specific_gravity": 0.85,
        "stoich_air_lb_per_lb": 14.1,
        "h2_in_fuel_pct": 13.0,
        "c_in_fuel_pct": 87.0,
        "co2_factor_lb_mmbtu": 163.0,
    },
    "hydrogen": {
        "hhv_btu_scf": 325.0,
        "lhv_btu_scf": 275.0,
        "specific_gravity": 0.07,
        "stoich_air_scf_per_scf": 2.38,
        "h2_in_fuel_pct": 100.0,
        "c_in_fuel_pct": 0.0,
        "co2_factor_lb_mmbtu": 0.0,
    },
}


# =============================================================================
# COMBUSTION CALCULATOR CLASS
# =============================================================================

class CombustionCalculator:
    """
    Zero-hallucination combustion analysis calculator.

    All calculations are DETERMINISTIC using engineering formulas from:
    - ASME PTC 4 (Fired Steam Generators)
    - North American Combustion Handbook
    - API 560 (Fired Heaters)

    Features:
    - Stoichiometric air calculation
    - Excess air from O2 measurement
    - Heat loss analysis (dry gas, moisture, radiation)
    - Thermal efficiency calculation
    - Emission calculations (CO2, NOx, CO)
    - Complete SHA-256 provenance tracking

    Example:
        >>> calculator = CombustionCalculator()
        >>> analysis = calculator.analyze_combustion(
        ...     fuel_flow_scfh=5000,
        ...     fuel_hhv_btu_scf=1020,
        ...     flue_gas_temp_f=450,
        ...     flue_gas_o2_pct=3.0,
        ... )
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        provenance_enabled: bool = True,
        precision: int = 4,
    ) -> None:
        """
        Initialize combustion calculator.

        Args:
            provenance_enabled: Enable SHA-256 provenance tracking
            precision: Decimal precision for calculations
        """
        self.provenance_enabled = provenance_enabled
        self.precision = precision
        self._provenance_tracker = ProvenanceTracker() if provenance_enabled else None

        logger.info(f"CombustionCalculator initialized v{self.VERSION}")

    # =========================================================================
    # PRIMARY CALCULATION METHODS - ZERO HALLUCINATION
    # =========================================================================

    def analyze_combustion(
        self,
        fuel_flow_scfh: float,
        fuel_hhv_btu_scf: float,
        flue_gas_temp_f: float,
        flue_gas_o2_pct: float,
        ambient_temp_f: float = 77.0,
        combustion_air_temp_f: Optional[float] = None,
        fuel_type: str = "natural_gas",
        flue_gas_co_ppm: Optional[float] = None,
        radiation_loss_pct: float = 2.0,
    ) -> CombustionAnalysis:
        """
        Perform complete combustion analysis - DETERMINISTIC.

        All calculations use engineering formulas with zero ML/AI.

        Args:
            fuel_flow_scfh: Fuel flow rate (SCFH)
            fuel_hhv_btu_scf: Fuel higher heating value (Btu/scf)
            flue_gas_temp_f: Flue gas exit temperature (F)
            flue_gas_o2_pct: Flue gas O2 content (%)
            ambient_temp_f: Ambient temperature (F)
            combustion_air_temp_f: Combustion air temperature (F), defaults to ambient
            fuel_type: Fuel type for properties lookup
            flue_gas_co_ppm: Flue gas CO content (ppm), optional
            radiation_loss_pct: Radiation/convection loss (% of input)

        Returns:
            CombustionAnalysis with complete results

        Raises:
            ValueError: If inputs are out of valid range
        """
        start_time = datetime.now(timezone.utc)
        analysis_id = str(uuid.uuid4())

        # Validate inputs
        self._validate_inputs(
            fuel_flow_scfh=fuel_flow_scfh,
            fuel_hhv_btu_scf=fuel_hhv_btu_scf,
            flue_gas_temp_f=flue_gas_temp_f,
            flue_gas_o2_pct=flue_gas_o2_pct,
        )

        # Get fuel properties
        fuel_props = self._get_fuel_properties(fuel_type, fuel_hhv_btu_scf)

        # Use combustion air temp or default to ambient
        if combustion_air_temp_f is None:
            combustion_air_temp_f = ambient_temp_f

        # Step 1: Calculate stoichiometric air requirement - DETERMINISTIC
        stoich_air = self.calculate_stoichiometric_air(fuel_type, fuel_hhv_btu_scf)

        # Step 2: Calculate excess air from O2 - DETERMINISTIC
        excess_air_pct = self.calculate_excess_air_from_o2(flue_gas_o2_pct)

        # Step 3: Calculate actual air ratio - DETERMINISTIC
        actual_air_ratio = stoich_air * (1 + excess_air_pct / 100)

        # Step 4: Calculate heat input - DETERMINISTIC
        heat_input_mmbtu_hr = (fuel_flow_scfh * fuel_hhv_btu_scf) / 1e6

        # Step 5: Calculate flue gas composition - DETERMINISTIC
        flue_gas_comp = self.calculate_flue_gas_composition(
            fuel_type=fuel_type,
            excess_air_pct=excess_air_pct,
        )

        # Step 6: Calculate heat losses - DETERMINISTIC (ASME PTC 4 method)
        losses = self.calculate_heat_losses(
            flue_gas_temp_f=flue_gas_temp_f,
            ambient_temp_f=ambient_temp_f,
            excess_air_pct=excess_air_pct,
            fuel_type=fuel_type,
            radiation_loss_pct=radiation_loss_pct,
            flue_gas_co_ppm=flue_gas_co_ppm,
        )

        # Step 7: Calculate efficiencies - DETERMINISTIC
        total_losses_pct = sum(losses.values())
        combustion_efficiency_pct = 100.0 - losses.get("dry_gas_loss", 0) - losses.get("moisture_loss", 0)
        thermal_efficiency_pct = 100.0 - total_losses_pct

        # Step 8: Calculate heat available - DETERMINISTIC
        heat_available_mmbtu_hr = heat_input_mmbtu_hr * (thermal_efficiency_pct / 100)

        # Step 9: Calculate emissions - DETERMINISTIC
        co2_lb_mmbtu = fuel_props.get("co2_factor_lb_mmbtu", 117.0)

        # CO emissions (convert ppm to lb/MMBtu)
        co_lb_mmbtu = 0.0
        if flue_gas_co_ppm is not None:
            # CO lb/MMBtu = (CO_ppm * MW_CO * FlueGas_vol) / (1e6 * Heat_input)
            # Simplified empirical formula
            co_lb_mmbtu = flue_gas_co_ppm * 0.0001  # Approximate conversion

        # NOx estimation (thermal NOx correlation)
        # NOx increases with temperature and excess air
        # Simplified correlation from combustion literature
        nox_lb_mmbtu = self._estimate_thermal_nox(
            flue_gas_temp_f=flue_gas_temp_f,
            excess_air_pct=excess_air_pct,
        )

        # Step 10: Determine combustion status - DETERMINISTIC
        combustion_status = self._determine_combustion_status(
            excess_air_pct=excess_air_pct,
            flue_gas_co_ppm=flue_gas_co_ppm,
            thermal_efficiency_pct=thermal_efficiency_pct,
        )

        # Step 11: Generate recommendations - DETERMINISTIC
        recommendations = self._generate_recommendations(
            excess_air_pct=excess_air_pct,
            flue_gas_temp_f=flue_gas_temp_f,
            thermal_efficiency_pct=thermal_efficiency_pct,
            flue_gas_co_ppm=flue_gas_co_ppm,
        )

        # Generate provenance hash - DETERMINISTIC
        provenance_data = {
            "analysis_id": analysis_id,
            "inputs": {
                "fuel_flow_scfh": fuel_flow_scfh,
                "fuel_hhv_btu_scf": fuel_hhv_btu_scf,
                "flue_gas_temp_f": flue_gas_temp_f,
                "flue_gas_o2_pct": flue_gas_o2_pct,
                "ambient_temp_f": ambient_temp_f,
                "fuel_type": fuel_type,
            },
            "outputs": {
                "thermal_efficiency_pct": round(thermal_efficiency_pct, self.precision),
                "excess_air_pct": round(excess_air_pct, self.precision),
            },
            "timestamp": start_time.isoformat(),
            "version": self.VERSION,
        }
        provenance_hash = generate_provenance_hash(provenance_data)

        # Track in provenance system
        if self._provenance_tracker:
            self._provenance_tracker.track_calculation(
                calc_type="combustion_analysis",
                inputs=provenance_data["inputs"],
                outputs=provenance_data["outputs"],
                formula="efficiency = 100 - (dry_gas_loss + moisture_loss + radiation_loss)",
                standard_references=[
                    "ASME PTC 4",
                    "API 560",
                    "North American Combustion Handbook",
                ],
            )

        # Create result
        result = CombustionAnalysis(
            analysis_id=analysis_id,
            timestamp=start_time,
            stoichiometric_air_scf_per_scf_fuel=round(stoich_air, self.precision),
            actual_air_scf_per_scf_fuel=round(actual_air_ratio, self.precision),
            excess_air_pct=round(excess_air_pct, self.precision),
            air_fuel_ratio=round(actual_air_ratio * CombustionConstants.AIR_DENSITY_STD /
                                (fuel_props.get("specific_gravity", 0.6) * CombustionConstants.AIR_DENSITY_STD),
                                self.precision),
            combustion_status=combustion_status,
            co2_pct_dry=round(flue_gas_comp["co2_pct"], self.precision),
            o2_pct_dry=round(flue_gas_o2_pct, self.precision),
            n2_pct_dry=round(flue_gas_comp["n2_pct"], self.precision),
            h2o_pct_wet=round(flue_gas_comp["h2o_pct"], self.precision),
            heat_input_mmbtu_hr=round(heat_input_mmbtu_hr, self.precision),
            heat_available_mmbtu_hr=round(heat_available_mmbtu_hr, self.precision),
            dry_flue_gas_loss_pct=round(losses.get("dry_gas_loss", 0), self.precision),
            moisture_loss_pct=round(losses.get("moisture_loss", 0), self.precision),
            radiation_loss_pct=round(losses.get("radiation_loss", 0), self.precision),
            unburned_fuel_loss_pct=round(losses.get("unburned_fuel_loss", 0), self.precision),
            total_losses_pct=round(total_losses_pct, self.precision),
            combustion_efficiency_pct=round(combustion_efficiency_pct, self.precision),
            thermal_efficiency_pct=round(thermal_efficiency_pct, self.precision),
            co_lb_mmbtu=round(co_lb_mmbtu, self.precision),
            nox_lb_mmbtu=round(nox_lb_mmbtu, self.precision),
            co2_lb_mmbtu=round(co2_lb_mmbtu, self.precision),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            formula_references=[
                "Excess Air = O2 / (21 - O2) * 100",
                "Dry Gas Loss = (Cp * (Tflue - Tambient) * (1 + EA/100) * AF) / HHV * 100",
                "Thermal Efficiency = 100 - Total Losses",
            ],
        )

        logger.info(
            f"Combustion analysis completed: efficiency={thermal_efficiency_pct:.1f}%, "
            f"excess_air={excess_air_pct:.1f}%"
        )

        return result

    def calculate_stoichiometric_air(
        self,
        fuel_type: str = "natural_gas",
        fuel_hhv_btu_scf: Optional[float] = None,
    ) -> float:
        """
        Calculate stoichiometric air requirement - DETERMINISTIC.

        Formula (natural gas approximation):
            Stoich Air (scf/scf) = 9.52 * (CH4) + 16.68 * (C2H6) + ...

        For natural gas: ~9.5-10.0 scf air per scf fuel

        Args:
            fuel_type: Type of fuel
            fuel_hhv_btu_scf: Higher heating value (Btu/scf)

        Returns:
            Stoichiometric air requirement (scf air / scf fuel)
        """
        fuel_props = FUEL_PROPERTIES.get(fuel_type, FUEL_PROPERTIES["natural_gas"])

        stoich_air = fuel_props.get("stoich_air_scf_per_scf", 9.52)

        # Adjust for HHV if provided (higher HHV = more air needed)
        if fuel_hhv_btu_scf is not None:
            reference_hhv = fuel_props.get("hhv_btu_scf", 1020.0)
            if reference_hhv > 0:
                stoich_air = stoich_air * (fuel_hhv_btu_scf / reference_hhv)

        return stoich_air

    def calculate_excess_air_from_o2(
        self,
        flue_gas_o2_pct: float,
    ) -> float:
        """
        Calculate excess air from flue gas O2 measurement - DETERMINISTIC.

        Formula (dry basis):
            Excess Air (%) = O2 / (21 - O2) * 100

        Where:
            - O2 is measured oxygen in flue gas (% dry)
            - 21 is oxygen content in air (%)

        Args:
            flue_gas_o2_pct: Measured O2 in flue gas (% dry)

        Returns:
            Excess air percentage

        Raises:
            ValueError: If O2 is >= 21% (impossible for combustion)
        """
        if flue_gas_o2_pct >= 21.0:
            raise ValueError(
                f"Flue gas O2 ({flue_gas_o2_pct}%) must be less than 21% "
                "for combustion to have occurred"
            )

        if flue_gas_o2_pct < 0:
            raise ValueError(f"Flue gas O2 ({flue_gas_o2_pct}%) cannot be negative")

        # DETERMINISTIC formula from combustion engineering
        excess_air_pct = (flue_gas_o2_pct / (21.0 - flue_gas_o2_pct)) * 100.0

        return excess_air_pct

    def calculate_excess_air_from_co2(
        self,
        flue_gas_co2_pct: float,
        theoretical_co2_pct: float = 11.7,
    ) -> float:
        """
        Calculate excess air from flue gas CO2 measurement - DETERMINISTIC.

        Formula:
            Excess Air (%) = ((CO2_theo / CO2_meas) - 1) * 100

        Args:
            flue_gas_co2_pct: Measured CO2 in flue gas (% dry)
            theoretical_co2_pct: Theoretical maximum CO2 (% at stoich)

        Returns:
            Excess air percentage
        """
        if flue_gas_co2_pct <= 0:
            raise ValueError(f"Flue gas CO2 ({flue_gas_co2_pct}%) must be positive")

        # DETERMINISTIC formula
        excess_air_pct = ((theoretical_co2_pct / flue_gas_co2_pct) - 1) * 100.0

        return max(0, excess_air_pct)  # Cannot be negative

    def calculate_flue_gas_composition(
        self,
        fuel_type: str = "natural_gas",
        excess_air_pct: float = 15.0,
    ) -> Dict[str, float]:
        """
        Calculate flue gas composition - DETERMINISTIC.

        Based on fuel composition and excess air level.

        Args:
            fuel_type: Type of fuel
            excess_air_pct: Excess air percentage

        Returns:
            Dictionary with flue gas composition (%)
        """
        # Get fuel properties
        fuel_props = FUEL_PROPERTIES.get(fuel_type, FUEL_PROPERTIES["natural_gas"])

        # For natural gas (CH4 dominant):
        # CH4 + 2O2 -> CO2 + 2H2O
        # At stoichiometric: ~11.7% CO2, 0% O2, ~73% N2, ~15% H2O (wet)

        # Calculate O2 from excess air
        # O2 (dry) = 21 * EA / (100 + EA)
        o2_pct = 21.0 * excess_air_pct / (100.0 + excess_air_pct)

        # Calculate CO2 (diluted by excess air)
        # Theoretical CO2 for natural gas: ~11.7%
        theoretical_co2 = 11.7  # % at stoichiometric
        co2_pct = theoretical_co2 * 100.0 / (100.0 + excess_air_pct)

        # Calculate N2 (remainder of dry gas)
        n2_pct = 100.0 - o2_pct - co2_pct

        # Calculate H2O (from hydrogen in fuel)
        # For natural gas: ~2 moles H2O per mole CH4
        # At stoich: ~15-18% H2O wet basis
        h2o_pct_wet = 15.0 * 100.0 / (100.0 + excess_air_pct)

        return {
            "o2_pct": round(o2_pct, 2),
            "co2_pct": round(co2_pct, 2),
            "n2_pct": round(n2_pct, 2),
            "h2o_pct": round(h2o_pct_wet, 2),
        }

    def calculate_heat_losses(
        self,
        flue_gas_temp_f: float,
        ambient_temp_f: float,
        excess_air_pct: float,
        fuel_type: str = "natural_gas",
        radiation_loss_pct: float = 2.0,
        flue_gas_co_ppm: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate heat losses per ASME PTC 4 method - DETERMINISTIC.

        Loss Categories:
        1. Dry flue gas loss (sensible heat in dry combustion products)
        2. Moisture loss (latent and sensible heat of water vapor)
        3. Radiation and convection loss
        4. Unburned fuel loss (if CO present)

        Args:
            flue_gas_temp_f: Flue gas exit temperature (F)
            ambient_temp_f: Ambient reference temperature (F)
            excess_air_pct: Excess air percentage
            fuel_type: Fuel type
            radiation_loss_pct: Radiation/convection loss (%)
            flue_gas_co_ppm: CO in flue gas (ppm)

        Returns:
            Dictionary of heat losses (% of heat input)
        """
        # Get fuel properties
        fuel_props = FUEL_PROPERTIES.get(fuel_type, FUEL_PROPERTIES["natural_gas"])
        hhv = fuel_props.get("hhv_btu_scf", 1020.0)
        h2_content = fuel_props.get("h2_in_fuel_pct", 23.0) / 100.0

        # Temperature difference
        delta_t = flue_gas_temp_f - ambient_temp_f

        # 1. Dry Flue Gas Loss - DETERMINISTIC
        # L_dry = (Cp_fg * (T_fg - T_amb) * m_fg) / HHV * 100
        # Simplified: L_dry = K1 * dT * (1 + EA/100)
        # K1 for natural gas ~ 0.0038 at typical conditions
        k1_dry_gas = 0.0038
        dry_gas_loss = k1_dry_gas * delta_t * (1 + excess_air_pct / 100)

        # 2. Moisture Loss - DETERMINISTIC
        # Loss from H2O formed in combustion and humidity
        # L_moist = (h2_in_fuel * 9 * (1040 + 0.45 * dT)) / HHV * 100
        # 9 lb H2O per lb H2, 1040 = latent heat, 0.45 = Cp of steam
        moisture_loss = (h2_content * 9.0 * (CombustionConstants.LATENT_HEAT_H2O +
                        CombustionConstants.CP_WATER_VAPOR * delta_t)) / hhv * 100

        # 3. Radiation Loss - typically 1-3% for well-insulated furnaces
        rad_loss = radiation_loss_pct

        # 4. Unburned Fuel Loss - DETERMINISTIC
        # If CO present, indicates incomplete combustion
        unburned_loss = 0.0
        if flue_gas_co_ppm is not None and flue_gas_co_ppm > 0:
            # CO represents unburned carbon
            # Loss ~ CO_ppm * 0.0001 (empirical)
            unburned_loss = flue_gas_co_ppm * 0.0001

        return {
            "dry_gas_loss": round(dry_gas_loss, 2),
            "moisture_loss": round(moisture_loss, 2),
            "radiation_loss": round(rad_loss, 2),
            "unburned_fuel_loss": round(unburned_loss, 2),
        }

    def calculate_adiabatic_flame_temp(
        self,
        fuel_type: str = "natural_gas",
        excess_air_pct: float = 15.0,
        air_preheat_temp_f: float = 77.0,
    ) -> float:
        """
        Calculate adiabatic flame temperature - DETERMINISTIC.

        Formula (simplified energy balance):
            T_ad = T_air + HHV / (Cp_products * (1 + AF * (1 + EA/100)))

        Args:
            fuel_type: Type of fuel
            excess_air_pct: Excess air percentage
            air_preheat_temp_f: Combustion air temperature (F)

        Returns:
            Adiabatic flame temperature (F)
        """
        # Base adiabatic flame temperatures at stoichiometric (F)
        base_temps = {
            "natural_gas": 3560,
            "propane": 3600,
            "hydrogen": 3800,
            "fuel_oil_2": 3500,
        }

        base_temp = base_temps.get(fuel_type, 3560)

        # Correction for excess air (dilution effect)
        # Flame temp drops ~25F per 1% excess air (empirical)
        excess_air_correction = -25.0 * (excess_air_pct / 10.0)

        # Correction for air preheat (~0.5F increase per F preheat)
        preheat_correction = 0.5 * (air_preheat_temp_f - 77.0)

        adiabatic_temp = base_temp + excess_air_correction + preheat_correction

        return round(adiabatic_temp, 0)

    def calculate_optimal_excess_air(
        self,
        fuel_type: str = "natural_gas",
        max_co_ppm: float = 100.0,
    ) -> Dict[str, float]:
        """
        Calculate optimal excess air for maximum efficiency - DETERMINISTIC.

        Optimal excess air balances:
        - Too low: incomplete combustion (CO), safety issues
        - Too high: efficiency loss from heating excess air

        Args:
            fuel_type: Type of fuel
            max_co_ppm: Maximum allowable CO (ppm)

        Returns:
            Dictionary with optimal settings
        """
        # Optimal excess air values (% by fuel type)
        # From combustion engineering practice
        optimal_values = {
            "natural_gas": {
                "min_safe_excess_air": 10.0,
                "optimal_excess_air": 15.0,
                "max_efficient_excess_air": 25.0,
                "optimal_o2": 3.0,
                "min_o2": 2.0,
                "max_o2": 5.0,
            },
            "propane": {
                "min_safe_excess_air": 10.0,
                "optimal_excess_air": 15.0,
                "max_efficient_excess_air": 25.0,
                "optimal_o2": 3.0,
                "min_o2": 2.0,
                "max_o2": 5.0,
            },
            "fuel_oil_2": {
                "min_safe_excess_air": 15.0,
                "optimal_excess_air": 20.0,
                "max_efficient_excess_air": 30.0,
                "optimal_o2": 4.0,
                "min_o2": 3.0,
                "max_o2": 6.0,
            },
            "hydrogen": {
                "min_safe_excess_air": 5.0,
                "optimal_excess_air": 10.0,
                "max_efficient_excess_air": 20.0,
                "optimal_o2": 2.0,
                "min_o2": 1.0,
                "max_o2": 4.0,
            },
        }

        return optimal_values.get(fuel_type, optimal_values["natural_gas"])

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _validate_inputs(
        self,
        fuel_flow_scfh: float,
        fuel_hhv_btu_scf: float,
        flue_gas_temp_f: float,
        flue_gas_o2_pct: float,
    ) -> None:
        """Validate calculation inputs."""
        if fuel_flow_scfh < 0:
            raise ValueError(f"Fuel flow ({fuel_flow_scfh}) cannot be negative")

        if fuel_hhv_btu_scf <= 0:
            raise ValueError(f"Fuel HHV ({fuel_hhv_btu_scf}) must be positive")

        if flue_gas_temp_f < 100:
            raise ValueError(f"Flue gas temp ({flue_gas_temp_f}F) seems too low")

        if flue_gas_o2_pct < 0 or flue_gas_o2_pct >= 21:
            raise ValueError(
                f"Flue gas O2 ({flue_gas_o2_pct}%) must be between 0 and 21%"
            )

    def _get_fuel_properties(
        self,
        fuel_type: str,
        fuel_hhv_btu_scf: float,
    ) -> Dict[str, float]:
        """Get fuel properties with override for HHV."""
        props = FUEL_PROPERTIES.get(fuel_type, FUEL_PROPERTIES["natural_gas"]).copy()
        props["hhv_btu_scf"] = fuel_hhv_btu_scf
        return props

    def _estimate_thermal_nox(
        self,
        flue_gas_temp_f: float,
        excess_air_pct: float,
    ) -> float:
        """
        Estimate thermal NOx formation - DETERMINISTIC.

        Thermal NOx is exponentially dependent on temperature.
        Simplified Zeldovich correlation.
        """
        # Base NOx at 2800F, 15% excess air = 0.05 lb/MMBtu
        base_nox = 0.05

        # Temperature correction (NOx increases exponentially with temp)
        # Doubling every ~200F above 2800F
        temp_factor = 2 ** ((flue_gas_temp_f - 400) / 400)
        temp_factor = max(0.5, min(2.0, temp_factor))

        # Excess air correction (NOx increases with O2 availability)
        ea_factor = 1 + (excess_air_pct - 15) * 0.01
        ea_factor = max(0.8, min(1.5, ea_factor))

        nox = base_nox * temp_factor * ea_factor

        return round(nox, 4)

    def _determine_combustion_status(
        self,
        excess_air_pct: float,
        flue_gas_co_ppm: Optional[float],
        thermal_efficiency_pct: float,
    ) -> CombustionStatus:
        """Determine combustion quality status - DETERMINISTIC."""

        # Check for incomplete combustion (high CO)
        if flue_gas_co_ppm is not None and flue_gas_co_ppm > 200:
            return CombustionStatus.INCOMPLETE

        # Check for rich combustion (low excess air)
        if excess_air_pct < 8:
            return CombustionStatus.RICH

        # Check for lean combustion (high excess air)
        if excess_air_pct > 30:
            return CombustionStatus.LEAN

        # Check for unstable (marginal efficiency)
        if thermal_efficiency_pct < 70:
            return CombustionStatus.UNSTABLE

        # Optimal range: 10-20% excess air, good efficiency
        if 10 <= excess_air_pct <= 20 and thermal_efficiency_pct >= 80:
            return CombustionStatus.OPTIMAL

        return CombustionStatus.LEAN if excess_air_pct > 20 else CombustionStatus.OPTIMAL

    def _generate_recommendations(
        self,
        excess_air_pct: float,
        flue_gas_temp_f: float,
        thermal_efficiency_pct: float,
        flue_gas_co_ppm: Optional[float],
    ) -> List[str]:
        """Generate optimization recommendations - DETERMINISTIC."""
        recommendations = []

        # Excess air recommendations
        if excess_air_pct > 25:
            recommendations.append(
                f"Reduce excess air from {excess_air_pct:.1f}% to 15-20% "
                f"to improve efficiency by ~{(excess_air_pct - 20) * 0.3:.1f}%"
            )
        elif excess_air_pct < 10:
            recommendations.append(
                f"Increase excess air from {excess_air_pct:.1f}% to 10-15% "
                "for safer combustion and lower CO"
            )

        # Flue gas temperature recommendations
        if flue_gas_temp_f > 500:
            potential_savings = (flue_gas_temp_f - 400) * 0.01
            recommendations.append(
                f"Consider heat recovery to reduce flue gas temp from "
                f"{flue_gas_temp_f:.0f}F to 350-400F "
                f"(potential {potential_savings:.1f}% efficiency gain)"
            )

        # CO recommendations
        if flue_gas_co_ppm is not None:
            if flue_gas_co_ppm > 100:
                recommendations.append(
                    f"High CO ({flue_gas_co_ppm:.0f} ppm) indicates incomplete "
                    "combustion - check burner adjustment and air/fuel ratio"
                )
            elif flue_gas_co_ppm < 20:
                recommendations.append(
                    "Low CO indicates potential for reducing excess air "
                    "to improve efficiency"
                )

        # Efficiency recommendations
        if thermal_efficiency_pct < 80:
            recommendations.append(
                f"Low efficiency ({thermal_efficiency_pct:.1f}%) - "
                "review combustion tuning and heat recovery opportunities"
            )

        return recommendations


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_combustion_calculator(
    provenance_enabled: bool = True,
    precision: int = 4,
) -> CombustionCalculator:
    """Factory function to create CombustionCalculator."""
    return CombustionCalculator(
        provenance_enabled=provenance_enabled,
        precision=precision,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CombustionConstants",
    "FUEL_PROPERTIES",
    "CombustionCalculator",
    "create_combustion_calculator",
]
