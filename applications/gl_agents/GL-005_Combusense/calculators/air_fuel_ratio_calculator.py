# -*- coding: utf-8 -*-
"""
Air-Fuel Ratio Calculator for GL-005 CombustionControlAgent

Calculates stoichiometric air-fuel ratios, excess air, and lambda (λ) values
for various fuel types. Zero-hallucination design using combustion chemistry.

Reference Standards:
- NFPA 86: Standard for Ovens and Furnaces
- API Standard 560: Fired Heaters for General Refinery Service
- ISO 13577: Industrial Furnaces and Associated Processing Equipment
- Turns: An Introduction to Combustion (Combustion Chemistry)

Mathematical Formulas:
- Stoichiometric Air: A_stoich = (O2_required / 0.21) * (ρ_air / ρ_fuel)
- O2 Required: O2 = C + H/4 - O/2 + S (kmol O2 per kmol fuel)
- Excess Air: EA (%) = (A_actual - A_stoich) / A_stoich * 100
- Lambda (λ): λ = A_actual / A_stoich = (21 - O2_meas) / (21 - O2_stoich)
- Air-Fuel Ratio: AFR = ṁ_air / ṁ_fuel (mass basis)
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import math
import logging
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class FuelType(str, Enum):
    """Supported fuel types"""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    DIESEL = "diesel"
    FUEL_OIL = "fuel_oil"
    METHANE = "methane"
    ETHANE = "ethane"
    BUTANE = "butane"
    HYDROGEN = "hydrogen"
    CUSTOM = "custom"


@dataclass
class FuelComposition:
    """Fuel elemental composition (mass basis)"""
    carbon_percent: float  # C (%)
    hydrogen_percent: float  # H (%)
    oxygen_percent: float  # O (%)
    sulfur_percent: float  # S (%)
    nitrogen_percent: float  # N (%)
    ash_percent: float  # Ash (%)
    moisture_percent: float  # H2O (%)

    def validate(self) -> bool:
        """Validate composition sums to ~100%"""
        total = (
            self.carbon_percent + self.hydrogen_percent + self.oxygen_percent +
            self.sulfur_percent + self.nitrogen_percent + self.ash_percent +
            self.moisture_percent
        )
        return 98 <= total <= 102  # Allow 2% tolerance


@dataclass
class StoichiometricProperties:
    """Stoichiometric combustion properties"""
    air_fuel_ratio_mass: float  # kg air / kg fuel
    air_fuel_ratio_molar: float  # kmol air / kmol fuel
    o2_required_kg_per_kg_fuel: float  # kg O2 / kg fuel
    theoretical_air_kg_per_kg_fuel: float  # kg air / kg fuel (same as AFR)
    stoichiometric_o2_percent: float  # O2% in flue gas at stoichiometric
    stoichiometric_co2_percent: float  # CO2% in flue gas at stoichiometric


class AirFuelRatioInput(BaseModel):
    """Input parameters for air-fuel ratio calculations"""

    # Fuel properties
    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel"
    )
    fuel_composition: Optional[Dict[str, float]] = Field(
        None,
        description="Fuel composition (C, H, O, S, N, ash, moisture) in mass %"
    )
    fuel_flow_rate_kg_per_hr: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Fuel flow rate"
    )

    # Air flow
    air_flow_rate_kg_per_hr: float = Field(
        ...,
        ge=0,
        le=1000000,
        description="Air flow rate"
    )

    # Measured flue gas composition
    flue_gas_o2_percent_dry: Optional[float] = Field(
        None,
        ge=0,
        le=21,
        description="Measured O2 in flue gas (dry basis)"
    )
    flue_gas_co2_percent_dry: Optional[float] = Field(
        None,
        ge=0,
        le=20,
        description="Measured CO2 in flue gas (dry basis)"
    )

    # Operating conditions
    ambient_temperature_c: float = Field(
        default=25.0,
        ge=-50,
        le=60
    )
    ambient_pressure_pa: float = Field(
        default=101325,
        ge=80000,
        le=110000
    )
    combustion_temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=2000,
        description="Combustion zone temperature"
    )

    # Target parameters
    target_excess_air_percent: float = Field(
        default=15.0,
        ge=0,
        le=100,
        description="Target excess air percentage"
    )
    target_o2_percent: Optional[float] = Field(
        None,
        ge=0,
        le=21,
        description="Target O2 percentage in flue gas"
    )


class AirFuelRatioOutput(BaseModel):
    """Air-fuel ratio calculation results"""

    # Stoichiometric properties
    stoichiometric_afr_mass: float = Field(
        ...,
        description="Stoichiometric air-fuel ratio (kg air / kg fuel)"
    )
    stoichiometric_afr_molar: float = Field(
        ...,
        description="Stoichiometric AFR (kmol air / kmol fuel)"
    )
    o2_required_kg_per_kg_fuel: float = Field(
        ...,
        description="Theoretical O2 required per kg fuel"
    )

    # Actual operating conditions
    actual_afr_mass: float = Field(
        ...,
        description="Actual air-fuel ratio (kg air / kg fuel)"
    )
    excess_air_percent: float = Field(
        ...,
        description="Excess air percentage"
    )
    lambda_value: float = Field(
        ...,
        description="Lambda (λ) = actual AFR / stoichiometric AFR"
    )

    # Flue gas composition (calculated or measured)
    flue_gas_o2_percent_dry: float = Field(
        ...,
        description="O2 in dry flue gas (%)"
    )
    flue_gas_co2_percent_dry: float = Field(
        ...,
        description="CO2 in dry flue gas (%)"
    )
    flue_gas_n2_percent_dry: float = Field(
        ...,
        description="N2 in dry flue gas (%)"
    )

    # Operating assessment
    combustion_quality: str = Field(
        ...,
        description="rich, stoichiometric, or lean"
    )
    is_optimal: bool = Field(
        ...,
        description="Whether operating at optimal excess air"
    )

    # Air flow recommendations
    recommended_air_flow_kg_per_hr: float = Field(
        ...,
        description="Recommended air flow for target excess air"
    )
    air_flow_adjustment_percent: float = Field(
        ...,
        description="Required air flow adjustment (%)"
    )

    # Fuel-specific data
    fuel_type: str
    fuel_composition_used: Dict[str, float] = Field(
        ...,
        description="Fuel composition used in calculations"
    )

    # Performance metrics
    combustion_completeness_percent: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Estimated combustion completeness"
    )
    theoretical_flame_temperature_c: Optional[float] = Field(
        None,
        description="Theoretical adiabatic flame temperature"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Air-fuel ratio recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about current operation"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for calculation provenance"
    )


class AirFuelRatioCalculator:
    """
    Air-fuel ratio calculator for combustion systems.

    Calculates stoichiometric air requirements and actual operating conditions
    based on fuel composition and measured flue gas analysis.

    Key Calculations:
        1. Stoichiometric Air-Fuel Ratio (from fuel composition)
        2. Actual Air-Fuel Ratio (from measurements)
        3. Excess Air Percentage
        4. Lambda (λ) value
        5. Flue Gas Composition

    Combustion Chemistry:
        Complete combustion of hydrocarbon fuel:
            CₓHᵧOᵦSᵟ + a(O₂ + 3.76N₂) → xCO₂ + (y/2)H₂O + δSO₂ + 3.76aN₂

        Where:
            a = O2 required (kmol) = x + y/4 - β/2 + δ
            Air required = a * (1 + 3.76) / 0.21 kmol air per kmol fuel

    Reference:
        - Air is 21% O2, 79% N2 by volume (23% O2, 77% N2 by mass)
        - Molecular weight: Air ≈ 28.97 kg/kmol
    """

    # Molecular weights (kg/kmol)
    MW = {
        'C': 12.011,
        'H': 1.008,
        'O': 15.999,
        'S': 32.065,
        'N': 14.007,
        'Air': 28.97,
        'O2': 31.998,
        'N2': 28.014,
        'CO2': 44.01,
        'H2O': 18.015,
        'SO2': 64.064
    }

    # Standard fuel compositions (mass %)
    STANDARD_FUELS = {
        FuelType.NATURAL_GAS: {
            'C': 74.9, 'H': 25.0, 'O': 0.0, 'S': 0.0, 'N': 0.1, 'ash': 0.0, 'moisture': 0.0
        },
        FuelType.PROPANE: {
            'C': 81.7, 'H': 18.2, 'O': 0.0, 'S': 0.0, 'N': 0.1, 'ash': 0.0, 'moisture': 0.0
        },
        FuelType.METHANE: {
            'C': 74.9, 'H': 25.1, 'O': 0.0, 'S': 0.0, 'N': 0.0, 'ash': 0.0, 'moisture': 0.0
        },
        FuelType.DIESEL: {
            'C': 86.0, 'H': 13.0, 'O': 0.5, 'S': 0.3, 'N': 0.2, 'ash': 0.0, 'moisture': 0.0
        },
        FuelType.FUEL_OIL: {
            'C': 87.0, 'H': 11.5, 'O': 0.5, 'S': 0.8, 'N': 0.2, 'ash': 0.0, 'moisture': 0.0
        },
        FuelType.HYDROGEN: {
            'C': 0.0, 'H': 100.0, 'O': 0.0, 'S': 0.0, 'N': 0.0, 'ash': 0.0, 'moisture': 0.0
        }
    }

    # Optimal excess air ranges by fuel type (%)
    OPTIMAL_EXCESS_AIR = {
        FuelType.NATURAL_GAS: (10, 20),
        FuelType.PROPANE: (10, 20),
        FuelType.DIESEL: (15, 25),
        FuelType.FUEL_OIL: (15, 30),
        FuelType.HYDROGEN: (5, 15)
    }

    def __init__(self):
        """Initialize air-fuel ratio calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate_air_fuel_ratio(
        self,
        afr_input: AirFuelRatioInput
    ) -> AirFuelRatioOutput:
        """
        Calculate air-fuel ratio and related parameters.

        Algorithm:
            1. Get fuel composition (from database or custom)
            2. Calculate stoichiometric air-fuel ratio
            3. Calculate actual air-fuel ratio from measurements
            4. Calculate excess air and lambda
            5. Calculate/validate flue gas composition
            6. Generate recommendations

        Args:
            afr_input: Air-fuel ratio input parameters

        Returns:
            AirFuelRatioOutput with complete AFR analysis
        """
        self.logger.info(f"Calculating air-fuel ratio for {afr_input.fuel_type}")

        # Step 1: Get fuel composition
        fuel_comp = self._get_fuel_composition(
            afr_input.fuel_type,
            afr_input.fuel_composition
        )

        # Step 2: Calculate stoichiometric properties
        stoich_props = self._calculate_stoichiometric_properties(fuel_comp)

        # Step 3: Calculate actual air-fuel ratio
        actual_afr = afr_input.air_flow_rate_kg_per_hr / afr_input.fuel_flow_rate_kg_per_hr

        # Step 4: Calculate excess air and lambda
        excess_air_percent = ((actual_afr - stoich_props.air_fuel_ratio_mass) /
                             stoich_props.air_fuel_ratio_mass * 100)

        lambda_value = actual_afr / stoich_props.air_fuel_ratio_mass

        # Step 5: Calculate flue gas composition
        if afr_input.flue_gas_o2_percent_dry is not None:
            # Use measured O2
            o2_percent = afr_input.flue_gas_o2_percent_dry
            # Calculate lambda from O2 measurement
            lambda_from_o2 = self._calculate_lambda_from_o2(
                o2_percent,
                stoich_props.stoichiometric_o2_percent
            )
            # Use measured lambda to refine calculations
            lambda_value = lambda_from_o2
            excess_air_percent = (lambda_value - 1) * 100
        else:
            # Calculate O2 from excess air
            o2_percent = self._calculate_o2_from_excess_air(
                excess_air_percent,
                stoich_props.stoichiometric_o2_percent
            )

        # Calculate CO2 in flue gas
        co2_percent = self._calculate_co2_in_flue_gas(
            fuel_comp,
            excess_air_percent
        )

        # Calculate N2 (balance)
        n2_percent = 100 - o2_percent - co2_percent

        # Step 6: Assess combustion quality
        combustion_quality = self._assess_combustion_quality(lambda_value)

        # Step 7: Check if operating at optimal excess air
        is_optimal = self._check_optimal_operation(
            afr_input.fuel_type,
            excess_air_percent
        )

        # Step 8: Calculate recommended air flow
        target_lambda = 1 + (afr_input.target_excess_air_percent / 100)
        recommended_air_flow = (
            target_lambda * stoich_props.air_fuel_ratio_mass *
            afr_input.fuel_flow_rate_kg_per_hr
        )

        air_flow_adjustment = (
            (recommended_air_flow - afr_input.air_flow_rate_kg_per_hr) /
            afr_input.air_flow_rate_kg_per_hr * 100
        )

        # Step 9: Estimate combustion completeness
        combustion_completeness = self._estimate_combustion_completeness(
            o2_percent,
            excess_air_percent
        )

        # Step 10: Calculate adiabatic flame temperature (optional)
        flame_temp = self._calculate_adiabatic_flame_temperature(
            fuel_comp,
            excess_air_percent,
            afr_input.ambient_temperature_c
        ) if afr_input.combustion_temperature_c is None else afr_input.combustion_temperature_c

        # Step 11: Generate recommendations and warnings
        recommendations = self._generate_afr_recommendations(
            excess_air_percent,
            lambda_value,
            afr_input.fuel_type,
            is_optimal
        )

        warnings = self._generate_warnings(
            excess_air_percent,
            lambda_value,
            o2_percent
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            afr_input,
            actual_afr,
            excess_air_percent,
            lambda_value,
            fuel_comp
        )

        return AirFuelRatioOutput(
            stoichiometric_afr_mass=self._round_decimal(stoich_props.air_fuel_ratio_mass, 4),
            stoichiometric_afr_molar=self._round_decimal(stoich_props.air_fuel_ratio_molar, 4),
            o2_required_kg_per_kg_fuel=self._round_decimal(stoich_props.o2_required_kg_per_kg_fuel, 4),
            actual_afr_mass=self._round_decimal(actual_afr, 4),
            excess_air_percent=self._round_decimal(excess_air_percent, 2),
            lambda_value=self._round_decimal(lambda_value, 4),
            flue_gas_o2_percent_dry=self._round_decimal(o2_percent, 2),
            flue_gas_co2_percent_dry=self._round_decimal(co2_percent, 2),
            flue_gas_n2_percent_dry=self._round_decimal(n2_percent, 2),
            combustion_quality=combustion_quality,
            is_optimal=is_optimal,
            recommended_air_flow_kg_per_hr=self._round_decimal(recommended_air_flow, 2),
            air_flow_adjustment_percent=self._round_decimal(air_flow_adjustment, 2),
            fuel_type=afr_input.fuel_type.value,
            fuel_composition_used=fuel_comp.__dict__,
            combustion_completeness_percent=self._round_decimal(combustion_completeness, 2),
            theoretical_flame_temperature_c=self._round_decimal(flame_temp, 1) if flame_temp else None,
            recommendations=recommendations,
            warnings=warnings,
            provenance_hash=provenance_hash
        )

    def _get_fuel_composition(
        self,
        fuel_type: FuelType,
        custom_composition: Optional[Dict[str, float]]
    ) -> FuelComposition:
        """
        Get fuel composition from database or custom input.

        Args:
            fuel_type: Type of fuel
            custom_composition: Custom composition (if provided)

        Returns:
            FuelComposition object
        """
        if custom_composition and fuel_type == FuelType.CUSTOM:
            comp = FuelComposition(
                carbon_percent=custom_composition.get('C', 0),
                hydrogen_percent=custom_composition.get('H', 0),
                oxygen_percent=custom_composition.get('O', 0),
                sulfur_percent=custom_composition.get('S', 0),
                nitrogen_percent=custom_composition.get('N', 0),
                ash_percent=custom_composition.get('ash', 0),
                moisture_percent=custom_composition.get('moisture', 0)
            )
        else:
            # Use standard composition
            std_comp = self.STANDARD_FUELS.get(fuel_type, self.STANDARD_FUELS[FuelType.NATURAL_GAS])
            comp = FuelComposition(
                carbon_percent=std_comp['C'],
                hydrogen_percent=std_comp['H'],
                oxygen_percent=std_comp['O'],
                sulfur_percent=std_comp['S'],
                nitrogen_percent=std_comp['N'],
                ash_percent=std_comp['ash'],
                moisture_percent=std_comp['moisture']
            )

        if not comp.validate():
            self.logger.warning("Fuel composition does not sum to 100%")

        return comp

    def _calculate_stoichiometric_properties(
        self,
        fuel_comp: FuelComposition
    ) -> StoichiometricProperties:
        """
        Calculate stoichiometric air-fuel ratio from fuel composition.

        Combustion equation:
            C + O2 → CO2  (requires 1 mol O2 per mol C)
            2H2 + O2 → 2H2O  (requires 0.5 mol O2 per mol H2)
            S + O2 → SO2  (requires 1 mol O2 per mol S)

        O2 required (kg) = 2.67*C + 8*H - O + S (per kg fuel)

        Air required (kg) = O2_required / 0.23 (air is 23% O2 by mass)

        Args:
            fuel_comp: Fuel composition

        Returns:
            StoichiometricProperties object
        """
        # Convert mass % to mass fractions
        C = fuel_comp.carbon_percent / 100
        H = fuel_comp.hydrogen_percent / 100
        O = fuel_comp.oxygen_percent / 100
        S = fuel_comp.sulfur_percent / 100

        # Calculate O2 required (kg O2 per kg fuel)
        # Using atomic mass ratios:
        # C: 32/12 = 2.67
        # H: 16/2 = 8 (but H2 → H2O requires 8 kg O2 per kg H)
        # O: already present, subtract
        # S: 32/32 = 1
        o2_required = 2.67 * C + 8 * H - O + S

        # Calculate air required (kg air per kg fuel)
        # Air is 23.15% O2 by mass
        air_required = o2_required / 0.2315

        # Calculate molar air-fuel ratio (simplified)
        # Assuming average molecular weight
        afr_molar = air_required * 15 / self.MW['Air']  # Approximate

        # Stoichiometric O2 in flue gas (nearly zero, but ~0.5% due to dissociation)
        stoich_o2 = 0.5

        # Calculate stoichiometric CO2 in flue gas
        # CO2 produced (kg per kg fuel) = 44/12 * C = 3.67 * C
        co2_produced = 3.67 * C

        # Total flue gas (approximate)
        flue_gas_total = 1 + air_required  # fuel + air
        stoich_co2 = (co2_produced / flue_gas_total) * 100

        return StoichiometricProperties(
            air_fuel_ratio_mass=air_required,
            air_fuel_ratio_molar=afr_molar,
            o2_required_kg_per_kg_fuel=o2_required,
            theoretical_air_kg_per_kg_fuel=air_required,
            stoichiometric_o2_percent=stoich_o2,
            stoichiometric_co2_percent=stoich_co2
        )

    def _calculate_lambda_from_o2(
        self,
        o2_measured: float,
        o2_stoich: float
    ) -> float:
        """
        Calculate lambda from measured O2.

        Formula:
            λ = (21 - O2_stoich) / (21 - O2_measured)

        Args:
            o2_measured: Measured O2 (%)
            o2_stoich: Stoichiometric O2 (%)

        Returns:
            Lambda value
        """
        if o2_measured >= 21:
            return 1.0

        lambda_val = (21 - o2_stoich) / (21 - o2_measured)
        return lambda_val

    def _calculate_o2_from_excess_air(
        self,
        excess_air_percent: float,
        o2_stoich: float
    ) -> float:
        """
        Calculate O2 in flue gas from excess air.

        Formula:
            O2 = 21 * EA / (100 + EA) (approximate)

        Args:
            excess_air_percent: Excess air (%)
            o2_stoich: Stoichiometric O2 (%)

        Returns:
            O2 percentage in flue gas
        """
        ea_fraction = excess_air_percent / 100
        o2_percent = 21 * ea_fraction / (1 + ea_fraction)
        return o2_percent

    def _calculate_co2_in_flue_gas(
        self,
        fuel_comp: FuelComposition,
        excess_air_percent: float
    ) -> float:
        """
        Calculate CO2 in flue gas.

        Args:
            fuel_comp: Fuel composition
            excess_air_percent: Excess air (%)

        Returns:
            CO2 percentage in flue gas (dry basis)
        """
        # Carbon in fuel (mass fraction)
        C = fuel_comp.carbon_percent / 100

        # CO2 produced per kg fuel
        co2_produced = 3.67 * C  # kg CO2 per kg fuel

        # Total flue gas (simplified)
        stoich_air = 2.67 * C + 8 * (fuel_comp.hydrogen_percent / 100)
        stoich_air = stoich_air / 0.2315
        actual_air = stoich_air * (1 + excess_air_percent / 100)
        total_flue_gas = 1 + actual_air

        # CO2 percentage (volume basis, approximate from mass basis)
        co2_percent = (co2_produced / total_flue_gas) * 100 * 0.7  # Conversion factor

        return min(co2_percent, 20)  # Cap at 20%

    def _assess_combustion_quality(self, lambda_value: float) -> str:
        """Assess combustion quality based on lambda"""
        if lambda_value < 0.95:
            return "rich"  # Insufficient air
        elif lambda_value > 1.05:
            return "lean"  # Excess air
        else:
            return "stoichiometric"

    def _check_optimal_operation(
        self,
        fuel_type: FuelType,
        excess_air_percent: float
    ) -> bool:
        """Check if operating in optimal excess air range"""
        optimal_range = self.OPTIMAL_EXCESS_AIR.get(
            fuel_type,
            (10, 25)  # Default range
        )
        return optimal_range[0] <= excess_air_percent <= optimal_range[1]

    def _estimate_combustion_completeness(
        self,
        o2_percent: float,
        excess_air_percent: float
    ) -> float:
        """
        Estimate combustion completeness.

        Higher O2 and excess air → more complete combustion
        """
        if excess_air_percent < 0:
            # Rich combustion - incomplete
            return 80.0
        elif excess_air_percent < 5:
            return 90.0
        elif excess_air_percent < 30:
            return 99.0
        else:
            return 99.5

    def _calculate_adiabatic_flame_temperature(
        self,
        fuel_comp: FuelComposition,
        excess_air_percent: float,
        ambient_temp_c: float
    ) -> float:
        """
        Estimate adiabatic flame temperature (simplified).

        More excess air → lower flame temperature
        """
        # Base flame temperature (approximate)
        base_temp = 1900  # °C for typical hydrocarbon

        # Correction for excess air (dilution effect)
        temp_reduction = excess_air_percent * 5  # °C per % excess air

        flame_temp = base_temp - temp_reduction + (ambient_temp_c - 25)

        return max(flame_temp, 800)  # Minimum 800°C

    def _generate_afr_recommendations(
        self,
        excess_air_percent: float,
        lambda_value: float,
        fuel_type: FuelType,
        is_optimal: bool
    ) -> List[str]:
        """Generate air-fuel ratio recommendations"""
        recommendations = []

        if is_optimal:
            recommendations.append("Operating at optimal excess air for fuel type")
        else:
            optimal_range = self.OPTIMAL_EXCESS_AIR.get(fuel_type, (10, 25))
            if excess_air_percent < optimal_range[0]:
                recommendations.append(
                    f"Increase excess air to {optimal_range[0]}-{optimal_range[1]}% for optimal efficiency"
                )
            else:
                recommendations.append(
                    f"Reduce excess air to {optimal_range[0]}-{optimal_range[1]}% to improve efficiency"
                )

        if lambda_value < 0.95:
            recommendations.append("Rich combustion - increase air flow to avoid CO formation")

        if excess_air_percent > 40:
            recommendations.append("Excessive air flow - reduce to improve thermal efficiency")

        return recommendations

    def _generate_warnings(
        self,
        excess_air_percent: float,
        lambda_value: float,
        o2_percent: float
    ) -> List[str]:
        """Generate warnings about air-fuel ratio"""
        warnings = []

        if lambda_value < 0.9:
            warnings.append("CRITICAL: Insufficient air - risk of incomplete combustion and CO formation")

        if lambda_value < 0.95:
            warnings.append("WARNING: Operating fuel-rich - monitor CO emissions closely")

        if excess_air_percent < 5:
            warnings.append("WARNING: Low excess air - risk of flame instability")

        if excess_air_percent > 50:
            warnings.append("WARNING: Very high excess air - significant efficiency loss")

        if o2_percent < 1:
            warnings.append("CRITICAL: Very low O2 - immediate adjustment required")

        return warnings

    def _calculate_provenance(
        self,
        afr_input: AirFuelRatioInput,
        actual_afr: float,
        excess_air: float,
        lambda_value: float,
        fuel_comp: FuelComposition
    ) -> str:
        """Calculate SHA-256 hash for complete audit trail"""
        provenance_data = {
            'fuel_type': afr_input.fuel_type.value,
            'fuel_flow': afr_input.fuel_flow_rate_kg_per_hr,
            'air_flow': afr_input.air_flow_rate_kg_per_hr,
            'actual_afr': actual_afr,
            'excess_air': excess_air,
            'lambda': lambda_value,
            'fuel_composition': fuel_comp.__dict__
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
