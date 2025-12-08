# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Fuel Impact Analyzer

This module implements fuel quality impact analysis on burner component degradation.
Analyzes how fuel properties affect burner component life through deterministic
engineering correlations.

Key capabilities:
    - Fuel quality scoring (0-100 scale)
    - Contaminant impact factors (sulfur, vanadium, sodium)
    - Fouling rate prediction (Kern-Seaton model)
    - Coking tendency index calculation
    - Corrosion rate estimation (Arrhenius kinetics)
    - Life reduction factor computation
    - Fuel switching impact analysis

ZERO-HALLUCINATION COMPLIANCE:
    All calculations use deterministic engineering formulas:
    - Kern-Seaton fouling model
    - Arrhenius rate equations for corrosion
    - API 571 damage mechanism correlations
    - Industry empirical correlations from field data

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.fuel_impact import (
    ...     FuelImpactAnalyzer
    ... )
    >>> analyzer = FuelImpactAnalyzer()
    >>> quality = analyzer.calculate_fuel_quality_score(fuel_properties)
    >>> impact = analyzer.analyze_degradation_impact(fuel_properties, conditions)
    >>> print(f"Life reduction factor: {impact.life_reduction_factor:.2f}")

References:
    - API 571: Damage Mechanisms Affecting Fixed Equipment
    - API 530: Calculation of Heater-Tube Thickness
    - NACE SP0472: Methods and Controls to Prevent In-Service Environmental Cracking
    - Kern-Seaton Model: Kern & Seaton, Brit. Chem. Eng., 1959
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Gas constant for Arrhenius calculations (kJ/mol-K)
GAS_CONSTANT_R = 8.314e-3

# Reference temperature for rate calculations (K)
REFERENCE_TEMPERATURE_K = 1073.15  # 800 C

# Typical activation energies for degradation mechanisms (kJ/mol)
ACTIVATION_ENERGIES = {
    "sulfur_corrosion": 45.0,
    "vanadium_corrosion": 55.0,
    "sodium_corrosion": 50.0,
    "oxidation": 35.0,
    "carburization": 120.0,
    "coking": 85.0,
}

# Contaminant impact multipliers (from industry data)
CONTAMINANT_MULTIPLIERS = {
    "sulfur": 1.5,      # Per % increase from baseline
    "vanadium": 2.0,    # Per 10 ppm increase
    "sodium": 2.5,      # Per 10 ppm increase
    "potassium": 2.0,   # Per 10 ppm increase
    "ash": 1.2,         # Per % increase
    "water": 0.8,       # Per % increase
    "nitrogen": 0.3,    # Per % increase
}

# Fuel quality baseline values
BASELINE_FUEL_PROPERTIES = {
    "sulfur_pct": 0.5,
    "vanadium_ppm": 10.0,
    "sodium_ppm": 5.0,
    "potassium_ppm": 5.0,
    "ash_pct": 0.02,
    "water_pct": 0.05,
    "nitrogen_pct": 0.1,
    "carbon_residue_pct": 0.1,
    "asphaltenes_pct": 1.0,
    "heating_value_mj_kg": 42.5,
}


# =============================================================================
# ENUMS
# =============================================================================

class FuelType(str, Enum):
    """Fuel types for burner systems."""
    NATURAL_GAS = "natural_gas"
    REFINERY_GAS = "refinery_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    RESIDUAL_FUEL_OIL = "residual_fuel_oil"
    CRUDE_OIL = "crude_oil"
    DIESEL = "diesel"
    LPG = "lpg"
    HYDROGEN = "hydrogen"
    SYNGAS = "syngas"


class DamageMechanism(str, Enum):
    """Damage mechanisms from fuel contaminants."""
    SULFIDATION = "sulfidation"
    VANADIUM_ATTACK = "vanadium_attack"
    ASH_DEPOSITION = "ash_deposition"
    COKING = "coking"
    FLAME_IMPINGEMENT = "flame_impingement"
    OXIDATION = "oxidation"
    CARBURIZATION = "carburization"
    EROSION = "erosion"
    HOT_CORROSION = "hot_corrosion"


class ImpactSeverity(str, Enum):
    """Fuel impact severity classification."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class FoulingLevel(str, Enum):
    """Fouling severity levels."""
    CLEAN = "clean"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FuelProperties:
    """Fuel properties for impact analysis."""
    fuel_type: FuelType = FuelType.NATURAL_GAS

    # Sulfur content
    sulfur_pct: float = 0.0          # Weight %
    h2s_ppm: float = 0.0             # For gas fuels

    # Metals (primarily for liquid fuels)
    vanadium_ppm: float = 0.0
    sodium_ppm: float = 0.0
    potassium_ppm: float = 0.0
    nickel_ppm: float = 0.0
    iron_ppm: float = 0.0
    calcium_ppm: float = 0.0

    # Ash and residue
    ash_pct: float = 0.0             # Weight %
    carbon_residue_pct: float = 0.0  # Conradson/Ramsbottom
    asphaltenes_pct: float = 0.0     # n-Heptane insolubles

    # Water and contamination
    water_pct: float = 0.0           # Weight %
    sediment_pct: float = 0.0        # BS&W for liquids

    # Heating value
    heating_value_mj_kg: float = 42.5  # Lower heating value
    density_kg_m3: float = 850.0       # At 15C for liquids

    # Gas composition (mol % for gas fuels)
    methane_pct: float = 0.0
    ethane_pct: float = 0.0
    propane_pct: float = 0.0
    hydrogen_pct: float = 0.0
    co_pct: float = 0.0
    co2_pct: float = 0.0
    nitrogen_pct: float = 0.0


@dataclass
class OperatingConditions:
    """Burner operating conditions."""
    flame_temperature_c: float = 1200.0
    flue_gas_temperature_c: float = 350.0
    tube_metal_temperature_c: float = 450.0
    excess_air_pct: float = 15.0
    firing_rate_pct: float = 80.0
    operating_hours_per_year: float = 8000.0
    thermal_cycles_per_year: float = 100.0


@dataclass
class FoulingResult:
    """Fouling prediction result."""
    fouling_factor_m2k_w: float
    asymptotic_fouling: float
    time_constant_hours: float
    fouling_level: FoulingLevel
    cleaning_interval_hours: Optional[float]
    efficiency_loss_pct: float


@dataclass
class CokingResult:
    """Coking tendency analysis result."""
    coking_index: float
    coke_formation_rate_g_m2_h: float
    decoking_interval_hours: Optional[float]
    severity: ImpactSeverity


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FuelQualityScore(BaseModel):
    """Fuel quality score result."""

    overall_score: float = Field(
        ..., ge=0, le=100,
        description="Overall fuel quality score (0-100, 100=best)"
    )

    # Component scores
    sulfur_score: float = Field(..., ge=0, le=100, description="Sulfur penalty score")
    metals_score: float = Field(..., ge=0, le=100, description="Metals penalty score")
    ash_score: float = Field(..., ge=0, le=100, description="Ash penalty score")
    water_score: float = Field(..., ge=0, le=100, description="Water penalty score")
    heating_value_score: float = Field(..., ge=0, le=100, description="Heating value score")

    # Classification
    quality_class: str = Field(..., description="Quality classification")
    fuel_type: FuelType = Field(..., description="Fuel type analyzed")

    # Recommendations
    concerns: List[str] = Field(
        default_factory=list,
        description="Identified quality concerns"
    )

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class DegradationImpact(BaseModel):
    """Fuel-induced degradation impact result."""

    life_reduction_factor: float = Field(
        ..., ge=0.1, le=10.0,
        description="Life reduction multiplier (1.0=no impact, >1=shorter life)"
    )

    # Damage mechanism contributions
    sulfidation_contribution: float = Field(
        ..., ge=0, le=1,
        description="Sulfidation damage contribution"
    )
    vanadium_contribution: float = Field(
        ..., ge=0, le=1,
        description="Vanadium attack contribution"
    )
    ash_contribution: float = Field(
        ..., ge=0, le=1,
        description="Ash deposition contribution"
    )
    coking_contribution: float = Field(
        ..., ge=0, le=1,
        description="Coking contribution"
    )

    # Corrosion rates
    corrosion_rate_mm_year: float = Field(
        ..., ge=0,
        description="Estimated corrosion rate (mm/year)"
    )

    # Fouling
    fouling_factor_m2k_w: float = Field(
        ..., ge=0,
        description="Fouling thermal resistance"
    )

    # Impact classification
    overall_severity: ImpactSeverity = Field(
        ..., description="Overall impact severity"
    )

    primary_mechanism: DamageMechanism = Field(
        ..., description="Primary damage mechanism"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Mitigation recommendations"
    )

    # Estimated costs
    estimated_annual_impact_usd: Optional[float] = Field(
        default=None,
        description="Estimated annual cost impact"
    )

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class FuelSwitchingImpact(BaseModel):
    """Impact analysis for switching between fuels."""

    original_fuel: FuelType = Field(..., description="Original fuel type")
    new_fuel: FuelType = Field(..., description="New fuel type")

    # Life impact
    life_change_factor: float = Field(
        ...,
        description="Life change factor (>1=improvement, <1=degradation)"
    )

    # Quality comparison
    quality_score_change: float = Field(
        ...,
        description="Change in quality score"
    )

    # Economic analysis
    efficiency_change_pct: float = Field(
        ...,
        description="Combustion efficiency change (%)"
    )
    fuel_cost_impact_pct: Optional[float] = Field(
        default=None,
        description="Fuel cost change (%)"
    )

    # Risk assessment
    new_risks: List[str] = Field(
        default_factory=list,
        description="New risks introduced"
    )
    mitigated_risks: List[str] = Field(
        default_factory=list,
        description="Risks mitigated by switch"
    )

    # Implementation
    required_modifications: List[str] = Field(
        default_factory=list,
        description="Required equipment modifications"
    )

    recommendation: str = Field(..., description="Overall recommendation")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


# =============================================================================
# FUEL IMPACT ANALYZER CLASS
# =============================================================================

class FuelImpactAnalyzer:
    """
    Fuel quality impact on burner degradation.

    Analyzes how fuel properties affect burner component life:
    - Contaminant effects (sulfur, vanadium, sodium)
    - Ash deposition and fouling rates
    - Coking tendency from heavy hydrocarbons
    - Moisture impact on flame stability
    - Heating value variation effects

    All calculations are DETERMINISTIC using established
    engineering correlations for ZERO-HALLUCINATION compliance.

    Attributes:
        reference_temp_k: Reference temperature for rate calculations
        baseline_properties: Baseline fuel properties for comparison

    Example:
        >>> analyzer = FuelImpactAnalyzer()
        >>> fuel = FuelProperties(fuel_type=FuelType.FUEL_OIL_NO6, sulfur_pct=2.5)
        >>> quality = analyzer.calculate_fuel_quality_score(fuel)
        >>> print(f"Quality score: {quality.overall_score:.1f}")
    """

    def __init__(
        self,
        reference_temp_k: float = REFERENCE_TEMPERATURE_K,
        baseline_properties: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize FuelImpactAnalyzer.

        Args:
            reference_temp_k: Reference temperature for rate calculations
            baseline_properties: Custom baseline fuel properties
        """
        self.reference_temp_k = reference_temp_k
        self.baseline = baseline_properties or BASELINE_FUEL_PROPERTIES.copy()

        logger.info(
            f"FuelImpactAnalyzer initialized: ref_temp={reference_temp_k:.0f}K"
        )

    def calculate_fuel_quality_score(
        self,
        fuel: FuelProperties
    ) -> FuelQualityScore:
        """
        Calculate comprehensive fuel quality score.

        Scores from 0-100 where 100 is ideal quality.
        Lower scores indicate worse fuel quality.

        Args:
            fuel: Fuel properties

        Returns:
            FuelQualityScore with component scores and recommendations
        """
        logger.info(f"Calculating quality score for {fuel.fuel_type.value}")

        concerns: List[str] = []

        # Sulfur score (100 = no sulfur, 0 = very high)
        sulfur_score = self._calculate_sulfur_score(fuel, concerns)

        # Metals score (vanadium, sodium, potassium)
        metals_score = self._calculate_metals_score(fuel, concerns)

        # Ash score
        ash_score = self._calculate_ash_score(fuel, concerns)

        # Water/contamination score
        water_score = self._calculate_water_score(fuel, concerns)

        # Heating value score (consistency with baseline)
        hv_score = self._calculate_heating_value_score(fuel, concerns)

        # Weighted overall score
        weights = {
            "sulfur": 0.30,
            "metals": 0.25,
            "ash": 0.20,
            "water": 0.10,
            "heating_value": 0.15,
        }

        overall = (
            weights["sulfur"] * sulfur_score +
            weights["metals"] * metals_score +
            weights["ash"] * ash_score +
            weights["water"] * water_score +
            weights["heating_value"] * hv_score
        )

        # Quality classification
        if overall >= 90:
            quality_class = "Premium"
        elif overall >= 75:
            quality_class = "Standard"
        elif overall >= 60:
            quality_class = "Acceptable"
        elif overall >= 40:
            quality_class = "Marginal"
        else:
            quality_class = "Poor"

        # Provenance
        provenance_hash = self._calculate_provenance(
            "quality_score",
            {
                "fuel_type": fuel.fuel_type.value,
                "sulfur": fuel.sulfur_pct,
                "vanadium": fuel.vanadium_ppm,
                "overall": overall,
            }
        )

        logger.info(
            f"Quality score: {overall:.1f} ({quality_class}) for {fuel.fuel_type.value}"
        )

        return FuelQualityScore(
            overall_score=round(overall, 1),
            sulfur_score=round(sulfur_score, 1),
            metals_score=round(metals_score, 1),
            ash_score=round(ash_score, 1),
            water_score=round(water_score, 1),
            heating_value_score=round(hv_score, 1),
            quality_class=quality_class,
            fuel_type=fuel.fuel_type,
            concerns=concerns,
            provenance_hash=provenance_hash,
        )

    def _calculate_sulfur_score(
        self,
        fuel: FuelProperties,
        concerns: List[str]
    ) -> float:
        """Calculate sulfur penalty score."""
        # For gas fuels, convert H2S ppm to equivalent sulfur %
        if fuel.fuel_type in [FuelType.NATURAL_GAS, FuelType.REFINERY_GAS]:
            # H2S ppm to weight % sulfur (approximate)
            effective_sulfur = fuel.h2s_ppm * 32 / (1e6 * 16)  # Molecular weight ratio
        else:
            effective_sulfur = fuel.sulfur_pct

        # Score: 100 at 0%, decreases with sulfur content
        # 0.5% baseline, 3% is severe for liquid fuels
        if effective_sulfur <= 0.01:
            score = 100.0
        elif effective_sulfur <= 0.5:
            score = 100 - 20 * (effective_sulfur / 0.5)
        elif effective_sulfur <= 1.5:
            score = 80 - 30 * ((effective_sulfur - 0.5) / 1.0)
        elif effective_sulfur <= 3.0:
            score = 50 - 35 * ((effective_sulfur - 1.5) / 1.5)
        else:
            score = max(0, 15 - 5 * (effective_sulfur - 3.0))

        # Add concerns
        if effective_sulfur > 2.0:
            concerns.append(
                f"HIGH SULFUR: {effective_sulfur:.2f}% - severe sulfidation risk, "
                "consider desulfurization or fuel additive"
            )
        elif effective_sulfur > 1.0:
            concerns.append(
                f"ELEVATED SULFUR: {effective_sulfur:.2f}% - increased corrosion rate, "
                "monitor tube wall thickness"
            )

        return max(0, min(100, score))

    def _calculate_metals_score(
        self,
        fuel: FuelProperties,
        concerns: List[str]
    ) -> float:
        """Calculate metals penalty score."""
        # Vanadium is most critical for high-temperature corrosion
        v_score = 100 - min(80, fuel.vanadium_ppm * 0.8)  # 100 ppm = 20

        # Sodium and potassium cause hot corrosion
        na_k_combined = fuel.sodium_ppm + fuel.potassium_ppm
        na_score = 100 - min(60, na_k_combined * 2)  # 30 ppm = 40

        # Nickel indicates catalyst fines (FCC slurry oil)
        ni_score = 100 - min(40, fuel.nickel_ppm * 0.4)

        # Weighted combination (vanadium most critical)
        metals_score = 0.5 * v_score + 0.35 * na_score + 0.15 * ni_score

        # Concerns
        if fuel.vanadium_ppm > 100:
            concerns.append(
                f"HIGH VANADIUM: {fuel.vanadium_ppm:.0f} ppm - "
                "severe hot corrosion risk above 600C, use vanadium inhibitor"
            )
        elif fuel.vanadium_ppm > 50:
            concerns.append(
                f"ELEVATED VANADIUM: {fuel.vanadium_ppm:.0f} ppm - "
                "increased refractory attack, monitor tube temperatures"
            )

        if na_k_combined > 20:
            concerns.append(
                f"HIGH ALKALI METALS: Na+K={na_k_combined:.0f} ppm - "
                "accelerated hot corrosion, reduce flame temperature"
            )

        return max(0, min(100, metals_score))

    def _calculate_ash_score(
        self,
        fuel: FuelProperties,
        concerns: List[str]
    ) -> float:
        """Calculate ash penalty score."""
        ash_pct = fuel.ash_pct

        if ash_pct <= 0.01:
            score = 100.0
        elif ash_pct <= 0.05:
            score = 100 - 20 * (ash_pct / 0.05)
        elif ash_pct <= 0.2:
            score = 80 - 40 * ((ash_pct - 0.05) / 0.15)
        else:
            score = max(0, 40 - 50 * (ash_pct - 0.2))

        # Carbon residue adds to fouling
        cr_penalty = min(30, fuel.carbon_residue_pct * 20)
        score -= cr_penalty

        # Asphaltenes increase coking
        asph_penalty = min(20, fuel.asphaltenes_pct * 4)
        score -= asph_penalty

        if fuel.ash_pct > 0.1:
            concerns.append(
                f"HIGH ASH: {fuel.ash_pct:.2f}% - increased fouling rate, "
                "reduce cleaning intervals"
            )

        if fuel.carbon_residue_pct > 1.0:
            concerns.append(
                f"HIGH CARBON RESIDUE: {fuel.carbon_residue_pct:.1f}% - "
                "coking tendency, ensure proper atomization"
            )

        return max(0, min(100, score))

    def _calculate_water_score(
        self,
        fuel: FuelProperties,
        concerns: List[str]
    ) -> float:
        """Calculate water/contamination score."""
        water_pct = fuel.water_pct
        sediment_pct = fuel.sediment_pct

        if water_pct <= 0.05:
            water_score = 100.0
        elif water_pct <= 0.3:
            water_score = 100 - 40 * (water_pct / 0.3)
        elif water_pct <= 1.0:
            water_score = 60 - 40 * ((water_pct - 0.3) / 0.7)
        else:
            water_score = max(0, 20 - 10 * (water_pct - 1.0))

        # Sediment penalty
        sed_penalty = min(30, sediment_pct * 100)
        score = water_score - sed_penalty

        if water_pct > 0.5:
            concerns.append(
                f"HIGH WATER: {water_pct:.2f}% - flame instability risk, "
                "improve fuel handling and storage"
            )

        if sediment_pct > 0.1:
            concerns.append(
                f"HIGH SEDIMENT: {sediment_pct:.2f}% - strainer clogging risk, "
                "improve filtration"
            )

        return max(0, min(100, score))

    def _calculate_heating_value_score(
        self,
        fuel: FuelProperties,
        concerns: List[str]
    ) -> float:
        """Calculate heating value consistency score."""
        hv = fuel.heating_value_mj_kg
        baseline_hv = self.baseline["heating_value_mj_kg"]

        # Score based on deviation from baseline
        deviation_pct = abs(hv - baseline_hv) / baseline_hv * 100

        if deviation_pct <= 2:
            score = 100.0
        elif deviation_pct <= 5:
            score = 100 - 10 * (deviation_pct - 2) / 3
        elif deviation_pct <= 10:
            score = 90 - 20 * (deviation_pct - 5) / 5
        elif deviation_pct <= 20:
            score = 70 - 30 * (deviation_pct - 10) / 10
        else:
            score = max(0, 40 - (deviation_pct - 20))

        if deviation_pct > 10:
            concerns.append(
                f"HEATING VALUE DEVIATION: {deviation_pct:.1f}% from baseline - "
                "adjust burner controls for stable combustion"
            )

        return max(0, min(100, score))

    def analyze_degradation_impact(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> DegradationImpact:
        """
        Analyze fuel-induced degradation impact on burner components.

        Calculates life reduction factor and identifies primary damage
        mechanisms based on fuel contaminants and operating conditions.

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            DegradationImpact with life reduction and mechanism analysis
        """
        logger.info(
            f"Analyzing degradation impact for {fuel.fuel_type.value} "
            f"at {conditions.flame_temperature_c}C"
        )

        # Calculate individual mechanism contributions
        sulfidation = self._calculate_sulfidation_impact(fuel, conditions)
        vanadium = self._calculate_vanadium_impact(fuel, conditions)
        ash = self._calculate_ash_impact(fuel, conditions)
        coking = self._calculate_coking_impact(fuel, conditions)

        # Calculate corrosion rate (mm/year)
        corrosion_rate = self._calculate_corrosion_rate(fuel, conditions)

        # Calculate fouling factor
        fouling = self.predict_fouling_rate(fuel, conditions)

        # Calculate life reduction factor (multiplicative contributions)
        # LRF = 1 + sum of individual impacts
        life_reduction = 1.0 + sulfidation + vanadium + ash + coking

        # Clamp to reasonable range
        life_reduction = max(0.5, min(5.0, life_reduction))

        # Normalize contributions
        total_contribution = sulfidation + vanadium + ash + coking
        if total_contribution > 0:
            sulfidation_norm = sulfidation / total_contribution
            vanadium_norm = vanadium / total_contribution
            ash_norm = ash / total_contribution
            coking_norm = coking / total_contribution
        else:
            sulfidation_norm = vanadium_norm = ash_norm = coking_norm = 0.0

        # Determine primary damage mechanism
        contributions = {
            DamageMechanism.SULFIDATION: sulfidation,
            DamageMechanism.VANADIUM_ATTACK: vanadium,
            DamageMechanism.ASH_DEPOSITION: ash,
            DamageMechanism.COKING: coking,
        }
        primary_mechanism = max(contributions, key=contributions.get)

        # Determine severity
        if life_reduction >= 3.0:
            severity = ImpactSeverity.SEVERE
        elif life_reduction >= 2.0:
            severity = ImpactSeverity.HIGH
        elif life_reduction >= 1.5:
            severity = ImpactSeverity.MODERATE
        elif life_reduction >= 1.1:
            severity = ImpactSeverity.LOW
        else:
            severity = ImpactSeverity.NEGLIGIBLE

        # Generate recommendations
        recommendations = self._generate_recommendations(
            fuel, conditions, life_reduction, primary_mechanism, corrosion_rate
        )

        # Estimate annual impact cost
        base_maintenance_cost = 50000  # USD
        annual_impact = base_maintenance_cost * (life_reduction - 1.0)

        # Provenance
        provenance_hash = self._calculate_provenance(
            "degradation_impact",
            {
                "fuel_type": fuel.fuel_type.value,
                "life_reduction": life_reduction,
                "corrosion_rate": corrosion_rate,
                "primary_mechanism": primary_mechanism.value,
            }
        )

        logger.info(
            f"Degradation analysis: LRF={life_reduction:.2f}, "
            f"primary={primary_mechanism.value}, severity={severity.value}"
        )

        return DegradationImpact(
            life_reduction_factor=round(life_reduction, 3),
            sulfidation_contribution=round(sulfidation_norm, 3),
            vanadium_contribution=round(vanadium_norm, 3),
            ash_contribution=round(ash_norm, 3),
            coking_contribution=round(coking_norm, 3),
            corrosion_rate_mm_year=round(corrosion_rate, 4),
            fouling_factor_m2k_w=round(fouling.fouling_factor_m2k_w, 6),
            overall_severity=severity,
            primary_mechanism=primary_mechanism,
            recommendations=recommendations,
            estimated_annual_impact_usd=round(annual_impact, 0),
            provenance_hash=provenance_hash,
        )

    def _calculate_sulfidation_impact(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> float:
        """
        Calculate sulfidation damage contribution.

        Sulfidation rate follows modified Arrhenius:
        Rate = k * [S]^n * exp(-Ea/RT)

        where n ~ 0.5 for H2S and 0.7 for organic sulfur.

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            Sulfidation impact factor (0 = none, >0 = increased degradation)
        """
        sulfur = fuel.sulfur_pct
        baseline = self.baseline["sulfur_pct"]

        if sulfur <= 0.01:
            return 0.0

        # Temperature factor (Arrhenius)
        temp_k = conditions.flame_temperature_c + 273.15
        ea = ACTIVATION_ENERGIES["sulfur_corrosion"]
        temp_factor = math.exp(-ea / (GAS_CONSTANT_R * temp_k)) / \
                      math.exp(-ea / (GAS_CONSTANT_R * self.reference_temp_k))

        # Sulfur concentration factor (power law, n=0.5)
        conc_factor = (sulfur / max(0.1, baseline)) ** 0.5

        # Combined impact (scaled to reasonable range)
        impact = CONTAMINANT_MULTIPLIERS["sulfur"] * conc_factor * temp_factor * 0.3

        return max(0, min(1.5, impact))

    def _calculate_vanadium_impact(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> float:
        """
        Calculate vanadium attack contribution.

        Vanadium pentoxide (V2O5) is highly corrosive above ~600C.
        Forms low-melting point ash that attacks refractory and tubes.

        Impact is exponential above critical temperature.

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            Vanadium impact factor
        """
        vanadium = fuel.vanadium_ppm
        baseline = self.baseline["vanadium_ppm"]

        if vanadium <= 1.0:
            return 0.0

        # Critical temperature effect (onset ~600C, severe ~700C)
        temp_c = conditions.tube_metal_temperature_c
        if temp_c < 550:
            temp_factor = 0.1
        elif temp_c < 650:
            temp_factor = 0.1 + 0.9 * (temp_c - 550) / 100
        elif temp_c < 750:
            temp_factor = 1.0 + (temp_c - 650) / 100
        else:
            temp_factor = 2.0 + (temp_c - 750) / 50

        # Concentration factor
        conc_factor = (vanadium / max(1.0, baseline))

        # Combined impact
        impact = CONTAMINANT_MULTIPLIERS["vanadium"] * conc_factor * temp_factor * 0.05

        return max(0, min(2.0, impact))

    def _calculate_ash_impact(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> float:
        """
        Calculate ash deposition impact.

        Ash causes:
        - Fouling (thermal resistance increase)
        - Erosion from impinging particles
        - Hot corrosion with alkali metals

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            Ash impact factor
        """
        ash = fuel.ash_pct
        baseline = self.baseline["ash_pct"]

        if ash <= 0.005:
            return 0.0

        # Ash composition matters (sodium/potassium worsen impact)
        alkali_factor = 1.0 + (fuel.sodium_ppm + fuel.potassium_ppm) / 100

        # Velocity factor (higher velocity = more erosion)
        velocity_factor = (conditions.firing_rate_pct / 80) ** 1.5

        # Concentration factor
        conc_factor = (ash / max(0.01, baseline))

        # Combined impact
        impact = (
            CONTAMINANT_MULTIPLIERS["ash"] *
            conc_factor * alkali_factor * velocity_factor * 0.15
        )

        return max(0, min(1.0, impact))

    def _calculate_coking_impact(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> float:
        """
        Calculate coking tendency impact.

        Coking from:
        - High carbon residue fuels
        - Asphaltene decomposition
        - Poor atomization/combustion

        Uses modified Conradson Carbon Residue correlation.

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            Coking impact factor
        """
        ccr = fuel.carbon_residue_pct
        asph = fuel.asphaltenes_pct

        if ccr <= 0.05 and asph <= 0.5:
            return 0.0

        # Carbon residue factor (exponential above threshold)
        if ccr <= 0.5:
            ccr_factor = ccr * 2
        else:
            ccr_factor = 1.0 + (ccr - 0.5) * 3

        # Asphaltene factor
        asph_factor = 1.0 + asph * 0.3

        # Temperature factor (higher temp reduces coking due to combustion)
        temp_c = conditions.flame_temperature_c
        if temp_c >= 1300:
            temp_factor = 0.7
        elif temp_c >= 1100:
            temp_factor = 0.7 + 0.3 * (1300 - temp_c) / 200
        else:
            temp_factor = 1.0 + (1100 - temp_c) / 200

        # Combined impact
        impact = ccr_factor * asph_factor * temp_factor * 0.1

        return max(0, min(1.5, impact))

    def _calculate_corrosion_rate(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> float:
        """
        Calculate estimated corrosion rate (mm/year).

        Combines sulfidation and hot corrosion rates using
        API 571 correlations and Nelson curves.

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            Corrosion rate in mm/year
        """
        # Sulfidation corrosion (modified McConomy curves)
        sulfur = fuel.sulfur_pct
        temp_c = conditions.tube_metal_temperature_c

        # Base rate at reference conditions (0.5% S, 450C)
        base_rate = 0.05  # mm/year

        # Temperature correction (Arrhenius)
        temp_k = temp_c + 273.15
        ref_temp_k = 450 + 273.15
        ea = ACTIVATION_ENERGIES["sulfur_corrosion"]
        temp_correction = math.exp(
            -ea / GAS_CONSTANT_R * (1/temp_k - 1/ref_temp_k)
        )

        # Sulfur concentration correction
        sulfur_correction = (sulfur / 0.5) ** 0.5

        # Sulfidation rate
        sulfidation_rate = base_rate * temp_correction * sulfur_correction

        # Vanadium/hot corrosion (if applicable)
        if fuel.vanadium_ppm > 10 and temp_c > 550:
            v_rate = 0.02 * (fuel.vanadium_ppm / 50) * (
                (temp_c - 550) / 200
            ) ** 2
        else:
            v_rate = 0.0

        # Total rate
        total_rate = sulfidation_rate + v_rate

        return max(0, min(5.0, total_rate))

    def predict_fouling_rate(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> FoulingResult:
        """
        Predict fouling rate using Kern-Seaton model.

        Kern-Seaton asymptotic fouling model:
        Rf(t) = Rf_inf * (1 - exp(-t/tau))

        where:
        - Rf_inf = asymptotic fouling resistance
        - tau = time constant

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            FoulingResult with fouling predictions
        """
        logger.debug("Predicting fouling rate with Kern-Seaton model")

        # Base asymptotic fouling (m2-K/W)
        # Typical values: 0.0001 - 0.0005 for fuel oil
        base_fouling = 0.0002

        # Ash contribution (proportional to ash content)
        ash_factor = 1.0 + fuel.ash_pct * 50  # 1% ash -> 50x increase

        # Velocity factor (higher velocity reduces deposition)
        velocity = conditions.firing_rate_pct / 100
        velocity_factor = 1.0 / max(0.5, velocity) ** 0.4

        # Temperature factor (higher wall temp can increase or decrease)
        wall_temp = conditions.tube_metal_temperature_c
        if wall_temp < 300:
            temp_factor = 1.0
        elif wall_temp < 500:
            temp_factor = 1.0 + (wall_temp - 300) / 400  # Increases
        else:
            temp_factor = 1.5 - (wall_temp - 500) / 500  # Decreases (sintering)
        temp_factor = max(0.5, temp_factor)

        # Asymptotic fouling resistance
        rf_inf = base_fouling * ash_factor * velocity_factor * temp_factor

        # Time constant (hours) - typically 1000-5000 hours
        # Faster buildup with more ash
        base_tau = 2000  # hours
        tau = base_tau / (1 + fuel.ash_pct * 20)
        tau = max(500, min(10000, tau))

        # Calculate current fouling (assuming steady-state)
        # Using 90% of asymptotic as "current"
        rf_current = rf_inf * 0.9

        # Efficiency loss (approximate: 1% per 0.0001 m2-K/W)
        efficiency_loss = rf_current * 10000  # Percentage points

        # Fouling level classification
        if rf_current < 0.0001:
            level = FoulingLevel.CLEAN
        elif rf_current < 0.0002:
            level = FoulingLevel.LIGHT
        elif rf_current < 0.0004:
            level = FoulingLevel.MODERATE
        elif rf_current < 0.0008:
            level = FoulingLevel.HEAVY
        else:
            level = FoulingLevel.SEVERE

        # Cleaning interval (based on reaching moderate fouling)
        target_rf = 0.0003  # Target fouling resistance
        if rf_inf > target_rf:
            cleaning_interval = tau * math.log(rf_inf / (rf_inf - target_rf))
            cleaning_interval = max(1000, min(20000, cleaning_interval))
        else:
            cleaning_interval = None  # No cleaning needed

        return FoulingResult(
            fouling_factor_m2k_w=rf_current,
            asymptotic_fouling=rf_inf,
            time_constant_hours=tau,
            fouling_level=level,
            cleaning_interval_hours=cleaning_interval,
            efficiency_loss_pct=round(efficiency_loss, 2),
        )

    def calculate_coking_index(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> CokingResult:
        """
        Calculate coking tendency index.

        Coking index based on:
        - Conradson Carbon Residue (CCR)
        - Asphaltene content
        - Operating temperature
        - Residence time

        Args:
            fuel: Fuel properties
            conditions: Operating conditions

        Returns:
            CokingResult with coking tendency analysis
        """
        ccr = fuel.carbon_residue_pct
        asph = fuel.asphaltenes_pct

        # Coking index formula (empirical)
        # CI = CCR * (1 + 0.5*Asph) * temp_factor
        temp_c = conditions.flame_temperature_c

        # Temperature factor (coking peaks at intermediate temps)
        if temp_c < 400:
            temp_factor = 0.5 + temp_c / 800
        elif temp_c < 700:
            temp_factor = 1.0 + (temp_c - 400) / 600
        else:
            temp_factor = 1.5 - (temp_c - 700) / 1000

        temp_factor = max(0.3, min(2.0, temp_factor))

        # Calculate coking index (0-10 scale)
        coking_index = ccr * (1 + 0.5 * asph) * temp_factor
        coking_index = min(10, coking_index * 5)  # Scale to 0-10

        # Coke formation rate (g/m2/h) - empirical
        # Base rate ~ 0.1 g/m2/h at CCR=1%
        base_rate = 0.1
        coke_rate = base_rate * ccr * (1 + asph * 0.3) * temp_factor

        # Severity classification
        if coking_index < 1:
            severity = ImpactSeverity.NEGLIGIBLE
        elif coking_index < 2:
            severity = ImpactSeverity.LOW
        elif coking_index < 4:
            severity = ImpactSeverity.MODERATE
        elif coking_index < 7:
            severity = ImpactSeverity.HIGH
        else:
            severity = ImpactSeverity.SEVERE

        # Decoking interval (based on acceptable coke buildup)
        max_coke_thickness_mm = 2.0
        coke_density = 1.5  # g/cm3 = 1500 kg/m3
        max_coke_loading = max_coke_thickness_mm * coke_density * 1000  # g/m2

        if coke_rate > 0:
            decoking_interval = max_coke_loading / coke_rate
            decoking_interval = max(1000, min(50000, decoking_interval))
        else:
            decoking_interval = None

        return CokingResult(
            coking_index=round(coking_index, 2),
            coke_formation_rate_g_m2_h=round(coke_rate, 4),
            decoking_interval_hours=decoking_interval,
            severity=severity,
        )

    def analyze_fuel_switching(
        self,
        original_fuel: FuelProperties,
        new_fuel: FuelProperties,
        conditions: OperatingConditions
    ) -> FuelSwitchingImpact:
        """
        Analyze impact of switching between fuels.

        Compares degradation impacts and identifies risks/benefits
        of fuel switching.

        Args:
            original_fuel: Current fuel properties
            new_fuel: Proposed new fuel properties
            conditions: Operating conditions

        Returns:
            FuelSwitchingImpact analysis
        """
        logger.info(
            f"Analyzing fuel switch: {original_fuel.fuel_type.value} -> "
            f"{new_fuel.fuel_type.value}"
        )

        # Calculate quality scores
        original_quality = self.calculate_fuel_quality_score(original_fuel)
        new_quality = self.calculate_fuel_quality_score(new_fuel)
        quality_change = new_quality.overall_score - original_quality.overall_score

        # Calculate degradation impacts
        original_impact = self.analyze_degradation_impact(original_fuel, conditions)
        new_impact = self.analyze_degradation_impact(new_fuel, conditions)

        # Life change factor (>1 = improvement)
        life_change = original_impact.life_reduction_factor / new_impact.life_reduction_factor

        # Efficiency change (from heating value and combustion quality)
        hv_ratio = new_fuel.heating_value_mj_kg / original_fuel.heating_value_mj_kg
        efficiency_change = (hv_ratio - 1) * 100  # Percentage

        # Identify new risks
        new_risks = []
        mitigated_risks = []

        # Sulfur comparison
        if new_fuel.sulfur_pct > original_fuel.sulfur_pct * 1.5:
            new_risks.append(
                f"Increased sulfur: {original_fuel.sulfur_pct:.2f}% -> "
                f"{new_fuel.sulfur_pct:.2f}% (corrosion risk)"
            )
        elif new_fuel.sulfur_pct < original_fuel.sulfur_pct * 0.7:
            mitigated_risks.append(
                f"Reduced sulfur: {original_fuel.sulfur_pct:.2f}% -> "
                f"{new_fuel.sulfur_pct:.2f}%"
            )

        # Vanadium comparison
        if new_fuel.vanadium_ppm > original_fuel.vanadium_ppm * 2:
            new_risks.append(
                f"Increased vanadium: {original_fuel.vanadium_ppm:.0f} -> "
                f"{new_fuel.vanadium_ppm:.0f} ppm (hot corrosion risk)"
            )
        elif new_fuel.vanadium_ppm < original_fuel.vanadium_ppm * 0.5:
            mitigated_risks.append(
                f"Reduced vanadium: {original_fuel.vanadium_ppm:.0f} -> "
                f"{new_fuel.vanadium_ppm:.0f} ppm"
            )

        # Ash comparison
        if new_fuel.ash_pct > original_fuel.ash_pct * 2:
            new_risks.append(
                f"Increased ash: {original_fuel.ash_pct:.3f}% -> "
                f"{new_fuel.ash_pct:.3f}% (fouling risk)"
            )

        # Water comparison
        if new_fuel.water_pct > 0.5 and new_fuel.water_pct > original_fuel.water_pct * 1.5:
            new_risks.append(
                f"Increased water: {new_fuel.water_pct:.2f}% (flame stability risk)"
            )

        # Required modifications
        required_mods = []

        # Fuel type change modifications
        if original_fuel.fuel_type != new_fuel.fuel_type:
            if new_fuel.fuel_type in [FuelType.FUEL_OIL_NO6, FuelType.RESIDUAL_FUEL_OIL]:
                required_mods.append("Fuel heating system for viscosity control")
                required_mods.append("Steam atomization capability")
            if original_fuel.fuel_type in [FuelType.NATURAL_GAS, FuelType.LPG]:
                required_mods.append("Liquid fuel supply system")
                required_mods.append("Atomizer installation")

        # Additive requirements
        if new_fuel.vanadium_ppm > 50:
            required_mods.append("Vanadium inhibitor injection system")
        if new_fuel.sulfur_pct > 2.0:
            required_mods.append("Increased corrosion monitoring")

        # Generate recommendation
        if life_change >= 1.2:
            recommendation = "RECOMMENDED: Fuel switch improves component life"
        elif life_change >= 0.9:
            recommendation = "ACCEPTABLE: Minimal impact on component life"
        elif life_change >= 0.7:
            recommendation = (
                "CAUTION: Fuel switch reduces component life - "
                "evaluate mitigation strategies"
            )
        else:
            recommendation = (
                "NOT RECOMMENDED: Significant life reduction - "
                "consider alternatives"
            )

        # Provenance
        provenance_hash = self._calculate_provenance(
            "fuel_switch",
            {
                "original": original_fuel.fuel_type.value,
                "new": new_fuel.fuel_type.value,
                "life_change": life_change,
            }
        )

        return FuelSwitchingImpact(
            original_fuel=original_fuel.fuel_type,
            new_fuel=new_fuel.fuel_type,
            life_change_factor=round(life_change, 3),
            quality_score_change=round(quality_change, 1),
            efficiency_change_pct=round(efficiency_change, 2),
            new_risks=new_risks,
            mitigated_risks=mitigated_risks,
            required_modifications=required_mods,
            recommendation=recommendation,
            provenance_hash=provenance_hash,
        )

    def calculate_life_reduction_from_contaminants(
        self,
        fuel: FuelProperties
    ) -> Dict[str, float]:
        """
        Calculate life reduction factors for each contaminant.

        Uses the formula:
        LRF_i = 1 + impact_coefficient * (concentration - baseline) / baseline

        Args:
            fuel: Fuel properties

        Returns:
            Dictionary of contaminant -> life reduction factor
        """
        reductions = {}

        # Sulfur
        sulfur_deviation = (fuel.sulfur_pct - self.baseline["sulfur_pct"]) / \
                           max(0.1, self.baseline["sulfur_pct"])
        reductions["sulfur"] = 1.0 + max(0, sulfur_deviation * 0.5)

        # Vanadium
        v_deviation = (fuel.vanadium_ppm - self.baseline["vanadium_ppm"]) / \
                      max(1, self.baseline["vanadium_ppm"])
        reductions["vanadium"] = 1.0 + max(0, v_deviation * 0.3)

        # Sodium
        na_deviation = (fuel.sodium_ppm - self.baseline["sodium_ppm"]) / \
                       max(1, self.baseline["sodium_ppm"])
        reductions["sodium"] = 1.0 + max(0, na_deviation * 0.25)

        # Ash
        ash_deviation = (fuel.ash_pct - self.baseline["ash_pct"]) / \
                        max(0.01, self.baseline["ash_pct"])
        reductions["ash"] = 1.0 + max(0, ash_deviation * 0.2)

        # Carbon residue
        cr_deviation = (fuel.carbon_residue_pct - self.baseline["carbon_residue_pct"]) / \
                       max(0.1, self.baseline["carbon_residue_pct"])
        reductions["carbon_residue"] = 1.0 + max(0, cr_deviation * 0.15)

        # Combined life reduction factor (multiplicative)
        combined_lrf = 1.0
        for lrf in reductions.values():
            combined_lrf *= lrf

        reductions["combined"] = round(combined_lrf, 3)

        return reductions

    def _generate_recommendations(
        self,
        fuel: FuelProperties,
        conditions: OperatingConditions,
        life_reduction: float,
        primary_mechanism: DamageMechanism,
        corrosion_rate: float
    ) -> List[str]:
        """Generate mitigation recommendations based on analysis."""
        recommendations = []

        # Severity-based recommendations
        if life_reduction >= 2.0:
            recommendations.append(
                "CRITICAL: Consider fuel specification change or "
                "additive treatment program"
            )

        # Mechanism-specific recommendations
        if primary_mechanism == DamageMechanism.SULFIDATION:
            if fuel.sulfur_pct > 1.5:
                recommendations.append(
                    "HIGH SULFUR: Consider desulfurization, fuel blending, "
                    "or corrosion inhibitor (e.g., magnesium-based)"
                )
            recommendations.append(
                f"SULFIDATION: Increase tube thickness inspection frequency. "
                f"Estimated corrosion rate: {corrosion_rate:.3f} mm/year"
            )

        elif primary_mechanism == DamageMechanism.VANADIUM_ATTACK:
            recommendations.append(
                "VANADIUM: Apply vanadium inhibitor (Mg:V ratio 3:1 minimum). "
                "Monitor tube metal temperatures - keep below 650C"
            )
            if conditions.tube_metal_temperature_c > 650:
                recommendations.append(
                    f"REDUCE TEMPERATURE: Current TMT {conditions.tube_metal_temperature_c}C "
                    "exceeds safe limit for vanadium-bearing fuels"
                )

        elif primary_mechanism == DamageMechanism.ASH_DEPOSITION:
            recommendations.append(
                "ASH FOULING: Implement regular sootblowing schedule. "
                "Consider online cleaning during operation"
            )
            recommendations.append(
                "FUEL TREATMENT: Consider ash modifier additive to "
                "raise ash fusion temperature"
            )

        elif primary_mechanism == DamageMechanism.COKING:
            if fuel.carbon_residue_pct > 1.0:
                recommendations.append(
                    "COKING RISK: Ensure proper atomization temperature. "
                    f"CCR={fuel.carbon_residue_pct:.1f}% requires "
                    "enhanced fuel heating"
                )
            recommendations.append(
                "COMBUSTION: Maintain 15-20% excess air to ensure complete combustion"
            )

        # Operating condition recommendations
        if conditions.thermal_cycles_per_year > 200:
            recommendations.append(
                f"THERMAL CYCLING: {conditions.thermal_cycles_per_year:.0f} cycles/year "
                "is high. Consider startup/shutdown optimization"
            )

        if conditions.firing_rate_pct > 95:
            recommendations.append(
                "HIGH FIRING RATE: Consider load reduction to extend component life"
            )

        # Water content
        if fuel.water_pct > 0.5:
            recommendations.append(
                f"WATER CONTENT: {fuel.water_pct:.2f}% may cause flame instability. "
                "Improve fuel dehydration"
            )

        return recommendations

    def _calculate_provenance(
        self,
        calculation_type: str,
        inputs: Dict[str, Any]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        input_str = "|".join(f"{k}:{v}" for k, v in sorted(inputs.items()))
        provenance_str = f"fuel_impact|{calculation_type}|{input_str}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_fuel_quality_check(
    sulfur_pct: float = 0.5,
    vanadium_ppm: float = 10.0,
    ash_pct: float = 0.02,
    fuel_type: FuelType = FuelType.FUEL_OIL_NO2
) -> FuelQualityScore:
    """
    Quick fuel quality check with common parameters.

    Args:
        sulfur_pct: Sulfur content (weight %)
        vanadium_ppm: Vanadium content (ppm)
        ash_pct: Ash content (weight %)
        fuel_type: Fuel type

    Returns:
        FuelQualityScore

    Example:
        >>> score = quick_fuel_quality_check(sulfur_pct=2.0, vanadium_ppm=50)
        >>> print(f"Quality: {score.overall_score:.0f} ({score.quality_class})")
    """
    fuel = FuelProperties(
        fuel_type=fuel_type,
        sulfur_pct=sulfur_pct,
        vanadium_ppm=vanadium_ppm,
        ash_pct=ash_pct,
    )

    analyzer = FuelImpactAnalyzer()
    return analyzer.calculate_fuel_quality_score(fuel)


def calculate_sulfur_corrosion_rate(
    sulfur_pct: float,
    temperature_c: float
) -> float:
    """
    Calculate sulfidation corrosion rate.

    Uses McConomy curves correlation.

    Args:
        sulfur_pct: Sulfur content (weight %)
        temperature_c: Metal temperature (C)

    Returns:
        Corrosion rate in mm/year

    Example:
        >>> rate = calculate_sulfur_corrosion_rate(2.0, 500)
        >>> print(f"Corrosion rate: {rate:.3f} mm/year")
    """
    # Modified McConomy correlation
    base_rate = 0.05  # mm/year at 0.5% S, 450C

    temp_k = temperature_c + 273.15
    ref_temp_k = 450 + 273.15
    ea = ACTIVATION_ENERGIES["sulfur_corrosion"]

    temp_factor = math.exp(
        -ea / GAS_CONSTANT_R * (1/temp_k - 1/ref_temp_k)
    )
    sulfur_factor = (sulfur_pct / 0.5) ** 0.5

    rate = base_rate * temp_factor * sulfur_factor

    return max(0, min(5.0, rate))


def get_vanadium_inhibitor_ratio(
    vanadium_ppm: float
) -> float:
    """
    Get recommended Mg:V ratio for vanadium inhibition.

    Industry standard: minimum 3:1 Mg:V ratio by weight.

    Args:
        vanadium_ppm: Vanadium content in fuel

    Returns:
        Recommended Mg dosage (ppm)

    Example:
        >>> mg_dose = get_vanadium_inhibitor_ratio(100)
        >>> print(f"Recommended Mg dose: {mg_dose:.0f} ppm")
    """
    if vanadium_ppm <= 5:
        ratio = 2.0  # Lower ratio for low vanadium
    elif vanadium_ppm <= 50:
        ratio = 3.0  # Standard ratio
    else:
        ratio = 3.5  # Higher ratio for high vanadium

    return vanadium_ppm * ratio


def estimate_fouling_efficiency_loss(
    ash_pct: float,
    operating_hours: float,
    firing_rate_pct: float = 80.0
) -> float:
    """
    Estimate efficiency loss from fouling.

    Args:
        ash_pct: Fuel ash content (%)
        operating_hours: Hours since last cleaning
        firing_rate_pct: Firing rate (%)

    Returns:
        Efficiency loss (percentage points)

    Example:
        >>> loss = estimate_fouling_efficiency_loss(0.1, 5000)
        >>> print(f"Efficiency loss: {loss:.1f}%")
    """
    # Kern-Seaton simplified model
    tau = 2000 / (1 + ash_pct * 20)  # Time constant
    rf_inf = 0.0002 * (1 + ash_pct * 50)  # Asymptotic fouling

    rf = rf_inf * (1 - math.exp(-operating_hours / tau))

    # Efficiency loss: ~1% per 0.0001 m2-K/W
    efficiency_loss = rf * 10000

    # Firing rate correction (higher rate = more deposition)
    rate_factor = (firing_rate_pct / 80) ** 0.5

    return efficiency_loss * rate_factor
