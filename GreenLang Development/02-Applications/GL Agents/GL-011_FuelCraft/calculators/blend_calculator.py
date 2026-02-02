"""
GL-011 FuelCraft - Blend Calculator

Deterministic calculations for fuel blending:
- Linear blending for heating values (mass-weighted)
- Non-linear property blending (viscosity, flash point)
- Quality constraint validation (sulfur, ash, water)
- Safety constraint checking (flash point, vapor pressure)

Standards:
- ASTM D341 (Viscosity-Temperature Charts)
- ASTM D92/D93 (Flash Point)
- IMO MEPC.320(74) (MARPOL Fuel Sulfur)
- ISO 8217 (Marine Fuels)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import math


class BlendingMethod(Enum):
    """Blending calculation method."""
    LINEAR_MASS = "linear_mass"          # Linear on mass fraction
    LINEAR_VOLUME = "linear_volume"      # Linear on volume fraction
    LINEAR_ENERGY = "linear_energy"      # Linear on energy fraction
    REFUTAS_VISCOSITY = "refutas"        # ASTM D341 viscosity blending


class ConstraintType(Enum):
    """Type of blend constraint."""
    QUALITY = "quality"   # Product quality constraint
    SAFETY = "safety"     # Safety-related constraint
    REGULATORY = "regulatory"  # Regulatory compliance


@dataclass(frozen=True)
class QualityConstraint:
    """
    Quality constraint for blend validation.
    """
    property_name: str
    min_value: Optional[Decimal]
    max_value: Optional[Decimal]
    unit: str
    constraint_type: ConstraintType
    standard: str  # Reference standard
    is_mandatory: bool = True

    def validate(self, value: Decimal) -> Tuple[bool, str]:
        """Check if value meets constraint."""
        if self.min_value is not None and value < self.min_value:
            return False, f"{self.property_name} ({value}) below minimum ({self.min_value})"
        if self.max_value is not None and value > self.max_value:
            return False, f"{self.property_name} ({value}) exceeds maximum ({self.max_value})"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "min_value": str(self.min_value) if self.min_value else None,
            "max_value": str(self.max_value) if self.max_value else None,
            "unit": self.unit,
            "constraint_type": self.constraint_type.value,
            "standard": self.standard,
            "is_mandatory": self.is_mandatory
        }


@dataclass(frozen=True)
class SafetyConstraint:
    """
    Safety constraint for blend validation.
    """
    property_name: str
    min_value: Optional[Decimal]
    max_value: Optional[Decimal]
    unit: str
    safety_standard: str  # NFPA, IMO, etc.
    sil_level: Optional[int] = None  # SIL rating if applicable

    def validate(self, value: Decimal) -> Tuple[bool, str]:
        """Check if value meets safety constraint."""
        if self.min_value is not None and value < self.min_value:
            return False, f"SAFETY: {self.property_name} ({value}) below safe minimum ({self.min_value})"
        if self.max_value is not None and value > self.max_value:
            return False, f"SAFETY: {self.property_name} ({value}) exceeds safe maximum ({self.max_value})"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "min_value": str(self.min_value) if self.min_value else None,
            "max_value": str(self.max_value) if self.max_value else None,
            "unit": self.unit,
            "safety_standard": self.safety_standard,
            "sil_level": self.sil_level
        }


# Default quality constraints for marine fuels (ISO 8217)
DEFAULT_QUALITY_CONSTRAINTS: List[QualityConstraint] = [
    QualityConstraint(
        property_name="sulfur_content",
        min_value=None,
        max_value=Decimal("0.50"),  # IMO 2020 global cap
        unit="wt%",
        constraint_type=ConstraintType.REGULATORY,
        standard="IMO MEPC.320(74)"
    ),
    QualityConstraint(
        property_name="water_content",
        min_value=None,
        max_value=Decimal("0.50"),
        unit="vol%",
        constraint_type=ConstraintType.QUALITY,
        standard="ISO 8217:2017"
    ),
    QualityConstraint(
        property_name="ash_content",
        min_value=None,
        max_value=Decimal("0.10"),
        unit="wt%",
        constraint_type=ConstraintType.QUALITY,
        standard="ISO 8217:2017"
    ),
    QualityConstraint(
        property_name="viscosity_50c",
        min_value=Decimal("10.0"),
        max_value=Decimal("700.0"),
        unit="cSt",
        constraint_type=ConstraintType.QUALITY,
        standard="ISO 8217:2017"
    ),
]

# Default safety constraints
DEFAULT_SAFETY_CONSTRAINTS: List[SafetyConstraint] = [
    SafetyConstraint(
        property_name="flash_point",
        min_value=Decimal("60.0"),  # SOLAS requirement
        max_value=None,
        unit="C",
        safety_standard="SOLAS II-2/4"
    ),
    SafetyConstraint(
        property_name="vapor_pressure",
        min_value=None,
        max_value=Decimal("100.0"),  # kPa
        unit="kPa",
        safety_standard="NFPA 30"
    ),
]


@dataclass
class BlendComponent:
    """
    Single component in a fuel blend.
    """
    component_id: str
    fuel_type: str
    mass_kg: Decimal
    lhv_mj_kg: Decimal
    hhv_mj_kg: Decimal
    density_kg_m3: Decimal
    # Quality properties
    sulfur_wt_pct: Decimal
    ash_wt_pct: Decimal
    water_vol_pct: Decimal
    viscosity_50c_cst: Decimal
    # Safety properties
    flash_point_c: Decimal
    vapor_pressure_kpa: Decimal
    # Carbon intensity
    carbon_intensity_kg_co2e_mj: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "fuel_type": self.fuel_type,
            "mass_kg": str(self.mass_kg),
            "lhv_mj_kg": str(self.lhv_mj_kg),
            "hhv_mj_kg": str(self.hhv_mj_kg),
            "density_kg_m3": str(self.density_kg_m3),
            "sulfur_wt_pct": str(self.sulfur_wt_pct),
            "ash_wt_pct": str(self.ash_wt_pct),
            "water_vol_pct": str(self.water_vol_pct),
            "viscosity_50c_cst": str(self.viscosity_50c_cst),
            "flash_point_c": str(self.flash_point_c),
            "vapor_pressure_kpa": str(self.vapor_pressure_kpa),
            "carbon_intensity_kg_co2e_mj": str(self.carbon_intensity_kg_co2e_mj)
        }


@dataclass
class BlendInput:
    """Input for blend calculation."""
    components: List[BlendComponent]
    blend_fractions: List[Decimal]  # Mass fractions (must sum to 1)
    quality_constraints: List[QualityConstraint] = field(default_factory=list)
    safety_constraints: List[SafetyConstraint] = field(default_factory=list)


@dataclass
class BlendResult:
    """
    Result of blend calculation with validation.
    """
    # Blended properties
    total_mass_kg: Decimal
    total_energy_mj: Decimal
    blend_lhv_mj_kg: Decimal
    blend_hhv_mj_kg: Decimal
    blend_density_kg_m3: Decimal
    # Blended quality
    blend_sulfur_wt_pct: Decimal
    blend_ash_wt_pct: Decimal
    blend_water_vol_pct: Decimal
    blend_viscosity_50c_cst: Decimal
    # Blended safety
    blend_flash_point_c: Decimal
    blend_vapor_pressure_kpa: Decimal
    # Carbon intensity
    blend_carbon_intensity: Decimal  # kgCO2e/MJ
    total_emissions_kg_co2e: Decimal
    # Validation
    quality_valid: bool
    quality_violations: List[str]
    safety_valid: bool
    safety_violations: List[str]
    # Provenance
    component_contributions: List[Dict[str, Any]]
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "total_mass_kg": str(self.total_mass_kg),
            "total_energy_mj": str(self.total_energy_mj),
            "blend_lhv_mj_kg": str(self.blend_lhv_mj_kg),
            "blend_sulfur_wt_pct": str(self.blend_sulfur_wt_pct),
            "blend_carbon_intensity": str(self.blend_carbon_intensity),
            "quality_valid": self.quality_valid,
            "safety_valid": self.safety_valid,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mass_kg": str(self.total_mass_kg),
            "total_energy_mj": str(self.total_energy_mj),
            "blend_lhv_mj_kg": str(self.blend_lhv_mj_kg),
            "blend_hhv_mj_kg": str(self.blend_hhv_mj_kg),
            "blend_density_kg_m3": str(self.blend_density_kg_m3),
            "blend_sulfur_wt_pct": str(self.blend_sulfur_wt_pct),
            "blend_ash_wt_pct": str(self.blend_ash_wt_pct),
            "blend_water_vol_pct": str(self.blend_water_vol_pct),
            "blend_viscosity_50c_cst": str(self.blend_viscosity_50c_cst),
            "blend_flash_point_c": str(self.blend_flash_point_c),
            "blend_vapor_pressure_kpa": str(self.blend_vapor_pressure_kpa),
            "blend_carbon_intensity": str(self.blend_carbon_intensity),
            "total_emissions_kg_co2e": str(self.total_emissions_kg_co2e),
            "quality_valid": self.quality_valid,
            "quality_violations": self.quality_violations,
            "safety_valid": self.safety_valid,
            "safety_violations": self.safety_violations,
            "component_contributions": self.component_contributions,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat()
        }


class BlendCalculator:
    """
    Deterministic fuel blend calculator.

    Provides ZERO-HALLUCINATION calculations for:
    - Linear blending of heating values (mass-weighted)
    - Non-linear viscosity blending (ASTM D341 Refutas)
    - Quality property blending (sulfur, ash, water)
    - Safety constraint validation (flash point)
    - Carbon intensity blending

    All calculations use Decimal arithmetic.
    """

    NAME: str = "BlendCalculator"
    VERSION: str = "1.0.0"

    PRECISION: int = 6

    def __init__(
        self,
        quality_constraints: Optional[List[QualityConstraint]] = None,
        safety_constraints: Optional[List[SafetyConstraint]] = None
    ):
        """
        Initialize calculator.

        Args:
            quality_constraints: Quality constraints to validate
            safety_constraints: Safety constraints to validate
        """
        self._quality_constraints = quality_constraints or DEFAULT_QUALITY_CONSTRAINTS
        self._safety_constraints = safety_constraints or DEFAULT_SAFETY_CONSTRAINTS

    def calculate(
        self,
        blend_input: BlendInput,
        precision: int = 6
    ) -> BlendResult:
        """
        Calculate blend properties - DETERMINISTIC.

        Args:
            blend_input: Blend components and fractions
            precision: Output decimal places

        Returns:
            BlendResult with full provenance

        Raises:
            ValueError: If fractions don't sum to 1 or component mismatch
        """
        components = blend_input.components
        fractions = blend_input.blend_fractions

        # Validate inputs
        if len(components) != len(fractions):
            raise ValueError("Number of components must match number of fractions")

        fraction_sum = sum(fractions)
        if abs(fraction_sum - Decimal("1.0")) > Decimal("0.0001"):
            raise ValueError(f"Blend fractions must sum to 1.0, got {fraction_sum}")

        steps: List[Dict[str, Any]] = []
        contributions: List[Dict[str, Any]] = []

        # Step 1: Calculate total mass and component contributions
        total_mass = Decimal("0")
        for i, (comp, frac) in enumerate(zip(components, fractions)):
            total_mass += comp.mass_kg * frac
            contributions.append({
                "component_id": comp.component_id,
                "fraction": str(frac),
                "mass_kg": str(comp.mass_kg * frac)
            })

        steps.append({
            "step": 1,
            "operation": "calculate_total_mass",
            "total_mass_kg": str(total_mass)
        })

        # Step 2: Calculate mass-weighted heating values (LINEAR)
        blend_lhv = Decimal("0")
        blend_hhv = Decimal("0")
        for comp, frac in zip(components, fractions):
            blend_lhv += comp.lhv_mj_kg * frac
            blend_hhv += comp.hhv_mj_kg * frac

        steps.append({
            "step": 2,
            "operation": "linear_blend_heating_values",
            "method": "mass_weighted",
            "blend_lhv_mj_kg": str(blend_lhv),
            "blend_hhv_mj_kg": str(blend_hhv)
        })

        # Step 3: Calculate total energy
        total_energy = total_mass * blend_lhv

        # Step 4: Calculate mass-weighted density
        blend_density = Decimal("0")
        for comp, frac in zip(components, fractions):
            blend_density += comp.density_kg_m3 * frac

        steps.append({
            "step": 4,
            "operation": "linear_blend_density",
            "blend_density_kg_m3": str(blend_density)
        })

        # Step 5: Calculate quality properties (LINEAR for most)
        blend_sulfur = self._linear_blend(
            [(comp.sulfur_wt_pct, frac) for comp, frac in zip(components, fractions)]
        )
        blend_ash = self._linear_blend(
            [(comp.ash_wt_pct, frac) for comp, frac in zip(components, fractions)]
        )
        blend_water = self._linear_blend(
            [(comp.water_vol_pct, frac) for comp, frac in zip(components, fractions)]
        )

        steps.append({
            "step": 5,
            "operation": "linear_blend_quality",
            "blend_sulfur_wt_pct": str(blend_sulfur),
            "blend_ash_wt_pct": str(blend_ash),
            "blend_water_vol_pct": str(blend_water)
        })

        # Step 6: Calculate viscosity (NON-LINEAR - Refutas method)
        blend_viscosity = self._refutas_blend_viscosity(
            [(comp.viscosity_50c_cst, frac) for comp, frac in zip(components, fractions)]
        )

        steps.append({
            "step": 6,
            "operation": "refutas_blend_viscosity",
            "method": "ASTM D341",
            "blend_viscosity_50c_cst": str(blend_viscosity)
        })

        # Step 7: Calculate flash point (NON-LINEAR - conservative)
        blend_flash = self._blend_flash_point(
            [(comp.flash_point_c, frac) for comp, frac in zip(components, fractions)]
        )

        # Step 8: Calculate vapor pressure (Raoult's Law approximation)
        blend_vapor = self._linear_blend(
            [(comp.vapor_pressure_kpa, frac) for comp, frac in zip(components, fractions)]
        )

        steps.append({
            "step": 8,
            "operation": "blend_safety_properties",
            "blend_flash_point_c": str(blend_flash),
            "blend_vapor_pressure_kpa": str(blend_vapor)
        })

        # Step 9: Calculate energy-weighted carbon intensity
        blend_ci = self._energy_weighted_carbon_intensity(components, fractions)
        total_emissions = total_energy * blend_ci

        steps.append({
            "step": 9,
            "operation": "energy_weighted_carbon_intensity",
            "blend_ci_kg_co2e_mj": str(blend_ci),
            "total_emissions_kg_co2e": str(total_emissions)
        })

        # Step 10: Validate constraints
        quality_valid, quality_violations = self._validate_quality(
            blend_sulfur, blend_ash, blend_water, blend_viscosity,
            blend_input.quality_constraints or self._quality_constraints
        )
        safety_valid, safety_violations = self._validate_safety(
            blend_flash, blend_vapor,
            blend_input.safety_constraints or self._safety_constraints
        )

        steps.append({
            "step": 10,
            "operation": "validate_constraints",
            "quality_valid": quality_valid,
            "safety_valid": safety_valid
        })

        # Apply precision
        quantize_str = "0." + "0" * precision

        return BlendResult(
            total_mass_kg=total_mass.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            total_energy_mj=total_energy.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_lhv_mj_kg=blend_lhv.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_hhv_mj_kg=blend_hhv.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_density_kg_m3=blend_density.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_sulfur_wt_pct=blend_sulfur.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_ash_wt_pct=blend_ash.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_water_vol_pct=blend_water.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_viscosity_50c_cst=blend_viscosity.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_flash_point_c=blend_flash.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_vapor_pressure_kpa=blend_vapor.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            blend_carbon_intensity=blend_ci.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            total_emissions_kg_co2e=total_emissions.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            quality_valid=quality_valid,
            quality_violations=quality_violations,
            safety_valid=safety_valid,
            safety_violations=safety_violations,
            component_contributions=contributions,
            calculation_steps=steps
        )

    def _linear_blend(
        self,
        values_fractions: List[Tuple[Decimal, Decimal]]
    ) -> Decimal:
        """
        Calculate linear mass-weighted blend.

        Args:
            values_fractions: List of (value, fraction) tuples

        Returns:
            Blended value
        """
        result = Decimal("0")
        for value, fraction in values_fractions:
            result += value * fraction
        return result

    def _refutas_blend_viscosity(
        self,
        viscosities_fractions: List[Tuple[Decimal, Decimal]]
    ) -> Decimal:
        """
        Calculate blended viscosity using Refutas method (ASTM D341).

        VBI = 14.534 * ln(ln(viscosity + 0.8)) + 10.975
        blend_VBI = sum(fraction_i * VBI_i)
        blend_viscosity = exp(exp((blend_VBI - 10.975) / 14.534)) - 0.8

        Args:
            viscosities_fractions: List of (viscosity_cSt, fraction) tuples

        Returns:
            Blended viscosity (cSt)
        """
        # Calculate Viscosity Blending Index for each component
        vbi_total = Decimal("0")

        for visc, frac in viscosities_fractions:
            visc_float = float(visc)
            if visc_float <= 0:
                raise ValueError("Viscosity must be positive")

            # VBI = 14.534 * ln(ln(v + 0.8)) + 10.975
            inner = math.log(visc_float + 0.8)
            if inner <= 0:
                raise ValueError(f"Invalid viscosity {visc} for Refutas calculation")

            vbi = 14.534 * math.log(inner) + 10.975
            vbi_total += Decimal(str(vbi)) * frac

        # Convert back from VBI to viscosity
        vbi_float = float(vbi_total)
        exp_arg = (vbi_float - 10.975) / 14.534
        blend_visc = math.exp(math.exp(exp_arg)) - 0.8

        return Decimal(str(blend_visc))

    def _blend_flash_point(
        self,
        flash_points_fractions: List[Tuple[Decimal, Decimal]]
    ) -> Decimal:
        """
        Calculate blended flash point - CONSERVATIVE estimate.

        Uses minimum flash point approach for safety.
        More sophisticated methods use Thomas correlation.

        Args:
            flash_points_fractions: List of (flash_point_C, fraction) tuples

        Returns:
            Estimated blend flash point (C)
        """
        # Conservative: use minimum flash point of significant components
        # (components with >5% fraction)
        significant_flashes = [
            fp for fp, frac in flash_points_fractions
            if frac > Decimal("0.05")
        ]

        if not significant_flashes:
            # All small fractions - use weighted average
            return self._linear_blend(flash_points_fractions)

        # Return minimum for safety
        return min(significant_flashes)

    def _energy_weighted_carbon_intensity(
        self,
        components: List[BlendComponent],
        fractions: List[Decimal]
    ) -> Decimal:
        """
        Calculate energy-weighted blend carbon intensity.

        CI_blend = sum(E_i * CI_i) / sum(E_i)
        where E_i is energy contribution of component i

        Args:
            components: Blend components
            fractions: Mass fractions

        Returns:
            Blended carbon intensity (kgCO2e/MJ)
        """
        total_weighted_ci = Decimal("0")
        total_energy = Decimal("0")

        for comp, frac in zip(components, fractions):
            energy_i = comp.mass_kg * frac * comp.lhv_mj_kg
            total_weighted_ci += energy_i * comp.carbon_intensity_kg_co2e_mj
            total_energy += energy_i

        if total_energy == Decimal("0"):
            return Decimal("0")

        return total_weighted_ci / total_energy

    def _validate_quality(
        self,
        sulfur: Decimal,
        ash: Decimal,
        water: Decimal,
        viscosity: Decimal,
        constraints: List[QualityConstraint]
    ) -> Tuple[bool, List[str]]:
        """Validate quality constraints."""
        violations = []
        property_map = {
            "sulfur_content": sulfur,
            "ash_content": ash,
            "water_content": water,
            "viscosity_50c": viscosity
        }

        for constraint in constraints:
            if constraint.property_name in property_map:
                value = property_map[constraint.property_name]
                valid, msg = constraint.validate(value)
                if not valid:
                    violations.append(msg)

        return len(violations) == 0, violations

    def _validate_safety(
        self,
        flash_point: Decimal,
        vapor_pressure: Decimal,
        constraints: List[SafetyConstraint]
    ) -> Tuple[bool, List[str]]:
        """Validate safety constraints."""
        violations = []
        property_map = {
            "flash_point": flash_point,
            "vapor_pressure": vapor_pressure
        }

        for constraint in constraints:
            if constraint.property_name in property_map:
                value = property_map[constraint.property_name]
                valid, msg = constraint.validate(value)
                if not valid:
                    violations.append(msg)

        return len(violations) == 0, violations
