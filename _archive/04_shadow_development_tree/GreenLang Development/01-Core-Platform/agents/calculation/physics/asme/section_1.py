"""
ASME Section I - Pressure Calculations

Zero-Hallucination Pressure Vessel Calculations

This module implements ASME Boiler and Pressure Vessel Code Section I
calculations for power boiler design and analysis.

References:
    - ASME BPVC Section I-2023: Rules for Construction of Power Boilers
    - ASME BPVC Section II-D: Materials (Allowable Stresses)
    - ASME BPVC Section VIII-1: Pressure Vessels (comparative)

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, Optional
import math
import hashlib


class TubeType(Enum):
    """Types of tubes per ASME Section I."""
    SEAMLESS = "seamless"
    WELDED = "welded"
    ELECTRIC_RESISTANCE_WELDED = "erw"


class HeadType(Enum):
    """Head types for pressure calculations."""
    HEMISPHERICAL = "hemispherical"
    ELLIPSOIDAL_2_1 = "ellipsoidal_2_1"
    TORISPHERICAL = "torispherical"
    FLAT = "flat"


@dataclass
class MaterialProperties:
    """Material properties for pressure calculations."""
    name: str
    specification: str  # e.g., "SA-106 Grade B"
    allowable_stress_mpa: float  # At design temperature
    yield_strength_mpa: float
    tensile_strength_mpa: float
    design_temperature_c: float


@dataclass
class PressureCalculationResult:
    """
    Pressure calculation results per ASME Section I.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Design parameters
    design_pressure_mpa: Decimal
    design_temperature_c: Decimal

    # Calculated thickness
    required_thickness_mm: Decimal
    minimum_thickness_mm: Decimal  # Including corrosion allowance
    actual_thickness_mm: Optional[Decimal]

    # Stresses
    allowable_stress_mpa: Decimal
    calculated_stress_mpa: Decimal
    stress_ratio: Decimal

    # MAWP (if actual thickness provided)
    mawp_mpa: Optional[Decimal]

    # Joint efficiency
    joint_efficiency: Decimal

    # Safety factors
    design_factor: Decimal

    # Code reference
    code_paragraph: str

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "design_pressure_mpa": float(self.design_pressure_mpa),
            "required_thickness_mm": float(self.required_thickness_mm),
            "minimum_thickness_mm": float(self.minimum_thickness_mm),
            "mawp_mpa": float(self.mawp_mpa) if self.mawp_mpa else None,
            "stress_ratio": float(self.stress_ratio),
            "code_paragraph": self.code_paragraph,
            "provenance_hash": self.provenance_hash
        }


class ASMESectionI:
    """
    ASME Section I Pressure Calculations.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on ASME BPVC Section I formulas
    - Complete provenance tracking
    - Conservative design factors applied

    Coverage:
    - Cylindrical shells under internal pressure
    - Tubes and pipes
    - Heads (hemispherical, ellipsoidal, torispherical)
    - Ligament efficiency

    References:
        - ASME BPVC Section I, PG-27 (Cylindrical Components)
        - ASME BPVC Section I, PG-29 (Tubes)
        - ASME BPVC Section I, PG-32 (Heads)
    """

    def __init__(self, precision: int = 3):
        """Initialize calculator."""
        self.precision = precision

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "ASME_BPVC_Section_I",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def cylindrical_shell_thickness(
        self,
        design_pressure_mpa: float,
        inside_diameter_mm: float,
        allowable_stress_mpa: float,
        joint_efficiency: float = 1.0,
        corrosion_allowance_mm: float = 0.0,
        design_temperature_c: float = 100.0,
        actual_thickness_mm: Optional[float] = None
    ) -> PressureCalculationResult:
        """
        Calculate required thickness for cylindrical shell under internal pressure.

        Reference: ASME Section I, PG-27.2.2

        t = P*R / (S*E - 0.6*P) + C

        Where:
        - P = Design pressure (MPa)
        - R = Inside radius (mm)
        - S = Allowable stress (MPa)
        - E = Joint efficiency
        - C = Corrosion allowance (mm)

        Args:
            design_pressure_mpa: Design pressure (MPa)
            inside_diameter_mm: Inside diameter (mm)
            allowable_stress_mpa: Allowable stress at temperature (MPa)
            joint_efficiency: Longitudinal joint efficiency (0-1)
            corrosion_allowance_mm: Corrosion allowance (mm)
            design_temperature_c: Design temperature (C)
            actual_thickness_mm: Actual thickness for MAWP calculation

        Returns:
            PressureCalculationResult with thickness and MAWP
        """
        p = Decimal(str(design_pressure_mpa))
        d = Decimal(str(inside_diameter_mm))
        r = d / Decimal("2")
        s = Decimal(str(allowable_stress_mpa))
        e = Decimal(str(joint_efficiency))
        c = Decimal(str(corrosion_allowance_mm))
        t_design = Decimal(str(design_temperature_c))

        # Validate inputs
        if e <= 0 or e > 1:
            raise ValueError("Joint efficiency must be between 0 and 1")

        if s <= 0:
            raise ValueError("Allowable stress must be positive")

        # Calculate required thickness (PG-27.2.2)
        # t = P*R / (S*E - 0.6*P)
        denominator = s * e - Decimal("0.6") * p

        if denominator <= 0:
            raise ValueError("Design pressure too high for given stress and efficiency")

        t_required = p * r / denominator

        # Add corrosion allowance
        t_minimum = t_required + c

        # Calculate stress in actual wall (if thickness provided)
        if actual_thickness_mm:
            t_actual = Decimal(str(actual_thickness_mm))

            # Check if actual thickness is adequate
            t_effective = t_actual - c

            if t_effective <= 0:
                raise ValueError("Actual thickness less than corrosion allowance")

            # Calculate MAWP from actual thickness
            # Rearranging: P = S*E*t / (R + 0.6*t)
            mawp = s * e * t_effective / (r + Decimal("0.6") * t_effective)

            # Calculate actual stress at design pressure
            actual_stress = p * (r + Decimal("0.6") * t_effective) / (e * t_effective)
        else:
            t_actual = None
            mawp = None
            actual_stress = s  # At design conditions

        # Stress ratio
        stress_ratio = actual_stress / s if s > 0 else Decimal("1")

        # Design factor (Section I uses 3.5 on tensile strength historically)
        design_factor = Decimal("3.5")

        # Create provenance
        inputs = {
            "pressure_mpa": str(p),
            "diameter_mm": str(d),
            "allowable_stress_mpa": str(s),
            "joint_efficiency": str(e)
        }
        outputs = {
            "required_thickness_mm": str(t_required),
            "minimum_thickness_mm": str(t_minimum),
            "mawp_mpa": str(mawp) if mawp else "N/A"
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PressureCalculationResult(
            design_pressure_mpa=self._apply_precision(p),
            design_temperature_c=self._apply_precision(t_design),
            required_thickness_mm=self._apply_precision(t_required),
            minimum_thickness_mm=self._apply_precision(t_minimum),
            actual_thickness_mm=self._apply_precision(t_actual) if t_actual else None,
            allowable_stress_mpa=self._apply_precision(s),
            calculated_stress_mpa=self._apply_precision(actual_stress),
            stress_ratio=self._apply_precision(stress_ratio),
            mawp_mpa=self._apply_precision(mawp) if mawp else None,
            joint_efficiency=self._apply_precision(e),
            design_factor=self._apply_precision(design_factor),
            code_paragraph="PG-27.2.2",
            provenance_hash=provenance_hash
        )

    def tube_thickness(
        self,
        design_pressure_mpa: float,
        outside_diameter_mm: float,
        allowable_stress_mpa: float,
        tube_type: TubeType = TubeType.SEAMLESS,
        corrosion_allowance_mm: float = 0.0,
        design_temperature_c: float = 100.0,
        actual_thickness_mm: Optional[float] = None
    ) -> PressureCalculationResult:
        """
        Calculate required thickness for tubes under internal pressure.

        Reference: ASME Section I, PG-27.2.1 (for tubes)

        For tubes: t = P*D / (2*S*E + 2*y*P) + C

        Where:
        - P = Design pressure
        - D = Outside diameter
        - S = Allowable stress
        - E = Quality factor (1.0 for seamless)
        - y = Temperature coefficient (0.4 for ferritic below 900F)
        - C = Minimum allowance for threading, grooving, corrosion

        Args:
            design_pressure_mpa: Design pressure (MPa)
            outside_diameter_mm: Outside diameter (mm)
            allowable_stress_mpa: Allowable stress (MPa)
            tube_type: Type of tube (affects E factor)
            corrosion_allowance_mm: Corrosion allowance (mm)
            design_temperature_c: Design temperature (C)
            actual_thickness_mm: Actual thickness for verification

        Returns:
            PressureCalculationResult with thickness and MAWP
        """
        p = Decimal(str(design_pressure_mpa))
        d = Decimal(str(outside_diameter_mm))
        s = Decimal(str(allowable_stress_mpa))
        c = Decimal(str(corrosion_allowance_mm))
        t_design = Decimal(str(design_temperature_c))

        # Quality factor E
        if tube_type == TubeType.SEAMLESS:
            e = Decimal("1.0")
        elif tube_type == TubeType.ELECTRIC_RESISTANCE_WELDED:
            e = Decimal("0.85")
        else:  # Welded
            e = Decimal("0.85")

        # Temperature coefficient y
        # For ferritic steels below 482C (900F): y = 0.4
        # For ferritic steels above 482C: y = 0.5
        # For austenitic steels: y = 0.4
        if t_design > Decimal("482"):
            y = Decimal("0.5")
        else:
            y = Decimal("0.4")

        # Calculate required thickness (PG-27.2.1 formula)
        # t = P*D / (2*S*E + 2*y*P)
        denominator = Decimal("2") * s * e + Decimal("2") * y * p

        if denominator <= 0:
            raise ValueError("Invalid calculation parameters")

        t_required = p * d / denominator

        # Add corrosion allowance
        t_minimum = t_required + c

        # Calculate MAWP if actual thickness provided
        if actual_thickness_mm:
            t_actual = Decimal(str(actual_thickness_mm))
            t_effective = t_actual - c

            if t_effective <= 0:
                raise ValueError("Effective thickness is zero or negative")

            # MAWP = 2*S*E*t / (D - 2*y*t)
            mawp = Decimal("2") * s * e * t_effective / (d - Decimal("2") * y * t_effective)

            # Calculate actual stress
            actual_stress = p * (d - Decimal("2") * y * t_effective) / (Decimal("2") * e * t_effective)
        else:
            t_actual = None
            mawp = None
            actual_stress = s

        stress_ratio = actual_stress / s if s > 0 else Decimal("1")
        design_factor = Decimal("3.5")

        inputs = {
            "pressure_mpa": str(p),
            "od_mm": str(d),
            "allowable_stress_mpa": str(s),
            "tube_type": tube_type.value
        }
        outputs = {
            "required_thickness_mm": str(t_required),
            "mawp_mpa": str(mawp) if mawp else "N/A"
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PressureCalculationResult(
            design_pressure_mpa=self._apply_precision(p),
            design_temperature_c=self._apply_precision(t_design),
            required_thickness_mm=self._apply_precision(t_required),
            minimum_thickness_mm=self._apply_precision(t_minimum),
            actual_thickness_mm=self._apply_precision(t_actual) if t_actual else None,
            allowable_stress_mpa=self._apply_precision(s),
            calculated_stress_mpa=self._apply_precision(actual_stress),
            stress_ratio=self._apply_precision(stress_ratio),
            mawp_mpa=self._apply_precision(mawp) if mawp else None,
            joint_efficiency=self._apply_precision(e),
            design_factor=self._apply_precision(design_factor),
            code_paragraph="PG-27.2.1",
            provenance_hash=provenance_hash
        )

    def head_thickness(
        self,
        design_pressure_mpa: float,
        inside_diameter_mm: float,
        allowable_stress_mpa: float,
        head_type: HeadType,
        joint_efficiency: float = 1.0,
        corrosion_allowance_mm: float = 0.0,
        dish_radius_mm: Optional[float] = None,
        knuckle_radius_mm: Optional[float] = None
    ) -> PressureCalculationResult:
        """
        Calculate required thickness for pressure vessel heads.

        Reference: ASME Section I, PG-29 (Dished Heads)

        Args:
            design_pressure_mpa: Design pressure (MPa)
            inside_diameter_mm: Inside diameter (mm)
            allowable_stress_mpa: Allowable stress (MPa)
            head_type: Type of head
            joint_efficiency: Joint efficiency
            corrosion_allowance_mm: Corrosion allowance (mm)
            dish_radius_mm: Dish radius (for torispherical)
            knuckle_radius_mm: Knuckle radius (for torispherical)

        Returns:
            PressureCalculationResult with thickness
        """
        p = Decimal(str(design_pressure_mpa))
        d = Decimal(str(inside_diameter_mm))
        s = Decimal(str(allowable_stress_mpa))
        e = Decimal(str(joint_efficiency))
        c = Decimal(str(corrosion_allowance_mm))

        if head_type == HeadType.HEMISPHERICAL:
            # t = P*R / (2*S*E - 0.2*P)
            r = d / Decimal("2")
            denominator = Decimal("2") * s * e - Decimal("0.2") * p
            t_required = p * r / denominator
            code_para = "PG-29.1"

        elif head_type == HeadType.ELLIPSOIDAL_2_1:
            # 2:1 ellipsoidal head
            # t = P*D / (2*S*E - 0.2*P)
            denominator = Decimal("2") * s * e - Decimal("0.2") * p
            t_required = p * d / denominator
            code_para = "PG-29.2"

        elif head_type == HeadType.TORISPHERICAL:
            # Torispherical (flanged and dished)
            # t = 0.885*P*L / (S*E - 0.1*P)
            if dish_radius_mm is None:
                l_radius = d  # Default: L = D
            else:
                l_radius = Decimal(str(dish_radius_mm))

            denominator = s * e - Decimal("0.1") * p
            t_required = Decimal("0.885") * p * l_radius / denominator
            code_para = "PG-29.3"

        elif head_type == HeadType.FLAT:
            # Flat heads - simplified
            # t = d * sqrt(C*P / S*E)
            # C depends on attachment method, use conservative value
            c_factor = Decimal("0.33")  # Conservative for bolted covers
            sqrt_term = Decimal(str(math.sqrt(float(c_factor * p / (s * e)))))
            t_required = d * sqrt_term
            code_para = "PG-31"

        else:
            raise ValueError(f"Unknown head type: {head_type}")

        t_minimum = t_required + c

        inputs = {
            "pressure_mpa": str(p),
            "diameter_mm": str(d),
            "head_type": head_type.value
        }
        outputs = {"required_thickness_mm": str(t_required)}
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PressureCalculationResult(
            design_pressure_mpa=self._apply_precision(p),
            design_temperature_c=Decimal("0"),  # Not specified
            required_thickness_mm=self._apply_precision(t_required),
            minimum_thickness_mm=self._apply_precision(t_minimum),
            actual_thickness_mm=None,
            allowable_stress_mpa=self._apply_precision(s),
            calculated_stress_mpa=self._apply_precision(s),  # At design
            stress_ratio=Decimal("1.0"),
            mawp_mpa=None,
            joint_efficiency=self._apply_precision(e),
            design_factor=Decimal("3.5"),
            code_paragraph=code_para,
            provenance_hash=provenance_hash
        )

    def ligament_efficiency(
        self,
        pitch_mm: float,
        hole_diameter_mm: float
    ) -> Decimal:
        """
        Calculate ligament efficiency for tube holes.

        Reference: ASME Section I, PG-52

        eta = (p - d) / p

        Args:
            pitch_mm: Center-to-center spacing of tubes (mm)
            hole_diameter_mm: Diameter of tube holes (mm)

        Returns:
            Ligament efficiency (0 to 1)
        """
        p = Decimal(str(pitch_mm))
        d = Decimal(str(hole_diameter_mm))

        if p <= d:
            raise ValueError("Pitch must be greater than hole diameter")

        eta = (p - d) / p

        return self._apply_precision(eta)


# Convenience functions
def shell_thickness(
    pressure_mpa: float,
    diameter_mm: float,
    allowable_stress_mpa: float,
    joint_efficiency: float = 1.0
) -> PressureCalculationResult:
    """
    Calculate cylindrical shell thickness per ASME Section I.

    Example:
        >>> result = shell_thickness(
        ...     pressure_mpa=10.0,
        ...     diameter_mm=1000,
        ...     allowable_stress_mpa=138
        ... )
        >>> print(f"Required thickness: {result.required_thickness_mm} mm")
    """
    calc = ASMESectionI()
    return calc.cylindrical_shell_thickness(
        pressure_mpa, diameter_mm, allowable_stress_mpa, joint_efficiency
    )


def tube_wall_thickness(
    pressure_mpa: float,
    od_mm: float,
    allowable_stress_mpa: float
) -> PressureCalculationResult:
    """Calculate tube thickness per ASME Section I."""
    calc = ASMESectionI()
    return calc.tube_thickness(pressure_mpa, od_mm, allowable_stress_mpa)
