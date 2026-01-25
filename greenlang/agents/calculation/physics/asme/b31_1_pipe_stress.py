"""
ASME B31.1 - Power Piping Stress Calculations

Zero-Hallucination Pipe Stress Analysis

This module implements ASME B31.1 "Power Piping" stress calculations
for design and analysis of power piping systems including:
- Internal pressure stresses
- Sustained load stresses (pressure + weight)
- Thermal expansion stress range
- Occasional load stresses
- Code compliance verification

References:
    - ASME B31.1-2022: Power Piping
    - ASME B31.1, Chapter II: Design
    - ASME B31.1, Paragraph 104: Pressure Design
    - ASME B31.1, Paragraph 119: Flexibility and Stress Intensification

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import hashlib


class PipeMaterial(Enum):
    """Common piping materials per ASME B31.1."""
    CARBON_STEEL_A106_B = "carbon_steel_a106_b"  # SA-106 Grade B
    CARBON_STEEL_A53_B = "carbon_steel_a53_b"  # SA-53 Grade B
    LOW_ALLOY_P11 = "low_alloy_p11"  # 1.25Cr-0.5Mo (P11)
    LOW_ALLOY_P22 = "low_alloy_p22"  # 2.25Cr-1Mo (P22)
    LOW_ALLOY_P91 = "low_alloy_p91"  # 9Cr-1Mo-V (P91)
    SS_304 = "ss_304"  # Type 304 SS
    SS_304H = "ss_304h"  # Type 304H SS
    SS_316 = "ss_316"  # Type 316 SS
    SS_321 = "ss_321"  # Type 321 SS
    SS_347 = "ss_347"  # Type 347 SS


class PipeSchedule(Enum):
    """Common pipe schedules."""
    SCH_10 = "sch_10"
    SCH_20 = "sch_20"
    SCH_30 = "sch_30"
    SCH_40 = "sch_40"
    SCH_60 = "sch_60"
    SCH_80 = "sch_80"
    SCH_100 = "sch_100"
    SCH_120 = "sch_120"
    SCH_140 = "sch_140"
    SCH_160 = "sch_160"
    SCH_STD = "sch_std"
    SCH_XS = "sch_xs"
    SCH_XXS = "sch_xxs"


class LoadCategory(Enum):
    """Load categories per ASME B31.1."""
    SUSTAINED = "sustained"
    EXPANSION = "expansion"
    OCCASIONAL = "occasional"


@dataclass
class PipeGeometry:
    """Pipe geometry parameters."""
    outside_diameter_mm: float
    wall_thickness_mm: float
    mill_tolerance_percent: float = 12.5  # Typical -12.5% mill tolerance
    corrosion_allowance_mm: float = 0.0

    @property
    def inside_diameter_mm(self) -> float:
        """Calculate inside diameter."""
        return self.outside_diameter_mm - 2 * self.wall_thickness_mm

    @property
    def mean_diameter_mm(self) -> float:
        """Calculate mean diameter."""
        return self.outside_diameter_mm - self.wall_thickness_mm

    @property
    def minimum_wall_thickness_mm(self) -> float:
        """Calculate minimum wall thickness after mill tolerance."""
        return self.wall_thickness_mm * (1 - self.mill_tolerance_percent / 100)

    @property
    def effective_wall_thickness_mm(self) -> float:
        """Calculate effective wall thickness (min - corrosion)."""
        return self.minimum_wall_thickness_mm - self.corrosion_allowance_mm

    @property
    def metal_area_mm2(self) -> float:
        """Calculate metal cross-sectional area."""
        return math.pi / 4 * (
            self.outside_diameter_mm ** 2 - self.inside_diameter_mm ** 2
        )

    @property
    def section_modulus_mm3(self) -> float:
        """Calculate section modulus for bending."""
        d_o = self.outside_diameter_mm
        d_i = self.inside_diameter_mm
        return math.pi / 32 * (d_o ** 4 - d_i ** 4) / d_o


@dataclass
class LoadData:
    """Applied load data for stress calculations."""
    internal_pressure_mpa: float = 0.0
    axial_force_n: float = 0.0  # Axial force from sustained loads
    bending_moment_nm: float = 0.0  # In-plane + out-plane combined
    torsional_moment_nm: float = 0.0  # Torsional moment
    thermal_bending_moment_nm: float = 0.0  # From thermal expansion
    occasional_force_n: float = 0.0  # Earthquake, wind, etc.
    occasional_moment_nm: float = 0.0


@dataclass
class B311StressResult:
    """
    ASME B31.1 pipe stress calculation results.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Geometry
    pipe_od_mm: Decimal
    wall_thickness_mm: Decimal
    effective_thickness_mm: Decimal

    # Primary stresses
    hoop_stress_mpa: Decimal  # Circumferential stress from pressure
    longitudinal_stress_mpa: Decimal  # Combined longitudinal stress
    radial_stress_mpa: Decimal  # Radial stress (typically -P at inner wall)

    # Code stresses
    sustained_stress_mpa: Decimal  # S_L per B31.1 Eq. 11A
    expansion_stress_range_mpa: Decimal  # S_E per B31.1 Eq. 13
    occasional_stress_mpa: Decimal  # For occasional loads

    # Allowable stresses
    allowable_stress_hot_mpa: Decimal  # S_h at operating temperature
    allowable_stress_cold_mpa: Decimal  # S_c at ambient
    allowable_expansion_range_mpa: Decimal  # S_A per B31.1 Eq. 1
    allowable_sustained_mpa: Decimal  # Usually S_h
    allowable_occasional_mpa: Decimal  # 1.15*S_h or 1.2*S_h

    # Stress ratios (actual / allowable)
    sustained_stress_ratio: Decimal
    expansion_stress_ratio: Decimal
    occasional_stress_ratio: Decimal

    # Compliance flags
    sustained_acceptable: bool
    expansion_acceptable: bool
    occasional_acceptable: bool
    overall_acceptable: bool

    # Stress reduction factor
    stress_reduction_factor_f: Decimal  # f factor for expansion

    # Code references
    code_paragraph: str
    code_edition: str

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "pipe_od_mm": float(self.pipe_od_mm),
            "wall_thickness_mm": float(self.wall_thickness_mm),
            "hoop_stress_mpa": float(self.hoop_stress_mpa),
            "sustained_stress_mpa": float(self.sustained_stress_mpa),
            "expansion_stress_range_mpa": float(self.expansion_stress_range_mpa),
            "sustained_stress_ratio": float(self.sustained_stress_ratio),
            "expansion_stress_ratio": float(self.expansion_stress_ratio),
            "overall_acceptable": self.overall_acceptable,
            "provenance_hash": self.provenance_hash
        }


@dataclass
class MinimumThicknessResult:
    """Result from minimum wall thickness calculation."""
    required_thickness_mm: Decimal
    pressure_design_thickness_mm: Decimal
    mill_tolerance_mm: Decimal
    corrosion_allowance_mm: Decimal
    total_minimum_thickness_mm: Decimal
    nearest_schedule: Optional[str]
    code_paragraph: str
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "required_thickness_mm": float(self.required_thickness_mm),
            "total_minimum_thickness_mm": float(self.total_minimum_thickness_mm),
            "nearest_schedule": self.nearest_schedule,
            "code_paragraph": self.code_paragraph,
            "provenance_hash": self.provenance_hash
        }


class ASMEB311PipeStress:
    """
    ASME B31.1 Power Piping Stress Calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on ASME B31.1-2022 formulas
    - Complete provenance tracking
    - Conservative design approach

    Calculation Methods:
    1. Hoop stress from internal pressure (Para. 104.1.2)
    2. Sustained stress (Para. 104.8.1)
    3. Thermal expansion stress range (Para. 104.8.3)
    4. Occasional stress (Para. 104.8.2)

    Key Code Equations:
    - Eq. 3: Pressure design thickness
    - Eq. 11A: Sustained stress
    - Eq. 12: Expansion stress components
    - Eq. 13: Expansion stress range

    References:
        - ASME B31.1-2022, Chapter II (Design)
        - ASME B31.1-2022, Para. 102.3 (Allowable Stresses)
        - ASME B31.1-2022, Para. 104 (Pressure Design)
    """

    # Allowable stress tables for common materials (MPa at temperature in C)
    # Reference: ASME B31.1, Mandatory Appendix A
    ALLOWABLE_STRESS = {
        PipeMaterial.CARBON_STEEL_A106_B: {
            38: Decimal("117.9"),   # 100F
            93: Decimal("117.9"),   # 200F
            149: Decimal("117.9"),  # 300F
            204: Decimal("117.9"),  # 400F
            260: Decimal("117.9"),  # 500F
            316: Decimal("113.8"),  # 600F
            343: Decimal("106.2"),  # 650F
            371: Decimal("93.1"),   # 700F
            399: Decimal("75.2"),   # 750F
            427: Decimal("55.2"),   # 800F
            454: Decimal("37.2"),   # 850F
        },
        PipeMaterial.LOW_ALLOY_P22: {
            38: Decimal("117.9"),
            93: Decimal("117.9"),
            149: Decimal("117.9"),
            204: Decimal("117.9"),
            260: Decimal("117.9"),
            316: Decimal("117.9"),
            343: Decimal("117.9"),
            371: Decimal("117.9"),
            399: Decimal("115.1"),
            427: Decimal("110.3"),
            454: Decimal("103.4"),
            482: Decimal("93.1"),
            510: Decimal("76.5"),
            538: Decimal("57.9"),
            566: Decimal("40.7"),
            593: Decimal("27.6"),
        },
        PipeMaterial.LOW_ALLOY_P91: {
            38: Decimal("124.1"),
            149: Decimal("124.1"),
            260: Decimal("124.1"),
            316: Decimal("124.1"),
            371: Decimal("124.1"),
            399: Decimal("124.1"),
            427: Decimal("124.1"),
            454: Decimal("122.0"),
            482: Decimal("118.6"),
            510: Decimal("113.1"),
            538: Decimal("104.1"),
            566: Decimal("89.6"),
            593: Decimal("71.0"),
            621: Decimal("51.7"),
        },
        PipeMaterial.SS_304H: {
            38: Decimal("115.1"),
            149: Decimal("110.3"),
            260: Decimal("101.4"),
            316: Decimal("97.2"),
            371: Decimal("93.1"),
            427: Decimal("89.6"),
            482: Decimal("86.9"),
            538: Decimal("84.8"),
            593: Decimal("83.4"),
            649: Decimal("81.4"),
            704: Decimal("77.2"),
            760: Decimal("57.2"),
            816: Decimal("35.2"),
        },
    }

    # Modulus of elasticity (MPa) at temperature
    # Reference: ASME B31.1, Mandatory Appendix A
    ELASTIC_MODULUS = {
        PipeMaterial.CARBON_STEEL_A106_B: {
            21: Decimal("203000"),   # 70F
            93: Decimal("199000"),   # 200F
            149: Decimal("195000"),  # 300F
            204: Decimal("192000"),  # 400F
            260: Decimal("187000"),  # 500F
            316: Decimal("183000"),  # 600F
            371: Decimal("178000"),  # 700F
            427: Decimal("173000"),  # 800F
        },
        PipeMaterial.LOW_ALLOY_P22: {
            21: Decimal("207000"),
            149: Decimal("199000"),
            260: Decimal("193000"),
            371: Decimal("185000"),
            482: Decimal("176000"),
            538: Decimal("170000"),
            593: Decimal("163000"),
        },
        PipeMaterial.LOW_ALLOY_P91: {
            21: Decimal("207000"),
            149: Decimal("199000"),
            260: Decimal("193000"),
            371: Decimal("185000"),
            482: Decimal("176000"),
            538: Decimal("170000"),
            593: Decimal("163000"),
        },
        PipeMaterial.SS_304H: {
            21: Decimal("195000"),
            149: Decimal("189000"),
            260: Decimal("183000"),
            371: Decimal("177000"),
            482: Decimal("170000"),
            538: Decimal("165000"),
            593: Decimal("161000"),
            649: Decimal("156000"),
            704: Decimal("151000"),
        },
    }

    # Mean coefficient of thermal expansion (mm/mm/C x 10^-6)
    # Reference: ASME B31.1, Mandatory Appendix A
    THERMAL_EXPANSION_COEFF = {
        PipeMaterial.CARBON_STEEL_A106_B: Decimal("12.1"),  # Average
        PipeMaterial.LOW_ALLOY_P22: Decimal("12.6"),
        PipeMaterial.LOW_ALLOY_P91: Decimal("11.2"),
        PipeMaterial.SS_304H: Decimal("17.3"),
    }

    def __init__(self, precision: int = 2):
        """
        Initialize ASME B31.1 calculator.

        Args:
            precision: Decimal places for output (default 2)
        """
        self.precision = precision
        self._code_edition = "ASME B31.1-2022"

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding to decimal value."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_data = {
            "method": "ASME_B31.1_Power_Piping",
            "code_edition": self._code_edition,
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _interpolate_property(
        self,
        property_table: Dict[int, Decimal],
        temperature_c: float
    ) -> Decimal:
        """Interpolate material property from temperature table."""
        t = float(temperature_c)
        temps = sorted(property_table.keys())

        # Check bounds - use boundary value if outside range
        if t <= temps[0]:
            return property_table[temps[0]]
        if t >= temps[-1]:
            return property_table[temps[-1]]

        # Find bracketing temperatures for interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= t <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                v1, v2 = property_table[t1], property_table[t2]

                # Linear interpolation
                fraction = Decimal(str((t - t1) / (t2 - t1)))
                return v1 + fraction * (v2 - v1)

        return property_table[temps[-1]]

    def get_allowable_stress(
        self,
        material: PipeMaterial,
        temperature_c: float
    ) -> Decimal:
        """
        Get allowable stress at temperature per ASME B31.1.

        Reference: ASME B31.1, Mandatory Appendix A

        Args:
            material: Pipe material
            temperature_c: Design temperature (Celsius)

        Returns:
            Allowable stress in MPa
        """
        if material in self.ALLOWABLE_STRESS:
            stress = self._interpolate_property(
                self.ALLOWABLE_STRESS[material], temperature_c
            )
        else:
            # Default to carbon steel if material not in table
            stress = self._interpolate_property(
                self.ALLOWABLE_STRESS[PipeMaterial.CARBON_STEEL_A106_B],
                temperature_c
            )
        return self._apply_precision(stress)

    def get_elastic_modulus(
        self,
        material: PipeMaterial,
        temperature_c: float
    ) -> Decimal:
        """
        Get modulus of elasticity at temperature.

        Reference: ASME B31.1, Mandatory Appendix A

        Args:
            material: Pipe material
            temperature_c: Temperature (Celsius)

        Returns:
            Elastic modulus in MPa
        """
        if material in self.ELASTIC_MODULUS:
            modulus = self._interpolate_property(
                self.ELASTIC_MODULUS[material], temperature_c
            )
        else:
            # Default to carbon steel
            modulus = self._interpolate_property(
                self.ELASTIC_MODULUS[PipeMaterial.CARBON_STEEL_A106_B],
                temperature_c
            )
        return self._apply_precision(modulus)

    def calculate_hoop_stress(
        self,
        internal_pressure_mpa: float,
        outside_diameter_mm: float,
        wall_thickness_mm: float
    ) -> Decimal:
        """
        Calculate hoop stress from internal pressure.

        ZERO-HALLUCINATION: Deterministic Barlow formula.

        Reference: ASME B31.1, Para. 104.1.2

        Formula:
            S_h = P * D_o / (2 * t_m)

        Where:
            P = Internal pressure (MPa)
            D_o = Outside diameter (mm)
            t_m = Minimum wall thickness (mm)

        Args:
            internal_pressure_mpa: Internal pressure (MPa)
            outside_diameter_mm: Pipe outside diameter (mm)
            wall_thickness_mm: Minimum wall thickness (mm)

        Returns:
            Hoop stress in MPa
        """
        p = Decimal(str(internal_pressure_mpa))
        d_o = Decimal(str(outside_diameter_mm))
        t_m = Decimal(str(wall_thickness_mm))

        if t_m <= 0:
            raise ValueError("Wall thickness must be positive")

        # Barlow formula for thin-walled cylinder (B31.1 basis)
        s_h = p * d_o / (Decimal("2") * t_m)

        return self._apply_precision(s_h)

    def calculate_longitudinal_stress(
        self,
        internal_pressure_mpa: float,
        outside_diameter_mm: float,
        wall_thickness_mm: float,
        axial_force_n: float = 0.0,
        bending_moment_nm: float = 0.0
    ) -> Decimal:
        """
        Calculate longitudinal stress from pressure, weight, and bending.

        ZERO-HALLUCINATION: Deterministic stress calculation.

        Reference: ASME B31.1, Para. 104.8.1, Eq. 11A

        Formula:
            S_L = P * D_o / (4 * t_m) + F_a / A_m + M_b / Z

        Where:
            P = Internal pressure
            D_o = Outside diameter
            t_m = Minimum wall thickness
            F_a = Axial force (weight, etc.)
            A_m = Metal cross-sectional area
            M_b = Resultant bending moment
            Z = Section modulus

        Args:
            internal_pressure_mpa: Internal pressure (MPa)
            outside_diameter_mm: Outside diameter (mm)
            wall_thickness_mm: Minimum wall thickness (mm)
            axial_force_n: Axial force from sustained loads (N)
            bending_moment_nm: Resultant bending moment (N-mm)

        Returns:
            Longitudinal stress in MPa
        """
        p = Decimal(str(internal_pressure_mpa))
        d_o = Decimal(str(outside_diameter_mm))
        t_m = Decimal(str(wall_thickness_mm))
        f_a = Decimal(str(axial_force_n))
        m_b = Decimal(str(abs(bending_moment_nm)))  # Use absolute value

        if t_m <= 0:
            raise ValueError("Wall thickness must be positive")

        # Calculate geometric properties
        d_i = d_o - Decimal("2") * t_m

        # Metal area: A_m = pi/4 * (D_o^2 - D_i^2)
        pi = Decimal(str(math.pi))
        a_m = pi / Decimal("4") * (d_o ** 2 - d_i ** 2)

        # Section modulus: Z = pi/32 * (D_o^4 - D_i^4) / D_o
        z = pi / Decimal("32") * (d_o ** 4 - d_i ** 4) / d_o

        # Longitudinal stress components
        # 1. Pressure stress: P * D_o / (4 * t_m)
        s_p = p * d_o / (Decimal("4") * t_m)

        # 2. Axial stress: F_a / A_m (convert N to MPa using mm^2)
        s_a = f_a / a_m if a_m > 0 else Decimal("0")

        # 3. Bending stress: M_b / Z (convert N-mm to MPa using mm^3)
        s_b = m_b / z if z > 0 else Decimal("0")

        # Total longitudinal stress
        s_l = s_p + s_a + s_b

        return self._apply_precision(s_l)

    def calculate_sustained_stress(
        self,
        internal_pressure_mpa: float,
        outside_diameter_mm: float,
        wall_thickness_mm: float,
        axial_force_n: float,
        bending_moment_nm: float,
        stress_intensification_factor: float = 1.0
    ) -> Decimal:
        """
        Calculate sustained stress per ASME B31.1.

        ZERO-HALLUCINATION: Deterministic per B31.1 Eq. 11A.

        Reference: ASME B31.1, Para. 104.8.1, Equation 11A

        Formula:
            S_L = P*D_o/(4*t_m) + 0.75*i*M_A/Z <= S_h

        Where:
            P = Internal design pressure
            D_o = Outside diameter
            t_m = Minimum wall thickness
            i = Stress intensification factor (>= 1.0)
            M_A = Resultant moment from sustained loads (sqrt(M_i^2 + M_o^2))
            Z = Section modulus
            S_h = Allowable stress at hot (operating) temperature

        Args:
            internal_pressure_mpa: Design pressure (MPa)
            outside_diameter_mm: Outside diameter (mm)
            wall_thickness_mm: Minimum wall thickness (mm)
            axial_force_n: Axial force (N)
            bending_moment_nm: Resultant bending moment (N-mm)
            stress_intensification_factor: SIF (>= 1.0)

        Returns:
            Sustained stress in MPa
        """
        p = Decimal(str(internal_pressure_mpa))
        d_o = Decimal(str(outside_diameter_mm))
        t_m = Decimal(str(wall_thickness_mm))
        m_a = Decimal(str(abs(bending_moment_nm)))
        i = Decimal(str(max(1.0, stress_intensification_factor)))

        if t_m <= 0:
            raise ValueError("Wall thickness must be positive")

        # Calculate section modulus
        d_i = d_o - Decimal("2") * t_m
        pi = Decimal(str(math.pi))
        z = pi / Decimal("32") * (d_o ** 4 - d_i ** 4) / d_o

        # Pressure term
        s_pressure = p * d_o / (Decimal("4") * t_m)

        # Moment term with SIF
        # Note: 0.75 factor per B31.1 for converting between stress types
        s_moment = Decimal("0.75") * i * m_a / z if z > 0 else Decimal("0")

        # Total sustained stress
        s_l = s_pressure + s_moment

        return self._apply_precision(s_l)

    def calculate_expansion_stress_range(
        self,
        outside_diameter_mm: float,
        wall_thickness_mm: float,
        thermal_bending_moment_nm: float,
        torsional_moment_nm: float,
        stress_intensification_factor: float = 1.0
    ) -> Decimal:
        """
        Calculate thermal expansion stress range per ASME B31.1.

        ZERO-HALLUCINATION: Deterministic per B31.1 Eq. 13.

        Reference: ASME B31.1, Para. 104.8.3, Equations 12 and 13

        Formula:
            S_E = sqrt(S_b^2 + 4*S_t^2)

        Where:
            S_b = i * M_c / Z = Resultant bending stress
            S_t = M_t / (2*Z) = Torsional stress
            M_c = sqrt(M_i^2 + M_o^2) = Range of resultant moments
            M_t = Range of torsional moment
            i = Stress intensification factor
            Z = Section modulus

        Args:
            outside_diameter_mm: Outside diameter (mm)
            wall_thickness_mm: Minimum wall thickness (mm)
            thermal_bending_moment_nm: Resultant thermal bending moment range (N-mm)
            torsional_moment_nm: Torsional moment range (N-mm)
            stress_intensification_factor: SIF (>= 1.0)

        Returns:
            Expansion stress range S_E in MPa
        """
        d_o = Decimal(str(outside_diameter_mm))
        t_m = Decimal(str(wall_thickness_mm))
        m_c = Decimal(str(abs(thermal_bending_moment_nm)))
        m_t = Decimal(str(abs(torsional_moment_nm)))
        i = Decimal(str(max(1.0, stress_intensification_factor)))

        if t_m <= 0:
            raise ValueError("Wall thickness must be positive")

        # Calculate section modulus
        d_i = d_o - Decimal("2") * t_m
        pi = Decimal(str(math.pi))
        z = pi / Decimal("32") * (d_o ** 4 - d_i ** 4) / d_o

        if z <= 0:
            raise ValueError("Section modulus must be positive")

        # Bending stress: S_b = i * M_c / Z
        s_b = i * m_c / z

        # Torsional stress: S_t = M_t / (2*Z)
        s_t = m_t / (Decimal("2") * z)

        # Combined expansion stress range: S_E = sqrt(S_b^2 + 4*S_t^2)
        s_e_squared = s_b ** 2 + Decimal("4") * s_t ** 2
        s_e = Decimal(str(math.sqrt(float(s_e_squared))))

        return self._apply_precision(s_e)

    def calculate_allowable_stress_range(
        self,
        allowable_stress_cold_mpa: float,
        allowable_stress_hot_mpa: float,
        stress_reduction_factor: float = 1.0
    ) -> Decimal:
        """
        Calculate allowable expansion stress range per ASME B31.1.

        ZERO-HALLUCINATION: Deterministic per B31.1 Eq. 1.

        Reference: ASME B31.1, Para. 102.3.2, Equation 1

        Formula:
            S_A = f * (1.25 * S_c + 0.25 * S_h)

        Where:
            f = Stress range reduction factor (see Para. 102.3.2(c))
            S_c = Basic allowable stress at minimum (cold) temperature
            S_h = Basic allowable stress at maximum (hot) temperature

        Note: If S_L (sustained) is less than S_h, the difference may
              be added to S_A per Equation 1b, but this function uses
              the basic Equation 1.

        Args:
            allowable_stress_cold_mpa: S_c - allowable at cold temp (MPa)
            allowable_stress_hot_mpa: S_h - allowable at hot temp (MPa)
            stress_reduction_factor: f - reduction factor for cycles

        Returns:
            Allowable expansion stress range S_A in MPa
        """
        s_c = Decimal(str(allowable_stress_cold_mpa))
        s_h = Decimal(str(allowable_stress_hot_mpa))
        f = Decimal(str(stress_reduction_factor))

        # Validate f factor (0 < f <= 1)
        if f <= 0 or f > 1:
            raise ValueError("Stress reduction factor must be between 0 and 1")

        # S_A = f * (1.25 * S_c + 0.25 * S_h)
        s_a = f * (Decimal("1.25") * s_c + Decimal("0.25") * s_h)

        return self._apply_precision(s_a)

    def get_stress_reduction_factor(self, total_cycles: int) -> Decimal:
        """
        Get stress reduction factor f based on number of cycles.

        Reference: ASME B31.1, Para. 102.3.2(c), Table 102.3.2(c)

        Args:
            total_cycles: Total number of equivalent full temperature cycles

        Returns:
            Stress reduction factor f
        """
        # Table 102.3.2(c) - Stress Reduction Factors
        if total_cycles <= 7000:
            return Decimal("1.0")
        elif total_cycles <= 14000:
            return Decimal("0.9")
        elif total_cycles <= 22000:
            return Decimal("0.8")
        elif total_cycles <= 45000:
            return Decimal("0.7")
        elif total_cycles <= 100000:
            return Decimal("0.6")
        else:
            return Decimal("0.5")

    def calculate_minimum_wall_thickness(
        self,
        internal_pressure_mpa: float,
        outside_diameter_mm: float,
        allowable_stress_mpa: float,
        joint_efficiency: float = 1.0,
        y_factor: float = 0.4,
        mill_tolerance_percent: float = 12.5,
        corrosion_allowance_mm: float = 0.0
    ) -> MinimumThicknessResult:
        """
        Calculate minimum wall thickness for internal pressure.

        ZERO-HALLUCINATION: Deterministic per B31.1 Eq. 3.

        Reference: ASME B31.1, Para. 104.1.2, Equation 3

        Formula:
            t_m = P * D_o / (2 * (S*E + P*y)) + A

        Where:
            t_m = Minimum required wall thickness
            P = Internal design pressure
            D_o = Outside diameter
            S = Allowable stress at design temperature
            E = Quality factor (joint efficiency)
            y = Coefficient from Table 104.1.2(A) (typically 0.4)
            A = Additional thickness for corrosion, erosion, etc.

        The ordered thickness must also account for:
        - Mill tolerance (typically -12.5%)
        - Mechanical allowances

        Args:
            internal_pressure_mpa: Design pressure (MPa)
            outside_diameter_mm: Outside diameter (mm)
            allowable_stress_mpa: Allowable stress at temperature (MPa)
            joint_efficiency: E - Joint efficiency (0 to 1)
            y_factor: y coefficient (0.4 for ferritic < 482C)
            mill_tolerance_percent: Mill tolerance percentage
            corrosion_allowance_mm: Corrosion/erosion allowance (mm)

        Returns:
            MinimumThicknessResult with thickness breakdown
        """
        p = Decimal(str(internal_pressure_mpa))
        d_o = Decimal(str(outside_diameter_mm))
        s = Decimal(str(allowable_stress_mpa))
        e = Decimal(str(joint_efficiency))
        y = Decimal(str(y_factor))
        mt = Decimal(str(mill_tolerance_percent))
        ca = Decimal(str(corrosion_allowance_mm))

        # Validate inputs
        if e <= 0 or e > 1:
            raise ValueError("Joint efficiency must be between 0 and 1")
        if s <= 0:
            raise ValueError("Allowable stress must be positive")
        if d_o <= 0:
            raise ValueError("Diameter must be positive")

        # Calculate pressure design thickness
        # t = P * D_o / (2 * (S*E + P*y))
        denominator = Decimal("2") * (s * e + p * y)
        if denominator <= 0:
            raise ValueError("Invalid stress/pressure combination")

        t_pressure = p * d_o / denominator

        # Add corrosion allowance to get required thickness
        t_required = t_pressure + ca

        # Account for mill tolerance to get ordered thickness
        # t_ordered = t_required / (1 - mill_tolerance/100)
        mt_factor = Decimal("1") - mt / Decimal("100")
        if mt_factor <= 0:
            raise ValueError("Mill tolerance too high")

        t_ordered = t_required / mt_factor

        # Calculate mill tolerance amount
        mt_amount = t_ordered - t_required

        # Create provenance
        inputs = {
            "pressure_mpa": str(p),
            "od_mm": str(d_o),
            "allowable_stress_mpa": str(s),
            "joint_efficiency": str(e),
            "y_factor": str(y)
        }
        outputs = {
            "pressure_design_thickness_mm": str(t_pressure),
            "required_thickness_mm": str(t_required),
            "ordered_thickness_mm": str(t_ordered)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return MinimumThicknessResult(
            required_thickness_mm=self._apply_precision(t_required),
            pressure_design_thickness_mm=self._apply_precision(t_pressure),
            mill_tolerance_mm=self._apply_precision(mt_amount),
            corrosion_allowance_mm=self._apply_precision(ca),
            total_minimum_thickness_mm=self._apply_precision(t_ordered),
            nearest_schedule=None,  # Could be determined with schedule tables
            code_paragraph="Para. 104.1.2, Eq. 3",
            provenance_hash=provenance_hash
        )

    def analyze_piping_stress(
        self,
        geometry: PipeGeometry,
        loads: LoadData,
        material: PipeMaterial,
        design_temperature_c: float,
        ambient_temperature_c: float = 21.0,
        stress_intensification_factor: float = 1.0,
        total_cycles: int = 7000,
        occasional_load_factor: float = 1.15
    ) -> B311StressResult:
        """
        Complete piping stress analysis per ASME B31.1.

        ZERO-HALLUCINATION: Deterministic stress analysis.

        This method performs complete code compliance checks for:
        1. Sustained stress (pressure + weight) vs S_h
        2. Expansion stress range vs S_A
        3. Occasional stress vs k*S_h (k = 1.15 or 1.2)

        References:
            - ASME B31.1, Para. 104.8.1 (Sustained)
            - ASME B31.1, Para. 104.8.2 (Occasional)
            - ASME B31.1, Para. 104.8.3 (Expansion)
            - ASME B31.1, Para. 102.3.2 (Allowable Stress Range)

        Args:
            geometry: Pipe geometry (OD, wall thickness, tolerances)
            loads: Applied loads (pressure, forces, moments)
            material: Pipe material
            design_temperature_c: Operating/design temperature (C)
            ambient_temperature_c: Ambient/cold temperature (C)
            stress_intensification_factor: SIF for fittings
            total_cycles: Expected thermal cycles over design life
            occasional_load_factor: k factor (1.15 for <1% time, 1.2 for <10%)

        Returns:
            B311StressResult with complete stress analysis
        """
        # Get effective wall thickness
        t_m = geometry.effective_wall_thickness_mm
        d_o = geometry.outside_diameter_mm

        if t_m <= 0:
            raise ValueError("Effective wall thickness must be positive")

        # Get material properties
        s_h = self.get_allowable_stress(material, design_temperature_c)
        s_c = self.get_allowable_stress(material, ambient_temperature_c)

        # Get stress reduction factor
        f = self.get_stress_reduction_factor(total_cycles)

        # ================================================================
        # CALCULATE PRIMARY STRESSES
        # ================================================================

        # Hoop stress: S_h = P * D_o / (2 * t_m)
        hoop_stress = self.calculate_hoop_stress(
            loads.internal_pressure_mpa, d_o, t_m
        )

        # Longitudinal stress from pressure
        s_longitudinal = self.calculate_longitudinal_stress(
            loads.internal_pressure_mpa,
            d_o,
            t_m,
            loads.axial_force_n,
            loads.bending_moment_nm
        )

        # Radial stress (at inner wall) = -P
        radial_stress = -Decimal(str(loads.internal_pressure_mpa))

        # ================================================================
        # SUSTAINED STRESS (Para. 104.8.1, Eq. 11A)
        # ================================================================

        sustained_stress = self.calculate_sustained_stress(
            loads.internal_pressure_mpa,
            d_o,
            t_m,
            loads.axial_force_n,
            loads.bending_moment_nm,
            stress_intensification_factor
        )

        # Allowable for sustained: S_h
        sustained_allowable = s_h
        sustained_ratio = sustained_stress / sustained_allowable if sustained_allowable > 0 else Decimal("999")
        sustained_acceptable = sustained_stress <= sustained_allowable

        # ================================================================
        # EXPANSION STRESS RANGE (Para. 104.8.3, Eq. 13)
        # ================================================================

        expansion_stress = self.calculate_expansion_stress_range(
            d_o,
            t_m,
            loads.thermal_bending_moment_nm,
            loads.torsional_moment_nm,
            stress_intensification_factor
        )

        # Allowable expansion stress range: S_A = f * (1.25*S_c + 0.25*S_h)
        expansion_allowable = self.calculate_allowable_stress_range(
            float(s_c), float(s_h), float(f)
        )
        expansion_ratio = expansion_stress / expansion_allowable if expansion_allowable > 0 else Decimal("999")
        expansion_acceptable = expansion_stress <= expansion_allowable

        # ================================================================
        # OCCASIONAL STRESS (Para. 104.8.2)
        # ================================================================

        # Calculate occasional stress by adding occasional loads to sustained
        d_i = Decimal(str(d_o)) - Decimal("2") * Decimal(str(t_m))
        pi = Decimal(str(math.pi))
        z = pi / Decimal("32") * (Decimal(str(d_o)) ** 4 - d_i ** 4) / Decimal(str(d_o))

        # Occasional stress = sustained + occasional moment/Z
        if z > 0:
            occasional_moment_stress = (
                Decimal("0.75") *
                Decimal(str(stress_intensification_factor)) *
                Decimal(str(abs(loads.occasional_moment_nm))) / z
            )
        else:
            occasional_moment_stress = Decimal("0")

        occasional_stress = sustained_stress + occasional_moment_stress

        # Allowable for occasional: k * S_h (k = 1.15 or 1.2)
        k = Decimal(str(occasional_load_factor))
        occasional_allowable = k * s_h
        occasional_ratio = occasional_stress / occasional_allowable if occasional_allowable > 0 else Decimal("999")
        occasional_acceptable = occasional_stress <= occasional_allowable

        # ================================================================
        # OVERALL COMPLIANCE
        # ================================================================

        overall_acceptable = (
            sustained_acceptable and
            expansion_acceptable and
            occasional_acceptable
        )

        # Create provenance
        inputs = {
            "od_mm": str(d_o),
            "wall_mm": str(t_m),
            "pressure_mpa": str(loads.internal_pressure_mpa),
            "material": material.value,
            "temp_c": str(design_temperature_c),
            "sif": str(stress_intensification_factor)
        }
        outputs = {
            "sustained_mpa": str(sustained_stress),
            "expansion_mpa": str(expansion_stress),
            "occasional_mpa": str(occasional_stress),
            "acceptable": str(overall_acceptable)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return B311StressResult(
            pipe_od_mm=self._apply_precision(Decimal(str(d_o))),
            wall_thickness_mm=self._apply_precision(Decimal(str(geometry.wall_thickness_mm))),
            effective_thickness_mm=self._apply_precision(Decimal(str(t_m))),
            hoop_stress_mpa=self._apply_precision(hoop_stress),
            longitudinal_stress_mpa=self._apply_precision(s_longitudinal),
            radial_stress_mpa=self._apply_precision(radial_stress),
            sustained_stress_mpa=self._apply_precision(sustained_stress),
            expansion_stress_range_mpa=self._apply_precision(expansion_stress),
            occasional_stress_mpa=self._apply_precision(occasional_stress),
            allowable_stress_hot_mpa=self._apply_precision(s_h),
            allowable_stress_cold_mpa=self._apply_precision(s_c),
            allowable_expansion_range_mpa=self._apply_precision(expansion_allowable),
            allowable_sustained_mpa=self._apply_precision(sustained_allowable),
            allowable_occasional_mpa=self._apply_precision(occasional_allowable),
            sustained_stress_ratio=self._apply_precision(sustained_ratio),
            expansion_stress_ratio=self._apply_precision(expansion_ratio),
            occasional_stress_ratio=self._apply_precision(occasional_ratio),
            sustained_acceptable=sustained_acceptable,
            expansion_acceptable=expansion_acceptable,
            occasional_acceptable=occasional_acceptable,
            overall_acceptable=overall_acceptable,
            stress_reduction_factor_f=self._apply_precision(f),
            code_paragraph="Para. 104.8",
            code_edition=self._code_edition,
            provenance_hash=provenance_hash
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def pipe_hoop_stress(
    pressure_mpa: float,
    od_mm: float,
    wall_mm: float
) -> Decimal:
    """
    Calculate pipe hoop stress per ASME B31.1.

    Example:
        >>> stress = pipe_hoop_stress(10.0, 323.9, 9.53)
        >>> print(f"Hoop stress: {stress} MPa")
    """
    calc = ASMEB311PipeStress()
    return calc.calculate_hoop_stress(pressure_mpa, od_mm, wall_mm)


def pipe_sustained_stress(
    pressure_mpa: float,
    od_mm: float,
    wall_mm: float,
    axial_force_n: float = 0.0,
    bending_moment_nm: float = 0.0,
    sif: float = 1.0
) -> Decimal:
    """
    Calculate pipe sustained stress per ASME B31.1.

    Example:
        >>> stress = pipe_sustained_stress(10.0, 323.9, 9.53, 0, 50000, 1.5)
        >>> print(f"Sustained stress: {stress} MPa")
    """
    calc = ASMEB311PipeStress()
    return calc.calculate_sustained_stress(
        pressure_mpa, od_mm, wall_mm, axial_force_n, bending_moment_nm, sif
    )


def pipe_expansion_stress(
    od_mm: float,
    wall_mm: float,
    thermal_moment_nm: float,
    torsional_moment_nm: float = 0.0,
    sif: float = 1.0
) -> Decimal:
    """
    Calculate pipe expansion stress range per ASME B31.1.

    Example:
        >>> stress = pipe_expansion_stress(323.9, 9.53, 100000, 20000, 1.5)
        >>> print(f"Expansion stress: {stress} MPa")
    """
    calc = ASMEB311PipeStress()
    return calc.calculate_expansion_stress_range(
        od_mm, wall_mm, thermal_moment_nm, torsional_moment_nm, sif
    )


def pipe_allowable_stress_range(
    s_cold_mpa: float,
    s_hot_mpa: float,
    cycles: int = 7000
) -> Decimal:
    """
    Calculate allowable expansion stress range per ASME B31.1.

    Example:
        >>> s_a = pipe_allowable_stress_range(117.9, 93.1, 10000)
        >>> print(f"Allowable range: {s_a} MPa")
    """
    calc = ASMEB311PipeStress()
    f = calc.get_stress_reduction_factor(cycles)
    return calc.calculate_allowable_stress_range(s_cold_mpa, s_hot_mpa, float(f))


def pipe_minimum_thickness(
    pressure_mpa: float,
    od_mm: float,
    allowable_stress_mpa: float,
    corrosion_mm: float = 0.0
) -> MinimumThicknessResult:
    """
    Calculate minimum pipe wall thickness per ASME B31.1.

    Example:
        >>> result = pipe_minimum_thickness(10.0, 323.9, 117.9, 1.5)
        >>> print(f"Minimum thickness: {result.total_minimum_thickness_mm} mm")
    """
    calc = ASMEB311PipeStress()
    return calc.calculate_minimum_wall_thickness(
        pressure_mpa, od_mm, allowable_stress_mpa,
        corrosion_allowance_mm=corrosion_mm
    )


def analyze_pipe_stress(
    od_mm: float,
    wall_mm: float,
    pressure_mpa: float,
    material: str = "carbon_steel_a106_b",
    temperature_c: float = 371.0,
    bending_moment_nm: float = 0.0,
    thermal_moment_nm: float = 0.0,
    sif: float = 1.0
) -> B311StressResult:
    """
    Complete pipe stress analysis per ASME B31.1.

    Example:
        >>> result = analyze_pipe_stress(
        ...     od_mm=323.9,
        ...     wall_mm=9.53,
        ...     pressure_mpa=10.0,
        ...     material="carbon_steel_a106_b",
        ...     temperature_c=371,
        ...     bending_moment_nm=50000,
        ...     thermal_moment_nm=100000,
        ...     sif=1.5
        ... )
        >>> print(f"Acceptable: {result.overall_acceptable}")
    """
    calc = ASMEB311PipeStress()

    # Map string to enum
    material_map = {
        "carbon_steel_a106_b": PipeMaterial.CARBON_STEEL_A106_B,
        "carbon_steel_a53_b": PipeMaterial.CARBON_STEEL_A53_B,
        "p11": PipeMaterial.LOW_ALLOY_P11,
        "p22": PipeMaterial.LOW_ALLOY_P22,
        "p91": PipeMaterial.LOW_ALLOY_P91,
        "ss_304": PipeMaterial.SS_304,
        "ss_304h": PipeMaterial.SS_304H,
        "ss_316": PipeMaterial.SS_316,
    }
    mat = material_map.get(material.lower(), PipeMaterial.CARBON_STEEL_A106_B)

    # Create geometry and loads
    geometry = PipeGeometry(
        outside_diameter_mm=od_mm,
        wall_thickness_mm=wall_mm
    )

    loads = LoadData(
        internal_pressure_mpa=pressure_mpa,
        bending_moment_nm=bending_moment_nm,
        thermal_bending_moment_nm=thermal_moment_nm
    )

    return calc.analyze_piping_stress(
        geometry=geometry,
        loads=loads,
        material=mat,
        design_temperature_c=temperature_c,
        stress_intensification_factor=sif
    )
