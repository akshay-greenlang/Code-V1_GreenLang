"""
GL-015 INSULSCAN Heat Loss Calculator

ZERO-HALLUCINATION calculation engine implementing ASTM C680 standard formulas
for heat loss from bare and insulated surfaces.

Physical Constants (NIST Reference):
    - Stefan-Boltzmann constant: 5.670374419e-8 W/(m^2*K^4)
    - Absolute zero offset: 273.15 K

Heat Transfer Mechanisms:
    1. Radiation: q_rad = epsilon * sigma * A * (T_s^4 - T_a^4)
    2. Natural Convection: h_nc = f(Gr, Pr, geometry)
    3. Forced Convection: h_fc = f(Re, Pr, geometry)

Standards Compliance: ASTM C680, ISO 12241

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math


# Physical Constants (NIST values - DO NOT MODIFY)
STEFAN_BOLTZMANN = Decimal("5.670374419E-8")  # W/(m^2*K^4)
ABSOLUTE_ZERO_OFFSET = Decimal("273.15")  # K


class SurfaceType(Enum):
    """Surface geometry types for convection correlations."""
    HORIZONTAL_PIPE = "horizontal_pipe"
    VERTICAL_PIPE = "vertical_pipe"
    HORIZONTAL_FLAT = "horizontal_flat_up"
    HORIZONTAL_FLAT_DOWN = "horizontal_flat_down"
    VERTICAL_FLAT = "vertical_flat"


@dataclass(frozen=True)
class HeatLossResult:
    """
    Immutable result container with full provenance tracking.

    Attributes:
        total_heat_loss_w: Total heat loss in watts
        convection_loss_w: Convective heat loss component
        radiation_loss_w: Radiative heat loss component
        convection_coefficient_w_m2k: Combined convection coefficient
        provenance_hash: SHA-256 hash for audit trail
        calculation_inputs: Dictionary of all input parameters
    """
    total_heat_loss_w: Decimal
    convection_loss_w: Decimal
    radiation_loss_w: Decimal
    convection_coefficient_w_m2k: Decimal
    provenance_hash: str
    calculation_inputs: Dict[str, Any]


class HeatLossCalculator:
    """
    ASTM C680 compliant heat loss calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic formulas from ASTM C680
    - No LLM inference in calculation path
    - Complete provenance tracking with SHA-256 hashes
    - Bit-perfect reproducibility (same input -> same output)

    Example Usage:
        >>> calc = HeatLossCalculator()
        >>> result = calc.calculate_bare_surface_loss(
        ...     surface_temp_c=150.0,
        ...     ambient_temp_c=25.0,
        ...     surface_area_m2=1.0,
        ...     wind_speed_ms=0.0,
        ...     surface_type=SurfaceType.HORIZONTAL_PIPE,
        ...     emissivity=0.9
        ... )
        >>> isinstance(result.total_heat_loss_w, Decimal)
        True
        >>> result.total_heat_loss_w > 0
        True

    Determinism Test:
        >>> calc = HeatLossCalculator()
        >>> r1 = calc.calculate_bare_surface_loss(100.0, 20.0, 2.0, 0.0, SurfaceType.VERTICAL_FLAT)
        >>> r2 = calc.calculate_bare_surface_loss(100.0, 20.0, 2.0, 0.0, SurfaceType.VERTICAL_FLAT)
        >>> r1.total_heat_loss_w == r2.total_heat_loss_w
        True
        >>> r1.provenance_hash == r2.provenance_hash
        True
    """

    # Default emissivity values by surface condition (ASTM C680 Table 1)
    DEFAULT_EMISSIVITY = {
        "oxidized_steel": Decimal("0.79"),
        "galvanized_steel_new": Decimal("0.28"),
        "galvanized_steel_weathered": Decimal("0.88"),
        "aluminum_polished": Decimal("0.05"),
        "aluminum_oxidized": Decimal("0.15"),
        "aluminum_jacket": Decimal("0.10"),
        "stainless_steel": Decimal("0.45"),
        "painted_surface": Decimal("0.90"),
        "canvas_jacket": Decimal("0.90"),
        "default": Decimal("0.90"),
    }

    # Air properties at film temperature (simplified - use lookup for production)
    # Reference: ASHRAE Fundamentals 2021
    AIR_THERMAL_CONDUCTIVITY = Decimal("0.0262")  # W/(m*K) at 300K
    AIR_KINEMATIC_VISCOSITY = Decimal("1.568E-5")  # m^2/s at 300K
    AIR_PRANDTL = Decimal("0.707")  # dimensionless at 300K
    GRAVITY = Decimal("9.81")  # m/s^2

    def __init__(self, precision: int = 6):
        """
        Initialize calculator with specified decimal precision.

        Args:
            precision: Number of decimal places for output (default: 6)
        """
        self.precision = precision
        self._quantize_str = "0." + "0" * precision

    def calculate_bare_surface_loss(
        self,
        surface_temp_c: float,
        ambient_temp_c: float,
        surface_area_m2: float,
        wind_speed_ms: float = 0.0,
        surface_type: SurfaceType = SurfaceType.HORIZONTAL_PIPE,
        emissivity: float = 0.9,
        characteristic_length_m: Optional[float] = None
    ) -> HeatLossResult:
        """
        Calculate heat loss from bare (uninsulated) surface.

        ASTM C680 Section 6.1: Combined convection and radiation heat transfer.

        Args:
            surface_temp_c: Surface temperature in Celsius
            ambient_temp_c: Ambient temperature in Celsius
            surface_area_m2: Surface area in square meters
            wind_speed_ms: Wind speed in m/s (0 for natural convection)
            surface_type: Geometry type for convection correlations
            emissivity: Surface emissivity (0-1), default 0.9 for painted
            characteristic_length_m: Characteristic length for convection
                                    (defaults to sqrt(area) for flat surfaces)

        Returns:
            HeatLossResult with total heat loss and provenance

        Example - Natural Convection:
            >>> calc = HeatLossCalculator()
            >>> result = calc.calculate_bare_surface_loss(
            ...     surface_temp_c=100.0,
            ...     ambient_temp_c=25.0,
            ...     surface_area_m2=1.0,
            ...     wind_speed_ms=0.0,
            ...     surface_type=SurfaceType.VERTICAL_FLAT
            ... )
            >>> 800 < float(result.total_heat_loss_w) < 1200
            True

        Example - Forced Convection:
            >>> calc = HeatLossCalculator()
            >>> result = calc.calculate_bare_surface_loss(
            ...     surface_temp_c=100.0,
            ...     ambient_temp_c=25.0,
            ...     surface_area_m2=1.0,
            ...     wind_speed_ms=5.0,
            ...     surface_type=SurfaceType.HORIZONTAL_PIPE
            ... )
            >>> float(result.convection_loss_w) > 1000
            True
        """
        # Convert inputs to Decimal for precision
        T_s = Decimal(str(surface_temp_c))
        T_a = Decimal(str(ambient_temp_c))
        A = Decimal(str(surface_area_m2))
        V = Decimal(str(wind_speed_ms))
        eps = Decimal(str(emissivity))

        # Validate inputs
        self._validate_temperature_inputs(T_s, T_a)
        self._validate_positive("surface_area_m2", A)
        self._validate_range("emissivity", eps, Decimal("0"), Decimal("1"))

        # Convert to Kelvin
        T_s_k = T_s + ABSOLUTE_ZERO_OFFSET
        T_a_k = T_a + ABSOLUTE_ZERO_OFFSET

        # Temperature difference
        delta_T = T_s - T_a

        # Calculate characteristic length
        if characteristic_length_m is not None:
            L = Decimal(str(characteristic_length_m))
        else:
            L = A.sqrt()  # Default for flat surfaces

        # Calculate radiation heat loss (Stefan-Boltzmann law)
        q_rad = self._calculate_radiation_loss(eps, A, T_s_k, T_a_k)

        # Calculate convection heat loss
        if V > Decimal("0.1"):  # Forced convection threshold
            h_conv = self._calculate_forced_convection_coefficient(V, L, surface_type)
        else:
            h_conv = self._calculate_natural_convection_coefficient(
                T_s_k, T_a_k, L, surface_type
            )

        q_conv = h_conv * A * delta_T

        # Total heat loss
        q_total = q_rad + q_conv

        # Apply precision
        q_total = self._apply_precision(q_total)
        q_rad = self._apply_precision(q_rad)
        q_conv = self._apply_precision(q_conv)
        h_conv = self._apply_precision(h_conv)

        # Build provenance
        inputs = {
            "surface_temp_c": str(surface_temp_c),
            "ambient_temp_c": str(ambient_temp_c),
            "surface_area_m2": str(surface_area_m2),
            "wind_speed_ms": str(wind_speed_ms),
            "surface_type": surface_type.value,
            "emissivity": str(emissivity),
            "characteristic_length_m": str(characteristic_length_m),
        }

        provenance_hash = self._calculate_provenance_hash(
            "bare_surface_loss", inputs, str(q_total)
        )

        return HeatLossResult(
            total_heat_loss_w=q_total,
            convection_loss_w=q_conv,
            radiation_loss_w=q_rad,
            convection_coefficient_w_m2k=h_conv,
            provenance_hash=provenance_hash,
            calculation_inputs=inputs
        )

    def calculate_insulated_surface_loss(
        self,
        operating_temp_c: float,
        ambient_temp_c: float,
        insulation_type: str,
        thickness_mm: float,
        surface_area_m2: float,
        pipe_outer_diameter_mm: Optional[float] = None,
        surface_emissivity: float = 0.9,
        wind_speed_ms: float = 0.0
    ) -> HeatLossResult:
        """
        Calculate heat loss from insulated surface.

        ASTM C680 Section 6.2: Iterative calculation for insulation outer
        surface temperature, then combined heat transfer.

        Args:
            operating_temp_c: Process/pipe temperature in Celsius
            ambient_temp_c: Ambient temperature in Celsius
            insulation_type: Type of insulation (see THERMAL_CONDUCTIVITY_TABLE)
            thickness_mm: Insulation thickness in millimeters
            surface_area_m2: Outer surface area in square meters
            pipe_outer_diameter_mm: Pipe OD for cylindrical geometry (None for flat)
            surface_emissivity: Jacket surface emissivity
            wind_speed_ms: Wind speed in m/s

        Returns:
            HeatLossResult with total heat loss and provenance

        Example - Mineral Wool on Flat Surface:
            >>> calc = HeatLossCalculator()
            >>> result = calc.calculate_insulated_surface_loss(
            ...     operating_temp_c=200.0,
            ...     ambient_temp_c=25.0,
            ...     insulation_type="mineral_wool",
            ...     thickness_mm=50.0,
            ...     surface_area_m2=1.0
            ... )
            >>> float(result.total_heat_loss_w) < 200
            True

        Example - Calcium Silicate on Pipe:
            >>> calc = HeatLossCalculator()
            >>> result = calc.calculate_insulated_surface_loss(
            ...     operating_temp_c=300.0,
            ...     ambient_temp_c=25.0,
            ...     insulation_type="calcium_silicate",
            ...     thickness_mm=75.0,
            ...     surface_area_m2=2.0,
            ...     pipe_outer_diameter_mm=168.3
            ... )
            >>> 100 < float(result.total_heat_loss_w) < 500
            True
        """
        # Convert inputs to Decimal
        T_op = Decimal(str(operating_temp_c))
        T_a = Decimal(str(ambient_temp_c))
        t = Decimal(str(thickness_mm)) / Decimal("1000")  # Convert to meters
        A = Decimal(str(surface_area_m2))
        eps = Decimal(str(surface_emissivity))
        V = Decimal(str(wind_speed_ms))

        # Validate inputs
        self._validate_temperature_inputs(T_op, T_a)
        self._validate_positive("thickness_mm", Decimal(str(thickness_mm)))
        self._validate_positive("surface_area_m2", A)

        # Get thermal conductivity
        k = self._get_thermal_conductivity(insulation_type, T_op)

        # Calculate thermal resistance
        if pipe_outer_diameter_mm is not None:
            # Cylindrical geometry
            r_i = Decimal(str(pipe_outer_diameter_mm)) / Decimal("2000")  # m
            r_o = r_i + t
            # R = ln(r_o/r_i) / (2 * pi * k * L)
            # For unit area calculation, use per-meter basis
            R_ins = (r_o / r_i).ln() / (Decimal("2") * Decimal(str(math.pi)) * k)
        else:
            # Flat geometry: R = t / k
            R_ins = t / k

        # Iterative solution for surface temperature (ASTM C680 Section 6.2.3)
        T_s = self._solve_surface_temperature(
            T_op, T_a, R_ins, A, eps, V, max_iterations=50
        )

        # Convert temperatures to Kelvin
        T_s_k = T_s + ABSOLUTE_ZERO_OFFSET
        T_a_k = T_a + ABSOLUTE_ZERO_OFFSET

        # Calculate heat loss at converged surface temperature
        delta_T_surface = T_s - T_a

        # Radiation loss
        q_rad = self._calculate_radiation_loss(eps, A, T_s_k, T_a_k)

        # Convection loss
        L = A.sqrt()
        if V > Decimal("0.1"):
            h_conv = self._calculate_forced_convection_coefficient(
                V, L, SurfaceType.HORIZONTAL_PIPE
            )
        else:
            h_conv = self._calculate_natural_convection_coefficient(
                T_s_k, T_a_k, L, SurfaceType.HORIZONTAL_PIPE
            )

        q_conv = h_conv * A * delta_T_surface

        # Total heat loss
        q_total = q_rad + q_conv

        # Apply precision
        q_total = self._apply_precision(q_total)
        q_rad = self._apply_precision(q_rad)
        q_conv = self._apply_precision(q_conv)
        h_conv = self._apply_precision(h_conv)

        # Build provenance
        inputs = {
            "operating_temp_c": str(operating_temp_c),
            "ambient_temp_c": str(ambient_temp_c),
            "insulation_type": insulation_type,
            "thickness_mm": str(thickness_mm),
            "surface_area_m2": str(surface_area_m2),
            "pipe_outer_diameter_mm": str(pipe_outer_diameter_mm),
            "surface_emissivity": str(surface_emissivity),
            "wind_speed_ms": str(wind_speed_ms),
            "calculated_surface_temp_c": str(T_s),
            "thermal_conductivity_w_mk": str(k),
        }

        provenance_hash = self._calculate_provenance_hash(
            "insulated_surface_loss", inputs, str(q_total)
        )

        return HeatLossResult(
            total_heat_loss_w=q_total,
            convection_loss_w=q_conv,
            radiation_loss_w=q_rad,
            convection_coefficient_w_m2k=h_conv,
            provenance_hash=provenance_hash,
            calculation_inputs=inputs
        )

    def _calculate_radiation_loss(
        self,
        emissivity: Decimal,
        area: Decimal,
        T_surface_k: Decimal,
        T_ambient_k: Decimal
    ) -> Decimal:
        """
        Calculate radiation heat loss using Stefan-Boltzmann law.

        Formula: q = epsilon * sigma * A * (T_s^4 - T_a^4)

        Reference: ASTM C680 Section 5.2.1

        >>> calc = HeatLossCalculator()
        >>> q = calc._calculate_radiation_loss(
        ...     Decimal("0.9"),
        ...     Decimal("1.0"),
        ...     Decimal("373.15"),  # 100C
        ...     Decimal("298.15")   # 25C
        ... )
        >>> 500 < float(q) < 700
        True
        """
        T_s_4 = T_surface_k ** 4
        T_a_4 = T_ambient_k ** 4

        q_rad = emissivity * STEFAN_BOLTZMANN * area * (T_s_4 - T_a_4)

        return q_rad

    def _calculate_natural_convection_coefficient(
        self,
        T_surface_k: Decimal,
        T_ambient_k: Decimal,
        characteristic_length: Decimal,
        surface_type: SurfaceType
    ) -> Decimal:
        """
        Calculate natural convection heat transfer coefficient.

        Uses Churchill-Chu correlation for vertical surfaces and
        Morgan correlation for horizontal cylinders.

        Reference: ASTM C680 Section 5.2.2, Incropera 6th Ed.

        >>> calc = HeatLossCalculator()
        >>> h = calc._calculate_natural_convection_coefficient(
        ...     Decimal("373.15"),
        ...     Decimal("298.15"),
        ...     Decimal("1.0"),
        ...     SurfaceType.VERTICAL_FLAT
        ... )
        >>> 3 < float(h) < 15
        True
        """
        # Film temperature
        T_film = (T_surface_k + T_ambient_k) / Decimal("2")

        # Temperature difference
        delta_T = abs(T_surface_k - T_ambient_k)

        # Volumetric expansion coefficient (ideal gas approximation)
        beta = Decimal("1") / T_film

        # Grashof number: Gr = g * beta * delta_T * L^3 / nu^2
        Gr = (
            self.GRAVITY * beta * delta_T *
            (characteristic_length ** 3) /
            (self.AIR_KINEMATIC_VISCOSITY ** 2)
        )

        # Rayleigh number: Ra = Gr * Pr
        Ra = Gr * self.AIR_PRANDTL

        # Nusselt number correlation based on geometry
        if surface_type == SurfaceType.VERTICAL_FLAT:
            # Churchill-Chu correlation for vertical plates
            Nu = self._churchill_chu_vertical(Ra)
        elif surface_type == SurfaceType.HORIZONTAL_PIPE:
            # Churchill-Chu correlation for horizontal cylinders
            Nu = self._churchill_chu_horizontal_cylinder(Ra)
        elif surface_type == SurfaceType.HORIZONTAL_FLAT:
            # Hot surface facing up
            Nu = self._mcadams_horizontal_up(Ra)
        elif surface_type == SurfaceType.HORIZONTAL_FLAT_DOWN:
            # Hot surface facing down
            Nu = self._mcadams_horizontal_down(Ra)
        else:
            # Default to vertical
            Nu = self._churchill_chu_vertical(Ra)

        # Heat transfer coefficient: h = Nu * k / L
        h = Nu * self.AIR_THERMAL_CONDUCTIVITY / characteristic_length

        return h

    def _calculate_forced_convection_coefficient(
        self,
        wind_speed: Decimal,
        characteristic_length: Decimal,
        surface_type: SurfaceType
    ) -> Decimal:
        """
        Calculate forced convection heat transfer coefficient.

        Uses Hilpert correlation for cylinders and flat plate correlation.

        Reference: ASTM C680 Section 5.2.3

        >>> calc = HeatLossCalculator()
        >>> h = calc._calculate_forced_convection_coefficient(
        ...     Decimal("5.0"),
        ...     Decimal("0.1"),
        ...     SurfaceType.HORIZONTAL_PIPE
        ... )
        >>> 20 < float(h) < 100
        True
        """
        # Reynolds number: Re = V * L / nu
        Re = wind_speed * characteristic_length / self.AIR_KINEMATIC_VISCOSITY

        # Nusselt number based on geometry
        if surface_type in [SurfaceType.HORIZONTAL_PIPE, SurfaceType.VERTICAL_PIPE]:
            # Hilpert correlation for cylinder in cross-flow
            Nu = self._hilpert_correlation(Re)
        else:
            # Flat plate correlation
            Nu = self._flat_plate_forced(Re)

        # Heat transfer coefficient
        h = Nu * self.AIR_THERMAL_CONDUCTIVITY / characteristic_length

        return h

    def _churchill_chu_vertical(self, Ra: Decimal) -> Decimal:
        """
        Churchill-Chu correlation for natural convection on vertical surfaces.

        Valid for 10^-1 < Ra < 10^12
        Nu = {0.825 + 0.387*Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2
        """
        # Simplified for Pr = 0.707 (air)
        denominator = Decimal("1") + (Decimal("0.492") / self.AIR_PRANDTL) ** (Decimal("9") / Decimal("16"))
        denominator = denominator ** (Decimal("8") / Decimal("27"))

        Ra_sixth = self._decimal_power(Ra, Decimal("1") / Decimal("6"))

        Nu = (Decimal("0.825") + Decimal("0.387") * Ra_sixth / denominator) ** 2

        return Nu

    def _churchill_chu_horizontal_cylinder(self, Ra: Decimal) -> Decimal:
        """
        Churchill-Chu correlation for horizontal cylinders.

        Nu = {0.60 + 0.387*Ra^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2
        """
        denominator = Decimal("1") + (Decimal("0.559") / self.AIR_PRANDTL) ** (Decimal("9") / Decimal("16"))
        denominator = denominator ** (Decimal("8") / Decimal("27"))

        Ra_sixth = self._decimal_power(Ra, Decimal("1") / Decimal("6"))

        Nu = (Decimal("0.60") + Decimal("0.387") * Ra_sixth / denominator) ** 2

        return Nu

    def _mcadams_horizontal_up(self, Ra: Decimal) -> Decimal:
        """McAdams correlation for hot horizontal surface facing up."""
        if Ra < Decimal("1E7"):
            return Decimal("0.54") * self._decimal_power(Ra, Decimal("0.25"))
        else:
            return Decimal("0.15") * self._decimal_power(Ra, Decimal("1") / Decimal("3"))

    def _mcadams_horizontal_down(self, Ra: Decimal) -> Decimal:
        """McAdams correlation for hot horizontal surface facing down."""
        return Decimal("0.27") * self._decimal_power(Ra, Decimal("0.25"))

    def _hilpert_correlation(self, Re: Decimal) -> Decimal:
        """
        Hilpert correlation for forced convection on cylinders.

        Nu = C * Re^m * Pr^(1/3)
        """
        # Determine C and m based on Reynolds number range
        if Re < Decimal("4"):
            C, m = Decimal("0.989"), Decimal("0.330")
        elif Re < Decimal("40"):
            C, m = Decimal("0.911"), Decimal("0.385")
        elif Re < Decimal("4000"):
            C, m = Decimal("0.683"), Decimal("0.466")
        elif Re < Decimal("40000"):
            C, m = Decimal("0.193"), Decimal("0.618")
        else:
            C, m = Decimal("0.027"), Decimal("0.805")

        Pr_third = self._decimal_power(self.AIR_PRANDTL, Decimal("1") / Decimal("3"))
        Re_m = self._decimal_power(Re, m)

        Nu = C * Re_m * Pr_third

        return Nu

    def _flat_plate_forced(self, Re: Decimal) -> Decimal:
        """Flat plate forced convection correlation."""
        if Re < Decimal("5E5"):
            # Laminar: Nu = 0.664 * Re^0.5 * Pr^(1/3)
            return Decimal("0.664") * self._decimal_power(Re, Decimal("0.5")) * \
                   self._decimal_power(self.AIR_PRANDTL, Decimal("1") / Decimal("3"))
        else:
            # Turbulent: Nu = 0.037 * Re^0.8 * Pr^(1/3)
            return Decimal("0.037") * self._decimal_power(Re, Decimal("0.8")) * \
                   self._decimal_power(self.AIR_PRANDTL, Decimal("1") / Decimal("3"))

    def _solve_surface_temperature(
        self,
        T_operating: Decimal,
        T_ambient: Decimal,
        R_insulation: Decimal,
        area: Decimal,
        emissivity: Decimal,
        wind_speed: Decimal,
        max_iterations: int = 50,
        tolerance: Decimal = Decimal("0.01")
    ) -> Decimal:
        """
        Iteratively solve for insulation outer surface temperature.

        ASTM C680 Section 6.2.3: Balance heat conducted through insulation
        with heat lost from surface by convection + radiation.
        """
        # Initial guess: linear interpolation
        T_s = T_ambient + (T_operating - T_ambient) * Decimal("0.1")

        for _ in range(max_iterations):
            T_s_k = T_s + ABSOLUTE_ZERO_OFFSET
            T_a_k = T_ambient + ABSOLUTE_ZERO_OFFSET

            # Heat loss from surface
            q_rad = self._calculate_radiation_loss(emissivity, area, T_s_k, T_a_k)

            L = area.sqrt()
            if wind_speed > Decimal("0.1"):
                h_conv = self._calculate_forced_convection_coefficient(
                    wind_speed, L, SurfaceType.HORIZONTAL_PIPE
                )
            else:
                h_conv = self._calculate_natural_convection_coefficient(
                    T_s_k, T_a_k, L, SurfaceType.HORIZONTAL_PIPE
                )

            q_conv = h_conv * area * (T_s - T_ambient)
            q_surface = q_rad + q_conv

            # Heat conducted through insulation
            # q = (T_op - T_s) / R_ins * A
            q_conducted = (T_operating - T_s) / R_insulation * area

            # Energy balance error
            error = q_conducted - q_surface

            # Adjust surface temperature
            # Use secant method approximation
            h_total = h_conv + self._linearized_radiation_coefficient(
                emissivity, T_s_k, T_a_k
            )

            delta_T_s = error / (area / R_insulation + h_total * area)
            T_s_new = T_s + delta_T_s

            # Check convergence
            if abs(T_s_new - T_s) < tolerance:
                return T_s_new

            T_s = T_s_new

        return T_s

    def _linearized_radiation_coefficient(
        self,
        emissivity: Decimal,
        T_surface_k: Decimal,
        T_ambient_k: Decimal
    ) -> Decimal:
        """
        Linearized radiation heat transfer coefficient.

        h_rad = epsilon * sigma * (T_s^2 + T_a^2) * (T_s + T_a)
        """
        T_s_2 = T_surface_k ** 2
        T_a_2 = T_ambient_k ** 2

        h_rad = emissivity * STEFAN_BOLTZMANN * (T_s_2 + T_a_2) * (T_surface_k + T_ambient_k)

        return h_rad

    def _get_thermal_conductivity(
        self,
        insulation_type: str,
        mean_temp_c: Decimal
    ) -> Decimal:
        """
        Get thermal conductivity for insulation type at given temperature.

        Uses temperature-dependent correlations from manufacturer data.
        """
        # Thermal conductivity database (W/m-K at 24C reference)
        # Temperature correction: k(T) = k_24 * (1 + alpha * (T - 24))
        CONDUCTIVITY_DATA = {
            "mineral_wool": {
                "k_24": Decimal("0.040"),
                "alpha": Decimal("0.0004"),
            },
            "calcium_silicate": {
                "k_24": Decimal("0.065"),
                "alpha": Decimal("0.0003"),
            },
            "fiberglass": {
                "k_24": Decimal("0.038"),
                "alpha": Decimal("0.0004"),
            },
            "cellular_glass": {
                "k_24": Decimal("0.048"),
                "alpha": Decimal("0.0002"),
            },
            "perlite": {
                "k_24": Decimal("0.055"),
                "alpha": Decimal("0.0003"),
            },
            "polyurethane_foam": {
                "k_24": Decimal("0.025"),
                "alpha": Decimal("0.0006"),
            },
            "phenolic_foam": {
                "k_24": Decimal("0.022"),
                "alpha": Decimal("0.0005"),
            },
            "aerogel": {
                "k_24": Decimal("0.015"),
                "alpha": Decimal("0.0003"),
            },
        }

        insulation_lower = insulation_type.lower().replace(" ", "_")

        if insulation_lower not in CONDUCTIVITY_DATA:
            raise ValueError(
                f"Unknown insulation type: {insulation_type}. "
                f"Available types: {list(CONDUCTIVITY_DATA.keys())}"
            )

        data = CONDUCTIVITY_DATA[insulation_lower]
        k_24 = data["k_24"]
        alpha = data["alpha"]

        # Mean temperature for conductivity
        T_mean = mean_temp_c

        # Temperature-corrected conductivity
        k = k_24 * (Decimal("1") + alpha * (T_mean - Decimal("24")))

        return k

    def _decimal_power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate decimal power using logarithms for non-integer exponents."""
        if base <= 0:
            return Decimal("0")

        # Use float for intermediate calculation, convert back to Decimal
        result = float(base) ** float(exponent)
        return Decimal(str(result))

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply configured precision using ROUND_HALF_UP."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _validate_temperature_inputs(self, T_surface: Decimal, T_ambient: Decimal) -> None:
        """Validate temperature inputs are physically reasonable."""
        if T_surface < Decimal("-273.15"):
            raise ValueError(f"Surface temperature {T_surface}C below absolute zero")
        if T_ambient < Decimal("-273.15"):
            raise ValueError(f"Ambient temperature {T_ambient}C below absolute zero")
        if T_surface < T_ambient:
            raise ValueError(
                f"Surface temperature ({T_surface}C) must be >= ambient ({T_ambient}C) "
                "for heat loss calculation"
            )

    def _validate_positive(self, name: str, value: Decimal) -> None:
        """Validate value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def _validate_range(
        self,
        name: str,
        value: Decimal,
        min_val: Decimal,
        max_val: Decimal
    ) -> None:
        """Validate value is within range."""
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

    def _calculate_provenance_hash(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        result: str
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Ensures bit-perfect reproducibility and tamper detection.
        """
        provenance_data = {
            "calculator": "HeatLossCalculator",
            "version": "1.0.0",
            "standard": "ASTM C680",
            "calculation_type": calculation_type,
            "inputs": inputs,
            "result": result,
        }

        # Deterministic JSON serialization
        provenance_str = json.dumps(provenance_data, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
