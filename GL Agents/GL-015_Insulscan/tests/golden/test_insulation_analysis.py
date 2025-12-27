"""
GL-015 InsulScan: Insulation Analysis Golden Value Tests.

Reference Standards:
- ASTM C680: Standard Practice for Estimate of Heat Gain/Loss for Insulated Pipe
- ASTM C585: Standard Practice for Inner/Outer Diameters of Thermal Insulation
- ASTM C335: Standard Test Method for Steady-State Heat Transfer Properties
- ASHRAE Fundamentals: Chapter 26 - Heat, Air, and Moisture Control
- 3E Plus (NAIMA): North American Insulation Manufacturers Association Software

These golden tests validate insulation thermal performance, heat loss calculations,
economic thickness optimization, and condensation prevention analysis.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

import pytest

# =============================================================================
# GOLDEN VALUE REFERENCE DATA - INSULATION THERMAL PROPERTIES
# =============================================================================


@dataclass(frozen=True)
class InsulationGoldenValue:
    """Immutable golden value for insulation validation."""

    description: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    source: str
    temp_range_c: Tuple[Decimal, Decimal]


# Thermal Conductivity (k) Values at Mean Temperature
# Reference: ASTM C680, Manufacturer Data, ASHRAE Fundamentals
THERMAL_CONDUCTIVITY: Dict[str, InsulationGoldenValue] = {
    # Mineral/Slag Wool (ASTM C547)
    'mineral_wool_50c': InsulationGoldenValue(
        'Mineral Wool k at 50°C',
        Decimal('0.038'),
        'W/m-K',
        Decimal('10'),
        'ASTM C547',
        (Decimal('-18'), Decimal('650')),
    ),
    'mineral_wool_100c': InsulationGoldenValue(
        'Mineral Wool k at 100°C',
        Decimal('0.045'),
        'W/m-K',
        Decimal('10'),
        'ASTM C547',
        (Decimal('-18'), Decimal('650')),
    ),
    'mineral_wool_200c': InsulationGoldenValue(
        'Mineral Wool k at 200°C',
        Decimal('0.058'),
        'W/m-K',
        Decimal('10'),
        'ASTM C547',
        (Decimal('-18'), Decimal('650')),
    ),
    # Calcium Silicate (ASTM C533)
    'calcium_silicate_50c': InsulationGoldenValue(
        'Calcium Silicate k at 50°C',
        Decimal('0.055'),
        'W/m-K',
        Decimal('10'),
        'ASTM C533',
        (Decimal('38'), Decimal('650')),
    ),
    'calcium_silicate_200c': InsulationGoldenValue(
        'Calcium Silicate k at 200°C',
        Decimal('0.070'),
        'W/m-K',
        Decimal('10'),
        'ASTM C533',
        (Decimal('38'), Decimal('650')),
    ),
    'calcium_silicate_400c': InsulationGoldenValue(
        'Calcium Silicate k at 400°C',
        Decimal('0.095'),
        'W/m-K',
        Decimal('10'),
        'ASTM C533',
        (Decimal('38'), Decimal('650')),
    ),
    # Cellular Glass (ASTM C552)
    'cellular_glass_24c': InsulationGoldenValue(
        'Cellular Glass k at 24°C',
        Decimal('0.043'),
        'W/m-K',
        Decimal('10'),
        'ASTM C552',
        (Decimal('-268'), Decimal('430')),
    ),
    'cellular_glass_100c': InsulationGoldenValue(
        'Cellular Glass k at 100°C',
        Decimal('0.055'),
        'W/m-K',
        Decimal('10'),
        'ASTM C552',
        (Decimal('-268'), Decimal('430')),
    ),
    # Fiberglass (ASTM C547)
    'fiberglass_24c': InsulationGoldenValue(
        'Fiberglass k at 24°C',
        Decimal('0.033'),
        'W/m-K',
        Decimal('10'),
        'ASTM C547',
        (Decimal('-18'), Decimal('450')),
    ),
    'fiberglass_100c': InsulationGoldenValue(
        'Fiberglass k at 100°C',
        Decimal('0.041'),
        'W/m-K',
        Decimal('10'),
        'ASTM C547',
        (Decimal('-18'), Decimal('450')),
    ),
    # Polyisocyanurate (PIR) Foam
    'pir_foam_24c': InsulationGoldenValue(
        'PIR Foam k at 24°C',
        Decimal('0.023'),
        'W/m-K',
        Decimal('10'),
        'ASTM C591',
        (Decimal('-73'), Decimal('149')),
    ),
    # Aerogel Blanket
    'aerogel_24c': InsulationGoldenValue(
        'Aerogel Blanket k at 24°C',
        Decimal('0.015'),
        'W/m-K',
        Decimal('15'),
        'Manufacturer',
        (Decimal('-200'), Decimal('650')),
    ),
    # Perlite
    'perlite_100c': InsulationGoldenValue(
        'Expanded Perlite k at 100°C',
        Decimal('0.055'),
        'W/m-K',
        Decimal('10'),
        'ASTM C610',
        (Decimal('-18'), Decimal('650')),
    ),
}


# Surface Emissivity Values
# Reference: ASHRAE Fundamentals Table 26.1
SURFACE_EMISSIVITY: Dict[str, InsulationGoldenValue] = {
    'aluminum_bright': InsulationGoldenValue(
        'Bright Aluminum Emissivity',
        Decimal('0.04'),
        'dimensionless',
        Decimal('25'),
        'ASHRAE',
        (Decimal('0'), Decimal('200')),
    ),
    'aluminum_oxidized': InsulationGoldenValue(
        'Oxidized Aluminum Emissivity',
        Decimal('0.12'),
        'dimensionless',
        Decimal('20'),
        'ASHRAE',
        (Decimal('0'), Decimal('300')),
    ),
    'stainless_steel': InsulationGoldenValue(
        'Stainless Steel Emissivity',
        Decimal('0.50'),
        'dimensionless',
        Decimal('20'),
        'ASHRAE',
        (Decimal('0'), Decimal('500')),
    ),
    'galvanized_steel': InsulationGoldenValue(
        'Galvanized Steel Emissivity',
        Decimal('0.28'),
        'dimensionless',
        Decimal('20'),
        'ASHRAE',
        (Decimal('0'), Decimal('200')),
    ),
    'painted_surface': InsulationGoldenValue(
        'Painted Surface Emissivity',
        Decimal('0.90'),
        'dimensionless',
        Decimal('10'),
        'ASHRAE',
        (Decimal('0'), Decimal('100')),
    ),
    'canvas_jacket': InsulationGoldenValue(
        'Canvas/Cloth Jacket Emissivity',
        Decimal('0.90'),
        'dimensionless',
        Decimal('10'),
        'ASHRAE',
        (Decimal('0'), Decimal('100')),
    ),
}


# Standard Pipe Insulation Thicknesses (ASTM C585)
STANDARD_THICKNESSES: Dict[str, InsulationGoldenValue] = {
    '1_inch': InsulationGoldenValue(
        '1" Insulation Thickness',
        Decimal('25.4'),
        'mm',
        Decimal('0'),
        'ASTM C585',
        (Decimal('-273'), Decimal('1000')),
    ),
    '1.5_inch': InsulationGoldenValue(
        '1.5" Insulation Thickness',
        Decimal('38.1'),
        'mm',
        Decimal('0'),
        'ASTM C585',
        (Decimal('-273'), Decimal('1000')),
    ),
    '2_inch': InsulationGoldenValue(
        '2" Insulation Thickness',
        Decimal('50.8'),
        'mm',
        Decimal('0'),
        'ASTM C585',
        (Decimal('-273'), Decimal('1000')),
    ),
    '2.5_inch': InsulationGoldenValue(
        '2.5" Insulation Thickness',
        Decimal('63.5'),
        'mm',
        Decimal('0'),
        'ASTM C585',
        (Decimal('-273'), Decimal('1000')),
    ),
    '3_inch': InsulationGoldenValue(
        '3" Insulation Thickness',
        Decimal('76.2'),
        'mm',
        Decimal('0'),
        'ASTM C585',
        (Decimal('-273'), Decimal('1000')),
    ),
    '4_inch': InsulationGoldenValue(
        '4" Insulation Thickness',
        Decimal('101.6'),
        'mm',
        Decimal('0'),
        'ASTM C585',
        (Decimal('-273'), Decimal('1000')),
    ),
}


# =============================================================================
# DETERMINISTIC CALCULATION FUNCTIONS
# =============================================================================


def calculate_flat_surface_r_value(
    thickness_m: Decimal,
    conductivity_w_mk: Decimal,
) -> Decimal:
    """
    Calculate R-value for flat insulation.

    R = thickness / k

    Args:
        thickness_m: Insulation thickness (m)
        conductivity_w_mk: Thermal conductivity (W/m-K)

    Returns:
        R-value (m²-K/W)

    Reference: ASHRAE Fundamentals
    """
    if conductivity_w_mk <= 0:
        raise ValueError('Thermal conductivity must be positive')
    if thickness_m < 0:
        raise ValueError('Thickness cannot be negative')

    r_value = thickness_m / conductivity_w_mk
    return r_value.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_pipe_r_value(
    inner_radius_m: Decimal,
    outer_radius_m: Decimal,
    conductivity_w_mk: Decimal,
) -> Decimal:
    """
    Calculate R-value for cylindrical (pipe) insulation.

    R = ln(r_outer/r_inner) / (2*π*k)

    Args:
        inner_radius_m: Inner radius (m) - pipe OD/2
        outer_radius_m: Outer radius (m) - insulated OD/2
        conductivity_w_mk: Thermal conductivity (W/m-K)

    Returns:
        R-value per unit length (m-K/W)

    Reference: ASTM C680
    """
    if conductivity_w_mk <= 0:
        raise ValueError('Thermal conductivity must be positive')
    if outer_radius_m <= inner_radius_m:
        raise ValueError('Outer radius must exceed inner radius')

    ratio = float(outer_radius_m / inner_radius_m)
    ln_ratio = Decimal(str(math.log(ratio)))
    two_pi = Decimal(str(2 * math.pi))

    r_value = ln_ratio / (two_pi * conductivity_w_mk)
    return r_value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_flat_heat_loss(
    delta_t: Decimal,
    r_total: Decimal,
    area_m2: Decimal,
) -> Decimal:
    """
    Calculate heat loss through flat insulation.

    Q = A * ΔT / R_total

    Args:
        delta_t: Temperature difference (K or °C)
        r_total: Total thermal resistance (m²-K/W)
        area_m2: Surface area (m²)

    Returns:
        Heat loss (W)

    Reference: ASHRAE Fundamentals
    """
    if r_total <= 0:
        raise ValueError('Total R-value must be positive')

    q = area_m2 * delta_t / r_total
    return q.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_pipe_heat_loss(
    t_pipe: Decimal,
    t_ambient: Decimal,
    r_insulation: Decimal,
    r_surface: Decimal,
    length_m: Decimal,
) -> Decimal:
    """
    Calculate heat loss from insulated pipe per unit length.

    Q = (T_pipe - T_ambient) / (R_insulation + R_surface) * L

    Args:
        t_pipe: Pipe surface temperature (°C)
        t_ambient: Ambient temperature (°C)
        r_insulation: Insulation R-value (m-K/W per m length)
        r_surface: Surface film R-value (m-K/W per m length)
        length_m: Pipe length (m)

    Returns:
        Heat loss (W)

    Reference: ASTM C680
    """
    r_total = r_insulation + r_surface
    if r_total <= 0:
        raise ValueError('Total R-value must be positive')

    delta_t = t_pipe - t_ambient
    q = (delta_t / r_total) * length_m
    return q.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_surface_film_coefficient(
    wind_speed_m_s: Decimal,
    emissivity: Decimal,
    surface_temp_c: Decimal,
    ambient_temp_c: Decimal,
) -> Decimal:
    """
    Calculate combined surface film coefficient (natural + forced convection + radiation).

    h_total = h_conv + h_rad

    Simplified model for outdoor surfaces:
    h_conv ≈ 5.7 + 3.8 * V (for V < 5 m/s)
    h_rad ≈ ε * σ * 4 * T_avg³

    Args:
        wind_speed_m_s: Wind speed (m/s)
        emissivity: Surface emissivity (0-1)
        surface_temp_c: Surface temperature (°C)
        ambient_temp_c: Ambient temperature (°C)

    Returns:
        Combined h coefficient (W/m²-K)

    Reference: ASHRAE Fundamentals Chapter 26
    """
    # Convection component (simplified McAdams)
    h_conv = Decimal('5.7') + Decimal('3.8') * wind_speed_m_s

    # Radiation component (linearized)
    t_surface_k = surface_temp_c + Decimal('273.15')
    t_ambient_k = ambient_temp_c + Decimal('273.15')
    t_avg_k = (t_surface_k + t_ambient_k) / Decimal('2')

    # Stefan-Boltzmann constant
    sigma = Decimal('5.67e-8')

    # Linearized radiation coefficient
    h_rad = emissivity * sigma * Decimal('4') * (t_avg_k ** 3)

    h_total = h_conv + h_rad
    return h_total.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_critical_radius(
    conductivity_w_mk: Decimal,
    h_surface_w_m2k: Decimal,
) -> Decimal:
    """
    Calculate critical radius of insulation.

    r_critical = k / h

    Adding insulation below critical radius INCREASES heat loss.

    Args:
        conductivity_w_mk: Insulation thermal conductivity (W/m-K)
        h_surface_w_m2k: Surface heat transfer coefficient (W/m²-K)

    Returns:
        Critical radius (m)

    Reference: Incropera & DeWitt
    """
    if h_surface_w_m2k <= 0:
        raise ValueError('Heat transfer coefficient must be positive')

    r_crit = conductivity_w_mk / h_surface_w_m2k
    return r_crit.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def calculate_economic_thickness(
    conductivity_w_mk: Decimal,
    pipe_od_m: Decimal,
    t_pipe_c: Decimal,
    t_ambient_c: Decimal,
    energy_cost_per_kwh: Decimal,
    insulation_cost_per_m3: Decimal,
    operating_hours_per_year: Decimal,
    payback_years: Decimal,
) -> Decimal:
    """
    Calculate economic thickness of insulation (simplified).

    Balances annual energy cost savings against insulation investment.

    Args:
        conductivity_w_mk: Insulation k-value (W/m-K)
        pipe_od_m: Pipe outer diameter (m)
        t_pipe_c: Pipe temperature (°C)
        t_ambient_c: Ambient temperature (°C)
        energy_cost_per_kwh: Energy cost ($/kWh)
        insulation_cost_per_m3: Insulation cost ($/m³)
        operating_hours_per_year: Annual operating hours
        payback_years: Target payback period (years)

    Returns:
        Economic thickness (m)

    Reference: 3E Plus (NAIMA)
    """
    # This is a simplified calculation; real economic thickness
    # requires iterative optimization

    delta_t = t_pipe_c - t_ambient_c
    if delta_t <= 0:
        return Decimal('0')

    # Estimate based on energy savings vs. insulation cost ratio
    # Actual 3E Plus uses much more sophisticated optimization
    energy_factor = (
        energy_cost_per_kwh * operating_hours_per_year / Decimal('1000')
    )
    cost_ratio = energy_factor / insulation_cost_per_m3

    # Simplified economic thickness estimate
    base_thickness = (conductivity_w_mk * delta_t * cost_ratio * payback_years) ** Decimal('0.5')

    # Practical limits
    min_thickness = Decimal('0.025')  # 1 inch minimum
    max_thickness = Decimal('0.150')  # 6 inch maximum

    thickness = max(min_thickness, min(base_thickness, max_thickness))
    return thickness.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_condensation_prevention_thickness(
    pipe_temp_c: Decimal,
    ambient_temp_c: Decimal,
    relative_humidity_pct: Decimal,
    conductivity_w_mk: Decimal,
    h_surface: Decimal,
    pipe_od_m: Decimal,
) -> Decimal:
    """
    Calculate minimum insulation thickness to prevent condensation.

    Surface temperature must exceed dew point.

    Args:
        pipe_temp_c: Pipe temperature (°C) - typically cold for condensation risk
        ambient_temp_c: Ambient temperature (°C)
        relative_humidity_pct: Relative humidity (%)
        conductivity_w_mk: Insulation k-value (W/m-K)
        h_surface: Surface heat transfer coefficient (W/m²-K)
        pipe_od_m: Pipe outer diameter (m)

    Returns:
        Minimum thickness (m)

    Reference: ASHRAE Fundamentals
    """
    # Calculate dew point using Magnus formula approximation
    rh = float(relative_humidity_pct) / 100.0
    t_amb = float(ambient_temp_c)

    # Magnus formula constants
    a = 17.27
    b = 237.7

    gamma = a * t_amb / (b + t_amb) + math.log(rh)
    t_dew = Decimal(str(b * gamma / (a - gamma)))

    if pipe_temp_c >= t_dew:
        return Decimal('0')  # No insulation needed

    # Required surface temperature (slightly above dew point)
    t_surface_required = t_dew + Decimal('1')

    # Calculate required R-value
    delta_t_total = ambient_temp_c - pipe_temp_c
    delta_t_surface = ambient_temp_c - t_surface_required

    if delta_t_surface <= 0:
        return Decimal('0.150')  # Maximum practical thickness

    # R_surface = 1/(h*A) for cylinder, simplified
    r_surface = Decimal('1') / h_surface

    # Required insulation R
    r_ratio = delta_t_total / delta_t_surface
    r_insulation_required = r_surface * (r_ratio - Decimal('1'))

    # Convert R to thickness (simplified for flat)
    thickness = r_insulation_required * conductivity_w_mk

    # Practical limits
    thickness = max(Decimal('0.013'), min(thickness, Decimal('0.150')))

    return thickness.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def calculate_heat_loss_bare_pipe(
    pipe_od_m: Decimal,
    t_pipe_c: Decimal,
    t_ambient_c: Decimal,
    h_surface: Decimal,
    length_m: Decimal,
) -> Decimal:
    """
    Calculate heat loss from bare (uninsulated) pipe.

    Q = h * A * ΔT = h * π * D * L * ΔT

    Args:
        pipe_od_m: Pipe outer diameter (m)
        t_pipe_c: Pipe surface temperature (°C)
        t_ambient_c: Ambient temperature (°C)
        h_surface: Surface heat transfer coefficient (W/m²-K)
        length_m: Pipe length (m)

    Returns:
        Heat loss (W)

    Reference: ASTM C680
    """
    pi = Decimal(str(math.pi))
    area = pi * pipe_od_m * length_m
    delta_t = t_pipe_c - t_ambient_c

    q = h_surface * area * delta_t
    return q.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_energy_savings(
    q_bare_w: Decimal,
    q_insulated_w: Decimal,
    operating_hours: Decimal,
) -> Decimal:
    """
    Calculate annual energy savings from insulation.

    Savings = (Q_bare - Q_insulated) * hours / 1000

    Args:
        q_bare_w: Bare pipe heat loss (W)
        q_insulated_w: Insulated pipe heat loss (W)
        operating_hours: Annual operating hours

    Returns:
        Energy savings (kWh/year)

    Reference: 3E Plus
    """
    savings_w = q_bare_w - q_insulated_w
    savings_kwh = savings_w * operating_hours / Decimal('1000')
    return savings_kwh.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_k_at_temperature(
    k_ref: Decimal,
    t_ref_c: Decimal,
    t_mean_c: Decimal,
    coefficient: Decimal = Decimal('0.0002'),
) -> Decimal:
    """
    Calculate thermal conductivity at different temperature.

    k(T) = k_ref * [1 + coefficient * (T - T_ref)]

    Args:
        k_ref: Reference k-value at T_ref (W/m-K)
        t_ref_c: Reference temperature (°C)
        t_mean_c: Mean operating temperature (°C)
        coefficient: Temperature coefficient (1/K)

    Returns:
        k-value at T_mean (W/m-K)

    Reference: ASTM C680
    """
    k = k_ref * (Decimal('1') + coefficient * (t_mean_c - t_ref_c))
    return k.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================


def generate_provenance_hash(
    calculation_name: str,
    inputs: Dict[str, str],
    output: str,
    reference: str,
) -> str:
    """Generate SHA-256 hash for calculation provenance."""
    provenance_data = {
        'calculation': calculation_name,
        'inputs': inputs,
        'output': output,
        'reference': reference,
        'version': '1.0.0',
    }
    json_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# GOLDEN VALUE TESTS
# =============================================================================


class TestThermalConductivity:
    """Test thermal conductivity reference values."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'material,expected_k',
        [
            ('mineral_wool_50c', Decimal('0.038')),
            ('mineral_wool_100c', Decimal('0.045')),
            ('calcium_silicate_50c', Decimal('0.055')),
            ('fiberglass_24c', Decimal('0.033')),
            ('pir_foam_24c', Decimal('0.023')),
            ('aerogel_24c', Decimal('0.015')),
        ],
    )
    def test_conductivity_values(self, material: str, expected_k: Decimal) -> None:
        """Verify thermal conductivity against ASTM references."""
        golden = THERMAL_CONDUCTIVITY[material]
        assert golden.value == expected_k, (
            f'Expected k={expected_k} for {material}, got {golden.value}'
        )

    @pytest.mark.golden
    def test_conductivity_temperature_ordering(self) -> None:
        """Verify conductivity increases with temperature."""
        k_50 = THERMAL_CONDUCTIVITY['mineral_wool_50c'].value
        k_100 = THERMAL_CONDUCTIVITY['mineral_wool_100c'].value
        k_200 = THERMAL_CONDUCTIVITY['mineral_wool_200c'].value

        assert k_50 < k_100 < k_200, (
            'Mineral wool k must increase with temperature'
        )


class TestRValueCalculations:
    """Test R-value calculations."""

    @pytest.mark.golden
    def test_flat_r_value_2inch_mineral_wool(self) -> None:
        """R-value for 2" mineral wool at 50°C."""
        # 2" = 50.8 mm = 0.0508 m
        # k = 0.038 W/m-K
        # R = 0.0508 / 0.038 = 1.337 m²-K/W

        r = calculate_flat_surface_r_value(
            Decimal('0.0508'),
            Decimal('0.038')
        )

        expected = Decimal('1.337')
        tolerance = Decimal('0.01')

        assert abs(r - expected) <= tolerance, (
            f'Expected R~{expected}, got {r}'
        )

    @pytest.mark.golden
    def test_flat_r_value_1inch_fiberglass(self) -> None:
        """R-value for 1" fiberglass at 24°C."""
        # 1" = 25.4 mm = 0.0254 m
        # k = 0.033 W/m-K
        # R = 0.0254 / 0.033 = 0.770 m²-K/W

        r = calculate_flat_surface_r_value(
            Decimal('0.0254'),
            Decimal('0.033')
        )

        expected = Decimal('0.770')
        tolerance = Decimal('0.01')

        assert abs(r - expected) <= tolerance, (
            f'Expected R~{expected}, got {r}'
        )

    @pytest.mark.golden
    def test_pipe_r_value_calculation(self) -> None:
        """R-value for pipe insulation (cylindrical)."""
        # 4" NPS pipe: OD = 114.3 mm, r_inner = 0.057 m
        # 2" insulation: r_outer = 0.057 + 0.0508 = 0.1078 m
        # k = 0.040 W/m-K
        # R = ln(0.1078/0.057) / (2*π*0.040)

        r = calculate_pipe_r_value(
            Decimal('0.057'),
            Decimal('0.1078'),
            Decimal('0.040')
        )

        # Expected ~2.54 m-K/W per meter length
        assert Decimal('2.4') < r < Decimal('2.7'), f'Pipe R-value {r} out of range'


class TestHeatLossCalculations:
    """Test heat loss calculations."""

    @pytest.mark.golden
    def test_flat_heat_loss(self) -> None:
        """Heat loss through flat surface."""
        # ΔT = 100°C, R = 2.0 m²-K/W, A = 10 m²
        # Q = 10 * 100 / 2.0 = 500 W

        q = calculate_flat_heat_loss(
            Decimal('100'),
            Decimal('2.0'),
            Decimal('10')
        )

        assert q == Decimal('500.0'), f'Expected Q=500.0 W, got {q}'

    @pytest.mark.golden
    def test_bare_pipe_heat_loss(self) -> None:
        """Heat loss from bare 4" NPS steam pipe."""
        # D = 0.1143 m, T_pipe = 150°C, T_amb = 25°C, h = 15 W/m²-K, L = 10 m
        # A = π * 0.1143 * 10 = 3.59 m²
        # Q = 15 * 3.59 * 125 = 6731 W

        q = calculate_heat_loss_bare_pipe(
            Decimal('0.1143'),
            Decimal('150'),
            Decimal('25'),
            Decimal('15'),
            Decimal('10')
        )

        expected = Decimal('6731')
        tolerance = Decimal('100')

        assert abs(q - expected) <= tolerance, (
            f'Expected Q~{expected} W, got {q}'
        )


class TestSurfaceFilmCoefficient:
    """Test surface film coefficient calculations."""

    @pytest.mark.golden
    def test_still_air_coefficient(self) -> None:
        """Combined h in still air."""
        # Wind = 0, emissivity = 0.9, T_surf = 50°C, T_amb = 25°C
        # h_conv ≈ 5.7
        # h_rad ≈ 0.9 * 5.67e-8 * 4 * 310³ ≈ 6.1
        # h_total ≈ 11.8

        h = calculate_surface_film_coefficient(
            Decimal('0'),
            Decimal('0.9'),
            Decimal('50'),
            Decimal('25')
        )

        assert Decimal('10') < h < Decimal('15'), f'Still air h={h} out of range'

    @pytest.mark.golden
    def test_windy_conditions_coefficient(self) -> None:
        """Combined h at 4 m/s wind."""
        # Wind = 4 m/s
        # h_conv ≈ 5.7 + 3.8*4 = 20.9
        # Plus radiation ≈ 6
        # h_total ≈ 27

        h = calculate_surface_film_coefficient(
            Decimal('4'),
            Decimal('0.9'),
            Decimal('50'),
            Decimal('25')
        )

        assert Decimal('20') < h < Decimal('35'), f'Windy h={h} out of range'


class TestCriticalRadius:
    """Test critical radius calculations."""

    @pytest.mark.golden
    def test_critical_radius_typical(self) -> None:
        """Critical radius for mineral wool with h=10."""
        # k = 0.040 W/m-K, h = 10 W/m²-K
        # r_crit = 0.040 / 10 = 0.004 m = 4 mm

        r_crit = calculate_critical_radius(
            Decimal('0.040'),
            Decimal('10')
        )

        assert r_crit == Decimal('0.0040'), f'Expected r_crit=0.0040, got {r_crit}'

    @pytest.mark.golden
    def test_critical_radius_small_pipe(self) -> None:
        """Critical radius vs. 1/4" pipe (OD 13.7mm = 6.85mm radius)."""
        # For most insulations with k~0.04 and h~10
        # r_crit ≈ 4 mm < 6.85 mm
        # Therefore insulation will always reduce heat loss

        r_crit = calculate_critical_radius(
            Decimal('0.040'),
            Decimal('10')
        )

        pipe_radius = Decimal('0.00685')
        assert r_crit < pipe_radius, (
            'Critical radius should be less than small pipe radius'
        )


class TestEnergySavings:
    """Test energy savings calculations."""

    @pytest.mark.golden
    def test_annual_savings(self) -> None:
        """Annual energy savings from insulation."""
        # Q_bare = 5000 W, Q_insulated = 500 W
        # Hours = 8760 h/year
        # Savings = (5000 - 500) * 8760 / 1000 = 39420 kWh/year

        savings = calculate_energy_savings(
            Decimal('5000'),
            Decimal('500'),
            Decimal('8760')
        )

        expected = Decimal('39420')
        assert savings == expected, f'Expected savings={expected}, got {savings}'

    @pytest.mark.golden
    def test_savings_efficiency_ratio(self) -> None:
        """Verify typical 80-95% heat loss reduction with insulation."""
        q_bare = Decimal('5000')
        q_insulated = Decimal('500')

        reduction_pct = (q_bare - q_insulated) / q_bare * Decimal('100')

        assert reduction_pct == Decimal('90'), (
            f'Expected 90% reduction, got {reduction_pct}%'
        )


class TestConductivityTemperatureCorrection:
    """Test k-value temperature correction."""

    @pytest.mark.golden
    def test_k_correction_higher_temp(self) -> None:
        """k increases at higher temperatures."""
        k_ref = Decimal('0.040')
        t_ref = Decimal('50')
        t_mean = Decimal('150')
        coefficient = Decimal('0.0002')

        # k = 0.040 * (1 + 0.0002 * 100) = 0.040 * 1.02 = 0.0408

        k = calculate_k_at_temperature(k_ref, t_ref, t_mean, coefficient)

        expected = Decimal('0.0408')
        assert k == expected, f'Expected k={expected}, got {k}'

    @pytest.mark.golden
    def test_k_correction_same_temp(self) -> None:
        """No correction at reference temperature."""
        k_ref = Decimal('0.040')
        t_ref = Decimal('50')

        k = calculate_k_at_temperature(k_ref, t_ref, t_ref)

        assert k == k_ref, f'k at T_ref should equal k_ref'


class TestStandardThicknesses:
    """Test ASTM C585 standard thicknesses."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'thickness_name,expected_mm',
        [
            ('1_inch', Decimal('25.4')),
            ('1.5_inch', Decimal('38.1')),
            ('2_inch', Decimal('50.8')),
            ('3_inch', Decimal('76.2')),
            ('4_inch', Decimal('101.6')),
        ],
    )
    def test_standard_thickness_values(
        self, thickness_name: str, expected_mm: Decimal
    ) -> None:
        """Verify standard insulation thicknesses."""
        golden = STANDARD_THICKNESSES[thickness_name]
        assert golden.value == expected_mm, (
            f'Expected {expected_mm} mm for {thickness_name}'
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.golden
    def test_zero_conductivity_rejected(self) -> None:
        """Reject zero thermal conductivity."""
        with pytest.raises(ValueError, match='positive'):
            calculate_flat_surface_r_value(Decimal('0.05'), Decimal('0'))

    @pytest.mark.golden
    def test_negative_thickness_rejected(self) -> None:
        """Reject negative thickness."""
        with pytest.raises(ValueError, match='negative'):
            calculate_flat_surface_r_value(Decimal('-0.05'), Decimal('0.04'))

    @pytest.mark.golden
    def test_invalid_radii_rejected(self) -> None:
        """Reject outer radius <= inner radius."""
        with pytest.raises(ValueError, match='Outer radius'):
            calculate_pipe_r_value(Decimal('0.1'), Decimal('0.05'), Decimal('0.04'))


class TestDeterminism:
    """Verify calculation determinism for regulatory compliance."""

    @pytest.mark.golden
    def test_r_value_determinism(self) -> None:
        """Verify R-value calculation is deterministic."""
        results = [
            calculate_flat_surface_r_value(Decimal('0.0508'), Decimal('0.040'))
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'R-value calculation must be deterministic'

    @pytest.mark.golden
    def test_heat_loss_determinism(self) -> None:
        """Verify heat loss calculation is deterministic."""
        results = [
            calculate_flat_heat_loss(
                Decimal('75'),
                Decimal('1.5'),
                Decimal('20')
            )
            for _ in range(100)
        ]

        assert len(set(results)) == 1, 'Heat loss calculation must be deterministic'

    @pytest.mark.golden
    def test_provenance_hash_determinism(self) -> None:
        """Verify provenance hashes are deterministic."""
        hashes = [
            generate_provenance_hash(
                'r_value',
                {'thickness': '0.0508', 'k': '0.040'},
                '1.270',
                'ASTM C680',
            )
            for _ in range(100)
        ]

        assert len(set(hashes)) == 1, 'Provenance hash must be deterministic'


class TestEmissivityValues:
    """Test surface emissivity reference values."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        'surface,min_eps,max_eps',
        [
            ('aluminum_bright', Decimal('0.02'), Decimal('0.10')),
            ('stainless_steel', Decimal('0.20'), Decimal('0.80')),
            ('painted_surface', Decimal('0.80'), Decimal('0.95')),
        ],
    )
    def test_emissivity_ranges(
        self, surface: str, min_eps: Decimal, max_eps: Decimal
    ) -> None:
        """Verify emissivity values are within expected ranges."""
        golden = SURFACE_EMISSIVITY[surface]

        assert min_eps <= golden.value <= max_eps, (
            f'{surface} emissivity {golden.value} outside [{min_eps}, {max_eps}]'
        )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_golden_values() -> Dict[str, List[Dict]]:
    """Export all golden values for documentation."""
    export_data = {
        'thermal_conductivity': [],
        'surface_emissivity': [],
        'standard_thicknesses': [],
        'metadata': {
            'version': '1.0.0',
            'references': ['ASTM C680', 'ASTM C585', 'ASHRAE Fundamentals'],
            'agent': 'GL-015_InsulScan',
        },
    }

    for material, golden in THERMAL_CONDUCTIVITY.items():
        export_data['thermal_conductivity'].append(
            {
                'material': material,
                'description': golden.description,
                'k': str(golden.value),
                'unit': golden.unit,
                'source': golden.source,
                'temp_range': [str(golden.temp_range_c[0]), str(golden.temp_range_c[1])],
            }
        )

    for surface, golden in SURFACE_EMISSIVITY.items():
        export_data['surface_emissivity'].append(
            {
                'surface': surface,
                'emissivity': str(golden.value),
                'source': golden.source,
            }
        )

    return export_data


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
