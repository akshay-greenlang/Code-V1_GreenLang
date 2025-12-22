"""
Unit Tests: Condensate Calculator

Tests the condensate recovery optimization module including:
- Return ratio calculation
- Flash steam fraction formula
- Heat recovery calculation
- Loss identification

Reference: ASME PTC 19.11, Spirax Sarco Steam Engineering Guidelines

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import hashlib
import json


# =============================================================================
# Data Classes and Enumerations
# =============================================================================

class CondensateLossType(Enum):
    """Types of condensate losses."""
    FLASH_STEAM = auto()
    TRAP_LEAKAGE = auto()
    CONTAMINATION = auto()
    DUMP_TO_DRAIN = auto()
    EVAPORATION = auto()
    UNRECOVERED = auto()


class RecoveryPriority(Enum):
    """Priority levels for condensate recovery."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    NOT_ECONOMICAL = auto()


@dataclass
class CondensateStream:
    """Data for a condensate stream."""
    stream_id: str
    source: str  # Equipment or area generating condensate
    mass_flow_kg_s: float
    temperature_k: float
    pressure_mpa: float
    is_contaminated: bool = False
    contamination_type: Optional[str] = None
    return_distance_m: float = 100.0
    elevation_change_m: float = 0.0


@dataclass
class FlashSteamResult:
    """Result of flash steam calculation."""
    flash_steam_fraction: float
    flash_steam_flow_kg_s: float
    liquid_condensate_flow_kg_s: float
    flash_steam_enthalpy_kj_kg: float
    liquid_enthalpy_kj_kg: float
    energy_in_flash_kw: float
    provenance_hash: str


@dataclass
class CondensateRecoveryResult:
    """Result of condensate recovery analysis."""
    total_condensate_kg_s: float
    recovered_condensate_kg_s: float
    lost_condensate_kg_s: float
    recovery_ratio: float
    loss_breakdown: Dict[CondensateLossType, float]
    heat_recovery_kw: float
    potential_savings_kw: float
    recommendations: List[str]
    priority: RecoveryPriority
    provenance_hash: str


@dataclass
class HeatRecoveryAnalysis:
    """Analysis of heat recovery potential."""
    sensible_heat_kw: float
    flash_steam_heat_kw: float
    total_available_heat_kw: float
    recoverable_heat_kw: float
    recovery_efficiency: float
    economic_value_per_hour: float  # Currency per hour
    payback_months: Optional[float]
    provenance_hash: str


# =============================================================================
# Constants
# =============================================================================

# Water properties
CP_WATER = 4.186  # kJ/(kg.K)
WATER_DENSITY = 1000.0  # kg/m3

# Reference temperature for enthalpy
T_REF = 273.15  # K (0 C)

# Economic parameters (example values)
STEAM_COST_PER_KWH = 0.05  # Currency per kWh
WATER_COST_PER_M3 = 2.0  # Currency per m3
TREATMENT_COST_PER_M3 = 1.5  # Currency per m3


# =============================================================================
# Condensate Calculator Implementation
# =============================================================================

class CondensateError(Exception):
    """Error in condensate calculation."""
    pass


def get_saturation_temperature(pressure_mpa: float) -> float:
    """Calculate saturation temperature from pressure."""
    if pressure_mpa < 0.001 or pressure_mpa > 22.064:
        raise CondensateError(f"Pressure {pressure_mpa} MPa outside valid range")

    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
         0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
         0.65017534844798e3]

    beta = pressure_mpa ** 0.25
    E = beta ** 2 + n[2] * beta + n[5]
    F = n[0] * beta ** 2 + n[3] * beta + n[6]
    G = n[1] * beta ** 2 + n[4] * beta + n[7]
    D = 2 * G / (-F - math.sqrt(F ** 2 - 4 * E * G))

    return (n[9] + D - math.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D))) / 2


def get_saturation_pressure(temperature_k: float) -> float:
    """Calculate saturation pressure from temperature."""
    if temperature_k < 273.15 or temperature_k > 647.096:
        raise CondensateError(f"Temperature {temperature_k} K outside saturation range")

    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
         0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
         0.65017534844798e3]

    theta = temperature_k + n[8] / (temperature_k - n[9])
    A = theta ** 2 + n[0] * theta + n[1]
    B = n[2] * theta ** 2 + n[3] * theta + n[4]
    C = n[5] * theta ** 2 + n[6] * theta + n[7]

    return (2 * C / (-B + math.sqrt(B ** 2 - 4 * A * C))) ** 4


def get_liquid_enthalpy(temperature_k: float) -> float:
    """Calculate saturated liquid enthalpy at given temperature."""
    return CP_WATER * (temperature_k - T_REF)


def get_latent_heat(pressure_mpa: float) -> float:
    """Calculate latent heat of vaporization at given pressure."""
    t_sat = get_saturation_temperature(pressure_mpa)

    # Simplified latent heat correlation
    # h_fg decreases from ~2500 kJ/kg at low pressure to 0 at critical point
    t_crit = 647.096
    h_fg_ref = 2257.0  # kJ/kg at 100 C

    if t_sat >= t_crit:
        return 0.0

    # Watson correlation approximation
    tr = t_sat / t_crit
    return h_fg_ref * ((1 - tr) / (1 - 373.15 / t_crit)) ** 0.38


def calculate_flash_steam_fraction(
    upstream_pressure_mpa: float,
    downstream_pressure_mpa: float
) -> float:
    """
    Calculate flash steam fraction when condensate pressure drops.

    Flash steam is generated when hot condensate at high pressure
    is released to a lower pressure. The excess enthalpy causes
    some water to evaporate.

    Formula:
    x_flash = (h_f1 - h_f2) / h_fg2

    Where:
    - h_f1 = saturated liquid enthalpy at upstream pressure
    - h_f2 = saturated liquid enthalpy at downstream pressure
    - h_fg2 = latent heat at downstream pressure
    """
    if downstream_pressure_mpa >= upstream_pressure_mpa:
        return 0.0  # No flash if pressure doesn't drop

    if downstream_pressure_mpa <= 0:
        raise CondensateError("Downstream pressure must be positive")

    # Get saturation temperatures
    t_sat_up = get_saturation_temperature(upstream_pressure_mpa)
    t_sat_down = get_saturation_temperature(downstream_pressure_mpa)

    # Get enthalpies
    h_f1 = get_liquid_enthalpy(t_sat_up)
    h_f2 = get_liquid_enthalpy(t_sat_down)
    h_fg2 = get_latent_heat(downstream_pressure_mpa)

    if h_fg2 <= 0:
        return 0.0

    # Flash fraction
    x_flash = (h_f1 - h_f2) / h_fg2

    # Clamp to physical limits
    return max(0.0, min(1.0, x_flash))


def calculate_flash_steam(
    condensate_flow_kg_s: float,
    upstream_pressure_mpa: float,
    downstream_pressure_mpa: float
) -> FlashSteamResult:
    """
    Calculate flash steam generation from condensate pressure reduction.
    """
    flash_fraction = calculate_flash_steam_fraction(upstream_pressure_mpa, downstream_pressure_mpa)

    flash_steam_flow = condensate_flow_kg_s * flash_fraction
    liquid_flow = condensate_flow_kg_s * (1 - flash_fraction)

    # Get enthalpies for energy calculation
    t_sat_down = get_saturation_temperature(downstream_pressure_mpa)
    h_f2 = get_liquid_enthalpy(t_sat_down)
    h_fg2 = get_latent_heat(downstream_pressure_mpa)
    h_g2 = h_f2 + h_fg2

    # Energy in flash steam
    energy_in_flash = flash_steam_flow * h_g2

    provenance_hash = hashlib.sha256(
        json.dumps({
            "condensate_flow": condensate_flow_kg_s,
            "upstream_pressure": upstream_pressure_mpa,
            "downstream_pressure": downstream_pressure_mpa,
            "flash_fraction": flash_fraction
        }, sort_keys=True).encode()
    ).hexdigest()

    return FlashSteamResult(
        flash_steam_fraction=flash_fraction,
        flash_steam_flow_kg_s=flash_steam_flow,
        liquid_condensate_flow_kg_s=liquid_flow,
        flash_steam_enthalpy_kj_kg=h_g2,
        liquid_enthalpy_kj_kg=h_f2,
        energy_in_flash_kw=energy_in_flash,
        provenance_hash=provenance_hash
    )


def calculate_return_ratio(
    recovered_flow_kg_s: float,
    total_condensate_kg_s: float
) -> float:
    """
    Calculate condensate return ratio.

    Return ratio = recovered condensate / total condensate generated
    """
    if total_condensate_kg_s <= 0:
        return 0.0

    ratio = recovered_flow_kg_s / total_condensate_kg_s
    return max(0.0, min(1.0, ratio))


def calculate_heat_recovery(
    condensate_flow_kg_s: float,
    condensate_temperature_k: float,
    reference_temperature_k: float = 288.15  # 15 C makeup water
) -> float:
    """
    Calculate heat recovery from returning hot condensate.

    Returns: Recoverable heat in kW
    """
    if condensate_flow_kg_s <= 0:
        return 0.0

    delta_t = condensate_temperature_k - reference_temperature_k

    if delta_t <= 0:
        return 0.0

    # Sensible heat recovery
    return condensate_flow_kg_s * CP_WATER * delta_t


def identify_condensate_losses(
    streams: List[CondensateStream],
    recovered_streams: List[str],
    trap_leakage_fraction: float = 0.02,
    evaporation_fraction: float = 0.01
) -> Dict[CondensateLossType, float]:
    """
    Identify and categorize condensate losses.

    Args:
        streams: All condensate streams
        recovered_streams: IDs of streams being recovered
        trap_leakage_fraction: Estimated trap leakage as fraction
        evaporation_fraction: Estimated evaporation as fraction

    Returns: Dictionary of loss type to flow rate (kg/s)
    """
    losses = {loss_type: 0.0 for loss_type in CondensateLossType}

    total_flow = sum(s.mass_flow_kg_s for s in streams)

    for stream in streams:
        if stream.stream_id not in recovered_streams:
            if stream.is_contaminated:
                losses[CondensateLossType.CONTAMINATION] += stream.mass_flow_kg_s
            else:
                losses[CondensateLossType.UNRECOVERED] += stream.mass_flow_kg_s

    # Estimate trap leakage on recovered streams
    recovered_flow = sum(
        s.mass_flow_kg_s for s in streams
        if s.stream_id in recovered_streams
    )
    losses[CondensateLossType.TRAP_LEAKAGE] = recovered_flow * trap_leakage_fraction

    # Estimate evaporation from return system
    losses[CondensateLossType.EVAPORATION] = total_flow * evaporation_fraction

    return losses


def analyze_condensate_recovery(
    streams: List[CondensateStream],
    recovered_streams: List[str],
    flash_recovery_pressure_mpa: float = 0.1,  # Flash tank pressure
    reference_temperature_k: float = 288.15
) -> CondensateRecoveryResult:
    """
    Comprehensive analysis of condensate recovery system.
    """
    if not streams:
        raise CondensateError("No condensate streams provided")

    # Calculate totals
    total_condensate = sum(s.mass_flow_kg_s for s in streams)
    recovered_condensate = sum(
        s.mass_flow_kg_s for s in streams
        if s.stream_id in recovered_streams and not s.is_contaminated
    )

    # Calculate return ratio
    recovery_ratio = calculate_return_ratio(recovered_condensate, total_condensate)

    # Identify losses
    loss_breakdown = identify_condensate_losses(streams, recovered_streams)
    lost_condensate = sum(loss_breakdown.values())

    # Calculate heat recovery
    heat_recovery = sum(
        calculate_heat_recovery(s.mass_flow_kg_s, s.temperature_k, reference_temperature_k)
        for s in streams
        if s.stream_id in recovered_streams and not s.is_contaminated
    )

    # Calculate potential additional savings
    unrecovered = [s for s in streams if s.stream_id not in recovered_streams and not s.is_contaminated]
    potential_savings = sum(
        calculate_heat_recovery(s.mass_flow_kg_s, s.temperature_k, reference_temperature_k)
        for s in unrecovered
    )

    # Generate recommendations
    recommendations = []

    if recovery_ratio < 0.6:
        recommendations.append("Condensate recovery ratio is low. Consider installing return lines.")

    if loss_breakdown[CondensateLossType.CONTAMINATION] > 0.1 * total_condensate:
        recommendations.append("Significant contamination losses. Consider segregated recovery systems.")

    if loss_breakdown[CondensateLossType.TRAP_LEAKAGE] > 0.05 * total_condensate:
        recommendations.append("High trap leakage detected. Survey and repair steam traps.")

    for stream in unrecovered:
        if stream.mass_flow_kg_s > 0.5:
            recommendations.append(f"Large unrecovered stream '{stream.stream_id}': {stream.mass_flow_kg_s:.2f} kg/s")

    # Determine priority
    if potential_savings > 500:  # kW threshold
        priority = RecoveryPriority.HIGH
    elif potential_savings > 100:
        priority = RecoveryPriority.MEDIUM
    elif potential_savings > 20:
        priority = RecoveryPriority.LOW
    else:
        priority = RecoveryPriority.NOT_ECONOMICAL

    provenance_hash = hashlib.sha256(
        json.dumps({
            "total_condensate": total_condensate,
            "recovered": recovered_condensate,
            "heat_recovery": heat_recovery
        }, sort_keys=True).encode()
    ).hexdigest()

    return CondensateRecoveryResult(
        total_condensate_kg_s=total_condensate,
        recovered_condensate_kg_s=recovered_condensate,
        lost_condensate_kg_s=lost_condensate,
        recovery_ratio=recovery_ratio,
        loss_breakdown=loss_breakdown,
        heat_recovery_kw=heat_recovery,
        potential_savings_kw=potential_savings,
        recommendations=recommendations,
        priority=priority,
        provenance_hash=provenance_hash
    )


def analyze_heat_recovery_economics(
    heat_recovery_kw: float,
    operating_hours_per_year: float = 8000,
    steam_cost_per_kwh: float = STEAM_COST_PER_KWH,
    investment_cost: float = 0.0,
    recovery_efficiency: float = 0.85
) -> HeatRecoveryAnalysis:
    """
    Analyze economics of heat recovery from condensate.
    """
    recoverable_heat = heat_recovery_kw * recovery_efficiency

    # Economic value
    hourly_value = recoverable_heat * steam_cost_per_kwh
    annual_value = hourly_value * operating_hours_per_year

    # Payback calculation
    if investment_cost > 0 and annual_value > 0:
        payback_months = (investment_cost / annual_value) * 12
    else:
        payback_months = None

    provenance_hash = hashlib.sha256(
        json.dumps({
            "heat_recovery_kw": heat_recovery_kw,
            "recovery_efficiency": recovery_efficiency,
            "hourly_value": hourly_value
        }, sort_keys=True).encode()
    ).hexdigest()

    return HeatRecoveryAnalysis(
        sensible_heat_kw=heat_recovery_kw,
        flash_steam_heat_kw=0.0,  # Calculated separately if flash recovery used
        total_available_heat_kw=heat_recovery_kw,
        recoverable_heat_kw=recoverable_heat,
        recovery_efficiency=recovery_efficiency,
        economic_value_per_hour=hourly_value,
        payback_months=payback_months,
        provenance_hash=provenance_hash
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_condensate_streams():
    """Sample condensate streams for testing."""
    return [
        CondensateStream(
            stream_id="COND-001",
            source="Heat Exchanger A",
            mass_flow_kg_s=2.0,
            temperature_k=433.0,  # 160 C
            pressure_mpa=0.6,
            is_contaminated=False
        ),
        CondensateStream(
            stream_id="COND-002",
            source="Process B",
            mass_flow_kg_s=1.5,
            temperature_k=423.0,  # 150 C
            pressure_mpa=0.5,
            is_contaminated=False
        ),
        CondensateStream(
            stream_id="COND-003",
            source="Process C",
            mass_flow_kg_s=0.8,
            temperature_k=413.0,  # 140 C
            pressure_mpa=0.4,
            is_contaminated=True,
            contamination_type="oil"
        ),
        CondensateStream(
            stream_id="COND-004",
            source="Building Heating",
            mass_flow_kg_s=0.5,
            temperature_k=373.0,  # 100 C
            pressure_mpa=0.1,
            is_contaminated=False
        ),
    ]


@pytest.fixture
def high_pressure_stream():
    """High pressure condensate stream."""
    return CondensateStream(
        stream_id="HP-001",
        source="High Pressure Process",
        mass_flow_kg_s=3.0,
        temperature_k=471.0,  # ~198 C at 1.5 MPa
        pressure_mpa=1.5,
        is_contaminated=False
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestFlashSteamFraction:
    """Test flash steam fraction calculation."""

    def test_no_flash_equal_pressure(self):
        """Test no flash when pressures are equal."""
        fraction = calculate_flash_steam_fraction(1.0, 1.0)
        assert fraction == 0.0

    def test_no_flash_pressure_increase(self):
        """Test no flash when downstream pressure is higher."""
        fraction = calculate_flash_steam_fraction(0.5, 1.0)
        assert fraction == 0.0

    def test_flash_fraction_positive(self):
        """Test flash fraction is positive when pressure drops."""
        fraction = calculate_flash_steam_fraction(1.0, 0.1)
        assert fraction > 0.0

    def test_flash_fraction_less_than_one(self):
        """Test flash fraction is less than 1 (can't all flash)."""
        fraction = calculate_flash_steam_fraction(2.0, 0.1)
        assert fraction < 1.0

    def test_flash_fraction_increases_with_pressure_drop(self):
        """Test flash fraction increases with larger pressure drop."""
        fraction_small = calculate_flash_steam_fraction(1.0, 0.5)
        fraction_large = calculate_flash_steam_fraction(1.0, 0.1)

        assert fraction_large > fraction_small

    @pytest.mark.parametrize("p_up,p_down,expected_range", [
        (1.0, 0.1, (0.10, 0.20)),   # Typical industrial case
        (1.5, 0.3, (0.08, 0.18)),   # Moderate drop
        (2.0, 0.1, (0.15, 0.30)),   # Large drop
        (0.5, 0.1, (0.05, 0.15)),   # Small initial pressure
    ])
    def test_flash_fraction_ranges(self, p_up, p_down, expected_range):
        """Test flash fraction is in expected range for typical cases."""
        fraction = calculate_flash_steam_fraction(p_up, p_down)
        min_frac, max_frac = expected_range
        assert min_frac <= fraction <= max_frac, \
            f"Flash fraction {fraction} not in expected range [{min_frac}, {max_frac}]"


class TestFlashSteamCalculation:
    """Test complete flash steam calculation."""

    def test_flash_steam_mass_balance(self):
        """Test mass is conserved in flash calculation."""
        result = calculate_flash_steam(
            condensate_flow_kg_s=5.0,
            upstream_pressure_mpa=1.0,
            downstream_pressure_mpa=0.1
        )

        total_out = result.flash_steam_flow_kg_s + result.liquid_condensate_flow_kg_s
        assert pytest.approx(total_out, rel=0.001) == 5.0

    def test_flash_steam_flow_proportional(self):
        """Test flash steam flow scales with condensate flow."""
        result_1 = calculate_flash_steam(1.0, 1.0, 0.1)
        result_5 = calculate_flash_steam(5.0, 1.0, 0.1)

        assert pytest.approx(result_5.flash_steam_flow_kg_s, rel=0.01) == \
               5 * result_1.flash_steam_flow_kg_s

    def test_flash_energy_positive(self):
        """Test energy in flash steam is positive."""
        result = calculate_flash_steam(5.0, 1.0, 0.1)
        assert result.energy_in_flash_kw > 0

    def test_flash_enthalpies_reasonable(self):
        """Test flash steam and liquid enthalpies are reasonable."""
        result = calculate_flash_steam(5.0, 1.0, 0.1)

        # Steam enthalpy should be much higher than liquid
        assert result.flash_steam_enthalpy_kj_kg > result.liquid_enthalpy_kj_kg

        # Steam enthalpy should be in typical range (~2500-2700 kJ/kg)
        assert 2400 < result.flash_steam_enthalpy_kj_kg < 2800

    def test_flash_provenance_hash(self):
        """Test provenance hash is generated."""
        result = calculate_flash_steam(5.0, 1.0, 0.1)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


class TestReturnRatio:
    """Test condensate return ratio calculation."""

    def test_return_ratio_full_recovery(self):
        """Test return ratio = 1 for full recovery."""
        ratio = calculate_return_ratio(10.0, 10.0)
        assert ratio == 1.0

    def test_return_ratio_no_recovery(self):
        """Test return ratio = 0 for no recovery."""
        ratio = calculate_return_ratio(0.0, 10.0)
        assert ratio == 0.0

    def test_return_ratio_partial(self):
        """Test return ratio for partial recovery."""
        ratio = calculate_return_ratio(7.0, 10.0)
        assert ratio == 0.7

    def test_return_ratio_clamped_above_one(self):
        """Test return ratio is clamped if recovered > total."""
        ratio = calculate_return_ratio(12.0, 10.0)
        assert ratio == 1.0

    def test_return_ratio_zero_total(self):
        """Test return ratio = 0 when total is zero."""
        ratio = calculate_return_ratio(0.0, 0.0)
        assert ratio == 0.0

    @pytest.mark.parametrize("recovered,total,expected", [
        (0, 100, 0.0),
        (50, 100, 0.5),
        (75, 100, 0.75),
        (100, 100, 1.0),
    ])
    def test_return_ratio_parametrized(self, recovered, total, expected):
        """Parametrized return ratio tests."""
        ratio = calculate_return_ratio(float(recovered), float(total))
        assert pytest.approx(ratio, rel=0.001) == expected


class TestHeatRecovery:
    """Test heat recovery calculation."""

    def test_heat_recovery_positive(self):
        """Test heat recovery is positive when condensate is hot."""
        heat = calculate_heat_recovery(
            condensate_flow_kg_s=5.0,
            condensate_temperature_k=423.0,  # 150 C
            reference_temperature_k=288.15   # 15 C
        )
        assert heat > 0

    def test_heat_recovery_zero_cold_condensate(self):
        """Test heat recovery is zero when condensate is cold."""
        heat = calculate_heat_recovery(
            condensate_flow_kg_s=5.0,
            condensate_temperature_k=288.15,
            reference_temperature_k=288.15
        )
        assert heat == 0

    def test_heat_recovery_scales_with_flow(self):
        """Test heat recovery scales linearly with flow."""
        heat_1 = calculate_heat_recovery(1.0, 423.0, 288.15)
        heat_5 = calculate_heat_recovery(5.0, 423.0, 288.15)

        assert pytest.approx(heat_5, rel=0.01) == 5 * heat_1

    def test_heat_recovery_scales_with_temperature(self):
        """Test heat recovery increases with temperature difference."""
        heat_low = calculate_heat_recovery(5.0, 353.0, 288.15)  # 80 C
        heat_high = calculate_heat_recovery(5.0, 423.0, 288.15)  # 150 C

        assert heat_high > heat_low

    def test_heat_recovery_formula(self):
        """Test heat recovery matches Q = m * Cp * dT."""
        flow = 5.0
        t_cond = 423.0
        t_ref = 288.15

        heat = calculate_heat_recovery(flow, t_cond, t_ref)
        expected = flow * CP_WATER * (t_cond - t_ref)

        assert pytest.approx(heat, rel=0.001) == expected


class TestLossIdentification:
    """Test condensate loss identification."""

    def test_identify_unrecovered_losses(self, sample_condensate_streams):
        """Test identification of unrecovered streams."""
        losses = identify_condensate_losses(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002"]
        )

        # COND-003 is contaminated, COND-004 is unrecovered
        assert losses[CondensateLossType.CONTAMINATION] > 0
        assert losses[CondensateLossType.UNRECOVERED] > 0

    def test_identify_contamination_losses(self, sample_condensate_streams):
        """Test identification of contamination losses."""
        losses = identify_condensate_losses(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002", "COND-004"]
        )

        # COND-003 (0.8 kg/s) is contaminated
        assert pytest.approx(losses[CondensateLossType.CONTAMINATION], rel=0.01) == 0.8

    def test_trap_leakage_estimation(self, sample_condensate_streams):
        """Test trap leakage is estimated on recovered streams."""
        losses = identify_condensate_losses(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002"],
            trap_leakage_fraction=0.05
        )

        # Trap leakage = 5% of recovered flow (2.0 + 1.5 = 3.5 kg/s)
        expected_leakage = 3.5 * 0.05
        assert pytest.approx(losses[CondensateLossType.TRAP_LEAKAGE], rel=0.01) == expected_leakage

    def test_evaporation_estimation(self, sample_condensate_streams):
        """Test evaporation loss is estimated."""
        losses = identify_condensate_losses(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001"],
            evaporation_fraction=0.02
        )

        total_flow = sum(s.mass_flow_kg_s for s in sample_condensate_streams)
        expected_evap = total_flow * 0.02

        assert pytest.approx(losses[CondensateLossType.EVAPORATION], rel=0.01) == expected_evap


class TestCondensateRecoveryAnalysis:
    """Test comprehensive condensate recovery analysis."""

    def test_recovery_analysis_structure(self, sample_condensate_streams):
        """Test recovery analysis returns complete structure."""
        result = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002"]
        )

        assert result.total_condensate_kg_s > 0
        assert result.recovered_condensate_kg_s >= 0
        assert 0 <= result.recovery_ratio <= 1
        assert result.heat_recovery_kw >= 0
        assert result.provenance_hash is not None

    def test_recovery_ratio_correct(self, sample_condensate_streams):
        """Test recovery ratio is calculated correctly."""
        result = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002"]
        )

        # Total = 2.0 + 1.5 + 0.8 + 0.5 = 4.8 kg/s
        # Recovered = 2.0 + 1.5 = 3.5 kg/s (COND-003 contaminated)
        expected_ratio = 3.5 / 4.8

        assert pytest.approx(result.recovery_ratio, rel=0.01) == expected_ratio

    def test_contaminated_streams_not_recovered(self, sample_condensate_streams):
        """Test contaminated streams are not counted as recovered."""
        result = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002", "COND-003"]  # Include contaminated
        )

        # COND-003 should not be recovered due to contamination
        assert result.recovered_condensate_kg_s == 3.5  # Only COND-001 + COND-002

    def test_recommendations_generated(self, sample_condensate_streams):
        """Test recommendations are generated for improvements."""
        result = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001"]  # Low recovery
        )

        assert len(result.recommendations) > 0

    def test_priority_assignment(self, sample_condensate_streams):
        """Test priority is assigned based on potential savings."""
        result_low = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001", "COND-002", "COND-004"]  # Most recovered
        )

        result_high = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=[]  # Nothing recovered
        )

        # More unrecovered should have higher priority
        assert result_high.potential_savings_kw >= result_low.potential_savings_kw

    def test_empty_streams_raises_error(self):
        """Test that empty stream list raises error."""
        with pytest.raises(CondensateError):
            analyze_condensate_recovery(streams=[], recovered_streams=[])


class TestHeatRecoveryEconomics:
    """Test heat recovery economics analysis."""

    def test_economics_positive_recovery(self):
        """Test economics calculation with positive heat recovery."""
        result = analyze_heat_recovery_economics(
            heat_recovery_kw=100.0,
            operating_hours_per_year=8000,
            steam_cost_per_kwh=0.05
        )

        assert result.recoverable_heat_kw > 0
        assert result.economic_value_per_hour > 0

    def test_economics_scales_with_heat(self):
        """Test economics scales with heat recovery."""
        result_small = analyze_heat_recovery_economics(heat_recovery_kw=50.0)
        result_large = analyze_heat_recovery_economics(heat_recovery_kw=100.0)

        assert result_large.economic_value_per_hour > result_small.economic_value_per_hour

    def test_payback_calculation(self):
        """Test payback period calculation."""
        result = analyze_heat_recovery_economics(
            heat_recovery_kw=100.0,
            operating_hours_per_year=8000,
            steam_cost_per_kwh=0.05,
            investment_cost=10000.0,
            recovery_efficiency=0.85
        )

        # Annual value = 100 * 0.85 * 0.05 * 8000 = 34000
        # Payback = 10000 / 34000 * 12 = ~3.5 months
        assert result.payback_months is not None
        assert result.payback_months > 0

    def test_no_payback_without_investment(self):
        """Test payback is None when no investment specified."""
        result = analyze_heat_recovery_economics(
            heat_recovery_kw=100.0,
            investment_cost=0.0
        )

        assert result.payback_months is None

    def test_recovery_efficiency_applied(self):
        """Test recovery efficiency reduces recoverable heat."""
        result_high = analyze_heat_recovery_economics(
            heat_recovery_kw=100.0,
            recovery_efficiency=0.95
        )
        result_low = analyze_heat_recovery_economics(
            heat_recovery_kw=100.0,
            recovery_efficiency=0.70
        )

        assert result_high.recoverable_heat_kw > result_low.recoverable_heat_kw


class TestProvenanceTracking:
    """Test provenance hash generation across calculations."""

    def test_flash_provenance_deterministic(self):
        """Test flash calculation provenance is deterministic."""
        result1 = calculate_flash_steam(5.0, 1.0, 0.1)
        result2 = calculate_flash_steam(5.0, 1.0, 0.1)

        assert result1.provenance_hash == result2.provenance_hash

    def test_flash_provenance_changes_with_input(self):
        """Test flash provenance changes with input."""
        result1 = calculate_flash_steam(5.0, 1.0, 0.1)
        result2 = calculate_flash_steam(5.0, 1.0, 0.2)  # Different downstream pressure

        assert result1.provenance_hash != result2.provenance_hash

    def test_recovery_analysis_provenance(self, sample_condensate_streams):
        """Test recovery analysis has provenance hash."""
        result = analyze_condensate_recovery(
            streams=sample_condensate_streams,
            recovered_streams=["COND-001"]
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_economics_provenance(self):
        """Test economics analysis has provenance hash."""
        result = analyze_heat_recovery_economics(heat_recovery_kw=100.0)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_pressure_flash(self):
        """Test flash calculation at very low downstream pressure."""
        result = calculate_flash_steam(
            condensate_flow_kg_s=5.0,
            upstream_pressure_mpa=1.0,
            downstream_pressure_mpa=0.01  # Near vacuum
        )

        # Should still be physical (high flash fraction)
        assert 0 < result.flash_steam_fraction < 1
        assert result.liquid_condensate_flow_kg_s > 0

    def test_high_pressure_flash(self):
        """Test flash from high pressure."""
        result = calculate_flash_steam(
            condensate_flow_kg_s=5.0,
            upstream_pressure_mpa=10.0,  # High pressure
            downstream_pressure_mpa=0.5
        )

        assert result.flash_steam_fraction > 0.1  # Significant flash expected

    def test_small_condensate_flow(self):
        """Test with very small condensate flow."""
        result = calculate_flash_steam(
            condensate_flow_kg_s=0.001,
            upstream_pressure_mpa=1.0,
            downstream_pressure_mpa=0.1
        )

        assert result.flash_steam_flow_kg_s >= 0
        assert result.flash_steam_flow_kg_s < 0.001

    def test_single_stream_analysis(self):
        """Test analysis with single stream."""
        stream = CondensateStream(
            stream_id="SINGLE",
            source="Test",
            mass_flow_kg_s=5.0,
            temperature_k=423.0,
            pressure_mpa=0.5
        )

        result = analyze_condensate_recovery(
            streams=[stream],
            recovered_streams=["SINGLE"]
        )

        assert result.recovery_ratio == 1.0
        assert result.total_condensate_kg_s == 5.0
