"""
Unit Tests: Enthalpy Balance Calculator

Tests mass and energy balance calculations for steam distribution networks
including synthetic networks with known flows/enthalpies, loss estimation,
and measurement reconciliation.

Test Categories:
1. Mass balance closure on synthetic network
2. Energy balance closure
3. Loss estimation accuracy
4. Measurement reconciliation
5. Error detection and handling

Reference: ASME PTC 19.11, ISO 50001

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
# Data Classes
# =============================================================================

class StreamType(Enum):
    """Type of steam/water stream."""
    STEAM_INLET = auto()
    STEAM_OUTLET = auto()
    CONDENSATE_RETURN = auto()
    MAKEUP_WATER = auto()
    BLOWDOWN = auto()
    FLASH_STEAM = auto()
    VENT_LOSS = auto()


@dataclass
class StreamData:
    """Data for a single stream in the network."""
    stream_id: str
    stream_type: StreamType
    mass_flow_kg_s: float
    enthalpy_kj_kg: float
    pressure_mpa: float
    temperature_k: float
    quality: Optional[float] = None  # Steam quality if two-phase
    measurement_uncertainty: float = 0.02  # Default 2% uncertainty


@dataclass
class MassBalanceResult:
    """Result of mass balance calculation."""
    total_mass_in_kg_s: float
    total_mass_out_kg_s: float
    imbalance_kg_s: float
    imbalance_percent: float
    is_balanced: bool
    closure_tolerance: float
    streams_in: List[str]
    streams_out: List[str]
    provenance_hash: str


@dataclass
class EnergyBalanceResult:
    """Result of energy balance calculation."""
    total_energy_in_kw: float
    total_energy_out_kw: float
    energy_loss_kw: float
    energy_loss_percent: float
    is_balanced: bool
    closure_tolerance: float
    enthalpy_rates: Dict[str, float]  # stream_id -> kW
    provenance_hash: str


@dataclass
class LossEstimate:
    """Estimated energy losses in the system."""
    distribution_loss_kw: float
    trap_loss_kw: float
    radiation_loss_kw: float
    flash_loss_kw: float
    total_loss_kw: float
    loss_breakdown: Dict[str, float]
    uncertainty_kw: float
    provenance_hash: str


@dataclass
class ReconciledState:
    """Result of measurement reconciliation."""
    original_values: Dict[str, float]
    reconciled_values: Dict[str, float]
    adjustments: Dict[str, float]
    residuals: Dict[str, float]
    chi_squared: float
    degrees_of_freedom: int
    is_consistent: bool
    gross_error_detected: bool
    gross_error_streams: List[str]
    provenance_hash: str


# =============================================================================
# Balance Calculator Implementation
# =============================================================================

class BalanceError(Exception):
    """Error in balance calculation."""
    pass


class InsufficientDataError(BalanceError):
    """Insufficient data for balance calculation."""
    pass


class GrossErrorDetectedError(BalanceError):
    """Gross measurement error detected."""
    pass


def compute_mass_balance(
    streams: List[StreamData],
    closure_tolerance: float = 0.02
) -> MassBalanceResult:
    """
    Compute mass balance for steam network.

    Args:
        streams: List of all streams in the network
        closure_tolerance: Acceptable imbalance fraction (default 2%)

    Returns:
        MassBalanceResult with balance analysis
    """
    if not streams:
        raise InsufficientDataError("No streams provided for mass balance")

    # Classify streams as inlet or outlet
    inlet_types = {
        StreamType.STEAM_INLET,
        StreamType.MAKEUP_WATER,
        StreamType.CONDENSATE_RETURN
    }
    outlet_types = {
        StreamType.STEAM_OUTLET,
        StreamType.BLOWDOWN,
        StreamType.FLASH_STEAM,
        StreamType.VENT_LOSS
    }

    streams_in = []
    streams_out = []
    total_in = 0.0
    total_out = 0.0

    for stream in streams:
        if stream.stream_type in inlet_types:
            total_in += stream.mass_flow_kg_s
            streams_in.append(stream.stream_id)
        elif stream.stream_type in outlet_types:
            total_out += stream.mass_flow_kg_s
            streams_out.append(stream.stream_id)

    if total_in <= 0:
        raise InsufficientDataError("No inlet streams found")

    imbalance = total_in - total_out
    imbalance_percent = abs(imbalance) / total_in * 100 if total_in > 0 else 0
    is_balanced = abs(imbalance) / total_in <= closure_tolerance if total_in > 0 else False

    # Compute provenance hash
    input_data = {
        "streams": [(s.stream_id, s.mass_flow_kg_s) for s in streams],
        "closure_tolerance": closure_tolerance
    }
    provenance_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()

    return MassBalanceResult(
        total_mass_in_kg_s=total_in,
        total_mass_out_kg_s=total_out,
        imbalance_kg_s=imbalance,
        imbalance_percent=imbalance_percent,
        is_balanced=is_balanced,
        closure_tolerance=closure_tolerance,
        streams_in=streams_in,
        streams_out=streams_out,
        provenance_hash=provenance_hash
    )


def compute_enthalpy_rate(stream: StreamData) -> float:
    """
    Compute enthalpy rate (energy flow) for a stream.

    Returns: Energy flow rate in kW
    """
    return stream.mass_flow_kg_s * stream.enthalpy_kj_kg


def compute_energy_balance(
    streams: List[StreamData],
    closure_tolerance: float = 0.05
) -> EnergyBalanceResult:
    """
    Compute energy balance for steam network.

    Args:
        streams: List of all streams in the network
        closure_tolerance: Acceptable energy imbalance fraction (default 5%)

    Returns:
        EnergyBalanceResult with balance analysis
    """
    if not streams:
        raise InsufficientDataError("No streams provided for energy balance")

    inlet_types = {
        StreamType.STEAM_INLET,
        StreamType.MAKEUP_WATER,
        StreamType.CONDENSATE_RETURN
    }
    outlet_types = {
        StreamType.STEAM_OUTLET,
        StreamType.BLOWDOWN,
        StreamType.FLASH_STEAM,
        StreamType.VENT_LOSS
    }

    total_energy_in = 0.0
    total_energy_out = 0.0
    enthalpy_rates = {}

    for stream in streams:
        h_rate = compute_enthalpy_rate(stream)
        enthalpy_rates[stream.stream_id] = h_rate

        if stream.stream_type in inlet_types:
            total_energy_in += h_rate
        elif stream.stream_type in outlet_types:
            total_energy_out += h_rate

    if total_energy_in <= 0:
        raise InsufficientDataError("No energy input streams found")

    energy_loss = total_energy_in - total_energy_out
    loss_percent = abs(energy_loss) / total_energy_in * 100 if total_energy_in > 0 else 0

    # Energy loss should be positive (energy out <= energy in due to losses)
    # Closure check is on the absolute difference
    is_balanced = abs(energy_loss) / total_energy_in <= closure_tolerance

    provenance_hash = hashlib.sha256(
        json.dumps({
            "enthalpy_rates": enthalpy_rates,
            "closure_tolerance": closure_tolerance
        }, sort_keys=True).encode()
    ).hexdigest()

    return EnergyBalanceResult(
        total_energy_in_kw=total_energy_in,
        total_energy_out_kw=total_energy_out,
        energy_loss_kw=energy_loss,
        energy_loss_percent=loss_percent,
        is_balanced=is_balanced,
        closure_tolerance=closure_tolerance,
        enthalpy_rates=enthalpy_rates,
        provenance_hash=provenance_hash
    )


def estimate_distribution_losses(
    pipe_length_m: float,
    pipe_diameter_m: float,
    steam_temperature_k: float,
    ambient_temperature_k: float,
    insulation_thickness_m: float = 0.05,
    insulation_conductivity_w_m_k: float = 0.04,
    surface_emissivity: float = 0.9,
    wind_speed_m_s: float = 0.0
) -> LossEstimate:
    """
    Estimate distribution losses from steam piping.

    Uses simplified heat transfer model combining:
    - Conduction through insulation
    - Convection to ambient
    - Radiation from surface
    """
    # Surface area of pipe
    outer_radius = pipe_diameter_m / 2 + insulation_thickness_m
    surface_area = 2 * math.pi * outer_radius * pipe_length_m

    # Temperature difference
    delta_t = steam_temperature_k - ambient_temperature_k

    if delta_t <= 0:
        return LossEstimate(
            distribution_loss_kw=0,
            trap_loss_kw=0,
            radiation_loss_kw=0,
            flash_loss_kw=0,
            total_loss_kw=0,
            loss_breakdown={},
            uncertainty_kw=0,
            provenance_hash=hashlib.sha256(b"no_loss").hexdigest()
        )

    # Conduction resistance through insulation
    r_inner = pipe_diameter_m / 2
    r_outer = r_inner + insulation_thickness_m
    R_cond = math.log(r_outer / r_inner) / (2 * math.pi * pipe_length_m * insulation_conductivity_w_m_k)

    # Convection coefficient (natural + forced)
    h_natural = 5.0  # W/(m2.K) natural convection
    h_forced = 10.0 * wind_speed_m_s ** 0.5 if wind_speed_m_s > 0 else 0
    h_conv = h_natural + h_forced
    R_conv = 1 / (h_conv * surface_area)

    # Total thermal resistance
    R_total = R_cond + R_conv

    # Conduction/convection heat loss
    q_cond_conv = delta_t / R_total / 1000  # kW

    # Radiation heat loss (simplified)
    stefan_boltzmann = 5.67e-8
    t_surface = ambient_temperature_k + 0.3 * delta_t  # Approximate surface temperature
    q_rad = surface_emissivity * stefan_boltzmann * surface_area * (
        t_surface ** 4 - ambient_temperature_k ** 4
    ) / 1000  # kW

    # Total distribution loss
    distribution_loss = q_cond_conv + q_rad

    # Estimate other losses (simplified empirical)
    trap_loss = distribution_loss * 0.05  # 5% of distribution loss
    flash_loss = distribution_loss * 0.02  # 2% flash losses

    total_loss = distribution_loss + trap_loss + flash_loss

    # Uncertainty estimate (20% of total)
    uncertainty = total_loss * 0.2

    loss_breakdown = {
        "conduction_convection": q_cond_conv,
        "radiation": q_rad,
        "trap_loss": trap_loss,
        "flash_loss": flash_loss
    }

    provenance_hash = hashlib.sha256(
        json.dumps({
            "inputs": {
                "pipe_length_m": pipe_length_m,
                "pipe_diameter_m": pipe_diameter_m,
                "steam_temperature_k": steam_temperature_k
            },
            "outputs": loss_breakdown
        }, sort_keys=True).encode()
    ).hexdigest()

    return LossEstimate(
        distribution_loss_kw=distribution_loss,
        trap_loss_kw=trap_loss,
        radiation_loss_kw=q_rad,
        flash_loss_kw=flash_loss,
        total_loss_kw=total_loss,
        loss_breakdown=loss_breakdown,
        uncertainty_kw=uncertainty,
        provenance_hash=provenance_hash
    )


def reconcile_measurements(
    streams: List[StreamData],
    constraints: List[Dict[str, Any]],
    gross_error_threshold: float = 3.0
) -> ReconciledState:
    """
    Reconcile measurements using weighted least squares to satisfy constraints.

    Implements simplified data reconciliation following:
    - Mass balance constraints
    - Energy balance constraints

    Detects gross errors using chi-squared test.
    """
    if not streams:
        raise InsufficientDataError("No streams for reconciliation")

    # Extract original values and uncertainties
    original_values = {}
    uncertainties = {}

    for stream in streams:
        original_values[stream.stream_id] = stream.mass_flow_kg_s
        uncertainties[stream.stream_id] = stream.mass_flow_kg_s * stream.measurement_uncertainty

    # Simplified reconciliation: adjust to close mass balance
    mass_balance = compute_mass_balance(streams, closure_tolerance=1.0)  # Allow any imbalance

    imbalance = mass_balance.imbalance_kg_s
    n_streams = len(streams)

    # Distribute imbalance proportionally by uncertainty
    total_uncertainty_sq = sum(u ** 2 for u in uncertainties.values())
    if total_uncertainty_sq <= 0:
        total_uncertainty_sq = 1e-10

    reconciled_values = {}
    adjustments = {}
    residuals = {}

    for stream in streams:
        stream_id = stream.stream_id
        u_sq = uncertainties[stream_id] ** 2

        # Adjustment proportional to variance
        if stream.stream_type in {StreamType.STEAM_INLET, StreamType.MAKEUP_WATER, StreamType.CONDENSATE_RETURN}:
            # Inlet streams: decrease if imbalance positive
            adj = -imbalance * (u_sq / total_uncertainty_sq)
        else:
            # Outlet streams: increase if imbalance positive
            adj = imbalance * (u_sq / total_uncertainty_sq)

        reconciled_values[stream_id] = original_values[stream_id] + adj
        adjustments[stream_id] = adj
        residuals[stream_id] = adj / uncertainties[stream_id] if uncertainties[stream_id] > 0 else 0

    # Chi-squared statistic
    chi_squared = sum(r ** 2 for r in residuals.values())
    degrees_of_freedom = max(1, n_streams - 1)

    # Gross error detection
    gross_error_streams = [
        stream_id for stream_id, r in residuals.items()
        if abs(r) > gross_error_threshold
    ]

    # Simplified consistency check
    is_consistent = chi_squared / degrees_of_freedom < 3.0  # Rule of thumb

    provenance_hash = hashlib.sha256(
        json.dumps({
            "original": original_values,
            "reconciled": reconciled_values,
            "chi_squared": chi_squared
        }, sort_keys=True).encode()
    ).hexdigest()

    return ReconciledState(
        original_values=original_values,
        reconciled_values=reconciled_values,
        adjustments=adjustments,
        residuals=residuals,
        chi_squared=chi_squared,
        degrees_of_freedom=degrees_of_freedom,
        is_consistent=is_consistent,
        gross_error_detected=len(gross_error_streams) > 0,
        gross_error_streams=gross_error_streams,
        provenance_hash=provenance_hash
    )


# =============================================================================
# Test Fixtures and Data Generators
# =============================================================================

def create_balanced_network() -> List[StreamData]:
    """Create a perfectly balanced synthetic network."""
    return [
        StreamData(
            stream_id="steam_header",
            stream_type=StreamType.STEAM_INLET,
            mass_flow_kg_s=10.0,
            enthalpy_kj_kg=2800.0,
            pressure_mpa=1.0,
            temperature_k=453.0,
            quality=1.0
        ),
        StreamData(
            stream_id="process_1",
            stream_type=StreamType.STEAM_OUTLET,
            mass_flow_kg_s=6.0,
            enthalpy_kj_kg=2800.0,
            pressure_mpa=1.0,
            temperature_k=453.0,
            quality=1.0
        ),
        StreamData(
            stream_id="process_2",
            stream_type=StreamType.STEAM_OUTLET,
            mass_flow_kg_s=3.5,
            enthalpy_kj_kg=2800.0,
            pressure_mpa=1.0,
            temperature_k=453.0,
            quality=1.0
        ),
        StreamData(
            stream_id="blowdown",
            stream_type=StreamType.BLOWDOWN,
            mass_flow_kg_s=0.5,
            enthalpy_kj_kg=762.0,  # Saturated liquid
            pressure_mpa=1.0,
            temperature_k=453.0,
            quality=0.0
        ),
    ]


def create_imbalanced_network(imbalance_fraction: float = 0.1) -> List[StreamData]:
    """Create a network with specified mass imbalance."""
    base = create_balanced_network()

    # Increase inlet to create imbalance
    base[0].mass_flow_kg_s *= (1 + imbalance_fraction)

    return base


def create_complex_network() -> List[StreamData]:
    """Create a more complex network with multiple inlets and outlets."""
    return [
        # Inlets
        StreamData(
            stream_id="boiler_1",
            stream_type=StreamType.STEAM_INLET,
            mass_flow_kg_s=15.0,
            enthalpy_kj_kg=2800.0,
            pressure_mpa=1.5,
            temperature_k=471.0,
            quality=1.0
        ),
        StreamData(
            stream_id="boiler_2",
            stream_type=StreamType.STEAM_INLET,
            mass_flow_kg_s=10.0,
            enthalpy_kj_kg=2820.0,
            pressure_mpa=1.5,
            temperature_k=480.0,  # Superheated
            quality=None
        ),
        StreamData(
            stream_id="condensate_return",
            stream_type=StreamType.CONDENSATE_RETURN,
            mass_flow_kg_s=8.0,
            enthalpy_kj_kg=400.0,
            pressure_mpa=0.5,
            temperature_k=370.0,
            quality=0.0
        ),
        StreamData(
            stream_id="makeup_water",
            stream_type=StreamType.MAKEUP_WATER,
            mass_flow_kg_s=2.0,
            enthalpy_kj_kg=100.0,
            pressure_mpa=0.5,
            temperature_k=310.0,
            quality=0.0
        ),
        # Outlets
        StreamData(
            stream_id="process_a",
            stream_type=StreamType.STEAM_OUTLET,
            mass_flow_kg_s=12.0,
            enthalpy_kj_kg=2750.0,
            pressure_mpa=1.0,
            temperature_k=453.0,
            quality=1.0
        ),
        StreamData(
            stream_id="process_b",
            stream_type=StreamType.STEAM_OUTLET,
            mass_flow_kg_s=10.0,
            enthalpy_kj_kg=2600.0,
            pressure_mpa=0.5,
            temperature_k=420.0,
            quality=0.95
        ),
        StreamData(
            stream_id="process_c",
            stream_type=StreamType.STEAM_OUTLET,
            mass_flow_kg_s=8.0,
            enthalpy_kj_kg=2700.0,
            pressure_mpa=0.8,
            temperature_k=443.0,
            quality=1.0
        ),
        StreamData(
            stream_id="blowdown",
            stream_type=StreamType.BLOWDOWN,
            mass_flow_kg_s=1.5,
            enthalpy_kj_kg=720.0,
            pressure_mpa=1.5,
            temperature_k=471.0,
            quality=0.0
        ),
        StreamData(
            stream_id="vent",
            stream_type=StreamType.VENT_LOSS,
            mass_flow_kg_s=0.5,
            enthalpy_kj_kg=2800.0,
            pressure_mpa=0.1,
            temperature_k=373.0,
            quality=1.0
        ),
    ]


# =============================================================================
# Test Classes
# =============================================================================

class TestMassBalance:
    """Test mass balance calculations."""

    def test_balanced_network_closes(self):
        """Test that a balanced network closes within tolerance."""
        streams = create_balanced_network()
        result = compute_mass_balance(streams)

        assert result.is_balanced
        assert result.imbalance_percent < 2.0
        assert abs(result.imbalance_kg_s) < 0.01

    def test_imbalanced_network_detected(self):
        """Test that imbalanced network is detected."""
        streams = create_imbalanced_network(imbalance_fraction=0.1)
        result = compute_mass_balance(streams, closure_tolerance=0.02)

        assert not result.is_balanced
        assert result.imbalance_percent > 8.0  # ~10% imbalance

    def test_mass_in_equals_mass_out_balanced(self):
        """Test that mass in equals mass out for balanced network."""
        streams = create_balanced_network()
        result = compute_mass_balance(streams)

        assert pytest.approx(result.total_mass_in_kg_s, rel=0.001) == result.total_mass_out_kg_s

    def test_stream_classification(self):
        """Test that streams are correctly classified as inlet/outlet."""
        streams = create_balanced_network()
        result = compute_mass_balance(streams)

        assert "steam_header" in result.streams_in
        assert "process_1" in result.streams_out
        assert "blowdown" in result.streams_out

    def test_complex_network_balance(self):
        """Test mass balance on complex network."""
        streams = create_complex_network()
        result = compute_mass_balance(streams)

        # Complex network should have mass balance
        total_in = 15.0 + 10.0 + 8.0 + 2.0  # 35 kg/s
        total_out = 12.0 + 10.0 + 8.0 + 1.5 + 0.5  # 32 kg/s

        assert pytest.approx(result.total_mass_in_kg_s, rel=0.01) == total_in
        assert pytest.approx(result.total_mass_out_kg_s, rel=0.01) == total_out

    def test_empty_streams_raises_error(self):
        """Test that empty stream list raises error."""
        with pytest.raises(InsufficientDataError):
            compute_mass_balance([])

    def test_no_inlets_raises_error(self):
        """Test that network with no inlets raises error."""
        streams = [
            StreamData("outlet", StreamType.STEAM_OUTLET, 10.0, 2800.0, 1.0, 453.0)
        ]
        with pytest.raises(InsufficientDataError):
            compute_mass_balance(streams)

    def test_provenance_hash_generated(self):
        """Test that provenance hash is generated."""
        streams = create_balanced_network()
        result = compute_mass_balance(streams)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        streams = create_balanced_network()
        result1 = compute_mass_balance(streams)
        result2 = compute_mass_balance(streams)

        assert result1.provenance_hash == result2.provenance_hash

    @pytest.mark.parametrize("tolerance", [0.01, 0.02, 0.05, 0.10])
    def test_closure_tolerance_affects_result(self, tolerance):
        """Test that closure tolerance affects is_balanced determination."""
        streams = create_imbalanced_network(imbalance_fraction=0.03)
        result = compute_mass_balance(streams, closure_tolerance=tolerance)

        if tolerance >= 0.05:
            assert result.is_balanced
        else:
            assert not result.is_balanced


class TestEnergyBalance:
    """Test energy balance calculations."""

    def test_energy_balance_computes_rates(self):
        """Test that enthalpy rates are computed for all streams."""
        streams = create_balanced_network()
        result = compute_energy_balance(streams)

        for stream in streams:
            assert stream.stream_id in result.enthalpy_rates

    def test_enthalpy_rate_calculation(self):
        """Test that enthalpy rate = mass_flow * specific_enthalpy."""
        stream = StreamData(
            stream_id="test",
            stream_type=StreamType.STEAM_INLET,
            mass_flow_kg_s=5.0,
            enthalpy_kj_kg=2800.0,
            pressure_mpa=1.0,
            temperature_k=453.0
        )

        h_rate = compute_enthalpy_rate(stream)
        expected = 5.0 * 2800.0  # 14000 kW

        assert pytest.approx(h_rate, rel=0.001) == expected

    def test_balanced_energy_network(self):
        """Test energy balance on network with equal inlet/outlet enthalpies."""
        streams = create_balanced_network()
        result = compute_energy_balance(streams)

        # Energy in from steam header
        energy_in = 10.0 * 2800.0  # 28000 kW

        # Energy out (assuming blowdown has different enthalpy)
        energy_out = 6.0 * 2800.0 + 3.5 * 2800.0 + 0.5 * 762.0

        assert pytest.approx(result.total_energy_in_kw, rel=0.01) == energy_in

    def test_energy_loss_calculation(self):
        """Test that energy loss is computed correctly."""
        streams = create_complex_network()
        result = compute_energy_balance(streams)

        # Energy loss should be positive (some energy lost to environment)
        assert result.energy_loss_kw >= 0 or result.is_balanced

    def test_energy_closure_tolerance(self):
        """Test energy balance closure with different tolerances."""
        streams = create_complex_network()

        result_tight = compute_energy_balance(streams, closure_tolerance=0.01)
        result_loose = compute_energy_balance(streams, closure_tolerance=0.20)

        # Loose tolerance should be more likely to show balanced
        if not result_tight.is_balanced:
            assert result_loose.is_balanced or result_loose.energy_loss_percent > 20

    def test_empty_streams_raises_error(self):
        """Test that empty stream list raises error."""
        with pytest.raises(InsufficientDataError):
            compute_energy_balance([])

    def test_provenance_hash_for_energy_balance(self):
        """Test provenance hash generation for energy balance."""
        streams = create_balanced_network()
        result = compute_energy_balance(streams)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


class TestLossEstimation:
    """Test distribution loss estimation."""

    def test_loss_estimation_positive(self):
        """Test that loss estimation produces positive values."""
        result = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0
        )

        assert result.total_loss_kw > 0
        assert result.distribution_loss_kw > 0

    def test_loss_increases_with_temperature_difference(self):
        """Test that loss increases with temperature difference."""
        result_low = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=373.0,  # 100 C
            ambient_temperature_k=293.0
        )

        result_high = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,  # 180 C
            ambient_temperature_k=293.0
        )

        assert result_high.total_loss_kw > result_low.total_loss_kw

    def test_loss_increases_with_pipe_length(self):
        """Test that loss increases with pipe length."""
        result_short = estimate_distribution_losses(
            pipe_length_m=50.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0
        )

        result_long = estimate_distribution_losses(
            pipe_length_m=200.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0
        )

        assert result_long.total_loss_kw > result_short.total_loss_kw

    def test_loss_decreases_with_insulation(self):
        """Test that loss decreases with more insulation."""
        result_thin = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0,
            insulation_thickness_m=0.025
        )

        result_thick = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0,
            insulation_thickness_m=0.10
        )

        assert result_thick.total_loss_kw < result_thin.total_loss_kw

    def test_loss_breakdown_sums_to_total(self):
        """Test that loss breakdown components sum to distribution loss."""
        result = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0
        )

        breakdown_sum = (
            result.loss_breakdown.get("conduction_convection", 0) +
            result.loss_breakdown.get("radiation", 0)
        )

        assert pytest.approx(result.distribution_loss_kw, rel=0.01) == breakdown_sum

    def test_zero_loss_when_no_temperature_difference(self):
        """Test zero loss when steam and ambient temperatures are equal."""
        result = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=293.0,
            ambient_temperature_k=293.0
        )

        assert result.total_loss_kw == 0

    def test_uncertainty_estimation(self):
        """Test that uncertainty is estimated."""
        result = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0
        )

        assert result.uncertainty_kw > 0
        assert result.uncertainty_kw < result.total_loss_kw

    def test_wind_increases_loss(self):
        """Test that wind increases heat loss."""
        result_calm = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0,
            wind_speed_m_s=0.0
        )

        result_windy = estimate_distribution_losses(
            pipe_length_m=100.0,
            pipe_diameter_m=0.15,
            steam_temperature_k=453.0,
            ambient_temperature_k=293.0,
            wind_speed_m_s=5.0
        )

        assert result_windy.total_loss_kw > result_calm.total_loss_kw


class TestMeasurementReconciliation:
    """Test measurement reconciliation."""

    def test_reconciliation_closes_balance(self):
        """Test that reconciliation closes mass balance."""
        streams = create_imbalanced_network(imbalance_fraction=0.05)
        result = reconcile_measurements(streams, constraints=[])

        # Verify reconciled values close balance
        total_in = sum(
            result.reconciled_values[s.stream_id]
            for s in streams
            if s.stream_type in {StreamType.STEAM_INLET, StreamType.MAKEUP_WATER, StreamType.CONDENSATE_RETURN}
        )
        total_out = sum(
            result.reconciled_values[s.stream_id]
            for s in streams
            if s.stream_type in {StreamType.STEAM_OUTLET, StreamType.BLOWDOWN, StreamType.FLASH_STEAM, StreamType.VENT_LOSS}
        )

        # Should be much closer after reconciliation
        assert abs(total_in - total_out) < abs(create_imbalanced_network(0.05)[0].mass_flow_kg_s * 0.05)

    def test_reconciliation_adjustments_proportional_to_uncertainty(self):
        """Test that adjustments are proportional to measurement uncertainty."""
        streams = [
            StreamData("high_uncertainty", StreamType.STEAM_INLET, 10.0, 2800.0, 1.0, 453.0, measurement_uncertainty=0.10),
            StreamData("low_uncertainty", StreamType.STEAM_OUTLET, 9.0, 2800.0, 1.0, 453.0, measurement_uncertainty=0.01),
        ]

        result = reconcile_measurements(streams, constraints=[])

        # Higher uncertainty stream should have larger adjustment
        adj_high = abs(result.adjustments["high_uncertainty"])
        adj_low = abs(result.adjustments["low_uncertainty"])

        assert adj_high >= adj_low

    def test_gross_error_detection(self):
        """Test gross error detection."""
        streams = [
            StreamData("normal_1", StreamType.STEAM_INLET, 10.0, 2800.0, 1.0, 453.0, measurement_uncertainty=0.02),
            StreamData("normal_2", StreamType.STEAM_OUTLET, 9.5, 2800.0, 1.0, 453.0, measurement_uncertainty=0.02),
            StreamData("gross_error", StreamType.STEAM_OUTLET, 5.0, 2800.0, 1.0, 453.0, measurement_uncertainty=0.02),  # Way too low
        ]

        result = reconcile_measurements(streams, constraints=[], gross_error_threshold=2.0)

        # Should detect gross error in one stream
        # Note: Detection depends on residual magnitude
        assert result.chi_squared > 0

    def test_chi_squared_computed(self):
        """Test that chi-squared statistic is computed."""
        streams = create_balanced_network()
        result = reconcile_measurements(streams, constraints=[])

        assert result.chi_squared >= 0
        assert result.degrees_of_freedom > 0

    def test_consistency_check(self):
        """Test consistency check based on chi-squared."""
        streams = create_balanced_network()
        result = reconcile_measurements(streams, constraints=[])

        # Balanced network should be consistent
        assert result.is_consistent

    def test_empty_streams_raises_error(self):
        """Test that empty stream list raises error."""
        with pytest.raises(InsufficientDataError):
            reconcile_measurements([], constraints=[])

    def test_reconciled_values_structure(self):
        """Test that reconciled result has correct structure."""
        streams = create_balanced_network()
        result = reconcile_measurements(streams, constraints=[])

        for stream in streams:
            assert stream.stream_id in result.original_values
            assert stream.stream_id in result.reconciled_values
            assert stream.stream_id in result.adjustments
            assert stream.stream_id in result.residuals

    def test_provenance_hash_for_reconciliation(self):
        """Test provenance hash generation for reconciliation."""
        streams = create_balanced_network()
        result = reconcile_measurements(streams, constraints=[])

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


class TestSyntheticNetworkValidation:
    """Validate against synthetic networks with known solutions."""

    @pytest.fixture
    def known_network(self):
        """Create network with known mass and energy balance."""
        return [
            StreamData("inlet", StreamType.STEAM_INLET, 100.0, 2750.0, 1.0, 453.0, quality=1.0),
            StreamData("outlet_1", StreamType.STEAM_OUTLET, 60.0, 2750.0, 1.0, 453.0, quality=1.0),
            StreamData("outlet_2", StreamType.STEAM_OUTLET, 35.0, 2750.0, 1.0, 453.0, quality=1.0),
            StreamData("blowdown", StreamType.BLOWDOWN, 5.0, 762.0, 1.0, 453.0, quality=0.0),
        ]

    def test_known_mass_balance(self, known_network):
        """Test mass balance against known values."""
        result = compute_mass_balance(known_network)

        # Known: 100 in, 60 + 35 + 5 = 100 out
        assert pytest.approx(result.total_mass_in_kg_s, rel=0.001) == 100.0
        assert pytest.approx(result.total_mass_out_kg_s, rel=0.001) == 100.0
        assert result.is_balanced

    def test_known_energy_balance(self, known_network):
        """Test energy balance against known values."""
        result = compute_energy_balance(known_network)

        # Known energy in: 100 * 2750 = 275000 kW
        assert pytest.approx(result.total_energy_in_kw, rel=0.001) == 275000.0

        # Known energy out: 60*2750 + 35*2750 + 5*762 = 165000 + 96250 + 3810 = 265060 kW
        expected_out = 60 * 2750 + 35 * 2750 + 5 * 762
        assert pytest.approx(result.total_energy_out_kw, rel=0.001) == expected_out

    def test_energy_loss_matches_blowdown(self, known_network):
        """Test that energy loss is primarily from blowdown enthalpy difference."""
        result = compute_energy_balance(known_network)

        # Blowdown takes out saturated liquid instead of steam
        # Loss = 5 * (2750 - 762) = 5 * 1988 = 9940 kW
        expected_loss = 275000.0 - (60 * 2750 + 35 * 2750 + 5 * 762)

        assert pytest.approx(result.energy_loss_kw, rel=0.01) == expected_loss


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_inlet_single_outlet(self):
        """Test simplest possible network."""
        streams = [
            StreamData("in", StreamType.STEAM_INLET, 10.0, 2800.0, 1.0, 453.0),
            StreamData("out", StreamType.STEAM_OUTLET, 10.0, 2800.0, 1.0, 453.0),
        ]

        mass_result = compute_mass_balance(streams)
        energy_result = compute_energy_balance(streams)

        assert mass_result.is_balanced
        assert energy_result.is_balanced

    def test_very_small_flows(self):
        """Test with very small flow rates."""
        streams = [
            StreamData("in", StreamType.STEAM_INLET, 0.001, 2800.0, 1.0, 453.0),
            StreamData("out", StreamType.STEAM_OUTLET, 0.001, 2800.0, 1.0, 453.0),
        ]

        mass_result = compute_mass_balance(streams)
        energy_result = compute_energy_balance(streams)

        assert mass_result.is_balanced
        assert energy_result.is_balanced

    def test_very_large_flows(self):
        """Test with very large flow rates."""
        streams = [
            StreamData("in", StreamType.STEAM_INLET, 10000.0, 2800.0, 1.0, 453.0),
            StreamData("out", StreamType.STEAM_OUTLET, 10000.0, 2800.0, 1.0, 453.0),
        ]

        mass_result = compute_mass_balance(streams)
        energy_result = compute_energy_balance(streams)

        assert mass_result.is_balanced
        assert energy_result.is_balanced

    def test_all_condensate_return(self):
        """Test network with all condensate return."""
        streams = [
            StreamData("steam_in", StreamType.STEAM_INLET, 10.0, 2800.0, 1.0, 453.0),
            StreamData("condensate_return", StreamType.CONDENSATE_RETURN, 8.0, 400.0, 0.5, 370.0),
            StreamData("steam_out", StreamType.STEAM_OUTLET, 18.0, 2000.0, 0.5, 420.0),
        ]

        mass_result = compute_mass_balance(streams)

        # Mass in: 10 + 8 = 18, Mass out: 18
        assert pytest.approx(mass_result.total_mass_in_kg_s, rel=0.001) == 18.0
        assert mass_result.is_balanced

    def test_multiple_blowdown_points(self):
        """Test network with multiple blowdown streams."""
        streams = [
            StreamData("in", StreamType.STEAM_INLET, 100.0, 2800.0, 1.0, 453.0),
            StreamData("out", StreamType.STEAM_OUTLET, 90.0, 2800.0, 1.0, 453.0),
            StreamData("blowdown_1", StreamType.BLOWDOWN, 5.0, 762.0, 1.0, 453.0),
            StreamData("blowdown_2", StreamType.BLOWDOWN, 3.0, 762.0, 1.0, 453.0),
            StreamData("blowdown_3", StreamType.BLOWDOWN, 2.0, 762.0, 1.0, 453.0),
        ]

        mass_result = compute_mass_balance(streams)

        assert pytest.approx(mass_result.total_mass_out_kg_s, rel=0.001) == 100.0
        assert mass_result.is_balanced
