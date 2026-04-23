"""
Unit Tests: Steam Separator Efficiency Calculator

Tests the steam separator (moisture separator) efficiency calculations:
1. Separation efficiency calculation
2. Mass balance verification
3. Outlet quality prediction
4. Condensate rate calculation
5. Performance degradation detection

Reference: ASME and separator manufacturer specifications
Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone


# =============================================================================
# Constants
# =============================================================================

# Separator efficiency bounds
MIN_EFFICIENCY = 0.0
MAX_EFFICIENCY = 1.0
DESIGN_EFFICIENCY = 0.95  # Typical design efficiency
DEGRADED_EFFICIENCY_THRESHOLD = 0.85

# Mass balance tolerance
MASS_BALANCE_TOLERANCE = 0.001  # 0.1% tolerance

# Minimum flow thresholds
MIN_INLET_FLOW_KG_S = 0.001
MIN_STEAM_QUALITY = 0.0
MAX_STEAM_QUALITY = 1.0


# =============================================================================
# Enumerations
# =============================================================================

class SeparatorStatus(Enum):
    """Separator operational status."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    MAINTENANCE_REQUIRED = "maintenance_required"
    OFFLINE = "offline"


class SeparatorType(Enum):
    """Type of steam separator."""
    CENTRIFUGAL = "centrifugal"
    BAFFLE = "baffle"
    MESH = "mesh"
    CYCLONE = "cyclone"
    CHEVRON = "chevron"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SeparatorConfig:
    """Configuration for steam separator."""
    separator_id: str
    separator_type: SeparatorType
    design_efficiency: float
    design_pressure_mpa: float
    design_flow_kg_s: float
    min_turndown_ratio: float = 0.25
    max_turnup_ratio: float = 1.25


@dataclass
class SeparatorInput:
    """Input conditions to separator."""
    inlet_flow_kg_s: float
    inlet_dryness_fraction: float
    inlet_pressure_mpa: float
    inlet_temperature_k: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class SeparatorOutput:
    """Output conditions from separator."""
    steam_flow_kg_s: float
    steam_dryness_fraction: float
    condensate_flow_kg_s: float
    steam_pressure_mpa: float
    separation_efficiency: float
    mass_balance_error: float


@dataclass
class SeparatorPerformance:
    """Separator performance assessment."""
    current_efficiency: float
    design_efficiency: float
    efficiency_ratio: float
    status: SeparatorStatus
    degradation_percent: float
    estimated_steam_loss_kg_s: float
    maintenance_recommendation: str
    provenance_hash: str


# =============================================================================
# Separator Calculator Implementation
# =============================================================================

class SeparatorCalculationError(Exception):
    """Error in separator calculation."""
    pass


def validate_separator_inputs(
    inlet_flow_kg_s: float,
    inlet_dryness_fraction: float,
    inlet_pressure_mpa: float,
) -> List[str]:
    """
    Validate separator input parameters.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    if inlet_flow_kg_s < 0:
        errors.append(f"Inlet flow cannot be negative: {inlet_flow_kg_s}")

    if inlet_flow_kg_s < MIN_INLET_FLOW_KG_S:
        errors.append(f"Inlet flow below minimum threshold: {inlet_flow_kg_s} < {MIN_INLET_FLOW_KG_S}")

    if inlet_dryness_fraction < MIN_STEAM_QUALITY:
        errors.append(f"Dryness fraction cannot be negative: {inlet_dryness_fraction}")

    if inlet_dryness_fraction > MAX_STEAM_QUALITY:
        errors.append(f"Dryness fraction cannot exceed 1.0: {inlet_dryness_fraction}")

    if inlet_pressure_mpa <= 0:
        errors.append(f"Pressure must be positive: {inlet_pressure_mpa}")

    return errors


def calculate_separation_efficiency(
    inlet_dryness_fraction: float,
    outlet_dryness_fraction: float,
) -> float:
    """
    Calculate separation efficiency.

    Efficiency = (x_out - x_in) / (1 - x_in)

    This represents the fraction of moisture removed.

    Args:
        inlet_dryness_fraction: Steam quality at inlet
        outlet_dryness_fraction: Steam quality at outlet

    Returns:
        Separation efficiency (0 to 1)
    """
    if inlet_dryness_fraction >= 1.0:
        # Already dry steam, no separation possible
        return 0.0

    if outlet_dryness_fraction < inlet_dryness_fraction:
        # Output quality lower than input - something is wrong
        return 0.0

    efficiency = (outlet_dryness_fraction - inlet_dryness_fraction) / (1.0 - inlet_dryness_fraction)

    return max(0.0, min(1.0, efficiency))


def calculate_outlet_quality(
    inlet_dryness_fraction: float,
    separation_efficiency: float,
) -> float:
    """
    Calculate outlet steam quality given inlet quality and efficiency.

    x_out = x_in + efficiency * (1 - x_in)

    Args:
        inlet_dryness_fraction: Steam quality at inlet
        separation_efficiency: Separator efficiency (0 to 1)

    Returns:
        Outlet steam quality
    """
    if inlet_dryness_fraction >= 1.0:
        return 1.0

    outlet_quality = inlet_dryness_fraction + separation_efficiency * (1.0 - inlet_dryness_fraction)

    return max(0.0, min(1.0, outlet_quality))


def calculate_mass_flows(
    inlet_flow_kg_s: float,
    inlet_dryness_fraction: float,
    outlet_dryness_fraction: float,
) -> Tuple[float, float, float]:
    """
    Calculate outlet steam and condensate mass flows.

    Based on mass and quality balance:
    m_in = m_steam + m_condensate
    m_in * x_in = m_steam * x_out + m_condensate * 0

    Args:
        inlet_flow_kg_s: Total inlet flow
        inlet_dryness_fraction: Inlet steam quality
        outlet_dryness_fraction: Outlet steam quality

    Returns:
        Tuple of (steam_flow, condensate_flow, mass_balance_error)
    """
    if inlet_flow_kg_s <= 0:
        return 0.0, 0.0, 0.0

    if outlet_dryness_fraction <= 0:
        # All condensate (extreme case)
        return 0.0, inlet_flow_kg_s, 0.0

    # From mass and quality balance:
    # m_steam = m_in * x_in / x_out
    steam_flow = inlet_flow_kg_s * inlet_dryness_fraction / outlet_dryness_fraction

    # Condensate is the difference
    condensate_flow = inlet_flow_kg_s - steam_flow

    # Verify mass balance
    total_out = steam_flow + condensate_flow
    mass_balance_error = abs(total_out - inlet_flow_kg_s) / inlet_flow_kg_s if inlet_flow_kg_s > 0 else 0.0

    # Clamp to physical limits
    steam_flow = max(0.0, min(inlet_flow_kg_s, steam_flow))
    condensate_flow = max(0.0, inlet_flow_kg_s - steam_flow)

    return steam_flow, condensate_flow, mass_balance_error


def calculate_separator_output(
    config: SeparatorConfig,
    input_conditions: SeparatorInput,
    actual_efficiency: Optional[float] = None,
) -> SeparatorOutput:
    """
    Calculate separator output conditions.

    Args:
        config: Separator configuration
        input_conditions: Input conditions
        actual_efficiency: Actual efficiency (uses design if not provided)

    Returns:
        SeparatorOutput with calculated flows and quality
    """
    # Validate inputs
    errors = validate_separator_inputs(
        input_conditions.inlet_flow_kg_s,
        input_conditions.inlet_dryness_fraction,
        input_conditions.inlet_pressure_mpa,
    )

    if errors:
        raise SeparatorCalculationError("; ".join(errors))

    # Use design efficiency if actual not provided
    efficiency = actual_efficiency if actual_efficiency is not None else config.design_efficiency

    # Check if operating within design range
    flow_ratio = input_conditions.inlet_flow_kg_s / config.design_flow_kg_s
    if flow_ratio < config.min_turndown_ratio or flow_ratio > config.max_turnup_ratio:
        # Efficiency degrades outside design range
        if flow_ratio < config.min_turndown_ratio:
            efficiency *= 0.8  # Reduced efficiency at low flows
        else:
            efficiency *= 0.9  # Reduced efficiency at high flows

    # Calculate outlet quality
    outlet_quality = calculate_outlet_quality(
        input_conditions.inlet_dryness_fraction,
        efficiency,
    )

    # Calculate mass flows
    steam_flow, condensate_flow, mass_error = calculate_mass_flows(
        input_conditions.inlet_flow_kg_s,
        input_conditions.inlet_dryness_fraction,
        outlet_quality,
    )

    # Pressure drop (simplified - typically 1-5% pressure drop)
    pressure_drop_fraction = 0.02  # 2% pressure drop
    outlet_pressure = input_conditions.inlet_pressure_mpa * (1 - pressure_drop_fraction)

    return SeparatorOutput(
        steam_flow_kg_s=steam_flow,
        steam_dryness_fraction=outlet_quality,
        condensate_flow_kg_s=condensate_flow,
        steam_pressure_mpa=outlet_pressure,
        separation_efficiency=efficiency,
        mass_balance_error=mass_error,
    )


def assess_separator_performance(
    config: SeparatorConfig,
    measured_inlet_quality: float,
    measured_outlet_quality: float,
    inlet_flow_kg_s: float,
) -> SeparatorPerformance:
    """
    Assess separator performance against design specifications.

    Args:
        config: Separator configuration
        measured_inlet_quality: Measured inlet steam quality
        measured_outlet_quality: Measured outlet steam quality
        inlet_flow_kg_s: Inlet flow rate

    Returns:
        SeparatorPerformance assessment
    """
    # Calculate actual efficiency
    actual_efficiency = calculate_separation_efficiency(
        measured_inlet_quality,
        measured_outlet_quality,
    )

    # Compare to design
    efficiency_ratio = actual_efficiency / config.design_efficiency if config.design_efficiency > 0 else 0

    # Determine status
    if actual_efficiency >= DEGRADED_EFFICIENCY_THRESHOLD * config.design_efficiency:
        status = SeparatorStatus.NORMAL
        degradation = 0.0
    elif actual_efficiency >= 0.7 * config.design_efficiency:
        status = SeparatorStatus.DEGRADED
        degradation = (1 - efficiency_ratio) * 100
    else:
        status = SeparatorStatus.MAINTENANCE_REQUIRED
        degradation = (1 - efficiency_ratio) * 100

    # Estimate steam loss due to degradation
    expected_outlet = calculate_outlet_quality(measured_inlet_quality, config.design_efficiency)
    moisture_not_removed = expected_outlet - measured_outlet_quality
    steam_loss = inlet_flow_kg_s * moisture_not_removed if moisture_not_removed > 0 else 0

    # Generate maintenance recommendation
    if status == SeparatorStatus.NORMAL:
        recommendation = "No maintenance required. Continue normal operation."
    elif status == SeparatorStatus.DEGRADED:
        recommendation = f"Performance degraded by {degradation:.1f}%. Schedule inspection during next outage."
    else:
        recommendation = f"Performance significantly degraded by {degradation:.1f}%. Immediate inspection recommended."

    # Calculate provenance hash
    inputs = {
        "separator_id": config.separator_id,
        "measured_inlet_quality": round(measured_inlet_quality, 10),
        "measured_outlet_quality": round(measured_outlet_quality, 10),
        "inlet_flow_kg_s": round(inlet_flow_kg_s, 10),
    }
    provenance_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    return SeparatorPerformance(
        current_efficiency=actual_efficiency,
        design_efficiency=config.design_efficiency,
        efficiency_ratio=efficiency_ratio,
        status=status,
        degradation_percent=degradation,
        estimated_steam_loss_kg_s=steam_loss,
        maintenance_recommendation=recommendation,
        provenance_hash=provenance_hash,
    )


def estimate_design_efficiency(separator_type: SeparatorType) -> float:
    """
    Estimate design efficiency based on separator type.

    Args:
        separator_type: Type of separator

    Returns:
        Estimated design efficiency
    """
    efficiencies = {
        SeparatorType.CENTRIFUGAL: 0.95,
        SeparatorType.CYCLONE: 0.93,
        SeparatorType.MESH: 0.90,
        SeparatorType.CHEVRON: 0.92,
        SeparatorType.BAFFLE: 0.85,
    }

    return efficiencies.get(separator_type, 0.90)


# =============================================================================
# Test Classes
# =============================================================================

class TestSeparationEfficiencyCalculation:
    """Tests for separation efficiency calculation."""

    def test_perfect_separation(self):
        """Test efficiency = 1 when outlet is completely dry."""
        efficiency = calculate_separation_efficiency(0.8, 1.0)
        assert efficiency == pytest.approx(1.0)

    def test_no_separation(self):
        """Test efficiency = 0 when no quality improvement."""
        efficiency = calculate_separation_efficiency(0.8, 0.8)
        assert efficiency == pytest.approx(0.0)

    def test_partial_separation(self):
        """Test partial separation calculation."""
        # If inlet is 0.8 and outlet is 0.9:
        # efficiency = (0.9 - 0.8) / (1.0 - 0.8) = 0.1 / 0.2 = 0.5
        efficiency = calculate_separation_efficiency(0.8, 0.9)
        assert efficiency == pytest.approx(0.5)

    def test_dry_inlet_returns_zero(self):
        """Test that dry inlet (x=1) returns zero efficiency."""
        efficiency = calculate_separation_efficiency(1.0, 1.0)
        assert efficiency == 0.0

    def test_degraded_output_returns_zero(self):
        """Test that degraded output (outlet < inlet) returns zero."""
        efficiency = calculate_separation_efficiency(0.9, 0.8)
        assert efficiency == 0.0

    @pytest.mark.parametrize("inlet,outlet,expected", [
        (0.7, 0.7, 0.0),   # No separation
        (0.7, 0.85, 0.5),  # 50% separation
        (0.7, 1.0, 1.0),   # Perfect separation
        (0.5, 0.75, 0.5),  # 50% separation at lower quality
        (0.9, 0.95, 0.5),  # 50% separation at higher quality
    ])
    def test_efficiency_parametrized(self, inlet, outlet, expected):
        """Parametrized test for efficiency calculation."""
        efficiency = calculate_separation_efficiency(inlet, outlet)
        assert efficiency == pytest.approx(expected, abs=0.01)

    def test_efficiency_bounded(self):
        """Test that efficiency is always in [0, 1]."""
        for inlet in [0.5, 0.7, 0.8, 0.9]:
            for outlet in [0.5, 0.7, 0.8, 0.9, 1.0]:
                if outlet >= inlet:
                    efficiency = calculate_separation_efficiency(inlet, outlet)
                    assert 0.0 <= efficiency <= 1.0


class TestOutletQualityCalculation:
    """Tests for outlet quality calculation."""

    def test_perfect_efficiency_gives_dry_steam(self):
        """Test that 100% efficiency gives dry steam."""
        outlet = calculate_outlet_quality(0.8, 1.0)
        assert outlet == pytest.approx(1.0)

    def test_zero_efficiency_gives_same_quality(self):
        """Test that 0% efficiency gives same quality as inlet."""
        outlet = calculate_outlet_quality(0.8, 0.0)
        assert outlet == pytest.approx(0.8)

    def test_partial_efficiency(self):
        """Test partial efficiency outlet calculation."""
        # x_out = 0.8 + 0.5 * (1.0 - 0.8) = 0.8 + 0.1 = 0.9
        outlet = calculate_outlet_quality(0.8, 0.5)
        assert outlet == pytest.approx(0.9)

    def test_dry_inlet_stays_dry(self):
        """Test that dry inlet stays dry regardless of efficiency."""
        outlet = calculate_outlet_quality(1.0, 0.5)
        assert outlet == pytest.approx(1.0)

    @pytest.mark.parametrize("inlet,efficiency,expected", [
        (0.7, 0.0, 0.7),
        (0.7, 0.5, 0.85),
        (0.7, 1.0, 1.0),
        (0.9, 0.5, 0.95),
    ])
    def test_outlet_quality_parametrized(self, inlet, efficiency, expected):
        """Parametrized test for outlet quality calculation."""
        outlet = calculate_outlet_quality(inlet, efficiency)
        assert outlet == pytest.approx(expected, abs=0.01)


class TestMassFlowCalculation:
    """Tests for mass flow calculation."""

    def test_dry_inlet_no_condensate(self):
        """Test that dry inlet produces no condensate."""
        steam, condensate, error = calculate_mass_flows(100.0, 1.0, 1.0)

        assert steam == pytest.approx(100.0)
        assert condensate == pytest.approx(0.0)
        assert error < MASS_BALANCE_TOLERANCE

    def test_mass_balance_conserved(self):
        """Test that mass is conserved."""
        steam, condensate, error = calculate_mass_flows(100.0, 0.8, 0.95)

        total_out = steam + condensate
        assert total_out == pytest.approx(100.0, rel=0.01)
        assert error < MASS_BALANCE_TOLERANCE

    def test_condensate_increases_with_moisture(self):
        """Test that more moisture produces more condensate."""
        _, condensate_low, _ = calculate_mass_flows(100.0, 0.9, 0.95)
        _, condensate_high, _ = calculate_mass_flows(100.0, 0.7, 0.95)

        assert condensate_high > condensate_low

    def test_zero_flow_returns_zeros(self):
        """Test that zero inlet flow returns zeros."""
        steam, condensate, error = calculate_mass_flows(0.0, 0.8, 0.95)

        assert steam == 0.0
        assert condensate == 0.0

    @pytest.mark.parametrize("inlet_flow,inlet_x,outlet_x", [
        (100.0, 0.9, 0.98),
        (50.0, 0.8, 0.95),
        (200.0, 0.7, 0.90),
    ])
    def test_mass_balance_various_conditions(self, inlet_flow, inlet_x, outlet_x):
        """Parametrized test for mass balance."""
        steam, condensate, error = calculate_mass_flows(inlet_flow, inlet_x, outlet_x)

        # Mass balance
        assert steam + condensate == pytest.approx(inlet_flow, rel=0.01)

        # Quality balance: inlet vapor = outlet vapor
        vapor_in = inlet_flow * inlet_x
        vapor_out = steam * outlet_x
        assert vapor_in == pytest.approx(vapor_out, rel=0.01)


class TestInputValidation:
    """Tests for input validation."""

    def test_negative_flow_rejected(self):
        """Test that negative flow is rejected."""
        errors = validate_separator_inputs(-10.0, 0.9, 1.0)
        assert len(errors) > 0
        assert any("negative" in e.lower() for e in errors)

    def test_below_minimum_flow_rejected(self):
        """Test that below-minimum flow is rejected."""
        errors = validate_separator_inputs(0.0001, 0.9, 1.0)
        assert len(errors) > 0
        assert any("minimum" in e.lower() or "threshold" in e.lower() for e in errors)

    def test_negative_quality_rejected(self):
        """Test that negative quality is rejected."""
        errors = validate_separator_inputs(100.0, -0.1, 1.0)
        assert len(errors) > 0

    def test_quality_over_one_rejected(self):
        """Test that quality > 1 is rejected."""
        errors = validate_separator_inputs(100.0, 1.1, 1.0)
        assert len(errors) > 0

    def test_negative_pressure_rejected(self):
        """Test that negative pressure is rejected."""
        errors = validate_separator_inputs(100.0, 0.9, -1.0)
        assert len(errors) > 0

    def test_valid_inputs_pass(self):
        """Test that valid inputs pass validation."""
        errors = validate_separator_inputs(100.0, 0.9, 1.0)
        assert len(errors) == 0


class TestSeparatorOutput:
    """Tests for separator output calculation."""

    @pytest.fixture
    def standard_config(self) -> SeparatorConfig:
        """Create standard separator configuration."""
        return SeparatorConfig(
            separator_id="SEP-001",
            separator_type=SeparatorType.CENTRIFUGAL,
            design_efficiency=0.95,
            design_pressure_mpa=5.0,
            design_flow_kg_s=100.0,
            min_turndown_ratio=0.25,
            max_turnup_ratio=1.25,
        )

    @pytest.fixture
    def standard_input(self) -> SeparatorInput:
        """Create standard input conditions."""
        return SeparatorInput(
            inlet_flow_kg_s=100.0,
            inlet_dryness_fraction=0.85,
            inlet_pressure_mpa=5.0,
            inlet_temperature_k=536.67,
        )

    def test_output_quality_improves(self, standard_config, standard_input):
        """Test that output quality is better than input."""
        output = calculate_separator_output(standard_config, standard_input)

        assert output.steam_dryness_fraction > standard_input.inlet_dryness_fraction

    def test_mass_balance_in_output(self, standard_config, standard_input):
        """Test mass balance in output."""
        output = calculate_separator_output(standard_config, standard_input)

        total_out = output.steam_flow_kg_s + output.condensate_flow_kg_s
        assert total_out == pytest.approx(standard_input.inlet_flow_kg_s, rel=0.01)

    def test_pressure_drop_applied(self, standard_config, standard_input):
        """Test that pressure drop is applied."""
        output = calculate_separator_output(standard_config, standard_input)

        assert output.steam_pressure_mpa < standard_input.inlet_pressure_mpa

    def test_efficiency_reduced_at_low_flow(self, standard_config):
        """Test that efficiency is reduced at low flow."""
        low_flow_input = SeparatorInput(
            inlet_flow_kg_s=10.0,  # 10% of design flow
            inlet_dryness_fraction=0.85,
            inlet_pressure_mpa=5.0,
            inlet_temperature_k=536.67,
        )

        output = calculate_separator_output(standard_config, low_flow_input)

        assert output.separation_efficiency < standard_config.design_efficiency

    def test_efficiency_reduced_at_high_flow(self, standard_config):
        """Test that efficiency is reduced at high flow."""
        high_flow_input = SeparatorInput(
            inlet_flow_kg_s=150.0,  # 150% of design flow
            inlet_dryness_fraction=0.85,
            inlet_pressure_mpa=5.0,
            inlet_temperature_k=536.67,
        )

        output = calculate_separator_output(standard_config, high_flow_input)

        assert output.separation_efficiency < standard_config.design_efficiency

    def test_invalid_input_raises_error(self, standard_config):
        """Test that invalid input raises error."""
        invalid_input = SeparatorInput(
            inlet_flow_kg_s=-10.0,  # Negative flow
            inlet_dryness_fraction=0.85,
            inlet_pressure_mpa=5.0,
            inlet_temperature_k=536.67,
        )

        with pytest.raises(SeparatorCalculationError):
            calculate_separator_output(standard_config, invalid_input)


class TestPerformanceAssessment:
    """Tests for separator performance assessment."""

    @pytest.fixture
    def standard_config(self) -> SeparatorConfig:
        """Create standard separator configuration."""
        return SeparatorConfig(
            separator_id="SEP-001",
            separator_type=SeparatorType.CENTRIFUGAL,
            design_efficiency=0.95,
            design_pressure_mpa=5.0,
            design_flow_kg_s=100.0,
        )

    def test_normal_performance(self, standard_config):
        """Test assessment of normal performance."""
        # Inlet 0.85, outlet 0.9925 -> efficiency = 0.95
        performance = assess_separator_performance(
            standard_config,
            measured_inlet_quality=0.85,
            measured_outlet_quality=0.9925,
            inlet_flow_kg_s=100.0,
        )

        assert performance.status == SeparatorStatus.NORMAL
        assert performance.efficiency_ratio > 0.95

    def test_degraded_performance(self, standard_config):
        """Test assessment of degraded performance."""
        # Lower outlet quality indicating degraded efficiency
        performance = assess_separator_performance(
            standard_config,
            measured_inlet_quality=0.85,
            measured_outlet_quality=0.92,  # Lower than expected
            inlet_flow_kg_s=100.0,
        )

        assert performance.status == SeparatorStatus.DEGRADED
        assert performance.degradation_percent > 0

    def test_maintenance_required_performance(self, standard_config):
        """Test assessment of severely degraded performance."""
        # Very low outlet quality
        performance = assess_separator_performance(
            standard_config,
            measured_inlet_quality=0.85,
            measured_outlet_quality=0.88,  # Much lower than expected
            inlet_flow_kg_s=100.0,
        )

        assert performance.status == SeparatorStatus.MAINTENANCE_REQUIRED
        assert "immediate" in performance.maintenance_recommendation.lower()

    def test_provenance_hash_generated(self, standard_config):
        """Test that provenance hash is generated."""
        performance = assess_separator_performance(
            standard_config,
            measured_inlet_quality=0.85,
            measured_outlet_quality=0.95,
            inlet_flow_kg_s=100.0,
        )

        assert performance.provenance_hash
        assert len(performance.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, standard_config):
        """Test that provenance hash is deterministic."""
        perf1 = assess_separator_performance(
            standard_config, 0.85, 0.95, 100.0,
        )
        perf2 = assess_separator_performance(
            standard_config, 0.85, 0.95, 100.0,
        )

        assert perf1.provenance_hash == perf2.provenance_hash


class TestDesignEfficiencyEstimation:
    """Tests for design efficiency estimation by separator type."""

    @pytest.mark.parametrize("sep_type,expected_min", [
        (SeparatorType.CENTRIFUGAL, 0.93),
        (SeparatorType.CYCLONE, 0.91),
        (SeparatorType.MESH, 0.88),
        (SeparatorType.CHEVRON, 0.90),
        (SeparatorType.BAFFLE, 0.83),
    ])
    def test_efficiency_by_type(self, sep_type, expected_min):
        """Test that efficiency estimation is reasonable by type."""
        efficiency = estimate_design_efficiency(sep_type)

        assert efficiency >= expected_min
        assert efficiency <= 1.0

    def test_centrifugal_highest_efficiency(self):
        """Test that centrifugal type has highest efficiency."""
        efficiencies = {t: estimate_design_efficiency(t) for t in SeparatorType}

        assert efficiencies[SeparatorType.CENTRIFUGAL] == max(efficiencies.values())

    def test_baffle_lowest_efficiency(self):
        """Test that baffle type has lowest efficiency."""
        efficiencies = {t: estimate_design_efficiency(t) for t in SeparatorType}

        assert efficiencies[SeparatorType.BAFFLE] == min(efficiencies.values())


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_output_calculation_identical(self):
        """Test that repeated calculations are identical."""
        config = SeparatorConfig(
            separator_id="SEP-001",
            separator_type=SeparatorType.CENTRIFUGAL,
            design_efficiency=0.95,
            design_pressure_mpa=5.0,
            design_flow_kg_s=100.0,
        )

        input_cond = SeparatorInput(
            inlet_flow_kg_s=100.0,
            inlet_dryness_fraction=0.85,
            inlet_pressure_mpa=5.0,
            inlet_temperature_k=536.67,
        )

        results = [calculate_separator_output(config, input_cond) for _ in range(10)]

        first = results[0]
        for result in results[1:]:
            assert result.steam_flow_kg_s == first.steam_flow_kg_s
            assert result.steam_dryness_fraction == first.steam_dryness_fraction
            assert result.separation_efficiency == first.separation_efficiency


class TestPhysicalReasonableness:
    """Tests for physical reasonableness of results."""

    def test_outlet_quality_never_exceeds_one(self):
        """Test that outlet quality never exceeds 1.0."""
        for inlet_x in [0.5, 0.7, 0.8, 0.9, 0.95]:
            for efficiency in [0.5, 0.8, 0.95, 1.0]:
                outlet = calculate_outlet_quality(inlet_x, efficiency)
                assert outlet <= 1.0

    def test_efficiency_improves_quality(self):
        """Test that positive efficiency always improves quality."""
        inlet_x = 0.8

        for efficiency in [0.1, 0.3, 0.5, 0.7, 0.9]:
            outlet = calculate_outlet_quality(inlet_x, efficiency)
            assert outlet >= inlet_x

    def test_steam_flow_less_than_inlet(self):
        """Test that steam flow is always <= inlet flow."""
        for inlet_flow in [50.0, 100.0, 200.0]:
            for inlet_x in [0.7, 0.8, 0.9]:
                steam, condensate, _ = calculate_mass_flows(inlet_flow, inlet_x, 0.95)
                assert steam <= inlet_flow
                assert condensate >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
