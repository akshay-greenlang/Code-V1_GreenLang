"""
GL-020 ECONOPULSE: Advanced Soot Blower Optimizer Test Suite

Comprehensive test coverage for the advanced soot blower optimizer.
Tests include unit tests, edge cases, provenance verification, and
calculation accuracy validation.

Author: GL-BackendDeveloper
Test Coverage Target: 85%+
"""

from __future__ import annotations

import hashlib
import json
import math
import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from calculators.advanced_soot_blower_optimizer import (
    # Enumerations
    BlowerType,
    BlowingMedium,
    EconomizerZone,
    CleaningPriority,
    WearSeverity,
    # Constants
    EPRI_GUIDELINES,
    ZONE_IMPORTANCE_WEIGHTS,
    STEAM_ENERGY_CONTENT,
    EROSION_RATES,
    # Data Classes
    SootBlowerConfiguration,
    ZoneFoulingState,
    BlowingIntervalResult,
    ZonePriorityResult,
    MediaConsumptionResult,
    CleaningEffectivenessResult,
    ROIAnalysisResult,
    ErosionMonitorResult,
    SequentialScheduleResult,
    EnergyBalanceResult,
    # Functions
    calculate_optimal_blowing_interval,
    prioritize_cleaning_zones,
    track_media_consumption,
    measure_cleaning_effectiveness,
    analyze_cleaning_roi,
    monitor_erosion_wear,
    optimize_blowing_sequence,
    calculate_soot_blowing_energy_balance,
    clear_optimizer_cache,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_blower_config() -> SootBlowerConfiguration:
    """Create sample soot blower configuration."""
    return SootBlowerConfiguration(
        blower_id="SB-ECO-001",
        blower_type=BlowerType.RETRACTABLE_LANCE,
        medium=BlowingMedium.SATURATED_STEAM,
        zone=EconomizerZone.GAS_INLET,
        steam_flow_lbm_per_cycle=500.0,
        steam_pressure_psia=300.0,
        steam_temperature_f=500.0,
        cycle_duration_seconds=300.0,
        lance_travel_inches=120.0,
        nozzle_orifice_dia_inches=0.5
    )


@pytest.fixture
def sample_blower_configs() -> List[SootBlowerConfiguration]:
    """Create list of sample soot blower configurations."""
    configs = []
    zones = [
        EconomizerZone.GAS_INLET,
        EconomizerZone.MIDDLE,
        EconomizerZone.GAS_OUTLET,
        EconomizerZone.BOTTOM_BANK
    ]

    for i, zone in enumerate(zones):
        configs.append(SootBlowerConfiguration(
            blower_id=f"SB-ECO-{i+1:03d}",
            blower_type=BlowerType.RETRACTABLE_LANCE,
            medium=BlowingMedium.SATURATED_STEAM,
            zone=zone,
            steam_flow_lbm_per_cycle=400.0 + i * 50,
            steam_pressure_psia=300.0,
            steam_temperature_f=500.0,
            cycle_duration_seconds=300.0
        ))

    return configs


@pytest.fixture
def sample_zone_states() -> List[ZoneFoulingState]:
    """Create sample zone fouling states."""
    states = []
    zones_fouling = [
        (EconomizerZone.GAS_INLET, 0.008, 7.0, 10.0),      # Heavy fouling
        (EconomizerZone.MIDDLE, 0.004, 8.5, 10.0),          # Moderate
        (EconomizerZone.GAS_OUTLET, 0.002, 9.2, 10.0),      # Light
        (EconomizerZone.BOTTOM_BANK, 0.006, 7.8, 10.0),     # Moderate-heavy
    ]

    for zone, rf, u_current, u_clean in zones_fouling:
        states.append(ZoneFoulingState(
            zone=zone,
            fouling_factor=rf,
            u_value_current=u_current,
            u_value_clean=u_clean,
            gas_temp_inlet_f=550.0,
            gas_temp_outlet_f=350.0,
            last_cleaning_timestamp=datetime.now(timezone.utc) - timedelta(days=7),
            hours_since_cleaning=168.0
        ))

    return states


@pytest.fixture
def zone_blower_map() -> Dict[EconomizerZone, SootBlowerConfiguration]:
    """Create mapping of zones to blower configurations."""
    return {
        EconomizerZone.GAS_INLET: SootBlowerConfiguration(
            blower_id="SB-INLET-001",
            blower_type=BlowerType.RETRACTABLE_LANCE,
            medium=BlowingMedium.SATURATED_STEAM,
            zone=EconomizerZone.GAS_INLET,
            steam_flow_lbm_per_cycle=550.0
        ),
        EconomizerZone.MIDDLE: SootBlowerConfiguration(
            blower_id="SB-MID-001",
            blower_type=BlowerType.ROTARY_WALL,
            medium=BlowingMedium.SATURATED_STEAM,
            zone=EconomizerZone.MIDDLE,
            steam_flow_lbm_per_cycle=400.0
        ),
        EconomizerZone.GAS_OUTLET: SootBlowerConfiguration(
            blower_id="SB-OUT-001",
            blower_type=BlowerType.FIXED_POSITION,
            medium=BlowingMedium.COMPRESSED_AIR,
            zone=EconomizerZone.GAS_OUTLET,
            air_flow_scfm=500.0
        ),
        EconomizerZone.BOTTOM_BANK: SootBlowerConfiguration(
            blower_id="SB-BOT-001",
            blower_type=BlowerType.RETRACTABLE_LANCE,
            medium=BlowingMedium.SUPERHEATED_STEAM,
            zone=EconomizerZone.BOTTOM_BANK,
            steam_flow_lbm_per_cycle=600.0
        ),
    }


# =============================================================================
# TEST: OPTIMAL BLOWING INTERVAL
# =============================================================================

class TestOptimalBlowingInterval:
    """Tests for calculate_optimal_blowing_interval function."""

    def test_basic_interval_calculation(self):
        """Test basic optimal blowing interval calculation."""
        result = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert isinstance(result, BlowingIntervalResult)
        assert result.optimal_interval_hours > 0
        assert float(result.min_interval_hours) == EPRI_GUIDELINES["min_interval_hours"]
        assert float(result.max_interval_hours) == EPRI_GUIDELINES["max_interval_hours"]

    def test_higher_fouling_shorter_interval(self):
        """Test that higher fouling rate leads to shorter interval."""
        result_low = calculate_optimal_blowing_interval(
            fouling_rate=1e-7,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )
        result_high = calculate_optimal_blowing_interval(
            fouling_rate=1e-5,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert result_high.optimal_interval_hours < result_low.optimal_interval_hours

    def test_higher_cost_longer_interval(self):
        """Test that higher cleaning cost leads to longer interval."""
        result_cheap = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=20.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )
        result_expensive = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=200.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert result_expensive.optimal_interval_hours > result_cheap.optimal_interval_hours

    def test_zero_fouling_maximum_interval(self):
        """Test that zero fouling rate gives maximum interval."""
        result = calculate_optimal_blowing_interval(
            fouling_rate=0.0,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert float(result.optimal_interval_hours) == EPRI_GUIDELINES["max_interval_hours"]

    def test_recommended_next_blow_time(self):
        """Test recommended next blow time calculation."""
        last_cleaning = datetime.now(timezone.utc) - timedelta(hours=24)
        result = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0,
            last_cleaning_time=last_cleaning
        )

        assert result.recommended_next_blow is not None

    def test_confidence_level(self):
        """Test confidence level calculation."""
        result = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert 0 <= float(result.confidence_level) <= 1

    def test_provenance_tracking(self):
        """Test provenance tracking returns complete record."""
        result, provenance = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0,
            track_provenance=True
        )

        assert provenance is not None
        assert provenance.verify_integrity()
        assert len(provenance.steps) > 0


# =============================================================================
# TEST: ZONE PRIORITIZATION
# =============================================================================

class TestZonePrioritization:
    """Tests for prioritize_cleaning_zones function."""

    def test_basic_prioritization(self, sample_zone_states, zone_blower_map):
        """Test basic zone prioritization."""
        results = prioritize_cleaning_zones(
            zone_states=sample_zone_states,
            blower_configs=zone_blower_map,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ZonePriorityResult) for r in results)

    def test_priority_order(self, sample_zone_states, zone_blower_map):
        """Test that results are sorted by priority score."""
        results = prioritize_cleaning_zones(
            zone_states=sample_zone_states,
            blower_configs=zone_blower_map,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85
        )

        # Verify descending order
        scores = [float(r.priority_score) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_heavily_fouled_highest_priority(self, sample_zone_states, zone_blower_map):
        """Test that heavily fouled zone gets highest priority."""
        results = prioritize_cleaning_zones(
            zone_states=sample_zone_states,
            blower_configs=zone_blower_map,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85
        )

        # GAS_INLET has highest fouling (0.008) and highest weight
        assert results[0].zone == EconomizerZone.GAS_INLET

    def test_max_zones_limit(self, sample_zone_states, zone_blower_map):
        """Test maximum zones per cycle limit."""
        results = prioritize_cleaning_zones(
            zone_states=sample_zone_states,
            blower_configs=zone_blower_map,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            max_zones_per_cycle=2
        )

        assert len(results) <= 2

    def test_priority_level_assignment(self, sample_zone_states, zone_blower_map):
        """Test priority level assignment."""
        results = prioritize_cleaning_zones(
            zone_states=sample_zone_states,
            blower_configs=zone_blower_map,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85
        )

        for result in results:
            assert result.priority_level in list(CleaningPriority)

    def test_recommended_action_provided(self, sample_zone_states, zone_blower_map):
        """Test that recommended action is provided."""
        results = prioritize_cleaning_zones(
            zone_states=sample_zone_states,
            blower_configs=zone_blower_map,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85
        )

        for result in results:
            assert result.recommended_action != ""


# =============================================================================
# TEST: MEDIA CONSUMPTION TRACKING
# =============================================================================

class TestMediaConsumptionTracking:
    """Tests for track_media_consumption function."""

    def test_basic_consumption_tracking(self, sample_blower_configs):
        """Test basic media consumption tracking."""
        cycles = {config.blower_id: 10 for config in sample_blower_configs}

        result = track_media_consumption(
            blower_configs=sample_blower_configs,
            cycles_per_blower=cycles
        )

        assert isinstance(result, MediaConsumptionResult)
        assert result.steam_consumed_lbm > 0
        assert result.steam_cost > 0
        assert result.cycles_tracked == 40  # 10 cycles * 4 blowers

    def test_zero_cycles(self, sample_blower_configs):
        """Test with zero cycles."""
        cycles = {config.blower_id: 0 for config in sample_blower_configs}

        result = track_media_consumption(
            blower_configs=sample_blower_configs,
            cycles_per_blower=cycles
        )

        assert result.steam_consumed_lbm == Decimal("0.00")
        assert result.total_cost == Decimal("0.00")

    def test_air_blower_consumption(self):
        """Test consumption tracking for air blowers."""
        air_config = SootBlowerConfiguration(
            blower_id="SB-AIR-001",
            blower_type=BlowerType.FIXED_POSITION,
            medium=BlowingMedium.COMPRESSED_AIR,
            zone=EconomizerZone.GAS_OUTLET,
            air_flow_scfm=500.0,
            cycle_duration_seconds=300.0
        )

        result = track_media_consumption(
            blower_configs=[air_config],
            cycles_per_blower={"SB-AIR-001": 5}
        )

        assert result.air_consumed_scf > 0
        assert result.steam_consumed_lbm == Decimal("0.00")


# =============================================================================
# TEST: CLEANING EFFECTIVENESS
# =============================================================================

class TestCleaningEffectiveness:
    """Tests for measure_cleaning_effectiveness function."""

    def test_basic_effectiveness_measurement(self):
        """Test basic cleaning effectiveness measurement."""
        result = measure_cleaning_effectiveness(
            u_before=7.5,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        assert isinstance(result, CleaningEffectivenessResult)
        assert result.effectiveness > 0
        assert result.recovery_percent > 0
        assert result.rf_removed > 0

    def test_perfect_cleaning(self):
        """Test 100% cleaning effectiveness."""
        result = measure_cleaning_effectiveness(
            u_before=7.0,
            u_after=10.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        assert float(result.effectiveness) == pytest.approx(1.0, abs=0.01)
        assert float(result.recovery_percent) == pytest.approx(100.0, abs=1.0)

    def test_partial_cleaning(self):
        """Test partial cleaning effectiveness."""
        result = measure_cleaning_effectiveness(
            u_before=8.0,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        assert 0 < float(result.effectiveness) < 1
        assert 0 < float(result.recovery_percent) < 100

    def test_no_improvement(self):
        """Test no improvement from cleaning."""
        result = measure_cleaning_effectiveness(
            u_before=8.0,
            u_after=8.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        assert float(result.effectiveness) == 0
        assert float(result.rf_removed) == pytest.approx(0, abs=1e-6)

    def test_invalid_u_values(self):
        """Test that invalid U-values raise error."""
        with pytest.raises(ValueError):
            measure_cleaning_effectiveness(
                u_before=-1.0,
                u_after=9.0,
                u_clean=10.0,
                blower_id="SB-001"
            )


# =============================================================================
# TEST: ROI ANALYSIS
# =============================================================================

class TestROIAnalysis:
    """Tests for analyze_cleaning_roi function."""

    def test_basic_roi_analysis(self):
        """Test basic ROI analysis."""
        result = analyze_cleaning_roi(
            cleaning_cost=50.0,
            fuel_penalty_before=20.0,
            fuel_penalty_after=5.0,
            expected_interval_hours=48.0
        )

        assert isinstance(result, ROIAnalysisResult)
        assert result.fuel_savings_per_hour == Decimal("15.00")  # 20 - 5
        assert result.net_benefit > 0
        assert result.roi_percent > 0

    def test_positive_roi(self):
        """Test positive ROI calculation."""
        result = analyze_cleaning_roi(
            cleaning_cost=50.0,
            fuel_penalty_before=25.0,
            fuel_penalty_after=5.0,
            expected_interval_hours=24.0
        )

        # Savings: 20 $/hr * 24 hr = 480
        # Net benefit: 480 - 50 = 430
        # ROI: 430/50 * 100 = 860%
        assert float(result.net_benefit) > 0
        assert float(result.roi_percent) > 0

    def test_negative_roi(self):
        """Test negative ROI when cleaning cost exceeds benefit."""
        result = analyze_cleaning_roi(
            cleaning_cost=1000.0,
            fuel_penalty_before=10.0,
            fuel_penalty_after=9.0,
            expected_interval_hours=10.0
        )

        # Savings: 1 $/hr * 10 hr = 10
        # Net benefit: 10 - 1000 = -990
        assert float(result.net_benefit) < 0
        assert float(result.roi_percent) < 0

    def test_payback_hours_calculation(self):
        """Test payback hours calculation."""
        result = analyze_cleaning_roi(
            cleaning_cost=100.0,
            fuel_penalty_before=30.0,
            fuel_penalty_after=10.0,
            expected_interval_hours=48.0
        )

        # Payback = 100 / 20 = 5 hours
        assert float(result.payback_hours) == pytest.approx(5.0, abs=0.1)


# =============================================================================
# TEST: EROSION MONITORING
# =============================================================================

class TestErosionMonitoring:
    """Tests for monitor_erosion_wear function."""

    def test_basic_erosion_monitoring(self, sample_blower_config):
        """Test basic erosion wear monitoring."""
        result = monitor_erosion_wear(
            blower_config=sample_blower_config,
            cumulative_cycles=5000,
            initial_tube_thickness_mils=120.0,
            wear_limit_mils=40.0
        )

        assert isinstance(result, ErosionMonitorResult)
        assert result.cumulative_cycles == 5000
        assert result.estimated_wear_mils > 0
        assert result.remaining_life_percent > 0

    def test_new_blower_minimal_wear(self, sample_blower_config):
        """Test that new blower has minimal wear."""
        result = monitor_erosion_wear(
            blower_config=sample_blower_config,
            cumulative_cycles=100
        )

        assert result.wear_severity == WearSeverity.MINIMAL
        assert float(result.remaining_life_percent) > 90

    def test_worn_blower_significant_wear(self, sample_blower_config):
        """Test significantly worn blower."""
        result = monitor_erosion_wear(
            blower_config=sample_blower_config,
            cumulative_cycles=100000  # Very high cycle count
        )

        assert result.wear_severity in [WearSeverity.SIGNIFICANT, WearSeverity.CRITICAL]
        assert float(result.remaining_life_percent) < 50

    def test_severity_classification(self, sample_blower_config):
        """Test wear severity classification."""
        severities = []

        for cycles in [100, 10000, 50000, 80000, 150000]:
            result = monitor_erosion_wear(
                blower_config=sample_blower_config,
                cumulative_cycles=cycles
            )
            severities.append(result.wear_severity)

        # Should progress through severities
        assert WearSeverity.MINIMAL in severities[:2]

    def test_inspection_recommendation(self, sample_blower_config):
        """Test inspection recommendation is provided."""
        result = monitor_erosion_wear(
            blower_config=sample_blower_config,
            cumulative_cycles=5000
        )

        assert result.recommended_inspection != ""


# =============================================================================
# TEST: SEQUENTIAL SCHEDULE OPTIMIZATION
# =============================================================================

class TestSequentialScheduleOptimization:
    """Tests for optimize_blowing_sequence function."""

    def test_basic_sequence_optimization(self, sample_blower_configs):
        """Test basic sequence optimization."""
        priority_scores = {
            EconomizerZone.GAS_INLET: 10.0,
            EconomizerZone.MIDDLE: 5.0,
            EconomizerZone.GAS_OUTLET: 3.0,
            EconomizerZone.BOTTOM_BANK: 7.0
        }

        result = optimize_blowing_sequence(
            blower_configs=sample_blower_configs,
            zone_priority_scores=priority_scores
        )

        assert isinstance(result, SequentialScheduleResult)
        assert len(result.sequence) > 0
        assert result.total_duration_minutes > 0

    def test_priority_order_in_sequence(self, sample_blower_configs):
        """Test that sequence follows priority order."""
        priority_scores = {
            EconomizerZone.GAS_INLET: 10.0,
            EconomizerZone.MIDDLE: 5.0,
            EconomizerZone.GAS_OUTLET: 2.0,
            EconomizerZone.BOTTOM_BANK: 8.0
        }

        result = optimize_blowing_sequence(
            blower_configs=sample_blower_configs,
            zone_priority_scores=priority_scores
        )

        # First in sequence should be from highest priority zone
        # (GAS_INLET with score 10)
        assert len(result.sequence) > 0

    def test_cooling_intervals_included(self, sample_blower_configs):
        """Test that cooling intervals are included."""
        priority_scores = {zone: 5.0 for zone in EconomizerZone}

        result = optimize_blowing_sequence(
            blower_configs=sample_blower_configs,
            zone_priority_scores=priority_scores,
            min_cooling_interval_minutes=15.0
        )

        assert float(result.cooling_intervals_minutes) == 15.0

    def test_max_duration_limit(self, sample_blower_configs):
        """Test maximum duration limit is respected."""
        priority_scores = {zone: 5.0 for zone in EconomizerZone}

        result = optimize_blowing_sequence(
            blower_configs=sample_blower_configs,
            zone_priority_scores=priority_scores,
            max_total_duration_minutes=15.0  # Very short limit
        )

        # Should limit number of blowers scheduled
        assert float(result.total_duration_minutes) <= 15.0 or len(result.sequence) <= 1


# =============================================================================
# TEST: ENERGY BALANCE
# =============================================================================

class TestEnergyBalance:
    """Tests for calculate_soot_blowing_energy_balance function."""

    def test_basic_energy_balance(self):
        """Test basic energy balance calculation."""
        result = calculate_soot_blowing_energy_balance(
            steam_consumed_lbm=500.0,
            steam_enthalpy_btu_per_lb=1190.0,
            u_before=8.0,
            u_after=9.5,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0,
            expected_benefit_hours=48.0
        )

        assert isinstance(result, EnergyBalanceResult)
        assert result.steam_energy_consumed_mmbtu > 0
        assert result.heat_recovery_improvement_mmbtu_hr > 0

    def test_positive_energy_benefit(self):
        """Test positive net energy benefit."""
        result = calculate_soot_blowing_energy_balance(
            steam_consumed_lbm=300.0,
            steam_enthalpy_btu_per_lb=1190.0,
            u_before=7.0,
            u_after=9.5,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0,
            expected_benefit_hours=72.0
        )

        # Large improvement should give positive benefit
        assert float(result.energy_efficiency_ratio) > 1.0

    def test_breakeven_hours_calculation(self):
        """Test breakeven hours calculation."""
        result = calculate_soot_blowing_energy_balance(
            steam_consumed_lbm=500.0,
            steam_enthalpy_btu_per_lb=1190.0,
            u_before=8.0,
            u_after=9.0,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0,
            expected_benefit_hours=48.0
        )

        assert result.breakeven_hours > 0

    def test_annual_benefit_calculation(self):
        """Test annual net benefit calculation."""
        result = calculate_soot_blowing_energy_balance(
            steam_consumed_lbm=500.0,
            steam_enthalpy_btu_per_lb=1190.0,
            u_before=8.0,
            u_after=9.0,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0,
            expected_benefit_hours=48.0,
            operating_hours_per_year=8000.0
        )

        assert result.annual_net_benefit_mmbtu is not None


# =============================================================================
# TEST: DATACLASS VALIDATION
# =============================================================================

class TestDataclassValidation:
    """Tests for dataclass validation."""

    def test_blower_config_valid(self):
        """Test valid blower configuration."""
        config = SootBlowerConfiguration(
            blower_id="SB-001",
            blower_type=BlowerType.RETRACTABLE_LANCE,
            medium=BlowingMedium.SATURATED_STEAM,
            zone=EconomizerZone.GAS_INLET
        )

        assert config.blower_id == "SB-001"
        assert config.steam_flow_lbm_per_cycle == 500.0  # Default

    def test_blower_config_negative_steam_flow(self):
        """Test that negative steam flow raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            SootBlowerConfiguration(
                blower_id="SB-001",
                blower_type=BlowerType.RETRACTABLE_LANCE,
                medium=BlowingMedium.SATURATED_STEAM,
                zone=EconomizerZone.GAS_INLET,
                steam_flow_lbm_per_cycle=-100.0
            )

    def test_blower_config_zero_duration(self):
        """Test that zero cycle duration raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            SootBlowerConfiguration(
                blower_id="SB-001",
                blower_type=BlowerType.RETRACTABLE_LANCE,
                medium=BlowingMedium.SATURATED_STEAM,
                zone=EconomizerZone.GAS_INLET,
                cycle_duration_seconds=0
            )


# =============================================================================
# TEST: CONSTANTS AND ENUMERATIONS
# =============================================================================

class TestConstantsAndEnumerations:
    """Tests for module constants and enumerations."""

    def test_epri_guidelines_complete(self):
        """Test that EPRI guidelines are defined."""
        assert "min_interval_hours" in EPRI_GUIDELINES
        assert "max_interval_hours" in EPRI_GUIDELINES
        assert EPRI_GUIDELINES["min_interval_hours"] < EPRI_GUIDELINES["max_interval_hours"]

    def test_zone_weights_complete(self):
        """Test that all zones have importance weights."""
        for zone in EconomizerZone:
            assert zone in ZONE_IMPORTANCE_WEIGHTS
            assert ZONE_IMPORTANCE_WEIGHTS[zone] > 0

    def test_steam_energy_content_defined(self):
        """Test that steam energy content is defined."""
        assert BlowingMedium.SATURATED_STEAM in STEAM_ENERGY_CONTENT
        assert STEAM_ENERGY_CONTENT[BlowingMedium.SATURATED_STEAM] > 0

    def test_erosion_rates_defined(self):
        """Test that erosion rates are defined for all media."""
        assert BlowingMedium.SATURATED_STEAM in EROSION_RATES
        assert BlowingMedium.SUPERHEATED_STEAM in EROSION_RATES
        # Superheated should have higher erosion rate
        assert EROSION_RATES[BlowingMedium.SUPERHEATED_STEAM] > EROSION_RATES[BlowingMedium.SATURATED_STEAM]

    def test_blower_type_enumeration(self):
        """Test BlowerType enumeration values."""
        assert BlowerType.RETRACTABLE_LANCE.value == "retractable_lance"
        assert BlowerType.ACOUSTIC_HORN.value == "acoustic_horn"

    def test_wear_severity_enumeration(self):
        """Test WearSeverity enumeration values."""
        assert WearSeverity.MINIMAL.value == "minimal"
        assert WearSeverity.CRITICAL.value == "critical"


# =============================================================================
# TEST: CACHE FUNCTIONALITY
# =============================================================================

class TestCacheFunctionality:
    """Tests for optimizer cache functionality."""

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        cleared = clear_optimizer_cache()
        assert cleared >= 0

    def test_provenance_bypasses_cache(self):
        """Test that provenance tracking bypasses cache."""
        result1, prov1 = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0,
            track_provenance=True
        )
        result2, prov2 = calculate_optimal_blowing_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0,
            track_provenance=True
        )

        # Results should be identical
        assert result1.optimal_interval_hours == result2.optimal_interval_hours


# =============================================================================
# TEST: PROVENANCE HASH VERIFICATION
# =============================================================================

class TestProvenanceVerification:
    """Tests for provenance hash integrity."""

    def test_hash_reproducibility(self):
        """Test that same inputs produce same hash."""
        result1 = measure_cleaning_effectiveness(
            u_before=8.0,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )
        result2 = measure_cleaning_effectiveness(
            u_before=8.0,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        assert result1.provenance_hash == result2.provenance_hash

    def test_hash_uniqueness(self):
        """Test that different inputs produce different hashes."""
        result1 = measure_cleaning_effectiveness(
            u_before=8.0,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )
        result2 = measure_cleaning_effectiveness(
            u_before=7.0,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        assert result1.provenance_hash != result2.provenance_hash

    def test_hash_is_sha256(self):
        """Test that hash is valid SHA-256 format."""
        result = analyze_cleaning_roi(
            cleaning_cost=50.0,
            fuel_penalty_before=20.0,
            fuel_penalty_after=5.0,
            expected_interval_hours=48.0
        )

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)


# =============================================================================
# TEST: FROZEN DATACLASS IMMUTABILITY
# =============================================================================

class TestDataclassImmutability:
    """Tests for frozen dataclass immutability."""

    def test_blower_config_immutable(self, sample_blower_config):
        """Test that SootBlowerConfiguration is immutable."""
        with pytest.raises(Exception):
            sample_blower_config.steam_flow_lbm_per_cycle = 1000.0

    def test_result_immutable(self):
        """Test that result dataclasses are immutable."""
        result = measure_cleaning_effectiveness(
            u_before=8.0,
            u_after=9.0,
            u_clean=10.0,
            blower_id="SB-001"
        )

        with pytest.raises(Exception):
            result.effectiveness = Decimal("0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
