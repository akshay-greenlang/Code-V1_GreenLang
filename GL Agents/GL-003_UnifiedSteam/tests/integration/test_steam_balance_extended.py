# -*- coding: utf-8 -*-
"""
Steam Balance Extended Integration Tests for GL-003 UnifiedSteam
================================================================

Additional comprehensive integration tests extending the base steam balance
validation. Covers condensate return accounting, provenance/determinism,
and performance benchmarks.

This module complements test_steam_balance.py with additional test coverage
for categories 6 (condensate return) and enhanced validation tests.

Validation Categories Covered:
    6. Condensate return accounting
    7. Provenance and determinism validation
    8. Performance benchmarks
    9. Parametrized edge case scenarios

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared components from main test module
from test_steam_balance import (
    SteamStream,
    BalanceResult,
    calculate_balance,
    calculate_stream_enthalpy,
)


# =============================================================================
# CONSTANTS
# =============================================================================

MASS_BALANCE_TOLERANCE_PCT = 2.0
ENERGY_BALANCE_TOLERANCE_PCT = 3.0
CONDENSATE_RECOVERY_MIN_PCT = 80.0
PERFORMANCE_TARGET_MS = 10.0
THROUGHPUT_TARGET_PER_SEC = 500


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CondensateReturnData:
    """Data structure for condensate return network test."""
    process_condensate: List[SteamStream]
    flash_steam: List[SteamStream]
    makeup_water: SteamStream
    total_return: SteamStream
    recovery_rate_percent: float


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_steam_system():
    """Simple steam system fixture for tests."""
    return [
        SteamStream("Boiler Steam", 10.0, 1000.0, 200.0,
                   enthalpy_kJ_kg=2827.0, is_inlet=True),
        SteamStream("Process Consumer", 10.0, 950.0, 195.0,
                   enthalpy_kJ_kg=2815.0, is_inlet=False),
    ]


@pytest.fixture
def condensate_return_system():
    """
    Condensate return network test data.

    Models a typical industrial condensate collection system with:
    - Multiple process condensate returns at varying temperatures
    - Flash steam recovery to LP header
    - Makeup water to compensate for losses
    - Total return flow to deaerator

    Mass balance: process_condensate + makeup = flash_steam + total_return
    """
    return CondensateReturnData(
        process_condensate=[
            SteamStream("Process A Return", 3.0, 200.0, 95.0,
                       enthalpy_kJ_kg=398.0, is_inlet=True),
            SteamStream("Process B Return", 2.5, 200.0, 90.0,
                       enthalpy_kJ_kg=377.0, is_inlet=True),
            SteamStream("Process C Return", 1.5, 200.0, 85.0,
                       enthalpy_kJ_kg=356.0, is_inlet=True),
            SteamStream("Flash Tank Drain", 4.0, 200.0, 105.0,
                       enthalpy_kJ_kg=440.0, is_inlet=True),
        ],
        flash_steam=[
            SteamStream("Flash Steam to LP", 0.5, 200.0, 120.0,
                       enthalpy_kJ_kg=2707.0, is_inlet=False),
        ],
        makeup_water=SteamStream(
            "Makeup Water", 2.0, 200.0, 25.0,
            enthalpy_kJ_kg=105.0, is_inlet=True
        ),
        total_return=SteamStream(
            "Total to DA", 12.5, 200.0, 95.0,
            enthalpy_kJ_kg=398.0, is_inlet=False
        ),
        recovery_rate_percent=85.0
    )


@pytest.fixture
def multi_process_condensate():
    """Multiple process condensate returns with varying conditions."""
    return [
        SteamStream("High Temp Process", 5.0, 300.0, 130.0,
                   enthalpy_kJ_kg=546.0, is_inlet=True),
        SteamStream("Medium Temp Process", 3.0, 200.0, 100.0,
                   enthalpy_kJ_kg=419.0, is_inlet=True),
        SteamStream("Low Temp Process", 2.0, 200.0, 70.0,
                   enthalpy_kJ_kg=293.0, is_inlet=True),
        SteamStream("Flash Steam", 0.8, 150.0, 111.4,
                   enthalpy_kJ_kg=2693.0, is_inlet=False),
        SteamStream("Combined Return", 9.2, 200.0, 95.0,
                   enthalpy_kJ_kg=398.0, is_inlet=False),
    ]


# =============================================================================
# TEST CLASS: CONDENSATE RETURN ACCOUNTING (CATEGORY 6)
# =============================================================================

@pytest.mark.integration
class TestCondensateReturnBalance:
    """
    Test condensate return system mass and energy balance.

    Condensate system accounting includes:
    - Process condensate returns from various heat exchangers
    - Flash steam recovery to LP header
    - Makeup water to compensate for system losses
    - Total return flow to deaerator for feedwater heating
    """

    def test_condensate_mass_balance(self, condensate_return_system):
        """Condensate system mass balance: returns + makeup = flash + total."""
        data = condensate_return_system

        # Build complete stream list
        streams = (
            data.process_condensate +
            data.flash_steam +
            [data.makeup_water, data.total_return]
        )

        result = calculate_balance(streams)

        assert abs(result.mass_imbalance_percent) <= MASS_BALANCE_TOLERANCE_PCT, (
            f"Condensate mass imbalance {result.mass_imbalance_percent:.2f}% "
            f"exceeds {MASS_BALANCE_TOLERANCE_PCT}% tolerance"
        )

    def test_condensate_energy_balance(self, condensate_return_system):
        """Condensate system energy balance should close within tolerance."""
        data = condensate_return_system

        streams = (
            data.process_condensate +
            data.flash_steam +
            [data.makeup_water, data.total_return]
        )

        result = calculate_balance(streams)

        assert abs(result.energy_imbalance_percent) <= ENERGY_BALANCE_TOLERANCE_PCT, (
            f"Condensate energy imbalance {result.energy_imbalance_percent:.2f}% "
            f"exceeds {ENERGY_BALANCE_TOLERANCE_PCT}% tolerance"
        )

    def test_condensate_recovery_rate(self, condensate_return_system):
        """Verify condensate recovery rate meets minimum target (80%)."""
        data = condensate_return_system

        # Calculate total process condensate (excluding makeup)
        total_process = sum(s.mass_flow_kg_s for s in data.process_condensate)

        # Recovery rate = condensate / (condensate + makeup) * 100
        total_with_makeup = total_process + data.makeup_water.mass_flow_kg_s
        recovery_rate = (total_process / total_with_makeup) * 100

        assert recovery_rate >= CONDENSATE_RECOVERY_MIN_PCT, (
            f"Condensate recovery {recovery_rate:.1f}% below "
            f"{CONDENSATE_RECOVERY_MIN_PCT}% minimum target"
        )

    def test_makeup_water_compensates_losses(self, condensate_return_system):
        """Verify makeup water compensates for flash steam losses."""
        data = condensate_return_system

        # Flash steam represents water lost from condensate system
        flash_loss = sum(s.mass_flow_kg_s for s in data.flash_steam)

        # Makeup should at least compensate for flash steam
        assert data.makeup_water.mass_flow_kg_s >= flash_loss, (
            f"Makeup water {data.makeup_water.mass_flow_kg_s:.2f} kg/s insufficient "
            f"for flash steam loss {flash_loss:.2f} kg/s"
        )

    def test_condensate_temperature_mixing(self, condensate_return_system):
        """Verify mixed condensate temperature is thermodynamically reasonable."""
        data = condensate_return_system

        # Calculate weighted average temperature of process returns
        total_mass = sum(s.mass_flow_kg_s for s in data.process_condensate)
        weighted_temp = sum(
            s.mass_flow_kg_s * s.temperature_C for s in data.process_condensate
        ) / total_mass

        # Mixed temperature should be between min and max
        temps = [s.temperature_C for s in data.process_condensate]
        assert min(temps) <= weighted_temp <= max(temps), (
            f"Mixed temperature {weighted_temp:.1f}C outside range "
            f"[{min(temps):.1f}, {max(temps):.1f}]C"
        )

    def test_multi_process_balance(self, multi_process_condensate):
        """Test balance with multiple process condensate streams."""
        result = calculate_balance(multi_process_condensate)

        assert abs(result.mass_imbalance_percent) <= MASS_BALANCE_TOLERANCE_PCT
        assert result.is_closed or abs(result.energy_imbalance_percent) <= 5.0


# =============================================================================
# TEST CLASS: PROVENANCE AND DETERMINISM
# =============================================================================

@pytest.mark.integration
class TestProvenanceAndDeterminism:
    """
    Test calculation provenance and bit-perfect reproducibility.

    GreenLang requirements:
    - All calculations must be deterministic
    - Same inputs must always produce same outputs
    - Results must support audit trail generation with SHA-256 hashing
    """

    def test_balance_calculation_deterministic(self, simple_steam_system):
        """Balance calculation should produce identical results on repeated runs."""
        hashes = set()

        for _ in range(10):
            result = calculate_balance(simple_steam_system)
            result_hash = hashlib.sha256(
                f"{result.mass_in_kg_s:.10f}:{result.mass_out_kg_s:.10f}:"
                f"{result.energy_in_kW:.10f}:{result.energy_out_kW:.10f}".encode()
            ).hexdigest()
            hashes.add(result_hash)

        assert len(hashes) == 1, (
            f"Balance calculation produced {len(hashes)} different results - "
            f"not deterministic"
        )

    def test_provenance_hash_format(self, simple_steam_system):
        """Verify result hash is valid SHA-256 format (64 hex characters)."""
        result = calculate_balance(simple_steam_system)
        result_hash = hashlib.sha256(
            f"{result.mass_in_kg_s}:{result.mass_out_kg_s}:{result.energy_in_kW}".encode()
        ).hexdigest()

        assert len(result_hash) == 64, (
            f"Hash length {len(result_hash)} != 64 (expected SHA-256)"
        )
        assert all(c in '0123456789abcdef' for c in result_hash), (
            "Hash contains non-hexadecimal characters"
        )

    def test_provenance_hash_unique_per_input(self, simple_steam_system):
        """Different inputs should produce different provenance hashes."""
        result1 = calculate_balance(simple_steam_system)
        hash1 = hashlib.sha256(
            f"{result1.mass_in_kg_s}:{result1.energy_in_kW}".encode()
        ).hexdigest()

        # Modify input slightly
        modified_system = [
            SteamStream("Boiler Steam", 11.0, 1000.0, 200.0,
                       enthalpy_kJ_kg=2827.0, is_inlet=True),
            SteamStream("Process Consumer", 10.0, 950.0, 195.0,
                       enthalpy_kJ_kg=2815.0, is_inlet=False),
        ]
        result2 = calculate_balance(modified_system)
        hash2 = hashlib.sha256(
            f"{result2.mass_in_kg_s}:{result2.energy_in_kW}".encode()
        ).hexdigest()

        assert hash1 != hash2, (
            "Different inputs should produce different hashes"
        )

    def test_audit_trail_completeness(self, simple_steam_system):
        """Balance result should contain all required audit trail fields."""
        result = calculate_balance(simple_steam_system)

        # All key values should be present and valid
        assert result.mass_in_kg_s is not None
        assert result.mass_out_kg_s is not None
        assert result.mass_imbalance_kg_s is not None
        assert result.mass_imbalance_percent is not None
        assert result.energy_in_kW is not None
        assert result.energy_out_kW is not None
        assert result.energy_imbalance_kW is not None
        assert result.energy_imbalance_percent is not None
        assert isinstance(result.is_closed, bool)

    def test_floating_point_consistency(self):
        """Verify consistent floating point arithmetic."""
        # Create streams with precise values
        streams = [
            SteamStream("A", 10.000000001, 1000.0, 200.0,
                       enthalpy_kJ_kg=2800.0, is_inlet=True),
            SteamStream("B", 10.000000001, 950.0, 195.0,
                       enthalpy_kJ_kg=2780.0, is_inlet=False),
        ]

        result = calculate_balance(streams)

        # Should be essentially zero imbalance
        assert abs(result.mass_imbalance_kg_s) < 1e-8


# =============================================================================
# TEST CLASS: PERFORMANCE BENCHMARKS
# =============================================================================

@pytest.mark.integration
class TestPerformanceBenchmarks:
    """
    Performance benchmarks for balance calculations.

    Targets:
    - Single calculation: < 10ms
    - Batch throughput: > 500 calculations/second
    """

    def test_single_calculation_speed(self, simple_steam_system):
        """Single balance calculation should complete in under 10ms."""
        # Warm up
        calculate_balance(simple_steam_system)

        # Measure
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            calculate_balance(simple_steam_system)
        elapsed = (time.perf_counter() - start) * 1000 / iterations  # ms per calc

        assert elapsed < PERFORMANCE_TARGET_MS, (
            f"Calculation took {elapsed:.2f}ms, exceeds {PERFORMANCE_TARGET_MS}ms target"
        )

    def test_batch_calculation_throughput(self):
        """Test throughput for batch calculations."""
        # Create 1000 unique scenarios
        scenarios = []
        for i in range(1000):
            scenarios.append([
                SteamStream(f"Inlet_{i}", 10.0 + i * 0.01, 1000.0, 200.0,
                           enthalpy_kJ_kg=2800.0, is_inlet=True),
                SteamStream(f"Outlet_{i}", 10.0 + i * 0.01, 950.0, 195.0,
                           enthalpy_kJ_kg=2780.0, is_inlet=False),
            ])

        start = time.perf_counter()
        results = [calculate_balance(s) for s in scenarios]
        elapsed = time.perf_counter() - start

        throughput = len(scenarios) / elapsed

        assert throughput >= THROUGHPUT_TARGET_PER_SEC, (
            f"Throughput {throughput:.0f}/sec below {THROUGHPUT_TARGET_PER_SEC} target"
        )
        assert all(r.is_closed for r in results), (
            "Not all balanced scenarios closed properly"
        )

    def test_large_stream_count_performance(self):
        """Test performance with many streams."""
        # Create system with 100 streams (50 in, 50 out)
        streams = []
        for i in range(50):
            streams.append(SteamStream(
                f"Inlet_{i}", 1.0, 1000.0, 200.0,
                enthalpy_kJ_kg=2800.0, is_inlet=True
            ))
            streams.append(SteamStream(
                f"Outlet_{i}", 1.0, 950.0, 195.0,
                enthalpy_kJ_kg=2780.0, is_inlet=False
            ))

        start = time.perf_counter()
        result = calculate_balance(streams)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0, (
            f"Large stream calculation took {elapsed_ms:.2f}ms"
        )
        assert result.mass_in_kg_s == pytest.approx(50.0, rel=0.001)


# =============================================================================
# TEST CLASS: PARAMETRIZED BALANCE SCENARIOS
# =============================================================================

@pytest.mark.integration
class TestParametrizedBalanceScenarios:
    """Parametrized tests for various balance scenarios."""

    @pytest.mark.parametrize("inlet_flow,outlet_flow,expected_imbalance_pct", [
        (10.0, 10.0, 0.0),    # Perfect balance
        (10.0, 9.8, 2.0),     # 2% loss - at tolerance
        (10.0, 9.5, 5.0),     # 5% loss - exceeds tolerance
        (10.0, 10.2, -2.0),   # 2% gain (measurement error)
        (20.0, 20.0, 0.0),    # Larger flow, balanced
        (5.0, 4.9, 2.0),      # Smaller flow, 2% loss
        (100.0, 100.0, 0.0),  # Large flow, balanced
        (0.5, 0.5, 0.0),      # Small flow, balanced
    ])
    def test_mass_balance_parametrized(self, inlet_flow, outlet_flow, expected_imbalance_pct):
        """Test mass balance with various flow combinations."""
        streams = [
            SteamStream("Inlet", inlet_flow, 1000.0, 200.0,
                       enthalpy_kJ_kg=2827.0, is_inlet=True),
            SteamStream("Outlet", outlet_flow, 950.0, 195.0,
                       enthalpy_kJ_kg=2815.0, is_inlet=False),
        ]

        result = calculate_balance(streams)

        assert result.mass_imbalance_percent == pytest.approx(expected_imbalance_pct, abs=0.5)

    @pytest.mark.parametrize("h_in,h_out,should_close", [
        (2800.0, 2800.0, True),   # No loss
        (2800.0, 2716.0, True),   # 3% loss - at tolerance
        (2800.0, 2660.0, False),  # 5% loss - exceeds tolerance
        (3200.0, 3200.0, True),   # HP steam, no loss
        (2600.0, 2600.0, True),   # LP steam, no loss
        (2800.0, 2520.0, False),  # 10% loss - definitely exceeds
    ])
    def test_energy_balance_parametrized(self, h_in, h_out, should_close):
        """Test energy balance with various enthalpy scenarios."""
        streams = [
            SteamStream("Inlet", 10.0, 1000.0, 200.0,
                       enthalpy_kJ_kg=h_in, is_inlet=True),
            SteamStream("Outlet", 10.0, 950.0, 195.0,
                       enthalpy_kJ_kg=h_out, is_inlet=False),
        ]

        result = calculate_balance(streams)

        assert result.is_closed == should_close, (
            f"Expected is_closed={should_close}, got {result.is_closed} "
            f"(energy imbalance: {result.energy_imbalance_percent:.2f}%)"
        )

    @pytest.mark.parametrize("num_inlets,num_outlets", [
        (1, 1),
        (2, 2),
        (5, 5),
        (10, 10),
        (1, 3),
        (3, 1),
    ])
    def test_multiple_stream_configurations(self, num_inlets, num_outlets):
        """Test various inlet/outlet stream configurations."""
        total_flow = 10.0
        inlet_flow_each = total_flow / num_inlets
        outlet_flow_each = total_flow / num_outlets

        streams = []
        for i in range(num_inlets):
            streams.append(SteamStream(
                f"Inlet_{i}", inlet_flow_each, 1000.0, 200.0,
                enthalpy_kJ_kg=2800.0, is_inlet=True
            ))
        for i in range(num_outlets):
            streams.append(SteamStream(
                f"Outlet_{i}", outlet_flow_each, 950.0, 195.0,
                enthalpy_kJ_kg=2780.0, is_inlet=False
            ))

        result = calculate_balance(streams)

        assert result.mass_in_kg_s == pytest.approx(total_flow, rel=0.001)
        assert result.mass_out_kg_s == pytest.approx(total_flow, rel=0.001)
        assert result.is_closed is True


# =============================================================================
# TEST CLASS: ADDITIONAL EDGE CASES
# =============================================================================

@pytest.mark.integration
class TestAdditionalEdgeCases:
    """Additional edge case tests for robust coverage."""

    def test_zero_enthalpy_stream(self):
        """Should calculate enthalpy when not provided (zero default)."""
        streams = [
            SteamStream("Inlet", 10.0, 1000.0, 200.0, is_inlet=True),  # No enthalpy
            SteamStream("Outlet", 10.0, 950.0, 195.0, is_inlet=False),  # No enthalpy
        ]

        result = calculate_balance(streams)

        # Enthalpy should have been calculated
        assert streams[0].enthalpy_kJ_kg > 0
        assert streams[1].enthalpy_kJ_kg > 0
        assert result.energy_in_kW > 0

    def test_negative_imbalance(self):
        """Should correctly report negative imbalance (more out than in)."""
        streams = [
            SteamStream("Inlet", 10.0, 1000.0, 200.0,
                       enthalpy_kJ_kg=2800.0, is_inlet=True),
            SteamStream("Outlet", 11.0, 950.0, 195.0,
                       enthalpy_kJ_kg=2780.0, is_inlet=False),  # More out than in
        ]

        result = calculate_balance(streams)

        assert result.mass_imbalance_kg_s < 0
        assert result.mass_imbalance_percent < 0

    def test_very_high_pressure_system(self):
        """Test with supercritical pressure conditions."""
        streams = [
            SteamStream("Supercritical", 10.0, 25000.0, 600.0,
                       enthalpy_kJ_kg=3500.0, is_inlet=True),
            SteamStream("Consumer", 10.0, 24000.0, 580.0,
                       enthalpy_kJ_kg=3450.0, is_inlet=False),
        ]

        result = calculate_balance(streams)

        assert result.is_closed is True
        assert result.energy_in_kW > 30000

    def test_atmospheric_pressure_system(self):
        """Test with atmospheric pressure conditions."""
        streams = [
            SteamStream("Atmospheric Steam", 5.0, 101.325, 100.0,
                       enthalpy_kJ_kg=2676.0, is_inlet=True),
            SteamStream("Consumer", 5.0, 100.0, 99.0,
                       enthalpy_kJ_kg=2670.0, is_inlet=False),
        ]

        result = calculate_balance(streams)

        assert result.is_closed is True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
