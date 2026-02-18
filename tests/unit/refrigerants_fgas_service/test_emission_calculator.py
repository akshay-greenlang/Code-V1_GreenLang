# -*- coding: utf-8 -*-
"""
Unit tests for EmissionCalculatorEngine - AGENT-MRV-002 Engine 2

Tests equipment-based, mass-balance, screening, direct measurement,
and top-down calculation methodologies with provenance tracking.

Target: 60+ tests, 700+ lines.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List

import pytest

from greenlang.refrigerants_fgas.emission_calculator import (
    EmissionCalculatorEngine,
)
from greenlang.refrigerants_fgas.models import (
    CalculationMethod,
    CalculationResult,
    BatchCalculationResponse,
    EquipmentProfile,
    EquipmentType,
    EquipmentStatus,
    GasEmission,
    MassBalanceData,
    RefrigerantType,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> EmissionCalculatorEngine:
    """Create a fresh EmissionCalculatorEngine."""
    return EmissionCalculatorEngine()


@pytest.fixture
def single_profile() -> EquipmentProfile:
    """Single active equipment profile."""
    return EquipmentProfile(
        equipment_id="eq_test_001",
        equipment_type=EquipmentType.COMMERCIAL_AC,
        refrigerant_type=RefrigerantType.R_410A,
        charge_kg=10.0,
        equipment_count=1,
        status=EquipmentStatus.ACTIVE,
    )


@pytest.fixture
def gwp_values() -> Dict[RefrigerantType, Decimal]:
    """Sample GWP values for tests."""
    return {
        RefrigerantType.R_410A: Decimal("2088"),
        RefrigerantType.R_134A: Decimal("1530"),
        RefrigerantType.R_32: Decimal("771"),
        RefrigerantType.SF6_GAS: Decimal("25200"),
    }


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestEmissionCalculatorInit:
    """Tests for engine initialization."""

    def test_initialization(self):
        """Engine initializes with defaults."""
        engine = EmissionCalculatorEngine()
        assert engine is not None
        assert len(engine) == 0

    def test_custom_config(self):
        """Engine accepts custom configuration."""
        engine = EmissionCalculatorEngine(config={"precision": 6, "gwp_source": "AR5"})
        assert engine._precision == 6
        assert engine._gwp_source_label == "AR5"

    def test_repr(self, engine: EmissionCalculatorEngine):
        """repr includes precision and count."""
        r = repr(engine)
        assert "EmissionCalculatorEngine" in r
        assert "precision=" in r


# ===========================================================================
# Test: Equipment-Based Calculation
# ===========================================================================


class TestEquipmentBased:
    """Tests for equipment-based emission calculations."""

    def test_equipment_based_single(
        self,
        engine: EmissionCalculatorEngine,
        single_profile: EquipmentProfile,
        gwp_values: Dict[RefrigerantType, Decimal],
    ):
        """Single equipment: charge * count * leak_rate * GWP / 1000."""
        results = engine.calculate_equipment_based(
            profiles=[single_profile],
            gwp_values=gwp_values,
        )
        assert len(results) == 1
        e = results[0]
        assert isinstance(e, GasEmission)
        assert e.refrigerant_type == RefrigerantType.R_410A
        assert e.loss_kg > 0
        assert e.emissions_tco2e > 0

    def test_equipment_based_multiple_units(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Equipment count multiplies the loss."""
        profile = EquipmentProfile(
            equipment_id="eq_multi",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
            equipment_count=5,
            status=EquipmentStatus.ACTIVE,
        )
        results = engine.calculate_equipment_based(
            profiles=[profile], gwp_values=gwp_values,
        )
        assert len(results) == 1
        # Loss should be 5x the single-unit loss
        single = EquipmentProfile(
            equipment_id="eq_single",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
            equipment_count=1,
            status=EquipmentStatus.ACTIVE,
        )
        single_results = engine.calculate_equipment_based(
            profiles=[single], gwp_values=gwp_values,
        )
        assert results[0].loss_kg == pytest.approx(
            single_results[0].loss_kg * 5, abs=0.01
        )

    def test_equipment_based_custom_leak_rate(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Custom leak rate from profile overrides default."""
        profile = EquipmentProfile(
            equipment_id="eq_custom_lr",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=100.0,
            equipment_count=1,
            status=EquipmentStatus.ACTIVE,
            custom_leak_rate=0.10,  # 10%
        )
        results = engine.calculate_equipment_based(
            profiles=[profile], gwp_values=gwp_values,
        )
        # loss = 100 * 1 * 0.10 = 10 kg
        assert results[0].loss_kg == pytest.approx(10.0, abs=0.01)

    def test_equipment_based_skips_decommissioned(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Decommissioned equipment is skipped."""
        profile = EquipmentProfile(
            equipment_id="eq_decom",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=50.0,
            status=EquipmentStatus.DECOMMISSIONED,
        )
        results = engine.calculate_equipment_based(
            profiles=[profile], gwp_values=gwp_values,
        )
        assert len(results) == 0

    def test_equipment_based_skips_inactive(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Inactive equipment is skipped."""
        profile = EquipmentProfile(
            equipment_id="eq_inactive",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=50.0,
            status=EquipmentStatus.INACTIVE,
        )
        results = engine.calculate_equipment_based(
            profiles=[profile], gwp_values=gwp_values,
        )
        assert len(results) == 0

    def test_equipment_based_includes_maintenance(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Maintenance-status equipment is included."""
        profile = EquipmentProfile(
            equipment_id="eq_maint",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=50.0,
            status=EquipmentStatus.MAINTENANCE,
        )
        results = engine.calculate_equipment_based(
            profiles=[profile], gwp_values=gwp_values,
        )
        assert len(results) == 1

    def test_equipment_based_missing_gwp_raises(
        self, engine: EmissionCalculatorEngine
    ):
        """Missing GWP for a refrigerant type raises ValueError."""
        profile = EquipmentProfile(
            equipment_id="eq_no_gwp",
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=50.0,
        )
        with pytest.raises(ValueError, match="GWP value not provided"):
            engine.calculate_equipment_based(
                profiles=[profile], gwp_values={},
            )


# ===========================================================================
# Test: Mass Balance Calculation
# ===========================================================================


class TestMassBalance:
    """Tests for mass-balance emission calculations."""

    def test_mass_balance_positive_loss(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Positive loss: BI + Purchases - Sales + Acq - Divest - EI - Cap."""
        data = MassBalanceData(
            refrigerant_type=RefrigerantType.R_134A,
            beginning_inventory_kg=500.0,
            purchases_kg=100.0,
            sales_kg=20.0,
            acquisitions_kg=10.0,
            divestitures_kg=5.0,
            ending_inventory_kg=450.0,
            capacity_change_kg=30.0,
        )
        results = engine.calculate_mass_balance(
            data=[data], gwp_values=gwp_values,
        )
        assert len(results) == 1
        # Loss = 500 + 100 - 20 + 10 - 5 - 450 - 30 = 105
        assert results[0].loss_kg == pytest.approx(105.0, abs=0.01)
        # Emissions = 105 * 1530 / 1000 = 160.65 tCO2e
        assert results[0].emissions_tco2e == pytest.approx(160.65, abs=0.01)

    def test_mass_balance_zero_loss(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """When inventory balances exactly, loss is zero."""
        data = MassBalanceData(
            refrigerant_type=RefrigerantType.R_134A,
            beginning_inventory_kg=500.0,
            purchases_kg=0.0,
            sales_kg=0.0,
            ending_inventory_kg=500.0,
        )
        results = engine.calculate_mass_balance(
            data=[data], gwp_values=gwp_values,
        )
        assert results[0].loss_kg == pytest.approx(0.0, abs=0.001)
        assert results[0].emissions_tco2e == pytest.approx(0.0, abs=0.001)

    def test_mass_balance_negative_clamped_to_zero(
        self, engine: EmissionCalculatorEngine, gwp_values
    ):
        """Negative loss (net accumulation) is clamped to zero."""
        data = MassBalanceData(
            refrigerant_type=RefrigerantType.R_134A,
            beginning_inventory_kg=100.0,
            purchases_kg=0.0,
            ending_inventory_kg=200.0,
        )
        results = engine.calculate_mass_balance(
            data=[data], gwp_values=gwp_values,
        )
        # raw loss = 100 - 200 = -100, clamped to 0
        assert results[0].loss_kg == 0.0
        assert results[0].emissions_tco2e == 0.0

    def test_mass_balance_missing_gwp_raises(
        self, engine: EmissionCalculatorEngine
    ):
        """Missing GWP raises ValueError."""
        data = MassBalanceData(
            refrigerant_type=RefrigerantType.R_134A,
            beginning_inventory_kg=100.0,
            ending_inventory_kg=50.0,
        )
        with pytest.raises(ValueError, match="GWP value not provided"):
            engine.calculate_mass_balance(data=[data], gwp_values={})


# ===========================================================================
# Test: Screening Calculation
# ===========================================================================


class TestScreening:
    """Tests for screening emission calculations."""

    def test_screening_calculation(self, engine: EmissionCalculatorEngine):
        """Screening: charge * leak_rate * GWP / 1000."""
        results = engine.calculate_screening(
            total_charge_kg=Decimal("200"),
            ref_type=RefrigerantType.R_404A,
            equip_type=EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED,
            leak_rate=Decimal("0.20"),
            gwp=Decimal("3922"),
        )
        assert len(results) == 1
        # Loss = 200 * 0.20 = 40 kg
        assert results[0].loss_kg == pytest.approx(40.0, abs=0.01)
        # Emissions = 40 * 3922 / 1000 = 156.88 tCO2e
        assert results[0].emissions_tco2e == pytest.approx(156.88, abs=0.01)

    def test_screening_zero_charge_raises(self, engine: EmissionCalculatorEngine):
        """Zero charge raises ValueError."""
        with pytest.raises(ValueError, match="total_charge_kg must be > 0"):
            engine.calculate_screening(
                total_charge_kg=Decimal("0"),
                ref_type=RefrigerantType.R_134A,
                equip_type=EquipmentType.COMMERCIAL_AC,
                leak_rate=Decimal("0.06"),
                gwp=Decimal("1530"),
            )

    def test_screening_negative_gwp_raises(self, engine: EmissionCalculatorEngine):
        """Negative GWP raises ValueError."""
        with pytest.raises(ValueError, match="gwp must be >= 0"):
            engine.calculate_screening(
                total_charge_kg=Decimal("100"),
                ref_type=RefrigerantType.R_134A,
                equip_type=EquipmentType.COMMERCIAL_AC,
                leak_rate=Decimal("0.06"),
                gwp=Decimal("-1"),
            )

    def test_screening_invalid_leak_rate_raises(
        self, engine: EmissionCalculatorEngine
    ):
        """Leak rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="leak_rate must be in"):
            engine.calculate_screening(
                total_charge_kg=Decimal("100"),
                ref_type=RefrigerantType.R_134A,
                equip_type=EquipmentType.COMMERCIAL_AC,
                leak_rate=Decimal("1.5"),
                gwp=Decimal("1530"),
            )


# ===========================================================================
# Test: Direct Measurement Calculation
# ===========================================================================


class TestDirectMeasurement:
    """Tests for direct measurement calculations."""

    def test_direct_measurement(self, engine: EmissionCalculatorEngine):
        """Direct measurement: loss * GWP / 1000."""
        results = engine.calculate_direct_measurement(
            loss_kg=Decimal("5.0"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        assert len(results) == 1
        assert results[0].loss_kg == pytest.approx(5.0, abs=0.001)
        # 5 * 1530 / 1000 = 7.65
        assert results[0].emissions_tco2e == pytest.approx(7.65, abs=0.001)

    def test_direct_measurement_negative_loss_raises(
        self, engine: EmissionCalculatorEngine
    ):
        """Negative loss raises ValueError."""
        with pytest.raises(ValueError, match="loss_kg must be >= 0"):
            engine.calculate_direct_measurement(
                loss_kg=Decimal("-1"),
                ref_type=RefrigerantType.R_134A,
                gwp=Decimal("1530"),
            )

    def test_direct_measurement_zero_loss(self, engine: EmissionCalculatorEngine):
        """Zero loss produces zero emissions."""
        results = engine.calculate_direct_measurement(
            loss_kg=Decimal("0"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        assert results[0].emissions_tco2e == 0.0


# ===========================================================================
# Test: Top-Down Calculation
# ===========================================================================


class TestTopDown:
    """Tests for top-down emission calculations."""

    def test_top_down_calculation(self, engine: EmissionCalculatorEngine):
        """Top-down: (purchases - recovered) * GWP / 1000."""
        results = engine.calculate_top_down(
            purchases_kg=Decimal("100"),
            recovered_kg=Decimal("30"),
            ref_type=RefrigerantType.R_410A,
            gwp=Decimal("2088"),
        )
        assert len(results) == 1
        # net loss = 100 - 30 = 70
        assert results[0].loss_kg == pytest.approx(70.0, abs=0.01)
        # 70 * 2088 / 1000 = 146.16
        assert results[0].emissions_tco2e == pytest.approx(146.16, abs=0.01)

    def test_top_down_recovered_exceeds_purchases(
        self, engine: EmissionCalculatorEngine
    ):
        """When recovered > purchases, net loss clamped to zero."""
        results = engine.calculate_top_down(
            purchases_kg=Decimal("30"),
            recovered_kg=Decimal("50"),
            ref_type=RefrigerantType.R_410A,
            gwp=Decimal("2088"),
        )
        assert results[0].loss_kg == 0.0
        assert results[0].emissions_tco2e == 0.0

    def test_top_down_negative_purchases_raises(
        self, engine: EmissionCalculatorEngine
    ):
        """Negative purchases raises ValueError."""
        with pytest.raises(ValueError, match="purchases_kg must be >= 0"):
            engine.calculate_top_down(
                purchases_kg=Decimal("-10"),
                recovered_kg=Decimal("0"),
                ref_type=RefrigerantType.R_134A,
                gwp=Decimal("1530"),
            )


# ===========================================================================
# Test: Unified calculate() Entry Point
# ===========================================================================


class TestUnifiedCalculate:
    """Tests for the calculate() dispatch method."""

    def test_calculate_screening_via_dispatch(
        self, engine: EmissionCalculatorEngine
    ):
        """calculate() dispatches to screening correctly."""
        result = engine.calculate(
            method=CalculationMethod.SCREENING,
            total_charge_kg=Decimal("100"),
            ref_type=RefrigerantType.R_134A,
            equip_type=EquipmentType.COMMERCIAL_AC,
            leak_rate=Decimal("0.06"),
            gwp=Decimal("1530"),
        )
        assert isinstance(result, CalculationResult)
        assert result.method == CalculationMethod.SCREENING
        assert result.total_emissions_tco2e > 0

    def test_calculate_produces_provenance_hash(
        self, engine: EmissionCalculatorEngine
    ):
        """calculate() produces a non-empty provenance hash."""
        result = engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("5"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_calculate_produces_trace(self, engine: EmissionCalculatorEngine):
        """calculate() produces a non-empty calculation trace."""
        result = engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("5"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        assert len(result.calculation_trace) > 0

    def test_calculate_tco2e_conversion(self, engine: EmissionCalculatorEngine):
        """total_emissions_tco2e = total loss * GWP / 1000."""
        result = engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("10"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        # 10 * 1530 / 1000 = 15.3
        assert result.total_emissions_tco2e == pytest.approx(15.3, abs=0.01)


# ===========================================================================
# Test: Batch Calculation
# ===========================================================================


class TestBatchCalculation:
    """Tests for batch emission calculations."""

    def test_batch_calculation(self, engine: EmissionCalculatorEngine):
        """Batch processes multiple inputs."""
        inputs = [
            {
                "method": CalculationMethod.DIRECT_MEASUREMENT,
                "loss_kg": Decimal("5"),
                "ref_type": RefrigerantType.R_134A,
                "gwp": Decimal("1530"),
            },
            {
                "method": CalculationMethod.DIRECT_MEASUREMENT,
                "loss_kg": Decimal("10"),
                "ref_type": RefrigerantType.R_134A,
                "gwp": Decimal("1530"),
            },
        ]
        response = engine.calculate_batch(inputs)
        assert isinstance(response, BatchCalculationResponse)
        assert response.success_count == 2
        assert response.failure_count == 0
        assert response.total_emissions_tco2e > 0
        assert len(response.provenance_hash) == 64

    def test_batch_with_failure(self, engine: EmissionCalculatorEngine):
        """Batch handles partial failures gracefully."""
        inputs = [
            {
                "method": CalculationMethod.DIRECT_MEASUREMENT,
                "loss_kg": Decimal("5"),
                "ref_type": RefrigerantType.R_134A,
                "gwp": Decimal("1530"),
            },
            {
                "method": CalculationMethod.SCREENING,
                # Missing required params
            },
        ]
        response = engine.calculate_batch(inputs)
        assert response.success_count == 1
        assert response.failure_count == 1


# ===========================================================================
# Test: Aggregation
# ===========================================================================


class TestAggregation:
    """Tests for result aggregation with control approaches."""

    def test_aggregate_operational_control(
        self, engine: EmissionCalculatorEngine
    ):
        """Operational control applies 100% share."""
        result1 = engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("10"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        agg = engine.aggregate_results(
            results=[result1],
            control_approach="operational",
            share=Decimal("1.0"),
        )
        assert Decimal(agg["total_emissions_tco2e"]) == Decimal(
            str(result1.total_emissions_tco2e)
        )

    def test_aggregate_equity_share(self, engine: EmissionCalculatorEngine):
        """Equity share reduces proportionally."""
        result1 = engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("10"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        agg = engine.aggregate_results(
            results=[result1],
            control_approach="equity",
            share=Decimal("0.5"),
        )
        # 50% of 15.3 = 7.65
        assert float(Decimal(agg["total_emissions_tco2e"])) == pytest.approx(
            7.65, abs=0.01
        )

    def test_aggregate_invalid_approach_raises(
        self, engine: EmissionCalculatorEngine
    ):
        """Invalid control approach raises ValueError."""
        with pytest.raises(ValueError, match="control_approach"):
            engine.aggregate_results([], control_approach="invalid")

    def test_aggregate_provenance_hash(self, engine: EmissionCalculatorEngine):
        """Aggregation produces a provenance hash."""
        agg = engine.aggregate_results([], control_approach="operational")
        assert len(agg["provenance_hash"]) == 64


# ===========================================================================
# Test: History and Stats
# ===========================================================================


class TestHistoryAndStats:
    """Tests for calculation history and statistics."""

    def test_history_grows(self, engine: EmissionCalculatorEngine):
        """History count increases with calculations."""
        assert len(engine.get_history()) == 0
        engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("1"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        assert len(engine.get_history()) == 1

    def test_clear_resets_history(self, engine: EmissionCalculatorEngine):
        """clear() removes all history."""
        engine.calculate(
            method=CalculationMethod.DIRECT_MEASUREMENT,
            loss_kg=Decimal("1"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        engine.clear()
        assert len(engine) == 0

    def test_get_stats(self, engine: EmissionCalculatorEngine):
        """get_stats() returns expected keys."""
        stats = engine.get_stats()
        assert "total_calculations" in stats
        assert "precision" in stats
        assert "gwp_source" in stats


# ===========================================================================
# Test: Decimal Precision
# ===========================================================================


class TestDecimalPrecision:
    """Tests for bit-perfect decimal arithmetic."""

    def test_gwp_application_correctness(self, engine: EmissionCalculatorEngine):
        """GWP multiplication is precise to configured precision."""
        results = engine.calculate_direct_measurement(
            loss_kg=Decimal("1.123456789"),
            ref_type=RefrigerantType.R_134A,
            gwp=Decimal("1530"),
        )
        # loss_kg rounded to 3dp = 1.123
        # emissions_kg = 1.123 * 1530 = 1718.190
        # emissions_tco2e = 1718.190 / 1000 = 1.718 (3dp)
        assert results[0].loss_kg == pytest.approx(1.123, abs=0.001)
        assert results[0].emissions_tco2e == pytest.approx(1.718, abs=0.001)
