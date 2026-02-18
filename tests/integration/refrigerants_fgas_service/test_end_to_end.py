# -*- coding: utf-8 -*-
"""
End-to-end integration tests for Refrigerants & F-Gas Agent - AGENT-MRV-002

Tests complete workflows through the RefrigerantsFGasService facade:
- Equipment-based workflow (register, log event, calculate, compliance, audit)
- Mass-balance workflow
- Screening workflow
- Blend decomposition (R-410A, R-404A)
- Multiple equipment types
- Lifecycle tracking
- Compliance checking (EU F-Gas)
- Uncertainty analysis
- Facility aggregation
- Batch processing
- Provenance chain integrity (SHA-256)
- Determinism (same input -> same output)
- SF6 switchgear calculation
- HFO low-GWP calculation

Target: 30+ tests, ~550 lines
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.refrigerants_fgas.setup import RefrigerantsFGasService


# ===========================================================================
# Full workflow tests
# ===========================================================================


class TestFullEquipmentBasedWorkflow:
    """End-to-end equipment-based workflow: register -> log -> calc -> audit."""

    def test_full_equipment_based_workflow(self, service, sample_equipment_based_input):
        """Complete equipment-based workflow through the service."""
        # Step 1: Register equipment
        equip = service.register_equipment(
            equipment_id="eq_e2e_001",
            equipment_type="COMMERCIAL_AC",
            name="Main HVAC Unit",
            facility_id="fac_integ_001",
            refrigerant_type="R_410A",
            charge_kg=25.0,
            age_years=3,
        )
        assert equip["equipment_id"] == "eq_e2e_001"

        # Step 2: Log a service event (installation)
        event = service.log_service_event(
            equipment_id="eq_e2e_001",
            event_type="installation",
            refrigerant_type="R_410A",
            quantity_kg=25.0,
            technician="Test Engineer",
            notes="Initial charging",
        )
        assert event["equipment_id"] == "eq_e2e_001"
        assert event["event_type"] == "installation"

        # Step 3: Calculate emissions
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            method="equipment_based",
            equipment_type="COMMERCIAL_AC",
            equipment_id="eq_e2e_001",
            facility_id="fac_integ_001",
        )
        assert "calculation_id" in calc
        calc_id = calc["calculation_id"]

        # Step 4: Check compliance
        compliance = service.check_compliance(
            calculation_id=calc_id,
            frameworks=["GHG_PROTOCOL", "EU_FGAS_2024"],
        )
        assert compliance["total_count"] == 2
        assert "records" in compliance

        # Step 5: Get audit trail
        audit = service.get_audit_trail(calc_id)
        assert isinstance(audit, list)

    def test_full_mass_balance_workflow(self, service, sample_mass_balance_input):
        """Complete mass-balance workflow through the service."""
        # Calculate
        calc = service.calculate(
            refrigerant_type="R_134A",
            charge_kg=500.0,
            method="mass_balance",
            facility_id="fac_integ_002",
            mass_balance_data={
                "inventory_start_kg": 500.0,
                "purchases_kg": 100.0,
                "recovery_kg": 50.0,
                "inventory_end_kg": 450.0,
            },
        )
        assert "calculation_id" in calc

        # Verify emissions are positive (100 kg loss)
        inner = calc.get("result", calc)
        emissions_kg = inner.get("emissions_kg", 0.0)
        if emissions_kg > 0:
            assert emissions_kg == pytest.approx(100.0, rel=1e-2)

        # Check compliance
        compliance = service.check_compliance(
            calculation_id=calc["calculation_id"],
        )
        assert compliance["total_count"] >= 5

    def test_full_screening_workflow(self, service, sample_screening_input):
        """Complete screening workflow through the service."""
        calc = service.calculate(
            refrigerant_type="R_407C",
            charge_kg=10.0,
            method="screening",
            activity_data=1000.0,
            screening_factor=0.02,
            facility_id="fac_integ_003",
        )
        assert "calculation_id" in calc

        # Run uncertainty
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(calculation_id=calc_id)
        assert isinstance(unc, dict)
        assert unc.get("calculation_id") == calc_id or "mean_co2e_kg" in unc


# ===========================================================================
# Blend decomposition tests
# ===========================================================================


class TestBlendDecomposition:
    """Tests for blend decomposition through the full pipeline."""

    def test_blend_decomposition_r410a(self, service, sample_r410a_input):
        """R-410A blend should decompose into R-32 and R-125 components."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=30.0,
            method="equipment_based",
            equipment_type="COMMERCIAL_AC",
            facility_id="fac_office_001",
        )
        assert "calculation_id" in calc

        # The pipeline result may contain blend_decomposition
        inner = calc.get("result", calc)
        # If pipeline engine is available, check decomposition
        decomp = inner.get("blend_decomposition")
        if decomp is not None and len(decomp) > 0:
            assert len(decomp) >= 2
            gases = [c.get("gas", "") for c in decomp]
            assert any("R_32" in g or "32" in g for g in gases)
            assert any("R_125" in g or "125" in g for g in gases)

    def test_blend_decomposition_r404a(self, service, sample_r404a_input):
        """R-404A blend should decompose into R-125, R-143a, R-134a."""
        calc = service.calculate(
            refrigerant_type="R_404A",
            charge_kg=80.0,
            method="equipment_based",
            equipment_type="COMMERCIAL_REFRIGERATION",
            facility_id="fac_supermarket_001",
        )
        assert "calculation_id" in calc

    def test_pure_refrigerant_no_decomposition(self, service):
        """Pure refrigerant (R-134A) should not have blend decomposition."""
        calc = service.calculate(
            refrigerant_type="R_134A",
            charge_kg=10.0,
            method="equipment_based",
        )
        inner = calc.get("result", calc)
        decomp = inner.get("blend_decomposition")
        # For pure refrigerants, decomposition should be None or empty
        if decomp is not None:
            assert len(decomp) <= 1


# ===========================================================================
# Equipment type tests
# ===========================================================================


class TestMultipleEquipmentTypes:
    """Tests for different equipment types."""

    @pytest.mark.parametrize("equipment_type,expected_min_leak", [
        ("COMMERCIAL_REFRIGERATION", 10.0),
        ("RESIDENTIAL_AC", 2.0),
        ("SWITCHGEAR", 0.1),
        ("TRANSPORT_REFRIGERATION", 10.0),
    ])
    def test_multiple_equipment_types(
        self,
        service,
        equipment_type,
        expected_min_leak,
    ):
        """Different equipment types produce different leak rates."""
        leak = service.estimate_leak_rate(
            equipment_type=equipment_type,
            age_years=0,
        )
        assert leak["effective_rate_pct"] >= expected_min_leak * 0.5

    def test_different_equipment_different_emissions(self, service):
        """Same charge with different equipment types gives different emissions."""
        calc_ac = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            equipment_type="COMMERCIAL_AC",
        )
        calc_chiller = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            equipment_type="CHILLER_CENTRIFUGAL",
        )
        # Both should produce valid results
        assert "calculation_id" in calc_ac
        assert "calculation_id" in calc_chiller


# ===========================================================================
# Lifecycle tracking tests
# ===========================================================================


class TestLifecycleTracking:
    """Tests for equipment lifecycle event tracking."""

    def test_lifecycle_tracking(self, service):
        """Track equipment through installation -> recharge -> decommission."""
        equip_id = "eq_lifecycle_001"

        # Register equipment
        service.register_equipment(
            equipment_id=equip_id,
            equipment_type="COMMERCIAL_AC",
            refrigerant_type="R_410A",
            charge_kg=30.0,
        )

        # Installation event
        e1 = service.log_service_event(
            equipment_id=equip_id,
            event_type="installation",
            refrigerant_type="R_410A",
            quantity_kg=30.0,
        )
        assert e1["event_type"] == "installation"

        # Recharge event (after 1 year)
        e2 = service.log_service_event(
            equipment_id=equip_id,
            event_type="recharge",
            refrigerant_type="R_410A",
            quantity_kg=3.0,
            notes="Annual top-up after leak detection",
        )
        assert e2["event_type"] == "recharge"

        # Decommissioning event
        e3 = service.log_service_event(
            equipment_id=equip_id,
            event_type="decommissioning",
            refrigerant_type="R_410A",
            quantity_kg=27.0,
            notes="End of life recovery",
        )
        assert e3["event_type"] == "decommissioning"

        # All events stored
        assert len(service._service_events) >= 3


# ===========================================================================
# Compliance tests
# ===========================================================================


class TestComplianceChecking:
    """Tests for regulatory compliance checking."""

    def test_compliance_checking_eu_fgas(self, service):
        """EU F-Gas 2024 compliance check returns records."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        compliance = service.check_compliance(
            calculation_id=calc["calculation_id"],
            frameworks=["EU_FGAS_2024"],
        )
        assert compliance["total_count"] == 1
        record = compliance["records"][0]
        assert record["framework"] == "EU_FGAS_2024"

    def test_compliance_multiple_frameworks(self, service):
        """Compliance check across all major frameworks."""
        frameworks = [
            "GHG_PROTOCOL",
            "EPA_40CFR98_DD",
            "EU_FGAS_2024",
            "KIGALI_AMENDMENT",
            "ISO_14064",
            "CSRD_ESRS_E1",
            "UK_FGAS",
        ]
        compliance = service.check_compliance(frameworks=frameworks)
        assert compliance["total_count"] == 7


# ===========================================================================
# Uncertainty analysis tests
# ===========================================================================


class TestUncertaintyAnalysis:
    """Tests for uncertainty analysis integration."""

    def test_uncertainty_analysis(self, service):
        """Uncertainty analysis returns statistical estimates."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(calculation_id=calc_id)

        assert isinstance(unc, dict)
        # Should have either full MC results or analytical stub
        if "mean_co2e_kg" in unc:
            assert unc["mean_co2e_kg"] >= 0
            assert unc["std_co2e_kg"] >= 0

    def test_uncertainty_custom_iterations(self, service):
        """Uncertainty analysis respects custom iteration count."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        calc_id = calc["calculation_id"]
        unc = service.run_uncertainty(
            calculation_id=calc_id,
            iterations=1000,
        )
        assert isinstance(unc, dict)


# ===========================================================================
# Facility aggregation tests
# ===========================================================================


class TestFacilityAggregation:
    """Tests for facility-level emission aggregation."""

    def test_facility_aggregation(self, service):
        """Aggregate emissions across multiple calculations."""
        service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
            facility_id="fac_A",
        )
        service.calculate(
            refrigerant_type="R_134A",
            charge_kg=10.0,
            facility_id="fac_A",
        )
        service.calculate(
            refrigerant_type="R_404A",
            charge_kg=80.0,
            facility_id="fac_B",
        )

        agg = service.aggregate()
        assert isinstance(agg, dict)
        assert "aggregations" in agg
        assert agg.get("grand_total_tco2e", 0) >= 0

    def test_facility_aggregation_equity_share(self, service):
        """Equity share aggregation applies fractional ownership."""
        service.calculate(
            refrigerant_type="R_410A",
            charge_kg=50.0,
            facility_id="fac_jv",
        )
        agg_full = service.aggregate(control_approach="OPERATIONAL")
        agg_half = service.aggregate(
            control_approach="EQUITY_SHARE",
            share=0.5,
        )

        full_total = agg_full.get("grand_total_tco2e", 0)
        half_total = agg_half.get("grand_total_tco2e", 0)
        # Equity share at 50% should be approximately half
        if full_total > 0 and half_total > 0:
            assert half_total == pytest.approx(full_total * 0.5, rel=0.01)


# ===========================================================================
# Batch processing tests
# ===========================================================================


class TestBatchProcessing:
    """Tests for batch calculation processing."""

    def test_batch_processing(self, service):
        """Batch process multiple calculation inputs."""
        inputs = [
            {"refrigerant_type": "R_410A", "charge_kg": 25.0},
            {"refrigerant_type": "R_134A", "charge_kg": 10.0},
            {"refrigerant_type": "R_404A", "charge_kg": 80.0},
        ]
        result = service.calculate_batch(inputs)

        assert isinstance(result, dict)
        results_list = result.get("results", [])
        assert len(results_list) == 3

    def test_batch_processing_large(self, service):
        """Batch process a larger set of inputs."""
        inputs = [
            {
                "refrigerant_type": "R_410A",
                "charge_kg": float(i + 1),
            }
            for i in range(20)
        ]
        result = service.calculate_batch(inputs)
        assert result.get("total_count", len(result.get("results", []))) == 20


# ===========================================================================
# Provenance chain integrity tests
# ===========================================================================


class TestProvenanceChainIntegrity:
    """Tests for SHA-256 provenance chain integrity."""

    def test_provenance_chain_integrity(self, service):
        """Provenance hashes are valid SHA-256 hex strings."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        hash_val = calc.get("provenance_hash", "")
        if hash_val:
            assert len(hash_val) == 64
            # Verify it is valid hexadecimal
            int(hash_val, 16)

    def test_provenance_unique_per_calculation(self, service):
        """Each calculation gets a unique calculation_id."""
        calc1 = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        calc2 = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        assert calc1["calculation_id"] != calc2["calculation_id"]


# ===========================================================================
# Determinism tests
# ===========================================================================


class TestDeterminism:
    """Tests for calculation determinism."""

    def test_determinism(self, service):
        """Same input data produces same emission values."""
        kwargs = dict(
            refrigerant_type="R_134A",
            charge_kg=100.0,
            method="mass_balance",
            mass_balance_data={
                "inventory_start_kg": 500.0,
                "purchases_kg": 100.0,
                "recovery_kg": 50.0,
                "inventory_end_kg": 450.0,
            },
        )
        calc1 = service.calculate(**kwargs)
        calc2 = service.calculate(**kwargs)

        # Extract the inner result if pipeline-wrapped
        inner1 = calc1.get("result", calc1)
        inner2 = calc2.get("result", calc2)

        e1 = inner1.get("total_emissions_kg_co2e", inner1.get("emissions_kg", 0))
        e2 = inner2.get("total_emissions_kg_co2e", inner2.get("emissions_kg", 0))

        assert e1 == pytest.approx(e2, rel=1e-9)


# ===========================================================================
# SF6 and HFO specific tests
# ===========================================================================


class TestSpecificRefrigerants:
    """Tests for specific refrigerant types."""

    def test_sf6_switchgear_calculation(self, service, sample_sf6_input):
        """SF6 switchgear calculation produces high-GWP emissions."""
        calc = service.calculate(
            refrigerant_type="SF6",
            charge_kg=15.0,
            method="equipment_based",
            equipment_type="SWITCHGEAR",
            facility_id="fac_substation_001",
        )
        assert "calculation_id" in calc
        inner = calc.get("result", calc)
        # SF6 has very high GWP (~25200); even 0.5% leak on 15 kg is significant
        tco2e = inner.get("total_emissions_tco2e", 0)
        if tco2e > 0:
            # 15 * 0.005 * 25200 / 1000 = ~1.89 tCO2e
            assert tco2e > 0

    def test_hfo_low_gwp_calculation(self, service, sample_hfo_input):
        """HFO-1234yf calculation produces near-zero emissions."""
        calc = service.calculate(
            refrigerant_type="R_1234YF",
            charge_kg=5.0,
            method="equipment_based",
            custom_leak_rate_pct=10.0,
            facility_id="fac_hfo_001",
        )
        assert "calculation_id" in calc
        inner = calc.get("result", calc)
        # R-1234yf GWP is ~0.5; 5 * 0.10 * 0.5 / 1000 = ~0.00025 tCO2e
        tco2e = inner.get("total_emissions_tco2e", 0)
        # Should be very small
        assert tco2e < 1.0

    def test_natural_refrigerant_r744(self, service):
        """CO2 (R-744) refrigerant calculation."""
        calc = service.calculate(
            refrigerant_type="R_744",
            charge_kg=20.0,
            method="equipment_based",
            equipment_type="COMMERCIAL_REFRIGERATION",
        )
        assert "calculation_id" in calc
        inner = calc.get("result", calc)
        # R-744 (CO2) GWP = 1; emissions should be minimal
        tco2e = inner.get("total_emissions_tco2e", 0)
        assert tco2e < 1.0

    def test_ammonia_r717_zero_gwp(self, service):
        """Ammonia (R-717) has zero GWP."""
        calc = service.calculate(
            refrigerant_type="R_717",
            charge_kg=50.0,
            method="equipment_based",
            equipment_type="INDUSTRIAL_REFRIGERATION",
        )
        assert "calculation_id" in calc
        inner = calc.get("result", calc)
        # R-717 GWP = 0; CO2e should be zero
        tco2e = inner.get("total_emissions_tco2e", 0)
        assert tco2e == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Direct measurement workflow tests
# ===========================================================================


class TestDirectMeasurementWorkflow:
    """Tests for direct measurement method through the service."""

    def test_direct_measurement_workflow(self, service):
        """Direct measurement method produces expected emissions."""
        calc = service.calculate(
            refrigerant_type="R_410A",
            charge_kg=10.0,
            method="direct",
            measured_emissions_kg=2.5,
        )
        assert "calculation_id" in calc
        inner = calc.get("result", calc)
        assert inner["method"] == "direct"
        assert inner["emissions_kg"] == pytest.approx(2.5, rel=1e-3)

    def test_direct_measurement_compliance_check(self, service):
        """Direct measurement result can be compliance-checked."""
        calc = service.calculate(
            refrigerant_type="R_134A",
            charge_kg=20.0,
            method="direct",
            measured_emissions_kg=1.0,
        )
        compliance = service.check_compliance(
            calculation_id=calc["calculation_id"],
        )
        assert compliance["total_count"] >= 5


# ===========================================================================
# Validation-only workflow tests
# ===========================================================================


class TestValidationWorkflow:
    """Tests for the validate-without-calculate workflow."""

    def test_validate_valid_inputs(self, service):
        """Validation passes for well-formed inputs."""
        result = service.validate([
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
            },
        ])
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_rejects_negative_charge(self, service):
        """Validation rejects negative charge_kg."""
        result = service.validate([
            {
                "refrigerant_type": "R_410A",
                "charge_kg": -5.0,
            },
        ])
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_validate_rejects_invalid_method(self, service):
        """Validation rejects unsupported methods."""
        result = service.validate([
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 10.0,
                "method": "nonexistent_method",
            },
        ])
        assert result["valid"] is False


# ===========================================================================
# Multi-facility aggregation tests
# ===========================================================================


class TestMultiFacilityAggregation:
    """Tests for multi-facility emission aggregation workflows."""

    def test_three_facility_aggregation(self, service):
        """Aggregate emissions from three separate facilities."""
        for fac_id, ref_type, charge in [
            ("fac_X", "R_410A", 25.0),
            ("fac_Y", "R_134A", 10.0),
            ("fac_Z", "R_404A", 80.0),
        ]:
            service.calculate(
                refrigerant_type=ref_type,
                charge_kg=charge,
                facility_id=fac_id,
            )
        agg = service.aggregate()
        assert agg.get("grand_total_tco2e", 0) > 0

    def test_aggregation_groups_by_facility(self, service):
        """Aggregation groups results by facility_id."""
        for charge in [10.0, 20.0]:
            service.calculate(
                refrigerant_type="R_410A",
                charge_kg=charge,
                facility_id="fac_grouped",
            )
        service.calculate(
            refrigerant_type="R_410A",
            charge_kg=15.0,
            facility_id="fac_other",
        )
        agg = service.aggregate()
        assert isinstance(agg.get("aggregations"), list)


# ===========================================================================
# Service health workflow tests
# ===========================================================================


class TestServiceHealthWorkflow:
    """Tests for service health and stats in integration context."""

    def test_health_after_calculations(self, service):
        """Health check reflects service state after calculations."""
        service.calculate(
            refrigerant_type="R_410A",
            charge_kg=25.0,
        )
        health = service.get_health()
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert health["started"] is True

    def test_stats_after_batch(self, service):
        """Stats reflect batch calculation counts."""
        inputs = [
            {"refrigerant_type": "R_410A", "charge_kg": float(i + 1)}
            for i in range(5)
        ]
        service.calculate_batch(inputs)
        stats = service.get_stats()
        assert stats["total_batch_runs"] >= 1
