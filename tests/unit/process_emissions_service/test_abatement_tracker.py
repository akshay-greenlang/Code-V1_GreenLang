# -*- coding: utf-8 -*-
"""
Unit tests for AbatementTrackerEngine - AGENT-MRV-004 Process Emissions Agent

Tests abatement technology registration, efficiency retrieval with age
degradation, combined train efficiency, applicable technologies lookup,
performance tracking, cost analysis, regulatory minimum compliance, and
edge cases.

50 tests across 7 test classes.

Author: GreenLang QA Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

import pytest

from greenlang.process_emissions.abatement_tracker import (
    AbatementTrackerEngine,
    AbatementTechnology,
    AbatementTechnologyType,
    AbatementPerformance,
    AbatementCostAnalysis,
    MaintenanceStatus,
    VerificationStatus,
    MonitoringFrequency,
    _TECHNOLOGY_DEFAULTS,
    _PROCESS_TECHNOLOGY_MATRIX,
    _REGULATORY_MINIMUMS,
)


# =========================================================================
# Decimal helpers
# =========================================================================

_D = Decimal
_ZERO = _D("0")
_ONE = _D("1")


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def engine() -> AbatementTrackerEngine:
    """Create a fresh AbatementTrackerEngine."""
    return AbatementTrackerEngine(config={
        "default_spec_tolerance_pct": 10,
    })


@pytest.fixture
def nscr_id(engine: AbatementTrackerEngine) -> str:
    """Register a standard NSCR technology and return its ID."""
    return engine.register_technology({
        "name": "NSCR Catalyst A",
        "technology_type": "NSCR",
        "process_type": "NITRIC_ACID",
        "target_gas": "N2O",
        "destruction_efficiency": 0.85,
        "installation_date": "2026-01-01",
    })


@pytest.fixture
def thermal_id(engine: AbatementTrackerEngine) -> str:
    """Register a thermal destruction technology."""
    return engine.register_technology({
        "name": "Thermal Destructor B",
        "technology_type": "THERMAL_DESTRUCTION",
        "process_type": "ADIPIC_ACID",
        "target_gas": "N2O",
        "destruction_efficiency": 0.975,
        "installation_date": "2026-01-01",
    })


@pytest.fixture
def cwpb_id(engine: AbatementTrackerEngine) -> str:
    """Register a CWPB aluminum PFC abatement technology."""
    return engine.register_technology({
        "name": "CWPB Anode Control",
        "technology_type": "CWPB",
        "process_type": "ALUMINUM_PREBAKE",
        "target_gas": "CF4",
        "destruction_efficiency": 0.90,
        "installation_date": "2026-01-01",
    })


@pytest.fixture
def pou_id(engine: AbatementTrackerEngine) -> str:
    """Register a POU semiconductor abatement technology."""
    return engine.register_technology({
        "name": "POU Abatement Unit",
        "technology_type": "POU_ABATEMENT",
        "process_type": "SEMICONDUCTOR",
        "target_gas": "CF4",
        "destruction_efficiency": 0.95,
        "installation_date": "2026-01-01",
    })


# =========================================================================
# TestAbatementEfficiency (15 tests)
# =========================================================================

class TestAbatementEfficiency:
    """Tests for get_abatement_efficiency() for each technology type."""

    @pytest.mark.parametrize("tech_type,expected_default", [
        ("NSCR", _D("0.85")),
        ("SCR", _D("0.78")),
        ("EXTENDED_ABSORPTION", _D("0.20")),
        ("THERMAL_DESTRUCTION", _D("0.975")),
        ("CATALYTIC_REDUCTION", _D("0.94")),
        ("CWPB", _D("0.90")),
        ("POINT_FEED_PREBAKE", _D("0.80")),
        ("POU_ABATEMENT", _D("0.95")),
        ("REMOTE_PLASMA_CLEAN", _D("0.90")),
        ("SF6_RECOVERY", _D("0.95")),
        ("SF6_LEAK_REPAIR", _D("0.65")),
        ("POST_COMBUSTION_CAPTURE", _D("0.90")),
        ("OXY_FUEL_CAPTURE", _D("0.95")),
        ("WET_SCRUBBING", _D("0.85")),
        ("DRY_SCRUBBING", _D("0.75")),
    ])
    def test_default_efficiency_per_type(
        self, engine: AbatementTrackerEngine,
        tech_type: str, expected_default: Decimal
    ):
        """Default efficiency matches technology database value."""
        tech_id = engine.register_technology({
            "technology_type": tech_type,
            "process_type": "NITRIC_ACID",  # generic
            "target_gas": "N2O",
            "installation_date": "2026-02-18",
        })
        eff = engine.get_abatement_efficiency(tech_id)
        # For a newly installed tech (age ~0), eff should be close to default
        assert eff == pytest.approx(expected_default, abs=_D("0.005"))

    def test_custom_efficiency_override(self, engine: AbatementTrackerEngine):
        """Custom destruction_efficiency overrides the default."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.92,
            "installation_date": "2026-02-18",
        })
        eff = engine.get_abatement_efficiency(tech_id)
        assert eff >= _D("0.91") and eff <= _D("0.93")

    def test_unknown_tech_id_raises(self, engine: AbatementTrackerEngine):
        """Getting efficiency for unknown tech_id raises ValueError."""
        with pytest.raises(ValueError):
            engine.get_abatement_efficiency("nonexistent_id")

    def test_efficiency_clamped_0_to_1(self, engine: AbatementTrackerEngine):
        """Efficiency is always in [0, 1] range."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 1.5,  # should be clamped
            "installation_date": "2026-01-01",
        })
        eff = engine.get_abatement_efficiency(tech_id)
        assert eff >= _ZERO and eff <= _ONE

    def test_age_degradation_reduces_efficiency(
        self, engine: AbatementTrackerEngine
    ):
        """Older technology has lower efficiency due to degradation."""
        tech_id_new = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.85,
            "installation_date": "2026-01-01",
        })
        tech_id_old = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.85,
            "installation_date": "2016-01-01",  # 10 years old
        })
        eff_new = engine.get_abatement_efficiency(tech_id_new)
        eff_old = engine.get_abatement_efficiency(tech_id_old)
        assert eff_old < eff_new


# =========================================================================
# TestCombinedAbatement (10 tests)
# =========================================================================

class TestCombinedAbatement:
    """Tests for calculate_combined_efficiency() for series trains."""

    def test_single_technology_combined(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Single technology combined efficiency equals its own efficiency."""
        result = engine.calculate_combined_efficiency([nscr_id])
        single_eff = engine.get_abatement_efficiency(nscr_id)
        assert result["combined_efficiency"] == single_eff

    def test_two_technologies_combined(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Two technologies: combined = 1 - (1-e1)(1-e2)."""
        scr_id = engine.register_technology({
            "technology_type": "SCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.78,
            "installation_date": "2026-01-01",
        })
        result = engine.calculate_combined_efficiency([nscr_id, scr_id])
        e1 = engine.get_abatement_efficiency(nscr_id)
        e2 = engine.get_abatement_efficiency(scr_id)
        expected = (_ONE - (_ONE - e1) * (_ONE - e2)).quantize(
            _D("0.0001"), rounding=ROUND_HALF_UP
        )
        assert result["combined_efficiency"] == expected

    def test_three_technologies_combined(
        self, engine: AbatementTrackerEngine
    ):
        """Three technologies: combined = 1 - (1-e1)(1-e2)(1-e3)."""
        ids = []
        for i, eff in enumerate([0.80, 0.70, 0.50]):
            tid = engine.register_technology({
                "technology_type": "NSCR",
                "process_type": "NITRIC_ACID",
                "target_gas": "N2O",
                "destruction_efficiency": eff,
                "installation_date": "2026-01-01",
            })
            ids.append(tid)
        result = engine.calculate_combined_efficiency(ids)
        # Combined > max individual
        assert result["combined_efficiency"] > _D("0.80")

    def test_combined_efficiency_higher_than_individual(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Combined efficiency of two techs is higher than either alone."""
        ext_id = engine.register_technology({
            "technology_type": "EXTENDED_ABSORPTION",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "installation_date": "2026-01-01",
        })
        result = engine.calculate_combined_efficiency([nscr_id, ext_id])
        e1 = engine.get_abatement_efficiency(nscr_id)
        e2 = engine.get_abatement_efficiency(ext_id)
        assert result["combined_efficiency"] > max(e1, e2)

    def test_combined_returns_individual_efficiencies(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Result includes individual efficiencies for each technology."""
        result = engine.calculate_combined_efficiency([nscr_id])
        assert nscr_id in result["individual_efficiencies"]

    def test_combined_returns_provenance_hash(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Result includes a provenance hash."""
        result = engine.calculate_combined_efficiency([nscr_id])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_combined_empty_list_raises(self, engine: AbatementTrackerEngine):
        """Empty technology_ids list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_combined_efficiency([])

    def test_combined_unknown_id_raises(self, engine: AbatementTrackerEngine):
        """Unknown technology ID raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_combined_efficiency(["nonexistent_id"])

    def test_combined_processing_time(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Result includes positive processing_time_ms."""
        result = engine.calculate_combined_efficiency([nscr_id])
        assert result["processing_time_ms"] > 0

    def test_combined_technology_count(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Result includes correct technology_count."""
        ext_id = engine.register_technology({
            "technology_type": "EXTENDED_ABSORPTION",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "installation_date": "2026-01-01",
        })
        result = engine.calculate_combined_efficiency([nscr_id, ext_id])
        assert result["technology_count"] == 2


# =========================================================================
# TestApplicableTechnologies (8 tests)
# =========================================================================

class TestApplicableTechnologies:
    """Tests for get_applicable_technologies() for different process types."""

    def test_nitric_acid_technologies(self, engine: AbatementTrackerEngine):
        """Nitric acid has NSCR, SCR, extended absorption, catalytic reduction."""
        techs = engine.get_applicable_technologies("NITRIC_ACID")
        tech_types = {t["technology_type"] for t in techs}
        assert "NSCR" in tech_types
        assert "SCR" in tech_types
        assert "EXTENDED_ABSORPTION" in tech_types
        assert "CATALYTIC_REDUCTION" in tech_types

    def test_adipic_acid_technologies(self, engine: AbatementTrackerEngine):
        """Adipic acid has thermal destruction and catalytic reduction."""
        techs = engine.get_applicable_technologies("ADIPIC_ACID")
        tech_types = {t["technology_type"] for t in techs}
        assert "THERMAL_DESTRUCTION" in tech_types
        assert "CATALYTIC_REDUCTION" in tech_types

    def test_aluminum_prebake_technologies(self, engine: AbatementTrackerEngine):
        """Aluminum prebake has CWPB, point-feed, SF6 recovery."""
        techs = engine.get_applicable_technologies("ALUMINUM_PREBAKE")
        tech_types = {t["technology_type"] for t in techs}
        assert "CWPB" in tech_types
        assert "POINT_FEED_PREBAKE" in tech_types

    def test_semiconductor_technologies(self, engine: AbatementTrackerEngine):
        """Semiconductor has POU abatement, RPC, SF6 recovery."""
        techs = engine.get_applicable_technologies("SEMICONDUCTOR")
        tech_types = {t["technology_type"] for t in techs}
        assert "POU_ABATEMENT" in tech_types
        assert "REMOTE_PLASMA_CLEAN" in tech_types

    def test_cement_technologies(self, engine: AbatementTrackerEngine):
        """Cement has post-combustion, oxy-fuel, wet and dry scrubbing."""
        techs = engine.get_applicable_technologies("CEMENT")
        tech_types = {t["technology_type"] for t in techs}
        assert "POST_COMBUSTION_CAPTURE" in tech_types
        assert "OXY_FUEL_CAPTURE" in tech_types

    def test_gas_filter_narrows_results(self, engine: AbatementTrackerEngine):
        """Gas filter returns only technologies for that specific gas."""
        all_techs = engine.get_applicable_technologies("ALUMINUM_PREBAKE")
        cf4_techs = engine.get_applicable_technologies(
            "ALUMINUM_PREBAKE", target_gas="CF4"
        )
        assert len(cf4_techs) <= len(all_techs)
        for t in cf4_techs:
            assert "CF4" in t["applicable_gases"]

    def test_unknown_process_returns_empty(self, engine: AbatementTrackerEngine):
        """Unknown process type returns empty list."""
        techs = engine.get_applicable_technologies("NONEXISTENT_PROCESS")
        assert techs == []

    def test_applicable_has_efficiency_ranges(
        self, engine: AbatementTrackerEngine
    ):
        """Each applicable technology includes efficiency range data."""
        techs = engine.get_applicable_technologies("NITRIC_ACID")
        for t in techs:
            assert "efficiency_low" in t
            assert "efficiency_high" in t
            assert "default_efficiency" in t


# =========================================================================
# TestPerformanceTracking (5 tests)
# =========================================================================

class TestPerformanceTracking:
    """Tests for track_performance() with age degradation."""

    def test_record_performance(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Performance record captures measured vs expected efficiency."""
        record = engine.track_performance(
            nscr_id, measured_efficiency=0.83,
            measurement_date="2026-02-15",
            notes="Routine monthly test",
        )
        assert isinstance(record, AbatementPerformance)
        assert record.technology_id == nscr_id
        assert record.measured_efficiency == _D("0.83")
        assert record.notes == "Routine monthly test"

    def test_within_spec_flag(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Measurement close to expected is within spec."""
        record = engine.track_performance(nscr_id, measured_efficiency=0.84)
        assert record.is_within_spec is True

    def test_out_of_spec_flag(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Measurement far below expected is out of spec."""
        record = engine.track_performance(nscr_id, measured_efficiency=0.50)
        assert record.is_within_spec is False

    def test_performance_history(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Multiple measurements are stored in history."""
        engine.track_performance(nscr_id, measured_efficiency=0.84)
        engine.track_performance(nscr_id, measured_efficiency=0.83)
        engine.track_performance(nscr_id, measured_efficiency=0.82)
        history = engine.get_performance_history(nscr_id)
        assert len(history) == 3

    def test_degraded_status_auto_detect(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Significant underperformance triggers DEGRADED status."""
        engine.track_performance(nscr_id, measured_efficiency=0.50)
        tech_data = engine.get_technology(nscr_id)
        assert tech_data["maintenance_status"] == MaintenanceStatus.DEGRADED.value


# =========================================================================
# TestCostAnalysis (5 tests)
# =========================================================================

class TestCostAnalysis:
    """Tests for get_cost_analysis() financial calculations."""

    def test_basic_cost_analysis(self, engine: AbatementTrackerEngine):
        """Cost analysis produces valid financial metrics."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.85,
            "installation_date": "2026-01-01",
            "capex_usd": 500000,
            "annual_opex_usd": 50000,
            "annual_abated_tco2e": 10000,
        })
        analysis = engine.get_cost_analysis([tech_id])
        assert isinstance(analysis, AbatementCostAnalysis)
        assert analysis.total_capex_usd == _D("500000")
        assert analysis.total_annual_opex_usd == _D("50000")

    def test_levelized_cost_calculated(self, engine: AbatementTrackerEngine):
        """Levelized cost is (annualized_capex + opex) / abated."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "capex_usd": 1000000,
            "annual_opex_usd": 100000,
            "annual_abated_tco2e": 20000,
            "installation_date": "2026-01-01",
        })
        analysis = engine.get_cost_analysis(
            [tech_id], analysis_period_years=10
        )
        # annualized_capex = 1000000/10 = 100000
        # levelized = (100000 + 100000) / 20000 = 10.0
        assert analysis.levelized_cost_per_tco2e == _D("10.00")

    def test_npv_calculation(self, engine: AbatementTrackerEngine):
        """NPV is computed using discount rate over analysis period."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "capex_usd": 100000,
            "annual_opex_usd": 10000,
            "annual_abated_tco2e": 5000,
            "installation_date": "2026-01-01",
        })
        analysis = engine.get_cost_analysis(
            [tech_id],
            carbon_price_usd_per_tco2e=50.0,
            discount_rate=0.08,
            analysis_period_years=10,
        )
        assert analysis.npv_usd != _ZERO
        assert analysis.discount_rate == _D("0.08")

    def test_payback_period(self, engine: AbatementTrackerEngine):
        """Payback = capex / annual_savings."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "capex_usd": 500000,
            "annual_opex_usd": 50000,
            "annual_abated_tco2e": 10000,
            "installation_date": "2026-01-01",
        })
        analysis = engine.get_cost_analysis(
            [tech_id], carbon_price_usd_per_tco2e=100.0
        )
        # annual_savings = 10000*100 - 50000 = 950000
        # payback = 500000 / 950000 = ~0.53 years
        assert analysis.payback_years > _ZERO
        assert analysis.payback_years < _D("2")

    def test_empty_tech_ids_raises(self, engine: AbatementTrackerEngine):
        """Empty technology_ids raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.get_cost_analysis([])


# =========================================================================
# TestRegulatoryMinimum (5 tests)
# =========================================================================

class TestRegulatoryMinimum:
    """Tests for check_regulatory_minimum() compliance checks."""

    def test_compliant_nscr_eu_ets(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """NSCR at 85% meets EU ETS minimum (70% for N2O nitric acid)."""
        result = engine.check_regulatory_minimum(nscr_id, "EU_ETS_MRR")
        assert result["status"] == "COMPLIANT"

    def test_non_compliant_low_efficiency(
        self, engine: AbatementTrackerEngine
    ):
        """Low-efficiency tech fails regulatory minimum check."""
        tech_id = engine.register_technology({
            "technology_type": "EXTENDED_ABSORPTION",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.20,
            "installation_date": "2026-01-01",
        })
        result = engine.check_regulatory_minimum(tech_id, "EU_ETS_MRR")
        assert result["status"] == "NON_COMPLIANT"
        assert len(result["recommendations"]) > 0

    def test_not_applicable_framework(
        self, engine: AbatementTrackerEngine
    ):
        """Process/gas combo not in framework returns NOT_APPLICABLE."""
        tech_id = engine.register_technology({
            "technology_type": "WET_SCRUBBING",
            "process_type": "CEMENT",
            "target_gas": "CO2",
            "installation_date": "2026-01-01",
        })
        result = engine.check_regulatory_minimum(tech_id, "GHG_PROTOCOL")
        assert result["status"] == "NOT_APPLICABLE"

    def test_unknown_framework_raises(
        self, engine: AbatementTrackerEngine, nscr_id: str
    ):
        """Unknown framework raises ValueError."""
        with pytest.raises(ValueError, match="Unknown framework"):
            engine.check_regulatory_minimum(nscr_id, "BOGUS_FRAMEWORK")

    def test_regulatory_check_has_gap(
        self, engine: AbatementTrackerEngine
    ):
        """Non-compliant result includes the efficiency gap."""
        tech_id = engine.register_technology({
            "technology_type": "EXTENDED_ABSORPTION",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.20,
            "installation_date": "2026-01-01",
        })
        result = engine.check_regulatory_minimum(tech_id, "EU_ETS_MRR")
        assert result["gap"] is not None
        gap_val = _D(result["gap"])
        assert gap_val > _ZERO


# =========================================================================
# TestEdgeCases (5 tests)
# =========================================================================

class TestEdgeCases:
    """Tests for 0%/100% efficiency, empty inputs, max capacity."""

    def test_zero_efficiency_registration(self, engine: AbatementTrackerEngine):
        """Technology with 0% efficiency can be registered."""
        tech_id = engine.register_technology({
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 0.0,
            "installation_date": "2026-01-01",
        })
        eff = engine.get_abatement_efficiency(tech_id)
        assert eff == _ZERO

    def test_100_percent_efficiency(self, engine: AbatementTrackerEngine):
        """Technology with 100% efficiency registers correctly."""
        tech_id = engine.register_technology({
            "technology_type": "THERMAL_DESTRUCTION",
            "process_type": "ADIPIC_ACID",
            "target_gas": "N2O",
            "destruction_efficiency": 1.0,
            "installation_date": "2026-02-18",
        })
        eff = engine.get_abatement_efficiency(tech_id)
        assert eff <= _ONE

    def test_unknown_technology_type_raises(
        self, engine: AbatementTrackerEngine
    ):
        """Unknown technology_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown technology_type"):
            engine.register_technology({
                "technology_type": "MAGIC_SCRUBBER",
                "process_type": "CEMENT",
                "target_gas": "CO2",
            })

    def test_duplicate_technology_id_raises(
        self, engine: AbatementTrackerEngine
    ):
        """Registering a duplicate technology_id raises ValueError."""
        engine.register_technology({
            "technology_id": "dup_001",
            "technology_type": "NSCR",
            "process_type": "NITRIC_ACID",
            "target_gas": "N2O",
            "installation_date": "2026-01-01",
        })
        with pytest.raises(ValueError, match="already exists"):
            engine.register_technology({
                "technology_id": "dup_001",
                "technology_type": "NSCR",
                "process_type": "NITRIC_ACID",
                "target_gas": "N2O",
            })

    def test_missing_required_fields_raises(
        self, engine: AbatementTrackerEngine
    ):
        """Missing required fields raise ValueError."""
        with pytest.raises(ValueError):
            engine.register_technology({
                "technology_type": "NSCR",
                # missing process_type and target_gas
            })
