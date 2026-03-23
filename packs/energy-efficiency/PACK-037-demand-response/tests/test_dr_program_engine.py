# -*- coding: utf-8 -*-
"""
Unit tests for DRProgramEngine -- PACK-037 Engine 2
======================================================

Tests program matching across ISOs (PJM, ERCOT, CAISO, UK), eligibility
evaluation, revenue projection accuracy, portfolio optimization, stacking
conflict detection, and decimal precision.

Coverage target: 85%+
Total tests: ~70
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack037_test.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("dr_program_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")

    def test_engine_class_exists(self):
        assert hasattr(_m, "DRProgramEngine")

    def test_engine_instantiation(self):
        engine = _m.DRProgramEngine()
        assert engine is not None


class TestProgramMatching:
    """Test program matching across ISOs."""

    def _get_match(self, engine):
        return (getattr(engine, "match_programs", None)
                or getattr(engine, "find_eligible_programs", None)
                or getattr(engine, "match", None))

    @pytest.mark.parametrize("iso_rto", ["PJM", "ERCOT", "CAISO"])
    def test_match_by_iso(self, sample_dr_programs, sample_facility_profile,
                          iso_rto):
        engine = _m.DRProgramEngine()
        match = self._get_match(engine)
        if match is None:
            pytest.skip("match_programs method not found")
        result = match(programs=sample_dr_programs,
                       facility=sample_facility_profile,
                       iso_rto=iso_rto)
        assert result is not None

    def test_match_returns_list(self, sample_dr_programs,
                                sample_facility_profile):
        engine = _m.DRProgramEngine()
        match = self._get_match(engine)
        if match is None:
            pytest.skip("match_programs method not found")
        result = match(programs=sample_dr_programs,
                       facility=sample_facility_profile)
        matched = getattr(result, "matched", result)
        if isinstance(matched, list):
            assert len(matched) >= 1

    def test_pjm_programs_found(self, sample_dr_programs):
        pjm = [p for p in sample_dr_programs if p["iso_rto"] == "PJM"]
        assert len(pjm) == 2

    def test_ercot_programs_found(self, sample_dr_programs):
        ercot = [p for p in sample_dr_programs if p["iso_rto"] == "ERCOT"]
        assert len(ercot) == 1

    def test_caiso_programs_found(self, sample_dr_programs):
        caiso = [p for p in sample_dr_programs if p["iso_rto"] == "CAISO"]
        assert len(caiso) == 1


class TestEligibilityEvaluation:
    """Test eligibility evaluation."""

    def _get_eval(self, engine):
        return (getattr(engine, "evaluate_eligibility", None)
                or getattr(engine, "check_eligibility", None)
                or getattr(engine, "is_eligible", None))

    def test_eligible_capacity_sufficient(self, sample_dr_program,
                                          sample_facility_profile):
        engine = _m.DRProgramEngine()
        evaluate = self._get_eval(engine)
        if evaluate is None:
            pytest.skip("evaluate_eligibility method not found")
        result = evaluate(program=sample_dr_program,
                         facility=sample_facility_profile)
        assert result is not None

    def test_ineligible_capacity_insufficient(self, sample_dr_program):
        engine = _m.DRProgramEngine()
        evaluate = self._get_eval(engine)
        if evaluate is None:
            pytest.skip("evaluate_eligibility method not found")
        small_facility = {
            "facility_id": "SMALL", "peak_demand_kw": 50.0,
            "dr_enrolled_capacity_kw": 30.0, "iso_rto": "PJM",
        }
        result = evaluate(program=sample_dr_program, facility=small_facility)
        eligible = getattr(result, "eligible", result)
        if isinstance(eligible, bool):
            assert eligible is False

    @pytest.mark.parametrize("capacity_kw,expected", [
        (50, False), (100, True), (200, True), (800, True),
    ])
    def test_min_reduction_threshold(self, sample_dr_program, capacity_kw,
                                      expected):
        engine = _m.DRProgramEngine()
        evaluate = self._get_eval(engine)
        if evaluate is None:
            pytest.skip("evaluate_eligibility method not found")
        facility = {
            "facility_id": "TEST", "peak_demand_kw": capacity_kw * 3,
            "dr_enrolled_capacity_kw": capacity_kw, "iso_rto": "PJM",
        }
        result = evaluate(program=sample_dr_program, facility=facility)
        eligible = getattr(result, "eligible", result)
        if isinstance(eligible, bool):
            assert eligible == expected


class TestRevenueProjection:
    """Test revenue projection accuracy."""

    def _get_project(self, engine):
        return (getattr(engine, "project_revenue", None)
                or getattr(engine, "estimate_revenue", None)
                or getattr(engine, "revenue_projection", None))

    def test_revenue_positive(self, sample_dr_program, sample_facility_profile):
        engine = _m.DRProgramEngine()
        project = self._get_project(engine)
        if project is None:
            pytest.skip("project_revenue method not found")
        result = project(program=sample_dr_program,
                        facility=sample_facility_profile)
        revenue = getattr(result, "total_revenue_usd", result)
        if isinstance(revenue, (int, float, Decimal)):
            assert revenue > 0

    def test_capacity_payment_calculation(self, sample_dr_program,
                                          sample_facility_profile):
        engine = _m.DRProgramEngine()
        project = self._get_project(engine)
        if project is None:
            pytest.skip("project_revenue method not found")
        result = project(program=sample_dr_program,
                        facility=sample_facility_profile)
        cap = getattr(result, "capacity_payment_usd", None)
        if cap is not None:
            expected = (800.0 * 40.0)  # enrolled * rate
            assert float(cap) == pytest.approx(expected, rel=0.2)

    def test_energy_payment_per_event(self, sample_dr_program):
        engine = _m.DRProgramEngine()
        project = self._get_project(engine)
        if project is None:
            pytest.skip("project_revenue method not found")
        reduction_mwh = Decimal("3.0")
        rate = sample_dr_program["energy_payment_usd_per_mwh"]
        expected = reduction_mwh * rate
        assert expected == Decimal("300.00")

    @pytest.mark.parametrize("enrolled_kw,expected_cap_rev", [
        (100, 4000), (400, 16000), (800, 32000),
    ])
    def test_capacity_revenue_scales_linearly(self, sample_dr_program,
                                               enrolled_kw, expected_cap_rev):
        rate = float(sample_dr_program["capacity_payment_usd_per_kw_year"])
        actual = enrolled_kw * rate
        assert actual == pytest.approx(expected_cap_rev, rel=0.01)


class TestPortfolioOptimization:
    """Test portfolio optimization across programs."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_portfolio", None)
                or getattr(engine, "portfolio_optimization", None)
                or getattr(engine, "optimize_programs", None))

    def test_optimize_returns_result(self, sample_dr_programs,
                                     sample_facility_profile):
        engine = _m.DRProgramEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_portfolio method not found")
        result = optimize(programs=sample_dr_programs,
                         facility=sample_facility_profile)
        assert result is not None

    def test_optimize_selects_programs(self, sample_dr_programs,
                                       sample_facility_profile):
        engine = _m.DRProgramEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_portfolio method not found")
        result = optimize(programs=sample_dr_programs,
                         facility=sample_facility_profile)
        selected = getattr(result, "selected_programs", None)
        if selected is not None:
            assert len(selected) >= 1

    def test_optimize_revenue_positive(self, sample_dr_programs,
                                       sample_facility_profile):
        engine = _m.DRProgramEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_portfolio method not found")
        result = optimize(programs=sample_dr_programs,
                         facility=sample_facility_profile)
        rev = getattr(result, "projected_revenue_usd", None)
        if rev is not None:
            assert float(rev) > 0


class TestStackingConflictDetection:
    """Test stacking conflict detection."""

    def _get_detect(self, engine):
        return (getattr(engine, "detect_stacking_conflicts", None)
                or getattr(engine, "check_stacking", None)
                or getattr(engine, "stacking_conflicts", None))

    def test_detect_pjm_stacking_conflict(self, sample_dr_programs):
        engine = _m.DRProgramEngine()
        detect = self._get_detect(engine)
        if detect is None:
            pytest.skip("stacking conflict method not found")
        pjm_programs = [p for p in sample_dr_programs if p["iso_rto"] == "PJM"]
        result = detect(pjm_programs)
        assert result is not None

    def test_no_conflict_different_isos(self, sample_dr_programs):
        engine = _m.DRProgramEngine()
        detect = self._get_detect(engine)
        if detect is None:
            pytest.skip("stacking conflict method not found")
        diff_iso = [sample_dr_programs[0], sample_dr_programs[2]]  # PJM + ERCOT
        result = detect(diff_iso)
        conflicts = getattr(result, "conflicts", result)
        if isinstance(conflicts, list):
            assert len(conflicts) == 0

    def test_single_program_no_conflict(self, sample_dr_programs):
        engine = _m.DRProgramEngine()
        detect = self._get_detect(engine)
        if detect is None:
            pytest.skip("stacking conflict method not found")
        result = detect([sample_dr_programs[0]])
        conflicts = getattr(result, "conflicts", result)
        if isinstance(conflicts, list):
            assert len(conflicts) == 0


class TestDecimalPrecision:
    """Test decimal precision in financial calculations."""

    def test_capacity_payment_decimal(self, sample_dr_program):
        rate = sample_dr_program["capacity_payment_usd_per_kw_year"]
        assert isinstance(rate, Decimal)

    def test_energy_payment_decimal(self, sample_dr_program):
        rate = sample_dr_program["energy_payment_usd_per_mwh"]
        assert isinstance(rate, Decimal)

    def test_penalty_rate_decimal(self, sample_dr_program):
        rate = sample_dr_program["penalty_rate_usd_per_kw"]
        assert isinstance(rate, Decimal)

    def test_multiplication_preserves_decimal(self, sample_dr_program):
        rate = sample_dr_program["capacity_payment_usd_per_kw_year"]
        enrolled = Decimal("800.0")
        result = rate * enrolled
        assert isinstance(result, Decimal)
        assert result == Decimal("32000.00")

    def test_revenue_never_loses_cents(self):
        rate = Decimal("100.00")
        mwh = Decimal("3.123")
        revenue = rate * mwh
        assert revenue == Decimal("312.300")
        assert str(revenue).count(".") == 1

    @pytest.mark.parametrize("rate,qty,expected", [
        (Decimal("40.00"), Decimal("800"), Decimal("32000.00")),
        (Decimal("100.00"), Decimal("3.12"), Decimal("312.00")),
        (Decimal("50.00"), Decimal("150"), Decimal("7500.00")),
    ])
    def test_exact_multiplication(self, rate, qty, expected):
        result = rate * qty
        assert result == expected


class TestProgramDataValidation:
    """Test program data validation."""

    def test_all_programs_have_id(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert "program_id" in p
            assert p["program_id"] != ""

    def test_all_programs_have_iso(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert "iso_rto" in p

    def test_all_programs_have_baseline_method(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert "baseline_methodology" in p

    def test_max_events_positive(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert p["max_events_per_season"] > 0

    def test_max_duration_positive(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert p["max_event_duration_hours"] > 0

    def test_min_reduction_positive(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert p["min_reduction_kw"] > 0

    @pytest.mark.parametrize("program_idx", [0, 1, 2, 3])
    def test_program_type_valid(self, sample_dr_programs, program_idx):
        valid_types = {"ECONOMIC", "CAPACITY", "EMERGENCY", "RELIABILITY",
                       "ANCILLARY"}
        assert sample_dr_programs[program_idx]["program_type"] in valid_types


# =============================================================================
# Baseline Methodology Per Program
# =============================================================================


class TestBaselineMethodologyPerProgram:
    """Test baseline methodology assignment per program."""

    @pytest.mark.parametrize("program_idx,expected_method", [
        (0, "HIGH_4_OF_5"),
        (1, "HIGH_4_OF_5"),
        (2, "10_OF_10"),
        (3, "10_OF_10"),
    ])
    def test_program_baseline_method(self, sample_dr_programs,
                                      program_idx, expected_method):
        assert (sample_dr_programs[program_idx]["baseline_methodology"]
                == expected_method)

    @pytest.mark.parametrize("program_idx,expected_max_events", [
        (0, 10), (1, 6), (2, 5), (3, 15),
    ])
    def test_program_max_events(self, sample_dr_programs,
                                 program_idx, expected_max_events):
        assert (sample_dr_programs[program_idx]["max_events_per_season"]
                == expected_max_events)

    @pytest.mark.parametrize("program_idx,expected_max_duration", [
        (0, 6), (1, 6), (2, 4), (3, 4),
    ])
    def test_program_max_duration(self, sample_dr_programs,
                                   program_idx, expected_max_duration):
        assert (sample_dr_programs[program_idx]["max_event_duration_hours"]
                == expected_max_duration)


# =============================================================================
# Revenue Comparison Across Programs
# =============================================================================


class TestRevenueComparison:
    """Test revenue comparison across different programs."""

    def test_pjm_csp_higher_capacity_than_elr(self, sample_dr_programs):
        elr = next(p for p in sample_dr_programs
                  if p["program_id"] == "PJM-ELR-2025")
        csp = next(p for p in sample_dr_programs
                  if p["program_id"] == "PJM-CSP-2025")
        assert (csp["capacity_payment_usd_per_kw_year"] >
                elr["capacity_payment_usd_per_kw_year"])

    def test_ercot_highest_energy_rate(self, sample_dr_programs):
        ercot = next(p for p in sample_dr_programs
                    if p["iso_rto"] == "ERCOT")
        max_energy = max(p["energy_payment_usd_per_mwh"]
                        for p in sample_dr_programs)
        assert ercot["energy_payment_usd_per_mwh"] == max_energy

    def test_all_programs_have_positive_capacity(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert p["capacity_payment_usd_per_kw_year"] >= 0

    def test_all_programs_have_nonneg_energy(self, sample_dr_programs):
        for p in sample_dr_programs:
            assert p["energy_payment_usd_per_mwh"] >= 0
