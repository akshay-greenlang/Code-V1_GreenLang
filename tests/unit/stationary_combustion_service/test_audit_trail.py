# -*- coding: utf-8 -*-
"""
Unit tests for AuditTrailEngine (Engine 6)

AGENT-MRV-001 Stationary Combustion Agent

Tests audit step recording, audit trail retrieval, compliance mapping for
all 6 regulatory frameworks, audit report generation, compliance validation,
JSON/CSV export, statistics, clearing, and entry counting.
40+ tests covering all public methods and edge cases.

Target: 85%+ code coverage of audit_trail.py
"""

import csv
import io
import json
import threading

import pytest

from greenlang.stationary_combustion.audit_trail import (
    AuditTrailEngine,
    AuditEntry,
    COMPLIANCE_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a fresh AuditTrailEngine with no config."""
    return AuditTrailEngine(config=None)


@pytest.fixture
def populated_engine(engine):
    """Engine with a 6-step calculation already recorded."""
    steps = [
        ("input_validation", {"fuel_type": "NATURAL_GAS", "quantity": 1000}, {"validated": True}),
        ("unit_conversion", {"quantity": 1000, "unit": "m3"}, {"energy_gj": 38.3}),
        ("ef_selection", {"fuel": "NATURAL_GAS", "tier": "TIER_1"}, {"ef_co2": 56.1}),
        ("emission_calculation", {"energy_gj": 38.3, "ef_co2": 56.1}, {"co2_kg": 2148.63}),
        ("gwp_application", {"co2_kg": 2148.63, "ch4_kg": 0.1, "n2o_kg": 0.01}, {"co2e_kg": 2155.0}),
        ("aggregation", {"co2e_kg": 2155.0}, {"co2e_tonnes": 2.155}),
    ]
    for i, (name, inp, out) in enumerate(steps, start=1):
        ef_used = {"value": 56.1, "source": "EPA", "tier": "TIER_1"} if name == "ef_selection" else None
        method_ref = "GHG Protocol Ch5" if name == "emission_calculation" else None
        engine.record_step(
            calculation_id="CALC-001",
            step_number=i,
            step_name=name,
            input_data=inp,
            output_data=out,
            emission_factor_used=ef_used,
            methodology_reference=method_ref,
        )
    return engine


@pytest.fixture
def compliant_calc_result():
    """A calculation result dict that satisfies most compliance checks."""
    return {
        "total_co2e_tonnes": 2.155,
        "total_co2_tonnes": 2.148,
        "total_ch4_tonnes": 0.004,
        "total_n2o_tonnes": 0.003,
        "total_biogenic_co2_tonnes": 0.0,
        "provenance_hash": "a" * 64,
        "emission_factors": [{"source": "EPA", "value": 56.1}],
        "emissions_by_fuel": {"NATURAL_GAS": 2.155},
        "uncertainty": {"cv": 0.05},
    }


@pytest.fixture
def minimal_calc_result():
    """A minimal calculation result that fails most compliance checks."""
    return {"total_co2e_tonnes": 1.0}


# ---------------------------------------------------------------------------
# TestAuditTrailInit
# ---------------------------------------------------------------------------


class TestAuditTrailInit:
    """Test AuditTrailEngine initialisation."""

    def test_default_initialisation(self, engine):
        """Engine initialises with empty store and zero count."""
        assert engine._audit_store == {}
        assert engine._entry_count == 0

    def test_config_stored(self):
        """Config object is stored on the engine."""
        cfg = {"some_key": "some_value"}
        eng = AuditTrailEngine(config=cfg)
        assert eng._config == cfg

    def test_lock_is_reentrant(self, engine):
        """Engine lock is an RLock (reentrant)."""
        assert isinstance(engine._lock, type(threading.RLock()))


# ---------------------------------------------------------------------------
# TestRecordStep
# ---------------------------------------------------------------------------


class TestRecordStep:
    """Test record_step method."""

    def test_returns_audit_entry(self, engine):
        """record_step returns an AuditEntry dataclass."""
        entry = engine.record_step(
            "CALC-001", 1, "input_validation",
            {"fuel": "DIESEL"}, {"valid": True},
        )
        assert isinstance(entry, AuditEntry)

    def test_entry_fields_populated(self, engine):
        """All fields of the returned AuditEntry are populated."""
        entry = engine.record_step(
            calculation_id="CALC-002",
            step_number=3,
            step_name="ef_selection",
            input_data={"fuel": "COAL"},
            output_data={"ef": 94.6},
            emission_factor_used={"value": 94.6, "source": "EPA"},
            methodology_reference="EPA Table C-1",
        )
        assert entry.calculation_id == "CALC-002"
        assert entry.step_number == 3
        assert entry.step_name == "ef_selection"
        assert entry.input_data == {"fuel": "COAL"}
        assert entry.output_data == {"ef": 94.6}
        assert entry.emission_factor_used == {"value": 94.6, "source": "EPA"}
        assert entry.methodology_reference == "EPA Table C-1"

    def test_provenance_hash_is_sha256(self, engine):
        """Entry provenance hash is a 64-char hex string (SHA-256)."""
        entry = engine.record_step(
            "CALC-001", 1, "test", {"a": 1}, {"b": 2},
        )
        assert len(entry.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in entry.provenance_hash)

    def test_provenance_hash_deterministic(self, engine):
        """Same input produces the same provenance hash."""
        e1 = engine.record_step("C1", 1, "step", {"x": 1}, {"y": 2})
        # Create a fresh engine to avoid step-count effects
        eng2 = AuditTrailEngine()
        e2 = eng2.record_step("C1", 1, "step", {"x": 1}, {"y": 2})
        assert e1.provenance_hash == e2.provenance_hash

    def test_different_input_different_hash(self, engine):
        """Different input data produces different hashes."""
        e1 = engine.record_step("C1", 1, "step", {"x": 1}, {"y": 2})
        e2 = engine.record_step("C1", 2, "step", {"x": 99}, {"y": 2})
        assert e1.provenance_hash != e2.provenance_hash

    def test_timestamp_present(self, engine):
        """Entry has an ISO-format timestamp."""
        entry = engine.record_step("C1", 1, "test", {}, {})
        assert "T" in entry.timestamp

    def test_entry_count_increments(self, engine):
        """Each record_step call increments the entry count."""
        assert engine.get_entry_count() == 0
        engine.record_step("C1", 1, "s1", {}, {})
        assert engine.get_entry_count() == 1
        engine.record_step("C1", 2, "s2", {}, {})
        assert engine.get_entry_count() == 2

    def test_optional_fields_default_none(self, engine):
        """emission_factor_used and methodology_reference default to None."""
        entry = engine.record_step("C1", 1, "test", {"a": 1}, {"b": 2})
        assert entry.emission_factor_used is None
        assert entry.methodology_reference is None


# ---------------------------------------------------------------------------
# TestGetAuditTrail
# ---------------------------------------------------------------------------


class TestGetAuditTrail:
    """Test get_audit_trail method."""

    def test_returns_list(self, engine):
        """Returns a list even for unknown calc_id."""
        trail = engine.get_audit_trail("NONEXISTENT")
        assert isinstance(trail, list)
        assert len(trail) == 0

    def test_returns_all_steps(self, populated_engine):
        """Returns all recorded steps for a calculation."""
        trail = populated_engine.get_audit_trail("CALC-001")
        assert len(trail) == 6

    def test_steps_in_order(self, populated_engine):
        """Steps are returned in recording order."""
        trail = populated_engine.get_audit_trail("CALC-001")
        step_numbers = [e.step_number for e in trail]
        assert step_numbers == [1, 2, 3, 4, 5, 6]

    def test_returns_copy(self, populated_engine):
        """Returned list is a copy (not a reference to internal state)."""
        trail1 = populated_engine.get_audit_trail("CALC-001")
        trail2 = populated_engine.get_audit_trail("CALC-001")
        assert trail1 is not trail2

    def test_multiple_calculations_isolated(self, engine):
        """Different calc_ids have independent trails."""
        engine.record_step("CALC-A", 1, "s1", {}, {})
        engine.record_step("CALC-B", 1, "s1", {}, {})
        engine.record_step("CALC-A", 2, "s2", {}, {})

        trail_a = engine.get_audit_trail("CALC-A")
        trail_b = engine.get_audit_trail("CALC-B")
        assert len(trail_a) == 2
        assert len(trail_b) == 1


# ---------------------------------------------------------------------------
# TestEntryCount
# ---------------------------------------------------------------------------


class TestEntryCount:
    """Test get_entry_count method."""

    def test_starts_at_zero(self, engine):
        """Fresh engine has zero entries."""
        assert engine.get_entry_count() == 0

    def test_counts_across_calculations(self, engine):
        """Count includes entries from all calculations."""
        engine.record_step("C1", 1, "s1", {}, {})
        engine.record_step("C2", 1, "s1", {}, {})
        engine.record_step("C3", 1, "s1", {}, {})
        assert engine.get_entry_count() == 3

    def test_count_matches_populated(self, populated_engine):
        """Count matches the number of recorded steps in fixture."""
        assert populated_engine.get_entry_count() == 6


# ---------------------------------------------------------------------------
# TestComplianceMapping
# ---------------------------------------------------------------------------


class TestComplianceMapping:
    """Test get_compliance_mapping method."""

    def test_all_frameworks_returned(self, engine):
        """When framework=None, all 6 frameworks are returned."""
        mapping = engine.get_compliance_mapping()
        assert len(mapping) == 6
        expected_keys = {"GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1", "EPA_40CFR98", "EU_ETS", "UK_SECR"}
        assert set(mapping.keys()) == expected_keys

    def test_single_framework(self, engine):
        """Requesting a single framework returns only that one."""
        mapping = engine.get_compliance_mapping("GHG_PROTOCOL")
        assert len(mapping) == 1
        assert "GHG_PROTOCOL" in mapping

    def test_unknown_framework_returns_empty(self, engine):
        """Unknown framework returns empty dict."""
        mapping = engine.get_compliance_mapping("UNKNOWN_FRAMEWORK")
        assert mapping == {}

    def test_case_insensitive_lookup(self, engine):
        """Framework lookup is case-insensitive (uppercased internally)."""
        mapping = engine.get_compliance_mapping("ghg_protocol")
        assert "GHG_PROTOCOL" in mapping

    def test_ghg_protocol_has_5_requirements(self, engine):
        """GHG Protocol has exactly 5 requirements."""
        mapping = engine.get_compliance_mapping("GHG_PROTOCOL")
        reqs = mapping["GHG_PROTOCOL"]["requirements"]
        assert len(reqs) == 5

    def test_iso_14064_has_2_requirements(self, engine):
        """ISO 14064 has exactly 2 requirements."""
        mapping = engine.get_compliance_mapping("ISO_14064")
        reqs = mapping["ISO_14064"]["requirements"]
        assert len(reqs) == 2

    def test_csrd_esrs_e1_has_3_requirements(self, engine):
        """CSRD ESRS E1 has exactly 3 requirements."""
        mapping = engine.get_compliance_mapping("CSRD_ESRS_E1")
        reqs = mapping["CSRD_ESRS_E1"]["requirements"]
        assert len(reqs) == 3

    def test_epa_40cfr98_has_3_requirements(self, engine):
        """EPA 40 CFR 98 has exactly 3 requirements."""
        mapping = engine.get_compliance_mapping("EPA_40CFR98")
        reqs = mapping["EPA_40CFR98"]["requirements"]
        assert len(reqs) == 3

    def test_eu_ets_has_2_requirements(self, engine):
        """EU ETS has exactly 2 requirements."""
        mapping = engine.get_compliance_mapping("EU_ETS")
        reqs = mapping["EU_ETS"]["requirements"]
        assert len(reqs) == 2

    def test_uk_secr_has_1_requirement(self, engine):
        """UK SECR has exactly 1 requirement."""
        mapping = engine.get_compliance_mapping("UK_SECR")
        reqs = mapping["UK_SECR"]["requirements"]
        assert len(reqs) == 1

    def test_total_requirements_is_16(self):
        """All 6 frameworks have 5+2+3+3+2+1 = 16 requirements total."""
        total = sum(len(fw["requirements"]) for fw in COMPLIANCE_MAP.values())
        assert total == 16


# ---------------------------------------------------------------------------
# TestGenerateAuditReport
# ---------------------------------------------------------------------------


class TestGenerateAuditReport:
    """Test generate_audit_report method."""

    def test_report_contains_required_keys(self, populated_engine):
        """Report dict has all expected keys."""
        report = populated_engine.generate_audit_report("CALC-001")
        expected_keys = {
            "calculation_id", "framework", "standard", "chapters",
            "total_steps", "steps", "emission_factors",
            "methodology_references", "compliance_requirements",
            "generated_at", "report_hash",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_report_calc_id(self, populated_engine):
        """Report contains the correct calculation_id."""
        report = populated_engine.generate_audit_report("CALC-001")
        assert report["calculation_id"] == "CALC-001"

    def test_report_default_framework_ghg(self, populated_engine):
        """Default framework is GHG_PROTOCOL."""
        report = populated_engine.generate_audit_report("CALC-001")
        assert report["framework"] == "GHG_PROTOCOL"

    def test_report_step_count(self, populated_engine):
        """total_steps matches the number of recorded steps."""
        report = populated_engine.generate_audit_report("CALC-001")
        assert report["total_steps"] == 6

    def test_report_steps_are_dicts(self, populated_engine):
        """Steps in the report are serialized as dictionaries."""
        report = populated_engine.generate_audit_report("CALC-001")
        assert len(report["steps"]) == 6
        for step in report["steps"]:
            assert isinstance(step, dict)
            assert "step_name" in step

    def test_report_emission_factors_extracted(self, populated_engine):
        """Emission factors are extracted from steps that have them."""
        report = populated_engine.generate_audit_report("CALC-001")
        # Only the ef_selection step has an emission_factor_used
        assert len(report["emission_factors"]) == 1
        assert report["emission_factors"][0]["source"] == "EPA"

    def test_report_methodology_references_extracted(self, populated_engine):
        """Methodology references are extracted from steps that have them."""
        report = populated_engine.generate_audit_report("CALC-001")
        assert len(report["methodology_references"]) == 1
        assert "GHG Protocol Ch5" in report["methodology_references"]

    def test_report_compliance_requirements_populated(self, populated_engine):
        """Report includes compliance requirements for the framework."""
        report = populated_engine.generate_audit_report("CALC-001", framework="GHG_PROTOCOL")
        assert len(report["compliance_requirements"]) == 5

    def test_report_hash_is_sha256(self, populated_engine):
        """Report hash is a 64-char hex string."""
        report = populated_engine.generate_audit_report("CALC-001")
        assert len(report["report_hash"]) == 64

    def test_report_for_different_framework(self, populated_engine):
        """Report can be generated for a different framework."""
        report = populated_engine.generate_audit_report("CALC-001", framework="ISO_14064")
        assert report["framework"] == "ISO_14064"
        assert "ISO 14064-1:2018" in report["standard"]

    def test_report_for_nonexistent_calc(self, engine):
        """Report for nonexistent calc has zero steps."""
        report = engine.generate_audit_report("NONEXISTENT")
        assert report["total_steps"] == 0
        assert report["steps"] == []


# ---------------------------------------------------------------------------
# TestValidateCompliance
# ---------------------------------------------------------------------------


class TestValidateCompliance:
    """Test validate_compliance method."""

    def test_ghg_protocol_full_compliance(self, engine, compliant_calc_result):
        """Fully compliant result passes all GHG Protocol requirements."""
        result = engine.validate_compliance(compliant_calc_result, "GHG_PROTOCOL")
        assert result["overall_compliant"] is True
        assert result["requirements_met"] == 5
        assert result["requirements_checked"] == 5

    def test_iso_14064_full_compliance(self, engine, compliant_calc_result):
        """Fully compliant result passes all ISO 14064 requirements."""
        result = engine.validate_compliance(compliant_calc_result, "ISO_14064")
        assert result["overall_compliant"] is True
        assert result["requirements_met"] == 2

    def test_csrd_esrs_e1_full_compliance(self, engine, compliant_calc_result):
        """Fully compliant result passes all CSRD ESRS E1 requirements."""
        result = engine.validate_compliance(compliant_calc_result, "CSRD_ESRS_E1")
        assert result["overall_compliant"] is True
        assert result["requirements_met"] == 3

    def test_epa_40cfr98_full_compliance(self, engine, compliant_calc_result):
        """Fully compliant result passes all EPA 40 CFR 98 requirements."""
        result = engine.validate_compliance(compliant_calc_result, "EPA_40CFR98")
        assert result["overall_compliant"] is True
        assert result["requirements_met"] == 3

    def test_eu_ets_full_compliance(self, engine, compliant_calc_result):
        """Fully compliant result passes all EU ETS requirements."""
        result = engine.validate_compliance(compliant_calc_result, "EU_ETS")
        assert result["overall_compliant"] is True
        assert result["requirements_met"] == 2

    def test_uk_secr_full_compliance(self, engine, compliant_calc_result):
        """Fully compliant result passes UK SECR requirement."""
        result = engine.validate_compliance(compliant_calc_result, "UK_SECR")
        assert result["overall_compliant"] is True
        assert result["requirements_met"] == 1

    def test_minimal_result_fails_ghg_protocol(self, engine, minimal_calc_result):
        """Minimal result fails some GHG Protocol checks."""
        result = engine.validate_compliance(minimal_calc_result, "GHG_PROTOCOL")
        assert result["requirements_met"] < result["requirements_checked"]

    def test_missing_gas_decomposition_fails_ghg_sc_02(self, engine):
        """Missing gas decomposition fails GHG-SC-02."""
        calc = {"total_co2e_tonnes": 1.0}
        result = engine.validate_compliance(calc, "GHG_PROTOCOL")
        details_by_id = {d["id"]: d for d in result["details"]}
        assert details_by_id["GHG-SC-02"]["compliant"] is False

    def test_missing_biogenic_fails_ghg_sc_03(self, engine):
        """Missing biogenic CO2 field fails GHG-SC-03."""
        calc = {"total_co2e_tonnes": 1.0}
        result = engine.validate_compliance(calc, "GHG_PROTOCOL")
        details_by_id = {d["id"]: d for d in result["details"]}
        assert details_by_id["GHG-SC-03"]["compliant"] is False

    def test_missing_provenance_fails_ghg_sc_05(self, engine):
        """Missing provenance_hash fails GHG-SC-05."""
        calc = {"total_co2e_tonnes": 1.0}
        result = engine.validate_compliance(calc, "GHG_PROTOCOL")
        details_by_id = {d["id"]: d for d in result["details"]}
        assert details_by_id["GHG-SC-05"]["compliant"] is False

    def test_missing_uncertainty_fails_iso_sc_02(self, engine):
        """Missing uncertainty field fails ISO-SC-02."""
        calc = {"total_co2e_tonnes": 1.0}
        result = engine.validate_compliance(calc, "ISO_14064")
        details_by_id = {d["id"]: d for d in result["details"]}
        assert details_by_id["ISO-SC-02"]["compliant"] is False

    def test_unknown_framework_returns_error(self, engine, compliant_calc_result):
        """Unknown framework returns non-compliant with error."""
        result = engine.validate_compliance(compliant_calc_result, "UNKNOWN")
        assert result["overall_compliant"] is False
        assert result["requirements_checked"] == 0
        assert "error" in result

    def test_details_contain_evidence(self, engine, compliant_calc_result):
        """Each detail has id, desc, how, compliant, evidence."""
        result = engine.validate_compliance(compliant_calc_result, "GHG_PROTOCOL")
        for detail in result["details"]:
            assert "id" in detail
            assert "desc" in detail
            assert "how" in detail
            assert "compliant" in detail
            assert "evidence" in detail

    def test_epa_sc_01_always_compliant(self, engine):
        """EPA-SC-01 (tier methodology) always returns compliant."""
        result = engine.validate_compliance({}, "EPA_40CFR98")
        details_by_id = {d["id"]: d for d in result["details"]}
        assert details_by_id["EPA-SC-01"]["compliant"] is True

    def test_ets_sc_01_always_compliant(self, engine):
        """ETS-SC-01 (NCV basis) always returns compliant."""
        result = engine.validate_compliance({}, "EU_ETS")
        details_by_id = {d["id"]: d for d in result["details"]}
        assert details_by_id["ETS-SC-01"]["compliant"] is True


# ---------------------------------------------------------------------------
# TestExportJSON
# ---------------------------------------------------------------------------


class TestExportJSON:
    """Test export_audit_json method."""

    def test_valid_json(self, populated_engine):
        """Exported string is valid JSON."""
        json_str = populated_engine.export_audit_json("CALC-001")
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_correct_entry_count(self, populated_engine):
        """JSON contains the correct number of entries."""
        parsed = json.loads(populated_engine.export_audit_json("CALC-001"))
        assert len(parsed) == 6

    def test_entries_have_required_fields(self, populated_engine):
        """Each JSON entry has the required fields."""
        parsed = json.loads(populated_engine.export_audit_json("CALC-001"))
        required = {"calculation_id", "step_number", "step_name", "input_data",
                     "output_data", "timestamp", "provenance_hash"}
        for entry in parsed:
            assert required.issubset(set(entry.keys()))

    def test_empty_calc_id_returns_empty_array(self, engine):
        """Nonexistent calc_id returns an empty JSON array."""
        json_str = engine.export_audit_json("NONEXISTENT")
        assert json.loads(json_str) == []


# ---------------------------------------------------------------------------
# TestExportCSV
# ---------------------------------------------------------------------------


class TestExportCSV:
    """Test export_audit_csv method."""

    def test_csv_has_header(self, populated_engine):
        """CSV output starts with a header row."""
        csv_str = populated_engine.export_audit_csv("CALC-001")
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        expected_header = [
            "calculation_id", "step_number", "step_name",
            "timestamp", "methodology_reference", "provenance_hash",
        ]
        assert header == expected_header

    def test_csv_row_count(self, populated_engine):
        """CSV has header + 6 data rows."""
        csv_str = populated_engine.export_audit_csv("CALC-001")
        lines = csv_str.strip().split("\n")
        assert len(lines) == 7  # 1 header + 6 data rows

    def test_csv_data_values(self, populated_engine):
        """CSV data rows contain correct values."""
        csv_str = populated_engine.export_audit_csv("CALC-001")
        reader = csv.reader(io.StringIO(csv_str))
        next(reader)  # Skip header
        first_row = next(reader)
        assert first_row[0] == "CALC-001"
        assert first_row[1] == "1"
        assert first_row[2] == "input_validation"

    def test_empty_calc_returns_header_only(self, engine):
        """Nonexistent calc_id returns CSV with header only."""
        csv_str = engine.export_audit_csv("NONEXISTENT")
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # Header only


# ---------------------------------------------------------------------------
# TestAuditStatistics
# ---------------------------------------------------------------------------


class TestAuditStatistics:
    """Test get_audit_statistics method."""

    def test_empty_statistics(self, engine):
        """Fresh engine has zero-valued statistics."""
        stats = engine.get_audit_statistics()
        assert stats["total_entries"] == 0
        assert stats["total_calculations"] == 0
        assert stats["entries_by_step_name"] == {}
        assert stats["entries_by_calculation"] == {}
        assert stats["average_steps_per_calculation"] == 0.0

    def test_populated_statistics(self, populated_engine):
        """Populated engine has correct statistics."""
        stats = populated_engine.get_audit_statistics()
        assert stats["total_entries"] == 6
        assert stats["total_calculations"] == 1
        assert stats["average_steps_per_calculation"] == 6.0
        assert stats["entries_by_calculation"]["CALC-001"] == 6

    def test_entries_by_step_name(self, populated_engine):
        """Entries by step name tallied correctly."""
        stats = populated_engine.get_audit_statistics()
        by_step = stats["entries_by_step_name"]
        assert by_step["input_validation"] == 1
        assert by_step["ef_selection"] == 1
        assert by_step["aggregation"] == 1

    def test_multiple_calculations(self, engine):
        """Statistics across multiple calculations."""
        engine.record_step("C1", 1, "s1", {}, {})
        engine.record_step("C1", 2, "s2", {}, {})
        engine.record_step("C2", 1, "s1", {}, {})

        stats = engine.get_audit_statistics()
        assert stats["total_entries"] == 3
        assert stats["total_calculations"] == 2
        assert stats["average_steps_per_calculation"] == 1.5


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------


class TestClear:
    """Test clear method."""

    def test_clear_empties_store(self, populated_engine):
        """clear() removes all audit data."""
        assert populated_engine.get_entry_count() > 0
        populated_engine.clear()
        assert populated_engine.get_entry_count() == 0

    def test_clear_empties_all_trails(self, populated_engine):
        """After clear, get_audit_trail returns empty for all calc_ids."""
        populated_engine.clear()
        assert populated_engine.get_audit_trail("CALC-001") == []

    def test_clear_resets_statistics(self, populated_engine):
        """After clear, statistics are back to zero."""
        populated_engine.clear()
        stats = populated_engine.get_audit_statistics()
        assert stats["total_entries"] == 0
        assert stats["total_calculations"] == 0

    def test_can_record_after_clear(self, populated_engine):
        """Recording works normally after clear()."""
        populated_engine.clear()
        populated_engine.record_step("C2", 1, "test", {}, {})
        assert populated_engine.get_entry_count() == 1


# ---------------------------------------------------------------------------
# TestAuditEntryToDict
# ---------------------------------------------------------------------------


class TestAuditEntryToDict:
    """Test AuditEntry.to_dict serialisation."""

    def test_to_dict_keys(self, engine):
        """to_dict returns all expected keys."""
        entry = engine.record_step("C1", 1, "test", {"a": 1}, {"b": 2})
        d = entry.to_dict()
        expected_keys = {
            "calculation_id", "step_number", "step_name",
            "input_data", "output_data", "emission_factor_used",
            "methodology_reference", "timestamp", "provenance_hash",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self, engine):
        """to_dict values match the dataclass fields."""
        entry = engine.record_step(
            "C1", 1, "test",
            {"fuel": "DIESEL"}, {"valid": True},
            emission_factor_used={"ef": 2.68},
            methodology_reference="ISO 14064",
        )
        d = entry.to_dict()
        assert d["calculation_id"] == "C1"
        assert d["step_number"] == 1
        assert d["input_data"] == {"fuel": "DIESEL"}
        assert d["emission_factor_used"] == {"ef": 2.68}
        assert d["methodology_reference"] == "ISO 14064"


# ---------------------------------------------------------------------------
# TestMultipleCalculations
# ---------------------------------------------------------------------------


class TestMultipleCalculations:
    """Test handling of multiple independent calculations."""

    def test_independent_trails(self, engine):
        """Two calculations maintain independent audit trails."""
        engine.record_step("CALC-A", 1, "s1", {"a": 1}, {"r": 10})
        engine.record_step("CALC-A", 2, "s2", {"a": 2}, {"r": 20})
        engine.record_step("CALC-B", 1, "s1", {"b": 1}, {"r": 100})

        trail_a = engine.get_audit_trail("CALC-A")
        trail_b = engine.get_audit_trail("CALC-B")
        assert len(trail_a) == 2
        assert len(trail_b) == 1
        assert trail_a[0].input_data == {"a": 1}
        assert trail_b[0].input_data == {"b": 1}

    def test_total_count_across_calcs(self, engine):
        """get_entry_count sums across all calculations."""
        engine.record_step("A", 1, "s", {}, {})
        engine.record_step("B", 1, "s", {}, {})
        engine.record_step("C", 1, "s", {}, {})
        assert engine.get_entry_count() == 3

    def test_reports_per_calculation(self, engine):
        """Audit reports are generated per calculation."""
        engine.record_step("CALC-X", 1, "s1", {}, {})
        engine.record_step("CALC-Y", 1, "s1", {}, {})
        engine.record_step("CALC-Y", 2, "s2", {}, {})

        report_x = engine.generate_audit_report("CALC-X")
        report_y = engine.generate_audit_report("CALC-Y")
        assert report_x["total_steps"] == 1
        assert report_y["total_steps"] == 2


# ---------------------------------------------------------------------------
# TestComplianceMapConstants
# ---------------------------------------------------------------------------


class TestComplianceMapConstants:
    """Test COMPLIANCE_MAP constant structure."""

    def test_all_frameworks_present(self):
        """COMPLIANCE_MAP has exactly 6 frameworks."""
        assert len(COMPLIANCE_MAP) == 6

    @pytest.mark.parametrize("fw_key,expected_standard", [
        ("GHG_PROTOCOL", "GHG Protocol Corporate Standard (Rev. 2015)"),
        ("ISO_14064", "ISO 14064-1:2018"),
        ("CSRD_ESRS_E1", "ESRS E1 Climate Change (July 2023)"),
        ("EPA_40CFR98", "40 CFR Part 98 Subpart C"),
        ("EU_ETS", "EU ETS MRR (2018/2066)"),
        ("UK_SECR", "UK SECR (2019)"),
    ])
    def test_framework_standard_names(self, fw_key, expected_standard):
        """Each framework has the correct standard name."""
        assert COMPLIANCE_MAP[fw_key]["standard"] == expected_standard

    @pytest.mark.parametrize("fw_key", [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
        "EPA_40CFR98", "EU_ETS", "UK_SECR",
    ])
    def test_framework_has_chapters(self, fw_key):
        """Each framework has at least one chapter reference."""
        assert len(COMPLIANCE_MAP[fw_key]["chapters"]) >= 1

    @pytest.mark.parametrize("fw_key", [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
        "EPA_40CFR98", "EU_ETS", "UK_SECR",
    ])
    def test_requirement_structure(self, fw_key):
        """Each requirement has id, desc, how fields."""
        for req in COMPLIANCE_MAP[fw_key]["requirements"]:
            assert "id" in req
            assert "desc" in req
            assert "how" in req


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test concurrent access to the engine."""

    def test_concurrent_recording(self):
        """Multiple threads can record steps concurrently without error."""
        engine = AuditTrailEngine()
        errors = []

        def worker(calc_id, num_steps):
            try:
                for i in range(1, num_steps + 1):
                    engine.record_step(calc_id, i, f"step_{i}", {"i": i}, {"r": i * 10})
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(f"CALC-{t}", 5))
            for t in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine.get_entry_count() == 50  # 10 calcs * 5 steps

    def test_concurrent_read_write(self):
        """Concurrent reads and writes do not cause errors."""
        engine = AuditTrailEngine()

        def writer():
            for i in range(20):
                engine.record_step(f"C-{i}", 1, "test", {}, {})

        def reader():
            for _ in range(20):
                engine.get_audit_trail("C-0")
                engine.get_entry_count()
                engine.get_audit_statistics()

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # No exceptions means thread safety is working
        assert engine.get_entry_count() == 20


# ---------------------------------------------------------------------------
# TestComputeHash
# ---------------------------------------------------------------------------


class TestComputeHash:
    """Test the internal _compute_hash static method."""

    def test_deterministic(self):
        """Same data produces same hash."""
        data = {"a": 1, "b": "test"}
        h1 = AuditTrailEngine._compute_hash(data)
        h2 = AuditTrailEngine._compute_hash(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        """Different data produces different hash."""
        h1 = AuditTrailEngine._compute_hash({"a": 1})
        h2 = AuditTrailEngine._compute_hash({"a": 2})
        assert h1 != h2

    def test_hash_is_64_hex_chars(self):
        """Hash is 64 hex characters (SHA-256)."""
        h = AuditTrailEngine._compute_hash({"test": True})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_sorted_keys(self):
        """Hash uses sorted keys for determinism regardless of dict order."""
        h1 = AuditTrailEngine._compute_hash({"b": 2, "a": 1})
        h2 = AuditTrailEngine._compute_hash({"a": 1, "b": 2})
        assert h1 == h2
