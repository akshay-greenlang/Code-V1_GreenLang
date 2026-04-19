# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Demo Configuration Tests
=================================================================

Tests demo_config.yaml: YAML structure, battery identity, carbon
footprint, recycled content, performance, supply chain, labelling,
end-of-life, conformity, and reporting sections.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

from pathlib import Path
from typing import Any, Dict

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
DEMO_CONFIG_PATH = PACK_ROOT / "config" / "demo" / "demo_config.yaml"

# ---------------------------------------------------------------------------
# Load YAML
# ---------------------------------------------------------------------------

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]


def _load_demo_config() -> Dict[str, Any]:
    if yaml is None:
        pytest.skip("PyYAML not installed")
    if not DEMO_CONFIG_PATH.exists():
        pytest.skip(f"demo_config.yaml not found at {DEMO_CONFIG_PATH}")
    with open(DEMO_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def demo() -> Dict[str, Any]:
    return _load_demo_config()


# =========================================================================
# File Existence and Top-Level Structure
# =========================================================================


class TestDemoStructure:
    """Validate demo config file exists and has expected sections."""

    def test_demo_file_exists(self):
        assert DEMO_CONFIG_PATH.exists()

    def test_top_level_sections(self, demo):
        expected = [
            "battery",
            "preset",
            "carbon_footprint",
            "recycled_content",
            "performance",
            "supply_chain",
            "labelling",
            "eol",
            "conformity",
            "reporting",
        ]
        for section in expected:
            assert section in demo, f"Missing top-level section: {section}"

    def test_preset_value(self, demo):
        assert demo["preset"] == "ev_battery"


# =========================================================================
# Battery Identity
# =========================================================================


class TestDemoBatteryIdentity:
    """Validate battery identity section."""

    def test_unique_identifier(self, demo):
        bat = demo["battery"]
        assert isinstance(bat["unique_identifier"], str)
        assert len(bat["unique_identifier"]) > 10

    def test_model(self, demo):
        bat = demo["battery"]
        assert bat["model"] == "EB-75-NMC811"

    def test_manufacturer(self, demo):
        bat = demo["battery"]
        assert bat["manufacturer"] == "EuroBattery GmbH"

    def test_manufacturing_country(self, demo):
        bat = demo["battery"]
        assert bat["manufacturing_country"] == "DE"

    def test_category_is_ev(self, demo):
        bat = demo["battery"]
        assert bat["category"] == "EV"

    def test_chemistry(self, demo):
        bat = demo["battery"]
        assert bat["chemistry"] == "NMC811"

    def test_rated_capacity_kwh(self, demo):
        bat = demo["battery"]
        assert bat["rated_capacity_kwh"] == 75.0

    def test_rated_capacity_ah(self, demo):
        bat = demo["battery"]
        assert bat["rated_capacity_ah"] == 201.0

    def test_nominal_voltage(self, demo):
        bat = demo["battery"]
        assert bat["nominal_voltage_v"] == 373.0

    def test_weight(self, demo):
        bat = demo["battery"]
        assert bat["weight_kg"] == 485.0

    def test_dimensions(self, demo):
        bat = demo["battery"]
        dims = bat.get("dimensions_mm", {})
        assert "length" in dims
        assert "width" in dims
        assert "height" in dims

    def test_cells_count(self, demo):
        bat = demo["battery"]
        assert bat["cells_count"] == 288

    def test_modules_count(self, demo):
        bat = demo["battery"]
        assert bat["modules_count"] == 24

    def test_intended_market(self, demo):
        bat = demo["battery"]
        assert bat["intended_market"] == "EU"


# =========================================================================
# Carbon Footprint (Art. 7)
# =========================================================================


class TestDemoCarbonFootprint:
    """Validate carbon footprint sample data."""

    def test_total_kgco2e_per_kwh(self, demo):
        cf = demo["carbon_footprint"]
        assert cf["total_kgco2e_per_kwh"] == 61.3

    def test_lifecycle_breakdown_keys(self, demo):
        cf = demo["carbon_footprint"]
        lb = cf["lifecycle_breakdown"]
        expected_keys = [
            "raw_material_extraction_kgco2e_per_kwh",
            "manufacturing_kgco2e_per_kwh",
            "distribution_kgco2e_per_kwh",
            "end_of_life_kgco2e_per_kwh",
        ]
        for key in expected_keys:
            assert key in lb, f"Missing lifecycle key: {key}"

    def test_lifecycle_stages_sum_approximately(self, demo):
        cf = demo["carbon_footprint"]
        lb = cf["lifecycle_breakdown"]
        total = sum(lb.values())
        declared = cf["total_kgco2e_per_kwh"]
        # Allow small rounding tolerance
        assert abs(total - declared) < 1.0, (
            f"Lifecycle stages sum ({total}) should approximate total ({declared})"
        )

    def test_performance_class(self, demo):
        cf = demo["carbon_footprint"]
        assert cf["performance_class"] == "CLASS_B"

    def test_methodology(self, demo):
        cf = demo["carbon_footprint"]
        assert cf["methodology"] == "EU_BATTERY_REG_DA"

    def test_third_party_verifier(self, demo):
        cf = demo["carbon_footprint"]
        assert isinstance(cf["third_party_verifier"], str)
        assert len(cf["third_party_verifier"]) > 3

    def test_data_quality_rating(self, demo):
        cf = demo["carbon_footprint"]
        assert 1.0 <= cf["data_quality_rating"] <= 5.0

    def test_renewable_percentage(self, demo):
        cf = demo["carbon_footprint"]
        mix = cf.get("manufacturing_electricity_mix", {})
        assert mix["renewable_pct"] + mix["grid_pct"] == 100.0


# =========================================================================
# Recycled Content (Art. 8)
# =========================================================================


class TestDemoRecycledContent:
    """Validate recycled content data."""

    def test_cobalt_pct(self, demo):
        rc = demo["recycled_content"]
        assert rc["cobalt_pct"] == 18.2

    def test_lithium_pct(self, demo):
        rc = demo["recycled_content"]
        assert rc["lithium_pct"] == 8.5

    def test_nickel_pct(self, demo):
        rc = demo["recycled_content"]
        assert rc["nickel_pct"] == 7.1

    def test_lead_pct(self, demo):
        rc = demo["recycled_content"]
        assert rc["lead_pct"] == 0.0

    def test_verification_method(self, demo):
        rc = demo["recycled_content"]
        assert rc["verification_method"] == "MASS_BALANCE"

    def test_phase_1_2031_compliant(self, demo):
        rc = demo["recycled_content"]
        assert rc["phase_1_2031_compliant"] is True

    def test_cobalt_meets_2031_target(self, demo):
        rc = demo["recycled_content"]
        # 2031 target for cobalt is 16%
        assert rc["cobalt_pct"] >= 16.0

    def test_lithium_meets_2031_target(self, demo):
        rc = demo["recycled_content"]
        # 2031 target for lithium is 6%
        assert rc["lithium_pct"] >= 6.0

    def test_nickel_meets_2031_target(self, demo):
        rc = demo["recycled_content"]
        # 2031 target for nickel is 6%
        assert rc["nickel_pct"] >= 6.0


# =========================================================================
# Performance and Durability (Art. 10, 14)
# =========================================================================


class TestDemoPerformance:
    """Validate performance sample data."""

    def test_initial_capacity_kwh(self, demo):
        perf = demo["performance"]
        assert perf["initial_rated_capacity_kwh"] == 75.0

    def test_initial_capacity_ah(self, demo):
        perf = demo["performance"]
        assert perf["initial_rated_capacity_ah"] == 201.0

    def test_power_capability(self, demo):
        perf = demo["performance"]
        assert perf["initial_power_capability_kw"] == 220.0

    def test_expected_cycle_life(self, demo):
        perf = demo["performance"]
        assert perf["expected_cycle_life"] >= 1000

    def test_warranty_years(self, demo):
        perf = demo["performance"]
        assert perf["warranty_years"] == 8

    def test_round_trip_efficiency(self, demo):
        perf = demo["performance"]
        assert 80.0 <= perf["round_trip_efficiency_pct"] <= 100.0

    def test_operating_temperature_range(self, demo):
        perf = demo["performance"]
        temp = perf.get("operating_temperature_range_c", {})
        assert temp["min"] <= 0
        assert temp["max"] >= 40

    def test_test_standard(self, demo):
        perf = demo["performance"]
        assert "IEC" in perf["test_standard"]


# =========================================================================
# Supply Chain Due Diligence (Art. 48)
# =========================================================================


class TestDemoSupplyChain:
    """Validate supply chain DD data."""

    def test_framework(self, demo):
        sc = demo["supply_chain"]
        assert sc["framework"] == "OECD_MINERALS"

    def test_materials_exist(self, demo):
        sc = demo["supply_chain"]
        materials = sc.get("materials", {})
        for mineral in ["cobalt", "lithium", "nickel", "natural_graphite"]:
            assert mineral in materials, f"Missing mineral: {mineral}"

    def test_cobalt_source_countries(self, demo):
        sc = demo["supply_chain"]
        cobalt = sc["materials"]["cobalt"]
        assert isinstance(cobalt["source_countries"], list)
        assert "CD" in cobalt["source_countries"]

    def test_cobalt_risk_level(self, demo):
        sc = demo["supply_chain"]
        cobalt = sc["materials"]["cobalt"]
        assert cobalt["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_lithium_risk_level_low(self, demo):
        sc = demo["supply_chain"]
        lithium = sc["materials"]["lithium"]
        assert lithium["risk_level"] == "LOW"

    def test_tier_suppliers_assessed(self, demo):
        sc = demo["supply_chain"]
        assert sc["tier_1_suppliers_assessed"] > 0
        assert sc["tier_2_suppliers_assessed"] > 0

    def test_each_material_has_refiner(self, demo):
        sc = demo["supply_chain"]
        for name, mineral in sc["materials"].items():
            assert "refiner" in mineral, f"Material '{name}' missing 'refiner'"
            assert "refiner_country" in mineral, f"Material '{name}' missing 'refiner_country'"

    def test_certifications(self, demo):
        sc = demo["supply_chain"]
        for name, mineral in sc["materials"].items():
            assert "certification" in mineral, f"Material '{name}' missing certification"


# =========================================================================
# Labelling (Art. 13)
# =========================================================================


class TestDemoLabelling:
    """Validate labelling compliance data."""

    def test_ce_marking(self, demo):
        lab = demo["labelling"]
        assert lab["ce_marking"] is True

    def test_qr_code(self, demo):
        lab = demo["labelling"]
        assert lab["qr_code"] is True

    def test_qr_code_url(self, demo):
        lab = demo["labelling"]
        assert isinstance(lab["qr_code_url"], str)
        assert lab["qr_code_url"].startswith("https://")

    def test_collection_symbol(self, demo):
        lab = demo["labelling"]
        assert lab["collection_symbol"] is True

    def test_capacity_label(self, demo):
        lab = demo["labelling"]
        assert lab["capacity_label"] is True
        assert "kWh" in lab["capacity_label_value"]

    def test_carbon_footprint_class_label(self, demo):
        lab = demo["labelling"]
        assert lab["carbon_footprint_class_label"] is True
        assert lab["carbon_footprint_class_value"] == "CLASS_B"

    def test_all_elements_compliant(self, demo):
        lab = demo["labelling"]
        assert lab["all_elements_compliant"] is True


# =========================================================================
# End-of-Life (EOL)
# =========================================================================


class TestDemoEndOfLife:
    """Validate end-of-life data."""

    def test_take_back_registered(self, demo):
        eol = demo["eol"]
        assert eol["take_back_registered"] is True

    def test_designated_recycler(self, demo):
        eol = demo["eol"]
        assert isinstance(eol["designated_recycler"], str)

    def test_second_life_eligible(self, demo):
        eol = demo["eol"]
        assert eol["second_life_eligible"] is True

    def test_recycling_efficiency_target(self, demo):
        eol = demo["eol"]
        assert eol["recycling_efficiency_target_pct"] >= 65.0

    def test_material_recovery_commitments(self, demo):
        eol = demo["eol"]
        recovery = eol.get("material_recovery_commitments", {})
        assert recovery["cobalt_recovery_pct"] >= 90.0
        assert recovery["lithium_recovery_pct"] >= 50.0
        assert recovery["nickel_recovery_pct"] >= 90.0


# =========================================================================
# Conformity Assessment
# =========================================================================


class TestDemoConformity:
    """Validate conformity assessment data."""

    def test_selected_module(self, demo):
        conf = demo["conformity"]
        assert conf["selected_module"] == "MODULE_H"

    def test_notified_body(self, demo):
        conf = demo["conformity"]
        assert isinstance(conf["notified_body"], str)
        assert len(conf["notified_body"]) > 3

    def test_notified_body_number(self, demo):
        conf = demo["conformity"]
        assert conf["notified_body_number"] is not None

    def test_certificate_number(self, demo):
        conf = demo["conformity"]
        assert isinstance(conf["certificate_number"], str)

    def test_quality_management_system(self, demo):
        conf = demo["conformity"]
        assert "IATF" in conf["quality_management_system"] or "ISO" in conf["quality_management_system"]

    def test_technical_file_complete(self, demo):
        conf = demo["conformity"]
        assert conf["technical_file_complete"] is True


# =========================================================================
# Reporting Configuration
# =========================================================================


class TestDemoReporting:
    """Validate reporting configuration."""

    def test_output_formats(self, demo):
        rep = demo["reporting"]
        formats = rep["output_formats"]
        assert isinstance(formats, list)
        assert "PDF" in formats
        assert "JSON" in formats

    def test_sha256_provenance(self, demo):
        rep = demo["reporting"]
        assert rep["sha256_provenance"] is True

    def test_audit_trail_report(self, demo):
        rep = demo["reporting"]
        assert rep["audit_trail_report"] is True

    def test_retention_years(self, demo):
        rep = demo["reporting"]
        assert rep["retention_years"] >= 10

    def test_passport_api_format(self, demo):
        rep = demo["reporting"]
        assert rep["passport_api_format"] == "JSON_LD"


# =========================================================================
# Cross-Section Consistency
# =========================================================================


class TestDemoCrossSectionConsistency:
    """Validate consistency across demo config sections."""

    def test_capacity_consistent(self, demo):
        """Battery capacity in battery section matches performance section."""
        bat_kwh = demo["battery"]["rated_capacity_kwh"]
        perf_kwh = demo["performance"]["initial_rated_capacity_kwh"]
        assert bat_kwh == perf_kwh

    def test_capacity_ah_consistent(self, demo):
        """Battery Ah in battery section matches performance section."""
        bat_ah = demo["battery"]["rated_capacity_ah"]
        perf_ah = demo["performance"]["initial_rated_capacity_ah"]
        assert bat_ah == perf_ah

    def test_carbon_footprint_class_matches_label(self, demo):
        """Carbon footprint class in CF section matches labelling section."""
        cf_class = demo["carbon_footprint"]["performance_class"]
        label_class = demo["labelling"]["carbon_footprint_class_value"]
        assert cf_class == label_class

    def test_battery_category_matches_preset(self, demo):
        """EV battery category should use ev_battery preset."""
        category = demo["battery"]["category"]
        preset = demo["preset"]
        assert category == "EV"
        assert preset == "ev_battery"
