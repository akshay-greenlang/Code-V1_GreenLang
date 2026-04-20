# -*- coding: utf-8 -*-
"""Phase F10 — polish, classification extensions, CI invariants."""
from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# factor_family inference
# --------------------------------------------------------------------------


class TestFactorFamilyInference:
    def _module(self):
        import importlib.util, sys
        path = Path("scripts/populate_factor_family.py")
        spec = importlib.util.spec_from_file_location("populate_factor_family", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["populate_factor_family"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_grid_electricity_gets_grid_intensity(self):
        mod = self._module()
        assert mod.infer_factor_family({"fuel_type": "grid_electricity"}) == "grid_intensity"

    def test_truck_freight_gets_transport_lane(self):
        mod = self._module()
        assert mod.infer_factor_family(
            {"activity_type": "road freight truck"}
        ) == "transport_lane"

    def test_landfill_gets_waste_treatment(self):
        mod = self._module()
        assert mod.infer_factor_family(
            {"activity_type": "landfill", "tags": ["waste"]}
        ) == "waste_treatment"

    def test_refrigerant_gets_refrigerant_gwp(self):
        mod = self._module()
        assert mod.infer_factor_family(
            {"fuel_type": "refrigerant R-134a"}
        ) == "refrigerant_gwp"

    def test_steel_gets_material_embodied(self):
        mod = self._module()
        assert mod.infer_factor_family(
            {"fuel_type": "hot-rolled steel coil"}
        ) == "material_embodied"

    def test_biochar_gets_land_use_removals(self):
        mod = self._module()
        assert mod.infer_factor_family(
            {"activity_type": "biochar application"}
        ) == "land_use_removals"

    def test_heating_value_row(self):
        mod = self._module()
        assert mod.infer_factor_family(
            {"fuel_type": "natural_gas", "hcv_mj_per_kg": 55.0}
        ) == "heating_value"

    def test_scope1_default_is_emissions(self):
        mod = self._module()
        assert mod.infer_factor_family({"scope": "1", "fuel_type": "diesel"}) == "emissions"

    def test_unrecognized_falls_back_to_emissions(self):
        mod = self._module()
        assert mod.infer_factor_family({}) == "emissions"


# --------------------------------------------------------------------------
# classifications — GICS / BICS alias
# --------------------------------------------------------------------------


class TestClassificationsExtensions:
    def test_bics_aliases_to_gics(self):
        from greenlang.factors.mapping import map_classification
        r = map_classification("BICS", "55101010")
        assert r.canonical is not None
        assert r.canonical["label"] == "Electric power generation"


# --------------------------------------------------------------------------
# CI invariants checker
# --------------------------------------------------------------------------


class TestInvariantsChecker:
    def _module(self):
        import importlib.util, sys
        path = Path("scripts/check_factor_invariants.py")
        spec = importlib.util.spec_from_file_location("check_factor_invariants", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["check_factor_invariants"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_flags_missing_factor_family(self, tmp_path: Path):
        mod = self._module()
        bad = tmp_path / "emission_factors_test.yaml"
        bad.write_text(
            "- factor_id: X\n"
            "  fuel_type: diesel\n"
            "  factor_status: certified\n"
            "  valid_from: 2024-01-01\n"
            "  release_version: '1'\n",
            encoding="utf-8",
        )
        issues = mod.check_file(bad)
        assert any("missing factor_family" in i for i in issues)

    def test_flags_deprecated_without_replacement(self, tmp_path: Path):
        mod = self._module()
        bad = tmp_path / "emission_factors_test.yaml"
        bad.write_text(
            "- factor_id: Y\n"
            "  fuel_type: diesel\n"
            "  factor_family: emissions\n"
            "  factor_status: deprecated\n"
            "  valid_from: 2024-01-01\n"
            "  release_version: '1'\n",
            encoding="utf-8",
        )
        issues = mod.check_file(bad)
        assert any("deprecated without replacement" in i for i in issues)

    def test_clean_file_passes(self, tmp_path: Path):
        mod = self._module()
        clean = tmp_path / "emission_factors_test.yaml"
        clean.write_text(
            "- factor_id: Z\n"
            "  fuel_type: diesel\n"
            "  factor_family: emissions\n"
            "  factor_status: certified\n"
            "  valid_from: 2024-01-01\n"
            "  release_version: '1'\n",
            encoding="utf-8",
        )
        assert mod.check_file(clean) == []
