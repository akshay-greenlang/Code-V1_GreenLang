# -*- coding: utf-8 -*-
"""Wave 5 catalog-expansion parser tests.

Covers the five Wave-5 parsers that land Safe-to-Certify ingest for:
- EU CBAM Annex IV (flat sector rollup)
- EXIOBASE v3 multi-regional EE-IO
- UK ONS/DEFRA Environmental Accounts (SIC-based spend proxies)
- EEA European waste statistics
- IPCC 2006 Vol 5 (Waste) — India-parameterised defaults

Each parser must:
 1. Produce >= 1 record for the canonical payload shape
 2. Emit every field in the canonical-v1 envelope (16 fields)
 3. Carry license + citation + attribution through to each record
 4. Stamp `source_id` matching the Wave-5 registry row
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.data.canonical_v2 import FactorFamily
from greenlang.data.emission_factor_record import (
    Boundary,
    GWPSet,
    GeographyLevel,
    Methodology,
    Scope,
)

from greenlang.factors.ingestion.parsers.cbam_default_values import (
    parse_cbam_default_values,
)
from greenlang.factors.ingestion.parsers.defra_uk_env_accounts import (
    parse_defra_uk_env_accounts,
)
from greenlang.factors.ingestion.parsers.eea_waste_stats import (
    parse_eea_waste_stats,
)
from greenlang.factors.ingestion.parsers.exiobase_v3 import (
    parse_exiobase_v3,
)
from greenlang.factors.ingestion.parsers.ipcc_waste_vol5_in import (
    parse_ipcc_waste_vol5_in,
)

# ---------------------------------------------------------------------------
# Canonical-v1 envelope (16 mandatory fields)
# ---------------------------------------------------------------------------
# Shared contract across every Wave-5 parser. Subset covered by the N5 gate
# in bootstrap.py lives inside this list.
CANONICAL_V1_FIELDS = (
    "factor_id",
    "fuel_type",
    "unit",
    "geography",
    "geography_level",
    "vectors",
    "gwp_100yr",
    "scope",
    "boundary",
    "provenance",
    "valid_from",
    "valid_to",
    "uncertainty_95ci",
    "dqs",
    "license_info",
    "source_id",
)


def _assert_canonical_v1(rec: Dict[str, Any], expected_source_id: str) -> None:
    """Check all 16 envelope fields present + license + source_id."""
    missing = [k for k in CANONICAL_V1_FIELDS if k not in rec]
    assert not missing, (
        f"record missing canonical-v1 fields {missing}; "
        f"factor_id={rec.get('factor_id')}"
    )

    # Enum-style fields validate as non-empty strings
    assert rec["geography_level"] in {lv.value for lv in GeographyLevel}
    assert rec["scope"] in {s.value for s in Scope}
    assert rec["boundary"] in {b.value for b in Boundary}

    # Vectors must carry at least a CO2 key with a non-None numeric value
    vectors = rec["vectors"]
    assert isinstance(vectors, dict)
    assert "CO2" in vectors
    assert vectors["CO2"] is not None

    # GWP block shape
    gwp = rec["gwp_100yr"]
    assert isinstance(gwp, dict)
    assert gwp.get("gwp_set") in {g.value for g in GWPSet}

    # Provenance chain
    prov = rec["provenance"]
    assert isinstance(prov, dict)
    assert prov.get("source_org")
    assert prov.get("source_year")
    assert prov.get("methodology") in {m.value for m in Methodology}

    # License + attribution carry-through
    lic = rec["license_info"]
    assert isinstance(lic, dict)
    assert lic.get("license")
    assert lic.get("attribution_required") is True

    # source_id matches the registry row
    assert rec["source_id"] == expected_source_id


# ---------------------------------------------------------------------------
# CBAM flat sector rollup
# ---------------------------------------------------------------------------

class TestCBAMDefaultValues:
    @pytest.fixture
    def payload(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "source": "EU Commission",
                "version": "2024",
            },
            "sectors": {
                "steel": {
                    "unit": "kg_product",
                    "by_country": {
                        "CN": {"direct": 2.20, "indirect": 0.60},
                        "IN": {"direct": 2.55, "indirect": 0.65},
                    },
                },
                "aluminium": {
                    "unit": "kg_product",
                    "by_country": {
                        "CN": {"direct": 2.40, "indirect": 12.50},
                    },
                },
            },
        }

    def test_parses_at_least_one(self, payload):
        out = parse_cbam_default_values(payload)
        assert len(out) >= 1

    def test_canonical_envelope(self, payload):
        out = parse_cbam_default_values(payload)
        assert len(out) == 3
        for rec in out:
            _assert_canonical_v1(rec, "cbam_default_values")
            assert rec["factor_family"] == FactorFamily.MATERIAL_EMBODIED.value

    def test_sector_rollup_shape(self, payload):
        out = parse_cbam_default_values(payload)
        ids = {r["factor_id"] for r in out}
        assert "EF:CBAM:steel:CN:2024:v1" in ids
        assert "EF:CBAM:steel:IN:2024:v1" in ids
        assert "EF:CBAM:aluminium:CN:2024:v1" in ids


# ---------------------------------------------------------------------------
# EXIOBASE v3
# ---------------------------------------------------------------------------

class TestExiobaseV3:
    @pytest.fixture
    def payload(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "source": "EXIOBASE Consortium",
                "version": "3.8.2",
                "vintage_year": 2022,
            },
            "sectors": [
                {
                    "sector_code": "1712",
                    "sector_name": "Manufacture of paper and paper products",
                    "region": "EU",
                    "kg_co2e_per_eur": 0.65,
                },
                {
                    "sector_code": "J62",
                    "sector_name": "Computer programming",
                    "region": "IN",
                    "kg_co2e_per_usd": 0.14,
                },
            ],
        }

    def test_parses_at_least_one(self, payload):
        out = parse_exiobase_v3(payload)
        assert len(out) >= 2

    def test_canonical_envelope(self, payload):
        out = parse_exiobase_v3(payload)
        for rec in out:
            _assert_canonical_v1(rec, "exiobase_v3")
            assert rec["factor_family"] == FactorFamily.FINANCE_PROXY.value


# ---------------------------------------------------------------------------
# DEFRA UK Environmental Accounts
# ---------------------------------------------------------------------------

class TestDefraUKEnvAccounts:
    @pytest.fixture
    def payload(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "source": "UK ONS / DEFRA",
                "vintage_year": 2022,
            },
            "sectors": [
                {"sic_code": "69", "sic_name": "Legal and accounting", "kg_co2e_per_gbp": 0.05},
                {"sic_code": "42", "sic_name": "Civil engineering", "kg_co2e_per_gbp": 0.52},
                {"sic_code": "46", "sic_name": "Wholesale trade", "kg_co2e_per_gbp": 0.22},
            ],
        }

    def test_parses_at_least_one(self, payload):
        assert len(parse_defra_uk_env_accounts(payload)) >= 1

    def test_canonical_envelope(self, payload):
        out = parse_defra_uk_env_accounts(payload)
        assert len(out) == 3
        for rec in out:
            _assert_canonical_v1(rec, "defra_uk_env_accounts")
            assert rec["factor_family"] == FactorFamily.FINANCE_PROXY.value
            assert rec["unit"] == "gbp"

    def test_id_shape(self, payload):
        out = parse_defra_uk_env_accounts(payload)
        ids = {r["factor_id"] for r in out}
        assert "EF:DEFRA_IO:69:UK:2022:v1" in ids


# ---------------------------------------------------------------------------
# EEA European waste statistics
# ---------------------------------------------------------------------------

class TestEEAWasteStats:
    @pytest.fixture
    def payload(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "source": "EEA",
                "vintage_year": 2022,
            },
            "streams": [
                {"treatment": "recycling", "material": "plastic", "country": "EU", "kg_co2e_per_tonne": 320.0},
                {"treatment": "incineration", "material": "msw", "country": "DE", "kg_co2e_per_tonne": 365.0},
                {"treatment": "landfill", "material": "msw", "country": "FR", "kg_co2e_per_tonne": 510.0},
            ],
        }

    def test_parses_at_least_one(self, payload):
        out = parse_eea_waste_stats(payload)
        assert len(out) >= 1

    def test_canonical_envelope(self, payload):
        out = parse_eea_waste_stats(payload)
        assert len(out) == 3
        for rec in out:
            _assert_canonical_v1(rec, "eea_waste_stats")
            assert rec["factor_family"] == FactorFamily.WASTE_TREATMENT.value


# ---------------------------------------------------------------------------
# IPCC 2006 Vol 5 (Waste) — India
# ---------------------------------------------------------------------------

class TestIPCCWasteVol5IN:
    @pytest.fixture
    def payload(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "source": "IPCC 2006 GL Vol 5",
                "vintage_year": 2006,
                "country": "IN",
                "climate_zone": "tropical_wet",
            },
            "categories": {
                "swds": [
                    {"stream": "msw_food", "doc_fraction": 0.15, "mcf": 0.8, "kg_ch4_per_tonne": 80.0},
                    {"stream": "msw_paper", "doc_fraction": 0.40, "mcf": 0.8, "kg_ch4_per_tonne": 213.0},
                ],
                "incineration": [
                    {"stream": "msw", "kg_co2_fossil_per_tonne": 900.0, "fossil_fraction": 0.39},
                ],
                "biological": [
                    {"stream": "compost_msw", "kg_ch4_per_tonne": 4.0, "kg_n2o_per_tonne": 0.3},
                ],
                "wastewater": [
                    {"stream": "domestic", "mcf": 0.8, "kg_ch4_per_m3": 0.288, "kg_n2o_per_m3": 0.0003},
                ],
            },
        }

    def test_parses_at_least_one(self, payload):
        out = parse_ipcc_waste_vol5_in(payload)
        assert len(out) >= 1

    def test_covers_all_four_categories(self, payload):
        out = parse_ipcc_waste_vol5_in(payload)
        assert len(out) == 5  # 2 swds + 1 incin + 1 bio + 1 wastewater
        id_tokens = {r["factor_id"].split(":")[2].split("_")[0] for r in out}
        assert "swds" in id_tokens
        assert "incineration" in id_tokens
        assert "biological" in id_tokens
        assert "wastewater" in id_tokens

    def test_canonical_envelope(self, payload):
        out = parse_ipcc_waste_vol5_in(payload)
        for rec in out:
            _assert_canonical_v1(rec, "ipcc_waste_vol5_in")
            assert rec["factor_family"] == FactorFamily.WASTE_TREATMENT.value
            assert rec["geography"] == "IN"

    def test_provenance_carries_ipcc_attribution(self, payload):
        out = parse_ipcc_waste_vol5_in(payload)
        for rec in out:
            prov = rec["provenance"]
            assert "IPCC" in prov["source_org"]
            assert rec["license_info"]["license"] == "IPCC-Guideline"


# ---------------------------------------------------------------------------
# Bootstrap wiring: each Wave-5 source must be registered in SOURCE_SPECS
# ---------------------------------------------------------------------------

def test_bootstrap_source_specs_wire_wave5():
    from greenlang.factors.ingestion.bootstrap import SOURCE_SPECS

    wired = {s.source_id for s in SOURCE_SPECS}
    wave5 = {
        "cbam_default_values",
        "exiobase_v3",
        "defra_uk_env_accounts",
        "eea_waste_stats",
        "ipcc_waste_vol5_in",
    }
    missing = wave5 - wired
    assert not missing, f"Wave-5 sources missing from SOURCE_SPECS: {missing}"
