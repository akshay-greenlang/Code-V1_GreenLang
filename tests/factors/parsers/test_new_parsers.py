# -*- coding: utf-8 -*-
"""Phase F9 — New source parsers (PACT / EC3 / freight / PCAF / LSR / waste)."""
from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
    RedistributionClass,
)


# --------------------------------------------------------------------------
# PACT
# --------------------------------------------------------------------------


class TestPACT:
    def test_product_footprint_parses(self):
        from greenlang.factors.ingestion.parsers.pact_product_data import parse_pact_rows

        rows = [
            {
                "id": "urn:gl:pact:product:STEEL-001",
                "productName": "Hot-rolled steel coil",
                "productCategoryCpc": "41237",
                "pcf": {
                    "declaredUnit": "kg",
                    "pCfExcludingBiogenic": "2.34",
                    "biogenicCarbonEmissions": "0.05",
                    "geographyCountrySubdivision": "DE",
                },
                "companyName": "Acme Steel",
                "periodCoveredStart": "2024-01-01",
                "periodCoveredEnd": "2024-12-31",
                "version": 2,
                "pcfSpec": "2.0.0",
            }
        ]
        records = parse_pact_rows(rows)
        assert len(records) == 1
        rec = records[0]
        assert rec.factor_family == FactorFamily.MATERIAL_EMBODIED.value
        assert rec.method_profile == MethodProfile.PRODUCT_CARBON.value
        assert rec.formula_type == FormulaType.LCA.value
        assert rec.redistribution_class == RedistributionClass.RESTRICTED.value

    def test_invalid_pact_row_skipped(self):
        from greenlang.factors.ingestion.parsers.pact_product_data import parse_pact_rows
        assert parse_pact_rows([{"broken": True}]) == []


# --------------------------------------------------------------------------
# EC3 / EPD
# --------------------------------------------------------------------------


class TestEC3:
    def test_epd_parses(self):
        from greenlang.factors.ingestion.parsers.ec3_epd import parse_ec3_epd_rows

        rows = [
            {
                "epd_id": "EPD-REG-20240001-EN",
                "product_name": "Ready-mix concrete 30 MPa",
                "category": "concrete",
                "functional_unit": "m3",
                "declared_unit_co2e_kg": 278.0,
                "modules_reported": ["A1", "A2", "A3"],
                "program_operator": "EPD International",
                "verification_date": "2024-03-15",
                "valid_until": "2029-03-15",
                "country": "SE",
                "manufacturer": "NCC AB",
            }
        ]
        rec = parse_ec3_epd_rows(rows)[0]
        assert rec.factor_family == FactorFamily.MATERIAL_EMBODIED.value
        assert rec.boundary.value == "cradle_to_gate"
        assert rec.redistribution_class == RedistributionClass.LICENSED.value

    def test_full_lifecycle_modules_bump_boundary(self):
        from greenlang.factors.ingestion.parsers.ec3_epd import parse_ec3_epd_rows
        rec = parse_ec3_epd_rows(
            [
                {
                    "epd_id": "X", "product_name": "X", "category": "X",
                    "functional_unit": "kg",
                    "declared_unit_co2e_kg": 1.0,
                    "modules_reported": ["A1", "A2", "A3", "C4"],
                    "program_operator": "OP", "verification_date": "2024-01-01",
                    "valid_until": "2029-01-01", "country": "DE", "manufacturer": "M",
                }
            ]
        )[0]
        assert rec.boundary.value == "cradle_to_grave"


# --------------------------------------------------------------------------
# Freight lanes
# --------------------------------------------------------------------------


class TestFreight:
    def test_lane_parses(self):
        from greenlang.factors.ingestion.parsers.freight_lanes import (
            parse_freight_lane_rows,
        )

        rows = [
            {
                "lane_id": "TRUCK-40T-EU-DIESEL",
                "mode": "road",
                "vehicle_class": "heavy_truck_40t",
                "fuel": "diesel",
                "payload_utilization": 0.7,
                "empty_running_factor": 1.25,
                "wtt_gco2e_per_tkm": 21.0,
                "ttw_gco2e_per_tkm": 75.0,
                "wtw_gco2e_per_tkm": 96.0,
                "valid_year": 2024,
                "geography": "EU",
            }
        ]
        rec = parse_freight_lane_rows(rows)[0]
        assert rec.factor_family == FactorFamily.TRANSPORT_LANE.value
        assert rec.method_profile == MethodProfile.FREIGHT_ISO_14083.value
        assert rec.formula_type == FormulaType.TRANSPORT_CHAIN.value
        # 96 g/tkm = 0.096 kg/tkm
        assert float(rec.vectors.CO2) == pytest.approx(0.096, abs=1e-4)


# --------------------------------------------------------------------------
# PCAF
# --------------------------------------------------------------------------


class TestPCAF:
    def test_proxy_row_parses(self):
        from greenlang.factors.ingestion.parsers.pcaf_proxies import parse_pcaf_rows

        rows = [
            {
                "asset_class": "listed_equity",
                "sector_nace": "D35.11",
                "geography": "EU",
                "intensity_tco2e_per_m_eur_revenue": 450.0,
                "pcaf_dqs_score": 4,
                "year": 2024,
            }
        ]
        rec = parse_pcaf_rows(rows)[0]
        assert rec.factor_family == FactorFamily.FINANCE_PROXY.value
        assert rec.method_profile == MethodProfile.FINANCE_PROXY.value
        assert rec.formula_type == FormulaType.SPEND_PROXY.value
        # 450 tCO2e/M€ = 0.45 kg CO2e/€ (coefficient check)
        assert float(rec.vectors.CO2) == pytest.approx(0.45, rel=1e-6)


# --------------------------------------------------------------------------
# LSR / Removals
# --------------------------------------------------------------------------


class TestLSR:
    def test_removal_row_parses(self):
        from greenlang.factors.ingestion.parsers.lsr_removals import parse_lsr_rows

        rows = [
            {
                "activity_id": "biochar-application-2024",
                "activity_type": "biochar_application",
                "removal_rate_kg_co2e_per_kg": -2.5,
                "permanence_class": "long_term",
                "reversal_risk": "low",
                "geography": "EU",
                "year": 2024,
            }
        ]
        rec = parse_lsr_rows(rows)[0]
        assert rec.factor_family == FactorFamily.LAND_USE_REMOVALS.value
        assert rec.method_profile == MethodProfile.LAND_REMOVALS.value
        assert rec.formula_type == FormulaType.CARBON_BUDGET.value
        # Magnitude stored in CO2; sign carried via the 'removal' tag.
        assert float(rec.vectors.CO2) == pytest.approx(2.5)
        assert "removal" in rec.tags


# --------------------------------------------------------------------------
# Waste treatment
# --------------------------------------------------------------------------


class TestWaste:
    def test_landfill_row_parses(self):
        from greenlang.factors.ingestion.parsers.waste_treatment import parse_waste_rows

        rows = [
            {
                "waste_type": "mixed_msw",
                "treatment": "landfill",
                "country": "US",
                "ch4_kg_per_t": 62.0,
                "co2_fossil_kg_per_t": 0.0,
                "biogenic_co2_kg_per_t": 320.0,
                "n2o_kg_per_t": 0.0,
                "year": 2024,
            }
        ]
        rec = parse_waste_rows(rows)[0]
        assert rec.factor_family == FactorFamily.WASTE_TREATMENT.value
        # biogenic reported separately
        assert float(rec.vectors.biogenic_CO2) == pytest.approx(320.0)
        assert float(rec.vectors.CH4) == pytest.approx(62.0)
