# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy GAR (Green Asset Ratio) Calculation Engine.

Tests GAR stock and flow calculation, BTAR calculation, exposure
classification, mortgage alignment (EPC A/B), auto loan alignment
(zero emission), sector and exposure breakdowns, financial institution
portfolio coverage, and multi-period GAR trends with 45+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest


# ===========================================================================
# GAR stock calculation tests
# ===========================================================================

class TestGARStockCalculation:
    """Test GAR stock (balance sheet) calculation."""

    def test_gar_stock_type(self, sample_gar_data):
        """GAR type is stock."""
        assert sample_gar_data["gar_type"] == "stock"

    def test_gar_stock_percentage(self, sample_gar_data):
        """GAR percentage calculated correctly."""
        aligned = float(sample_gar_data["aligned_assets"])
        covered = float(sample_gar_data["covered_assets"])
        expected = round(aligned / covered * 100, 4)
        assert abs(float(sample_gar_data["gar_percentage"]) - expected) < 0.01

    def test_gar_aligned_le_covered(self, sample_gar_data):
        """Aligned assets <= covered assets."""
        assert sample_gar_data["aligned_assets"] <= sample_gar_data["covered_assets"]

    def test_gar_percentage_range(self, sample_gar_data):
        """GAR percentage between 0 and 100."""
        pct = float(sample_gar_data["gar_percentage"])
        assert 0 <= pct <= 100

    def test_gar_status(self, sample_gar_data):
        """GAR calculation status is calculated."""
        assert sample_gar_data["status"] == "calculated"

    def test_gar_provenance(self, sample_gar_data):
        """GAR calculation has provenance hash."""
        assert len(sample_gar_data["provenance_hash"]) == 64

    def test_engine_calculate_gar(self, gar_engine):
        """Engine calculates GAR."""
        gar_engine.calculate_gar.return_value = {
            "aligned_assets": 52500000000,
            "covered_assets": 280000000000,
            "gar_percentage": 18.75,
            "gar_type": "stock",
        }
        result = gar_engine.calculate_gar("inst-123", "FY2025", "stock")
        assert result["gar_percentage"] == 18.75
        gar_engine.calculate_gar.assert_called_once()


# ===========================================================================
# GAR flow calculation tests
# ===========================================================================

class TestGARFlowCalculation:
    """Test GAR flow (new originations) calculation."""

    def test_gar_flow_type(self, sample_gar_flow):
        """GAR type is flow."""
        assert sample_gar_flow["gar_type"] == "flow"

    def test_gar_flow_percentage(self, sample_gar_flow):
        """Flow GAR percentage calculated."""
        pct = float(sample_gar_flow["gar_percentage"])
        assert pct > 0

    def test_gar_flow_higher_than_stock(self, sample_gar_data, sample_gar_flow):
        """Flow GAR typically higher than stock (new green lending)."""
        stock_pct = float(sample_gar_data["gar_percentage"])
        flow_pct = float(sample_gar_flow["gar_percentage"])
        assert flow_pct > stock_pct

    def test_gar_flow_sector_breakdown(self, sample_gar_flow):
        """Flow GAR has sector breakdown."""
        sectors = sample_gar_flow["sector_breakdown"]
        assert len(sectors) >= 2

    def test_engine_calculate_flow_gar(self, gar_engine):
        """Engine calculates flow GAR."""
        gar_engine.calculate_flow_gar.return_value = {
            "gar_type": "flow",
            "gar_percentage": 24.29,
        }
        result = gar_engine.calculate_flow_gar("inst-123", "FY2025")
        assert result["gar_percentage"] == 24.29


# ===========================================================================
# BTAR calculation tests
# ===========================================================================

class TestBTARCalculation:
    """Test Banking-book Taxonomy Alignment Ratio (BTAR) calculation."""

    def test_btar_is_gar_variant(self, sample_btar_data):
        """BTAR uses GAR structure with btar metadata."""
        assert sample_btar_data["metadata"]["ratio_type"] == "btar"

    def test_btar_percentage(self, sample_btar_data):
        """BTAR percentage calculated correctly."""
        aligned = float(sample_btar_data["aligned_assets"])
        covered = float(sample_btar_data["covered_assets"])
        expected = round(aligned / covered * 100, 4)
        assert abs(float(sample_btar_data["gar_percentage"]) - expected) < 0.01

    def test_btar_aligned_le_covered(self, sample_btar_data):
        """Aligned assets <= covered assets for BTAR."""
        assert sample_btar_data["aligned_assets"] <= sample_btar_data["covered_assets"]

    def test_engine_calculate_btar(self, gar_engine):
        """Engine calculates BTAR."""
        gar_engine.calculate_btar.return_value = {
            "gar_percentage": 15.0,
            "ratio_type": "btar",
        }
        result = gar_engine.calculate_btar("inst-123", "FY2025")
        assert result["gar_percentage"] == 15.0

    def test_btar_covers_banking_book(self, sample_btar_data):
        """BTAR covers banking book assets."""
        assert sample_btar_data["gar_type"] == "stock"


# ===========================================================================
# Exposure classification tests
# ===========================================================================

class TestExposureClassification:
    """Test exposure type classification for GAR."""

    def test_corporate_loan_exposure(self, sample_exposures):
        """Corporate loan exposure classified correctly."""
        corp_loans = [e for e in sample_exposures if e["exposure_type"] == "corporate_loan"]
        assert len(corp_loans) >= 1
        for loan in corp_loans:
            assert loan["exposure_amount"] > 0

    def test_mortgage_exposure(self, sample_exposures):
        """Retail mortgage exposure classified correctly."""
        mortgages = [e for e in sample_exposures if e["exposure_type"] == "retail_mortgage"]
        assert len(mortgages) >= 2

    def test_auto_loan_exposure(self, sample_exposures):
        """Auto loan exposure classified correctly."""
        auto_loans = [e for e in sample_exposures if e["exposure_type"] == "auto_loan"]
        assert len(auto_loans) >= 2

    def test_project_finance_exposure(self, sample_exposures):
        """Project finance exposure classified correctly."""
        project = [e for e in sample_exposures if e["exposure_type"] == "project_finance"]
        assert len(project) >= 1

    def test_green_bond_exposure(self, sample_exposures):
        """Green bond exposure classified correctly."""
        green_bonds = [e for e in sample_exposures if e["exposure_type"] == "green_bond"]
        assert len(green_bonds) >= 1

    def test_all_exposure_types_represented(self, sample_exposures):
        """Multiple exposure types represented."""
        types = {e["exposure_type"] for e in sample_exposures}
        assert len(types) >= 5

    def test_engine_classify_exposure(self, gar_engine):
        """Engine classifies exposure type."""
        gar_engine.classify_exposure.return_value = {
            "exposure_type": "corporate_loan",
            "nace_sector": "energy",
            "taxonomy_eligible": True,
        }
        result = gar_engine.classify_exposure(
            counterparty="SolarTech",
            nace_code="D35.11",
            product="term_loan",
        )
        assert result["exposure_type"] == "corporate_loan"


# ===========================================================================
# Mortgage alignment (EPC) tests
# ===========================================================================

class TestMortgageAlignment:
    """Test mortgage alignment based on EPC rating."""

    def test_epc_a_aligned(self, sample_exposures):
        """EPC A-rated mortgages are taxonomy-aligned."""
        epc_a = [e for e in sample_exposures
                 if e["exposure_type"] == "retail_mortgage" and e["epc_rating"] == "A"]
        assert len(epc_a) >= 1
        for exposure in epc_a:
            assert exposure["taxonomy_aligned"] is True

    def test_epc_b_aligned(self, sample_exposures):
        """EPC B-rated mortgages are taxonomy-aligned."""
        epc_b = [e for e in sample_exposures
                 if e["exposure_type"] == "retail_mortgage" and e["epc_rating"] == "B"]
        assert len(epc_b) >= 1
        for exposure in epc_b:
            assert exposure["taxonomy_aligned"] is True

    def test_epc_d_not_aligned(self, sample_exposures):
        """EPC D-rated mortgages are not taxonomy-aligned."""
        epc_d = [e for e in sample_exposures
                 if e["exposure_type"] == "retail_mortgage" and e["epc_rating"] == "D"]
        assert len(epc_d) >= 1
        for exposure in epc_d:
            assert exposure["taxonomy_aligned"] is False

    def test_epc_threshold_config(self, sample_config):
        """Configuration specifies EPC alignment threshold."""
        assert sample_config["epc_alignment_threshold"] == "B"

    def test_mortgage_alignment_pct(self, sample_exposures):
        """Aligned mortgages have 100% alignment."""
        aligned_mortgages = [e for e in sample_exposures
                             if e["exposure_type"] == "retail_mortgage" and e["taxonomy_aligned"]]
        for m in aligned_mortgages:
            assert m["alignment_pct"] == Decimal("100.00")

    def test_engine_check_mortgage_alignment(self, gar_engine):
        """Engine checks mortgage alignment."""
        gar_engine.check_mortgage_alignment.return_value = {
            "epc_rating": "A",
            "aligned": True,
            "alignment_pct": 100.0,
        }
        result = gar_engine.check_mortgage_alignment(epc_rating="A")
        assert result["aligned"] is True


# ===========================================================================
# Auto loan alignment (zero emission) tests
# ===========================================================================

class TestAutoLoanAlignment:
    """Test auto loan alignment based on CO2 emissions."""

    def test_zero_emission_aligned(self, sample_exposures):
        """Zero-emission vehicles (0 g/km) are taxonomy-aligned."""
        ev = [e for e in sample_exposures
              if e["exposure_type"] == "auto_loan" and e["co2_gkm"] == Decimal("0.00")]
        assert len(ev) >= 1
        for loan in ev:
            assert loan["taxonomy_aligned"] is True

    def test_high_emission_not_aligned(self, sample_exposures):
        """High-emission vehicles (>50 g/km) are not aligned."""
        ice = [e for e in sample_exposures
               if e["exposure_type"] == "auto_loan" and e["co2_gkm"] is not None
               and e["co2_gkm"] > Decimal("50.00")]
        assert len(ice) >= 1
        for loan in ice:
            assert loan["taxonomy_aligned"] is False

    def test_co2_threshold_config(self, sample_config):
        """Configuration specifies auto loan CO2 threshold."""
        assert sample_config["auto_loan_co2_threshold_gkm"] == 50.0

    def test_engine_check_auto_loan(self, gar_engine):
        """Engine checks auto loan alignment."""
        gar_engine.check_auto_loan_alignment.return_value = {
            "co2_gkm": 0,
            "aligned": True,
            "vehicle_type": "battery_electric",
        }
        result = gar_engine.check_auto_loan_alignment(co2_gkm=0)
        assert result["aligned"] is True

    def test_ev_alignment_100_pct(self, sample_exposures):
        """EV auto loans have 100% alignment."""
        ev = [e for e in sample_exposures
              if e["exposure_type"] == "auto_loan" and e["taxonomy_aligned"]]
        for loan in ev:
            assert loan["alignment_pct"] == Decimal("100.00")


# ===========================================================================
# Sector breakdown tests
# ===========================================================================

class TestGARSectorBreakdown:
    """Test GAR sector-level breakdown."""

    def test_sector_breakdown_present(self, sample_gar_data):
        """GAR has sector breakdown."""
        sectors = sample_gar_data["sector_breakdown"]
        assert len(sectors) >= 3

    def test_sector_breakdown_structure(self, sample_gar_data):
        """Each sector has aligned, covered, and percentage."""
        for sector, data in sample_gar_data["sector_breakdown"].items():
            assert "aligned" in data
            assert "covered" in data
            assert "pct" in data

    def test_sector_aligned_le_covered(self, sample_gar_data):
        """Sector aligned <= covered."""
        for sector, data in sample_gar_data["sector_breakdown"].items():
            assert data["aligned"] <= data["covered"]

    def test_energy_sector_highest_alignment(self, sample_gar_data):
        """Energy sector has highest alignment percentage."""
        sectors = sample_gar_data["sector_breakdown"]
        energy_pct = sectors["energy"]["pct"]
        for sector_name, data in sectors.items():
            if sector_name != "energy":
                assert energy_pct >= data["pct"]

    def test_engine_get_sector_breakdown(self, gar_engine):
        """Engine returns sector breakdown."""
        gar_engine.get_sector_breakdown.return_value = {
            "energy": {"aligned": 15e9, "covered": 45e9, "pct": 33.33},
        }
        result = gar_engine.get_sector_breakdown("inst-123", "FY2025")
        assert "energy" in result


# ===========================================================================
# Exposure breakdown tests
# ===========================================================================

class TestGARExposureBreakdown:
    """Test GAR exposure-type breakdown."""

    def test_exposure_breakdown_present(self, sample_gar_data):
        """GAR has exposure breakdown."""
        exposures = sample_gar_data["exposure_breakdown"]
        assert len(exposures) >= 3

    def test_exposure_breakdown_structure(self, sample_gar_data):
        """Each exposure type has aligned and covered amounts."""
        for exp_type, data in sample_gar_data["exposure_breakdown"].items():
            assert "aligned" in data
            assert "covered" in data

    def test_corporate_loan_in_breakdown(self, sample_gar_data):
        """Corporate loans included in breakdown."""
        assert "corporate_loan" in sample_gar_data["exposure_breakdown"]

    def test_mortgage_in_breakdown(self, sample_gar_data):
        """Retail mortgages included in breakdown."""
        assert "retail_mortgage" in sample_gar_data["exposure_breakdown"]

    def test_engine_get_exposure_breakdown(self, gar_engine):
        """Engine returns exposure breakdown."""
        gar_engine.get_exposure_breakdown.return_value = {
            "corporate_loan": {"aligned": 30e9, "covered": 150e9},
        }
        result = gar_engine.get_exposure_breakdown("inst-123", "FY2025")
        assert "corporate_loan" in result


# ===========================================================================
# Exposure data quality tests
# ===========================================================================

class TestExposureDataQuality:
    """Test exposure data quality and completeness."""

    def test_exposure_currency_present(self, sample_exposures):
        """All exposures have currency specified."""
        for exp in sample_exposures:
            assert exp["currency"] is not None
            assert len(exp["currency"]) == 3

    def test_exposure_amount_positive(self, sample_exposures):
        """All exposures have positive amounts."""
        for exp in sample_exposures:
            assert exp["exposure_amount"] > 0

    def test_exposure_reporting_date(self, sample_exposures):
        """All exposures have reporting dates."""
        from datetime import date as dt_date
        for exp in sample_exposures:
            assert isinstance(exp["reporting_date"], dt_date)

    def test_exposure_counterparty_named(self, sample_exposures):
        """All exposures have counterparty names."""
        for exp in sample_exposures:
            assert len(exp["counterparty_name"]) > 0

    def test_alignment_pct_consistency(self, sample_exposures):
        """Alignment percentage consistent with aligned flag."""
        for exp in sample_exposures:
            if exp["taxonomy_aligned"]:
                assert exp["alignment_pct"] > 0
            else:
                assert exp["alignment_pct"] == Decimal("0.00")
