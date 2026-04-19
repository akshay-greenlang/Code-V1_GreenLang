# -*- coding: utf-8 -*-
"""
Unit tests for PortfolioCarbonFootprintEngine (PACK-010 SFDR Article 8).

Tests WACI calculation, carbon footprint, financed emissions, attribution
analysis, sector breakdown, top contributors, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_cf_mod = _import_from_path(
    "portfolio_carbon_footprint",
    str(ENGINES_DIR / "portfolio_carbon_footprint.py"),
)

PortfolioCarbonFootprintEngine = _cf_mod.PortfolioCarbonFootprintEngine
HoldingEmissions = _cf_mod.HoldingEmissions
WACIResult = _cf_mod.WACIResult
CarbonFootprintResult = _cf_mod.CarbonFootprintResult
FinancedEmissionsResult = _cf_mod.FinancedEmissionsResult
TemperatureAlignment = _cf_mod.TemperatureAlignment
SectorBreakdown = _cf_mod.SectorBreakdown
CarbonSummary = _cf_mod.CarbonSummary
CarbonMethodology = _cf_mod.CarbonMethodology
AttributionMethod = _cf_mod.AttributionMethod
ScopeCoverage = _cf_mod.ScopeCoverage
DataQuality = _cf_mod.DataQuality
TemperatureMethodology = _cf_mod.TemperatureMethodology

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_holding(
    holding_id: str = "H1",
    company_name: str = "TestCo",
    scope1: float = 1000.0,
    scope2: float = 500.0,
    scope3: float = 5000.0,
    revenue: float = 50_000_000.0,
    evic: float = 100_000_000.0,
    holding_value: float = 2_000_000.0,
    weight_pct: float = 20.0,
    sector: str = "Industrials",
    country: str = "DE",
    data_quality: DataQuality = DataQuality.ESTIMATED_SECTOR,
) -> HoldingEmissions:
    return HoldingEmissions(
        holding_id=holding_id,
        company_name=company_name,
        scope1=scope1,
        scope2=scope2,
        scope3=scope3,
        revenue=revenue,
        evic=evic,
        holding_value=holding_value,
        weight_pct=weight_pct,
        data_quality=data_quality,
        sector=sector,
        country=country,
    )


def _sample_holdings() -> list:
    """Build a portfolio of 5 holdings for carbon testing."""
    return [
        _make_holding("H1", "GreenTech", 200, 100, 1000, 30e6, 60e6, 3e6, 30.0,
                       "Technology", "DE"),
        _make_holding("H2", "OilMajor", 50000, 10000, 200000, 150e6, 200e6, 2e6, 20.0,
                       "Energy", "FR"),
        _make_holding("H3", "CleanBank", 10, 50, 500, 20e6, 40e6, 2e6, 20.0,
                       "Financials", "NL"),
        _make_holding("H4", "SteelCo", 30000, 5000, 80000, 80e6, 120e6, 1.5e6, 15.0,
                       "Materials", "SE"),
        _make_holding("H5", "Pharma", 150, 200, 3000, 40e6, 200e6, 1.5e6, 15.0,
                       "Health Care", "DK"),
    ]


# ===================================================================
# TEST CLASS
# ===================================================================


class TestPortfolioCarbonFootprintEngine:
    """Unit tests for PortfolioCarbonFootprintEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_default_initialization(self):
        """Test engine initializes with default config."""
        engine = PortfolioCarbonFootprintEngine()
        assert engine is not None

    def test_engine_custom_config(self):
        """Test engine initializes with custom config."""
        config = {"currency": "EUR"}
        engine = PortfolioCarbonFootprintEngine(config)
        assert engine is not None

    # ---------------------------------------------------------------
    # 2. calculate_waci
    # ---------------------------------------------------------------

    def test_calculate_waci_returns_result(self):
        """Test WACI calculation returns WACIResult."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1_2)
        assert isinstance(result, WACIResult)

    def test_waci_positive_value(self):
        """Test WACI value is positive for non-zero emissions portfolio."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1_2)
        assert result.waci_value > 0

    def test_waci_scope_1_2_3_higher_than_1_2(self):
        """Test WACI with Scope 1+2+3 is higher than Scope 1+2 only."""
        engine = PortfolioCarbonFootprintEngine()
        r12 = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1_2)
        r123 = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1_2_3)
        assert r123.waci_value >= r12.waci_value

    def test_waci_provenance_hash(self):
        """Test WACI result has valid provenance hash."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1_2)
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    # ---------------------------------------------------------------
    # 3. calculate_carbon_footprint
    # ---------------------------------------------------------------

    def test_carbon_footprint_returns_result(self):
        """Test carbon footprint calculation returns CarbonFootprintResult."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.calculate_carbon_footprint(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2,
        )
        assert isinstance(result, CarbonFootprintResult)

    def test_carbon_footprint_positive(self):
        """Test carbon footprint value is positive."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.calculate_carbon_footprint(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2,
        )
        assert result.carbon_footprint > 0

    # ---------------------------------------------------------------
    # 4. calculate_financed_emissions
    # ---------------------------------------------------------------

    def test_financed_emissions_returns_result(self):
        """Test financed emissions calculation returns FinancedEmissionsResult."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.calculate_financed_emissions(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2,
        )
        assert isinstance(result, FinancedEmissionsResult)

    # ---------------------------------------------------------------
    # 5. attribution_analysis
    # ---------------------------------------------------------------

    def test_attribution_analysis_returns_temperature_alignment(self):
        """Test attribution analysis returns TemperatureAlignment."""
        engine = PortfolioCarbonFootprintEngine()
        result = engine.attribution_analysis(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2,
        )
        assert isinstance(result, TemperatureAlignment)
        assert hasattr(result, "implied_temperature_rise")
        assert result.implied_temperature_rise > 0

    # ---------------------------------------------------------------
    # 6. get_sector_breakdown
    # ---------------------------------------------------------------

    def test_sector_breakdown_returns_result(self):
        """Test sector breakdown returns sector-level emissions."""
        engine = PortfolioCarbonFootprintEngine()
        breakdown = engine.get_sector_breakdown(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2,
        )
        assert isinstance(breakdown, (list, dict))

    # ---------------------------------------------------------------
    # 7. get_top_contributors
    # ---------------------------------------------------------------

    def test_top_contributors_returns_list(self):
        """Test top contributors returns ranked list."""
        engine = PortfolioCarbonFootprintEngine()
        top = engine.get_top_contributors(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2, count=3,
        )
        assert isinstance(top, list)
        assert len(top) <= 3

    # ---------------------------------------------------------------
    # 8. generate_carbon_summary
    # ---------------------------------------------------------------

    def test_generate_carbon_summary(self):
        """Test carbon summary generation returns CarbonSummary."""
        engine = PortfolioCarbonFootprintEngine()
        summary = engine.generate_carbon_summary(
            _sample_holdings(), ScopeCoverage.SCOPE_1_2,
        )
        assert isinstance(summary, CarbonSummary)

    # ---------------------------------------------------------------
    # 9. ScopeCoverage enum
    # ---------------------------------------------------------------

    def test_scope_coverage_enum(self):
        """Test ScopeCoverage enum has expected values."""
        vals = {s.value for s in ScopeCoverage}
        assert "scope_1" in vals
        assert "scope_1_2" in vals
        assert "scope_1_2_3" in vals

    # ---------------------------------------------------------------
    # 10. CarbonMethodology enum
    # ---------------------------------------------------------------

    def test_carbon_methodology_enum(self):
        """Test CarbonMethodology enum has expected values."""
        vals = {m.value for m in CarbonMethodology}
        assert "pcaf" in vals

    # ---------------------------------------------------------------
    # 11. DataQuality enum
    # ---------------------------------------------------------------

    def test_data_quality_enum(self):
        """Test DataQuality enum has expected values."""
        vals = {q.value for q in DataQuality}
        assert "reported_verified" in vals
        assert "estimated_sector" in vals
        assert len(vals) >= 2

    # ---------------------------------------------------------------
    # 12. Deterministic WACI
    # ---------------------------------------------------------------

    def test_waci_deterministic(self):
        """Test same inputs produce identical WACI values."""
        engine = PortfolioCarbonFootprintEngine()
        holdings = _sample_holdings()
        r1 = engine.calculate_waci(holdings, ScopeCoverage.SCOPE_1_2)
        r2 = engine.calculate_waci(holdings, ScopeCoverage.SCOPE_1_2)
        # WACI value must be identical for same inputs
        assert r1.waci_value == r2.waci_value
        # Note: provenance_hash includes timestamp so may differ between calls

    # ---------------------------------------------------------------
    # 13. AttributionMethod enum
    # ---------------------------------------------------------------

    def test_attribution_method_enum(self):
        """Test AttributionMethod enum has expected values."""
        vals = {a.value for a in AttributionMethod}
        assert "evic" in vals

    # ---------------------------------------------------------------
    # 14. HoldingEmissions model
    # ---------------------------------------------------------------

    def test_holding_emissions_model(self):
        """Test HoldingEmissions model construction."""
        h = _make_holding()
        assert h.holding_id == "H1"
        assert h.scope1 == 1000.0
        assert h.scope2 == 500.0
        assert h.revenue == 50_000_000.0

    # ---------------------------------------------------------------
    # 15. Scope 1 only coverage
    # ---------------------------------------------------------------

    def test_scope_1_only_waci(self):
        """Test WACI with Scope 1 only is lower than Scope 1+2."""
        engine = PortfolioCarbonFootprintEngine()
        r1 = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1)
        r12 = engine.calculate_waci(_sample_holdings(), ScopeCoverage.SCOPE_1_2)
        assert r12.waci_value >= r1.waci_value
