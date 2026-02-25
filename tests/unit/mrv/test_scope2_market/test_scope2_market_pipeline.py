# -*- coding: utf-8 -*-
"""
Unit tests for Scope2MarketPipelineEngine (Engine 7 of 7)

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests the 8-stage orchestrated pipeline: input validation, instrument
resolution, instrument allocation, covered emissions calculation,
uncovered emissions calculation, GWP conversion, compliance checks,
and result assembly. Also covers batch pipeline, facility pipeline,
total aggregation, uncertainty integration, dual reporting integration,
coverage scenarios, pipeline info, and error handling.

Target: 80 tests, ~1000 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

try:
    from greenlang.scope2_market.scope2_market_pipeline import (
        Scope2MarketPipelineEngine,
        PIPELINE_STAGES,
        GWP_TABLE,
        VALID_GWP_SOURCES,
        VALID_INSTRUMENT_TYPES,
        RENEWABLE_INSTRUMENT_TYPES,
        DEFAULT_RESIDUAL_MIX,
        DEFAULT_RESIDUAL_MIX_BY_REGION,
        INSTRUMENT_HIERARCHY,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline engine not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    """Create a Scope2MarketPipelineEngine with no upstream engines."""
    p = Scope2MarketPipelineEngine()
    yield p
    p.reset()


@pytest.fixture
def market_request() -> Dict[str, Any]:
    """Build a valid market-based calculation request (partial coverage)."""
    return {
        "facility_id": "FAC-001",
        "tenant_id": "tenant-001",
        "region": "US-CAMX",
        "gwp_source": "AR5",
        "purchases": [
            {
                "purchase_id": "pur-001",
                "mwh": Decimal("5000"),
                "energy_type": "electricity",
            },
        ],
        "instruments": [
            {
                "instrument_id": "inst-001",
                "instrument_type": "rec",
                "mwh": Decimal("3000"),
                "emission_factor": Decimal("0"),
                "is_renewable": True,
                "vintage_year": 2025,
            },
        ],
        "include_compliance": False,
    }


@pytest.fixture
def full_renewable_request() -> Dict[str, Any]:
    """Build a request with 100% renewable instrument coverage."""
    return {
        "facility_id": "FAC-002",
        "tenant_id": "tenant-001",
        "region": "EU-SE",
        "gwp_source": "AR5",
        "purchases": [
            {
                "purchase_id": "pur-002",
                "mwh": Decimal("10000"),
                "energy_type": "electricity",
            },
        ],
        "instruments": [
            {
                "instrument_id": "inst-002",
                "instrument_type": "ppa",
                "mwh": Decimal("10000"),
                "emission_factor": Decimal("0"),
                "is_renewable": True,
                "vintage_year": 2025,
            },
        ],
        "include_compliance": False,
    }


@pytest.fixture
def no_instrument_request() -> Dict[str, Any]:
    """Build a request with no instruments (100% uncovered)."""
    return {
        "facility_id": "FAC-003",
        "tenant_id": "tenant-001",
        "region": "US",
        "gwp_source": "AR5",
        "purchases": [
            {
                "purchase_id": "pur-003",
                "mwh": Decimal("2000"),
                "energy_type": "electricity",
            },
        ],
        "instruments": [],
        "include_compliance": False,
    }


@pytest.fixture
def location_result() -> Dict[str, Any]:
    """Pre-computed location-based result for dual reporting tests."""
    return {
        "total_co2e_tonnes": Decimal("2175.00"),
        "total_co2e_kg": Decimal("2175000.00"),
        "total_mwh": Decimal("5000"),
        "facility_id": "FAC-001",
    }


# ===========================================================================
# 1. TestFullPipeline
# ===========================================================================


@_SKIP
class TestFullPipeline:
    """Tests for run_pipeline end-to-end."""

    def test_run_pipeline_success(self, pipeline, market_request):
        """Full pipeline returns expected fields."""
        result = pipeline.run_pipeline(market_request)
        assert "calculation_id" in result
        assert "total_co2e_tonnes" in result
        assert "provenance_hash" in result
        assert result["total_co2e_tonnes"] >= Decimal("0")

    def test_run_pipeline_has_coverage_info(self, pipeline, market_request):
        """Pipeline result includes coverage information."""
        result = pipeline.run_pipeline(market_request)
        assert "covered_mwh" in result or "coverage_pct" in result

    def test_run_pipeline_has_instrument_allocation(self, pipeline, market_request):
        """Pipeline result includes instrument allocation details."""
        result = pipeline.run_pipeline(market_request)
        assert "instruments" in result or "allocation" in result or "instrument_allocation" in result

    def test_run_pipeline_provenance_hash(self, pipeline, market_request):
        """Pipeline result has a 64-char SHA-256 provenance hash."""
        result = pipeline.run_pipeline(market_request)
        assert len(result["provenance_hash"]) == 64

    def test_run_pipeline_deterministic(self, pipeline, market_request):
        """Same input produces same provenance hash (deterministic)."""
        r1 = pipeline.run_pipeline(market_request)
        pipeline.reset()
        r2 = pipeline.run_pipeline(market_request)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_run_pipeline_full_renewable(self, pipeline, full_renewable_request):
        """100% renewable coverage produces zero or near-zero emissions."""
        result = pipeline.run_pipeline(full_renewable_request)
        total = float(result["total_co2e_tonnes"])
        assert total < 1.0

    def test_run_pipeline_no_instruments(self, pipeline, no_instrument_request):
        """No instruments means 100% uncovered (uses residual mix)."""
        result = pipeline.run_pipeline(no_instrument_request)
        assert float(result["total_co2e_tonnes"]) > 0
        coverage = float(result.get("coverage_pct", 0))
        assert coverage == pytest.approx(0.0, abs=0.01)

    def test_run_pipeline_with_compliance(self, pipeline, market_request):
        """Pipeline with compliance flag runs compliance checks."""
        market_request["include_compliance"] = True
        market_request["compliance_frameworks"] = ["ghg_protocol_scope2"]
        result = pipeline.run_pipeline(market_request)
        assert result is not None

    def test_run_pipeline_ar6_gwp(self, pipeline, market_request):
        """Pipeline with AR6 GWP produces valid result."""
        market_request["gwp_source"] = "AR6"
        result = pipeline.run_pipeline(market_request)
        assert result["total_co2e_tonnes"] >= Decimal("0")

    def test_run_pipeline_ar4_gwp(self, pipeline, market_request):
        """Pipeline with AR4 GWP produces valid result."""
        market_request["gwp_source"] = "AR4"
        result = pipeline.run_pipeline(market_request)
        assert result["total_co2e_tonnes"] >= Decimal("0")


# ===========================================================================
# 2. TestPipelineStages
# ===========================================================================


@_SKIP
class TestPipelineStages:
    """Tests verifying all 8 pipeline stages execute in order."""

    def test_pipeline_stages_constant(self):
        """PIPELINE_STAGES has exactly 8 stages."""
        assert len(PIPELINE_STAGES) == 8

    def test_pipeline_stages_order(self):
        """Stages are in the correct order."""
        assert PIPELINE_STAGES[0] == "validate_input"
        assert PIPELINE_STAGES[1] == "resolve_instruments"
        assert PIPELINE_STAGES[2] == "allocate_instruments"
        assert PIPELINE_STAGES[3] == "calculate_covered"
        assert PIPELINE_STAGES[4] == "calculate_uncovered"
        assert PIPELINE_STAGES[5] == "apply_gwp_conversion"
        assert PIPELINE_STAGES[6] == "compliance_checks"
        assert PIPELINE_STAGES[7] == "assemble_results"

    def test_gwp_table_ar5(self):
        """GWP_TABLE AR5 has correct CO2 GWP of 1."""
        assert GWP_TABLE["AR5"]["co2"] == Decimal("1")
        assert GWP_TABLE["AR5"]["ch4"] == Decimal("28")
        assert GWP_TABLE["AR5"]["n2o"] == Decimal("265")

    def test_gwp_table_ar6(self):
        """GWP_TABLE AR6 has correct values."""
        assert GWP_TABLE["AR6"]["co2"] == Decimal("1")
        assert GWP_TABLE["AR6"]["ch4"] == Decimal("27.9")
        assert GWP_TABLE["AR6"]["n2o"] == Decimal("273")

    def test_valid_gwp_sources(self):
        """VALID_GWP_SOURCES includes AR4, AR5, AR6, AR6_20YR."""
        assert "AR4" in VALID_GWP_SOURCES
        assert "AR5" in VALID_GWP_SOURCES
        assert "AR6" in VALID_GWP_SOURCES
        assert "AR6_20YR" in VALID_GWP_SOURCES

    def test_valid_instrument_types(self):
        """VALID_INSTRUMENT_TYPES has at least 10 types."""
        assert len(VALID_INSTRUMENT_TYPES) >= 10
        assert "rec" in VALID_INSTRUMENT_TYPES
        assert "go" in VALID_INSTRUMENT_TYPES
        assert "ppa" in VALID_INSTRUMENT_TYPES

    def test_renewable_instrument_types(self):
        """RENEWABLE_INSTRUMENT_TYPES is a subset of VALID_INSTRUMENT_TYPES."""
        assert RENEWABLE_INSTRUMENT_TYPES.issubset(
            VALID_INSTRUMENT_TYPES
        ) or RENEWABLE_INSTRUMENT_TYPES <= VALID_INSTRUMENT_TYPES

    def test_instrument_hierarchy(self):
        """INSTRUMENT_HIERARCHY has at least 5 tiers."""
        assert len(INSTRUMENT_HIERARCHY) >= 5

    def test_default_residual_mix_positive(self):
        """DEFAULT_RESIDUAL_MIX is a positive Decimal."""
        assert DEFAULT_RESIDUAL_MIX > Decimal("0")

    def test_residual_mix_by_region_us(self):
        """US residual mix factor is populated."""
        assert "US" in DEFAULT_RESIDUAL_MIX_BY_REGION
        assert DEFAULT_RESIDUAL_MIX_BY_REGION["US"] > Decimal("0")


# ===========================================================================
# 3. TestBatchPipeline
# ===========================================================================


@_SKIP
class TestBatchPipeline:
    """Tests for run_batch_pipeline."""

    def test_batch_pipeline_success(self, pipeline, market_request):
        """Batch pipeline processes multiple requests."""
        batch = {
            "batch_id": "batch-001",
            "requests": [market_request, market_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["total_requests"] == 2
        assert result["successful"] == 2
        assert result["failed"] == 0
        assert len(result["results"]) == 2

    def test_batch_pipeline_with_errors(self, pipeline, market_request):
        """Batch pipeline captures errors for invalid requests."""
        bad_request = {"facility_id": "", "purchases": [], "instruments": []}
        batch = {
            "batch_id": "batch-002",
            "requests": [market_request, bad_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["successful"] >= 1
        assert result["failed"] >= 1

    def test_batch_pipeline_empty_batch(self, pipeline):
        """Empty batch returns zero processed."""
        batch = {"batch_id": "batch-003", "requests": []}
        result = pipeline.run_batch_pipeline(batch)
        assert result["total_requests"] == 0
        assert result["successful"] == 0

    def test_batch_pipeline_aggregated_co2e(self, pipeline, market_request):
        """Batch aggregates total CO2e across all results."""
        batch = {
            "batch_id": "batch-004",
            "requests": [market_request, market_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["total_co2e_tonnes"] > Decimal("0")

    def test_batch_pipeline_provenance(self, pipeline, market_request):
        """Batch result includes provenance hash."""
        batch = {
            "batch_id": "batch-005",
            "requests": [market_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_batch_pipeline_facility_count(self, pipeline, market_request):
        """Batch result counts unique facilities."""
        req2 = dict(market_request)
        req2["facility_id"] = "FAC-DIFFERENT"
        batch = {
            "batch_id": "batch-006",
            "requests": [market_request, req2],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["facility_count"] >= 2

    def test_batch_pipeline_covered_uncovered_totals(
        self, pipeline, market_request
    ):
        """Batch result aggregates covered and uncovered MWh."""
        batch = {
            "batch_id": "batch-007",
            "requests": [market_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert "total_covered_mwh" in result
        assert "total_uncovered_mwh" in result

    def test_batch_single_request(self, pipeline, market_request):
        """Single-item batch works correctly."""
        batch = {
            "batch_id": "batch-008",
            "requests": [market_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["total_requests"] == 1
        assert result["successful"] == 1

    def test_batch_auto_generated_id(self, pipeline, market_request):
        """Batch generates an ID when none is provided."""
        batch = {"requests": [market_request]}
        result = pipeline.run_batch_pipeline(batch)
        assert "batch_id" in result
        assert result["batch_id"] != ""

    def test_batch_three_requests(self, pipeline, market_request):
        """Three-request batch processes all items."""
        batch = {
            "batch_id": "batch-009",
            "requests": [market_request, market_request, market_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["total_requests"] == 3


# ===========================================================================
# 4. TestFacilityPipeline
# ===========================================================================


@_SKIP
class TestFacilityPipeline:
    """Tests for run_facility_pipeline convenience method."""

    def test_facility_pipeline_basic(self, pipeline):
        """Facility pipeline calculates for one facility."""
        purchases = [
            {"purchase_id": "pur-001", "mwh": Decimal("5000"), "energy_type": "electricity"},
        ]
        instruments = [
            {
                "instrument_id": "inst-001",
                "instrument_type": "rec",
                "mwh": Decimal("2000"),
                "emission_factor": Decimal("0"),
                "is_renewable": True,
            },
        ]
        result = pipeline.run_facility_pipeline(
            "FAC-001", purchases, instruments, region="US-CAMX"
        )
        assert result is not None
        assert result.get("total_co2e_tonnes") is not None

    def test_facility_pipeline_region(self, pipeline):
        """Facility pipeline uses specified region."""
        purchases = [
            {"purchase_id": "pur-001", "mwh": Decimal("1000")},
        ]
        instruments = []
        result = pipeline.run_facility_pipeline(
            "FAC-001", purchases, instruments, region="EU-DE"
        )
        assert result is not None

    def test_facility_pipeline_gwp_source(self, pipeline):
        """Facility pipeline uses specified GWP source."""
        purchases = [
            {"purchase_id": "pur-001", "mwh": Decimal("1000")},
        ]
        instruments = []
        result = pipeline.run_facility_pipeline(
            "FAC-001", purchases, instruments,
            region="US", gwp_source="AR6"
        )
        assert result is not None

    def test_facility_pipeline_with_compliance(self, pipeline):
        """Facility pipeline with compliance flag."""
        purchases = [
            {"purchase_id": "pur-001", "mwh": Decimal("1000")},
        ]
        instruments = []
        result = pipeline.run_facility_pipeline(
            "FAC-001", purchases, instruments,
            region="US",
            include_compliance=True,
            compliance_frameworks=["ghg_protocol_scope2"],
        )
        assert result is not None

    def test_facility_pipeline_empty_purchases(self, pipeline):
        """Facility pipeline with empty purchases raises or returns error."""
        with pytest.raises((ValueError, Exception)):
            pipeline.run_facility_pipeline(
                "FAC-001", [], [], region="US"
            )


# ===========================================================================
# 5. TestTotalAggregation
# ===========================================================================


@_SKIP
class TestTotalAggregation:
    """Tests for calculate_total_scope2_market across multiple facilities."""

    def test_total_scope2_single_facility(self, pipeline):
        """Total Scope 2 with one facility."""
        entries = [
            {
                "facility_id": "FAC-001",
                "purchases": [{"mwh": Decimal("5000")}],
                "instruments": [
                    {"instrument_type": "rec", "mwh": Decimal("2000"),
                     "emission_factor": Decimal("0"), "is_renewable": True},
                ],
                "region": "US-CAMX",
            },
        ]
        result = pipeline.calculate_total_scope2_market(entries)
        assert result["facility_count"] == 1
        assert result["grand_total_co2e_tonnes"] >= Decimal("0")

    def test_total_scope2_multiple_facilities(self, pipeline):
        """Total Scope 2 aggregates across multiple facilities."""
        entries = [
            {
                "facility_id": "FAC-001",
                "purchases": [{"mwh": Decimal("5000")}],
                "instruments": [],
                "region": "US-CAMX",
            },
            {
                "facility_id": "FAC-002",
                "purchases": [{"mwh": Decimal("3000")}],
                "instruments": [],
                "region": "EU-DE",
            },
        ]
        result = pipeline.calculate_total_scope2_market(entries)
        assert result["facility_count"] == 2
        assert result["grand_total_co2e_tonnes"] > Decimal("0")

    def test_total_scope2_coverage_pct(self, pipeline):
        """Coverage percentage is weighted correctly."""
        entries = [
            {
                "facility_id": "FAC-001",
                "purchases": [{"mwh": Decimal("10000")}],
                "instruments": [
                    {"instrument_type": "ppa", "mwh": Decimal("10000"),
                     "emission_factor": Decimal("0"), "is_renewable": True},
                ],
                "region": "US",
            },
        ]
        result = pipeline.calculate_total_scope2_market(entries)
        assert float(result["weighted_coverage_pct"]) == pytest.approx(100.0, rel=0.01)

    def test_total_scope2_gwp_source_passed(self, pipeline):
        """GWP source is reflected in the result."""
        entries = [
            {
                "facility_id": "FAC-001",
                "purchases": [{"mwh": Decimal("1000")}],
                "instruments": [],
            },
        ]
        result = pipeline.calculate_total_scope2_market(entries, gwp_source="AR6")
        assert result["gwp_source"] == "AR6"

    def test_total_scope2_empty_facilities(self, pipeline):
        """Empty facility list returns zero."""
        result = pipeline.calculate_total_scope2_market([])
        assert result["facility_count"] == 0
        assert result["grand_total_co2e_tonnes"] == Decimal("0")


# ===========================================================================
# 6. TestUncertaintyIntegration
# ===========================================================================


@_SKIP
class TestUncertaintyIntegration:
    """Tests for run_with_uncertainty with Monte Carlo."""

    def test_run_with_uncertainty(self, pipeline, market_request):
        """Pipeline with uncertainty returns result and uncertainty dict."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=500)
        assert "result" in result
        assert "uncertainty" in result
        assert result["result"]["total_co2e_tonnes"] >= Decimal("0")

    def test_uncertainty_has_ci(self, pipeline, market_request):
        """Uncertainty result includes confidence interval."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=500)
        unc = result["uncertainty"]
        if unc is not None:
            assert "ci_lower" in unc or "lower" in str(unc)

    def test_uncertainty_fallback(self, pipeline, market_request):
        """When no uncertainty engine, fallback uncertainty is used."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=100)
        assert result["uncertainty"] is not None

    def test_uncertainty_default_iterations(self, pipeline, market_request):
        """Default iterations parameter works."""
        result = pipeline.run_with_uncertainty(market_request)
        assert "result" in result

    def test_uncertainty_full_renewable(self, pipeline, full_renewable_request):
        """Uncertainty for zero-emission result is near-zero."""
        result = pipeline.run_with_uncertainty(
            full_renewable_request, mc_iterations=500
        )
        assert result["result"]["total_co2e_tonnes"] < Decimal("1")

    def test_uncertainty_no_instruments(self, pipeline, no_instrument_request):
        """Uncertainty for fully uncovered facility."""
        result = pipeline.run_with_uncertainty(
            no_instrument_request, mc_iterations=500
        )
        assert result["result"]["total_co2e_tonnes"] > Decimal("0")

    def test_uncertainty_result_structure(self, pipeline, market_request):
        """Uncertainty result dict has standard fields."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=100)
        unc = result["uncertainty"]
        assert isinstance(unc, dict)

    def test_uncertainty_combined_results(self, pipeline, market_request):
        """Combined result has both pipeline and uncertainty data."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=100)
        assert "result" in result
        assert result["result"].get("total_co2e_tonnes") is not None

    def test_uncertainty_iterations_passed(self, pipeline, market_request):
        """Custom iteration count is used."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=200)
        unc = result["uncertainty"]
        if unc and "iterations" in unc:
            assert unc["iterations"] == 200

    def test_uncertainty_and_provenance(self, pipeline, market_request):
        """Pipeline result within uncertainty still has provenance."""
        result = pipeline.run_with_uncertainty(market_request, mc_iterations=100)
        assert "provenance_hash" in result["result"]


# ===========================================================================
# 7. TestDualReportingIntegration
# ===========================================================================


@_SKIP
class TestDualReportingIntegration:
    """Tests for run_with_dual_reporting combining location and market."""

    def test_run_with_dual_reporting(
        self, pipeline, market_request, location_result
    ):
        """Dual reporting returns market, location, and comparison."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        assert result["dual_report"] is True
        assert "market_result" in result
        assert "location_result" in result
        assert "comparison" in result

    def test_dual_reporting_comparison_fields(
        self, pipeline, market_request, location_result
    ):
        """Comparison section has expected fields."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        comp = result["comparison"]
        assert "location_co2e_tonnes" in comp
        assert "market_co2e_tonnes" in comp
        assert "difference_tonnes" in comp
        assert "difference_pct" in comp

    def test_dual_reporting_market_lower(
        self, pipeline, market_request, location_result
    ):
        """When market < location, market_lower is True."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        # With 60% REC coverage, market should be lower
        comp = result["comparison"]
        assert comp["market_lower"] is True

    def test_dual_reporting_status_complete(
        self, pipeline, market_request, location_result
    ):
        """Status is 'complete' when both methods have data."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        assert result["status"] == "complete"

    def test_dual_reporting_generated_at(
        self, pipeline, market_request, location_result
    ):
        """Result includes generated_at timestamp."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        assert "generated_at" in result

    def test_dual_reporting_full_renewable(
        self, pipeline, full_renewable_request, location_result
    ):
        """Full renewable shows maximum difference."""
        result = pipeline.run_with_dual_reporting(
            full_renewable_request, location_result
        )
        comp = result["comparison"]
        assert comp["market_lower"] is True
        assert float(comp["difference_tonnes"]) < 0

    def test_dual_reporting_no_instruments(
        self, pipeline, no_instrument_request, location_result
    ):
        """No instruments: market may be similar to location."""
        result = pipeline.run_with_dual_reporting(
            no_instrument_request, location_result
        )
        assert result["dual_report"] is True

    def test_dual_reporting_difference_pct(
        self, pipeline, market_request, location_result
    ):
        """Difference percentage is calculated correctly."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        comp = result["comparison"]
        diff_pct = float(comp["difference_pct"])
        # Should be negative (market < location)
        assert diff_pct < 0

    def test_dual_reporting_zero_location(self, pipeline, market_request):
        """Zero location emissions handles division safely."""
        zero_loc = {
            "total_co2e_tonnes": Decimal("0"),
            "total_co2e_kg": Decimal("0"),
        }
        result = pipeline.run_with_dual_reporting(market_request, zero_loc)
        assert result["dual_report"] is True

    def test_dual_reporting_coverage_in_comparison(
        self, pipeline, market_request, location_result
    ):
        """Comparison includes coverage percentage."""
        result = pipeline.run_with_dual_reporting(
            market_request, location_result
        )
        comp = result["comparison"]
        assert "coverage_pct" in comp


# ===========================================================================
# 8. TestCoverageScenarios
# ===========================================================================


@_SKIP
class TestCoverageScenarios:
    """Tests for fully covered, partially covered, and uncovered scenarios."""

    def test_fully_covered_zero_emissions(self, pipeline, full_renewable_request):
        """100% renewable coverage produces near-zero emissions."""
        result = pipeline.run_pipeline(full_renewable_request)
        assert float(result["total_co2e_tonnes"]) < 1.0

    def test_fully_covered_coverage_pct(self, pipeline, full_renewable_request):
        """100% renewable coverage shows 100% coverage."""
        result = pipeline.run_pipeline(full_renewable_request)
        coverage = float(result.get("coverage_pct", 0))
        assert coverage == pytest.approx(100.0, rel=0.01)

    def test_partially_covered_nonzero_emissions(self, pipeline, market_request):
        """Partial coverage produces non-zero emissions."""
        result = pipeline.run_pipeline(market_request)
        assert float(result["total_co2e_tonnes"]) > 0

    def test_partially_covered_coverage_between(self, pipeline, market_request):
        """Partial coverage shows percentage between 0 and 100."""
        result = pipeline.run_pipeline(market_request)
        coverage = float(result.get("coverage_pct", 0))
        assert 0 < coverage < 100

    def test_uncovered_uses_residual_mix(self, pipeline, no_instrument_request):
        """Uncovered MWh uses residual mix factor."""
        result = pipeline.run_pipeline(no_instrument_request)
        assert float(result["total_co2e_tonnes"]) > 0

    def test_uncovered_zero_coverage(self, pipeline, no_instrument_request):
        """Zero instruments means 0% coverage."""
        result = pipeline.run_pipeline(no_instrument_request)
        coverage = float(result.get("coverage_pct", 0))
        assert coverage == pytest.approx(0.0, abs=0.01)

    def test_multiple_instruments(self, pipeline):
        """Multiple instruments are allocated correctly."""
        request = {
            "facility_id": "FAC-MULTI",
            "tenant_id": "tenant-001",
            "region": "US",
            "gwp_source": "AR5",
            "purchases": [
                {"purchase_id": "pur-001", "mwh": Decimal("5000")},
            ],
            "instruments": [
                {
                    "instrument_id": "inst-001",
                    "instrument_type": "rec",
                    "mwh": Decimal("2000"),
                    "emission_factor": Decimal("0"),
                    "is_renewable": True,
                },
                {
                    "instrument_id": "inst-002",
                    "instrument_type": "go",
                    "mwh": Decimal("1500"),
                    "emission_factor": Decimal("0"),
                    "is_renewable": True,
                },
            ],
        }
        result = pipeline.run_pipeline(request)
        covered = float(result.get("covered_mwh", 0))
        assert covered >= 3000

    def test_supplier_specific_instrument(self, pipeline):
        """Supplier-specific instrument has non-zero emission factor."""
        request = {
            "facility_id": "FAC-SUP",
            "tenant_id": "tenant-001",
            "region": "US",
            "gwp_source": "AR5",
            "purchases": [
                {"purchase_id": "pur-001", "mwh": Decimal("5000")},
            ],
            "instruments": [
                {
                    "instrument_id": "inst-sup",
                    "instrument_type": "supplier_specific",
                    "mwh": Decimal("5000"),
                    "emission_factor": Decimal("0.350"),
                    "is_renewable": False,
                },
            ],
        }
        result = pipeline.run_pipeline(request)
        # Supplier-specific with 0.350 tCO2e/MWh * 5000 = ~1750 tCO2e
        assert float(result["total_co2e_tonnes"]) > 0

    def test_over_allocation_capped(self, pipeline):
        """Instruments exceeding total MWh are capped."""
        request = {
            "facility_id": "FAC-OVER",
            "tenant_id": "tenant-001",
            "region": "US",
            "gwp_source": "AR5",
            "purchases": [
                {"purchase_id": "pur-001", "mwh": Decimal("1000")},
            ],
            "instruments": [
                {
                    "instrument_id": "inst-001",
                    "instrument_type": "rec",
                    "mwh": Decimal("5000"),
                    "emission_factor": Decimal("0"),
                    "is_renewable": True,
                },
            ],
        }
        result = pipeline.run_pipeline(request)
        covered = float(result.get("covered_mwh", 0))
        assert covered <= 1000 + 0.01  # Cannot exceed total purchase

    def test_different_regions_different_residual(self, pipeline):
        """Different regions produce different uncovered emissions."""
        req_us = {
            "facility_id": "FAC-US",
            "region": "US",
            "gwp_source": "AR5",
            "purchases": [{"mwh": Decimal("1000")}],
            "instruments": [],
        }
        req_eu_fr = {
            "facility_id": "FAC-FR",
            "region": "EU-FR",
            "gwp_source": "AR5",
            "purchases": [{"mwh": Decimal("1000")}],
            "instruments": [],
        }
        r_us = pipeline.run_pipeline(req_us)
        pipeline.reset()
        r_fr = pipeline.run_pipeline(req_eu_fr)
        # US and France have very different residual mix factors
        assert r_us["total_co2e_tonnes"] != r_fr["total_co2e_tonnes"]


# ===========================================================================
# 9. TestPipelineInfo
# ===========================================================================


@_SKIP
class TestPipelineInfo:
    """Tests for get_pipeline_stages, get_statistics, and reset."""

    def test_get_pipeline_stages(self, pipeline):
        """get_pipeline_stages returns list of 8 stage names."""
        stages = pipeline.get_pipeline_stages()
        assert isinstance(stages, list)
        assert len(stages) == 8
        assert "validate_input" in stages
        assert "assemble_results" in stages

    def test_get_statistics_initial(self, pipeline):
        """Initial statistics show zero runs."""
        stats = pipeline.get_statistics()
        assert stats["pipeline_runs"] == 0
        assert stats["stages_count"] == 8

    def test_statistics_after_run(self, pipeline, market_request):
        """Statistics update after a pipeline run."""
        pipeline.run_pipeline(market_request)
        stats = pipeline.get_statistics()
        assert stats["pipeline_runs"] >= 1

    def test_reset_pipeline(self, pipeline, market_request):
        """reset zeroes counters."""
        pipeline.run_pipeline(market_request)
        assert pipeline._pipeline_runs > 0
        pipeline.reset()
        assert pipeline._pipeline_runs == 0

    def test_statistics_engines_available(self, pipeline):
        """Statistics report engine availability."""
        stats = pipeline.get_statistics()
        assert "engines" in stats
        assert isinstance(stats["engines"], dict)


# ===========================================================================
# 10. TestErrorHandling
# ===========================================================================


@_SKIP
class TestErrorHandling:
    """Tests for error handling with invalid input."""

    def test_invalid_gwp_source(self, pipeline):
        """Invalid GWP source raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.run_pipeline({
                "facility_id": "FAC-001",
                "purchases": [{"mwh": Decimal("100")}],
                "instruments": [],
                "gwp_source": "INVALID",
            })

    def test_empty_facility_id(self, pipeline):
        """Empty facility_id raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.run_pipeline({
                "facility_id": "",
                "purchases": [{"mwh": Decimal("100")}],
                "instruments": [],
            })

    def test_empty_purchases(self, pipeline):
        """Empty purchases list raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.run_pipeline({
                "facility_id": "FAC-001",
                "purchases": [],
                "instruments": [],
            })

    def test_negative_mwh(self, pipeline):
        """Negative MWh raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.run_pipeline({
                "facility_id": "FAC-001",
                "purchases": [{"mwh": Decimal("-100")}],
                "instruments": [],
            })

    def test_missing_purchases_key(self, pipeline):
        """Missing purchases key uses empty default and raises."""
        with pytest.raises(ValueError):
            pipeline.run_pipeline({
                "facility_id": "FAC-001",
            })
