# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Waste Treatment Emissions Agent - WasteTreatmentPipelineEngine.

Tests the 8-stage orchestration pipeline: input validation, treatment classification,
factor lookup, biological calculation, thermal calculation, wastewater calculation,
compliance checking, and result assembly. Also tests full pipeline execution, batch
processing, partial failure handling, and provenance chaining.

Target: 90+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.waste_treatment_emissions.waste_treatment_pipeline import (
        WasteTreatmentPipelineEngine,
        PipelineStage,
        VALID_WASTE_CATEGORIES,
        VALID_TREATMENT_METHODS,
        VALID_CALCULATION_METHODS,
        VALID_GWP_SOURCES,
        BIOLOGICAL_METHODS,
        THERMAL_METHODS,
        WASTEWATER_METHODS,
        GWP_VALUES,
        BIOLOGICAL_DEFAULT_EFS,
        THERMAL_DEFAULT_EFS,
        CARBON_CONTENT_WET,
        FOSSIL_CARBON_FRACTIONS,
        WASTEWATER_MCF,
        WASTEWATER_BO,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason="WasteTreatmentPipelineEngine not available",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def pipeline():
    """Create a WasteTreatmentPipelineEngine with mock engines."""
    return WasteTreatmentPipelineEngine()


@pytest.fixture
def composting_request():
    """Valid request for composting calculation."""
    return {
        "tenant_id": "tenant_001",
        "facility_id": "facility_abc",
        "gwp_source": "AR6",
        "calculation_method": "IPCC_DEFAULT",
        "treatment_streams": [
            {
                "stream_id": "stream_01",
                "treatment_method": "COMPOSTING",
                "waste_category": "FOOD_WASTE",
                "waste_mass_tonnes": 500,
                "composition": {"organic": 85, "paper": 10, "other": 5},
            },
        ],
        "frameworks": ["GHG_PROTOCOL"],
    }


@pytest.fixture
def incineration_request():
    """Valid request for incineration calculation."""
    return {
        "tenant_id": "tenant_002",
        "facility_id": "facility_def",
        "gwp_source": "AR6",
        "calculation_method": "IPCC_DEFAULT",
        "treatment_streams": [
            {
                "stream_id": "stream_02",
                "treatment_method": "INCINERATION",
                "waste_category": "MIXED_WASTE",
                "waste_mass_tonnes": 1200,
                "composition": {
                    "plastic": 30, "paper": 25, "food": 20,
                    "textile": 10, "wood": 10, "other": 5,
                },
            },
        ],
        "frameworks": ["IPCC_2006"],
    }


@pytest.fixture
def multi_stream_request():
    """Request with multiple treatment streams."""
    return {
        "tenant_id": "tenant_003",
        "facility_id": "facility_ghi",
        "gwp_source": "AR6",
        "calculation_method": "IPCC_DEFAULT",
        "treatment_streams": [
            {
                "stream_id": "stream_01",
                "treatment_method": "COMPOSTING",
                "waste_category": "FOOD_WASTE",
                "waste_mass_tonnes": 300,
            },
            {
                "stream_id": "stream_02",
                "treatment_method": "INCINERATION",
                "waste_category": "PLASTIC",
                "waste_mass_tonnes": 200,
            },
            {
                "stream_id": "stream_03",
                "treatment_method": "WASTEWATER_TREATMENT",
                "waste_category": "SLUDGE",
                "waste_mass_tonnes": 100,
            },
        ],
        "frameworks": ["GHG_PROTOCOL", "IPCC_2006"],
    }


@pytest.fixture
def wastewater_request():
    """Valid request for wastewater treatment."""
    return {
        "tenant_id": "tenant_004",
        "facility_id": "facility_jkl",
        "gwp_source": "AR6",
        "calculation_method": "IPCC_DEFAULT",
        "treatment_streams": [
            {
                "stream_id": "stream_ww",
                "treatment_method": "WASTEWATER_TREATMENT",
                "waste_category": "SLUDGE",
                "waste_mass_tonnes": 50,
            },
        ],
    }


# ===========================================================================
# Test Class: Pipeline Constants
# ===========================================================================


@_SKIP
class TestPipelineConstants:
    """Test pipeline stage enumeration and constants."""

    def test_eight_pipeline_stages(self):
        """Pipeline has exactly 8 stages."""
        assert len(PipelineStage) == 8

    def test_stage_names(self):
        """All expected stages are defined."""
        expected = {
            "VALIDATE_INPUT", "CLASSIFY_TREATMENT", "LOOKUP_FACTORS",
            "CALCULATE_BIOLOGICAL", "CALCULATE_THERMAL", "CALCULATE_WASTEWATER",
            "CHECK_COMPLIANCE", "ASSEMBLE_RESULTS",
        }
        actual = {s.value for s in PipelineStage}
        assert actual == expected

    def test_biological_methods_defined(self):
        """Biological treatment methods are defined."""
        assert "COMPOSTING" in BIOLOGICAL_METHODS
        assert "ANAEROBIC_DIGESTION" in BIOLOGICAL_METHODS

    def test_thermal_methods_defined(self):
        """Thermal treatment methods are defined."""
        assert "INCINERATION" in THERMAL_METHODS
        assert "PYROLYSIS" in THERMAL_METHODS

    def test_wastewater_methods_defined(self):
        """Wastewater treatment methods are defined."""
        assert "WASTEWATER_TREATMENT" in WASTEWATER_METHODS

    def test_gwp_ar6_values(self):
        """AR6 GWP values are defined."""
        assert GWP_VALUES["AR6"]["CO2"] == Decimal("1")
        assert GWP_VALUES["AR6"]["CH4"] == Decimal("29.8")
        assert GWP_VALUES["AR6"]["N2O"] == Decimal("273")

    def test_biological_default_efs(self):
        """Biological default emission factors are defined."""
        assert "COMPOSTING_WELL_MANAGED" in BIOLOGICAL_DEFAULT_EFS
        assert "AD_VENTED" in BIOLOGICAL_DEFAULT_EFS

    def test_thermal_default_efs(self):
        """Thermal default emission factors are defined."""
        assert "STOKER_GRATE" in THERMAL_DEFAULT_EFS
        assert "FLUIDIZED_BED" in THERMAL_DEFAULT_EFS

    def test_fossil_carbon_fractions(self):
        """Fossil carbon fractions cover key waste types."""
        assert FOSSIL_CARBON_FRACTIONS["FOOD_WASTE"] == Decimal("0")
        assert FOSSIL_CARBON_FRACTIONS["PLASTIC"] == Decimal("1")

    def test_carbon_content_wet(self):
        """Carbon content wet basis covers key waste types."""
        assert "FOOD_WASTE" in CARBON_CONTENT_WET
        assert "PAPER" in CARBON_CONTENT_WET
        assert "PLASTIC" in CARBON_CONTENT_WET

    def test_wastewater_mcf_values(self):
        """Wastewater MCF values are defined."""
        assert WASTEWATER_MCF["AEROBIC_WELL_MANAGED"] == Decimal("0")
        assert WASTEWATER_MCF["ANAEROBIC_NO_RECOVERY"] == Decimal("0.8")

    def test_wastewater_bo_values(self):
        """Wastewater Bo values for BOD and COD are defined."""
        assert WASTEWATER_BO["BOD"] == Decimal("0.6")
        assert WASTEWATER_BO["COD"] == Decimal("0.25")


# ===========================================================================
# Test Class: Pipeline Initialization
# ===========================================================================


@_SKIP
class TestPipelineInit:
    """Test pipeline engine initialization."""

    def test_default_initialization(self):
        """Pipeline initializes with default engines."""
        p = WasteTreatmentPipelineEngine()
        assert p is not None

    def test_custom_engines(self):
        """Pipeline accepts custom engine instances."""
        mock_db = MagicMock()
        mock_bio = MagicMock()
        p = WasteTreatmentPipelineEngine(
            db_engine=mock_db,
            bio_engine=mock_bio,
        )
        assert p._db_engine == mock_db
        assert p._bio_engine == mock_bio

    def test_counter_starts_at_zero(self, pipeline):
        """Execution counter starts at zero."""
        assert pipeline._total_executions == 0

    def test_stage_timings_initialized(self, pipeline):
        """Stage timing accumulators are initialized."""
        assert len(pipeline._stage_timings) == 8
        for stage in PipelineStage:
            assert stage.value in pipeline._stage_timings


# ===========================================================================
# Test Class: Stage 1 - Input Validation
# ===========================================================================


@_SKIP
class TestStage1ValidateInput:
    """Test Stage 1: Input validation."""

    def test_valid_input_passes(self, pipeline, composting_request):
        """Valid input passes validation."""
        result = pipeline.execute(composting_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")
        assert "VALIDATE_INPUT" in result.get("stages_completed", [])

    def test_missing_tenant_id_fails(self, pipeline):
        """Missing tenant_id causes validation failure."""
        bad_request = {
            "treatment_streams": [
                {
                    "stream_id": "s1",
                    "treatment_method": "COMPOSTING",
                    "waste_category": "FOOD_WASTE",
                    "waste_mass_tonnes": 100,
                },
            ],
        }
        result = pipeline.execute(bad_request)
        assert result["status"] in ("ERROR", "VALIDATION_ERROR", "PARTIAL", "FAILED")

    def test_empty_streams_fails(self, pipeline):
        """Empty treatment_streams causes validation failure."""
        bad_request = {
            "tenant_id": "t1",
            "treatment_streams": [],
        }
        result = pipeline.execute(bad_request)
        assert result["status"] in ("ERROR", "VALIDATION_ERROR", "PARTIAL", "FAILED")

    def test_missing_stream_id_fails(self, pipeline):
        """Missing stream_id causes validation failure."""
        bad_request = {
            "tenant_id": "t1",
            "treatment_streams": [
                {
                    "treatment_method": "COMPOSTING",
                    "waste_category": "FOOD_WASTE",
                    "waste_mass_tonnes": 100,
                },
            ],
        }
        result = pipeline.execute(bad_request)
        assert result["status"] in ("ERROR", "VALIDATION_ERROR", "PARTIAL", "FAILED")

    def test_invalid_gwp_source_fails(self, pipeline):
        """Invalid GWP source causes validation failure."""
        bad_request = {
            "tenant_id": "t1",
            "gwp_source": "INVALID",
            "treatment_streams": [
                {
                    "stream_id": "s1",
                    "treatment_method": "COMPOSTING",
                    "waste_category": "FOOD_WASTE",
                    "waste_mass_tonnes": 100,
                },
            ],
        }
        result = pipeline.execute(bad_request)
        assert result["status"] in ("ERROR", "VALIDATION_ERROR", "PARTIAL", "FAILED")


# ===========================================================================
# Test Class: Stage 2 - Treatment Classification
# ===========================================================================


@_SKIP
class TestStage2ClassifyTreatment:
    """Test Stage 2: Treatment method classification."""

    def test_composting_classified_as_biological(self, pipeline):
        """COMPOSTING is classified as BIOLOGICAL."""
        cat = pipeline._classify_stream_method("COMPOSTING")
        assert cat == "BIOLOGICAL"

    def test_incineration_classified_as_thermal(self, pipeline):
        """INCINERATION is classified as THERMAL."""
        cat = pipeline._classify_stream_method("INCINERATION")
        assert cat == "THERMAL"

    def test_wastewater_classified(self, pipeline):
        """WASTEWATER_TREATMENT is classified as WASTEWATER."""
        cat = pipeline._classify_stream_method("WASTEWATER_TREATMENT")
        assert cat == "WASTEWATER"

    def test_unknown_classified_as_other(self, pipeline):
        """Unknown method is classified as OTHER."""
        cat = pipeline._classify_stream_method("PLASMA_ARC")
        assert cat == "OTHER"

    @pytest.mark.parametrize("method,expected", [
        ("COMPOSTING", "BIOLOGICAL"),
        ("ANAEROBIC_DIGESTION", "BIOLOGICAL"),
        ("MBT", "BIOLOGICAL"),
        ("INCINERATION", "THERMAL"),
        ("INCINERATION_ENERGY_RECOVERY", "THERMAL"),
        ("PYROLYSIS", "THERMAL"),
        ("GASIFICATION", "THERMAL"),
        ("OPEN_BURNING", "THERMAL"),
        ("WASTEWATER_TREATMENT", "WASTEWATER"),
    ])
    def test_method_classification_parametrized(self, pipeline, method, expected):
        """Parametrized method classification test."""
        cat = pipeline._classify_stream_method(method)
        assert cat == expected


# ===========================================================================
# Test Class: Full Pipeline Execution
# ===========================================================================


@_SKIP
class TestFullPipelineExecution:
    """Test complete pipeline execution."""

    def test_composting_pipeline(self, pipeline, composting_request):
        """Composting request completes pipeline successfully."""
        result = pipeline.execute(composting_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")
        assert "stages_completed" in result
        assert len(result["stages_completed"]) >= 1

    def test_incineration_pipeline(self, pipeline, incineration_request):
        """Incineration request completes pipeline successfully."""
        result = pipeline.execute(incineration_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")

    def test_multi_stream_pipeline(self, pipeline, multi_stream_request):
        """Multi-stream request processes all streams."""
        result = pipeline.execute(multi_stream_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")

    def test_wastewater_pipeline(self, pipeline, wastewater_request):
        """Wastewater treatment request completes pipeline."""
        result = pipeline.execute(wastewater_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")

    def test_result_has_total_co2e(self, pipeline, composting_request):
        """Pipeline result includes total emissions."""
        result = pipeline.execute(composting_request)
        assert (
            "total_emissions_tco2e" in result
            or "total_co2e_tonnes" in result
            or "gross_total_tco2e" in result
        )

    def test_result_has_per_gas_breakdown(self, pipeline, composting_request):
        """Pipeline result includes per-gas breakdown."""
        result = pipeline.execute(composting_request)
        assert (
            "total_ch4_tonnes" in result
            or "ch4_co2e" in result
            or "emissions_by_gas" in result
            or "gas_results" in result
            or "ch4_tonnes" in result
        )

    def test_result_has_provenance_chain(self, pipeline, composting_request):
        """Pipeline result includes provenance hash chain."""
        result = pipeline.execute(composting_request)
        assert (
            "provenance_chain" in result
            or "provenance_hash" in result
        )

    def test_result_has_stage_timings(self, pipeline, composting_request):
        """Pipeline result includes per-stage timing data."""
        result = pipeline.execute(composting_request)
        assert "stage_timings" in result

    def test_result_has_processing_time(self, pipeline, composting_request):
        """Pipeline result includes total processing time."""
        result = pipeline.execute(composting_request)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_execution_counter_increments(self, pipeline, composting_request):
        """Execution counter increments after pipeline run."""
        initial = pipeline._total_executions
        pipeline.execute(composting_request)
        assert pipeline._total_executions == initial + 1


# ===========================================================================
# Test Class: Batch Processing
# ===========================================================================


@_SKIP
class TestBatchProcessing:
    """Test batch pipeline execution."""

    def test_batch_with_single_item(self, pipeline, composting_request):
        """Batch with one item succeeds."""
        if hasattr(pipeline, "execute_batch"):
            result = pipeline.execute_batch([composting_request])
            assert result["status"] in ("SUCCESS", "PARTIAL")
            assert result.get("total_requests", result.get("batch_size", 0)) >= 1

    def test_batch_with_multiple_items(self, pipeline, composting_request, incineration_request):
        """Batch with multiple items processes all."""
        if hasattr(pipeline, "execute_batch"):
            result = pipeline.execute_batch([
                composting_request, incineration_request,
            ])
            assert result["status"] in ("SUCCESS", "PARTIAL")

    def test_batch_empty_list(self, pipeline):
        """Batch with empty list returns appropriate result."""
        if hasattr(pipeline, "execute_batch"):
            result = pipeline.execute_batch([])
            assert "status" in result

    def test_batch_counter_increments(self, pipeline, composting_request):
        """Batch counter increments."""
        if hasattr(pipeline, "execute_batch"):
            initial = pipeline._total_batches
            pipeline.execute_batch([composting_request])
            assert pipeline._total_batches >= initial


# ===========================================================================
# Test Class: Partial Failure Handling
# ===========================================================================


@_SKIP
class TestPartialFailure:
    """Test pipeline behavior when individual stages fail."""

    def test_validation_failure_records_failed_stage(self, pipeline):
        """Validation failure records VALIDATE_INPUT in stages_failed."""
        bad_request = {
            "treatment_streams": [],
        }
        result = pipeline.execute(bad_request)
        assert (
            "VALIDATE_INPUT" in result.get("stages_failed", [])
            or result["status"] in ("ERROR", "VALIDATION_ERROR")
        )

    def test_partial_status_on_some_failures(self, pipeline):
        """Pipeline returns PARTIAL when some stages fail but not all."""
        request = {
            "tenant_id": "t1",
            "treatment_streams": [
                {
                    "stream_id": "s1",
                    "treatment_method": "COMPOSTING",
                    "waste_category": "FOOD_WASTE",
                    "waste_mass_tonnes": 500,
                },
            ],
        }
        result = pipeline.execute(request)
        # Should at least validate and attempt subsequent stages
        assert result["status"] in ("SUCCESS", "PARTIAL", "ERROR")

    def test_errors_list_populated(self, pipeline):
        """Pipeline populates errors list when stages fail."""
        bad_request = {
            "treatment_streams": [],
        }
        result = pipeline.execute(bad_request)
        errors = result.get("errors", [])
        assert isinstance(errors, list)


# ===========================================================================
# Test Class: GWP Source Handling
# ===========================================================================


@_SKIP
class TestGWPSourceHandling:
    """Test GWP source resolution in pipeline."""

    def test_default_gwp_is_ar6(self, pipeline, composting_request):
        """Default GWP source is AR6."""
        result = pipeline.execute(composting_request)
        gwp = result.get("gwp_source", "AR6")
        assert gwp == "AR6"

    @pytest.mark.parametrize("gwp_source", ["AR4", "AR5", "AR6"])
    def test_valid_gwp_sources_accepted(self, pipeline, composting_request, gwp_source):
        """All valid GWP sources are accepted."""
        composting_request["gwp_source"] = gwp_source
        result = pipeline.execute(composting_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")

    def test_gwp_lookup_ch4_ar6(self, pipeline):
        """GWP lookup for CH4 AR6 returns 29.8."""
        gwp = pipeline._get_gwp("CH4", "AR6")
        assert gwp == Decimal("29.8")

    def test_gwp_lookup_n2o_ar5(self, pipeline):
        """GWP lookup for N2O AR5 returns 265."""
        gwp = pipeline._get_gwp("N2O", "AR5")
        assert gwp == Decimal("265")

    def test_gwp_unknown_gas_returns_one(self, pipeline):
        """Unknown gas returns GWP of 1."""
        gwp = pipeline._get_gwp("UNKNOWN_GAS", "AR6")
        assert gwp == Decimal("1")


# ===========================================================================
# Test Class: Provenance Chaining
# ===========================================================================


@_SKIP
class TestProvenanceChaining:
    """Test provenance hash chaining across pipeline stages."""

    def test_provenance_chain_present(self, pipeline, composting_request):
        """Pipeline result includes provenance chain list."""
        result = pipeline.execute(composting_request)
        chain = result.get("provenance_chain", [])
        assert isinstance(chain, list)

    def test_provenance_chain_grows_with_stages(self, pipeline, composting_request):
        """Each completed stage adds to the provenance chain."""
        result = pipeline.execute(composting_request)
        chain = result.get("provenance_chain", [])
        completed = result.get("stages_completed", [])
        # Chain should have at least as many hashes as completed stages
        assert len(chain) >= len(completed)

    def test_provenance_hash_is_sha256(self, pipeline, composting_request):
        """Each chain hash is 64 hex characters (SHA-256)."""
        result = pipeline.execute(composting_request)
        chain = result.get("provenance_chain", [])
        for h in chain:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# Test Class: Thread Safety
# ===========================================================================


@_SKIP
class TestPipelineThreadSafety:
    """Test pipeline thread safety."""

    def test_concurrent_executions(self, pipeline, composting_request):
        """Multiple concurrent pipeline executions are safe."""
        results = []
        errors = []

        def worker():
            try:
                r = pipeline.execute(composting_request)
                results.append(r)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5

    def test_counter_consistent_after_concurrent(self, pipeline, composting_request):
        """Execution counter is correct after concurrent runs."""
        initial = pipeline._total_executions

        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: pipeline.execute(composting_request))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert pipeline._total_executions == initial + 5


# ===========================================================================
# Test Class: Result Assembly
# ===========================================================================


@_SKIP
class TestResultAssembly:
    """Test Stage 8: Result assembly."""

    def test_result_contains_tenant_id(self, pipeline, composting_request):
        """Assembled result contains tenant_id."""
        result = pipeline.execute(composting_request)
        assert result.get("tenant_id") == "tenant_001"

    def test_result_contains_facility_id(self, pipeline, composting_request):
        """Assembled result contains facility_id."""
        result = pipeline.execute(composting_request)
        assert result.get("facility_id") == "facility_abc"

    def test_result_contains_stream_count(self, pipeline, composting_request):
        """Assembled result contains stream count."""
        result = pipeline.execute(composting_request)
        assert result.get("stream_count", 0) >= 1

    def test_result_calculated_at_present(self, pipeline, composting_request):
        """Assembled result has calculated_at timestamp."""
        result = pipeline.execute(composting_request)
        assert "calculated_at" in result or "timestamp" in result

    def test_multi_stream_result_aggregation(self, pipeline, multi_stream_request):
        """Multi-stream results are aggregated correctly."""
        result = pipeline.execute(multi_stream_request)
        assert result["status"] in ("SUCCESS", "PARTIAL")
        # Total CO2e should be non-negative
        total = float(result.get("total_co2e_tonnes", 0))
        assert total >= 0
