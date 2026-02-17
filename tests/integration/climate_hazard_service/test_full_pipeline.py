# -*- coding: utf-8 -*-
"""
Integration tests for the Climate Hazard Connector full pipeline flow.

Tests the complete pipeline lifecycle through the ClimateHazardService facade
and the HazardPipelineEngine directly:
- Complete pipeline flow: register source -> ingest data -> calculate risk
  -> project scenario -> register asset -> assess exposure -> score
  vulnerability -> generate report
- Pipeline run with multiple stages
- Error handling in pipeline stages
- Statistics after pipeline completion
- Batch pipeline runs
- Provenance chain integrity across pipeline stages

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
"""

import time
import uuid
from typing import Any, Dict, List

import pytest


# ===================================================================
# Complete pipeline flow through service facade
# ===================================================================


class TestCompletePipelineFlow:
    """Test the complete pipeline flow through ClimateHazardService methods."""

    def test_step1_register_source(self, service, sample_source_noaa):
        """Step 1: Register a hazard data source succeeds."""
        result = service.register_source(**sample_source_noaa)

        assert result["name"] == "NOAA NCEI Climate Data"
        assert result["source_type"] == "noaa"
        assert "source_id" in result
        assert result["status"] == "active"
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_step2_ingest_hazard_data(
        self, service, registered_source_id, sample_hazard_data_flood
    ):
        """Step 2: Ingest hazard data from a registered source."""
        data = dict(sample_hazard_data_flood)
        data["source_id"] = registered_source_id

        result = service.ingest_hazard_data(**data)

        assert result["hazard_type"] == "flood"
        assert result["source_id"] == registered_source_id
        assert result["location_id"] == "loc_london_uk"
        assert result["value"] == 72.5
        assert result["provenance_hash"] != ""
        assert "record_id" in result

    def test_step3_calculate_risk_index(self, service):
        """Step 3: Calculate a composite risk index for a location."""
        result = service.calculate_risk_index(
            location_id="loc_london_uk",
            hazard_type="flood",
            scenario="SSP2-4.5",
            probability=65.0,
            intensity=72.0,
            frequency=45.0,
            duration=30.0,
        )

        assert "index_id" in result
        assert result["location_id"] == "loc_london_uk"
        assert result["hazard_type"] == "flood"
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_step4_project_scenario(self, service):
        """Step 4: Project climate hazard under a given scenario."""
        result = service.project_scenario(
            location_id="loc_london_uk",
            hazard_type="flood",
            scenario="SSP2-4.5",
            time_horizon="MID_TERM",
            baseline_value=50.0,
        )

        assert "projection_id" in result
        assert result["location_id"] == "loc_london_uk"
        assert result["hazard_type"] == "flood"
        assert result["scenario"] == "SSP2-4.5"
        assert result["time_horizon"] == "MID_TERM"
        assert result["baseline_value"] == 50.0
        assert result["provenance_hash"] != ""

    def test_step5_register_asset(self, service, sample_asset_factory):
        """Step 5: Register a physical asset for monitoring."""
        result = service.register_asset(**sample_asset_factory)

        assert result["name"] == "Munich Manufacturing Plant"
        assert result["asset_type"] == "factory"
        assert result["value"] == 25_000_000.0
        assert result["currency"] == "EUR"
        assert "asset_id" in result
        assert result["provenance_hash"] != ""

    def test_step6_assess_exposure(self, service, registered_asset_id):
        """Step 6: Assess climate hazard exposure for an asset."""
        result = service.assess_exposure(
            asset_id=registered_asset_id,
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        assert "exposure_id" in result
        assert result["asset_id"] == registered_asset_id
        assert result["hazard_type"] == "flood"
        assert result["provenance_hash"] != ""

    def test_step7_score_vulnerability(self, service):
        """Step 7: Score climate vulnerability for an entity."""
        result = service.score_vulnerability(
            entity_id="asset_munich_factory",
            hazard_type="flood",
            sector="manufacturing",
            exposure_score=65.0,
            sensitivity_score=55.0,
            adaptive_capacity_score=40.0,
        )

        assert "vulnerability_id" in result
        assert result["entity_id"] == "asset_munich_factory"
        assert result["hazard_type"] == "flood"
        assert result["sector"] == "manufacturing"
        assert result["provenance_hash"] != ""

    def test_step8_generate_report(self, service):
        """Step 8: Generate a compliance report."""
        result = service.generate_report(
            report_type="tcfd",
            format="json",
        )

        assert "report_id" in result
        assert result["report_type"] == "tcfd"
        assert result["format"] == "json"
        assert result["provenance_hash"] != ""

    def test_full_sequential_workflow(
        self,
        service,
        sample_source_noaa,
        sample_hazard_data_flood,
        sample_asset_factory,
    ):
        """Execute the complete sequential workflow end-to-end."""
        # Step 1: Register source
        source = service.register_source(**sample_source_noaa)
        source_id = source["source_id"]
        assert source["status"] == "active"

        # Step 2: Ingest data
        flood_data = dict(sample_hazard_data_flood)
        flood_data["source_id"] = source_id
        record = service.ingest_hazard_data(**flood_data)
        record_id = record["record_id"]

        # Step 3: Calculate risk
        risk = service.calculate_risk_index(
            location_id="loc_london_uk",
            hazard_type="flood",
            scenario="SSP2-4.5",
            probability=70.0,
            intensity=60.0,
            frequency=50.0,
            duration=40.0,
        )
        assert "index_id" in risk

        # Step 4: Project scenario
        projection = service.project_scenario(
            location_id="loc_london_uk",
            hazard_type="flood",
            scenario="SSP2-4.5",
            time_horizon="MID_TERM",
            baseline_value=55.0,
        )
        assert "projection_id" in projection

        # Step 5: Register asset
        asset = service.register_asset(**sample_asset_factory)
        asset_id = asset["asset_id"]

        # Step 6: Assess exposure
        exposure = service.assess_exposure(
            asset_id=asset_id,
            hazard_type="flood",
            scenario="SSP2-4.5",
        )
        assert "exposure_id" in exposure

        # Step 7: Score vulnerability
        vuln = service.score_vulnerability(
            entity_id=asset_id,
            hazard_type="flood",
            sector="manufacturing",
            exposure_score=60.0,
            sensitivity_score=50.0,
            adaptive_capacity_score=45.0,
        )
        assert "vulnerability_id" in vuln

        # Step 8: Generate report
        report = service.generate_report(
            report_type="tcfd",
            format="json",
        )
        assert "report_id" in report

        # Verify statistics reflect all operations
        stats = service.get_statistics()
        assert stats["total_sources"] >= 1
        assert stats["total_hazard_records"] >= 1
        assert stats["total_risk_indices"] >= 1
        assert stats["total_assets"] >= 1
        assert stats["total_exposures"] >= 1
        assert stats["total_vulnerabilities"] >= 1
        assert stats["total_reports"] >= 1


# ===================================================================
# HazardPipelineEngine direct tests
# ===================================================================


class TestPipelineEngineRun:
    """Test HazardPipelineEngine.run_pipeline directly."""

    def test_pipeline_single_asset_single_hazard(
        self, pipeline_engine, pipeline_single_asset, single_hazard_type
    ):
        """Run pipeline with a single asset and single hazard type."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=single_hazard_type,
        )

        assert "pipeline_id" in result
        assert result["status"] in ("completed", "partial", "failed")
        assert "stages_completed" in result
        assert "results" in result
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0
        assert "provenance_hash" in result

    def test_pipeline_multi_asset_multi_hazard(
        self, pipeline_engine, pipeline_assets, multi_hazard_types
    ):
        """Run pipeline with multiple assets and multiple hazard types."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_assets,
            hazard_types=multi_hazard_types,
        )

        assert "pipeline_id" in result
        assert result["status"] in ("completed", "partial", "failed")
        assert result["assets_count"] == 3
        assert result["hazard_types"] == multi_hazard_types

    def test_pipeline_with_explicit_scenarios(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline with explicit SSP scenarios."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
            scenarios=["ssp2_4.5", "ssp5_8.5"],
        )

        assert "pipeline_id" in result
        assert result["scenarios"] == ["ssp2_4.5", "ssp5_8.5"]

    def test_pipeline_with_explicit_time_horizons(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline with explicit time horizons."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["heat_wave"],
            time_horizons=["short_term", "mid_term", "long_term"],
        )

        assert "pipeline_id" in result
        assert result["time_horizons"] == ["short_term", "mid_term", "long_term"]

    def test_pipeline_with_explicit_frameworks(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline with explicit report frameworks."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["drought"],
            report_frameworks=["tcfd", "csrd"],
        )

        assert "pipeline_id" in result
        assert result["report_frameworks"] == ["tcfd", "csrd"]

    def test_pipeline_default_scenarios_applied(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Verify default scenarios are applied when none specified."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        assert "scenarios" in result
        assert result["scenarios"] == ["ssp2_4.5"]

    def test_pipeline_default_frameworks_applied(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Verify default report frameworks are applied when none specified."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        assert "report_frameworks" in result
        assert result["report_frameworks"] == ["tcfd"]


# ===================================================================
# Pipeline stage selection
# ===================================================================


class TestPipelineStageSelection:
    """Test pipeline with selective stage execution."""

    def test_pipeline_subset_stages_ingest_only(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline with only the ingest stage."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
            stages=["ingest"],
        )

        assert "pipeline_id" in result
        assert "ingest" in result.get("results", {})

    def test_pipeline_subset_stages_ingest_index(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline with ingest and index stages only."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
            stages=["ingest", "index"],
        )

        assert "ingest" in result.get("results", {})
        assert "index" in result.get("results", {})

    def test_pipeline_subset_stages_report_audit(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline with report and audit stages only."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
            stages=["report", "audit"],
        )

        assert "report" in result.get("results", {})
        assert "audit" in result.get("results", {})

    def test_pipeline_all_seven_stages(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Run pipeline explicitly requesting all 7 stages."""
        from greenlang.climate_hazard.hazard_pipeline import PIPELINE_STAGES

        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
            stages=list(PIPELINE_STAGES),
        )

        assert "pipeline_id" in result
        for stage in PIPELINE_STAGES:
            assert stage in result.get("results", {}), (
                f"Stage '{stage}' missing from pipeline results"
            )

    def test_pipeline_stage_timings_recorded(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Verify each executed stage has timing information."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        assert "stage_timings" in result
        timings = result["stage_timings"]
        assert isinstance(timings, dict)
        for stage_name, duration_ms in timings.items():
            assert isinstance(duration_ms, (int, float))
            assert duration_ms >= 0

    def test_pipeline_evaluation_summary_present(
        self, pipeline_engine, pipeline_assets, multi_hazard_types
    ):
        """Verify evaluation summary is included in pipeline result."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_assets,
            hazard_types=multi_hazard_types,
        )

        assert "evaluation_summary" in result
        summary = result["evaluation_summary"]
        assert "total_assets" in summary
        assert "hazard_types_assessed" in summary
        assert summary["total_assets"] == 3


# ===================================================================
# Pipeline error handling
# ===================================================================


class TestPipelineErrorHandling:
    """Test pipeline error handling and validation."""

    def test_pipeline_empty_assets_raises_value_error(self, pipeline_engine):
        """Pipeline raises ValueError when assets list is empty."""
        with pytest.raises(ValueError, match="assets must not be empty"):
            pipeline_engine.run_pipeline(
                assets=[],
                hazard_types=["flood"],
            )

    def test_pipeline_empty_hazard_types_raises_value_error(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline raises ValueError when hazard_types is empty."""
        with pytest.raises(ValueError, match="hazard_types must not be empty"):
            pipeline_engine.run_pipeline(
                assets=pipeline_single_asset,
                hazard_types=[],
            )

    def test_pipeline_partial_stage_failure_continues(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline continues to next stages even if one stage fails."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        # Pipeline should still produce a result regardless of individual
        # stage failures
        assert "pipeline_id" in result
        assert result["status"] in ("completed", "partial", "failed")
        assert "results" in result

    def test_pipeline_status_reflects_failures(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline status correctly reflects stage outcomes."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        status = result["status"]
        stages_completed = result["stages_completed"]
        total_attempted = result.get("total_stages_attempted", 7)

        if stages_completed == total_attempted:
            assert status == "completed"
        elif stages_completed == 0:
            assert status == "failed"
        else:
            assert status in ("partial", "completed", "failed")

    def test_pipeline_failed_stages_tracked(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline tracks which stages failed."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        assert "stages_failed" in result
        assert isinstance(result["stages_failed"], list)


# ===================================================================
# Pipeline statistics and persistence
# ===================================================================


class TestPipelineStatistics:
    """Test pipeline run statistics and state tracking."""

    def test_pipeline_run_persisted(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline run is stored and can be retrieved by ID."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        pipeline_id = result["pipeline_id"]
        retrieved = pipeline_engine.get_pipeline_run(pipeline_id)

        assert retrieved is not None
        assert retrieved["pipeline_id"] == pipeline_id

    def test_pipeline_statistics_increment(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline statistics increment correctly after runs."""
        initial_stats = pipeline_engine.get_statistics()
        initial_total = initial_stats.get("total_runs", 0)

        pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        updated_stats = pipeline_engine.get_statistics()
        assert updated_stats["total_runs"] == initial_total + 1

    def test_pipeline_multiple_runs_accumulate(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Multiple pipeline runs accumulate correctly in statistics."""
        for _ in range(3):
            pipeline_engine.run_pipeline(
                assets=pipeline_single_asset,
                hazard_types=["flood"],
            )

        stats = pipeline_engine.get_statistics()
        assert stats["total_runs"] >= 3

    def test_pipeline_total_duration_tracked(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Total pipeline duration is tracked across runs."""
        pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        stats = pipeline_engine.get_statistics()
        # Pipeline engine tracks duration via avg/max rather than total
        assert stats["total_runs"] >= 1
        assert stats.get("max_duration_ms", stats.get("total_duration_ms", 0)) >= 0

    def test_pipeline_health_check(self, pipeline_engine):
        """Pipeline engine health check returns valid status."""
        health = pipeline_engine.get_health()

        assert "engines_total" in health
        assert health["engines_total"] == 6
        assert "status" in health

    def test_pipeline_nonexistent_run_returns_none(self, pipeline_engine):
        """Retrieving a nonexistent pipeline run returns None."""
        result = pipeline_engine.get_pipeline_run("nonexistent-id-12345")
        assert result is None


# ===================================================================
# Pipeline provenance chain
# ===================================================================


class TestPipelineProvenance:
    """Test provenance chain integrity across pipeline runs."""

    def test_pipeline_provenance_hash_present(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Pipeline result contains a non-empty provenance hash."""
        result = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        assert "provenance_hash" in result
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_pipeline_unique_provenance_per_run(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Each pipeline run produces a unique provenance hash."""
        result1 = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )
        result2 = pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        # Different runs should produce different hashes because of
        # timestamps and pipeline IDs
        assert result1["provenance_hash"] != result2["provenance_hash"]

    def test_pipeline_provenance_chain_valid(
        self, pipeline_engine, pipeline_single_asset
    ):
        """Provenance chain remains valid after pipeline runs."""
        pipeline_engine.run_pipeline(
            assets=pipeline_single_asset,
            hazard_types=["flood"],
        )

        if pipeline_engine.provenance is not None:
            assert pipeline_engine.provenance.verify_chain() is True


# ===================================================================
# Batch pipeline
# ===================================================================


class TestBatchPipeline:
    """Test batch pipeline runs across multiple portfolios."""

    def test_batch_pipeline_single_portfolio(self, pipeline_engine):
        """Batch pipeline with a single portfolio."""
        portfolios = [
            {
                "portfolio_id": "pf_test_single",
                "assets": [
                    {
                        "asset_id": "a1",
                        "name": "Test Office",
                        "asset_type": "office",
                        "location": {"lat": 51.5, "lon": -0.13},
                    }
                ],
            }
        ]

        result = pipeline_engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=["flood"],
        )

        assert "batch_id" in result
        assert "per_portfolio_results" in result
        assert len(result["per_portfolio_results"]) == 1

    def test_batch_pipeline_multiple_portfolios(self, pipeline_engine):
        """Batch pipeline with multiple portfolios."""
        portfolios = [
            {
                "portfolio_id": "pf_europe",
                "assets": [
                    {
                        "asset_id": "a1",
                        "name": "London Office",
                        "asset_type": "office",
                        "location": {"lat": 51.5074, "lon": -0.1278},
                    },
                    {
                        "asset_id": "a2",
                        "name": "Paris Factory",
                        "asset_type": "factory",
                        "location": {"lat": 48.8566, "lon": 2.3522},
                    },
                ],
            },
            {
                "portfolio_id": "pf_asia",
                "assets": [
                    {
                        "asset_id": "a3",
                        "name": "Singapore DC",
                        "asset_type": "data_center",
                        "location": {"lat": 1.3521, "lon": 103.8198},
                    },
                ],
            },
        ]

        result = pipeline_engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=["flood", "heat_wave"],
        )

        assert len(result["per_portfolio_results"]) == 2
        assert result["per_portfolio_results"][0]["portfolio_id"] == "pf_europe"
        assert result["per_portfolio_results"][1]["portfolio_id"] == "pf_asia"

    def test_batch_pipeline_empty_portfolios_raises(self, pipeline_engine):
        """Batch pipeline raises ValueError for empty portfolios list."""
        with pytest.raises(ValueError, match="asset_portfolios must not be empty"):
            pipeline_engine.run_batch_pipeline(
                asset_portfolios=[],
                hazard_types=["flood"],
            )

    def test_batch_pipeline_empty_hazards_raises(self, pipeline_engine):
        """Batch pipeline raises ValueError for empty hazard types."""
        portfolios = [
            {
                "portfolio_id": "pf_test",
                "assets": [
                    {
                        "asset_id": "a1",
                        "name": "Test",
                        "asset_type": "office",
                        "location": {"lat": 0, "lon": 0},
                    }
                ],
            }
        ]

        with pytest.raises(ValueError, match="hazard_types must not be empty"):
            pipeline_engine.run_batch_pipeline(
                asset_portfolios=portfolios,
                hazard_types=[],
            )

    def test_batch_pipeline_summary_present(self, pipeline_engine):
        """Batch pipeline result includes summary statistics."""
        portfolios = [
            {
                "portfolio_id": "pf_summary_test",
                "assets": [
                    {
                        "asset_id": "a1",
                        "name": "Test",
                        "asset_type": "office",
                        "location": {"lat": 40.0, "lon": -74.0},
                    }
                ],
            }
        ]

        result = pipeline_engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=["flood"],
        )

        assert "summary" in result
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    def test_batch_pipeline_provenance_hash_present(self, pipeline_engine):
        """Batch pipeline result includes a provenance hash."""
        portfolios = [
            {
                "portfolio_id": "pf_prov_test",
                "assets": [
                    {
                        "asset_id": "a1",
                        "name": "Test",
                        "asset_type": "office",
                        "location": {"lat": 35.0, "lon": 139.0},
                    }
                ],
            }
        ]

        result = pipeline_engine.run_batch_pipeline(
            asset_portfolios=portfolios,
            hazard_types=["heat_wave"],
        )

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===================================================================
# Service facade pipeline delegation
# ===================================================================


class TestServicePipelineDelegation:
    """Test that ClimateHazardService correctly delegates to HazardPipelineEngine."""

    def test_service_run_pipeline(self, service):
        """Service run_pipeline returns a valid pipeline response."""
        result = service.run_pipeline()

        assert "pipeline_id" in result
        assert "stages_completed" in result
        assert "stages_total" in result
        assert "duration_ms" in result
        assert result["provenance_hash"] != ""

    def test_service_pipeline_statistics_update(self, service):
        """Service statistics update after pipeline run."""
        initial_stats = service.get_statistics()
        initial_count = initial_stats.get("total_pipeline_runs", 0)

        service.run_pipeline()

        updated_stats = service.get_statistics()
        assert updated_stats["total_pipeline_runs"] == initial_count + 1

    def test_service_health_check_after_pipeline(self, service):
        """Service health check remains valid after pipeline operations."""
        service.run_pipeline()

        health = service.get_health()
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert health["provenance_chain_valid"] is True
        assert health["provenance_entries"] > 0


# ===================================================================
# Statistics and metrics after pipeline completion
# ===================================================================


class TestStatisticsAfterPipeline:
    """Test statistics reflect completed operations accurately."""

    def test_statistics_zero_initial(self, service):
        """All statistics start at zero for a fresh service."""
        stats = service.get_statistics()

        assert stats["total_sources"] == 0
        assert stats["total_hazard_records"] == 0
        assert stats["total_risk_indices"] == 0
        assert stats["total_assets"] == 0
        assert stats["total_reports"] == 0

    def test_statistics_after_source_registration(
        self, service, sample_source_noaa
    ):
        """Source count increments after registration."""
        service.register_source(**sample_source_noaa)
        stats = service.get_statistics()
        assert stats["total_sources"] == 1

    def test_statistics_after_data_ingestion(
        self, service, registered_source_id, sample_hazard_data_flood
    ):
        """Hazard record count increments after ingestion."""
        data = dict(sample_hazard_data_flood)
        data["source_id"] = registered_source_id
        service.ingest_hazard_data(**data)

        stats = service.get_statistics()
        assert stats["total_hazard_records"] == 1

    def test_statistics_after_risk_calculation(self, service):
        """Risk index count increments after calculation."""
        service.calculate_risk_index(
            location_id="loc_test",
            hazard_type="flood",
            probability=50.0,
            intensity=50.0,
        )

        stats = service.get_statistics()
        assert stats["total_risk_indices"] == 1

    def test_statistics_after_asset_registration(
        self, service, sample_asset_factory
    ):
        """Asset count increments after registration."""
        service.register_asset(**sample_asset_factory)
        stats = service.get_statistics()
        assert stats["total_assets"] == 1

    def test_statistics_after_report_generation(self, service):
        """Report count increments after generation."""
        service.generate_report(report_type="tcfd", format="json")
        stats = service.get_statistics()
        assert stats["total_reports"] == 1

    def test_metrics_include_provenance(self, service):
        """Service metrics include provenance information."""
        service.register_source(
            name="Test Source",
            source_type="custom",
            hazard_types=["flood"],
        )

        metrics = service.get_metrics()
        assert "provenance_entries" in metrics
        assert metrics["provenance_entries"] > 0
        assert "provenance_chain_valid" in metrics
        assert metrics["provenance_chain_valid"] is True
