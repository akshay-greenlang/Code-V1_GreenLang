# -*- coding: utf-8 -*-
"""
End-to-end integration tests for the Climate Hazard Connector service.

Tests complete real-world workflow scenarios through the
ClimateHazardService facade:
- Multi-hazard assessment workflows
- Portfolio exposure assessment across multiple assets
- Compliance report generation for each framework (TCFD, CSRD, EU Taxonomy)
- TCFD-specific disclosure scenarios
- CSRD/ESRS-specific reporting scenarios
- Location comparison and ranking workflows
- Cross-cutting provenance and audit trail validation

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
"""

import uuid
from typing import Any, Dict, List

import pytest


# ===================================================================
# End-to-end workflow scenarios
# ===================================================================


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow scenarios."""

    def test_e2e_climate_risk_assessment_for_single_facility(
        self,
        service,
        sample_source_noaa,
        sample_hazard_data_flood,
        sample_asset_factory,
    ):
        """End-to-end climate risk assessment for a single facility."""
        # Register data source
        source = service.register_source(**sample_source_noaa)
        source_id = source["source_id"]

        # Ingest hazard data
        flood_data = dict(sample_hazard_data_flood)
        flood_data["source_id"] = source_id
        record = service.ingest_hazard_data(**flood_data)

        # Register asset
        asset = service.register_asset(**sample_asset_factory)
        asset_id = asset["asset_id"]

        # Calculate risk
        risk = service.calculate_risk_index(
            location_id="loc_munich_de",
            hazard_type="flood",
            scenario="SSP2-4.5",
            probability=55.0,
            intensity=40.0,
            frequency=30.0,
            duration=25.0,
        )

        # Project scenario
        projection = service.project_scenario(
            location_id="loc_munich_de",
            hazard_type="flood",
            scenario="SSP2-4.5",
            time_horizon="MID_TERM",
            baseline_value=45.0,
        )

        # Assess exposure
        exposure = service.assess_exposure(
            asset_id=asset_id,
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        # Score vulnerability
        vuln = service.score_vulnerability(
            entity_id=asset_id,
            hazard_type="flood",
            sector="manufacturing",
            exposure_score=55.0,
            sensitivity_score=45.0,
            adaptive_capacity_score=60.0,
        )

        # Generate TCFD report
        report = service.generate_report(
            report_type="tcfd",
            format="json",
        )

        # Validate end-to-end results
        assert source["status"] == "active"
        assert record["hazard_type"] == "flood"
        assert asset["name"] == "Munich Manufacturing Plant"
        assert "index_id" in risk
        assert "projection_id" in projection
        assert "exposure_id" in exposure
        assert "vulnerability_id" in vuln
        assert report["report_type"] == "tcfd"

    def test_e2e_multiple_sources_same_location(
        self, service, sample_source_noaa, sample_source_copernicus
    ):
        """Register multiple data sources covering the same location."""
        source1 = service.register_source(**sample_source_noaa)
        source2 = service.register_source(**sample_source_copernicus)

        # Ingest data from both sources for the same location
        service.ingest_hazard_data(
            source_id=source1["source_id"],
            hazard_type="flood",
            location_id="loc_london_uk",
            value=70.0,
            unit="mm/day",
        )
        service.ingest_hazard_data(
            source_id=source2["source_id"],
            hazard_type="heat_wave",
            location_id="loc_london_uk",
            value=38.5,
            unit="degrees_celsius",
        )

        # Query data for the location
        records = service.query_hazard_data(location_id="loc_london_uk")
        assert len(records) == 2

        # Verify sources are listed
        sources = service.list_sources()
        assert len(sources) == 2

    def test_e2e_iterative_risk_refinement(self, service):
        """Calculate risk multiple times with different parameters to refine."""
        location = "loc_coastal_city"

        # Initial conservative estimate
        risk_v1 = service.calculate_risk_index(
            location_id=location,
            hazard_type="sea_level_rise",
            scenario="SSP2-4.5",
            probability=40.0,
            intensity=30.0,
            frequency=20.0,
            duration=50.0,
        )

        # Revised estimate with updated data
        risk_v2 = service.calculate_risk_index(
            location_id=location,
            hazard_type="sea_level_rise",
            scenario="SSP5-8.5",
            probability=70.0,
            intensity=65.0,
            frequency=45.0,
            duration=80.0,
        )

        # Both should have valid provenance
        assert risk_v1["provenance_hash"] != ""
        assert risk_v2["provenance_hash"] != ""
        # Different inputs should produce different hashes
        assert risk_v1["provenance_hash"] != risk_v2["provenance_hash"]

    def test_e2e_source_to_query_roundtrip(
        self, service, sample_source_noaa
    ):
        """Register source, ingest data, query back, verify data integrity."""
        source = service.register_source(**sample_source_noaa)
        source_id = source["source_id"]

        # Get source by ID
        retrieved = service.get_source(source_id)
        assert retrieved is not None
        assert retrieved["name"] == "NOAA NCEI Climate Data"

        # Ingest and query
        service.ingest_hazard_data(
            source_id=source_id,
            hazard_type="storm",
            location_id="loc_houston_us",
            value=130.0,
            unit="knots",
        )

        records = service.query_hazard_data(
            hazard_type="storm",
            source_id=source_id,
        )
        assert len(records) == 1
        assert records[0]["value"] == 130.0


# ===================================================================
# Multi-hazard assessment
# ===================================================================


class TestMultiHazardAssessment:
    """Test multi-hazard composite risk assessment workflows."""

    def test_multi_hazard_risk_calculation(self, service):
        """Calculate multi-hazard composite risk index for a location."""
        result = service.calculate_multi_hazard(
            location_id="loc_tokyo_jp",
            hazard_types=["flood", "earthquake", "tropical_cyclone"],
            scenario="SSP2-4.5",
        )

        assert "assessment_id" in result
        assert result["location_id"] == "loc_tokyo_jp"
        assert result["hazard_types"] == ["flood", "earthquake", "tropical_cyclone"]
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_multi_hazard_with_all_types(self, service, all_hazard_types):
        """Multi-hazard assessment with all 13 supported hazard types."""
        result = service.calculate_multi_hazard(
            location_id="loc_comprehensive",
            hazard_types=all_hazard_types,
            scenario="SSP3-7.0",
        )

        assert len(result["hazard_types"]) == 13
        assert result["provenance_hash"] != ""

    def test_multi_hazard_different_scenarios(self, service):
        """Multi-hazard assessment under different SSP scenarios."""
        hazard_types = ["flood", "heat_wave"]

        result_ssp2 = service.calculate_multi_hazard(
            location_id="loc_berlin_de",
            hazard_types=hazard_types,
            scenario="SSP2-4.5",
        )

        result_ssp5 = service.calculate_multi_hazard(
            location_id="loc_berlin_de",
            hazard_types=hazard_types,
            scenario="SSP5-8.5",
        )

        # Both should produce valid results
        assert result_ssp2["provenance_hash"] != ""
        assert result_ssp5["provenance_hash"] != ""
        # Different scenarios produce different assessment IDs
        assert result_ssp2["assessment_id"] != result_ssp5["assessment_id"]

    def test_multi_hazard_single_type_degenerates(self, service):
        """Multi-hazard with a single hazard type still produces valid results."""
        result = service.calculate_multi_hazard(
            location_id="loc_oslo_no",
            hazard_types=["cold_wave"],
            scenario="SSP1-2.6",
        )

        assert result["hazard_types"] == ["cold_wave"]
        assert "assessment_id" in result

    def test_multi_hazard_compound_hazard(self, service):
        """Multi-hazard assessment including compound hazard type."""
        result = service.calculate_multi_hazard(
            location_id="loc_jakarta_id",
            hazard_types=["flood", "sea_level_rise", "compound"],
            scenario="SSP5-8.5",
        )

        assert "compound" in result["hazard_types"]
        assert result["provenance_hash"] != ""


# ===================================================================
# Portfolio exposure assessment
# ===================================================================


class TestPortfolioExposureAssessment:
    """Test portfolio-level exposure assessment workflows."""

    def test_portfolio_exposure_single_asset(
        self, service, sample_asset_factory
    ):
        """Portfolio exposure assessment with a single registered asset."""
        asset = service.register_asset(**sample_asset_factory)
        asset_id = asset["asset_id"]

        result = service.assess_portfolio_exposure(
            asset_ids=[asset_id],
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        assert "portfolio_id" in result
        assert result["asset_count"] == 1
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_portfolio_exposure_multiple_assets(
        self,
        service,
        sample_asset_factory,
        sample_asset_warehouse,
        sample_asset_office,
    ):
        """Portfolio exposure with multiple diverse assets."""
        a1 = service.register_asset(**sample_asset_factory)
        a2 = service.register_asset(**sample_asset_warehouse)
        a3 = service.register_asset(**sample_asset_office)

        result = service.assess_portfolio_exposure(
            asset_ids=[a1["asset_id"], a2["asset_id"], a3["asset_id"]],
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        assert result["asset_count"] == 3
        assert result["provenance_hash"] != ""

    def test_portfolio_exposure_different_hazards(
        self, service, sample_asset_factory
    ):
        """Assess portfolio exposure for different hazard types."""
        asset = service.register_asset(**sample_asset_factory)
        asset_id = asset["asset_id"]

        exposure_flood = service.assess_portfolio_exposure(
            asset_ids=[asset_id],
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        exposure_heat = service.assess_portfolio_exposure(
            asset_ids=[asset_id],
            hazard_type="heat_wave",
            scenario="SSP2-4.5",
        )

        assert exposure_flood["portfolio_id"] != exposure_heat["portfolio_id"]
        assert exposure_flood["provenance_hash"] != exposure_heat["provenance_hash"]

    def test_portfolio_exposure_empty_asset_list(self, service):
        """Portfolio exposure with empty asset_ids returns valid response."""
        result = service.assess_portfolio_exposure(
            asset_ids=[],
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        assert result["asset_count"] == 0
        assert "portfolio_id" in result

    def test_portfolio_assets_listed_after_registration(
        self, service, sample_asset_factory, sample_asset_office
    ):
        """Registered assets appear in list_assets results."""
        service.register_asset(**sample_asset_factory)
        service.register_asset(**sample_asset_office)

        assets = service.list_assets()
        assert len(assets) == 2

    def test_portfolio_assets_filtered_by_type(
        self, service, sample_asset_factory, sample_asset_office
    ):
        """list_assets filters correctly by asset_type."""
        service.register_asset(**sample_asset_factory)
        service.register_asset(**sample_asset_office)

        factories = service.list_assets(asset_type="factory")
        assert len(factories) == 1
        assert factories[0]["asset_type"] == "factory"

        offices = service.list_assets(asset_type="office")
        assert len(offices) == 1
        assert offices[0]["asset_type"] == "office"


# ===================================================================
# Compliance report generation
# ===================================================================


class TestComplianceReportGeneration:
    """Test compliance report generation for each framework."""

    def test_generate_tcfd_report_json(self, service):
        """Generate TCFD compliance report in JSON format."""
        result = service.generate_report(
            report_type="tcfd",
            format="json",
        )

        assert result["report_type"] == "tcfd"
        assert result["format"] == "json"
        assert "report_id" in result
        assert result["provenance_hash"] != ""

    def test_generate_csrd_report_json(self, service):
        """Generate CSRD/ESRS compliance report in JSON format."""
        result = service.generate_report(
            report_type="csrd",
            format="json",
        )

        assert result["report_type"] == "csrd"
        assert result["format"] == "json"
        assert "report_id" in result
        assert result["provenance_hash"] != ""

    def test_generate_eu_taxonomy_report(self, service):
        """Generate EU Taxonomy compliance report."""
        result = service.generate_report(
            report_type="eu_taxonomy",
            format="json",
        )

        assert result["report_type"] == "eu_taxonomy"
        assert "report_id" in result
        assert result["provenance_hash"] != ""

    def test_generate_physical_risk_report(self, service):
        """Generate physical risk report."""
        result = service.generate_report(
            report_type="physical_risk",
            format="json",
        )

        assert result["report_type"] == "physical_risk"

    def test_generate_transition_risk_report(self, service):
        """Generate transition risk report."""
        result = service.generate_report(
            report_type="transition_risk",
            format="json",
        )

        assert result["report_type"] == "transition_risk"

    def test_generate_portfolio_summary_report(self, service):
        """Generate portfolio summary report."""
        result = service.generate_report(
            report_type="portfolio_summary",
            format="json",
        )

        assert result["report_type"] == "portfolio_summary"

    def test_generate_hotspot_analysis_report(self, service):
        """Generate hotspot analysis report."""
        result = service.generate_report(
            report_type="hotspot_analysis",
            format="json",
        )

        assert result["report_type"] == "hotspot_analysis"

    def test_generate_compliance_summary_report(self, service):
        """Generate compliance summary report."""
        result = service.generate_report(
            report_type="compliance_summary",
            format="json",
        )

        assert result["report_type"] == "compliance_summary"

    def test_generate_report_csv_format(self, service):
        """Generate report in CSV format."""
        result = service.generate_report(
            report_type="tcfd",
            format="csv",
        )

        assert result["format"] == "csv"

    def test_generate_report_markdown_format(self, service):
        """Generate report in Markdown format."""
        result = service.generate_report(
            report_type="tcfd",
            format="markdown",
        )

        assert result["format"] == "markdown"

    def test_report_retrieval_by_id(self, service):
        """Generate a report and retrieve it by ID."""
        generated = service.generate_report(
            report_type="tcfd",
            format="json",
        )
        report_id = generated["report_id"]

        retrieved = service.get_report(report_id)
        assert retrieved is not None
        assert retrieved["report_id"] == report_id
        assert retrieved["report_type"] == "tcfd"

    def test_report_not_found_returns_none(self, service):
        """Retrieving a nonexistent report returns None."""
        result = service.get_report("nonexistent-report-id")
        assert result is None

    def test_multiple_reports_independent_ids(self, service):
        """Multiple report generations produce unique report IDs."""
        r1 = service.generate_report(report_type="tcfd", format="json")
        r2 = service.generate_report(report_type="csrd", format="json")
        r3 = service.generate_report(report_type="eu_taxonomy", format="json")

        ids = {r1["report_id"], r2["report_id"], r3["report_id"]}
        assert len(ids) == 3  # All unique

    def test_report_with_parameters(self, service):
        """Generate report with additional parameters."""
        result = service.generate_report(
            report_type="tcfd",
            format="json",
            parameters={
                "include_scenarios": ["SSP2-4.5", "SSP5-8.5"],
                "time_horizons": ["MID_TERM", "LONG_TERM"],
            },
        )

        assert "report_id" in result
        assert result["report_type"] == "tcfd"


# ===================================================================
# TCFD-specific scenarios
# ===================================================================


class TestTCFDScenarios:
    """Test TCFD (Task Force on Climate-related Financial Disclosures) scenarios."""

    def test_tcfd_governance_data_flow(self, service):
        """TCFD Governance: register assets, assess risk, generate report."""
        # Register multiple assets
        assets = []
        for i in range(3):
            asset = service.register_asset(
                name=f"TCFD Governance Asset {i}",
                asset_type="facility",
                location_id=f"loc_tcfd_gov_{i}",
                value=10_000_000.0 * (i + 1),
                currency="USD",
                sector="energy",
            )
            assets.append(asset)

        # Calculate risk for each
        for asset in assets:
            service.calculate_risk_index(
                location_id=asset["location_id"],
                hazard_type="flood",
                scenario="SSP2-4.5",
                probability=50.0 + i * 10,
                intensity=40.0 + i * 5,
            )

        # Generate TCFD report
        report = service.generate_report(
            report_type="tcfd",
            format="json",
            parameters={"pillar": "governance"},
        )

        assert report["report_type"] == "tcfd"
        stats = service.get_statistics()
        assert stats["total_assets"] >= 3

    def test_tcfd_strategy_scenario_analysis(self, service):
        """TCFD Strategy: scenario analysis across SSP pathways."""
        location = "loc_tcfd_strategy"

        for scenario in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
            service.project_scenario(
                location_id=location,
                hazard_type="temperature_change",
                scenario=scenario,
                time_horizon="LONG_TERM",
                baseline_value=1.2,
            )

        scenarios = service.list_scenarios()
        assert len(scenarios) == 3

    def test_tcfd_risk_management_workflow(self, service):
        """TCFD Risk Management: identify, assess, and manage climate risks."""
        # Identify risks via multi-hazard assessment
        multi = service.calculate_multi_hazard(
            location_id="loc_tcfd_risk_mgmt",
            hazard_types=["flood", "heat_wave", "drought"],
            scenario="SSP2-4.5",
        )

        # Assess individual exposures
        for hazard in ["flood", "heat_wave", "drought"]:
            service.calculate_risk_index(
                location_id="loc_tcfd_risk_mgmt",
                hazard_type=hazard,
                scenario="SSP2-4.5",
                probability=60.0,
                intensity=55.0,
            )

        # Score vulnerability
        vuln = service.score_vulnerability(
            entity_id="entity_tcfd_risk_mgmt",
            hazard_type="flood",
            sector="financial_services",
            exposure_score=60.0,
            sensitivity_score=50.0,
            adaptive_capacity_score=70.0,
        )

        assert "vulnerability_id" in vuln
        assert multi["provenance_hash"] != ""

    def test_tcfd_metrics_and_targets(self, service):
        """TCFD Metrics & Targets: track KPIs across multiple facilities."""
        facilities = [
            ("Site A", "loc_site_a", "factory"),
            ("Site B", "loc_site_b", "warehouse"),
            ("Site C", "loc_site_c", "office"),
        ]

        asset_ids = []
        for name, loc, atype in facilities:
            asset = service.register_asset(
                name=name,
                asset_type=atype,
                location_id=loc,
                value=5_000_000.0,
                sector="manufacturing",
            )
            asset_ids.append(asset["asset_id"])

        # Assess exposure for portfolio
        portfolio = service.assess_portfolio_exposure(
            asset_ids=asset_ids,
            hazard_type="heat_wave",
            scenario="SSP2-4.5",
        )

        assert portfolio["asset_count"] == 3

        # Generate metrics report
        report = service.generate_report(
            report_type="tcfd",
            format="json",
            parameters={"pillar": "metrics_and_targets"},
        )
        assert report["report_type"] == "tcfd"


# ===================================================================
# CSRD/ESRS-specific scenarios
# ===================================================================


class TestCSRDScenarios:
    """Test CSRD/ESRS (Corporate Sustainability Reporting Directive) scenarios."""

    def test_csrd_e1_climate_change_disclosure(self, service):
        """CSRD ESRS E1: Climate change disclosure requirements."""
        # Register source for CSRD data
        source = service.register_source(
            name="CSRD E1 Climate Data",
            source_type="custom",
            hazard_types=["temperature_change", "precipitation_change"],
            region="European Union",
        )

        # Ingest climate data
        service.ingest_hazard_data(
            source_id=source["source_id"],
            hazard_type="temperature_change",
            location_id="loc_eu_hq",
            scenario="SSP2-4.5",
            value=2.1,
            unit="degrees_celsius",
        )

        # Generate CSRD report
        report = service.generate_report(
            report_type="csrd",
            format="json",
            parameters={"standard": "ESRS_E1"},
        )

        assert report["report_type"] == "csrd"
        assert "report_id" in report

    def test_csrd_double_materiality_assessment(self, service):
        """CSRD double materiality: financial and impact materiality."""
        # Financial materiality -- exposure assessment
        asset = service.register_asset(
            name="EU HQ",
            asset_type="office",
            location_id="loc_eu_hq",
            value=20_000_000.0,
            currency="EUR",
            sector="financial_services",
        )

        exposure = service.assess_exposure(
            asset_id=asset["asset_id"],
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        # Impact materiality -- vulnerability scoring
        vuln = service.score_vulnerability(
            entity_id=asset["asset_id"],
            hazard_type="flood",
            sector="financial_services",
            exposure_score=55.0,
            sensitivity_score=40.0,
            adaptive_capacity_score=65.0,
        )

        assert "exposure_id" in exposure
        assert "vulnerability_id" in vuln

    def test_csrd_multi_location_eu_operations(self, service):
        """CSRD: assess across multiple EU operation locations."""
        eu_locations = [
            ("Berlin Office", "loc_berlin_de", "office"),
            ("Paris Warehouse", "loc_paris_fr", "warehouse"),
            ("Milan Factory", "loc_milan_it", "factory"),
            ("Amsterdam DC", "loc_amsterdam_nl", "data_center"),
        ]

        asset_ids = []
        for name, loc, atype in eu_locations:
            asset = service.register_asset(
                name=name,
                asset_type=atype,
                location_id=loc,
                value=10_000_000.0,
                currency="EUR",
                sector="mixed",
            )
            asset_ids.append(asset["asset_id"])

        portfolio = service.assess_portfolio_exposure(
            asset_ids=asset_ids,
            hazard_type="heat_wave",
            scenario="SSP3-7.0",
        )

        assert portfolio["asset_count"] == 4

        report = service.generate_report(
            report_type="csrd",
            format="json",
            parameters={"scope": "eu_operations"},
        )
        assert report["report_type"] == "csrd"


# ===================================================================
# Location comparison and ranking
# ===================================================================


class TestLocationComparison:
    """Test location comparison and risk ranking workflows."""

    def test_compare_two_locations(self, service):
        """Compare risk indices between two locations."""
        result = service.compare_locations(
            location_ids=["loc_london_uk", "loc_miami_us"],
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        assert "comparison_id" in result
        assert result["location_ids"] == ["loc_london_uk", "loc_miami_us"]
        assert result["hazard_type"] == "flood"
        assert result["provenance_hash"] != ""

    def test_compare_multiple_locations(self, service):
        """Compare risk indices across five locations."""
        locations = [
            "loc_london_uk",
            "loc_miami_us",
            "loc_tokyo_jp",
            "loc_mumbai_in",
            "loc_amsterdam_nl",
        ]

        result = service.compare_locations(
            location_ids=locations,
            hazard_type="sea_level_rise",
            scenario="SSP5-8.5",
        )

        assert len(result["location_ids"]) == 5
        assert result["provenance_hash"] != ""

    def test_compare_locations_different_hazards(self, service):
        """Separate comparisons for different hazard types."""
        locations = ["loc_berlin_de", "loc_madrid_es"]

        comp_flood = service.compare_locations(
            location_ids=locations,
            hazard_type="flood",
            scenario="SSP2-4.5",
        )

        comp_drought = service.compare_locations(
            location_ids=locations,
            hazard_type="drought",
            scenario="SSP2-4.5",
        )

        assert comp_flood["comparison_id"] != comp_drought["comparison_id"]
        assert comp_flood["hazard_type"] == "flood"
        assert comp_drought["hazard_type"] == "drought"


# ===================================================================
# Scenario projection workflows
# ===================================================================


class TestScenarioProjectionWorkflow:
    """Test climate scenario projection workflows."""

    def test_project_all_ssp_scenarios(self, service, ssp_scenarios):
        """Project hazard under all four SSP scenarios."""
        projections = []
        for scenario in ssp_scenarios:
            proj = service.project_scenario(
                location_id="loc_scenario_test",
                hazard_type="temperature_change",
                scenario=scenario,
                time_horizon="LONG_TERM",
                baseline_value=1.0,
            )
            projections.append(proj)

        assert len(projections) == 4
        # Each projection should have a unique ID
        ids = {p["projection_id"] for p in projections}
        assert len(ids) == 4

    def test_project_all_time_horizons(self, service, time_horizons):
        """Project hazard across short, mid, and long-term horizons."""
        projections = []
        for horizon in time_horizons:
            proj = service.project_scenario(
                location_id="loc_horizon_test",
                hazard_type="sea_level_rise",
                scenario="SSP5-8.5",
                time_horizon=horizon,
                baseline_value=0.2,
            )
            projections.append(proj)

        assert len(projections) == 3

    def test_list_scenarios_filtered(self, service):
        """List projections filtered by scenario pathway."""
        service.project_scenario(
            location_id="loc_filter_test",
            hazard_type="flood",
            scenario="SSP2-4.5",
            time_horizon="MID_TERM",
            baseline_value=50.0,
        )
        service.project_scenario(
            location_id="loc_filter_test",
            hazard_type="flood",
            scenario="SSP5-8.5",
            time_horizon="MID_TERM",
            baseline_value=50.0,
        )

        ssp2_results = service.list_scenarios(scenario="SSP2-4.5")
        assert len(ssp2_results) >= 1
        assert all(s["scenario"] == "SSP2-4.5" for s in ssp2_results)

    def test_list_scenarios_filtered_by_horizon(self, service):
        """List projections filtered by time horizon."""
        service.project_scenario(
            location_id="loc_horizon_filter",
            hazard_type="drought",
            scenario="SSP3-7.0",
            time_horizon="SHORT_TERM",
            baseline_value=30.0,
        )
        service.project_scenario(
            location_id="loc_horizon_filter",
            hazard_type="drought",
            scenario="SSP3-7.0",
            time_horizon="LONG_TERM",
            baseline_value=30.0,
        )

        short_term = service.list_scenarios(time_horizon="SHORT_TERM")
        assert len(short_term) >= 1
        assert all(s["time_horizon"] == "SHORT_TERM" for s in short_term)


# ===================================================================
# Source management workflows
# ===================================================================


class TestSourceManagementWorkflow:
    """Test source management lifecycle workflows."""

    def test_register_multiple_source_types(self, service):
        """Register sources of different types."""
        source_configs = [
            {"name": "NOAA Source", "source_type": "noaa", "hazard_types": ["flood"]},
            {"name": "Copernicus Source", "source_type": "copernicus", "hazard_types": ["heat_wave"]},
            {"name": "IPCC Source", "source_type": "ipcc", "hazard_types": ["temperature_change"]},
            {"name": "NASA Source", "source_type": "nasa", "hazard_types": ["wildfire"]},
            {"name": "Custom Source", "source_type": "custom", "hazard_types": ["drought"]},
        ]

        for config in source_configs:
            result = service.register_source(**config)
            assert result["status"] == "active"
            assert result["source_type"] == config["source_type"]

        sources = service.list_sources()
        assert len(sources) == 5

    def test_source_registration_requires_name(self, service):
        """Source registration fails without a name."""
        with pytest.raises(ValueError, match="name must not be empty"):
            service.register_source(
                name="",
                source_type="custom",
                hazard_types=["flood"],
            )

    def test_list_sources_pagination(self, service):
        """Source listing respects limit and offset."""
        for i in range(5):
            service.register_source(
                name=f"Source {i}",
                source_type="custom",
                hazard_types=["flood"],
            )

        page1 = service.list_sources(limit=2, offset=0)
        assert len(page1) == 2

        page2 = service.list_sources(limit=2, offset=2)
        assert len(page2) == 2

        page3 = service.list_sources(limit=2, offset=4)
        assert len(page3) == 1

    def test_get_nonexistent_source_returns_none(self, service):
        """Getting a nonexistent source returns None."""
        result = service.get_source("nonexistent-source-id")
        assert result is None


# ===================================================================
# Vulnerability scoring workflows
# ===================================================================


class TestVulnerabilityScoringWorkflow:
    """Test vulnerability scoring across sectors and hazard types."""

    def test_vulnerability_across_sectors(self, service):
        """Score vulnerability across different economic sectors."""
        sectors = [
            "manufacturing",
            "financial_services",
            "technology",
            "energy",
            "agriculture",
        ]

        results = []
        for sector in sectors:
            result = service.score_vulnerability(
                entity_id=f"entity_{sector}",
                hazard_type="heat_wave",
                sector=sector,
                exposure_score=50.0,
                sensitivity_score=60.0,
                adaptive_capacity_score=45.0,
            )
            results.append(result)
            assert result["sector"] == sector

        assert len(results) == 5

    def test_vulnerability_provenance_unique(self, service):
        """Each vulnerability scoring produces a unique provenance hash."""
        v1 = service.score_vulnerability(
            entity_id="entity_v1",
            hazard_type="flood",
            sector="energy",
            exposure_score=70.0,
            sensitivity_score=60.0,
            adaptive_capacity_score=40.0,
        )

        v2 = service.score_vulnerability(
            entity_id="entity_v2",
            hazard_type="drought",
            sector="agriculture",
            exposure_score=80.0,
            sensitivity_score=75.0,
            adaptive_capacity_score=30.0,
        )

        assert v1["provenance_hash"] != v2["provenance_hash"]


# ===================================================================
# Cross-cutting provenance validation
# ===================================================================


class TestCrossCuttingProvenance:
    """Validate provenance chain integrity across multiple operations."""

    def test_provenance_chain_integrity_after_workflow(
        self, service, sample_source_noaa, sample_asset_factory
    ):
        """Provenance chain stays valid after a complex workflow."""
        # Execute multiple operations
        service.register_source(**sample_source_noaa)
        service.register_asset(**sample_asset_factory)
        service.calculate_risk_index(
            location_id="loc_prov_test",
            hazard_type="flood",
            probability=50.0,
            intensity=50.0,
        )
        service.project_scenario(
            location_id="loc_prov_test",
            hazard_type="flood",
            scenario="SSP2-4.5",
            time_horizon="MID_TERM",
            baseline_value=40.0,
        )
        service.generate_report(report_type="tcfd", format="json")

        # Verify chain is still valid
        assert service.provenance.verify_chain() is True

    def test_provenance_entry_count_increments(self, service):
        """Provenance entry count grows with each operation."""
        initial_count = service.provenance.entry_count

        service.register_source(
            name="Provenance Count Test",
            source_type="custom",
            hazard_types=["flood"],
        )

        assert service.provenance.entry_count > initial_count

    def test_provenance_hash_is_sha256(self, service):
        """All provenance hashes are valid 64-character SHA-256 hex strings."""
        result = service.register_source(
            name="Hash Format Test",
            source_type="custom",
            hazard_types=["flood"],
        )

        prov_hash = result["provenance_hash"]
        assert len(prov_hash) == 64
        # Validate it is valid hex
        int(prov_hash, 16)


# ===================================================================
# Health check end-to-end
# ===================================================================


class TestHealthCheckEndToEnd:
    """Test service health check returns meaningful status."""

    def test_health_check_structure(self, service):
        """Health check response contains all expected fields."""
        health = service.get_health()

        assert "status" in health
        assert "engines" in health
        assert "engines_available" in health
        assert "engines_total" in health
        assert "started" in health
        assert "statistics" in health
        assert "provenance_chain_valid" in health
        assert "provenance_entries" in health
        assert "timestamp" in health

    def test_health_check_engines_enumerated(self, service):
        """Health check enumerates all seven engine statuses."""
        health = service.get_health()

        expected_engines = [
            "hazard_database",
            "risk_index",
            "scenario_projector",
            "exposure_assessor",
            "vulnerability_scorer",
            "compliance_reporter",
            "hazard_pipeline",
        ]

        for engine_name in expected_engines:
            assert engine_name in health["engines"], (
                f"Engine '{engine_name}' missing from health check"
            )
            assert health["engines"][engine_name] in ("available", "unavailable")

    def test_health_check_valid_statuses(self, service):
        """Health check overall status is one of the valid values."""
        health = service.get_health()
        assert health["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_check_provenance_chain_valid(self, service):
        """Health check reports provenance chain as valid for fresh service."""
        health = service.get_health()
        assert health["provenance_chain_valid"] is True
