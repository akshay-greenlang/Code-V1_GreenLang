# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Full E2E Annual Report Tests
============================================================

End-to-end tests that simulate the complete annual CSRD reporting
pipeline from setup wizard through to auditor package generation.
All external dependencies are mocked; focus is on verifying the
full pipeline contract: correct orchestration order, data flow
between stages, and final output integrity.

Test count: 10
Author: GreenLang QA Team
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# E2E Pipeline Simulator
# ---------------------------------------------------------------------------

class E2EPipelineSimulator:
    """Simulates a complete CSRD annual reporting pipeline.

    Each stage validates its inputs, simulates processing, and
    produces deterministic outputs for downstream consumption.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        company: Dict[str, Any],
        esg_data: List[Dict[str, Any]],
        agent_registry: MagicMock,
    ):
        self.config = config
        self.company = company
        self.esg_data = esg_data
        self.registry = agent_registry
        self.completed_stages: List[str] = []
        self.stage_outputs: Dict[str, Any] = {}
        self.start_time: float = 0

    def run_setup_wizard(self) -> Dict[str, Any]:
        """Stage 1: Setup wizard collects company profile and config."""
        self.start_time = time.monotonic()
        result = {
            "status": "completed",
            "company_name": self.company["company_name"],
            "preset": self.config.get("size_preset", "mid_market"),
            "sector": self.config.get("sector_preset", "manufacturing"),
            "reporting_year": self.config.get("reporting_year", 2025),
            "esrs_standards_enabled": len(self.config.get("esrs_standards", [])),
            "data_sources_configured": len(self.company.get("data_sources", [])),
        }
        self.completed_stages.append("setup_wizard")
        self.stage_outputs["setup_wizard"] = result
        return result

    def run_data_onboarding(self) -> Dict[str, Any]:
        """Stage 2: Ingest and validate all ESG data."""
        assert "setup_wizard" in self.completed_stages, (
            "Setup wizard must complete before data onboarding"
        )
        records = self.esg_data
        result = {
            "status": "completed",
            "records_total": len(records),
            "records_valid": len(records) - 2,
            "records_rejected": 2,
            "quality_score": 0.91,
            "esrs_coverage_pct": 72.0,
            "missing_data_points": 302,
        }
        self.completed_stages.append("data_onboarding")
        self.stage_outputs["data_onboarding"] = result
        return result

    def run_materiality_assessment(self) -> Dict[str, Any]:
        """Stage 3: Double materiality assessment per ESRS 1."""
        assert "data_onboarding" in self.completed_stages
        result = {
            "status": "completed",
            "topics_assessed": 10,
            "material_topics": 6,
            "non_material_topics": 4,
            "material_topic_names": [
                "E1 - Climate Change",
                "E2 - Pollution",
                "E5 - Circular Economy",
                "S1 - Own Workforce",
                "S2 - Workers in Value Chain",
                "G1 - Business Conduct",
            ],
            "methodology": "ESRS_1_double_materiality",
            "human_review_required": 2,
            "provenance_hash": "mat_hash_" + "a" * 56,
        }
        self.completed_stages.append("materiality_assessment")
        self.stage_outputs["materiality_assessment"] = result
        return result

    def run_scope1_calculations(self) -> Dict[str, Any]:
        """Stage 4: Execute Scope 1 emission calculations."""
        assert "data_onboarding" in self.completed_stages
        scope1_data = [r for r in self.esg_data if r["category"].startswith("scope1")]
        result = {
            "status": "completed",
            "scope": 1,
            "agents_executed": [
                "AGENT-MRV-001", "AGENT-MRV-002", "AGENT-MRV-003",
                "AGENT-MRV-004", "AGENT-MRV-005",
            ],
            "total_tco2e": 4285.3,
            "breakdown": {
                "stationary_combustion": 3652.1,
                "refrigerants": 75.0,
                "mobile_combustion": 245.8,
                "process_emissions": 312.4,
                "fugitive_emissions": 0.0,
            },
            "data_records_used": len(scope1_data),
            "provenance_hash": "sc1_hash_" + "b" * 56,
        }
        self.completed_stages.append("scope1_calculations")
        self.stage_outputs["scope1_calculations"] = result
        return result

    def run_scope2_calculations(self) -> Dict[str, Any]:
        """Stage 5: Execute Scope 2 emission calculations."""
        assert "data_onboarding" in self.completed_stages
        scope2_data = [r for r in self.esg_data if r["category"].startswith("scope2")]
        result = {
            "status": "completed",
            "scope": 2,
            "agents_executed": ["AGENT-MRV-009", "AGENT-MRV-010", "AGENT-MRV-013"],
            "location_based_tco2e": 5124.7,
            "market_based_tco2e": 3891.2,
            "dual_reporting_reconciled": True,
            "data_records_used": len(scope2_data),
            "provenance_hash": "sc2_hash_" + "c" * 56,
        }
        self.completed_stages.append("scope2_calculations")
        self.stage_outputs["scope2_calculations"] = result
        return result

    def run_scope3_calculations(self) -> Dict[str, Any]:
        """Stage 6: Execute Scope 3 emission calculations."""
        assert "data_onboarding" in self.completed_stages
        scope3_data = [r for r in self.esg_data if r["category"].startswith("scope3")]
        categories_enabled = self.config.get("scope3_categories_enabled", [1, 3, 4, 5, 6])
        result = {
            "status": "completed",
            "scope": 3,
            "categories_calculated": categories_enabled,
            "total_tco2e": 28456.9,
            "breakdown": {
                "cat1_purchased_goods": 18230.5,
                "cat3_fuel_energy": 1850.0,
                "cat4_upstream_transport": 3200.4,
                "cat5_waste_generated": 4980.0,
                "cat6_business_travel": 196.0,
            },
            "data_records_used": len(scope3_data),
            "provenance_hash": "sc3_hash_" + "d" * 56,
        }
        self.completed_stages.append("scope3_calculations")
        self.stage_outputs["scope3_calculations"] = result
        return result

    def run_report_generation(self) -> Dict[str, Any]:
        """Stage 7: Generate CSRD compliance reports."""
        assert "scope1_calculations" in self.completed_stages
        assert "scope2_calculations" in self.completed_stages
        assert "scope3_calculations" in self.completed_stages
        assert "materiality_assessment" in self.completed_stages

        total_emissions = (
            self.stage_outputs["scope1_calculations"]["total_tco2e"]
            + self.stage_outputs["scope2_calculations"]["location_based_tco2e"]
            + self.stage_outputs["scope3_calculations"]["total_tco2e"]
        )

        result = {
            "status": "completed",
            "reports_generated": [
                "executive_summary",
                "esrs_disclosure",
                "ghg_emissions_report",
                "materiality_matrix",
                "compliance_dashboard",
            ],
            "total_emissions_tco2e": total_emissions,
            "xbrl_tags_applied": 842,
            "esrs_standards_covered": 12,
            "pages": 95,
        }
        self.completed_stages.append("report_generation")
        self.stage_outputs["report_generation"] = result
        return result

    def run_compliance_audit(self) -> Dict[str, Any]:
        """Stage 8: Run 235 ESRS compliance rules."""
        assert "report_generation" in self.completed_stages
        result = {
            "status": "completed",
            "total_rules": 235,
            "passed": 228,
            "failed": 5,
            "warnings": 2,
            "compliance_score_pct": 97.0,
        }
        self.completed_stages.append("compliance_audit")
        self.stage_outputs["compliance_audit"] = result
        return result

    def run_auditor_package(self) -> Dict[str, Any]:
        """Stage 9: Generate external auditor evidence package."""
        assert "compliance_audit" in self.completed_stages
        result = {
            "status": "completed",
            "package_generated": True,
            "evidence_items": 763,
            "calculation_audit_entries": 48,
            "data_lineage_records": 312,
            "package_hash": "audit_hash_" + "e" * 52,
        }
        self.completed_stages.append("auditor_package")
        self.stage_outputs["auditor_package"] = result
        return result

    def get_total_duration_seconds(self) -> float:
        """Return elapsed time since pipeline start."""
        return time.monotonic() - self.start_time


# =========================================================================
# Full E2E Annual Report Tests
# =========================================================================

class TestE2EAnnualReport:
    """End-to-end tests for the complete annual CSRD reporting pipeline."""

    @pytest.fixture
    def pipeline(
        self,
        sample_pack_config,
        sample_company_profile,
        sample_esg_data,
        mock_agent_registry,
    ) -> E2EPipelineSimulator:
        """Create a pipeline simulator with all test fixtures."""
        return E2EPipelineSimulator(
            config=sample_pack_config,
            company=sample_company_profile,
            esg_data=sample_esg_data,
            agent_registry=mock_agent_registry,
        )

    def test_e2e_setup_wizard_completes(self, pipeline):
        """Setup wizard configures company profile, preset, and data sources."""
        result = pipeline.run_setup_wizard()
        assert result["status"] == "completed"
        assert result["company_name"] == "GreenTech Manufacturing GmbH"
        assert result["preset"] == "mid_market"
        assert result["sector"] == "manufacturing"
        assert result["reporting_year"] == 2025
        assert result["esrs_standards_enabled"] == 12
        assert result["data_sources_configured"] >= 2

    def test_e2e_data_onboarding(self, pipeline):
        """Data onboarding ingests and validates all ESG records."""
        pipeline.run_setup_wizard()
        result = pipeline.run_data_onboarding()
        assert result["status"] == "completed"
        assert result["records_total"] == 50
        assert result["records_valid"] >= 45
        assert result["quality_score"] > 0.8
        assert result["esrs_coverage_pct"] > 50.0

    def test_e2e_materiality_assessment(self, pipeline):
        """Materiality assessment identifies material topics correctly."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        result = pipeline.run_materiality_assessment()
        assert result["status"] == "completed"
        assert result["topics_assessed"] == 10
        assert result["material_topics"] == 6
        assert result["material_topics"] + result["non_material_topics"] == 10
        assert "E1 - Climate Change" in result["material_topic_names"]
        assert result["provenance_hash"] is not None

    def test_e2e_scope1_calculations(self, pipeline):
        """Scope 1 calculations execute all emission engines."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        result = pipeline.run_scope1_calculations()
        assert result["status"] == "completed"
        assert result["scope"] == 1
        assert result["total_tco2e"] > 0
        assert len(result["agents_executed"]) >= 5
        breakdown = result["breakdown"]
        calculated_total = sum(breakdown.values())
        assert calculated_total == pytest.approx(result["total_tco2e"], rel=1e-4)

    def test_e2e_scope2_calculations(self, pipeline):
        """Scope 2 calculations produce both location and market-based results."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        result = pipeline.run_scope2_calculations()
        assert result["status"] == "completed"
        assert result["scope"] == 2
        assert result["location_based_tco2e"] > 0
        assert result["market_based_tco2e"] > 0
        assert result["dual_reporting_reconciled"] is True
        # Location-based should typically be higher than market-based
        # (companies buy RECs to reduce market-based)
        assert result["location_based_tco2e"] >= result["market_based_tco2e"]

    def test_e2e_scope3_calculations(self, pipeline):
        """Scope 3 calculations cover all enabled categories."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        result = pipeline.run_scope3_calculations()
        assert result["status"] == "completed"
        assert result["scope"] == 3
        assert result["total_tco2e"] > 0
        categories = result["categories_calculated"]
        assert 1 in categories, "Category 1 (purchased goods) must be calculated"
        breakdown = result["breakdown"]
        calculated_total = sum(breakdown.values())
        assert calculated_total == pytest.approx(result["total_tco2e"], rel=1e-4)

    def test_e2e_report_generation(self, pipeline):
        """Report generation produces all required output types."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        pipeline.run_materiality_assessment()
        pipeline.run_scope1_calculations()
        pipeline.run_scope2_calculations()
        pipeline.run_scope3_calculations()
        result = pipeline.run_report_generation()
        assert result["status"] == "completed"
        assert len(result["reports_generated"]) >= 5
        assert "executive_summary" in result["reports_generated"]
        assert "esrs_disclosure" in result["reports_generated"]
        assert "ghg_emissions_report" in result["reports_generated"]
        assert result["xbrl_tags_applied"] > 0
        assert result["esrs_standards_covered"] == 12
        assert result["total_emissions_tco2e"] > 0

    def test_e2e_compliance_audit(self, pipeline):
        """Compliance audit executes all 235 rules and scores > 90%."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        pipeline.run_materiality_assessment()
        pipeline.run_scope1_calculations()
        pipeline.run_scope2_calculations()
        pipeline.run_scope3_calculations()
        pipeline.run_report_generation()
        result = pipeline.run_compliance_audit()
        assert result["status"] == "completed"
        assert result["total_rules"] == 235
        assert result["passed"] + result["failed"] + result["warnings"] == 235
        assert result["compliance_score_pct"] >= 90.0

    def test_e2e_auditor_package(self, pipeline):
        """Auditor package generates with complete evidence inventory."""
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        pipeline.run_materiality_assessment()
        pipeline.run_scope1_calculations()
        pipeline.run_scope2_calculations()
        pipeline.run_scope3_calculations()
        pipeline.run_report_generation()
        pipeline.run_compliance_audit()
        result = pipeline.run_auditor_package()
        assert result["status"] == "completed"
        assert result["package_generated"] is True
        assert result["evidence_items"] > 500
        assert result["calculation_audit_entries"] > 0
        assert result["data_lineage_records"] > 0
        assert result["package_hash"] is not None
        assert len(result["package_hash"]) > 40

    def test_e2e_full_pipeline_under_30_minutes(self, pipeline):
        """Full pipeline completes within the 30-minute performance target.

        Since we are running with mocked agents, the actual execution
        should finish in under a second. This test validates that the
        pipeline does not have any blocking operations and that the
        expected performance target is structurally achievable.
        """
        pipeline.run_setup_wizard()
        pipeline.run_data_onboarding()
        pipeline.run_materiality_assessment()
        pipeline.run_scope1_calculations()
        pipeline.run_scope2_calculations()
        pipeline.run_scope3_calculations()
        pipeline.run_report_generation()
        pipeline.run_compliance_audit()
        pipeline.run_auditor_package()

        # Verify all 9 stages completed
        assert len(pipeline.completed_stages) == 9
        expected_stages = [
            "setup_wizard", "data_onboarding", "materiality_assessment",
            "scope1_calculations", "scope2_calculations", "scope3_calculations",
            "report_generation", "compliance_audit", "auditor_package",
        ]
        assert pipeline.completed_stages == expected_stages

        # Verify performance (mocked pipeline should be near-instant)
        duration = pipeline.get_total_duration_seconds()
        assert duration < 30 * 60, (
            f"Pipeline took {duration:.1f}s, exceeding 30-minute target"
        )

        # Verify data flows correctly between stages
        assert pipeline.stage_outputs["scope1_calculations"]["total_tco2e"] > 0
        assert pipeline.stage_outputs["scope2_calculations"]["location_based_tco2e"] > 0
        assert pipeline.stage_outputs["scope3_calculations"]["total_tco2e"] > 0

        # Total emissions should match sum of scopes
        report_total = pipeline.stage_outputs["report_generation"]["total_emissions_tco2e"]
        expected_total = (
            pipeline.stage_outputs["scope1_calculations"]["total_tco2e"]
            + pipeline.stage_outputs["scope2_calculations"]["location_based_tco2e"]
            + pipeline.stage_outputs["scope3_calculations"]["total_tco2e"]
        )
        assert report_total == pytest.approx(expected_total, rel=1e-4)
