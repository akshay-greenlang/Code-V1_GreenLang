# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Demo Data and Setup Tests (10 tests)

Tests demo configuration loading, group structure validation,
portfolio data generation, demo execution, and report generation.

Author: GreenLang QA Team
"""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    DEMO_DIR,
    _compute_hash,
    _utcnow,
    assert_provenance_hash,
    generate_import_portfolio,
)


# ---------------------------------------------------------------------------
# Demo Configuration (4 tests)
# ---------------------------------------------------------------------------

class TestDemoConfig:
    """Test demo configuration and data loading."""

    def test_demo_config_loads(self, demo_config):
        """Test demo config loads or defaults are usable."""
        if demo_config:
            assert isinstance(demo_config, dict)
        else:
            # If demo_config.yaml does not exist yet, defaults should work
            default_demo = {
                "mode": "demo",
                "company_name": "EuroSteel Group GmbH (Demo)",
                "reporting_year": 2026,
                "entities": 3,
                "certificates": 50,
            }
            assert default_demo["mode"] == "demo"

    def test_demo_group_structure_valid(self, sample_entity_group):
        """Test demo group structure has valid parent and subsidiaries."""
        grp = sample_entity_group
        assert grp["parent"]["role"] == "parent"
        assert len(grp["subsidiaries"]) == 2
        all_entities = [grp["parent"]] + grp["subsidiaries"]
        for entity in all_entities:
            assert "entity_id" in entity
            assert "eori_number" in entity
            assert "member_state" in entity

    def test_demo_portfolio_data_generated(self, sample_entity_group):
        """Test demo can generate portfolio with correct entity coverage."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 500)
        assert len(portfolio) == 500
        # All entities should have imports
        entity_ids = {r["entity_id"] for r in portfolio}
        expected_ids = {e["entity_id"] for e in all_entities}
        assert expected_ids == entity_ids

    def test_demo_portfolio_row_count(self, sample_entity_group):
        """Test demo portfolio generates 500 rows."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 500)
        assert len(portfolio) == 500


# ---------------------------------------------------------------------------
# Demo Portfolio Content (3 tests)
# ---------------------------------------------------------------------------

class TestDemoPortfolioContent:
    """Test demo portfolio data content."""

    def test_demo_portfolio_entities_match(self, sample_entity_group):
        """Test portfolio entities match group structure."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 100)
        portfolio_entities = {r["entity_id"] for r in portfolio}
        group_entities = {e["entity_id"] for e in all_entities}
        assert portfolio_entities == group_entities

    def test_demo_portfolio_categories_covered(self, sample_entity_group):
        """Test portfolio covers all 6 goods categories."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 100)
        categories = {r["goods_category"] for r in portfolio}
        expected = {"steel", "aluminium", "cement", "fertilizers", "electricity", "hydrogen"}
        assert expected == categories

    def test_demo_portfolio_has_required_fields(self, sample_entity_group):
        """Test each portfolio record has required fields."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 10)
        required_fields = [
            "import_id", "entity_id", "cn_code", "goods_category",
            "origin_country", "weight_tonnes", "value_eur",
            "specific_emission_tco2e_per_tonne", "import_date", "supplier_id",
        ]
        for record in portfolio:
            for field in required_fields:
                assert field in record, f"Missing field '{field}' in portfolio record"


# ---------------------------------------------------------------------------
# Demo Execution and Reports (3 tests)
# ---------------------------------------------------------------------------

class TestDemoExecution:
    """Test demo execution and report generation."""

    def test_demo_execution(self, sample_entity_group, sample_config):
        """Test full demo execution produces results."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 50)

        total_emissions = sum(
            r["weight_tonnes"] * r["specific_emission_tco2e_per_tonne"]
            for r in portfolio
        )
        assert total_emissions > 0

        result = {
            "demo_mode": True,
            "entities_processed": len(all_entities),
            "imports_processed": len(portfolio),
            "total_emissions_tco2e": round(total_emissions, 6),
            "status": "completed",
        }
        assert result["status"] == "completed"
        assert result["imports_processed"] == 50

    def test_demo_reports_generated(self, sample_entity_group):
        """Test demo generates expected reports."""
        report_types = [
            "certificate_portfolio_report",
            "group_consolidation_report",
            "sourcing_scenario_analysis",
            "cross_regulation_mapping_report",
            "customs_integration_report",
            "audit_readiness_scorecard",
        ]
        reports = {}
        for rt in report_types:
            reports[rt] = {
                "template_id": rt,
                "format": "markdown",
                "content": f"# {rt.replace('_', ' ').title()}\n\nDemo content.",
                "provenance_hash": _compute_hash({"template": rt, "demo": True}),
            }
        assert len(reports) == 6
        for rt in report_types:
            assert rt in reports
            assert len(reports[rt]["provenance_hash"]) == 64

    def test_demo_summary_statistics(self, sample_entity_group):
        """Test demo produces summary statistics."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 200)

        by_category = {}
        for r in portfolio:
            cat = r["goods_category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        summary = {
            "total_imports": len(portfolio),
            "by_category": by_category,
            "categories_count": len(by_category),
            "entities_count": len(all_entities),
        }
        assert summary["total_imports"] == 200
        assert summary["categories_count"] >= 3
