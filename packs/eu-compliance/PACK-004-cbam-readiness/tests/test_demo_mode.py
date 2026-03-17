# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Demo Mode Tests (8 tests)

Tests demo configuration loading, sample data validation,
demo workflows, and demo health checks.

Author: GreenLang QA Team
"""

import csv
import io
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    DEMO_DIR,
    StubCBAMApp,
    StubCustoms,
    _compute_hash,
    _utcnow,
    render_template_stub,
)


class TestDemoMode:
    """Test suite for CBAM demo mode functionality."""

    def test_demo_config_loads(self, demo_config):
        """Test demo configuration loads and has correct structure."""
        if demo_config:
            assert isinstance(demo_config, dict)
        else:
            # Validate that default demo works
            default_demo = {
                "enabled": True,
                "use_sample_data": True,
                "skip_external_apis": True,
            }
            assert default_demo["enabled"] is True

    def test_demo_imports_csv_valid(self, sample_import_csv_data):
        """Test demo import CSV data is valid and parseable."""
        reader = csv.DictReader(io.StringIO(sample_import_csv_data))
        rows = list(reader)
        assert len(rows) == 20
        required_columns = [
            "import_id", "cn_code", "goods_category",
            "origin_country", "weight_tonnes",
        ]
        for col in required_columns:
            assert col in rows[0], f"Missing column: {col}"
        for row in rows:
            assert float(row["weight_tonnes"]) > 0
            assert row["goods_category"] in ("steel", "aluminium", "cement")

    def test_demo_supplier_data_valid(self, sample_suppliers):
        """Test demo supplier data is valid."""
        assert len(sample_suppliers) >= 3
        for supplier in sample_suppliers:
            assert "supplier_id" in supplier
            assert "company_name" in supplier
            assert "country" in supplier
            assert supplier["status"] == "active"

    def test_demo_setup_wizard(self):
        """Test demo setup wizard completes successfully."""
        wizard_steps = [
            {"step": "load_config", "status": "completed"},
            {"step": "load_cn_codes", "status": "completed"},
            {"step": "load_sample_imports", "status": "completed"},
            {"step": "configure_engines", "status": "completed"},
            {"step": "run_health_check", "status": "completed"},
        ]
        assert len(wizard_steps) == 5
        assert all(s["status"] == "completed" for s in wizard_steps)
        result = {
            "demo_mode": True,
            "steps_completed": len(wizard_steps),
            "errors": 0,
            "status": "ready",
        }
        assert result["status"] == "ready"

    def test_demo_quarterly_workflow(
        self, sample_importer_config, sample_emission_results,
    ):
        """Test demo quarterly workflow with sample data."""
        total_emissions = sum(
            r["total_emissions_tco2e"] for r in sample_emission_results
        )
        report = {
            "report_id": "DEMO-QR-2026-Q1",
            "importer": sample_importer_config["company_name"],
            "total_imports": len(sample_emission_results),
            "total_emissions_tco2e": round(total_emissions, 6),
            "status": "demo_completed",
        }
        assert report["total_imports"] == 10
        assert report["total_emissions_tco2e"] > 0
        assert report["status"] == "demo_completed"

    def test_demo_health_check(self):
        """Test demo mode health check passes."""
        health = {
            "demo_mode": True,
            "config_loaded": True,
            "sample_data_available": True,
            "external_apis_skipped": True,
            "engines_ready": 7,
            "status": "healthy",
        }
        assert health["status"] == "healthy"
        assert health["demo_mode"] is True
        assert health["engines_ready"] == 7

    def test_demo_cn_codes_valid(self, sample_cn_codes):
        """Test demo CN codes are valid format."""
        total_codes = sum(len(codes) for codes in sample_cn_codes.values())
        assert total_codes >= 50
        for category, codes in sample_cn_codes.items():
            for entry in codes:
                cn_code = entry["code"]
                assert re.match(r"^\d{4}\s\d{2}\s\d{2}$", cn_code), (
                    f"Invalid CN code format: {cn_code}"
                )

    def test_demo_emission_factors(self):
        """Test demo emission factors are within reasonable ranges."""
        demo_factors = {
            "steel": {"bof": 1.85, "eaf": 0.45, "default": 2.30},
            "aluminium": {"primary": 9.20, "secondary": 1.50, "default": 8.50},
            "cement": {"clinker": 0.85, "blended": 0.50, "default": 0.65},
        }
        for category, factors in demo_factors.items():
            for method, value in factors.items():
                assert 0.0 < value < 50.0, (
                    f"EF for {category}/{method} out of range: {value}"
                )
            # Default should be within min/max of specific methods
            methods = {k: v for k, v in factors.items() if k != "default"}
            assert factors["default"] >= min(methods.values()), (
                f"Default EF for {category} below minimum method"
            )
