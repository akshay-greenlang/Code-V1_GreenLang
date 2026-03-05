# -*- coding: utf-8 -*-
"""
Unit tests for ISO14064Platform -- Service Facade (setup.py).

Tests platform initialization, health check, platform info, engine
composition, and shared store wiring with 20+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ISO14064AppConfig,
    ISOCategory,
    ReportFormat,
)
from services.setup import ISO14064Platform
from services.boundary_manager import BoundaryManager
from services.quantification_engine import QuantificationEngine
from services.removals_tracker import RemovalsTracker
from services.significance_engine import SignificanceEngine
from services.uncertainty_engine import UncertaintyEngine
from services.base_year_manager import BaseYearManager
from services.report_generator import ReportGenerator
from services.management_plan import ManagementPlanEngine
from services.verification_workflow import VerificationWorkflow
from services.crosswalk_engine import CrosswalkEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def platform():
    """Fresh ISO14064Platform instance."""
    return ISO14064Platform()


@pytest.fixture
def custom_platform():
    """Platform with custom configuration."""
    config = ISO14064AppConfig(
        monte_carlo_iterations=500,
        reporting_year=2025,
    )
    return ISO14064Platform(config)


# ===========================================================================
# Tests
# ===========================================================================


class TestPlatformInitialization:
    """Test platform initialization and engine wiring."""

    def test_platform_creates_successfully(self, platform):
        assert platform is not None
        assert platform.config is not None

    def test_default_config_applied(self, platform):
        assert isinstance(platform.config, ISO14064AppConfig)

    def test_custom_config_applied(self, custom_platform):
        assert custom_platform.config.monte_carlo_iterations == 500
        assert custom_platform.config.reporting_year == 2025

    def test_all_12_engines_wired(self, platform):
        assert isinstance(platform.boundary, BoundaryManager)
        assert isinstance(platform.quantification, QuantificationEngine)
        assert isinstance(platform.removals, RemovalsTracker)
        # aggregator is CategoryAggregator -- just check it exists
        assert platform.aggregator is not None
        assert isinstance(platform.significance, SignificanceEngine)
        assert isinstance(platform.uncertainty, UncertaintyEngine)
        # quality is QualityManagement -- just check it exists
        assert platform.quality is not None
        assert isinstance(platform.base_year, BaseYearManager)
        assert isinstance(platform.reporter, ReportGenerator)
        assert isinstance(platform.management, ManagementPlanEngine)
        assert isinstance(platform.verification, VerificationWorkflow)
        assert isinstance(platform.crosswalk, CrosswalkEngine)


class TestHealthCheck:
    """Test platform health check."""

    def test_health_check_status(self, platform):
        health = platform.health_check()
        assert health["status"] == "healthy"

    def test_health_check_version(self, platform):
        health = platform.health_check()
        assert health["version"] == platform.config.version

    def test_health_check_standard(self, platform):
        health = platform.health_check()
        assert health["standard"] == "ISO 14064-1:2018"

    def test_health_check_engine_count(self, platform):
        health = platform.health_check()
        assert health["engine_count"] == 12

    def test_health_check_all_engines_ok(self, platform):
        health = platform.health_check()
        engines = health["engines"]
        assert len(engines) == 12
        for engine_name, status in engines.items():
            assert status == "ok", f"Engine {engine_name} not ok"

    def test_health_check_category_count(self, platform):
        health = platform.health_check()
        assert health["categories"] == 6

    def test_health_check_mrv_agents(self, platform):
        health = platform.health_check()
        assert health["mrv_agents_integrated"] == 28

    def test_health_check_counts_start_at_zero(self, platform):
        health = platform.health_check()
        assert health["inventory_count"] == 0
        assert health["organization_count"] == 0


class TestPlatformInfo:
    """Test platform metadata."""

    def test_platform_info_name(self, platform):
        info = platform.get_platform_info()
        assert info["name"] == platform.config.app_name

    def test_platform_info_standard(self, platform):
        info = platform.get_platform_info()
        assert info["standard"] == "ISO 14064-1:2018"
        assert info["verification_standard"] == "ISO 14064-3:2019"

    def test_platform_info_engine_count(self, platform):
        info = platform.get_platform_info()
        assert info["engine_count"] == 12

    def test_platform_info_iso_categories(self, platform):
        info = platform.get_platform_info()
        assert info["iso_categories"] == 6

    def test_platform_info_mrv_agents(self, platform):
        info = platform.get_platform_info()
        assert info["mrv_agents_integrated"] == 28

    def test_platform_info_mandatory_elements(self, platform):
        info = platform.get_platform_info()
        assert info["mandatory_reporting_elements"] == 14

    def test_platform_info_supported_formats(self, platform):
        info = platform.get_platform_info()
        formats = info["supported_formats"]
        assert len(formats) == len(ReportFormat)
        assert "json" in formats

    def test_platform_info_categories_list(self, platform):
        info = platform.get_platform_info()
        cats = info["categories"]
        assert len(cats) == 6
        assert ISOCategory.CATEGORY_1_DIRECT.value in cats


class TestSharedStoreWiring:
    """Test that engines share the same in-memory stores."""

    def test_boundary_creates_org_visible_in_health(self, platform):
        platform.boundary.create_organization("Test Corp", "tech", "US")
        health = platform.health_check()
        assert health["organization_count"] == 1

    def test_boundary_creates_inventory_visible_in_health(self, platform):
        org = platform.boundary.create_organization("Test Corp", "tech", "US")
        platform.boundary.create_inventory(org.id, 2025)
        health = platform.health_check()
        assert health["inventory_count"] == 1

    def test_multiple_orgs_counted(self, platform):
        platform.boundary.create_organization("Org A", "energy", "US")
        platform.boundary.create_organization("Org B", "energy", "DE")
        health = platform.health_check()
        assert health["organization_count"] == 2
