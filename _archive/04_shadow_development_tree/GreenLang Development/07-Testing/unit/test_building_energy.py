# -*- coding: utf-8 -*-
"""
Unit Tests for Building Energy Performance Agent

Comprehensive test suite with 20 test cases covering:
- CalculateEuiTool (7 tests)
- LookupBpsThresholdTool (7 tests)
- CheckBpsComplianceTool (6 tests)

Target: 85%+ coverage for building energy tools
Run with: pytest tests/unit/test_building_energy.py -v --cov=generated/energy_performance_v1

Author: GL-TestEngineer
Version: 1.0.0

Building Performance Standards (BPS) are regulations that require buildings
to meet energy efficiency thresholds, commonly measured in Energy Use Intensity (EUI).
"""

import pytest
import asyncio
import hashlib
import json
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "generated" / "energy_performance_v1"))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def calculate_eui_tool():
    """Create CalculateEuiTool instance."""
    from generated.energy_performance_v1.tools import CalculateEuiTool
    return CalculateEuiTool()


@pytest.fixture
def lookup_bps_threshold_tool():
    """Create LookupBpsThresholdTool instance."""
    from generated.energy_performance_v1.tools import LookupBpsThresholdTool
    return LookupBpsThresholdTool()


@pytest.fixture
def check_bps_compliance_tool():
    """Create CheckBpsComplianceTool instance."""
    from generated.energy_performance_v1.tools import CheckBpsComplianceTool
    return CheckBpsComplianceTool()


@pytest.fixture
def mock_bps_database():
    """Create mock BPS threshold database."""
    mock_db = Mock()
    mock_threshold = Mock()
    mock_threshold.building_type = "office"
    mock_threshold.climate_zone = "4A"
    mock_threshold.threshold_kwh_per_sqm = 80.0
    mock_threshold.ghg_threshold_kgco2e_per_sqm = 5.4
    mock_threshold.source = "NYC Local Law 97 2024-2029"
    mock_threshold.jurisdiction = "NYC"
    mock_threshold.effective_date = "2024-01-01"
    mock_threshold.notes = "Office buildings in NYC Climate Zone 4A"
    mock_db.lookup.return_value = mock_threshold
    mock_db.list_building_types.return_value = [
        "office", "residential", "retail", "industrial",
        "warehouse", "hotel", "hospital", "school"
    ]
    return mock_db


@pytest.fixture
def sample_building_types():
    """Sample building types for testing."""
    return ["office", "residential", "retail", "industrial", "warehouse", "hotel", "hospital", "school"]


@pytest.fixture
def sample_climate_zones():
    """Sample ASHRAE climate zones for testing."""
    return ["1A", "2A", "3A", "4A", "5A", "6A", "7"]


# =============================================================================
# CalculateEuiTool Tests (7 tests)
# =============================================================================

class TestCalculateEuiTool:
    """Test suite for CalculateEuiTool - 7 test cases."""

    @pytest.mark.unit
    @pytest.mark.building_energy
    def test_tool_initialization(self, calculate_eui_tool):
        """UT-BE-001: Test tool initializes correctly."""
        assert calculate_eui_tool is not None
        assert calculate_eui_tool.name == "calculate_eui"
        assert calculate_eui_tool.safe is True
        assert "Energy Use Intensity" in calculate_eui_tool.description

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_missing_energy_consumption_raises_error(self, calculate_eui_tool):
        """UT-BE-002: Test missing energy_consumption_kwh raises ValueError."""
        params = {"floor_area_sqm": 1000.0}

        with pytest.raises(ValueError) as exc_info:
            await calculate_eui_tool.execute(params)

        assert "energy_consumption_kwh" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_missing_floor_area_raises_error(self, calculate_eui_tool):
        """UT-BE-003: Test missing floor_area_sqm raises ValueError."""
        params = {"energy_consumption_kwh": 80000.0}

        with pytest.raises(ValueError) as exc_info:
            await calculate_eui_tool.execute(params)

        assert "floor_area_sqm" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_basic_eui_calculation(self, calculate_eui_tool):
        """UT-BE-004: Test basic EUI calculation."""
        params = {
            "energy_consumption_kwh": 80000.0,  # 80,000 kWh/year
            "floor_area_sqm": 1000.0,  # 1,000 sqm
        }

        result = await calculate_eui_tool.execute(params)

        assert result is not None
        assert "eui_kwh_per_sqm" in result
        assert "calculation_formula" in result

        # Verify calculation: 80000 / 1000 = 80 kWh/sqm/year
        expected_eui = 80000.0 / 1000.0
        assert abs(result["eui_kwh_per_sqm"] - expected_eui) < 0.0001

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_zero_floor_area_raises_error(self, calculate_eui_tool):
        """UT-BE-005: Test zero floor area raises ValueError."""
        params = {
            "energy_consumption_kwh": 80000.0,
            "floor_area_sqm": 0.0,
        }

        with pytest.raises(ValueError) as exc_info:
            await calculate_eui_tool.execute(params)

        assert "positive" in str(exc_info.value).lower() or "floor" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_eui_calculation_determinism(self, calculate_eui_tool):
        """UT-BE-006: Test EUI calculation is deterministic."""
        params = {
            "energy_consumption_kwh": 123456.789,
            "floor_area_sqm": 987.654,
        }

        results = []
        for _ in range(10):
            result = await calculate_eui_tool.execute(params)
            results.append(result["eui_kwh_per_sqm"])

        # All results must be identical
        assert all(r == results[0] for r in results), "EUI results are not deterministic"

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_eui_result_includes_provenance(self, calculate_eui_tool):
        """UT-BE-007: Test EUI result includes provenance hash and timestamp."""
        params = {
            "energy_consumption_kwh": 80000.0,
            "floor_area_sqm": 1000.0,
        }

        result = await calculate_eui_tool.execute(params)

        assert "result_hash" in result
        assert len(result["result_hash"]) == 64  # SHA-256 hex length
        assert "executed_at" in result
        # Should be a valid ISO timestamp
        datetime.fromisoformat(result["executed_at"])


# =============================================================================
# LookupBpsThresholdTool Tests (7 tests)
# =============================================================================

class TestLookupBpsThresholdTool:
    """Test suite for LookupBpsThresholdTool - 7 test cases."""

    @pytest.mark.unit
    @pytest.mark.building_energy
    def test_tool_initialization(self, lookup_bps_threshold_tool):
        """UT-BE-008: Test tool initializes correctly."""
        assert lookup_bps_threshold_tool is not None
        assert lookup_bps_threshold_tool.name == "lookup_bps_threshold"
        assert lookup_bps_threshold_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_missing_building_type_raises_error(self, lookup_bps_threshold_tool):
        """UT-BE-009: Test missing building_type raises ValueError."""
        params = {"climate_zone": "4A"}

        with pytest.raises(ValueError) as exc_info:
            await lookup_bps_threshold_tool.execute(params)

        assert "building_type" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_valid_lookup_returns_threshold(self, lookup_bps_threshold_tool, mock_bps_database):
        """UT-BE-010: Test valid lookup returns threshold data."""
        with patch('generated.energy_performance_v1.tools.get_bps_database', return_value=mock_bps_database):
            params = {
                "building_type": "office",
                "climate_zone": "4A",
            }

            result = await lookup_bps_threshold_tool.execute(params)

            assert result is not None
            assert "threshold_kwh_per_sqm" in result
            assert result["threshold_kwh_per_sqm"] == 80.0
            assert "source" in result
            assert "jurisdiction" in result

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_lookup_without_climate_zone_uses_default(self, lookup_bps_threshold_tool, mock_bps_database):
        """UT-BE-011: Test lookup without climate zone uses default threshold."""
        with patch('generated.energy_performance_v1.tools.get_bps_database', return_value=mock_bps_database):
            params = {"building_type": "office"}

            result = await lookup_bps_threshold_tool.execute(params)

            assert result is not None
            assert "threshold_kwh_per_sqm" in result

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_invalid_building_type_raises_error(self, lookup_bps_threshold_tool):
        """UT-BE-012: Test invalid building type raises appropriate error."""
        mock_db = Mock()
        mock_db.lookup.return_value = None
        mock_db.list_building_types.return_value = ["office", "residential"]

        with patch('generated.energy_performance_v1.tools.get_bps_database', return_value=mock_db):
            params = {"building_type": "invalid_type_xyz"}

            with pytest.raises(ValueError) as exc_info:
                await lookup_bps_threshold_tool.execute(params)

            assert "not found" in str(exc_info.value).lower()
            # Should include available types in error message
            assert "office" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_lookup_includes_ghg_threshold(self, lookup_bps_threshold_tool, mock_bps_database):
        """UT-BE-013: Test lookup result includes GHG threshold."""
        with patch('generated.energy_performance_v1.tools.get_bps_database', return_value=mock_bps_database):
            params = {"building_type": "office", "climate_zone": "4A"}

            result = await lookup_bps_threshold_tool.execute(params)

            assert "ghg_threshold_kgco2e_per_sqm" in result
            assert result["ghg_threshold_kgco2e_per_sqm"] == 5.4

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_lookup_includes_effective_date(self, lookup_bps_threshold_tool, mock_bps_database):
        """UT-BE-014: Test lookup result includes effective date."""
        with patch('generated.energy_performance_v1.tools.get_bps_database', return_value=mock_bps_database):
            params = {"building_type": "office", "climate_zone": "4A"}

            result = await lookup_bps_threshold_tool.execute(params)

            assert "effective_date" in result
            assert result["effective_date"] == "2024-01-01"


# =============================================================================
# CheckBpsComplianceTool Tests (6 tests)
# =============================================================================

class TestCheckBpsComplianceTool:
    """Test suite for CheckBpsComplianceTool - 6 test cases."""

    @pytest.mark.unit
    @pytest.mark.building_energy
    def test_tool_initialization(self, check_bps_compliance_tool):
        """UT-BE-015: Test tool initializes correctly."""
        assert check_bps_compliance_tool is not None
        assert check_bps_compliance_tool.name == "check_bps_compliance"
        assert check_bps_compliance_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_compliant_building(self, check_bps_compliance_tool):
        """UT-BE-016: Test compliant building returns compliant=True."""
        params = {
            "actual_eui": 70.0,  # 70 kWh/sqm/year (below threshold)
            "threshold_eui": 80.0,  # 80 kWh/sqm/year
        }

        result = await check_bps_compliance_tool.execute(params)

        assert result["compliant"] is True
        assert result["gap_kwh_per_sqm"] < 0  # Negative gap means under threshold
        assert result["compliance_status"] == "COMPLIANT"

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_non_compliant_building(self, check_bps_compliance_tool):
        """UT-BE-017: Test non-compliant building returns compliant=False."""
        params = {
            "actual_eui": 100.0,  # 100 kWh/sqm/year (above threshold)
            "threshold_eui": 80.0,  # 80 kWh/sqm/year
        }

        result = await check_bps_compliance_tool.execute(params)

        assert result["compliant"] is False
        assert result["gap_kwh_per_sqm"] > 0  # Positive gap means over threshold
        assert result["gap_kwh_per_sqm"] == 20.0  # 100 - 80 = 20
        assert result["compliance_status"] == "NON-COMPLIANT"

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_exactly_at_threshold_is_compliant(self, check_bps_compliance_tool):
        """UT-BE-018: Test building exactly at threshold is compliant."""
        params = {
            "actual_eui": 80.0,  # Exactly at threshold
            "threshold_eui": 80.0,
        }

        result = await check_bps_compliance_tool.execute(params)

        assert result["compliant"] is True
        assert result["gap_kwh_per_sqm"] == 0.0
        assert result["compliance_status"] == "COMPLIANT"

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_percentage_difference_calculation(self, check_bps_compliance_tool):
        """UT-BE-019: Test percentage difference is calculated correctly."""
        params = {
            "actual_eui": 100.0,  # 25% over threshold
            "threshold_eui": 80.0,
        }

        result = await check_bps_compliance_tool.execute(params)

        # Gap = 20, Threshold = 80, Percentage = 20/80 * 100 = 25%
        assert "percentage_difference" in result
        assert abs(result["percentage_difference"] - 25.0) < 0.01

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_negative_eui_raises_error(self, check_bps_compliance_tool):
        """UT-BE-020: Test negative EUI raises ValueError."""
        params = {
            "actual_eui": -10.0,  # Invalid negative value
            "threshold_eui": 80.0,
        }

        with pytest.raises(ValueError) as exc_info:
            await check_bps_compliance_tool.execute(params)

        assert "negative" in str(exc_info.value).lower()


# =============================================================================
# BPS Compliance End-to-End Tests
# =============================================================================

class TestBPSComplianceWorkflow:
    """Test suite for BPS compliance end-to-end workflow."""

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.asyncio
    async def test_full_compliance_workflow(
        self,
        calculate_eui_tool,
        lookup_bps_threshold_tool,
        check_bps_compliance_tool,
        mock_bps_database
    ):
        """Test full BPS compliance assessment workflow."""
        # Step 1: Calculate actual EUI
        eui_params = {
            "energy_consumption_kwh": 70000.0,  # 70,000 kWh/year
            "floor_area_sqm": 1000.0,  # 1,000 sqm
        }
        eui_result = await calculate_eui_tool.execute(eui_params)

        assert eui_result["eui_kwh_per_sqm"] == 70.0

        # Step 2: Look up threshold
        with patch('generated.energy_performance_v1.tools.get_bps_database', return_value=mock_bps_database):
            threshold_params = {"building_type": "office", "climate_zone": "4A"}
            threshold_result = await lookup_bps_threshold_tool.execute(threshold_params)

            assert threshold_result["threshold_kwh_per_sqm"] == 80.0

        # Step 3: Check compliance
        compliance_params = {
            "actual_eui": eui_result["eui_kwh_per_sqm"],
            "threshold_eui": threshold_result["threshold_kwh_per_sqm"],
        }
        compliance_result = await check_bps_compliance_tool.execute(compliance_params)

        assert compliance_result["compliant"] is True
        assert compliance_result["gap_kwh_per_sqm"] == -10.0  # 10 kWh/sqm under threshold


# =============================================================================
# Tool Registry Tests
# =============================================================================

class TestBuildingEnergyToolRegistry:
    """Test suite for building energy tool registry functionality."""

    @pytest.mark.unit
    @pytest.mark.building_energy
    def test_get_tool_returns_correct_tool(self):
        """Test get_tool function returns correct tool instances."""
        from generated.energy_performance_v1.tools import get_tool

        eui_tool = get_tool("calculate_eui")
        threshold_tool = get_tool("lookup_bps_threshold")
        compliance_tool = get_tool("check_bps_compliance")

        assert eui_tool is not None
        assert threshold_tool is not None
        assert compliance_tool is not None

        assert eui_tool.name == "calculate_eui"
        assert threshold_tool.name == "lookup_bps_threshold"
        assert compliance_tool.name == "check_bps_compliance"

    @pytest.mark.unit
    @pytest.mark.building_energy
    def test_list_tools_returns_all_tools(self):
        """Test list_tools returns all available tools."""
        from generated.energy_performance_v1.tools import list_tools

        tools = list_tools()

        assert "calculate_eui" in tools
        assert "lookup_bps_threshold" in tools
        assert "check_bps_compliance" in tools
        assert len(tools) == 3


# =============================================================================
# Provenance and Determinism Tests
# =============================================================================

class TestBuildingEnergyProvenance:
    """Test suite for building energy provenance tracking."""

    @pytest.mark.unit
    @pytest.mark.building_energy
    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_compliance_determinism(self, check_bps_compliance_tool):
        """Test compliance check is deterministic across multiple calls."""
        params = {
            "actual_eui": 85.5,
            "threshold_eui": 80.0,
        }

        results = []
        for _ in range(5):
            result = await check_bps_compliance_tool.execute(params)
            results.append((
                result["compliant"],
                result["gap_kwh_per_sqm"],
                result["percentage_difference"]
            ))

        # All results should be identical
        assert all(r == results[0] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
