# -*- coding: utf-8 -*-
"""Shared fixtures for AGENT-MRV-013 Dual Reporting Reconciliation tests."""

import pytest
import sys
from decimal import Decimal
from typing import Any, Dict, List


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all singletons before each test."""
    modules = [
        "greenlang.agents.mrv.dual_reporting_reconciliation.config",
        "greenlang.agents.mrv.dual_reporting_reconciliation.metrics",
        "greenlang.agents.mrv.dual_reporting_reconciliation.provenance",
        "greenlang.agents.mrv.dual_reporting_reconciliation.dual_result_collector",
        "greenlang.agents.mrv.dual_reporting_reconciliation.discrepancy_analyzer",
        "greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer",
        "greenlang.agents.mrv.dual_reporting_reconciliation.reporting_table_generator",
        "greenlang.agents.mrv.dual_reporting_reconciliation.trend_analyzer",
        "greenlang.agents.mrv.dual_reporting_reconciliation.compliance_checker",
        "greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline",
        "greenlang.agents.mrv.dual_reporting_reconciliation.setup",
    ]
    for mod_name in modules:
        try:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "reset"):
                mod.reset()
            elif mod and hasattr(mod, "reset_service"):
                mod.reset_service()
        except Exception:
            pass
    yield


@pytest.fixture
def sample_location_result() -> Dict[str, Any]:
    """Return a sample location-based upstream result.

    Matches the UpstreamResult Pydantic model field requirements:
    agent, method, energy_type (enums), tenant_id, period_start, period_end (required).
    """
    return {
        "agent": "mrv_009",
        "facility_id": "FAC-001",
        "energy_type": "electricity",
        "method": "location_based",
        "emissions_tco2e": Decimal("1250.50"),
        "energy_quantity_mwh": Decimal("5000.0"),
        "ef_used": Decimal("0.2501"),
        "ef_source": "eGRID 2023 CAMX",
        "ef_hierarchy": "grid_average",
        "tier": "tier_1",
        "gwp_source": "AR5",
        "provenance_hash": "abc123def456",
        "tenant_id": "tenant-001",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "region": "US-CAMX",
    }


@pytest.fixture
def sample_market_result() -> Dict[str, Any]:
    """Return a sample market-based upstream result.

    Uses ``supplier_no_cert`` for ef_hierarchy since ``supplier_specific``
    is not a valid EFHierarchyPriority enum value.
    """
    return {
        "agent": "mrv_010",
        "facility_id": "FAC-001",
        "energy_type": "electricity",
        "method": "market_based",
        "emissions_tco2e": Decimal("625.25"),
        "energy_quantity_mwh": Decimal("5000.0"),
        "ef_used": Decimal("0.12505"),
        "ef_source": "Supplier Disclosure 2024",
        "ef_hierarchy": "supplier_no_cert",
        "tier": "tier_3",
        "gwp_source": "AR5",
        "provenance_hash": "def456ghi789",
        "tenant_id": "tenant-001",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "region": "US-CAMX",
    }


@pytest.fixture
def sample_upstream_results(
    sample_location_result,
    sample_market_result,
) -> List[Dict[str, Any]]:
    """Return a list of location + market upstream results."""
    return [sample_location_result, sample_market_result]


@pytest.fixture
def sample_steam_location_result() -> Dict[str, Any]:
    """Return a sample location-based steam result."""
    return {
        "agent": "mrv_011",
        "facility_id": "FAC-002",
        "energy_type": "steam",
        "method": "location_based",
        "emissions_tco2e": Decimal("350.75"),
        "energy_quantity_mwh": Decimal("2000.0"),
        "ef_used": Decimal("0.175375"),
        "ef_source": "IEA 2023",
        "ef_hierarchy": "grid_average",
        "tier": "tier_1",
        "gwp_source": "AR5",
        "provenance_hash": "steam_loc_hash",
        "tenant_id": "tenant-001",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
    }


@pytest.fixture
def sample_steam_market_result() -> Dict[str, Any]:
    """Return a sample market-based steam result."""
    return {
        "agent": "mrv_011",
        "facility_id": "FAC-002",
        "energy_type": "steam",
        "method": "market_based",
        "emissions_tco2e": Decimal("300.50"),
        "energy_quantity_mwh": Decimal("2000.0"),
        "ef_used": Decimal("0.15025"),
        "ef_source": "Supplier CHP Allocation",
        "ef_hierarchy": "supplier_no_cert",
        "tier": "tier_2",
        "gwp_source": "AR5",
        "provenance_hash": "steam_mkt_hash",
        "tenant_id": "tenant-001",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
    }


@pytest.fixture
def sample_reconciliation_request(
    sample_upstream_results,
) -> Dict[str, Any]:
    """Return a complete reconciliation request dictionary."""
    return {
        "tenant_id": "tenant-001",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "upstream_results": sample_upstream_results,
        "frameworks": ["ghg_protocol", "csrd_esrs"],
        "include_trends": True,
        "include_compliance": True,
    }


@pytest.fixture
def sample_multi_facility_results(
    sample_location_result,
    sample_market_result,
    sample_steam_location_result,
    sample_steam_market_result,
) -> List[Dict[str, Any]]:
    """Return upstream results across multiple facilities and energy types."""
    return [
        sample_location_result,
        sample_market_result,
        sample_steam_location_result,
        sample_steam_market_result,
    ]


@pytest.fixture
def sample_trend_data() -> List[Dict[str, Any]]:
    """Return sample multi-period trend data points."""
    return [
        {
            "period": "2021",
            "period_start": "2021-01-01",
            "period_end": "2021-12-31",
            "location_tco2e": 2000.0,
            "market_tco2e": 1800.0,
        },
        {
            "period": "2022",
            "period_start": "2022-01-01",
            "period_end": "2022-12-31",
            "location_tco2e": 1900.0,
            "market_tco2e": 1500.0,
        },
        {
            "period": "2023",
            "period_start": "2023-01-01",
            "period_end": "2023-12-31",
            "location_tco2e": 1850.0,
            "market_tco2e": 1200.0,
        },
        {
            "period": "2024",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "location_tco2e": 1800.0,
            "market_tco2e": 1000.0,
        },
    ]


@pytest.fixture
def sample_batch_request(sample_upstream_results) -> Dict[str, Any]:
    """Return a sample batch reconciliation request."""
    return {
        "batch_id": "BATCH-001",
        "tenant_id": "tenant-001",
        "periods": [
            {
                "period_start": "2023-01-01",
                "period_end": "2023-12-31",
                "upstream_results": sample_upstream_results,
            },
            {
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
                "upstream_results": sample_upstream_results,
            },
        ],
    }
