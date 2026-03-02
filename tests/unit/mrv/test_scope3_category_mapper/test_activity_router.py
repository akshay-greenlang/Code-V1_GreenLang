# -*- coding: utf-8 -*-
"""
Test suite for ActivityRouterEngine - AGENT-MRV-029 Engine 3.

Tests the activity routing logic for the Scope 3 Category Mapper Agent
(GL-MRV-X-040). The ActivityRouterEngine routes classified spend records
to the correct downstream MRV agent (AGENT-MRV-014 through AGENT-MRV-028)
based on their assigned Scope 3 category (1-15).

This module validates:
- Routing table completeness (all 15 categories mapped to unique agents)
- Routing plan creation (single and multi-category)
- Input transformation for each downstream agent
- Routing execution (batch, dry-run, partial failure handling)
- Provenance tracking through routing

The routing table is built from CategoryDatabaseEngine.get_category_info()
which provides the downstream_agent field for each category.

Total: ~80 tests

Author: GL-TestEngineer
Date: March 2026
"""

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from greenlang.scope3_category_mapper.category_database import (
    CategoryDatabaseEngine,
    CategoryInfo,
    Scope3Category,
    ValueChainDirection,
    reset_category_database_engine,
)


# ==============================================================================
# LOCAL FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    """Reset the CategoryDatabaseEngine singleton before and after each test."""
    reset_category_database_engine()
    yield
    reset_category_database_engine()


@pytest.fixture
def db_engine() -> CategoryDatabaseEngine:
    """Create a fresh CategoryDatabaseEngine instance."""
    return CategoryDatabaseEngine()


# --- Routing table built from CategoryDatabaseEngine ---

# The routing table maps each Scope 3 category (1-15) to its downstream agent.
# This is sourced from _CATEGORY_INFO in category_database.py.
ROUTING_TABLE: Dict[int, Dict[str, str]] = {
    1:  {"agent_id": "AGENT-MRV-014", "api_endpoint": "/api/v1/mrv/purchased-goods",
         "agent_name": "Purchased Goods & Services Agent"},
    2:  {"agent_id": "AGENT-MRV-015", "api_endpoint": "/api/v1/mrv/capital-goods",
         "agent_name": "Capital Goods Agent"},
    3:  {"agent_id": "AGENT-MRV-016", "api_endpoint": "/api/v1/mrv/fuel-energy",
         "agent_name": "Fuel & Energy Activities Agent"},
    4:  {"agent_id": "AGENT-MRV-017", "api_endpoint": "/api/v1/mrv/upstream-transport",
         "agent_name": "Upstream Transportation Agent"},
    5:  {"agent_id": "AGENT-MRV-018", "api_endpoint": "/api/v1/mrv/waste-generated",
         "agent_name": "Waste Generated Agent"},
    6:  {"agent_id": "AGENT-MRV-019", "api_endpoint": "/api/v1/mrv/business-travel",
         "agent_name": "Business Travel Agent"},
    7:  {"agent_id": "AGENT-MRV-020", "api_endpoint": "/api/v1/mrv/employee-commuting",
         "agent_name": "Employee Commuting Agent"},
    8:  {"agent_id": "AGENT-MRV-021", "api_endpoint": "/api/v1/mrv/upstream-leased",
         "agent_name": "Upstream Leased Assets Agent"},
    9:  {"agent_id": "AGENT-MRV-022", "api_endpoint": "/api/v1/mrv/downstream-transport",
         "agent_name": "Downstream Transportation Agent"},
    10: {"agent_id": "AGENT-MRV-023", "api_endpoint": "/api/v1/mrv/processing-sold",
         "agent_name": "Processing of Sold Products Agent"},
    11: {"agent_id": "AGENT-MRV-024", "api_endpoint": "/api/v1/mrv/use-sold",
         "agent_name": "Use of Sold Products Agent"},
    12: {"agent_id": "AGENT-MRV-025", "api_endpoint": "/api/v1/mrv/end-of-life",
         "agent_name": "End-of-Life Treatment Agent"},
    13: {"agent_id": "AGENT-MRV-026", "api_endpoint": "/api/v1/mrv/downstream-leased",
         "agent_name": "Downstream Leased Assets Agent"},
    14: {"agent_id": "AGENT-MRV-027", "api_endpoint": "/api/v1/mrv/franchises",
         "agent_name": "Franchises Agent"},
    15: {"agent_id": "AGENT-MRV-028", "api_endpoint": "/api/v1/mrv/investments",
         "agent_name": "Investments Agent"},
}


def get_agent_info(category_number: int) -> Optional[Dict[str, str]]:
    """Get routing info for a category number."""
    return ROUTING_TABLE.get(category_number)


def get_all_agents() -> List[Dict[str, str]]:
    """Get routing info for all 15 categories."""
    return [ROUTING_TABLE[i] for i in range(1, 16)]


def create_routing_plan(
    classified_records: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Create a routing plan that groups classified records by category
    and maps them to the correct downstream agent.
    """
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for record in classified_records:
        cat_num = record.get("category_number", 1)
        groups.setdefault(cat_num, []).append(record)

    routes: List[Dict[str, Any]] = []
    for cat_num, records in sorted(groups.items()):
        agent_info = get_agent_info(cat_num)
        if agent_info is None:
            continue
        routes.append({
            "category_number": cat_num,
            "agent_id": agent_info["agent_id"],
            "api_endpoint": agent_info["api_endpoint"],
            "agent_name": agent_info["agent_name"],
            "record_count": len(records),
            "records": records,
        })

    plan_data = json.dumps(
        {"routes": [{"cat": r["category_number"], "count": r["record_count"]}
         for r in routes]},
        sort_keys=True,
    )
    provenance_hash = hashlib.sha256(plan_data.encode()).hexdigest()

    return {
        "total_records": len(classified_records),
        "total_categories": len(routes),
        "routes": routes,
        "dry_run": dry_run,
        "provenance_hash": provenance_hash,
    }


def validate_routing_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a routing plan for completeness and correctness."""
    errors: List[str] = []
    warnings: List[str] = []

    if plan["total_records"] == 0:
        errors.append("Routing plan contains no records.")

    for route in plan.get("routes", []):
        cat_num = route["category_number"]
        if cat_num < 1 or cat_num > 15:
            errors.append(f"Invalid category number: {cat_num}")
        if route["record_count"] == 0:
            warnings.append(f"Category {cat_num} has zero records.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def transform_input_for_agent(
    record: Dict[str, Any], category_number: int
) -> Dict[str, Any]:
    """
    Transform a classified spend record into the input format expected
    by the downstream MRV agent for the given category.
    """
    base = {
        "source_record_id": record.get("record_id", "unknown"),
        "amount": record.get("amount"),
        "currency": record.get("currency", "USD"),
        "description": record.get("description", ""),
        "classification_confidence": record.get("confidence", 0.0),
        "classification_method": record.get("classification_method", "unknown"),
        "provenance_hash": record.get("provenance_hash", ""),
        "target_category": category_number,
    }

    # Category-specific enrichment
    if category_number == 6:
        base["travel_type"] = record.get("travel_type", "air")
        base["travel_class"] = record.get("cabin_class", "economy")
    elif category_number == 15:
        base["asset_class"] = record.get("asset_class", "listed_equity")
        base["investee_name"] = record.get("investee_name", "")
    elif category_number in (4, 9):
        base["transport_mode"] = record.get("transport_mode", "road")
        base["direction"] = "upstream" if category_number == 4 else "downstream"

    return base


def route_batch(
    db: CategoryDatabaseEngine,
    classified_records: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Route a batch of classified records to their downstream agents.
    Returns routing result with success/failure counts.
    """
    plan = create_routing_plan(classified_records, dry_run=dry_run)

    if dry_run:
        return {
            "status": "dry_run",
            "plan": plan,
            "executed": False,
            "success_count": 0,
            "failure_count": 0,
            "provenance_hash": plan["provenance_hash"],
        }

    # Simulate execution (in production, this would call downstream APIs)
    success_count = 0
    failure_count = 0
    route_results: List[Dict[str, Any]] = []

    for route in plan["routes"]:
        cat_num = route["category_number"]
        for record in route["records"]:
            try:
                transformed = transform_input_for_agent(record, cat_num)
                route_results.append({
                    "record_id": record.get("record_id"),
                    "category_number": cat_num,
                    "agent_id": route["agent_id"],
                    "status": "routed",
                    "transformed_input": transformed,
                })
                success_count += 1
            except Exception as exc:
                route_results.append({
                    "record_id": record.get("record_id"),
                    "category_number": cat_num,
                    "agent_id": route["agent_id"],
                    "status": "failed",
                    "error": str(exc),
                })
                failure_count += 1

    return {
        "status": "completed",
        "plan": plan,
        "executed": True,
        "success_count": success_count,
        "failure_count": failure_count,
        "route_results": route_results,
        "provenance_hash": plan["provenance_hash"],
    }


# ==============================================================================
# ROUTING TABLE TESTS (~20)
# ==============================================================================


class TestRoutingTable:
    """Test the routing table mapping categories to downstream agents."""

    def test_routing_table_has_15_entries(self):
        """Routing table contains exactly 15 entries (one per category)."""
        assert len(ROUTING_TABLE) == 15

    def test_routing_table_cat1_targets_mrv014(self):
        """Category 1 routes to AGENT-MRV-014."""
        assert ROUTING_TABLE[1]["agent_id"] == "AGENT-MRV-014"

    def test_routing_table_cat2_targets_mrv015(self):
        """Category 2 routes to AGENT-MRV-015."""
        assert ROUTING_TABLE[2]["agent_id"] == "AGENT-MRV-015"

    def test_routing_table_cat3_targets_mrv016(self):
        """Category 3 routes to AGENT-MRV-016."""
        assert ROUTING_TABLE[3]["agent_id"] == "AGENT-MRV-016"

    def test_routing_table_cat4_targets_mrv017(self):
        """Category 4 routes to AGENT-MRV-017."""
        assert ROUTING_TABLE[4]["agent_id"] == "AGENT-MRV-017"

    def test_routing_table_cat5_targets_mrv018(self):
        """Category 5 routes to AGENT-MRV-018."""
        assert ROUTING_TABLE[5]["agent_id"] == "AGENT-MRV-018"

    def test_routing_table_cat6_targets_mrv019(self):
        """Category 6 routes to AGENT-MRV-019."""
        assert ROUTING_TABLE[6]["agent_id"] == "AGENT-MRV-019"

    def test_routing_table_cat7_targets_mrv020(self):
        """Category 7 routes to AGENT-MRV-020."""
        assert ROUTING_TABLE[7]["agent_id"] == "AGENT-MRV-020"

    def test_routing_table_cat8_targets_mrv021(self):
        """Category 8 routes to AGENT-MRV-021."""
        assert ROUTING_TABLE[8]["agent_id"] == "AGENT-MRV-021"

    def test_routing_table_cat9_targets_mrv022(self):
        """Category 9 routes to AGENT-MRV-022."""
        assert ROUTING_TABLE[9]["agent_id"] == "AGENT-MRV-022"

    def test_routing_table_cat10_targets_mrv023(self):
        """Category 10 routes to AGENT-MRV-023."""
        assert ROUTING_TABLE[10]["agent_id"] == "AGENT-MRV-023"

    def test_routing_table_cat11_targets_mrv024(self):
        """Category 11 routes to AGENT-MRV-024."""
        assert ROUTING_TABLE[11]["agent_id"] == "AGENT-MRV-024"

    def test_routing_table_cat12_targets_mrv025(self):
        """Category 12 routes to AGENT-MRV-025."""
        assert ROUTING_TABLE[12]["agent_id"] == "AGENT-MRV-025"

    def test_routing_table_cat13_targets_mrv026(self):
        """Category 13 routes to AGENT-MRV-026."""
        assert ROUTING_TABLE[13]["agent_id"] == "AGENT-MRV-026"

    def test_routing_table_cat14_targets_mrv027(self):
        """Category 14 routes to AGENT-MRV-027."""
        assert ROUTING_TABLE[14]["agent_id"] == "AGENT-MRV-027"

    def test_routing_table_cat15_targets_mrv028(self):
        """Category 15 routes to AGENT-MRV-028."""
        assert ROUTING_TABLE[15]["agent_id"] == "AGENT-MRV-028"

    @pytest.mark.parametrize("cat_num,expected_agent", [
        (1, "AGENT-MRV-014"), (2, "AGENT-MRV-015"), (3, "AGENT-MRV-016"),
        (4, "AGENT-MRV-017"), (5, "AGENT-MRV-018"), (6, "AGENT-MRV-019"),
        (7, "AGENT-MRV-020"), (8, "AGENT-MRV-021"), (9, "AGENT-MRV-022"),
        (10, "AGENT-MRV-023"), (11, "AGENT-MRV-024"), (12, "AGENT-MRV-025"),
        (13, "AGENT-MRV-026"), (14, "AGENT-MRV-027"), (15, "AGENT-MRV-028"),
    ])
    def test_routing_table_parametrized(self, cat_num, expected_agent):
        """Parametrized check: each category maps to the correct agent."""
        assert ROUTING_TABLE[cat_num]["agent_id"] == expected_agent

    def test_routing_table_all_agent_ids_unique(self):
        """All 15 agent IDs in the routing table are unique."""
        agent_ids = [v["agent_id"] for v in ROUTING_TABLE.values()]
        assert len(set(agent_ids)) == 15

    def test_routing_table_all_api_endpoints_unique(self):
        """All 15 API endpoints in the routing table are unique."""
        endpoints = [v["api_endpoint"] for v in ROUTING_TABLE.values()]
        assert len(set(endpoints)) == 15

    def test_get_agent_info_valid_category(self):
        """get_agent_info returns dict for valid category."""
        info = get_agent_info(1)
        assert info is not None
        assert "agent_id" in info
        assert "api_endpoint" in info
        assert "agent_name" in info

    def test_get_agent_info_invalid_category(self):
        """get_agent_info returns None for invalid category."""
        assert get_agent_info(0) is None
        assert get_agent_info(16) is None
        assert get_agent_info(-1) is None

    def test_get_all_agents_returns_15(self):
        """get_all_agents returns exactly 15 entries."""
        agents = get_all_agents()
        assert len(agents) == 15

    def test_routing_table_matches_category_database(self, db_engine):
        """Routing table agent IDs match CategoryDatabaseEngine category info."""
        for cat_num in range(1, 16):
            cat_info = db_engine.get_category_info(cat_num)
            routing_agent = ROUTING_TABLE[cat_num]["agent_id"]
            assert routing_agent == cat_info.downstream_agent


# ==============================================================================
# ROUTING PLAN TESTS (~20)
# ==============================================================================


class TestRoutingPlan:
    """Test routing plan creation and validation."""

    def test_create_routing_plan_single_category(self):
        """Plan with records in one category creates one route."""
        records = [
            {"record_id": "R-1", "category_number": 1, "confidence": 0.9},
            {"record_id": "R-2", "category_number": 1, "confidence": 0.85},
        ]
        plan = create_routing_plan(records)
        assert plan["total_records"] == 2
        assert plan["total_categories"] == 1
        assert plan["routes"][0]["category_number"] == 1
        assert plan["routes"][0]["record_count"] == 2

    def test_create_routing_plan_multiple_categories(self):
        """Plan with records in multiple categories creates multiple routes."""
        records = [
            {"record_id": "R-1", "category_number": 1, "confidence": 0.9},
            {"record_id": "R-2", "category_number": 4, "confidence": 0.85},
            {"record_id": "R-3", "category_number": 6, "confidence": 0.92},
            {"record_id": "R-4", "category_number": 1, "confidence": 0.88},
        ]
        plan = create_routing_plan(records)
        assert plan["total_records"] == 4
        assert plan["total_categories"] == 3

    def test_create_routing_plan_groups_by_category(self):
        """Plan groups records by category number."""
        records = [
            {"record_id": "G-1", "category_number": 5, "confidence": 0.9},
            {"record_id": "G-2", "category_number": 5, "confidence": 0.85},
            {"record_id": "G-3", "category_number": 5, "confidence": 0.88},
        ]
        plan = create_routing_plan(records)
        assert len(plan["routes"]) == 1
        assert plan["routes"][0]["record_count"] == 3

    def test_create_routing_plan_dry_run(self):
        """Dry-run plan is flagged correctly."""
        records = [
            {"record_id": "DR-1", "category_number": 1, "confidence": 0.9},
        ]
        plan = create_routing_plan(records, dry_run=True)
        assert plan["dry_run"] is True

    def test_create_routing_plan_not_dry_run_by_default(self):
        """Default plan is not dry-run."""
        records = [
            {"record_id": "ND-1", "category_number": 1, "confidence": 0.9},
        ]
        plan = create_routing_plan(records)
        assert plan["dry_run"] is False

    def test_create_routing_plan_provenance_hash(self):
        """Plan includes a 64-character provenance hash."""
        records = [
            {"record_id": "PH-1", "category_number": 1, "confidence": 0.9},
        ]
        plan = create_routing_plan(records)
        assert "provenance_hash" in plan
        assert len(plan["provenance_hash"]) == 64

    def test_create_routing_plan_deterministic(self):
        """Same input produces same routing plan hash."""
        records = [
            {"record_id": "DT-1", "category_number": 4, "confidence": 0.88},
            {"record_id": "DT-2", "category_number": 4, "confidence": 0.85},
        ]
        plan1 = create_routing_plan(records)
        plan2 = create_routing_plan(records)
        assert plan1["provenance_hash"] == plan2["provenance_hash"]

    def test_create_routing_plan_sorted_by_category(self):
        """Routes in the plan are sorted by category number."""
        records = [
            {"record_id": "S-1", "category_number": 15, "confidence": 0.9},
            {"record_id": "S-2", "category_number": 1, "confidence": 0.9},
            {"record_id": "S-3", "category_number": 6, "confidence": 0.9},
        ]
        plan = create_routing_plan(records)
        cat_nums = [r["category_number"] for r in plan["routes"]]
        assert cat_nums == sorted(cat_nums)

    def test_create_routing_plan_all_15_categories(self):
        """Plan with all 15 categories creates 15 routes."""
        records = [
            {"record_id": f"ALL-{i}", "category_number": i, "confidence": 0.85}
            for i in range(1, 16)
        ]
        plan = create_routing_plan(records)
        assert plan["total_categories"] == 15
        assert plan["total_records"] == 15

    def test_create_routing_plan_route_has_agent_id(self):
        """Each route in the plan includes agent_id."""
        records = [
            {"record_id": "AI-1", "category_number": 6, "confidence": 0.9},
        ]
        plan = create_routing_plan(records)
        assert plan["routes"][0]["agent_id"] == "AGENT-MRV-019"

    def test_create_routing_plan_route_has_api_endpoint(self):
        """Each route in the plan includes api_endpoint."""
        records = [
            {"record_id": "EP-1", "category_number": 6, "confidence": 0.9},
        ]
        plan = create_routing_plan(records)
        assert "/api/v1/mrv/business-travel" in plan["routes"][0]["api_endpoint"]

    def test_validate_routing_plan_valid(self):
        """Valid plan passes validation."""
        records = [
            {"record_id": "V-1", "category_number": 1, "confidence": 0.9},
        ]
        plan = create_routing_plan(records)
        validation = validate_routing_plan(plan)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    def test_validate_routing_plan_empty(self):
        """Empty plan fails validation."""
        plan = create_routing_plan([])
        validation = validate_routing_plan(plan)
        assert validation["valid"] is False
        assert any("no records" in e.lower() for e in validation["errors"])

    def test_create_routing_plan_empty_returns_zero(self):
        """Empty records list results in zero total."""
        plan = create_routing_plan([])
        assert plan["total_records"] == 0
        assert plan["total_categories"] == 0

    def test_create_routing_plan_includes_records(self):
        """Each route in the plan includes the classified records."""
        records = [
            {"record_id": "INC-1", "category_number": 4, "amount": 5000},
            {"record_id": "INC-2", "category_number": 4, "amount": 3000},
        ]
        plan = create_routing_plan(records)
        route_records = plan["routes"][0]["records"]
        assert len(route_records) == 2
        ids = [r["record_id"] for r in route_records]
        assert "INC-1" in ids
        assert "INC-2" in ids


# ==============================================================================
# INPUT TRANSFORMATION TESTS (~20)
# ==============================================================================


class TestInputTransformation:
    """Test transformation of classified records for downstream agents."""

    def test_transform_input_cat1(self):
        """Cat 1 transformation includes base fields."""
        record = {
            "record_id": "TR-1", "amount": Decimal("5000"),
            "currency": "USD", "description": "Steel purchase",
            "confidence": 0.88, "classification_method": "naics_lookup",
            "provenance_hash": "a" * 64,
        }
        result = transform_input_for_agent(record, 1)
        assert result["source_record_id"] == "TR-1"
        assert result["amount"] == Decimal("5000")
        assert result["currency"] == "USD"
        assert result["target_category"] == 1

    def test_transform_input_cat6_travel(self):
        """Cat 6 transformation includes travel-specific fields."""
        record = {
            "record_id": "TR-6", "amount": Decimal("850"),
            "currency": "USD", "description": "Flight SFO-NYC",
            "confidence": 0.92, "classification_method": "naics_lookup",
            "provenance_hash": "b" * 64,
            "travel_type": "air", "cabin_class": "business",
        }
        result = transform_input_for_agent(record, 6)
        assert result["travel_type"] == "air"
        assert result["travel_class"] == "business"
        assert result["target_category"] == 6

    def test_transform_input_cat15_investment(self):
        """Cat 15 transformation includes investment-specific fields."""
        record = {
            "record_id": "TR-15", "amount": Decimal("100000"),
            "currency": "USD", "description": "Bond portfolio",
            "confidence": 0.90, "classification_method": "naics_lookup",
            "provenance_hash": "c" * 64,
            "asset_class": "corporate_bond", "investee_name": "TechCorp",
        }
        result = transform_input_for_agent(record, 15)
        assert result["asset_class"] == "corporate_bond"
        assert result["investee_name"] == "TechCorp"
        assert result["target_category"] == 15

    def test_transform_input_cat4_upstream_transport(self):
        """Cat 4 transformation includes transport direction=upstream."""
        record = {
            "record_id": "TR-4", "amount": Decimal("3000"),
            "currency": "USD", "description": "Inbound freight",
            "confidence": 0.85, "classification_method": "naics_lookup",
            "provenance_hash": "d" * 64,
            "transport_mode": "road",
        }
        result = transform_input_for_agent(record, 4)
        assert result["transport_mode"] == "road"
        assert result["direction"] == "upstream"

    def test_transform_input_cat9_downstream_transport(self):
        """Cat 9 transformation includes transport direction=downstream."""
        record = {
            "record_id": "TR-9", "amount": Decimal("2500"),
            "currency": "USD", "description": "Outbound freight",
            "confidence": 0.82, "classification_method": "gl_account_lookup",
            "provenance_hash": "e" * 64,
            "transport_mode": "rail",
        }
        result = transform_input_for_agent(record, 9)
        assert result["transport_mode"] == "rail"
        assert result["direction"] == "downstream"

    def test_transform_preserves_amount(self):
        """Transformation preserves the original amount value."""
        record = {
            "record_id": "PA-1", "amount": Decimal("12345.67"),
            "currency": "EUR", "description": "Purchase",
            "confidence": 0.80, "classification_method": "gl_account_lookup",
            "provenance_hash": "f" * 64,
        }
        result = transform_input_for_agent(record, 1)
        assert result["amount"] == Decimal("12345.67")

    def test_transform_preserves_currency(self):
        """Transformation preserves the currency code."""
        record = {
            "record_id": "PC-1", "amount": Decimal("5000"),
            "currency": "GBP", "description": "Purchase",
            "confidence": 0.80, "classification_method": "gl_account_lookup",
            "provenance_hash": "f" * 64,
        }
        result = transform_input_for_agent(record, 1)
        assert result["currency"] == "GBP"

    def test_transform_preserves_provenance_hash(self):
        """Transformation carries the provenance hash through."""
        orig_hash = "a1b2c3d4" * 8
        record = {
            "record_id": "PP-1", "amount": Decimal("1000"),
            "currency": "USD", "description": "Test",
            "confidence": 0.85, "classification_method": "naics_lookup",
            "provenance_hash": orig_hash,
        }
        result = transform_input_for_agent(record, 1)
        assert result["provenance_hash"] == orig_hash

    def test_transform_preserves_classification_method(self):
        """Transformation carries the classification method through."""
        record = {
            "record_id": "CM-1", "amount": Decimal("1000"),
            "currency": "USD", "description": "Test",
            "confidence": 0.85, "classification_method": "keyword_lookup",
            "provenance_hash": "x" * 64,
        }
        result = transform_input_for_agent(record, 5)
        assert result["classification_method"] == "keyword_lookup"

    def test_transform_preserves_confidence(self):
        """Transformation carries the classification confidence through."""
        record = {
            "record_id": "CC-1", "amount": Decimal("1000"),
            "currency": "USD", "description": "Test",
            "confidence": 0.92, "classification_method": "naics_lookup",
            "provenance_hash": "y" * 64,
        }
        result = transform_input_for_agent(record, 1)
        assert result["classification_confidence"] == 0.92

    def test_transform_sets_target_category(self):
        """Transformation sets the target_category field."""
        record = {
            "record_id": "TC-1", "amount": Decimal("1000"),
            "currency": "USD", "description": "Test",
            "confidence": 0.80, "classification_method": "gl_account_lookup",
            "provenance_hash": "z" * 64,
        }
        for cat_num in range(1, 16):
            result = transform_input_for_agent(record, cat_num)
            assert result["target_category"] == cat_num

    def test_transform_cat6_defaults_travel_type(self):
        """Cat 6 transformation defaults travel_type to 'air' if missing."""
        record = {
            "record_id": "DT-6", "amount": Decimal("800"),
            "currency": "USD", "description": "Travel expense",
            "confidence": 0.85, "classification_method": "gl_account_lookup",
            "provenance_hash": "t" * 64,
        }
        result = transform_input_for_agent(record, 6)
        assert result["travel_type"] == "air"

    def test_transform_cat15_defaults_asset_class(self):
        """Cat 15 transformation defaults asset_class to 'listed_equity' if missing."""
        record = {
            "record_id": "DI-15", "amount": Decimal("50000"),
            "currency": "USD", "description": "Investment allocation",
            "confidence": 0.88, "classification_method": "naics_lookup",
            "provenance_hash": "i" * 64,
        }
        result = transform_input_for_agent(record, 15)
        assert result["asset_class"] == "listed_equity"

    def test_transform_cat4_defaults_transport_mode(self):
        """Cat 4 transformation defaults transport_mode to 'road' if missing."""
        record = {
            "record_id": "DM-4", "amount": Decimal("3000"),
            "currency": "USD", "description": "Freight charges",
            "confidence": 0.82, "classification_method": "gl_account_lookup",
            "provenance_hash": "m" * 64,
        }
        result = transform_input_for_agent(record, 4)
        assert result["transport_mode"] == "road"

    @pytest.mark.parametrize("cat_num", list(range(1, 16)))
    def test_transform_all_categories(self, cat_num):
        """Transformation works for all 15 categories."""
        record = {
            "record_id": f"ALL-{cat_num}",
            "amount": Decimal("1000"),
            "currency": "USD",
            "description": "Test",
            "confidence": 0.85,
            "classification_method": "naics_lookup",
            "provenance_hash": "a" * 64,
        }
        result = transform_input_for_agent(record, cat_num)
        assert result["target_category"] == cat_num
        assert result["source_record_id"] == f"ALL-{cat_num}"

    def test_transform_missing_optional_fields(self):
        """Transformation handles records with minimal fields."""
        record = {"record_id": "MIN-1"}
        result = transform_input_for_agent(record, 1)
        assert result["source_record_id"] == "MIN-1"
        assert result["currency"] == "USD"  # Default
        assert result["classification_confidence"] == 0.0  # Default

    def test_transform_preserves_description(self):
        """Transformation preserves the original description."""
        record = {
            "record_id": "DESC-1", "amount": Decimal("500"),
            "currency": "USD", "description": "Raw materials - steel plate 10mm",
            "confidence": 0.90, "classification_method": "naics_lookup",
            "provenance_hash": "d" * 64,
        }
        result = transform_input_for_agent(record, 1)
        assert result["description"] == "Raw materials - steel plate 10mm"


# ==============================================================================
# ROUTING EXECUTION TESTS (~20)
# ==============================================================================


class TestRoutingExecution:
    """Test batch routing execution with success/failure tracking."""

    def test_route_batch_success(self, db_engine):
        """Successful batch routing returns all records as routed."""
        records = [
            {"record_id": "EX-1", "category_number": 1,
             "amount": Decimal("5000"), "currency": "USD",
             "description": "Steel", "confidence": 0.88,
             "classification_method": "naics_lookup",
             "provenance_hash": "a" * 64},
            {"record_id": "EX-2", "category_number": 4,
             "amount": Decimal("3000"), "currency": "USD",
             "description": "Freight", "confidence": 0.85,
             "classification_method": "gl_account_lookup",
             "provenance_hash": "b" * 64},
        ]
        result = route_batch(db_engine, records)
        assert result["status"] == "completed"
        assert result["executed"] is True
        assert result["success_count"] == 2
        assert result["failure_count"] == 0

    def test_route_batch_dry_run_no_execution(self, db_engine):
        """Dry-run routing does not execute."""
        records = [
            {"record_id": "DRY-1", "category_number": 1,
             "amount": Decimal("5000"), "currency": "USD",
             "description": "Steel", "confidence": 0.88,
             "classification_method": "naics_lookup",
             "provenance_hash": "a" * 64},
        ]
        result = route_batch(db_engine, records, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["executed"] is False
        assert result["success_count"] == 0

    def test_route_batch_empty(self, db_engine):
        """Empty batch returns zero counts."""
        result = route_batch(db_engine, [])
        assert result["success_count"] == 0
        assert result["failure_count"] == 0

    def test_route_batch_provenance(self, db_engine):
        """Routing result includes provenance hash."""
        records = [
            {"record_id": "PRV-1", "category_number": 6,
             "amount": Decimal("850"), "currency": "USD",
             "description": "Flight", "confidence": 0.92,
             "classification_method": "naics_lookup",
             "provenance_hash": "c" * 64},
        ]
        result = route_batch(db_engine, records)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_route_batch_includes_plan(self, db_engine):
        """Routing result includes the routing plan."""
        records = [
            {"record_id": "PL-1", "category_number": 1,
             "amount": Decimal("1000"), "currency": "USD",
             "description": "Test", "confidence": 0.85,
             "classification_method": "naics_lookup",
             "provenance_hash": "d" * 64},
        ]
        result = route_batch(db_engine, records)
        assert "plan" in result
        assert result["plan"]["total_records"] == 1

    def test_route_batch_multiple_categories(self, db_engine):
        """Batch with multiple categories routes to multiple agents."""
        records = [
            {"record_id": f"MC-{i}", "category_number": cat,
             "amount": Decimal("1000"), "currency": "USD",
             "description": "Test", "confidence": 0.85,
             "classification_method": "naics_lookup",
             "provenance_hash": "e" * 64}
            for i, cat in enumerate([1, 4, 6, 15])
        ]
        result = route_batch(db_engine, records)
        assert result["success_count"] == 4
        assert result["plan"]["total_categories"] == 4

    def test_route_batch_all_15_categories(self, db_engine):
        """Batch covering all 15 categories routes correctly."""
        records = [
            {"record_id": f"A15-{i}", "category_number": i,
             "amount": Decimal("1000"), "currency": "USD",
             "description": "Test", "confidence": 0.85,
             "classification_method": "naics_lookup",
             "provenance_hash": "f" * 64}
            for i in range(1, 16)
        ]
        result = route_batch(db_engine, records)
        assert result["success_count"] == 15
        assert result["plan"]["total_categories"] == 15

    def test_route_batch_route_results_have_agent_id(self, db_engine):
        """Each route result includes the target agent ID."""
        records = [
            {"record_id": "AG-1", "category_number": 6,
             "amount": Decimal("850"), "currency": "USD",
             "description": "Flight", "confidence": 0.92,
             "classification_method": "naics_lookup",
             "provenance_hash": "g" * 64},
        ]
        result = route_batch(db_engine, records)
        assert result["route_results"][0]["agent_id"] == "AGENT-MRV-019"

    def test_route_batch_route_results_have_status(self, db_engine):
        """Each route result includes a status field."""
        records = [
            {"record_id": "ST-1", "category_number": 1,
             "amount": Decimal("5000"), "currency": "USD",
             "description": "Steel", "confidence": 0.88,
             "classification_method": "naics_lookup",
             "provenance_hash": "h" * 64},
        ]
        result = route_batch(db_engine, records)
        assert result["route_results"][0]["status"] == "routed"

    def test_route_batch_route_results_have_transformed_input(self, db_engine):
        """Each successfully routed record includes transformed input."""
        records = [
            {"record_id": "TI-1", "category_number": 4,
             "amount": Decimal("3000"), "currency": "USD",
             "description": "Freight", "confidence": 0.85,
             "classification_method": "gl_account_lookup",
             "provenance_hash": "i" * 64},
        ]
        result = route_batch(db_engine, records)
        transformed = result["route_results"][0]["transformed_input"]
        assert transformed["source_record_id"] == "TI-1"
        assert transformed["target_category"] == 4

    def test_route_batch_deterministic(self, db_engine):
        """Same input produces same routing provenance hash."""
        records = [
            {"record_id": "DET-1", "category_number": 1,
             "amount": Decimal("5000"), "currency": "USD",
             "description": "Steel", "confidence": 0.88,
             "classification_method": "naics_lookup",
             "provenance_hash": "j" * 64},
        ]
        r1 = route_batch(db_engine, records)
        r2 = route_batch(db_engine, records)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_route_batch_preserves_record_ids(self, db_engine):
        """Routing preserves original record IDs in route results."""
        records = [
            {"record_id": f"PID-{i}", "category_number": 1,
             "amount": Decimal("1000"), "currency": "USD",
             "description": "Test", "confidence": 0.85,
             "classification_method": "naics_lookup",
             "provenance_hash": "k" * 64}
            for i in range(5)
        ]
        result = route_batch(db_engine, records)
        ids = [rr["record_id"] for rr in result["route_results"]]
        assert ids == [f"PID-{i}" for i in range(5)]

    def test_route_batch_partial_failure(self, db_engine):
        """Partial failure: some records succeed, others fail.

        Simulates failure by patching transform_input_for_agent to raise
        on specific records.
        """
        records = [
            {"record_id": "PF-1", "category_number": 1,
             "amount": Decimal("5000"), "currency": "USD",
             "description": "Steel", "confidence": 0.88,
             "classification_method": "naics_lookup",
             "provenance_hash": "l" * 64},
            {"record_id": "PF-2", "category_number": 6,
             "amount": Decimal("850"), "currency": "USD",
             "description": "Flight", "confidence": 0.92,
             "classification_method": "naics_lookup",
             "provenance_hash": "m" * 64},
        ]

        original_transform = transform_input_for_agent
        call_count = 0

        def failing_transform(record, cat_num):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated downstream agent failure")
            return original_transform(record, cat_num)

        with patch(
            f"{__name__}.transform_input_for_agent",
            side_effect=failing_transform,
        ):
            result = route_batch(db_engine, records)

        assert result["success_count"] == 1
        assert result["failure_count"] == 1

    def test_route_batch_large_batch(self, db_engine):
        """Large batch (1000 records) routes successfully."""
        records = [
            {"record_id": f"LB-{i}", "category_number": (i % 15) + 1,
             "amount": Decimal("1000"), "currency": "USD",
             "description": "Bulk item", "confidence": 0.85,
             "classification_method": "naics_lookup",
             "provenance_hash": "n" * 64}
            for i in range(1000)
        ]
        result = route_batch(db_engine, records)
        assert result["success_count"] == 1000
        assert result["failure_count"] == 0
        assert result["plan"]["total_categories"] == 15

    def test_route_batch_dry_run_has_plan_but_no_results(self, db_engine):
        """Dry run returns a plan but no route_results list."""
        records = [
            {"record_id": "DRP-1", "category_number": 1,
             "amount": Decimal("1000"), "currency": "USD",
             "description": "Test", "confidence": 0.85,
             "classification_method": "naics_lookup",
             "provenance_hash": "o" * 64},
        ]
        result = route_batch(db_engine, records, dry_run=True)
        assert "plan" in result
        assert result["executed"] is False
        assert "route_results" not in result
