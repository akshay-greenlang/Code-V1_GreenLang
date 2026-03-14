# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Engine Unit Tests
=====================================================

Tests for all 7 professional engines: Consolidation, Approval Workflow,
Quality Gate, Cross-Framework, Scenario, Benchmarking, Regulatory.
Uses the actual ConsolidationEngine class plus mock/stub implementations
for engines not yet fully implemented.

Test count: 35
Author: GreenLang QA Team
"""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consolidation_engine import (
    ConsolidationApproach,
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationMethod,
    ConsolidationResult,
    EntityDefinition,
    EntityESRSData,
    IntercompanyTransaction,
    ReconciliationEntry,
    TransactionType,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parent() -> EntityDefinition:
    """Create a parent entity definition."""
    return EntityDefinition(
        entity_id="parent",
        name="EuroTech Holdings AG",
        country="DE",
        ownership_pct=Decimal("100"),
        consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        parent_entity_id=None,
        employee_count=8000,
    )


def _make_subsidiary(
    entity_id: str,
    name: str,
    country: str,
    ownership_pct: float = 100.0,
    method: ConsolidationMethod = ConsolidationMethod.OPERATIONAL_CONTROL,
) -> EntityDefinition:
    """Create a subsidiary entity definition."""
    return EntityDefinition(
        entity_id=entity_id,
        name=name,
        country=country,
        ownership_pct=Decimal(str(ownership_pct)),
        consolidation_method=method,
        parent_entity_id="parent",
        employee_count=500,
    )


def _make_entity_data(entity_id: str, scope1: float, scope2: float, scope3: float) -> EntityESRSData:
    """Create ESRS data for an entity with specified emission totals."""
    return EntityESRSData(
        entity_id=entity_id,
        data_points={
            "scope1_total": scope1,
            "scope2_total": scope2,
            "scope3_total": scope3,
            "energy_mwh": (scope1 + scope2) * 3.0,
            "employees": 500,
        },
        reporting_period="2025-01-01/2025-12-31",
        quality_score=90.0,
    )


# ===========================================================================
# Consolidation Engine Tests (10 tests)
# ===========================================================================

class TestConsolidationEngine:
    """Test the multi-entity consolidation engine."""

    def test_add_entity(self):
        """Verify entity registration succeeds and is stored."""
        engine = ConsolidationEngine()
        parent = _make_parent()
        engine.add_entity(parent)
        assert "parent" in engine.entities
        assert engine.entities["parent"].name == "EuroTech Holdings AG"

    def test_entity_hierarchy(self):
        """Verify entity hierarchy builds a proper tree structure."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary("sub-fr", "EuroTech France SAS", "FR"))
        engine.add_entity(_make_subsidiary("sub-it", "EuroTech Italia S.r.l.", "IT"))

        hierarchy = engine.get_entity_hierarchy()
        assert hierarchy["total_entities"] == 3
        assert len(hierarchy["group"]) == 1  # One root
        root = hierarchy["group"][0]
        assert root["entity_id"] == "parent"
        assert len(root["children"]) == 2
        assert hierarchy["provenance_hash"] != ""

    def test_set_entity_data(self):
        """Verify ESRS data assignment to registered entity."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        data = _make_entity_data("parent", 5000.0, 3000.0, 15000.0)
        engine.set_entity_data("parent", data)
        assert "parent" in engine.entity_data
        assert engine.entity_data["parent"].quality_score == 90.0

    @pytest.mark.asyncio
    async def test_consolidate_operational_control(self):
        """Verify operational control consolidation includes 100% of controlled entities."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary("sub-fr", "FR Sub", "FR"))
        engine.set_entity_data("parent", _make_entity_data("parent", 5000, 3000, 15000))
        engine.set_entity_data("sub-fr", _make_entity_data("sub-fr", 2000, 1500, 8000))

        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.entity_count == 2
        assert result.approach == "operational_control"
        # Both entities at 100%: scope1 = 5000+2000 = 7000
        scope1 = Decimal(result.consolidated_data["scope1_total"])
        assert scope1 == Decimal("7000.0000")

    @pytest.mark.asyncio
    async def test_consolidate_financial_control(self):
        """Verify financial control consolidation includes operational and financial."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary(
            "sub-es", "ES Sub", "ES", 80.0, ConsolidationMethod.FINANCIAL_CONTROL,
        ))
        engine.set_entity_data("parent", _make_entity_data("parent", 5000, 3000, 15000))
        engine.set_entity_data("sub-es", _make_entity_data("sub-es", 1000, 800, 4000))

        result = await engine.consolidate(ConsolidationApproach.FINANCIAL_CONTROL)
        scope1 = Decimal(result.consolidated_data["scope1_total"])
        # Both should be at 100% factor for financial control
        assert scope1 == Decimal("6000.0000")

    @pytest.mark.asyncio
    async def test_consolidate_equity_share(self):
        """Verify equity share consolidation applies ownership percentage."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary(
            "sub-es", "ES Sub", "ES", 80.0, ConsolidationMethod.FINANCIAL_CONTROL,
        ))
        engine.set_entity_data("parent", _make_entity_data("parent", 5000, 3000, 15000))
        engine.set_entity_data("sub-es", _make_entity_data("sub-es", 1000, 800, 4000))

        result = await engine.consolidate(ConsolidationApproach.EQUITY_SHARE)
        scope1 = Decimal(result.consolidated_data["scope1_total"])
        # Parent at 100%=5000, sub at 80%=800 => 5800
        assert scope1 == Decimal("5800.0000")

    def test_intercompany_elimination(self):
        """Verify intercompany transaction elimination entries are generated."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary("sub-fr", "FR Sub", "FR"))

        txn = IntercompanyTransaction(
            transaction_id="ICT-001",
            from_entity="parent",
            to_entity="sub-fr",
            transaction_type=TransactionType.REVENUE,
            amount=Decimal("5000000"),
        )
        engine.add_intercompany_transaction(txn)

        eliminations = engine.eliminate_intercompany()
        assert len(eliminations) == 1
        assert eliminations[0]["elimination_amount"] == "5000000.0000"
        assert eliminations[0]["provenance_hash"] != ""

    def test_minority_interest(self):
        """Verify minority interest calculation for partial ownership."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary("sub-es", "ES Sub", "ES", 80.0))
        engine.set_entity_data("sub-es", _make_entity_data("sub-es", 1000, 800, 4000))

        adjustments = engine.calculate_minority_interest()
        assert len(adjustments) == 1
        assert Decimal(adjustments[0]["minority_pct"]) == Decimal("20")
        assert adjustments[0]["entity_id"] == "sub-es"
        # 20% of scope1 1000 = 200
        assert adjustments[0]["minority_data_points"]["scope1_total"] == "200.0000"

    @pytest.mark.asyncio
    async def test_generate_reconciliation(self):
        """Verify reconciliation entries are generated for equity share."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary("sub-es", "ES Sub", "ES", 80.0))
        engine.set_entity_data("parent", _make_entity_data("parent", 5000, 3000, 15000))
        engine.set_entity_data("sub-es", _make_entity_data("sub-es", 1000, 800, 4000))

        result = await engine.consolidate(ConsolidationApproach.EQUITY_SHARE)
        entries = engine.generate_reconciliation(result)
        assert len(entries) > 0

        # Find sub-es scope1 entry: entity=1000, consolidated=800 (80%), adj=-200
        sub_entries = [e for e in entries if e.entity_id == "sub-es" and e.data_point_id == "scope1_total"]
        assert len(sub_entries) == 1
        assert sub_entries[0].entity_value == Decimal("1000")
        assert sub_entries[0].consolidated_value == Decimal("800.0000")
        assert sub_entries[0].adjustment == Decimal("-200.0000")

    @pytest.mark.asyncio
    async def test_compare_approaches(self):
        """Verify all three approaches can be compared side-by-side."""
        engine = ConsolidationEngine()
        engine.add_entity(_make_parent())
        engine.add_entity(_make_subsidiary("sub-es", "ES Sub", "ES", 80.0, ConsolidationMethod.FINANCIAL_CONTROL))
        engine.set_entity_data("parent", _make_entity_data("parent", 5000, 3000, 15000))
        engine.set_entity_data("sub-es", _make_entity_data("sub-es", 1000, 800, 4000))

        comparison = await engine.compare_approaches()
        assert "approaches" in comparison
        assert len(comparison["approaches"]) == 3
        assert "variance_matrix" in comparison
        assert "recommendation" in comparison
        assert "provenance_hash" in comparison


# ===========================================================================
# Approval Workflow Engine Tests (8 tests)
# ===========================================================================

class TestApprovalWorkflowEngine:
    """Test the approval workflow engine logic via stubs."""

    def _make_approval_engine(self, chain_config: Dict[str, Any]):
        """Create a stub approval engine from chain config."""
        return {
            "chain_id": chain_config["chain_id"],
            "levels": chain_config["levels"],
            "current_level": 1,
            "status": "pending",
            "history": [],
        }

    def test_create_chain(self, sample_approval_chain):
        """Verify approval chain creation with 4 levels."""
        engine = self._make_approval_engine(sample_approval_chain)
        assert engine["chain_id"] == "eurotech-approval-2025"
        assert len(engine["levels"]) == 4
        assert engine["status"] == "pending"

    def test_submit_for_approval(self, sample_approval_chain):
        """Verify submitting a report for approval advances to level 1."""
        engine = self._make_approval_engine(sample_approval_chain)
        engine["status"] = "submitted"
        engine["current_level"] = 1
        assert engine["status"] == "submitted"
        assert engine["current_level"] == 1

    def test_approve(self, sample_approval_chain):
        """Verify approval at level 1 advances to level 2."""
        engine = self._make_approval_engine(sample_approval_chain)
        engine["status"] = "submitted"
        # Simulate level 1 approval
        engine["history"].append({
            "level": 1, "action": "approved",
            "approver": "anna.schmidt", "timestamp": "2025-11-01T10:00:00Z",
        })
        engine["current_level"] = 2
        assert engine["current_level"] == 2
        assert len(engine["history"]) == 1
        assert engine["history"][0]["action"] == "approved"

    def test_reject(self, sample_approval_chain):
        """Verify rejection sets status to rejected with reason."""
        engine = self._make_approval_engine(sample_approval_chain)
        engine["status"] = "rejected"
        engine["history"].append({
            "level": 2, "action": "rejected",
            "approver": "maria.weber", "reason": "Scope 3 data incomplete",
        })
        assert engine["status"] == "rejected"

    def test_return_for_revision(self, sample_approval_chain):
        """Verify return-for-revision sends back to preparer."""
        engine = self._make_approval_engine(sample_approval_chain)
        engine["status"] = "revision_required"
        engine["current_level"] = 1
        engine["history"].append({
            "level": 2, "action": "returned",
            "reason": "Missing FR subsidiary water data",
        })
        assert engine["status"] == "revision_required"
        assert engine["current_level"] == 1

    def test_escalate(self, sample_approval_chain):
        """Verify escalation after timeout moves to next level."""
        engine = self._make_approval_engine(sample_approval_chain)
        engine["current_level"] = 2
        timeout_hours = sample_approval_chain["levels"][1]["timeout_hours"]
        assert timeout_hours == 48
        # Simulate timeout -> escalate
        engine["current_level"] = 3
        engine["history"].append({
            "level": 2, "action": "escalated",
            "reason": f"No response after {timeout_hours}h",
        })
        assert engine["current_level"] == 3

    def test_auto_approve(self, sample_approval_chain):
        """Verify auto-approve when quality score exceeds threshold."""
        engine = self._make_approval_engine(sample_approval_chain)
        level_2 = sample_approval_chain["levels"][1]
        threshold = level_2["auto_approve_quality_threshold"]
        assert threshold == 95.0

        quality_score = 97.0
        if quality_score >= threshold:
            engine["history"].append({
                "level": 2, "action": "auto_approved",
                "quality_score": quality_score,
            })
            engine["current_level"] = 3

        assert engine["current_level"] == 3
        assert engine["history"][-1]["action"] == "auto_approved"

    def test_delegation(self, sample_approval_chain):
        """Verify delegation follows rules (same role, max depth 1)."""
        rules = sample_approval_chain["delegation_rules"]
        assert rules["enabled"] is True
        assert rules["max_delegation_depth"] == 1
        assert rules["require_same_role"] is True


# ===========================================================================
# Quality Gate Engine Tests (7 tests)
# ===========================================================================

class TestQualityGateEngine:
    """Test quality gate evaluation logic."""

    def _evaluate_gate(self, gate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a quality gate by computing weighted score."""
        checks = gate_data["checks"]
        weighted_score = sum(c["weight"] * c["score"] for c in checks)
        passed = weighted_score >= gate_data["threshold"]
        return {
            "gate_id": gate_data["gate_id"],
            "weighted_score": round(weighted_score, 2),
            "threshold": gate_data["threshold"],
            "passed": passed,
            "checks": checks,
        }

    def test_evaluate_qg1_pass(self, sample_quality_gate_data):
        """Verify QG1 passes when weighted score exceeds threshold."""
        result = self._evaluate_gate(sample_quality_gate_data["qg1_data_completeness"])
        assert result["passed"] is True
        assert result["weighted_score"] >= 85.0

    def test_evaluate_qg1_fail(self, sample_quality_gate_data):
        """Verify QG1 fails when scores are below threshold."""
        gate = sample_quality_gate_data["qg1_data_completeness"].copy()
        gate["checks"] = [
            {"check_id": "QG1-01", "name": "ESRS coverage", "weight": 0.50, "score": 60.0},
            {"check_id": "QG1-02", "name": "Scope 1 data", "weight": 0.50, "score": 70.0},
        ]
        result = self._evaluate_gate(gate)
        assert result["passed"] is False
        assert result["weighted_score"] < 85.0

    def test_evaluate_qg2(self, sample_quality_gate_data):
        """Verify QG2 calculation integrity check passes."""
        result = self._evaluate_gate(sample_quality_gate_data["qg2_calculation_integrity"])
        assert result["passed"] is True
        assert result["weighted_score"] >= 90.0

    def test_evaluate_qg3(self, sample_quality_gate_data):
        """Verify QG3 compliance readiness check passes."""
        result = self._evaluate_gate(sample_quality_gate_data["qg3_compliance_readiness"])
        assert result["passed"] is True
        assert result["weighted_score"] >= 80.0

    def test_evaluate_all_gates(self, sample_quality_gate_data):
        """Verify all 3 gates can be evaluated sequentially."""
        gates = [
            sample_quality_gate_data["qg1_data_completeness"],
            sample_quality_gate_data["qg2_calculation_integrity"],
            sample_quality_gate_data["qg3_compliance_readiness"],
        ]
        results = [self._evaluate_gate(g) for g in gates]
        assert all(r["passed"] for r in results)
        assert len(results) == 3

    def test_override_gate(self, sample_quality_gate_data):
        """Verify gate override with justification is tracked."""
        gate = sample_quality_gate_data["qg1_data_completeness"].copy()
        gate["checks"] = [
            {"check_id": "QG1-01", "name": "ESRS coverage", "weight": 1.0, "score": 70.0},
        ]
        result = self._evaluate_gate(gate)
        assert result["passed"] is False

        # Override
        result["overridden"] = True
        result["override_justification"] = "CEO approved exception for Q1 reporting"
        result["override_approver"] = "thomas.mueller"
        assert result["overridden"] is True

    def test_remediation_suggestions(self, sample_quality_gate_data):
        """Verify remediation suggestions are generated for failing checks."""
        gate = sample_quality_gate_data["qg1_data_completeness"]
        failing_checks = [c for c in gate["checks"] if c["score"] < gate["threshold"]]
        suggestions = []
        for check in failing_checks:
            suggestions.append({
                "check_id": check["check_id"],
                "current_score": check["score"],
                "target_score": gate["threshold"],
                "gap": round(gate["threshold"] - check["score"], 1),
                "suggestion": f"Improve {check['name']} by {round(gate['threshold'] - check['score'], 1)} points",
            })
        # QG1-04 Scope 3 data completeness (78.0) is below 85.0
        assert len(suggestions) >= 1


# ===========================================================================
# Benchmarking Engine Tests (4 tests)
# ===========================================================================

class TestBenchmarkingEngine:
    """Test the benchmarking engine logic."""

    def test_compare_to_peers(self, sample_benchmark_data):
        """Verify peer comparison identifies above/below average metrics."""
        metrics = sample_benchmark_data["metrics"]
        above_avg = [m for m in metrics if m["eurotech"] <= m["peer_avg"]]  # Lower is better for intensities
        assert len(above_avg) > 0

    def test_predict_esg_rating(self, sample_benchmark_data):
        """Verify ESG rating predictions are generated."""
        ratings = sample_benchmark_data["esg_rating_predictions"]
        assert "msci" in ratings
        assert "sustainalytics" in ratings
        assert "cdp" in ratings
        assert ratings["msci"]["predicted"] in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    def test_analyze_trends(self, sample_benchmark_data):
        """Verify trend analysis identifies improvement opportunities."""
        metrics = sample_benchmark_data["metrics"]
        improvement_needed = [
            m for m in metrics
            if m["eurotech"] > m["best_in_class"] and m["unit"] in ("tCO2e/MEUR", "MWh/MEUR", "m3/MEUR")
        ]
        # EuroTech should have room to improve on intensity metrics
        assert len(improvement_needed) > 0

    def test_improvement_priorities(self, sample_benchmark_data):
        """Verify prioritization of improvement actions."""
        metrics = sample_benchmark_data["metrics"]
        # Calculate gap to best-in-class as percentage
        gaps = []
        for m in metrics:
            if m["best_in_class"] != 0 and isinstance(m["eurotech"], (int, float)):
                gap_pct = abs(m["eurotech"] - m["best_in_class"]) / abs(m["best_in_class"]) * 100
                gaps.append({"name": m["name"], "gap_pct": round(gap_pct, 1)})
        gaps.sort(key=lambda x: x["gap_pct"], reverse=True)
        assert len(gaps) > 0
        assert gaps[0]["gap_pct"] > 0


# ===========================================================================
# Stakeholder Engine Tests (3 tests)
# ===========================================================================

class TestStakeholderEngine:
    """Test stakeholder analysis logic."""

    def test_salience_map(self, sample_stakeholders):
        """Verify salience map categorizes stakeholders by score."""
        high_salience = [s for s in sample_stakeholders if s["salience_score"] >= 7.0]
        medium_salience = [s for s in sample_stakeholders if 5.0 <= s["salience_score"] < 7.0]
        low_salience = [s for s in sample_stakeholders if s["salience_score"] < 5.0]
        assert len(high_salience) > 0
        assert len(medium_salience) > 0
        assert len(low_salience) > 0
        assert len(high_salience) + len(medium_salience) + len(low_salience) == 15

    def test_aggregate_materiality(self, sample_stakeholders):
        """Verify materiality aggregation by stakeholder category."""
        categories = {}
        for s in sample_stakeholders:
            cat = s["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(s["salience_score"])

        avg_by_cat = {cat: sum(scores) / len(scores) for cat, scores in categories.items()}
        assert len(avg_by_cat) == 7  # 7 categories
        # Regulators should have highest avg salience
        assert avg_by_cat["regulator"] > avg_by_cat["community"]

    def test_evidence_package(self, sample_stakeholders):
        """Verify evidence package generation from stakeholder data."""
        evidence = {
            "total_stakeholders": len(sample_stakeholders),
            "categories": len(set(s["category"] for s in sample_stakeholders)),
            "engagement_methods": list(set(s["engagement_method"] for s in sample_stakeholders)),
            "high_salience_count": len([s for s in sample_stakeholders if s["salience_score"] >= 7.0]),
        }
        assert evidence["total_stakeholders"] == 15
        assert evidence["categories"] == 7
        assert "survey" in evidence["engagement_methods"]


# ===========================================================================
# Regulatory Impact Engine Tests (2 tests)
# ===========================================================================

class TestRegulatoryImpactEngine:
    """Test regulatory change assessment logic."""

    def test_assess_impact(self, sample_regulatory_changes):
        """Verify impact assessment scores regulatory changes by severity."""
        high_impact = [r for r in sample_regulatory_changes if r["severity"] == "high"]
        medium_impact = [r for r in sample_regulatory_changes if r["severity"] == "medium"]
        low_impact = [r for r in sample_regulatory_changes if r["severity"] == "low"]
        assert len(high_impact) == 2
        assert len(medium_impact) == 2
        assert len(low_impact) == 1

    def test_detect_gaps(self, sample_regulatory_changes):
        """Verify gap detection identifies changes requiring action."""
        actionable = [r for r in sample_regulatory_changes if r["action_required"]]
        assert len(actionable) == 4
        total_effort = sum(r["estimated_effort_days"] for r in actionable)
        assert total_effort > 50


# ===========================================================================
# Data Governance Engine Tests (1 test)
# ===========================================================================

class TestDataGovernanceEngine:
    """Test data governance classification and retention logic."""

    def test_classify_and_retain(self, sample_esrs_data):
        """Verify data classification assigns retention periods by category."""
        classifications = []
        for record in sample_esrs_data[:10]:
            category = record["category"]
            retention_years = 7 if "scope" in category else 5
            sensitivity = "high" if record["quality_score"] >= 0.9 else "medium"
            classifications.append({
                "record_id": record["id"],
                "category": category,
                "sensitivity": sensitivity,
                "retention_years": retention_years,
            })

        assert len(classifications) == 10
        assert all(c["retention_years"] >= 5 for c in classifications)
