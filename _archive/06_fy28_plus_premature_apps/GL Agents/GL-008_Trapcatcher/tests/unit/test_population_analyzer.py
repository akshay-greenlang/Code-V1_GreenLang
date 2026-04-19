"""
Unit Tests: Population Analyzer
Tests for fleet health metrics, failure trends, and priority ranking.
Author: GL-TestEngineer
"""
import pytest
import hashlib
import json
from typing import Dict, List, Any
from conftest import TrapType, TrapFailureMode, MaintenancePriority, MockTrapData


class PopulationAnalyzer:
    def __init__(self):
        self.failure_threshold = 0.15

    def calculate_fleet_health(self, traps: List[MockTrapData], diagnoses: List[Dict]) -> Dict[str, Any]:
        total = len(traps)
        if total == 0: return {"total": 0, "healthy": 0, "failed": 0, "failure_rate": 0.0}
        failed = sum(1 for d in diagnoses if d["failure_mode"] != "NORMAL")
        return {"total": total, "healthy": total - failed, "failed": failed, "failure_rate": failed / total * 100}

    def analyze_failure_trends(self, historical_data: List[Dict]) -> Dict[str, Any]:
        if not historical_data: return {"trend": "stable", "slope": 0.0}
        rates = [d.get("failure_rate", 0) for d in historical_data]
        if len(rates) < 2: return {"trend": "stable", "slope": 0.0}
        slope = (rates[-1] - rates[0]) / len(rates)
        trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        return {"trend": trend, "slope": slope}

    def prioritize_traps(self, diagnoses: List[Dict]) -> List[Dict]:
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
        return sorted(diagnoses, key=lambda x: (priority_order.get(x.get("priority", "NONE"), 4), -x.get("energy_loss_kw", 0)))

    def pareto_analysis(self, diagnoses: List[Dict]) -> Dict[str, Any]:
        if not diagnoses: return {"top_20_percent": [], "contribution": 0.0}
        sorted_diag = sorted(diagnoses, key=lambda x: x.get("annual_cost_usd", 0), reverse=True)
        top_count = max(1, len(sorted_diag) // 5)
        top_traps = sorted_diag[:top_count]
        total_cost = sum(d.get("annual_cost_usd", 0) for d in diagnoses)
        top_cost = sum(d.get("annual_cost_usd", 0) for d in top_traps)
        contribution = (top_cost / total_cost * 100) if total_cost > 0 else 0.0
        return {"top_20_percent": [t["trap_id"] for t in top_traps], "contribution": contribution}

    def optimize_survey_frequency(self, fleet_health: Dict) -> str:
        rate = fleet_health.get("failure_rate", 0)
        if rate > 20: return "monthly"
        elif rate > 10: return "quarterly"
        else: return "semi-annual"


@pytest.fixture
def analyzer(): return PopulationAnalyzer()


class TestFleetHealthMetrics:
    def test_healthy_fleet(self, analyzer, trap_fleet):
        diagnoses = [{"trap_id": t.trap_id, "failure_mode": "NORMAL"} for t in trap_fleet[:3]]
        diagnoses += [{"trap_id": t.trap_id, "failure_mode": "FAILED_OPEN"} for t in trap_fleet[3:]]
        result = analyzer.calculate_fleet_health(trap_fleet, diagnoses)
        assert result["total"] == len(trap_fleet)
        assert result["healthy"] >= 0

    def test_empty_fleet(self, analyzer):
        result = analyzer.calculate_fleet_health([], [])
        assert result["total"] == 0


class TestFailureTrendAnalysis:
    def test_increasing_trend(self, analyzer):
        data = [{"failure_rate": 5}, {"failure_rate": 8}, {"failure_rate": 12}, {"failure_rate": 18}]
        result = analyzer.analyze_failure_trends(data)
        assert result["trend"] == "increasing"

    def test_stable_trend(self, analyzer):
        data = [{"failure_rate": 10}, {"failure_rate": 10}, {"failure_rate": 10}]
        result = analyzer.analyze_failure_trends(data)
        assert result["trend"] == "stable"


class TestPriorityRanking:
    def test_priority_order(self, analyzer):
        diagnoses = [
            {"trap_id": "T1", "priority": "LOW", "energy_loss_kw": 10},
            {"trap_id": "T2", "priority": "CRITICAL", "energy_loss_kw": 100},
            {"trap_id": "T3", "priority": "HIGH", "energy_loss_kw": 50},
        ]
        result = analyzer.prioritize_traps(diagnoses)
        assert result[0]["priority"] == "CRITICAL"


class TestParetoAnalysis:
    def test_pareto(self, analyzer):
        diagnoses = [{"trap_id": f"T{i}", "annual_cost_usd": (10-i)*100} for i in range(10)]
        result = analyzer.pareto_analysis(diagnoses)
        assert "top_20_percent" in result
        assert result["contribution"] > 0


class TestSurveyFrequency:
    def test_high_failure_rate(self, analyzer):
        assert analyzer.optimize_survey_frequency({"failure_rate": 25}) == "monthly"

    def test_low_failure_rate(self, analyzer):
        assert analyzer.optimize_survey_frequency({"failure_rate": 5}) == "semi-annual"
