"""
Unit tests for GL-EUDR-APP v1.0 Risk Aggregation Engine.

Tests risk score computation, country risk lookups, risk heatmap
generation, alert management, trend tracking, and mitigation
recommendation generation.

Test count target: 25+ tests
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Risk Aggregation Engine (self-contained for testing)
# ---------------------------------------------------------------------------

HIGH_RISK_COUNTRIES = {
    "BRA", "IDN", "COD", "COG", "CMR", "MYS", "PNG", "BOL", "PER", "COL"
}
LOW_RISK_COUNTRIES = {"DEU", "SWE", "NOR", "FIN", "AUT", "FRA", "NLD", "JPN"}

RISK_WEIGHTS = {
    "satellite": 0.35,
    "country": 0.25,
    "supplier": 0.20,
    "document": 0.20,
}

RISK_LEVELS = {
    "critical": (0.9, 1.0),
    "high": (0.7, 0.9),
    "standard": (0.4, 0.7),
    "low": (0.0, 0.4),
}


class RiskAggregator:
    """Aggregates risk from multiple sources for EUDR compliance."""

    def __init__(self):
        self._assessments: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []

    def assess_risk(self, satellite_risk: float, country_risk: float,
                    supplier_risk: float, document_risk: float,
                    plot_id: Optional[str] = None,
                    supplier_id: Optional[str] = None) -> Dict[str, Any]:
        """Combine four risk sources into an overall risk assessment."""
        for name, val in [("satellite", satellite_risk), ("country", country_risk),
                          ("supplier", supplier_risk), ("document", document_risk)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name}_risk must be in [0, 1], got {val}")

        overall = (
            satellite_risk * RISK_WEIGHTS["satellite"] +
            country_risk * RISK_WEIGHTS["country"] +
            supplier_risk * RISK_WEIGHTS["supplier"] +
            document_risk * RISK_WEIGHTS["document"]
        )
        overall = round(min(1.0, max(0.0, overall)), 4)

        level = "low"
        for lvl, (lo, hi) in RISK_LEVELS.items():
            if lo <= overall < hi or (lvl == "critical" and overall >= hi - 0.1):
                level = lvl

        # More precise level determination
        if overall >= 0.9:
            level = "critical"
        elif overall >= 0.7:
            level = "high"
        elif overall >= 0.4:
            level = "standard"
        else:
            level = "low"

        factors = []
        if satellite_risk > 0.5:
            factors.append({"type": "satellite", "score": satellite_risk, "note": "Elevated satellite risk"})
        if country_risk > 0.5:
            factors.append({"type": "country", "score": country_risk, "note": "High-risk country"})
        if supplier_risk > 0.5:
            factors.append({"type": "supplier", "score": supplier_risk, "note": "Supplier compliance issues"})
        if document_risk > 0.5:
            factors.append({"type": "document", "score": document_risk, "note": "Document verification gaps"})

        assessment = {
            "assessment_id": f"ra_{uuid.uuid4().hex[:12]}",
            "satellite_risk": satellite_risk,
            "country_risk": country_risk,
            "supplier_risk": supplier_risk,
            "document_risk": document_risk,
            "overall_risk": overall,
            "risk_level": level,
            "factors": factors,
            "recommendations": self._generate_recommendations(level, factors),
            "plot_id": plot_id,
            "supplier_id": supplier_id,
            "assessed_at": datetime.now(timezone.utc),
        }
        self._assessments.append(assessment)
        return assessment

    def get_country_risk(self, country_iso3: str) -> Dict[str, Any]:
        """Look up country-level deforestation risk."""
        code = country_iso3.upper()
        if code in HIGH_RISK_COUNTRIES:
            return {"country": code, "risk_level": "high", "score": 0.8}
        elif code in LOW_RISK_COUNTRIES:
            return {"country": code, "risk_level": "low", "score": 0.15}
        else:
            return {"country": code, "risk_level": "standard", "score": 0.4}

    def get_risk_heatmap(self, countries: List[str],
                         commodities: List[str]) -> Dict[str, Any]:
        """Generate a risk heatmap matrix (country x commodity)."""
        matrix = {}
        for country in countries:
            cr = self.get_country_risk(country)
            matrix[country] = {}
            for commodity in commodities:
                # Commodity risk modifier
                commodity_mod = 0.1
                if commodity in ("oil_palm", "soya", "cattle"):
                    commodity_mod = 0.3
                elif commodity in ("cocoa", "coffee"):
                    commodity_mod = 0.2
                score = min(1.0, cr["score"] + commodity_mod)
                matrix[country][commodity] = round(score, 4)
        return {"matrix": matrix, "countries": countries, "commodities": commodities}

    def add_alert(self, alert_type: str, severity: str, message: str,
                  plot_id: Optional[str] = None,
                  supplier_id: Optional[str] = None,
                  details: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a risk alert."""
        alert = {
            "alert_id": f"alert_{uuid.uuid4().hex[:12]}",
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "plot_id": plot_id,
            "supplier_id": supplier_id,
            "details": details or {},
            "acknowledged": False,
            "created_at": datetime.now(timezone.utc),
        }
        self._alerts.append(alert)
        return alert

    def get_alerts(self, min_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get risk alerts, optionally filtered by minimum severity."""
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        results = self._alerts[:]
        if min_level:
            min_sev = severity_order.get(min_level, 0)
            results = [a for a in results if severity_order.get(a["severity"], 0) >= min_sev]
        results.sort(key=lambda a: severity_order.get(a["severity"], 0), reverse=True)
        return results

    def get_risk_trends(self, supplier_id: Optional[str] = None,
                        days: int = 30) -> List[Dict[str, Any]]:
        """Get risk assessment time series."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        results = [
            a for a in self._assessments
            if a["assessed_at"] >= cutoff
            and (supplier_id is None or a.get("supplier_id") == supplier_id)
        ]
        results.sort(key=lambda a: a["assessed_at"])
        return [
            {"date": a["assessed_at"].isoformat(), "overall_risk": a["overall_risk"],
             "risk_level": a["risk_level"]}
            for a in results
        ]

    def _generate_recommendations(self, level: str,
                                  factors: List[Dict]) -> List[str]:
        """Generate mitigation recommendations based on risk level and factors."""
        recs = []
        if level == "critical":
            recs.append("IMMEDIATE: Suspend imports until risk is mitigated")
            recs.append("Engage third-party verification immediately")
        if level in ("critical", "high"):
            recs.append("Conduct enhanced due diligence")
            recs.append("Request supplier site inspection")
        for f in factors:
            if f["type"] == "satellite":
                recs.append("Commission fresh satellite analysis for affected plots")
            if f["type"] == "document":
                recs.append("Request updated compliance documentation")
        if level == "low":
            recs.append("Continue standard monitoring")
        return recs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aggregator():
    return RiskAggregator()


# ---------------------------------------------------------------------------
# TestAssessRisk
# ---------------------------------------------------------------------------

class TestAssessRisk:

    def test_combines_four_risk_sources(self, aggregator):
        result = aggregator.assess_risk(0.6, 0.8, 0.4, 0.3)
        assert "overall_risk" in result
        assert "risk_level" in result

    def test_weights_sum_to_one(self):
        total = sum(RISK_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_overall_in_zero_one(self, aggregator):
        result = aggregator.assess_risk(1.0, 1.0, 1.0, 1.0)
        assert 0.0 <= result["overall_risk"] <= 1.0
        result = aggregator.assess_risk(0.0, 0.0, 0.0, 0.0)
        assert result["overall_risk"] == 0.0

    def test_weighted_calculation(self, aggregator):
        result = aggregator.assess_risk(0.5, 0.5, 0.5, 0.5)
        expected = 0.5 * 0.35 + 0.5 * 0.25 + 0.5 * 0.20 + 0.5 * 0.20
        assert result["overall_risk"] == pytest.approx(expected, abs=0.001)

    def test_critical_level(self, aggregator):
        result = aggregator.assess_risk(1.0, 1.0, 1.0, 1.0)
        assert result["risk_level"] == "critical"

    def test_low_level(self, aggregator):
        result = aggregator.assess_risk(0.1, 0.1, 0.1, 0.1)
        assert result["risk_level"] == "low"

    def test_out_of_bounds_raises(self, aggregator):
        with pytest.raises(ValueError):
            aggregator.assess_risk(1.5, 0.5, 0.5, 0.5)
        with pytest.raises(ValueError):
            aggregator.assess_risk(0.5, -0.1, 0.5, 0.5)

    def test_factors_populated(self, aggregator):
        result = aggregator.assess_risk(0.8, 0.9, 0.1, 0.1)
        types = [f["type"] for f in result["factors"]]
        assert "satellite" in types
        assert "country" in types

    def test_assessment_stored(self, aggregator):
        aggregator.assess_risk(0.5, 0.5, 0.5, 0.5)
        assert len(aggregator._assessments) == 1


# ---------------------------------------------------------------------------
# TestCountryRisk
# ---------------------------------------------------------------------------

class TestCountryRisk:

    def test_high_risk_countries(self, aggregator):
        for code in ["BRA", "IDN", "COD"]:
            result = aggregator.get_country_risk(code)
            assert result["risk_level"] == "high"
            assert result["score"] == 0.8

    def test_low_risk_countries(self, aggregator):
        for code in ["DEU", "SWE", "NOR"]:
            result = aggregator.get_country_risk(code)
            assert result["risk_level"] == "low"
            assert result["score"] == 0.15

    def test_unknown_country_default(self, aggregator):
        result = aggregator.get_country_risk("ZZZ")
        assert result["risk_level"] == "standard"
        assert result["score"] == 0.4


# ---------------------------------------------------------------------------
# TestRiskHeatmap
# ---------------------------------------------------------------------------

class TestRiskHeatmap:

    def test_matrix_structure(self, aggregator):
        result = aggregator.get_risk_heatmap(["BRA", "DEU"], ["soya", "wood"])
        assert "BRA" in result["matrix"]
        assert "DEU" in result["matrix"]
        assert "soya" in result["matrix"]["BRA"]
        assert "wood" in result["matrix"]["DEU"]

    def test_all_values_in_zero_one(self, aggregator):
        result = aggregator.get_risk_heatmap(
            ["BRA", "IDN", "DEU"], ["soya", "oil_palm", "wood", "cocoa"]
        )
        for country, commodities in result["matrix"].items():
            for commodity, score in commodities.items():
                assert 0.0 <= score <= 1.0, f"{country}/{commodity} score {score} out of bounds"

    def test_high_risk_combination(self, aggregator):
        result = aggregator.get_risk_heatmap(["BRA"], ["oil_palm"])
        # BRA (0.8) + oil_palm modifier (0.3) = capped at 1.0
        assert result["matrix"]["BRA"]["oil_palm"] == 1.0


# ---------------------------------------------------------------------------
# TestRiskAlerts
# ---------------------------------------------------------------------------

class TestRiskAlerts:

    def test_create_alert(self, aggregator):
        alert = aggregator.add_alert("deforestation", "high", "Deforestation detected")
        assert alert["alert_id"].startswith("alert_")
        assert alert["acknowledged"] is False

    def test_filters_by_min_level(self, aggregator):
        aggregator.add_alert("deforestation", "low", "Low risk")
        aggregator.add_alert("deforestation", "high", "High risk")
        aggregator.add_alert("deforestation", "critical", "Critical risk")
        alerts = aggregator.get_alerts(min_level="high")
        assert len(alerts) == 2
        for a in alerts:
            assert a["severity"] in ("high", "critical")

    def test_sorted_by_severity(self, aggregator):
        aggregator.add_alert("a", "low", "Low")
        aggregator.add_alert("b", "critical", "Critical")
        aggregator.add_alert("c", "medium", "Medium")
        alerts = aggregator.get_alerts()
        assert alerts[0]["severity"] == "critical"
        assert alerts[-1]["severity"] == "low"

    def test_includes_affected_entities(self, aggregator):
        alert = aggregator.add_alert("deforestation", "high", "Detected",
                                     plot_id="plot_123", supplier_id="sup_456")
        assert alert["plot_id"] == "plot_123"
        assert alert["supplier_id"] == "sup_456"


# ---------------------------------------------------------------------------
# TestRiskTrends
# ---------------------------------------------------------------------------

class TestRiskTrends:

    def test_returns_time_series(self, aggregator):
        aggregator.assess_risk(0.5, 0.5, 0.5, 0.5, supplier_id="s1")
        aggregator.assess_risk(0.6, 0.6, 0.6, 0.6, supplier_id="s1")
        trends = aggregator.get_risk_trends(supplier_id="s1")
        assert len(trends) == 2

    def test_ascending_date_order(self, aggregator):
        aggregator.assess_risk(0.3, 0.3, 0.3, 0.3)
        aggregator.assess_risk(0.5, 0.5, 0.5, 0.5)
        trends = aggregator.get_risk_trends()
        assert len(trends) >= 2
        assert trends[0]["date"] <= trends[1]["date"]


# ---------------------------------------------------------------------------
# TestMitigations
# ---------------------------------------------------------------------------

class TestMitigations:

    def test_high_risk_generates_recommendations(self, aggregator):
        result = aggregator.assess_risk(0.9, 0.9, 0.9, 0.9)
        assert len(result["recommendations"]) > 0
        recs_str = " ".join(result["recommendations"]).lower()
        assert "due diligence" in recs_str or "immediate" in recs_str

    def test_low_risk_minimal_recommendations(self, aggregator):
        result = aggregator.assess_risk(0.1, 0.1, 0.1, 0.1)
        assert len(result["recommendations"]) >= 1
        assert "standard monitoring" in result["recommendations"][0].lower()

    def test_satellite_factor_specific_rec(self, aggregator):
        result = aggregator.assess_risk(0.9, 0.3, 0.3, 0.3)
        recs_str = " ".join(result["recommendations"]).lower()
        assert "satellite" in recs_str

    def test_document_factor_specific_rec(self, aggregator):
        result = aggregator.assess_risk(0.3, 0.3, 0.3, 0.9)
        recs_str = " ".join(result["recommendations"]).lower()
        assert "documentation" in recs_str
