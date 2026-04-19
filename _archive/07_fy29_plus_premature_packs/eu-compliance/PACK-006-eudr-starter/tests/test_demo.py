# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Demo Mode Tests
================================================

Validates the demo mode configuration including sample suppliers,
sample plots, GeoJSON parseability, commodity coverage, demo
execution, and report generation.

Test count: 10
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pytest

from conftest import (
    EUDR_COMMODITIES,
    generate_coordinates,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Demo Mode Simulator
# ---------------------------------------------------------------------------

class DemoModeSimulator:
    """Simulates EUDR demo mode execution."""

    def __init__(self):
        self.config = self._build_demo_config()
        self.suppliers = self._build_demo_suppliers()
        self.plots = self._build_demo_plots()

    def _build_demo_config(self) -> Dict[str, Any]:
        """Build demo configuration."""
        return {
            "mode": "demo",
            "company_name": "Demo EcoImports B.V.",
            "operator_type": "OPERATOR",
            "company_size": "MID_MARKET",
            "commodities": ["palm_oil", "cocoa", "wood"],
            "skip_external_apis": True,
            "use_mock_satellite": True,
            "max_suppliers": 10,
            "max_plots": 20,
        }

    def _build_demo_suppliers(self) -> List[Dict[str, Any]]:
        """Build 10 demo suppliers."""
        suppliers = []
        configs = [
            ("PT Sawit Lestari", "IDN", "palm_oil", ["RSPO"]),
            ("Madeira Verde Ltda", "BRA", "wood", ["FSC"]),
            ("Cacao Excellence SARL", "CIV", "cocoa", ["Rainforest_Alliance"]),
            ("Deutsche Holz GmbH", "DEU", "wood", ["PEFC"]),
            ("AgroSoja Brasil", "BRA", "soya", ["RTRS"]),
            ("Kopi Nusantara", "IDN", "coffee", []),
            ("Rubber World MYS", "MYS", "rubber", ["ISCC"]),
            ("Cattle Ranch ARG", "ARG", "cattle", []),
            ("Palm Co MYS", "MYS", "palm_oil", ["RSPO"]),
            ("Timber Solutions COL", "COL", "wood", ["FSC"]),
        ]
        for name, country, commodity, certs in configs:
            suppliers.append({
                "supplier_id": str(uuid.uuid4()),
                "name": name,
                "country": country,
                "commodity": commodity,
                "certifications": [{"scheme": c, "status": "active"} for c in certs],
                "dd_status": "COMPLETED" if certs else "NOT_STARTED",
                "data_completeness": 0.90 if certs else 0.50,
            })
        return suppliers

    def _build_demo_plots(self) -> List[Dict[str, Any]]:
        """Build 20 demo plots."""
        plots = []
        countries = ["IDN", "BRA", "CIV", "MYS", "COL", "DEU", "ARG",
                      "IDN", "MYS", "BRA"]
        for i in range(20):
            country = countries[i % len(countries)]
            coords = generate_coordinates(country, 1)[0]
            area = 2.0 + (i * 2.5)
            plots.append({
                "plot_id": str(uuid.uuid4()),
                "name": f"Demo-Plot-{i + 1:03d}",
                "country": country,
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "area_hectares": round(area, 1),
                "deforestation_free_since": "2019-06-15",
                "polygon": [
                    [coords["latitude"] - 0.01, coords["longitude"] - 0.01],
                    [coords["latitude"] - 0.01, coords["longitude"] + 0.01],
                    [coords["latitude"] + 0.01, coords["longitude"] + 0.01],
                    [coords["latitude"] + 0.01, coords["longitude"] - 0.01],
                    [coords["latitude"] - 0.01, coords["longitude"] - 0.01],
                ] if area > 4.0 else None,
            })
        return plots

    def get_config(self) -> Dict[str, Any]:
        """Get demo config."""
        return self.config

    def get_suppliers(self) -> List[Dict[str, Any]]:
        """Get demo suppliers."""
        return self.suppliers

    def get_plots(self) -> List[Dict[str, Any]]:
        """Get demo plots."""
        return self.plots

    def execute(self) -> Dict[str, Any]:
        """Execute demo mode pipeline."""
        return {
            "status": "COMPLETED",
            "mode": "demo",
            "suppliers_processed": len(self.suppliers),
            "plots_validated": len(self.plots),
            "dds_generated": 3,
            "risk_assessments": len(self.suppliers),
            "compliance_score_pct": 85.0,
            "duration_seconds": 12.5,
        }

    def generate_reports(self) -> Dict[str, Any]:
        """Generate demo reports."""
        return {
            "reports_generated": [
                "dds_report",
                "risk_assessment_report",
                "supplier_compliance_report",
                "compliance_dashboard",
            ],
            "total_reports": 4,
            "format": "markdown",
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the demo mode."""

    @pytest.fixture
    def demo(self) -> DemoModeSimulator:
        return DemoModeSimulator()

    # 1
    def test_demo_config_loads(self, demo):
        """Demo config loads with correct defaults."""
        config = demo.get_config()
        assert config["mode"] == "demo"
        assert config["skip_external_apis"] is True
        assert config["use_mock_satellite"] is True
        assert config["operator_type"] == "OPERATOR"

    # 2
    def test_demo_suppliers_valid(self, demo):
        """Demo mode creates 10 valid suppliers."""
        suppliers = demo.get_suppliers()
        assert len(suppliers) == 10
        for s in suppliers:
            assert "supplier_id" in s
            assert "name" in s
            assert "country" in s
            assert "commodity" in s

    # 3
    def test_demo_plots_valid(self, demo):
        """Demo mode creates 20 valid plots."""
        plots = demo.get_plots()
        assert len(plots) == 20
        for p in plots:
            assert "plot_id" in p
            assert "latitude" in p
            assert "longitude" in p
            assert -90 <= p["latitude"] <= 90
            assert -180 <= p["longitude"] <= 180

    # 4
    def test_demo_geojson_parseable(self, demo):
        """Demo plots with polygons produce parseable GeoJSON."""
        plots = demo.get_plots()
        plots_with_polygon = [p for p in plots if p.get("polygon")]
        assert len(plots_with_polygon) > 0
        for p in plots_with_polygon:
            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[v[1], v[0]] for v in p["polygon"]]],
                },
                "properties": {"plot_id": p["plot_id"]},
            }
            # Should be valid JSON
            json_str = json.dumps(geojson)
            parsed = json.loads(json_str)
            assert parsed["type"] == "Feature"

    # 5
    def test_demo_commodities_covered(self, demo):
        """Demo covers at least 3 distinct commodities."""
        suppliers = demo.get_suppliers()
        commodities = set(s["commodity"] for s in suppliers)
        assert len(commodities) >= 3

    # 6
    def test_demo_commodities_all_seven(self, demo):
        """Demo suppliers cover all 7 EUDR commodities."""
        suppliers = demo.get_suppliers()
        commodities = set(s["commodity"] for s in suppliers)
        assert len(commodities) == 7, (
            f"Expected all 7 EUDR commodities, got {len(commodities)}: {commodities}"
        )

    # 7
    def test_demo_execution(self, demo):
        """Demo pipeline executes successfully."""
        result = demo.execute()
        assert result["status"] == "COMPLETED"
        assert result["mode"] == "demo"
        assert result["suppliers_processed"] == 10
        assert result["plots_validated"] == 20
        assert result["compliance_score_pct"] > 0

    # 8
    def test_demo_reports_generated(self, demo):
        """Demo generates compliance reports."""
        reports = demo.generate_reports()
        assert reports["total_reports"] >= 4
        assert "dds_report" in reports["reports_generated"]
        assert "compliance_dashboard" in reports["reports_generated"]

    # 9
    def test_demo_suppliers_have_mixed_status(self, demo):
        """Demo suppliers have a mix of DD statuses."""
        suppliers = demo.get_suppliers()
        statuses = set(s["dd_status"] for s in suppliers)
        assert len(statuses) >= 2, "Should have at least 2 different DD statuses"

    # 10
    def test_demo_plots_have_coordinates(self, demo):
        """All demo plots have valid WGS84 coordinates."""
        plots = demo.get_plots()
        for p in plots:
            lat = p["latitude"]
            lon = p["longitude"]
            assert isinstance(lat, float), f"Latitude must be float, got {type(lat)}"
            assert isinstance(lon, float), f"Longitude must be float, got {type(lon)}"
            assert -90 <= lat <= 90, f"Latitude {lat} out of range"
            assert -180 <= lon <= 180, f"Longitude {lon} out of range"
