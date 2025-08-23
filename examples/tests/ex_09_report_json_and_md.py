"""Example 09: ReportAgent JSON and Markdown generation."""

import pytest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.report_agent import ReportAgent
except Exception:
    ReportAgent = None

from examples.utils.normalizers import normalize_text

@pytest.mark.example
def test_report_json_and_markdown():
    """Generate both JSON and Markdown reports."""
    if ReportAgent is None:
        pytest.skip("ReportAgent not importable")
    
    payload = {
        "emissions": {
            "total_co2e_kg": 1091800.0,
            "by_fuel": {"electricity": 1065000.0, "diesel": 26800.0}
        },
        "intensity": {"co2e_per_sqft": 21.84},
        "benchmark": {"rating": "good", "threshold": 25.0},
        "recommendations": ["Install solar panels", "Switch to LED lighting"]
    }
    
    # JSON report
    out_json = ReportAgent().run({**payload, "format": "json"})
    assert out_json["success"] is True
    report = out_json["data"]["report"]
    data = json.loads(report) if isinstance(report, str) else report
    assert data["total_emissions_kg"] == 1091800.0
    
    # Markdown report
    out_md = ReportAgent().run({**payload, "format": "markdown"})
    assert out_md["success"] is True
    md_text = normalize_text(out_md["data"]["report"])
    assert "Total Emissions" in md_text or "total" in md_text.lower()