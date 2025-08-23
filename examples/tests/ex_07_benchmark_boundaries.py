"""Example 07: BenchmarkAgent threshold boundaries."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.benchmark_agent import BenchmarkAgent
except Exception:
    BenchmarkAgent = None

@pytest.mark.example
def test_benchmark_threshold_boundaries():
    """Test benchmark rating at different intensity levels."""
    if BenchmarkAgent is None:
        pytest.skip("BenchmarkAgent not importable")
    
    # Excellent performance
    out_excellent = BenchmarkAgent().run({
        "co2e_per_sqft": 10.0,
        "building_type": "commercial_office"
    })
    assert out_excellent["success"] is True
    assert "excellent" in out_excellent["data"]["rating"].lower()
    
    # Poor performance
    out_poor = BenchmarkAgent().run({
        "co2e_per_sqft": 50.0,
        "building_type": "commercial_office"
    })
    assert out_poor["success"] is True
    assert "poor" in out_poor["data"]["rating"].lower() or "needs" in out_poor["data"]["rating"].lower()