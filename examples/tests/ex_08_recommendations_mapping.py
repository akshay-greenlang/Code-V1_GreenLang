"""Example 08: RecommendationAgent deterministic mapping."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.recommendation_agent import RecommendationAgent
except Exception:
    RecommendationAgent = None

@pytest.mark.example
def test_recommendations_deterministic_mapping():
    """Same input → same recommendations."""
    if RecommendationAgent is None:
        pytest.skip("RecommendationAgent not importable")
    
    payload = {
        "rating": "needs_improvement",
        "by_fuel": {
            "electricity": 1065000.0,
            "diesel": 26800.0
        },
        "co2e_per_sqft": 21.8
    }
    
    out1 = RecommendationAgent().run(payload)
    out2 = RecommendationAgent().run(payload)
    
    assert out1["success"] is True
    assert out2["success"] is True
    assert out1["data"]["recommendations"] == out2["data"]["recommendations"]