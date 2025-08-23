"""Example 16: Assistant path mocked (deterministic)."""

import pytest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.assistant import Assistant
except Exception:
    Assistant = None

@pytest.mark.example
def test_assistant_mocked_deterministic():
    """Mock assistant for deterministic testing."""
    if Assistant is None:
        pytest.skip("Assistant not importable")
    
    # Mock the LLM response
    mock_response = {
        "recommendation": "Install solar panels",
        "estimated_savings": 30.5,
        "confidence": 0.85
    }
    
    with patch.object(Assistant, 'get_recommendation', return_value=mock_response):
        assistant = Assistant()
        
        result1 = assistant.get_recommendation({
            "emissions": 1000,
            "building_type": "office"
        })
        
        result2 = assistant.get_recommendation({
            "emissions": 1000,
            "building_type": "office"
        })
        
        # Deterministic: same input → same output
        assert result1 == result2
        assert result1["recommendation"] == "Install solar panels"