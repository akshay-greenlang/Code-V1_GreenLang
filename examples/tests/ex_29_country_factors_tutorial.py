"""Example 29: Test for adding new country factor set tutorial."""

import pytest
import json
import tempfile
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from examples.tutorials.add_country_factors import add_country_factors
except Exception:
    add_country_factors = None

@pytest.mark.example
def test_add_country_factors():
    """Test adding new country emission factors."""
    if add_country_factors is None:
        pytest.skip("add_country_factors tutorial not available")
    
    # Create temporary dataset file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        # Write initial data
        initial_data = {
            "US": {
                "electricity": {"emission_factor": 0.42, "unit": "kgCO2e/kWh"}
            }
        }
        json.dump(initial_data, f)
    
    try:
        # Add Brazil factors
        brazil_factors = {
            "electricity": {
                "emission_factor": 0.074,
                "unit": "kgCO2e/kWh",
                "source": "Brazil National Energy Balance 2023"
            },
            "natural_gas": {
                "emission_factor": 5.3,
                "unit": "kgCO2e/therm",
                "source": "Brazil National Energy Balance 2023"
            }
        }
        
        result = add_country_factors("BR", brazil_factors, temp_path)
        assert result is True
        
        # Verify factors were added
        with open(temp_path, 'r') as f:
            updated_data = json.load(f)
        
        assert "BR" in updated_data
        assert "US" in updated_data  # Original data preserved
        assert updated_data["BR"]["electricity"]["emission_factor"] == 0.074
        assert updated_data["BR"]["natural_gas"]["emission_factor"] == 5.3
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.example
def test_load_custom_country_factors():
    """Test loading custom country factors from file."""
    # Load the example custom factors
    with open("examples/fixtures/custom_country_factors.json", "r") as f:
        custom_factors = json.load(f)
    
    # Verify structure
    assert "BR" in custom_factors
    assert "electricity" in custom_factors["BR"]
    assert custom_factors["BR"]["electricity"]["emission_factor"] == 0.074
    assert "notes" in custom_factors["BR"]["electricity"]