"""Tutorial: Add a new country factor set.

This example shows how to add emission factors for a new country
(Brazil in this case) to the GreenLang system.
"""

import json
import os

def add_country_factors(country_code, factors_dict, dataset_path=None):
    """Add emission factors for a new country.
    
    Args:
        country_code: ISO country code (e.g., 'BR' for Brazil)
        factors_dict: Dictionary of fuel types and their emission factors
        dataset_path: Path to the global emission factors file
    """
    if dataset_path is None:
        dataset_path = "greenlang/data/global_emission_factors.json"
    
    # Load existing factors
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    # Add new country factors
    data[country_code] = factors_dict
    
    # Save updated dataset
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return True

# Example usage: Add Brazil factors
brazil_factors = {
    "electricity": {
        "emission_factor": 0.074,
        "unit": "kgCO2e/kWh",
        "source": "Brazil National Energy Balance 2023"
    },
    "natural_gas": {
        "emission_factor": 5.3,
        "unit": "kgCO2e/therm"
    }
}