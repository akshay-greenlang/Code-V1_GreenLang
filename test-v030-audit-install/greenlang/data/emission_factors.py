from typing import Optional
import json
import os


class EmissionFactors:
    """Manages emission factors for different fuel types and regions"""

    DEFAULT_FACTORS = {
        "US": {
            "electricity": {"kWh": 0.385, "MWh": 385.0, "GWh": 385000.0},
            "natural_gas": {
                "therms": 5.3,
                "ccf": 5.3,
                "mcf": 53.0,
                "m3": 1.89,
                "MMBtu": 53.06,
            },
            "diesel": {"gallons": 10.21, "liters": 2.68, "kg": 3.16},
            "gasoline": {"gallons": 8.78, "liters": 2.31, "kg": 3.16},
            "propane": {"gallons": 5.76, "liters": 1.51, "kg": 2.98},
            "fuel_oil": {"gallons": 10.16, "liters": 2.68, "kg": 3.16},
            "coal": {"tons": 2086.0, "kg": 2.086, "lbs": 0.946},
            "biomass": {"tons": 1500.0, "kg": 1.5},
        },
        "EU": {
            "electricity": {"kWh": 0.233, "MWh": 233.0, "GWh": 233000.0},
            "natural_gas": {"therms": 5.3, "m3": 1.89, "MMBtu": 53.06},
        },
        "UK": {
            "electricity": {"kWh": 0.212, "MWh": 212.0, "GWh": 212000.0},
            "natural_gas": {"therms": 5.3, "m3": 1.89, "MMBtu": 53.06},
        },
    }

    def __init__(self, custom_factors_path: Optional[str] = None):
        self.factors = self.DEFAULT_FACTORS.copy()

        if custom_factors_path and os.path.exists(custom_factors_path):
            self._load_custom_factors(custom_factors_path)

    def _load_custom_factors(self, path: str):
        try:
            with open(path, "r") as f:
                custom_factors = json.load(f)
                self.factors.update(custom_factors)
        except Exception as e:
            print(f"Error loading custom factors: {e}")

    def get_factor(
        self, fuel_type: str, unit: str, region: str = "US"
    ) -> Optional[float]:
        region_factors = self.factors.get(region, self.factors["US"])

        if fuel_type in region_factors:
            fuel_factors = region_factors[fuel_type]
            return fuel_factors.get(unit)

        return None

    def get_available_regions(self) -> list:
        return list(self.factors.keys())

    def get_available_fuels(self, region: str = "US") -> list:
        region_factors = self.factors.get(region, {})
        return list(region_factors.keys())

    def get_available_units(self, fuel_type: str, region: str = "US") -> list:
        region_factors = self.factors.get(region, {})
        fuel_factors = region_factors.get(fuel_type, {})
        return list(fuel_factors.keys())

    def add_factor(self, region: str, fuel_type: str, unit: str, factor: float):
        if region not in self.factors:
            self.factors[region] = {}
        if fuel_type not in self.factors[region]:
            self.factors[region][fuel_type] = {}
        self.factors[region][fuel_type][unit] = factor

    def save_to_file(self, path: str):
        with open(path, "w") as f:
            json.dump(self.factors, f, indent=2)
