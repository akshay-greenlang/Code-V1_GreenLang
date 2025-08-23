"""Dataset utilities for loading emission factors from the global dataset."""

import json
import os

CANDIDATE_PATHS = [
    # packaged
    os.path.join(os.path.dirname(__file__), "../../greenlang/data/global_emission_factors.json"),
    os.path.join(os.path.dirname(__file__), "../../greenlang/greenlang/data/global_emission_factors.json"),
    # editable / repo
    os.path.join(os.getcwd(), "greenlang/data/global_emission_factors.json"),
    os.path.join(os.getcwd(), "data/global_emission_factors.json"),
]

def load_emission_factor(country: str, fuel: str, unit: str) -> float:
    """Return emission factor (kgCO2e per unit) from dataset — no hard-coded numbers."""
    for path in CANDIDATE_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = data.get(country, {}).get(fuel)
            if not rec or "emission_factor" not in rec:
                raise KeyError(f"Missing factor for {country=} {fuel=} in {path}")
            return float(rec["emission_factor"])
    raise FileNotFoundError("global_emission_factors.json not found (update CANDIDATE_PATHS).")