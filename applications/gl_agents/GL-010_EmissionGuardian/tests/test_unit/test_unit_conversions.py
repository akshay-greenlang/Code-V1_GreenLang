"""
Unit Conversion Calculator Tests for GL-010 EmissionsGuardian
Tests for emission unit conversion calculations.
"""

import pytest
from decimal import Decimal
import hashlib
import json


class EmissionCalculator:
    """Emission calculation utilities per EPA 40 CFR Part 75."""
    
    F_FACTORS = {"coal": Decimal("9780"), "oil": Decimal("9190"), "gas": Decimal("8710")}
    MOLECULAR_WEIGHTS = {"NOx": Decimal("46.01"), "SO2": Decimal("64.06"), "CO": Decimal("28.01")}
    MOLAR_VOLUME_SCF = Decimal("385.326")
    REFERENCE_O2_PERCENT = Decimal("7.0")
    
    def __init__(self):
        self.calculation_trace = []
    def ppm_to_lb_per_mmbtu(self, concentration_ppm, pollutant, fuel_type, o2_percent_dry):
        mw = self.MOLECULAR_WEIGHTS.get(pollutant)
        fd = self.F_FACTORS.get(fuel_type)
        if mw is None or fd is None:
            raise ValueError(f"Unknown pollutant {pollutant} or fuel type {fuel_type}")
        k_factor = (mw * fd) / (self.MOLAR_VOLUME_SCF * Decimal("1000000"))
        o2_correction = Decimal("20.9") / (Decimal("20.9") - o2_percent_dry)
        emission_rate = (k_factor * concentration_ppm * o2_correction).quantize(Decimal("0.000001"))
        calc_data = {"concentration_ppm": str(concentration_ppm), "pollutant": pollutant,
            "fuel_type": fuel_type, "o2_percent_dry": str(o2_percent_dry), "emission_rate": str(emission_rate)}
        provenance_hash = hashlib.sha256(json.dumps(calc_data, sort_keys=True).encode()).hexdigest()
        self.calculation_trace.append({"operation": "ppm_to_lb_per_mmbtu", "inputs": calc_data, "hash": provenance_hash})
        return emission_rate, provenance_hash

    def wet_to_dry_correction(self, concentration_wet, moisture_percent):
        if moisture_percent >= Decimal("100"):
            raise ValueError("Moisture percent must be less than 100")
        dry_concentration = (concentration_wet / (Decimal("1") - moisture_percent / Decimal("100"))).quantize(Decimal("0.01"))
        calc_data = {"concentration_wet": str(concentration_wet), "moisture_percent": str(moisture_percent), "concentration_dry": str(dry_concentration)}
        provenance_hash = hashlib.sha256(json.dumps(calc_data, sort_keys=True).encode()).hexdigest()
        self.calculation_trace.append({"operation": "wet_to_dry_correction", "inputs": calc_data, "hash": provenance_hash})
        return dry_concentration, provenance_hash