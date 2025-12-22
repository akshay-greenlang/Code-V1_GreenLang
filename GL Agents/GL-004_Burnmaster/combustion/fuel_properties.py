"""
Fuel Properties Module for GL-004 BURNMASTER

This module provides fuel property databases and calculations for combustion
optimization. All calculations are deterministic and auditable with complete
provenance tracking.

Supports:
- Natural gas (various compositions)
- Refinery gas (high H2, olefins)
- Hydrogen blends (0-100% H2)
- Liquid fuels (diesel, fuel oil)

Reference Standards:
- ISO 6976: Natural gas - Calculation of calorific values
- ASTM D3588: Standard practice for calculating heat value
- GPA 2172: Calculation of gross heating value
"""

from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import hashlib
import json


class FuelType(Enum):
    """Enumeration of supported fuel types."""
    NATURAL_GAS = "natural_gas"
    REFINERY_GAS = "refinery_gas"
    HYDROGEN_BLEND = "hydrogen_blend"
    PURE_HYDROGEN = "pure_hydrogen"
    DIESEL = "diesel"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    PROPANE = "propane"
    BUTANE = "butane"
    LPG = "lpg"
    COKE_OVEN_GAS = "coke_oven_gas"
    BLAST_FURNACE_GAS = "blast_furnace_gas"


@dataclass(frozen=True)
class FuelComposition:
    """Fuel composition in mole/volume percent (0-100 on dry basis)."""
    ch4: float = 0.0
    c2h6: float = 0.0
    c3h8: float = 0.0
    c4h10: float = 0.0
    c5h12: float = 0.0
    c6h14: float = 0.0
    h2: float = 0.0
    co: float = 0.0
    co2: float = 0.0
    n2: float = 0.0
    o2: float = 0.0
    h2s: float = 0.0
    c2h4: float = 0.0
    c3h6: float = 0.0

    def __post_init__(self):
        total = self.total_percent()
        if abs(total - 100.0) > 0.5:
            raise ValueError(f"Fuel composition must sum to 100%, got {total:.2f}%")

    def total_percent(self) -> float:
        return (self.ch4 + self.c2h6 + self.c3h8 + self.c4h10 + self.c5h12 +
                self.c6h14 + self.h2 + self.co + self.co2 + self.n2 + self.o2 +
                self.h2s + self.c2h4 + self.c3h6)

    def to_dict(self) -> Dict[str, float]:
        return {"CH4": self.ch4, "C2H6": self.c2h6, "C3H8": self.c3h8,
                "C4H10": self.c4h10, "C5H12": self.c5h12, "C6H14": self.c6h14,
                "H2": self.h2, "CO": self.co, "CO2": self.co2, "N2": self.n2,
                "O2": self.o2, "H2S": self.h2s, "C2H4": self.c2h4, "C3H6": self.c3h6}


@dataclass
class FuelProperties:
    """Complete fuel properties for combustion calculations."""
    fuel_type: FuelType
    composition: FuelComposition
    hhv: float
    lhv: float
    molecular_weight: float
    specific_gravity: float
    density: float
    stoichiometric_afr: float
    stoichiometric_afr_vol: float
    adiabatic_flame_temp: float
    flammability_lower: float
    flammability_upper: float
    flame_speed: float
    co2_factor: float
    source: str = "GreenLang Fuel Database v1.0"
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        object.__setattr__(self, 'provenance_hash', self._compute_provenance_hash())

    def _compute_provenance_hash(self) -> str:
        data = {"fuel_type": self.fuel_type.value, "hhv": str(self.hhv),
                "molecular_weight": str(self.molecular_weight)}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class FuelQuality:
    """Assessment of fuel quality from combustion analysis."""
    estimated_hhv: float
    estimated_afr: float
    quality_score: float
    deviation_from_design: float
    confidence: float
    warnings: list = field(default_factory=list)


# Component Properties Database - NIST values
MOLECULAR_WEIGHTS: Dict[str, float] = {
    "CH4": 16.043, "C2H6": 30.069, "C3H8": 44.096, "C4H10": 58.122,
    "C5H12": 72.149, "C6H14": 86.175, "H2": 2.016, "CO": 28.010,
    "CO2": 44.009, "N2": 28.014, "O2": 31.999, "H2S": 34.081,
    "C2H4": 28.053, "C3H6": 42.080, "AIR": 28.964
}

# Higher Heating Values (MJ/Nm3 at 15.6C, 101.325 kPa) - ISO 6976
HHV_NM3: Dict[str, float] = {
    "CH4": 39.82, "C2H6": 70.29, "C3H8": 101.24, "C4H10": 133.12,
    "C5H12": 164.51, "C6H14": 195.89, "H2": 12.75, "CO": 12.63,
    "CO2": 0.0, "N2": 0.0, "O2": 0.0, "H2S": 25.30,
    "C2H4": 63.41, "C3H6": 93.58
}

# Lower Heating Values (MJ/Nm3)
LHV_NM3: Dict[str, float] = {
    "CH4": 35.88, "C2H6": 64.36, "C3H8": 93.18, "C4H10": 122.91,
    "C5H12": 152.15, "C6H14": 181.39, "H2": 10.79, "CO": 12.63,
    "CO2": 0.0, "N2": 0.0, "O2": 0.0, "H2S": 23.39,
    "C2H4": 59.45, "C3H6": 87.60
}

# Stoichiometric O2 requirement (mol O2 / mol fuel)
STOICH_O2: Dict[str, float] = {
    "CH4": 2.0, "C2H6": 3.5, "C3H8": 5.0, "C4H10": 6.5, "C5H12": 8.0,
    "C6H14": 9.5, "H2": 0.5, "CO": 0.5, "H2S": 1.5, "C2H4": 3.0,
    "C3H6": 4.5, "CO2": 0.0, "N2": 0.0, "O2": -1.0
}

# CO2 produced (mol CO2 / mol fuel)
CO2_PRODUCED: Dict[str, float] = {
    "CH4": 1.0, "C2H6": 2.0, "C3H8": 3.0, "C4H10": 4.0, "C5H12": 5.0,
    "C6H14": 6.0, "H2": 0.0, "CO": 1.0, "CO2": 0.0, "N2": 0.0,
    "O2": 0.0, "H2S": 0.0, "C2H4": 2.0, "C3H6": 3.0
}

# H2O produced (mol H2O / mol fuel)
H2O_PRODUCED: Dict[str, float] = {
    "CH4": 2.0, "C2H6": 3.0, "C3H8": 4.0, "C4H10": 5.0, "C5H12": 6.0,
    "C6H14": 7.0, "H2": 1.0, "CO": 0.0, "CO2": 0.0, "N2": 0.0,
    "O2": 0.0, "H2S": 1.0, "C2H4": 2.0, "C3H6": 3.0
}

# Adiabatic flame temperatures at stoichiometric (K)
ADIABATIC_FLAME_TEMP: Dict[str, float] = {
    "CH4": 2223.0, "C2H6": 2244.0, "C3H8": 2261.0, "C4H10": 2270.0,
    "H2": 2400.0, "CO": 2381.0, "C2H4": 2375.0, "C3H6": 2334.0
}

# Laminar flame speed (m/s) at stoichiometric, 1 atm, 25C
LAMINAR_FLAME_SPEED: Dict[str, float] = {
    "CH4": 0.40, "C2H6": 0.43, "C3H8": 0.43, "C4H10": 0.45,
    "H2": 3.10, "CO": 0.48, "C2H4": 0.68, "C3H6": 0.52
}

# Standard Fuel Compositions
STANDARD_COMPOSITIONS: Dict[FuelType, FuelComposition] = {
    FuelType.NATURAL_GAS: FuelComposition(
        ch4=94.0, c2h6=3.0, c3h8=1.0, c4h10=0.3, co2=0.7, n2=1.0
    ),
    FuelType.REFINERY_GAS: FuelComposition(
        ch4=35.0, c2h6=10.0, c3h8=5.0, h2=30.0, c2h4=8.0, c3h6=5.0, co=2.0, n2=5.0
    ),
    FuelType.HYDROGEN_BLEND: FuelComposition(ch4=80.0, h2=20.0),
    FuelType.PURE_HYDROGEN: FuelComposition(h2=100.0),
    FuelType.PROPANE: FuelComposition(c3h8=97.0, c4h10=2.0, c2h6=1.0),
    FuelType.BUTANE: FuelComposition(c4h10=98.0, c3h8=1.5, c5h12=0.5),
    FuelType.LPG: FuelComposition(c3h8=60.0, c4h10=40.0),
    FuelType.COKE_OVEN_GAS: FuelComposition(
        h2=55.0, ch4=25.0, co=6.0, co2=2.0, n2=10.0, c2h4=2.0
    ),
    FuelType.BLAST_FURNACE_GAS: FuelComposition(co=23.0, co2=22.0, n2=54.0, h2=1.0),
}


def compute_molecular_weight(composition: FuelComposition) -> float:
    """Compute mixture molecular weight from composition. Deterministic: YES"""
    comp_dict = composition.to_dict()
    mw = sum((mole_pct / 100.0) * MOLECULAR_WEIGHTS.get(comp, 0)
             for comp, mole_pct in comp_dict.items())
    return round(mw, 4)


def compute_heating_values(composition: FuelComposition) -> tuple:
    """Compute mixture HHV and LHV from composition. Deterministic: YES"""
    comp_dict = composition.to_dict()
    hhv = sum((mole_pct / 100.0) * HHV_NM3.get(comp, 0)
              for comp, mole_pct in comp_dict.items())
    lhv = sum((mole_pct / 100.0) * LHV_NM3.get(comp, 0)
              for comp, mole_pct in comp_dict.items())
    return round(hhv, 3), round(lhv, 3)


def compute_specific_gravity(composition: FuelComposition) -> float:
    """Compute specific gravity relative to air. Deterministic: YES"""
    mw = compute_molecular_weight(composition)
    return round(mw / MOLECULAR_WEIGHTS["AIR"], 4)


def compute_wobbe_index(hhv: float, specific_gravity: float) -> float:
    """Compute Wobbe Index for fuel interchangeability. Deterministic: YES"""
    if specific_gravity <= 0:
        raise ValueError(f"Specific gravity must be positive, got {specific_gravity}")
    return round(float(hhv / np.sqrt(specific_gravity)), 3)


def compute_stoichiometric_afr_from_composition(composition: FuelComposition) -> tuple:
    """Compute stoichiometric air-fuel ratio from composition. Deterministic: YES"""
    comp_dict = composition.to_dict()
    total_o2 = sum((mole_pct / 100.0) * STOICH_O2.get(comp, 0)
                   for comp, mole_pct in comp_dict.items())
    air_vol = total_o2 / 0.2095
    mw_fuel = compute_molecular_weight(composition)
    air_mass = air_vol * (MOLECULAR_WEIGHTS["AIR"] / mw_fuel)
    return round(air_mass, 3), round(air_vol, 3)


def compute_co2_emission_factor(composition: FuelComposition) -> float:
    """Compute CO2 emission factor in kg CO2 / GJ (HHV basis). Deterministic: YES"""
    comp_dict = composition.to_dict()
    total_co2 = sum((mole_pct / 100.0) * CO2_PRODUCED.get(comp, 0)
                    for comp, mole_pct in comp_dict.items())
    total_co2 += composition.co2 / 100.0
    kg_co2_per_nm3 = total_co2 * MOLECULAR_WEIGHTS["CO2"] / 22.414
    hhv, _ = compute_heating_values(composition)
    return round(kg_co2_per_nm3 / (hhv / 1000.0), 2) if hhv > 0 else 0.0


def _estimate_adiabatic_flame_temp(composition: FuelComposition) -> float:
    """Estimate adiabatic flame temperature from composition."""
    comp_dict = composition.to_dict()
    total = sum(mole_pct for comp, mole_pct in comp_dict.items()
                if comp in ADIABATIC_FLAME_TEMP)
    weighted = sum(mole_pct * ADIABATIC_FLAME_TEMP.get(comp, 0)
                   for comp, mole_pct in comp_dict.items()
                   if comp in ADIABATIC_FLAME_TEMP)
    if total > 0:
        avg = weighted / total
        correction = 1.0 - 0.5 * (composition.co2 + composition.n2) / 100.0
        return round(avg * correction, 1)
    return 2000.0


def _estimate_flame_speed(composition: FuelComposition) -> float:
    """Estimate laminar flame speed from composition."""
    comp_dict = composition.to_dict()
    total = sum(mole_pct for comp, mole_pct in comp_dict.items()
                if comp in LAMINAR_FLAME_SPEED)
    weighted = sum(mole_pct * LAMINAR_FLAME_SPEED.get(comp, 0)
                   for comp, mole_pct in comp_dict.items()
                   if comp in LAMINAR_FLAME_SPEED)
    return round(weighted / total, 3) if total > 0 else 0.40


def _estimate_flammability_limits(composition: FuelComposition) -> tuple:
    """Estimate flammability limits using Le Chatelier's rule."""
    LFL = {"CH4": 5.0, "C2H6": 3.0, "C3H8": 2.1, "C4H10": 1.8, "C5H12": 1.4,
           "C6H14": 1.2, "H2": 4.0, "CO": 12.5, "C2H4": 2.7, "C3H6": 2.0, "H2S": 4.0}
    UFL = {"CH4": 15.0, "C2H6": 12.4, "C3H8": 9.5, "C4H10": 8.4, "C5H12": 7.8,
           "C6H14": 7.4, "H2": 75.0, "CO": 74.0, "C2H4": 36.0, "C3H6": 11.1, "H2S": 44.0}
    comp_dict = composition.to_dict()
    inv_lfl = sum(mole_pct / LFL[comp] for comp, mole_pct in comp_dict.items()
                  if comp in LFL and mole_pct > 0)
    inv_ufl = sum(mole_pct / UFL[comp] for comp, mole_pct in comp_dict.items()
                  if comp in UFL and mole_pct > 0)
    total = sum(mole_pct for comp, mole_pct in comp_dict.items()
                if comp in LFL and mole_pct > 0)
    if inv_lfl > 0 and inv_ufl > 0:
        return round(total / inv_lfl, 2), round(total / inv_ufl, 2)
    return 5.0, 15.0


def compute_fuel_properties(
    composition: FuelComposition,
    fuel_type: FuelType = FuelType.NATURAL_GAS
) -> FuelProperties:
    """Compute complete fuel properties from composition. Deterministic: YES"""
    mw = compute_molecular_weight(composition)
    hhv, lhv = compute_heating_values(composition)
    sg = compute_specific_gravity(composition)
    afr_mass, afr_vol = compute_stoichiometric_afr_from_composition(composition)
    co2_factor = compute_co2_emission_factor(composition)
    density = mw / 22.414
    aft = _estimate_adiabatic_flame_temp(composition)
    flame_speed = _estimate_flame_speed(composition)
    lfl, ufl = _estimate_flammability_limits(composition)
    return FuelProperties(
        fuel_type=fuel_type, composition=composition, hhv=hhv, lhv=lhv,
        molecular_weight=mw, specific_gravity=sg, density=round(density, 4),
        stoichiometric_afr=afr_mass, stoichiometric_afr_vol=afr_vol,
        adiabatic_flame_temp=aft, flammability_lower=lfl, flammability_upper=ufl,
        flame_speed=flame_speed, co2_factor=co2_factor
    )


def get_fuel_properties(fuel_type: FuelType) -> FuelProperties:
    """Get complete fuel properties for a standard fuel type. Deterministic: YES"""
    if fuel_type not in STANDARD_COMPOSITIONS:
        raise ValueError(f"Fuel type {fuel_type} not in standard database.")
    return compute_fuel_properties(STANDARD_COMPOSITIONS[fuel_type], fuel_type)


def estimate_fuel_quality_from_o2_co(
    o2: float,
    co: float,
    baseline_o2: float = 3.0,
    baseline_co: float = 50.0,
    fuel_type: FuelType = FuelType.NATURAL_GAS
) -> FuelQuality:
    """Estimate fuel quality deviation from stack O2 and CO readings. Deterministic: YES"""
    if o2 < 0 or o2 > 21:
        raise ValueError(f"O2 must be 0-21%, got {o2}")
    if co < 0:
        raise ValueError(f"CO must be non-negative, got {co}")
    baseline = get_fuel_properties(fuel_type)
    o2_dev = o2 - baseline_o2
    correction = 1.0 - (o2_dev * 0.04)
    est_hhv = baseline.hhv * correction
    est_afr = baseline.stoichiometric_afr * correction
    o2_score = max(0, 1.0 - abs(o2_dev) / 3.0)
    co_score = max(0, 1.0 - abs(co - baseline_co) / 500.0)
    quality = (o2_score + co_score) / 2.0
    deviation = abs((est_hhv - baseline.hhv) / baseline.hhv * 100)
    warnings = []
    if o2 < 1.5:
        warnings.append("Very low O2 - risk of incomplete combustion")
    if o2 > 6.0:
        warnings.append("High O2 - excessive excess air")
    if co > 200:
        warnings.append("High CO - possible combustion issues")
    if co > 500:
        warnings.append("CRITICAL: Very high CO")
    confidence = 0.7 * (0.8 if abs(o2_dev) > 2.0 else 1.0)
    return FuelQuality(
        estimated_hhv=round(est_hhv, 2), estimated_afr=round(est_afr, 3),
        quality_score=round(quality, 3), deviation_from_design=round(deviation, 2),
        confidence=round(confidence, 3), warnings=warnings
    )


def validate_fuel_composition(composition: Dict[str, float]) -> tuple:
    """Validate a fuel composition dictionary. Deterministic: YES"""
    errors = []
    valid = set(MOLECULAR_WEIGHTS.keys()) - {"AIR"}
    for comp in composition:
        if comp not in valid:
            errors.append(f"Unknown: {comp}")
    for comp, value in composition.items():
        if value < 0:
            errors.append(f"{comp}: negative")
        if value > 100:
            errors.append(f"{comp}: >100%")
    total = sum(composition.values())
    if abs(total - 100.0) > 0.5:
        errors.append(f"Sum={total}%, expected 100%")
    return len(errors) == 0, errors


def validate_fuel_properties(props: FuelProperties) -> tuple:
    """Validate computed fuel properties are physically reasonable. Deterministic: YES"""
    errors = []
    if props.hhv < props.lhv:
        errors.append("HHV < LHV")
    if props.hhv < 0 or props.hhv > 250:
        errors.append(f"HHV out of range: {props.hhv}")
    if props.molecular_weight < 2 or props.molecular_weight > 200:
        errors.append(f"MW out of range")
    if props.specific_gravity < 0.05 or props.specific_gravity > 5:
        errors.append(f"SG out of range")
    if props.stoichiometric_afr < 0.5 or props.stoichiometric_afr > 40:
        errors.append(f"AFR out of range")
    if props.adiabatic_flame_temp < 1500 or props.adiabatic_flame_temp > 3000:
        errors.append(f"AFT out of range")
    return len(errors) == 0, errors
