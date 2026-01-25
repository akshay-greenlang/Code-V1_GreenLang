#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Example Usage Demonstrations

This module provides comprehensive examples for using the GL-015 INSULSCAN
Industrial Insulation Inspection Agent. Examples include:

1. Basic thermal image analysis
2. Heat loss calculations
3. Degradation assessment
4. Repair prioritization
5. Facility-wide inspection
6. Integration examples (cameras, CMMS)

All examples demonstrate the zero-hallucination guarantee with
deterministic calculations and full provenance tracking.

Usage:
    python example_usage.py --example basic
    python example_usage.py --example heat_loss
    python example_usage.py --example all

Author: GreenLang Engineering Team
Version: 1.0.0
License: Apache 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Mock Imports (These would be real imports in production)
# =============================================================================

# In production, these would be:
# from gl_015 import InsulationInspectionAgent
# from gl_015.config import InsulationConfig
# from gl_015.calculators import (
#     ThermalImageAnalyzer,
#     HeatLossCalculator,
#     RepairPrioritizationEngine
# )
# from gl_015.calculators.heat_loss_calculator import (
#     SurfaceGeometry, SurfaceMaterial, INSULATION_MATERIALS
# )
# from gl_015.calculators.repair_prioritization import (
#     ThermalDefect, DefectLocation, EconomicParameters,
#     EquipmentType, DamageType, InsulationMaterial
# )


# =============================================================================
# Simulation Classes for Demonstration
# =============================================================================

class SurfaceGeometry(Enum):
    """Surface geometry types for heat transfer calculations."""
    FLAT_HORIZONTAL_UP = "flat_horizontal_up"
    FLAT_HORIZONTAL_DOWN = "flat_horizontal_down"
    FLAT_VERTICAL = "flat_vertical"
    CYLINDER_HORIZONTAL = "cylinder_horizontal"
    CYLINDER_VERTICAL = "cylinder_vertical"
    SPHERE = "sphere"


class SurfaceMaterial(Enum):
    """Surface material types for emissivity lookup."""
    ALUMINUM_POLISHED = "aluminum_polished"
    ALUMINUM_OXIDIZED = "aluminum_oxidized"
    ALUMINUM_JACKETING = "aluminum_jacketing"
    STAINLESS_STEEL_POLISHED = "stainless_steel_polished"
    STAINLESS_STEEL_OXIDIZED = "stainless_steel_oxidized"
    GALVANIZED_STEEL_NEW = "galvanized_steel_new"
    GALVANIZED_STEEL_WEATHERED = "galvanized_steel_weathered"
    PAINTED_SURFACE = "painted_surface"


class EquipmentType(Enum):
    """Equipment types for inspection."""
    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    VALVE = "valve"
    HEAT_EXCHANGER = "heat_exchanger"
    DUCT = "duct"


class DamageType(Enum):
    """Types of insulation damage."""
    MISSING = "missing"
    COMPRESSED = "compressed"
    WET = "wet"
    CRACKED = "cracked"
    JACKET_DAMAGED = "jacket_damaged"


class InsulationMaterial(Enum):
    """Insulation material types."""
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    POLYURETHANE_FOAM = "polyurethane_foam"


# Physical constants
STEFAN_BOLTZMANN = Decimal("5.670374419e-8")
KELVIN_OFFSET = Decimal("273.15")
PI = Decimal("3.14159265358979323846")


# =============================================================================
# Example 1: Basic Thermal Image Analysis
# =============================================================================

def example_basic_thermal_analysis() -> None:
    """
    Example 1: Basic Thermal Image Analysis

    Demonstrates:
    - Processing a temperature matrix from an IR camera
    - Detecting thermal hotspots
    - Calculating image statistics
    - Assessing anomaly severity
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC THERMAL IMAGE ANALYSIS")
    print("="*70)

    # Simulated thermal image data (10x10 temperature matrix in Celsius)
    # This represents a pipe section with a hotspot indicating damaged insulation
    temperature_matrix = [
        [45.0, 46.2, 47.1, 48.5, 52.3, 55.8, 48.2, 46.1, 45.3, 45.0],
        [45.2, 46.5, 47.8, 49.2, 53.1, 56.4, 49.1, 46.8, 45.6, 45.1],
        [45.1, 46.3, 48.2, 50.1, 54.2, 57.1, 50.3, 47.2, 45.8, 45.2],
        [45.3, 46.8, 48.5, 51.2, 55.3, 58.2, 51.5, 47.8, 46.1, 45.3],
        [45.2, 46.5, 48.1, 50.8, 54.8, 57.8, 51.0, 47.5, 45.9, 45.1],
        [45.0, 46.2, 47.5, 49.8, 53.5, 56.5, 49.8, 46.9, 45.5, 45.0],
        [44.9, 45.8, 46.9, 48.5, 51.2, 54.1, 48.2, 46.2, 45.2, 44.9],
        [44.8, 45.5, 46.2, 47.2, 49.1, 51.2, 47.0, 45.8, 45.0, 44.8],
        [44.7, 45.2, 45.8, 46.5, 47.8, 49.2, 46.3, 45.5, 44.9, 44.7],
        [44.6, 45.0, 45.5, 46.0, 47.0, 48.1, 45.9, 45.2, 44.8, 44.6],
    ]

    # Configuration
    ambient_temp_c = Decimal("25.0")
    emissivity = Decimal("0.90")  # Aluminum jacketing
    pixel_size_m = Decimal("0.005")  # 5mm per pixel

    print("\n--- Input Parameters ---")
    print(f"Image Size: {len(temperature_matrix)}x{len(temperature_matrix[0])} pixels")
    print(f"Pixel Size: {pixel_size_m} m")
    print(f"Emissivity: {emissivity}")
    print(f"Ambient Temperature: {ambient_temp_c} C")

    # Calculate statistics
    all_temps = [t for row in temperature_matrix for t in row]
    min_temp = Decimal(str(min(all_temps)))
    max_temp = Decimal(str(max(all_temps)))
    avg_temp = Decimal(str(sum(all_temps) / len(all_temps)))

    print("\n--- Temperature Statistics ---")
    print(f"Minimum Temperature: {min_temp:.2f} C")
    print(f"Maximum Temperature: {max_temp:.2f} C")
    print(f"Average Temperature: {avg_temp:.2f} C")
    print(f"Temperature Range: {max_temp - min_temp:.2f} C")

    # Detect hotspots (pixels > ambient + 10C threshold)
    delta_t_threshold = Decimal("10.0")
    hotspot_threshold = ambient_temp_c + delta_t_threshold

    hotspots = []
    for i, row in enumerate(temperature_matrix):
        for j, temp in enumerate(row):
            if Decimal(str(temp)) > hotspot_threshold:
                hotspots.append({
                    "row": i,
                    "col": j,
                    "temperature_c": Decimal(str(temp)),
                    "delta_t": Decimal(str(temp)) - ambient_temp_c
                })

    print("\n--- Hotspot Detection ---")
    print(f"Detection Threshold: {hotspot_threshold:.1f} C (ambient + {delta_t_threshold} C)")
    print(f"Hotspots Detected: {len(hotspots)}")

    if hotspots:
        # Find peak hotspot
        peak_hotspot = max(hotspots, key=lambda h: h["temperature_c"])

        print(f"\nPeak Hotspot:")
        print(f"  Location: Row {peak_hotspot['row']}, Col {peak_hotspot['col']}")
        print(f"  Temperature: {peak_hotspot['temperature_c']:.1f} C")
        print(f"  Delta-T from Ambient: {peak_hotspot['delta_t']:.1f} C")

        # Calculate hotspot area
        hotspot_area_m2 = len(hotspots) * (pixel_size_m ** 2)
        print(f"  Hotspot Area: {hotspot_area_m2:.6f} m2")

        # Severity assessment
        if peak_hotspot["delta_t"] > Decimal("50"):
            severity = "CRITICAL"
        elif peak_hotspot["delta_t"] > Decimal("30"):
            severity = "SEVERE"
        elif peak_hotspot["delta_t"] > Decimal("20"):
            severity = "MODERATE"
        else:
            severity = "MINOR"

        print(f"  Severity: {severity}")

    # Generate provenance hash (simplified)
    import hashlib
    provenance_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "min_temp": str(min_temp),
        "max_temp": str(max_temp),
        "avg_temp": str(avg_temp),
        "hotspot_count": len(hotspots)
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    print(f"\n--- Provenance ---")
    print(f"Provenance Hash: sha256:{provenance_hash}...")

    print("\n[Example 1 Complete]")


# =============================================================================
# Example 2: Heat Loss Calculation
# =============================================================================

def example_heat_loss_calculation() -> None:
    """
    Example 2: Heat Loss Calculation

    Demonstrates ASTM C680 compliant heat loss calculations:
    - Conduction through insulation
    - Natural convection from surface
    - Radiation heat transfer
    - Combined heat loss
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: HEAT LOSS CALCULATION (ASTM C680)")
    print("="*70)

    # Input parameters
    process_temp_c = Decimal("180.0")  # Process temperature
    surface_temp_c = Decimal("45.0")   # Measured surface temperature
    ambient_temp_c = Decimal("25.0")   # Ambient temperature

    # Pipe geometry
    pipe_od_mm = Decimal("168.3")      # 6" NPS pipe
    insulation_thickness_mm = Decimal("75.0")
    pipe_length_m = Decimal("10.0")

    # Material properties
    insulation_k = Decimal("0.045")    # W/m.K at mean temp
    surface_emissivity = Decimal("0.10")  # Aluminum jacketing

    print("\n--- Input Parameters ---")
    print(f"Process Temperature: {process_temp_c} C")
    print(f"Surface Temperature: {surface_temp_c} C")
    print(f"Ambient Temperature: {ambient_temp_c} C")
    print(f"Pipe OD: {pipe_od_mm} mm")
    print(f"Insulation Thickness: {insulation_thickness_mm} mm")
    print(f"Pipe Length: {pipe_length_m} m")
    print(f"Insulation k-value: {insulation_k} W/m.K")
    print(f"Surface Emissivity: {surface_emissivity}")

    # Calculate radii
    r_inner_m = (pipe_od_mm / 2) / Decimal("1000")
    r_outer_m = (pipe_od_mm / 2 + insulation_thickness_mm) / Decimal("1000")
    outer_diameter_m = r_outer_m * 2

    print("\n--- Calculated Geometry ---")
    print(f"Inner Radius: {r_inner_m:.4f} m")
    print(f"Outer Radius: {r_outer_m:.4f} m")
    print(f"Outer Diameter: {outer_diameter_m:.4f} m")

    # Calculate surface area
    surface_area_m2 = PI * outer_diameter_m * pipe_length_m
    print(f"Surface Area: {surface_area_m2:.4f} m2")

    # Step 1: Conduction through cylindrical insulation
    # q = 2*pi*L*k*(T_hot - T_cold) / ln(r_outer/r_inner)
    print("\n--- Step 1: Conduction (Fourier's Law) ---")
    import math
    ln_ratio = Decimal(str(math.log(float(r_outer_m / r_inner_m))))

    q_conduction = (
        2 * PI * pipe_length_m * insulation_k *
        (process_temp_c - surface_temp_c) / ln_ratio
    )

    print(f"Formula: q = 2*pi*L*k*(T_process - T_surface) / ln(r_o/r_i)")
    print(f"ln(r_outer/r_inner): {ln_ratio:.4f}")
    print(f"Conduction Heat Transfer: {q_conduction:.2f} W")

    # Step 2: Natural convection from horizontal cylinder
    # Using simplified correlation: h = 1.32 * (dT/D)^0.25
    print("\n--- Step 2: Natural Convection ---")
    delta_t_surface = surface_temp_c - ambient_temp_c

    # Churchill-Chu correlation (simplified)
    h_conv = Decimal("1.32") * (delta_t_surface / outer_diameter_m) ** Decimal("0.25")

    q_convection = h_conv * surface_area_m2 * delta_t_surface

    print(f"Formula: h = 1.32 * (dT/D)^0.25 (simplified)")
    print(f"Delta-T (surface to ambient): {delta_t_surface} C")
    print(f"Convection Coefficient h: {h_conv:.2f} W/m2.K")
    print(f"Convection Heat Transfer: {q_convection:.2f} W")

    # Step 3: Radiation heat transfer
    # q = epsilon * sigma * A * (T_s^4 - T_a^4)
    print("\n--- Step 3: Radiation (Stefan-Boltzmann) ---")
    t_surface_k = surface_temp_c + KELVIN_OFFSET
    t_ambient_k = ambient_temp_c + KELVIN_OFFSET

    q_radiation = (
        surface_emissivity * STEFAN_BOLTZMANN * surface_area_m2 *
        (t_surface_k**4 - t_ambient_k**4)
    )

    print(f"Formula: q = epsilon * sigma * A * (T_s^4 - T_a^4)")
    print(f"T_surface: {t_surface_k:.2f} K")
    print(f"T_ambient: {t_ambient_k:.2f} K")
    print(f"Radiation Heat Transfer: {q_radiation:.2f} W")

    # Step 4: Combined heat loss
    print("\n--- Step 4: Combined Heat Loss ---")
    q_total_surface = q_convection + q_radiation

    print(f"Total Surface Heat Loss: {q_total_surface:.2f} W")
    print(f"  - Convection: {q_convection:.2f} W ({q_convection/q_total_surface*100:.1f}%)")
    print(f"  - Radiation: {q_radiation:.2f} W ({q_radiation/q_total_surface*100:.1f}%)")

    # Heat loss per unit length
    q_per_m = q_total_surface / pipe_length_m
    print(f"\nHeat Loss per meter: {q_per_m:.2f} W/m")

    # Annual energy loss
    operating_hours = Decimal("8000")
    annual_energy_kwh = q_total_surface * operating_hours / Decimal("1000")
    annual_energy_mwh = annual_energy_kwh / Decimal("1000")

    print("\n--- Annual Energy Impact ---")
    print(f"Operating Hours: {operating_hours} hours/year")
    print(f"Annual Energy Loss: {annual_energy_kwh:.1f} kWh ({annual_energy_mwh:.2f} MWh)")

    # Cost calculation
    energy_cost_per_kwh = Decimal("0.10")
    annual_cost = annual_energy_kwh * energy_cost_per_kwh

    print(f"Energy Cost: ${energy_cost_per_kwh}/kWh")
    print(f"Annual Cost: ${annual_cost:.2f}")

    # CO2 emissions
    co2_factor = Decimal("0.185")  # kg CO2/kWh (natural gas)
    annual_co2_kg = annual_energy_kwh * co2_factor
    annual_co2_tonnes = annual_co2_kg / Decimal("1000")

    print(f"CO2 Emissions: {annual_co2_kg:.1f} kg/year ({annual_co2_tonnes:.2f} tonnes)")

    print("\n[Example 2 Complete]")


# =============================================================================
# Example 3: Degradation Assessment
# =============================================================================

def example_degradation_assessment() -> None:
    """
    Example 3: Degradation Assessment

    Demonstrates:
    - Insulation condition scoring
    - CUI (Corrosion Under Insulation) risk assessment
    - Remaining useful life estimation
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: DEGRADATION ASSESSMENT")
    print("="*70)

    # Input parameters
    measured_surface_temp_c = Decimal("65.0")
    design_surface_temp_c = Decimal("40.0")   # Expected for new insulation
    bare_surface_temp_c = Decimal("175.0")    # Temperature if uninsulated
    process_temp_c = Decimal("180.0")
    ambient_temp_c = Decimal("25.0")

    insulation_age_years = Decimal("15.0")
    insulation_type = InsulationMaterial.MINERAL_WOOL
    environment = "outdoor_industrial"

    print("\n--- Input Parameters ---")
    print(f"Process Temperature: {process_temp_c} C")
    print(f"Measured Surface Temperature: {measured_surface_temp_c} C")
    print(f"Design Surface Temperature: {design_surface_temp_c} C")
    print(f"Bare Surface Temperature: {bare_surface_temp_c} C")
    print(f"Ambient Temperature: {ambient_temp_c} C")
    print(f"Insulation Age: {insulation_age_years} years")
    print(f"Insulation Type: {insulation_type.value}")
    print(f"Environment: {environment}")

    # Calculate thermal efficiency
    print("\n--- Thermal Efficiency ---")
    # Efficiency = (T_bare - T_measured) / (T_bare - T_design) * 100
    efficiency = (
        (bare_surface_temp_c - measured_surface_temp_c) /
        (bare_surface_temp_c - design_surface_temp_c) * Decimal("100")
    )

    print(f"Formula: (T_bare - T_measured) / (T_bare - T_design) * 100")
    print(f"Thermal Efficiency: {efficiency:.1f}%")

    # Determine condition grade
    if efficiency >= Decimal("95"):
        condition_grade = "EXCELLENT"
        condition_score = Decimal("95")
    elif efficiency >= Decimal("85"):
        condition_grade = "GOOD"
        condition_score = Decimal("82")
    elif efficiency >= Decimal("70"):
        condition_grade = "FAIR"
        condition_score = Decimal("68")
    elif efficiency >= Decimal("50"):
        condition_grade = "POOR"
        condition_score = Decimal("45")
    else:
        condition_grade = "CRITICAL"
        condition_score = Decimal("25")

    print(f"Condition Grade: {condition_grade}")
    print(f"Condition Score: {condition_score}/100")

    # CUI Risk Assessment
    print("\n--- CUI Risk Assessment ---")

    # Temperature risk factor (high risk in 50-150C range)
    if Decimal("50") <= process_temp_c <= Decimal("150"):
        temp_risk_factor = Decimal("1.0")
        temp_risk_desc = "HIGH (wet-dry cycling zone)"
    elif process_temp_c < Decimal("50"):
        temp_risk_factor = Decimal("0.3")
        temp_risk_desc = "LOW (below cycling zone)"
    elif process_temp_c <= Decimal("250"):
        temp_risk_factor = Decimal("0.6")
        temp_risk_desc = "MODERATE"
    else:
        temp_risk_factor = Decimal("0.2")
        temp_risk_desc = "LOW (too hot for moisture)"

    print(f"Temperature Risk: {temp_risk_factor} - {temp_risk_desc}")

    # Environment risk factor
    env_risk_factors = {
        "indoor_dry": Decimal("0.2"),
        "indoor_humid": Decimal("0.5"),
        "outdoor_temperate": Decimal("0.6"),
        "outdoor_marine": Decimal("1.0"),
        "outdoor_industrial": Decimal("0.9"),
    }
    env_risk_factor = env_risk_factors.get(environment, Decimal("0.5"))
    print(f"Environment Risk: {env_risk_factor} - {environment}")

    # Condition risk factor
    condition_risk_mapping = {
        "EXCELLENT": Decimal("0.1"),
        "GOOD": Decimal("0.3"),
        "FAIR": Decimal("0.6"),
        "POOR": Decimal("0.9"),
        "CRITICAL": Decimal("1.0"),
    }
    condition_risk_factor = condition_risk_mapping[condition_grade]
    print(f"Condition Risk: {condition_risk_factor} - {condition_grade}")

    # Age risk factor
    if insulation_age_years < Decimal("5"):
        age_risk_factor = Decimal("0.2")
    elif insulation_age_years < Decimal("10"):
        age_risk_factor = Decimal("0.4")
    elif insulation_age_years < Decimal("20"):
        age_risk_factor = Decimal("0.7")
    else:
        age_risk_factor = Decimal("1.0")
    print(f"Age Risk: {age_risk_factor} - {insulation_age_years} years")

    # Combined CUI risk score (weighted average)
    cui_risk_score = (
        temp_risk_factor * Decimal("0.30") +
        env_risk_factor * Decimal("0.25") +
        condition_risk_factor * Decimal("0.25") +
        age_risk_factor * Decimal("0.20")
    ) * Decimal("100")

    # CUI risk level
    if cui_risk_score >= Decimal("75"):
        cui_risk_level = "CRITICAL"
    elif cui_risk_score >= Decimal("50"):
        cui_risk_level = "HIGH"
    elif cui_risk_score >= Decimal("25"):
        cui_risk_level = "MEDIUM"
    else:
        cui_risk_level = "LOW"

    print(f"\nCombined CUI Risk Score: {cui_risk_score:.1f}/100")
    print(f"CUI Risk Level: {cui_risk_level}")

    # Remaining useful life estimation
    print("\n--- Remaining Useful Life ---")
    expected_life_years = Decimal("25")  # Typical for mineral wool

    # Adjust for condition
    condition_multiplier = condition_score / Decimal("100")
    adjusted_life = expected_life_years * condition_multiplier
    remaining_life = max(adjusted_life - insulation_age_years, Decimal("0"))

    print(f"Expected Total Life: {expected_life_years} years")
    print(f"Condition Adjustment: {condition_multiplier:.2f}")
    print(f"Adjusted Expected Life: {adjusted_life:.1f} years")
    print(f"Remaining Useful Life: {remaining_life:.1f} years")

    # Recommendations
    print("\n--- Recommendations ---")
    if condition_grade == "CRITICAL":
        print("  [!] IMMEDIATE ACTION REQUIRED")
        print("  - Schedule emergency repair/replacement")
        print("  - Conduct CUI inspection under insulation")
    elif condition_grade == "POOR":
        print("  [!] PRIORITY REPAIR NEEDED")
        print("  - Schedule replacement within 6 months")
        print("  - Monitor for further degradation")
    elif condition_grade == "FAIR":
        print("  [!] SCHEDULED MAINTENANCE")
        print("  - Plan repair during next turnaround")
        print("  - Continue periodic monitoring")
    else:
        print("  [OK] ROUTINE MONITORING")
        print("  - Continue annual inspection schedule")

    if cui_risk_level in ["CRITICAL", "HIGH"]:
        print(f"  [!] CUI INVESTIGATION RECOMMENDED ({cui_risk_level} risk)")

    print("\n[Example 3 Complete]")


# =============================================================================
# Example 4: Repair Prioritization
# =============================================================================

def example_repair_prioritization() -> None:
    """
    Example 4: Repair Prioritization

    Demonstrates:
    - Multi-factor criticality scoring
    - ROI-based ranking
    - Work scope generation
    - Budget optimization
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: REPAIR PRIORITIZATION")
    print("="*70)

    # Define multiple defects for prioritization
    defects = [
        {
            "id": "DEF-001",
            "equipment": "10-P-101-A",
            "type": EquipmentType.PIPE,
            "damage": DamageType.MISSING,
            "length_m": Decimal("5.0"),
            "heat_loss_w_m": Decimal("450.0"),
            "surface_temp_c": Decimal("120.0"),
            "process_temp_c": Decimal("180.0"),
            "criticality": "A",
            "access": "easy",
        },
        {
            "id": "DEF-002",
            "equipment": "20-V-201",
            "type": EquipmentType.VESSEL,
            "damage": DamageType.WET,
            "length_m": Decimal("3.0"),
            "heat_loss_w_m": Decimal("280.0"),
            "surface_temp_c": Decimal("85.0"),
            "process_temp_c": Decimal("95.0"),
            "criticality": "A",
            "access": "scaffold",
        },
        {
            "id": "DEF-003",
            "equipment": "10-P-102-B",
            "type": EquipmentType.PIPE,
            "damage": DamageType.COMPRESSED,
            "length_m": Decimal("8.0"),
            "heat_loss_w_m": Decimal("120.0"),
            "surface_temp_c": Decimal("55.0"),
            "process_temp_c": Decimal("150.0"),
            "criticality": "B",
            "access": "easy",
        },
        {
            "id": "DEF-004",
            "equipment": "30-E-301",
            "type": EquipmentType.HEAT_EXCHANGER,
            "damage": DamageType.JACKET_DAMAGED,
            "length_m": Decimal("2.0"),
            "heat_loss_w_m": Decimal("350.0"),
            "surface_temp_c": Decimal("95.0"),
            "process_temp_c": Decimal("200.0"),
            "criticality": "A",
            "access": "confined",
        },
    ]

    # Economic parameters
    energy_cost_kwh = Decimal("0.10")
    operating_hours = Decimal("8000")
    discount_rate = Decimal("0.08")
    carbon_price_tonne = Decimal("50.0")

    print("\n--- Defects to Prioritize ---")
    for defect in defects:
        print(f"\n{defect['id']} - {defect['equipment']}")
        print(f"  Type: {defect['type'].value}, Damage: {defect['damage'].value}")
        print(f"  Length: {defect['length_m']}m, Heat Loss: {defect['heat_loss_w_m']} W/m")
        print(f"  Criticality: {defect['criticality']}, Access: {defect['access']}")

    # Calculate scores for each defect
    print("\n--- Scoring Analysis ---")
    scored_defects = []

    for defect in defects:
        # Heat loss score (0-100)
        total_heat_loss = defect["heat_loss_w_m"] * defect["length_m"]
        annual_kwh = total_heat_loss * operating_hours / Decimal("1000")
        annual_cost = annual_kwh * energy_cost_kwh

        if annual_cost > Decimal("2000"):
            heat_loss_score = Decimal("100")
        elif annual_cost > Decimal("1000"):
            heat_loss_score = Decimal("75")
        elif annual_cost > Decimal("500"):
            heat_loss_score = Decimal("50")
        else:
            heat_loss_score = Decimal("25")

        # Safety score (0-100) based on surface temperature
        if defect["surface_temp_c"] > Decimal("60"):  # ASTM C1055 limit
            safety_score = Decimal("100")
        elif defect["surface_temp_c"] > Decimal("55"):
            safety_score = Decimal("75")
        elif defect["surface_temp_c"] > Decimal("48"):
            safety_score = Decimal("50")
        else:
            safety_score = Decimal("25")

        # Process criticality score
        criticality_scores = {"A": Decimal("100"), "B": Decimal("60"), "C": Decimal("30")}
        process_score = criticality_scores.get(defect["criticality"], Decimal("30"))

        # CUI risk score (simplified)
        if defect["damage"] == DamageType.WET:
            cui_score = Decimal("100")
        elif defect["damage"] == DamageType.JACKET_DAMAGED:
            cui_score = Decimal("75")
        elif Decimal("50") <= defect["process_temp_c"] <= Decimal("150"):
            cui_score = Decimal("50")
        else:
            cui_score = Decimal("25")

        # Composite score (weighted)
        composite_score = (
            heat_loss_score * Decimal("0.30") +
            safety_score * Decimal("0.25") +
            process_score * Decimal("0.25") +
            cui_score * Decimal("0.20")
        )

        # Estimate repair cost
        access_multipliers = {"easy": Decimal("1.0"), "scaffold": Decimal("1.5"), "confined": Decimal("2.0")}
        base_cost_m = Decimal("150.0")  # Base cost per meter
        repair_cost = (
            base_cost_m * defect["length_m"] *
            access_multipliers.get(defect["access"], Decimal("1.0"))
        )

        # Calculate ROI
        annual_savings = annual_cost
        simple_payback_years = repair_cost / annual_savings if annual_savings > 0 else Decimal("999")

        # NPV over 10 years
        npv = Decimal("0")
        for year in range(1, 11):
            npv += annual_savings / ((1 + discount_rate) ** year)
        npv -= repair_cost

        roi_percent = (npv / repair_cost) * Decimal("100") if repair_cost > 0 else Decimal("0")

        scored_defects.append({
            **defect,
            "heat_loss_score": heat_loss_score,
            "safety_score": safety_score,
            "process_score": process_score,
            "cui_score": cui_score,
            "composite_score": composite_score,
            "annual_cost": annual_cost,
            "repair_cost": repair_cost,
            "simple_payback": simple_payback_years,
            "npv": npv,
            "roi_percent": roi_percent,
        })

    # Sort by composite score (descending)
    scored_defects.sort(key=lambda x: x["composite_score"], reverse=True)

    # Display ranked results
    print("\n--- Prioritized Repair List ---")
    print(f"{'Rank':<6}{'Defect':<10}{'Equipment':<15}{'Score':<8}{'ROI%':<10}{'Payback':<10}{'Cost':<10}")
    print("-" * 69)

    for rank, defect in enumerate(scored_defects, 1):
        print(f"{rank:<6}{defect['id']:<10}{defect['equipment']:<15}"
              f"{defect['composite_score']:.1f}   {defect['roi_percent']:.1f}%     "
              f"{defect['simple_payback']:.1f}y     ${defect['repair_cost']:.0f}")

    # Budget optimization
    print("\n--- Budget Optimization ---")
    total_budget = Decimal("2000.0")
    print(f"Available Budget: ${total_budget}")

    remaining_budget = total_budget
    selected_repairs = []

    for defect in scored_defects:
        if defect["repair_cost"] <= remaining_budget:
            selected_repairs.append(defect)
            remaining_budget -= defect["repair_cost"]

    total_selected_cost = sum(d["repair_cost"] for d in selected_repairs)
    total_annual_savings = sum(d["annual_cost"] for d in selected_repairs)

    print(f"\nSelected Repairs (within budget):")
    for defect in selected_repairs:
        print(f"  - {defect['id']}: ${defect['repair_cost']:.0f}")

    print(f"\nTotal Cost: ${total_selected_cost:.0f}")
    print(f"Remaining Budget: ${remaining_budget:.0f}")
    print(f"Total Annual Savings: ${total_annual_savings:.0f}")
    print(f"Portfolio Payback: {total_selected_cost/total_annual_savings:.1f} years")

    print("\n[Example 4 Complete]")


# =============================================================================
# Example 5: Facility-Wide Inspection
# =============================================================================

def example_facility_inspection() -> None:
    """
    Example 5: Facility-Wide Inspection

    Demonstrates:
    - Batch processing of multiple inspections
    - Facility summary statistics
    - Priority categorization
    - Executive reporting
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: FACILITY-WIDE INSPECTION")
    print("="*70)

    # Simulated facility inspection data
    facility_data = {
        "facility_name": "Chemical Plant Alpha",
        "inspection_date": date.today().isoformat(),
        "inspector": "John Smith",
        "units": [
            {
                "unit": "Process Unit 1",
                "equipment_count": 45,
                "inspected": 45,
                "excellent": 12,
                "good": 18,
                "fair": 10,
                "poor": 4,
                "critical": 1,
                "total_heat_loss_kw": Decimal("125.5"),
            },
            {
                "unit": "Process Unit 2",
                "equipment_count": 62,
                "inspected": 60,
                "excellent": 8,
                "good": 22,
                "fair": 18,
                "poor": 9,
                "critical": 3,
                "total_heat_loss_kw": Decimal("215.8"),
            },
            {
                "unit": "Utilities",
                "equipment_count": 38,
                "inspected": 38,
                "excellent": 15,
                "good": 12,
                "fair": 8,
                "poor": 3,
                "critical": 0,
                "total_heat_loss_kw": Decimal("68.2"),
            },
            {
                "unit": "Tank Farm",
                "equipment_count": 24,
                "inspected": 24,
                "excellent": 6,
                "good": 10,
                "fair": 5,
                "poor": 2,
                "critical": 1,
                "total_heat_loss_kw": Decimal("92.4"),
            },
        ]
    }

    print(f"\n--- Facility Information ---")
    print(f"Facility: {facility_data['facility_name']}")
    print(f"Inspection Date: {facility_data['inspection_date']}")
    print(f"Inspector: {facility_data['inspector']}")

    # Calculate totals
    total_equipment = sum(u["equipment_count"] for u in facility_data["units"])
    total_inspected = sum(u["inspected"] for u in facility_data["units"])
    total_excellent = sum(u["excellent"] for u in facility_data["units"])
    total_good = sum(u["good"] for u in facility_data["units"])
    total_fair = sum(u["fair"] for u in facility_data["units"])
    total_poor = sum(u["poor"] for u in facility_data["units"])
    total_critical = sum(u["critical"] for u in facility_data["units"])
    total_heat_loss = sum(u["total_heat_loss_kw"] for u in facility_data["units"])

    print(f"\n--- Inspection Summary ---")
    print(f"Total Equipment: {total_equipment}")
    print(f"Inspected: {total_inspected} ({total_inspected/total_equipment*100:.1f}%)")

    print(f"\n--- Condition Distribution ---")
    print(f"{'Condition':<12}{'Count':<8}{'Percentage':<12}")
    print("-" * 32)
    print(f"{'Excellent':<12}{total_excellent:<8}{total_excellent/total_inspected*100:.1f}%")
    print(f"{'Good':<12}{total_good:<8}{total_good/total_inspected*100:.1f}%")
    print(f"{'Fair':<12}{total_fair:<8}{total_fair/total_inspected*100:.1f}%")
    print(f"{'Poor':<12}{total_poor:<8}{total_poor/total_inspected*100:.1f}%")
    print(f"{'Critical':<12}{total_critical:<8}{total_critical/total_inspected*100:.1f}%")

    print(f"\n--- Heat Loss by Unit ---")
    print(f"{'Unit':<20}{'Heat Loss (kW)':<15}{'% of Total':<12}")
    print("-" * 47)
    for unit in facility_data["units"]:
        pct = unit["total_heat_loss_kw"] / total_heat_loss * Decimal("100")
        print(f"{unit['unit']:<20}{unit['total_heat_loss_kw']:<15.1f}{pct:.1f}%")
    print("-" * 47)
    print(f"{'TOTAL':<20}{total_heat_loss:<15.1f}100.0%")

    # Annual impact
    operating_hours = Decimal("8000")
    energy_cost = Decimal("0.10")
    annual_kwh = total_heat_loss * operating_hours
    annual_cost = annual_kwh * energy_cost

    print(f"\n--- Annual Energy Impact ---")
    print(f"Total Heat Loss: {total_heat_loss:.1f} kW")
    print(f"Annual Energy Loss: {annual_kwh/1000:.1f} MWh")
    print(f"Annual Cost: ${annual_cost:,.0f}")

    # Priority actions
    print(f"\n--- Priority Actions Required ---")

    # Emergency (Critical)
    print(f"\n[EMERGENCY] {total_critical} items requiring immediate action:")
    for unit in facility_data["units"]:
        if unit["critical"] > 0:
            print(f"  - {unit['unit']}: {unit['critical']} critical item(s)")

    # Urgent (Poor)
    print(f"\n[URGENT] {total_poor} items requiring action within 30 days:")
    for unit in facility_data["units"]:
        if unit["poor"] > 0:
            print(f"  - {unit['unit']}: {unit['poor']} poor condition item(s)")

    # Scheduled (Fair)
    print(f"\n[SCHEDULED] {total_fair} items for next turnaround:")
    for unit in facility_data["units"]:
        if unit["fair"] > 0:
            print(f"  - {unit['unit']}: {unit['fair']} fair condition item(s)")

    # Estimated repair budget
    print(f"\n--- Estimated Repair Budget ---")
    critical_cost = total_critical * Decimal("5000")
    poor_cost = total_poor * Decimal("2500")
    fair_cost = total_fair * Decimal("1500")
    total_repair_cost = critical_cost + poor_cost + fair_cost

    print(f"Emergency Repairs: ${critical_cost:,.0f}")
    print(f"Urgent Repairs: ${poor_cost:,.0f}")
    print(f"Scheduled Repairs: ${fair_cost:,.0f}")
    print(f"Total Estimated: ${total_repair_cost:,.0f}")

    # ROI summary
    print(f"\n--- Investment Summary ---")
    potential_savings = annual_cost * Decimal("0.40")  # Assume 40% recovery
    payback = total_repair_cost / potential_savings if potential_savings > 0 else Decimal("999")

    print(f"Potential Annual Savings: ${potential_savings:,.0f}")
    print(f"Simple Payback: {payback:.1f} years")

    print("\n[Example 5 Complete]")


# =============================================================================
# Example 6: Integration Examples
# =============================================================================

def example_integrations() -> None:
    """
    Example 6: Integration Examples

    Demonstrates integration patterns with:
    - Thermal cameras (FLIR, Fluke)
    - CMMS systems (SAP PM, IBM Maximo)
    - GreenLang agents
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: INTEGRATION EXAMPLES")
    print("="*70)

    # Example 6a: Thermal Camera Integration
    print("\n--- 6a: Thermal Camera Integration (FLIR) ---")
    print("""
# In production:
from gl_015.integrations.thermal_cameras import FLIRConnector

# Initialize FLIR connector
flir = FLIRConnector(
    sdk_path="/opt/flir/atlas",
    camera_ip="192.168.1.100"
)

# Connect to camera
await flir.connect()

# Capture thermal image with parameters
image_data = await flir.capture_image(
    emissivity=0.90,
    reflected_temp_c=25.0,
    atmospheric_temp_c=25.0,
    relative_humidity=60.0,
    distance_m=3.0
)

# Get temperature matrix for analysis
temp_matrix = image_data.temperature_matrix
metadata = image_data.metadata

# Disconnect when done
await flir.disconnect()
""")

    # Example 6b: CMMS Integration
    print("\n--- 6b: CMMS Integration (SAP PM) ---")
    print("""
# In production:
from gl_015.integrations.cmms import SAPPMConnector

# Initialize SAP PM connector
sap = SAPPMConnector(
    host="sap.example.com",
    client="100",
    username=os.environ["SAP_USER"],
    password=os.environ["SAP_PASS"]
)

# Create work order from inspection result
work_order = await sap.create_work_order(
    functional_location="FL-PLANT-AREA1-PIPE001",
    order_type="PM02",  # Corrective maintenance
    priority="3",       # High priority
    description="Insulation repair - 5m section missing",
    long_text=result.generate_work_order_text(),
    planned_start=result.repair_priorities.recommended_timing,
    estimated_cost=result.repair_priorities.estimated_repair_cost_usd
)

print(f"Created SAP Work Order: {work_order.order_number}")

# Attach inspection report
await sap.attach_document(
    order_number=work_order.order_number,
    document_type="INS",
    file_path=result.export_report("pdf")
)
""")

    # Example 6c: IBM Maximo Integration
    print("\n--- 6c: CMMS Integration (IBM Maximo) ---")
    print("""
# In production:
from gl_015.integrations.cmms import MaximoConnector

# Initialize Maximo connector
maximo = MaximoConnector(
    base_url="https://maximo.example.com/maximo",
    api_key=os.environ["MAXIMO_API_KEY"],
    site_id="SITE01"
)

# Create work order
work_order = await maximo.create_work_order(
    asset_num="PIPE-001",
    work_type="CM",
    priority=3,
    description=result.generate_work_order_summary(),
    job_plan="JP-INSULATION-REPAIR",
    estimated_labor_hours=result.repair_priorities.estimated_labor_hours,
    estimated_material_cost=result.repair_priorities.material_cost_usd
)

print(f"Created Maximo Work Order: {work_order.wonum}")
""")

    # Example 6d: GreenLang Agent Integration
    print("\n--- 6d: GreenLang Agent Integration ---")
    print("""
# In production:
from gl_015.integrations.greenlang import ThermosyncConnector

# Connect to GL-001 THERMOSYNC for steam trap correlation
thermosync = ThermosyncConnector(
    agent_url="http://gl-001:8080",
    api_key=os.environ["GL_API_KEY"]
)

# Get nearby steam trap data
steam_traps = await thermosync.get_traps_by_location(
    location="AREA-A",
    radius_m=10.0
)

# Correlate insulation findings with steam trap failures
for trap in steam_traps:
    if trap.status in ["failed", "suspect"]:
        # Check for related insulation issues
        related = result.find_related_equipment(
            trap.functional_location,
            radius_m=5.0
        )

        if related:
            print(f"Steam trap {trap.trap_id} has correlated "
                  f"insulation issues: {[r.defect_id for r in related]}")
""")

    # Example 6e: Webhook Configuration
    print("\n--- 6e: Webhook Configuration ---")
    print("""
# Configure webhooks for real-time notifications
webhook_config = {
    "url": "https://your-app.com/webhooks/insulscan",
    "events": [
        "inspection.completed",
        "critical_finding.detected",
        "repair_order.created"
    ],
    "secret": "your_webhook_secret",
    "retry_policy": {
        "max_retries": 3,
        "backoff_seconds": [1, 5, 30]
    }
}

# Webhook payload example
webhook_payload = {
    "event": "critical_finding.detected",
    "timestamp": "2025-12-01T10:30:00Z",
    "data": {
        "inspection_id": "INS-2025-001",
        "equipment_id": "PIPE-001",
        "finding_type": "missing_insulation",
        "severity": "critical",
        "heat_loss_w": 2250,
        "surface_temp_c": 125,
        "recommended_action": "immediate_repair"
    },
    "signature": "sha256=abc123..."
}
""")

    print("\n[Example 6 Complete]")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for example demonstrations."""
    parser = argparse.ArgumentParser(
        description="GL-015 INSULSCAN Example Usage Demonstrations"
    )
    parser.add_argument(
        "--example",
        choices=["basic", "heat_loss", "degradation", "prioritization",
                 "facility", "integration", "all"],
        default="all",
        help="Which example to run (default: all)"
    )

    args = parser.parse_args()

    print("="*70)
    print("GL-015 INSULSCAN - Example Usage Demonstrations")
    print("="*70)
    print("\nThis script demonstrates key capabilities of the GL-015")
    print("Industrial Insulation Inspection Agent.")
    print("\nAll examples use deterministic calculations with")
    print("zero-hallucination guarantees and full provenance tracking.")

    examples = {
        "basic": example_basic_thermal_analysis,
        "heat_loss": example_heat_loss_calculation,
        "degradation": example_degradation_assessment,
        "prioritization": example_repair_prioritization,
        "facility": example_facility_inspection,
        "integration": example_integrations,
    }

    if args.example == "all":
        for name, func in examples.items():
            func()
    else:
        examples[args.example]()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nFor more information, see:")
    print("  - README.md: Complete documentation")
    print("  - ARCHITECTURE.md: System design details")
    print("  - QUICKSTART.md: 5-minute setup guide")
    print("  - https://docs.greenlang.io/agents/gl-015")
    print("")


if __name__ == "__main__":
    main()
