"""
Steam Quality Module - Wet Steam and Dryness Fraction Calculations

This module provides deterministic calculations for steam quality (dryness
fraction) in two-phase (wet steam) systems, including:
- Wet steam property calculations
- State validation
- Quality inference from process data

Key Concepts:
- Quality (x): Mass fraction of vapor in two-phase mixture (0 = all liquid, 1 = all vapor)
- Wet steam: Two-phase mixture where 0 < x < 1
- Saturation: Boundary between liquid and vapor phases

Author: GL-CalculatorEngineer
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import hashlib
import json
import math


@dataclass
class ValidationResult:
    """
    Result of steam state validation.
    """
    is_valid: bool
    state_description: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Detailed validation checks
    temperature_check: bool = True
    pressure_check: bool = True
    quality_check: bool = True
    consistency_check: bool = True

    # Suggested corrections
    suggested_state: Optional[str] = None
    suggested_quality: Optional[float] = None

    # Provenance
    provenance_hash: str = ""


@dataclass
class InferredQuality:
    """
    Result of quality inference from process data.
    """
    # Inferred quality
    quality_x: float               # Best estimate of dryness fraction
    quality_uncertainty: float     # Uncertainty in quality estimate

    # Confidence
    confidence_level: float        # 0-1 confidence in inference
    inference_method: str          # Method used for inference

    # Supporting evidence
    evidence: List[str]            # Reasoning for inference
    alternative_estimates: List[Tuple[float, str]]  # (value, method) alternatives

    # State classification
    inferred_state: str            # "subcooled", "wet_steam", "superheated"

    # Provenance
    provenance_hash: str = ""


class QualityInferenceMethod(Enum):
    """Methods for inferring steam quality."""
    ENTHALPY = "enthalpy"           # From h = hf + x*hfg
    ENTROPY = "entropy"             # From s = sf + x*sfg
    SPECIFIC_VOLUME = "volume"      # From v = vf + x*(vg-vf)
    TEMPERATURE = "temperature"     # From T vs Tsat comparison
    PROCESS = "process"             # From process type (throttling, etc.)


# =============================================================================
# WET STEAM PROPERTY CALCULATIONS
# =============================================================================

def compute_wet_steam_enthalpy(
    pressure_kpa: float,
    dryness_x: float,
) -> float:
    """
    Compute specific enthalpy of wet steam (two-phase mixture).

    DETERMINISTIC: Same inputs always produce same output.

    Formula: h = hf + x * hfg = hf + x * (hg - hf)

    Args:
        pressure_kpa: Pressure in kPa
        dryness_x: Dryness fraction (quality), 0 <= x <= 1

    Returns:
        Specific enthalpy in kJ/kg

    Raises:
        ValueError: If quality is outside [0, 1] or pressure invalid
    """
    from .iapws_if97 import (
        kpa_to_mpa,
        region4_saturation_properties,
        region4_mixture_enthalpy,
    )

    # Validate and clamp quality
    if dryness_x < 0:
        raise ValueError(f"Quality cannot be negative: {dryness_x}")
    if dryness_x > 1:
        raise ValueError(f"Quality cannot exceed 1: {dryness_x}")

    P_mpa = kpa_to_mpa(pressure_kpa)

    # Use IAPWS-IF97 mixture enthalpy
    h = region4_mixture_enthalpy(P_mpa, dryness_x)

    return h


def compute_wet_steam_entropy(
    pressure_kpa: float,
    dryness_x: float,
) -> float:
    """
    Compute specific entropy of wet steam (two-phase mixture).

    DETERMINISTIC: Same inputs always produce same output.

    Formula: s = sf + x * sfg = sf + x * (sg - sf)

    Args:
        pressure_kpa: Pressure in kPa
        dryness_x: Dryness fraction (quality), 0 <= x <= 1

    Returns:
        Specific entropy in kJ/(kg*K)

    Raises:
        ValueError: If quality is outside [0, 1] or pressure invalid
    """
    from .iapws_if97 import (
        kpa_to_mpa,
        region4_mixture_entropy,
    )

    # Validate quality
    if dryness_x < 0:
        raise ValueError(f"Quality cannot be negative: {dryness_x}")
    if dryness_x > 1:
        raise ValueError(f"Quality cannot exceed 1: {dryness_x}")

    P_mpa = kpa_to_mpa(pressure_kpa)

    # Use IAPWS-IF97 mixture entropy
    s = region4_mixture_entropy(P_mpa, dryness_x)

    return s


def compute_wet_steam_specific_volume(
    pressure_kpa: float,
    dryness_x: float,
) -> float:
    """
    Compute specific volume of wet steam (two-phase mixture).

    DETERMINISTIC: Same inputs always produce same output.

    Formula: v = vf + x * (vg - vf)

    Args:
        pressure_kpa: Pressure in kPa
        dryness_x: Dryness fraction (quality), 0 <= x <= 1

    Returns:
        Specific volume in m^3/kg

    Raises:
        ValueError: If quality is outside [0, 1] or pressure invalid
    """
    from .iapws_if97 import (
        kpa_to_mpa,
        region4_mixture_specific_volume,
    )

    # Validate quality
    if dryness_x < 0:
        raise ValueError(f"Quality cannot be negative: {dryness_x}")
    if dryness_x > 1:
        raise ValueError(f"Quality cannot exceed 1: {dryness_x}")

    P_mpa = kpa_to_mpa(pressure_kpa)

    # Use IAPWS-IF97 mixture specific volume
    v = region4_mixture_specific_volume(P_mpa, dryness_x)

    return v


# =============================================================================
# STATE VALIDATION
# =============================================================================

def validate_steam_state(
    pressure_kpa: float,
    temperature_c: float,
    quality_x: Optional[float] = None,
    enthalpy_kj_kg: Optional[float] = None,
    tolerance_c: float = 1.0,
) -> ValidationResult:
    """
    Validate steam state for physical consistency.

    DETERMINISTIC: Same inputs always produce same output.

    Validation checks:
    1. Temperature vs saturation temperature
    2. Quality bounds [0, 1]
    3. Enthalpy consistency with quality
    4. Physical state consistency

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius
        quality_x: Optional steam quality (dryness fraction)
        enthalpy_kj_kg: Optional specific enthalpy
        tolerance_c: Temperature tolerance for saturation detection

    Returns:
        ValidationResult with detailed validation status
    """
    from .iapws_if97 import (
        kpa_to_mpa,
        celsius_to_kelvin,
        kelvin_to_celsius,
        get_saturation_temperature,
        region4_saturation_properties,
        IF97_CONSTANTS,
        REGION_BOUNDARIES,
    )

    errors = []
    warnings = []
    temperature_check = True
    pressure_check = True
    quality_check = True
    consistency_check = True

    # Convert units
    P_mpa = kpa_to_mpa(pressure_kpa)
    T_k = celsius_to_kelvin(temperature_c)

    # ==========================================================================
    # Pressure validation
    # ==========================================================================
    P_MIN = REGION_BOUNDARIES["P_MIN"]
    P_MAX = REGION_BOUNDARIES["P_MAX_1_2"]
    P_CRIT = IF97_CONSTANTS["P_CRIT"]

    if P_mpa < P_MIN:
        errors.append(f"Pressure {pressure_kpa:.2f} kPa below minimum ({P_MIN*1000:.4f} kPa)")
        pressure_check = False
    elif P_mpa > P_MAX:
        errors.append(f"Pressure {pressure_kpa:.2f} kPa exceeds maximum ({P_MAX*1000:.0f} kPa)")
        pressure_check = False

    # ==========================================================================
    # Temperature validation
    # ==========================================================================
    T_MIN = REGION_BOUNDARIES["T_MIN"]
    T_MAX = REGION_BOUNDARIES["T_MAX_2"]

    if T_k < T_MIN:
        errors.append(f"Temperature {temperature_c:.2f} C below minimum (0 C)")
        temperature_check = False
    elif T_k > T_MAX:
        errors.append(f"Temperature {temperature_c:.2f} C exceeds maximum ({T_MAX-273.15:.0f} C)")
        temperature_check = False

    # ==========================================================================
    # Saturation comparison
    # ==========================================================================
    try:
        T_sat_k = get_saturation_temperature(P_mpa)
        T_sat_c = kelvin_to_celsius(T_sat_k)

        delta_T = temperature_c - T_sat_c

        if delta_T < -tolerance_c:
            # Subcooled liquid
            state_description = f"Subcooled liquid ({abs(delta_T):.1f} C below saturation)"
            suggested_state = "compressed_liquid"

            # Quality check for subcooled
            if quality_x is not None and quality_x > 0:
                warnings.append(
                    f"Quality {quality_x:.3f} provided but T < Tsat indicates subcooled liquid. "
                    f"Quality should be undefined or 0."
                )
                consistency_check = False

        elif abs(delta_T) <= tolerance_c:
            # At or near saturation - wet steam region
            state_description = f"Saturated / wet steam (T within {tolerance_c} C of Tsat)"
            suggested_state = "wet_steam"

            # Quality is required for wet steam
            if quality_x is None:
                warnings.append(
                    "At saturation temperature - quality (x) required to fully define state"
                )

        else:
            # Superheated vapor
            state_description = f"Superheated vapor ({delta_T:.1f} C above saturation)"
            suggested_state = "superheated_vapor"

            # Quality check for superheated
            if quality_x is not None and quality_x < 1:
                warnings.append(
                    f"Quality {quality_x:.3f} provided but T > Tsat indicates superheated vapor. "
                    f"Quality should be undefined or 1."
                )
                consistency_check = False

    except ValueError:
        # Above critical pressure
        if P_mpa > P_CRIT:
            state_description = "Supercritical fluid (above critical pressure)"
            suggested_state = "supercritical"
            if quality_x is not None:
                warnings.append("Quality is undefined for supercritical fluid")
        else:
            state_description = "Unknown state (outside saturation range)"
            suggested_state = None

    # ==========================================================================
    # Quality validation
    # ==========================================================================
    suggested_quality = None

    if quality_x is not None:
        if quality_x < 0:
            errors.append(f"Quality {quality_x:.4f} is negative (must be >= 0)")
            quality_check = False
            suggested_quality = 0.0
        elif quality_x > 1:
            errors.append(f"Quality {quality_x:.4f} exceeds 1 (must be <= 1)")
            quality_check = False
            suggested_quality = 1.0
        elif quality_x < 0.001:
            warnings.append(
                f"Quality {quality_x:.4f} very low - essentially saturated liquid"
            )
        elif quality_x > 0.999:
            warnings.append(
                f"Quality {quality_x:.4f} very high - essentially saturated vapor"
            )

    # ==========================================================================
    # Enthalpy consistency check
    # ==========================================================================
    if enthalpy_kj_kg is not None and quality_x is not None:
        try:
            sat = region4_saturation_properties(P_mpa)
            h_calculated = sat.hf + quality_x * sat.hfg

            h_error = abs(enthalpy_kj_kg - h_calculated)
            h_error_percent = h_error / h_calculated * 100 if h_calculated > 0 else 0

            if h_error_percent > 5:
                warnings.append(
                    f"Enthalpy {enthalpy_kj_kg:.1f} kJ/kg inconsistent with "
                    f"quality {quality_x:.3f} (expected {h_calculated:.1f} kJ/kg, "
                    f"error {h_error_percent:.1f}%)"
                )
                consistency_check = False

                # Suggest corrected quality
                suggested_quality = (enthalpy_kj_kg - sat.hf) / sat.hfg
                suggested_quality = max(0, min(1, suggested_quality))

        except ValueError:
            pass  # Can't check consistency outside saturation range

    # ==========================================================================
    # Create provenance hash
    # ==========================================================================
    provenance_hash = _compute_provenance({
        "pressure_kpa": pressure_kpa,
        "temperature_c": temperature_c,
        "quality_x": quality_x,
        "enthalpy_kj_kg": enthalpy_kj_kg,
        "is_valid": len(errors) == 0,
    })

    return ValidationResult(
        is_valid=len(errors) == 0,
        state_description=state_description,
        warnings=warnings,
        errors=errors,
        temperature_check=temperature_check,
        pressure_check=pressure_check,
        quality_check=quality_check,
        consistency_check=consistency_check,
        suggested_state=suggested_state,
        suggested_quality=suggested_quality,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# QUALITY INFERENCE
# =============================================================================

def infer_quality_from_process(
    process_data: Dict[str, Any],
    confidence_threshold: float = 0.7,
) -> InferredQuality:
    """
    Infer steam quality from process measurements and context.

    DETERMINISTIC: Same inputs always produce same output.

    Uses multiple methods to estimate quality when direct measurement
    is not available:
    1. Enthalpy-based: x = (h - hf) / hfg
    2. Entropy-based: x = (s - sf) / sfg
    3. Temperature-based: Compare T to Tsat
    4. Process-based: Use known process characteristics

    Args:
        process_data: Dictionary containing available measurements:
            - pressure_kpa: Pressure in kPa (required)
            - temperature_c: Temperature in Celsius (optional)
            - enthalpy_kj_kg: Specific enthalpy (optional)
            - entropy_kj_kgk: Specific entropy (optional)
            - specific_volume_m3_kg: Specific volume (optional)
            - process_type: Type of process (optional)
            - upstream_quality: Quality of upstream steam (optional)
        confidence_threshold: Minimum confidence for valid inference

    Returns:
        InferredQuality with best estimate and uncertainty
    """
    from .iapws_if97 import (
        kpa_to_mpa,
        celsius_to_kelvin,
        kelvin_to_celsius,
        get_saturation_temperature,
        region4_saturation_properties,
    )

    pressure_kpa = process_data.get("pressure_kpa")
    if pressure_kpa is None:
        raise ValueError("pressure_kpa is required for quality inference")

    P_mpa = kpa_to_mpa(pressure_kpa)

    # Get saturation properties
    try:
        sat = region4_saturation_properties(P_mpa)
        T_sat_c = kelvin_to_celsius(sat.temperature_k)
    except ValueError:
        # Above critical - quality is undefined
        return InferredQuality(
            quality_x=float('nan'),
            quality_uncertainty=float('nan'),
            confidence_level=0.0,
            inference_method="none",
            evidence=["Pressure above critical - quality undefined"],
            alternative_estimates=[],
            inferred_state="supercritical",
            provenance_hash=_compute_provenance(process_data),
        )

    # Collect quality estimates from different methods
    estimates = []
    evidence = []

    # ==========================================================================
    # Method 1: Temperature-based
    # ==========================================================================
    temperature_c = process_data.get("temperature_c")
    if temperature_c is not None:
        delta_T = temperature_c - T_sat_c

        if delta_T < -1.0:
            # Subcooled liquid
            x_temp = 0.0
            confidence_temp = 0.9
            evidence.append(f"T={temperature_c:.1f}C < Tsat={T_sat_c:.1f}C: subcooled liquid")
            estimates.append((x_temp, confidence_temp, QualityInferenceMethod.TEMPERATURE))

        elif delta_T > 1.0:
            # Superheated vapor
            x_temp = 1.0
            confidence_temp = 0.9
            evidence.append(f"T={temperature_c:.1f}C > Tsat={T_sat_c:.1f}C: superheated vapor")
            estimates.append((x_temp, confidence_temp, QualityInferenceMethod.TEMPERATURE))

        else:
            # At saturation - cannot determine quality from T alone
            evidence.append(f"T={temperature_c:.1f}C at Tsat: wet steam, quality unknown from T")

    # ==========================================================================
    # Method 2: Enthalpy-based (most reliable)
    # ==========================================================================
    enthalpy_kj_kg = process_data.get("enthalpy_kj_kg")
    if enthalpy_kj_kg is not None:
        hf = sat.hf
        hfg = sat.hfg
        hg = sat.hg

        if enthalpy_kj_kg < hf:
            x_h = 0.0
            confidence_h = 0.95
            evidence.append(f"h={enthalpy_kj_kg:.1f} < hf={hf:.1f}: subcooled liquid")
        elif enthalpy_kj_kg > hg:
            x_h = 1.0
            confidence_h = 0.95
            evidence.append(f"h={enthalpy_kj_kg:.1f} > hg={hg:.1f}: superheated vapor")
        else:
            x_h = (enthalpy_kj_kg - hf) / hfg
            x_h = max(0, min(1, x_h))  # Clamp to [0, 1]
            confidence_h = 0.95  # Enthalpy is most reliable
            evidence.append(f"x = (h - hf) / hfg = ({enthalpy_kj_kg:.1f} - {hf:.1f}) / {hfg:.1f} = {x_h:.4f}")

        estimates.append((x_h, confidence_h, QualityInferenceMethod.ENTHALPY))

    # ==========================================================================
    # Method 3: Entropy-based
    # ==========================================================================
    entropy_kj_kgk = process_data.get("entropy_kj_kgk")
    if entropy_kj_kgk is not None:
        sf = sat.sf
        sfg = sat.sfg
        sg = sat.sg

        if entropy_kj_kgk < sf:
            x_s = 0.0
            confidence_s = 0.85
            evidence.append(f"s={entropy_kj_kgk:.4f} < sf={sf:.4f}: subcooled liquid")
        elif entropy_kj_kgk > sg:
            x_s = 1.0
            confidence_s = 0.85
            evidence.append(f"s={entropy_kj_kgk:.4f} > sg={sg:.4f}: superheated vapor")
        else:
            x_s = (entropy_kj_kgk - sf) / sfg
            x_s = max(0, min(1, x_s))
            confidence_s = 0.85
            evidence.append(f"x = (s - sf) / sfg = ({entropy_kj_kgk:.4f} - {sf:.4f}) / {sfg:.4f} = {x_s:.4f}")

        estimates.append((x_s, confidence_s, QualityInferenceMethod.ENTROPY))

    # ==========================================================================
    # Method 4: Specific volume-based
    # ==========================================================================
    specific_volume = process_data.get("specific_volume_m3_kg")
    if specific_volume is not None:
        vf = sat.vf
        vg = sat.vg
        vfg = vg - vf

        if specific_volume < vf:
            x_v = 0.0
            confidence_v = 0.7
            evidence.append(f"v={specific_volume:.6f} < vf={vf:.6f}: compressed liquid")
        elif specific_volume > vg:
            x_v = 1.0
            confidence_v = 0.7
            evidence.append(f"v={specific_volume:.6f} > vg={vg:.6f}: superheated vapor")
        else:
            x_v = (specific_volume - vf) / vfg
            x_v = max(0, min(1, x_v))
            confidence_v = 0.75
            evidence.append(f"x = (v - vf) / vfg = ({specific_volume:.6f} - {vf:.6f}) / {vfg:.6f} = {x_v:.4f}")

        estimates.append((x_v, confidence_v, QualityInferenceMethod.SPECIFIC_VOLUME))

    # ==========================================================================
    # Method 5: Process-based
    # ==========================================================================
    process_type = process_data.get("process_type", "").lower()
    upstream_quality = process_data.get("upstream_quality")

    if process_type == "throttling":
        # Throttling is isenthalpic - quality may increase
        if upstream_quality is not None:
            # After throttling, enthalpy is conserved
            evidence.append(f"Throttling process: isenthalpic (h constant)")
            # Quality would be determined by enthalpy method

    elif process_type == "desuperheating":
        # Desuperheating reduces superheat
        x_p = 1.0  # Still saturated vapor
        confidence_p = 0.7
        evidence.append("Desuperheating: output near saturation")
        estimates.append((x_p, confidence_p, QualityInferenceMethod.PROCESS))

    elif process_type == "flash":
        # Flash steam generation
        flash_pressure_ratio = process_data.get("flash_pressure_ratio", 1.0)
        if flash_pressure_ratio < 1.0:
            # Approximate flash steam quality
            # x_flash ~ (hf_high - hf_low) / hfg_low
            evidence.append(f"Flash process: quality depends on pressure ratio")

    elif process_type == "condensate":
        # Condensate is saturated liquid
        x_p = 0.0
        confidence_p = 0.9
        evidence.append("Condensate stream: saturated liquid (x = 0)")
        estimates.append((x_p, confidence_p, QualityInferenceMethod.PROCESS))

    # ==========================================================================
    # Combine estimates
    # ==========================================================================
    if not estimates:
        # No methods available - return uncertain result
        return InferredQuality(
            quality_x=0.5,  # Default uncertain value
            quality_uncertainty=0.5,
            confidence_level=0.0,
            inference_method="none",
            evidence=["Insufficient data for quality inference"],
            alternative_estimates=[],
            inferred_state="unknown",
            provenance_hash=_compute_provenance(process_data),
        )

    # Weighted average of estimates
    total_weight = sum(conf for _, conf, _ in estimates)
    weighted_sum = sum(x * conf for x, conf, _ in estimates)
    best_estimate = weighted_sum / total_weight

    # Uncertainty from spread of estimates
    if len(estimates) > 1:
        variance = sum(
            conf * (x - best_estimate) ** 2
            for x, conf, _ in estimates
        ) / total_weight
        uncertainty = math.sqrt(variance)
    else:
        # Single estimate - use method-specific uncertainty
        uncertainty = 1 - estimates[0][1]  # Higher confidence = lower uncertainty

    # Best method (highest confidence)
    best_method = max(estimates, key=lambda x: x[1])
    inference_method = best_method[2].value

    # Overall confidence
    confidence_level = min(total_weight / len(estimates), 0.99)

    # Determine state
    if best_estimate < 0.001:
        inferred_state = "subcooled"
    elif best_estimate > 0.999:
        inferred_state = "superheated"
    else:
        inferred_state = "wet_steam"

    # Alternative estimates
    alternative_estimates = [
        (x, method.value) for x, _, method in estimates
    ]

    # Provenance
    provenance_hash = _compute_provenance({
        **process_data,
        "best_estimate": best_estimate,
        "uncertainty": uncertainty,
    })

    return InferredQuality(
        quality_x=best_estimate,
        quality_uncertainty=uncertainty,
        confidence_level=confidence_level,
        inference_method=inference_method,
        evidence=evidence,
        alternative_estimates=alternative_estimates,
        inferred_state=inferred_state,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clamp_quality(quality_x: float, warn: bool = True) -> float:
    """
    Clamp quality to valid range [0, 1].

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        quality_x: Quality value to clamp
        warn: Whether to print warning for clamped values

    Returns:
        Clamped quality value
    """
    if quality_x < 0:
        if warn:
            print(f"Warning: Quality {quality_x:.4f} clamped to 0")
        return 0.0
    elif quality_x > 1:
        if warn:
            print(f"Warning: Quality {quality_x:.4f} clamped to 1")
        return 1.0
    return quality_x


def quality_from_enthalpy(
    pressure_kpa: float,
    enthalpy_kj_kg: float,
) -> Tuple[float, bool]:
    """
    Calculate quality from pressure and enthalpy.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_kpa: Pressure in kPa
        enthalpy_kj_kg: Specific enthalpy in kJ/kg

    Returns:
        Tuple of (quality, is_two_phase)
        - quality is clamped to [0, 1]
        - is_two_phase is False if outside saturation region
    """
    from .iapws_if97 import kpa_to_mpa, region4_saturation_properties

    P_mpa = kpa_to_mpa(pressure_kpa)

    try:
        sat = region4_saturation_properties(P_mpa)
    except ValueError:
        return (float('nan'), False)

    if enthalpy_kj_kg < sat.hf:
        return (0.0, False)  # Subcooled
    elif enthalpy_kj_kg > sat.hg:
        return (1.0, False)  # Superheated
    else:
        x = (enthalpy_kj_kg - sat.hf) / sat.hfg
        return (x, True)


def is_near_zero_flow(mass_flow_kg_s: float, threshold: float = 0.001) -> bool:
    """
    Check if mass flow is near zero (below measurement threshold).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        mass_flow_kg_s: Mass flow rate in kg/s
        threshold: Threshold for "near zero" in kg/s

    Returns:
        True if flow is below threshold
    """
    return abs(mass_flow_kg_s) < threshold


def _compute_provenance(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 provenance hash.

    DETERMINISTIC: Same inputs always produce same hash.
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()
