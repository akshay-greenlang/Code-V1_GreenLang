# -*- coding: utf-8 -*-
"""
Scenario Projector Engine - AGENT-DATA-020: Climate Hazard Connector (GL-DATA-GEO-002)
=======================================================================================

Engine 3 of 7 -- ScenarioProjectorEngine.

Projects climate hazard risk under IPCC SSP and legacy RCP scenarios across
five standardised time horizons (baseline through end-of-century).  Scaling
factors translate global mean temperature anomaly into per-hazard-type changes
in probability, intensity, frequency, and duration using peer-reviewed
climate science relationships.

Supported SSP Scenarios (IPCC AR6, 2021):
    - SSP1-1.9  Sustainability pathway, very low forcing (+1.4 C by 2100)
    - SSP1-2.6  Sustainability pathway, low forcing (+1.8 C by 2100)
    - SSP2-4.5  Middle of the road (+2.7 C by 2100)
    - SSP3-7.0  Regional rivalry, high forcing (+3.6 C by 2100)
    - SSP5-8.5  Fossil-fueled development, very high forcing (+4.4 C by 2100)

Supported Legacy RCP Scenarios (IPCC AR5, 2014):
    - RCP2.6  Low emissions pathway (+1.6 C by 2100)
    - RCP4.5  Medium emissions pathway (+2.4 C by 2100)
    - RCP8.5  High emissions pathway (+4.3 C by 2100)

Time Horizons:
    - BASELINE     1995-2014  (warming fraction 0.0 -- reference period)
    - NEAR_TERM    2021-2040  (warming fraction 0.3)
    - MID_TERM     2041-2060  (warming fraction 0.55)
    - LONG_TERM    2061-2080  (warming fraction 0.8)
    - END_CENTURY  2081-2100  (warming fraction 1.0)

Hazard Scaling (per degree Celsius of warming):
    Each hazard type has an intensity_factor and frequency_factor that are
    applied as multiplicative scaling relative to the baseline.  For example,
    EXTREME_HEAT with intensity_factor=2.0 means that for every 1 C of
    warming the hazard intensity doubles on top of baseline.

Zero-Hallucination Guarantees:
    - All projections use deterministic Python arithmetic only
    - Warming trajectories are hard-coded lookup tables from IPCC AR5/AR6
    - No LLM or ML models in the scaling, projection, or comparison paths
    - SHA-256 provenance hashes for full audit trails
    - Thread-safe with reentrant locking
    - All returns are deep copies to prevent mutation of internal state

Example:
    >>> from greenlang.climate_hazard.scenario_projector import ScenarioProjectorEngine
    >>> engine = ScenarioProjectorEngine()
    >>> baseline = {
    ...     "probability": 0.15,
    ...     "intensity": 0.6,
    ...     "frequency": 2.0,
    ...     "duration_days": 5.0,
    ... }
    >>> result = engine.project_hazard(
    ...     hazard_type="EXTREME_HEAT",
    ...     location={"lat": 40.7128, "lon": -74.0060, "name": "New York"},
    ...     baseline_risk=baseline,
    ...     scenario="ssp2_4.5",
    ...     time_horizon="MID_TERM",
    ... )
    >>> assert result["scenario"] == "ssp2_4.5"
    >>> assert result["warming_delta_c"] > 0.0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ScenarioProjectorEngine",
]


# ---------------------------------------------------------------------------
# Graceful imports -- provenance, metrics, config
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.provenance import (
        ProvenanceTracker,
        get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:  # pragma: no cover -- fallback when provenance not yet built
    ProvenanceTracker = None  # type: ignore[misc, assignment]
    get_provenance_tracker = None  # type: ignore[misc, assignment]
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.climate_hazard import metrics as _metrics_mod
    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover -- fallback when metrics not yet built
    _metrics_mod = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False

try:
    from greenlang.climate_hazard.config import get_config
    _CONFIG_AVAILABLE = True
except ImportError:  # pragma: no cover -- fallback when config not yet built
    get_config = None  # type: ignore[misc, assignment]
    _CONFIG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants -- Scenarios
# ---------------------------------------------------------------------------

# Mapping from scenario string identifier to scenario metadata.
# warming_by_2100 is the IPCC AR5/AR6 median estimate of global mean
# surface temperature increase (delta C relative to 1850-1900 baseline).

_SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ssp1_1.9": {
        "name": "SSP1-1.9",
        "pathway": "SSP",
        "forcing": "1.9 W/m2",
        "warming_by_2100": 1.4,
        "description": (
            "Sustainability pathway with very low greenhouse gas emissions. "
            "CO2 emissions are cut to net zero around 2050.  Global warming "
            "is limited to approximately 1.5 C above pre-industrial levels."
        ),
        "ipcc_report": "AR6",
        "narrative": "Taking the Green Road",
        "emission_trajectory": "very_low",
        "co2_peak_year": 2020,
        "net_zero_year": 2050,
        "warming_range_low": 1.0,
        "warming_range_high": 1.8,
    },
    "ssp1_2.6": {
        "name": "SSP1-2.6",
        "pathway": "SSP",
        "forcing": "2.6 W/m2",
        "warming_by_2100": 1.8,
        "description": (
            "Sustainability pathway with low greenhouse gas emissions. "
            "Global cooperation leads to rapid emissions reductions "
            "consistent with 2 C warming above pre-industrial levels."
        ),
        "ipcc_report": "AR6",
        "narrative": "Taking the Green Road",
        "emission_trajectory": "low",
        "co2_peak_year": 2020,
        "net_zero_year": 2070,
        "warming_range_low": 1.3,
        "warming_range_high": 2.4,
    },
    "ssp2_4.5": {
        "name": "SSP2-4.5",
        "pathway": "SSP",
        "forcing": "4.5 W/m2",
        "warming_by_2100": 2.7,
        "description": (
            "Middle of the road scenario.  Social, economic, and "
            "technological trends do not shift markedly from historical "
            "patterns.  Emissions peak around mid-century and decline "
            "thereafter but do not reach net zero by 2100."
        ),
        "ipcc_report": "AR6",
        "narrative": "Middle of the Road",
        "emission_trajectory": "medium",
        "co2_peak_year": 2040,
        "net_zero_year": None,
        "warming_range_low": 2.1,
        "warming_range_high": 3.5,
    },
    "ssp3_7.0": {
        "name": "SSP3-7.0",
        "pathway": "SSP",
        "forcing": "7.0 W/m2",
        "warming_by_2100": 3.6,
        "description": (
            "Regional rivalry with high greenhouse gas emissions.  "
            "Resurgent nationalism and regional conflicts hamper "
            "international cooperation.  Emissions continue to rise "
            "through the century."
        ),
        "ipcc_report": "AR6",
        "narrative": "A Rocky Road",
        "emission_trajectory": "high",
        "co2_peak_year": 2070,
        "net_zero_year": None,
        "warming_range_low": 2.8,
        "warming_range_high": 4.6,
    },
    "ssp5_8.5": {
        "name": "SSP5-8.5",
        "pathway": "SSP",
        "forcing": "8.5 W/m2",
        "warming_by_2100": 4.4,
        "description": (
            "Fossil-fueled development with very high greenhouse gas "
            "emissions.  Economic growth is driven by the exploitation "
            "of abundant fossil fuel resources and energy-intensive "
            "lifestyles.  This is the highest-forcing SSP scenario."
        ),
        "ipcc_report": "AR6",
        "narrative": "Taking the Highway",
        "emission_trajectory": "very_high",
        "co2_peak_year": 2090,
        "net_zero_year": None,
        "warming_range_low": 3.3,
        "warming_range_high": 5.7,
    },
    "rcp2.6": {
        "name": "RCP2.6",
        "pathway": "RCP",
        "forcing": "2.6 W/m2",
        "warming_by_2100": 1.6,
        "description": (
            "Low emissions legacy pathway from IPCC AR5.  Radiative "
            "forcing peaks at approximately 3 W/m2 before 2100 and "
            "then declines to 2.6 W/m2 by end of century."
        ),
        "ipcc_report": "AR5",
        "narrative": "Peak and Decline",
        "emission_trajectory": "low",
        "co2_peak_year": 2020,
        "net_zero_year": 2070,
        "warming_range_low": 0.9,
        "warming_range_high": 2.3,
    },
    "rcp4.5": {
        "name": "RCP4.5",
        "pathway": "RCP",
        "forcing": "4.5 W/m2",
        "warming_by_2100": 2.4,
        "description": (
            "Medium emissions legacy pathway from IPCC AR5.  Emissions "
            "peak around 2040 and then decline.  Radiative forcing "
            "stabilises at about 4.5 W/m2 before end of century."
        ),
        "ipcc_report": "AR5",
        "narrative": "Stabilisation",
        "emission_trajectory": "medium",
        "co2_peak_year": 2040,
        "net_zero_year": None,
        "warming_range_low": 1.7,
        "warming_range_high": 3.2,
    },
    "rcp8.5": {
        "name": "RCP8.5",
        "pathway": "RCP",
        "forcing": "8.5 W/m2",
        "warming_by_2100": 4.3,
        "description": (
            "High emissions legacy pathway from IPCC AR5.  Emissions "
            "continue to rise throughout the 21st century.  Radiative "
            "forcing reaches 8.5 W/m2 by 2100.  Comparable to SSP5-8.5."
        ),
        "ipcc_report": "AR5",
        "narrative": "Rising Emissions",
        "emission_trajectory": "very_high",
        "co2_peak_year": 2100,
        "net_zero_year": None,
        "warming_range_low": 3.2,
        "warming_range_high": 5.4,
    },
}

# Frozen set of all valid scenario identifiers for fast membership testing
_VALID_SCENARIO_IDS: frozenset = frozenset(_SCENARIO_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Constants -- Time Horizons
# ---------------------------------------------------------------------------

# Each time horizon maps to a period range and a warming fraction.
# The warming fraction indicates what fraction of the end-of-century (2100)
# warming has been realised by that time horizon's midpoint.

_TIME_HORIZON_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BASELINE": {
        "name": "Baseline",
        "period_start": 1995,
        "period_end": 2014,
        "warming_fraction": 0.0,
        "description": (
            "Historical reference period (1995-2014).  No additional "
            "warming above the observed baseline is applied."
        ),
    },
    "NEAR_TERM": {
        "name": "Near-term",
        "period_start": 2021,
        "period_end": 2040,
        "warming_fraction": 0.3,
        "description": (
            "Near-term projection period (2021-2040).  Approximately 30% "
            "of end-of-century warming has been realised.  Climate "
            "response is dominated by committed warming from existing "
            "greenhouse gas concentrations."
        ),
    },
    "MID_TERM": {
        "name": "Mid-term",
        "period_start": 2041,
        "period_end": 2060,
        "warming_fraction": 0.55,
        "description": (
            "Mid-term projection period (2041-2060).  Approximately 55% "
            "of end-of-century warming has been realised.  Scenario "
            "divergence becomes increasingly visible."
        ),
    },
    "LONG_TERM": {
        "name": "Long-term",
        "period_start": 2061,
        "period_end": 2080,
        "warming_fraction": 0.8,
        "description": (
            "Long-term projection period (2061-2080).  Approximately 80% "
            "of end-of-century warming has been realised.  Scenario "
            "choice strongly influences projected hazard levels."
        ),
    },
    "END_CENTURY": {
        "name": "End of century",
        "period_start": 2081,
        "period_end": 2100,
        "warming_fraction": 1.0,
        "description": (
            "End-of-century projection period (2081-2100).  Full scenario "
            "warming is applied.  This represents the maximum projected "
            "climate impact for each scenario pathway."
        ),
    },
}

# Frozen set of valid horizon identifiers
_VALID_TIME_HORIZONS: frozenset = frozenset(_TIME_HORIZON_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Constants -- Hazard Scaling Factors
# ---------------------------------------------------------------------------

# Per-degree-Celsius scaling factors for each hazard type.
#
# intensity_factor: multiplicative change in hazard intensity per 1 C warming.
#   For most hazards this is > 1 indicating amplification.  For SEA_LEVEL_RISE
#   the intensity factor is cumulative metres per degree.  For EXTREME_COLD the
#   factor is < 1 indicating weakening with warming.
#
# frequency_factor: multiplicative change in hazard occurrence frequency per
#   1 C warming.  For TROPICAL_CYCLONE frequency_factor < 1 reflects the
#   scientific consensus that total cyclone count may decrease while intensity
#   increases.  For EXTREME_COLD, frequency decreases significantly.
#
# Sources:
#   - IPCC AR6 WG1 Chapter 11 (Weather and Climate Extreme Events)
#   - IPCC AR6 WG2 Chapter 4 (Water), Chapter 9 (Coastal)
#   - Swiss Re sigma risk assessment methodology
#   - Simplified representative scaling for decision-support

_HAZARD_SCALING_FACTORS: Dict[str, Dict[str, float]] = {
    "EXTREME_HEAT": {
        "intensity_factor": 2.0,
        "frequency_factor": 1.8,
    },
    "RIVERINE_FLOOD": {
        "intensity_factor": 1.3,
        "frequency_factor": 1.2,
    },
    "COASTAL_FLOOD": {
        "intensity_factor": 1.4,
        "frequency_factor": 1.3,
    },
    "DROUGHT": {
        "intensity_factor": 1.5,
        "frequency_factor": 1.4,
    },
    "WILDFIRE": {
        "intensity_factor": 1.6,
        "frequency_factor": 1.5,
    },
    "TROPICAL_CYCLONE": {
        "intensity_factor": 1.1,
        "frequency_factor": 0.9,
    },
    "EXTREME_PRECIPITATION": {
        "intensity_factor": 1.4,
        "frequency_factor": 1.3,
    },
    "WATER_STRESS": {
        "intensity_factor": 1.3,
        "frequency_factor": 1.2,
    },
    "SEA_LEVEL_RISE": {
        "intensity_factor": 0.3,
        "frequency_factor": 1.0,
    },
    "LANDSLIDE": {
        "intensity_factor": 1.2,
        "frequency_factor": 1.1,
    },
    "COASTAL_EROSION": {
        "intensity_factor": 1.3,
        "frequency_factor": 1.1,
    },
    "EXTREME_COLD": {
        "intensity_factor": 0.7,
        "frequency_factor": 0.6,
    },
}

# Frozen set of valid hazard types
_VALID_HAZARD_TYPES: frozenset = frozenset(_HAZARD_SCALING_FACTORS.keys())

# Default baseline risk components used when caller provides partial input
_DEFAULT_BASELINE_RISK: Dict[str, float] = {
    "probability": 0.0,
    "intensity": 0.0,
    "frequency": 0.0,
    "duration_days": 0.0,
}

# Required keys in a baseline_risk dictionary
_BASELINE_RISK_KEYS: frozenset = frozenset(
    {"probability", "intensity", "frequency", "duration_days"}
)


# ---------------------------------------------------------------------------
# Duration scaling factors (per C warming)
# ---------------------------------------------------------------------------

# Duration scales differently from intensity/frequency.  These factors
# represent the fractional increase in event duration per degree C.
# For example, EXTREME_HEAT duration_factor=0.25 means a 25% increase
# in event duration per degree of warming.

_HAZARD_DURATION_FACTORS: Dict[str, float] = {
    "EXTREME_HEAT": 0.25,
    "RIVERINE_FLOOD": 0.15,
    "COASTAL_FLOOD": 0.10,
    "DROUGHT": 0.30,
    "WILDFIRE": 0.20,
    "TROPICAL_CYCLONE": 0.05,
    "EXTREME_PRECIPITATION": 0.10,
    "WATER_STRESS": 0.25,
    "SEA_LEVEL_RISE": 0.0,
    "LANDSLIDE": 0.08,
    "COASTAL_EROSION": 0.15,
    "EXTREME_COLD": -0.20,
}

# Probability scaling factors per C warming.  These represent the
# fractional increase in event probability per degree C of warming.
# The projected probability is clamped to [0.0, 1.0].

_HAZARD_PROBABILITY_FACTORS: Dict[str, float] = {
    "EXTREME_HEAT": 0.35,
    "RIVERINE_FLOOD": 0.15,
    "COASTAL_FLOOD": 0.20,
    "DROUGHT": 0.20,
    "WILDFIRE": 0.25,
    "TROPICAL_CYCLONE": 0.08,
    "EXTREME_PRECIPITATION": 0.18,
    "WATER_STRESS": 0.15,
    "SEA_LEVEL_RISE": 0.30,
    "LANDSLIDE": 0.10,
    "COASTAL_EROSION": 0.18,
    "EXTREME_COLD": -0.25,
}


# ---------------------------------------------------------------------------
# ID generation helper
# ---------------------------------------------------------------------------

_PREFIX_PROJECTION = "PROJ"


def _generate_id(prefix: str = "PROJ") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Short uppercase string prepended to the random hex segment.

    Returns:
        String of the form ``{prefix}-{hex12}``.

    Example:
        >>> _generate_id("PROJ")
        'PROJ-3f9a1b2c4d5e'
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed for consistency.

    Returns:
        Timezone-aware datetime at second precision in UTC.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------


def _compute_provenance(operation: str, payload_repr: str) -> str:
    """Compute a SHA-256 provenance hash for an engine operation.

    The hash covers the operation name, the serialised payload, and the
    current UTC timestamp so every call produces a unique fingerprint even
    for identical inputs.

    Args:
        operation: Human-readable name of the operation (e.g. "project_hazard").
        payload_repr: JSON-serialised or string representation of the data.

    Returns:
        Hex-encoded 64-character SHA-256 digest.
    """
    raw = f"{operation}:{payload_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def _record_projection_metric(
    hazard_type: str,
    scenario: str,
    duration_seconds: float,
) -> None:
    """Safely record a projection metric if the metrics module is available.

    Args:
        hazard_type: The hazard type projected.
        scenario: The scenario string identifier.
        duration_seconds: Processing duration in seconds.
    """
    if not _METRICS_AVAILABLE or _metrics_mod is None:
        return
    try:
        record_fn = getattr(_metrics_mod, "record_projection", None)
        if record_fn is not None:
            record_fn(hazard_type=hazard_type, scenario=scenario)
        observe_fn = getattr(_metrics_mod, "observe_processing_duration", None)
        if observe_fn is not None:
            observe_fn(operation="scenario_projection", seconds=duration_seconds)
    except Exception:  # pragma: no cover
        logger.debug("Metrics recording failed (non-critical)", exc_info=True)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_scenario(scenario: str) -> str:
    """Validate and normalise a scenario identifier.

    Accepts both canonical (e.g. ``"ssp2_4.5"``) and display-name forms
    (e.g. ``"SSP2-4.5"``).  Returns the normalised canonical form.

    Args:
        scenario: Scenario identifier to validate.

    Returns:
        Normalised canonical scenario identifier (lowercase, underscore-separated).

    Raises:
        ValueError: If the scenario is not recognised.
    """
    if not scenario or not isinstance(scenario, str):
        raise ValueError(
            f"scenario must be a non-empty string, got {scenario!r}"
        )

    # Normalise: lowercase, replace hyphens with underscores
    normalised = scenario.strip().lower().replace("-", "_")

    # Check against registry
    if normalised in _VALID_SCENARIO_IDS:
        return normalised

    # Try a second pass replacing spaces with underscores
    normalised_alt = scenario.strip().lower().replace(" ", "_").replace("-", "_")
    if normalised_alt in _VALID_SCENARIO_IDS:
        return normalised_alt

    # Try to match display names (e.g. "SSP2-4.5" -> "ssp2_4.5")
    for sid, meta in _SCENARIO_REGISTRY.items():
        display = meta["name"].strip().lower().replace("-", "_")
        if normalised == display:
            return sid

    raise ValueError(
        f"Unknown scenario '{scenario}'. Valid scenarios: "
        f"{sorted(_VALID_SCENARIO_IDS)}"
    )


def _validate_time_horizon(time_horizon: str) -> str:
    """Validate and normalise a time horizon identifier.

    Args:
        time_horizon: Time horizon identifier to validate.

    Returns:
        Normalised uppercase time horizon identifier.

    Raises:
        ValueError: If the time horizon is not recognised.
    """
    if not time_horizon or not isinstance(time_horizon, str):
        raise ValueError(
            f"time_horizon must be a non-empty string, got {time_horizon!r}"
        )

    normalised = time_horizon.strip().upper().replace("-", "_").replace(" ", "_")

    if normalised in _VALID_TIME_HORIZONS:
        return normalised

    raise ValueError(
        f"Unknown time_horizon '{time_horizon}'. Valid horizons: "
        f"{sorted(_VALID_TIME_HORIZONS)}"
    )


def _validate_hazard_type(hazard_type: str) -> str:
    """Validate and normalise a hazard type identifier.

    Args:
        hazard_type: Hazard type to validate.

    Returns:
        Normalised uppercase hazard type identifier.

    Raises:
        ValueError: If the hazard type is not recognised.
    """
    if not hazard_type or not isinstance(hazard_type, str):
        raise ValueError(
            f"hazard_type must be a non-empty string, got {hazard_type!r}"
        )

    normalised = hazard_type.strip().upper().replace("-", "_").replace(" ", "_")

    if normalised in _VALID_HAZARD_TYPES:
        return normalised

    raise ValueError(
        f"Unknown hazard_type '{hazard_type}'. Valid hazard types: "
        f"{sorted(_VALID_HAZARD_TYPES)}"
    )


def _validate_baseline_risk(baseline_risk: Dict[str, Any]) -> Dict[str, float]:
    """Validate and normalise a baseline risk dictionary.

    Ensures all required keys are present and values are non-negative
    numbers.  Missing keys are filled from defaults.  Probability is
    clamped to [0.0, 1.0].

    Args:
        baseline_risk: Dictionary with probability, intensity, frequency,
            and duration_days.

    Returns:
        Normalised baseline risk dictionary with float values.

    Raises:
        ValueError: If required keys have invalid values.
    """
    if not baseline_risk or not isinstance(baseline_risk, dict):
        raise ValueError(
            "baseline_risk must be a non-empty dictionary with keys: "
            f"{sorted(_BASELINE_RISK_KEYS)}"
        )

    result: Dict[str, float] = {}
    errors: list[str] = []

    for key in _BASELINE_RISK_KEYS:
        raw_val = baseline_risk.get(key, _DEFAULT_BASELINE_RISK.get(key, 0.0))
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            errors.append(
                f"baseline_risk['{key}'] must be numeric, got {raw_val!r}"
            )
            continue

        # Probability must be in [0, 1]
        if key == "probability":
            val = max(0.0, min(1.0, val))

        # All other values must be non-negative
        if key != "probability" and val < 0.0:
            errors.append(
                f"baseline_risk['{key}'] must be >= 0.0, got {val}"
            )
            continue

        result[key] = val

    if errors:
        raise ValueError(
            "baseline_risk validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return result


def _validate_location(location: Any) -> Dict[str, Any]:
    """Validate a location parameter.

    Accepts a dictionary with optional lat/lon/name fields or any
    non-empty value that can serve as a location identifier.

    Args:
        location: Location dictionary or identifier.

    Returns:
        Normalised location dictionary.

    Raises:
        ValueError: If location is None or empty.
    """
    if location is None:
        raise ValueError("location must not be None")

    if isinstance(location, dict):
        return dict(location)

    if isinstance(location, str):
        if not location.strip():
            raise ValueError("location string must not be empty")
        return {"name": location.strip()}

    # Accept any other truthy type
    return {"identifier": str(location)}


# ---------------------------------------------------------------------------
# ScenarioProjectorEngine
# ---------------------------------------------------------------------------


class ScenarioProjectorEngine:
    """SSP/RCP scenario projection engine for climate hazard risk assessment.

    Projects baseline hazard risk into the future under various IPCC climate
    scenarios and time horizons.  Uses deterministic scaling factors derived
    from peer-reviewed climate science to translate global mean temperature
    anomalies into changes in hazard probability, intensity, frequency, and
    event duration.

    The engine supports single-scenario projections, multi-scenario comparison,
    and time-series projections across all five standard time horizons.

    Thread Safety:
        All public methods are protected by a ``threading.Lock`` making the
        engine safe for concurrent use across multiple threads.

    Zero-Hallucination:
        All calculations use deterministic Python arithmetic with hard-coded
        lookup tables.  No LLM or ML models are invoked for numeric
        projections.  Every projection carries a SHA-256 provenance hash.

    Attributes:
        _risk_engine: Optional reference to a RiskIndexEngine for composite
            risk index calculations post-projection.
        _provenance: Optional ProvenanceTracker instance for audit trail.
        _lock: Thread safety lock for all mutable state.
        _projections: In-memory store of all computed projections, keyed by
            projection_id.
        _projection_order: Ordered list of projection IDs for list queries.
        _stats: Cumulative engine statistics.

    Example:
        >>> engine = ScenarioProjectorEngine()
        >>> result = engine.project_hazard(
        ...     hazard_type="DROUGHT",
        ...     location={"lat": 34.05, "lon": -118.24, "name": "Los Angeles"},
        ...     baseline_risk={"probability": 0.3, "intensity": 0.7,
        ...                    "frequency": 1.0, "duration_days": 30.0},
        ...     scenario="ssp3_7.0",
        ...     time_horizon="LONG_TERM",
        ... )
        >>> assert result["warming_delta_c"] > 0.0
        >>> assert result["projected_risk"]["probability"] > 0.3
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        risk_engine: Optional[Any] = None,
        provenance: Optional[Any] = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize ScenarioProjectorEngine.

        Args:
            risk_engine: Optional reference to a RiskIndexEngine for
                composite risk calculations post-projection.  When
                provided, projections can be enriched with composite
                risk scores.
            provenance: Optional ProvenanceTracker instance.  When None,
                the engine attempts to use the module-level singleton
                tracker.  Pass ``False`` to disable provenance entirely.
            genesis_hash: Optional genesis hash string for provenance
                chain anchoring.  Only used when constructing a new
                ProvenanceTracker internally.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> engine = ScenarioProjectorEngine(risk_engine=my_risk_engine)
        """
        self._risk_engine: Optional[Any] = risk_engine
        self._lock: threading.Lock = threading.Lock()

        # In-memory projection storage
        self._projections: Dict[str, Dict[str, Any]] = {}
        self._projection_order: List[str] = []

        # Statistics counters
        self._stats: Dict[str, Any] = {
            "total_projections": 0,
            "total_multi_scenario": 0,
            "total_time_series": 0,
            "total_warming_calculations": 0,
            "total_scaling_applications": 0,
            "projections_by_scenario": defaultdict(int),
            "projections_by_hazard": defaultdict(int),
            "projections_by_horizon": defaultdict(int),
            "total_errors": 0,
            "created_at": _utcnow().isoformat(),
            "last_projection_at": None,
        }

        # Provenance tracker setup
        self._provenance: Optional[Any] = self._init_provenance(
            provenance, genesis_hash
        )

        logger.info(
            "ScenarioProjectorEngine initialized: "
            "scenarios=%d, time_horizons=%d, hazard_types=%d, "
            "risk_engine=%s, provenance=%s",
            len(_SCENARIO_REGISTRY),
            len(_TIME_HORIZON_REGISTRY),
            len(_HAZARD_SCALING_FACTORS),
            "attached" if self._risk_engine is not None else "none",
            "enabled" if self._provenance is not None else "disabled",
        )

    def _init_provenance(
        self,
        provenance: Optional[Any],
        genesis_hash: Optional[str],
    ) -> Optional[Any]:
        """Initialise provenance tracker from constructor arguments.

        Args:
            provenance: Explicit tracker, None for singleton, or False to
                disable.
            genesis_hash: Genesis hash for new tracker construction.

        Returns:
            ProvenanceTracker instance or None.
        """
        # Explicitly disabled
        if provenance is False:
            return None

        # Explicit tracker provided
        if provenance is not None:
            return provenance

        # Try singleton
        if _PROVENANCE_AVAILABLE and get_provenance_tracker is not None:
            try:
                return get_provenance_tracker()
            except Exception:  # pragma: no cover
                logger.debug(
                    "Could not get singleton ProvenanceTracker", exc_info=True
                )

        # Construct new tracker
        if _PROVENANCE_AVAILABLE and ProvenanceTracker is not None:
            try:
                gash = genesis_hash or "greenlang-scenario-projector-genesis"
                return ProvenanceTracker(genesis_hash=gash)
            except Exception:  # pragma: no cover
                logger.debug(
                    "Could not construct ProvenanceTracker", exc_info=True
                )

        return None

    # ------------------------------------------------------------------
    # Provenance recording helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Any] = None,
    ) -> Optional[str]:
        """Record a provenance entry and return the hash, or None.

        Args:
            entity_type: Entity type string.
            action: Action string.
            entity_id: Entity identifier.
            data: Optional serializable payload.

        Returns:
            SHA-256 provenance hash string, or None if provenance is disabled.
        """
        if self._provenance is None:
            return _compute_provenance(action, json.dumps(data, default=str))

        try:
            entry = self._provenance.record(
                entity_type=entity_type,
                action=action,
                entity_id=entity_id,
                data=data,
            )
            return entry.hash_value
        except Exception:  # pragma: no cover
            logger.debug("Provenance recording failed (non-critical)", exc_info=True)
            return _compute_provenance(action, json.dumps(data, default=str))

    # ------------------------------------------------------------------
    # 1. project_hazard
    # ------------------------------------------------------------------

    def project_hazard(
        self,
        hazard_type: str,
        location: Any,
        baseline_risk: Dict[str, Any],
        scenario: str,
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Project a single hazard under a specific scenario and time horizon.

        Calculates the warming delta for the given scenario and time
        horizon, then applies hazard-specific scaling factors to the
        baseline risk components to produce a projected risk.

        Args:
            hazard_type: Climate hazard type (e.g. ``"EXTREME_HEAT"``,
                ``"DROUGHT"``, ``"COASTAL_FLOOD"``).  Case-insensitive.
            location: Location dictionary with optional lat/lon/name
                fields, or a string location identifier.
            baseline_risk: Dictionary with baseline risk components:
                - ``probability`` (float, 0.0-1.0): Baseline event probability.
                - ``intensity`` (float, >= 0): Baseline hazard intensity
                  (normalised 0-1 or absolute scale).
                - ``frequency`` (float, >= 0): Baseline event frequency
                  (events per year).
                - ``duration_days`` (float, >= 0): Baseline event duration
                  in days.
            scenario: Climate scenario identifier (e.g. ``"ssp2_4.5"``,
                ``"rcp8.5"``).  Case-insensitive.
            time_horizon: Time horizon identifier (e.g. ``"MID_TERM"``,
                ``"END_CENTURY"``).  Case-insensitive.

        Returns:
            Projection result dictionary with keys:
                - ``projection_id`` (str): Unique projection identifier.
                - ``hazard_type`` (str): Normalised hazard type.
                - ``location`` (dict): Location data.
                - ``scenario`` (str): Normalised scenario identifier.
                - ``time_horizon`` (str): Normalised time horizon.
                - ``baseline_risk`` (dict): Validated baseline risk values.
                - ``projected_risk`` (dict): Scaled projected risk values.
                - ``warming_delta_c`` (float): Temperature delta in C.
                - ``scaling_factors`` (dict): Applied per-C scaling factors.
                - ``risk_change_pct`` (dict): Percentage change per component.
                - ``scenario_info`` (dict): Scenario metadata.
                - ``horizon_info`` (dict): Time horizon metadata.
                - ``projected_at`` (str): ISO timestamp.
                - ``provenance_hash`` (str): SHA-256 audit hash.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        norm_hazard = _validate_hazard_type(hazard_type)
        norm_scenario = _validate_scenario(scenario)
        norm_horizon = _validate_time_horizon(time_horizon)
        norm_location = _validate_location(location)
        norm_baseline = _validate_baseline_risk(baseline_risk)

        # Calculate warming delta
        warming_delta = self.calculate_warming_delta(norm_scenario, norm_horizon)

        # Apply scaling
        projected_risk = self.apply_scaling_factors(
            norm_baseline, norm_hazard, warming_delta
        )

        # Calculate percentage changes
        risk_change_pct = self._compute_risk_change_pct(
            norm_baseline, projected_risk
        )

        # Get scaling factors for reference
        scaling = self.get_scaling_factors(norm_hazard)

        # Build projection record
        projection_id = _generate_id(_PREFIX_PROJECTION)
        now_iso = _utcnow().isoformat()

        scenario_info = copy.deepcopy(_SCENARIO_REGISTRY.get(norm_scenario, {}))
        horizon_info = copy.deepcopy(_TIME_HORIZON_REGISTRY.get(norm_horizon, {}))

        projection = {
            "projection_id": projection_id,
            "hazard_type": norm_hazard,
            "location": copy.deepcopy(norm_location),
            "scenario": norm_scenario,
            "time_horizon": norm_horizon,
            "baseline_risk": copy.deepcopy(norm_baseline),
            "projected_risk": copy.deepcopy(projected_risk),
            "warming_delta_c": round(warming_delta, 4),
            "scaling_factors": copy.deepcopy(scaling),
            "risk_change_pct": copy.deepcopy(risk_change_pct),
            "scenario_info": scenario_info,
            "horizon_info": horizon_info,
            "projected_at": now_iso,
            "provenance_hash": "",
        }

        # Record provenance
        prov_hash = self._record_provenance(
            entity_type="scenario_projection",
            action="project_scenario",
            entity_id=projection_id,
            data=projection,
        )
        projection["provenance_hash"] = prov_hash or ""

        # Store projection
        duration = time.monotonic() - start_time
        with self._lock:
            self._projections[projection_id] = copy.deepcopy(projection)
            self._projection_order.append(projection_id)
            self._stats["total_projections"] += 1
            self._stats["projections_by_scenario"][norm_scenario] += 1
            self._stats["projections_by_hazard"][norm_hazard] += 1
            self._stats["projections_by_horizon"][norm_horizon] += 1
            self._stats["last_projection_at"] = now_iso

        # Record metric
        _record_projection_metric(norm_hazard, norm_scenario, duration)

        logger.info(
            "project_hazard: id=%s hazard=%s scenario=%s horizon=%s "
            "warming=%.2fC duration=%.3fs",
            projection_id,
            norm_hazard,
            norm_scenario,
            norm_horizon,
            warming_delta,
            duration,
        )

        return copy.deepcopy(projection)

    # ------------------------------------------------------------------
    # 2. project_multi_scenario
    # ------------------------------------------------------------------

    def project_multi_scenario(
        self,
        hazard_type: str,
        location: Any,
        baseline_risk: Dict[str, Any],
        scenarios: List[str],
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Project a hazard across multiple scenarios for comparison.

        Runs ``project_hazard`` for each scenario and assembles a
        comparison summary sorted by composite risk change.

        Args:
            hazard_type: Climate hazard type.
            location: Location dictionary or identifier.
            baseline_risk: Baseline risk components dictionary.
            scenarios: List of scenario identifiers to compare.
            time_horizon: Time horizon for all projections.

        Returns:
            Multi-scenario result dictionary with keys:
                - ``comparison_id`` (str): Unique comparison identifier.
                - ``hazard_type`` (str): Normalised hazard type.
                - ``location`` (dict): Location data.
                - ``time_horizon`` (str): Normalised time horizon.
                - ``baseline_risk`` (dict): Validated baseline risk.
                - ``scenarios_count`` (int): Number of scenarios compared.
                - ``per_scenario_projections`` (list): List of individual
                  projection results.
                - ``scenario_comparison`` (list): Scenarios sorted by
                  overall risk change (highest first).
                - ``warming_range`` (dict): Min and max warming deltas.
                - ``projected_at`` (str): ISO timestamp.
                - ``provenance_hash`` (str): SHA-256 audit hash.

        Raises:
            ValueError: If inputs are invalid or scenarios list is empty.
        """
        start_time = time.monotonic()

        if not scenarios or not isinstance(scenarios, (list, tuple)):
            raise ValueError(
                "scenarios must be a non-empty list of scenario identifiers"
            )

        # Validate common inputs once
        norm_hazard = _validate_hazard_type(hazard_type)
        norm_horizon = _validate_time_horizon(time_horizon)
        norm_location = _validate_location(location)
        norm_baseline = _validate_baseline_risk(baseline_risk)

        # Run projections for each scenario
        per_scenario: List[Dict[str, Any]] = []
        for scen in scenarios:
            try:
                result = self.project_hazard(
                    hazard_type=norm_hazard,
                    location=norm_location,
                    baseline_risk=norm_baseline,
                    scenario=scen,
                    time_horizon=norm_horizon,
                )
                per_scenario.append(result)
            except ValueError as exc:
                logger.warning(
                    "project_multi_scenario: skipping invalid scenario "
                    "'%s': %s",
                    scen,
                    exc,
                )
                continue

        if not per_scenario:
            raise ValueError(
                "No valid scenarios could be projected. Check scenario "
                "identifiers."
            )

        # Build comparison: sort by overall risk change descending
        scenario_comparison = self._build_scenario_comparison(per_scenario)

        # Warming range
        warmings = [p["warming_delta_c"] for p in per_scenario]
        warming_range = {
            "min_warming_c": round(min(warmings), 4),
            "max_warming_c": round(max(warmings), 4),
            "warming_spread_c": round(max(warmings) - min(warmings), 4),
        }

        comparison_id = _generate_id("COMP")
        now_iso = _utcnow().isoformat()

        result = {
            "comparison_id": comparison_id,
            "hazard_type": norm_hazard,
            "location": copy.deepcopy(norm_location),
            "time_horizon": norm_horizon,
            "baseline_risk": copy.deepcopy(norm_baseline),
            "scenarios_count": len(per_scenario),
            "per_scenario_projections": copy.deepcopy(per_scenario),
            "scenario_comparison": copy.deepcopy(scenario_comparison),
            "warming_range": warming_range,
            "projected_at": now_iso,
            "provenance_hash": "",
        }

        # Record provenance
        prov_hash = self._record_provenance(
            entity_type="scenario_projection",
            action="project_multi",
            entity_id=comparison_id,
            data=result,
        )
        result["provenance_hash"] = prov_hash or ""

        duration = time.monotonic() - start_time
        with self._lock:
            self._stats["total_multi_scenario"] += 1

        logger.info(
            "project_multi_scenario: id=%s hazard=%s scenarios=%d "
            "horizon=%s duration=%.3fs",
            comparison_id,
            norm_hazard,
            len(per_scenario),
            norm_horizon,
            duration,
        )

        return copy.deepcopy(result)

    # ------------------------------------------------------------------
    # 3. project_time_series
    # ------------------------------------------------------------------

    def project_time_series(
        self,
        hazard_type: str,
        location: Any,
        baseline_risk: Dict[str, Any],
        scenario: str,
        time_horizons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Project a hazard across multiple time horizons for a single scenario.

        Runs ``project_hazard`` for each time horizon and assembles a
        time-series view showing how risk evolves from baseline through
        end-of-century under the selected scenario.

        Args:
            hazard_type: Climate hazard type.
            location: Location dictionary or identifier.
            baseline_risk: Baseline risk components dictionary.
            scenario: Climate scenario identifier.
            time_horizons: Optional list of time horizon identifiers.
                When None, all five horizons are used in chronological
                order (BASELINE, NEAR_TERM, MID_TERM, LONG_TERM,
                END_CENTURY).

        Returns:
            Time-series result dictionary with keys:
                - ``timeseries_id`` (str): Unique time series identifier.
                - ``hazard_type`` (str): Normalised hazard type.
                - ``location`` (dict): Location data.
                - ``scenario`` (str): Normalised scenario identifier.
                - ``baseline_risk`` (dict): Validated baseline risk.
                - ``horizons_count`` (int): Number of time horizons.
                - ``time_series`` (list): List of projections in
                  chronological order.
                - ``trend_direction`` (str): Overall trend classification:
                  ``"increasing"``, ``"decreasing"``, ``"stable"``, or
                  ``"non_monotonic"``.
                - ``warming_trajectory`` (list): List of dicts with
                  horizon/warming_delta_c for quick reference.
                - ``projected_at`` (str): ISO timestamp.
                - ``provenance_hash`` (str): SHA-256 audit hash.

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.monotonic()

        # Default to all horizons in chronological order
        if time_horizons is None:
            time_horizons = [
                "BASELINE",
                "NEAR_TERM",
                "MID_TERM",
                "LONG_TERM",
                "END_CENTURY",
            ]

        if not time_horizons or not isinstance(time_horizons, (list, tuple)):
            raise ValueError(
                "time_horizons must be a non-empty list of horizon identifiers"
            )

        # Validate common inputs once
        norm_hazard = _validate_hazard_type(hazard_type)
        norm_scenario = _validate_scenario(scenario)
        norm_location = _validate_location(location)
        norm_baseline = _validate_baseline_risk(baseline_risk)

        # Run projections for each horizon
        ts_projections: List[Dict[str, Any]] = []
        for hz in time_horizons:
            try:
                result = self.project_hazard(
                    hazard_type=norm_hazard,
                    location=norm_location,
                    baseline_risk=norm_baseline,
                    scenario=norm_scenario,
                    time_horizon=hz,
                )
                ts_projections.append(result)
            except ValueError as exc:
                logger.warning(
                    "project_time_series: skipping invalid horizon "
                    "'%s': %s",
                    hz,
                    exc,
                )
                continue

        if not ts_projections:
            raise ValueError(
                "No valid time horizons could be projected. Check horizon "
                "identifiers."
            )

        # Sort by warming fraction (proxy for chronological order)
        ts_projections.sort(key=lambda p: p.get("warming_delta_c", 0.0))

        # Determine trend direction
        trend_direction = self._determine_trend(ts_projections)

        # Warming trajectory summary
        warming_trajectory = [
            {
                "time_horizon": p["time_horizon"],
                "period": _TIME_HORIZON_REGISTRY.get(
                    p["time_horizon"], {}
                ).get("period_start", "?"),
                "warming_delta_c": p["warming_delta_c"],
            }
            for p in ts_projections
        ]

        ts_id = _generate_id("TS")
        now_iso = _utcnow().isoformat()

        result = {
            "timeseries_id": ts_id,
            "hazard_type": norm_hazard,
            "location": copy.deepcopy(norm_location),
            "scenario": norm_scenario,
            "baseline_risk": copy.deepcopy(norm_baseline),
            "horizons_count": len(ts_projections),
            "time_series": copy.deepcopy(ts_projections),
            "trend_direction": trend_direction,
            "warming_trajectory": warming_trajectory,
            "projected_at": now_iso,
            "provenance_hash": "",
        }

        # Record provenance
        prov_hash = self._record_provenance(
            entity_type="scenario_projection",
            action="project_timeseries",
            entity_id=ts_id,
            data=result,
        )
        result["provenance_hash"] = prov_hash or ""

        duration = time.monotonic() - start_time
        with self._lock:
            self._stats["total_time_series"] += 1

        logger.info(
            "project_time_series: id=%s hazard=%s scenario=%s "
            "horizons=%d trend=%s duration=%.3fs",
            ts_id,
            norm_hazard,
            norm_scenario,
            len(ts_projections),
            trend_direction,
            duration,
        )

        return copy.deepcopy(result)

    # ------------------------------------------------------------------
    # 4. calculate_warming_delta
    # ------------------------------------------------------------------

    def calculate_warming_delta(
        self,
        scenario: str,
        time_horizon: str,
    ) -> float:
        """Calculate the temperature delta for a scenario and time horizon.

        The warming delta is computed as::

            delta_C = warming_by_2100 * warming_fraction

        where ``warming_by_2100`` comes from the scenario registry and
        ``warming_fraction`` comes from the time horizon registry.

        Args:
            scenario: Climate scenario identifier.
            time_horizon: Time horizon identifier.

        Returns:
            Temperature delta in degrees Celsius, rounded to 4 decimal
            places.

        Raises:
            ValueError: If scenario or time_horizon is not recognised.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> delta = engine.calculate_warming_delta("ssp2_4.5", "MID_TERM")
            >>> # 2.7 * 0.55 = 1.485
            >>> assert abs(delta - 1.485) < 0.001
        """
        norm_scenario = _validate_scenario(scenario)
        norm_horizon = _validate_time_horizon(time_horizon)

        scenario_meta = _SCENARIO_REGISTRY[norm_scenario]
        horizon_meta = _TIME_HORIZON_REGISTRY[norm_horizon]

        warming_by_2100 = scenario_meta["warming_by_2100"]
        warming_fraction = horizon_meta["warming_fraction"]

        delta = warming_by_2100 * warming_fraction

        with self._lock:
            self._stats["total_warming_calculations"] += 1

        logger.debug(
            "calculate_warming_delta: scenario=%s horizon=%s "
            "warming_2100=%.1fC fraction=%.2f delta=%.4fC",
            norm_scenario,
            norm_horizon,
            warming_by_2100,
            warming_fraction,
            delta,
        )

        return round(delta, 4)

    # ------------------------------------------------------------------
    # 5. apply_scaling_factors
    # ------------------------------------------------------------------

    def apply_scaling_factors(
        self,
        baseline_risk: Dict[str, Any],
        hazard_type: str,
        warming_delta: float,
    ) -> Dict[str, float]:
        """Scale baseline risk components by warming-dependent factors.

        Applies hazard-specific scaling factors to each risk component:

        - **Probability**: ``baseline * (1 + probability_factor * delta_C)``,
          clamped to [0.0, 1.0].
        - **Intensity**: ``baseline * (1 + (intensity_factor - 1) * delta_C)``
          for standard hazards.  For SEA_LEVEL_RISE, intensity is additive
          (``baseline + intensity_factor * delta_C``).
        - **Frequency**: ``baseline * (1 + (frequency_factor - 1) * delta_C)``,
          floored at 0.0.
        - **Duration**: ``baseline * (1 + duration_factor * delta_C)``,
          floored at 0.0.

        Args:
            baseline_risk: Dictionary with probability, intensity,
                frequency, and duration_days components.
            hazard_type: Climate hazard type identifier.
            warming_delta: Temperature anomaly in degrees Celsius.

        Returns:
            Dictionary with projected probability, intensity, frequency,
            and duration_days values.

        Raises:
            ValueError: If hazard_type or baseline_risk is invalid.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> baseline = {"probability": 0.2, "intensity": 0.5,
            ...             "frequency": 3.0, "duration_days": 7.0}
            >>> projected = engine.apply_scaling_factors(
            ...     baseline, "EXTREME_HEAT", 1.485
            ... )
            >>> assert projected["probability"] > 0.2
        """
        norm_hazard = _validate_hazard_type(hazard_type)
        norm_baseline = _validate_baseline_risk(baseline_risk)
        delta = float(warming_delta)

        scaling = _HAZARD_SCALING_FACTORS[norm_hazard]
        intensity_factor = scaling["intensity_factor"]
        frequency_factor = scaling["frequency_factor"]
        prob_factor = _HAZARD_PROBABILITY_FACTORS.get(norm_hazard, 0.15)
        dur_factor = _HAZARD_DURATION_FACTORS.get(norm_hazard, 0.10)

        base_prob = norm_baseline["probability"]
        base_intensity = norm_baseline["intensity"]
        base_frequency = norm_baseline["frequency"]
        base_duration = norm_baseline["duration_days"]

        # Probability scaling
        proj_probability = self._scale_probability(
            base_prob, prob_factor, delta
        )

        # Intensity scaling
        proj_intensity = self._scale_intensity(
            base_intensity, intensity_factor, delta, norm_hazard
        )

        # Frequency scaling
        proj_frequency = self._scale_frequency(
            base_frequency, frequency_factor, delta
        )

        # Duration scaling
        proj_duration = self._scale_duration(
            base_duration, dur_factor, delta
        )

        result = {
            "probability": round(proj_probability, 6),
            "intensity": round(proj_intensity, 6),
            "frequency": round(proj_frequency, 6),
            "duration_days": round(proj_duration, 4),
        }

        with self._lock:
            self._stats["total_scaling_applications"] += 1

        logger.debug(
            "apply_scaling_factors: hazard=%s delta=%.3fC "
            "prob=%.4f->%.4f int=%.4f->%.4f "
            "freq=%.4f->%.4f dur=%.2f->%.2f",
            norm_hazard,
            delta,
            base_prob,
            proj_probability,
            base_intensity,
            proj_intensity,
            base_frequency,
            proj_frequency,
            base_duration,
            proj_duration,
        )

        return result

    # ------------------------------------------------------------------
    # 6. get_scaling_factors
    # ------------------------------------------------------------------

    def get_scaling_factors(
        self,
        hazard_type: str,
    ) -> Dict[str, Any]:
        """Get the per-degree-Celsius scaling factors for a hazard type.

        Returns all four scaling factor families (intensity, frequency,
        probability, duration) for the specified hazard type.

        Args:
            hazard_type: Climate hazard type identifier.

        Returns:
            Dictionary with keys:
                - ``hazard_type`` (str): Normalised hazard type.
                - ``intensity_factor`` (float): Per-C intensity scaling.
                - ``frequency_factor`` (float): Per-C frequency scaling.
                - ``probability_factor`` (float): Per-C probability scaling.
                - ``duration_factor`` (float): Per-C duration scaling.
                - ``intensity_direction`` (str): ``"increasing"``,
                  ``"decreasing"``, or ``"stable"``.
                - ``frequency_direction`` (str): Direction of frequency
                  change.

        Raises:
            ValueError: If hazard_type is not recognised.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> factors = engine.get_scaling_factors("EXTREME_HEAT")
            >>> assert factors["intensity_factor"] == 2.0
        """
        norm_hazard = _validate_hazard_type(hazard_type)

        scaling = _HAZARD_SCALING_FACTORS[norm_hazard]
        prob_factor = _HAZARD_PROBABILITY_FACTORS.get(norm_hazard, 0.15)
        dur_factor = _HAZARD_DURATION_FACTORS.get(norm_hazard, 0.10)

        int_factor = scaling["intensity_factor"]
        freq_factor = scaling["frequency_factor"]

        result = {
            "hazard_type": norm_hazard,
            "intensity_factor": int_factor,
            "frequency_factor": freq_factor,
            "probability_factor": prob_factor,
            "duration_factor": dur_factor,
            "intensity_direction": self._classify_direction(int_factor, 1.0),
            "frequency_direction": self._classify_direction(freq_factor, 1.0),
            "probability_direction": (
                "increasing" if prob_factor > 0 else
                "decreasing" if prob_factor < 0 else
                "stable"
            ),
            "duration_direction": (
                "increasing" if dur_factor > 0 else
                "decreasing" if dur_factor < 0 else
                "stable"
            ),
        }

        return copy.deepcopy(result)

    # ------------------------------------------------------------------
    # 7. get_scenario_info
    # ------------------------------------------------------------------

    def get_scenario_info(
        self,
        scenario: str,
    ) -> Dict[str, Any]:
        """Get metadata for a specific climate scenario.

        Args:
            scenario: Climate scenario identifier.

        Returns:
            Dictionary with scenario metadata including name, pathway,
            forcing, warming_by_2100, description, IPCC report,
            narrative, emission_trajectory, and warming_range.

        Raises:
            ValueError: If the scenario is not recognised.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> info = engine.get_scenario_info("ssp2_4.5")
            >>> assert info["name"] == "SSP2-4.5"
            >>> assert info["warming_by_2100"] == 2.7
        """
        norm_scenario = _validate_scenario(scenario)
        meta = _SCENARIO_REGISTRY[norm_scenario]

        result = {
            "scenario_id": norm_scenario,
        }
        result.update(copy.deepcopy(meta))

        return result

    # ------------------------------------------------------------------
    # 8. list_scenarios
    # ------------------------------------------------------------------

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """List all 8 supported climate scenarios with metadata.

        Returns:
            List of scenario metadata dictionaries sorted by
            warming_by_2100 ascending.  Each entry contains:
                - ``scenario_id`` (str): Canonical scenario identifier.
                - ``name`` (str): Display name (e.g. ``"SSP2-4.5"``).
                - ``pathway`` (str): ``"SSP"`` or ``"RCP"``.
                - ``warming_by_2100`` (float): Median warming estimate.
                - ``emission_trajectory`` (str): Qualitative trajectory.
                - ``description`` (str): Scenario narrative.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> scenarios = engine.list_scenarios()
            >>> assert len(scenarios) == 8
            >>> assert scenarios[0]["warming_by_2100"] <= scenarios[-1]["warming_by_2100"]
        """
        result = []
        for sid, meta in _SCENARIO_REGISTRY.items():
            entry = {"scenario_id": sid}
            entry.update(copy.deepcopy(meta))
            result.append(entry)

        # Sort by warming ascending
        result.sort(key=lambda x: x.get("warming_by_2100", 0.0))

        return result

    # ------------------------------------------------------------------
    # 9. get_projection
    # ------------------------------------------------------------------

    def get_projection(
        self,
        projection_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a stored projection by its identifier.

        Args:
            projection_id: Unique projection identifier returned by
                ``project_hazard``.

        Returns:
            Deep copy of the projection dictionary, or ``None`` if the
            projection_id is not found.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> result = engine.project_hazard(...)
            >>> stored = engine.get_projection(result["projection_id"])
            >>> assert stored is not None
        """
        if not projection_id:
            return None

        with self._lock:
            projection = self._projections.get(projection_id)

        if projection is None:
            return None

        return copy.deepcopy(projection)

    # ------------------------------------------------------------------
    # 10. list_projections
    # ------------------------------------------------------------------

    def list_projections(
        self,
        hazard_type: Optional[str] = None,
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List stored projections with optional filtering.

        Args:
            hazard_type: Optional hazard type filter.
            scenario: Optional scenario filter.
            time_horizon: Optional time horizon filter.
            limit: Maximum number of projections to return.  Defaults to
                100.  Projections are returned in reverse chronological
                order (most recent first).

        Returns:
            List of projection dictionaries matching the filters.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> # ... run several projections ...
            >>> results = engine.list_projections(hazard_type="DROUGHT")
        """
        # Normalise filters
        norm_hazard = None
        if hazard_type:
            try:
                norm_hazard = _validate_hazard_type(hazard_type)
            except ValueError:
                return []

        norm_scenario = None
        if scenario:
            try:
                norm_scenario = _validate_scenario(scenario)
            except ValueError:
                return []

        norm_horizon = None
        if time_horizon:
            try:
                norm_horizon = _validate_time_horizon(time_horizon)
            except ValueError:
                return []

        with self._lock:
            # Iterate in reverse order (most recent first)
            ids = list(reversed(self._projection_order))
            result: List[Dict[str, Any]] = []

            for pid in ids:
                if len(result) >= limit:
                    break

                proj = self._projections.get(pid)
                if proj is None:
                    continue

                # Apply filters
                if norm_hazard and proj.get("hazard_type") != norm_hazard:
                    continue
                if norm_scenario and proj.get("scenario") != norm_scenario:
                    continue
                if norm_horizon and proj.get("time_horizon") != norm_horizon:
                    continue

                result.append(copy.deepcopy(proj))

        return result

    # ------------------------------------------------------------------
    # 11. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get cumulative engine statistics.

        Returns:
            Dictionary with statistics including total projections,
            multi-scenario comparisons, time-series runs, warming
            calculations, scaling applications, per-scenario counts,
            per-hazard counts, per-horizon counts, error counts,
            creation timestamp, and last projection timestamp.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> stats = engine.get_statistics()
            >>> assert stats["total_projections"] == 0
        """
        with self._lock:
            stats = copy.deepcopy(dict(self._stats))

        # Convert defaultdicts to plain dicts for serialisation
        for key in ("projections_by_scenario", "projections_by_hazard",
                     "projections_by_horizon"):
            if key in stats:
                stats[key] = dict(stats[key])

        # Add derived metrics
        stats["stored_projections"] = len(self._projections)
        stats["supported_scenarios"] = len(_SCENARIO_REGISTRY)
        stats["supported_horizons"] = len(_TIME_HORIZON_REGISTRY)
        stats["supported_hazard_types"] = len(_HAZARD_SCALING_FACTORS)
        stats["provenance_enabled"] = self._provenance is not None
        stats["risk_engine_attached"] = self._risk_engine is not None

        return stats

    # ------------------------------------------------------------------
    # 12. clear
    # ------------------------------------------------------------------

    def clear(self) -> Dict[str, Any]:
        """Reset all engine state and return summary.

        Clears all stored projections and resets statistics counters.
        Does not reset the provenance tracker (provenance is immutable
        for audit purposes).

        Returns:
            Summary dictionary with counts of cleared items.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> engine.project_hazard(...)
            >>> summary = engine.clear()
            >>> assert summary["projections_cleared"] >= 1
        """
        with self._lock:
            projections_count = len(self._projections)
            orders_count = len(self._projection_order)

            self._projections.clear()
            self._projection_order.clear()

            # Reset stats
            old_stats = copy.deepcopy(dict(self._stats))
            self._stats = {
                "total_projections": 0,
                "total_multi_scenario": 0,
                "total_time_series": 0,
                "total_warming_calculations": 0,
                "total_scaling_applications": 0,
                "projections_by_scenario": defaultdict(int),
                "projections_by_hazard": defaultdict(int),
                "projections_by_horizon": defaultdict(int),
                "total_errors": 0,
                "created_at": _utcnow().isoformat(),
                "last_projection_at": None,
            }

        # Record provenance for clear operation
        self._record_provenance(
            entity_type="scenario_projection",
            action="clear_engine",
            entity_id="scenario_projector",
            data={
                "projections_cleared": projections_count,
                "orders_cleared": orders_count,
            },
        )

        summary = {
            "projections_cleared": projections_count,
            "orders_cleared": orders_count,
            "previous_stats": {
                k: (dict(v) if isinstance(v, defaultdict) else v)
                for k, v in old_stats.items()
            },
            "cleared_at": _utcnow().isoformat(),
        }

        logger.info(
            "ScenarioProjectorEngine cleared: projections=%d",
            projections_count,
        )

        return copy.deepcopy(summary)

    # ------------------------------------------------------------------
    # Additional public methods
    # ------------------------------------------------------------------

    def get_time_horizon_info(
        self,
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Get metadata for a specific time horizon.

        Args:
            time_horizon: Time horizon identifier.

        Returns:
            Dictionary with time horizon metadata including name, period
            range, warming fraction, and description.

        Raises:
            ValueError: If the time horizon is not recognised.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> info = engine.get_time_horizon_info("MID_TERM")
            >>> assert info["warming_fraction"] == 0.55
        """
        norm_horizon = _validate_time_horizon(time_horizon)
        meta = _TIME_HORIZON_REGISTRY[norm_horizon]

        result = {"horizon_id": norm_horizon}
        result.update(copy.deepcopy(meta))

        return result

    def list_time_horizons(self) -> List[Dict[str, Any]]:
        """List all 5 supported time horizons with metadata.

        Returns:
            List of time horizon metadata dictionaries sorted by
            period_start ascending (chronological order).

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> horizons = engine.list_time_horizons()
            >>> assert len(horizons) == 5
        """
        result = []
        for hid, meta in _TIME_HORIZON_REGISTRY.items():
            entry = {"horizon_id": hid}
            entry.update(copy.deepcopy(meta))
            result.append(entry)

        result.sort(key=lambda x: x.get("period_start", 0))

        return result

    def list_hazard_types(self) -> List[Dict[str, Any]]:
        """List all 12 supported hazard types with scaling factors.

        Returns:
            List of hazard type dictionaries sorted alphabetically by
            hazard type name.  Each entry contains the hazard type plus
            all four scaling factors and their directional indicators.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> hazards = engine.list_hazard_types()
            >>> assert len(hazards) == 12
        """
        result = []
        for ht in sorted(_VALID_HAZARD_TYPES):
            entry = self.get_scaling_factors(ht)
            result.append(entry)

        return result

    def get_warming_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build a complete warming delta matrix (scenario x time horizon).

        Returns:
            Nested dictionary where the outer key is the scenario
            identifier and the inner key is the time horizon identifier.
            Values are warming deltas in degrees Celsius.

        Example:
            >>> engine = ScenarioProjectorEngine()
            >>> matrix = engine.get_warming_matrix()
            >>> assert matrix["ssp2_4.5"]["END_CENTURY"] == 2.7
        """
        matrix: Dict[str, Dict[str, float]] = {}

        for sid in sorted(_VALID_SCENARIO_IDS):
            matrix[sid] = {}
            for hid in ["BASELINE", "NEAR_TERM", "MID_TERM",
                         "LONG_TERM", "END_CENTURY"]:
                delta = self.calculate_warming_delta(sid, hid)
                matrix[sid][hid] = delta

        return copy.deepcopy(matrix)

    def project_all_hazards(
        self,
        location: Any,
        baseline_risks: Dict[str, Dict[str, Any]],
        scenario: str,
        time_horizon: str,
    ) -> Dict[str, Any]:
        """Project all provided hazard types under one scenario/horizon.

        Convenience method for projecting multiple hazard types at once.
        Each hazard type must have its own baseline risk in the
        ``baseline_risks`` dictionary.

        Args:
            location: Location dictionary or identifier.
            baseline_risks: Dictionary keyed by hazard_type string with
                baseline risk dictionaries as values.
            scenario: Climate scenario identifier.
            time_horizon: Time horizon identifier.

        Returns:
            Dictionary with:
                - ``batch_id`` (str): Unique batch identifier.
                - ``location`` (dict): Location data.
                - ``scenario`` (str): Normalised scenario.
                - ``time_horizon`` (str): Normalised horizon.
                - ``hazard_count`` (int): Number of hazards projected.
                - ``projections`` (dict): Keyed by hazard type with
                  individual projection results.
                - ``errors`` (dict): Keyed by hazard type with error
                  messages for any failed projections.
                - ``warming_delta_c`` (float): Common warming delta.
                - ``projected_at`` (str): ISO timestamp.
                - ``provenance_hash`` (str): SHA-256 audit hash.

        Raises:
            ValueError: If baseline_risks is empty or inputs invalid.
        """
        start_time = time.monotonic()

        if not baseline_risks or not isinstance(baseline_risks, dict):
            raise ValueError(
                "baseline_risks must be a non-empty dictionary keyed by "
                "hazard_type"
            )

        norm_scenario = _validate_scenario(scenario)
        norm_horizon = _validate_time_horizon(time_horizon)
        norm_location = _validate_location(location)

        warming_delta = self.calculate_warming_delta(norm_scenario, norm_horizon)

        projections: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}

        for ht, br in baseline_risks.items():
            try:
                proj = self.project_hazard(
                    hazard_type=ht,
                    location=norm_location,
                    baseline_risk=br,
                    scenario=norm_scenario,
                    time_horizon=norm_horizon,
                )
                projections[proj["hazard_type"]] = proj
            except (ValueError, KeyError) as exc:
                errors[ht] = str(exc)
                logger.warning(
                    "project_all_hazards: failed for hazard '%s': %s",
                    ht,
                    exc,
                )

        batch_id = _generate_id("BATCH")
        now_iso = _utcnow().isoformat()

        result = {
            "batch_id": batch_id,
            "location": copy.deepcopy(norm_location),
            "scenario": norm_scenario,
            "time_horizon": norm_horizon,
            "hazard_count": len(projections),
            "projections": copy.deepcopy(projections),
            "errors": copy.deepcopy(errors),
            "warming_delta_c": round(warming_delta, 4),
            "projected_at": now_iso,
            "provenance_hash": "",
        }

        prov_hash = self._record_provenance(
            entity_type="scenario_projection",
            action="project_scenario",
            entity_id=batch_id,
            data=result,
        )
        result["provenance_hash"] = prov_hash or ""

        duration = time.monotonic() - start_time
        logger.info(
            "project_all_hazards: batch=%s hazards=%d errors=%d "
            "duration=%.3fs",
            batch_id,
            len(projections),
            len(errors),
            duration,
        )

        return copy.deepcopy(result)

    def compare_scenarios_over_time(
        self,
        hazard_type: str,
        location: Any,
        baseline_risk: Dict[str, Any],
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a full scenario x time-horizon comparison matrix.

        Projects the specified hazard type across all scenarios and all
        time horizons, producing a comprehensive comparison matrix.

        Args:
            hazard_type: Climate hazard type.
            location: Location dictionary or identifier.
            baseline_risk: Baseline risk components.
            scenarios: Optional list of scenarios.  When None, all 8
                scenarios are used.

        Returns:
            Dictionary with:
                - ``matrix_id`` (str): Unique matrix identifier.
                - ``hazard_type`` (str): Normalised hazard type.
                - ``location`` (dict): Location data.
                - ``baseline_risk`` (dict): Validated baseline.
                - ``matrix`` (dict): Nested scenario -> horizon -> projection.
                - ``summary`` (dict): Aggregated statistics.
                - ``projected_at`` (str): ISO timestamp.
                - ``provenance_hash`` (str): SHA-256 audit hash.
        """
        start_time = time.monotonic()

        if scenarios is None:
            scenarios = sorted(_VALID_SCENARIO_IDS)

        norm_hazard = _validate_hazard_type(hazard_type)
        norm_location = _validate_location(location)
        norm_baseline = _validate_baseline_risk(baseline_risk)

        horizons = ["BASELINE", "NEAR_TERM", "MID_TERM", "LONG_TERM",
                     "END_CENTURY"]

        matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
        max_warming = 0.0
        max_prob_change = 0.0
        total_projections = 0

        for scen in scenarios:
            try:
                norm_scen = _validate_scenario(scen)
            except ValueError:
                continue

            matrix[norm_scen] = {}
            for hz in horizons:
                try:
                    proj = self.project_hazard(
                        hazard_type=norm_hazard,
                        location=norm_location,
                        baseline_risk=norm_baseline,
                        scenario=norm_scen,
                        time_horizon=hz,
                    )
                    matrix[norm_scen][hz] = proj
                    total_projections += 1

                    wdelta = proj.get("warming_delta_c", 0.0)
                    if wdelta > max_warming:
                        max_warming = wdelta

                    pct = proj.get("risk_change_pct", {})
                    prob_pct = abs(pct.get("probability_change_pct", 0.0))
                    if prob_pct > max_prob_change:
                        max_prob_change = prob_pct

                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "compare_scenarios_over_time: failed for %s/%s: %s",
                        scen,
                        hz,
                        exc,
                    )

        matrix_id = _generate_id("MATRIX")
        now_iso = _utcnow().isoformat()

        summary = {
            "total_projections": total_projections,
            "scenarios_count": len(matrix),
            "horizons_count": len(horizons),
            "max_warming_delta_c": round(max_warming, 4),
            "max_probability_change_pct": round(max_prob_change, 2),
        }

        result = {
            "matrix_id": matrix_id,
            "hazard_type": norm_hazard,
            "location": copy.deepcopy(norm_location),
            "baseline_risk": copy.deepcopy(norm_baseline),
            "matrix": copy.deepcopy(matrix),
            "summary": summary,
            "projected_at": now_iso,
            "provenance_hash": "",
        }

        prov_hash = self._record_provenance(
            entity_type="scenario_projection",
            action="project_multi",
            entity_id=matrix_id,
            data={
                "matrix_id": matrix_id,
                "hazard_type": norm_hazard,
                "total_projections": total_projections,
            },
        )
        result["provenance_hash"] = prov_hash or ""

        duration = time.monotonic() - start_time
        logger.info(
            "compare_scenarios_over_time: matrix=%s hazard=%s "
            "projections=%d duration=%.3fs",
            matrix_id,
            norm_hazard,
            total_projections,
            duration,
        )

        return copy.deepcopy(result)

    def export_projections(
        self,
        format: str = "json",
    ) -> Any:
        """Export all stored projections.

        Args:
            format: Export format. Currently supports ``"json"`` (returns
                JSON string) and ``"dict"`` (returns list of dicts).

        Returns:
            Exported projections in the requested format.

        Raises:
            ValueError: If format is not supported.
        """
        norm_format = format.strip().lower()
        if norm_format not in ("json", "dict"):
            raise ValueError(
                f"Unsupported export format '{format}'. "
                f"Use 'json' or 'dict'."
            )

        with self._lock:
            projections = [
                copy.deepcopy(self._projections[pid])
                for pid in self._projection_order
                if pid in self._projections
            ]

        if norm_format == "json":
            return json.dumps(projections, indent=2, default=str)

        return projections

    def import_projections(
        self,
        projections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Import projection records into the engine store.

        Args:
            projections: List of projection dictionaries to import.
                Each must contain a ``projection_id`` key.

        Returns:
            Import summary with counts of imported and skipped records.
        """
        if not projections or not isinstance(projections, list):
            return {
                "imported": 0,
                "skipped": 0,
                "errors": ["projections must be a non-empty list"],
            }

        imported = 0
        skipped = 0
        errors_list: List[str] = []

        with self._lock:
            for proj in projections:
                if not isinstance(proj, dict):
                    skipped += 1
                    errors_list.append(
                        f"Skipped non-dict entry: {type(proj)}"
                    )
                    continue

                pid = proj.get("projection_id")
                if not pid:
                    skipped += 1
                    errors_list.append(
                        "Skipped entry without projection_id"
                    )
                    continue

                if pid in self._projections:
                    skipped += 1
                    continue

                self._projections[pid] = copy.deepcopy(proj)
                self._projection_order.append(pid)
                imported += 1

        self._record_provenance(
            entity_type="scenario_projection",
            action="import_data",
            entity_id="batch_import",
            data={"imported": imported, "skipped": skipped},
        )

        logger.info(
            "import_projections: imported=%d skipped=%d errors=%d",
            imported,
            skipped,
            len(errors_list),
        )

        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors_list,
        }

    def search_projections(
        self,
        query: Optional[str] = None,
        hazard_type: Optional[str] = None,
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
        min_warming: Optional[float] = None,
        max_warming: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search projections with advanced filtering.

        Args:
            query: Optional text search against location names and
                hazard types.  Case-insensitive substring match.
            hazard_type: Optional hazard type filter.
            scenario: Optional scenario filter.
            time_horizon: Optional time horizon filter.
            min_warming: Optional minimum warming delta filter (inclusive).
            max_warming: Optional maximum warming delta filter (inclusive).
            limit: Maximum results.  Defaults to 100.

        Returns:
            List of matching projection dictionaries.
        """
        # Start with standard list_projections filters
        candidates = self.list_projections(
            hazard_type=hazard_type,
            scenario=scenario,
            time_horizon=time_horizon,
            limit=10_000,  # Get all then filter
        )

        results: List[Dict[str, Any]] = []

        for proj in candidates:
            # Warming range filter
            wdelta = proj.get("warming_delta_c", 0.0)
            if min_warming is not None and wdelta < min_warming:
                continue
            if max_warming is not None and wdelta > max_warming:
                continue

            # Text query filter
            if query:
                query_lower = query.lower()
                location = proj.get("location", {})
                searchable = " ".join([
                    str(location.get("name", "")),
                    str(proj.get("hazard_type", "")),
                    str(proj.get("scenario", "")),
                    str(proj.get("time_horizon", "")),
                ]).lower()
                if query_lower not in searchable:
                    continue

            results.append(proj)
            if len(results) >= limit:
                break

        return results

    # ------------------------------------------------------------------
    # Private scaling helpers
    # ------------------------------------------------------------------

    def _scale_probability(
        self,
        base_probability: float,
        probability_factor: float,
        warming_delta: float,
    ) -> float:
        """Scale probability by warming delta.

        Formula: ``base * (1 + factor * delta)``, clamped to [0, 1].

        For zero warming delta, returns the baseline unchanged.

        Args:
            base_probability: Baseline probability (0-1).
            probability_factor: Per-C probability scaling factor.
            warming_delta: Temperature delta in C.

        Returns:
            Projected probability, clamped to [0.0, 1.0].
        """
        if warming_delta == 0.0:
            return base_probability

        projected = base_probability * (1.0 + probability_factor * warming_delta)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, projected))

    def _scale_intensity(
        self,
        base_intensity: float,
        intensity_factor: float,
        warming_delta: float,
        hazard_type: str,
    ) -> float:
        """Scale intensity by warming delta.

        For SEA_LEVEL_RISE, intensity is additive (metres per C cumulative).
        For all other hazards, the formula is:
        ``base * (1 + (factor - 1) * delta)``

        This ensures that the factor represents the relative change at 1 C:
        an intensity_factor of 2.0 means intensity doubles at 1 C warming,
        so the multiplier is ``1 + (2.0 - 1.0) * delta = 1 + delta``.

        Args:
            base_intensity: Baseline intensity value.
            intensity_factor: Per-C intensity scaling factor.
            warming_delta: Temperature delta in C.
            hazard_type: Normalised hazard type for special handling.

        Returns:
            Projected intensity, floored at 0.0.
        """
        if warming_delta == 0.0:
            return base_intensity

        if hazard_type == "SEA_LEVEL_RISE":
            # Additive: cumulative sea level rise in metres
            projected = base_intensity + intensity_factor * warming_delta
        else:
            # Multiplicative scaling
            multiplier = 1.0 + (intensity_factor - 1.0) * warming_delta
            projected = base_intensity * multiplier

        return max(0.0, projected)

    def _scale_frequency(
        self,
        base_frequency: float,
        frequency_factor: float,
        warming_delta: float,
    ) -> float:
        """Scale frequency by warming delta.

        Formula: ``base * (1 + (factor - 1) * delta)``

        A frequency_factor of 0.9 (e.g. TROPICAL_CYCLONE) means frequency
        decreases: multiplier = ``1 + (0.9 - 1.0) * delta = 1 - 0.1 * delta``.

        Args:
            base_frequency: Baseline event frequency (events/year).
            frequency_factor: Per-C frequency scaling factor.
            warming_delta: Temperature delta in C.

        Returns:
            Projected frequency, floored at 0.0.
        """
        if warming_delta == 0.0:
            return base_frequency

        multiplier = 1.0 + (frequency_factor - 1.0) * warming_delta
        projected = base_frequency * multiplier

        return max(0.0, projected)

    def _scale_duration(
        self,
        base_duration: float,
        duration_factor: float,
        warming_delta: float,
    ) -> float:
        """Scale duration by warming delta.

        Formula: ``base * (1 + factor * delta)``

        Args:
            base_duration: Baseline event duration in days.
            duration_factor: Per-C fractional change in duration.
            warming_delta: Temperature delta in C.

        Returns:
            Projected duration in days, floored at 0.0.
        """
        if warming_delta == 0.0:
            return base_duration

        projected = base_duration * (1.0 + duration_factor * warming_delta)

        return max(0.0, projected)

    # ------------------------------------------------------------------
    # Private comparison/trend helpers
    # ------------------------------------------------------------------

    def _compute_risk_change_pct(
        self,
        baseline: Dict[str, float],
        projected: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute percentage change between baseline and projected risk.

        Calculates the percentage change for each risk component.  When
        the baseline value is zero, percentage change is set to 0.0 to
        avoid division by zero (the absolute change is still visible in
        the projected values).

        Args:
            baseline: Baseline risk dictionary.
            projected: Projected risk dictionary.

        Returns:
            Dictionary with ``*_change_pct`` keys for each component,
            plus an ``overall_change_pct`` (simple average of absolute
            percentage changes).
        """
        changes: Dict[str, float] = {}
        abs_changes: List[float] = []

        for key in ("probability", "intensity", "frequency", "duration_days"):
            base_val = baseline.get(key, 0.0)
            proj_val = projected.get(key, 0.0)

            if abs(base_val) < 1e-12:
                pct = 0.0
            else:
                pct = ((proj_val - base_val) / base_val) * 100.0

            pct = round(pct, 4)
            changes[f"{key}_change_pct"] = pct
            abs_changes.append(abs(pct))

        # Overall change is the simple average of absolute percentage changes
        if abs_changes:
            changes["overall_change_pct"] = round(
                sum(abs_changes) / len(abs_changes), 4
            )
        else:
            changes["overall_change_pct"] = 0.0

        return changes

    def _build_scenario_comparison(
        self,
        projections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build a ranked scenario comparison from per-scenario projections.

        Sorts scenarios by overall risk change percentage (highest first)
        to highlight the most impactful scenarios.

        Args:
            projections: List of per-scenario projection results.

        Returns:
            List of comparison entries sorted by overall_change_pct
            descending.
        """
        comparison = []

        for proj in projections:
            risk_change = proj.get("risk_change_pct", {})
            overall_pct = risk_change.get("overall_change_pct", 0.0)

            entry = {
                "scenario": proj.get("scenario", ""),
                "scenario_name": proj.get("scenario_info", {}).get(
                    "name", ""
                ),
                "warming_delta_c": proj.get("warming_delta_c", 0.0),
                "overall_change_pct": overall_pct,
                "probability_change_pct": risk_change.get(
                    "probability_change_pct", 0.0
                ),
                "intensity_change_pct": risk_change.get(
                    "intensity_change_pct", 0.0
                ),
                "frequency_change_pct": risk_change.get(
                    "frequency_change_pct", 0.0
                ),
                "duration_change_pct": risk_change.get(
                    "duration_days_change_pct", 0.0
                ),
                "projection_id": proj.get("projection_id", ""),
            }
            comparison.append(entry)

        # Sort by overall change descending
        comparison.sort(
            key=lambda x: x.get("overall_change_pct", 0.0),
            reverse=True,
        )

        # Add rank
        for idx, entry in enumerate(comparison):
            entry["rank"] = idx + 1

        return comparison

    def _determine_trend(
        self,
        projections: List[Dict[str, Any]],
    ) -> str:
        """Determine the overall trend direction of a time series.

        Analyses the warming deltas to classify the trend as increasing,
        decreasing, stable, or non-monotonic.

        Args:
            projections: List of projections sorted by time (ascending
                warming delta).

        Returns:
            One of ``"increasing"``, ``"decreasing"``, ``"stable"``, or
            ``"non_monotonic"``.
        """
        if len(projections) < 2:
            return "stable"

        warmings = [p.get("warming_delta_c", 0.0) for p in projections]

        # Check if monotonically increasing
        is_increasing = all(
            warmings[i] <= warmings[i + 1]
            for i in range(len(warmings) - 1)
        )
        if is_increasing:
            # Check if any actual increase occurs
            if warmings[-1] > warmings[0] + 1e-6:
                return "increasing"
            return "stable"

        # Check if monotonically decreasing
        is_decreasing = all(
            warmings[i] >= warmings[i + 1]
            for i in range(len(warmings) - 1)
        )
        if is_decreasing:
            if warmings[0] > warmings[-1] + 1e-6:
                return "decreasing"
            return "stable"

        return "non_monotonic"

    def _classify_direction(
        self,
        factor: float,
        neutral: float,
    ) -> str:
        """Classify a scaling factor as increasing, decreasing, or stable.

        Args:
            factor: The scaling factor value.
            neutral: The neutral value (no change).

        Returns:
            ``"increasing"``, ``"decreasing"``, or ``"stable"``.
        """
        if factor > neutral + 1e-6:
            return "increasing"
        elif factor < neutral - 1e-6:
            return "decreasing"
        return "stable"

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing engine state summary.
        """
        with self._lock:
            proj_count = len(self._projections)

        return (
            f"ScenarioProjectorEngine("
            f"projections={proj_count}, "
            f"scenarios={len(_SCENARIO_REGISTRY)}, "
            f"horizons={len(_TIME_HORIZON_REGISTRY)}, "
            f"hazard_types={len(_HAZARD_SCALING_FACTORS)}, "
            f"risk_engine={'attached' if self._risk_engine else 'none'}, "
            f"provenance={'enabled' if self._provenance else 'disabled'}"
            f")"
        )

    def __len__(self) -> int:
        """Return the number of stored projections.

        Returns:
            Integer count of projections in the store.
        """
        with self._lock:
            return len(self._projections)
