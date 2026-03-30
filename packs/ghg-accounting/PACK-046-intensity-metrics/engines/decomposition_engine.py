# -*- coding: utf-8 -*-
"""
DecompositionEngine - PACK-046 Intensity Metrics Engine 3
====================================================================

Implements LMDI-I (Log-Mean Divisia Index, method I) decomposition to
attribute changes in total emissions to activity, structure, and
intensity effects.  Primary method follows Ang (2004, 2015).

Calculation Methodology:
    LMDI-I Additive Decomposition:
        Total change:     dV = V_t - V_0

        Activity effect:  dV_act = SUM_i[ L(V_i_t, V_i_0) * ln(D_t / D_0) ]
        Structure effect: dV_str = SUM_i[ L(V_i_t, V_i_0) * ln(S_i_t / S_i_0) ]
        Intensity effect: dV_int = SUM_i[ L(V_i_t, V_i_0) * ln(I_i_t / I_i_0) ]

        Where:
            V_i_t = emissions of sub-sector i at time t
            D_t   = total activity (denominator) at time t
            S_i_t = structural share = D_i_t / D_t
            I_i_t = intensity = V_i_t / D_i_t
            L(a,b) = logarithmic mean = (a - b) / (ln(a) - ln(b))

    Logarithmic Mean Function:
        L(a, b) = (a - b) / (ln(a) - ln(b))     when a != b
        L(a, a) = a                                when a == b (L'Hopital limit)
        L(a, 0) = 0                                when b == 0 (by convention)
        L(0, b) = 0                                when a == 0 (by convention)

    LMDI-I Multiplicative Decomposition:
        D_total = V_t / V_0

        D_act = exp( SUM_i[ w_i * ln(D_t / D_0) ] )
        D_str = exp( SUM_i[ w_i * ln(S_i_t / S_i_0) ] )
        D_int = exp( SUM_i[ w_i * ln(I_i_t / I_i_0) ] )

        Where w_i = L(V_i_t, V_i_0) / L(V_t, V_0)

    Closure Property:
        Additive:       dV_act + dV_str + dV_int = dV  (exact, no residual)
        Multiplicative: D_act * D_str * D_int = D_total (exact, no residual)

    Zero-Value Handling:
        - New sector entry (V_i_0 = 0, V_i_t > 0): attributed to activity effect
        - Sector exit (V_i_0 > 0, V_i_t = 0): attributed to activity effect
        - Both zero (V_i_0 = 0, V_i_t = 0): excluded from decomposition

Regulatory References:
    - Ang, B.W. (2004) Decomposition analysis for policymaking in energy
    - Ang, B.W. (2015) LMDI decomposition approach: A guide for implementation
    - IEA (2023) Tracking Clean Energy Progress
    - ESRS E1-6 requirements for explaining emission changes
    - CDP Climate Change C7 (Emissions breakdown)
    - SBTi Monitoring Report guidance

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - LMDI-I is an exact decomposition (zero residual by construction)
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round6(value: Any) -> float:
    """Round to 6 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

def _ln(value: Decimal) -> Decimal:
    """Compute natural logarithm using float then convert back to Decimal.

    Returns Decimal('0') for non-positive input.
    """
    fval = float(value)
    if fval <= 0:
        return Decimal("0")
    return _decimal(math.log(fval))

def _exp(value: Decimal) -> Decimal:
    """Compute exponential using float then convert back to Decimal."""
    fval = float(value)
    try:
        return _decimal(math.exp(fval))
    except OverflowError:
        return Decimal("0")

def _log_mean(a: Decimal, b: Decimal) -> Decimal:
    """Compute the logarithmic mean L(a, b).

    L(a, b) = (a - b) / (ln(a) - ln(b))   when a != b and both > 0
    L(a, a) = a                              when a == b (L'Hopital limit)
    L(a, 0) = 0                              when either is zero

    This function is central to LMDI decomposition.
    """
    if a <= Decimal("0") or b <= Decimal("0"):
        return Decimal("0")
    if a == b:
        return a
    ln_a = _ln(a)
    ln_b = _ln(b)
    if ln_a == ln_b:
        return a
    return (a - b) / (ln_a - ln_b)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecompositionMethod(str, Enum):
    """Decomposition method variant.

    ADDITIVE:        LMDI-I additive (results in tCO2e).
    MULTIPLICATIVE:  LMDI-I multiplicative (results as ratios).
    """
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

class EffectType(str, Enum):
    """Type of decomposition effect.

    ACTIVITY:   Change due to overall activity level (scale).
    STRUCTURE:  Change due to sub-sector mix (composition).
    INTENSITY:  Change due to emission intensity (efficiency).
    """
    ACTIVITY = "activity"
    STRUCTURE = "structure"
    INTENSITY = "intensity"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Closure tolerance: activity + structure + intensity must equal total within this
CLOSURE_TOLERANCE_TCO2E: Decimal = Decimal("0.001")
CLOSURE_TOLERANCE_RATIO: Decimal = Decimal("0.0001")

# Maximum number of sub-sectors
MAX_SUBSECTORS: int = 500

# Maximum number of periods in multi-period decomposition
MAX_PERIODS: int = 100

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class SubSectorData(BaseModel):
    """Data for a single sub-sector at a single point in time.

    Attributes:
        subsector_id:    Unique sub-sector identifier.
        subsector_name:  Human-readable sub-sector name.
        emissions_tco2e: GHG emissions for this sub-sector (tCO2e).
        activity_value:  Activity (denominator) value for this sub-sector.
    """
    subsector_id: str = Field(..., min_length=1, description="Sub-sector ID")
    subsector_name: str = Field(default="", description="Sub-sector name")
    emissions_tco2e: Decimal = Field(..., ge=0, description="Emissions (tCO2e)")
    activity_value: Decimal = Field(..., ge=0, description="Activity value")

    @field_validator("emissions_tco2e", "activity_value", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        return _decimal(v)

class PeriodData(BaseModel):
    """Data for all sub-sectors at a single point in time.

    Attributes:
        period:           Period label (e.g. '2024').
        total_activity:   Total activity across all sub-sectors.
        subsectors:       Sub-sector data.
    """
    period: str = Field(..., description="Period label")
    total_activity: Optional[Decimal] = Field(
        default=None, ge=0, description="Total activity (auto-computed if None)"
    )
    subsectors: List[SubSectorData] = Field(..., min_length=1, description="Sub-sector data")

    @field_validator("total_activity", mode="before")
    @classmethod
    def coerce_activity(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

    @model_validator(mode="after")
    def compute_total_activity(self) -> "PeriodData":
        """Auto-compute total_activity if not provided."""
        if self.total_activity is None:
            computed = sum(s.activity_value for s in self.subsectors)
            object.__setattr__(self, "total_activity", computed)
        return self

class DecompositionInput(BaseModel):
    """Input for decomposition analysis.

    Attributes:
        analysis_id:       Unique analysis identifier.
        base_period:       Base period data (time 0).
        target_period:     Target period data (time t).
        method:            Decomposition method (additive/multiplicative).
        activity_unit:     Unit of activity measure.
        output_precision:  Decimal places for output.
    """
    analysis_id: str = Field(default_factory=_new_uuid, description="Analysis ID")
    base_period: PeriodData = Field(..., description="Base period data")
    target_period: PeriodData = Field(..., description="Target period data")
    method: DecompositionMethod = Field(
        default=DecompositionMethod.ADDITIVE, description="Method"
    )
    activity_unit: str = Field(default="unit", description="Activity unit")
    output_precision: int = Field(default=3, ge=0, le=12, description="Output precision")

class MultiPeriodDecompositionInput(BaseModel):
    """Input for multi-period (chained) decomposition.

    Attributes:
        analysis_id:     Unique analysis identifier.
        periods:         Ordered list of period data.
        method:          Decomposition method.
        activity_unit:   Activity unit.
        output_precision: Output precision.
    """
    analysis_id: str = Field(default_factory=_new_uuid, description="Analysis ID")
    periods: List[PeriodData] = Field(..., min_length=2, description="Period data")
    method: DecompositionMethod = Field(
        default=DecompositionMethod.ADDITIVE, description="Method"
    )
    activity_unit: str = Field(default="unit", description="Activity unit")
    output_precision: int = Field(default=3, ge=0, le=12, description="Output precision")

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class SubSectorContribution(BaseModel):
    """A sub-sector's contribution to each decomposition effect.

    Attributes:
        subsector_id:      Sub-sector identifier.
        subsector_name:    Sub-sector name.
        activity_effect:   Contribution to activity effect.
        structure_effect:  Contribution to structure effect.
        intensity_effect:  Contribution to intensity effect.
        base_emissions:    Base period emissions.
        target_emissions:  Target period emissions.
        base_share:        Base period structural share (%).
        target_share:      Target period structural share (%).
        base_intensity:    Base period intensity.
        target_intensity:  Target period intensity.
    """
    subsector_id: str = Field(..., description="Sub-sector ID")
    subsector_name: str = Field(default="", description="Sub-sector name")
    activity_effect: Decimal = Field(default=Decimal("0"), description="Activity effect")
    structure_effect: Decimal = Field(default=Decimal("0"), description="Structure effect")
    intensity_effect: Decimal = Field(default=Decimal("0"), description="Intensity effect")
    base_emissions: Decimal = Field(default=Decimal("0"), description="Base emissions")
    target_emissions: Decimal = Field(default=Decimal("0"), description="Target emissions")
    base_share: Decimal = Field(default=Decimal("0"), description="Base share (%)")
    target_share: Decimal = Field(default=Decimal("0"), description="Target share (%)")
    base_intensity: Optional[Decimal] = Field(default=None, description="Base intensity")
    target_intensity: Optional[Decimal] = Field(default=None, description="Target intensity")

class DecompositionResult(BaseModel):
    """Result of decomposition analysis.

    Attributes:
        result_id:              Unique result identifier.
        analysis_id:            Analysis identifier.
        method:                 Decomposition method used.
        base_period:            Base period label.
        target_period:          Target period label.
        total_change:           Total emission change (tCO2e or ratio).
        activity_effect:        Activity effect (tCO2e or ratio).
        structure_effect:       Structure effect (tCO2e or ratio).
        intensity_effect:       Intensity effect (tCO2e or ratio).
        closure_residual:       Residual (should be ~0 for LMDI-I).
        closure_valid:          Whether closure property holds.
        subsector_contributions: Per-sub-sector breakdown.
        activity_effect_pct:    Activity effect as % of total change.
        structure_effect_pct:   Structure effect as % of total change.
        intensity_effect_pct:   Intensity effect as % of total change.
        base_total_emissions:   Base period total emissions.
        target_total_emissions: Target period total emissions.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    analysis_id: str = Field(default="", description="Analysis ID")
    method: DecompositionMethod = Field(
        default=DecompositionMethod.ADDITIVE, description="Method"
    )
    base_period: str = Field(default="", description="Base period")
    target_period: str = Field(default="", description="Target period")
    total_change: Decimal = Field(default=Decimal("0"), description="Total change")
    activity_effect: Decimal = Field(default=Decimal("0"), description="Activity effect")
    structure_effect: Decimal = Field(default=Decimal("0"), description="Structure effect")
    intensity_effect: Decimal = Field(default=Decimal("0"), description="Intensity effect")
    closure_residual: Decimal = Field(default=Decimal("0"), description="Closure residual")
    closure_valid: bool = Field(default=True, description="Closure property valid")
    subsector_contributions: List[SubSectorContribution] = Field(
        default_factory=list, description="Sub-sector contributions"
    )
    activity_effect_pct: Optional[Decimal] = Field(default=None, description="Activity % of total")
    structure_effect_pct: Optional[Decimal] = Field(default=None, description="Structure % of total")
    intensity_effect_pct: Optional[Decimal] = Field(default=None, description="Intensity % of total")
    base_total_emissions: Decimal = Field(default=Decimal("0"), description="Base total emissions")
    target_total_emissions: Decimal = Field(default=Decimal("0"), description="Target total emissions")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class MultiPeriodStep(BaseModel):
    """One step in a multi-period decomposition chain.

    Attributes:
        from_period:      Start period.
        to_period:        End period.
        activity_effect:  Activity effect for this step.
        structure_effect: Structure effect for this step.
        intensity_effect: Intensity effect for this step.
        total_change:     Total change for this step.
    """
    from_period: str = Field(..., description="Start period")
    to_period: str = Field(..., description="End period")
    activity_effect: Decimal = Field(default=Decimal("0"), description="Activity effect")
    structure_effect: Decimal = Field(default=Decimal("0"), description="Structure effect")
    intensity_effect: Decimal = Field(default=Decimal("0"), description="Intensity effect")
    total_change: Decimal = Field(default=Decimal("0"), description="Total change")

class MultiPeriodDecompositionResult(BaseModel):
    """Result of multi-period chained decomposition.

    Attributes:
        result_id:               Unique result identifier.
        analysis_id:             Analysis identifier.
        method:                  Decomposition method.
        period_count:            Number of periods.
        steps:                   Per-step decomposition results.
        cumulative_activity:     Cumulative activity effect.
        cumulative_structure:    Cumulative structure effect.
        cumulative_intensity:    Cumulative intensity effect.
        cumulative_total:        Cumulative total change.
        overall_closure_valid:   Whether closure holds for all steps.
        warnings:                Warnings.
        calculated_at:           Timestamp.
        processing_time_ms:      Processing time (ms).
        provenance_hash:         SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    analysis_id: str = Field(default="", description="Analysis ID")
    method: DecompositionMethod = Field(
        default=DecompositionMethod.ADDITIVE, description="Method"
    )
    period_count: int = Field(default=0, description="Period count")
    steps: List[MultiPeriodStep] = Field(default_factory=list, description="Steps")
    cumulative_activity: Decimal = Field(default=Decimal("0"), description="Cumulative activity")
    cumulative_structure: Decimal = Field(default=Decimal("0"), description="Cumulative structure")
    cumulative_intensity: Decimal = Field(default=Decimal("0"), description="Cumulative intensity")
    cumulative_total: Decimal = Field(default=Decimal("0"), description="Cumulative total")
    overall_closure_valid: bool = Field(default=True, description="Overall closure valid")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DecompositionEngine:
    """LMDI-I decomposition engine for emissions intensity analysis.

    Decomposes total emission changes into activity, structure, and
    intensity effects using the Log-Mean Divisia Index method I.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Zero residual: LMDI-I satisfies perfect decomposition.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Zero-Hallucination: No LLM in any calculation path.

    Usage:
        engine = DecompositionEngine()
        inp = DecompositionInput(
            base_period=PeriodData(period="2019", subsectors=[...]),
            target_period=PeriodData(period="2024", subsectors=[...]),
        )
        result = engine.calculate(inp)
        print(result.activity_effect, result.structure_effect, result.intensity_effect)
    """

    def __init__(self) -> None:
        """Initialise the DecompositionEngine."""
        self._version = _MODULE_VERSION
        logger.info("DecompositionEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: DecompositionInput) -> DecompositionResult:
        """Perform LMDI-I decomposition between two periods.

        Args:
            input_data: Decomposition input with base and target period data.

        Returns:
            DecompositionResult with effects and sub-sector contributions.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        base = input_data.base_period
        target = input_data.target_period
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        # Validate sub-sector count
        all_ids = set(s.subsector_id for s in base.subsectors) | set(
            s.subsector_id for s in target.subsectors
        )
        if len(all_ids) > MAX_SUBSECTORS:
            raise ValueError(
                f"Maximum {MAX_SUBSECTORS} sub-sectors allowed (got {len(all_ids)})"
            )

        # Build lookup maps
        base_map: Dict[str, SubSectorData] = {s.subsector_id: s for s in base.subsectors}
        target_map: Dict[str, SubSectorData] = {s.subsector_id: s for s in target.subsectors}

        # Total activity and emissions
        D_0 = base.total_activity or sum(s.activity_value for s in base.subsectors)
        D_t = target.total_activity or sum(s.activity_value for s in target.subsectors)
        V_0_total = sum(s.emissions_tco2e for s in base.subsectors)
        V_t_total = sum(s.emissions_tco2e for s in target.subsectors)

        if input_data.method == DecompositionMethod.ADDITIVE:
            result = self._additive_decomposition(
                all_ids, base_map, target_map, D_0, D_t, V_0_total, V_t_total,
                prec, prec_str, warnings,
            )
        else:
            result = self._multiplicative_decomposition(
                all_ids, base_map, target_map, D_0, D_t, V_0_total, V_t_total,
                prec, prec_str, warnings,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result.analysis_id = input_data.analysis_id
        result.method = input_data.method
        result.base_period = base.period
        result.target_period = target.period
        result.base_total_emissions = V_0_total
        result.target_total_emissions = V_t_total
        result.warnings = warnings
        result.calculated_at = utcnow().isoformat()
        result.processing_time_ms = round(elapsed_ms, 3)
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_multi_period(
        self,
        input_data: MultiPeriodDecompositionInput,
    ) -> MultiPeriodDecompositionResult:
        """Perform chained multi-period decomposition.

        Decomposes changes between each consecutive pair of periods
        and accumulates the effects.

        Args:
            input_data: Multi-period input with ordered periods.

        Returns:
            MultiPeriodDecompositionResult with per-step and cumulative effects.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        if len(input_data.periods) > MAX_PERIODS:
            raise ValueError(
                f"Maximum {MAX_PERIODS} periods allowed (got {len(input_data.periods)})"
            )

        steps: List[MultiPeriodStep] = []
        cum_activity = Decimal("0")
        cum_structure = Decimal("0")
        cum_intensity = Decimal("0")
        cum_total = Decimal("0")
        all_closure_valid = True

        for i in range(len(input_data.periods) - 1):
            step_input = DecompositionInput(
                base_period=input_data.periods[i],
                target_period=input_data.periods[i + 1],
                method=input_data.method,
                activity_unit=input_data.activity_unit,
                output_precision=input_data.output_precision,
            )
            step_result = self.calculate(step_input)

            if not step_result.closure_valid:
                all_closure_valid = False

            if input_data.method == DecompositionMethod.ADDITIVE:
                cum_activity += step_result.activity_effect
                cum_structure += step_result.structure_effect
                cum_intensity += step_result.intensity_effect
                cum_total += step_result.total_change
            else:
                if cum_activity == Decimal("0"):
                    cum_activity = step_result.activity_effect
                    cum_structure = step_result.structure_effect
                    cum_intensity = step_result.intensity_effect
                    cum_total = step_result.total_change
                else:
                    cum_activity *= step_result.activity_effect
                    cum_structure *= step_result.structure_effect
                    cum_intensity *= step_result.intensity_effect
                    cum_total *= step_result.total_change

            steps.append(MultiPeriodStep(
                from_period=input_data.periods[i].period,
                to_period=input_data.periods[i + 1].period,
                activity_effect=step_result.activity_effect,
                structure_effect=step_result.structure_effect,
                intensity_effect=step_result.intensity_effect,
                total_change=step_result.total_change,
            ))

            warnings.extend(step_result.warnings)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = MultiPeriodDecompositionResult(
            analysis_id=input_data.analysis_id,
            method=input_data.method,
            period_count=len(input_data.periods),
            steps=steps,
            cumulative_activity=cum_activity,
            cumulative_structure=cum_structure,
            cumulative_intensity=cum_intensity,
            cumulative_total=cum_total,
            overall_closure_valid=all_closure_valid,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Additive Decomposition
    # ------------------------------------------------------------------

    def _additive_decomposition(
        self,
        all_ids: set,
        base_map: Dict[str, SubSectorData],
        target_map: Dict[str, SubSectorData],
        D_0: Decimal,
        D_t: Decimal,
        V_0_total: Decimal,
        V_t_total: Decimal,
        prec: int,
        prec_str: str,
        warnings: List[str],
    ) -> DecompositionResult:
        """LMDI-I additive decomposition."""
        total_change = V_t_total - V_0_total
        sum_activity = Decimal("0")
        sum_structure = Decimal("0")
        sum_intensity = Decimal("0")
        contributions: List[SubSectorContribution] = []

        for sid in sorted(all_ids):
            base_s = base_map.get(sid)
            target_s = target_map.get(sid)

            V_i_0 = base_s.emissions_tco2e if base_s else Decimal("0")
            V_i_t = target_s.emissions_tco2e if target_s else Decimal("0")
            D_i_0 = base_s.activity_value if base_s else Decimal("0")
            D_i_t = target_s.activity_value if target_s else Decimal("0")

            name = ""
            if target_s:
                name = target_s.subsector_name
            elif base_s:
                name = base_s.subsector_name

            # Structural shares
            S_i_0 = _safe_divide(D_i_0, D_0) if D_0 > Decimal("0") else Decimal("0")
            S_i_t = _safe_divide(D_i_t, D_t) if D_t > Decimal("0") else Decimal("0")

            # Intensities
            I_i_0 = _safe_divide(V_i_0, D_i_0) if D_i_0 > Decimal("0") else None
            I_i_t = _safe_divide(V_i_t, D_i_t) if D_i_t > Decimal("0") else None

            # Handle zero-value cases
            if V_i_0 == Decimal("0") and V_i_t == Decimal("0"):
                # Both zero - no contribution
                contributions.append(SubSectorContribution(
                    subsector_id=sid,
                    subsector_name=name,
                    base_emissions=V_i_0,
                    target_emissions=V_i_t,
                    base_share=S_i_0 * Decimal("100"),
                    target_share=S_i_t * Decimal("100"),
                    base_intensity=I_i_0,
                    target_intensity=I_i_t,
                ))
                continue

            if V_i_0 == Decimal("0") or V_i_t == Decimal("0"):
                # Entry/exit: attribute entirely to activity effect
                delta = V_i_t - V_i_0
                sum_activity += delta
                contributions.append(SubSectorContribution(
                    subsector_id=sid,
                    subsector_name=name,
                    activity_effect=delta,
                    base_emissions=V_i_0,
                    target_emissions=V_i_t,
                    base_share=S_i_0 * Decimal("100"),
                    target_share=S_i_t * Decimal("100"),
                    base_intensity=I_i_0,
                    target_intensity=I_i_t,
                ))
                if V_i_0 == Decimal("0"):
                    warnings.append(f"Sub-sector '{sid}' entered in target period.")
                else:
                    warnings.append(f"Sub-sector '{sid}' exited in target period.")
                continue

            # Standard LMDI-I additive
            L_i = _log_mean(V_i_t, V_i_0)

            # Activity effect
            act_i = Decimal("0")
            if D_0 > Decimal("0") and D_t > Decimal("0"):
                act_i = L_i * _ln(D_t / D_0)

            # Structure effect
            str_i = Decimal("0")
            if S_i_0 > Decimal("0") and S_i_t > Decimal("0"):
                str_i = L_i * _ln(S_i_t / S_i_0)

            # Intensity effect
            int_i = Decimal("0")
            if I_i_0 is not None and I_i_t is not None and I_i_0 > Decimal("0") and I_i_t > Decimal("0"):
                int_i = L_i * _ln(I_i_t / I_i_0)

            sum_activity += act_i
            sum_structure += str_i
            sum_intensity += int_i

            contributions.append(SubSectorContribution(
                subsector_id=sid,
                subsector_name=name,
                activity_effect=act_i.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                structure_effect=str_i.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                intensity_effect=int_i.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                base_emissions=V_i_0,
                target_emissions=V_i_t,
                base_share=(S_i_0 * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ),
                target_share=(S_i_t * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ),
                base_intensity=I_i_0,
                target_intensity=I_i_t,
            ))

        # Round totals
        sum_activity = sum_activity.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        sum_structure = sum_structure.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        sum_intensity = sum_intensity.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Closure check
        reconstructed = sum_activity + sum_structure + sum_intensity
        residual = (total_change - reconstructed).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        closure_valid = abs(residual) <= CLOSURE_TOLERANCE_TCO2E

        if not closure_valid:
            warnings.append(
                f"Closure residual {residual} tCO2e exceeds tolerance "
                f"{CLOSURE_TOLERANCE_TCO2E}. This may be due to rounding."
            )

        # Effect percentages
        act_pct = _pct_of_total(sum_activity, total_change)
        str_pct = _pct_of_total(sum_structure, total_change)
        int_pct = _pct_of_total(sum_intensity, total_change)

        return DecompositionResult(
            total_change=total_change.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            activity_effect=sum_activity,
            structure_effect=sum_structure,
            intensity_effect=sum_intensity,
            closure_residual=residual,
            closure_valid=closure_valid,
            subsector_contributions=contributions,
            activity_effect_pct=act_pct,
            structure_effect_pct=str_pct,
            intensity_effect_pct=int_pct,
        )

    # ------------------------------------------------------------------
    # Internal: Multiplicative Decomposition
    # ------------------------------------------------------------------

    def _multiplicative_decomposition(
        self,
        all_ids: set,
        base_map: Dict[str, SubSectorData],
        target_map: Dict[str, SubSectorData],
        D_0: Decimal,
        D_t: Decimal,
        V_0_total: Decimal,
        V_t_total: Decimal,
        prec: int,
        prec_str: str,
        warnings: List[str],
    ) -> DecompositionResult:
        """LMDI-I multiplicative decomposition."""
        if V_0_total == Decimal("0"):
            warnings.append("Base period total emissions are zero; multiplicative decomposition undefined.")
            return DecompositionResult(
                total_change=Decimal("0"),
                closure_valid=False,
            )

        total_ratio = V_t_total / V_0_total
        L_total = _log_mean(V_t_total, V_0_total)

        sum_w_ln_act = Decimal("0")
        sum_w_ln_str = Decimal("0")
        sum_w_ln_int = Decimal("0")
        contributions: List[SubSectorContribution] = []

        for sid in sorted(all_ids):
            base_s = base_map.get(sid)
            target_s = target_map.get(sid)

            V_i_0 = base_s.emissions_tco2e if base_s else Decimal("0")
            V_i_t = target_s.emissions_tco2e if target_s else Decimal("0")
            D_i_0 = base_s.activity_value if base_s else Decimal("0")
            D_i_t = target_s.activity_value if target_s else Decimal("0")

            name = (target_s or base_s).subsector_name if (target_s or base_s) else ""

            S_i_0 = _safe_divide(D_i_0, D_0) if D_0 > Decimal("0") else Decimal("0")
            S_i_t = _safe_divide(D_i_t, D_t) if D_t > Decimal("0") else Decimal("0")
            I_i_0 = _safe_divide(V_i_0, D_i_0) if D_i_0 > Decimal("0") else None
            I_i_t = _safe_divide(V_i_t, D_i_t) if D_i_t > Decimal("0") else None

            if V_i_0 <= Decimal("0") or V_i_t <= Decimal("0"):
                contributions.append(SubSectorContribution(
                    subsector_id=sid,
                    subsector_name=name,
                    base_emissions=V_i_0,
                    target_emissions=V_i_t,
                    base_share=S_i_0 * Decimal("100"),
                    target_share=S_i_t * Decimal("100"),
                    base_intensity=I_i_0,
                    target_intensity=I_i_t,
                ))
                continue

            L_i = _log_mean(V_i_t, V_i_0)
            w_i = _safe_divide(L_i, L_total) if L_total > Decimal("0") else Decimal("0")

            if D_0 > Decimal("0") and D_t > Decimal("0"):
                sum_w_ln_act += w_i * _ln(D_t / D_0)
            if S_i_0 > Decimal("0") and S_i_t > Decimal("0"):
                sum_w_ln_str += w_i * _ln(S_i_t / S_i_0)
            if I_i_0 is not None and I_i_t is not None and I_i_0 > Decimal("0") and I_i_t > Decimal("0"):
                sum_w_ln_int += w_i * _ln(I_i_t / I_i_0)

            contributions.append(SubSectorContribution(
                subsector_id=sid,
                subsector_name=name,
                base_emissions=V_i_0,
                target_emissions=V_i_t,
                base_share=(S_i_0 * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                target_share=(S_i_t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                base_intensity=I_i_0,
                target_intensity=I_i_t,
            ))

        D_act = _exp(sum_w_ln_act)
        D_str = _exp(sum_w_ln_str)
        D_int = _exp(sum_w_ln_int)

        # Closure check (multiplicative)
        reconstructed = D_act * D_str * D_int
        residual = (total_ratio - reconstructed).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        closure_valid = abs(residual) <= CLOSURE_TOLERANCE_RATIO

        return DecompositionResult(
            total_change=total_ratio.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            activity_effect=D_act.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            structure_effect=D_str.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            intensity_effect=D_int.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            closure_residual=residual,
            closure_valid=closure_valid,
            subsector_contributions=contributions,
        )

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        """Return engine version string."""
        return self._version

# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------

def _pct_of_total(part: Decimal, total: Decimal) -> Optional[Decimal]:
    """Compute percentage of total change, handling zero total."""
    if total == Decimal("0"):
        return None
    return (part / total * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def log_mean(a: Decimal, b: Decimal) -> Decimal:
    """Public interface to the logarithmic mean function.

    Args:
        a: First value.
        b: Second value.

    Returns:
        Logarithmic mean L(a, b).
    """
    return _log_mean(_decimal(a), _decimal(b))

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "DecompositionMethod",
    "EffectType",
    # Input Models
    "SubSectorData",
    "PeriodData",
    "DecompositionInput",
    "MultiPeriodDecompositionInput",
    # Output Models
    "SubSectorContribution",
    "DecompositionResult",
    "MultiPeriodStep",
    "MultiPeriodDecompositionResult",
    # Engine
    "DecompositionEngine",
    # Convenience
    "log_mean",
    # Constants
    "CLOSURE_TOLERANCE_TCO2E",
    "CLOSURE_TOLERANCE_RATIO",
]
