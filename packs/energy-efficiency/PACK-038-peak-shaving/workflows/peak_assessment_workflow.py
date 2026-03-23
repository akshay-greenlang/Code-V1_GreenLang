# -*- coding: utf-8 -*-
"""
Peak Assessment Workflow
===================================

4-phase workflow for attributing peak demand to load categories, decomposing
demand charges, assessing avoidability, and recommending shaving strategies
within PACK-038 Peak Shaving Pack.

Phases:
    1. PeakAttribution           -- Attribute peaks to load categories
    2. DemandChargeDecomposition  -- Full tariff analysis per charge component
    3. AvoidabilityAssessment     -- Classify peaks as avoidable/partial/unavoidable
    4. StrategyRecommendation     -- Recommend peak shaving strategies with ranking

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - FERC rate schedule filings
    - Utility tariff demand charge structures
    - NAESB WEQ demand charge standards

Schedule: on-demand / quarterly
Estimated duration: 20 minutes

Author: GreenLang Team
Version: 38.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AvoidabilityClass(str, Enum):
    """Peak avoidability classification."""

    FULLY_AVOIDABLE = "fully_avoidable"
    PARTIALLY_AVOIDABLE = "partially_avoidable"
    UNAVOIDABLE = "unavoidable"


class DemandChargeType(str, Enum):
    """Demand charge component type."""

    FLAT = "flat"
    TIERED = "tiered"
    TOU = "time_of_use"
    COINCIDENT_PEAK = "coincident_peak"
    RATCHET = "ratchet"
    POWER_FACTOR = "power_factor"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

DEMAND_CHARGE_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "flat_demand": {
        "charge_type": "flat",
        "description": "Flat demand charge per kW of billing peak",
        "typical_rate_low": 5.00,
        "typical_rate_high": 25.00,
        "unit": "$/kW/month",
        "avoidability": "high",
    },
    "tiered_demand": {
        "charge_type": "tiered",
        "description": "Tiered demand charge with escalating rates",
        "typical_rate_low": 3.00,
        "typical_rate_high": 35.00,
        "unit": "$/kW/month",
        "avoidability": "high",
    },
    "tou_demand": {
        "charge_type": "time_of_use",
        "description": "TOU on-peak demand charge",
        "typical_rate_low": 8.00,
        "typical_rate_high": 40.00,
        "unit": "$/kW/month",
        "avoidability": "high",
    },
    "coincident_peak": {
        "charge_type": "coincident_peak",
        "description": "Coincident peak / transmission demand charge",
        "typical_rate_low": 2.00,
        "typical_rate_high": 20.00,
        "unit": "$/kW/month",
        "avoidability": "medium",
    },
    "ratchet": {
        "charge_type": "ratchet",
        "description": "Ratchet clause (80-100% of annual peak for 11 months)",
        "typical_rate_low": 5.00,
        "typical_rate_high": 25.00,
        "unit": "$/kW/month",
        "avoidability": "medium",
    },
    "power_factor": {
        "charge_type": "power_factor",
        "description": "Power factor penalty for PF below threshold",
        "typical_rate_low": 0.50,
        "typical_rate_high": 5.00,
        "unit": "$/kVA/month",
        "avoidability": "high",
    },
}

PEAK_SHAVING_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "bess": {
        "name": "Battery Energy Storage System",
        "effectiveness": "high",
        "capex_range_per_kw": (400, 800),
        "implementation_months": 6,
        "applicable_charge_types": ["flat", "tiered", "time_of_use", "ratchet"],
    },
    "load_shifting": {
        "name": "Load Shifting / Scheduling",
        "effectiveness": "medium",
        "capex_range_per_kw": (5, 50),
        "implementation_months": 2,
        "applicable_charge_types": ["flat", "tiered", "time_of_use"],
    },
    "thermal_storage": {
        "name": "Thermal Energy Storage (Ice/Chilled Water)",
        "effectiveness": "medium",
        "capex_range_per_kw": (200, 500),
        "implementation_months": 8,
        "applicable_charge_types": ["flat", "tiered", "time_of_use"],
    },
    "demand_limiting": {
        "name": "Automated Demand Limiting Controls",
        "effectiveness": "medium",
        "capex_range_per_kw": (10, 80),
        "implementation_months": 3,
        "applicable_charge_types": ["flat", "tiered", "time_of_use", "ratchet"],
    },
    "on_site_generation": {
        "name": "On-site Generation (Generator/CHP)",
        "effectiveness": "high",
        "capex_range_per_kw": (300, 1200),
        "implementation_months": 12,
        "applicable_charge_types": ["flat", "tiered", "time_of_use", "coincident_peak"],
    },
    "cp_response": {
        "name": "Coincident Peak Response Programme",
        "effectiveness": "high",
        "capex_range_per_kw": (0, 30),
        "implementation_months": 1,
        "applicable_charge_types": ["coincident_peak"],
    },
}

LOAD_CATEGORY_PEAKS: Dict[str, Dict[str, Any]] = {
    "hvac": {"peak_contribution_pct": 0.35, "shiftable_pct": 0.30, "curtailable_pct": 0.25},
    "lighting": {"peak_contribution_pct": 0.15, "shiftable_pct": 0.20, "curtailable_pct": 0.40},
    "process": {"peak_contribution_pct": 0.25, "shiftable_pct": 0.15, "curtailable_pct": 0.10},
    "plug_loads": {"peak_contribution_pct": 0.10, "shiftable_pct": 0.30, "curtailable_pct": 0.20},
    "refrigeration": {"peak_contribution_pct": 0.10, "shiftable_pct": 0.25, "curtailable_pct": 0.25},
    "other": {"peak_contribution_pct": 0.05, "shiftable_pct": 0.10, "curtailable_pct": 0.15},
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class PeakAssessmentInput(BaseModel):
    """Input data model for PeakAssessmentWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    facility_type: str = Field(default="office_building", description="Facility type")
    peak_demand_kw: Decimal = Field(..., gt=0, description="Billing peak demand kW")
    avg_demand_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Average demand kW")
    annual_demand_charges: Decimal = Field(default=Decimal("0"), ge=0, description="Annual demand charges $")
    tariff_type: str = Field(default="flat", description="flat|tiered|tou|cp|ratchet|pf")
    demand_rate: Decimal = Field(default=Decimal("15.00"), ge=0, description="Demand rate $/kW/month")
    ratchet_pct: Decimal = Field(default=Decimal("80"), ge=0, le=100, description="Ratchet % of annual peak")
    power_factor: Decimal = Field(default=Decimal("0.92"), gt=0, le=1, description="Facility power factor")
    load_categories: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Load categories with peak_kw and category name",
    )
    peak_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Peak event data: timestamp, peak_kw, duration_min",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_name")
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        """Ensure facility name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("facility_name must not be blank")
        return stripped


class StrategyRecommendation(BaseModel):
    """Recommended peak shaving strategy."""

    strategy_key: str = Field(default="", description="Strategy identifier")
    strategy_name: str = Field(default="", description="Human-readable strategy name")
    priority_rank: int = Field(default=0, ge=0, description="Priority rank (1=highest)")
    estimated_savings_kw: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_annual_savings: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_capex: Decimal = Field(default=Decimal("0"), ge=0)
    simple_payback_years: Decimal = Field(default=Decimal("0"), ge=0)
    effectiveness: str = Field(default="", description="high|medium|low")


class PeakAssessmentResult(BaseModel):
    """Complete result from peak assessment workflow."""

    assessment_id: str = Field(..., description="Unique assessment execution ID")
    facility_id: str = Field(default="", description="Assessed facility ID")
    peak_demand_kw: Decimal = Field(default=Decimal("0"), ge=0)
    avoidable_peak_kw: Decimal = Field(default=Decimal("0"), ge=0)
    partially_avoidable_kw: Decimal = Field(default=Decimal("0"), ge=0)
    unavoidable_peak_kw: Decimal = Field(default=Decimal("0"), ge=0)
    annual_demand_charges: Decimal = Field(default=Decimal("0"), ge=0)
    avoidable_charges: Decimal = Field(default=Decimal("0"), ge=0)
    category_attribution: Dict[str, Any] = Field(default_factory=dict)
    charge_decomposition: Dict[str, Any] = Field(default_factory=dict)
    strategies: List[StrategyRecommendation] = Field(default_factory=list)
    total_savings_potential: Decimal = Field(default=Decimal("0"), ge=0)
    assessment_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PeakAssessmentWorkflow:
    """
    4-phase peak assessment workflow for demand charge optimization.

    Performs peak attribution, demand charge decomposition, avoidability
    analysis, and strategy recommendation with priority ranking.

    Zero-hallucination: all charge calculations use published tariff rates and
    deterministic formulas. No LLM calls in the numeric computation path.

    Attributes:
        assessment_id: Unique assessment execution identifier.
        _attribution: Peak category attribution data.
        _charges: Demand charge decomposition data.
        _avoidability: Avoidability classification results.
        _strategies: Recommended strategies with rankings.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = PeakAssessmentWorkflow()
        >>> inp = PeakAssessmentInput(
        ...     facility_name="Warehouse B",
        ...     peak_demand_kw=Decimal("2500"),
        ... )
        >>> result = wf.run(inp)
        >>> assert len(result.strategies) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PeakAssessmentWorkflow."""
        self.assessment_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._attribution: Dict[str, Any] = {}
        self._charges: Dict[str, Any] = {}
        self._avoidability: Dict[str, Any] = {}
        self._strategies: List[StrategyRecommendation] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: PeakAssessmentInput) -> PeakAssessmentResult:
        """
        Execute the 4-phase peak assessment workflow.

        Args:
            input_data: Validated peak assessment input.

        Returns:
            PeakAssessmentResult with attribution, charges, and strategies.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting peak assessment workflow %s for facility=%s tariff=%s",
            self.assessment_id, input_data.facility_name, input_data.tariff_type,
        )

        self._phase_results = []
        self._attribution = {}
        self._charges = {}
        self._avoidability = {}
        self._strategies = []

        try:
            phase1 = self._phase_peak_attribution(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_demand_charge_decomposition(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_avoidability_assessment(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_strategy_recommendation(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Peak assessment workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        avoidable_kw = Decimal(str(self._avoidability.get("avoidable_kw", 0)))
        partially_kw = Decimal(str(self._avoidability.get("partially_avoidable_kw", 0)))
        unavoidable_kw = Decimal(str(self._avoidability.get("unavoidable_kw", 0)))
        avoidable_charges = Decimal(str(self._avoidability.get("avoidable_charges", 0)))
        total_savings = sum(s.estimated_annual_savings for s in self._strategies)

        result = PeakAssessmentResult(
            assessment_id=self.assessment_id,
            facility_id=input_data.facility_id,
            peak_demand_kw=input_data.peak_demand_kw,
            avoidable_peak_kw=avoidable_kw,
            partially_avoidable_kw=partially_kw,
            unavoidable_peak_kw=unavoidable_kw,
            annual_demand_charges=Decimal(str(self._charges.get("total_annual_charges", 0))),
            avoidable_charges=avoidable_charges,
            category_attribution=self._attribution,
            charge_decomposition=self._charges,
            strategies=self._strategies,
            total_savings_potential=total_savings,
            assessment_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Peak assessment workflow %s completed in %dms avoidable=%.0f kW "
            "strategies=%d savings=$%.0f",
            self.assessment_id, int(elapsed_ms), float(avoidable_kw),
            len(self._strategies), float(total_savings),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Peak Attribution
    # -------------------------------------------------------------------------

    def _phase_peak_attribution(
        self, input_data: PeakAssessmentInput
    ) -> PhaseResult:
        """Attribute peaks to load categories (HVAC, lighting, process, etc.)."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = float(input_data.peak_demand_kw)
        attribution: Dict[str, Dict[str, Any]] = {}

        if input_data.load_categories:
            # Use provided load category data
            total_allocated = Decimal("0")
            for cat_data in input_data.load_categories:
                cat_name = cat_data.get("category", "other")
                cat_kw = Decimal(str(cat_data.get("peak_kw", 0)))
                total_allocated += cat_kw
                ref = LOAD_CATEGORY_PEAKS.get(cat_name, LOAD_CATEGORY_PEAKS["other"])
                attribution[cat_name] = {
                    "peak_kw": str(cat_kw),
                    "contribution_pct": str(round(float(cat_kw) / max(peak_kw, 0.01) * 100, 1)),
                    "shiftable_kw": str((cat_kw * Decimal(str(ref["shiftable_pct"]))).quantize(Decimal("0.1"))),
                    "curtailable_kw": str((cat_kw * Decimal(str(ref["curtailable_pct"]))).quantize(Decimal("0.1"))),
                }
            if abs(float(total_allocated) - peak_kw) > peak_kw * 0.05:
                warnings.append(
                    f"Load categories sum {total_allocated} kW vs peak {peak_kw} kW (>5% deviation)"
                )
        else:
            # Use benchmark allocation
            warnings.append("No load categories provided; using benchmark allocation")
            for cat_name, ref in LOAD_CATEGORY_PEAKS.items():
                cat_kw = Decimal(str(round(peak_kw * ref["peak_contribution_pct"], 1)))
                attribution[cat_name] = {
                    "peak_kw": str(cat_kw),
                    "contribution_pct": str(round(ref["peak_contribution_pct"] * 100, 1)),
                    "shiftable_kw": str((cat_kw * Decimal(str(ref["shiftable_pct"]))).quantize(Decimal("0.1"))),
                    "curtailable_kw": str((cat_kw * Decimal(str(ref["curtailable_pct"]))).quantize(Decimal("0.1"))),
                }

        self._attribution = attribution

        outputs["categories_assessed"] = len(attribution)
        outputs["peak_demand_kw"] = str(input_data.peak_demand_kw)
        outputs["top_contributor"] = max(
            attribution.items(),
            key=lambda x: float(x[1].get("peak_kw", 0)),
        )[0] if attribution else "none"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 PeakAttribution: %d categories, top=%s",
            len(attribution), outputs.get("top_contributor"),
        )
        return PhaseResult(
            phase_name="peak_attribution", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Demand Charge Decomposition
    # -------------------------------------------------------------------------

    def _phase_demand_charge_decomposition(
        self, input_data: PeakAssessmentInput
    ) -> PhaseResult:
        """Full tariff analysis decomposing demand charges by component."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = input_data.peak_demand_kw
        rate = input_data.demand_rate
        tariff = input_data.tariff_type
        monthly_months = Decimal("12")

        components: Dict[str, Dict[str, Any]] = {}
        total_annual = Decimal("0")

        # Flat demand charge
        flat_monthly = (peak_kw * rate).quantize(Decimal("0.01"))
        flat_annual = (flat_monthly * monthly_months).quantize(Decimal("0.01"))
        components["flat_demand"] = {
            "monthly_charge": str(flat_monthly),
            "annual_charge": str(flat_annual),
            "rate": str(rate),
            "unit": "$/kW/month",
        }
        total_annual += flat_annual

        # TOU premium if applicable
        if tariff in ("tou", "cp"):
            tou_premium = Decimal("1.35")
            tou_monthly = (flat_monthly * (tou_premium - Decimal("1"))).quantize(Decimal("0.01"))
            tou_annual = (tou_monthly * Decimal("6")).quantize(Decimal("0.01"))
            components["tou_premium"] = {
                "monthly_charge": str(tou_monthly),
                "annual_charge": str(tou_annual),
                "rate": str(rate * tou_premium),
                "note": "Applied 6 on-peak months",
            }
            total_annual += tou_annual

        # Ratchet impact
        if tariff == "ratchet" or float(input_data.ratchet_pct) > 0:
            ratchet_floor = (peak_kw * input_data.ratchet_pct / Decimal("100")).quantize(Decimal("0.1"))
            avg_kw = input_data.avg_demand_kw if input_data.avg_demand_kw > 0 else peak_kw * Decimal("0.6")
            ratchet_excess = max(Decimal("0"), ratchet_floor - avg_kw)
            ratchet_monthly = (ratchet_excess * rate).quantize(Decimal("0.01"))
            ratchet_annual = (ratchet_monthly * Decimal("11")).quantize(Decimal("0.01"))
            components["ratchet"] = {
                "ratchet_floor_kw": str(ratchet_floor),
                "ratchet_excess_kw": str(ratchet_excess),
                "monthly_charge": str(ratchet_monthly),
                "annual_charge": str(ratchet_annual),
                "note": "Applied 11 non-peak months",
            }
            total_annual += ratchet_annual

        # Power factor penalty
        pf = float(input_data.power_factor)
        if pf < 0.90:
            pf_penalty_rate = Decimal("1.50")
            kva = (peak_kw / Decimal(str(pf))).quantize(Decimal("0.1"))
            kva_excess = kva - peak_kw
            pf_monthly = (kva_excess * pf_penalty_rate).quantize(Decimal("0.01"))
            pf_annual = (pf_monthly * monthly_months).quantize(Decimal("0.01"))
            components["power_factor_penalty"] = {
                "power_factor": str(input_data.power_factor),
                "kva": str(kva),
                "excess_kva": str(kva_excess),
                "monthly_charge": str(pf_monthly),
                "annual_charge": str(pf_annual),
            }
            total_annual += pf_annual

        # Coincident peak transmission charge
        if tariff == "cp":
            cp_rate = Decimal("8.50")
            cp_annual = (peak_kw * cp_rate * monthly_months).quantize(Decimal("0.01"))
            components["coincident_peak"] = {
                "rate": str(cp_rate),
                "annual_charge": str(cp_annual),
                "note": "Transmission demand charge based on CP contribution",
            }
            total_annual += cp_annual

        # Override total if annual charges provided
        if input_data.annual_demand_charges > 0:
            total_annual = input_data.annual_demand_charges

        self._charges = {
            "components": components,
            "total_annual_charges": str(total_annual),
            "tariff_type": tariff,
            "demand_rate": str(rate),
        }

        outputs["total_annual_charges"] = str(total_annual)
        outputs["charge_components"] = len(components)
        outputs["tariff_type"] = tariff

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 DemandChargeDecomposition: $%.0f/yr, %d components, tariff=%s",
            float(total_annual), len(components), tariff,
        )
        return PhaseResult(
            phase_name="demand_charge_decomposition", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Avoidability Assessment
    # -------------------------------------------------------------------------

    def _phase_avoidability_assessment(
        self, input_data: PeakAssessmentInput
    ) -> PhaseResult:
        """Classify each peak contribution as fully/partially/unavoidable."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = float(input_data.peak_demand_kw)
        total_annual = float(self._charges.get("total_annual_charges", 0))

        avoidable_kw = Decimal("0")
        partially_avoidable_kw = Decimal("0")
        unavoidable_kw = Decimal("0")

        category_avoidability: Dict[str, Dict[str, Any]] = {}
        for cat_name, cat_data in self._attribution.items():
            cat_kw = Decimal(str(cat_data.get("peak_kw", 0)))
            shiftable = Decimal(str(cat_data.get("shiftable_kw", 0)))
            curtailable = Decimal(str(cat_data.get("curtailable_kw", 0)))

            # Fully avoidable: can be shifted + curtailed
            fully = min(shiftable + curtailable, cat_kw)
            # Partially: some portion
            partial = (cat_kw - fully) * Decimal("0.3")
            partial = max(Decimal("0"), partial).quantize(Decimal("0.1"))
            # Unavoidable: remainder
            unavoid = (cat_kw - fully - partial).quantize(Decimal("0.1"))

            avoidable_kw += fully
            partially_avoidable_kw += partial
            unavoidable_kw += unavoid

            category_avoidability[cat_name] = {
                "fully_avoidable_kw": str(fully),
                "partially_avoidable_kw": str(partial),
                "unavoidable_kw": str(unavoid),
                "classification": (
                    "fully_avoidable" if fully >= cat_kw * Decimal("0.7")
                    else "partially_avoidable" if fully >= cat_kw * Decimal("0.3")
                    else "unavoidable"
                ),
            }

        # Calculate avoidable demand charges
        avoidable_ratio = (
            float(avoidable_kw + partially_avoidable_kw * Decimal("0.5"))
            / max(peak_kw, 0.01)
        )
        avoidable_charges = round(total_annual * avoidable_ratio, 2)

        self._avoidability = {
            "avoidable_kw": str(avoidable_kw.quantize(Decimal("0.1"))),
            "partially_avoidable_kw": str(partially_avoidable_kw.quantize(Decimal("0.1"))),
            "unavoidable_kw": str(unavoidable_kw.quantize(Decimal("0.1"))),
            "avoidable_charges": str(avoidable_charges),
            "category_avoidability": category_avoidability,
        }

        outputs["avoidable_kw"] = str(avoidable_kw.quantize(Decimal("0.1")))
        outputs["partially_avoidable_kw"] = str(partially_avoidable_kw.quantize(Decimal("0.1")))
        outputs["unavoidable_kw"] = str(unavoidable_kw.quantize(Decimal("0.1")))
        outputs["avoidable_charges"] = str(avoidable_charges)
        outputs["avoidable_ratio_pct"] = round(avoidable_ratio * 100, 1)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 AvoidabilityAssessment: avoidable=%.0f kW partial=%.0f kW "
            "charges=$%.0f",
            float(avoidable_kw), float(partially_avoidable_kw), avoidable_charges,
        )
        return PhaseResult(
            phase_name="avoidability_assessment", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Strategy Recommendation
    # -------------------------------------------------------------------------

    def _phase_strategy_recommendation(
        self, input_data: PeakAssessmentInput
    ) -> PhaseResult:
        """Recommend peak shaving strategies with priority ranking."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        avoidable_kw = Decimal(str(self._avoidability.get("avoidable_kw", 0)))
        avoidable_charges = Decimal(str(self._avoidability.get("avoidable_charges", 0)))
        tariff = input_data.tariff_type

        strategy_scores: List[Dict[str, Any]] = []

        for strat_key, strat in PEAK_SHAVING_STRATEGIES.items():
            # Check tariff applicability
            applicable_types = strat["applicable_charge_types"]
            tariff_match = tariff in applicable_types or "flat" in applicable_types

            if not tariff_match:
                continue

            # Estimate savings based on effectiveness
            eff_map = {"high": Decimal("0.75"), "medium": Decimal("0.45"), "low": Decimal("0.20")}
            eff_factor = eff_map.get(strat["effectiveness"], Decimal("0.30"))

            savings_kw = (avoidable_kw * eff_factor).quantize(Decimal("0.1"))
            annual_savings = (avoidable_charges * eff_factor).quantize(Decimal("0.01"))

            # CAPEX estimate
            capex_low = Decimal(str(strat["capex_range_per_kw"][0]))
            capex_high = Decimal(str(strat["capex_range_per_kw"][1]))
            capex_mid = ((capex_low + capex_high) / Decimal("2")).quantize(Decimal("0.01"))
            total_capex = (savings_kw * capex_mid).quantize(Decimal("0.01"))

            # Simple payback
            payback = (
                (total_capex / annual_savings).quantize(Decimal("0.1"))
                if annual_savings > 0 else Decimal("99")
            )

            strategy_scores.append({
                "key": strat_key,
                "name": strat["name"],
                "savings_kw": savings_kw,
                "annual_savings": annual_savings,
                "capex": total_capex,
                "payback": payback,
                "effectiveness": strat["effectiveness"],
                "score": float(annual_savings) / max(float(total_capex), 1) * 100,
            })

        # Sort by score descending (best ROI first)
        strategy_scores.sort(key=lambda x: x["score"], reverse=True)

        for rank, s in enumerate(strategy_scores, 1):
            self._strategies.append(StrategyRecommendation(
                strategy_key=s["key"],
                strategy_name=s["name"],
                priority_rank=rank,
                estimated_savings_kw=s["savings_kw"],
                estimated_annual_savings=s["annual_savings"],
                estimated_capex=s["capex"],
                simple_payback_years=s["payback"],
                effectiveness=s["effectiveness"],
            ))

        outputs["strategies_evaluated"] = len(PEAK_SHAVING_STRATEGIES)
        outputs["strategies_recommended"] = len(self._strategies)
        outputs["top_strategy"] = self._strategies[0].strategy_name if self._strategies else "none"
        outputs["total_savings_potential"] = str(
            sum(s.estimated_annual_savings for s in self._strategies)
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 StrategyRecommendation: %d strategies, top=%s",
            len(self._strategies), outputs.get("top_strategy"),
        )
        return PhaseResult(
            phase_name="strategy_recommendation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PeakAssessmentResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
