# -*- coding: utf-8 -*-
"""
Procurement Analysis Workflow
===================================

4-phase energy procurement analysis workflow within PACK-036 Utility
Analysis Pack.  Orchestrates market assessment, load profiling, strategy
development, and procurement report generation for competitive energy
supply procurement.

Phases:
    1. MarketAssessment      -- Analyse current market conditions, benchmark
                                 contract terms, compare fixed vs. index
                                 pricing, evaluate renewable energy options
    2. LoadProfiling          -- Build procurement-grade load profile, calculate
                                 load shape characteristics, hourly factors,
                                 and capacity requirements
    3. StrategyDevelopment    -- Develop procurement strategy: contract
                                 structure, term length, hedging approach,
                                 renewable mix, and risk mitigation
    4. ProcurementReport     -- Generate procurement report with RFP
                                 specifications, evaluation criteria,
                                 and contract term recommendations

The workflow follows GreenLang zero-hallucination principles: all
pricing comparisons use deterministic arithmetic, market benchmarks from
published indices, and contract cost modelling via formula evaluation.
No LLM calls in the numeric computation path.

Schedule: annually / on contract expiry
Estimated duration: 25 minutes

Regulatory References:
    - FERC wholesale market regulations
    - EU Directive 2019/944 (electricity market design)
    - PJM/ERCOT/CAISO wholesale market rules
    - RE100 corporate renewable procurement guidance
    - GHG Protocol Scope 2 market-based methodology

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import RiskLevel

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {k: v for k, v in s.items()
             if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()

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

class ContractStructure(str, Enum):
    """Energy contract structure classification."""
    FIXED_PRICE = "fixed_price"
    INDEX_PLUS = "index_plus"
    BLOCK_AND_INDEX = "block_and_index"
    HEAT_RATE = "heat_rate"
    LOAD_FOLLOWING = "load_following"
    FULL_REQUIREMENTS = "full_requirements"
    SHAPED = "shaped"

class ContractTerm(str, Enum):
    """Contract duration classification."""
    SPOT = "spot"
    SHORT_TERM = "short_term_1_yr"
    MEDIUM_TERM = "medium_term_2_3_yr"
    LONG_TERM = "long_term_5_yr"
    VERY_LONG_TERM = "very_long_term_10_yr"

class RenewableOption(str, Enum):
    """Renewable energy procurement option."""
    NONE = "none"
    GREEN_TARIFF = "green_tariff"
    REC_PURCHASE = "rec_purchase"
    VPPA = "virtual_ppa"
    PHYSICAL_PPA = "physical_ppa"
    ON_SITE = "on_site"
    COMMUNITY_SOLAR = "community_solar"

class MarketRegion(str, Enum):
    """Wholesale market region."""
    PJM = "pjm"
    ERCOT = "ercot"
    CAISO = "caiso"
    NYISO = "nyiso"
    ISO_NE = "iso_ne"
    MISO = "miso"
    SPP = "spp"
    EUROPE = "europe"
    UK = "uk"
    AUSTRALIA = "australia"
    OTHER = "other"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Wholesale energy price benchmarks by region ($/MWh, Q1 2025)
# Source: EIA, AEMO, Ofgem published wholesale indices
WHOLESALE_PRICE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "pjm": {"average": 42.0, "peak": 58.0, "off_peak": 28.0},
    "ercot": {"average": 38.0, "peak": 55.0, "off_peak": 22.0},
    "caiso": {"average": 52.0, "peak": 72.0, "off_peak": 32.0},
    "nyiso": {"average": 48.0, "peak": 65.0, "off_peak": 30.0},
    "iso_ne": {"average": 50.0, "peak": 68.0, "off_peak": 32.0},
    "miso": {"average": 35.0, "peak": 48.0, "off_peak": 24.0},
    "spp": {"average": 32.0, "peak": 45.0, "off_peak": 22.0},
    "europe": {"average": 65.0, "peak": 88.0, "off_peak": 42.0},
    "uk": {"average": 70.0, "peak": 95.0, "off_peak": 45.0},
    "australia": {"average": 55.0, "peak": 75.0, "off_peak": 35.0},
    "other": {"average": 50.0, "peak": 65.0, "off_peak": 30.0},
}

# REC/GO prices by region ($/MWh)
REC_PRICES: Dict[str, Dict[str, float]] = {
    "pjm": {"wind": 2.5, "solar": 15.0, "national": 1.5},
    "ercot": {"wind": 1.0, "solar": 8.0, "national": 1.0},
    "caiso": {"wind": 3.0, "solar": 5.0, "national": 2.0},
    "europe": {"wind": 3.0, "solar": 4.0, "national": 2.5},
    "uk": {"wind": 5.0, "solar": 6.0, "national": 4.0},
    "other": {"wind": 3.0, "solar": 8.0, "national": 2.0},
}

# Contract structure risk profiles
CONTRACT_RISK_PROFILES: Dict[str, Dict[str, Any]] = {
    "fixed_price": {"price_risk": "low", "volume_risk": "high", "typical_premium_pct": 8.0},
    "index_plus": {"price_risk": "high", "volume_risk": "low", "typical_premium_pct": 2.0},
    "block_and_index": {"price_risk": "medium", "volume_risk": "medium", "typical_premium_pct": 5.0},
    "heat_rate": {"price_risk": "medium", "volume_risk": "low", "typical_premium_pct": 4.0},
    "load_following": {"price_risk": "medium", "volume_risk": "low", "typical_premium_pct": 6.0},
    "full_requirements": {"price_risk": "low", "volume_risk": "low", "typical_premium_pct": 10.0},
    "shaped": {"price_risk": "medium", "volume_risk": "medium", "typical_premium_pct": 7.0},
}

# Load shape classification thresholds
LOAD_SHAPE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "flat": (0.85, 1.0),
    "moderate_peaking": (0.60, 0.85),
    "strong_peaking": (0.40, 0.60),
    "extreme_peaking": (0.0, 0.40),
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CurrentContract(BaseModel):
    """Current energy supply contract details.

    Attributes:
        contract_id: Contract identifier.
        supplier_name: Current supplier name.
        contract_structure: Contract pricing structure.
        rate_per_kwh: Current effective rate.
        demand_charge_per_kw: Demand charge rate.
        start_date: Contract start date (YYYY-MM-DD).
        end_date: Contract end date (YYYY-MM-DD).
        annual_volume_kwh: Contracted annual volume.
        renewable_pct: Current renewable content percentage.
        early_termination_fee: ETF amount if applicable.
    """
    contract_id: str = Field(default="")
    supplier_name: str = Field(default="")
    contract_structure: ContractStructure = Field(default=ContractStructure.FIXED_PRICE)
    rate_per_kwh: float = Field(default=0.10, ge=0.0)
    demand_charge_per_kw: float = Field(default=0.0, ge=0.0)
    start_date: str = Field(default="")
    end_date: str = Field(default="")
    annual_volume_kwh: float = Field(default=0.0, ge=0.0)
    renewable_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    early_termination_fee: float = Field(default=0.0, ge=0.0)

class ProcurementRecommendation(BaseModel):
    """A procurement strategy recommendation.

    Attributes:
        recommendation_id: Unique identifier.
        rank: Priority rank.
        contract_structure: Recommended contract structure.
        contract_term: Recommended term length.
        renewable_option: Recommended renewable approach.
        estimated_rate_per_kwh: Estimated achieved rate.
        estimated_annual_cost: Estimated annual cost.
        savings_vs_current: Savings compared to current contract.
        savings_pct: Savings as percentage.
        risk_level: Overall risk assessment.
        rationale: Recommendation rationale.
        key_contract_terms: Suggested contract terms.
    """
    recommendation_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:8]}")
    rank: int = Field(default=0, ge=0)
    contract_structure: str = Field(default="")
    contract_term: str = Field(default="")
    renewable_option: str = Field(default="none")
    estimated_rate_per_kwh: float = Field(default=0.0)
    estimated_annual_cost: float = Field(default=0.0)
    savings_vs_current: float = Field(default=0.0)
    savings_pct: float = Field(default=0.0)
    risk_level: str = Field(default="medium")
    rationale: str = Field(default="")
    key_contract_terms: List[str] = Field(default_factory=list)

class ProcurementAnalysisInput(BaseModel):
    """Input data model for ProcurementAnalysisWorkflow.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        market_region: Wholesale market region.
        current_contract: Current supply contract details.
        annual_consumption_kwh: Annual energy consumption.
        peak_demand_kw: Peak demand in kW.
        load_factor: Current load factor.
        renewable_target_pct: Target renewable energy percentage.
        risk_tolerance: Organisation risk tolerance.
        preferred_term: Preferred contract term length.
        budget_ceiling_per_kwh: Maximum acceptable rate.
        include_renewable_options: Whether to evaluate renewables.
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    market_region: MarketRegion = Field(default=MarketRegion.PJM)
    current_contract: CurrentContract = Field(default_factory=CurrentContract)
    annual_consumption_kwh: float = Field(default=0.0, ge=0.0)
    peak_demand_kw: float = Field(default=0.0, ge=0.0)
    load_factor: float = Field(default=0.50, ge=0.0, le=1.0)
    renewable_target_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_tolerance: RiskLevel = Field(default=RiskLevel.MEDIUM)
    preferred_term: ContractTerm = Field(default=ContractTerm.MEDIUM_TERM)
    budget_ceiling_per_kwh: float = Field(default=0.0, ge=0.0)
    include_renewable_options: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class ProcurementAnalysisResult(BaseModel):
    """Complete result from procurement analysis workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="procurement_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    current_annual_cost: float = Field(default=0.0)
    best_estimated_cost: float = Field(default=0.0)
    potential_savings: float = Field(default=0.0)
    savings_pct: float = Field(default=0.0)
    recommendations: List[ProcurementRecommendation] = Field(default_factory=list)
    market_assessment: Dict[str, Any] = Field(default_factory=dict)
    load_profile: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ProcurementAnalysisWorkflow:
    """
    4-phase energy procurement analysis workflow.

    Evaluates market conditions, load profiles, and contract structures to
    develop optimal procurement strategy. Each phase produces a PhaseResult
    with SHA-256 provenance hash.

    Phases:
        1. MarketAssessment    - Analyse market prices and benchmarks
        2. LoadProfiling       - Build procurement-grade load profile
        3. StrategyDevelopment - Develop procurement strategies
        4. ProcurementReport   - Generate RFP and contract recommendations

    Zero-hallucination: all pricing uses published wholesale indices and
    deterministic contract cost formulas.

    Example:
        >>> wf = ProcurementAnalysisWorkflow()
        >>> inp = ProcurementAnalysisInput(
        ...     facility_id="fac-001",
        ...     annual_consumption_kwh=2000000,
        ...     market_region=MarketRegion.PJM,
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ProcurementAnalysisWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self._market_data: Dict[str, Any] = {}
        self._load_profile: Dict[str, Any] = {}
        self._recommendations: List[ProcurementRecommendation] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(self, input_data: ProcurementAnalysisInput) -> ProcurementAnalysisResult:
        """Execute the 4-phase procurement analysis workflow.

        Args:
            input_data: Validated procurement analysis input.

        Returns:
            ProcurementAnalysisResult with recommendations and market data.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting procurement analysis %s for facility=%s region=%s",
            self.workflow_id, input_data.facility_id, input_data.market_region.value,
        )

        self._phase_results = []
        self._market_data = {}
        self._load_profile = {}
        self._recommendations = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_1_market_assessment(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_2_load_profiling(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_3_strategy_development(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_4_procurement_report(input_data)
            self._phase_results.append(phase4)

            failed_count = sum(
                1 for p in self._phase_results if p.status == PhaseStatus.FAILED
            )
            if failed_count == 0:
                overall_status = WorkflowStatus.COMPLETED
            elif failed_count < len(self._phase_results):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error("Procurement analysis failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        current_cost = input_data.current_contract.rate_per_kwh * input_data.annual_consumption_kwh
        best_cost = self._recommendations[0].estimated_annual_cost if self._recommendations else current_cost
        savings = current_cost - best_cost
        savings_pct = (savings / current_cost * 100.0) if current_cost > 0 else 0.0

        result = ProcurementAnalysisResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            current_annual_cost=round(current_cost, 2),
            best_estimated_cost=round(best_cost, 2),
            potential_savings=round(max(0.0, savings), 2),
            savings_pct=round(max(0.0, savings_pct), 2),
            recommendations=self._recommendations,
            market_assessment=self._market_data,
            load_profile=self._load_profile,
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Procurement analysis %s completed in %.2fs: current=$%.2f "
            "best=$%.2f savings=$%.2f",
            self.workflow_id, elapsed, current_cost, best_cost, savings,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Market Assessment
    # -------------------------------------------------------------------------

    def _phase_1_market_assessment(
        self, input_data: ProcurementAnalysisInput
    ) -> PhaseResult:
        """Analyse current market conditions and benchmark pricing.

        Args:
            input_data: Procurement analysis input.

        Returns:
            PhaseResult with market assessment outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        region = input_data.market_region.value
        prices = WHOLESALE_PRICE_BENCHMARKS.get(
            region, WHOLESALE_PRICE_BENCHMARKS["other"]
        )

        # Convert from $/MWh to $/kWh
        avg_price = prices["average"] / 1000.0
        peak_price = prices["peak"] / 1000.0
        off_peak_price = prices["off_peak"] / 1000.0

        # Compare to current contract
        current_rate = input_data.current_contract.rate_per_kwh
        rate_vs_market = ((current_rate - avg_price) / avg_price * 100.0) if avg_price > 0 else 0.0

        # Market volatility indicator (based on peak/off-peak spread)
        spread = peak_price - off_peak_price
        volatility = "low" if spread < 0.02 else ("medium" if spread < 0.04 else "high")

        # REC pricing
        rec_prices = REC_PRICES.get(region, REC_PRICES.get("other", {}))

        self._market_data = {
            "region": region,
            "wholesale_avg_per_kwh": round(avg_price, 4),
            "wholesale_peak_per_kwh": round(peak_price, 4),
            "wholesale_off_peak_per_kwh": round(off_peak_price, 4),
            "current_rate_per_kwh": round(current_rate, 4),
            "rate_vs_market_pct": round(rate_vs_market, 2),
            "market_volatility": volatility,
            "peak_off_peak_spread": round(spread, 4),
            "rec_prices": {k: round(v / 1000.0, 4) for k, v in rec_prices.items()},
        }

        outputs.update(self._market_data)
        outputs["market_position"] = (
            "above_market" if rate_vs_market > 5.0
            else ("at_market" if abs(rate_vs_market) <= 5.0 else "below_market")
        )

        if rate_vs_market > 10.0:
            warnings.append(
                f"Current rate is {rate_vs_market:.1f}% above market average; "
                f"significant savings opportunity"
            )

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 MarketAssessment: region=%s avg=$%.4f/kWh vs_market=%.1f%% (%.3fs)",
            region, avg_price, rate_vs_market, elapsed,
        )
        return PhaseResult(
            phase_name="market_assessment", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Load Profiling
    # -------------------------------------------------------------------------

    def _phase_2_load_profiling(
        self, input_data: ProcurementAnalysisInput
    ) -> PhaseResult:
        """Build procurement-grade load profile.

        Args:
            input_data: Procurement analysis input.

        Returns:
            PhaseResult with load profile outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_kwh = input_data.annual_consumption_kwh
        peak_kw = input_data.peak_demand_kw
        load_factor = input_data.load_factor

        if annual_kwh == 0:
            annual_kwh = input_data.current_contract.annual_volume_kwh

        if peak_kw == 0 and load_factor > 0:
            peak_kw = annual_kwh / (load_factor * 8760.0)

        if load_factor == 0 and peak_kw > 0:
            load_factor = annual_kwh / (peak_kw * 8760.0)

        # Load shape classification
        load_shape = "flat"
        for shape, (low, high) in LOAD_SHAPE_THRESHOLDS.items():
            if low <= load_factor < high:
                load_shape = shape
                break

        # Estimated TOU distribution based on load shape
        if load_factor >= 0.75:
            on_peak_pct = 30.0
            off_peak_pct = 42.0
        elif load_factor >= 0.55:
            on_peak_pct = 38.0
            off_peak_pct = 35.0
        else:
            on_peak_pct = 45.0
            off_peak_pct = 28.0
        mid_peak_pct = 100.0 - on_peak_pct - off_peak_pct

        # Average demand
        avg_demand = annual_kwh / 8760.0

        self._load_profile = {
            "annual_consumption_kwh": round(annual_kwh, 2),
            "annual_consumption_mwh": round(annual_kwh / 1000.0, 2),
            "peak_demand_kw": round(peak_kw, 2),
            "average_demand_kw": round(avg_demand, 2),
            "load_factor": round(load_factor, 4),
            "load_shape": load_shape,
            "on_peak_pct": on_peak_pct,
            "mid_peak_pct": mid_peak_pct,
            "off_peak_pct": off_peak_pct,
        }

        outputs.update(self._load_profile)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 LoadProfiling: %.0f MWh, %.0f kW, LF=%.2f, shape=%s (%.3fs)",
            annual_kwh / 1000.0, peak_kw, load_factor, load_shape, elapsed,
        )
        return PhaseResult(
            phase_name="load_profiling", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Strategy Development
    # -------------------------------------------------------------------------

    def _phase_3_strategy_development(
        self, input_data: ProcurementAnalysisInput
    ) -> PhaseResult:
        """Develop procurement strategies with cost estimates.

        Args:
            input_data: Procurement analysis input.

        Returns:
            PhaseResult with strategy outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        recommendations: List[ProcurementRecommendation] = []

        annual_kwh = self._load_profile.get("annual_consumption_kwh", 0.0)
        current_rate = input_data.current_contract.rate_per_kwh
        current_cost = current_rate * annual_kwh
        market_avg = self._market_data.get("wholesale_avg_per_kwh", 0.05)

        # Evaluate each contract structure
        for structure in ContractStructure:
            risk_profile = CONTRACT_RISK_PROFILES.get(
                structure.value, CONTRACT_RISK_PROFILES["fixed_price"]
            )
            premium_pct = risk_profile["typical_premium_pct"] / 100.0
            estimated_rate = market_avg * (1.0 + premium_pct)

            # Add renewable premium if target > 0
            renewable_opt = RenewableOption.NONE.value
            if input_data.include_renewable_options and input_data.renewable_target_pct > 0:
                rec_prices = self._market_data.get("rec_prices", {})
                rec_cost = rec_prices.get("wind", 0.003)
                renewable_fraction = input_data.renewable_target_pct / 100.0
                estimated_rate += rec_cost * renewable_fraction
                renewable_opt = RenewableOption.REC_PURCHASE.value

            estimated_cost = estimated_rate * annual_kwh
            savings = current_cost - estimated_cost
            savings_pct = (savings / current_cost * 100.0) if current_cost > 0 else 0.0

            # Risk assessment
            price_risk = risk_profile["price_risk"]
            if input_data.risk_tolerance == RiskLevel.LOW and price_risk == "high":
                risk_level = "high"
            elif input_data.risk_tolerance == RiskLevel.HIGH and price_risk == "low":
                risk_level = "low"
            else:
                risk_level = price_risk

            # Key contract terms
            terms = self._get_contract_terms(
                structure, input_data.preferred_term, annual_kwh
            )

            recommendations.append(ProcurementRecommendation(
                rank=0,
                contract_structure=structure.value,
                contract_term=input_data.preferred_term.value,
                renewable_option=renewable_opt,
                estimated_rate_per_kwh=round(estimated_rate, 4),
                estimated_annual_cost=round(estimated_cost, 2),
                savings_vs_current=round(savings, 2),
                savings_pct=round(savings_pct, 2),
                risk_level=risk_level,
                rationale=self._get_rationale(structure, savings_pct, risk_level),
                key_contract_terms=terms,
            ))

        # Sort by savings descending
        recommendations.sort(key=lambda r: r.savings_vs_current, reverse=True)
        for i, rec in enumerate(recommendations, start=1):
            rec.rank = i

        self._recommendations = recommendations

        outputs["strategies_evaluated"] = len(recommendations)
        outputs["positive_savings_count"] = sum(
            1 for r in recommendations if r.savings_vs_current > 0
        )
        if recommendations:
            outputs["best_structure"] = recommendations[0].contract_structure
            outputs["best_savings"] = round(recommendations[0].savings_vs_current, 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 StrategyDevelopment: %d strategies, best savings=$%.2f (%.3fs)",
            len(recommendations),
            recommendations[0].savings_vs_current if recommendations else 0.0,
            elapsed,
        )
        return PhaseResult(
            phase_name="strategy_development", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _get_contract_terms(
        self,
        structure: ContractStructure,
        term: ContractTerm,
        annual_kwh: float,
    ) -> List[str]:
        """Generate key contract terms for a strategy.

        Args:
            structure: Contract structure type.
            term: Contract term length.
            annual_kwh: Annual consumption volume.

        Returns:
            List of contract term strings.
        """
        terms = [
            f"Contract term: {term.value.replace('_', ' ')}",
            f"Annual volume: {annual_kwh / 1000.0:.0f} MWh",
            "Volume tolerance: +/- 10%",
            "Payment terms: Net 30",
        ]

        if structure == ContractStructure.FIXED_PRICE:
            terms.append("Fixed all-in price for contract duration")
            terms.append("No index exposure; budget certainty")
        elif structure == ContractStructure.INDEX_PLUS:
            terms.append("Index-based pricing with fixed adder")
            terms.append("Monthly settlement against day-ahead index")
        elif structure == ContractStructure.BLOCK_AND_INDEX:
            terms.append("70% fixed block, 30% index exposure")
            terms.append("Partial price certainty with upside potential")

        return terms

    def _get_rationale(
        self,
        structure: ContractStructure,
        savings_pct: float,
        risk_level: str,
    ) -> str:
        """Generate deterministic rationale for a recommendation.

        Args:
            structure: Contract structure type.
            savings_pct: Savings percentage.
            risk_level: Risk level assessment.

        Returns:
            Rationale string.
        """
        rationales: Dict[str, str] = {
            "fixed_price": "Budget certainty with full price protection",
            "index_plus": "Lowest cost with full market exposure",
            "block_and_index": "Balanced approach with partial price protection",
            "heat_rate": "Gas-linked pricing for gas-heavy regions",
            "load_following": "Volume flexibility for variable loads",
            "full_requirements": "Comprehensive supply with no volume risk",
            "shaped": "Custom load-shaped pricing for optimal match",
        }
        base = rationales.get(structure.value, "Competitive energy supply option")
        if savings_pct > 10.0:
            return f"{base}. Significant savings potential of {savings_pct:.0f}%."
        elif savings_pct > 0:
            return f"{base}. Moderate savings of {savings_pct:.0f}%."
        return f"{base}. At or above current cost."

    # -------------------------------------------------------------------------
    # Phase 4: Procurement Report
    # -------------------------------------------------------------------------

    def _phase_4_procurement_report(
        self, input_data: ProcurementAnalysisInput
    ) -> PhaseResult:
        """Generate procurement report with RFP specifications.

        Args:
            input_data: Procurement analysis input.

        Returns:
            PhaseResult with report outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"

        # RFP specifications
        rfp_specs: Dict[str, Any] = {
            "volume_mwh": round(
                self._load_profile.get("annual_consumption_mwh", 0.0), 0
            ),
            "peak_demand_kw": round(
                self._load_profile.get("peak_demand_kw", 0.0), 0
            ),
            "load_factor": self._load_profile.get("load_factor", 0.50),
            "contract_term": input_data.preferred_term.value,
            "renewable_target_pct": input_data.renewable_target_pct,
            "market_region": input_data.market_region.value,
            "delivery_point": "utility_meter",
            "settlement": "monthly",
        }

        # Evaluation criteria
        eval_criteria = [
            {"criterion": "Price competitiveness", "weight_pct": 40},
            {"criterion": "Contract flexibility", "weight_pct": 15},
            {"criterion": "Renewable content", "weight_pct": 15},
            {"criterion": "Supplier creditworthiness", "weight_pct": 15},
            {"criterion": "Risk management features", "weight_pct": 15},
        ]

        # Timeline
        timeline = [
            {"phase": "RFP issuance", "timeline": "Week 1"},
            {"phase": "Supplier questions", "timeline": "Week 2"},
            {"phase": "Bid submission deadline", "timeline": "Week 4"},
            {"phase": "Bid evaluation", "timeline": "Weeks 5-6"},
            {"phase": "Finalist presentations", "timeline": "Week 7"},
            {"phase": "Contract negotiation", "timeline": "Weeks 8-10"},
            {"phase": "Contract execution", "timeline": "Week 12"},
        ]

        outputs["report_id"] = report_id
        outputs["generated_at"] = utcnow().isoformat()
        outputs["rfp_specifications"] = rfp_specs
        outputs["evaluation_criteria"] = eval_criteria
        outputs["procurement_timeline"] = timeline
        outputs["recommendation_count"] = len(self._recommendations)
        outputs["methodology"] = [
            "Wholesale market price benchmarking (EIA/AEMO/Ofgem indices)",
            "Load shape analysis for contract structure matching",
            "Contract structure cost modelling with risk premiums",
            "REC/GO pricing for renewable procurement options",
            "Risk-adjusted savings comparison",
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 ProcurementReport: report=%s, %d criteria (%.3fs)",
            report_id, len(eval_criteria), elapsed,
        )
        return PhaseResult(
            phase_name="procurement_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )
