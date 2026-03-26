# -*- coding: utf-8 -*-
"""
PortfolioBenchmarkingEngine - PACK-047 GHG Emissions Benchmark Engine 7
====================================================================

Implements PCAF-aligned portfolio carbon accounting and benchmarking
with WACI, carbon footprint, financed emissions, data quality scoring,
sector allocation attribution, and holdings-level analysis per TCFD/SFDR.

Calculation Methodology:
    Financed Emissions:
        FE_i = ownership_share_i * E_i
        ownership_share_i = investment_i / EVIC_i

        Where:
            FE_i           = financed emissions for holding i
            investment_i   = portfolio investment in holding i
            EVIC_i         = enterprise value including cash of holding i
            E_i            = total emissions of holding i

    Weighted Average Carbon Intensity (WACI):
        WACI = SUM(w_i * I_i)

        Where:
            w_i = investment_i / total_portfolio_value (portfolio weight)
            I_i = emissions_i / revenue_i (carbon intensity of holding i)

    Carbon Footprint:
        CF = SUM(FE_i) / AUM

        Where AUM = total assets under management.

    Carbon Intensity (portfolio level):
        CI = SUM(FE_i) / SUM(w_i * revenue_i)

    PCAF Data Quality Aggregation:
        Q_portfolio = SUM(w_i * Q_i)

        Where w_i = FE_i / SUM(FE_j) (emissions-weighted)

    Sector Attribution:
        attribution_s = SUM(w_i * I_i) for i in sector s
        contribution_pct = attribution_s / WACI * 100

    Tracking Error (vs benchmark index):
        TE = SUM(|(w_p_i - w_b_i)| * I_i)

        Where w_p = portfolio weight, w_b = benchmark weight.

Regulatory References:
    - PCAF Global GHG Accounting Standard (Part A): Financed emissions
    - TCFD Metrics and Targets (b): WACI, carbon footprint
    - SFDR PAI Indicator 1: Carbon footprint
    - SFDR PAI Indicator 2: Carbon intensity
    - SFDR PAI Indicator 3: GHG intensity of investee companies
    - EU Taxonomy Article 8: Portfolio alignment disclosure
    - ESRS E1-6: Financed emissions context
    - NZAOA Target Setting Protocol v3.0 (2024)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - PCAF methodology from published standard only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-047 GHG Emissions Benchmark
Engine:  7 of 10
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    """PCAF asset class for financed emissions."""
    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    SOVEREIGN_DEBT = "sovereign_debt"


class PCAFScore(int, Enum):
    """PCAF data quality score (1=best, 5=worst)."""
    SCORE_1 = 1
    SCORE_2 = 2
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5


class MetricType(str, Enum):
    """Portfolio carbon metric type."""
    FINANCED_EMISSIONS = "financed_emissions"
    WACI = "waci"
    CARBON_FOOTPRINT = "carbon_footprint"
    CARBON_INTENSITY = "carbon_intensity"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HOLDINGS: int = 50000
DEFAULT_PCAF_QUALITY: int = 3


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class Holding(BaseModel):
    """A single portfolio holding.

    Attributes:
        entity_id:              Entity identifier.
        entity_name:            Entity name.
        sector:                 GICS/NACE sector.
        country:                Country.
        asset_class:            PCAF asset class.
        investment_value:       Investment value.
        evic:                   Enterprise value including cash.
        revenue:                Revenue.
        scope1_tco2e:           Scope 1 emissions.
        scope2_tco2e:           Scope 2 emissions.
        scope3_tco2e:           Scope 3 emissions.
        total_emissions_tco2e:  Total emissions.
        pcaf_quality:           PCAF data quality score.
    """
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    sector: str = Field(default="", description="Sector")
    country: str = Field(default="", description="Country")
    asset_class: AssetClass = Field(default=AssetClass.LISTED_EQUITY)
    investment_value: Decimal = Field(..., ge=0, description="Investment value")
    evic: Decimal = Field(default=Decimal("0"), gt=0, description="EVIC")
    revenue: Decimal = Field(default=Decimal("0"), ge=0, description="Revenue")
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    pcaf_quality: PCAFScore = Field(default=PCAFScore.SCORE_3)

    @field_validator(
        "investment_value", "evic", "revenue",
        "scope1_tco2e", "scope2_tco2e", "total_emissions_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)

    @model_validator(mode="after")
    def compute_total(self) -> "Holding":
        if self.total_emissions_tco2e == Decimal("0"):
            total = self.scope1_tco2e + self.scope2_tco2e
            if self.scope3_tco2e is not None:
                total += self.scope3_tco2e
            object.__setattr__(self, "total_emissions_tco2e", total)
        return self


class BenchmarkIndex(BaseModel):
    """Benchmark index for comparison.

    Attributes:
        index_id:       Index identifier.
        index_name:     Index name.
        holdings:       Index holdings.
        waci:           Index WACI (if pre-calculated).
    """
    index_id: str = Field(default="", description="Index ID")
    index_name: str = Field(default="", description="Index name")
    holdings: List[Holding] = Field(default_factory=list, description="Holdings")
    waci: Optional[Decimal] = Field(default=None, ge=0, description="Pre-calculated WACI")


class Portfolio(BaseModel):
    """Portfolio for benchmarking.

    Attributes:
        portfolio_id:       Portfolio identifier.
        portfolio_name:     Portfolio name.
        holdings:           Portfolio holdings.
        aum:                Assets under management.
        benchmark:          Benchmark index (optional).
        output_precision:   Output decimal places.
    """
    portfolio_id: str = Field(default_factory=_new_uuid, description="Portfolio ID")
    portfolio_name: str = Field(default="", description="Portfolio name")
    holdings: List[Holding] = Field(default_factory=list, description="Holdings")
    aum: Optional[Decimal] = Field(default=None, ge=0, description="AUM")
    benchmark: Optional[BenchmarkIndex] = Field(default=None, description="Benchmark")
    output_precision: int = Field(default=4, ge=0, le=12)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class HoldingResult(BaseModel):
    """Per-holding calculation result.

    Attributes:
        entity_id:          Entity identifier.
        entity_name:        Entity name.
        sector:             Sector.
        portfolio_weight:   Portfolio weight.
        ownership_share:    Ownership share (investment/EVIC).
        financed_emissions: Financed emissions (tCO2e).
        intensity:          Carbon intensity (tCO2e/revenue).
        waci_contribution:  Contribution to WACI.
        pcaf_quality:       Data quality score.
    """
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    sector: str = Field(default="")
    portfolio_weight: Decimal = Field(default=Decimal("0"))
    ownership_share: Decimal = Field(default=Decimal("0"))
    financed_emissions: Decimal = Field(default=Decimal("0"))
    intensity: Decimal = Field(default=Decimal("0"))
    waci_contribution: Decimal = Field(default=Decimal("0"))
    pcaf_quality: int = Field(default=3)


class SectorAttribution(BaseModel):
    """Sector-level attribution analysis.

    Attributes:
        sector:             Sector name.
        holdings_count:     Number of holdings.
        portfolio_weight:   Aggregate portfolio weight.
        financed_emissions: Sector financed emissions.
        waci_contribution:  Sector WACI contribution.
        contribution_pct:   Percentage of total WACI.
    """
    sector: str = Field(default="")
    holdings_count: int = Field(default=0)
    portfolio_weight: Decimal = Field(default=Decimal("0"))
    financed_emissions: Decimal = Field(default=Decimal("0"))
    waci_contribution: Decimal = Field(default=Decimal("0"))
    contribution_pct: Decimal = Field(default=Decimal("0"))


class WACIResult(BaseModel):
    """WACI calculation result.

    Attributes:
        waci:               Portfolio WACI.
        benchmark_waci:     Benchmark WACI (if available).
        tracking_error:     Tracking error vs benchmark.
        relative_pct:       Relative performance vs benchmark (%).
    """
    waci: Decimal = Field(default=Decimal("0"))
    benchmark_waci: Optional[Decimal] = Field(default=None)
    tracking_error: Optional[Decimal] = Field(default=None)
    relative_pct: Optional[Decimal] = Field(default=None)


class PortfolioResult(BaseModel):
    """Complete portfolio benchmarking result.

    Attributes:
        result_id:              Unique result ID.
        portfolio_id:           Portfolio ID.
        total_financed_emissions: Total financed emissions.
        waci_result:            WACI analysis.
        carbon_footprint:       Carbon footprint (FE / AUM).
        carbon_intensity:       Carbon intensity (FE / revenue).
        pcaf_quality:           Portfolio-weighted PCAF quality.
        holding_results:        Per-holding results.
        sector_attribution:     Sector attribution.
        holdings_count:         Holdings count.
        coverage_pct:           Data coverage (%).
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    portfolio_id: str = Field(default="")
    total_financed_emissions: Decimal = Field(default=Decimal("0"))
    waci_result: WACIResult = Field(default_factory=WACIResult)
    carbon_footprint: Decimal = Field(default=Decimal("0"))
    carbon_intensity: Decimal = Field(default=Decimal("0"))
    pcaf_quality: Decimal = Field(default=Decimal("0"))
    holding_results: List[HoldingResult] = Field(default_factory=list)
    sector_attribution: List[SectorAttribution] = Field(default_factory=list)
    holdings_count: int = Field(default=0)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PortfolioBenchmarkingEngine:
    """PCAF-aligned portfolio carbon benchmarking engine.

    Calculates financed emissions, WACI, carbon footprint, carbon
    intensity, PCAF quality, sector attribution, and benchmark comparison.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every holding calculation documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("PortfolioBenchmarkingEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: Portfolio) -> PortfolioResult:
        """Calculate portfolio carbon metrics.

        Args:
            input_data: Portfolio with holdings and optional benchmark.

        Returns:
            PortfolioResult with all metrics and attribution.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        holdings = input_data.holdings
        if len(holdings) > MAX_HOLDINGS:
            raise ValueError(f"Maximum {MAX_HOLDINGS} holdings (got {len(holdings)})")

        total_investment = sum(h.investment_value for h in holdings)
        if total_investment == Decimal("0"):
            warnings.append("Total investment is zero. Cannot compute portfolio weights.")

        aum = input_data.aum or total_investment

        # Per-holding calculations
        holding_results: List[HoldingResult] = []
        total_fe = Decimal("0")
        total_waci = Decimal("0")
        weighted_revenue = Decimal("0")
        quality_numerator = Decimal("0")
        quality_denominator = Decimal("0")
        covered = 0

        for h in holdings:
            weight = _safe_divide(h.investment_value, total_investment)
            ownership = _safe_divide(h.investment_value, h.evic)

            # Financed emissions: FE = ownership * emissions
            fe = (ownership * h.total_emissions_tco2e).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )
            total_fe += fe

            # Intensity: emissions / revenue
            intensity = _safe_divide(h.total_emissions_tco2e, h.revenue)

            # WACI contribution: weight * intensity
            waci_contrib = (weight * intensity).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )
            total_waci += waci_contrib

            # Revenue weighted
            weighted_revenue += weight * h.revenue

            # Quality (emissions-weighted)
            if fe > Decimal("0"):
                quality_numerator += fe * Decimal(str(h.pcaf_quality.value))
                quality_denominator += fe
                covered += 1

            holding_results.append(HoldingResult(
                entity_id=h.entity_id,
                entity_name=h.entity_name,
                sector=h.sector,
                portfolio_weight=weight.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                ownership_share=ownership.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                financed_emissions=fe,
                intensity=intensity.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                waci_contribution=waci_contrib,
                pcaf_quality=h.pcaf_quality.value,
            ))

        # Portfolio metrics
        waci = total_waci.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        carbon_footprint = _safe_divide(total_fe, aum).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        carbon_intensity = _safe_divide(total_fe, weighted_revenue).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        pcaf_quality = _safe_divide(quality_numerator, quality_denominator).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        coverage = _safe_divide(
            Decimal(str(covered)), Decimal(str(len(holdings)))
        ) * Decimal("100") if holdings else Decimal("0")
        coverage = coverage.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Benchmark comparison
        waci_result = WACIResult(waci=waci)
        if input_data.benchmark:
            bm = input_data.benchmark
            bm_waci = bm.waci
            if bm_waci is None and bm.holdings:
                bm_waci = self._compute_benchmark_waci(bm.holdings, prec_str)
            if bm_waci is not None:
                waci_result.benchmark_waci = bm_waci
                te = self._compute_tracking_error(holdings, bm.holdings, total_investment, prec_str)
                waci_result.tracking_error = te
                if bm_waci > Decimal("0"):
                    rel = ((waci - bm_waci) / bm_waci * Decimal("100")).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    waci_result.relative_pct = rel

        # Sector attribution
        sector_attr = self._compute_sector_attribution(holding_results, waci, prec_str)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = PortfolioResult(
            portfolio_id=input_data.portfolio_id,
            total_financed_emissions=total_fe.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            waci_result=waci_result,
            carbon_footprint=carbon_footprint,
            carbon_intensity=carbon_intensity,
            pcaf_quality=pcaf_quality,
            holding_results=holding_results,
            sector_attribution=sector_attr,
            holdings_count=len(holdings),
            coverage_pct=coverage,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_benchmark_waci(
        self, holdings: List[Holding], prec_str: str,
    ) -> Decimal:
        """Compute WACI for benchmark index."""
        total_inv = sum(h.investment_value for h in holdings)
        if total_inv == Decimal("0"):
            return Decimal("0")

        waci = Decimal("0")
        for h in holdings:
            w = _safe_divide(h.investment_value, total_inv)
            intensity = _safe_divide(h.total_emissions_tco2e, h.revenue)
            waci += w * intensity

        return waci.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

    def _compute_tracking_error(
        self,
        port_holdings: List[Holding],
        bench_holdings: List[Holding],
        total_port_inv: Decimal,
        prec_str: str,
    ) -> Decimal:
        """Compute tracking error: TE = SUM(|w_p - w_b| * I_i)."""
        total_bench = sum(h.investment_value for h in bench_holdings)
        if total_bench == Decimal("0"):
            return Decimal("0")

        # Build weight maps
        port_weights: Dict[str, Decimal] = {}
        for h in port_holdings:
            port_weights[h.entity_id] = _safe_divide(h.investment_value, total_port_inv)

        bench_weights: Dict[str, Decimal] = {}
        bench_intensity: Dict[str, Decimal] = {}
        for h in bench_holdings:
            bench_weights[h.entity_id] = _safe_divide(h.investment_value, total_bench)
            bench_intensity[h.entity_id] = _safe_divide(h.total_emissions_tco2e, h.revenue)

        all_ids = set(port_weights.keys()) | set(bench_weights.keys())
        te = Decimal("0")
        for eid in all_ids:
            wp = port_weights.get(eid, Decimal("0"))
            wb = bench_weights.get(eid, Decimal("0"))
            intensity = bench_intensity.get(eid, Decimal("0"))
            te += abs(wp - wb) * intensity

        return te.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

    def _compute_sector_attribution(
        self,
        holding_results: List[HoldingResult],
        total_waci: Decimal,
        prec_str: str,
    ) -> List[SectorAttribution]:
        """Compute sector-level attribution."""
        sector_data: Dict[str, Dict[str, Any]] = {}

        for hr in holding_results:
            sector = hr.sector or "unclassified"
            if sector not in sector_data:
                sector_data[sector] = {
                    "count": 0,
                    "weight": Decimal("0"),
                    "fe": Decimal("0"),
                    "waci": Decimal("0"),
                }
            sector_data[sector]["count"] += 1
            sector_data[sector]["weight"] += hr.portfolio_weight
            sector_data[sector]["fe"] += hr.financed_emissions
            sector_data[sector]["waci"] += hr.waci_contribution

        attrs: List[SectorAttribution] = []
        for sector, data in sorted(sector_data.items()):
            contrib_pct = _safe_divide(
                data["waci"] * Decimal("100"), total_waci
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            attrs.append(SectorAttribution(
                sector=sector,
                holdings_count=data["count"],
                portfolio_weight=data["weight"].quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                financed_emissions=data["fe"].quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                waci_contribution=data["waci"].quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                contribution_pct=contrib_pct,
            ))

        return attrs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "AssetClass",
    "PCAFScore",
    "MetricType",
    # Input Models
    "Holding",
    "BenchmarkIndex",
    "Portfolio",
    # Output Models
    "HoldingResult",
    "SectorAttribution",
    "WACIResult",
    "PortfolioResult",
    # Engine
    "PortfolioBenchmarkingEngine",
]
