"""
GAR Calculation Engine -- Green Asset Ratio & BTAR Computation

Implements Green Asset Ratio (GAR) calculation for stock and flow portfolios,
Banking Book Taxonomy Alignment Ratio (BTAR), exposure classification,
sector-level GAR breakdown, covered-asset computation with exclusions,
mortgage and auto-loan alignment assessment, EBA template generation,
GAR trend analysis, GAR-BTAR comparison, and asset-class summaries per
EBA Pillar 3 ESG ITS (EBA/ITS/2022/01).

GAR formula:
    Numerator   = Taxonomy-aligned on-balance-sheet covered assets
    Denominator = Total on-balance-sheet covered assets
                  (excluding sovereign, central bank, trading book)

BTAR extends GAR by including non-NFRD/non-CSRD exposures in both
numerator and denominator, providing a broader alignment picture.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852, Article 8
    - Delegated Regulation (EU) 2021/2178 (Article 8 Disclosures)
    - EBA Pillar 3 ESG ITS (EBA/ITS/2022/01) -- Templates 6-10
    - Commission FAQ on GAR calculation (June 2023)
    - ECB Guide on Climate-Related and Environmental Risks (2020)

Example:
    >>> from services.config import TaxonomyAppConfig
    >>> engine = GARCalculationEngine(TaxonomyAppConfig())
    >>> result = engine.calculate_gar_stock("bank-1", "2025", exposures)
    >>> print(result.gar_pct)
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import (
    EPCRating,
    EPC_RATING_SCORES,
    ExposureType,
    GAR_EXPOSURE_TYPES,
    REPORTING_TEMPLATES,
    ReportTemplate,
    TaxonomyAppConfig,
)
from .models import (
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class GARResult(BaseModel):
    """Base GAR calculation result."""

    institution_id: str = Field(...)
    period: str = Field(...)
    gar_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    numerator_eur: float = Field(default=0.0, ge=0.0)
    denominator_eur: float = Field(default=0.0, ge=0.0)
    total_assets_eur: float = Field(default=0.0, ge=0.0)
    excluded_eur: float = Field(default=0.0, ge=0.0)
    covered_assets_eur: float = Field(default=0.0, ge=0.0)
    aligned_assets_eur: float = Field(default=0.0, ge=0.0)
    eligible_assets_eur: float = Field(default=0.0, ge=0.0)
    exposure_count: int = Field(default=0)
    by_objective: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    calculated_at: datetime = Field(default_factory=_now)


class GARStockResult(GARResult):
    """GAR calculation result for stock (existing) portfolio."""

    calculation_type: str = Field(default="stock")
    weighted_average_maturity_years: Optional[float] = Field(None)


class GARFlowResult(GARResult):
    """GAR calculation result for flow (new originations) portfolio."""

    calculation_type: str = Field(default="flow")
    new_originations_count: int = Field(default=0)
    new_originations_eur: float = Field(default=0.0, ge=0.0)


class BTARResult(BaseModel):
    """Banking Book Taxonomy Alignment Ratio result."""

    institution_id: str = Field(...)
    period: str = Field(...)
    btar_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    numerator_eur: float = Field(default=0.0, ge=0.0)
    denominator_eur: float = Field(default=0.0, ge=0.0)
    nfrd_aligned_eur: float = Field(default=0.0, ge=0.0)
    non_nfrd_aligned_eur: float = Field(default=0.0, ge=0.0)
    total_nfrd_eur: float = Field(default=0.0, ge=0.0)
    total_non_nfrd_eur: float = Field(default=0.0, ge=0.0)
    gar_comparison_pct: float = Field(default=0.0)
    exposure_count: int = Field(default=0)
    provenance_hash: str = Field(default="")
    calculated_at: datetime = Field(default_factory=_now)


class ExposureClassification(BaseModel):
    """Classification result for a single exposure."""

    exposure_id: str = Field(...)
    exposure_type: str = Field(default="corporate_loan")
    is_covered: bool = Field(default=True)
    is_excluded: bool = Field(default=False)
    exclusion_reason: Optional[str] = Field(None)
    is_nfrd_scope: bool = Field(default=True)
    is_taxonomy_eligible: bool = Field(default=False)
    is_taxonomy_aligned: bool = Field(default=False)
    alignment_reason: str = Field(default="")


class SectorGARBreakdown(BaseModel):
    """GAR breakdown by NACE sector."""

    institution_id: str = Field(...)
    period: str = Field(...)
    sectors: List[Dict[str, Any]] = Field(default_factory=list)
    total_covered_eur: float = Field(default=0.0, ge=0.0)
    total_aligned_eur: float = Field(default=0.0, ge=0.0)
    sector_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


class EBATemplateResult(BaseModel):
    """Generated EBA Pillar 3 template data."""

    institution_id: str = Field(...)
    period: str = Field(...)
    template_number: int = Field(...)
    template_name: str = Field(default="")
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_now)


class AssetClassSummary(BaseModel):
    """GAR summary broken down by asset class."""

    institution_id: str = Field(...)
    period: str = Field(...)
    asset_classes: List[Dict[str, Any]] = Field(default_factory=list)
    total_covered_eur: float = Field(default=0.0, ge=0.0)
    total_aligned_eur: float = Field(default=0.0, ge=0.0)
    overall_gar_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Internal Exposure model (in-memory)
# ---------------------------------------------------------------------------

class _Exposure(BaseModel):
    """Internal exposure record for in-memory GAR calculation."""

    id: str = Field(default_factory=_new_id)
    institution_id: str = Field(...)
    period: str = Field(...)
    counterparty_name: Optional[str] = Field(None)
    exposure_type: str = Field(default=ExposureType.CORPORATE_LOAN.value)
    gross_carrying_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    is_on_balance_sheet: bool = Field(default=True)
    nace_sector: Optional[str] = Field(None)
    is_nfrd_scope: bool = Field(default=True)
    is_taxonomy_eligible: Optional[bool] = Field(None)
    is_taxonomy_aligned: Optional[bool] = Field(None)
    aligned_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    objective: Optional[str] = Field(None)
    epc_rating: Optional[str] = Field(None, description="For mortgages: A-G")
    co2_gkm: Optional[Decimal] = Field(None, description="For auto loans: gCO2/km")
    vehicle_type: Optional[str] = Field(None)
    country: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Exclusion categories (sovereign, central bank, trading book, interbank)
# ---------------------------------------------------------------------------

_GAR_EXCLUSION_CATEGORIES: Dict[str, str] = {
    "sovereign": "Sovereign exposures excluded per EBA ITS Article 7(1)",
    "central_bank": "Central bank exposures excluded per EBA ITS Article 7(1)",
    "trading_book": "Trading book exposures excluded per EBA ITS Article 7(2)",
    "interbank_on_demand": "Interbank on-demand deposits excluded per EBA ITS",
    "derivatives": "Derivatives excluded from GAR denominator",
}


# ---------------------------------------------------------------------------
# GARCalculationEngine
# ---------------------------------------------------------------------------

class GARCalculationEngine:
    """
    Green Asset Ratio (GAR) and BTAR calculation engine.

    Computes GAR for stock and flow portfolios per EBA Pillar 3 ESG ITS,
    classifies exposures, applies sovereign/central-bank/trading-book
    exclusions, assesses mortgage and auto-loan alignment, generates EBA
    templates 6-10, and provides trend analysis and comparisons.

    Attributes:
        config: Application configuration.
        _exposures: In-memory exposures keyed by institution_id.
        _gar_results: Cached GAR results keyed by (institution_id, period).

    Example:
        >>> engine = GARCalculationEngine(TaxonomyAppConfig())
        >>> stock = engine.calculate_gar_stock("bank-1", "2025", exposures)
        >>> print(stock.gar_pct)
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """Initialize the GARCalculationEngine."""
        self.config = config or TaxonomyAppConfig()
        self._exposures: Dict[str, List[_Exposure]] = {}
        self._gar_results: Dict[str, GARResult] = {}
        logger.info("GARCalculationEngine initialized")

    # ------------------------------------------------------------------
    # Exposure Registration
    # ------------------------------------------------------------------

    def register_exposure(self, exposure: _Exposure) -> None:
        """
        Register a financial exposure for GAR computation.

        Args:
            exposure: Internal Exposure model instance.
        """
        key = exposure.institution_id
        self._exposures.setdefault(key, []).append(exposure)

    def register_exposures(self, exposures: List[_Exposure]) -> None:
        """
        Register multiple exposures at once.

        Args:
            exposures: List of _Exposure model instances.
        """
        for exp in exposures:
            self.register_exposure(exp)

    # ------------------------------------------------------------------
    # GAR Stock
    # ------------------------------------------------------------------

    def calculate_gar_stock(
        self,
        institution_id: str,
        period: str,
        exposures: Optional[List[Dict[str, Any]]] = None,
    ) -> GARStockResult:
        """
        Calculate GAR for existing (stock) on-balance-sheet portfolio.

        The GAR stock measures the proportion of taxonomy-aligned assets
        in the institution's existing covered portfolio at the reporting
        date.

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period (e.g. "2025").
            exposures: Optional list of exposure dicts to register on-the-fly.

        Returns:
            GARStockResult with numerator, denominator, and GAR percentage.
        """
        start = datetime.utcnow()

        if exposures:
            for exp_data in exposures:
                exp = _Exposure(institution_id=institution_id, period=period, **exp_data)
                self.register_exposure(exp)

        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]

        total_assets = sum(float(e.gross_carrying_amount_eur) for e in all_exposures)

        # Apply exclusions
        covered, excluded_amt = self._apply_exclusions(all_exposures)
        denominator = sum(float(e.gross_carrying_amount_eur) for e in covered)

        # Numerator = aligned assets among covered
        aligned = [e for e in covered if e.is_taxonomy_aligned]
        numerator = sum(float(e.aligned_amount_eur) for e in aligned)

        # Eligible (broader than aligned)
        eligible = [e for e in covered if e.is_taxonomy_eligible]
        eligible_amt = sum(float(e.gross_carrying_amount_eur) for e in eligible)

        gar_pct = (numerator / denominator * 100.0) if denominator > 0 else 0.0

        # By objective breakdown
        by_objective = self._compute_objective_breakdown(aligned)

        provenance = _sha256(
            f"gar_stock:{institution_id}:{period}:{numerator}:{denominator}"
        )

        result = GARStockResult(
            institution_id=institution_id,
            period=period,
            gar_pct=round(gar_pct, 4),
            numerator_eur=round(numerator, 2),
            denominator_eur=round(denominator, 2),
            total_assets_eur=round(total_assets, 2),
            excluded_eur=round(excluded_amt, 2),
            covered_assets_eur=round(denominator, 2),
            aligned_assets_eur=round(numerator, 2),
            eligible_assets_eur=round(eligible_amt, 2),
            exposure_count=len(all_exposures),
            by_objective=by_objective,
            provenance_hash=provenance,
        )

        cache_key = f"{institution_id}:{period}:stock"
        self._gar_results[cache_key] = result

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "GAR stock for %s period %s: %.4f%% (%d exposures) in %.1f ms",
            institution_id, period, gar_pct, len(all_exposures), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # GAR Flow
    # ------------------------------------------------------------------

    def calculate_gar_flow(
        self,
        institution_id: str,
        period: str,
        new_originations: Optional[List[Dict[str, Any]]] = None,
    ) -> GARFlowResult:
        """
        Calculate GAR for new originations (flow) during the reporting period.

        The GAR flow measures the proportion of taxonomy-aligned new lending
        and investment originations during the period, reflecting the
        institution's transition trajectory.

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period.
            new_originations: Optional list of new exposure dicts.

        Returns:
            GARFlowResult with flow-specific metrics.
        """
        start = datetime.utcnow()

        if new_originations:
            for orig in new_originations:
                exp = _Exposure(
                    institution_id=institution_id,
                    period=period,
                    **orig,
                )
                self.register_exposure(exp)

        all_flow = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]

        covered, excluded_amt = self._apply_exclusions(all_flow)
        denominator = sum(float(e.gross_carrying_amount_eur) for e in covered)

        aligned = [e for e in covered if e.is_taxonomy_aligned]
        numerator = sum(float(e.aligned_amount_eur) for e in aligned)

        eligible = [e for e in covered if e.is_taxonomy_eligible]
        eligible_amt = sum(float(e.gross_carrying_amount_eur) for e in eligible)

        gar_pct = (numerator / denominator * 100.0) if denominator > 0 else 0.0
        by_objective = self._compute_objective_breakdown(aligned)

        total_assets = sum(float(e.gross_carrying_amount_eur) for e in all_flow)

        provenance = _sha256(
            f"gar_flow:{institution_id}:{period}:{numerator}:{denominator}"
        )

        result = GARFlowResult(
            institution_id=institution_id,
            period=period,
            gar_pct=round(gar_pct, 4),
            numerator_eur=round(numerator, 2),
            denominator_eur=round(denominator, 2),
            total_assets_eur=round(total_assets, 2),
            excluded_eur=round(excluded_amt, 2),
            covered_assets_eur=round(denominator, 2),
            aligned_assets_eur=round(numerator, 2),
            eligible_assets_eur=round(eligible_amt, 2),
            exposure_count=len(all_flow),
            by_objective=by_objective,
            new_originations_count=len(all_flow),
            new_originations_eur=round(total_assets, 2),
            provenance_hash=provenance,
        )

        cache_key = f"{institution_id}:{period}:flow"
        self._gar_results[cache_key] = result

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "GAR flow for %s period %s: %.4f%% (%d originations) in %.1f ms",
            institution_id, period, gar_pct, len(all_flow), elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # BTAR
    # ------------------------------------------------------------------

    def calculate_btar(
        self,
        institution_id: str,
        period: str,
        extended_exposures: Optional[List[Dict[str, Any]]] = None,
    ) -> BTARResult:
        """
        Calculate the Banking Book Taxonomy Alignment Ratio (BTAR).

        BTAR extends GAR by including non-NFRD/non-CSRD counterparty
        exposures in both numerator and denominator. This is a voluntary
        EBA Template 9 disclosure providing a broader alignment view.

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period.
            extended_exposures: Optional list of exposure dicts including
                non-NFRD exposures.

        Returns:
            BTARResult with extended alignment metrics.
        """
        start = datetime.utcnow()

        if extended_exposures:
            for exp_data in extended_exposures:
                exp = _Exposure(
                    institution_id=institution_id,
                    period=period,
                    **exp_data,
                )
                self.register_exposure(exp)

        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]

        covered, _ = self._apply_exclusions(all_exposures)

        # Split NFRD vs non-NFRD
        nfrd = [e for e in covered if e.is_nfrd_scope]
        non_nfrd = [e for e in covered if not e.is_nfrd_scope]

        nfrd_total = sum(float(e.gross_carrying_amount_eur) for e in nfrd)
        non_nfrd_total = sum(float(e.gross_carrying_amount_eur) for e in non_nfrd)

        nfrd_aligned = sum(
            float(e.aligned_amount_eur) for e in nfrd if e.is_taxonomy_aligned
        )
        non_nfrd_aligned = sum(
            float(e.aligned_amount_eur) for e in non_nfrd if e.is_taxonomy_aligned
        )

        numerator = nfrd_aligned + non_nfrd_aligned
        denominator = nfrd_total + non_nfrd_total
        btar_pct = (numerator / denominator * 100.0) if denominator > 0 else 0.0

        gar_denom = nfrd_total
        gar_pct = (nfrd_aligned / gar_denom * 100.0) if gar_denom > 0 else 0.0

        provenance = _sha256(
            f"btar:{institution_id}:{period}:{numerator}:{denominator}"
        )

        result = BTARResult(
            institution_id=institution_id,
            period=period,
            btar_pct=round(btar_pct, 4),
            numerator_eur=round(numerator, 2),
            denominator_eur=round(denominator, 2),
            nfrd_aligned_eur=round(nfrd_aligned, 2),
            non_nfrd_aligned_eur=round(non_nfrd_aligned, 2),
            total_nfrd_eur=round(nfrd_total, 2),
            total_non_nfrd_eur=round(non_nfrd_total, 2),
            gar_comparison_pct=round(gar_pct, 4),
            exposure_count=len(covered),
            provenance_hash=provenance,
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "BTAR for %s period %s: %.4f%% (GAR=%.4f%%) in %.1f ms",
            institution_id, period, btar_pct, gar_pct, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Exposure Classification
    # ------------------------------------------------------------------

    def classify_exposure(
        self, exposure_data: Dict[str, Any],
    ) -> ExposureClassification:
        """
        Classify a single exposure by type and taxonomy alignment.

        Determines whether the exposure is covered (not excluded), its
        type (corporate/retail/mortgage/auto/project), and whether it
        qualifies as taxonomy-eligible or aligned.

        Args:
            exposure_data: Dict with exposure attributes.

        Returns:
            ExposureClassification with type and alignment determination.
        """
        exp_type = exposure_data.get("exposure_type", ExposureType.CORPORATE_LOAN.value)
        exp_id = exposure_data.get("id", _new_id())
        is_nfrd = exposure_data.get("is_nfrd_scope", True)
        is_on_bs = exposure_data.get("is_on_balance_sheet", True)

        # Check exclusions based on config flags
        is_excluded = False
        exclusion_reason = None
        if self.config.exclude_sovereign_exposures and exp_type == "sovereign":
            is_excluded = True
            exclusion_reason = _GAR_EXCLUSION_CATEGORIES["sovereign"]
        if self.config.exclude_trading_book and exp_type == "trading_book":
            is_excluded = True
            exclusion_reason = _GAR_EXCLUSION_CATEGORIES["trading_book"]

        is_covered = is_on_bs and not is_excluded

        # Determine alignment based on exposure type
        is_eligible = exposure_data.get("is_taxonomy_eligible", False)
        is_aligned = exposure_data.get("is_taxonomy_aligned", False)
        alignment_reason = ""

        # Mortgage alignment via EPC
        if exp_type == ExposureType.RETAIL_MORTGAGE.value:
            epc = exposure_data.get("epc_rating", "")
            country = exposure_data.get("country", "EU")
            if epc:
                is_aligned = self.assess_mortgage_alignment(epc, country)
                is_eligible = True
                alignment_reason = (
                    f"EPC rating {epc}: {'aligned' if is_aligned else 'not aligned'}"
                )

        # Auto loan alignment via CO2
        if exp_type == ExposureType.AUTO_LOAN.value:
            co2 = exposure_data.get("co2_gkm")
            vtype = exposure_data.get("vehicle_type", "car")
            if co2 is not None:
                is_aligned = self.assess_auto_loan_alignment(
                    Decimal(str(co2)), vtype,
                )
                is_eligible = True
                alignment_reason = (
                    f"CO2 {co2} g/km, type={vtype}: "
                    f"{'aligned' if is_aligned else 'not aligned'}"
                )

        return ExposureClassification(
            exposure_id=exp_id,
            exposure_type=exp_type,
            is_covered=is_covered,
            is_excluded=is_excluded,
            exclusion_reason=exclusion_reason,
            is_nfrd_scope=is_nfrd,
            is_taxonomy_eligible=is_eligible,
            is_taxonomy_aligned=is_aligned,
            alignment_reason=alignment_reason,
        )

    # ------------------------------------------------------------------
    # Sector GAR Breakdown
    # ------------------------------------------------------------------

    def get_sector_gar_breakdown(
        self, institution_id: str, period: str,
    ) -> SectorGARBreakdown:
        """
        Break down GAR by NACE sector for detailed analysis.

        Groups covered exposures by their NACE sector code and computes
        the aligned amount and sector-level GAR for each group.

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period.

        Returns:
            SectorGARBreakdown with per-sector alignment data.
        """
        start = datetime.utcnow()

        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]
        covered, _ = self._apply_exclusions(all_exposures)

        sector_map: Dict[str, Dict[str, float]] = {}
        for exp in covered:
            sector = exp.nace_sector or "UNKNOWN"
            if sector not in sector_map:
                sector_map[sector] = {
                    "total_eur": 0.0,
                    "aligned_eur": 0.0,
                    "count": 0,
                }
            sector_map[sector]["total_eur"] += float(exp.gross_carrying_amount_eur)
            sector_map[sector]["count"] += 1
            if exp.is_taxonomy_aligned:
                sector_map[sector]["aligned_eur"] += float(exp.aligned_amount_eur)

        sectors: List[Dict[str, Any]] = []
        total_covered = 0.0
        total_aligned = 0.0

        for sector_code, data in sorted(sector_map.items()):
            sector_gar = (
                data["aligned_eur"] / data["total_eur"] * 100.0
                if data["total_eur"] > 0 else 0.0
            )
            sectors.append({
                "nace_sector": sector_code,
                "total_eur": round(data["total_eur"], 2),
                "aligned_eur": round(data["aligned_eur"], 2),
                "gar_pct": round(sector_gar, 4),
                "exposure_count": int(data["count"]),
            })
            total_covered += data["total_eur"]
            total_aligned += data["aligned_eur"]

        provenance = _sha256(
            f"sector_gar:{institution_id}:{period}:{len(sectors)}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Sector GAR for %s period %s: %d sectors in %.1f ms",
            institution_id, period, len(sectors), elapsed,
        )

        return SectorGARBreakdown(
            institution_id=institution_id,
            period=period,
            sectors=sectors,
            total_covered_eur=round(total_covered, 2),
            total_aligned_eur=round(total_aligned, 2),
            sector_count=len(sectors),
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Covered Assets Calculation
    # ------------------------------------------------------------------

    def calculate_covered_assets(
        self,
        institution_id: str,
        total_assets: float,
        exclusions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate covered assets by applying standard exclusions.

        Subtracts sovereign, central bank, and trading book exposures
        from total assets to derive the GAR denominator.

        Args:
            institution_id: Financial institution identifier.
            total_assets: Total on-balance-sheet assets in EUR.
            exclusions: Dict mapping exclusion category to EUR amount.

        Returns:
            Dict with total, excluded, and covered asset values.
        """
        exclusions = exclusions or {}

        sovereign_excl = exclusions.get("sovereign", 0.0)
        central_bank_excl = exclusions.get("central_bank", 0.0)
        trading_book_excl = exclusions.get("trading_book", 0.0)
        interbank_excl = exclusions.get("interbank_on_demand", 0.0)
        derivatives_excl = exclusions.get("derivatives", 0.0)

        total_excluded = (
            sovereign_excl + central_bank_excl + trading_book_excl
            + interbank_excl + derivatives_excl
        )
        covered = max(total_assets - total_excluded, 0.0)

        result = {
            "institution_id": institution_id,
            "total_assets_eur": round(total_assets, 2),
            "exclusions": {
                "sovereign": round(sovereign_excl, 2),
                "central_bank": round(central_bank_excl, 2),
                "trading_book": round(trading_book_excl, 2),
                "interbank_on_demand": round(interbank_excl, 2),
                "derivatives": round(derivatives_excl, 2),
            },
            "total_excluded_eur": round(total_excluded, 2),
            "covered_assets_eur": round(covered, 2),
            "covered_pct": round(
                (covered / total_assets * 100.0) if total_assets > 0 else 0.0, 2,
            ),
        }

        logger.info(
            "Covered assets for %s: %.0f total, %.0f excluded, %.0f covered",
            institution_id, total_assets, total_excluded, covered,
        )
        return result

    # ------------------------------------------------------------------
    # Mortgage Alignment
    # ------------------------------------------------------------------

    def assess_mortgage_alignment(
        self, epc_rating: str, country: str = "EU",
    ) -> bool:
        """
        Assess taxonomy alignment for a mortgage exposure based on EPC.

        EPC class A (or top 15% of national stock) qualifies as
        taxonomy-aligned per Activity 7.7 TSC.

        Args:
            epc_rating: Energy Performance Certificate rating (A-G).
            country: ISO country code (used for national top-15% reference).

        Returns:
            True if the mortgage is taxonomy-aligned.
        """
        rating = epc_rating.upper().strip()
        threshold = self.config.epc_alignment_threshold

        # Compare EPC rating scores: higher score = better rating
        rating_score = EPC_RATING_SCORES.get(EPCRating(rating), 0)
        threshold_score = EPC_RATING_SCORES.get(threshold, 7)

        is_aligned = rating_score >= threshold_score
        logger.debug(
            "Mortgage EPC %s in %s: aligned=%s (threshold=%s)",
            rating, country, is_aligned, threshold.value,
        )
        return is_aligned

    # ------------------------------------------------------------------
    # Auto Loan Alignment
    # ------------------------------------------------------------------

    def assess_auto_loan_alignment(
        self, co2_gkm: Decimal, vehicle_type: str = "car",
    ) -> bool:
        """
        Assess taxonomy alignment for an auto loan exposure.

        Zero direct (tailpipe) CO2 emissions vehicles are aligned.
        Per Activity 6.5 TSC, only zero-emission vehicles qualify.

        Args:
            co2_gkm: CO2 emissions in grams per kilometre.
            vehicle_type: Vehicle type (car, van, truck).

        Returns:
            True if the auto loan is taxonomy-aligned.
        """
        # Zero emission is always aligned per Activity 6.5 TSC
        if co2_gkm <= Decimal("0"):
            return True

        return False

    # ------------------------------------------------------------------
    # EBA Template Generation
    # ------------------------------------------------------------------

    def generate_eba_template(
        self,
        institution_id: str,
        period: str,
        template_number: int,
    ) -> EBATemplateResult:
        """
        Generate EBA Pillar 3 disclosure template data.

        Supports Templates 6 (GAR summary), 7 (sector information),
        8 (household exposures), 9 (BTAR - voluntary), and 10 (other
        climate actions).

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period.
            template_number: Template number (6-10).

        Returns:
            EBATemplateResult with template rows and summary.
        """
        start = datetime.utcnow()

        # Resolve template name from config
        template_key_map = {
            6: ReportTemplate.EBA_TEMPLATE_6,
            7: ReportTemplate.EBA_TEMPLATE_7,
            8: ReportTemplate.EBA_TEMPLATE_8,
            9: ReportTemplate.EBA_TEMPLATE_9,
            10: ReportTemplate.EBA_TEMPLATE_10,
        }
        tmpl_enum = template_key_map.get(template_number)
        tmpl_def = REPORTING_TEMPLATES.get(tmpl_enum, {}) if tmpl_enum else {}
        template_name = tmpl_def.get("name", f"Template {template_number}")

        rows: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}

        if template_number == 6:
            rows, summary = self._generate_template_6(institution_id, period)
        elif template_number == 7:
            rows, summary = self._generate_template_7(institution_id, period)
        elif template_number == 8:
            rows, summary = self._generate_template_8(institution_id, period)
        elif template_number == 9:
            rows, summary = self._generate_template_9(institution_id, period)
        elif template_number == 10:
            rows, summary = self._generate_template_10(institution_id, period)
        else:
            logger.warning("Unknown template number: %d", template_number)

        provenance = _sha256(
            f"eba_template:{institution_id}:{period}:{template_number}:{len(rows)}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "EBA template %d for %s period %s: %d rows in %.1f ms",
            template_number, institution_id, period, len(rows), elapsed,
        )

        return EBATemplateResult(
            institution_id=institution_id,
            period=period,
            template_number=template_number,
            template_name=template_name,
            rows=rows,
            summary=summary,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # GAR Trends
    # ------------------------------------------------------------------

    def get_gar_trends(
        self, institution_id: str, periods: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve GAR trend data across multiple reporting periods.

        Args:
            institution_id: Financial institution identifier.
            periods: List of reporting period strings to compare.

        Returns:
            List of dicts with period, GAR %, and delta from prior period.
        """
        trends: List[Dict[str, Any]] = []
        prev_gar: Optional[float] = None

        for period in sorted(periods):
            cache_key = f"{institution_id}:{period}:stock"
            result = self._gar_results.get(cache_key)

            if result:
                gar = result.gar_pct
            else:
                stock = self.calculate_gar_stock(institution_id, period)
                gar = stock.gar_pct

            delta = (gar - prev_gar) if prev_gar is not None else 0.0
            trends.append({
                "period": period,
                "gar_pct": round(gar, 4),
                "delta_pct": round(delta, 4),
                "direction": (
                    "improving" if delta > 0
                    else "declining" if delta < 0
                    else "stable"
                ),
            })
            prev_gar = gar

        logger.info(
            "GAR trends for %s: %d periods", institution_id, len(trends),
        )
        return trends

    # ------------------------------------------------------------------
    # GAR vs BTAR Comparison
    # ------------------------------------------------------------------

    def compare_gar_btar(
        self, institution_id: str, period: str,
    ) -> Dict[str, Any]:
        """
        Compare GAR and BTAR for the same institution and period.

        Highlights the difference in alignment ratios when non-NFRD
        exposures are included (BTAR) versus excluded (GAR).

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period.

        Returns:
            Dict with GAR, BTAR, difference, and interpretation.
        """
        stock_key = f"{institution_id}:{period}:stock"
        gar_result = self._gar_results.get(stock_key)
        gar_pct = gar_result.gar_pct if gar_result else 0.0

        btar = self.calculate_btar(institution_id, period)
        btar_pct = btar.btar_pct

        difference = btar_pct - gar_pct

        if abs(difference) < 0.5:
            interpretation = (
                "GAR and BTAR are closely aligned, indicating non-NFRD exposures "
                "have similar alignment characteristics to NFRD-scope exposures."
            )
        elif difference > 0:
            interpretation = (
                f"BTAR is {difference:.2f}pp higher than GAR, suggesting non-NFRD "
                f"counterparties have stronger taxonomy alignment than NFRD-scope "
                f"counterparties."
            )
        else:
            interpretation = (
                f"BTAR is {abs(difference):.2f}pp lower than GAR, indicating "
                f"non-NFRD counterparties are less taxonomy-aligned than "
                f"NFRD-scope counterparties."
            )

        return {
            "institution_id": institution_id,
            "period": period,
            "gar_pct": round(gar_pct, 4),
            "btar_pct": round(btar_pct, 4),
            "difference_pp": round(difference, 4),
            "interpretation": interpretation,
            "gar_denominator_eur": (
                gar_result.denominator_eur if gar_result else 0.0
            ),
            "btar_denominator_eur": btar.denominator_eur,
        }

    # ------------------------------------------------------------------
    # Asset Class Summary
    # ------------------------------------------------------------------

    def get_asset_class_summary(
        self, institution_id: str, period: str,
    ) -> AssetClassSummary:
        """
        Summarize GAR metrics by asset class.

        Groups covered exposures by their exposure_type and computes
        per-class totals, aligned amounts, and class-level GAR.

        Args:
            institution_id: Financial institution identifier.
            period: Reporting period.

        Returns:
            AssetClassSummary with per-class breakdown.
        """
        start = datetime.utcnow()

        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]
        covered, _ = self._apply_exclusions(all_exposures)

        class_map: Dict[str, Dict[str, float]] = {}
        for exp in covered:
            cls = exp.exposure_type
            if cls not in class_map:
                class_map[cls] = {"total_eur": 0.0, "aligned_eur": 0.0, "count": 0}
            class_map[cls]["total_eur"] += float(exp.gross_carrying_amount_eur)
            class_map[cls]["count"] += 1
            if exp.is_taxonomy_aligned:
                class_map[cls]["aligned_eur"] += float(exp.aligned_amount_eur)

        asset_classes: List[Dict[str, Any]] = []
        total_covered = 0.0
        total_aligned = 0.0

        for cls_name, data in sorted(class_map.items()):
            cls_gar = (
                data["aligned_eur"] / data["total_eur"] * 100.0
                if data["total_eur"] > 0 else 0.0
            )
            asset_classes.append({
                "asset_class": cls_name,
                "total_eur": round(data["total_eur"], 2),
                "aligned_eur": round(data["aligned_eur"], 2),
                "gar_pct": round(cls_gar, 4),
                "exposure_count": int(data["count"]),
            })
            total_covered += data["total_eur"]
            total_aligned += data["aligned_eur"]

        overall = (
            total_aligned / total_covered * 100.0 if total_covered > 0 else 0.0
        )

        provenance = _sha256(
            f"asset_class:{institution_id}:{period}:{len(asset_classes)}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Asset class summary for %s period %s: %d classes in %.1f ms",
            institution_id, period, len(asset_classes), elapsed,
        )

        return AssetClassSummary(
            institution_id=institution_id,
            period=period,
            asset_classes=asset_classes,
            total_covered_eur=round(total_covered, 2),
            total_aligned_eur=round(total_aligned, 2),
            overall_gar_pct=round(overall, 4),
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_exclusions(
        self, exposures: List[_Exposure],
    ) -> Tuple[List[_Exposure], float]:
        """
        Apply GAR exclusions to a set of exposures.

        Removes sovereign and trading book exposures (per config flags)
        from the covered set and returns the excluded amount.

        Args:
            exposures: Full list of on-balance-sheet exposures.

        Returns:
            Tuple of (covered exposures, total excluded EUR amount).
        """
        covered: List[_Exposure] = []
        excluded_amt = 0.0

        # Build set of excluded types based on config
        excluded_types: set = set()
        if self.config.exclude_sovereign_exposures:
            excluded_types.add("sovereign")
            excluded_types.add("central_bank")
        if self.config.exclude_trading_book:
            excluded_types.add("trading_book")

        for exp in exposures:
            if exp.exposure_type in excluded_types:
                excluded_amt += float(exp.gross_carrying_amount_eur)
            else:
                covered.append(exp)

        return covered, excluded_amt

    def _compute_objective_breakdown(
        self, aligned_exposures: List[_Exposure],
    ) -> Dict[str, float]:
        """
        Break down aligned amounts by environmental objective.

        Args:
            aligned_exposures: List of taxonomy-aligned exposures.

        Returns:
            Dict mapping objective name to aligned EUR total.
        """
        breakdown: Dict[str, float] = {}
        for exp in aligned_exposures:
            obj = exp.objective or "unspecified"
            breakdown[obj] = breakdown.get(obj, 0.0) + float(exp.aligned_amount_eur)

        return {k: round(v, 2) for k, v in sorted(breakdown.items())}

    def _generate_template_6(
        self, institution_id: str, period: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate Template 6: Summary of GAR KPIs."""
        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]
        covered, excluded_amt = self._apply_exclusions(all_exposures)
        total_covered = sum(float(e.gross_carrying_amount_eur) for e in covered)

        aligned_mit = sum(
            float(e.aligned_amount_eur) for e in covered
            if e.is_taxonomy_aligned and e.objective == "climate_mitigation"
        )
        aligned_adp = sum(
            float(e.aligned_amount_eur) for e in covered
            if e.is_taxonomy_aligned and e.objective == "climate_adaptation"
        )
        total_aligned = sum(
            float(e.aligned_amount_eur) for e in covered
            if e.is_taxonomy_aligned
        )

        rows = [
            {
                "row_id": 1,
                "description": "Total covered assets (denominator)",
                "amount_eur": round(total_covered, 2),
            },
            {
                "row_id": 2,
                "description": "Climate change mitigation - aligned",
                "amount_eur": round(aligned_mit, 2),
                "pct_of_covered": round(
                    aligned_mit / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
                ),
            },
            {
                "row_id": 3,
                "description": "Climate change adaptation - aligned",
                "amount_eur": round(aligned_adp, 2),
                "pct_of_covered": round(
                    aligned_adp / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
                ),
            },
            {
                "row_id": 4,
                "description": "Total taxonomy-aligned (numerator)",
                "amount_eur": round(total_aligned, 2),
                "pct_of_covered": round(
                    total_aligned / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
                ),
            },
        ]

        summary = {
            "total_covered_eur": round(total_covered, 2),
            "total_aligned_eur": round(total_aligned, 2),
            "gar_pct": round(
                total_aligned / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
            ),
        }

        return rows, summary

    def _generate_template_7(
        self, institution_id: str, period: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate Template 7: GAR Sector Information."""
        breakdown = self.get_sector_gar_breakdown(institution_id, period)

        rows = [
            {
                "row_id": idx + 1,
                "nace_sector": s["nace_sector"],
                "total_eur": s["total_eur"],
                "aligned_eur": s["aligned_eur"],
                "gar_pct": s["gar_pct"],
                "exposure_count": s["exposure_count"],
            }
            for idx, s in enumerate(breakdown.sectors)
        ]

        summary = {
            "total_covered_eur": breakdown.total_covered_eur,
            "total_aligned_eur": breakdown.total_aligned_eur,
            "sector_count": breakdown.sector_count,
        }

        return rows, summary

    def _generate_template_8(
        self, institution_id: str, period: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate Template 8: GAR Household Exposures."""
        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]
        total_assets = sum(float(e.gross_carrying_amount_eur) for e in all_exposures)
        covered, excluded_amt = self._apply_exclusions(all_exposures)
        total_covered = sum(float(e.gross_carrying_amount_eur) for e in covered)
        total_aligned = sum(
            float(e.aligned_amount_eur) for e in covered if e.is_taxonomy_aligned
        )
        total_eligible = sum(
            float(e.gross_carrying_amount_eur) for e in covered
            if e.is_taxonomy_eligible
        )

        rows = [
            {
                "row_id": 1,
                "description": "Total on-balance-sheet assets",
                "amount_eur": round(total_assets, 2),
            },
            {
                "row_id": 2,
                "description": "Excluded assets",
                "amount_eur": round(excluded_amt, 2),
            },
            {
                "row_id": 3,
                "description": "Covered assets (denominator)",
                "amount_eur": round(total_covered, 2),
            },
            {
                "row_id": 4,
                "description": "Taxonomy-eligible assets",
                "amount_eur": round(total_eligible, 2),
                "pct_of_covered": round(
                    total_eligible / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
                ),
            },
            {
                "row_id": 5,
                "description": "Taxonomy-aligned assets (numerator)",
                "amount_eur": round(total_aligned, 2),
                "pct_of_covered": round(
                    total_aligned / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
                ),
            },
        ]

        summary = {
            "total_assets_eur": round(total_assets, 2),
            "excluded_eur": round(excluded_amt, 2),
            "covered_eur": round(total_covered, 2),
            "eligible_eur": round(total_eligible, 2),
            "aligned_eur": round(total_aligned, 2),
            "gar_pct": round(
                total_aligned / total_covered * 100.0 if total_covered > 0 else 0.0, 4,
            ),
        }

        return rows, summary

    def _generate_template_9(
        self, institution_id: str, period: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate Template 9: BTAR (voluntary)."""
        btar = self.calculate_btar(institution_id, period)

        rows = [
            {
                "row_id": 1,
                "description": "NFRD-scope exposures",
                "amount_eur": btar.total_nfrd_eur,
                "aligned_eur": btar.nfrd_aligned_eur,
            },
            {
                "row_id": 2,
                "description": "Non-NFRD exposures",
                "amount_eur": btar.total_non_nfrd_eur,
                "aligned_eur": btar.non_nfrd_aligned_eur,
            },
            {
                "row_id": 3,
                "description": "Total (BTAR denominator)",
                "amount_eur": btar.denominator_eur,
                "aligned_eur": btar.numerator_eur,
            },
            {
                "row_id": 4,
                "description": "BTAR",
                "btar_pct": btar.btar_pct,
            },
        ]

        summary = {
            "btar_pct": btar.btar_pct,
            "gar_comparison_pct": btar.gar_comparison_pct,
            "difference_pp": round(btar.btar_pct - btar.gar_comparison_pct, 4),
        }

        return rows, summary

    def _generate_template_10(
        self, institution_id: str, period: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate Template 10: Other climate change mitigating actions."""
        all_exposures = [
            e for e in self._exposures.get(institution_id, [])
            if e.period == period and e.is_on_balance_sheet
        ]
        covered, _ = self._apply_exclusions(all_exposures)

        eligible_not_aligned = [
            e for e in covered
            if e.is_taxonomy_eligible and not e.is_taxonomy_aligned
        ]
        total_ena = sum(
            float(e.gross_carrying_amount_eur) for e in eligible_not_aligned
        )

        rows = [
            {
                "row_id": 1,
                "description": "Taxonomy-eligible but not aligned exposures",
                "amount_eur": round(total_ena, 2),
                "count": len(eligible_not_aligned),
            },
        ]

        summary = {
            "eligible_not_aligned_eur": round(total_ena, 2),
            "count": len(eligible_not_aligned),
        }

        return rows, summary
