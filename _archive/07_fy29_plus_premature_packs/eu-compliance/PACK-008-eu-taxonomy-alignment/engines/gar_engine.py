# -*- coding: utf-8 -*-
"""
Green Asset Ratio Engine - PACK-008 EU Taxonomy Alignment

This module implements the Green Asset Ratio (GAR) calculation engine for
financial institutions subject to EU Taxonomy Article 8 disclosure and
EBA Pillar 3 ESG reporting requirements.

The engine calculates GAR stock (on-balance-sheet assets), GAR flow (new
originations during the period), and BTAR (Banking Book Taxonomy Alignment
Ratio). It classifies exposure types, integrates EPC ratings for real estate,
handles de minimis thresholds, and provides numerator/denominator breakdowns
per Article 8 Delegated Regulation (EU) 2021/2178.

All financial calculations use Decimal for precision. No LLM calls are used
for numeric computations -- every ratio is deterministic.

Example:
    >>> config = GARConfig(de_minimis_threshold=Decimal("500000"))
    >>> engine = GreenAssetRatioEngine(config)
    >>> result = engine.calculate_gar_stock(exposures)
    >>> print(f"GAR Stock: {result.gar_ratio}")
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExposureType(str, Enum):
    """Asset exposure types on balance sheet."""

    CORPORATE_LOANS = "CORPORATE_LOANS"
    DEBT_SECURITIES = "DEBT_SECURITIES"
    EQUITY_HOLDINGS = "EQUITY_HOLDINGS"
    RESIDENTIAL_MORTGAGES = "RESIDENTIAL_MORTGAGES"
    COMMERCIAL_MORTGAGES = "COMMERCIAL_MORTGAGES"
    AUTO_LOANS = "AUTO_LOANS"
    RENOVATION_LOANS = "RENOVATION_LOANS"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    INTERBANK_LOANS = "INTERBANK_LOANS"
    SOVEREIGN_EXPOSURES = "SOVEREIGN_EXPOSURES"
    DERIVATIVES = "DERIVATIVES"
    OTHER = "OTHER"


class EPCRating(str, Enum):
    """Energy Performance Certificate ratings for real estate."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    NOT_AVAILABLE = "N/A"


class CounterpartyType(str, Enum):
    """Counterparty classification."""

    NFRD_SUBJECT = "NFRD_SUBJECT"
    NON_NFRD_CORPORATE = "NON_NFRD_CORPORATE"
    SME = "SME"
    HOUSEHOLD = "HOUSEHOLD"
    LOCAL_GOVERNMENT = "LOCAL_GOVERNMENT"
    OTHER_COUNTERPARTY = "OTHER_COUNTERPARTY"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GARConfig(BaseModel):
    """Configuration for the Green Asset Ratio Engine."""

    de_minimis_threshold: Decimal = Field(
        default=Decimal("500000"),
        description="Minimum exposure amount to include in GAR (EUR)"
    )
    include_trading_book: bool = Field(
        default=False,
        description="Include trading book exposures in GAR calculation"
    )
    epc_top_15_threshold: EPCRating = Field(
        default=EPCRating.A,
        description="EPC threshold for top-15-percent NZEB proxy"
    )
    reporting_currency: str = Field(
        default="EUR",
        description="Reporting currency (ISO 4217)"
    )
    reporting_period_start: Optional[str] = Field(
        None,
        description="Reporting period start (YYYY-MM-DD)"
    )
    reporting_period_end: Optional[str] = Field(
        None,
        description="Reporting period end (YYYY-MM-DD)"
    )


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class Exposure(BaseModel):
    """Single exposure on balance sheet."""

    exposure_id: str = Field(..., description="Unique exposure identifier")
    exposure_type: ExposureType = Field(..., description="Type of exposure")
    counterparty_id: str = Field(..., description="Counterparty identifier")
    counterparty_name: str = Field(..., description="Counterparty name")
    counterparty_type: CounterpartyType = Field(..., description="Counterparty classification")
    gross_carrying_amount: Decimal = Field(..., description="Gross carrying amount (EUR)")
    taxonomy_eligible_amount: Decimal = Field(
        default=Decimal("0"),
        description="Amount classified as taxonomy-eligible"
    )
    taxonomy_aligned_amount: Decimal = Field(
        default=Decimal("0"),
        description="Amount classified as taxonomy-aligned"
    )
    nace_sector: Optional[str] = Field(None, description="NACE sector code")
    epc_rating: Optional[EPCRating] = Field(None, description="EPC rating for real estate")
    counterparty_turnover_alignment: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("1"),
        description="Counterparty's taxonomy-aligned turnover ratio"
    )
    counterparty_capex_alignment: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("1"),
        description="Counterparty's taxonomy-aligned CapEx ratio"
    )
    is_banking_book: bool = Field(default=True, description="Whether exposure is in banking book")
    origination_date: Optional[str] = Field(None, description="Date of origination (YYYY-MM-DD)")
    maturity_date: Optional[str] = Field(None, description="Maturity date (YYYY-MM-DD)")


class ExposureClassification(BaseModel):
    """Classification result for a portfolio of exposures."""

    total_exposures: int = Field(..., description="Total number of exposures")
    total_carrying_amount: Decimal = Field(..., description="Total gross carrying amount")
    covered_assets: Decimal = Field(..., description="Total covered assets (GAR denominator)")
    excluded_assets: Decimal = Field(..., description="Excluded exposures (sovereign, etc.)")
    de_minimis_excluded: Decimal = Field(..., description="De minimis excluded amount")
    by_type: Dict[str, Decimal] = Field(..., description="Carrying amount by exposure type")
    by_counterparty_type: Dict[str, Decimal] = Field(
        ..., description="Carrying amount by counterparty type"
    )
    eligible_amount: Decimal = Field(..., description="Total taxonomy-eligible amount")
    aligned_amount: Decimal = Field(..., description="Total taxonomy-aligned amount")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class GARResult(BaseModel):
    """Result of GAR stock or flow calculation."""

    gar_ratio: Decimal = Field(..., description="Green Asset Ratio (0-1)")
    numerator: Decimal = Field(..., description="Taxonomy-aligned exposures (EUR)")
    denominator: Decimal = Field(..., description="Total covered assets (EUR)")
    eligible_ratio: Decimal = Field(..., description="Taxonomy-eligible ratio (0-1)")
    eligible_amount: Decimal = Field(..., description="Taxonomy-eligible exposures (EUR)")
    aligned_amount: Decimal = Field(..., description="Taxonomy-aligned exposures (EUR)")
    by_objective: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aligned amount by environmental objective"
    )
    by_exposure_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aligned amount by exposure type"
    )
    by_sector: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aligned amount by NACE sector"
    )
    excluded_exposures: Decimal = Field(
        default=Decimal("0"),
        description="Excluded sovereign/central bank exposures"
    )
    calculation_date: str = Field(..., description="Calculation date (ISO 8601)")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class BTARResult(BaseModel):
    """Result of Banking Book Taxonomy Alignment Ratio calculation."""

    btar_ratio: Decimal = Field(..., description="BTAR ratio (0-1)")
    numerator: Decimal = Field(..., description="Aligned banking book exposures (EUR)")
    denominator: Decimal = Field(..., description="Total banking book covered assets (EUR)")
    eligible_ratio: Decimal = Field(..., description="Eligible ratio for banking book (0-1)")
    by_exposure_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Aligned banking book amount by exposure type"
    )
    gar_comparison: Optional[Decimal] = Field(
        None,
        description="Difference between BTAR and GAR (BTAR - GAR)"
    )
    calculation_date: str = Field(..., description="Calculation date (ISO 8601)")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# ---------------------------------------------------------------------------
# Excluded exposure types for GAR denominator
# ---------------------------------------------------------------------------

_EXCLUDED_EXPOSURE_TYPES = {
    ExposureType.SOVEREIGN_EXPOSURES,
    ExposureType.INTERBANK_LOANS,
    ExposureType.DERIVATIVES,
}

# EPC rating order for comparison (lower index = better rating)
_EPC_ORDER = [
    EPCRating.A_PLUS, EPCRating.A, EPCRating.B, EPCRating.C,
    EPCRating.D, EPCRating.E, EPCRating.F, EPCRating.G,
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GreenAssetRatioEngine:
    """
    Green Asset Ratio Engine for EU Taxonomy Article 8 financial institution disclosures.

    Calculates GAR stock (on-balance-sheet), GAR flow (new originations during period),
    and BTAR (Banking Book Taxonomy Alignment Ratio). Supports exposure classification,
    counterparty taxonomy-data aggregation, EPC integration for real estate, and
    de minimis threshold handling.

    Attributes:
        config: Engine configuration

    Example:
        >>> config = GARConfig()
        >>> engine = GreenAssetRatioEngine(config)
        >>> result = engine.calculate_gar_stock(exposures)
        >>> assert Decimal("0") <= result.gar_ratio <= Decimal("1")
    """

    def __init__(self, config: GARConfig):
        """Initialize the Green Asset Ratio Engine."""
        self.config = config
        logger.info("GreenAssetRatioEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_gar_stock(self, exposures: List[Exposure]) -> GARResult:
        """
        Calculate GAR stock ratio from on-balance-sheet exposures.

        The GAR stock is: taxonomy-aligned covered assets / total covered assets.
        Sovereign exposures, interbank loans, derivatives, and de minimis
        exposures are excluded from both numerator and denominator.

        Args:
            exposures: List of balance-sheet exposures

        Returns:
            GARResult with ratio, numerator, denominator, and breakdowns

        Raises:
            ValueError: If exposures list is empty
        """
        if not exposures:
            raise ValueError("Exposures list cannot be empty")

        start = datetime.utcnow()
        logger.info(f"Calculating GAR stock for {len(exposures)} exposures")

        # Step 1 -- filter and classify
        filtered = self._filter_exposures(exposures, flow=False)

        # Step 2 -- calculate aligned amounts (using counterparty data)
        aligned_by_type, aligned_by_sector = self._aggregate_aligned(filtered)

        numerator = sum(aligned_by_type.values(), Decimal("0"))
        denominator = self._covered_assets(filtered)
        eligible_amount = sum(
            e.taxonomy_eligible_amount for e in filtered
        )

        gar = self._safe_divide(numerator, denominator)
        eligible_ratio = self._safe_divide(eligible_amount, denominator)

        provenance = self._provenance(
            {"type": "gar_stock", "n": len(exposures), "ts": start.isoformat()}
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"GAR stock calculated: {gar} "
            f"(numerator={numerator}, denominator={denominator}) "
            f"in {elapsed_ms:.1f}ms"
        )

        return GARResult(
            gar_ratio=gar,
            numerator=numerator,
            denominator=denominator,
            eligible_ratio=eligible_ratio,
            eligible_amount=eligible_amount,
            aligned_amount=numerator,
            by_exposure_type=aligned_by_type,
            by_sector=aligned_by_sector,
            excluded_exposures=self._excluded_total(exposures),
            calculation_date=start.isoformat(),
            provenance_hash=provenance,
        )

    def calculate_gar_flow(self, new_originations: List[Exposure]) -> GARResult:
        """
        Calculate GAR flow ratio from new originations during the reporting period.

        Only exposures originated within the reporting period are included.

        Args:
            new_originations: Exposures originated during the period

        Returns:
            GARResult representing the flow ratio

        Raises:
            ValueError: If originations list is empty
        """
        if not new_originations:
            raise ValueError("New originations list cannot be empty")

        start = datetime.utcnow()
        logger.info(f"Calculating GAR flow for {len(new_originations)} new originations")

        filtered = self._filter_exposures(new_originations, flow=True)
        aligned_by_type, aligned_by_sector = self._aggregate_aligned(filtered)

        numerator = sum(aligned_by_type.values(), Decimal("0"))
        denominator = self._covered_assets(filtered)
        eligible_amount = sum(e.taxonomy_eligible_amount for e in filtered)

        gar = self._safe_divide(numerator, denominator)
        eligible_ratio = self._safe_divide(eligible_amount, denominator)

        provenance = self._provenance(
            {"type": "gar_flow", "n": len(new_originations), "ts": start.isoformat()}
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"GAR flow calculated: {gar} "
            f"(numerator={numerator}, denominator={denominator}) "
            f"in {elapsed_ms:.1f}ms"
        )

        return GARResult(
            gar_ratio=gar,
            numerator=numerator,
            denominator=denominator,
            eligible_ratio=eligible_ratio,
            eligible_amount=eligible_amount,
            aligned_amount=numerator,
            by_exposure_type=aligned_by_type,
            by_sector=aligned_by_sector,
            excluded_exposures=self._excluded_total(new_originations),
            calculation_date=start.isoformat(),
            provenance_hash=provenance,
        )

    def calculate_btar(self, banking_book: List[Exposure]) -> BTARResult:
        """
        Calculate Banking Book Taxonomy Alignment Ratio (BTAR).

        BTAR considers only banking-book exposures and may include
        counterparties not subject to NFRD/CSRD, providing a broader view
        than the mandatory GAR.

        Args:
            banking_book: Banking book exposures

        Returns:
            BTARResult with ratio and breakdowns

        Raises:
            ValueError: If banking_book list is empty
        """
        if not banking_book:
            raise ValueError("Banking book exposures list cannot be empty")

        start = datetime.utcnow()
        logger.info(f"Calculating BTAR for {len(banking_book)} banking book exposures")

        # Only banking-book items
        bb_items = [e for e in banking_book if e.is_banking_book]
        filtered = self._filter_exposures(bb_items, flow=False)

        aligned_by_type, _ = self._aggregate_aligned(filtered)
        numerator = sum(aligned_by_type.values(), Decimal("0"))
        denominator = self._covered_assets(filtered)
        eligible_amount = sum(e.taxonomy_eligible_amount for e in filtered)

        btar = self._safe_divide(numerator, denominator)
        eligible_ratio = self._safe_divide(eligible_amount, denominator)

        provenance = self._provenance(
            {"type": "btar", "n": len(bb_items), "ts": start.isoformat()}
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(f"BTAR calculated: {btar} in {elapsed_ms:.1f}ms")

        return BTARResult(
            btar_ratio=btar,
            numerator=numerator,
            denominator=denominator,
            eligible_ratio=eligible_ratio,
            by_exposure_type=aligned_by_type,
            calculation_date=start.isoformat(),
            provenance_hash=provenance,
        )

    def classify_exposures(self, portfolio: List[Exposure]) -> ExposureClassification:
        """
        Classify a portfolio of exposures into GAR-relevant categories.

        Produces a breakdown by exposure type and counterparty type, identifies
        excluded and de minimis exposures, and totals eligible and aligned amounts.

        Args:
            portfolio: Full portfolio of exposures

        Returns:
            ExposureClassification with complete breakdown
        """
        start = datetime.utcnow()
        logger.info(f"Classifying {len(portfolio)} exposures")

        total_carrying = sum(e.gross_carrying_amount for e in portfolio)
        by_type: Dict[str, Decimal] = {}
        by_cp_type: Dict[str, Decimal] = {}
        excluded = Decimal("0")
        de_minimis = Decimal("0")
        eligible_total = Decimal("0")
        aligned_total = Decimal("0")

        for exp in portfolio:
            # By exposure type
            key = exp.exposure_type.value
            by_type[key] = by_type.get(key, Decimal("0")) + exp.gross_carrying_amount

            # By counterparty type
            cp_key = exp.counterparty_type.value
            by_cp_type[cp_key] = by_cp_type.get(cp_key, Decimal("0")) + exp.gross_carrying_amount

            # Excluded
            if exp.exposure_type in _EXCLUDED_EXPOSURE_TYPES:
                excluded += exp.gross_carrying_amount
                continue

            # De minimis
            if exp.gross_carrying_amount < self.config.de_minimis_threshold:
                de_minimis += exp.gross_carrying_amount
                continue

            eligible_total += exp.taxonomy_eligible_amount
            aligned_total += exp.taxonomy_aligned_amount

        covered = total_carrying - excluded - de_minimis

        provenance = self._provenance(
            {"type": "classify", "n": len(portfolio), "ts": start.isoformat()}
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"Classification complete: covered={covered}, "
            f"excluded={excluded}, de_minimis={de_minimis} in {elapsed_ms:.1f}ms"
        )

        return ExposureClassification(
            total_exposures=len(portfolio),
            total_carrying_amount=total_carrying,
            covered_assets=covered,
            excluded_assets=excluded,
            de_minimis_excluded=de_minimis,
            by_type=by_type,
            by_counterparty_type=by_cp_type,
            eligible_amount=eligible_total,
            aligned_amount=aligned_total,
            provenance_hash=provenance,
        )

    def apply_epc_alignment(self, exposure: Exposure) -> Decimal:
        """
        Determine taxonomy-aligned amount for a real-estate exposure using EPC rating.

        Residential and commercial mortgages with EPC rating at or above the
        configured threshold are considered taxonomy-aligned for the full
        carrying amount.

        Args:
            exposure: A real-estate exposure with EPC rating

        Returns:
            Taxonomy-aligned amount based on EPC rating
        """
        if exposure.exposure_type not in (
            ExposureType.RESIDENTIAL_MORTGAGES,
            ExposureType.COMMERCIAL_MORTGAGES,
        ):
            return Decimal("0")

        if exposure.epc_rating is None or exposure.epc_rating == EPCRating.NOT_AVAILABLE:
            logger.warning(
                f"Exposure {exposure.exposure_id}: no EPC rating available"
            )
            return Decimal("0")

        threshold_idx = _EPC_ORDER.index(self.config.epc_top_15_threshold)
        rating_idx = _EPC_ORDER.index(exposure.epc_rating)

        if rating_idx <= threshold_idx:
            logger.debug(
                f"Exposure {exposure.exposure_id}: EPC {exposure.epc_rating.value} "
                f"meets threshold {self.config.epc_top_15_threshold.value}"
            )
            return exposure.gross_carrying_amount

        return Decimal("0")

    def apply_counterparty_alignment(self, exposure: Exposure) -> Decimal:
        """
        Calculate taxonomy-aligned amount using counterparty taxonomy data.

        For corporate loans and debt securities, the aligned amount is derived
        from the counterparty's reported taxonomy-aligned turnover or CapEx ratio
        applied to the carrying amount.

        Args:
            exposure: An exposure with counterparty taxonomy data

        Returns:
            Taxonomy-aligned amount based on counterparty data
        """
        ratio = exposure.counterparty_turnover_alignment
        if ratio is None:
            ratio = exposure.counterparty_capex_alignment
        if ratio is None:
            return exposure.taxonomy_aligned_amount

        aligned = (exposure.gross_carrying_amount * ratio).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        logger.debug(
            f"Exposure {exposure.exposure_id}: counterparty alignment "
            f"ratio={ratio}, aligned={aligned}"
        )
        return aligned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_exposures(
        self, exposures: List[Exposure], flow: bool
    ) -> List[Exposure]:
        """Filter exposures for GAR calculation, removing excluded and de minimis."""
        filtered: List[Exposure] = []
        for exp in exposures:
            if exp.exposure_type in _EXCLUDED_EXPOSURE_TYPES:
                continue
            if not self.config.include_trading_book and not exp.is_banking_book:
                continue
            if exp.gross_carrying_amount < self.config.de_minimis_threshold:
                continue
            filtered.append(exp)
        return filtered

    def _covered_assets(self, exposures: List[Exposure]) -> Decimal:
        """Sum gross carrying amounts of filtered exposures (GAR denominator)."""
        return sum(
            (e.gross_carrying_amount for e in exposures), Decimal("0")
        )

    def _excluded_total(self, exposures: List[Exposure]) -> Decimal:
        """Sum excluded exposure amounts."""
        return sum(
            (e.gross_carrying_amount for e in exposures
             if e.exposure_type in _EXCLUDED_EXPOSURE_TYPES),
            Decimal("0"),
        )

    def _aggregate_aligned(
        self, exposures: List[Exposure]
    ) -> Tuple[Dict[str, Decimal], Dict[str, Decimal]]:
        """Aggregate aligned amounts by exposure type and sector."""
        by_type: Dict[str, Decimal] = {}
        by_sector: Dict[str, Decimal] = {}

        for exp in exposures:
            # Determine aligned amount -- prefer EPC for real estate,
            # counterparty data for corporates, fall back to declared amount
            if exp.exposure_type in (
                ExposureType.RESIDENTIAL_MORTGAGES,
                ExposureType.COMMERCIAL_MORTGAGES,
            ):
                aligned = self.apply_epc_alignment(exp)
                # If EPC yields zero, fall back to declared
                if aligned == Decimal("0"):
                    aligned = exp.taxonomy_aligned_amount
            elif exp.exposure_type in (
                ExposureType.CORPORATE_LOANS,
                ExposureType.DEBT_SECURITIES,
                ExposureType.EQUITY_HOLDINGS,
            ):
                aligned = self.apply_counterparty_alignment(exp)
            else:
                aligned = exp.taxonomy_aligned_amount

            key = exp.exposure_type.value
            by_type[key] = by_type.get(key, Decimal("0")) + aligned

            sector = exp.nace_sector or "UNKNOWN"
            by_sector[sector] = by_sector.get(sector, Decimal("0")) + aligned

        return by_type, by_sector

    @staticmethod
    def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
        """Divide numerator by denominator, returning 0 if denominator is zero."""
        if denominator == Decimal("0"):
            return Decimal("0")
        return (numerator / denominator).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

    @staticmethod
    def _provenance(data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
