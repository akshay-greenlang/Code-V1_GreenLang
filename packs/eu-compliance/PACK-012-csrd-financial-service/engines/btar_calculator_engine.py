# -*- coding: utf-8 -*-
"""
BTARCalculatorEngine - PACK-012 CSRD Financial Service Engine 4
=================================================================

Banking Book Taxonomy Alignment Ratio (BTAR) calculator for credit
institutions.  The BTAR extends the GAR to cover the FULL banking
book, including exposures that the GAR excludes (SMEs not subject
to CSRD, non-EU counterparties, derivatives, intangibles, etc.).

Where the GAR only counts verified Taxonomy alignment data from
CSRD-reporting counterparties, the BTAR allows estimation methods
for counterparties that do not yet report under CSRD.

Core Formulas:
    BTAR = Estimated Aligned Banking Book / Total Banking Book
    Estimation Ratio   = Estimated Exposures / Total Banking Book
    Data Coverage      = Exposures with Data / Total Banking Book
    BTAR-GAR Gap       = BTAR - GAR (should be >= 0)
    Confidence Score   = Weighted avg of estimation confidence

Estimation Methodologies:
    1. CSRD-reported data (highest confidence, same as GAR)
    2. Sector proxy (NACE sector average alignment)
    3. Internal ESG scoring (proprietary models)
    4. Third-party data providers (MSCI, ISS, Sustainalytics)
    5. Geographic proxy (country-level alignment rates)
    6. Conservative default (0% alignment)

Banking Book Scope (Full):
    - All GAR-covered assets
    - SMEs not subject to CSRD
    - Non-EU counterparties
    - Interbank exposures
    - Derivatives (banking book hedging)
    - Intangible assets
    - Other assets not in GAR denominator

Regulatory References:
    - EBA Report on the Role of ESG Risks (EBA/REP/2021/18)
    - EBA Pillar 3 ITS on ESG Disclosures (EBA/ITS/2022/01)
    - EU Taxonomy Regulation 2020/852
    - CRR Article 449a
    - ECB Guide on climate-related and environmental risks

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - Estimation methods are rule-based lookups (no LLM)
    - Confidence scoring uses deterministic weighted averages
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage or 0.0 on zero denominator.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EstimationType(str, Enum):
    """Estimation methodology types for BTAR."""
    CSRD_REPORTED = "csrd_reported"         # Direct CSRD Taxonomy data
    SECTOR_PROXY = "sector_proxy"           # NACE sector average alignment
    INTERNAL_ESG = "internal_esg"           # Internal ESG scoring model
    THIRD_PARTY = "third_party"             # Third-party data provider
    GEOGRAPHIC_PROXY = "geographic_proxy"   # Country-level proxy
    CONSERVATIVE_DEFAULT = "conservative_default"  # 0% alignment default


class ExposureCategory(str, Enum):
    """Banking book exposure categories."""
    GAR_COVERED_NFC = "gar_covered_nfc"               # NFCs in GAR scope
    GAR_COVERED_HOUSEHOLD = "gar_covered_household"    # Households in GAR scope
    SME_NON_CSRD = "sme_non_csrd"                     # SMEs not subject to CSRD
    NON_EU_COUNTERPARTY = "non_eu_counterparty"        # Non-EU entities
    INTERBANK = "interbank"                             # Interbank exposures
    SOVEREIGN = "sovereign"                             # Sovereign exposures
    DERIVATIVE_HEDGE = "derivative_hedge"               # Banking book derivatives
    INTANGIBLE = "intangible"                           # Intangible assets
    TRADING_BOOK = "trading_book"                       # Trading book (excluded from BB)
    OTHER = "other"                                     # Other banking book items


class ConfidenceLevel(str, Enum):
    """Confidence level for estimated alignment."""
    HIGH = "high"           # CSRD-reported data
    MEDIUM_HIGH = "medium_high"  # Third-party verified
    MEDIUM = "medium"       # Internal ESG / sector proxy
    LOW = "low"             # Geographic proxy
    VERY_LOW = "very_low"   # Conservative default


# ---------------------------------------------------------------------------
# Default Sector Proxy Alignment Rates
# ---------------------------------------------------------------------------

# Estimated Taxonomy alignment rates by NACE sector (%)
# Based on EBA reports and industry analysis
SECTOR_PROXY_ALIGNMENT: Dict[str, float] = {
    "A": 5.0,     # Agriculture
    "B": 2.0,     # Mining
    "C": 12.0,    # Manufacturing
    "D": 25.0,    # Electricity, gas
    "E": 18.0,    # Water, waste
    "F": 15.0,    # Construction
    "G": 6.0,     # Wholesale/retail
    "H": 8.0,     # Transport
    "I": 4.0,     # Accommodation
    "J": 20.0,    # ICT
    "K": 10.0,    # Financial
    "L": 12.0,    # Real estate
    "M": 15.0,    # Professional
    "N": 8.0,     # Administrative
    "O": 5.0,     # Public admin
    "P": 10.0,    # Education
    "Q": 8.0,     # Health
    "R": 3.0,     # Arts
    "S": 5.0,     # Other services
    "T": 2.0,     # Households
    "U": 1.0,     # Extraterritorial
}

# Geographic proxy: country-level alignment estimate (%)
GEOGRAPHIC_PROXY_ALIGNMENT: Dict[str, float] = {
    "DE": 18.0,  # Germany
    "FR": 20.0,  # France
    "NL": 22.0,  # Netherlands
    "SE": 25.0,  # Sweden
    "DK": 24.0,  # Denmark
    "FI": 22.0,  # Finland
    "AT": 16.0,  # Austria
    "BE": 15.0,  # Belgium
    "ES": 12.0,  # Spain
    "IT": 10.0,  # Italy
    "PT": 8.0,   # Portugal
    "GR": 6.0,   # Greece
    "IE": 18.0,  # Ireland
    "LU": 20.0,  # Luxembourg
    "PL": 8.0,   # Poland
    "CZ": 10.0,  # Czech Republic
    "RO": 5.0,   # Romania
    "HU": 7.0,   # Hungary
    "US": 12.0,  # United States
    "GB": 16.0,  # United Kingdom
    "CH": 18.0,  # Switzerland
    "NO": 22.0,  # Norway
    "JP": 10.0,  # Japan
    "CN": 8.0,   # China
    "IN": 4.0,   # India
    "BR": 6.0,   # Brazil
    "AU": 12.0,  # Australia
    "CA": 14.0,  # Canada
    "KR": 12.0,  # South Korea
}

# Confidence scores by estimation type (0.0 to 1.0)
ESTIMATION_CONFIDENCE: Dict[str, float] = {
    EstimationType.CSRD_REPORTED.value: 0.95,
    EstimationType.THIRD_PARTY.value: 0.75,
    EstimationType.INTERNAL_ESG.value: 0.65,
    EstimationType.SECTOR_PROXY.value: 0.50,
    EstimationType.GEOGRAPHIC_PROXY.value: 0.35,
    EstimationType.CONSERVATIVE_DEFAULT.value: 0.10,
}


# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------


class BankingBookData(BaseModel):
    """Input data for a single banking book exposure.

    Contains the exposure details, counterparty information, and
    any available Taxonomy alignment data or estimation inputs.

    Attributes:
        exposure_id: Unique exposure identifier.
        exposure_name: Name / description of the exposure.
        exposure_category: Banking book category.
        carrying_amount: Carrying amount (EUR).
        nace_sector: NACE sector code.
        country: Country of counterparty (ISO 3166-1).
        is_sme: Whether the counterparty is an SME.
        is_eu: Whether the counterparty is EU-domiciled.
        is_csrd_subject: Whether subject to CSRD reporting.
        reported_turnover_aligned_pct: CSRD-reported turnover alignment.
        reported_capex_aligned_pct: CSRD-reported CapEx alignment.
        reported_opex_aligned_pct: CSRD-reported OpEx alignment.
        taxonomy_eligible_pct: Reported Taxonomy eligibility.
        estimation_type: Estimation method to use (if not CSRD-reported).
        third_party_alignment_pct: Third-party provided alignment estimate.
        third_party_provider: Name of third-party data provider.
        internal_esg_score: Internal ESG score (0-100).
        internal_esg_aligned_pct: Internal model alignment estimate.
        is_in_gar_scope: Whether this exposure is in the GAR scope.
        gar_aligned_pct: GAR alignment percentage (if in scope).
    """
    exposure_id: str = Field(default_factory=_new_uuid, description="Unique exposure ID")
    exposure_name: str = Field(default="", description="Exposure name / description")
    exposure_category: ExposureCategory = Field(
        default=ExposureCategory.OTHER,
        description="Banking book category",
    )
    carrying_amount: float = Field(
        default=0.0, ge=0.0, description="Carrying amount (EUR)",
    )
    nace_sector: str = Field(default="", description="NACE sector code")
    country: str = Field(default="", description="Country (ISO 3166)")
    is_sme: bool = Field(default=False, description="Whether counterparty is SME")
    is_eu: bool = Field(default=True, description="Whether counterparty is EU-domiciled")
    is_csrd_subject: bool = Field(
        default=False, description="Whether subject to CSRD",
    )
    # CSRD-reported alignment data
    reported_turnover_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CSRD-reported turnover alignment (%)",
    )
    reported_capex_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CSRD-reported CapEx alignment (%)",
    )
    reported_opex_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CSRD-reported OpEx alignment (%)",
    )
    taxonomy_eligible_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Reported Taxonomy eligibility (%)",
    )
    # Estimation inputs
    estimation_type: Optional[EstimationType] = Field(
        default=None,
        description="Estimation method (if not CSRD-reported)",
    )
    third_party_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Third-party alignment estimate (%)",
    )
    third_party_provider: str = Field(
        default="", description="Third-party data provider name",
    )
    internal_esg_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Internal ESG score (0-100)",
    )
    internal_esg_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Internal model alignment estimate (%)",
    )
    # GAR linkage
    is_in_gar_scope: bool = Field(
        default=False, description="Whether in GAR scope",
    )
    gar_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="GAR alignment percentage (if in scope)",
    )


class SectorProxyResult(BaseModel):
    """Result of sector proxy estimation for an exposure.

    Attributes:
        exposure_id: Exposure identifier.
        nace_sector: NACE sector used.
        sector_alignment_pct: Sector average alignment.
        applied_alignment_pct: Alignment applied to exposure.
        confidence: Confidence score (0.0-1.0).
        source: Source of the sector data.
    """
    exposure_id: str = Field(default="", description="Exposure identifier")
    nace_sector: str = Field(default="", description="NACE sector")
    sector_alignment_pct: float = Field(
        default=0.0, description="Sector average alignment (%)",
    )
    applied_alignment_pct: float = Field(
        default=0.0, description="Applied alignment (%)",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence score",
    )
    source: str = Field(default="PCAF/EBA sector averages", description="Data source")


class EstimationMethodology(BaseModel):
    """Estimation methodology applied to a single exposure.

    Attributes:
        exposure_id: Exposure identifier.
        estimation_type: Method used.
        estimated_alignment_pct: Estimated alignment.
        confidence: Confidence score.
        data_source: Source of estimation data.
        methodology_note: Description of the methodology.
        provenance_hash: SHA-256 hash.
    """
    exposure_id: str = Field(default="", description="Exposure identifier")
    estimation_type: EstimationType = Field(description="Estimation method")
    estimated_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Estimated alignment (%)",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence score",
    )
    data_source: str = Field(default="", description="Source of estimation data")
    methodology_note: str = Field(default="", description="Methodology description")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class DataCoverageReport(BaseModel):
    """Data coverage statistics for the banking book.

    Attributes:
        total_exposures: Total number of exposures.
        total_amount: Total banking book amount (EUR).
        csrd_reported_count: Exposures with CSRD data.
        csrd_reported_amount: Amount with CSRD data (EUR).
        csrd_reported_pct: CSRD-reported as % of total.
        estimated_count: Exposures with estimated alignment.
        estimated_amount: Estimated amount (EUR).
        estimated_pct: Estimated as % of total.
        no_data_count: Exposures with no alignment data.
        no_data_amount: Amount with no data (EUR).
        no_data_pct: No-data as % of total.
        estimation_method_breakdown: Count by estimation method.
    """
    total_exposures: int = Field(default=0, ge=0, description="Total exposures")
    total_amount: float = Field(default=0.0, description="Total amount (EUR)")
    csrd_reported_count: int = Field(default=0, ge=0, description="CSRD-reported count")
    csrd_reported_amount: float = Field(default=0.0, description="CSRD-reported amount")
    csrd_reported_pct: float = Field(default=0.0, description="CSRD-reported (%)")
    estimated_count: int = Field(default=0, ge=0, description="Estimated count")
    estimated_amount: float = Field(default=0.0, description="Estimated amount")
    estimated_pct: float = Field(default=0.0, description="Estimated (%)")
    no_data_count: int = Field(default=0, ge=0, description="No-data count")
    no_data_amount: float = Field(default=0.0, description="No-data amount")
    no_data_pct: float = Field(default=0.0, description="No-data (%)")
    estimation_method_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by estimation method",
    )


class BTARvsGARReconciliation(BaseModel):
    """Reconciliation between BTAR and GAR.

    The BTAR should always be >= 0.  The difference between BTAR
    and GAR represents estimated alignment of non-GAR exposures.

    Attributes:
        gar_turnover_pct: GAR turnover alignment (%).
        btar_turnover_pct: BTAR turnover alignment (%).
        gap_turnover_pct: BTAR - GAR turnover difference.
        gar_covered_amount: GAR covered assets (EUR).
        btar_total_amount: BTAR total banking book (EUR).
        additional_aligned_amount: Additional aligned from estimation.
        reconciliation_status: Whether BTAR >= GAR (expected).
        explanation: Explanation of the gap.
        provenance_hash: SHA-256 hash.
    """
    gar_turnover_pct: float = Field(default=0.0, description="GAR turnover (%)")
    btar_turnover_pct: float = Field(default=0.0, description="BTAR turnover (%)")
    gap_turnover_pct: float = Field(default=0.0, description="BTAR - GAR gap")
    gar_covered_amount: float = Field(default=0.0, description="GAR covered (EUR)")
    btar_total_amount: float = Field(default=0.0, description="BTAR total (EUR)")
    additional_aligned_amount: float = Field(
        default=0.0, description="Additional aligned from estimation (EUR)",
    )
    reconciliation_status: str = Field(
        default="", description="PASS if BTAR >= GAR",
    )
    explanation: str = Field(default="", description="Explanation of gap")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class BTARResult(BaseModel):
    """Complete BTAR calculation result.

    Top-level result containing the BTAR ratio, estimation details,
    data coverage report, and GAR reconciliation.

    Attributes:
        result_id: Unique result identifier.
        reporting_year: Reporting year.
        total_banking_book: Total banking book amount (EUR).
        total_aligned_estimated: Total estimated aligned amount (EUR).
        btar_turnover_pct: BTAR turnover alignment (%).
        btar_capex_pct: BTAR CapEx alignment (%).
        btar_opex_pct: BTAR OpEx alignment (%).
        weighted_confidence_score: Portfolio weighted confidence.
        estimation_methodologies: Per-exposure estimation details.
        data_coverage: Data coverage report.
        category_breakdown: Breakdown by exposure category.
        gar_reconciliation: BTAR vs GAR reconciliation.
        total_exposures: Total number of exposures.
        methodology_notes: Methodology notes.
        processing_time_ms: Processing time (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    reporting_year: int = Field(default=2024, description="Reporting year")
    total_banking_book: float = Field(
        default=0.0, description="Total banking book (EUR)",
    )
    total_aligned_estimated: float = Field(
        default=0.0, description="Total estimated aligned (EUR)",
    )
    btar_turnover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="BTAR turnover alignment (%)",
    )
    btar_capex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="BTAR CapEx alignment (%)",
    )
    btar_opex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="BTAR OpEx alignment (%)",
    )
    weighted_confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Portfolio weighted confidence score",
    )
    estimation_methodologies: List[EstimationMethodology] = Field(
        default_factory=list,
        description="Per-exposure estimation details",
    )
    data_coverage: Optional[DataCoverageReport] = Field(
        default=None, description="Data coverage report",
    )
    category_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Breakdown by exposure category",
    )
    gar_reconciliation: Optional[BTARvsGARReconciliation] = Field(
        default=None, description="BTAR vs GAR reconciliation",
    )
    total_exposures: int = Field(default=0, ge=0, description="Total exposures")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes",
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class BTARConfig(BaseModel):
    """Configuration for the BTARCalculatorEngine.

    Controls estimation methodologies, proxy data, confidence scoring,
    and GAR reconciliation parameters.

    Attributes:
        reporting_year: Reporting year.
        sector_proxy_alignment: NACE sector proxy alignment rates.
        geographic_proxy_alignment: Country proxy alignment rates.
        estimation_confidence: Confidence scores by estimation type.
        default_estimation_type: Default estimation method for unknowns.
        sme_default_alignment_pct: Default alignment for SMEs.
        non_eu_default_alignment_pct: Default alignment for non-EU.
        interbank_alignment_pct: Alignment for interbank exposures.
        sovereign_alignment_pct: Alignment for sovereign exposures.
        internal_esg_score_to_alignment: Mapping of ESG score ranges to alignment.
        gar_turnover_pct: GAR turnover % (for reconciliation).
        gar_covered_amount: GAR covered assets (EUR) for reconciliation.
        precision_decimal_places: Decimal places for rounding.
    """
    reporting_year: int = Field(default=2024, description="Reporting year")
    sector_proxy_alignment: Dict[str, float] = Field(
        default_factory=lambda: dict(SECTOR_PROXY_ALIGNMENT),
        description="NACE sector proxy alignment rates (%)",
    )
    geographic_proxy_alignment: Dict[str, float] = Field(
        default_factory=lambda: dict(GEOGRAPHIC_PROXY_ALIGNMENT),
        description="Country proxy alignment rates (%)",
    )
    estimation_confidence: Dict[str, float] = Field(
        default_factory=lambda: dict(ESTIMATION_CONFIDENCE),
        description="Confidence scores by estimation type",
    )
    default_estimation_type: EstimationType = Field(
        default=EstimationType.SECTOR_PROXY,
        description="Default estimation method for unknowns",
    )
    sme_default_alignment_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Default alignment for SMEs without data",
    )
    non_eu_default_alignment_pct: float = Field(
        default=3.0, ge=0.0, le=100.0,
        description="Default alignment for non-EU counterparties",
    )
    interbank_alignment_pct: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Estimated alignment for interbank exposures",
    )
    sovereign_alignment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Alignment for sovereign exposures (typically 0%)",
    )
    internal_esg_score_to_alignment: Dict[str, float] = Field(
        default_factory=lambda: {
            "0-20": 0.0,
            "21-40": 5.0,
            "41-60": 15.0,
            "61-80": 30.0,
            "81-100": 50.0,
        },
        description="Mapping of ESG score ranges to alignment estimates",
    )
    gar_turnover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="GAR turnover % for reconciliation",
    )
    gar_covered_amount: float = Field(
        default=0.0, ge=0.0,
        description="GAR covered assets (EUR) for reconciliation",
    )
    precision_decimal_places: int = Field(
        default=4, ge=0, le=10,
        description="Decimal places for rounding",
    )


# ---------------------------------------------------------------------------
# Model rebuilds for forward references
# ---------------------------------------------------------------------------

BankingBookData.model_rebuild()
SectorProxyResult.model_rebuild()
EstimationMethodology.model_rebuild()
DataCoverageReport.model_rebuild()
BTARvsGARReconciliation.model_rebuild()
BTARResult.model_rebuild()
BTARConfig.model_rebuild()


# ---------------------------------------------------------------------------
# BTARCalculatorEngine
# ---------------------------------------------------------------------------


class BTARCalculatorEngine:
    """
    Banking Book Taxonomy Alignment Ratio calculator.

    Extends the GAR to cover the full banking book by applying
    estimation methodologies to exposures not covered by CSRD
    reporting.  Provides a more comprehensive view of the bank's
    Taxonomy alignment than the GAR alone.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Python arithmetic
        - Estimation methods are rule-based lookups
        - Confidence scoring uses deterministic weighted averages
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Example:
        >>> config = BTARConfig(reporting_year=2024, gar_turnover_pct=8.5)
        >>> engine = BTARCalculatorEngine(config)
        >>> exposures = [BankingBookData(
        ...     carrying_amount=50_000_000,
        ...     exposure_category=ExposureCategory.SME_NON_CSRD,
        ...     nace_sector="C",
        ...     country="DE",
        ... )]
        >>> result = engine.calculate_btar(exposures)
        >>> assert result.btar_turnover_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize BTARCalculatorEngine.

        Args:
            config: Optional BTARConfig or dict.
        """
        if config and isinstance(config, dict):
            self.config = BTARConfig(**config)
        elif config and isinstance(config, BTARConfig):
            self.config = config
        else:
            self.config = BTARConfig()

        self._exposures: List[BankingBookData] = []

        logger.info(
            "BTARCalculatorEngine initialized (version=%s, year=%d)",
            _MODULE_VERSION,
            self.config.reporting_year,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_btar(
        self,
        exposures: List[BankingBookData],
    ) -> BTARResult:
        """Calculate the Banking Book Taxonomy Alignment Ratio.

        Pipeline:
        1. Classify each exposure (CSRD-reported vs estimated)
        2. Apply appropriate estimation methodology
        3. Calculate alignment amounts
        4. Sum to get total BTAR
        5. Build data coverage report
        6. Compute weighted confidence score
        7. Reconcile with GAR
        8. Generate category breakdown

        Args:
            exposures: List of BankingBookData for the full banking book.

        Returns:
            BTARResult with full breakdown and provenance.

        Raises:
            ValueError: If exposures list is empty.
        """
        start = _utcnow()

        if not exposures:
            raise ValueError("Exposures list cannot be empty")

        self._exposures = exposures

        logger.info("Calculating BTAR for %d exposures", len(exposures))

        # Step 1-2: Estimate alignment for each exposure
        estimations: List[EstimationMethodology] = []
        aligned_amounts_turnover: Dict[str, float] = {}
        aligned_amounts_capex: Dict[str, float] = {}
        aligned_amounts_opex: Dict[str, float] = {}

        for exp in exposures:
            est = self._estimate_alignment(exp)
            estimations.append(est)

            # Apply estimated alignment to carrying amount
            amt = exp.carrying_amount
            if est.estimation_type == EstimationType.CSRD_REPORTED:
                aligned_amounts_turnover[exp.exposure_id] = amt * (exp.reported_turnover_aligned_pct / 100.0)
                aligned_amounts_capex[exp.exposure_id] = amt * (exp.reported_capex_aligned_pct / 100.0)
                aligned_amounts_opex[exp.exposure_id] = amt * (exp.reported_opex_aligned_pct / 100.0)
            else:
                # For estimated, apply same rate across all variants
                aligned_pct = est.estimated_alignment_pct / 100.0
                aligned_amounts_turnover[exp.exposure_id] = amt * aligned_pct
                aligned_amounts_capex[exp.exposure_id] = amt * aligned_pct
                aligned_amounts_opex[exp.exposure_id] = amt * aligned_pct

        # Step 3: Totals
        total_bb = sum(e.carrying_amount for e in exposures)
        total_aligned_turnover = sum(aligned_amounts_turnover.values())
        total_aligned_capex = sum(aligned_amounts_capex.values())
        total_aligned_opex = sum(aligned_amounts_opex.values())

        # Step 4: BTAR percentages
        btar_turnover = _safe_pct(total_aligned_turnover, total_bb)
        btar_capex = _safe_pct(total_aligned_capex, total_bb)
        btar_opex = _safe_pct(total_aligned_opex, total_bb)

        # Step 5: Data coverage
        coverage = self._build_data_coverage(exposures, estimations)

        # Step 6: Weighted confidence
        confidence = self._compute_weighted_confidence(exposures, estimations)

        # Step 7: GAR reconciliation
        reconciliation = self._reconcile_with_gar(
            btar_turnover, total_bb, total_aligned_turnover,
        )

        # Step 8: Category breakdown
        category_breakdown = self._build_category_breakdown(
            exposures, estimations, aligned_amounts_turnover,
        )

        # Methodology notes
        notes = self._generate_methodology_notes(exposures, estimations, coverage)

        end = _utcnow()
        processing_ms = (end - start).total_seconds() * 1000.0

        result = BTARResult(
            reporting_year=self.config.reporting_year,
            total_banking_book=_round_val(total_bb, 2),
            total_aligned_estimated=_round_val(total_aligned_turnover, 2),
            btar_turnover_pct=_round_val(btar_turnover, 4),
            btar_capex_pct=_round_val(btar_capex, 4),
            btar_opex_pct=_round_val(btar_opex, 4),
            weighted_confidence_score=_round_val(confidence, 4),
            estimation_methodologies=estimations,
            data_coverage=coverage,
            category_breakdown=category_breakdown,
            gar_reconciliation=reconciliation,
            total_exposures=len(exposures),
            methodology_notes=notes,
            processing_time_ms=_round_val(processing_ms, 2),
        )

        result.provenance_hash = _compute_hash(result)
        logger.info(
            "BTAR: turnover=%.4f%%, confidence=%.4f, "
            "total_bb=%.2f, aligned=%.2f",
            result.btar_turnover_pct,
            result.weighted_confidence_score,
            result.total_banking_book,
            result.total_aligned_estimated,
        )
        return result

    def estimate_single_exposure(
        self, exposure: BankingBookData,
    ) -> EstimationMethodology:
        """Estimate Taxonomy alignment for a single exposure.

        Args:
            exposure: BankingBookData.

        Returns:
            EstimationMethodology with the estimation result.
        """
        return self._estimate_alignment(exposure)

    def get_sector_proxy(self, nace_sector: str) -> SectorProxyResult:
        """Look up sector proxy alignment rate.

        Args:
            nace_sector: NACE sector code.

        Returns:
            SectorProxyResult with the lookup result.
        """
        nace = nace_sector.upper()[:1] if nace_sector else ""
        rate = self.config.sector_proxy_alignment.get(nace, 0.0)
        confidence = self.config.estimation_confidence.get(
            EstimationType.SECTOR_PROXY.value, 0.50,
        )

        return SectorProxyResult(
            nace_sector=nace,
            sector_alignment_pct=rate,
            applied_alignment_pct=rate,
            confidence=confidence,
            source="PCAF/EBA sector averages",
        )

    # ------------------------------------------------------------------
    # Internal: Estimation Logic
    # ------------------------------------------------------------------

    def _estimate_alignment(
        self, exposure: BankingBookData,
    ) -> EstimationMethodology:
        """Estimate alignment for a single exposure.

        Decision tree:
        1. If CSRD-reported data available -> use directly
        2. If third-party data available -> use third-party
        3. If internal ESG score available -> use internal model
        4. If sector known -> use sector proxy
        5. If country known -> use geographic proxy
        6. Otherwise -> conservative default (0%)

        Args:
            exposure: BankingBookData.

        Returns:
            EstimationMethodology.
        """
        # Priority 1: CSRD-reported
        if exposure.is_csrd_subject and exposure.reported_turnover_aligned_pct > 0:
            return self._build_estimation(
                exposure,
                EstimationType.CSRD_REPORTED,
                exposure.reported_turnover_aligned_pct,
                "CSRD-reported Taxonomy alignment data",
            )

        # Priority 2: Explicit estimation type set
        if exposure.estimation_type:
            return self._apply_estimation_type(exposure, exposure.estimation_type)

        # Priority 3: Third-party data
        if exposure.third_party_alignment_pct > 0:
            return self._build_estimation(
                exposure,
                EstimationType.THIRD_PARTY,
                exposure.third_party_alignment_pct,
                f"Third-party: {exposure.third_party_provider or 'unspecified'}",
            )

        # Priority 4: Internal ESG
        if exposure.internal_esg_score > 0:
            aligned = self._esg_score_to_alignment(exposure.internal_esg_score)
            return self._build_estimation(
                exposure,
                EstimationType.INTERNAL_ESG,
                aligned,
                f"Internal ESG score: {exposure.internal_esg_score:.1f}",
            )

        # Priority 5: Sector proxy
        nace = exposure.nace_sector.upper()[:1] if exposure.nace_sector else ""
        if nace and nace in self.config.sector_proxy_alignment:
            rate = self.config.sector_proxy_alignment[nace]
            return self._build_estimation(
                exposure,
                EstimationType.SECTOR_PROXY,
                rate,
                f"Sector proxy: NACE {nace} = {rate:.1f}%",
            )

        # Priority 6: Geographic proxy
        country = exposure.country.upper()[:2] if exposure.country else ""
        if country and country in self.config.geographic_proxy_alignment:
            rate = self.config.geographic_proxy_alignment[country]
            return self._build_estimation(
                exposure,
                EstimationType.GEOGRAPHIC_PROXY,
                rate,
                f"Geographic proxy: {country} = {rate:.1f}%",
            )

        # Priority 7: Category-specific defaults
        if exposure.exposure_category == ExposureCategory.SME_NON_CSRD:
            return self._build_estimation(
                exposure,
                EstimationType.CONSERVATIVE_DEFAULT,
                self.config.sme_default_alignment_pct,
                "SME default alignment",
            )

        if exposure.exposure_category == ExposureCategory.NON_EU_COUNTERPARTY:
            return self._build_estimation(
                exposure,
                EstimationType.CONSERVATIVE_DEFAULT,
                self.config.non_eu_default_alignment_pct,
                "Non-EU default alignment",
            )

        if exposure.exposure_category == ExposureCategory.INTERBANK:
            return self._build_estimation(
                exposure,
                EstimationType.SECTOR_PROXY,
                self.config.interbank_alignment_pct,
                "Interbank default alignment",
            )

        if exposure.exposure_category == ExposureCategory.SOVEREIGN:
            return self._build_estimation(
                exposure,
                EstimationType.CONSERVATIVE_DEFAULT,
                self.config.sovereign_alignment_pct,
                "Sovereign exposure (typically 0%)",
            )

        # Fallback: conservative default
        return self._build_estimation(
            exposure,
            EstimationType.CONSERVATIVE_DEFAULT,
            0.0,
            "No data available, conservative 0% default",
        )

    def _apply_estimation_type(
        self,
        exposure: BankingBookData,
        est_type: EstimationType,
    ) -> EstimationMethodology:
        """Apply a specific estimation type to an exposure.

        Args:
            exposure: BankingBookData.
            est_type: Estimation type to apply.

        Returns:
            EstimationMethodology.
        """
        if est_type == EstimationType.CSRD_REPORTED:
            return self._build_estimation(
                exposure, est_type,
                exposure.reported_turnover_aligned_pct,
                "CSRD-reported data",
            )

        if est_type == EstimationType.THIRD_PARTY:
            return self._build_estimation(
                exposure, est_type,
                exposure.third_party_alignment_pct,
                f"Third-party: {exposure.third_party_provider}",
            )

        if est_type == EstimationType.INTERNAL_ESG:
            aligned = (
                exposure.internal_esg_aligned_pct
                if exposure.internal_esg_aligned_pct > 0
                else self._esg_score_to_alignment(exposure.internal_esg_score)
            )
            return self._build_estimation(
                exposure, est_type, aligned,
                f"Internal ESG model (score={exposure.internal_esg_score:.1f})",
            )

        if est_type == EstimationType.SECTOR_PROXY:
            nace = exposure.nace_sector.upper()[:1] if exposure.nace_sector else ""
            rate = self.config.sector_proxy_alignment.get(nace, 0.0)
            return self._build_estimation(
                exposure, est_type, rate,
                f"Sector proxy: NACE {nace}",
            )

        if est_type == EstimationType.GEOGRAPHIC_PROXY:
            country = exposure.country.upper()[:2] if exposure.country else ""
            rate = self.config.geographic_proxy_alignment.get(country, 0.0)
            return self._build_estimation(
                exposure, est_type, rate,
                f"Geographic proxy: {country}",
            )

        # Conservative default
        return self._build_estimation(
            exposure,
            EstimationType.CONSERVATIVE_DEFAULT,
            0.0,
            "Conservative default (0%)",
        )

    def _build_estimation(
        self,
        exposure: BankingBookData,
        est_type: EstimationType,
        alignment_pct: float,
        note: str,
    ) -> EstimationMethodology:
        """Build an EstimationMethodology result.

        Args:
            exposure: The exposure being estimated.
            est_type: Estimation type.
            alignment_pct: Estimated alignment percentage.
            note: Methodology note.

        Returns:
            EstimationMethodology.
        """
        confidence = self.config.estimation_confidence.get(
            est_type.value, 0.10,
        )

        result = EstimationMethodology(
            exposure_id=exposure.exposure_id,
            estimation_type=est_type,
            estimated_alignment_pct=_round_val(alignment_pct, 4),
            confidence=confidence,
            data_source=note,
            methodology_note=note,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _esg_score_to_alignment(self, score: float) -> float:
        """Convert internal ESG score (0-100) to alignment estimate.

        Uses the configured score-to-alignment mapping.

        Args:
            score: Internal ESG score (0-100).

        Returns:
            Estimated alignment percentage.
        """
        for range_str, alignment in self.config.internal_esg_score_to_alignment.items():
            parts = range_str.split("-")
            if len(parts) == 2:
                low = float(parts[0])
                high = float(parts[1])
                if low <= score <= high:
                    return alignment
        return 0.0

    # ------------------------------------------------------------------
    # Internal: Data Coverage
    # ------------------------------------------------------------------

    def _build_data_coverage(
        self,
        exposures: List[BankingBookData],
        estimations: List[EstimationMethodology],
    ) -> DataCoverageReport:
        """Build data coverage report.

        Args:
            exposures: All exposures.
            estimations: All estimation results.

        Returns:
            DataCoverageReport.
        """
        total_amt = sum(e.carrying_amount for e in exposures)
        est_map = {e.exposure_id: e for e in estimations}

        csrd_count = 0
        csrd_amt = 0.0
        estimated_count = 0
        estimated_amt = 0.0
        no_data_count = 0
        no_data_amt = 0.0
        method_breakdown: Dict[str, int] = defaultdict(int)

        for exp in exposures:
            est = est_map.get(exp.exposure_id)
            if not est:
                no_data_count += 1
                no_data_amt += exp.carrying_amount
                continue

            method_breakdown[est.estimation_type.value] += 1

            if est.estimation_type == EstimationType.CSRD_REPORTED:
                csrd_count += 1
                csrd_amt += exp.carrying_amount
            elif est.estimation_type == EstimationType.CONSERVATIVE_DEFAULT and est.estimated_alignment_pct == 0:
                no_data_count += 1
                no_data_amt += exp.carrying_amount
            else:
                estimated_count += 1
                estimated_amt += exp.carrying_amount

        return DataCoverageReport(
            total_exposures=len(exposures),
            total_amount=_round_val(total_amt, 2),
            csrd_reported_count=csrd_count,
            csrd_reported_amount=_round_val(csrd_amt, 2),
            csrd_reported_pct=_round_val(_safe_pct(csrd_amt, total_amt), 2),
            estimated_count=estimated_count,
            estimated_amount=_round_val(estimated_amt, 2),
            estimated_pct=_round_val(_safe_pct(estimated_amt, total_amt), 2),
            no_data_count=no_data_count,
            no_data_amount=_round_val(no_data_amt, 2),
            no_data_pct=_round_val(_safe_pct(no_data_amt, total_amt), 2),
            estimation_method_breakdown=dict(method_breakdown),
        )

    # ------------------------------------------------------------------
    # Internal: Confidence Score
    # ------------------------------------------------------------------

    def _compute_weighted_confidence(
        self,
        exposures: List[BankingBookData],
        estimations: List[EstimationMethodology],
    ) -> float:
        """Compute exposure-weighted average confidence score.

        Args:
            exposures: All exposures.
            estimations: All estimation results.

        Returns:
            Weighted confidence score (0.0-1.0).
        """
        total_amt = sum(e.carrying_amount for e in exposures)
        if total_amt <= 0:
            return 0.0

        est_map = {e.exposure_id: e for e in estimations}
        weighted_sum = 0.0

        for exp in exposures:
            est = est_map.get(exp.exposure_id)
            if est:
                weight = exp.carrying_amount / total_amt
                weighted_sum += weight * est.confidence

        return max(0.0, min(1.0, weighted_sum))

    # ------------------------------------------------------------------
    # Internal: GAR Reconciliation
    # ------------------------------------------------------------------

    def _reconcile_with_gar(
        self,
        btar_turnover_pct: float,
        total_bb: float,
        total_aligned: float,
    ) -> BTARvsGARReconciliation:
        """Reconcile BTAR with GAR.

        The BTAR should typically be close to or slightly above
        the GAR, since it includes estimated alignment for exposures
        the GAR excludes.

        Args:
            btar_turnover_pct: BTAR turnover percentage.
            total_bb: Total banking book.
            total_aligned: Total aligned amount.

        Returns:
            BTARvsGARReconciliation.
        """
        gar = self.config.gar_turnover_pct
        gap = btar_turnover_pct - gar

        gar_covered = self.config.gar_covered_amount
        additional_aligned = total_aligned - (gar_covered * (gar / 100.0)) if gar_covered > 0 else total_aligned

        if btar_turnover_pct >= gar:
            status = "PASS"
            explanation = (
                f"BTAR ({btar_turnover_pct:.2f}%) >= GAR ({gar:.2f}%). "
                f"Gap of {gap:.2f}pp represents estimated alignment "
                f"of non-GAR exposures."
            )
        else:
            status = "WARNING"
            explanation = (
                f"BTAR ({btar_turnover_pct:.2f}%) < GAR ({gar:.2f}%). "
                f"This may indicate that non-GAR exposures are dragging "
                f"the overall alignment below the GAR level. Review "
                f"estimation methodologies."
            )

        result = BTARvsGARReconciliation(
            gar_turnover_pct=_round_val(gar, 4),
            btar_turnover_pct=_round_val(btar_turnover_pct, 4),
            gap_turnover_pct=_round_val(gap, 4),
            gar_covered_amount=_round_val(gar_covered, 2),
            btar_total_amount=_round_val(total_bb, 2),
            additional_aligned_amount=_round_val(max(0, additional_aligned), 2),
            reconciliation_status=status,
            explanation=explanation,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Category Breakdown
    # ------------------------------------------------------------------

    def _build_category_breakdown(
        self,
        exposures: List[BankingBookData],
        estimations: List[EstimationMethodology],
        aligned_amounts: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build breakdown by exposure category.

        Args:
            exposures: All exposures.
            estimations: All estimation results.
            aligned_amounts: Turnover-aligned amounts by exposure ID.

        Returns:
            List of category breakdown dicts.
        """
        total_bb = sum(e.carrying_amount for e in exposures)
        est_map = {e.exposure_id: e for e in estimations}

        groups: Dict[str, List[BankingBookData]] = defaultdict(list)
        for e in exposures:
            groups[e.exposure_category.value].append(e)

        breakdowns: List[Dict[str, Any]] = []
        for cat, group in sorted(groups.items()):
            cat_amount = sum(e.carrying_amount for e in group)
            cat_aligned = sum(aligned_amounts.get(e.exposure_id, 0.0) for e in group)

            # Average confidence
            confidences = [
                est_map[e.exposure_id].confidence
                for e in group
                if e.exposure_id in est_map
            ]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            breakdowns.append({
                "category": cat,
                "exposure_count": len(group),
                "total_amount": _round_val(cat_amount, 2),
                "aligned_amount": _round_val(cat_aligned, 2),
                "alignment_pct": _round_val(_safe_pct(cat_aligned, cat_amount), 4),
                "weight_in_bb_pct": _round_val(_safe_pct(cat_amount, total_bb), 4),
                "avg_confidence": _round_val(avg_conf, 4),
            })

        return breakdowns

    # ------------------------------------------------------------------
    # Internal: Methodology Notes
    # ------------------------------------------------------------------

    def _generate_methodology_notes(
        self,
        exposures: List[BankingBookData],
        estimations: List[EstimationMethodology],
        coverage: DataCoverageReport,
    ) -> List[str]:
        """Generate methodology disclosure notes.

        Args:
            exposures: All exposures.
            estimations: All estimation results.
            coverage: Data coverage report.

        Returns:
            List of methodology note strings.
        """
        notes: List[str] = [
            "Methodology: EBA ITS on Pillar 3 ESG Disclosures (BTAR extension)",
            f"Reporting year: {self.config.reporting_year}",
            f"Total banking book exposures: {len(exposures)}",
            f"CSRD-reported: {coverage.csrd_reported_count} ({coverage.csrd_reported_pct:.1f}%)",
            f"Estimated: {coverage.estimated_count} ({coverage.estimated_pct:.1f}%)",
            f"No data: {coverage.no_data_count} ({coverage.no_data_pct:.1f}%)",
        ]

        # Estimation method distribution
        method_counts: Dict[str, int] = defaultdict(int)
        for est in estimations:
            method_counts[est.estimation_type.value] += 1
        for method, count in sorted(method_counts.items()):
            notes.append(f"Estimation method {method}: {count} exposure(s)")

        # Category distribution
        cat_counts: Dict[str, int] = defaultdict(int)
        for exp in exposures:
            cat_counts[exp.exposure_category.value] += 1
        for cat, count in sorted(cat_counts.items()):
            notes.append(f"Exposure category {cat}: {count}")

        # GAR reconciliation
        if self.config.gar_turnover_pct > 0:
            notes.append(
                f"GAR reference: {self.config.gar_turnover_pct:.2f}% "
                f"(covered: {self.config.gar_covered_amount:,.0f} EUR)"
            )

        return notes
