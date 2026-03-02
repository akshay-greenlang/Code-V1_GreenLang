# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Certificate Engine - Data Models v1.1

Pydantic models, enums, and constants for CBAM certificate obligation
calculation, EU ETS price tracking, free allocation phase-out, and
carbon price deduction verification.

Per EU CBAM Regulation 2023/956 (Definitive Period Jan 2026+):
  - Article 21: CBAM certificates (1 certificate = 1 tCO2e)
  - Article 22: Certificate price (weekly EU ETS auction average)
  - Article 23: Quarterly holding requirement (50% of estimated annual)
  - Article 24: Surrender of certificates (31 May each year)
  - Article 26: Carbon price paid in country of origin (deduction)
  - Article 31: Free allocation phase-out (2026-2034, declining to zero)

All monetary and emissions values use Decimal with ROUND_HALF_UP to prevent
floating-point drift in regulatory/financial calculations.

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class CertificateStatus(str, Enum):
    """
    CBAM certificate lifecycle status.
    Per Regulation 2023/956 Articles 21-24.
    """

    REQUIRED = "required"
    PURCHASED = "purchased"
    SURRENDERED = "surrendered"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether this status represents a terminal state."""
        return self in {CertificateStatus.SURRENDERED, CertificateStatus.CANCELLED}


class PriceSource(str, Enum):
    """
    Source of EU ETS price data.
    Per Regulation 2023/956 Article 22(1).
    """

    WEEKLY_AUCTION = "weekly_auction"
    QUARTERLY_AVERAGE = "quarterly_average"
    MANUAL = "manual"

    @property
    def description(self) -> str:
        """Human-readable source description."""
        descriptions = {
            PriceSource.WEEKLY_AUCTION: "Weekly average of EU ETS allowance auction closing prices",
            PriceSource.QUARTERLY_AVERAGE: "Volume-weighted quarterly average auction price",
            PriceSource.MANUAL: "Manually entered price (admin override)",
        }
        return descriptions[self]


class AllocationPhase(str, Enum):
    """
    Free allocation phase-out stage.
    Per Regulation 2023/956 Article 31.
    """

    PHASE1_100PCT = "phase1_100pct"
    PHASE2_97_5PCT = "phase2_97_5pct"
    PHASE3_95PCT = "phase3_95pct"
    PHASE4_90PCT = "phase4_90pct"
    PHASE5_77_5PCT = "phase5_77_5pct"
    PHASE6_51_5PCT = "phase6_51_5pct"
    PHASE7_39PCT = "phase7_39pct"
    PHASE8_26_5PCT = "phase8_26_5pct"
    PHASE9_14PCT = "phase9_14pct"
    PHASE_OUT_0PCT = "phase_out_0pct"

    @property
    def label(self) -> str:
        """Human-readable phase label."""
        labels = {
            AllocationPhase.PHASE1_100PCT: "Pre-CBAM (100% free allocation)",
            AllocationPhase.PHASE2_97_5PCT: "2026 (97.5% free allocation)",
            AllocationPhase.PHASE3_95PCT: "2027 (95% free allocation)",
            AllocationPhase.PHASE4_90PCT: "2028 (90% free allocation)",
            AllocationPhase.PHASE5_77_5PCT: "2029 (77.5% free allocation)",
            AllocationPhase.PHASE6_51_5PCT: "2030 (51.5% free allocation)",
            AllocationPhase.PHASE7_39PCT: "2031 (39% free allocation)",
            AllocationPhase.PHASE8_26_5PCT: "2032 (26.5% free allocation)",
            AllocationPhase.PHASE9_14PCT: "2033 (14% free allocation)",
            AllocationPhase.PHASE_OUT_0PCT: "2034+ (0% free allocation)",
        }
        return labels[self]


class CarbonPricingScheme(str, Enum):
    """
    Types of carbon pricing schemes recognized for CBAM deductions.
    Per Regulation 2023/956 Article 26.
    """

    ETS = "ets"
    CARBON_TAX = "carbon_tax"
    HYBRID = "hybrid"
    NONE = "none"

    @property
    def description(self) -> str:
        """Human-readable scheme description."""
        descriptions = {
            CarbonPricingScheme.ETS: "Emissions Trading System (cap-and-trade)",
            CarbonPricingScheme.CARBON_TAX: "Direct carbon tax/levy",
            CarbonPricingScheme.HYBRID: "Combined ETS and carbon tax",
            CarbonPricingScheme.NONE: "No carbon pricing mechanism in place",
        }
        return descriptions[self]


class DeductionStatus(str, Enum):
    """
    Carbon price deduction verification status.
    Per Regulation 2023/956 Article 26(2).
    """

    PENDING = "pending"
    VERIFIED = "verified"
    APPROVED = "approved"
    REJECTED = "rejected"

    @property
    def is_eligible(self) -> bool:
        """Whether the deduction can be applied."""
        return self in {DeductionStatus.VERIFIED, DeductionStatus.APPROVED}


# ============================================================================
# CONSTANTS
# ============================================================================

# One CBAM certificate represents 1 tonne of CO2 equivalent
CERTIFICATE_UNIT_TCO2E: Decimal = Decimal("1")

# Quarterly holding requirement: 50% of estimated annual obligation
# Per Regulation 2023/956 Article 23
QUARTERLY_HOLDING_PCT: Decimal = Decimal("50")

# Free allocation phase-out schedule: year -> percentage of free allocation remaining
# Per Regulation 2023/956 Article 31 and Annex V
PHASE_OUT_SCHEDULE: Dict[int, Decimal] = {
    2025: Decimal("100"),     # Pre-CBAM definitive period
    2026: Decimal("97.5"),    # First year of definitive period
    2027: Decimal("95.0"),
    2028: Decimal("90.0"),
    2029: Decimal("77.5"),
    2030: Decimal("51.5"),
    2031: Decimal("39.0"),
    2032: Decimal("26.5"),
    2033: Decimal("14.0"),
    2034: Decimal("0"),       # Full phase-out complete
}

# Product-specific benchmarks (tCO2e per tonne of product)
# Per EU ETS benchmarking decisions, mapped to CBAM CN codes
# These are the EU ETS free allocation benchmark values
PRODUCT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "25231000": {
        "product_name": "Cement clinker",
        "benchmark_value": Decimal("0.766"),
        "unit": "tCO2e/t clinker",
        "source": "EU ETS Benchmark Decision 2021/927, Annex I",
    },
    "72031000": {
        "product_name": "Hot metal (pig iron)",
        "benchmark_value": Decimal("1.328"),
        "unit": "tCO2e/t hot metal",
        "source": "EU ETS Benchmark Decision 2021/927, Annex I",
    },
    "76011000": {
        "product_name": "Unwrought aluminium",
        "benchmark_value": Decimal("1.514"),
        "unit": "tCO2e/t aluminium",
        "source": "EU ETS Benchmark Decision 2021/927, Annex I",
    },
    "28141000": {
        "product_name": "Ammonia (anhydrous)",
        "benchmark_value": Decimal("1.619"),
        "unit": "tCO2e/t ammonia",
        "source": "EU ETS Benchmark Decision 2021/927, Annex I",
    },
    "28041000": {
        "product_name": "Hydrogen",
        "benchmark_value": Decimal("8.850"),
        "unit": "tCO2e/t hydrogen",
        "source": "EU ETS Benchmark Decision 2021/927, Annex I",
    },
    "72081000": {
        "product_name": "Hot-rolled steel products",
        "benchmark_value": Decimal("1.328"),
        "unit": "tCO2e/t product",
        "source": "EU ETS Benchmark Decision 2021/927 (hot metal benchmark)",
    },
    "72044100": {
        "product_name": "Ferrous waste and scrap (EAF steel)",
        "benchmark_value": Decimal("0.283"),
        "unit": "tCO2e/t product",
        "source": "EU ETS Benchmark Decision 2021/927 (EAF carbon steel)",
    },
    "31021000": {
        "product_name": "Urea",
        "benchmark_value": Decimal("1.619"),
        "unit": "tCO2e/t product",
        "source": "EU ETS Benchmark Decision 2021/927 (ammonia benchmark)",
    },
}

# EU ETS price range boundaries for sanity checks (EUR per tCO2e)
ETS_PRICE_MIN: Decimal = Decimal("20")
ETS_PRICE_MAX: Decimal = Decimal("200")

# Maximum deduction cannot exceed gross obligation
MAX_DEDUCTION_RATIO: Decimal = Decimal("1.0")

# ECB exchange rates snapshot (illustrative, updated periodically)
# In production, these would come from ECB API or database
ECB_EXCHANGE_RATES: Dict[str, Decimal] = {
    "EUR": Decimal("1.000000"),
    "USD": Decimal("0.920000"),
    "GBP": Decimal("1.160000"),
    "CHF": Decimal("1.050000"),
    "CNY": Decimal("0.127000"),
    "CAD": Decimal("0.680000"),
    "JPY": Decimal("0.006100"),
    "KRW": Decimal("0.000690"),
    "TRY": Decimal("0.028000"),
    "INR": Decimal("0.011000"),
    "BRL": Decimal("0.183000"),
    "ZAR": Decimal("0.051000"),
    "AUD": Decimal("0.600000"),
    "NOK": Decimal("0.086000"),
    "SEK": Decimal("0.087000"),
    "DKK": Decimal("0.134000"),
    "PLN": Decimal("0.230000"),
    "CZK": Decimal("0.040000"),
    "HUF": Decimal("0.002600"),
    "MXN": Decimal("0.055000"),
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ETSPrice(BaseModel):
    """
    EU ETS allowance price record.

    Per Regulation 2023/956 Article 22(1), CBAM certificate prices are
    derived from the weekly average of EU ETS allowance auction closing
    prices on the common auction platform.

    Attributes:
        date: Reference date for this price record.
        price_eur_per_tco2e: Price in EUR per tonne CO2 equivalent.
        source: How the price was obtained (auction/average/manual).
        volume_weighted: Whether the price is volume-weighted.
        period: Description of the averaging period (e.g., "2026-W05").
    """

    date: date = Field(
        ...,
        description="Reference date for this price record"
    )
    price_eur_per_tco2e: Decimal = Field(
        ...,
        ge=0,
        description="Price in EUR per tonne CO2 equivalent"
    )
    source: PriceSource = Field(
        ...,
        description="Price data source"
    )
    volume_weighted: bool = Field(
        default=False,
        description="Whether the price is volume-weighted"
    )
    period: str = Field(
        default="",
        description="Averaging period label (e.g., '2026-W05', '2026Q1')"
    )

    @field_validator("price_eur_per_tco2e")
    @classmethod
    def validate_price_range(cls, v: Decimal) -> Decimal:
        """Warn if price is outside expected range (but do not reject)."""
        if v < ETS_PRICE_MIN or v > ETS_PRICE_MAX:
            logger.warning(
                "ETS price EUR %.2f is outside expected range [%s-%s]",
                v, ETS_PRICE_MIN, ETS_PRICE_MAX,
            )
        return v

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "date": "2026-02-01",
                "price_eur_per_tco2e": "72.50",
                "source": "weekly_auction",
                "volume_weighted": True,
                "period": "2026-W05",
            }
        }


class FreeAllocationFactor(BaseModel):
    """
    Free allocation benchmark for a CBAM product.

    Per Regulation 2023/956 Article 31, free allocation to EU producers
    is phased out as CBAM is phased in. The benchmark value represents
    the EU ETS free allocation for the equivalent EU product.

    Attributes:
        product_benchmark: Product name for this benchmark.
        benchmark_value_tCO2e: Benchmark value in tCO2e per tonne of product.
        allocation_percentage: Free allocation percentage for the year.
        phase: Phase-out stage identifier.
        year: Reference year for this factor.
        cn_codes: List of CN codes this benchmark applies to.
    """

    product_benchmark: str = Field(
        ...,
        min_length=1,
        description="Product benchmark name"
    )
    benchmark_value_tCO2e: Decimal = Field(
        ...,
        ge=0,
        description="Benchmark value in tCO2e per tonne of product"
    )
    allocation_percentage: Decimal = Field(
        ...,
        ge=0,
        le=100,
        description="Free allocation percentage for the year"
    )
    phase: AllocationPhase = Field(
        ...,
        description="Phase-out stage identifier"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Reference year"
    )
    cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes this benchmark applies to"
    )

    @property
    def effective_allocation_tCO2e(self) -> Decimal:
        """Compute the effective free allocation per tonne of product."""
        pct = self.allocation_percentage / Decimal("100")
        return (self.benchmark_value_tCO2e * pct).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "product_benchmark": "Cement clinker",
                "benchmark_value_tCO2e": "0.766",
                "allocation_percentage": "97.5",
                "phase": "phase2_97_5pct",
                "year": 2026,
                "cn_codes": ["25231000", "25231010"],
            }
        }


class CarbonPriceDeduction(BaseModel):
    """
    Carbon price paid in country of origin, eligible for CBAM deduction.

    Per Regulation 2023/956 Article 26, importers may deduct carbon prices
    effectively paid in the country of origin for embedded emissions. The
    deduction requires verification and evidence documentation.

    Attributes:
        deduction_id: Unique deduction identifier.
        importer_id: Importer EORI number or internal identifier.
        installation_id: Installation identifier in country of origin.
        country: ISO 3166-1 alpha-2 country code.
        pricing_scheme: Type of carbon pricing mechanism.
        carbon_price_paid_eur: Amount paid, converted to EUR.
        carbon_price_paid_local: Amount paid in local currency.
        exchange_rate: EUR exchange rate used for conversion.
        currency: Local currency code (ISO 4217).
        tonnes_covered: Tonnes of CO2e covered by this payment.
        deduction_per_tonne_eur: Effective deduction per tCO2e in EUR.
        evidence_docs: List of evidence document references.
        verification_status: Current verification status.
        verified_by: Verifier identifier (if verified).
        verified_at: Verification timestamp (if verified).
        year: Reference year for this deduction.
        provenance_hash: SHA-256 hash for audit trail.
    """

    deduction_id: str = Field(
        ...,
        min_length=5,
        description="Unique deduction identifier"
    )
    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer EORI or internal identifier"
    )
    installation_id: str = Field(
        ...,
        min_length=1,
        description="Installation identifier in country of origin"
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )
    pricing_scheme: CarbonPricingScheme = Field(
        ...,
        description="Type of carbon pricing mechanism"
    )
    carbon_price_paid_eur: Decimal = Field(
        ...,
        ge=0,
        description="Amount paid converted to EUR"
    )
    carbon_price_paid_local: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Amount paid in local currency"
    )
    exchange_rate: Decimal = Field(
        default=Decimal("1.0"),
        gt=0,
        description="EUR exchange rate used for conversion"
    )
    currency: str = Field(
        default="EUR",
        min_length=3,
        max_length=3,
        description="Local currency code (ISO 4217)"
    )
    tonnes_covered: Decimal = Field(
        ...,
        gt=0,
        description="Tonnes of CO2e covered by this payment"
    )
    deduction_per_tonne_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Effective deduction per tCO2e in EUR"
    )
    evidence_docs: List[str] = Field(
        default_factory=list,
        description="List of evidence document references"
    )
    verification_status: DeductionStatus = Field(
        default=DeductionStatus.PENDING,
        description="Current verification status"
    )
    verified_by: Optional[str] = Field(
        default=None,
        description="Verifier identifier"
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="Verification timestamp (UTC)"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Reference year for this deduction"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def compute_deduction_per_tonne(self) -> "CarbonPriceDeduction":
        """Auto-compute deduction per tonne if not provided."""
        if (
            self.deduction_per_tonne_eur == Decimal("0")
            and self.tonnes_covered > 0
            and self.carbon_price_paid_eur > 0
        ):
            computed = (
                self.carbon_price_paid_eur / self.tonnes_covered
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            object.__setattr__(self, "deduction_per_tonne_eur", computed)
        return self

    @model_validator(mode="after")
    def validate_verified_fields(self) -> "CarbonPriceDeduction":
        """Validate that verified_by and verified_at are set when status is verified+."""
        if self.verification_status.is_eligible and self.verified_by is None:
            raise ValueError(
                f"verified_by must be set when status is "
                f"{self.verification_status.value}"
            )
        return self

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of deduction content for audit trail."""
        payload = {
            "deduction_id": self.deduction_id,
            "importer_id": self.importer_id,
            "installation_id": self.installation_id,
            "country": self.country,
            "pricing_scheme": self.pricing_scheme.value,
            "carbon_price_paid_eur": str(self.carbon_price_paid_eur),
            "tonnes_covered": str(self.tonnes_covered),
            "deduction_per_tonne_eur": str(self.deduction_per_tonne_eur),
            "year": self.year,
        }
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "deduction_id": "CPD-2026-NL123-001",
                "importer_id": "NL123456789012",
                "installation_id": "TR-INSTALL-001",
                "country": "TR",
                "pricing_scheme": "ets",
                "carbon_price_paid_eur": "15000.00",
                "carbon_price_paid_local": "500000.00",
                "exchange_rate": "0.030",
                "currency": "TRY",
                "tonnes_covered": "1000",
                "deduction_per_tonne_eur": "15.00",
                "evidence_docs": ["tax-receipt-2026-001.pdf"],
                "verification_status": "pending",
                "year": 2026,
            }
        }


class CertificateObligation(BaseModel):
    """
    Complete CBAM certificate obligation for an importer-product-year combination.

    Per Regulation 2023/956 Articles 21-24, the net obligation is computed as:
        net = gross - free_allocation_deduction - carbon_price_deduction
        cost = net x ETS_price

    All financial values use Decimal with ROUND_HALF_UP.

    Attributes:
        obligation_id: Unique obligation identifier.
        importer_id: Importer EORI or internal identifier.
        year: Obligation year.
        cn_code: Combined Nomenclature code for the goods.
        quantity_mt: Total imported quantity in metric tonnes.
        embedded_emissions_tCO2e: Total embedded emissions.
        gross_certificates_required: Gross certificate count (= embedded emissions).
        free_allocation_deduction: Deduction from EU free allocation.
        carbon_price_deduction_eur: Deduction from carbon price paid in origin.
        net_certificates_required: Net certificates after all deductions.
        certificate_cost_eur: Estimated cost at current ETS price.
        ets_price_used: ETS price used for cost calculation.
        calculation_date: Date of this calculation.
        provenance_hash: SHA-256 hash for audit trail.
    """

    obligation_id: str = Field(
        ...,
        min_length=5,
        description="Unique obligation identifier"
    )
    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer EORI or internal identifier"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Obligation year"
    )
    cn_code: str = Field(
        default="",
        description="CN code (empty for aggregate obligation)"
    )
    country_of_origin: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code (empty for aggregate)"
    )
    quantity_mt: Decimal = Field(
        ...,
        ge=0,
        description="Total imported quantity in metric tonnes"
    )
    embedded_emissions_tCO2e: Decimal = Field(
        ...,
        ge=0,
        description="Total embedded emissions in tCO2e"
    )
    gross_certificates_required: Decimal = Field(
        ...,
        ge=0,
        description="Gross certificates required (= embedded emissions)"
    )
    free_allocation_deduction: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Deduction from EU free allocation phase-out"
    )
    carbon_price_deduction_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Deduction value from carbon price in country of origin (EUR)"
    )
    carbon_price_deduction_tCO2e: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Carbon price deduction in equivalent tCO2e certificates"
    )
    net_certificates_required: Decimal = Field(
        ...,
        ge=0,
        description="Net certificates after all deductions"
    )
    certificate_cost_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Estimated cost at current ETS price"
    )
    ets_price_used: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="ETS price used for cost calculation (EUR/tCO2e)"
    )
    calculation_date: date = Field(
        default_factory=date.today,
        description="Date of this calculation"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def validate_deductions_within_gross(self) -> "CertificateObligation":
        """Validate that total deductions do not exceed gross certificates."""
        total_deductions = (
            self.free_allocation_deduction + self.carbon_price_deduction_tCO2e
        )
        if total_deductions > self.gross_certificates_required:
            logger.warning(
                "Total deductions (%.3f) exceed gross certificates (%.3f). "
                "Net will be clamped to zero.",
                total_deductions,
                self.gross_certificates_required,
            )
        return self

    @model_validator(mode="after")
    def validate_net_consistency(self) -> "CertificateObligation":
        """Validate net = max(0, gross - deductions)."""
        total_deductions = (
            self.free_allocation_deduction + self.carbon_price_deduction_tCO2e
        )
        expected_net = max(
            Decimal("0"),
            self.gross_certificates_required - total_deductions,
        )
        tolerance = Decimal("0.01")
        if abs(self.net_certificates_required - expected_net) > tolerance:
            raise ValueError(
                f"net_certificates_required ({self.net_certificates_required}) "
                f"does not match expected ({expected_net}). "
                f"gross={self.gross_certificates_required}, "
                f"free_alloc={self.free_allocation_deduction}, "
                f"carbon_deduction={self.carbon_price_deduction_tCO2e}"
            )
        return self

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of obligation content for audit trail."""
        payload = {
            "obligation_id": self.obligation_id,
            "importer_id": self.importer_id,
            "year": self.year,
            "cn_code": self.cn_code,
            "quantity_mt": str(self.quantity_mt),
            "embedded_emissions_tCO2e": str(self.embedded_emissions_tCO2e),
            "gross_certificates_required": str(self.gross_certificates_required),
            "free_allocation_deduction": str(self.free_allocation_deduction),
            "carbon_price_deduction_tCO2e": str(self.carbon_price_deduction_tCO2e),
            "net_certificates_required": str(self.net_certificates_required),
            "ets_price_used": str(self.ets_price_used),
        }
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "obligation_id": "OBL-2026-NL123-72031000",
                "importer_id": "NL123456789012",
                "year": 2026,
                "cn_code": "72031000",
                "quantity_mt": "1500.000",
                "embedded_emissions_tCO2e": "3000.000",
                "gross_certificates_required": "3000.000",
                "free_allocation_deduction": "97.050",
                "carbon_price_deduction_tCO2e": "200.000",
                "net_certificates_required": "2702.950",
                "certificate_cost_eur": "195963.88",
                "ets_price_used": "72.50",
            }
        }


class QuarterlyHolding(BaseModel):
    """
    Quarterly certificate holding compliance check.

    Per Regulation 2023/956 Article 23, importers must hold certificates
    covering at least 50% of their estimated annual obligation at the
    end of each quarter.

    Attributes:
        quarter: Quarter identifier (Q1-Q4).
        year: Reference year.
        importer_id: Importer identifier.
        holding_required: Number of certificates required (50% of annual estimate).
        certificates_held: Number of certificates currently held.
        compliant: Whether the holding requirement is met.
        shortfall: Number of certificates short (0 if compliant).
    """

    quarter: str = Field(
        ...,
        description="Quarter identifier (Q1/Q2/Q3/Q4)"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Reference year"
    )
    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer identifier"
    )
    holding_required: Decimal = Field(
        ...,
        ge=0,
        description="Certificates required (50% of estimated annual obligation)"
    )
    certificates_held: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Certificates currently held"
    )
    compliant: bool = Field(
        default=False,
        description="Whether the holding requirement is met"
    )
    shortfall: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Certificate shortfall (0 if compliant)"
    )

    @model_validator(mode="after")
    def compute_compliance(self) -> "QuarterlyHolding":
        """Auto-compute compliance status and shortfall."""
        is_compliant = self.certificates_held >= self.holding_required
        gap = max(Decimal("0"), self.holding_required - self.certificates_held)
        object.__setattr__(self, "compliant", is_compliant)
        object.__setattr__(self, "shortfall", gap)
        return self

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "quarter": "Q1",
                "year": 2026,
                "importer_id": "NL123456789012",
                "holding_required": "1500.000",
                "certificates_held": "1600.000",
                "compliant": True,
                "shortfall": "0",
            }
        }


class CertificateSummary(BaseModel):
    """
    Annual certificate obligation summary for an importer.

    Aggregates all obligations across CN codes, applies deductions,
    and computes total cost and quarterly holding requirements.

    Attributes:
        importer_id: Importer identifier.
        year: Obligation year.
        total_gross: Total gross certificates required.
        total_free_allocation: Total free allocation deduction.
        total_carbon_deductions: Total carbon price deductions (tCO2e).
        total_net: Total net certificates required.
        total_cost_eur: Total estimated cost in EUR.
        quarterly_holdings_required: Quarterly holding requirement (50%).
        certificates_held: Certificates currently held.
        shortfall: Total shortfall (0 if sufficient).
        obligations_by_cn: Breakdown by CN code.
        ets_price_used: ETS price used for cost calculation.
    """

    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer identifier"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Obligation year"
    )
    total_gross: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total gross certificates required"
    )
    total_free_allocation: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total free allocation deduction"
    )
    total_carbon_deductions: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total carbon price deductions (tCO2e)"
    )
    total_net: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total net certificates required"
    )
    total_cost_eur: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total estimated cost in EUR"
    )
    quarterly_holdings_required: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Quarterly holding requirement (50% of annual)"
    )
    certificates_held: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Certificates currently held"
    )
    shortfall: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total shortfall"
    )
    obligations_by_cn: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Breakdown of obligations by CN code"
    )
    ets_price_used: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="ETS price used for cost calculation (EUR/tCO2e)"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of summary for audit trail"
    )

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of summary content for audit trail."""
        payload = {
            "importer_id": self.importer_id,
            "year": self.year,
            "total_gross": str(self.total_gross),
            "total_free_allocation": str(self.total_free_allocation),
            "total_carbon_deductions": str(self.total_carbon_deductions),
            "total_net": str(self.total_net),
            "total_cost_eur": str(self.total_cost_eur),
        }
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "importer_id": "NL123456789012",
                "year": 2026,
                "total_gross": "5000.000",
                "total_free_allocation": "200.000",
                "total_carbon_deductions": "300.000",
                "total_net": "4500.000",
                "total_cost_eur": "326250.00",
                "quarterly_holdings_required": "2250.000",
                "certificates_held": "2500.000",
                "shortfall": "0",
                "ets_price_used": "72.50",
            }
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_sha256(data: Union[str, bytes, dict]) -> str:
    """
    Compute SHA-256 hash of arbitrary data.

    This is a ZERO-HALLUCINATION deterministic computation.

    Args:
        data: String, bytes, or dict to hash. Dicts are JSON-serialized
              with sorted keys for deterministic output.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def quantize_decimal(
    value: Union[Decimal, float, str],
    places: int = 3,
) -> Decimal:
    """
    Quantize a value to the specified decimal places using ROUND_HALF_UP.

    Standard rounding method for CBAM regulatory and financial calculations.

    Args:
        value: The value to quantize.
        places: Number of decimal places (default: 3).

    Returns:
        Quantized Decimal value.
    """
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    quantizer = Decimal(10) ** -places
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)
