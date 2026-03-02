# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Certificate Engine v1.1

CBAM certificate requirement calculation, EU ETS price integration,
free allocation adjustments, and carbon price deduction tracking.
Per EU CBAM Regulation 2023/956 Definitive Period (Jan 2026+).

Architecture:
    CertificateCalculatorEngine  - Gross/net obligation, quarterly holdings,
                                   cost projections, CN-code/country breakdowns
    ETSPriceService              - EU ETS auction price retrieval, trend analysis,
                                   volume-weighted averages, manual entry
    FreeAllocationEngine         - Product benchmarks, phase-out schedule (2026-2034),
                                   declining deduction calculations
    CarbonPriceDeductionEngine   - Third-country carbon price registration,
                                   ECB FX conversion, verification workflow

Design Principles:
    - ZERO HALLUCINATION: All financial and emissions calculations use
      deterministic Python Decimal arithmetic with ROUND_HALF_UP
    - Thread safety: RLock-based singletons for calculator and ETS service
    - Provenance: SHA-256 hashing on every obligation and deduction record
    - Regulatory alignment: EU CBAM Regulation 2023/956 Articles 21-27
    - Auditability: Full version history and verification status tracking

CBAM Certificate Key Concepts:
    - Each CBAM certificate = 1 tCO2e of embedded emissions
    - Gross certificates = quantity x embedded emissions per tonne
    - Free allocation deduction = EU ETS benchmark-based allocation (declining)
    - Carbon price deduction = carbon price already paid in country of origin
    - Net certificates = gross - free allocation - carbon price deductions
    - Certificate price = weekly EU ETS auction average
    - Quarterly holding requirement = 50% of estimated annual obligation
    - Phase-out schedule: free allocation declines from 97.5% (2026) to 0% (2034)

Usage:
    >>> from certificate_engine import (
    ...     CertificateCalculatorEngine,
    ...     ETSPriceService,
    ...     FreeAllocationEngine,
    ...     CarbonPriceDeductionEngine,
    ... )
    >>>
    >>> calculator = CertificateCalculatorEngine()
    >>> obligation = calculator.calculate_annual_obligation(
    ...     importer_id="NL123456789012",
    ...     year=2026,
    ...     shipments=[...],
    ... )
    >>> print(f"Net certificates: {obligation.net_certificates_required}")
    >>> print(f"Estimated cost: EUR {obligation.certificate_cost_eur}")
    >>>
    >>> ets = ETSPriceService()
    >>> price = ets.get_current_price()
    >>> print(f"Current ETS price: EUR {price.price_eur_per_tco2e}/tCO2e")
    >>>
    >>> free_alloc = FreeAllocationEngine()
    >>> schedule = free_alloc.get_phase_out_schedule()
    >>> print(f"2026 allocation: {schedule[2026]}%")

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

# Version metadata
__version__ = "1.1.0"
__author__ = "GreenLang CBAM Team"
__license__ = "Proprietary"

# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------

from .models import (
    # Enums
    CertificateStatus,
    PriceSource,
    AllocationPhase,
    CarbonPricingScheme,
    DeductionStatus,
    # Models
    CertificateObligation,
    ETSPrice,
    FreeAllocationFactor,
    CarbonPriceDeduction,
    CertificateSummary,
    QuarterlyHolding,
    # Constants
    CERTIFICATE_UNIT_TCO2E,
    QUARTERLY_HOLDING_PCT,
    PHASE_OUT_SCHEDULE,
    PRODUCT_BENCHMARKS,
    # Helpers
    compute_sha256,
    quantize_decimal,
)

from .certificate_calculator import CertificateCalculatorEngine
from .ets_price_service import ETSPriceService
from .free_allocation import FreeAllocationEngine
from .carbon_price_deduction import CarbonPriceDeductionEngine

# ---------------------------------------------------------------------------
# Module-level __all__ for explicit public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Enums
    "CertificateStatus",
    "PriceSource",
    "AllocationPhase",
    "CarbonPricingScheme",
    "DeductionStatus",
    # Models
    "CertificateObligation",
    "ETSPrice",
    "FreeAllocationFactor",
    "CarbonPriceDeduction",
    "CertificateSummary",
    "QuarterlyHolding",
    # Constants
    "CERTIFICATE_UNIT_TCO2E",
    "QUARTERLY_HOLDING_PCT",
    "PHASE_OUT_SCHEDULE",
    "PRODUCT_BENCHMARKS",
    # Helpers
    "compute_sha256",
    "quantize_decimal",
    # Engines
    "CertificateCalculatorEngine",
    "ETSPriceService",
    "FreeAllocationEngine",
    "CarbonPriceDeductionEngine",
]
