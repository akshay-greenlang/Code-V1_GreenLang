# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness - Computation Engines
==============================================

Seven specialized engines providing the computational backbone for
CBAM (Carbon Border Adjustment Mechanism) compliance:

    1. CBAMCalculationEngine       - Embedded emissions calculation per CBAM Annex III
    2. CertificateEngine           - Certificate obligation, free allocation, cost projection
    3. QuarterlyReportingEngine    - Quarterly CBAM report assembly and XML generation
    4. SupplierManagementEngine    - Supplier/installation registration and data workflow
    5. DeMinimisEngine             - De minimis threshold tracking (50 t/sector)
    6. VerificationEngine          - Accredited verifier management and findings workflow
    7. PolicyComplianceEngine      - 50+ CBAM compliance rules and scoring

All engines follow GreenLang zero-hallucination principles:
    - Deterministic arithmetic for all emission/financial calculations
    - SHA-256 provenance hashing on every result
    - Pydantic validation at input and output boundaries
    - No LLM involvement in any numeric calculation path

Each engine is self-contained (no cross-engine imports) and wraps the
existing GL-CBAM-APP modules to provide pack-level orchestration without
duplicating business logic.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from __future__ import annotations

# Engine 1: CBAM Calculation
from packs.eu_compliance.PACK_004_cbam_readiness.engines.cbam_calculation_engine import (
    CBAMCalculationEngine,
    CBAMGoodsCategory,
    CalculationMethod,
    EmissionInput,
    EmissionResult,
)

# Engine 2: Certificate Obligation
from packs.eu_compliance.PACK_004_cbam_readiness.engines.certificate_engine import (
    CarbonPriceDeduction,
    CertificateEngine,
    CertificateObligation,
    CostProjection,
    FreeAllocationSchedule,
    QuarterlyHolding,
)

# Engine 3: Quarterly Reporting
from packs.eu_compliance.PACK_004_cbam_readiness.engines.quarterly_reporting_engine import (
    GoodsEntry,
    QuarterlyPeriod,
    QuarterlyReport,
    QuarterlyReportingEngine,
)

# Engine 4: Supplier Management
from packs.eu_compliance.PACK_004_cbam_readiness.engines.supplier_management_engine import (
    EmissionSubmission,
    Installation,
    SupplierManagementEngine,
    SupplierProfile,
)

# Engine 5: De Minimis Threshold
from packs.eu_compliance.PACK_004_cbam_readiness.engines.deminimis_engine import (
    DeMinimisAssessment,
    DeMinimisEngine,
    SectorGroup,
    ThresholdStatus,
)

# Engine 6: Verification
from packs.eu_compliance.PACK_004_cbam_readiness.engines.verification_engine import (
    AccreditedVerifier,
    VerificationEngine,
    VerificationEngagement,
    VerificationFinding,
)

# Engine 7: Policy Compliance
from packs.eu_compliance.PACK_004_cbam_readiness.engines.policy_compliance_engine import (
    ComplianceAssessment,
    ComplianceCheckResult,
    ComplianceRule,
    PolicyComplianceEngine,
)

__all__: list[str] = [
    # Engine 1: CBAM Calculation
    "CBAMCalculationEngine",
    "CBAMGoodsCategory",
    "CalculationMethod",
    "EmissionInput",
    "EmissionResult",
    # Engine 2: Certificate Obligation
    "CertificateEngine",
    "CertificateObligation",
    "FreeAllocationSchedule",
    "CarbonPriceDeduction",
    "CostProjection",
    "QuarterlyHolding",
    # Engine 3: Quarterly Reporting
    "QuarterlyReportingEngine",
    "QuarterlyPeriod",
    "QuarterlyReport",
    "GoodsEntry",
    # Engine 4: Supplier Management
    "SupplierManagementEngine",
    "SupplierProfile",
    "Installation",
    "EmissionSubmission",
    # Engine 5: De Minimis Threshold
    "DeMinimisEngine",
    "SectorGroup",
    "ThresholdStatus",
    "DeMinimisAssessment",
    # Engine 6: Verification
    "VerificationEngine",
    "AccreditedVerifier",
    "VerificationEngagement",
    "VerificationFinding",
    # Engine 7: Policy Compliance
    "PolicyComplianceEngine",
    "ComplianceRule",
    "ComplianceCheckResult",
    "ComplianceAssessment",
]
