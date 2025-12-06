"""
GL-012: Carbon Offset Verification Agent

This package provides the Carbon Offset Verification Agent for validating
carbon credits/offsets against major registries and scoring quality using
ICVCM Core Carbon Principles.

Features:
- 6 major carbon registries (Verra VCS, Gold Standard, ACR, CAR, Plan Vivo, Puro.earth)
- ICVCM Core Carbon Principles quality scoring (5 dimensions with weights):
    - Additionality (30%): Would project happen without credits?
    - Permanence (25%): Risk of reversal
    - MRV (20%): Measurement, Reporting, Verification quality
    - Co-benefits (15%): SDG alignment, community impact
    - Governance (10%): Registry standards, third-party verification
- Project existence and retirement status verification
- Credit vintage validation
- Double counting prevention (corresponding adjustments)
- Buffer pool adequacy assessment
- Complete SHA-256 provenance tracking

Example:
    >>> from gl_012_carbon_offset import (
    ...     CarbonOffsetAgent,
    ...     CarbonOffsetInput,
    ...     CarbonOffsetOutput,
    ...     CarbonRegistry,
    ...     CarbonCredit,
    ... )
    >>> agent = CarbonOffsetAgent()
    >>> result = agent.run(CarbonOffsetInput(
    ...     project_id="VCS-1234",
    ...     registry=CarbonRegistry.VERRA_VCS,
    ...     credits=[
    ...         CarbonCredit(
    ...             serial_number="VCS-1234-2023-001",
    ...             vintage_year=2023,
    ...             quantity_tco2e=100.0
    ...         )
    ...     ]
    ... ))
    >>> print(f"Status: {result.verification_status}")
    >>> print(f"Quality: {result.quality_score}/100 ({result.quality_rating})")
"""

from .agent import (
    # Main agent
    CarbonOffsetAgent,
    # Input/Output models
    CarbonOffsetInput,
    CarbonOffsetOutput,
    # Enumerations
    CarbonRegistry,
    ProjectType,
    VerificationStatus,
    RiskLevel,
    RetirementStatus,
    CorrespondingAdjustmentStatus,
    # Supporting models
    CarbonCredit,
    ProjectDetails,
    ICVCMScoreBreakdown,
    VerificationCheck,
    CreditVerificationResult,
    RiskAssessment,
    # Data constants
    REGISTRY_STANDARDS,
    PROJECT_TYPE_PROFILES,
    PACK_SPEC,
)

__all__ = [
    # Main agent
    "CarbonOffsetAgent",
    # Input/Output models
    "CarbonOffsetInput",
    "CarbonOffsetOutput",
    # Enumerations
    "CarbonRegistry",
    "ProjectType",
    "VerificationStatus",
    "RiskLevel",
    "RetirementStatus",
    "CorrespondingAdjustmentStatus",
    # Supporting models
    "CarbonCredit",
    "ProjectDetails",
    "ICVCMScoreBreakdown",
    "VerificationCheck",
    "CreditVerificationResult",
    "RiskAssessment",
    # Data constants
    "REGISTRY_STANDARDS",
    "PROJECT_TYPE_PROFILES",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "offsets/carbon_verification_v1"
