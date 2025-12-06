"""
GL-008: Green Claims Verification Agent

This package provides the Green Claims Verification Agent for validating
environmental claims against the EU Green Claims Directive and detecting
potential greenwashing patterns.

Features:
- 16 claim types (carbon_neutral, net_zero, eco_friendly, etc.)
- 5-dimension substantiation scoring
- 7 greenwashing red flag detection patterns
- EU Green Claims Directive compliance validation
- Complete SHA-256 provenance tracking

Example:
    >>> from gl_008_green_claims import (
    ...     GreenClaimsAgent,
    ...     GreenClaimsInput,
    ...     GreenClaimsOutput,
    ...     ClaimType,
    ...     EvidenceItem,
    ... )
    >>> agent = GreenClaimsAgent()
    >>> result = agent.run(GreenClaimsInput(
    ...     claim_text="Our product is carbon neutral",
    ...     claim_type=ClaimType.CARBON_NEUTRAL,
    ...     evidence_items=[
    ...         EvidenceItem(
    ...             evidence_type="certification",
    ...             description="ISO 14064 certification",
    ...             is_third_party=True
    ...         )
    ...     ]
    ... ))
    >>> print(f"Valid: {result.claim_valid}, Score: {result.substantiation_score}")
"""

from .agent import (
    # Main agent
    GreenClaimsAgent,
    # Input/Output models
    GreenClaimsInput,
    GreenClaimsOutput,
    # Enumerations
    ClaimType,
    GreenwashingRedFlag,
    ValidationStatus,
    SubstantiationLevel,
    ComplianceFramework,
    # Supporting models
    EvidenceItem,
    ClaimScope,
    SubstantiationScoreBreakdown,
    GreenwashingDetection,
    FrameworkComplianceResult,
    ComplianceReport,
    ClaimRequirements,
    # Data constants
    EU_GREEN_CLAIMS_REQUIREMENTS,
    GREENWASHING_PATTERNS,
    VALID_CERTIFICATIONS,
    PACK_SPEC,
)

__all__ = [
    # Main agent
    "GreenClaimsAgent",
    # Input/Output models
    "GreenClaimsInput",
    "GreenClaimsOutput",
    # Enumerations
    "ClaimType",
    "GreenwashingRedFlag",
    "ValidationStatus",
    "SubstantiationLevel",
    "ComplianceFramework",
    # Supporting models
    "EvidenceItem",
    "ClaimScope",
    "SubstantiationScoreBreakdown",
    "GreenwashingDetection",
    "FrameworkComplianceResult",
    "ComplianceReport",
    "ClaimRequirements",
    # Data constants
    "EU_GREEN_CLAIMS_REQUIREMENTS",
    "GREENWASHING_PATTERNS",
    "VALID_CERTIFICATIONS",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "regulatory/green_claims_v1"
