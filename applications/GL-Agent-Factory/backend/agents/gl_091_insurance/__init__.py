"""GL-091: Insurance Optimizer Agent (INSURANCE-OPT)"""

from .agent import (
    InsuranceOptimizerAgent,
    InsuranceInput,
    InsurancePolicy,
    RiskExposure,
    AssetInventory,
    ClaimsHistory,
    InsuranceOutput,
    CoverageGap,
    InsuranceRecommendation,
    PolicyAnalysis,
    RiskAnalysis,
    ProvenanceRecord,
    PolicyType,
    RiskCategory,
    CoverageLevel,
    RecommendationType,
    PACK_SPEC,
)

__all__ = [
    "InsuranceOptimizerAgent",
    "InsuranceInput",
    "InsurancePolicy",
    "RiskExposure",
    "AssetInventory",
    "ClaimsHistory",
    "InsuranceOutput",
    "CoverageGap",
    "InsuranceRecommendation",
    "PolicyAnalysis",
    "RiskAnalysis",
    "ProvenanceRecord",
    "PolicyType",
    "RiskCategory",
    "CoverageLevel",
    "RecommendationType",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-091"
__agent_name__ = "INSURANCE-OPT"
