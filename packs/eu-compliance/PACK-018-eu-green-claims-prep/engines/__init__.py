# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep - Calculation Engines
=====================================================

Eight deterministic, zero-hallucination engines for EU Green Claims
Directive (COM/2023/166) and Empowering Consumers Directive (2024/825)
compliance preparation.

Engines:
    1. ClaimSubstantiationEngine   - Articles 3-4 substantiation scoring
    2. ComparativeClaimsEngine     - Article 5 comparative claim validation
    3. LifecycleAssessmentEngine   - PEF lifecycle impact assessment
    4. LabelComplianceEngine       - Articles 6-9 label governance
    5. EvidenceChainEngine         - Evidence chain construction and validation
    6. GreenwashingDetectionEngine - Greenwashing risk screening
    7. TraderObligationEngine      - Articles 3-8 trader obligation tracking
    8. GreenClaimsBenchmarkEngine  - Cross-portfolio scoring and maturity
"""

from .claim_substantiation_engine import ClaimSubstantiationEngine
from .comparative_claims_engine import ComparativeClaimsEngine
from .lifecycle_assessment_engine import LifecycleAssessmentEngine
from .label_compliance_engine import LabelComplianceEngine
from .evidence_chain_engine import EvidenceChainEngine
from .greenwashing_detection_engine import GreenwashingDetectionEngine
from .trader_obligation_engine import TraderObligationEngine
from .green_claims_benchmark_engine import GreenClaimsBenchmarkEngine

__all__ = [
    "ClaimSubstantiationEngine",
    "ComparativeClaimsEngine",
    "LifecycleAssessmentEngine",
    "LabelComplianceEngine",
    "EvidenceChainEngine",
    "GreenwashingDetectionEngine",
    "TraderObligationEngine",
    "GreenClaimsBenchmarkEngine",
]
