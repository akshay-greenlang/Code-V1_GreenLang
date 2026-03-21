# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep - Engines Module
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

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-018"
__pack_name__: str = "EU Green Claims Prep Pack"
__engines_count__: int = 8

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Claim Substantiation
# ---------------------------------------------------------------------------
_ENGINE_1_SYMBOLS: list[str] = ["ClaimSubstantiationEngine"]
try:
    from .claim_substantiation_engine import ClaimSubstantiationEngine
    _loaded_engines.append("ClaimSubstantiationEngine")
except ImportError as e:
    logger.debug("Engine 1 (ClaimSubstantiationEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 2: Comparative Claims
# ---------------------------------------------------------------------------
_ENGINE_2_SYMBOLS: list[str] = ["ComparativeClaimsEngine"]
try:
    from .comparative_claims_engine import ComparativeClaimsEngine
    _loaded_engines.append("ComparativeClaimsEngine")
except ImportError as e:
    logger.debug("Engine 2 (ComparativeClaimsEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 3: Lifecycle Assessment
# ---------------------------------------------------------------------------
_ENGINE_3_SYMBOLS: list[str] = ["LifecycleAssessmentEngine"]
try:
    from .lifecycle_assessment_engine import LifecycleAssessmentEngine
    _loaded_engines.append("LifecycleAssessmentEngine")
except ImportError as e:
    logger.debug("Engine 3 (LifecycleAssessmentEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 4: Label Compliance
# ---------------------------------------------------------------------------
_ENGINE_4_SYMBOLS: list[str] = ["LabelComplianceEngine"]
try:
    from .label_compliance_engine import LabelComplianceEngine
    _loaded_engines.append("LabelComplianceEngine")
except ImportError as e:
    logger.debug("Engine 4 (LabelComplianceEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 5: Evidence Chain
# ---------------------------------------------------------------------------
_ENGINE_5_SYMBOLS: list[str] = ["EvidenceChainEngine"]
try:
    from .evidence_chain_engine import EvidenceChainEngine
    _loaded_engines.append("EvidenceChainEngine")
except ImportError as e:
    logger.debug("Engine 5 (EvidenceChainEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 6: Greenwashing Detection
# ---------------------------------------------------------------------------
_ENGINE_6_SYMBOLS: list[str] = ["GreenwashingDetectionEngine"]
try:
    from .greenwashing_detection_engine import GreenwashingDetectionEngine
    _loaded_engines.append("GreenwashingDetectionEngine")
except ImportError as e:
    logger.debug("Engine 6 (GreenwashingDetectionEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 7: Trader Obligation
# ---------------------------------------------------------------------------
_ENGINE_7_SYMBOLS: list[str] = ["TraderObligationEngine"]
try:
    from .trader_obligation_engine import TraderObligationEngine
    _loaded_engines.append("TraderObligationEngine")
except ImportError as e:
    logger.debug("Engine 7 (TraderObligationEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 8: Green Claims Benchmark
# ---------------------------------------------------------------------------
_ENGINE_8_SYMBOLS: list[str] = ["GreenClaimsBenchmarkEngine"]
try:
    from .green_claims_benchmark_engine import GreenClaimsBenchmarkEngine
    _loaded_engines.append("GreenClaimsBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 8 (GreenClaimsBenchmarkEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_ENGINE_1_SYMBOLS,
    *_ENGINE_2_SYMBOLS,
    *_ENGINE_3_SYMBOLS,
    *_ENGINE_4_SYMBOLS,
    *_ENGINE_5_SYMBOLS,
    *_ENGINE_6_SYMBOLS,
    *_ENGINE_7_SYMBOLS,
    *_ENGINE_8_SYMBOLS,
    "get_loaded_engines",
    "get_engine_count",
]


def get_loaded_engines() -> list[str]:
    """Return list of successfully loaded engine class names."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-018 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
