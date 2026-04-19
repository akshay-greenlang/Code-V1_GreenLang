# -*- coding: utf-8 -*-
"""
PACK-024 Carbon Neutral Pack - Engines
=======================================

10 calculation engines for the complete carbon neutrality lifecycle.

Engines:
    1. FootprintQuantificationEngine   -- ISO 14064-1:2018 GHG quantification
    2. CarbonMgmtPlanEngine            -- Reduction-first MACC planning
    3. CreditQualityEngine             -- 12-dimension ICVCM CCP scoring
    4. PortfolioOptimizationEngine     -- Markowitz-inspired credit allocation
    5. RegistryRetirementEngine        -- 6-registry retirement tracking
    6. NeutralizationBalanceEngine     -- ISO 14068-1 / PAS 2060 balance
    7. ClaimsSubstantiationEngine      -- VCMI Claims Code validation
    8. VerificationPackageEngine       -- ISAE 3410 evidence assembly
    9. AnnualCycleEngine               -- Multi-year cycle management
    10. PermanenceRiskEngine           -- Buffer pool risk assessment

Regulatory Basis:
    ISO 14068-1:2023 (Carbon neutrality)
    PAS 2060:2014 (Demonstrating carbon neutrality)
    ISO 14064-1:2018 (GHG quantification)
    ICVCM Core Carbon Principles (2023)
    VCMI Claims Code of Practice (2023)
    ISAE 3410 (Assurance on GHG statements)

Pack Tier: Professional (PACK-024)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-024"
__pack_name__: str = "Carbon Neutral Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Footprint Quantification
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "FootprintQuantificationEngine",
]

try:
    from .footprint_quantification_engine import FootprintQuantificationEngine
    _loaded_engines.append("FootprintQuantificationEngine")
except ImportError as e:
    logger.debug("Engine 1 (FootprintQuantificationEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Carbon Management Plan
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "CarbonMgmtPlanEngine",
]

try:
    from .carbon_mgmt_plan_engine import CarbonMgmtPlanEngine
    _loaded_engines.append("CarbonMgmtPlanEngine")
except ImportError as e:
    logger.debug("Engine 2 (CarbonMgmtPlanEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Credit Quality
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "CreditQualityEngine",
]

try:
    from .credit_quality_engine import CreditQualityEngine
    _loaded_engines.append("CreditQualityEngine")
except ImportError as e:
    logger.debug("Engine 3 (CreditQualityEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Portfolio Optimization
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "PortfolioOptimizationEngine",
]

try:
    from .portfolio_optimization_engine import PortfolioOptimizationEngine
    _loaded_engines.append("PortfolioOptimizationEngine")
except ImportError as e:
    logger.debug("Engine 4 (PortfolioOptimizationEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Registry Retirement
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "RegistryRetirementEngine",
]

try:
    from .registry_retirement_engine import RegistryRetirementEngine
    _loaded_engines.append("RegistryRetirementEngine")
except ImportError as e:
    logger.debug("Engine 5 (RegistryRetirementEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Neutralization Balance
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "NeutralizationBalanceEngine",
]

try:
    from .neutralization_balance_engine import NeutralizationBalanceEngine
    _loaded_engines.append("NeutralizationBalanceEngine")
except ImportError as e:
    logger.debug("Engine 6 (NeutralizationBalanceEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Claims Substantiation
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "ClaimsSubstantiationEngine",
]

try:
    from .claims_substantiation_engine import ClaimsSubstantiationEngine
    _loaded_engines.append("ClaimsSubstantiationEngine")
except ImportError as e:
    logger.debug("Engine 7 (ClaimsSubstantiationEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Verification Package
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "VerificationPackageEngine",
]

try:
    from .verification_package_engine import VerificationPackageEngine
    _loaded_engines.append("VerificationPackageEngine")
except ImportError as e:
    logger.debug("Engine 8 (VerificationPackageEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Annual Cycle
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "AnnualCycleEngine",
]

try:
    from .annual_cycle_engine import AnnualCycleEngine
    _loaded_engines.append("AnnualCycleEngine")
except ImportError as e:
    logger.debug("Engine 9 (AnnualCycleEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Permanence Risk
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "PermanenceRiskEngine",
]

try:
    from .permanence_risk_engine import PermanenceRiskEngine
    _loaded_engines.append("PermanenceRiskEngine")
except ImportError as e:
    logger.debug("Engine 10 (PermanenceRiskEngine) not available: %s", e)
    _ENGINE_10_SYMBOLS = []


# ===================================================================
# Public API - dynamically collected from successfully loaded engines
# ===================================================================

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
    *_ENGINE_9_SYMBOLS,
    *_ENGINE_10_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-024 Carbon Neutral engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
