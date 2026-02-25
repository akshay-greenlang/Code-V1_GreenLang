# -*- coding: utf-8 -*-
"""
GreenLang Capital Goods Agent - AGENT-MRV-015 (GL-MRV-S3-002)

Scope 3 Category 2 capital goods emission estimation
covering four calculation methods (spend-based EEIO, average-data physical EFs,
supplier-specific EPD/PCF/CDP, and hybrid multi-method), asset classification
with capitalization threshold enforcement, no-depreciation accounting rule,
CapEx volatility context, double-counting prevention (vs Category 1 and Scope 1/2),
and compliance checking against seven regulatory frameworks.

Engines:
    1. CapitalAssetDatabaseEngine       - EEIO factors, physical EFs, asset classification
    2. SpendBasedCalculatorEngine        - CapEx × EEIO factor calculations
    3. AverageDataCalculatorEngine       - Quantity × physical EF calculations
    4. SupplierSpecificCalculatorEngine  - EPD/PCF/CDP supplier calculations
    5. HybridAggregatorEngine           - Multi-method aggregation, hot-spot analysis
    6. ComplianceCheckerEngine          - 7-framework regulatory compliance
    7. CapitalGoodsPipelineEngine       - 10-stage pipeline orchestration

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "CapitalAssetDatabaseEngine",
    "SpendBasedCalculatorEngine",
    "AverageDataCalculatorEngine",
    "SupplierSpecificCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "CapitalGoodsPipelineEngine",
]

VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Graceful engine imports - engines may not yet be implemented
# ---------------------------------------------------------------------------

try:
    from greenlang.capital_goods.capital_asset_database import (
        CapitalAssetDatabaseEngine,
    )
except ImportError:
    CapitalAssetDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.capital_goods.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.capital_goods.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.capital_goods.supplier_specific_calculator import (
        SupplierSpecificCalculatorEngine,
    )
except ImportError:
    SupplierSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.capital_goods.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.capital_goods.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.capital_goods.capital_goods_pipeline import (
        CapitalGoodsPipelineEngine,
    )
except ImportError:
    CapitalGoodsPipelineEngine = None  # type: ignore[assignment,misc]
