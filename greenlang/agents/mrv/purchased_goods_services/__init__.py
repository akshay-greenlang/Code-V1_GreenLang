# -*- coding: utf-8 -*-
"""
GreenLang Purchased Goods & Services Agent - AGENT-MRV-014 (GL-MRV-S3-001)

Scope 3 Category 1 purchased goods and services emission estimation
covering four calculation methods (spend-based EEIO, average-data physical EFs,
supplier-specific EPD/PCF/CDP, and hybrid multi-method), classification cross-mapping
(NAICS/NACE/ISIC/UNSPSC), currency conversion, margin adjustment, data quality
indicator (DQI) scoring, hot-spot analysis, double-counting prevention, and
compliance checking against seven regulatory frameworks.

Engines:
    1. ProcurementDatabaseEngine        - EEIO factors, physical EFs, classification mapping
    2. SpendBasedCalculatorEngine        - Spend × EEIO factor calculations
    3. AverageDataCalculatorEngine       - Quantity × physical EF calculations
    4. SupplierSpecificCalculatorEngine  - EPD/PCF/CDP supplier calculations
    5. HybridAggregatorEngine           - Multi-method aggregation, hot-spot analysis
    6. ComplianceCheckerEngine          - 7-framework regulatory compliance
    7. PurchasedGoodsPipelineEngine     - 10-stage pipeline orchestration

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "ProcurementDatabaseEngine",
    "SpendBasedCalculatorEngine",
    "AverageDataCalculatorEngine",
    "SupplierSpecificCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "PurchasedGoodsPipelineEngine",
]

VERSION: str = "1.0.0"

# Graceful engine imports
try:
    from greenlang.agents.mrv.purchased_goods_services.procurement_database import ProcurementDatabaseEngine
except ImportError:
    ProcurementDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.purchased_goods_services.spend_based_calculator import SpendBasedCalculatorEngine
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.purchased_goods_services.average_data_calculator import AverageDataCalculatorEngine
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.purchased_goods_services.supplier_specific_calculator import SupplierSpecificCalculatorEngine
except ImportError:
    SupplierSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.purchased_goods_services.hybrid_aggregator import HybridAggregatorEngine
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.purchased_goods_services.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.purchased_goods_services.purchased_goods_pipeline import PurchasedGoodsPipelineEngine
except ImportError:
    PurchasedGoodsPipelineEngine = None  # type: ignore[assignment,misc]
