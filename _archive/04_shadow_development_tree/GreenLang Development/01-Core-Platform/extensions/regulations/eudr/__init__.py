# -*- coding: utf-8 -*-
"""
greenlang/regulations/eudr/__init__.py

EU Deforestation Regulation (EUDR) Module

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires companies
placing specific commodities on the EU market to ensure products are:
1. Deforestation-free (not produced on land deforested after Dec 31, 2020)
2. Produced in accordance with relevant legislation of country of production
3. Covered by a due diligence statement

Covered Commodities:
- Cattle
- Cocoa
- Coffee
- Oil palm
- Rubber
- Soya
- Wood
- And derived products (leather, chocolate, furniture, etc.)

Key Dates:
- Entry into force: June 29, 2023
- Large operators compliance: December 30, 2024
- SME compliance: June 30, 2025

This module provides:
- Data models for EUDR compliance data
- Geolocation validation for production plots
- Due diligence statement generation
- Risk assessment calculations
- Integration with GreenLang's emission tracking

Author: GreenLang Framework Team
Date: November 2025
"""

from greenlang.regulations.eudr.models import (
    EUDRCommodity,
    EUDRProduct,
    GeolocationData,
    ProductionPlot,
    DueDiligenceStatement,
    RiskAssessment,
    SupplierDeclaration,
    EUDRComplianceStatus,
)

from greenlang.regulations.eudr.validators import (
    validate_geolocation,
    validate_deforestation_free,
    validate_production_legality,
)

from greenlang.regulations.eudr.risk_engine import (
    calculate_country_risk,
    calculate_commodity_risk,
    calculate_overall_risk,
)

__all__ = [
    # Models
    "EUDRCommodity",
    "EUDRProduct",
    "GeolocationData",
    "ProductionPlot",
    "DueDiligenceStatement",
    "RiskAssessment",
    "SupplierDeclaration",
    "EUDRComplianceStatus",
    # Validators
    "validate_geolocation",
    "validate_deforestation_free",
    "validate_production_legality",
    # Risk Engine
    "calculate_country_risk",
    "calculate_commodity_risk",
    "calculate_overall_risk",
]
