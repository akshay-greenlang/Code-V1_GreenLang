# -*- coding: utf-8 -*-
"""
SB 253 Climate Disclosure Agent
================================

California SB 253 (Climate Corporate Data Accountability Act) compliance agent.
Calculates Scope 1, 2, and 3 GHG emissions aligned with GHG Protocol standards.

Key Deadlines:
    - June 30, 2026: Scope 1 & 2 reporting required
    - June 30, 2027: Scope 3 reporting required

Features:
    - Automated GHG calculation (Scope 1, 2, 3)
    - California-specific grid factors (CAMX = 0.254 kg CO2e/kWh)
    - Third-party assurance-ready audit trails
    - CARB portal integration
    - Multi-state support (CA, CO, WA)

5-Agent Pipeline:
    1. DataCollectionAgent - Automated data ingestion from ERP/utilities
    2. CalculationAgent - GHG Protocol calculations (zero hallucination)
    3. AssuranceReadyAgent - SHA-256 audit trails
    4. MultiStateFilingAgent - CARB, CDPHE, ECY portal integration
    5. ThirdPartyAssuranceAgent - Big 4 audit package generation

Emission Factor Sources:
    - EPA eGRID 2023 (electricity grid)
    - EPA GHG Emission Factors Hub 2024 (fuel combustion)
    - IPCC AR6 (GWP-100 values)
    - EPA EEIO (spend-based Scope 3)
    - DEFRA 2024 (travel, commuting)
    - GLEC Framework (transportation)

Author: GreenLang Framework Team
Version: 1.0.0
Date: 2025-12-04
"""

from .agent_spec import (
    SB253DisclosureAgent,
    DataCollectionAgent,
    CalculationAgent,
    AssuranceReadyAgent,
    MultiStateFilingAgent,
    ThirdPartyAssuranceAgent,
)

__all__ = [
    "SB253DisclosureAgent",
    "DataCollectionAgent",
    "CalculationAgent",
    "AssuranceReadyAgent",
    "MultiStateFilingAgent",
    "ThirdPartyAssuranceAgent",
]

__version__ = "1.0.0"
__regulation__ = "California SB 253 (Climate Corporate Data Accountability Act)"
__deadline_scope_1_2__ = "2026-06-30"
__deadline_scope_3__ = "2027-06-30"
