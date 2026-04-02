# -*- coding: utf-8 -*-
"""
GL-DATA-X-004: ERP/Finance Connector Agent - Re-export Shim
=============================================================

Backward-compatibility shim that re-exports all core enums, models,
constants, and the ERPConnectorAgent class from the canonical locations
in the ``erp_connector`` package.

Canonical sources:
    - Enums, models, constants: ``greenlang.agents.data.erp_connector.models``
    - Agent class: ``greenlang.agents.data.erp_connector.agent``

All new code should import directly from the package sub-modules.
"""

# Core enums, models, and constants
from greenlang.agents.data.erp_connector.models import (  # noqa: F401
    DEFAULT_EMISSION_FACTORS,
    ERPConnectionConfig,
    ERPQueryInput,
    ERPQueryOutput,
    ERPSystem,
    InventoryItem,
    MaterialMapping,
    PurchaseOrder,
    PurchaseOrderLine,
    Scope3Category,
    SpendCategory,
    SpendRecord,
    SPEND_TO_SCOPE3_MAPPING,
    TransactionType,
    VendorMapping,
)

# Agent class
from greenlang.agents.data.erp_connector.agent import (  # noqa: F401
    ERPConnectorAgent,
)

__all__ = [
    # Enums
    "ERPSystem",
    "Scope3Category",
    "TransactionType",
    "SpendCategory",
    # Models
    "ERPConnectionConfig",
    "VendorMapping",
    "MaterialMapping",
    "PurchaseOrderLine",
    "PurchaseOrder",
    "SpendRecord",
    "InventoryItem",
    "ERPQueryInput",
    "ERPQueryOutput",
    # Constants
    "SPEND_TO_SCOPE3_MAPPING",
    "DEFAULT_EMISSION_FACTORS",
    # Agent
    "ERPConnectorAgent",
]
