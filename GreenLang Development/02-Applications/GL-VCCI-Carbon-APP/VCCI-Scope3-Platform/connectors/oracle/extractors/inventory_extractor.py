# -*- coding: utf-8 -*-
"""
Oracle Inventory Management Extractor

Extracts data from Oracle Fusion Cloud Inventory Management including:
    - Material Transactions (/fscmRestApi/resources/11.13.18.05/materialTransactions)
    - Onhand Quantities
    - Lots and Serial Numbers
    - Subinventory Transfers

Use Cases:
    - Stock movement emissions tracking
    - Inventory holding emissions
    - Category 4: Logistics and transfers

Carbon Impact: MEDIUM

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class MaterialTransactionData(BaseModel):
    """Oracle Material Transaction data model."""
    TransactionId: int
    TransactionDate: str
    TransactionType: str
    ItemNumber: str
    OrganizationCode: str
    Subinventory: Optional[str] = None
    TransactionQuantity: float
    TransactionUOM: str
    SourceType: Optional[str] = None
    SourceReference: Optional[str] = None
    LastUpdateDate: Optional[str] = None


class OnhandQuantityData(BaseModel):
    """Oracle Onhand Quantity data model."""
    ItemNumber: str
    OrganizationCode: str
    Subinventory: str
    Locator: Optional[str] = None
    OnhandQuantity: float
    AvailableQuantity: float
    ReservedQuantity: Optional[float] = None
    UOMCode: str


class InventoryExtractor(BaseExtractor):
    """Oracle Inventory Management Extractor."""

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        super().__init__(client, config)
        self.service_name = "INV"
        self._current_entity_set = "materialTransactions"

    def get_entity_set_name(self) -> str:
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        return "LastUpdateDate"

    def extract_material_transactions(
        self,
        organization: Optional[str] = None,
        transaction_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Material Transactions."""
        self._current_entity_set = "materialTransactions"

        additional_filters = []
        if organization:
            additional_filters.append(f"OrganizationCode = '{organization}'")
        if transaction_type:
            additional_filters.append(f"TransactionType = '{transaction_type}'")
        if date_from:
            additional_filters.append(f"TransactionDate >= '{date_from}'")
        if date_to:
            additional_filters.append(f"TransactionDate <= '{date_to}'")

        logger.info(f"Extracting Material Transactions")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_onhand_quantities(
        self,
        organization: Optional[str] = None,
        item_number: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Onhand Quantities."""
        self._current_entity_set = "onhandQuantities"

        additional_filters = []
        if organization:
            additional_filters.append(f"OrganizationCode = '{organization}'")
        if item_number:
            additional_filters.append(f"ItemNumber = '{item_number}'")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )
