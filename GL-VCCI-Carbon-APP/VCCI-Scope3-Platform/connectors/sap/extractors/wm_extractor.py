"""
SAP Warehouse Management (WM) Extractor

Extracts data from SAP S/4HANA Warehouse Management module including:
    - Warehouse Orders (API_WHSE_ORDER_SRV)
    - Warehouse Tasks
    - Storage Bin Data
    - Warehouse Transfers

Use Cases:
    - Category 4: Warehouse-to-warehouse transfers and logistics emissions
    - Material handling equipment emissions
    - Storage energy consumption

Carbon Impact: MEDIUM

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class WarehouseOrderData(BaseModel):
    """SAP Warehouse Order data model."""
    WarehouseOrder: str
    Warehouse: str
    WarehouseOrderType: Optional[str] = None
    WarehouseOrderCategory: Optional[str] = None
    SourceStorageType: Optional[str] = None
    SourceStorageBin: Optional[str] = None
    DestinationStorageType: Optional[str] = None
    DestinationStorageBin: Optional[str] = None
    Material: Optional[str] = None
    Quantity: Optional[float] = None
    QuantityUnit: Optional[str] = None
    WarehouseOrderStatus: Optional[str] = None
    CreationDate: Optional[str] = None
    ChangedOn: Optional[str] = None


class WarehouseTaskData(BaseModel):
    """SAP Warehouse Task data model."""
    WarehouseTask: str
    WarehouseOrder: str
    Warehouse: str
    SourceStorageType: Optional[str] = None
    DestinationStorageType: Optional[str] = None
    Material: Optional[str] = None
    ActualQuantity: Optional[float] = None
    QuantityUnit: Optional[str] = None
    TaskStatus: Optional[str] = None
    ProcessingTime: Optional[float] = None


class WMExtractor(BaseExtractor):
    """Warehouse Management (WM) Extractor."""

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        super().__init__(client, config)
        self.service_name = "WM"
        self._current_entity_set = "A_WarehouseOrder"

    def get_entity_set_name(self) -> str:
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        return "ChangedOn"

    def extract_warehouse_orders(
        self,
        warehouse: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Warehouse Orders."""
        self._current_entity_set = "A_WarehouseOrder"

        additional_filters = []
        if warehouse:
            additional_filters.append(f"Warehouse eq '{warehouse}'")
        if date_from:
            additional_filters.append(f"CreationDate ge datetime'{date_from}'")
        if date_to:
            additional_filters.append(f"CreationDate le datetime'{date_to}'")
        if status:
            additional_filters.append(f"WarehouseOrderStatus eq '{status}'")

        logger.info(f"Extracting Warehouse Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_warehouse_tasks(
        self,
        warehouse_order: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Warehouse Tasks."""
        self._current_entity_set = "A_WarehouseTask"

        additional_filters = []
        if warehouse_order:
            additional_filters.append(f"WarehouseOrder eq '{warehouse_order}'")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )
