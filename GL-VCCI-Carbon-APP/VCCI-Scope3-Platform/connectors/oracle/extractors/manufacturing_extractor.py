"""
Oracle Manufacturing Extractor

Extracts data from Oracle Fusion Cloud Manufacturing including:
    - Work Orders (/fscmRestApi/resources/11.13.18.05/workOrders)
    - Work Order Operations
    - Work Order Materials
    - Production Lots

Use Cases:
    - Production emissions tracking
    - Manufacturing energy consumption
    - Category 1: Manufactured goods emissions

Carbon Impact: HIGH

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class WorkOrderData(BaseModel):
    """Oracle Work Order data model."""
    WorkOrderId: int
    WorkOrderNumber: str
    ItemNumber: Optional[str] = None
    ItemDescription: Optional[str] = None
    OrganizationCode: str
    StatusCode: Optional[str] = None
    OrderQuantity: float
    CompletedQuantity: Optional[float] = None
    ScrapQuantity: Optional[float] = None
    UOMCode: str
    ScheduledStartDate: Optional[str] = None
    ScheduledCompletionDate: Optional[str] = None
    ActualStartDate: Optional[str] = None
    ActualCompletionDate: Optional[str] = None
    LastUpdateDate: Optional[str] = None


class WorkOrderOperationData(BaseModel):
    """Oracle Work Order Operation data model."""
    WorkOrderOperationId: int
    WorkOrderId: int
    OperationSequence: int
    OperationCode: Optional[str] = None
    DepartmentCode: Optional[str] = None
    ResourceCode: Optional[str] = None
    StandardOperationTime: Optional[float] = None
    ActualOperationTime: Optional[float] = None


class ManufacturingExtractor(BaseExtractor):
    """Oracle Manufacturing Extractor."""

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        super().__init__(client, config)
        self.service_name = "MFG"
        self._current_entity_set = "workOrders"

    def get_entity_set_name(self) -> str:
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        return "LastUpdateDate"

    def extract_work_orders(
        self,
        organization: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Work Orders from Oracle."""
        self._current_entity_set = "workOrders"

        additional_filters = []
        if organization:
            additional_filters.append(f"OrganizationCode = '{organization}'")
        if status:
            additional_filters.append(f"StatusCode = '{status}'")
        if date_from:
            additional_filters.append(f"ScheduledStartDate >= '{date_from}'")
        if date_to:
            additional_filters.append(f"ScheduledStartDate <= '{date_to}'")

        logger.info(f"Extracting Work Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_work_order_operations(
        self,
        work_order_id: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Work Order Operations."""
        self._current_entity_set = "workOrderOperations"

        additional_filters = []
        if work_order_id:
            additional_filters.append(f"WorkOrderId = {work_order_id}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )
