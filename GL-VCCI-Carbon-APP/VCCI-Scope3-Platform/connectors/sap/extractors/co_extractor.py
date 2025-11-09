"""
SAP Controlling (CO) Extractor

Extracts data from SAP S/4HANA Controlling module including:
    - Cost Centers (API_CONTROLLINGAREA_SRV)
    - Cost Elements
    - Internal Orders
    - Activity Types

Use Cases:
    - Cost center emissions allocation
    - Activity-based costing for carbon accounting
    - Overhead allocation for indirect emissions

Carbon Impact: HIGH

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class CostCenterData(BaseModel):
    """SAP Cost Center data model."""
    CostCenter: str
    ControllingArea: str
    CostCenterName: Optional[str] = None
    CostCenterCategory: Optional[str] = None
    ValidityStartDate: Optional[str] = None
    ValidityEndDate: Optional[str] = None
    CompanyCode: Optional[str] = None
    ProfitCenter: Optional[str] = None
    ResponsiblePerson: Optional[str] = None
    CostCenterCurrency: Optional[str] = None
    ChangedOn: Optional[str] = None


class InternalOrderData(BaseModel):
    """SAP Internal Order data model."""
    InternalOrder: str
    OrderType: str
    OrderDescription: Optional[str] = None
    ControllingArea: str
    CostCenter: Optional[str] = None
    CompanyCode: Optional[str] = None
    PlannedCosts: Optional[float] = None
    ActualCosts: Optional[float] = None
    OrderCurrency: Optional[str] = None
    OrderStartDate: Optional[str] = None
    OrderEndDate: Optional[str] = None
    ChangedOn: Optional[str] = None


class COExtractor(BaseExtractor):
    """Controlling (CO) Extractor."""

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        super().__init__(client, config)
        self.service_name = "CO"
        self._current_entity_set = "A_CostCenter"

    def get_entity_set_name(self) -> str:
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        return "ChangedOn"

    def extract_cost_centers(
        self,
        controlling_area: Optional[str] = None,
        company_code: Optional[str] = None,
        valid_on_date: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Cost Centers."""
        self._current_entity_set = "A_CostCenter"

        additional_filters = []
        if controlling_area:
            additional_filters.append(f"ControllingArea eq '{controlling_area}'")
        if company_code:
            additional_filters.append(f"CompanyCode eq '{company_code}'")
        if valid_on_date:
            additional_filters.append(f"ValidityStartDate le datetime'{valid_on_date}'")
            additional_filters.append(f"ValidityEndDate ge datetime'{valid_on_date}'")

        logger.info(f"Extracting Cost Centers")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_internal_orders(
        self,
        controlling_area: Optional[str] = None,
        order_type: Optional[str] = None,
        date_from: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Internal Orders."""
        self._current_entity_set = "A_InternalOrder"

        additional_filters = []
        if controlling_area:
            additional_filters.append(f"ControllingArea eq '{controlling_area}'")
        if order_type:
            additional_filters.append(f"OrderType eq '{order_type}'")
        if date_from:
            additional_filters.append(f"OrderStartDate ge datetime'{date_from}'")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )
