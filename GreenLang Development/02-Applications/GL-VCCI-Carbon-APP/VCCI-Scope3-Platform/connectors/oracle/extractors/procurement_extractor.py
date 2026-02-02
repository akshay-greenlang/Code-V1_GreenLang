# -*- coding: utf-8 -*-
"""
Oracle Procurement Cloud Extractor

Extracts data from Oracle Fusion Procurement Cloud including:
    - Purchase Orders (/purchaseOrders)
    - Purchase Requisitions (/purchaseRequisitions)
    - Suppliers (/suppliers)

Supports delta extraction by LastUpdateDate field and provides field selection
for performance optimization. Uses Oracle REST API query syntax (q parameter).

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class PurchaseOrderData(BaseModel):
    """Oracle Purchase Order data model.

    Maps to /purchaseOrders REST endpoint response.
    """
    POHeaderId: int
    OrderNumber: str
    DocumentStatus: Optional[str] = None
    BuyerId: Optional[int] = None
    BuyerName: Optional[str] = None
    SupplierId: Optional[int] = None
    SupplierName: Optional[str] = None
    SupplierSiteId: Optional[int] = None
    OrderedDate: Optional[str] = None
    BU: Optional[str] = None  # Business Unit
    BUName: Optional[str] = None
    Currency: Optional[str] = None
    TotalAmount: Optional[float] = None
    PaymentTerms: Optional[str] = None
    FreightTerms: Optional[str] = None
    FOBPoint: Optional[str] = None
    CreatedBy: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdatedBy: Optional[str] = None
    LastUpdateDate: Optional[str] = None  # For delta extraction


class PurchaseOrderLineData(BaseModel):
    """Oracle Purchase Order Line data model.

    Maps to /purchaseOrderLines REST endpoint response.
    """
    POLineId: int
    POHeaderId: int
    LineNumber: int
    ItemId: Optional[int] = None
    ItemDescription: Optional[str] = None
    ItemNumber: Optional[str] = None
    CategoryId: Optional[int] = None
    CategoryName: Optional[str] = None
    UOM: Optional[str] = None
    Quantity: Optional[float] = None
    UnitPrice: Optional[float] = None
    LineAmount: Optional[float] = None
    TaxAmount: Optional[float] = None
    ShipToLocationId: Optional[int] = None
    ShipToLocationCode: Optional[str] = None
    NeedByDate: Optional[str] = None
    PromisedDate: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdateDate: Optional[str] = None


class PurchaseRequisitionData(BaseModel):
    """Oracle Purchase Requisition data model.

    Maps to /purchaseRequisitions REST endpoint response.
    """
    RequisitionHeaderId: int
    RequisitionNumber: str
    DocumentStatus: Optional[str] = None
    PreparerId: Optional[int] = None
    PreparerName: Optional[str] = None
    BU: Optional[str] = None
    BUName: Optional[str] = None
    Description: Optional[str] = None
    JustificationText: Optional[str] = None
    CreatedBy: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdatedBy: Optional[str] = None
    LastUpdateDate: Optional[str] = None  # For delta extraction


class PurchaseRequisitionLineData(BaseModel):
    """Oracle Purchase Requisition Line data model.

    Maps to /purchaseRequisitionLines REST endpoint response.
    """
    RequisitionLineId: int
    RequisitionHeaderId: int
    LineNumber: int
    ItemId: Optional[int] = None
    ItemDescription: Optional[str] = None
    ItemNumber: Optional[str] = None
    CategoryId: Optional[int] = None
    CategoryName: Optional[str] = None
    UOM: Optional[str] = None
    Quantity: Optional[float] = None
    UnitPrice: Optional[float] = None
    Amount: Optional[float] = None
    Currency: Optional[str] = None
    NeedByDate: Optional[str] = None
    SuggestedSupplierId: Optional[int] = None
    SuggestedSupplierName: Optional[str] = None
    DestinationTypeCode: Optional[str] = None
    DeliverToLocationId: Optional[int] = None
    CreationDate: Optional[str] = None
    LastUpdateDate: Optional[str] = None


class SupplierData(BaseModel):
    """Oracle Supplier Master data model.

    Maps to /suppliers REST endpoint response.
    """
    SupplierId: int
    SupplierName: str
    SupplierNumber: Optional[str] = None
    TaxOrganizationType: Optional[str] = None
    BusinessRelationship: Optional[str] = None
    DUNSNumber: Optional[str] = None
    TaxPayerId: Optional[str] = None
    VATRegistrationNumber: Optional[str] = None
    AddressLine1: Optional[str] = None
    AddressLine2: Optional[str] = None
    City: Optional[str] = None
    State: Optional[str] = None
    Province: Optional[str] = None
    Country: Optional[str] = None
    PostalCode: Optional[str] = None
    CreatedBy: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdatedBy: Optional[str] = None
    LastUpdateDate: Optional[str] = None  # For delta extraction


class ProcurementExtractor(BaseExtractor):
    """Procurement Cloud Extractor.

    Extracts procurement data from Oracle Fusion Procurement Cloud.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize Procurement extractor.

        Args:
            client: Oracle REST client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.base_url = "/fscmRestApi/resources/11.13.18.05"  # Oracle Procurement REST API base
        self._current_resource = "/purchaseOrders"  # Default

    def get_resource_path(self) -> str:
        """Get current REST resource path."""
        return f"{self.base_url}{self._current_resource}"

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "LastUpdateDate"

    def extract_purchase_orders(
        self,
        business_unit: Optional[str] = None,
        supplier_id: Optional[int] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Purchase Orders from Oracle.

        Args:
            business_unit: Filter by business unit code
            supplier_id: Filter by supplier ID
            status: Filter by status (e.g., 'APPROVED', 'OPEN')
            date_from: Filter by ordered date from (ISO format)
            date_to: Filter by ordered date to (ISO format)

        Yields:
            Purchase Order records as dictionaries
        """
        self._current_resource = "/purchaseOrders"

        additional_filters = []

        if business_unit:
            additional_filters.append(f"BU='{business_unit}'")
        if supplier_id:
            additional_filters.append(f"SupplierId={supplier_id}")
        if status:
            additional_filters.append(f"DocumentStatus='{status}'")
        if date_from:
            additional_filters.append(f"OrderedDate>='{date_from}'")
        if date_to:
            additional_filters.append(f"OrderedDate<='{date_to}'")

        logger.info(f"Extracting Purchase Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="OrderedDate:desc"
        )

    def extract_purchase_order_lines(
        self,
        po_header_id: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Purchase Order Lines from Oracle.

        Args:
            po_header_id: Filter by specific PO header ID

        Yields:
            Purchase Order Line records as dictionaries
        """
        self._current_resource = "/purchaseOrderLines"

        additional_filters = []
        if po_header_id:
            additional_filters.append(f"POHeaderId={po_header_id}")

        logger.info(f"Extracting Purchase Order Lines for PO: {po_header_id or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="POHeaderId:asc,LineNumber:asc"
        )

    def extract_purchase_requisitions(
        self,
        business_unit: Optional[str] = None,
        preparer_id: Optional[int] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Purchase Requisitions from Oracle.

        Args:
            business_unit: Filter by business unit code
            preparer_id: Filter by preparer user ID
            status: Filter by status (e.g., 'APPROVED', 'IN_PROCESS')
            date_from: Filter by creation date from (ISO format)
            date_to: Filter by creation date to (ISO format)

        Yields:
            Purchase Requisition records as dictionaries
        """
        self._current_resource = "/purchaseRequisitions"

        additional_filters = []

        if business_unit:
            additional_filters.append(f"BU='{business_unit}'")
        if preparer_id:
            additional_filters.append(f"PreparerId={preparer_id}")
        if status:
            additional_filters.append(f"DocumentStatus='{status}'")
        if date_from:
            additional_filters.append(f"CreationDate>='{date_from}'")
        if date_to:
            additional_filters.append(f"CreationDate<='{date_to}'")

        logger.info(f"Extracting Purchase Requisitions with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="CreationDate:desc"
        )

    def extract_purchase_requisition_lines(
        self,
        requisition_header_id: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Purchase Requisition Lines from Oracle.

        Args:
            requisition_header_id: Filter by specific requisition header ID

        Yields:
            Purchase Requisition Line records as dictionaries
        """
        self._current_resource = "/purchaseRequisitionLines"

        additional_filters = []
        if requisition_header_id:
            additional_filters.append(f"RequisitionHeaderId={requisition_header_id}")

        logger.info(f"Extracting Requisition Lines for Req: {requisition_header_id or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="RequisitionHeaderId:asc,LineNumber:asc"
        )

    def extract_suppliers(
        self,
        country: Optional[str] = None,
        status: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Supplier Master data from Oracle.

        Args:
            country: Filter by country code (ISO 2-letter)
            status: Filter by business relationship status

        Yields:
            Supplier master records as dictionaries
        """
        self._current_resource = "/suppliers"

        additional_filters = []
        if country:
            additional_filters.append(f"Country='{country}'")
        if status:
            additional_filters.append(f"BusinessRelationship='{status}'")

        logger.info(f"Extracting Supplier Master data")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="SupplierName:asc"
        )
