# -*- coding: utf-8 -*-
"""
SAP Materials Management (MM) Extractor

Extracts data from SAP S/4HANA Materials Management module including:
    - Purchase Orders (MM_PUR_PO_MAINT_V2_SRV)
    - Goods Receipts (API_MATERIAL_DOCUMENT_SRV)
    - Vendor Master Data (MD_SUPPLIER_MASTER_SRV)
    - Material Master Data (API_MATERIAL_STOCK_SRV)

Supports delta extraction by ChangedOn field and provides field selection
for performance optimization.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class PurchaseOrderData(BaseModel):
    """SAP Purchase Order data model.

    Maps to A_PurchaseOrder entity from MM_PUR_PO_MAINT_V2_SRV.
    """
    PurchaseOrder: str
    PurchaseOrderType: Optional[str] = None
    Vendor: str
    VendorName: Optional[str] = None
    PurchaseOrderDate: Optional[str] = None
    CompanyCode: Optional[str] = None
    PurchasingOrganization: Optional[str] = None
    PurchasingGroup: Optional[str] = None
    DocumentCurrency: Optional[str] = None
    ExchangeRate: Optional[float] = None
    PaymentTerms: Optional[str] = None
    IncotermsClassification: Optional[str] = None
    IncotermsLocation1: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class PurchaseOrderItemData(BaseModel):
    """SAP Purchase Order Item data model.

    Maps to A_PurchaseOrderItem entity.
    """
    PurchaseOrder: str
    PurchaseOrderItem: str
    Material: Optional[str] = None
    MaterialGroup: Optional[str] = None
    PurchaseOrderItemText: Optional[str] = None
    Plant: Optional[str] = None
    StorageLocation: Optional[str] = None
    OrderQuantity: Optional[float] = None
    PurchaseOrderQuantityUnit: Optional[str] = None
    NetPriceAmount: Optional[float] = None
    NetPriceQuantity: Optional[float] = None
    OrderPriceUnit: Optional[str] = None
    NetAmount: Optional[float] = None
    GrossAmount: Optional[float] = None
    TaxCode: Optional[str] = None
    AccountAssignmentCategory: Optional[str] = None
    GLAccount: Optional[str] = None
    CostCenter: Optional[str] = None
    DeliveryDate: Optional[str] = None


class GoodsReceiptData(BaseModel):
    """SAP Goods Receipt data model.

    Maps to A_MaterialDocumentHeader entity from API_MATERIAL_DOCUMENT_SRV.
    """
    MaterialDocument: str
    MaterialDocumentYear: str
    MaterialDocumentHeaderText: Optional[str] = None
    DocumentDate: Optional[str] = None
    PostingDate: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    CreationTime: Optional[str] = None
    ReferenceDocument: Optional[str] = None
    GoodsMovementCode: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class GoodsReceiptItemData(BaseModel):
    """SAP Goods Receipt Item data model.

    Maps to A_MaterialDocumentItem entity.
    """
    MaterialDocument: str
    MaterialDocumentYear: str
    MaterialDocumentItem: str
    Material: Optional[str] = None
    Plant: Optional[str] = None
    StorageLocation: Optional[str] = None
    Batch: Optional[str] = None
    GoodsMovementType: Optional[str] = None
    Supplier: Optional[str] = None
    PurchaseOrder: Optional[str] = None
    PurchaseOrderItem: Optional[str] = None
    QuantityInEntryUnit: Optional[float] = None
    EntryUnit: Optional[str] = None
    QuantityInBaseUnit: Optional[float] = None
    MaterialBaseUnit: Optional[str] = None
    DeliveryNote: Optional[str] = None
    ShipmentNumber: Optional[str] = None


class VendorMasterData(BaseModel):
    """SAP Vendor Master data model.

    Maps to A_Supplier entity from MD_SUPPLIER_MASTER_SRV.
    """
    Supplier: str
    SupplierName: Optional[str] = None
    SupplierFullName: Optional[str] = None
    OrganizationBPName1: Optional[str] = None
    OrganizationBPName2: Optional[str] = None
    Country: Optional[str] = None
    Region: Optional[str] = None
    CityName: Optional[str] = None
    PostalCode: Optional[str] = None
    StreetName: Optional[str] = None
    TaxNumber1: Optional[str] = None
    VATRegistration: Optional[str] = None
    IndustryCode1: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class MaterialMasterData(BaseModel):
    """SAP Material Master data model.

    Maps to A_Material entity from API_MATERIAL_STOCK_SRV.
    """
    Material: str
    MaterialType: Optional[str] = None
    MaterialGroup: Optional[str] = None
    MaterialBaseUnit: Optional[str] = None
    MaterialDescription: Optional[str] = None
    IndustryStandardName: Optional[str] = None
    ProductHierarchy: Optional[str] = None
    GrossWeight: Optional[float] = None
    NetWeight: Optional[float] = None
    WeightUnit: Optional[str] = None
    Volume: Optional[float] = None
    VolumeUnit: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class MMExtractor(BaseExtractor):
    """Materials Management (MM) Extractor.

    Extracts procurement and inventory data from SAP S/4HANA MM module.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize MM extractor.

        Args:
            client: SAP OData client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "MM"
        self._current_entity_set = "A_PurchaseOrder"  # Default

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "ChangedOn"

    def extract_purchase_orders(
        self,
        company_code: Optional[str] = None,
        vendor: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Purchase Orders from SAP.

        Args:
            company_code: Filter by company code
            vendor: Filter by vendor number
            date_from: Filter by PO date from (ISO format)
            date_to: Filter by PO date to (ISO format)

        Yields:
            Purchase Order records as dictionaries
        """
        self._current_entity_set = "A_PurchaseOrder"

        additional_filters = []

        if company_code:
            additional_filters.append(f"CompanyCode eq '{company_code}'")
        if vendor:
            additional_filters.append(f"Vendor eq '{vendor}'")
        if date_from:
            additional_filters.append(f"PurchaseOrderDate ge datetime'{date_from}'")
        if date_to:
            additional_filters.append(f"PurchaseOrderDate le datetime'{date_to}'")

        logger.info(f"Extracting Purchase Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="PurchaseOrderDate desc"
        )

    def extract_purchase_order_items(
        self,
        purchase_order: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Purchase Order Items from SAP.

        Args:
            purchase_order: Filter by specific PO number

        Yields:
            Purchase Order Item records as dictionaries
        """
        self._current_entity_set = "A_PurchaseOrderItem"

        additional_filters = []
        if purchase_order:
            additional_filters.append(f"PurchaseOrder eq '{purchase_order}'")

        logger.info(f"Extracting Purchase Order Items for PO: {purchase_order or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_goods_receipts(
        self,
        posting_date_from: Optional[str] = None,
        posting_date_to: Optional[str] = None,
        plant: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Goods Receipt headers from SAP.

        Args:
            posting_date_from: Filter by posting date from (ISO format)
            posting_date_to: Filter by posting date to (ISO format)
            plant: Filter by plant code

        Yields:
            Goods Receipt header records as dictionaries
        """
        self._current_entity_set = "A_MaterialDocumentHeader"

        additional_filters = []

        # Filter for goods receipts (movement type 101)
        additional_filters.append("GoodsMovementCode eq '1'")

        if posting_date_from:
            additional_filters.append(f"PostingDate ge datetime'{posting_date_from}'")
        if posting_date_to:
            additional_filters.append(f"PostingDate le datetime'{posting_date_to}'")

        logger.info(f"Extracting Goods Receipts with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters,
            order_by="PostingDate desc"
        )

    def extract_goods_receipt_items(
        self,
        material_document: Optional[str] = None,
        material_document_year: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Goods Receipt items from SAP.

        Args:
            material_document: Filter by material document number
            material_document_year: Filter by material document year

        Yields:
            Goods Receipt item records as dictionaries
        """
        self._current_entity_set = "A_MaterialDocumentItem"

        additional_filters = []

        if material_document:
            additional_filters.append(f"MaterialDocument eq '{material_document}'")
        if material_document_year:
            additional_filters.append(f"MaterialDocumentYear eq '{material_document_year}'")

        logger.info(f"Extracting Goods Receipt Items")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_vendors(
        self,
        country: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Vendor Master data from SAP.

        Args:
            country: Filter by country code (ISO 2-letter)

        Yields:
            Vendor master records as dictionaries
        """
        self._current_entity_set = "A_Supplier"

        additional_filters = []
        if country:
            additional_filters.append(f"Country eq '{country}'")

        logger.info(f"Extracting Vendor Master data")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="Supplier asc"
        )

    def extract_materials(
        self,
        material_type: Optional[str] = None,
        material_group: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Material Master data from SAP.

        Args:
            material_type: Filter by material type
            material_group: Filter by material group

        Yields:
            Material master records as dictionaries
        """
        self._current_entity_set = "A_Material"

        additional_filters = []
        if material_type:
            additional_filters.append(f"MaterialType eq '{material_type}'")
        if material_group:
            additional_filters.append(f"MaterialGroup eq '{material_group}'")

        logger.info(f"Extracting Material Master data")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="Material asc"
        )
