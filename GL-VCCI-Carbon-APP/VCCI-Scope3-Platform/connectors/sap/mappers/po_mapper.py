"""
Purchase Order Mapper

Maps SAP S/4HANA Purchase Order data to VCCI procurement_v1.0.json schema.

This mapper transforms SAP MM Purchase Order and Purchase Order Item data
into the standardized procurement schema used by the VCCI Scope 3 Carbon Platform
for Category 1 (Purchased Goods and Services) emissions calculations.

Field Mappings:
    SAP Field → VCCI Schema Field
    - PurchaseOrder + PurchaseOrderItem → procurement_id (PROC-{PO}-{Item})
    - Vendor → supplier_id_erp
    - VendorName → supplier_name
    - Material / PurchaseOrderItemText → product_name
    - Material → product_code
    - OrderQuantity → quantity
    - PurchaseOrderQuantityUnit → unit
    - NetAmount → spend_usd (with currency conversion)
    - PurchaseOrderDate → transaction_date
    - CompanyCode → metadata.custom_fields.company_code
    - Plant → metadata.custom_fields.plant

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ProcurementRecord(BaseModel):
    """VCCI Procurement data model matching procurement_v1.0.json schema."""

    procurement_id: str
    transaction_date: str
    supplier_name: str
    product_name: str
    quantity: float
    unit: str
    spend_usd: float

    # Optional fields
    tenant_id: Optional[str] = None
    reporting_year: Optional[int] = None
    supplier_id_erp: Optional[str] = None
    supplier_country: Optional[str] = None
    supplier_region: Optional[str] = None
    product_code: Optional[str] = None
    product_category: Optional[str] = None
    spend_currency_original: Optional[str] = None
    spend_amount_original: Optional[float] = None
    exchange_rate_to_usd: Optional[float] = None
    facility_location: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class PurchaseOrderMapper:
    """Maps SAP Purchase Order data to VCCI procurement schema."""

    # SAP to VCCI unit mapping
    UNIT_MAPPING = {
        "KG": "kg",
        "G": "kg",  # Convert grams to kg
        "TO": "tonnes",
        "T": "tonnes",
        "LB": "lbs",
        "L": "liters",
        "GAL": "gallons",
        "M3": "m3",
        "KWH": "kWh",
        "MWH": "MWh",
        "EA": "items",
        "PC": "items",
        "ST": "items",
        "UN": "units",
    }

    # Currency conversion rates (USD base)
    # In production, this would call a currency API or service
    CURRENCY_RATES = {
        "USD": 1.0,
        "EUR": 1.10,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CNY": 0.14,
        "INR": 0.012,
        "CAD": 0.74,
        "AUD": 0.65,
    }

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize Purchase Order mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized PurchaseOrderMapper for tenant: {tenant_id}")

    def _standardize_unit(self, sap_unit: Optional[str]) -> str:
        """Convert SAP unit to VCCI standard unit.

        Args:
            sap_unit: SAP unit of measure

        Returns:
            Standardized VCCI unit
        """
        if not sap_unit:
            return "units"

        sap_unit_upper = sap_unit.upper().strip()
        return self.UNIT_MAPPING.get(sap_unit_upper, "units")

    def _convert_currency(
        self,
        amount: float,
        from_currency: str,
        exchange_rate: Optional[float] = None
    ) -> tuple[float, float]:
        """Convert amount to USD.

        Args:
            amount: Amount in original currency
            from_currency: Source currency code
            exchange_rate: Explicit exchange rate (if available from SAP)

        Returns:
            Tuple of (amount_in_usd, exchange_rate_used)
        """
        if from_currency == "USD":
            return amount, 1.0

        # Use explicit exchange rate from SAP if available
        if exchange_rate and exchange_rate > 0:
            return amount * exchange_rate, exchange_rate

        # Fall back to static rates (in production, use currency API)
        rate = self.CURRENCY_RATES.get(from_currency.upper(), 1.0)
        logger.debug(f"Converting {amount} {from_currency} to USD using rate {rate}")

        return amount * rate, rate

    def _generate_procurement_id(self, po_number: str, item_number: str) -> str:
        """Generate VCCI procurement ID from SAP PO and item.

        Args:
            po_number: SAP Purchase Order number
            item_number: SAP Purchase Order Item number

        Returns:
            VCCI procurement ID (format: PROC-{PO}-{Item})
        """
        # Pad item number to 5 digits
        item_padded = item_number.zfill(5)
        return f"PROC-{po_number}-{item_padded}"

    def _extract_reporting_year(self, transaction_date: str) -> int:
        """Extract reporting year from transaction date.

        Args:
            transaction_date: ISO date string (YYYY-MM-DD)

        Returns:
            Year as integer
        """
        try:
            return int(transaction_date[:4])
        except (ValueError, TypeError):
            return datetime.now().year

    def map_purchase_order(
        self,
        po_header: Dict[str, Any],
        po_item: Dict[str, Any],
        vendor_data: Optional[Dict[str, Any]] = None
    ) -> ProcurementRecord:
        """Map SAP PO header + item to VCCI procurement record.

        Args:
            po_header: SAP Purchase Order header data
            po_item: SAP Purchase Order Item data
            vendor_data: Optional vendor master data for enrichment

        Returns:
            ProcurementRecord matching procurement_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not po_header.get("PurchaseOrder"):
            raise ValueError("Missing required field: PurchaseOrder")
        if not po_item.get("PurchaseOrderItem"):
            raise ValueError("Missing required field: PurchaseOrderItem")

        # Generate procurement ID
        procurement_id = self._generate_procurement_id(
            po_header["PurchaseOrder"],
            po_item["PurchaseOrderItem"]
        )

        # Transaction date
        transaction_date = po_header.get("PurchaseOrderDate", "")
        if not transaction_date:
            transaction_date = datetime.now().strftime("%Y-%m-%d")
            logger.warning(f"PO {po_header['PurchaseOrder']}: Missing PurchaseOrderDate, using today")

        # Supplier information
        supplier_id_erp = po_header.get("Vendor", "")
        supplier_name = po_header.get("VendorName") or vendor_data.get("SupplierName", "") if vendor_data else ""

        if not supplier_name:
            supplier_name = f"Vendor {supplier_id_erp}"
            logger.warning(f"PO {po_header['PurchaseOrder']}: Missing vendor name, using ID")

        # Product information
        product_code = po_item.get("Material", "")
        product_name = po_item.get("PurchaseOrderItemText") or po_item.get("Material", "Unknown Product")

        # Quantity and unit
        quantity = po_item.get("OrderQuantity", 0.0)
        if quantity <= 0:
            logger.warning(f"PO {po_header['PurchaseOrder']} item {po_item['PurchaseOrderItem']}: Invalid quantity")
            quantity = 1.0

        sap_unit = po_item.get("PurchaseOrderQuantityUnit", "")
        unit = self._standardize_unit(sap_unit)

        # Amount and currency conversion
        net_amount = po_item.get("NetAmount", 0.0)
        currency = po_header.get("DocumentCurrency", "USD")
        exchange_rate_sap = po_header.get("ExchangeRate")

        spend_usd, exchange_rate = self._convert_currency(
            net_amount,
            currency,
            exchange_rate_sap
        )

        # Vendor country and region
        vendor_country = None
        vendor_region = None
        if vendor_data:
            vendor_country = vendor_data.get("Country")
            # Map country to region (simplified)
            if vendor_country in ["US", "CA", "MX"]:
                vendor_region = "US"
            elif vendor_country in ["DE", "FR", "GB", "IT", "ES", "NL", "BE"]:
                vendor_region = "EU"
            elif vendor_country in ["CN", "JP", "KR", "IN", "SG", "TH"]:
                vendor_region = "APAC"

        # Metadata
        metadata = {
            "source_system": "SAP_S4HANA",
            "source_document_id": f"PO-{po_header['PurchaseOrder']}-{po_item['PurchaseOrderItem']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "sap-mm-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields (SAP-specific)
        custom_fields = {
            "company_code": po_header.get("CompanyCode"),
            "plant": po_item.get("Plant"),
            "purchasing_organization": po_header.get("PurchasingOrganization"),
            "purchasing_group": po_header.get("PurchasingGroup"),
            "material_group": po_item.get("MaterialGroup"),
            "storage_location": po_item.get("StorageLocation"),
            "cost_center": po_item.get("CostCenter"),
            "gl_account": po_item.get("GLAccount"),
            "payment_terms": po_header.get("PaymentTerms"),
            "incoterms": po_header.get("IncotermsClassification"),
            "delivery_date": po_item.get("DeliveryDate"),
        }

        # Build procurement record
        record = ProcurementRecord(
            procurement_id=procurement_id,
            tenant_id=self.tenant_id,
            transaction_date=transaction_date,
            reporting_year=self._extract_reporting_year(transaction_date),
            supplier_name=supplier_name,
            supplier_id_erp=supplier_id_erp,
            supplier_country=vendor_country,
            supplier_region=vendor_region,
            product_name=product_name,
            product_code=product_code,
            quantity=quantity,
            unit=unit,
            spend_usd=spend_usd,
            spend_currency_original=currency,
            spend_amount_original=net_amount,
            exchange_rate_to_usd=exchange_rate,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped PO {procurement_id}: {supplier_name}, ${spend_usd:.2f}")

        return record

    def map_batch(
        self,
        po_data: List[Dict[str, Any]],
        vendor_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ProcurementRecord]:
        """Map a batch of PO records.

        Args:
            po_data: List of dicts containing 'header' and 'item' keys
            vendor_lookup: Optional dict mapping vendor ID to vendor master data

        Returns:
            List of mapped ProcurementRecord objects
        """
        records = []
        vendor_lookup = vendor_lookup or {}

        for po in po_data:
            try:
                header = po.get("header", {})
                item = po.get("item", {})
                vendor_id = header.get("Vendor")
                vendor_data = vendor_lookup.get(vendor_id) if vendor_id else None

                record = self.map_purchase_order(header, item, vendor_data)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping PO record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(po_data)} PO records")
        return records
