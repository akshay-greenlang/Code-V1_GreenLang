# -*- coding: utf-8 -*-
"""
Purchase Order Mapper

Maps Oracle Fusion Procurement Cloud Purchase Order data to VCCI procurement_v1.0.json schema.

This mapper transforms Oracle Purchase Order and Purchase Order Line data
into the standardized procurement schema used by the VCCI Scope 3 Carbon Platform
for Category 1 (Purchased Goods and Services) emissions calculations.

Field Mappings:
    Oracle Field → VCCI Schema Field
    - POHeaderId + LineNumber → procurement_id (PROC-{POHeaderId}-{LineNumber})
    - SupplierId → supplier_id_erp
    - SupplierName → supplier_name
    - ItemDescription → product_name
    - ItemNumber → product_code
    - Quantity → quantity
    - UOM → unit
    - LineAmount → spend_usd (with currency conversion)
    - OrderedDate → transaction_date
    - BU (Business Unit) → custom_fields.business_unit

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.determinism import DeterministicClock

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
    """Maps Oracle Purchase Order data to VCCI procurement schema."""

    # Oracle to VCCI unit mapping
    UNIT_MAPPING = {
        "KG": "kg",
        "G": "kg",  # Convert grams to kg
        "TON": "tonnes",
        "MT": "tonnes",  # Metric tons
        "LB": "lbs",
        "L": "liters",
        "LTR": "liters",
        "GAL": "gallons",
        "M3": "m3",
        "KWH": "kWh",
        "MWH": "MWh",
        "EA": "items",
        "EACH": "items",
        "PC": "items",
        "PCS": "items",
        "UN": "units",
        "UNIT": "units",
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

    def _standardize_unit(self, oracle_unit: Optional[str]) -> str:
        """Convert Oracle unit to VCCI standard unit.

        Args:
            oracle_unit: Oracle unit of measure

        Returns:
            Standardized VCCI unit
        """
        if not oracle_unit:
            return "units"

        oracle_unit_upper = oracle_unit.upper().strip()
        return self.UNIT_MAPPING.get(oracle_unit_upper, "units")

    def _convert_currency(
        self,
        amount: float,
        from_currency: str
    ) -> tuple[float, float]:
        """Convert amount to USD.

        Args:
            amount: Amount in original currency
            from_currency: Source currency code

        Returns:
            Tuple of (amount_in_usd, exchange_rate_used)
        """
        if from_currency == "USD":
            return amount, 1.0

        # Use static rates (in production, use currency API)
        rate = self.CURRENCY_RATES.get(from_currency.upper(), 1.0)
        logger.debug(f"Converting {amount} {from_currency} to USD using rate {rate}")

        return amount * rate, rate

    def _generate_procurement_id(self, po_header_id: int, line_number: int) -> str:
        """Generate VCCI procurement ID from Oracle PO header ID and line number.

        Args:
            po_header_id: Oracle PO Header ID
            line_number: Oracle PO Line Number

        Returns:
            VCCI procurement ID (format: PROC-{POHeaderId}-{LineNumber})
        """
        # Pad line number to 5 digits
        line_padded = str(line_number).zfill(5)
        return f"PROC-{po_header_id}-{line_padded}"

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
            return DeterministicClock.now().year

    def map_purchase_order(
        self,
        po_header: Dict[str, Any],
        po_line: Dict[str, Any],
        supplier_data: Optional[Dict[str, Any]] = None
    ) -> ProcurementRecord:
        """Map Oracle PO header + line to VCCI procurement record.

        Args:
            po_header: Oracle Purchase Order header data
            po_line: Oracle Purchase Order Line data
            supplier_data: Optional supplier master data for enrichment

        Returns:
            ProcurementRecord matching procurement_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not po_header.get("POHeaderId"):
            raise ValueError("Missing required field: POHeaderId")
        if not po_line.get("LineNumber"):
            raise ValueError("Missing required field: LineNumber")

        # Generate procurement ID
        procurement_id = self._generate_procurement_id(
            po_header["POHeaderId"],
            po_line["LineNumber"]
        )

        # Transaction date
        transaction_date = po_header.get("OrderedDate", "")
        if not transaction_date:
            transaction_date = DeterministicClock.now().strftime("%Y-%m-%d")
            logger.warning(f"PO {po_header['POHeaderId']}: Missing OrderedDate, using today")

        # Supplier information
        supplier_id_erp = str(po_header.get("SupplierId", ""))
        supplier_name = po_header.get("SupplierName") or (
            supplier_data.get("SupplierName", "") if supplier_data else ""
        )

        if not supplier_name:
            supplier_name = f"Supplier {supplier_id_erp}"
            logger.warning(f"PO {po_header['POHeaderId']}: Missing supplier name, using ID")

        # Product information
        product_code = po_line.get("ItemNumber", "")
        product_name = po_line.get("ItemDescription") or po_line.get("ItemNumber", "Unknown Product")

        # Quantity and unit
        quantity = po_line.get("Quantity", 0.0)
        if quantity <= 0:
            logger.warning(f"PO {po_header['POHeaderId']} line {po_line['LineNumber']}: Invalid quantity")
            quantity = 1.0

        oracle_unit = po_line.get("UOM", "")
        unit = self._standardize_unit(oracle_unit)

        # Amount and currency conversion
        line_amount = po_line.get("LineAmount", 0.0)
        currency = po_header.get("Currency", "USD")

        spend_usd, exchange_rate = self._convert_currency(line_amount, currency)

        # Supplier country and region
        vendor_country = None
        vendor_region = None
        if supplier_data:
            vendor_country = supplier_data.get("Country")
            # Map country to region (simplified)
            if vendor_country in ["US", "CA", "MX"]:
                vendor_region = "US"
            elif vendor_country in ["DE", "FR", "GB", "IT", "ES", "NL", "BE"]:
                vendor_region = "EU"
            elif vendor_country in ["CN", "JP", "KR", "IN", "SG", "TH"]:
                vendor_region = "APAC"

        # Metadata
        metadata = {
            "source_system": "Oracle_Fusion",
            "source_document_id": f"PO-{po_header['OrderNumber']}-{po_line['LineNumber']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "oracle-procurement-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields (Oracle-specific)
        custom_fields = {
            "po_header_id": po_header.get("POHeaderId"),
            "po_number": po_header.get("OrderNumber"),
            "po_line_id": po_line.get("POLineId"),
            "business_unit": po_header.get("BU"),
            "business_unit_name": po_header.get("BUName"),
            "buyer_id": po_header.get("BuyerId"),
            "buyer_name": po_header.get("BuyerName"),
            "document_status": po_header.get("DocumentStatus"),
            "category_id": po_line.get("CategoryId"),
            "category_name": po_line.get("CategoryName"),
            "ship_to_location": po_line.get("ShipToLocationCode"),
            "payment_terms": po_header.get("PaymentTerms"),
            "freight_terms": po_header.get("FreightTerms"),
            "need_by_date": po_line.get("NeedByDate"),
            "promised_date": po_line.get("PromisedDate"),
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
            spend_amount_original=line_amount,
            exchange_rate_to_usd=exchange_rate,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped PO {procurement_id}: {supplier_name}, ${spend_usd:.2f}")

        return record

    def map_batch(
        self,
        po_data: List[Dict[str, Any]],
        supplier_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ProcurementRecord]:
        """Map a batch of PO records.

        Args:
            po_data: List of dicts containing 'header' and 'line' keys
            supplier_lookup: Optional dict mapping supplier ID to supplier master data

        Returns:
            List of mapped ProcurementRecord objects
        """
        records = []
        supplier_lookup = supplier_lookup or {}

        for po in po_data:
            try:
                header = po.get("header", {})
                line = po.get("line", {})
                supplier_id = str(header.get("SupplierId", ""))
                supplier_data = supplier_lookup.get(supplier_id) if supplier_id else None

                record = self.map_purchase_order(header, line, supplier_data)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping PO record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(po_data)} PO records")
        return records
