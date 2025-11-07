"""
Purchase Requisition Mapper

Maps Oracle Fusion Procurement Cloud Purchase Requisition data to VCCI procurement_v1.0.json schema.

This mapper transforms Oracle Purchase Requisition and Purchase Requisition Line data
into the standardized procurement schema. Requisitions represent pre-purchase requests
that may or may not be converted to Purchase Orders.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RequisitionMapper:
    """Maps Oracle Purchase Requisition data to VCCI procurement schema."""

    # Unit mapping (same as PO mapper)
    UNIT_MAPPING = {
        "KG": "kg",
        "G": "kg",
        "TON": "tonnes",
        "MT": "tonnes",
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

    # Currency conversion rates
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
        """Initialize Requisition mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized RequisitionMapper for tenant: {tenant_id}")

    def _standardize_unit(self, oracle_unit: Optional[str]) -> str:
        """Convert Oracle unit to VCCI standard unit."""
        if not oracle_unit:
            return "units"
        oracle_unit_upper = oracle_unit.upper().strip()
        return self.UNIT_MAPPING.get(oracle_unit_upper, "units")

    def _convert_currency(self, amount: float, from_currency: str) -> tuple[float, float]:
        """Convert amount to USD."""
        if from_currency == "USD":
            return amount, 1.0
        rate = self.CURRENCY_RATES.get(from_currency.upper(), 1.0)
        return amount * rate, rate

    def _generate_procurement_id(self, req_header_id: int, line_number: int) -> str:
        """Generate VCCI procurement ID from Oracle Requisition header ID and line number."""
        line_padded = str(line_number).zfill(5)
        return f"PROC-REQ-{req_header_id}-{line_padded}"

    def _extract_reporting_year(self, transaction_date: str) -> int:
        """Extract reporting year from transaction date."""
        try:
            return int(transaction_date[:4])
        except (ValueError, TypeError):
            return datetime.now().year

    def map_purchase_requisition(
        self,
        req_header: Dict[str, Any],
        req_line: Dict[str, Any],
        supplier_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Map Oracle Requisition header + line to VCCI procurement record.

        Args:
            req_header: Oracle Purchase Requisition header data
            req_line: Oracle Purchase Requisition Line data
            supplier_data: Optional suggested supplier master data

        Returns:
            Procurement record dictionary matching procurement_v1.0.json schema
        """
        # Generate procurement ID
        procurement_id = self._generate_procurement_id(
            req_header["RequisitionHeaderId"],
            req_line["LineNumber"]
        )

        # Transaction date (use creation date for requisitions)
        transaction_date = req_header.get("CreationDate", "")
        if not transaction_date:
            transaction_date = datetime.now().strftime("%Y-%m-%d")

        # Supplier information (may be suggested, not final)
        supplier_id_erp = str(req_line.get("SuggestedSupplierId", "")) if req_line.get("SuggestedSupplierId") else ""
        supplier_name = req_line.get("SuggestedSupplierName", "")

        if not supplier_name and supplier_data:
            supplier_name = supplier_data.get("SupplierName", "")

        if not supplier_name:
            supplier_name = "Unknown Supplier (Requisition)"

        # Product information
        product_code = req_line.get("ItemNumber", "")
        product_name = req_line.get("ItemDescription") or req_line.get("ItemNumber", "Unknown Product")

        # Quantity and unit
        quantity = req_line.get("Quantity", 1.0)
        if quantity <= 0:
            quantity = 1.0

        unit = self._standardize_unit(req_line.get("UOM", ""))

        # Amount and currency conversion
        amount = req_line.get("Amount", 0.0)
        currency = req_line.get("Currency", "USD")
        spend_usd, exchange_rate = self._convert_currency(amount, currency)

        # Metadata
        metadata = {
            "source_system": "Oracle_Fusion",
            "source_document_id": f"REQ-{req_header['RequisitionNumber']}-{req_line['LineNumber']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": True,  # Requisitions may need review
            "manual_review_reason": "Purchase Requisition - may not be converted to PO",
            "created_by": "oracle-procurement-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields
        custom_fields = {
            "document_type": "REQUISITION",
            "requisition_header_id": req_header.get("RequisitionHeaderId"),
            "requisition_number": req_header.get("RequisitionNumber"),
            "requisition_line_id": req_line.get("RequisitionLineId"),
            "business_unit": req_header.get("BU"),
            "business_unit_name": req_header.get("BUName"),
            "preparer_id": req_header.get("PreparerId"),
            "preparer_name": req_header.get("PreparerName"),
            "document_status": req_header.get("DocumentStatus"),
            "justification": req_header.get("JustificationText"),
            "category_id": req_line.get("CategoryId"),
            "category_name": req_line.get("CategoryName"),
            "need_by_date": req_line.get("NeedByDate"),
            "deliver_to_location": req_line.get("DeliverToLocationId"),
            "destination_type": req_line.get("DestinationTypeCode"),
        }

        # Build procurement record
        record = {
            "procurement_id": procurement_id,
            "tenant_id": self.tenant_id,
            "transaction_date": transaction_date,
            "reporting_year": self._extract_reporting_year(transaction_date),
            "supplier_name": supplier_name,
            "supplier_id_erp": supplier_id_erp,
            "product_name": product_name,
            "product_code": product_code,
            "quantity": quantity,
            "unit": unit,
            "spend_usd": spend_usd,
            "spend_currency_original": currency,
            "spend_amount_original": amount,
            "exchange_rate_to_usd": exchange_rate,
            "metadata": metadata,
            "custom_fields": custom_fields,
        }

        logger.debug(f"Mapped Requisition {procurement_id}: {supplier_name}, ${spend_usd:.2f}")

        return record

    def map_batch(
        self,
        requisition_data: List[Dict[str, Any]],
        supplier_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Map a batch of Requisition records."""
        records = []
        supplier_lookup = supplier_lookup or {}

        for req in requisition_data:
            try:
                header = req.get("header", {})
                line = req.get("line", {})
                supplier_id = str(line.get("SuggestedSupplierId", ""))
                supplier_data = supplier_lookup.get(supplier_id) if supplier_id else None

                record = self.map_purchase_requisition(header, line, supplier_data)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping Requisition record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(requisition_data)} Requisition records")
        return records
