# -*- coding: utf-8 -*-
"""
Goods Receipt Mapper

Maps SAP S/4HANA Goods Receipt data to VCCI logistics_v1.0.json schema.

This mapper transforms SAP MM Material Document (Goods Receipt) data
into the standardized logistics schema for tracking inbound shipments
and their associated Scope 3 Category 4 (Upstream Transportation) emissions.

Field Mappings:
    SAP Field → VCCI Schema Field
    - MaterialDocument + MaterialDocumentYear → shipment_id (SHIP-GR-{Doc}-{Year})
    - PostingDate → shipment_date
    - Plant → destination.location_name
    - Supplier → carrier_information.carrier_name (if applicable)
    - QuantityInBaseUnit → weight_tonnes (estimated from quantity)
    - DeliveryNote → metadata.source_document_id

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class LogisticsRecord(BaseModel):
    """VCCI Logistics data model matching logistics_v1.0.json schema."""

    shipment_id: str
    shipment_date: str
    transport_mode: str
    calculation_method: str

    # Optional fields
    tenant_id: Optional[str] = None
    reporting_year: Optional[int] = None
    origin: Optional[Dict[str, Any]] = None
    destination: Optional[Dict[str, Any]] = None
    distance_km: Optional[float] = None
    distance_source: Optional[str] = None
    weight_tonnes: Optional[float] = None
    weight_source: Optional[str] = None
    carrier_information: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class GoodsReceiptMapper:
    """Maps SAP Goods Receipt data to VCCI logistics schema."""

    # Unit to kg conversion factors
    WEIGHT_CONVERSION = {
        "KG": 1.0,
        "G": 0.001,
        "TO": 1000.0,
        "T": 1000.0,
        "LB": 0.453592,
    }

    # Default weight estimation by material type (kg per unit)
    # This is a fallback when actual weight is not available
    DEFAULT_WEIGHTS = {
        "ROH": 100.0,  # Raw materials
        "HALB": 50.0,  # Semi-finished goods
        "FERT": 25.0,  # Finished goods
        "HAWA": 10.0,  # Trading goods
    }

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize Goods Receipt mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized GoodsReceiptMapper for tenant: {tenant_id}")

    def _generate_shipment_id(self, material_doc: str, doc_year: str) -> str:
        """Generate VCCI shipment ID from SAP material document.

        Args:
            material_doc: SAP Material Document number
            doc_year: Material Document year

        Returns:
            VCCI shipment ID (format: SHIP-GR-{Doc}-{Year})
        """
        return f"SHIP-GR-{material_doc}-{doc_year}"

    def _estimate_weight_tonnes(
        self,
        quantity: float,
        unit: str,
        material_type: Optional[str] = None,
        actual_weight: Optional[float] = None,
        weight_unit: Optional[str] = None
    ) -> tuple[float, str]:
        """Estimate shipment weight in tonnes.

        Args:
            quantity: Quantity received
            unit: Unit of measure
            material_type: SAP material type (for estimation)
            actual_weight: Actual weight from material master
            weight_unit: Unit of actual weight

        Returns:
            Tuple of (weight_in_tonnes, source)
        """
        # Use actual weight if available
        if actual_weight and actual_weight > 0 and weight_unit:
            conversion = self.WEIGHT_CONVERSION.get(weight_unit.upper(), 1.0)
            weight_kg = actual_weight * conversion
            return weight_kg / 1000.0, "Material_Master"

        # Estimate based on quantity and material type
        default_kg_per_unit = self.DEFAULT_WEIGHTS.get(material_type or "FERT", 25.0)
        estimated_kg = quantity * default_kg_per_unit

        logger.debug(
            f"Estimated weight: {quantity} {unit} x {default_kg_per_unit} kg/unit = {estimated_kg} kg"
        )

        return estimated_kg / 1000.0, "Estimated"

    def _extract_reporting_year(self, shipment_date: str) -> int:
        """Extract reporting year from shipment date.

        Args:
            shipment_date: ISO date string (YYYY-MM-DD)

        Returns:
            Year as integer
        """
        try:
            return int(shipment_date[:4])
        except (ValueError, TypeError):
            return DeterministicClock.now().year

    def map_goods_receipt(
        self,
        gr_header: Dict[str, Any],
        gr_item: Dict[str, Any],
        plant_data: Optional[Dict[str, Any]] = None,
        supplier_data: Optional[Dict[str, Any]] = None,
        material_data: Optional[Dict[str, Any]] = None
    ) -> LogisticsRecord:
        """Map SAP Goods Receipt to VCCI logistics record.

        Args:
            gr_header: SAP Material Document header data
            gr_item: SAP Material Document Item data
            plant_data: Optional plant master data for destination info
            supplier_data: Optional supplier data for origin info
            material_data: Optional material master data for weight

        Returns:
            LogisticsRecord matching logistics_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not gr_header.get("MaterialDocument"):
            raise ValueError("Missing required field: MaterialDocument")
        if not gr_header.get("MaterialDocumentYear"):
            raise ValueError("Missing required field: MaterialDocumentYear")

        # Generate shipment ID
        shipment_id = self._generate_shipment_id(
            gr_header["MaterialDocument"],
            gr_header["MaterialDocumentYear"]
        )

        # Shipment date
        shipment_date = gr_header.get("PostingDate", "")
        if not shipment_date:
            shipment_date = DeterministicClock.now().strftime("%Y-%m-%d")
            logger.warning(f"GR {gr_header['MaterialDocument']}: Missing PostingDate, using today")

        # Destination (receiving plant)
        plant = gr_item.get("Plant", "")
        destination = {
            "location_name": plant_data.get("PlantName", f"Plant {plant}") if plant_data else f"Plant {plant}",
            "city": plant_data.get("CityName") if plant_data else None,
            "country": plant_data.get("Country") if plant_data else None,
            "postal_code": plant_data.get("PostalCode") if plant_data else None,
        }

        # Origin (supplier location)
        supplier = gr_item.get("Supplier", "")
        origin = None
        if supplier and supplier_data:
            origin = {
                "location_name": supplier_data.get("SupplierName", f"Supplier {supplier}"),
                "city": supplier_data.get("CityName"),
                "country": supplier_data.get("Country"),
                "postal_code": supplier_data.get("PostalCode"),
            }

        # Weight estimation
        quantity = gr_item.get("QuantityInBaseUnit", 0.0)
        unit = gr_item.get("MaterialBaseUnit", "")
        material_type = material_data.get("MaterialType") if material_data else None
        actual_weight = material_data.get("NetWeight") if material_data else None
        weight_unit = material_data.get("WeightUnit") if material_data else None

        weight_tonnes, weight_source = self._estimate_weight_tonnes(
            quantity, unit, material_type, actual_weight, weight_unit
        )

        # Carrier information (if available from delivery note or shipment)
        carrier_info = None
        if gr_item.get("ShipmentNumber"):
            carrier_info = {
                "carrier_name": f"Shipment {gr_item['ShipmentNumber']}",
            }

        # Metadata
        metadata = {
            "source_system": "SAP_S4HANA",
            "source_document_id": f"MIGO-{gr_header['MaterialDocument']}-{gr_header['MaterialDocumentYear']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "sap-mm-gr-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields
        custom_fields = {
            "material_document": gr_header["MaterialDocument"],
            "material_document_year": gr_header["MaterialDocumentYear"],
            "material": gr_item.get("Material"),
            "plant": plant,
            "storage_location": gr_item.get("StorageLocation"),
            "batch": gr_item.get("Batch"),
            "purchase_order": gr_item.get("PurchaseOrder"),
            "purchase_order_item": gr_item.get("PurchaseOrderItem"),
            "goods_movement_type": gr_item.get("GoodsMovementType"),
            "delivery_note": gr_item.get("DeliveryNote"),
            "shipment_number": gr_item.get("ShipmentNumber"),
        }

        # Build logistics record
        record = LogisticsRecord(
            shipment_id=shipment_id,
            tenant_id=self.tenant_id,
            shipment_date=shipment_date,
            reporting_year=self._extract_reporting_year(shipment_date),
            transport_mode="Road_Truck_GreaterThan17t",  # Default assumption
            calculation_method="distance_based",  # Will need distance data
            origin=origin,
            destination=destination,
            weight_tonnes=weight_tonnes,
            weight_source=weight_source,
            carrier_information=carrier_info,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped GR {shipment_id}: {weight_tonnes:.2f} tonnes to {plant}")

        return record

    def map_batch(
        self,
        gr_data: List[Dict[str, Any]],
        plant_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        supplier_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        material_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[LogisticsRecord]:
        """Map a batch of Goods Receipt records.

        Args:
            gr_data: List of dicts containing 'header' and 'item' keys
            plant_lookup: Optional dict mapping plant code to plant data
            supplier_lookup: Optional dict mapping supplier ID to supplier data
            material_lookup: Optional dict mapping material ID to material data

        Returns:
            List of mapped LogisticsRecord objects
        """
        records = []
        plant_lookup = plant_lookup or {}
        supplier_lookup = supplier_lookup or {}
        material_lookup = material_lookup or {}

        for gr in gr_data:
            try:
                header = gr.get("header", {})
                item = gr.get("item", {})
                plant = item.get("Plant")
                supplier = item.get("Supplier")
                material = item.get("Material")

                plant_data = plant_lookup.get(plant) if plant else None
                supplier_data = supplier_lookup.get(supplier) if supplier else None
                material_data = material_lookup.get(material) if material else None

                record = self.map_goods_receipt(
                    header, item, plant_data, supplier_data, material_data
                )
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping GR record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(gr_data)} GR records")
        return records
