# -*- coding: utf-8 -*-
"""
Delivery Mapper

Maps SAP S/4HANA Outbound Delivery data to VCCI logistics_v1.0.json schema.

This mapper transforms SAP SD Outbound Delivery data into the standardized
logistics schema for tracking outbound shipments and their associated
Scope 3 Category 9 (Downstream Transportation) emissions.

Field Mappings:
    SAP Field → VCCI Schema Field
    - OutboundDelivery → shipment_id (SHIP-OD-{Delivery})
    - DeliveryDate / ActualDeliveryDate → shipment_date
    - ShippingPoint → origin.location_name
    - ShipToParty / ReceivingPlant → destination.location_name
    - TotalWeight → weight_tonnes
    - Route → custom_fields.route
    - MeansOfTransport → transport_mode mapping

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


class DeliveryMapper:
    """Maps SAP Outbound Delivery data to VCCI logistics schema."""

    # SAP transport type to VCCI transport mode mapping
    TRANSPORT_MODE_MAPPING = {
        "01": "Road_Truck_GreaterThan17t",  # Road freight
        "02": "Rail_Freight",  # Rail
        "03": "Sea_Freight_Container",  # Sea
        "04": "Air_Freight_ShortHaul",  # Air
        "05": "Road_Truck_7.5to17t",  # Small truck
        "06": "Pipeline",  # Pipeline
    }

    # Weight unit conversion to tonnes
    WEIGHT_CONVERSION = {
        "KG": 0.001,
        "G": 0.000001,
        "TO": 1.0,
        "T": 1.0,
        "LB": 0.000453592,
    }

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize Delivery mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized DeliveryMapper for tenant: {tenant_id}")

    def _generate_shipment_id(self, outbound_delivery: str) -> str:
        """Generate VCCI shipment ID from SAP outbound delivery.

        Args:
            outbound_delivery: SAP Outbound Delivery number

        Returns:
            VCCI shipment ID (format: SHIP-OD-{Delivery})
        """
        return f"SHIP-OD-{outbound_delivery}"

    def _map_transport_mode(
        self,
        shipping_type: Optional[str],
        means_of_transport_type: Optional[str]
    ) -> str:
        """Map SAP transport type to VCCI transport mode.

        Args:
            shipping_type: SAP shipping type code
            means_of_transport_type: SAP means of transport type

        Returns:
            VCCI transport mode
        """
        # Try to map from shipping type first
        if shipping_type:
            mode = self.TRANSPORT_MODE_MAPPING.get(shipping_type)
            if mode:
                return mode

        # Try means of transport type
        if means_of_transport_type:
            mode = self.TRANSPORT_MODE_MAPPING.get(means_of_transport_type)
            if mode:
                return mode

        # Default to road truck
        logger.debug(f"Unknown transport type, defaulting to Road_Truck_GreaterThan17t")
        return "Road_Truck_GreaterThan17t"

    def _convert_weight_to_tonnes(
        self,
        weight: float,
        weight_unit: str
    ) -> float:
        """Convert weight to tonnes.

        Args:
            weight: Weight value
            weight_unit: SAP weight unit

        Returns:
            Weight in tonnes
        """
        if not weight or weight <= 0:
            return 0.0

        conversion = self.WEIGHT_CONVERSION.get(weight_unit.upper(), 1.0)
        return weight * conversion

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

    def map_outbound_delivery(
        self,
        delivery_header: Dict[str, Any],
        delivery_items: Optional[List[Dict[str, Any]]] = None,
        shipping_point_data: Optional[Dict[str, Any]] = None,
        customer_data: Optional[Dict[str, Any]] = None
    ) -> LogisticsRecord:
        """Map SAP Outbound Delivery to VCCI logistics record.

        Args:
            delivery_header: SAP Outbound Delivery header data
            delivery_items: Optional list of delivery item data
            shipping_point_data: Optional shipping point master data
            customer_data: Optional customer master data for destination

        Returns:
            LogisticsRecord matching logistics_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not delivery_header.get("OutboundDelivery"):
            raise ValueError("Missing required field: OutboundDelivery")

        # Generate shipment ID
        shipment_id = self._generate_shipment_id(delivery_header["OutboundDelivery"])

        # Shipment date (prefer actual over planned)
        shipment_date = delivery_header.get("ActualDeliveryDate") or delivery_header.get("DeliveryDate", "")
        if not shipment_date:
            shipment_date = DeterministicClock.now().strftime("%Y-%m-%d")
            logger.warning(f"Delivery {delivery_header['OutboundDelivery']}: Missing delivery date, using today")

        # Transport mode
        transport_mode = self._map_transport_mode(
            delivery_header.get("ShippingType"),
            delivery_header.get("MeansOfTransportType")
        )

        # Origin (shipping point)
        shipping_point = delivery_header.get("ShippingPoint", "")
        origin = {
            "location_name": shipping_point_data.get("ShippingPointName", f"Shipping Point {shipping_point}") if shipping_point_data else f"Shipping Point {shipping_point}",
            "city": shipping_point_data.get("CityName") if shipping_point_data else None,
            "country": shipping_point_data.get("Country") if shipping_point_data else None,
        }

        # Destination (ship-to party)
        ship_to = delivery_header.get("ShipToParty", "")
        destination = None
        if ship_to and customer_data:
            destination = {
                "location_name": customer_data.get("CustomerName", delivery_header.get("ShipToPartyName", f"Customer {ship_to}")),
                "city": customer_data.get("CityName"),
                "country": customer_data.get("Country"),
                "postal_code": customer_data.get("PostalCode"),
            }
        elif delivery_header.get("ReceivingPlant"):
            destination = {
                "location_name": delivery_header.get("ReceivingPlantName", f"Plant {delivery_header['ReceivingPlant']}"),
            }

        # Weight
        total_weight = delivery_header.get("TotalWeight", 0.0)
        weight_unit = delivery_header.get("WeightUnit", "KG")
        weight_tonnes = self._convert_weight_to_tonnes(total_weight, weight_unit)
        weight_source = "Delivery_Document" if total_weight > 0 else "Estimated"

        # Carrier information
        carrier_info = None
        if delivery_header.get("Route"):
            carrier_info = {
                "carrier_name": delivery_header.get("RouteDescription", f"Route {delivery_header['Route']}"),
            }

        # Metadata
        metadata = {
            "source_system": "SAP_S4HANA",
            "source_document_id": f"ODN-{delivery_header['OutboundDelivery']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "sap-sd-delivery-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields
        custom_fields = {
            "outbound_delivery": delivery_header["OutboundDelivery"],
            "delivery_type": delivery_header.get("DeliveryDocumentType"),
            "shipping_point": shipping_point,
            "ship_to_party": ship_to,
            "sold_to_party": delivery_header.get("SoldToParty"),
            "receiving_plant": delivery_header.get("ReceivingPlant"),
            "route": delivery_header.get("Route"),
            "shipping_condition": delivery_header.get("ShippingCondition"),
            "incoterms": delivery_header.get("IncotermsClassification"),
            "means_of_transport": delivery_header.get("MeansOfTransport"),
            "total_volume": delivery_header.get("TotalVolume"),
            "volume_unit": delivery_header.get("VolumeUnit"),
        }

        # Add item details if provided
        if delivery_items:
            custom_fields["item_count"] = len(delivery_items)
            custom_fields["materials"] = [
                item.get("Material") for item in delivery_items if item.get("Material")
            ]

        # Build logistics record
        record = LogisticsRecord(
            shipment_id=shipment_id,
            tenant_id=self.tenant_id,
            shipment_date=shipment_date,
            reporting_year=self._extract_reporting_year(shipment_date),
            transport_mode=transport_mode,
            calculation_method="distance_based",  # Will need distance/route data
            origin=origin,
            destination=destination,
            weight_tonnes=weight_tonnes,
            weight_source=weight_source,
            carrier_information=carrier_info,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped Delivery {shipment_id}: {weight_tonnes:.2f} tonnes via {transport_mode}")

        return record

    def map_batch(
        self,
        delivery_data: List[Dict[str, Any]],
        shipping_point_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        customer_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[LogisticsRecord]:
        """Map a batch of Outbound Delivery records.

        Args:
            delivery_data: List of dicts containing 'header' and optional 'items' keys
            shipping_point_lookup: Optional dict mapping shipping point to location data
            customer_lookup: Optional dict mapping customer ID to customer data

        Returns:
            List of mapped LogisticsRecord objects
        """
        records = []
        shipping_point_lookup = shipping_point_lookup or {}
        customer_lookup = customer_lookup or {}

        for delivery in delivery_data:
            try:
                header = delivery.get("header", {})
                items = delivery.get("items", [])
                shipping_point = header.get("ShippingPoint")
                ship_to = header.get("ShipToParty")

                shipping_point_data = shipping_point_lookup.get(shipping_point) if shipping_point else None
                customer_data = customer_lookup.get(ship_to) if ship_to else None

                record = self.map_outbound_delivery(
                    header, items, shipping_point_data, customer_data
                )
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping Delivery record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(delivery_data)} Delivery records")
        return records
