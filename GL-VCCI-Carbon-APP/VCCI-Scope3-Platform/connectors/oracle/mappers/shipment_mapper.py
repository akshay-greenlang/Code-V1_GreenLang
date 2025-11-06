"""
Shipment Mapper

Maps Oracle Fusion SCM Cloud Shipment data to VCCI logistics_v1.0.json schema.

This mapper transforms Oracle Shipment data into the standardized logistics schema
for tracking shipments and their associated Scope 3 Category 4 (Upstream Transportation)
emissions.

Field Mappings:
    Oracle Field → VCCI Schema Field
    - ShipmentId → shipment_id (SHIP-{ShipmentId})
    - ShipmentDate → shipment_date
    - TransportMode → transport_mode
    - OriginLocation → origin
    - DestinationLocation → destination
    - ShipmentWeight → weight_tonnes
    - FreightCost → spend_usd

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

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
    spend_usd: Optional[float] = None
    spend_currency_original: Optional[str] = None
    spend_amount_original: Optional[float] = None
    exchange_rate_to_usd: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ShipmentMapper:
    """Maps Oracle Shipment data to VCCI logistics schema."""

    # Oracle transport mode to VCCI transport mode mapping
    TRANSPORT_MODE_MAPPING = {
        "TRUCK": "Road_Truck_GreaterThan17t",
        "LTL": "Road_Truck_7.5to17t",  # Less Than Truckload
        "FTL": "Road_Truck_GreaterThan17t",  # Full Truckload
        "RAIL": "Rail_Freight",
        "SEA": "Sea_Freight_Container",
        "OCEAN": "Sea_Freight_Container",
        "AIR": "Air_Freight_LongHaul",
        "PARCEL": "Road_Truck_LessThan7.5t",
        "COURIER": "Road_Truck_LessThan7.5t",
    }

    # Weight unit conversion to tonnes
    WEIGHT_CONVERSION = {
        "KG": 0.001,
        "G": 0.000001,
        "TON": 1.0,
        "MT": 1.0,
        "T": 1.0,
        "LB": 0.000453592,
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
        """Initialize Shipment mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized ShipmentMapper for tenant: {tenant_id}")

    def _generate_shipment_id(self, shipment_id: int) -> str:
        """Generate VCCI shipment ID from Oracle shipment ID."""
        return f"SHIP-{shipment_id}"

    def _map_transport_mode(self, oracle_mode: Optional[str]) -> str:
        """Map Oracle transport mode to VCCI transport mode."""
        if not oracle_mode:
            logger.debug("Missing transport mode, defaulting to Road_Truck_GreaterThan17t")
            return "Road_Truck_GreaterThan17t"

        mode = self.TRANSPORT_MODE_MAPPING.get(oracle_mode.upper())
        if mode:
            return mode

        logger.debug(f"Unknown transport mode '{oracle_mode}', defaulting to Road_Truck_GreaterThan17t")
        return "Road_Truck_GreaterThan17t"

    def _convert_weight_to_tonnes(self, weight: float, weight_unit: str) -> float:
        """Convert weight to tonnes."""
        if not weight or weight <= 0:
            return 0.0

        conversion = self.WEIGHT_CONVERSION.get(weight_unit.upper(), 1.0)
        return weight * conversion

    def _convert_currency(self, amount: float, from_currency: str) -> tuple[float, float]:
        """Convert amount to USD."""
        if from_currency == "USD":
            return amount, 1.0
        rate = self.CURRENCY_RATES.get(from_currency.upper(), 1.0)
        return amount * rate, rate

    def _extract_reporting_year(self, shipment_date: str) -> int:
        """Extract reporting year from shipment date."""
        try:
            return int(shipment_date[:4])
        except (ValueError, TypeError):
            return datetime.now().year

    def map_shipment(
        self,
        shipment_header: Dict[str, Any],
        shipment_lines: Optional[List[Dict[str, Any]]] = None
    ) -> LogisticsRecord:
        """Map Oracle Shipment to VCCI logistics record.

        Args:
            shipment_header: Oracle Shipment header data
            shipment_lines: Optional list of shipment line data

        Returns:
            LogisticsRecord matching logistics_v1.0.json schema
        """
        # Generate shipment ID
        shipment_id = self._generate_shipment_id(shipment_header["ShipmentId"])

        # Shipment date (prefer actual over planned)
        shipment_date = shipment_header.get("ActualDeliveryDate") or shipment_header.get("ShipmentDate", "")
        if not shipment_date:
            shipment_date = datetime.now().strftime("%Y-%m-%d")

        # Transport mode
        transport_mode = self._map_transport_mode(shipment_header.get("TransportMode"))

        # Origin location
        origin = {
            "location_name": shipment_header.get("OriginLocationCode", "Unknown Origin"),
            "city": shipment_header.get("OriginCity"),
            "state_province": shipment_header.get("OriginState"),
            "country": shipment_header.get("OriginCountry"),
            "postal_code": shipment_header.get("OriginPostalCode"),
        }

        # Destination location
        destination = {
            "location_name": shipment_header.get("DestinationLocationCode", "Unknown Destination"),
            "city": shipment_header.get("DestinationCity"),
            "state_province": shipment_header.get("DestinationState"),
            "country": shipment_header.get("DestinationCountry"),
            "postal_code": shipment_header.get("DestinationPostalCode"),
        }

        # Weight
        shipment_weight = shipment_header.get("ShipmentWeight", 0.0)
        weight_unit = shipment_header.get("ShipmentWeightUOM", "KG")
        weight_tonnes = self._convert_weight_to_tonnes(shipment_weight, weight_unit)
        weight_source = "Shipment_Document" if shipment_weight > 0 else "Estimated"

        # Freight cost
        freight_cost = shipment_header.get("FreightCost", 0.0)
        freight_currency = shipment_header.get("FreightCurrency", "USD")
        spend_usd, exchange_rate = self._convert_currency(freight_cost, freight_currency)

        # Carrier information
        carrier_info = None
        if shipment_header.get("CarrierName"):
            carrier_info = {
                "carrier_name": shipment_header.get("CarrierName"),
                "carrier_id_erp": str(shipment_header.get("CarrierId", "")),
            }

        # Metadata
        metadata = {
            "source_system": "Oracle_Fusion",
            "source_document_id": f"SHIP-{shipment_header['ShipmentNumber']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "oracle-scm-shipment-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields
        custom_fields = {
            "shipment_id": shipment_header.get("ShipmentId"),
            "shipment_number": shipment_header.get("ShipmentNumber"),
            "shipment_status": shipment_header.get("ShipmentStatus"),
            "carrier_service_level": shipment_header.get("CarrierServiceLevel"),
            "origin_location_id": shipment_header.get("OriginLocationId"),
            "destination_location_id": shipment_header.get("DestinationLocationId"),
            "planned_delivery_date": shipment_header.get("PlannedDeliveryDate"),
            "actual_delivery_date": shipment_header.get("ActualDeliveryDate"),
            "shipment_volume": shipment_header.get("ShipmentVolume"),
            "shipment_volume_uom": shipment_header.get("ShipmentVolumeUOM"),
        }

        # Add line item details if provided
        if shipment_lines:
            custom_fields["line_count"] = len(shipment_lines)
            custom_fields["items"] = [
                {
                    "item_number": line.get("ItemNumber"),
                    "item_description": line.get("ItemDescription"),
                    "quantity": line.get("Quantity"),
                }
                for line in shipment_lines[:10]  # Limit to first 10 items
            ]

        # Build logistics record
        record = LogisticsRecord(
            shipment_id=shipment_id,
            tenant_id=self.tenant_id,
            shipment_date=shipment_date,
            reporting_year=self._extract_reporting_year(shipment_date),
            transport_mode=transport_mode,
            calculation_method="distance_based",  # Will need distance data
            origin=origin,
            destination=destination,
            weight_tonnes=weight_tonnes,
            weight_source=weight_source,
            carrier_information=carrier_info,
            spend_usd=spend_usd,
            spend_currency_original=freight_currency,
            spend_amount_original=freight_cost,
            exchange_rate_to_usd=exchange_rate,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped Shipment {shipment_id}: {weight_tonnes:.2f} tonnes via {transport_mode}")

        return record

    def map_batch(
        self,
        shipment_data: List[Dict[str, Any]]
    ) -> List[LogisticsRecord]:
        """Map a batch of Shipment records."""
        records = []

        for shipment in shipment_data:
            try:
                header = shipment.get("header", {})
                lines = shipment.get("lines", [])

                record = self.map_shipment(header, lines)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping Shipment record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(shipment_data)} Shipment records")
        return records
