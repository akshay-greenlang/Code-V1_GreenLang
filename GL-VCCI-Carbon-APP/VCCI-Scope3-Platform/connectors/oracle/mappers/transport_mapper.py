"""
Transportation Mapper

Maps Oracle Fusion SCM Cloud Transportation Order data to VCCI logistics_v1.0.json schema.

This mapper transforms Oracle Transportation Order data into the standardized logistics schema
for tracking transportation movements and their associated Scope 3 Category 4
(Upstream Transportation) emissions.

Transportation Orders typically contain more detailed route and vehicle information
compared to Shipments.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TransportMapper:
    """Maps Oracle Transportation Order data to VCCI logistics schema."""

    # Oracle transport mode to VCCI transport mode mapping
    TRANSPORT_MODE_MAPPING = {
        "TRUCK": "Road_Truck_GreaterThan17t",
        "SMALL_TRUCK": "Road_Truck_LessThan7.5t",
        "MEDIUM_TRUCK": "Road_Truck_7.5to17t",
        "HEAVY_TRUCK": "Road_Truck_GreaterThan17t",
        "RAIL": "Rail_Freight",
        "SEA": "Sea_Freight_Container",
        "OCEAN": "Sea_Freight_Container",
        "AIR": "Air_Freight_LongHaul",
        "AIR_DOMESTIC": "Air_Freight_ShortHaul",
        "AIR_INTERNATIONAL": "Air_Freight_LongHaul",
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
        """Initialize Transportation mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized TransportMapper for tenant: {tenant_id}")

    def _generate_shipment_id(self, transport_order_id: int) -> str:
        """Generate VCCI shipment ID from Oracle transportation order ID."""
        return f"SHIP-TO-{transport_order_id}"

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

    def _convert_distance_to_km(self, distance_miles: Optional[float], distance_km: Optional[float]) -> Optional[float]:
        """Convert distance to kilometers."""
        if distance_km and distance_km > 0:
            return distance_km
        if distance_miles and distance_miles > 0:
            return distance_miles * 1.60934  # Miles to km
        return None

    def _extract_reporting_year(self, shipment_date: str) -> int:
        """Extract reporting year from shipment date."""
        try:
            return int(shipment_date[:4])
        except (ValueError, TypeError):
            return datetime.now().year

    def map_transportation_order(
        self,
        transport_order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map Oracle Transportation Order to VCCI logistics record.

        Args:
            transport_order: Oracle Transportation Order data

        Returns:
            Logistics record dictionary matching logistics_v1.0.json schema
        """
        # Generate shipment ID
        shipment_id = self._generate_shipment_id(transport_order["TransportationOrderId"])

        # Shipment date (use pickup date or actual arrival date)
        shipment_date = transport_order.get("ActualArrivalDate") or transport_order.get("PickupDate", "")
        if not shipment_date:
            shipment_date = datetime.now().strftime("%Y-%m-%d")

        # Transport mode
        transport_mode = self._map_transport_mode(transport_order.get("TransportMode"))

        # Origin location
        origin = {
            "location_name": transport_order.get("OriginLocationCode", "Unknown Origin"),
            "city": transport_order.get("OriginCity"),
            "country": transport_order.get("OriginCountry"),
        }

        # Destination location
        destination = {
            "location_name": transport_order.get("DestinationLocationCode", "Unknown Destination"),
            "city": transport_order.get("DestinationCity"),
            "country": transport_order.get("DestinationCountry"),
        }

        # Distance (prefer km, convert from miles if needed)
        distance_km = self._convert_distance_to_km(
            transport_order.get("DistanceMiles"),
            transport_order.get("DistanceKM")
        )
        distance_source = "Carrier_Provided" if distance_km else None

        # Weight
        total_weight = transport_order.get("TotalWeight", 0.0)
        weight_unit = transport_order.get("WeightUOM", "KG")
        weight_tonnes = self._convert_weight_to_tonnes(total_weight, weight_unit)
        weight_source = "Transportation_Order" if total_weight > 0 else "Estimated"

        # Freight cost
        total_charge = transport_order.get("TotalCharge", 0.0)
        freight_currency = transport_order.get("FreightCurrency", "USD")
        spend_usd, exchange_rate = self._convert_currency(total_charge, freight_currency)

        # Carrier information
        carrier_info = None
        if transport_order.get("CarrierName"):
            carrier_info = {
                "carrier_name": transport_order.get("CarrierName"),
                "carrier_id_erp": str(transport_order.get("CarrierId", "")),
            }

        # Vehicle specifications
        vehicle_specs = None
        if transport_order.get("VehicleType"):
            vehicle_specs = {
                "vehicle_type": transport_order.get("VehicleType"),
                "equipment_number": transport_order.get("EquipmentNumber"),
            }

        # Determine calculation method
        calculation_method = "distance_based" if distance_km else "spend_based"

        # Metadata
        metadata = {
            "source_system": "Oracle_Fusion",
            "source_document_id": f"TO-{transport_order['OrderNumber']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "oracle-scm-transport-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields
        custom_fields = {
            "transportation_order_id": transport_order.get("TransportationOrderId"),
            "order_number": transport_order.get("OrderNumber"),
            "order_status": transport_order.get("OrderStatus"),
            "order_type": transport_order.get("OrderType"),
            "service_level": transport_order.get("ServiceLevel"),
            "vehicle_type": transport_order.get("VehicleType"),
            "equipment_number": transport_order.get("EquipmentNumber"),
            "origin_location_id": transport_order.get("OriginLocationId"),
            "destination_location_id": transport_order.get("DestinationLocationId"),
            "pickup_date": transport_order.get("PickupDate"),
            "planned_arrival_date": transport_order.get("PlannedArrivalDate"),
            "actual_arrival_date": transport_order.get("ActualArrivalDate"),
            "distance_miles": transport_order.get("DistanceMiles"),
            "total_volume": transport_order.get("TotalVolume"),
            "volume_uom": transport_order.get("VolumeUOM"),
            "freight_charge": transport_order.get("FreightCharge"),
            "fuel_surcharge": transport_order.get("FuelSurcharge"),
        }

        # Build logistics record
        record = {
            "shipment_id": shipment_id,
            "tenant_id": self.tenant_id,
            "shipment_date": shipment_date,
            "reporting_year": self._extract_reporting_year(shipment_date),
            "transport_mode": transport_mode,
            "calculation_method": calculation_method,
            "origin": origin,
            "destination": destination,
            "distance_km": distance_km,
            "distance_source": distance_source,
            "weight_tonnes": weight_tonnes,
            "weight_source": weight_source,
            "carrier_information": carrier_info,
            "vehicle_specifications": vehicle_specs,
            "spend_usd": spend_usd,
            "spend_currency_original": freight_currency,
            "spend_amount_original": total_charge,
            "exchange_rate_to_usd": exchange_rate,
            "metadata": metadata,
            "custom_fields": custom_fields,
        }

        logger.debug(
            f"Mapped Transportation Order {shipment_id}: {weight_tonnes:.2f} tonnes, "
            f"{distance_km or 0:.0f} km via {transport_mode}"
        )

        return record

    def map_batch(
        self,
        transport_orders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map a batch of Transportation Order records."""
        records = []

        for order in transport_orders:
            try:
                record = self.map_transportation_order(order)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping Transportation Order: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(transport_orders)} Transportation Order records")
        return records
