"""
Transportation Order Mapper

Maps SAP S/4HANA Transportation Order data to VCCI logistics_v1.0.json schema.

This mapper transforms SAP Transportation Management (TM) Transportation Order
data into the standardized logistics schema for tracking transportation activities
and their associated Scope 3 Category 4 & 9 emissions.

Field Mappings:
    SAP Field → VCCI Schema Field
    - TransportationOrder → shipment_id (SHIP-TO-{Order})
    - TransportationExecutionDate → shipment_date
    - TransportationMode → transport_mode (with mapping)
    - TotalDistance → distance_km
    - TotalWeight → weight_tonnes
    - Carrier → carrier_information.carrier_name
    - FreightCost → spend_usd (for spend-based calculation)
    - TranspOrdStop (origin/destination) → origin/destination

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
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
    vehicle_specifications: Optional[Dict[str, Any]] = None
    carrier_information: Optional[Dict[str, Any]] = None
    spend_usd: Optional[float] = None
    spend_currency_original: Optional[str] = None
    spend_amount_original: Optional[float] = None
    exchange_rate_to_usd: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class TransportMapper:
    """Maps SAP Transportation Order data to VCCI logistics schema."""

    # SAP transportation mode to VCCI transport mode mapping
    TRANSPORT_MODE_MAPPING = {
        "01": "Road_Truck_GreaterThan17t",  # Road
        "02": "Rail_Freight",  # Rail
        "03": "Sea_Freight_Container",  # Sea
        "04": "Air_Freight_LongHaul",  # Air
        "05": "Sea_Freight_Bulk",  # Sea bulk
        "06": "Air_Freight_ShortHaul",  # Air short-haul
        "07": "Pipeline",  # Pipeline
        "ROAD": "Road_Truck_GreaterThan17t",
        "RAIL": "Rail_Freight",
        "SEA": "Sea_Freight_Container",
        "AIR": "Air_Freight_LongHaul",
        "PIPELINE": "Pipeline",
    }

    # Weight unit conversion to tonnes
    WEIGHT_CONVERSION = {
        "KG": 0.001,
        "G": 0.000001,
        "TO": 1.0,
        "T": 1.0,
        "LB": 0.000453592,
    }

    # Distance unit conversion to km
    DISTANCE_CONVERSION = {
        "KM": 1.0,
        "M": 0.001,
        "MI": 1.60934,  # Miles to km
        "NMI": 1.852,   # Nautical miles to km
    }

    # Currency conversion rates (USD base)
    CURRENCY_RATES = {
        "USD": 1.0,
        "EUR": 1.10,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CNY": 0.14,
    }

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize Transportation Order mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized TransportMapper for tenant: {tenant_id}")

    def _generate_shipment_id(self, transport_order: str) -> str:
        """Generate VCCI shipment ID from SAP transportation order.

        Args:
            transport_order: SAP Transportation Order number

        Returns:
            VCCI shipment ID (format: SHIP-TO-{Order})
        """
        return f"SHIP-TO-{transport_order}"

    def _map_transport_mode(
        self,
        sap_mode: Optional[str],
        sap_mode_category: Optional[str]
    ) -> str:
        """Map SAP transportation mode to VCCI transport mode.

        Args:
            sap_mode: SAP transportation mode code
            sap_mode_category: SAP transportation mode category

        Returns:
            VCCI transport mode
        """
        # Try mode first
        if sap_mode:
            mode = self.TRANSPORT_MODE_MAPPING.get(sap_mode.upper())
            if mode:
                return mode

        # Try category
        if sap_mode_category:
            mode = self.TRANSPORT_MODE_MAPPING.get(sap_mode_category.upper())
            if mode:
                return mode

        # Default to road
        logger.debug(f"Unknown transport mode, defaulting to Road_Truck_GreaterThan17t")
        return "Road_Truck_GreaterThan17t"

    def _convert_weight_to_tonnes(self, weight: float, weight_unit: str) -> float:
        """Convert weight to tonnes."""
        if not weight or weight <= 0:
            return 0.0
        conversion = self.WEIGHT_CONVERSION.get(weight_unit.upper(), 1.0)
        return weight * conversion

    def _convert_distance_to_km(self, distance: float, distance_unit: str) -> float:
        """Convert distance to kilometers."""
        if not distance or distance <= 0:
            return 0.0
        conversion = self.DISTANCE_CONVERSION.get(distance_unit.upper(), 1.0)
        return distance * conversion

    def _convert_currency(
        self,
        amount: float,
        from_currency: str
    ) -> tuple[float, float]:
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

    def map_transportation_order(
        self,
        transport_order: Dict[str, Any],
        stops: Optional[List[Dict[str, Any]]] = None,
        carrier_data: Optional[Dict[str, Any]] = None
    ) -> LogisticsRecord:
        """Map SAP Transportation Order to VCCI logistics record.

        Args:
            transport_order: SAP Transportation Order header data
            stops: Optional list of transportation order stops (origin/destination)
            carrier_data: Optional carrier master data

        Returns:
            LogisticsRecord matching logistics_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not transport_order.get("TransportationOrder"):
            raise ValueError("Missing required field: TransportationOrder")

        # Generate shipment ID
        shipment_id = self._generate_shipment_id(transport_order["TransportationOrder"])

        # Shipment date (prefer actual over planned)
        shipment_date = (
            transport_order.get("TransportationExecutionDate") or
            transport_order.get("TransportationPlanningDate", "")
        )
        if not shipment_date:
            shipment_date = datetime.now().strftime("%Y-%m-%d")
            logger.warning(f"TO {transport_order['TransportationOrder']}: Missing execution date, using today")

        # Transport mode
        transport_mode = self._map_transport_mode(
            transport_order.get("TransportationMode"),
            transport_order.get("TransportationModeCategory")
        )

        # Origin and destination from stops
        origin = None
        destination = None
        if stops:
            # Sort stops by sequence
            sorted_stops = sorted(stops, key=lambda s: int(s.get("TranspOrdStopSequenceNumber", 0)))

            # First stop is origin (LOAD)
            if sorted_stops:
                first_stop = sorted_stops[0]
                origin = {
                    "location_name": first_stop.get("LocationName", first_stop.get("LocationID", "Unknown")),
                    "city": first_stop.get("CityName"),
                    "country": first_stop.get("Country"),
                    "postal_code": first_stop.get("PostalCode"),
                    "coordinates": {
                        "latitude": first_stop.get("Latitude"),
                        "longitude": first_stop.get("Longitude"),
                    } if first_stop.get("Latitude") else None,
                }

            # Last stop is destination (UNLOAD)
            if len(sorted_stops) > 1:
                last_stop = sorted_stops[-1]
                destination = {
                    "location_name": last_stop.get("LocationName", last_stop.get("LocationID", "Unknown")),
                    "city": last_stop.get("CityName"),
                    "country": last_stop.get("Country"),
                    "postal_code": last_stop.get("PostalCode"),
                    "coordinates": {
                        "latitude": last_stop.get("Latitude"),
                        "longitude": last_stop.get("Longitude"),
                    } if last_stop.get("Latitude") else None,
                }

        # Distance
        total_distance = transport_order.get("TotalDistance", 0.0)
        distance_unit = transport_order.get("DistanceUnit", "KM")
        distance_km = self._convert_distance_to_km(total_distance, distance_unit)
        distance_source = "GPS_Telematics" if distance_km > 0 else "Manual_Estimate"

        # Weight
        total_weight = transport_order.get("TotalWeight", 0.0)
        weight_unit = transport_order.get("WeightUnit", "KG")
        weight_tonnes = self._convert_weight_to_tonnes(total_weight, weight_unit)
        weight_source = "Weigh_Station" if weight_tonnes > 0 else "Estimated"

        # Determine calculation method
        calculation_method = "distance_based"
        if distance_km > 0 and weight_tonnes > 0:
            calculation_method = "distance_based"
        elif transport_order.get("FreightCost", 0.0) > 0:
            calculation_method = "spend_based"

        # Carrier information
        carrier_id = transport_order.get("Carrier", "")
        carrier_info = None
        if carrier_id:
            carrier_info = {
                "carrier_name": carrier_data.get("CarrierName", transport_order.get("CarrierName", f"Carrier {carrier_id}")) if carrier_data else transport_order.get("CarrierName", f"Carrier {carrier_id}"),
                "carrier_id_erp": carrier_id,
            }

        # Freight cost (for spend-based method)
        freight_cost = transport_order.get("FreightCost", 0.0)
        freight_currency = transport_order.get("FreightCostCurrency", "USD")
        spend_usd = None
        spend_currency_original = None
        spend_amount_original = None
        exchange_rate = None

        if freight_cost > 0:
            spend_usd, exchange_rate = self._convert_currency(freight_cost, freight_currency)
            spend_currency_original = freight_currency
            spend_amount_original = freight_cost

        # Vehicle specifications
        vehicle_specs = None
        if transport_order.get("VehicleType"):
            vehicle_specs = {
                "vehicle_type": transport_order.get("VehicleType"),
            }

        # Metadata
        metadata = {
            "source_system": "SAP_TM",
            "source_document_id": f"TO-{transport_order['TransportationOrder']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "sap-tm-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields
        custom_fields = {
            "transportation_order": transport_order["TransportationOrder"],
            "transportation_order_type": transport_order.get("TransportationOrderType"),
            "transportation_service_level": transport_order.get("TransportationServiceLevel"),
            "planned_departure": transport_order.get("PlannedDepartureDateTime"),
            "actual_departure": transport_order.get("ActualDepartureDateTime"),
            "planned_arrival": transport_order.get("PlannedArrivalDateTime"),
            "actual_arrival": transport_order.get("ActualArrivalDateTime"),
            "vehicle_registration": transport_order.get("VehicleRegistrationNumber"),
            "total_volume": transport_order.get("TotalVolume"),
            "volume_unit": transport_order.get("VolumeUnit"),
        }

        # Build logistics record
        record = LogisticsRecord(
            shipment_id=shipment_id,
            tenant_id=self.tenant_id,
            shipment_date=shipment_date,
            reporting_year=self._extract_reporting_year(shipment_date),
            transport_mode=transport_mode,
            calculation_method=calculation_method,
            origin=origin,
            destination=destination,
            distance_km=distance_km if distance_km > 0 else None,
            distance_source=distance_source if distance_km > 0 else None,
            weight_tonnes=weight_tonnes if weight_tonnes > 0 else None,
            weight_source=weight_source if weight_tonnes > 0 else None,
            vehicle_specifications=vehicle_specs,
            carrier_information=carrier_info,
            spend_usd=spend_usd,
            spend_currency_original=spend_currency_original,
            spend_amount_original=spend_amount_original,
            exchange_rate_to_usd=exchange_rate,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(
            f"Mapped Transport Order {shipment_id}: {distance_km:.1f} km, "
            f"{weight_tonnes:.2f} tonnes via {transport_mode}"
        )

        return record

    def map_batch(
        self,
        transport_data: List[Dict[str, Any]],
        carrier_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[LogisticsRecord]:
        """Map a batch of Transportation Order records.

        Args:
            transport_data: List of dicts containing 'order' and optional 'stops' keys
            carrier_lookup: Optional dict mapping carrier ID to carrier data

        Returns:
            List of mapped LogisticsRecord objects
        """
        records = []
        carrier_lookup = carrier_lookup or {}

        for transport in transport_data:
            try:
                order = transport.get("order", {})
                stops = transport.get("stops", [])
                carrier_id = order.get("Carrier")

                carrier_data = carrier_lookup.get(carrier_id) if carrier_id else None

                record = self.map_transportation_order(order, stops, carrier_data)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping Transport Order record: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(transport_data)} Transport Order records")
        return records
