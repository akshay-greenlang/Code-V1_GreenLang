"""
Oracle Supply Chain Management (SCM) Extractor

Extracts data from Oracle Fusion SCM Cloud including:
    - Shipments (/shipments)
    - Transportation Orders (/transportationOrders)

Supports delta extraction by LastUpdateDate field and provides field selection
for performance optimization. Used for Category 4 (Upstream Transportation) emissions.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class ShipmentData(BaseModel):
    """Oracle Shipment data model.

    Maps to /shipments REST endpoint response.
    """
    ShipmentId: int
    ShipmentNumber: str
    ShipmentStatus: Optional[str] = None
    CarrierId: Optional[int] = None
    CarrierName: Optional[str] = None
    CarrierServiceLevel: Optional[str] = None
    OriginLocationId: Optional[int] = None
    OriginLocationCode: Optional[str] = None
    OriginAddress: Optional[str] = None
    OriginCity: Optional[str] = None
    OriginState: Optional[str] = None
    OriginCountry: Optional[str] = None
    OriginPostalCode: Optional[str] = None
    DestinationLocationId: Optional[int] = None
    DestinationLocationCode: Optional[str] = None
    DestinationAddress: Optional[str] = None
    DestinationCity: Optional[str] = None
    DestinationState: Optional[str] = None
    DestinationCountry: Optional[str] = None
    DestinationPostalCode: Optional[str] = None
    ShipmentDate: Optional[str] = None
    PlannedDeliveryDate: Optional[str] = None
    ActualDeliveryDate: Optional[str] = None
    ShipmentWeight: Optional[float] = None
    ShipmentWeightUOM: Optional[str] = None
    ShipmentVolume: Optional[float] = None
    ShipmentVolumeUOM: Optional[str] = None
    FreightCost: Optional[float] = None
    FreightCurrency: Optional[str] = None
    TransportMode: Optional[str] = None
    CreatedBy: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdatedBy: Optional[str] = None
    LastUpdateDate: Optional[str] = None  # For delta extraction


class ShipmentLineData(BaseModel):
    """Oracle Shipment Line data model.

    Maps to /shipmentLines REST endpoint response.
    """
    ShipmentLineId: int
    ShipmentId: int
    LineNumber: int
    ItemId: Optional[int] = None
    ItemNumber: Optional[str] = None
    ItemDescription: Optional[str] = None
    Quantity: Optional[float] = None
    QuantityUOM: Optional[str] = None
    Weight: Optional[float] = None
    WeightUOM: Optional[str] = None
    Volume: Optional[float] = None
    VolumeUOM: Optional[str] = None
    PackagingUnit: Optional[str] = None
    NumberOfPackages: Optional[int] = None
    SourceOrderId: Optional[int] = None
    SourceOrderNumber: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdateDate: Optional[str] = None


class TransportationOrderData(BaseModel):
    """Oracle Transportation Order data model.

    Maps to /transportationOrders REST endpoint response.
    """
    TransportationOrderId: int
    OrderNumber: str
    OrderStatus: Optional[str] = None
    OrderType: Optional[str] = None
    CarrierId: Optional[int] = None
    CarrierName: Optional[str] = None
    ServiceLevel: Optional[str] = None
    TransportMode: Optional[str] = None
    VehicleType: Optional[str] = None
    EquipmentNumber: Optional[str] = None
    OriginLocationId: Optional[int] = None
    OriginLocationCode: Optional[str] = None
    OriginCity: Optional[str] = None
    OriginCountry: Optional[str] = None
    DestinationLocationId: Optional[int] = None
    DestinationLocationCode: Optional[str] = None
    DestinationCity: Optional[str] = None
    DestinationCountry: Optional[str] = None
    PickupDate: Optional[str] = None
    PlannedArrivalDate: Optional[str] = None
    ActualArrivalDate: Optional[str] = None
    DistanceKM: Optional[float] = None
    DistanceMiles: Optional[float] = None
    TotalWeight: Optional[float] = None
    WeightUOM: Optional[str] = None
    TotalVolume: Optional[float] = None
    VolumeUOM: Optional[str] = None
    FreightCharge: Optional[float] = None
    FreightCurrency: Optional[str] = None
    FuelSurcharge: Optional[float] = None
    TotalCharge: Optional[float] = None
    CreatedBy: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdatedBy: Optional[str] = None
    LastUpdateDate: Optional[str] = None  # For delta extraction


class SCMExtractor(BaseExtractor):
    """Supply Chain Management (SCM) Extractor.

    Extracts logistics and transportation data from Oracle Fusion SCM Cloud.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize SCM extractor.

        Args:
            client: Oracle REST client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.base_url = "/fscmRestApi/resources/11.13.18.05"  # Oracle SCM REST API base
        self._current_resource = "/shipments"  # Default

    def get_resource_path(self) -> str:
        """Get current REST resource path."""
        return f"{self.base_url}{self._current_resource}"

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "LastUpdateDate"

    def extract_shipments(
        self,
        carrier_id: Optional[int] = None,
        origin_country: Optional[str] = None,
        destination_country: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Shipments from Oracle.

        Args:
            carrier_id: Filter by carrier ID
            origin_country: Filter by origin country code
            destination_country: Filter by destination country code
            status: Filter by shipment status (e.g., 'IN_TRANSIT', 'DELIVERED')
            date_from: Filter by shipment date from (ISO format)
            date_to: Filter by shipment date to (ISO format)

        Yields:
            Shipment records as dictionaries
        """
        self._current_resource = "/shipments"

        additional_filters = []

        if carrier_id:
            additional_filters.append(f"CarrierId={carrier_id}")
        if origin_country:
            additional_filters.append(f"OriginCountry='{origin_country}'")
        if destination_country:
            additional_filters.append(f"DestinationCountry='{destination_country}'")
        if status:
            additional_filters.append(f"ShipmentStatus='{status}'")
        if date_from:
            additional_filters.append(f"ShipmentDate>='{date_from}'")
        if date_to:
            additional_filters.append(f"ShipmentDate<='{date_to}'")

        logger.info(f"Extracting Shipments with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="ShipmentDate:desc"
        )

    def extract_shipment_lines(
        self,
        shipment_id: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Shipment Lines from Oracle.

        Args:
            shipment_id: Filter by specific shipment ID

        Yields:
            Shipment Line records as dictionaries
        """
        self._current_resource = "/shipmentLines"

        additional_filters = []
        if shipment_id:
            additional_filters.append(f"ShipmentId={shipment_id}")

        logger.info(f"Extracting Shipment Lines for Shipment: {shipment_id or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="ShipmentId:asc,LineNumber:asc"
        )

    def extract_transportation_orders(
        self,
        carrier_id: Optional[int] = None,
        transport_mode: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Transportation Orders from Oracle.

        Args:
            carrier_id: Filter by carrier ID
            transport_mode: Filter by transport mode (e.g., 'TRUCK', 'RAIL', 'AIR', 'SEA')
            status: Filter by order status (e.g., 'PLANNED', 'IN_TRANSIT', 'COMPLETED')
            date_from: Filter by pickup date from (ISO format)
            date_to: Filter by pickup date to (ISO format)

        Yields:
            Transportation Order records as dictionaries
        """
        self._current_resource = "/transportationOrders"

        additional_filters = []

        if carrier_id:
            additional_filters.append(f"CarrierId={carrier_id}")
        if transport_mode:
            additional_filters.append(f"TransportMode='{transport_mode}'")
        if status:
            additional_filters.append(f"OrderStatus='{status}'")
        if date_from:
            additional_filters.append(f"PickupDate>='{date_from}'")
        if date_to:
            additional_filters.append(f"PickupDate<='{date_to}'")

        logger.info(f"Extracting Transportation Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="PickupDate:desc"
        )
