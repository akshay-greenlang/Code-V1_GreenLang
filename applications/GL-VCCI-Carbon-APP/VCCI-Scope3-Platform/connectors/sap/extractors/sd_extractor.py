# -*- coding: utf-8 -*-
"""
SAP Sales & Distribution (SD) Extractor

Extracts data from SAP S/4HANA Sales & Distribution module including:
    - Outbound Deliveries (API_OUTBOUND_DELIVERY_SRV)
    - Transportation Orders (API_TRANSPORTATION_ORDER_SRV)

Supports delta extraction and maps delivery and transportation data
for Scope 3 Category 4 (Upstream Transportation) and Category 9 (Downstream Transportation).

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class OutboundDeliveryData(BaseModel):
    """SAP Outbound Delivery Header data model.

    Maps to A_OutbDeliveryHeader entity from API_OUTBOUND_DELIVERY_SRV.
    """
    OutboundDelivery: str
    DeliveryDocumentType: Optional[str] = None
    DeliveryDate: Optional[str] = None
    ActualDeliveryDate: Optional[str] = None
    ShippingPoint: Optional[str] = None
    ShipToParty: Optional[str] = None
    ShipToPartyName: Optional[str] = None
    SoldToParty: Optional[str] = None
    SoldToPartyName: Optional[str] = None
    ReceivingPlant: Optional[str] = None
    ReceivingPlantName: Optional[str] = None
    ShippingCondition: Optional[str] = None
    ShippingType: Optional[str] = None
    Route: Optional[str] = None
    RouteDescription: Optional[str] = None
    TotalWeight: Optional[float] = None
    WeightUnit: Optional[str] = None
    TotalVolume: Optional[float] = None
    VolumeUnit: Optional[str] = None
    IncotermsClassification: Optional[str] = None
    IncotermsLocation1: Optional[str] = None
    MeansOfTransport: Optional[str] = None
    MeansOfTransportType: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    CreationTime: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    LastChangeDate: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class OutboundDeliveryItemData(BaseModel):
    """SAP Outbound Delivery Item data model.

    Maps to A_OutbDeliveryItem entity.
    """
    OutboundDelivery: str
    OutboundDeliveryItem: str
    Material: Optional[str] = None
    MaterialName: Optional[str] = None
    MaterialGroup: Optional[str] = None
    Batch: Optional[str] = None
    ActualDeliveryQuantity: Optional[float] = None
    DeliveryQuantityUnit: Optional[str] = None
    ItemGrossWeight: Optional[float] = None
    ItemWeightUnit: Optional[str] = None
    ItemVolume: Optional[float] = None
    ItemVolumeUnit: Optional[str] = None
    Plant: Optional[str] = None
    StorageLocation: Optional[str] = None
    ReferenceSDDocument: Optional[str] = None
    ReferenceSDDocumentItem: Optional[str] = None


class TransportationOrderData(BaseModel):
    """SAP Transportation Order data model.

    Maps to A_TransportationOrder entity from API_TRANSPORTATION_ORDER_SRV.
    """
    TransportationOrder: str
    TransportationOrderType: Optional[str] = None
    TransportationMode: Optional[str] = None
    TransportationModeCategory: Optional[str] = None
    Carrier: Optional[str] = None
    CarrierName: Optional[str] = None
    TransportationServiceLevel: Optional[str] = None
    TransportationPlanningDate: Optional[str] = None
    TransportationExecutionDate: Optional[str] = None
    TranspOrdExecutionStartDateTime: Optional[str] = None
    TranspOrdExecutionEndDateTime: Optional[str] = None
    PlannedDepartureDateTime: Optional[str] = None
    ActualDepartureDateTime: Optional[str] = None
    PlannedArrivalDateTime: Optional[str] = None
    ActualArrivalDateTime: Optional[str] = None
    TotalDistance: Optional[float] = None
    DistanceUnit: Optional[str] = None
    TotalWeight: Optional[float] = None
    WeightUnit: Optional[str] = None
    TotalVolume: Optional[float] = None
    VolumeUnit: Optional[str] = None
    FreightCost: Optional[float] = None
    FreightCostCurrency: Optional[str] = None
    VehicleRegistrationNumber: Optional[str] = None
    VehicleType: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    LastChangeDateTime: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class TransportationOrderStopData(BaseModel):
    """SAP Transportation Order Stop data model.

    Maps to A_TranspOrdStop entity - represents origin/destination stops.
    """
    TransportationOrder: str
    TranspOrdStopSequenceNumber: str
    StopType: Optional[str] = None  # 'LOAD' or 'UNLOAD'
    LocationID: Optional[str] = None
    LocationName: Optional[str] = None
    Country: Optional[str] = None
    Region: Optional[str] = None
    CityName: Optional[str] = None
    PostalCode: Optional[str] = None
    StreetName: Optional[str] = None
    AddressTimeZone: Optional[str] = None
    Latitude: Optional[float] = None
    Longitude: Optional[float] = None
    PlannedArrivalDateTime: Optional[str] = None
    ActualArrivalDateTime: Optional[str] = None
    PlannedDepartureDateTime: Optional[str] = None
    ActualDepartureDateTime: Optional[str] = None


class SDExtractor(BaseExtractor):
    """Sales & Distribution (SD) Extractor.

    Extracts delivery and transportation data from SAP S/4HANA SD module.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize SD extractor.

        Args:
            client: SAP OData client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "SD"
        self._current_entity_set = "A_OutbDeliveryHeader"  # Default

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "ChangedOn"

    def extract_outbound_deliveries(
        self,
        shipping_point: Optional[str] = None,
        delivery_date_from: Optional[str] = None,
        delivery_date_to: Optional[str] = None,
        ship_to_party: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Outbound Delivery headers from SAP.

        Args:
            shipping_point: Filter by shipping point
            delivery_date_from: Filter by delivery date from (ISO format)
            delivery_date_to: Filter by delivery date to (ISO format)
            ship_to_party: Filter by ship-to party

        Yields:
            Outbound Delivery header records as dictionaries
        """
        self._current_entity_set = "A_OutbDeliveryHeader"

        additional_filters = []

        if shipping_point:
            additional_filters.append(f"ShippingPoint eq '{shipping_point}'")
        if ship_to_party:
            additional_filters.append(f"ShipToParty eq '{ship_to_party}'")
        if delivery_date_from:
            additional_filters.append(f"DeliveryDate ge datetime'{delivery_date_from}'")
        if delivery_date_to:
            additional_filters.append(f"DeliveryDate le datetime'{delivery_date_to}'")

        logger.info(f"Extracting Outbound Deliveries with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="DeliveryDate desc"
        )

    def extract_outbound_delivery_items(
        self,
        outbound_delivery: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Outbound Delivery Items from SAP.

        Args:
            outbound_delivery: Filter by specific delivery number

        Yields:
            Outbound Delivery Item records as dictionaries
        """
        self._current_entity_set = "A_OutbDeliveryItem"

        additional_filters = []
        if outbound_delivery:
            additional_filters.append(f"OutboundDelivery eq '{outbound_delivery}'")

        logger.info(f"Extracting Outbound Delivery Items for: {outbound_delivery or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_transportation_orders(
        self,
        carrier: Optional[str] = None,
        transportation_mode: Optional[str] = None,
        execution_date_from: Optional[str] = None,
        execution_date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Transportation Orders from SAP.

        Args:
            carrier: Filter by carrier ID
            transportation_mode: Filter by mode (e.g., 'ROAD', 'SEA', 'AIR', 'RAIL')
            execution_date_from: Filter by execution date from (ISO format)
            execution_date_to: Filter by execution date to (ISO format)

        Yields:
            Transportation Order records as dictionaries
        """
        self._current_entity_set = "A_TransportationOrder"

        additional_filters = []

        if carrier:
            additional_filters.append(f"Carrier eq '{carrier}'")
        if transportation_mode:
            additional_filters.append(f"TransportationMode eq '{transportation_mode}'")
        if execution_date_from:
            additional_filters.append(
                f"TransportationExecutionDate ge datetime'{execution_date_from}'"
            )
        if execution_date_to:
            additional_filters.append(
                f"TransportationExecutionDate le datetime'{execution_date_to}'"
            )

        logger.info(f"Extracting Transportation Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="TransportationExecutionDate desc"
        )

    def extract_transportation_order_stops(
        self,
        transportation_order: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Transportation Order Stops (origin/destination) from SAP.

        Args:
            transportation_order: Filter by specific transportation order number

        Yields:
            Transportation Order Stop records as dictionaries
        """
        self._current_entity_set = "A_TranspOrdStop"

        additional_filters = []
        if transportation_order:
            additional_filters.append(f"TransportationOrder eq '{transportation_order}'")

        logger.info(
            f"Extracting Transportation Order Stops for: {transportation_order or 'All'}"
        )

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="TranspOrdStopSequenceNumber asc"
        )

    def extract_deliveries_with_transport(
        self,
        delivery_date_from: Optional[str] = None,
        delivery_date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract outbound deliveries that have associated transportation data.

        This is a convenience method for extracting complete logistics chain data.

        Args:
            delivery_date_from: Filter by delivery date from (ISO format)
            delivery_date_to: Filter by delivery date to (ISO format)

        Yields:
            Outbound Delivery records with transportation information
        """
        # Extract deliveries
        deliveries = list(self.extract_outbound_deliveries(
            delivery_date_from=delivery_date_from,
            delivery_date_to=delivery_date_to
        ))

        logger.info(f"Extracted {len(deliveries)} outbound deliveries")

        # For each delivery, enrich with items (if needed)
        for delivery in deliveries:
            delivery_number = delivery.get("OutboundDelivery")

            # Add items to delivery
            items = list(self.extract_outbound_delivery_items(
                outbound_delivery=delivery_number
            ))

            delivery["items"] = items

            yield delivery
