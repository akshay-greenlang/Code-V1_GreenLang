# -*- coding: utf-8 -*-
"""
GreenLang Transport MRV Agents
==============================

This package provides MRV (Monitoring, Reporting, Verification) agents
for transport sector emissions measurement and reporting.

Agents:
- GL-MRV-TRN-001: RoadTransportMRVAgent - Fleet and vehicle emissions
- GL-MRV-TRN-002: AviationMRVAgent - Aviation emissions
- GL-MRV-TRN-003: MaritimeMRVAgent - Shipping emissions
- GL-MRV-TRN-004: RailMRVAgent - Rail transport emissions
- GL-MRV-TRN-005: LastMileMRVAgent - Last-mile delivery emissions
- GL-MRV-TRN-006: EVFleetMRVAgent - Electric vehicle fleet emissions
- GL-MRV-TRN-007: LogisticsMRVAgent - Logistics and supply chain emissions
- GL-MRV-TRN-008: BusinessTravelMRVAgent - Employee travel emissions

All agents follow the CRITICAL PATH pattern with zero-hallucination guarantee.
"""

from greenlang.agents.mrv.transport.road_transport import (
    RoadTransportMRVAgent,
    RoadTransportInput,
    RoadTransportOutput,
    VehicleRecord,
    FleetRecord,
)
from greenlang.agents.mrv.transport.aviation import (
    AviationMRVAgent,
    AviationInput,
    AviationOutput,
    FlightRecord,
)
from greenlang.agents.mrv.transport.maritime import (
    MaritimeMRVAgent,
    MaritimeInput,
    MaritimeOutput,
    VoyageRecord,
)
from greenlang.agents.mrv.transport.rail import (
    RailMRVAgent,
    RailInput,
    RailOutput,
    RailShipmentRecord,
)
from greenlang.agents.mrv.transport.last_mile import (
    LastMileMRVAgent,
    LastMileInput,
    LastMileOutput,
    DeliveryRecord,
)
from greenlang.agents.mrv.transport.ev_fleet import (
    EVFleetMRVAgent,
    EVFleetInput,
    EVFleetOutput,
    EVVehicleRecord,
)
from greenlang.agents.mrv.transport.logistics import (
    LogisticsMRVAgent,
    LogisticsInput,
    LogisticsOutput,
    LogisticsShipmentRecord,
)
from greenlang.agents.mrv.transport.business_travel import (
    BusinessTravelMRVAgent,
    BusinessTravelInput,
    BusinessTravelOutput,
    TravelRecord,
)

__all__ = [
    # Road Transport (GL-MRV-TRN-001)
    "RoadTransportMRVAgent",
    "RoadTransportInput",
    "RoadTransportOutput",
    "VehicleRecord",
    "FleetRecord",
    # Aviation (GL-MRV-TRN-002)
    "AviationMRVAgent",
    "AviationInput",
    "AviationOutput",
    "FlightRecord",
    # Maritime (GL-MRV-TRN-003)
    "MaritimeMRVAgent",
    "MaritimeInput",
    "MaritimeOutput",
    "VoyageRecord",
    # Rail (GL-MRV-TRN-004)
    "RailMRVAgent",
    "RailInput",
    "RailOutput",
    "RailShipmentRecord",
    # Last Mile (GL-MRV-TRN-005)
    "LastMileMRVAgent",
    "LastMileInput",
    "LastMileOutput",
    "DeliveryRecord",
    # EV Fleet (GL-MRV-TRN-006)
    "EVFleetMRVAgent",
    "EVFleetInput",
    "EVFleetOutput",
    "EVVehicleRecord",
    # Logistics (GL-MRV-TRN-007)
    "LogisticsMRVAgent",
    "LogisticsInput",
    "LogisticsOutput",
    "LogisticsShipmentRecord",
    # Business Travel (GL-MRV-TRN-008)
    "BusinessTravelMRVAgent",
    "BusinessTravelInput",
    "BusinessTravelOutput",
    "TravelRecord",
]
