# -*- coding: utf-8 -*-
"""
Business Travel Agent Package - AGENT-MRV-019

GHG Protocol Scope 3, Category 6: Business Travel.
Calculates emissions from transportation of employees for business-related
activities in vehicles not owned or operated by the reporting company,
plus hotel accommodation during business travel.

Agent ID: GL-MRV-S3-006
Package: greenlang.agents.mrv.business_travel
API: /api/v1/business-travel
DB Migration: V070
Metrics Prefix: gl_bt_
Table Prefix: gl_bt_

Supported Transport Modes:
    - Air (domestic/short-haul/long-haul, radiative forcing, cabin class)
    - Rail (national/international/metro/light rail/high-speed)
    - Road (13 vehicle types, distance-based and fuel-based)
    - Bus (local bus, coach)
    - Taxi (regular, black cab)
    - Ferry (foot passenger, car passenger)
    - Motorcycle
    - Hotel accommodation (16 countries, 4 classes)

Calculation Methods:
    - Supplier-specific (primary data from carriers/hotels)
    - Distance-based (distance x mode-specific EF)
    - Average-data (industry average EFs)
    - Spend-based (EEIO factors with CPI deflation)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "BusinessTravelDatabaseEngine",
    "AirTravelCalculatorEngine",
    "GroundTransportCalculatorEngine",
    "HotelStayCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "ComplianceCheckerEngine",
    "BusinessTravelPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-006"
AGENT_COMPONENT: str = "AGENT-MRV-019"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_bt_"

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.business_travel.business_travel_database import BusinessTravelDatabaseEngine
except ImportError:
    BusinessTravelDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.business_travel.air_travel_calculator import AirTravelCalculatorEngine
except ImportError:
    AirTravelCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.business_travel.ground_transport_calculator import GroundTransportCalculatorEngine
except ImportError:
    GroundTransportCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.business_travel.hotel_stay_calculator import HotelStayCalculatorEngine
except ImportError:
    HotelStayCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.business_travel.spend_based_calculator import SpendBasedCalculatorEngine
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.business_travel.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.business_travel.business_travel_pipeline import BusinessTravelPipelineEngine
except ImportError:
    BusinessTravelPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.agents.mrv.business_travel.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
