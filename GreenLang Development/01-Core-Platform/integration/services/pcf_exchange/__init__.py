# -*- coding: utf-8 -*-
"""
PCF Exchange Service
Product Carbon Footprint Exchange Protocol Support

This service provides integrations with major PCF exchange standards:
- PACT Pathfinder v2.0 (Partnership for Carbon Transparency)
- Catena-X PCF Exchange
- SAP Sustainability Data Exchange (SDX)

Enables import/export of product carbon footprints across supply chains
with validation, versioning, and data quality checks.

Version: 1.0.0
"""

from greenlang.services.pcf_exchange.service import PCFExchangeService
from greenlang.services.pcf_exchange.pact_client import PACTPathfinderClient
from greenlang.services.pcf_exchange.catenax_client import CatenaXClient
from greenlang.services.pcf_exchange.models import (
    PCFDataModel,
    PCFExchangeRequest,
    PCFExchangeResponse,
    DataQualityRating,
)

__version__ = "1.0.0"

__all__ = [
    "PCFExchangeService",
    "PACTPathfinderClient",
    "CatenaXClient",
    "PCFDataModel",
    "PCFExchangeRequest",
    "PCFExchangeResponse",
    "DataQualityRating",
]
