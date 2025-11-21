# -*- coding: utf-8 -*-
"""
Oracle Fusion Cloud Mappers

Mappers for transforming Oracle Fusion Cloud data to VCCI schemas.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

from .po_mapper import PurchaseOrderMapper, ProcurementRecord
from .requisition_mapper import RequisitionMapper
from .shipment_mapper import ShipmentMapper
from .transport_mapper import TransportMapper

__all__ = [
    "PurchaseOrderMapper",
    "ProcurementRecord",
    "RequisitionMapper",
    "ShipmentMapper",
    "TransportMapper",
]
