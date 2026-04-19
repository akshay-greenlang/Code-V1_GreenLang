# -*- coding: utf-8 -*-
"""
SAP Data Mappers Package

This package contains data mappers that transform SAP S/4HANA data structures
into VCCI platform JSON schemas (procurement_v1.0, logistics_v1.0).

Mappers handle:
    - Field mapping and transformation
    - Data type conversion
    - Unit standardization
    - Missing field handling
    - Metadata enrichment

Modules:
    - po_mapper: Purchase Orders → procurement_v1.0.json
    - goods_receipt_mapper: Goods Receipts → logistics_v1.0.json
    - delivery_mapper: Outbound Deliveries → logistics_v1.0.json
    - transport_mapper: Transportation Orders → logistics_v1.0.json

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

from .po_mapper import PurchaseOrderMapper
from .goods_receipt_mapper import GoodsReceiptMapper
from .delivery_mapper import DeliveryMapper
from .transport_mapper import TransportMapper

__all__ = [
    "PurchaseOrderMapper",
    "GoodsReceiptMapper",
    "DeliveryMapper",
    "TransportMapper",
]
