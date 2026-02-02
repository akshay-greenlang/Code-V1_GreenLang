# -*- coding: utf-8 -*-
"""
SAP Extractors Package

This package contains SAP S/4HANA OData extractors for the GL-VCCI Scope 3 Carbon Platform.
Extractors pull data from SAP modules (MM, SD, FI) and prepare it for mapping to VCCI schemas.

Modules:
    - base: Abstract base extractor with common extraction logic
    - mm_extractor: Materials Management (Purchase Orders, Goods Receipts, Vendor/Material Master)
    - sd_extractor: Sales & Distribution (Outbound Deliveries, Transportation Orders)
    - fi_extractor: Financial Accounting (Fixed Assets)

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

from .base import BaseExtractor, ExtractionConfig, ExtractionResult
from .mm_extractor import MMExtractor
from .sd_extractor import SDExtractor
from .fi_extractor import FIExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "MMExtractor",
    "SDExtractor",
    "FIExtractor",
]
