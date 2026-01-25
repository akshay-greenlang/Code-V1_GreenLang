# -*- coding: utf-8 -*-
"""
Oracle Fusion Cloud Extractors

Extractors for Oracle Fusion Cloud REST APIs.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

from .base import BaseExtractor, ExtractionConfig, ExtractionResult
from .procurement_extractor import ProcurementExtractor
from .scm_extractor import SCMExtractor
from .financials_extractor import FinancialsExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "ProcurementExtractor",
    "SCMExtractor",
    "FinancialsExtractor",
]
