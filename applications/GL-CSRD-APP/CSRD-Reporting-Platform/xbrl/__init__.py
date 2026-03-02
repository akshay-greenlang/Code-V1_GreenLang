# -*- coding: utf-8 -*-
"""
GL-CSRD-APP XBRL/iXBRL Engine
==============================

ESEF-compliant XBRL taxonomy mapping, iXBRL generation, and validation
for CSRD/ESRS sustainability reporting.

This module provides a production-grade XBRL engine supporting:

1. **Taxonomy Mapping** - EFRAG ESRS XBRL taxonomy with 1,082+ elements,
   22 dimensions, multi-language labels, and filing indicators.

2. **iXBRL Generation** - Inline XBRL document generation compliant with
   ESEF Reporting Manual v7, including ix:header, contexts, units, facts,
   continuation blocks, and ESEF package creation.

3. **XBRL Validation** - Comprehensive validation covering taxonomy,
   context, unit, fact, calculation linkbase, dimension, filing indicator,
   ESEF RTS compliance, cross-reference, consistency, and completeness.

All components follow GreenLang's zero-hallucination principle: numeric
calculations and regulatory compliance checks use deterministic logic only.

Architecture:
    TaxonomyMapper (singleton) <-- loads taxonomy JSON data
         |
    IXBRLGenerator            <-- produces iXBRL XHTML + ESEF packages
         |
    XBRLValidator             <-- validates XBRL instances

Security:
    - XXE protection in all XML parsing operations
    - Input size validation to prevent DoS attacks
    - No external entity resolution

Version: 1.1.0
Author: GreenLang CSRD Team
License: MIT
"""

from xbrl.taxonomy_mapper import (
    TaxonomyElement,
    DimensionInfo,
    DimensionMember,
    TaxonomyMapper,
)
from xbrl.ixbrl_generator import (
    IXBRLContext,
    IXBRLUnit,
    IXBRLFact,
    IXBRLGenerator,
    ESEFPackager,
)
from xbrl.xbrl_validator import (
    ValidationSeverity,
    ValidationResult,
    ValidationReport,
    XBRLValidator,
)

__version__ = "1.1.0"
__author__ = "GreenLang CSRD Team"

__all__ = [
    # Taxonomy mapping
    "TaxonomyElement",
    "DimensionInfo",
    "DimensionMember",
    "TaxonomyMapper",
    # iXBRL generation
    "IXBRLContext",
    "IXBRLUnit",
    "IXBRLFact",
    "IXBRLGenerator",
    "ESEFPackager",
    # Validation
    "ValidationSeverity",
    "ValidationResult",
    "ValidationReport",
    "XBRLValidator",
    # Module metadata
    "__version__",
    "__author__",
]
