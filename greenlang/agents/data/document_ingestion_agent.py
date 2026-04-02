# -*- coding: utf-8 -*-
"""
GL-DATA-X-001: Document Ingestion & OCR Agent -- Re-export Shim
================================================================

This module is a backward-compatible re-export shim. All canonical
definitions now live inside the ``pdf_extractor`` sub-package:

- Enums and Pydantic models: ``pdf_extractor.models``
- Agent class: ``pdf_extractor.document_ingestion``

Existing imports such as::

    from greenlang.agents.data.document_ingestion_agent import (
        DocumentIngestionAgent, DocumentType, ...
    )

continue to work unchanged.
"""

from greenlang.agents.data.pdf_extractor.models import (  # noqa: F401
    DocumentType,
    ExtractionStatus,
    OCREngine,
    BoundingBox,
    ExtractedField,
    LineItem,
    InvoiceData,
    ManifestData,
    UtilityBillData,
    DocumentIngestionInput,
    DocumentIngestionOutput,
)

from greenlang.agents.data.pdf_extractor.document_ingestion import (  # noqa: F401
    DocumentIngestionAgent,
)

__all__ = [
    "DocumentType",
    "ExtractionStatus",
    "OCREngine",
    "BoundingBox",
    "ExtractedField",
    "LineItem",
    "InvoiceData",
    "ManifestData",
    "UtilityBillData",
    "DocumentIngestionInput",
    "DocumentIngestionOutput",
    "DocumentIngestionAgent",
]
