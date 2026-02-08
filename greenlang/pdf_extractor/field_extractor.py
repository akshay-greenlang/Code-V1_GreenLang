# -*- coding: utf-8 -*-
"""
Field Extractor - AGENT-DATA-001: PDF & Invoice Extractor

Pattern-based field extraction engine that applies regex patterns to
extracted text and returns typed values with confidence scores.  Ships
with comprehensive built-in pattern sets for invoices, shipping manifests,
utility bills, receipts, and purchase orders.

Features:
    - Multi-pattern matching per field with best-match selection
    - Typed value parsing (date, numeric, currency, string)
    - Multiple date format support (ISO, US, EU, long-form)
    - Currency symbol and thousand-separator handling
    - Line-item table extraction
    - Custom pattern registration
    - Per-extraction confidence scoring
    - Thread-safe statistics

Zero-Hallucination Guarantees:
    - All extractions are regex-based, never LLM-inferred
    - Confidence reflects pattern match quality, not prediction
    - Unmatched fields return None with confidence 0.0

Example:
    >>> from greenlang.pdf_extractor.field_extractor import FieldExtractor
    >>> extractor = FieldExtractor()
    >>> fields, meta = extractor.extract_fields(
    ...     text, document_type="invoice",
    ... )
    >>> for f in fields:
    ...     print(f.field_name, f.value, f.confidence)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractedField",
    "LineItem",
    "FieldExtractor",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """A single extracted field with value and confidence."""

    field_name: str = Field(..., description="Canonical field name")
    raw_value: str = Field(default="", description="Raw matched text")
    value: Any = Field(default=None, description="Parsed typed value")
    field_type: str = Field(default="string", description="Field type")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Extraction confidence",
    )
    pattern_used: str = Field(
        default="", description="Regex pattern that matched",
    )

    model_config = {"extra": "forbid"}


class LineItem(BaseModel):
    """A single line item from a tabular section."""

    line_number: int = Field(default=0, ge=0, description="1-based line number")
    description: str = Field(default="", description="Item description")
    quantity: Optional[float] = Field(None, description="Quantity")
    unit: Optional[str] = Field(None, description="Unit of measure")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    amount: Optional[float] = Field(None, description="Line total")
    raw_text: str = Field(default="", description="Original line text")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Built-in pattern dictionaries
# ---------------------------------------------------------------------------

_INVOICE_PATTERNS: Dict[str, List[str]] = {
    "invoice_number": [
        r"(?i)invoice\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
        r"(?i)inv\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
        r"(?i)bill\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
    ],
    "invoice_date": [
        r"(?i)invoice\s*date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)invoice\s*date\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
        r"(?i)date\s*of\s*invoice\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)invoice\s*date\s*[:\-]?\s*(\w+\s+\d{1,2},?\s+\d{4})",
    ],
    "due_date": [
        r"(?i)due\s*date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)due\s*date\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
        r"(?i)payment\s*due\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)due\s*date\s*[:\-]?\s*(\w+\s+\d{1,2},?\s+\d{4})",
    ],
    "vendor_name": [
        r"(?i)(?:vendor|supplier|from|seller|billed?\s*by)\s*[:\-]?\s*(.+?)(?:\n|$)",
        r"(?i)company\s*name\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "vendor_address": [
        r"(?i)(?:vendor|supplier|seller)\s*address\s*[:\-]?\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
    ],
    "vendor_tax_id": [
        r"(?i)(?:tax\s*id|tin|ein|vat\s*(?:no\.?|number|id)|gst\s*(?:no\.?|number))\s*[:\-]?\s*(\S+)",
    ],
    "buyer_name": [
        r"(?i)(?:buyer|customer|bill\s*to|sold\s*to|client)\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "buyer_address": [
        r"(?i)(?:buyer|bill\s*to|sold\s*to)\s*address\s*[:\-]?\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
    ],
    "subtotal": [
        r"(?i)sub\s*-?\s*total\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
        r"(?i)net\s*(?:amount|total)\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
    "tax_amount": [
        r"(?i)(?:tax|vat|gst|hst)\s*(?:amount)?\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
        r"(?i)(?:sales\s*tax)\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
    "total_amount": [
        r"(?i)(?:grand\s*)?total\s*(?:due|amount|payable)?\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
        r"(?i)amount\s*due\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
        r"(?i)balance\s*due\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
    "currency": [
        r"(?i)currency\s*[:\-]?\s*([A-Z]{3})",
        r"(USD|EUR|GBP|CAD|AUD|JPY|CHF|INR|CNY)",
    ],
    "payment_terms": [
        r"(?i)(?:payment\s*)?terms?\s*[:\-]?\s*(net\s*\d+|due\s*on\s*receipt|cod|cia|[\w\s]+days)",
    ],
    "po_number": [
        r"(?i)(?:p\.?o\.?\s*(?:#|no\.?|number)|purchase\s*order)\s*[:\-]?\s*(\S+)",
    ],
}

_MANIFEST_PATTERNS: Dict[str, List[str]] = {
    "manifest_number": [
        r"(?i)(?:manifest|bol|b\/l|bill\s*of\s*lading)\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
    ],
    "shipment_date": [
        r"(?i)(?:ship(?:ment|ping)?|departure)\s*date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)(?:ship(?:ment|ping)?|departure)\s*date\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
    ],
    "carrier_name": [
        r"(?i)carrier\s*(?:name)?\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "origin": [
        r"(?i)(?:origin|ship\s*from|port\s*of\s*loading)\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "destination": [
        r"(?i)(?:destination|ship\s*to|port\s*of\s*discharge|deliver\s*to)\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "shipper_name": [
        r"(?i)shipper\s*(?:name)?\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "consignee_name": [
        r"(?i)consignee\s*(?:name)?\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "total_weight": [
        r"(?i)(?:total|gross)\s*weight\s*[:\-]?\s*([\d,]+\.?\d*)",
    ],
    "weight_unit": [
        r"(?i)weight\s*unit\s*[:\-]?\s*(kg|lbs?|tonnes?|tons?|mt)",
        r"(?i)(?:total|gross)\s*weight\s*[:\-]?\s*[\d,]+\.?\d*\s*(kg|lbs?|tonnes?|tons?|mt)",
    ],
    "total_pieces": [
        r"(?i)(?:total\s*)?(?:pieces|packages|cartons|units)\s*[:\-]?\s*(\d+)",
    ],
    "vehicle_id": [
        r"(?i)(?:vehicle|truck|trailer)\s*(?:id|#|no\.?|number)\s*[:\-]?\s*(\S+)",
    ],
    "seal_numbers": [
        r"(?i)seal\s*(?:#|no\.?|number)s?\s*[:\-]?\s*(\S+(?:\s*[,;]\s*\S+)*)",
    ],
}

_UTILITY_BILL_PATTERNS: Dict[str, List[str]] = {
    "account_number": [
        r"(?i)account\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
    ],
    "billing_period_start": [
        r"(?i)(?:billing|service)\s*period\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)from\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
    ],
    "billing_period_end": [
        r"(?i)(?:billing|service)\s*period\s*.*?(?:to|through|-)\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)to\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
    ],
    "utility_type": [
        r"(?i)(electric(?:ity)?|gas|natural\s*gas|water|sewer|waste|telecom)",
    ],
    "meter_number": [
        r"(?i)meter\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
    ],
    "previous_reading": [
        r"(?i)(?:previous|prior|last)\s*(?:meter\s*)?reading\s*[:\-]?\s*([\d,]+\.?\d*)",
    ],
    "current_reading": [
        r"(?i)(?:current|present|new)\s*(?:meter\s*)?reading\s*[:\-]?\s*([\d,]+\.?\d*)",
    ],
    "consumption": [
        r"(?i)(?:consumption|usage|used)\s*[:\-]?\s*([\d,]+\.?\d*)",
    ],
    "consumption_unit": [
        r"(?i)(?:consumption|usage)\s*.*?(kWh|MWh|therms?|ccf|mcf|gallons?|cubic\s*(?:feet|meters?))",
    ],
    "rate": [
        r"(?i)rate\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
    "total_amount": [
        r"(?i)(?:total|amount)\s*(?:due|payable)?\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
}

_RECEIPT_PATTERNS: Dict[str, List[str]] = {
    "store_name": [
        r"(?i)(?:store|shop|merchant)\s*(?:name)?\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "receipt_date": [
        r"(?i)date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)date\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
    ],
    "receipt_number": [
        r"(?i)(?:receipt|transaction)\s*(?:#|no\.?|number)\s*[:\-]?\s*(\S+)",
    ],
    "total_amount": [
        r"(?i)total\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
    "payment_method": [
        r"(?i)(?:payment|paid\s*by|method)\s*[:\-]?\s*(cash|credit|debit|visa|mastercard|amex|check|cheque|wire|eft|ach)",
    ],
}

_PURCHASE_ORDER_PATTERNS: Dict[str, List[str]] = {
    "po_number": [
        r"(?i)(?:p\.?o\.?\s*(?:#|no\.?|number)|purchase\s*order\s*(?:#|no\.?|number)?)\s*[:\-]?\s*(\S+)",
    ],
    "po_date": [
        r"(?i)(?:p\.?o\.?\s*date|order\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?i)(?:p\.?o\.?\s*date|order\s*date)\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
    ],
    "vendor_name": [
        r"(?i)(?:vendor|supplier)\s*(?:name)?\s*[:\-]?\s*(.+?)(?:\n|$)",
    ],
    "ship_to": [
        r"(?i)ship\s*to\s*[:\-]?\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
    ],
    "total_amount": [
        r"(?i)(?:total|order\s*total)\s*[:\-]?\s*[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)",
    ],
}

_ALL_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "invoice": _INVOICE_PATTERNS,
    "manifest": _MANIFEST_PATTERNS,
    "utility_bill": _UTILITY_BILL_PATTERNS,
    "receipt": _RECEIPT_PATTERNS,
    "purchase_order": _PURCHASE_ORDER_PATTERNS,
}

# Field-type mapping for automatic type inference per field name
_FIELD_TYPES: Dict[str, str] = {
    "invoice_date": "date",
    "due_date": "date",
    "shipment_date": "date",
    "receipt_date": "date",
    "po_date": "date",
    "billing_period_start": "date",
    "billing_period_end": "date",
    "subtotal": "numeric",
    "tax_amount": "numeric",
    "total_amount": "currency",
    "total_weight": "numeric",
    "total_pieces": "numeric",
    "quantity": "numeric",
    "unit_price": "currency",
    "amount": "currency",
    "rate": "numeric",
    "previous_reading": "numeric",
    "current_reading": "numeric",
    "consumption": "numeric",
}

# Date formats to try in order
_DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%m.%d.%Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%y",
    "%d/%m/%y",
]

# Line-item table patterns
_LINE_ITEM_PATTERN = re.compile(
    r"^(.+?)\s+"               # description
    r"([\d,]+\.?\d*)\s+"       # quantity
    r"(?:(\w+)\s+)?"           # optional unit
    r"[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)\s+"  # unit_price
    r"[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)\s*$",  # amount
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# FieldExtractor
# ---------------------------------------------------------------------------


class FieldExtractor:
    """Pattern-based field extraction with confidence scoring.

    Applies regex patterns against document text to extract structured
    fields.  Each field may have multiple patterns; the best-scoring match
    is selected.  Values are parsed into typed representations (date,
    numeric, currency) where applicable.

    Attributes:
        _patterns: Combined pattern dictionary keyed by document type.
        _custom_patterns: User-registered custom patterns.
        _lock: Threading lock for statistics.
        _stats: Extraction statistics.

    Example:
        >>> ext = FieldExtractor()
        >>> fields, meta = ext.extract_fields(text, "invoice")
        >>> print([(f.field_name, f.value) for f in fields])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise FieldExtractor.

        Args:
            config: Optional configuration dict.  Recognised keys:
                - ``confidence_threshold``: float (default 0.7)
                - ``max_line_items``: int (default 500)
        """
        self._config = config or {}
        self._patterns: Dict[str, Dict[str, List[str]]] = {
            k: dict(v) for k, v in _ALL_PATTERNS.items()
        }
        self._custom_patterns: Dict[str, List[str]] = {}
        self._default_threshold: float = self._config.get(
            "confidence_threshold", 0.7,
        )
        self._max_line_items: int = self._config.get("max_line_items", 500)
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "total_extractions": 0,
            "fields_extracted": 0,
            "by_type": {},
            "total_confidence": 0.0,
            "errors": 0,
        }
        logger.info(
            "FieldExtractor initialised: doc_types=%d, threshold=%.2f",
            len(self._patterns), self._default_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_fields(
        self,
        text: str,
        document_type: str,
        confidence_threshold: float = 0.7,
        template: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[ExtractedField], Dict[str, Any]]:
        """Extract all fields from text for a given document type.

        Args:
            text: Document text to extract from.
            document_type: Document type key (e.g. "invoice").
            confidence_threshold: Minimum confidence to include a field.
            template: Optional custom pattern template to use instead
                      of built-in patterns.

        Returns:
            Tuple of (list of ExtractedField, metadata dict).
        """
        start = time.monotonic()

        patterns = template or self.get_default_patterns(document_type)
        if not patterns:
            logger.warning("No patterns for document_type '%s'", document_type)
            return [], {"document_type": document_type, "fields_found": 0}

        fields: List[ExtractedField] = []
        for field_name, field_patterns in patterns.items():
            field_type = _FIELD_TYPES.get(field_name, "string")
            result = self.extract_field(text, field_name, field_patterns, field_type)
            if result and result.confidence >= confidence_threshold:
                fields.append(result)

        # Include custom patterns
        for field_name, custom_pats in self._custom_patterns.items():
            field_type = _FIELD_TYPES.get(field_name, "string")
            result = self.extract_field(text, field_name, custom_pats, field_type)
            if result and result.confidence >= confidence_threshold:
                fields.append(result)

        elapsed_ms = (time.monotonic() - start) * 1000
        with self._lock:
            self._stats["total_extractions"] += 1
            self._stats["fields_extracted"] += len(fields)
            type_stats = self._stats["by_type"].setdefault(
                document_type, {"calls": 0, "fields": 0},
            )
            type_stats["calls"] += 1
            type_stats["fields"] += len(fields)
            for f in fields:
                self._stats["total_confidence"] += f.confidence

        metadata = {
            "document_type": document_type,
            "fields_found": len(fields),
            "extraction_time_ms": round(elapsed_ms, 2),
            "confidence_threshold": confidence_threshold,
            "provenance_hash": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        }
        logger.info(
            "Extracted %d fields for '%s' (%.1f ms)",
            len(fields), document_type, elapsed_ms,
        )
        return fields, metadata

    def extract_field(
        self,
        text: str,
        field_name: str,
        patterns: List[str],
        field_type: str = "string",
    ) -> Optional[ExtractedField]:
        """Extract a single field using a list of regex patterns.

        Tries each pattern in order and returns the best-confidence match.

        Args:
            text: Document text.
            field_name: Canonical field name.
            patterns: List of regex pattern strings.
            field_type: Expected type ("string", "date", "numeric", "currency").

        Returns:
            ExtractedField or None if no pattern matched.
        """
        best_raw: Optional[str] = None
        best_confidence: float = 0.0
        best_pattern: str = ""

        for pattern in patterns:
            result = self._apply_pattern(text, pattern)
            if result is not None:
                raw_val, conf = result
                if conf > best_confidence:
                    best_raw = raw_val
                    best_confidence = conf
                    best_pattern = pattern

        if best_raw is None:
            return None

        parsed_value, parse_confidence = self.parse_field_value(
            field_name, best_raw, field_type,
        )
        # Combined confidence = match * parse
        combined = best_confidence * parse_confidence

        return ExtractedField(
            field_name=field_name,
            raw_value=best_raw,
            value=parsed_value,
            field_type=field_type,
            confidence=round(combined, 4),
            pattern_used=best_pattern,
        )

    def extract_line_items(
        self,
        text: str,
        document_type: str,
    ) -> List[LineItem]:
        """Extract tabular line items from document text.

        Applies multiple table patterns to find rows with description,
        quantity, unit price, and amount columns.

        Args:
            text: Document text.
            document_type: Document type for pattern selection.

        Returns:
            List of LineItem objects.
        """
        items: List[LineItem] = []
        matches = _LINE_ITEM_PATTERN.findall(text)

        for idx, match in enumerate(matches[: self._max_line_items]):
            desc = match[0].strip()
            qty = self._safe_float(match[1])
            unit = match[2].strip() if match[2] else None
            unit_price = self._safe_float(match[3])
            amount = self._safe_float(match[4])

            items.append(LineItem(
                line_number=idx + 1,
                description=desc,
                quantity=qty,
                unit=unit,
                unit_price=unit_price,
                amount=amount,
                raw_text=" ".join(match).strip(),
            ))

        # Fallback: simpler two-column pattern (description + amount)
        if not items:
            simple = re.compile(
                r"^(.{3,40}?)\s+[\$\u20ac\u00a3]?\s*([\d,]+\.?\d{2})\s*$",
                re.MULTILINE,
            )
            for idx, m in enumerate(
                simple.finditer(text)[: self._max_line_items]
                if hasattr(simple.finditer(text), "__getitem__")
                else list(simple.finditer(text))[: self._max_line_items]
            ):
                items.append(LineItem(
                    line_number=idx + 1,
                    description=m.group(1).strip(),
                    amount=self._safe_float(m.group(2)),
                    raw_text=m.group(0).strip(),
                ))

        logger.debug("Extracted %d line items for '%s'", len(items), document_type)
        return items

    def register_custom_patterns(
        self,
        field_name: str,
        patterns: List[str],
    ) -> None:
        """Register custom extraction patterns for a field.

        Args:
            field_name: Field name to register patterns for.
            patterns: List of regex pattern strings.
        """
        self._custom_patterns[field_name] = patterns
        logger.info(
            "Registered %d custom patterns for '%s'",
            len(patterns), field_name,
        )

    def get_default_patterns(
        self,
        document_type: str,
    ) -> Dict[str, List[str]]:
        """Get built-in patterns for a document type.

        Args:
            document_type: Document type key.

        Returns:
            Dictionary of field_name -> pattern list, or empty dict.
        """
        return dict(self._patterns.get(document_type, {}))

    def get_statistics(self) -> Dict[str, Any]:
        """Return extraction statistics.

        Returns:
            Dictionary with total extractions, per-type breakdowns,
            and average confidence.
        """
        with self._lock:
            total_fields = self._stats["fields_extracted"]
            total_conf = self._stats["total_confidence"]
            avg_conf = total_conf / total_fields if total_fields > 0 else 0.0
            return {
                "total_extractions": self._stats["total_extractions"],
                "fields_extracted": total_fields,
                "avg_confidence": round(avg_conf, 4),
                "by_type": dict(self._stats["by_type"]),
                "errors": self._stats["errors"],
                "custom_patterns_registered": len(self._custom_patterns),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Pattern matching internals
    # ------------------------------------------------------------------

    def _apply_pattern(
        self,
        text: str,
        pattern: str,
    ) -> Optional[Tuple[str, float]]:
        """Apply a single regex pattern and return match + confidence.

        Confidence is calculated based on match specificity:
        - Exact group match in short context = 0.95
        - Match with surrounding noise = 0.80
        - Partial match = 0.65

        Args:
            text: Document text.
            pattern: Regex pattern string.

        Returns:
            Tuple of (matched_group, confidence) or None.
        """
        try:
            match = re.search(pattern, text)
            if match is None:
                return None

            captured = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if not captured:
                return None

            # Confidence heuristic based on match quality
            full_span = match.end() - match.start()
            group_len = len(captured)
            ratio = group_len / max(full_span, 1)

            if ratio > 0.5:
                confidence = 0.95
            elif ratio > 0.25:
                confidence = 0.80
            else:
                confidence = 0.65

            return captured, confidence
        except re.error as exc:
            logger.warning("Regex error for pattern '%s': %s", pattern, exc)
            with self._lock:
                self._stats["errors"] += 1
            return None

    def parse_field_value(
        self,
        field_name: str,
        raw_value: str,
        field_type: str,
    ) -> Tuple[Any, float]:
        """Parse a raw string into a typed value.

        Args:
            field_name: Field name (for context-specific parsing).
            raw_value: Raw extracted string.
            field_type: Target type.

        Returns:
            Tuple of (parsed_value, parse_confidence).
        """
        if field_type == "date":
            return self._parse_date(raw_value)
        if field_type == "numeric":
            return self._parse_numeric(raw_value)
        if field_type == "currency":
            return self._parse_currency(raw_value)
        # Default: string
        return raw_value.strip(), 1.0

    def _parse_date(self, raw_value: str) -> Tuple[Optional[str], float]:
        """Parse a date string into ISO format.

        Args:
            raw_value: Raw date string.

        Returns:
            Tuple of (ISO date string or None, confidence).
        """
        cleaned = raw_value.strip()
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.strftime("%Y-%m-%d"), 0.95
            except ValueError:
                continue
        # Could not parse
        logger.debug("Could not parse date: '%s'", raw_value)
        return raw_value, 0.50

    def _parse_numeric(self, raw_value: str) -> Tuple[Optional[float], float]:
        """Parse a numeric string, removing thousand separators.

        Args:
            raw_value: Raw numeric string.

        Returns:
            Tuple of (float value or None, confidence).
        """
        cleaned = raw_value.strip().replace(",", "")
        try:
            return float(cleaned), 0.95
        except ValueError:
            logger.debug("Could not parse numeric: '%s'", raw_value)
            return None, 0.30

    def _parse_currency(self, raw_value: str) -> Tuple[Optional[float], float]:
        """Parse a currency string, removing symbols and separators.

        Args:
            raw_value: Raw currency string.

        Returns:
            Tuple of (float value or None, confidence).
        """
        cleaned = re.sub(r"[^\d.,\-]", "", raw_value)
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned), 0.95
        except ValueError:
            logger.debug("Could not parse currency: '%s'", raw_value)
            return None, 0.30

    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert a string to float.

        Args:
            value: String to convert.

        Returns:
            Float value or None.
        """
        try:
            return float(value.replace(",", ""))
        except (ValueError, AttributeError):
            return None
