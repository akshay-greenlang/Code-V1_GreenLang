# -*- coding: utf-8 -*-
"""
Document Classifier - AGENT-DATA-001: PDF & Invoice Extractor

Document type classification engine that uses keyword scoring,
structural pattern analysis, and filename heuristics to determine
document types.  Returns a confidence-scored classification suitable
for routing documents to the correct processor (InvoiceProcessor,
ManifestProcessor, etc.).

Features:
    - Multi-signal classification (keywords, structure, filename)
    - Configurable scoring weights for each signal
    - Batch classification for pipeline throughput
    - Custom keyword registration per document type
    - Thread-safe statistics collection
    - Deterministic scoring (no ML or LLM in the path)

Zero-Hallucination Guarantees:
    - Classification is deterministic keyword + structure scoring
    - No ML models or LLM calls; purely pattern-based
    - Confidence reflects observed keyword density, not prediction

Example:
    >>> from greenlang.pdf_extractor.document_classifier import DocumentClassifier
    >>> classifier = DocumentClassifier()
    >>> doc_type, confidence = classifier.classify(invoice_text)
    >>> print(doc_type, confidence)
    invoice 0.92

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentType",
    "DocumentClassifier",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    """Classifiable document types."""

    INVOICE = "invoice"
    MANIFEST = "manifest"
    UTILITY_BILL = "utility_bill"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Built-in keyword dictionaries
# ---------------------------------------------------------------------------

_KEYWORD_DICTIONARIES: Dict[DocumentType, List[str]] = {
    DocumentType.INVOICE: [
        "invoice", "inv", "bill to", "sold to", "remit to",
        "invoice number", "invoice date", "due date", "payment terms",
        "subtotal", "tax", "total due", "amount due", "balance due",
        "vendor", "supplier", "purchase order", "po number",
        "net 30", "net 60", "upon receipt",
        "tax id", "ein", "vat", "gst",
        "line item", "quantity", "unit price", "discount",
    ],
    DocumentType.MANIFEST: [
        "manifest", "bill of lading", "bol", "b/l",
        "shipper", "consignee", "carrier", "notify party",
        "port of loading", "port of discharge",
        "vessel", "voyage", "container", "seal",
        "gross weight", "net weight", "tare weight",
        "number of packages", "description of goods",
        "freight", "prepaid", "collect",
        "hazardous", "dangerous goods", "un number",
        "origin", "destination", "shipment date",
        "vehicle id", "truck", "trailer",
    ],
    DocumentType.UTILITY_BILL: [
        "utility", "electric", "electricity", "gas", "water",
        "sewer", "waste", "telecom",
        "meter number", "meter reading", "previous reading",
        "current reading", "consumption", "usage",
        "kwh", "mwh", "therms", "ccf", "gallons",
        "billing period", "service period", "service address",
        "rate", "tariff", "tier", "baseline",
        "account number", "customer number",
    ],
    DocumentType.RECEIPT: [
        "receipt", "sales receipt", "transaction",
        "store", "register", "cashier",
        "subtotal", "tax", "total", "change",
        "cash", "credit card", "debit card",
        "visa", "mastercard", "amex",
        "thank you", "return policy",
        "item", "qty", "price",
    ],
    DocumentType.PURCHASE_ORDER: [
        "purchase order", "po number", "po date",
        "order number", "order date",
        "ship to", "bill to", "vendor",
        "requisition", "authorized by", "approved by",
        "delivery date", "expected delivery",
        "terms and conditions", "shipping method",
        "quantity", "unit price", "total",
        "fob", "freight terms",
    ],
}

# Structural patterns (header/table indicators) per document type
_STRUCTURAL_PATTERNS: Dict[DocumentType, List[str]] = {
    DocumentType.INVOICE: [
        r"(?i)^\s*invoice\s*$",
        r"(?i)invoice\s*#\s*\S+",
        r"(?i)bill\s*to\s*:",
        r"(?i)item\s+(?:description|desc)\s+(?:qty|quantity)\s+(?:price|rate)\s+(?:amount|total)",
        r"(?i)subtotal\s*[\$\u20ac\u00a3]?\s*[\d,]+",
    ],
    DocumentType.MANIFEST: [
        r"(?i)^\s*(?:bill\s*of\s*lading|manifest|bol)\s*$",
        r"(?i)shipper\s*:",
        r"(?i)consignee\s*:",
        r"(?i)description\s+.*weight\s+.*(?:pieces|packages)",
        r"(?i)seal\s*(?:#|no)",
    ],
    DocumentType.UTILITY_BILL: [
        r"(?i)meter\s*reading",
        r"(?i)billing\s*period",
        r"(?i)(?:previous|current)\s*reading\s*[\d,]+",
        r"(?i)consumption\s*[\d,]+\s*(?:kwh|therms|ccf|gallons)",
        r"(?i)service\s*address",
    ],
    DocumentType.RECEIPT: [
        r"(?i)^\s*receipt\s*$",
        r"(?i)(?:store|register)\s*#?\s*\d+",
        r"(?i)cash(?:ier)?\s*:",
        r"(?i)change\s*[\$\u20ac\u00a3]?\s*[\d,]+",
    ],
    DocumentType.PURCHASE_ORDER: [
        r"(?i)^\s*purchase\s*order\s*$",
        r"(?i)p\.?o\.?\s*#\s*\S+",
        r"(?i)ship\s*to\s*:",
        r"(?i)authorized\s*(?:by|signature)",
        r"(?i)delivery\s*date",
    ],
}

# Filename patterns
_FILENAME_PATTERNS: Dict[str, DocumentType] = {
    r"(?i)inv(?:oice)?[\-_\s]": DocumentType.INVOICE,
    r"(?i)manifest|bol|bill[\-_]?of[\-_]?lading": DocumentType.MANIFEST,
    r"(?i)utility|electric|gas[\-_]?bill|water[\-_]?bill": DocumentType.UTILITY_BILL,
    r"(?i)receipt|rcpt": DocumentType.RECEIPT,
    r"(?i)p[\.\-_]?o[\.\-_]?|purchase[\-_]?order": DocumentType.PURCHASE_ORDER,
}

# Default scoring weights
_DEFAULT_WEIGHTS = {
    "keyword": 0.50,
    "structure": 0.35,
    "filename": 0.15,
}


# ---------------------------------------------------------------------------
# DocumentClassifier
# ---------------------------------------------------------------------------


class DocumentClassifier:
    """Document type classifier using keyword and structural scoring.

    Combines three scoring signals -- keyword density, structural pattern
    matches, and filename heuristics -- to produce a confidence-scored
    document type classification.

    Attributes:
        _keyword_dicts: Keyword lists per document type.
        _weights: Scoring weights for each signal.
        _lock: Threading lock for statistics.
        _stats: Classification statistics.

    Example:
        >>> cls = DocumentClassifier()
        >>> doc_type, conf = cls.classify(text, file_name="INV-2024-001.pdf")
        >>> print(doc_type, conf)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DocumentClassifier.

        Args:
            config: Optional configuration dict.  Recognised keys:
                - ``weights``: dict with keyword/structure/filename weights
                - ``min_confidence``: float threshold for UNKNOWN fallback
        """
        self._config = config or {}
        self._keyword_dicts: Dict[DocumentType, List[str]] = {
            k: list(v) for k, v in _KEYWORD_DICTIONARIES.items()
        }
        self._weights: Dict[str, float] = self._config.get(
            "weights", dict(_DEFAULT_WEIGHTS),
        )
        self._min_confidence: float = self._config.get("min_confidence", 0.25)
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "total_classifications": 0,
            "by_type": {},
            "total_confidence": 0.0,
            "unknown_count": 0,
        }
        logger.info(
            "DocumentClassifier initialised: types=%d, weights=%s",
            len(self._keyword_dicts), self._weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        file_name: Optional[str] = None,
    ) -> Tuple[DocumentType, float]:
        """Classify a document by its text content and optional filename.

        Args:
            text: Document text.
            file_name: Optional original file name.

        Returns:
            Tuple of (DocumentType, confidence).
        """
        start = time.monotonic()

        scores: Dict[DocumentType, float] = {}
        text_lower = text.lower()

        kw_weight = self._weights.get("keyword", 0.50)
        st_weight = self._weights.get("structure", 0.35)
        fn_weight = self._weights.get("filename", 0.15)

        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue

            kw_score = self._score_keywords(text_lower, doc_type)
            st_score = self._score_structure(text, doc_type)
            fn_score = 0.0

            if file_name:
                fn_type = self._score_filename(file_name)
                fn_score = 1.0 if fn_type == doc_type else 0.0

            combined = (
                kw_score * kw_weight
                + st_score * st_weight
                + fn_score * fn_weight
            )
            scores[doc_type] = combined

        # Select best
        best_type = DocumentType.UNKNOWN
        best_score = 0.0
        for dt, score in scores.items():
            if score > best_score:
                best_score = score
                best_type = dt

        # Apply minimum confidence threshold
        if best_score < self._min_confidence:
            best_type = DocumentType.UNKNOWN
            best_score = round(best_score, 4)

        confidence = round(min(best_score, 1.0), 4)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Update stats
        with self._lock:
            self._stats["total_classifications"] += 1
            self._stats["total_confidence"] += confidence
            type_key = best_type.value
            self._stats["by_type"][type_key] = (
                self._stats["by_type"].get(type_key, 0) + 1
            )
            if best_type == DocumentType.UNKNOWN:
                self._stats["unknown_count"] += 1

        logger.info(
            "Classified document as '%s' (confidence=%.4f, %.1f ms)",
            best_type.value, confidence, elapsed_ms,
        )
        return best_type, confidence

    def classify_batch(
        self,
        texts: List[str],
        file_names: Optional[List[Optional[str]]] = None,
    ) -> List[Tuple[DocumentType, float]]:
        """Classify a batch of documents.

        Args:
            texts: List of document texts.
            file_names: Optional parallel list of file names.

        Returns:
            List of (DocumentType, confidence) tuples.
        """
        results: List[Tuple[DocumentType, float]] = []
        names = file_names or [None] * len(texts)
        for text, name in zip(texts, names):
            results.append(self.classify(text, name))
        return results

    def get_keyword_dictionary(
        self,
        doc_type: DocumentType,
    ) -> List[str]:
        """Get the keyword list for a document type.

        Args:
            doc_type: Document type.

        Returns:
            List of keyword strings.
        """
        return list(self._keyword_dicts.get(doc_type, []))

    def register_keywords(
        self,
        doc_type: DocumentType,
        keywords: List[str],
    ) -> None:
        """Add custom keywords for a document type.

        Args:
            doc_type: Document type to add keywords for.
            keywords: List of keyword strings to add.
        """
        existing = self._keyword_dicts.setdefault(doc_type, [])
        for kw in keywords:
            lower_kw = kw.lower()
            if lower_kw not in existing:
                existing.append(lower_kw)
        logger.info(
            "Registered %d keywords for '%s' (total: %d)",
            len(keywords), doc_type.value, len(existing),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Return classification statistics.

        Returns:
            Dictionary with total classifications, per-type counts,
            and average confidence.
        """
        with self._lock:
            total = self._stats["total_classifications"]
            avg_conf = (
                self._stats["total_confidence"] / total
                if total > 0
                else 0.0
            )
            return {
                "total_classifications": total,
                "avg_confidence": round(avg_conf, 4),
                "by_type": dict(self._stats["by_type"]),
                "unknown_count": self._stats["unknown_count"],
                "registered_types": len(self._keyword_dicts),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _score_keywords(
        self,
        text_lower: str,
        doc_type: DocumentType,
    ) -> float:
        """Score text against keyword list for a document type.

        Score is the fraction of keywords found in the text, with
        weighting for multi-word phrases (they are more specific).

        Args:
            text_lower: Lowercased document text.
            doc_type: Document type to score against.

        Returns:
            Score between 0.0 and 1.0.
        """
        keywords = self._keyword_dicts.get(doc_type, [])
        if not keywords:
            return 0.0

        total_weight = 0.0
        matched_weight = 0.0

        for kw in keywords:
            # Multi-word phrases get higher weight
            word_count = len(kw.split())
            weight = 1.0 + (word_count - 1) * 0.5

            total_weight += weight
            if kw in text_lower:
                matched_weight += weight

        if total_weight == 0.0:
            return 0.0

        raw_score = matched_weight / total_weight
        # Apply a sigmoid-like curve to spread scores
        # This avoids very low scores from large keyword lists
        return min(raw_score * 2.0, 1.0)

    def _score_structure(
        self,
        text: str,
        doc_type: DocumentType,
    ) -> float:
        """Score structural patterns for a document type.

        Each structural pattern is searched; the score is the fraction
        of patterns that match.

        Args:
            text: Document text (original case).
            doc_type: Document type to score against.

        Returns:
            Score between 0.0 and 1.0.
        """
        patterns = _STRUCTURAL_PATTERNS.get(doc_type, [])
        if not patterns:
            return 0.0

        matches = 0
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.MULTILINE):
                    matches += 1
            except re.error:
                continue

        return matches / len(patterns)

    def _score_filename(
        self,
        file_name: str,
    ) -> Optional[DocumentType]:
        """Attempt classification from filename alone.

        Args:
            file_name: File name string.

        Returns:
            Matched DocumentType or None.
        """
        base = os.path.basename(file_name)
        for pattern, doc_type in _FILENAME_PATTERNS.items():
            if re.search(pattern, base):
                return doc_type
        return None
