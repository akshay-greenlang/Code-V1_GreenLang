# -*- coding: utf-8 -*-
"""
Invoice Processor - AGENT-DATA-001: PDF & Invoice Extractor

Invoice-specific extraction engine that wraps FieldExtractor with
invoice domain logic: header extraction, multi-format line-item parsing,
totals extraction, payment information, and cross-validation of
financial figures (line items sum, tax calculation, total reconciliation).

Features:
    - Full invoice header extraction (invoice #, dates, vendor, buyer)
    - Multi-format line-item table parsing
    - Totals extraction with subtotal/tax/total breakdown
    - Payment info extraction (terms, bank details)
    - Cross-validation of financial consistency
    - Weighted confidence scoring across all extracted fields
    - SHA-256 provenance hashing

Zero-Hallucination Guarantees:
    - All numeric validations are deterministic arithmetic
    - Tax validation uses extracted values, never assumed rates
    - Totals are verified by summing, never estimated

Example:
    >>> from greenlang.pdf_extractor.invoice_processor import InvoiceProcessor
    >>> from greenlang.pdf_extractor.field_extractor import FieldExtractor
    >>> extractor = FieldExtractor()
    >>> processor = InvoiceProcessor(field_extractor=extractor)
    >>> result = processor.process_invoice(invoice_text)
    >>> print(result.invoice_number, result.total_amount)

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.pdf_extractor.field_extractor import (
    ExtractedField,
    FieldExtractor,
    LineItem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationResult",
    "InvoiceExtraction",
    "InvoiceProcessor",
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


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    rule_name: str = Field(..., description="Validation rule identifier")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(default="", description="Human-readable message")
    severity: str = Field(
        default="warning",
        description="Severity: info, warning, error",
    )
    expected: Optional[str] = Field(None, description="Expected value")
    actual: Optional[str] = Field(None, description="Actual value")

    model_config = {"extra": "forbid"}


class InvoiceExtraction(BaseModel):
    """Complete invoice extraction result."""

    invoice_number: Optional[str] = Field(None)
    invoice_date: Optional[str] = Field(None)
    due_date: Optional[str] = Field(None)
    vendor_name: Optional[str] = Field(None)
    vendor_address: Optional[str] = Field(None)
    vendor_tax_id: Optional[str] = Field(None)
    buyer_name: Optional[str] = Field(None)
    buyer_address: Optional[str] = Field(None)
    po_number: Optional[str] = Field(None)
    currency: Optional[str] = Field(None)
    payment_terms: Optional[str] = Field(None)
    subtotal: Optional[float] = Field(None)
    tax_amount: Optional[float] = Field(None)
    total_amount: Optional[float] = Field(None)
    line_items: List[LineItem] = Field(default_factory=list)
    header_fields: Dict[str, Any] = Field(default_factory=dict)
    payment_info: Dict[str, Any] = Field(default_factory=dict)
    validations: List[ValidationResult] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_hash: str = Field(default="")
    extracted_at: datetime = Field(default_factory=_utcnow)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# InvoiceProcessor
# ---------------------------------------------------------------------------


class InvoiceProcessor:
    """Invoice-specific extraction and validation engine.

    Uses FieldExtractor for pattern matching and adds invoice domain
    logic including multi-format table parsing, financial validation,
    and confidence scoring.

    Attributes:
        _field_extractor: FieldExtractor instance for pattern matching.
        _config: Configuration dictionary.
        _lock: Threading lock for statistics.
        _stats: Processing statistics.

    Example:
        >>> processor = InvoiceProcessor()
        >>> result = processor.process_invoice(text)
        >>> assert result.total_amount is not None
    """

    # Confidence weights for different field categories
    _CONFIDENCE_WEIGHTS: Dict[str, float] = {
        "invoice_number": 2.0,
        "invoice_date": 1.5,
        "vendor_name": 1.5,
        "total_amount": 2.0,
        "subtotal": 1.0,
        "tax_amount": 1.0,
        "buyer_name": 1.0,
        "due_date": 0.8,
        "currency": 0.5,
        "po_number": 0.5,
        "payment_terms": 0.5,
        "vendor_tax_id": 0.5,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        field_extractor: Optional[FieldExtractor] = None,
    ) -> None:
        """Initialise InvoiceProcessor.

        Args:
            config: Optional configuration dict.
            field_extractor: Optional FieldExtractor instance.
                Creates a new one if None.
        """
        self._config = config or {}
        self._field_extractor = field_extractor or FieldExtractor(config)
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "invoices_processed": 0,
            "line_items_extracted": 0,
            "validations_run": 0,
            "validation_failures": 0,
            "errors": 0,
        }
        logger.info("InvoiceProcessor initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_invoice(
        self,
        text: str,
        confidence_threshold: float = 0.7,
        template: Optional[Dict[str, List[str]]] = None,
    ) -> InvoiceExtraction:
        """Perform full invoice extraction and validation.

        Args:
            text: Extracted document text.
            confidence_threshold: Minimum field confidence.
            template: Optional custom pattern template.

        Returns:
            InvoiceExtraction with all fields, line items, and validations.
        """
        start = time.monotonic()

        # Step 1: Extract all fields
        fields, meta = self._field_extractor.extract_fields(
            text, "invoice", confidence_threshold, template,
        )

        # Step 2: Build field lookup
        field_map = {f.field_name: f for f in fields}

        # Step 3: Extract header
        header = self.extract_header(text, field_map)

        # Step 4: Extract line items
        line_items = self.extract_line_items(text)

        # Step 5: Extract totals
        totals = self.extract_totals(text, field_map)

        # Step 6: Extract payment info
        payment_info = self.extract_payment_info(text, field_map)

        # Step 7: Build invoice data
        invoice = InvoiceExtraction(
            invoice_number=header.get("invoice_number"),
            invoice_date=header.get("invoice_date"),
            due_date=header.get("due_date"),
            vendor_name=header.get("vendor_name"),
            vendor_address=header.get("vendor_address"),
            vendor_tax_id=header.get("vendor_tax_id"),
            buyer_name=header.get("buyer_name"),
            buyer_address=header.get("buyer_address"),
            po_number=header.get("po_number"),
            currency=header.get("currency"),
            payment_terms=payment_info.get("payment_terms"),
            subtotal=totals.get("subtotal"),
            tax_amount=totals.get("tax_amount"),
            total_amount=totals.get("total_amount"),
            line_items=line_items,
            header_fields=header,
            payment_info=payment_info,
        )

        # Step 8: Validate
        validations = self.validate_invoice(invoice, line_items)
        invoice.validations = validations

        # Step 9: Calculate confidence
        field_confidences = {f.field_name: f.confidence for f in fields}
        invoice.overall_confidence = self.calculate_confidence(field_confidences)

        # Step 10: Provenance hash
        invoice.provenance_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        elapsed_ms = (time.monotonic() - start) * 1000
        with self._lock:
            self._stats["invoices_processed"] += 1
            self._stats["line_items_extracted"] += len(line_items)
            self._stats["validations_run"] += len(validations)
            self._stats["validation_failures"] += sum(
                1 for v in validations if not v.passed
            )

        logger.info(
            "Invoice processed: invoice_number=%s, fields=%d, "
            "line_items=%d, validations=%d (%.1f ms)",
            invoice.invoice_number,
            len(fields),
            len(line_items),
            len(validations),
            elapsed_ms,
        )
        return invoice

    def extract_header(
        self,
        text: str,
        field_map: Optional[Dict[str, ExtractedField]] = None,
    ) -> Dict[str, Any]:
        """Extract invoice header fields.

        Args:
            text: Document text.
            field_map: Optional pre-extracted field map.

        Returns:
            Dictionary of header field values.
        """
        if field_map is None:
            fields, _ = self._field_extractor.extract_fields(text, "invoice")
            field_map = {f.field_name: f for f in fields}

        header_keys = [
            "invoice_number", "invoice_date", "due_date",
            "vendor_name", "vendor_address", "vendor_tax_id",
            "buyer_name", "buyer_address", "currency", "po_number",
        ]

        header: Dict[str, Any] = {}
        for key in header_keys:
            field = field_map.get(key)
            header[key] = field.value if field else None

        return header

    def extract_line_items(self, text: str) -> List[LineItem]:
        """Extract invoice line items.

        Delegates to FieldExtractor and applies invoice-specific
        heuristics for multi-format table detection.

        Args:
            text: Document text.

        Returns:
            List of LineItem objects.
        """
        items = self._field_extractor.extract_line_items(text, "invoice")

        # Additional invoice-specific pattern: numbered lines
        if not items:
            numbered_pat = re.compile(
                r"^\s*(\d+)\.\s+(.+?)\s+"
                r"([\d,]+\.?\d*)\s+"
                r"[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)\s+"
                r"[\$\u20ac\u00a3]?\s*([\d,]+\.?\d*)\s*$",
                re.MULTILINE,
            )
            for m in numbered_pat.finditer(text):
                items.append(LineItem(
                    line_number=int(m.group(1)),
                    description=m.group(2).strip(),
                    quantity=self._safe_float(m.group(3)),
                    unit_price=self._safe_float(m.group(4)),
                    amount=self._safe_float(m.group(5)),
                    raw_text=m.group(0).strip(),
                ))

        return items

    def extract_totals(
        self,
        text: str,
        field_map: Optional[Dict[str, ExtractedField]] = None,
    ) -> Dict[str, Optional[float]]:
        """Extract subtotal, tax, and total amounts.

        Args:
            text: Document text.
            field_map: Optional pre-extracted field map.

        Returns:
            Dictionary with subtotal, tax_amount, total_amount.
        """
        if field_map is None:
            fields, _ = self._field_extractor.extract_fields(text, "invoice")
            field_map = {f.field_name: f for f in fields}

        return {
            "subtotal": self._get_float_value(field_map, "subtotal"),
            "tax_amount": self._get_float_value(field_map, "tax_amount"),
            "total_amount": self._get_float_value(field_map, "total_amount"),
        }

    def extract_payment_info(
        self,
        text: str,
        field_map: Optional[Dict[str, ExtractedField]] = None,
    ) -> Dict[str, Any]:
        """Extract payment-related information.

        Args:
            text: Document text.
            field_map: Optional pre-extracted field map.

        Returns:
            Dictionary with payment terms and bank details.
        """
        if field_map is None:
            fields, _ = self._field_extractor.extract_fields(text, "invoice")
            field_map = {f.field_name: f for f in fields}

        payment: Dict[str, Any] = {
            "payment_terms": None,
            "bank_name": None,
            "account_number": None,
            "routing_number": None,
            "iban": None,
            "swift_bic": None,
        }

        terms_field = field_map.get("payment_terms")
        if terms_field:
            payment["payment_terms"] = terms_field.value

        # Bank detail patterns
        bank_patterns = {
            "bank_name": r"(?i)bank\s*(?:name)?\s*[:\-]?\s*(.+?)(?:\n|$)",
            "account_number": r"(?i)(?:account|acct)\s*(?:#|no\.?|number)\s*[:\-]?\s*(\d[\d\s\-]+)",
            "routing_number": r"(?i)(?:routing|aba|sort\s*code)\s*(?:#|no\.?|number)?\s*[:\-]?\s*(\d[\d\-]+)",
            "iban": r"(?i)iban\s*[:\-]?\s*([A-Z]{2}\d{2}[A-Z0-9\s]{10,30})",
            "swift_bic": r"(?i)(?:swift|bic)\s*[:\-]?\s*([A-Z]{6}[A-Z0-9]{2,5})",
        }

        for key, pattern in bank_patterns.items():
            m = re.search(pattern, text)
            if m:
                payment[key] = m.group(1).strip()

        return payment

    def validate_invoice(
        self,
        invoice_data: InvoiceExtraction,
        line_items: Optional[List[LineItem]] = None,
    ) -> List[ValidationResult]:
        """Cross-validate invoice financial consistency.

        Args:
            invoice_data: Extracted invoice data.
            line_items: Optional line items (uses invoice_data.line_items
                if None).

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []
        items = line_items if line_items is not None else invoice_data.line_items

        # Required fields
        results.extend(self._validate_required_fields(invoice_data))

        # Totals consistency
        results.extend(self._validate_totals(invoice_data))

        # Line items sum vs subtotal
        if items and invoice_data.subtotal is not None:
            results.extend(self._validate_line_items_sum(
                items, invoice_data.subtotal,
            ))

        # Tax calculation
        if (invoice_data.subtotal is not None
                and invoice_data.tax_amount is not None
                and invoice_data.total_amount is not None):
            result = self._validate_tax_calculation(
                invoice_data.subtotal,
                None,  # tax_rate not extracted; only structural check
                invoice_data.tax_amount,
            )
            if result:
                results.append(result)

        # Date ordering
        results.extend(self._validate_date_ordering(invoice_data))

        return results

    def calculate_confidence(
        self,
        field_confidences: Dict[str, float],
    ) -> float:
        """Calculate weighted average confidence across fields.

        Args:
            field_confidences: Dict of field_name -> confidence.

        Returns:
            Weighted average confidence (0.0-1.0).
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for field, confidence in field_confidences.items():
            weight = self._CONFIDENCE_WEIGHTS.get(field, 0.5)
            weighted_sum += confidence * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics.

        Returns:
            Dictionary with counts and rates.
        """
        with self._lock:
            total = self._stats["invoices_processed"]
            return {
                "invoices_processed": total,
                "line_items_extracted": self._stats["line_items_extracted"],
                "validations_run": self._stats["validations_run"],
                "validation_failures": self._stats["validation_failures"],
                "validation_pass_rate": round(
                    1.0 - (
                        self._stats["validation_failures"]
                        / max(self._stats["validations_run"], 1)
                    ), 4,
                ),
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_required_fields(
        self,
        invoice: InvoiceExtraction,
    ) -> List[ValidationResult]:
        """Validate that critical fields are present."""
        results: List[ValidationResult] = []
        required = {
            "invoice_number": invoice.invoice_number,
            "invoice_date": invoice.invoice_date,
            "total_amount": invoice.total_amount,
        }
        for name, value in required.items():
            results.append(ValidationResult(
                rule_name=f"required_{name}",
                passed=value is not None,
                message=(
                    f"{name} present"
                    if value is not None
                    else f"{name} is missing"
                ),
                severity="error" if value is None else "info",
            ))
        return results

    def _validate_totals(
        self,
        invoice: InvoiceExtraction,
    ) -> List[ValidationResult]:
        """Validate total = subtotal + tax."""
        results: List[ValidationResult] = []
        if (invoice.subtotal is not None
                and invoice.tax_amount is not None
                and invoice.total_amount is not None):
            expected = round(invoice.subtotal + invoice.tax_amount, 2)
            actual = round(invoice.total_amount, 2)
            tolerance = 0.02  # allow 2 cent rounding
            passed = abs(expected - actual) <= tolerance
            results.append(ValidationResult(
                rule_name="totals_consistency",
                passed=passed,
                message=(
                    "Total matches subtotal + tax"
                    if passed
                    else f"Total mismatch: expected {expected}, got {actual}"
                ),
                severity="error" if not passed else "info",
                expected=str(expected),
                actual=str(actual),
            ))
        return results

    def _validate_line_items_sum(
        self,
        items: List[LineItem],
        subtotal: float,
    ) -> List[ValidationResult]:
        """Validate line items sum equals subtotal."""
        computed = self._compute_expected_total(items)
        if computed is None:
            return []
        tolerance = 0.05
        passed = abs(computed - subtotal) <= tolerance
        return [ValidationResult(
            rule_name="line_items_sum",
            passed=passed,
            message=(
                "Line items sum matches subtotal"
                if passed
                else f"Line items sum ({computed}) != subtotal ({subtotal})"
            ),
            severity="warning" if not passed else "info",
            expected=str(subtotal),
            actual=str(computed),
        )]

    def _validate_tax_calculation(
        self,
        subtotal: float,
        tax_rate: Optional[float],
        tax_amount: float,
    ) -> Optional[ValidationResult]:
        """Validate tax amount is non-negative and plausible.

        Does NOT assume a tax rate; only checks structural validity.
        """
        if tax_amount < 0:
            return ValidationResult(
                rule_name="tax_non_negative",
                passed=False,
                message=f"Tax amount is negative: {tax_amount}",
                severity="error",
            )
        if subtotal > 0 and tax_amount > subtotal:
            return ValidationResult(
                rule_name="tax_plausibility",
                passed=False,
                message=f"Tax ({tax_amount}) exceeds subtotal ({subtotal})",
                severity="warning",
            )
        return None

    def _validate_date_ordering(
        self,
        invoice: InvoiceExtraction,
    ) -> List[ValidationResult]:
        """Validate invoice_date <= due_date if both are present."""
        if invoice.invoice_date and invoice.due_date:
            try:
                inv_dt = datetime.strptime(invoice.invoice_date, "%Y-%m-%d")
                due_dt = datetime.strptime(invoice.due_date, "%Y-%m-%d")
                passed = inv_dt <= due_dt
                return [ValidationResult(
                    rule_name="date_ordering",
                    passed=passed,
                    message=(
                        "Invoice date <= due date"
                        if passed
                        else f"Invoice date ({invoice.invoice_date}) > "
                             f"due date ({invoice.due_date})"
                    ),
                    severity="warning" if not passed else "info",
                )]
            except ValueError:
                pass
        return []

    def _compute_expected_total(
        self,
        line_items: List[LineItem],
    ) -> Optional[float]:
        """Sum line item amounts.

        Args:
            line_items: List of line items.

        Returns:
            Sum of amounts, or None if no amounts.
        """
        amounts = [
            item.amount for item in line_items if item.amount is not None
        ]
        if not amounts:
            return None
        return round(sum(amounts), 2)

    def _get_float_value(
        self,
        field_map: Dict[str, ExtractedField],
        field_name: str,
    ) -> Optional[float]:
        """Safely get float value from field map."""
        field = field_map.get(field_name)
        if field is None or field.value is None:
            return None
        try:
            return float(field.value)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float."""
        try:
            return float(value.replace(",", ""))
        except (ValueError, AttributeError):
            return None
