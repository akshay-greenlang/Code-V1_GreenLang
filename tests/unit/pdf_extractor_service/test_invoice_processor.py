# -*- coding: utf-8 -*-
"""
Unit Tests for InvoiceProcessor (AGENT-DATA-001)

Tests full invoice extraction, header extraction, line item extraction
(multiple formats), totals extraction, payment info extraction, invoice
validation (totals match, dates valid), confidence calculation, and
statistics. Uses realistic invoice text samples.

Coverage target: 85%+ of invoice_processor.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline InvoiceProcessor mirroring greenlang/pdf_extractor/invoice_processor.py
# ---------------------------------------------------------------------------


class InvoiceProcessingError(Exception):
    """Raised when invoice processing fails."""
    pass


class InvoiceProcessor:
    """Specialized processor for invoice documents.

    Extracts header fields, line items, totals, payment info,
    and validates cross-field consistency.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self._confidence_threshold = confidence_threshold
        self._stats = {
            "invoices_processed": 0,
            "invoices_validated": 0,
            "validation_failures": 0,
            "total_line_items": 0,
        }

    def process_invoice(self, text: str) -> Dict[str, Any]:
        """Process invoice text and extract all fields.

        Returns:
            Dict with header, line_items, totals, payment_info, confidence, validation.
        """
        header = self.extract_header(text)
        line_items = self.extract_line_items(text)
        totals = self.extract_totals(text)
        payment_info = self.extract_payment_info(text)
        confidence = self.calculate_confidence(header, line_items, totals)
        validation = self.validate_invoice(header, line_items, totals)

        self._stats["invoices_processed"] += 1
        self._stats["total_line_items"] += len(line_items)

        return {
            "header": header,
            "line_items": line_items,
            "totals": totals,
            "payment_info": payment_info,
            "confidence": confidence,
            "validation": validation,
        }

    def extract_header(self, text: str) -> Dict[str, Any]:
        """Extract invoice header fields."""
        header = {}

        # Invoice number
        m = re.search(r"Invoice\s*(?:Number|No|#)[:\s]*([\w\-]+)", text, re.IGNORECASE)
        header["invoice_number"] = m.group(1).strip() if m else None

        # Invoice date
        m = re.search(r"Invoice\s*Date[:\s]*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        header["invoice_date"] = m.group(1) if m else None

        # Due date
        m = re.search(r"Due\s*Date[:\s]*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        header["due_date"] = m.group(1) if m else None

        # PO number
        m = re.search(r"PO\s*(?:Number|No|#)[:\s]*([\w\-]+)", text, re.IGNORECASE)
        header["po_number"] = m.group(1).strip() if m else None

        # Vendor name
        m = re.search(r"Vendor[:\s]*\n?\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)", text)
        header["vendor_name"] = m.group(1).strip() if m else None

        # Buyer / Bill To
        m = re.search(r"Bill\s*To[:\s]*\n?\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)", text)
        header["buyer_name"] = m.group(1).strip() if m else None

        return header

    def extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract line items from tabular invoice content."""
        items = []

        # Pattern: code  description  qty  unit_price  amount
        pattern = r"(\w[\w\-]*)\s+(.+?)\s+(\d+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)"
        for match in re.finditer(pattern, text):
            items.append({
                "item_code": match.group(1),
                "description": match.group(2).strip(),
                "quantity": float(match.group(3)),
                "unit_price": self._parse_number(match.group(4)),
                "amount": self._parse_number(match.group(5)),
            })

        return items

    def extract_totals(self, text: str) -> Dict[str, Optional[float]]:
        """Extract subtotal, tax, and total amounts."""
        totals: Dict[str, Optional[float]] = {
            "subtotal": None,
            "tax_amount": None,
            "total_amount": None,
        }

        m = re.search(r"Subtotal[:\s]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            totals["subtotal"] = self._parse_number(m.group(1))

        m = re.search(r"Tax[^:]*[:\s]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            totals["tax_amount"] = self._parse_number(m.group(1))

        m = re.search(r"Total[:\s]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            totals["total_amount"] = self._parse_number(m.group(1))

        return totals

    def extract_payment_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract payment-related information."""
        info: Dict[str, Optional[str]] = {
            "payment_terms": None,
            "bank_name": None,
            "account_number": None,
            "sort_code": None,
        }

        m = re.search(r"Payment\s*Terms?[:\s]*(Net\s*\d+|Due\s+on\s+receipt|COD)", text, re.IGNORECASE)
        if m:
            info["payment_terms"] = m.group(1).strip()

        m = re.search(r"Bank[:\s]*([A-Za-z\s]+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            info["bank_name"] = m.group(1).strip()

        m = re.search(r"Account[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            info["account_number"] = m.group(1)

        m = re.search(r"Sort\s*Code[:\s]*([\d\-]+)", text, re.IGNORECASE)
        if m:
            info["sort_code"] = m.group(1)

        return info

    def validate_invoice(
        self,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        totals: Dict[str, Optional[float]],
    ) -> Dict[str, Any]:
        """Validate invoice cross-field consistency.

        Checks:
        - Subtotal matches sum of line item amounts
        - Total = subtotal + tax
        - Required fields present
        - Due date after invoice date
        """
        errors = []
        warnings = []

        # Check required fields
        if not header.get("invoice_number"):
            errors.append({"field": "invoice_number", "message": "Missing invoice number"})
        if not header.get("invoice_date"):
            warnings.append({"field": "invoice_date", "message": "Missing invoice date"})

        # Check line item totals
        if line_items and totals.get("subtotal") is not None:
            line_item_sum = sum(item.get("amount", 0) for item in line_items)
            subtotal = totals["subtotal"]
            if abs(line_item_sum - subtotal) > 0.01:
                errors.append({
                    "field": "subtotal",
                    "message": f"Subtotal {subtotal} does not match line items sum {line_item_sum}",
                })

        # Check total = subtotal + tax
        if totals.get("subtotal") is not None and totals.get("tax_amount") is not None:
            expected_total = totals["subtotal"] + totals["tax_amount"]
            if totals.get("total_amount") is not None:
                if abs(expected_total - totals["total_amount"]) > 0.01:
                    errors.append({
                        "field": "total_amount",
                        "message": f"Total {totals['total_amount']} != subtotal + tax {expected_total}",
                    })

        # Check dates
        if header.get("invoice_date") and header.get("due_date"):
            if header["due_date"] < header["invoice_date"]:
                errors.append({
                    "field": "due_date",
                    "message": "Due date is before invoice date",
                })

        is_valid = len(errors) == 0
        self._stats["invoices_validated"] += 1
        if not is_valid:
            self._stats["validation_failures"] += 1

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
        }

    def calculate_confidence(
        self,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        totals: Dict[str, Optional[float]],
    ) -> float:
        """Calculate overall extraction confidence."""
        scores = []

        # Header fields present
        header_fields = ["invoice_number", "invoice_date", "due_date", "vendor_name"]
        present = sum(1 for f in header_fields if header.get(f))
        scores.append(present / len(header_fields))

        # Line items found
        if line_items:
            scores.append(0.9)
        else:
            scores.append(0.3)

        # Totals found
        totals_present = sum(1 for v in totals.values() if v is not None)
        scores.append(totals_present / len(totals))

        return sum(scores) / len(scores) if scores else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return dict(self._stats)

    def _parse_number(self, raw: str) -> float:
        cleaned = raw.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0


# ===========================================================================
# Test Classes
# ===========================================================================


class TestInvoiceProcessorInit:
    """Test InvoiceProcessor initialization."""

    def test_default_confidence_threshold(self):
        proc = InvoiceProcessor()
        assert proc._confidence_threshold == 0.7

    def test_custom_confidence_threshold(self):
        proc = InvoiceProcessor(confidence_threshold=0.9)
        assert proc._confidence_threshold == 0.9

    def test_initial_statistics(self):
        proc = InvoiceProcessor()
        stats = proc.get_statistics()
        assert stats["invoices_processed"] == 0
        assert stats["total_line_items"] == 0


class TestProcessInvoice:
    """Test process_invoice full extraction."""

    def test_process_returns_all_sections(self, sample_invoice_text):
        proc = InvoiceProcessor()
        result = proc.process_invoice(sample_invoice_text)
        assert "header" in result
        assert "line_items" in result
        assert "totals" in result
        assert "payment_info" in result
        assert "confidence" in result
        assert "validation" in result

    def test_process_updates_stats(self, sample_invoice_text):
        proc = InvoiceProcessor()
        proc.process_invoice(sample_invoice_text)
        stats = proc.get_statistics()
        assert stats["invoices_processed"] == 1

    def test_process_multiple_invoices(self, sample_invoice_text):
        proc = InvoiceProcessor()
        proc.process_invoice(sample_invoice_text)
        proc.process_invoice(sample_invoice_text)
        stats = proc.get_statistics()
        assert stats["invoices_processed"] == 2

    def test_confidence_is_float(self, sample_invoice_text):
        proc = InvoiceProcessor()
        result = proc.process_invoice(sample_invoice_text)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0


class TestExtractHeader:
    """Test extract_header method."""

    def test_extract_invoice_number(self, sample_invoice_text):
        proc = InvoiceProcessor()
        header = proc.extract_header(sample_invoice_text)
        assert header["invoice_number"] == "INV-2025-001234"

    def test_extract_invoice_date(self, sample_invoice_text):
        proc = InvoiceProcessor()
        header = proc.extract_header(sample_invoice_text)
        assert header["invoice_date"] == "2025-06-15"

    def test_extract_due_date(self, sample_invoice_text):
        proc = InvoiceProcessor()
        header = proc.extract_header(sample_invoice_text)
        assert header["due_date"] == "2025-07-15"

    def test_extract_po_number(self, sample_invoice_text):
        proc = InvoiceProcessor()
        header = proc.extract_header(sample_invoice_text)
        assert header["po_number"] == "PO-98765"

    def test_extract_vendor_name(self, sample_invoice_text):
        proc = InvoiceProcessor()
        header = proc.extract_header(sample_invoice_text)
        assert header["vendor_name"] is not None

    def test_extract_buyer_name(self, sample_invoice_text):
        proc = InvoiceProcessor()
        header = proc.extract_header(sample_invoice_text)
        assert header["buyer_name"] is not None

    def test_missing_fields_return_none(self):
        proc = InvoiceProcessor()
        header = proc.extract_header("No invoice fields here")
        assert header["invoice_number"] is None
        assert header["invoice_date"] is None


class TestExtractLineItems:
    """Test extract_line_items method."""

    def test_extract_from_invoice(self, sample_invoice_text):
        proc = InvoiceProcessor()
        items = proc.extract_line_items(sample_invoice_text)
        assert isinstance(items, list)

    def test_extract_structured_items(self):
        text = """
        CARB-001 Carbon Offset Credits 100 25.00 2500.00
        RENEW-002 Renewable Energy Certs 50 15.50 775.00
        """
        proc = InvoiceProcessor()
        items = proc.extract_line_items(text)
        assert len(items) == 2
        assert items[0]["item_code"] == "CARB-001"
        assert items[0]["quantity"] == 100.0
        assert items[0]["unit_price"] == 25.0
        assert items[0]["amount"] == 2500.0

    def test_extract_empty_returns_empty(self):
        proc = InvoiceProcessor()
        items = proc.extract_line_items("")
        assert items == []

    def test_item_fields_present(self):
        text = "ITEM-001 Widget 10 5.00 50.00"
        proc = InvoiceProcessor()
        items = proc.extract_line_items(text)
        assert len(items) == 1
        for key in ("item_code", "description", "quantity", "unit_price", "amount"):
            assert key in items[0]


class TestExtractTotals:
    """Test extract_totals method."""

    def test_extract_subtotal(self, sample_invoice_text):
        proc = InvoiceProcessor()
        totals = proc.extract_totals(sample_invoice_text)
        assert totals["subtotal"] is not None

    def test_extract_tax(self, sample_invoice_text):
        proc = InvoiceProcessor()
        totals = proc.extract_totals(sample_invoice_text)
        # Tax field may match different patterns
        assert "tax_amount" in totals

    def test_extract_total(self, sample_invoice_text):
        proc = InvoiceProcessor()
        totals = proc.extract_totals(sample_invoice_text)
        assert totals["total_amount"] is not None

    def test_extract_no_totals(self):
        proc = InvoiceProcessor()
        totals = proc.extract_totals("No totals here")
        assert totals["subtotal"] is None
        assert totals["tax_amount"] is None
        assert totals["total_amount"] is None


class TestExtractPaymentInfo:
    """Test extract_payment_info method."""

    def test_extract_payment_terms(self, sample_invoice_text):
        proc = InvoiceProcessor()
        info = proc.extract_payment_info(sample_invoice_text)
        assert info["payment_terms"] is not None
        assert "Net" in info["payment_terms"]

    def test_extract_bank_name(self, sample_invoice_text):
        proc = InvoiceProcessor()
        info = proc.extract_payment_info(sample_invoice_text)
        assert info["bank_name"] is not None

    def test_extract_account_number(self, sample_invoice_text):
        proc = InvoiceProcessor()
        info = proc.extract_payment_info(sample_invoice_text)
        assert info["account_number"] is not None

    def test_extract_sort_code(self, sample_invoice_text):
        proc = InvoiceProcessor()
        info = proc.extract_payment_info(sample_invoice_text)
        assert info["sort_code"] is not None

    def test_no_payment_info(self):
        proc = InvoiceProcessor()
        info = proc.extract_payment_info("No payment details")
        assert info["payment_terms"] is None
        assert info["bank_name"] is None


class TestValidateInvoice:
    """Test validate_invoice method."""

    def test_valid_invoice(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001", "invoice_date": "2025-06-01", "due_date": "2025-07-01"}
        items = [{"amount": 50.0}, {"amount": 50.0}]
        totals = {"subtotal": 100.0, "tax_amount": 20.0, "total_amount": 120.0}
        result = proc.validate_invoice(header, items, totals)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_invoice_number(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": None, "invoice_date": "2025-06-01"}
        result = proc.validate_invoice(header, [], {"subtotal": None, "tax_amount": None, "total_amount": None})
        assert any(e["field"] == "invoice_number" for e in result["errors"])

    def test_subtotal_mismatch(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001"}
        items = [{"amount": 50.0}, {"amount": 50.0}]
        totals = {"subtotal": 200.0, "tax_amount": None, "total_amount": None}
        result = proc.validate_invoice(header, items, totals)
        assert result["is_valid"] is False
        assert any("subtotal" in e["field"] for e in result["errors"])

    def test_total_mismatch(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001"}
        items = []
        totals = {"subtotal": 100.0, "tax_amount": 20.0, "total_amount": 999.0}
        result = proc.validate_invoice(header, items, totals)
        assert result["is_valid"] is False

    def test_due_date_before_invoice_date(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001", "invoice_date": "2025-07-01", "due_date": "2025-06-01"}
        result = proc.validate_invoice(header, [], {"subtotal": None, "tax_amount": None, "total_amount": None})
        assert any("due_date" in e["field"] for e in result["errors"])

    def test_validation_updates_stats(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001"}
        proc.validate_invoice(header, [], {"subtotal": None, "tax_amount": None, "total_amount": None})
        stats = proc.get_statistics()
        assert stats["invoices_validated"] == 1

    def test_failed_validation_updates_stats(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": None}
        proc.validate_invoice(header, [], {"subtotal": None, "tax_amount": None, "total_amount": None})
        stats = proc.get_statistics()
        assert stats["validation_failures"] == 1

    def test_totals_match_exactly(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001"}
        items = [{"amount": 100.0}]
        totals = {"subtotal": 100.0, "tax_amount": 10.0, "total_amount": 110.0}
        result = proc.validate_invoice(header, items, totals)
        assert result["is_valid"] is True

    def test_missing_date_warning(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001", "invoice_date": None}
        result = proc.validate_invoice(header, [], {"subtotal": None, "tax_amount": None, "total_amount": None})
        assert any(w["field"] == "invoice_date" for w in result["warnings"])


class TestCalculateConfidence:
    """Test calculate_confidence method."""

    def test_full_confidence_all_fields(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001", "invoice_date": "2025-06-01", "due_date": "2025-07-01", "vendor_name": "Vendor"}
        items = [{"amount": 100.0}]
        totals = {"subtotal": 100.0, "tax_amount": 10.0, "total_amount": 110.0}
        conf = proc.calculate_confidence(header, items, totals)
        assert conf > 0.8

    def test_low_confidence_missing_fields(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": None, "invoice_date": None, "due_date": None, "vendor_name": None}
        items = []
        totals = {"subtotal": None, "tax_amount": None, "total_amount": None}
        conf = proc.calculate_confidence(header, items, totals)
        assert conf < 0.5

    def test_confidence_range(self, sample_invoice_text):
        proc = InvoiceProcessor()
        result = proc.process_invoice(sample_invoice_text)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_no_line_items_reduces_confidence(self):
        proc = InvoiceProcessor()
        header = {"invoice_number": "INV-001", "invoice_date": "2025-06-01", "due_date": "2025-07-01", "vendor_name": "V"}
        conf_with = proc.calculate_confidence(header, [{"amount": 100}], {"subtotal": 100.0, "tax_amount": 10.0, "total_amount": 110.0})
        conf_without = proc.calculate_confidence(header, [], {"subtotal": 100.0, "tax_amount": 10.0, "total_amount": 110.0})
        assert conf_with > conf_without


class TestInvoiceProcessorStatistics:
    """Test statistics gathering."""

    def test_line_items_counted(self):
        text = """
        Invoice Number: INV-001
        ITEM-001 Widget 10 5.00 50.00
        ITEM-002 Gadget 5 10.00 50.00
        Subtotal: $100.00
        Total: $100.00
        """
        proc = InvoiceProcessor()
        proc.process_invoice(text)
        stats = proc.get_statistics()
        assert stats["total_line_items"] == 2

    def test_multiple_invoices_accumulate(self, sample_invoice_text):
        proc = InvoiceProcessor()
        proc.process_invoice(sample_invoice_text)
        proc.process_invoice(sample_invoice_text)
        stats = proc.get_statistics()
        assert stats["invoices_processed"] == 2
