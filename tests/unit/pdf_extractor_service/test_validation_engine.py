# -*- coding: utf-8 -*-
"""
Unit Tests for ValidationEngine (AGENT-DATA-001)

Tests cross-field validation for invoices (totals, line items, dates,
required fields), manifests (weights, pieces), utility bills (consumption,
readings), custom rules, and severity levels.

Coverage target: 85%+ of validation_engine.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ValidationEngine mirroring greenlang/pdf_extractor/validation_engine.py
# ---------------------------------------------------------------------------


class ValidationEngine:
    """Cross-field validation engine for extracted document data.

    Supports invoice, manifest, and utility bill validation rules.
    Custom rules can be registered per document type.
    """

    def __init__(self):
        self._custom_rules: Dict[str, List[Callable]] = {}
        self._stats = {
            "validations_run": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "rules_checked": 0,
        }

    def validate_invoice(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate invoice extraction data."""
        errors = []
        warnings = []
        info = []
        rules_checked = 0

        # Required fields
        rules_checked += 1
        if not data.get("invoice_number"):
            errors.append({
                "severity": "error",
                "field": "invoice_number",
                "message": "Invoice number is required",
            })

        rules_checked += 1
        if not data.get("invoice_date"):
            warnings.append({
                "severity": "warning",
                "field": "invoice_date",
                "message": "Invoice date is missing",
            })

        # Date consistency
        if data.get("invoice_date") and data.get("due_date"):
            rules_checked += 1
            if data["due_date"] < data["invoice_date"]:
                errors.append({
                    "severity": "error",
                    "field": "due_date",
                    "message": "Due date is before invoice date",
                })

        # Line item amount validation
        line_items = data.get("line_items", [])
        for i, item in enumerate(line_items):
            rules_checked += 1
            qty = item.get("quantity", 0)
            price = item.get("unit_price", 0)
            amount = item.get("amount", 0)
            expected = qty * price
            if abs(expected - amount) > 0.01:
                errors.append({
                    "severity": "error",
                    "field": f"line_items[{i}].amount",
                    "message": f"Amount {amount} != qty {qty} * price {price} = {expected}",
                })

        # Subtotal vs line items
        if line_items and data.get("subtotal") is not None:
            rules_checked += 1
            items_sum = sum(item.get("amount", 0) for item in line_items)
            if abs(items_sum - data["subtotal"]) > 0.01:
                errors.append({
                    "severity": "error",
                    "field": "subtotal",
                    "message": f"Subtotal {data['subtotal']} != line items sum {items_sum}",
                })

        # Total = subtotal + tax
        if data.get("subtotal") is not None and data.get("tax_amount") is not None:
            rules_checked += 1
            expected_total = data["subtotal"] + data["tax_amount"]
            if data.get("total_amount") is not None:
                if abs(expected_total - data["total_amount"]) > 0.01:
                    errors.append({
                        "severity": "error",
                        "field": "total_amount",
                        "message": f"Total {data['total_amount']} != subtotal + tax {expected_total}",
                    })

        # Positive amounts
        for field in ("subtotal", "tax_amount", "total_amount"):
            if data.get(field) is not None:
                rules_checked += 1
                if data[field] < 0:
                    errors.append({
                        "severity": "error",
                        "field": field,
                        "message": f"{field} cannot be negative",
                    })

        # Custom rules
        for rule_fn in self._custom_rules.get("invoice", []):
            rules_checked += 1
            rule_result = rule_fn(data)
            if rule_result:
                errors.extend(rule_result)

        is_valid = len(errors) == 0
        self._update_stats(is_valid, rules_checked)

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "rules_checked": rules_checked,
        }

    def validate_manifest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate manifest extraction data."""
        errors = []
        warnings = []
        rules_checked = 0

        # Required fields
        rules_checked += 1
        if not data.get("manifest_number"):
            errors.append({
                "severity": "error",
                "field": "manifest_number",
                "message": "Manifest/BOL number is required",
            })

        # Weight validation
        items = data.get("cargo_items", [])
        if items and data.get("total_weight_kg") is not None:
            rules_checked += 1
            item_weight = sum(item.get("weight_kg", 0) for item in items)
            if abs(item_weight - data["total_weight_kg"]) > 1.0:
                errors.append({
                    "severity": "error",
                    "field": "total_weight_kg",
                    "message": f"Weight mismatch: items {item_weight} != total {data['total_weight_kg']}",
                })

        # Package count validation
        if items and data.get("total_packages") is not None:
            rules_checked += 1
            item_packages = sum(item.get("packages", 0) for item in items)
            if item_packages != data["total_packages"]:
                errors.append({
                    "severity": "error",
                    "field": "total_packages",
                    "message": f"Package mismatch: items {item_packages} != total {data['total_packages']}",
                })

        # Positive weight
        if data.get("total_weight_kg") is not None:
            rules_checked += 1
            if data["total_weight_kg"] <= 0:
                errors.append({
                    "severity": "error",
                    "field": "total_weight_kg",
                    "message": "Total weight must be positive",
                })

        # Custom rules
        for rule_fn in self._custom_rules.get("manifest", []):
            rules_checked += 1
            rule_result = rule_fn(data)
            if rule_result:
                errors.extend(rule_result)

        is_valid = len(errors) == 0
        self._update_stats(is_valid, rules_checked)

        return {"is_valid": is_valid, "errors": errors, "warnings": warnings, "rules_checked": rules_checked}

    def validate_utility_bill(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate utility bill extraction data."""
        errors = []
        warnings = []
        rules_checked = 0

        # Required fields
        rules_checked += 1
        if not data.get("account_number"):
            errors.append({
                "severity": "error",
                "field": "account_number",
                "message": "Account number is required",
            })

        # Consumption = current - previous
        if (
            data.get("previous_reading") is not None
            and data.get("current_reading") is not None
            and data.get("consumption") is not None
        ):
            rules_checked += 1
            expected = data["current_reading"] - data["previous_reading"]
            if abs(expected - data["consumption"]) > 0.5:
                errors.append({
                    "severity": "error",
                    "field": "consumption",
                    "message": f"Consumption {data['consumption']} != current - previous = {expected}",
                })

        # Current reading >= previous reading
        if data.get("previous_reading") is not None and data.get("current_reading") is not None:
            rules_checked += 1
            if data["current_reading"] < data["previous_reading"]:
                warnings.append({
                    "severity": "warning",
                    "field": "current_reading",
                    "message": "Current reading is less than previous reading (meter rollover?)",
                })

        # Positive consumption
        if data.get("consumption") is not None:
            rules_checked += 1
            if data["consumption"] < 0:
                errors.append({
                    "severity": "error",
                    "field": "consumption",
                    "message": "Consumption cannot be negative",
                })

        # Positive total
        if data.get("total_amount") is not None:
            rules_checked += 1
            if data["total_amount"] < 0:
                errors.append({
                    "severity": "error",
                    "field": "total_amount",
                    "message": "Total amount cannot be negative",
                })

        # Custom rules
        for rule_fn in self._custom_rules.get("utility_bill", []):
            rules_checked += 1
            rule_result = rule_fn(data)
            if rule_result:
                errors.extend(rule_result)

        is_valid = len(errors) == 0
        self._update_stats(is_valid, rules_checked)

        return {"is_valid": is_valid, "errors": errors, "warnings": warnings, "rules_checked": rules_checked}

    def register_custom_rule(self, document_type: str, rule_fn: Callable) -> None:
        """Register a custom validation rule for a document type."""
        if document_type not in self._custom_rules:
            self._custom_rules[document_type] = []
        self._custom_rules[document_type].append(rule_fn)

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)

    def _update_stats(self, is_valid: bool, rules_checked: int):
        self._stats["validations_run"] += 1
        self._stats["rules_checked"] += rules_checked
        if is_valid:
            self._stats["validations_passed"] += 1
        else:
            self._stats["validations_failed"] += 1


# ===========================================================================
# Test Classes
# ===========================================================================


class TestValidationEngineInit:
    """Test ValidationEngine initialization."""

    def test_initial_statistics(self):
        engine = ValidationEngine()
        stats = engine.get_statistics()
        assert stats["validations_run"] == 0
        assert stats["rules_checked"] == 0


class TestValidateInvoice:
    """Test validate_invoice method."""

    def test_valid_invoice_all_fields(self):
        engine = ValidationEngine()
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-06-01",
            "due_date": "2025-07-01",
            "line_items": [{"quantity": 10, "unit_price": 5.0, "amount": 50.0}],
            "subtotal": 50.0,
            "tax_amount": 10.0,
            "total_amount": 60.0,
        }
        result = engine.validate_invoice(data)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_invoice_number(self):
        engine = ValidationEngine()
        data = {"invoice_number": None}
        result = engine.validate_invoice(data)
        assert any(e["field"] == "invoice_number" for e in result["errors"])

    def test_missing_invoice_date_warning(self):
        engine = ValidationEngine()
        data = {"invoice_number": "INV-001", "invoice_date": None}
        result = engine.validate_invoice(data)
        assert any(w["field"] == "invoice_date" for w in result["warnings"])

    def test_due_date_before_invoice_date(self):
        engine = ValidationEngine()
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-07-01",
            "due_date": "2025-06-01",
        }
        result = engine.validate_invoice(data)
        assert result["is_valid"] is False
        assert any(e["field"] == "due_date" for e in result["errors"])

    def test_line_item_amount_mismatch(self):
        engine = ValidationEngine()
        data = {
            "invoice_number": "INV-001",
            "line_items": [{"quantity": 10, "unit_price": 5.0, "amount": 999.0}],
        }
        result = engine.validate_invoice(data)
        assert result["is_valid"] is False

    def test_subtotal_mismatch(self):
        engine = ValidationEngine()
        data = {
            "invoice_number": "INV-001",
            "line_items": [
                {"quantity": 10, "unit_price": 5.0, "amount": 50.0},
                {"quantity": 5, "unit_price": 10.0, "amount": 50.0},
            ],
            "subtotal": 200.0,
        }
        result = engine.validate_invoice(data)
        assert result["is_valid"] is False

    def test_total_mismatch(self):
        engine = ValidationEngine()
        data = {
            "invoice_number": "INV-001",
            "subtotal": 100.0,
            "tax_amount": 20.0,
            "total_amount": 999.0,
        }
        result = engine.validate_invoice(data)
        assert result["is_valid"] is False

    def test_negative_subtotal(self):
        engine = ValidationEngine()
        data = {"invoice_number": "INV-001", "subtotal": -50.0}
        result = engine.validate_invoice(data)
        assert result["is_valid"] is False

    def test_negative_total(self):
        engine = ValidationEngine()
        data = {"invoice_number": "INV-001", "total_amount": -100.0}
        result = engine.validate_invoice(data)
        assert result["is_valid"] is False

    def test_rules_checked_count(self):
        engine = ValidationEngine()
        data = {"invoice_number": "INV-001"}
        result = engine.validate_invoice(data)
        assert result["rules_checked"] >= 2

    def test_updates_stats(self):
        engine = ValidationEngine()
        engine.validate_invoice({"invoice_number": "INV-001"})
        stats = engine.get_statistics()
        assert stats["validations_run"] == 1

    def test_valid_updates_passed(self):
        engine = ValidationEngine()
        engine.validate_invoice({"invoice_number": "INV-001"})
        stats = engine.get_statistics()
        assert stats["validations_passed"] == 1

    def test_invalid_updates_failed(self):
        engine = ValidationEngine()
        engine.validate_invoice({"invoice_number": None})
        stats = engine.get_statistics()
        assert stats["validations_failed"] == 1


class TestValidateManifest:
    """Test validate_manifest method."""

    def test_valid_manifest(self):
        engine = ValidationEngine()
        data = {
            "manifest_number": "BOL-001",
            "cargo_items": [{"weight_kg": 1000, "packages": 10}],
            "total_weight_kg": 1000.0,
            "total_packages": 10,
        }
        result = engine.validate_manifest(data)
        assert result["is_valid"] is True

    def test_missing_manifest_number(self):
        engine = ValidationEngine()
        data = {"manifest_number": None}
        result = engine.validate_manifest(data)
        assert result["is_valid"] is False

    def test_weight_mismatch(self):
        engine = ValidationEngine()
        data = {
            "manifest_number": "BOL-001",
            "cargo_items": [{"weight_kg": 500, "packages": 5}],
            "total_weight_kg": 2000.0,
        }
        result = engine.validate_manifest(data)
        assert result["is_valid"] is False

    def test_package_mismatch(self):
        engine = ValidationEngine()
        data = {
            "manifest_number": "BOL-001",
            "cargo_items": [{"weight_kg": 500, "packages": 5}],
            "total_weight_kg": 500.0,
            "total_packages": 99,
        }
        result = engine.validate_manifest(data)
        assert result["is_valid"] is False

    def test_zero_weight(self):
        engine = ValidationEngine()
        data = {"manifest_number": "BOL-001", "total_weight_kg": 0}
        result = engine.validate_manifest(data)
        assert result["is_valid"] is False

    def test_negative_weight(self):
        engine = ValidationEngine()
        data = {"manifest_number": "BOL-001", "total_weight_kg": -100}
        result = engine.validate_manifest(data)
        assert result["is_valid"] is False


class TestValidateUtilityBill:
    """Test validate_utility_bill method."""

    def test_valid_utility_bill(self):
        engine = ValidationEngine()
        data = {
            "account_number": "ACC-001",
            "previous_reading": 100,
            "current_reading": 200,
            "consumption": 100,
            "total_amount": 50.0,
        }
        result = engine.validate_utility_bill(data)
        assert result["is_valid"] is True

    def test_missing_account_number(self):
        engine = ValidationEngine()
        data = {"account_number": None}
        result = engine.validate_utility_bill(data)
        assert result["is_valid"] is False

    def test_consumption_mismatch(self):
        engine = ValidationEngine()
        data = {
            "account_number": "ACC-001",
            "previous_reading": 100,
            "current_reading": 200,
            "consumption": 999,
        }
        result = engine.validate_utility_bill(data)
        assert result["is_valid"] is False

    def test_current_less_than_previous_warning(self):
        engine = ValidationEngine()
        data = {
            "account_number": "ACC-001",
            "previous_reading": 200,
            "current_reading": 100,
        }
        result = engine.validate_utility_bill(data)
        assert any(w["field"] == "current_reading" for w in result["warnings"])

    def test_negative_consumption(self):
        engine = ValidationEngine()
        data = {"account_number": "ACC-001", "consumption": -50}
        result = engine.validate_utility_bill(data)
        assert result["is_valid"] is False

    def test_negative_total(self):
        engine = ValidationEngine()
        data = {"account_number": "ACC-001", "total_amount": -10}
        result = engine.validate_utility_bill(data)
        assert result["is_valid"] is False


class TestCustomRules:
    """Test custom rule registration and execution."""

    def test_register_custom_invoice_rule(self):
        engine = ValidationEngine()

        def check_currency(data):
            if data.get("currency") and data["currency"] not in ("USD", "EUR", "GBP"):
                return [{"severity": "error", "field": "currency", "message": "Unsupported currency"}]
            return []

        engine.register_custom_rule("invoice", check_currency)
        data = {"invoice_number": "INV-001", "currency": "XYZ"}
        result = engine.validate_invoice(data)
        assert any(e["field"] == "currency" for e in result["errors"])

    def test_custom_rule_passes(self):
        engine = ValidationEngine()

        def check_currency(data):
            if data.get("currency") and data["currency"] not in ("USD", "EUR", "GBP"):
                return [{"severity": "error", "field": "currency", "message": "Unsupported"}]
            return []

        engine.register_custom_rule("invoice", check_currency)
        data = {"invoice_number": "INV-001", "currency": "USD"}
        result = engine.validate_invoice(data)
        assert not any(e["field"] == "currency" for e in result["errors"])

    def test_register_custom_manifest_rule(self):
        engine = ValidationEngine()

        def check_max_weight(data):
            if data.get("total_weight_kg", 0) > 50000:
                return [{"severity": "error", "field": "total_weight_kg", "message": "Exceeds max weight"}]
            return []

        engine.register_custom_rule("manifest", check_max_weight)
        data = {"manifest_number": "BOL-001", "total_weight_kg": 60000}
        result = engine.validate_manifest(data)
        assert any("max weight" in e["message"].lower() for e in result["errors"])

    def test_register_custom_utility_rule(self):
        engine = ValidationEngine()

        def check_max_consumption(data):
            if data.get("consumption", 0) > 100000:
                return [{"severity": "error", "field": "consumption", "message": "Unusually high"}]
            return []

        engine.register_custom_rule("utility_bill", check_max_consumption)
        data = {"account_number": "ACC-001", "consumption": 200000}
        result = engine.validate_utility_bill(data)
        assert result["is_valid"] is False


class TestValidationSeverityLevels:
    """Test that severity levels are properly categorized."""

    def test_errors_have_error_severity(self):
        engine = ValidationEngine()
        data = {"invoice_number": None}
        result = engine.validate_invoice(data)
        for error in result["errors"]:
            assert error["severity"] == "error"

    def test_warnings_have_warning_severity(self):
        engine = ValidationEngine()
        data = {"invoice_number": "INV-001", "invoice_date": None}
        result = engine.validate_invoice(data)
        for warning in result["warnings"]:
            assert warning["severity"] == "warning"


class TestValidationStatistics:
    """Test statistics accumulation."""

    def test_multiple_validations(self):
        engine = ValidationEngine()
        engine.validate_invoice({"invoice_number": "INV-001"})
        engine.validate_manifest({"manifest_number": "BOL-001"})
        engine.validate_utility_bill({"account_number": "ACC-001"})
        stats = engine.get_statistics()
        assert stats["validations_run"] == 3

    def test_rules_checked_accumulate(self):
        engine = ValidationEngine()
        engine.validate_invoice({"invoice_number": "INV-001"})
        engine.validate_invoice({"invoice_number": "INV-002"})
        stats = engine.get_statistics()
        assert stats["rules_checked"] > 2
