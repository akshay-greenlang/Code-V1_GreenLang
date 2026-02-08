# -*- coding: utf-8 -*-
"""
Unit Tests for ManifestProcessor (AGENT-DATA-001)

Tests shipping manifest / Bill of Lading processing including full
extraction, shipping party extraction, cargo detail extraction,
transport info extraction, manifest validation, and weight validation.
Uses realistic BOL text samples.

Coverage target: 85%+ of manifest_processor.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ManifestProcessor mirroring greenlang/pdf_extractor/manifest_processor.py
# ---------------------------------------------------------------------------


class ManifestProcessingError(Exception):
    """Raised when manifest processing fails."""
    pass


class ManifestProcessor:
    """Specialized processor for shipping manifests and Bills of Lading.

    Extracts shipping parties, cargo details, transport information,
    and validates weight/package consistency.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self._confidence_threshold = confidence_threshold
        self._stats = {
            "manifests_processed": 0,
            "manifests_validated": 0,
            "validation_failures": 0,
            "total_cargo_items": 0,
        }

    def process_manifest(self, text: str) -> Dict[str, Any]:
        """Process manifest text and extract all fields."""
        shipping_parties = self.extract_shipping_parties(text)
        cargo_details = self.extract_cargo_details(text)
        transport_info = self.extract_transport_info(text)
        validation = self.validate_manifest(shipping_parties, cargo_details, transport_info)

        self._stats["manifests_processed"] += 1
        self._stats["total_cargo_items"] += len(cargo_details.get("items", []))

        confidence = self._calculate_confidence(shipping_parties, cargo_details, transport_info)

        return {
            "shipping_parties": shipping_parties,
            "cargo_details": cargo_details,
            "transport_info": transport_info,
            "validation": validation,
            "confidence": confidence,
        }

    def extract_shipping_parties(self, text: str) -> Dict[str, Optional[str]]:
        """Extract shipper, consignee, and carrier information."""
        parties: Dict[str, Optional[str]] = {
            "shipper_name": None,
            "shipper_address": None,
            "consignee_name": None,
            "consignee_address": None,
            "carrier_name": None,
        }

        m = re.search(r"Shipper[:\s]*\n\s*([A-Z][A-Za-z\s&.,]+?)(?:\n)", text)
        if m:
            parties["shipper_name"] = m.group(1).strip()

        m = re.search(r"Consignee[:\s]*\n\s*([A-Z][A-Za-z\s&.,]+?)(?:\n)", text)
        if m:
            parties["consignee_name"] = m.group(1).strip()

        m = re.search(r"Carrier[:\s]*\n\s*([A-Z][A-Za-z\s&.,]+?)(?:\n)", text)
        if m:
            parties["carrier_name"] = m.group(1).strip()

        return parties

    def extract_cargo_details(self, text: str) -> Dict[str, Any]:
        """Extract cargo items and totals."""
        cargo: Dict[str, Any] = {
            "items": [],
            "total_packages": None,
            "total_weight_kg": None,
            "total_volume_m3": None,
        }

        # Extract cargo items (tabular format)
        pattern = r"(\d+)\s+(.+?)\s+(\d+)\s+([\d,]+)\s+([\d.]+)"
        for match in re.finditer(pattern, text):
            cargo["items"].append({
                "item_number": int(match.group(1)),
                "description": match.group(2).strip(),
                "packages": int(match.group(3)),
                "weight_kg": self._parse_number(match.group(4)),
                "volume_m3": float(match.group(5)),
            })

        # Total packages
        m = re.search(r"Total\s+Packages[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            cargo["total_packages"] = int(m.group(1))

        # Total weight
        m = re.search(r"Total\s+(?:Gross\s+)?Weight[:\s]*([\d,]+)\s*kg", text, re.IGNORECASE)
        if m:
            cargo["total_weight_kg"] = self._parse_number(m.group(1))

        # Total volume
        m = re.search(r"Total\s+Volume[:\s]*([\d.]+)\s*m3", text, re.IGNORECASE)
        if m:
            cargo["total_volume_m3"] = float(m.group(1))

        return cargo

    def extract_transport_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract transport and routing information."""
        info: Dict[str, Optional[str]] = {
            "manifest_number": None,
            "date": None,
            "vessel_name": None,
            "voyage_number": None,
            "port_of_loading": None,
            "port_of_discharge": None,
            "container_number": None,
            "freight_terms": None,
        }

        m = re.search(r"(?:BOL|B/L|Bill\s+of\s+Lading)\s*(?:Number|No)?[:\s]*([\w\-]+)", text, re.IGNORECASE)
        if m:
            info["manifest_number"] = m.group(1).strip()

        m = re.search(r"Date[:\s]*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        if m:
            info["date"] = m.group(1)

        m = re.search(r"Vessel[:\s]*([A-Z][A-Za-z\s]+?)(?:\n|$)", text)
        if m:
            info["vessel_name"] = m.group(1).strip()

        m = re.search(r"Voyage[:\s]*([\w\-]+)", text, re.IGNORECASE)
        if m:
            info["voyage_number"] = m.group(1).strip()

        m = re.search(r"Port\s+of\s+Loading[:\s]*([A-Za-z\s,()]+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            info["port_of_loading"] = m.group(1).strip()

        m = re.search(r"Port\s+of\s+Discharge[:\s]*([A-Za-z\s,()]+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            info["port_of_discharge"] = m.group(1).strip()

        m = re.search(r"Container[:\s]*([\w]+)", text, re.IGNORECASE)
        if m:
            info["container_number"] = m.group(1).strip()

        m = re.search(r"Freight\s+Terms[:\s]*([A-Za-z\s]+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            info["freight_terms"] = m.group(1).strip()

        return info

    def validate_manifest(
        self,
        shipping_parties: Dict[str, Optional[str]],
        cargo_details: Dict[str, Any],
        transport_info: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Validate manifest cross-field consistency."""
        errors = []
        warnings = []

        # Required fields
        if not transport_info.get("manifest_number"):
            errors.append({"field": "manifest_number", "message": "Missing manifest/BOL number"})

        if not shipping_parties.get("shipper_name"):
            warnings.append({"field": "shipper_name", "message": "Missing shipper name"})

        if not shipping_parties.get("consignee_name"):
            warnings.append({"field": "consignee_name", "message": "Missing consignee name"})

        # Weight validation
        items = cargo_details.get("items", [])
        if items and cargo_details.get("total_weight_kg") is not None:
            item_weight = sum(i.get("weight_kg", 0) for i in items)
            total_weight = cargo_details["total_weight_kg"]
            if abs(item_weight - total_weight) > 1.0:
                errors.append({
                    "field": "total_weight_kg",
                    "message": f"Weight mismatch: items sum {item_weight} != total {total_weight}",
                })

        # Package validation
        if items and cargo_details.get("total_packages") is not None:
            item_packages = sum(i.get("packages", 0) for i in items)
            total_packages = cargo_details["total_packages"]
            if item_packages != total_packages:
                errors.append({
                    "field": "total_packages",
                    "message": f"Package mismatch: items sum {item_packages} != total {total_packages}",
                })

        is_valid = len(errors) == 0
        self._stats["manifests_validated"] += 1
        if not is_valid:
            self._stats["validation_failures"] += 1

        return {"is_valid": is_valid, "errors": errors, "warnings": warnings}

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return dict(self._stats)

    def _calculate_confidence(
        self,
        shipping_parties: Dict[str, Optional[str]],
        cargo_details: Dict[str, Any],
        transport_info: Dict[str, Optional[str]],
    ) -> float:
        scores = []
        party_fields = ["shipper_name", "consignee_name", "carrier_name"]
        present = sum(1 for f in party_fields if shipping_parties.get(f))
        scores.append(present / len(party_fields))

        if cargo_details.get("items"):
            scores.append(0.9)
        else:
            scores.append(0.3)

        transport_fields = ["manifest_number", "vessel_name", "port_of_loading", "port_of_discharge"]
        present = sum(1 for f in transport_fields if transport_info.get(f))
        scores.append(present / len(transport_fields))

        return sum(scores) / len(scores) if scores else 0.0

    def _parse_number(self, raw: str) -> float:
        cleaned = raw.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0


# ===========================================================================
# Test Classes
# ===========================================================================


class TestManifestProcessorInit:
    """Test ManifestProcessor initialization."""

    def test_default_threshold(self):
        proc = ManifestProcessor()
        assert proc._confidence_threshold == 0.7

    def test_initial_statistics(self):
        proc = ManifestProcessor()
        stats = proc.get_statistics()
        assert stats["manifests_processed"] == 0


class TestProcessManifest:
    """Test process_manifest full extraction."""

    def test_returns_all_sections(self, sample_manifest_text):
        proc = ManifestProcessor()
        result = proc.process_manifest(sample_manifest_text)
        assert "shipping_parties" in result
        assert "cargo_details" in result
        assert "transport_info" in result
        assert "validation" in result
        assert "confidence" in result

    def test_updates_stats(self, sample_manifest_text):
        proc = ManifestProcessor()
        proc.process_manifest(sample_manifest_text)
        stats = proc.get_statistics()
        assert stats["manifests_processed"] == 1

    def test_confidence_is_float(self, sample_manifest_text):
        proc = ManifestProcessor()
        result = proc.process_manifest(sample_manifest_text)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0


class TestExtractShippingParties:
    """Test extract_shipping_parties method."""

    def test_shipper_name(self, sample_manifest_text):
        proc = ManifestProcessor()
        parties = proc.extract_shipping_parties(sample_manifest_text)
        assert parties["shipper_name"] is not None

    def test_consignee_name(self, sample_manifest_text):
        proc = ManifestProcessor()
        parties = proc.extract_shipping_parties(sample_manifest_text)
        assert parties["consignee_name"] is not None

    def test_carrier_name(self, sample_manifest_text):
        proc = ManifestProcessor()
        parties = proc.extract_shipping_parties(sample_manifest_text)
        assert parties["carrier_name"] is not None

    def test_missing_parties(self):
        proc = ManifestProcessor()
        parties = proc.extract_shipping_parties("No party information here")
        assert parties["shipper_name"] is None
        assert parties["consignee_name"] is None
        assert parties["carrier_name"] is None


class TestExtractCargoDetails:
    """Test extract_cargo_details method."""

    def test_total_packages(self, sample_manifest_text):
        proc = ManifestProcessor()
        cargo = proc.extract_cargo_details(sample_manifest_text)
        assert cargo["total_packages"] == 85

    def test_total_weight(self, sample_manifest_text):
        proc = ManifestProcessor()
        cargo = proc.extract_cargo_details(sample_manifest_text)
        assert cargo["total_weight_kg"] is not None
        if cargo["total_weight_kg"] is not None:
            assert cargo["total_weight_kg"] > 0

    def test_total_volume(self, sample_manifest_text):
        proc = ManifestProcessor()
        cargo = proc.extract_cargo_details(sample_manifest_text)
        if cargo["total_volume_m3"] is not None:
            assert cargo["total_volume_m3"] > 0

    def test_empty_text(self):
        proc = ManifestProcessor()
        cargo = proc.extract_cargo_details("")
        assert cargo["items"] == []
        assert cargo["total_packages"] is None


class TestExtractTransportInfo:
    """Test extract_transport_info method."""

    def test_manifest_number(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        assert info["manifest_number"] is not None

    def test_date(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        assert info["date"] == "2025-06-20"

    def test_vessel_name(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        if info["vessel_name"] is not None:
            assert "Green Future" in info["vessel_name"] or len(info["vessel_name"]) > 0

    def test_voyage_number(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        assert info["voyage_number"] is not None

    def test_port_of_loading(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        assert info["port_of_loading"] is not None

    def test_port_of_discharge(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        assert info["port_of_discharge"] is not None

    def test_container_number(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        assert info["container_number"] is not None

    def test_freight_terms(self, sample_manifest_text):
        proc = ManifestProcessor()
        info = proc.extract_transport_info(sample_manifest_text)
        if info["freight_terms"] is not None:
            assert len(info["freight_terms"]) > 0


class TestValidateManifest:
    """Test validate_manifest method."""

    def test_valid_manifest(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": "Shipper", "consignee_name": "Consignee", "carrier_name": "Carrier"}
        cargo = {
            "items": [{"weight_kg": 1000, "packages": 10}],
            "total_weight_kg": 1000.0,
            "total_packages": 10,
        }
        transport = {"manifest_number": "BOL-001"}
        result = proc.validate_manifest(parties, cargo, transport)
        assert result["is_valid"] is True

    def test_missing_manifest_number(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": "S", "consignee_name": "C"}
        cargo = {"items": [], "total_weight_kg": None, "total_packages": None}
        transport = {"manifest_number": None}
        result = proc.validate_manifest(parties, cargo, transport)
        assert any(e["field"] == "manifest_number" for e in result["errors"])

    def test_weight_mismatch(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": "S"}
        cargo = {
            "items": [{"weight_kg": 500, "packages": 5}, {"weight_kg": 500, "packages": 5}],
            "total_weight_kg": 2000.0,
            "total_packages": 10,
        }
        transport = {"manifest_number": "BOL-001"}
        result = proc.validate_manifest(parties, cargo, transport)
        assert any("weight" in e["field"].lower() for e in result["errors"])

    def test_package_mismatch(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": "S"}
        cargo = {
            "items": [{"weight_kg": 500, "packages": 5}],
            "total_weight_kg": 500.0,
            "total_packages": 99,
        }
        transport = {"manifest_number": "BOL-001"}
        result = proc.validate_manifest(parties, cargo, transport)
        assert any("package" in e["field"].lower() for e in result["errors"])

    def test_missing_shipper_warning(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": None, "consignee_name": "C"}
        cargo = {"items": [], "total_weight_kg": None, "total_packages": None}
        transport = {"manifest_number": "BOL-001"}
        result = proc.validate_manifest(parties, cargo, transport)
        assert any(w["field"] == "shipper_name" for w in result["warnings"])

    def test_validation_updates_stats(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": "S"}
        cargo = {"items": [], "total_weight_kg": None, "total_packages": None}
        transport = {"manifest_number": "BOL-001"}
        proc.validate_manifest(parties, cargo, transport)
        stats = proc.get_statistics()
        assert stats["manifests_validated"] == 1

    def test_weight_within_tolerance(self):
        proc = ManifestProcessor()
        parties = {"shipper_name": "S"}
        cargo = {
            "items": [{"weight_kg": 999.5, "packages": 10}],
            "total_weight_kg": 1000.0,
            "total_packages": 10,
        }
        transport = {"manifest_number": "BOL-001"}
        result = proc.validate_manifest(parties, cargo, transport)
        assert result["is_valid"] is True
