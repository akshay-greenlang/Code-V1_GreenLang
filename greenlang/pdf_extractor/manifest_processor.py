# -*- coding: utf-8 -*-
"""
Manifest Processor - AGENT-DATA-001: PDF & Invoice Extractor

Shipping manifest and Bill-of-Lading (BOL) extraction engine that wraps
FieldExtractor with logistics domain logic: shipping party extraction,
cargo detail parsing, transport information, and weight/piece-count
cross-validation.

Features:
    - Shipping party extraction (shipper, consignee, carrier)
    - Multi-format cargo table parsing (description, weight, pieces)
    - Transport info (vehicle, seal numbers, container IDs)
    - Cross-validation of weight totals and piece counts
    - SHA-256 provenance hashing for audit trails
    - Weighted confidence scoring

Zero-Hallucination Guarantees:
    - All weight validations are deterministic arithmetic
    - Piece counts are summed, never estimated
    - Carrier/shipper names are pattern-matched, never inferred

Example:
    >>> from greenlang.pdf_extractor.manifest_processor import ManifestProcessor
    >>> from greenlang.pdf_extractor.field_extractor import FieldExtractor
    >>> extractor = FieldExtractor()
    >>> processor = ManifestProcessor(field_extractor=extractor)
    >>> result = processor.process_manifest(manifest_text)
    >>> print(result.manifest_number, result.total_weight)

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
    "ManifestValidationResult",
    "ManifestExtraction",
    "ManifestProcessor",
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


class ManifestValidationResult(BaseModel):
    """Result of a single manifest validation check."""

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


class ManifestExtraction(BaseModel):
    """Complete manifest / BOL extraction result."""

    manifest_number: Optional[str] = Field(None)
    shipment_date: Optional[str] = Field(None)
    carrier_name: Optional[str] = Field(None)
    origin: Optional[str] = Field(None)
    destination: Optional[str] = Field(None)
    shipper_name: Optional[str] = Field(None)
    consignee_name: Optional[str] = Field(None)
    total_weight: Optional[float] = Field(None)
    weight_unit: Optional[str] = Field(None)
    total_pieces: Optional[int] = Field(None)
    vehicle_id: Optional[str] = Field(None)
    seal_numbers: Optional[List[str]] = Field(default_factory=list)
    shipping_parties: Dict[str, Any] = Field(default_factory=dict)
    cargo_items: List[LineItem] = Field(default_factory=list)
    transport_info: Dict[str, Any] = Field(default_factory=dict)
    validations: List[ManifestValidationResult] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_hash: str = Field(default="")
    extracted_at: datetime = Field(default_factory=_utcnow)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Cargo line-item patterns
# ---------------------------------------------------------------------------

_CARGO_ITEM_PATTERN = re.compile(
    r"^(.+?)\s+"                             # description
    r"([\d,]+\.?\d*)\s+"                     # weight
    r"(kg|lbs?|tonnes?|tons?|mt)?\s*"        # optional unit
    r"(\d+)\s*$",                            # pieces
    re.MULTILINE | re.IGNORECASE,
)

_CARGO_SIMPLE_PATTERN = re.compile(
    r"^(.{3,50}?)\s+([\d,]+\.?\d*)\s*(kg|lbs?|tonnes?|tons?|mt)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ManifestProcessor
# ---------------------------------------------------------------------------


class ManifestProcessor:
    """Shipping manifest and BOL extraction and validation engine.

    Uses FieldExtractor for pattern matching and adds logistics domain
    logic including cargo table parsing, transport information
    extraction, and weight/piece-count cross-validation.

    Attributes:
        _field_extractor: FieldExtractor instance.
        _config: Configuration dictionary.
        _lock: Threading lock for statistics.
        _stats: Processing statistics.

    Example:
        >>> processor = ManifestProcessor()
        >>> result = processor.process_manifest(text)
        >>> print(result.carrier_name, result.total_weight)
    """

    # Confidence weights
    _CONFIDENCE_WEIGHTS: Dict[str, float] = {
        "manifest_number": 2.0,
        "shipment_date": 1.5,
        "carrier_name": 1.5,
        "total_weight": 2.0,
        "origin": 1.0,
        "destination": 1.0,
        "shipper_name": 1.0,
        "consignee_name": 1.0,
        "vehicle_id": 0.5,
        "seal_numbers": 0.5,
        "total_pieces": 1.0,
        "weight_unit": 0.5,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        field_extractor: Optional[FieldExtractor] = None,
    ) -> None:
        """Initialise ManifestProcessor.

        Args:
            config: Optional configuration dict.
            field_extractor: Optional FieldExtractor instance.
        """
        self._config = config or {}
        self._field_extractor = field_extractor or FieldExtractor(config)
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "manifests_processed": 0,
            "cargo_items_extracted": 0,
            "validations_run": 0,
            "validation_failures": 0,
            "errors": 0,
        }
        logger.info("ManifestProcessor initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_manifest(
        self,
        text: str,
        confidence_threshold: float = 0.7,
        template: Optional[Dict[str, List[str]]] = None,
    ) -> ManifestExtraction:
        """Perform full manifest extraction and validation.

        Args:
            text: Extracted document text.
            confidence_threshold: Minimum field confidence.
            template: Optional custom pattern template.

        Returns:
            ManifestExtraction with all fields and validations.
        """
        start = time.monotonic()

        # Step 1: Extract fields
        fields, meta = self._field_extractor.extract_fields(
            text, "manifest", confidence_threshold, template,
        )
        field_map = {f.field_name: f for f in fields}

        # Step 2: Shipping parties
        parties = self.extract_shipping_parties(text, field_map)

        # Step 3: Cargo details
        cargo_items = self.extract_cargo_details(text)

        # Step 4: Transport info
        transport = self.extract_transport_info(text, field_map)

        # Step 5: Parse seal numbers
        seal_raw = self._get_value(field_map, "seal_numbers")
        seal_numbers: List[str] = []
        if seal_raw and isinstance(seal_raw, str):
            seal_numbers = [
                s.strip() for s in re.split(r"[,;]", seal_raw)
                if s.strip()
            ]

        # Step 6: Build result
        total_weight_val = self._get_float(field_map, "total_weight")
        total_pieces_raw = self._get_value(field_map, "total_pieces")
        total_pieces_val: Optional[int] = None
        if total_pieces_raw is not None:
            try:
                total_pieces_val = int(float(str(total_pieces_raw)))
            except (ValueError, TypeError):
                pass

        manifest = ManifestExtraction(
            manifest_number=self._get_str(field_map, "manifest_number"),
            shipment_date=self._get_str(field_map, "shipment_date"),
            carrier_name=parties.get("carrier_name"),
            origin=parties.get("origin"),
            destination=parties.get("destination"),
            shipper_name=parties.get("shipper_name"),
            consignee_name=parties.get("consignee_name"),
            total_weight=total_weight_val,
            weight_unit=self._get_str(field_map, "weight_unit"),
            total_pieces=total_pieces_val,
            vehicle_id=transport.get("vehicle_id"),
            seal_numbers=seal_numbers,
            shipping_parties=parties,
            cargo_items=cargo_items,
            transport_info=transport,
        )

        # Step 7: Validate
        validations = self.validate_manifest(manifest)
        manifest.validations = validations

        # Step 8: Confidence
        field_confidences = {f.field_name: f.confidence for f in fields}
        manifest.overall_confidence = self._calculate_confidence(
            field_confidences,
        )

        # Step 9: Provenance
        manifest.provenance_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        elapsed_ms = (time.monotonic() - start) * 1000
        with self._lock:
            self._stats["manifests_processed"] += 1
            self._stats["cargo_items_extracted"] += len(cargo_items)
            self._stats["validations_run"] += len(validations)
            self._stats["validation_failures"] += sum(
                1 for v in validations if not v.passed
            )

        logger.info(
            "Manifest processed: number=%s, cargo=%d, validations=%d (%.1f ms)",
            manifest.manifest_number,
            len(cargo_items),
            len(validations),
            elapsed_ms,
        )
        return manifest

    def extract_shipping_parties(
        self,
        text: str,
        field_map: Optional[Dict[str, ExtractedField]] = None,
    ) -> Dict[str, Any]:
        """Extract shipping party information.

        Args:
            text: Document text.
            field_map: Optional pre-extracted field map.

        Returns:
            Dictionary with shipper, consignee, carrier, origin, destination.
        """
        if field_map is None:
            fields, _ = self._field_extractor.extract_fields(text, "manifest")
            field_map = {f.field_name: f for f in fields}

        return {
            "shipper_name": self._get_str(field_map, "shipper_name"),
            "consignee_name": self._get_str(field_map, "consignee_name"),
            "carrier_name": self._get_str(field_map, "carrier_name"),
            "origin": self._get_str(field_map, "origin"),
            "destination": self._get_str(field_map, "destination"),
        }

    def extract_cargo_details(self, text: str) -> List[LineItem]:
        """Extract cargo items with weights and piece counts.

        Args:
            text: Document text.

        Returns:
            List of LineItem objects representing cargo lines.
        """
        items: List[LineItem] = []

        # Primary pattern: description + weight + unit + pieces
        for idx, m in enumerate(_CARGO_ITEM_PATTERN.finditer(text)):
            items.append(LineItem(
                line_number=idx + 1,
                description=m.group(1).strip(),
                quantity=self._safe_float(m.group(4)),  # pieces
                unit=m.group(3) if m.group(3) else None,
                amount=self._safe_float(m.group(2)),    # weight as amount
                raw_text=m.group(0).strip(),
            ))

        # Fallback: simpler description + weight + unit
        if not items:
            for idx, m in enumerate(_CARGO_SIMPLE_PATTERN.finditer(text)):
                items.append(LineItem(
                    line_number=idx + 1,
                    description=m.group(1).strip(),
                    unit=m.group(3),
                    amount=self._safe_float(m.group(2)),  # weight
                    raw_text=m.group(0).strip(),
                ))

        logger.debug("Extracted %d cargo items", len(items))
        return items

    def extract_transport_info(
        self,
        text: str,
        field_map: Optional[Dict[str, ExtractedField]] = None,
    ) -> Dict[str, Any]:
        """Extract transport-related information.

        Args:
            text: Document text.
            field_map: Optional pre-extracted field map.

        Returns:
            Dictionary with vehicle_id, seal_numbers, container info.
        """
        if field_map is None:
            fields, _ = self._field_extractor.extract_fields(text, "manifest")
            field_map = {f.field_name: f for f in fields}

        info: Dict[str, Any] = {
            "vehicle_id": self._get_str(field_map, "vehicle_id"),
            "seal_numbers": self._get_str(field_map, "seal_numbers"),
        }

        # Additional container patterns
        container_pat = re.compile(
            r"(?i)container\s*(?:#|no\.?|number|id)\s*[:\-]?\s*(\S+)",
        )
        m = container_pat.search(text)
        info["container_id"] = m.group(1) if m else None

        # Mode of transport
        mode_pat = re.compile(
            r"(?i)(?:mode|transport\s*mode)\s*[:\-]?\s*(road|rail|sea|air|ocean|truck|ship|vessel|barge)",
        )
        m = mode_pat.search(text)
        info["transport_mode"] = m.group(1).lower() if m else None

        return info

    def validate_manifest(
        self,
        manifest_data: ManifestExtraction,
    ) -> List[ManifestValidationResult]:
        """Validate manifest data for consistency.

        Args:
            manifest_data: Extracted manifest data.

        Returns:
            List of ManifestValidationResult objects.
        """
        results: List[ManifestValidationResult] = []

        # Required fields
        results.extend(self._validate_required(manifest_data))

        # Weight total
        if manifest_data.cargo_items and manifest_data.total_weight is not None:
            result = self._validate_weight_total(
                manifest_data.cargo_items,
                manifest_data.total_weight,
            )
            if result:
                results.append(result)

        # Piece count
        if manifest_data.cargo_items and manifest_data.total_pieces is not None:
            results.append(self._validate_piece_count(
                manifest_data.cargo_items,
                manifest_data.total_pieces,
            ))

        # Origin != destination
        if manifest_data.origin and manifest_data.destination:
            same = (
                manifest_data.origin.strip().lower()
                == manifest_data.destination.strip().lower()
            )
            results.append(ManifestValidationResult(
                rule_name="origin_destination_differ",
                passed=not same,
                message=(
                    "Origin and destination are different"
                    if not same
                    else "Origin and destination are identical"
                ),
                severity="warning" if same else "info",
            ))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics.

        Returns:
            Dictionary with counts and rates.
        """
        with self._lock:
            total = self._stats["manifests_processed"]
            return {
                "manifests_processed": total,
                "cargo_items_extracted": self._stats["cargo_items_extracted"],
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

    def _validate_required(
        self,
        manifest: ManifestExtraction,
    ) -> List[ManifestValidationResult]:
        """Validate required manifest fields are present."""
        results: List[ManifestValidationResult] = []
        required = {
            "manifest_number": manifest.manifest_number,
            "shipment_date": manifest.shipment_date,
            "carrier_name": manifest.carrier_name,
        }
        for name, value in required.items():
            results.append(ManifestValidationResult(
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

    def _validate_weight_total(
        self,
        items: List[LineItem],
        declared_total: float,
    ) -> Optional[ManifestValidationResult]:
        """Validate cargo item weights sum to declared total.

        Args:
            items: Cargo line items (weight stored in amount field).
            declared_total: Declared total weight.

        Returns:
            ManifestValidationResult or None if no weights to sum.
        """
        weights = [
            item.amount for item in items if item.amount is not None
        ]
        if not weights:
            return None

        computed = round(sum(weights), 2)
        # Allow 1% tolerance for rounding
        tolerance = max(declared_total * 0.01, 0.5)
        passed = abs(computed - declared_total) <= tolerance

        return ManifestValidationResult(
            rule_name="weight_total",
            passed=passed,
            message=(
                "Weight total matches cargo items"
                if passed
                else f"Weight mismatch: items sum={computed}, "
                     f"declared={declared_total}"
            ),
            severity="warning" if not passed else "info",
            expected=str(declared_total),
            actual=str(computed),
        )

    def _validate_piece_count(
        self,
        items: List[LineItem],
        declared_total: int,
    ) -> ManifestValidationResult:
        """Validate cargo item piece counts sum to declared total."""
        pieces = [
            int(item.quantity) for item in items
            if item.quantity is not None
        ]
        computed = sum(pieces)
        passed = computed == declared_total

        return ManifestValidationResult(
            rule_name="piece_count",
            passed=passed,
            message=(
                "Piece count matches cargo items"
                if passed
                else f"Piece count mismatch: items sum={computed}, "
                     f"declared={declared_total}"
            ),
            severity="warning" if not passed else "info",
            expected=str(declared_total),
            actual=str(computed),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        field_confidences: Dict[str, float],
    ) -> float:
        """Calculate weighted average confidence."""
        total_weight = 0.0
        weighted_sum = 0.0
        for field, confidence in field_confidences.items():
            weight = self._CONFIDENCE_WEIGHTS.get(field, 0.5)
            weighted_sum += confidence * weight
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    def _get_value(
        self,
        field_map: Dict[str, ExtractedField],
        field_name: str,
    ) -> Any:
        """Get parsed value from field map."""
        field = field_map.get(field_name)
        return field.value if field else None

    def _get_str(
        self,
        field_map: Dict[str, ExtractedField],
        field_name: str,
    ) -> Optional[str]:
        """Get string value from field map."""
        val = self._get_value(field_map, field_name)
        return str(val) if val is not None else None

    def _get_float(
        self,
        field_map: Dict[str, ExtractedField],
        field_name: str,
    ) -> Optional[float]:
        """Safely get float from field map."""
        val = self._get_value(field_map, field_name)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float."""
        try:
            return float(value.replace(",", ""))
        except (ValueError, AttributeError):
            return None
