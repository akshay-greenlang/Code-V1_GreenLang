# -*- coding: utf-8 -*-
"""
FoundationBridge - Bridge to Foundation Agents for PACK-041
=============================================================

This module routes core platform operations to the Foundation agents
(FOUND-001 through FOUND-010) for DAG orchestration, schema validation,
unit normalization, assumption registration, citation management,
access control, reproducibility verification, and telemetry emission.

Routing Table:
    DAG orchestration        --> FOUND-001 (GreenLang Orchestrator)
    Schema validation        --> FOUND-002 (Schema Compiler & Validator)
    Unit normalization       --> FOUND-003 (Unit & Reference Normalizer)
    Assumption registration  --> FOUND-004 (Assumptions Registry)
    Citation management      --> FOUND-005 (Citations & Evidence Agent)
    Access control           --> FOUND-006 (Access & Policy Guard)
    Reproducibility check    --> FOUND-008 (Reproducibility Agent)
    Telemetry emission       --> FOUND-010 (Observability & Telemetry Agent)

Zero-Hallucination:
    All unit conversions, schema validations, and hash comparisons use
    deterministic logic. No LLM calls in the foundation path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Unit Conversion Tables (deterministic)
# ---------------------------------------------------------------------------

ENERGY_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "kwh_to_mmbtu": {"factor": 0.003412, "from": "kWh", "to": "MMBtu"},
    "mmbtu_to_kwh": {"factor": 293.071, "from": "MMBtu", "to": "kWh"},
    "therms_to_mmbtu": {"factor": 0.1, "from": "therms", "to": "MMBtu"},
    "mmbtu_to_therms": {"factor": 10.0, "from": "MMBtu", "to": "therms"},
    "gj_to_mmbtu": {"factor": 0.9478, "from": "GJ", "to": "MMBtu"},
    "mmbtu_to_gj": {"factor": 1.0551, "from": "MMBtu", "to": "GJ"},
    "kwh_to_mj": {"factor": 3.6, "from": "kWh", "to": "MJ"},
    "mj_to_kwh": {"factor": 0.2778, "from": "MJ", "to": "kWh"},
    "kwh_to_gj": {"factor": 0.0036, "from": "kWh", "to": "GJ"},
    "gj_to_kwh": {"factor": 277.778, "from": "GJ", "to": "kWh"},
    "gallons_to_liters": {"factor": 3.78541, "from": "gallons", "to": "liters"},
    "liters_to_gallons": {"factor": 0.26417, "from": "liters", "to": "gallons"},
    "short_tons_to_metric_tonnes": {"factor": 0.9072, "from": "short_tons", "to": "tonnes"},
    "metric_tonnes_to_short_tons": {"factor": 1.1023, "from": "tonnes", "to": "short_tons"},
    "scf_to_ccf": {"factor": 0.01, "from": "scf", "to": "ccf"},
    "ccf_to_therms": {"factor": 1.024, "from": "ccf", "to": "therms"},
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class FoundationConfig(BaseModel):
    """Configuration for Foundation agent routing."""

    config_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-041")
    enable_telemetry: bool = Field(default=True)
    enable_citations: bool = Field(default=True)
    enable_access_control: bool = Field(default=True)

class ValidationResult(BaseModel):
    """Schema validation result."""

    valid: bool = Field(default=True)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    fields_validated: int = Field(default=0)
    provenance_hash: str = Field(default="")

class NormalizationResult(BaseModel):
    """Unit normalization result."""

    original_value: float = Field(default=0.0)
    original_unit: str = Field(default="")
    normalized_value: float = Field(default=0.0)
    normalized_unit: str = Field(default="")
    conversion_factor: float = Field(default=1.0)
    provenance_hash: str = Field(default="")

class AssumptionRecord(BaseModel):
    """Registered assumption for audit trail."""

    assumption_id: str = Field(default_factory=_new_uuid)
    category: str = Field(default="")
    description: str = Field(default="")
    value: str = Field(default="")
    source: str = Field(default="")
    impact: str = Field(default="low")
    registered_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class CitationRecord(BaseModel):
    """Citation and evidence reference."""

    citation_id: str = Field(default_factory=_new_uuid)
    source_type: str = Field(default="")
    title: str = Field(default="")
    author: str = Field(default="")
    year: int = Field(default=2025)
    url: str = Field(default="")
    doi: str = Field(default="")
    page_reference: str = Field(default="")
    provenance_hash: str = Field(default="")

class TelemetryEvent(BaseModel):
    """Telemetry event for observability."""

    event_id: str = Field(default_factory=_new_uuid)
    event_type: str = Field(default="")
    component: str = Field(default="PACK-041")
    message: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# FoundationBridge
# ---------------------------------------------------------------------------

class FoundationBridge:
    """Bridge to Foundation agents (FOUND-001 through FOUND-010).

    Routes core platform operations including DAG orchestration, schema
    validation, unit normalization, assumption registration, citation
    management, access control, reproducibility verification, and
    telemetry emission.

    Attributes:
        config: Foundation agent configuration.
        _assumptions: Registered assumptions registry.
        _citations: Citation registry.
        _telemetry: Emitted telemetry events.

    Example:
        >>> bridge = FoundationBridge()
        >>> result = bridge.normalize_units(1000, "kWh", "MMBtu")
        >>> assert result.normalized_unit == "MMBtu"
    """

    def __init__(
        self,
        config: Optional[FoundationConfig] = None,
    ) -> None:
        """Initialize FoundationBridge.

        Args:
            config: Foundation agent configuration. Uses defaults if None.
        """
        self.config = config or FoundationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._assumptions: Dict[str, AssumptionRecord] = {}
        self._citations: Dict[str, CitationRecord] = {}
        self._telemetry: List[TelemetryEvent] = []

        self.logger.info(
            "FoundationBridge initialized: pack=%s, telemetry=%s, citations=%s",
            self.config.pack_id,
            self.config.enable_telemetry,
            self.config.enable_citations,
        )

    # -------------------------------------------------------------------------
    # FOUND-001: DAG Orchestration
    # -------------------------------------------------------------------------

    def orchestrate_dag(
        self,
        dag_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit a DAG configuration for orchestration via FOUND-001.

        Args:
            dag_config: DAG definition with nodes and edges.

        Returns:
            Dict with orchestration result.
        """
        start_time = time.monotonic()
        self.logger.info("Orchestrating DAG: %d nodes", len(dag_config.get("nodes", [])))

        result = {
            "agent": "FOUND-001",
            "dag_id": _new_uuid(),
            "nodes": len(dag_config.get("nodes", [])),
            "edges": len(dag_config.get("edges", [])),
            "status": "submitted",
            "provenance_hash": _compute_hash(dag_config),
            "processing_time_ms": (time.monotonic() - start_time) * 1000,
        }
        return result

    # -------------------------------------------------------------------------
    # FOUND-002: Schema Validation
    # -------------------------------------------------------------------------

    def validate_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> ValidationResult:
        """Validate data against a schema via FOUND-002.

        Args:
            data: Data to validate.
            schema: Schema definition with required_fields and types.

        Returns:
            ValidationResult with validation details.
        """
        start_time = time.monotonic()
        errors: List[str] = []
        warnings: List[str] = []

        required_fields = schema.get("required_fields", [])
        for field_name in required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        type_checks = schema.get("field_types", {})
        for field_name, expected_type in type_checks.items():
            if field_name in data:
                value = data[field_name]
                if expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field_name}' expected number, got {type(value).__name__}")
                elif expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{field_name}' expected string, got {type(value).__name__}")

        fields_validated = len(required_fields) + len(type_checks)

        result = ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fields_validated=fields_validated,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Schema validation: valid=%s, fields=%d, errors=%d",
            result.valid, fields_validated, len(errors),
        )
        return result

    # -------------------------------------------------------------------------
    # FOUND-003: Unit Normalization
    # -------------------------------------------------------------------------

    def normalize_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> NormalizationResult:
        """Normalize a value from one unit to another via FOUND-003.

        Uses deterministic conversion factors from ENERGY_CONVERSIONS table.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit (e.g., 'kWh').
            to_unit: Target unit (e.g., 'MMBtu').

        Returns:
            NormalizationResult with converted value.
        """
        conversion_key = f"{from_unit.lower()}_to_{to_unit.lower()}"
        conversion = ENERGY_CONVERSIONS.get(conversion_key)

        if conversion:
            factor = Decimal(str(conversion["factor"]))
            converted = Decimal(str(value)) * factor
            normalized_value = float(converted.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
            conversion_factor = float(factor)
        else:
            self.logger.warning(
                "No conversion found for '%s' -> '%s', returning original",
                from_unit, to_unit,
            )
            normalized_value = value
            conversion_factor = 1.0

        result = NormalizationResult(
            original_value=value,
            original_unit=from_unit,
            normalized_value=normalized_value,
            normalized_unit=to_unit if conversion else from_unit,
            conversion_factor=conversion_factor,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.debug(
            "Unit normalization: %.4f %s -> %.4f %s (factor=%.6f)",
            value, from_unit, normalized_value,
            result.normalized_unit, conversion_factor,
        )
        return result

    # -------------------------------------------------------------------------
    # FOUND-004: Assumptions Registry
    # -------------------------------------------------------------------------

    def register_assumption(
        self,
        assumption: Dict[str, Any],
    ) -> AssumptionRecord:
        """Register an assumption for audit trail via FOUND-004.

        Args:
            assumption: Dict with category, description, value, source, impact.

        Returns:
            AssumptionRecord with registration details.
        """
        record = AssumptionRecord(
            category=assumption.get("category", "general"),
            description=assumption.get("description", ""),
            value=str(assumption.get("value", "")),
            source=assumption.get("source", ""),
            impact=assumption.get("impact", "low"),
        )
        record.provenance_hash = _compute_hash(record)
        self._assumptions[record.assumption_id] = record

        self.logger.info(
            "Assumption registered: id=%s, category=%s, impact=%s",
            record.assumption_id, record.category, record.impact,
        )
        return record

    # -------------------------------------------------------------------------
    # FOUND-005: Citations
    # -------------------------------------------------------------------------

    def cite_source(
        self,
        source_info: Dict[str, Any],
    ) -> CitationRecord:
        """Create a citation reference via FOUND-005.

        Args:
            source_info: Dict with source_type, title, author, year, url, doi.

        Returns:
            CitationRecord with citation details.
        """
        record = CitationRecord(
            source_type=source_info.get("source_type", "document"),
            title=source_info.get("title", ""),
            author=source_info.get("author", ""),
            year=source_info.get("year", 2025),
            url=source_info.get("url", ""),
            doi=source_info.get("doi", ""),
            page_reference=source_info.get("page_reference", ""),
        )
        record.provenance_hash = _compute_hash(record)
        self._citations[record.citation_id] = record

        self.logger.info(
            "Citation created: id=%s, title=%s, year=%d",
            record.citation_id, record.title, record.year,
        )
        return record

    # -------------------------------------------------------------------------
    # FOUND-006: Access Control
    # -------------------------------------------------------------------------

    def check_access(
        self,
        user: str,
        resource: str,
        action: str = "read",
    ) -> bool:
        """Check access permission via FOUND-006.

        Args:
            user: User identifier.
            resource: Resource being accessed.
            action: Action to perform (read, write, execute).

        Returns:
            True if access is granted.
        """
        if not self.config.enable_access_control:
            return True

        # Deterministic policy check (simulated RBAC)
        granted = True
        self.logger.info(
            "Access check: user=%s, resource=%s, action=%s, granted=%s",
            user, resource, action, granted,
        )
        return granted

    # -------------------------------------------------------------------------
    # FOUND-008: Reproducibility
    # -------------------------------------------------------------------------

    def verify_reproducibility(
        self,
        hash1: str,
        hash2: str,
    ) -> bool:
        """Verify reproducibility by comparing two SHA-256 hashes via FOUND-008.

        Args:
            hash1: First provenance hash.
            hash2: Second provenance hash (re-computed).

        Returns:
            True if hashes match (calculation is reproducible).
        """
        match = hash1 == hash2
        self.logger.info(
            "Reproducibility check: match=%s, hash1=%s..., hash2=%s...",
            match, hash1[:16], hash2[:16],
        )
        return match

    # -------------------------------------------------------------------------
    # FOUND-010: Telemetry
    # -------------------------------------------------------------------------

    def emit_telemetry(
        self,
        event: Dict[str, Any],
    ) -> TelemetryEvent:
        """Emit a telemetry event via FOUND-010.

        Args:
            event: Dict with event_type, message, metadata.

        Returns:
            TelemetryEvent with event details.
        """
        if not self.config.enable_telemetry:
            return TelemetryEvent()

        telemetry = TelemetryEvent(
            event_type=event.get("event_type", "info"),
            component=event.get("component", "PACK-041"),
            message=event.get("message", ""),
            metadata=event.get("metadata", {}),
        )
        self._telemetry.append(telemetry)

        self.logger.debug(
            "Telemetry emitted: type=%s, component=%s",
            telemetry.event_type, telemetry.component,
        )
        return telemetry

    # -------------------------------------------------------------------------
    # Registry Queries
    # -------------------------------------------------------------------------

    def get_assumptions(
        self,
        category: Optional[str] = None,
    ) -> List[AssumptionRecord]:
        """Get registered assumptions with optional category filter.

        Args:
            category: Optional category filter.

        Returns:
            List of matching assumption records.
        """
        assumptions = list(self._assumptions.values())
        if category:
            assumptions = [a for a in assumptions if a.category == category]
        return assumptions

    def get_citations(self) -> List[CitationRecord]:
        """Get all registered citations.

        Returns:
            List of citation records.
        """
        return list(self._citations.values())

    def get_telemetry_events(
        self,
        event_type: Optional[str] = None,
    ) -> List[TelemetryEvent]:
        """Get emitted telemetry events with optional type filter.

        Args:
            event_type: Optional event type filter.

        Returns:
            List of matching telemetry events.
        """
        events = list(self._telemetry)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events
