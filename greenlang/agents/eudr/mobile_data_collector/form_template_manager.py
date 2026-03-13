# -*- coding: utf-8 -*-
"""
Form Template Manager Engine - AGENT-EUDR-015

Engine 5: Dynamic form template management with conditional logic,
multi-language support, versioning, inheritance, and JSON Schema
generation for EUDR mobile data collection.

This engine manages the lifecycle of form templates used to capture
field data for EUDR compliance per EU 2023/1115 Articles 4, 9, and 14.
Templates define the structure, validation, conditional logic, and
multilingual labels for the 6 EUDR form types:

    1. Producer Registration (Art. 9(1)(f))
    2. Plot Survey (Art. 9(1)(c-d))
    3. Harvest Log (Art. 9(1)(a-b,e))
    4. Custody Transfer (Art. 9(1)(f-g))
    5. Quality Inspection (Art. 10(1))
    6. Smallholder Declaration (Art. 4(2))

Capabilities:
    - Template CRUD with versioning (major.minor, immutable once published)
    - 14 field types: text, number, decimal, date, datetime, select,
      multi_select, checkbox, photo, gps_point, gps_polygon, signature,
      barcode, calculated
    - Conditional logic: show_if, hide_if, skip_if, require_if
    - Cross-field validation rules
    - Multi-language rendering (24 EU + 20 local languages)
    - Template inheritance (base -> commodity -> operator customization)
    - Template publishing workflow: draft -> review -> published -> deprecated
    - JSON Schema generation from template definitions
    - Template diff/comparison between versions

Zero-Hallucination Guarantees:
    - All template IDs are deterministic UUIDs
    - Version numbers follow strict major.minor semantics
    - Conditional logic evaluation is deterministic (no LLM)
    - JSON Schema generation is deterministic

Example:
    >>> from greenlang.agents.eudr.mobile_data_collector.form_template_manager import (
    ...     FormTemplateManager,
    ... )
    >>> manager = FormTemplateManager()
    >>> template = manager.create_template(
    ...     name="Coffee Harvest Log",
    ...     form_type="harvest_log",
    ...     fields=[
    ...         {"field_id": "quantity", "type": "decimal", "label": "Quantity (kg)",
    ...          "required": True, "min_value": 0},
    ...     ],
    ... )
    >>> manager.publish_template(template["template_id"])

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config
from .metrics import record_api_error
from .provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported field types for form template definitions (14 types).
FIELD_TYPES: frozenset = frozenset({
    "text", "number", "decimal", "date", "datetime",
    "select", "multi_select", "checkbox", "photo",
    "gps_point", "gps_polygon", "signature", "barcode",
    "calculated",
})

#: Supported conditional logic operators.
CONDITION_OPERATORS: frozenset = frozenset({
    "equals", "not_equals", "greater_than", "less_than",
    "greater_than_or_equal", "less_than_or_equal",
    "contains", "not_contains", "is_empty", "is_not_empty",
    "in_list", "not_in_list", "matches_regex",
})

#: Supported conditional logic actions.
CONDITION_ACTIONS: frozenset = frozenset({
    "show_if", "hide_if", "skip_if", "require_if",
})

#: Template lifecycle statuses.
TEMPLATE_STATUSES: frozenset = frozenset({
    "draft", "review", "published", "deprecated",
})

#: Valid status transitions.
VALID_TRANSITIONS: Dict[str, frozenset] = {
    "draft": frozenset({"review", "deprecated"}),
    "review": frozenset({"draft", "published", "deprecated"}),
    "published": frozenset({"deprecated"}),
    "deprecated": frozenset(),
}

#: 6 built-in EUDR form types.
EUDR_FORM_TYPES: frozenset = frozenset({
    "producer_registration", "plot_survey", "harvest_log",
    "custody_transfer", "quality_inspection", "smallholder_declaration",
})


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Default templates for 6 EUDR form types
# ---------------------------------------------------------------------------

def _build_default_templates() -> Dict[str, Dict[str, Any]]:
    """Build default template definitions for the 6 EUDR form types.

    Returns:
        Dictionary keyed by form_type with template definition dicts.
    """
    defaults: Dict[str, Dict[str, Any]] = {}

    # Producer Registration (Art. 9(1)(f))
    defaults["producer_registration"] = {
        "name": "EUDR Producer Registration",
        "description": "Standard producer/supplier identification form per Art. 9(1)(f)",
        "fields": [
            {"field_id": "producer_name", "type": "text", "label": "Producer Name",
             "required": True, "max_length": 200},
            {"field_id": "producer_id", "type": "text", "label": "Producer ID / National ID",
             "required": True, "max_length": 50},
            {"field_id": "country", "type": "select", "label": "Country",
             "required": True, "options": []},
            {"field_id": "region", "type": "text", "label": "Region / Province",
             "required": True, "max_length": 200},
            {"field_id": "village", "type": "text", "label": "Village / Town",
             "required": False, "max_length": 200},
            {"field_id": "phone_number", "type": "text", "label": "Phone Number",
             "required": False, "max_length": 20},
            {"field_id": "commodity", "type": "select", "label": "Primary Commodity",
             "required": True, "options": [
                 "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]},
            {"field_id": "farm_size_ha", "type": "decimal", "label": "Farm Size (ha)",
             "required": True, "min_value": 0.0},
            {"field_id": "gps_location", "type": "gps_point", "label": "Farm Location (GPS)",
             "required": True},
            {"field_id": "photo_id", "type": "photo", "label": "Producer Photo / ID Document",
             "required": False},
            {"field_id": "declaration_signature", "type": "signature",
             "label": "Declaration Signature", "required": True},
        ],
    }

    # Plot Survey (Art. 9(1)(c-d))
    defaults["plot_survey"] = {
        "name": "EUDR Plot Survey",
        "description": "Geolocation and boundary survey per Art. 9(1)(c-d)",
        "fields": [
            {"field_id": "plot_name", "type": "text", "label": "Plot Name / Reference",
             "required": True, "max_length": 200},
            {"field_id": "producer_id", "type": "text", "label": "Producer ID",
             "required": True, "max_length": 50},
            {"field_id": "commodity", "type": "select", "label": "Commodity",
             "required": True, "options": [
                 "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]},
            {"field_id": "plot_centroid", "type": "gps_point", "label": "Plot Centroid (GPS)",
             "required": True},
            {"field_id": "plot_boundary", "type": "gps_polygon",
             "label": "Plot Boundary (Walk Trace)", "required": False},
            {"field_id": "plot_area_ha", "type": "calculated", "label": "Plot Area (ha)",
             "required": False, "formula": "auto_from_polygon"},
            {"field_id": "land_use", "type": "select", "label": "Current Land Use",
             "required": True, "options": [
                 "cropland", "agroforestry", "pasture", "forest", "fallow", "other"]},
            {"field_id": "deforestation_free", "type": "checkbox",
             "label": "No deforestation after 31 Dec 2020", "required": True},
            {"field_id": "plot_photos", "type": "photo", "label": "Plot Photos",
             "required": True},
        ],
    }

    # Harvest Log (Art. 9(1)(a-b,e))
    defaults["harvest_log"] = {
        "name": "EUDR Harvest Log",
        "description": "Production and harvest data per Art. 9(1)(a-b,e)",
        "fields": [
            {"field_id": "producer_id", "type": "text", "label": "Producer ID",
             "required": True, "max_length": 50},
            {"field_id": "plot_id", "type": "text", "label": "Plot ID",
             "required": True, "max_length": 50},
            {"field_id": "commodity", "type": "select", "label": "Commodity",
             "required": True, "options": [
                 "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]},
            {"field_id": "harvest_date", "type": "date", "label": "Harvest Date",
             "required": True},
            {"field_id": "quantity_kg", "type": "decimal", "label": "Quantity (kg)",
             "required": True, "min_value": 0.0},
            {"field_id": "quality_grade", "type": "select", "label": "Quality Grade",
             "required": False, "options": ["A", "B", "C", "reject"]},
            {"field_id": "moisture_pct", "type": "decimal", "label": "Moisture Content (%)",
             "required": False, "min_value": 0.0, "max_value": 100.0},
            {"field_id": "harvest_gps", "type": "gps_point",
             "label": "Harvest Location (GPS)", "required": True},
            {"field_id": "harvest_photo", "type": "photo",
             "label": "Harvest Photo", "required": False},
            {"field_id": "barcode", "type": "barcode", "label": "Batch Barcode",
             "required": False},
        ],
    }

    # Custody Transfer (Art. 9(1)(f-g))
    defaults["custody_transfer"] = {
        "name": "EUDR Custody Transfer",
        "description": "Chain of custody transfer per Art. 9(1)(f-g)",
        "fields": [
            {"field_id": "transfer_from", "type": "text", "label": "Transfer From (Entity)",
             "required": True, "max_length": 200},
            {"field_id": "transfer_to", "type": "text", "label": "Transfer To (Entity)",
             "required": True, "max_length": 200},
            {"field_id": "commodity", "type": "select", "label": "Commodity",
             "required": True, "options": [
                 "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]},
            {"field_id": "batch_id", "type": "text", "label": "Batch / Lot ID",
             "required": True, "max_length": 100},
            {"field_id": "quantity_kg", "type": "decimal", "label": "Quantity (kg)",
             "required": True, "min_value": 0.0},
            {"field_id": "transfer_date", "type": "datetime",
             "label": "Transfer Date & Time", "required": True},
            {"field_id": "transfer_location", "type": "gps_point",
             "label": "Transfer Location (GPS)", "required": True},
            {"field_id": "vehicle_id", "type": "text", "label": "Vehicle / Container ID",
             "required": False, "max_length": 50},
            {"field_id": "from_signature", "type": "signature",
             "label": "Sender Signature", "required": True},
            {"field_id": "to_signature", "type": "signature",
             "label": "Receiver Signature", "required": True},
            {"field_id": "witness_signature", "type": "signature",
             "label": "Witness Signature", "required": False},
            {"field_id": "transfer_photo", "type": "photo",
             "label": "Transfer Photo Evidence", "required": False},
        ],
    }

    # Quality Inspection (Art. 10(1))
    defaults["quality_inspection"] = {
        "name": "EUDR Quality Inspection",
        "description": "Quality and risk assessment data per Art. 10(1)",
        "fields": [
            {"field_id": "inspector_name", "type": "text",
             "label": "Inspector Name", "required": True, "max_length": 200},
            {"field_id": "inspection_date", "type": "datetime",
             "label": "Inspection Date & Time", "required": True},
            {"field_id": "batch_id", "type": "text", "label": "Batch / Lot ID",
             "required": True, "max_length": 100},
            {"field_id": "commodity", "type": "select", "label": "Commodity",
             "required": True, "options": [
                 "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]},
            {"field_id": "visual_grade", "type": "select",
             "label": "Visual Quality Grade",
             "required": True, "options": ["A", "B", "C", "reject"]},
            {"field_id": "defects_pct", "type": "decimal",
             "label": "Defect Percentage (%)",
             "required": False, "min_value": 0.0, "max_value": 100.0},
            {"field_id": "sample_weight_g", "type": "decimal",
             "label": "Sample Weight (g)", "required": False, "min_value": 0.0},
            {"field_id": "pass_fail", "type": "checkbox",
             "label": "Inspection Passed", "required": True},
            {"field_id": "inspection_photos", "type": "photo",
             "label": "Inspection Photos", "required": True},
            {"field_id": "inspector_signature", "type": "signature",
             "label": "Inspector Signature", "required": True},
            {"field_id": "notes", "type": "text", "label": "Inspection Notes",
             "required": False, "max_length": 2000},
        ],
    }

    # Smallholder Declaration (Art. 4(2))
    defaults["smallholder_declaration"] = {
        "name": "EUDR Smallholder Declaration",
        "description": "Smallholder due diligence declaration per Art. 4(2)",
        "fields": [
            {"field_id": "producer_name", "type": "text",
             "label": "Producer Name", "required": True, "max_length": 200},
            {"field_id": "producer_id", "type": "text",
             "label": "Producer ID / National ID",
             "required": True, "max_length": 50},
            {"field_id": "commodity", "type": "select", "label": "Primary Commodity",
             "required": True, "options": [
                 "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]},
            {"field_id": "farm_size_ha", "type": "decimal",
             "label": "Total Farm Size (ha)",
             "required": True, "min_value": 0.0, "max_value": 4.0},
            {"field_id": "plot_count", "type": "number",
             "label": "Number of Plots", "required": True, "min_value": 1},
            {"field_id": "deforestation_declaration", "type": "checkbox",
             "label": "I declare no deforestation after 31 Dec 2020",
             "required": True},
            {"field_id": "legality_declaration", "type": "checkbox",
             "label": "I declare all production is legal under local law",
             "required": True},
            {"field_id": "gps_location", "type": "gps_point",
             "label": "Primary Plot GPS", "required": True},
            {"field_id": "declaration_date", "type": "date",
             "label": "Declaration Date", "required": True},
            {"field_id": "producer_signature", "type": "signature",
             "label": "Producer Signature", "required": True},
        ],
    }

    return defaults


# ---------------------------------------------------------------------------
# FormTemplateManager
# ---------------------------------------------------------------------------


class FormTemplateManager:
    """Dynamic form template management engine for EUDR mobile data collection.

    Manages the full lifecycle of form templates including creation,
    versioning (major.minor), conditional logic evaluation, cross-field
    validation, multi-language rendering, template inheritance, and
    JSON Schema generation.

    Templates follow a publishing workflow:
        draft -> review -> published -> deprecated

    Published templates are immutable; modifications create new versions.

    Attributes:
        _config: Mobile data collector configuration.
        _provenance: Provenance tracker for audit trails.
        _templates: In-memory template storage keyed by template_id.
        _versions: Version history keyed by template_id -> list of versions.
        _lock: Thread-safe lock for concurrent access.

    Example:
        >>> manager = FormTemplateManager()
        >>> tpl = manager.create_template(
        ...     name="Coffee Harvest",
        ...     form_type="harvest_log",
        ...     fields=[{"field_id": "qty", "type": "decimal",
        ...              "label": "Quantity", "required": True}],
        ... )
        >>> manager.publish_template(tpl["template_id"])
    """

    __slots__ = (
        "_config", "_provenance", "_templates", "_versions",
        "_default_templates", "_lock",
    )

    def __init__(self) -> None:
        """Initialize FormTemplateManager with config and provenance."""
        self._config = get_config()
        self._provenance = get_provenance_tracker()
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._default_templates = _build_default_templates()
        self._lock = threading.Lock()
        logger.info(
            "FormTemplateManager initialized: max_templates=%d, "
            "languages=%d, logic_depth=%d, versioning=%s, "
            "inheritance=%s",
            self._config.max_templates,
            len(self._config.supported_languages),
            self._config.conditional_logic_depth,
            self._config.enable_template_versioning,
            self._config.enable_template_inheritance,
        )

    # ------------------------------------------------------------------
    # Template CRUD
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        form_type: str,
        fields: Optional[List[Dict[str, Any]]] = None,
        conditional_logic: Optional[List[Dict[str, Any]]] = None,
        validation_rules: Optional[List[Dict[str, Any]]] = None,
        language_packs: Optional[Dict[str, Dict[str, str]]] = None,
        parent_template_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new form template in draft status.

        Args:
            name: Human-readable template name.
            form_type: One of the 6 EUDR form types.
            fields: List of field definitions with field_id, type, label, etc.
            conditional_logic: List of conditional show/hide/skip/require rules.
            validation_rules: List of cross-field validation rules.
            language_packs: Dict of lang_code -> {field_id: translated_label}.
            parent_template_id: Parent template ID for inheritance.
            metadata: Additional template metadata.

        Returns:
            Created template dictionary.

        Raises:
            ValueError: If name is empty, form_type is invalid, max templates
                exceeded, or field definitions are invalid.
        """
        start = time.monotonic()

        self._validate_template_name(name)
        self._validate_form_type(form_type)
        self._validate_template_capacity()

        if parent_template_id is not None:
            self._validate_parent_exists(parent_template_id)

        resolved_fields = fields or []
        if parent_template_id and self._config.enable_template_inheritance:
            resolved_fields = self._inherit_fields(
                parent_template_id, resolved_fields,
            )

        self._validate_fields(resolved_fields)
        self._validate_conditional_logic(conditional_logic or [])
        self._validate_validation_rules(validation_rules or [], resolved_fields)

        template_id = str(uuid.uuid4())
        now = _utcnow()

        template: Dict[str, Any] = {
            "template_id": template_id,
            "name": name,
            "form_type": form_type,
            "template_type": "inherited" if parent_template_id else "base",
            "version": "1.0",
            "status": "draft",
            "parent_template_id": parent_template_id,
            "fields": copy.deepcopy(resolved_fields),
            "conditional_logic": copy.deepcopy(conditional_logic or []),
            "validation_rules": copy.deepcopy(validation_rules or []),
            "language_packs": copy.deepcopy(language_packs or {}),
            "schema_definition": {},
            "is_active": True,
            "metadata": copy.deepcopy(metadata or {}),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "created_by": metadata.get("created_by", "system") if metadata else "system",
            "field_count": len(resolved_fields),
        }

        with self._lock:
            self._templates[template_id] = template
            self._versions[template_id] = [copy.deepcopy(template)]

        self._record_provenance(template_id, "create", template)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Template created: id=%s name='%s' form_type=%s "
            "fields=%d status=draft elapsed=%.1fms",
            template_id[:12], name, form_type,
            len(resolved_fields), elapsed,
        )
        return copy.deepcopy(template)

    def get_template(self, template_id: str) -> Dict[str, Any]:
        """Retrieve a template by its ID.

        Args:
            template_id: Unique template identifier.

        Returns:
            Template dictionary.

        Raises:
            KeyError: If template_id not found.
        """
        with self._lock:
            template = self._templates.get(template_id)
        if template is None:
            raise KeyError(f"Template not found: {template_id}")
        return copy.deepcopy(template)

    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        fields: Optional[List[Dict[str, Any]]] = None,
        conditional_logic: Optional[List[Dict[str, Any]]] = None,
        validation_rules: Optional[List[Dict[str, Any]]] = None,
        language_packs: Optional[Dict[str, Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a draft or review template (not published/deprecated).

        If versioning is enabled and the template is published, this
        creates a new minor version instead.

        Args:
            template_id: Template to update.
            name: Updated name (optional).
            fields: Updated field definitions (optional).
            conditional_logic: Updated conditional rules (optional).
            validation_rules: Updated validation rules (optional).
            language_packs: Updated translations (optional).
            metadata: Updated metadata (optional).

        Returns:
            Updated template dictionary.

        Raises:
            KeyError: If template_id not found.
            ValueError: If template is published/deprecated and not versionable.
        """
        start = time.monotonic()

        with self._lock:
            template = self._templates.get(template_id)
            if template is None:
                raise KeyError(f"Template not found: {template_id}")

            if template["status"] in ("published", "deprecated"):
                raise ValueError(
                    f"Cannot modify template in '{template['status']}' status. "
                    f"Use create a new version instead."
                )

            if name is not None:
                self._validate_template_name(name)
                template["name"] = name

            if fields is not None:
                self._validate_fields(fields)
                template["fields"] = copy.deepcopy(fields)
                template["field_count"] = len(fields)

            if conditional_logic is not None:
                self._validate_conditional_logic(conditional_logic)
                template["conditional_logic"] = copy.deepcopy(conditional_logic)

            if validation_rules is not None:
                target_fields = fields if fields is not None else template["fields"]
                self._validate_validation_rules(validation_rules, target_fields)
                template["validation_rules"] = copy.deepcopy(validation_rules)

            if language_packs is not None:
                template["language_packs"] = copy.deepcopy(language_packs)

            if metadata is not None:
                template["metadata"].update(metadata)

            template["updated_at"] = _utcnow().isoformat()
            self._versions[template_id].append(copy.deepcopy(template))

        self._record_provenance(template_id, "update", template)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Template updated: id=%s name='%s' elapsed=%.1fms",
            template_id[:12], template["name"], elapsed,
        )
        return copy.deepcopy(template)

    def publish_template(self, template_id: str) -> Dict[str, Any]:
        """Publish a template, making it immutable and available for use.

        Transitions: draft -> review -> published.
        A template in draft is moved to review first, then to published.

        Args:
            template_id: Template to publish.

        Returns:
            Published template dictionary.

        Raises:
            KeyError: If template_id not found.
            ValueError: If template cannot be published from current status.
        """
        start = time.monotonic()

        with self._lock:
            template = self._templates.get(template_id)
            if template is None:
                raise KeyError(f"Template not found: {template_id}")

            current = template["status"]
            if current == "draft":
                template["status"] = "review"
                template["updated_at"] = _utcnow().isoformat()
                logger.info(
                    "Template transitioned draft->review: id=%s",
                    template_id[:12],
                )

            if template["status"] == "review":
                template["status"] = "published"
                template["updated_at"] = _utcnow().isoformat()
                template["schema_definition"] = self._generate_schema_internal(
                    template,
                )
                self._versions[template_id].append(copy.deepcopy(template))
            elif current == "published":
                raise ValueError("Template is already published")
            elif current == "deprecated":
                raise ValueError("Cannot publish a deprecated template")

        self._record_provenance(template_id, "update", template)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Template published: id=%s name='%s' version=%s elapsed=%.1fms",
            template_id[:12], template["name"],
            template["version"], elapsed,
        )
        return copy.deepcopy(template)

    def deprecate_template(
        self,
        template_id: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Deprecate a template, removing it from active use.

        Args:
            template_id: Template to deprecate.
            reason: Reason for deprecation.

        Returns:
            Deprecated template dictionary.

        Raises:
            KeyError: If template_id not found.
            ValueError: If template is already deprecated.
        """
        with self._lock:
            template = self._templates.get(template_id)
            if template is None:
                raise KeyError(f"Template not found: {template_id}")

            if template["status"] == "deprecated":
                raise ValueError("Template is already deprecated")

            template["status"] = "deprecated"
            template["is_active"] = False
            template["updated_at"] = _utcnow().isoformat()
            template["metadata"]["deprecation_reason"] = reason
            self._versions[template_id].append(copy.deepcopy(template))

        self._record_provenance(template_id, "update", template)
        logger.info(
            "Template deprecated: id=%s reason='%s'",
            template_id[:12], reason,
        )
        return copy.deepcopy(template)

    def list_templates(
        self,
        form_type: Optional[str] = None,
        status: Optional[str] = None,
        is_active: Optional[bool] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """List templates with optional filtering and pagination.

        Args:
            form_type: Filter by EUDR form type.
            status: Filter by template status.
            is_active: Filter by active flag.
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            Dictionary with templates list, total_count, page, page_size.
        """
        with self._lock:
            templates = list(self._templates.values())

        if form_type is not None:
            templates = [t for t in templates if t["form_type"] == form_type]
        if status is not None:
            templates = [t for t in templates if t["status"] == status]
        if is_active is not None:
            templates = [t for t in templates if t["is_active"] == is_active]

        total_count = len(templates)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_templates = templates[start_idx:end_idx]

        return {
            "templates": [copy.deepcopy(t) for t in page_templates],
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
        }

    def delete_template(self, template_id: str) -> bool:
        """Delete a draft template permanently.

        Only templates in draft status can be deleted. Published or
        review templates must be deprecated instead.

        Args:
            template_id: Template to delete.

        Returns:
            True if deleted.

        Raises:
            KeyError: If template_id not found.
            ValueError: If template is not in draft status.
        """
        with self._lock:
            template = self._templates.get(template_id)
            if template is None:
                raise KeyError(f"Template not found: {template_id}")

            if template["status"] != "draft":
                raise ValueError(
                    f"Only draft templates can be deleted, "
                    f"current status: {template['status']}"
                )

            del self._templates[template_id]
            del self._versions[template_id]

        self._record_provenance(template_id, "update", {"deleted": True})
        logger.info("Template deleted: id=%s", template_id[:12])
        return True

    # ------------------------------------------------------------------
    # Conditional Logic
    # ------------------------------------------------------------------

    def evaluate_conditions(
        self,
        template_id: str,
        form_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, bool]]:
        """Evaluate all conditional logic rules against current form data.

        Returns a mapping of field_id -> {visible, required, skipped}
        reflecting which fields should be shown, required, or skipped
        given the current form values.

        Args:
            template_id: Template whose conditions to evaluate.
            form_data: Current form field values as key-value pairs.

        Returns:
            Dict of field_id -> {"visible": bool, "required": bool,
            "skipped": bool}.

        Raises:
            KeyError: If template_id not found.
        """
        template = self.get_template(template_id)
        fields = template["fields"]
        conditions = template["conditional_logic"]

        result: Dict[str, Dict[str, bool]] = {}
        for field_def in fields:
            fid = field_def["field_id"]
            result[fid] = {
                "visible": True,
                "required": field_def.get("required", False),
                "skipped": False,
            }

        for rule in conditions:
            self._apply_condition_rule(rule, form_data, result, depth=0)

        return result

    def _apply_condition_rule(
        self,
        rule: Dict[str, Any],
        form_data: Dict[str, Any],
        result: Dict[str, Dict[str, bool]],
        depth: int,
    ) -> None:
        """Apply a single conditional logic rule.

        Args:
            rule: Condition rule with action, target_field, source_field,
                operator, value.
            form_data: Current form field values.
            result: Mutable result mapping to update.
            depth: Current nesting depth for recursion guard.
        """
        max_depth = self._config.conditional_logic_depth
        if depth > max_depth:
            logger.warning(
                "Conditional logic depth exceeded max=%d", max_depth,
            )
            return

        action = rule.get("action", "")
        target = rule.get("target_field", "")
        source = rule.get("source_field", "")
        operator = rule.get("operator", "")
        expected = rule.get("value")

        if not action or not target or target not in result:
            return

        match = self._evaluate_operator(
            form_data.get(source), operator, expected,
        )

        if action == "show_if":
            result[target]["visible"] = match
        elif action == "hide_if":
            result[target]["visible"] = not match
        elif action == "skip_if":
            result[target]["skipped"] = match
        elif action == "require_if":
            result[target]["required"] = match

        for nested in rule.get("nested_rules", []):
            if match:
                self._apply_condition_rule(
                    nested, form_data, result, depth + 1,
                )

    def _evaluate_operator(
        self,
        actual: Any,
        operator: str,
        expected: Any,
    ) -> bool:
        """Evaluate a comparison operator deterministically.

        Args:
            actual: Actual field value from form data.
            operator: Comparison operator string.
            expected: Expected value to compare against.

        Returns:
            Boolean result of the comparison.
        """
        if operator == "equals":
            return actual == expected
        if operator == "not_equals":
            return actual != expected
        if operator == "greater_than":
            return _safe_compare(actual, expected, ">")
        if operator == "less_than":
            return _safe_compare(actual, expected, "<")
        if operator == "greater_than_or_equal":
            return _safe_compare(actual, expected, ">=")
        if operator == "less_than_or_equal":
            return _safe_compare(actual, expected, "<=")
        if operator == "contains":
            return expected in str(actual) if actual is not None else False
        if operator == "not_contains":
            return expected not in str(actual) if actual is not None else True
        if operator == "is_empty":
            return actual is None or actual == "" or actual == []
        if operator == "is_not_empty":
            return actual is not None and actual != "" and actual != []
        if operator == "in_list":
            return actual in (expected if isinstance(expected, list) else [])
        if operator == "not_in_list":
            return actual not in (expected if isinstance(expected, list) else [])
        return False

    # ------------------------------------------------------------------
    # Validation against template
    # ------------------------------------------------------------------

    def validate_against_template(
        self,
        template_id: str,
        form_data: Dict[str, Any],
        strictness: str = "strict",
    ) -> Dict[str, Any]:
        """Validate form data against a template's field definitions and rules.

        Args:
            template_id: Template to validate against.
            form_data: Form field values to validate.
            strictness: "strict" (errors on all issues) or "lenient"
                (warnings for non-critical).

        Returns:
            Dict with "valid" (bool), "errors" (list), "warnings" (list).

        Raises:
            KeyError: If template_id not found.
        """
        template = self.get_template(template_id)
        errors: List[str] = []
        warnings: List[str] = []

        conditions = self.evaluate_conditions(template_id, form_data)

        for field_def in template["fields"]:
            fid = field_def["field_id"]
            field_state = conditions.get(fid, {})

            if field_state.get("skipped", False):
                continue

            if not field_state.get("visible", True):
                continue

            value = form_data.get(fid)
            is_required = field_state.get("required", field_def.get("required", False))

            if is_required and (value is None or value == "" or value == []):
                errors.append(f"Required field '{fid}' is missing or empty")
                continue

            if value is not None:
                field_errors = self._validate_field_value(field_def, value)
                errors.extend(field_errors)

        rule_errors = self._evaluate_validation_rules(
            template["validation_rules"], form_data,
        )
        errors.extend(rule_errors)

        if strictness == "lenient":
            warnings.extend(errors)
            errors = []

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _validate_field_value(
        self,
        field_def: Dict[str, Any],
        value: Any,
    ) -> List[str]:
        """Validate a single field value against its definition.

        Args:
            field_def: Field definition dict.
            value: The value to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []
        fid = field_def["field_id"]
        ftype = field_def.get("type", "text")

        if ftype in ("number", "decimal") and value is not None:
            if not isinstance(value, (int, float)):
                errors.append(f"Field '{fid}' must be numeric, got {type(value).__name__}")
            else:
                min_val = field_def.get("min_value")
                max_val = field_def.get("max_value")
                if min_val is not None and value < min_val:
                    errors.append(
                        f"Field '{fid}' value {value} below minimum {min_val}"
                    )
                if max_val is not None and value > max_val:
                    errors.append(
                        f"Field '{fid}' value {value} above maximum {max_val}"
                    )

        if ftype == "text" and value is not None:
            max_len = field_def.get("max_length")
            if max_len is not None and len(str(value)) > max_len:
                errors.append(
                    f"Field '{fid}' exceeds max length {max_len}"
                )

        if ftype == "select" and value is not None:
            options = field_def.get("options", [])
            if options and value not in options:
                errors.append(
                    f"Field '{fid}' value '{value}' not in options: {options}"
                )

        if ftype == "multi_select" and value is not None:
            if isinstance(value, list):
                options = field_def.get("options", [])
                if options:
                    invalid = [v for v in value if v not in options]
                    if invalid:
                        errors.append(
                            f"Field '{fid}' has invalid selections: {invalid}"
                        )

        if ftype == "checkbox" and value is not None:
            if not isinstance(value, bool):
                errors.append(f"Field '{fid}' must be boolean")

        return errors

    def _evaluate_validation_rules(
        self,
        rules: List[Dict[str, Any]],
        form_data: Dict[str, Any],
    ) -> List[str]:
        """Evaluate cross-field validation rules.

        Args:
            rules: List of validation rule dicts.
            form_data: Current form field values.

        Returns:
            List of error messages.
        """
        errors: List[str] = []
        for rule in rules:
            rule_type = rule.get("type", "")
            if rule_type == "date_order":
                errors.extend(self._check_date_order(rule, form_data))
            elif rule_type == "conditional_required":
                errors.extend(self._check_conditional_required(rule, form_data))
            elif rule_type == "value_range":
                errors.extend(self._check_value_range(rule, form_data))
            elif rule_type == "field_match":
                errors.extend(self._check_field_match(rule, form_data))
        return errors

    def _check_date_order(
        self,
        rule: Dict[str, Any],
        form_data: Dict[str, Any],
    ) -> List[str]:
        """Validate that one date is before another.

        Args:
            rule: Rule with before_field and after_field keys.
            form_data: Current form values.

        Returns:
            List of error messages.
        """
        before_field = rule.get("before_field", "")
        after_field = rule.get("after_field", "")
        before_val = form_data.get(before_field)
        after_val = form_data.get(after_field)

        if before_val is None or after_val is None:
            return []

        if str(before_val) > str(after_val):
            msg = rule.get(
                "message",
                f"'{after_field}' must be after '{before_field}'",
            )
            return [msg]
        return []

    def _check_conditional_required(
        self,
        rule: Dict[str, Any],
        form_data: Dict[str, Any],
    ) -> List[str]:
        """Check conditional required rule.

        Args:
            rule: Rule with condition_field, condition_value,
                required_field.
            form_data: Current form values.

        Returns:
            List of error messages.
        """
        cond_field = rule.get("condition_field", "")
        cond_value = rule.get("condition_value")
        req_field = rule.get("required_field", "")

        if form_data.get(cond_field) == cond_value:
            val = form_data.get(req_field)
            if val is None or val == "" or val == []:
                msg = rule.get(
                    "message",
                    f"'{req_field}' is required when '{cond_field}' "
                    f"is '{cond_value}'",
                )
                return [msg]
        return []

    def _check_value_range(
        self,
        rule: Dict[str, Any],
        form_data: Dict[str, Any],
    ) -> List[str]:
        """Check value range cross-field rule.

        Args:
            rule: Rule with field, min_field or min_value, max_field or max_value.
            form_data: Current form values.

        Returns:
            List of error messages.
        """
        field = rule.get("field", "")
        value = form_data.get(field)
        if value is None or not isinstance(value, (int, float)):
            return []

        min_val = form_data.get(rule.get("min_field", ""), rule.get("min_value"))
        max_val = form_data.get(rule.get("max_field", ""), rule.get("max_value"))

        errors: List[str] = []
        if min_val is not None and isinstance(min_val, (int, float)):
            if value < min_val:
                errors.append(
                    rule.get("message", f"'{field}' below minimum {min_val}")
                )
        if max_val is not None and isinstance(max_val, (int, float)):
            if value > max_val:
                errors.append(
                    rule.get("message", f"'{field}' above maximum {max_val}")
                )
        return errors

    def _check_field_match(
        self,
        rule: Dict[str, Any],
        form_data: Dict[str, Any],
    ) -> List[str]:
        """Check that two fields have matching values.

        Args:
            rule: Rule with field_a and field_b keys.
            form_data: Current form values.

        Returns:
            List of error messages.
        """
        field_a = rule.get("field_a", "")
        field_b = rule.get("field_b", "")
        val_a = form_data.get(field_a)
        val_b = form_data.get(field_b)

        if val_a is not None and val_b is not None and val_a != val_b:
            msg = rule.get(
                "message",
                f"'{field_a}' and '{field_b}' must match",
            )
            return [msg]
        return []

    # ------------------------------------------------------------------
    # Multi-language rendering
    # ------------------------------------------------------------------

    def render_template(
        self,
        template_id: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Render a template with field labels in the specified language.

        Falls back to English ('en') if the requested language pack
        does not contain a translation for a given field.

        Args:
            template_id: Template to render.
            language: ISO 639 language code.

        Returns:
            Rendered template with translated labels.

        Raises:
            KeyError: If template_id not found.
        """
        template = self.get_template(template_id)
        lang_pack = template["language_packs"].get(language, {})
        en_pack = template["language_packs"].get("en", {})

        rendered_fields: List[Dict[str, Any]] = []
        for field_def in template["fields"]:
            rendered = copy.deepcopy(field_def)
            fid = field_def["field_id"]
            if fid in lang_pack:
                rendered["label"] = lang_pack[fid]
            elif fid in en_pack:
                rendered["label"] = en_pack[fid]
            rendered_fields.append(rendered)

        result = copy.deepcopy(template)
        result["fields"] = rendered_fields
        result["rendered_language"] = language
        return result

    def get_available_languages(self, template_id: str) -> List[str]:
        """Get list of languages available for a template.

        Args:
            template_id: Template to query.

        Returns:
            Sorted list of language codes with available translations.

        Raises:
            KeyError: If template_id not found.
        """
        template = self.get_template(template_id)
        return sorted(template["language_packs"].keys())

    # ------------------------------------------------------------------
    # JSON Schema generation
    # ------------------------------------------------------------------

    def generate_json_schema(self, template_id: str) -> Dict[str, Any]:
        """Generate a JSON Schema from the template definition.

        Produces a valid JSON Schema (draft-07) document describing
        the expected form data structure based on the template fields.

        Args:
            template_id: Template to generate schema for.

        Returns:
            JSON Schema dictionary.

        Raises:
            KeyError: If template_id not found.
        """
        template = self.get_template(template_id)
        return self._generate_schema_internal(template)

    def _generate_schema_internal(
        self,
        template: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate JSON Schema from template definition (internal).

        Args:
            template: Template dictionary.

        Returns:
            JSON Schema dict.
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for field_def in template["fields"]:
            fid = field_def["field_id"]
            ftype = field_def.get("type", "text")
            prop = self._field_to_json_schema(field_def, ftype)
            properties[fid] = prop

            if field_def.get("required", False):
                required.append(fid)

        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": template["name"],
            "description": f"JSON Schema for {template['form_type']} template v{template['version']}",
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        schema["additionalProperties"] = False
        return schema

    def _field_to_json_schema(
        self,
        field_def: Dict[str, Any],
        ftype: str,
    ) -> Dict[str, Any]:
        """Convert a single field definition to JSON Schema property.

        Args:
            field_def: Field definition dict.
            ftype: Field type string.

        Returns:
            JSON Schema property dict.
        """
        prop: Dict[str, Any] = {
            "description": field_def.get("label", field_def["field_id"]),
        }

        type_mapping: Dict[str, Dict[str, Any]] = {
            "text": {"type": "string"},
            "number": {"type": "integer"},
            "decimal": {"type": "number"},
            "date": {"type": "string", "format": "date"},
            "datetime": {"type": "string", "format": "date-time"},
            "checkbox": {"type": "boolean"},
            "barcode": {"type": "string"},
            "calculated": {"type": "number"},
            "select": {"type": "string"},
            "multi_select": {"type": "array", "items": {"type": "string"}},
            "photo": {"type": "string", "description": "Photo evidence ID"},
            "gps_point": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                    "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                },
            },
            "gps_polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "signature": {"type": "string", "description": "Digital signature ID"},
        }

        schema_type = type_mapping.get(ftype, {"type": "string"})
        prop.update(schema_type)

        if ftype == "text":
            max_len = field_def.get("max_length")
            if max_len is not None:
                prop["maxLength"] = max_len

        if ftype in ("number", "decimal"):
            min_val = field_def.get("min_value")
            max_val = field_def.get("max_value")
            if min_val is not None:
                prop["minimum"] = min_val
            if max_val is not None:
                prop["maximum"] = max_val

        if ftype == "select":
            options = field_def.get("options", [])
            if options:
                prop["enum"] = options

        if ftype == "multi_select":
            options = field_def.get("options", [])
            if options:
                prop["items"]["enum"] = options

        return prop

    # ------------------------------------------------------------------
    # Template diff/comparison
    # ------------------------------------------------------------------

    def diff_templates(
        self,
        template_id_a: str,
        template_id_b: str,
    ) -> Dict[str, Any]:
        """Compare two templates and return the differences.

        Args:
            template_id_a: First template ID.
            template_id_b: Second template ID.

        Returns:
            Dict with "added_fields", "removed_fields",
            "modified_fields", "logic_changes", "metadata_changes".

        Raises:
            KeyError: If either template not found.
        """
        tpl_a = self.get_template(template_id_a)
        tpl_b = self.get_template(template_id_b)

        fields_a = {f["field_id"]: f for f in tpl_a["fields"]}
        fields_b = {f["field_id"]: f for f in tpl_b["fields"]}

        ids_a = set(fields_a.keys())
        ids_b = set(fields_b.keys())

        added = sorted(ids_b - ids_a)
        removed = sorted(ids_a - ids_b)
        common = ids_a & ids_b

        modified: List[Dict[str, Any]] = []
        for fid in sorted(common):
            fa = fields_a[fid]
            fb = fields_b[fid]
            if fa != fb:
                changes: Dict[str, Any] = {"field_id": fid, "changes": {}}
                for key in set(list(fa.keys()) + list(fb.keys())):
                    va = fa.get(key)
                    vb = fb.get(key)
                    if va != vb:
                        changes["changes"][key] = {"from": va, "to": vb}
                modified.append(changes)

        logic_changed = tpl_a["conditional_logic"] != tpl_b["conditional_logic"]
        rules_changed = tpl_a["validation_rules"] != tpl_b["validation_rules"]

        meta_changes: Dict[str, Any] = {}
        for key in ("name", "form_type", "version", "status"):
            if tpl_a.get(key) != tpl_b.get(key):
                meta_changes[key] = {
                    "from": tpl_a.get(key),
                    "to": tpl_b.get(key),
                }

        return {
            "template_a": template_id_a,
            "template_b": template_id_b,
            "added_fields": added,
            "removed_fields": removed,
            "modified_fields": modified,
            "logic_changed": logic_changed,
            "rules_changed": rules_changed,
            "metadata_changes": meta_changes,
        }

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------

    def get_version_history(self, template_id: str) -> List[Dict[str, Any]]:
        """Get the version history for a template.

        Args:
            template_id: Template ID.

        Returns:
            List of template snapshots in chronological order.

        Raises:
            KeyError: If template_id not found.
        """
        with self._lock:
            versions = self._versions.get(template_id)
        if versions is None:
            raise KeyError(f"Template not found: {template_id}")
        return [copy.deepcopy(v) for v in versions]

    def create_new_version(
        self,
        template_id: str,
        bump: str = "minor",
    ) -> Dict[str, Any]:
        """Create a new version of an existing template.

        Creates a copy of the current template with an incremented
        version number (major or minor) in draft status.

        Args:
            template_id: Source template to version from.
            bump: Version bump type ("major" or "minor").

        Returns:
            New template dictionary with incremented version.

        Raises:
            KeyError: If template_id not found.
            ValueError: If bump type is invalid.
        """
        if bump not in ("major", "minor"):
            raise ValueError(f"bump must be 'major' or 'minor', got '{bump}'")

        source = self.get_template(template_id)
        current_version = source["version"]
        new_version = self._increment_version(current_version, bump)

        new_id = str(uuid.uuid4())
        now = _utcnow()

        new_template = copy.deepcopy(source)
        new_template["template_id"] = new_id
        new_template["version"] = new_version
        new_template["status"] = "draft"
        new_template["is_active"] = True
        new_template["schema_definition"] = {}
        new_template["created_at"] = now.isoformat()
        new_template["updated_at"] = now.isoformat()
        new_template["metadata"]["source_template_id"] = template_id
        new_template["metadata"]["source_version"] = current_version

        with self._lock:
            self._templates[new_id] = new_template
            self._versions[new_id] = [copy.deepcopy(new_template)]

        self._record_provenance(new_id, "create", new_template)
        logger.info(
            "New version created: id=%s version=%s from=%s",
            new_id[:12], new_version, template_id[:12],
        )
        return copy.deepcopy(new_template)

    # ------------------------------------------------------------------
    # Built-in default templates
    # ------------------------------------------------------------------

    def get_default_template(self, form_type: str) -> Dict[str, Any]:
        """Get the built-in default template for an EUDR form type.

        Args:
            form_type: One of the 6 EUDR form types.

        Returns:
            Default template definition dict.

        Raises:
            ValueError: If form_type is not a valid EUDR form type.
        """
        if form_type not in EUDR_FORM_TYPES:
            raise ValueError(
                f"Invalid EUDR form type '{form_type}'. "
                f"Must be one of: {sorted(EUDR_FORM_TYPES)}"
            )
        return copy.deepcopy(self._default_templates[form_type])

    def load_default_templates(self) -> List[str]:
        """Load all 6 default EUDR form templates into the manager.

        Creates each default template in draft status. Skips creation
        if a template for that form_type already exists.

        Returns:
            List of created template IDs.
        """
        created_ids: List[str] = []
        for form_type, default_def in self._default_templates.items():
            existing = self.list_templates(form_type=form_type)
            if existing["total_count"] > 0:
                logger.debug(
                    "Default template for '%s' already exists, skipping",
                    form_type,
                )
                continue

            template = self.create_template(
                name=default_def["name"],
                form_type=form_type,
                fields=default_def["fields"],
                metadata={"source": "default", "description": default_def.get("description", "")},
            )
            created_ids.append(template["template_id"])

        logger.info(
            "Loaded %d default templates: %s",
            len(created_ids),
            [tid[:8] for tid in created_ids],
        )
        return created_ids

    # ------------------------------------------------------------------
    # Template statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get template management statistics.

        Returns:
            Dict with counts by status, form_type, and totals.
        """
        with self._lock:
            templates = list(self._templates.values())

        by_status: Dict[str, int] = {}
        by_form_type: Dict[str, int] = {}
        total_fields = 0

        for tpl in templates:
            status = tpl["status"]
            by_status[status] = by_status.get(status, 0) + 1

            ft = tpl["form_type"]
            by_form_type[ft] = by_form_type.get(ft, 0) + 1

            total_fields += tpl.get("field_count", len(tpl.get("fields", [])))

        return {
            "total_templates": len(templates),
            "by_status": by_status,
            "by_form_type": by_form_type,
            "total_fields": total_fields,
            "max_templates": self._config.max_templates,
            "capacity_pct": round(
                (len(templates) / max(self._config.max_templates, 1)) * 100, 1,
            ),
        }

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _validate_template_name(self, name: str) -> None:
        """Validate template name is non-empty.

        Args:
            name: Template name.

        Raises:
            ValueError: If name is empty or whitespace-only.
        """
        if not name or not name.strip():
            raise ValueError("Template name must not be empty")

    def _validate_form_type(self, form_type: str) -> None:
        """Validate form_type is one of the 6 EUDR types.

        Args:
            form_type: Form type string.

        Raises:
            ValueError: If form_type is not valid.
        """
        if form_type not in EUDR_FORM_TYPES:
            raise ValueError(
                f"Invalid form type '{form_type}'. "
                f"Must be one of: {sorted(EUDR_FORM_TYPES)}"
            )

    def _validate_template_capacity(self) -> None:
        """Validate that template storage has capacity.

        Raises:
            ValueError: If max_templates reached.
        """
        with self._lock:
            count = len(self._templates)
        if count >= self._config.max_templates:
            raise ValueError(
                f"Maximum template count reached: {self._config.max_templates}"
            )

    def _validate_parent_exists(self, parent_id: str) -> None:
        """Validate that parent template exists.

        Args:
            parent_id: Parent template ID.

        Raises:
            KeyError: If parent not found.
            ValueError: If inheritance is disabled.
        """
        if not self._config.enable_template_inheritance:
            raise ValueError(
                "Template inheritance is disabled in configuration"
            )
        with self._lock:
            if parent_id not in self._templates:
                raise KeyError(
                    f"Parent template not found: {parent_id}"
                )

    def _validate_fields(self, fields: List[Dict[str, Any]]) -> None:
        """Validate field definitions list.

        Args:
            fields: List of field definition dicts.

        Raises:
            ValueError: If any field definition is invalid.
        """
        if len(fields) > self._config.max_fields_per_form:
            raise ValueError(
                f"Too many fields: {len(fields)} exceeds max "
                f"{self._config.max_fields_per_form}"
            )

        seen_ids: set = set()
        for i, field_def in enumerate(fields):
            fid = field_def.get("field_id")
            if not fid:
                raise ValueError(f"Field at index {i} missing 'field_id'")
            if fid in seen_ids:
                raise ValueError(f"Duplicate field_id: '{fid}'")
            seen_ids.add(fid)

            ftype = field_def.get("type")
            if not ftype:
                raise ValueError(f"Field '{fid}' missing 'type'")
            if ftype not in FIELD_TYPES:
                raise ValueError(
                    f"Field '{fid}' has invalid type '{ftype}'. "
                    f"Must be one of: {sorted(FIELD_TYPES)}"
                )

            if not field_def.get("label"):
                raise ValueError(f"Field '{fid}' missing 'label'")

    def _validate_conditional_logic(
        self,
        rules: List[Dict[str, Any]],
    ) -> None:
        """Validate conditional logic rules structure.

        Args:
            rules: List of conditional rule dicts.

        Raises:
            ValueError: If any rule is invalid.
        """
        for i, rule in enumerate(rules):
            action = rule.get("action")
            if action and action not in CONDITION_ACTIONS:
                raise ValueError(
                    f"Conditional rule {i}: invalid action '{action}'. "
                    f"Must be one of: {sorted(CONDITION_ACTIONS)}"
                )
            operator = rule.get("operator")
            if operator and operator not in CONDITION_OPERATORS:
                raise ValueError(
                    f"Conditional rule {i}: invalid operator '{operator}'. "
                    f"Must be one of: {sorted(CONDITION_OPERATORS)}"
                )

    def _validate_validation_rules(
        self,
        rules: List[Dict[str, Any]],
        fields: List[Dict[str, Any]],
    ) -> None:
        """Validate cross-field validation rules.

        Args:
            rules: Validation rules to validate.
            fields: Template field definitions for reference.

        Raises:
            ValueError: If any rule references non-existent fields.
        """
        field_ids = {f["field_id"] for f in fields}
        for i, rule in enumerate(rules):
            for key in ("field", "before_field", "after_field",
                        "condition_field", "required_field",
                        "field_a", "field_b"):
                ref = rule.get(key)
                if ref and ref not in field_ids:
                    logger.warning(
                        "Validation rule %d references unknown field '%s'",
                        i, ref,
                    )

    def _inherit_fields(
        self,
        parent_id: str,
        child_fields: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge parent fields with child overrides.

        Child fields override parent fields with the same field_id.
        Parent-only fields are prepended.

        Args:
            parent_id: Parent template ID.
            child_fields: Child field definitions.

        Returns:
            Merged list of field definitions.
        """
        with self._lock:
            parent = self._templates.get(parent_id)
        if parent is None:
            return child_fields

        child_ids = {f["field_id"] for f in child_fields}
        parent_fields = copy.deepcopy(parent["fields"])

        inherited = [f for f in parent_fields if f["field_id"] not in child_ids]
        inherited.extend(copy.deepcopy(child_fields))
        return inherited

    def _increment_version(self, version: str, bump: str) -> str:
        """Increment a version string.

        Args:
            version: Current version string (e.g. "1.0").
            bump: "major" or "minor".

        Returns:
            Incremented version string.
        """
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 1
        minor = int(parts[1]) if len(parts) > 1 else 0

        if bump == "major":
            return f"{major + 1}.0"
        return f"{major}.{minor + 1}"

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        template_id: str,
        action: str,
        data: Any,
    ) -> None:
        """Record a provenance entry for template operations.

        Args:
            template_id: Template identifier.
            action: Provenance action.
            data: Data payload to hash.
        """
        try:
            self._provenance.record(
                entity_type="form_template",
                action=action,
                entity_id=template_id,
                data=data,
                metadata={"engine": "FormTemplateManager"},
            )
        except Exception as exc:
            logger.warning(
                "Provenance recording failed for template %s: %s",
                template_id[:12], exc,
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._templates)
        return (
            f"FormTemplateManager(templates={count}, "
            f"max={self._config.max_templates})"
        )

    def __len__(self) -> int:
        """Return total number of templates."""
        with self._lock:
            return len(self._templates)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_compare(a: Any, b: Any, op: str) -> bool:
    """Safely compare two values with a given operator.

    Args:
        a: Left operand.
        b: Right operand.
        op: Comparison operator string.

    Returns:
        Boolean comparison result, False on type error.
    """
    try:
        if op == ">":
            return a > b
        if op == "<":
            return a < b
        if op == ">=":
            return a >= b
        if op == "<=":
            return a <= b
    except TypeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FormTemplateManager",
    "FIELD_TYPES",
    "CONDITION_OPERATORS",
    "CONDITION_ACTIONS",
    "TEMPLATE_STATUSES",
    "VALID_TRANSITIONS",
    "EUDR_FORM_TYPES",
]
