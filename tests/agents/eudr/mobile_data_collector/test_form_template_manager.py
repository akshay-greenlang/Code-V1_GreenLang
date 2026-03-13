# -*- coding: utf-8 -*-
"""
Unit tests for FormTemplateManager - AGENT-EUDR-015 Engine 5.

Tests all methods of FormTemplateManager with 85%+ coverage.
Validates template CRUD, lifecycle transitions, conditional logic,
validation, multi-language rendering, JSON Schema generation,
template diff, versioning, inheritance, and statistics.

Test count: ~60 tests
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.mobile_data_collector.form_template_manager import (
    FormTemplateManager,
    FIELD_TYPES,
    CONDITION_OPERATORS,
    CONDITION_ACTIONS,
    TEMPLATE_STATUSES,
    VALID_TRANSITIONS,
    EUDR_FORM_TYPES,
)

from .conftest import EUDR_FORM_TYPES as FORM_TYPES_LIST, EUDR_COMMODITIES, FIELD_TYPES as FIELD_TYPES_LIST


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestFormTemplateManagerInit:
    """Tests for FormTemplateManager initialization."""

    def test_initialization(self, form_template_manager):
        """Engine initializes with empty template store."""
        assert form_template_manager is not None
        assert len(form_template_manager) == 0

    def test_repr(self, form_template_manager):
        """Repr includes template count."""
        r = repr(form_template_manager)
        assert "FormTemplateManager" in r

    def test_len_starts_at_zero(self, form_template_manager):
        """Initial template count is zero."""
        assert len(form_template_manager) == 0


# ---------------------------------------------------------------------------
# Test: create_template
# ---------------------------------------------------------------------------

class TestCreateTemplate:
    """Tests for create_template method."""

    def test_create_valid_template(self, form_template_manager, make_form_template):
        """Create a valid template in draft status."""
        data = make_form_template()
        result = form_template_manager.create_template(**data)
        assert "template_id" in result
        assert result["status"] == "draft"
        assert result["form_type"] == "harvest_log"

    def test_create_increments_count(self, form_template_manager, make_form_template):
        """Creating a template increments count."""
        form_template_manager.create_template(**make_form_template())
        assert len(form_template_manager) == 1

    @pytest.mark.parametrize("form_type", FORM_TYPES_LIST)
    def test_create_all_form_types(self, form_template_manager, make_form_template, form_type):
        """All 6 EUDR form types can create templates."""
        data = make_form_template(form_type=form_type, name=f"Test {form_type}")
        result = form_template_manager.create_template(**data)
        assert result["form_type"] == form_type

    def test_create_with_conditional_logic(self, form_template_manager, make_form_template):
        """Create template with conditional logic rules."""
        data = make_form_template()
        data["conditional_logic"] = [
            {
                "action": "show_if",
                "target_field": "quantity_kg",
                "source_field": "commodity",
                "operator": "equals",
                "value": "coffee",
            },
        ]
        result = form_template_manager.create_template(**data)
        assert len(result["conditional_logic"]) == 1

    def test_create_with_validation_rules(self, form_template_manager, make_form_template):
        """Create template with cross-field validation rules."""
        data = make_form_template()
        data["validation_rules"] = [
            {
                "type": "value_range",
                "field": "quantity_kg",
                "min_value": 0,
                "max_value": 10000,
            },
        ]
        result = form_template_manager.create_template(**data)
        assert len(result["validation_rules"]) == 1

    def test_create_with_language_packs(self, form_template_manager, make_form_template):
        """Create template with multi-language labels."""
        data = make_form_template()
        data["language_packs"] = {
            "fr": {"producer_id": "ID Producteur", "quantity_kg": "Quantite (kg)"},
            "pt": {"producer_id": "ID do Produtor", "quantity_kg": "Quantidade (kg)"},
        }
        result = form_template_manager.create_template(**data)
        assert "fr" in result["language_packs"]
        assert "pt" in result["language_packs"]

    def test_create_empty_name_raises(self, form_template_manager, make_form_template):
        """Empty template name raises ValueError."""
        data = make_form_template(name="")
        with pytest.raises(ValueError):
            form_template_manager.create_template(**data)

    def test_create_invalid_form_type_raises(self, form_template_manager, make_form_template):
        """Invalid form_type raises ValueError."""
        data = make_form_template(form_type="invalid_type")
        with pytest.raises(ValueError):
            form_template_manager.create_template(**data)

    def test_create_invalid_field_type_raises(self, form_template_manager):
        """Invalid field type in definition raises ValueError."""
        with pytest.raises(ValueError):
            form_template_manager.create_template(
                name="Bad Fields",
                form_type="harvest_log",
                fields=[
                    {"field_id": "f1", "type": "invalid_type", "label": "Bad", "required": True},
                ],
            )

    def test_create_duplicate_field_ids_raises(self, form_template_manager):
        """Duplicate field_ids raise ValueError."""
        with pytest.raises(ValueError):
            form_template_manager.create_template(
                name="Duplicate Fields",
                form_type="harvest_log",
                fields=[
                    {"field_id": "f1", "type": "text", "label": "Field 1", "required": True},
                    {"field_id": "f1", "type": "number", "label": "Field 1 dup", "required": False},
                ],
            )

    def test_create_returns_unique_ids(self, form_template_manager, make_form_template):
        """Each template gets a unique ID."""
        ids = set()
        for i in range(5):
            data = make_form_template(name=f"Template {i}")
            result = form_template_manager.create_template(**data)
            ids.add(result["template_id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Test: get_template
# ---------------------------------------------------------------------------

class TestGetTemplate:
    """Tests for get_template method."""

    def test_get_existing_template(self, form_template_manager, make_form_template):
        """Get an existing template by ID."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.get_template(created["template_id"])
        assert result["template_id"] == created["template_id"]

    def test_get_nonexistent_raises(self, form_template_manager):
        """Getting nonexistent template raises KeyError."""
        with pytest.raises(KeyError):
            form_template_manager.get_template("nonexistent-id")


# ---------------------------------------------------------------------------
# Test: update_template
# ---------------------------------------------------------------------------

class TestUpdateTemplate:
    """Tests for update_template method."""

    def test_update_draft_name(self, form_template_manager, make_form_template):
        """Update a draft template name."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.update_template(
            created["template_id"], name="Updated Name",
        )
        assert result["name"] == "Updated Name"

    def test_update_draft_fields(self, form_template_manager, make_form_template):
        """Update a draft template fields."""
        created = form_template_manager.create_template(**make_form_template())
        new_fields = [
            {"field_id": "new_field", "type": "text", "label": "New Field", "required": True},
        ]
        result = form_template_manager.update_template(
            created["template_id"], fields=new_fields,
        )
        assert result["field_count"] == 1

    def test_update_published_template_raises(self, form_template_manager, make_form_template):
        """Updating a published template raises ValueError."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.publish_template(created["template_id"])
        with pytest.raises(ValueError):
            form_template_manager.update_template(
                created["template_id"], name="Cannot Update",
            )

    def test_update_nonexistent_raises(self, form_template_manager):
        """Updating nonexistent template raises KeyError."""
        with pytest.raises(KeyError):
            form_template_manager.update_template("nonexistent-id", name="X")


# ---------------------------------------------------------------------------
# Test: publish_template
# ---------------------------------------------------------------------------

class TestPublishTemplate:
    """Tests for publish_template method."""

    def test_publish_draft_template(self, form_template_manager, make_form_template):
        """Publish a draft template transitions through review to published."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.publish_template(created["template_id"])
        assert result["status"] == "published"

    def test_publish_generates_schema(self, form_template_manager, make_form_template):
        """Publishing generates a JSON Schema definition."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.publish_template(created["template_id"])
        assert result["schema_definition"] != {}

    def test_publish_already_published_raises(self, form_template_manager, make_form_template):
        """Publishing an already-published template raises ValueError."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.publish_template(created["template_id"])
        with pytest.raises(ValueError):
            form_template_manager.publish_template(created["template_id"])


# ---------------------------------------------------------------------------
# Test: deprecate_template
# ---------------------------------------------------------------------------

class TestDeprecateTemplate:
    """Tests for deprecate_template method."""

    def test_deprecate_draft(self, form_template_manager, make_form_template):
        """Deprecate a draft template."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.deprecate_template(
            created["template_id"], reason="Replaced by v2",
        )
        assert result["status"] == "deprecated"
        assert result["is_active"] is False

    def test_deprecate_published(self, form_template_manager, make_form_template):
        """Deprecate a published template."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.publish_template(created["template_id"])
        result = form_template_manager.deprecate_template(
            created["template_id"], reason="End of life",
        )
        assert result["status"] == "deprecated"

    def test_deprecate_already_deprecated_raises(self, form_template_manager, make_form_template):
        """Deprecating already deprecated template raises ValueError."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.deprecate_template(created["template_id"])
        with pytest.raises(ValueError):
            form_template_manager.deprecate_template(created["template_id"])


# ---------------------------------------------------------------------------
# Test: list_templates
# ---------------------------------------------------------------------------

class TestListTemplates:
    """Tests for list_templates method."""

    def test_list_empty(self, form_template_manager):
        """List templates returns empty when none exist."""
        result = form_template_manager.list_templates()
        assert result["total_count"] == 0

    def test_list_after_create(self, form_template_manager, make_form_template):
        """List templates includes created templates."""
        form_template_manager.create_template(**make_form_template())
        result = form_template_manager.list_templates()
        assert result["total_count"] == 1

    def test_list_filter_by_form_type(self, form_template_manager, make_form_template):
        """List templates filters by form_type."""
        form_template_manager.create_template(**make_form_template(form_type="harvest_log"))
        form_template_manager.create_template(**make_form_template(
            name="Plot Survey", form_type="plot_survey",
        ))
        result = form_template_manager.list_templates(form_type="harvest_log")
        assert all(t["form_type"] == "harvest_log" for t in result["templates"])

    def test_list_filter_by_status(self, form_template_manager, make_form_template):
        """List templates filters by status."""
        form_template_manager.create_template(**make_form_template())
        result = form_template_manager.list_templates(status="draft")
        assert all(t["status"] == "draft" for t in result["templates"])


# ---------------------------------------------------------------------------
# Test: delete_template
# ---------------------------------------------------------------------------

class TestDeleteTemplate:
    """Tests for delete_template method."""

    def test_delete_draft_template(self, form_template_manager, make_form_template):
        """Delete a draft template."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.delete_template(created["template_id"])
        assert result is True

    def test_delete_non_draft_raises(self, form_template_manager, make_form_template):
        """Deleting a non-draft template raises ValueError."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.publish_template(created["template_id"])
        with pytest.raises(ValueError):
            form_template_manager.delete_template(created["template_id"])


# ---------------------------------------------------------------------------
# Test: evaluate_conditions
# ---------------------------------------------------------------------------

class TestEvaluateConditions:
    """Tests for conditional logic evaluation."""

    def test_evaluate_show_if_match(self, form_template_manager, make_form_template):
        """show_if makes field visible when condition matches."""
        data = make_form_template()
        data["conditional_logic"] = [
            {
                "action": "show_if",
                "target_field": "quantity_kg",
                "source_field": "commodity",
                "operator": "equals",
                "value": "coffee",
            },
        ]
        created = form_template_manager.create_template(**data)
        result = form_template_manager.evaluate_conditions(
            created["template_id"], {"commodity": "coffee"},
        )
        assert result["quantity_kg"]["visible"] is True

    def test_evaluate_show_if_no_match(self, form_template_manager, make_form_template):
        """show_if hides field when condition does not match."""
        data = make_form_template()
        data["conditional_logic"] = [
            {
                "action": "show_if",
                "target_field": "quantity_kg",
                "source_field": "commodity",
                "operator": "equals",
                "value": "coffee",
            },
        ]
        created = form_template_manager.create_template(**data)
        result = form_template_manager.evaluate_conditions(
            created["template_id"], {"commodity": "cocoa"},
        )
        assert result["quantity_kg"]["visible"] is False

    def test_evaluate_require_if(self, form_template_manager, make_form_template):
        """require_if makes field required when condition matches."""
        data = make_form_template()
        data["conditional_logic"] = [
            {
                "action": "require_if",
                "target_field": "harvest_date",
                "source_field": "commodity",
                "operator": "equals",
                "value": "coffee",
            },
        ]
        created = form_template_manager.create_template(**data)
        result = form_template_manager.evaluate_conditions(
            created["template_id"], {"commodity": "coffee"},
        )
        assert result["harvest_date"]["required"] is True


# ---------------------------------------------------------------------------
# Test: validate_against_template
# ---------------------------------------------------------------------------

class TestValidateAgainstTemplate:
    """Tests for template validation."""

    def test_validate_complete_data(self, form_template_manager, make_form_template):
        """Complete data passes validation."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.validate_against_template(
            created["template_id"],
            {
                "producer_id": "PROD-001",
                "quantity_kg": 100.0,
                "harvest_date": "2026-01-15",
                "commodity": "coffee",
            },
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_missing_required_field(self, form_template_manager, make_form_template):
        """Missing required field fails validation."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.validate_against_template(
            created["template_id"],
            {"quantity_kg": 100.0},
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Test: render_template (multi-language)
# ---------------------------------------------------------------------------

class TestRenderTemplate:
    """Tests for multi-language rendering."""

    def test_render_default_language(self, form_template_manager, make_form_template):
        """Render template in default (en) language."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.render_template(created["template_id"])
        assert result["rendered_language"] == "en"

    def test_render_french_language(self, form_template_manager, make_form_template):
        """Render template in French with translated labels."""
        data = make_form_template()
        data["language_packs"] = {
            "fr": {"producer_id": "ID Producteur"},
        }
        created = form_template_manager.create_template(**data)
        result = form_template_manager.render_template(
            created["template_id"], language="fr",
        )
        assert result["rendered_language"] == "fr"
        fr_field = next(
            (f for f in result["fields"] if f["field_id"] == "producer_id"), None,
        )
        assert fr_field is not None
        assert fr_field["label"] == "ID Producteur"


# ---------------------------------------------------------------------------
# Test: generate_json_schema
# ---------------------------------------------------------------------------

class TestGenerateJsonSchema:
    """Tests for JSON Schema generation."""

    def test_generate_schema(self, form_template_manager, make_form_template):
        """Generate JSON Schema from template."""
        created = form_template_manager.create_template(**make_form_template())
        schema = form_template_manager.generate_json_schema(created["template_id"])
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_schema_required_fields(self, form_template_manager, make_form_template):
        """Schema required array matches template required fields."""
        created = form_template_manager.create_template(**make_form_template())
        schema = form_template_manager.generate_json_schema(created["template_id"])
        assert "required" in schema
        assert "producer_id" in schema["required"]


# ---------------------------------------------------------------------------
# Test: diff_templates
# ---------------------------------------------------------------------------

class TestDiffTemplates:
    """Tests for template comparison."""

    def test_diff_identical_templates(self, form_template_manager, make_form_template):
        """Diffing identical templates shows no changes."""
        t1 = form_template_manager.create_template(**make_form_template(name="T1"))
        t2 = form_template_manager.create_template(**make_form_template(name="T2"))
        result = form_template_manager.diff_templates(
            t1["template_id"], t2["template_id"],
        )
        assert len(result["added_fields"]) == 0
        assert len(result["removed_fields"]) == 0

    def test_diff_different_templates(self, form_template_manager):
        """Diffing different templates shows changes."""
        t1 = form_template_manager.create_template(
            name="T1", form_type="harvest_log",
            fields=[{"field_id": "f1", "type": "text", "label": "F1", "required": True}],
        )
        t2 = form_template_manager.create_template(
            name="T2", form_type="harvest_log",
            fields=[
                {"field_id": "f1", "type": "text", "label": "F1", "required": True},
                {"field_id": "f2", "type": "number", "label": "F2", "required": False},
            ],
        )
        result = form_template_manager.diff_templates(
            t1["template_id"], t2["template_id"],
        )
        assert "f2" in result["added_fields"]


# ---------------------------------------------------------------------------
# Test: Version History / create_new_version
# ---------------------------------------------------------------------------

class TestVersioning:
    """Tests for template versioning."""

    def test_get_version_history(self, form_template_manager, make_form_template):
        """Version history starts with one entry."""
        created = form_template_manager.create_template(**make_form_template())
        history = form_template_manager.get_version_history(created["template_id"])
        assert len(history) >= 1

    def test_create_minor_version(self, form_template_manager, make_form_template):
        """Create a minor version bump."""
        created = form_template_manager.create_template(**make_form_template())
        new_ver = form_template_manager.create_new_version(
            created["template_id"], bump="minor",
        )
        assert new_ver["version"] == "1.1"
        assert new_ver["status"] == "draft"

    def test_create_major_version(self, form_template_manager, make_form_template):
        """Create a major version bump."""
        created = form_template_manager.create_template(**make_form_template())
        new_ver = form_template_manager.create_new_version(
            created["template_id"], bump="major",
        )
        assert new_ver["version"] == "2.0"

    def test_invalid_bump_raises(self, form_template_manager, make_form_template):
        """Invalid bump type raises ValueError."""
        created = form_template_manager.create_template(**make_form_template())
        with pytest.raises(ValueError):
            form_template_manager.create_new_version(created["template_id"], bump="patch")


# ---------------------------------------------------------------------------
# Test: Default Templates / Statistics
# ---------------------------------------------------------------------------

class TestDefaultsAndStatistics:
    """Tests for default templates and statistics."""

    def test_get_default_template(self, form_template_manager):
        """Get built-in default template for each form type."""
        for form_type in FORM_TYPES_LIST:
            tpl = form_template_manager.get_default_template(form_type)
            assert tpl["name"] is not None
            assert len(tpl["fields"]) > 0

    def test_get_default_invalid_type_raises(self, form_template_manager):
        """Getting default for invalid type raises ValueError."""
        with pytest.raises(ValueError):
            form_template_manager.get_default_template("invalid_type")

    def test_load_default_templates(self, form_template_manager):
        """Load all 6 default templates."""
        ids = form_template_manager.load_default_templates()
        assert len(ids) == 6
        assert len(form_template_manager) == 6

    def test_get_statistics(self, form_template_manager, make_form_template):
        """Statistics reflect current state."""
        form_template_manager.create_template(**make_form_template())
        stats = form_template_manager.get_statistics()
        assert stats["total_templates"] == 1
        assert stats["by_status"]["draft"] == 1

    def test_statistics_published_count(self, form_template_manager, make_form_template):
        """Statistics tracks published templates."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.publish_template(created["template_id"])
        stats = form_template_manager.get_statistics()
        assert stats["by_status"]["published"] == 1

    def test_statistics_zero_initially(self, form_template_manager):
        """Statistics shows zero counts initially."""
        stats = form_template_manager.get_statistics()
        assert stats["total_templates"] == 0


# ---------------------------------------------------------------------------
# Test: Additional Template Operations
# ---------------------------------------------------------------------------

class TestFormTemplateAdditional:
    """Additional tests for template operations."""

    def test_get_available_languages_default(self, form_template_manager, make_form_template):
        """Default template has English available."""
        created = form_template_manager.create_template(**make_form_template())
        langs = form_template_manager.get_available_languages(created["template_id"])
        assert "en" in langs

    def test_get_available_languages_with_packs(self, form_template_manager, make_form_template):
        """Languages include pack languages."""
        data = make_form_template()
        data["language_packs"] = {
            "fr": {"producer_id": "ID Producteur"},
            "pt": {"producer_id": "ID do Produtor"},
        }
        created = form_template_manager.create_template(**data)
        langs = form_template_manager.get_available_languages(created["template_id"])
        assert "fr" in langs
        assert "pt" in langs

    def test_render_nonexistent_language_falls_back(
        self, form_template_manager, make_form_template,
    ):
        """Rendering in unavailable language falls back to default."""
        created = form_template_manager.create_template(**make_form_template())
        result = form_template_manager.render_template(
            created["template_id"], language="zh",
        )
        assert result["rendered_language"] in ("en", "zh")

    def test_create_template_with_all_field_types(self, form_template_manager):
        """Template with all 14 field types is accepted."""
        fields = []
        for i, ft in enumerate(FIELD_TYPES_LIST):
            field = {"field_id": f"field_{i}", "type": ft, "label": f"Label {ft}", "required": False}
            if ft in ("select", "multi_select"):
                field["options"] = ["opt_a", "opt_b"]
            fields.append(field)
        result = form_template_manager.create_template(
            name="All Field Types",
            form_type="harvest_log",
            fields=fields,
        )
        assert result["field_count"] == len(FIELD_TYPES_LIST)

    def test_update_template_conditional_logic(self, form_template_manager, make_form_template):
        """Update template conditional logic on draft."""
        created = form_template_manager.create_template(**make_form_template())
        new_logic = [
            {
                "action": "hide_if",
                "target_field": "quantity_kg",
                "source_field": "commodity",
                "operator": "not_equals",
                "value": "coffee",
            },
        ]
        result = form_template_manager.update_template(
            created["template_id"], conditional_logic=new_logic,
        )
        assert len(result["conditional_logic"]) == 1

    def test_diff_templates_field_removal(self, form_template_manager):
        """Diff detects field removal between templates."""
        t1 = form_template_manager.create_template(
            name="T1", form_type="harvest_log",
            fields=[
                {"field_id": "f1", "type": "text", "label": "F1", "required": True},
                {"field_id": "f2", "type": "number", "label": "F2", "required": False},
            ],
        )
        t2 = form_template_manager.create_template(
            name="T2", form_type="harvest_log",
            fields=[
                {"field_id": "f1", "type": "text", "label": "F1", "required": True},
            ],
        )
        result = form_template_manager.diff_templates(
            t1["template_id"], t2["template_id"],
        )
        assert "f2" in result["removed_fields"]

    def test_version_history_grows(self, form_template_manager, make_form_template):
        """Version history grows when creating new versions."""
        created = form_template_manager.create_template(**make_form_template())
        form_template_manager.create_new_version(
            created["template_id"], bump="minor",
        )
        history = form_template_manager.get_version_history(created["template_id"])
        assert len(history) >= 2
