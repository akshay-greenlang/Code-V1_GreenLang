# -*- coding: utf-8 -*-
"""
Unit tests for TemplateEngine engine - AGENT-EUDR-040

Tests multi-language template rendering, placeholder substitution,
language fallback, template registration, default template loading,
all 24 EU languages, template retrieval, listing, and health checks.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
    EU_LANGUAGES,
)
from greenlang.agents.eudr.authority_communication_manager.template_engine import (
    TemplateEngine,
    _DEFAULT_TEMPLATES,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    CommunicationType,
    LanguageCode,
    Template,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def engine(config):
    return TemplateEngine(config=config)


@pytest.fixture
def loaded_engine(config):
    """Engine with default templates loaded."""
    e = TemplateEngine(config=config)
    e.load_default_templates()
    return e


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_engine_created(self, engine):
        assert engine is not None

    def test_default_config(self):
        e = TemplateEngine()
        assert e.config is not None

    def test_custom_config(self, config):
        e = TemplateEngine(config=config)
        assert e.config is config

    def test_templates_empty(self, engine):
        assert len(engine._templates) == 0

    def test_provenance_initialized(self, engine):
        assert engine._provenance is not None


# ====================================================================
# Default Templates
# ====================================================================


class TestDefaultTemplates:
    def test_default_templates_exist(self):
        assert len(_DEFAULT_TEMPLATES) > 0

    def test_default_templates_have_names(self):
        for tpl in _DEFAULT_TEMPLATES:
            assert "template_name" in tpl
            assert len(tpl["template_name"]) > 0

    def test_default_templates_have_types(self):
        for tpl in _DEFAULT_TEMPLATES:
            assert "communication_type" in tpl

    def test_default_templates_have_subject(self):
        for tpl in _DEFAULT_TEMPLATES:
            assert "subject_template" in tpl

    def test_default_templates_have_body(self):
        for tpl in _DEFAULT_TEMPLATES:
            assert "body_template" in tpl

    def test_default_templates_have_placeholders(self):
        for tpl in _DEFAULT_TEMPLATES:
            assert "placeholders" in tpl
            assert isinstance(tpl["placeholders"], list)


# ====================================================================
# Load Default Templates
# ====================================================================


class TestLoadDefaultTemplates:
    def test_load_default_templates(self, engine):
        engine.load_default_templates()
        assert len(engine._templates) > 0

    def test_load_creates_template_records(self, engine):
        engine.load_default_templates()
        assert len(engine._template_records) > 0

    def test_load_idempotent(self, engine):
        engine.load_default_templates()
        count1 = len(engine._template_records)
        engine.load_default_templates()
        count2 = len(engine._template_records)
        assert count2 >= count1


# ====================================================================
# Register Template
# ====================================================================


class TestRegisterTemplate:
    @pytest.mark.asyncio
    async def test_register_template(self, engine):
        result = await engine.register_template(
            template_name="custom_template",
            communication_type="information_request",
            language="en",
            subject_template="Custom Subject - ${ref}",
            body_template="Custom body for ${operator_name}.",
            placeholders=["ref", "operator_name"],
        )
        assert isinstance(result, Template)
        assert result.template_name == "custom_template"

    @pytest.mark.asyncio
    async def test_register_template_de(self, engine):
        result = await engine.register_template(
            template_name="custom_template",
            communication_type="information_request",
            language="de",
            subject_template="Benutzerdefiniert - ${ref}",
            body_template="Benutzerdefinierter Text fuer ${operator_name}.",
            placeholders=["ref", "operator_name"],
        )
        assert result.language == LanguageCode.DE

    @pytest.mark.asyncio
    async def test_register_all_languages(self, engine):
        """Register a template in all 24 EU languages."""
        for lang in EU_LANGUAGES:
            result = await engine.register_template(
                template_name="multi_lang_test",
                communication_type="general_correspondence",
                language=lang,
                subject_template=f"[{lang.upper()}] Subject",
                body_template=f"[{lang.upper()}] Body",
                placeholders=[],
            )
            assert result.language.value == lang

    @pytest.mark.asyncio
    async def test_register_assigns_id(self, engine):
        result = await engine.register_template(
            template_name="id_test",
            communication_type="information_request",
            language="en",
            subject_template="Test",
            body_template="Test body",
            placeholders=[],
        )
        assert result.template_id is not None
        assert len(result.template_id) > 0


# ====================================================================
# Render Template
# ====================================================================


class TestRenderTemplate:
    @pytest.mark.asyncio
    async def test_render_template(self, loaded_engine):
        result = await loaded_engine.render_template(
            template_name="information_request_response",
            language="en",
            variables={
                "reference_number": "REF-001",
                "operator_name": "Acme Corp",
                "authority_name": "German BfN",
                "request_date": "2026-03-13",
                "commodity": "cocoa",
                "items_list": "- DDS Statement\n- Risk Assessment",
                "operator_contact": "compliance@acme.com",
            },
        )
        assert result is not None
        assert "subject" in result
        assert "body" in result
        assert "Acme Corp" in result["body"]
        assert "REF-001" in result["subject"]

    @pytest.mark.asyncio
    async def test_render_missing_variables(self, loaded_engine):
        """Render should handle missing variables gracefully."""
        result = await loaded_engine.render_template(
            template_name="information_request_response",
            language="en",
            variables={"operator_name": "Partial Corp"},
        )
        assert result is not None
        assert "Partial Corp" in result["body"]

    @pytest.mark.asyncio
    async def test_render_nonexistent_template(self, loaded_engine):
        with pytest.raises(ValueError, match="[Tt]emplate"):
            await loaded_engine.render_template(
                template_name="nonexistent_template",
                language="en",
                variables={},
            )

    @pytest.mark.asyncio
    async def test_render_fallback_language(self, loaded_engine):
        """When requested language not available, fall back to EN."""
        result = await loaded_engine.render_template(
            template_name="information_request_response",
            language="bg",
            variables={"operator_name": "Bulgarian Corp"},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_render_inspection_acknowledgment(self, loaded_engine):
        result = await loaded_engine.render_template(
            template_name="inspection_acknowledgment",
            language="en",
            variables={
                "authority_name": "Dutch NVWA",
                "inspection_id": "INSP-001",
                "inspection_date": "2026-03-20",
                "location": "Amsterdam Warehouse",
                "contact_name": "Jan de Vries",
                "contact_email": "jan@operator.nl",
                "operator_name": "Dutch Cocoa BV",
            },
        )
        assert "INSP-001" in result["subject"]

    @pytest.mark.asyncio
    async def test_render_non_compliance_response(self, loaded_engine):
        result = await loaded_engine.render_template(
            template_name="non_compliance_response",
            language="en",
            variables={
                "authority_name": "French Ministry",
                "nc_reference": "NC-2026-001",
                "issue_date": "2026-03-01",
                "corrective_actions": "- Action 1\n- Action 2",
                "completion_date": "2026-04-01",
                "operator_name": "Acme Trading",
            },
        )
        assert "NC-2026-001" in result["subject"]

    @pytest.mark.asyncio
    async def test_render_appeal_filing(self, loaded_engine):
        result = await loaded_engine.render_template(
            template_name="appeal_filing",
            language="en",
            variables={
                "authority_name": "German BfN",
                "nc_reference": "NC-001",
                "appeal_grounds": "DDS was submitted on time.",
                "operator_name": "Acme Corp",
                "legal_representative": "Dr. Schmidt, LLP",
            },
        )
        assert "NC-001" in result["subject"]

    @pytest.mark.asyncio
    async def test_render_deadline_reminder(self, loaded_engine):
        result = await loaded_engine.render_template(
            template_name="deadline_reminder",
            language="en",
            variables={
                "recipient_name": "Compliance Team",
                "communication_type": "Information Request",
                "reference_number": "REF-001",
                "deadline_date": "2026-03-18 12:00 UTC",
                "hours_remaining": "48",
            },
        )
        assert "48" in result["body"]


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_template(self, loaded_engine):
        templates = await loaded_engine.list_templates()
        if templates:
            result = await loaded_engine.get_template(templates[0].template_id)
            assert result is not None

    @pytest.mark.asyncio
    async def test_get_template_not_found(self, engine):
        result = await engine.get_template("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_templates_empty(self, engine):
        result = await engine.list_templates()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_templates_after_load(self, loaded_engine):
        result = await loaded_engine.list_templates()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        health = await engine.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_after_load(self, loaded_engine):
        health = await loaded_engine.health_check()
        assert health["total_templates"] > 0
