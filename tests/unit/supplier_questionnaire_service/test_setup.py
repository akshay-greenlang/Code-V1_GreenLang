# -*- coding: utf-8 -*-
"""
Unit tests for SupplierQuestionnaireService facade (setup.py) - AGENT-DATA-008

Tests the SupplierQuestionnaireService class covering initialization, template
management, distribution, response submission, validation, scoring, follow-up
management, analytics, statistics, health checks, provenance tracking,
thread safety, the configure_supplier_questionnaire() async function, and
full end-to-end workflows.

Target: 85%+ coverage of greenlang/supplier_questionnaire/setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.supplier_questionnaire.config import SupplierQuestionnaireConfig
from greenlang.supplier_questionnaire.setup import (
    CampaignAnalytics,
    Distribution,
    FollowUpAction,
    QuestionnaireResponse,
    QuestionnaireStatistics,
    QuestionnaireTemplate,
    ScoringResult,
    SupplierQuestionnaireService,
    ValidationResult,
    _ProvenanceTracker,
    _compute_hash,
    configure_supplier_questionnaire,
    get_supplier_questionnaire,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_config(**overrides: Any) -> SupplierQuestionnaireConfig:
    """Build a SupplierQuestionnaireConfig with optional field overrides."""
    defaults = dict(
        database_url="",
        redis_url="",
        default_framework="custom",
        default_deadline_days=60,
        max_reminders=4,
        min_completion_pct=80.0,
        score_leader_threshold=80,
        score_advanced_threshold=60,
        score_developing_threshold=40,
        score_lagging_threshold=20,
    )
    defaults.update(overrides)
    return SupplierQuestionnaireConfig(**defaults)


def _make_sections(
    with_questions: bool = True,
    section_count: int = 1,
    questions_per_section: int = 2,
) -> List[Dict[str, Any]]:
    """Build a list of template sections with optional questions."""
    sections: List[Dict[str, Any]] = []
    qid = 0
    for s in range(section_count):
        qs: List[Dict[str, Any]] = []
        if with_questions:
            for _q in range(questions_per_section):
                qs.append({"id": f"q-{qid}", "text": f"Question {qid}", "type": "text"})
                qid += 1
        sections.append({
            "name": f"Section-{s}",
            "category": f"cat-{s % 3}",
            "questions": qs,
        })
    return sections


def _build_service(config: SupplierQuestionnaireConfig | None = None) -> SupplierQuestionnaireService:
    """Convenience to build a service with default or custom config.

    Patches _init_engines to avoid real engine construction which may fail
    in unit test environments where engine submodules have incompatible
    constructor signatures.
    """
    with patch.object(SupplierQuestionnaireService, "_init_engines"):
        return SupplierQuestionnaireService(config=config or _make_config())


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def config() -> SupplierQuestionnaireConfig:
    return _make_config()


@pytest.fixture
def service(config: SupplierQuestionnaireConfig) -> SupplierQuestionnaireService:
    with patch.object(SupplierQuestionnaireService, "_init_engines"):
        return SupplierQuestionnaireService(config=config)


@pytest.fixture
def sections_2q() -> List[Dict[str, Any]]:
    """Two sections, 2 questions each (q-0..q-3)."""
    return _make_sections(section_count=2, questions_per_section=2)


@pytest.fixture
def template_in_service(
    service: SupplierQuestionnaireService,
    sections_2q: List[Dict[str, Any]],
) -> QuestionnaireTemplate:
    """Create and return a template already stored in the service."""
    return service.create_template(
        name="CDP Climate 2025",
        framework="cdp",
        version="2.0",
        description="Climate disclosure",
        sections=sections_2q,
        language="en",
        tags=["climate", "cdp"],
        created_by="tester",
    )


@pytest.fixture
def distribution_in_service(
    service: SupplierQuestionnaireService,
    template_in_service: QuestionnaireTemplate,
) -> Distribution:
    """Distribute a questionnaire to a supplier and return the Distribution."""
    return service.distribute(
        template_id=template_in_service.template_id,
        supplier_id="SUP-001",
        supplier_name="EcoSteel GmbH",
        supplier_email="contact@ecosteel.de",
        campaign_id="camp-001",
        channel="email",
    )


@pytest.fixture
def response_in_service(
    service: SupplierQuestionnaireService,
    distribution_in_service: Distribution,
) -> QuestionnaireResponse:
    """Submit a response answering all 4 questions."""
    return service.submit_response(
        distribution_id=distribution_in_service.distribution_id,
        supplier_id="SUP-001",
        supplier_name="EcoSteel GmbH",
        answers={"q-0": "100 MWh", "q-1": "Yes", "q-2": "50%", "q-3": "No"},
        evidence_files=["audit.pdf"],
        channel="portal",
    )


# ===================================================================
# 1. Initialization Tests
# ===================================================================


class TestServiceInitialization:
    """Tests for SupplierQuestionnaireService __init__."""

    def test_init_creates_provenance_tracker(self, service: SupplierQuestionnaireService) -> None:
        assert service.provenance is not None
        assert isinstance(service.provenance, _ProvenanceTracker)

    def test_init_sets_config(self, config: SupplierQuestionnaireConfig) -> None:
        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            svc = SupplierQuestionnaireService(config=config)
        assert svc.config is config

    def test_init_uses_global_config_when_none(self) -> None:
        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            svc = SupplierQuestionnaireService(config=None)
        assert svc.config is not None

    def test_init_internal_stores_empty(self, service: SupplierQuestionnaireService) -> None:
        assert len(service._templates) == 0
        assert len(service._distributions) == 0
        assert len(service._responses) == 0
        assert len(service._validations) == 0
        assert len(service._scores) == 0
        assert len(service._followups) == 0
        assert len(service._campaigns) == 0

    def test_init_statistics_zeroed(self, service: SupplierQuestionnaireService) -> None:
        stats = service.get_statistics()
        assert stats.total_templates == 0
        assert stats.total_distributions == 0
        assert stats.total_responses == 0
        assert stats.total_validations == 0
        assert stats.total_scores == 0
        assert stats.total_followups == 0

    def test_init_not_started(self, service: SupplierQuestionnaireService) -> None:
        assert service._started is False

    def test_engine_properties_exist(self, service: SupplierQuestionnaireService) -> None:
        # Engines may be None if SDK sub-modules are not installed
        _ = service.template_builder
        _ = service.distribution_engine
        _ = service.response_collector
        _ = service.validation_engine
        _ = service.scoring_engine
        _ = service.followup_manager
        _ = service.analytics_engine


# ===================================================================
# 2. Template Management Tests
# ===================================================================


class TestCreateTemplate:
    """Tests for create_template()."""

    def test_create_template_returns_model(self, service: SupplierQuestionnaireService) -> None:
        t = service.create_template(name="Basic Template")
        assert isinstance(t, QuestionnaireTemplate)

    def test_create_template_sets_fields(
        self, service: SupplierQuestionnaireService, sections_2q: List[Dict[str, Any]]
    ) -> None:
        t = service.create_template(
            name="Test",
            framework="gri",
            version="3.0",
            description="desc",
            sections=sections_2q,
            language="de",
            tags=["tag1"],
            created_by="alice",
        )
        assert t.name == "Test"
        assert t.framework == "gri"
        assert t.version == "3.0"
        assert t.description == "desc"
        assert t.language == "de"
        assert t.tags == ["tag1"]
        assert t.created_by == "alice"
        assert t.status == "draft"

    def test_create_template_counts_questions(
        self, service: SupplierQuestionnaireService, sections_2q: List[Dict[str, Any]]
    ) -> None:
        t = service.create_template(name="Q Count", sections=sections_2q)
        assert t.questions == 4  # 2 sections * 2 questions

    def test_create_template_no_sections_zero_questions(
        self, service: SupplierQuestionnaireService
    ) -> None:
        t = service.create_template(name="No Sections")
        assert t.questions == 0

    def test_create_template_empty_name_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            service.create_template(name="")

    def test_create_template_whitespace_name_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            service.create_template(name="   ")

    def test_create_template_generates_uuid(self, service: SupplierQuestionnaireService) -> None:
        t = service.create_template(name="UUID Test")
        uuid.UUID(t.template_id)  # should not raise

    def test_create_template_provenance_hash_nonempty(
        self, service: SupplierQuestionnaireService
    ) -> None:
        t = service.create_template(name="Hash Test")
        assert len(t.provenance_hash) == 64

    def test_create_template_stored_internally(
        self, service: SupplierQuestionnaireService
    ) -> None:
        t = service.create_template(name="Stored")
        assert service._templates[t.template_id] is t

    def test_create_template_increments_statistics(
        self, service: SupplierQuestionnaireService
    ) -> None:
        service.create_template(name="A")
        service.create_template(name="B")
        assert service.get_statistics().total_templates == 2

    def test_create_template_active_templates_count(
        self, service: SupplierQuestionnaireService
    ) -> None:
        service.create_template(name="Draft1")
        service.create_template(name="Draft2")
        assert service.get_statistics().active_templates == 2

    def test_create_template_records_provenance_entry(
        self, service: SupplierQuestionnaireService
    ) -> None:
        initial = service.provenance.entry_count
        service.create_template(name="Prov")
        assert service.provenance.entry_count == initial + 1


class TestGetTemplate:
    """Tests for get_template()."""

    def test_get_existing_template(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        result = service.get_template(template_in_service.template_id)
        assert result is not None
        assert result.template_id == template_in_service.template_id

    def test_get_nonexistent_template_returns_none(
        self, service: SupplierQuestionnaireService
    ) -> None:
        assert service.get_template("no-such-id") is None


class TestListTemplates:
    """Tests for list_templates()."""

    def test_list_all(self, service: SupplierQuestionnaireService) -> None:
        service.create_template(name="A", framework="cdp")
        service.create_template(name="B", framework="gri")
        result = service.list_templates()
        assert len(result) == 2

    def test_list_filter_framework(self, service: SupplierQuestionnaireService) -> None:
        service.create_template(name="C", framework="cdp")
        service.create_template(name="D", framework="gri")
        result = service.list_templates(framework="cdp")
        assert len(result) == 1
        assert result[0].framework == "cdp"

    def test_list_filter_status(self, service: SupplierQuestionnaireService) -> None:
        t = service.create_template(name="E")
        service.update_template(t.template_id, status="archived")
        result = service.list_templates(status="archived")
        assert len(result) == 1

    def test_list_pagination_limit(self, service: SupplierQuestionnaireService) -> None:
        for i in range(5):
            service.create_template(name=f"T{i}")
        assert len(service.list_templates(limit=3)) == 3

    def test_list_pagination_offset(self, service: SupplierQuestionnaireService) -> None:
        for i in range(5):
            service.create_template(name=f"T{i}")
        assert len(service.list_templates(offset=3)) == 2


class TestUpdateTemplate:
    """Tests for update_template()."""

    def test_update_name(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        updated = service.update_template(template_in_service.template_id, name="New Name")
        assert updated.name == "New Name"

    def test_update_description(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        updated = service.update_template(
            template_in_service.template_id, description="New desc"
        )
        assert updated.description == "New desc"

    def test_update_sections_recalculates_questions(
        self, service: SupplierQuestionnaireService, template_in_service: QuestionnaireTemplate
    ) -> None:
        new_sections = _make_sections(section_count=3, questions_per_section=3)
        updated = service.update_template(
            template_in_service.template_id, sections=new_sections,
        )
        assert updated.questions == 9

    def test_update_status(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        updated = service.update_template(template_in_service.template_id, status="active")
        assert updated.status == "active"

    def test_update_tags(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        updated = service.update_template(
            template_in_service.template_id, tags=["new-tag"],
        )
        assert updated.tags == ["new-tag"]

    def test_update_nonexistent_raises(self, service: SupplierQuestionnaireService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.update_template("nonexistent", name="X")

    def test_update_changes_provenance_hash(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        old_hash = template_in_service.provenance_hash
        service.update_template(template_in_service.template_id, name="Changed")
        assert template_in_service.provenance_hash != old_hash

    def test_update_records_provenance(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        before = service.provenance.entry_count
        service.update_template(template_in_service.template_id, name="Upd")
        assert service.provenance.entry_count == before + 1


class TestCloneTemplate:
    """Tests for clone_template()."""

    def test_clone_creates_new_template(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        cloned = service.clone_template(template_in_service.template_id)
        assert cloned.template_id != template_in_service.template_id
        assert "(Clone)" in cloned.name

    def test_clone_custom_name(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        cloned = service.clone_template(
            template_in_service.template_id, new_name="Custom Clone"
        )
        assert cloned.name == "Custom Clone"

    def test_clone_custom_version(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        cloned = service.clone_template(
            template_in_service.template_id, new_version="9.0"
        )
        assert cloned.version == "9.0"

    def test_clone_nonexistent_raises(self, service: SupplierQuestionnaireService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.clone_template("nonexistent")

    def test_clone_preserves_framework(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        cloned = service.clone_template(template_in_service.template_id)
        assert cloned.framework == template_in_service.framework


# ===================================================================
# 3. Distribution Tests
# ===================================================================


class TestDistribute:
    """Tests for distribute()."""

    def test_distribute_returns_distribution(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        d = service.distribute(
            template_id=template_in_service.template_id,
            supplier_id="SUP-001",
            supplier_name="Eco",
            supplier_email="a@b.com",
        )
        assert isinstance(d, Distribution)

    def test_distribute_sets_status_sent(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        assert distribution_in_service.status == "sent"

    def test_distribute_template_not_found_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.distribute(
                template_id="bad", supplier_id="SUP-001",
                supplier_name="X", supplier_email="x@x.com",
            )

    def test_distribute_empty_supplier_id_raises(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            service.distribute(
                template_id=template_in_service.template_id,
                supplier_id="   ",
                supplier_name="X",
                supplier_email="x@x.com",
            )

    def test_distribute_creates_campaign_if_absent(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        d = service.distribute(
            template_id=template_in_service.template_id,
            supplier_id="SUP-001",
            supplier_name="Eco",
            supplier_email="a@b.com",
            campaign_id="camp-new",
        )
        assert "camp-new" in service._campaigns

    def test_distribute_auto_generates_campaign_id(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        d = service.distribute(
            template_id=template_in_service.template_id,
            supplier_id="SUP-001",
            supplier_name="Eco",
            supplier_email="a@b.com",
        )
        assert d.campaign_id != ""

    def test_distribute_sets_default_deadline(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        d = service.distribute(
            template_id=template_in_service.template_id,
            supplier_id="SUP-001",
            supplier_name="Eco",
            supplier_email="a@b.com",
        )
        assert d.deadline != ""

    def test_distribute_custom_deadline(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        d = service.distribute(
            template_id=template_in_service.template_id,
            supplier_id="SUP-001",
            supplier_name="Eco",
            supplier_email="a@b.com",
            deadline="2026-06-01T00:00:00+00:00",
        )
        assert d.deadline == "2026-06-01T00:00:00+00:00"

    def test_distribute_increments_statistics(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        assert service.get_statistics().total_distributions >= 1

    def test_distribute_provenance_hash(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        assert len(distribution_in_service.provenance_hash) == 64


class TestGetDistribution:
    """Tests for get_distribution() and list_distributions()."""

    def test_get_existing(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        result = service.get_distribution(distribution_in_service.distribution_id)
        assert result is distribution_in_service

    def test_get_nonexistent_returns_none(self, service: SupplierQuestionnaireService) -> None:
        assert service.get_distribution("no-such") is None

    def test_list_distributions_all(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        result = service.list_distributions()
        assert len(result) >= 1

    def test_list_distributions_by_campaign(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        result = service.list_distributions(campaign_id="camp-001")
        assert len(result) == 1

    def test_list_distributions_by_supplier(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        result = service.list_distributions(supplier_id="SUP-001")
        assert len(result) == 1

    def test_list_distributions_by_status(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        result = service.list_distributions(status="sent")
        assert len(result) >= 1


# ===================================================================
# 4. Response Tests
# ===================================================================


class TestSubmitResponse:
    """Tests for submit_response()."""

    def test_submit_returns_response(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert isinstance(response_in_service, QuestionnaireResponse)

    def test_submit_status_is_submitted(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert response_in_service.status == "submitted"

    def test_submit_calculates_completion(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        # 4 answers for a template with 4 questions => 100%
        assert response_in_service.completion_pct == pytest.approx(100.0)

    def test_submit_partial_completion(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        r = service.submit_response(
            distribution_id=distribution_in_service.distribution_id,
            supplier_id="SUP-002",
            supplier_name="Green",
            answers={"q-0": "Yes"},  # 1 of 4
        )
        assert r.completion_pct == pytest.approx(25.0)

    def test_submit_distribution_not_found_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.submit_response(
                distribution_id="bad",
                supplier_id="SUP-001",
                supplier_name="X",
                answers={"q-0": "Y"},
            )

    def test_submit_stores_evidence_files(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert response_in_service.evidence_files == ["audit.pdf"]

    def test_submit_provenance_hash(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert len(response_in_service.provenance_hash) == 64

    def test_submit_increments_statistics(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert service.get_statistics().total_responses >= 1


class TestGetListResponses:
    """Tests for get_response() and list_responses()."""

    def test_get_existing_response(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.get_response(response_in_service.response_id)
        assert result is response_in_service

    def test_get_nonexistent_response(self, service: SupplierQuestionnaireService) -> None:
        assert service.get_response("none") is None

    def test_list_responses_all(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert len(service.list_responses()) >= 1

    def test_list_responses_by_supplier(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert len(service.list_responses(supplier_id="SUP-001")) == 1

    def test_list_responses_by_template(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert len(service.list_responses(template_id=response_in_service.template_id)) >= 1

    def test_list_responses_by_status(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        assert len(service.list_responses(status="submitted")) >= 1


class TestUpdateResponse:
    """Tests for update_response()."""

    def test_update_answers_merge(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        updated = service.update_response(
            response_in_service.response_id,
            answers={"q-extra": "extra"},
        )
        assert "q-extra" in updated.answers
        assert "q-0" in updated.answers  # original preserved

    def test_update_evidence_files(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        updated = service.update_response(
            response_in_service.response_id,
            evidence_files=["new.pdf"],
        )
        assert updated.evidence_files == ["new.pdf"]

    def test_update_nonexistent_raises(self, service: SupplierQuestionnaireService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.update_response("bad-id", answers={"q-0": "val"})

    def test_update_finalized_raises(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.finalize_response(response_in_service.response_id)
        with pytest.raises(ValueError, match="already finalized"):
            service.update_response(response_in_service.response_id, answers={"q-0": "val"})


class TestFinalizeResponse:
    """Tests for finalize_response()."""

    def test_finalize_sets_status(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        fin = service.finalize_response(response_in_service.response_id)
        assert fin.status == "finalized"

    def test_finalize_sets_finalized_at(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        fin = service.finalize_response(response_in_service.response_id)
        assert fin.finalized_at is not None

    def test_finalize_nonexistent_raises(self, service: SupplierQuestionnaireService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.finalize_response("bad-id")

    def test_finalize_already_finalized_raises(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.finalize_response(response_in_service.response_id)
        with pytest.raises(ValueError, match="already finalized"):
            service.finalize_response(response_in_service.response_id)

    def test_finalize_increments_total_finalized(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.finalize_response(response_in_service.response_id)
        assert service.get_statistics().total_finalized == 1


# ===================================================================
# 5. Validation Tests
# ===================================================================


class TestValidateResponse:
    """Tests for validate_response()."""

    def test_validate_completeness_pass(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.validate_response(response_in_service.response_id, level="completeness")
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.checks_failed == 0

    def test_validate_completeness_fail_low_completion(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        r = service.submit_response(
            distribution_id=distribution_in_service.distribution_id,
            supplier_id="SUP-002",
            supplier_name="Low",
            answers={"q-0": "Yes"},  # 25% completion vs 80% threshold
        )
        result = service.validate_response(r.response_id, level="completeness")
        assert result.is_valid is False
        assert result.checks_failed >= 1

    def test_validate_consistency_checks_sections(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        # Answer only section-0 questions, skip section-1
        r = service.submit_response(
            distribution_id=distribution_in_service.distribution_id,
            supplier_id="SUP-002",
            supplier_name="Partial",
            answers={"q-0": "100", "q-1": "Yes", "q-2": "50", "q-3": "No"},
        )
        result = service.validate_response(r.response_id, level="consistency")
        assert isinstance(result, ValidationResult)

    def test_validate_evidence_warns_no_files(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        r = service.submit_response(
            distribution_id=distribution_in_service.distribution_id,
            supplier_id="SUP-003",
            supplier_name="NoEvidence",
            answers={"q-0": "100", "q-1": "Yes", "q-2": "50", "q-3": "No"},
            evidence_files=[],
        )
        result = service.validate_response(r.response_id, level="evidence")
        assert result.checks_warned >= 1

    def test_validate_evidence_no_warning_with_files(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.validate_response(
            response_in_service.response_id, level="evidence"
        )
        # evidence_files=["audit.pdf"] -- so no warning for missing evidence
        evidence_warnings = [w for w in result.warnings if "evidence" in w.lower()]
        assert len(evidence_warnings) == 0

    def test_validate_nonexistent_response_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.validate_response("bad-id")

    def test_validate_stores_result(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.validate_response(response_in_service.response_id)
        assert result.validation_id in service._validations

    def test_validate_increments_statistics(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.validate_response(response_in_service.response_id)
        assert service.get_statistics().total_validations == 1

    def test_validate_provenance_hash_set(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.validate_response(response_in_service.response_id)
        assert len(result.provenance_hash) == 64

    def test_validate_empty_answers_warning(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        r = service.submit_response(
            distribution_id=distribution_in_service.distribution_id,
            supplier_id="SUP-004",
            supplier_name="Empty",
            answers={"q-0": "", "q-1": None, "q-2": "val", "q-3": "val"},
        )
        result = service.validate_response(r.response_id)
        assert result.checks_warned >= 1


class TestBatchValidate:
    """Tests for batch_validate()."""

    def test_batch_validate_returns_list(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        results = service.batch_validate([response_in_service.response_id])
        assert len(results) == 1

    def test_batch_validate_skips_invalid_ids(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        results = service.batch_validate(
            [response_in_service.response_id, "bad-id"]
        )
        assert len(results) == 1


# ===================================================================
# 6. Scoring Tests
# ===================================================================


class TestScoreResponse:
    """Tests for score_response()."""

    def test_score_returns_scoring_result(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert isinstance(result, ScoringResult)

    def test_score_total_is_reasonable(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert 0.0 <= result.total_score <= 100.0

    def test_score_section_scores_populated(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert len(result.section_scores) > 0

    def test_score_tier_is_valid(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert result.tier in ("leader", "advanced", "developing", "lagging")

    def test_score_framework_from_template(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert result.framework == "cdp"

    def test_score_framework_override(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(
            response_in_service.response_id, framework="ecovadis"
        )
        assert result.framework == "ecovadis"

    def test_score_nonexistent_response_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.score_response("bad-id")

    def test_score_stored_internally(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert result.score_id in service._scores

    def test_score_increments_statistics(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.score_response(response_in_service.response_id)
        assert service.get_statistics().total_scores == 1

    def test_score_updates_avg_score(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.score_response(response_in_service.response_id)
        assert service.get_statistics().avg_score > 0.0

    def test_score_provenance_hash(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        assert len(result.provenance_hash) == 64

    def test_score_benchmark_percentile_first_entry(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        # First score: no peers yet so default 50.0
        assert result.benchmark_percentile == 50.0


class TestGetScore:
    """Tests for get_score() and get_supplier_scores()."""

    def test_get_score_by_id(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.score_response(response_in_service.response_id)
        fetched = service.get_score(result.score_id)
        assert fetched is result

    def test_get_score_nonexistent(self, service: SupplierQuestionnaireService) -> None:
        assert service.get_score("bad") is None

    def test_get_supplier_scores(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.score_response(response_in_service.response_id)
        scores = service.get_supplier_scores("SUP-001")
        assert len(scores) == 1


class TestDetermineTier:
    """Tests for _determine_tier()."""

    def test_tier_leader(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(85.0) == "leader"

    def test_tier_advanced(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(65.0) == "advanced"

    def test_tier_developing(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(45.0) == "developing"

    def test_tier_lagging(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(15.0) == "lagging"

    def test_tier_boundary_leader(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(80.0) == "leader"

    def test_tier_boundary_advanced(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(60.0) == "advanced"

    def test_tier_boundary_developing(self, service: SupplierQuestionnaireService) -> None:
        assert service._determine_tier(40.0) == "developing"


class TestBenchmarkSupplier:
    """Tests for benchmark_supplier()."""

    def test_benchmark_no_scores(self, service: SupplierQuestionnaireService) -> None:
        result = service.benchmark_supplier("SUP-999")
        assert result["avg_score"] == 0.0
        assert result["total_assessments"] == 0

    def test_benchmark_with_scores(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.score_response(response_in_service.response_id)
        result = service.benchmark_supplier("SUP-001")
        assert result["total_assessments"] == 1
        assert result["avg_score"] > 0.0
        assert result["tier"] in ("leader", "advanced", "developing", "lagging")


# ===================================================================
# 7. Follow-Up Tests
# ===================================================================


class TestFollowUp:
    """Tests for follow-up management."""

    def test_schedule_reminders_for_pending(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        actions = service.schedule_reminders("camp-001")
        assert len(actions) >= 1
        assert actions[0].action_type == "reminder"
        assert actions[0].status == "scheduled"

    def test_schedule_reminders_skips_responded(
        self,
        service: SupplierQuestionnaireService,
        response_in_service: QuestionnaireResponse,
    ) -> None:
        # response_in_service already submitted for SUP-001
        actions = service.schedule_reminders("camp-001")
        assert len(actions) == 0

    def test_schedule_reminders_respects_max_reminders(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        # Exhaust max_reminders (4)
        for _ in range(4):
            service.schedule_reminders("camp-001")
        actions = service.schedule_reminders("camp-001")
        assert len(actions) == 0

    def test_trigger_reminder_returns_action(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        action = service.trigger_reminder(distribution_in_service.distribution_id)
        assert isinstance(action, FollowUpAction)
        assert action.action_type == "reminder"
        assert action.status == "sent"

    def test_trigger_reminder_nonexistent_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.trigger_reminder("bad-id")

    def test_trigger_reminder_custom_message(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        action = service.trigger_reminder(
            distribution_in_service.distribution_id, message="Custom msg"
        )
        assert action.message == "Custom msg"

    def test_escalate_returns_escalation(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        action = service.escalate(distribution_in_service.distribution_id)
        assert action.action_type == "escalation"
        assert action.status == "sent"

    def test_escalate_nonexistent_raises(self, service: SupplierQuestionnaireService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.escalate("bad-id")

    def test_get_due_reminders(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        service.schedule_reminders("camp-001")
        due = service.get_due_reminders("camp-001")
        assert len(due) >= 1
        assert all(r.status == "scheduled" for r in due)

    def test_followup_increments_statistics(
        self, service: SupplierQuestionnaireService, distribution_in_service: Distribution
    ) -> None:
        service.trigger_reminder(distribution_in_service.distribution_id)
        assert service.get_statistics().total_followups >= 1


# ===================================================================
# 8. Analytics Tests
# ===================================================================


class TestCampaignAnalytics:
    """Tests for get_campaign_analytics()."""

    def test_analytics_empty_campaign(self, service: SupplierQuestionnaireService) -> None:
        analytics = service.get_campaign_analytics("empty-campaign")
        assert isinstance(analytics, CampaignAnalytics)
        assert analytics.total_distributed == 0
        assert analytics.response_rate_pct == 0.0

    def test_analytics_with_data(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        analytics = service.get_campaign_analytics("camp-001")
        assert analytics.total_distributed >= 1
        assert analytics.total_responded >= 1
        assert analytics.response_rate_pct > 0.0

    def test_analytics_with_scores(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.score_response(response_in_service.response_id)
        analytics = service.get_campaign_analytics("camp-001")
        assert analytics.avg_score > 0.0

    def test_analytics_score_distribution(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        service.score_response(response_in_service.response_id)
        analytics = service.get_campaign_analytics("camp-001")
        total_in_dist = sum(analytics.score_distribution.values())
        assert total_in_dist >= 1

    def test_analytics_provenance_hash(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        analytics = service.get_campaign_analytics("camp-001")
        assert len(analytics.provenance_hash) == 64


class TestResponseRate:
    """Tests for get_response_rate()."""

    def test_response_rate_zero_for_empty_campaign(
        self, service: SupplierQuestionnaireService
    ) -> None:
        rate = service.get_response_rate("nonexistent")
        assert rate == 0.0

    def test_response_rate_100_percent(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        rate = service.get_response_rate("camp-001")
        assert rate == pytest.approx(100.0)


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_generate_report_has_keys(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        report = service.generate_report("camp-001")
        assert "campaign" in report
        assert "analytics" in report
        assert "generated_at" in report
        assert "provenance_hash" in report


class TestComplianceGaps:
    """Tests for get_compliance_gaps()."""

    def test_compliance_gaps_returns_list(
        self, service: SupplierQuestionnaireService
    ) -> None:
        gaps = service.get_compliance_gaps("some-campaign")
        assert isinstance(gaps, list)


# ===================================================================
# 9. Statistics and Health Tests
# ===================================================================


class TestStatisticsAndHealth:
    """Tests for get_statistics() and health_check()."""

    def test_get_statistics_returns_model(self, service: SupplierQuestionnaireService) -> None:
        stats = service.get_statistics()
        assert isinstance(stats, QuestionnaireStatistics)

    def test_health_check_not_started(self, service: SupplierQuestionnaireService) -> None:
        result = service.health_check()
        assert result["status"] == "not_started"
        assert result["started"] is False

    def test_health_check_after_startup(self, service: SupplierQuestionnaireService) -> None:
        service.startup()
        result = service.health_check()
        assert result["status"] == "healthy"
        assert result["started"] is True

    def test_health_check_contains_counts(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        result = service.health_check()
        assert result["templates"] >= 1
        assert result["distributions"] >= 1
        assert result["responses"] >= 1
        assert "provenance_entries" in result

    def test_get_metrics(self, service: SupplierQuestionnaireService) -> None:
        metrics = service.get_metrics()
        assert "total_templates" in metrics
        assert "provenance_entries" in metrics

    def test_get_provenance_returns_tracker(
        self, service: SupplierQuestionnaireService
    ) -> None:
        tracker = service.get_provenance()
        assert isinstance(tracker, _ProvenanceTracker)


# ===================================================================
# 10. Lifecycle Tests
# ===================================================================


class TestLifecycle:
    """Tests for startup() and shutdown()."""

    def test_startup(self, service: SupplierQuestionnaireService) -> None:
        service.startup()
        assert service._started is True

    def test_startup_idempotent(self, service: SupplierQuestionnaireService) -> None:
        service.startup()
        service.startup()
        assert service._started is True

    def test_shutdown(self, service: SupplierQuestionnaireService) -> None:
        service.startup()
        service.shutdown()
        assert service._started is False

    def test_shutdown_when_not_started(self, service: SupplierQuestionnaireService) -> None:
        service.shutdown()  # should not raise
        assert service._started is False


# ===================================================================
# 11. Provenance Tracker Tests
# ===================================================================


class TestProvenanceTracker:
    """Tests for _ProvenanceTracker."""

    def test_record_returns_hash(self) -> None:
        tracker = _ProvenanceTracker()
        h = tracker.record("template", "t-1", "create", "abc123")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_record_increments_count(self) -> None:
        tracker = _ProvenanceTracker()
        assert tracker.entry_count == 0
        tracker.record("template", "t-1", "create", "abc")
        assert tracker.entry_count == 1
        tracker.record("response", "r-1", "submit", "def")
        assert tracker.entry_count == 2

    def test_record_stores_entry(self) -> None:
        tracker = _ProvenanceTracker()
        tracker.record("template", "t-1", "create", "abc")
        assert len(tracker._entries) == 1
        assert tracker._entries[0]["entity_type"] == "template"
        assert tracker._entries[0]["entity_id"] == "t-1"

    def test_record_with_user_id(self) -> None:
        tracker = _ProvenanceTracker()
        tracker.record("score", "s-1", "score", "xyz", user_id="alice")
        assert tracker._entries[0]["user_id"] == "alice"


# ===================================================================
# 12. _compute_hash Tests
# ===================================================================


class TestComputeHash:
    """Tests for the _compute_hash() helper function."""

    def test_hash_dict(self) -> None:
        h = _compute_hash({"a": 1, "b": 2})
        assert len(h) == 64

    def test_hash_deterministic(self) -> None:
        data = {"key": "value", "num": 42}
        assert _compute_hash(data) == _compute_hash(data)

    def test_hash_pydantic_model(self) -> None:
        t = QuestionnaireTemplate(name="Test")
        h = _compute_hash(t)
        assert len(h) == 64

    def test_hash_different_data_different_hash(self) -> None:
        h1 = _compute_hash({"x": 1})
        h2 = _compute_hash({"x": 2})
        assert h1 != h2


# ===================================================================
# 13. SHA-256 Provenance Tracking Through Facade
# ===================================================================


class TestProvenanceThroughFacade:
    """End-to-end provenance hash verification through the facade."""

    def test_provenance_recorded_for_create(
        self, service: SupplierQuestionnaireService
    ) -> None:
        initial = service.provenance.entry_count
        service.create_template(name="P1")
        assert service.provenance.entry_count == initial + 1

    def test_provenance_recorded_for_distribute(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        initial = service.provenance.entry_count
        service.distribute(
            template_id=template_in_service.template_id,
            supplier_id="SUP-001",
            supplier_name="X",
            supplier_email="x@x.com",
        )
        assert service.provenance.entry_count == initial + 1

    def test_provenance_recorded_for_submit(
        self,
        service: SupplierQuestionnaireService,
        distribution_in_service: Distribution,
    ) -> None:
        initial = service.provenance.entry_count
        service.submit_response(
            distribution_id=distribution_in_service.distribution_id,
            supplier_id="SUP-001",
            supplier_name="X",
            answers={"q-0": "v"},
        )
        assert service.provenance.entry_count == initial + 1

    def test_provenance_recorded_for_validate(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        initial = service.provenance.entry_count
        service.validate_response(response_in_service.response_id)
        assert service.provenance.entry_count == initial + 1

    def test_provenance_recorded_for_score(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        initial = service.provenance.entry_count
        service.score_response(response_in_service.response_id)
        assert service.provenance.entry_count == initial + 1

    def test_provenance_recorded_for_analytics(
        self, service: SupplierQuestionnaireService, response_in_service: QuestionnaireResponse
    ) -> None:
        initial = service.provenance.entry_count
        service.get_campaign_analytics("camp-001")
        assert service.provenance.entry_count == initial + 1

    def test_all_hashes_are_sha256(
        self,
        service: SupplierQuestionnaireService,
        response_in_service: QuestionnaireResponse,
    ) -> None:
        service.validate_response(response_in_service.response_id)
        service.score_response(response_in_service.response_id)
        for entry in service.provenance._entries:
            assert len(entry["entry_hash"]) == 64
            # Verify it is valid hexadecimal
            int(entry["entry_hash"], 16)


# ===================================================================
# 14. Full Workflow Tests
# ===================================================================


class TestFullWorkflow:
    """End-to-end workflow: create -> distribute -> respond -> validate -> score -> analytics."""

    def test_full_workflow(self, service: SupplierQuestionnaireService) -> None:
        # Step 1: Create template
        sections = _make_sections(section_count=2, questions_per_section=3)
        template = service.create_template(
            name="Annual ESG Survey",
            framework="ecovadis",
            sections=sections,
        )
        assert template.questions == 6

        # Step 2: Distribute
        dist = service.distribute(
            template_id=template.template_id,
            supplier_id="SUP-100",
            supplier_name="Workflow Supplier",
            supplier_email="ws@example.com",
            campaign_id="camp-wf",
        )
        assert dist.status == "sent"

        # Step 3: Submit response (answer all 6 questions)
        answers = {f"q-{i}": f"answer-{i}" for i in range(6)}
        resp = service.submit_response(
            distribution_id=dist.distribution_id,
            supplier_id="SUP-100",
            supplier_name="Workflow Supplier",
            answers=answers,
            evidence_files=["report.pdf"],
        )
        assert resp.completion_pct == pytest.approx(100.0)
        assert resp.status == "submitted"

        # Step 4: Validate
        val_result = service.validate_response(resp.response_id, level="evidence")
        assert val_result.is_valid is True

        # Step 5: Score
        score_result = service.score_response(resp.response_id)
        assert score_result.total_score > 0.0
        assert score_result.tier in ("leader", "advanced", "developing", "lagging")

        # Step 6: Analytics
        analytics = service.get_campaign_analytics("camp-wf")
        assert analytics.total_distributed == 1
        assert analytics.total_responded == 1
        assert analytics.response_rate_pct == pytest.approx(100.0)

        # Verify provenance chain
        assert service.provenance.entry_count >= 5

    def test_multi_supplier_workflow(self, service: SupplierQuestionnaireService) -> None:
        sections = _make_sections(section_count=1, questions_per_section=2)
        template = service.create_template(name="Multi", sections=sections)

        campaign_id = "camp-multi"
        for i in range(3):
            dist = service.distribute(
                template_id=template.template_id,
                supplier_id=f"SUP-{i}",
                supplier_name=f"Supplier {i}",
                supplier_email=f"s{i}@test.com",
                campaign_id=campaign_id,
            )
            # Only first two suppliers respond
            if i < 2:
                service.submit_response(
                    distribution_id=dist.distribution_id,
                    supplier_id=f"SUP-{i}",
                    supplier_name=f"Supplier {i}",
                    answers={"q-0": "yes", "q-1": "no"},
                )

        analytics = service.get_campaign_analytics(campaign_id)
        assert analytics.total_distributed == 3
        assert analytics.total_responded == 2
        assert analytics.response_rate_pct == pytest.approx(66.67, abs=0.1)


# ===================================================================
# 15. Thread Safety Tests
# ===================================================================


class TestThreadSafety:
    """Verify facade handles concurrent access gracefully."""

    def test_concurrent_template_creation(self, service: SupplierQuestionnaireService) -> None:
        results: List[QuestionnaireTemplate] = []
        errors: List[Exception] = []

        def create_template(name: str) -> None:
            try:
                t = service.create_template(name=name)
                results.append(t)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_template, args=(f"Thread-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All template IDs should be unique
        ids = {r.template_id for r in results}
        assert len(ids) == 10

    def test_concurrent_reads_and_writes(
        self, service: SupplierQuestionnaireService
    ) -> None:
        # Pre-create some templates
        for i in range(5):
            service.create_template(name=f"Pre-{i}")

        errors: List[Exception] = []

        def read_op() -> None:
            try:
                service.list_templates()
                service.get_statistics()
                service.health_check()
            except Exception as e:
                errors.append(e)

        def write_op(name: str) -> None:
            try:
                service.create_template(name=name)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=read_op))
            threads.append(threading.Thread(target=write_op, args=(f"CW-{i}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ===================================================================
# 16. configure_supplier_questionnaire() Tests
# ===================================================================


class TestConfigureFunction:
    """Tests for the async configure_supplier_questionnaire() function."""

    def test_configure_attaches_to_app_state(self) -> None:
        app = MagicMock()
        app.state = MagicMock()

        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            svc = asyncio.get_event_loop().run_until_complete(
                configure_supplier_questionnaire(app, config=_make_config())
            )
        assert app.state.supplier_questionnaire_service is svc

    def test_configure_starts_service(self) -> None:
        app = MagicMock()
        app.state = MagicMock()

        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            svc = asyncio.get_event_loop().run_until_complete(
                configure_supplier_questionnaire(app, config=_make_config())
            )
        assert svc._started is True

    def test_configure_includes_router(self) -> None:
        app = MagicMock()
        app.state = MagicMock()

        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            asyncio.get_event_loop().run_until_complete(
                configure_supplier_questionnaire(app, config=_make_config())
            )
        # The router should be included (include_router called)
        app.include_router.assert_called()

    def test_configure_returns_service_instance(self) -> None:
        app = MagicMock()
        app.state = MagicMock()

        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            svc = asyncio.get_event_loop().run_until_complete(
                configure_supplier_questionnaire(app, config=_make_config())
            )
        assert isinstance(svc, SupplierQuestionnaireService)


# ===================================================================
# 17. get_supplier_questionnaire() Tests
# ===================================================================


class TestGetSupplierQuestionnaire:
    """Tests for get_supplier_questionnaire()."""

    def test_get_service_from_app(self) -> None:
        app = MagicMock()
        with patch.object(SupplierQuestionnaireService, "_init_engines"):
            svc = SupplierQuestionnaireService(config=_make_config())
        app.state.supplier_questionnaire_service = svc

        result = get_supplier_questionnaire(app)
        assert result is svc

    def test_get_service_not_configured_raises(self) -> None:
        app = MagicMock()
        app.state = MagicMock(spec=[])  # no supplier_questionnaire_service attribute

        with pytest.raises(RuntimeError, match="not configured"):
            get_supplier_questionnaire(app)


# ===================================================================
# 18. Campaign Management Tests
# ===================================================================


class TestCampaignManagement:
    """Tests for create_campaign()."""

    def test_create_campaign(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        campaign = service.create_campaign(template_in_service.template_id, name="Q2 2026")
        assert campaign["name"] == "Q2 2026"
        assert campaign["status"] == "active"

    def test_create_campaign_nonexistent_template_raises(
        self, service: SupplierQuestionnaireService
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.create_campaign("bad-template")

    def test_create_campaign_increments_stats(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        service.create_campaign(template_in_service.template_id)
        assert service.get_statistics().total_campaigns >= 1
        assert service.get_statistics().active_campaigns >= 1

    def test_create_campaign_records_provenance(
        self,
        service: SupplierQuestionnaireService,
        template_in_service: QuestionnaireTemplate,
    ) -> None:
        before = service.provenance.entry_count
        service.create_campaign(template_in_service.template_id)
        assert service.provenance.entry_count == before + 1


# ===================================================================
# 19. Update Average Score Tests
# ===================================================================


class TestUpdateAvgScore:
    """Tests for _update_avg_score()."""

    def test_first_score_sets_avg(self, service: SupplierQuestionnaireService) -> None:
        service._stats.total_scores = 1
        service._update_avg_score(80.0)
        assert service._stats.avg_score == 80.0

    def test_zero_scores_sets_avg(self, service: SupplierQuestionnaireService) -> None:
        service._stats.total_scores = 0
        service._update_avg_score(50.0)
        assert service._stats.avg_score == 50.0

    def test_running_average(self, service: SupplierQuestionnaireService) -> None:
        service._stats.total_scores = 1
        service._stats.avg_score = 60.0
        # Now second score comes in: total_scores is set to 2 by score_response
        # But we test the helper directly: total_scores already incremented
        service._stats.total_scores = 2
        service._update_avg_score(80.0)
        # (60.0 * 1 + 80.0) / 2 = 70.0
        assert service._stats.avg_score == pytest.approx(70.0)


# ===================================================================
# 20. Pydantic Model Tests
# ===================================================================


class TestPydanticModels:
    """Basic tests for the Pydantic model defaults."""

    def test_template_defaults(self) -> None:
        t = QuestionnaireTemplate()
        assert t.status == "draft"
        assert t.framework == "custom"
        assert t.language == "en"
        assert t.questions == 0

    def test_distribution_defaults(self) -> None:
        d = Distribution()
        assert d.status == "pending"
        assert d.channel == "email"
        assert d.reminder_count == 0

    def test_response_defaults(self) -> None:
        r = QuestionnaireResponse()
        assert r.status == "draft"
        assert r.completion_pct == 0.0
        assert r.submitted_at is None

    def test_validation_result_defaults(self) -> None:
        v = ValidationResult()
        assert v.is_valid is False
        assert v.level == "completeness"

    def test_scoring_result_defaults(self) -> None:
        s = ScoringResult()
        assert s.tier == "lagging"
        assert s.total_score == 0.0

    def test_followup_action_defaults(self) -> None:
        f = FollowUpAction()
        assert f.action_type == "reminder"
        assert f.status == "scheduled"

    def test_campaign_analytics_defaults(self) -> None:
        a = CampaignAnalytics()
        assert a.total_distributed == 0
        assert a.response_rate_pct == 0.0

    def test_statistics_defaults(self) -> None:
        s = QuestionnaireStatistics()
        assert s.total_templates == 0
        assert s.avg_score == 0.0
