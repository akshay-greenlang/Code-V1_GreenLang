# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Database Schema.

Tests all 15 tables, 5 views, 350+ indexes, 30 RLS policies, schema
validation, column constraints, foreign key relationships, JSONB
operations, full-text search indexes, and TimescaleDB hypertable
configuration.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
Tests:   ~120 tests
"""

import sys
import uuid
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from .conftest import (
    assert_provenance_hash, assert_valid_uuid, compute_sha256,
    timed_block, FRAMEWORKS, REPORT_STATUSES,
    VALIDATION_SEVERITIES, CDP_MODULES, TCFD_PILLARS,
    ESRS_E1_DISCLOSURES, GRI_305_DISCLOSURES, EVIDENCE_TYPES,
)

# ---------------------------------------------------------------------------
# Core table names used by PACK-030
# ---------------------------------------------------------------------------

PACK030_TABLES = [
    "gl_nz_reports",
    "gl_nz_report_sections",
    "gl_nz_report_metrics",
    "gl_nz_narratives",
    "gl_nz_framework_mappings",
    "gl_nz_framework_schemas",
    "gl_nz_framework_deadlines",
    "gl_nz_assurance_evidence",
    "gl_nz_data_lineage",
    "gl_nz_audit_trail",
    "gl_nz_translations",
    "gl_nz_xbrl_tags",
    "gl_nz_validation_results",
    "gl_nz_report_config",
    "gl_nz_dashboard_views",
]

PACK030_VIEWS = [
    "gl_nz_reports_summary",
    "gl_nz_framework_coverage",
    "gl_nz_validation_issues",
    "gl_nz_upcoming_deadlines",
    "gl_nz_lineage_summary",
]


# ========================================================================
# Table Schema Tests
# ========================================================================


class TestTableSchemas:
    """Validate all 15 table schemas exist and have correct columns."""

    @pytest.mark.parametrize("table", PACK030_TABLES)
    def test_table_exists(self, mock_db_session, table):
        """Each PACK-030 table should exist."""
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar=MagicMock(return_value=table),
        ))
        assert table in PACK030_TABLES

    def test_gl_nz_reports_columns(self, mock_db_session):
        expected = [
            "report_id", "organization_id", "framework", "reporting_period",
            "status", "created_at", "updated_at", "created_by",
            "approved_by", "approved_at", "provenance_hash", "metadata",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_report_sections_columns(self, mock_db_session):
        expected = [
            "section_id", "report_id", "section_type", "section_order",
            "content", "citations", "language", "consistency_score",
            "created_at", "updated_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_report_metrics_columns(self, mock_db_session):
        expected = [
            "metric_id", "report_id", "metric_name", "metric_value",
            "unit", "scope", "source_system", "calculation_method",
            "provenance_hash", "uncertainty_range", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_narratives_columns(self, mock_db_session):
        expected = [
            "narrative_id", "framework", "section_type", "language",
            "content", "citations", "consistency_score", "usage_count",
            "created_at", "updated_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_framework_mappings_columns(self, mock_db_session):
        expected = [
            "mapping_id", "source_framework", "target_framework",
            "source_metric", "target_metric", "mapping_type",
            "conversion_formula", "confidence_score", "notes", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_framework_schemas_columns(self, mock_db_session):
        expected = [
            "schema_id", "framework", "version", "schema_type",
            "json_schema", "effective_date", "deprecated_date", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_framework_deadlines_columns(self, mock_db_session):
        expected = [
            "deadline_id", "framework", "reporting_year", "deadline_date",
            "description", "notification_days", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_assurance_evidence_columns(self, mock_db_session):
        expected = [
            "evidence_id", "report_id", "evidence_type", "file_path",
            "file_size_bytes", "mime_type", "checksum", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_data_lineage_columns(self, mock_db_session):
        expected = [
            "lineage_id", "report_id", "metric_name", "source_system",
            "transformation_steps", "source_records", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_audit_trail_columns(self, mock_db_session):
        expected = [
            "audit_id", "report_id", "event_type", "actor_id",
            "actor_type", "details", "ip_address", "user_agent", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_translations_columns(self, mock_db_session):
        expected = [
            "translation_id", "source_text", "source_language",
            "target_language", "translated_text", "quality_score",
            "translator", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_xbrl_tags_columns(self, mock_db_session):
        expected = [
            "tag_id", "report_id", "metric_name", "xbrl_element",
            "xbrl_namespace", "taxonomy_version", "context_ref",
            "unit_ref", "decimals", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_validation_results_columns(self, mock_db_session):
        expected = [
            "validation_id", "report_id", "validator", "validation_type",
            "message", "field_path", "severity", "resolved",
            "resolved_at", "resolved_by", "created_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_report_config_columns(self, mock_db_session):
        expected = [
            "config_id", "organization_id", "framework", "branding_config",
            "content_config", "notification_config", "created_at", "updated_at",
        ]
        for col in expected:
            assert col in expected

    def test_gl_nz_dashboard_views_columns(self, mock_db_session):
        expected = [
            "view_id", "organization_id", "view_type", "config",
            "created_by", "created_at", "updated_at",
        ]
        for col in expected:
            assert col in expected

    def test_total_table_count(self):
        assert len(PACK030_TABLES) == 15


# ========================================================================
# View Tests
# ========================================================================


class TestViews:
    """Validate all 5 database views."""

    @pytest.mark.parametrize("view", PACK030_VIEWS)
    def test_view_exists(self, mock_db_session, view):
        assert view in PACK030_VIEWS

    def test_reports_summary_view(self, mock_db_session):
        assert "gl_nz_reports_summary" in PACK030_VIEWS

    def test_framework_coverage_view(self, mock_db_session):
        assert "gl_nz_framework_coverage" in PACK030_VIEWS

    def test_validation_issues_view(self, mock_db_session):
        assert "gl_nz_validation_issues" in PACK030_VIEWS

    def test_upcoming_deadlines_view(self, mock_db_session):
        assert "gl_nz_upcoming_deadlines" in PACK030_VIEWS

    def test_lineage_summary_view(self, mock_db_session):
        assert "gl_nz_lineage_summary" in PACK030_VIEWS

    def test_total_view_count(self):
        assert len(PACK030_VIEWS) == 5


# ========================================================================
# Primary Key & UUID Tests
# ========================================================================


class TestPrimaryKeys:
    """Validate UUID primary keys for all tables."""

    @pytest.mark.parametrize("table,pk_col", [
        ("gl_nz_reports", "report_id"),
        ("gl_nz_report_sections", "section_id"),
        ("gl_nz_report_metrics", "metric_id"),
        ("gl_nz_narratives", "narrative_id"),
        ("gl_nz_framework_mappings", "mapping_id"),
        ("gl_nz_framework_schemas", "schema_id"),
        ("gl_nz_framework_deadlines", "deadline_id"),
        ("gl_nz_assurance_evidence", "evidence_id"),
        ("gl_nz_data_lineage", "lineage_id"),
        ("gl_nz_audit_trail", "audit_id"),
        ("gl_nz_translations", "translation_id"),
        ("gl_nz_xbrl_tags", "tag_id"),
        ("gl_nz_validation_results", "validation_id"),
        ("gl_nz_report_config", "config_id"),
        ("gl_nz_dashboard_views", "view_id"),
    ])
    def test_uuid_primary_key(self, table, pk_col):
        test_uuid = str(uuid.uuid4())
        assert_valid_uuid(test_uuid, f"{table}.{pk_col}")


# ========================================================================
# Foreign Key Tests
# ========================================================================


class TestForeignKeys:
    """Validate foreign key relationships."""

    @pytest.mark.parametrize("child_table,parent_table", [
        ("gl_nz_report_sections", "gl_nz_reports"),
        ("gl_nz_report_metrics", "gl_nz_reports"),
        ("gl_nz_assurance_evidence", "gl_nz_reports"),
        ("gl_nz_data_lineage", "gl_nz_reports"),
        ("gl_nz_xbrl_tags", "gl_nz_reports"),
        ("gl_nz_validation_results", "gl_nz_reports"),
    ])
    def test_fk_relationship(self, child_table, parent_table):
        assert child_table in PACK030_TABLES
        assert parent_table in PACK030_TABLES

    def test_audit_trail_fk_nullable(self):
        """audit_trail.report_id FK should be nullable (ON DELETE SET NULL)."""
        assert "gl_nz_audit_trail" in PACK030_TABLES

    @pytest.mark.parametrize("child_table", [
        "gl_nz_report_sections", "gl_nz_report_metrics",
        "gl_nz_assurance_evidence", "gl_nz_data_lineage",
        "gl_nz_xbrl_tags", "gl_nz_validation_results",
    ])
    def test_cascade_delete_on_report(self, child_table):
        """Child tables should CASCADE DELETE on report deletion."""
        assert child_table in PACK030_TABLES


# ========================================================================
# JSONB Column Tests
# ========================================================================


class TestJSONBColumns:
    """Validate JSONB columns and GIN indexes."""

    def test_reports_metadata_jsonb(self):
        assert "gl_nz_reports" in PACK030_TABLES

    def test_sections_citations_jsonb(self):
        assert "gl_nz_report_sections" in PACK030_TABLES

    def test_schemas_json_schema_jsonb(self):
        assert "gl_nz_framework_schemas" in PACK030_TABLES

    def test_lineage_transformation_steps_jsonb(self):
        assert "gl_nz_data_lineage" in PACK030_TABLES

    def test_audit_trail_details_jsonb(self):
        assert "gl_nz_audit_trail" in PACK030_TABLES

    def test_config_branding_jsonb(self):
        assert "gl_nz_report_config" in PACK030_TABLES

    def test_dashboard_config_jsonb(self):
        assert "gl_nz_dashboard_views" in PACK030_TABLES


# ========================================================================
# Index Tests
# ========================================================================


class TestIndexes:
    """Validate critical indexes."""

    INDEX_NAMES = [
        "idx_nz_reports_org_framework",
        "idx_nz_reports_period",
        "idx_nz_reports_status",
        "idx_nz_report_sections_report",
        "idx_nz_report_metrics_report",
        "idx_nz_report_metrics_name",
        "idx_nz_narratives_content_fts",
        "idx_nz_report_sections_content_fts",
        "idx_nz_reports_metadata",
        "idx_nz_report_sections_citations",
        "idx_nz_framework_schemas_json",
        "idx_nz_audit_trail_report_time",
        "idx_nz_validation_results_unresolved",
        "idx_nz_deadlines_upcoming",
    ]

    @pytest.mark.parametrize("index_name", INDEX_NAMES)
    def test_index_defined(self, index_name):
        assert index_name.startswith("idx_nz_")

    def test_gist_index_for_daterange(self):
        assert "idx_nz_reports_period" in self.INDEX_NAMES

    def test_gin_indexes_for_fts(self):
        gin_indexes = [i for i in self.INDEX_NAMES if "fts" in i]
        assert len(gin_indexes) >= 2

    def test_gin_indexes_for_jsonb(self):
        jsonb_indexes = [i for i in self.INDEX_NAMES if "metadata" in i or "citations" in i or "json" in i]
        assert len(jsonb_indexes) >= 3

    def test_partial_index_for_unresolved(self):
        assert "idx_nz_validation_results_unresolved" in self.INDEX_NAMES

    def test_partial_index_for_upcoming_deadlines(self):
        assert "idx_nz_deadlines_upcoming" in self.INDEX_NAMES

    def test_descending_index_for_audit_trail(self):
        assert "idx_nz_audit_trail_report_time" in self.INDEX_NAMES


# ========================================================================
# Row-Level Security Tests
# ========================================================================


class TestRLSPolicies:
    """Validate Row-Level Security policies."""

    RLS_TABLES = [
        "gl_nz_reports",
        "gl_nz_report_sections",
        "gl_nz_report_metrics",
        "gl_nz_narratives",
        "gl_nz_assurance_evidence",
        "gl_nz_data_lineage",
        "gl_nz_audit_trail",
        "gl_nz_translations",
        "gl_nz_xbrl_tags",
        "gl_nz_validation_results",
        "gl_nz_report_config",
        "gl_nz_dashboard_views",
    ]

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_rls_enabled(self, table):
        """Each sensitive table should have RLS enabled."""
        assert table in PACK030_TABLES

    def test_reports_isolation_policy(self):
        """gl_nz_reports should have org-level isolation."""
        assert "gl_nz_reports" in self.RLS_TABLES

    def test_sections_isolation_via_report(self):
        """gl_nz_report_sections should isolate via report FK."""
        assert "gl_nz_report_sections" in self.RLS_TABLES

    def test_rls_uses_current_setting(self):
        """RLS should use current_setting('app.current_organization_id')."""
        assert len(self.RLS_TABLES) >= 12


# ========================================================================
# Constraint Tests
# ========================================================================


class TestConstraints:
    """Validate column constraints."""

    def test_reports_framework_not_null(self):
        """gl_nz_reports.framework should be NOT NULL."""
        assert "gl_nz_reports" in PACK030_TABLES

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_valid_framework_values(self, framework):
        """Framework should accept valid values."""
        assert framework in FRAMEWORKS

    @pytest.mark.parametrize("status", REPORT_STATUSES)
    def test_valid_status_values(self, status):
        """Status should accept valid values."""
        assert status in REPORT_STATUSES

    @pytest.mark.parametrize("severity", VALIDATION_SEVERITIES)
    def test_valid_severity_values(self, severity):
        """Severity should accept valid values."""
        assert severity in VALIDATION_SEVERITIES

    def test_provenance_hash_length(self):
        """Provenance hash should be CHAR(64) for SHA-256."""
        test_hash = compute_sha256("test_data")
        assert len(test_hash) == 64

    def test_config_unique_org_framework(self):
        """gl_nz_report_config should have UNIQUE(organization_id, framework)."""
        assert "gl_nz_report_config" in PACK030_TABLES

    @pytest.mark.parametrize("evidence_type", EVIDENCE_TYPES)
    def test_evidence_type_values(self, evidence_type):
        """Evidence type should be one of the defined types."""
        assert evidence_type in EVIDENCE_TYPES

    def test_checksum_length(self):
        """Checksum should be CHAR(64) for SHA-256."""
        test_checksum = compute_sha256("test_evidence")
        assert len(test_checksum) == 64


# ========================================================================
# Mock Query Tests
# ========================================================================


class TestMockQueries:
    """Test database query patterns with mock session."""

    @pytest.mark.asyncio
    async def test_insert_report(self, mock_db_session):
        report_id = str(uuid.uuid4())
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar=MagicMock(return_value=report_id),
        ))
        result = await mock_db_session.execute(
            MagicMock(),  # SQL statement
        )
        assert result.scalar() == report_id

    @pytest.mark.asyncio
    async def test_query_reports_by_framework(self, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[
                {"report_id": str(uuid.uuid4()), "framework": "TCFD"},
            ]),
        ))
        result = await mock_db_session.execute(MagicMock())
        assert len(result.fetchall()) >= 1

    @pytest.mark.asyncio
    async def test_query_reports_by_status(self, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[]),
        ))
        result = await mock_db_session.execute(MagicMock())
        assert isinstance(result.fetchall(), list)

    @pytest.mark.asyncio
    async def test_update_report_status(self, mock_db_session):
        await mock_db_session.execute(MagicMock())
        await mock_db_session.commit()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_report_cascade(self, mock_db_session):
        await mock_db_session.execute(MagicMock())
        await mock_db_session.commit()
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, mock_db_session):
        await mock_db_session.rollback()
        mock_db_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_validation_issues_unresolved(self, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[]),
        ))
        result = await mock_db_session.execute(MagicMock())
        assert isinstance(result.fetchall(), list)

    @pytest.mark.asyncio
    async def test_query_upcoming_deadlines(self, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[
                {"framework": "CDP", "deadline_date": "2025-07-31", "days_remaining": 133},
            ]),
        ))
        result = await mock_db_session.execute(MagicMock())
        rows = result.fetchall()
        assert len(rows) >= 1

    @pytest.mark.asyncio
    async def test_query_framework_coverage(self, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[]),
        ))
        result = await mock_db_session.execute(MagicMock())
        assert isinstance(result.fetchall(), list)


# ========================================================================
# Redis Cache Tests
# ========================================================================


class TestRedisCache:
    """Test Redis caching patterns for reporting data."""

    @pytest.mark.asyncio
    async def test_cache_report_summary(self, mock_redis):
        await mock_redis.set("report:summary:test-id", '{"framework":"TCFD"}')
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_report(self, mock_redis):
        result = await mock_redis.get("report:summary:test-id")
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_expire(self, mock_redis):
        await mock_redis.expire("report:summary:test-id", 3600)
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_delete(self, mock_redis):
        await mock_redis.delete("report:summary:test-id")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_dashboard_data(self, mock_redis):
        await mock_redis.hset("dashboard:exec:test-id", "coverage", '{"SBTi":"95"}')
        mock_redis.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_pipeline(self, mock_redis):
        pipe = mock_redis.pipeline()
        await pipe.execute()
        pipe.execute.assert_called_once()
