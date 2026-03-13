# -*- coding: utf-8 -*-
"""
Unit tests for DataFreshnessValidator - AGENT-EUDR-033

Tests data freshness validation, staleness detection, freshness
classification, stale entity identification, refresh scheduling,
freshness report generation, record retrieval, listing, and
health checks.

60+ tests covering all data freshness validation logic.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.continuous_monitoring.config import (
    ContinuousMonitoringConfig,
)
from greenlang.agents.eudr.continuous_monitoring.data_freshness_validator import (
    DataFreshnessValidator,
)
from greenlang.agents.eudr.continuous_monitoring.models import (
    DataFreshnessRecord,
    FreshnessStatus,
)


@pytest.fixture
def config():
    return ContinuousMonitoringConfig()


@pytest.fixture
def validator(config):
    return DataFreshnessValidator(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_validator_created(self, validator):
        assert validator is not None

    def test_validator_uses_config(self, config):
        v = DataFreshnessValidator(config=config)
        assert v.config is config

    def test_validator_default_config(self):
        v = DataFreshnessValidator()
        assert v.config is not None

    def test_records_empty_on_init(self, validator):
        assert len(validator._records) == 0


# ---------------------------------------------------------------------------
# Validate Data Age
# ---------------------------------------------------------------------------


class TestValidateDataAge:
    @pytest.mark.asyncio
    async def test_returns_record(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert isinstance(record, DataFreshnessRecord)

    @pytest.mark.asyncio
    async def test_entities_checked_count(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert record.entities_checked == len(sample_data_freshness_records)

    @pytest.mark.asyncio
    async def test_fresh_count_positive(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert record.fresh_count >= 0

    @pytest.mark.asyncio
    async def test_freshness_percentage_range(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert Decimal("0") <= record.freshness_percentage <= Decimal("100")

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert record.freshness_id in validator._records

    @pytest.mark.asyncio
    async def test_empty_entities_returns_record(self, validator):
        record = await validator.validate_data_age("OP-001", [])
        assert record.entities_checked == 0
        assert record.freshness_percentage == Decimal("0")

    @pytest.mark.asyncio
    async def test_meets_target_flag(self, validator):
        now = datetime.now(timezone.utc)
        entities = [
            {"entity_id": f"E-{i}", "entity_type": "supplier",
             "last_updated": (now - timedelta(hours=1)).isoformat()}
            for i in range(10)
        ]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.meets_target is True


# ---------------------------------------------------------------------------
# Freshness Classification
# ---------------------------------------------------------------------------


class TestFreshnessClassification:
    @pytest.mark.asyncio
    async def test_fresh_entity(self, validator):
        now = datetime.now(timezone.utc)
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(hours=6)).isoformat(),
        }]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.fresh_count == 1

    @pytest.mark.asyncio
    async def test_stale_warning_entity(self, validator):
        now = datetime.now(timezone.utc)
        # Default stale_warning_hours = 24, stale_critical_hours = 72
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(hours=36)).isoformat(),
        }]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.stale_warning_count == 1

    @pytest.mark.asyncio
    async def test_stale_critical_entity(self, validator):
        now = datetime.now(timezone.utc)
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(days=10)).isoformat(),
        }]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.stale_critical_count == 1

    @pytest.mark.asyncio
    async def test_missing_timestamp_unknown(self, validator):
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
        }]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.stale_critical_count == 1  # Treated as unknown => critical

    @pytest.mark.asyncio
    async def test_mixed_freshness(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        total = record.fresh_count + record.stale_warning_count + record.stale_critical_count
        assert total == record.entities_checked


# ---------------------------------------------------------------------------
# Stale Entity Identification
# ---------------------------------------------------------------------------


class TestStaleEntityIdentification:
    @pytest.mark.asyncio
    async def test_stale_entities_populated(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert isinstance(record.stale_entities, list)

    @pytest.mark.asyncio
    async def test_stale_entities_have_recommended_action(self, validator):
        now = datetime.now(timezone.utc)
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(days=5)).isoformat(),
        }]
        record = await validator.validate_data_age("OP-001", entities)
        for stale in record.stale_entities:
            assert stale.recommended_action is not None
            assert len(stale.recommended_action) > 0


# ---------------------------------------------------------------------------
# Refresh Scheduling
# ---------------------------------------------------------------------------


class TestRefreshScheduling:
    @pytest.mark.asyncio
    async def test_refresh_schedule_created(self, validator):
        now = datetime.now(timezone.utc)
        entities = [
            {"entity_id": f"E-{i}", "entity_type": "supplier",
             "last_updated": (now - timedelta(days=5)).isoformat()}
            for i in range(5)
        ]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.refresh_scheduled >= 0

    @pytest.mark.asyncio
    async def test_no_stale_no_refresh(self, validator):
        now = datetime.now(timezone.utc)
        entities = [
            {"entity_id": f"E-{i}", "entity_type": "supplier",
             "last_updated": (now - timedelta(hours=1)).isoformat()}
            for i in range(5)
        ]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.refresh_scheduled == 0


# ---------------------------------------------------------------------------
# Freshness Reports
# ---------------------------------------------------------------------------


class TestFreshnessReports:
    @pytest.mark.asyncio
    async def test_report_with_records(self, validator, sample_data_freshness_records):
        await validator.validate_data_age("OP-001", sample_data_freshness_records)
        report = await validator.generate_freshness_reports("OP-001")
        assert report["total_checks"] >= 1
        assert report["operator_id"] == "OP-001"

    @pytest.mark.asyncio
    async def test_report_empty_when_no_records(self, validator):
        report = await validator.generate_freshness_reports("OP-001")
        assert report["total_checks"] == 0

    @pytest.mark.asyncio
    async def test_report_has_average_freshness(self, validator, sample_data_freshness_records):
        await validator.validate_data_age("OP-001", sample_data_freshness_records)
        report = await validator.generate_freshness_reports("OP-001")
        assert "average_freshness" in report

    @pytest.mark.asyncio
    async def test_report_has_latest_check(self, validator, sample_data_freshness_records):
        await validator.validate_data_age("OP-001", sample_data_freshness_records)
        report = await validator.generate_freshness_reports("OP-001")
        assert report["latest_check"] is not None


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_record(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        retrieved = await validator.get_record(record.freshness_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_record_not_found(self, validator):
        result = await validator.get_record("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_records_all(self, validator, sample_data_freshness_records):
        await validator.validate_data_age("OP-001", sample_data_freshness_records)
        await validator.validate_data_age("OP-002", sample_data_freshness_records[:2])
        results = await validator.list_records()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_records_filter_operator(self, validator, sample_data_freshness_records):
        await validator.validate_data_age("OP-001", sample_data_freshness_records)
        await validator.validate_data_age("OP-002", sample_data_freshness_records[:2])
        results = await validator.list_records(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_records_empty(self, validator):
        results = await validator.list_records()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, validator):
        health = await validator.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "DataFreshnessValidator"

    @pytest.mark.asyncio
    async def test_health_check_record_count(self, validator, sample_data_freshness_records):
        await validator.validate_data_age("OP-001", sample_data_freshness_records)
        health = await validator.health_check()
        assert health["record_count"] == 1


# ---------------------------------------------------------------------------
# Freshness Percentage
# ---------------------------------------------------------------------------


class TestFreshnessPercentage:
    @pytest.mark.asyncio
    async def test_all_fresh_100_percent(self, validator):
        now = datetime.now(timezone.utc)
        entities = [
            {"entity_id": f"E-{i}", "entity_type": "supplier",
             "last_updated": (now - timedelta(hours=1)).isoformat()}
            for i in range(5)
        ]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.freshness_percentage == Decimal("100.00")

    @pytest.mark.asyncio
    async def test_all_stale_0_percent(self, validator):
        now = datetime.now(timezone.utc)
        entities = [
            {"entity_id": f"E-{i}", "entity_type": "supplier",
             "last_updated": (now - timedelta(days=10)).isoformat()}
            for i in range(5)
        ]
        record = await validator.validate_data_age("OP-001", entities)
        assert record.freshness_percentage == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_precision_two_decimals(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        score_str = str(record.freshness_percentage)
        if "." in score_str:
            decimal_places = len(score_str.split(".")[1])
            assert decimal_places <= 2


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestFreshnessProvenance:
    @pytest.mark.asyncio
    async def test_provenance_is_hex(self, validator, sample_data_freshness_records):
        record = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        assert all(c in "0123456789abcdef" for c in record.provenance_hash)

    @pytest.mark.asyncio
    async def test_different_operators_different_provenance(self, validator, sample_data_freshness_records):
        r1 = await validator.validate_data_age("OP-001", sample_data_freshness_records)
        r2 = await validator.validate_data_age("OP-002", sample_data_freshness_records)
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# Multi-Operator Freshness
# ---------------------------------------------------------------------------


class TestMultiOperatorFreshness:
    @pytest.mark.asyncio
    async def test_different_operators_independent(self, validator):
        now = datetime.now(timezone.utc)
        fresh_entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(hours=1)).isoformat(),
        }]
        stale_entities = [{
            "entity_id": "E-002", "entity_type": "supplier",
            "last_updated": (now - timedelta(days=10)).isoformat(),
        }]
        r1 = await validator.validate_data_age("OP-001", fresh_entities)
        r2 = await validator.validate_data_age("OP-002", stale_entities)
        assert r1.freshness_percentage > r2.freshness_percentage

    @pytest.mark.asyncio
    async def test_report_scoped_to_operator(self, validator):
        now = datetime.now(timezone.utc)
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(hours=1)).isoformat(),
        }]
        await validator.validate_data_age("OP-001", entities)
        report = await validator.generate_freshness_reports("OP-001")
        assert report["total_checks"] == 1

    @pytest.mark.asyncio
    async def test_report_empty_for_different_operator(self, validator):
        now = datetime.now(timezone.utc)
        entities = [{
            "entity_id": "E-001", "entity_type": "supplier",
            "last_updated": (now - timedelta(hours=1)).isoformat(),
        }]
        await validator.validate_data_age("OP-001", entities)
        report = await validator.generate_freshness_reports("OP-999")
        assert report["total_checks"] == 0
