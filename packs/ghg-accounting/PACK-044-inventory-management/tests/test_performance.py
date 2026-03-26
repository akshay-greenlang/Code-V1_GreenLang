# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Performance Tests
==========================================

Tests engine performance: creation speed, transition throughput,
batch processing, and memory-conscious operation.

Target: 15+ test cases.
"""

import time
from datetime import date
from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_period_mod = _load_engine("inventory_period")
_dc_mod = _load_engine("data_collection")
_qm_mod = _load_engine("quality_management")
_cm_mod = _load_engine("change_management")
_iv_mod = _load_engine("inventory_versioning")

InventoryPeriodEngine = _period_mod.InventoryPeriodEngine
PeriodStatus = _period_mod.PeriodStatus
MilestoneStatus = _period_mod.MilestoneStatus

DataCollectionEngine = _dc_mod.DataCollectionEngine
DataScope = _dc_mod.DataScope

QualityManagementEngine = _qm_mod.QualityManagementEngine

ChangeManagementEngine = _cm_mod.ChangeManagementEngine
ChangeRequest = _cm_mod.ChangeRequest
AffectedSource = _cm_mod.AffectedSource
ChangeCategory = _cm_mod.ChangeCategory

InventoryVersioningEngine = _iv_mod.InventoryVersioningEngine


# ===================================================================
# Period Engine Performance
# ===================================================================


class TestPeriodEnginePerformance:
    """Performance tests for InventoryPeriodEngine."""

    def test_create_100_periods_under_2s(self):
        engine = InventoryPeriodEngine()
        t0 = time.perf_counter()
        for i in range(100):
            engine.create_period(
                organisation_id=f"org-{i % 10}",
                period_name=f"FY{2000 + i}",
                start_date=date(2000 + i, 1, 1),
                end_date=date(2000 + i, 12, 31),
            )
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"100 period creations took {elapsed:.2f}s"

    def test_transition_throughput(self):
        engine = InventoryPeriodEngine()
        periods = []
        for i in range(20):
            r = engine.create_period(
                "org-001", f"P{i}",
                date(2000 + i, 1, 1), date(2000 + i, 12, 31),
            )
            pid = r.period.period_id
            p = engine.get_period(pid)
            for ms in p.milestones:
                if ms.phase == "planning":
                    engine.update_milestone(
                        pid, ms.milestone_id,
                        status=MilestoneStatus.COMPLETED,
                    )
                    break
            periods.append(pid)

        t0 = time.perf_counter()
        for pid in periods:
            engine.transition(pid, PeriodStatus.DATA_COLLECTION)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"20 transitions took {elapsed:.2f}s"

    def test_list_periods_performance(self):
        engine = InventoryPeriodEngine()
        for i in range(50):
            engine.create_period(
                "org-001", f"P{i}",
                date(2000 + i, 1, 1), date(2000 + i, 12, 31),
            )
        t0 = time.perf_counter()
        result = engine.list_periods()
        elapsed = time.perf_counter() - t0
        assert len(result) == 50
        assert elapsed < 0.5


# ===================================================================
# Data Collection Engine Performance
# ===================================================================


class TestDataCollectionPerformance:
    """Performance tests for DataCollectionEngine."""

    def test_create_campaign_with_50_requests(self):
        engine = DataCollectionEngine()
        r = engine.create_campaign(
            period_id="per-001",
            organisation_id="org-001",
            campaign_name="Large Campaign",
        )
        cid = r.campaign.campaign_id
        t0 = time.perf_counter()
        for i in range(50):
            engine.add_request(
                campaign_id=cid,
                scope=DataScope.SCOPE_1,
                category=f"category_{i}",
                facility_id=f"FAC-{i:03d}",
            )
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"50 request additions took {elapsed:.2f}s"


# ===================================================================
# Quality Engine Performance
# ===================================================================


class TestQualityEnginePerformance:
    """Performance tests for QualityManagementEngine."""

    def test_run_checks_under_100ms(self):
        engine = QualityManagementEngine()
        inputs = {f"COMP-{i:03d}": True for i in range(1, 6)}
        inputs.update({f"CONS-{i:03d}": True for i in range(1, 5)})
        inputs.update({f"ACCU-{i:03d}": True for i in range(1, 6)})
        inputs.update({f"TRAN-{i:03d}": True for i in range(1, 6)})
        t0 = time.perf_counter()
        engine.run_checks("per-001", "org-001", inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Quality checks took {elapsed_ms:.1f}ms"


# ===================================================================
# Change Management Performance
# ===================================================================


class TestChangeManagementPerformance:
    """Performance tests for ChangeManagementEngine."""

    def test_batch_assess_20_changes(self):
        engine = ChangeManagementEngine()
        requests = []
        for i in range(20):
            requests.append(ChangeRequest(
                title=f"Change {i}",
                category=ChangeCategory.METHODOLOGICAL,
                total_inventory_tco2e=Decimal("55000"),
                affected_sources=[
                    AffectedSource(
                        source_id=f"SRC-{i}",
                        scope="scope1",
                        old_value_tco2e=Decimal("500"),
                        new_value_tco2e=Decimal("490"),
                        delta_tco2e=Decimal("-10"),
                    ),
                ],
            ))
        t0 = time.perf_counter()
        results = engine.batch_assess(requests)
        elapsed = time.perf_counter() - t0
        assert len(results) == 20
        assert elapsed < 2.0, f"20 assessments took {elapsed:.2f}s"


# ===================================================================
# Versioning Performance
# ===================================================================


class TestVersioningPerformance:
    """Performance tests for InventoryVersioningEngine."""

    def test_create_20_version_chain(self):
        engine = InventoryVersioningEngine()
        data = {"scope1_total": 10000, "scope2_total": 5000}
        r = engine.create_version(
            inventory_id="inv-001", reporting_year=2025,
            data=data, created_by="user",
        )
        t0 = time.perf_counter()
        prev = r.version
        for i in range(19):
            new_data = dict(data)
            new_data["scope1_total"] = 10000 - (i * 100)
            r = engine.create_next_version(prev, new_data)
            prev = r.version
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"19 version creations took {elapsed:.2f}s"

    def test_diff_computation_speed(self):
        engine = InventoryVersioningEngine()
        data_v1 = {f"field_{i}": i * 100 for i in range(50)}
        data_v2 = {f"field_{i}": i * 100 + 10 for i in range(50)}
        r1 = engine.create_version("inv-001", 2025, data_v1, "user")
        r2 = engine.create_next_version(r1.version, data_v2)
        t0 = time.perf_counter()
        diff = engine.compute_diff(r1.version, r2.version)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"Diff on 50 fields took {elapsed_ms:.1f}ms"


# ===================================================================
# Processing Time Validation
# ===================================================================


class TestProcessingTimeField:
    """Tests that processing_time_ms is populated on results."""

    def test_period_creation_time(self):
        engine = InventoryPeriodEngine()
        r = engine.create_period("o", "P1", date(2025, 1, 1), date(2025, 12, 31))
        assert r.processing_time_ms > Decimal("0")

    def test_version_creation_time(self):
        engine = InventoryVersioningEngine()
        r = engine.create_version("inv-001", 2025, {"a": 1}, "user")
        assert r.processing_time_ms > Decimal("0")
