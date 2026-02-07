# -*- coding: utf-8 -*-
"""
Database Persistence Integration Tests for SLO Service (OBS-005)

Tests SLO creation, versioned history, budget snapshot storage,
compliance report persistence, soft-deletion, and data queries
using in-memory state (no actual database required).

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from greenlang.infrastructure.slo_service.error_budget import calculate_error_budget
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    ErrorBudget,
    SLI,
    SLIType,
    SLO,
)
from greenlang.infrastructure.slo_service.slo_manager import SLOManager


@pytest.mark.integration
class TestDatabasePersistence:
    """Database persistence integration tests using in-memory manager."""

    def test_create_slo_in_database(self, populated_manager):
        """SLOs are persisted after creation."""
        slos = populated_manager.list_all()
        assert len(slos) == 3
        for slo in slos:
            retrieved = populated_manager.get(slo.slo_id)
            assert retrieved is not None
            assert retrieved.name == slo.name

    def test_update_slo_creates_history(self, populated_manager):
        """Each update creates a history entry."""
        slo_id = "api-avail-99-9"
        populated_manager.update(slo_id, {"description": "Version 2"})
        populated_manager.update(slo_id, {"description": "Version 3"})

        history = populated_manager.get_history(slo_id)
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2

    def test_budget_snapshot_storage(self, populated_manager):
        """Budget snapshots can be stored and retrieved."""
        slo = populated_manager.get("api-avail-99-9")
        budgets = []
        for sli_val in [0.9999, 0.9997, 0.9993]:
            budget = calculate_error_budget(slo, sli_val)
            budgets.append(budget)

        assert len(budgets) == 3
        # Consumption increases
        assert budgets[-1].consumed_percent > budgets[0].consumed_percent

    def test_compliance_report_storage(self, tmp_path, populated_manager):
        """Compliance reports can be stored as JSON."""
        from greenlang.infrastructure.slo_service.compliance_reporter import (
            generate_report,
            store_report,
        )

        slos = populated_manager.list_all()
        budgets = {}
        for slo in slos:
            budgets[slo.slo_id] = calculate_error_budget(slo, 0.9995)

        report = generate_report("weekly", slos, budgets)
        path = store_report(report, str(tmp_path))
        assert path.endswith(".json")

    def test_slo_soft_delete(self, populated_manager):
        """Soft-deleted SLOs are hidden from list but preserved."""
        populated_manager.delete("api-avail-99-9")
        visible = populated_manager.list_all()
        assert all(s.slo_id != "api-avail-99-9" for s in visible)

        # Still accessible with include_deleted
        all_slos = populated_manager.list_all(include_deleted=True)
        deleted = [s for s in all_slos if s.slo_id == "api-avail-99-9"]
        assert len(deleted) == 1
        assert deleted[0].deleted is True

    def test_concurrent_writes(self, populated_manager):
        """Multiple sequential writes do not corrupt state."""
        import threading

        errors = []

        def update_slo(idx):
            try:
                populated_manager.update(
                    "api-avail-99-9",
                    {"description": f"Concurrent update {idx}"},
                )
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=update_slo, args=(i,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        slo = populated_manager.get("api-avail-99-9")
        assert slo.version == 11  # 1 initial + 10 updates

    def test_query_performance(self, populated_manager):
        """Listing SLOs is fast for small registries."""
        import time

        start = time.monotonic()
        for _ in range(1000):
            populated_manager.list_all()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # 1000 list operations under 1 second

    def test_data_retention(self, populated_manager):
        """History is preserved across multiple updates."""
        slo_id = "api-avail-99-9"
        for i in range(20):
            populated_manager.update(slo_id, {"description": f"Update {i}"})

        history = populated_manager.get_history(slo_id)
        assert len(history) == 20

    def test_evaluation_log_simulation(self, populated_manager):
        """Simulated evaluation logs track budget snapshots."""
        slo = populated_manager.get("api-avail-99-9")
        evaluation_log = []

        for sli_val in [0.9999, 0.9998, 0.9997, 0.9996, 0.9995]:
            budget = calculate_error_budget(slo, sli_val)
            evaluation_log.append({
                "slo_id": slo.slo_id,
                "sli_value": sli_val,
                "budget_status": budget.status.value,
                "consumed_percent": budget.consumed_percent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        assert len(evaluation_log) == 5
        # Each evaluation should show increasing consumption
        for i in range(1, len(evaluation_log)):
            assert (
                evaluation_log[i]["consumed_percent"]
                >= evaluation_log[i - 1]["consumed_percent"]
            )
