# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for SLO Service (OBS-005)

Tests full SLO lifecycle, YAML import to evaluation, budget tracking,
burn rate alerting, compliance reporting, and rule/dashboard generation
with mocked Prometheus and Redis.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.slo_service.alerting_bridge import AlertingBridge
from greenlang.infrastructure.slo_service.burn_rate import (
    evaluate_burn_rate_windows,
    generate_all_burn_rate_rules,
)
from greenlang.infrastructure.slo_service.compliance_reporter import (
    generate_report,
)
from greenlang.infrastructure.slo_service.error_budget import (
    calculate_error_budget,
    check_budget_policy,
)
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    ErrorBudget,
    SLI,
    SLIType,
    SLO,
    SLOWindow,
)
from greenlang.infrastructure.slo_service.recording_rules import (
    generate_all_recording_rules,
    write_recording_rules_file,
)
from greenlang.infrastructure.slo_service.alert_rules import (
    generate_all_alert_rules,
    write_alert_rules_file,
)
from greenlang.infrastructure.slo_service.dashboard_generator import (
    write_dashboards,
)
from greenlang.infrastructure.slo_service.slo_manager import SLOManager


@pytest.mark.integration
class TestFullSLOLifecycle:
    """Full SLO lifecycle tests."""

    def test_full_slo_lifecycle(self):
        """Create, update, evaluate, and delete an SLO."""
        mgr = SLOManager()

        # Create
        sli = SLI(
            name="test_avail",
            sli_type=SLIType.AVAILABILITY,
            good_query="good",
            total_query="total",
        )
        slo = SLO(
            slo_id="lifecycle-test",
            name="Lifecycle Test SLO",
            service="test-service",
            sli=sli,
            target=99.9,
        )
        created = mgr.create(slo)
        assert created.version == 1

        # Update
        updated = mgr.update("lifecycle-test", {"description": "Updated"})
        assert updated.version == 2
        assert updated.description == "Updated"

        # Evaluate budget
        budget = calculate_error_budget(updated, current_sli=0.9995)
        assert budget.status in (BudgetStatus.HEALTHY, BudgetStatus.WARNING,
                                  BudgetStatus.CRITICAL, BudgetStatus.EXHAUSTED)

        # Delete
        assert mgr.delete("lifecycle-test") is True
        assert mgr.get("lifecycle-test") is None

    def test_yaml_import_to_evaluation(self, tmp_path):
        """Import SLOs from YAML and evaluate their budgets."""
        import yaml

        yaml_data = {
            "slos": [
                {
                    "slo_id": "yaml-import-1",
                    "name": "YAML Import SLO 1",
                    "service": "test-svc",
                    "target": 99.9,
                    "sli": {
                        "name": "avail",
                        "sli_type": "availability",
                        "good_query": "good",
                        "total_query": "total",
                    },
                },
            ]
        }

        yaml_file = tmp_path / "slos.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        mgr = SLOManager()
        loaded = mgr.load_from_yaml(str(yaml_file))
        assert len(loaded) == 1

        budget = calculate_error_budget(loaded[0], current_sli=0.9998)
        assert budget.slo_id == "yaml-import-1"
        assert budget.remaining_percent > 0

    def test_budget_tracking_over_time(self, populated_manager):
        """Budget changes over multiple SLI evaluations."""
        slo = populated_manager.get("api-avail-99-9")
        assert slo is not None

        sli_values = [0.9999, 0.9997, 0.9993, 0.9990]
        budgets = []
        for sli_val in sli_values:
            budget = calculate_error_budget(slo, sli_val)
            budgets.append(budget)

        # Budget should be increasingly consumed
        for i in range(1, len(budgets)):
            assert budgets[i].consumed_percent >= budgets[i - 1].consumed_percent

    def test_burn_rate_alert_trigger(self, populated_manager):
        """Fast burn rate fires an alert."""
        slo = populated_manager.get("api-avail-99-9")
        burn_rates = {
            "fast": {"long": 16.0, "short": 17.0},
            "medium": {"long": 3.0, "short": 2.0},
            "slow": {"long": 0.5, "short": 0.3},
        }
        alerts = evaluate_burn_rate_windows(slo, burn_rates)
        assert len(alerts) >= 1
        assert alerts[0].severity == "critical"

    def test_compliance_report_generation(self, populated_manager):
        """Generate a compliance report from populated SLOs."""
        slos = populated_manager.list_all()
        budgets = {}
        for slo in slos:
            budget = calculate_error_budget(slo, current_sli=0.9995)
            budgets[slo.slo_id] = budget

        report = generate_report("weekly", slos, budgets)
        assert report.total_slos == len(slos)
        assert len(report.entries) == len(slos)

    def test_recording_rule_generation_and_validation(self, tmp_path, populated_manager):
        """Generate and validate recording rules."""
        slos = populated_manager.list_all()
        output = str(tmp_path / "recording_rules.yaml")
        path = write_recording_rules_file(slos, output)
        assert Path(path).exists()

        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data["groups"]) == 3

    def test_alert_rule_generation_and_validation(self, tmp_path, populated_manager):
        """Generate and validate alert rules."""
        slos = populated_manager.list_all()
        output = str(tmp_path / "alert_rules.yaml")
        path = write_alert_rules_file(slos, output)
        assert Path(path).exists()

    def test_dashboard_generation(self, tmp_path, populated_manager):
        """Generate Grafana dashboards."""
        slos = populated_manager.list_all()
        paths = write_dashboards(slos, str(tmp_path))
        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()

    def test_slo_version_history(self, populated_manager):
        """SLO updates create version history entries."""
        populated_manager.update("api-avail-99-9", {"description": "v2"})
        populated_manager.update("api-avail-99-9", {"description": "v3"})

        history = populated_manager.get_history("api-avail-99-9")
        assert len(history) == 2

    def test_budget_policy_enforcement(self, populated_manager):
        """Budget policy produces correct actions."""
        slo = populated_manager.get("api-avail-99-9")
        budget = calculate_error_budget(slo, current_sli=0.998)
        result = check_budget_policy(budget, "freeze_deployments")
        assert result["action_required"] is True

    def test_alerting_bridge_integration(self, populated_manager):
        """Alerting bridge dispatches alerts for burn rates."""
        slo = populated_manager.get("api-avail-99-9")
        bridge = AlertingBridge(enabled=True)

        from greenlang.infrastructure.slo_service.models import BurnRateAlert
        alert = BurnRateAlert(
            slo_id=slo.slo_id,
            slo_name=slo.name,
            burn_window="fast",
            burn_rate_long=16.0,
            burn_rate_short=17.0,
            threshold=14.4,
            severity="critical",
            service=slo.service,
            message="Integration test alert",
        )
        result = bridge.fire_burn_rate_alert(alert, slo)
        assert result["dispatched"] is True

    def test_multi_slo_evaluation(self, populated_manager):
        """Multiple SLOs can be evaluated and budgets calculated."""
        slos = populated_manager.list_all()
        results = []
        for slo in slos:
            budget = calculate_error_budget(slo, current_sli=0.9995)
            results.append({
                "slo_id": slo.slo_id,
                "budget_status": budget.status.value,
                "consumed_percent": budget.consumed_percent,
            })

        assert len(results) == 3
        for r in results:
            assert "budget_status" in r
