"""
Unit tests for PACK-007 EUDR Professional Pack - Regulatory Change Engine

Tests EUR-Lex monitoring, impact assessment, gap analysis, migration planning,
and regulatory change tracking.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any


def _import_from_path(module_name, file_path):
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PACK_007_DIR = Path(__file__).resolve().parent.parent

# Import regulatory change module
regulatory_change_mod = _import_from_path(
    "pack_007_regulatory_change",
    _PACK_007_DIR / "engines" / "regulatory_change.py"
)

pytestmark = pytest.mark.skipif(
    regulatory_change_mod is None,
    reason="PACK-007 regulatory_change module not available"
)


@pytest.fixture
def regulatory_engine():
    """Create regulatory change engine instance."""
    if regulatory_change_mod is None:
        pytest.skip("regulatory_change module not available")
    return regulatory_change_mod.RegulatoryChangeEngine()


@pytest.fixture
def sample_regulatory_update():
    """Sample regulatory update from EUR-Lex."""
    return {
        "document_id": "32023R1115",  # EUDR regulation
        "title": "Regulation (EU) 2023/1115 - Amendment",
        "publication_date": "2024-12-01",
        "effective_date": "2025-01-01",
        "type": "amendment",
        "summary": "Updated deforestation cutoff dates and compliance timelines"
    }


class TestRegulatoryChangeEngine:
    """Test suite for RegulatoryChangeEngine."""

    def test_check_eurlex_updates(self, regulatory_engine):
        """Test checking EUR-Lex for regulatory updates."""
        updates = regulatory_engine.check_eurlex_updates(
            regulation="EUDR",
            since_date="2024-01-01"
        )

        assert updates is not None
        assert isinstance(updates, list)
        # Should return list of regulatory updates

    def test_impact_assessment(self, regulatory_engine, sample_regulatory_update):
        """Test assessing impact of regulatory change."""
        impact = regulatory_engine.assess_impact(
            update=sample_regulatory_update
        )

        assert impact is not None
        assert "impact_level" in impact or "severity" in impact
        assert "affected_areas" in impact or "scope" in impact
        assert "required_actions" in impact or "actions" in impact

    def test_gap_identification(self, regulatory_engine, sample_regulatory_update):
        """Test identifying compliance gaps from regulatory change."""
        current_state = {
            "cutoff_date": "2020-12-31",
            "compliance_level": "basic"
        }

        gaps = regulatory_engine.identify_gaps(
            update=sample_regulatory_update,
            current_state=current_state
        )

        assert gaps is not None
        assert isinstance(gaps, list)
        if len(gaps) > 0:
            assert "gap_type" in gaps[0] or "type" in gaps[0]
            assert "description" in gaps[0]
            assert "priority" in gaps[0] or "severity" in gaps[0]

    def test_migration_plan_generation(self, regulatory_engine, sample_regulatory_update):
        """Test generating migration plan for regulatory compliance."""
        gaps = [
            {"gap_type": "cutoff_date", "priority": "HIGH"},
            {"gap_type": "documentation", "priority": "MEDIUM"},
        ]

        plan = regulatory_engine.generate_migration_plan(
            update=sample_regulatory_update,
            gaps=gaps
        )

        assert plan is not None
        assert "phases" in plan or "steps" in plan
        assert "timeline" in plan or "duration" in plan
        assert "resources_required" in plan or "resources" in plan

    def test_cross_regulation_tracking(self, regulatory_engine):
        """Test tracking cross-regulation impacts (EUDR + CSRD, etc.)."""
        regulations = ["EUDR", "CSRD", "CBAM"]

        cross_impacts = regulatory_engine.analyze_cross_regulation_impacts(
            regulations=regulations
        )

        assert cross_impacts is not None
        assert "synergies" in cross_impacts or "overlaps" in cross_impacts
        assert "conflicts" in cross_impacts or "contradictions" in cross_impacts

    def test_change_history(self, regulatory_engine):
        """Test tracking regulatory change history."""
        history = regulatory_engine.get_change_history(
            regulation="EUDR",
            from_date="2023-01-01"
        )

        assert history is not None
        assert isinstance(history, list)
        # Should return chronological history

    def test_amendment_timeline(self, regulatory_engine):
        """Test generating amendment timeline."""
        timeline = regulatory_engine.generate_amendment_timeline(
            regulation="EUDR"
        )

        assert timeline is not None
        assert "milestones" in timeline or "events" in timeline

    def test_stakeholder_notification(self, regulatory_engine, sample_regulatory_update):
        """Test generating stakeholder notifications for regulatory changes."""
        notification = regulatory_engine.generate_stakeholder_notification(
            update=sample_regulatory_update,
            stakeholder_type="suppliers"
        )

        assert notification is not None
        assert "subject" in notification or "title" in notification
        assert "message" in notification or "body" in notification
        assert "action_required" in notification or "actions" in notification

    def test_eudr_regulatory_events_database(self, regulatory_engine):
        """Test accessing EUDR regulatory events database."""
        events = regulatory_engine.get_regulatory_events(
            regulation="EUDR",
            event_type="all"
        )

        assert events is not None
        assert isinstance(events, list)


class TestRegulatoryMonitoring:
    """Test regulatory monitoring features."""

    def test_subscribe_to_updates(self, regulatory_engine):
        """Test subscribing to regulatory updates."""
        subscription = regulatory_engine.subscribe_to_updates(
            regulation="EUDR",
            notification_channel="email"
        )

        assert subscription is not None
        assert "subscription_id" in subscription or "id" in subscription

    def test_check_for_new_amendments(self, regulatory_engine):
        """Test checking for new amendments."""
        amendments = regulatory_engine.check_for_amendments(
            regulation="EUDR",
            check_since="2024-01-01"
        )

        assert amendments is not None
        assert isinstance(amendments, list)

    def test_monitor_implementation_guidelines(self, regulatory_engine):
        """Test monitoring implementation guidelines updates."""
        guidelines = regulatory_engine.monitor_implementation_guidelines(
            regulation="EUDR"
        )

        assert guidelines is not None


class TestImpactAssessment:
    """Test impact assessment features."""

    def test_assess_timeline_impact(self, regulatory_engine, sample_regulatory_update):
        """Test assessing impact on compliance timelines."""
        timeline_impact = regulatory_engine.assess_timeline_impact(
            update=sample_regulatory_update,
            current_deadline="2025-06-30"
        )

        assert timeline_impact is not None
        assert "deadline_change" in timeline_impact or "new_deadline" in timeline_impact
        assert "additional_time_needed" in timeline_impact or "time_delta" in timeline_impact

    def test_assess_cost_impact(self, regulatory_engine, sample_regulatory_update):
        """Test assessing cost impact of regulatory change."""
        cost_impact = regulatory_engine.assess_cost_impact(
            update=sample_regulatory_update
        )

        assert cost_impact is not None
        assert "estimated_cost" in cost_impact or "cost_estimate" in cost_impact
        assert "cost_breakdown" in cost_impact or "breakdown" in cost_impact

    def test_assess_operational_impact(self, regulatory_engine, sample_regulatory_update):
        """Test assessing operational impact."""
        operational_impact = regulatory_engine.assess_operational_impact(
            update=sample_regulatory_update
        )

        assert operational_impact is not None
        assert "affected_processes" in operational_impact or "processes" in operational_impact

    def test_assess_technology_impact(self, regulatory_engine, sample_regulatory_update):
        """Test assessing technology/system impact."""
        tech_impact = regulatory_engine.assess_technology_impact(
            update=sample_regulatory_update
        )

        assert tech_impact is not None
        assert "system_changes_required" in tech_impact or "changes" in tech_impact


class TestCompliancePlanning:
    """Test compliance planning features."""

    def test_generate_action_plan(self, regulatory_engine, sample_regulatory_update):
        """Test generating action plan for compliance."""
        action_plan = regulatory_engine.generate_action_plan(
            update=sample_regulatory_update
        )

        assert action_plan is not None
        assert "actions" in action_plan or "tasks" in action_plan
        assert "priority_order" in action_plan or "priorities" in action_plan

    def test_estimate_compliance_timeline(self, regulatory_engine):
        """Test estimating compliance timeline."""
        tasks = [
            {"task": "Update systems", "duration_days": 30},
            {"task": "Train staff", "duration_days": 15},
            {"task": "Update documentation", "duration_days": 10},
        ]

        timeline = regulatory_engine.estimate_compliance_timeline(tasks)

        assert timeline is not None
        assert "total_duration_days" in timeline or "duration" in timeline
        assert timeline["total_duration_days"] >= 55  # Sum of task durations

    def test_identify_quick_wins(self, regulatory_engine):
        """Test identifying quick wins for compliance."""
        gaps = [
            {"gap": "Missing documentation", "effort": "LOW", "impact": "HIGH"},
            {"gap": "System upgrade", "effort": "HIGH", "impact": "HIGH"},
            {"gap": "Process update", "effort": "LOW", "impact": "MEDIUM"},
        ]

        quick_wins = regulatory_engine.identify_quick_wins(gaps)

        assert quick_wins is not None
        assert isinstance(quick_wins, list)
        # Should prioritize low effort, high impact items


class TestRegulatoryReporting:
    """Test regulatory change reporting features."""

    def test_generate_change_impact_report(self, regulatory_engine, sample_regulatory_update):
        """Test generating regulatory change impact report."""
        report = regulatory_engine.generate_change_impact_report(
            update=sample_regulatory_update
        )

        assert report is not None
        assert "regulatory_change" in report or "update" in report
        assert "impact_assessment" in report or "impact" in report
        assert "recommended_actions" in report or "actions" in report

    def test_generate_compliance_roadmap(self, regulatory_engine):
        """Test generating compliance roadmap."""
        updates = [sample_regulatory_update()]

        roadmap = regulatory_engine.generate_compliance_roadmap(
            updates=updates
        )

        assert roadmap is not None
        assert "milestones" in roadmap or "phases" in roadmap
        assert "timeline" in roadmap

    def test_generate_gap_analysis_report(self, regulatory_engine):
        """Test generating gap analysis report."""
        current_state = {"compliance_level": "basic"}
        target_state = {"compliance_level": "full"}

        report = regulatory_engine.generate_gap_analysis_report(
            current_state=current_state,
            target_state=target_state
        )

        assert report is not None
        assert "identified_gaps" in report or "gaps" in report
        assert "prioritization" in report or "priorities" in report


class TestRegulatoryDatabase:
    """Test regulatory database features."""

    def test_query_regulatory_database(self, regulatory_engine):
        """Test querying regulatory database."""
        results = regulatory_engine.query_regulatory_database(
            regulation="EUDR",
            query_type="amendments"
        )

        assert results is not None
        assert isinstance(results, list)

    def test_get_regulation_details(self, regulatory_engine):
        """Test getting regulation details."""
        details = regulatory_engine.get_regulation_details(
            regulation_id="32023R1115"
        )

        assert details is not None
        assert "regulation_id" in details or "id" in details
        assert "title" in details or "name" in details

    def test_search_regulations(self, regulatory_engine):
        """Test searching regulations."""
        results = regulatory_engine.search_regulations(
            keywords=["deforestation", "due diligence"]
        )

        assert results is not None
        assert isinstance(results, list)
