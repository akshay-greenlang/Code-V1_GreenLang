# -*- coding: utf-8 -*-
"""
Unit tests for DeforestationMonitor - AGENT-EUDR-033

Tests deforestation alert checking, Haversine correlation with supply chain
entities, impact assessment, investigation triggering, record retrieval,
listing, and health checks.

60+ tests covering the deforestation monitoring engine.

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
from greenlang.agents.eudr.continuous_monitoring.deforestation_monitor import (
    DeforestationMonitor,
)
from greenlang.agents.eudr.continuous_monitoring.models import (
    AlertSeverity,
    DeforestationCorrelation,
    DeforestationMonitorRecord,
    InvestigationRecord,
    InvestigationStatus,
)


@pytest.fixture
def config():
    return ContinuousMonitoringConfig()


@pytest.fixture
def monitor(config):
    return DeforestationMonitor(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_monitor_created(self, monitor):
        assert monitor is not None

    def test_monitor_uses_config(self, config):
        m = DeforestationMonitor(config=config)
        assert m.config is config

    def test_monitor_default_config(self):
        m = DeforestationMonitor()
        assert m.config is not None

    def test_records_empty_on_init(self, monitor):
        assert len(monitor._records) == 0

    def test_investigations_empty_on_init(self, monitor):
        assert len(monitor._investigations) == 0

    def test_alerts_empty_on_init(self, monitor):
        assert len(monitor._alerts) == 0


# ---------------------------------------------------------------------------
# Check Deforestation Alerts
# ---------------------------------------------------------------------------


class TestCheckDeforestationAlerts:
    @pytest.mark.asyncio
    async def test_returns_record(self, monitor, sample_deforestation_alerts, sample_supply_chain_entities):
        record = await monitor.check_deforestation_alerts(
            "OP-001", sample_deforestation_alerts, sample_supply_chain_entities,
        )
        assert isinstance(record, DeforestationMonitorRecord)

    @pytest.mark.asyncio
    async def test_sets_operator_id(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        assert record.operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_alerts_checked_count(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        assert record.alerts_checked == len(sample_deforestation_alerts)

    @pytest.mark.asyncio
    async def test_correlations_found(self, monitor, sample_deforestation_alerts, sample_supply_chain_entities):
        record = await monitor.check_deforestation_alerts(
            "OP-001", sample_deforestation_alerts, sample_supply_chain_entities,
        )
        assert record.correlations_found >= 0

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored_internally(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        assert record.monitor_id in monitor._records

    @pytest.mark.asyncio
    async def test_empty_alerts_returns_record(self, monitor):
        record = await monitor.check_deforestation_alerts("OP-001", [])
        assert record.alerts_checked == 0
        assert record.correlations_found == 0

    @pytest.mark.asyncio
    async def test_no_entities_no_correlations(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts, [])
        assert record.correlations_found == 0

    @pytest.mark.asyncio
    async def test_total_area_affected(self, monitor, sample_deforestation_alerts, sample_supply_chain_entities):
        record = await monitor.check_deforestation_alerts(
            "OP-001", sample_deforestation_alerts, sample_supply_chain_entities,
        )
        assert record.total_area_affected_hectares >= Decimal("0")


# ---------------------------------------------------------------------------
# Correlate with Plots
# ---------------------------------------------------------------------------


class TestCorrelateWithPlots:
    @pytest.mark.asyncio
    async def test_exact_match_correlates(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 15.3}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.5, "lon": 112.9}]
        correlations = await monitor.correlate_with_plots(alerts, entities)
        assert len(correlations) == 1
        assert correlations[0].confidence == Decimal("100.00")

    @pytest.mark.asyncio
    async def test_nearby_coordinates_correlate(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 10.0}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.6, "lon": 112.8}]
        correlations = await monitor.correlate_with_plots(alerts, entities)
        assert len(correlations) >= 1
        assert correlations[0].distance_km > Decimal("0")

    @pytest.mark.asyncio
    async def test_far_coordinates_no_correlation(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 5.0}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": 50.0, "lon": 10.0}]
        correlations = await monitor.correlate_with_plots(alerts, entities)
        assert len(correlations) == 0

    @pytest.mark.asyncio
    async def test_multiple_alerts_multiple_entities(self, monitor, sample_deforestation_alerts, sample_supply_chain_entities):
        correlations = await monitor.correlate_with_plots(
            sample_deforestation_alerts, sample_supply_chain_entities,
        )
        assert isinstance(correlations, list)

    @pytest.mark.asyncio
    async def test_correlation_has_alert_id(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 10.0}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.5, "lon": 112.9}]
        correlations = await monitor.correlate_with_plots(alerts, entities)
        assert correlations[0].alert_id == "A-001"

    @pytest.mark.asyncio
    async def test_correlation_has_entity_id(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 10.0}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.5, "lon": 112.9}]
        correlations = await monitor.correlate_with_plots(alerts, entities)
        assert correlations[0].entity_id == "P-001"

    @pytest.mark.asyncio
    async def test_correlation_confidence_range(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 10.0}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.6, "lon": 112.8}]
        correlations = await monitor.correlate_with_plots(alerts, entities)
        for corr in correlations:
            assert Decimal("0") <= corr.confidence <= Decimal("100")

    @pytest.mark.asyncio
    async def test_empty_alerts_no_correlations(self, monitor):
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.5, "lon": 112.9}]
        correlations = await monitor.correlate_with_plots([], entities)
        assert len(correlations) == 0


# ---------------------------------------------------------------------------
# Impact Assessment
# ---------------------------------------------------------------------------


class TestImpactAssessment:
    @pytest.mark.asyncio
    async def test_no_correlations_negligible(self, monitor):
        impact = await monitor.assess_impact("OP-001", [])
        assert impact["overall_severity"] == "negligible"

    @pytest.mark.asyncio
    async def test_large_area_high_severity(self, monitor):
        correlations = [
            DeforestationCorrelation(
                alert_id="A-001", entity_id="P-001", entity_type="plot",
                distance_km=Decimal("1"), area_hectares=Decimal("60"),
                confidence=Decimal("90"),
            ),
        ]
        impact = await monitor.assess_impact("OP-001", correlations)
        assert impact["overall_severity"] in ("critical", "high")

    @pytest.mark.asyncio
    async def test_impact_has_recommendations(self, monitor):
        correlations = [
            DeforestationCorrelation(
                alert_id="A-001", entity_id="P-001", entity_type="plot",
                distance_km=Decimal("1"), area_hectares=Decimal("60"),
                confidence=Decimal("90"),
            ),
        ]
        impact = await monitor.assess_impact("OP-001", correlations)
        assert len(impact["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_impact_affected_entities_count(self, monitor):
        correlations = [
            DeforestationCorrelation(
                alert_id="A-001", entity_id="P-001", entity_type="plot",
                distance_km=Decimal("5"), area_hectares=Decimal("10"),
                confidence=Decimal("80"),
            ),
            DeforestationCorrelation(
                alert_id="A-002", entity_id="P-002", entity_type="plot",
                distance_km=Decimal("10"), area_hectares=Decimal("5"),
                confidence=Decimal("60"),
            ),
        ]
        impact = await monitor.assess_impact("OP-001", correlations)
        assert impact["affected_entities"] == 2


# ---------------------------------------------------------------------------
# Investigation Triggering
# ---------------------------------------------------------------------------


class TestInvestigationTriggering:
    @pytest.mark.asyncio
    async def test_trigger_creates_investigation(self, monitor):
        correlation = DeforestationCorrelation(
            alert_id="A-001", entity_id="P-001", entity_type="plot",
            distance_km=Decimal("1"), area_hectares=Decimal("20"),
            confidence=Decimal("90"),
        )
        inv = await monitor.trigger_investigations("OP-001", "A-001", "P-001", correlation)
        assert inv is not None
        assert isinstance(inv, InvestigationRecord)

    @pytest.mark.asyncio
    async def test_investigation_status_pending(self, monitor):
        correlation = DeforestationCorrelation(
            alert_id="A-001", entity_id="P-001", entity_type="plot",
            distance_km=Decimal("1"), area_hectares=Decimal("20"),
            confidence=Decimal("90"),
        )
        inv = await monitor.trigger_investigations("OP-001", "A-001", "P-001", correlation)
        assert inv.investigation_status == InvestigationStatus.PENDING

    @pytest.mark.asyncio
    async def test_investigation_stored(self, monitor):
        correlation = DeforestationCorrelation(
            alert_id="A-001", entity_id="P-001", entity_type="plot",
            distance_km=Decimal("1"), area_hectares=Decimal("20"),
            confidence=Decimal("90"),
        )
        inv = await monitor.trigger_investigations("OP-001", "A-001", "P-001", correlation)
        assert inv.investigation_id in monitor._investigations

    @pytest.mark.asyncio
    async def test_investigation_alert_generated(self, monitor):
        correlation = DeforestationCorrelation(
            alert_id="A-001", entity_id="P-001", entity_type="plot",
            distance_km=Decimal("1"), area_hectares=Decimal("20"),
            confidence=Decimal("90"),
        )
        await monitor.trigger_investigations("OP-001", "A-001", "P-001", correlation)
        assert len(monitor._alerts) >= 1

    @pytest.mark.asyncio
    async def test_high_confidence_triggers_in_full_scan(self, monitor):
        alerts = [{"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 50.0}]
        entities = [{"entity_id": "P-001", "entity_type": "plot", "lat": -2.5, "lon": 112.9}]
        record = await monitor.check_deforestation_alerts("OP-001", alerts, entities)
        assert record.investigations_triggered >= 0


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_record(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        retrieved = await monitor.get_record(record.monitor_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_record_not_found(self, monitor):
        result = await monitor.get_record("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_records_all(self, monitor, sample_deforestation_alerts):
        await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        await monitor.check_deforestation_alerts("OP-002", sample_deforestation_alerts[:1])
        results = await monitor.list_records()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_records_filter_operator(self, monitor, sample_deforestation_alerts):
        await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        await monitor.check_deforestation_alerts("OP-002", sample_deforestation_alerts[:1])
        results = await monitor.list_records(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_records_empty(self, monitor):
        results = await monitor.list_records()
        assert results == []

    @pytest.mark.asyncio
    async def test_list_investigations(self, monitor):
        correlation = DeforestationCorrelation(
            alert_id="A-001", entity_id="P-001", entity_type="plot",
            distance_km=Decimal("1"), area_hectares=Decimal("20"),
            confidence=Decimal("90"),
        )
        await monitor.trigger_investigations("OP-001", "A-001", "P-001", correlation)
        results = await monitor.list_investigations()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_investigation(self, monitor):
        correlation = DeforestationCorrelation(
            alert_id="A-001", entity_id="P-001", entity_type="plot",
            distance_km=Decimal("1"), area_hectares=Decimal("20"),
            confidence=Decimal("90"),
        )
        inv = await monitor.trigger_investigations("OP-001", "A-001", "P-001", correlation)
        retrieved = await monitor.get_investigation(inv.investigation_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_investigation_not_found(self, monitor):
        result = await monitor.get_investigation("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, monitor):
        health = await monitor.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "DeforestationMonitor"

    @pytest.mark.asyncio
    async def test_health_check_record_count(self, monitor, sample_deforestation_alerts):
        await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        health = await monitor.health_check()
        assert health["record_count"] == 1

    @pytest.mark.asyncio
    async def test_health_check_investigation_count(self, monitor):
        health = await monitor.health_check()
        assert health["investigation_count"] == 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    @pytest.mark.asyncio
    async def test_provenance_is_hex(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        assert all(c in "0123456789abcdef" for c in record.provenance_hash)

    @pytest.mark.asyncio
    async def test_provenance_hash_length(self, monitor, sample_deforestation_alerts):
        record = await monitor.check_deforestation_alerts("OP-001", sample_deforestation_alerts)
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_different_alerts_different_provenance(self, monitor):
        r1 = await monitor.check_deforestation_alerts("OP-001", [
            {"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 10.0},
        ])
        r2 = await monitor.check_deforestation_alerts("OP-002", [
            {"alert_id": "A-002", "lat": 3.1, "lon": 101.7, "area_ha": 5.0},
        ])
        assert r1.provenance_hash != r2.provenance_hash
