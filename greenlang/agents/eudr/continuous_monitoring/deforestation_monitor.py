# -*- coding: utf-8 -*-
"""
Deforestation Monitor Engine - AGENT-EUDR-033

Integrates with EUDR-020 (Deforestation Alert System) for real-time
deforestation alert correlation with supply chain entities, impact
assessment, and investigation triggering.

Zero-Hallucination:
    - All distance correlations use Haversine formula
    - Impact assessment uses deterministic area thresholds
    - Investigation triggers are rule-based, not ML-predicted

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ContinuousMonitoringConfig, get_config
from .models import (
    AGENT_ID,
    AlertSeverity,
    DeforestationCorrelation,
    DeforestationMonitorRecord,
    InvestigationRecord,
    InvestigationStatus,
    MonitoringAlert,
    MonitoringScope,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class DeforestationMonitor:
    """Deforestation monitoring and correlation engine.

    Checks deforestation alerts from upstream EUDR-020, correlates them
    with supply chain entities (plots, suppliers), assesses impact, and
    triggers investigations when thresholds are exceeded.

    Example:
        >>> monitor = DeforestationMonitor()
        >>> record = await monitor.check_deforestation_alerts(
        ...     operator_id="OP-001",
        ...     alerts=[{"alert_id": "A-001", "lat": -2.5, "lon": 28.3, "area_ha": 15.0}],
        ...     supply_chain_entities=[{"entity_id": "P-001", "lat": -2.5, "lon": 28.3}],
        ... )
        >>> assert record.correlations_found >= 0
    """

    def __init__(
        self, config: Optional[ContinuousMonitoringConfig] = None,
    ) -> None:
        """Initialize DeforestationMonitor engine."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._records: Dict[str, DeforestationMonitorRecord] = {}
        self._investigations: Dict[str, InvestigationRecord] = {}
        self._alerts: Dict[str, MonitoringAlert] = {}
        logger.info("DeforestationMonitor engine initialized")

    async def check_deforestation_alerts(
        self,
        operator_id: str,
        alerts: List[Dict[str, Any]],
        supply_chain_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> DeforestationMonitorRecord:
        """Check deforestation alerts and correlate with supply chain.

        Args:
            operator_id: Operator identifier.
            alerts: Deforestation alerts from EUDR-020.
            supply_chain_entities: Supply chain entities for correlation.

        Returns:
            DeforestationMonitorRecord with check results.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        monitor_id = str(uuid.uuid4())
        entities = supply_chain_entities or []

        # Correlate alerts with supply chain entities
        correlations = await self.correlate_with_plots(alerts, entities)

        # Assess impact
        impact = await self.assess_impact(operator_id, correlations)

        # Trigger investigations for high-impact correlations
        investigation_ids: List[str] = []
        for corr in correlations:
            if corr.confidence >= self.config.investigation_auto_trigger_threshold * 100:
                inv = await self.trigger_investigations(
                    operator_id, corr.alert_id, corr.entity_id, corr,
                )
                if inv:
                    investigation_ids.append(inv.investigation_id)

        total_area = sum(
            (c.area_hectares for c in correlations), Decimal("0"),
        )

        record = DeforestationMonitorRecord(
            monitor_id=monitor_id,
            operator_id=operator_id,
            alerts_checked=len(alerts),
            correlations_found=len(correlations),
            investigations_triggered=len(investigation_ids),
            total_area_affected_hectares=total_area,
            correlations=correlations,
            impact_assessment=impact,
            investigation_ids=investigation_ids,
            checked_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "monitor_id": monitor_id,
            "operator_id": operator_id,
            "alerts_checked": len(alerts),
            "correlations_found": len(correlations),
            "created_at": now.isoformat(),
        })

        self._provenance.record(
            entity_type="deforestation_monitor",
            action="check",
            entity_id=monitor_id,
            actor=AGENT_ID,
            metadata={
                "operator_id": operator_id,
                "alerts_checked": len(alerts),
                "correlations": len(correlations),
                "investigations": len(investigation_ids),
            },
        )

        self._records[monitor_id] = record
        elapsed = time.monotonic() - start_time
        logger.info(
            "Deforestation check %s: %d alerts, %d correlations, %d investigations (%.3fs)",
            monitor_id, len(alerts), len(correlations), len(investigation_ids), elapsed,
        )
        return record

    async def correlate_with_plots(
        self,
        alerts: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> List[DeforestationCorrelation]:
        """Correlate deforestation alerts with supply chain entities.

        Uses Haversine distance to match alerts with nearby plots
        and suppliers within the configured correlation window.

        Args:
            alerts: Deforestation alert data.
            entities: Supply chain entities with coordinates.

        Returns:
            List of found correlations.
        """
        correlations: List[DeforestationCorrelation] = []
        max_distance_km = Decimal("50.0")  # 50km correlation radius

        for alert in alerts:
            alert_id = alert.get("alert_id", str(uuid.uuid4()))
            alert_lat = float(alert.get("lat", 0))
            alert_lon = float(alert.get("lon", 0))
            area_ha = Decimal(str(alert.get("area_ha", 0)))

            for entity in entities:
                entity_id = entity.get("entity_id", "")
                entity_type = entity.get("entity_type", "plot")
                entity_lat = float(entity.get("lat", 0))
                entity_lon = float(entity.get("lon", 0))

                distance = self._haversine_km(
                    alert_lat, alert_lon, entity_lat, entity_lon,
                )
                distance_dec = Decimal(str(round(distance, 3)))

                if distance_dec <= max_distance_km:
                    # Confidence inversely proportional to distance
                    confidence = self._calculate_correlation_confidence(
                        distance_dec, max_distance_km,
                    )
                    correlations.append(DeforestationCorrelation(
                        alert_id=alert_id,
                        entity_id=entity_id,
                        entity_type=entity_type,
                        distance_km=distance_dec,
                        area_hectares=area_ha,
                        confidence=confidence,
                    ))

        return correlations

    async def assess_impact(
        self,
        operator_id: str,
        correlations: List[DeforestationCorrelation],
    ) -> Dict[str, Any]:
        """Assess the impact of correlated deforestation events.

        Args:
            operator_id: Operator identifier.
            correlations: List of deforestation correlations.

        Returns:
            Impact assessment dictionary.
        """
        if not correlations:
            return {
                "overall_severity": "negligible",
                "total_area_hectares": "0",
                "affected_entities": 0,
                "high_confidence_count": 0,
                "recommendations": [],
            }

        total_area = sum((c.area_hectares for c in correlations), Decimal("0"))
        severity = self.config.get_deforestation_severity(total_area)
        high_confidence = sum(1 for c in correlations if c.confidence >= Decimal("75"))

        # Unique affected entities
        affected_entities = len({c.entity_id for c in correlations})

        recommendations = []
        if severity in ("critical", "high"):
            recommendations.append({
                "action": "Immediately suspend sourcing from affected plots",
                "priority": "critical",
                "deadline_days": 1,
            })
            recommendations.append({
                "action": "Notify competent authority per EUDR Article 31",
                "priority": "critical",
                "deadline_days": 3,
            })
        if severity == "moderate":
            recommendations.append({
                "action": "Initiate enhanced due diligence for affected entities",
                "priority": "high",
                "deadline_days": 7,
            })

        return {
            "overall_severity": severity,
            "total_area_hectares": str(total_area),
            "affected_entities": affected_entities,
            "high_confidence_count": high_confidence,
            "recommendations": recommendations,
        }

    async def trigger_investigations(
        self,
        operator_id: str,
        alert_id: str,
        entity_id: str,
        correlation: DeforestationCorrelation,
    ) -> Optional[InvestigationRecord]:
        """Trigger an investigation for a high-confidence correlation.

        Args:
            operator_id: Operator identifier.
            alert_id: Deforestation alert ID.
            entity_id: Affected entity ID.
            correlation: The correlation that triggered investigation.

        Returns:
            InvestigationRecord if triggered, None otherwise.
        """
        investigation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        severity = self.config.get_deforestation_severity(correlation.area_hectares)

        investigation = InvestigationRecord(
            investigation_id=investigation_id,
            operator_id=operator_id,
            trigger_alert_id=alert_id,
            investigation_status=InvestigationStatus.PENDING,
            investigation_type="deforestation_correlation",
            findings=[],
            recommendations=[],
            created_at=now,
        )

        investigation.provenance_hash = self._provenance.compute_hash({
            "investigation_id": investigation_id,
            "alert_id": alert_id,
            "entity_id": entity_id,
            "created_at": now.isoformat(),
        })

        self._investigations[investigation_id] = investigation

        # Generate alert
        alert_severity = (
            AlertSeverity.CRITICAL if severity in ("critical", "high")
            else AlertSeverity.HIGH
        )
        alert = MonitoringAlert(
            alert_id=str(uuid.uuid4()),
            operator_id=operator_id,
            source_engine=MonitoringScope.DEFORESTATION,
            severity=alert_severity,
            title=f"Investigation triggered: deforestation near {entity_id}",
            description=(
                f"Deforestation alert {alert_id} correlated with entity "
                f"{entity_id} at distance {correlation.distance_km} km. "
                f"Affected area: {correlation.area_hectares} ha."
            ),
            entity_id=entity_id,
            entity_type=correlation.entity_type,
        )
        self._alerts[alert.alert_id] = alert

        logger.info(
            "Investigation %s triggered for alert %s / entity %s",
            investigation_id, alert_id, entity_id,
        )
        return investigation

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance between two coordinates in km."""
        r = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    @staticmethod
    def _calculate_correlation_confidence(
        distance_km: Decimal, max_distance_km: Decimal,
    ) -> Decimal:
        """Calculate correlation confidence inversely proportional to distance.

        Args:
            distance_km: Distance between alert and entity.
            max_distance_km: Maximum correlation distance.

        Returns:
            Confidence score 0-100.
        """
        if max_distance_km <= 0:
            return Decimal("0")
        ratio = distance_km / max_distance_km
        confidence = (Decimal("1") - ratio) * Decimal("100")
        return max(Decimal("0"), min(Decimal("100"), confidence.quantize(Decimal("0.01"))))

    async def get_record(self, monitor_id: str) -> Optional[DeforestationMonitorRecord]:
        """Retrieve a deforestation monitor record by ID."""
        return self._records.get(monitor_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
    ) -> List[DeforestationMonitorRecord]:
        """List deforestation monitor records with optional filters."""
        results = list(self._records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        return results

    async def get_investigation(self, investigation_id: str) -> Optional[InvestigationRecord]:
        """Retrieve an investigation record by ID."""
        return self._investigations.get(investigation_id)

    async def list_investigations(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[InvestigationRecord]:
        """List investigation records with optional filters."""
        results = list(self._investigations.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if status:
            results = [r for r in results if r.investigation_status.value == status]
        return results

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "DeforestationMonitor",
            "status": "healthy",
            "record_count": len(self._records),
            "investigation_count": len(self._investigations),
        }
