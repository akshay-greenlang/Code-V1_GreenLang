# -*- coding: utf-8 -*-
"""
Continuous Monitoring Workflow
================================

Four-phase continuous monitoring cycle for real-time EUDR compliance tracking.

This workflow implements 24/7 automated monitoring of:
- Deforestation alert signals (satellite imagery, news, NGO reports)
- Supplier performance changes (certifications, audits, violations)
- Regulatory updates (EUDR guidance, country benchmarking changes)
- Supply chain disruptions (geopolitical events, natural disasters)

Phases:
    1. Signal Collection - Aggregate signals from multiple data sources
    2. Event Detection - Identify significant events requiring attention
    3. Alert Generation - Create prioritized alerts for stakeholders
    4. Escalation Management - Route alerts to appropriate response teams

Regulatory Context:
    EUDR Article 10 requires operators to update Due Diligence Statements
    if "substantive changes" occur. This workflow ensures continuous monitoring
    to detect such changes and trigger timely DDS updates.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    SIGNAL_COLLECTION = "signal_collection"
    EVENT_DETECTION = "event_detection"
    ALERT_GENERATION = "alert_generation"
    ESCALATION_MANAGEMENT = "escalation_management"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SignalSource(str, Enum):
    """Signal data sources."""
    SATELLITE = "satellite"
    NEWS_FEED = "news_feed"
    NGO_REPORT = "ngo_report"
    SUPPLIER_UPDATE = "supplier_update"
    REGULATORY = "regulatory"
    INTERNAL = "internal"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EscalationTier(str, Enum):
    """Escalation tiers."""
    TIER_1_ANALYST = "tier_1_analyst"
    TIER_2_MANAGER = "tier_2_manager"
    TIER_3_EXECUTIVE = "tier_3_executive"
    AUTOMATED = "automated"


# =============================================================================
# DATA MODELS
# =============================================================================


class ContinuousMonitoringConfig(BaseModel):
    """Configuration for continuous monitoring workflow."""
    monitoring_window_hours: int = Field(default=24, ge=1, description="Monitoring time window")
    signal_sources: List[SignalSource] = Field(
        default_factory=lambda: list(SignalSource),
        description="Active signal sources",
    )
    alert_threshold_score: float = Field(default=50.0, ge=0.0, le=100.0, description="Alert threshold")
    auto_escalate_critical: bool = Field(default=True, description="Auto-escalate critical alerts")
    operator_id: Optional[str] = Field(None, description="Operator context")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: ContinuousMonitoringConfig = Field(default_factory=ContinuousMonitoringConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the continuous monitoring workflow."""
    workflow_name: str = Field(default="continuous_monitoring", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    signals_collected: int = Field(default=0, ge=0, description="Total signals collected")
    events_detected: int = Field(default=0, ge=0, description="Significant events detected")
    alerts_generated: int = Field(default=0, ge=0, description="Alerts created")
    critical_alerts: int = Field(default=0, ge=0, description="Critical severity alerts")
    escalations: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation actions")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# CONTINUOUS MONITORING WORKFLOW
# =============================================================================


class ContinuousMonitoringWorkflow:
    """
    Four-phase continuous monitoring workflow.

    Implements real-time EUDR compliance monitoring with:
    - Multi-source signal aggregation (satellite, news, supplier, regulatory)
    - Machine learning event detection
    - Risk-based alert prioritization
    - Multi-tier escalation management

    Example:
        >>> config = ContinuousMonitoringConfig(
        ...     monitoring_window_hours=24,
        ...     alert_threshold_score=60.0,
        ... )
        >>> workflow = ContinuousMonitoringWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.overall_status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[ContinuousMonitoringConfig] = None) -> None:
        """Initialize the continuous monitoring workflow."""
        self.config = config or ContinuousMonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.ContinuousMonitoringWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 4-phase continuous monitoring workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with signals, events, alerts, and escalations.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting continuous monitoring workflow execution_id=%s window=%dh",
            context.execution_id,
            self.config.monitoring_window_hours,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.SIGNAL_COLLECTION, self._phase_1_signal_collection),
            (Phase.EVENT_DETECTION, self._phase_2_event_detection),
            (Phase.ALERT_GENERATION, self._phase_3_alert_generation),
            (Phase.ESCALATION_MANAGEMENT, self._phase_4_escalation_management),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Extract final outputs
        signals_collected = len(context.state.get("signals", []))
        events = context.state.get("events", [])
        alerts = context.state.get("alerts", [])
        escalations = context.state.get("escalations", [])
        critical_alerts = len([a for a in alerts if a.get("severity") == AlertSeverity.CRITICAL.value])

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "window_hours": self.config.monitoring_window_hours,
        })

        self.logger.info(
            "Continuous monitoring workflow finished execution_id=%s status=%s "
            "signals=%d events=%d alerts=%d",
            context.execution_id,
            overall_status.value,
            signals_collected,
            len(events),
            len(alerts),
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            signals_collected=signals_collected,
            events_detected=len(events),
            alerts_generated=len(alerts),
            critical_alerts=critical_alerts,
            escalations=escalations,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Signal Collection
    # -------------------------------------------------------------------------

    async def _phase_1_signal_collection(self, context: WorkflowContext) -> PhaseResult:
        """
        Aggregate signals from all configured data sources.

        Signal sources:
        - Satellite: Deforestation alerts (GLAD, RADD, JJ-FAST)
        - News: Media mentions of suppliers, countries, commodities
        - NGO Reports: Investigations, violations, campaigns
        - Supplier Updates: Certification changes, audit results
        - Regulatory: EUDR guidance updates, country benchmarking
        - Internal: User reports, data quality issues
        """
        phase = Phase.SIGNAL_COLLECTION
        self.logger.info(
            "Collecting signals from %d sources (window=%dh)",
            len(self.config.signal_sources),
            self.config.monitoring_window_hours,
        )

        # Simulate signal collection (replace with actual API calls)
        await asyncio.sleep(0.1)

        signals: List[Dict[str, Any]] = []
        window_start = datetime.utcnow() - timedelta(hours=self.config.monitoring_window_hours)

        for source in self.config.signal_sources:
            signal_count = random.randint(0, 50)
            for i in range(signal_count):
                signals.append({
                    "signal_id": f"SIG-{uuid.uuid4().hex[:8]}",
                    "source": source.value,
                    "timestamp": window_start + timedelta(hours=random.uniform(0, self.config.monitoring_window_hours)),
                    "score": random.uniform(0, 100),
                    "content": self._generate_signal_content(source),
                    "entities": self._extract_entities(source),
                })

        context.state["signals"] = signals

        # Group signals by source
        by_source = {}
        for s in signals:
            src = s["source"]
            by_source[src] = by_source.get(src, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "signal_count": len(signals),
            "sources": list(by_source.keys()),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "signals_collected": len(signals),
                "by_source": by_source,
                "window_start": window_start.isoformat(),
                "window_end": datetime.utcnow().isoformat(),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Event Detection
    # -------------------------------------------------------------------------

    async def _phase_2_event_detection(self, context: WorkflowContext) -> PhaseResult:
        """
        Identify significant events from signal stream.

        Event detection logic:
        - Clustering: Group related signals (same supplier, same plot, same timeframe)
        - Scoring: Calculate event significance score
        - Filtering: Retain only events above threshold
        - Classification: Categorize event type (deforestation, certification, regulatory, etc.)
        """
        phase = Phase.EVENT_DETECTION
        signals = context.state.get("signals", [])

        self.logger.info("Detecting events from %d signals", len(signals))

        # Simulate event detection (replace with ML model)
        events: List[Dict[str, Any]] = []

        # Cluster signals by entity and time proximity
        clusters = self._cluster_signals(signals)

        for cluster in clusters:
            # Calculate cluster significance
            significance = self._calculate_event_significance(cluster)

            if significance >= self.config.alert_threshold_score:
                event = {
                    "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
                    "event_type": self._classify_event_type(cluster),
                    "significance_score": round(significance, 2),
                    "signal_count": len(cluster),
                    "entities": self._aggregate_entities(cluster),
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": self._generate_event_description(cluster),
                }
                events.append(event)

        context.state["events"] = events

        # Group events by type
        by_type = {}
        for e in events:
            et = e["event_type"]
            by_type[et] = by_type.get(et, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "event_count": len(events),
            "types": list(by_type.keys()),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "events_detected": len(events),
                "by_type": by_type,
                "threshold": self.config.alert_threshold_score,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Alert Generation
    # -------------------------------------------------------------------------

    async def _phase_3_alert_generation(self, context: WorkflowContext) -> PhaseResult:
        """
        Create prioritized alerts for stakeholders.

        Alert generation:
        - Map events to alert severity (critical, high, medium, low, info)
        - Deduplicate similar alerts
        - Enrich with contextual information (supplier profile, DDS status)
        - Format for notification channels (email, Slack, dashboard)
        """
        phase = Phase.ALERT_GENERATION
        events = context.state.get("events", [])

        self.logger.info("Generating alerts from %d events", len(events))

        alerts: List[Dict[str, Any]] = []

        for event in events:
            severity = self._determine_alert_severity(event)

            alert = {
                "alert_id": f"ALT-{uuid.uuid4().hex[:8]}",
                "event_id": event["event_id"],
                "severity": severity.value,
                "title": self._generate_alert_title(event),
                "description": event.get("description", ""),
                "entities": event.get("entities", {}),
                "significance_score": event.get("significance_score", 0.0),
                "created_at": datetime.utcnow().isoformat(),
                "status": "open",
                "assigned_to": None,
            }
            alerts.append(alert)

        context.state["alerts"] = alerts

        # Group by severity
        by_severity = {}
        for a in alerts:
            sev = a["severity"]
            by_severity[sev] = by_severity.get(sev, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "alert_count": len(alerts),
            "severities": by_severity,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "alerts_generated": len(alerts),
                "by_severity": by_severity,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Escalation Management
    # -------------------------------------------------------------------------

    async def _phase_4_escalation_management(self, context: WorkflowContext) -> PhaseResult:
        """
        Route alerts to appropriate response teams.

        Escalation logic:
        - Critical alerts -> Tier 3 (Executive)
        - High alerts -> Tier 2 (Manager)
        - Medium/Low alerts -> Tier 1 (Analyst)
        - Info alerts -> Automated handling (log only)

        Auto-actions:
        - Critical deforestation alerts: Suspend supplier, trigger DDS review
        - Regulatory alerts: Notify compliance team, schedule policy review
        - Supplier alerts: Flag for next audit cycle
        """
        phase = Phase.ESCALATION_MANAGEMENT
        alerts = context.state.get("alerts", [])

        self.logger.info("Managing escalation for %d alerts", len(alerts))

        escalations: List[Dict[str, Any]] = []

        for alert in alerts:
            severity = alert["severity"]
            tier = self._determine_escalation_tier(severity)

            escalation = {
                "escalation_id": f"ESC-{uuid.uuid4().hex[:8]}",
                "alert_id": alert["alert_id"],
                "tier": tier.value,
                "assigned_at": datetime.utcnow().isoformat(),
                "auto_actions": self._generate_auto_actions(alert),
            }

            # Auto-escalate critical alerts if configured
            if severity == AlertSeverity.CRITICAL.value and self.config.auto_escalate_critical:
                escalation["tier"] = EscalationTier.TIER_3_EXECUTIVE.value
                escalation["auto_escalated"] = True

            escalations.append(escalation)

        context.state["escalations"] = escalations

        # Group by tier
        by_tier = {}
        for e in escalations:
            tier = e["tier"]
            by_tier[tier] = by_tier.get(tier, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "escalation_count": len(escalations),
            "tiers": by_tier,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "escalations_created": len(escalations),
                "by_tier": by_tier,
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_signal_content(self, source: SignalSource) -> str:
        """Generate realistic signal content for testing."""
        templates = {
            SignalSource.SATELLITE: "Deforestation alert detected in plot PLOT-{id}",
            SignalSource.NEWS_FEED: "Media report: Supplier {id} linked to deforestation",
            SignalSource.NGO_REPORT: "NGO investigation: Country {id} forest violations",
            SignalSource.SUPPLIER_UPDATE: "Supplier {id} certification expired",
            SignalSource.REGULATORY: "EUDR guidance update: Country {id} reclassified",
            SignalSource.INTERNAL: "Data quality issue: Missing geolocation for supplier {id}",
        }
        template = templates.get(source, "Signal from {id}")
        return template.format(id=uuid.uuid4().hex[:6].upper())

    def _extract_entities(self, source: SignalSource) -> Dict[str, List[str]]:
        """Extract entities from signal."""
        return {
            "suppliers": [f"SUP-{uuid.uuid4().hex[:8]}"] if random.random() > 0.5 else [],
            "plots": [f"PLOT-{uuid.uuid4().hex[:8]}"] if random.random() > 0.7 else [],
            "countries": [random.choice(["BR", "ID", "CO", "PE", "MY"])] if random.random() > 0.6 else [],
        }

    def _cluster_signals(self, signals: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster related signals."""
        # Simplified: random clustering for demonstration
        clusters = []
        current_cluster = []
        for sig in signals:
            current_cluster.append(sig)
            if len(current_cluster) >= random.randint(1, 5):
                clusters.append(current_cluster)
                current_cluster = []
        if current_cluster:
            clusters.append(current_cluster)
        return clusters

    def _calculate_event_significance(self, cluster: List[Dict[str, Any]]) -> float:
        """Calculate event significance score."""
        if not cluster:
            return 0.0
        avg_score = sum(s.get("score", 0) for s in cluster) / len(cluster)
        cluster_bonus = min(20.0, len(cluster) * 2.0)  # More signals = higher significance
        return min(100.0, avg_score + cluster_bonus)

    def _classify_event_type(self, cluster: List[Dict[str, Any]]) -> str:
        """Classify event type from signal cluster."""
        sources = [s.get("source") for s in cluster]
        if SignalSource.SATELLITE.value in sources:
            return "deforestation_alert"
        elif SignalSource.SUPPLIER_UPDATE.value in sources:
            return "supplier_change"
        elif SignalSource.REGULATORY.value in sources:
            return "regulatory_update"
        elif SignalSource.NGO_REPORT.value in sources:
            return "compliance_investigation"
        return "general_risk_event"

    def _aggregate_entities(self, cluster: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Aggregate entities across signal cluster."""
        all_entities: Dict[str, set] = {"suppliers": set(), "plots": set(), "countries": set()}
        for sig in cluster:
            entities = sig.get("entities", {})
            for key in all_entities:
                all_entities[key].update(entities.get(key, []))
        return {k: list(v) for k, v in all_entities.items()}

    def _generate_event_description(self, cluster: List[Dict[str, Any]]) -> str:
        """Generate human-readable event description."""
        event_type = self._classify_event_type(cluster)
        signal_count = len(cluster)
        return f"{event_type.replace('_', ' ').title()} detected from {signal_count} signal(s)"

    def _determine_alert_severity(self, event: Dict[str, Any]) -> AlertSeverity:
        """Determine alert severity from event."""
        score = event.get("significance_score", 0.0)
        event_type = event.get("event_type", "")

        if score >= 90.0 or event_type == "deforestation_alert":
            return AlertSeverity.CRITICAL
        elif score >= 70.0:
            return AlertSeverity.HIGH
        elif score >= 50.0:
            return AlertSeverity.MEDIUM
        elif score >= 30.0:
            return AlertSeverity.LOW
        return AlertSeverity.INFO

    def _generate_alert_title(self, event: Dict[str, Any]) -> str:
        """Generate alert title."""
        event_type = event.get("event_type", "event")
        entities = event.get("entities", {})
        supplier_count = len(entities.get("suppliers", []))
        if supplier_count > 0:
            return f"{event_type.replace('_', ' ').title()} affecting {supplier_count} supplier(s)"
        return event_type.replace('_', ' ').title()

    def _determine_escalation_tier(self, severity: str) -> EscalationTier:
        """Determine escalation tier from severity."""
        if severity == AlertSeverity.CRITICAL.value:
            return EscalationTier.TIER_3_EXECUTIVE
        elif severity == AlertSeverity.HIGH.value:
            return EscalationTier.TIER_2_MANAGER
        elif severity in (AlertSeverity.MEDIUM.value, AlertSeverity.LOW.value):
            return EscalationTier.TIER_1_ANALYST
        return EscalationTier.AUTOMATED

    def _generate_auto_actions(self, alert: Dict[str, Any]) -> List[str]:
        """Generate automated actions for alert."""
        actions = []
        severity = alert.get("severity")
        event_type = alert.get("event_id", "").split("-")[0]

        if severity == AlertSeverity.CRITICAL.value:
            actions.append("Suspend affected suppliers pending investigation")
            actions.append("Trigger DDS review and update workflow")
            actions.append("Notify executive stakeholder committee")

        if "deforestation" in alert.get("title", "").lower():
            actions.append("Request satellite imagery analysis")
            actions.append("Schedule supplier audit")

        if not actions:
            actions.append("Log alert for review")

        return actions

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
