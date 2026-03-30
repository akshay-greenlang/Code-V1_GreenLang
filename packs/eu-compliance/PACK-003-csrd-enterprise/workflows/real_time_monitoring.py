# -*- coding: utf-8 -*-
"""
Real-Time Monitoring Workflow
================================

4-phase continuous IoT monitoring workflow for facility-level emissions
tracking. Registers IoT devices, processes data streams, detects anomalies
in real-time, and dispatches alerts through multiple channels.

Phases:
    1. Device Registration: Register IoT sensors, validate protocols, assign IDs
    2. Stream Processing: Ingest and aggregate IoT telemetry into emission estimates
    3. Anomaly Detection: Real-time anomaly detection with configurable thresholds
    4. Alert Dispatch: Multi-channel alerting (webhook, email, Slack, Teams)

Author: GreenLang Team
Version: 3.0.0
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MonitoringStatus(str, Enum):
    """Monitoring session status."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AlertChannel(str, Enum):
    """Alert dispatch channels."""

    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"


class IoTProtocol(str, Enum):
    """Supported IoT communication protocols."""

    MQTT = "MQTT"
    HTTP = "HTTP"
    OPCUA = "OPCUA"
    MODBUS = "MODBUS"


class EscalationLevel(str, Enum):
    """Escalation levels for alert routing."""

    L1_OPERATOR = "L1_operator"
    L2_ENGINEER = "L2_engineer"
    L3_MANAGER = "L3_manager"
    L4_EXECUTIVE = "L4_executive"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class DeviceConfig(BaseModel):
    """IoT device configuration."""

    device_id: str = Field(default_factory=lambda: f"dev-{uuid.uuid4().hex[:8]}")
    device_type: str = Field(..., description="Device type (meter, sensor, gauge)")
    protocol: IoTProtocol = Field(default=IoTProtocol.MQTT, description="Communication protocol")
    location: str = Field(default="", description="Physical location description")
    measurement_type: str = Field(
        default="energy", description="What the device measures (energy, gas, water, etc.)"
    )
    unit: str = Field(default="kWh", description="Measurement unit")
    emission_factor_id: str = Field(
        default="", description="Emission factor for conversion to tCO2e"
    )
    polling_interval_seconds: int = Field(
        default=60, ge=1, le=3600, description="Data polling interval"
    )
    alert_thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Alert thresholds by severity"
    )


class AlertRule(BaseModel):
    """Alert rule configuration."""

    rule_id: str = Field(default_factory=lambda: f"rule-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Rule display name")
    condition: str = Field(
        ..., description="Condition expression (e.g., 'value > threshold')"
    )
    threshold: float = Field(..., description="Numeric threshold for the condition")
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM, description="Alert severity")
    channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.EMAIL],
        description="Alert dispatch channels",
    )
    cooldown_minutes: int = Field(
        default=15, ge=1, description="Minimum minutes between repeated alerts"
    )
    escalation_after_minutes: int = Field(
        default=60, ge=0, description="Minutes before escalation (0 = no escalation)"
    )


class EscalationRule(BaseModel):
    """Escalation rule for unacknowledged alerts."""

    severity: AlertSeverity = Field(..., description="Alert severity to escalate")
    escalation_path: List[EscalationLevel] = Field(
        default_factory=lambda: [
            EscalationLevel.L1_OPERATOR,
            EscalationLevel.L2_ENGINEER,
            EscalationLevel.L3_MANAGER,
        ],
        description="Escalation path from first to last",
    )
    escalation_interval_minutes: int = Field(
        default=30, ge=5, description="Minutes between escalation levels"
    )


class MonitoringConfig(BaseModel):
    """Configuration for a monitoring session."""

    facility_id: str = Field(..., description="Facility identifier")
    tenant_id: str = Field(default="", description="Tenant isolation ID")
    devices: List[DeviceConfig] = Field(
        default_factory=list, description="IoT devices to register"
    )
    alert_rules: List[AlertRule] = Field(
        default_factory=list, description="Alert rules to apply"
    )
    escalation_rules: List[EscalationRule] = Field(
        default_factory=list, description="Escalation rules"
    )
    aggregation_interval_seconds: int = Field(
        default=300, ge=60, le=3600, description="Metric aggregation interval"
    )
    anomaly_detection_method: str = Field(
        default="z_score", description="Anomaly detection method (z_score, iqr, isolation_forest)"
    )
    anomaly_threshold: float = Field(
        default=3.0, ge=1.0, le=10.0, description="Anomaly detection threshold (z-score)"
    )
    channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.EMAIL, AlertChannel.SLACK],
        description="Default alert channels",
    )
    webhook_url: str = Field(default="", description="Webhook endpoint URL")
    slack_webhook_url: str = Field(default="", description="Slack webhook URL")
    teams_webhook_url: str = Field(default="", description="Teams webhook URL")
    email_recipients: List[str] = Field(
        default_factory=list, description="Alert email recipients"
    )


class AlertRecord(BaseModel):
    """Record of a dispatched alert."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_id: str = Field(default="", description="Source device")
    rule_id: str = Field(default="", description="Triggering rule")
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)
    message: str = Field(default="", description="Alert message")
    value: float = Field(default=0.0, description="Value that triggered the alert")
    threshold: float = Field(default=0.0, description="Threshold that was exceeded")
    channels_dispatched: List[str] = Field(default_factory=list)
    acknowledged: bool = Field(default=False)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.L1_OPERATOR)


class MonitoringSession(BaseModel):
    """Active monitoring session state."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(default="", description="Parent workflow ID")
    facility_id: str = Field(default="", description="Monitored facility")
    status: MonitoringStatus = Field(default=MonitoringStatus.ACTIVE)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    devices_registered: int = Field(default=0)
    alerts_dispatched: int = Field(default=0)
    data_points_processed: int = Field(default=0)
    current_emission_rate_tco2e_hr: float = Field(
        default=0.0, description="Current emission rate in tCO2e per hour"
    )
    provenance_hash: str = Field(default="")


class MonitoringSummary(BaseModel):
    """Summary produced when monitoring is stopped."""

    session_id: str = Field(..., description="Session identifier")
    facility_id: str = Field(default="")
    status: MonitoringStatus = Field(default=MonitoringStatus.STOPPED)
    started_at: Optional[datetime] = Field(None)
    stopped_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    total_data_points: int = Field(default=0)
    total_alerts: int = Field(default=0)
    alerts_by_severity: Dict[str, int] = Field(default_factory=dict)
    total_emissions_tco2e: float = Field(default=0.0)
    average_emission_rate_tco2e_hr: float = Field(default=0.0)
    anomalies_detected: int = Field(default=0)
    alert_history: List[AlertRecord] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RealTimeMonitoringWorkflow:
    """
    4-phase continuous IoT monitoring workflow.

    Registers IoT devices at a facility, processes incoming telemetry
    streams into emission estimates, detects anomalies in real-time,
    and dispatches multi-channel alerts with escalation support.

    Supports start/stop lifecycle for continuous monitoring sessions.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig.
        _sessions: Active monitoring sessions keyed by session_id.
        _alert_history: Alert records for the current execution.

    Example:
        >>> workflow = RealTimeMonitoringWorkflow()
        >>> config = MonitoringConfig(
        ...     facility_id="facility-001",
        ...     devices=[DeviceConfig(device_type="meter", measurement_type="electricity")],
        ... )
        >>> session = await workflow.execute(config)
        >>> assert session.status == MonitoringStatus.ACTIVE
        >>> summary = await workflow.stop_monitoring(session.session_id)
        >>> assert summary.status == MonitoringStatus.STOPPED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the real-time monitoring workflow.

        Args:
            config: Optional EnterprisePackConfig for engine resolution.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sessions: Dict[str, MonitoringSession] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._alert_history: List[AlertRecord] = []
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, facility_id: str = "", config: Optional[MonitoringConfig] = None
    ) -> MonitoringSession:
        """
        Execute the 4-phase monitoring setup and create an active session.

        Args:
            facility_id: Facility identifier (used if config not provided).
            config: Full monitoring configuration.

        Returns:
            MonitoringSession with session ID, device count, and status.
        """
        if config is None:
            config = MonitoringConfig(facility_id=facility_id or "default-facility")

        self.logger.info(
            "Starting real-time monitoring workflow %s for facility=%s devices=%d",
            self.workflow_id, config.facility_id, len(config.devices),
        )

        phase_results: List[PhaseResult] = []

        # Phase 1: Device Registration
        p1 = await self._phase_1_device_registration(config)
        phase_results.append(p1)
        if p1.status == PhaseStatus.FAILED:
            return MonitoringSession(
                workflow_id=self.workflow_id,
                facility_id=config.facility_id,
                status=MonitoringStatus.ERROR,
                provenance_hash=self._hash_data({"error": p1.errors}),
            )

        # Phase 2: Stream Processing Setup
        p2 = await self._phase_2_stream_processing(config)
        phase_results.append(p2)

        # Phase 3: Anomaly Detection Setup
        p3 = await self._phase_3_anomaly_detection(config)
        phase_results.append(p3)

        # Phase 4: Alert Dispatch Setup
        p4 = await self._phase_4_alert_dispatch(config)
        phase_results.append(p4)

        # Create session
        session = MonitoringSession(
            workflow_id=self.workflow_id,
            facility_id=config.facility_id,
            status=MonitoringStatus.ACTIVE,
            devices_registered=p1.outputs.get("devices_registered", 0),
            provenance_hash=self._hash_data({
                "phases": [p.provenance_hash for p in phase_results]
            }),
        )

        self._sessions[session.session_id] = session
        self._context[session.session_id] = {
            "config": config,
            "phase_results": phase_results,
        }

        self.logger.info(
            "Monitoring session %s created for facility %s with %d devices",
            session.session_id, config.facility_id, session.devices_registered,
        )

        return session

    async def start_monitoring(self, session_id: str) -> MonitoringSession:
        """
        Start the continuous monitoring loop for a session.

        Launches a background task that continuously processes device
        telemetry, detects anomalies, and dispatches alerts.

        Args:
            session_id: Active session to start monitoring.

        Returns:
            Updated MonitoringSession with ACTIVE status.

        Raises:
            ValueError: If session_id is not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status == MonitoringStatus.ACTIVE and session_id in self._monitoring_tasks:
            self.logger.warning("Monitoring already active for session %s", session_id)
            return session

        session.status = MonitoringStatus.ACTIVE
        ctx = self._context.get(session_id, {})
        config = ctx.get("config")

        # Launch background monitoring task
        task = asyncio.create_task(self._monitoring_loop(session_id, config))
        self._monitoring_tasks[session_id] = task

        self.logger.info("Monitoring loop started for session %s", session_id)
        return session

    async def stop_monitoring(self, session_id: str) -> MonitoringSummary:
        """
        Stop the continuous monitoring loop and produce a summary.

        Args:
            session_id: Active session to stop.

        Returns:
            MonitoringSummary with aggregated metrics, alert history, and totals.

        Raises:
            ValueError: If session_id is not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Cancel monitoring task if running
        task = self._monitoring_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        session.status = MonitoringStatus.STOPPED
        stopped_at = datetime.utcnow()
        duration = (stopped_at - session.started_at).total_seconds()

        # Build alert summary
        session_alerts = [
            a for a in self._alert_history
            if a.device_id in self._context.get(session_id, {}).get("device_ids", [])
            or True  # Include all alerts for this session
        ]
        alerts_by_severity: Dict[str, int] = {}
        for alert in session_alerts:
            sev = alert.severity.value
            alerts_by_severity[sev] = alerts_by_severity.get(sev, 0) + 1

        summary = MonitoringSummary(
            session_id=session_id,
            facility_id=session.facility_id,
            status=MonitoringStatus.STOPPED,
            started_at=session.started_at,
            stopped_at=stopped_at,
            duration_seconds=duration,
            total_data_points=session.data_points_processed,
            total_alerts=session.alerts_dispatched,
            alerts_by_severity=alerts_by_severity,
            total_emissions_tco2e=session.data_points_processed * 0.001,
            average_emission_rate_tco2e_hr=(
                session.data_points_processed * 0.001 / (duration / 3600.0)
                if duration > 0 else 0.0
            ),
            anomalies_detected=len([
                a for a in session_alerts if "anomaly" in a.message.lower()
            ]),
            alert_history=session_alerts[-100:],  # Last 100 alerts
            provenance_hash=self._hash_data({
                "session_id": session_id,
                "total_points": session.data_points_processed,
                "total_alerts": session.alerts_dispatched,
            }),
        )

        self.logger.info(
            "Monitoring stopped for session %s: %d data points, %d alerts, %.1fs",
            session_id, session.data_points_processed,
            session.alerts_dispatched, duration,
        )

        return summary

    # -------------------------------------------------------------------------
    # Phase 1: Device Registration
    # -------------------------------------------------------------------------

    async def _phase_1_device_registration(
        self, config: MonitoringConfig
    ) -> PhaseResult:
        """
        Register IoT devices and validate communication protocols.

        For each device, validates the protocol configuration, assigns a
        unique device ID, registers with the telemetry ingestion service,
        and performs a connectivity test.

        Steps:
            1. Validate device configurations
            2. Register devices with telemetry service
            3. Test device connectivity
            4. Assign emission factors for conversion
        """
        phase_name = "device_registration"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        registered_devices: List[Dict[str, Any]] = []
        device_ids: List[str] = []

        for device in config.devices:
            # Step 1: Validate
            validation = self._validate_device_config(device)
            if not validation.get("valid", False):
                warnings.append(
                    f"Device {device.device_id} validation warning: {validation.get('reason', '')}"
                )

            # Step 2: Register
            registration = await self._register_device(
                config.facility_id, device, config.tenant_id
            )
            registered_devices.append(registration)
            device_ids.append(device.device_id)

            # Step 3: Connectivity test
            connected = await self._test_device_connectivity(device)
            if not connected:
                warnings.append(f"Device {device.device_id} connectivity test failed")

        outputs["devices_registered"] = len(registered_devices)
        outputs["device_ids"] = device_ids
        outputs["protocols_used"] = list({d.protocol.value for d in config.devices})

        # Step 4: Assign emission factors
        ef_assignments = await self._assign_emission_factors(config.devices)
        outputs["emission_factors_assigned"] = ef_assignments.get("assigned_count", 0)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Stream Processing
    # -------------------------------------------------------------------------

    async def _phase_2_stream_processing(
        self, config: MonitoringConfig
    ) -> PhaseResult:
        """
        Set up stream processing pipeline for IoT telemetry.

        Configures the data ingestion pipeline to receive device telemetry,
        aggregate readings at configurable intervals, and convert raw
        measurements into emission estimates using assigned emission factors.

        Steps:
            1. Configure ingestion pipeline (per protocol)
            2. Set up aggregation windows
            3. Configure emission factor application
            4. Validate pipeline with test data
        """
        phase_name = "stream_processing"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Configure pipeline
        pipeline = await self._configure_ingestion_pipeline(config)
        outputs["pipeline_id"] = pipeline.get("pipeline_id", "")
        outputs["protocols_configured"] = pipeline.get("protocols", [])

        # Step 2: Aggregation windows
        outputs["aggregation_interval_seconds"] = config.aggregation_interval_seconds
        outputs["aggregation_method"] = "time_weighted_average"

        # Step 3: Emission factor application
        ef_config = await self._configure_emission_conversion(config)
        outputs["conversion_configured"] = ef_config.get("configured", False)
        outputs["conversion_rules"] = ef_config.get("rule_count", 0)

        # Step 4: Pipeline validation
        test_result = await self._validate_pipeline(pipeline.get("pipeline_id", ""))
        outputs["pipeline_validated"] = test_result.get("valid", False)
        if not test_result.get("valid", False):
            warnings.append("Pipeline validation completed with warnings")

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Anomaly Detection
    # -------------------------------------------------------------------------

    async def _phase_3_anomaly_detection(
        self, config: MonitoringConfig
    ) -> PhaseResult:
        """
        Set up real-time anomaly detection with configurable thresholds.

        Configures anomaly detection models (z-score, IQR, or isolation forest)
        for each device stream. Establishes baseline from historical data
        and sets detection thresholds.

        Steps:
            1. Load historical baselines for each device
            2. Configure anomaly detection model
            3. Set detection thresholds per device
            4. Validate detector with synthetic anomalies
        """
        phase_name = "anomaly_detection"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Load baselines
        baselines = await self._load_device_baselines(config)
        outputs["baselines_loaded"] = baselines.get("count", 0)

        # Step 2: Configure model
        detector = await self._configure_anomaly_detector(
            config.anomaly_detection_method, config.anomaly_threshold
        )
        outputs["detection_method"] = config.anomaly_detection_method
        outputs["threshold"] = config.anomaly_threshold
        outputs["detector_id"] = detector.get("detector_id", "")

        # Step 3: Per-device thresholds
        device_thresholds: Dict[str, Dict[str, float]] = {}
        for device in config.devices:
            thresholds = device.alert_thresholds or {
                "critical": config.anomaly_threshold * 2,
                "high": config.anomaly_threshold * 1.5,
                "medium": config.anomaly_threshold,
                "low": config.anomaly_threshold * 0.5,
            }
            device_thresholds[device.device_id] = thresholds

        outputs["device_thresholds"] = device_thresholds

        # Step 4: Validation
        validation = await self._validate_anomaly_detector(detector)
        outputs["detector_validated"] = validation.get("valid", False)
        outputs["false_positive_rate"] = validation.get("false_positive_rate", 0.0)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Alert Dispatch
    # -------------------------------------------------------------------------

    async def _phase_4_alert_dispatch(
        self, config: MonitoringConfig
    ) -> PhaseResult:
        """
        Set up multi-channel alert dispatch and escalation rules.

        Configures alert routing to webhook, email, Slack, and Teams channels.
        Sets up escalation rules for unacknowledged alerts and cooldown periods
        to prevent alert fatigue.

        Steps:
            1. Validate and configure alert channels
            2. Register alert rules with cooldown settings
            3. Configure escalation paths
            4. Send test alert to each channel
        """
        phase_name = "alert_dispatch"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Configure channels
        configured_channels: List[str] = []
        for channel in config.channels:
            success = await self._configure_alert_channel(channel, config)
            if success:
                configured_channels.append(channel.value)
            else:
                warnings.append(f"Alert channel {channel.value} configuration incomplete")

        outputs["channels_configured"] = configured_channels

        # Step 2: Register alert rules
        rules_registered = 0
        for rule in config.alert_rules:
            await self._register_alert_rule(rule, config.facility_id)
            rules_registered += 1

        outputs["alert_rules_registered"] = rules_registered
        outputs["default_cooldown_minutes"] = 15

        # Step 3: Escalation paths
        escalation_configured = 0
        for esc_rule in config.escalation_rules:
            await self._configure_escalation(esc_rule)
            escalation_configured += 1

        outputs["escalation_rules_configured"] = escalation_configured
        outputs["escalation_levels"] = [e.value for e in EscalationLevel]

        # Step 4: Test alerts
        test_results: Dict[str, bool] = {}
        for channel in configured_channels:
            result = await self._send_test_alert(channel, config.facility_id)
            test_results[channel] = result

        outputs["test_alerts_sent"] = test_results
        outputs["all_channels_tested"] = all(test_results.values())

        if not all(test_results.values()):
            failed_channels = [ch for ch, ok in test_results.items() if not ok]
            warnings.append(f"Test alert failed for channels: {failed_channels}")

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Monitoring Loop
    # -------------------------------------------------------------------------

    async def _monitoring_loop(
        self, session_id: str, config: MonitoringConfig
    ) -> None:
        """
        Continuous monitoring loop that processes telemetry and dispatches alerts.

        This runs as a background asyncio task until cancelled or stopped.
        Each iteration: read devices -> aggregate -> detect anomalies -> alert.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        self.logger.info("Entering monitoring loop for session %s", session_id)
        interval = config.aggregation_interval_seconds if config else 300

        try:
            while session.status == MonitoringStatus.ACTIVE:
                # Read telemetry from all devices
                readings = await self._read_device_telemetry(config)
                session.data_points_processed += len(readings)

                # Aggregate into emission estimate
                aggregated = await self._aggregate_readings(readings, config)
                session.current_emission_rate_tco2e_hr = aggregated.get(
                    "emission_rate_tco2e_hr", 0.0
                )

                # Check for anomalies
                anomalies = await self._check_for_anomalies(readings, config)

                # Dispatch alerts for anomalies
                for anomaly in anomalies:
                    alert = await self._dispatch_alert(anomaly, config)
                    if alert:
                        self._alert_history.append(alert)
                        session.alerts_dispatched += 1

                # Wait for next interval
                await asyncio.sleep(min(interval, 5))

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled for session %s", session_id)
            raise
        except Exception as exc:
            self.logger.error(
                "Monitoring loop error for session %s: %s",
                session_id, str(exc), exc_info=True,
            )
            session.status = MonitoringStatus.ERROR

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    def _validate_device_config(self, device: DeviceConfig) -> Dict[str, Any]:
        """Validate device configuration."""
        return {"valid": True}

    async def _register_device(
        self, facility_id: str, device: DeviceConfig, tenant_id: str
    ) -> Dict[str, Any]:
        """Register a device with the telemetry service."""
        return {"device_id": device.device_id, "registered": True}

    async def _test_device_connectivity(self, device: DeviceConfig) -> bool:
        """Test device connectivity via its protocol."""
        return True

    async def _assign_emission_factors(
        self, devices: List[DeviceConfig]
    ) -> Dict[str, Any]:
        """Assign emission factors to devices for tCO2e conversion."""
        return {"assigned_count": len(devices)}

    async def _configure_ingestion_pipeline(
        self, config: MonitoringConfig
    ) -> Dict[str, Any]:
        """Configure the data ingestion pipeline."""
        return {
            "pipeline_id": f"pipe-{uuid.uuid4().hex[:8]}",
            "protocols": list({d.protocol.value for d in config.devices}),
        }

    async def _configure_emission_conversion(
        self, config: MonitoringConfig
    ) -> Dict[str, Any]:
        """Configure emission factor application in the pipeline."""
        return {"configured": True, "rule_count": len(config.devices)}

    async def _validate_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Validate pipeline with test data."""
        return {"valid": True, "test_records_processed": 10}

    async def _load_device_baselines(
        self, config: MonitoringConfig
    ) -> Dict[str, Any]:
        """Load historical baselines for anomaly detection."""
        return {"count": len(config.devices)}

    async def _configure_anomaly_detector(
        self, method: str, threshold: float
    ) -> Dict[str, Any]:
        """Configure the anomaly detection model."""
        return {"detector_id": f"det-{uuid.uuid4().hex[:8]}", "method": method}

    async def _validate_anomaly_detector(self, detector: Dict) -> Dict[str, Any]:
        """Validate anomaly detector with synthetic anomalies."""
        return {"valid": True, "false_positive_rate": 0.02}

    async def _configure_alert_channel(
        self, channel: AlertChannel, config: MonitoringConfig
    ) -> bool:
        """Configure a single alert channel."""
        return True

    async def _register_alert_rule(
        self, rule: AlertRule, facility_id: str
    ) -> Dict[str, Any]:
        """Register an alert rule."""
        return {"registered": True, "rule_id": rule.rule_id}

    async def _configure_escalation(self, rule: EscalationRule) -> Dict[str, Any]:
        """Configure an escalation rule."""
        return {"configured": True}

    async def _send_test_alert(
        self, channel: str, facility_id: str
    ) -> bool:
        """Send a test alert to a channel."""
        return True

    async def _read_device_telemetry(
        self, config: MonitoringConfig
    ) -> List[Dict[str, Any]]:
        """Read current telemetry from all devices."""
        return [
            {"device_id": d.device_id, "value": 100.0, "timestamp": datetime.utcnow().isoformat()}
            for d in (config.devices if config else [])
        ]

    async def _aggregate_readings(
        self, readings: List[Dict], config: MonitoringConfig
    ) -> Dict[str, Any]:
        """Aggregate readings into emission estimates."""
        total_value = sum(r.get("value", 0.0) for r in readings)
        return {"emission_rate_tco2e_hr": total_value * 0.0005, "readings_aggregated": len(readings)}

    async def _check_for_anomalies(
        self, readings: List[Dict], config: MonitoringConfig
    ) -> List[Dict[str, Any]]:
        """Check readings for anomalies."""
        return []

    async def _dispatch_alert(
        self, anomaly: Dict[str, Any], config: MonitoringConfig
    ) -> Optional[AlertRecord]:
        """Dispatch an alert for an anomaly."""
        return AlertRecord(
            device_id=anomaly.get("device_id", ""),
            severity=AlertSeverity(anomaly.get("severity", "medium")),
            message=f"Anomaly detected: {anomaly.get('description', '')}",
            value=anomaly.get("value", 0.0),
            threshold=anomaly.get("threshold", 0.0),
            channels_dispatched=[c.value for c in (config.channels if config else [])],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
