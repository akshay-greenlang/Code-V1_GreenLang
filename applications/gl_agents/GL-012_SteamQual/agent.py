"""
GL-012_SteamQual - Main Agent Entry Point

Steam Quality Controller for industrial steam systems.
Provides real-time quality monitoring, estimation, and control recommendations.

This agent integrates with GL-003 UNIFIEDSTEAM as a domain module.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json

from .core import SteamQualConfig, SteamQualOrchestrator
from .models import (
    SteamMeasurement,
    QualityEstimate,
    CarryoverRiskAssessment,
    QualityState,
    QualityConstraints,
    QualityEvent,
    ProcessData,
)
from .calculators import (
    DrynessFractionCalculator,
    CarryoverRiskCalculator,
    SeparatorEfficiencyCalculator,
)
from .estimators import SoftSensor, CarryoverDetector
from .monitoring import QualityMetrics, QualityAlerting
from .explainability import QualityExplainer, RootCauseAnalyzer


@dataclass
class SteamQualAgent:
    """
    Steam Quality Controller Agent

    Agent ID: GL-012
    Version: 1.0.0

    Provides:
    - Real-time steam dryness fraction estimation
    - Moisture carryover risk assessment
    - Separator/scrubber efficiency monitoring
    - Quality event detection and alerting
    - Control recommendations (advisory mode)
    - Integration with GL-003 UNIFIEDSTEAM optimizer

    Standards:
    - ASME PTC 19.11 Steam Quality
    - IAPWS-IF97 Steam Tables

    Primary Calculations:
    - Dryness fraction x = (h - hf) / hfg
    - Carryover risk scoring
    - Separator efficiency estimation
    - Uncertainty propagation

    Attributes:
        config: Agent configuration
        track_provenance: Enable SHA-256 provenance tracking
    """

    AGENT_ID = "GL-012"
    VERSION = "1.0.0"
    NAME = "SteamQual"

    config: Optional[SteamQualConfig] = None
    track_provenance: bool = True

    # Internal components (initialized lazily)
    _orchestrator: Optional[SteamQualOrchestrator] = field(default=None, repr=False)
    _dryness_calculator: Optional[DrynessFractionCalculator] = field(default=None, repr=False)
    _carryover_calculator: Optional[CarryoverRiskCalculator] = field(default=None, repr=False)
    _separator_calculator: Optional[SeparatorEfficiencyCalculator] = field(default=None, repr=False)
    _soft_sensor: Optional[SoftSensor] = field(default=None, repr=False)
    _carryover_detector: Optional[CarryoverDetector] = field(default=None, repr=False)
    _metrics: Optional[QualityMetrics] = field(default=None, repr=False)
    _alerting: Optional[QualityAlerting] = field(default=None, repr=False)
    _explainer: Optional[QualityExplainer] = field(default=None, repr=False)
    _calculation_history: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize agent components."""
        if self.config is None:
            self.config = SteamQualConfig()

        self._initialize_components()

    def _initialize_components(self):
        """Initialize internal components."""
        # Calculators
        self._dryness_calculator = DrynessFractionCalculator(
            agent_id=self.AGENT_ID,
            track_provenance=self.track_provenance,
        )
        self._carryover_calculator = CarryoverRiskCalculator(
            agent_id=self.AGENT_ID,
            track_provenance=self.track_provenance,
        )
        self._separator_calculator = SeparatorEfficiencyCalculator(
            agent_id=self.AGENT_ID,
            track_provenance=self.track_provenance,
        )

        # Estimators
        self._soft_sensor = SoftSensor(agent_id=self.AGENT_ID)
        self._carryover_detector = CarryoverDetector(agent_id=self.AGENT_ID)

        # Monitoring
        self._metrics = QualityMetrics()
        self._alerting = QualityAlerting(config=self.config)

        # Explainability
        self._explainer = QualityExplainer(agent_id=self.AGENT_ID)

        # Orchestrator
        self._orchestrator = SteamQualOrchestrator(
            config=self.config,
            dryness_calculator=self._dryness_calculator,
            carryover_calculator=self._carryover_calculator,
            separator_calculator=self._separator_calculator,
            soft_sensor=self._soft_sensor,
            carryover_detector=self._carryover_detector,
            alerting=self._alerting,
            track_provenance=self.track_provenance,
        )

    def estimate_quality(
        self,
        measurement: SteamMeasurement,
        use_soft_sensor: bool = True,
    ) -> QualityEstimate:
        """
        Estimate steam quality (dryness fraction) from measurement.

        Args:
            measurement: Steam pressure, temperature, and flow data
            use_soft_sensor: Use hybrid soft sensor for improved accuracy

        Returns:
            QualityEstimate with dryness fraction, uncertainty, and provenance
        """
        start_time = datetime.now(timezone.utc)

        if use_soft_sensor:
            result = self._soft_sensor.estimate(measurement)
        else:
            result = self._dryness_calculator.calculate_from_measurement(measurement)

        # Track provenance
        if self.track_provenance:
            self._record_calculation("estimate_quality", measurement, result, start_time)

        # Update metrics
        self._metrics.record_quality_estimate(result)

        return result

    def assess_carryover_risk(
        self,
        process_data: ProcessData,
    ) -> CarryoverRiskAssessment:
        """
        Assess moisture carryover risk.

        Args:
            process_data: Process measurements including drum level, load, etc.

        Returns:
            CarryoverRiskAssessment with risk score and contributing factors
        """
        start_time = datetime.now(timezone.utc)

        result = self._carryover_calculator.assess_risk(process_data)

        # Check for carryover events
        events = self._carryover_detector.detect(process_data)
        for event in events:
            self._alerting.emit_event(event)

        # Track provenance
        if self.track_provenance:
            self._record_calculation("assess_carryover_risk", process_data, result, start_time)

        # Update metrics
        self._metrics.record_carryover_risk(result)

        return result

    def get_quality_state(self, header_id: str) -> QualityState:
        """
        Get current quality state for a header.

        Args:
            header_id: Identifier for the steam header

        Returns:
            QualityState for GL-003 optimizer integration
        """
        return self._orchestrator.get_quality_state(header_id)

    def get_quality_constraints(self, header_id: str) -> QualityConstraints:
        """
        Get quality constraints for optimizer.

        Args:
            header_id: Identifier for the steam header

        Returns:
            QualityConstraints for GL-003 optimizer
        """
        return self._orchestrator.get_quality_constraints(header_id)

    def get_events(
        self,
        since: Optional[datetime] = None,
        severity_min: Optional[str] = None,
    ) -> List[QualityEvent]:
        """
        Get quality events.

        Args:
            since: Filter events after this timestamp
            severity_min: Minimum severity level (S0, S1, S2, S3)

        Returns:
            List of quality events
        """
        return self._alerting.get_events(since=since, severity_min=severity_min)

    def explain(self, result: QualityEstimate) -> Dict[str, Any]:
        """
        Generate explanation for a quality estimate.

        Args:
            result: Quality estimate to explain

        Returns:
            Explanation with methodology, feature importance, and confidence
        """
        return self._explainer.explain(result)

    def analyze_root_cause(self, event: QualityEvent) -> Dict[str, Any]:
        """
        Analyze root cause of a quality event.

        Args:
            event: Quality event to analyze

        Returns:
            Root cause analysis with contributing factors and recommendations
        """
        analyzer = RootCauseAnalyzer(agent_id=self.AGENT_ID)
        return analyzer.analyze(event)

    def get_recommendations(
        self,
        quality_state: QualityState,
    ) -> List[Dict[str, Any]]:
        """
        Get control recommendations for quality improvement.

        Args:
            quality_state: Current quality state

        Returns:
            List of prioritized recommendations with expected impact
        """
        return self._orchestrator.get_recommendations(quality_state)

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get calculation audit trail."""
        return self._calculation_history.copy()

    def _record_calculation(
        self,
        calculation_type: str,
        inputs: Any,
        outputs: Any,
        start_time: datetime,
    ) -> None:
        """Record calculation in audit trail."""
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(outputs)
        computation_hash = self._compute_hash({
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "calculation_type": calculation_type,
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
        })

        self._calculation_history.append({
            "timestamp": start_time.isoformat(),
            "calculation_type": calculation_type,
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "computation_hash": computation_hash,
            "execution_time_ms": execution_time_ms,
        })

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        elif hasattr(data, '__dict__'):
            data = data.__dict__
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# Convenience function for quick usage
def create_agent(
    config: Optional[Dict[str, Any]] = None,
    track_provenance: bool = True,
) -> SteamQualAgent:
    """
    Create a SteamQual agent instance.

    Args:
        config: Optional configuration overrides
        track_provenance: Enable SHA-256 provenance tracking

    Returns:
        Configured SteamQualAgent instance
    """
    agent_config = SteamQualConfig(**(config or {}))
    return SteamQualAgent(config=agent_config, track_provenance=track_provenance)
