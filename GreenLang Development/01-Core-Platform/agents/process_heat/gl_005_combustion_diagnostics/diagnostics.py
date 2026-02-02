# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Combustion Diagnostics Agent
==============================================

This is the main agent class for the GL-005 COMBUSENSE (Combustion Diagnostics)
agent. It orchestrates all diagnostic components and provides a unified interface
for combustion analysis.

AGENT BOUNDARY:
    GL-005 is a DIAGNOSTICS-ONLY agent. It:
    - READS sensor data (from GL-018 or direct sensors)
    - ANALYZES combustion quality, anomalies, and equipment health
    - RECOMMENDS actions (but does NOT execute them)
    - GENERATES work orders (for CMMS integration)
    - PROVIDES long-term trending and compliance reporting

    GL-005 does NOT:
    - Execute control actions
    - Modify setpoints
    - Send commands to equipment
    - Override safety systems

DATA FLOW:
    Sensors --> GL-018 (Control) --> GL-005 (Diagnostics)
                     |                    |
                     v                    v
               Control Actions       Recommendations
                                    Work Orders
                                    Reports

ZERO-HALLUCINATION GUARANTEE:
    All calculations are deterministic.
    All algorithms use documented engineering methods.
    Full audit trail with SHA-256 provenance hashes.
    No AI/ML in critical calculation paths.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Intelligence imports for LLM capabilities
from greenlang.agents.intelligence_mixin import IntelligenceMixin, IntelligenceConfig
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

from greenlang.agents.process_heat.shared.base_agent import (
    AgentConfig,
    AgentState,
    BaseProcessHeatAgent,
    ProcessingMetadata,
    SafetyLevel,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    GL005Config,
    DiagnosticMode,
    FuelCategory,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    AnalysisStatus,
    CMMSWorkOrder,
    CombustionOperatingData,
    DiagnosticsInput,
    DiagnosticsOutput,
    FlueGasReading,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.combustion_quality import (
    CombustionQualityCalculator,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.anomaly_detection import (
    CombustionAnomalyDetector,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.fuel_characterization import (
    FuelCharacterizationEngine,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.maintenance_advisor import (
    MaintenanceAdvisor,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.trending import (
    TrendingEngine,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GL-005 COMBUSTION DIAGNOSTICS AGENT
# =============================================================================

class CombustionDiagnosticsAgent(IntelligenceMixin, BaseProcessHeatAgent[DiagnosticsInput, DiagnosticsOutput]):
    """
    GL-005 COMBUSENSE Combustion Diagnostics Agent.

    This agent provides comprehensive combustion diagnostics including:
    - Combustion Quality Index (CQI) calculation
    - Anomaly detection (SPC + ML + rule-based)
    - Fuel characterization from flue gas
    - Maintenance prediction and work order generation
    - Long-term trending and compliance reporting

    BOUNDARY DEFINITION:
        This agent is DIAGNOSTICS ONLY. It analyzes data and generates
        recommendations but does NOT execute any control actions.

    Example:
        >>> config = GL005Config(
        ...     agent_id="GL005-BOILER-01",
        ...     equipment_id="BLR-001",
        ... )
        >>> agent = CombustionDiagnosticsAgent(config)
        >>> await agent.start()
        >>>
        >>> result = agent.process(diagnostics_input)
        >>> print(f"CQI: {result.cqi.cqi_score}")
        >>> print(f"Anomalies: {result.anomaly_detection.total_anomalies}")
    """

    # Agent metadata
    AGENT_TYPE = "GL-005"
    AGENT_NAME = "COMBUSENSE"
    AGENT_VERSION = "1.0.0"
    AGENT_DESCRIPTION = "Combustion Diagnostics and Health Monitoring"

    def __init__(self, config: GL005Config) -> None:
        """
        Initialize GL-005 Combustion Diagnostics Agent.

        Args:
            config: GL-005 agent configuration
        """
        # Create base agent config
        base_config = AgentConfig(
            agent_id=config.agent_id,
            agent_type=self.AGENT_TYPE,
            name=config.agent_name,
            version=config.version,
        )

        # Initialize base agent (SIL-2 for diagnostics)
        super().__init__(base_config, safety_level=SafetyLevel.SIL_2)

        self.gl005_config = config
        self._baseline_cqi: Optional[float] = None
        self._historical_readings: List[FlueGasReading] = []

        # Initialize diagnostic components
        self._cqi_calculator = CombustionQualityCalculator(config.cqi)
        self._anomaly_detector = CombustionAnomalyDetector(config.anomaly_detection)
        self._fuel_engine = FuelCharacterizationEngine(config.fuel_characterization)
        self._maintenance_advisor = MaintenanceAdvisor(
            config.maintenance,
            config.equipment_id,
        )
        self._trending_engine = TrendingEngine(config.trending)

        # Processing state
        self._last_cqi_result = None
        self._last_anomaly_result = None
        self._processing_count = 0

        logger.info(
            f"GL-005 COMBUSENSE Agent initialized: {config.agent_id} "
            f"(equipment={config.equipment_id}, mode={config.mode.value})"
        )

        # Initialize intelligence with ADVANCED level for diagnostics
        self._init_intelligence(IntelligenceConfig(
            domain_context="combustion diagnostics and equipment health monitoring",
            regulatory_context="IEC 61511, SIL 2",
            enable_explanations=True,
            enable_recommendations=True,
            enable_anomaly_detection=True,
        ))

    # =========================================================================
    # INTELLIGENCE INTERFACE METHODS
    # =========================================================================

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return ADVANCED intelligence level for diagnostics."""
        return IntelligenceLevel.ADVANCED

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return advanced intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    async def _init_connections(self) -> None:
        """Initialize connections (placeholder for data source integration)."""
        # In production, this would connect to:
        # - GL-018 data stream
        # - OPC-UA/Modbus sensors
        # - CMMS API (if enabled)
        logger.info(f"GL-005 data source: {self.gl005_config.data_source_agent}")

    async def _close_connections(self) -> None:
        """Close connections."""
        # Flush trending data
        logger.info("GL-005 connections closed")

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def validate_input(self, input_data: DiagnosticsInput) -> bool:
        """
        Validate input data for diagnostics.

        Args:
            input_data: Diagnostics input

        Returns:
            True if valid
        """
        # Check equipment ID matches
        if input_data.equipment_id != self.gl005_config.equipment_id:
            logger.warning(
                f"Equipment ID mismatch: expected {self.gl005_config.equipment_id}, "
                f"got {input_data.equipment_id}"
            )
            return False

        # Check sensor data quality
        if input_data.flue_gas.data_quality_flag == "bad":
            logger.warning("Flue gas data quality flagged as BAD")
            return False

        # Check sensor status
        if input_data.flue_gas.sensor_status == "fault":
            logger.warning("Sensor status indicates FAULT")
            return False

        return True

    # =========================================================================
    # OUTPUT VALIDATION
    # =========================================================================

    def validate_output(self, output_data: DiagnosticsOutput) -> bool:
        """
        Validate output data.

        Args:
            output_data: Diagnostics output

        Returns:
            True if valid
        """
        # Verify provenance hash is present
        if not output_data.provenance_hash:
            logger.warning("Output missing provenance hash")
            return False

        # Verify processing time is reasonable
        if output_data.processing_time_ms > 30000:  # 30 seconds
            logger.warning(f"Processing time excessive: {output_data.processing_time_ms}ms")

        return True

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    def process(self, input_data: DiagnosticsInput) -> DiagnosticsOutput:
        """
        Main diagnostics processing method.

        Orchestrates all diagnostic analyses and produces a comprehensive
        output with CQI, anomalies, fuel characterization, and maintenance
        recommendations.

        IMPORTANT: This method is DIAGNOSTICS ONLY. It does not execute
        any control actions.

        Args:
            input_data: Validated diagnostics input

        Returns:
            Complete diagnostics output
        """
        start_time = time.time()
        self._processing_count += 1

        logger.info(
            f"GL-005 processing #{self._processing_count} for {input_data.equipment_id}"
        )

        audit_trail = []
        alerts = []
        recommendations = []
        control_suggestions = []
        work_orders = []

        # Initialize result placeholders
        cqi_result = None
        anomaly_result = None
        fuel_result = None
        maintenance_result = None
        compliance_result = None

        # Step 1: CQI Calculation
        if input_data.run_cqi_analysis:
            try:
                cqi_result = self._cqi_calculator.calculate(
                    input_data.flue_gas,
                    baseline_cqi=self._baseline_cqi,
                )
                self._last_cqi_result = cqi_result
                self._trending_engine.add_cqi_result(cqi_result)

                audit_trail.append({
                    "step": "cqi_calculation",
                    "status": "success",
                    "score": cqi_result.cqi_score,
                    "rating": cqi_result.cqi_rating.value,
                })

                # Generate alerts for poor CQI
                if cqi_result.cqi_score < 60:
                    alerts.append({
                        "type": "cqi_warning",
                        "severity": "warning" if cqi_result.cqi_score >= 40 else "alarm",
                        "message": f"CQI score {cqi_result.cqi_score:.1f} indicates poor combustion quality",
                    })

                # Control suggestions based on CQI
                if cqi_result.excess_air_pct > 30:
                    control_suggestions.append({
                        "parameter": "excess_air",
                        "action": "reduce",
                        "reason": f"Excess air {cqi_result.excess_air_pct:.1f}% is high",
                        "target": "15-25%",
                    })

            except Exception as e:
                logger.error(f"CQI calculation failed: {e}", exc_info=True)
                audit_trail.append({
                    "step": "cqi_calculation",
                    "status": "error",
                    "error": str(e),
                })

        # Step 2: Anomaly Detection
        if input_data.run_anomaly_detection:
            try:
                # Update baseline if we have historical data
                if input_data.historical_readings:
                    self._anomaly_detector.initialize_baseline(
                        input_data.historical_readings
                    )

                anomaly_result = self._anomaly_detector.detect(input_data.flue_gas)
                self._last_anomaly_result = anomaly_result

                audit_trail.append({
                    "step": "anomaly_detection",
                    "status": "success",
                    "anomalies_found": anomaly_result.total_anomalies,
                })

                # Generate alerts for anomalies
                for anomaly in anomaly_result.anomalies:
                    alerts.append({
                        "type": f"anomaly_{anomaly.anomaly_type.value}",
                        "severity": anomaly.severity.value,
                        "message": f"{anomaly.anomaly_type.value}: {anomaly.potential_causes[0] if anomaly.potential_causes else 'Unknown cause'}",
                    })

                    # Add recommended actions
                    recommendations.extend(anomaly.recommended_actions)

            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}", exc_info=True)
                audit_trail.append({
                    "step": "anomaly_detection",
                    "status": "error",
                    "error": str(e),
                })

        # Step 3: Fuel Characterization
        if input_data.run_fuel_characterization:
            try:
                fuel_result = self._fuel_engine.characterize(
                    input_data.flue_gas,
                    expected_fuel=self.gl005_config.primary_fuel,
                )

                audit_trail.append({
                    "step": "fuel_characterization",
                    "status": "success",
                    "fuel_type": fuel_result.primary_fuel.fuel_category.value,
                    "confidence": fuel_result.primary_fuel.confidence,
                })

                # Alert if fuel doesn't match expected
                if not fuel_result.matches_configured_fuel:
                    alerts.append({
                        "type": "fuel_mismatch",
                        "severity": "warning",
                        "message": f"Detected fuel ({fuel_result.primary_fuel.fuel_category.value}) "
                                  f"differs from configured ({self.gl005_config.primary_fuel.value})",
                    })

            except Exception as e:
                logger.error(f"Fuel characterization failed: {e}", exc_info=True)
                audit_trail.append({
                    "step": "fuel_characterization",
                    "status": "error",
                    "error": str(e),
                })

        # Step 4: Maintenance Advisory
        if input_data.run_maintenance_prediction:
            try:
                maintenance_result = self._maintenance_advisor.analyze(
                    input_data.flue_gas,
                    input_data.operating_data,
                    cqi_result,
                    anomaly_result,
                )

                audit_trail.append({
                    "step": "maintenance_advisory",
                    "status": "success",
                    "health_score": maintenance_result.equipment_health_score,
                    "recommendations": len(maintenance_result.recommendations),
                })

                # Add maintenance recommendations
                for rec in maintenance_result.recommendations:
                    recommendations.append(rec.title)

                # Alert for urgent maintenance
                if maintenance_result.urgent_actions_required:
                    alerts.append({
                        "type": "urgent_maintenance",
                        "severity": "alarm",
                        "message": "Urgent maintenance action required - see recommendations",
                    })

                # Get work orders if CMMS enabled
                # (Work orders are generated internally by MaintenanceAdvisor)

            except Exception as e:
                logger.error(f"Maintenance advisory failed: {e}", exc_info=True)
                audit_trail.append({
                    "step": "maintenance_advisory",
                    "status": "error",
                    "error": str(e),
                })

        # Step 5: Update trending data
        self._trending_engine.add_flue_gas_reading(input_data.flue_gas)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Determine overall status
        status = AnalysisStatus.SUCCESS
        if any(entry["status"] == "error" for entry in audit_trail):
            status = AnalysisStatus.PARTIAL

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data, cqi_result, anomaly_result, fuel_result, maintenance_result
        )

        # Build output
        output = DiagnosticsOutput(
            request_id=input_data.request_id,
            equipment_id=input_data.equipment_id,
            agent_id=self.gl005_config.agent_id,
            agent_version=self.AGENT_VERSION,
            status=status,
            processing_time_ms=round(processing_time_ms, 2),
            cqi=cqi_result,
            anomaly_detection=anomaly_result,
            fuel_characterization=fuel_result,
            maintenance_advisory=maintenance_result,
            compliance=compliance_result,
            work_orders=work_orders,
            alerts=alerts,
            recommendations=list(set(recommendations)),  # Deduplicate
            control_suggestions=control_suggestions,
            input_timestamp=input_data.flue_gas.timestamp,
            output_timestamp=datetime.now(timezone.utc),
            provenance_hash=provenance_hash,
            audit_trail=audit_trail,
        )

        logger.info(
            f"GL-005 processing complete: CQI={cqi_result.cqi_score if cqi_result else 'N/A'}, "
            f"anomalies={anomaly_result.total_anomalies if anomaly_result else 'N/A'}, "
            f"health={maintenance_result.equipment_health_score if maintenance_result else 'N/A'}, "
            f"time={processing_time_ms:.1f}ms"
        )

        # Generate intelligent explanation of diagnostic results
        output.explanation = self.generate_explanation(
            input_data={
                "equipment_id": input_data.equipment_id,
                "o2_pct": input_data.flue_gas.oxygen_pct,
                "co_ppm": input_data.flue_gas.co_ppm
            },
            output_data={
                "cqi_score": cqi_result.cqi_score if cqi_result else None,
                "anomalies": anomaly_result.total_anomalies if anomaly_result else 0,
                "health_score": maintenance_result.equipment_health_score if maintenance_result else None,
                "alerts_count": len(alerts)
            },
            calculation_steps=[
                f"CQI analysis: {cqi_result.cqi_rating.value if cqi_result else 'N/A'}",
                f"Anomaly detection: {anomaly_result.total_anomalies if anomaly_result else 0} found",
                f"Maintenance recommendations: {len(recommendations)}"
            ]
        )

        return output

    # =========================================================================
    # BASELINE MANAGEMENT
    # =========================================================================

    def set_cqi_baseline(self, baseline_cqi: float) -> None:
        """
        Set CQI baseline for trend comparison.

        Args:
            baseline_cqi: Baseline CQI score (clean equipment condition)
        """
        self._baseline_cqi = baseline_cqi
        logger.info(f"GL-005 CQI baseline set: {baseline_cqi}")

    def set_maintenance_baselines(
        self,
        stack_temp: float,
        efficiency: float,
        co_ppm: float,
    ) -> None:
        """
        Set maintenance prediction baselines.

        Args:
            stack_temp: Baseline stack temperature (clean condition)
            efficiency: Baseline efficiency
            co_ppm: Baseline CO level (new burner)
        """
        self._maintenance_advisor.set_baselines(stack_temp, efficiency, co_ppm)
        logger.info(
            f"GL-005 maintenance baselines set: "
            f"stack_temp={stack_temp}C, efficiency={efficiency}%, co={co_ppm}ppm"
        )

    def initialize_anomaly_baseline(
        self,
        historical_readings: List[FlueGasReading],
    ) -> None:
        """
        Initialize anomaly detection baseline from historical data.

        Args:
            historical_readings: Historical flue gas readings
        """
        self._anomaly_detector.initialize_baseline(historical_readings)
        self._historical_readings = historical_readings.copy()
        logger.info(
            f"GL-005 anomaly baseline initialized from {len(historical_readings)} readings"
        )

    # =========================================================================
    # TRENDING ACCESS
    # =========================================================================

    def get_cqi_trend(self, days: int = 30) -> Dict[str, Any]:
        """
        Get CQI trend analysis.

        Args:
            days: Number of days to analyze

        Returns:
            Trend analysis result as dict
        """
        trend = self._trending_engine.get_cqi_trend(days)
        return {
            "parameter": trend.parameter,
            "direction": trend.direction.value,
            "slope_per_day": trend.slope,
            "percent_change": trend.percent_change,
            "is_significant": trend.is_significant,
            "data_points": trend.data_points,
        }

    def get_trend_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive trend summary.

        Returns:
            Summary of all parameter trends
        """
        return self._trending_engine.get_trend_summary()

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _calculate_provenance_hash(
        self,
        input_data: DiagnosticsInput,
        cqi_result,
        anomaly_result,
        fuel_result,
        maintenance_result,
    ) -> str:
        """
        Calculate SHA-256 provenance hash for complete audit trail.

        Args:
            input_data: Input data
            cqi_result: CQI result
            anomaly_result: Anomaly detection result
            fuel_result: Fuel characterization result
            maintenance_result: Maintenance advisory result

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            "agent": {
                "id": self.gl005_config.agent_id,
                "type": self.AGENT_TYPE,
                "version": self.AGENT_VERSION,
            },
            "input": {
                "equipment_id": input_data.equipment_id,
                "request_id": input_data.request_id,
                "timestamp": input_data.flue_gas.timestamp.isoformat(),
                "o2": input_data.flue_gas.oxygen_pct,
                "co": input_data.flue_gas.co_ppm,
            },
            "output": {
                "cqi": cqi_result.cqi_score if cqi_result else None,
                "anomalies": anomaly_result.total_anomalies if anomaly_result else None,
                "fuel": fuel_result.primary_fuel.fuel_category.value if fuel_result else None,
                "health": maintenance_result.equipment_health_score if maintenance_result else None,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    # =========================================================================
    # AUDIT TRAIL
    # =========================================================================

    def get_component_audit_trails(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get audit trails from all diagnostic components.

        Returns:
            Dictionary of component audit trails
        """
        return {
            "cqi": self._cqi_calculator.get_audit_trail(),
            "anomaly_detection": self._anomaly_detector.get_audit_trail(),
            "fuel_characterization": self._fuel_engine.get_audit_trail(),
            "maintenance_advisory": self._maintenance_advisor.get_audit_trail(),
            "trending": self._trending_engine.get_audit_trail(),
        }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent processing statistics.

        Returns:
            Dictionary of agent statistics
        """
        stats = self.get_stats()
        stats.update({
            "processing_count": self._processing_count,
            "baseline_cqi": self._baseline_cqi,
            "historical_readings_count": len(self._historical_readings),
            "last_cqi": self._last_cqi_result.cqi_score if self._last_cqi_result else None,
            "last_anomaly_count": (
                self._last_anomaly_result.total_anomalies
                if self._last_anomaly_result
                else None
            ),
        })
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_combustion_diagnostics_agent(
    agent_id: str,
    equipment_id: str,
    fuel_type: FuelCategory = FuelCategory.NATURAL_GAS,
    mode: DiagnosticMode = DiagnosticMode.REAL_TIME,
) -> CombustionDiagnosticsAgent:
    """
    Factory function to create a GL-005 Combustion Diagnostics Agent.

    Args:
        agent_id: Unique agent identifier
        equipment_id: Target equipment identifier
        fuel_type: Primary fuel type
        mode: Diagnostic operating mode

    Returns:
        Configured CombustionDiagnosticsAgent

    Example:
        >>> agent = create_combustion_diagnostics_agent(
        ...     agent_id="GL005-BOILER-01",
        ...     equipment_id="BLR-001",
        ...     fuel_type=FuelCategory.NATURAL_GAS,
        ... )
    """
    config = GL005Config(
        agent_id=agent_id,
        equipment_id=equipment_id,
        primary_fuel=fuel_type,
        mode=mode,
    )

    return CombustionDiagnosticsAgent(config)
