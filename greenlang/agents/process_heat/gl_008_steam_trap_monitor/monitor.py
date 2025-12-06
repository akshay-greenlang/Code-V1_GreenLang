# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Steam Trap Monitoring Agent

This module implements the main SteamTrapMonitorAgent for comprehensive
steam trap monitoring, diagnostics, and management. It integrates all
component modules to provide a complete solution for steam trap programs.

Features:
    - Multi-method diagnostics (ultrasonic, temperature, visual)
    - Zero-hallucination calculations (deterministic only)
    - DOE Best Practices compliance
    - Spirax Sarco methodology
    - ASME B16.34 compliance checking
    - Economic analysis with ROI calculations
    - Survey route optimization (TSP)
    - Wireless sensor network integration
    - SHA-256 provenance tracking

Standards:
    - DOE Steam System Best Practices
    - Spirax Sarco Application Guides
    - ASME B16.34 Valve Pressure-Temperature Ratings
    - ISO 6552 Automatic Steam Traps

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor import (
    ...     SteamTrapMonitorAgent,
    ...     SteamTrapMonitorConfig,
    ...     TrapDiagnosticInput,
    ... )
    >>> config = SteamTrapMonitorConfig(
    ...     plant_id="PLANT-001",
    ...     steam_pressure_psig=150.0,
    ... )
    >>> agent = SteamTrapMonitorAgent(config)
    >>> result = agent.process(diagnostic_input)
    >>> print(f"Status: {result.condition.status}")
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import time
import uuid

from pydantic import BaseModel, Field

# Import from shared base
from greenlang.agents.process_heat.shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig,
    AgentCapability,
    AgentState,
    SafetyLevel,
    ProcessingError,
    ValidationError,
)

# Import local modules
from .config import (
    SteamTrapMonitorConfig,
    TrapType,
    TrapApplication,
    FailureMode,
    DiagnosticMethod,
    TrapTypeConfig,
    DiagnosticThresholds,
    EconomicsConfig,
)
from .schemas import (
    TrapDiagnosticInput,
    TrapDiagnosticOutput,
    TrapCondition,
    TrapHealthScore,
    SteamLossEstimate,
    MaintenanceRecommendation,
    FailureModeProbability,
    TrapStatus,
    DiagnosisConfidence,
    TrendDirection,
    MaintenancePriority,
    CondensateLoadInput,
    CondensateLoadOutput,
    TrapSurveyInput,
    SurveyRouteOutput,
    TrapStatusSummary,
    EconomicAnalysisOutput,
    UltrasonicReading,
    TemperatureReading,
)
from .trap_types import TrapTypeClassifier, TrapCharacteristics
from .condensate_load import CondensateLoadCalculator
from .failure_diagnostics import TrapDiagnosticsEngine
from .survey_management import TrapSurveyManager, TrapPopulationManager
from .wireless_sensors import WirelessSensorNetwork
from .economics import EconomicAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

class SteamTrapAgentConfig(AgentConfig):
    """Extended agent configuration for Steam Trap Monitor."""

    steam_trap_config: SteamTrapMonitorConfig = Field(
        ...,
        description="Steam trap monitoring configuration"
    )


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class SteamTrapMonitorAgent(BaseProcessHeatAgent[TrapDiagnosticInput, TrapDiagnosticOutput]):
    """
    GL-008 TRAPCATCHER - Steam Trap Monitoring Agent.

    This agent provides comprehensive steam trap monitoring capabilities
    including diagnostics, economic analysis, survey management, and
    wireless sensor integration. All calculations follow zero-hallucination
    principles with deterministic formulas only.

    Attributes:
        steam_config: Steam trap monitoring configuration
        trap_classifier: Trap type classification engine
        condensate_calc: Condensate load calculator
        diagnostics_engine: Multi-method diagnostics engine
        survey_manager: Survey planning and route optimization
        wireless_network: Wireless sensor network manager
        economic_analyzer: Economic impact analyzer
        population_manager: Trap population registry

    Example:
        >>> config = SteamTrapMonitorConfig(
        ...     plant_id="PLANT-001",
        ...     steam_pressure_psig=150.0,
        ...     economics=EconomicsConfig(steam_cost_per_mlb=15.00),
        ... )
        >>> agent_config = SteamTrapAgentConfig(
        ...     agent_type="GL-008",
        ...     name="TRAPCATCHER",
        ...     steam_trap_config=config,
        ... )
        >>> agent = SteamTrapMonitorAgent(agent_config)
        >>> await agent.start()
        >>> result = agent.process(diagnostic_input)
    """

    # Agent metadata
    AGENT_ID = "GL-008"
    AGENT_NAME = "TRAPCATCHER"
    AGENT_VERSION = "1.0.0"
    AGENT_DESCRIPTION = "Steam Trap Monitoring and Diagnostics Agent"

    # DOE Best Practices compliance
    DOE_COMPLIANCE = True
    SPIRAX_SARCO_METHODOLOGY = True
    ASME_B16_34_COMPLIANCE = True

    # Performance targets
    TARGET_ACCURACY_PCT = 95.0
    TARGET_COVERAGE_PCT = 100.0
    MAX_PROCESSING_TIME_MS = 500.0

    def __init__(
        self,
        config: Union[SteamTrapAgentConfig, SteamTrapMonitorConfig],
        safety_level: SafetyLevel = SafetyLevel.SIL_2,
    ) -> None:
        """
        Initialize the Steam Trap Monitor Agent.

        Args:
            config: Agent configuration or steam trap configuration
            safety_level: Safety Integrity Level (default SIL-2)
        """
        # Handle both config types
        if isinstance(config, SteamTrapMonitorConfig):
            agent_config = SteamTrapAgentConfig(
                agent_type=self.AGENT_ID,
                name=self.AGENT_NAME,
                version=self.AGENT_VERSION,
                capabilities={
                    AgentCapability.REAL_TIME_MONITORING,
                    AgentCapability.PREDICTIVE_ANALYTICS,
                    AgentCapability.OPTIMIZATION,
                    AgentCapability.COMPLIANCE_REPORTING,
                },
                steam_trap_config=config,
            )
            self.steam_config = config
        else:
            agent_config = config
            self.steam_config = config.steam_trap_config

        # Initialize base class
        super().__init__(agent_config, safety_level)

        # Initialize component engines
        self._init_components()

        # Processing statistics
        self._processing_count = 0
        self._total_processing_time_ms = 0.0
        self._failures_detected = 0
        self._total_steam_loss_lb_hr = 0.0

        logger.info(
            f"Initialized {self.AGENT_ID} {self.AGENT_NAME} v{self.AGENT_VERSION} "
            f"for plant {self.steam_config.plant_id}"
        )

    def _init_components(self) -> None:
        """Initialize all component engines."""
        # Trap type classifier
        self.trap_classifier = TrapTypeClassifier()

        # Condensate load calculator
        self.condensate_calc = CondensateLoadCalculator(
            steam_pressure_psig=self.steam_config.steam_pressure_psig
        )

        # Diagnostics engine with thresholds from config
        self.diagnostics_engine = TrapDiagnosticsEngine(
            ultrasonic_thresholds=self.steam_config.diagnostics.ultrasonic,
            temperature_thresholds=self.steam_config.diagnostics.temperature,
        )

        # Population manager for trap registry
        self.population_manager = TrapPopulationManager()

        # Survey manager with config
        self.survey_manager = TrapSurveyManager(
            population_manager=self.population_manager,
            config=self.steam_config.survey,
        )

        # Wireless sensor network
        self.wireless_network = WirelessSensorNetwork(
            config=self.steam_config.wireless
        )

        # Economic analyzer
        self.economic_analyzer = EconomicAnalyzer(
            config=self.steam_config.economics
        )

        logger.debug("All component engines initialized")

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def process(self, input_data: TrapDiagnosticInput) -> TrapDiagnosticOutput:
        """
        Process steam trap diagnostic analysis.

        This is the main entry point for trap diagnostics. It performs
        multi-method analysis, calculates steam losses, and generates
        maintenance recommendations.

        Args:
            input_data: Diagnostic input with sensor readings and trap info

        Returns:
            Comprehensive diagnostic output with status, losses, recommendations

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If diagnostic processing fails

        Example:
            >>> result = agent.process(diagnostic_input)
            >>> if result.condition.status == TrapStatus.FAILED_OPEN:
            ...     print(f"Steam loss: {result.steam_loss.steam_loss_lb_hr} lb/hr")
        """
        start_time = time.perf_counter()

        try:
            # Validate input with safety guard
            with self.safety_guard():
                if not self.validate_input(input_data):
                    raise ValidationError(
                        f"Input validation failed for trap {input_data.trap_info.trap_id}"
                    )

                # Track processing state
                self.state = AgentState.PROCESSING
                self._processing_count += 1

                # Step 1: Run diagnostics
                diagnostic_result = self._run_diagnostics(input_data)

                # Step 2: Calculate steam losses if failed
                steam_loss = self._calculate_steam_loss(
                    input_data, diagnostic_result
                )

                # Step 3: Calculate health score
                health_score = self._calculate_health_score(
                    input_data, diagnostic_result, steam_loss
                )

                # Step 4: Generate failure mode probabilities
                failure_probs = self._get_failure_probabilities(diagnostic_result)

                # Step 5: Generate recommendations
                recommendations = self._generate_recommendations(
                    input_data, diagnostic_result, steam_loss
                )

                # Step 6: Check ASME B16.34 compliance
                asme_compliant = self._check_asme_compliance(input_data)

                # Calculate processing time
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                self._total_processing_time_ms += processing_time_ms

                # Track failures
                if diagnostic_result.status != TrapStatus.GOOD:
                    self._failures_detected += 1
                    self._total_steam_loss_lb_hr += steam_loss.steam_loss_lb_hr

                # Build output
                output = TrapDiagnosticOutput(
                    request_id=input_data.request_id,
                    trap_id=input_data.trap_info.trap_id,
                    timestamp=datetime.now(timezone.utc),
                    status="success",
                    processing_time_ms=processing_time_ms,
                    condition=diagnostic_result,
                    health_score=health_score,
                    failure_probabilities=failure_probs,
                    steam_loss=steam_loss,
                    recommendations=recommendations,
                    diagnostic_methods_used=self._get_methods_used(input_data),
                    sensor_data_quality=self._assess_data_quality(input_data),
                    asme_b16_34_compliant=asme_compliant,
                    pressure_rating_adequate=self._check_pressure_rating(input_data),
                )

                # Calculate provenance hash
                if self.steam_config.provenance_tracking:
                    output.provenance_hash = self.calculate_provenance_hash(
                        input_data, output
                    )

                # Validate output
                if not self.validate_output(output):
                    logger.warning(
                        f"Output validation failed for trap {input_data.trap_info.trap_id}"
                    )

                # Return to ready state
                self.state = AgentState.READY

                logger.info(
                    f"Processed trap {input_data.trap_info.trap_id}: "
                    f"{output.condition.status.value} "
                    f"(confidence: {output.condition.confidence.value})"
                )

                return output

        except ValidationError:
            self.state = AgentState.ERROR
            raise

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(
                f"Processing failed for trap {input_data.trap_info.trap_id}: {e}",
                exc_info=True
            )
            raise ProcessingError(f"Diagnostic processing failed: {str(e)}") from e

    def validate_input(self, input_data: TrapDiagnosticInput) -> bool:
        """
        Validate diagnostic input data.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not input_data.trap_info.trap_id:
                logger.warning("Missing trap_id in input")
                return False

            if not input_data.trap_info.trap_type:
                logger.warning("Missing trap_type in input")
                return False

            # Check steam pressure is reasonable
            if input_data.steam_pressure_psig <= 0:
                logger.warning(f"Invalid steam pressure: {input_data.steam_pressure_psig}")
                return False

            if input_data.steam_pressure_psig > 1000:
                logger.warning(f"Steam pressure exceeds maximum: {input_data.steam_pressure_psig}")
                return False

            # Must have at least one diagnostic method
            has_ultrasonic = len(input_data.ultrasonic_readings) > 0
            has_temperature = len(input_data.temperature_readings) > 0
            has_visual = input_data.visual_inspection is not None

            if not (has_ultrasonic or has_temperature or has_visual):
                logger.warning("No diagnostic data provided (ultrasonic, temperature, or visual)")
                return False

            # Validate sensor readings
            for reading in input_data.ultrasonic_readings:
                if reading.decibel_level_db < 0 or reading.decibel_level_db > 120:
                    logger.warning(f"Invalid ultrasonic reading: {reading.decibel_level_db} dB")
                    return False

            for reading in input_data.temperature_readings:
                if reading.inlet_temp_f < 32 or reading.inlet_temp_f > 1000:
                    logger.warning(f"Invalid inlet temperature: {reading.inlet_temp_f} F")
                    return False

            return True

        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False

    def validate_output(self, output_data: TrapDiagnosticOutput) -> bool:
        """
        Validate diagnostic output data.

        Args:
            output_data: Output data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not output_data.trap_id:
                logger.warning("Missing trap_id in output")
                return False

            # Validate condition
            if output_data.condition is None:
                logger.warning("Missing condition in output")
                return False

            # Validate health score range
            if not (0 <= output_data.health_score.overall_score <= 100):
                logger.warning(
                    f"Invalid health score: {output_data.health_score.overall_score}"
                )
                return False

            # Validate steam loss values are non-negative
            if output_data.steam_loss.steam_loss_lb_hr < 0:
                logger.warning(
                    f"Negative steam loss: {output_data.steam_loss.steam_loss_lb_hr}"
                )
                return False

            # Validate processing time
            if output_data.processing_time_ms < 0:
                logger.warning(
                    f"Negative processing time: {output_data.processing_time_ms}"
                )
                return False

            if output_data.processing_time_ms > self.MAX_PROCESSING_TIME_MS * 10:
                logger.warning(
                    f"Excessive processing time: {output_data.processing_time_ms}ms"
                )

            return True

        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return False

    # =========================================================================
    # DIAGNOSTIC METHODS
    # =========================================================================

    def _run_diagnostics(
        self, input_data: TrapDiagnosticInput
    ) -> TrapCondition:
        """
        Run comprehensive diagnostics on trap sensor data.

        Uses multi-method analysis combining ultrasonic, temperature,
        and visual inspection data per Spirax Sarco methodology.

        Args:
            input_data: Diagnostic input with sensor readings

        Returns:
            TrapCondition with status and confidence
        """
        # Get trap type configuration for adjusted thresholds
        trap_type_config = self.steam_config.get_trap_type_config(
            input_data.trap_info.trap_type
        )

        # Prepare diagnostic data
        ultrasonic_data = None
        if input_data.ultrasonic_readings:
            # Use most recent reading
            latest = sorted(
                input_data.ultrasonic_readings,
                key=lambda x: x.timestamp,
                reverse=True
            )[0]
            ultrasonic_data = {
                "decibel_level_db": latest.decibel_level_db,
                "cycling_detected": latest.cycling_detected,
                "continuous_flow_detected": latest.continuous_flow_detected,
                "background_noise_db": latest.background_noise_db,
            }

        temperature_data = None
        if input_data.temperature_readings:
            latest = sorted(
                input_data.temperature_readings,
                key=lambda x: x.timestamp,
                reverse=True
            )[0]
            # Calculate saturation temperature
            sat_temp = self.steam_config.get_saturation_temperature(
                input_data.steam_pressure_psig
            )
            temperature_data = {
                "inlet_temp_f": latest.inlet_temp_f,
                "outlet_temp_f": latest.outlet_temp_f,
                "delta_t_f": latest.delta_t_f or (latest.inlet_temp_f - latest.outlet_temp_f),
                "saturation_temp_f": sat_temp,
                "subcooling_f": sat_temp - latest.outlet_temp_f if sat_temp else None,
            }

        visual_data = None
        if input_data.visual_inspection:
            visual_data = {
                "visible_steam_discharge": input_data.visual_inspection.visible_steam_discharge,
                "condensate_visible": input_data.visual_inspection.condensate_visible,
                "trap_cycling_observed": input_data.visual_inspection.trap_cycling_observed,
                "leaks_detected": input_data.visual_inspection.leaks_detected,
            }

        # Run diagnostics engine
        result = self.diagnostics_engine.diagnose(
            trap_type=input_data.trap_info.trap_type,
            ultrasonic_data=ultrasonic_data,
            temperature_data=temperature_data,
            visual_data=visual_data,
            trap_type_config=trap_type_config,
        )

        return result

    def _calculate_steam_loss(
        self,
        input_data: TrapDiagnosticInput,
        condition: TrapCondition,
    ) -> SteamLossEstimate:
        """
        Calculate steam loss for failed traps.

        Uses Napier's equation for orifice flow:
        W = 51.45 * A * P (lb/hr)

        This is a ZERO-HALLUCINATION calculation using deterministic formulas.

        Args:
            input_data: Diagnostic input
            condition: Trap condition from diagnostics

        Returns:
            SteamLossEstimate with hourly/annual losses and costs
        """
        # No loss for good traps
        if condition.status == TrapStatus.GOOD:
            return SteamLossEstimate(
                calculation_method="none",
                operating_hours_per_year=self.steam_config.economics.operating_hours_per_year,
            )

        # Get orifice size
        orifice_size_in = input_data.trap_info.orifice_size_in
        if orifice_size_in is None:
            # Estimate from connection size
            orifice_size_in = input_data.trap_info.connection_size_in * 0.25

        # Calculate using economic analyzer
        loss_result = self.economic_analyzer.calculate_trap_loss(
            orifice_diameter_in=orifice_size_in,
            steam_pressure_psig=input_data.steam_pressure_psig,
            failure_mode=condition.status.value,
            operating_hours=self.steam_config.economics.operating_hours_per_year,
        )

        return SteamLossEstimate(
            steam_loss_lb_hr=loss_result.get("steam_loss_lb_hr", 0.0),
            steam_loss_lb_year=loss_result.get("steam_loss_lb_year", 0.0),
            energy_loss_mmbtu_hr=loss_result.get("energy_loss_mmbtu_hr", 0.0),
            energy_loss_mmbtu_year=loss_result.get("energy_loss_mmbtu_year", 0.0),
            cost_per_hour_usd=loss_result.get("cost_per_hour_usd", 0.0),
            cost_per_year_usd=loss_result.get("cost_per_year_usd", 0.0),
            co2_emissions_lb_hr=loss_result.get("co2_emissions_lb_hr", 0.0),
            co2_emissions_tons_year=loss_result.get("co2_emissions_tons_year", 0.0),
            calculation_method="napier_orifice_flow",
            orifice_diameter_in=orifice_size_in,
            operating_hours_per_year=self.steam_config.economics.operating_hours_per_year,
        )

    def _calculate_health_score(
        self,
        input_data: TrapDiagnosticInput,
        condition: TrapCondition,
        steam_loss: SteamLossEstimate,
    ) -> TrapHealthScore:
        """
        Calculate overall trap health score (0-100).

        Scoring algorithm:
        - Good trap: Base 100, adjusted by age and maintenance
        - Failed open: 0-20 based on severity
        - Failed closed: 10-30 based on severity
        - Leaking: 20-50 based on leak rate

        Args:
            input_data: Diagnostic input
            condition: Trap condition
            steam_loss: Steam loss estimate

        Returns:
            TrapHealthScore with category and component scores
        """
        # Base score by status
        status_scores = {
            TrapStatus.GOOD: 100.0,
            TrapStatus.LEAKING: 40.0,
            TrapStatus.FAILED_OPEN: 10.0,
            TrapStatus.FAILED_CLOSED: 20.0,
            TrapStatus.COLD: 30.0,
            TrapStatus.FLOODED: 25.0,
            TrapStatus.UNKNOWN: 50.0,
        }

        base_score = status_scores.get(condition.status, 50.0)

        # Adjust based on confidence
        confidence_multipliers = {
            DiagnosisConfidence.HIGH: 1.0,
            DiagnosisConfidence.MEDIUM: 0.9,
            DiagnosisConfidence.LOW: 0.8,
            DiagnosisConfidence.UNCERTAIN: 0.7,
        }
        confidence_mult = confidence_multipliers.get(condition.confidence, 0.8)

        # Component scores
        thermal_score = 100.0
        if condition.status in {TrapStatus.FAILED_OPEN, TrapStatus.LEAKING}:
            # Reduce thermal score based on steam loss
            loss_penalty = min(steam_loss.steam_loss_lb_hr * 2, 90)
            thermal_score = max(10.0, 100.0 - loss_penalty)

        mechanical_score = 100.0
        if condition.status != TrapStatus.GOOD:
            mechanical_score = 50.0  # Failed internals
        if input_data.visual_inspection:
            if input_data.visual_inspection.trap_body_condition == "damaged":
                mechanical_score = max(20.0, mechanical_score - 30)
            if input_data.visual_inspection.leaks_detected:
                mechanical_score = max(10.0, mechanical_score - 40)

        operational_score = base_score

        # Calculate overall score
        overall_score = (
            thermal_score * 0.4 +
            mechanical_score * 0.3 +
            operational_score * 0.3
        ) * confidence_mult

        # Determine category
        if overall_score >= 90:
            category = "excellent"
        elif overall_score >= 75:
            category = "good"
        elif overall_score >= 50:
            category = "fair"
        elif overall_score >= 25:
            category = "poor"
        else:
            category = "critical"

        # Estimate days to critical (simple linear model)
        days_to_critical = None
        if overall_score > 25:
            # Assume 1% degradation per month if not repaired
            degradation_rate = 1.0  # points per month
            points_to_critical = overall_score - 25
            days_to_critical = int((points_to_critical / degradation_rate) * 30)

        return TrapHealthScore(
            overall_score=round(overall_score, 1),
            category=category,
            thermal_efficiency_score=round(thermal_score, 1),
            mechanical_condition_score=round(mechanical_score, 1),
            operational_score=round(operational_score, 1),
            trend=TrendDirection.STABLE if condition.status == TrapStatus.GOOD else TrendDirection.DEGRADING,
            days_to_critical=days_to_critical,
        )

    def _get_failure_probabilities(
        self, condition: TrapCondition
    ) -> List[FailureModeProbability]:
        """
        Get detailed failure mode probabilities.

        Args:
            condition: Trap condition from diagnostics

        Returns:
            List of failure mode probabilities with explanations
        """
        probs = []

        # Failed open probability
        probs.append(FailureModeProbability(
            failure_mode="failed_open",
            probability=condition.failed_open_probability,
            confidence=condition.confidence,
            indicators=[ev for ev in condition.evidence if "open" in ev.lower() or "blow" in ev.lower()],
            contradictors=[ic for ic in condition.inconsistencies if "open" in ic.lower()],
        ))

        # Failed closed probability
        probs.append(FailureModeProbability(
            failure_mode="failed_closed",
            probability=condition.failed_closed_probability,
            confidence=condition.confidence,
            indicators=[ev for ev in condition.evidence if "closed" in ev.lower() or "blocked" in ev.lower()],
            contradictors=[ic for ic in condition.inconsistencies if "closed" in ic.lower()],
        ))

        # Leaking probability
        probs.append(FailureModeProbability(
            failure_mode="leaking",
            probability=condition.leaking_probability,
            confidence=condition.confidence,
            indicators=[ev for ev in condition.evidence if "leak" in ev.lower()],
            contradictors=[ic for ic in condition.inconsistencies if "leak" in ic.lower()],
        ))

        return probs

    def _generate_recommendations(
        self,
        input_data: TrapDiagnosticInput,
        condition: TrapCondition,
        steam_loss: SteamLossEstimate,
    ) -> List[MaintenanceRecommendation]:
        """
        Generate maintenance recommendations based on diagnosis.

        Args:
            input_data: Diagnostic input
            condition: Trap condition
            steam_loss: Steam loss estimate

        Returns:
            Prioritized list of maintenance recommendations
        """
        recommendations = []

        if condition.status == TrapStatus.GOOD:
            # Good trap - schedule next survey
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.SCHEDULED,
                action="ROUTINE_SURVEY",
                description="Schedule routine survey per DOE annual recommendation",
                reason="Trap operating correctly",
                deadline_hours=None,
            ))
            return recommendations

        # Failed trap - prioritize by severity
        if condition.status == TrapStatus.FAILED_OPEN:
            priority = MaintenancePriority.URGENT
            if steam_loss.cost_per_year_usd > 10000:
                priority = MaintenancePriority.EMERGENCY

            recommendations.append(MaintenanceRecommendation(
                priority=priority,
                action="REPLACE_TRAP",
                description=f"Replace failed-open trap. Steam loss: {steam_loss.steam_loss_lb_hr:.1f} lb/hr",
                reason=f"Failed open condition causing ${steam_loss.cost_per_year_usd:,.0f}/year in losses",
                deadline_hours=24 if priority == MaintenancePriority.EMERGENCY else 48,
                estimated_duration_hours=self.steam_config.economics.average_repair_hours,
                estimated_cost_usd=self.steam_config.economics.average_replacement_cost_usd,
                parts_required=["Replacement steam trap", "Gaskets", "Pipe dope"],
                potential_savings_usd=steam_loss.cost_per_year_usd,
            ))

        elif condition.status == TrapStatus.FAILED_CLOSED:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.HIGH,
                action="REPAIR_OR_REPLACE",
                description="Repair or replace blocked trap to restore condensate drainage",
                reason="Failed closed condition causing condensate backup and potential waterhammer",
                deadline_hours=48,
                estimated_duration_hours=self.steam_config.economics.average_repair_hours,
                estimated_cost_usd=self.steam_config.economics.average_repair_cost_usd,
                parts_required=["Trap internals or replacement trap", "Strainer screen", "Gaskets"],
            ))

        elif condition.status == TrapStatus.LEAKING:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.MEDIUM,
                action="REPAIR_TRAP",
                description=f"Repair leaking trap. Steam loss: {steam_loss.steam_loss_lb_hr:.1f} lb/hr",
                reason=f"Leaking trap causing ${steam_loss.cost_per_year_usd:,.0f}/year in losses",
                deadline_hours=168,  # 1 week
                estimated_duration_hours=self.steam_config.economics.average_repair_hours * 0.75,
                estimated_cost_usd=self.steam_config.economics.average_repair_cost_usd * 0.5,
                parts_required=["Valve seat or disc", "Gaskets"],
                potential_savings_usd=steam_loss.cost_per_year_usd,
            ))

        elif condition.status in {TrapStatus.COLD, TrapStatus.FLOODED}:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.HIGH,
                action="INVESTIGATE",
                description="Investigate cold/flooded condition - check isolation valves and strainer",
                reason="No flow detected - may be isolated or blocked upstream",
                deadline_hours=48,
                estimated_duration_hours=1.0,
                parts_required=["May need strainer cleaning or valve repair"],
            ))

        # Add insulation recommendation if damaged
        if input_data.visual_inspection:
            if input_data.visual_inspection.insulation_condition in ["damaged", "missing"]:
                recommendations.append(MaintenanceRecommendation(
                    priority=MaintenancePriority.LOW,
                    action="REPAIR_INSULATION",
                    description="Repair or replace damaged insulation",
                    reason="Missing insulation increases heat loss and diagnostic difficulty",
                    parts_required=["Insulation material", "Jacketing"],
                ))

        return recommendations

    # =========================================================================
    # COMPLIANCE METHODS
    # =========================================================================

    def _check_asme_compliance(self, input_data: TrapDiagnosticInput) -> bool:
        """
        Check ASME B16.34 pressure-temperature compliance.

        Args:
            input_data: Diagnostic input

        Returns:
            True if compliant, False otherwise
        """
        if not self.steam_config.asme_b16_34_compliance:
            return True  # Skip if not required

        # Get operating conditions
        pressure = input_data.steam_pressure_psig
        temperature = self.steam_config.get_saturation_temperature(pressure)

        # Get trap rating
        trap_rating = input_data.trap_info.pressure_rating_psig

        # Check if operating pressure exceeds rating
        if pressure > trap_rating * 0.9:  # 90% margin
            logger.warning(
                f"Trap {input_data.trap_info.trap_id} operating near pressure rating: "
                f"{pressure} psig vs {trap_rating} psig rated"
            )
            return False

        return self.trap_classifier.check_asme_b16_34_compliance(
            trap_rating_class=150 if trap_rating <= 285 else 300,
            operating_temp_f=temperature,
            operating_pressure_psig=pressure,
        )

    def _check_pressure_rating(self, input_data: TrapDiagnosticInput) -> bool:
        """
        Check if trap pressure rating is adequate.

        Args:
            input_data: Diagnostic input

        Returns:
            True if adequate, False otherwise
        """
        operating_pressure = input_data.steam_pressure_psig
        rating = input_data.trap_info.pressure_rating_psig

        # Require 10% margin
        return rating >= operating_pressure * 1.1

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_methods_used(self, input_data: TrapDiagnosticInput) -> List[str]:
        """Get list of diagnostic methods used."""
        methods = []
        if input_data.ultrasonic_readings:
            methods.append("ultrasonic")
        if input_data.temperature_readings:
            methods.append("temperature_differential")
        if input_data.visual_inspection:
            methods.append("visual_inspection")
        return methods

    def _assess_data_quality(self, input_data: TrapDiagnosticInput) -> float:
        """
        Assess overall sensor data quality.

        Args:
            input_data: Diagnostic input

        Returns:
            Quality score 0.0 to 1.0
        """
        quality_scores = []

        for reading in input_data.ultrasonic_readings:
            quality_scores.append(reading.quality_score)

        for reading in input_data.temperature_readings:
            quality_scores.append(reading.quality_score)

        if not quality_scores:
            return 0.5  # Unknown quality

        return sum(quality_scores) / len(quality_scores)

    # =========================================================================
    # BATCH PROCESSING METHODS
    # =========================================================================

    def process_batch(
        self,
        inputs: List[TrapDiagnosticInput],
        batch_size: int = 100,
    ) -> List[TrapDiagnosticOutput]:
        """
        Process multiple traps in batch.

        Args:
            inputs: List of diagnostic inputs
            batch_size: Maximum batch size for memory efficiency

        Returns:
            List of diagnostic outputs
        """
        results = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]

            for input_data in batch:
                try:
                    result = self.process(input_data)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Batch processing failed for trap {input_data.trap_info.trap_id}: {e}"
                    )
                    # Continue with next trap

        return results

    # =========================================================================
    # CONDENSATE LOAD METHODS
    # =========================================================================

    def calculate_condensate_load(
        self, input_data: CondensateLoadInput
    ) -> CondensateLoadOutput:
        """
        Calculate condensate load for trap sizing.

        Args:
            input_data: Condensate load calculation input

        Returns:
            CondensateLoadOutput with startup/operating loads
        """
        start_time = time.perf_counter()

        result = self.condensate_calc.calculate(
            application=input_data.application,
            steam_pressure_psig=input_data.steam_pressure_psig,
            pipe_diameter_in=input_data.pipe_diameter_in,
            pipe_length_ft=input_data.pipe_length_ft,
            pipe_material=input_data.pipe_material,
            heat_transfer_rate_btu_hr=input_data.heat_transfer_rate_btu_hr,
            ambient_temperature_f=input_data.ambient_temperature_f,
            insulation_thickness_in=input_data.insulation_thickness_in,
            insulation_type=input_data.insulation_type,
            calculate_startup=input_data.calculate_startup,
            calculate_operating=input_data.calculate_operating,
            startup_time_minutes=input_data.startup_time_minutes,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        output = CondensateLoadOutput(
            request_id=input_data.request_id,
            startup_load_lb_hr=result.get("startup_load_lb_hr", 0.0),
            operating_load_lb_hr=result.get("operating_load_lb_hr", 0.0),
            peak_load_lb_hr=result.get("peak_load_lb_hr", 0.0),
            safety_factor=result.get("safety_factor", 2.0),
            design_load_lb_hr=result.get("design_load_lb_hr", 0.0),
            recommended_trap_capacity_lb_hr=result.get("recommended_capacity_lb_hr", 0.0),
            recommended_trap_types=result.get("recommended_trap_types", []),
            calculation_method=result.get("calculation_method", ""),
            formula_reference=result.get("formula_reference", ""),
            warnings=result.get("warnings", []),
        )

        if self.steam_config.provenance_tracking:
            output.provenance_hash = self.calculate_provenance_hash(input_data, output)

        return output

    # =========================================================================
    # SURVEY METHODS
    # =========================================================================

    def plan_survey(self, input_data: TrapSurveyInput) -> SurveyRouteOutput:
        """
        Plan optimized survey routes using TSP.

        Args:
            input_data: Survey planning input

        Returns:
            Optimized survey routes
        """
        return self.survey_manager.plan_survey(
            trap_ids=input_data.trap_ids,
            trap_locations=input_data.trap_locations,
            trap_areas=input_data.trap_areas,
            max_traps_per_route=input_data.max_traps_per_route,
            available_hours=input_data.available_hours,
            minutes_per_trap=input_data.minutes_per_trap,
        )

    # =========================================================================
    # ECONOMIC ANALYSIS METHODS
    # =========================================================================

    def analyze_economics(
        self,
        diagnostic_results: List[TrapDiagnosticOutput],
    ) -> EconomicAnalysisOutput:
        """
        Analyze economic impact of trap failures.

        Args:
            diagnostic_results: List of diagnostic outputs

        Returns:
            Economic analysis with ROI calculations
        """
        return self.economic_analyzer.analyze_portfolio(diagnostic_results)

    # =========================================================================
    # STATUS SUMMARY METHODS
    # =========================================================================

    def get_plant_summary(self) -> TrapStatusSummary:
        """
        Get summary of trap population status.

        Returns:
            TrapStatusSummary with population statistics
        """
        return self.population_manager.get_summary(
            plant_id=self.steam_config.plant_id
        )

    # =========================================================================
    # AGENT STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        avg_time = (
            self._total_processing_time_ms / self._processing_count
            if self._processing_count > 0
            else 0.0
        )

        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "agent_version": self.AGENT_VERSION,
            "plant_id": self.steam_config.plant_id,
            "processing_count": self._processing_count,
            "average_processing_time_ms": round(avg_time, 2),
            "total_processing_time_ms": round(self._total_processing_time_ms, 2),
            "failures_detected": self._failures_detected,
            "total_steam_loss_lb_hr": round(self._total_steam_loss_lb_hr, 2),
            "state": self.state.name,
            "is_ready": self.is_ready,
            "is_safe": self.is_safe,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_steam_trap_monitor(
    plant_id: str,
    steam_pressure_psig: float = 150.0,
    steam_cost_per_mlb: float = 12.50,
    **kwargs,
) -> SteamTrapMonitorAgent:
    """
    Factory function to create a configured Steam Trap Monitor agent.

    Args:
        plant_id: Plant/facility identifier
        steam_pressure_psig: Operating steam pressure
        steam_cost_per_mlb: Steam cost per 1000 lb
        **kwargs: Additional configuration parameters

    Returns:
        Configured SteamTrapMonitorAgent instance

    Example:
        >>> agent = create_steam_trap_monitor(
        ...     plant_id="PLANT-001",
        ...     steam_pressure_psig=150.0,
        ...     steam_cost_per_mlb=15.00,
        ... )
    """
    # Create economics config
    economics = EconomicsConfig(
        steam_cost_per_mlb=steam_cost_per_mlb,
        steam_pressure_psig=steam_pressure_psig,
        **{k: v for k, v in kwargs.items() if hasattr(EconomicsConfig, k)},
    )

    # Create main config
    config = SteamTrapMonitorConfig(
        plant_id=plant_id,
        steam_pressure_psig=steam_pressure_psig,
        economics=economics,
    )

    return SteamTrapMonitorAgent(config)
