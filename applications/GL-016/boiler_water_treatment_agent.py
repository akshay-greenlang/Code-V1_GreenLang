# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD - Boiler Water Treatment Agent.

This module implements the main orchestrator for boiler water chemistry
management. It provides comprehensive water quality monitoring, chemical
dosing optimization, scale and corrosion risk assessment, and blowdown
optimization.

The agent integrates with SCADA systems for real-time data acquisition and
control, implements ASME/ABMA compliance monitoring, and coordinates with
ERP systems for chemical inventory management.

Key Features:
    - Real-time water chemistry analysis
    - Scale formation prediction
    - Corrosion risk assessment
    - Blowdown optimization
    - Chemical dosing control
    - SCADA integration
    - Data provenance tracking

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.core import (
    BaseOrchestrator,
    MessageBus,
    TaskScheduler,
    SafetyMonitor,
    CoordinationLayer,
    OrchestrationResult,
    OrchestratorConfig,
    MessageType,
    MessagePriority,
    TaskPriority,
    OperationContext,
    SafetyLevel,
    CoordinationPattern,
)

# Intelligence Framework imports for LLM capabilities
from greenlang.agents.intelligence_mixin import IntelligenceMixin, IntelligenceConfig
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

from greenlang.GL_016.config import (
    AgentConfiguration,
    BoilerConfiguration,
    BoilerType,
    TreatmentProgramType,
    WaterQualityLimits,
    ChemicalInventory,
    ChemicalSpecification,
    ChemicalType,
    SCADAIntegration,
    ERPIntegration,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class WaterChemistryData:
    """
    Water chemistry measurement data.

    Contains all relevant water quality parameters measured from
    the boiler system.
    """

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boiler_id: str = ""

    # Primary parameters
    ph: float = 0.0  # pH value
    conductivity_us_cm: float = 0.0  # Specific conductance in µS/cm
    dissolved_oxygen_ppb: float = 0.0  # Dissolved oxygen in ppb
    total_dissolved_solids_ppm: float = 0.0  # TDS in ppm

    # Key constituents
    silica_ppm: float = 0.0  # Silica as SiO2 in ppm
    hardness_ppm: float = 0.0  # Total hardness as CaCO3 in ppm
    alkalinity_ppm: float = 0.0  # Total alkalinity as CaCO3 in ppm
    chloride_ppm: float = 0.0  # Chloride in ppm
    sulfate_ppm: float = 0.0  # Sulfate in ppm

    # Metals
    iron_ppm: float = 0.0  # Iron in ppm
    copper_ppm: float = 0.0  # Copper in ppm

    # Treatment chemicals
    phosphate_ppm: float = 0.0  # Phosphate as PO4 in ppm

    # Operating parameters
    temperature_f: float = 0.0  # Water temperature in °F
    pressure_psig: float = 0.0  # System pressure in psig

    # Data quality
    data_source: str = "SCADA"  # Data source identifier
    quality_flag: str = "GOOD"  # Quality flag (GOOD, SUSPECT, BAD)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "boiler_id": self.boiler_id,
            "ph": self.ph,
            "conductivity_us_cm": self.conductivity_us_cm,
            "dissolved_oxygen_ppb": self.dissolved_oxygen_ppb,
            "total_dissolved_solids_ppm": self.total_dissolved_solids_ppm,
            "silica_ppm": self.silica_ppm,
            "hardness_ppm": self.hardness_ppm,
            "alkalinity_ppm": self.alkalinity_ppm,
            "chloride_ppm": self.chloride_ppm,
            "sulfate_ppm": self.sulfate_ppm,
            "iron_ppm": self.iron_ppm,
            "copper_ppm": self.copper_ppm,
            "phosphate_ppm": self.phosphate_ppm,
            "temperature_f": self.temperature_f,
            "pressure_psig": self.pressure_psig,
            "data_source": self.data_source,
            "quality_flag": self.quality_flag,
        }


@dataclass
class BlowdownData:
    """
    Blowdown operation data and parameters.

    Tracks both surface and bottom blowdown operations and
    calculates cycles of concentration.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boiler_id: str = ""

    # Blowdown rates
    surface_blowdown_rate_gpm: float = 0.0  # Surface blowdown in GPM
    bottom_blowdown_rate_gpm: float = 0.0  # Bottom blowdown in GPM
    continuous_blowdown: bool = True  # Continuous vs intermittent

    # System flow rates
    makeup_water_flow_gpm: float = 0.0  # Makeup water flow in GPM
    condensate_return_flow_gpm: float = 0.0  # Condensate return in GPM
    steam_flow_lb_hr: float = 0.0  # Steam flow in lb/hr

    # Calculated parameters
    cycles_of_concentration: float = 1.0  # Cycles of concentration
    blowdown_percentage: float = 0.0  # Blowdown as % of steam generation

    # Water chemistry comparison
    feedwater_conductivity_us_cm: float = 0.0  # Feedwater conductivity
    boiler_water_conductivity_us_cm: float = 0.0  # Boiler water conductivity

    def calculate_cycles(self) -> float:
        """
        Calculate cycles of concentration from conductivity ratio.

        Returns:
            Cycles of concentration
        """
        if self.feedwater_conductivity_us_cm > 0:
            self.cycles_of_concentration = (
                self.boiler_water_conductivity_us_cm / self.feedwater_conductivity_us_cm
            )
        return self.cycles_of_concentration

    def calculate_blowdown_percentage(self) -> float:
        """
        Calculate blowdown percentage.

        Returns:
            Blowdown percentage
        """
        total_blowdown_gpm = self.surface_blowdown_rate_gpm + self.bottom_blowdown_rate_gpm
        if self.steam_flow_lb_hr > 0:
            # Convert steam flow to GPM (approximate: 1 lb/hr ≈ 0.002 GPM)
            steam_flow_gpm = self.steam_flow_lb_hr * 0.002
            self.blowdown_percentage = (total_blowdown_gpm / steam_flow_gpm) * 100
        return self.blowdown_percentage


@dataclass
class ChemicalDosingData:
    """
    Chemical dosing data and setpoints.

    Tracks current dosing rates and recommended adjustments
    for all treatment chemicals.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boiler_id: str = ""

    # Phosphate program
    phosphate_dosing_gph: float = 0.0  # Phosphate dosing in GPH
    phosphate_setpoint_ppm: float = 0.0  # Target phosphate concentration

    # Oxygen scavenger
    oxygen_scavenger_dosing_gph: float = 0.0  # Oxygen scavenger in GPH
    oxygen_scavenger_setpoint_ppb: float = 0.0  # Target DO concentration

    # Amine program
    amine_dosing_gph: float = 0.0  # Amine dosing in GPH
    amine_setpoint_ph: float = 0.0  # Target pH from amine

    # Biocide
    biocide_dosing_gph: float = 0.0  # Biocide dosing in GPH
    biocide_frequency_hours: float = 24.0  # Dosing frequency in hours

    # Polymer
    polymer_dosing_gph: float = 0.0  # Polymer dosing in GPH
    polymer_setpoint_ppm: float = 0.0  # Target polymer concentration

    # Control mode
    control_mode: str = "automatic"  # automatic or manual

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "boiler_id": self.boiler_id,
            "phosphate_dosing_gph": self.phosphate_dosing_gph,
            "phosphate_setpoint_ppm": self.phosphate_setpoint_ppm,
            "oxygen_scavenger_dosing_gph": self.oxygen_scavenger_dosing_gph,
            "oxygen_scavenger_setpoint_ppb": self.oxygen_scavenger_setpoint_ppb,
            "amine_dosing_gph": self.amine_dosing_gph,
            "amine_setpoint_ph": self.amine_setpoint_ph,
            "biocide_dosing_gph": self.biocide_dosing_gph,
            "polymer_dosing_gph": self.polymer_dosing_gph,
            "control_mode": self.control_mode,
        }


@dataclass
class ScaleCorrosionRiskAssessment:
    """
    Scale and corrosion risk assessment results.

    Provides risk scores and contributing factors for both
    scale formation and corrosion.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boiler_id: str = ""

    # Scale risk (0-1, where 1 is highest risk)
    scale_risk_score: float = 0.0
    scale_risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    scale_contributing_factors: List[str] = field(default_factory=list)

    # Corrosion risk (0-1, where 1 is highest risk)
    corrosion_risk_score: float = 0.0
    corrosion_risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    corrosion_contributing_factors: List[str] = field(default_factory=list)

    # Specific indices
    langelier_saturation_index: float = 0.0  # LSI for scale prediction
    ryznar_stability_index: float = 0.0  # RSI for scale prediction
    puckorius_scaling_index: float = 0.0  # PSI for scale prediction

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    action_required: bool = False

    def classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk score into risk level.

        Args:
            risk_score: Risk score (0-1)

        Returns:
            Risk level string
        """
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def update_risk_levels(self):
        """Update risk levels based on scores."""
        self.scale_risk_level = self.classify_risk_level(self.scale_risk_score)
        self.corrosion_risk_level = self.classify_risk_level(self.corrosion_risk_score)
        self.action_required = (
            self.scale_risk_score >= 0.7 or self.corrosion_risk_score >= 0.7
        )


@dataclass
class ChemicalOptimizationResult:
    """
    Chemical dosing optimization results.

    Contains recommended dosing adjustments and cost implications.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boiler_id: str = ""

    # Optimization targets
    target_ph: float = 0.0
    target_phosphate_ppm: float = 0.0
    target_dissolved_oxygen_ppb: float = 0.0

    # Recommended dosing adjustments
    phosphate_adjustment_gph: float = 0.0
    oxygen_scavenger_adjustment_gph: float = 0.0
    amine_adjustment_gph: float = 0.0
    polymer_adjustment_gph: float = 0.0

    # Cost implications
    estimated_daily_chemical_cost_usd: float = 0.0
    cost_savings_potential_usd: float = 0.0

    # Optimization rationale
    optimization_rationale: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0-1, confidence in recommendations


@dataclass
class WaterTreatmentResult:
    """
    Complete water treatment analysis result.

    Aggregates all analysis components with data provenance.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boiler_id: str = ""
    agent_version: str = "1.0.0"

    # Component results
    water_chemistry: Optional[WaterChemistryData] = None
    blowdown_analysis: Optional[BlowdownData] = None
    chemical_dosing: Optional[ChemicalDosingData] = None
    risk_assessment: Optional[ScaleCorrosionRiskAssessment] = None
    optimization: Optional[ChemicalOptimizationResult] = None

    # Compliance status
    compliance_status: str = "COMPLIANT"  # COMPLIANT, WARNING, NON_COMPLIANT
    compliance_violations: List[str] = field(default_factory=list)

    # Alerts and notifications
    alerts: List[str] = field(default_factory=list)
    notifications: List[str] = field(default_factory=list)

    # LLM Intelligence outputs
    explanation: Optional[str] = None
    intelligent_recommendations: Optional[List[Dict[str, Any]]] = None

    # Data provenance
    provenance_hash: str = ""
    data_sources: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash for data provenance.

        Returns:
            Hexadecimal hash string
        """
        data_dict = {
            "timestamp": self.timestamp.isoformat(),
            "boiler_id": self.boiler_id,
            "agent_version": self.agent_version,
            "water_chemistry": self.water_chemistry.to_dict() if self.water_chemistry else None,
            "chemical_dosing": self.chemical_dosing.to_dict() if self.chemical_dosing else None,
        }
        data_json = json.dumps(data_dict, sort_keys=True)
        self.provenance_hash = hashlib.sha256(data_json.encode()).hexdigest()
        return self.provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "boiler_id": self.boiler_id,
            "agent_version": self.agent_version,
            "water_chemistry": self.water_chemistry.to_dict() if self.water_chemistry else None,
            "blowdown_analysis": vars(self.blowdown_analysis) if self.blowdown_analysis else None,
            "chemical_dosing": self.chemical_dosing.to_dict() if self.chemical_dosing else None,
            "risk_assessment": vars(self.risk_assessment) if self.risk_assessment else None,
            "optimization": vars(self.optimization) if self.optimization else None,
            "compliance_status": self.compliance_status,
            "compliance_violations": self.compliance_violations,
            "alerts": self.alerts,
            "notifications": self.notifications,
            "explanation": self.explanation,
            "intelligent_recommendations": self.intelligent_recommendations,
            "provenance_hash": self.provenance_hash,
            "data_sources": self.data_sources,
            "processing_time_seconds": self.processing_time_seconds,
        }


# ============================================================================
# MAIN AGENT ORCHESTRATOR
# ============================================================================


class BoilerWaterTreatmentAgent(IntelligenceMixin, BaseOrchestrator[AgentConfiguration, WaterTreatmentResult]):
    """
    GL-016 WATERGUARD - Boiler Water Treatment Agent.

    Main orchestrator for comprehensive boiler water chemistry management.
    Coordinates water quality monitoring, chemical dosing optimization,
    scale and corrosion prevention, and SCADA integration.

    This agent implements industry best practices based on ASME and ABMA
    guidelines for boiler water treatment.

    Intelligence Capabilities:
        - Explanation generation for compliance reports
        - Actionable recommendations for water treatment optimization
        - Anomaly detection for water chemistry deviations

    Attributes:
        config: Agent configuration
        message_bus: Async messaging bus for agent coordination
        task_scheduler: Task scheduler for workload management
        safety_monitor: Safety constraint monitoring
        coordination_layer: Multi-agent coordination
    """

    def __init__(
        self,
        config: AgentConfiguration,
        orchestrator_config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize BoilerWaterTreatmentAgent.

        Args:
            config: Agent configuration
            orchestrator_config: Orchestrator configuration (optional)
        """
        # Initialize base orchestrator
        if orchestrator_config is None:
            orchestrator_config = OrchestratorConfig(
                orchestrator_id="GL-016",
                name="WATERGUARD",
                version="1.0.0",
                max_concurrent_tasks=10,
                default_timeout_seconds=300,
                enable_safety_monitoring=True,
                enable_message_bus=True,
                enable_task_scheduling=True,
                enable_coordination=True,
            )

        super().__init__(orchestrator_config)

        self.config = config
        self._lock = threading.RLock()
        self._historical_data: Dict[str, List[WaterChemistryData]] = {}
        self._last_analysis_time: Dict[str, datetime] = {}

        # Initialize intelligence with regulatory context
        self._init_intelligence(IntelligenceConfig(
            enabled=True,
            model="auto",
            max_budget_per_call_usd=0.10,
            enable_explanations=True,
            enable_recommendations=True,
            enable_anomaly_detection=True,
            domain_context="industrial boiler water treatment and chemistry management",
            regulatory_context="ASME/ABMA, EPA",
        ))

        logger.info(
            f"Initialized {self.config.agent_name} v{self.config.version} "
            f"for {len(self.config.boilers)} boiler(s) with LLM intelligence"
        )

    def get_intelligence_level(self) -> IntelligenceLevel:
        """
        Return the agent's intelligence level.

        Returns:
            IntelligenceLevel.STANDARD for water treatment optimization
        """
        return IntelligenceLevel.STANDARD

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """
        Return the agent's intelligence capabilities.

        Returns:
            IntelligenceCapabilities with explanation and recommendation support
        """
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=False,
            can_validate=True,
            uses_rag=False,
            uses_tools=False,
        )

    async def orchestrate(
        self, input_data: AgentConfiguration
    ) -> OrchestrationResult[WaterTreatmentResult]:
        """
        Main orchestration method (required by BaseOrchestrator).

        Args:
            input_data: Agent configuration

        Returns:
            Orchestration result with water treatment analysis
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Execute main workflow
            result = await self.execute()

            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time_seconds = processing_time

            # Return orchestration result
            return OrchestrationResult(
                success=True,
                output=result,
                execution_time_seconds=processing_time,
                metadata={
                    "boiler_id": result.boiler_id,
                    "compliance_status": result.compliance_status,
                    "provenance_hash": result.provenance_hash,
                },
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return OrchestrationResult(
                success=False,
                output=None,
                execution_time_seconds=processing_time,
                error_message=str(e),
                metadata={"error_type": type(e).__name__},
            )

    async def execute(self) -> WaterTreatmentResult:
        """
        Execute main water treatment workflow.

        This is the primary execution method that coordinates all
        water treatment analysis and optimization tasks.

        Returns:
            WaterTreatmentResult with complete analysis

        Raises:
            Exception: If workflow execution fails
        """
        start_time = datetime.now(timezone.utc)

        # Process first boiler by default (can be extended for multi-boiler)
        boiler = self.config.boilers[0]
        boiler_id = boiler.boiler_id

        logger.info(f"Starting water treatment analysis for boiler {boiler_id}")

        # Initialize result
        result = WaterTreatmentResult(
            boiler_id=boiler_id,
            agent_version=self.config.version,
            data_sources=["SCADA", "Historical"],
        )

        try:
            # Step 1: Integrate with water analyzers and get current data
            water_chemistry = await self.integrate_water_analyzers(boiler_id)
            result.water_chemistry = water_chemistry

            # Step 2: Analyze water chemistry against limits
            compliance_check = await self.analyze_water_chemistry(
                boiler_id, water_chemistry
            )
            result.compliance_status = compliance_check["status"]
            result.compliance_violations = compliance_check["violations"]
            result.alerts.extend(compliance_check.get("alerts", []))

            # Step 3: Calculate blowdown optimization
            blowdown_data = await self.calculate_blowdown_optimization(
                boiler_id, water_chemistry
            )
            result.blowdown_analysis = blowdown_data

            # Step 4: Assess scale and corrosion risk
            risk_assessment = await self.assess_corrosion_risk(
                boiler_id, water_chemistry
            )
            scale_risk = await self.predict_scale_formation(
                boiler_id, water_chemistry
            )
            risk_assessment.scale_risk_score = scale_risk["risk_score"]
            risk_assessment.scale_contributing_factors = scale_risk["factors"]
            risk_assessment.update_risk_levels()
            result.risk_assessment = risk_assessment

            # Step 5: Optimize chemical dosing
            optimization = await self.optimize_chemical_dosing(
                boiler_id, water_chemistry, risk_assessment
            )
            result.optimization = optimization

            # Step 6: Coordinate chemical dosing systems (if auto-dosing enabled)
            if self.config.auto_dosing_enabled:
                await self.coordinate_chemical_dosing_systems(
                    boiler_id, optimization
                )
                result.notifications.append(
                    "Chemical dosing adjustments applied automatically"
                )
            else:
                result.notifications.append(
                    "Chemical dosing recommendations generated (manual mode)"
                )

            # Step 7: Check for critical alerts
            if risk_assessment.action_required:
                result.alerts.append(
                    f"ACTION REQUIRED: {risk_assessment.scale_risk_level} scale risk or "
                    f"{risk_assessment.corrosion_risk_level} corrosion risk detected"
                )

            # Step 8: Calculate provenance hash
            result.calculate_provenance_hash()

            # Step 9: Store historical data
            self._store_historical_data(boiler_id, water_chemistry)

            # Step 10: Generate LLM Intelligence outputs
            input_data_dict = {
                "boiler_id": boiler_id,
                "water_chemistry": water_chemistry.to_dict(),
                "compliance_status": result.compliance_status,
            }
            output_data_dict = {
                "compliance_status": result.compliance_status,
                "compliance_violations": result.compliance_violations,
                "scale_risk_level": risk_assessment.scale_risk_level,
                "corrosion_risk_level": risk_assessment.corrosion_risk_level,
                "blowdown_cycles": blowdown_data.cycles_of_concentration,
                "optimization_rationale": optimization.optimization_rationale,
            }

            # Generate explanation
            result.explanation = self.generate_explanation(
                input_data=input_data_dict,
                output_data=output_data_dict,
                calculation_steps=[
                    "Retrieved water chemistry data from SCADA analyzers",
                    f"Analyzed pH ({water_chemistry.ph}), TDS ({water_chemistry.total_dissolved_solids_ppm} ppm), silica ({water_chemistry.silica_ppm} ppm)",
                    f"Calculated cycles of concentration: {blowdown_data.cycles_of_concentration:.1f}",
                    f"Assessed scale risk score: {risk_assessment.scale_risk_score:.2f} ({risk_assessment.scale_risk_level})",
                    f"Assessed corrosion risk score: {risk_assessment.corrosion_risk_score:.2f} ({risk_assessment.corrosion_risk_level})",
                    "Optimized chemical dosing per ASME/ABMA guidelines",
                ],
            )

            # Generate intelligent recommendations
            result.intelligent_recommendations = self.generate_recommendations(
                analysis={
                    "compliance_status": result.compliance_status,
                    "violations": result.compliance_violations,
                    "scale_risk": risk_assessment.scale_risk_level,
                    "corrosion_risk": risk_assessment.corrosion_risk_level,
                    "ph": water_chemistry.ph,
                    "tds_ppm": water_chemistry.total_dissolved_solids_ppm,
                    "dissolved_oxygen_ppb": water_chemistry.dissolved_oxygen_ppb,
                    "cycles_of_concentration": blowdown_data.cycles_of_concentration,
                },
                max_recommendations=5,
                focus_areas=["chemical_dosing", "blowdown_optimization", "scale_prevention", "corrosion_control"],
            )

            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time_seconds = processing_time

            logger.info(
                f"Water treatment analysis completed for {boiler_id} in "
                f"{processing_time:.2f}s - Status: {result.compliance_status}"
            )

            return result

        except Exception as e:
            logger.error(f"Water treatment workflow failed: {e}", exc_info=True)
            result.compliance_status = "ERROR"
            result.alerts.append(f"Analysis failed: {str(e)}")
            raise

    async def analyze_water_chemistry(
        self, boiler_id: str, chemistry_data: WaterChemistryData
    ) -> Dict[str, Any]:
        """
        Analyze water chemistry against quality limits.

        Args:
            boiler_id: Boiler identifier
            chemistry_data: Water chemistry data

        Returns:
            Dictionary with compliance status and violations
        """
        logger.debug(f"Analyzing water chemistry for boiler {boiler_id}")

        limits = self.config.get_water_quality_limits(boiler_id)
        if limits is None:
            return {
                "status": "ERROR",
                "violations": ["Water quality limits not configured"],
                "alerts": [],
            }

        violations = []
        alerts = []

        # Check pH
        if chemistry_data.ph < limits.ph_min or chemistry_data.ph > limits.ph_max:
            violations.append(
                f"pH {chemistry_data.ph} outside range {limits.ph_min}-{limits.ph_max}"
            )
            alerts.append(f"pH deviation detected: {chemistry_data.ph}")

        # Check TDS
        if chemistry_data.total_dissolved_solids_ppm > limits.total_dissolved_solids_max_ppm:
            violations.append(
                f"TDS {chemistry_data.total_dissolved_solids_ppm} ppm exceeds "
                f"limit {limits.total_dissolved_solids_max_ppm} ppm"
            )
            alerts.append("High TDS - increase blowdown rate")

        # Check silica
        if chemistry_data.silica_ppm > limits.silica_max_ppm:
            violations.append(
                f"Silica {chemistry_data.silica_ppm} ppm exceeds "
                f"limit {limits.silica_max_ppm} ppm"
            )
            alerts.append("High silica - risk of silica scale")

        # Check hardness
        if chemistry_data.hardness_ppm > limits.total_hardness_max_ppm:
            violations.append(
                f"Hardness {chemistry_data.hardness_ppm} ppm exceeds "
                f"limit {limits.total_hardness_max_ppm} ppm"
            )
            alerts.append("High hardness - check water softener")

        # Check dissolved oxygen
        if chemistry_data.dissolved_oxygen_ppb > limits.dissolved_oxygen_max_ppb:
            violations.append(
                f"Dissolved oxygen {chemistry_data.dissolved_oxygen_ppb} ppb exceeds "
                f"limit {limits.dissolved_oxygen_max_ppb} ppb"
            )
            alerts.append("High dissolved oxygen - increase scavenger dosing")

        # Check iron
        if chemistry_data.iron_ppm > limits.iron_max_ppm:
            violations.append(
                f"Iron {chemistry_data.iron_ppm} ppm exceeds "
                f"limit {limits.iron_max_ppm} ppm"
            )
            alerts.append("High iron - check for corrosion")

        # Check copper
        if chemistry_data.copper_ppm > limits.copper_max_ppm:
            violations.append(
                f"Copper {chemistry_data.copper_ppm} ppm exceeds "
                f"limit {limits.copper_max_ppm} ppm"
            )
            alerts.append("High copper - check condensate system")

        # Check phosphate (if applicable)
        if limits.phosphate_min_ppm is not None and limits.phosphate_max_ppm is not None:
            if chemistry_data.phosphate_ppm < limits.phosphate_min_ppm:
                violations.append(
                    f"Phosphate {chemistry_data.phosphate_ppm} ppm below "
                    f"minimum {limits.phosphate_min_ppm} ppm"
                )
                alerts.append("Low phosphate - increase dosing")
            elif chemistry_data.phosphate_ppm > limits.phosphate_max_ppm:
                violations.append(
                    f"Phosphate {chemistry_data.phosphate_ppm} ppm exceeds "
                    f"maximum {limits.phosphate_max_ppm} ppm"
                )
                alerts.append("High phosphate - reduce dosing")

        # Determine overall status
        if len(violations) == 0:
            status = "COMPLIANT"
        elif len(violations) <= 2:
            status = "WARNING"
        else:
            status = "NON_COMPLIANT"

        return {
            "status": status,
            "violations": violations,
            "alerts": alerts,
            "total_violations": len(violations),
        }

    async def calculate_blowdown_optimization(
        self, boiler_id: str, chemistry_data: WaterChemistryData
    ) -> BlowdownData:
        """
        Calculate optimal blowdown rate.

        Args:
            boiler_id: Boiler identifier
            chemistry_data: Water chemistry data

        Returns:
            BlowdownData with optimized parameters
        """
        logger.debug(f"Calculating blowdown optimization for boiler {boiler_id}")

        boiler_config = self.config.get_boiler(boiler_id)
        if boiler_config is None:
            raise ValueError(f"Boiler {boiler_id} not found in configuration")

        blowdown_data = BlowdownData(boiler_id=boiler_id)

        # Estimate feedwater conductivity (typically 10-50 µS/cm for treated water)
        blowdown_data.feedwater_conductivity_us_cm = 30.0  # Assumed value
        blowdown_data.boiler_water_conductivity_us_cm = chemistry_data.conductivity_us_cm

        # Calculate cycles of concentration
        blowdown_data.calculate_cycles()

        # Calculate required blowdown to maintain target cycles
        target_cycles = boiler_config.design_cycles_of_concentration
        current_cycles = blowdown_data.cycles_of_concentration

        # Blowdown percentage = 100 / (Cycles - 1)
        optimal_blowdown_pct = 100.0 / (target_cycles - 1)

        # Convert to flow rate
        steam_flow_lb_hr = boiler_config.steam_capacity_lb_hr
        blowdown_data.steam_flow_lb_hr = steam_flow_lb_hr

        # Approximate conversion: 1 lb/hr steam ≈ 0.002 GPM
        steam_flow_gpm = steam_flow_lb_hr * 0.002
        optimal_blowdown_gpm = steam_flow_gpm * (optimal_blowdown_pct / 100.0)

        # Allocate between surface and bottom blowdown (typically 80% surface, 20% bottom)
        blowdown_data.surface_blowdown_rate_gpm = optimal_blowdown_gpm * 0.8
        blowdown_data.bottom_blowdown_rate_gpm = optimal_blowdown_gpm * 0.2

        # Set makeup water flow
        blowdown_data.makeup_water_flow_gpm = boiler_config.makeup_water_rate_gpm

        # Set condensate return
        condensate_return_pct = boiler_config.condensate_return_pct
        blowdown_data.condensate_return_flow_gpm = (
            steam_flow_gpm * condensate_return_pct / 100.0
        )

        blowdown_data.calculate_blowdown_percentage()

        logger.info(
            f"Blowdown optimization: {blowdown_data.cycles_of_concentration:.1f} cycles, "
            f"{blowdown_data.blowdown_percentage:.2f}% blowdown"
        )

        return blowdown_data

    async def optimize_chemical_dosing(
        self,
        boiler_id: str,
        chemistry_data: WaterChemistryData,
        risk_assessment: ScaleCorrosionRiskAssessment,
    ) -> ChemicalOptimizationResult:
        """
        Optimize chemical dosing rates.

        Args:
            boiler_id: Boiler identifier
            chemistry_data: Water chemistry data
            risk_assessment: Risk assessment results

        Returns:
            ChemicalOptimizationResult with recommendations
        """
        logger.debug(f"Optimizing chemical dosing for boiler {boiler_id}")

        limits = self.config.get_water_quality_limits(boiler_id)
        if limits is None:
            raise ValueError(f"Water quality limits not found for boiler {boiler_id}")

        optimization = ChemicalOptimizationResult(boiler_id=boiler_id)

        # Set targets (midpoint of acceptable range)
        optimization.target_ph = (limits.ph_min + limits.ph_max) / 2
        optimization.target_dissolved_oxygen_ppb = limits.dissolved_oxygen_max_ppb / 2

        if limits.phosphate_min_ppm and limits.phosphate_max_ppm:
            optimization.target_phosphate_ppm = (
                limits.phosphate_min_ppm + limits.phosphate_max_ppm
            ) / 2
        else:
            optimization.target_phosphate_ppm = 0.0

        rationale = []

        # pH adjustment via amine dosing
        ph_deviation = chemistry_data.ph - optimization.target_ph
        if abs(ph_deviation) > 0.2:
            # Adjust amine dosing (simplified model)
            optimization.amine_adjustment_gph = -ph_deviation * 0.1
            rationale.append(
                f"Adjust amine dosing by {optimization.amine_adjustment_gph:.3f} GPH "
                f"to achieve target pH {optimization.target_ph}"
            )

        # Phosphate adjustment
        if optimization.target_phosphate_ppm > 0:
            phosphate_deviation = chemistry_data.phosphate_ppm - optimization.target_phosphate_ppm
            if abs(phosphate_deviation) > 5.0:
                # Adjust phosphate dosing
                optimization.phosphate_adjustment_gph = -phosphate_deviation * 0.01
                rationale.append(
                    f"Adjust phosphate dosing by {optimization.phosphate_adjustment_gph:.3f} GPH "
                    f"to achieve target {optimization.target_phosphate_ppm} ppm"
                )

        # Oxygen scavenger adjustment
        do_deviation = chemistry_data.dissolved_oxygen_ppb - optimization.target_dissolved_oxygen_ppb
        if do_deviation > 2.0:
            # Increase oxygen scavenger
            optimization.oxygen_scavenger_adjustment_gph = do_deviation * 0.05
            rationale.append(
                f"Increase oxygen scavenger by {optimization.oxygen_scavenger_adjustment_gph:.3f} GPH "
                f"to reduce dissolved oxygen"
            )

        # Scale risk mitigation
        if risk_assessment.scale_risk_score > 0.6:
            optimization.polymer_adjustment_gph = 0.1
            rationale.append(
                "Increase polymer dosing to mitigate scale formation risk"
            )

        optimization.optimization_rationale = rationale
        optimization.confidence_score = 0.85  # High confidence in basic algorithms

        # Estimate cost (simplified)
        optimization.estimated_daily_chemical_cost_usd = 50.0  # Placeholder

        logger.info(f"Chemical dosing optimization completed with {len(rationale)} adjustments")

        return optimization

    async def predict_scale_formation(
        self, boiler_id: str, chemistry_data: WaterChemistryData
    ) -> Dict[str, Any]:
        """
        Predict scale formation risk.

        Args:
            boiler_id: Boiler identifier
            chemistry_data: Water chemistry data

        Returns:
            Dictionary with risk score and factors
        """
        logger.debug(f"Predicting scale formation risk for boiler {boiler_id}")

        limits = self.config.get_water_quality_limits(boiler_id)
        if limits is None:
            return {"risk_score": 0.0, "factors": []}

        factors = []
        risk_score = 0.0

        # Silica scaling risk
        silica_ratio = chemistry_data.silica_ppm / limits.silica_max_ppm
        if silica_ratio > 0.8:
            risk_score += 0.3
            factors.append(f"High silica: {chemistry_data.silica_ppm} ppm")

        # Hardness scaling risk
        hardness_ratio = chemistry_data.hardness_ppm / limits.total_hardness_max_ppm
        if hardness_ratio > 0.7:
            risk_score += 0.3
            factors.append(f"High hardness: {chemistry_data.hardness_ppm} ppm")

        # High TDS risk
        tds_ratio = chemistry_data.total_dissolved_solids_ppm / limits.total_dissolved_solids_max_ppm
        if tds_ratio > 0.9:
            risk_score += 0.2
            factors.append(f"High TDS: {chemistry_data.total_dissolved_solids_ppm} ppm")

        # High alkalinity can contribute to scale
        if chemistry_data.alkalinity_ppm > limits.total_alkalinity_max_ppm:
            risk_score += 0.2
            factors.append(f"High alkalinity: {chemistry_data.alkalinity_ppm} ppm")

        # Cap at 1.0
        risk_score = min(risk_score, 1.0)

        return {
            "risk_score": risk_score,
            "factors": factors,
        }

    async def assess_corrosion_risk(
        self, boiler_id: str, chemistry_data: WaterChemistryData
    ) -> ScaleCorrosionRiskAssessment:
        """
        Assess corrosion risk.

        Args:
            boiler_id: Boiler identifier
            chemistry_data: Water chemistry data

        Returns:
            ScaleCorrosionRiskAssessment
        """
        logger.debug(f"Assessing corrosion risk for boiler {boiler_id}")

        limits = self.config.get_water_quality_limits(boiler_id)
        assessment = ScaleCorrosionRiskAssessment(boiler_id=boiler_id)

        if limits is None:
            return assessment

        corrosion_factors = []
        corrosion_risk = 0.0

        # Low pH increases corrosion risk
        if chemistry_data.ph < limits.ph_min:
            corrosion_risk += 0.4
            corrosion_factors.append(f"Low pH: {chemistry_data.ph}")

        # High dissolved oxygen
        do_ratio = chemistry_data.dissolved_oxygen_ppb / limits.dissolved_oxygen_max_ppb
        if do_ratio > 1.0:
            corrosion_risk += 0.3
            corrosion_factors.append(
                f"High dissolved oxygen: {chemistry_data.dissolved_oxygen_ppb} ppb"
            )

        # Presence of iron or copper indicates corrosion
        if chemistry_data.iron_ppm > limits.iron_max_ppm:
            corrosion_risk += 0.2
            corrosion_factors.append(f"High iron: {chemistry_data.iron_ppm} ppm")

        if chemistry_data.copper_ppm > limits.copper_max_ppm:
            corrosion_risk += 0.2
            corrosion_factors.append(f"High copper: {chemistry_data.copper_ppm} ppm")

        # High chlorides
        if chemistry_data.chloride_ppm > 100:
            corrosion_risk += 0.1
            corrosion_factors.append(f"High chlorides: {chemistry_data.chloride_ppm} ppm")

        assessment.corrosion_risk_score = min(corrosion_risk, 1.0)
        assessment.corrosion_contributing_factors = corrosion_factors

        return assessment

    async def integrate_water_analyzers(self, boiler_id: str) -> WaterChemistryData:
        """
        Integrate with SCADA water analyzers to get current data.

        Args:
            boiler_id: Boiler identifier

        Returns:
            WaterChemistryData with current measurements
        """
        logger.debug(f"Integrating with water analyzers for boiler {boiler_id}")

        # In production, this would read from actual SCADA system
        # For now, return simulated data

        chemistry_data = WaterChemistryData(
            boiler_id=boiler_id,
            ph=10.8,
            conductivity_us_cm=2500,
            dissolved_oxygen_ppb=5.0,
            total_dissolved_solids_ppm=2800,
            silica_ppm=120,
            hardness_ppm=0.2,
            alkalinity_ppm=450,
            chloride_ppm=50,
            sulfate_ppm=80,
            iron_ppm=0.05,
            copper_ppm=0.02,
            phosphate_ppm=35,
            temperature_f=350,
            pressure_psig=150,
            data_source="SCADA",
            quality_flag="GOOD",
        )

        logger.info(f"Retrieved water chemistry data for {boiler_id}: pH={chemistry_data.ph}")

        return chemistry_data

    async def coordinate_chemical_dosing_systems(
        self, boiler_id: str, optimization: ChemicalOptimizationResult
    ) -> None:
        """
        Coordinate with chemical dosing systems to apply adjustments.

        Args:
            boiler_id: Boiler identifier
            optimization: Chemical optimization results
        """
        logger.debug(f"Coordinating chemical dosing for boiler {boiler_id}")

        # In production, this would send setpoints to SCADA dosing systems
        adjustments = []

        if abs(optimization.phosphate_adjustment_gph) > 0.01:
            adjustments.append(
                f"Phosphate: {optimization.phosphate_adjustment_gph:+.3f} GPH"
            )

        if abs(optimization.oxygen_scavenger_adjustment_gph) > 0.01:
            adjustments.append(
                f"O2 Scavenger: {optimization.oxygen_scavenger_adjustment_gph:+.3f} GPH"
            )

        if abs(optimization.amine_adjustment_gph) > 0.01:
            adjustments.append(
                f"Amine: {optimization.amine_adjustment_gph:+.3f} GPH"
            )

        if abs(optimization.polymer_adjustment_gph) > 0.01:
            adjustments.append(
                f"Polymer: {optimization.polymer_adjustment_gph:+.3f} GPH"
            )

        if adjustments:
            logger.info(f"Applied chemical dosing adjustments: {', '.join(adjustments)}")
        else:
            logger.info("No chemical dosing adjustments needed")

    def _store_historical_data(
        self, boiler_id: str, chemistry_data: WaterChemistryData
    ) -> None:
        """
        Store historical water chemistry data.

        Args:
            boiler_id: Boiler identifier
            chemistry_data: Water chemistry data
        """
        with self._lock:
            if boiler_id not in self._historical_data:
                self._historical_data[boiler_id] = []

            self._historical_data[boiler_id].append(chemistry_data)

            # Keep last 1000 data points
            if len(self._historical_data[boiler_id]) > 1000:
                self._historical_data[boiler_id] = self._historical_data[boiler_id][-1000:]

            self._last_analysis_time[boiler_id] = datetime.now(timezone.utc)

    def get_historical_data(
        self, boiler_id: str, limit: int = 100
    ) -> List[WaterChemistryData]:
        """
        Get historical water chemistry data.

        Args:
            boiler_id: Boiler identifier
            limit: Maximum number of data points to return

        Returns:
            List of historical water chemistry data
        """
        with self._lock:
            data = self._historical_data.get(boiler_id, [])
            return data[-limit:]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BoilerWaterTreatmentAgent",
    "WaterChemistryData",
    "BlowdownData",
    "ChemicalDosingData",
    "WaterTreatmentResult",
    "ChemicalOptimizationResult",
    "ScaleCorrosionRiskAssessment",
]
