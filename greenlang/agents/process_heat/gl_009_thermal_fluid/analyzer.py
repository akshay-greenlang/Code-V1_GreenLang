"""
GL-009 THERMALIQ Agent - Main Thermal Fluid Analyzer

This module provides the main ThermalFluidAnalyzer class that orchestrates
all thermal fluid system analysis including property calculations, exergy
analysis, degradation monitoring, expansion tank sizing, heat transfer
analysis, and safety monitoring.

Features:
    - Comprehensive thermal fluid property calculations
    - Exergy (2nd Law) efficiency analysis
    - Fluid degradation monitoring with remaining life estimation
    - Expansion tank sizing validation per API 660
    - Heat transfer coefficient calculations
    - High temperature safety interlock monitoring
    - SHA-256 provenance tracking for audit trails
    - Zero-hallucination deterministic calculations

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid import (
    ...     ThermalFluidAnalyzer,
    ...     ThermalFluidConfig,
    ...     create_default_config,
    ... )
    >>> config = create_default_config(system_id="TF-001")
    >>> analyzer = ThermalFluidAnalyzer(config)
    >>> result = analyzer.process(input_data)
    >>> print(f"Exergy efficiency: {result.exergy_analysis.exergy_efficiency_pct}%")
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import hashlib
import json
import logging
import time

from pydantic import BaseModel, Field

# Import from shared base
from ..shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig,
    AgentCapability,
    SafetyLevel,
    ProcessingError,
    ValidationError,
)

# Import local modules
from .config import (
    ThermalFluidConfig,
    create_default_config,
)
from .schemas import (
    ThermalFluidType,
    ThermalFluidInput,
    ThermalFluidOutput,
    FluidProperties,
    ExergyAnalysis,
    DegradationAnalysis,
    HeatTransferAnalysis,
    ExpansionTankSizing,
    SafetyAnalysis,
    OptimizationRecommendation,
    FluidLabAnalysis,
    ExpansionTankData,
    OptimizationStatus,
    SafetyStatus,
)
from .fluid_properties import ThermalFluidPropertyDatabase
from .exergy import ExergyAnalyzer
from .degradation import DegradationMonitor
from .expansion_tank import ExpansionTankAnalyzer
from .heat_transfer import HeatTransferCalculator
from .safety import SafetyMonitor

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class ThermalFluidAnalyzer(BaseProcessHeatAgent[ThermalFluidInput, ThermalFluidOutput]):
    """
    GL-009 THERMALIQ Thermal Fluid Systems Analyzer.

    This agent provides comprehensive analysis of thermal fluid (hot oil)
    systems including exergy-based efficiency analysis, fluid degradation
    monitoring, expansion tank sizing, heat transfer calculations, and
    safety interlock monitoring.

    Unlike steam systems which use 1st Law (energy) efficiency, thermal
    fluid systems benefit from 2nd Law (exergy) analysis to identify
    irreversibilities and optimization opportunities.

    Capabilities:
        - Thermal fluid property database (20+ fluids)
        - Exergy (2nd Law) efficiency analysis
        - Fluid degradation monitoring (viscosity, flash point, TAN, etc.)
        - Expansion tank sizing per API 660
        - Heat transfer coefficient calculations (Dittus-Boelter, Gnielinski)
        - Film temperature monitoring
        - Flash point and auto-ignition safety margins
        - SIL-2 safety interlock recommendations

    All calculations are deterministic with zero hallucination and complete
    SHA-256 provenance tracking for regulatory compliance.

    Example:
        >>> config = create_default_config(
        ...     system_id="TF-001",
        ...     fluid_type=ThermalFluidType.THERMINOL_66,
        ...     design_temperature_f=600.0,
        ... )
        >>> analyzer = ThermalFluidAnalyzer(config)
        >>> result = analyzer.process(input_data)
        >>> print(f"Safety status: {result.safety_analysis.safety_status}")
    """

    # Agent metadata
    AGENT_TYPE = "GL-009"
    AGENT_NAME = "THERMALIQ Thermal Fluid Analyzer"
    AGENT_VERSION = "1.0.0"

    def __init__(
        self,
        config: ThermalFluidConfig,
        safety_level: SafetyLevel = SafetyLevel.SIL_2,
    ) -> None:
        """
        Initialize the Thermal Fluid Analyzer.

        Args:
            config: Thermal fluid system configuration
            safety_level: Safety Integrity Level (default SIL-2)
        """
        # Create agent config for base class
        agent_config = AgentConfig(
            agent_id=config.agent_id,
            agent_type=self.AGENT_TYPE,
            name=self.AGENT_NAME,
            version=self.AGENT_VERSION,
            capabilities={
                AgentCapability.REAL_TIME_MONITORING,
                AgentCapability.OPTIMIZATION,
                AgentCapability.COMPLIANCE_REPORTING,
                AgentCapability.PREDICTIVE_ANALYTICS,
            },
        )

        # Initialize base class
        super().__init__(agent_config, safety_level)

        # Store configuration
        self.tf_config = config

        # Initialize sub-analyzers
        self._init_property_database()
        self._init_exergy_analyzer()
        self._init_degradation_monitor()
        self._init_expansion_tank_analyzer()
        self._init_heat_transfer_calculator()
        self._init_safety_monitor()

        # Calculation counter for provenance
        self._calculation_count = 0

        logger.info(
            f"ThermalFluidAnalyzer initialized: {config.system_id}, "
            f"fluid={config.fluid_type}, design_temp={config.design_temperature_f}F"
        )

    def _init_property_database(self) -> None:
        """Initialize fluid property database."""
        self.property_db = ThermalFluidPropertyDatabase()

    def _init_exergy_analyzer(self) -> None:
        """Initialize exergy analyzer."""
        self.exergy_analyzer = ExergyAnalyzer(
            reference_temp_f=self.tf_config.exergy.reference_temperature_f,
            fluid_type=self.tf_config.fluid_type,
        )

    def _init_degradation_monitor(self) -> None:
        """Initialize degradation monitor."""
        self.degradation_monitor = DegradationMonitor(
            fluid_type=self.tf_config.fluid_type,
            thresholds=self.tf_config.degradation,
        )

    def _init_expansion_tank_analyzer(self) -> None:
        """Initialize expansion tank analyzer."""
        self.expansion_analyzer = ExpansionTankAnalyzer(
            fluid_type=self.tf_config.fluid_type,
            pump_config=self.tf_config.pump,
            tank_config=self.tf_config.expansion_tank,
        )

    def _init_heat_transfer_calculator(self) -> None:
        """Initialize heat transfer calculator."""
        self.heat_transfer_calc = HeatTransferCalculator(
            fluid_type=self.tf_config.fluid_type,
        )

    def _init_safety_monitor(self) -> None:
        """Initialize safety monitor."""
        self.safety_monitor = SafetyMonitor(
            fluid_type=self.tf_config.fluid_type,
            config=self.tf_config.safety,
        )

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def process(
        self,
        input_data: ThermalFluidInput,
    ) -> ThermalFluidOutput:
        """
        Main processing method for thermal fluid analysis.

        This method orchestrates all sub-analyzers to provide comprehensive
        thermal fluid system analysis with exergy efficiency, degradation
        assessment, and safety monitoring.

        Args:
            input_data: Validated input data with current operating conditions

        Returns:
            ThermalFluidOutput with complete analysis

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        start_time = time.time()
        self._calculation_count = 0

        logger.info(f"Processing thermal fluid analysis for {input_data.system_id}")

        try:
            with self.safety_guard():
                # Step 1: Validate input
                if not self.validate_input(input_data):
                    raise ValidationError("Input validation failed")

                # Step 2: Get fluid properties
                fluid_props = self._get_fluid_properties(input_data)
                self._calculation_count += 1

                # Step 3: Safety analysis (always first priority)
                safety_analysis = self._perform_safety_analysis(input_data)
                self._calculation_count += 1

                # Step 4: Exergy analysis
                exergy_analysis = None
                if self.tf_config.exergy.enabled:
                    exergy_analysis = self._perform_exergy_analysis(input_data)
                    self._calculation_count += 1

                # Step 5: Heat transfer analysis
                heat_transfer_analysis = self._perform_heat_transfer_analysis(input_data)
                self._calculation_count += 1

                # Step 6: Expansion tank analysis
                expansion_analysis = None
                if input_data.expansion_tank_level_pct is not None:
                    expansion_analysis = self._perform_expansion_analysis(input_data)
                    self._calculation_count += 1

                # Step 7: Calculate KPIs
                kpis = self._calculate_kpis(
                    input_data,
                    fluid_props,
                    exergy_analysis,
                    safety_analysis,
                )

                # Step 8: Generate recommendations
                recommendations = self._generate_recommendations(
                    input_data,
                    fluid_props,
                    exergy_analysis,
                    heat_transfer_analysis,
                    expansion_analysis,
                    safety_analysis,
                )

                # Step 9: Collect alerts and warnings
                alerts, warnings = self._collect_alerts_warnings(
                    safety_analysis,
                    heat_transfer_analysis,
                    expansion_analysis,
                )

                # Step 10: Determine overall status
                overall_status = self._determine_overall_status(
                    safety_analysis,
                    heat_transfer_analysis,
                )

                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000

                # Calculate provenance hash
                provenance_hash = self._calculate_provenance_hash(input_data)

                # Create output
                output = ThermalFluidOutput(
                    system_id=input_data.system_id,
                    status="success",
                    overall_status=overall_status,
                    processing_time_ms=round(processing_time_ms, 2),
                    fluid_properties=fluid_props,
                    exergy_analysis=exergy_analysis,
                    degradation_analysis=None,  # Requires lab data
                    heat_transfer_analysis=heat_transfer_analysis,
                    expansion_tank_analysis=expansion_analysis,
                    safety_analysis=safety_analysis,
                    recommendations=recommendations,
                    kpis=kpis,
                    alerts=alerts,
                    warnings=warnings,
                    provenance_hash=provenance_hash,
                    calculation_count=self._calculation_count,
                    metadata={
                        "fluid_type": self.tf_config.fluid_type.value,
                        "design_temperature_f": self.tf_config.design_temperature_f,
                        "exergy_enabled": self.tf_config.exergy.enabled,
                    },
                )

                # Validate output
                if not self.validate_output(output):
                    raise ProcessingError("Output validation failed")

                logger.info(
                    f"Thermal fluid analysis complete: {overall_status.value}, "
                    f"{self._calculation_count} calculations, "
                    f"{processing_time_ms:.1f}ms"
                )

                return output

        except Exception as e:
            logger.error(f"Thermal fluid analysis failed: {e}", exc_info=True)
            raise ProcessingError(f"Thermal fluid analysis failed: {str(e)}") from e

    def validate_input(self, input_data: ThermalFluidInput) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input to validate

        Returns:
            True if valid
        """
        errors = []

        # Check temperature range
        max_bulk = self.property_db.get_max_bulk_temp(self.tf_config.fluid_type)
        if input_data.bulk_temperature_f > max_bulk:
            errors.append(
                f"Bulk temp {input_data.bulk_temperature_f}F exceeds "
                f"max {max_bulk}F for {self.tf_config.fluid_type}"
            )

        # Check flow rate
        if input_data.flow_rate_gpm <= 0:
            errors.append("Flow rate must be positive")

        if errors:
            for error in errors:
                logger.warning(f"Validation error: {error}")
            return False

        return True

    def validate_output(self, output_data: ThermalFluidOutput) -> bool:
        """
        Validate output data.

        Args:
            output_data: Output to validate

        Returns:
            True if valid
        """
        # Check for required fields
        if output_data.fluid_properties is None:
            logger.warning("Missing fluid properties in output")
            return False

        if output_data.safety_analysis is None:
            logger.warning("Missing safety analysis in output")
            return False

        # Check provenance hash
        if not output_data.provenance_hash:
            logger.warning("Missing provenance hash")
            return False

        return True

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def _get_fluid_properties(
        self,
        input_data: ThermalFluidInput,
    ) -> FluidProperties:
        """Get fluid properties at operating conditions."""
        return self.property_db.get_properties(
            fluid_type=self.tf_config.fluid_type,
            temperature_f=input_data.bulk_temperature_f,
        )

    def _perform_safety_analysis(
        self,
        input_data: ThermalFluidInput,
    ) -> SafetyAnalysis:
        """Perform safety interlock analysis."""
        return self.safety_monitor.analyze(input_data)

    def _perform_exergy_analysis(
        self,
        input_data: ThermalFluidInput,
    ) -> Optional[ExergyAnalysis]:
        """Perform exergy (2nd Law) analysis."""
        if input_data.inlet_temperature_f is None or input_data.outlet_temperature_f is None:
            # Use bulk temp with assumed delta-T
            hot_temp = input_data.bulk_temperature_f
            cold_temp = input_data.bulk_temperature_f - 50.0
        else:
            hot_temp = input_data.outlet_temperature_f
            cold_temp = input_data.inlet_temperature_f

        # Estimate heat duty if not provided
        if input_data.heater_duty_btu_hr is None:
            props = self.property_db.get_properties(
                self.tf_config.fluid_type,
                input_data.bulk_temperature_f
            )
            # Q = m_dot * Cp * dT
            # Convert GPM to lb/hr: GPM * 8.33 * SG * 60
            sg = props.density_lb_ft3 / 62.4
            mass_flow = input_data.flow_rate_gpm * 8.33 * sg * 60
            heat_duty = mass_flow * props.specific_heat_btu_lb_f * abs(hot_temp - cold_temp)
        else:
            heat_duty = input_data.heater_duty_btu_hr

        return self.exergy_analyzer.analyze_system(
            hot_temp_f=hot_temp,
            cold_temp_f=cold_temp,
            heat_duty_btu_hr=heat_duty,
            heater_efficiency_pct=85.0,  # Typical value
        )

    def _perform_heat_transfer_analysis(
        self,
        input_data: ThermalFluidInput,
    ) -> Optional[HeatTransferAnalysis]:
        """Perform heat transfer coefficient analysis."""
        # Estimate velocity from flow rate and pipe size
        pipe_id_in = self.tf_config.piping.main_header_size_in * 0.9  # Approximate ID

        # Area in ft2
        import math
        area_ft2 = math.pi * (pipe_id_in / 24) ** 2

        # Flow in ft3/s
        flow_ft3_s = input_data.flow_rate_gpm / 7.48 / 60

        velocity_ft_s = flow_ft3_s / area_ft2

        return self.heat_transfer_calc.calculate_film_coefficient(
            temperature_f=input_data.bulk_temperature_f,
            velocity_ft_s=velocity_ft_s,
            tube_id_in=self.tf_config.heater.coil_tube_id_in,
        )

    def _perform_expansion_analysis(
        self,
        input_data: ThermalFluidInput,
    ) -> Optional[ExpansionTankSizing]:
        """Perform expansion tank analysis."""
        return self.expansion_analyzer.analyze(
            tank_volume_gallons=self.tf_config.expansion_tank.volume_gallons,
            system_volume_gallons=self.tf_config.system_volume_gallons,
            cold_temp_f=70.0,  # Standard cold fill
            hot_temp_f=input_data.bulk_temperature_f,
            current_level_pct=input_data.expansion_tank_level_pct,
        )

    def analyze_degradation(
        self,
        lab_analysis: FluidLabAnalysis,
        baseline: Optional[FluidLabAnalysis] = None,
    ) -> DegradationAnalysis:
        """
        Analyze fluid degradation from laboratory results.

        This method is called separately when lab data is available.

        Args:
            lab_analysis: Current laboratory analysis
            baseline: Optional baseline (new fluid) analysis

        Returns:
            DegradationAnalysis with condition assessment
        """
        return self.degradation_monitor.analyze(lab_analysis, baseline)

    # =========================================================================
    # KPI AND RECOMMENDATION METHODS
    # =========================================================================

    def _calculate_kpis(
        self,
        input_data: ThermalFluidInput,
        fluid_props: FluidProperties,
        exergy_analysis: Optional[ExergyAnalysis],
        safety_analysis: SafetyAnalysis,
    ) -> Dict[str, float]:
        """Calculate key performance indicators."""
        kpis = {
            "bulk_temperature_f": round(input_data.bulk_temperature_f, 1),
            "flow_rate_gpm": round(input_data.flow_rate_gpm, 1),
            "fluid_viscosity_cst": round(fluid_props.kinematic_viscosity_cst, 2),
            "fluid_density_lb_ft3": round(fluid_props.density_lb_ft3, 2),
            "prandtl_number": round(fluid_props.prandtl_number, 1),
            "bulk_temp_margin_f": round(safety_analysis.bulk_temp_margin_f, 1),
            "film_temp_margin_f": round(safety_analysis.film_temp_margin_f, 1),
            "flash_point_margin_f": round(safety_analysis.flash_point_margin_f, 1),
        }

        if exergy_analysis:
            kpis["exergy_efficiency_pct"] = round(exergy_analysis.exergy_efficiency_pct, 1)
            kpis["carnot_efficiency_pct"] = round(exergy_analysis.carnot_efficiency_pct, 1)
            kpis["exergy_destruction_btu_hr"] = round(exergy_analysis.exergy_destruction_btu_hr, 0)

        return kpis

    def _generate_recommendations(
        self,
        input_data: ThermalFluidInput,
        fluid_props: FluidProperties,
        exergy_analysis: Optional[ExergyAnalysis],
        heat_transfer: Optional[HeatTransferAnalysis],
        expansion: Optional[ExpansionTankSizing],
        safety: SafetyAnalysis,
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []

        # Safety recommendations
        for rec in safety.safety_recommendations:
            recommendations.append(OptimizationRecommendation(
                category="safety",
                priority=1,
                title="Safety Recommendation",
                description=rec,
            ))

        # Exergy recommendations
        if exergy_analysis and exergy_analysis.exergy_efficiency_pct < 40:
            potential_savings = (
                exergy_analysis.exergy_destruction_btu_hr *
                0.20 *  # 20% improvement potential
                8000 *  # operating hours
                self.tf_config.fuel_cost_usd_mmbtu /
                1_000_000
            )
            recommendations.append(OptimizationRecommendation(
                category="efficiency",
                priority=2,
                title="Improve Exergy Efficiency",
                description=(
                    f"Current exergy efficiency {exergy_analysis.exergy_efficiency_pct:.1f}% "
                    "is below target. Consider heat integration opportunities."
                ),
                current_value=exergy_analysis.exergy_efficiency_pct,
                recommended_value=50.0,
                estimated_annual_savings_usd=round(potential_savings, 0),
            ))

        # Heat transfer recommendations
        if heat_transfer and heat_transfer.warnings:
            for warning in heat_transfer.warnings:
                recommendations.append(OptimizationRecommendation(
                    category="heat_transfer",
                    priority=2,
                    title="Heat Transfer Issue",
                    description=warning,
                ))

        # Expansion tank recommendations
        if expansion and expansion.recommendations:
            for rec in expansion.recommendations:
                recommendations.append(OptimizationRecommendation(
                    category="expansion_tank",
                    priority=3,
                    title="Expansion Tank",
                    description=rec,
                ))

        # High viscosity recommendation
        if fluid_props.kinematic_viscosity_cst > 50:
            recommendations.append(OptimizationRecommendation(
                category="operations",
                priority=3,
                title="High Viscosity",
                description=(
                    f"Fluid viscosity {fluid_props.kinematic_viscosity_cst:.1f} cSt "
                    "is elevated. Consider increasing temperature or fluid changeout."
                ),
                current_value=fluid_props.kinematic_viscosity_cst,
            ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations

    def _collect_alerts_warnings(
        self,
        safety: SafetyAnalysis,
        heat_transfer: Optional[HeatTransferAnalysis],
        expansion: Optional[ExpansionTankSizing],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Collect all alerts and warnings."""
        alerts = []
        warnings = []

        # Safety alerts/trips
        for trip in safety.active_trips:
            alerts.append({
                "type": "TRIP",
                "severity": "critical",
                "message": trip,
            })

        for alarm in safety.active_alarms:
            if "CRITICAL" in alarm:
                alerts.append({
                    "type": "ALARM",
                    "severity": "high",
                    "message": alarm,
                })
            else:
                warnings.append(alarm)

        # Heat transfer warnings
        if heat_transfer:
            warnings.extend(heat_transfer.warnings)

        # Expansion tank warnings
        if expansion and not expansion.sizing_adequate:
            warnings.append(
                f"Expansion tank undersized: {expansion.actual_volume_gallons:.0f} gal "
                f"vs {expansion.required_volume_gallons:.0f} gal required"
            )

        return alerts, warnings

    def _determine_overall_status(
        self,
        safety: SafetyAnalysis,
        heat_transfer: Optional[HeatTransferAnalysis],
    ) -> OptimizationStatus:
        """Determine overall system status."""
        if safety.safety_status in [SafetyStatus.TRIP, SafetyStatus.EMERGENCY_SHUTDOWN]:
            return OptimizationStatus.CRITICAL

        if safety.safety_status == SafetyStatus.ALARM:
            return OptimizationStatus.CRITICAL

        if safety.safety_status == SafetyStatus.WARNING:
            return OptimizationStatus.SUBOPTIMAL

        if heat_transfer and heat_transfer.warnings:
            return OptimizationStatus.SUBOPTIMAL

        return OptimizationStatus.OPTIMAL

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _calculate_provenance_hash(
        self,
        input_data: ThermalFluidInput,
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "agent_id": self.tf_config.agent_id,
            "agent_type": self.AGENT_TYPE,
            "agent_version": self.AGENT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_id": input_data.system_id,
            "fluid_type": self.tf_config.fluid_type.value,
            "bulk_temperature_f": input_data.bulk_temperature_f,
            "flow_rate_gpm": input_data.flow_rate_gpm,
            "calculation_count": self._calculation_count,
        }

        data_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_fluid_properties(
        self,
        temperature_f: float,
    ) -> FluidProperties:
        """
        Get fluid properties at specified temperature.

        Args:
            temperature_f: Temperature (F)

        Returns:
            FluidProperties at temperature
        """
        return self.property_db.get_properties(
            self.tf_config.fluid_type,
            temperature_f,
        )

    def calculate_exergy_efficiency(
        self,
        hot_temp_f: float,
        cold_temp_f: float,
        heat_duty_btu_hr: float,
    ) -> ExergyAnalysis:
        """
        Calculate exergy efficiency for given conditions.

        Args:
            hot_temp_f: Hot supply temperature (F)
            cold_temp_f: Cold return temperature (F)
            heat_duty_btu_hr: Heat duty (BTU/hr)

        Returns:
            ExergyAnalysis with results
        """
        return self.exergy_analyzer.analyze_system(
            hot_temp_f=hot_temp_f,
            cold_temp_f=cold_temp_f,
            heat_duty_btu_hr=heat_duty_btu_hr,
        )

    def check_safety_status(
        self,
        input_data: ThermalFluidInput,
    ) -> SafetyAnalysis:
        """
        Perform quick safety status check.

        Args:
            input_data: Current operating data

        Returns:
            SafetyAnalysis with status
        """
        return self.safety_monitor.analyze(input_data)
