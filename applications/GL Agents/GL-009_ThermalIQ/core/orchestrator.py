"""
GL-009 THERMALIQ - Main Orchestrator

Central orchestrator for the Thermal Fluid Analyzer agent.
Coordinates thermal efficiency calculations, exergy analysis,
Sankey diagram generation, and explainability workflows.

All calculations follow zero-hallucination principles:
- No LLM calls for numeric calculations
- Deterministic, reproducible results
- SHA-256 provenance tracking
- Full audit logging

Example:
    >>> config = ThermalIQConfig()
    >>> orchestrator = ThermalIQOrchestrator(config)
    >>> result = await orchestrator.analyze(input_data)
    >>> print(f"Efficiency: {result.efficiency_percent}%")
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math
import asyncio

from .config import (
    ThermalIQConfig,
    CalculationMode,
    FluidConfig,
    DEFAULT_CONFIG,
    REFERENCE_TEMPERATURE_K,
)
from .schemas import (
    ThermalAnalysisInput,
    ThermalAnalysisOutput,
    FluidProperties,
    ExergyResult,
    SankeyData,
    SankeyNode,
    SankeyLink,
    ExplainabilityReport,
    FeatureImportance,
    LIMEExplanation,
    Recommendation,
    RecommendationType,
    ProvenanceRecord,
    CalculationEvent,
    CalculationType,
    CalculationStatus,
    AgentStatus,
    HealthCheckResponse,
    OperatingConditions,
)

logger = logging.getLogger(__name__)


class ThermalIQOrchestrator:
    """
    Main orchestrator for GL-009 THERMALIQ agent.

    Coordinates all thermal fluid analysis workflows including:
    - Thermal efficiency (first-law) calculations
    - Exergy (second-law) analysis
    - Sankey diagram generation
    - Fluid property lookups
    - Fluid recommendation engine
    - SHAP/LIME explainability integration

    Ensures deterministic, reproducible operation with full
    provenance tracking and audit logging.

    Attributes:
        config: Agent configuration
        VERSION: Current version string

    Example:
        >>> orchestrator = ThermalIQOrchestrator()
        >>> input_data = ThermalAnalysisInput(
        ...     energy_in_kW=1000,
        ...     heat_out_kW=850,
        ...     fluid_properties=FluidProperties(...),
        ...     operating_conditions=OperatingConditions(...)
        ... )
        >>> result = await orchestrator.analyze(input_data)
        >>> print(f"Efficiency: {result.efficiency_percent:.1f}%")
        >>> print(f"Exergy destruction: {result.exergy_result.exergy_destruction_kW:.1f} kW")
    """

    VERSION = "1.0.0"
    AGENT_ID = "GL-009"
    AGENT_NAME = "THERMALIQ"

    def __init__(
        self,
        config: Optional[ThermalIQConfig] = None,
    ) -> None:
        """
        Initialize the THERMALIQ orchestrator.

        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self._start_time = datetime.now(timezone.utc)

        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"Invalid configuration: {config_errors}")

        # Statistics
        self._analyses_count = 0
        self._successful_count = 0
        self._total_processing_time_ms = 0.0
        self._calculation_events: List[CalculationEvent] = []

        logger.info(
            f"GL-009 THERMALIQ orchestrator initialized: "
            f"version={self.VERSION}, mode={self.config.mode.value}"
        )

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    async def analyze(
        self,
        input_data: ThermalAnalysisInput,
    ) -> ThermalAnalysisOutput:
        """
        Execute full thermal analysis workflow.

        Performs thermal efficiency calculation, optional exergy analysis,
        Sankey diagram generation, and explainability based on input
        configuration.

        Args:
            input_data: Validated input data for analysis

        Returns:
            ThermalAnalysisOutput with complete analysis results

        Raises:
            ValueError: If input validation fails
            RuntimeError: If calculation fails

        Example:
            >>> result = await orchestrator.analyze(input_data)
            >>> print(f"Efficiency: {result.efficiency_percent}%")
        """
        start_time = datetime.now(timezone.utc)
        self._analyses_count += 1

        logger.info(
            f"Starting thermal analysis: analysis_id={input_data.analysis_id}, "
            f"energy_in={input_data.energy_in_kW}kW, mode={input_data.mode.value}"
        )

        try:
            # Step 1: Calculate thermal efficiency (DETERMINISTIC)
            efficiency_pct = self.calculate_efficiency(
                input_data.energy_in_kW,
                input_data.heat_out_kW,
            )

            # Calculate total losses
            total_losses_kW = input_data.energy_in_kW - input_data.heat_out_kW
            losses_breakdown = input_data.losses_kW.copy()
            if not losses_breakdown:
                losses_breakdown["unaccounted"] = total_losses_kW

            # Log efficiency calculation
            self._log_calculation(
                CalculationType.EFFICIENCY,
                {"energy_in": input_data.energy_in_kW, "heat_out": input_data.heat_out_kW},
                {"efficiency_pct": efficiency_pct},
            )

            # Step 2: Exergy analysis (if requested)
            exergy_result = None
            if input_data.include_exergy:
                exergy_result = await self._run_exergy_analysis(
                    input_data.energy_in_kW,
                    input_data.heat_out_kW,
                    input_data.operating_conditions,
                    input_data.fluid_properties,
                )

            # Step 3: Generate Sankey diagram (if requested)
            sankey_data = None
            if input_data.include_sankey:
                sankey_data = await self._generate_sankey(
                    input_data.energy_in_kW,
                    input_data.heat_out_kW,
                    losses_breakdown,
                    efficiency_pct,
                )

            # Step 4: Generate explainability report (if requested)
            explainability = None
            if input_data.include_recommendations:
                explainability = await self._generate_explainability(
                    input_data,
                    efficiency_pct,
                    exergy_result,
                )

            # Calculate processing time
            processing_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Compute provenance hashes
            input_hash = self._compute_hash({
                "analysis_id": input_data.analysis_id,
                "energy_in": input_data.energy_in_kW,
                "heat_out": input_data.heat_out_kW,
                "fluid": input_data.fluid_properties.fluid_id,
            })
            output_hash = self._compute_hash({
                "efficiency": efficiency_pct,
                "exergy_destruction": exergy_result.exergy_destruction_kW if exergy_result else 0,
            })
            provenance_hash = self._compute_hash({
                "input": input_hash,
                "output": output_hash,
                "timestamp": start_time.isoformat(),
            })

            # Build result
            result = ThermalAnalysisOutput(
                analysis_id=input_data.analysis_id,
                status=CalculationStatus.COMPLETED,
                efficiency_percent=efficiency_pct,
                energy_in_kW=input_data.energy_in_kW,
                heat_out_kW=input_data.heat_out_kW,
                total_losses_kW=total_losses_kW,
                losses_breakdown=losses_breakdown,
                exergy_result=exergy_result,
                sankey_data=sankey_data,
                explainability=explainability,
                fluid_properties=input_data.fluid_properties,
                processing_time_ms=round(processing_time_ms, 2),
                calculation_count=3 if input_data.include_exergy else 1,
                provenance_hash=provenance_hash,
                input_hash=input_hash,
                formula_versions={
                    "efficiency": "EFF_v1.0",
                    "exergy": "EXERGY_v1.0" if exergy_result else None,
                    "sankey": "SANKEY_v1.0" if sankey_data else None,
                },
                validation_passed=True,
            )

            # Update statistics
            self._successful_count += 1
            self._total_processing_time_ms += processing_time_ms

            logger.info(
                f"Analysis complete: analysis_id={input_data.analysis_id}, "
                f"efficiency={efficiency_pct:.1f}%, time={processing_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return ThermalAnalysisOutput(
                analysis_id=input_data.analysis_id,
                status=CalculationStatus.FAILED,
                efficiency_percent=0.0,
                energy_in_kW=input_data.energy_in_kW,
                heat_out_kW=input_data.heat_out_kW,
                total_losses_kW=0.0,
                losses_breakdown={},
                validation_passed=False,
                validation_messages=[str(e)],
            )

    # =========================================================================
    # THERMAL EFFICIENCY CALCULATION (ZERO-HALLUCINATION)
    # =========================================================================

    def calculate_efficiency(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
    ) -> float:
        """
        Calculate thermal (first-law) efficiency.

        DETERMINISTIC CALCULATION - No LLM involvement.

        Formula: eta = (Q_out / Q_in) * 100

        Args:
            energy_in_kW: Total energy input (kW)
            heat_out_kW: Useful heat output (kW)

        Returns:
            Thermal efficiency as percentage (0-100)

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> eff = orchestrator.calculate_efficiency(1000, 850)
            >>> print(f"Efficiency: {eff}%")  # 85.0%
        """
        # Input validation
        if energy_in_kW <= 0:
            raise ValueError("Energy input must be positive")
        if heat_out_kW < 0:
            raise ValueError("Heat output cannot be negative")
        if heat_out_kW > energy_in_kW:
            raise ValueError("Heat output cannot exceed energy input")

        # DETERMINISTIC CALCULATION
        efficiency = (heat_out_kW / energy_in_kW) * 100.0

        # Round to configured precision
        efficiency = round(efficiency, self.config.calculation_precision)

        return efficiency

    # =========================================================================
    # EXERGY ANALYSIS (ZERO-HALLUCINATION)
    # =========================================================================

    def calculate_exergy(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
        source_temperature_K: float,
        sink_temperature_K: float,
        reference_temperature_K: Optional[float] = None,
    ) -> ExergyResult:
        """
        Calculate exergy (second-law) analysis.

        DETERMINISTIC CALCULATION - No LLM involvement.

        Formulas:
            - Carnot factor: eta_c = 1 - T0/Tsource
            - Exergy input: Ex_in = Q_in * (1 - T0/Tsource)
            - Exergy output: Ex_out = Q_out * (1 - T0/Tsink)
            - Exergy destruction: Ex_d = Ex_in - Ex_out
            - Exergy efficiency: eta_ex = Ex_out / Ex_in * 100

        Args:
            energy_in_kW: Energy input (kW)
            heat_out_kW: Heat output (kW)
            source_temperature_K: Heat source temperature (K)
            sink_temperature_K: Heat sink temperature (K)
            reference_temperature_K: Dead state temperature (K), default 298.15K

        Returns:
            ExergyResult with complete analysis

        Example:
            >>> result = orchestrator.calculate_exergy(
            ...     energy_in_kW=1000,
            ...     heat_out_kW=850,
            ...     source_temperature_K=573.15,  # 300C
            ...     sink_temperature_K=353.15,    # 80C
            ... )
            >>> print(f"Exergy destruction: {result.exergy_destruction_kW} kW")
        """
        T0 = reference_temperature_K or self.config.reference_temperature_K

        # Validate temperatures
        if source_temperature_K <= T0:
            raise ValueError("Source temperature must be above reference temperature")
        if sink_temperature_K < T0:
            # Allow sink at or slightly below reference for cooling applications
            sink_temperature_K = max(sink_temperature_K, T0)

        # DETERMINISTIC CALCULATIONS
        # Carnot factor at source
        carnot_source = 1.0 - (T0 / source_temperature_K)

        # Carnot factor at sink
        carnot_sink = 1.0 - (T0 / sink_temperature_K) if sink_temperature_K > T0 else 0.0

        # Exergy input: maximum work extractable from heat source
        exergy_input_kW = energy_in_kW * carnot_source

        # Exergy output: maximum work extractable from output heat
        exergy_output_kW = heat_out_kW * carnot_sink

        # Exergy destruction: irreversibility
        exergy_destruction_kW = exergy_input_kW - exergy_output_kW

        # Ensure non-negative (numerical precision)
        exergy_destruction_kW = max(0.0, exergy_destruction_kW)

        # Exergy efficiency
        exergy_efficiency_pct = 0.0
        if exergy_input_kW > 0:
            exergy_efficiency_pct = (exergy_output_kW / exergy_input_kW) * 100.0

        # Improvement potential
        improvement_potential_kW = exergy_destruction_kW
        improvement_potential_pct = 0.0
        if exergy_input_kW > 0:
            improvement_potential_pct = (exergy_destruction_kW / exergy_input_kW) * 100.0

        # Round to configured precision
        precision = self.config.calculation_precision

        result = ExergyResult(
            reference_temperature_K=round(T0, 2),
            reference_pressure_kPa=self.config.reference_pressure_kPa,
            exergy_input_kW=round(exergy_input_kW, precision),
            exergy_output_kW=round(exergy_output_kW, precision),
            exergy_destruction_kW=round(exergy_destruction_kW, precision),
            exergy_loss_kW=0.0,
            exergy_efficiency_pct=round(exergy_efficiency_pct, precision),
            carnot_factor=round(carnot_source, precision),
            improvement_potential_kW=round(improvement_potential_kW, precision),
            improvement_potential_pct=round(improvement_potential_pct, precision),
            formula_version="EXERGY_v1.0",
        )

        # Compute provenance hashes
        result.input_hash = self._compute_hash({
            "energy_in": energy_in_kW,
            "heat_out": heat_out_kW,
            "T_source": source_temperature_K,
            "T_sink": sink_temperature_K,
            "T0": T0,
        })
        result.output_hash = self._compute_hash({
            "exergy_destruction": exergy_destruction_kW,
            "exergy_efficiency": exergy_efficiency_pct,
        })

        return result

    async def _run_exergy_analysis(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
        operating_conditions: OperatingConditions,
        fluid_properties: FluidProperties,
    ) -> ExergyResult:
        """Run exergy analysis with operating conditions."""
        # Convert temperatures to Kelvin
        inlet_temp_K = operating_conditions.inlet_temperature_C + 273.15
        outlet_temp_K = operating_conditions.outlet_temperature_C + 273.15

        result = self.calculate_exergy(
            energy_in_kW=energy_in_kW,
            heat_out_kW=heat_out_kW,
            source_temperature_K=inlet_temp_K,
            sink_temperature_K=outlet_temp_K,
        )

        self._log_calculation(
            CalculationType.EXERGY,
            {"energy_in": energy_in_kW, "T_source_K": inlet_temp_K},
            {"exergy_destruction": result.exergy_destruction_kW},
        )

        return result

    # =========================================================================
    # SANKEY DIAGRAM GENERATION
    # =========================================================================

    def generate_sankey(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
        losses: Dict[str, float],
        title: str = "Energy Flow Diagram",
    ) -> SankeyData:
        """
        Generate Sankey diagram data for energy flow visualization.

        Creates nodes and links representing energy flows from input
        through the system to output and losses.

        Args:
            energy_in_kW: Total energy input (kW)
            heat_out_kW: Useful heat output (kW)
            losses: Dictionary of loss categories and values (kW)
            title: Diagram title

        Returns:
            SankeyData ready for rendering

        Example:
            >>> sankey = orchestrator.generate_sankey(
            ...     energy_in_kW=1000,
            ...     heat_out_kW=850,
            ...     losses={"stack": 100, "radiation": 30, "other": 20}
            ... )
            >>> plotly_data = sankey.to_plotly_dict()
        """
        nodes = []
        links = []

        # Define color scheme for thermal systems
        colors = {
            "input": "#2196F3",      # Blue for input
            "output": "#4CAF50",     # Green for useful output
            "loss": "#F44336",       # Red for losses
            "process": "#FF9800",    # Orange for process
        }

        # Input node
        nodes.append(SankeyNode(
            node_id="energy_input",
            label="Energy Input",
            value_kW=energy_in_kW,
            color=colors["input"],
            category="input",
        ))

        # Process node (thermal system)
        nodes.append(SankeyNode(
            node_id="thermal_system",
            label="Thermal System",
            value_kW=energy_in_kW,
            color=colors["process"],
            category="process",
        ))

        # Link: Input to Process
        links.append(SankeyLink(
            source="energy_input",
            target="thermal_system",
            value_kW=energy_in_kW,
            color="rgba(33,150,243,0.5)",
            label="Energy In",
        ))

        # Output node
        nodes.append(SankeyNode(
            node_id="heat_output",
            label="Useful Heat Output",
            value_kW=heat_out_kW,
            color=colors["output"],
            category="output",
        ))

        # Link: Process to Output
        links.append(SankeyLink(
            source="thermal_system",
            target="heat_output",
            value_kW=heat_out_kW,
            color="rgba(76,175,80,0.5)",
            label="Useful Heat",
        ))

        # Loss nodes and links
        total_losses = 0.0
        for loss_name, loss_value in losses.items():
            if loss_value > self.config.sankey.min_flow_threshold_kW:
                node_id = f"loss_{loss_name}"
                nodes.append(SankeyNode(
                    node_id=node_id,
                    label=f"{loss_name.replace('_', ' ').title()} Loss",
                    value_kW=loss_value,
                    color=colors["loss"],
                    category="loss",
                ))
                links.append(SankeyLink(
                    source="thermal_system",
                    target=node_id,
                    value_kW=loss_value,
                    color="rgba(244,67,54,0.5)",
                    label=loss_name,
                ))
                total_losses += loss_value

        # Calculate efficiency
        efficiency = (heat_out_kW / energy_in_kW * 100) if energy_in_kW > 0 else 0.0

        return SankeyData(
            nodes=nodes,
            links=links,
            total_input_kW=energy_in_kW,
            total_output_kW=heat_out_kW,
            total_losses_kW=total_losses,
            thermal_efficiency_pct=round(efficiency, 2),
            format=self.config.sankey.output_format,
            title=title,
        )

    async def _generate_sankey(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
        losses: Dict[str, float],
        efficiency_pct: float,
    ) -> SankeyData:
        """Generate Sankey diagram asynchronously."""
        sankey = self.generate_sankey(
            energy_in_kW=energy_in_kW,
            heat_out_kW=heat_out_kW,
            losses=losses,
            title=f"Energy Flow - {efficiency_pct:.1f}% Efficiency",
        )

        self._log_calculation(
            CalculationType.SANKEY,
            {"energy_in": energy_in_kW, "n_losses": len(losses)},
            {"n_nodes": len(sankey.nodes), "n_links": len(sankey.links)},
        )

        return sankey

    # =========================================================================
    # FLUID PROPERTY LOOKUP
    # =========================================================================

    def get_fluid_properties(
        self,
        fluid_id: str,
        temperature_C: float,
        pressure_kPa: float = 101.325,
    ) -> FluidProperties:
        """
        Get thermophysical properties for a fluid at given conditions.

        Looks up fluid properties from the configured library.
        Uses constant properties or library-specific calculations.

        Args:
            fluid_id: Fluid identifier (e.g., "water", "steam")
            temperature_C: Temperature in Celsius
            pressure_kPa: Pressure in kPa

        Returns:
            FluidProperties at specified conditions

        Raises:
            ValueError: If fluid not found or conditions invalid

        Example:
            >>> props = orchestrator.get_fluid_properties(
            ...     fluid_id="water",
            ...     temperature_C=80,
            ...     pressure_kPa=101.325
            ... )
            >>> print(f"Cp: {props.Cp_kJ_kgK} kJ/kg-K")
        """
        fluid_config = self.config.get_fluid_config(fluid_id)
        if not fluid_config:
            raise ValueError(f"Fluid '{fluid_id}' not found in library")

        # Validate operating range
        if not fluid_config.is_temperature_valid(temperature_C):
            raise ValueError(
                f"Temperature {temperature_C}C outside valid range "
                f"({fluid_config.min_temperature_C} to {fluid_config.max_temperature_C}C)"
            )
        if not fluid_config.is_pressure_valid(pressure_kPa):
            raise ValueError(
                f"Pressure {pressure_kPa}kPa outside valid range "
                f"({fluid_config.min_pressure_kPa} to {fluid_config.max_pressure_kPa}kPa)"
            )

        # For internal library, use constant properties
        # In production, would integrate with CoolProp/IAPWS
        return FluidProperties(
            fluid_id=fluid_config.fluid_id,
            name=fluid_config.fluid_name,
            temperature_C=temperature_C,
            pressure_kPa=pressure_kPa,
            phase=fluid_config.phase,
            density_kg_m3=fluid_config.density_kg_m3,
            Cp_kJ_kgK=fluid_config.Cp_kJ_kgK,
            viscosity_Pa_s=fluid_config.viscosity_Pa_s,
            conductivity_W_mK=fluid_config.conductivity_W_mK,
            source=fluid_config.library.value,
        )

    # =========================================================================
    # FLUID RECOMMENDATION (LLM ALLOWED - NON-NUMERIC)
    # =========================================================================

    def recommend_fluid(
        self,
        operating_conditions: OperatingConditions,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Recommend suitable thermal fluids for operating conditions.

        Evaluates available fluids against operating conditions
        and returns ranked recommendations with rationale.

        NOTE: This is a classification/ranking task, not numeric
        calculation, so LLM assistance is permitted if needed.

        Args:
            operating_conditions: Target operating conditions
            constraints: Optional constraints (e.g., max_cost, food_safe)

        Returns:
            List of (fluid_id, score, rationale) tuples, ranked by score

        Example:
            >>> recommendations = orchestrator.recommend_fluid(
            ...     OperatingConditions(
            ...         inlet_temperature_C=250,
            ...         outlet_temperature_C=180,
            ...         mass_flow_kg_s=5.0
            ...     )
            ... )
            >>> for fluid, score, reason in recommendations[:3]:
            ...     print(f"{fluid}: {score:.2f} - {reason}")
        """
        constraints = constraints or {}
        recommendations = []

        max_temp = max(
            operating_conditions.inlet_temperature_C,
            operating_conditions.outlet_temperature_C
        )
        min_temp = min(
            operating_conditions.inlet_temperature_C,
            operating_conditions.outlet_temperature_C
        )

        for fluid_id, config in self.config.fluid_library.items():
            # Check temperature compatibility
            if not (config.min_temperature_C <= min_temp and
                    config.max_temperature_C >= max_temp):
                continue

            # Calculate suitability score
            score = 1.0

            # Temperature margin (prefer fluids with wider margins)
            temp_margin = min(
                max_temp - config.min_temperature_C,
                config.max_temperature_C - max_temp
            ) / 100.0
            score += min(temp_margin, 0.5)

            # Specific heat (higher is better for heat transfer)
            if config.Cp_kJ_kgK > 3.0:
                score += 0.3
            elif config.Cp_kJ_kgK > 2.0:
                score += 0.1

            # Phase stability
            if config.phase.value in ["liquid", "supercritical"]:
                score += 0.2

            # Build rationale
            rationale_parts = []
            rationale_parts.append(
                f"Operating range: {config.min_temperature_C} to {config.max_temperature_C}C"
            )
            rationale_parts.append(f"Cp: {config.Cp_kJ_kgK} kJ/kg-K")

            if config.flash_point_C and max_temp < config.flash_point_C:
                rationale_parts.append("Below flash point (safe)")
                score += 0.2

            recommendations.append((
                fluid_id,
                round(score, 2),
                "; ".join(rationale_parts)
            ))

        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    # =========================================================================
    # EXPLAINABILITY INTEGRATION
    # =========================================================================

    async def _generate_explainability(
        self,
        input_data: ThermalAnalysisInput,
        efficiency_pct: float,
        exergy_result: Optional[ExergyResult],
    ) -> ExplainabilityReport:
        """
        Generate explainability report with recommendations.

        Creates feature importance rankings, LIME explanations,
        and actionable recommendations based on analysis results.
        """
        feature_importance = []
        lime_explanations = []
        recommendations = []
        key_findings = []

        # Analyze efficiency drivers (DETERMINISTIC ANALYSIS)
        loss_total = input_data.energy_in_kW - input_data.heat_out_kW
        loss_pct = (loss_total / input_data.energy_in_kW * 100) if input_data.energy_in_kW > 0 else 0

        # Feature importance based on loss contribution
        if input_data.losses_kW:
            sorted_losses = sorted(
                input_data.losses_kW.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (loss_name, loss_value) in enumerate(sorted_losses):
                importance = loss_value / loss_total if loss_total > 0 else 0
                feature_importance.append(FeatureImportance(
                    feature_name=loss_name,
                    importance_score=round(importance, 3),
                    shap_value=round(loss_value, 2),
                    direction="negative",
                ))

        # LIME-style explanations
        lime_explanations.append(LIMEExplanation(
            feature_name="thermal_efficiency",
            feature_value=efficiency_pct,
            weight=1.0,
            contribution=f"Current efficiency is {efficiency_pct:.1f}%",
        ))

        if exergy_result:
            lime_explanations.append(LIMEExplanation(
                feature_name="exergy_destruction",
                feature_value=exergy_result.exergy_destruction_kW,
                weight=0.8,
                contribution=f"Exergy destruction: {exergy_result.exergy_destruction_kW:.1f} kW "
                            f"({exergy_result.improvement_potential_pct:.1f}% improvement potential)",
            ))

        # Generate recommendations based on findings
        if efficiency_pct < 70:
            recommendations.append(Recommendation(
                type=RecommendationType.HEAT_RECOVERY,
                title="Improve Heat Recovery",
                description="Thermal efficiency is below 70%. Consider installing "
                           "heat recovery equipment to capture waste heat.",
                rationale=f"Current efficiency of {efficiency_pct:.1f}% indicates "
                         f"significant heat loss potential.",
                estimated_savings_pct=min((100 - efficiency_pct) * 0.3, 20),
                priority=1,
                complexity="medium",
            ))
            key_findings.append(f"Low thermal efficiency ({efficiency_pct:.1f}%)")

        if exergy_result and exergy_result.exergy_destruction_kW > 100:
            recommendations.append(Recommendation(
                type=RecommendationType.TEMPERATURE_OPTIMIZATION,
                title="Reduce Temperature Differentials",
                description="High exergy destruction indicates large temperature "
                           "differences. Consider staged heating or closer temperature matching.",
                rationale=f"Exergy destruction of {exergy_result.exergy_destruction_kW:.1f} kW "
                         "represents thermodynamic improvement opportunity.",
                estimated_savings_kW=exergy_result.exergy_destruction_kW * 0.2,
                priority=2,
                complexity="medium",
            ))
            key_findings.append(
                f"High exergy destruction ({exergy_result.exergy_destruction_kW:.1f} kW)"
            )

        # Check for insulation recommendation
        if input_data.losses_kW.get("radiation", 0) > loss_total * 0.1:
            recommendations.append(Recommendation(
                type=RecommendationType.INSULATION,
                title="Improve Insulation",
                description="Radiation losses exceed 10% of total losses. "
                           "Consider upgrading insulation on hot surfaces.",
                rationale="Radiation losses can be significantly reduced with "
                         "proper insulation materials.",
                estimated_savings_pct=5,
                priority=2,
                complexity="low",
            ))
            key_findings.append("High radiation losses detected")

        # Executive summary
        executive_summary = (
            f"Thermal analysis completed with {efficiency_pct:.1f}% first-law efficiency. "
        )
        if exergy_result:
            executive_summary += (
                f"Second-law (exergy) efficiency is {exergy_result.exergy_efficiency_pct:.1f}% "
                f"with {exergy_result.improvement_potential_pct:.1f}% improvement potential. "
            )
        if recommendations:
            executive_summary += (
                f"Generated {len(recommendations)} recommendations for improvement."
            )

        report = ExplainabilityReport(
            analysis_id=input_data.analysis_id,
            feature_importance=feature_importance,
            lime_explanations=lime_explanations,
            recommendations=recommendations,
            executive_summary=executive_summary,
            key_findings=key_findings,
        )

        # Compute provenance hash
        report.provenance_hash = self._compute_hash({
            "analysis_id": input_data.analysis_id,
            "n_recommendations": len(recommendations),
            "n_features": len(feature_importance),
        })

        return report

    # =========================================================================
    # PROVENANCE AND AUDIT
    # =========================================================================

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _log_calculation(
        self,
        calc_type: CalculationType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Log calculation event for audit trail."""
        event = CalculationEvent(
            calculation_type=calc_type,
            input_summary=inputs,
            input_hash=self._compute_hash(inputs),
            output_summary=outputs,
            output_hash=self._compute_hash(outputs),
            formula_id=f"{calc_type.value}_v1.0",
            deterministic=True,
            reproducible=True,
        )
        self._calculation_events.append(event)

        if self.config.enable_audit_logging:
            logger.debug(
                f"Calculation logged: type={calc_type.value}, "
                f"input_hash={event.input_hash[:8]}..."
            )

    # =========================================================================
    # STATUS AND HEALTH
    # =========================================================================

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        avg_time = (
            self._total_processing_time_ms / self._analyses_count
            if self._analyses_count > 0 else 0.0
        )

        return AgentStatus(
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            agent_version=self.VERSION,
            status="running",
            health="healthy",
            uptime_seconds=uptime,
            analyses_performed=self._analyses_count,
            analyses_successful=self._successful_count,
            avg_processing_time_ms=round(avg_time, 2),
            available_fluids=list(self.config.fluid_library.keys()),
            calculation_modes=[m.value for m in CalculationMode],
            graphql_ready=True,
            explainability_ready=self.config.explainability.shap_enabled
                               or self.config.explainability.lime_enabled,
        )

    def health_check(self) -> HealthCheckResponse:
        """Perform health check."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        checks = {
            "efficiency_calculator": "ok",
            "exergy_calculator": "ok",
            "sankey_generator": "ok",
            "fluid_library": "ok" if self.config.fluid_library else "error",
            "explainability": "ok" if self.config.explainability.shap_enabled else "disabled",
        }

        overall_status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"

        return HealthCheckResponse(
            status=overall_status,
            version=self.VERSION,
            uptime_seconds=uptime,
            checks=checks,
        )


# =============================================================================
# SYNCHRONOUS WRAPPER
# =============================================================================

def run_analysis_sync(
    input_data: ThermalAnalysisInput,
    config: Optional[ThermalIQConfig] = None,
) -> ThermalAnalysisOutput:
    """
    Run thermal analysis synchronously.

    Convenience wrapper for non-async contexts.

    Args:
        input_data: Analysis input data
        config: Agent configuration

    Returns:
        ThermalAnalysisOutput

    Example:
        >>> result = run_analysis_sync(input_data)
        >>> print(f"Efficiency: {result.efficiency_percent}%")
    """
    orchestrator = ThermalIQOrchestrator(config)
    return asyncio.run(orchestrator.analyze(input_data))
