"""
GL-009 THERMALIQ - Request Handlers

Handlers for processing analysis requests, fluid property lookups,
Sankey diagram generation, and explainability operations.

All handlers follow zero-hallucination principles for numeric
calculations and provide full audit logging.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import hashlib
import json

from .config import (
    ThermalIQConfig,
    FluidConfig,
    FluidPhase,
    CalculationMode,
    ExplainabilityMethod,
    DEFAULT_CONFIG,
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
    OperatingConditions,
    APIResponse,
    CalculationStatus,
    CalculationType,
    CalculationEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create successful validation result."""
        return cls(is_valid=True, errors=[], warnings=warnings or [])

    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create failed validation result."""
        return cls(is_valid=False, errors=errors, warnings=warnings or [])


# =============================================================================
# ANALYSIS HANDLER
# =============================================================================

class AnalysisHandler:
    """
    Handler for processing thermal analysis requests.

    Validates inputs, coordinates calculations, and formats
    responses for the thermal analysis workflow.

    Attributes:
        config: Agent configuration

    Example:
        >>> handler = AnalysisHandler()
        >>> validation = handler.validate_input(input_data)
        >>> if validation.is_valid:
        ...     result = await handler.process(input_data)
    """

    def __init__(self, config: Optional[ThermalIQConfig] = None) -> None:
        """
        Initialize analysis handler.

        Args:
            config: Agent configuration
        """
        self.config = config or DEFAULT_CONFIG
        self._request_count = 0
        self._validation_errors = 0

    def validate_input(
        self,
        input_data: ThermalAnalysisInput,
    ) -> ValidationResult:
        """
        Validate thermal analysis input data.

        Performs comprehensive validation including:
        - Energy balance check
        - Fluid property validation
        - Operating condition bounds
        - Safety limit checks

        Args:
            input_data: Input data to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Energy balance validation
        if input_data.energy_in_kW <= 0:
            errors.append("Energy input must be positive")

        if input_data.heat_out_kW < 0:
            errors.append("Heat output cannot be negative")

        if input_data.heat_out_kW > input_data.energy_in_kW:
            errors.append(
                f"Heat output ({input_data.heat_out_kW} kW) cannot exceed "
                f"energy input ({input_data.energy_in_kW} kW)"
            )

        # Loss balance check
        total_specified_losses = sum(input_data.losses_kW.values())
        implied_losses = input_data.energy_in_kW - input_data.heat_out_kW

        if total_specified_losses > 0:
            loss_diff = abs(total_specified_losses - implied_losses)
            loss_tolerance = implied_losses * 0.05  # 5% tolerance

            if loss_diff > loss_tolerance:
                warnings.append(
                    f"Specified losses ({total_specified_losses:.1f} kW) differ from "
                    f"implied losses ({implied_losses:.1f} kW) by {loss_diff:.1f} kW"
                )

        # Fluid property validation
        fluid = input_data.fluid_properties
        fluid_config = self.config.get_fluid_config(fluid.fluid_id)

        if not fluid_config:
            warnings.append(
                f"Fluid '{fluid.fluid_id}' not in library, using provided properties"
            )
        else:
            # Check temperature range
            if not fluid_config.is_temperature_valid(fluid.temperature_C):
                errors.append(
                    f"Fluid temperature {fluid.temperature_C}C outside valid range "
                    f"({fluid_config.min_temperature_C} to {fluid_config.max_temperature_C}C)"
                )

            # Check pressure range
            if not fluid_config.is_pressure_valid(fluid.pressure_kPa):
                errors.append(
                    f"Fluid pressure {fluid.pressure_kPa} kPa outside valid range "
                    f"({fluid_config.min_pressure_kPa} to {fluid_config.max_pressure_kPa} kPa)"
                )

        # Operating condition validation
        ops = input_data.operating_conditions

        if ops.inlet_temperature_C < -200 or ops.inlet_temperature_C > 1500:
            errors.append(
                f"Inlet temperature {ops.inlet_temperature_C}C outside valid range (-200 to 1500C)"
            )

        if ops.outlet_temperature_C < -200 or ops.outlet_temperature_C > 1500:
            errors.append(
                f"Outlet temperature {ops.outlet_temperature_C}C outside valid range (-200 to 1500C)"
            )

        if ops.mass_flow_kg_s < 0:
            errors.append("Mass flow rate cannot be negative")

        if ops.mass_flow_kg_s == 0:
            warnings.append("Mass flow rate is zero - no heat transfer will occur")

        # Safety checks
        if ops.inlet_temperature_C > self.config.safety.max_temperature_C:
            warnings.append(
                f"Inlet temperature exceeds safety limit ({self.config.safety.max_temperature_C}C)"
            )

        # Efficiency plausibility check
        if input_data.energy_in_kW > 0:
            implied_efficiency = (input_data.heat_out_kW / input_data.energy_in_kW) * 100
            if implied_efficiency > 99:
                warnings.append(
                    f"Implied efficiency ({implied_efficiency:.1f}%) is unusually high"
                )
            if implied_efficiency < 10:
                warnings.append(
                    f"Implied efficiency ({implied_efficiency:.1f}%) is unusually low"
                )

        if errors:
            self._validation_errors += 1
            return ValidationResult.failure(errors, warnings)

        return ValidationResult.success(warnings)

    def build_request(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
        fluid_id: str,
        inlet_temp_C: float,
        outlet_temp_C: float,
        mass_flow_kg_s: float,
        losses: Optional[Dict[str, float]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ThermalAnalysisInput:
        """
        Build analysis request from raw parameters.

        Args:
            energy_in_kW: Energy input (kW)
            heat_out_kW: Heat output (kW)
            fluid_id: Fluid identifier
            inlet_temp_C: Inlet temperature (C)
            outlet_temp_C: Outlet temperature (C)
            mass_flow_kg_s: Mass flow rate (kg/s)
            losses: Optional loss breakdown
            options: Additional options

        Returns:
            ThermalAnalysisInput ready for processing
        """
        options = options or {}

        # Get fluid properties
        fluid_config = self.config.get_fluid_config(fluid_id)
        if fluid_config:
            fluid_properties = FluidProperties(
                fluid_id=fluid_id,
                name=fluid_config.fluid_name,
                temperature_C=inlet_temp_C,
                pressure_kPa=options.get("pressure_kPa", 101.325),
                phase=fluid_config.phase,
                density_kg_m3=fluid_config.density_kg_m3,
                Cp_kJ_kgK=fluid_config.Cp_kJ_kgK,
                viscosity_Pa_s=fluid_config.viscosity_Pa_s,
                conductivity_W_mK=fluid_config.conductivity_W_mK,
            )
        else:
            # Use default water properties
            fluid_properties = FluidProperties(
                fluid_id=fluid_id,
                name=fluid_id,
                temperature_C=inlet_temp_C,
                pressure_kPa=options.get("pressure_kPa", 101.325),
                phase=FluidPhase.LIQUID,
                density_kg_m3=1000.0,
                Cp_kJ_kgK=4.186,
            )

        # Build operating conditions
        operating_conditions = OperatingConditions(
            inlet_temperature_C=inlet_temp_C,
            outlet_temperature_C=outlet_temp_C,
            mass_flow_kg_s=mass_flow_kg_s,
            pressure_kPa=options.get("pressure_kPa", 101.325),
            ambient_temperature_C=options.get("ambient_temp_C", 25.0),
        )

        # Build request
        return ThermalAnalysisInput(
            energy_in_kW=energy_in_kW,
            heat_out_kW=heat_out_kW,
            losses_kW=losses or {},
            fluid_properties=fluid_properties,
            operating_conditions=operating_conditions,
            mode=CalculationMode(options.get("mode", "full_analysis")),
            include_exergy=options.get("include_exergy", True),
            include_sankey=options.get("include_sankey", True),
            include_recommendations=options.get("include_recommendations", True),
        )

    def format_response(
        self,
        result: ThermalAnalysisOutput,
        include_details: bool = True,
    ) -> APIResponse:
        """
        Format analysis result as API response.

        Args:
            result: Analysis result
            include_details: Include detailed breakdown

        Returns:
            APIResponse with formatted data
        """
        self._request_count += 1

        data = {
            "analysis_id": result.analysis_id,
            "status": result.status.value,
            "efficiency_percent": result.efficiency_percent,
            "energy_in_kW": result.energy_in_kW,
            "heat_out_kW": result.heat_out_kW,
            "total_losses_kW": result.total_losses_kW,
            "processing_time_ms": result.processing_time_ms,
        }

        if include_details:
            data["losses_breakdown"] = result.losses_breakdown

            if result.exergy_result:
                data["exergy"] = {
                    "exergy_efficiency_pct": result.exergy_result.exergy_efficiency_pct,
                    "exergy_destruction_kW": result.exergy_result.exergy_destruction_kW,
                    "improvement_potential_pct": result.exergy_result.improvement_potential_pct,
                    "carnot_factor": result.exergy_result.carnot_factor,
                }

            if result.sankey_data:
                data["sankey"] = {
                    "node_count": len(result.sankey_data.nodes),
                    "link_count": len(result.sankey_data.links),
                    "format": result.sankey_data.format.value if hasattr(result.sankey_data.format, 'value') else result.sankey_data.format,
                }

            if result.explainability:
                data["recommendations"] = [
                    {
                        "type": r.type.value if hasattr(r.type, 'value') else r.type,
                        "title": r.title,
                        "priority": r.priority,
                        "estimated_savings_pct": r.estimated_savings_pct,
                    }
                    for r in result.explainability.recommendations
                ]
                data["key_findings"] = result.explainability.key_findings

            data["provenance_hash"] = result.provenance_hash

        warnings = result.validation_messages or []

        return APIResponse(
            success=result.status == CalculationStatus.COMPLETED,
            message=f"Analysis {result.status.value}",
            data=data,
            warnings=warnings,
            request_id=result.analysis_id,
            processing_time_ms=result.processing_time_ms,
        )


# =============================================================================
# FLUID PROPERTY HANDLER
# =============================================================================

class FluidPropertyHandler:
    """
    Handler for fluid property lookups and recommendations.

    Provides access to the fluid property library and
    generates fluid selection recommendations.

    Attributes:
        config: Agent configuration
        fluid_library: Available fluid configurations
    """

    def __init__(self, config: Optional[ThermalIQConfig] = None) -> None:
        """Initialize fluid property handler."""
        self.config = config or DEFAULT_CONFIG

    def get_available_fluids(self) -> List[Dict[str, Any]]:
        """
        Get list of available fluids in library.

        Returns:
            List of fluid dictionaries with basic info
        """
        fluids = []
        for fluid_id, config in self.config.fluid_library.items():
            fluids.append({
                "fluid_id": fluid_id,
                "name": config.fluid_name,
                "phase": config.phase.value,
                "temp_range_C": [config.min_temperature_C, config.max_temperature_C],
                "Cp_kJ_kgK": config.Cp_kJ_kgK,
            })
        return fluids

    def lookup_properties(
        self,
        fluid_id: str,
        temperature_C: float,
        pressure_kPa: float = 101.325,
    ) -> Tuple[Optional[FluidProperties], List[str]]:
        """
        Look up fluid properties at specified conditions.

        Args:
            fluid_id: Fluid identifier
            temperature_C: Temperature (C)
            pressure_kPa: Pressure (kPa)

        Returns:
            Tuple of (FluidProperties or None, list of errors)
        """
        errors = []

        fluid_config = self.config.get_fluid_config(fluid_id)
        if not fluid_config:
            errors.append(f"Fluid '{fluid_id}' not found in library")
            return None, errors

        if not fluid_config.is_temperature_valid(temperature_C):
            errors.append(
                f"Temperature {temperature_C}C outside valid range "
                f"({fluid_config.min_temperature_C} to {fluid_config.max_temperature_C}C)"
            )
            return None, errors

        if not fluid_config.is_pressure_valid(pressure_kPa):
            errors.append(
                f"Pressure {pressure_kPa} kPa outside valid range "
                f"({fluid_config.min_pressure_kPa} to {fluid_config.max_pressure_kPa} kPa)"
            )
            return None, errors

        # Return properties (using constant values for internal library)
        properties = FluidProperties(
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

        return properties, errors

    def validate_operating_conditions(
        self,
        fluid_id: str,
        conditions: OperatingConditions,
    ) -> ValidationResult:
        """
        Validate operating conditions for a specific fluid.

        Args:
            fluid_id: Fluid identifier
            conditions: Operating conditions

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        fluid_config = self.config.get_fluid_config(fluid_id)
        if not fluid_config:
            return ValidationResult.failure([f"Fluid '{fluid_id}' not found"])

        # Check temperature range
        max_temp = max(conditions.inlet_temperature_C, conditions.outlet_temperature_C)
        min_temp = min(conditions.inlet_temperature_C, conditions.outlet_temperature_C)

        if max_temp > fluid_config.max_temperature_C:
            errors.append(
                f"Maximum temperature ({max_temp}C) exceeds fluid limit "
                f"({fluid_config.max_temperature_C}C)"
            )

        if min_temp < fluid_config.min_temperature_C:
            errors.append(
                f"Minimum temperature ({min_temp}C) below fluid limit "
                f"({fluid_config.min_temperature_C}C)"
            )

        # Check for phase changes
        if fluid_config.phase == FluidPhase.LIQUID and max_temp > 100:
            warnings.append(
                "Temperature may cause phase change for liquid fluid"
            )

        # Check flash point for combustible fluids
        if fluid_config.flash_point_C and max_temp > fluid_config.flash_point_C:
            errors.append(
                f"Temperature ({max_temp}C) exceeds flash point "
                f"({fluid_config.flash_point_C}C) - fire hazard!"
            )

        # Check film temperature
        if fluid_config.max_film_temperature_C and max_temp > fluid_config.max_film_temperature_C * 0.9:
            warnings.append(
                f"Operating near maximum film temperature "
                f"({fluid_config.max_film_temperature_C}C)"
            )

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult.success(warnings)

    def format_response(
        self,
        properties: Optional[FluidProperties],
        errors: List[str],
    ) -> APIResponse:
        """Format fluid property lookup as API response."""
        if errors:
            return APIResponse(
                success=False,
                message="Property lookup failed",
                errors=errors,
            )

        if properties:
            data = {
                "fluid_id": properties.fluid_id,
                "name": properties.name,
                "temperature_C": properties.temperature_C,
                "pressure_kPa": properties.pressure_kPa,
                "phase": properties.phase.value if hasattr(properties.phase, 'value') else properties.phase,
                "density_kg_m3": properties.density_kg_m3,
                "Cp_kJ_kgK": properties.Cp_kJ_kgK,
                "viscosity_Pa_s": properties.viscosity_Pa_s,
                "conductivity_W_mK": properties.conductivity_W_mK,
                "source": properties.source,
            }
            return APIResponse(
                success=True,
                message="Properties retrieved successfully",
                data=data,
            )

        return APIResponse(
            success=False,
            message="No properties available",
        )


# =============================================================================
# SANKEY HANDLER
# =============================================================================

class SankeyHandler:
    """
    Handler for Sankey diagram generation and formatting.

    Generates Sankey diagram data in various formats for
    energy flow visualization.
    """

    def __init__(self, config: Optional[ThermalIQConfig] = None) -> None:
        """Initialize Sankey handler."""
        self.config = config or DEFAULT_CONFIG

    def validate_sankey_request(
        self,
        energy_in_kW: float,
        heat_out_kW: float,
        losses: Dict[str, float],
    ) -> ValidationResult:
        """
        Validate Sankey diagram request.

        Args:
            energy_in_kW: Energy input
            heat_out_kW: Heat output
            losses: Loss breakdown

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        if energy_in_kW <= 0:
            errors.append("Energy input must be positive")

        if heat_out_kW < 0:
            errors.append("Heat output cannot be negative")

        if heat_out_kW > energy_in_kW:
            errors.append("Heat output cannot exceed energy input")

        # Check loss balance
        total_losses = sum(losses.values())
        implied_losses = energy_in_kW - heat_out_kW

        if total_losses > 0 and abs(total_losses - implied_losses) > implied_losses * 0.1:
            warnings.append(
                f"Loss sum ({total_losses:.1f} kW) differs from implied losses "
                f"({implied_losses:.1f} kW)"
            )

        if any(v < 0 for v in losses.values()):
            errors.append("Loss values cannot be negative")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult.success(warnings)

    def generate_plotly_data(self, sankey: SankeyData) -> Dict[str, Any]:
        """
        Generate Plotly-compatible Sankey data.

        Args:
            sankey: SankeyData object

        Returns:
            Dictionary ready for Plotly.js
        """
        return sankey.to_plotly_dict()

    def generate_d3_data(self, sankey: SankeyData) -> Dict[str, Any]:
        """
        Generate D3.js-compatible Sankey data.

        Args:
            sankey: SankeyData object

        Returns:
            Dictionary ready for D3.js Sankey
        """
        return {
            "nodes": [
                {"name": n.label, "id": n.node_id, "value": n.value_kW}
                for n in sankey.nodes
            ],
            "links": [
                {
                    "source": l.source,
                    "target": l.target,
                    "value": l.value_kW,
                }
                for l in sankey.links
            ],
        }

    def format_response(
        self,
        sankey: SankeyData,
        output_format: str = "plotly",
    ) -> APIResponse:
        """Format Sankey data as API response."""
        if output_format == "plotly":
            data = self.generate_plotly_data(sankey)
        elif output_format == "d3":
            data = self.generate_d3_data(sankey)
        else:
            data = {
                "nodes": [n.dict() for n in sankey.nodes],
                "links": [l.dict() for l in sankey.links],
            }

        return APIResponse(
            success=True,
            message="Sankey diagram generated",
            data={
                "diagram_id": sankey.diagram_id,
                "format": output_format,
                "total_input_kW": sankey.total_input_kW,
                "total_output_kW": sankey.total_output_kW,
                "efficiency_pct": sankey.thermal_efficiency_pct,
                "sankey_data": data,
            },
        )


# =============================================================================
# EXPLAINABILITY HANDLER
# =============================================================================

class ExplainabilityHandler:
    """
    Handler for SHAP/LIME explainability operations.

    Generates feature importance rankings, local explanations,
    and actionable recommendations.
    """

    def __init__(self, config: Optional[ThermalIQConfig] = None) -> None:
        """Initialize explainability handler."""
        self.config = config or DEFAULT_CONFIG

    def get_available_methods(self) -> List[str]:
        """Get list of available explainability methods."""
        methods = []
        if self.config.explainability.shap_enabled:
            methods.append("shap")
        if self.config.explainability.lime_enabled:
            methods.append("lime")
        return methods

    def generate_feature_importance(
        self,
        input_data: ThermalAnalysisInput,
        output_data: ThermalAnalysisOutput,
    ) -> List[FeatureImportance]:
        """
        Generate feature importance ranking.

        DETERMINISTIC - based on input/output analysis.

        Args:
            input_data: Analysis input
            output_data: Analysis output

        Returns:
            List of FeatureImportance sorted by importance
        """
        features = []

        # Analyze contribution of each input to efficiency
        total_energy = input_data.energy_in_kW
        if total_energy <= 0:
            return features

        # Heat output contribution
        heat_ratio = input_data.heat_out_kW / total_energy
        features.append(FeatureImportance(
            feature_name="heat_output",
            importance_score=heat_ratio,
            shap_value=input_data.heat_out_kW,
            direction="positive",
        ))

        # Loss contributions
        total_losses = total_energy - input_data.heat_out_kW
        if total_losses > 0:
            for loss_name, loss_value in input_data.losses_kW.items():
                importance = loss_value / total_energy
                features.append(FeatureImportance(
                    feature_name=f"loss_{loss_name}",
                    importance_score=importance,
                    shap_value=-loss_value,  # Negative impact
                    direction="negative",
                ))

        # Operating condition contributions
        temp_range = abs(
            input_data.operating_conditions.inlet_temperature_C -
            input_data.operating_conditions.outlet_temperature_C
        )
        temp_importance = min(temp_range / 200.0, 0.3)  # Normalized
        features.append(FeatureImportance(
            feature_name="temperature_differential",
            importance_score=temp_importance,
            shap_value=temp_range,
            direction="neutral",
        ))

        # Sort by importance
        features.sort(key=lambda x: abs(x.importance_score), reverse=True)

        return features

    def generate_recommendations(
        self,
        efficiency_pct: float,
        exergy_result: Optional[ExergyResult],
        losses: Dict[str, float],
    ) -> List[Recommendation]:
        """
        Generate improvement recommendations.

        DETERMINISTIC - rule-based recommendations.

        Args:
            efficiency_pct: Current thermal efficiency
            exergy_result: Exergy analysis results
            losses: Loss breakdown

        Returns:
            List of Recommendations sorted by priority
        """
        recommendations = []

        # Low efficiency recommendation
        if efficiency_pct < 70:
            savings_potential = min((100 - efficiency_pct) * 0.25, 15)
            recommendations.append(Recommendation(
                type=RecommendationType.HEAT_RECOVERY,
                title="Implement Heat Recovery",
                description=(
                    f"Current efficiency of {efficiency_pct:.1f}% indicates "
                    "significant waste heat. Consider installing economizers, "
                    "recuperators, or waste heat boilers."
                ),
                rationale=(
                    "Heat recovery can capture 20-40% of waste heat, "
                    "significantly improving overall efficiency."
                ),
                estimated_savings_pct=savings_potential,
                priority=1,
                complexity="medium",
            ))

        # High exergy destruction
        if exergy_result and exergy_result.exergy_destruction_kW > 50:
            recommendations.append(Recommendation(
                type=RecommendationType.TEMPERATURE_OPTIMIZATION,
                title="Optimize Temperature Approach",
                description=(
                    f"Exergy destruction of {exergy_result.exergy_destruction_kW:.1f} kW "
                    "indicates large temperature driving forces. Consider staged heating "
                    "or closer temperature matching."
                ),
                rationale=(
                    "Reducing temperature differentials minimizes irreversibility "
                    "and improves thermodynamic efficiency."
                ),
                estimated_savings_kW=exergy_result.exergy_destruction_kW * 0.15,
                priority=2,
                complexity="medium",
            ))

        # Specific loss recommendations
        total_losses = sum(losses.values())
        if total_losses > 0:
            # Radiation losses
            radiation_loss = losses.get("radiation", 0)
            if radiation_loss > total_losses * 0.15:
                recommendations.append(Recommendation(
                    type=RecommendationType.INSULATION,
                    title="Upgrade Insulation",
                    description=(
                        f"Radiation losses ({radiation_loss:.1f} kW) exceed 15% of total. "
                        "Consider upgrading insulation or adding reflective barriers."
                    ),
                    rationale="Proper insulation can reduce radiation losses by 50-80%.",
                    estimated_savings_pct=radiation_loss / total_losses * 0.6 * 100,
                    priority=2,
                    complexity="low",
                ))

            # Stack/flue losses
            stack_loss = losses.get("stack", 0) + losses.get("flue", 0)
            if stack_loss > total_losses * 0.3:
                recommendations.append(Recommendation(
                    type=RecommendationType.HEAT_RECOVERY,
                    title="Install Economizer",
                    description=(
                        f"Flue gas losses ({stack_loss:.1f} kW) are significant. "
                        "Consider installing an economizer to preheat incoming streams."
                    ),
                    rationale=(
                        "Economizers typically recover 5-10% of input energy "
                        "from flue gases."
                    ),
                    estimated_savings_pct=8,
                    priority=1,
                    complexity="medium",
                ))

        # Maintenance recommendation for aging systems
        if efficiency_pct < 60 and not recommendations:
            recommendations.append(Recommendation(
                type=RecommendationType.MAINTENANCE,
                title="Schedule Maintenance Inspection",
                description=(
                    "Very low efficiency may indicate fouling, scaling, or "
                    "equipment degradation. Schedule maintenance inspection."
                ),
                rationale="Fouling can reduce heat transfer by 20-30%.",
                estimated_savings_pct=10,
                priority=1,
                complexity="low",
            ))

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)

        return recommendations[:self.config.explainability.max_recommendations]

    def format_response(
        self,
        report: ExplainabilityReport,
    ) -> APIResponse:
        """Format explainability report as API response."""
        data = {
            "report_id": report.report_id,
            "analysis_id": report.analysis_id,
            "feature_importance": [
                {
                    "feature": f.feature_name,
                    "importance": f.importance_score,
                    "direction": f.direction,
                }
                for f in report.feature_importance[:10]
            ],
            "recommendations": [
                {
                    "type": r.type.value if hasattr(r.type, 'value') else r.type,
                    "title": r.title,
                    "description": r.description,
                    "priority": r.priority,
                    "estimated_savings_pct": r.estimated_savings_pct,
                    "complexity": r.complexity,
                }
                for r in report.recommendations
            ],
            "executive_summary": report.executive_summary,
            "key_findings": report.key_findings,
        }

        return APIResponse(
            success=True,
            message="Explainability report generated",
            data=data,
        )


# =============================================================================
# CALLBACK HANDLER
# =============================================================================

class CallbackHandler:
    """
    Handler for async callbacks and event notifications.

    Manages event subscriptions and notification dispatch
    for the thermal analysis workflow.
    """

    def __init__(self) -> None:
        """Initialize callback handler."""
        self._callbacks: Dict[str, List[Callable]] = {
            "analysis_started": [],
            "analysis_completed": [],
            "analysis_failed": [],
            "efficiency_calculated": [],
            "exergy_calculated": [],
            "sankey_generated": [],
            "recommendation_generated": [],
        }
        self._event_count = 0

    def register(
        self,
        event: str,
        callback: Callable,
    ) -> bool:
        """
        Register callback for event.

        Args:
            event: Event name
            callback: Callback function

        Returns:
            True if registered successfully
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for event: {event}")
            return True
        logger.warning(f"Unknown event type: {event}")
        return False

    def unregister(
        self,
        event: str,
        callback: Callable,
    ) -> bool:
        """
        Unregister callback for event.

        Args:
            event: Event name
            callback: Callback function

        Returns:
            True if unregistered successfully
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            return True
        return False

    def emit(
        self,
        event: str,
        data: Any,
    ) -> int:
        """
        Emit event to registered callbacks.

        Args:
            event: Event name
            data: Event data

        Returns:
            Number of callbacks invoked
        """
        callbacks = self._callbacks.get(event, [])
        invoked = 0

        for callback in callbacks:
            try:
                callback(data)
                invoked += 1
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

        self._event_count += 1
        return invoked

    def get_registered_events(self) -> List[str]:
        """Get list of events with registered callbacks."""
        return [
            event for event, callbacks in self._callbacks.items()
            if callbacks
        ]

    @property
    def event_count(self) -> int:
        """Get total events emitted."""
        return self._event_count
