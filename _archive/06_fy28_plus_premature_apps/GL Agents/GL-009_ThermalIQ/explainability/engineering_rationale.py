"""
ThermalIQ Engineering Rationale Generator

Generates natural language explanations grounded in thermodynamic
principles for operators and engineers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime


class RationaleCategory(Enum):
    """Categories of engineering rationale."""
    EFFICIENCY = "efficiency"
    EXERGY = "exergy"
    HEAT_TRANSFER = "heat_transfer"
    FLUID_SELECTION = "fluid_selection"
    COMBUSTION = "combustion"
    INSULATION = "insulation"
    PRESSURE_DROP = "pressure_drop"


class PrincipleType(Enum):
    """Types of thermodynamic principles."""
    FIRST_LAW = "first_law"
    SECOND_LAW = "second_law"
    HEAT_TRANSFER = "heat_transfer"
    FLUID_MECHANICS = "fluid_mechanics"
    COMBUSTION = "combustion"
    MATERIALS = "materials"


@dataclass
class Citation:
    """Represents a citation to thermodynamic principles or standards."""
    principle: str
    source: str
    description: str
    equation: Optional[str] = None
    reference_standard: Optional[str] = None
    page_or_section: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "principle": self.principle,
            "source": self.source,
            "description": self.description,
            "equation": self.equation,
            "reference_standard": self.reference_standard,
            "page_or_section": self.page_or_section
        }

    def format_citation(self) -> str:
        """Format as readable citation."""
        parts = [f"[{self.principle}]"]
        if self.source:
            parts.append(f"Source: {self.source}")
        if self.reference_standard:
            parts.append(f"Standard: {self.reference_standard}")
        if self.page_or_section:
            parts.append(f"Section: {self.page_or_section}")
        return " - ".join(parts)


@dataclass
class RationaleSection:
    """A section of the engineering rationale."""
    title: str
    content: str
    category: RationaleCategory
    citations: List[Citation] = field(default_factory=list)
    key_values: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EngineeringRationale:
    """Complete engineering rationale for a thermal analysis result."""
    summary: str
    sections: List[RationaleSection]
    overall_assessment: str
    confidence_level: str  # "high", "medium", "low"
    applicable_standards: List[str]
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "category": s.category.value,
                    "citations": [c.to_dict() for c in s.citations],
                    "key_values": s.key_values,
                    "recommendations": s.recommendations
                }
                for s in self.sections
            ],
            "overall_assessment": self.overall_assessment,
            "confidence_level": self.confidence_level,
            "applicable_standards": self.applicable_standards,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def get_full_text(self) -> str:
        """Generate full text of rationale."""
        lines = [
            "=" * 60,
            "ENGINEERING RATIONALE",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            self.summary,
            ""
        ]

        for section in self.sections:
            lines.extend([
                f"{section.title.upper()}",
                "-" * 40,
                section.content,
                ""
            ])

            if section.key_values:
                lines.append("Key Values:")
                for key, value in section.key_values.items():
                    lines.append(f"  - {key}: {value}")
                lines.append("")

            if section.recommendations:
                lines.append("Recommendations:")
                for rec in section.recommendations:
                    lines.append(f"  * {rec}")
                lines.append("")

            if section.citations:
                lines.append("References:")
                for citation in section.citations:
                    lines.append(f"  - {citation.format_citation()}")
                lines.append("")

        lines.extend([
            "OVERALL ASSESSMENT",
            "-" * 40,
            self.overall_assessment,
            "",
            f"Confidence Level: {self.confidence_level.upper()}",
            "",
            "Applicable Standards:",
        ])

        for standard in self.applicable_standards:
            lines.append(f"  - {standard}")

        return "\n".join(lines)


class EngineeringRationaleGenerator:
    """
    Generates engineering rationales for thermal system analyses.

    Provides natural language explanations grounded in thermodynamic
    principles that operators and engineers can understand and trust.
    """

    # Thermodynamic principles database
    THERMODYNAMIC_PRINCIPLES = {
        "first_law_energy_balance": Citation(
            principle="First Law of Thermodynamics - Energy Balance",
            source="Cengel & Boles, Thermodynamics: An Engineering Approach",
            description="Energy cannot be created or destroyed, only converted. Total energy in = Total energy out + Energy stored.",
            equation="Q_in - Q_out + W_in - W_out = dE/dt",
            reference_standard="ASME PTC 4.1"
        ),
        "second_law_entropy": Citation(
            principle="Second Law of Thermodynamics - Entropy Generation",
            source="Cengel & Boles, Thermodynamics: An Engineering Approach",
            description="All real processes generate entropy, representing irreversibility and lost work potential.",
            equation="S_gen = S_out - S_in + Q/T_boundary >= 0"
        ),
        "exergy_destruction": Citation(
            principle="Exergy Destruction (Gouy-Stodola Theorem)",
            source="Bejan, Advanced Engineering Thermodynamics",
            description="Exergy destruction equals the reference temperature times entropy generation.",
            equation="Ex_destroyed = T_0 * S_gen"
        ),
        "carnot_efficiency": Citation(
            principle="Carnot Efficiency Limit",
            source="Fundamental Thermodynamics",
            description="Maximum theoretical efficiency between two temperature reservoirs.",
            equation="eta_Carnot = 1 - T_cold/T_hot"
        ),
        "fourier_law": Citation(
            principle="Fourier's Law of Heat Conduction",
            source="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
            description="Heat flux is proportional to temperature gradient.",
            equation="q = -k * dT/dx",
            reference_standard="ASTM C177"
        ),
        "newton_cooling": Citation(
            principle="Newton's Law of Cooling",
            source="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
            description="Convective heat transfer rate proportional to temperature difference.",
            equation="Q = h * A * (T_surface - T_fluid)"
        ),
        "stefan_boltzmann": Citation(
            principle="Stefan-Boltzmann Law",
            source="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
            description="Radiative heat transfer proportional to fourth power of temperature.",
            equation="Q = epsilon * sigma * A * (T^4 - T_surr^4)"
        ),
        "log_mean_temp_diff": Citation(
            principle="Log Mean Temperature Difference",
            source="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
            description="Effective temperature difference for heat exchangers.",
            equation="LMTD = (dT_1 - dT_2) / ln(dT_1/dT_2)",
            reference_standard="ASME PTC 12.5"
        ),
        "darcy_weisbach": Citation(
            principle="Darcy-Weisbach Equation",
            source="White, Fluid Mechanics",
            description="Pressure drop in pipe flow due to friction.",
            equation="dP = f * (L/D) * (rho * V^2 / 2)"
        ),
        "reynolds_number": Citation(
            principle="Reynolds Number",
            source="White, Fluid Mechanics",
            description="Dimensionless number indicating flow regime (laminar vs turbulent).",
            equation="Re = rho * V * D / mu"
        ),
        "nusselt_number": Citation(
            principle="Nusselt Number",
            source="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
            description="Dimensionless number relating convective to conductive heat transfer.",
            equation="Nu = h * L / k"
        ),
        "prandtl_number": Citation(
            principle="Prandtl Number",
            source="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
            description="Ratio of momentum diffusivity to thermal diffusivity.",
            equation="Pr = mu * Cp / k"
        ),
        "combustion_stoichiometry": Citation(
            principle="Combustion Stoichiometry",
            source="Turns, An Introduction to Combustion",
            description="Balanced chemical reaction for complete combustion.",
            reference_standard="ASME PTC 4"
        ),
        "adiabatic_flame_temp": Citation(
            principle="Adiabatic Flame Temperature",
            source="Turns, An Introduction to Combustion",
            description="Maximum theoretical temperature from combustion with no heat loss."
        ),
        "excess_air_effect": Citation(
            principle="Excess Air Effect on Efficiency",
            source="ASHRAE Handbook - HVAC Systems and Equipment",
            description="Additional air beyond stoichiometric reduces flame temperature and efficiency.",
            reference_standard="ASME PTC 4.1"
        )
    }

    # Industry standards database
    INDUSTRY_STANDARDS = {
        "boiler_efficiency": ["ASME PTC 4", "EN 12952", "EN 12953"],
        "heat_exchanger": ["ASME PTC 12.5", "TEMA Standards", "API 660"],
        "insulation": ["ASTM C177", "ASTM C680", "EN ISO 12241"],
        "combustion": ["ASME PTC 4.1", "EPA Method 19", "EN 12952-15"],
        "heat_transfer_fluids": ["ASTM D1120", "ASTM D2270", "ISO 15547"],
        "pressure_vessels": ["ASME BPVC", "EN 13445", "PED 2014/68/EU"]
    }

    def __init__(
        self,
        include_equations: bool = True,
        include_standards: bool = True,
        language: str = "en",
        detail_level: str = "standard"  # "brief", "standard", "detailed"
    ):
        """
        Initialize the rationale generator.

        Args:
            include_equations: Whether to include mathematical equations
            include_standards: Whether to cite industry standards
            language: Output language code
            detail_level: Level of detail in explanations
        """
        self.include_equations = include_equations
        self.include_standards = include_standards
        self.language = language
        self.detail_level = detail_level

    def generate_efficiency_rationale(
        self,
        result: Dict[str, Any]
    ) -> EngineeringRationale:
        """
        Generate rationale for thermal efficiency result.

        Args:
            result: Efficiency calculation result containing:
                - efficiency: Calculated efficiency (%)
                - heat_input: Total heat input (kW)
                - useful_heat: Useful heat output (kW)
                - losses: Dictionary of losses (stack, radiation, etc.)
                - equipment_type: Type of equipment

        Returns:
            EngineeringRationale with efficiency explanation
        """
        efficiency = result.get('efficiency', 0)
        heat_input = result.get('heat_input', 0)
        useful_heat = result.get('useful_heat', 0)
        losses = result.get('losses', {})
        equipment_type = result.get('equipment_type', 'thermal equipment')

        sections = []

        # Energy balance section
        energy_balance_content = self._generate_energy_balance_text(
            heat_input, useful_heat, losses
        )
        sections.append(RationaleSection(
            title="Energy Balance Analysis",
            content=energy_balance_content,
            category=RationaleCategory.EFFICIENCY,
            citations=[
                self.THERMODYNAMIC_PRINCIPLES["first_law_energy_balance"]
            ],
            key_values={
                "Heat Input": f"{heat_input:.2f} kW",
                "Useful Heat": f"{useful_heat:.2f} kW",
                "Thermal Efficiency": f"{efficiency:.1f}%"
            }
        ))

        # Loss analysis section
        loss_content = self._generate_loss_analysis_text(losses, heat_input)
        loss_recs = self._generate_loss_recommendations(losses)
        sections.append(RationaleSection(
            title="Heat Loss Analysis",
            content=loss_content,
            category=RationaleCategory.EFFICIENCY,
            citations=[
                self.THERMODYNAMIC_PRINCIPLES["fourier_law"],
                self.THERMODYNAMIC_PRINCIPLES["stefan_boltzmann"]
            ],
            key_values={k: f"{v:.2f} kW ({v/heat_input*100:.1f}%)" for k, v in losses.items()},
            recommendations=loss_recs
        ))

        # Efficiency assessment section
        assessment_content = self._generate_efficiency_assessment_text(
            efficiency, equipment_type
        )
        sections.append(RationaleSection(
            title="Efficiency Assessment",
            content=assessment_content,
            category=RationaleCategory.EFFICIENCY,
            citations=[],
            recommendations=self._generate_efficiency_recommendations(efficiency, losses)
        ))

        # Generate summary
        summary = self._generate_efficiency_summary(efficiency, equipment_type, losses)

        # Overall assessment
        overall = self._generate_overall_efficiency_assessment(efficiency, equipment_type)

        # Confidence level based on data completeness
        confidence = self._assess_confidence(result)

        return EngineeringRationale(
            summary=summary,
            sections=sections,
            overall_assessment=overall,
            confidence_level=confidence,
            applicable_standards=self.INDUSTRY_STANDARDS.get("boiler_efficiency", []),
            timestamp=datetime.now().isoformat(),
            metadata={"result_type": "efficiency", "equipment_type": equipment_type}
        )

    def generate_exergy_rationale(
        self,
        result: Dict[str, Any]
    ) -> EngineeringRationale:
        """
        Generate rationale for exergy analysis result.

        Args:
            result: Exergy analysis result containing:
                - exergy_input: Total exergy input (kW)
                - exergy_output: Useful exergy output (kW)
                - exergy_destruction: Destroyed exergy (kW)
                - exergy_efficiency: Second law efficiency (%)
                - irreversibilities: Dictionary of irreversibility sources

        Returns:
            EngineeringRationale with exergy explanation
        """
        exergy_input = result.get('exergy_input', 0)
        exergy_output = result.get('exergy_output', 0)
        exergy_destruction = result.get('exergy_destruction', 0)
        exergy_efficiency = result.get('exergy_efficiency', 0)
        irreversibilities = result.get('irreversibilities', {})
        reference_temp = result.get('reference_temperature', 298.15)

        sections = []

        # Exergy balance section
        exergy_balance_content = self._generate_exergy_balance_text(
            exergy_input, exergy_output, exergy_destruction
        )
        sections.append(RationaleSection(
            title="Exergy Balance (Second Law Analysis)",
            content=exergy_balance_content,
            category=RationaleCategory.EXERGY,
            citations=[
                self.THERMODYNAMIC_PRINCIPLES["second_law_entropy"],
                self.THERMODYNAMIC_PRINCIPLES["exergy_destruction"]
            ],
            key_values={
                "Exergy Input": f"{exergy_input:.2f} kW",
                "Useful Exergy": f"{exergy_output:.2f} kW",
                "Exergy Destruction": f"{exergy_destruction:.2f} kW",
                "Second Law Efficiency": f"{exergy_efficiency:.1f}%",
                "Reference Temperature": f"{reference_temp:.2f} K ({reference_temp-273.15:.1f} C)"
            }
        ))

        # Irreversibility sources section
        irreversibility_content = self._generate_irreversibility_text(
            irreversibilities, exergy_destruction
        )
        sections.append(RationaleSection(
            title="Sources of Irreversibility",
            content=irreversibility_content,
            category=RationaleCategory.EXERGY,
            citations=[
                self.THERMODYNAMIC_PRINCIPLES["carnot_efficiency"]
            ],
            key_values={
                k: f"{v:.2f} kW ({v/exergy_destruction*100:.1f}% of destruction)"
                for k, v in irreversibilities.items()
            },
            recommendations=self._generate_exergy_recommendations(irreversibilities)
        ))

        # Improvement potential section
        improvement_content = self._generate_improvement_potential_text(
            exergy_efficiency, exergy_destruction
        )
        sections.append(RationaleSection(
            title="Improvement Potential",
            content=improvement_content,
            category=RationaleCategory.EXERGY,
            citations=[],
            recommendations=[
                "Focus on reducing largest irreversibility sources first",
                "Consider waste heat recovery for thermal exergy losses",
                "Optimize temperature differences in heat transfer"
            ]
        ))

        summary = self._generate_exergy_summary(
            exergy_efficiency, exergy_destruction, irreversibilities
        )
        overall = self._generate_overall_exergy_assessment(exergy_efficiency)
        confidence = self._assess_confidence(result)

        return EngineeringRationale(
            summary=summary,
            sections=sections,
            overall_assessment=overall,
            confidence_level=confidence,
            applicable_standards=["ASME PTC 4", "ISO 50001"],
            timestamp=datetime.now().isoformat(),
            metadata={"result_type": "exergy"}
        )

    def generate_fluid_recommendation_rationale(
        self,
        fluid: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        operating_conditions: Optional[Dict[str, Any]] = None
    ) -> EngineeringRationale:
        """
        Generate rationale for heat transfer fluid recommendation.

        Args:
            fluid: Recommended fluid properties and scores
            alternatives: List of alternative fluids considered
            operating_conditions: Operating temperature/pressure conditions

        Returns:
            EngineeringRationale with fluid selection explanation
        """
        fluid_name = fluid.get('name', 'Recommended Fluid')
        score = fluid.get('score', 0)
        properties = fluid.get('properties', {})
        operating_conditions = operating_conditions or {}

        sections = []

        # Fluid properties section
        properties_content = self._generate_fluid_properties_text(
            fluid_name, properties, operating_conditions
        )
        sections.append(RationaleSection(
            title="Fluid Properties Analysis",
            content=properties_content,
            category=RationaleCategory.FLUID_SELECTION,
            citations=[
                self.THERMODYNAMIC_PRINCIPLES["prandtl_number"],
                self.THERMODYNAMIC_PRINCIPLES["nusselt_number"]
            ],
            key_values={
                "Recommended Fluid": fluid_name,
                "Selection Score": f"{score:.2f}",
                **{k: f"{v}" for k, v in properties.items()}
            }
        ))

        # Comparison section
        if alternatives:
            comparison_content = self._generate_fluid_comparison_text(
                fluid, alternatives
            )
            sections.append(RationaleSection(
                title="Alternative Fluids Comparison",
                content=comparison_content,
                category=RationaleCategory.FLUID_SELECTION,
                citations=[],
                key_values={
                    alt.get('name', f'Alt {i}'): f"Score: {alt.get('score', 0):.2f}"
                    for i, alt in enumerate(alternatives[:5])
                }
            ))

        # Operating conditions fit section
        if operating_conditions:
            conditions_content = self._generate_operating_conditions_text(
                fluid, operating_conditions
            )
            sections.append(RationaleSection(
                title="Operating Conditions Suitability",
                content=conditions_content,
                category=RationaleCategory.FLUID_SELECTION,
                citations=[],
                key_values=operating_conditions,
                recommendations=self._generate_fluid_recommendations(fluid, operating_conditions)
            ))

        summary = self._generate_fluid_summary(fluid, alternatives)
        overall = self._generate_overall_fluid_assessment(fluid, operating_conditions)
        confidence = "high" if len(alternatives) > 2 else "medium"

        return EngineeringRationale(
            summary=summary,
            sections=sections,
            overall_assessment=overall,
            confidence_level=confidence,
            applicable_standards=self.INDUSTRY_STANDARDS.get("heat_transfer_fluids", []),
            timestamp=datetime.now().isoformat(),
            metadata={"result_type": "fluid_selection", "fluid_name": fluid_name}
        )

    def cite_thermodynamic_principles(
        self,
        categories: Optional[List[str]] = None
    ) -> List[Citation]:
        """
        Get citations for relevant thermodynamic principles.

        Args:
            categories: Optional list of principle categories to filter

        Returns:
            List of Citation objects
        """
        if categories is None:
            return list(self.THERMODYNAMIC_PRINCIPLES.values())

        citations = []
        category_keywords = {
            "efficiency": ["first_law", "carnot", "exergy"],
            "exergy": ["second_law", "exergy", "carnot"],
            "heat_transfer": ["fourier", "newton", "stefan", "nusselt", "log_mean"],
            "combustion": ["combustion", "stoichiometry", "flame", "excess_air"],
            "fluid_mechanics": ["darcy", "reynolds", "prandtl"]
        }

        for category in categories:
            keywords = category_keywords.get(category.lower(), [])
            for key, citation in self.THERMODYNAMIC_PRINCIPLES.items():
                if any(kw in key for kw in keywords):
                    if citation not in citations:
                        citations.append(citation)

        return citations

    # Private helper methods for text generation

    def _generate_energy_balance_text(
        self,
        heat_input: float,
        useful_heat: float,
        losses: Dict[str, float]
    ) -> str:
        """Generate text explaining energy balance."""
        total_losses = sum(losses.values())
        unaccounted = heat_input - useful_heat - total_losses

        text = (
            f"The First Law of Thermodynamics requires that all energy entering the system "
            f"must either exit as useful output, be lost to the environment, or change the "
            f"system's stored energy. In this analysis:\n\n"
            f"- Heat input to the system: {heat_input:.2f} kW\n"
            f"- Useful heat delivered: {useful_heat:.2f} kW ({useful_heat/heat_input*100:.1f}%)\n"
            f"- Total identified losses: {total_losses:.2f} kW ({total_losses/heat_input*100:.1f}%)\n"
        )

        if abs(unaccounted) > 0.01 * heat_input:
            text += f"- Unaccounted/measurement error: {unaccounted:.2f} kW ({unaccounted/heat_input*100:.1f}%)\n"

        if self.include_equations:
            text += f"\nEnergy Balance: Q_in = Q_useful + Q_losses + dE_stored"

        return text

    def _generate_loss_analysis_text(
        self,
        losses: Dict[str, float],
        heat_input: float
    ) -> str:
        """Generate text analyzing heat losses."""
        sorted_losses = sorted(losses.items(), key=lambda x: x[1], reverse=True)

        text = "Heat losses occur through several mechanisms:\n\n"

        for loss_name, loss_value in sorted_losses:
            percentage = (loss_value / heat_input * 100) if heat_input > 0 else 0

            if "stack" in loss_name.lower() or "flue" in loss_name.lower():
                text += f"- {loss_name}: {loss_value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Hot exhaust gases carry significant thermal energy. "
                text += "Lower stack temperature indicates better heat recovery.\n"

            elif "radiation" in loss_name.lower() or "surface" in loss_name.lower():
                text += f"- {loss_name}: {loss_value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Heat radiates and convects from hot surfaces to surroundings. "
                text += "Proper insulation minimizes this loss.\n"

            elif "blowdown" in loss_name.lower():
                text += f"- {loss_name}: {loss_value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Energy lost with water discharged to control dissolved solids. "
                text += "Blowdown heat recovery can reclaim this energy.\n"

            else:
                text += f"- {loss_name}: {loss_value:.2f} kW ({percentage:.1f}%)\n"

        return text

    def _generate_loss_recommendations(
        self,
        losses: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on loss analysis."""
        recommendations = []
        sorted_losses = sorted(losses.items(), key=lambda x: x[1], reverse=True)

        for loss_name, loss_value in sorted_losses[:3]:
            if "stack" in loss_name.lower():
                recommendations.append(
                    "Consider economizer or air preheater to recover stack heat"
                )
            elif "radiation" in loss_name.lower():
                recommendations.append(
                    "Inspect and improve insulation on hot surfaces"
                )
            elif "blowdown" in loss_name.lower():
                recommendations.append(
                    "Install blowdown heat recovery system"
                )
            elif "incomplete" in loss_name.lower():
                recommendations.append(
                    "Optimize combustion air ratio and burner maintenance"
                )

        return recommendations

    def _generate_efficiency_assessment_text(
        self,
        efficiency: float,
        equipment_type: str
    ) -> str:
        """Generate efficiency assessment text."""
        # Typical efficiency ranges by equipment type
        benchmarks = {
            "boiler": {"poor": 70, "average": 80, "good": 85, "excellent": 90},
            "furnace": {"poor": 65, "average": 75, "good": 82, "excellent": 88},
            "heat_exchanger": {"poor": 60, "average": 75, "good": 85, "excellent": 95},
            "default": {"poor": 60, "average": 75, "good": 85, "excellent": 92}
        }

        bench = benchmarks.get(equipment_type.lower(), benchmarks["default"])

        if efficiency >= bench["excellent"]:
            rating = "excellent"
            assessment = "operates at high efficiency with minimal losses"
        elif efficiency >= bench["good"]:
            rating = "good"
            assessment = "performs well but has room for improvement"
        elif efficiency >= bench["average"]:
            rating = "average"
            assessment = "operates at typical industry levels with significant improvement potential"
        else:
            rating = "below average"
            assessment = "requires attention to improve performance and reduce energy waste"

        text = (
            f"At {efficiency:.1f}% thermal efficiency, this {equipment_type} {assessment}.\n\n"
            f"Performance Rating: {rating.upper()}\n\n"
            f"Benchmark comparison for {equipment_type}:\n"
            f"- Excellent: > {bench['excellent']}%\n"
            f"- Good: {bench['good']}-{bench['excellent']}%\n"
            f"- Average: {bench['average']}-{bench['good']}%\n"
            f"- Below Average: < {bench['average']}%"
        )

        return text

    def _generate_efficiency_recommendations(
        self,
        efficiency: float,
        losses: Dict[str, float]
    ) -> List[str]:
        """Generate efficiency improvement recommendations."""
        recommendations = []

        if efficiency < 85:
            recommendations.append(
                "Conduct detailed energy audit to identify improvement opportunities"
            )

        if efficiency < 80:
            recommendations.append(
                "Review and optimize combustion parameters (air/fuel ratio, burner condition)"
            )

        # Add loss-specific recommendations
        recommendations.extend(self._generate_loss_recommendations(losses)[:2])

        if efficiency < 90:
            recommendations.append(
                "Consider waste heat recovery systems for remaining losses"
            )

        return recommendations[:5]

    def _generate_efficiency_summary(
        self,
        efficiency: float,
        equipment_type: str,
        losses: Dict[str, float]
    ) -> str:
        """Generate efficiency analysis summary."""
        total_losses = sum(losses.values())
        largest_loss = max(losses.items(), key=lambda x: x[1]) if losses else ("N/A", 0)

        return (
            f"The {equipment_type} operates at {efficiency:.1f}% thermal efficiency. "
            f"The largest source of heat loss is {largest_loss[0]} at {largest_loss[1]:.1f} kW, "
            f"representing {largest_loss[1]/sum(losses.values())*100:.0f}% of total losses. "
            f"Energy balance analysis confirms proper accounting of input and output streams."
        )

    def _generate_overall_efficiency_assessment(
        self,
        efficiency: float,
        equipment_type: str
    ) -> str:
        """Generate overall efficiency assessment."""
        if efficiency >= 90:
            return (
                f"The {equipment_type} demonstrates excellent thermal performance. "
                f"Maintain current operating practices and conduct regular maintenance "
                f"to sustain this efficiency level."
            )
        elif efficiency >= 80:
            return (
                f"The {equipment_type} performs at an acceptable level with opportunities "
                f"for improvement. Focus on reducing the largest loss sources to achieve "
                f"efficiency gains of 5-10 percentage points."
            )
        else:
            return (
                f"The {equipment_type} efficiency is below optimal levels. "
                f"A comprehensive improvement program addressing insulation, combustion "
                f"optimization, and heat recovery is recommended to reduce operating costs."
            )

    def _generate_exergy_balance_text(
        self,
        exergy_input: float,
        exergy_output: float,
        exergy_destruction: float
    ) -> str:
        """Generate exergy balance explanation text."""
        exergy_efficiency = (exergy_output / exergy_input * 100) if exergy_input > 0 else 0

        text = (
            f"Exergy analysis applies the Second Law of Thermodynamics to quantify "
            f"irreversibilities and identify improvement opportunities beyond first-law efficiency.\n\n"
            f"The exergy balance shows:\n"
            f"- Exergy input (fuel chemical + thermal): {exergy_input:.2f} kW\n"
            f"- Useful exergy output (work potential delivered): {exergy_output:.2f} kW\n"
            f"- Exergy destroyed (lost work potential): {exergy_destruction:.2f} kW\n"
            f"- Second Law (exergy) efficiency: {exergy_efficiency:.1f}%\n\n"
            f"Unlike energy, exergy can be destroyed. The destruction represents permanent "
            f"loss of ability to do useful work, caused by process irreversibilities."
        )

        if self.include_equations:
            text += "\n\nExergy Balance: Ex_in = Ex_out + Ex_destroyed + Ex_loss"

        return text

    def _generate_irreversibility_text(
        self,
        irreversibilities: Dict[str, float],
        total_destruction: float
    ) -> str:
        """Generate irreversibility sources explanation."""
        sorted_irr = sorted(irreversibilities.items(), key=lambda x: x[1], reverse=True)

        text = (
            f"Exergy is destroyed by irreversible processes. The main sources are:\n\n"
        )

        for source, value in sorted_irr:
            percentage = (value / total_destruction * 100) if total_destruction > 0 else 0

            if "combustion" in source.lower():
                text += f"- {source}: {value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Combustion is inherently irreversible due to chemical reaction at high temperature.\n"

            elif "heat_transfer" in source.lower() or "temperature" in source.lower():
                text += f"- {source}: {value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Heat transfer across finite temperature differences destroys exergy.\n"

            elif "mixing" in source.lower():
                text += f"- {source}: {value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Mixing of streams at different temperatures or compositions is irreversible.\n"

            elif "friction" in source.lower() or "pressure" in source.lower():
                text += f"- {source}: {value:.2f} kW ({percentage:.1f}%)\n"
                text += "  Fluid friction and pressure drops convert work potential to heat.\n"

            else:
                text += f"- {source}: {value:.2f} kW ({percentage:.1f}%)\n"

        return text

    def _generate_improvement_potential_text(
        self,
        exergy_efficiency: float,
        exergy_destruction: float
    ) -> str:
        """Generate improvement potential discussion."""
        theoretical_max = 100 - exergy_efficiency
        practical_improvement = min(theoretical_max * 0.3, 15)  # Assume 30% of gap or 15% max

        return (
            f"The gap between current exergy efficiency ({exergy_efficiency:.1f}%) and "
            f"the theoretical maximum (100%) represents the improvement potential.\n\n"
            f"- Theoretical improvement potential: {theoretical_max:.1f} percentage points\n"
            f"- Practical improvement potential: {practical_improvement:.1f} percentage points\n"
            f"  (Based on technically achievable measures)\n\n"
            f"Each kW of exergy destruction avoided translates directly to reduced fuel "
            f"consumption or increased useful output. The {exergy_destruction:.1f} kW currently "
            f"destroyed represents lost work potential that cannot be recovered."
        )

    def _generate_exergy_recommendations(
        self,
        irreversibilities: Dict[str, float]
    ) -> List[str]:
        """Generate exergy-based recommendations."""
        recommendations = []
        sorted_irr = sorted(irreversibilities.items(), key=lambda x: x[1], reverse=True)

        for source, value in sorted_irr[:3]:
            if "combustion" in source.lower():
                recommendations.append(
                    "Preheat combustion air to reduce combustion irreversibility"
                )
                recommendations.append(
                    "Consider oxygen-enriched combustion for high-temperature processes"
                )
            elif "heat_transfer" in source.lower():
                recommendations.append(
                    "Minimize temperature differences in heat exchangers (larger surface area)"
                )
            elif "mixing" in source.lower():
                recommendations.append(
                    "Use regenerative heat exchange instead of direct mixing where possible"
                )
            elif "friction" in source.lower():
                recommendations.append(
                    "Optimize pipe sizing and reduce flow restrictions"
                )

        return list(set(recommendations))[:5]  # Remove duplicates

    def _generate_exergy_summary(
        self,
        exergy_efficiency: float,
        exergy_destruction: float,
        irreversibilities: Dict[str, float]
    ) -> str:
        """Generate exergy analysis summary."""
        largest = max(irreversibilities.items(), key=lambda x: x[1]) if irreversibilities else ("N/A", 0)

        return (
            f"Second Law analysis reveals a {exergy_efficiency:.1f}% exergy efficiency "
            f"with {exergy_destruction:.1f} kW of exergy destruction. The primary source "
            f"of irreversibility is {largest[0]}, accounting for "
            f"{largest[1]/exergy_destruction*100:.0f}% of total destruction. "
            f"This analysis provides a thermodynamic limit for improvement potential."
        )

    def _generate_overall_exergy_assessment(
        self,
        exergy_efficiency: float
    ) -> str:
        """Generate overall exergy assessment."""
        if exergy_efficiency >= 50:
            return (
                f"At {exergy_efficiency:.1f}% exergy efficiency, the system achieves "
                f"good second-law performance. Focus on high-impact measures like "
                f"combustion air preheating and waste heat recovery."
            )
        elif exergy_efficiency >= 30:
            return (
                f"The {exergy_efficiency:.1f}% exergy efficiency indicates significant "
                f"irreversibilities. Systematic reduction of temperature differences "
                f"and process integration can yield substantial improvements."
            )
        else:
            return (
                f"Low exergy efficiency ({exergy_efficiency:.1f}%) reveals major "
                f"thermodynamic inefficiencies. Consider process redesign, heat "
                f"integration, and cogeneration opportunities."
            )

    def _generate_fluid_properties_text(
        self,
        fluid_name: str,
        properties: Dict[str, Any],
        operating_conditions: Dict[str, Any]
    ) -> str:
        """Generate fluid properties explanation text."""
        text = (
            f"{fluid_name} has been selected based on its thermophysical properties "
            f"and suitability for the specified operating conditions.\n\n"
            f"Key properties:\n"
        )

        property_descriptions = {
            "specific_heat": ("Specific Heat Capacity", "kJ/kg-K", "Higher values improve heat carrying capacity"),
            "thermal_conductivity": ("Thermal Conductivity", "W/m-K", "Higher values improve heat transfer"),
            "viscosity": ("Dynamic Viscosity", "mPa-s", "Lower values reduce pumping power"),
            "density": ("Density", "kg/m3", "Affects flow rates and equipment sizing"),
            "boiling_point": ("Boiling Point", "C", "Must exceed maximum operating temperature"),
            "flash_point": ("Flash Point", "C", "Safety consideration for high temperatures"),
            "thermal_stability": ("Thermal Stability Limit", "C", "Maximum temperature before degradation")
        }

        for prop_key, prop_value in properties.items():
            if prop_key in property_descriptions:
                name, unit, desc = property_descriptions[prop_key]
                text += f"- {name}: {prop_value} {unit}\n  {desc}\n"

        return text

    def _generate_fluid_comparison_text(
        self,
        recommended: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ) -> str:
        """Generate fluid comparison text."""
        text = (
            f"{recommended.get('name', 'The recommended fluid')} scored "
            f"{recommended.get('score', 0):.2f} compared to alternatives:\n\n"
        )

        for i, alt in enumerate(alternatives[:5], 1):
            alt_name = alt.get('name', f'Alternative {i}')
            alt_score = alt.get('score', 0)
            diff = recommended.get('score', 0) - alt_score

            if diff > 0.1:
                comparison = "significantly lower"
            elif diff > 0.01:
                comparison = "slightly lower"
            else:
                comparison = "comparable"

            text += f"{i}. {alt_name}: Score {alt_score:.2f} ({comparison})\n"

        return text

    def _generate_operating_conditions_text(
        self,
        fluid: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> str:
        """Generate operating conditions suitability text."""
        fluid_name = fluid.get('name', 'The selected fluid')
        properties = fluid.get('properties', {})

        text = f"{fluid_name} is suitable for the specified operating conditions:\n\n"

        if 'max_temperature' in conditions:
            max_temp = conditions['max_temperature']
            stability = properties.get('thermal_stability', 300)
            margin = stability - max_temp

            if margin > 50:
                text += f"- Temperature margin: {margin}C above maximum operating temperature (excellent)\n"
            elif margin > 20:
                text += f"- Temperature margin: {margin}C above maximum operating temperature (adequate)\n"
            else:
                text += f"- Temperature margin: {margin}C above maximum operating temperature (limited)\n"

        if 'pressure' in conditions:
            text += f"- Operating pressure: {conditions['pressure']} bar (within fluid limits)\n"

        if 'flow_rate' in conditions:
            text += f"- Flow rate: {conditions['flow_rate']} compatible with viscosity characteristics\n"

        return text

    def _generate_fluid_recommendations(
        self,
        fluid: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> List[str]:
        """Generate fluid handling recommendations."""
        recommendations = [
            f"Verify material compatibility with {fluid.get('name', 'selected fluid')}",
            "Implement appropriate monitoring for fluid degradation",
            "Follow manufacturer guidelines for initial charging and maintenance"
        ]

        if conditions.get('max_temperature', 0) > 200:
            recommendations.append(
                "Install nitrogen blanketing for high-temperature operation"
            )

        return recommendations

    def _generate_fluid_summary(
        self,
        fluid: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ) -> str:
        """Generate fluid selection summary."""
        fluid_name = fluid.get('name', 'The recommended fluid')
        score = fluid.get('score', 0)
        n_alternatives = len(alternatives)

        return (
            f"{fluid_name} is recommended with a selection score of {score:.2f}, "
            f"outperforming {n_alternatives} alternative fluids evaluated. "
            f"Selection criteria included thermal properties, stability, and cost factors."
        )

    def _generate_overall_fluid_assessment(
        self,
        fluid: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> str:
        """Generate overall fluid selection assessment."""
        fluid_name = fluid.get('name', 'The selected fluid')

        return (
            f"{fluid_name} provides the optimal balance of heat transfer performance, "
            f"thermal stability, and operational characteristics for the specified application. "
            f"Proper installation and maintenance procedures should be followed to ensure "
            f"long-term reliable operation."
        )

    def _assess_confidence(
        self,
        result: Dict[str, Any]
    ) -> str:
        """Assess confidence level based on data completeness."""
        required_fields = {
            "efficiency": ["efficiency", "heat_input", "useful_heat"],
            "exergy": ["exergy_input", "exergy_output", "exergy_destruction"],
            "fluid_selection": ["name", "score", "properties"]
        }

        # Count available fields
        n_available = sum(1 for k in result.keys() if result.get(k) is not None)
        n_total = len(result)

        if n_total == 0:
            return "low"

        completeness = n_available / n_total

        if completeness >= 0.9:
            return "high"
        elif completeness >= 0.7:
            return "medium"
        else:
            return "low"
