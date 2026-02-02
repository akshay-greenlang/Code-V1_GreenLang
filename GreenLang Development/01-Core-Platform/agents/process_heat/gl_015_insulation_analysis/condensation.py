"""
GL-015 INSULSCAN - Condensation Prevention Analyzer

Analyzes cold insulation systems for condensation prevention,
including dew point calculations, vapor barrier requirements,
and minimum insulation thickness.

All calculations are DETERMINISTIC - zero hallucination.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    CondensationConfig,
    InsulationAnalysisConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    InsulationLayer,
    CondensationAnalysisResult,
    GeometryType,
    ServiceType,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterialDatabase,
    InsulationMaterial,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.heat_loss import (
    HeatLossCalculator,
)

logger = logging.getLogger(__name__)


@dataclass
class DewPointResult:
    """Dew point calculation result."""
    dew_point_f: float
    dew_point_c: float
    relative_humidity_pct: float
    dry_bulb_temp_f: float
    wet_bulb_temp_f: Optional[float]


@dataclass
class VaporBarrierAnalysis:
    """Vapor barrier analysis result."""
    required: bool
    recommended_perm_rating: float
    location: str  # innermost, between_layers
    material_suggestions: List[str]
    condensation_point_in_system: Optional[float]


class CondensationAnalyzer:
    """
    Condensation prevention analyzer for cold insulation systems.

    Calculates dew point temperature and determines minimum insulation
    thickness and vapor barrier requirements to prevent condensation
    on cold surfaces.

    Key Principles:
        - Surface temperature must stay above ambient dew point
        - Vapor barrier prevents moisture migration into insulation
        - Design for worst-case humidity conditions

    Features:
        - Dew point calculation (Magnus formula)
        - Minimum thickness for condensation prevention
        - Vapor barrier requirements
        - Inter-layer condensation analysis
        - Cryogenic service considerations

    Attributes:
        heat_loss_calc: Heat loss calculator
        material_db: Material database
        condensation_config: Condensation configuration

    Example:
        >>> analyzer = CondensationAnalyzer(config)
        >>> result = analyzer.analyze(cold_pipe_input)
        >>> if result.condensation_risk:
        ...     print(f"Need {result.minimum_thickness_for_prevention_in} inches")
    """

    def __init__(
        self,
        config: InsulationAnalysisConfig,
        material_database: Optional[InsulationMaterialDatabase] = None,
        heat_loss_calculator: Optional[HeatLossCalculator] = None,
    ) -> None:
        """
        Initialize the condensation analyzer.

        Args:
            config: Analysis configuration
            material_database: Material database
            heat_loss_calculator: Heat loss calculator
        """
        self.config = config
        self.condensation = config.condensation
        self.material_db = material_database or InsulationMaterialDatabase()
        self.heat_loss_calc = heat_loss_calculator or HeatLossCalculator(
            material_database=self.material_db,
        )
        self._calculation_count = 0

        logger.info("CondensationAnalyzer initialized")

    def analyze(
        self,
        input_data: InsulationInput,
    ) -> CondensationAnalysisResult:
        """
        Perform condensation prevention analysis.

        Args:
            input_data: Insulation analysis input

        Returns:
            CondensationAnalysisResult
        """
        self._calculation_count += 1
        logger.debug(f"Analyzing condensation for {input_data.item_id}")

        # Calculate dew point
        dew_point = self._calculate_dew_point(
            dry_bulb_temp_f=input_data.ambient_temperature_f,
            relative_humidity_pct=input_data.relative_humidity_pct,
        )

        # Calculate current surface temperature
        if input_data.insulation_layers:
            heat_loss_result = self.heat_loss_calc.calculate_heat_loss(input_data)
            surface_temp_f = heat_loss_result.outer_surface_temperature_f
        else:
            # Bare surface - use operating temperature
            surface_temp_f = input_data.operating_temperature_f

        # Calculate margin above dew point
        margin = surface_temp_f - dew_point.dew_point_f

        # Determine condensation risk
        condensation_risk = margin < self.condensation.design_dew_point_margin_f

        # Risk level
        if margin < 0:
            risk_level = "high"
        elif margin < 5:
            risk_level = "medium"
        elif margin < 10:
            risk_level = "low"
        else:
            risk_level = "none"

        # Calculate minimum thickness for prevention
        min_thickness = self._calculate_minimum_thickness_for_prevention(
            input_data=input_data,
            target_surface_temp_f=dew_point.dew_point_f + self.condensation.design_dew_point_margin_f,
        )

        # Vapor barrier analysis
        vapor_analysis = self._analyze_vapor_barrier_requirements(
            input_data=input_data,
            dew_point_f=dew_point.dew_point_f,
        )

        # Current thickness
        current_thickness = sum(
            layer.thickness_in for layer in input_data.insulation_layers
        )

        return CondensationAnalysisResult(
            ambient_dew_point_f=round(dew_point.dew_point_f, 1),
            surface_temperature_f=round(surface_temp_f, 1),
            margin_above_dew_point_f=round(margin, 1),
            condensation_risk=condensation_risk,
            condensation_risk_level=risk_level,
            minimum_thickness_for_prevention_in=round(min_thickness, 2),
            vapor_barrier_required=vapor_analysis.required,
            vapor_barrier_location=vapor_analysis.location,
            current_thickness_in=current_thickness,
            additional_thickness_needed_in=round(
                max(0, min_thickness - current_thickness), 2
            ),
            has_adequate_vapor_barrier=self._check_vapor_barrier(input_data),
        )

    def _calculate_dew_point(
        self,
        dry_bulb_temp_f: float,
        relative_humidity_pct: float,
    ) -> DewPointResult:
        """
        Calculate dew point temperature using Magnus formula.

        The Magnus formula provides accurate dew point calculation
        for typical ambient conditions.

        Td = b * alpha / (a - alpha)
        where alpha = a*T/(b+T) + ln(RH/100)

        Constants for T in Celsius:
        a = 17.27
        b = 237.7

        Args:
            dry_bulb_temp_f: Dry bulb temperature (F)
            relative_humidity_pct: Relative humidity (%)

        Returns:
            DewPointResult
        """
        # Convert to Celsius
        T_c = (dry_bulb_temp_f - 32) * 5 / 9
        RH = relative_humidity_pct / 100

        # Magnus formula constants
        a = 17.27
        b = 237.7

        # Avoid log(0)
        RH = max(RH, 0.01)

        # Calculate alpha
        alpha = a * T_c / (b + T_c) + math.log(RH)

        # Calculate dew point in Celsius
        Td_c = b * alpha / (a - alpha)

        # Convert to Fahrenheit
        Td_f = Td_c * 9 / 5 + 32

        return DewPointResult(
            dew_point_f=Td_f,
            dew_point_c=Td_c,
            relative_humidity_pct=relative_humidity_pct,
            dry_bulb_temp_f=dry_bulb_temp_f,
            wet_bulb_temp_f=None,  # Could calculate if needed
        )

    def _calculate_minimum_thickness_for_prevention(
        self,
        input_data: InsulationInput,
        target_surface_temp_f: float,
    ) -> float:
        """
        Calculate minimum thickness to prevent condensation.

        For cold service, surface temperature must stay above dew point.

        Args:
            input_data: Input data
            target_surface_temp_f: Target surface temperature (above dew point)

        Returns:
            Minimum thickness (inches)
        """
        # Select appropriate material for cold service
        if input_data.insulation_layers:
            material_id = input_data.insulation_layers[0].material_id
        else:
            material = self._select_cold_service_material(
                input_data.operating_temperature_f
            )
            material_id = material.material_id

        # Bisection search for minimum thickness
        low = 0.25
        high = 8.0
        tolerance = 0.1

        # Handle case where operating temp is above target
        if input_data.operating_temperature_f >= target_surface_temp_f:
            return 0.0  # No insulation needed for cold prevention

        while high - low > tolerance:
            mid = (low + high) / 2

            test_input = input_data.copy(deep=True)
            test_input.insulation_layers = [
                InsulationLayer(
                    layer_number=1,
                    material_id=material_id,
                    thickness_in=mid,
                )
            ]

            result = self.heat_loss_calc.calculate_heat_loss(test_input)
            surface_temp = result.outer_surface_temperature_f

            # For cold service, surface should be WARMER than target (near ambient)
            # Heat flows INTO the cold pipe, raising surface temp toward ambient
            if surface_temp < target_surface_temp_f:
                # Need MORE insulation to raise surface temp
                low = mid
            else:
                # Can use less insulation
                high = mid

        return high

    def _analyze_vapor_barrier_requirements(
        self,
        input_data: InsulationInput,
        dew_point_f: float,
    ) -> VaporBarrierAnalysis:
        """
        Analyze vapor barrier requirements.

        For cold service below ambient dew point, vapor barrier is critical
        to prevent moisture migration into insulation.

        Args:
            input_data: Input data
            dew_point_f: Ambient dew point temperature

        Returns:
            VaporBarrierAnalysis
        """
        operating_temp = input_data.operating_temperature_f

        # Vapor barrier always required for cold service below dew point
        required = operating_temp < dew_point_f

        # Cryogenic service needs extra attention
        is_cryogenic = operating_temp < self.condensation.cryogenic_threshold_f

        # Recommended perm rating
        if is_cryogenic:
            perm_rating = 0.005  # Very low permeance
        elif operating_temp < 0:
            perm_rating = 0.01
        else:
            perm_rating = self.condensation.vapor_barrier_perm_rating

        # Location
        location = "innermost"  # Always on warm side (outer for cold service)

        # Material suggestions
        if is_cryogenic:
            materials = [
                "Polyisocyanurate with foil facing",
                "Cellular glass (no vapor barrier needed)",
                "Stainless steel jacket with sealed joints",
            ]
        else:
            materials = [
                "All Service Jacket (ASJ)",
                "Polyethylene film",
                "Aluminum foil laminate",
                "Mastic coating",
            ]

        # Check for condensation point within insulation
        condensation_point = None
        if input_data.insulation_layers:
            # Temperature profile through insulation could have point below dew point
            heat_loss_result = self.heat_loss_calc.calculate_heat_loss(input_data)
            temps = heat_loss_result.layer_temperatures_f

            cumulative_thickness = 0.0
            for i, temp in enumerate(temps):
                if temp < dew_point_f:
                    # Condensation could occur at this depth
                    if i > 0:
                        condensation_point = cumulative_thickness
                    break
                if i < len(input_data.insulation_layers):
                    cumulative_thickness += input_data.insulation_layers[i].thickness_in

        return VaporBarrierAnalysis(
            required=required,
            recommended_perm_rating=perm_rating,
            location=location,
            material_suggestions=materials,
            condensation_point_in_system=condensation_point,
        )

    def _select_cold_service_material(
        self,
        operating_temp_f: float,
    ) -> InsulationMaterial:
        """Select appropriate material for cold service."""
        # For cryogenic
        if operating_temp_f < -100:
            candidates = [
                "cellular_glass_7pcf",
                "polyurethane_2pcf",
                "aerogel_blanket_8pcf",
            ]
        # For cold
        elif operating_temp_f < 32:
            candidates = [
                "cellular_glass_7pcf",
                "polyurethane_2pcf",
                "elastomeric_4pcf",
                "phenolic_3pcf",
            ]
        else:
            candidates = [
                "elastomeric_4pcf",
                "fiberglass_3pcf",
                "mineral_wool_8pcf",
            ]

        for material_id in candidates:
            material = self.material_db.get_material(material_id)
            if material:
                return material

        # Default
        return self.material_db.get_material("cellular_glass_7pcf")

    def _check_vapor_barrier(self, input_data: InsulationInput) -> bool:
        """
        Check if current system has adequate vapor barrier.

        This is a simplified check - in production would need
        more detailed specification data.
        """
        # Check if any layer material has built-in vapor barrier
        for layer in input_data.insulation_layers:
            material = self.material_db.get_material(layer.material_id)
            if material:
                # Cellular glass and closed-cell foams don't need vapor barrier
                if material.moisture_resistant and not material.requires_vapor_barrier:
                    return True

        # Check jacketing
        if input_data.jacketing:
            # Metal jacketing with proper sealing can act as vapor barrier
            if input_data.jacketing.jacketing_type.value in ["aluminum", "stainless_steel"]:
                return True

        return False

    def calculate_design_conditions(
        self,
        location_climate: str = "humid_subtropical",
    ) -> Dict[str, float]:
        """
        Get design conditions for condensation prevention.

        Returns worst-case ambient conditions for design.

        Args:
            location_climate: Climate zone

        Returns:
            Design conditions dictionary
        """
        # Design conditions by climate zone (99th percentile humidity)
        climate_data = {
            "humid_subtropical": {
                "design_temp_f": 95,
                "design_rh_pct": 95,
                "dew_point_f": 78,
            },
            "humid_continental": {
                "design_temp_f": 90,
                "design_rh_pct": 90,
                "dew_point_f": 75,
            },
            "marine": {
                "design_temp_f": 85,
                "design_rh_pct": 85,
                "dew_point_f": 68,
            },
            "arid": {
                "design_temp_f": 105,
                "design_rh_pct": 50,
                "dew_point_f": 60,
            },
            "tropical": {
                "design_temp_f": 95,
                "design_rh_pct": 98,
                "dew_point_f": 82,
            },
        }

        return climate_data.get(
            location_climate,
            climate_data["humid_subtropical"]
        )

    def generate_condensation_report(
        self,
        input_data: InsulationInput,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive condensation analysis report.

        Args:
            input_data: Input data

        Returns:
            Report dictionary
        """
        result = self.analyze(input_data)

        # Calculate dew point for report
        dew_point = self._calculate_dew_point(
            input_data.ambient_temperature_f,
            input_data.relative_humidity_pct,
        )

        report = {
            "item_id": input_data.item_id,
            "item_name": input_data.item_name,
            "service_type": input_data.service_type.value,
            "operating_temperature_f": input_data.operating_temperature_f,
            "ambient_conditions": {
                "temperature_f": input_data.ambient_temperature_f,
                "relative_humidity_pct": input_data.relative_humidity_pct,
                "dew_point_f": round(dew_point.dew_point_f, 1),
                "dew_point_c": round(dew_point.dew_point_c, 1),
            },
            "surface_analysis": {
                "current_surface_temp_f": result.surface_temperature_f,
                "margin_above_dew_point_f": result.margin_above_dew_point_f,
            },
            "condensation_risk": {
                "at_risk": result.condensation_risk,
                "risk_level": result.condensation_risk_level,
            },
            "insulation_requirements": {
                "current_thickness_in": result.current_thickness_in,
                "minimum_thickness_in": result.minimum_thickness_for_prevention_in,
                "additional_needed_in": result.additional_thickness_needed_in,
            },
            "vapor_barrier": {
                "required": result.vapor_barrier_required,
                "location": result.vapor_barrier_location,
                "adequate_current": result.has_adequate_vapor_barrier,
            },
            "recommendations": [],
        }

        # Add recommendations
        if result.condensation_risk:
            report["recommendations"].append(
                f"Add {result.additional_thickness_needed_in} inches of insulation"
            )
        if result.vapor_barrier_required and not result.has_adequate_vapor_barrier:
            report["recommendations"].append(
                "Install vapor barrier on warm side of insulation"
            )
        if input_data.operating_temperature_f < -100:
            report["recommendations"].append(
                "Use cellular glass or aerogel for cryogenic service"
            )

        return report

    @property
    def calculation_count(self) -> int:
        """Get total calculations performed."""
        return self._calculation_count
