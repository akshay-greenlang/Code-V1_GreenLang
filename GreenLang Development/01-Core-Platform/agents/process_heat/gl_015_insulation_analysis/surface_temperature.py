"""
GL-015 INSULSCAN - Surface Temperature Calculator

Calculates surface temperatures and checks OSHA compliance for
personnel protection (60C/140F touchable surface limit).

Reference: OSHA 29 CFR 1910.261, ASTM C1055

All calculations are DETERMINISTIC - zero hallucination.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    SafetyConfig,
    InsulationAnalysisConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    InsulationLayer,
    SurfaceTemperatureResult,
    GeometryType,
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
class BurnRiskAssessment:
    """Burn risk assessment from surface contact."""
    risk_level: str  # none, low, medium, high, extreme
    time_to_injury_sec: Optional[float]
    recommended_protection: List[str]
    contact_safe_duration_sec: Optional[float]


class SurfaceTemperatureCalculator:
    """
    Surface temperature calculator with OSHA compliance checking.

    Calculates outer surface temperature of insulated systems and
    determines compliance with OSHA touchable surface limits.
    Also calculates minimum insulation thickness for compliance.

    OSHA Limits:
        - 60C (140F) for touchable surfaces per 29 CFR 1910.261
        - Based on 1-second contact threshold for burn injury

    Features:
        - Surface temperature calculation
        - OSHA 60C compliance checking
        - Minimum thickness for compliance
        - Burn risk assessment
        - Personnel protection recommendations

    Attributes:
        heat_loss_calc: Heat loss calculator
        material_db: Material database
        safety_config: Safety configuration

    Example:
        >>> calc = SurfaceTemperatureCalculator(config)
        >>> result = calc.check_compliance(input_data)
        >>> if not result.is_compliant:
        ...     print(f"Need {result.additional_thickness_needed_in} more inches")
    """

    # OSHA touchable surface limits
    OSHA_LIMIT_C = 60.0
    OSHA_LIMIT_F = 140.0

    # Contact burn thresholds (ASTM C1055)
    # Time to burn injury (seconds) at various temperatures
    BURN_THRESHOLDS_METAL = {
        48.0: 60.0,    # 48C: 60 seconds
        51.0: 10.0,    # 51C: 10 seconds
        54.0: 5.0,     # 54C: 5 seconds
        60.0: 1.0,     # 60C: 1 second
        66.0: 0.5,     # 66C: 0.5 seconds
        70.0: 0.1,     # 70C: 0.1 seconds
    }

    BURN_THRESHOLDS_NON_METAL = {
        48.0: 480.0,   # 48C: 8 minutes
        51.0: 60.0,    # 51C: 60 seconds
        60.0: 10.0,    # 60C: 10 seconds
        66.0: 5.0,     # 66C: 5 seconds
        70.0: 1.0,     # 70C: 1 second
    }

    def __init__(
        self,
        config: InsulationAnalysisConfig,
        material_database: Optional[InsulationMaterialDatabase] = None,
        heat_loss_calculator: Optional[HeatLossCalculator] = None,
    ) -> None:
        """
        Initialize the surface temperature calculator.

        Args:
            config: Analysis configuration
            material_database: Material database
            heat_loss_calculator: Heat loss calculator
        """
        self.config = config
        self.safety = config.safety
        self.material_db = material_database or InsulationMaterialDatabase()
        self.heat_loss_calc = heat_loss_calculator or HeatLossCalculator(
            material_database=self.material_db,
        )
        self._calculation_count = 0

        logger.info("SurfaceTemperatureCalculator initialized")

    def check_compliance(
        self,
        input_data: InsulationInput,
    ) -> SurfaceTemperatureResult:
        """
        Check OSHA surface temperature compliance.

        Args:
            input_data: Insulation analysis input

        Returns:
            SurfaceTemperatureResult with compliance status
        """
        self._calculation_count += 1
        logger.debug(f"Checking surface temperature for {input_data.item_id}")

        # Calculate current heat loss and surface temperature
        heat_loss_result = self.heat_loss_calc.calculate_heat_loss(input_data)
        surface_temp_f = heat_loss_result.outer_surface_temperature_f
        surface_temp_c = (surface_temp_f - 32) * 5 / 9

        # Get OSHA limits from config
        osha_limit_f = self.safety.max_touch_temperature_f
        osha_limit_c = self.safety.max_touch_temperature_c

        # Check compliance
        is_compliant = surface_temp_f <= osha_limit_f
        margin_f = osha_limit_f - surface_temp_f
        margin_c = osha_limit_c - surface_temp_c

        # Get current thickness
        current_thickness = sum(
            layer.thickness_in for layer in input_data.insulation_layers
        )

        # Calculate minimum thickness for compliance if not compliant
        min_thickness = None
        additional_needed = 0.0

        if not is_compliant or margin_f < 10:
            min_thickness = self._calculate_minimum_thickness_for_compliance(
                input_data=input_data,
                target_surface_temp_f=osha_limit_f - 10,  # 10F safety margin
            )
            additional_needed = max(0, min_thickness - current_thickness)

        # Assess burn risk
        burn_risk = self._assess_burn_risk(
            surface_temp_c=surface_temp_c,
            jacketing_material=input_data.jacketing.jacketing_type.value if input_data.jacketing else "aluminum",
        )

        # Personnel protection requirements
        protection_required = surface_temp_f > self.safety.warning_threshold_c * 9/5 + 32
        recommended_protection = burn_risk.recommended_protection

        return SurfaceTemperatureResult(
            calculated_surface_temp_f=round(surface_temp_f, 1),
            calculated_surface_temp_c=round(surface_temp_c, 1),
            osha_limit_temp_f=osha_limit_f,
            osha_limit_temp_c=osha_limit_c,
            is_compliant=is_compliant,
            margin_f=round(margin_f, 1),
            margin_c=round(margin_c, 1),
            minimum_thickness_for_compliance_in=round(min_thickness, 2) if min_thickness else None,
            current_thickness_in=current_thickness,
            additional_thickness_needed_in=round(additional_needed, 2),
            contact_burn_risk=burn_risk.risk_level,
            time_to_burn_injury_sec=burn_risk.time_to_injury_sec,
            personnel_protection_required=protection_required,
            recommended_protection=recommended_protection,
        )

    def _calculate_minimum_thickness_for_compliance(
        self,
        input_data: InsulationInput,
        target_surface_temp_f: float,
    ) -> float:
        """
        Calculate minimum insulation thickness for OSHA compliance.

        Uses bisection method to find thickness that achieves target
        surface temperature.

        Args:
            input_data: Input data
            target_surface_temp_f: Target surface temperature

        Returns:
            Minimum thickness (inches)
        """
        # Determine material to use
        if input_data.insulation_layers:
            material_id = input_data.insulation_layers[0].material_id
        else:
            # Select appropriate material
            material = self._select_material_for_temperature(
                input_data.operating_temperature_f
            )
            material_id = material.material_id

        # Bisection search
        low = 0.5  # Minimum practical thickness
        high = 12.0  # Maximum practical thickness
        tolerance = 0.1  # inches

        while high - low > tolerance:
            mid = (low + high) / 2

            # Create test input with this thickness
            test_input = input_data.copy(deep=True)
            test_input.insulation_layers = [
                InsulationLayer(
                    layer_number=1,
                    material_id=material_id,
                    thickness_in=mid,
                )
            ]

            # Calculate surface temperature
            result = self.heat_loss_calc.calculate_heat_loss(test_input)
            surface_temp = result.outer_surface_temperature_f

            if surface_temp > target_surface_temp_f:
                # Need more insulation
                low = mid
            else:
                # Can use less insulation
                high = mid

        return high

    def calculate_surface_temperature(
        self,
        input_data: InsulationInput,
    ) -> float:
        """
        Calculate outer surface temperature.

        Args:
            input_data: Insulation analysis input

        Returns:
            Surface temperature (F)
        """
        result = self.heat_loss_calc.calculate_heat_loss(input_data)
        return result.outer_surface_temperature_f

    def _assess_burn_risk(
        self,
        surface_temp_c: float,
        jacketing_material: str = "aluminum",
    ) -> BurnRiskAssessment:
        """
        Assess burn risk from surface contact.

        Based on ASTM C1055 contact burn thresholds.

        Args:
            surface_temp_c: Surface temperature (C)
            jacketing_material: Surface material type

        Returns:
            BurnRiskAssessment
        """
        # Select threshold table based on material
        if jacketing_material.lower() in ["aluminum", "stainless_steel", "galvanized"]:
            thresholds = self.BURN_THRESHOLDS_METAL
        else:
            thresholds = self.BURN_THRESHOLDS_NON_METAL

        # Find time to injury
        time_to_injury = None
        temps = sorted(thresholds.keys())

        if surface_temp_c <= temps[0]:
            time_to_injury = None  # No burn risk
        elif surface_temp_c >= temps[-1]:
            time_to_injury = thresholds[temps[-1]]
        else:
            # Interpolate
            for i in range(len(temps) - 1):
                if temps[i] <= surface_temp_c < temps[i + 1]:
                    t1, t2 = temps[i], temps[i + 1]
                    time1, time2 = thresholds[t1], thresholds[t2]
                    # Log interpolation for time
                    if time1 > 0 and time2 > 0:
                        log_time = (
                            math.log(time1) +
                            (math.log(time2) - math.log(time1)) *
                            (surface_temp_c - t1) / (t2 - t1)
                        )
                        time_to_injury = math.exp(log_time)
                    break

        # Determine risk level
        if time_to_injury is None or surface_temp_c < 43:
            risk_level = "none"
        elif time_to_injury > 60:
            risk_level = "low"
        elif time_to_injury > 10:
            risk_level = "medium"
        elif time_to_injury > 1:
            risk_level = "high"
        else:
            risk_level = "extreme"

        # Recommended protection
        protection = []
        if risk_level == "none":
            protection = []
        elif risk_level == "low":
            protection = ["Post warning sign"]
        elif risk_level == "medium":
            protection = ["Post warning sign", "Wear gloves when touching"]
        elif risk_level == "high":
            protection = [
                "Post danger sign",
                "Wear heat-resistant gloves",
                "Install guard or barrier",
            ]
        else:  # extreme
            protection = [
                "Install permanent barrier/guard",
                "Lockout/tagout for contact",
                "Add additional insulation",
                "PPE required: Heat-resistant suit",
            ]

        return BurnRiskAssessment(
            risk_level=risk_level,
            time_to_injury_sec=round(time_to_injury, 2) if time_to_injury else None,
            recommended_protection=protection,
            contact_safe_duration_sec=max(0, (time_to_injury or 999) - 0.5) if time_to_injury else None,
        )

    def _select_material_for_temperature(
        self,
        operating_temp_f: float,
    ) -> InsulationMaterial:
        """Select appropriate material for operating temperature."""
        candidates = self.material_db.get_recommended_materials(
            operating_temp_f=operating_temp_f,
            cold_service=operating_temp_f < 60,
        )

        if candidates:
            return candidates[0]

        # Default fallback
        return self.material_db.get_material("mineral_wool_8pcf")

    def calculate_thickness_for_target_temperature(
        self,
        input_data: InsulationInput,
        target_surface_temp_f: float,
        material_id: Optional[str] = None,
    ) -> float:
        """
        Calculate thickness needed to achieve target surface temperature.

        Args:
            input_data: Input data
            target_surface_temp_f: Target surface temperature
            material_id: Material to use (auto-selects if None)

        Returns:
            Required thickness (inches)
        """
        if material_id is None:
            material = self._select_material_for_temperature(
                input_data.operating_temperature_f
            )
            material_id = material.material_id

        # Bisection search
        low = 0.5
        high = 12.0
        tolerance = 0.1

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

            if surface_temp > target_surface_temp_f:
                low = mid
            else:
                high = mid

        return round(high, 1)

    def generate_compliance_report(
        self,
        input_data: InsulationInput,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Args:
            input_data: Input data

        Returns:
            Dictionary with compliance report
        """
        result = self.check_compliance(input_data)

        report = {
            "item_id": input_data.item_id,
            "item_name": input_data.item_name,
            "operating_temperature_f": input_data.operating_temperature_f,
            "ambient_temperature_f": input_data.ambient_temperature_f,
            "current_insulation_thickness_in": result.current_thickness_in,
            "surface_temperature": {
                "calculated_f": result.calculated_surface_temp_f,
                "calculated_c": result.calculated_surface_temp_c,
                "osha_limit_f": result.osha_limit_temp_f,
                "osha_limit_c": result.osha_limit_temp_c,
            },
            "compliance": {
                "is_compliant": result.is_compliant,
                "margin_f": result.margin_f,
                "margin_c": result.margin_c,
                "status": "PASS" if result.is_compliant else "FAIL",
            },
            "corrective_action": None,
            "burn_risk": {
                "level": result.contact_burn_risk,
                "time_to_injury_sec": result.time_to_burn_injury_sec,
            },
            "personnel_protection": {
                "required": result.personnel_protection_required,
                "recommendations": result.recommended_protection,
            },
        }

        if not result.is_compliant:
            report["corrective_action"] = {
                "minimum_thickness_in": result.minimum_thickness_for_compliance_in,
                "additional_thickness_needed_in": result.additional_thickness_needed_in,
                "urgency": "immediate" if result.contact_burn_risk == "extreme" else "scheduled",
            }

        return report

    @property
    def calculation_count(self) -> int:
        """Get total calculations performed."""
        return self._calculation_count
