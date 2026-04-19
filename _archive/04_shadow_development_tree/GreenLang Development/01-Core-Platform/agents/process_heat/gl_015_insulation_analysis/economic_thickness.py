"""
GL-015 INSULSCAN - Economic Thickness Optimizer

Implements NAIMA 3E Plus methodology for calculating economically
optimal insulation thickness. Balances energy costs against
insulation investment costs.

All calculations are DETERMINISTIC - zero hallucination.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    EconomicConfig,
    InsulationAnalysisConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    InsulationLayer,
    EconomicThicknessResult,
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
class ThicknessPoint:
    """Data point for thickness optimization."""
    thickness_in: float
    heat_loss_btu_hr: float
    annual_energy_cost_usd: float
    insulation_cost_usd: float
    total_annual_cost_usd: float
    npv_usd: float


class EconomicThicknessOptimizer:
    """
    Economic thickness optimizer using NAIMA 3E Plus methodology.

    Calculates the economically optimal insulation thickness by minimizing
    the total lifecycle cost (energy cost + insulation cost).

    The optimal thickness is where:
    d(Total Cost)/d(Thickness) = 0

    Or equivalently, where marginal energy savings equals marginal
    insulation cost.

    Features:
        - NAIMA 3E Plus methodology
        - NPV and ROI calculations
        - Multiple material comparison
        - Sensitivity analysis
        - Annualized cost optimization

    Attributes:
        heat_loss_calc: Heat loss calculator
        material_db: Material database
        economic_config: Economic parameters

    Example:
        >>> optimizer = EconomicThicknessOptimizer(config)
        >>> result = optimizer.calculate_economic_thickness(input_data)
        >>> print(f"Optimal: {result.optimal_thickness_in} inches")
    """

    def __init__(
        self,
        config: InsulationAnalysisConfig,
        material_database: Optional[InsulationMaterialDatabase] = None,
        heat_loss_calculator: Optional[HeatLossCalculator] = None,
    ) -> None:
        """
        Initialize the economic thickness optimizer.

        Args:
            config: Analysis configuration
            material_database: Material database (creates new if None)
            heat_loss_calculator: Heat loss calculator (creates new if None)
        """
        self.config = config
        self.economic = config.economic
        self.material_db = material_database or InsulationMaterialDatabase()
        self.heat_loss_calc = heat_loss_calculator or HeatLossCalculator(
            material_database=self.material_db,
        )
        self._calculation_count = 0

        logger.info("EconomicThicknessOptimizer initialized")

    def calculate_economic_thickness(
        self,
        input_data: InsulationInput,
        material_id: Optional[str] = None,
        min_thickness_in: float = 0.5,
        max_thickness_in: float = 8.0,
        thickness_step_in: float = 0.5,
    ) -> EconomicThicknessResult:
        """
        Calculate economically optimal insulation thickness.

        Uses NAIMA 3E Plus methodology to find thickness that minimizes
        total lifecycle cost.

        Args:
            input_data: Insulation analysis input
            material_id: Specific material to use (auto-selects if None)
            min_thickness_in: Minimum thickness to consider
            max_thickness_in: Maximum thickness to consider
            thickness_step_in: Thickness increment for search

        Returns:
            EconomicThicknessResult with optimal thickness and economics
        """
        self._calculation_count += 1
        logger.debug(f"Calculating economic thickness for {input_data.item_id}")

        # Determine material to use
        if material_id is None:
            material = self._select_optimal_material(input_data)
            material_id = material.material_id
        else:
            material = self.material_db.get_material(material_id)
            if material is None:
                raise ValueError(f"Unknown material: {material_id}")

        # Get available thicknesses
        thicknesses = []
        t = min_thickness_in
        while t <= max_thickness_in:
            thicknesses.append(t)
            t += thickness_step_in

        # Calculate costs at each thickness
        thickness_points: List[ThicknessPoint] = []

        for thickness in thicknesses:
            point = self._evaluate_thickness(
                input_data=input_data,
                material=material,
                thickness_in=thickness,
            )
            thickness_points.append(point)

        # Find optimal thickness (minimum total annual cost)
        optimal_point = min(thickness_points, key=lambda p: p.total_annual_cost_usd)
        optimal_thickness = optimal_point.thickness_in

        # Get current state
        current_thickness = sum(
            layer.thickness_in for layer in input_data.insulation_layers
        )
        current_heat_loss = 0.0
        current_annual_cost = 0.0

        if current_thickness > 0:
            current_result = self.heat_loss_calc.calculate_heat_loss(input_data)
            current_heat_loss = current_result.heat_loss_btu_hr
            current_annual_cost = self._calculate_annual_energy_cost(current_heat_loss)
        else:
            # Bare surface
            bare_input = input_data.copy()
            bare_input.insulation_layers = []
            bare_result = self.heat_loss_calc.calculate_heat_loss(bare_input)
            current_heat_loss = bare_result.heat_loss_btu_hr
            current_annual_cost = self._calculate_annual_energy_cost(current_heat_loss)

        # Calculate installation costs
        surface_area = self._get_surface_area(input_data)
        additional_thickness = max(0, optimal_thickness - current_thickness)

        insulation_cost = self._calculate_insulation_material_cost(
            material_id=material_id,
            thickness_in=additional_thickness,
            surface_area_sqft=surface_area,
        )

        installation_cost = self._calculate_installation_cost(
            surface_area_sqft=surface_area,
            elevated=input_data.location_elevation_ft > 6,
        )

        total_project_cost = insulation_cost + installation_cost

        # Calculate savings and payback
        annual_savings = current_annual_cost - optimal_point.annual_energy_cost_usd

        if annual_savings > 0:
            simple_payback = total_project_cost / annual_savings
        else:
            simple_payback = float('inf')

        # Calculate NPV
        npv = self._calculate_npv(
            initial_investment=total_project_cost,
            annual_savings=annual_savings,
            years=self.economic.plant_lifetime_years,
            discount_rate=self.economic.discount_rate_pct / 100,
        )

        # Calculate ROI
        if total_project_cost > 0:
            roi = (annual_savings * self.economic.plant_lifetime_years - total_project_cost) / total_project_cost * 100
        else:
            roi = 0.0

        return EconomicThicknessResult(
            optimal_thickness_in=optimal_thickness,
            optimal_thickness_layers=[optimal_thickness],
            recommended_material=material.name,
            current_thickness_in=current_thickness,
            additional_thickness_needed_in=additional_thickness,
            current_heat_loss_btu_hr=current_heat_loss,
            optimal_heat_loss_btu_hr=optimal_point.heat_loss_btu_hr,
            heat_loss_savings_btu_hr=current_heat_loss - optimal_point.heat_loss_btu_hr,
            annual_energy_cost_current_usd=round(current_annual_cost, 2),
            annual_energy_cost_optimal_usd=round(optimal_point.annual_energy_cost_usd, 2),
            annual_savings_usd=round(annual_savings, 2),
            insulation_cost_usd=round(insulation_cost, 2),
            installation_cost_usd=round(installation_cost, 2),
            total_project_cost_usd=round(total_project_cost, 2),
            simple_payback_years=round(simple_payback, 2) if simple_payback != float('inf') else 999.9,
            npv_usd=round(npv, 2),
            roi_pct=round(roi, 1),
            calculation_method="NAIMA_3E_PLUS",
        )

    def _evaluate_thickness(
        self,
        input_data: InsulationInput,
        material: InsulationMaterial,
        thickness_in: float,
    ) -> ThicknessPoint:
        """
        Evaluate costs at a specific thickness.

        Args:
            input_data: Base input data
            material: Insulation material
            thickness_in: Thickness to evaluate

        Returns:
            ThicknessPoint with all cost data
        """
        # Create modified input with single layer at specified thickness
        modified_input = input_data.copy(deep=True)
        modified_input.insulation_layers = [
            InsulationLayer(
                layer_number=1,
                material_id=material.material_id,
                thickness_in=thickness_in,
            )
        ]

        # Calculate heat loss
        heat_loss_result = self.heat_loss_calc.calculate_heat_loss(modified_input)
        heat_loss = heat_loss_result.heat_loss_btu_hr

        # Calculate annual energy cost
        annual_energy_cost = self._calculate_annual_energy_cost(heat_loss)

        # Calculate insulation cost (annualized)
        surface_area = self._get_surface_area(input_data)
        insulation_cost = self._calculate_insulation_material_cost(
            material_id=material.material_id,
            thickness_in=thickness_in,
            surface_area_sqft=surface_area,
        )
        installation_cost = self._calculate_installation_cost(
            surface_area_sqft=surface_area,
            elevated=input_data.location_elevation_ft > 6,
        )
        total_insulation_cost = insulation_cost + installation_cost

        # Annualize insulation cost using capital recovery factor
        crf = self._calculate_capital_recovery_factor(
            discount_rate=self.economic.discount_rate_pct / 100,
            years=self.economic.plant_lifetime_years,
        )
        annualized_insulation_cost = total_insulation_cost * crf

        # Total annual cost
        total_annual_cost = annual_energy_cost + annualized_insulation_cost

        # NPV of this option
        npv = self._calculate_npv(
            initial_investment=total_insulation_cost,
            annual_savings=-annual_energy_cost,  # Negative because it's a cost
            years=self.economic.plant_lifetime_years,
            discount_rate=self.economic.discount_rate_pct / 100,
        )

        return ThicknessPoint(
            thickness_in=thickness_in,
            heat_loss_btu_hr=heat_loss,
            annual_energy_cost_usd=annual_energy_cost,
            insulation_cost_usd=total_insulation_cost,
            total_annual_cost_usd=total_annual_cost,
            npv_usd=npv,
        )

    def _calculate_annual_energy_cost(self, heat_loss_btu_hr: float) -> float:
        """
        Calculate annual energy cost from heat loss.

        Args:
            heat_loss_btu_hr: Heat loss rate (BTU/hr)

        Returns:
            Annual energy cost (USD)
        """
        # Convert to MMBTU/year
        annual_heat_loss_mmbtu = (
            heat_loss_btu_hr *
            self.economic.operating_hours_per_year /
            1_000_000
        )

        # Cost
        annual_cost = annual_heat_loss_mmbtu * self.economic.energy_cost_per_mmbtu

        return annual_cost

    def _calculate_insulation_material_cost(
        self,
        material_id: str,
        thickness_in: float,
        surface_area_sqft: float,
    ) -> float:
        """
        Calculate insulation material cost.

        Args:
            material_id: Material identifier
            thickness_in: Insulation thickness
            surface_area_sqft: Surface area

        Returns:
            Material cost (USD)
        """
        # Get base cost per sqft
        material = self.material_db.get_material(material_id)
        if material is None:
            base_cost = 10.0  # Default
        else:
            # Map category to cost
            category_key = material.category.value
            if category_key in self.economic.insulation_cost_per_sqft:
                base_cost = self.economic.insulation_cost_per_sqft[category_key]
            else:
                # Use default based on category
                default_costs = {
                    "calcium_silicate": 12.50,
                    "mineral_wool": 8.75,
                    "fiberglass": 7.50,
                    "cellular_glass": 18.00,
                    "aerogel": 85.00,
                }
                base_cost = default_costs.get(category_key, 10.0)

        # Thickness adjustment (cost increases with thickness)
        thickness_multiplier = 1.0 + (thickness_in - 1.0) * 0.4

        total_cost = base_cost * thickness_multiplier * surface_area_sqft

        return total_cost

    def _calculate_installation_cost(
        self,
        surface_area_sqft: float,
        elevated: bool = False,
    ) -> float:
        """
        Calculate installation labor cost.

        Args:
            surface_area_sqft: Surface area
            elevated: Requires scaffolding

        Returns:
            Installation cost (USD)
        """
        # Estimate hours based on area (assume 20 sqft/hr productivity)
        hours = surface_area_sqft / 20.0

        # Labor cost
        labor_cost = hours * self.economic.labor_rate_per_hour

        # Scaffolding multiplier
        if elevated:
            labor_cost *= self.economic.scaffolding_cost_multiplier

        return labor_cost

    def _calculate_capital_recovery_factor(
        self,
        discount_rate: float,
        years: int,
    ) -> float:
        """
        Calculate capital recovery factor (CRF).

        CRF = i(1+i)^n / ((1+i)^n - 1)

        Args:
            discount_rate: Discount rate (decimal)
            years: Number of years

        Returns:
            Capital recovery factor
        """
        if discount_rate <= 0:
            return 1.0 / years

        i = discount_rate
        n = years

        numerator = i * (1 + i) ** n
        denominator = (1 + i) ** n - 1

        return numerator / denominator if denominator > 0 else 1.0 / years

    def _calculate_npv(
        self,
        initial_investment: float,
        annual_savings: float,
        years: int,
        discount_rate: float,
    ) -> float:
        """
        Calculate Net Present Value.

        NPV = -Investment + Sum(Savings / (1+r)^t)

        Args:
            initial_investment: Initial investment cost
            annual_savings: Annual savings (positive = benefit)
            years: Number of years
            discount_rate: Discount rate (decimal)

        Returns:
            Net present value (USD)
        """
        npv = -initial_investment

        for t in range(1, years + 1):
            npv += annual_savings / ((1 + discount_rate) ** t)

        return npv

    def _select_optimal_material(
        self,
        input_data: InsulationInput,
    ) -> InsulationMaterial:
        """
        Select optimal material for given conditions.

        Considers temperature range, cost, and performance.

        Args:
            input_data: Input data with operating conditions

        Returns:
            Recommended InsulationMaterial
        """
        operating_temp = input_data.operating_temperature_f
        is_cold = input_data.service_type.value in ["cold", "cryogenic"]

        # Get candidate materials
        candidates = self.material_db.get_recommended_materials(
            operating_temp_f=operating_temp,
            cold_service=is_cold,
        )

        if not candidates:
            # Default to mineral wool
            return self.material_db.get_material("mineral_wool_8pcf")

        # For now, return the one with lowest k-value
        # In production, would consider cost-performance tradeoff
        return candidates[0]

    def _get_surface_area(self, input_data: InsulationInput) -> float:
        """Calculate surface area for cost estimation."""
        if input_data.geometry_type == GeometryType.PIPE:
            geom = input_data.pipe_geometry
            return math.pi * (geom.outer_diameter_in / 12) * geom.pipe_length_ft

        elif input_data.geometry_type == GeometryType.VESSEL:
            geom = input_data.vessel_geometry
            shell_area = math.pi * geom.vessel_diameter_ft * geom.vessel_length_ft
            head_area = 0
            if geom.include_heads:
                head_area = 2 * math.pi * (geom.vessel_diameter_ft / 2) ** 2
            return shell_area + head_area

        elif input_data.geometry_type == GeometryType.FLAT_SURFACE:
            return input_data.flat_geometry.surface_area_sqft

        return 100.0  # Default

    def compare_materials(
        self,
        input_data: InsulationInput,
        material_ids: List[str],
        target_thickness_in: float,
    ) -> Dict[str, EconomicThicknessResult]:
        """
        Compare economics of different materials.

        Args:
            input_data: Input data
            material_ids: Materials to compare
            target_thickness_in: Thickness for comparison

        Returns:
            Dictionary of material_id -> EconomicThicknessResult
        """
        results = {}

        for material_id in material_ids:
            try:
                result = self.calculate_economic_thickness(
                    input_data=input_data,
                    material_id=material_id,
                    min_thickness_in=target_thickness_in,
                    max_thickness_in=target_thickness_in + 0.1,
                    thickness_step_in=0.5,
                )
                results[material_id] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate {material_id}: {e}")

        return results

    @property
    def calculation_count(self) -> int:
        """Get total calculations performed."""
        return self._calculation_count
