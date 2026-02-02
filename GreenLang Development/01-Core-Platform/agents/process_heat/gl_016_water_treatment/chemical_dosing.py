"""
GL-016 WATERGUARD Agent - Chemical Dosing Optimization Module

Implements chemical dosing optimization including:
- Oxygen scavenger dosing (sulfite, hydrazine, carbohydrazide)
- Phosphate dosing for scale prevention
- Amine dosing for condensate protection
- Cost optimization

All calculations are deterministic with zero hallucination.

References:
    - ASME Water Treatment Guidelines
    - Chemical Supplier Technical Bulletins
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    ChemicalDosingInput,
    ChemicalDosingOutput,
    ChemicalType,
    TreatmentProgram,
)
from greenlang.agents.process_heat.gl_016_water_treatment.config import (
    get_scavenger_config,
    get_amine_config,
    get_phosphate_config,
    OxygenScavengerConfig,
    AmineConfig,
    PhosphateTreatmentConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class DosingConstants:
    """Constants for chemical dosing calculations."""

    # Conversion factors
    PPM_TO_LB_PER_KGAL = 0.00834
    LB_PER_GAL_WATER = 8.34

    # Hours
    HOURS_PER_DAY = 24
    HOURS_PER_YEAR = 8760

    # Oxygen scavenger stoichiometry (lb scavenger per lb O2)
    SCAVENGER_STOICH = {
        ChemicalType.SULFITE: 7.88,
        ChemicalType.HYDRAZINE: 1.0,
        ChemicalType.CARBOHYDRAZIDE: 1.39,
        ChemicalType.ERYTHORBIC_ACID: 5.5,
    }

    # Recommended excess factors
    SCAVENGER_EXCESS = {
        ChemicalType.SULFITE: 1.5,  # 50% excess
        ChemicalType.HYDRAZINE: 2.0,  # 100% excess
        ChemicalType.CARBOHYDRAZIDE: 1.75,
        ChemicalType.ERYTHORBIC_ACID: 1.5,
    }


# =============================================================================
# CHEMICAL DOSING OPTIMIZER CLASS
# =============================================================================

class ChemicalDosingOptimizer:
    """
    Optimizes chemical dosing for water treatment.

    Calculates optimal dosing rates for oxygen scavengers, phosphate,
    and amines based on water conditions and treatment targets.

    Example:
        >>> optimizer = ChemicalDosingOptimizer()
        >>> result = optimizer.optimize(dosing_input)
        >>> print(f"Scavenger rate: {result.scavenger_feed_rate_lb_hr} lb/hr")
    """

    def __init__(self) -> None:
        """Initialize ChemicalDosingOptimizer."""
        logger.info("ChemicalDosingOptimizer initialized")

    def optimize(self, input_data: ChemicalDosingInput) -> ChemicalDosingOutput:
        """
        Optimize chemical dosing rates.

        Args:
            input_data: Chemical dosing input data

        Returns:
            ChemicalDosingOutput with optimized dosing rates
        """
        start_time = datetime.now(timezone.utc)
        logger.debug("Optimizing chemical dosing")

        # Calculate oxygen scavenger requirements
        scavenger_dose, scavenger_rate, scavenger_ratio = self._calculate_scavenger_dosing(
            input_data
        )
        scavenger_change = scavenger_dose - input_data.current_scavenger_dose_ppm

        # Calculate phosphate requirements
        phosphate_dose, phosphate_rate = self._calculate_phosphate_dosing(input_data)
        phosphate_change = phosphate_dose - input_data.current_phosphate_dose_ppm

        # Calculate amine requirements
        amine_dose, amine_rate, amine_change = self._calculate_amine_dosing(input_data)

        # Calculate current and optimized costs
        current_cost, optimized_cost = self._calculate_costs(
            input_data,
            scavenger_dose,
            phosphate_dose,
            amine_dose,
        )
        cost_savings = current_cost - optimized_cost
        annual_savings = cost_savings * 365

        # Check if within recommended ranges
        within_ranges = self._check_dosing_ranges(
            input_data,
            scavenger_dose,
            phosphate_dose,
            amine_dose,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            input_data,
            scavenger_dose,
            scavenger_change,
            phosphate_dose,
            phosphate_change,
            amine_dose,
            amine_change,
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(input_data)
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return ChemicalDosingOutput(
            timestamp=datetime.now(timezone.utc),
            # Scavenger
            scavenger_dose_recommended_ppm=round(scavenger_dose, 2),
            scavenger_dose_change_ppm=round(scavenger_change, 2),
            scavenger_feed_rate_lb_hr=round(scavenger_rate, 3),
            scavenger_ratio_to_o2=round(scavenger_ratio, 1),
            # Phosphate
            phosphate_dose_recommended_ppm=round(phosphate_dose, 2),
            phosphate_dose_change_ppm=round(phosphate_change, 2),
            phosphate_feed_rate_lb_hr=round(phosphate_rate, 3),
            # Amine
            amine_dose_recommended_ppm=round(amine_dose, 2) if amine_dose else None,
            amine_dose_change_ppm=round(amine_change, 2) if amine_change else None,
            amine_feed_rate_lb_hr=round(amine_rate, 3) if amine_rate else None,
            # Costs
            current_chemical_cost_per_day=round(current_cost, 2),
            optimized_chemical_cost_per_day=round(optimized_cost, 2),
            cost_savings_per_day=round(cost_savings, 2),
            annual_savings_usd=round(annual_savings, 0),
            # Status
            within_recommended_ranges=within_ranges,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def _calculate_scavenger_dosing(
        self, input_data: ChemicalDosingInput
    ) -> Tuple[float, float, float]:
        """
        Calculate oxygen scavenger dosing requirement.

        Returns:
            Tuple of (dose_ppm, feed_rate_lb_hr, ratio_to_o2)
        """
        scavenger_type = input_data.current_scavenger_type
        feedwater_do = input_data.feedwater_do_ppb
        feedwater_flow = input_data.feedwater_flow_lb_hr
        target_residual = input_data.target_scavenger_residual_ppm

        # Get scavenger configuration
        config = get_scavenger_config(scavenger_type)

        # Get stoichiometric ratio
        if config:
            stoich_ratio = config.stoichiometric_ratio
            excess_factor = 1 + (config.recommended_excess_pct / 100)
        else:
            stoich_ratio = DosingConstants.SCAVENGER_STOICH.get(
                scavenger_type, 7.88  # Default to sulfite
            )
            excess_factor = DosingConstants.SCAVENGER_EXCESS.get(
                scavenger_type, 1.5
            )

        # Convert O2 from ppb to ppm
        o2_ppm = feedwater_do / 1000

        # Calculate stoichiometric requirement
        # Scavenger needed (ppm) = O2 (ppm) * stoichiometric ratio
        stoich_dose = o2_ppm * stoich_ratio

        # Apply excess factor
        reaction_dose = stoich_dose * excess_factor

        # Add residual requirement
        total_dose = reaction_dose + target_residual

        # Calculate actual ratio used
        actual_ratio = total_dose / o2_ppm if o2_ppm > 0 else stoich_ratio * excess_factor

        # Calculate feed rate (lb/hr)
        # Feed rate = Dose (ppm) * Flow (lb/hr) / 1,000,000
        feed_rate = total_dose * feedwater_flow / 1_000_000

        return total_dose, feed_rate, actual_ratio

    def _calculate_phosphate_dosing(
        self, input_data: ChemicalDosingInput
    ) -> Tuple[float, float]:
        """
        Calculate phosphate dosing requirement.

        Returns:
            Tuple of (dose_ppm, feed_rate_lb_hr)
        """
        current_phosphate = input_data.boiler_phosphate_ppm
        target_phosphate = input_data.target_phosphate_ppm
        blowdown_rate = input_data.blowdown_rate_pct
        feedwater_flow = input_data.feedwater_flow_lb_hr
        makeup_flow = input_data.makeup_water_flow_lb_hr

        # Get phosphate configuration
        phosphate_config = get_phosphate_config(input_data.treatment_program)

        if phosphate_config:
            target_phosphate = phosphate_config.phosphate_target_ppm

        # Calculate phosphate loss rate through blowdown
        # Loss rate (ppm in FW) = Target PO4 * Blowdown% / 100
        loss_rate = target_phosphate * (blowdown_rate / 100)

        # Calculate required feed to maintain target
        # Need to account for concentration factor (cycles)
        if blowdown_rate > 0:
            cycles = 100 / blowdown_rate
        else:
            cycles = 10  # Default

        # Dose in feedwater to achieve target in boiler
        required_dose = loss_rate

        # Calculate feed rate (lb/hr)
        feed_rate = required_dose * feedwater_flow / 1_000_000

        return required_dose, feed_rate

    def _calculate_amine_dosing(
        self, input_data: ChemicalDosingInput
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate amine dosing requirement for condensate protection.

        Returns:
            Tuple of (dose_ppm, feed_rate_lb_hr, change_ppm)
        """
        if input_data.condensate_ph is None or input_data.current_amine_type is None:
            return None, None, None

        amine_type = input_data.current_amine_type
        current_dose = input_data.current_amine_dose_ppm
        current_ph = input_data.condensate_ph
        target_ph = input_data.target_condensate_ph
        feedwater_flow = input_data.feedwater_flow_lb_hr

        # Get amine configuration
        amine_config = get_amine_config(amine_type)

        if amine_config:
            typical_dose = amine_config.typical_dose_ppm
            target_ph_range = amine_config.target_ph_range
        else:
            typical_dose = 5.0
            target_ph_range = (8.5, 9.0)

        # Estimate required dose adjustment based on pH deviation
        ph_deviation = target_ph - current_ph

        if abs(ph_deviation) < 0.1:
            # pH is acceptable, maintain current dose
            recommended_dose = current_dose
        elif ph_deviation > 0:
            # Need to raise pH - increase amine
            # Rough estimate: 1 ppm amine change = 0.1 pH units
            dose_increase = ph_deviation * 10 * typical_dose / (target_ph_range[1] - target_ph_range[0])
            recommended_dose = current_dose + dose_increase
        else:
            # Need to lower pH - decrease amine
            dose_decrease = abs(ph_deviation) * 10 * typical_dose / (target_ph_range[1] - target_ph_range[0])
            recommended_dose = max(0, current_dose - dose_decrease)

        # Limit to practical range
        recommended_dose = max(typical_dose * 0.5, min(recommended_dose, typical_dose * 3))

        dose_change = recommended_dose - current_dose

        # Calculate feed rate
        feed_rate = recommended_dose * feedwater_flow / 1_000_000

        return recommended_dose, feed_rate, dose_change

    def _calculate_costs(
        self,
        input_data: ChemicalDosingInput,
        scavenger_dose: float,
        phosphate_dose: float,
        amine_dose: Optional[float],
    ) -> Tuple[float, float]:
        """
        Calculate current and optimized daily chemical costs.

        Returns:
            Tuple of (current_cost_per_day, optimized_cost_per_day)
        """
        feedwater_flow = input_data.feedwater_flow_lb_hr
        hours_per_day = DosingConstants.HOURS_PER_DAY

        # Calculate daily water volume
        daily_water_lb = feedwater_flow * hours_per_day
        daily_water_gal = daily_water_lb / DosingConstants.LB_PER_GAL_WATER
        daily_water_kgal = daily_water_gal / 1000

        # Current costs
        current_scavenger_lb = (
            input_data.current_scavenger_dose_ppm *
            daily_water_lb / 1_000_000
        )
        current_scavenger_cost = current_scavenger_lb * input_data.scavenger_cost_per_lb

        current_phosphate_lb = (
            input_data.current_phosphate_dose_ppm *
            daily_water_lb / 1_000_000
        )
        current_phosphate_cost = current_phosphate_lb * input_data.phosphate_cost_per_lb

        current_amine_cost = 0
        if input_data.current_amine_dose_ppm > 0:
            current_amine_lb = (
                input_data.current_amine_dose_ppm *
                daily_water_lb / 1_000_000
            )
            current_amine_cost = current_amine_lb * input_data.amine_cost_per_lb

        current_total = current_scavenger_cost + current_phosphate_cost + current_amine_cost

        # Optimized costs
        optimized_scavenger_lb = scavenger_dose * daily_water_lb / 1_000_000
        optimized_scavenger_cost = optimized_scavenger_lb * input_data.scavenger_cost_per_lb

        optimized_phosphate_lb = phosphate_dose * daily_water_lb / 1_000_000
        optimized_phosphate_cost = optimized_phosphate_lb * input_data.phosphate_cost_per_lb

        optimized_amine_cost = 0
        if amine_dose:
            optimized_amine_lb = amine_dose * daily_water_lb / 1_000_000
            optimized_amine_cost = optimized_amine_lb * input_data.amine_cost_per_lb

        optimized_total = (
            optimized_scavenger_cost +
            optimized_phosphate_cost +
            optimized_amine_cost
        )

        return current_total, optimized_total

    def _check_dosing_ranges(
        self,
        input_data: ChemicalDosingInput,
        scavenger_dose: float,
        phosphate_dose: float,
        amine_dose: Optional[float],
    ) -> bool:
        """Check if dosing rates are within recommended ranges."""
        within_range = True

        # Check scavenger
        scavenger_config = get_scavenger_config(input_data.current_scavenger_type)
        if scavenger_config:
            if scavenger_dose < scavenger_config.residual_min_ppm:
                within_range = False
            if scavenger_dose > scavenger_config.residual_max_ppm * 2:
                within_range = False

        # Check phosphate
        phosphate_config = get_phosphate_config(input_data.treatment_program)
        if phosphate_config:
            if phosphate_dose > phosphate_config.phosphate_max_ppm * 2:
                within_range = False

        return within_range

    def _generate_recommendations(
        self,
        input_data: ChemicalDosingInput,
        scavenger_dose: float,
        scavenger_change: float,
        phosphate_dose: float,
        phosphate_change: float,
        amine_dose: Optional[float],
        amine_change: Optional[float],
    ) -> List[str]:
        """Generate dosing recommendations."""
        recommendations = []

        # Scavenger recommendations
        if abs(scavenger_change) > 5:
            if scavenger_change > 0:
                recommendations.append(
                    f"Increase oxygen scavenger dose by {scavenger_change:.1f} ppm "
                    f"to maintain adequate residual"
                )
            else:
                recommendations.append(
                    f"Reduce oxygen scavenger dose by {abs(scavenger_change):.1f} ppm "
                    f"to avoid over-treatment"
                )

        # Check if DO is too high
        if input_data.feedwater_do_ppb > 7:
            recommendations.append(
                f"Feedwater DO ({input_data.feedwater_do_ppb} ppb) exceeds limit - "
                f"verify deaerator operation before increasing scavenger"
            )

        # Phosphate recommendations
        if abs(phosphate_change) > 2:
            if phosphate_change > 0:
                recommendations.append(
                    f"Increase phosphate dose by {phosphate_change:.1f} ppm "
                    f"to maintain control range"
                )
            else:
                recommendations.append(
                    f"Reduce phosphate dose by {abs(phosphate_change):.1f} ppm"
                )

        # Amine recommendations
        if amine_dose and amine_change:
            if abs(amine_change) > 1:
                if amine_change > 0:
                    recommendations.append(
                        f"Increase amine dose by {amine_change:.1f} ppm "
                        f"to raise condensate pH"
                    )
                else:
                    recommendations.append(
                        f"Reduce amine dose by {abs(amine_change):.1f} ppm "
                        f"to lower condensate pH"
                    )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Current dosing rates are optimal - continue monitoring"
            )

        return recommendations

    def _calculate_provenance_hash(self, input_data: ChemicalDosingInput) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data_str = json.dumps(input_data.dict(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


def calculate_chemical_feed_rate(
    dose_ppm: float,
    flow_rate_lb_hr: float,
    product_strength_pct: float = 100.0,
) -> float:
    """
    Calculate chemical feed rate from dose and flow.

    Args:
        dose_ppm: Desired dose in ppm
        flow_rate_lb_hr: Water flow rate (lb/hr)
        product_strength_pct: Product concentration (%)

    Returns:
        Chemical feed rate (lb/hr)
    """
    # Feed rate = (Dose * Flow) / (1,000,000 * Strength)
    feed_rate = (dose_ppm * flow_rate_lb_hr) / (1_000_000 * product_strength_pct / 100)
    return feed_rate
