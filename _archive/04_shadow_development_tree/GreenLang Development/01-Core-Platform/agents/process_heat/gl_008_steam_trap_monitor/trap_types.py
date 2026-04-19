# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Trap Type Classification Module

This module provides steam trap type classification, selection criteria,
and application guidance based on DOE Best Practices and Spirax Sarco
engineering guidelines.

Features:
    - Trap type characteristics and operating principles
    - Selection criteria based on application requirements
    - Optimal application guide for each trap type
    - ASME B16.34 pressure rating validation

Standards:
    - DOE Steam System Best Practices
    - Spirax Sarco Application Guide
    - ASME B16.34 Valve Ratings

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.trap_types import (
    ...     TrapTypeClassifier,
    ...     TrapSelectionCriteria,
    ... )
    >>> classifier = TrapTypeClassifier()
    >>> criteria = TrapSelectionCriteria(
    ...     application="heat_exchanger",
    ...     steam_pressure_psig=150,
    ...     condensate_load_lb_hr=500,
    ... )
    >>> recommendations = classifier.select_trap_type(criteria)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    TrapType,
    TrapApplication,
    TrapTypeConfig,
    TrapTypeDefaults,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TRAP CHARACTERISTICS
# =============================================================================

@dataclass
class TrapCharacteristics:
    """
    Detailed characteristics of a steam trap type.

    This class encapsulates the operating principles, performance
    characteristics, advantages, disadvantages, and typical applications
    for each trap type based on engineering references.

    Attributes:
        trap_type: Trap type identifier
        operating_principle: Description of how the trap operates
        discharge_pattern: Continuous, intermittent, or thermostatic
        subcooling_f: Degrees of subcooling at discharge
        air_venting: Air venting capability rating
        advantages: List of advantages
        disadvantages: List of disadvantages
        typical_applications: Recommended applications
    """

    trap_type: TrapType
    operating_principle: str
    discharge_pattern: str
    subcooling_f: float
    air_venting: str
    max_pressure_psig: float
    max_capacity_lb_hr: float
    typical_service_life_years: float
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    typical_applications: List[TrapApplication] = field(default_factory=list)
    contraindicated_applications: List[TrapApplication] = field(default_factory=list)


# Default trap characteristics based on Spirax Sarco and DOE guides
TRAP_CHARACTERISTICS: Dict[TrapType, TrapCharacteristics] = {
    TrapType.FLOAT_THERMOSTATIC: TrapCharacteristics(
        trap_type=TrapType.FLOAT_THERMOSTATIC,
        operating_principle=(
            "Float rises and falls with condensate level, modulating valve "
            "to discharge condensate continuously. Thermostatic air vent "
            "discharges air and non-condensable gases."
        ),
        discharge_pattern="continuous",
        subcooling_f=0.0,
        air_venting="excellent",
        max_pressure_psig=465,
        max_capacity_lb_hr=30000,
        typical_service_life_years=7,
        advantages=[
            "Continuous condensate discharge at steam temperature",
            "Excellent air venting capability",
            "High capacity range",
            "Responds immediately to load changes",
            "Minimal steam loss on failure",
        ],
        disadvantages=[
            "Susceptible to waterhammer damage",
            "Not suitable for superheated steam",
            "Relatively complex internal mechanism",
            "Higher initial cost",
            "Can be damaged by dirt/debris",
        ],
        typical_applications=[
            TrapApplication.PROCESS,
            TrapApplication.HEAT_EXCHANGER,
            TrapApplication.COIL,
            TrapApplication.JACKETED_VESSEL,
        ],
        contraindicated_applications=[
            TrapApplication.DRIP_LEG,  # Waterhammer risk
            TrapApplication.TRACER,     # Overkill for small loads
        ],
    ),

    TrapType.INVERTED_BUCKET: TrapCharacteristics(
        trap_type=TrapType.INVERTED_BUCKET,
        operating_principle=(
            "Inverted bucket floats in condensate; steam entering bucket "
            "causes it to rise and close valve. As steam condenses, bucket "
            "sinks and valve opens to discharge condensate intermittently."
        ),
        discharge_pattern="intermittent",
        subcooling_f=3.0,
        air_venting="good",
        max_pressure_psig=600,
        max_capacity_lb_hr=50000,
        typical_service_life_years=15,
        advantages=[
            "Very robust and durable (longest service life)",
            "Resistant to waterhammer",
            "Tolerant to dirt and debris",
            "Wide pressure and capacity range",
            "Simple mechanism, easy to maintain",
            "Suitable for superheated steam",
        ],
        disadvantages=[
            "Intermittent discharge causes slight condensate backup",
            "Can lose prime (water seal)",
            "Air venting limited by small vent hole",
            "Larger physical size",
            "Slight subcooling of condensate",
        ],
        typical_applications=[
            TrapApplication.DRIP_LEG,
            TrapApplication.PROCESS,
            TrapApplication.REBOILER,
            TrapApplication.AUTOCLAVE,
        ],
        contraindicated_applications=[],
    ),

    TrapType.THERMOSTATIC: TrapCharacteristics(
        trap_type=TrapType.THERMOSTATIC,
        operating_principle=(
            "Bellows or bimetallic element contracts when cooled by condensate, "
            "opening valve. Element expands when heated by steam, closing valve. "
            "Operates based on temperature difference from saturation."
        ),
        discharge_pattern="thermostatic",
        subcooling_f=20.0,
        air_venting="excellent",
        max_pressure_psig=465,
        max_capacity_lb_hr=5000,
        typical_service_life_years=5,
        advantages=[
            "Excellent air venting",
            "Compact size",
            "Self-draining",
            "Passes large amounts of air/CO2 on startup",
            "Lower cost",
            "Simple design",
        ],
        disadvantages=[
            "Requires significant subcooling (backup of condensate)",
            "Bellows can be damaged by waterhammer",
            "Not suitable for modulating applications",
            "Slow response to load changes",
            "Limited capacity",
        ],
        typical_applications=[
            TrapApplication.TRACER,
            TrapApplication.UNIT_HEATER,
        ],
        contraindicated_applications=[
            TrapApplication.PROCESS,       # Subcooling unacceptable
            TrapApplication.HEAT_EXCHANGER,  # Need immediate discharge
        ],
    ),

    TrapType.THERMODYNAMIC: TrapCharacteristics(
        trap_type=TrapType.THERMODYNAMIC,
        operating_principle=(
            "Disc trap operates on velocity difference between steam and "
            "condensate. High-velocity flash steam holds disc closed; as "
            "steam condenses, pressure drops and disc opens."
        ),
        discharge_pattern="intermittent",
        subcooling_f=5.0,
        air_venting="fair",
        max_pressure_psig=600,
        max_capacity_lb_hr=2500,
        typical_service_life_years=5,
        advantages=[
            "Very compact and lightweight",
            "Resistant to waterhammer",
            "Resistant to freezing",
            "Wide pressure range",
            "Low cost",
            "Easy to install and maintain",
            "Works in any orientation",
        ],
        disadvantages=[
            "Noisy operation (cycling)",
            "Poor air venting",
            "Wear on disc and seat",
            "Requires minimum pressure differential",
            "Higher steam loss than mechanical traps",
            "Can be affected by back pressure",
        ],
        typical_applications=[
            TrapApplication.DRIP_LEG,
            TrapApplication.TRACER,
        ],
        contraindicated_applications=[
            TrapApplication.PROCESS,  # Steam loss concerns
            TrapApplication.HEAT_EXCHANGER,  # Poor modulation
        ],
    ),

    TrapType.BIMETALLIC: TrapCharacteristics(
        trap_type=TrapType.BIMETALLIC,
        operating_principle=(
            "Bimetallic strips or discs deflect when heated, closing valve. "
            "When cooled by subcooled condensate, they return to open position. "
            "Provides high subcooling but very robust operation."
        ),
        discharge_pattern="thermostatic",
        subcooling_f=40.0,
        air_venting="good",
        max_pressure_psig=600,
        max_capacity_lb_hr=10000,
        typical_service_life_years=10,
        advantages=[
            "Very robust construction",
            "Resistant to waterhammer",
            "Resistant to freezing (drains on shutdown)",
            "Wide operating range",
            "Good air venting",
            "Adjustable subcooling",
        ],
        disadvantages=[
            "High subcooling (significant condensate backup)",
            "Slow response to load changes",
            "Can stick in open or closed position",
            "Condensate temperature lower than steam temperature",
        ],
        typical_applications=[
            TrapApplication.TRACER,
            TrapApplication.DRIP_LEG,
        ],
        contraindicated_applications=[
            TrapApplication.PROCESS,
            TrapApplication.HEAT_EXCHANGER,
        ],
    ),
}


# =============================================================================
# SELECTION CRITERIA
# =============================================================================

class TrapSelectionCriteria(BaseModel):
    """Criteria for trap type selection."""

    application: TrapApplication = Field(
        ...,
        description="Primary application type"
    )
    steam_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Operating steam pressure"
    )
    back_pressure_psig: float = Field(
        default=0.0,
        ge=0,
        description="Back pressure at discharge"
    )
    condensate_load_lb_hr: float = Field(
        ...,
        gt=0,
        description="Expected condensate load"
    )

    # Operating requirements
    require_immediate_discharge: bool = Field(
        default=False,
        description="Must discharge at steam temperature"
    )
    require_air_venting: bool = Field(
        default=True,
        description="Require good air venting"
    )
    waterhammer_risk: bool = Field(
        default=False,
        description="Waterhammer present in system"
    )
    superheated_steam: bool = Field(
        default=False,
        description="Superheated steam application"
    )
    modulating_load: bool = Field(
        default=False,
        description="Load varies significantly"
    )

    # Environment
    outdoor_installation: bool = Field(
        default=False,
        description="Outdoor installation (freezing risk)"
    )
    dirt_in_system: bool = Field(
        default=False,
        description="Dirty steam or condensate"
    )

    # Preferences
    max_subcooling_acceptable_f: float = Field(
        default=50.0,
        ge=0,
        description="Maximum acceptable subcooling"
    )
    prefer_low_maintenance: bool = Field(
        default=True,
        description="Prefer low-maintenance options"
    )
    budget_constraint: str = Field(
        default="standard",
        description="Budget: economy, standard, premium"
    )

    class Config:
        use_enum_values = True


class TrapRecommendation(BaseModel):
    """Trap type recommendation with scoring."""

    trap_type: TrapType = Field(..., description="Recommended trap type")
    suitability_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Suitability score (0-100)"
    )
    ranking: int = Field(..., ge=1, description="Recommendation ranking")
    reasons_for: List[str] = Field(
        default_factory=list,
        description="Reasons supporting selection"
    )
    reasons_against: List[str] = Field(
        default_factory=list,
        description="Potential concerns"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Additional notes"
    )
    sizing_recommendation: Optional[str] = Field(
        default=None,
        description="Sizing guidance"
    )

    class Config:
        use_enum_values = True


class TrapApplicationGuide(BaseModel):
    """Application guide for trap selection."""

    application: TrapApplication = Field(..., description="Application type")
    description: str = Field(..., description="Application description")
    primary_requirements: List[str] = Field(
        default_factory=list,
        description="Primary requirements"
    )
    recommended_trap_types: List[TrapType] = Field(
        default_factory=list,
        description="Recommended trap types in order"
    )
    avoid_trap_types: List[TrapType] = Field(
        default_factory=list,
        description="Trap types to avoid"
    )
    typical_sizing_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Typical safety factor"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Application notes"
    )

    class Config:
        use_enum_values = True


# Application guides based on DOE and Spirax Sarco
APPLICATION_GUIDES: Dict[TrapApplication, TrapApplicationGuide] = {
    TrapApplication.DRIP_LEG: TrapApplicationGuide(
        application=TrapApplication.DRIP_LEG,
        description="Steam main drip legs - removing condensate from distribution",
        primary_requirements=[
            "Resistance to waterhammer",
            "Ability to handle flash steam",
            "Durability in cycling service",
        ],
        recommended_trap_types=[
            TrapType.INVERTED_BUCKET,
            TrapType.THERMODYNAMIC,
            TrapType.BIMETALLIC,
        ],
        avoid_trap_types=[
            TrapType.FLOAT_THERMOSTATIC,  # Waterhammer damage
        ],
        typical_sizing_factor=3.0,  # Higher for startup loads
        notes=[
            "Install at every 100-150 ft and low points",
            "Pocket should be at least 2x pipe diameter",
            "Consider strainer upstream",
        ],
    ),

    TrapApplication.HEAT_EXCHANGER: TrapApplicationGuide(
        application=TrapApplication.HEAT_EXCHANGER,
        description="Heat exchangers requiring maximum heat transfer",
        primary_requirements=[
            "Immediate discharge at steam temperature",
            "Modulating capacity for varying loads",
            "Good air venting for startup",
        ],
        recommended_trap_types=[
            TrapType.FLOAT_THERMOSTATIC,
            TrapType.INVERTED_BUCKET,
        ],
        avoid_trap_types=[
            TrapType.THERMOSTATIC,   # Subcooling reduces heat transfer
            TrapType.BIMETALLIC,      # Subcooling reduces heat transfer
            TrapType.THERMODYNAMIC,   # Poor modulation
        ],
        typical_sizing_factor=2.0,
        notes=[
            "Size for maximum load plus safety factor",
            "Consider stall conditions",
            "Install vacuum breaker if needed",
        ],
    ),

    TrapApplication.PROCESS: TrapApplicationGuide(
        application=TrapApplication.PROCESS,
        description="Process equipment requiring precise temperature control",
        primary_requirements=[
            "Immediate discharge",
            "High capacity",
            "Reliable operation",
        ],
        recommended_trap_types=[
            TrapType.FLOAT_THERMOSTATIC,
            TrapType.INVERTED_BUCKET,
        ],
        avoid_trap_types=[
            TrapType.THERMOSTATIC,
            TrapType.BIMETALLIC,
        ],
        typical_sizing_factor=2.0,
        notes=[
            "Consider redundant traps for critical processes",
            "Install isolation valves for maintenance",
        ],
    ),

    TrapApplication.TRACER: TrapApplicationGuide(
        application=TrapApplication.TRACER,
        description="Steam tracing for freeze protection and viscosity control",
        primary_requirements=[
            "Compact size",
            "Low cost",
            "Freeze resistance",
        ],
        recommended_trap_types=[
            TrapType.THERMOSTATIC,
            TrapType.THERMODYNAMIC,
            TrapType.BIMETALLIC,
        ],
        avoid_trap_types=[
            TrapType.FLOAT_THERMOSTATIC,  # Overkill
        ],
        typical_sizing_factor=2.0,
        notes=[
            "Subcooling acceptable for tracing",
            "Consider manifold installations",
            "Ensure adequate pitch for drainage",
        ],
    ),

    TrapApplication.UNIT_HEATER: TrapApplicationGuide(
        application=TrapApplication.UNIT_HEATER,
        description="Unit heaters and fan coils",
        primary_requirements=[
            "Good air venting",
            "Compact size",
            "Low cost",
        ],
        recommended_trap_types=[
            TrapType.THERMOSTATIC,
            TrapType.FLOAT_THERMOSTATIC,
        ],
        avoid_trap_types=[
            TrapType.THERMODYNAMIC,  # Poor air venting
        ],
        typical_sizing_factor=2.0,
        notes=[
            "Air binding common issue",
            "Consider vacuum breakers",
        ],
    ),

    TrapApplication.REBOILER: TrapApplicationGuide(
        application=TrapApplication.REBOILER,
        description="Distillation column reboilers",
        primary_requirements=[
            "High capacity",
            "Reliable operation",
            "Handle varying loads",
        ],
        recommended_trap_types=[
            TrapType.INVERTED_BUCKET,
            TrapType.FLOAT_THERMOSTATIC,
        ],
        avoid_trap_types=[
            TrapType.THERMOSTATIC,
            TrapType.THERMODYNAMIC,
        ],
        typical_sizing_factor=2.0,
        notes=[
            "Often require multiple traps",
            "Consider stall conditions",
        ],
    ),

    TrapApplication.AUTOCLAVE: TrapApplicationGuide(
        application=TrapApplication.AUTOCLAVE,
        description="Autoclaves and sterilizers",
        primary_requirements=[
            "Excellent air removal",
            "Handle temperature cycling",
            "Reliable operation",
        ],
        recommended_trap_types=[
            TrapType.INVERTED_BUCKET,
            TrapType.FLOAT_THERMOSTATIC,
        ],
        avoid_trap_types=[
            TrapType.THERMODYNAMIC,
        ],
        typical_sizing_factor=3.0,  # High startup loads
        notes=[
            "Air removal critical for sterilization",
            "Pressure/temperature cycling stresses traps",
        ],
    ),
}


# =============================================================================
# TRAP TYPE CLASSIFIER
# =============================================================================

class TrapTypeClassifier:
    """
    Steam trap type classifier and selection advisor.

    This class provides trap type classification, selection recommendations,
    and application guidance based on DOE Best Practices and Spirax Sarco
    engineering guidelines.

    All recommendations are deterministic (zero-hallucination) based on
    engineering lookup tables and decision trees.

    Example:
        >>> classifier = TrapTypeClassifier()
        >>> criteria = TrapSelectionCriteria(
        ...     application=TrapApplication.HEAT_EXCHANGER,
        ...     steam_pressure_psig=150,
        ...     condensate_load_lb_hr=500,
        ... )
        >>> recommendations = classifier.select_trap_type(criteria)
        >>> print(recommendations[0].trap_type)
    """

    def __init__(self) -> None:
        """Initialize the trap type classifier."""
        self._characteristics = TRAP_CHARACTERISTICS
        self._application_guides = APPLICATION_GUIDES
        self._selection_count = 0

        logger.info("TrapTypeClassifier initialized")

    def get_trap_characteristics(
        self,
        trap_type: TrapType,
    ) -> Optional[TrapCharacteristics]:
        """
        Get detailed characteristics for a trap type.

        Args:
            trap_type: Trap type to look up

        Returns:
            TrapCharacteristics or None if not found
        """
        return self._characteristics.get(trap_type)

    def get_application_guide(
        self,
        application: TrapApplication,
    ) -> Optional[TrapApplicationGuide]:
        """
        Get application guide for a specific application.

        Args:
            application: Application type

        Returns:
            TrapApplicationGuide or None if not found
        """
        return self._application_guides.get(application)

    def select_trap_type(
        self,
        criteria: TrapSelectionCriteria,
    ) -> List[TrapRecommendation]:
        """
        Select appropriate trap types based on criteria.

        Uses deterministic scoring based on application requirements,
        operating conditions, and trap characteristics.

        Args:
            criteria: Selection criteria

        Returns:
            List of TrapRecommendation, sorted by suitability score

        Example:
            >>> criteria = TrapSelectionCriteria(
            ...     application=TrapApplication.PROCESS,
            ...     steam_pressure_psig=150,
            ...     condensate_load_lb_hr=500,
            ... )
            >>> recs = classifier.select_trap_type(criteria)
        """
        self._selection_count += 1
        recommendations = []

        # Get application guide
        guide = self._application_guides.get(criteria.application)

        for trap_type, characteristics in self._characteristics.items():
            score = 100.0
            reasons_for = []
            reasons_against = []
            notes = []

            # Check pressure rating
            if criteria.steam_pressure_psig > characteristics.max_pressure_psig:
                score = 0.0
                reasons_against.append(
                    f"Pressure {criteria.steam_pressure_psig} psig exceeds "
                    f"max rating {characteristics.max_pressure_psig} psig"
                )
                continue

            # Check capacity
            if criteria.condensate_load_lb_hr > characteristics.max_capacity_lb_hr:
                score = 0.0
                reasons_against.append(
                    f"Load {criteria.condensate_load_lb_hr} lb/hr exceeds "
                    f"max capacity {characteristics.max_capacity_lb_hr} lb/hr"
                )
                continue

            # Application suitability from guide
            if guide:
                if trap_type in guide.recommended_trap_types:
                    rank = guide.recommended_trap_types.index(trap_type)
                    bonus = 30 - (rank * 10)  # First choice +30, second +20, etc.
                    score += bonus
                    reasons_for.append(
                        f"Recommended for {criteria.application} applications"
                    )

                if trap_type in guide.avoid_trap_types:
                    score -= 50
                    reasons_against.append(
                        f"Not recommended for {criteria.application} applications"
                    )

            # Subcooling check
            if criteria.require_immediate_discharge:
                if characteristics.subcooling_f > 5:
                    score -= 40
                    reasons_against.append(
                        f"Subcooling of {characteristics.subcooling_f}F "
                        "conflicts with immediate discharge requirement"
                    )
                else:
                    score += 15
                    reasons_for.append("Provides immediate discharge")

            if characteristics.subcooling_f > criteria.max_subcooling_acceptable_f:
                score -= 30
                reasons_against.append(
                    f"Subcooling {characteristics.subcooling_f}F exceeds "
                    f"maximum acceptable {criteria.max_subcooling_acceptable_f}F"
                )

            # Air venting
            if criteria.require_air_venting:
                if characteristics.air_venting == "excellent":
                    score += 15
                    reasons_for.append("Excellent air venting capability")
                elif characteristics.air_venting == "good":
                    score += 10
                    reasons_for.append("Good air venting capability")
                elif characteristics.air_venting == "fair":
                    score -= 10
                    reasons_against.append("Limited air venting capability")
                elif characteristics.air_venting == "poor":
                    score -= 25
                    reasons_against.append("Poor air venting capability")

            # Waterhammer
            if criteria.waterhammer_risk:
                # Check advantages/disadvantages for waterhammer info
                if any("waterhammer" in adv.lower() and "resist" in adv.lower()
                       for adv in characteristics.advantages):
                    score += 20
                    reasons_for.append("Resistant to waterhammer")
                if any("waterhammer" in dis.lower() for dis in characteristics.disadvantages):
                    score -= 30
                    reasons_against.append("Susceptible to waterhammer damage")

            # Superheated steam
            if criteria.superheated_steam:
                if trap_type == TrapType.INVERTED_BUCKET:
                    score += 15
                    reasons_for.append("Suitable for superheated steam")
                elif trap_type == TrapType.FLOAT_THERMOSTATIC:
                    score -= 20
                    reasons_against.append("Not suitable for superheated steam")

            # Modulating load
            if criteria.modulating_load:
                if characteristics.discharge_pattern == "continuous":
                    score += 15
                    reasons_for.append("Continuous modulating discharge")
                elif characteristics.discharge_pattern == "intermittent":
                    score -= 5
                    notes.append("Intermittent discharge may cause slight backup")

            # Outdoor/freezing
            if criteria.outdoor_installation:
                if any("freez" in adv.lower() for adv in characteristics.advantages):
                    score += 15
                    reasons_for.append("Resistant to freezing")
                if trap_type == TrapType.FLOAT_THERMOSTATIC:
                    score -= 10
                    reasons_against.append("Water in body can freeze")

            # Dirt tolerance
            if criteria.dirt_in_system:
                if any("dirt" in adv.lower() for adv in characteristics.advantages):
                    score += 15
                    reasons_for.append("Tolerant to dirt and debris")
                if any("dirt" in dis.lower() or "debris" in dis.lower()
                       for dis in characteristics.disadvantages):
                    score -= 15
                    reasons_against.append("Sensitive to dirt/debris")

            # Maintenance preference
            if criteria.prefer_low_maintenance:
                score += characteristics.typical_service_life_years * 2
                if characteristics.typical_service_life_years >= 10:
                    reasons_for.append(
                        f"Long service life ({characteristics.typical_service_life_years} years)"
                    )

            # Clamp score to valid range
            score = max(0, min(100, score))

            # Create recommendation
            recommendation = TrapRecommendation(
                trap_type=trap_type,
                suitability_score=round(score, 1),
                ranking=0,  # Set after sorting
                reasons_for=reasons_for,
                reasons_against=reasons_against,
                notes=notes,
                sizing_recommendation=self._get_sizing_guidance(
                    trap_type, criteria
                ),
            )
            recommendations.append(recommendation)

        # Sort by score and assign rankings
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        for i, rec in enumerate(recommendations):
            rec.ranking = i + 1

        logger.debug(
            f"Trap selection completed. Top recommendation: "
            f"{recommendations[0].trap_type if recommendations else 'None'}"
        )

        return recommendations

    def _get_sizing_guidance(
        self,
        trap_type: TrapType,
        criteria: TrapSelectionCriteria,
    ) -> str:
        """Generate sizing guidance for the trap type."""
        guide = self._application_guides.get(criteria.application)
        safety_factor = guide.typical_sizing_factor if guide else 2.0

        design_capacity = criteria.condensate_load_lb_hr * safety_factor

        characteristics = self._characteristics.get(trap_type)
        if characteristics:
            if design_capacity > characteristics.max_capacity_lb_hr * 0.8:
                return (
                    f"Design capacity {design_capacity:.0f} lb/hr "
                    f"(SF={safety_factor}). Consider parallel traps."
                )

        return (
            f"Design capacity {design_capacity:.0f} lb/hr "
            f"(load {criteria.condensate_load_lb_hr:.0f} x SF={safety_factor})"
        )

    def validate_trap_for_application(
        self,
        trap_type: TrapType,
        application: TrapApplication,
        pressure_psig: float,
        load_lb_hr: float,
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a trap type is suitable for an application.

        Args:
            trap_type: Trap type to validate
            application: Target application
            pressure_psig: Operating pressure
            load_lb_hr: Condensate load

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        characteristics = self._characteristics.get(trap_type)

        if not characteristics:
            return False, [f"Unknown trap type: {trap_type}"]

        # Pressure check
        if pressure_psig > characteristics.max_pressure_psig:
            issues.append(
                f"Pressure {pressure_psig} psig exceeds "
                f"rating {characteristics.max_pressure_psig} psig"
            )

        # Capacity check
        if load_lb_hr > characteristics.max_capacity_lb_hr:
            issues.append(
                f"Load {load_lb_hr} lb/hr exceeds "
                f"capacity {characteristics.max_capacity_lb_hr} lb/hr"
            )

        # Application check
        if application in characteristics.contraindicated_applications:
            issues.append(
                f"Trap type not recommended for {application} applications"
            )

        return len(issues) == 0, issues

    def check_asme_b16_34_compliance(
        self,
        pressure_psig: float,
        temperature_f: float,
        trap_rating_class: int,
    ) -> Tuple[bool, str]:
        """
        Check ASME B16.34 pressure-temperature rating compliance.

        Args:
            pressure_psig: Operating pressure
            temperature_f: Operating temperature
            trap_rating_class: Trap class rating (150, 300, 600, etc.)

        Returns:
            Tuple of (is_compliant, message)
        """
        # ASME B16.34 Class pressure ratings at temperature
        # Simplified table for carbon steel (adjust for material)
        class_ratings = {
            150: {
                100: 285, 200: 260, 300: 230, 400: 200,
                500: 170, 600: 140, 650: 125, 700: 110,
            },
            300: {
                100: 740, 200: 675, 300: 600, 400: 530,
                500: 455, 600: 375, 650: 340, 700: 300,
            },
            600: {
                100: 1480, 200: 1350, 300: 1195, 400: 1055,
                500: 905, 600: 750, 650: 675, 700: 600,
            },
        }

        if trap_rating_class not in class_ratings:
            return False, f"Unknown class rating: {trap_rating_class}"

        ratings = class_ratings[trap_rating_class]

        # Find applicable temperature rating
        applicable_pressure = None
        for temp in sorted(ratings.keys()):
            if temperature_f <= temp:
                applicable_pressure = ratings[temp]
                break
        else:
            # Use highest temperature rating
            applicable_pressure = ratings[max(ratings.keys())]

        if pressure_psig <= applicable_pressure:
            return True, (
                f"Class {trap_rating_class} rated for {applicable_pressure} psig "
                f"at {temperature_f}F - COMPLIANT"
            )
        else:
            return False, (
                f"Class {trap_rating_class} rated for only {applicable_pressure} psig "
                f"at {temperature_f}F - EXCEEDS RATING"
            )

    @property
    def selection_count(self) -> int:
        """Get total selection count."""
        return self._selection_count
