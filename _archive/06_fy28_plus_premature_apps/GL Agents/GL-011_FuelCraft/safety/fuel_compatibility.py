# -*- coding: utf-8 -*-
"""
FuelCompatibilityMatrix - Fuel co-mingling and segregation rules for GL-011 FuelCraft.

This module implements the FuelCompatibilityMatrix for GL-011 FuelCraft, providing
fuel compatibility verification, co-mingling rules, segregation requirements, and
contamination risk scoring for safe fuel blending operations.

Reference Standards:
    - ASTM D4814: Standard Specification for Automotive Spark-Ignition Engine Fuel
    - ASTM D975: Standard Specification for Diesel Fuel
    - ASTM D396: Standard Specification for Fuel Oils
    - API 2610: Design, Construction, Operation, Maintenance, and Inspection of
                Terminal and Tank Facilities
    - NFPA 30: Flammable and Combustible Liquids Code

Safety Philosophy:
    - Fail-closed: Unknown fuel combinations are blocked by default
    - Zero-hallucination: All compatibility rules from deterministic lookup
    - Provenance tracking: SHA-256 hashes for all compatibility decisions

Example:
    >>> matrix = FuelCompatibilityMatrix()
    >>> result = matrix.check_compatibility("HFO", "LSFO")
    >>> if result.can_co_mingle:
    ...     proceed_with_blending()
    >>> else:
    ...     raise SegregationRequired(result.reason)

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class FuelCategory(str, Enum):
    """Fuel category classification per ASTM standards."""
    DISTILLATE = "distillate"
    RESIDUAL = "residual"
    GASOLINE = "gasoline"
    BIOFUEL = "biofuel"
    LNG = "lng"
    LPG = "lpg"
    CRUDE = "crude"
    SPECIALTY = "specialty"


class CompatibilityLevel(str, Enum):
    """Compatibility level for fuel combinations."""
    FULLY_COMPATIBLE = "fully_compatible"
    CONDITIONALLY_COMPATIBLE = "conditional"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class ContaminationRisk(str, Enum):
    """Contamination risk level."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SegregationType(str, Enum):
    """Type of segregation required."""
    NONE = "none"
    DEDICATED_TANK = "dedicated"
    PHYSICAL_BARRIER = "barrier"
    TIME_SEPARATION = "time"
    FLUSH_REQUIRED = "flush"


class FuelType(BaseModel):
    """Definition of a fuel type with properties."""
    fuel_id: str = Field(..., description="Unique fuel identifier")
    fuel_name: str = Field(..., description="Human-readable fuel name")
    category: FuelCategory = Field(..., description="Fuel category")
    flash_point_min_c: Optional[float] = Field(None, ge=-50, le=300)
    flash_point_max_c: Optional[float] = Field(None, ge=-50, le=300)
    density_min_kg_m3: Optional[float] = Field(None, ge=500, le=1100)
    density_max_kg_m3: Optional[float] = Field(None, ge=500, le=1100)
    viscosity_min_cst: Optional[float] = Field(None, ge=0.1, le=700)
    viscosity_max_cst: Optional[float] = Field(None, ge=0.1, le=700)
    sulfur_max_pct: Optional[float] = Field(None, ge=0, le=5)
    water_sensitive: bool = Field(False)
    oxidation_sensitive: bool = Field(False)
    wax_content: bool = Field(False)
    asphaltene_content: bool = Field(False)
    astm_standard: Optional[str] = Field(None)
    iso_standard: Optional[str] = Field(None)

    class Config:
        frozen = True


class CompatibilityRule(BaseModel):
    """Rule defining compatibility between two fuel types."""
    rule_id: str = Field(..., description="Unique rule identifier")
    fuel_a: str = Field(..., description="First fuel type ID")
    fuel_b: str = Field(..., description="Second fuel type ID")
    compatibility: CompatibilityLevel = Field(..., description="Compatibility level")
    conditions: List[str] = Field(default_factory=list)
    max_blend_ratio: Optional[float] = Field(None, ge=0, le=1)
    contamination_risk: ContaminationRisk = Field(ContaminationRisk.NONE)
    segregation_type: SegregationType = Field(SegregationType.NONE)
    technical_reason: str = Field("")
    reference_standard: Optional[str] = Field(None)

    class Config:
        frozen = True


class CompatibilityCheckResult(BaseModel):
    """Result of a fuel compatibility check."""
    check_id: str = Field(..., description="Unique check identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fuel_a: str = Field(..., description="First fuel type")
    fuel_b: str = Field(..., description="Second fuel type")
    can_co_mingle: bool = Field(..., description="Whether co-mingling is allowed")
    compatibility_level: CompatibilityLevel = Field(...)
    contamination_risk: ContaminationRisk = Field(...)
    contamination_score: float = Field(..., ge=0, le=100)
    conditions: List[str] = Field(default_factory=list)
    segregation_required: SegregationType = Field(...)
    max_blend_ratio: Optional[float] = Field(None)
    reason: str = Field(...)
    technical_details: str = Field("")
    rule_id: Optional[str] = Field(None)
    provenance_hash: str = Field(...)


DEFAULT_FUEL_TYPES: Dict[str, FuelType] = {
    "HFO": FuelType(
        fuel_id="HFO", fuel_name="Heavy Fuel Oil", category=FuelCategory.RESIDUAL,
        flash_point_min_c=60, flash_point_max_c=100, density_min_kg_m3=920,
        density_max_kg_m3=1010, viscosity_min_cst=50, viscosity_max_cst=700,
        sulfur_max_pct=3.5, water_sensitive=True, asphaltene_content=True,
        astm_standard="ASTM D396 Grade 6"
    ),
    "VLSFO": FuelType(
        fuel_id="VLSFO", fuel_name="Very Low Sulfur Fuel Oil", category=FuelCategory.RESIDUAL,
        flash_point_min_c=60, flash_point_max_c=100, density_min_kg_m3=880,
        density_max_kg_m3=1000, viscosity_min_cst=20, viscosity_max_cst=500,
        sulfur_max_pct=0.5, water_sensitive=True, asphaltene_content=True,
        astm_standard="ISO 8217:2017"
    ),
    "MGO": FuelType(
        fuel_id="MGO", fuel_name="Marine Gas Oil", category=FuelCategory.DISTILLATE,
        flash_point_min_c=60, flash_point_max_c=100, density_min_kg_m3=820,
        density_max_kg_m3=890, viscosity_min_cst=2, viscosity_max_cst=6,
        sulfur_max_pct=1.5, water_sensitive=True, wax_content=True,
        astm_standard="ISO 8217 DMA"
    ),
    "ULSD": FuelType(
        fuel_id="ULSD", fuel_name="Ultra Low Sulfur Diesel", category=FuelCategory.DISTILLATE,
        flash_point_min_c=52, flash_point_max_c=80, density_min_kg_m3=820,
        density_max_kg_m3=860, viscosity_min_cst=1.9, viscosity_max_cst=4.1,
        sulfur_max_pct=0.0015, water_sensitive=True, wax_content=True,
        astm_standard="ASTM D975 S15"
    ),
    "FAME": FuelType(
        fuel_id="FAME", fuel_name="Fatty Acid Methyl Ester", category=FuelCategory.BIOFUEL,
        flash_point_min_c=100, flash_point_max_c=170, density_min_kg_m3=860,
        density_max_kg_m3=900, viscosity_min_cst=3.5, viscosity_max_cst=5.0,
        sulfur_max_pct=0.001, water_sensitive=True, oxidation_sensitive=True,
        astm_standard="ASTM D6751"
    ),
    "MOGAS": FuelType(
        fuel_id="MOGAS", fuel_name="Motor Gasoline", category=FuelCategory.GASOLINE,
        flash_point_min_c=-43, flash_point_max_c=-38, density_min_kg_m3=720,
        density_max_kg_m3=775, viscosity_min_cst=0.4, viscosity_max_cst=0.8,
        sulfur_max_pct=0.01, water_sensitive=True, astm_standard="ASTM D4814"
    ),
    "JET_A": FuelType(
        fuel_id="JET_A", fuel_name="Jet A Aviation Fuel", category=FuelCategory.DISTILLATE,
        flash_point_min_c=38, flash_point_max_c=55, density_min_kg_m3=775,
        density_max_kg_m3=840, viscosity_min_cst=1, viscosity_max_cst=8,
        sulfur_max_pct=0.3, water_sensitive=True, oxidation_sensitive=True,
        astm_standard="ASTM D1655"
    ),
}

DEFAULT_COMPATIBILITY_RULES: List[CompatibilityRule] = [
    CompatibilityRule(
        rule_id="R001", fuel_a="HFO", fuel_b="VLSFO",
        compatibility=CompatibilityLevel.CONDITIONALLY_COMPATIBLE,
        conditions=["Perform asphaltene stability test", "Maintain min 40C", "Max 30% blend"],
        max_blend_ratio=0.3, contamination_risk=ContaminationRisk.MEDIUM,
        technical_reason="Paraffinic VLSFO may destabilize asphaltenes in HFO",
        reference_standard="ISO 8217:2017 Annex B"
    ),
    CompatibilityRule(
        rule_id="D001", fuel_a="MGO", fuel_b="ULSD",
        compatibility=CompatibilityLevel.FULLY_COMPATIBLE,
        contamination_risk=ContaminationRisk.NONE,
        technical_reason="Compatible distillate fuels", reference_standard="ASTM D975"
    ),
    CompatibilityRule(
        rule_id="B001", fuel_a="ULSD", fuel_b="FAME",
        compatibility=CompatibilityLevel.CONDITIONALLY_COMPATIBLE,
        conditions=["Limit FAME to 7% (B7)", "Check microbial growth", "Monitor oxidation"],
        max_blend_ratio=0.07, contamination_risk=ContaminationRisk.LOW,
        technical_reason="FAME blending per EN 590", reference_standard="EN 590, ASTM D6751"
    ),
    CompatibilityRule(
        rule_id="G001", fuel_a="MOGAS", fuel_b="ULSD",
        compatibility=CompatibilityLevel.INCOMPATIBLE,
        contamination_risk=ContaminationRisk.CRITICAL,
        segregation_type=SegregationType.DEDICATED_TANK,
        technical_reason="Gasoline in diesel lowers flash point - critical hazard",
        reference_standard="NFPA 30, API 2610"
    ),
    CompatibilityRule(
        rule_id="G002", fuel_a="MOGAS", fuel_b="JET_A",
        compatibility=CompatibilityLevel.INCOMPATIBLE,
        contamination_risk=ContaminationRisk.CRITICAL,
        segregation_type=SegregationType.DEDICATED_TANK,
        technical_reason="Gasoline in jet fuel is a critical safety hazard",
        reference_standard="ASTM D1655, ATA 103"
    ),
    CompatibilityRule(
        rule_id="J001", fuel_a="JET_A", fuel_b="ULSD",
        compatibility=CompatibilityLevel.INCOMPATIBLE,
        contamination_risk=ContaminationRisk.HIGH,
        segregation_type=SegregationType.DEDICATED_TANK,
        technical_reason="Aviation fuel requires dedicated storage",
        reference_standard="ASTM D1655"
    ),
]


class FuelCompatibilityMatrix:
    """
    FuelCompatibilityMatrix for safe fuel co-mingling decisions.

    Implements fail-closed behavior for unknown combinations and provides
    SHA-256 provenance tracking for all compatibility decisions.
    """

    def __init__(
        self,
        fuel_types: Optional[Dict[str, FuelType]] = None,
        compatibility_rules: Optional[List[CompatibilityRule]] = None
    ):
        """Initialize FuelCompatibilityMatrix."""
        self._fuel_types: Dict[str, FuelType] = fuel_types or dict(DEFAULT_FUEL_TYPES)
        self._rules: Dict[str, CompatibilityRule] = {}

        rules = compatibility_rules or DEFAULT_COMPATIBILITY_RULES
        for rule in rules:
            key_ab = self._make_key(rule.fuel_a, rule.fuel_b)
            key_ba = self._make_key(rule.fuel_b, rule.fuel_a)
            self._rules[key_ab] = rule
            self._rules[key_ba] = rule

        logger.info(
            f"FuelCompatibilityMatrix initialized with {len(self._fuel_types)} fuel types "
            f"and {len(rules)} compatibility rules"
        )

    def check_compatibility(
        self, fuel_a: str, fuel_b: str, proposed_ratio: Optional[float] = None
    ) -> CompatibilityCheckResult:
        """
        Check compatibility between two fuel types.

        FAIL-CLOSED: Unknown combinations return INCOMPATIBLE.
        """
        start_time = datetime.now(timezone.utc)

        if fuel_a not in self._fuel_types:
            raise ValueError(f"Unknown fuel type: {fuel_a}")
        if fuel_b not in self._fuel_types:
            raise ValueError(f"Unknown fuel type: {fuel_b}")

        if fuel_a == fuel_b:
            check_id = self._generate_check_id(fuel_a, fuel_b, start_time)
            return CompatibilityCheckResult(
                check_id=check_id, fuel_a=fuel_a, fuel_b=fuel_b,
                can_co_mingle=True, compatibility_level=CompatibilityLevel.FULLY_COMPATIBLE,
                contamination_risk=ContaminationRisk.NONE, contamination_score=0.0,
                segregation_required=SegregationType.NONE, reason="Same fuel type",
                provenance_hash=self._calculate_provenance_hash(check_id, fuel_a, fuel_b, "SAME")
            )

        key = self._make_key(fuel_a, fuel_b)
        rule = self._rules.get(key)

        if rule is None:
            check_id = self._generate_check_id(fuel_a, fuel_b, start_time)
            logger.warning(f"[SAFETY] Unknown combination {fuel_a}/{fuel_b} - BLOCKING")
            return CompatibilityCheckResult(
                check_id=check_id, fuel_a=fuel_a, fuel_b=fuel_b,
                can_co_mingle=False, compatibility_level=CompatibilityLevel.UNKNOWN,
                contamination_risk=ContaminationRisk.HIGH, contamination_score=80.0,
                segregation_required=SegregationType.DEDICATED_TANK,
                reason="Unknown combination - segregation required (fail-closed)",
                provenance_hash=self._calculate_provenance_hash(check_id, fuel_a, fuel_b, "UNKNOWN")
            )

        check_id = self._generate_check_id(fuel_a, fuel_b, start_time)
        contamination_score = self._calculate_contamination_score(rule.contamination_risk)

        ratio_exceeded = False
        if proposed_ratio is not None and rule.max_blend_ratio is not None:
            if proposed_ratio > rule.max_blend_ratio:
                ratio_exceeded = True

        if rule.compatibility == CompatibilityLevel.INCOMPATIBLE:
            can_co_mingle = False
            reason = f"Incompatible: {rule.technical_reason}"
        elif rule.compatibility == CompatibilityLevel.CONDITIONALLY_COMPATIBLE:
            if ratio_exceeded:
                can_co_mingle = False
                reason = f"Ratio {proposed_ratio:.1%} exceeds max {rule.max_blend_ratio:.1%}"
            else:
                can_co_mingle = True
                reason = f"Conditional: {rule.technical_reason}"
        else:
            can_co_mingle = True
            reason = f"Compatible: {rule.technical_reason}"

        return CompatibilityCheckResult(
            check_id=check_id, fuel_a=fuel_a, fuel_b=fuel_b,
            can_co_mingle=can_co_mingle, compatibility_level=rule.compatibility,
            contamination_risk=rule.contamination_risk, contamination_score=contamination_score,
            conditions=list(rule.conditions), segregation_required=rule.segregation_type,
            max_blend_ratio=rule.max_blend_ratio, reason=reason,
            technical_details=rule.technical_reason, rule_id=rule.rule_id,
            provenance_hash=self._calculate_provenance_hash(check_id, fuel_a, fuel_b, rule.rule_id)
        )

    def get_segregation_requirements(self, fuel_id: str) -> Dict[str, SegregationType]:
        """Get segregation requirements for a fuel against all others."""
        if fuel_id not in self._fuel_types:
            raise ValueError(f"Unknown fuel type: {fuel_id}")

        requirements: Dict[str, SegregationType] = {}
        for other_fuel in self._fuel_types.keys():
            if other_fuel == fuel_id:
                continue
            key = self._make_key(fuel_id, other_fuel)
            rule = self._rules.get(key)
            requirements[other_fuel] = (
                rule.segregation_type if rule else SegregationType.DEDICATED_TANK
            )
        return requirements

    def get_compatible_fuels(self, fuel_id: str, include_conditional: bool = True) -> List[str]:
        """Get list of fuels compatible with the specified fuel."""
        if fuel_id not in self._fuel_types:
            raise ValueError(f"Unknown fuel type: {fuel_id}")

        compatible: List[str] = [fuel_id]
        for other_fuel in self._fuel_types.keys():
            if other_fuel == fuel_id:
                continue
            key = self._make_key(fuel_id, other_fuel)
            rule = self._rules.get(key)
            if rule is None:
                continue
            if rule.compatibility == CompatibilityLevel.FULLY_COMPATIBLE:
                compatible.append(other_fuel)
            elif include_conditional and rule.compatibility == CompatibilityLevel.CONDITIONALLY_COMPATIBLE:
                compatible.append(other_fuel)
        return compatible

    def get_contamination_risk_score(self, fuel_a: str, fuel_b: str) -> float:
        """Get contamination risk score for a fuel combination."""
        if fuel_a == fuel_b:
            return 0.0
        key = self._make_key(fuel_a, fuel_b)
        rule = self._rules.get(key)
        if rule is None:
            return 80.0
        return self._calculate_contamination_score(rule.contamination_risk)

    def register_fuel_type(self, fuel: FuelType) -> None:
        """Register a new fuel type."""
        self._fuel_types[fuel.fuel_id] = fuel
        logger.info(f"Registered fuel type: {fuel.fuel_id}")

    def register_compatibility_rule(self, rule: CompatibilityRule) -> None:
        """Register a new compatibility rule."""
        key_ab = self._make_key(rule.fuel_a, rule.fuel_b)
        key_ba = self._make_key(rule.fuel_b, rule.fuel_a)
        self._rules[key_ab] = rule
        self._rules[key_ba] = rule
        logger.info(f"Registered rule {rule.rule_id}: {rule.fuel_a}/{rule.fuel_b}")

    def get_fuel_type(self, fuel_id: str) -> Optional[FuelType]:
        """Get a fuel type by ID."""
        return self._fuel_types.get(fuel_id)

    def list_fuel_types(self) -> List[str]:
        """List all registered fuel type IDs."""
        return list(self._fuel_types.keys())

    def _make_key(self, fuel_a: str, fuel_b: str) -> str:
        """Create key for fuel pair."""
        return f"{fuel_a}|{fuel_b}"

    def _generate_check_id(self, fuel_a: str, fuel_b: str, timestamp: datetime) -> str:
        """Generate unique check ID."""
        data = f"{fuel_a}|{fuel_b}|{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _calculate_contamination_score(self, risk: ContaminationRisk) -> float:
        """Convert risk level to numeric score."""
        scores = {
            ContaminationRisk.NONE: 0.0, ContaminationRisk.LOW: 20.0,
            ContaminationRisk.MEDIUM: 50.0, ContaminationRisk.HIGH: 80.0,
            ContaminationRisk.CRITICAL: 100.0,
        }
        return scores.get(risk, 80.0)

    def _calculate_provenance_hash(self, check_id: str, fuel_a: str, fuel_b: str, rule_id: str) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {"check_id": check_id, "fuel_a": fuel_a, "fuel_b": fuel_b, "rule_id": rule_id, "version": "1.0.0"}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
