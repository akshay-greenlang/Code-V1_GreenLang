# -*- coding: utf-8 -*-
"""
ESGCharacteristicsEngine - PACK-010 SFDR Article 8 Engine 5

Define, track, and report on promoted environmental and social
characteristics per SFDR Article 8 requirements. This engine manages
the lifecycle of ESG characteristics: definition, binding element
enforcement, attainment measurement, and benchmark comparison.

SFDR Article 8 Context:
    Article 8 products promote environmental or social characteristics,
    or a combination thereof, provided that the investee companies follow
    good governance practices. The promoted characteristics must be clearly
    defined, measurable, and reported against in periodic disclosures.

Binding Elements (RTS Annex II):
    - Hard constraints that must be met at all times
    - Minimum thresholds for environmental/social metrics
    - Exclusion criteria that form part of the investment strategy
    - Must be disclosed in pre-contractual documents

Sustainability Indicators (RTS Annex II, Template Table 1):
    - Quantitative measures used to assess attainment
    - Linked to each promoted characteristic
    - Reported in periodic disclosures with methodology

Zero-Hallucination:
    - All attainment calculations use deterministic arithmetic
    - No LLM involvement in numeric calculation paths
    - Benchmark comparisons use exact formula-based differences
    - SHA-256 provenance hash on every result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value within a given range.

    Args:
        value: The value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(min_val, min(value, max_val))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CharacteristicType(str, Enum):
    """Type of ESG characteristic promoted under Article 8."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"


class CharacteristicStatus(str, Enum):
    """Lifecycle status of a characteristic."""
    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class AttainmentStatus(str, Enum):
    """Whether a characteristic target has been met."""
    ATTAINED = "attained"
    PARTIALLY_ATTAINED = "partially_attained"
    NOT_ATTAINED = "not_attained"
    NOT_MEASURED = "not_measured"


class BindingElementStatus(str, Enum):
    """Enforcement status of a binding element."""
    COMPLIANT = "compliant"
    BREACHED = "breached"
    WARNING = "warning"
    NOT_EVALUATED = "not_evaluated"


class MeasurementFrequency(str, Enum):
    """Frequency of sustainability indicator measurement."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"


class BenchmarkType(str, Enum):
    """Type of reference benchmark for comparison."""
    DESIGNATED_REFERENCE = "designated_reference"
    EU_CLIMATE_TRANSITION = "eu_climate_transition"
    EU_PARIS_ALIGNED = "eu_paris_aligned"
    BROAD_MARKET = "broad_market"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Pre-defined Characteristic Definitions
# ---------------------------------------------------------------------------


ENVIRONMENTAL_CHARACTERISTICS: Dict[str, Dict[str, Any]] = {
    "climate_mitigation": {
        "name": "Climate Change Mitigation",
        "description": "Contribute to climate change mitigation through GHG emission reduction",
        "default_metric": "carbon_intensity_tco2e_per_m_revenue",
        "default_target": 30.0,
        "unit": "% reduction vs benchmark",
        "taxonomy_objective": "climate_change_mitigation",
    },
    "water_stewardship": {
        "name": "Water Stewardship",
        "description": "Promote sustainable water use and protection of water resources",
        "default_metric": "water_intensity_m3_per_m_revenue",
        "default_target": 20.0,
        "unit": "% reduction vs benchmark",
        "taxonomy_objective": "sustainable_use_water_marine",
    },
    "biodiversity_protection": {
        "name": "Biodiversity Protection",
        "description": "Protect and restore biodiversity and ecosystems",
        "default_metric": "biodiversity_impact_score",
        "default_target": 70.0,
        "unit": "score (0-100)",
        "taxonomy_objective": "biodiversity_ecosystems",
    },
    "pollution_prevention": {
        "name": "Pollution Prevention",
        "description": "Prevent and control pollution to air, water, and soil",
        "default_metric": "pollution_incidents_per_year",
        "default_target": 0.0,
        "unit": "incidents",
        "taxonomy_objective": "pollution_prevention_control",
    },
    "circular_economy": {
        "name": "Circular Economy",
        "description": "Transition to a circular economy through waste reduction and recycling",
        "default_metric": "waste_recycling_rate_pct",
        "default_target": 60.0,
        "unit": "% recycled",
        "taxonomy_objective": "circular_economy",
    },
    "renewable_energy": {
        "name": "Renewable Energy",
        "description": "Promote renewable energy generation and consumption",
        "default_metric": "renewable_energy_share_pct",
        "default_target": 50.0,
        "unit": "% of total energy",
        "taxonomy_objective": "climate_change_mitigation",
    },
    "energy_efficiency": {
        "name": "Energy Efficiency",
        "description": "Improve energy efficiency across operations and products",
        "default_metric": "energy_intensity_mwh_per_m_revenue",
        "default_target": 25.0,
        "unit": "% improvement vs baseline",
        "taxonomy_objective": "climate_change_mitigation",
    },
    "clean_transport": {
        "name": "Clean Transportation",
        "description": "Support transition to clean and sustainable transport modes",
        "default_metric": "ev_fleet_pct",
        "default_target": 30.0,
        "unit": "% electric/hybrid fleet",
        "taxonomy_objective": "climate_change_mitigation",
    },
    "sustainable_agriculture": {
        "name": "Sustainable Agriculture",
        "description": "Promote sustainable agricultural practices and food systems",
        "default_metric": "sustainable_sourcing_pct",
        "default_target": 50.0,
        "unit": "% sustainably sourced",
        "taxonomy_objective": "biodiversity_ecosystems",
    },
    "ocean_conservation": {
        "name": "Ocean Conservation",
        "description": "Protect marine ecosystems and promote sustainable ocean use",
        "default_metric": "marine_impact_score",
        "default_target": 65.0,
        "unit": "score (0-100)",
        "taxonomy_objective": "sustainable_use_water_marine",
    },
    "forest_protection": {
        "name": "Forest Protection",
        "description": "Protect forests and promote sustainable forestry practices",
        "default_metric": "deforestation_free_supply_chain_pct",
        "default_target": 100.0,
        "unit": "% deforestation-free",
        "taxonomy_objective": "biodiversity_ecosystems",
    },
    "waste_reduction": {
        "name": "Waste Reduction",
        "description": "Minimize waste generation and promote waste hierarchy",
        "default_metric": "waste_to_landfill_tonnes_per_m_revenue",
        "default_target": 15.0,
        "unit": "% reduction vs baseline",
        "taxonomy_objective": "circular_economy",
    },
    "green_buildings": {
        "name": "Green Buildings",
        "description": "Promote energy-efficient and sustainable building practices",
        "default_metric": "green_certified_buildings_pct",
        "default_target": 40.0,
        "unit": "% certified green",
        "taxonomy_objective": "climate_change_mitigation",
    },
    "air_quality": {
        "name": "Air Quality Improvement",
        "description": "Reduce air pollutant emissions and improve air quality",
        "default_metric": "nox_sox_emissions_tonnes",
        "default_target": 20.0,
        "unit": "% reduction vs baseline",
        "taxonomy_objective": "pollution_prevention_control",
    },
    "soil_health": {
        "name": "Soil Health",
        "description": "Protect and restore soil health and prevent contamination",
        "default_metric": "contaminated_sites_remediated_pct",
        "default_target": 80.0,
        "unit": "% remediated",
        "taxonomy_objective": "pollution_prevention_control",
    },
}

SOCIAL_CHARACTERISTICS: Dict[str, Dict[str, Any]] = {
    "labor_rights": {
        "name": "Labor Rights Protection",
        "description": "Respect and promote fundamental labor rights per ILO conventions",
        "default_metric": "labor_rights_compliance_score",
        "default_target": 80.0,
        "unit": "score (0-100)",
        "un_sdg": [8],
    },
    "health_safety": {
        "name": "Health & Safety",
        "description": "Ensure workplace health and safety standards",
        "default_metric": "ltir_per_million_hours",
        "default_target": 2.0,
        "unit": "LTIR",
        "un_sdg": [3, 8],
    },
    "diversity_inclusion": {
        "name": "Diversity & Inclusion",
        "description": "Promote diversity, equity, and inclusion in the workplace",
        "default_metric": "board_gender_diversity_pct",
        "default_target": 33.0,
        "unit": "% female board members",
        "un_sdg": [5, 10],
    },
    "fair_wages": {
        "name": "Fair Wages",
        "description": "Ensure fair and living wages across operations and supply chain",
        "default_metric": "living_wage_coverage_pct",
        "default_target": 100.0,
        "unit": "% employees at living wage",
        "un_sdg": [1, 8],
    },
    "community_development": {
        "name": "Community Development",
        "description": "Support local community development and stakeholder engagement",
        "default_metric": "community_investment_pct_revenue",
        "default_target": 1.0,
        "unit": "% of revenue",
        "un_sdg": [11],
    },
    "education_access": {
        "name": "Education Access",
        "description": "Promote access to quality education and training",
        "default_metric": "training_hours_per_employee",
        "default_target": 40.0,
        "unit": "hours per employee per year",
        "un_sdg": [4],
    },
    "anti_discrimination": {
        "name": "Anti-Discrimination",
        "description": "Zero tolerance for discrimination based on protected characteristics",
        "default_metric": "discrimination_incidents",
        "default_target": 0.0,
        "unit": "reported incidents",
        "un_sdg": [5, 10],
    },
    "supply_chain_labor": {
        "name": "Supply Chain Labor Standards",
        "description": "Ensure labor standards compliance throughout the supply chain",
        "default_metric": "supplier_audit_compliance_pct",
        "default_target": 90.0,
        "unit": "% suppliers compliant",
        "un_sdg": [8, 12],
    },
    "data_privacy": {
        "name": "Data Privacy & Protection",
        "description": "Protect personal data and privacy rights of stakeholders",
        "default_metric": "data_breach_incidents",
        "default_target": 0.0,
        "unit": "incidents per year",
        "un_sdg": [16],
    },
    "indigenous_rights": {
        "name": "Indigenous Rights",
        "description": "Respect and protect indigenous peoples' rights and lands",
        "default_metric": "fpic_compliance_pct",
        "default_target": 100.0,
        "unit": "% operations with FPIC",
        "un_sdg": [10, 16],
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SustainabilityIndicator(BaseModel):
    """Quantitative indicator used to measure ESG characteristic attainment.

    Per SFDR RTS Annex II, each promoted characteristic must have at least
    one sustainability indicator with a defined methodology and data source.
    """
    indicator_id: str = Field(default_factory=_new_uuid, description="Unique indicator identifier")
    indicator_name: str = Field(description="Human-readable indicator name")
    characteristic_id: str = Field(description="Linked characteristic identifier")
    methodology: str = Field(default="", description="Measurement methodology description")
    data_source: str = Field(default="", description="Primary data source for this indicator")
    frequency: MeasurementFrequency = Field(
        default=MeasurementFrequency.QUARTERLY,
        description="Measurement frequency"
    )
    unit: str = Field(default="", description="Unit of measurement")
    current_value: Optional[float] = Field(default=None, description="Most recent measurement value")
    previous_value: Optional[float] = Field(default=None, description="Previous period value")
    target_value: Optional[float] = Field(default=None, description="Target value")
    measurement_date: Optional[datetime] = Field(default=None, description="Date of current measurement")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BindingElement(BaseModel):
    """Binding element that constitutes a hard constraint in the strategy.

    Per SFDR RTS, binding elements are commitments that must be met at all
    times. Breaching a binding element is a compliance violation.
    """
    element_id: str = Field(default_factory=_new_uuid, description="Unique element identifier")
    commitment_name: str = Field(description="Name of the binding commitment")
    characteristic_id: str = Field(description="Linked characteristic identifier")
    minimum_threshold: float = Field(description="Minimum threshold that must be maintained")
    maximum_threshold: Optional[float] = Field(
        default=None, description="Maximum threshold (for metrics where lower is better)"
    )
    measurement_method: str = Field(default="", description="How compliance is measured")
    current_value: Optional[float] = Field(default=None, description="Current measured value")
    status: BindingElementStatus = Field(
        default=BindingElementStatus.NOT_EVALUATED,
        description="Current compliance status"
    )
    last_evaluated: Optional[datetime] = Field(default=None, description="Last evaluation timestamp")
    breach_count: int = Field(default=0, description="Number of times this element has been breached")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("minimum_threshold", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> float:
        """Coerce threshold to float."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


class CharacteristicDefinition(BaseModel):
    """Definition of a promoted ESG characteristic.

    Each Article 8 product must clearly define which environmental and/or
    social characteristics it promotes, with measurable targets and
    sustainability indicators.
    """
    characteristic_id: str = Field(default_factory=_new_uuid, description="Unique characteristic identifier")
    name: str = Field(description="Characteristic display name")
    characteristic_type: CharacteristicType = Field(description="Environmental or Social")
    description: str = Field(default="", description="Detailed description of the characteristic")
    metric: str = Field(default="", description="Primary metric used to track this characteristic")
    target: Optional[float] = Field(default=None, description="Target value for the metric")
    unit: str = Field(default="", description="Unit of measurement")
    status: CharacteristicStatus = Field(
        default=CharacteristicStatus.ACTIVE, description="Lifecycle status"
    )
    binding: bool = Field(default=False, description="Whether this has binding elements")
    taxonomy_objective: Optional[str] = Field(
        default=None, description="Linked EU Taxonomy environmental objective"
    )
    un_sdgs: List[int] = Field(default_factory=list, description="Linked UN Sustainable Development Goals")
    indicators: List[SustainabilityIndicator] = Field(
        default_factory=list, description="Sustainability indicators for measurement"
    )
    binding_elements: List[BindingElement] = Field(
        default_factory=list, description="Binding elements (hard constraints)"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last update timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class AttainmentResult(BaseModel):
    """Result of measuring characteristic attainment against target.

    Tracks actual vs target performance and calculates attainment percentage
    for periodic disclosure reporting.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    characteristic_id: str = Field(description="Measured characteristic")
    characteristic_name: str = Field(default="", description="Characteristic display name")
    characteristic_type: CharacteristicType = Field(description="Environmental or Social")
    target_value: float = Field(description="Target value for the period")
    actual_value: float = Field(description="Actual measured value")
    attainment_pct: float = Field(description="Attainment percentage (actual/target * 100)")
    status: AttainmentStatus = Field(description="Attainment status")
    period_start: Optional[datetime] = Field(default=None, description="Measurement period start")
    period_end: Optional[datetime] = Field(default=None, description="Measurement period end")
    delta_vs_previous: Optional[float] = Field(
        default=None, description="Change vs previous period"
    )
    notes: str = Field(default="", description="Additional context")
    measured_at: datetime = Field(default_factory=_utcnow, description="Measurement timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BenchmarkComparison(BaseModel):
    """Comparison of product ESG performance vs reference benchmark.

    Per SFDR RTS, Article 8 products must explain how the promoted
    characteristics differ from the designated reference benchmark.
    """
    comparison_id: str = Field(default_factory=_new_uuid, description="Unique comparison identifier")
    characteristic_id: str = Field(description="Compared characteristic")
    characteristic_name: str = Field(default="", description="Characteristic display name")
    benchmark_name: str = Field(default="", description="Reference benchmark name")
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.BROAD_MARKET, description="Type of reference benchmark"
    )
    product_value: float = Field(description="Product metric value")
    benchmark_value: float = Field(description="Benchmark metric value")
    absolute_difference: float = Field(description="Product value minus benchmark value")
    relative_difference_pct: float = Field(
        description="Percentage difference vs benchmark"
    )
    outperforms: bool = Field(description="Whether product outperforms benchmark")
    unit: str = Field(default="", description="Unit of measurement")
    comparison_date: datetime = Field(default_factory=_utcnow, description="Comparison timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class StrategyValidationResult(BaseModel):
    """Result of validating the overall ESG strategy configuration.

    Checks that the Article 8 product has properly defined characteristics,
    binding elements, and sustainability indicators per SFDR requirements.
    """
    validation_id: str = Field(default_factory=_new_uuid, description="Unique validation identifier")
    valid: bool = Field(description="Whether the strategy passes all checks")
    total_characteristics: int = Field(default=0, description="Total defined characteristics")
    environmental_count: int = Field(default=0, description="Environmental characteristics count")
    social_count: int = Field(default=0, description="Social characteristics count")
    binding_elements_count: int = Field(default=0, description="Total binding elements")
    indicators_count: int = Field(default=0, description="Total sustainability indicators")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    checks_passed: List[str] = Field(default_factory=list, description="Checks that passed")
    validated_at: datetime = Field(default_factory=_utcnow, description="Validation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CharacteristicsSummary(BaseModel):
    """Summary of all promoted characteristics and their status.

    Used for pre-contractual and periodic disclosure reporting.
    """
    summary_id: str = Field(default_factory=_new_uuid, description="Unique summary identifier")
    product_name: str = Field(default="", description="Financial product name")
    total_characteristics: int = Field(default=0, description="Total active characteristics")
    environmental_characteristics: int = Field(default=0, description="Active environmental count")
    social_characteristics: int = Field(default=0, description="Active social count")
    binding_elements_total: int = Field(default=0, description="Total binding elements")
    binding_elements_compliant: int = Field(default=0, description="Compliant binding elements")
    binding_elements_breached: int = Field(default=0, description="Breached binding elements")
    overall_attainment_pct: float = Field(default=0.0, description="Average attainment across characteristics")
    attainment_results: List[AttainmentResult] = Field(
        default_factory=list, description="Per-characteristic attainment"
    )
    benchmark_comparisons: List[BenchmarkComparison] = Field(
        default_factory=list, description="Benchmark comparisons"
    )
    generated_at: datetime = Field(default_factory=_utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class ESGCharacteristicsConfig(BaseModel):
    """Configuration for the ESGCharacteristicsEngine.

    Controls thresholds, benchmark settings, and default parameters
    for characteristic definition and measurement.
    """
    product_name: str = Field(default="", description="Financial product name")
    product_isin: str = Field(default="", description="Product ISIN identifier")
    attainment_threshold_high: float = Field(
        default=90.0, description="Threshold above which attainment is considered full"
    )
    attainment_threshold_partial: float = Field(
        default=50.0, description="Threshold above which attainment is partial"
    )
    binding_element_warning_margin: float = Field(
        default=10.0, description="Percentage margin for binding element warnings"
    )
    default_benchmark_name: str = Field(
        default="MSCI World", description="Default reference benchmark name"
    )
    default_benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.BROAD_MARKET, description="Default benchmark type"
    )
    require_binding_elements: bool = Field(
        default=True, description="Whether at least one binding element is required"
    )
    min_characteristics: int = Field(
        default=1, description="Minimum number of characteristics required"
    )
    max_characteristics: int = Field(
        default=50, description="Maximum number of characteristics allowed"
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

ESGCharacteristicsConfig.model_rebuild()
SustainabilityIndicator.model_rebuild()
BindingElement.model_rebuild()
CharacteristicDefinition.model_rebuild()
AttainmentResult.model_rebuild()
BenchmarkComparison.model_rebuild()
StrategyValidationResult.model_rebuild()
CharacteristicsSummary.model_rebuild()


# ---------------------------------------------------------------------------
# ESGCharacteristicsEngine
# ---------------------------------------------------------------------------


class ESGCharacteristicsEngine:
    """
    ESG characteristics management engine for SFDR Article 8 products.

    Manages the full lifecycle of promoted environmental and social
    characteristics: definition from pre-defined catalogs or custom inputs,
    binding element enforcement, attainment measurement, benchmark
    comparison, and strategy validation.

    Attributes:
        config: Engine configuration parameters.
        _characteristics: In-memory store of defined characteristics.
        _measurements: Historical measurement records.

    Example:
        >>> engine = ESGCharacteristicsEngine()
        >>> chars = engine.define_characteristics(["climate_mitigation", "labor_rights"])
        >>> engine.add_binding_element(chars[0].characteristic_id, {
        ...     "commitment_name": "Carbon Intensity Reduction",
        ...     "minimum_threshold": 30.0,
        ...     "measurement_method": "WACI vs benchmark"
        ... })
        >>> results = engine.measure_attainment({"climate_mitigation_id": 35.0})
        >>> assert results[0].status == AttainmentStatus.ATTAINED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESGCharacteristicsEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = ESGCharacteristicsConfig(**config)
        elif config and isinstance(config, ESGCharacteristicsConfig):
            self.config = config
        else:
            self.config = ESGCharacteristicsConfig()

        self._characteristics: Dict[str, CharacteristicDefinition] = {}
        self._measurements: Dict[str, List[AttainmentResult]] = defaultdict(list)
        self._benchmark_data: Dict[str, Dict[str, float]] = {}

        logger.info(
            "ESGCharacteristicsEngine initialized (version=%s, product=%s)",
            _MODULE_VERSION,
            self.config.product_name,
        )

    # ------------------------------------------------------------------
    # Characteristic Definition
    # ------------------------------------------------------------------

    def define_characteristics(
        self,
        characteristic_keys: List[str],
        custom_targets: Optional[Dict[str, float]] = None,
    ) -> List[CharacteristicDefinition]:
        """Define promoted characteristics from pre-defined catalogs.

        Looks up characteristics from the environmental and social catalogs
        and creates formal definitions with optional custom targets.

        Args:
            characteristic_keys: List of catalog keys (e.g. "climate_mitigation").
            custom_targets: Optional mapping of key to custom target value.

        Returns:
            List of created CharacteristicDefinition objects.

        Raises:
            ValueError: If a key is not found in either catalog.
        """
        start = _utcnow()
        custom_targets = custom_targets or {}
        created: List[CharacteristicDefinition] = []

        for key in characteristic_keys:
            catalog_entry = self._lookup_catalog(key)
            if catalog_entry is None:
                raise ValueError(
                    f"Characteristic key '{key}' not found in environmental or social catalogs"
                )

            char_type = (
                CharacteristicType.ENVIRONMENTAL
                if key in ENVIRONMENTAL_CHARACTERISTICS
                else CharacteristicType.SOCIAL
            )

            target = custom_targets.get(key, catalog_entry.get("default_target"))

            definition = CharacteristicDefinition(
                name=catalog_entry["name"],
                characteristic_type=char_type,
                description=catalog_entry["description"],
                metric=catalog_entry.get("default_metric", ""),
                target=target,
                unit=catalog_entry.get("unit", ""),
                taxonomy_objective=catalog_entry.get("taxonomy_objective"),
                un_sdgs=catalog_entry.get("un_sdg", []),
            )
            definition.provenance_hash = _compute_hash(definition)
            self._characteristics[definition.characteristic_id] = definition
            created.append(definition)

            logger.info(
                "Defined characteristic: %s (%s, target=%.2f)",
                definition.name,
                char_type.value,
                target if target is not None else 0.0,
            )

        logger.info(
            "Defined %d characteristics in %dms",
            len(created),
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return created

    def define_custom_characteristic(
        self,
        name: str,
        characteristic_type: CharacteristicType,
        description: str,
        metric: str,
        target: float,
        unit: str = "",
        taxonomy_objective: Optional[str] = None,
        un_sdgs: Optional[List[int]] = None,
    ) -> CharacteristicDefinition:
        """Define a custom characteristic not in the pre-defined catalogs.

        Args:
            name: Characteristic display name.
            characteristic_type: ENVIRONMENTAL or SOCIAL.
            description: Detailed description.
            metric: Primary metric identifier.
            target: Target value.
            unit: Unit of measurement.
            taxonomy_objective: Optional linked Taxonomy objective.
            un_sdgs: Optional linked UN SDGs.

        Returns:
            Created CharacteristicDefinition.
        """
        definition = CharacteristicDefinition(
            name=name,
            characteristic_type=characteristic_type,
            description=description,
            metric=metric,
            target=target,
            unit=unit,
            taxonomy_objective=taxonomy_objective,
            un_sdgs=un_sdgs or [],
        )
        definition.provenance_hash = _compute_hash(definition)
        self._characteristics[definition.characteristic_id] = definition

        logger.info(
            "Defined custom characteristic: %s (%s)", name, characteristic_type.value
        )
        return definition

    def get_characteristic(self, characteristic_id: str) -> Optional[CharacteristicDefinition]:
        """Retrieve a characteristic by its identifier.

        Args:
            characteristic_id: The unique identifier of the characteristic.

        Returns:
            CharacteristicDefinition if found, None otherwise.
        """
        return self._characteristics.get(characteristic_id)

    def list_characteristics(
        self,
        char_type: Optional[CharacteristicType] = None,
        status: Optional[CharacteristicStatus] = None,
    ) -> List[CharacteristicDefinition]:
        """List all defined characteristics with optional filtering.

        Args:
            char_type: Filter by ENVIRONMENTAL or SOCIAL.
            status: Filter by lifecycle status.

        Returns:
            Filtered list of CharacteristicDefinition objects.
        """
        results = list(self._characteristics.values())
        if char_type is not None:
            results = [c for c in results if c.characteristic_type == char_type]
        if status is not None:
            results = [c for c in results if c.status == status]
        return results

    # ------------------------------------------------------------------
    # Binding Elements
    # ------------------------------------------------------------------

    def add_binding_element(
        self,
        characteristic_id: str,
        element_data: Dict[str, Any],
    ) -> BindingElement:
        """Add a binding element to a characteristic.

        Binding elements are hard constraints that must be met at all times.
        They form the enforceable part of the Article 8 investment strategy.

        Args:
            characteristic_id: Target characteristic identifier.
            element_data: Dictionary with binding element fields:
                - commitment_name (str): Name of the commitment.
                - minimum_threshold (float): Minimum threshold.
                - maximum_threshold (float, optional): Maximum threshold.
                - measurement_method (str): How compliance is measured.

        Returns:
            Created BindingElement.

        Raises:
            ValueError: If characteristic_id is not found.
        """
        char = self._characteristics.get(characteristic_id)
        if char is None:
            raise ValueError(f"Characteristic '{characteristic_id}' not found")

        element = BindingElement(
            commitment_name=element_data.get("commitment_name", ""),
            characteristic_id=characteristic_id,
            minimum_threshold=element_data.get("minimum_threshold", 0.0),
            maximum_threshold=element_data.get("maximum_threshold"),
            measurement_method=element_data.get("measurement_method", ""),
        )
        element.provenance_hash = _compute_hash(element)

        char.binding_elements.append(element)
        char.binding = True
        char.updated_at = _utcnow()

        logger.info(
            "Added binding element '%s' to characteristic '%s' (threshold=%.2f)",
            element.commitment_name,
            char.name,
            element.minimum_threshold,
        )
        return element

    def evaluate_binding_elements(
        self,
        values: Dict[str, float],
    ) -> List[BindingElement]:
        """Evaluate all binding elements against current values.

        For each binding element, checks whether the current measured value
        meets the minimum threshold (or stays below maximum threshold where
        lower is better).

        Args:
            values: Mapping of characteristic_id to current measured value.

        Returns:
            List of all BindingElement objects with updated status.
        """
        start = _utcnow()
        evaluated: List[BindingElement] = []

        for char in self._characteristics.values():
            current_value = values.get(char.characteristic_id)
            for element in char.binding_elements:
                if current_value is not None:
                    element.current_value = current_value
                    element.last_evaluated = _utcnow()
                    element.status = self._evaluate_single_binding(
                        element, current_value
                    )
                else:
                    element.status = BindingElementStatus.NOT_EVALUATED
                element.provenance_hash = _compute_hash(element)
                evaluated.append(element)

        breached = sum(1 for e in evaluated if e.status == BindingElementStatus.BREACHED)
        logger.info(
            "Evaluated %d binding elements (%d breached) in %dms",
            len(evaluated),
            breached,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return evaluated

    def get_binding_elements(
        self,
        characteristic_id: Optional[str] = None,
    ) -> List[BindingElement]:
        """Retrieve binding elements, optionally filtered by characteristic.

        Args:
            characteristic_id: If provided, filter to this characteristic only.

        Returns:
            List of BindingElement objects.
        """
        elements: List[BindingElement] = []
        for char in self._characteristics.values():
            if characteristic_id and char.characteristic_id != characteristic_id:
                continue
            elements.extend(char.binding_elements)
        return elements

    # ------------------------------------------------------------------
    # Sustainability Indicators
    # ------------------------------------------------------------------

    def add_sustainability_indicator(
        self,
        characteristic_id: str,
        indicator_data: Dict[str, Any],
    ) -> SustainabilityIndicator:
        """Add a sustainability indicator to a characteristic.

        Per SFDR RTS, each promoted characteristic must have at least one
        sustainability indicator with methodology and data source.

        Args:
            characteristic_id: Target characteristic identifier.
            indicator_data: Dictionary with indicator fields.

        Returns:
            Created SustainabilityIndicator.

        Raises:
            ValueError: If characteristic_id is not found.
        """
        char = self._characteristics.get(characteristic_id)
        if char is None:
            raise ValueError(f"Characteristic '{characteristic_id}' not found")

        frequency = indicator_data.get("frequency", "quarterly")
        if isinstance(frequency, str):
            frequency = MeasurementFrequency(frequency)

        indicator = SustainabilityIndicator(
            indicator_name=indicator_data.get("indicator_name", ""),
            characteristic_id=characteristic_id,
            methodology=indicator_data.get("methodology", ""),
            data_source=indicator_data.get("data_source", ""),
            frequency=frequency,
            unit=indicator_data.get("unit", ""),
            target_value=indicator_data.get("target_value"),
        )
        indicator.provenance_hash = _compute_hash(indicator)

        char.indicators.append(indicator)
        char.updated_at = _utcnow()

        logger.info(
            "Added indicator '%s' to characteristic '%s'",
            indicator.indicator_name,
            char.name,
        )
        return indicator

    def get_sustainability_indicators(
        self,
        characteristic_id: Optional[str] = None,
    ) -> List[SustainabilityIndicator]:
        """Retrieve sustainability indicators, optionally filtered.

        Args:
            characteristic_id: If provided, filter to this characteristic.

        Returns:
            List of SustainabilityIndicator objects.
        """
        indicators: List[SustainabilityIndicator] = []
        for char in self._characteristics.values():
            if characteristic_id and char.characteristic_id != characteristic_id:
                continue
            indicators.extend(char.indicators)
        return indicators

    def update_indicator_value(
        self,
        indicator_id: str,
        value: float,
        measurement_date: Optional[datetime] = None,
    ) -> Optional[SustainabilityIndicator]:
        """Update a sustainability indicator with a new measurement.

        Args:
            indicator_id: Target indicator identifier.
            value: New measurement value.
            measurement_date: Optional date of measurement (defaults to now).

        Returns:
            Updated SustainabilityIndicator, or None if not found.
        """
        for char in self._characteristics.values():
            for indicator in char.indicators:
                if indicator.indicator_id == indicator_id:
                    indicator.previous_value = indicator.current_value
                    indicator.current_value = value
                    indicator.measurement_date = measurement_date or _utcnow()
                    indicator.provenance_hash = _compute_hash(indicator)
                    logger.info(
                        "Updated indicator '%s' value: %.4f -> %.4f",
                        indicator.indicator_name,
                        indicator.previous_value if indicator.previous_value else 0.0,
                        value,
                    )
                    return indicator
        return None

    # ------------------------------------------------------------------
    # Attainment Measurement
    # ------------------------------------------------------------------

    def measure_attainment(
        self,
        actual_values: Dict[str, float],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> List[AttainmentResult]:
        """Measure attainment for all characteristics against actuals.

        Calculates the attainment percentage for each characteristic by
        comparing the actual measured value against the target. Results are
        stored in the measurement history and hashed for provenance.

        Attainment formula:
            attainment_pct = (actual / target) * 100  (for higher-is-better)
            attainment_pct = (target / actual) * 100  (for lower-is-better)

        Args:
            actual_values: Mapping of characteristic_id to actual value.
            period_start: Start of measurement period.
            period_end: End of measurement period.

        Returns:
            List of AttainmentResult objects.
        """
        start = _utcnow()
        results: List[AttainmentResult] = []

        for char_id, char in self._characteristics.items():
            if char.status != CharacteristicStatus.ACTIVE:
                continue
            if char_id not in actual_values:
                continue

            actual = actual_values[char_id]
            target = char.target if char.target is not None else 0.0

            attainment_pct = self._calculate_attainment_pct(
                actual, target, char.metric
            )
            status = self._determine_attainment_status(attainment_pct)

            # Compute delta vs previous measurement
            delta = None
            previous = self._measurements.get(char_id)
            if previous and len(previous) > 0:
                delta = actual - previous[-1].actual_value

            result = AttainmentResult(
                characteristic_id=char_id,
                characteristic_name=char.name,
                characteristic_type=char.characteristic_type,
                target_value=target,
                actual_value=actual,
                attainment_pct=round(attainment_pct, 2),
                status=status,
                period_start=period_start,
                period_end=period_end,
                delta_vs_previous=round(delta, 4) if delta is not None else None,
            )
            result.provenance_hash = _compute_hash(result)
            results.append(result)
            self._measurements[char_id].append(result)

        logger.info(
            "Measured attainment for %d characteristics in %dms",
            len(results),
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return results

    def get_attainment_history(
        self,
        characteristic_id: str,
    ) -> List[AttainmentResult]:
        """Retrieve historical attainment measurements for a characteristic.

        Args:
            characteristic_id: Target characteristic identifier.

        Returns:
            List of historical AttainmentResult objects.
        """
        return list(self._measurements.get(characteristic_id, []))

    # ------------------------------------------------------------------
    # Benchmark Comparison
    # ------------------------------------------------------------------

    def set_benchmark_data(
        self,
        benchmark_name: str,
        benchmark_values: Dict[str, float],
    ) -> None:
        """Set benchmark reference data for comparison.

        Args:
            benchmark_name: Name of the reference benchmark.
            benchmark_values: Mapping of characteristic_id to benchmark value.
        """
        self._benchmark_data[benchmark_name] = benchmark_values
        logger.info(
            "Set benchmark data for '%s' with %d values",
            benchmark_name,
            len(benchmark_values),
        )

    def compare_to_benchmark(
        self,
        actual_values: Dict[str, float],
        benchmark_name: Optional[str] = None,
        benchmark_type: Optional[BenchmarkType] = None,
    ) -> List[BenchmarkComparison]:
        """Compare product performance against reference benchmark.

        For each characteristic, calculates absolute and relative differences
        between the product value and the benchmark value.

        Args:
            actual_values: Mapping of characteristic_id to product value.
            benchmark_name: Name of benchmark (defaults to config default).
            benchmark_type: Type of benchmark (defaults to config default).

        Returns:
            List of BenchmarkComparison objects.

        Raises:
            ValueError: If benchmark data has not been set.
        """
        start = _utcnow()
        bm_name = benchmark_name or self.config.default_benchmark_name
        bm_type = benchmark_type or self.config.default_benchmark_type

        bm_values = self._benchmark_data.get(bm_name)
        if bm_values is None:
            raise ValueError(
                f"Benchmark data for '{bm_name}' has not been set. "
                f"Call set_benchmark_data() first."
            )

        comparisons: List[BenchmarkComparison] = []
        for char_id, product_val in actual_values.items():
            char = self._characteristics.get(char_id)
            if char is None:
                continue

            bm_val = bm_values.get(char_id, 0.0)
            abs_diff = product_val - bm_val
            rel_diff = _safe_divide(abs_diff, abs(bm_val), 0.0) * 100.0

            # Determine outperformance based on metric type
            outperforms = self._determine_outperformance(
                product_val, bm_val, char.metric
            )

            comparison = BenchmarkComparison(
                characteristic_id=char_id,
                characteristic_name=char.name,
                benchmark_name=bm_name,
                benchmark_type=bm_type,
                product_value=round(product_val, 4),
                benchmark_value=round(bm_val, 4),
                absolute_difference=round(abs_diff, 4),
                relative_difference_pct=round(rel_diff, 2),
                outperforms=outperforms,
                unit=char.unit,
            )
            comparison.provenance_hash = _compute_hash(comparison)
            comparisons.append(comparison)

        logger.info(
            "Compared %d characteristics to benchmark '%s' in %dms",
            len(comparisons),
            bm_name,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return comparisons

    # ------------------------------------------------------------------
    # Strategy Validation
    # ------------------------------------------------------------------

    def validate_strategy(self) -> StrategyValidationResult:
        """Validate the overall ESG strategy configuration.

        Performs comprehensive checks per SFDR Article 8 requirements:
        - At least one characteristic must be defined
        - Both environmental and social types should be considered
        - Binding elements should be present if required
        - Each characteristic should have at least one indicator

        Returns:
            StrategyValidationResult with errors, warnings, and passed checks.
        """
        start = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        checks_passed: List[str] = []

        active_chars = self.list_characteristics(status=CharacteristicStatus.ACTIVE)
        env_chars = [c for c in active_chars if c.characteristic_type == CharacteristicType.ENVIRONMENTAL]
        soc_chars = [c for c in active_chars if c.characteristic_type == CharacteristicType.SOCIAL]

        # Check 1: Minimum characteristics
        if len(active_chars) < self.config.min_characteristics:
            errors.append(
                f"At least {self.config.min_characteristics} characteristic(s) required, "
                f"found {len(active_chars)}"
            )
        else:
            checks_passed.append(
                f"Minimum characteristics met ({len(active_chars)} >= {self.config.min_characteristics})"
            )

        # Check 2: Maximum characteristics
        if len(active_chars) > self.config.max_characteristics:
            errors.append(
                f"Maximum {self.config.max_characteristics} characteristics allowed, "
                f"found {len(active_chars)}"
            )
        else:
            checks_passed.append("Characteristics count within maximum limit")

        # Check 3: At least one type present
        if len(active_chars) > 0 and len(env_chars) == 0 and len(soc_chars) == 0:
            errors.append("No environmental or social characteristics are active")
        else:
            if len(env_chars) > 0:
                checks_passed.append(f"{len(env_chars)} environmental characteristics defined")
            if len(soc_chars) > 0:
                checks_passed.append(f"{len(soc_chars)} social characteristics defined")

        # Check 4: Type diversity warning
        if len(env_chars) > 0 and len(soc_chars) == 0:
            warnings.append(
                "Only environmental characteristics defined; consider adding social characteristics"
            )
        elif len(soc_chars) > 0 and len(env_chars) == 0:
            warnings.append(
                "Only social characteristics defined; consider adding environmental characteristics"
            )

        # Check 5: Binding elements
        total_binding = sum(len(c.binding_elements) for c in active_chars)
        if self.config.require_binding_elements and total_binding == 0:
            errors.append(
                "At least one binding element is required but none are defined"
            )
        elif total_binding > 0:
            checks_passed.append(f"{total_binding} binding elements defined")

        # Check 6: Sustainability indicators
        total_indicators = sum(len(c.indicators) for c in active_chars)
        chars_without_indicators = [c for c in active_chars if len(c.indicators) == 0]
        if chars_without_indicators:
            warnings.append(
                f"{len(chars_without_indicators)} characteristic(s) have no sustainability indicators: "
                + ", ".join(c.name for c in chars_without_indicators)
            )
        if total_indicators > 0:
            checks_passed.append(f"{total_indicators} sustainability indicators defined")

        # Check 7: Targets defined
        chars_without_targets = [c for c in active_chars if c.target is None]
        if chars_without_targets:
            warnings.append(
                f"{len(chars_without_targets)} characteristic(s) have no target value: "
                + ", ".join(c.name for c in chars_without_targets)
            )
        else:
            checks_passed.append("All characteristics have target values")

        is_valid = len(errors) == 0

        result = StrategyValidationResult(
            valid=is_valid,
            total_characteristics=len(active_chars),
            environmental_count=len(env_chars),
            social_count=len(soc_chars),
            binding_elements_count=total_binding,
            indicators_count=total_indicators,
            errors=errors,
            warnings=warnings,
            checks_passed=checks_passed,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Strategy validation %s (%d errors, %d warnings) in %dms",
            "PASSED" if is_valid else "FAILED",
            len(errors),
            len(warnings),
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Summary & Reporting
    # ------------------------------------------------------------------

    def get_characteristics_summary(
        self,
        actual_values: Optional[Dict[str, float]] = None,
        benchmark_name: Optional[str] = None,
    ) -> CharacteristicsSummary:
        """Generate a comprehensive summary of all characteristics.

        Combines attainment results, binding element status, and benchmark
        comparisons into a single summary for disclosure reporting.

        Args:
            actual_values: Optional current values for attainment calculation.
            benchmark_name: Optional benchmark for comparison.

        Returns:
            CharacteristicsSummary with all metrics.
        """
        start = _utcnow()
        active_chars = self.list_characteristics(status=CharacteristicStatus.ACTIVE)
        env_count = sum(
            1 for c in active_chars
            if c.characteristic_type == CharacteristicType.ENVIRONMENTAL
        )
        soc_count = sum(
            1 for c in active_chars
            if c.characteristic_type == CharacteristicType.SOCIAL
        )

        # Binding elements summary
        all_bindings = self.get_binding_elements()
        compliant = sum(
            1 for b in all_bindings if b.status == BindingElementStatus.COMPLIANT
        )
        breached = sum(
            1 for b in all_bindings if b.status == BindingElementStatus.BREACHED
        )

        # Attainment results
        attainment_results: List[AttainmentResult] = []
        if actual_values:
            attainment_results = self.measure_attainment(actual_values)

        # Benchmark comparisons
        comparisons: List[BenchmarkComparison] = []
        bm_name = benchmark_name or self.config.default_benchmark_name
        if actual_values and bm_name in self._benchmark_data:
            comparisons = self.compare_to_benchmark(actual_values, bm_name)

        # Overall attainment
        overall_att = 0.0
        if attainment_results:
            overall_att = sum(r.attainment_pct for r in attainment_results) / len(
                attainment_results
            )

        summary = CharacteristicsSummary(
            product_name=self.config.product_name,
            total_characteristics=len(active_chars),
            environmental_characteristics=env_count,
            social_characteristics=soc_count,
            binding_elements_total=len(all_bindings),
            binding_elements_compliant=compliant,
            binding_elements_breached=breached,
            overall_attainment_pct=round(overall_att, 2),
            attainment_results=attainment_results,
            benchmark_comparisons=comparisons,
        )
        summary.provenance_hash = _compute_hash(summary)

        logger.info(
            "Generated characteristics summary: %d chars, %.1f%% attainment in %dms",
            len(active_chars),
            overall_att,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return summary

    # ------------------------------------------------------------------
    # Characteristic Management
    # ------------------------------------------------------------------

    def update_characteristic_status(
        self,
        characteristic_id: str,
        new_status: CharacteristicStatus,
    ) -> Optional[CharacteristicDefinition]:
        """Update the lifecycle status of a characteristic.

        Args:
            characteristic_id: Target characteristic identifier.
            new_status: New status to apply.

        Returns:
            Updated CharacteristicDefinition, or None if not found.
        """
        char = self._characteristics.get(characteristic_id)
        if char is None:
            logger.warning("Characteristic '%s' not found for status update", characteristic_id)
            return None

        old_status = char.status
        char.status = new_status
        char.updated_at = _utcnow()
        char.provenance_hash = _compute_hash(char)

        logger.info(
            "Updated characteristic '%s' status: %s -> %s",
            char.name,
            old_status.value,
            new_status.value,
        )
        return char

    def remove_characteristic(self, characteristic_id: str) -> bool:
        """Remove a characteristic definition entirely.

        Args:
            characteristic_id: Target characteristic identifier.

        Returns:
            True if removed, False if not found.
        """
        if characteristic_id in self._characteristics:
            char = self._characteristics.pop(characteristic_id)
            self._measurements.pop(characteristic_id, None)
            logger.info("Removed characteristic: %s", char.name)
            return True
        return False

    # ------------------------------------------------------------------
    # Available Catalogs
    # ------------------------------------------------------------------

    def list_available_environmental(self) -> Dict[str, Dict[str, Any]]:
        """Return the catalog of available environmental characteristics.

        Returns:
            Dictionary of environmental characteristic definitions.
        """
        return dict(ENVIRONMENTAL_CHARACTERISTICS)

    def list_available_social(self) -> Dict[str, Dict[str, Any]]:
        """Return the catalog of available social characteristics.

        Returns:
            Dictionary of social characteristic definitions.
        """
        return dict(SOCIAL_CHARACTERISTICS)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _lookup_catalog(self, key: str) -> Optional[Dict[str, Any]]:
        """Look up a characteristic key in the pre-defined catalogs.

        Args:
            key: Catalog key string.

        Returns:
            Catalog entry dict if found, None otherwise.
        """
        if key in ENVIRONMENTAL_CHARACTERISTICS:
            return ENVIRONMENTAL_CHARACTERISTICS[key]
        if key in SOCIAL_CHARACTERISTICS:
            return SOCIAL_CHARACTERISTICS[key]
        return None

    def _evaluate_single_binding(
        self,
        element: BindingElement,
        current_value: float,
    ) -> BindingElementStatus:
        """Evaluate a single binding element against a measured value.

        For minimum-threshold elements (higher is better):
            - COMPLIANT if current >= minimum
            - WARNING if current >= minimum * (1 - margin%)
            - BREACHED otherwise

        For maximum-threshold elements (lower is better):
            - COMPLIANT if current <= maximum
            - WARNING if current <= maximum * (1 + margin%)
            - BREACHED otherwise

        Args:
            element: The binding element to evaluate.
            current_value: Current measured value.

        Returns:
            Evaluated BindingElementStatus.
        """
        margin_factor = self.config.binding_element_warning_margin / 100.0

        if element.maximum_threshold is not None:
            # Lower is better (e.g., pollution incidents)
            max_thresh = element.maximum_threshold
            warning_thresh = max_thresh * (1.0 + margin_factor)
            if current_value <= max_thresh:
                return BindingElementStatus.COMPLIANT
            elif current_value <= warning_thresh:
                return BindingElementStatus.WARNING
            else:
                element.breach_count += 1
                return BindingElementStatus.BREACHED
        else:
            # Higher is better (e.g., renewable energy %)
            min_thresh = element.minimum_threshold
            warning_thresh = min_thresh * (1.0 - margin_factor)
            if current_value >= min_thresh:
                return BindingElementStatus.COMPLIANT
            elif current_value >= warning_thresh:
                return BindingElementStatus.WARNING
            else:
                element.breach_count += 1
                return BindingElementStatus.BREACHED

    def _calculate_attainment_pct(
        self,
        actual: float,
        target: float,
        metric: str,
    ) -> float:
        """Calculate attainment percentage based on metric direction.

        For most metrics, higher actual values mean better performance:
            attainment = (actual / target) * 100

        For metrics where lower is better (incidents, emissions intensity):
            attainment = (target / actual) * 100

        Args:
            actual: Actual measured value.
            target: Target value.
            metric: Metric identifier to determine direction.

        Returns:
            Attainment percentage (0-200, capped).
        """
        lower_is_better_keywords = [
            "incidents", "intensity", "ltir", "breach", "contaminated",
            "pollution", "waste_to_landfill", "nox_sox",
        ]

        is_lower_better = any(kw in metric.lower() for kw in lower_is_better_keywords)

        if is_lower_better:
            if actual == 0.0 and target == 0.0:
                return 100.0
            if actual == 0.0:
                return 200.0  # Perfect for zero-incidents type metrics
            pct = _safe_divide(target, actual, 0.0) * 100.0
        else:
            pct = _safe_divide(actual, target, 0.0) * 100.0

        return _clamp(pct, 0.0, 200.0)

    def _determine_attainment_status(self, attainment_pct: float) -> AttainmentStatus:
        """Determine attainment status from percentage.

        Args:
            attainment_pct: Calculated attainment percentage.

        Returns:
            Appropriate AttainmentStatus enum value.
        """
        if attainment_pct >= self.config.attainment_threshold_high:
            return AttainmentStatus.ATTAINED
        elif attainment_pct >= self.config.attainment_threshold_partial:
            return AttainmentStatus.PARTIALLY_ATTAINED
        else:
            return AttainmentStatus.NOT_ATTAINED

    def _determine_outperformance(
        self,
        product_val: float,
        benchmark_val: float,
        metric: str,
    ) -> bool:
        """Determine if product outperforms benchmark for a given metric.

        Args:
            product_val: Product metric value.
            benchmark_val: Benchmark metric value.
            metric: Metric identifier to determine direction.

        Returns:
            True if product outperforms benchmark.
        """
        lower_is_better_keywords = [
            "incidents", "intensity", "ltir", "breach", "contaminated",
            "pollution", "waste_to_landfill", "nox_sox",
        ]

        is_lower_better = any(kw in metric.lower() for kw in lower_is_better_keywords)

        if is_lower_better:
            return product_val < benchmark_val
        else:
            return product_val > benchmark_val
