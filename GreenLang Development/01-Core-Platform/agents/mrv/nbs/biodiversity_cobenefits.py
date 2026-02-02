# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-007: Biodiversity Co-benefits MRV Agent
==================================================

This agent measures, reports, and verifies biodiversity co-benefits from
nature-based solutions projects following international standards.

Capabilities:
    - Species richness and diversity metrics
    - Habitat quality assessment
    - Ecosystem services quantification
    - IUCN Red List species tracking
    - Connectivity and corridor metrics
    - Co-benefit scoring for carbon projects

Methodologies:
    - CBD (Convention on Biological Diversity) indicators
    - TNFD (Taskforce on Nature-related Financial Disclosures)
    - CCB (Climate, Community & Biodiversity) Standards
    - Gold Standard biodiversity requirements
    - IUCN ecosystem assessment framework

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class BiodiversityIndicator(str, Enum):
    """Biodiversity indicators for assessment."""
    SPECIES_RICHNESS = "species_richness"
    SHANNON_DIVERSITY = "shannon_diversity"
    SIMPSON_DIVERSITY = "simpson_diversity"
    ENDEMISM_RATE = "endemism_rate"
    THREATENED_SPECIES = "threatened_species"
    HABITAT_AREA = "habitat_area"
    HABITAT_QUALITY = "habitat_quality"
    CONNECTIVITY = "connectivity"
    INTACT_FOREST_LANDSCAPE = "intact_forest_landscape"
    MEAN_SPECIES_ABUNDANCE = "mean_species_abundance"


class EcosystemService(str, Enum):
    """Ecosystem services from NBS projects."""
    CARBON_SEQUESTRATION = "carbon_sequestration"
    WATER_REGULATION = "water_regulation"
    EROSION_CONTROL = "erosion_control"
    POLLINATION = "pollination"
    NUTRIENT_CYCLING = "nutrient_cycling"
    PEST_CONTROL = "pest_control"
    RECREATION = "recreation"
    CULTURAL_VALUE = "cultural_value"
    AIR_QUALITY = "air_quality"
    FLOOD_PROTECTION = "flood_protection"


class IUCNCategory(str, Enum):
    """IUCN Red List categories."""
    EXTINCT = "EX"
    EXTINCT_IN_WILD = "EW"
    CRITICALLY_ENDANGERED = "CR"
    ENDANGERED = "EN"
    VULNERABLE = "VU"
    NEAR_THREATENED = "NT"
    LEAST_CONCERN = "LC"
    DATA_DEFICIENT = "DD"
    NOT_EVALUATED = "NE"


class HabitatType(str, Enum):
    """Habitat types for assessment."""
    PRIMARY_FOREST = "primary_forest"
    SECONDARY_FOREST = "secondary_forest"
    PLANTED_FOREST = "planted_forest"
    GRASSLAND_NATURAL = "grassland_natural"
    GRASSLAND_MANAGED = "grassland_managed"
    WETLAND = "wetland"
    COASTAL = "coastal"
    AQUATIC_FRESHWATER = "aquatic_freshwater"
    AQUATIC_MARINE = "aquatic_marine"
    AGRICULTURAL = "agricultural"
    URBAN = "urban"


class AssessmentTier(str, Enum):
    """Assessment methodology tiers."""
    TIER_1 = "tier_1"  # Proxy-based
    TIER_2 = "tier_2"  # Sample-based
    TIER_3 = "tier_3"  # Comprehensive survey


# =============================================================================
# Default Values and Scoring
# =============================================================================

# Habitat quality scores (0-100)
HABITAT_QUALITY_SCORES: Dict[HabitatType, float] = {
    HabitatType.PRIMARY_FOREST: 100.0,
    HabitatType.SECONDARY_FOREST: 75.0,
    HabitatType.PLANTED_FOREST: 45.0,
    HabitatType.GRASSLAND_NATURAL: 80.0,
    HabitatType.GRASSLAND_MANAGED: 50.0,
    HabitatType.WETLAND: 90.0,
    HabitatType.COASTAL: 85.0,
    HabitatType.AQUATIC_FRESHWATER: 85.0,
    HabitatType.AQUATIC_MARINE: 90.0,
    HabitatType.AGRICULTURAL: 25.0,
    HabitatType.URBAN: 10.0,
}

# IUCN category weights for threatened species scoring
IUCN_WEIGHTS: Dict[IUCNCategory, float] = {
    IUCNCategory.CRITICALLY_ENDANGERED: 5.0,
    IUCNCategory.ENDANGERED: 4.0,
    IUCNCategory.VULNERABLE: 3.0,
    IUCNCategory.NEAR_THREATENED: 2.0,
    IUCNCategory.LEAST_CONCERN: 1.0,
    IUCNCategory.DATA_DEFICIENT: 1.5,
}

# Ecosystem service value ranges (USD/ha/yr) - for reference only
ECOSYSTEM_SERVICE_VALUES: Dict[EcosystemService, Dict[str, float]] = {
    EcosystemService.CARBON_SEQUESTRATION: {"min": 50, "max": 500},
    EcosystemService.WATER_REGULATION: {"min": 100, "max": 1000},
    EcosystemService.EROSION_CONTROL: {"min": 50, "max": 300},
    EcosystemService.POLLINATION: {"min": 200, "max": 600},
    EcosystemService.FLOOD_PROTECTION: {"min": 500, "max": 5000},
}


# =============================================================================
# Pydantic Models
# =============================================================================

class SpeciesRecord(BaseModel):
    """Species observation record."""

    species_name: str = Field(..., description="Scientific name")
    common_name: Optional[str] = Field(None, description="Common name")
    iucn_category: Optional[IUCNCategory] = Field(None, description="IUCN status")
    is_endemic: bool = Field(default=False, description="Endemic to region")
    count: int = Field(default=1, ge=0, description="Number observed")
    taxonomic_group: Optional[str] = Field(None, description="Taxonomic group")


class HabitatAssessment(BaseModel):
    """Habitat assessment data."""

    habitat_type: HabitatType = Field(..., description="Habitat type")
    area_ha: float = Field(..., gt=0, description="Area in hectares")
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    fragmentation_index: Optional[float] = Field(None, ge=0, le=1)
    edge_density_m_per_ha: Optional[float] = Field(None, ge=0)
    core_area_ha: Optional[float] = Field(None, ge=0)


class BiodiversityAssessment(BaseModel):
    """Biodiversity assessment input data."""

    assessment_id: str = Field(..., description="Assessment identifier")
    assessment_date: date = Field(..., description="Assessment date")
    site_id: str = Field(..., description="Site identifier")
    area_ha: float = Field(..., gt=0, description="Assessment area")

    # Species data
    species_records: List[SpeciesRecord] = Field(
        default_factory=list, description="Species observations"
    )

    # Habitat data
    habitats: List[HabitatAssessment] = Field(
        default_factory=list, description="Habitat assessments"
    )

    # Direct metrics (if measured)
    measured_shannon_index: Optional[float] = Field(None, ge=0)
    measured_simpson_index: Optional[float] = Field(None, ge=0, le=1)
    measured_connectivity_score: Optional[float] = Field(None, ge=0, le=100)

    # Ecosystem services present
    ecosystem_services: List[EcosystemService] = Field(default_factory=list)


class CobenefitScore(BaseModel):
    """Biodiversity co-benefit score."""

    indicator: BiodiversityIndicator = Field(...)
    score: float = Field(..., ge=0, le=100, description="Score 0-100")
    raw_value: Optional[float] = Field(None, description="Raw metric value")
    interpretation: str = Field(..., description="Score interpretation")

    # Uncertainty
    uncertainty_percent: float = Field(default=25.0, ge=0)
    data_quality: str = Field(
        default="moderate",
        description="Data quality: high, moderate, low"
    )


class BiodiversityInput(BaseModel):
    """Input for Biodiversity Co-benefits MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    reporting_year: int = Field(..., ge=1990)

    assessments: List[BiodiversityAssessment] = Field(..., min_length=1)
    target_tier: AssessmentTier = Field(default=AssessmentTier.TIER_1)

    # Baseline comparison
    baseline_species_count: Optional[int] = Field(None, ge=0)
    baseline_habitat_quality: Optional[float] = Field(None, ge=0, le=100)

    # Standards to report against
    report_ccb_metrics: bool = Field(default=True)
    report_tnfd_metrics: bool = Field(default=False)


class BiodiversityOutput(BaseModel):
    """Output from Biodiversity Co-benefits MRV Agent."""

    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary metrics
    total_area_ha: float = Field(..., ge=0)
    total_species_count: int = Field(..., ge=0)
    threatened_species_count: int = Field(..., ge=0)
    endemic_species_count: int = Field(..., ge=0)

    # Diversity indices
    shannon_diversity_index: Optional[float] = Field(None)
    simpson_diversity_index: Optional[float] = Field(None)

    # Habitat metrics
    average_habitat_quality: float = Field(..., ge=0, le=100)
    habitat_diversity_score: float = Field(..., ge=0, le=100)
    connectivity_score: Optional[float] = Field(None, ge=0, le=100)

    # Ecosystem services
    ecosystem_services_present: List[EcosystemService] = Field(default_factory=list)
    ecosystem_services_score: float = Field(..., ge=0, le=100)

    # Overall co-benefit score
    overall_biodiversity_score: float = Field(..., ge=0, le=100)
    biodiversity_tier: str = Field(
        ..., description="Biodiversity tier: exceptional, high, moderate, low"
    )

    # Additionality (vs baseline)
    species_additionality: Optional[int] = Field(None)
    habitat_quality_improvement: Optional[float] = Field(None)

    # Detailed scores
    indicator_scores: List[CobenefitScore] = Field(...)

    # Uncertainty
    overall_uncertainty_percent: float = Field(...)
    data_quality_rating: str = Field(...)

    # Methodology
    methodology_tier: AssessmentTier = Field(...)
    provenance_hash: str = Field(...)
    calculation_trace: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# Biodiversity Co-benefits MRV Agent
# =============================================================================

class BiodiversityCobenefitsMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-007: Biodiversity Co-benefits MRV Agent

    Measures, reports, and verifies biodiversity co-benefits from NBS projects.
    CRITICAL PATH agent with zero-hallucination guarantee.

    Metrics Calculated:
        - Species richness and diversity
        - Threatened species presence
        - Habitat quality and connectivity
        - Ecosystem services
        - Overall biodiversity co-benefit score

    Usage:
        agent = BiodiversityCobenefitsMRVAgent()
        result = agent.execute({
            "project_id": "BIO-001",
            "assessments": [...],
        })
    """

    AGENT_ID = "GL-MRV-NBS-007"
    AGENT_NAME = "Biodiversity Co-benefits MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="BiodiversityCobenefitsMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Biodiversity co-benefits MRV for NBS projects"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Biodiversity Co-benefits MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute biodiversity assessment."""
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            agent_input = BiodiversityInput(**inputs)
            calculation_trace.append(
                f"Processing {len(agent_input.assessments)} biodiversity assessments"
            )

            total_area = sum(a.area_ha for a in agent_input.assessments)

            # Aggregate species records
            all_species = []
            for assessment in agent_input.assessments:
                all_species.extend(assessment.species_records)

            # Deduplicate species
            unique_species = {s.species_name: s for s in all_species}
            total_species = len(unique_species)

            # Count threatened and endemic species
            threatened_count = sum(
                1 for s in unique_species.values()
                if s.iucn_category in (
                    IUCNCategory.CRITICALLY_ENDANGERED,
                    IUCNCategory.ENDANGERED,
                    IUCNCategory.VULNERABLE
                )
            )
            endemic_count = sum(
                1 for s in unique_species.values() if s.is_endemic
            )

            calculation_trace.append(
                f"Total species: {total_species}, "
                f"Threatened: {threatened_count}, Endemic: {endemic_count}"
            )

            # Calculate diversity indices
            shannon_index = self._calculate_shannon_index(all_species)
            simpson_index = self._calculate_simpson_index(all_species)

            # Calculate habitat metrics
            all_habitats = []
            for assessment in agent_input.assessments:
                all_habitats.extend(assessment.habitats)

            avg_habitat_quality = self._calculate_habitat_quality(all_habitats)
            habitat_diversity = self._calculate_habitat_diversity(all_habitats)
            connectivity = self._calculate_connectivity(
                agent_input.assessments, calculation_trace
            )

            # Ecosystem services
            all_services = set()
            for assessment in agent_input.assessments:
                all_services.update(assessment.ecosystem_services)
            services_score = (len(all_services) / len(EcosystemService)) * 100

            # Calculate indicator scores
            indicator_scores = self._calculate_indicator_scores(
                total_species=total_species,
                threatened_count=threatened_count,
                endemic_count=endemic_count,
                shannon_index=shannon_index,
                simpson_index=simpson_index,
                habitat_quality=avg_habitat_quality,
                connectivity=connectivity,
                tier=agent_input.target_tier,
                calculation_trace=calculation_trace
            )

            # Overall biodiversity score (weighted average)
            overall_score = sum(s.score for s in indicator_scores) / len(indicator_scores)

            # Determine biodiversity tier
            if overall_score >= 80:
                biodiversity_tier = "exceptional"
            elif overall_score >= 60:
                biodiversity_tier = "high"
            elif overall_score >= 40:
                biodiversity_tier = "moderate"
            else:
                biodiversity_tier = "low"

            # Additionality
            species_additionality = None
            habitat_improvement = None
            if agent_input.baseline_species_count is not None:
                species_additionality = total_species - agent_input.baseline_species_count
            if agent_input.baseline_habitat_quality is not None:
                habitat_improvement = avg_habitat_quality - agent_input.baseline_habitat_quality

            # Uncertainty
            uncertainty = {
                AssessmentTier.TIER_1: 50.0,
                AssessmentTier.TIER_2: 30.0,
                AssessmentTier.TIER_3: 15.0,
            }.get(agent_input.target_tier, 50.0)

            data_quality = {
                AssessmentTier.TIER_1: "low",
                AssessmentTier.TIER_2: "moderate",
                AssessmentTier.TIER_3: "high",
            }.get(agent_input.target_tier, "low")

            provenance_hash = self._calculate_provenance_hash(
                inputs, indicator_scores, calculation_trace
            )

            output = BiodiversityOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_species_count=total_species,
                threatened_species_count=threatened_count,
                endemic_species_count=endemic_count,
                shannon_diversity_index=shannon_index,
                simpson_diversity_index=simpson_index,
                average_habitat_quality=avg_habitat_quality,
                habitat_diversity_score=habitat_diversity,
                connectivity_score=connectivity,
                ecosystem_services_present=list(all_services),
                ecosystem_services_score=services_score,
                overall_biodiversity_score=overall_score,
                biodiversity_tier=biodiversity_tier,
                species_additionality=species_additionality,
                habitat_quality_improvement=habitat_improvement,
                indicator_scores=indicator_scores,
                overall_uncertainty_percent=uncertainty,
                data_quality_rating=data_quality,
                methodology_tier=agent_input.target_tier,
                provenance_hash=provenance_hash,
                calculation_trace=calculation_trace,
                warnings=warnings
            )

            self._capture_audit_entry(
                operation="biodiversity_cobenefits_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Biodiversity assessment failed: {str(e)}", exc_info=True)
            raise

    def _calculate_shannon_index(
        self,
        species_records: List[SpeciesRecord]
    ) -> Optional[float]:
        """Calculate Shannon diversity index: H = -sum(pi * ln(pi))"""
        if not species_records:
            return None

        total_count = sum(s.count for s in species_records)
        if total_count == 0:
            return None

        # Calculate proportions
        proportions = [s.count / total_count for s in species_records if s.count > 0]

        # Shannon index
        h = -sum(p * math.log(p) for p in proportions if p > 0)
        return round(h, 3)

    def _calculate_simpson_index(
        self,
        species_records: List[SpeciesRecord]
    ) -> Optional[float]:
        """Calculate Simpson diversity index: D = 1 - sum(pi^2)"""
        if not species_records:
            return None

        total_count = sum(s.count for s in species_records)
        if total_count == 0:
            return None

        proportions = [s.count / total_count for s in species_records if s.count > 0]
        d = 1 - sum(p ** 2 for p in proportions)
        return round(d, 3)

    def _calculate_habitat_quality(
        self,
        habitats: List[HabitatAssessment]
    ) -> float:
        """Calculate area-weighted habitat quality score."""
        if not habitats:
            return 50.0  # Default

        total_area = sum(h.area_ha for h in habitats)
        if total_area == 0:
            return 50.0

        weighted_sum = 0.0
        for h in habitats:
            quality = h.quality_score
            if quality is None:
                quality = HABITAT_QUALITY_SCORES.get(h.habitat_type, 50.0)
            weighted_sum += quality * h.area_ha

        return round(weighted_sum / total_area, 1)

    def _calculate_habitat_diversity(
        self,
        habitats: List[HabitatAssessment]
    ) -> float:
        """Calculate habitat type diversity score."""
        if not habitats:
            return 0.0

        unique_types = len(set(h.habitat_type for h in habitats))
        max_types = len(HabitatType)

        return round((unique_types / max_types) * 100, 1)

    def _calculate_connectivity(
        self,
        assessments: List[BiodiversityAssessment],
        calculation_trace: List[str]
    ) -> Optional[float]:
        """Calculate connectivity score if data available."""
        scores = []
        for a in assessments:
            if a.measured_connectivity_score is not None:
                scores.append(a.measured_connectivity_score)
            elif a.habitats:
                # Estimate from habitat fragmentation
                frag_indices = [
                    h.fragmentation_index for h in a.habitats
                    if h.fragmentation_index is not None
                ]
                if frag_indices:
                    avg_frag = sum(frag_indices) / len(frag_indices)
                    connectivity = (1 - avg_frag) * 100
                    scores.append(connectivity)

        if scores:
            return round(sum(scores) / len(scores), 1)
        return None

    def _calculate_indicator_scores(
        self,
        total_species: int,
        threatened_count: int,
        endemic_count: int,
        shannon_index: Optional[float],
        simpson_index: Optional[float],
        habitat_quality: float,
        connectivity: Optional[float],
        tier: AssessmentTier,
        calculation_trace: List[str]
    ) -> List[CobenefitScore]:
        """Calculate individual indicator scores."""
        scores = []

        # Species richness score (0-100)
        # Score based on typical ranges (0-200 species)
        richness_score = min(100, (total_species / 200) * 100)
        scores.append(CobenefitScore(
            indicator=BiodiversityIndicator.SPECIES_RICHNESS,
            score=richness_score,
            raw_value=float(total_species),
            interpretation=self._interpret_richness(total_species)
        ))

        # Threatened species score
        if threatened_count > 0:
            # Having threatened species indicates high conservation value
            threatened_score = min(100, 50 + (threatened_count * 10))
        else:
            threatened_score = 50  # Neutral if none found
        scores.append(CobenefitScore(
            indicator=BiodiversityIndicator.THREATENED_SPECIES,
            score=threatened_score,
            raw_value=float(threatened_count),
            interpretation=f"{threatened_count} IUCN threatened species"
        ))

        # Diversity indices
        if shannon_index is not None:
            # Shannon typically ranges 0-4+
            diversity_score = min(100, (shannon_index / 4) * 100)
            scores.append(CobenefitScore(
                indicator=BiodiversityIndicator.SHANNON_DIVERSITY,
                score=diversity_score,
                raw_value=shannon_index,
                interpretation=self._interpret_shannon(shannon_index)
            ))

        # Habitat quality
        scores.append(CobenefitScore(
            indicator=BiodiversityIndicator.HABITAT_QUALITY,
            score=habitat_quality,
            raw_value=habitat_quality,
            interpretation=self._interpret_habitat_quality(habitat_quality)
        ))

        # Connectivity
        if connectivity is not None:
            scores.append(CobenefitScore(
                indicator=BiodiversityIndicator.CONNECTIVITY,
                score=connectivity,
                raw_value=connectivity,
                interpretation=self._interpret_connectivity(connectivity)
            ))

        # Endemic species (bonus indicator)
        if endemic_count > 0:
            endemism_score = min(100, 60 + (endemic_count * 8))
            scores.append(CobenefitScore(
                indicator=BiodiversityIndicator.ENDEMISM_RATE,
                score=endemism_score,
                raw_value=float(endemic_count),
                interpretation=f"{endemic_count} endemic species"
            ))

        calculation_trace.append(
            f"Calculated {len(scores)} biodiversity indicators"
        )

        return scores

    def _interpret_richness(self, count: int) -> str:
        """Interpret species richness."""
        if count >= 100:
            return "Very high species richness"
        elif count >= 50:
            return "High species richness"
        elif count >= 20:
            return "Moderate species richness"
        else:
            return "Low species richness"

    def _interpret_shannon(self, h: float) -> str:
        """Interpret Shannon diversity index."""
        if h >= 3.0:
            return "Very high diversity"
        elif h >= 2.0:
            return "High diversity"
        elif h >= 1.0:
            return "Moderate diversity"
        else:
            return "Low diversity"

    def _interpret_habitat_quality(self, score: float) -> str:
        """Interpret habitat quality score."""
        if score >= 80:
            return "Excellent habitat quality"
        elif score >= 60:
            return "Good habitat quality"
        elif score >= 40:
            return "Moderate habitat quality"
        else:
            return "Poor habitat quality"

    def _interpret_connectivity(self, score: float) -> str:
        """Interpret connectivity score."""
        if score >= 80:
            return "Highly connected landscape"
        elif score >= 60:
            return "Moderately connected"
        elif score >= 40:
            return "Fragmented landscape"
        else:
            return "Highly fragmented"

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        scores: List[CobenefitScore],
        trace: List[str]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        content = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "timestamp": DeterministicClock.now().isoformat(),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()


def create_biodiversity_cobenefits_agent(
    enable_audit_trail: bool = True
) -> BiodiversityCobenefitsMRVAgent:
    """Create a Biodiversity Co-benefits MRV Agent instance."""
    return BiodiversityCobenefitsMRVAgent(enable_audit_trail=enable_audit_trail)
