"""
Data Quality Framework
======================

Comprehensive data quality framework for emission factor management.
Implements Data Quality Indicators (DQI), Pedigree Matrix scoring,
and multi-dimensional quality assessment per ISO 14044/14067 guidelines.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum, IntEnum
from datetime import datetime, date, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import logging
import statistics
import math

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PEDIGREE MATRIX IMPLEMENTATION
# =============================================================================

class PedigreeScore(IntEnum):
    """
    Pedigree Matrix scores (1-5 scale).
    Lower score = higher quality.
    Based on Weidema & Wesnaes (1996) methodology.
    """
    EXCELLENT = 1  # Best quality
    GOOD = 2
    FAIR = 3
    POOR = 4
    VERY_POOR = 5  # Worst quality


@dataclass
class PedigreeMatrix:
    """
    Pedigree Matrix for emission factor data quality assessment.

    The Pedigree Matrix provides a semi-quantitative approach to
    assessing data quality across 5 dimensions, widely used in LCA.

    Dimensions:
    1. Reliability - How was the data collected/verified?
    2. Completeness - Is the sample representative?
    3. Temporal correlation - How recent is the data?
    4. Geographic correlation - How locally representative?
    5. Technological correlation - How technology-specific?

    Reference: Weidema & Wesnaes (1996), Ecoinvent guidelines
    """

    reliability: PedigreeScore = PedigreeScore.FAIR
    completeness: PedigreeScore = PedigreeScore.FAIR
    temporal_correlation: PedigreeScore = PedigreeScore.FAIR
    geographic_correlation: PedigreeScore = PedigreeScore.FAIR
    technological_correlation: PedigreeScore = PedigreeScore.FAIR

    @classmethod
    def from_scores(
        cls,
        reliability: int = 3,
        completeness: int = 3,
        temporal: int = 3,
        geographic: int = 3,
        technological: int = 3
    ) -> 'PedigreeMatrix':
        """Create from integer scores (1-5)."""
        return cls(
            reliability=PedigreeScore(max(1, min(5, reliability))),
            completeness=PedigreeScore(max(1, min(5, completeness))),
            temporal_correlation=PedigreeScore(max(1, min(5, temporal))),
            geographic_correlation=PedigreeScore(max(1, min(5, geographic))),
            technological_correlation=PedigreeScore(max(1, min(5, technological))),
        )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            'reliability': self.reliability.value,
            'completeness': self.completeness.value,
            'temporal_correlation': self.temporal_correlation.value,
            'geographic_correlation': self.geographic_correlation.value,
            'technological_correlation': self.technological_correlation.value,
        }

    def calculate_dqi(self) -> float:
        """
        Calculate Data Quality Indicator (0-100, higher is better).

        Converts 1-5 scores to 0-100 scale where higher = better quality.
        """
        scores = [
            self.reliability.value,
            self.completeness.value,
            self.temporal_correlation.value,
            self.geographic_correlation.value,
            self.technological_correlation.value,
        ]

        # Invert scores: 1 becomes 5, 5 becomes 1
        inverted = [(6 - s) for s in scores]

        # Calculate percentage (5 dimensions * max score 5 = 25)
        dqi = (sum(inverted) / 25) * 100

        return round(dqi, 2)

    def calculate_uncertainty_factor(self) -> float:
        """
        Calculate geometric standard deviation factor from pedigree scores.

        Based on the pedigree matrix uncertainty approach from Ecoinvent.
        Returns a factor to apply to base uncertainty.
        """
        # Uncertainty contribution factors (squared geometric SD)
        uncertainty_contributions = {
            1: 0.0,      # No additional uncertainty
            2: 0.0006,   # Small uncertainty
            3: 0.002,    # Moderate uncertainty
            4: 0.008,    # High uncertainty
            5: 0.04,     # Very high uncertainty
        }

        total_variance = (
            uncertainty_contributions[self.reliability.value] +
            uncertainty_contributions[self.completeness.value] +
            uncertainty_contributions[self.temporal_correlation.value] +
            uncertainty_contributions[self.geographic_correlation.value] +
            uncertainty_contributions[self.technological_correlation.value]
        )

        # Geometric standard deviation factor
        gsd_factor = math.exp(math.sqrt(total_variance))

        return round(gsd_factor, 4)

    @property
    def quality_tier(self) -> str:
        """Determine quality tier from DQI."""
        dqi = self.calculate_dqi()
        if dqi >= 80:
            return "tier_3"  # High quality
        elif dqi >= 60:
            return "tier_2"  # Medium quality
        else:
            return "tier_1"  # Low quality (default)


# Pedigree Matrix Scoring Guidelines
PEDIGREE_GUIDELINES = {
    'reliability': {
        1: "Verified data based on measurements",
        2: "Verified data partly based on assumptions or non-verified data based on measurements",
        3: "Non-verified data partly based on qualified estimates",
        4: "Qualified estimate (e.g., by industrial expert)",
        5: "Non-qualified estimate",
    },
    'completeness': {
        1: "Representative data from all sites relevant for market considered over adequate time",
        2: "Representative data from >50% of sites relevant for market over adequate time",
        3: "Representative data from <50% of sites relevant for market over adequate time",
        4: "Representative data from only one site relevant for market or some sites but shorter time",
        5: "Representativeness unknown or data from a small number of sites and shorter time",
    },
    'temporal_correlation': {
        1: "Less than 3 years difference to year of study",
        2: "Less than 6 years difference",
        3: "Less than 10 years difference",
        4: "Less than 15 years difference",
        5: "Age of data unknown or more than 15 years difference",
    },
    'geographic_correlation': {
        1: "Data from area under study",
        2: "Average data from larger area in which study area is included",
        3: "Data from area with similar production conditions",
        4: "Data from area with slightly similar production conditions",
        5: "Data from unknown or very different area",
    },
    'technological_correlation': {
        1: "Data from enterprises, processes and materials under study",
        2: "Data from processes/materials under study but from different enterprises",
        3: "Data from processes/materials under study but different technology",
        4: "Data on related processes/materials",
        5: "Data on related processes on laboratory scale or from different technology",
    },
}


# =============================================================================
# DATA QUALITY INDICATORS
# =============================================================================

class DataQualityIndicator(str, Enum):
    """Data Quality Indicators (DQI) categories."""
    RELIABILITY = "reliability"
    COMPLETENESS = "completeness"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    TECHNOLOGICAL = "technological"
    UNCERTAINTY = "uncertainty"
    CONSISTENCY = "consistency"
    TRANSPARENCY = "transparency"


@dataclass
class TemporalRepresentativeness:
    """Assess temporal representativeness of emission factor."""
    reference_year: int
    current_year: int = field(default_factory=lambda: datetime.now().year)
    data_collection_start: Optional[date] = None
    data_collection_end: Optional[date] = None

    def get_age_years(self) -> int:
        """Get age of data in years."""
        return self.current_year - self.reference_year

    def get_score(self) -> PedigreeScore:
        """Calculate temporal representativeness score."""
        age = self.get_age_years()

        if age <= 3:
            return PedigreeScore.EXCELLENT
        elif age <= 6:
            return PedigreeScore.GOOD
        elif age <= 10:
            return PedigreeScore.FAIR
        elif age <= 15:
            return PedigreeScore.POOR
        else:
            return PedigreeScore.VERY_POOR

    def is_current(self, threshold_years: int = 5) -> bool:
        """Check if data is current within threshold."""
        return self.get_age_years() <= threshold_years


@dataclass
class GeographicRepresentativeness:
    """Assess geographic representativeness of emission factor."""
    factor_region: str  # Region the factor represents
    target_region: str  # Region we want to apply it to
    factor_country: Optional[str] = None
    target_country: Optional[str] = None

    # Known region hierarchies
    REGION_HIERARCHY = {
        'global': 0,
        'continental': 1,  # europe, asia, americas, etc.
        'regional': 2,     # EU, OECD, etc.
        'country': 3,
        'state': 4,
        'local': 5,
    }

    def get_score(self) -> PedigreeScore:
        """Calculate geographic representativeness score."""
        # Exact match
        if self._regions_match():
            return PedigreeScore.EXCELLENT

        # Same country
        if self.factor_country and self.target_country:
            if self.factor_country == self.target_country:
                return PedigreeScore.GOOD

        # Same continent/region
        if self._similar_region():
            return PedigreeScore.FAIR

        # Different region but similar conditions
        if self._compatible_regions():
            return PedigreeScore.POOR

        # Very different regions
        return PedigreeScore.VERY_POOR

    def _regions_match(self) -> bool:
        """Check if regions match exactly."""
        return (
            self.factor_region.lower() == self.target_region.lower() or
            (self.factor_country and self.target_country and
             self.factor_country.lower() == self.target_country.lower())
        )

    def _similar_region(self) -> bool:
        """Check if regions are similar."""
        # Define similar region groups
        similar_groups = [
            {'europe', 'eu', 'oecd_europe', 'western_europe'},
            {'asia', 'asia_pacific', 'east_asia'},
            {'north_america', 'usa', 'canada'},
            {'latin_america', 'south_america'},
        ]

        factor_lower = self.factor_region.lower()
        target_lower = self.target_region.lower()

        for group in similar_groups:
            if factor_lower in group and target_lower in group:
                return True

        return False

    def _compatible_regions(self) -> bool:
        """Check if regions have compatible production conditions."""
        # Global factors are compatible with everything
        if self.factor_region.lower() in ['global', 'world', 'glo']:
            return True
        return False


@dataclass
class TechnologicalRepresentativeness:
    """Assess technological representativeness of emission factor."""
    factor_technology: Optional[str] = None
    target_technology: Optional[str] = None
    factor_production_route: Optional[str] = None
    target_production_route: Optional[str] = None
    is_generic: bool = False
    is_industry_average: bool = False

    def get_score(self) -> PedigreeScore:
        """Calculate technological representativeness score."""
        # Exact technology match
        if self._technology_matches():
            return PedigreeScore.EXCELLENT

        # Same production route, different specifics
        if self._similar_production_route():
            return PedigreeScore.GOOD

        # Industry average
        if self.is_industry_average:
            return PedigreeScore.FAIR

        # Generic/default values
        if self.is_generic:
            return PedigreeScore.POOR

        # Unknown technology
        return PedigreeScore.VERY_POOR

    def _technology_matches(self) -> bool:
        """Check if technologies match."""
        if not self.factor_technology or not self.target_technology:
            return False
        return self.factor_technology.lower() == self.target_technology.lower()

    def _similar_production_route(self) -> bool:
        """Check if production routes are similar."""
        if not self.factor_production_route or not self.target_production_route:
            return False
        return self.factor_production_route.lower() == self.target_production_route.lower()


# =============================================================================
# QUALITY ASSESSMENT ENGINE
# =============================================================================

class QualityAssessmentResult(BaseModel):
    """Result of quality assessment."""
    factor_id: str
    overall_dqi: float = Field(..., ge=0, le=100)
    quality_tier: str
    pedigree_scores: Dict[str, int]
    uncertainty_factor: float
    is_acceptable: bool
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    assessment_details: Dict[str, Any] = Field(default_factory=dict)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class DataQualityAssessor:
    """
    Comprehensive data quality assessment engine.

    Implements:
    - Pedigree matrix scoring
    - DQI calculation
    - Uncertainty factor derivation
    - Quality tier assignment
    - Regulatory compliance checking (CBAM, CSRD)
    """

    # Minimum DQI thresholds by use case
    DQI_THRESHOLDS = {
        'cbam_reporting': 70,
        'csrd_reporting': 60,
        'financial_reporting': 80,
        'internal_reporting': 50,
        'screening': 30,
    }

    def __init__(
        self,
        min_acceptable_dqi: float = 50.0,
        require_uncertainty: bool = False,
        require_source_verification: bool = True,
    ):
        """
        Initialize quality assessor.

        Args:
            min_acceptable_dqi: Minimum DQI score to pass quality check
            require_uncertainty: Whether uncertainty data is required
            require_source_verification: Whether source verification is required
        """
        self.min_acceptable_dqi = min_acceptable_dqi
        self.require_uncertainty = require_uncertainty
        self.require_source_verification = require_source_verification

    def assess_emission_factor(
        self,
        factor: Dict[str, Any],
        target_context: Optional[Dict[str, Any]] = None,
        use_case: str = 'internal_reporting'
    ) -> QualityAssessmentResult:
        """
        Assess quality of an emission factor.

        Args:
            factor: Emission factor data
            target_context: Context for applying the factor
            use_case: Intended use case (affects threshold)

        Returns:
            QualityAssessmentResult with detailed assessment
        """
        issues = []
        recommendations = []

        # 1. Calculate pedigree matrix scores
        pedigree = self._calculate_pedigree(factor, target_context)

        # 2. Calculate overall DQI
        overall_dqi = pedigree.calculate_dqi()

        # 3. Calculate uncertainty factor
        uncertainty_factor = pedigree.calculate_uncertainty_factor()

        # 4. Determine quality tier
        quality_tier = pedigree.quality_tier

        # 5. Check acceptability
        threshold = self.DQI_THRESHOLDS.get(use_case, self.min_acceptable_dqi)
        is_acceptable = overall_dqi >= threshold

        # 6. Generate issues and recommendations
        issues, recommendations = self._generate_feedback(
            pedigree, factor, target_context, use_case
        )

        return QualityAssessmentResult(
            factor_id=factor.get('factor_id', 'unknown'),
            overall_dqi=overall_dqi,
            quality_tier=quality_tier,
            pedigree_scores=pedigree.to_dict(),
            uncertainty_factor=uncertainty_factor,
            is_acceptable=is_acceptable,
            issues=issues,
            recommendations=recommendations,
            assessment_details={
                'use_case': use_case,
                'threshold': threshold,
                'pedigree_guidelines': {
                    dim: PEDIGREE_GUIDELINES[dim][score]
                    for dim, score in pedigree.to_dict().items()
                }
            },
        )

    def _calculate_pedigree(
        self,
        factor: Dict[str, Any],
        target_context: Optional[Dict[str, Any]] = None
    ) -> PedigreeMatrix:
        """Calculate pedigree matrix for emission factor."""
        target = target_context or {}

        # Get existing quality scores if available
        quality = factor.get('quality', {})

        if quality:
            return PedigreeMatrix.from_scores(
                reliability=quality.get('reliability_score', 3),
                completeness=quality.get('completeness_score', 3),
                temporal=quality.get('temporal_score', 3),
                geographic=quality.get('geographic_score', 3),
                technological=quality.get('technology_score', 3),
            )

        # Calculate from factor attributes
        # Temporal representativeness
        reference_year = factor.get('reference_year', datetime.now().year - 5)
        temporal = TemporalRepresentativeness(reference_year=reference_year)
        temporal_score = temporal.get_score()

        # Geographic representativeness
        geo = GeographicRepresentativeness(
            factor_region=factor.get('region', 'global'),
            target_region=target.get('region', factor.get('region', 'global')),
            factor_country=factor.get('country_code'),
            target_country=target.get('country_code'),
        )
        geographic_score = geo.get_score()

        # Technological representativeness
        tech = TechnologicalRepresentativeness(
            factor_production_route=factor.get('production_route'),
            target_production_route=target.get('production_route'),
            is_generic=factor.get('is_generic', False),
            is_industry_average=True,  # Most factors are industry averages
        )
        technological_score = tech.get_score()

        # Reliability based on source
        source = factor.get('source', {})
        source_type = source.get('source_type', 'unknown')
        reliability_score = self._assess_source_reliability(source_type)

        # Completeness - default to Fair if unknown
        completeness_score = PedigreeScore.FAIR

        return PedigreeMatrix(
            reliability=reliability_score,
            completeness=completeness_score,
            temporal_correlation=temporal_score,
            geographic_correlation=geographic_score,
            technological_correlation=technological_score,
        )

    def _assess_source_reliability(self, source_type: str) -> PedigreeScore:
        """Assess reliability based on data source type."""
        source_reliability = {
            'defra': PedigreeScore.GOOD,
            'epa_egrid': PedigreeScore.GOOD,
            'epa_ghg': PedigreeScore.GOOD,
            'ecoinvent': PedigreeScore.GOOD,
            'iea': PedigreeScore.GOOD,
            'ipcc_ar6': PedigreeScore.EXCELLENT,
            'ipcc_ar7': PedigreeScore.EXCELLENT,
            'world_bank': PedigreeScore.FAIR,
            'fao': PedigreeScore.FAIR,
            'customer_provided': PedigreeScore.POOR,
            'calculated': PedigreeScore.FAIR,
        }

        return source_reliability.get(source_type.lower(), PedigreeScore.POOR)

    def _generate_feedback(
        self,
        pedigree: PedigreeMatrix,
        factor: Dict[str, Any],
        target_context: Optional[Dict[str, Any]],
        use_case: str
    ) -> Tuple[List[str], List[str]]:
        """Generate issues and recommendations based on assessment."""
        issues = []
        recommendations = []

        scores = pedigree.to_dict()

        # Check each dimension
        for dim, score in scores.items():
            if score >= 4:
                issues.append(f"Poor {dim.replace('_', ' ')} (score: {score})")

                # Add specific recommendations
                if dim == 'temporal_correlation':
                    recommendations.append("Consider updating to more recent emission factors")
                elif dim == 'geographic_correlation':
                    recommendations.append("Consider using region-specific emission factors")
                elif dim == 'technological_correlation':
                    recommendations.append("Consider using technology-specific emission factors")
                elif dim == 'reliability':
                    recommendations.append("Consider using verified/measured data sources")
                elif dim == 'completeness':
                    recommendations.append("Consider using more comprehensive data sources")

        # Use case specific checks
        if use_case == 'cbam_reporting':
            if not factor.get('cbam_eligible'):
                issues.append("Factor not marked as CBAM eligible")
            if pedigree.calculate_dqi() < 70:
                issues.append("DQI below CBAM threshold (70)")

        if use_case == 'csrd_reporting':
            if not factor.get('csrd_compliant'):
                issues.append("Factor not marked as CSRD compliant")
            if not factor.get('source'):
                issues.append("Missing source documentation for CSRD audit trail")

        return issues, recommendations

    def assess_batch(
        self,
        factors: List[Dict[str, Any]],
        target_context: Optional[Dict[str, Any]] = None,
        use_case: str = 'internal_reporting'
    ) -> Dict[str, Any]:
        """
        Assess quality of multiple emission factors.

        Returns aggregate statistics and individual assessments.
        """
        results = []
        dqi_scores = []

        for factor in factors:
            result = self.assess_emission_factor(factor, target_context, use_case)
            results.append(result)
            dqi_scores.append(result.overall_dqi)

        # Aggregate statistics
        avg_dqi = statistics.mean(dqi_scores) if dqi_scores else 0
        min_dqi = min(dqi_scores) if dqi_scores else 0
        max_dqi = max(dqi_scores) if dqi_scores else 0
        acceptable_count = sum(1 for r in results if r.is_acceptable)

        tier_distribution = {}
        for result in results:
            tier = result.quality_tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        return {
            'total_factors': len(factors),
            'acceptable_count': acceptable_count,
            'acceptance_rate': (acceptable_count / len(factors) * 100) if factors else 0,
            'average_dqi': round(avg_dqi, 2),
            'min_dqi': round(min_dqi, 2),
            'max_dqi': round(max_dqi, 2),
            'tier_distribution': tier_distribution,
            'individual_results': [r.dict() for r in results],
        }


# =============================================================================
# COMPLETENESS CHECKS
# =============================================================================

class CompletenessChecker:
    """Check data completeness for emission factor datasets."""

    # Required fields by regulatory framework
    REQUIRED_FIELDS = {
        'base': [
            'factor_id', 'factor_value', 'factor_unit',
            'industry', 'region', 'scope_type', 'reference_year'
        ],
        'cbam': [
            'factor_id', 'factor_value', 'factor_unit',
            'product_code', 'country_code', 'production_route',
            'source', 'reference_year'
        ],
        'csrd': [
            'factor_id', 'factor_value', 'factor_unit',
            'industry', 'region', 'scope_type',
            'source', 'quality', 'reference_year'
        ],
        'ghg_protocol': [
            'factor_id', 'factor_value', 'factor_unit',
            'ghg_type', 'scope_type', 'reference_year'
        ],
    }

    def check_completeness(
        self,
        factors: List[Dict[str, Any]],
        framework: str = 'base'
    ) -> Dict[str, Any]:
        """
        Check completeness of emission factor dataset.

        Returns:
            Completeness metrics and missing field report
        """
        required_fields = self.REQUIRED_FIELDS.get(framework, self.REQUIRED_FIELDS['base'])

        field_completeness = {field: 0 for field in required_fields}
        incomplete_records = []
        complete_count = 0

        for i, factor in enumerate(factors):
            missing_fields = []

            for field in required_fields:
                value = self._get_nested_field(factor, field)
                if value is not None and value != '':
                    field_completeness[field] += 1
                else:
                    missing_fields.append(field)

            if missing_fields:
                incomplete_records.append({
                    'index': i,
                    'factor_id': factor.get('factor_id'),
                    'missing_fields': missing_fields,
                })
            else:
                complete_count += 1

        total = len(factors)

        return {
            'framework': framework,
            'total_records': total,
            'complete_records': complete_count,
            'incomplete_records': total - complete_count,
            'completeness_rate': round((complete_count / total * 100) if total else 0, 2),
            'field_completeness': {
                field: round((count / total * 100) if total else 0, 2)
                for field, count in field_completeness.items()
            },
            'incomplete_record_summary': incomplete_records[:10],  # First 10
        }

    def _get_nested_field(self, data: Dict, field: str) -> Any:
        """Get nested field value using dot notation."""
        parts = field.split('.')
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cbam_quality_assessor() -> DataQualityAssessor:
    """Create assessor configured for CBAM reporting."""
    return DataQualityAssessor(
        min_acceptable_dqi=70,
        require_uncertainty=True,
        require_source_verification=True,
    )


def create_csrd_quality_assessor() -> DataQualityAssessor:
    """Create assessor configured for CSRD reporting."""
    return DataQualityAssessor(
        min_acceptable_dqi=60,
        require_uncertainty=False,
        require_source_verification=True,
    )


def create_screening_quality_assessor() -> DataQualityAssessor:
    """Create assessor configured for screening/estimation."""
    return DataQualityAssessor(
        min_acceptable_dqi=30,
        require_uncertainty=False,
        require_source_verification=False,
    )
