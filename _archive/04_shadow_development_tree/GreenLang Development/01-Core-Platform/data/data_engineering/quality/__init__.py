"""
Data Quality Framework
======================

Comprehensive data quality scoring across multiple dimensions.
Includes Pedigree Matrix, DQI, and regulatory compliance assessment.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from greenlang.data_engineering.quality.scoring import (
    DataQualityScorer,
    QualityDimension,
    QualityScore,
    DimensionScore,
    create_emission_factor_scorer,
    create_cbam_scorer,
)
from greenlang.data_engineering.quality.quality_framework import (
    PedigreeMatrix,
    PedigreeScore,
    DataQualityIndicator,
    TemporalRepresentativeness,
    GeographicRepresentativeness,
    TechnologicalRepresentativeness,
    DataQualityAssessor,
    QualityAssessmentResult,
    CompletenessChecker,
    create_cbam_quality_assessor,
    create_csrd_quality_assessor,
    create_screening_quality_assessor,
    PEDIGREE_GUIDELINES,
)

__all__ = [
    # Scoring
    "DataQualityScorer",
    "QualityDimension",
    "QualityScore",
    "DimensionScore",
    "create_emission_factor_scorer",
    "create_cbam_scorer",
    # Pedigree Matrix
    "PedigreeMatrix",
    "PedigreeScore",
    "PEDIGREE_GUIDELINES",
    # DQI
    "DataQualityIndicator",
    "TemporalRepresentativeness",
    "GeographicRepresentativeness",
    "TechnologicalRepresentativeness",
    # Assessment
    "DataQualityAssessor",
    "QualityAssessmentResult",
    "CompletenessChecker",
    # Factory functions
    "create_cbam_quality_assessor",
    "create_csrd_quality_assessor",
    "create_screening_quality_assessor",
]
