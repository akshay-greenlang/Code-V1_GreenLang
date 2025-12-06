"""
Satellite Image Analysis Module.

Provides analysis algorithms for satellite imagery:
- Vegetation indices (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI)
- Change detection (bi-temporal and multi-temporal)
- Forest loss area calculations
"""

from greenlang.satellite.analysis.vegetation_indices import (
    VegetationIndexCalculator,
    IndexType,
    IndexResult,
    calculate_ndvi,
    calculate_evi,
    calculate_ndwi,
    calculate_nbr,
    calculate_savi,
    calculate_msavi,
    calculate_ndmi,
    resample_to_match,
    VegetationIndicesError,
    MissingBandError,
)

from greenlang.satellite.analysis.change_detection import (
    BiTemporalChangeDetector,
    MultiTemporalAnalyzer,
    ChangeDetectionResult,
    ChangeType,
    ChangeThresholds,
    calculate_ndvi_difference,
    calculate_nbr_difference,
    classify_change,
    generate_change_report,
    calculate_forest_loss_area,
    ChangeDetectionError,
    TemporalMismatchError,
)

__all__ = [
    # Vegetation Indices
    "VegetationIndexCalculator",
    "IndexType",
    "IndexResult",
    "calculate_ndvi",
    "calculate_evi",
    "calculate_ndwi",
    "calculate_nbr",
    "calculate_savi",
    "calculate_msavi",
    "calculate_ndmi",
    "resample_to_match",
    "VegetationIndicesError",
    "MissingBandError",

    # Change Detection
    "BiTemporalChangeDetector",
    "MultiTemporalAnalyzer",
    "ChangeDetectionResult",
    "ChangeType",
    "ChangeThresholds",
    "calculate_ndvi_difference",
    "calculate_nbr_difference",
    "classify_change",
    "generate_change_report",
    "calculate_forest_loss_area",
    "ChangeDetectionError",
    "TemporalMismatchError",
]
