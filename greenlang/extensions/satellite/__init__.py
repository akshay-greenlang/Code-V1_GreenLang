"""
GreenLang Satellite Analysis Module.

Provides satellite imagery analysis capabilities for EUDR compliance
verification, including:

- Sentinel-2 and Landsat imagery access
- Vegetation index calculations (NDVI, EVI, NDWI, NBR)
- Forest cover classification
- Change detection for deforestation monitoring
- Integration with Global Forest Watch alerts
- Complete analysis pipeline for EUDR compliance

Example usage:
    from greenlang.satellite import quick_analysis
    from datetime import datetime

    result = quick_analysis(
        min_lon=-55.0, min_lat=-10.0,
        max_lon=-54.5, max_lat=-9.5,
        pre_date=datetime(2020, 1, 1),
        post_date=datetime(2023, 1, 1),
        use_mock=True
    )

    print(f"Forest loss: {result.change_result.forest_loss_ha:.2f} ha")
    print(f"Compliance: {result.eudr_compliance['compliance_status']}")
"""

from greenlang.satellite.clients.sentinel2_client import (
    Sentinel2Client,
    Sentinel2Image,
    Sentinel2Band,
    BoundingBox,
    SearchResult as Sentinel2SearchResult,
    SceneClassification,
)

from greenlang.satellite.clients.landsat_client import (
    LandsatClient,
    LandsatImage,
    LandsatBand,
    HarmonizedSatelliteClient,
    SearchResult as LandsatSearchResult,
    LandsatQAFlags,
)

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
)

from greenlang.satellite.analysis.change_detection import (
    BiTemporalChangeDetector,
    MultiTemporalAnalyzer,
    ChangeDetectionResult,
    ChangeType,
    ChangeThresholds,
    calculate_ndvi_difference,
    calculate_nbr_difference,
    generate_change_report,
    calculate_forest_loss_area,
)

from greenlang.satellite.models.forest_classifier import (
    ForestClassifier,
    ForestClassificationResult,
    ClassificationThresholds,
    LandCoverClass,
    AdaptiveThresholdClassifier,
)

from greenlang.satellite.alerts.deforestation_alert import (
    DeforestationAlertSystem,
    GlobalForestWatchClient,
    DeforestationAlert,
    AlertAggregation,
    AlertSource,
    AlertSeverity,
    AlertConfidence,
    GeoPolygon,
    create_polygon_from_bbox,
    create_polygon_from_coordinates,
)

from greenlang.satellite.pipeline.analysis_pipeline import (
    DeforestationAnalysisPipeline,
    PipelineConfig,
    PipelineProgress,
    PipelineStage,
    AnalysisResult,
    create_pipeline,
    quick_analysis,
)

__all__ = [
    # Clients
    "Sentinel2Client",
    "Sentinel2Image",
    "Sentinel2Band",
    "BoundingBox",
    "Sentinel2SearchResult",
    "SceneClassification",
    "LandsatClient",
    "LandsatImage",
    "LandsatBand",
    "LandsatSearchResult",
    "LandsatQAFlags",
    "HarmonizedSatelliteClient",

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

    # Change Detection
    "BiTemporalChangeDetector",
    "MultiTemporalAnalyzer",
    "ChangeDetectionResult",
    "ChangeType",
    "ChangeThresholds",
    "calculate_ndvi_difference",
    "calculate_nbr_difference",
    "generate_change_report",
    "calculate_forest_loss_area",

    # Forest Classifier
    "ForestClassifier",
    "ForestClassificationResult",
    "ClassificationThresholds",
    "LandCoverClass",
    "AdaptiveThresholdClassifier",

    # Alerts
    "DeforestationAlertSystem",
    "GlobalForestWatchClient",
    "DeforestationAlert",
    "AlertAggregation",
    "AlertSource",
    "AlertSeverity",
    "AlertConfidence",
    "GeoPolygon",
    "create_polygon_from_bbox",
    "create_polygon_from_coordinates",

    # Pipeline
    "DeforestationAnalysisPipeline",
    "PipelineConfig",
    "PipelineProgress",
    "PipelineStage",
    "AnalysisResult",
    "create_pipeline",
    "quick_analysis",
]

__version__ = "0.1.0"
