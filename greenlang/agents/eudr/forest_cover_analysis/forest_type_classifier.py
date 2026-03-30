# -*- coding: utf-8 -*-
"""
Forest Type Classifier Engine - AGENT-EUDR-004: Forest Cover Analysis (Feature 2)

Classifies forest areas into 10 regulatory categories required for EUDR
compliance using five deterministic classification methods: spectral
signature matching, phenological profiling, structural analysis, multi-
temporal combination, and weighted ensemble voting.

Zero-Hallucination Guarantees:
    - All classifications use deterministic arithmetic (no ML/LLM).
    - Spectral matching: minimum distance and spectral angle mapping
      against static reference libraries.
    - Phenological profiling: template correlation against 12-month
      NDVI profiles.
    - Structural analysis: rule-based decision tree on canopy metrics.
    - Ensemble: weighted majority vote of all methods.
    - SHA-256 provenance hashes on all result objects.
    - No ML/LLM used for any classification computation.

Forest Types (10):
    PRIMARY_TROPICAL:      Undisturbed primary tropical moist forest.
    SECONDARY_TROPICAL:    Regenerated tropical forest (>20 years).
    TROPICAL_DRY:          Deciduous or semi-deciduous tropical forest.
    TEMPERATE_BROADLEAF:   Temperate broadleaf deciduous/mixed forest.
    TEMPERATE_CONIFEROUS:  Temperate needleleaf evergreen forest.
    BOREAL:                High-latitude coniferous forest (taiga).
    MANGROVE:              Coastal tidal forest ecosystem.
    PLANTATION:            Monoculture tree plantation (eucalyptus, pine).
    AGROFORESTRY:          Trees integrated with agricultural production.
    NON_FOREST:            Does not meet forest definition.

Classification Methods (5):
    1. Spectral Signature: Multi-band reflectance matching to reference
       spectral libraries using minimum distance and spectral angle.
    2. Phenological: NDVI time-series profile shape matching against
       10 phenological templates (evergreen, deciduous, bimodal, etc.).
    3. Structural: Rule-based analysis of canopy height, density, and
       texture metrics (primary vs plantation discrimination).
    4. Multi-Temporal: Combined spectral signatures from dry + wet season
       for improved discrimination of deciduous vs evergreen types.
    5. Ensemble: Configurable weighted vote of all 4 methods above.

EUDR Relevance:
    - Article 2(1): Deforestation = conversion of forest to non-forest.
      Classification identifies what TYPE of forest existed.
    - Article 2(4): Forest definition excludes agricultural tree
      plantations. Oil palm and rubber monocultures are NOT forests.
    - Article 9: Geolocation-based evidence of forest type.
    - Article 10: Risk assessment considers forest type sensitivity.

Performance Targets:
    - Single plot classification: <100ms
    - Spectral signature matching: <15ms
    - Phenological profiling: <20ms
    - Structural analysis: <10ms
    - Multi-temporal classification: <25ms
    - Ensemble classification: <80ms
    - Batch classification (100 plots): <5 seconds

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Feature 2: Forest Type Classification)
Agent ID: GL-EUDR-FCA-004
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ForestType(str, Enum):
    """EUDR regulatory forest type classification.

    Ten categories covering all forest types relevant to EUDR
    deforestation-free verification. Includes NON_FOREST for
    completeness of classification output.
    """

    PRIMARY_TROPICAL = "PRIMARY_TROPICAL"
    SECONDARY_TROPICAL = "SECONDARY_TROPICAL"
    TROPICAL_DRY = "TROPICAL_DRY"
    TEMPERATE_BROADLEAF = "TEMPERATE_BROADLEAF"
    TEMPERATE_CONIFEROUS = "TEMPERATE_CONIFEROUS"
    BOREAL = "BOREAL"
    MANGROVE = "MANGROVE"
    PLANTATION = "PLANTATION"
    AGROFORESTRY = "AGROFORESTRY"
    NON_FOREST = "NON_FOREST"

class ClassificationMethod(str, Enum):
    """Available forest type classification methods."""

    SPECTRAL_SIGNATURE = "SPECTRAL_SIGNATURE"
    PHENOLOGICAL = "PHENOLOGICAL"
    STRUCTURAL = "STRUCTURAL"
    MULTI_TEMPORAL = "MULTI_TEMPORAL"
    ENSEMBLE = "ENSEMBLE"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: FAO minimum canopy cover percentage for forest classification.
FAO_CANOPY_THRESHOLD_PCT: float = 10.0

#: FAO minimum tree height at maturity (metres).
FAO_HEIGHT_THRESHOLD_M: float = 5.0

#: FAO minimum contiguous forest area (hectares).
FAO_AREA_THRESHOLD_HA: float = 0.5

#: Spectral angle threshold (degrees) for "close match".
SPECTRAL_ANGLE_CLOSE_DEG: float = 10.0

#: Spectral angle threshold (degrees) for "acceptable match".
SPECTRAL_ANGLE_ACCEPT_DEG: float = 20.0

#: Number of months in phenological profiles.
MONTHS_PER_YEAR: int = 12

#: EUDR-excluded commodities that are NOT forests per Article 2(4).
EUDR_COMMODITY_EXCLUSIONS: Dict[str, str] = {
    "oil_palm": (
        "Oil palm plantations are excluded from the EUDR forest "
        "definition per Article 2(4). Oil palm is an agricultural "
        "tree crop, not a forest."
    ),
    "rubber_monoculture": (
        "Rubber monoculture plantations are excluded from the EUDR "
        "forest definition per Article 2(4). Monoculture rubber "
        "is classified as agricultural land use."
    ),
}

# ---------------------------------------------------------------------------
# Reference Spectral Libraries (6 bands: Blue, Green, Red, NIR, SWIR1, SWIR2)
# ---------------------------------------------------------------------------
# Derived from USGS Spectral Library Version 7, Sentinel-2 land cover
# validation studies, and MODIS land cover type product (MCD12Q1).

SPECTRAL_LIBRARY: Dict[str, List[float]] = {
    ForestType.PRIMARY_TROPICAL.value: [0.020, 0.035, 0.025, 0.420, 0.170, 0.070],
    ForestType.SECONDARY_TROPICAL.value: [0.025, 0.040, 0.030, 0.380, 0.180, 0.085],
    ForestType.TROPICAL_DRY.value: [0.030, 0.050, 0.045, 0.340, 0.200, 0.110],
    ForestType.TEMPERATE_BROADLEAF.value: [0.025, 0.045, 0.035, 0.390, 0.190, 0.090],
    ForestType.TEMPERATE_CONIFEROUS.value: [0.020, 0.035, 0.028, 0.350, 0.160, 0.075],
    ForestType.BOREAL.value: [0.022, 0.038, 0.032, 0.320, 0.150, 0.070],
    ForestType.MANGROVE.value: [0.025, 0.040, 0.030, 0.350, 0.160, 0.080],
    ForestType.PLANTATION.value: [0.028, 0.045, 0.038, 0.360, 0.195, 0.095],
    ForestType.AGROFORESTRY.value: [0.035, 0.055, 0.050, 0.310, 0.210, 0.120],
    ForestType.NON_FOREST.value: [0.080, 0.100, 0.120, 0.180, 0.250, 0.220],
}

#: Number of spectral bands in the reference library.
NUM_SPECTRAL_BANDS: int = 6

# ---------------------------------------------------------------------------
# Phenological Templates (12 monthly NDVI values)
# ---------------------------------------------------------------------------
# Each template represents the characteristic annual NDVI curve shape.
# Values are peak-normalized NDVI (0.0 to 1.0 scale relative to biome max).
# Source: MODIS 16-day NDVI composites (MOD13Q1) averaged 2015-2020.

PHENOLOGICAL_TEMPLATES: Dict[str, List[float]] = {
    ForestType.PRIMARY_TROPICAL.value: [
        0.92, 0.93, 0.94, 0.94, 0.93, 0.92,
        0.91, 0.90, 0.91, 0.92, 0.92, 0.92,
    ],
    ForestType.SECONDARY_TROPICAL.value: [
        0.85, 0.87, 0.88, 0.88, 0.87, 0.85,
        0.83, 0.82, 0.83, 0.85, 0.85, 0.85,
    ],
    ForestType.TROPICAL_DRY.value: [
        0.45, 0.50, 0.60, 0.75, 0.85, 0.90,
        0.88, 0.80, 0.70, 0.55, 0.48, 0.45,
    ],
    ForestType.TEMPERATE_BROADLEAF.value: [
        0.25, 0.30, 0.45, 0.70, 0.85, 0.90,
        0.88, 0.80, 0.60, 0.40, 0.30, 0.25,
    ],
    ForestType.TEMPERATE_CONIFEROUS.value: [
        0.55, 0.58, 0.62, 0.70, 0.78, 0.82,
        0.82, 0.80, 0.75, 0.68, 0.60, 0.55,
    ],
    ForestType.BOREAL.value: [
        0.20, 0.22, 0.30, 0.50, 0.70, 0.80,
        0.82, 0.78, 0.60, 0.40, 0.25, 0.20,
    ],
    ForestType.MANGROVE.value: [
        0.78, 0.80, 0.82, 0.83, 0.82, 0.80,
        0.78, 0.77, 0.78, 0.79, 0.79, 0.78,
    ],
    ForestType.PLANTATION.value: [
        0.60, 0.65, 0.72, 0.78, 0.82, 0.85,
        0.84, 0.80, 0.75, 0.68, 0.63, 0.60,
    ],
    ForestType.AGROFORESTRY.value: [
        0.35, 0.40, 0.55, 0.70, 0.78, 0.82,
        0.80, 0.72, 0.60, 0.48, 0.40, 0.35,
    ],
    ForestType.NON_FOREST.value: [
        0.15, 0.18, 0.25, 0.40, 0.55, 0.60,
        0.58, 0.50, 0.38, 0.25, 0.18, 0.15,
    ],
}

# ---------------------------------------------------------------------------
# Structural Parameters per Forest Type
# ---------------------------------------------------------------------------
# Each type has expected ranges for:
#   canopy_height_m: (min, max) expected canopy height
#   canopy_density_pct: (min, max) expected canopy density
#   texture_complexity: (min, max) GLCM-like complexity score [0, 1]
#   height_uniformity: (min, max) coefficient of variation of height

STRUCTURAL_PARAMETERS: Dict[str, Dict[str, Tuple[float, float]]] = {
    ForestType.PRIMARY_TROPICAL.value: {
        "canopy_height_m": (25.0, 60.0),
        "canopy_density_pct": (75.0, 100.0),
        "texture_complexity": (0.6, 1.0),
        "height_uniformity": (0.3, 0.8),
    },
    ForestType.SECONDARY_TROPICAL.value: {
        "canopy_height_m": (12.0, 30.0),
        "canopy_density_pct": (55.0, 85.0),
        "texture_complexity": (0.4, 0.7),
        "height_uniformity": (0.2, 0.5),
    },
    ForestType.TROPICAL_DRY.value: {
        "canopy_height_m": (8.0, 25.0),
        "canopy_density_pct": (30.0, 70.0),
        "texture_complexity": (0.3, 0.6),
        "height_uniformity": (0.2, 0.5),
    },
    ForestType.TEMPERATE_BROADLEAF.value: {
        "canopy_height_m": (15.0, 40.0),
        "canopy_density_pct": (60.0, 90.0),
        "texture_complexity": (0.4, 0.7),
        "height_uniformity": (0.2, 0.5),
    },
    ForestType.TEMPERATE_CONIFEROUS.value: {
        "canopy_height_m": (15.0, 50.0),
        "canopy_density_pct": (50.0, 85.0),
        "texture_complexity": (0.3, 0.6),
        "height_uniformity": (0.15, 0.4),
    },
    ForestType.BOREAL.value: {
        "canopy_height_m": (5.0, 25.0),
        "canopy_density_pct": (30.0, 70.0),
        "texture_complexity": (0.2, 0.5),
        "height_uniformity": (0.15, 0.4),
    },
    ForestType.MANGROVE.value: {
        "canopy_height_m": (3.0, 20.0),
        "canopy_density_pct": (50.0, 90.0),
        "texture_complexity": (0.3, 0.6),
        "height_uniformity": (0.2, 0.5),
    },
    ForestType.PLANTATION.value: {
        "canopy_height_m": (8.0, 35.0),
        "canopy_density_pct": (60.0, 95.0),
        "texture_complexity": (0.05, 0.25),
        "height_uniformity": (0.02, 0.15),
    },
    ForestType.AGROFORESTRY.value: {
        "canopy_height_m": (3.0, 20.0),
        "canopy_density_pct": (20.0, 60.0),
        "texture_complexity": (0.3, 0.7),
        "height_uniformity": (0.3, 0.7),
    },
    ForestType.NON_FOREST.value: {
        "canopy_height_m": (0.0, 5.0),
        "canopy_density_pct": (0.0, 10.0),
        "texture_complexity": (0.0, 0.3),
        "height_uniformity": (0.0, 1.0),
    },
}

# ---------------------------------------------------------------------------
# Default ensemble weights
# ---------------------------------------------------------------------------

DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    ClassificationMethod.SPECTRAL_SIGNATURE.value: 0.25,
    ClassificationMethod.PHENOLOGICAL.value: 0.25,
    ClassificationMethod.STRUCTURAL.value: 0.25,
    ClassificationMethod.MULTI_TEMPORAL.value: 0.25,
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ForestClassificationResult:
    """Result of forest type classification for a single plot.

    Attributes:
        result_id: Unique identifier for this result.
        plot_id: Identifier of the analyzed plot.
        primary_type: Most likely forest type.
        secondary_type: Second most likely forest type.
        primary_confidence: Confidence in the primary classification.
        secondary_confidence: Confidence in the secondary classification.
        is_forest_eudr: Whether the plot qualifies as forest per EUDR.
        method_used: Classification method employed.
        method_scores: Per-type scores from the classification method.
        inter_method_agreement: Agreement score across methods (if ensemble).
        spectral_angle_deg: Spectral angle to closest match (if applicable).
        phenological_correlation: Correlation to best phenological template.
        structural_match_score: Structural parameter match score.
        commodity_exclusion: Whether the plot is excluded as commodity
            plantation per EUDR Article 2(4).
        commodity_exclusion_reason: Reason for exclusion (if applicable).
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of classification.
        metadata: Additional contextual fields.
    """

    result_id: str = ""
    plot_id: str = ""
    primary_type: str = ForestType.NON_FOREST.value
    secondary_type: str = ForestType.NON_FOREST.value
    primary_confidence: float = 0.0
    secondary_confidence: float = 0.0
    is_forest_eudr: bool = False
    method_used: str = ""
    method_scores: Dict[str, float] = field(default_factory=dict)
    inter_method_agreement: float = 0.0
    spectral_angle_deg: float = 0.0
    phenological_correlation: float = 0.0
    structural_match_score: float = 0.0
    commodity_exclusion: bool = False
    commodity_exclusion_reason: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "result_id": self.result_id,
            "plot_id": self.plot_id,
            "primary_type": self.primary_type,
            "secondary_type": self.secondary_type,
            "primary_confidence": self.primary_confidence,
            "secondary_confidence": self.secondary_confidence,
            "is_forest_eudr": self.is_forest_eudr,
            "method_used": self.method_used,
            "method_scores": self.method_scores,
            "inter_method_agreement": self.inter_method_agreement,
            "spectral_angle_deg": self.spectral_angle_deg,
            "phenological_correlation": self.phenological_correlation,
            "structural_match_score": self.structural_match_score,
            "commodity_exclusion": self.commodity_exclusion,
            "commodity_exclusion_reason": self.commodity_exclusion_reason,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

@dataclass
class ClassificationInput:
    """Input data for forest type classification.

    Attributes:
        plot_id: Unique identifier for the plot.
        multi_band_reflectance: Mean reflectance per band (6 bands).
        ndvi_time_series: 12 monthly NDVI values (Jan-Dec).
        canopy_height_m: Mean canopy height in metres.
        canopy_density_pct: Canopy density percentage.
        texture_complexity: GLCM texture complexity score [0, 1].
        height_uniformity: Coefficient of variation of canopy height.
        dry_season_reflectance: Optional 6-band reflectance (dry season).
        wet_season_reflectance: Optional 6-band reflectance (wet season).
        area_ha: Plot area in hectares.
        commodity_type: Optional EUDR commodity (for exclusion check).
        latitude: Optional latitude for biome inference.
        longitude: Optional longitude for biome inference.
    """

    plot_id: str = ""
    multi_band_reflectance: List[float] = field(default_factory=list)
    ndvi_time_series: List[float] = field(default_factory=list)
    canopy_height_m: float = 0.0
    canopy_density_pct: float = 0.0
    texture_complexity: float = 0.0
    height_uniformity: float = 0.0
    dry_season_reflectance: List[float] = field(default_factory=list)
    wet_season_reflectance: List[float] = field(default_factory=list)
    area_ha: float = 1.0
    commodity_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# ---------------------------------------------------------------------------
# ForestTypeClassifier
# ---------------------------------------------------------------------------

class ForestTypeClassifier:
    """Production-grade forest type classifier for EUDR compliance.

    Classifies forest areas into 10 regulatory categories using
    deterministic spectral, phenological, structural, multi-temporal,
    and ensemble methods. All computations are zero-hallucination
    with full SHA-256 provenance tracking.

    Example::

        classifier = ForestTypeClassifier()
        input_data = ClassificationInput(
            plot_id="plot-001",
            multi_band_reflectance=[0.02, 0.04, 0.03, 0.42, 0.17, 0.07],
            ndvi_time_series=[0.92, 0.93, 0.94, 0.94, 0.93, 0.92,
                              0.91, 0.90, 0.91, 0.92, 0.92, 0.92],
            canopy_height_m=35.0,
            canopy_density_pct=85.0,
            texture_complexity=0.7,
            height_uniformity=0.5,
        )
        result = classifier.classify_plot(input_data)
        assert result.primary_type == ForestType.PRIMARY_TROPICAL.value
        assert result.provenance_hash != ""

    Attributes:
        ensemble_weights: Weights for ensemble voting across methods.
    """

    def __init__(
        self,
        config: Any = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the ForestTypeClassifier.

        Args:
            config: Optional configuration object.
            ensemble_weights: Optional custom weights for ensemble method.
                Keys must match ClassificationMethod values (excluding
                ENSEMBLE). Weights must sum to 1.0.
        """
        if ensemble_weights is not None:
            weight_sum = sum(ensemble_weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(
                    f"ensemble_weights must sum to 1.0, got {weight_sum:.4f}"
                )
            self.ensemble_weights = ensemble_weights
        else:
            self.ensemble_weights = dict(DEFAULT_ENSEMBLE_WEIGHTS)

        self.config = config

        logger.info(
            "ForestTypeClassifier initialized: ensemble_weights=%s, "
            "module_version=%s",
            self.ensemble_weights,
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Main Entry Points
    # ------------------------------------------------------------------

    def classify_plot(
        self,
        input_data: ClassificationInput,
        method: ClassificationMethod = ClassificationMethod.ENSEMBLE,
    ) -> ForestClassificationResult:
        """Classify a single plot into a forest type.

        Runs the selected classification method and returns a result
        with primary and secondary type, confidence scores, EUDR
        forest check, commodity exclusion check, and provenance hash.

        Args:
            input_data: Classification input data.
            method: Classification method to use.

        Returns:
            ForestClassificationResult with type, confidence, and
            provenance hash.

        Raises:
            ValueError: If plot_id is empty or required data is missing.
        """
        start_time = time.monotonic()

        self._validate_input(input_data, method)

        result_id = _generate_id()
        timestamp = utcnow().isoformat()

        # Dispatch to classification method
        scores, method_metadata = self._dispatch_classification(
            input_data, method,
        )

        # Extract primary and secondary types
        sorted_types = sorted(
            scores.items(), key=lambda x: x[1], reverse=True,
        )
        primary_type = sorted_types[0][0] if sorted_types else ForestType.NON_FOREST.value
        primary_conf = sorted_types[0][1] if sorted_types else 0.0
        secondary_type = sorted_types[1][0] if len(sorted_types) > 1 else ForestType.NON_FOREST.value
        secondary_conf = sorted_types[1][1] if len(sorted_types) > 1 else 0.0

        # Check EUDR forest status
        is_forest = self.is_forest_per_eudr(
            canopy_density_pct=input_data.canopy_density_pct,
            canopy_height_m=input_data.canopy_height_m,
            area_ha=input_data.area_ha,
            forest_type=primary_type,
        )

        # Check commodity exclusions
        exclusion, exclusion_reason = self.get_commodity_exclusions(
            commodity_type=input_data.commodity_type,
            forest_type=primary_type,
        )

        # If excluded as commodity, override forest status
        if exclusion:
            is_forest = False

        # Extract method-specific metrics
        spectral_angle = method_metadata.get("spectral_angle_deg", 0.0)
        pheno_corr = method_metadata.get("phenological_correlation", 0.0)
        struct_score = method_metadata.get("structural_match_score", 0.0)
        agreement = method_metadata.get("inter_method_agreement", 0.0)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = ForestClassificationResult(
            result_id=result_id,
            plot_id=input_data.plot_id,
            primary_type=primary_type,
            secondary_type=secondary_type,
            primary_confidence=round(primary_conf, 4),
            secondary_confidence=round(secondary_conf, 4),
            is_forest_eudr=is_forest,
            method_used=method.value,
            method_scores={k: round(v, 4) for k, v in scores.items()},
            inter_method_agreement=round(agreement, 4),
            spectral_angle_deg=round(spectral_angle, 4),
            phenological_correlation=round(pheno_corr, 4),
            structural_match_score=round(struct_score, 4),
            commodity_exclusion=exclusion,
            commodity_exclusion_reason=exclusion_reason,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=timestamp,
            metadata=method_metadata,
        )

        result.provenance_hash = self._compute_result_hash(result)

        logger.info(
            "Forest classified: plot=%s, primary=%s (%.2f), "
            "secondary=%s (%.2f), eudr_forest=%s, method=%s, %.2fms",
            input_data.plot_id,
            primary_type, primary_conf,
            secondary_type, secondary_conf,
            is_forest, method.value, elapsed_ms,
        )

        return result

    def batch_classify(
        self,
        inputs: List[ClassificationInput],
        method: ClassificationMethod = ClassificationMethod.ENSEMBLE,
    ) -> List[ForestClassificationResult]:
        """Classify multiple plots for forest type.

        Args:
            inputs: List of classification inputs.
            method: Classification method for all plots.

        Returns:
            List of ForestClassificationResult objects.

        Raises:
            ValueError: If inputs list is empty.
        """
        if not inputs:
            raise ValueError("inputs list must not be empty")

        start_time = time.monotonic()
        results: List[ForestClassificationResult] = []

        for i, input_data in enumerate(inputs):
            try:
                result = self.classify_plot(input_data, method=method)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "batch_classify: failed on plot[%d] id=%s: %s",
                    i, input_data.plot_id, str(exc),
                )
                error_result = self._create_error_result(
                    input_data, method, str(exc),
                )
                results.append(error_result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "batch_classify complete: %d plots, %.2fms total",
            len(inputs), elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: EUDR Forest Check
    # ------------------------------------------------------------------

    def is_forest_per_eudr(
        self,
        canopy_density_pct: float,
        canopy_height_m: float,
        area_ha: float,
        forest_type: str = "",
    ) -> bool:
        """Check if a plot qualifies as forest per EUDR Article 2(4).

        FAO/EUDR criteria:
            1. Canopy cover >= 10%
            2. Tree height potential >= 5m at maturity
            3. Contiguous area >= 0.5 hectares
            4. NOT an agricultural tree plantation (palm, rubber monoculture)

        Args:
            canopy_density_pct: Canopy cover percentage.
            canopy_height_m: Mean canopy height in metres.
            area_ha: Plot area in hectares.
            forest_type: Classified forest type (used for exclusion check).

        Returns:
            True if the plot meets EUDR forest criteria.
        """
        if canopy_density_pct < FAO_CANOPY_THRESHOLD_PCT:
            return False

        if canopy_height_m < FAO_HEIGHT_THRESHOLD_M:
            return False

        if area_ha < FAO_AREA_THRESHOLD_HA:
            return False

        # Exclude non-forest classification
        if forest_type == ForestType.NON_FOREST.value:
            return False

        return True

    # ------------------------------------------------------------------
    # Public API: Commodity Exclusions
    # ------------------------------------------------------------------

    def get_commodity_exclusions(
        self,
        commodity_type: Optional[str],
        forest_type: str = "",
    ) -> Tuple[bool, str]:
        """Check if a plot should be excluded per EUDR Article 2(4).

        Oil palm and rubber monoculture plantations are NOT considered
        forests under EUDR even if they meet canopy cover thresholds.

        Args:
            commodity_type: Type of commodity associated with the plot
                (e.g., "oil_palm", "rubber_monoculture").
            forest_type: Classified forest type for cross-reference.

        Returns:
            Tuple of (is_excluded, reason_string).
        """
        if commodity_type is None:
            return False, ""

        commodity_lower = commodity_type.lower().strip()
        if commodity_lower in EUDR_COMMODITY_EXCLUSIONS:
            return True, EUDR_COMMODITY_EXCLUSIONS[commodity_lower]

        return False, ""

    # ------------------------------------------------------------------
    # Public API: Classification Methods
    # ------------------------------------------------------------------

    def spectral_signature_classify(
        self,
        reflectance: List[float],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Classify forest type by comparing spectral signature to library.

        Computes both Euclidean distance and Spectral Angle Mapper (SAM)
        scores between the observed reflectance and each reference
        spectral signature. The final score combines both metrics.

        Spectral Angle Mapper:
            angle = arccos(dot(obs, ref) / (||obs|| * ||ref||))
            Smaller angle = closer match.

        Args:
            reflectance: 6-band mean reflectance values
                [Blue, Green, Red, NIR, SWIR1, SWIR2].

        Returns:
            Tuple of (type_scores, metadata) where type_scores maps
            ForestType names to similarity scores [0, 1].

        Raises:
            ValueError: If reflectance does not have 6 bands.
        """
        start_time = time.monotonic()

        if len(reflectance) != NUM_SPECTRAL_BANDS:
            raise ValueError(
                f"Expected {NUM_SPECTRAL_BANDS} bands, "
                f"got {len(reflectance)}"
            )

        obs_norm = self._vector_norm(reflectance)
        scores: Dict[str, float] = {}
        best_angle = 180.0
        best_distance = float("inf")

        for type_name, ref_spectrum in SPECTRAL_LIBRARY.items():
            # Euclidean distance
            distance = self._euclidean_distance(reflectance, ref_spectrum)

            # Spectral angle
            angle = self._spectral_angle(reflectance, ref_spectrum, obs_norm)

            # Convert to similarity score [0, 1]
            # Angle-based: 0 degrees = 1.0, 90 degrees = 0.0
            angle_score = max(0.0, 1.0 - (angle / 90.0))

            # Distance-based: normalize by max possible distance
            # Max distance for 6 bands with range [0, 0.5] is ~1.22
            dist_score = max(0.0, 1.0 - (distance / 1.22))

            # Combined score (60% angle, 40% distance)
            combined = 0.6 * angle_score + 0.4 * dist_score
            scores[type_name] = combined

            if angle < best_angle:
                best_angle = angle
            if distance < best_distance:
                best_distance = distance

        # Normalize scores to sum to 1.0
        scores = self._normalize_scores(scores)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": ClassificationMethod.SPECTRAL_SIGNATURE.value,
            "spectral_angle_deg": round(best_angle, 4),
            "best_distance": round(best_distance, 6),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        return scores, metadata

    def phenological_classify(
        self,
        ndvi_time_series: List[float],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Classify forest type using NDVI time-series profile matching.

        Computes Pearson correlation between the observed 12-month NDVI
        profile and each phenological template. Also considers the
        seasonality amplitude (max - min) to distinguish evergreen from
        deciduous types.

        Template matching rules:
            - Evergreen tropical: flat profile, high baseline (low amplitude)
            - Deciduous tropical: unimodal peak, moderate amplitude
            - Temperate deciduous: strong seasonal peak (high amplitude)
            - Boreal: short growing season, extreme amplitude

        Args:
            ndvi_time_series: 12 monthly NDVI values (January to December).

        Returns:
            Tuple of (type_scores, metadata).

        Raises:
            ValueError: If time series does not have 12 values.
        """
        start_time = time.monotonic()

        if len(ndvi_time_series) != MONTHS_PER_YEAR:
            raise ValueError(
                f"Expected {MONTHS_PER_YEAR} monthly values, "
                f"got {len(ndvi_time_series)}"
            )

        # Normalize observed values to [0, 1] range
        obs_min = min(ndvi_time_series)
        obs_max = max(ndvi_time_series)
        obs_range = obs_max - obs_min

        if obs_range < 1e-10:
            # Flat profile: all values nearly identical
            normalized_obs = [0.5] * MONTHS_PER_YEAR
        else:
            normalized_obs = [
                (v - obs_min) / obs_range for v in ndvi_time_series
            ]

        # Compute amplitude (seasonality indicator)
        amplitude = obs_range

        scores: Dict[str, float] = {}
        best_corr = -1.0

        for type_name, template in PHENOLOGICAL_TEMPLATES.items():
            # Pearson correlation between normalized obs and template
            correlation = self._pearson_correlation(
                normalized_obs, template,
            )

            # Amplitude penalty: penalize mismatch in seasonality
            template_range = max(template) - min(template)
            amplitude_diff = abs(amplitude - template_range)
            amplitude_penalty = min(0.3, amplitude_diff * 0.5)

            # Combined score
            score = max(0.0, (correlation + 1.0) / 2.0 - amplitude_penalty)
            scores[type_name] = score

            if correlation > best_corr:
                best_corr = correlation

        # Normalize scores
        scores = self._normalize_scores(scores)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": ClassificationMethod.PHENOLOGICAL.value,
            "phenological_correlation": round(best_corr, 4),
            "observed_amplitude": round(amplitude, 4),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        return scores, metadata

    def structural_classify(
        self,
        canopy_height_m: float,
        canopy_density_pct: float,
        texture_complexity: float,
        height_uniformity: float,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Classify forest type using structural canopy metrics.

        Uses a rule-based approach comparing observed structural
        parameters against expected ranges for each forest type.
        Key discriminators:
            - Primary forest: tall, dense, complex texture.
            - Plantation: uniform height and spacing (low complexity).
            - Agroforestry: mixed height, moderate density.

        Args:
            canopy_height_m: Mean canopy height in metres.
            canopy_density_pct: Canopy density percentage.
            texture_complexity: GLCM texture complexity [0, 1].
            height_uniformity: Coefficient of variation of height.

        Returns:
            Tuple of (type_scores, metadata).
        """
        start_time = time.monotonic()

        observed = {
            "canopy_height_m": canopy_height_m,
            "canopy_density_pct": canopy_density_pct,
            "texture_complexity": texture_complexity,
            "height_uniformity": height_uniformity,
        }

        scores: Dict[str, float] = {}
        best_score = 0.0

        for type_name, params in STRUCTURAL_PARAMETERS.items():
            score = self._structural_match_score(observed, params)
            scores[type_name] = score
            if score > best_score:
                best_score = score

        # Normalize scores
        scores = self._normalize_scores(scores)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": ClassificationMethod.STRUCTURAL.value,
            "structural_match_score": round(best_score, 4),
            "observed_metrics": {
                k: round(v, 4) for k, v in observed.items()
            },
            "processing_time_ms": round(elapsed_ms, 2),
        }

        return scores, metadata

    def multi_temporal_classify(
        self,
        dry_season_reflectance: List[float],
        wet_season_reflectance: List[float],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Classify using combined dry and wet season spectral data.

        Improved discrimination of deciduous vs evergreen types by
        analyzing spectral change between seasons. Deciduous forests
        show large reflectance changes; evergreen forests remain stable.

        Args:
            dry_season_reflectance: 6-band reflectance from dry season.
            wet_season_reflectance: 6-band reflectance from wet season.

        Returns:
            Tuple of (type_scores, metadata).

        Raises:
            ValueError: If either reflectance array has incorrect length.
        """
        start_time = time.monotonic()

        if len(dry_season_reflectance) != NUM_SPECTRAL_BANDS:
            raise ValueError(
                f"dry_season_reflectance must have {NUM_SPECTRAL_BANDS} "
                f"bands, got {len(dry_season_reflectance)}"
            )
        if len(wet_season_reflectance) != NUM_SPECTRAL_BANDS:
            raise ValueError(
                f"wet_season_reflectance must have {NUM_SPECTRAL_BANDS} "
                f"bands, got {len(wet_season_reflectance)}"
            )

        # Compute spectral change magnitude per band
        band_changes: List[float] = []
        for b in range(NUM_SPECTRAL_BANDS):
            change = abs(
                wet_season_reflectance[b] - dry_season_reflectance[b]
            )
            band_changes.append(change)

        # Overall spectral change magnitude
        total_change = sum(band_changes) / NUM_SPECTRAL_BANDS

        # Compute mean reflectance of both seasons combined
        mean_reflectance = [
            (dry_season_reflectance[b] + wet_season_reflectance[b]) / 2.0
            for b in range(NUM_SPECTRAL_BANDS)
        ]

        # Classify the mean spectrum against the library
        spectral_scores, _ = self.spectral_signature_classify(
            mean_reflectance,
        )

        # Adjust scores based on seasonal change pattern
        adjusted_scores = self._adjust_for_seasonality(
            spectral_scores, total_change, band_changes,
        )

        # Normalize
        adjusted_scores = self._normalize_scores(adjusted_scores)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": ClassificationMethod.MULTI_TEMPORAL.value,
            "spectral_change_magnitude": round(total_change, 6),
            "band_changes": [round(c, 6) for c in band_changes],
            "processing_time_ms": round(elapsed_ms, 2),
        }

        return adjusted_scores, metadata

    def ensemble_classify(
        self,
        input_data: ClassificationInput,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Classify using weighted vote of all 4 methods.

        Runs each available method and combines their scores using
        configurable weights (default equal). Methods that cannot run
        (due to missing data) are skipped and their weight is
        redistributed.

        Args:
            input_data: Complete classification input data.

        Returns:
            Tuple of (type_scores, metadata) with agreement score.
        """
        start_time = time.monotonic()

        method_results: Dict[str, Dict[str, float]] = {}
        method_meta: Dict[str, Any] = {}
        active_weights: Dict[str, float] = {}

        # Run spectral classification if reflectance available
        if (
            len(input_data.multi_band_reflectance) == NUM_SPECTRAL_BANDS
        ):
            scores, meta = self.spectral_signature_classify(
                input_data.multi_band_reflectance,
            )
            method_results[ClassificationMethod.SPECTRAL_SIGNATURE.value] = scores
            method_meta["spectral"] = meta
            active_weights[ClassificationMethod.SPECTRAL_SIGNATURE.value] = (
                self.ensemble_weights.get(
                    ClassificationMethod.SPECTRAL_SIGNATURE.value, 0.25,
                )
            )

        # Run phenological classification if time series available
        if len(input_data.ndvi_time_series) == MONTHS_PER_YEAR:
            scores, meta = self.phenological_classify(
                input_data.ndvi_time_series,
            )
            method_results[ClassificationMethod.PHENOLOGICAL.value] = scores
            method_meta["phenological"] = meta
            active_weights[ClassificationMethod.PHENOLOGICAL.value] = (
                self.ensemble_weights.get(
                    ClassificationMethod.PHENOLOGICAL.value, 0.25,
                )
            )

        # Run structural classification if height/density available
        if input_data.canopy_height_m > 0 or input_data.canopy_density_pct > 0:
            scores, meta = self.structural_classify(
                canopy_height_m=input_data.canopy_height_m,
                canopy_density_pct=input_data.canopy_density_pct,
                texture_complexity=input_data.texture_complexity,
                height_uniformity=input_data.height_uniformity,
            )
            method_results[ClassificationMethod.STRUCTURAL.value] = scores
            method_meta["structural"] = meta
            active_weights[ClassificationMethod.STRUCTURAL.value] = (
                self.ensemble_weights.get(
                    ClassificationMethod.STRUCTURAL.value, 0.25,
                )
            )

        # Run multi-temporal if both season data available
        if (
            len(input_data.dry_season_reflectance) == NUM_SPECTRAL_BANDS
            and len(input_data.wet_season_reflectance) == NUM_SPECTRAL_BANDS
        ):
            scores, meta = self.multi_temporal_classify(
                input_data.dry_season_reflectance,
                input_data.wet_season_reflectance,
            )
            method_results[ClassificationMethod.MULTI_TEMPORAL.value] = scores
            method_meta["multi_temporal"] = meta
            active_weights[ClassificationMethod.MULTI_TEMPORAL.value] = (
                self.ensemble_weights.get(
                    ClassificationMethod.MULTI_TEMPORAL.value, 0.25,
                )
            )

        if not method_results:
            # No methods could run: return NON_FOREST with zero confidence
            empty_scores = {ft.value: 0.0 for ft in ForestType}
            empty_scores[ForestType.NON_FOREST.value] = 1.0
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return empty_scores, {
                "method": ClassificationMethod.ENSEMBLE.value,
                "methods_used": 0,
                "inter_method_agreement": 0.0,
                "processing_time_ms": round(elapsed_ms, 2),
            }

        # Normalize active weights to sum to 1.0
        weight_sum = sum(active_weights.values())
        if weight_sum > 1e-10:
            active_weights = {
                k: v / weight_sum for k, v in active_weights.items()
            }

        # Weighted combination of scores
        combined_scores: Dict[str, float] = {ft.value: 0.0 for ft in ForestType}
        for method_name, scores in method_results.items():
            weight = active_weights.get(method_name, 0.0)
            for type_name, score in scores.items():
                combined_scores[type_name] = (
                    combined_scores.get(type_name, 0.0) + weight * score
                )

        # Compute inter-method agreement
        agreement = self._compute_agreement(method_results)

        # Normalize combined scores
        combined_scores = self._normalize_scores(combined_scores)

        # Propagate best metrics from sub-methods
        best_spectral_angle = method_meta.get("spectral", {}).get(
            "spectral_angle_deg", 0.0,
        )
        best_pheno_corr = method_meta.get("phenological", {}).get(
            "phenological_correlation", 0.0,
        )
        best_struct_score = method_meta.get("structural", {}).get(
            "structural_match_score", 0.0,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": ClassificationMethod.ENSEMBLE.value,
            "methods_used": len(method_results),
            "active_weights": {
                k: round(v, 4) for k, v in active_weights.items()
            },
            "inter_method_agreement": round(agreement, 4),
            "spectral_angle_deg": best_spectral_angle,
            "phenological_correlation": best_pheno_corr,
            "structural_match_score": best_struct_score,
            "processing_time_ms": round(elapsed_ms, 2),
        }

        return combined_scores, metadata

    # ------------------------------------------------------------------
    # Internal: Method Dispatch
    # ------------------------------------------------------------------

    def _dispatch_classification(
        self,
        input_data: ClassificationInput,
        method: ClassificationMethod,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Dispatch to the selected classification method.

        Args:
            input_data: Classification input data.
            method: Selected classification method.

        Returns:
            Tuple of (type_scores, metadata).
        """
        if method == ClassificationMethod.SPECTRAL_SIGNATURE:
            return self.spectral_signature_classify(
                input_data.multi_band_reflectance,
            )
        elif method == ClassificationMethod.PHENOLOGICAL:
            return self.phenological_classify(
                input_data.ndvi_time_series,
            )
        elif method == ClassificationMethod.STRUCTURAL:
            return self.structural_classify(
                canopy_height_m=input_data.canopy_height_m,
                canopy_density_pct=input_data.canopy_density_pct,
                texture_complexity=input_data.texture_complexity,
                height_uniformity=input_data.height_uniformity,
            )
        elif method == ClassificationMethod.MULTI_TEMPORAL:
            return self.multi_temporal_classify(
                input_data.dry_season_reflectance,
                input_data.wet_season_reflectance,
            )
        elif method == ClassificationMethod.ENSEMBLE:
            return self.ensemble_classify(input_data)
        else:
            raise ValueError(f"Unsupported method: {method}")

    # ------------------------------------------------------------------
    # Internal: Vector Math
    # ------------------------------------------------------------------

    def _vector_norm(self, vector: List[float]) -> float:
        """Compute the L2 norm (Euclidean length) of a vector.

        Args:
            vector: List of float values.

        Returns:
            L2 norm of the vector.
        """
        return math.sqrt(sum(v * v for v in vector))

    def _euclidean_distance(
        self,
        vec_a: List[float],
        vec_b: List[float],
    ) -> float:
        """Compute Euclidean distance between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector (same length as vec_a).

        Returns:
            Euclidean distance.
        """
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
        )

    def _spectral_angle(
        self,
        observed: List[float],
        reference: List[float],
        obs_norm: float = 0.0,
    ) -> float:
        """Compute spectral angle between observed and reference vectors.

        SAM = arccos(dot(obs, ref) / (||obs|| * ||ref||))

        Args:
            observed: Observed reflectance vector.
            reference: Reference spectral signature.
            obs_norm: Pre-computed norm of observed (0 = compute it).

        Returns:
            Spectral angle in degrees.
        """
        dot_product = sum(
            o * r for o, r in zip(observed, reference)
        )
        if obs_norm < 1e-10:
            obs_norm = self._vector_norm(observed)
        ref_norm = self._vector_norm(reference)

        denominator = obs_norm * ref_norm
        if denominator < 1e-10:
            return 90.0

        cos_angle = dot_product / denominator
        # Clamp to [-1, 1] for numerical stability
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    def _pearson_correlation(
        self,
        series_a: List[float],
        series_b: List[float],
    ) -> float:
        """Compute Pearson correlation coefficient between two series.

        Args:
            series_a: First time series.
            series_b: Second time series (same length).

        Returns:
            Pearson r in [-1, 1].
        """
        n = len(series_a)
        if n == 0 or n != len(series_b):
            return 0.0

        mean_a = sum(series_a) / n
        mean_b = sum(series_b) / n

        cov = sum(
            (series_a[i] - mean_a) * (series_b[i] - mean_b)
            for i in range(n)
        )
        var_a = sum((v - mean_a) ** 2 for v in series_a)
        var_b = sum((v - mean_b) ** 2 for v in series_b)

        denominator = math.sqrt(var_a * var_b)
        if denominator < 1e-10:
            return 0.0

        return cov / denominator

    # ------------------------------------------------------------------
    # Internal: Score Helpers
    # ------------------------------------------------------------------

    def _normalize_scores(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Normalize scores so they sum to 1.0.

        Args:
            scores: Raw scores per forest type.

        Returns:
            Normalized scores summing to 1.0.
        """
        total = sum(scores.values())
        if total < 1e-10:
            # Return uniform distribution
            n = len(scores)
            return {k: 1.0 / n for k in scores} if n > 0 else scores

        return {k: v / total for k, v in scores.items()}

    def _structural_match_score(
        self,
        observed: Dict[str, float],
        expected_ranges: Dict[str, Tuple[float, float]],
    ) -> float:
        """Compute how well observed metrics match expected ranges.

        For each metric, computes a score based on whether the value
        falls within, near, or far from the expected range.

        Args:
            observed: Observed structural metrics.
            expected_ranges: Expected (min, max) ranges per metric.

        Returns:
            Match score in [0, 1].
        """
        total_score = 0.0
        metric_count = 0

        for metric_name, (exp_min, exp_max) in expected_ranges.items():
            obs_val = observed.get(metric_name, 0.0)
            exp_range = exp_max - exp_min

            if exp_range < 1e-10:
                # Degenerate range: exact match check
                if abs(obs_val - exp_min) < 0.01:
                    total_score += 1.0
                else:
                    total_score += 0.0
            elif exp_min <= obs_val <= exp_max:
                # Within range: full score
                total_score += 1.0
            else:
                # Outside range: score decreases with distance
                if obs_val < exp_min:
                    distance = exp_min - obs_val
                else:
                    distance = obs_val - exp_max
                penalty = distance / max(exp_range, 1.0)
                total_score += max(0.0, 1.0 - penalty)

            metric_count += 1

        return total_score / metric_count if metric_count > 0 else 0.0

    def _adjust_for_seasonality(
        self,
        spectral_scores: Dict[str, float],
        total_change: float,
        band_changes: List[float],
    ) -> Dict[str, float]:
        """Adjust spectral scores based on seasonal change pattern.

        Low seasonal change (< 0.02) boosts evergreen types.
        High seasonal change (> 0.05) boosts deciduous types.

        Args:
            spectral_scores: Base scores from spectral classification.
            total_change: Mean spectral change across bands.
            band_changes: Per-band change magnitudes.

        Returns:
            Adjusted scores.
        """
        adjusted = dict(spectral_scores)

        # Evergreen types (low seasonality expected)
        evergreen_types = {
            ForestType.PRIMARY_TROPICAL.value,
            ForestType.SECONDARY_TROPICAL.value,
            ForestType.TEMPERATE_CONIFEROUS.value,
            ForestType.MANGROVE.value,
        }

        # Deciduous types (high seasonality expected)
        deciduous_types = {
            ForestType.TROPICAL_DRY.value,
            ForestType.TEMPERATE_BROADLEAF.value,
            ForestType.BOREAL.value,
            ForestType.AGROFORESTRY.value,
        }

        if total_change < 0.02:
            # Low change: boost evergreen
            for ft in evergreen_types:
                if ft in adjusted:
                    adjusted[ft] *= 1.3
            for ft in deciduous_types:
                if ft in adjusted:
                    adjusted[ft] *= 0.7
        elif total_change > 0.05:
            # High change: boost deciduous
            for ft in deciduous_types:
                if ft in adjusted:
                    adjusted[ft] *= 1.3
            for ft in evergreen_types:
                if ft in adjusted:
                    adjusted[ft] *= 0.7

        return adjusted

    def _compute_agreement(
        self,
        method_results: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute inter-method agreement score.

        Measures how much the different methods agree on the primary
        forest type. Score of 1.0 = all methods agree on the same
        primary type. Score near 0.0 = high disagreement.

        Args:
            method_results: Scores from each method.

        Returns:
            Agreement score in [0, 1].
        """
        if len(method_results) < 2:
            return 1.0

        # Get the primary type from each method
        primary_types: List[str] = []
        for scores in method_results.values():
            if scores:
                best = max(scores, key=scores.get)  # type: ignore[arg-type]
                primary_types.append(best)

        if not primary_types:
            return 0.0

        # Count the most common primary type
        type_counts: Dict[str, int] = {}
        for t in primary_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        max_count = max(type_counts.values())
        agreement = max_count / len(primary_types)

        return agreement

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        input_data: ClassificationInput,
        method: ClassificationMethod,
    ) -> None:
        """Validate classification input for the selected method.

        Args:
            input_data: Classification input data.
            method: Selected method.

        Raises:
            ValueError: If required data is missing.
        """
        if not input_data.plot_id:
            raise ValueError("plot_id must not be empty")

        if method == ClassificationMethod.SPECTRAL_SIGNATURE:
            if len(input_data.multi_band_reflectance) != NUM_SPECTRAL_BANDS:
                raise ValueError(
                    f"multi_band_reflectance must have {NUM_SPECTRAL_BANDS} "
                    f"bands for SPECTRAL_SIGNATURE method"
                )
        elif method == ClassificationMethod.PHENOLOGICAL:
            if len(input_data.ndvi_time_series) != MONTHS_PER_YEAR:
                raise ValueError(
                    f"ndvi_time_series must have {MONTHS_PER_YEAR} values "
                    f"for PHENOLOGICAL method"
                )
        elif method == ClassificationMethod.MULTI_TEMPORAL:
            if (
                len(input_data.dry_season_reflectance) != NUM_SPECTRAL_BANDS
                or len(input_data.wet_season_reflectance) != NUM_SPECTRAL_BANDS
            ):
                raise ValueError(
                    f"Both dry/wet season reflectance must have "
                    f"{NUM_SPECTRAL_BANDS} bands for MULTI_TEMPORAL method"
                )
        # STRUCTURAL and ENSEMBLE have no strict requirements

    # ------------------------------------------------------------------
    # Internal: Error Result
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        input_data: ClassificationInput,
        method: ClassificationMethod,
        error_msg: str,
    ) -> ForestClassificationResult:
        """Create an error result for a failed classification.

        Args:
            input_data: Input that caused the error.
            method: Method that was attempted.
            error_msg: Error message.

        Returns:
            ForestClassificationResult with zero confidence.
        """
        result = ForestClassificationResult(
            result_id=_generate_id(),
            plot_id=input_data.plot_id,
            primary_type=ForestType.NON_FOREST.value,
            secondary_type=ForestType.NON_FOREST.value,
            primary_confidence=0.0,
            secondary_confidence=0.0,
            is_forest_eudr=False,
            method_used=method.value,
            timestamp=utcnow().isoformat(),
            metadata={"error": error_msg},
        )
        result.provenance_hash = self._compute_result_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_result_hash(
        self,
        result: ForestClassificationResult,
    ) -> str:
        """Compute SHA-256 provenance hash for a classification result.

        Args:
            result: ForestClassificationResult to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "result_id": result.result_id,
            "plot_id": result.plot_id,
            "primary_type": result.primary_type,
            "secondary_type": result.secondary_type,
            "primary_confidence": result.primary_confidence,
            "secondary_confidence": result.secondary_confidence,
            "is_forest_eudr": result.is_forest_eudr,
            "method_used": result.method_used,
            "commodity_exclusion": result.commodity_exclusion,
            "timestamp": result.timestamp,
        }
        return _compute_hash(hash_data)

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "ForestType",
    "ClassificationMethod",
    # Constants
    "FAO_CANOPY_THRESHOLD_PCT",
    "FAO_HEIGHT_THRESHOLD_M",
    "FAO_AREA_THRESHOLD_HA",
    "EUDR_COMMODITY_EXCLUSIONS",
    "SPECTRAL_LIBRARY",
    "PHENOLOGICAL_TEMPLATES",
    "STRUCTURAL_PARAMETERS",
    "DEFAULT_ENSEMBLE_WEIGHTS",
    # Data classes
    "ForestClassificationResult",
    "ClassificationInput",
    # Engine
    "ForestTypeClassifier",
]
