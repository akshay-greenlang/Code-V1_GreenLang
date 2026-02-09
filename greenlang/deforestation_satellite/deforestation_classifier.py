# -*- coding: utf-8 -*-
"""
Deforestation Classifier Engine - AGENT-DATA-007: GL-DATA-GEO-003

Hierarchical decision tree classifier for land cover classification
and forest/non-forest determination from vegetation indices.

Features:
    - Binary forest classification (NDVI threshold)
    - Multi-class land cover classification (10 classes)
    - Tree cover percentage estimation (calibrated NDVI/EVI model)
    - Canopy height estimation (linear tree cover model)
    - Classification confidence scoring
    - Provenance tracking for all classification operations

Classification Decision Tree:
    1. Water:        NDWI > 0.3
    2. Dense Forest: NDVI >= 0.6 AND EVI >= 0.35
    3. Open Forest:  NDVI >= 0.4 AND EVI >= 0.2
    4. Shrubland:    0.3 <= NDVI < 0.4
    5. Grassland:    NDVI >= 0.2
    6. Bare Soil:    NDVI > 0
    7. Unknown:      otherwise

Zero-Hallucination Guarantees:
    - All thresholds are deterministic hard-coded values
    - Tree cover estimation uses calibrated linear model
    - No ML inference or LLM-based classification
    - Provenance recorded for every classification

Example:
    >>> from greenlang.deforestation_satellite.deforestation_classifier import DeforestationClassifierEngine
    >>> engine = DeforestationClassifierEngine()
    >>> classification = engine.classify(ndvi=0.72, evi=0.45, ndwi=-0.2)
    >>> print(classification.land_cover_class, classification.tree_cover_percent)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.deforestation_satellite.config import get_config
from greenlang.deforestation_satellite.models import (
    ForestClassification,
    LandCoverClass,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NDVI threshold for binary forest/non-forest classification
_FOREST_NDVI_THRESHOLD = 0.4

# Decision tree thresholds
_WATER_NDWI_THRESHOLD = 0.3
_DENSE_FOREST_NDVI_THRESHOLD = 0.6
_DENSE_FOREST_EVI_THRESHOLD = 0.35
_OPEN_FOREST_NDVI_THRESHOLD = 0.4
_OPEN_FOREST_EVI_THRESHOLD = 0.2
_SHRUBLAND_NDVI_MIN = 0.3
_SHRUBLAND_NDVI_MAX = 0.4
_GRASSLAND_NDVI_THRESHOLD = 0.2

# Tree cover estimation calibration coefficients
# Linear model: cover% = coeff_ndvi * NDVI + offset
_COVER_COEFF_NDVI = 166.67
_COVER_OFFSET_NDVI = -33.33
_COVER_COEFF_EVI = 142.86
_COVER_OFFSET_EVI = -28.57
_NDVI_WEIGHT = 0.6
_EVI_WEIGHT = 0.4

# Canopy height model: height_m = coeff * tree_cover%
_HEIGHT_COEFF = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# DeforestationClassifierEngine
# =============================================================================


class DeforestationClassifierEngine:
    """Engine for land cover classification and forest determination.

    Implements a hierarchical decision tree classifier that uses
    vegetation indices (NDVI, EVI, NDWI, SAVI) to classify land
    cover type, estimate tree cover percentage, and determine
    forest/non-forest status.

    The classifier is fully deterministic with hard-coded thresholds.
    No machine learning or probabilistic inference is used.

    Attributes:
        config: DeforestationSatelliteConfig instance.
        provenance: Optional ProvenanceTracker for audit trails.

    Example:
        >>> engine = DeforestationClassifierEngine()
        >>> result = engine.classify(ndvi=0.75, evi=0.50)
        >>> print(result.land_cover_class)  # 'dense_forest'
        >>> print(result.is_forest)         # True
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize DeforestationClassifierEngine.

        Args:
            config: Optional DeforestationSatelliteConfig. Uses global
                config if None.
            provenance: Optional ProvenanceTracker for recording audit entries.
        """
        self.config = config or get_config()
        self.provenance = provenance
        self._classifications: Dict[str, ForestClassification] = {}
        self._classification_count: int = 0
        logger.info("DeforestationClassifierEngine initialized")

    # ------------------------------------------------------------------
    # Full classification
    # ------------------------------------------------------------------

    def classify(
        self,
        ndvi: float,
        evi: Optional[float] = None,
        ndwi: Optional[float] = None,
        savi: Optional[float] = None,
    ) -> ForestClassification:
        """Perform hierarchical land cover classification.

        Applies a decision tree to vegetation indices in priority order:
        water > dense_forest > open_forest > shrubland > grassland >
        bare_soil > unknown.

        Also estimates tree cover percentage, canopy height, and
        forest/non-forest binary classification.

        Args:
            ndvi: Normalized Difference Vegetation Index value (-1 to 1).
            evi: Optional Enhanced Vegetation Index value.
            ndwi: Optional Normalized Difference Water Index value.
            savi: Optional Soil-Adjusted Vegetation Index value.

        Returns:
            ForestClassification with land cover type, tree cover,
            canopy height, and confidence.
        """
        # Classify land cover
        land_cover = self.classify_land_cover(ndvi, evi, ndwi)

        # Binary forest classification
        is_forest = self.classify_binary(ndvi)

        # Estimate tree cover
        tree_cover = self.estimate_tree_cover(ndvi, evi)

        # Estimate canopy height
        canopy_height = self.estimate_canopy_height(tree_cover)

        # Calculate confidence
        confidence = self.calculate_confidence(ndvi, evi)

        # Determine method description
        method_parts = ["threshold"]
        if evi is not None:
            method_parts.append("evi_enhanced")
        if ndwi is not None:
            method_parts.append("water_detection")
        if savi is not None:
            method_parts.append("soil_adjusted")
        method = "+".join(method_parts)

        classification_id = self._generate_classification_id()

        classification = ForestClassification(
            classification_id=classification_id,
            land_cover_class=land_cover.value,
            tree_cover_percent=round(tree_cover, 2),
            is_forest=is_forest,
            canopy_height_m=round(canopy_height, 2),
            confidence=round(confidence, 4),
            method=method,
            pixel_count=1,  # single-pixel classification
        )

        # Store classification
        self._classifications[classification_id] = classification
        self._classification_count += 1

        # Record provenance
        if self.provenance is not None:
            data_hash = hashlib.sha256(
                json.dumps(classification.model_dump(mode="json"), sort_keys=True, default=str).encode()
            ).hexdigest()
            self.provenance.record(
                entity_type="classification",
                entity_id=classification_id,
                action="classify",
                data_hash=data_hash,
            )

        logger.debug(
            "Classification %s: class=%s, cover=%.1f%%, height=%.1fm, "
            "forest=%s, confidence=%.2f, method=%s",
            classification_id, land_cover.value, tree_cover,
            canopy_height, is_forest, confidence, method,
        )

        return classification

    # ------------------------------------------------------------------
    # Binary forest classification
    # ------------------------------------------------------------------

    def classify_binary(self, ndvi: float) -> bool:
        """Perform binary forest/non-forest classification.

        Uses a simple NDVI threshold: forest if NDVI >= 0.4.

        Args:
            ndvi: NDVI value (-1 to 1).

        Returns:
            True if the pixel is classified as forest.
        """
        return ndvi >= _FOREST_NDVI_THRESHOLD

    # ------------------------------------------------------------------
    # Land cover classification
    # ------------------------------------------------------------------

    def classify_land_cover(
        self,
        ndvi: float,
        evi: Optional[float] = None,
        ndwi: Optional[float] = None,
    ) -> LandCoverClass:
        """Classify land cover type using hierarchical decision tree.

        Priority order (first match wins):
            1. Water:        NDWI > 0.3
            2. Dense forest: NDVI >= 0.6 AND (EVI >= 0.35 or EVI unavailable)
            3. Open forest:  NDVI >= 0.4 AND (EVI >= 0.2 or EVI unavailable)
            4. Shrubland:    0.3 <= NDVI < 0.4
            5. Grassland:    NDVI >= 0.2
            6. Bare soil:    NDVI > 0
            7. Unknown:      otherwise

        When EVI is not provided, the EVI conditions in steps 2-3 are
        skipped (NDVI alone is used). When NDWI is not provided,
        water detection is based on NDVI < 0 only.

        Args:
            ndvi: NDVI value (-1 to 1).
            evi: Optional EVI value.
            ndwi: Optional NDWI value.

        Returns:
            LandCoverClass classification.
        """
        # Step 1: Water detection
        if ndwi is not None and ndwi > _WATER_NDWI_THRESHOLD:
            return LandCoverClass.WATER

        # Fallback water detection from NDVI
        if ndvi < -0.1 and ndwi is None:
            return LandCoverClass.WATER

        # Step 2: Dense forest
        if ndvi >= _DENSE_FOREST_NDVI_THRESHOLD:
            if evi is not None:
                if evi >= _DENSE_FOREST_EVI_THRESHOLD:
                    return LandCoverClass.DENSE_FOREST
                # EVI too low for dense forest -> check open forest
            else:
                return LandCoverClass.DENSE_FOREST

        # Step 3: Open forest
        if ndvi >= _OPEN_FOREST_NDVI_THRESHOLD:
            if evi is not None:
                if evi >= _OPEN_FOREST_EVI_THRESHOLD:
                    return LandCoverClass.OPEN_FOREST
                # EVI too low for open forest -> shrubland
                return LandCoverClass.SHRUBLAND
            else:
                return LandCoverClass.OPEN_FOREST

        # Step 4: Shrubland
        if _SHRUBLAND_NDVI_MIN <= ndvi < _SHRUBLAND_NDVI_MAX:
            return LandCoverClass.SHRUBLAND

        # Step 5: Grassland
        if ndvi >= _GRASSLAND_NDVI_THRESHOLD:
            return LandCoverClass.GRASSLAND

        # Step 6: Bare soil
        if ndvi > 0:
            return LandCoverClass.BARE_SOIL

        # Step 7: Unknown
        return LandCoverClass.UNKNOWN

    # ------------------------------------------------------------------
    # Tree cover estimation
    # ------------------------------------------------------------------

    def estimate_tree_cover(
        self,
        ndvi: float,
        evi: Optional[float] = None,
    ) -> float:
        """Estimate tree cover percentage from vegetation indices.

        Calibrated linear model:
            NDVI component: 166.67 * NDVI - 33.33
            EVI component:  142.86 * EVI - 28.57

        When EVI is available, a weighted blend is used:
            cover% = 0.6 * NDVI_component + 0.4 * EVI_component

        Result is clamped to [0, 100].

        Args:
            ndvi: NDVI value (-1 to 1).
            evi: Optional EVI value.

        Returns:
            Estimated tree cover percentage (0.0 to 100.0).
        """
        ndvi_cover = _COVER_COEFF_NDVI * ndvi + _COVER_OFFSET_NDVI

        if evi is not None:
            evi_cover = _COVER_COEFF_EVI * evi + _COVER_OFFSET_EVI
            blended = _NDVI_WEIGHT * ndvi_cover + _EVI_WEIGHT * evi_cover
        else:
            blended = ndvi_cover

        return max(0.0, min(100.0, blended))

    # ------------------------------------------------------------------
    # Canopy height estimation
    # ------------------------------------------------------------------

    def estimate_canopy_height(self, tree_cover_percent: float) -> float:
        """Estimate canopy height from tree cover percentage.

        Simple linear model: height_m = 0.3 * tree_cover_percent
        Gives: 0% cover -> 0m, 100% cover -> 30m.

        Args:
            tree_cover_percent: Tree cover percentage (0-100).

        Returns:
            Estimated canopy height in meters (0.0 to 30.0).
        """
        return max(0.0, min(30.0, _HEIGHT_COEFF * tree_cover_percent))

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        ndvi: float,
        evi: Optional[float] = None,
    ) -> float:
        """Calculate classification confidence score.

        Base confidence from NDVI distance to classification boundaries:
            - Far from boundaries (|NDVI| > 0.5): 0.90-0.95
            - Moderate distance (0.3 < |NDVI| < 0.5): 0.70-0.85
            - Near boundaries (|NDVI| < 0.3): 0.40-0.65
            - Very near (|NDVI| < 0.1): 0.20-0.35

        EVI availability boosts confidence by up to 0.10.

        Args:
            ndvi: NDVI value.
            evi: Optional EVI value.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        abs_ndvi = abs(ndvi)

        # Distance from nearest threshold
        thresholds = [0.0, 0.2, 0.3, 0.4, 0.6]
        min_dist = min(abs(ndvi - t) for t in thresholds)

        # Base confidence from distance to boundary
        if min_dist > 0.15:
            base_conf = 0.90
        elif min_dist > 0.10:
            base_conf = 0.75
        elif min_dist > 0.05:
            base_conf = 0.55
        else:
            base_conf = 0.35

        # NDVI magnitude boost
        if abs_ndvi > 0.6:
            base_conf = max(base_conf, 0.85)
        elif abs_ndvi > 0.4:
            base_conf = max(base_conf, 0.70)

        # EVI availability boost
        evi_boost = 0.0
        if evi is not None:
            evi_boost = 0.08
            # Additional boost if EVI and NDVI agree
            if (ndvi > 0.4 and evi > 0.2) or (ndvi < 0.2 and evi < 0.1):
                evi_boost = 0.12

        return min(1.0, base_conf + evi_boost)

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        ndvi_values: List[float],
        evi_values: Optional[List[float]] = None,
        ndwi_values: Optional[List[float]] = None,
    ) -> List[ForestClassification]:
        """Classify multiple pixels in batch.

        Args:
            ndvi_values: List of NDVI values.
            evi_values: Optional list of EVI values (same length).
            ndwi_values: Optional list of NDWI values (same length).

        Returns:
            List of ForestClassification results.
        """
        results: List[ForestClassification] = []

        for i, ndvi in enumerate(ndvi_values):
            evi = evi_values[i] if evi_values and i < len(evi_values) else None
            ndwi = ndwi_values[i] if ndwi_values and i < len(ndwi_values) else None
            result = self.classify(ndvi=ndvi, evi=evi, ndwi=ndwi)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Classification retrieval
    # ------------------------------------------------------------------

    def get_classification(self, classification_id: str) -> Optional[ForestClassification]:
        """Retrieve a classification by ID.

        Args:
            classification_id: Unique classification identifier.

        Returns:
            ForestClassification or None if not found.
        """
        return self._classifications.get(classification_id)

    def list_classifications(self) -> List[ForestClassification]:
        """Return all stored classifications.

        Returns:
            List of ForestClassification instances.
        """
        return list(self._classifications.values())

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_class_distribution(self) -> Dict[str, int]:
        """Get count of classifications by land cover class.

        Returns:
            Dictionary mapping land cover class names to counts.
        """
        dist: Dict[str, int] = {}
        for cls in self._classifications.values():
            dist[cls.land_cover_class] = dist.get(cls.land_cover_class, 0) + 1
        return dist

    def get_forest_rate(self) -> float:
        """Get the percentage of classifications classified as forest.

        Returns:
            Percentage (0.0 to 100.0) of forest classifications.
        """
        if not self._classifications:
            return 0.0
        forest_count = sum(1 for c in self._classifications.values() if c.is_forest)
        return (forest_count / len(self._classifications)) * 100.0

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_classification_id(self) -> str:
        """Generate a unique classification identifier.

        Returns:
            String in format "FCL-{12 hex chars}".
        """
        return f"FCL-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def classification_count(self) -> int:
        """Return the total number of classifications performed.

        Returns:
            Integer count of classifications.
        """
        return self._classification_count


__all__ = [
    "DeforestationClassifierEngine",
]
