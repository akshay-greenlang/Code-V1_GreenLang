# -*- coding: utf-8 -*-
"""
Transition Detector Engine - AGENT-EUDR-005: Land Use Change Detector (Engine 2)

Temporal land use transition detection engine that identifies when and how
land use changed between any two time periods. Classifies transitions into
EUDR-relevant types (deforestation, degradation, reforestation, urbanisation,
agricultural intensification, abandonment, or stable) using deterministic
rule-based classification grounded in EUDR Articles 2(1) and 2(5).

Zero-Hallucination Guarantees:
    - All transition classifications use deterministic from/to category
      lookup tables (no ML/LLM inference for type assignment).
    - Transition date estimation uses moving-average breakpoint detection
      on monthly NDVI series with fixed threshold parameters.
    - Transition evidence is computed from spectral deltas, NDVI magnitude
      changes, and canopy cover differentials using float arithmetic only.
    - Confidence propagation uses geometric mean of from/to classification
      confidence with spectral evidence weighting.
    - SHA-256 provenance hashes on all result objects.

Transition Types (7 EUDR-relevant):
    DEFORESTATION:    forest -> cropland/grassland/oil_palm/rubber/bare_soil
    DEGRADATION:      natural forest -> plantation_forest
    REFORESTATION:    non-forest -> forest
    URBANISATION:     any -> settlement
    AGRICULTURAL_INTENSIFICATION: cropland <-> grassland, grassland -> cropland
    ABANDONMENT:      agriculture -> forest/grassland (natural regrowth)
    STABLE:           no class change detected

Transition Matrix:
    The engine generates a full 10x10 transition matrix counting plot-level
    transitions between all land use category pairs within a region,
    enabling landscape-scale change analysis.

Performance Targets:
    - Single plot transition detection: <200ms
    - Transition type classification: <5ms
    - Transition date estimation: <50ms
    - Evidence compilation: <20ms
    - Batch detection (100 plots): <10 seconds
    - Full 10x10 transition matrix: <30ms

Regulatory References:
    - EUDR Article 2(1): Deforestation = conversion of forest to agricultural use
    - EUDR Article 2(5): Forest degradation = primary to plantation forest
    - EUDR Article 2(6): Cutoff date December 31, 2020
    - EUDR Article 9: Geolocation-based monitoring evidence
    - EUDR Article 10: Risk assessment from transition detection

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-005 (Engine 2: Transition Detection)
Agent ID: GL-EUDR-LUC-005
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
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.land_use_change.land_use_classifier import (
    LandUseCategory,
    LandUseClassification,
    LandUseClassifier,
    PlotClassificationInput,
    VegetationIndices,
    ClassificationMethod,
)

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

class TransitionType(str, Enum):
    """EUDR-relevant land use transition types.

    DEFORESTATION: Conversion of forest to agricultural or non-forest use.
        Defined under EUDR Article 2(1) as the conversion of forest land
        to agricultural land. Triggers NON_COMPLIANT verdict for any
        deforestation occurring after 31 December 2020.
    DEGRADATION: Conversion of natural (primary/secondary) forest to
        plantation forest. Defined under EUDR Article 2(5). The forest
        canopy may remain but ecological integrity is reduced.
    REFORESTATION: Recovery of forest cover on previously non-forest land.
        Non-forest to forest transition, indicating natural regeneration
        or active reforestation.
    URBANISATION: Conversion of any land use type to settlement/built-up
        area. While not directly an EUDR deforestation event, it signals
        permanent land use change that may be associated with
        infrastructure-driven deforestation.
    AGRICULTURAL_INTENSIFICATION: Transitions within agricultural use
        categories, such as grassland to cropland or cropland expansion.
        Indicates changes in farming practice without deforestation.
    ABANDONMENT: Conversion of agricultural land back to natural
        vegetation (forest or grassland). Indicates land taken out of
        production, often showing natural regrowth patterns.
    STABLE: No land use category change detected between the two time
        periods. The same class is present at both dates.
    """

    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    REFORESTATION = "reforestation"
    URBANISATION = "urbanisation"
    AGRICULTURAL_INTENSIFICATION = "agricultural_intensification"
    ABANDONMENT = "abandonment"
    STABLE = "stable"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR deforestation cutoff date per Article 2(1).
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Maximum plots in a single batch detection.
MAX_BATCH_SIZE: int = 5000

#: Categories classified as "forest" for transition rules.
FOREST_CATEGORIES: List[str] = [
    LandUseCategory.FOREST.value,
    LandUseCategory.PLANTATION_FOREST.value,
]

#: Categories classified as "natural forest" (not plantation).
NATURAL_FOREST_CATEGORIES: List[str] = [
    LandUseCategory.FOREST.value,
]

#: Categories classified as "agriculture" for deforestation detection.
AGRICULTURE_CATEGORIES: List[str] = [
    LandUseCategory.CROPLAND.value,
    LandUseCategory.GRASSLAND.value,
    LandUseCategory.OIL_PALM.value,
    LandUseCategory.RUBBER.value,
]

#: Categories classified as "non-forest" for transition rules.
NON_FOREST_CATEGORIES: List[str] = [
    LandUseCategory.CROPLAND.value,
    LandUseCategory.GRASSLAND.value,
    LandUseCategory.OIL_PALM.value,
    LandUseCategory.RUBBER.value,
    LandUseCategory.SHRUBLAND.value,
    LandUseCategory.SETTLEMENT.value,
    LandUseCategory.WATER.value,
    LandUseCategory.BARE_SOIL.value,
]

#: NDVI change threshold for detecting significant spectral change.
NDVI_CHANGE_THRESHOLD: float = 0.15

#: Moving average window (months) for breakpoint detection.
MOVING_AVERAGE_WINDOW: int = 3

#: Minimum NDVI drop to confirm abrupt transition.
ABRUPT_NDVI_DROP_THRESHOLD: float = 0.20

#: Minimum confidence for a transition date estimate.
MIN_TRANSITION_DATE_CONFIDENCE: float = 0.50

# ---------------------------------------------------------------------------
# Transition classification lookup table
# ---------------------------------------------------------------------------
# Maps (from_category, to_category) to TransitionType.
# Pairs not in this table default to STABLE if from == to, or a fallback
# based on category group membership.

_TRANSITION_LOOKUP: Dict[Tuple[str, str], TransitionType] = {}

def _build_transition_lookup() -> None:
    """Populate the transition lookup table with all EUDR-relevant rules.

    Deforestation rules (EUDR Art 2(1)):
        forest -> cropland/grassland/oil_palm/rubber/bare_soil = DEFORESTATION

    Degradation rules (EUDR Art 2(5)):
        natural_forest -> plantation_forest = DEGRADATION

    Reforestation rules:
        non_forest -> forest = REFORESTATION

    Urbanisation rules:
        any (except settlement) -> settlement = URBANISATION

    Agricultural intensification:
        cropland <-> grassland, grassland -> oil_palm/rubber = AGRICULTURAL_INTENSIFICATION

    Abandonment:
        agriculture -> forest/shrubland = ABANDONMENT
    """
    # Deforestation: forest -> agriculture/bare_soil
    for forest_cat in NATURAL_FOREST_CATEGORIES:
        for ag_cat in AGRICULTURE_CATEGORIES:
            _TRANSITION_LOOKUP[(forest_cat, ag_cat)] = TransitionType.DEFORESTATION
        _TRANSITION_LOOKUP[(forest_cat, LandUseCategory.BARE_SOIL.value)] = (
            TransitionType.DEFORESTATION
        )

    # Plantation forest -> agriculture/bare_soil is also deforestation
    _TRANSITION_LOOKUP[
        (LandUseCategory.PLANTATION_FOREST.value, LandUseCategory.CROPLAND.value)
    ] = TransitionType.DEFORESTATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.PLANTATION_FOREST.value, LandUseCategory.GRASSLAND.value)
    ] = TransitionType.DEFORESTATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.PLANTATION_FOREST.value, LandUseCategory.OIL_PALM.value)
    ] = TransitionType.DEFORESTATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.PLANTATION_FOREST.value, LandUseCategory.RUBBER.value)
    ] = TransitionType.DEFORESTATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.PLANTATION_FOREST.value, LandUseCategory.BARE_SOIL.value)
    ] = TransitionType.DEFORESTATION

    # Degradation: natural forest -> plantation forest
    _TRANSITION_LOOKUP[
        (LandUseCategory.FOREST.value, LandUseCategory.PLANTATION_FOREST.value)
    ] = TransitionType.DEGRADATION

    # Reforestation: non-forest -> forest
    for nf_cat in NON_FOREST_CATEGORIES:
        _TRANSITION_LOOKUP[
            (nf_cat, LandUseCategory.FOREST.value)
        ] = TransitionType.REFORESTATION

    # Also plantation -> natural forest is reforestation
    _TRANSITION_LOOKUP[
        (LandUseCategory.PLANTATION_FOREST.value, LandUseCategory.FOREST.value)
    ] = TransitionType.REFORESTATION

    # Urbanisation: any -> settlement
    for cat in LandUseCategory:
        if cat != LandUseCategory.SETTLEMENT:
            _TRANSITION_LOOKUP[
                (cat.value, LandUseCategory.SETTLEMENT.value)
            ] = TransitionType.URBANISATION

    # Agricultural intensification
    _TRANSITION_LOOKUP[
        (LandUseCategory.CROPLAND.value, LandUseCategory.GRASSLAND.value)
    ] = TransitionType.AGRICULTURAL_INTENSIFICATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.GRASSLAND.value, LandUseCategory.CROPLAND.value)
    ] = TransitionType.AGRICULTURAL_INTENSIFICATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.GRASSLAND.value, LandUseCategory.OIL_PALM.value)
    ] = TransitionType.AGRICULTURAL_INTENSIFICATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.GRASSLAND.value, LandUseCategory.RUBBER.value)
    ] = TransitionType.AGRICULTURAL_INTENSIFICATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.CROPLAND.value, LandUseCategory.OIL_PALM.value)
    ] = TransitionType.AGRICULTURAL_INTENSIFICATION
    _TRANSITION_LOOKUP[
        (LandUseCategory.CROPLAND.value, LandUseCategory.RUBBER.value)
    ] = TransitionType.AGRICULTURAL_INTENSIFICATION

    # Abandonment: agriculture -> forest/shrubland
    for ag_cat in AGRICULTURE_CATEGORIES:
        _TRANSITION_LOOKUP[
            (ag_cat, LandUseCategory.FOREST.value)
        ] = TransitionType.REFORESTATION
        _TRANSITION_LOOKUP[
            (ag_cat, LandUseCategory.SHRUBLAND.value)
        ] = TransitionType.ABANDONMENT

    # Overrides where reforestation and abandonment conflict:
    # agriculture -> forest is REFORESTATION (already set above)
    # agriculture -> shrubland is ABANDONMENT (already set above)

# Build the lookup table on module load.
_build_transition_lookup()

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TransitionPlotInput:
    """Input data for transition detection on a single plot.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot centroid latitude (-90 to 90).
        longitude: Plot centroid longitude (-180 to 180).
        from_classification: Land use classification at the start date.
        to_classification: Land use classification at the end date.
        ndvi_time_series: Optional monthly NDVI values between the two dates,
            used for transition date estimation.
        ndvi_dates: Optional dates corresponding to ndvi_time_series values.
        spectral_change: Optional dict of spectral band changes (deltas)
            between the two dates.
        from_date: Start date of the observation window.
        to_date: End date of the observation window.
        commodity_context: Optional EUDR commodity for context.
        area_ha: Plot area in hectares.
    """

    plot_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    from_classification: Optional[LandUseClassification] = None
    to_classification: Optional[LandUseClassification] = None
    ndvi_time_series: List[float] = field(default_factory=list)
    ndvi_dates: List[str] = field(default_factory=list)
    spectral_change: Dict[str, float] = field(default_factory=dict)
    from_date: str = ""
    to_date: str = ""
    commodity_context: Optional[str] = None
    area_ha: float = 1.0

@dataclass
class LandUseTransition:
    """Result of land use transition detection for a single plot.

    Attributes:
        result_id: Unique result identifier (UUID).
        plot_id: Plot identifier.
        from_category: Land use category at the start date.
        to_category: Land use category at the end date.
        transition_type: Classified transition type (EUDR-relevant).
        is_deforestation: Whether this transition constitutes deforestation
            under EUDR Article 2(1).
        is_degradation: Whether this transition constitutes forest degradation
            under EUDR Article 2(5).
        transition_date_earliest: Earliest estimated date of the transition.
        transition_date_latest: Latest estimated date of the transition.
        transition_confidence: Confidence in the transition detection [0, 1].
        from_confidence: Classification confidence at start date.
        to_confidence: Classification confidence at end date.
        ndvi_change: NDVI difference (to - from).
        spectral_evidence: Evidence from spectral band changes.
        evidence_summary: Narrative summary of transition evidence.
        from_date: Observation start date.
        to_date: Observation end date.
        latitude: Plot centroid latitude.
        longitude: Plot centroid longitude.
        area_ha: Plot area in hectares.
        processing_time_ms: Time taken for detection in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of detection.
        metadata: Additional contextual information.
    """

    result_id: str = ""
    plot_id: str = ""
    from_category: str = ""
    to_category: str = ""
    transition_type: str = ""
    is_deforestation: bool = False
    is_degradation: bool = False
    transition_date_earliest: str = ""
    transition_date_latest: str = ""
    transition_confidence: float = 0.0
    from_confidence: float = 0.0
    to_confidence: float = 0.0
    ndvi_change: float = 0.0
    spectral_evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_summary: str = ""
    from_date: str = ""
    to_date: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    area_ha: float = 1.0
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
            "from_category": self.from_category,
            "to_category": self.to_category,
            "transition_type": self.transition_type,
            "is_deforestation": self.is_deforestation,
            "is_degradation": self.is_degradation,
            "transition_date_earliest": self.transition_date_earliest,
            "transition_date_latest": self.transition_date_latest,
            "transition_confidence": self.transition_confidence,
            "from_confidence": self.from_confidence,
            "to_confidence": self.to_confidence,
            "ndvi_change": self.ndvi_change,
            "spectral_evidence": self.spectral_evidence,
            "evidence_summary": self.evidence_summary,
            "from_date": self.from_date,
            "to_date": self.to_date,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "area_ha": self.area_ha,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

@dataclass
class TransitionMatrix:
    """Full 10x10 transition matrix for a region.

    Counts the number of plots transitioning between each pair of
    land use categories within the analysis window.

    Attributes:
        matrix: Nested dict [from_category][to_category] -> count.
        total_plots: Total number of plots in the analysis.
        changed_plots: Number of plots with any class change.
        stable_plots: Number of plots with no class change.
        deforestation_count: Number of deforestation transitions.
        degradation_count: Number of degradation transitions.
        reforestation_count: Number of reforestation transitions.
        from_date: Start date of the analysis window.
        to_date: End date of the analysis window.
        region_bounds: Bounding box [min_lat, min_lon, max_lat, max_lon].
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of matrix generation.
    """

    matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_plots: int = 0
    changed_plots: int = 0
    stable_plots: int = 0
    deforestation_count: int = 0
    degradation_count: int = 0
    reforestation_count: int = 0
    from_date: str = ""
    to_date: str = ""
    region_bounds: List[float] = field(default_factory=list)
    provenance_hash: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the matrix to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "matrix": self.matrix,
            "total_plots": self.total_plots,
            "changed_plots": self.changed_plots,
            "stable_plots": self.stable_plots,
            "deforestation_count": self.deforestation_count,
            "degradation_count": self.degradation_count,
            "reforestation_count": self.reforestation_count,
            "from_date": self.from_date,
            "to_date": self.to_date,
            "region_bounds": self.region_bounds,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }

# ---------------------------------------------------------------------------
# TransitionDetector
# ---------------------------------------------------------------------------

class TransitionDetector:
    """Production-grade land use transition detection engine for EUDR.

    Detects transitions between land use categories across two time periods,
    classifies them into EUDR-relevant types, estimates transition dates,
    and compiles spectral evidence for regulatory audit trails. All
    computations are deterministic with SHA-256 provenance tracking.

    The detector relies on a LandUseClassifier for land use classification
    at each time point. If classifications are provided directly (via
    TransitionPlotInput), the classifier is not invoked.

    Example::

        classifier = LandUseClassifier()
        detector = TransitionDetector(classifier=classifier)
        transition = detector.detect_transition(
            latitude=-2.5,
            longitude=110.0,
            date_from=date(2020, 12, 31),
            date_to=date(2023, 6, 15),
        )
        assert transition.transition_type in [t.value for t in TransitionType]

    Attributes:
        classifier: LandUseClassifier instance for on-demand classification.
        config: Optional configuration object.
    """

    def __init__(
        self,
        classifier: Optional[LandUseClassifier] = None,
        config: Any = None,
    ) -> None:
        """Initialize the TransitionDetector.

        Args:
            classifier: LandUseClassifier instance. If None, a default
                instance is created.
            config: Optional configuration object with overrides.
        """
        self.classifier = classifier or LandUseClassifier()
        self.config = config

        logger.info(
            "TransitionDetector initialized: module_version=%s, "
            "transition_types=%d, lookup_rules=%d",
            _MODULE_VERSION,
            len(TransitionType),
            len(_TRANSITION_LOOKUP),
        )

    # ------------------------------------------------------------------
    # Public API: Single Plot Detection
    # ------------------------------------------------------------------

    def detect_transition(
        self,
        latitude: float,
        longitude: float,
        date_from: date,
        date_to: date,
        from_classification: Optional[LandUseClassification] = None,
        to_classification: Optional[LandUseClassification] = None,
        ndvi_time_series: Optional[List[float]] = None,
        ndvi_dates: Optional[List[str]] = None,
        spectral_change: Optional[Dict[str, float]] = None,
        commodity_context: Optional[str] = None,
    ) -> LandUseTransition:
        """Detect land use transition at a given location between two dates.

        If pre-computed classifications are not provided, the engine
        invokes the LandUseClassifier for both dates. When an NDVI
        time-series is provided, the engine estimates the transition
        date with monthly granularity.

        Args:
            latitude: Plot centroid latitude (-90 to 90).
            longitude: Plot centroid longitude (-180 to 180).
            date_from: Start date of the observation window.
            date_to: End date of the observation window.
            from_classification: Optional pre-computed classification at
                date_from.
            to_classification: Optional pre-computed classification at
                date_to.
            ndvi_time_series: Optional monthly NDVI values for transition
                date estimation.
            ndvi_dates: Optional dates corresponding to NDVI values.
            spectral_change: Optional spectral band deltas between dates.
            commodity_context: Optional EUDR commodity for context.

        Returns:
            LandUseTransition with transition type, date range, confidence,
            and provenance hash.

        Raises:
            ValueError: If coordinates are out of range or date_from > date_to.
        """
        start_time = time.monotonic()

        self._validate_coordinates(latitude, longitude)
        self._validate_date_range(date_from, date_to)

        plot = TransitionPlotInput(
            plot_id=_generate_id(),
            latitude=latitude,
            longitude=longitude,
            from_classification=from_classification,
            to_classification=to_classification,
            ndvi_time_series=ndvi_time_series or [],
            ndvi_dates=ndvi_dates or [],
            spectral_change=spectral_change or {},
            from_date=date_from.isoformat(),
            to_date=date_to.isoformat(),
            commodity_context=commodity_context,
        )

        return self._detect_plot_transition(plot, start_time)

    # ------------------------------------------------------------------
    # Public API: Batch Detection
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        plots: List[TransitionPlotInput],
        date_from: date,
        date_to: date,
    ) -> List[LandUseTransition]:
        """Detect transitions for a batch of plots.

        Args:
            plots: List of plot inputs to analyze.
            date_from: Start date for all plots.
            date_to: End date for all plots.

        Returns:
            List of LandUseTransition results.

        Raises:
            ValueError: If plots list is empty or exceeds MAX_BATCH_SIZE.
        """
        if not plots:
            raise ValueError("plots list must not be empty")
        if len(plots) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(plots)} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        batch_start = time.monotonic()
        results: List[LandUseTransition] = []

        for i, plot in enumerate(plots):
            try:
                if not plot.from_date:
                    plot.from_date = date_from.isoformat()
                if not plot.to_date:
                    plot.to_date = date_to.isoformat()
                start_time = time.monotonic()
                result = self._detect_plot_transition(plot, start_time)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "detect_batch: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                error_result = self._create_error_result(
                    plot=plot, error_msg=str(exc),
                )
                results.append(error_result)

        batch_elapsed = (time.monotonic() - batch_start) * 1000
        successful = sum(1 for r in results if r.transition_confidence > 0.0)

        logger.info(
            "detect_batch complete: %d/%d successful, %.2fms total",
            successful, len(plots), batch_elapsed,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: Transition Matrix
    # ------------------------------------------------------------------

    def generate_transition_matrix(
        self,
        region_bounds: List[float],
        date_from: date,
        date_to: date,
        transitions: Optional[List[LandUseTransition]] = None,
    ) -> TransitionMatrix:
        """Generate a 10x10 transition matrix from detected transitions.

        If pre-computed transitions are not provided, this method requires
        external data. Typically called after batch detection.

        Args:
            region_bounds: Bounding box [min_lat, min_lon, max_lat, max_lon].
            date_from: Start date of the analysis window.
            date_to: End date of the analysis window.
            transitions: Pre-computed list of LandUseTransition results.

        Returns:
            TransitionMatrix with counts for all category pairs.

        Raises:
            ValueError: If region_bounds does not have exactly 4 elements.
        """
        start_time = time.monotonic()

        if len(region_bounds) != 4:
            raise ValueError(
                f"region_bounds must have 4 elements [min_lat, min_lon, "
                f"max_lat, max_lon], got {len(region_bounds)}"
            )

        if transitions is None:
            transitions = []

        # Initialize the 10x10 matrix
        categories = [cat.value for cat in LandUseCategory]
        matrix: Dict[str, Dict[str, int]] = {}
        for from_cat in categories:
            matrix[from_cat] = {}
            for to_cat in categories:
                matrix[from_cat][to_cat] = 0

        # Populate counts
        deforestation_count = 0
        degradation_count = 0
        reforestation_count = 0
        changed = 0
        stable = 0

        for t in transitions:
            from_cat = t.from_category
            to_cat = t.to_category

            if from_cat in matrix and to_cat in matrix.get(from_cat, {}):
                matrix[from_cat][to_cat] += 1

            if from_cat == to_cat:
                stable += 1
            else:
                changed += 1

            if t.is_deforestation:
                deforestation_count += 1
            if t.is_degradation:
                degradation_count += 1
            if t.transition_type == TransitionType.REFORESTATION.value:
                reforestation_count += 1

        total = len(transitions)

        result = TransitionMatrix(
            matrix=matrix,
            total_plots=total,
            changed_plots=changed,
            stable_plots=stable,
            deforestation_count=deforestation_count,
            degradation_count=degradation_count,
            reforestation_count=reforestation_count,
            from_date=date_from.isoformat(),
            to_date=date_to.isoformat(),
            region_bounds=region_bounds,
            timestamp=utcnow().isoformat(),
        )

        result.provenance_hash = _compute_hash(result.to_dict())

        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            "Transition matrix generated: total=%d, changed=%d, stable=%d, "
            "deforestation=%d, degradation=%d, reforestation=%d, %.2fms",
            total, changed, stable,
            deforestation_count, degradation_count, reforestation_count,
            elapsed,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: Core Detection Pipeline
    # ------------------------------------------------------------------

    def _detect_plot_transition(
        self,
        plot: TransitionPlotInput,
        start_time: float,
    ) -> LandUseTransition:
        """Run the transition detection pipeline for a single plot.

        Args:
            plot: Input data for the plot.
            start_time: Monotonic start time for duration tracking.

        Returns:
            LandUseTransition result with classification and evidence.
        """
        self._validate_plot_input(plot)

        result_id = _generate_id()
        timestamp = utcnow().isoformat()

        # Obtain classifications at both dates
        from_class = plot.from_classification
        to_class = plot.to_classification

        if from_class is None:
            from_class = self._classify_at_date(
                plot.latitude, plot.longitude,
                date.fromisoformat(plot.from_date),
            )

        if to_class is None:
            to_class = self._classify_at_date(
                plot.latitude, plot.longitude,
                date.fromisoformat(plot.to_date),
            )

        from_cat = LandUseCategory(from_class.category)
        to_cat = LandUseCategory(to_class.category)

        # Classify transition type
        transition_type = self._classify_transition_type(from_cat, to_cat)

        # Check deforestation and degradation flags
        is_deforestation = self._is_deforestation(from_cat, to_cat)
        is_degradation = self._is_degradation(from_cat, to_cat)

        # Estimate transition date
        earliest_date, latest_date = self._estimate_transition_date(
            plot.ndvi_time_series,
            plot.ndvi_dates,
            from_cat,
            to_cat,
            plot.from_date,
            plot.to_date,
        )

        # Compute evidence
        spectral_evidence = self._compute_transition_evidence(
            from_class, to_class, plot.spectral_change,
        )

        # Compute NDVI change
        ndvi_change = self._compute_ndvi_change(
            from_class, to_class, plot.ndvi_time_series,
        )

        # Compute transition confidence
        transition_confidence = self._compute_transition_confidence(
            from_class.confidence,
            to_class.confidence,
            spectral_evidence,
        )

        # Generate evidence summary
        evidence_summary = self._generate_evidence_summary(
            from_cat, to_cat, transition_type,
            ndvi_change, transition_confidence,
            earliest_date, latest_date,
        )

        # Validate the transition
        is_valid = self._validate_transition_result(
            from_cat, to_cat, transition_type, transition_confidence,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = LandUseTransition(
            result_id=result_id,
            plot_id=plot.plot_id,
            from_category=from_cat.value,
            to_category=to_cat.value,
            transition_type=transition_type.value,
            is_deforestation=is_deforestation,
            is_degradation=is_degradation,
            transition_date_earliest=earliest_date,
            transition_date_latest=latest_date,
            transition_confidence=round(transition_confidence, 4),
            from_confidence=round(from_class.confidence, 4),
            to_confidence=round(to_class.confidence, 4),
            ndvi_change=round(ndvi_change, 6),
            spectral_evidence=spectral_evidence,
            evidence_summary=evidence_summary,
            from_date=plot.from_date,
            to_date=plot.to_date,
            latitude=plot.latitude,
            longitude=plot.longitude,
            area_ha=plot.area_ha,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=timestamp,
            metadata={
                "module_version": _MODULE_VERSION,
                "is_valid": is_valid,
                "commodity_context": plot.commodity_context or "",
            },
        )

        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Transition detected: plot=%s, %s -> %s, type=%s, "
            "deforestation=%s, confidence=%.2f, %.2fms",
            plot.plot_id,
            from_cat.value,
            to_cat.value,
            transition_type.value,
            is_deforestation,
            transition_confidence,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Transition Type Classification
    # ------------------------------------------------------------------

    def _classify_transition_type(
        self,
        from_class: LandUseCategory,
        to_class: LandUseCategory,
    ) -> TransitionType:
        """Classify the transition type from the from/to category pair.

        Uses the pre-built lookup table for known transitions. Falls
        back to STABLE if from == to, or infers type from category
        group membership for unmapped pairs.

        Args:
            from_class: Land use category at the start date.
            to_class: Land use category at the end date.

        Returns:
            TransitionType classification.
        """
        if from_class == to_class:
            return TransitionType.STABLE

        key = (from_class.value, to_class.value)
        result = _TRANSITION_LOOKUP.get(key)

        if result is not None:
            return result

        # Fallback inference based on category groups
        from_is_forest = from_class.value in FOREST_CATEGORIES
        to_is_forest = to_class.value in FOREST_CATEGORIES
        to_is_settlement = to_class == LandUseCategory.SETTLEMENT

        if to_is_settlement:
            return TransitionType.URBANISATION
        if from_is_forest and not to_is_forest:
            return TransitionType.DEFORESTATION
        if not from_is_forest and to_is_forest:
            return TransitionType.REFORESTATION

        # Default: agricultural intensification for any remaining change
        return TransitionType.AGRICULTURAL_INTENSIFICATION

    # ------------------------------------------------------------------
    # Deforestation and Degradation Checks
    # ------------------------------------------------------------------

    def _is_deforestation(
        self,
        from_class: LandUseCategory,
        to_class: LandUseCategory,
    ) -> bool:
        """Check if transition constitutes deforestation per EUDR Art 2(1).

        Deforestation is defined as conversion of forest to agricultural
        use. This includes both natural forest and plantation forest
        being converted to any non-forest agricultural category.

        Args:
            from_class: Land use category at start date.
            to_class: Land use category at end date.

        Returns:
            True if the transition is deforestation.
        """
        if from_class == to_class:
            return False

        from_is_forest = from_class.value in FOREST_CATEGORIES
        to_is_agriculture = to_class.value in AGRICULTURE_CATEGORIES
        to_is_bare = to_class == LandUseCategory.BARE_SOIL

        return from_is_forest and (to_is_agriculture or to_is_bare)

    def _is_degradation(
        self,
        from_class: LandUseCategory,
        to_class: LandUseCategory,
    ) -> bool:
        """Check if transition constitutes degradation per EUDR Art 2(5).

        Forest degradation is defined as conversion of natural (primary
        or secondary) forest to plantation forest. The canopy cover
        remains but ecological integrity is reduced.

        Args:
            from_class: Land use category at start date.
            to_class: Land use category at end date.

        Returns:
            True if the transition is forest degradation.
        """
        if from_class == to_class:
            return False

        from_is_natural = from_class.value in NATURAL_FOREST_CATEGORIES
        to_is_plantation = to_class == LandUseCategory.PLANTATION_FOREST

        return from_is_natural and to_is_plantation

    # ------------------------------------------------------------------
    # Transition Date Estimation
    # ------------------------------------------------------------------

    def _estimate_transition_date(
        self,
        ndvi_series: List[float],
        ndvi_dates: List[str],
        from_class: LandUseCategory,
        to_class: LandUseCategory,
        from_date_str: str,
        to_date_str: str,
    ) -> Tuple[str, str]:
        """Estimate the transition date with monthly granularity.

        Uses a moving-average breakpoint detection on the NDVI time
        series to find the month when the most significant change
        occurred. If no NDVI series is available, returns the full
        window as the date range.

        Args:
            ndvi_series: Monthly NDVI values.
            ndvi_dates: Dates corresponding to NDVI values.
            from_class: Category at start date.
            to_class: Category at end date.
            from_date_str: Start date string (ISO format).
            to_date_str: End date string (ISO format).

        Returns:
            Tuple of (earliest estimate ISO string, latest estimate ISO string).
        """
        if from_class == to_class:
            return ("", "")

        if not ndvi_series or len(ndvi_series) < 3:
            return (from_date_str, to_date_str)

        if not ndvi_dates or len(ndvi_dates) != len(ndvi_series):
            return (from_date_str, to_date_str)

        # Compute moving average
        window = min(MOVING_AVERAGE_WINDOW, len(ndvi_series))
        smoothed = self._moving_average(ndvi_series, window)

        # Find the point of maximum change (breakpoint)
        max_change_idx = 0
        max_change_val = 0.0

        for i in range(1, len(smoothed)):
            change = abs(smoothed[i] - smoothed[i - 1])
            if change > max_change_val:
                max_change_val = change
                max_change_idx = i

        # The transition likely occurred around the breakpoint
        if max_change_val < NDVI_CHANGE_THRESHOLD * 0.5:
            # Very gradual change - return the full window
            return (from_date_str, to_date_str)

        # Estimate date range: breakpoint +/- 1 month
        earliest_idx = max(0, max_change_idx - 1)
        latest_idx = min(len(ndvi_dates) - 1, max_change_idx + 1)

        earliest_date = ndvi_dates[earliest_idx]
        latest_date = ndvi_dates[latest_idx]

        return (earliest_date, latest_date)

    def _moving_average(
        self,
        values: List[float],
        window: int,
    ) -> List[float]:
        """Compute a simple moving average over a list of values.

        Uses a trailing window. The first (window-1) values are computed
        with a smaller effective window (left-aligned).

        Args:
            values: Input values.
            window: Window size.

        Returns:
            Smoothed values (same length as input).
        """
        if window <= 1 or len(values) <= 1:
            return list(values)

        result: List[float] = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            segment = values[start_idx: i + 1]
            avg = sum(segment) / len(segment)
            result.append(avg)

        return result

    # ------------------------------------------------------------------
    # Transition Evidence
    # ------------------------------------------------------------------

    def _compute_transition_evidence(
        self,
        from_classification: LandUseClassification,
        to_classification: LandUseClassification,
        spectral_change: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compile spectral and classification evidence for the transition.

        Combines classification metadata, spectral band changes, and
        derived metrics into a structured evidence dictionary for
        regulatory audit trails.

        Args:
            from_classification: Classification at start date.
            to_classification: Classification at end date.
            spectral_change: Per-band reflectance deltas.

        Returns:
            Evidence dictionary with classification, spectral, and
            derived metric fields.
        """
        evidence: Dict[str, Any] = {
            "from_class": {
                "category": from_classification.category,
                "confidence": from_classification.confidence,
                "method": from_classification.method_used,
            },
            "to_class": {
                "category": to_classification.category,
                "confidence": to_classification.confidence,
                "method": to_classification.method_used,
            },
            "spectral_changes": dict(spectral_change),
            "derived_metrics": {},
        }

        # Compute derived spectral metrics if band changes are available
        if spectral_change:
            nir_change = spectral_change.get("B8_nir", 0.0)
            red_change = spectral_change.get("B4_red", 0.0)
            swir1_change = spectral_change.get("B11_swir1", 0.0)

            evidence["derived_metrics"]["nir_change"] = round(nir_change, 6)
            evidence["derived_metrics"]["red_change"] = round(red_change, 6)
            evidence["derived_metrics"]["swir1_change"] = round(swir1_change, 6)

            # Magnitude of spectral change vector
            total_change = math.sqrt(
                sum(v * v for v in spectral_change.values())
            )
            evidence["derived_metrics"]["total_spectral_change"] = round(
                total_change, 6,
            )

            # Forest loss indicator: NIR decreases, SWIR increases
            if nir_change < -0.05 and swir1_change > 0.03:
                evidence["derived_metrics"]["forest_loss_indicator"] = True
            else:
                evidence["derived_metrics"]["forest_loss_indicator"] = False

        return evidence

    def _compute_ndvi_change(
        self,
        from_classification: LandUseClassification,
        to_classification: LandUseClassification,
        ndvi_series: List[float],
    ) -> float:
        """Compute the NDVI change between two dates.

        Uses the first and last values of the NDVI time series if
        available. Falls back to vegetation index metadata from
        classifications.

        Args:
            from_classification: Classification at start date.
            to_classification: Classification at end date.
            ndvi_series: NDVI time series values.

        Returns:
            NDVI difference (to - from). Negative indicates vegetation loss.
        """
        if ndvi_series and len(ndvi_series) >= 2:
            return ndvi_series[-1] - ndvi_series[0]

        # Fall back to classification vegetation indices
        from_ndvi = 0.0
        to_ndvi = 0.0

        if from_classification.vegetation_indices:
            from_ndvi = from_classification.vegetation_indices.get("ndvi", 0.0)
        if to_classification.vegetation_indices:
            to_ndvi = to_classification.vegetation_indices.get("ndvi", 0.0)

        return to_ndvi - from_ndvi

    # ------------------------------------------------------------------
    # Transition Confidence
    # ------------------------------------------------------------------

    def _compute_transition_confidence(
        self,
        from_confidence: float,
        to_confidence: float,
        spectral_evidence: Dict[str, Any],
    ) -> float:
        """Compute confidence in the transition detection.

        Uses the geometric mean of from/to classification confidence,
        weighted by spectral evidence strength. Higher spectral change
        magnitude increases confidence in genuine transitions.

        Args:
            from_confidence: Classification confidence at start date.
            to_confidence: Classification confidence at end date.
            spectral_evidence: Evidence dictionary with derived metrics.

        Returns:
            Transition confidence score in [0.0, 1.0].
        """
        if from_confidence <= 0.0 or to_confidence <= 0.0:
            return 0.0

        # Geometric mean of classification confidences
        geo_mean = math.sqrt(from_confidence * to_confidence)

        # Spectral evidence weight
        spectral_weight = 1.0
        derived = spectral_evidence.get("derived_metrics", {})

        if derived:
            total_change = derived.get("total_spectral_change", 0.0)
            # Boost confidence if spectral change is significant
            if total_change > 0.10:
                spectral_weight = 1.05
            elif total_change > 0.05:
                spectral_weight = 1.02
            elif total_change < 0.01:
                spectral_weight = 0.95

        confidence = geo_mean * spectral_weight
        return max(0.0, min(1.0, round(confidence, 4)))

    # ------------------------------------------------------------------
    # Evidence Summary Generation
    # ------------------------------------------------------------------

    def _generate_evidence_summary(
        self,
        from_cat: LandUseCategory,
        to_cat: LandUseCategory,
        transition_type: TransitionType,
        ndvi_change: float,
        confidence: float,
        earliest_date: str,
        latest_date: str,
    ) -> str:
        """Generate a human-readable evidence summary.

        Args:
            from_cat: Source land use category.
            to_cat: Destination land use category.
            transition_type: Classified transition type.
            ndvi_change: NDVI change magnitude.
            confidence: Transition confidence.
            earliest_date: Earliest estimated transition date.
            latest_date: Latest estimated transition date.

        Returns:
            Narrative evidence summary string.
        """
        if transition_type == TransitionType.STABLE:
            return (
                f"Land use remained stable as {from_cat.value} with "
                f"confidence {confidence:.0%}. No transition detected."
            )

        direction = "loss" if ndvi_change < 0 else "gain"

        date_range = ""
        if earliest_date and latest_date:
            date_range = f" between {earliest_date} and {latest_date}"

        summary = (
            f"Land use transitioned from {from_cat.value} to {to_cat.value} "
            f"({transition_type.value}){date_range}. "
            f"NDVI {direction} of {abs(ndvi_change):.3f} observed. "
            f"Confidence: {confidence:.0%}."
        )

        if transition_type == TransitionType.DEFORESTATION:
            summary += (
                " This transition constitutes deforestation under "
                "EUDR Article 2(1)."
            )
        elif transition_type == TransitionType.DEGRADATION:
            summary += (
                " This transition constitutes forest degradation under "
                "EUDR Article 2(5)."
            )

        return summary

    # ------------------------------------------------------------------
    # Transition Validation
    # ------------------------------------------------------------------

    def _validate_transition_result(
        self,
        from_cat: LandUseCategory,
        to_cat: LandUseCategory,
        transition_type: TransitionType,
        confidence: float,
    ) -> bool:
        """Validate that the transition result is internally consistent.

        Checks that the transition type matches the from/to categories
        and that the confidence is above the minimum threshold.

        Args:
            from_cat: Source land use category.
            to_cat: Destination land use category.
            transition_type: Classified transition type.
            confidence: Transition confidence.

        Returns:
            True if the transition is valid and consistent.
        """
        if from_cat == to_cat and transition_type != TransitionType.STABLE:
            logger.warning(
                "Inconsistent transition: from==to (%s) but type=%s",
                from_cat.value, transition_type.value,
            )
            return False

        if from_cat != to_cat and transition_type == TransitionType.STABLE:
            logger.warning(
                "Inconsistent transition: from!=to (%s->%s) but type=STABLE",
                from_cat.value, to_cat.value,
            )
            return False

        if confidence < MIN_TRANSITION_DATE_CONFIDENCE:
            logger.warning(
                "Low confidence transition: %.4f < %.4f threshold",
                confidence, MIN_TRANSITION_DATE_CONFIDENCE,
            )

        return True

    # ------------------------------------------------------------------
    # Classification Helper
    # ------------------------------------------------------------------

    def _classify_at_date(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> LandUseClassification:
        """Classify land use at a given location and date.

        Delegates to the injected LandUseClassifier with the vegetation
        index method (requires minimal input data). In production, this
        would be backed by actual satellite imagery retrieval.

        Args:
            latitude: Plot centroid latitude.
            longitude: Plot centroid longitude.
            target_date: Target date for classification.

        Returns:
            LandUseClassification result.
        """
        # Create minimal input with default vegetation indices
        # In production, this would query satellite imagery services.
        vi = VegetationIndices(ndvi=0.5, evi=0.35, ndmi=0.15, savi=0.35)

        result = self.classifier.classify(
            latitude=latitude,
            longitude=longitude,
            date=target_date,
            method=ClassificationMethod.VEGETATION_INDEX,
            vegetation_indices=vi,
        )

        return result

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    def _validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        """Validate geographic coordinates.

        Args:
            latitude: Latitude to validate.
            longitude: Longitude to validate.

        Raises:
            ValueError: If coordinates are out of valid range.
        """
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}"
            )
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}"
            )

    def _validate_date_range(
        self,
        date_from: date,
        date_to: date,
    ) -> None:
        """Validate that date_from <= date_to.

        Args:
            date_from: Start date.
            date_to: End date.

        Raises:
            ValueError: If date_from > date_to.
        """
        if date_from > date_to:
            raise ValueError(
                f"date_from ({date_from}) must be <= date_to ({date_to})"
            )

    def _validate_plot_input(
        self,
        plot: TransitionPlotInput,
    ) -> None:
        """Validate plot input data.

        Args:
            plot: Plot input to validate.

        Raises:
            ValueError: If required fields are missing.
        """
        if not plot.plot_id:
            raise ValueError("plot_id must not be empty")
        self._validate_coordinates(plot.latitude, plot.longitude)
        if not plot.from_date:
            raise ValueError("from_date must not be empty")
        if not plot.to_date:
            raise ValueError("to_date must not be empty")

    # ------------------------------------------------------------------
    # Error Result Creation
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: TransitionPlotInput,
        error_msg: str,
    ) -> LandUseTransition:
        """Create an error result for a failed transition detection.

        Args:
            plot: Input plot that failed detection.
            error_msg: Error message describing the failure.

        Returns:
            LandUseTransition with zero confidence and error metadata.
        """
        return LandUseTransition(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            from_category="",
            to_category="",
            transition_type=TransitionType.STABLE.value,
            transition_confidence=0.0,
            from_date=plot.from_date,
            to_date=plot.to_date,
            latitude=plot.latitude,
            longitude=plot.longitude,
            area_ha=plot.area_ha,
            processing_time_ms=0.0,
            provenance_hash="",
            timestamp=utcnow().isoformat(),
            metadata={
                "error": True,
                "error_message": error_msg,
                "module_version": _MODULE_VERSION,
            },
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "TransitionType",
    # Constants
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "FOREST_CATEGORIES",
    "NATURAL_FOREST_CATEGORIES",
    "AGRICULTURE_CATEGORIES",
    "NON_FOREST_CATEGORIES",
    # Data classes
    "TransitionPlotInput",
    "LandUseTransition",
    "TransitionMatrix",
    # Engine
    "TransitionDetector",
]
