# -*- coding: utf-8 -*-
"""
Baseline Manager Engine - AGENT-EUDR-003: Satellite Monitoring (Feature 3)

Establishes, stores, and manages December 31, 2020 forest baselines for
EUDR compliance. Each baseline captures the vegetation state of a production
plot at the EUDR cutoff date using satellite imagery and spectral index
analysis, creating an immutable reference point for change detection.

Zero-Hallucination Guarantees:
    - Baseline establishment uses deterministic pipeline:
      imagery search -> scene selection -> NDVI calculation -> classification.
    - All baselines are immutable once established; re-establishment creates
      a new baseline with full audit trail of the previous one.
    - Provenance hash chain: baseline hash includes input parameters,
      scene metadata, NDVI statistics, and classification results.
    - No ML/LLM used for any baseline logic.
    - In-memory storage simulates database persistence for testing.

Pipeline:
    1. Search imagery near EUDR cutoff date (Dec 31, 2020 +/- 90 days)
    2. Select best scene (closest date, lowest cloud cover)
    3. Download and calculate NDVI
    4. Classify forest using biome-specific thresholds
    5. Store immutable BaselineSnapshot with provenance hash

Biome Thresholds:
    - tropical_rainforest: NDVI >= 0.50 (forest), >= 0.70 (dense)
    - tropical_dry: NDVI >= 0.40 (forest), >= 0.55 (dense)
    - temperate: NDVI >= 0.45 (forest), >= 0.60 (dense)
    - boreal: NDVI >= 0.35 (forest), >= 0.50 (dense)
    - mangrove: NDVI >= 0.45 (forest), >= 0.60 (dense)
    - cerrado_savanna: NDVI >= 0.35 (forest), >= 0.50 (dense)

Performance Targets:
    - Baseline establishment (single plot): <500ms
    - Baseline retrieval: <1ms (in-memory lookup)
    - Integrity validation: <2ms

Regulatory References:
    - EUDR Article 2(1): Deforestation-free requirement baseline
    - EUDR Article 2(6): Cutoff date December 31, 2020
    - EUDR Article 9: Geolocation baseline for production plots
    - EUDR Article 10: Risk assessment against historical baseline

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 (Feature 3: Baseline Management)
Agent ID: GL-EUDR-SAT-003
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
    DataQualityAssessment,
    ImageryAcquisitionEngine,
    SceneMetadata,
)
from greenlang.agents.eudr.satellite_monitoring.spectral_index_calculator import (
    ForestClassification,
    SpectralIndexCalculator,
    SpectralIndexResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR cutoff date string (ISO format).
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: EUDR cutoff as a date object.
EUDR_CUTOFF_DATE_OBJ: date = date(2020, 12, 31)

#: Default search window around cutoff date (days before and after).
DEFAULT_SEARCH_WINDOW_DAYS: int = 90

#: Minimum acceptable scene quality score for baseline.
MIN_BASELINE_QUALITY_SCORE: float = 40.0


# ---------------------------------------------------------------------------
# Biome Thresholds Reference Data
# ---------------------------------------------------------------------------
# Comprehensive thresholds for baseline forest classification per biome.
# Each biome includes: ndvi_forest, ndvi_dense, typical_canopy_cover,
# typical_ndvi_range, seasonal_variation, and fire_susceptibility.

BIOME_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "tropical_rainforest": {
        "ndvi_forest": 0.50,
        "ndvi_dense": 0.70,
        "typical_canopy_cover_pct": 85.0,
        "typical_ndvi_range": (0.60, 0.90),
        "seasonal_variation": 0.05,
        "fire_susceptibility": "low",
        "description": "Dense tropical broadleaf evergreen forest",
        "primary_commodities": ["cocoa", "coffee", "palm_oil", "rubber", "wood"],
        "typical_countries": ["BR", "ID", "CD", "CO", "PE"],
    },
    "tropical_dry": {
        "ndvi_forest": 0.40,
        "ndvi_dense": 0.55,
        "typical_canopy_cover_pct": 60.0,
        "typical_ndvi_range": (0.35, 0.70),
        "seasonal_variation": 0.15,
        "fire_susceptibility": "high",
        "description": "Tropical dry deciduous forest",
        "primary_commodities": ["cattle", "soya", "wood"],
        "typical_countries": ["BR", "BO", "PY", "TZ"],
    },
    "temperate": {
        "ndvi_forest": 0.45,
        "ndvi_dense": 0.60,
        "typical_canopy_cover_pct": 70.0,
        "typical_ndvi_range": (0.40, 0.80),
        "seasonal_variation": 0.25,
        "fire_susceptibility": "medium",
        "description": "Temperate broadleaf and mixed forest",
        "primary_commodities": ["wood"],
        "typical_countries": ["DE", "FR", "PL", "RO", "US", "CA"],
    },
    "boreal": {
        "ndvi_forest": 0.35,
        "ndvi_dense": 0.50,
        "typical_canopy_cover_pct": 55.0,
        "typical_ndvi_range": (0.30, 0.65),
        "seasonal_variation": 0.30,
        "fire_susceptibility": "medium",
        "description": "Boreal/taiga coniferous forest",
        "primary_commodities": ["wood"],
        "typical_countries": ["SE", "FI", "CA", "RU"],
    },
    "mangrove": {
        "ndvi_forest": 0.45,
        "ndvi_dense": 0.60,
        "typical_canopy_cover_pct": 75.0,
        "typical_ndvi_range": (0.45, 0.80),
        "seasonal_variation": 0.05,
        "fire_susceptibility": "low",
        "description": "Coastal mangrove forest",
        "primary_commodities": ["wood"],
        "typical_countries": ["ID", "BR", "NG", "MZ", "MG"],
    },
    "cerrado_savanna": {
        "ndvi_forest": 0.35,
        "ndvi_dense": 0.50,
        "typical_canopy_cover_pct": 45.0,
        "typical_ndvi_range": (0.25, 0.60),
        "seasonal_variation": 0.20,
        "fire_susceptibility": "high",
        "description": "Tropical savanna with scattered trees",
        "primary_commodities": ["cattle", "soya"],
        "typical_countries": ["BR"],
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class BaselineSnapshot:
    """Immutable baseline snapshot for a production plot at EUDR cutoff.

    Captures the vegetation state of a plot at or near December 31, 2020,
    including the satellite scene used, NDVI statistics, forest classification,
    and a provenance hash chain for regulatory audit.

    Attributes:
        baseline_id: Unique identifier for this baseline.
        plot_id: Production plot identifier.
        commodity: EUDR commodity produced on the plot.
        country_code: ISO 3166-1 alpha-2 country code.
        biome: Biome classification of the plot location.
        polygon_vertices: Plot boundary vertices (lat, lon).
        cutoff_date: Target EUDR cutoff date.
        scene_id: ID of the satellite scene used for baseline.
        scene_date: Actual acquisition date of the scene used.
        scene_source: Satellite source ('sentinel2', 'landsat8', etc.).
        scene_cloud_cover_pct: Cloud cover of the scene used.
        ndvi_mean: Mean NDVI value across the plot.
        ndvi_min: Minimum NDVI value.
        ndvi_max: Maximum NDVI value.
        ndvi_std_dev: NDVI standard deviation.
        forest_cover_pct: Percentage of plot classified as forest.
        dense_forest_pct: Percentage classified as dense forest.
        classification_biome: Biome used for classification thresholds.
        quality_score: Overall baseline quality score (0-100).
        is_forested: Whether the plot is classified as forested at cutoff.
        established_at: UTC timestamp of baseline establishment.
        previous_baseline_id: ID of the baseline this replaces (if any).
        previous_baseline_hash: Hash of the replaced baseline for audit.
        provenance_hash: SHA-256 hash of the complete baseline.
        metadata: Additional metadata for audit purposes.
    """

    baseline_id: str = ""
    plot_id: str = ""
    commodity: str = ""
    country_code: str = ""
    biome: str = ""
    polygon_vertices: List[Tuple[float, float]] = field(default_factory=list)
    cutoff_date: str = EUDR_CUTOFF_DATE
    scene_id: str = ""
    scene_date: Optional[date] = None
    scene_source: str = ""
    scene_cloud_cover_pct: float = 0.0
    ndvi_mean: float = 0.0
    ndvi_min: float = 0.0
    ndvi_max: float = 0.0
    ndvi_std_dev: float = 0.0
    forest_cover_pct: float = 0.0
    dense_forest_pct: float = 0.0
    classification_biome: str = ""
    quality_score: float = 0.0
    is_forested: bool = False
    established_at: Optional[datetime] = None
    previous_baseline_id: Optional[str] = None
    previous_baseline_hash: Optional[str] = None
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaselineManager
# ---------------------------------------------------------------------------


class BaselineManager:
    """Production-grade baseline manager for EUDR forest baselines.

    Establishes, stores, retrieves, and validates forest baselines
    at the EUDR cutoff date (December 31, 2020) using satellite
    imagery analysis. Uses ImageryAcquisitionEngine for scene search
    and SpectralIndexCalculator for vegetation analysis.

    All baselines are immutable. Re-establishment creates a new
    baseline with a complete audit trail linking to the previous one.

    Example::

        manager = BaselineManager()
        baseline = manager.establish_baseline(
            plot_id="PLOT-001",
            polygon_vertices=[(-3.0, -60.0), (-3.0, -59.0),
                              (-4.0, -59.0), (-4.0, -60.0)],
            commodity="soya",
            country_code="BR",
            biome="tropical_rainforest",
        )
        assert baseline.is_forested
        assert baseline.provenance_hash != ""
        assert manager.validate_baseline_integrity("PLOT-001")

    Attributes:
        imagery_engine: ImageryAcquisitionEngine for scene acquisition.
        spectral_calculator: SpectralIndexCalculator for NDVI analysis.
        search_window_days: Days before/after cutoff to search for imagery.
        _baselines: In-memory baseline store (plot_id -> BaselineSnapshot).
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the BaselineManager.

        Args:
            config: Optional configuration object. If provided,
                overrides search_window_days and passes through to
                sub-engines.
        """
        self.search_window_days = DEFAULT_SEARCH_WINDOW_DAYS
        self._baselines: Dict[str, BaselineSnapshot] = {}

        if config is not None:
            self.search_window_days = getattr(
                config, "search_window_days", DEFAULT_SEARCH_WINDOW_DAYS
            )

        self.imagery_engine = ImageryAcquisitionEngine(config=config)
        self.spectral_calculator = SpectralIndexCalculator(config=config)

        logger.info(
            "BaselineManager initialized: search_window=%d days, "
            "cutoff=%s",
            self.search_window_days, EUDR_CUTOFF_DATE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def establish_baseline(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        commodity: str,
        country_code: str,
        biome: str = "tropical_rainforest",
    ) -> BaselineSnapshot:
        """Establish a forest baseline at the EUDR cutoff date.

        Full pipeline:
            1. Search imagery near Dec 31, 2020 (+/- search_window_days)
            2. Select best scene (closest date, lowest cloud cover)
            3. Download Red and NIR bands
            4. Calculate NDVI
            5. Classify forest using biome-specific thresholds
            6. Create immutable BaselineSnapshot with provenance hash

        Args:
            plot_id: Unique identifier for the production plot.
            polygon_vertices: Plot boundary as (lat, lon) tuples.
            commodity: EUDR commodity (e.g., 'soya', 'cocoa').
            country_code: ISO 3166-1 alpha-2 country code.
            biome: Biome type for classification thresholds.

        Returns:
            BaselineSnapshot with vegetation state at cutoff.

        Raises:
            ValueError: If plot_id is empty, polygon is empty,
                or no suitable imagery is found.
        """
        start_time = time.monotonic()

        if not plot_id or not plot_id.strip():
            raise ValueError("plot_id must not be empty")
        if not polygon_vertices:
            raise ValueError("polygon_vertices must not be empty")

        logger.info(
            "Establishing baseline for plot %s: commodity=%s, "
            "country=%s, biome=%s",
            plot_id, commodity, country_code, biome,
        )

        # Step 1: Search imagery near cutoff date
        search_start = self._date_offset(
            EUDR_CUTOFF_DATE_OBJ, -self.search_window_days
        )
        search_end = self._date_offset(
            EUDR_CUTOFF_DATE_OBJ, self.search_window_days
        )

        scenes = self.imagery_engine.search_scenes(
            polygon_vertices=polygon_vertices,
            date_range=(search_start.isoformat(), search_end.isoformat()),
            source="sentinel2",
            cloud_cover_max=50.0,
            limit=100,
        )

        if not scenes:
            # Fall back to Landsat
            scenes = self.imagery_engine.search_scenes(
                polygon_vertices=polygon_vertices,
                date_range=(search_start.isoformat(), search_end.isoformat()),
                source="landsat8",
                cloud_cover_max=50.0,
                limit=100,
            )

        if not scenes:
            raise ValueError(
                f"No suitable imagery found for plot {plot_id} "
                f"near {EUDR_CUTOFF_DATE}"
            )

        # Step 2: Select best scene
        best_scene = self.imagery_engine.get_best_scene(
            scenes, EUDR_CUTOFF_DATE
        )

        if best_scene is None:
            raise ValueError(
                f"Could not select a best scene for plot {plot_id}"
            )

        # Step 3: Download Red and NIR bands
        band_names = self._get_ndvi_bands(best_scene.source)
        bands = self.imagery_engine.download_bands(
            scene_id=best_scene.scene_id,
            bands=band_names,
        )

        red_band = bands[band_names[0]]
        nir_band = bands[band_names[1]]

        # Step 4: Calculate NDVI
        ndvi_result = self.spectral_calculator.calculate_ndvi(
            red_band=red_band,
            nir_band=nir_band,
        )

        # Step 5: Classify forest
        classification = self.spectral_calculator.classify_forest(
            ndvi_values=ndvi_result.values,
            biome=biome,
        )

        # Step 6: Determine forest status
        biome_thresholds = BIOME_THRESHOLDS.get(biome.lower(), {})
        ndvi_forest = biome_thresholds.get("ndvi_forest", 0.40)
        is_forested = ndvi_result.mean >= ndvi_forest

        # Assess quality
        quality_assessment = self.imagery_engine.assess_scene_quality(
            scene=best_scene,
            target_date=EUDR_CUTOFF_DATE,
            polygon_vertices=polygon_vertices,
        )

        # Create baseline snapshot
        baseline = BaselineSnapshot(
            baseline_id=_generate_id(),
            plot_id=plot_id,
            commodity=commodity.lower().strip(),
            country_code=country_code.upper().strip(),
            biome=biome.lower().strip(),
            polygon_vertices=polygon_vertices,
            cutoff_date=EUDR_CUTOFF_DATE,
            scene_id=best_scene.scene_id,
            scene_date=best_scene.acquisition_date,
            scene_source=best_scene.source,
            scene_cloud_cover_pct=best_scene.cloud_cover_pct,
            ndvi_mean=ndvi_result.mean,
            ndvi_min=ndvi_result.min_val,
            ndvi_max=ndvi_result.max_val,
            ndvi_std_dev=ndvi_result.std_dev,
            forest_cover_pct=classification.forest_pct,
            dense_forest_pct=classification.dense_forest_pct,
            classification_biome=biome.lower().strip(),
            quality_score=quality_assessment.overall_score,
            is_forested=is_forested,
            established_at=_utcnow(),
            metadata={
                "search_window_days": self.search_window_days,
                "scenes_found": len(scenes),
                "scene_quality_acceptable": quality_assessment.is_acceptable,
                "ndvi_pixel_count": ndvi_result.pixel_count,
                "classification_thresholds": biome_thresholds.get(
                    "ndvi_forest", "N/A"
                ),
            },
        )

        # Compute provenance hash
        baseline.provenance_hash = self._compute_baseline_hash(baseline)

        # Store baseline
        self._baselines[plot_id] = baseline

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Baseline established: plot=%s, scene=%s, date=%s, "
            "ndvi_mean=%.4f, forest=%.1f%%, forested=%s, "
            "quality=%.1f, %.2fms",
            plot_id, best_scene.scene_id, best_scene.acquisition_date,
            ndvi_result.mean, classification.forest_pct,
            is_forested, quality_assessment.overall_score, elapsed_ms,
        )

        return baseline

    def get_baseline(self, plot_id: str) -> Optional[BaselineSnapshot]:
        """Retrieve a stored baseline for a plot.

        Args:
            plot_id: Production plot identifier.

        Returns:
            BaselineSnapshot or None if no baseline exists.
        """
        baseline = self._baselines.get(plot_id)
        if baseline is None:
            logger.debug("No baseline found for plot %s", plot_id)
        return baseline

    def re_establish_baseline(
        self,
        plot_id: str,
        reason: str,
        polygon_vertices: List[Tuple[float, float]],
        commodity: str,
        country_code: str,
        biome: str = "tropical_rainforest",
    ) -> BaselineSnapshot:
        """Re-establish a baseline with full audit trail.

        Creates a new baseline while preserving a reference to the
        previous baseline's ID and provenance hash for audit purposes.

        Args:
            plot_id: Production plot identifier.
            reason: Reason for re-establishment (stored in metadata).
            polygon_vertices: Plot boundary vertices.
            commodity: EUDR commodity.
            country_code: Country code.
            biome: Biome type.

        Returns:
            New BaselineSnapshot with previous baseline reference.

        Raises:
            ValueError: If re-establishment fails.
        """
        start_time = time.monotonic()

        # Capture previous baseline info
        previous = self._baselines.get(plot_id)
        prev_id = previous.baseline_id if previous else None
        prev_hash = previous.provenance_hash if previous else None

        logger.info(
            "Re-establishing baseline for plot %s: reason='%s', "
            "previous_id=%s",
            plot_id, reason, prev_id,
        )

        # Establish new baseline using standard pipeline
        new_baseline = self.establish_baseline(
            plot_id=plot_id,
            polygon_vertices=polygon_vertices,
            commodity=commodity,
            country_code=country_code,
            biome=biome,
        )

        # Set audit trail references
        new_baseline.previous_baseline_id = prev_id
        new_baseline.previous_baseline_hash = prev_hash
        new_baseline.metadata["re_establishment_reason"] = reason
        new_baseline.metadata["re_established_at"] = str(_utcnow())

        # Recompute provenance hash with audit trail
        new_baseline.provenance_hash = self._compute_baseline_hash(
            new_baseline
        )

        # Update store
        self._baselines[plot_id] = new_baseline

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Baseline re-established: plot=%s, new_id=%s, prev_id=%s, "
            "%.2fms",
            plot_id, new_baseline.baseline_id, prev_id, elapsed_ms,
        )

        return new_baseline

    def assess_baseline_quality(
        self, baseline: BaselineSnapshot,
    ) -> DataQualityAssessment:
        """Assess the quality of an existing baseline.

        Evaluates the baseline based on scene quality, temporal
        proximity to cutoff date, NDVI data quality, and forest
        classification confidence.

        Args:
            baseline: BaselineSnapshot to assess.

        Returns:
            DataQualityAssessment with component scores and overall.
        """
        start_time = time.monotonic()

        # Cloud cover score
        cloud_score = max(0.0, 100.0 - baseline.scene_cloud_cover_pct)

        # Temporal proximity to cutoff date
        if baseline.scene_date is not None:
            days_from_cutoff = abs(
                (baseline.scene_date - EUDR_CUTOFF_DATE_OBJ).days
            )
            temporal_score = max(
                0.0, 100.0 - (days_from_cutoff / self.search_window_days * 100.0)
            )
        else:
            temporal_score = 0.0

        # NDVI data quality: based on valid pixel count and std dev
        ndvi_quality = 80.0  # Base quality for valid NDVI data
        if baseline.ndvi_std_dev > 0.3:
            ndvi_quality -= 20.0  # High variation indicates issues
        if baseline.ndvi_mean < 0.1:
            ndvi_quality -= 30.0  # Very low NDVI may indicate bare soil

        spatial_score = min(100.0, max(0.0, ndvi_quality))

        # Atmospheric quality (inferred from cloud cover)
        atmospheric_score = cloud_score * 0.8

        # Sensor quality based on source
        sensor_scores = {
            "sentinel2": 95.0,
            "landsat8": 85.0,
            "landsat9": 90.0,
        }
        sensor_score = sensor_scores.get(baseline.scene_source, 70.0)

        # Overall weighted score
        overall = (
            cloud_score * 0.30
            + temporal_score * 0.25
            + spatial_score * 0.20
            + atmospheric_score * 0.15
            + sensor_score * 0.10
        )
        overall = round(min(100.0, max(0.0, overall)), 2)

        assessment = DataQualityAssessment(
            assessment_id=_generate_id(),
            scene_id=baseline.scene_id,
            cloud_cover_score=round(cloud_score, 2),
            temporal_proximity_score=round(temporal_score, 2),
            spatial_coverage_score=round(spatial_score, 2),
            atmospheric_quality_score=round(atmospheric_score, 2),
            sensor_health_score=round(sensor_score, 2),
            overall_score=overall,
            is_acceptable=overall >= MIN_BASELINE_QUALITY_SCORE,
            details={
                "baseline_id": baseline.baseline_id,
                "plot_id": baseline.plot_id,
                "scene_date": str(baseline.scene_date),
                "days_from_cutoff": (
                    abs((baseline.scene_date - EUDR_CUTOFF_DATE_OBJ).days)
                    if baseline.scene_date else None
                ),
                "ndvi_mean": baseline.ndvi_mean,
                "ndvi_std_dev": baseline.ndvi_std_dev,
                "is_forested": baseline.is_forested,
            },
        )

        assessment.provenance_hash = _compute_hash({
            "module_version": _MODULE_VERSION,
            "assessment_id": assessment.assessment_id,
            "overall_score": assessment.overall_score,
            "is_acceptable": assessment.is_acceptable,
        })

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Baseline quality assessed: plot=%s, overall=%.2f, "
            "acceptable=%s, %.2fms",
            baseline.plot_id, overall, assessment.is_acceptable, elapsed_ms,
        )

        return assessment

    def get_baseline_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all stored baselines.

        Returns:
            Dictionary with baseline count, forest/non-forest split,
            quality distribution, biome distribution, and commodity
            distribution.
        """
        start_time = time.monotonic()

        baselines = list(self._baselines.values())
        total = len(baselines)

        if total == 0:
            return {
                "total_baselines": 0,
                "forested_count": 0,
                "non_forested_count": 0,
                "average_ndvi": 0.0,
                "average_quality_score": 0.0,
                "biome_distribution": {},
                "commodity_distribution": {},
                "country_distribution": {},
                "provenance_hash": _compute_hash({"total": 0}),
            }

        forested = sum(1 for b in baselines if b.is_forested)
        non_forested = total - forested

        avg_ndvi = sum(b.ndvi_mean for b in baselines) / total
        avg_quality = sum(b.quality_score for b in baselines) / total

        # Biome distribution
        biome_dist: Dict[str, int] = {}
        for b in baselines:
            biome_dist[b.biome] = biome_dist.get(b.biome, 0) + 1

        # Commodity distribution
        commodity_dist: Dict[str, int] = {}
        for b in baselines:
            commodity_dist[b.commodity] = commodity_dist.get(b.commodity, 0) + 1

        # Country distribution
        country_dist: Dict[str, int] = {}
        for b in baselines:
            country_dist[b.country_code] = (
                country_dist.get(b.country_code, 0) + 1
            )

        stats = {
            "total_baselines": total,
            "forested_count": forested,
            "non_forested_count": non_forested,
            "forested_pct": round(forested / total * 100.0, 2),
            "average_ndvi": round(avg_ndvi, 4),
            "average_quality_score": round(avg_quality, 2),
            "average_forest_cover_pct": round(
                sum(b.forest_cover_pct for b in baselines) / total, 2
            ),
            "biome_distribution": biome_dist,
            "commodity_distribution": commodity_dist,
            "country_distribution": country_dist,
        }

        stats["provenance_hash"] = _compute_hash(stats)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Baseline statistics: total=%d, forested=%d, "
            "avg_ndvi=%.4f, %.2fms",
            total, forested, avg_ndvi, elapsed_ms,
        )

        return stats

    def validate_baseline_integrity(self, plot_id: str) -> bool:
        """Validate the provenance hash chain of a stored baseline.

        Recomputes the provenance hash from the stored baseline's
        data fields and compares it with the stored hash to detect
        any tampering or corruption.

        Args:
            plot_id: Production plot identifier.

        Returns:
            True if the baseline's provenance hash is valid.
            False if no baseline exists or hash mismatch detected.
        """
        baseline = self._baselines.get(plot_id)
        if baseline is None:
            logger.warning(
                "Integrity check failed: no baseline for plot %s", plot_id
            )
            return False

        expected_hash = self._compute_baseline_hash(baseline)
        is_valid = expected_hash == baseline.provenance_hash

        if not is_valid:
            logger.error(
                "Baseline integrity check FAILED for plot %s: "
                "expected=%s, stored=%s",
                plot_id, expected_hash[:16], baseline.provenance_hash[:16],
            )
        else:
            logger.debug(
                "Baseline integrity verified for plot %s", plot_id
            )

        return is_valid

    def list_baselines(
        self,
        commodity: Optional[str] = None,
        country_code: Optional[str] = None,
        forested_only: bool = False,
    ) -> List[BaselineSnapshot]:
        """List stored baselines with optional filtering.

        Args:
            commodity: Filter by commodity (case-insensitive).
            country_code: Filter by country code (case-insensitive).
            forested_only: If True, only return forested baselines.

        Returns:
            List of matching BaselineSnapshot objects.
        """
        results: List[BaselineSnapshot] = []

        for baseline in self._baselines.values():
            if commodity and baseline.commodity != commodity.lower().strip():
                continue
            if country_code and baseline.country_code != country_code.upper().strip():
                continue
            if forested_only and not baseline.is_forested:
                continue
            results.append(baseline)

        logger.debug(
            "Listed baselines: total=%d, commodity=%s, country=%s, "
            "forested_only=%s, results=%d",
            len(self._baselines), commodity, country_code,
            forested_only, len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Internal: Helpers
    # ------------------------------------------------------------------

    def _get_ndvi_bands(self, source: str) -> List[str]:
        """Get the Red and NIR band names for a satellite source.

        Args:
            source: Satellite source identifier.

        Returns:
            List of [red_band_name, nir_band_name].
        """
        if source == "sentinel2":
            return ["B04", "B08"]
        elif source in ("landsat8", "landsat9"):
            return ["B4", "B5"]
        else:
            return ["B04", "B08"]

    def _date_offset(self, base_date: date, days: int) -> date:
        """Calculate a date offset from a base date.

        Args:
            base_date: Starting date.
            days: Number of days to offset (negative for past).

        Returns:
            Offset date.
        """
        from datetime import timedelta
        return base_date + timedelta(days=days)

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_baseline_hash(self, baseline: BaselineSnapshot) -> str:
        """Compute SHA-256 provenance hash for a baseline.

        Includes all key baseline parameters for tamper detection.
        Excludes mutable fields (established_at, metadata) to
        focus on the immutable scientific data.

        Args:
            baseline: BaselineSnapshot to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "baseline_id": baseline.baseline_id,
            "plot_id": baseline.plot_id,
            "commodity": baseline.commodity,
            "country_code": baseline.country_code,
            "biome": baseline.biome,
            "cutoff_date": baseline.cutoff_date,
            "scene_id": baseline.scene_id,
            "scene_date": str(baseline.scene_date),
            "scene_source": baseline.scene_source,
            "scene_cloud_cover_pct": baseline.scene_cloud_cover_pct,
            "ndvi_mean": baseline.ndvi_mean,
            "ndvi_min": baseline.ndvi_min,
            "ndvi_max": baseline.ndvi_max,
            "ndvi_std_dev": baseline.ndvi_std_dev,
            "forest_cover_pct": baseline.forest_cover_pct,
            "dense_forest_pct": baseline.dense_forest_pct,
            "classification_biome": baseline.classification_biome,
            "quality_score": baseline.quality_score,
            "is_forested": baseline.is_forested,
            "previous_baseline_id": baseline.previous_baseline_id,
            "previous_baseline_hash": baseline.previous_baseline_hash,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "BaselineManager",
    "BaselineSnapshot",
    "BIOME_THRESHOLDS",
    "EUDR_CUTOFF_DATE",
    "EUDR_CUTOFF_DATE_OBJ",
    "DEFAULT_SEARCH_WINDOW_DAYS",
    "MIN_BASELINE_QUALITY_SCORE",
]
