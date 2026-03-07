# -*- coding: utf-8 -*-
"""
CanopyHeightModeler - AGENT-EUDR-004: Forest Cover Analysis (Engine 1)

Estimates tree canopy height for FAO 5-metre threshold verification using
multi-source remote sensing data fusion. Combines lidar-derived heights
(GEDI, ICESat-2), global canopy height maps (ETH Zurich, Meta/WRI), and
texture-based proxies (Sentinel-2 GLCM) into a single fused estimate with
propagated uncertainty.

The FAO definition of forest requires trees capable of reaching 5 metres
at maturity. This engine provides deterministic evidence for that threshold
by fusing all available height observations and reporting whether the
estimated canopy height meets the requirement.

Data Sources (5):
    1. GEDI L2A/L2B (NASA) - 25m footprint lidar, RH95/RH98 metrics, +/-3m
    2. ICESat-2 ATL08 (NASA) - Photon-counting lidar, ~100m segments, +/-5m
    3. Sentinel-2 GLCM Texture - Texture contrast as height proxy, r^2~0.6
    4. ETH Zurich Global Map - Pre-computed 10m map (2020), +/-5m
    5. Meta/WRI Global Map - AI-derived 1m map (2023), +/-4m

Fusion Weights (Default):
    GEDI:     0.35 (highest accuracy, sparse coverage)
    ICESat2:  0.25 (orbital tracks only, moderate accuracy)
    ETH:      0.20 (wall-to-wall, moderate accuracy)
    Meta:     0.15 (wall-to-wall, AI-derived)
    Texture:  0.05 (proxy only, lowest accuracy)

FAO Threshold:
    Height at maturity >= 5.0 metres (forest definition per FAO/EUDR)

Zero-Hallucination Guarantees:
    - All height estimates come from source data, never LLM-generated.
    - Fusion uses deterministic weighted arithmetic.
    - Uncertainty propagation uses standard RMS formula.
    - FAO threshold check is a simple numeric comparison.
    - SHA-256 provenance hashes on all result objects.
    - No ML/LLM used for any height calculation.

Performance Targets:
    - Single plot height estimation: <50ms
    - Batch estimation (100 plots): <2 seconds
    - Multi-source fusion: <5ms

Regulatory References:
    - EUDR Article 2(1): Forest definition using FAO criteria
    - EUDR Article 2(5): Trees reaching 5m height at maturity
    - EUDR Article 9: Geolocation and spatial analysis requirements
    - FAO Global Forest Resources Assessment: 5m height threshold
    - IPCC 2006 Guidelines Vol 4: Forest land characterization

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 1: Canopy Height Modeling)
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
from typing import Any, Dict, List, Optional, Tuple

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
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Data to hash (dict or other JSON-serializable object).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "cht") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Constants: FAO Threshold
# ---------------------------------------------------------------------------

#: FAO forest definition minimum tree height in metres.
FAO_HEIGHT_THRESHOLD_M: float = 5.0

# ---------------------------------------------------------------------------
# Constants: Source Accuracy Specifications
# ---------------------------------------------------------------------------

#: Inherent uncertainty (metres, 1-sigma) for each height source.
SOURCE_UNCERTAINTY_M: Dict[str, float] = {
    "gedi": 3.0,
    "icesat2": 5.0,
    "eth_map": 5.0,
    "meta_map": 4.0,
    "texture": 8.0,
}

#: Default fusion weights (must sum to 1.0 across all 5 sources).
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "gedi": 0.35,
    "icesat2": 0.25,
    "eth_map": 0.20,
    "meta_map": 0.15,
    "texture": 0.05,
}

#: Spatial resolution of each source in metres.
SOURCE_RESOLUTION_M: Dict[str, float] = {
    "gedi": 25.0,
    "icesat2": 100.0,
    "eth_map": 10.0,
    "meta_map": 1.0,
    "texture": 10.0,
}

#: Coverage type for each source.
SOURCE_COVERAGE: Dict[str, str] = {
    "gedi": "sparse_orbital",
    "icesat2": "sparse_orbital",
    "eth_map": "wall_to_wall",
    "meta_map": "wall_to_wall",
    "texture": "wall_to_wall",
}

# ---------------------------------------------------------------------------
# Constants: Biome-Specific Calibration Offsets
# ---------------------------------------------------------------------------

#: Calibration offset (metres) added to raw height by biome.
#: Accounts for systematic biases in remote sensing height estimation
#: per vegetation type. Positive offset = source tends to underestimate.
BIOME_CALIBRATION_OFFSETS: Dict[str, Dict[str, float]] = {
    "tropical_rainforest": {
        "gedi": 0.0,
        "icesat2": -1.2,
        "eth_map": 0.5,
        "meta_map": 0.3,
        "texture": 2.0,
    },
    "tropical_dry": {
        "gedi": 0.0,
        "icesat2": -0.8,
        "eth_map": 0.3,
        "meta_map": 0.2,
        "texture": 1.5,
    },
    "temperate": {
        "gedi": 0.0,
        "icesat2": -0.5,
        "eth_map": 0.4,
        "meta_map": 0.2,
        "texture": 1.8,
    },
    "boreal": {
        "gedi": 0.0,
        "icesat2": -0.3,
        "eth_map": 0.2,
        "meta_map": 0.1,
        "texture": 1.2,
    },
    "mangrove": {
        "gedi": 0.5,
        "icesat2": -1.5,
        "eth_map": 0.8,
        "meta_map": 0.5,
        "texture": 2.5,
    },
    "plantation": {
        "gedi": 0.0,
        "icesat2": -0.5,
        "eth_map": 0.3,
        "meta_map": 0.2,
        "texture": 1.0,
    },
    "agroforestry": {
        "gedi": 0.0,
        "icesat2": -0.4,
        "eth_map": 0.4,
        "meta_map": 0.3,
        "texture": 1.5,
    },
}

#: Typical canopy height ranges (min, max) per biome in metres.
BIOME_HEIGHT_RANGES: Dict[str, Tuple[float, float]] = {
    "tropical_rainforest": (20.0, 60.0),
    "tropical_dry": (8.0, 25.0),
    "temperate": (15.0, 45.0),
    "boreal": (5.0, 25.0),
    "mangrove": (3.0, 30.0),
    "plantation": (8.0, 35.0),
    "agroforestry": (3.0, 20.0),
}

#: Default biome when none is specified.
DEFAULT_BIOME = "tropical_rainforest"

# ---------------------------------------------------------------------------
# Constants: GEDI Lidar Parameters
# ---------------------------------------------------------------------------

#: GEDI footprint diameter in metres.
GEDI_FOOTPRINT_M: float = 25.0

#: GEDI accuracy specification (1-sigma) in metres.
GEDI_ACCURACY_M: float = 3.0

#: RH metrics available from GEDI L2A.
GEDI_RH_METRICS: List[str] = [
    "RH25", "RH50", "RH75", "RH90", "RH95", "RH98", "RH100",
]

#: Default GEDI metric used for canopy top height.
GEDI_DEFAULT_METRIC: str = "RH98"

# ---------------------------------------------------------------------------
# Constants: ICESat-2 Parameters
# ---------------------------------------------------------------------------

#: ICESat-2 ATL08 segment length in metres.
ICESAT2_SEGMENT_M: float = 100.0

#: ICESat-2 accuracy specification (1-sigma) in metres.
ICESAT2_ACCURACY_M: float = 5.0

# ---------------------------------------------------------------------------
# Constants: Texture-Based Height Proxy
# ---------------------------------------------------------------------------

#: Texture-height regression coefficients by biome.
#: height_proxy = a * texture_contrast + b
#: Coefficients derived from published literature correlations.
TEXTURE_REGRESSION_COEFFICIENTS: Dict[str, Tuple[float, float]] = {
    "tropical_rainforest": (0.45, 5.0),
    "tropical_dry": (0.35, 3.0),
    "temperate": (0.40, 4.0),
    "boreal": (0.30, 2.5),
    "mangrove": (0.38, 2.0),
    "plantation": (0.32, 4.0),
    "agroforestry": (0.28, 2.5),
}

#: R-squared of texture-height correlation per biome.
TEXTURE_R_SQUARED: Dict[str, float] = {
    "tropical_rainforest": 0.60,
    "tropical_dry": 0.45,
    "temperate": 0.55,
    "boreal": 0.40,
    "mangrove": 0.50,
    "plantation": 0.50,
    "agroforestry": 0.35,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SourceHeightEstimate:
    """Height estimate from a single remote sensing source.

    Attributes:
        source: Source identifier (gedi, icesat2, eth_map, meta_map, texture).
        height_m: Estimated canopy height in metres.
        uncertainty_m: Uncertainty (1-sigma) in metres.
        resolution_m: Spatial resolution of the source.
        raw_value: Unprocessed value from the source.
        calibrated: Whether biome calibration was applied.
        biome: Biome used for calibration (if applied).
        metric: Specific metric used (e.g., RH98 for GEDI).
        observation_date: Date of observation (if available).
        quality_flag: Source-specific quality indicator (0-1, 1=best).
        provenance_hash: SHA-256 provenance hash.
    """

    source: str = ""
    height_m: float = 0.0
    uncertainty_m: float = 0.0
    resolution_m: float = 0.0
    raw_value: float = 0.0
    calibrated: bool = False
    biome: str = ""
    metric: str = ""
    observation_date: Optional[str] = None
    quality_flag: float = 1.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and reporting."""
        return {
            "source": self.source,
            "height_m": round(self.height_m, 2),
            "uncertainty_m": round(self.uncertainty_m, 2),
            "resolution_m": self.resolution_m,
            "raw_value": round(self.raw_value, 2),
            "calibrated": self.calibrated,
            "biome": self.biome,
            "metric": self.metric,
            "observation_date": self.observation_date,
            "quality_flag": round(self.quality_flag, 3),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CanopyHeightEstimate:
    """Fused canopy height estimate from multiple sources.

    Attributes:
        estimate_id: Unique identifier for this estimate.
        plot_id: Plot identifier.
        height_m: Fused canopy height in metres.
        uncertainty_m: Fused uncertainty (1-sigma) in metres.
        meets_fao_threshold: True if height >= 5.0m.
        fao_threshold_m: The FAO threshold used (5.0m).
        source_estimates: Individual source estimates used in fusion.
        sources_used: List of source names that contributed.
        fusion_weights: Weights used for fusion (re-normalized).
        biome: Biome context for this estimate.
        confidence_score: Overall confidence (0-1) based on source count
            and agreement.
        processing_time_ms: Time to compute this estimate.
        created_at: Timestamp of estimation.
        provenance_hash: SHA-256 provenance hash.
    """

    estimate_id: str = ""
    plot_id: str = ""
    height_m: float = 0.0
    uncertainty_m: float = 0.0
    meets_fao_threshold: bool = False
    fao_threshold_m: float = FAO_HEIGHT_THRESHOLD_M
    source_estimates: List[SourceHeightEstimate] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    biome: str = ""
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    created_at: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and reporting."""
        return {
            "estimate_id": self.estimate_id,
            "plot_id": self.plot_id,
            "height_m": round(self.height_m, 2),
            "uncertainty_m": round(self.uncertainty_m, 2),
            "meets_fao_threshold": self.meets_fao_threshold,
            "fao_threshold_m": self.fao_threshold_m,
            "sources_used": self.sources_used,
            "fusion_weights": {
                k: round(v, 4) for k, v in self.fusion_weights.items()
            },
            "biome": self.biome,
            "confidence_score": round(self.confidence_score, 3),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "created_at": self.created_at,
            "source_count": len(self.source_estimates),
        }


@dataclass
class BatchHeightResult:
    """Result of batch height estimation across multiple plots.

    Attributes:
        batch_id: Unique batch identifier.
        total_plots: Number of plots processed.
        completed: Number successfully processed.
        failed: Number that failed processing.
        estimates: List of individual plot estimates.
        summary: Aggregate statistics.
        processing_time_ms: Total batch processing time.
        provenance_hash: SHA-256 provenance hash.
    """

    batch_id: str = ""
    total_plots: int = 0
    completed: int = 0
    failed: int = 0
    estimates: List[CanopyHeightEstimate] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and reporting."""
        return {
            "batch_id": self.batch_id,
            "total_plots": self.total_plots,
            "completed": self.completed,
            "failed": self.failed,
            "summary": self.summary,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# CanopyHeightModeler
# ---------------------------------------------------------------------------


class CanopyHeightModeler:
    """Multi-source canopy height estimation engine for EUDR compliance.

    Fuses height observations from GEDI lidar, ICESat-2 lidar, ETH Zurich
    global canopy height map, Meta/WRI global canopy height map, and
    Sentinel-2 GLCM texture-based height proxies. All calculations are
    deterministic with SHA-256 provenance hashing.

    The primary output is a CanopyHeightEstimate indicating whether the
    estimated canopy height meets the FAO 5-metre threshold required by
    the EUDR definition of forest.

    Example::

        modeler = CanopyHeightModeler()
        gedi_data = {"rh98": 22.5, "quality": 0.9, "date": "2021-06-15"}
        estimate = modeler.estimate_plot_height(
            plot_id="PLOT-001",
            gedi_data=gedi_data,
            biome="tropical_rainforest",
        )
        assert estimate.meets_fao_threshold is True
        assert estimate.provenance_hash != ""

    Attributes:
        fusion_weights: Source fusion weights.
        biome: Default biome for calibration.
    """

    def __init__(
        self,
        fusion_weights: Optional[Dict[str, float]] = None,
        biome: str = DEFAULT_BIOME,
    ) -> None:
        """Initialize the CanopyHeightModeler.

        Args:
            fusion_weights: Optional custom fusion weights. Must contain
                keys from {gedi, icesat2, eth_map, meta_map, texture}
                and values summing to 1.0.
            biome: Default biome for calibration. Must be a key in
                BIOME_CALIBRATION_OFFSETS.

        Raises:
            ValueError: If fusion weights do not sum to 1.0 or biome
                is not recognized.
        """
        self.fusion_weights = dict(
            fusion_weights if fusion_weights is not None
            else DEFAULT_FUSION_WEIGHTS
        )
        self.biome = biome

        self._validate_fusion_weights(self.fusion_weights)
        self._validate_biome(self.biome)

        logger.info(
            "CanopyHeightModeler initialized: biome=%s, weights=%s",
            self.biome,
            {k: round(v, 3) for k, v in self.fusion_weights.items()},
        )

    # ------------------------------------------------------------------
    # Public API: Single-Source Estimation
    # ------------------------------------------------------------------

    def estimate_from_gedi(
        self,
        rh_metrics: Dict[str, float],
        quality: float = 1.0,
        observation_date: Optional[str] = None,
        metric: str = GEDI_DEFAULT_METRIC,
        biome: Optional[str] = None,
    ) -> SourceHeightEstimate:
        """Estimate canopy height from NASA GEDI L2A/L2B lidar data.

        Uses the specified RH (relative height) metric as the canopy
        top height. RH98 is the default, representing the 98th
        percentile return height which corresponds to canopy top.

        GEDI has 25m footprint diameter with +/-3m vertical accuracy.
        This is the highest-accuracy spaceborne lidar for forest height.

        Args:
            rh_metrics: Dictionary of RH metrics, e.g. {"RH98": 22.5}.
                Keys should be from GEDI_RH_METRICS.
            quality: Quality flag (0-1, 1=best). GEDI shots with low
                quality (beam sensitivity < 0.9) should use lower values.
            observation_date: ISO date string of the GEDI overpass.
            metric: Which RH metric to use for canopy height (default RH98).
            biome: Biome for calibration. Uses engine default if None.

        Returns:
            SourceHeightEstimate from GEDI.

        Raises:
            ValueError: If the specified metric is not in rh_metrics.
        """
        start_time = time.monotonic()
        effective_biome = biome or self.biome

        metric_upper = metric.upper()
        if metric_upper not in rh_metrics:
            available = list(rh_metrics.keys())
            raise ValueError(
                f"GEDI metric '{metric_upper}' not found in provided "
                f"rh_metrics. Available: {available}"
            )

        raw_height = rh_metrics[metric_upper]
        if raw_height < 0.0:
            raw_height = 0.0

        calibrated_height = self._apply_calibration(
            raw_height, "gedi", effective_biome
        )

        uncertainty = GEDI_ACCURACY_M
        if quality < 1.0:
            uncertainty = GEDI_ACCURACY_M / max(quality, 0.1)

        estimate = SourceHeightEstimate(
            source="gedi",
            height_m=round(calibrated_height, 2),
            uncertainty_m=round(uncertainty, 2),
            resolution_m=GEDI_FOOTPRINT_M,
            raw_value=round(raw_height, 2),
            calibrated=True,
            biome=effective_biome,
            metric=metric_upper,
            observation_date=observation_date,
            quality_flag=round(max(0.0, min(1.0, quality)), 3),
        )
        estimate.provenance_hash = _compute_hash(estimate.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "GEDI height estimate: %.2fm (raw=%.2f, metric=%s, "
            "uncertainty=+/-%.2fm, quality=%.2f), %.2fms",
            calibrated_height, raw_height, metric_upper,
            uncertainty, quality, elapsed_ms,
        )

        return estimate

    def estimate_from_icesat2(
        self,
        canopy_height_m: float,
        terrain_height_m: float = 0.0,
        n_photons: int = 0,
        segment_length_m: float = ICESAT2_SEGMENT_M,
        quality: float = 1.0,
        observation_date: Optional[str] = None,
        biome: Optional[str] = None,
    ) -> SourceHeightEstimate:
        """Estimate canopy height from ICESat-2 ATL08 vegetation data.

        ICESat-2 uses photon-counting lidar organized into ~100m along-
        track segments. The canopy height is derived from the difference
        between the canopy top (98th percentile photon height) and the
        terrain surface. Accuracy is approximately +/-5m.

        Args:
            canopy_height_m: ATL08 canopy height (h_canopy) in metres.
            terrain_height_m: ATL08 terrain height for reference.
            n_photons: Number of canopy photons (more = higher confidence).
            segment_length_m: Along-track segment length.
            quality: Quality flag (0-1, 1=best).
            observation_date: ISO date string of the overpass.
            biome: Biome for calibration. Uses engine default if None.

        Returns:
            SourceHeightEstimate from ICESat-2.
        """
        start_time = time.monotonic()
        effective_biome = biome or self.biome

        raw_height = max(0.0, canopy_height_m)

        calibrated_height = self._apply_calibration(
            raw_height, "icesat2", effective_biome
        )

        uncertainty = ICESAT2_ACCURACY_M
        if n_photons > 0:
            photon_factor = min(1.0, n_photons / 100.0)
            uncertainty = ICESAT2_ACCURACY_M / max(photon_factor, 0.1)
            uncertainty = min(uncertainty, ICESAT2_ACCURACY_M * 3.0)
        if quality < 1.0:
            uncertainty = uncertainty / max(quality, 0.1)

        effective_quality = quality
        if n_photons > 0:
            photon_quality = min(1.0, n_photons / 100.0)
            effective_quality = (quality + photon_quality) / 2.0

        estimate = SourceHeightEstimate(
            source="icesat2",
            height_m=round(calibrated_height, 2),
            uncertainty_m=round(min(uncertainty, 25.0), 2),
            resolution_m=segment_length_m,
            raw_value=round(raw_height, 2),
            calibrated=True,
            biome=effective_biome,
            metric="h_canopy_98pct",
            observation_date=observation_date,
            quality_flag=round(max(0.0, min(1.0, effective_quality)), 3),
        )
        estimate.provenance_hash = _compute_hash(estimate.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "ICESat-2 height estimate: %.2fm (raw=%.2f, terrain=%.2f, "
            "photons=%d, uncertainty=+/-%.2fm), %.2fms",
            calibrated_height, raw_height, terrain_height_m,
            n_photons, estimate.uncertainty_m, elapsed_ms,
        )

        return estimate

    def estimate_from_texture(
        self,
        texture_contrast: float,
        biome: Optional[str] = None,
        observation_date: Optional[str] = None,
    ) -> SourceHeightEstimate:
        """Estimate canopy height from Sentinel-2 GLCM texture metrics.

        Uses biome-specific linear regression of GLCM texture contrast
        to approximate canopy height. This is the lowest-accuracy source
        (r-squared ~0.35-0.60 depending on biome) but provides wall-to-wall
        coverage where lidar is unavailable.

        Regression: height_proxy = a * texture_contrast + b
        where (a, b) are biome-specific fitted coefficients.

        Args:
            texture_contrast: GLCM contrast value from Sentinel-2 imagery.
                Typically in range [0, 100] for 10m pixels.
            biome: Biome for regression coefficients. Uses engine default
                if None.
            observation_date: ISO date string of the Sentinel-2 acquisition.

        Returns:
            SourceHeightEstimate from texture proxy.

        Raises:
            ValueError: If biome has no regression coefficients.
        """
        start_time = time.monotonic()
        effective_biome = biome or self.biome

        if effective_biome not in TEXTURE_REGRESSION_COEFFICIENTS:
            raise ValueError(
                f"No texture regression coefficients for biome "
                f"'{effective_biome}'. Available: "
                f"{list(TEXTURE_REGRESSION_COEFFICIENTS.keys())}"
            )

        a_coeff, b_coeff = TEXTURE_REGRESSION_COEFFICIENTS[effective_biome]
        raw_height = a_coeff * texture_contrast + b_coeff
        raw_height = max(0.0, raw_height)

        r_squared = TEXTURE_R_SQUARED.get(effective_biome, 0.4)
        base_uncertainty = SOURCE_UNCERTAINTY_M["texture"]
        uncertainty = base_uncertainty * (1.0 - r_squared + 0.4)

        quality = r_squared

        calibrated_height = self._apply_calibration(
            raw_height, "texture", effective_biome
        )

        estimate = SourceHeightEstimate(
            source="texture",
            height_m=round(calibrated_height, 2),
            uncertainty_m=round(uncertainty, 2),
            resolution_m=SOURCE_RESOLUTION_M["texture"],
            raw_value=round(raw_height, 2),
            calibrated=True,
            biome=effective_biome,
            metric=f"GLCM_contrast(a={a_coeff},b={b_coeff})",
            observation_date=observation_date,
            quality_flag=round(quality, 3),
        )
        estimate.provenance_hash = _compute_hash(estimate.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Texture height estimate: %.2fm (contrast=%.2f, r2=%.2f, "
            "uncertainty=+/-%.2fm), %.2fms",
            calibrated_height, texture_contrast, r_squared,
            uncertainty, elapsed_ms,
        )

        return estimate

    def estimate_from_global_map_eth(
        self,
        height_value_m: float,
        pixel_count: int = 1,
        observation_date: Optional[str] = None,
        biome: Optional[str] = None,
    ) -> SourceHeightEstimate:
        """Look up canopy height from ETH Zurich global canopy height map.

        The ETH global canopy height map (Lang et al., 2022) provides
        10m resolution canopy height estimates globally, derived from
        Sentinel-2 imagery using deep learning. Reference year is 2020.

        This is a direct lookup from a pre-computed dataset, making it
        a zero-hallucination source with known accuracy (+/-5m).

        Args:
            height_value_m: Height value from the ETH map in metres.
            pixel_count: Number of 10m pixels averaged (more = lower
                uncertainty via spatial averaging).
            observation_date: Reference date (typically "2020-01-01").
            biome: Biome for calibration. Uses engine default if None.

        Returns:
            SourceHeightEstimate from ETH map.
        """
        start_time = time.monotonic()
        effective_biome = biome or self.biome

        raw_height = max(0.0, height_value_m)

        calibrated_height = self._apply_calibration(
            raw_height, "eth_map", effective_biome
        )

        base_uncertainty = SOURCE_UNCERTAINTY_M["eth_map"]
        if pixel_count > 1:
            uncertainty = base_uncertainty / math.sqrt(min(pixel_count, 100))
        else:
            uncertainty = base_uncertainty

        quality = min(1.0, 0.7 + 0.003 * min(pixel_count, 100))

        estimate = SourceHeightEstimate(
            source="eth_map",
            height_m=round(calibrated_height, 2),
            uncertainty_m=round(uncertainty, 2),
            resolution_m=SOURCE_RESOLUTION_M["eth_map"],
            raw_value=round(raw_height, 2),
            calibrated=True,
            biome=effective_biome,
            metric=f"ETH_Zurich_10m(n_pixels={pixel_count})",
            observation_date=observation_date or "2020-01-01",
            quality_flag=round(quality, 3),
        )
        estimate.provenance_hash = _compute_hash(estimate.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "ETH map height: %.2fm (raw=%.2f, pixels=%d, "
            "uncertainty=+/-%.2fm), %.2fms",
            calibrated_height, raw_height, pixel_count,
            uncertainty, elapsed_ms,
        )

        return estimate

    def estimate_from_global_map_meta(
        self,
        height_value_m: float,
        pixel_count: int = 1,
        observation_date: Optional[str] = None,
        biome: Optional[str] = None,
    ) -> SourceHeightEstimate:
        """Look up canopy height from Meta/WRI global canopy height map.

        The Meta/WRI high-resolution canopy height map (Tolan et al., 2023)
        provides 1m resolution canopy height estimates globally, derived
        from high-resolution optical imagery using AI. Reference year 2023.

        Despite AI derivation, the lookup is deterministic (same input
        coordinates always return the same pre-computed value).

        Args:
            height_value_m: Height value from the Meta/WRI map in metres.
            pixel_count: Number of 1m pixels averaged.
            observation_date: Reference date (typically "2023-01-01").
            biome: Biome for calibration. Uses engine default if None.

        Returns:
            SourceHeightEstimate from Meta/WRI map.
        """
        start_time = time.monotonic()
        effective_biome = biome or self.biome

        raw_height = max(0.0, height_value_m)

        calibrated_height = self._apply_calibration(
            raw_height, "meta_map", effective_biome
        )

        base_uncertainty = SOURCE_UNCERTAINTY_M["meta_map"]
        if pixel_count > 1:
            uncertainty = base_uncertainty / math.sqrt(min(pixel_count, 1000))
        else:
            uncertainty = base_uncertainty

        quality = min(1.0, 0.75 + 0.00025 * min(pixel_count, 1000))

        estimate = SourceHeightEstimate(
            source="meta_map",
            height_m=round(calibrated_height, 2),
            uncertainty_m=round(uncertainty, 2),
            resolution_m=SOURCE_RESOLUTION_M["meta_map"],
            raw_value=round(raw_height, 2),
            calibrated=True,
            biome=effective_biome,
            metric=f"Meta_WRI_1m(n_pixels={pixel_count})",
            observation_date=observation_date or "2023-01-01",
            quality_flag=round(quality, 3),
        )
        estimate.provenance_hash = _compute_hash(estimate.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Meta/WRI map height: %.2fm (raw=%.2f, pixels=%d, "
            "uncertainty=+/-%.2fm), %.2fms",
            calibrated_height, raw_height, pixel_count,
            uncertainty, elapsed_ms,
        )

        return estimate

    # ------------------------------------------------------------------
    # Public API: Multi-Source Fusion
    # ------------------------------------------------------------------

    def fuse_height_estimates(
        self,
        estimates: List[SourceHeightEstimate],
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Fuse height estimates from multiple sources using weighted average.

        Only sources with provided estimates contribute. Weights are
        re-normalized across available sources so they sum to 1.0.
        If a source appears more than once, only the first estimate is used.

        Uncertainty propagation: fused_uncertainty = sqrt(sum(w_i^2 * u_i^2))
        where w_i are normalized weights and u_i are source uncertainties.

        Args:
            estimates: List of SourceHeightEstimate objects to fuse.
            custom_weights: Optional custom weights override.

        Returns:
            Tuple of (fused_height_m, fused_uncertainty_m, used_weights).

        Raises:
            ValueError: If no estimates are provided.
        """
        if not estimates:
            raise ValueError("At least one height estimate is required for fusion")

        weights = dict(
            custom_weights if custom_weights is not None
            else self.fusion_weights
        )

        seen_sources: Dict[str, SourceHeightEstimate] = {}
        for est in estimates:
            if est.source not in seen_sources:
                seen_sources[est.source] = est

        available_weights: Dict[str, float] = {}
        for source, est in seen_sources.items():
            if source in weights:
                available_weights[source] = weights[source]
            else:
                available_weights[source] = 0.1

        weight_sum = sum(available_weights.values())
        if weight_sum <= 0:
            raise ValueError("All available source weights are zero")

        normalized_weights = {
            k: v / weight_sum for k, v in available_weights.items()
        }

        fused_height = 0.0
        for source, w in normalized_weights.items():
            fused_height += w * seen_sources[source].height_m

        fused_uncertainty_sq = 0.0
        for source, w in normalized_weights.items():
            u = seen_sources[source].uncertainty_m
            fused_uncertainty_sq += (w * u) ** 2

        fused_uncertainty = math.sqrt(fused_uncertainty_sq)

        logger.debug(
            "Fused %d sources: height=%.2fm, uncertainty=+/-%.2fm, "
            "weights=%s",
            len(seen_sources), fused_height, fused_uncertainty,
            {k: round(v, 3) for k, v in normalized_weights.items()},
        )

        return (
            round(fused_height, 2),
            round(fused_uncertainty, 2),
            {k: round(v, 4) for k, v in normalized_weights.items()},
        )

    # ------------------------------------------------------------------
    # Public API: FAO Threshold Check
    # ------------------------------------------------------------------

    def check_fao_height_threshold(
        self,
        height_m: float,
        uncertainty_m: float = 0.0,
    ) -> Tuple[bool, str]:
        """Check whether estimated height meets the FAO 5-metre threshold.

        The FAO definition of forest requires trees capable of reaching
        5 metres at maturity. This method performs a simple numeric
        comparison with optional consideration of uncertainty.

        Decision logic:
            - height >= 5.0m: MEETS_THRESHOLD
            - height + uncertainty >= 5.0m AND height < 5.0m: UNCERTAIN
            - height + uncertainty < 5.0m: BELOW_THRESHOLD

        Args:
            height_m: Estimated canopy height in metres.
            uncertainty_m: Height uncertainty (1-sigma) in metres.

        Returns:
            Tuple of (meets_threshold: bool, status: str).
        """
        threshold = FAO_HEIGHT_THRESHOLD_M

        if height_m >= threshold:
            return True, "MEETS_THRESHOLD"
        elif (height_m + uncertainty_m) >= threshold:
            return True, "UNCERTAIN_MEETS_THRESHOLD"
        else:
            return False, "BELOW_THRESHOLD"

    # ------------------------------------------------------------------
    # Public API: Main Entry Point
    # ------------------------------------------------------------------

    def estimate_plot_height(
        self,
        plot_id: str,
        gedi_data: Optional[Dict[str, Any]] = None,
        icesat2_data: Optional[Dict[str, Any]] = None,
        texture_data: Optional[Dict[str, Any]] = None,
        eth_map_data: Optional[Dict[str, Any]] = None,
        meta_map_data: Optional[Dict[str, Any]] = None,
        biome: Optional[str] = None,
    ) -> CanopyHeightEstimate:
        """Estimate canopy height for a single plot from all available sources.

        This is the primary entry point. It collects height estimates from
        all provided data sources, fuses them using weighted averaging, checks
        the FAO 5-metre threshold, and returns a complete CanopyHeightEstimate
        with provenance.

        Args:
            plot_id: Unique identifier for the plot.
            gedi_data: Optional GEDI data dict with keys:
                - rh_metrics (Dict[str, float]): RH percentile values.
                - quality (float): Quality flag 0-1.
                - date (str): Observation date.
                - metric (str): Which RH metric to use.
            icesat2_data: Optional ICESat-2 data dict with keys:
                - canopy_height_m (float): h_canopy value.
                - terrain_height_m (float): Terrain reference.
                - n_photons (int): Canopy photon count.
                - quality (float): Quality flag 0-1.
                - date (str): Observation date.
            texture_data: Optional texture data dict with keys:
                - contrast (float): GLCM contrast value.
                - date (str): Observation date.
            eth_map_data: Optional ETH map data dict with keys:
                - height_m (float): Map height value.
                - pixel_count (int): Number of averaged pixels.
                - date (str): Reference date.
            meta_map_data: Optional Meta/WRI map data dict with keys:
                - height_m (float): Map height value.
                - pixel_count (int): Number of averaged pixels.
                - date (str): Reference date.
            biome: Biome for calibration. Uses engine default if None.

        Returns:
            CanopyHeightEstimate with fused height, uncertainty, and
            FAO threshold determination.

        Raises:
            ValueError: If no data sources are provided.
        """
        start_time = time.monotonic()
        effective_biome = biome or self.biome

        source_estimates: List[SourceHeightEstimate] = []

        if gedi_data is not None:
            est = self._process_gedi_data(gedi_data, effective_biome)
            if est is not None:
                source_estimates.append(est)

        if icesat2_data is not None:
            est = self._process_icesat2_data(icesat2_data, effective_biome)
            if est is not None:
                source_estimates.append(est)

        if texture_data is not None:
            est = self._process_texture_data(texture_data, effective_biome)
            if est is not None:
                source_estimates.append(est)

        if eth_map_data is not None:
            est = self._process_eth_map_data(eth_map_data, effective_biome)
            if est is not None:
                source_estimates.append(est)

        if meta_map_data is not None:
            est = self._process_meta_map_data(meta_map_data, effective_biome)
            if est is not None:
                source_estimates.append(est)

        if not source_estimates:
            raise ValueError(
                f"No valid height data provided for plot '{plot_id}'. "
                f"At least one data source must be supplied."
            )

        fused_height, fused_uncertainty, used_weights = (
            self.fuse_height_estimates(source_estimates)
        )

        meets_threshold, threshold_status = self.check_fao_height_threshold(
            fused_height, fused_uncertainty
        )

        confidence = self._compute_confidence(
            source_estimates, fused_height, fused_uncertainty
        )

        sources_used = [e.source for e in source_estimates]
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = CanopyHeightEstimate(
            estimate_id=_generate_id("cht"),
            plot_id=plot_id,
            height_m=fused_height,
            uncertainty_m=fused_uncertainty,
            meets_fao_threshold=meets_threshold,
            fao_threshold_m=FAO_HEIGHT_THRESHOLD_M,
            source_estimates=source_estimates,
            sources_used=sources_used,
            fusion_weights=used_weights,
            biome=effective_biome,
            confidence_score=round(confidence, 3),
            processing_time_ms=round(elapsed_ms, 2),
            created_at=str(_utcnow()),
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Plot '%s' height estimate: %.2fm +/-%.2fm "
            "(FAO=%s, confidence=%.3f, sources=%s), %.2fms",
            plot_id, fused_height, fused_uncertainty,
            threshold_status, confidence,
            sources_used, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Batch Estimation
    # ------------------------------------------------------------------

    def batch_estimate(
        self,
        plots: List[Dict[str, Any]],
    ) -> BatchHeightResult:
        """Estimate canopy height for multiple plots.

        Each plot dict should contain:
            - plot_id (str): Unique plot identifier.
            - biome (str, optional): Biome override.
            - gedi_data, icesat2_data, texture_data, eth_map_data,
              meta_map_data: Source data dicts (all optional, at least
              one required per plot).

        Args:
            plots: List of plot data dictionaries.

        Returns:
            BatchHeightResult with individual estimates and summary.
        """
        start_time = time.monotonic()
        batch_id = _generate_id("cht-batch")

        estimates: List[CanopyHeightEstimate] = []
        failed = 0

        for plot_data in plots:
            plot_id = plot_data.get("plot_id", _generate_id("plot"))
            try:
                estimate = self.estimate_plot_height(
                    plot_id=plot_id,
                    gedi_data=plot_data.get("gedi_data"),
                    icesat2_data=plot_data.get("icesat2_data"),
                    texture_data=plot_data.get("texture_data"),
                    eth_map_data=plot_data.get("eth_map_data"),
                    meta_map_data=plot_data.get("meta_map_data"),
                    biome=plot_data.get("biome"),
                )
                estimates.append(estimate)
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Batch height estimation failed for plot '%s': %s",
                    plot_id, str(exc),
                )

        summary = self._compute_batch_summary(estimates)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = BatchHeightResult(
            batch_id=batch_id,
            total_plots=len(plots),
            completed=len(estimates),
            failed=failed,
            estimates=estimates,
            summary=summary,
            processing_time_ms=round(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Batch height estimation complete: %d/%d succeeded, "
            "%d failed, %.2fms",
            len(estimates), len(plots), failed, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: Data Processing Helpers
    # ------------------------------------------------------------------

    def _process_gedi_data(
        self,
        data: Dict[str, Any],
        biome: str,
    ) -> Optional[SourceHeightEstimate]:
        """Process GEDI data dictionary into a height estimate.

        Args:
            data: GEDI data dict.
            biome: Biome for calibration.

        Returns:
            SourceHeightEstimate or None if processing fails.
        """
        try:
            rh_metrics = data.get("rh_metrics", {})
            if not rh_metrics:
                rh98 = data.get("rh98")
                if rh98 is not None:
                    rh_metrics = {"RH98": float(rh98)}
                else:
                    logger.warning("GEDI data missing rh_metrics")
                    return None

            return self.estimate_from_gedi(
                rh_metrics=rh_metrics,
                quality=float(data.get("quality", 1.0)),
                observation_date=data.get("date"),
                metric=data.get("metric", GEDI_DEFAULT_METRIC),
                biome=biome,
            )
        except Exception as exc:
            logger.warning("Failed to process GEDI data: %s", str(exc))
            return None

    def _process_icesat2_data(
        self,
        data: Dict[str, Any],
        biome: str,
    ) -> Optional[SourceHeightEstimate]:
        """Process ICESat-2 data dictionary into a height estimate.

        Args:
            data: ICESat-2 data dict.
            biome: Biome for calibration.

        Returns:
            SourceHeightEstimate or None if processing fails.
        """
        try:
            canopy_height = data.get("canopy_height_m")
            if canopy_height is None:
                logger.warning("ICESat-2 data missing canopy_height_m")
                return None

            return self.estimate_from_icesat2(
                canopy_height_m=float(canopy_height),
                terrain_height_m=float(data.get("terrain_height_m", 0.0)),
                n_photons=int(data.get("n_photons", 0)),
                quality=float(data.get("quality", 1.0)),
                observation_date=data.get("date"),
                biome=biome,
            )
        except Exception as exc:
            logger.warning("Failed to process ICESat-2 data: %s", str(exc))
            return None

    def _process_texture_data(
        self,
        data: Dict[str, Any],
        biome: str,
    ) -> Optional[SourceHeightEstimate]:
        """Process texture data dictionary into a height estimate.

        Args:
            data: Texture data dict.
            biome: Biome for calibration.

        Returns:
            SourceHeightEstimate or None if processing fails.
        """
        try:
            contrast = data.get("contrast")
            if contrast is None:
                logger.warning("Texture data missing contrast")
                return None

            return self.estimate_from_texture(
                texture_contrast=float(contrast),
                biome=biome,
                observation_date=data.get("date"),
            )
        except Exception as exc:
            logger.warning("Failed to process texture data: %s", str(exc))
            return None

    def _process_eth_map_data(
        self,
        data: Dict[str, Any],
        biome: str,
    ) -> Optional[SourceHeightEstimate]:
        """Process ETH map data dictionary into a height estimate.

        Args:
            data: ETH map data dict.
            biome: Biome for calibration.

        Returns:
            SourceHeightEstimate or None if processing fails.
        """
        try:
            height = data.get("height_m")
            if height is None:
                logger.warning("ETH map data missing height_m")
                return None

            return self.estimate_from_global_map_eth(
                height_value_m=float(height),
                pixel_count=int(data.get("pixel_count", 1)),
                observation_date=data.get("date"),
                biome=biome,
            )
        except Exception as exc:
            logger.warning("Failed to process ETH map data: %s", str(exc))
            return None

    def _process_meta_map_data(
        self,
        data: Dict[str, Any],
        biome: str,
    ) -> Optional[SourceHeightEstimate]:
        """Process Meta/WRI map data dictionary into a height estimate.

        Args:
            data: Meta/WRI map data dict.
            biome: Biome for calibration.

        Returns:
            SourceHeightEstimate or None if processing fails.
        """
        try:
            height = data.get("height_m")
            if height is None:
                logger.warning("Meta/WRI map data missing height_m")
                return None

            return self.estimate_from_global_map_meta(
                height_value_m=float(height),
                pixel_count=int(data.get("pixel_count", 1)),
                observation_date=data.get("date"),
                biome=biome,
            )
        except Exception as exc:
            logger.warning("Failed to process Meta/WRI map data: %s", str(exc))
            return None

    # ------------------------------------------------------------------
    # Internal: Calibration
    # ------------------------------------------------------------------

    def _apply_calibration(
        self,
        raw_height: float,
        source: str,
        biome: str,
    ) -> float:
        """Apply biome-specific calibration offset to a raw height.

        Args:
            raw_height: Uncalibrated height in metres.
            source: Source identifier.
            biome: Biome for calibration lookup.

        Returns:
            Calibrated height in metres (clamped to >= 0).
        """
        offsets = BIOME_CALIBRATION_OFFSETS.get(biome, {})
        offset = offsets.get(source, 0.0)
        calibrated = raw_height + offset
        return max(0.0, calibrated)

    # ------------------------------------------------------------------
    # Internal: Confidence Scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        estimates: List[SourceHeightEstimate],
        fused_height: float,
        fused_uncertainty: float,
    ) -> float:
        """Compute overall confidence score for a fused height estimate.

        Confidence is based on:
        1. Number of contributing sources (more = higher).
        2. Source quality flags (higher = better).
        3. Agreement between sources (low spread = higher).
        4. Relative uncertainty (lower uncertainty / height = higher).

        Score is in range [0.0, 1.0].

        Args:
            estimates: Individual source estimates.
            fused_height: Fused height in metres.
            fused_uncertainty: Fused uncertainty in metres.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        n_sources = len(estimates)
        if n_sources == 0:
            return 0.0

        source_score = min(1.0, n_sources / 5.0) * 0.30

        avg_quality = sum(e.quality_flag for e in estimates) / n_sources
        quality_score = avg_quality * 0.25

        heights = [e.height_m for e in estimates]
        if n_sources > 1:
            mean_h = sum(heights) / n_sources
            variance = sum((h - mean_h) ** 2 for h in heights) / n_sources
            std_dev = math.sqrt(variance)
            max_expected_std = 10.0
            agreement_score = max(0.0, 1.0 - std_dev / max_expected_std) * 0.25
        else:
            agreement_score = 0.15

        if fused_height > 0:
            relative_unc = fused_uncertainty / fused_height
            uncertainty_score = max(0.0, 1.0 - relative_unc) * 0.20
        else:
            uncertainty_score = 0.0

        total = source_score + quality_score + agreement_score + uncertainty_score
        return min(1.0, max(0.0, total))

    # ------------------------------------------------------------------
    # Internal: Batch Summary
    # ------------------------------------------------------------------

    def _compute_batch_summary(
        self,
        estimates: List[CanopyHeightEstimate],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics for a batch of height estimates.

        Args:
            estimates: List of CanopyHeightEstimate objects.

        Returns:
            Dictionary with summary statistics.
        """
        if not estimates:
            return {
                "mean_height_m": 0.0,
                "min_height_m": 0.0,
                "max_height_m": 0.0,
                "std_dev_height_m": 0.0,
                "mean_uncertainty_m": 0.0,
                "meets_fao_count": 0,
                "below_fao_count": 0,
                "meets_fao_pct": 0.0,
                "mean_confidence": 0.0,
                "mean_sources_per_plot": 0.0,
            }

        heights = [e.height_m for e in estimates]
        n = len(heights)
        mean_h = sum(heights) / n
        min_h = min(heights)
        max_h = max(heights)

        if n > 1:
            variance = sum((h - mean_h) ** 2 for h in heights) / (n - 1)
            std_h = math.sqrt(variance)
        else:
            std_h = 0.0

        mean_unc = sum(e.uncertainty_m for e in estimates) / n
        meets = sum(1 for e in estimates if e.meets_fao_threshold)
        below = n - meets
        mean_conf = sum(e.confidence_score for e in estimates) / n
        mean_sources = sum(len(e.sources_used) for e in estimates) / n

        return {
            "mean_height_m": round(mean_h, 2),
            "min_height_m": round(min_h, 2),
            "max_height_m": round(max_h, 2),
            "std_dev_height_m": round(std_h, 2),
            "mean_uncertainty_m": round(mean_unc, 2),
            "meets_fao_count": meets,
            "below_fao_count": below,
            "meets_fao_pct": round(meets / n * 100.0, 1),
            "mean_confidence": round(mean_conf, 3),
            "mean_sources_per_plot": round(mean_sources, 1),
        }

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_fusion_weights(self, weights: Dict[str, float]) -> None:
        """Validate that fusion weights are non-negative and sum to ~1.0.

        Args:
            weights: Dictionary of source weights.

        Raises:
            ValueError: If any weight is negative or sum deviates
                from 1.0 by more than 0.01.
        """
        for source, w in weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Fusion weight for '{source}' must be >= 0, got {w}"
                )

        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {weight_sum:.4f}. "
                f"Weights: {weights}"
            )

    def _validate_biome(self, biome: str) -> None:
        """Validate that a biome is recognized.

        Args:
            biome: Biome name to validate.

        Raises:
            ValueError: If biome is not in the calibration table.
        """
        if biome not in BIOME_CALIBRATION_OFFSETS:
            raise ValueError(
                f"Unknown biome '{biome}'. Valid biomes: "
                f"{list(BIOME_CALIBRATION_OFFSETS.keys())}"
            )


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "CanopyHeightModeler",
    "CanopyHeightEstimate",
    "SourceHeightEstimate",
    "BatchHeightResult",
    "FAO_HEIGHT_THRESHOLD_M",
    "DEFAULT_FUSION_WEIGHTS",
    "SOURCE_UNCERTAINTY_M",
    "SOURCE_RESOLUTION_M",
    "SOURCE_COVERAGE",
    "BIOME_CALIBRATION_OFFSETS",
    "BIOME_HEIGHT_RANGES",
    "GEDI_FOOTPRINT_M",
    "GEDI_ACCURACY_M",
    "GEDI_RH_METRICS",
    "GEDI_DEFAULT_METRIC",
    "ICESAT2_SEGMENT_M",
    "ICESAT2_ACCURACY_M",
    "TEXTURE_REGRESSION_COEFFICIENTS",
    "TEXTURE_R_SQUARED",
]
