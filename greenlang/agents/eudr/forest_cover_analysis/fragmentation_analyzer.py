# -*- coding: utf-8 -*-
"""
FragmentationAnalyzer - AGENT-EUDR-004: Forest Cover Analysis (Engine 2)

Analyzes landscape-level forest fragmentation from binary forest/non-forest
masks to assess habitat quality, deforestation risk, and ecological integrity.
Computes six standard landscape ecology metrics: patch count, edge density,
core area, connectivity, shape complexity, and effective mesh size.

Fragmentation is a key indicator of forest degradation for EUDR compliance.
Even without outright deforestation, progressive fragmentation indicates
ongoing degradation that undermines ecosystem function and threatens
biodiversity corridors critical for EUDR-covered commodities.

Fragmentation Metrics (6):
    1. Patch Count: Number of discrete forest patches (8-connectivity).
    2. Edge Density: Total forest edge length / total landscape area.
    3. Core Area: Forest area excluding edge buffer (default 100m).
    4. Connectivity: Mean nearest-neighbor distance between patches.
    5. Shape Complexity: Mean perimeter-area ratio across patches.
    6. Effective Mesh Size: Sum of patch areas squared / total area.

Fragmentation Levels:
    INTACT:                  Single patch, >80% core, edge density <50
    SLIGHTLY_FRAGMENTED:     <=3 patches, >60% core
    MODERATELY_FRAGMENTED:   <=10 patches, >40% core
    HIGHLY_FRAGMENTED:       <=25 patches, >20% core
    SEVERELY_FRAGMENTED:     >25 patches OR <20% core

Zero-Hallucination Guarantees:
    - All metrics computed with deterministic integer/float arithmetic.
    - Patch identification uses standard flood-fill (BFS, 8-connectivity).
    - Edge detection uses deterministic neighbor counting.
    - Core area uses morphological erosion (pixel-level buffer).
    - No ML/LLM involvement in any metric calculation.
    - SHA-256 provenance hashes on all result objects.

Performance Targets:
    - Single plot fragmentation analysis (100x100 grid): <100ms
    - Batch analysis (100 plots): <5 seconds
    - Individual metric calculation: <20ms

Regulatory References:
    - EUDR Article 2(1): Forest degradation monitoring
    - EUDR Article 2(3): Degradation as reduction of ecosystem services
    - EUDR Article 9: Spatial analysis evidence requirements
    - EUDR Article 10: Risk assessment from landscape condition
    - FAO FRA 2020: Forest fragmentation as degradation indicator

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 2: Fragmentation Analysis)
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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _generate_id(prefix: str = "frag") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default pixel resolution in metres (Sentinel-2 10m).
DEFAULT_PIXEL_SIZE_M: float = 10.0

#: Default edge buffer in metres for core area calculation.
DEFAULT_EDGE_BUFFER_M: float = 100.0

#: 8-connectivity neighbor offsets (row_delta, col_delta).
NEIGHBORS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

#: 4-connectivity neighbor offsets for edge detection.
NEIGHBORS_4: List[Tuple[int, int]] = [
    (-1, 0), (0, -1), (0, 1), (1, 0),
]

# ---------------------------------------------------------------------------
# Constants: Fragmentation Classification Thresholds
# ---------------------------------------------------------------------------

#: Maximum patch count for INTACT classification.
INTACT_MAX_PATCHES: int = 1

#: Minimum core area percentage for INTACT classification.
INTACT_MIN_CORE_PCT: float = 80.0

#: Maximum edge density (m/ha) for INTACT classification.
INTACT_MAX_EDGE_DENSITY: float = 50.0

#: Maximum patch count for SLIGHTLY_FRAGMENTED.
SLIGHT_MAX_PATCHES: int = 3

#: Minimum core area percentage for SLIGHTLY_FRAGMENTED.
SLIGHT_MIN_CORE_PCT: float = 60.0

#: Maximum patch count for MODERATELY_FRAGMENTED.
MODERATE_MAX_PATCHES: int = 10

#: Minimum core area percentage for MODERATELY_FRAGMENTED.
MODERATE_MIN_CORE_PCT: float = 40.0

#: Maximum patch count for HIGHLY_FRAGMENTED.
HIGH_MAX_PATCHES: int = 25

#: Minimum core area percentage for HIGHLY_FRAGMENTED.
HIGH_MIN_CORE_PCT: float = 20.0

# ---------------------------------------------------------------------------
# Constants: Fragmentation Risk Mapping
# ---------------------------------------------------------------------------

#: Risk score for each fragmentation level (0-1 scale).
FRAGMENTATION_RISK_SCORES: Dict[str, float] = {
    "INTACT": 0.05,
    "SLIGHTLY_FRAGMENTED": 0.20,
    "MODERATELY_FRAGMENTED": 0.45,
    "HIGHLY_FRAGMENTED": 0.70,
    "SEVERELY_FRAGMENTED": 0.95,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PatchInfo:
    """Information about a single forest patch.

    Attributes:
        patch_id: Unique identifier for this patch.
        pixel_count: Number of pixels in the patch.
        area_ha: Area in hectares.
        perimeter_pixels: Number of edge pixels.
        perimeter_m: Perimeter length in metres.
        centroid_row: Centroid row coordinate.
        centroid_col: Centroid column coordinate.
        bounding_box: (min_row, min_col, max_row, max_col).
        is_core: Whether this patch contains core area.
        core_pixel_count: Number of core (non-edge) pixels.
        par: Perimeter-area ratio.
    """

    patch_id: int = 0
    pixel_count: int = 0
    area_ha: float = 0.0
    perimeter_pixels: int = 0
    perimeter_m: float = 0.0
    centroid_row: float = 0.0
    centroid_col: float = 0.0
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    is_core: bool = False
    core_pixel_count: int = 0
    par: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "patch_id": self.patch_id,
            "pixel_count": self.pixel_count,
            "area_ha": round(self.area_ha, 4),
            "perimeter_pixels": self.perimeter_pixels,
            "perimeter_m": round(self.perimeter_m, 2),
            "centroid_row": round(self.centroid_row, 2),
            "centroid_col": round(self.centroid_col, 2),
            "bounding_box": list(self.bounding_box),
            "is_core": self.is_core,
            "core_pixel_count": self.core_pixel_count,
            "par": round(self.par, 6),
        }


@dataclass
class FragmentationMetrics:
    """Complete set of fragmentation metrics for a landscape.

    Attributes:
        analysis_id: Unique identifier for this analysis.
        plot_id: Plot/landscape identifier.
        total_pixels: Total pixels in the analysis grid.
        forest_pixels: Number of forest pixels.
        non_forest_pixels: Number of non-forest pixels.
        forest_area_ha: Total forest area in hectares.
        total_area_ha: Total landscape area in hectares.
        forest_cover_pct: Forest cover percentage.
        pixel_size_m: Pixel edge length in metres.
        patch_count: Number of discrete forest patches.
        patches: Detailed info for each patch.
        patch_size_distribution: Dict with min/max/mean/median patch sizes.
        edge_density: Total edge length / total area (m/ha).
        total_edge_m: Total forest edge in metres.
        core_area_ha: Forest core area in hectares (excluding buffer).
        core_area_pct: Core area as percentage of total forest.
        edge_buffer_m: Buffer distance used for core area.
        mean_nn_distance_m: Mean nearest-neighbor distance between patches.
        min_nn_distance_m: Minimum nearest-neighbor distance.
        max_nn_distance_m: Maximum nearest-neighbor distance.
        mean_par: Mean perimeter-area ratio.
        effective_mesh_size_ha: Effective mesh size in hectares.
        fragmentation_level: Classification level.
        deforestation_risk_score: Risk score (0-1) from fragmentation.
        processing_time_ms: Processing time in milliseconds.
        created_at: Timestamp of analysis.
        provenance_hash: SHA-256 provenance hash.
    """

    analysis_id: str = ""
    plot_id: str = ""
    total_pixels: int = 0
    forest_pixels: int = 0
    non_forest_pixels: int = 0
    forest_area_ha: float = 0.0
    total_area_ha: float = 0.0
    forest_cover_pct: float = 0.0
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    patch_count: int = 0
    patches: List[PatchInfo] = field(default_factory=list)
    patch_size_distribution: Dict[str, float] = field(default_factory=dict)
    edge_density: float = 0.0
    total_edge_m: float = 0.0
    core_area_ha: float = 0.0
    core_area_pct: float = 0.0
    edge_buffer_m: float = DEFAULT_EDGE_BUFFER_M
    mean_nn_distance_m: float = 0.0
    min_nn_distance_m: float = 0.0
    max_nn_distance_m: float = 0.0
    mean_par: float = 0.0
    effective_mesh_size_ha: float = 0.0
    fragmentation_level: str = ""
    deforestation_risk_score: float = 0.0
    processing_time_ms: float = 0.0
    created_at: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and reporting."""
        return {
            "analysis_id": self.analysis_id,
            "plot_id": self.plot_id,
            "total_pixels": self.total_pixels,
            "forest_pixels": self.forest_pixels,
            "forest_area_ha": round(self.forest_area_ha, 4),
            "total_area_ha": round(self.total_area_ha, 4),
            "forest_cover_pct": round(self.forest_cover_pct, 2),
            "pixel_size_m": self.pixel_size_m,
            "patch_count": self.patch_count,
            "edge_density": round(self.edge_density, 4),
            "total_edge_m": round(self.total_edge_m, 2),
            "core_area_ha": round(self.core_area_ha, 4),
            "core_area_pct": round(self.core_area_pct, 2),
            "edge_buffer_m": self.edge_buffer_m,
            "mean_nn_distance_m": round(self.mean_nn_distance_m, 2),
            "mean_par": round(self.mean_par, 6),
            "effective_mesh_size_ha": round(self.effective_mesh_size_ha, 4),
            "fragmentation_level": self.fragmentation_level,
            "deforestation_risk_score": round(self.deforestation_risk_score, 3),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "created_at": self.created_at,
        }


@dataclass
class FragmentationComparison:
    """Comparison of fragmentation between two time periods.

    Attributes:
        comparison_id: Unique identifier.
        plot_id: Plot identifier.
        baseline_metrics: Metrics at cutoff date.
        current_metrics: Current metrics.
        delta_patch_count: Change in patch count.
        delta_edge_density: Change in edge density.
        delta_core_area_pct: Change in core area percentage.
        delta_mesh_size_ha: Change in effective mesh size.
        delta_risk_score: Change in risk score.
        baseline_level: Fragmentation level at cutoff.
        current_level: Current fragmentation level.
        trend: Overall trend (IMPROVING, STABLE, WORSENING).
        provenance_hash: SHA-256 provenance hash.
    """

    comparison_id: str = ""
    plot_id: str = ""
    baseline_metrics: Optional[FragmentationMetrics] = None
    current_metrics: Optional[FragmentationMetrics] = None
    delta_patch_count: int = 0
    delta_edge_density: float = 0.0
    delta_core_area_pct: float = 0.0
    delta_mesh_size_ha: float = 0.0
    delta_risk_score: float = 0.0
    baseline_level: str = ""
    current_level: str = ""
    trend: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "comparison_id": self.comparison_id,
            "plot_id": self.plot_id,
            "delta_patch_count": self.delta_patch_count,
            "delta_edge_density": round(self.delta_edge_density, 4),
            "delta_core_area_pct": round(self.delta_core_area_pct, 2),
            "delta_mesh_size_ha": round(self.delta_mesh_size_ha, 4),
            "delta_risk_score": round(self.delta_risk_score, 3),
            "baseline_level": self.baseline_level,
            "current_level": self.current_level,
            "trend": self.trend,
        }


# ---------------------------------------------------------------------------
# FragmentationAnalyzer
# ---------------------------------------------------------------------------


class FragmentationAnalyzer:
    """Landscape-level forest fragmentation analysis engine.

    Analyzes binary forest/non-forest masks to compute six standard
    landscape ecology metrics and classify the fragmentation level.
    All calculations are deterministic with SHA-256 provenance hashing.

    The input is a 2D grid (list of lists) where True/1 indicates forest
    and False/0 indicates non-forest. Default pixel resolution is 10m
    (Sentinel-2 based classification).

    Example::

        analyzer = FragmentationAnalyzer()
        forest_mask = [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
        metrics = analyzer.analyze_plot("PLOT-001", forest_mask)
        assert metrics.patch_count == 2
        assert metrics.fragmentation_level != ""
        assert metrics.provenance_hash != ""

    Attributes:
        pixel_size_m: Pixel edge length in metres.
        edge_buffer_m: Edge buffer for core area calculation.
    """

    def __init__(
        self,
        pixel_size_m: float = DEFAULT_PIXEL_SIZE_M,
        edge_buffer_m: float = DEFAULT_EDGE_BUFFER_M,
    ) -> None:
        """Initialize the FragmentationAnalyzer.

        Args:
            pixel_size_m: Pixel edge length in metres. Must be > 0.
            edge_buffer_m: Buffer distance for core area calculation
                in metres. Must be >= 0.

        Raises:
            ValueError: If pixel_size_m <= 0 or edge_buffer_m < 0.
        """
        if pixel_size_m <= 0:
            raise ValueError(
                f"pixel_size_m must be > 0, got {pixel_size_m}"
            )
        if edge_buffer_m < 0:
            raise ValueError(
                f"edge_buffer_m must be >= 0, got {edge_buffer_m}"
            )

        self.pixel_size_m = pixel_size_m
        self.edge_buffer_m = edge_buffer_m

        self._buffer_pixels = max(1, int(round(edge_buffer_m / pixel_size_m)))

        logger.info(
            "FragmentationAnalyzer initialized: pixel_size=%.1fm, "
            "edge_buffer=%.1fm (%d pixels)",
            pixel_size_m, edge_buffer_m, self._buffer_pixels,
        )

    # ------------------------------------------------------------------
    # Public API: Metric 1 - Patch Count
    # ------------------------------------------------------------------

    def count_patches(
        self,
        forest_mask: List[List[int]],
    ) -> Tuple[int, List[PatchInfo]]:
        """Count discrete forest patches using 8-connectivity BFS.

        A patch is a contiguous group of forest pixels (value 1) where
        pixels are connected via 8-connectivity (horizontally, vertically,
        and diagonally adjacent).

        Args:
            forest_mask: 2D grid of 0 (non-forest) and 1 (forest).

        Returns:
            Tuple of (patch_count, list of PatchInfo).

        Raises:
            ValueError: If forest_mask is empty.
        """
        start_time = time.monotonic()

        self._validate_mask(forest_mask)

        n_rows = len(forest_mask)
        n_cols = len(forest_mask[0])

        visited: List[List[bool]] = [
            [False] * n_cols for _ in range(n_rows)
        ]

        patches: List[PatchInfo] = []
        patch_id = 0

        for r in range(n_rows):
            for c in range(n_cols):
                if forest_mask[r][c] == 1 and not visited[r][c]:
                    patch_id += 1
                    pixels = self._bfs_flood_fill(
                        forest_mask, visited, r, c, n_rows, n_cols
                    )
                    patch_info = self._build_patch_info(
                        patch_id, pixels, forest_mask, n_rows, n_cols
                    )
                    patches.append(patch_info)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Patch count: %d patches found in %dx%d grid, %.2fms",
            len(patches), n_rows, n_cols, elapsed_ms,
        )

        return len(patches), patches

    # ------------------------------------------------------------------
    # Public API: Metric 2 - Edge Density
    # ------------------------------------------------------------------

    def compute_edge_density(
        self,
        forest_mask: List[List[int]],
    ) -> Tuple[float, float]:
        """Compute edge density: total forest edge length / total area.

        An edge pixel is any forest pixel (1) that has at least one
        non-forest neighbor (0) or is on the grid boundary, using
        4-connectivity for edge detection.

        Edge density = total_edge_m / total_area_ha.

        Args:
            forest_mask: 2D grid of 0/1.

        Returns:
            Tuple of (edge_density_m_per_ha, total_edge_m).

        Raises:
            ValueError: If forest_mask is empty.
        """
        start_time = time.monotonic()

        self._validate_mask(forest_mask)

        n_rows = len(forest_mask)
        n_cols = len(forest_mask[0])

        edge_segments = 0

        for r in range(n_rows):
            for c in range(n_cols):
                if forest_mask[r][c] != 1:
                    continue

                for dr, dc in NEIGHBORS_4:
                    nr, nc = r + dr, c + dc
                    if (
                        nr < 0 or nr >= n_rows
                        or nc < 0 or nc >= n_cols
                        or forest_mask[nr][nc] != 1
                    ):
                        edge_segments += 1

        total_edge_m = edge_segments * self.pixel_size_m
        total_area_ha = (n_rows * n_cols * self.pixel_size_m ** 2) / 10_000.0

        if total_area_ha > 0:
            edge_density = total_edge_m / total_area_ha
        else:
            edge_density = 0.0

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Edge density: %.2f m/ha (total_edge=%.1fm, "
            "area=%.4f ha), %.2fms",
            edge_density, total_edge_m, total_area_ha, elapsed_ms,
        )

        return round(edge_density, 4), round(total_edge_m, 2)

    # ------------------------------------------------------------------
    # Public API: Metric 3 - Core Area
    # ------------------------------------------------------------------

    def compute_core_area(
        self,
        forest_mask: List[List[int]],
        buffer_pixels: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute core forest area excluding edge buffer.

        Core area is the forest interior remaining after removing a
        buffer of specified width from all forest edges. This represents
        the interior habitat unaffected by edge effects.

        Core area % = core_area / total_forest_area * 100.

        The algorithm uses morphological erosion: a forest pixel is
        core only if all pixels within the buffer distance are also
        forest.

        Args:
            forest_mask: 2D grid of 0/1.
            buffer_pixels: Override buffer width in pixels. If None,
                uses the engine default (edge_buffer_m / pixel_size_m).

        Returns:
            Tuple of (core_area_ha, core_area_pct).

        Raises:
            ValueError: If forest_mask is empty.
        """
        start_time = time.monotonic()

        self._validate_mask(forest_mask)

        buf = buffer_pixels if buffer_pixels is not None else self._buffer_pixels
        n_rows = len(forest_mask)
        n_cols = len(forest_mask[0])

        distance_to_edge = self._compute_distance_to_non_forest(
            forest_mask, n_rows, n_cols
        )

        total_forest = 0
        core_count = 0
        pixel_area_ha = (self.pixel_size_m ** 2) / 10_000.0

        for r in range(n_rows):
            for c in range(n_cols):
                if forest_mask[r][c] == 1:
                    total_forest += 1
                    if distance_to_edge[r][c] >= buf:
                        core_count += 1

        core_area_ha = core_count * pixel_area_ha
        total_forest_ha = total_forest * pixel_area_ha

        if total_forest_ha > 0:
            core_pct = (core_area_ha / total_forest_ha) * 100.0
        else:
            core_pct = 0.0

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Core area: %.4f ha (%.1f%% of %.4f ha forest, "
            "buffer=%d px), %.2fms",
            core_area_ha, core_pct, total_forest_ha, buf, elapsed_ms,
        )

        return round(core_area_ha, 4), round(core_pct, 2)

    # ------------------------------------------------------------------
    # Public API: Metric 4 - Connectivity
    # ------------------------------------------------------------------

    def compute_connectivity(
        self,
        patches: List[PatchInfo],
    ) -> Tuple[float, float, float]:
        """Compute mean nearest-neighbor distance between patch centroids.

        Uses Euclidean distance between patch centroids scaled by
        pixel size. Lower distance indicates better connectivity
        between forest patches.

        Args:
            patches: List of PatchInfo with centroid coordinates.

        Returns:
            Tuple of (mean_nn_distance_m, min_nn_distance_m,
            max_nn_distance_m). Returns (0, 0, 0) if fewer than 2
            patches.
        """
        start_time = time.monotonic()

        if len(patches) < 2:
            logger.debug(
                "Connectivity: < 2 patches (%d), returning 0.0",
                len(patches),
            )
            return 0.0, 0.0, 0.0

        nn_distances: List[float] = []

        for i, patch_a in enumerate(patches):
            min_dist = float("inf")
            for j, patch_b in enumerate(patches):
                if i == j:
                    continue
                dist_pixels = math.sqrt(
                    (patch_a.centroid_row - patch_b.centroid_row) ** 2
                    + (patch_a.centroid_col - patch_b.centroid_col) ** 2
                )
                dist_m = dist_pixels * self.pixel_size_m
                if dist_m < min_dist:
                    min_dist = dist_m
            nn_distances.append(min_dist)

        mean_nn = sum(nn_distances) / len(nn_distances)
        min_nn = min(nn_distances)
        max_nn = max(nn_distances)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Connectivity: mean_nn=%.2fm, min=%.2fm, max=%.2fm, "
            "%d patches, %.2fms",
            mean_nn, min_nn, max_nn, len(patches), elapsed_ms,
        )

        return round(mean_nn, 2), round(min_nn, 2), round(max_nn, 2)

    # ------------------------------------------------------------------
    # Public API: Metric 5 - Shape Complexity
    # ------------------------------------------------------------------

    def compute_shape_complexity(
        self,
        patches: List[PatchInfo],
    ) -> float:
        """Compute mean perimeter-area ratio (PAR) across all patches.

        PAR = perimeter / area for each patch. A circle has the minimum
        PAR for a given area. Higher PAR indicates more irregular, complex
        shapes typical of fragmented or disturbed landscapes.

        Args:
            patches: List of PatchInfo with area and perimeter.

        Returns:
            Mean PAR across all patches. Returns 0.0 if no patches.
        """
        start_time = time.monotonic()

        if not patches:
            return 0.0

        par_values: List[float] = []
        for patch in patches:
            if patch.area_ha > 0:
                par = patch.perimeter_m / (patch.area_ha * 10_000.0)
                par_values.append(par)

        if not par_values:
            return 0.0

        mean_par = sum(par_values) / len(par_values)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Shape complexity: mean_par=%.6f, %d patches, %.2fms",
            mean_par, len(patches), elapsed_ms,
        )

        return round(mean_par, 6)

    # ------------------------------------------------------------------
    # Public API: Metric 6 - Effective Mesh Size
    # ------------------------------------------------------------------

    def compute_effective_mesh_size(
        self,
        patches: List[PatchInfo],
        total_area_ha: float,
    ) -> float:
        """Compute effective mesh size (MESH) landscape division index.

        MESH = sum(patch_area_i^2) / total_landscape_area

        This metric quantifies landscape division. High MESH indicates
        large, connected patches (low fragmentation). Low MESH indicates
        the landscape is divided into many small patches (high
        fragmentation).

        Args:
            patches: List of PatchInfo with area_ha.
            total_area_ha: Total landscape area in hectares.

        Returns:
            Effective mesh size in hectares. Returns 0.0 if total area
            is zero.
        """
        start_time = time.monotonic()

        if total_area_ha <= 0 or not patches:
            return 0.0

        sum_area_sq = sum(p.area_ha ** 2 for p in patches)
        mesh = sum_area_sq / total_area_ha

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Effective mesh size: %.4f ha, %d patches, "
            "total_area=%.4f ha, %.2fms",
            mesh, len(patches), total_area_ha, elapsed_ms,
        )

        return round(mesh, 4)

    # ------------------------------------------------------------------
    # Public API: Fragmentation Classification
    # ------------------------------------------------------------------

    def classify_fragmentation(
        self,
        patch_count: int,
        core_area_pct: float,
        edge_density: float,
    ) -> str:
        """Classify fragmentation level from metric combination.

        Classification Rules (evaluated in order):
            1. INTACT: patch_count <= 1, core_area > 80%, edge_density < 50
            2. SLIGHTLY_FRAGMENTED: patch_count <= 3, core_area > 60%
            3. MODERATELY_FRAGMENTED: patch_count <= 10, core_area > 40%
            4. HIGHLY_FRAGMENTED: patch_count <= 25, core_area > 20%
            5. SEVERELY_FRAGMENTED: all other cases

        Args:
            patch_count: Number of forest patches.
            core_area_pct: Core area as percentage of total forest.
            edge_density: Edge density in m/ha.

        Returns:
            Fragmentation level string.
        """
        if (
            patch_count <= INTACT_MAX_PATCHES
            and core_area_pct >= INTACT_MIN_CORE_PCT
            and edge_density < INTACT_MAX_EDGE_DENSITY
        ):
            return "INTACT"

        if (
            patch_count <= SLIGHT_MAX_PATCHES
            and core_area_pct >= SLIGHT_MIN_CORE_PCT
        ):
            return "SLIGHTLY_FRAGMENTED"

        if (
            patch_count <= MODERATE_MAX_PATCHES
            and core_area_pct >= MODERATE_MIN_CORE_PCT
        ):
            return "MODERATELY_FRAGMENTED"

        if (
            patch_count <= HIGH_MAX_PATCHES
            and core_area_pct >= HIGH_MIN_CORE_PCT
        ):
            return "HIGHLY_FRAGMENTED"

        return "SEVERELY_FRAGMENTED"

    # ------------------------------------------------------------------
    # Public API: Risk Assessment
    # ------------------------------------------------------------------

    def assess_risk_from_fragmentation(
        self,
        fragmentation_level: str,
        core_area_pct: float = 100.0,
        patch_count: int = 1,
    ) -> float:
        """Map fragmentation level to deforestation risk score.

        Higher fragmentation correlates with higher risk of further
        deforestation. The base risk comes from the level classification
        and is modulated by core area and patch count.

        Args:
            fragmentation_level: Classification level string.
            core_area_pct: Core area percentage for fine adjustment.
            patch_count: Patch count for fine adjustment.

        Returns:
            Risk score between 0.0 (no risk) and 1.0 (maximum risk).
        """
        base_risk = FRAGMENTATION_RISK_SCORES.get(
            fragmentation_level, 0.5
        )

        if core_area_pct > 0:
            core_modifier = max(0.0, (100.0 - core_area_pct) / 200.0)
        else:
            core_modifier = 0.5

        if patch_count > 1:
            patch_modifier = min(0.1, patch_count / 500.0)
        else:
            patch_modifier = 0.0

        risk = base_risk + core_modifier * 0.1 + patch_modifier
        return round(min(1.0, max(0.0, risk)), 3)

    # ------------------------------------------------------------------
    # Public API: Main Entry Point
    # ------------------------------------------------------------------

    def analyze_plot(
        self,
        plot_id: str,
        forest_mask: List[List[int]],
        edge_buffer_m: Optional[float] = None,
    ) -> FragmentationMetrics:
        """Perform complete fragmentation analysis on a single plot.

        Computes all six fragmentation metrics, classifies the
        fragmentation level, and assesses deforestation risk.

        Args:
            plot_id: Unique identifier for the plot.
            forest_mask: 2D grid of 0 (non-forest) and 1 (forest).
            edge_buffer_m: Optional override for edge buffer distance.

        Returns:
            FragmentationMetrics with all indices, level, and risk.

        Raises:
            ValueError: If forest_mask is empty.
        """
        start_time = time.monotonic()

        self._validate_mask(forest_mask)

        if edge_buffer_m is not None:
            buffer_px = max(1, int(round(edge_buffer_m / self.pixel_size_m)))
            effective_buffer_m = edge_buffer_m
        else:
            buffer_px = self._buffer_pixels
            effective_buffer_m = self.edge_buffer_m

        n_rows = len(forest_mask)
        n_cols = len(forest_mask[0])
        pixel_area_ha = (self.pixel_size_m ** 2) / 10_000.0
        total_area_ha = n_rows * n_cols * pixel_area_ha

        forest_pixels = sum(
            1 for r in range(n_rows) for c in range(n_cols)
            if forest_mask[r][c] == 1
        )
        non_forest_pixels = (n_rows * n_cols) - forest_pixels
        forest_area_ha = forest_pixels * pixel_area_ha
        forest_cover_pct = (
            (forest_pixels / (n_rows * n_cols) * 100.0)
            if (n_rows * n_cols) > 0 else 0.0
        )

        patch_count, patches = self.count_patches(forest_mask)

        edge_density, total_edge_m = self.compute_edge_density(forest_mask)

        core_area_ha, core_area_pct = self.compute_core_area(
            forest_mask, buffer_pixels=buffer_px
        )

        mean_nn, min_nn, max_nn = self.compute_connectivity(patches)

        mean_par = self.compute_shape_complexity(patches)

        mesh_size = self.compute_effective_mesh_size(patches, total_area_ha)

        frag_level = self.classify_fragmentation(
            patch_count, core_area_pct, edge_density
        )

        risk_score = self.assess_risk_from_fragmentation(
            frag_level, core_area_pct, patch_count
        )

        size_dist = self._compute_size_distribution(patches)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = FragmentationMetrics(
            analysis_id=_generate_id("frag"),
            plot_id=plot_id,
            total_pixels=n_rows * n_cols,
            forest_pixels=forest_pixels,
            non_forest_pixels=non_forest_pixels,
            forest_area_ha=round(forest_area_ha, 4),
            total_area_ha=round(total_area_ha, 4),
            forest_cover_pct=round(forest_cover_pct, 2),
            pixel_size_m=self.pixel_size_m,
            patch_count=patch_count,
            patches=patches,
            patch_size_distribution=size_dist,
            edge_density=edge_density,
            total_edge_m=total_edge_m,
            core_area_ha=core_area_ha,
            core_area_pct=core_area_pct,
            edge_buffer_m=effective_buffer_m,
            mean_nn_distance_m=mean_nn,
            min_nn_distance_m=min_nn,
            max_nn_distance_m=max_nn,
            mean_par=mean_par,
            effective_mesh_size_ha=mesh_size,
            fragmentation_level=frag_level,
            deforestation_risk_score=risk_score,
            processing_time_ms=round(elapsed_ms, 2),
            created_at=str(_utcnow()),
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Plot '%s' fragmentation: level=%s, patches=%d, "
            "edge_density=%.2f m/ha, core=%.1f%%, MESH=%.4f ha, "
            "risk=%.3f, %.2fms",
            plot_id, frag_level, patch_count, edge_density,
            core_area_pct, mesh_size, risk_score, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Temporal Comparison
    # ------------------------------------------------------------------

    def compare_fragmentation(
        self,
        plot_id: str,
        baseline_mask: List[List[int]],
        current_mask: List[List[int]],
        edge_buffer_m: Optional[float] = None,
    ) -> FragmentationComparison:
        """Compare fragmentation between two time periods.

        Computes fragmentation metrics for both the baseline (cutoff date)
        and current masks, then calculates deltas and determines the trend.

        Args:
            plot_id: Plot identifier.
            baseline_mask: Forest mask at cutoff date.
            current_mask: Current forest mask.
            edge_buffer_m: Optional edge buffer override.

        Returns:
            FragmentationComparison with deltas and trend.

        Raises:
            ValueError: If masks have different dimensions.
        """
        start_time = time.monotonic()

        if len(baseline_mask) != len(current_mask):
            raise ValueError(
                f"Baseline mask ({len(baseline_mask)} rows) and current "
                f"mask ({len(current_mask)} rows) must have same dimensions"
            )
        if baseline_mask and current_mask:
            if len(baseline_mask[0]) != len(current_mask[0]):
                raise ValueError(
                    f"Baseline mask ({len(baseline_mask[0])} cols) and "
                    f"current mask ({len(current_mask[0])} cols) must "
                    f"have same dimensions"
                )

        baseline_metrics = self.analyze_plot(
            f"{plot_id}_baseline", baseline_mask, edge_buffer_m
        )
        current_metrics = self.analyze_plot(
            f"{plot_id}_current", current_mask, edge_buffer_m
        )

        delta_patches = (
            current_metrics.patch_count - baseline_metrics.patch_count
        )
        delta_edge = (
            current_metrics.edge_density - baseline_metrics.edge_density
        )
        delta_core = (
            current_metrics.core_area_pct - baseline_metrics.core_area_pct
        )
        delta_mesh = (
            current_metrics.effective_mesh_size_ha
            - baseline_metrics.effective_mesh_size_ha
        )
        delta_risk = (
            current_metrics.deforestation_risk_score
            - baseline_metrics.deforestation_risk_score
        )

        trend = self._determine_trend(
            delta_patches, delta_core, delta_mesh, delta_risk
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        comparison = FragmentationComparison(
            comparison_id=_generate_id("frag-cmp"),
            plot_id=plot_id,
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            delta_patch_count=delta_patches,
            delta_edge_density=round(delta_edge, 4),
            delta_core_area_pct=round(delta_core, 2),
            delta_mesh_size_ha=round(delta_mesh, 4),
            delta_risk_score=round(delta_risk, 3),
            baseline_level=baseline_metrics.fragmentation_level,
            current_level=current_metrics.fragmentation_level,
            trend=trend,
        )
        comparison.provenance_hash = _compute_hash(comparison.to_dict())

        logger.info(
            "Plot '%s' fragmentation comparison: %s -> %s (trend=%s, "
            "delta_patches=%+d, delta_core=%+.1f%%), %.2fms",
            plot_id, baseline_metrics.fragmentation_level,
            current_metrics.fragmentation_level, trend,
            delta_patches, delta_core, elapsed_ms,
        )

        return comparison

    # ------------------------------------------------------------------
    # Internal: BFS Flood Fill
    # ------------------------------------------------------------------

    def _bfs_flood_fill(
        self,
        mask: List[List[int]],
        visited: List[List[bool]],
        start_r: int,
        start_c: int,
        n_rows: int,
        n_cols: int,
    ) -> List[Tuple[int, int]]:
        """Flood-fill from a starting pixel using BFS and 8-connectivity.

        Args:
            mask: Forest mask grid.
            visited: Visited tracking grid (modified in place).
            start_r: Starting row.
            start_c: Starting column.
            n_rows: Grid row count.
            n_cols: Grid column count.

        Returns:
            List of (row, col) tuples belonging to this patch.
        """
        queue: deque[Tuple[int, int]] = deque()
        queue.append((start_r, start_c))
        visited[start_r][start_c] = True
        pixels: List[Tuple[int, int]] = []

        while queue:
            r, c = queue.popleft()
            pixels.append((r, c))

            for dr, dc in NEIGHBORS_8:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < n_rows
                    and 0 <= nc < n_cols
                    and not visited[nr][nc]
                    and mask[nr][nc] == 1
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return pixels

    # ------------------------------------------------------------------
    # Internal: Patch Info Builder
    # ------------------------------------------------------------------

    def _build_patch_info(
        self,
        patch_id: int,
        pixels: List[Tuple[int, int]],
        mask: List[List[int]],
        n_rows: int,
        n_cols: int,
    ) -> PatchInfo:
        """Build PatchInfo from a list of pixel coordinates.

        Args:
            patch_id: Patch identifier.
            pixels: List of (row, col) coordinates.
            mask: Forest mask grid.
            n_rows: Grid row count.
            n_cols: Grid column count.

        Returns:
            Populated PatchInfo.
        """
        pixel_set: Set[Tuple[int, int]] = set(pixels)
        pixel_count = len(pixels)
        pixel_area_ha = (self.pixel_size_m ** 2) / 10_000.0
        area_ha = pixel_count * pixel_area_ha

        perimeter_pixels = 0
        for r, c in pixels:
            for dr, dc in NEIGHBORS_4:
                nr, nc = r + dr, c + dc
                if (
                    nr < 0 or nr >= n_rows
                    or nc < 0 or nc >= n_cols
                    or (nr, nc) not in pixel_set
                ):
                    perimeter_pixels += 1

        perimeter_m = perimeter_pixels * self.pixel_size_m

        sum_r = sum(r for r, c in pixels)
        sum_c = sum(c for r, c in pixels)
        centroid_row = sum_r / pixel_count if pixel_count > 0 else 0.0
        centroid_col = sum_c / pixel_count if pixel_count > 0 else 0.0

        rows = [r for r, c in pixels]
        cols = [c for r, c in pixels]
        bbox = (min(rows), min(cols), max(rows), max(cols))

        if area_ha > 0:
            par = perimeter_m / (area_ha * 10_000.0)
        else:
            par = 0.0

        return PatchInfo(
            patch_id=patch_id,
            pixel_count=pixel_count,
            area_ha=round(area_ha, 4),
            perimeter_pixels=perimeter_pixels,
            perimeter_m=round(perimeter_m, 2),
            centroid_row=round(centroid_row, 2),
            centroid_col=round(centroid_col, 2),
            bounding_box=bbox,
            par=round(par, 6),
        )

    # ------------------------------------------------------------------
    # Internal: Distance to Non-Forest (for Core Area)
    # ------------------------------------------------------------------

    def _compute_distance_to_non_forest(
        self,
        mask: List[List[int]],
        n_rows: int,
        n_cols: int,
    ) -> List[List[int]]:
        """Compute Manhattan distance from each forest pixel to nearest edge.

        Uses multi-source BFS from all non-forest/boundary pixels to
        compute the minimum distance (in pixels) from each forest pixel
        to the nearest non-forest pixel. Forest pixels with distance >=
        buffer_pixels are considered core.

        Args:
            mask: Forest mask grid.
            n_rows: Grid row count.
            n_cols: Grid column count.

        Returns:
            2D grid of distances (in pixels) to nearest non-forest.
        """
        INF = n_rows + n_cols + 1
        distance: List[List[int]] = [
            [INF] * n_cols for _ in range(n_rows)
        ]

        queue: deque[Tuple[int, int]] = deque()

        for r in range(n_rows):
            for c in range(n_cols):
                if mask[r][c] != 1:
                    distance[r][c] = 0
                    queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in NEIGHBORS_4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < n_rows
                    and 0 <= nc < n_cols
                    and distance[nr][nc] > distance[r][c] + 1
                ):
                    distance[nr][nc] = distance[r][c] + 1
                    queue.append((nr, nc))

        return distance

    # ------------------------------------------------------------------
    # Internal: Size Distribution
    # ------------------------------------------------------------------

    def _compute_size_distribution(
        self,
        patches: List[PatchInfo],
    ) -> Dict[str, float]:
        """Compute patch size distribution statistics.

        Args:
            patches: List of PatchInfo.

        Returns:
            Dict with min, max, mean, median, std_dev patch sizes in ha.
        """
        if not patches:
            return {
                "min_ha": 0.0,
                "max_ha": 0.0,
                "mean_ha": 0.0,
                "median_ha": 0.0,
                "std_dev_ha": 0.0,
            }

        sizes = sorted([p.area_ha for p in patches])
        n = len(sizes)
        mean_s = sum(sizes) / n

        if n % 2 == 1:
            median_s = sizes[n // 2]
        else:
            median_s = (sizes[n // 2 - 1] + sizes[n // 2]) / 2.0

        if n > 1:
            variance = sum((s - mean_s) ** 2 for s in sizes) / (n - 1)
            std_s = math.sqrt(variance)
        else:
            std_s = 0.0

        return {
            "min_ha": round(min(sizes), 4),
            "max_ha": round(max(sizes), 4),
            "mean_ha": round(mean_s, 4),
            "median_ha": round(median_s, 4),
            "std_dev_ha": round(std_s, 4),
        }

    # ------------------------------------------------------------------
    # Internal: Trend Determination
    # ------------------------------------------------------------------

    def _determine_trend(
        self,
        delta_patches: int,
        delta_core_pct: float,
        delta_mesh: float,
        delta_risk: float,
    ) -> str:
        """Determine fragmentation trend from metric deltas.

        Worsening indicators: more patches, less core, lower mesh, higher risk.
        Improving indicators: fewer patches, more core, higher mesh, lower risk.

        Decision uses a simple scoring system:
            +1 for each improving indicator, -1 for each worsening.

        Args:
            delta_patches: Change in patch count.
            delta_core_pct: Change in core area percentage.
            delta_mesh: Change in effective mesh size.
            delta_risk: Change in risk score.

        Returns:
            One of "IMPROVING", "STABLE", "WORSENING".
        """
        score = 0

        if delta_patches < -1:
            score += 1
        elif delta_patches > 1:
            score -= 1

        if delta_core_pct > 2.0:
            score += 1
        elif delta_core_pct < -2.0:
            score -= 1

        if delta_mesh > 0.01:
            score += 1
        elif delta_mesh < -0.01:
            score -= 1

        if delta_risk < -0.05:
            score += 1
        elif delta_risk > 0.05:
            score -= 1

        if score >= 2:
            return "IMPROVING"
        elif score <= -2:
            return "WORSENING"
        else:
            return "STABLE"

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_mask(self, mask: List[List[int]]) -> None:
        """Validate that a forest mask is non-empty and rectangular.

        Args:
            mask: 2D grid to validate.

        Raises:
            ValueError: If mask is empty or rows have inconsistent lengths.
        """
        if not mask:
            raise ValueError("Forest mask is empty (no rows)")
        if not mask[0]:
            raise ValueError("Forest mask is empty (no columns)")

        n_cols = len(mask[0])
        for i, row in enumerate(mask):
            if len(row) != n_cols:
                raise ValueError(
                    f"Forest mask row {i} has {len(row)} columns, "
                    f"expected {n_cols}"
                )


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "FragmentationAnalyzer",
    "FragmentationMetrics",
    "FragmentationComparison",
    "PatchInfo",
    "DEFAULT_PIXEL_SIZE_M",
    "DEFAULT_EDGE_BUFFER_M",
    "NEIGHBORS_8",
    "NEIGHBORS_4",
    "FRAGMENTATION_RISK_SCORES",
    "INTACT_MAX_PATCHES",
    "INTACT_MIN_CORE_PCT",
    "INTACT_MAX_EDGE_DENSITY",
]
