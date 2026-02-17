# -*- coding: utf-8 -*-
"""
Multivariate Outlier Detection Engine - AGENT-DATA-013

Pure-Python multivariate outlier detection methods including Mahalanobis
distance, Isolation Forest, Local Outlier Factor (LOF), and DBSCAN.
All implementations are self-contained with no external ML dependencies.

Zero-Hallucination: All calculations use deterministic Python
arithmetic. No LLM calls for numeric computations.

Example:
    >>> from greenlang.outlier_detector.multivariate_detector import MultivariateDetectorEngine
    >>> engine = MultivariateDetectorEngine()
    >>> results = engine.detect_mahalanobis(
    ...     records=[{"a": 1, "b": 2}, {"a": 100, "b": 200}],
    ...     columns=["a", "b"],
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from greenlang.outlier_detector.config import get_config
from greenlang.outlier_detector.models import (
    DetectionMethod,
    MultivariateResult,
    OutlierScore,
    SeverityLevel,
)
from greenlang.outlier_detector.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _severity_from_score(score: float) -> SeverityLevel:
    """Map normalised score to severity level."""
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    if score >= 0.80:
        return SeverityLevel.HIGH
    if score >= 0.60:
        return SeverityLevel.MEDIUM
    if score >= 0.40:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


class MultivariateDetectorEngine:
    """Pure-Python multivariate outlier detection engine.

    Implements four multivariate detection methods that operate on
    multiple columns simultaneously to detect outliers in
    multidimensional space.

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = MultivariateDetectorEngine()
        >>> results = engine.detect_lof(
        ...     records=[{"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 100, "b": 100}],
        ...     columns=["a", "b"],
        ... )
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MultivariateDetectorEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("MultivariateDetectorEngine initialized")

    # ------------------------------------------------------------------
    # Mahalanobis Distance
    # ------------------------------------------------------------------

    def detect_mahalanobis(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        threshold: Optional[float] = None,
    ) -> MultivariateResult:
        """Detect outliers using Mahalanobis distance.

        Computes the Mahalanobis distance of each point from the
        multivariate mean, using the inverse covariance matrix.
        Points with distance > threshold * sqrt(dimensions) are flagged.

        Args:
            records: List of record dictionaries.
            columns: Columns to use for detection.
            threshold: Distance threshold (default: 3.0).

        Returns:
            MultivariateResult with per-point scores.
        """
        start = time.time()
        t = threshold if threshold is not None else 3.0
        data = self._extract_matrix(records, columns)
        n = len(data)
        d = len(columns)

        if n < d + 2:
            return self._empty_multivariate_result(
                records, columns, DetectionMethod.MAHALANOBIS,
            )

        # Compute mean vector
        means = [0.0] * d
        for row in data:
            for j in range(d):
                means[j] += row[j]
        for j in range(d):
            means[j] /= n

        # Compute covariance matrix and invert
        cov = self._compute_covariance_matrix(data, means)
        inv_cov = self._invert_matrix(cov)

        if inv_cov is None:
            logger.warning("Covariance matrix is singular, using diagonal fallback")
            inv_cov = self._diagonal_inverse(cov, d)

        # Compute Mahalanobis distances
        distances: List[float] = []
        for row in data:
            diff = [row[j] - means[j] for j in range(d)]
            md = self._mahalanobis_distance(diff, inv_cov, d)
            distances.append(md)

        dist_threshold = t * math.sqrt(d)
        max_dist = max(distances) if distances else 1.0

        scores: List[OutlierScore] = []
        outlier_count = 0

        for i, md in enumerate(distances):
            score = min(1.0, md / (dist_threshold * 2.0)) if dist_threshold > 0 else 0.0
            is_outlier = md > dist_threshold

            if is_outlier:
                outlier_count += 1

            provenance_hash = self._provenance.build_hash({
                "method": "mahalanobis", "index": i,
                "distance": md, "threshold": dist_threshold,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name="|".join(columns),
                value=data[i] if i < len(data) else None,
                method=DetectionMethod.MAHALANOBIS,
                score=score,
                is_outlier=is_outlier,
                threshold=dist_threshold,
                severity=_severity_from_score(score),
                details={"mahalanobis_distance": md,
                         "dimensions": d},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        mean_dist = sum(distances) / n if n > 0 else 0.0
        result_hash = self._provenance.build_hash({
            "method": "mahalanobis", "n": n, "d": d,
            "outliers": outlier_count,
        })

        logger.debug("Mahalanobis: %d points, %d outliers, %.3fs",
                      n, outlier_count, elapsed)

        return MultivariateResult(
            method=DetectionMethod.MAHALANOBIS,
            columns=columns,
            total_points=n,
            outliers_found=outlier_count,
            scores=scores,
            distance_threshold=dist_threshold,
            mean_distance=mean_dist,
            max_distance=max_dist,
            confidence=min(1.0, 0.5 + n / 100.0 * 0.5),
            provenance_hash=result_hash,
        )

    # ------------------------------------------------------------------
    # Isolation Forest (Pure Python)
    # ------------------------------------------------------------------

    def detect_isolation_forest(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        n_trees: Optional[int] = None,
        sample_size: int = 256,
    ) -> MultivariateResult:
        """Detect outliers using a pure-Python Isolation Forest.

        Builds an ensemble of isolation trees. Points that are isolated
        in fewer splits have higher anomaly scores.

        Args:
            records: List of record dictionaries.
            columns: Columns to use for detection.
            n_trees: Number of trees (default from config).
            sample_size: Subsample size per tree (default 256).

        Returns:
            MultivariateResult with per-point anomaly scores.
        """
        start = time.time()
        trees = n_trees if n_trees is not None else self._config.isolation_trees
        data = self._extract_matrix(records, columns)
        n = len(data)
        d = len(columns)

        if n < 4:
            return self._empty_multivariate_result(
                records, columns, DetectionMethod.ISOLATION_FOREST,
            )

        max_depth = int(math.ceil(math.log2(max(sample_size, 2))))
        rng = random.Random(42)

        # Build trees
        forest: List[Dict[str, Any]] = []
        for _ in range(trees):
            sample_indices = rng.sample(
                range(n), min(sample_size, n),
            )
            sample = [data[i] for i in sample_indices]
            tree = self._build_isolation_tree(sample, 0, max_depth, d, rng)
            forest.append(tree)

        # Compute anomaly scores
        c_n = self._average_path_length(n)
        scores_list: List[OutlierScore] = []
        outlier_count = 0

        for i, point in enumerate(data):
            total_path = 0.0
            for tree in forest:
                total_path += self._path_length(point, tree, 0, d)
            avg_path = total_path / trees if trees > 0 else 0.0

            # Anomaly score: 2^(-avg_path / c(n))
            if c_n > 0:
                anomaly_score = 2.0 ** (-avg_path / c_n)
            else:
                anomaly_score = 0.5

            anomaly_score = min(1.0, max(0.0, anomaly_score))
            is_outlier = anomaly_score > 0.6

            if is_outlier:
                outlier_count += 1

            provenance_hash = self._provenance.build_hash({
                "method": "isolation_forest", "index": i,
                "avg_path": avg_path, "score": anomaly_score,
            })

            scores_list.append(OutlierScore(
                record_index=i,
                column_name="|".join(columns),
                value=point,
                method=DetectionMethod.ISOLATION_FOREST,
                score=anomaly_score,
                is_outlier=is_outlier,
                threshold=0.6,
                severity=_severity_from_score(anomaly_score),
                details={"avg_path_length": avg_path,
                         "expected_path_length": c_n,
                         "n_trees": trees},
                confidence=min(1.0, 0.4 + anomaly_score * 0.4 + trees / 200.0 * 0.2),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        all_scores = [s.score for s in scores_list]
        result_hash = self._provenance.build_hash({
            "method": "isolation_forest", "n": n, "trees": trees,
            "outliers": outlier_count,
        })

        logger.debug("Isolation Forest: %d points, %d trees, %d outliers, %.3fs",
                      n, trees, outlier_count, elapsed)

        return MultivariateResult(
            method=DetectionMethod.ISOLATION_FOREST,
            columns=columns,
            total_points=n,
            outliers_found=outlier_count,
            scores=scores_list,
            distance_threshold=0.6,
            mean_distance=sum(all_scores) / n if n > 0 else 0.0,
            max_distance=max(all_scores) if all_scores else 0.0,
            confidence=min(1.0, 0.5 + trees / 200.0 * 0.3 + n / 1000.0 * 0.2),
            provenance_hash=result_hash,
        )

    # ------------------------------------------------------------------
    # Local Outlier Factor (LOF)
    # ------------------------------------------------------------------

    def detect_lof(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        k: Optional[int] = None,
    ) -> MultivariateResult:
        """Detect outliers using Local Outlier Factor.

        LOF compares the local density of each point to its neighbors.
        Points with significantly lower density (LOF >> 1) are outliers.

        Args:
            records: List of record dictionaries.
            columns: Columns to use for detection.
            k: Number of neighbors (default from config).

        Returns:
            MultivariateResult with per-point LOF scores.
        """
        start = time.time()
        neighbors = k if k is not None else self._config.lof_neighbors
        data = self._extract_matrix(records, columns)
        n = len(data)

        if n < neighbors + 1:
            return self._empty_multivariate_result(
                records, columns, DetectionMethod.LOF,
            )

        # Compute pairwise distances
        dist_matrix = self._pairwise_distances(data)

        # k-distance and k-nearest neighbors for each point
        k_distances: List[float] = []
        k_neighbors: List[List[int]] = []

        for i in range(n):
            dists_i = [(dist_matrix[i][j], j) for j in range(n) if j != i]
            dists_i.sort(key=lambda x: x[0])
            knn = dists_i[:neighbors]
            k_dist = knn[-1][0] if knn else 0.0
            k_distances.append(k_dist)
            k_neighbors.append([idx for _, idx in knn])

        # Local reachability density (LRD)
        lrd: List[float] = []
        for i in range(n):
            reach_sum = 0.0
            for j in k_neighbors[i]:
                reach_dist = max(k_distances[j], dist_matrix[i][j])
                reach_sum += reach_dist
            if reach_sum > 0:
                lrd.append(neighbors / reach_sum)
            else:
                lrd.append(0.0)

        # LOF score
        lof_scores: List[float] = []
        for i in range(n):
            if lrd[i] > 0:
                neighbor_lrd_sum = sum(lrd[j] for j in k_neighbors[i])
                lof_val = (neighbor_lrd_sum / neighbors) / lrd[i]
            else:
                lof_val = 1.0
            lof_scores.append(lof_val)

        # Normalise LOF to 0-1 range
        max_lof = max(lof_scores) if lof_scores else 1.0
        min_lof = min(lof_scores) if lof_scores else 1.0
        lof_range = max_lof - min_lof if max_lof > min_lof else 1.0

        scores_list: List[OutlierScore] = []
        outlier_count = 0

        for i, lof_val in enumerate(lof_scores):
            # LOF > 1.5 is typically considered an outlier
            normalised = min(1.0, max(0.0, (lof_val - 1.0) / 1.0))
            is_outlier = lof_val > 1.5

            if is_outlier:
                outlier_count += 1

            provenance_hash = self._provenance.build_hash({
                "method": "lof", "index": i,
                "lof": lof_val, "k": neighbors,
            })

            scores_list.append(OutlierScore(
                record_index=i,
                column_name="|".join(columns),
                value=data[i] if i < len(data) else None,
                method=DetectionMethod.LOF,
                score=normalised,
                is_outlier=is_outlier,
                threshold=1.5,
                severity=_severity_from_score(normalised),
                details={"lof_score": lof_val, "lrd": lrd[i],
                         "k_distance": k_distances[i], "k": neighbors},
                confidence=min(1.0, 0.4 + normalised * 0.4 + n / 200.0 * 0.2),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        all_norm = [s.score for s in scores_list]
        result_hash = self._provenance.build_hash({
            "method": "lof", "n": n, "k": neighbors,
            "outliers": outlier_count,
        })

        logger.debug("LOF: %d points, k=%d, %d outliers, %.3fs",
                      n, neighbors, outlier_count, elapsed)

        return MultivariateResult(
            method=DetectionMethod.LOF,
            columns=columns,
            total_points=n,
            outliers_found=outlier_count,
            scores=scores_list,
            distance_threshold=1.5,
            mean_distance=sum(all_norm) / n if n > 0 else 0.0,
            max_distance=max(all_norm) if all_norm else 0.0,
            confidence=min(1.0, 0.5 + n / 200.0 * 0.3 + neighbors / 50.0 * 0.2),
            provenance_hash=result_hash,
        )

    # ------------------------------------------------------------------
    # DBSCAN (Density-Based)
    # ------------------------------------------------------------------

    def detect_dbscan(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> MultivariateResult:
        """Detect outliers using DBSCAN clustering.

        Noise points (not assigned to any cluster) are treated as outliers.

        Args:
            records: List of record dictionaries.
            columns: Columns to use for detection.
            eps: Neighborhood radius (default: auto-estimated).
            min_samples: Minimum cluster size (default: 5).

        Returns:
            MultivariateResult with per-point cluster/noise labels.
        """
        start = time.time()
        data = self._extract_matrix(records, columns)
        n = len(data)
        ms = min_samples if min_samples is not None else 5

        if n < ms + 1:
            return self._empty_multivariate_result(
                records, columns, DetectionMethod.DBSCAN,
            )

        # Auto-estimate eps if not provided
        if eps is None:
            eps = self._estimate_eps(data, ms)

        # DBSCAN clustering
        labels = [-1] * n  # -1 = unvisited
        cluster_id = 0
        visited = [False] * n

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._range_query(data, i, eps)

            if len(neighbors) < ms:
                labels[i] = -1  # Noise
            else:
                self._expand_cluster(
                    data, labels, visited, i, neighbors,
                    cluster_id, eps, ms,
                )
                cluster_id += 1

        # Noise points are outliers
        scores_list: List[OutlierScore] = []
        outlier_count = 0

        for i in range(n):
            is_noise = labels[i] == -1
            if is_noise:
                outlier_count += 1

            # Score: noise gets high score, cluster members get low
            score = 0.8 if is_noise else 0.1

            provenance_hash = self._provenance.build_hash({
                "method": "dbscan", "index": i,
                "cluster": labels[i], "is_noise": is_noise,
            })

            scores_list.append(OutlierScore(
                record_index=i,
                column_name="|".join(columns),
                value=data[i] if i < len(data) else None,
                method=DetectionMethod.DBSCAN,
                score=score,
                is_outlier=is_noise,
                threshold=eps,
                severity=_severity_from_score(score),
                details={"cluster_label": labels[i], "is_noise": is_noise,
                         "eps": eps, "min_samples": ms,
                         "n_clusters": cluster_id},
                confidence=min(1.0, 0.5 + n / 200.0 * 0.3),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        all_scores = [s.score for s in scores_list]
        result_hash = self._provenance.build_hash({
            "method": "dbscan", "n": n, "eps": eps, "ms": ms,
            "clusters": cluster_id, "outliers": outlier_count,
        })

        logger.debug("DBSCAN: %d points, %d clusters, %d noise, %.3fs",
                      n, cluster_id, outlier_count, elapsed)

        return MultivariateResult(
            method=DetectionMethod.DBSCAN,
            columns=columns,
            total_points=n,
            outliers_found=outlier_count,
            scores=scores_list,
            distance_threshold=eps,
            mean_distance=sum(all_scores) / n if n > 0 else 0.0,
            max_distance=max(all_scores) if all_scores else 0.0,
            confidence=min(1.0, 0.4 + n / 200.0 * 0.3 + cluster_id / 10.0 * 0.3),
            provenance_hash=result_hash,
        )

    # ------------------------------------------------------------------
    # Matrix utilities
    # ------------------------------------------------------------------

    def _compute_covariance_matrix(
        self,
        data: List[List[float]],
        means: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """Compute the covariance matrix of data.

        Args:
            data: N x D matrix.
            means: Pre-computed means (computed if None).

        Returns:
            D x D covariance matrix.
        """
        n = len(data)
        d = len(data[0]) if data else 0

        if means is None:
            means = [0.0] * d
            for row in data:
                for j in range(d):
                    means[j] += row[j]
            for j in range(d):
                means[j] /= n

        cov = [[0.0] * d for _ in range(d)]
        for row in data:
            for j in range(d):
                for k in range(d):
                    cov[j][k] += (row[j] - means[j]) * (row[k] - means[k])

        for j in range(d):
            for k in range(d):
                cov[j][k] /= max(n - 1, 1)

        return cov

    def _invert_matrix(
        self,
        matrix: List[List[float]],
    ) -> Optional[List[List[float]]]:
        """Invert a square matrix using Gauss-Jordan elimination.

        Args:
            matrix: D x D matrix.

        Returns:
            Inverted matrix or None if singular.
        """
        d = len(matrix)
        # Augment with identity
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(d)]
               for i, row in enumerate(matrix)]

        for col in range(d):
            # Find pivot
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, d):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row

            if max_val < 1e-12:
                return None  # Singular

            aug[col], aug[max_row] = aug[max_row], aug[col]

            # Scale pivot row
            pivot = aug[col][col]
            for j in range(2 * d):
                aug[col][j] /= pivot

            # Eliminate
            for row in range(d):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * d):
                    aug[row][j] -= factor * aug[col][j]

        # Extract inverse
        return [row[d:] for row in aug]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_matrix(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
    ) -> List[List[float]]:
        """Extract numeric matrix from records.

        Args:
            records: Record dictionaries.
            columns: Column names.

        Returns:
            N x D matrix of floats (rows with non-numeric values skipped).
        """
        matrix: List[List[float]] = []
        for rec in records:
            row: List[float] = []
            valid = True
            for col in columns:
                val = rec.get(col)
                if val is None:
                    valid = False
                    break
                try:
                    row.append(float(val))
                except (ValueError, TypeError):
                    valid = False
                    break
            if valid:
                matrix.append(row)
        return matrix

    def _mahalanobis_distance(
        self,
        diff: List[float],
        inv_cov: List[List[float]],
        d: int,
    ) -> float:
        """Compute Mahalanobis distance: sqrt(diff^T * inv_cov * diff).

        Args:
            diff: Difference vector (point - mean).
            inv_cov: Inverse covariance matrix.
            d: Dimensions.

        Returns:
            Mahalanobis distance.
        """
        # Compute inv_cov * diff
        temp = [0.0] * d
        for i in range(d):
            for j in range(d):
                temp[i] += inv_cov[i][j] * diff[j]

        # Compute diff^T * temp
        result = sum(diff[i] * temp[i] for i in range(d))
        return math.sqrt(max(0.0, result))

    def _diagonal_inverse(
        self,
        cov: List[List[float]],
        d: int,
    ) -> List[List[float]]:
        """Create diagonal-only inverse as fallback for singular matrix.

        Args:
            cov: Covariance matrix.
            d: Dimensions.

        Returns:
            Diagonal inverse matrix.
        """
        inv = [[0.0] * d for _ in range(d)]
        for i in range(d):
            if abs(cov[i][i]) > 1e-12:
                inv[i][i] = 1.0 / cov[i][i]
            else:
                inv[i][i] = 1.0
        return inv

    def _pairwise_distances(
        self,
        data: List[List[float]],
    ) -> List[List[float]]:
        """Compute pairwise Euclidean distance matrix.

        Args:
            data: N x D matrix.

        Returns:
            N x N distance matrix.
        """
        n = len(data)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum(
                    (data[i][k] - data[j][k]) ** 2
                    for k in range(len(data[i]))
                ))
                dist[i][j] = d
                dist[j][i] = d
        return dist

    def _euclidean_distance(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        """Compute Euclidean distance between two points.

        Args:
            a: First point.
            b: Second point.

        Returns:
            Euclidean distance.
        """
        return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(len(a))))

    def _build_isolation_tree(
        self,
        data: List[List[float]],
        depth: int,
        max_depth: int,
        d: int,
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Build a single isolation tree node.

        Args:
            data: Subsample data.
            depth: Current depth.
            max_depth: Maximum depth.
            d: Number of dimensions.
            rng: Random number generator.

        Returns:
            Tree node dictionary.
        """
        n = len(data)
        if n <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": n, "depth": depth}

        # Random feature and split
        feature = rng.randint(0, d - 1)
        vals = [row[feature] for row in data]
        min_val = min(vals)
        max_val = max(vals)

        if min_val == max_val:
            return {"type": "leaf", "size": n, "depth": depth}

        split = rng.uniform(min_val, max_val)
        left = [row for row in data if row[feature] < split]
        right = [row for row in data if row[feature] >= split]

        if not left or not right:
            return {"type": "leaf", "size": n, "depth": depth}

        return {
            "type": "split",
            "feature": feature,
            "split": split,
            "left": self._build_isolation_tree(left, depth + 1, max_depth, d, rng),
            "right": self._build_isolation_tree(right, depth + 1, max_depth, d, rng),
        }

    def _path_length(
        self,
        point: List[float],
        tree: Dict[str, Any],
        depth: int,
        d: int,
    ) -> float:
        """Compute path length for a point in an isolation tree.

        Args:
            point: Data point.
            tree: Tree node.
            depth: Current depth.
            d: Dimensions.

        Returns:
            Path length.
        """
        if tree["type"] == "leaf":
            size = tree["size"]
            return depth + self._average_path_length(size)

        feature = tree["feature"]
        split = tree["split"]

        if point[feature] < split:
            return self._path_length(point, tree["left"], depth + 1, d)
        return self._path_length(point, tree["right"], depth + 1, d)

    @staticmethod
    def _average_path_length(n: int) -> float:
        """Compute average path length for a BST with n nodes.

        c(n) = 2*H(n-1) - 2*(n-1)/n where H is harmonic number.

        Args:
            n: Number of nodes.

        Returns:
            Average path length.
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # H(n-1) approximation using Euler-Mascheroni
        h = math.log(n - 1) + 0.5772156649
        return 2.0 * h - 2.0 * (n - 1) / n

    def _range_query(
        self,
        data: List[List[float]],
        point_idx: int,
        eps: float,
    ) -> List[int]:
        """Find all points within eps distance of point_idx.

        Args:
            data: N x D matrix.
            point_idx: Index of the query point.
            eps: Radius.

        Returns:
            List of neighbor indices (including point_idx).
        """
        neighbors: List[int] = []
        for j in range(len(data)):
            if self._euclidean_distance(data[point_idx], data[j]) <= eps:
                neighbors.append(j)
        return neighbors

    def _expand_cluster(
        self,
        data: List[List[float]],
        labels: List[int],
        visited: List[bool],
        point_idx: int,
        neighbors: List[int],
        cluster_id: int,
        eps: float,
        min_samples: int,
    ) -> None:
        """Expand a DBSCAN cluster from a core point.

        Args:
            data: N x D matrix.
            labels: Cluster labels.
            visited: Visited flags.
            point_idx: Core point index.
            neighbors: Initial neighbor indices.
            cluster_id: Cluster ID to assign.
            eps: Neighborhood radius.
            min_samples: Minimum samples for core point.
        """
        labels[point_idx] = cluster_id
        queue = list(neighbors)
        i = 0

        while i < len(queue):
            q = queue[i]
            i += 1

            if not visited[q]:
                visited[q] = True
                q_neighbors = self._range_query(data, q, eps)
                if len(q_neighbors) >= min_samples:
                    queue.extend(q_neighbors)

            if labels[q] == -1:
                labels[q] = cluster_id

    def _estimate_eps(
        self,
        data: List[List[float]],
        min_samples: int,
    ) -> float:
        """Estimate eps using k-distance heuristic.

        Computes the k-th nearest neighbor distance for each point
        and uses the knee of the sorted distances as eps.

        Args:
            data: N x D matrix.
            min_samples: k value for k-distance.

        Returns:
            Estimated eps value.
        """
        n = len(data)
        k = min(min_samples, n - 1)
        k_dists: List[float] = []

        for i in range(n):
            dists = []
            for j in range(n):
                if i != j:
                    dists.append(self._euclidean_distance(data[i], data[j]))
            dists.sort()
            if k <= len(dists):
                k_dists.append(dists[k - 1])
            elif dists:
                k_dists.append(dists[-1])

        if not k_dists:
            return 1.0

        k_dists.sort()
        # Use the value at ~90th percentile of k-distances
        idx = int(0.9 * len(k_dists))
        return k_dists[min(idx, len(k_dists) - 1)]

    def _empty_multivariate_result(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        method: DetectionMethod,
    ) -> MultivariateResult:
        """Return empty result for insufficient data.

        Args:
            records: Input records.
            columns: Column names.
            method: Detection method.

        Returns:
            MultivariateResult with empty scores.
        """
        scores = []
        for i in range(len(records)):
            provenance_hash = self._provenance.build_hash({
                "method": method.value, "index": i,
                "reason": "insufficient_data",
            })
            scores.append(OutlierScore(
                record_index=i,
                column_name="|".join(columns),
                value=None,
                method=method,
                score=0.0,
                is_outlier=False,
                threshold=0.0,
                severity=SeverityLevel.INFO,
                details={"reason": "insufficient_data"},
                confidence=0.0,
                provenance_hash=provenance_hash,
            ))

        result_hash = self._provenance.build_hash({
            "method": method.value, "n": len(records),
            "reason": "insufficient_data",
        })

        return MultivariateResult(
            method=method,
            columns=columns,
            total_points=len(records),
            outliers_found=0,
            scores=scores,
            distance_threshold=0.0,
            mean_distance=0.0,
            max_distance=0.0,
            confidence=0.0,
            provenance_hash=result_hash,
        )


__all__ = [
    "MultivariateDetectorEngine",
]
