"""
GL-007 FurnacePulse - Hotspot Detector

Deterministic calculator for furnace tube metal temperature (TMT) monitoring,
hotspot detection, and tiered alert generation. Processes TMT readings to
identify spatial clustering of hot tubes and calculate rate-of-rise metrics.

Key Calculations:
    - Rate-of-rise (ROr) for each tube position
    - Spatial clustering detection using grid analysis
    - Time-above-threshold (TAT) tracking
    - Tiered alerts: Advisory/Warning/Urgent

Example:
    >>> detector = HotspotDetector(agent_id="GL-007")
    >>> readings = TMTReadings(
    ...     timestamp=datetime.now(timezone.utc),
    ...     tube_positions=[(0,0), (0,1), (1,0), (1,1)],
    ...     temperatures_c=[520, 580, 545, 610],
    ...     previous_temperatures_c=[515, 560, 540, 590]
    ... )
    >>> result = detector.calculate(readings)
    >>> for alert in result.result.alerts:
    ...     print(f"[{alert.level}] Tube {alert.tube_id}: {alert.message}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import math
import sys
import os

# Add framework path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Framework_GreenLang', 'shared'))

from calculator_base import DeterministicCalculator, CalculationResult


class AlertLevel(str, Enum):
    """Alert severity levels for hotspot detection."""
    NORMAL = "NORMAL"
    ADVISORY = "ADVISORY"  # Temperature approaching threshold
    WARNING = "WARNING"    # Temperature exceeded, requires attention
    URGENT = "URGENT"      # Critical temperature, immediate action required


@dataclass
class TubeThresholds:
    """
    Temperature thresholds for tube monitoring.

    Attributes:
        advisory_temp_c: Advisory threshold (approaching limit)
        warning_temp_c: Warning threshold (exceeded safe limit)
        urgent_temp_c: Urgent threshold (critical/emergency)
        rate_of_rise_limit_c_min: Maximum acceptable temperature rise rate
        cluster_distance: Maximum distance for spatial clustering
    """
    advisory_temp_c: float = 550.0
    warning_temp_c: float = 580.0
    urgent_temp_c: float = 620.0
    rate_of_rise_limit_c_min: float = 5.0  # deg C per minute
    cluster_distance: float = 2.0  # tube positions


@dataclass
class TMTReadings:
    """
    Tube Metal Temperature readings from furnace monitoring.

    Attributes:
        timestamp: Reading timestamp
        tube_positions: List of (row, col) tube positions
        temperatures_c: Current temperature readings in Celsius
        previous_temperatures_c: Previous temperature readings
        time_delta_seconds: Time between current and previous readings
        tube_ids: Optional tube identifiers
    """
    timestamp: datetime
    tube_positions: List[Tuple[int, int]]
    temperatures_c: List[float]
    previous_temperatures_c: Optional[List[float]] = None
    time_delta_seconds: float = 60.0  # Default 1 minute interval
    tube_ids: Optional[List[str]] = None

    def __post_init__(self):
        """Generate tube IDs if not provided."""
        if self.tube_ids is None:
            self.tube_ids = [
                f"T-R{pos[0]:02d}C{pos[1]:02d}"
                for pos in self.tube_positions
            ]


@dataclass
class TubeAnalysis:
    """
    Analysis results for a single tube.

    Attributes:
        tube_id: Tube identifier
        position: (row, col) position
        temperature_c: Current temperature
        rate_of_rise_c_min: Temperature change rate
        time_above_threshold_min: Time above warning threshold
        alert_level: Current alert level
        in_cluster: Whether tube is part of a hot cluster
        cluster_id: Cluster identifier if applicable
    """
    tube_id: str
    position: Tuple[int, int]
    temperature_c: float
    rate_of_rise_c_min: float
    time_above_threshold_min: float
    alert_level: AlertLevel
    in_cluster: bool = False
    cluster_id: Optional[int] = None


@dataclass
class HotspotCluster:
    """
    Spatial cluster of hot tubes.

    Attributes:
        cluster_id: Unique cluster identifier
        tube_ids: Tubes in this cluster
        center_position: Cluster centroid
        max_temperature_c: Maximum temperature in cluster
        avg_temperature_c: Average temperature in cluster
        size: Number of tubes in cluster
    """
    cluster_id: int
    tube_ids: List[str]
    center_position: Tuple[float, float]
    max_temperature_c: float
    avg_temperature_c: float
    size: int


@dataclass
class HotspotAlert:
    """
    Alert generated for hotspot condition.

    Attributes:
        alert_id: Unique alert identifier
        timestamp: Alert generation time
        level: Alert severity level
        tube_id: Affected tube identifier
        position: Tube position
        temperature_c: Current temperature
        threshold_c: Threshold that was exceeded
        rate_of_rise_c_min: Temperature change rate
        message: Human-readable alert message
        cluster_id: Cluster identifier if part of cluster
        recommended_action: Suggested operator action
    """
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    tube_id: str
    position: Tuple[int, int]
    temperature_c: float
    threshold_c: float
    rate_of_rise_c_min: float
    message: str
    cluster_id: Optional[int] = None
    recommended_action: Optional[str] = None


@dataclass
class IRImageHooks:
    """
    Hooks for IR image hotspot map analysis.

    Provides integration points for thermal imaging analysis
    without implementing image processing (zero-hallucination).

    Attributes:
        image_path: Path to IR image file
        calibration_matrix: Temperature calibration data
        roi_bounds: Region of interest boundaries
        pixel_to_tube_map: Mapping from image pixels to tube IDs
    """
    image_path: Optional[str] = None
    calibration_matrix: Optional[List[List[float]]] = None
    roi_bounds: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    pixel_to_tube_map: Optional[Dict[Tuple[int, int], str]] = None

    def is_configured(self) -> bool:
        """Check if IR integration is configured."""
        return all([
            self.image_path is not None,
            self.calibration_matrix is not None,
            self.pixel_to_tube_map is not None,
        ])


@dataclass
class HotspotAnalysis:
    """
    Complete hotspot analysis output.

    Attributes:
        timestamp: Analysis timestamp
        tube_analyses: Individual tube analysis results
        clusters: Detected hotspot clusters
        alerts: Generated alerts
        max_temperature_c: Maximum observed temperature
        avg_temperature_c: Average temperature
        tubes_above_advisory: Count of tubes above advisory threshold
        tubes_above_warning: Count of tubes above warning threshold
        tubes_above_urgent: Count of tubes above urgent threshold
        overall_status: Overall furnace status
        ir_image_hooks: Hooks for IR image integration
    """
    timestamp: datetime
    tube_analyses: List[TubeAnalysis]
    clusters: List[HotspotCluster]
    alerts: List[HotspotAlert]
    max_temperature_c: float
    avg_temperature_c: float
    tubes_above_advisory: int
    tubes_above_warning: int
    tubes_above_urgent: int
    overall_status: AlertLevel
    ir_image_hooks: Optional[IRImageHooks] = None


class HotspotDetector(DeterministicCalculator[TMTReadings, HotspotAnalysis]):
    """
    Deterministic calculator for furnace hotspot detection.

    Processes tube metal temperature (TMT) readings to detect hotspots,
    calculate rate-of-rise, identify spatial clusters, and generate
    tiered alerts. All calculations are deterministic with SHA-256
    provenance tracking.

    Features:
        - Rate-of-rise calculation for each tube
        - Spatial clustering using distance-based algorithm
        - Time-above-threshold tracking
        - Tiered alert generation (Advisory/Warning/Urgent)
        - IR image integration hooks

    Example:
        >>> detector = HotspotDetector(agent_id="GL-007")
        >>> readings = TMTReadings(
        ...     timestamp=datetime.now(timezone.utc),
        ...     tube_positions=[(0,0), (0,1), (1,0), (1,1)],
        ...     temperatures_c=[520, 580, 545, 610]
        ... )
        >>> result = detector.calculate(readings)
        >>> print(f"Max temp: {result.result.max_temperature_c}C")
    """

    NAME = "FurnaceHotspotDetector"
    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-007",
        track_provenance: bool = True,
        thresholds: Optional[TubeThresholds] = None,
    ):
        """
        Initialize hotspot detector.

        Args:
            agent_id: Agent identifier for provenance
            track_provenance: Whether to track calculation provenance
            thresholds: Custom temperature thresholds
        """
        super().__init__(agent_id, track_provenance)
        self.thresholds = thresholds or TubeThresholds()
        self._time_above_threshold: Dict[str, float] = {}  # tube_id -> minutes
        self._alert_counter = 0

    def _validate_inputs(self, inputs: TMTReadings) -> List[str]:
        """
        Validate TMT readings inputs.

        Args:
            inputs: TMT readings to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check array lengths match
        n_positions = len(inputs.tube_positions)
        n_temps = len(inputs.temperatures_c)

        if n_positions == 0:
            errors.append("No tube positions provided")

        if n_temps == 0:
            errors.append("No temperature readings provided")

        if n_positions != n_temps:
            errors.append(
                f"Mismatch: {n_positions} positions but {n_temps} temperatures"
            )

        if inputs.previous_temperatures_c is not None:
            n_prev = len(inputs.previous_temperatures_c)
            if n_prev != n_temps:
                errors.append(
                    f"Mismatch: {n_temps} current but {n_prev} previous temperatures"
                )

        # Validate temperature ranges
        for i, temp in enumerate(inputs.temperatures_c):
            if temp < 0:
                errors.append(f"Negative temperature at index {i}: {temp}C")
            elif temp > 1500:
                errors.append(f"Temperature exceeds max (1500C) at index {i}: {temp}C")

        # Validate time delta
        if inputs.time_delta_seconds <= 0:
            errors.append("time_delta_seconds must be positive")

        # Validate positions are unique
        positions_set = set(inputs.tube_positions)
        if len(positions_set) != n_positions:
            errors.append("Duplicate tube positions detected")

        return errors

    def _calculate(self, inputs: TMTReadings, **kwargs: Any) -> HotspotAnalysis:
        """
        Perform hotspot detection analysis.

        This is a DETERMINISTIC calculation:
        - Rate of rise = (T_current - T_previous) / time_delta
        - Clustering uses fixed distance threshold
        - Alert levels based on fixed thresholds

        Args:
            inputs: Validated TMT readings

        Returns:
            HotspotAnalysis with all detection results
        """
        # Step 1: Analyze individual tubes
        tube_analyses = self._analyze_tubes(inputs)

        # Step 2: Detect spatial clusters
        clusters = self._detect_clusters(tube_analyses)

        # Step 3: Update cluster assignments
        self._assign_clusters_to_tubes(tube_analyses, clusters)

        # Step 4: Generate alerts
        alerts = self._generate_alerts(tube_analyses, clusters, inputs.timestamp)

        # Step 5: Calculate summary statistics
        temperatures = inputs.temperatures_c
        max_temp = max(temperatures)
        avg_temp = sum(temperatures) / len(temperatures)

        tubes_above_advisory = sum(
            1 for t in temperatures if t >= self.thresholds.advisory_temp_c
        )
        tubes_above_warning = sum(
            1 for t in temperatures if t >= self.thresholds.warning_temp_c
        )
        tubes_above_urgent = sum(
            1 for t in temperatures if t >= self.thresholds.urgent_temp_c
        )

        # Determine overall status
        if tubes_above_urgent > 0:
            overall_status = AlertLevel.URGENT
        elif tubes_above_warning > 0:
            overall_status = AlertLevel.WARNING
        elif tubes_above_advisory > 0:
            overall_status = AlertLevel.ADVISORY
        else:
            overall_status = AlertLevel.NORMAL

        return HotspotAnalysis(
            timestamp=inputs.timestamp,
            tube_analyses=tube_analyses,
            clusters=clusters,
            alerts=alerts,
            max_temperature_c=round(max_temp, 2),
            avg_temperature_c=round(avg_temp, 2),
            tubes_above_advisory=tubes_above_advisory,
            tubes_above_warning=tubes_above_warning,
            tubes_above_urgent=tubes_above_urgent,
            overall_status=overall_status,
            ir_image_hooks=kwargs.get("ir_hooks"),
        )

    def _analyze_tubes(self, inputs: TMTReadings) -> List[TubeAnalysis]:
        """Analyze individual tube readings."""
        analyses = []
        time_delta_min = inputs.time_delta_seconds / 60.0

        for i, (pos, temp, tube_id) in enumerate(zip(
            inputs.tube_positions,
            inputs.temperatures_c,
            inputs.tube_ids
        )):
            # Calculate rate of rise
            rate_of_rise = 0.0
            if inputs.previous_temperatures_c is not None:
                prev_temp = inputs.previous_temperatures_c[i]
                rate_of_rise = (temp - prev_temp) / time_delta_min

            # Update time-above-threshold tracking
            if temp >= self.thresholds.warning_temp_c:
                current_tat = self._time_above_threshold.get(tube_id, 0.0)
                self._time_above_threshold[tube_id] = current_tat + time_delta_min
            else:
                self._time_above_threshold[tube_id] = 0.0

            time_above_threshold = self._time_above_threshold.get(tube_id, 0.0)

            # Determine alert level
            alert_level = self._determine_alert_level(temp, rate_of_rise)

            analyses.append(TubeAnalysis(
                tube_id=tube_id,
                position=pos,
                temperature_c=round(temp, 2),
                rate_of_rise_c_min=round(rate_of_rise, 3),
                time_above_threshold_min=round(time_above_threshold, 1),
                alert_level=alert_level,
                in_cluster=False,
                cluster_id=None,
            ))

        return analyses

    def _determine_alert_level(
        self,
        temperature_c: float,
        rate_of_rise_c_min: float,
    ) -> AlertLevel:
        """Determine alert level based on temperature and rate of rise."""
        # Check urgent conditions first
        if temperature_c >= self.thresholds.urgent_temp_c:
            return AlertLevel.URGENT

        # Check warning conditions
        if temperature_c >= self.thresholds.warning_temp_c:
            return AlertLevel.WARNING

        # Check advisory conditions (including high rate of rise)
        if (temperature_c >= self.thresholds.advisory_temp_c or
                rate_of_rise_c_min > self.thresholds.rate_of_rise_limit_c_min):
            return AlertLevel.ADVISORY

        return AlertLevel.NORMAL

    def _detect_clusters(
        self,
        tube_analyses: List[TubeAnalysis],
    ) -> List[HotspotCluster]:
        """
        Detect spatial clusters of hot tubes.

        Uses simple distance-based clustering:
        - Tubes above advisory threshold are candidates
        - Tubes within cluster_distance are grouped together
        """
        # Get hot tubes (above advisory threshold)
        hot_tubes = [
            t for t in tube_analyses
            if t.temperature_c >= self.thresholds.advisory_temp_c
        ]

        if not hot_tubes:
            return []

        # Build adjacency based on distance
        clusters = []
        visited: Set[str] = set()
        cluster_id = 0

        for tube in hot_tubes:
            if tube.tube_id in visited:
                continue

            # Start new cluster with BFS
            cluster_tubes = []
            queue = [tube]

            while queue:
                current = queue.pop(0)
                if current.tube_id in visited:
                    continue

                visited.add(current.tube_id)
                cluster_tubes.append(current)

                # Find neighbors
                for other in hot_tubes:
                    if other.tube_id not in visited:
                        dist = self._calculate_distance(
                            current.position, other.position
                        )
                        if dist <= self.thresholds.cluster_distance:
                            queue.append(other)

            # Create cluster if >= 2 tubes
            if len(cluster_tubes) >= 2:
                # Calculate cluster properties
                temps = [t.temperature_c for t in cluster_tubes]
                positions = [t.position for t in cluster_tubes]

                center_row = sum(p[0] for p in positions) / len(positions)
                center_col = sum(p[1] for p in positions) / len(positions)

                clusters.append(HotspotCluster(
                    cluster_id=cluster_id,
                    tube_ids=[t.tube_id for t in cluster_tubes],
                    center_position=(round(center_row, 2), round(center_col, 2)),
                    max_temperature_c=round(max(temps), 2),
                    avg_temperature_c=round(sum(temps) / len(temps), 2),
                    size=len(cluster_tubes),
                ))
                cluster_id += 1

        return clusters

    def _calculate_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2
        )

    def _assign_clusters_to_tubes(
        self,
        tube_analyses: List[TubeAnalysis],
        clusters: List[HotspotCluster],
    ) -> None:
        """Assign cluster IDs to tube analyses in place."""
        # Build tube_id -> cluster_id mapping
        tube_to_cluster: Dict[str, int] = {}
        for cluster in clusters:
            for tube_id in cluster.tube_ids:
                tube_to_cluster[tube_id] = cluster.cluster_id

        # Update tube analyses
        for analysis in tube_analyses:
            if analysis.tube_id in tube_to_cluster:
                analysis.in_cluster = True
                analysis.cluster_id = tube_to_cluster[analysis.tube_id]

    def _generate_alerts(
        self,
        tube_analyses: List[TubeAnalysis],
        clusters: List[HotspotCluster],
        timestamp: datetime,
    ) -> List[HotspotAlert]:
        """Generate alerts for hotspot conditions."""
        alerts = []

        for analysis in tube_analyses:
            if analysis.alert_level == AlertLevel.NORMAL:
                continue

            self._alert_counter += 1
            alert_id = f"HS-{timestamp.strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

            # Determine threshold and message
            if analysis.alert_level == AlertLevel.URGENT:
                threshold = self.thresholds.urgent_temp_c
                message = (
                    f"CRITICAL: Tube {analysis.tube_id} at {analysis.temperature_c}C "
                    f"exceeds urgent threshold ({threshold}C). "
                    f"Rate of rise: {analysis.rate_of_rise_c_min}C/min."
                )
                action = "Immediate inspection required. Consider emergency shutdown if rising."

            elif analysis.alert_level == AlertLevel.WARNING:
                threshold = self.thresholds.warning_temp_c
                message = (
                    f"WARNING: Tube {analysis.tube_id} at {analysis.temperature_c}C "
                    f"exceeds warning threshold ({threshold}C). "
                    f"TAT: {analysis.time_above_threshold_min} min."
                )
                action = "Schedule inspection within 4 hours. Monitor closely."

            else:  # ADVISORY
                threshold = self.thresholds.advisory_temp_c
                if analysis.rate_of_rise_c_min > self.thresholds.rate_of_rise_limit_c_min:
                    message = (
                        f"ADVISORY: Tube {analysis.tube_id} rate of rise "
                        f"({analysis.rate_of_rise_c_min}C/min) exceeds limit."
                    )
                else:
                    message = (
                        f"ADVISORY: Tube {analysis.tube_id} at {analysis.temperature_c}C "
                        f"approaching threshold ({threshold}C)."
                    )
                action = "Continue monitoring. Log for trend analysis."

            alerts.append(HotspotAlert(
                alert_id=alert_id,
                timestamp=timestamp,
                level=analysis.alert_level,
                tube_id=analysis.tube_id,
                position=analysis.position,
                temperature_c=analysis.temperature_c,
                threshold_c=threshold,
                rate_of_rise_c_min=analysis.rate_of_rise_c_min,
                message=message,
                cluster_id=analysis.cluster_id,
                recommended_action=action,
            ))

        # Add cluster-level alerts for large clusters
        for cluster in clusters:
            if cluster.size >= 3:  # Significant cluster
                self._alert_counter += 1
                alert_id = f"HS-CL-{timestamp.strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

                level = AlertLevel.WARNING
                if cluster.max_temperature_c >= self.thresholds.urgent_temp_c:
                    level = AlertLevel.URGENT

                alerts.append(HotspotAlert(
                    alert_id=alert_id,
                    timestamp=timestamp,
                    level=level,
                    tube_id=f"CLUSTER-{cluster.cluster_id}",
                    position=(int(cluster.center_position[0]), int(cluster.center_position[1])),
                    temperature_c=cluster.max_temperature_c,
                    threshold_c=self.thresholds.warning_temp_c,
                    rate_of_rise_c_min=0.0,
                    message=(
                        f"Spatial cluster detected: {cluster.size} tubes in region "
                        f"centered at {cluster.center_position}. "
                        f"Max temp: {cluster.max_temperature_c}C, "
                        f"Avg temp: {cluster.avg_temperature_c}C."
                    ),
                    cluster_id=cluster.cluster_id,
                    recommended_action=(
                        "Investigate possible burner misalignment, "
                        "refractory damage, or flow maldistribution."
                    ),
                ))

        # Sort alerts by severity (URGENT first)
        severity_order = {
            AlertLevel.URGENT: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ADVISORY: 2,
            AlertLevel.NORMAL: 3,
        }
        alerts.sort(key=lambda a: (severity_order[a.level], -a.temperature_c))

        return alerts

    def reset_tracking(self) -> None:
        """Reset time-above-threshold tracking and alert counter."""
        self._time_above_threshold.clear()
        self._alert_counter = 0

    def get_ir_image_hooks(self) -> IRImageHooks:
        """
        Get IR image integration hooks.

        Returns empty hooks - actual image processing should be done
        by specialized image processing module, not in calculator.

        Returns:
            IRImageHooks instance for integration
        """
        return IRImageHooks()

    def configure_ir_integration(
        self,
        image_path: str,
        calibration_matrix: List[List[float]],
        pixel_to_tube_map: Dict[Tuple[int, int], str],
        roi_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> IRImageHooks:
        """
        Configure IR image integration.

        This does not process images - it provides configuration
        hooks for external image processing to integrate with
        the detector.

        Args:
            image_path: Path to IR image file
            calibration_matrix: Temperature calibration data
            pixel_to_tube_map: Pixel to tube ID mapping
            roi_bounds: Optional region of interest

        Returns:
            Configured IRImageHooks
        """
        return IRImageHooks(
            image_path=image_path,
            calibration_matrix=calibration_matrix,
            pixel_to_tube_map=pixel_to_tube_map,
            roi_bounds=roi_bounds,
        )
