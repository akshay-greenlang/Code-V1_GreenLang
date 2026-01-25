"""
GL-013 PREDICTMAINT - Statistical Anomaly Detection Module

This module implements statistical anomaly detection methods for
predictive maintenance condition monitoring.

Key Features:
- Isolation Forest implementation
- CUSUM (Cumulative Sum) for trend detection
- Statistical Process Control (SPC)
- Mahalanobis distance for multivariate analysis
- Multivariate analysis
- Anomaly scoring and classification

Reference Standards:
- ISO 13379-1:2012 Condition monitoring
- ISO 16587:2004 Statistical process control
- ASTM E2587: Practice for control charts

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import math
import random
from collections import deque

from .constants import (
    DEFAULT_DECIMAL_PRECISION,
    Z_SCORES,
)
from .provenance import (
    ProvenanceBuilder,
    ProvenanceRecord,
    CalculationType,
    store_provenance,
)


# =============================================================================
# ENUMS
# =============================================================================

class AnomalyType(Enum):
    """Types of anomalies detected."""
    POINT_ANOMALY = auto()       # Single outlier point
    CONTEXTUAL = auto()          # Anomaly in specific context
    COLLECTIVE = auto()          # Group of related anomalies
    TREND_SHIFT = auto()         # Gradual drift
    LEVEL_SHIFT = auto()         # Sudden change in mean
    VARIANCE_CHANGE = auto()     # Change in variability
    PATTERN_ANOMALY = auto()     # Unusual pattern


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    INFO = auto()        # Informational, no action needed
    LOW = auto()         # Minor deviation, monitor
    MEDIUM = auto()      # Significant deviation, investigate
    HIGH = auto()        # Major deviation, action needed
    CRITICAL = auto()    # Severe, immediate action required


class ControlChartRule(Enum):
    """Western Electric / Nelson rules for SPC."""
    RULE_1 = auto()  # Point beyond 3 sigma
    RULE_2 = auto()  # 2 of 3 points beyond 2 sigma (same side)
    RULE_3 = auto()  # 4 of 5 points beyond 1 sigma (same side)
    RULE_4 = auto()  # 8 points in a row on one side of centerline
    RULE_5 = auto()  # 6 points in a row increasing or decreasing
    RULE_6 = auto()  # 14 points alternating up and down
    RULE_7 = auto()  # 15 points in a row within 1 sigma
    RULE_8 = auto()  # 8 points in a row beyond 1 sigma (either side)


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class AnomalyResult:
    """
    Result of anomaly detection analysis.

    Attributes:
        is_anomaly: Whether point is classified as anomaly
        anomaly_score: Score indicating degree of anomaly (0-1)
        anomaly_type: Type of anomaly if detected
        severity: Severity level
        z_score: Standard deviations from mean
        explanation: Human-readable explanation
        contributing_factors: Factors contributing to anomaly
        provenance_hash: SHA-256 hash
    """
    is_anomaly: bool
    anomaly_score: Decimal
    anomaly_type: Optional[AnomalyType]
    severity: AnomalySeverity
    z_score: Decimal
    explanation: str
    contributing_factors: Tuple[str, ...]
    confidence: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": str(self.anomaly_score),
            "anomaly_type": self.anomaly_type.name if self.anomaly_type else None,
            "severity": self.severity.name,
            "z_score": str(self.z_score),
            "explanation": self.explanation,
            "contributing_factors": list(self.contributing_factors),
            "confidence": str(self.confidence),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class CUSUMResult:
    """
    Result of CUSUM (Cumulative Sum) analysis.

    CUSUM detects small shifts in the process mean.
    """
    cusum_upper: Decimal
    cusum_lower: Decimal
    is_out_of_control: bool
    shift_detected: bool
    shift_direction: Optional[str]
    shift_magnitude: Optional[Decimal]
    decision_value: Decimal
    threshold: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class SPCResult:
    """
    Result of Statistical Process Control analysis.

    Evaluates data against control chart rules.
    """
    in_control: bool
    violated_rules: Tuple[ControlChartRule, ...]
    ucl: Decimal  # Upper Control Limit
    lcl: Decimal  # Lower Control Limit
    center_line: Decimal
    current_value: Decimal
    zone: str  # A, B, C, or Out
    process_capability: Optional[Decimal]
    provenance_hash: str = ""


@dataclass(frozen=True)
class MahalanobisResult:
    """
    Result of Mahalanobis distance calculation.

    Measures distance in multivariate space accounting for correlations.
    """
    distance: Decimal
    is_outlier: bool
    threshold: Decimal
    contributing_variables: Tuple[str, ...]
    chi_square_percentile: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class IsolationForestResult:
    """
    Result of Isolation Forest anomaly detection.

    Score interpretation:
    - Score close to 1: Anomaly
    - Score close to 0.5: Normal
    - Score close to 0: Normal (dense region)
    """
    anomaly_score: Decimal
    is_anomaly: bool
    average_path_length: Decimal
    expected_path_length: Decimal
    contamination_threshold: Decimal
    provenance_hash: str = ""


# =============================================================================
# ANOMALY DETECTOR
# =============================================================================

class AnomalyDetector:
    """
    Statistical anomaly detection for predictive maintenance.

    Implements multiple anomaly detection algorithms with
    complete provenance tracking.

    All calculations are:
    - Deterministic (for reproducibility)
    - Documented with provenance
    - Based on established statistical methods

    Reference: ISO 13379-1:2012

    Example:
        >>> detector = AnomalyDetector()
        >>> result = detector.detect_univariate_anomaly(
        ...     value=Decimal("105"),
        ...     historical_data=[100, 101, 99, 102, 100, 98, 101],
        ...     threshold_sigma=3.0
        ... )
        >>> print(f"Anomaly: {result.is_anomaly}, Score: {result.anomaly_score}")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        store_provenance_records: bool = True
    ):
        """
        Initialize Anomaly Detector.

        Args:
            precision: Decimal precision for calculations
            store_provenance_records: Whether to store provenance
        """
        self._precision = precision
        self._store_provenance = store_provenance_records
        self._cusum_state: Dict[str, Dict[str, Decimal]] = {}
        self._spc_state: Dict[str, deque] = {}

    # =========================================================================
    # UNIVARIATE ANOMALY DETECTION
    # =========================================================================

    def detect_univariate_anomaly(
        self,
        value: Union[Decimal, float, str],
        historical_data: List[Union[Decimal, float]],
        threshold_sigma: Union[Decimal, float, str] = "3.0",
        use_mad: bool = False
    ) -> AnomalyResult:
        """
        Detect anomalies in univariate data using z-score.

        Uses standard z-score or robust MAD-based detection:
        - Z-score: z = (x - mean) / std
        - MAD: z = 0.6745 * (x - median) / MAD

        Args:
            value: Current value to test
            historical_data: Historical reference data
            threshold_sigma: Number of standard deviations for threshold
            use_mad: Use Median Absolute Deviation (robust to outliers)

        Returns:
            AnomalyResult

        Reference:
            Grubbs, F.E. (1969). Procedures for detecting outlying
            observations in samples. Technometrics, 11(1), 1-21.

        Example:
            >>> detector = AnomalyDetector()
            >>> result = detector.detect_univariate_anomaly(
            ...     value=150,
            ...     historical_data=[100, 102, 98, 101, 99, 100],
            ...     threshold_sigma=3.0
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.ANOMALY_DETECTION)

        # Convert inputs
        x = self._to_decimal(value)
        data = [self._to_decimal(v) for v in historical_data]
        threshold = self._to_decimal(threshold_sigma)

        if len(data) < 3:
            raise ValueError("Need at least 3 historical data points")

        builder.add_input("value", x)
        builder.add_input("num_historical", len(data))
        builder.add_input("threshold_sigma", threshold)
        builder.add_input("use_mad", use_mad)

        # Step 1: Calculate location and scale statistics
        if use_mad:
            # Robust statistics
            median = self._median(data)
            mad = self._median_absolute_deviation(data)
            location = median
            scale = mad / Decimal("0.6745") if mad > Decimal("0") else Decimal("1")
            method = "MAD"
        else:
            # Standard statistics
            mean = sum(data) / Decimal(str(len(data)))
            variance = sum((v - mean) ** 2 for v in data) / Decimal(str(len(data) - 1))
            std_dev = self._sqrt(variance)
            location = mean
            scale = std_dev if std_dev > Decimal("0") else Decimal("1")
            method = "Z-Score"

        builder.add_step(
            step_number=1,
            operation="calculate",
            description=f"Calculate {method} statistics",
            inputs={"data_length": len(data)},
            output_name="statistics",
            output_value={"location": location, "scale": scale},
            formula="mean/median and std/MAD"
        )

        # Step 2: Calculate z-score
        z_score = (x - location) / scale

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate z-score",
            inputs={"x": x, "location": location, "scale": scale},
            output_name="z_score",
            output_value=z_score,
            formula="z = (x - location) / scale"
        )

        # Step 3: Determine if anomaly
        abs_z = abs(z_score)
        is_anomaly = abs_z > threshold

        # Calculate anomaly score (0 to 1)
        # Score = 1 - P(Z < |z|)
        if abs_z < Decimal("5"):
            p_value = Decimal("1") - self._normal_cdf(abs_z)
            anomaly_score = Decimal("1") - (Decimal("2") * p_value)  # Two-tailed
        else:
            anomaly_score = Decimal("1")  # Extreme outlier

        builder.add_step(
            step_number=3,
            operation="compare",
            description="Determine anomaly status",
            inputs={"abs_z": abs_z, "threshold": threshold},
            output_name="is_anomaly",
            output_value=is_anomaly
        )

        # Step 4: Classify anomaly type and severity
        if is_anomaly:
            if abs_z > Decimal("5"):
                severity = AnomalySeverity.CRITICAL
                anomaly_type = AnomalyType.POINT_ANOMALY
            elif abs_z > Decimal("4"):
                severity = AnomalySeverity.HIGH
                anomaly_type = AnomalyType.POINT_ANOMALY
            elif abs_z > Decimal("3"):
                severity = AnomalySeverity.MEDIUM
                anomaly_type = AnomalyType.POINT_ANOMALY
            else:
                severity = AnomalySeverity.LOW
                anomaly_type = AnomalyType.POINT_ANOMALY

            direction = "above" if z_score > Decimal("0") else "below"
            explanation = (
                f"Value {x} is {abs_z:.2f} standard deviations {direction} "
                f"the expected range (threshold: {threshold} sigma)"
            )
        else:
            severity = AnomalySeverity.INFO
            anomaly_type = None
            explanation = f"Value {x} is within normal range (z-score: {z_score:.2f})"

        # Calculate confidence
        confidence = min(Decimal("1"), abs_z / threshold) if is_anomaly else Decimal("1") - anomaly_score

        builder.add_output("is_anomaly", is_anomaly)
        builder.add_output("z_score", z_score)
        builder.add_output("anomaly_score", anomaly_score)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=self._apply_precision(anomaly_score, 4),
            anomaly_type=anomaly_type,
            severity=severity,
            z_score=self._apply_precision(z_score, 4),
            explanation=explanation,
            contributing_factors=("High deviation from mean",) if is_anomaly else (),
            confidence=self._apply_precision(confidence, 4),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # CUSUM (CUMULATIVE SUM)
    # =========================================================================

    def detect_cusum_shift(
        self,
        equipment_id: str,
        value: Union[Decimal, float, str],
        target: Union[Decimal, float, str],
        sigma: Union[Decimal, float, str],
        k: Union[Decimal, float, str] = "0.5",
        h: Union[Decimal, float, str] = "5.0"
    ) -> CUSUMResult:
        """
        Detect mean shift using CUSUM (Cumulative Sum) control chart.

        CUSUM is sensitive to small, persistent shifts in the mean.

        Upper CUSUM: C_t^+ = max(0, x_t - (mu_0 + K) + C_{t-1}^+)
        Lower CUSUM: C_t^- = max(0, (mu_0 - K) - x_t + C_{t-1}^-)

        Out of control when C^+ > H or C^- > H

        Args:
            equipment_id: Equipment identifier for state tracking
            value: Current observation
            target: Target/expected mean (mu_0)
            sigma: Process standard deviation
            k: Slack value (typically 0.5 sigma)
            h: Decision interval (typically 4-5 sigma)

        Returns:
            CUSUMResult

        Reference:
            Page, E.S. (1954). Continuous inspection schemes.
            Biometrika, 41(1/2), 100-115.

        Example:
            >>> detector = AnomalyDetector()
            >>> for value in [100, 101, 102, 105, 106, 107]:
            ...     result = detector.detect_cusum_shift(
            ...         equipment_id="PUMP-001",
            ...         value=value,
            ...         target=100,
            ...         sigma=2
            ...     )
        """
        builder = ProvenanceBuilder(CalculationType.ANOMALY_DETECTION)

        # Convert inputs
        x = self._to_decimal(value)
        mu_0 = self._to_decimal(target)
        sigma_val = self._to_decimal(sigma)
        K = self._to_decimal(k) * sigma_val  # Slack in units
        H = self._to_decimal(h) * sigma_val  # Decision interval in units

        builder.add_input("equipment_id", equipment_id)
        builder.add_input("value", x)
        builder.add_input("target", mu_0)
        builder.add_input("sigma", sigma_val)
        builder.add_input("k", k)
        builder.add_input("h", h)

        # Initialize or get state
        if equipment_id not in self._cusum_state:
            self._cusum_state[equipment_id] = {
                "C_plus": Decimal("0"),
                "C_minus": Decimal("0"),
                "n": 0
            }

        state = self._cusum_state[equipment_id]
        C_plus_prev = state["C_plus"]
        C_minus_prev = state["C_minus"]

        # Step 1: Calculate standardized value
        z = (x - mu_0) / sigma_val

        builder.add_step(
            step_number=1,
            operation="standardize",
            description="Standardize observation",
            inputs={"x": x, "mu_0": mu_0, "sigma": sigma_val},
            output_name="z",
            output_value=z,
            formula="z = (x - mu_0) / sigma"
        )

        # Step 2: Update upper CUSUM
        # C^+ = max(0, x - (mu_0 + K) + C^+_prev)
        C_plus = max(Decimal("0"), x - (mu_0 + K) + C_plus_prev)

        builder.add_step(
            step_number=2,
            operation="update",
            description="Update upper CUSUM",
            inputs={"x": x, "mu_0_plus_K": mu_0 + K, "C_plus_prev": C_plus_prev},
            output_name="C_plus",
            output_value=C_plus,
            formula="C^+ = max(0, x - (mu_0 + K) + C^+_prev)"
        )

        # Step 3: Update lower CUSUM
        # C^- = max(0, (mu_0 - K) - x + C^-_prev)
        C_minus = max(Decimal("0"), (mu_0 - K) - x + C_minus_prev)

        builder.add_step(
            step_number=3,
            operation="update",
            description="Update lower CUSUM",
            inputs={"mu_0_minus_K": mu_0 - K, "x": x, "C_minus_prev": C_minus_prev},
            output_name="C_minus",
            output_value=C_minus,
            formula="C^- = max(0, (mu_0 - K) - x + C^-_prev)"
        )

        # Step 4: Check for out of control
        is_out_of_control = C_plus > H or C_minus > H

        if C_plus > H:
            shift_detected = True
            shift_direction = "positive"
            shift_magnitude = C_plus / sigma_val
            decision_value = C_plus
        elif C_minus > H:
            shift_detected = True
            shift_direction = "negative"
            shift_magnitude = -C_minus / sigma_val
            decision_value = C_minus
        else:
            shift_detected = False
            shift_direction = None
            shift_magnitude = None
            decision_value = max(C_plus, C_minus)

        builder.add_step(
            step_number=4,
            operation="evaluate",
            description="Check control status",
            inputs={"C_plus": C_plus, "C_minus": C_minus, "H": H},
            output_name="is_out_of_control",
            output_value=is_out_of_control
        )

        # Update state
        state["C_plus"] = C_plus
        state["C_minus"] = C_minus
        state["n"] += 1

        builder.add_output("cusum_upper", C_plus)
        builder.add_output("cusum_lower", C_minus)
        builder.add_output("shift_detected", shift_detected)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return CUSUMResult(
            cusum_upper=self._apply_precision(C_plus, 4),
            cusum_lower=self._apply_precision(C_minus, 4),
            is_out_of_control=is_out_of_control,
            shift_detected=shift_detected,
            shift_direction=shift_direction,
            shift_magnitude=self._apply_precision(shift_magnitude, 4) if shift_magnitude else None,
            decision_value=self._apply_precision(decision_value, 4),
            threshold=self._apply_precision(H, 4),
            provenance_hash=provenance.final_hash
        )

    def reset_cusum(self, equipment_id: str) -> None:
        """Reset CUSUM state for equipment."""
        if equipment_id in self._cusum_state:
            del self._cusum_state[equipment_id]

    # =========================================================================
    # STATISTICAL PROCESS CONTROL (SPC)
    # =========================================================================

    def evaluate_spc(
        self,
        equipment_id: str,
        value: Union[Decimal, float, str],
        target: Optional[Union[Decimal, float, str]] = None,
        ucl: Optional[Union[Decimal, float, str]] = None,
        lcl: Optional[Union[Decimal, float, str]] = None,
        sigma: Optional[Union[Decimal, float, str]] = None,
        window_size: int = 20
    ) -> SPCResult:
        """
        Evaluate Statistical Process Control chart.

        Checks Western Electric / Nelson rules for special causes:
        1. Point beyond 3 sigma
        2. 2 of 3 points beyond 2 sigma (same side)
        3. 4 of 5 points beyond 1 sigma (same side)
        4. 8 points in a row on one side of centerline
        5. 6 points in a row increasing or decreasing
        6. 14 points alternating up and down
        7. 15 points in a row within 1 sigma
        8. 8 points in a row beyond 1 sigma (either side)

        Args:
            equipment_id: Equipment identifier for state tracking
            value: Current observation
            target: Center line (if None, uses sample mean)
            ucl: Upper Control Limit (if None, calculates 3-sigma)
            lcl: Lower Control Limit (if None, calculates 3-sigma)
            sigma: Standard deviation (if None, estimates from data)
            window_size: Size of moving window for statistics

        Returns:
            SPCResult

        Reference:
            ASTM E2587-16, Standard Practice for Use of Control Charts

        Example:
            >>> detector = AnomalyDetector()
            >>> for value in [100, 101, 99, 102, 98, 105, 95, 110]:
            ...     result = detector.evaluate_spc(
            ...         equipment_id="SENSOR-001",
            ...         value=value,
            ...         target=100,
            ...         sigma=3
            ...     )
        """
        builder = ProvenanceBuilder(CalculationType.ANOMALY_DETECTION)

        x = self._to_decimal(value)

        builder.add_input("equipment_id", equipment_id)
        builder.add_input("value", x)

        # Initialize or get history
        if equipment_id not in self._spc_state:
            self._spc_state[equipment_id] = deque(maxlen=window_size)

        history = self._spc_state[equipment_id]
        history.append(x)
        data = list(history)

        # Calculate or use provided statistics
        if target is not None:
            center = self._to_decimal(target)
        else:
            center = sum(data) / Decimal(str(len(data)))

        if sigma is not None:
            sigma_val = self._to_decimal(sigma)
        elif len(data) >= 3:
            mean = sum(data) / Decimal(str(len(data)))
            variance = sum((v - mean) ** 2 for v in data) / Decimal(str(len(data) - 1))
            sigma_val = self._sqrt(variance)
        else:
            sigma_val = Decimal("1")

        if ucl is not None:
            upper_limit = self._to_decimal(ucl)
        else:
            upper_limit = center + Decimal("3") * sigma_val

        if lcl is not None:
            lower_limit = self._to_decimal(lcl)
        else:
            lower_limit = center - Decimal("3") * sigma_val

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate control limits",
            inputs={"center": center, "sigma": sigma_val},
            output_name="limits",
            output_value={"ucl": upper_limit, "lcl": lower_limit}
        )

        # Determine zone
        one_sigma = sigma_val
        two_sigma = Decimal("2") * sigma_val
        three_sigma = Decimal("3") * sigma_val

        deviation = x - center
        abs_dev = abs(deviation)

        if abs_dev <= one_sigma:
            zone = "C"  # Within 1 sigma
        elif abs_dev <= two_sigma:
            zone = "B"  # 1-2 sigma
        elif abs_dev <= three_sigma:
            zone = "A"  # 2-3 sigma
        else:
            zone = "Out"  # Beyond 3 sigma

        builder.add_step(
            step_number=2,
            operation="classify",
            description="Determine zone",
            inputs={"deviation": deviation, "one_sigma": one_sigma},
            output_name="zone",
            output_value=zone
        )

        # Check Western Electric rules
        violated_rules = []

        # Rule 1: Point beyond 3 sigma
        if abs_dev > three_sigma:
            violated_rules.append(ControlChartRule.RULE_1)

        if len(data) >= 3:
            # Rule 2: 2 of 3 points beyond 2 sigma (same side)
            recent_3 = data[-3:]
            above_2sig = sum(1 for v in recent_3 if v - center > two_sigma)
            below_2sig = sum(1 for v in recent_3 if center - v > two_sigma)
            if above_2sig >= 2 or below_2sig >= 2:
                violated_rules.append(ControlChartRule.RULE_2)

        if len(data) >= 5:
            # Rule 3: 4 of 5 points beyond 1 sigma (same side)
            recent_5 = data[-5:]
            above_1sig = sum(1 for v in recent_5 if v - center > one_sigma)
            below_1sig = sum(1 for v in recent_5 if center - v > one_sigma)
            if above_1sig >= 4 or below_1sig >= 4:
                violated_rules.append(ControlChartRule.RULE_3)

        if len(data) >= 8:
            # Rule 4: 8 points in a row on one side
            recent_8 = data[-8:]
            all_above = all(v > center for v in recent_8)
            all_below = all(v < center for v in recent_8)
            if all_above or all_below:
                violated_rules.append(ControlChartRule.RULE_4)

        if len(data) >= 6:
            # Rule 5: 6 points in a row increasing or decreasing
            recent_6 = data[-6:]
            all_increasing = all(recent_6[i] < recent_6[i+1] for i in range(5))
            all_decreasing = all(recent_6[i] > recent_6[i+1] for i in range(5))
            if all_increasing or all_decreasing:
                violated_rules.append(ControlChartRule.RULE_5)

        in_control = len(violated_rules) == 0

        builder.add_step(
            step_number=3,
            operation="evaluate",
            description="Check Western Electric rules",
            inputs={"num_points": len(data)},
            output_name="violated_rules",
            output_value=[r.name for r in violated_rules]
        )

        # Calculate process capability if enough data
        if len(data) >= 10 and sigma_val > Decimal("0"):
            # Cp = (USL - LSL) / (6 * sigma)
            # Using control limits as spec limits for illustration
            cp = (upper_limit - lower_limit) / (Decimal("6") * sigma_val)
        else:
            cp = None

        builder.add_output("in_control", in_control)
        builder.add_output("zone", zone)
        builder.add_output("num_rules_violated", len(violated_rules))

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return SPCResult(
            in_control=in_control,
            violated_rules=tuple(violated_rules),
            ucl=self._apply_precision(upper_limit, 4),
            lcl=self._apply_precision(lower_limit, 4),
            center_line=self._apply_precision(center, 4),
            current_value=x,
            zone=zone,
            process_capability=self._apply_precision(cp, 4) if cp else None,
            provenance_hash=provenance.final_hash
        )

    def reset_spc(self, equipment_id: str) -> None:
        """Reset SPC history for equipment."""
        if equipment_id in self._spc_state:
            del self._spc_state[equipment_id]

    # =========================================================================
    # MAHALANOBIS DISTANCE
    # =========================================================================

    def calculate_mahalanobis_distance(
        self,
        observation: Dict[str, Union[Decimal, float]],
        reference_data: List[Dict[str, Union[Decimal, float]]],
        threshold_percentile: Union[Decimal, float, str] = "95"
    ) -> MahalanobisResult:
        """
        Calculate Mahalanobis distance for multivariate anomaly detection.

        The Mahalanobis distance accounts for correlations between variables:
            D = sqrt((x - mu)' * S^(-1) * (x - mu))

        Under multivariate normality, D^2 follows chi-square distribution
        with p degrees of freedom (p = number of variables).

        Args:
            observation: Current observation (dict of variable: value)
            reference_data: Historical reference data
            threshold_percentile: Chi-square percentile for threshold

        Returns:
            MahalanobisResult

        Reference:
            Mahalanobis, P.C. (1936). On the generalised distance in statistics.

        Example:
            >>> detector = AnomalyDetector()
            >>> obs = {"temp": 85, "vibration": 5.2, "current": 12.5}
            >>> ref = [{"temp": 80, "vibration": 4.0, "current": 10.0}, ...]
            >>> result = detector.calculate_mahalanobis_distance(obs, ref)
        """
        builder = ProvenanceBuilder(CalculationType.ANOMALY_DETECTION)

        if len(reference_data) < 5:
            raise ValueError("Need at least 5 reference observations")

        # Get variable names
        variables = list(observation.keys())
        n_vars = len(variables)

        builder.add_input("num_variables", n_vars)
        builder.add_input("num_reference", len(reference_data))
        builder.add_input("threshold_percentile", threshold_percentile)

        # Convert observation to Decimal list
        x = [self._to_decimal(observation[var]) for var in variables]

        # Convert reference data to matrix
        ref_matrix = []
        for ref in reference_data:
            row = [self._to_decimal(ref[var]) for var in variables]
            ref_matrix.append(row)

        # Step 1: Calculate means
        n_ref = len(ref_matrix)
        means = []
        for j in range(n_vars):
            col_mean = sum(ref_matrix[i][j] for i in range(n_ref)) / Decimal(str(n_ref))
            means.append(col_mean)

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate mean vector",
            inputs={"num_variables": n_vars},
            output_name="means",
            output_value=[str(m) for m in means]
        )

        # Step 2: Calculate covariance matrix
        cov_matrix = [[Decimal("0")] * n_vars for _ in range(n_vars)]
        for i in range(n_vars):
            for j in range(n_vars):
                cov = sum(
                    (ref_matrix[k][i] - means[i]) * (ref_matrix[k][j] - means[j])
                    for k in range(n_ref)
                ) / Decimal(str(n_ref - 1))
                cov_matrix[i][j] = cov

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate covariance matrix",
            inputs={"dimension": f"{n_vars}x{n_vars}"},
            output_name="covariance_matrix",
            output_value="Computed"
        )

        # Step 3: Calculate inverse covariance matrix (simplified for small dimensions)
        # For production, use proper matrix inversion
        if n_vars == 1:
            if cov_matrix[0][0] > Decimal("0"):
                inv_cov = [[Decimal("1") / cov_matrix[0][0]]]
            else:
                inv_cov = [[Decimal("1")]]
        elif n_vars == 2:
            # 2x2 matrix inversion
            det = cov_matrix[0][0] * cov_matrix[1][1] - cov_matrix[0][1] * cov_matrix[1][0]
            if abs(det) > Decimal("1e-10"):
                inv_cov = [
                    [cov_matrix[1][1] / det, -cov_matrix[0][1] / det],
                    [-cov_matrix[1][0] / det, cov_matrix[0][0] / det]
                ]
            else:
                # Singular matrix - use identity
                inv_cov = [[Decimal("1"), Decimal("0")], [Decimal("0"), Decimal("1")]]
        else:
            # For higher dimensions, use diagonal approximation
            inv_cov = [[Decimal("0")] * n_vars for _ in range(n_vars)]
            for i in range(n_vars):
                if cov_matrix[i][i] > Decimal("0"):
                    inv_cov[i][i] = Decimal("1") / cov_matrix[i][i]
                else:
                    inv_cov[i][i] = Decimal("1")

        # Step 4: Calculate Mahalanobis distance
        # D^2 = (x - mu)' * S^(-1) * (x - mu)
        diff = [x[i] - means[i] for i in range(n_vars)]

        # Matrix multiplication: diff' * inv_cov * diff
        temp = [sum(diff[j] * inv_cov[j][i] for j in range(n_vars)) for i in range(n_vars)]
        d_squared = sum(temp[i] * diff[i] for i in range(n_vars))
        distance = self._sqrt(max(d_squared, Decimal("0")))

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate Mahalanobis distance",
            inputs={"diff_vector": [str(d) for d in diff]},
            output_name="distance",
            output_value=distance,
            formula="D = sqrt((x-mu)' * S^(-1) * (x-mu))"
        )

        # Step 5: Determine threshold from chi-square distribution
        percentile = self._to_decimal(threshold_percentile) / Decimal("100")
        # Chi-square critical value (approximation for common cases)
        chi_sq_critical = {
            1: {"0.90": 2.706, "0.95": 3.841, "0.99": 6.635},
            2: {"0.90": 4.605, "0.95": 5.991, "0.99": 9.210},
            3: {"0.90": 6.251, "0.95": 7.815, "0.99": 11.345},
            4: {"0.90": 7.779, "0.95": 9.488, "0.99": 13.277},
            5: {"0.90": 9.236, "0.95": 11.070, "0.99": 15.086},
        }

        pct_key = str(threshold_percentile).replace("%", "")
        if pct_key.startswith("9") and len(pct_key) == 2:
            pct_key = "0." + pct_key
        else:
            pct_key = "0.95"  # Default

        if n_vars <= 5:
            chi_sq_thresh = Decimal(str(chi_sq_critical.get(n_vars, {}).get(pct_key, 10)))
        else:
            # Approximation for larger dimensions
            chi_sq_thresh = Decimal(str(n_vars)) + Decimal("2") * self._sqrt(Decimal(str(n_vars * 2)))

        threshold = self._sqrt(chi_sq_thresh)

        # Determine if outlier
        is_outlier = d_squared > chi_sq_thresh

        # Find contributing variables (those furthest from mean in standardized terms)
        contributions = []
        for i, var in enumerate(variables):
            if cov_matrix[i][i] > Decimal("0"):
                std = self._sqrt(cov_matrix[i][i])
                z = abs(diff[i]) / std
                contributions.append((var, z))
            else:
                contributions.append((var, abs(diff[i])))

        contributions.sort(key=lambda x: x[1], reverse=True)
        contributing_vars = tuple(v[0] for v in contributions[:min(3, n_vars)])

        # Calculate chi-square percentile of observed distance
        # This is an approximation
        chi_sq_pct = self._chi_sq_cdf(d_squared, n_vars)

        builder.add_output("distance", distance)
        builder.add_output("is_outlier", is_outlier)
        builder.add_output("contributing_variables", contributing_vars)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return MahalanobisResult(
            distance=self._apply_precision(distance, 4),
            is_outlier=is_outlier,
            threshold=self._apply_precision(threshold, 4),
            contributing_variables=contributing_vars,
            chi_square_percentile=self._apply_precision(chi_sq_pct * Decimal("100"), 2),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # ISOLATION FOREST (SIMPLIFIED)
    # =========================================================================

    def isolation_forest_score(
        self,
        observation: List[Union[Decimal, float]],
        reference_data: List[List[Union[Decimal, float]]],
        n_trees: int = 100,
        sample_size: Optional[int] = None,
        contamination: Union[Decimal, float, str] = "0.1",
        random_seed: int = 42
    ) -> IsolationForestResult:
        """
        Calculate Isolation Forest anomaly score.

        Isolation Forest isolates observations by randomly selecting
        a feature and randomly selecting a split value. Anomalies
        require fewer splits to isolate.

        Score(x) = 2^(-E(h(x)) / c(n))

        Where:
            h(x) = path length to isolate x
            c(n) = average path length in unsuccessful search in BST
            E[h(x)] = average of h(x) over all trees

        Args:
            observation: Current observation vector
            reference_data: Reference data points
            n_trees: Number of isolation trees
            sample_size: Subsample size for each tree
            contamination: Expected proportion of outliers
            random_seed: Random seed for reproducibility

        Returns:
            IsolationForestResult

        Reference:
            Liu, Ting, Zhou (2008). Isolation Forest. ICDM.

        Example:
            >>> detector = AnomalyDetector()
            >>> obs = [85.0, 5.2, 12.5]
            >>> ref = [[80, 4.0, 10], [82, 4.5, 11], ...]
            >>> result = detector.isolation_forest_score(obs, ref)
        """
        builder = ProvenanceBuilder(CalculationType.ANOMALY_DETECTION)

        # Set random seed for reproducibility (deterministic!)
        random.seed(random_seed)

        n_samples = len(reference_data)
        n_features = len(observation)

        if sample_size is None:
            sample_size = min(256, n_samples)

        cont = self._to_decimal(contamination)

        builder.add_input("n_samples", n_samples)
        builder.add_input("n_features", n_features)
        builder.add_input("n_trees", n_trees)
        builder.add_input("sample_size", sample_size)
        builder.add_input("contamination", cont)

        # Convert to Decimal
        obs = [self._to_decimal(v) for v in observation]
        ref = [[self._to_decimal(v) for v in row] for row in reference_data]

        # Step 1: Build isolation trees and calculate path lengths
        path_lengths = []

        for tree_idx in range(n_trees):
            # Sample data for this tree
            if n_samples > sample_size:
                indices = random.sample(range(n_samples), sample_size)
                tree_data = [ref[i] for i in indices]
            else:
                tree_data = ref.copy()

            # Calculate path length for observation
            path_length = self._isolation_path_length(obs, tree_data, 0, len(tree_data))
            path_lengths.append(path_length)

        builder.add_step(
            step_number=1,
            operation="build",
            description="Build isolation trees",
            inputs={"n_trees": n_trees, "sample_size": sample_size},
            output_name="path_lengths",
            output_value=[str(p) for p in path_lengths[:5]]  # First 5
        )

        # Step 2: Calculate average path length
        avg_path_length = sum(path_lengths) / Decimal(str(n_trees))

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate average path length",
            inputs={"num_trees": n_trees},
            output_name="avg_path_length",
            output_value=avg_path_length
        )

        # Step 3: Calculate expected path length for n samples
        # c(n) = 2H(n-1) - 2(n-1)/n, where H is harmonic number
        # Approximation: c(n) ~ 2(ln(n-1) + 0.5772) - 2(n-1)/n
        if sample_size > 2:
            n_minus_1 = sample_size - 1
            harmonic = Decimal(str(math.log(n_minus_1))) + Decimal("0.5772156649")
            expected_path = Decimal("2") * harmonic - Decimal("2") * Decimal(str(n_minus_1)) / Decimal(str(sample_size))
        else:
            expected_path = Decimal("1")

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate expected path length c(n)",
            inputs={"n": sample_size},
            output_name="expected_path",
            output_value=expected_path,
            formula="c(n) = 2H(n-1) - 2(n-1)/n"
        )

        # Step 4: Calculate anomaly score
        # Score = 2^(-avg_path / c(n))
        if expected_path > Decimal("0"):
            exponent = -avg_path_length / expected_path
            anomaly_score = self._power(Decimal("2"), exponent)
        else:
            anomaly_score = Decimal("0.5")

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate anomaly score",
            inputs={"avg_path": avg_path_length, "expected_path": expected_path},
            output_name="anomaly_score",
            output_value=anomaly_score,
            formula="Score = 2^(-E[h(x)]/c(n))",
            reference="Liu et al. (2008)"
        )

        # Step 5: Determine if anomaly based on contamination
        # Threshold = score at contamination percentile
        # Simplified: use fixed threshold based on contamination
        threshold = Decimal("0.5") + cont * Decimal("0.2")  # Simplified threshold
        is_anomaly = anomaly_score > threshold

        builder.add_output("anomaly_score", anomaly_score)
        builder.add_output("is_anomaly", is_anomaly)
        builder.add_output("avg_path_length", avg_path_length)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return IsolationForestResult(
            anomaly_score=self._apply_precision(anomaly_score, 4),
            is_anomaly=is_anomaly,
            average_path_length=self._apply_precision(avg_path_length, 4),
            expected_path_length=self._apply_precision(expected_path, 4),
            contamination_threshold=self._apply_precision(threshold, 4),
            provenance_hash=provenance.final_hash
        )

    def _isolation_path_length(
        self,
        point: List[Decimal],
        data: List[List[Decimal]],
        current_depth: int,
        current_size: int
    ) -> Decimal:
        """Calculate path length to isolate a point."""
        max_depth = 20  # Limit depth for efficiency

        if current_depth >= max_depth or current_size <= 1:
            return Decimal(str(current_depth)) + self._c(current_size)

        # Random feature and split
        n_features = len(point)
        feature_idx = random.randint(0, n_features - 1)

        feature_values = [row[feature_idx] for row in data]
        min_val = min(feature_values)
        max_val = max(feature_values)

        if min_val == max_val:
            return Decimal(str(current_depth)) + self._c(current_size)

        # Random split point
        split = min_val + Decimal(str(random.random())) * (max_val - min_val)

        # Partition data
        left_data = [row for row in data if row[feature_idx] < split]
        right_data = [row for row in data if row[feature_idx] >= split]

        # Recurse
        if point[feature_idx] < split:
            return self._isolation_path_length(point, left_data, current_depth + 1, len(left_data))
        else:
            return self._isolation_path_length(point, right_data, current_depth + 1, len(right_data))

    def _c(self, n: int) -> Decimal:
        """Average path length in unsuccessful search in BST."""
        if n <= 1:
            return Decimal("0")
        n_dec = Decimal(str(n))
        harmonic = Decimal(str(math.log(max(n - 1, 1)))) + Decimal("0.5772156649")
        return Decimal("2") * harmonic - Decimal("2") * (n_dec - Decimal("1")) / n_dec

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _sqrt(self, x: Decimal) -> Decimal:
        """Calculate square root."""
        if x < Decimal("0"):
            raise ValueError("Cannot take square root of negative number")
        if x == Decimal("0"):
            return Decimal("0")
        return Decimal(str(math.sqrt(float(x))))

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate e^x."""
        if x == Decimal("0"):
            return Decimal("1")
        if x < Decimal("-700"):
            return Decimal("0")
        return Decimal(str(math.exp(float(x))))

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate base^exponent."""
        if base == Decimal("0"):
            return Decimal("0") if exponent > Decimal("0") else Decimal("1")
        if exponent == Decimal("0"):
            return Decimal("1")
        return Decimal(str(math.pow(float(base), float(exponent))))

    def _median(self, data: List[Decimal]) -> Decimal:
        """Calculate median."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            return sorted_data[n // 2]
        else:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / Decimal("2")

    def _median_absolute_deviation(self, data: List[Decimal]) -> Decimal:
        """Calculate MAD."""
        med = self._median(data)
        deviations = [abs(x - med) for x in data]
        return self._median(deviations)

    def _normal_cdf(self, z: Decimal) -> Decimal:
        """Calculate standard normal CDF."""
        z_float = float(z)
        result = 0.5 * (1 + math.erf(z_float / math.sqrt(2)))
        return Decimal(str(result))

    def _chi_sq_cdf(self, x: Decimal, df: int) -> Decimal:
        """Approximate chi-square CDF."""
        # Using Wilson-Hilferty transformation
        if x <= Decimal("0"):
            return Decimal("0")
        x_float = float(x)
        df_float = float(df)
        z = ((x_float / df_float) ** (1/3) - (1 - 2/(9*df_float))) / math.sqrt(2/(9*df_float))
        return self._normal_cdf(Decimal(str(z)))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AnomalyType",
    "AnomalySeverity",
    "ControlChartRule",

    # Data classes
    "AnomalyResult",
    "CUSUMResult",
    "SPCResult",
    "MahalanobisResult",
    "IsolationForestResult",

    # Main class
    "AnomalyDetector",
]
