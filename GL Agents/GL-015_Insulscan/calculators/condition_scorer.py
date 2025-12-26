"""
GL-015 INSULSCAN Condition Scorer

ZERO-HALLUCINATION deterministic scoring algorithm for insulation condition
assessment. Produces repeatable 0-100 scores with severity classification.

Scoring Components:
    1. Age Factor (0-25 points)
    2. Thermal Performance Factor (0-35 points)
    3. Visual Inspection Factor (0-20 points)
    4. Thermal Image Analysis Factor (0-20 points)

Severity Classification:
    - GOOD: 80-100 (Continue monitoring)
    - FAIR: 60-79 (Plan maintenance)
    - POOR: 40-59 (Priority repair)
    - CRITICAL: 0-39 (Immediate action)

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class SeverityLevel(Enum):
    """Insulation condition severity levels."""
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class VisualDefectType(Enum):
    """Types of visual defects in insulation."""
    NONE = "none"
    MINOR_DAMAGE = "minor_damage"
    MOISTURE_STAINING = "moisture_staining"
    COMPRESSION = "compression"
    GAP_SMALL = "gap_small"
    GAP_LARGE = "gap_large"
    JACKET_DAMAGE = "jacket_damage"
    MISSING_SECTION = "missing_section"
    SAGGING = "sagging"
    CORROSION_UNDER_INSULATION = "corrosion_under_insulation"


class ThermalAnomalyType(Enum):
    """Types of thermal anomalies detected in thermal imaging."""
    NONE = "none"
    HOT_SPOT_MINOR = "hot_spot_minor"
    HOT_SPOT_MAJOR = "hot_spot_major"
    THERMAL_BRIDGING = "thermal_bridging"
    MOISTURE_INGRESS = "moisture_ingress"
    AIR_INFILTRATION = "air_infiltration"
    MISSING_INSULATION = "missing_insulation"


@dataclass(frozen=True)
class VisualInspectionData:
    """
    Visual inspection input data.

    Attributes:
        defects: List of observed defect types
        affected_area_percent: Percentage of surface affected
        jacket_condition_score: Jacket condition 0-10 (10=perfect)
        seal_condition_score: Seals and joints 0-10
    """
    defects: List[VisualDefectType]
    affected_area_percent: float
    jacket_condition_score: float
    seal_condition_score: float


@dataclass(frozen=True)
class ThermalImageData:
    """
    Thermal image analysis input data.

    Attributes:
        anomalies: List of detected thermal anomalies
        max_temperature_deviation_c: Max deviation from expected (Celsius)
        anomaly_area_percent: Percentage of surface with anomalies
        uniformity_score: Temperature uniformity 0-10 (10=uniform)
    """
    anomalies: List[ThermalAnomalyType]
    max_temperature_deviation_c: float
    anomaly_area_percent: float
    uniformity_score: float


@dataclass(frozen=True)
class ConditionScoreResult:
    """
    Immutable result container for condition scoring.

    Attributes:
        total_score: Overall condition score (0-100)
        severity: Severity classification
        age_score: Age component score (0-25)
        thermal_score: Thermal performance score (0-35)
        visual_score: Visual inspection score (0-20)
        thermal_image_score: Thermal image score (0-20)
        remaining_useful_life_years: Estimated RUL
        recommendations: List of maintenance recommendations
        provenance_hash: SHA-256 hash for audit trail
        calculation_inputs: Dictionary of all input parameters
    """
    total_score: Decimal
    severity: SeverityLevel
    age_score: Decimal
    thermal_score: Decimal
    visual_score: Decimal
    thermal_image_score: Decimal
    remaining_useful_life_years: Decimal
    recommendations: List[str]
    provenance_hash: str
    calculation_inputs: Dict[str, Any]


class InsulationConditionScorer:
    """
    Deterministic insulation condition scoring algorithm.

    ZERO-HALLUCINATION GUARANTEE:
    - Fixed scoring weights and thresholds
    - Deterministic calculations (same input -> same score)
    - No LLM inference in scoring path
    - Complete provenance tracking

    Scoring Weights:
        - Age: 25% (accounts for natural degradation)
        - Thermal Performance: 35% (primary functional indicator)
        - Visual Inspection: 20% (observable damage)
        - Thermal Imaging: 20% (non-visible issues)

    Example Usage:
        >>> scorer = InsulationConditionScorer()
        >>> result = scorer.calculate_condition_score(
        ...     age_years=5,
        ...     expected_life_years=25,
        ...     thermal_efficiency_percent=90,
        ...     visual_data=VisualInspectionData(
        ...         defects=[VisualDefectType.NONE],
        ...         affected_area_percent=0,
        ...         jacket_condition_score=9,
        ...         seal_condition_score=9
        ...     ),
        ...     thermal_data=ThermalImageData(
        ...         anomalies=[ThermalAnomalyType.NONE],
        ...         max_temperature_deviation_c=2,
        ...         anomaly_area_percent=0,
        ...         uniformity_score=9
        ...     )
        ... )
        >>> float(result.total_score) > 80
        True
        >>> result.severity == SeverityLevel.GOOD
        True

    Determinism Test:
        >>> scorer = InsulationConditionScorer()
        >>> visual = VisualInspectionData([VisualDefectType.MINOR_DAMAGE], 5, 7, 8)
        >>> thermal = ThermalImageData([ThermalAnomalyType.HOT_SPOT_MINOR], 10, 5, 7)
        >>> r1 = scorer.calculate_condition_score(10, 25, 75, visual, thermal)
        >>> r2 = scorer.calculate_condition_score(10, 25, 75, visual, thermal)
        >>> r1.total_score == r2.total_score
        True
        >>> r1.provenance_hash == r2.provenance_hash
        True
    """

    # Scoring weights (must sum to 100)
    WEIGHT_AGE = Decimal("25")
    WEIGHT_THERMAL = Decimal("35")
    WEIGHT_VISUAL = Decimal("20")
    WEIGHT_THERMAL_IMAGE = Decimal("20")

    # Severity thresholds
    THRESHOLD_GOOD = Decimal("80")
    THRESHOLD_FAIR = Decimal("60")
    THRESHOLD_POOR = Decimal("40")

    # Defect severity scores (deductions from max)
    DEFECT_SCORES: Dict[VisualDefectType, Decimal] = {
        VisualDefectType.NONE: Decimal("0"),
        VisualDefectType.MINOR_DAMAGE: Decimal("2"),
        VisualDefectType.MOISTURE_STAINING: Decimal("4"),
        VisualDefectType.COMPRESSION: Decimal("3"),
        VisualDefectType.GAP_SMALL: Decimal("3"),
        VisualDefectType.GAP_LARGE: Decimal("6"),
        VisualDefectType.JACKET_DAMAGE: Decimal("4"),
        VisualDefectType.MISSING_SECTION: Decimal("8"),
        VisualDefectType.SAGGING: Decimal("5"),
        VisualDefectType.CORROSION_UNDER_INSULATION: Decimal("10"),
    }

    # Thermal anomaly severity scores
    ANOMALY_SCORES: Dict[ThermalAnomalyType, Decimal] = {
        ThermalAnomalyType.NONE: Decimal("0"),
        ThermalAnomalyType.HOT_SPOT_MINOR: Decimal("3"),
        ThermalAnomalyType.HOT_SPOT_MAJOR: Decimal("7"),
        ThermalAnomalyType.THERMAL_BRIDGING: Decimal("5"),
        ThermalAnomalyType.MOISTURE_INGRESS: Decimal("6"),
        ThermalAnomalyType.AIR_INFILTRATION: Decimal("4"),
        ThermalAnomalyType.MISSING_INSULATION: Decimal("10"),
    }

    # Expected insulation lifespans by type (years)
    EXPECTED_LIFESPAN: Dict[str, int] = {
        "mineral_wool": 25,
        "calcium_silicate": 30,
        "fiberglass": 20,
        "cellular_glass": 35,
        "perlite": 25,
        "polyurethane_foam": 15,
        "phenolic_foam": 20,
        "aerogel": 25,
        "ceramic_fiber": 20,
        "default": 20,
    }

    def __init__(self, precision: int = 1):
        """
        Initialize scorer with specified decimal precision.

        Args:
            precision: Number of decimal places for scores (default: 1)
        """
        self.precision = precision
        self._quantize_str = "0." + "0" * precision

    def calculate_condition_score(
        self,
        age_years: float,
        expected_life_years: float,
        thermal_efficiency_percent: float,
        visual_data: VisualInspectionData,
        thermal_data: ThermalImageData,
        insulation_type: str = "default"
    ) -> ConditionScoreResult:
        """
        Calculate comprehensive insulation condition score.

        Args:
            age_years: Current age of insulation
            expected_life_years: Design life of insulation
            thermal_efficiency_percent: Current thermal efficiency (0-100)
            visual_data: Visual inspection data
            thermal_data: Thermal imaging data
            insulation_type: Type of insulation for RUL estimation

        Returns:
            ConditionScoreResult with score breakdown and recommendations

        Example - New Installation:
            >>> scorer = InsulationConditionScorer()
            >>> visual = VisualInspectionData([VisualDefectType.NONE], 0, 10, 10)
            >>> thermal = ThermalImageData([ThermalAnomalyType.NONE], 0, 0, 10)
            >>> result = scorer.calculate_condition_score(0, 25, 100, visual, thermal)
            >>> float(result.total_score) >= 95
            True

        Example - Aged Installation:
            >>> scorer = InsulationConditionScorer()
            >>> visual = VisualInspectionData(
            ...     [VisualDefectType.MOISTURE_STAINING, VisualDefectType.GAP_SMALL],
            ...     15, 6, 5
            ... )
            >>> thermal = ThermalImageData(
            ...     [ThermalAnomalyType.HOT_SPOT_MINOR, ThermalAnomalyType.THERMAL_BRIDGING],
            ...     20, 10, 5
            ... )
            >>> result = scorer.calculate_condition_score(15, 25, 60, visual, thermal)
            >>> 40 < float(result.total_score) < 70
            True

        Example - Critical Condition:
            >>> scorer = InsulationConditionScorer()
            >>> visual = VisualInspectionData(
            ...     [VisualDefectType.MISSING_SECTION, VisualDefectType.CORROSION_UNDER_INSULATION],
            ...     40, 2, 2
            ... )
            >>> thermal = ThermalImageData(
            ...     [ThermalAnomalyType.HOT_SPOT_MAJOR, ThermalAnomalyType.MISSING_INSULATION],
            ...     50, 35, 2
            ... )
            >>> result = scorer.calculate_condition_score(20, 20, 30, visual, thermal)
            >>> result.severity == SeverityLevel.CRITICAL
            True
        """
        # Convert inputs to Decimal
        age = Decimal(str(age_years))
        expected_life = Decimal(str(expected_life_years))
        thermal_eff = Decimal(str(thermal_efficiency_percent))

        # Validate inputs
        self._validate_non_negative("age_years", age)
        self._validate_positive("expected_life_years", expected_life)
        self._validate_range("thermal_efficiency_percent", thermal_eff,
                            Decimal("0"), Decimal("100"))

        # Calculate component scores
        age_score = self._calculate_age_score(age, expected_life)
        thermal_score = self._calculate_thermal_score(thermal_eff)
        visual_score = self._calculate_visual_score(visual_data)
        thermal_image_score = self._calculate_thermal_image_score(thermal_data)

        # Calculate total weighted score
        total_score = (
            age_score +
            thermal_score +
            visual_score +
            thermal_image_score
        )

        # Determine severity level
        severity = self._classify_severity(total_score)

        # Estimate remaining useful life
        rul = self._estimate_remaining_life(
            age, expected_life, total_score, insulation_type
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, age_score, thermal_score, visual_score,
            thermal_image_score, visual_data, thermal_data
        )

        # Apply precision
        total_score = self._apply_precision(total_score)
        age_score = self._apply_precision(age_score)
        thermal_score = self._apply_precision(thermal_score)
        visual_score = self._apply_precision(visual_score)
        thermal_image_score = self._apply_precision(thermal_image_score)
        rul = self._apply_precision(rul)

        # Build provenance
        inputs = {
            "age_years": str(age_years),
            "expected_life_years": str(expected_life_years),
            "thermal_efficiency_percent": str(thermal_efficiency_percent),
            "visual_defects": [d.value for d in visual_data.defects],
            "visual_affected_area_percent": visual_data.affected_area_percent,
            "visual_jacket_condition": visual_data.jacket_condition_score,
            "visual_seal_condition": visual_data.seal_condition_score,
            "thermal_anomalies": [a.value for a in thermal_data.anomalies],
            "thermal_max_deviation_c": thermal_data.max_temperature_deviation_c,
            "thermal_anomaly_area_percent": thermal_data.anomaly_area_percent,
            "thermal_uniformity_score": thermal_data.uniformity_score,
            "insulation_type": insulation_type,
        }

        provenance_hash = self._calculate_provenance_hash(
            "condition_score", inputs, str(total_score)
        )

        return ConditionScoreResult(
            total_score=total_score,
            severity=severity,
            age_score=age_score,
            thermal_score=thermal_score,
            visual_score=visual_score,
            thermal_image_score=thermal_image_score,
            remaining_useful_life_years=rul,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            calculation_inputs=inputs
        )

    def _calculate_age_score(
        self,
        age: Decimal,
        expected_life: Decimal
    ) -> Decimal:
        """
        Calculate age component score (0-25 points).

        Score decreases linearly with age until expected life,
        then drops more rapidly.

        >>> scorer = InsulationConditionScorer()
        >>> score = scorer._calculate_age_score(Decimal("0"), Decimal("25"))
        >>> float(score) == 25.0
        True
        >>> score = scorer._calculate_age_score(Decimal("12.5"), Decimal("25"))
        >>> 10 < float(score) < 15
        True
        """
        age_ratio = age / expected_life

        if age_ratio <= Decimal("1"):
            # Linear decrease: 25 at age 0, scales down
            # At 50% of life, score is 50% of max
            score = self.WEIGHT_AGE * (Decimal("1") - age_ratio * Decimal("0.6"))
        else:
            # Accelerated decrease past expected life
            excess_ratio = age_ratio - Decimal("1")
            base_score = self.WEIGHT_AGE * Decimal("0.4")
            penalty = min(excess_ratio * Decimal("0.8"), Decimal("1")) * base_score
            score = base_score - penalty

        return max(score, Decimal("0"))

    def _calculate_thermal_score(
        self,
        thermal_efficiency: Decimal
    ) -> Decimal:
        """
        Calculate thermal performance score (0-35 points).

        Based on thermal efficiency as percentage of design performance.

        >>> scorer = InsulationConditionScorer()
        >>> score = scorer._calculate_thermal_score(Decimal("100"))
        >>> float(score) == 35.0
        True
        >>> score = scorer._calculate_thermal_score(Decimal("70"))
        >>> 20 < float(score) < 30
        True
        """
        # Direct linear relationship
        score = self.WEIGHT_THERMAL * (thermal_efficiency / Decimal("100"))

        return max(score, Decimal("0"))

    def _calculate_visual_score(
        self,
        visual_data: VisualInspectionData
    ) -> Decimal:
        """
        Calculate visual inspection score (0-20 points).

        Deductions for defects, area affected, and condition scores.

        >>> scorer = InsulationConditionScorer()
        >>> data = VisualInspectionData([VisualDefectType.NONE], 0, 10, 10)
        >>> score = scorer._calculate_visual_score(data)
        >>> float(score) == 20.0
        True
        """
        score = self.WEIGHT_VISUAL

        # Deduct for each defect type
        for defect in visual_data.defects:
            score -= self.DEFECT_SCORES.get(defect, Decimal("2"))

        # Deduct for affected area (up to 5 points)
        area_penalty = min(
            Decimal(str(visual_data.affected_area_percent)) * Decimal("0.1"),
            Decimal("5")
        )
        score -= area_penalty

        # Factor in jacket and seal condition (each contributes up to 2.5 points)
        jacket_factor = Decimal(str(visual_data.jacket_condition_score)) / Decimal("10")
        seal_factor = Decimal(str(visual_data.seal_condition_score)) / Decimal("10")

        condition_bonus = (jacket_factor + seal_factor) * Decimal("2.5") - Decimal("5")
        score += condition_bonus

        return max(min(score, self.WEIGHT_VISUAL), Decimal("0"))

    def _calculate_thermal_image_score(
        self,
        thermal_data: ThermalImageData
    ) -> Decimal:
        """
        Calculate thermal imaging score (0-20 points).

        Based on anomalies detected and their severity.

        >>> scorer = InsulationConditionScorer()
        >>> data = ThermalImageData([ThermalAnomalyType.NONE], 0, 0, 10)
        >>> score = scorer._calculate_thermal_image_score(data)
        >>> float(score) == 20.0
        True
        """
        score = self.WEIGHT_THERMAL_IMAGE

        # Deduct for each anomaly type
        for anomaly in thermal_data.anomalies:
            score -= self.ANOMALY_SCORES.get(anomaly, Decimal("2"))

        # Deduct for temperature deviation (up to 5 points)
        # >30C deviation is severe
        deviation = Decimal(str(thermal_data.max_temperature_deviation_c))
        deviation_penalty = min(deviation / Decimal("6"), Decimal("5"))
        score -= deviation_penalty

        # Deduct for anomaly area (up to 3 points)
        area_penalty = min(
            Decimal(str(thermal_data.anomaly_area_percent)) * Decimal("0.1"),
            Decimal("3")
        )
        score -= area_penalty

        # Factor in uniformity score (up to 2 points bonus/penalty)
        uniformity = Decimal(str(thermal_data.uniformity_score))
        uniformity_factor = (uniformity / Decimal("10") - Decimal("0.5")) * Decimal("4")
        score += uniformity_factor

        return max(min(score, self.WEIGHT_THERMAL_IMAGE), Decimal("0"))

    def _classify_severity(self, score: Decimal) -> SeverityLevel:
        """
        Classify severity based on total score.

        >>> scorer = InsulationConditionScorer()
        >>> scorer._classify_severity(Decimal("85"))
        <SeverityLevel.GOOD: 'good'>
        >>> scorer._classify_severity(Decimal("65"))
        <SeverityLevel.FAIR: 'fair'>
        >>> scorer._classify_severity(Decimal("45"))
        <SeverityLevel.POOR: 'poor'>
        >>> scorer._classify_severity(Decimal("30"))
        <SeverityLevel.CRITICAL: 'critical'>
        """
        if score >= self.THRESHOLD_GOOD:
            return SeverityLevel.GOOD
        elif score >= self.THRESHOLD_FAIR:
            return SeverityLevel.FAIR
        elif score >= self.THRESHOLD_POOR:
            return SeverityLevel.POOR
        else:
            return SeverityLevel.CRITICAL

    def _estimate_remaining_life(
        self,
        age: Decimal,
        expected_life: Decimal,
        score: Decimal,
        insulation_type: str
    ) -> Decimal:
        """
        Estimate remaining useful life based on condition score.

        RUL = (expected_life - age) * (score / 100) * degradation_factor

        >>> scorer = InsulationConditionScorer()
        >>> rul = scorer._estimate_remaining_life(
        ...     Decimal("5"), Decimal("25"), Decimal("90"), "mineral_wool"
        ... )
        >>> 15 < float(rul) < 25
        True
        """
        # Base remaining life
        base_remaining = expected_life - age

        if base_remaining <= 0:
            # Past expected life - use score to estimate extension
            return max(score / Decimal("10") - Decimal("5"), Decimal("0"))

        # Adjust by condition score
        # Score of 100 = full remaining life
        # Score of 50 = half remaining life
        score_factor = score / Decimal("100")

        # Apply degradation curve (faster degradation at lower scores)
        if score < Decimal("60"):
            score_factor = score_factor ** Decimal("1.5")

        rul = base_remaining * score_factor

        return max(rul, Decimal("0"))

    def _generate_recommendations(
        self,
        severity: SeverityLevel,
        age_score: Decimal,
        thermal_score: Decimal,
        visual_score: Decimal,
        thermal_image_score: Decimal,
        visual_data: VisualInspectionData,
        thermal_data: ThermalImageData
    ) -> List[str]:
        """Generate actionable maintenance recommendations."""
        recommendations = []

        # Severity-based recommendations
        if severity == SeverityLevel.CRITICAL:
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED: Schedule emergency insulation repair/replacement"
            )
        elif severity == SeverityLevel.POOR:
            recommendations.append(
                "PRIORITY: Plan insulation maintenance within 30 days"
            )
        elif severity == SeverityLevel.FAIR:
            recommendations.append(
                "Schedule preventive maintenance within next quarter"
            )

        # Age-specific recommendations
        if age_score < self.WEIGHT_AGE * Decimal("0.4"):
            recommendations.append(
                "Insulation approaching end of service life - plan replacement"
            )

        # Thermal performance recommendations
        if thermal_score < self.WEIGHT_THERMAL * Decimal("0.6"):
            recommendations.append(
                "Thermal performance degraded - investigate root cause"
            )

        # Visual inspection recommendations
        for defect in visual_data.defects:
            if defect == VisualDefectType.MOISTURE_STAINING:
                recommendations.append(
                    "Moisture ingress detected - repair weatherproofing and check for CUI"
                )
            elif defect == VisualDefectType.MISSING_SECTION:
                recommendations.append(
                    "Missing insulation sections - install replacement material"
                )
            elif defect == VisualDefectType.CORROSION_UNDER_INSULATION:
                recommendations.append(
                    "CUI detected - immediate pipe inspection and remediation required"
                )
            elif defect == VisualDefectType.GAP_LARGE:
                recommendations.append(
                    "Large gaps in insulation - seal and fill to prevent energy loss"
                )

        # Thermal imaging recommendations
        for anomaly in thermal_data.anomalies:
            if anomaly == ThermalAnomalyType.HOT_SPOT_MAJOR:
                recommendations.append(
                    "Major hot spots detected - investigate insulation voids or damage"
                )
            elif anomaly == ThermalAnomalyType.MOISTURE_INGRESS:
                recommendations.append(
                    "Thermal imaging indicates moisture - conduct invasive inspection"
                )

        # General monitoring recommendation for good condition
        if severity == SeverityLevel.GOOD:
            recommendations.append(
                "Continue routine monitoring per maintenance schedule"
            )

        return recommendations

    def calculate_batch_scores(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Tuple[str, ConditionScoreResult]]:
        """
        Calculate condition scores for multiple items.

        Args:
            items: List of dictionaries with item_id and scoring parameters

        Returns:
            List of (item_id, ConditionScoreResult) tuples

        Example:
            >>> scorer = InsulationConditionScorer()
            >>> items = [
            ...     {
            ...         "item_id": "PIPE-001",
            ...         "age_years": 5,
            ...         "expected_life_years": 25,
            ...         "thermal_efficiency_percent": 95,
            ...         "visual_data": VisualInspectionData(
            ...             [VisualDefectType.NONE], 0, 9, 9
            ...         ),
            ...         "thermal_data": ThermalImageData(
            ...             [ThermalAnomalyType.NONE], 1, 0, 9
            ...         )
            ...     }
            ... ]
            >>> results = scorer.calculate_batch_scores(items)
            >>> len(results) == 1
            True
        """
        results = []

        for item in items:
            item_id = item.get("item_id", "unknown")

            result = self.calculate_condition_score(
                age_years=item["age_years"],
                expected_life_years=item["expected_life_years"],
                thermal_efficiency_percent=item["thermal_efficiency_percent"],
                visual_data=item["visual_data"],
                thermal_data=item["thermal_data"],
                insulation_type=item.get("insulation_type", "default")
            )

            results.append((item_id, result))

        return results

    def get_severity_statistics(
        self,
        results: List[ConditionScoreResult]
    ) -> Dict[SeverityLevel, int]:
        """
        Calculate severity distribution from batch results.

        >>> scorer = InsulationConditionScorer()
        >>> from decimal import Decimal
        >>> results = [
        ...     ConditionScoreResult(
        ...         total_score=Decimal("85"), severity=SeverityLevel.GOOD,
        ...         age_score=Decimal("20"), thermal_score=Decimal("30"),
        ...         visual_score=Decimal("18"), thermal_image_score=Decimal("17"),
        ...         remaining_useful_life_years=Decimal("15"), recommendations=[],
        ...         provenance_hash="abc", calculation_inputs={}
        ...     ),
        ...     ConditionScoreResult(
        ...         total_score=Decimal("55"), severity=SeverityLevel.POOR,
        ...         age_score=Decimal("10"), thermal_score=Decimal("20"),
        ...         visual_score=Decimal("12"), thermal_image_score=Decimal("13"),
        ...         remaining_useful_life_years=Decimal("5"), recommendations=[],
        ...         provenance_hash="def", calculation_inputs={}
        ...     )
        ... ]
        >>> stats = scorer.get_severity_statistics(results)
        >>> stats[SeverityLevel.GOOD]
        1
        >>> stats[SeverityLevel.POOR]
        1
        """
        stats = {
            SeverityLevel.GOOD: 0,
            SeverityLevel.FAIR: 0,
            SeverityLevel.POOR: 0,
            SeverityLevel.CRITICAL: 0,
        }

        for result in results:
            stats[result.severity] += 1

        return stats

    def _validate_positive(self, name: str, value: Decimal) -> None:
        """Validate value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def _validate_non_negative(self, name: str, value: Decimal) -> None:
        """Validate value is non-negative."""
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    def _validate_range(
        self,
        name: str,
        value: Decimal,
        min_val: Decimal,
        max_val: Decimal
    ) -> None:
        """Validate value is within range."""
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply configured precision using ROUND_HALF_UP."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance_hash(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        result: str
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "calculator": "InsulationConditionScorer",
            "version": "1.0.0",
            "calculation_type": calculation_type,
            "inputs": inputs,
            "result": result,
            "weights": {
                "age": str(self.WEIGHT_AGE),
                "thermal": str(self.WEIGHT_THERMAL),
                "visual": str(self.WEIGHT_VISUAL),
                "thermal_image": str(self.WEIGHT_THERMAL_IMAGE),
            }
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
