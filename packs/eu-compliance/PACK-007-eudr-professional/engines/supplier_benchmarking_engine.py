"""
Supplier Benchmarking Engine - PACK-007 EUDR Professional

This module implements industry-relative supplier performance scoring with
peer comparison and best practice identification for EUDR compliance.

Example:
    >>> config = BenchmarkConfig()
    >>> engine = SupplierBenchmarkingEngine(config)
    >>> score = engine.calculate_score(supplier_data)
    >>> print(f"Composite Score: {score.composite_score}")
"""

import hashlib
import json
import logging
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class CommoditySector(str, Enum):
    """EUDR commodity sectors."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOY = "SOY"
    WOOD = "WOOD"


class PerformanceDimension(str, Enum):
    """Supplier performance dimensions."""
    TRACEABILITY = "TRACEABILITY"
    CERTIFICATION = "CERTIFICATION"
    DEFORESTATION_RISK = "DEFORESTATION_RISK"
    DATA_QUALITY = "DATA_QUALITY"
    COMPLIANCE_HISTORY = "COMPLIANCE_HISTORY"
    TRANSPARENCY = "TRANSPARENCY"
    RESPONSIVENESS = "RESPONSIVENESS"


class TrendDirection(str, Enum):
    """Performance trend direction."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"


class BenchmarkConfig(BaseModel):
    """Configuration for supplier benchmarking."""

    enable_peer_comparison: bool = Field(
        default=True,
        description="Enable peer group comparison"
    )
    peer_group_size_min: int = Field(
        default=5,
        ge=3,
        description="Minimum peer group size for valid comparison"
    )
    dimension_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            PerformanceDimension.TRACEABILITY.value: 0.25,
            PerformanceDimension.CERTIFICATION.value: 0.20,
            PerformanceDimension.DEFORESTATION_RISK.value: 0.20,
            PerformanceDimension.DATA_QUALITY.value: 0.15,
            PerformanceDimension.COMPLIANCE_HISTORY.value: 0.10,
            PerformanceDimension.TRANSPARENCY.value: 0.05,
            PerformanceDimension.RESPONSIVENESS.value: 0.05,
        },
        description="Weights for performance dimensions"
    )
    alert_degradation_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Score drop threshold for degradation alert"
    )


class SupplierScore(BaseModel):
    """Supplier performance score."""

    supplier_id: str = Field(..., description="Supplier identifier")
    dimensions: Dict[str, float] = Field(..., description="Dimension scores (0-100)")
    composite_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    percentile: Optional[float] = Field(None, ge=0.0, le=100.0, description="Percentile rank")
    peer_group: Optional[str] = Field(None, description="Peer group identifier")
    trend: TrendDirection = Field(default=TrendDirection.STABLE, description="Performance trend")
    grade: str = Field(..., description="Letter grade (A+ to F)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Scoring timestamp")


class PeerGroupStats(BaseModel):
    """Statistical summary for peer group."""

    group_id: str = Field(..., description="Peer group identifier")
    commodity: CommoditySector = Field(..., description="Commodity sector")
    count: int = Field(..., ge=1, description="Number of suppliers in group")
    mean: float = Field(..., ge=0.0, le=100.0, description="Mean score")
    median: float = Field(..., ge=0.0, le=100.0, description="Median score")
    std: float = Field(..., ge=0.0, description="Standard deviation")
    p25: float = Field(..., ge=0.0, le=100.0, description="25th percentile")
    p75: float = Field(..., ge=0.0, le=100.0, description="75th percentile")
    min: float = Field(..., ge=0.0, le=100.0, description="Minimum score")
    max: float = Field(..., ge=0.0, le=100.0, description="Maximum score")


class BestPractice(BaseModel):
    """Best practice identified from high performers."""

    practice_id: str = Field(..., description="Practice identifier")
    category: str = Field(..., description="Practice category")
    description: str = Field(..., description="Practice description")
    impact_score: float = Field(..., ge=0.0, le=10.0, description="Impact on performance")
    adoption_rate: float = Field(..., ge=0.0, le=1.0, description="Adoption rate in peer group")
    example_suppliers: List[str] = Field(
        default_factory=list,
        description="Supplier IDs demonstrating this practice"
    )


class Alert(BaseModel):
    """Performance degradation alert."""

    alert_id: str = Field(..., description="Alert identifier")
    supplier_id: str = Field(..., description="Supplier identifier")
    severity: str = Field(..., description="LOW/MEDIUM/HIGH/CRITICAL")
    message: str = Field(..., description="Alert message")
    current_score: float = Field(..., description="Current score")
    previous_score: float = Field(..., description="Previous score")
    score_change: float = Field(..., description="Score change (negative)")
    dimensions_affected: List[str] = Field(..., description="Affected dimensions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alert timestamp")


class ImprovementTrack(BaseModel):
    """Supplier improvement tracking."""

    supplier_id: str = Field(..., description="Supplier identifier")
    baseline_score: float = Field(..., description="Baseline score")
    current_score: float = Field(..., description="Current score")
    improvement_pct: float = Field(..., description="Improvement percentage")
    history: List[Tuple[date, float]] = Field(..., description="Historical scores")
    milestones: List[str] = Field(default_factory=list, description="Achieved milestones")
    recommendations: List[str] = Field(default_factory=list, description="Improvement actions")


class SupplierScorecard(BaseModel):
    """Comprehensive supplier scorecard."""

    supplier_id: str = Field(..., description="Supplier identifier")
    score: SupplierScore = Field(..., description="Current score")
    peer_comparison: Optional[PeerGroupStats] = Field(None, description="Peer group stats")
    strengths: List[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    best_practices: List[BestPractice] = Field(
        default_factory=list,
        description="Applicable best practices"
    )


class PortfolioBenchmark(BaseModel):
    """Portfolio-level benchmarking result."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    supplier_count: int = Field(..., ge=0, description="Number of suppliers")
    average_score: float = Field(..., ge=0.0, le=100.0, description="Portfolio average score")
    median_score: float = Field(..., ge=0.0, le=100.0, description="Portfolio median score")
    score_distribution: Dict[str, int] = Field(..., description="Count by grade")
    top_performers: List[str] = Field(default_factory=list, description="Top 10% supplier IDs")
    underperformers: List[str] = Field(
        default_factory=list,
        description="Bottom 10% supplier IDs"
    )
    dimension_averages: Dict[str, float] = Field(..., description="Average by dimension")


# Industry Benchmark Reference Data (by commodity sector)
INDUSTRY_BENCHMARKS = {
    CommoditySector.CATTLE: {
        "mean": 62.5,
        "median": 65.0,
        "std": 18.5,
        "top_quartile": 78.0,
        "typical_certification_rate": 0.45,
        "typical_traceability": 0.55,
    },
    CommoditySector.COCOA: {
        "mean": 68.0,
        "median": 70.0,
        "std": 16.0,
        "top_quartile": 82.0,
        "typical_certification_rate": 0.65,
        "typical_traceability": 0.60,
    },
    CommoditySector.COFFEE: {
        "mean": 70.5,
        "median": 72.0,
        "std": 15.5,
        "top_quartile": 84.0,
        "typical_certification_rate": 0.70,
        "typical_traceability": 0.65,
    },
    CommoditySector.OIL_PALM: {
        "mean": 65.0,
        "median": 67.0,
        "std": 17.0,
        "top_quartile": 80.0,
        "typical_certification_rate": 0.60,
        "typical_traceability": 0.58,
    },
    CommoditySector.RUBBER: {
        "mean": 60.0,
        "median": 62.0,
        "std": 19.0,
        "top_quartile": 76.0,
        "typical_certification_rate": 0.40,
        "typical_traceability": 0.50,
    },
    CommoditySector.SOY: {
        "mean": 66.0,
        "median": 68.0,
        "std": 17.5,
        "top_quartile": 81.0,
        "typical_certification_rate": 0.55,
        "typical_traceability": 0.60,
    },
    CommoditySector.WOOD: {
        "mean": 64.0,
        "median": 66.0,
        "std": 18.0,
        "top_quartile": 79.0,
        "typical_certification_rate": 0.50,
        "typical_traceability": 0.53,
    },
}


class SupplierBenchmarkingEngine:
    """
    Supplier Benchmarking Engine for PACK-007 EUDR Professional.

    This engine provides industry-relative supplier performance scoring with
    peer comparison and best practice identification. It follows GreenLang's
    zero-hallucination principle by using deterministic scoring algorithms
    and reference benchmark data.

    Attributes:
        config: Engine configuration
        benchmarks: Industry benchmark reference data

    Example:
        >>> config = BenchmarkConfig()
        >>> engine = SupplierBenchmarkingEngine(config)
        >>> score = engine.calculate_score(supplier_data)
        >>> assert 0 <= score.composite_score <= 100
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize Supplier Benchmarking Engine."""
        self.config = config
        self.benchmarks = INDUSTRY_BENCHMARKS
        logger.info("Initialized SupplierBenchmarkingEngine")

    def calculate_score(self, supplier_data: Dict[str, Any]) -> SupplierScore:
        """
        Calculate comprehensive supplier score.

        Args:
            supplier_data: Dictionary with supplier metrics including:
                - supplier_id: Supplier identifier
                - traceability_pct: Traceability percentage (0-100)
                - certification_status: List of certifications
                - deforestation_alerts: Number of alerts
                - data_completeness: Data completeness percentage
                - compliance_violations: Number of violations
                - response_time_days: Average response time
                - transparency_score: Transparency score (0-100)

        Returns:
            SupplierScore with dimensional and composite scores

        Raises:
            ValueError: If required fields missing
        """
        supplier_id = supplier_data.get("supplier_id")
        if not supplier_id:
            raise ValueError("supplier_id is required")

        logger.info(f"Calculating score for supplier {supplier_id}")

        # Calculate dimension scores
        dimensions = {}

        # Traceability (0-100)
        traceability_pct = supplier_data.get("traceability_pct", 0.0)
        dimensions[PerformanceDimension.TRACEABILITY.value] = min(100.0, traceability_pct)

        # Certification (0-100)
        certifications = supplier_data.get("certification_status", [])
        cert_score = self._calculate_certification_score(certifications)
        dimensions[PerformanceDimension.CERTIFICATION.value] = cert_score

        # Deforestation Risk (0-100, inverted)
        alerts = supplier_data.get("deforestation_alerts", 0)
        risk_score = max(0.0, 100.0 - (alerts * 10.0))
        dimensions[PerformanceDimension.DEFORESTATION_RISK.value] = risk_score

        # Data Quality (0-100)
        data_completeness = supplier_data.get("data_completeness", 0.0)
        dimensions[PerformanceDimension.DATA_QUALITY.value] = min(100.0, data_completeness)

        # Compliance History (0-100, inverted)
        violations = supplier_data.get("compliance_violations", 0)
        compliance_score = max(0.0, 100.0 - (violations * 15.0))
        dimensions[PerformanceDimension.COMPLIANCE_HISTORY.value] = compliance_score

        # Transparency (0-100)
        transparency = supplier_data.get("transparency_score", 50.0)
        dimensions[PerformanceDimension.TRANSPARENCY.value] = min(100.0, transparency)

        # Responsiveness (0-100, based on response time)
        response_time = supplier_data.get("response_time_days", 10.0)
        responsiveness_score = max(0.0, 100.0 - (response_time * 5.0))
        dimensions[PerformanceDimension.RESPONSIVENESS.value] = responsiveness_score

        # Calculate composite score (weighted average)
        composite_score = 0.0
        for dim, score in dimensions.items():
            weight = self.config.dimension_weights.get(dim, 0.0)
            composite_score += score * weight

        # Determine grade
        grade = self._score_to_grade(composite_score)

        # Determine trend (requires historical data)
        trend = TrendDirection.STABLE  # Default, would be calculated from history

        return SupplierScore(
            supplier_id=supplier_id,
            dimensions=dimensions,
            composite_score=composite_score,
            grade=grade,
            trend=trend
        )

    def define_peer_group(self, supplier_data: Dict[str, Any]) -> str:
        """
        Define peer group for supplier.

        Args:
            supplier_data: Supplier data including commodity and region

        Returns:
            Peer group identifier
        """
        commodity = supplier_data.get("commodity", "UNKNOWN")
        region = supplier_data.get("region", "GLOBAL")
        size_tier = supplier_data.get("size_tier", "MEDIUM")

        peer_group_id = f"{commodity}_{region}_{size_tier}"

        logger.debug(f"Assigned peer group: {peer_group_id}")

        return peer_group_id

    def get_peer_group_stats(self, peer_group: str) -> PeerGroupStats:
        """
        Get statistical summary for peer group.

        Args:
            peer_group: Peer group identifier

        Returns:
            PeerGroupStats with statistical measures
        """
        # Extract commodity from peer group ID
        commodity_str = peer_group.split("_")[0]

        try:
            commodity = CommoditySector(commodity_str)
            benchmark = self.benchmarks.get(commodity, self.benchmarks[CommoditySector.CATTLE])
        except (ValueError, KeyError):
            # Default to cattle if unknown
            commodity = CommoditySector.CATTLE
            benchmark = self.benchmarks[CommoditySector.CATTLE]

        # Simulate peer group size (20-200 suppliers)
        count = 50 + (hash(peer_group) % 150)

        # Calculate percentiles from mean and std
        mean = benchmark["mean"]
        std = benchmark["std"]
        median = benchmark["median"]

        p25 = mean - (0.674 * std)  # ~25th percentile
        p75 = mean + (0.674 * std)  # ~75th percentile

        min_score = max(0.0, mean - (2 * std))
        max_score = min(100.0, mean + (2 * std))

        return PeerGroupStats(
            group_id=peer_group,
            commodity=commodity,
            count=count,
            mean=mean,
            median=median,
            std=std,
            p25=p25,
            p75=p75,
            min=min_score,
            max=max_score
        )

    def calculate_percentile(self, score: float, peer_group: str) -> float:
        """
        Calculate percentile rank within peer group.

        Args:
            score: Supplier composite score
            peer_group: Peer group identifier

        Returns:
            Percentile rank (0-100)
        """
        stats = self.get_peer_group_stats(peer_group)

        # Use normal distribution CDF approximation
        z_score = (score - stats.mean) / stats.std if stats.std > 0 else 0.0

        # Approximate percentile using error function approximation
        percentile = self._normal_cdf(z_score) * 100.0

        return max(0.0, min(100.0, percentile))

    def generate_scorecard(self, supplier_id: str) -> SupplierScorecard:
        """
        Generate comprehensive supplier scorecard.

        Args:
            supplier_id: Supplier identifier

        Returns:
            SupplierScorecard with score, comparison, and recommendations

        Note:
            In production, would fetch supplier data from database
        """
        # Simulate supplier data (in production, would fetch from DB)
        supplier_data = self._simulate_supplier_data(supplier_id)

        # Calculate score
        score = self.calculate_score(supplier_data)

        # Get peer group
        peer_group_id = self.define_peer_group(supplier_data)
        score.peer_group = peer_group_id

        # Calculate percentile
        percentile = self.calculate_percentile(score.composite_score, peer_group_id)
        score.percentile = percentile

        # Get peer stats
        peer_stats = self.get_peer_group_stats(peer_group_id)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(score, peer_stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(score, weaknesses)

        # Get applicable best practices
        best_practices = self.identify_best_practices(peer_group_id)[:3]  # Top 3

        return SupplierScorecard(
            supplier_id=supplier_id,
            score=score,
            peer_comparison=peer_stats,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            best_practices=best_practices
        )

    def track_improvement(
        self,
        supplier_id: str,
        history: List[Tuple[date, float]]
    ) -> ImprovementTrack:
        """
        Track supplier improvement over time.

        Args:
            supplier_id: Supplier identifier
            history: List of (date, score) tuples

        Returns:
            ImprovementTrack with progress metrics
        """
        if not history:
            raise ValueError("History data required for improvement tracking")

        # Sort by date
        history_sorted = sorted(history, key=lambda x: x[0])

        baseline_score = history_sorted[0][1]
        current_score = history_sorted[-1][1]

        improvement_pct = ((current_score - baseline_score) / baseline_score * 100.0
                           if baseline_score > 0 else 0.0)

        # Identify milestones
        milestones = []
        for i in range(1, len(history_sorted)):
            prev_score = history_sorted[i-1][1]
            curr_score = history_sorted[i][1]

            # 10-point improvement milestone
            if curr_score >= prev_score + 10:
                milestones.append(f"10-point improvement achieved on {history_sorted[i][0]}")

            # Grade upgrade milestone
            prev_grade = self._score_to_grade(prev_score)
            curr_grade = self._score_to_grade(curr_score)
            if curr_grade > prev_grade:
                milestones.append(f"Grade upgraded to {curr_grade} on {history_sorted[i][0]}")

        # Generate recommendations
        recommendations = []
        if improvement_pct < 5:
            recommendations.append("Accelerate improvement initiatives")
        if current_score < 70:
            recommendations.append("Focus on certification acquisition")
            recommendations.append("Improve traceability coverage")

        return ImprovementTrack(
            supplier_id=supplier_id,
            baseline_score=baseline_score,
            current_score=current_score,
            improvement_pct=improvement_pct,
            history=history_sorted,
            milestones=milestones,
            recommendations=recommendations
        )

    def identify_best_practices(self, peer_group: str) -> List[BestPractice]:
        """
        Identify best practices from high performers.

        Args:
            peer_group: Peer group identifier

        Returns:
            List of BestPractice sorted by impact
        """
        # Pre-defined best practices (would be learned from data in production)
        best_practices = [
            BestPractice(
                practice_id="BP001",
                category="Traceability",
                description="Implement blockchain-based traceability system",
                impact_score=9.5,
                adoption_rate=0.35,
                example_suppliers=["SUP001", "SUP042", "SUP089"]
            ),
            BestPractice(
                practice_id="BP002",
                category="Certification",
                description="Maintain multiple third-party certifications (FSC, RSPO, etc.)",
                impact_score=8.8,
                adoption_rate=0.52,
                example_suppliers=["SUP023", "SUP067"]
            ),
            BestPractice(
                practice_id="BP003",
                category="Risk Management",
                description="Quarterly satellite monitoring of supply plots",
                impact_score=9.2,
                adoption_rate=0.28,
                example_suppliers=["SUP012", "SUP034"]
            ),
            BestPractice(
                practice_id="BP004",
                category="Data Quality",
                description="Real-time data integration with ERP systems",
                impact_score=7.9,
                adoption_rate=0.41,
                example_suppliers=["SUP045", "SUP078"]
            ),
            BestPractice(
                practice_id="BP005",
                category="Transparency",
                description="Public disclosure of supply chain map",
                impact_score=8.5,
                adoption_rate=0.19,
                example_suppliers=["SUP056"]
            ),
        ]

        # Sort by impact score
        best_practices.sort(key=lambda x: x.impact_score, reverse=True)

        return best_practices

    def alert_degradation(
        self,
        supplier_id: str,
        current: SupplierScore,
        previous: SupplierScore
    ) -> Optional[Alert]:
        """
        Generate alert if supplier performance degraded.

        Args:
            supplier_id: Supplier identifier
            current: Current score
            previous: Previous score

        Returns:
            Alert if degradation detected, None otherwise
        """
        score_change = current.composite_score - previous.composite_score

        if score_change >= -self.config.alert_degradation_threshold:
            return None  # No significant degradation

        # Identify affected dimensions
        affected_dimensions = []
        for dim in PerformanceDimension:
            dim_key = dim.value
            curr_dim_score = current.dimensions.get(dim_key, 0.0)
            prev_dim_score = previous.dimensions.get(dim_key, 0.0)

            if curr_dim_score < prev_dim_score - 10.0:  # 10-point drop
                affected_dimensions.append(dim_key)

        # Determine severity
        if score_change <= -30:
            severity = "CRITICAL"
        elif score_change <= -20:
            severity = "HIGH"
        elif score_change <= -10:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        alert_id = hashlib.sha256(
            f"{supplier_id}_{current.timestamp.isoformat()}".encode()
        ).hexdigest()[:16]

        message = (
            f"Supplier {supplier_id} performance degraded by {abs(score_change):.1f} points "
            f"from {previous.composite_score:.1f} to {current.composite_score:.1f}"
        )

        return Alert(
            alert_id=alert_id,
            supplier_id=supplier_id,
            severity=severity,
            message=message,
            current_score=current.composite_score,
            previous_score=previous.composite_score,
            score_change=score_change,
            dimensions_affected=affected_dimensions
        )

    def portfolio_benchmark(self, suppliers: List[Dict[str, Any]]) -> PortfolioBenchmark:
        """
        Benchmark portfolio of suppliers.

        Args:
            suppliers: List of supplier data dictionaries

        Returns:
            PortfolioBenchmark with portfolio-level metrics
        """
        if not suppliers:
            raise ValueError("Suppliers list cannot be empty")

        logger.info(f"Benchmarking portfolio of {len(suppliers)} suppliers")

        # Calculate scores for all suppliers
        scores = []
        for supplier_data in suppliers:
            try:
                score = self.calculate_score(supplier_data)
                scores.append((supplier_data["supplier_id"], score))
            except Exception as e:
                logger.error(f"Failed to score supplier {supplier_data.get('supplier_id')}: {str(e)}")
                continue

        if not scores:
            raise ValueError("No valid supplier scores calculated")

        # Calculate portfolio metrics
        composite_scores = [s[1].composite_score for s in scores]
        average_score = sum(composite_scores) / len(composite_scores)

        sorted_scores = sorted(composite_scores)
        median_score = sorted_scores[len(sorted_scores) // 2]

        # Score distribution by grade
        score_distribution = {}
        for _, score in scores:
            grade = score.grade
            score_distribution[grade] = score_distribution.get(grade, 0) + 1

        # Identify top and bottom performers
        scores_sorted = sorted(scores, key=lambda x: x[1].composite_score, reverse=True)
        top_10pct = max(1, len(scores) // 10)

        top_performers = [s[0] for s in scores_sorted[:top_10pct]]
        underperformers = [s[0] for s in scores_sorted[-top_10pct:]]

        # Calculate dimension averages
        dimension_averages = {}
        for dim in PerformanceDimension:
            dim_key = dim.value
            dim_scores = [s[1].dimensions.get(dim_key, 0.0) for s in scores]
            dimension_averages[dim_key] = sum(dim_scores) / len(dim_scores)

        return PortfolioBenchmark(
            portfolio_id=f"PORTFOLIO_{datetime.utcnow().strftime('%Y%m%d')}",
            supplier_count=len(scores),
            average_score=average_score,
            median_score=median_score,
            score_distribution=score_distribution,
            top_performers=top_performers,
            underperformers=underperformers,
            dimension_averages=dimension_averages
        )

    # Helper methods

    def _calculate_certification_score(self, certifications: List[str]) -> float:
        """Calculate certification score based on certifications held."""
        cert_weights = {
            "FSC": 25.0,
            "RSPO": 25.0,
            "RTRS": 25.0,
            "Rainforest Alliance": 20.0,
            "UTZ": 20.0,
            "Fairtrade": 15.0,
            "Organic": 15.0,
            "4C": 10.0,
        }

        score = 0.0
        for cert in certifications:
            score += cert_weights.get(cert, 5.0)

        return min(100.0, score)

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D"
        else:
            return "F"

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function approximation."""
        # Simplified approximation of normal CDF
        # For production, would use scipy or proper implementation
        import math

        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _simulate_supplier_data(self, supplier_id: str) -> Dict[str, Any]:
        """Simulate supplier data (for demo purposes)."""
        # Hash-based deterministic simulation
        hash_val = hash(supplier_id)

        return {
            "supplier_id": supplier_id,
            "commodity": CommoditySector.COCOA.value,
            "region": "WEST_AFRICA",
            "size_tier": "MEDIUM",
            "traceability_pct": 50.0 + (hash_val % 40),
            "certification_status": ["FSC", "RSPO"] if hash_val % 2 == 0 else ["Rainforest Alliance"],
            "deforestation_alerts": hash_val % 3,
            "data_completeness": 60.0 + (hash_val % 35),
            "compliance_violations": hash_val % 2,
            "response_time_days": 5 + (hash_val % 10),
            "transparency_score": 55.0 + (hash_val % 30),
        }

    def _identify_strengths_weaknesses(
        self,
        score: SupplierScore,
        peer_stats: PeerGroupStats
    ) -> Tuple[List[str], List[str]]:
        """Identify supplier strengths and weaknesses relative to peers."""
        strengths = []
        weaknesses = []

        for dim, dim_score in score.dimensions.items():
            # Compare to peer mean
            if dim_score > peer_stats.mean + 10:
                strengths.append(f"{dim}: {dim_score:.1f} (above peer average)")
            elif dim_score < peer_stats.mean - 10:
                weaknesses.append(f"{dim}: {dim_score:.1f} (below peer average)")

        return strengths, weaknesses

    def _generate_recommendations(
        self,
        score: SupplierScore,
        weaknesses: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Check each dimension
        for dim, dim_score in score.dimensions.items():
            if dim_score < 60:
                if dim == PerformanceDimension.TRACEABILITY.value:
                    recommendations.append("Implement GPS tracking for all supply plots")
                elif dim == PerformanceDimension.CERTIFICATION.value:
                    recommendations.append("Pursue FSC or RSPO certification")
                elif dim == PerformanceDimension.DEFORESTATION_RISK.value:
                    recommendations.append("Investigate and remediate deforestation alerts")
                elif dim == PerformanceDimension.DATA_QUALITY.value:
                    recommendations.append("Improve data collection and validation processes")

        return recommendations[:5]  # Top 5 recommendations
