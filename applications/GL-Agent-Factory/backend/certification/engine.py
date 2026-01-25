"""
Certification Engine for GreenLang Agents

This module provides the main certification engine that evaluates agents
across all 12 certification dimensions.

Example:
    >>> engine = CertificationEngine()
    >>> result = engine.evaluate_agent(Path("path/to/agent"))
    >>> if result.certified:
    ...     print(f"Agent certified at {result.level} level")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .dimensions.base import BaseDimension, DimensionResult, DimensionStatus
from .dimensions import (
    DeterminismDimension,
    ProvenanceDimension,
    ZeroHallucinationDimension,
    AccuracyDimension,
    SourceVerificationDimension,
    UnitConsistencyDimension,
    RegulatoryComplianceDimension,
    SecurityDimension,
    PerformanceDimension,
    DocumentationDimension,
    CoverageDimension,
    ProductionReadinessDimension,
    ALL_DIMENSIONS,
)

logger = logging.getLogger(__name__)


class CertificationLevel(str, Enum):
    """Certification level based on score."""

    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"
    FAIL = "FAIL"


@dataclass
class CertificationResult:
    """
    Complete certification result for an agent.

    Attributes:
        agent_path: Path to the certified agent
        agent_id: Agent identifier
        agent_version: Agent version
        certified: Whether the agent passed certification
        level: Certification level (GOLD, SILVER, BRONZE, FAIL)
        overall_score: Overall score (0-100)
        dimension_results: Results for each dimension
        timestamp: When certification was performed
        certification_id: Unique certification identifier
    """

    agent_path: Path
    agent_id: str
    agent_version: str
    certified: bool
    level: CertificationLevel
    overall_score: float
    weighted_score: float
    dimensions_passed: int
    dimensions_failed: int
    dimensions_total: int
    dimension_results: List[DimensionResult] = field(default_factory=list)
    required_failures: List[str] = field(default_factory=list)
    remediation_summary: Dict[str, List[str]] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    certification_id: str = ""

    def __post_init__(self):
        """Generate certification ID."""
        if not self.certification_id:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            agent_safe = self.agent_id.replace("/", "_").replace("-", "_")
            self.certification_id = f"CERT_{agent_safe}_{timestamp_str}"

    @property
    def passed_dimensions(self) -> List[DimensionResult]:
        """Get list of passed dimensions."""
        return [d for d in self.dimension_results if d.status == DimensionStatus.PASS]

    @property
    def failed_dimensions(self) -> List[DimensionResult]:
        """Get list of failed dimensions."""
        return [d for d in self.dimension_results if d.status == DimensionStatus.FAIL]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_path": str(self.agent_path),
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "certified": self.certified,
            "level": self.level.value,
            "overall_score": self.overall_score,
            "weighted_score": self.weighted_score,
            "dimensions_passed": self.dimensions_passed,
            "dimensions_failed": self.dimensions_failed,
            "dimensions_total": self.dimensions_total,
            "dimension_results": [d.to_dict() for d in self.dimension_results],
            "required_failures": self.required_failures,
            "remediation_summary": self.remediation_summary,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "certification_id": self.certification_id,
        }


class CertificationEngine:
    """
    Main certification engine for evaluating agents.

    This engine runs all 12 certification dimensions and produces
    a comprehensive certification report.

    Certification Levels:
        - GOLD: 100% score, all required dimensions pass
        - SILVER: 95%+ score, all required dimensions pass
        - BRONZE: 85%+ score, all required dimensions pass
        - FAIL: Below 85% or required dimension fails

    Configuration:
        dimensions: List of dimension classes to evaluate
        gold_threshold: Score threshold for GOLD (default: 100.0)
        silver_threshold: Score threshold for SILVER (default: 95.0)
        bronze_threshold: Score threshold for BRONZE (default: 85.0)

    Example:
        >>> engine = CertificationEngine()
        >>> result = engine.evaluate_agent(Path("agents/my_agent"))
        >>> print(f"Certification: {result.level.value}")
    """

    # Default certification thresholds
    GOLD_THRESHOLD = 100.0
    SILVER_THRESHOLD = 95.0
    BRONZE_THRESHOLD = 85.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the certification engine.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Load thresholds from config
        self.gold_threshold = self.config.get("gold_threshold", self.GOLD_THRESHOLD)
        self.silver_threshold = self.config.get("silver_threshold", self.SILVER_THRESHOLD)
        self.bronze_threshold = self.config.get("bronze_threshold", self.BRONZE_THRESHOLD)

        # Initialize dimensions
        self.dimensions: List[BaseDimension] = []
        dimension_classes = self.config.get("dimensions", ALL_DIMENSIONS)

        for dim_class in dimension_classes:
            dim_config = self.config.get(f"dimension_{dim_class.DIMENSION_ID}", {})
            self.dimensions.append(dim_class(dim_config))

        logger.info(
            f"CertificationEngine initialized with {len(self.dimensions)} dimensions"
        )

    def evaluate_agent(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
        verbose: bool = True,
    ) -> CertificationResult:
        """
        Run all certification dimensions and return results.

        Args:
            agent_path: Path to agent directory
            agent: Optional pre-loaded agent instance
            sample_input: Optional sample input for testing
            verbose: Print progress information

        Returns:
            CertificationResult with comprehensive evaluation
        """
        start_time = datetime.utcnow()

        if verbose:
            self._print_header(agent_path)

        # Load agent metadata
        agent_id, agent_version = self._load_agent_metadata(agent_path)

        # Run all dimensions
        dimension_results: List[DimensionResult] = []
        required_failures: List[str] = []

        for i, dimension in enumerate(self.dimensions, 1):
            if verbose:
                print(f"\n[{i}/{len(self.dimensions)}] Evaluating {dimension.DIMENSION_NAME}...")

            try:
                result = dimension.evaluate(agent_path, agent, sample_input)
                dimension_results.append(result)

                # Track required dimension failures
                if dimension.REQUIRED_FOR_CERTIFICATION and result.status == DimensionStatus.FAIL:
                    required_failures.append(dimension.DIMENSION_NAME)

                if verbose:
                    status_icon = "PASS" if result.passed else "FAIL"
                    print(f"    [{status_icon}] {dimension.DIMENSION_NAME}: {result.score:.1f}/100")
                    if not result.passed and result.remediation:
                        print(f"    Remediation: {result.remediation[0][:80]}...")

            except Exception as e:
                logger.error(f"Dimension {dimension.DIMENSION_NAME} failed: {str(e)}")
                # Create error result
                error_result = DimensionResult(
                    dimension_id=dimension.DIMENSION_ID,
                    dimension_name=dimension.DIMENSION_NAME,
                    status=DimensionStatus.ERROR,
                    score=0.0,
                    checks_passed=0,
                    checks_failed=1,
                    checks_total=1,
                    execution_time_ms=0,
                    details={"error": str(e)},
                )
                dimension_results.append(error_result)

                if dimension.REQUIRED_FOR_CERTIFICATION:
                    required_failures.append(dimension.DIMENSION_NAME)

        # Calculate scores
        overall_score = self._calculate_overall_score(dimension_results)
        weighted_score = self._calculate_weighted_score(dimension_results)

        # Count passed/failed
        dimensions_passed = sum(
            1 for d in dimension_results if d.status == DimensionStatus.PASS
        )
        dimensions_failed = len(dimension_results) - dimensions_passed

        # Determine certification level
        certified, level = self._determine_certification(
            weighted_score,
            required_failures,
        )

        # Compile remediation summary
        remediation_summary = self._compile_remediation(dimension_results)

        # Calculate total execution time
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = CertificationResult(
            agent_path=agent_path,
            agent_id=agent_id,
            agent_version=agent_version,
            certified=certified,
            level=level,
            overall_score=overall_score,
            weighted_score=weighted_score,
            dimensions_passed=dimensions_passed,
            dimensions_failed=dimensions_failed,
            dimensions_total=len(dimension_results),
            dimension_results=dimension_results,
            required_failures=required_failures,
            remediation_summary=remediation_summary,
            execution_time_ms=execution_time_ms,
        )

        if verbose:
            self._print_summary(result)

        return result

    def evaluate_single_dimension(
        self,
        agent_path: Path,
        dimension_id: str,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate a single dimension.

        Args:
            agent_path: Path to agent directory
            dimension_id: Dimension ID (e.g., "D01")
            agent: Optional pre-loaded agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult for the specified dimension

        Raises:
            ValueError: If dimension ID is not found
        """
        for dimension in self.dimensions:
            if dimension.DIMENSION_ID == dimension_id:
                return dimension.evaluate(agent_path, agent, sample_input)

        raise ValueError(f"Dimension {dimension_id} not found")

    def get_dimension_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all certification dimensions.

        Returns:
            List of dimension information dictionaries
        """
        return [
            {
                "id": d.DIMENSION_ID,
                "name": d.DIMENSION_NAME,
                "description": d.DESCRIPTION,
                "weight": d.WEIGHT,
                "required": d.REQUIRED_FOR_CERTIFICATION,
            }
            for d in self.dimensions
        ]

    def _load_agent_metadata(self, agent_path: Path) -> tuple:
        """
        Load agent ID and version from metadata.

        Args:
            agent_path: Path to agent directory

        Returns:
            Tuple of (agent_id, agent_version)
        """
        agent_id = agent_path.name
        agent_version = "1.0.0"

        try:
            import yaml

            # Try pack.yaml
            pack_file = agent_path / "pack.yaml"
            if pack_file.exists():
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                pack_info = pack_spec.get("pack", {})
                agent_id = pack_info.get("id", agent_id)
                agent_version = pack_info.get("version", agent_version)

            # Try agent.yaml
            agent_yaml = agent_path / "agent.yaml"
            if agent_yaml.exists():
                with open(agent_yaml, "r", encoding="utf-8") as f:
                    agent_spec = yaml.safe_load(f)

                agent_id = agent_spec.get("id", agent_id)
                agent_version = agent_spec.get("version", agent_version)

        except Exception as e:
            logger.warning(f"Failed to load agent metadata: {str(e)}")

        return agent_id, agent_version

    def _calculate_overall_score(
        self,
        dimension_results: List[DimensionResult],
    ) -> float:
        """
        Calculate overall score (simple average).

        Args:
            dimension_results: List of dimension results

        Returns:
            Overall score (0-100)
        """
        if not dimension_results:
            return 0.0

        total_score = sum(d.score for d in dimension_results)
        return total_score / len(dimension_results)

    def _calculate_weighted_score(
        self,
        dimension_results: List[DimensionResult],
    ) -> float:
        """
        Calculate weighted score based on dimension weights.

        Args:
            dimension_results: List of dimension results

        Returns:
            Weighted score (0-100)
        """
        if not dimension_results:
            return 0.0

        # Get weights from dimensions
        weights = {}
        for dimension in self.dimensions:
            weights[dimension.DIMENSION_ID] = dimension.WEIGHT

        total_weight = 0.0
        weighted_sum = 0.0

        for result in dimension_results:
            weight = weights.get(result.dimension_id, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _determine_certification(
        self,
        weighted_score: float,
        required_failures: List[str],
    ) -> tuple:
        """
        Determine certification status and level.

        Args:
            weighted_score: Weighted score
            required_failures: List of failed required dimensions

        Returns:
            Tuple of (certified, level)
        """
        # Fail if any required dimension failed
        if required_failures:
            return False, CertificationLevel.FAIL

        # Determine level based on score
        if weighted_score >= self.gold_threshold:
            return True, CertificationLevel.GOLD
        elif weighted_score >= self.silver_threshold:
            return True, CertificationLevel.SILVER
        elif weighted_score >= self.bronze_threshold:
            return True, CertificationLevel.BRONZE
        else:
            return False, CertificationLevel.FAIL

    def _compile_remediation(
        self,
        dimension_results: List[DimensionResult],
    ) -> Dict[str, List[str]]:
        """
        Compile remediation suggestions from all dimensions.

        Args:
            dimension_results: List of dimension results

        Returns:
            Dictionary mapping dimension names to remediation lists
        """
        remediation: Dict[str, List[str]] = {}

        for result in dimension_results:
            if result.remediation:
                remediation[result.dimension_name] = result.remediation

        return remediation

    def _print_header(self, agent_path: Path) -> None:
        """Print certification header."""
        print("\n" + "=" * 80)
        print("GREENLANG AGENT CERTIFICATION")
        print("12-Dimension Evaluation Framework")
        print("=" * 80)
        print(f"Agent Path: {agent_path}")
        print(f"Started: {datetime.utcnow().isoformat()}")
        print("-" * 80)

    def _print_summary(self, result: CertificationResult) -> None:
        """Print certification summary."""
        print("\n" + "=" * 80)
        print("CERTIFICATION RESULTS")
        print("=" * 80)
        print(f"Agent ID:          {result.agent_id}")
        print(f"Version:           {result.agent_version}")
        print(f"Certification ID:  {result.certification_id}")
        print("-" * 80)
        print(f"Overall Score:     {result.overall_score:.2f}/100")
        print(f"Weighted Score:    {result.weighted_score:.2f}/100")
        print(f"Dimensions:        {result.dimensions_passed}/{result.dimensions_total} passed")
        print(f"Execution Time:    {result.execution_time_ms:.2f}ms")
        print("-" * 80)
        print(f"Certification:     {result.level.value}")
        print(f"Status:            {'CERTIFIED' if result.certified else 'NOT CERTIFIED'}")
        print("=" * 80)

        if result.required_failures:
            print("\nREQUIRED DIMENSIONS FAILED:")
            for failure in result.required_failures:
                print(f"  - {failure}")

        if result.remediation_summary:
            print("\nREMEDIATION REQUIRED:")
            for dimension, suggestions in result.remediation_summary.items():
                print(f"\n  {dimension}:")
                for suggestion in suggestions[:2]:
                    print(f"    - {suggestion[:100]}...")

        print()


def certify_agent(
    agent_path: Path,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> CertificationResult:
    """
    Convenience function to certify an agent.

    Args:
        agent_path: Path to agent directory
        config: Optional configuration
        verbose: Print progress

    Returns:
        CertificationResult
    """
    engine = CertificationEngine(config)
    return engine.evaluate_agent(agent_path, verbose=verbose)
