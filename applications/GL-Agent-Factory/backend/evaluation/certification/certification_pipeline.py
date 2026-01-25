"""
Agent Certification Pipeline for GreenLang AI Agent Factory

A comprehensive 12-dimension certification system for validating GreenLang agents.
Executes parallel evaluations, aggregates results, and generates certification reports.

Features:
- 12-dimension evaluation framework
- Parallel test execution for performance
- Weighted scoring system
- Configurable thresholds
- Pass/fail determination
- Integration with existing evaluation tools

Example:
    >>> from certification import CertificationPipeline, CertificationConfig
    >>> config = CertificationConfig(strict_mode=True)
    >>> pipeline = CertificationPipeline(config)
    >>> report = pipeline.certify_agent(agent, "path/to/pack.yaml")
    >>> if report.is_certified:
    ...     print(f"Agent certified at {report.certification_level} level")

"""

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from ..golden_test_runner import GoldenTestRunner, GoldenTestResult
from ..determinism_verifier import DeterminismVerifier, DeterminismResult

from .dimensions import (
    TechnicalAccuracyEvaluator,
    DataCredibilityEvaluator,
    SafetyComplianceEvaluator,
    RegulatoryAlignmentEvaluator,
    UncertaintyQuantificationEvaluator,
    ExplainabilityEvaluator,
    PerformanceEvaluator,
    RobustnessEvaluator,
    SecurityEvaluator,
    AuditabilityEvaluator,
    MaintainabilityEvaluator,
    OperabilityEvaluator,
    DIMENSION_WEIGHTS,
    DIMENSION_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class CertificationLevel(Enum):
    """Certification levels based on overall score."""
    PLATINUM = "PLATINUM"  # 98-100
    GOLD = "GOLD"          # 95-97
    SILVER = "SILVER"      # 90-94
    BRONZE = "BRONZE"      # 85-89
    PROVISIONAL = "PROVISIONAL"  # 75-84
    FAIL = "FAIL"          # Below 75


@dataclass
class CertificationConfig:
    """Configuration for the certification pipeline."""

    # Execution settings
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300

    # Scoring settings
    dimension_weights: Dict[str, float] = field(default_factory=lambda: DIMENSION_WEIGHTS.copy())
    dimension_thresholds: Dict[str, float] = field(default_factory=lambda: DIMENSION_THRESHOLDS.copy())

    # Certification thresholds
    platinum_threshold: float = 98.0
    gold_threshold: float = 95.0
    silver_threshold: float = 90.0
    bronze_threshold: float = 85.0
    provisional_threshold: float = 75.0

    # Strict mode requires all dimensions to pass their thresholds
    strict_mode: bool = True

    # Integration settings
    determinism_runs: int = 100
    golden_test_tolerance: float = 1e-9

    # Output settings
    generate_detailed_report: bool = True
    save_report_to_file: bool = True
    report_output_dir: Optional[str] = None

    def validate(self) -> bool:
        """Validate configuration."""
        # Check weights sum to 1.0
        weight_sum = sum(self.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Dimension weights must sum to 1.0, got {weight_sum}")
        return True


@dataclass
class DimensionResult:
    """Result from evaluating a single dimension."""
    dimension_name: str
    score: float  # 0-100
    weight: float  # Dimension weight for overall score
    weighted_score: float  # score * weight
    passed_threshold: bool
    threshold: float
    execution_time_ms: float
    test_count: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.test_count == 0:
            return 0.0
        return (self.tests_passed / self.test_count) * 100


@dataclass
class CertificationReport:
    """Complete certification report for an agent."""

    # Agent identification
    agent_id: str
    agent_version: str
    pack_yaml_path: str

    # Certification results
    is_certified: bool
    certification_level: CertificationLevel
    overall_score: float

    # Dimension results
    dimension_results: Dict[str, DimensionResult]
    dimensions_passed: int
    dimensions_failed: int

    # Existing evaluation integration
    golden_test_result: Optional[GoldenTestResult] = None
    determinism_result: Optional[DeterminismResult] = None

    # Metadata
    certification_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0
    config: Optional[CertificationConfig] = None

    # Findings and recommendations
    critical_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Validity
    valid_until: Optional[datetime] = None

    def __post_init__(self):
        """Generate certification ID and validity period."""
        if not self.certification_id:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            agent_safe = self.agent_id.replace("/", "_").replace("-", "_")
            hash_input = f"{self.agent_id}:{self.agent_version}:{timestamp_str}"
            cert_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
            self.certification_id = f"CERT_{agent_safe}_{cert_hash}_{timestamp_str}"

        if not self.valid_until:
            # Certifications valid for 90 days
            self.valid_until = self.timestamp + timedelta(days=90)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "certification_id": self.certification_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "is_certified": self.is_certified,
            "certification_level": self.certification_level.value,
            "overall_score": self.overall_score,
            "dimensions_passed": self.dimensions_passed,
            "dimensions_failed": self.dimensions_failed,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "execution_time_ms": self.execution_time_ms,
            "critical_findings": self.critical_findings,
            "recommendations": self.recommendations,
            "dimension_scores": {
                name: {
                    "score": result.score,
                    "weighted_score": result.weighted_score,
                    "passed": result.passed_threshold,
                    "threshold": result.threshold,
                }
                for name, result in self.dimension_results.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class CertificationPipeline:
    """
    Main certification pipeline for GreenLang agents.

    Orchestrates 12-dimension evaluation, integrates with existing
    evaluation tools, and generates comprehensive certification reports.
    """

    def __init__(self, config: Optional[CertificationConfig] = None):
        """
        Initialize certification pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or CertificationConfig()
        self.config.validate()

        # Initialize existing evaluation tools
        self.golden_test_runner = GoldenTestRunner(
            default_tolerance=self.config.golden_test_tolerance
        )
        self.determinism_verifier = DeterminismVerifier(tolerance=0.0)

        # Initialize dimension evaluators
        self._init_evaluators()

        logger.info("CertificationPipeline initialized")

    def _init_evaluators(self) -> None:
        """Initialize all dimension evaluators."""
        self.evaluators = {
            "technical_accuracy": TechnicalAccuracyEvaluator(),
            "data_credibility": DataCredibilityEvaluator(),
            "safety_compliance": SafetyComplianceEvaluator(),
            "regulatory_alignment": RegulatoryAlignmentEvaluator(),
            "uncertainty_quantification": UncertaintyQuantificationEvaluator(),
            "explainability": ExplainabilityEvaluator(),
            "performance": PerformanceEvaluator(),
            "robustness": RobustnessEvaluator(),
            "security": SecurityEvaluator(),
            "auditability": AuditabilityEvaluator(),
            "maintainability": MaintainabilityEvaluator(),
            "operability": OperabilityEvaluator(),
        }

    def certify_agent(
        self,
        agent: Any,
        pack_yaml_path: Union[str, Path],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True,
    ) -> CertificationReport:
        """
        Run full certification for an agent.

        Args:
            agent: Agent instance to certify
            pack_yaml_path: Path to agent's pack.yaml
            sample_inputs: Optional sample inputs for testing
            verbose: Print progress output

        Returns:
            CertificationReport with complete results
        """
        start_time = datetime.utcnow()
        pack_yaml_path = Path(pack_yaml_path)

        if verbose:
            self._print_header("AGENT CERTIFICATION PIPELINE")

        # Load pack specification
        pack_spec = self._load_pack_spec(pack_yaml_path)
        agent_id = pack_spec.get("pack", {}).get("id", "unknown")
        agent_version = pack_spec.get("pack", {}).get("version", "1.0.0")

        if verbose:
            logger.info(f"Certifying: {agent_id} v{agent_version}")

        # Get sample inputs
        if sample_inputs is None:
            sample_inputs = self._extract_sample_inputs(pack_spec)

        # Run existing evaluation tools integration
        if verbose:
            logger.info("\n[Phase 1] Running existing evaluation tools...")

        golden_result = self._run_golden_tests(pack_yaml_path, agent)
        determinism_result = self._run_determinism_verification(
            agent, sample_inputs[0] if sample_inputs else {}
        )

        # Run 12-dimension evaluation
        if verbose:
            logger.info("\n[Phase 2] Running 12-dimension evaluation...")

        dimension_results = self._evaluate_all_dimensions(
            agent, pack_spec, sample_inputs, golden_result, determinism_result, verbose
        )

        # Calculate overall score and certification level
        overall_score = self._calculate_overall_score(dimension_results)
        dimensions_passed = sum(
            1 for r in dimension_results.values() if r.passed_threshold
        )
        dimensions_failed = len(dimension_results) - dimensions_passed

        # Determine certification
        is_certified, cert_level = self._determine_certification(
            overall_score, dimension_results
        )

        # Collect findings and recommendations
        critical_findings, recommendations = self._collect_findings(dimension_results)

        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Build report
        report = CertificationReport(
            agent_id=agent_id,
            agent_version=agent_version,
            pack_yaml_path=str(pack_yaml_path),
            is_certified=is_certified,
            certification_level=cert_level,
            overall_score=overall_score,
            dimension_results=dimension_results,
            dimensions_passed=dimensions_passed,
            dimensions_failed=dimensions_failed,
            golden_test_result=golden_result,
            determinism_result=determinism_result,
            execution_time_ms=execution_time_ms,
            config=self.config,
            critical_findings=critical_findings,
            recommendations=recommendations,
        )

        if verbose:
            self._print_report(report)

        # Save report if configured
        if self.config.save_report_to_file:
            self._save_report(report, pack_yaml_path.parent)

        return report

    def _load_pack_spec(self, pack_yaml_path: Path) -> Dict[str, Any]:
        """Load pack.yaml specification."""
        if not pack_yaml_path.exists():
            raise FileNotFoundError(f"Pack file not found: {pack_yaml_path}")

        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _extract_sample_inputs(
        self, pack_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract sample inputs from pack specification."""
        inputs = []

        # Try golden tests
        golden_tests = pack_spec.get("tests", {}).get("golden", [])
        if golden_tests:
            for test in golden_tests[:5]:  # Take up to 5 samples
                if "input" in test:
                    inputs.append(test["input"])

        # Try old format
        if not inputs:
            old_golden = pack_spec.get("golden_tests", {}).get("test_cases", [])
            for test in old_golden[:5]:
                if "input" in test:
                    inputs.append(test["input"])

        # Default empty input
        if not inputs:
            inputs = [{}]

        return inputs

    def _run_golden_tests(
        self, pack_yaml_path: Path, agent: Any
    ) -> Optional[GoldenTestResult]:
        """Run golden test suite integration."""
        try:
            return self.golden_test_runner.run_tests(
                pack_yaml_path, agent, verbose=False
            )
        except Exception as e:
            logger.warning(f"Golden tests failed: {e}")
            return None

    def _run_determinism_verification(
        self, agent: Any, sample_input: Dict[str, Any]
    ) -> Optional[DeterminismResult]:
        """Run determinism verification integration."""
        try:
            return self.determinism_verifier.verify(
                agent,
                sample_input,
                num_runs=self.config.determinism_runs,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Determinism verification failed: {e}")
            return None

    def _evaluate_all_dimensions(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[GoldenTestResult],
        determinism_result: Optional[DeterminismResult],
        verbose: bool,
    ) -> Dict[str, DimensionResult]:
        """
        Evaluate all 12 dimensions.

        Uses parallel execution if configured.
        """
        context = {
            "agent": agent,
            "pack_spec": pack_spec,
            "sample_inputs": sample_inputs,
            "golden_result": golden_result,
            "determinism_result": determinism_result,
        }

        if self.config.parallel_execution:
            return self._evaluate_dimensions_parallel(context, verbose)
        else:
            return self._evaluate_dimensions_sequential(context, verbose)

    def _evaluate_dimensions_parallel(
        self, context: Dict[str, Any], verbose: bool
    ) -> Dict[str, DimensionResult]:
        """Evaluate dimensions in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_dimension, name, evaluator, context
                ): name
                for name, evaluator in self.evaluators.items()
            }

            for future in as_completed(futures):
                dim_name = futures[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results[dim_name] = result

                    if verbose:
                        status = "PASS" if result.passed_threshold else "FAIL"
                        logger.info(
                            f"  [{status}] {dim_name}: {result.score:.1f}/100"
                        )

                except Exception as e:
                    logger.error(f"Dimension {dim_name} failed: {e}")
                    results[dim_name] = self._create_failed_dimension_result(dim_name)

        return results

    def _evaluate_dimensions_sequential(
        self, context: Dict[str, Any], verbose: bool
    ) -> Dict[str, DimensionResult]:
        """Evaluate dimensions sequentially."""
        results = {}

        for name, evaluator in self.evaluators.items():
            try:
                result = self._evaluate_single_dimension(name, evaluator, context)
                results[name] = result

                if verbose:
                    status = "PASS" if result.passed_threshold else "FAIL"
                    logger.info(f"  [{status}] {name}: {result.score:.1f}/100")

            except Exception as e:
                logger.error(f"Dimension {name} failed: {e}")
                results[name] = self._create_failed_dimension_result(name)

        return results

    def _evaluate_single_dimension(
        self,
        dimension_name: str,
        evaluator: Any,
        context: Dict[str, Any],
    ) -> DimensionResult:
        """Evaluate a single dimension."""
        start_time = datetime.utcnow()

        # Run evaluator
        result = evaluator.evaluate(
            agent=context["agent"],
            pack_spec=context["pack_spec"],
            sample_inputs=context["sample_inputs"],
            golden_result=context.get("golden_result"),
            determinism_result=context.get("determinism_result"),
        )

        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Get weight and threshold
        weight = self.config.dimension_weights.get(dimension_name, 0.0)
        threshold = self.config.dimension_thresholds.get(dimension_name, 85.0)

        return DimensionResult(
            dimension_name=dimension_name,
            score=result.score,
            weight=weight,
            weighted_score=result.score * weight,
            passed_threshold=result.score >= threshold,
            threshold=threshold,
            execution_time_ms=execution_time_ms,
            test_count=result.test_count,
            tests_passed=result.tests_passed,
            tests_failed=result.tests_failed,
            details=result.details,
            findings=result.findings,
            recommendations=result.recommendations,
        )

    def _create_failed_dimension_result(self, dimension_name: str) -> DimensionResult:
        """Create a failed dimension result for error cases."""
        weight = self.config.dimension_weights.get(dimension_name, 0.0)
        threshold = self.config.dimension_thresholds.get(dimension_name, 85.0)

        return DimensionResult(
            dimension_name=dimension_name,
            score=0.0,
            weight=weight,
            weighted_score=0.0,
            passed_threshold=False,
            threshold=threshold,
            execution_time_ms=0.0,
            test_count=0,
            tests_passed=0,
            tests_failed=0,
            details={"error": "Evaluation failed"},
            findings=["Dimension evaluation failed due to error"],
            recommendations=["Review agent implementation and retry certification"],
        )

    def _calculate_overall_score(
        self, dimension_results: Dict[str, DimensionResult]
    ) -> float:
        """Calculate weighted overall score."""
        total_weighted = sum(r.weighted_score for r in dimension_results.values())
        return total_weighted

    def _determine_certification(
        self,
        overall_score: float,
        dimension_results: Dict[str, DimensionResult],
    ) -> Tuple[bool, CertificationLevel]:
        """Determine certification status and level."""

        # Check if all dimensions pass thresholds (strict mode)
        all_passed = all(r.passed_threshold for r in dimension_results.values())

        if self.config.strict_mode and not all_passed:
            return False, CertificationLevel.FAIL

        # Determine level based on score
        if overall_score >= self.config.platinum_threshold and all_passed:
            return True, CertificationLevel.PLATINUM
        elif overall_score >= self.config.gold_threshold and all_passed:
            return True, CertificationLevel.GOLD
        elif overall_score >= self.config.silver_threshold:
            return True, CertificationLevel.SILVER
        elif overall_score >= self.config.bronze_threshold:
            return True, CertificationLevel.BRONZE
        elif overall_score >= self.config.provisional_threshold:
            return True, CertificationLevel.PROVISIONAL
        else:
            return False, CertificationLevel.FAIL

    def _collect_findings(
        self, dimension_results: Dict[str, DimensionResult]
    ) -> Tuple[List[str], List[str]]:
        """Collect critical findings and recommendations."""
        critical = []
        recommendations = []

        for result in dimension_results.values():
            if not result.passed_threshold:
                critical.extend(result.findings)
            recommendations.extend(result.recommendations)

        return critical, recommendations

    def _save_report(self, report: CertificationReport, output_dir: Path) -> None:
        """Save certification report to file."""
        report_dir = Path(self.config.report_output_dir or output_dir / "certification_reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"{report.certification_id}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report.to_json())

        logger.info(f"Report saved to: {report_file}")

    def _print_header(self, title: str) -> None:
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)

    def _print_report(self, report: CertificationReport) -> None:
        """Print certification report summary."""
        print("\n" + "=" * 80)
        print(" CERTIFICATION REPORT")
        print("=" * 80)
        print(f" Agent ID:        {report.agent_id}")
        print(f" Version:         {report.agent_version}")
        print(f" Certification:   {report.certification_id}")
        print(f" Timestamp:       {report.timestamp.isoformat()}")
        print("-" * 80)
        print(f" Overall Score:   {report.overall_score:.2f}/100")
        print(f" Level:           {report.certification_level.value}")
        print(f" Status:          {'CERTIFIED' if report.is_certified else 'NOT CERTIFIED'}")
        print(f" Valid Until:     {report.valid_until.isoformat() if report.valid_until else 'N/A'}")
        print("=" * 80)

        print("\n DIMENSION SCORES:")
        print("-" * 80)
        for name, result in sorted(
            report.dimension_results.items(),
            key=lambda x: x[1].score,
            reverse=True,
        ):
            status = "PASS" if result.passed_threshold else "FAIL"
            bar = self._score_bar(result.score)
            print(f" [{status}] {name:<30} {result.score:>6.1f}/100 {bar}")

        if report.critical_findings:
            print("\n CRITICAL FINDINGS:")
            print("-" * 80)
            for finding in report.critical_findings[:10]:
                print(f"  - {finding}")

        if report.recommendations:
            print("\n TOP RECOMMENDATIONS:")
            print("-" * 80)
            for rec in report.recommendations[:5]:
                print(f"  - {rec}")

        print("\n" + "=" * 80)

        if report.is_certified:
            print(f" CERTIFICATION GRANTED: {report.certification_level.value}")
        else:
            print(" CERTIFICATION DENIED - Address findings and resubmit")

        print("=" * 80 + "\n")

    def _score_bar(self, score: float, width: int = 20) -> str:
        """Generate ASCII score bar."""
        filled = int((score / 100) * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"


# Convenience function for quick certification
def certify_agent(
    agent: Any,
    pack_yaml_path: Union[str, Path],
    config: Optional[CertificationConfig] = None,
    verbose: bool = True,
) -> CertificationReport:
    """
    Convenience function to certify an agent.

    Args:
        agent: Agent instance
        pack_yaml_path: Path to pack.yaml
        config: Optional configuration
        verbose: Print output

    Returns:
        CertificationReport
    """
    pipeline = CertificationPipeline(config)
    return pipeline.certify_agent(agent, pack_yaml_path, verbose=verbose)
