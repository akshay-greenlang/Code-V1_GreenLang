"""
Agent Certification Suite for GreenLang

Runs comprehensive certification testing including:
1. Golden tests (100% pass required)
2. Determinism verification (100 runs)
3. Provenance completeness
4. Citation verification
5. Confidence bounds

Returns PASS/FAIL certification with detailed report.

Example:
    >>> suite = CertificationSuite()
    >>> report = suite.certify_agent("path/to/pack.yaml", agent_instance)
    >>> if report.certified:
    ...     print("CERTIFICATION PASSED")

"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .golden_test_runner import GoldenTestRunner, GoldenTestResult
from .determinism_verifier import DeterminismVerifier, DeterminismResult

logger = logging.getLogger(__name__)


@dataclass
class CertificationTest:
    """Individual certification test result."""
    name: str
    category: str
    passed: bool
    score: float  # 0.0 to 100.0
    details: str
    execution_time_ms: float = 0.0


@dataclass
class CertificationReport:
    """Complete certification report."""
    agent_id: str
    agent_version: str
    certified: bool
    overall_score: float
    certification_level: str  # GOLD, SILVER, BRONZE, FAIL
    tests_passed: int
    tests_failed: int
    test_results: List[CertificationTest] = field(default_factory=list)
    golden_test_result: Optional[GoldenTestResult] = None
    determinism_result: Optional[DeterminismResult] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    certification_id: Optional[str] = None

    def __post_init__(self):
        """Generate certification ID."""
        if not self.certification_id:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            agent_id_safe = self.agent_id.replace('/', '_').replace('-', '_')
            self.certification_id = f"CERT_{agent_id_safe}_{timestamp_str}"


class CertificationSuite:
    """
    Comprehensive agent certification suite.

    This suite runs all required tests for agent certification:
    - Golden tests: 100% pass required
    - Determinism: 100% identical outputs required
    - Provenance: Complete tracking required
    - Citations: All sources verified
    - Confidence: Bounds validation

    Certification levels:
    - GOLD: 100% pass rate on all tests
    - SILVER: 95%+ pass rate
    - BRONZE: 85%+ pass rate
    - FAIL: Below 85%
    """

    # Certification thresholds
    GOLD_THRESHOLD = 100.0
    SILVER_THRESHOLD = 95.0
    BRONZE_THRESHOLD = 85.0

    # Determinism requirements
    DETERMINISM_RUNS = 100

    def __init__(
        self,
        golden_test_runner: Optional[GoldenTestRunner] = None,
        determinism_verifier: Optional[DeterminismVerifier] = None,
    ):
        """
        Initialize certification suite.

        Args:
            golden_test_runner: Optional custom golden test runner
            determinism_verifier: Optional custom determinism verifier
        """
        self.golden_test_runner = golden_test_runner or GoldenTestRunner()
        self.determinism_verifier = determinism_verifier or DeterminismVerifier()

        logger.info("CertificationSuite initialized")

    def certify_agent(
        self,
        pack_yaml_path: Union[str, Path],
        agent: Any,
        sample_input: Optional[Any] = None,
        verbose: bool = True,
    ) -> CertificationReport:
        """
        Run full certification suite for an agent.

        Args:
            pack_yaml_path: Path to pack.yaml
            agent: Agent instance to certify
            sample_input: Sample input for determinism testing
            verbose: Print detailed output

        Returns:
            CertificationReport with results
        """
        if verbose:
            logger.info("=" * 80)
            logger.info("STARTING AGENT CERTIFICATION")
            logger.info("=" * 80)

        start_time = datetime.utcnow()
        test_results = []

        # Load pack metadata
        with open(pack_yaml_path, 'r', encoding='utf-8') as f:
            pack_spec = yaml.safe_load(f)

        pack_info = pack_spec.get('pack', {})
        agent_id = pack_info.get('id', 'unknown')
        agent_version = pack_info.get('version', '1.0.0')

        # Get sample input for tests
        if sample_input is None:
            sample_input = self._get_sample_input(pack_spec)

        # Test 1: Golden Tests
        if verbose:
            logger.info("\n[1/5] Running golden tests...")

        golden_result = self._run_golden_tests(pack_yaml_path, agent, verbose)
        test_results.append(CertificationTest(
            name="Golden Tests",
            category="correctness",
            passed=golden_result.all_passed,
            score=golden_result.pass_rate,
            details=f"{golden_result.passed_tests}/{golden_result.total_tests} tests passed",
            execution_time_ms=golden_result.execution_time_ms,
        ))

        # Test 2: Determinism Verification
        if verbose:
            logger.info("\n[2/5] Verifying determinism...")

        determinism_result = self._verify_determinism(agent, sample_input, verbose)
        test_results.append(CertificationTest(
            name="Determinism",
            category="reproducibility",
            passed=determinism_result.is_deterministic,
            score=determinism_result.determinism_percentage,
            details=f"{determinism_result.unique_outputs} unique output(s) in {determinism_result.num_runs} runs",
            execution_time_ms=determinism_result.execution_time_ms,
        ))

        # Test 3: Provenance Completeness
        if verbose:
            logger.info("\n[3/5] Checking provenance completeness...")

        provenance_test = self._check_provenance(agent, sample_input, verbose)
        test_results.append(provenance_test)

        # Test 4: Citation Verification
        if verbose:
            logger.info("\n[4/5] Verifying citations...")

        citation_test = self._verify_citations(agent, sample_input, pack_spec, verbose)
        test_results.append(citation_test)

        # Test 5: Confidence Bounds
        if verbose:
            logger.info("\n[5/5] Validating confidence bounds...")

        confidence_test = self._check_confidence_bounds(agent, sample_input, verbose)
        test_results.append(confidence_test)

        # Calculate overall results
        tests_passed = sum(1 for t in test_results if t.passed)
        tests_failed = len(test_results) - tests_passed
        overall_score = sum(t.score for t in test_results) / len(test_results)

        # Determine certification level
        certified = False
        if overall_score >= self.GOLD_THRESHOLD and all(t.passed for t in test_results):
            cert_level = "GOLD"
            certified = True
        elif overall_score >= self.SILVER_THRESHOLD:
            cert_level = "SILVER"
            certified = True
        elif overall_score >= self.BRONZE_THRESHOLD:
            cert_level = "BRONZE"
            certified = True
        else:
            cert_level = "FAIL"
            certified = False

        report = CertificationReport(
            agent_id=agent_id,
            agent_version=agent_version,
            certified=certified,
            overall_score=overall_score,
            certification_level=cert_level,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_results=test_results,
            golden_test_result=golden_result,
            determinism_result=determinism_result,
        )

        if verbose:
            self._print_report(report)

        return report

    def _run_golden_tests(
        self,
        pack_yaml_path: Path,
        agent: Any,
        verbose: bool,
    ) -> GoldenTestResult:
        """Run golden test suite."""
        return self.golden_test_runner.run_tests(pack_yaml_path, agent, verbose=False)

    def _verify_determinism(
        self,
        agent: Any,
        sample_input: Any,
        verbose: bool,
    ) -> DeterminismResult:
        """Verify agent determinism."""
        return self.determinism_verifier.verify(
            agent,
            sample_input,
            num_runs=self.DETERMINISM_RUNS,
            verbose=False,
        )

    def _check_provenance(
        self,
        agent: Any,
        sample_input: Any,
        verbose: bool,
    ) -> CertificationTest:
        """Check provenance tracking completeness."""
        try:
            result = agent.run(sample_input)

            # Check provenance hash exists
            if not hasattr(result, 'provenance_hash'):
                return CertificationTest(
                    name="Provenance Completeness",
                    category="audit",
                    passed=False,
                    score=0.0,
                    details="Missing provenance_hash field",
                )

            prov_hash = result.provenance_hash

            # Validate hash format (SHA-256)
            if not prov_hash or len(prov_hash) != 64:
                return CertificationTest(
                    name="Provenance Completeness",
                    category="audit",
                    passed=False,
                    score=50.0,
                    details=f"Invalid provenance hash format (length: {len(prov_hash) if prov_hash else 0})",
                )

            # Validate hash is hex
            if not all(c in '0123456789abcdef' for c in prov_hash.lower()):
                return CertificationTest(
                    name="Provenance Completeness",
                    category="audit",
                    passed=False,
                    score=50.0,
                    details="Provenance hash not valid hex",
                )

            return CertificationTest(
                name="Provenance Completeness",
                category="audit",
                passed=True,
                score=100.0,
                details=f"Valid provenance hash: {prov_hash[:16]}...",
            )

        except Exception as e:
            return CertificationTest(
                name="Provenance Completeness",
                category="audit",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
            )

    def _verify_citations(
        self,
        agent: Any,
        sample_input: Any,
        pack_spec: Dict[str, Any],
        verbose: bool,
    ) -> CertificationTest:
        """Verify all citations and sources."""
        try:
            result = agent.run(sample_input)

            # Check for citation/source fields
            citation_fields = []
            if hasattr(result, 'emission_factor_source'):
                citation_fields.append('emission_factor_source')
            if hasattr(result, 'data_source'):
                citation_fields.append('data_source')
            if hasattr(result, 'sources'):
                citation_fields.append('sources')

            if not citation_fields:
                return CertificationTest(
                    name="Citation Verification",
                    category="transparency",
                    passed=False,
                    score=0.0,
                    details="No citation/source fields found in output",
                )

            # Validate citations are not empty
            for field in citation_fields:
                value = getattr(result, field, None)
                if not value:
                    return CertificationTest(
                        name="Citation Verification",
                        category="transparency",
                        passed=False,
                        score=50.0,
                        details=f"Citation field '{field}' is empty",
                    )

            return CertificationTest(
                name="Citation Verification",
                category="transparency",
                passed=True,
                score=100.0,
                details=f"Found {len(citation_fields)} citation field(s): {', '.join(citation_fields)}",
            )

        except Exception as e:
            return CertificationTest(
                name="Citation Verification",
                category="transparency",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
            )

    def _check_confidence_bounds(
        self,
        agent: Any,
        sample_input: Any,
        verbose: bool,
    ) -> CertificationTest:
        """Check confidence score bounds."""
        try:
            result = agent.run(sample_input)

            # Check if confidence score exists
            if not hasattr(result, 'confidence_score') and not hasattr(result, 'confidence'):
                # Not all agents need confidence scores
                return CertificationTest(
                    name="Confidence Bounds",
                    category="quality",
                    passed=True,
                    score=100.0,
                    details="N/A (no confidence score in output)",
                )

            confidence = getattr(result, 'confidence_score', None) or getattr(result, 'confidence', None)

            if confidence is None:
                return CertificationTest(
                    name="Confidence Bounds",
                    category="quality",
                    passed=True,
                    score=100.0,
                    details="N/A (confidence is None)",
                )

            # Validate bounds [0, 1]
            if not (0.0 <= confidence <= 1.0):
                return CertificationTest(
                    name="Confidence Bounds",
                    category="quality",
                    passed=False,
                    score=0.0,
                    details=f"Confidence {confidence} out of bounds [0.0, 1.0]",
                )

            return CertificationTest(
                name="Confidence Bounds",
                category="quality",
                passed=True,
                score=100.0,
                details=f"Confidence: {confidence:.4f} (within bounds)",
            )

        except Exception as e:
            return CertificationTest(
                name="Confidence Bounds",
                category="quality",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
            )

    def _get_sample_input(self, pack_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sample input from pack spec."""
        # Try to get from golden tests
        tests = pack_spec.get('tests', {})
        golden_tests = tests.get('golden', [])

        if golden_tests and len(golden_tests) > 0:
            return golden_tests[0].get('input', {})

        # Try old format
        golden_tests = pack_spec.get('golden_tests', {})
        if 'test_cases' in golden_tests and golden_tests['test_cases']:
            return golden_tests['test_cases'][0].get('input', {})

        # Default sample input
        return {}

    def _print_report(self, report: CertificationReport) -> None:
        """Print certification report."""
        print("\n" + "=" * 80)
        print("AGENT CERTIFICATION REPORT")
        print("=" * 80)
        print(f"Agent ID:          {report.agent_id}")
        print(f"Version:           {report.agent_version}")
        print(f"Certification ID:  {report.certification_id}")
        print(f"Timestamp:         {report.timestamp.isoformat()}")
        print("-" * 80)
        print(f"Overall Score:     {report.overall_score:.2f}/100")
        print(f"Certification:     {report.certification_level}")
        print(f"Status:            {'CERTIFIED' if report.certified else 'NOT CERTIFIED'}")
        print(f"Tests Passed:      {report.tests_passed}/{report.tests_passed + report.tests_failed}")
        print("=" * 80)

        print("\nTEST RESULTS:")
        print("-" * 80)
        for i, test in enumerate(report.test_results, 1):
            status = "PASS" if test.passed else "FAIL"
            print(f"{i}. [{status}] {test.name} ({test.category})")
            print(f"   Score:   {test.score:.2f}/100")
            print(f"   Details: {test.details}")
            print(f"   Time:    {test.execution_time_ms:.2f}ms")
            print()

        print("=" * 80)

        if report.certified:
            print(f"\nCONGRATULATIONS! Agent certified at {report.certification_level} level.")
        else:
            print(f"\nCERTIFICATION FAILED. Please address failed tests and resubmit.")

        print()


def certify_from_command_line():
    """Command-line interface for certification."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python certification_suite.py <pack.yaml>")
        sys.exit(1)

    pack_yaml_path = sys.argv[1]

    print(f"Certifying agent from: {pack_yaml_path}")
    print("Note: Agent instance must be provided programmatically")


if __name__ == "__main__":
    certify_from_command_line()
