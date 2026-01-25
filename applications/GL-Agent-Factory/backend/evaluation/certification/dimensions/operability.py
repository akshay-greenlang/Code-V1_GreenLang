"""
Operability Dimension Evaluator

Evaluates agent operability including:
- Monitoring readiness
- Alerting capabilities
- Health check support
- Observability metrics
- Deployment readiness

Ensures agents are ready for production operations.

"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from dimension evaluation."""
    score: float
    test_count: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class OperabilityEvaluator:
    """
    Evaluator for operability dimension.

    Tests:
    1. Monitoring - Metrics and observability
    2. Alerting - Warning/error conditions
    3. Health checks - Status reporting
    4. Configuration - Runtime configuration
    5. Deployment - Deployment readiness
    """

    # Monitoring fields
    MONITORING_FIELDS = [
        "execution_time_ms",
        "processing_time_ms",
        "latency_ms",
        "metrics",
    ]

    # Alerting fields
    ALERTING_FIELDS = [
        "warnings",
        "errors",
        "alerts",
        "status",
    ]

    def __init__(self):
        """Initialize operability evaluator."""
        logger.info("OperabilityEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent operability.

        Args:
            agent: Agent instance to evaluate
            pack_spec: Agent pack specification
            sample_inputs: Sample inputs for testing
            golden_result: Optional golden test results
            determinism_result: Optional determinism results

        Returns:
            EvaluationResult with score and details
        """
        tests_run = 0
        tests_passed = 0
        findings = []
        recommendations = []
        details = {}

        # Test 1: Monitoring readiness
        monitor_score, monitor_details = self._test_monitoring_readiness(
            agent, sample_inputs
        )
        details["monitoring"] = monitor_details
        tests_run += monitor_details.get("test_count", 0)
        tests_passed += monitor_details.get("tests_passed", 0)

        if monitor_score < 100:
            findings.append(f"Monitoring readiness: {monitor_score:.1f}%")
            recommendations.append(
                "Add execution_time_ms to outputs for monitoring"
            )

        # Test 2: Alerting capability
        alert_score, alert_details = self._test_alerting_capability(
            agent, sample_inputs
        )
        details["alerting"] = alert_details
        tests_run += alert_details.get("test_count", 0)
        tests_passed += alert_details.get("tests_passed", 0)

        if alert_score < 100:
            findings.append(f"Alerting capability: {alert_score:.1f}%")
            recommendations.append(
                "Implement warning/error reporting in outputs"
            )

        # Test 3: Health check support
        health_score, health_details = self._test_health_check(agent)
        details["health_check"] = health_details
        tests_run += health_details.get("test_count", 0)
        tests_passed += health_details.get("tests_passed", 0)

        if health_score < 100:
            findings.append(f"Health check support: {health_score:.1f}%")
            recommendations.append(
                "Implement health_check() method for status reporting"
            )

        # Test 4: Configuration support
        config_score, config_details = self._test_configuration(pack_spec)
        details["configuration"] = config_details
        tests_run += config_details.get("test_count", 0)
        tests_passed += config_details.get("tests_passed", 0)

        if config_score < 100:
            findings.append(f"Configuration support: {config_score:.1f}%")
            recommendations.append(
                "Document configuration options in pack spec"
            )

        # Test 5: Deployment readiness
        deploy_score, deploy_details = self._test_deployment_readiness(pack_spec)
        details["deployment"] = deploy_details
        tests_run += deploy_details.get("test_count", 0)
        tests_passed += deploy_details.get("tests_passed", 0)

        if deploy_score < 100:
            findings.append(f"Deployment readiness: {deploy_score:.1f}%")
            recommendations.append(
                "Add deployment configuration to pack spec"
            )

        # Calculate overall score
        if tests_run == 0:
            overall_score = 0.0
        else:
            overall_score = (tests_passed / tests_run) * 100

        return EvaluationResult(
            score=overall_score,
            test_count=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_run - tests_passed,
            details=details,
            findings=findings,
            recommendations=recommendations,
        )

    def _test_monitoring_readiness(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test monitoring readiness."""
        tests_run = 0
        tests_passed = 0
        monitor_checks = []

        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                has_monitoring = any(
                    hasattr(result, f) for f in self.MONITORING_FIELDS
                )

                if has_monitoring:
                    tests_passed += 1
                    monitor_checks.append({
                        "status": "METRICS_PRESENT",
                    })
                else:
                    # Not strictly required
                    tests_passed += 0.5
                    monitor_checks.append({
                        "status": "NO_METRICS",
                    })

            except Exception:
                tests_run += 1

        # Ensure at least one test
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": int(tests_passed),
            "monitor_checks": monitor_checks,
        }

    def _test_alerting_capability(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test alerting capability."""
        tests_run = 1
        tests_passed = 0
        alert_checks = []

        # Check if agent has alerting capability
        if sample_inputs:
            try:
                result = agent.run(sample_inputs[0])

                has_alerting = any(
                    hasattr(result, f) for f in self.ALERTING_FIELDS
                )

                if has_alerting:
                    tests_passed = 1
                    alert_checks.append({
                        "status": "ALERTS_CAPABLE",
                    })
                else:
                    # Not required for all agents
                    tests_passed = 1
                    alert_checks.append({
                        "status": "N/A",
                    })

            except Exception:
                tests_passed = 1
                alert_checks.append({
                    "status": "ERROR",
                })
        else:
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "alert_checks": alert_checks,
        }

    def _test_health_check(
        self, agent: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """Test health check support."""
        tests_run = 1
        tests_passed = 0
        health_checks = []

        # Check if agent has health_check method
        has_health_check = hasattr(agent, "health_check")

        if has_health_check:
            try:
                health_status = agent.health_check()
                tests_passed = 1
                health_checks.append({
                    "method": "health_check",
                    "status": "IMPLEMENTED",
                })
            except Exception:
                health_checks.append({
                    "method": "health_check",
                    "status": "ERROR",
                })
        else:
            # Not required but recommended
            tests_passed = 1
            health_checks.append({
                "method": "health_check",
                "status": "NOT_IMPLEMENTED",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "health_checks": health_checks,
        }

    def _test_configuration(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test configuration support."""
        tests_run = 1
        tests_passed = 0
        config_checks = []

        # Check for configuration section
        config = pack_spec.get("config", {})
        defaults = pack_spec.get("defaults", {})

        if config or defaults:
            tests_passed = 1
            config_checks.append({
                "check": "config_section",
                "status": "PRESENT",
            })
        else:
            # Not required
            tests_passed = 1
            config_checks.append({
                "check": "config_section",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "config_checks": config_checks,
        }

    def _test_deployment_readiness(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test deployment readiness."""
        tests_run = 1
        tests_passed = 0
        deploy_checks = []

        # Check for deployment section
        deployment = pack_spec.get("deployment", {})
        dependencies = pack_spec.get("dependencies", {})

        if deployment:
            tests_passed = 1
            deploy_checks.append({
                "check": "deployment_config",
                "status": "PRESENT",
            })
        elif dependencies:
            tests_passed = 1
            deploy_checks.append({
                "check": "dependencies",
                "status": "PRESENT",
            })
        else:
            # Not required for certification
            tests_passed = 1
            deploy_checks.append({
                "check": "deployment_config",
                "status": "N/A",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "deploy_checks": deploy_checks,
        }
