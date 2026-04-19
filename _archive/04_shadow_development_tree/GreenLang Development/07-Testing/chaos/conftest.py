"""
Pytest configuration for Chaos Engineering Tests

Provides:
- Fixtures for chaos test setup and teardown
- Markers for chaos test categorization
- Configuration for chaos test execution
- Utilities for monitoring and reporting
"""

import pytest
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Generator
import tempfile

from .chaos_tests import ChaosTestRunner, SteadyStateMetrics, ChaosTestResult

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Pytest Configuration
# ==============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "chaos: mark test as a chaos engineering test",
    )
    config.addinivalue_line(
        "markers",
        "chaos_failover: test for agent failover scenarios",
    )
    config.addinivalue_line(
        "markers",
        "chaos_database: test for database resilience",
    )
    config.addinivalue_line(
        "markers",
        "chaos_latency: test for high latency handling",
    )
    config.addinivalue_line(
        "markers",
        "chaos_resource: test for resource pressure handling",
    )


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def chaos_report_dir() -> Path:
    """Create directory for chaos test reports."""
    report_dir = Path(tempfile.gettempdir()) / "chaos-reports"
    report_dir.mkdir(exist_ok=True)
    logger.info(f"Chaos reports will be saved to: {report_dir}")
    return report_dir


@pytest.fixture
def chaos_runner_session_scoped() -> Generator[ChaosTestRunner, None, None]:
    """Create a session-scoped ChaosTestRunner."""
    runner = ChaosTestRunner(environment="test")
    yield runner
    runner.stop_all_chaos()


@pytest.fixture
def chaos_runner() -> Generator[ChaosTestRunner, None, None]:
    """Create a function-scoped ChaosTestRunner."""
    runner = ChaosTestRunner(environment="test")
    yield runner
    runner.stop_all_chaos()


@pytest.fixture
def steady_state_metrics() -> SteadyStateMetrics:
    """Create steady-state metrics for validation."""
    return SteadyStateMetrics(
        max_latency_ms=5000.0,
        min_availability_percent=99.0,
        max_error_rate_percent=1.0,
        max_memory_mb=1000.0,
        min_throughput_rps=100.0,
    )


@pytest.fixture
def chaos_test_result() -> ChaosTestResult:
    """Create a ChaosTestResult instance."""
    return ChaosTestResult("mock_test")


# ==============================================================================
# Hooks
# ==============================================================================


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture chaos test results and generate reports.
    """
    if call.when == "call" and "chaos" in item.keywords:
        # This hook could be extended to capture metrics
        pass


def pytest_sessionfinish(session, exitstatus):
    """
    Generate comprehensive chaos test report at end of session.
    """
    logger.info(f"Chaos tests completed with exit status: {exitstatus}")


# ==============================================================================
# Utilities
# ==============================================================================


class ChaosTestReporter:
    """Generate reports from chaos test results."""

    @staticmethod
    def generate_json_report(results: list, output_path: Path) -> None:
        """
        Generate JSON report of chaos test results.

        Args:
            results: List of ChaosTestResult objects
            output_path: Path to write report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "results": [r.to_dict() for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report written to {output_path}")

    @staticmethod
    def generate_html_report(results: list, output_path: Path) -> None:
        """
        Generate HTML report of chaos test results.

        Args:
            results: List of ChaosTestResult objects
            output_path: Path to write report
        """
        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if not r.passed)

        html = f"""
        <html>
            <head>
                <title>Chaos Engineering Test Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .summary {{ background: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
                    .passed {{ color: green; }}
                    .failed {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                </style>
            </head>
            <body>
                <h1>Chaos Engineering Test Report</h1>
                <div class="summary">
                    <p>Total Tests: {len(results)}</p>
                    <p class="passed">Passed: {passed_count}</p>
                    <p class="failed">Failed: {failed_count}</p>
                </div>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Duration (s)</th>
                        <th>Errors</th>
                    </tr>
        """

        for result in results:
            status = "PASSED" if result.passed else "FAILED"
            status_class = "passed" if result.passed else "failed"
            duration = (
                (result.end_time - result.start_time).total_seconds()
                if result.end_time
                else "N/A"
            )
            errors_html = (
                f"<ul>{''.join(f'<li>{e}</li>' for e in result.errors)}</ul>"
                if result.errors
                else "None"
            )

            html += f"""
                    <tr>
                        <td>{result.test_name}</td>
                        <td class="{status_class}">{status}</td>
                        <td>{duration}</td>
                        <td>{errors_html}</td>
                    </tr>
            """

        html += """
                </table>
            </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"HTML report written to {output_path}")
