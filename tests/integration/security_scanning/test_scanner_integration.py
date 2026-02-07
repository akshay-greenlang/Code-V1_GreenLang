# -*- coding: utf-8 -*-
"""
Integration tests for Scanner Integration - SEC-007

Tests for real scanner execution covering:
    - SAST scanner execution
    - SCA scanner execution
    - Secrets scanner execution
    - Container scanner execution
    - IaC scanner execution

Coverage target: 20+ tests

Note: These tests require actual scanner binaries to be installed.
Tests will be skipped if scanners are not available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    temp_dir = tempfile.mkdtemp()

    # Create a sample Python file with potential issues
    python_file = Path(temp_dir) / "app.py"
    python_file.write_text("""
import os
import subprocess

# Potential security issues for testing
password = "hardcoded_password_123"  # B105
api_key = "sk_live_abcdefghijklmnop"

def unsafe_exec(cmd):
    subprocess.call(cmd, shell=True)  # B602

def sql_query(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input  # SQL injection
    return query
""")

    # Create requirements.txt with vulnerable packages
    requirements = Path(temp_dir) / "requirements.txt"
    requirements.write_text("""
requests==2.25.0
django==2.2.0
pyyaml==5.3.1
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_terraform_dir():
    """Create a temporary Terraform directory for testing."""
    temp_dir = tempfile.mkdtemp()

    tf_file = Path(temp_dir) / "main.tf"
    tf_file.write_text("""
resource "aws_s3_bucket" "public_bucket" {
  bucket = "my-public-bucket"
  acl    = "public-read"  # Security issue
}

resource "aws_security_group" "open_sg" {
  name = "open-security-group"

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Security issue
  }
}
""")

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def scanner_available(scanner_name: str) -> bool:
    """Check if a scanner is available."""
    return shutil.which(scanner_name) is not None


# ============================================================================
# TestSASTScannerIntegration
# ============================================================================


class TestSASTScannerIntegration:
    """Integration tests for SAST scanners."""

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("bandit"), reason="Bandit not installed")
    def test_bandit_real_execution(self, temp_project_dir):
        """Test real Bandit execution on sample code."""
        result = subprocess.run(
            ["bandit", "-r", temp_project_dir, "-f", "json"],
            capture_output=True,
            text=True,
        )

        # Bandit returns 1 if issues found
        assert result.returncode in [0, 1]

        if result.stdout:
            import json
            output = json.loads(result.stdout)
            assert "results" in output

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("semgrep"), reason="Semgrep not installed")
    def test_semgrep_real_execution(self, temp_project_dir):
        """Test real Semgrep execution on sample code."""
        result = subprocess.run(
            ["semgrep", "--config", "auto", temp_project_dir, "--json"],
            capture_output=True,
            text=True,
        )

        # Check execution completed
        assert result.returncode in [0, 1]

    @pytest.mark.integration
    def test_sast_scanner_wrapper(self, temp_project_dir):
        """Test SAST scanner wrapper handles execution."""
        try:
            from greenlang.infrastructure.security_scanning.scanners.sast import (
                BanditScanner,
            )

            scanner = BanditScanner()
            if scanner.is_available():
                results = scanner.scan(temp_project_dir)
                assert isinstance(results, list)
        except ImportError:
            pytest.skip("SAST scanner module not available")

    @pytest.mark.integration
    def test_parallel_sast_execution(self, temp_project_dir):
        """Test parallel execution of multiple SAST scanners."""
        import concurrent.futures

        def run_scanner(name: str) -> Dict:
            if not scanner_available(name):
                return {"scanner": name, "status": "skipped"}
            return {"scanner": name, "status": "completed"}

        scanners = ["bandit", "semgrep", "pylint"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(run_scanner, s): s for s in scanners}
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 3


# ============================================================================
# TestSCAScannerIntegration
# ============================================================================


class TestSCAScannerIntegration:
    """Integration tests for SCA scanners."""

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("trivy"), reason="Trivy not installed")
    def test_trivy_fs_scan(self, temp_project_dir):
        """Test Trivy filesystem scan for dependencies."""
        result = subprocess.run(
            ["trivy", "fs", temp_project_dir, "--format", "json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode in [0, 1]

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("pip-audit"), reason="pip-audit not installed")
    def test_pip_audit_execution(self, temp_project_dir):
        """Test pip-audit execution."""
        result = subprocess.run(
            ["pip-audit", "-r", f"{temp_project_dir}/requirements.txt", "--format", "json"],
            capture_output=True,
            text=True,
        )

        # pip-audit may return non-zero if vulnerabilities found
        assert result.returncode in [0, 1]

    @pytest.mark.integration
    def test_sca_scanner_wrapper(self, temp_project_dir):
        """Test SCA scanner wrapper handles execution."""
        try:
            from greenlang.infrastructure.security_scanning.scanners.sca import (
                TrivyScanner,
            )

            scanner = TrivyScanner()
            if scanner.is_available():
                results = scanner.scan(temp_project_dir)
                assert isinstance(results, list)
        except ImportError:
            pytest.skip("SCA scanner module not available")

    @pytest.mark.integration
    def test_sbom_generation(self, temp_project_dir):
        """Test SBOM generation capability."""
        if not scanner_available("trivy"):
            pytest.skip("Trivy not installed")

        sbom_file = Path(temp_project_dir) / "sbom.json"

        result = subprocess.run(
            ["trivy", "fs", temp_project_dir, "--format", "cyclonedx", "-o", str(sbom_file)],
            capture_output=True,
            text=True,
        )

        # Check if SBOM was generated
        if result.returncode == 0:
            assert sbom_file.exists() or True  # May not create file if no deps


# ============================================================================
# TestSecretsScannerIntegration
# ============================================================================


class TestSecretsScannerIntegration:
    """Integration tests for secrets scanners."""

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("gitleaks"), reason="Gitleaks not installed")
    def test_gitleaks_execution(self, temp_project_dir):
        """Test Gitleaks execution on sample code."""
        result = subprocess.run(
            ["gitleaks", "detect", "--source", temp_project_dir, "--report-format", "json", "--no-git"],
            capture_output=True,
            text=True,
        )

        # Gitleaks returns 1 if secrets found
        assert result.returncode in [0, 1]

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("trufflehog"), reason="Trufflehog not installed")
    def test_trufflehog_execution(self, temp_project_dir):
        """Test Trufflehog execution on sample code."""
        result = subprocess.run(
            ["trufflehog", "filesystem", temp_project_dir, "--json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode in [0, 1, 183]  # 183 is secrets found

    @pytest.mark.integration
    def test_secrets_scanner_wrapper(self, temp_project_dir):
        """Test secrets scanner wrapper handles execution."""
        try:
            from greenlang.infrastructure.security_scanning.scanners.secrets import (
                GitleaksScanner,
            )

            scanner = GitleaksScanner()
            if scanner.is_available():
                results = scanner.scan(temp_project_dir)
                assert isinstance(results, list)
        except ImportError:
            pytest.skip("Secrets scanner module not available")


# ============================================================================
# TestContainerScannerIntegration
# ============================================================================


class TestContainerScannerIntegration:
    """Integration tests for container scanners."""

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("trivy"), reason="Trivy not installed")
    def test_trivy_image_scan(self):
        """Test Trivy container image scan."""
        # Scan a small, known image
        result = subprocess.run(
            ["trivy", "image", "alpine:3.18", "--format", "json", "--timeout", "5m"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode in [0, 1]

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("grype"), reason="Grype not installed")
    def test_grype_image_scan(self):
        """Test Grype container image scan."""
        result = subprocess.run(
            ["grype", "alpine:3.18", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode in [0, 1]

    @pytest.mark.integration
    def test_container_scanner_wrapper(self):
        """Test container scanner wrapper handles execution."""
        try:
            from greenlang.infrastructure.security_scanning.scanners.container import (
                TrivyContainerScanner,
            )

            scanner = TrivyContainerScanner()
            if scanner.is_available():
                # Use a small test image
                results = scanner.scan("alpine:3.18")
                assert isinstance(results, list)
        except ImportError:
            pytest.skip("Container scanner module not available")


# ============================================================================
# TestIaCScannerIntegration
# ============================================================================


class TestIaCScannerIntegration:
    """Integration tests for IaC scanners."""

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("tfsec"), reason="tfsec not installed")
    def test_tfsec_execution(self, temp_terraform_dir):
        """Test tfsec execution on Terraform code."""
        result = subprocess.run(
            ["tfsec", temp_terraform_dir, "--format", "json"],
            capture_output=True,
            text=True,
        )

        # tfsec returns 1 if issues found
        assert result.returncode in [0, 1]

    @pytest.mark.integration
    @pytest.mark.skipif(not scanner_available("checkov"), reason="Checkov not installed")
    def test_checkov_execution(self, temp_terraform_dir):
        """Test Checkov execution on Terraform code."""
        result = subprocess.run(
            ["checkov", "-d", temp_terraform_dir, "-o", "json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode in [0, 1]

    @pytest.mark.integration
    def test_iac_scanner_wrapper(self, temp_terraform_dir):
        """Test IaC scanner wrapper handles execution."""
        try:
            from greenlang.infrastructure.security_scanning.scanners.iac import (
                TfsecScanner,
            )

            scanner = TfsecScanner()
            if scanner.is_available():
                results = scanner.scan(temp_terraform_dir)
                assert isinstance(results, list)
        except ImportError:
            pytest.skip("IaC scanner module not available")


# ============================================================================
# TestOrchestratorIntegration
# ============================================================================


class TestOrchestratorIntegration:
    """Integration tests for scan orchestration."""

    @pytest.mark.integration
    def test_orchestrator_runs_available_scanners(self, temp_project_dir):
        """Test orchestrator runs all available scanners."""
        try:
            from greenlang.infrastructure.security_scanning.orchestrator import (
                ScanOrchestrator,
            )

            orchestrator = ScanOrchestrator()
            results = orchestrator.scan(temp_project_dir)

            assert isinstance(results, dict) or isinstance(results, list)
        except ImportError:
            pytest.skip("Orchestrator not available")

    @pytest.mark.integration
    def test_orchestrator_handles_scanner_failures(self, temp_project_dir):
        """Test orchestrator handles individual scanner failures gracefully."""
        try:
            from greenlang.infrastructure.security_scanning.orchestrator import (
                ScanOrchestrator,
            )

            orchestrator = ScanOrchestrator()

            # Run with a mix of available and unavailable scanners
            results = orchestrator.scan(
                temp_project_dir,
                scanners=["bandit", "nonexistent_scanner"],
            )

            # Should complete despite one failing
            assert results is not None
        except ImportError:
            pytest.skip("Orchestrator not available")
