# -*- coding: utf-8 -*-
"""
Secret Scanner Implementations - SEC-007

Secret detection scanners for finding exposed credentials, API keys,
and other sensitive data in source code and git history.

Scanners:
    - GitleaksScanner: Git-aware secret detection
    - TrufflehogScanner: Entropy and regex-based secret detection
    - DetectSecretsScanner: Pre-commit focused secret detection

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners.secrets import (
    ...     GitleaksScanner,
    ...     TrufflehogScanner,
    ... )
    >>> config = ScannerConfig(name="gitleaks", scanner_type=ScannerType.SECRETS)
    >>> scanner = GitleaksScanner(config)
    >>> result = await scanner.scan("/path/to/repo")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.security_scanning.config import (
    Severity,
    ScannerConfig,
    ScannerType,
)
from greenlang.infrastructure.security_scanning.models import (
    FileLocation,
    ScanFinding,
    ScanResult,
    ScanStatus,
    VulnerabilityInfo,
    RemediationInfo,
)
from greenlang.infrastructure.security_scanning.scanners.base import (
    BaseScanner,
    normalize_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gitleaks Scanner
# ---------------------------------------------------------------------------


class GitleaksScanner(BaseScanner):
    """Gitleaks scanner for secret detection.

    Gitleaks scans repositories for secrets using regex patterns and
    entropy analysis. Supports custom rules via TOML configuration.

    Severity Mapping:
        All secrets are treated as CRITICAL by default since exposed
        credentials require immediate attention.

    Example:
        >>> config = ScannerConfig(
        ...     name="gitleaks",
        ...     scanner_type=ScannerType.SECRETS,
        ...     config_file=".gitleaks.toml"
        ... )
        >>> scanner = GitleaksScanner(config)
        >>> result = await scanner.scan("/path/to/repo")
    """

    # Rule severity overrides (most secrets are critical)
    RULE_SEVERITY_MAP: Dict[str, Severity] = {
        "generic-api-key": Severity.HIGH,
        "generic-password": Severity.HIGH,
        "private-key": Severity.CRITICAL,
        "aws-access-key": Severity.CRITICAL,
        "github-token": Severity.CRITICAL,
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Gitleaks scan.

        Args:
            target_path: Path to scan (directory or git repo).

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Gitleaks not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Gitleaks exit codes:
            # 0 = no leaks found
            # 1 = leaks found
            # 2+ = error

            if exit_code > 1:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"Gitleaks failed with exit code {exit_code}",
                    exit_code=exit_code,
                    command=" ".join(command),
                    scan_path=target_path,
                )

            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("Gitleaks scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Gitleaks command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "gitleaks"]
        cmd.extend(["detect"])
        cmd.extend(["--source", target_path])
        cmd.extend(["--report-format", "json"])
        cmd.extend(["--report-path", "/dev/stdout"])  # Output to stdout

        # Don't scan git history by default (faster)
        if "--no-git" not in self.config.extra_args:
            cmd.append("--no-git")

        # Add config file if specified
        if self.config.config_file:
            cmd.extend(["--config", self.config.config_file])

        # Add baseline if exists
        import os

        baseline_path = os.path.join(target_path, ".gitleaks-baseline.json")
        if os.path.exists(baseline_path):
            cmd.extend(["--baseline-path", baseline_path])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Gitleaks JSON output.

        Args:
            raw_output: JSON output from Gitleaks.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            # Gitleaks outputs an array of findings
            data = json.loads(raw_output)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Gitleaks output: %s", e)
            return []

        findings = []
        for leak in data:
            finding = self._parse_leak(leak)
            if finding:
                findings.append(finding)

        logger.info("Gitleaks found %d secrets", len(findings))
        return findings

    def _parse_leak(self, leak: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single Gitleaks leak.

        Args:
            leak: Leak dictionary from Gitleaks.

        Returns:
            ScanFinding or None.
        """
        try:
            rule_id = leak.get("RuleID", "")
            description = leak.get("Description", rule_id)
            file_path = leak.get("File", "")
            line_number = leak.get("StartLine", 1)
            secret = leak.get("Secret", "")
            match = leak.get("Match", "")
            commit = leak.get("Commit", "")

            # Determine severity (secrets are generally critical)
            severity = self.RULE_SEVERITY_MAP.get(rule_id, Severity.CRITICAL)

            location = FileLocation(
                file_path=normalize_path(file_path),
                start_line=line_number,
                end_line=leak.get("EndLine", line_number),
                start_column=leak.get("StartColumn"),
                end_column=leak.get("EndColumn"),
                # Redact the actual secret in snippet
                snippet=self._redact_secret(match, secret) if match else None,
            )

            remediation = RemediationInfo(
                description=(
                    "1. Rotate the exposed credential immediately.\n"
                    "2. Remove the secret from the codebase.\n"
                    "3. Use environment variables or a secrets manager.\n"
                    "4. If committed to git, consider rewriting history."
                ),
                effort_estimate="high" if commit else "medium",
            )

            return ScanFinding(
                title=f"Exposed Secret: {description}",
                description=(
                    f"Potential secret detected in {file_path}:{line_number}. "
                    f"Rule: {rule_id}. "
                    f"{'Commit: ' + commit[:8] if commit else 'File scan only.'}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SECRETS,
                rule_id=rule_id,
                location=location,
                remediation_info=remediation,
                tags={"secret", "credential", rule_id.lower().replace("-", "_")},
                raw_data=self._redact_raw_data(leak),
            )

        except Exception as e:
            logger.warning("Failed to parse Gitleaks leak: %s", e)
            return None

    def _redact_secret(self, match: str, secret: str) -> str:
        """Redact secret from match string for safe logging.

        Args:
            match: Full match string.
            secret: The actual secret value.

        Returns:
            Redacted string.
        """
        if not secret or len(secret) < 4:
            return match.replace(secret, "[REDACTED]")

        # Keep first and last 2 chars for debugging
        redacted = secret[:2] + "*" * (len(secret) - 4) + secret[-2:]
        return match.replace(secret, redacted)

    def _redact_raw_data(self, leak: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from raw data.

        Args:
            leak: Original leak data.

        Returns:
            Redacted copy.
        """
        redacted = leak.copy()
        if "Secret" in redacted:
            secret = redacted["Secret"]
            if len(secret) > 4:
                redacted["Secret"] = secret[:2] + "[REDACTED]" + secret[-2:]
            else:
                redacted["Secret"] = "[REDACTED]"
        return redacted

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": "Secret detected in source code"},
                    "helpUri": "https://github.com/gitleaks/gitleaks",
                    "properties": {
                        "security-severity": "9.0",  # Secrets are high severity
                    },
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Gitleaks",
                            "version": self.version or "unknown",
                            "informationUri": "https://github.com/gitleaks/gitleaks",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://github.com/gitleaks/gitleaks"


# ---------------------------------------------------------------------------
# TruffleHog Scanner
# ---------------------------------------------------------------------------


class TrufflehogScanner(BaseScanner):
    """TruffleHog scanner for high-entropy secret detection.

    TruffleHog uses entropy analysis and regex patterns to find secrets.
    Supports scanning git repos, S3 buckets, and filesystems.

    Example:
        >>> config = ScannerConfig(name="trufflehog", scanner_type=ScannerType.SECRETS)
        >>> scanner = TrufflehogScanner(config)
        >>> result = await scanner.scan("/path/to/repo")
    """

    async def scan(self, target_path: str) -> ScanResult:
        """Execute TruffleHog scan.

        Args:
            target_path: Path to scan.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"TruffleHog not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # TruffleHog exits 0 even with findings (outputs JSON lines)
            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("TruffleHog scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build TruffleHog command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "trufflehog"]
        cmd.extend(["filesystem", target_path])
        cmd.extend(["--json"])

        # Only scan current state, not git history
        if "--only-verified" not in self.config.extra_args:
            cmd.append("--only-verified")

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse TruffleHog JSON lines output.

        Args:
            raw_output: JSON lines output from TruffleHog.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        findings = []

        # TruffleHog outputs JSON lines (one JSON object per line)
        for line in raw_output.strip().split("\n"):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
                finding = self._parse_result(result)
                if finding:
                    findings.append(finding)
            except json.JSONDecodeError:
                continue

        logger.info("TruffleHog found %d secrets", len(findings))
        return findings

    def _parse_result(self, result: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single TruffleHog result.

        Args:
            result: Result dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            detector_name = result.get("DetectorName", "unknown")
            raw = result.get("Raw", "")
            source_metadata = result.get("SourceMetadata", {})
            data = source_metadata.get("Data", {})

            file_path = data.get("Filesystem", {}).get("file", "")
            line = data.get("Filesystem", {}).get("line", 1)

            verified = result.get("Verified", False)

            # Verified secrets are critical, unverified are high
            severity = Severity.CRITICAL if verified else Severity.HIGH

            location = FileLocation(
                file_path=normalize_path(file_path) if file_path else "unknown",
                start_line=line,
            )

            remediation = RemediationInfo(
                description=(
                    f"{'VERIFIED ' if verified else ''}Secret detected.\n"
                    "1. Rotate this credential immediately.\n"
                    "2. Remove from codebase and use secrets manager."
                ),
                effort_estimate="high",
            )

            return ScanFinding(
                title=f"{'Verified ' if verified else ''}Secret: {detector_name}",
                description=(
                    f"{detector_name} credential detected. "
                    f"{'This secret has been verified as active!' if verified else ''}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SECRETS,
                rule_id=detector_name,
                location=location,
                remediation_info=remediation,
                tags={"secret", "trufflehog", detector_name.lower()},
                raw_data=self._redact_raw_data(result),
            )

        except Exception as e:
            logger.warning("Failed to parse TruffleHog result: %s", e)
            return None

    def _redact_raw_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from raw output.

        Args:
            result: Original result.

        Returns:
            Redacted copy.
        """
        redacted = result.copy()
        if "Raw" in redacted:
            raw = redacted["Raw"]
            if len(raw) > 8:
                redacted["Raw"] = raw[:4] + "[REDACTED]" + raw[-4:]
            else:
                redacted["Raw"] = "[REDACTED]"
        return redacted

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "helpUri": "https://github.com/trufflesecurity/trufflehog",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "TruffleHog",
                            "version": self.version or "unknown",
                            "informationUri": "https://github.com/trufflesecurity/trufflehog",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://github.com/trufflesecurity/trufflehog"


# ---------------------------------------------------------------------------
# Detect-Secrets Scanner
# ---------------------------------------------------------------------------


class DetectSecretsScanner(BaseScanner):
    """detect-secrets scanner for pre-commit secret detection.

    detect-secrets is designed for pre-commit hooks and maintains a
    baseline of known secrets to prevent new ones from being committed.

    Example:
        >>> config = ScannerConfig(name="detect-secrets", scanner_type=ScannerType.SECRETS)
        >>> scanner = DetectSecretsScanner(config)
        >>> result = await scanner.scan("/path/to/repo")
    """

    async def scan(self, target_path: str) -> ScanResult:
        """Execute detect-secrets scan.

        Args:
            target_path: Path to scan.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"detect-secrets not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                cwd=target_path,
            )

            findings = self.parse_results(stdout)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_path,
            )

        except Exception as e:
            logger.error("detect-secrets scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build detect-secrets command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "detect-secrets"]
        cmd.extend(["scan"])
        cmd.extend(["--all-files"])

        # Exclude common paths
        for path in self.config.exclude_paths:
            cmd.extend(["--exclude-files", path])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse detect-secrets JSON output.

        Args:
            raw_output: JSON output from detect-secrets.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse detect-secrets output: %s", e)
            return []

        findings = []
        results = data.get("results", {})

        for file_path, secrets in results.items():
            for secret in secrets:
                finding = self._parse_secret(secret, file_path)
                if finding:
                    findings.append(finding)

        logger.info("detect-secrets found %d secrets", len(findings))
        return findings

    def _parse_secret(
        self, secret: Dict[str, Any], file_path: str
    ) -> Optional[ScanFinding]:
        """Parse a single detect-secrets finding.

        Args:
            secret: Secret dictionary.
            file_path: File where secret was found.

        Returns:
            ScanFinding or None.
        """
        try:
            secret_type = secret.get("type", "unknown")
            line_number = secret.get("line_number", 1)
            hashed_secret = secret.get("hashed_secret", "")

            location = FileLocation(
                file_path=normalize_path(file_path),
                start_line=line_number,
            )

            remediation = RemediationInfo(
                description=(
                    f"Potential {secret_type} detected.\n"
                    "1. Verify if this is a real secret.\n"
                    "2. If real, rotate and remove from code.\n"
                    "3. If false positive, add to .secrets.baseline."
                ),
            )

            return ScanFinding(
                title=f"Potential Secret: {secret_type}",
                description=f"Potential {secret_type} detected at line {line_number}",
                severity=Severity.HIGH,
                scanner_name=self.name,
                scanner_type=ScannerType.SECRETS,
                rule_id=secret_type,
                location=location,
                remediation_info=remediation,
                tags={"secret", "detect-secrets", secret_type.lower().replace(" ", "_")},
                raw_data={"type": secret_type, "line_number": line_number},
            )

        except Exception as e:
            logger.warning("Failed to parse detect-secrets finding: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "helpUri": "https://github.com/Yelp/detect-secrets",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "detect-secrets",
                            "version": self.version or "unknown",
                            "informationUri": "https://github.com/Yelp/detect-secrets",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://github.com/Yelp/detect-secrets"
