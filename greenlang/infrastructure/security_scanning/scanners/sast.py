# -*- coding: utf-8 -*-
"""
SAST Scanner Implementations - SEC-007

Static Application Security Testing (SAST) scanners for Python codebases.
Implements Bandit, Semgrep, and CodeQL scanner integrations.

Scanners:
    - BanditScanner: Python-specific security linter
    - SemgrepScanner: Multi-language pattern matching
    - CodeQLScanner: Deep semantic analysis

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners.sast import (
    ...     BanditScanner,
    ...     SemgrepScanner,
    ... )
    >>> config = ScannerConfig(name="bandit", scanner_type=ScannerType.SAST)
    >>> scanner = BanditScanner(config)
    >>> result = await scanner.scan("/path/to/code")

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
    ScannerExecutionError,
    ScannerParseError,
    extract_cwe_from_text,
    normalize_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bandit Scanner
# ---------------------------------------------------------------------------


class BanditScanner(BaseScanner):
    """Python security linter using Bandit.

    Bandit finds common security issues in Python code by analyzing
    the AST. Supports SARIF and JSON output formats.

    Severity Mapping:
        - HIGH -> HIGH
        - MEDIUM -> MEDIUM
        - LOW -> LOW

    Example:
        >>> config = ScannerConfig(name="bandit", scanner_type=ScannerType.SAST)
        >>> scanner = BanditScanner(config)
        >>> result = await scanner.scan("/path/to/python/code")
    """

    # Bandit severity to normalized severity
    SEVERITY_MAP: Dict[str, Severity] = {
        "HIGH": Severity.HIGH,
        "MEDIUM": Severity.MEDIUM,
        "LOW": Severity.LOW,
    }

    # Common Bandit test IDs with CWE mappings
    BANDIT_CWE_MAP: Dict[str, str] = {
        "B101": "CWE-703",  # assert_used
        "B102": "CWE-78",   # exec_used
        "B103": "CWE-276",  # set_bad_file_permissions
        "B104": "CWE-200",  # hardcoded_bind_all_interfaces
        "B105": "CWE-798",  # hardcoded_password_string
        "B106": "CWE-798",  # hardcoded_password_funcarg
        "B107": "CWE-798",  # hardcoded_password_default
        "B108": "CWE-377",  # hardcoded_tmp_directory
        "B110": "CWE-703",  # try_except_pass
        "B112": "CWE-703",  # try_except_continue
        "B201": "CWE-94",   # flask_debug_true
        "B301": "CWE-502",  # pickle
        "B302": "CWE-78",   # marshal
        "B303": "CWE-327",  # md5
        "B304": "CWE-327",  # des
        "B305": "CWE-327",  # cipher
        "B306": "CWE-78",   # mktemp_q
        "B307": "CWE-78",   # eval
        "B308": "CWE-611",  # mark_safe
        "B310": "CWE-78",   # urllib_urlopen
        "B311": "CWE-330",  # random
        "B312": "CWE-295",  # telnetlib
        "B313": "CWE-611",  # xml_bad_cElementTree
        "B314": "CWE-611",  # xml_bad_ElementTree
        "B315": "CWE-611",  # xml_bad_expatreader
        "B316": "CWE-611",  # xml_bad_expatbuilder
        "B317": "CWE-611",  # xml_bad_sax
        "B318": "CWE-611",  # xml_bad_minidom
        "B319": "CWE-611",  # xml_bad_pulldom
        "B320": "CWE-611",  # xml_bad_etree
        "B321": "CWE-78",   # ftplib
        "B323": "CWE-295",  # unverified_context
        "B324": "CWE-327",  # hashlib_insecure_functions
        "B401": "CWE-295",  # import_telnetlib
        "B402": "CWE-78",   # import_ftplib
        "B403": "CWE-502",  # import_pickle
        "B404": "CWE-78",   # import_subprocess
        "B405": "CWE-611",  # import_xml_etree
        "B406": "CWE-611",  # import_xml_sax
        "B407": "CWE-611",  # import_xml_expat
        "B408": "CWE-611",  # import_xml_minidom
        "B409": "CWE-611",  # import_xml_pulldom
        "B410": "CWE-611",  # import_lxml
        "B411": "CWE-330",  # import_xmlrpclib
        "B412": "CWE-330",  # import_httpoxy
        "B413": "CWE-327",  # import_pycrypto
        "B501": "CWE-295",  # request_with_no_cert_validation
        "B502": "CWE-327",  # ssl_with_bad_version
        "B503": "CWE-327",  # ssl_with_bad_defaults
        "B504": "CWE-327",  # ssl_with_no_version
        "B505": "CWE-327",  # weak_cryptographic_key
        "B506": "CWE-611",  # yaml_load
        "B507": "CWE-20",   # ssh_no_host_key_verification
        "B508": "CWE-78",   # snmp_insecure_version
        "B509": "CWE-78",   # snmp_weak_cryptography
        "B601": "CWE-78",   # paramiko_calls
        "B602": "CWE-78",   # subprocess_popen_with_shell_equals_true
        "B603": "CWE-78",   # subprocess_without_shell_equals_true
        "B604": "CWE-78",   # any_other_function_with_shell_equals_true
        "B605": "CWE-78",   # start_process_with_a_shell
        "B606": "CWE-78",   # start_process_with_no_shell
        "B607": "CWE-78",   # start_process_with_partial_path
        "B608": "CWE-89",   # hardcoded_sql_expressions
        "B609": "CWE-78",   # linux_commands_wildcard_injection
        "B610": "CWE-89",   # django_extra_used
        "B611": "CWE-89",   # django_rawsql_used
        "B612": "CWE-338",  # logging_config_insecure_listen
        "B701": "CWE-79",   # jinja2_autoescape_false
        "B702": "CWE-79",   # use_of_mako_templates
        "B703": "CWE-79",   # django_mark_safe
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Bandit scanner on target path.

        Args:
            target_path: Path to Python code to scan.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Bandit not found: {self.config.executable}",
            )

        # Build command
        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                cwd=None,
            )

            # Bandit exit codes:
            # 0 = no issues
            # 1 = issues found
            # 2+ = execution error

            if exit_code > 1:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"Bandit failed with exit code {exit_code}",
                    exit_code=exit_code,
                    raw_output=stdout,
                    command=" ".join(command),
                    scan_path=target_path,
                )

            # Parse results
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
            logger.error("Bandit scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Bandit command with arguments.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "bandit"]
        cmd.extend(["-r", target_path])
        cmd.extend(["-f", "json"])

        # Add config file if specified
        if self.config.config_file:
            cmd.extend(["-c", self.config.config_file])

        # Add exclude paths
        if self.config.exclude_paths:
            excludes = ",".join(self.config.exclude_paths)
            cmd.extend(["--exclude", excludes])

        # Add skip rules
        if self.config.exclude_rules:
            skips = ",".join(self.config.exclude_rules)
            cmd.extend(["--skip", skips])

        # Add severity threshold
        if self.config.severity_threshold == Severity.HIGH:
            cmd.extend(["--severity-level", "high"])
        elif self.config.severity_threshold == Severity.MEDIUM:
            cmd.extend(["--severity-level", "medium"])

        # Add extra args
        cmd.extend(self.config.extra_args)

        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Bandit JSON output.

        Args:
            raw_output: JSON output from Bandit.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Bandit output: %s", e)
            return []

        findings = []
        results = data.get("results", [])

        for result in results:
            finding = self._parse_bandit_result(result)
            if finding:
                findings.append(finding)

        logger.info("Bandit found %d issues", len(findings))
        return findings

    def _parse_bandit_result(self, result: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single Bandit result.

        Args:
            result: Bandit result dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            severity_str = result.get("issue_severity", "MEDIUM")
            severity = self.SEVERITY_MAP.get(severity_str, Severity.MEDIUM)

            test_id = result.get("test_id", "")
            test_name = result.get("test_name", "")
            issue_text = result.get("issue_text", "")
            filename = result.get("filename", "")
            line_number = result.get("line_number", 1)
            code = result.get("code", "")

            # Get CWE mapping
            cwe_id = self.BANDIT_CWE_MAP.get(test_id)

            location = FileLocation(
                file_path=normalize_path(filename),
                start_line=line_number,
                snippet=code.strip() if code else None,
            )

            vuln_info = VulnerabilityInfo(
                cwe_id=cwe_id,
                description=issue_text,
            ) if cwe_id else None

            return ScanFinding(
                title=f"[{test_id}] {test_name}",
                description=issue_text,
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SAST,
                rule_id=test_id,
                location=location,
                vulnerability_info=vuln_info,
                tags={"python", "sast", test_id.lower()},
                raw_data=result,
            )

        except Exception as e:
            logger.warning("Failed to parse Bandit result: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format.

        Args:
            findings: List of findings.

        Returns:
            SARIF dictionary.
        """
        rules = {}
        results = []

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            # Build rule if not seen
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.title,
                    "shortDescription": {"text": finding.title},
                    "fullDescription": {"text": finding.description},
                    "helpUri": f"https://bandit.readthedocs.io/en/latest/plugins/{rule_id.lower()}.html",
                    "properties": {
                        "security-severity": str(finding.get_risk_score()),
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
                            "name": "Bandit",
                            "version": self.version or "unknown",
                            "informationUri": "https://bandit.readthedocs.io/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://bandit.readthedocs.io/"


# ---------------------------------------------------------------------------
# Semgrep Scanner
# ---------------------------------------------------------------------------


class SemgrepScanner(BaseScanner):
    """Multi-language SAST using Semgrep.

    Semgrep is a fast, open-source static analysis tool with a large
    rule repository. Supports custom rules in YAML format.

    Example:
        >>> config = ScannerConfig(name="semgrep", scanner_type=ScannerType.SAST)
        >>> scanner = SemgrepScanner(config)
        >>> result = await scanner.scan("/path/to/code")
    """

    # Semgrep severity mapping
    SEVERITY_MAP: Dict[str, Severity] = {
        "ERROR": Severity.HIGH,
        "WARNING": Severity.MEDIUM,
        "INFO": Severity.LOW,
    }

    async def scan(self, target_path: str) -> ScanResult:
        """Execute Semgrep scanner.

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
                error_message=f"Semgrep not found: {self.config.executable}",
            )

        command = self._build_command(target_path)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Semgrep exit codes:
            # 0 = success, no findings
            # 1 = success, findings present
            # 2+ = error

            if exit_code > 1:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"Semgrep failed with exit code {exit_code}",
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
            logger.error("Semgrep scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    def _build_command(self, target_path: str) -> List[str]:
        """Build Semgrep command.

        Args:
            target_path: Path to scan.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "semgrep"]
        cmd.extend(["scan", target_path])
        cmd.extend(["--json"])

        # Use auto config by default
        if not any("--config" in arg for arg in self.config.extra_args):
            cmd.extend(["--config", "auto"])

        # Add exclude paths
        for path in self.config.exclude_paths:
            cmd.extend(["--exclude", path])

        # Add extra args
        cmd.extend(self.config.extra_args)

        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse Semgrep JSON output.

        Args:
            raw_output: JSON output from Semgrep.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Semgrep output: %s", e)
            return []

        findings = []
        results = data.get("results", [])

        for result in results:
            finding = self._parse_semgrep_result(result)
            if finding:
                findings.append(finding)

        logger.info("Semgrep found %d issues", len(findings))
        return findings

    def _parse_semgrep_result(self, result: Dict[str, Any]) -> Optional[ScanFinding]:
        """Parse a single Semgrep result.

        Args:
            result: Semgrep result dictionary.

        Returns:
            ScanFinding or None.
        """
        try:
            check_id = result.get("check_id", "")
            extra = result.get("extra", {})
            message = extra.get("message", "")
            severity_str = extra.get("severity", "WARNING")
            severity = self.SEVERITY_MAP.get(severity_str.upper(), Severity.MEDIUM)

            path = result.get("path", "")
            start = result.get("start", {})
            end = result.get("end", {})

            location = FileLocation(
                file_path=normalize_path(path),
                start_line=start.get("line", 1),
                end_line=end.get("line", start.get("line", 1)),
                start_column=start.get("col"),
                end_column=end.get("col"),
            )

            # Extract CWE if present in metadata
            metadata = extra.get("metadata", {})
            cwe_id = None
            cwe_list = metadata.get("cwe", [])
            if cwe_list:
                cwe_id = cwe_list[0] if isinstance(cwe_list, list) else cwe_list

            vuln_info = VulnerabilityInfo(
                cwe_id=cwe_id,
                description=message,
                references=metadata.get("references", []),
            )

            return ScanFinding(
                title=check_id,
                description=message,
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SAST,
                rule_id=check_id,
                location=location,
                vulnerability_info=vuln_info,
                tags=set(metadata.get("tags", [])) | {"semgrep", "sast"},
                raw_data=result,
            )

        except Exception as e:
            logger.warning("Failed to parse Semgrep result: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format.

        Args:
            findings: List of findings.

        Returns:
            SARIF dictionary.
        """
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
                    "helpUri": "https://semgrep.dev/r",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Semgrep",
                            "version": self.version or "unknown",
                            "informationUri": "https://semgrep.dev/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://semgrep.dev/"


# ---------------------------------------------------------------------------
# CodeQL Scanner
# ---------------------------------------------------------------------------


class CodeQLScanner(BaseScanner):
    """Deep semantic analysis using GitHub CodeQL.

    CodeQL requires a pre-built database. This scanner assumes the
    database exists or runs the full database creation + analysis workflow.

    Note: CodeQL is typically run via GitHub Actions rather than locally.

    Example:
        >>> config = ScannerConfig(name="codeql", scanner_type=ScannerType.SAST)
        >>> scanner = CodeQLScanner(config)
        >>> result = await scanner.scan("/path/to/codeql-db")
    """

    async def scan(self, target_path: str) -> ScanResult:
        """Execute CodeQL analysis.

        Args:
            target_path: Path to CodeQL database or source code.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"CodeQL not found: {self.config.executable}",
            )

        # Check if target is a database or source
        import os

        is_database = os.path.isdir(target_path) and os.path.exists(
            os.path.join(target_path, "codeql-database.yml")
        )

        if not is_database:
            # Create database first
            db_result = await self._create_database(target_path)
            if db_result.status != ScanStatus.COMPLETED:
                return db_result
            database_path = os.path.join(target_path, ".codeql-db")
        else:
            database_path = target_path

        # Run analysis
        command = self._build_analyze_command(database_path)

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                timeout=self.config.timeout_seconds,
            )

            if exit_code != 0:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"CodeQL failed with exit code {exit_code}",
                    exit_code=exit_code,
                    command=" ".join(command),
                    scan_path=target_path,
                )

            # Parse SARIF output
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
            logger.error("CodeQL scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_path,
            )

    async def _create_database(self, source_path: str) -> ScanResult:
        """Create CodeQL database from source.

        Args:
            source_path: Path to source code.

        Returns:
            ScanResult indicating success/failure.
        """
        import os

        started_at = datetime.now(timezone.utc)
        db_path = os.path.join(source_path, ".codeql-db")

        command = [
            self.config.executable or "codeql",
            "database",
            "create",
            db_path,
            "--language=python",
            f"--source-root={source_path}",
            "--overwrite",
        ]

        try:
            stdout, stderr, exit_code = await self._run_command(
                command,
                timeout=1800,  # 30 minutes for database creation
            )

            if exit_code != 0:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=f"Database creation failed: {stderr}",
                    exit_code=exit_code,
                    scan_path=source_path,
                )

            return self._create_result(
                findings=[],
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                scan_path=source_path,
            )

        except Exception as e:
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=source_path,
            )

    def _build_analyze_command(self, database_path: str) -> List[str]:
        """Build CodeQL analyze command.

        Args:
            database_path: Path to CodeQL database.

        Returns:
            Command list.
        """
        cmd = [
            self.config.executable or "codeql",
            "database",
            "analyze",
            database_path,
            "--format=sarif-latest",
            "--output=-",  # stdout
        ]

        # Add query suites
        cmd.append("python-security-and-quality.qls")

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse CodeQL SARIF output.

        Args:
            raw_output: SARIF JSON output.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            sarif = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse CodeQL SARIF: %s", e)
            return []

        findings = []

        for run in sarif.get("runs", []):
            rules = {r["id"]: r for r in run.get("tool", {}).get("driver", {}).get("rules", [])}

            for result in run.get("results", []):
                finding = self._parse_sarif_result(result, rules)
                if finding:
                    findings.append(finding)

        logger.info("CodeQL found %d issues", len(findings))
        return findings

    def _parse_sarif_result(
        self, result: Dict[str, Any], rules: Dict[str, Any]
    ) -> Optional[ScanFinding]:
        """Parse a SARIF result into ScanFinding.

        Args:
            result: SARIF result object.
            rules: Rule definitions.

        Returns:
            ScanFinding or None.
        """
        try:
            rule_id = result.get("ruleId", "")
            rule = rules.get(rule_id, {})

            level = result.get("level", "warning")
            severity_map = {
                "error": Severity.HIGH,
                "warning": Severity.MEDIUM,
                "note": Severity.LOW,
                "none": Severity.INFO,
            }
            severity = severity_map.get(level, Severity.MEDIUM)

            message = result.get("message", {}).get("text", "")

            # Parse location
            locations = result.get("locations", [])
            location = None
            if locations:
                loc = locations[0].get("physicalLocation", {})
                artifact = loc.get("artifactLocation", {})
                region = loc.get("region", {})

                location = FileLocation(
                    file_path=artifact.get("uri", ""),
                    start_line=region.get("startLine", 1),
                    end_line=region.get("endLine"),
                    start_column=region.get("startColumn"),
                    end_column=region.get("endColumn"),
                    snippet=region.get("snippet", {}).get("text"),
                )

            # Extract CWE from rule properties
            cwe_id = None
            tags = rule.get("properties", {}).get("tags", [])
            for tag in tags:
                if tag.startswith("external/cwe/"):
                    cwe_id = tag.replace("external/cwe/", "").upper()
                    break

            vuln_info = VulnerabilityInfo(
                cwe_id=cwe_id,
                description=rule.get("fullDescription", {}).get("text", message),
            )

            return ScanFinding(
                title=rule.get("shortDescription", {}).get("text", rule_id),
                description=message,
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.SAST,
                rule_id=rule_id,
                location=location,
                vulnerability_info=vuln_info,
                tags={"codeql", "sast"} | set(tags),
                raw_data=result,
            )

        except Exception as e:
            logger.warning("Failed to parse CodeQL result: %s", e)
            return None

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format.

        Since CodeQL already outputs SARIF, this reconstructs it.

        Args:
            findings: List of findings.

        Returns:
            SARIF dictionary.
        """
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
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "CodeQL",
                            "version": self.version or "unknown",
                            "informationUri": "https://codeql.github.com/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://codeql.github.com/"
