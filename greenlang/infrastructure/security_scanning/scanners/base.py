# -*- coding: utf-8 -*-
"""
Base Scanner Abstract Class - SEC-007

Provides the abstract base class for all security scanners.
Implements common functionality for subprocess execution, timeout handling,
output parsing, and SARIF generation.

All scanner implementations must inherit from BaseScanner and implement:
    - scan(): Execute the scanner and return results
    - parse_results(): Parse raw output into ScanFinding objects
    - to_sarif(): Convert results to SARIF format

Example:
    >>> class MyScanner(BaseScanner):
    ...     async def scan(self, target_path: str) -> ScanResult:
    ...         output = await self._run_command(["my-scanner", target_path])
    ...         findings = self.parse_results(output)
    ...         return self._create_result(findings)
    ...
    ...     def parse_results(self, raw_output: str) -> List[ScanFinding]:
    ...         # Parse scanner-specific output
    ...         ...

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from greenlang.infrastructure.security_scanning.config import ScannerConfig
    from greenlang.infrastructure.security_scanning.models import (
        ScanFinding,
        ScanResult,
        ScanStatus,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ScannerError(Exception):
    """Base exception for scanner errors."""

    pass


class ScannerExecutionError(ScannerError):
    """Error during scanner execution."""

    def __init__(
        self,
        message: str,
        exit_code: Optional[int] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class ScannerTimeoutError(ScannerError):
    """Scanner execution timed out."""

    def __init__(self, message: str, timeout_seconds: int):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class ScannerNotFoundError(ScannerError):
    """Scanner executable not found."""

    def __init__(self, scanner_name: str, executable: str):
        super().__init__(f"Scanner '{scanner_name}' not found: {executable}")
        self.scanner_name = scanner_name
        self.executable = executable


class ScannerParseError(ScannerError):
    """Error parsing scanner output."""

    pass


# ---------------------------------------------------------------------------
# Base Scanner
# ---------------------------------------------------------------------------


class BaseScanner(ABC):
    """Abstract base class for security scanners.

    Provides common infrastructure for running external scanner tools,
    handling timeouts, parsing output, and generating SARIF reports.

    Subclasses must implement:
        - scan(): Main entry point for running the scanner
        - parse_results(): Convert raw output to ScanFinding objects
        - to_sarif(): Generate SARIF 2.1.0 output

    Attributes:
        config: Scanner configuration.
        name: Scanner name.
        version: Scanner version (detected at runtime).

    Example:
        >>> scanner = BanditScanner(config)
        >>> result = await scanner.scan("/path/to/code")
        >>> print(f"Found {len(result.findings)} issues")
    """

    def __init__(self, config: ScannerConfig) -> None:
        """Initialize the scanner.

        Args:
            config: Scanner configuration.
        """
        self.config = config
        self.name = config.name
        self._version: Optional[str] = None
        self._executable_path: Optional[str] = None

        logger.debug(
            "Initialized %s scanner  executable=%s  timeout=%ds",
            self.name,
            config.executable,
            config.timeout_seconds,
        )

    @property
    def version(self) -> Optional[str]:
        """Get scanner version (lazy-loaded).

        Returns:
            Version string or None if unavailable.
        """
        if self._version is None:
            self._version = self._detect_version()
        return self._version

    @property
    def executable_path(self) -> Optional[str]:
        """Get full path to scanner executable.

        Returns:
            Path to executable or None if not found.
        """
        if self._executable_path is None:
            self._executable_path = shutil.which(self.config.executable or self.name)
        return self._executable_path

    def is_available(self) -> bool:
        """Check if scanner is available on the system.

        Returns:
            True if scanner executable is found.
        """
        return self.executable_path is not None

    @abstractmethod
    async def scan(self, target_path: str) -> ScanResult:
        """Execute the scanner on the target path.

        Args:
            target_path: Path to scan (file, directory, or image).

        Returns:
            ScanResult containing findings and metadata.

        Raises:
            ScannerNotFoundError: If scanner is not installed.
            ScannerExecutionError: If scanner execution fails.
            ScannerTimeoutError: If scanner times out.
        """
        pass

    @abstractmethod
    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse scanner output into ScanFinding objects.

        Args:
            raw_output: Raw output from the scanner.

        Returns:
            List of parsed findings.

        Raises:
            ScannerParseError: If parsing fails.
        """
        pass

    @abstractmethod
    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF 2.1.0 format.

        Args:
            findings: List of findings to convert.

        Returns:
            SARIF-compatible dictionary.
        """
        pass

    async def _run_command(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        capture_stderr: bool = True,
    ) -> Tuple[str, str, int]:
        """Run a command asynchronously with timeout.

        Args:
            command: Command and arguments.
            cwd: Working directory.
            env: Environment variables.
            timeout: Timeout in seconds (uses config default if None).
            capture_stderr: Whether to capture stderr.

        Returns:
            Tuple of (stdout, stderr, exit_code).

        Raises:
            ScannerNotFoundError: If executable not found.
            ScannerTimeoutError: If command times out.
            ScannerExecutionError: If command fails unexpectedly.
        """
        timeout = timeout or self.config.timeout_seconds

        # Merge environment
        effective_env = dict(subprocess.os.environ)
        if self.config.environment:
            effective_env.update(self.config.environment)
        if env:
            effective_env.update(env)

        logger.debug(
            "Running scanner command  cmd=%s  cwd=%s  timeout=%ds",
            " ".join(command),
            cwd,
            timeout,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if capture_stderr else None,
                cwd=cwd,
                env=effective_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ScannerTimeoutError(
                    f"Scanner '{self.name}' timed out after {timeout}s",
                    timeout,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = (
                stderr_bytes.decode("utf-8", errors="replace")
                if stderr_bytes
                else ""
            )
            exit_code = process.returncode or 0

            logger.debug(
                "Scanner command completed  exit_code=%d  stdout_len=%d  stderr_len=%d",
                exit_code,
                len(stdout),
                len(stderr),
            )

            return stdout, stderr, exit_code

        except FileNotFoundError:
            raise ScannerNotFoundError(self.name, command[0])
        except OSError as e:
            raise ScannerExecutionError(
                f"Failed to execute '{self.name}': {e}",
                exit_code=None,
                stderr=str(e),
            )

    def _run_command_sync(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[str, str, int]:
        """Run a command synchronously with timeout.

        Args:
            command: Command and arguments.
            cwd: Working directory.
            env: Environment variables.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (stdout, stderr, exit_code).
        """
        timeout = timeout or self.config.timeout_seconds

        effective_env = dict(subprocess.os.environ)
        if self.config.environment:
            effective_env.update(self.config.environment)
        if env:
            effective_env.update(env)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=effective_env,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired:
            raise ScannerTimeoutError(
                f"Scanner '{self.name}' timed out after {timeout}s",
                timeout,
            )
        except FileNotFoundError:
            raise ScannerNotFoundError(self.name, command[0])

    def _detect_version(self) -> Optional[str]:
        """Detect scanner version.

        Returns:
            Version string or None.
        """
        if not self.is_available():
            return None

        version_flags = ["--version", "-v", "version"]
        for flag in version_flags:
            try:
                stdout, _, exit_code = self._run_command_sync(
                    [self.config.executable or self.name, flag],
                    timeout=10,
                )
                if exit_code == 0 and stdout.strip():
                    # Extract first line, common version format
                    return stdout.strip().split("\n")[0]
            except (ScannerError, Exception):
                continue

        return None

    def _create_result(
        self,
        findings: List[ScanFinding],
        status: ScanStatus,
        started_at: datetime,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        exit_code: Optional[int] = None,
        raw_output: Optional[str] = None,
        command: Optional[str] = None,
        scan_path: Optional[str] = None,
    ) -> ScanResult:
        """Create a ScanResult from findings.

        Args:
            findings: List of findings.
            status: Scan status.
            started_at: Start time.
            completed_at: Completion time (defaults to now).
            error_message: Error message if failed.
            exit_code: Scanner exit code.
            raw_output: Raw scanner output.
            command: Command that was executed.
            scan_path: Path that was scanned.

        Returns:
            Populated ScanResult.
        """
        from greenlang.infrastructure.security_scanning.models import ScanResult

        completed_at = completed_at or datetime.now(timezone.utc)
        duration = (completed_at - started_at).total_seconds()

        return ScanResult(
            scanner_name=self.name,
            scanner_type=self.config.scanner_type,
            status=status,
            findings=findings,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            error_message=error_message,
            exit_code=exit_code,
            raw_output=raw_output,
            command=command,
            scan_path=scan_path,
        )

    def _filter_by_severity(
        self, findings: List[ScanFinding]
    ) -> List[ScanFinding]:
        """Filter findings by severity threshold.

        Args:
            findings: List of findings.

        Returns:
            Filtered list based on config.severity_threshold.
        """
        from greenlang.infrastructure.security_scanning.config import Severity

        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        threshold_idx = severity_order.index(self.config.severity_threshold)

        return [
            f
            for f in findings
            if severity_order.index(f.severity) >= threshold_idx
        ]

    def _filter_by_rules(self, findings: List[ScanFinding]) -> List[ScanFinding]:
        """Filter out excluded rules.

        Args:
            findings: List of findings.

        Returns:
            Filtered list with excluded rules removed.
        """
        if not self.config.exclude_rules:
            return findings

        excluded = set(self.config.exclude_rules)
        return [f for f in findings if f.rule_id not in excluded]

    def _apply_filters(self, findings: List[ScanFinding]) -> List[ScanFinding]:
        """Apply all configured filters to findings.

        Args:
            findings: List of findings.

        Returns:
            Filtered findings.
        """
        filtered = self._filter_by_severity(findings)
        filtered = self._filter_by_rules(filtered)
        return filtered

    def _build_exclude_args(self) -> List[str]:
        """Build exclude path arguments for the scanner.

        Override in subclasses for scanner-specific syntax.

        Returns:
            List of exclude arguments.
        """
        args = []
        for path in self.config.exclude_paths:
            args.extend(["--exclude", path])
        return args

    def _parse_json_output(self, output: str) -> Any:
        """Parse JSON output safely.

        Args:
            output: JSON string.

        Returns:
            Parsed JSON data.

        Raises:
            ScannerParseError: If JSON parsing fails.
        """
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            raise ScannerParseError(f"Failed to parse JSON output: {e}")

    def _create_temp_file(
        self, suffix: str = ".json", content: Optional[str] = None
    ) -> str:
        """Create a temporary file for scanner output.

        Args:
            suffix: File suffix.
            content: Optional initial content.

        Returns:
            Path to temporary file.
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{self.name}_")
        if content:
            with open(path, "w") as f:
                f.write(content)
        else:
            subprocess.os.close(fd)
        return path

    def _get_sarif_tool_component(self) -> Dict[str, Any]:
        """Get SARIF tool component for this scanner.

        Returns:
            SARIF tool driver dictionary.
        """
        return {
            "driver": {
                "name": self.name,
                "version": self.version or "unknown",
                "informationUri": self._get_scanner_url(),
            }
        }

    def _get_scanner_url(self) -> str:
        """Get URL for scanner documentation.

        Override in subclasses for scanner-specific URLs.

        Returns:
            Scanner documentation URL.
        """
        return f"https://github.com/search?q={self.name}"

    def get_scanner_info(self) -> Dict[str, Any]:
        """Get scanner information.

        Returns:
            Dictionary with scanner metadata.
        """
        return {
            "name": self.name,
            "type": self.config.scanner_type.value,
            "version": self.version,
            "available": self.is_available(),
            "enabled": self.config.enabled,
            "executable": self.config.executable,
            "timeout_seconds": self.config.timeout_seconds,
        }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def normalize_path(path: str, base_path: Optional[str] = None) -> str:
    """Normalize a file path relative to base.

    Args:
        path: File path to normalize.
        base_path: Base path to make relative to.

    Returns:
        Normalized relative path.
    """
    p = Path(path)
    if base_path:
        try:
            p = p.relative_to(base_path)
        except ValueError:
            pass
    return str(p).replace("\\", "/")


def parse_sarif_severity(level: str) -> Severity:
    """Parse SARIF level to Severity enum.

    Args:
        level: SARIF level string.

    Returns:
        Corresponding Severity.
    """
    from greenlang.infrastructure.security_scanning.config import Severity

    level_map = {
        "error": Severity.HIGH,
        "warning": Severity.MEDIUM,
        "note": Severity.LOW,
        "none": Severity.INFO,
    }
    return level_map.get(level.lower(), Severity.MEDIUM)


def extract_cve_from_text(text: str) -> Optional[str]:
    """Extract CVE ID from text.

    Args:
        text: Text that may contain a CVE ID.

    Returns:
        CVE ID or None.
    """
    import re

    match = re.search(r"CVE-\d{4}-\d{4,}", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_cwe_from_text(text: str) -> Optional[str]:
    """Extract CWE ID from text.

    Args:
        text: Text that may contain a CWE ID.

    Returns:
        CWE ID or None.
    """
    import re

    match = re.search(r"CWE-\d+", text, re.IGNORECASE)
    return match.group(0).upper() if match else None
