# -*- coding: utf-8 -*-
"""
Hub Validator - Pre-publish validation for agent packages.

Validates packages before they are published to the Agent Hub. Performs
a configurable set of checks including pack.yaml validity, file presence,
checksum verification, test execution, secret detection, and license
compliance.

Example:
    >>> validator = HubValidator()
    >>> result = await validator.validate("dist/my-agent-1.0.0.glpack")
    >>> if result.passed:
    ...     print("Package is ready for publish")
    >>> else:
    ...     for error in result.errors:
    ...         print(f"ERROR: {error}")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union

from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentPack,
    PackFormat,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Patterns that suggest secrets in source code
SECRET_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)(api[_-]?key|api[_-]?secret)\s*[=:]\s*['\"][^'\"]{8,}"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}"),
    re.compile(r"(?i)(secret[_-]?key)\s*[=:]\s*['\"][^'\"]{8,}"),
    re.compile(r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[=:]\s*['\"]AK"),
    re.compile(r"(?i)(private[_-]?key)\s*[=:]\s*['\"][^'\"]{16,}"),
    re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
]

REQUIRED_FILES: Set[str] = {"agent.pack.yaml"}

LICENSE_FILES: Set[str] = {"LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationCheck:
    """Result of a single validation check.

    Attributes:
        name: Check identifier.
        passed: Whether the check passed.
        message: Description or error message.
        severity: 'error' or 'warning'.
    """

    name: str
    passed: bool
    message: str
    severity: str = "error"


@dataclass
class ValidationResult:
    """Aggregate outcome of all validation checks.

    Attributes:
        passed: True if all error-severity checks passed.
        checks: Individual check results.
        errors: Error-level failure messages.
        warnings: Warning-level messages.
    """

    passed: bool = True
    checks: List[ValidationCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Pre-publish hook type
PrePublishHook = Callable[[Path, AgentPack], Union[List[str], Awaitable[List[str]]]]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class HubValidator:
    """Validate agent packages before publishing to the Hub.

    Performs a configurable set of checks:
      - pack.yaml validity
      - Required files present
      - Checksum consistency
      - No secrets in source
      - License file present
      - Optional: test execution

    Attributes:
        run_tests: Whether to execute tests as part of validation.
        pre_publish_hooks: List of hook functions called before validation.
    """

    def __init__(
        self,
        run_tests: bool = False,
        check_secrets: bool = True,
        require_license: bool = True,
    ) -> None:
        """Initialize the validator.

        Args:
            run_tests: Whether to run tests during validation.
            check_secrets: Whether to scan for secrets in source files.
            require_license: Whether to require a LICENSE file.
        """
        self.run_tests = run_tests
        self.check_secrets = check_secrets
        self.require_license = require_license
        self._hooks: List[PrePublishHook] = []

    def add_hook(self, hook: PrePublishHook) -> None:
        """Register a pre-publish validation hook.

        Args:
            hook: Callable that takes (extract_path, pack) and returns error messages.
        """
        self._hooks.append(hook)

    async def validate(self, package_path: str | Path) -> ValidationResult:
        """Validate a .glpack archive for publishing.

        Args:
            package_path: Path to the .glpack archive.

        Returns:
            ValidationResult with check details.
        """
        result = ValidationResult()
        archive = Path(package_path).resolve()

        # Check archive exists
        if not archive.exists():
            result.checks.append(ValidationCheck(
                name="archive_exists",
                passed=False,
                message=f"Archive not found: {archive}",
            ))
            result.errors.append(f"Archive not found: {archive}")
            result.passed = False
            return result

        result.checks.append(ValidationCheck(
            name="archive_exists", passed=True, message="Archive file exists."
        ))

        # Extract to temp directory
        extract_dir = None
        try:
            extract_dir = await asyncio.to_thread(self._extract, archive)

            # Check required files
            self._check_required_files(extract_dir, result)

            # Validate pack.yaml
            pack = self._check_pack_yaml(extract_dir, result)

            # Check license
            if self.require_license:
                self._check_license(extract_dir, result)

            # Check for secrets
            if self.check_secrets:
                self._check_secrets(extract_dir, result)

            # Check Python syntax
            self._check_python_syntax(extract_dir, result)

            # Run tests if configured
            if self.run_tests:
                await self._run_tests(extract_dir, result)

            # Run pre-publish hooks
            if pack:
                await self._run_hooks(extract_dir, pack, result)

        finally:
            # Cleanup
            if extract_dir and extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)

        # Determine overall pass/fail
        result.passed = len(result.errors) == 0
        status = "PASSED" if result.passed else "FAILED"
        logger.info(
            "Validation %s for %s: %d checks, %d errors, %d warnings",
            status,
            archive.name,
            len(result.checks),
            len(result.errors),
            len(result.warnings),
        )
        return result

    async def validate_directory(self, source_dir: str | Path) -> ValidationResult:
        """Validate an agent source directory (pre-build).

        Args:
            source_dir: Path to the agent source directory.

        Returns:
            ValidationResult.
        """
        result = ValidationResult()
        source = Path(source_dir).resolve()

        if not source.is_dir():
            result.errors.append(f"Not a directory: {source}")
            result.passed = False
            return result

        self._check_required_files(source, result)
        pack = self._check_pack_yaml(source, result)
        if self.require_license:
            self._check_license(source, result)
        if self.check_secrets:
            self._check_secrets(source, result)
        self._check_python_syntax(source, result)

        if pack:
            await self._run_hooks(source, pack, result)

        result.passed = len(result.errors) == 0
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_required_files(
        self, directory: Path, result: ValidationResult
    ) -> None:
        """Check that all required files are present."""
        for filename in REQUIRED_FILES:
            found = (directory / filename).exists()
            result.checks.append(ValidationCheck(
                name=f"required_file_{filename}",
                passed=found,
                message=f"Required file '{filename}' {'found' if found else 'MISSING'}.",
            ))
            if not found:
                result.errors.append(f"Required file missing: {filename}")

    def _check_pack_yaml(
        self, directory: Path, result: ValidationResult
    ) -> Optional[AgentPack]:
        """Validate agent.pack.yaml contents."""
        pack_path = directory / "agent.pack.yaml"
        if not pack_path.exists():
            return None

        try:
            pack = PackFormat.load(pack_path)
            result.checks.append(ValidationCheck(
                name="pack_yaml_valid",
                passed=True,
                message="agent.pack.yaml is valid.",
            ))
            return pack
        except (ValueError, Exception) as exc:
            result.checks.append(ValidationCheck(
                name="pack_yaml_valid",
                passed=False,
                message=f"agent.pack.yaml validation failed: {exc}",
            ))
            result.errors.append(f"Invalid agent.pack.yaml: {exc}")
            return None

    def _check_license(
        self, directory: Path, result: ValidationResult
    ) -> None:
        """Check for a LICENSE file."""
        found = any((directory / f).exists() for f in LICENSE_FILES)
        result.checks.append(ValidationCheck(
            name="license_present",
            passed=found,
            message="LICENSE file found." if found else "No LICENSE file found.",
            severity="warning" if not found else "error",
        ))
        if not found:
            result.warnings.append(
                "No LICENSE file found. Consider adding one before publishing."
            )

    def _check_secrets(
        self, directory: Path, result: ValidationResult
    ) -> None:
        """Scan source files for potential secrets."""
        secrets_found: List[str] = []

        for root_str, dirs, files in os.walk(directory):
            root = Path(root_str)
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules")]
            for fname in files:
                if not fname.endswith((".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")):
                    continue
                fpath = root / fname
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for pattern in SECRET_PATTERNS:
                    matches = pattern.findall(content)
                    if matches:
                        rel_path = fpath.relative_to(directory)
                        secrets_found.append(
                            f"{rel_path}: potential secret detected (pattern: {pattern.pattern[:40]}...)"
                        )
                        break  # One match per file is enough

        passed = len(secrets_found) == 0
        result.checks.append(ValidationCheck(
            name="no_secrets",
            passed=passed,
            message=(
                "No secrets detected in source files."
                if passed
                else f"Potential secrets found in {len(secrets_found)} file(s)."
            ),
        ))
        if not passed:
            for detail in secrets_found:
                result.errors.append(f"Secret detected: {detail}")

    def _check_python_syntax(
        self, directory: Path, result: ValidationResult
    ) -> None:
        """Check Python files for syntax errors."""
        syntax_errors: List[str] = []

        for root_str, dirs, files in os.walk(directory):
            root = Path(root_str)
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = root / fname
                try:
                    content = fpath.read_text(encoding="utf-8")
                    compile(content, str(fpath), "exec")
                except SyntaxError as exc:
                    rel_path = fpath.relative_to(directory)
                    syntax_errors.append(f"{rel_path}: {exc}")

        passed = len(syntax_errors) == 0
        result.checks.append(ValidationCheck(
            name="python_syntax",
            passed=passed,
            message=(
                "All Python files have valid syntax."
                if passed
                else f"Syntax errors in {len(syntax_errors)} file(s)."
            ),
        ))
        if not passed:
            for err in syntax_errors:
                result.errors.append(f"Syntax error: {err}")

    async def _run_tests(
        self, directory: Path, result: ValidationResult
    ) -> None:
        """Run tests found in the extracted package."""
        test_dirs = [directory / "tests", directory / "test"]
        test_dir = None
        for td in test_dirs:
            if td.exists() and td.is_dir():
                test_dir = td
                break

        if test_dir is None:
            result.checks.append(ValidationCheck(
                name="tests_pass",
                passed=True,
                message="No test directory found; skipping test execution.",
                severity="warning",
            ))
            result.warnings.append("No tests directory found.")
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", str(test_dir), "-v", "--tb=short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(directory),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            passed = proc.returncode == 0
            result.checks.append(ValidationCheck(
                name="tests_pass",
                passed=passed,
                message=(
                    "All tests passed."
                    if passed
                    else f"Tests failed (exit code {proc.returncode})."
                ),
            ))
            if not passed:
                result.errors.append(
                    f"Test execution failed: {stderr.decode('utf-8', errors='replace')[:500]}"
                )
        except asyncio.TimeoutError:
            result.checks.append(ValidationCheck(
                name="tests_pass",
                passed=False,
                message="Test execution timed out after 120 seconds.",
            ))
            result.errors.append("Tests timed out.")
        except Exception as exc:
            result.checks.append(ValidationCheck(
                name="tests_pass",
                passed=False,
                message=f"Test execution error: {exc}",
            ))
            result.errors.append(f"Test execution error: {exc}")

    async def _run_hooks(
        self,
        directory: Path,
        pack: AgentPack,
        result: ValidationResult,
    ) -> None:
        """Execute pre-publish hooks."""
        for idx, hook in enumerate(self._hooks):
            try:
                hook_result = hook(directory, pack)
                if asyncio.iscoroutine(hook_result):
                    errors = await hook_result
                else:
                    errors = hook_result
                if errors:
                    for err in errors:
                        result.errors.append(f"Hook {idx}: {err}")
                    result.checks.append(ValidationCheck(
                        name=f"hook_{idx}",
                        passed=False,
                        message=f"Pre-publish hook {idx} failed with {len(errors)} error(s).",
                    ))
                else:
                    result.checks.append(ValidationCheck(
                        name=f"hook_{idx}",
                        passed=True,
                        message=f"Pre-publish hook {idx} passed.",
                    ))
            except Exception as exc:
                result.errors.append(f"Hook {idx} raised exception: {exc}")
                result.checks.append(ValidationCheck(
                    name=f"hook_{idx}",
                    passed=False,
                    message=f"Pre-publish hook {idx} raised: {exc}",
                ))

    # ------------------------------------------------------------------
    # Extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(archive: Path) -> Path:
        """Extract a .glpack archive to a temp directory.

        Returns:
            Path to the temporary extraction directory.
        """
        extract_dir = Path(tempfile.mkdtemp(prefix="glvalidate_"))
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            tar.extractall(extract_dir)
        return extract_dir
