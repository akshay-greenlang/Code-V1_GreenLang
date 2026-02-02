"""
Dimension 08: Security Verification

This dimension verifies that agents follow security best practices
including no secrets in code, input sanitization, and injection prevention.

Checks:
    - No secrets in code
    - Input sanitization
    - Injection prevention
    - Secure configuration

Example:
    >>> dimension = SecurityDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class SecurityDimension(BaseDimension):
    """
    Security Dimension Evaluator (D08).

    Verifies that agents follow security best practices.

    Configuration:
        check_secrets: Check for hardcoded secrets (default: True)
        check_injection: Check for injection vulnerabilities (default: True)
    """

    DIMENSION_ID = "D08"
    DIMENSION_NAME = "Security"
    DESCRIPTION = "Verifies no secrets in code, input sanitization, injection prevention"
    WEIGHT = 1.5
    REQUIRED_FOR_CERTIFICATION = True

    # Patterns that indicate potential secrets
    SECRET_PATTERNS = [
        (r"['\"](?:api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "API key"),
        (r"['\"](?:secret|password|passwd|pwd)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]", "Password/Secret"),
        (r"['\"](?:token|auth[_-]?token)['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_.-]{20,}['\"]", "Auth token"),
        (r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", "Private key"),
        (r"AWS[_A-Z]*\s*[:=]\s*['\"][A-Z0-9]{20}['\"]", "AWS credential"),
        (r"['\"]sk-[a-zA-Z0-9]{48}['\"]", "OpenAI API key"),
        (r"['\"]ghp_[a-zA-Z0-9]{36}['\"]", "GitHub token"),
        (r"(?:jdbc|mysql|postgresql)://[^:]+:[^@]+@", "Database connection string"),
    ]

    # Patterns that indicate potential injection vulnerabilities
    INJECTION_PATTERNS = [
        (r"eval\s*\(", "eval() usage"),
        (r"exec\s*\(", "exec() usage"),
        (r"os\.system\s*\(", "os.system() usage"),
        (r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", "subprocess with shell=True"),
        (r"__import__\s*\(", "dynamic import"),
        (r"pickle\.loads?\s*\(", "pickle deserialization"),
        (r"yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader", "unsafe YAML load"),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security dimension evaluator."""
        super().__init__(config)

        self.check_secrets = self.config.get("check_secrets", True)
        self.check_injection = self.config.get("check_injection", True)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate security for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with security evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting security evaluation")

        # Scan all Python files in agent directory
        python_files = list(agent_path.glob("**/*.py"))
        if not python_files:
            self._add_check(
                name="python_files_exist",
                passed=False,
                message="No Python files found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Combine all source code for analysis
        all_source = ""
        for py_file in python_files:
            try:
                all_source += py_file.read_text(encoding="utf-8") + "\n"
            except Exception as e:
                logger.warning(f"Failed to read {py_file}: {str(e)}")

        # Check 1: No hardcoded secrets
        if self.check_secrets:
            secret_check = self._check_hardcoded_secrets(all_source, python_files)
            self._add_check(
                name="no_hardcoded_secrets",
                passed=secret_check["no_secrets"],
                message="No hardcoded secrets found"
                if secret_check["no_secrets"]
                else f"Found {len(secret_check['secrets_found'])} potential secret(s)",
                severity="error" if not secret_check["no_secrets"] else "info",
                details=secret_check,
            )

        # Check 2: No injection vulnerabilities
        if self.check_injection:
            injection_check = self._check_injection_vulnerabilities(all_source)
            self._add_check(
                name="no_injection_vulnerabilities",
                passed=injection_check["no_vulnerabilities"],
                message="No injection vulnerabilities found"
                if injection_check["no_vulnerabilities"]
                else f"Found {len(injection_check['vulnerabilities'])} potential vulnerability(ies)",
                severity="error" if not injection_check["no_vulnerabilities"] else "info",
                details=injection_check,
            )

        # Check 3: Input validation present
        validation_check = self._check_input_validation(all_source)
        self._add_check(
            name="input_validation",
            passed=validation_check["has_validation"],
            message="Input validation is present"
            if validation_check["has_validation"]
            else "No input validation found",
            severity="warning",
            details=validation_check,
        )

        # Check 4: Secure configuration handling
        config_check = self._check_secure_configuration(all_source, agent_path)
        self._add_check(
            name="secure_configuration",
            passed=config_check["is_secure"],
            message="Configuration handling is secure"
            if config_check["is_secure"]
            else "Insecure configuration patterns found",
            severity="warning",
            details=config_check,
        )

        # Check 5: No .env files committed
        env_check = self._check_env_files(agent_path)
        self._add_check(
            name="no_env_files",
            passed=env_check["no_env_committed"],
            message="No .env files in repository"
            if env_check["no_env_committed"]
            else f"Found {len(env_check['env_files'])} .env file(s)",
            severity="warning" if not env_check["no_env_committed"] else "info",
            details=env_check,
        )

        # Check 6: Logging security
        logging_check = self._check_secure_logging(all_source)
        self._add_check(
            name="secure_logging",
            passed=logging_check["is_secure"],
            message="Logging practices are secure"
            if logging_check["is_secure"]
            else "Potential sensitive data in logs",
            severity="warning",
            details=logging_check,
        )

        # Check 7: Dependency security
        dependency_check = self._check_dependencies(agent_path)
        self._add_check(
            name="dependency_security",
            passed=dependency_check["is_secure"],
            message="Dependencies appear secure"
            if dependency_check["is_secure"]
            else "Dependency security concerns found",
            severity="warning",
            details=dependency_check,
        )

        # Check 8: Error handling security
        error_check = self._check_error_handling(all_source)
        self._add_check(
            name="secure_error_handling",
            passed=error_check["is_secure"],
            message="Error handling is secure"
            if error_check["is_secure"]
            else "Error handling may expose sensitive info",
            severity="warning",
            details=error_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "files_scanned": len(python_files),
                "total_lines": all_source.count("\n"),
            },
        )

    def _check_hardcoded_secrets(
        self,
        source_code: str,
        files: List[Path],
    ) -> Dict[str, Any]:
        """
        Check for hardcoded secrets in source code.

        Args:
            source_code: Combined source code
            files: List of Python files

        Returns:
            Dictionary with secret check results
        """
        result = {
            "no_secrets": True,
            "secrets_found": [],
        }

        for pattern, secret_type in self.SECRET_PATTERNS:
            matches = re.finditer(pattern, source_code, re.IGNORECASE)
            for match in matches:
                result["no_secrets"] = False
                result["secrets_found"].append({
                    "type": secret_type,
                    "pattern": pattern[:50],
                    "preview": match.group()[:30] + "...",
                })

        return result

    def _check_injection_vulnerabilities(self, source_code: str) -> Dict[str, Any]:
        """
        Check for injection vulnerabilities.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with injection check results
        """
        result = {
            "no_vulnerabilities": True,
            "vulnerabilities": [],
        }

        for pattern, vuln_type in self.INJECTION_PATTERNS:
            matches = re.finditer(pattern, source_code)
            for match in matches:
                # Check for safe usage patterns
                context_start = max(0, match.start() - 100)
                context = source_code[context_start:match.start()]

                # Skip if there's indication of safe usage
                if "# nosec" in context or "safe" in context.lower():
                    continue

                result["no_vulnerabilities"] = False
                result["vulnerabilities"].append({
                    "type": vuln_type,
                    "pattern": pattern,
                    "match": match.group(),
                })

        return result

    def _check_input_validation(self, source_code: str) -> Dict[str, Any]:
        """
        Check for input validation.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with validation check results
        """
        result = {
            "has_validation": False,
            "validation_methods": [],
        }

        validation_patterns = [
            (r"@validator", "Pydantic validator"),
            (r"@field_validator", "Pydantic field validator"),
            (r"isinstance\s*\(", "type checking"),
            (r"if\s+not\s+\w+\s*:", "null check"),
            (r"raise\s+ValueError", "ValueError validation"),
            (r"assert\s+", "assertion"),
            (r"Field\s*\(.*(?:ge|le|gt|lt|min_length|max_length)=", "Field constraints"),
        ]

        for pattern, method in validation_patterns:
            if re.search(pattern, source_code):
                result["has_validation"] = True
                result["validation_methods"].append(method)

        return result

    def _check_secure_configuration(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for secure configuration handling.

        Args:
            source_code: Combined source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with configuration check results
        """
        result = {
            "is_secure": True,
            "issues": [],
            "good_practices": [],
        }

        # Check for environment variable usage (good)
        if re.search(r"os\.environ|os\.getenv|environ\.get", source_code):
            result["good_practices"].append("Uses environment variables")

        # Check for hardcoded config in code (bad)
        if re.search(r"(?:host|port|url)\s*[:=]\s*['\"](?:localhost|127\.0\.0\.1|0\.0\.0\.0)", source_code):
            result["issues"].append("Hardcoded host/port")

        # Check for config file usage
        if re.search(r"config\.yaml|config\.json|settings\.py", source_code):
            result["good_practices"].append("Uses configuration file")

        result["is_secure"] = len(result["issues"]) == 0

        return result

    def _check_env_files(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for .env files in repository.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with env file check results
        """
        result = {
            "no_env_committed": True,
            "env_files": [],
        }

        env_patterns = [".env", ".env.local", ".env.production", "secrets.json", "credentials.json"]

        for pattern in env_patterns:
            for env_file in agent_path.glob(f"**/{pattern}"):
                result["no_env_committed"] = False
                result["env_files"].append(str(env_file.relative_to(agent_path)))

        return result

    def _check_secure_logging(self, source_code: str) -> Dict[str, Any]:
        """
        Check for secure logging practices.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with logging security check results
        """
        result = {
            "is_secure": True,
            "issues": [],
        }

        # Check for sensitive data in logs
        sensitive_patterns = [
            (r"logger?\.(?:info|debug|warning|error)\s*\([^)]*(?:password|secret|token|key)[^)]*\)", "Sensitive data in log"),
            (r"print\s*\([^)]*(?:password|secret|token|key)[^)]*\)", "Sensitive data in print"),
        ]

        for pattern, issue in sensitive_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["is_secure"] = False
                result["issues"].append(issue)

        return result

    def _check_dependencies(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check dependency security.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with dependency security check results
        """
        result = {
            "is_secure": True,
            "issues": [],
            "requirements_found": False,
        }

        # Check for requirements.txt
        req_file = agent_path / "requirements.txt"
        if req_file.exists():
            result["requirements_found"] = True
            try:
                content = req_file.read_text(encoding="utf-8")

                # Check for unpinned dependencies
                unpinned = re.findall(r"^([a-zA-Z0-9_-]+)\s*$", content, re.MULTILINE)
                if unpinned:
                    result["issues"].append(f"Unpinned dependencies: {', '.join(unpinned[:3])}")

                # Check for known vulnerable packages (simplified check)
                vulnerable = ["pyyaml<5.4", "requests<2.20"]
                for vuln in vulnerable:
                    if vuln in content.lower():
                        result["is_secure"] = False
                        result["issues"].append(f"Potentially vulnerable: {vuln}")

            except Exception:
                pass

        result["is_secure"] = len(result["issues"]) == 0 or not any("vulnerable" in i.lower() for i in result["issues"])

        return result

    def _check_error_handling(self, source_code: str) -> Dict[str, Any]:
        """
        Check error handling security.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with error handling check results
        """
        result = {
            "is_secure": True,
            "issues": [],
            "good_practices": [],
        }

        # Check for bare except clauses (bad)
        if re.search(r"except\s*:", source_code):
            result["issues"].append("Bare except clause (catches all exceptions)")

        # Check for exception info in responses (potentially bad)
        if re.search(r"str\s*\(\s*e\s*\)|exc_info|traceback", source_code, re.IGNORECASE):
            result["issues"].append("Exception details may be exposed")

        # Check for good error handling
        if re.search(r"except\s+\w+Error", source_code):
            result["good_practices"].append("Specific exception handling")

        if re.search(r"logger\.error.*exc_info=True", source_code):
            result["good_practices"].append("Logs exception info")

        # Secure if no critical issues
        result["is_secure"] = not any("bare except" in i.lower() for i in result["issues"])

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "python_files_exist": (
                "Create Python files for the agent implementation."
            ),
            "no_hardcoded_secrets": (
                "Remove hardcoded secrets:\n"
                "  1. Use environment variables: os.environ.get('API_KEY')\n"
                "  2. Use a secrets manager\n"
                "  3. Use .env files (not committed to git)"
            ),
            "no_injection_vulnerabilities": (
                "Fix injection vulnerabilities:\n"
                "  - Avoid eval(), exec(), os.system()\n"
                "  - Use subprocess with shell=False\n"
                "  - Use yaml.safe_load() instead of yaml.load()\n"
                "  - Avoid pickle for untrusted data"
            ),
            "input_validation": (
                "Add input validation:\n"
                "  - Use Pydantic models with validators\n"
                "  - Add type checking with isinstance()\n"
                "  - Validate ranges and formats"
            ),
            "secure_configuration": (
                "Secure configuration handling:\n"
                "  - Use environment variables for secrets\n"
                "  - Don't hardcode hostnames/ports\n"
                "  - Use configuration files for non-sensitive settings"
            ),
            "no_env_files": (
                "Don't commit .env files:\n"
                "  - Add .env to .gitignore\n"
                "  - Use .env.example for templates\n"
                "  - Use environment variables in production"
            ),
            "secure_logging": (
                "Secure logging practices:\n"
                "  - Don't log passwords, tokens, or keys\n"
                "  - Use log sanitization\n"
                "  - Configure log levels appropriately"
            ),
            "dependency_security": (
                "Secure dependencies:\n"
                "  - Pin all dependencies with versions\n"
                "  - Run 'pip-audit' to check for vulnerabilities\n"
                "  - Keep dependencies updated"
            ),
            "secure_error_handling": (
                "Secure error handling:\n"
                "  - Use specific exception types\n"
                "  - Don't expose stack traces to users\n"
                "  - Log exceptions securely"
            ),
        }

        return remediation_map.get(check.name)
