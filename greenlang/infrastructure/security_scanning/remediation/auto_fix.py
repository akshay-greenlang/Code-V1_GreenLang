# -*- coding: utf-8 -*-
"""
Auto-Fix Generator - SEC-007

Generates automated fixes for security vulnerabilities including:
- Dependency version bumps with GitHub PR creation
- Secret rotation task triggers
- Configuration hardening patches
- License violation resolution

Example:
    >>> config = AutoFixConfig(github_token="ghp_xxx")
    >>> generator = AutoFixGenerator(config)
    >>> fix = await generator.generate_dependency_fix(vulnerability)
    >>> pr_url = await generator.create_github_pr(fix)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Try to import httpx for async HTTP
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FixStatus(str, Enum):
    """Status of a fix."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FixType(str, Enum):
    """Type of fix."""

    DEPENDENCY_UPDATE = "dependency_update"
    SECRET_ROTATION = "secret_rotation"
    CONFIG_HARDENING = "config_hardening"
    LICENSE_RESOLUTION = "license_resolution"
    CODE_FIX = "code_fix"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FixPR:
    """Represents a pull request fix for a vulnerability.

    Attributes:
        fix_id: Unique identifier for this fix.
        fix_type: Type of fix.
        title: PR title.
        description: PR description/body.
        branch_name: Git branch name for the fix.
        base_branch: Base branch to merge into.
        files_changed: Dictionary of file path to new content.
        vulnerability_id: ID of the vulnerability being fixed.
        cve_id: CVE ID if applicable.
        severity: Severity of the vulnerability.
        package_name: Name of the package being updated.
        current_version: Current vulnerable version.
        fixed_version: Version with the fix.
        status: Current status of the fix.
        pr_url: URL of the created PR.
        created_at: When the fix was created.
        metadata: Additional metadata.
    """

    fix_id: str
    fix_type: FixType
    title: str
    description: str
    branch_name: str = ""
    base_branch: str = "main"
    files_changed: Dict[str, str] = field(default_factory=dict)
    vulnerability_id: Optional[str] = None
    cve_id: Optional[str] = None
    severity: str = "MEDIUM"
    package_name: Optional[str] = None
    current_version: Optional[str] = None
    fixed_version: Optional[str] = None
    status: FixStatus = FixStatus.PENDING
    pr_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate branch name if not provided."""
        if not self.branch_name:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            safe_package = re.sub(r"[^a-zA-Z0-9-]", "-", self.package_name or "fix")
            self.branch_name = f"security-fix/{safe_package}-{timestamp}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fix_id": self.fix_id,
            "fix_type": self.fix_type.value,
            "title": self.title,
            "description": self.description,
            "branch_name": self.branch_name,
            "base_branch": self.base_branch,
            "files_changed": list(self.files_changed.keys()),
            "vulnerability_id": self.vulnerability_id,
            "cve_id": self.cve_id,
            "severity": self.severity,
            "package_name": self.package_name,
            "current_version": self.current_version,
            "fixed_version": self.fixed_version,
            "status": self.status.value,
            "pr_url": self.pr_url,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RotationTask:
    """Represents a secret rotation task.

    Attributes:
        task_id: Unique identifier for this task.
        secret_path: Path to the secret in the secrets manager.
        secret_type: Type of secret (api_key, password, certificate, etc.).
        reason: Reason for rotation.
        finding_id: ID of the finding that triggered rotation.
        severity: Severity of the finding.
        status: Current status of the task.
        rotation_method: Method of rotation (manual, automated).
        scheduled_at: When rotation is scheduled.
        completed_at: When rotation was completed.
        metadata: Additional metadata.
    """

    task_id: str
    secret_path: str
    secret_type: str = "unknown"
    reason: str = ""
    finding_id: Optional[str] = None
    severity: str = "HIGH"
    status: FixStatus = FixStatus.PENDING
    rotation_method: str = "automated"
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "secret_path": self.secret_path,
            "secret_type": self.secret_type,
            "reason": self.reason,
            "finding_id": self.finding_id,
            "severity": self.severity,
            "status": self.status.value,
            "rotation_method": self.rotation_method,
            "scheduled_at": self.scheduled_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class ConfigPatch:
    """Represents a configuration hardening patch.

    Attributes:
        patch_id: Unique identifier for this patch.
        config_file: Path to the configuration file.
        description: Description of the hardening change.
        current_value: Current insecure value.
        recommended_value: Recommended secure value.
        finding_id: ID of the finding that triggered this.
        severity: Severity of the finding.
        status: Current status of the patch.
        auto_apply: Whether to auto-apply the patch.
        diff: The patch diff.
        metadata: Additional metadata.
    """

    patch_id: str
    config_file: str
    description: str
    current_value: Optional[str] = None
    recommended_value: Optional[str] = None
    finding_id: Optional[str] = None
    severity: str = "MEDIUM"
    status: FixStatus = FixStatus.PENDING
    auto_apply: bool = False
    diff: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patch_id": self.patch_id,
            "config_file": self.config_file,
            "description": self.description,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "finding_id": self.finding_id,
            "severity": self.severity,
            "status": self.status.value,
            "auto_apply": self.auto_apply,
            "has_diff": bool(self.diff),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AutoFixConfig:
    """Configuration for the auto-fix generator.

    Attributes:
        github_token: GitHub personal access token for API operations.
        github_api_url: GitHub API base URL.
        repo_owner: Repository owner (user or organization).
        repo_name: Repository name.
        default_base_branch: Default base branch for PRs.
        auto_merge_low_risk: Automatically merge low-risk fixes.
        create_draft_prs: Create PRs as drafts.
        add_security_labels: Add security labels to PRs.
        notify_on_fix: Notify when fixes are created.
        secrets_service_url: URL for secrets rotation service.
        max_concurrent_prs: Maximum concurrent fix PRs.
    """

    github_token: Optional[str] = None
    github_api_url: str = "https://api.github.com"
    repo_owner: str = ""
    repo_name: str = ""
    default_base_branch: str = "main"
    auto_merge_low_risk: bool = False
    create_draft_prs: bool = True
    add_security_labels: bool = True
    notify_on_fix: bool = True
    secrets_service_url: Optional[str] = None
    max_concurrent_prs: int = 5


# ---------------------------------------------------------------------------
# Auto-Fix Generator
# ---------------------------------------------------------------------------


class AutoFixGenerator:
    """Generates automated fixes for security vulnerabilities.

    Provides methods to generate dependency updates, secret rotation tasks,
    and configuration hardening patches. Can create GitHub PRs for fixes.

    Attributes:
        config: Auto-fix configuration.
        _fix_counter: Counter for generating unique fix IDs.

    Example:
        >>> config = AutoFixConfig(github_token="ghp_xxx", repo_owner="org", repo_name="repo")
        >>> generator = AutoFixGenerator(config)
        >>> fix = await generator.generate_dependency_fix(vuln)
        >>> pr_url = await generator.create_github_pr(fix)
    """

    def __init__(self, config: Optional[AutoFixConfig] = None) -> None:
        """Initialize the auto-fix generator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AutoFixConfig()
        self._fix_counter = 0

    def _generate_id(self, prefix: str = "FIX") -> str:
        """Generate a unique ID.

        Args:
            prefix: Prefix for the ID.

        Returns:
            Unique ID string.
        """
        self._fix_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix}-{timestamp}-{self._fix_counter:04d}"

    async def generate_dependency_fix(
        self,
        vulnerability: Dict[str, Any],
    ) -> FixPR:
        """Generate a dependency version bump fix for a vulnerability.

        Args:
            vulnerability: Vulnerability data containing package info.

        Returns:
            FixPR object ready for PR creation.

        Example:
            >>> vuln = {
            ...     "id": "VULN-001",
            ...     "cve_id": "CVE-2024-1234",
            ...     "package_name": "requests",
            ...     "current_version": "2.31.0",
            ...     "fixed_version": "2.32.0",
            ...     "severity": "HIGH",
            ...     "file_path": "requirements.txt",
            ... }
            >>> fix = await generator.generate_dependency_fix(vuln)
        """
        package_name = vulnerability.get("package_name", "unknown")
        current_version = vulnerability.get("current_version", "")
        fixed_version = vulnerability.get("fixed_version", "")
        cve_id = vulnerability.get("cve_id", "")
        severity = vulnerability.get("severity", "MEDIUM")
        file_path = vulnerability.get("file_path", "requirements.txt")

        # Generate fix content based on file type
        files_changed = self._generate_dependency_update_files(
            package_name, current_version, fixed_version, file_path
        )

        # Generate PR title and description
        title = f"security: Bump {package_name} from {current_version} to {fixed_version}"
        if cve_id:
            title = f"security({cve_id}): Bump {package_name} to {fixed_version}"

        description = self._generate_dependency_pr_description(
            vulnerability, package_name, current_version, fixed_version
        )

        fix = FixPR(
            fix_id=self._generate_id("DEP"),
            fix_type=FixType.DEPENDENCY_UPDATE,
            title=title,
            description=description,
            files_changed=files_changed,
            vulnerability_id=vulnerability.get("id"),
            cve_id=cve_id,
            severity=severity,
            package_name=package_name,
            current_version=current_version,
            fixed_version=fixed_version,
            base_branch=self.config.default_base_branch,
            metadata={
                "advisory_url": vulnerability.get("advisory_url"),
                "cvss_score": vulnerability.get("cvss_score"),
                "scanner": vulnerability.get("scanner"),
            },
        )

        logger.info(
            "Generated dependency fix: %s -> %s (%s)",
            package_name,
            fixed_version,
            fix.fix_id,
        )

        return fix

    def _generate_dependency_update_files(
        self,
        package_name: str,
        current_version: str,
        fixed_version: str,
        file_path: str,
    ) -> Dict[str, str]:
        """Generate updated dependency file content.

        Args:
            package_name: Name of the package.
            current_version: Current version.
            fixed_version: Fixed version.
            file_path: Path to the dependency file.

        Returns:
            Dictionary of file path to new content.
        """
        files: Dict[str, str] = {}

        # Handle different dependency file types
        if file_path.endswith("requirements.txt"):
            # Simple version pin update pattern
            old_pattern = f"{package_name}=={current_version}"
            new_content = f"{package_name}>={fixed_version}"
            files[file_path] = f"# Updated {package_name} to fix security vulnerability\n{new_content}\n"

        elif file_path.endswith("pyproject.toml"):
            # TOML format update
            files[file_path] = f'# Security update for {package_name}\n"{package_name}>={fixed_version}",\n'

        elif file_path.endswith("package.json"):
            # NPM package.json update
            files[file_path] = json.dumps({
                "dependencies": {
                    package_name: f"^{fixed_version}"
                }
            }, indent=2)

        elif file_path.endswith("Pipfile"):
            files[file_path] = f'{package_name} = ">={fixed_version}"\n'

        else:
            # Generic update
            files[file_path] = f"{package_name}>={fixed_version}\n"

        return files

    def _generate_dependency_pr_description(
        self,
        vulnerability: Dict[str, Any],
        package_name: str,
        current_version: str,
        fixed_version: str,
    ) -> str:
        """Generate PR description for dependency update.

        Args:
            vulnerability: Vulnerability data.
            package_name: Package name.
            current_version: Current version.
            fixed_version: Fixed version.

        Returns:
            Markdown formatted PR description.
        """
        cve_id = vulnerability.get("cve_id", "N/A")
        severity = vulnerability.get("severity", "UNKNOWN")
        cvss = vulnerability.get("cvss_score", "N/A")
        advisory = vulnerability.get("advisory_url", "")
        description = vulnerability.get("description", "No description available.")

        parts = [
            "## Security Update",
            "",
            f"Bumps **{package_name}** from `{current_version}` to `{fixed_version}` to address a security vulnerability.",
            "",
            "### Vulnerability Details",
            "",
            f"- **CVE ID**: {cve_id}",
            f"- **Severity**: {severity}",
            f"- **CVSS Score**: {cvss}",
            "",
            "### Description",
            "",
            description[:500] + ("..." if len(description) > 500 else ""),
            "",
        ]

        if advisory:
            parts.extend([
                "### References",
                "",
                f"- [Security Advisory]({advisory})",
                "",
            ])

        parts.extend([
            "### Testing",
            "",
            "- [ ] Unit tests pass",
            "- [ ] Integration tests pass",
            "- [ ] No breaking changes introduced",
            "",
            "---",
            "",
            "*This PR was automatically generated by GreenLang Security Scanner.*",
        ])

        return "\n".join(parts)

    async def generate_secret_rotation(
        self,
        finding: Dict[str, Any],
    ) -> RotationTask:
        """Generate a secret rotation task for a detected secret.

        Args:
            finding: Secret detection finding data.

        Returns:
            RotationTask ready for execution.

        Example:
            >>> finding = {
            ...     "id": "SECRET-001",
            ...     "secret_type": "api_key",
            ...     "secret_path": "config/api_keys/stripe",
            ...     "severity": "CRITICAL",
            ... }
            >>> task = await generator.generate_secret_rotation(finding)
        """
        secret_path = finding.get("secret_path", finding.get("file_path", "unknown"))
        secret_type = self._detect_secret_type(finding)

        task = RotationTask(
            task_id=self._generate_id("ROT"),
            secret_path=secret_path,
            secret_type=secret_type,
            reason=f"Secret exposed in {finding.get('file_path', 'source code')}",
            finding_id=finding.get("id"),
            severity=finding.get("severity", "HIGH"),
            rotation_method=self._determine_rotation_method(secret_type),
            metadata={
                "detection_rule": finding.get("rule_id"),
                "detector": finding.get("scanner"),
                "commit_sha": finding.get("commit_sha"),
                "file_path": finding.get("file_path"),
            },
        )

        logger.info(
            "Generated secret rotation task: %s (%s)",
            task.secret_path,
            task.task_id,
        )

        return task

    def _detect_secret_type(self, finding: Dict[str, Any]) -> str:
        """Detect the type of secret from finding data.

        Args:
            finding: Secret detection finding.

        Returns:
            Secret type string.
        """
        rule_id = finding.get("rule_id", "").lower()
        description = finding.get("description", "").lower()

        type_patterns = {
            "api_key": ["api_key", "apikey", "api-key"],
            "aws_access_key": ["aws", "access_key"],
            "github_token": ["github", "gh_token", "ghp_"],
            "password": ["password", "passwd", "pwd"],
            "private_key": ["private_key", "rsa", "ssh_key"],
            "certificate": ["certificate", "cert", "pem"],
            "jwt_secret": ["jwt", "secret_key"],
            "database_password": ["db_password", "database"],
            "slack_token": ["slack", "xoxb", "xoxp"],
            "stripe_key": ["stripe", "sk_live", "sk_test"],
        }

        for secret_type, patterns in type_patterns.items():
            if any(p in rule_id or p in description for p in patterns):
                return secret_type

        return "unknown"

    def _determine_rotation_method(self, secret_type: str) -> str:
        """Determine rotation method based on secret type.

        Args:
            secret_type: Type of secret.

        Returns:
            Rotation method (automated or manual).
        """
        automated_types = {
            "api_key",
            "aws_access_key",
            "github_token",
            "database_password",
        }

        return "automated" if secret_type in automated_types else "manual"

    async def generate_config_fix(
        self,
        finding: Dict[str, Any],
    ) -> ConfigPatch:
        """Generate a configuration hardening patch for a misconfiguration.

        Args:
            finding: IaC/configuration finding data.

        Returns:
            ConfigPatch with recommended fix.

        Example:
            >>> finding = {
            ...     "id": "IAC-001",
            ...     "file_path": "terraform/main.tf",
            ...     "rule_id": "AWS002",
            ...     "description": "S3 bucket is not encrypted",
            ...     "remediation": "Enable SSE-S3 encryption",
            ... }
            >>> patch = await generator.generate_config_fix(finding)
        """
        config_file = finding.get("file_path", "")
        rule_id = finding.get("rule_id", "")

        # Get recommended fix based on rule
        recommendation = self._get_config_recommendation(finding)

        patch = ConfigPatch(
            patch_id=self._generate_id("CFG"),
            config_file=config_file,
            description=finding.get("description", "Configuration hardening"),
            current_value=finding.get("current_value"),
            recommended_value=recommendation.get("value"),
            finding_id=finding.get("id"),
            severity=finding.get("severity", "MEDIUM"),
            auto_apply=recommendation.get("auto_apply", False),
            diff=recommendation.get("diff", ""),
            metadata={
                "rule_id": rule_id,
                "scanner": finding.get("scanner"),
                "remediation_guidance": finding.get("remediation"),
                "compliance_controls": recommendation.get("controls", []),
            },
        )

        logger.info(
            "Generated config fix: %s (%s)",
            config_file,
            patch.patch_id,
        )

        return patch

    def _get_config_recommendation(
        self, finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get configuration recommendation based on finding.

        Args:
            finding: Configuration finding.

        Returns:
            Recommendation dictionary with fix details.
        """
        rule_id = finding.get("rule_id", "").upper()

        # Common IaC security recommendations
        recommendations = {
            # AWS/Terraform
            "AWS002": {
                "value": 'encryption { sse_algorithm = "aws:kms" }',
                "diff": "+ encryption {\n+   sse_algorithm = \"aws:kms\"\n+ }",
                "auto_apply": True,
                "controls": ["CC6.7", "Art.32.1.a"],
            },
            "AWS017": {
                "value": "block_public_acls = true",
                "diff": "+ block_public_acls       = true\n+ block_public_policy     = true",
                "auto_apply": True,
                "controls": ["CC6.6"],
            },
            "AWS019": {
                "value": "encrypted = true",
                "diff": "+ encrypted = true",
                "auto_apply": True,
                "controls": ["CC6.7"],
            },
            # Kubernetes
            "KSV001": {
                "value": "runAsNonRoot: true",
                "diff": "+ securityContext:\n+   runAsNonRoot: true",
                "auto_apply": False,
                "controls": ["CC6.1"],
            },
            "KSV012": {
                "value": "readOnlyRootFilesystem: true",
                "diff": "+ securityContext:\n+   readOnlyRootFilesystem: true",
                "auto_apply": False,
                "controls": ["CC6.8"],
            },
        }

        default = {
            "value": finding.get("remediation", "See documentation"),
            "diff": "",
            "auto_apply": False,
            "controls": [],
        }

        return recommendations.get(rule_id, default)

    async def create_github_pr(
        self,
        fix: FixPR,
    ) -> Optional[str]:
        """Create a GitHub pull request for a fix.

        Args:
            fix: The FixPR object containing PR details.

        Returns:
            URL of the created PR, or None if creation failed.
        """
        if not self.config.github_token:
            logger.warning("No GitHub token configured, cannot create PR")
            fix.status = FixStatus.SKIPPED
            return None

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot create PR")
            fix.status = FixStatus.SKIPPED
            return None

        fix.status = FixStatus.IN_PROGRESS

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.config.github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                }

                # Step 1: Get the base branch SHA
                base_sha = await self._get_branch_sha(
                    client, headers, self.config.default_base_branch
                )
                if not base_sha:
                    fix.status = FixStatus.FAILED
                    return None

                # Step 2: Create a new branch
                branch_created = await self._create_branch(
                    client, headers, fix.branch_name, base_sha
                )
                if not branch_created:
                    fix.status = FixStatus.FAILED
                    return None

                # Step 3: Create/update files
                for file_path, content in fix.files_changed.items():
                    await self._create_or_update_file(
                        client, headers, fix.branch_name, file_path, content, fix.title
                    )

                # Step 4: Create the pull request
                pr_url = await self._create_pull_request(
                    client, headers, fix
                )

                if pr_url:
                    fix.pr_url = pr_url
                    fix.status = FixStatus.COMPLETED
                    logger.info("Created PR: %s", pr_url)

                    # Add labels if configured
                    if self.config.add_security_labels:
                        await self._add_labels(client, headers, pr_url)

                    return pr_url
                else:
                    fix.status = FixStatus.FAILED
                    return None

        except Exception as e:
            logger.error("Failed to create GitHub PR: %s", e)
            fix.status = FixStatus.FAILED
            fix.metadata["error"] = str(e)
            return None

    async def _get_branch_sha(
        self,
        client: "httpx.AsyncClient",
        headers: Dict[str, str],
        branch: str,
    ) -> Optional[str]:
        """Get the SHA of a branch.

        Args:
            client: HTTP client.
            headers: Request headers.
            branch: Branch name.

        Returns:
            Branch SHA or None.
        """
        url = f"{self.config.github_api_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/git/ref/heads/{branch}"

        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()["object"]["sha"]
            else:
                logger.error("Failed to get branch SHA: %s", response.text)
                return None
        except Exception as e:
            logger.error("Error getting branch SHA: %s", e)
            return None

    async def _create_branch(
        self,
        client: "httpx.AsyncClient",
        headers: Dict[str, str],
        branch_name: str,
        sha: str,
    ) -> bool:
        """Create a new branch.

        Args:
            client: HTTP client.
            headers: Request headers.
            branch_name: New branch name.
            sha: Base SHA.

        Returns:
            True if successful.
        """
        url = f"{self.config.github_api_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/git/refs"

        try:
            response = await client.post(
                url,
                headers=headers,
                json={
                    "ref": f"refs/heads/{branch_name}",
                    "sha": sha,
                },
            )
            if response.status_code in (200, 201):
                return True
            else:
                logger.error("Failed to create branch: %s", response.text)
                return False
        except Exception as e:
            logger.error("Error creating branch: %s", e)
            return False

    async def _create_or_update_file(
        self,
        client: "httpx.AsyncClient",
        headers: Dict[str, str],
        branch: str,
        file_path: str,
        content: str,
        message: str,
    ) -> bool:
        """Create or update a file in the repository.

        Args:
            client: HTTP client.
            headers: Request headers.
            branch: Branch name.
            file_path: File path in the repository.
            content: File content.
            message: Commit message.

        Returns:
            True if successful.
        """
        import base64

        url = f"{self.config.github_api_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/contents/{file_path}"

        try:
            # Check if file exists
            get_response = await client.get(
                url,
                headers=headers,
                params={"ref": branch},
            )

            payload = {
                "message": message,
                "content": base64.b64encode(content.encode()).decode(),
                "branch": branch,
            }

            if get_response.status_code == 200:
                # File exists, need to include SHA
                payload["sha"] = get_response.json()["sha"]

            response = await client.put(url, headers=headers, json=payload)

            if response.status_code in (200, 201):
                return True
            else:
                logger.error("Failed to update file: %s", response.text)
                return False

        except Exception as e:
            logger.error("Error updating file: %s", e)
            return False

    async def _create_pull_request(
        self,
        client: "httpx.AsyncClient",
        headers: Dict[str, str],
        fix: FixPR,
    ) -> Optional[str]:
        """Create the pull request.

        Args:
            client: HTTP client.
            headers: Request headers.
            fix: Fix PR details.

        Returns:
            PR URL or None.
        """
        url = f"{self.config.github_api_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/pulls"

        try:
            payload = {
                "title": fix.title,
                "body": fix.description,
                "head": fix.branch_name,
                "base": fix.base_branch,
                "draft": self.config.create_draft_prs,
            }

            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 201:
                return response.json()["html_url"]
            else:
                logger.error("Failed to create PR: %s", response.text)
                return None

        except Exception as e:
            logger.error("Error creating PR: %s", e)
            return None

    async def _add_labels(
        self,
        client: "httpx.AsyncClient",
        headers: Dict[str, str],
        pr_url: str,
    ) -> None:
        """Add security labels to the PR.

        Args:
            client: HTTP client.
            headers: Request headers.
            pr_url: PR URL.
        """
        # Extract PR number from URL
        pr_number = pr_url.split("/")[-1]
        url = f"{self.config.github_api_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/issues/{pr_number}/labels"

        try:
            await client.post(
                url,
                headers=headers,
                json={"labels": ["security", "automated", "dependencies"]},
            )
        except Exception as e:
            logger.warning("Failed to add labels: %s", e)

    async def trigger_secret_rotation(
        self,
        task: RotationTask,
    ) -> bool:
        """Trigger secret rotation via the secrets service.

        Args:
            task: Rotation task to execute.

        Returns:
            True if rotation was triggered successfully.
        """
        if not self.config.secrets_service_url:
            logger.warning("No secrets service URL configured")
            task.status = FixStatus.SKIPPED
            return False

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot trigger rotation")
            task.status = FixStatus.SKIPPED
            return False

        task.status = FixStatus.IN_PROGRESS

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.secrets_service_url}/api/v1/secrets/rotate/{task.secret_path}",
                    json={
                        "task_id": task.task_id,
                        "reason": task.reason,
                        "finding_id": task.finding_id,
                        "secret_type": task.secret_type,
                    },
                    timeout=30.0,
                )

                if response.status_code in (200, 202):
                    task.status = FixStatus.COMPLETED
                    task.completed_at = datetime.now(timezone.utc)
                    logger.info(
                        "Triggered secret rotation: %s (%s)",
                        task.secret_path,
                        task.task_id,
                    )
                    return True
                else:
                    task.status = FixStatus.FAILED
                    task.metadata["error"] = response.text
                    logger.error(
                        "Failed to trigger rotation: %s",
                        response.text,
                    )
                    return False

        except Exception as e:
            task.status = FixStatus.FAILED
            task.metadata["error"] = str(e)
            logger.error("Error triggering rotation: %s", e)
            return False
