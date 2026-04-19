#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Dependency Vulnerability Scanner

Scans all requirements.txt files for:
- Known CVEs (using safety and pip-audit)
- Outdated packages with security patches
- Unmaintained dependencies
- License compliance issues
- Typosquatting detection via PyPI API
- Security advisories via GitHub API

Usage:
    python scan_dependencies.py
    python scan_dependencies.py --auto-fix
    python scan_dependencies.py --report-only

Author: Security & Compliance Audit Team
Date: 2025-11-09
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import re
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class Vulnerability:
    """Vulnerability information"""
    package: str
    installed_version: str
    vulnerable_spec: str
    cve: Optional[str]
    severity: Severity
    description: str
    fixed_version: Optional[str]
    source: str  # safety, pip-audit, etc.


@dataclass
class LicenseIssue:
    """License compliance issue"""
    package: str
    version: str
    license: str
    issue: str
    severity: Severity


@dataclass
class TyposquattingIssue:
    """Potential typosquatting attack detection"""
    package: str
    similar_to: str
    similarity_score: float
    pypi_exists: bool
    issue: str
    severity: Severity


@dataclass
class GitHubSecurityAdvisory:
    """GitHub security advisory information"""
    package: str
    ghsa_id: str
    cve_id: Optional[str]
    severity: Severity
    summary: str
    description: str
    vulnerable_versions: str
    patched_versions: Optional[str]
    published_at: str
    updated_at: str
    url: str


@dataclass
class ScanResult:
    """Scan result"""
    timestamp: str
    total_packages: int
    vulnerabilities: List[Vulnerability]
    license_issues: List[LicenseIssue]
    outdated_packages: List[Dict[str, str]]
    unmaintained_packages: List[Dict[str, str]]
    typosquatting_issues: List[TyposquattingIssue] = field(default_factory=list)
    github_advisories: List[GitHubSecurityAdvisory] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)


class GitHubAPIClient:
    """
    GitHub API client for security-related queries.

    Handles:
    - Repository existence checks
    - Security advisory fetching
    - Rate limiting with exponential backoff
    - Authentication via token
    """

    # GitHub API base URLs
    REST_API_URL = "https://api.github.com"
    GRAPHQL_API_URL = "https://api.github.com/graphql"

    # Rate limiting settings
    RATE_LIMIT_WAIT_SECONDS = 60
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub API client.

        Args:
            token: GitHub personal access token (or from GITHUB_TOKEN env var)
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self._rate_limit_remaining = None
        self._rate_limit_reset = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "GreenLang-Security-Scanner/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _handle_rate_limit(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        if "X-RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in headers:
            self._rate_limit_reset = int(headers["X-RateLimit-Reset"])

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exhausted."""
        if self._rate_limit_remaining is not None and self._rate_limit_remaining <= 0:
            if self._rate_limit_reset:
                wait_time = max(0, self._rate_limit_reset - time.time())
                if wait_time > 0:
                    logger.warning(
                        f"GitHub API rate limit exceeded. "
                        f"Waiting {wait_time:.0f} seconds..."
                    )
                    time.sleep(min(wait_time, self.RATE_LIMIT_WAIT_SECONDS))

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[bytes] = None,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to GitHub API with retry and rate limiting.

        Args:
            url: API endpoint URL
            method: HTTP method
            data: Request body for POST requests
            retry_count: Current retry attempt

        Returns:
            JSON response data or None on failure
        """
        self._wait_for_rate_limit()

        try:
            request = urllib.request.Request(
                url,
                headers=self._get_headers(),
                method=method,
                data=data
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                # Update rate limit tracking
                self._handle_rate_limit(dict(response.headers))

                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
                return None

        except urllib.error.HTTPError as e:
            if e.code == 403 and "rate limit" in str(e.reason).lower():
                # Rate limit hit - wait and retry
                if retry_count < self.MAX_RETRIES:
                    wait_time = self.BACKOFF_FACTOR ** retry_count * 10
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    return self._make_request(url, method, data, retry_count + 1)
                logger.error("GitHub API rate limit exceeded after retries")
                return None
            elif e.code == 404:
                logger.debug(f"GitHub resource not found: {url}")
                return None
            elif e.code == 401:
                logger.warning("GitHub API authentication failed - using unauthenticated access")
                return None
            else:
                logger.error(f"GitHub API error: {e.code} {e.reason}")
                return None

        except urllib.error.URLError as e:
            logger.error(f"GitHub API connection error: {e.reason}")
            return None

        except Exception as e:
            logger.error(f"Unexpected GitHub API error: {e}")
            return None

    def check_repository_exists(self, owner: str, repo: str) -> bool:
        """
        Check if a GitHub repository exists.

        Args:
            owner: Repository owner/organization
            repo: Repository name

        Returns:
            True if repository exists, False otherwise
        """
        url = f"{self.REST_API_URL}/repos/{owner}/{repo}"
        result = self._make_request(url)
        return result is not None

    def get_security_advisories(
        self,
        package_name: str,
        ecosystem: str = "pip"
    ) -> List[Dict[str, Any]]:
        """
        Fetch security advisories for a package from GitHub Advisory Database.

        Args:
            package_name: Name of the package
            ecosystem: Package ecosystem (pip, npm, etc.)

        Returns:
            List of security advisories
        """
        # Use the GitHub Security Advisory API
        url = (
            f"{self.REST_API_URL}/advisories"
            f"?ecosystem={ecosystem}&affects={package_name}"
        )

        result = self._make_request(url)
        if result is None:
            return []

        return result if isinstance(result, list) else []

    def get_package_advisories_graphql(
        self,
        package_names: List[str],
        ecosystem: str = "PIP"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch security advisories for multiple packages using GraphQL.

        This is more efficient for bulk queries.

        Args:
            package_names: List of package names
            ecosystem: Package ecosystem (PIP, NPM, etc.)

        Returns:
            Dictionary mapping package names to their advisories
        """
        if not self.token:
            logger.warning(
                "GraphQL API requires authentication. "
                "Set GITHUB_TOKEN environment variable."
            )
            return {}

        # Build GraphQL query for security vulnerabilities
        query = """
        query($ecosystem: SecurityAdvisoryEcosystem!, $package: String!) {
            securityVulnerabilities(
                ecosystem: $ecosystem,
                package: $package,
                first: 100
            ) {
                nodes {
                    advisory {
                        ghsaId
                        summary
                        description
                        severity
                        publishedAt
                        updatedAt
                        permalink
                        identifiers {
                            type
                            value
                        }
                    }
                    package {
                        name
                    }
                    vulnerableVersionRange
                    firstPatchedVersion {
                        identifier
                    }
                }
            }
        }
        """

        advisories_by_package: Dict[str, List[Dict[str, Any]]] = {}

        for package_name in package_names:
            variables = {
                "ecosystem": ecosystem,
                "package": package_name
            }

            data = json.dumps({
                "query": query,
                "variables": variables
            }).encode("utf-8")

            result = self._make_request(
                self.GRAPHQL_API_URL,
                method="POST",
                data=data
            )

            if result and "data" in result:
                vulns = result["data"].get("securityVulnerabilities", {})
                nodes = vulns.get("nodes", [])
                if nodes:
                    advisories_by_package[package_name] = nodes

            # Small delay between requests to be nice to the API
            time.sleep(0.1)

        return advisories_by_package

    def verify_package_authenticity(
        self,
        package_name: str,
        repository_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify package authenticity by checking GitHub repository.

        Checks:
        - Repository exists
        - Has security policy
        - Has verified releases
        - Is not archived

        Args:
            package_name: Name of the package
            repository_url: Optional direct repository URL

        Returns:
            Dictionary with authenticity verification results
        """
        result = {
            "package": package_name,
            "repository_exists": False,
            "has_security_policy": False,
            "is_archived": False,
            "is_fork": False,
            "stars": 0,
            "last_updated": None,
            "authenticity_score": 0,
            "warnings": []
        }

        # Parse repository URL if provided
        if repository_url:
            match = re.match(
                r"https?://github\.com/([^/]+)/([^/]+)",
                repository_url
            )
            if match:
                owner, repo = match.groups()
                repo = repo.replace(".git", "")
            else:
                result["warnings"].append("Invalid GitHub repository URL")
                return result
        else:
            # Try to guess repository from package name
            owner = package_name
            repo = package_name

        # Check repository
        repo_url = f"{self.REST_API_URL}/repos/{owner}/{repo}"
        repo_data = self._make_request(repo_url)

        if repo_data:
            result["repository_exists"] = True
            result["is_archived"] = repo_data.get("archived", False)
            result["is_fork"] = repo_data.get("fork", False)
            result["stars"] = repo_data.get("stargazers_count", 0)
            result["last_updated"] = repo_data.get("updated_at")

            if result["is_archived"]:
                result["warnings"].append("Repository is archived")
            if result["is_fork"]:
                result["warnings"].append("Repository is a fork")

            # Check for security policy
            security_url = f"{repo_url}/community/profile"
            security_data = self._make_request(security_url)
            if security_data:
                files = security_data.get("files", {})
                if files.get("security"):
                    result["has_security_policy"] = True

            # Calculate authenticity score (0-100)
            score = 0
            if result["repository_exists"]:
                score += 30
            if result["has_security_policy"]:
                score += 20
            if not result["is_archived"]:
                score += 15
            if not result["is_fork"]:
                score += 10
            if result["stars"] > 100:
                score += 15
            elif result["stars"] > 10:
                score += 10
            if result["last_updated"]:
                # Check if updated in last year
                try:
                    updated = datetime.fromisoformat(
                        result["last_updated"].replace("Z", "+00:00")
                    )
                    if (datetime.now(updated.tzinfo) - updated).days < 365:
                        score += 10
                except Exception:
                    pass

            result["authenticity_score"] = score

        return result


class DependencyScanner:
    """
    Scan dependencies for security issues.

    Features:
    - CVE scanning via safety and pip-audit
    - PyPI API integration for package verification
    - Typosquatting detection using similarity matching
    - GitHub Advisory Database integration
    - License compliance checking
    - Unmaintained package detection
    """

    # Well-known popular packages for typosquatting comparison
    POPULAR_PACKAGES = {
        "requests", "numpy", "pandas", "django", "flask", "boto3",
        "pillow", "urllib3", "cryptography", "pyyaml", "jinja2",
        "sqlalchemy", "celery", "redis", "psycopg2", "pytest",
        "tensorflow", "torch", "scikit-learn", "matplotlib", "scipy",
        "beautifulsoup4", "selenium", "scrapy", "httpx", "aiohttp",
        "pydantic", "fastapi", "uvicorn", "gunicorn", "asyncio",
        "click", "typer", "rich", "tqdm", "black", "flake8", "mypy"
    }

    # Similarity threshold for typosquatting detection (0.0 - 1.0)
    TYPOSQUATTING_THRESHOLD = 0.85

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.requirements_files = []
        self.all_packages: Set[str] = set()
        self.github_client = GitHubAPIClient()

    def find_requirements_files(self) -> List[Path]:
        """Find all requirements*.txt files"""
        files = []

        # Find all requirements files
        for pattern in ["requirements*.txt", "**/requirements*.txt"]:
            files.extend(self.root_dir.glob(pattern))

        self.requirements_files = files
        logger.info(f"Found {len(files)} requirements files")

        return files

    async def scan_with_safety(self) -> List[Vulnerability]:
        """Scan with safety (PyUp vulnerability database)"""
        logger.info("Scanning with safety...")
        vulnerabilities = []

        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json", "--output", "json"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode in [0, 64]:  # 64 = vulnerabilities found
                try:
                    data = json.loads(result.stdout)

                    for vuln in data:
                        vulnerabilities.append(Vulnerability(
                            package=vuln.get("package"),
                            installed_version=vuln.get("installed_version"),
                            vulnerable_spec=vuln.get("vulnerable_spec"),
                            cve=vuln.get("cve"),
                            severity=self._map_severity(vuln.get("severity", "medium")),
                            description=vuln.get("advisory"),
                            fixed_version=vuln.get("fixed_in"),
                            source="safety"
                        ))

                except json.JSONDecodeError:
                    logger.warning("Failed to parse safety output")

        except FileNotFoundError:
            logger.warning("safety not installed. Install with: pip install safety")

        logger.info(f"Safety found {len(vulnerabilities)} vulnerabilities")
        return vulnerabilities

    async def scan_with_pip_audit(self) -> List[Vulnerability]:
        """Scan with pip-audit (OSV database)"""
        logger.info("Scanning with pip-audit...")
        vulnerabilities = []

        try:
            # Run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--local"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode in [0, 1]:  # 1 = vulnerabilities found
                try:
                    data = json.loads(result.stdout)

                    for vuln in data.get("vulnerabilities", []):
                        vulnerabilities.append(Vulnerability(
                            package=vuln.get("name"),
                            installed_version=vuln.get("version"),
                            vulnerable_spec=vuln.get("vuln_spec"),
                            cve=vuln.get("id"),
                            severity=self._map_severity(
                                vuln.get("severity", {}).get("level", "medium")
                            ),
                            description=vuln.get("description", ""),
                            fixed_version=vuln.get("fix_versions", [None])[0],
                            source="pip-audit"
                        ))

                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-audit output")

        except FileNotFoundError:
            logger.warning("pip-audit not installed. Install with: pip install pip-audit")

        logger.info(f"pip-audit found {len(vulnerabilities)} vulnerabilities")
        return vulnerabilities

    async def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages"""
        logger.info("Checking for outdated packages...")
        outdated = []

        try:
            # Run pip list --outdated
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)

                    for pkg in data:
                        # Check if security release
                        is_security = self._is_security_release(
                            pkg["name"],
                            pkg["version"],
                            pkg["latest_version"]
                        )

                        if is_security:
                            outdated.append({
                                "package": pkg["name"],
                                "current": pkg["version"],
                                "latest": pkg["latest_version"],
                                "type": pkg.get("latest_filetype", "unknown"),
                                "is_security_release": True
                            })

                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip list output")

        except FileNotFoundError:
            logger.error("pip not found")

        logger.info(f"Found {len(outdated)} outdated packages with security updates")
        return outdated

    async def check_licenses(self) -> List[LicenseIssue]:
        """Check for license compliance issues"""
        logger.info("Checking license compliance...")
        issues = []

        # Forbidden licenses (copyleft, restrictive)
        FORBIDDEN_LICENSES = [
            "GPL",
            "AGPL",
            "LGPL",
            "SSPL",
            "Commercial",
            "Proprietary"
        ]

        # Warning licenses (require review)
        WARNING_LICENSES = [
            "MPL",
            "EPL",
            "CPL"
        ]

        try:
            # Run pip-licenses
            result = subprocess.run(
                ["pip-licenses", "--format", "json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)

                    for pkg in data:
                        license_name = pkg.get("License", "Unknown")

                        # Check forbidden
                        if any(forbidden in license_name for forbidden in FORBIDDEN_LICENSES):
                            issues.append(LicenseIssue(
                                package=pkg["Name"],
                                version=pkg["Version"],
                                license=license_name,
                                issue=f"Forbidden license: {license_name}",
                                severity=Severity.HIGH
                            ))

                        # Check warning
                        elif any(warning in license_name for warning in WARNING_LICENSES):
                            issues.append(LicenseIssue(
                                package=pkg["Name"],
                                version=pkg["Version"],
                                license=license_name,
                                issue=f"License requires review: {license_name}",
                                severity=Severity.MEDIUM
                            ))

                        # Unknown license
                        elif license_name in ["UNKNOWN", "Unknown", ""]:
                            issues.append(LicenseIssue(
                                package=pkg["Name"],
                                version=pkg["Version"],
                                license="Unknown",
                                issue="License not specified",
                                severity=Severity.LOW
                            ))

                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-licenses output")

        except FileNotFoundError:
            logger.warning("pip-licenses not installed. Install with: pip install pip-licenses")

        logger.info(f"Found {len(issues)} license issues")
        return issues

    async def check_unmaintained(self) -> List[Dict[str, str]]:
        """Check for unmaintained packages"""
        logger.info("Checking for unmaintained packages...")
        unmaintained = []

        # Known unmaintained packages (update this list)
        UNMAINTAINED_PACKAGES = [
            "pycrypto",  # Replaced by pycryptodome
            "pyyaml",    # Check version < 5.4 (has vulns)
        ]

        try:
            result = subprocess.run(
                ["pip", "list", "--format", "json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                for pkg in data:
                    if pkg["name"].lower() in UNMAINTAINED_PACKAGES:
                        unmaintained.append({
                            "package": pkg["name"],
                            "version": pkg["version"],
                            "reason": "Package is no longer maintained",
                            "recommendation": self._get_replacement(pkg["name"])
                        })

        except Exception as e:
            logger.error(f"Failed to check unmaintained packages: {e}")

        return unmaintained

    def _map_severity(self, severity_str: str) -> Severity:
        """Map severity string to enum"""
        severity_map = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW,
            "informational": Severity.INFORMATIONAL
        }

        return severity_map.get(severity_str.lower(), Severity.MEDIUM)

    def _is_security_release(self, package: str, current: str, latest: str) -> bool:
        """
        Check if latest version is a security release by querying PyPI API.

        Queries the PyPI JSON API to check release notes and changelog for
        security-related keywords. Uses rate limiting to avoid API throttling.

        Args:
            package: Package name
            current: Currently installed version
            latest: Latest available version

        Returns:
            True if the latest version appears to be a security release
        """
        pypi_data = self._check_pypi_package(package, latest)
        if pypi_data is None:
            # If we cannot fetch data, assume it could be a security release
            # to err on the side of caution
            return True

        return pypi_data.get("is_security_release", False)

    def _check_pypi_package(self, package_name: str, version: str = None) -> Optional[Dict[str, Any]]:
        """
        Query PyPI JSON API to check package information and detect security releases.

        Args:
            package_name: Name of the package to check
            version: Specific version to check (optional, defaults to latest)

        Returns:
            Dictionary with package info and security indicators, or None on error
        """
        # Rate limiting: simple delay between requests
        time.sleep(0.1)  # 100ms delay to avoid rate limiting

        # Build URL for PyPI JSON API
        if version:
            url = f"https://pypi.org/pypi/{package_name}/{version}/json"
        else:
            url = f"https://pypi.org/pypi/{package_name}/json"

        try:
            # Create request with proper headers
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "GreenLang-Security-Scanner/1.0",
                    "Accept": "application/json"
                }
            )

            # Fetch data with timeout
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"PyPI API returned status {response.status} for {package_name}")
                    return None

                data = json.loads(response.read().decode("utf-8"))

            # Extract relevant information
            info = data.get("info", {})
            releases = data.get("releases", {})

            # Check for security-related keywords in description and changelog
            security_keywords = [
                "security", "vulnerability", "cve", "exploit", "patch",
                "fix", "xss", "injection", "csrf", "rce", "dos", "bypass",
                "authentication", "authorization", "privilege", "escalation"
            ]

            description = (info.get("description") or "").lower()
            summary = (info.get("summary") or "").lower()
            changelog_url = info.get("project_urls", {}).get("Changelog", "")

            # Check if description mentions security
            is_security_release = any(
                keyword in description or keyword in summary
                for keyword in security_keywords
            )

            # Check version-specific release info if available
            version_info = releases.get(version, [])
            if version_info:
                # Check upload comment or yanked status
                for release in version_info:
                    if release.get("yanked"):
                        # Yanked releases often indicate security issues
                        is_security_release = True
                        break

            result = {
                "package": package_name,
                "version": version or info.get("version"),
                "latest_version": info.get("version"),
                "summary": info.get("summary"),
                "author": info.get("author"),
                "license": info.get("license"),
                "home_page": info.get("home_page"),
                "project_url": info.get("project_url"),
                "requires_python": info.get("requires_python"),
                "is_security_release": is_security_release,
                "release_count": len(releases),
                "last_release_date": self._get_latest_release_date(releases)
            }

            logger.debug(f"PyPI check for {package_name}@{version}: security={is_security_release}")
            return result

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Package {package_name}@{version} not found on PyPI")
            elif e.code == 429:
                logger.warning(f"PyPI rate limit exceeded. Waiting before retry...")
                time.sleep(60)  # Wait 60 seconds on rate limit
                return self._check_pypi_package(package_name, version)  # Retry once
            else:
                logger.error(f"PyPI API HTTP error for {package_name}: {e.code} {e.reason}")
            return None

        except urllib.error.URLError as e:
            logger.error(f"PyPI API connection error for {package_name}: {e.reason}")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PyPI response for {package_name}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error checking PyPI for {package_name}: {e}")
            return None

    def _get_latest_release_date(self, releases: Dict[str, List]) -> Optional[str]:
        """Extract the latest release date from PyPI releases data."""
        latest_date = None
        for version, files in releases.items():
            for file_info in files:
                upload_time = file_info.get("upload_time")
                if upload_time:
                    if latest_date is None or upload_time > latest_date:
                        latest_date = upload_time
        return latest_date

    def _get_replacement(self, package: str) -> str:
        """Get recommended replacement for unmaintained package"""
        replacements = {
            "pycrypto": "pycryptodome or cryptography",
            "pyyaml": "ruamel.yaml or pyyaml >= 5.4"
        }

        return replacements.get(package.lower(), "Contact security team")

    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity ratio between two package names.

        Uses SequenceMatcher for fuzzy string matching.

        Args:
            name1: First package name
            name2: Second package name

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        # Normalize names for comparison
        name1 = name1.lower().replace("-", "").replace("_", "")
        name2 = name2.lower().replace("-", "").replace("_", "")
        return SequenceMatcher(None, name1, name2).ratio()

    def _check_package_exists_on_pypi(self, package_name: str) -> bool:
        """
        Check if a package exists on PyPI.

        Args:
            package_name: Name of the package to check

        Returns:
            True if package exists on PyPI, False otherwise
        """
        url = f"https://pypi.org/pypi/{package_name}/json"

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "GreenLang-Security-Scanner/1.0",
                    "Accept": "application/json"
                }
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            logger.debug(f"PyPI check error for {package_name}: {e.code}")
            return False

        except Exception as e:
            logger.debug(f"PyPI check failed for {package_name}: {e}")
            return False

    async def check_typosquatting(
        self,
        packages: Optional[List[str]] = None
    ) -> List[TyposquattingIssue]:
        """
        Detect potential typosquatting attacks by comparing installed packages
        against well-known popular packages.

        Typosquatting is when attackers create packages with names similar to
        popular packages to trick users into installing malicious code.

        Args:
            packages: List of package names to check (uses installed if None)

        Returns:
            List of potential typosquatting issues detected
        """
        logger.info("Checking for potential typosquatting attacks...")
        issues: List[TyposquattingIssue] = []

        # Get installed packages if not provided
        if packages is None:
            packages = list(self.all_packages)

        if not packages:
            try:
                result = subprocess.run(
                    ["pip", "list", "--format", "json"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    packages = [pkg["name"] for pkg in data]
            except Exception as e:
                logger.error(f"Failed to get installed packages: {e}")
                return issues

        # Check each package against popular packages
        for package in packages:
            package_lower = package.lower()

            # Skip if it's exactly a popular package
            if package_lower in {p.lower() for p in self.POPULAR_PACKAGES}:
                continue

            # Check similarity to popular packages
            for popular in self.POPULAR_PACKAGES:
                similarity = self._calculate_similarity(package, popular)

                if self.TYPOSQUATTING_THRESHOLD <= similarity < 1.0:
                    # High similarity but not exact match - potential typosquatting
                    pypi_exists = self._check_package_exists_on_pypi(package)

                    # Determine severity based on similarity and PyPI status
                    if similarity >= 0.95:
                        severity = Severity.CRITICAL
                    elif similarity >= 0.90:
                        severity = Severity.HIGH
                    else:
                        severity = Severity.MEDIUM

                    issue = TyposquattingIssue(
                        package=package,
                        similar_to=popular,
                        similarity_score=round(similarity, 3),
                        pypi_exists=pypi_exists,
                        issue=(
                            f"Package '{package}' is suspiciously similar to "
                            f"popular package '{popular}' (similarity: {similarity:.1%}). "
                            f"This could be a typosquatting attack."
                        ),
                        severity=severity
                    )
                    issues.append(issue)
                    logger.warning(
                        f"Potential typosquatting detected: {package} -> {popular} "
                        f"(similarity: {similarity:.1%})"
                    )

            # Add small delay to avoid overwhelming PyPI
            time.sleep(0.05)

        logger.info(f"Found {len(issues)} potential typosquatting issues")
        return issues

    async def check_github_advisories(
        self,
        packages: Optional[List[str]] = None
    ) -> List[GitHubSecurityAdvisory]:
        """
        Check packages against GitHub Security Advisory Database.

        Uses GitHub's GraphQL API for efficient bulk queries when authenticated,
        falls back to REST API otherwise.

        Args:
            packages: List of package names to check (uses installed if None)

        Returns:
            List of GitHub security advisories found for packages
        """
        logger.info("Checking GitHub Security Advisory Database...")
        advisories: List[GitHubSecurityAdvisory] = []

        # Get installed packages if not provided
        if packages is None:
            packages = list(self.all_packages)

        if not packages:
            try:
                result = subprocess.run(
                    ["pip", "list", "--format", "json"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    packages = [pkg["name"] for pkg in data]
            except Exception as e:
                logger.error(f"Failed to get installed packages: {e}")
                return advisories

        # Try GraphQL API for bulk queries if token is available
        if self.github_client.token:
            advisories_map = self.github_client.get_package_advisories_graphql(
                packages, ecosystem="PIP"
            )

            for package_name, vulns in advisories_map.items():
                for vuln in vulns:
                    advisory_data = vuln.get("advisory", {})

                    # Extract CVE ID if present
                    cve_id = None
                    for identifier in advisory_data.get("identifiers", []):
                        if identifier.get("type") == "CVE":
                            cve_id = identifier.get("value")
                            break

                    # Map severity
                    severity_str = advisory_data.get("severity", "MODERATE")
                    severity_map = {
                        "CRITICAL": Severity.CRITICAL,
                        "HIGH": Severity.HIGH,
                        "MODERATE": Severity.MEDIUM,
                        "LOW": Severity.LOW
                    }
                    severity = severity_map.get(severity_str, Severity.MEDIUM)

                    # Get patched version
                    patched = vuln.get("firstPatchedVersion", {})
                    patched_version = patched.get("identifier") if patched else None

                    advisory = GitHubSecurityAdvisory(
                        package=package_name,
                        ghsa_id=advisory_data.get("ghsaId", ""),
                        cve_id=cve_id,
                        severity=severity,
                        summary=advisory_data.get("summary", ""),
                        description=advisory_data.get("description", "")[:500],
                        vulnerable_versions=vuln.get("vulnerableVersionRange", ""),
                        patched_versions=patched_version,
                        published_at=advisory_data.get("publishedAt", ""),
                        updated_at=advisory_data.get("updatedAt", ""),
                        url=advisory_data.get("permalink", "")
                    )
                    advisories.append(advisory)
        else:
            # Fall back to REST API (slower, package by package)
            logger.info(
                "No GitHub token available, using REST API "
                "(set GITHUB_TOKEN for better performance)"
            )
            for package_name in packages[:50]:  # Limit to avoid rate limiting
                rest_advisories = self.github_client.get_security_advisories(
                    package_name, ecosystem="pip"
                )

                for adv in rest_advisories:
                    # Extract CVE ID
                    cve_id = None
                    for identifier in adv.get("identifiers", []):
                        if identifier.get("type") == "CVE":
                            cve_id = identifier.get("value")
                            break

                    severity_str = adv.get("severity", "moderate")
                    severity_map = {
                        "critical": Severity.CRITICAL,
                        "high": Severity.HIGH,
                        "moderate": Severity.MEDIUM,
                        "low": Severity.LOW
                    }
                    severity = severity_map.get(severity_str.lower(), Severity.MEDIUM)

                    advisory = GitHubSecurityAdvisory(
                        package=package_name,
                        ghsa_id=adv.get("ghsa_id", ""),
                        cve_id=cve_id,
                        severity=severity,
                        summary=adv.get("summary", ""),
                        description=adv.get("description", "")[:500],
                        vulnerable_versions=adv.get("vulnerable_version_range", ""),
                        patched_versions=adv.get("patched_versions"),
                        published_at=adv.get("published_at", ""),
                        updated_at=adv.get("updated_at", ""),
                        url=adv.get("html_url", "")
                    )
                    advisories.append(advisory)

                time.sleep(0.1)  # Rate limiting

        logger.info(f"Found {len(advisories)} GitHub security advisories")
        return advisories

    async def generate_report(self, scan_result: ScanResult) -> str:
        """Generate JSON report including all scan results."""
        report_path = self.root_dir / "security" / "reports" / "DEPENDENCY_VULNERABILITIES.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        report_data = {
            "timestamp": scan_result.timestamp,
            "total_packages": scan_result.total_packages,
            "summary": scan_result.summary,
            "vulnerabilities": [
                {
                    "package": v.package,
                    "installed_version": v.installed_version,
                    "vulnerable_spec": v.vulnerable_spec,
                    "cve": v.cve,
                    "severity": v.severity.value,
                    "description": v.description,
                    "fixed_version": v.fixed_version,
                    "source": v.source
                }
                for v in scan_result.vulnerabilities
            ],
            "license_issues": [
                {
                    "package": li.package,
                    "version": li.version,
                    "license": li.license,
                    "issue": li.issue,
                    "severity": li.severity.value
                }
                for li in scan_result.license_issues
            ],
            "typosquatting_issues": [
                {
                    "package": ti.package,
                    "similar_to": ti.similar_to,
                    "similarity_score": ti.similarity_score,
                    "pypi_exists": ti.pypi_exists,
                    "issue": ti.issue,
                    "severity": ti.severity.value
                }
                for ti in scan_result.typosquatting_issues
            ],
            "github_advisories": [
                {
                    "package": ga.package,
                    "ghsa_id": ga.ghsa_id,
                    "cve_id": ga.cve_id,
                    "severity": ga.severity.value,
                    "summary": ga.summary,
                    "description": ga.description,
                    "vulnerable_versions": ga.vulnerable_versions,
                    "patched_versions": ga.patched_versions,
                    "published_at": ga.published_at,
                    "updated_at": ga.updated_at,
                    "url": ga.url
                }
                for ga in scan_result.github_advisories
            ],
            "outdated_packages": scan_result.outdated_packages,
            "unmaintained_packages": scan_result.unmaintained_packages
        }

        # Write report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Report saved to: {report_path}")

        return str(report_path)

    async def create_pr_for_fixes(self, scan_result: ScanResult) -> bool:
        """
        Create a GitHub Pull Request with security fixes.

        This method:
        1. Creates a new branch for security fixes
        2. Updates requirements files with patched versions
        3. Commits the changes
        4. Creates a PR with detailed summary

        Args:
            scan_result: The scan result containing vulnerabilities to fix

        Returns:
            True if PR was created successfully, False otherwise
        """
        logger.info("Creating PR for security fixes...")

        # Check if we have GitHub token
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.error(
                "GITHUB_TOKEN environment variable not set. "
                "Cannot create PR without authentication."
            )
            return False

        # Check if we're in a git repository
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            if result.returncode != 0:
                logger.error("Not in a git repository")
                return False
        except FileNotFoundError:
            logger.error("Git not found")
            return False

        # Get current branch and repo info
        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            base_branch = result.stdout.strip()

            # Get remote URL to extract owner/repo
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            remote_url = result.stdout.strip()

            # Parse owner/repo from remote URL
            match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", remote_url)
            if not match:
                logger.error(f"Could not parse GitHub repo from URL: {remote_url}")
                return False

            owner, repo = match.groups()
            repo = repo.replace(".git", "")

        except Exception as e:
            logger.error(f"Failed to get git info: {e}")
            return False

        # Create branch name
        timestamp = DeterministicClock.utcnow().strftime("%Y%m%d-%H%M%S")
        branch_name = f"security-fixes-{timestamp}"

        # Step 1: Create new branch
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                capture_output=True,
                check=True,
                cwd=self.root_dir
            )
            logger.info(f"Created branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False

        # Step 2: Update requirements files with fixed versions
        updated_files = []
        fixes_applied = []

        for vuln in scan_result.vulnerabilities:
            if vuln.fixed_version:
                # Find and update requirements files
                for req_file in self.requirements_files:
                    try:
                        content = req_file.read_text()
                        # Match package with various version specifiers
                        pattern = rf"^{re.escape(vuln.package)}[=<>~!].*$"
                        replacement = f"{vuln.package}=={vuln.fixed_version}"

                        new_content, count = re.subn(
                            pattern,
                            replacement,
                            content,
                            flags=re.MULTILINE | re.IGNORECASE
                        )

                        if count > 0:
                            req_file.write_text(new_content)
                            if str(req_file) not in updated_files:
                                updated_files.append(str(req_file))
                            fixes_applied.append({
                                "package": vuln.package,
                                "from_version": vuln.installed_version,
                                "to_version": vuln.fixed_version,
                                "cve": vuln.cve,
                                "file": str(req_file)
                            })
                            logger.info(
                                f"Updated {vuln.package} to {vuln.fixed_version} "
                                f"in {req_file}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to update {req_file}: {e}")

        if not fixes_applied:
            logger.warning("No fixes could be applied")
            # Checkout back to original branch
            subprocess.run(
                ["git", "checkout", base_branch],
                capture_output=True,
                cwd=self.root_dir
            )
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                capture_output=True,
                cwd=self.root_dir
            )
            return False

        # Step 3: Commit the changes
        try:
            # Stage updated files
            for file_path in updated_files:
                subprocess.run(
                    ["git", "add", file_path],
                    capture_output=True,
                    check=True,
                    cwd=self.root_dir
                )

            # Create commit message
            commit_message = (
                f"fix(security): Update dependencies with security patches\n\n"
                f"This commit updates the following packages:\n"
            )
            for fix in fixes_applied:
                commit_message += (
                    f"- {fix['package']}: {fix['from_version']} -> {fix['to_version']}"
                )
                if fix['cve']:
                    commit_message += f" ({fix['cve']})"
                commit_message += "\n"

            commit_message += "\nGenerated by GreenLang Security Scanner"

            subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                check=True,
                cwd=self.root_dir
            )
            logger.info("Committed security fixes")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            subprocess.run(
                ["git", "checkout", base_branch],
                capture_output=True,
                cwd=self.root_dir
            )
            return False

        # Step 4: Push branch
        try:
            subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                capture_output=True,
                check=True,
                cwd=self.root_dir
            )
            logger.info(f"Pushed branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push branch: {e}")
            return False

        # Step 5: Create PR via GitHub API
        try:
            pr_title = f"Security: Update vulnerable dependencies ({len(fixes_applied)} fixes)"

            pr_body = "## Security Dependency Updates\n\n"
            pr_body += "This PR was automatically generated by the GreenLang Security Scanner.\n\n"

            pr_body += "### Vulnerabilities Fixed\n\n"
            pr_body += "| Package | From | To | CVE | Severity |\n"
            pr_body += "|---------|------|----|----|----------|\n"

            for fix in fixes_applied:
                vuln = next(
                    (v for v in scan_result.vulnerabilities
                     if v.package == fix['package']),
                    None
                )
                severity = vuln.severity.value if vuln else "unknown"
                pr_body += (
                    f"| {fix['package']} | {fix['from_version']} | "
                    f"{fix['to_version']} | {fix['cve'] or 'N/A'} | {severity} |\n"
                )

            pr_body += "\n### Files Modified\n\n"
            for file_path in updated_files:
                pr_body += f"- `{file_path}`\n"

            pr_body += "\n### Action Required\n\n"
            pr_body += "1. Review the dependency changes\n"
            pr_body += "2. Run tests to ensure compatibility\n"
            pr_body += "3. Merge when ready\n"

            # Create PR using GitHub API
            pr_data = json.dumps({
                "title": pr_title,
                "body": pr_body,
                "head": branch_name,
                "base": base_branch,
                "maintainer_can_modify": True
            }).encode("utf-8")

            pr_request = urllib.request.Request(
                f"https://api.github.com/repos/{owner}/{repo}/pulls",
                data=pr_data,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {github_token}",
                    "User-Agent": "GreenLang-Security-Scanner/1.0",
                    "Content-Type": "application/json",
                    "X-GitHub-Api-Version": "2022-11-28"
                },
                method="POST"
            )

            with urllib.request.urlopen(pr_request, timeout=30) as response:
                if response.status in (200, 201):
                    pr_response = json.loads(response.read().decode("utf-8"))
                    pr_url = pr_response.get("html_url", "")
                    logger.info(f"Created PR: {pr_url}")
                    print(f"\nPull Request created: {pr_url}\n")
                    return True
                else:
                    logger.error(f"Failed to create PR: {response.status}")
                    return False

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error(f"GitHub API error creating PR: {e.code} - {error_body}")
            return False
        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
            return False

        finally:
            # Checkout back to original branch
            subprocess.run(
                ["git", "checkout", base_branch],
                capture_output=True,
                cwd=self.root_dir
            )


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Scan dependencies for vulnerabilities")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory")
    parser.add_argument("--auto-fix", action="store_true", help="Create PR with fixes")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")
    parser.add_argument(
        "--skip-typosquatting",
        action="store_true",
        help="Skip typosquatting detection"
    )
    parser.add_argument(
        "--skip-github-advisories",
        action="store_true",
        help="Skip GitHub advisory checks"
    )

    args = parser.parse_args()

    scanner = DependencyScanner(args.root)

    # Find requirements files
    scanner.find_requirements_files()

    # Build list of scans to run
    scan_tasks = [
        scanner.scan_with_safety(),
        scanner.scan_with_pip_audit(),
        scanner.check_outdated_packages(),
        scanner.check_licenses(),
        scanner.check_unmaintained(),
    ]

    # Optionally add typosquatting and GitHub advisory scans
    if not args.skip_typosquatting:
        scan_tasks.append(scanner.check_typosquatting())
    if not args.skip_github_advisories:
        scan_tasks.append(scanner.check_github_advisories())

    # Run scans in parallel
    results = await asyncio.gather(*scan_tasks)

    # Unpack results
    safety_vulns = results[0]
    pip_audit_vulns = results[1]
    outdated = results[2]
    licenses = results[3]
    unmaintained = results[4]

    typosquatting_issues = results[5] if not args.skip_typosquatting else []
    github_advisories = (
        results[6] if not args.skip_github_advisories else
        (results[5] if args.skip_typosquatting else [])
    )

    # Handle case when both are skipped
    if args.skip_typosquatting and args.skip_github_advisories:
        typosquatting_issues = []
        github_advisories = []
    elif args.skip_typosquatting and not args.skip_github_advisories:
        github_advisories = results[5]

    # Combine vulnerabilities (deduplicate)
    all_vulns = safety_vulns + pip_audit_vulns
    unique_vulns = []
    seen = set()

    for v in all_vulns:
        key = (v.package, v.installed_version, v.cve)
        if key not in seen:
            seen.add(key)
            unique_vulns.append(v)

    # Calculate summary
    summary = {
        "total_vulnerabilities": len(unique_vulns),
        "critical": sum(1 for v in unique_vulns if v.severity == Severity.CRITICAL),
        "high": sum(1 for v in unique_vulns if v.severity == Severity.HIGH),
        "medium": sum(1 for v in unique_vulns if v.severity == Severity.MEDIUM),
        "low": sum(1 for v in unique_vulns if v.severity == Severity.LOW),
        "license_issues": len(licenses),
        "outdated_packages": len(outdated),
        "unmaintained_packages": len(unmaintained),
        "typosquatting_issues": len(typosquatting_issues),
        "github_advisories": len(github_advisories)
    }

    # Create scan result
    scan_result = ScanResult(
        timestamp=DeterministicClock.utcnow().isoformat(),
        total_packages=len(scanner.all_packages),
        vulnerabilities=unique_vulns,
        license_issues=licenses,
        outdated_packages=outdated,
        unmaintained_packages=unmaintained,
        typosquatting_issues=typosquatting_issues,
        github_advisories=github_advisories,
        summary=summary
    )

    # Generate report
    report_path = await scanner.generate_report(scan_result)

    # Print summary
    print("\n" + "="*80)
    print("DEPENDENCY SECURITY SCAN RESULTS")
    print("="*80)
    print(f"Timestamp: {scan_result.timestamp}")
    print(f"Total Packages: {scan_result.total_packages}")
    print(f"\nVulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  High: {summary['high']}")
    print(f"  Medium: {summary['medium']}")
    print(f"  Low: {summary['low']}")
    print(f"\nLicense Issues: {summary['license_issues']}")
    print(f"Outdated Packages (with security updates): {summary['outdated_packages']}")
    print(f"Unmaintained Packages: {summary['unmaintained_packages']}")
    print(f"Typosquatting Issues: {summary['typosquatting_issues']}")
    print(f"GitHub Security Advisories: {summary['github_advisories']}")
    print(f"\nReport: {report_path}")
    print("="*80 + "\n")

    # Fail CI if critical/high vulnerabilities
    if summary['critical'] > 0 or summary['high'] > 0:
        print("ERROR: Critical or High severity vulnerabilities found!")
        print("Please update vulnerable packages before deploying.")
        sys.exit(1)

    # Create PR if requested
    if args.auto_fix and not args.report_only:
        await scanner.create_pr_for_fixes(scan_result)

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
