# -*- coding: utf-8 -*-
"""
Container Scanner Implementations - SEC-007

Container image security scanners for vulnerability detection and
signature verification.

Scanners:
    - TrivyContainerScanner: Container image vulnerability scanning
    - GrypeScanner: Anchore Grype container scanning
    - CosignVerifier: Container image signature verification

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners.container import (
    ...     TrivyContainerScanner,
    ...     CosignVerifier,
    ... )
    >>> config = ScannerConfig(name="trivy-container", scanner_type=ScannerType.CONTAINER)
    >>> scanner = TrivyContainerScanner(config)
    >>> result = await scanner.scan("nginx:latest")

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
    ContainerLocation,
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
# Trivy Container Scanner
# ---------------------------------------------------------------------------


class TrivyContainerScanner(BaseScanner):
    """Trivy scanner for container image vulnerabilities.

    Scans container images for vulnerabilities in OS packages and
    application dependencies. Supports Docker, OCI, and tar archives.

    Example:
        >>> config = ScannerConfig(
        ...     name="trivy-container",
        ...     scanner_type=ScannerType.CONTAINER
        ... )
        >>> scanner = TrivyContainerScanner(config)
        >>> result = await scanner.scan("python:3.11-slim")
    """

    # Severity mapping
    SEVERITY_MAP: Dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH": Severity.HIGH,
        "MEDIUM": Severity.MEDIUM,
        "LOW": Severity.LOW,
        "UNKNOWN": Severity.INFO,
    }

    async def scan(self, target_image: str) -> ScanResult:
        """Scan a container image for vulnerabilities.

        Args:
            target_image: Image reference (e.g., "nginx:latest", "ghcr.io/org/image:v1").

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Trivy not found: {self.config.executable}",
            )

        command = self._build_command(target_image)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Trivy image scan exit codes similar to fs scan
            if exit_code > 1:
                return self._create_result(
                    findings=[],
                    status=ScanStatus.FAILED,
                    started_at=started_at,
                    error_message=stderr or f"Trivy failed with exit code {exit_code}",
                    exit_code=exit_code,
                    command=" ".join(command),
                    scan_path=target_image,
                )

            findings = self.parse_results(stdout, target_image)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_image,
            )

        except Exception as e:
            logger.error("Trivy container scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_image,
            )

    def _build_command(self, target_image: str) -> List[str]:
        """Build Trivy image command.

        Args:
            target_image: Image reference.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "trivy"]
        cmd.extend(["image", target_image])
        cmd.extend(["--format", "json"])
        cmd.extend(["--scanners", "vuln"])

        # Severity filter
        severity_filter = ["CRITICAL", "HIGH", "MEDIUM"]
        if self.config.severity_threshold == Severity.CRITICAL:
            severity_filter = ["CRITICAL"]
        elif self.config.severity_threshold == Severity.HIGH:
            severity_filter = ["CRITICAL", "HIGH"]

        cmd.extend(["--severity", ",".join(severity_filter)])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(
        self, raw_output: str, image_ref: str = ""
    ) -> List[ScanFinding]:
        """Parse Trivy image scan output.

        Args:
            raw_output: JSON output from Trivy.
            image_ref: Image reference for context.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Trivy output: %s", e)
            return []

        findings = []
        results = data.get("Results", [])

        for result in results:
            target = result.get("Target", "")
            target_type = result.get("Type", "")
            vulns = result.get("Vulnerabilities", []) or []

            for vuln in vulns:
                finding = self._parse_vulnerability(
                    vuln, target, target_type, image_ref
                )
                if finding:
                    findings.append(finding)

        logger.info("Trivy found %d container vulnerabilities", len(findings))
        return findings

    def _parse_vulnerability(
        self,
        vuln: Dict[str, Any],
        target: str,
        target_type: str,
        image_ref: str,
    ) -> Optional[ScanFinding]:
        """Parse a container vulnerability.

        Args:
            vuln: Vulnerability dictionary.
            target: Target within the image (OS, package manager).
            target_type: Type of target.
            image_ref: Container image reference.

        Returns:
            ScanFinding or None.
        """
        try:
            severity_str = vuln.get("Severity", "MEDIUM")
            severity = self.SEVERITY_MAP.get(severity_str, Severity.MEDIUM)

            vuln_id = vuln.get("VulnerabilityID", "")
            pkg_name = vuln.get("PkgName", "")
            installed_version = vuln.get("InstalledVersion", "")
            fixed_version = vuln.get("FixedVersion", "")
            title = vuln.get("Title", vuln_id)
            description = vuln.get("Description", "")

            # Layer info if available
            layer = vuln.get("Layer", {})
            layer_digest = layer.get("Digest")

            # CVSS info
            cvss_score = None
            cvss_vector = None
            cvss = vuln.get("CVSS", {})
            for source in ["nvd", "redhat", "ghsa"]:
                if source in cvss:
                    cvss_score = cvss[source].get("V3Score")
                    cvss_vector = cvss[source].get("V3Vector")
                    break

            container_location = ContainerLocation(
                image_ref=image_ref,
                layer_digest=layer_digest,
                package_path=f"{target}/{pkg_name}",
            )

            # Also create file location pointing to the target
            location = FileLocation(
                file_path=target,
                start_line=1,
            )

            vuln_info = VulnerabilityInfo(
                cve_id=vuln_id if vuln_id.startswith("CVE-") else None,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                description=description,
                references=vuln.get("References", []),
            )

            remediation = RemediationInfo(
                fixed_version=fixed_version,
                description=(
                    f"Update base image or package {pkg_name} to {fixed_version}"
                    if fixed_version
                    else "No fix available. Consider using a different base image."
                ),
                patch_available=bool(fixed_version),
                auto_fixable=bool(fixed_version),
            )

            return ScanFinding(
                title=f"{vuln_id}: {title}" if title != vuln_id else vuln_id,
                description=(
                    f"Container: {image_ref}\n"
                    f"Package: {pkg_name} {installed_version} ({target_type})\n"
                    f"{description}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.CONTAINER,
                rule_id=vuln_id,
                location=location,
                container_location=container_location,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"container", "image", target_type, pkg_name},
                raw_data=vuln,
            )

        except Exception as e:
            logger.warning("Failed to parse container vulnerability: %s", e)
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
                    "fullDescription": {"text": finding.description[:500]},
                    "helpUri": f"https://nvd.nist.gov/vuln/detail/{rule_id}",
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
                            "name": "Trivy",
                            "version": self.version or "unknown",
                            "informationUri": "https://trivy.dev/",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://trivy.dev/"


# ---------------------------------------------------------------------------
# Grype Scanner
# ---------------------------------------------------------------------------


class GrypeScanner(BaseScanner):
    """Anchore Grype scanner for container vulnerabilities.

    Grype is a vulnerability scanner for container images and filesystems
    from Anchore. Fast scanning with SBOM support.

    Example:
        >>> config = ScannerConfig(name="grype", scanner_type=ScannerType.CONTAINER)
        >>> scanner = GrypeScanner(config)
        >>> result = await scanner.scan("alpine:latest")
    """

    SEVERITY_MAP: Dict[str, Severity] = {
        "Critical": Severity.CRITICAL,
        "High": Severity.HIGH,
        "Medium": Severity.MEDIUM,
        "Low": Severity.LOW,
        "Negligible": Severity.INFO,
        "Unknown": Severity.INFO,
    }

    async def scan(self, target_image: str) -> ScanResult:
        """Scan a container image with Grype.

        Args:
            target_image: Image reference.

        Returns:
            ScanResult with findings.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Grype not found: {self.config.executable}",
            )

        command = self._build_command(target_image)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            # Grype exit codes:
            # 0 = no vulnerabilities above threshold
            # 1 = vulnerabilities found above threshold

            findings = self.parse_results(stdout, target_image)
            filtered_findings = self._apply_filters(findings)

            return self._create_result(
                findings=filtered_findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_image,
            )

        except Exception as e:
            logger.error("Grype scan failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_image,
            )

    def _build_command(self, target_image: str) -> List[str]:
        """Build Grype command.

        Args:
            target_image: Image reference.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "grype"]
        cmd.append(target_image)
        cmd.extend(["-o", "json"])

        # Add fail-on threshold if needed
        if self.config.severity_threshold == Severity.CRITICAL:
            cmd.extend(["--fail-on", "critical"])
        elif self.config.severity_threshold == Severity.HIGH:
            cmd.extend(["--fail-on", "high"])

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(
        self, raw_output: str, image_ref: str = ""
    ) -> List[ScanFinding]:
        """Parse Grype JSON output.

        Args:
            raw_output: JSON output from Grype.
            image_ref: Image reference.

        Returns:
            List of findings.
        """
        if not raw_output.strip():
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Grype output: %s", e)
            return []

        findings = []
        matches = data.get("matches", [])

        for match in matches:
            finding = self._parse_match(match, image_ref)
            if finding:
                findings.append(finding)

        logger.info("Grype found %d vulnerabilities", len(findings))
        return findings

    def _parse_match(
        self, match: Dict[str, Any], image_ref: str
    ) -> Optional[ScanFinding]:
        """Parse a Grype vulnerability match.

        Args:
            match: Match dictionary.
            image_ref: Image reference.

        Returns:
            ScanFinding or None.
        """
        try:
            vulnerability = match.get("vulnerability", {})
            artifact = match.get("artifact", {})

            vuln_id = vulnerability.get("id", "")
            severity_str = vulnerability.get("severity", "Medium")
            severity = self.SEVERITY_MAP.get(severity_str, Severity.MEDIUM)
            description = vulnerability.get("description", "")
            fix_versions = vulnerability.get("fix", {}).get("versions", [])

            pkg_name = artifact.get("name", "")
            version = artifact.get("version", "")
            pkg_type = artifact.get("type", "")

            # CVSS
            cvss = vulnerability.get("cvss", [])
            cvss_score = None
            cvss_vector = None
            if cvss:
                metrics = cvss[0].get("metrics", {})
                cvss_score = metrics.get("baseScore")
                cvss_vector = cvss[0].get("vector")

            container_location = ContainerLocation(
                image_ref=image_ref,
                package_path=f"{pkg_type}/{pkg_name}",
            )

            location = FileLocation(
                file_path=f"{pkg_type}/{pkg_name}",
                start_line=1,
            )

            vuln_info = VulnerabilityInfo(
                cve_id=vuln_id if vuln_id.startswith("CVE-") else None,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                description=description,
                references=vulnerability.get("urls", []),
            )

            remediation = RemediationInfo(
                fixed_version=fix_versions[0] if fix_versions else None,
                patch_available=bool(fix_versions),
                auto_fixable=bool(fix_versions),
            )

            return ScanFinding(
                title=f"{vuln_id}: {pkg_name} vulnerability",
                description=(
                    f"Package: {pkg_name}@{version} ({pkg_type})\n"
                    f"{description}"
                ),
                severity=severity,
                scanner_name=self.name,
                scanner_type=ScannerType.CONTAINER,
                rule_id=vuln_id,
                location=location,
                container_location=container_location,
                vulnerability_info=vuln_info,
                remediation_info=remediation,
                tags={"container", "grype", pkg_type, pkg_name},
                raw_data=match,
            )

        except Exception as e:
            logger.warning("Failed to parse Grype match: %s", e)
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
                    "fullDescription": {"text": finding.description[:500]},
                    "helpUri": f"https://nvd.nist.gov/vuln/detail/{rule_id}",
                }

            results.append(finding.to_sarif_result())

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Grype",
                            "version": self.version or "unknown",
                            "informationUri": "https://github.com/anchore/grype",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://github.com/anchore/grype"


# ---------------------------------------------------------------------------
# Cosign Verifier
# ---------------------------------------------------------------------------


class CosignVerifier(BaseScanner):
    """Cosign verifier for container image signatures.

    Verifies that container images have been signed using Sigstore/Cosign.
    Reports unsigned or incorrectly signed images as findings.

    Example:
        >>> config = ScannerConfig(name="cosign", scanner_type=ScannerType.CONTAINER)
        >>> verifier = CosignVerifier(config)
        >>> result = await verifier.scan("ghcr.io/org/signed-image:v1")
    """

    async def scan(self, target_image: str) -> ScanResult:
        """Verify container image signature.

        Args:
            target_image: Image reference to verify.

        Returns:
            ScanResult with verification status.
        """
        started_at = datetime.now(timezone.utc)

        if not self.is_available():
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=f"Cosign not found: {self.config.executable}",
            )

        command = self._build_command(target_image)

        try:
            stdout, stderr, exit_code = await self._run_command(command)

            findings = []

            if exit_code != 0:
                # Signature verification failed
                finding = ScanFinding(
                    title=f"Unsigned or Invalid Signature: {target_image}",
                    description=(
                        f"Container image {target_image} does not have a valid "
                        "signature or the signature verification failed. "
                        "Unsigned images may have been tampered with."
                    ),
                    severity=Severity.HIGH,
                    scanner_name=self.name,
                    scanner_type=ScannerType.CONTAINER,
                    rule_id="unsigned-image",
                    container_location=ContainerLocation(image_ref=target_image),
                    remediation_info=RemediationInfo(
                        description=(
                            "1. Sign the image using cosign sign.\n"
                            "2. Verify the signing key is trusted.\n"
                            "3. Re-push the signed image."
                        ),
                    ),
                    tags={"container", "signature", "supply-chain"},
                    raw_data={"stderr": stderr, "exit_code": exit_code},
                )
                findings.append(finding)

            return self._create_result(
                findings=findings,
                status=ScanStatus.COMPLETED,
                started_at=started_at,
                exit_code=exit_code,
                raw_output=stdout,
                command=" ".join(command),
                scan_path=target_image,
            )

        except Exception as e:
            logger.error("Cosign verification failed: %s", e, exc_info=True)
            return self._create_result(
                findings=[],
                status=ScanStatus.FAILED,
                started_at=started_at,
                error_message=str(e),
                scan_path=target_image,
            )

    def _build_command(self, target_image: str) -> List[str]:
        """Build cosign verify command.

        Args:
            target_image: Image reference.

        Returns:
            Command list.
        """
        cmd = [self.config.executable or "cosign"]
        cmd.extend(["verify", target_image])

        # Keyless verification (Sigstore)
        if "--key" not in self.config.extra_args:
            cmd.append("--certificate-oidc-issuer=https://accounts.google.com")
            cmd.append("--certificate-identity-regexp=.*")

        cmd.extend(self.config.extra_args)
        return cmd

    def parse_results(self, raw_output: str) -> List[ScanFinding]:
        """Parse cosign verification output.

        For cosign, we generate findings based on exit code, not output parsing.

        Args:
            raw_output: Output from cosign (usually JSON payloads on success).

        Returns:
            Empty list (findings generated in scan method).
        """
        return []

    def to_sarif(self, findings: List[ScanFinding]) -> Dict[str, Any]:
        """Convert findings to SARIF format."""
        results = [f.to_sarif_result() for f in findings]

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Cosign",
                            "version": self.version or "unknown",
                            "informationUri": "https://github.com/sigstore/cosign",
                            "rules": [
                                {
                                    "id": "unsigned-image",
                                    "name": "Unsigned Container Image",
                                    "shortDescription": {
                                        "text": "Container image is not signed"
                                    },
                                    "properties": {
                                        "security-severity": "7.0",
                                    },
                                }
                            ],
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_scanner_url(self) -> str:
        return "https://github.com/sigstore/cosign"
