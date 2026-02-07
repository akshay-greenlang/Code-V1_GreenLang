# -*- coding: utf-8 -*-
"""
Security Scan Orchestrator - SEC-007

Coordinates execution of all security scanners, aggregates results,
applies deduplication, and generates unified reports.

The orchestrator is the main entry point for running security scans.
It manages parallel execution, result aggregation, and output generation.

Example:
    >>> from greenlang.infrastructure.security_scanning import (
    ...     ScanOrchestrator,
    ...     ScanOrchestratorConfig,
    ... )
    >>> config = ScanOrchestratorConfig.from_environment()
    >>> orchestrator = ScanOrchestrator(config)
    >>> report = await orchestrator.scan("/path/to/code")
    >>> print(f"Found {report.get_total_finding_count()} issues")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type
from uuid import uuid4

from greenlang.infrastructure.security_scanning.config import (
    ScannerConfig,
    ScannerType,
    ScanOrchestratorConfig,
    Severity,
)
from greenlang.infrastructure.security_scanning.models import (
    ScanFinding,
    ScanReport,
    ScanResult,
    ScanStatus,
)
from greenlang.infrastructure.security_scanning.deduplication import (
    DeduplicationEngine,
    DeduplicationResult,
)
from greenlang.infrastructure.security_scanning.sarif_generator import (
    SARIFGenerator,
)
from greenlang.infrastructure.security_scanning.scanners.base import (
    BaseScanner,
    ScannerError,
)

# Optional PII scanner imports
try:
    from greenlang.infrastructure.security_scanning.pii_scanner import (
        PIIScanner,
        PIIFinding,
        ScanResult as PIIScanResult,
        DataClassification,
        get_pii_scanner,
    )
    PII_SCANNER_AVAILABLE = True
except ImportError:
    PIIScanner = None  # type: ignore
    PIIFinding = None  # type: ignore
    PIIScanResult = None  # type: ignore
    DataClassification = None  # type: ignore
    get_pii_scanner = None  # type: ignore
    PII_SCANNER_AVAILABLE = False

# Optional Hybrid PII scanner for ML-based detection
try:
    from greenlang.infrastructure.security_scanning.pii_ml import (
        HybridPIIScanner,
        get_presidio_scanner,
        PRESIDIO_AVAILABLE,
    )
    PII_ML_AVAILABLE = True
except ImportError:
    HybridPIIScanner = None  # type: ignore
    get_presidio_scanner = None  # type: ignore
    PRESIDIO_AVAILABLE = False
    PII_ML_AVAILABLE = False

# Optional PII alert routing
try:
    from greenlang.infrastructure.security_scanning.pii_alerts import (
        PIIAlertRouter,
        get_pii_alert_router,
    )
    PII_ALERTS_AVAILABLE = True
except ImportError:
    PIIAlertRouter = None  # type: ignore
    get_pii_alert_router = None  # type: ignore
    PII_ALERTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scanner Registry
# ---------------------------------------------------------------------------

# Lazy imports to avoid circular dependencies
_SCANNER_CLASSES: Dict[str, str] = {
    # SAST
    "bandit": "greenlang.infrastructure.security_scanning.scanners.sast.BanditScanner",
    "semgrep": "greenlang.infrastructure.security_scanning.scanners.sast.SemgrepScanner",
    "codeql": "greenlang.infrastructure.security_scanning.scanners.sast.CodeQLScanner",
    # SCA
    "trivy": "greenlang.infrastructure.security_scanning.scanners.sca.TrivyScanner",
    "snyk": "greenlang.infrastructure.security_scanning.scanners.sca.SnykScanner",
    "pip-audit": "greenlang.infrastructure.security_scanning.scanners.sca.PipAuditScanner",
    "safety": "greenlang.infrastructure.security_scanning.scanners.sca.SafetyScanner",
    # Secrets
    "gitleaks": "greenlang.infrastructure.security_scanning.scanners.secrets.GitleaksScanner",
    "trufflehog": "greenlang.infrastructure.security_scanning.scanners.secrets.TrufflehogScanner",
    "detect-secrets": "greenlang.infrastructure.security_scanning.scanners.secrets.DetectSecretsScanner",
    # Container
    "trivy-container": "greenlang.infrastructure.security_scanning.scanners.container.TrivyContainerScanner",
    "grype": "greenlang.infrastructure.security_scanning.scanners.container.GrypeScanner",
    "cosign": "greenlang.infrastructure.security_scanning.scanners.container.CosignVerifier",
    # IaC
    "tfsec": "greenlang.infrastructure.security_scanning.scanners.iac.TfsecScanner",
    "checkov": "greenlang.infrastructure.security_scanning.scanners.iac.CheckovScanner",
    "kubeconform": "greenlang.infrastructure.security_scanning.scanners.iac.KubeconformScanner",
    # PII (pattern-based and ML-based)
    "pii-regex": "greenlang.infrastructure.security_scanning.pii_scanner.PIIScanner",
    "pii-ml": "greenlang.infrastructure.security_scanning.pii_ml.HybridPIIScanner",
}


def _get_scanner_class(scanner_name: str) -> Optional[Type[BaseScanner]]:
    """Get scanner class by name.

    Args:
        scanner_name: Scanner name.

    Returns:
        Scanner class or None.
    """
    class_path = _SCANNER_CLASSES.get(scanner_name)
    if not class_path:
        return None

    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning("Failed to load scanner %s: %s", scanner_name, e)
        return None


# ---------------------------------------------------------------------------
# Scan Orchestrator
# ---------------------------------------------------------------------------


class ScanOrchestrator:
    """Orchestrates security scanning across multiple tools.

    Manages parallel execution of scanners, aggregates results,
    applies deduplication, and generates reports in multiple formats.

    Attributes:
        config: Orchestrator configuration.
        scanners: Instantiated scanner objects.
        dedup_engine: Deduplication engine.
        sarif_generator: SARIF report generator.

    Example:
        >>> config = ScanOrchestratorConfig()
        >>> orchestrator = ScanOrchestrator(config)
        >>> report = await orchestrator.scan("/path/to/code")
    """

    def __init__(self, config: Optional[ScanOrchestratorConfig] = None) -> None:
        """Initialize scan orchestrator.

        Args:
            config: Orchestrator configuration. Defaults to environment config.
        """
        self.config = config or ScanOrchestratorConfig.from_environment()
        self.scanners: Dict[str, BaseScanner] = {}
        self.dedup_engine = DeduplicationEngine(
            enable_cve_dedup=self.config.deduplication_enabled,
            enable_fingerprint_dedup=self.config.deduplication_enabled,
            enable_location_dedup=self.config.deduplication_enabled,
        )
        self.sarif_generator = SARIFGenerator()

        self._initialize_scanners()

        logger.info(
            "ScanOrchestrator initialized  "
            "scanners=%d  parallel=%d  timeout=%ds",
            len(self.scanners),
            self.config.parallel_scans,
            self.config.global_timeout_seconds,
        )

    def _initialize_scanners(self) -> None:
        """Initialize scanner instances from configuration."""
        for scanner_config in self.config.get_enabled_scanners():
            scanner_class = _get_scanner_class(scanner_config.name)
            if scanner_class:
                try:
                    scanner = scanner_class(scanner_config)
                    if scanner.is_available():
                        self.scanners[scanner_config.name] = scanner
                        logger.debug(
                            "Initialized scanner: %s (version: %s)",
                            scanner_config.name,
                            scanner.version,
                        )
                    else:
                        logger.warning(
                            "Scanner %s not available on system",
                            scanner_config.name,
                        )
                except Exception as e:
                    logger.error(
                        "Failed to initialize scanner %s: %s",
                        scanner_config.name,
                        e,
                    )

    async def scan(
        self,
        target_path: str,
        scanner_types: Optional[Set[ScannerType]] = None,
        specific_scanners: Optional[List[str]] = None,
    ) -> ScanReport:
        """Execute security scan on target path.

        Runs enabled scanners in parallel, aggregates results, and
        applies deduplication.

        Args:
            target_path: Path to scan.
            scanner_types: Specific scanner types to run. Defaults to config.
            specific_scanners: Specific scanner names to run. Overrides types.

        Returns:
            Aggregated scan report.
        """
        started_at = datetime.now(timezone.utc)

        logger.info("Starting security scan  path=%s", target_path)

        # Determine which scanners to run
        scanners_to_run = self._select_scanners(scanner_types, specific_scanners)

        if not scanners_to_run:
            logger.warning("No scanners available to run")
            return self._create_empty_report(target_path, started_at)

        logger.info(
            "Running %d scanners: %s",
            len(scanners_to_run),
            ", ".join(scanners_to_run.keys()),
        )

        # Execute scanners in parallel with concurrency limit
        results = await self._execute_scanners(scanners_to_run, target_path)

        # Aggregate results
        report = self._aggregate_results(results, target_path, started_at)

        # Apply deduplication
        if self.config.deduplication_enabled:
            report = self._apply_deduplication(report)

        # Generate outputs
        await self._generate_outputs(report)

        logger.info(
            "Security scan complete  "
            "findings=%d  critical=%d  high=%d  duration=%.1fs",
            report.get_total_finding_count(),
            len(report.get_critical_findings()),
            len(report.get_high_findings()),
            report.total_duration_seconds,
        )

        return report

    async def scan_container(
        self,
        image_ref: str,
        verify_signature: bool = True,
    ) -> ScanReport:
        """Scan a container image.

        Args:
            image_ref: Container image reference (e.g., "nginx:latest").
            verify_signature: Also verify image signature with cosign.

        Returns:
            Scan report for the container.
        """
        started_at = datetime.now(timezone.utc)

        logger.info("Starting container scan  image=%s", image_ref)

        # Select container scanners
        container_scanners = {
            name: scanner
            for name, scanner in self.scanners.items()
            if scanner.config.scanner_type == ScannerType.CONTAINER
        }

        if verify_signature and "cosign" not in container_scanners:
            logger.warning("Cosign not available for signature verification")

        results = await self._execute_scanners(container_scanners, image_ref)

        report = self._aggregate_results(results, image_ref, started_at)

        if self.config.deduplication_enabled:
            report = self._apply_deduplication(report)

        await self._generate_outputs(report)

        return report

    async def scan_pii(
        self,
        target_path: str,
        use_ml: bool = False,
        route_alerts: bool = False,
        extensions: Optional[List[str]] = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Scan a directory or file for PII (Personally Identifiable Information).

        This method provides dedicated PII scanning using pattern-based detection
        (regex) and optionally ML-based detection using Microsoft Presidio.

        Args:
            target_path: Path to scan (file or directory).
            use_ml: Enable ML-based detection (requires Presidio). Default False.
            route_alerts: Route findings to appropriate teams. Default False.
            extensions: File extensions to scan (defaults to common code/config).
            min_confidence: Minimum confidence threshold (0-1). Default 0.5.

        Returns:
            Dictionary containing:
                - findings: List of PII findings
                - summary: Statistics by classification and type
                - alerts_routed: Number of alerts sent (if route_alerts=True)
                - scan_duration_ms: Total scan duration

        Raises:
            ValueError: If PII scanner is not available.
            RuntimeError: If ML scanning requested but Presidio unavailable.

        Example:
            >>> result = await orchestrator.scan_pii("/path/to/code")
            >>> print(f"Found {len(result['findings'])} PII instances")
            >>> for f in result['findings']:
            ...     print(f"  {f.classification}: {f.pii_type} ({f.confidence_score})")
        """
        if not PII_SCANNER_AVAILABLE:
            raise ValueError(
                "PII scanner not available. Install greenlang with PII support."
            )

        started_at = datetime.now(timezone.utc)
        logger.info("Starting PII scan  path=%s  use_ml=%s", target_path, use_ml)

        all_findings: List[Any] = []
        alerts_routed = 0

        # Pattern-based scanning
        pii_scanner = get_pii_scanner()
        pii_scanner._min_confidence = min_confidence

        path = Path(target_path)
        if path.is_file():
            findings = pii_scanner.scan_file(str(path))
            all_findings.extend(findings)
        elif path.is_dir():
            result = pii_scanner.scan_directory(
                str(path),
                extensions=extensions,
            )
            all_findings.extend(result.findings)
        else:
            raise ValueError(f"Invalid target path: {target_path}")

        logger.debug("Pattern-based scan found %d findings", len(all_findings))

        # ML-based scanning (optional)
        if use_ml:
            if not PII_ML_AVAILABLE:
                raise RuntimeError(
                    "ML-based PII scanning requires Presidio. "
                    "Install with: pip install presidio-analyzer presidio-anonymizer"
                )

            if HybridPIIScanner is not None:
                try:
                    hybrid_scanner = HybridPIIScanner(min_confidence=min_confidence)
                    if path.is_file():
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        ml_findings = await hybrid_scanner.scan_async(content, str(path))
                        # ML findings are already deduplicated in HybridPIIScanner
                        logger.debug("ML scan found %d findings", len(ml_findings))
                        # Add ML findings (HybridPIIScanner handles deduplication)
                        all_findings.extend(ml_findings)
                    elif path.is_dir():
                        # Scan each file with hybrid scanner
                        for file_path in self._collect_files_for_pii(path, extensions):
                            try:
                                content = file_path.read_text(encoding="utf-8", errors="ignore")
                                ml_findings = await hybrid_scanner.scan_async(
                                    content, str(file_path)
                                )
                                all_findings.extend(ml_findings)
                            except Exception as e:
                                logger.warning("ML scan failed for %s: %s", file_path, e)
                except Exception as e:
                    logger.error("ML-based PII scanning failed: %s", e, exc_info=True)

        # Route alerts (optional)
        if route_alerts and PII_ALERTS_AVAILABLE and len(all_findings) > 0:
            try:
                alert_router = get_pii_alert_router()
                alerts = await alert_router.route_findings(all_findings)
                alerts_routed = len(alerts)
                logger.info("Routed %d PII alerts", alerts_routed)
            except Exception as e:
                logger.error("Failed to route PII alerts: %s", e)

        # Build summary
        summary = self._build_pii_summary(all_findings)

        duration_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000

        logger.info(
            "PII scan complete  findings=%d  classifications=%s  duration=%.1fms",
            len(all_findings),
            summary.get("by_classification", {}),
            duration_ms,
        )

        return {
            "findings": all_findings,
            "summary": summary,
            "alerts_routed": alerts_routed,
            "scan_duration_ms": duration_ms,
            "ml_enabled": use_ml,
            "target_path": target_path,
        }

    def _collect_files_for_pii(
        self,
        directory: Path,
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """Collect files for PII scanning.

        Args:
            directory: Directory to scan.
            extensions: File extensions to include.

        Returns:
            List of file paths to scan.
        """
        default_extensions = [
            ".py", ".js", ".ts", ".java", ".go", ".rb", ".php",
            ".yaml", ".yml", ".json", ".xml", ".toml", ".ini", ".cfg",
            ".env", ".conf", ".config", ".properties",
            ".md", ".txt", ".csv",
            ".tf", ".hcl",
            ".sh", ".bash", ".zsh",
        ]
        extensions = extensions or default_extensions

        exclude_dirs = {
            "node_modules", ".git", "venv", "__pycache__",
            "dist", "build", ".tox", ".pytest_cache",
        }

        files: List[Path] = []
        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                # Skip excluded directories
                if not any(part in exclude_dirs for part in file_path.parts):
                    files.append(file_path)

        return files

    def _build_pii_summary(self, findings: List[Any]) -> Dict[str, Any]:
        """Build summary statistics for PII findings.

        Args:
            findings: List of PII findings.

        Returns:
            Summary dictionary with counts by classification and type.
        """
        by_classification: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_risk: Dict[str, int] = {}

        for finding in findings:
            # Handle both PIIFinding and PIIEntity (from ML scanner)
            if hasattr(finding, "classification"):
                classification = str(finding.classification.value if hasattr(finding.classification, "value") else finding.classification)
                by_classification[classification] = by_classification.get(classification, 0) + 1

            if hasattr(finding, "pii_type"):
                pii_type = str(finding.pii_type.value if hasattr(finding.pii_type, "value") else finding.pii_type)
                by_type[pii_type] = by_type.get(pii_type, 0) + 1
            elif hasattr(finding, "entity_type"):
                # PIIEntity from Presidio
                entity_type = str(finding.entity_type)
                by_type[entity_type] = by_type.get(entity_type, 0) + 1

            if hasattr(finding, "exposure_risk"):
                risk = str(finding.exposure_risk)
                by_risk[risk] = by_risk.get(risk, 0) + 1

        return {
            "total_findings": len(findings),
            "by_classification": by_classification,
            "by_type": by_type,
            "by_risk": by_risk,
        }

    async def scan_with_pii(
        self,
        target_path: str,
        scanner_types: Optional[Set[ScannerType]] = None,
        specific_scanners: Optional[List[str]] = None,
        pii_scan_options: Optional[Dict[str, Any]] = None,
    ) -> ScanReport:
        """Execute security scan with integrated PII detection.

        Combines standard security scanning (SAST, SCA, secrets, etc.)
        with PII detection in a single unified scan.

        Args:
            target_path: Path to scan.
            scanner_types: Specific scanner types to run.
            specific_scanners: Specific scanner names to run.
            pii_scan_options: Options for PII scanning:
                - use_ml: Enable ML-based detection (default: False)
                - route_alerts: Route findings to teams (default: False)
                - extensions: File extensions to scan
                - min_confidence: Minimum confidence (default: 0.5)

        Returns:
            ScanReport with all findings including PII.

        Example:
            >>> report = await orchestrator.scan_with_pii(
            ...     "/path/to/code",
            ...     pii_scan_options={"use_ml": True, "route_alerts": True}
            ... )
        """
        # Run standard security scan
        report = await self.scan(target_path, scanner_types, specific_scanners)

        # Run PII scan if available
        if PII_SCANNER_AVAILABLE:
            pii_options = pii_scan_options or {}
            try:
                pii_result = await self.scan_pii(
                    target_path,
                    use_ml=pii_options.get("use_ml", False),
                    route_alerts=pii_options.get("route_alerts", False),
                    extensions=pii_options.get("extensions"),
                    min_confidence=pii_options.get("min_confidence", 0.5),
                )

                # Convert PII findings to ScanFinding format and add to report
                pii_findings = self._convert_pii_to_scan_findings(
                    pii_result.get("findings", []),
                    target_path,
                )
                report.all_findings.extend(pii_findings)

                # Add PII metadata to report
                report.metadata["pii_scan"] = {
                    "total_pii_findings": len(pii_result.get("findings", [])),
                    "summary": pii_result.get("summary", {}),
                    "alerts_routed": pii_result.get("alerts_routed", 0),
                    "ml_enabled": pii_result.get("ml_enabled", False),
                    "scan_duration_ms": pii_result.get("scan_duration_ms", 0),
                }

                logger.info(
                    "Added %d PII findings to security report",
                    len(pii_findings),
                )
            except Exception as e:
                logger.error("PII scan failed: %s", e, exc_info=True)
                report.metadata["pii_scan"] = {"error": str(e)}

        return report

    def _convert_pii_to_scan_findings(
        self,
        pii_findings: List[Any],
        target_path: str,
    ) -> List[ScanFinding]:
        """Convert PII findings to ScanFinding format.

        Args:
            pii_findings: List of PIIFinding objects.
            target_path: Base path for the scan.

        Returns:
            List of ScanFinding objects.
        """
        from greenlang.infrastructure.security_scanning.models import FileLocation

        scan_findings: List[ScanFinding] = []

        # Map PII risk levels to severities
        risk_to_severity = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW,
        }

        for pii in pii_findings:
            # Handle both PIIFinding and PIIEntity
            pii_type = getattr(pii, "pii_type", getattr(pii, "entity_type", "unknown"))
            if hasattr(pii_type, "value"):
                pii_type = pii_type.value

            classification = getattr(pii, "classification", "pii")
            if hasattr(classification, "value"):
                classification = classification.value

            risk = getattr(pii, "exposure_risk", "medium")
            severity = risk_to_severity.get(str(risk).lower(), Severity.MEDIUM)

            # Build location
            location = None
            file_path = getattr(pii, "file_path", None)
            if file_path:
                location = FileLocation(
                    file_path=file_path,
                    start_line=getattr(pii, "line_number", 1),
                    end_line=getattr(pii, "line_number", 1),
                    start_column=getattr(pii, "column_start", None),
                    end_column=getattr(pii, "column_end", None),
                )

            # Create ScanFinding
            finding = ScanFinding(
                finding_id=str(getattr(pii, "id", uuid4())),
                title=f"PII Detected: {pii_type} ({classification})",
                description=(
                    f"Detected {pii_type} data classified as {classification}. "
                    f"Confidence: {getattr(pii, 'confidence_score', 0.0):.0%}"
                ),
                severity=severity,
                scanner_name="pii-scanner",
                scanner_type=ScannerType.SECRETS,  # PII is treated as sensitive data
                rule_id=f"pii/{pii_type}",
                location=location,
                fingerprint=getattr(pii, "matched_text_hash", ""),
            )
            scan_findings.append(finding)

        return scan_findings

    def _select_scanners(
        self,
        scanner_types: Optional[Set[ScannerType]],
        specific_scanners: Optional[List[str]],
    ) -> Dict[str, BaseScanner]:
        """Select scanners to run based on criteria.

        Args:
            scanner_types: Scanner types to include.
            specific_scanners: Specific scanner names.

        Returns:
            Dictionary of scanner name to scanner instance.
        """
        if specific_scanners:
            return {
                name: self.scanners[name]
                for name in specific_scanners
                if name in self.scanners
            }

        types = scanner_types or self.config.enabled_scanner_types
        return {
            name: scanner
            for name, scanner in self.scanners.items()
            if scanner.config.scanner_type in types
        }

    async def _execute_scanners(
        self,
        scanners: Dict[str, BaseScanner],
        target: str,
    ) -> List[ScanResult]:
        """Execute scanners in parallel with concurrency limit.

        Args:
            scanners: Scanners to execute.
            target: Target path or image.

        Returns:
            List of scan results.
        """
        semaphore = asyncio.Semaphore(self.config.parallel_scans)

        async def run_scanner(
            name: str, scanner: BaseScanner
        ) -> ScanResult:
            async with semaphore:
                try:
                    logger.debug("Starting scanner: %s", name)
                    result = await asyncio.wait_for(
                        scanner.scan(target),
                        timeout=self.config.global_timeout_seconds,
                    )
                    logger.debug(
                        "Scanner %s completed: %d findings",
                        name,
                        len(result.findings),
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.error("Scanner %s timed out", name)
                    return ScanResult(
                        scanner_name=name,
                        scanner_type=scanner.config.scanner_type,
                        status=ScanStatus.TIMED_OUT,
                        error_message=f"Timed out after {self.config.global_timeout_seconds}s",
                        scan_path=target,
                    )
                except Exception as e:
                    logger.error("Scanner %s failed: %s", name, e, exc_info=True)
                    return ScanResult(
                        scanner_name=name,
                        scanner_type=scanner.config.scanner_type,
                        status=ScanStatus.FAILED,
                        error_message=str(e),
                        scan_path=target,
                    )

        # Run all scanners concurrently
        tasks = [
            run_scanner(name, scanner)
            for name, scanner in scanners.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any unexpected exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Unexpected scanner error: %s", result)
            elif isinstance(result, ScanResult):
                processed_results.append(result)

        return processed_results

    def _aggregate_results(
        self,
        results: List[ScanResult],
        target_path: str,
        started_at: datetime,
    ) -> ScanReport:
        """Aggregate scanner results into a report.

        Args:
            results: List of scanner results.
            target_path: Scanned target.
            started_at: Scan start time.

        Returns:
            Aggregated scan report.
        """
        completed_at = datetime.now(timezone.utc)

        all_findings = []
        scanners_run = []
        scanners_failed = []

        for result in results:
            scanners_run.append(result.scanner_name)
            all_findings.extend(result.findings)

            if result.status in (ScanStatus.FAILED, ScanStatus.TIMED_OUT):
                scanners_failed.append(result.scanner_name)

        # Get git info if available
        git_commit, git_branch = self._get_git_info(target_path)

        return ScanReport(
            scan_results=results,
            all_findings=all_findings,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            scan_path=target_path,
            git_commit=git_commit,
            git_branch=git_branch,
            scanners_run=scanners_run,
            scanners_failed=scanners_failed,
            deduplication_enabled=False,  # Will be set after dedup
        )

    def _apply_deduplication(self, report: ScanReport) -> ScanReport:
        """Apply deduplication to report findings.

        Args:
            report: Original report.

        Returns:
            Report with deduplicated findings.
        """
        dedup_result = self.dedup_engine.deduplicate(report.all_findings)

        report.all_findings = dedup_result.deduplicated_findings
        report.deduplication_enabled = True
        report.metadata["deduplication"] = {
            "original_count": dedup_result.original_count,
            "deduplicated_count": len(dedup_result.deduplicated_findings),
            "duplicates_removed": dedup_result.duplicate_count,
            "deduplication_rate": dedup_result.deduplication_rate,
        }

        return report

    async def _generate_outputs(self, report: ScanReport) -> None:
        """Generate output files from report.

        Args:
            report: Scan report.
        """
        # SARIF output
        if self.config.sarif_output_path:
            try:
                self.sarif_generator.save(
                    self.config.sarif_output_path,
                    report,
                )
            except Exception as e:
                logger.error("Failed to save SARIF: %s", e)

        # JSON output
        if self.config.json_output_path:
            try:
                self._save_json_report(report)
            except Exception as e:
                logger.error("Failed to save JSON: %s", e)

        # HTML output
        if self.config.html_output_path:
            try:
                self._save_html_report(report)
            except Exception as e:
                logger.error("Failed to save HTML: %s", e)

    def _save_json_report(self, report: ScanReport) -> None:
        """Save JSON report.

        Args:
            report: Scan report.
        """
        output = {
            "summary": report.get_summary(),
            "findings": [
                {
                    "id": f.finding_id,
                    "title": f.title,
                    "description": f.description,
                    "severity": f.severity.value,
                    "scanner": f.scanner_name,
                    "scanner_type": f.scanner_type.value,
                    "rule_id": f.rule_id,
                    "location": {
                        "file": f.location.file_path if f.location else None,
                        "line": f.location.start_line if f.location else None,
                    },
                    "cve": f.vulnerability_info.cve_id if f.vulnerability_info else None,
                    "cvss": f.vulnerability_info.cvss_score if f.vulnerability_info else None,
                    "fingerprint": f.fingerprint,
                }
                for f in report.all_findings
            ],
            "scanners": [
                {
                    "name": r.scanner_name,
                    "status": r.status.value,
                    "findings_count": len(r.findings),
                    "duration_seconds": r.duration_seconds,
                }
                for r in report.scan_results
            ],
        }

        path = Path(self.config.json_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("JSON report saved to %s", path)

    def _save_html_report(self, report: ScanReport) -> None:
        """Save HTML report.

        Args:
            report: Scan report.
        """
        summary = report.get_summary()
        counts = report.get_finding_counts_by_severity()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #fd7e14; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #28a745; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .severity-badge {{ padding: 4px 8px; border-radius: 4px; color: white; }}
        .severity-CRITICAL {{ background: #dc3545; }}
        .severity-HIGH {{ background: #fd7e14; }}
        .severity-MEDIUM {{ background: #ffc107; color: #333; }}
        .severity-LOW {{ background: #28a745; }}
        .severity-INFO {{ background: #17a2b8; }}
    </style>
</head>
<body>
    <h1>Security Scan Report</h1>
    <p>Generated: {datetime.now(timezone.utc).isoformat()}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Path:</strong> {summary['scan_path']}</p>
        <p><strong>Duration:</strong> {summary['duration_seconds']:.1f} seconds</p>
        <p><strong>Total Findings:</strong> {summary['total_findings']}</p>
        <ul>
            <li class="critical">Critical: {counts[Severity.CRITICAL]}</li>
            <li class="high">High: {counts[Severity.HIGH]}</li>
            <li class="medium">Medium: {counts[Severity.MEDIUM]}</li>
            <li class="low">Low: {counts[Severity.LOW]}</li>
        </ul>
        <p><strong>Scanners:</strong> {summary['scanners_run']} run, {summary['scanners_failed']} failed</p>
    </div>

    <h2>Findings</h2>
    <table>
        <tr>
            <th>Severity</th>
            <th>Title</th>
            <th>Location</th>
            <th>Scanner</th>
            <th>CVE</th>
        </tr>
"""

        for finding in report.all_findings:
            cve = finding.vulnerability_info.cve_id if finding.vulnerability_info else "-"
            location = f"{finding.location.file_path}:{finding.location.start_line}" if finding.location else "-"

            html += f"""        <tr>
            <td><span class="severity-badge severity-{finding.severity.value}">{finding.severity.value}</span></td>
            <td>{finding.title[:80]}</td>
            <td>{location}</td>
            <td>{finding.scanner_name}</td>
            <td>{cve}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""

        path = Path(self.config.html_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML report saved to %s", path)

    def _get_git_info(self, path: str) -> tuple[Optional[str], Optional[str]]:
        """Get git commit and branch info.

        Args:
            path: Path to check for git.

        Returns:
            Tuple of (commit_sha, branch_name).
        """
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=path,
                timeout=5,
            )
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=path,
                timeout=5,
            )

            return (
                commit.stdout.strip() if commit.returncode == 0 else None,
                branch.stdout.strip() if branch.returncode == 0 else None,
            )
        except Exception:
            return None, None

    def _create_empty_report(
        self,
        target_path: str,
        started_at: datetime,
    ) -> ScanReport:
        """Create empty report when no scanners are available.

        Args:
            target_path: Target path.
            started_at: Start time.

        Returns:
            Empty scan report.
        """
        return ScanReport(
            scan_results=[],
            all_findings=[],
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            scan_path=target_path,
            scanners_run=[],
            scanners_failed=[],
        )

    def get_available_scanners(self) -> List[Dict[str, Any]]:
        """Get list of available scanners.

        Returns:
            List of scanner info dictionaries.
        """
        return [
            scanner.get_scanner_info()
            for scanner in self.scanners.values()
        ]

    def check_blocking_findings(self, report: ScanReport) -> bool:
        """Check if report has blocking findings.

        Uses config.fail_on_severity to determine blocking threshold.

        Args:
            report: Scan report.

        Returns:
            True if blocking findings exist.
        """
        return report.has_blocking_findings(self.config.fail_on_severity)


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_orchestrator: Optional[ScanOrchestrator] = None


def get_orchestrator() -> ScanOrchestrator:
    """Get global orchestrator instance.

    Creates instance on first call using environment configuration.

    Returns:
        Global ScanOrchestrator instance.
    """
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ScanOrchestrator()
    return _global_orchestrator


def reset_orchestrator() -> None:
    """Reset global orchestrator instance.

    Useful for testing or reconfiguration.
    """
    global _global_orchestrator
    _global_orchestrator = None
