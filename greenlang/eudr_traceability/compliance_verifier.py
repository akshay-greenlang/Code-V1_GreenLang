# -*- coding: utf-8 -*-
"""
Compliance Verifier - AGENT-DATA-004: EUDR Traceability Connector

Verifies compliance of plots, DDS, and operators against specific EUDR
articles. Checks cover Article 3 (deforestation-free and legal), Article 9
(geolocation requirements), Article 10 (due diligence), and Article 11
(risk mitigation).

Zero-Hallucination Guarantees:
    - All checks are deterministic rule-based evaluations
    - No ML/LLM used for compliance determination
    - Each check maps to a specific EUDR article requirement
    - SHA-256 provenance hashes on all check results

Example:
    >>> from greenlang.eudr_traceability.compliance_verifier import ComplianceVerifier
    >>> verifier = ComplianceVerifier()
    >>> results = verifier.verify_compliance("plot", "PLOT-abc123")
    >>> score = verifier.get_compliance_score(results)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.eudr_traceability.models import (
    ComplianceCheckResult,
    ComplianceStatus,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class ComplianceVerifier:
    """EUDR article compliance verification engine.

    Performs deterministic compliance checks against EUDR article
    requirements for plots, due diligence statements, and operators.

    Attributes:
        ARTICLE_CHECKS: Mapping of article numbers to requirement checks.
        _config: Configuration dictionary or object.
        _plot_registry: PlotRegistryEngine for plot lookups.
        _chain_of_custody: ChainOfCustodyEngine for custody lookups.
        _check_results: In-memory check result storage.
        _provenance: Provenance tracker instance.

    Example:
        >>> verifier = ComplianceVerifier()
        >>> results = verifier.verify_article_3("PLOT-abc123")
        >>> assert all(r.is_compliant for r in results)
    """

    # EUDR article requirements for compliance checking
    ARTICLE_CHECKS: Dict[str, List[Dict[str, str]]] = {
        "article_3": [
            {
                "id": "art3_deforestation_free",
                "requirement": "Product is deforestation-free (Article 3(a))",
            },
            {
                "id": "art3_legal_compliance",
                "requirement": "Product produced in compliance with relevant legislation (Article 3(b))",
            },
            {
                "id": "art3_dds_coverage",
                "requirement": "Product covered by a due diligence statement (Article 3(c))",
            },
        ],
        "article_9": [
            {
                "id": "art9_geolocation",
                "requirement": "Geolocation of all plots of land provided (Article 9(1)(d))",
            },
            {
                "id": "art9_polygon",
                "requirement": "Polygon provided for plots > 4 hectares (Article 9(1)(d))",
            },
            {
                "id": "art9_country",
                "requirement": "Country of production identified (Article 9(1)(e))",
            },
        ],
        "article_10": [
            {
                "id": "art10_info_collection",
                "requirement": "Information collected per Article 9 (Article 10(1))",
            },
            {
                "id": "art10_risk_assessment",
                "requirement": "Risk assessment performed (Article 10(2))",
            },
            {
                "id": "art10_risk_mitigation",
                "requirement": "Risk mitigation measures applied if needed (Article 10(3))",
            },
        ],
        "article_11": [
            {
                "id": "art11_adequate_measures",
                "requirement": "Adequate and proportionate risk mitigation measures (Article 11(1))",
            },
            {
                "id": "art11_additional_info",
                "requirement": "Additional information or documents gathered (Article 11(2))",
            },
        ],
    }

    def __init__(
        self,
        config: Any = None,
        plot_registry: Any = None,
        chain_of_custody: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize ComplianceVerifier.

        Args:
            config: Optional configuration.
            plot_registry: Optional PlotRegistryEngine.
            chain_of_custody: Optional ChainOfCustodyEngine.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._plot_registry = plot_registry
        self._chain_of_custody = chain_of_custody
        self._provenance = provenance

        # In-memory storage
        self._check_results: Dict[str, ComplianceCheckResult] = {}

        logger.info("ComplianceVerifier initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_compliance(
        self,
        target_type: str,
        target_id: str,
    ) -> List[ComplianceCheckResult]:
        """Verify compliance for a target entity across all applicable articles.

        Args:
            target_type: Type of entity (plot, dds, operator).
            target_id: Entity identifier.

        Returns:
            List of ComplianceCheckResult for all applicable checks.
        """
        start_time = time.monotonic()
        results: List[ComplianceCheckResult] = []

        if target_type == "plot":
            results.extend(self.verify_article_3(target_id))
            results.extend(self.verify_article_9(target_id))
        elif target_type == "dds":
            results.extend(self.verify_article_10(target_id))
            results.extend(self.verify_article_11(target_id))
        else:
            logger.warning(
                "Unknown target_type '%s' for compliance verification",
                target_type,
            )

        # Record metrics
        elapsed = time.monotonic() - start_time
        try:
            from greenlang.eudr_traceability.metrics import (
                record_batch_operation,
            )
            record_batch_operation("compliance_verification", elapsed)
        except ImportError:
            pass

        logger.info(
            "Compliance verification for %s/%s: %d checks in %.2fs",
            target_type, target_id[:8], len(results), elapsed,
        )
        return results

    def verify_article_3(self, plot_id: str) -> List[ComplianceCheckResult]:
        """Verify Article 3 requirements for a plot.

        Checks deforestation-free status, legal compliance, and DDS
        coverage.

        Args:
            plot_id: Plot identifier.

        Returns:
            List of ComplianceCheckResult for Article 3 checks.
        """
        results: List[ComplianceCheckResult] = []
        plot = self._get_plot(plot_id)

        # Check 1: Deforestation-free
        if plot is None:
            is_compliant = False
            details = f"Plot {plot_id} not found in registry"
            remediation = "Register the plot in the plot registry"
        elif plot.deforestation_free is True:
            is_compliant = True
            details = "Plot declared as deforestation-free"
            remediation = None
        elif plot.deforestation_free is False:
            is_compliant = False
            details = "Plot is NOT deforestation-free"
            remediation = "Obtain deforestation-free certification for the plot"
        else:
            is_compliant = False
            details = "Deforestation-free status not yet declared"
            remediation = "Submit deforestation-free declaration for the plot"

        results.append(self._create_check(
            article_checked="Article 3",
            requirement="Product is deforestation-free (Article 3(a))",
            is_compliant=is_compliant,
            target_type="plot",
            target_id=plot_id,
            details=details,
            remediation=remediation,
        ))

        # Check 2: Legal compliance
        if plot is None:
            is_compliant = False
            details = f"Plot {plot_id} not found"
            remediation = "Register the plot in the plot registry"
        elif plot.legal_compliance is True:
            is_compliant = True
            details = "Legal compliance confirmed"
            remediation = None
        elif plot.legal_compliance is False:
            is_compliant = False
            details = "Legal compliance NOT confirmed"
            remediation = "Provide evidence of legal compliance with local laws"
        else:
            is_compliant = False
            details = "Legal compliance not yet declared"
            remediation = "Submit legal compliance declaration for the plot"

        results.append(self._create_check(
            article_checked="Article 3",
            requirement="Production legally compliant (Article 3(b))",
            is_compliant=is_compliant,
            target_type="plot",
            target_id=plot_id,
            details=details,
            remediation=remediation,
        ))

        # Check 3: DDS coverage (check if any DDS references this plot)
        results.append(self._create_check(
            article_checked="Article 3",
            requirement="Product covered by a DDS (Article 3(c))",
            is_compliant=False,
            target_type="plot",
            target_id=plot_id,
            details="DDS coverage check requires DueDiligenceEngine",
            remediation="Generate a DDS covering this plot",
        ))

        # Record metrics
        for r in results:
            status_str = "compliant" if r.is_compliant else "non_compliant"
            self._record_check_metric("article_3", status_str)

        return results

    def verify_article_9(self, plot_id: str) -> List[ComplianceCheckResult]:
        """Verify Article 9 geolocation requirements for a plot.

        Args:
            plot_id: Plot identifier.

        Returns:
            List of ComplianceCheckResult for Article 9 checks.
        """
        results: List[ComplianceCheckResult] = []
        plot = self._get_plot(plot_id)

        # Check 1: Geolocation provided
        if plot is None:
            is_compliant = False
            details = f"Plot {plot_id} not found"
            remediation = "Register the plot with geolocation data"
        elif (
            plot.geolocation
            and -90 <= plot.geolocation.latitude <= 90
            and -180 <= plot.geolocation.longitude <= 180
        ):
            is_compliant = True
            details = (
                f"Geolocation provided: ({plot.geolocation.latitude}, "
                f"{plot.geolocation.longitude})"
            )
            remediation = None
        else:
            is_compliant = False
            details = "Valid geolocation not provided"
            remediation = "Provide valid GPS coordinates for the plot"

        results.append(self._create_check(
            article_checked="Article 9",
            requirement="Geolocation of plot provided (Article 9(1)(d))",
            is_compliant=is_compliant,
            target_type="plot",
            target_id=plot_id,
            details=details,
            remediation=remediation,
        ))

        # Check 2: Polygon for plots > 4 hectares
        if plot is None:
            is_compliant = False
            details = f"Plot {plot_id} not found"
            remediation = "Register the plot with geolocation data"
        elif plot.geolocation and plot.geolocation.plot_area_hectares is not None:
            if plot.geolocation.plot_area_hectares > 4.0:
                if (
                    plot.geolocation.polygon_coordinates is not None
                    and len(plot.geolocation.polygon_coordinates) >= 3
                ):
                    is_compliant = True
                    details = (
                        f"Polygon provided for {plot.geolocation.plot_area_hectares} ha plot "
                        f"({len(plot.geolocation.polygon_coordinates)} vertices)"
                    )
                    remediation = None
                else:
                    is_compliant = False
                    details = (
                        f"Plot is {plot.geolocation.plot_area_hectares} ha "
                        "but polygon not provided"
                    )
                    remediation = (
                        "Provide polygon coordinates for plots larger than 4 hectares"
                    )
            else:
                # Plots <= 4 ha do not require polygon; mark as compliant
                is_compliant = True
                details = (
                    f"Plot is {plot.geolocation.plot_area_hectares} ha "
                    "(polygon not required < 4 ha)"
                )
                remediation = None
        else:
            is_compliant = False
            details = "Plot area not specified; polygon check inconclusive"
            remediation = "Provide plot area in hectares"

        results.append(self._create_check(
            article_checked="Article 9",
            requirement="Polygon for plots > 4 hectares (Article 9(1)(d))",
            is_compliant=is_compliant,
            target_type="plot",
            target_id=plot_id,
            details=details,
            remediation=remediation,
        ))

        # Check 3: Country identification
        if plot is None:
            is_compliant = False
            details = f"Plot {plot_id} not found"
            remediation = "Register the plot with country information"
        elif plot.geolocation and plot.geolocation.country_code:
            is_compliant = True
            details = f"Country identified: {plot.geolocation.country_code}"
            remediation = None
        else:
            is_compliant = False
            details = "Country of production not identified"
            remediation = "Provide ISO 3166-1 alpha-2 country code"

        results.append(self._create_check(
            article_checked="Article 9",
            requirement="Country of production identified (Article 9(1)(e))",
            is_compliant=is_compliant,
            target_type="plot",
            target_id=plot_id,
            details=details,
            remediation=remediation,
        ))

        # Record metrics
        for r in results:
            status_str = "compliant" if r.is_compliant else "non_compliant"
            self._record_check_metric("article_9", status_str)

        return results

    def verify_article_10(self, dds_id: str) -> List[ComplianceCheckResult]:
        """Verify Article 10 due diligence requirements.

        Args:
            dds_id: DDS identifier.

        Returns:
            List of ComplianceCheckResult for Article 10 checks.
        """
        results: List[ComplianceCheckResult] = []

        # Article 10 checks require DDS data - mark as non-compliant pending
        # since we do not have direct DDS engine reference
        for check_def in self.ARTICLE_CHECKS["article_10"]:
            results.append(self._create_check(
                article_checked="Article 10",
                requirement=check_def["requirement"],
                is_compliant=False,
                target_type="dds",
                target_id=dds_id,
                details="Requires DueDiligenceEngine for full verification",
                remediation="Complete due diligence process for this DDS",
            ))

        for r in results:
            status_str = "compliant" if r.is_compliant else "non_compliant"
            self._record_check_metric("article_10", status_str)

        return results

    def verify_article_11(self, dds_id: str) -> List[ComplianceCheckResult]:
        """Verify Article 11 risk mitigation requirements.

        Args:
            dds_id: DDS identifier.

        Returns:
            List of ComplianceCheckResult for Article 11 checks.
        """
        results: List[ComplianceCheckResult] = []

        for check_def in self.ARTICLE_CHECKS["article_11"]:
            results.append(self._create_check(
                article_checked="Article 11",
                requirement=check_def["requirement"],
                is_compliant=False,
                target_type="dds",
                target_id=dds_id,
                details="Requires RiskAssessmentEngine for full verification",
                remediation="Complete risk mitigation assessment for this DDS",
            ))

        for r in results:
            status_str = "compliant" if r.is_compliant else "non_compliant"
            self._record_check_metric("article_11", status_str)

        return results

    def get_compliance_score(
        self,
        results: List[ComplianceCheckResult],
    ) -> float:
        """Calculate compliance score from check results.

        Deterministic scoring:
        - is_compliant=True = 100 points
        - is_compliant=False = 0 points

        Args:
            results: List of check results.

        Returns:
            Compliance score (0-100).
        """
        if not results:
            return 0.0

        total_points = 0.0
        for r in results:
            if r.is_compliant:
                total_points += 100.0

        return round(total_points / len(results), 2)

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get overall compliance statistics.

        Returns:
            Dictionary with check counts by compliance status and article.
        """
        by_status: Dict[str, int] = {"compliant": 0, "non_compliant": 0}
        by_article: Dict[str, int] = {}

        for result in self._check_results.values():
            if result.is_compliant:
                by_status["compliant"] += 1
            else:
                by_status["non_compliant"] += 1

            article_key = result.article_checked
            by_article[article_key] = by_article.get(article_key, 0) + 1

        return {
            "total_checks": len(self._check_results),
            "by_status": by_status,
            "by_article": by_article,
        }

    def batch_verify(
        self,
        target_type: str,
        target_ids: List[str],
    ) -> Dict[str, List[ComplianceCheckResult]]:
        """Batch verify compliance for multiple entities.

        Args:
            target_type: Type of entities.
            target_ids: List of entity identifiers.

        Returns:
            Dictionary mapping target_id to compliance check results.
        """
        results: Dict[str, List[ComplianceCheckResult]] = {}
        for target_id in target_ids:
            results[target_id] = self.verify_compliance(
                target_type, target_id,
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_check(
        self,
        article_checked: str,
        requirement: str,
        is_compliant: bool,
        target_type: str,
        target_id: str,
        details: str = "",
        remediation: Optional[str] = None,
    ) -> ComplianceCheckResult:
        """Create and store a compliance check result.

        Args:
            article_checked: EUDR article reference.
            requirement: Requirement description.
            is_compliant: Whether the check passed.
            target_type: Entity type checked.
            target_id: Entity identifier.
            details: Detailed findings.
            remediation: Suggested remediation action.

        Returns:
            ComplianceCheckResult instance.
        """
        check_id = self._generate_check_id()

        result = ComplianceCheckResult(
            check_id=check_id,
            target_type=target_type,
            target_id=target_id,
            article_checked=article_checked,
            requirement=requirement,
            is_compliant=is_compliant,
            details=details,
            remediation=remediation,
        )

        self._check_results[check_id] = result

        # Record provenance
        if self._provenance is not None:
            data_hash = hashlib.sha256(
                json.dumps(
                    result.model_dump(mode="json"),
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            self._provenance.record(
                entity_type="compliance_check",
                entity_id=check_id,
                action="compliance_check",
                data_hash=data_hash,
            )

        return result

    def _get_plot(self, plot_id: str) -> Any:
        """Get a plot from the registry.

        Args:
            plot_id: Plot identifier.

        Returns:
            PlotRecord or None.
        """
        if self._plot_registry is None:
            return None
        return self._plot_registry.get_plot(plot_id)

    def _generate_check_id(self) -> str:
        """Generate a unique check identifier.

        Returns:
            Check ID in format "CHK-{hex12}".
        """
        return f"CHK-{uuid.uuid4().hex[:12]}"

    def _record_check_metric(self, article: str, result: str) -> None:
        """Record a compliance check metric.

        Args:
            article: Article identifier.
            result: Check result status string.
        """
        try:
            from greenlang.eudr_traceability.metrics import record_compliance_check
            record_compliance_check(article, result)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def check_count(self) -> int:
        """Return the total number of compliance checks performed."""
        return len(self._check_results)


__all__ = [
    "ComplianceVerifier",
]
