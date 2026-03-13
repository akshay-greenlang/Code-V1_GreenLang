# -*- coding: utf-8 -*-
"""
AGENT-EUDR-027: Information Gathering Agent - Completeness Validation Engine

Validates completeness of the 10 mandatory EUDR Article 9 information elements
required for due diligence statements. Each element (product description,
quantity, country of production, geolocation, production date range, supplier
identification, buyer identification, deforestation-free evidence, legal
production evidence, supply chain information) is scored individually and
combined into a weighted completeness score with gap analysis.

Production infrastructure includes:
    - Weighted completeness scoring with Decimal precision
    - Three-tier classification: INSUFFICIENT / PARTIAL / COMPLETE
    - Simplified due diligence mode with relaxed thresholds (Article 13)
    - Detailed gap report with per-element remediation actions
    - Gap severity classification (critical / high / medium / low)
    - Validation history tracking for audit trail
    - SHA-256 provenance hash on every completeness report

Zero-Hallucination Guarantees:
    - Completeness scores computed via deterministic weighted Decimal arithmetic
    - Classification thresholds applied as simple numeric comparisons
    - No LLM involvement in scoring, classification, or gap analysis
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 9(1): 10 mandatory information elements
    - EUDR Article 9(2): Completeness requirements for DDS submission
    - EUDR Article 10: Risk assessment requires complete information
    - EUDR Article 13: Simplified due diligence with reduced requirements
    - EUDR Article 31: 5-year record retention for validation results

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 5: Completeness Validation)
Agent ID: GL-EUDR-IGA-027
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.information_gathering.config import (
    InformationGatheringConfig,
    get_config,
)
from greenlang.agents.eudr.information_gathering.models import (
    Article9ElementName,
    Article9ElementStatus,
    CompletenessClassification,
    CompletenessReport,
    ElementStatus,
    EUDRCommodity,
    GapReport,
    GapReportItem,
)
from greenlang.agents.eudr.information_gathering.provenance import ProvenanceTracker
from greenlang.agents.eudr.information_gathering.metrics import (
    record_completeness_validation,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Simplified DD thresholds (Article 13 low-risk countries)
# ---------------------------------------------------------------------------

_SIMPLIFIED_DD_INSUFFICIENT_THRESHOLD = Decimal("50")
_SIMPLIFIED_DD_PARTIAL_THRESHOLD = Decimal("80")

# ---------------------------------------------------------------------------
# Remediation action lookup per element
# ---------------------------------------------------------------------------

_REMEDIATION_ACTIONS_MISSING: Dict[str, str] = {
    Article9ElementName.PRODUCT_DESCRIPTION.value: (
        "Request complete product description including HS/CN codes "
        "and derived product composition from the operator"
    ),
    Article9ElementName.QUANTITY.value: (
        "Obtain quantity data (weight in kg/tonnes) from commercial "
        "invoices, bills of lading, or customs declarations"
    ),
    Article9ElementName.COUNTRY_OF_PRODUCTION.value: (
        "Identify country of production through supplier declarations, "
        "certificates of origin, or customs records"
    ),
    Article9ElementName.GEOLOCATION.value: (
        "Collect GPS coordinates of production plots; use Mobile Data "
        "Collector Agent (EUDR-015) for field-level capture"
    ),
    Article9ElementName.PRODUCTION_DATE_RANGE.value: (
        "Establish production/harvest date range from supplier records, "
        "phytosanitary certificates, or processing logs"
    ),
    Article9ElementName.SUPPLIER_IDENTIFICATION.value: (
        "Obtain supplier legal entity name, registration number, and "
        "address from government or trade registries"
    ),
    Article9ElementName.BUYER_IDENTIFICATION.value: (
        "Record buyer/operator EORI number, legal entity name, and "
        "establishment address from internal records"
    ),
    Article9ElementName.DEFORESTATION_FREE_EVIDENCE.value: (
        "Obtain satellite imagery analysis, certification evidence, "
        "or field verification confirming no deforestation after 31/12/2020"
    ),
    Article9ElementName.LEGAL_PRODUCTION_EVIDENCE.value: (
        "Collect legal compliance documentation: land titles, concession "
        "permits, FLEGT licenses, or national forestry approvals"
    ),
    Article9ElementName.SUPPLY_CHAIN_INFORMATION.value: (
        "Map full supply chain from production to import using Chain of "
        "Custody Agent (EUDR-009) or supplier questionnaires"
    ),
}

_REMEDIATION_ACTIONS_PARTIAL: Dict[str, str] = {
    Article9ElementName.PRODUCT_DESCRIPTION.value: (
        "Complete missing product details: add HS code, composition "
        "breakdown, or derived product traceability"
    ),
    Article9ElementName.QUANTITY.value: (
        "Verify quantity accuracy against shipping documents; reconcile "
        "discrepancies between declared and actual volumes"
    ),
    Article9ElementName.COUNTRY_OF_PRODUCTION.value: (
        "Confirm country of production with additional source; resolve "
        "any discrepancy between supplier declaration and certificates"
    ),
    Article9ElementName.GEOLOCATION.value: (
        "Improve geolocation precision: upgrade from region-level to "
        "plot-level coordinates using GPS Coordinate Validator (EUDR-007)"
    ),
    Article9ElementName.PRODUCTION_DATE_RANGE.value: (
        "Narrow production date range with additional corroborating "
        "evidence from processing or transport records"
    ),
    Article9ElementName.SUPPLIER_IDENTIFICATION.value: (
        "Complete supplier profile: add missing registration number, "
        "alternative names, or verify address against registry"
    ),
    Article9ElementName.BUYER_IDENTIFICATION.value: (
        "Complete buyer identification: verify EORI number and update "
        "address if establishment has changed"
    ),
    Article9ElementName.DEFORESTATION_FREE_EVIDENCE.value: (
        "Strengthen deforestation-free evidence: obtain additional "
        "satellite analysis period or independent field verification"
    ),
    Article9ElementName.LEGAL_PRODUCTION_EVIDENCE.value: (
        "Supplement legal evidence: obtain additional permits or verify "
        "existing documents against national registry records"
    ),
    Article9ElementName.SUPPLY_CHAIN_INFORMATION.value: (
        "Extend supply chain mapping to additional tiers; verify "
        "intermediary identities and handling records"
    ),
}

# Severity classification per element when missing
_ELEMENT_SEVERITY: Dict[str, str] = {
    Article9ElementName.PRODUCT_DESCRIPTION.value: "high",
    Article9ElementName.QUANTITY.value: "high",
    Article9ElementName.COUNTRY_OF_PRODUCTION.value: "critical",
    Article9ElementName.GEOLOCATION.value: "critical",
    Article9ElementName.PRODUCTION_DATE_RANGE.value: "high",
    Article9ElementName.SUPPLIER_IDENTIFICATION.value: "critical",
    Article9ElementName.BUYER_IDENTIFICATION.value: "high",
    Article9ElementName.DEFORESTATION_FREE_EVIDENCE.value: "critical",
    Article9ElementName.LEGAL_PRODUCTION_EVIDENCE.value: "critical",
    Article9ElementName.SUPPLY_CHAIN_INFORMATION.value: "high",
}

# Estimated effort per element when missing
_ELEMENT_EFFORT: Dict[str, str] = {
    Article9ElementName.PRODUCT_DESCRIPTION.value: "1-2 days",
    Article9ElementName.QUANTITY.value: "1 day",
    Article9ElementName.COUNTRY_OF_PRODUCTION.value: "1-3 days",
    Article9ElementName.GEOLOCATION.value: "3-7 days",
    Article9ElementName.PRODUCTION_DATE_RANGE.value: "1-2 days",
    Article9ElementName.SUPPLIER_IDENTIFICATION.value: "2-5 days",
    Article9ElementName.BUYER_IDENTIFICATION.value: "1 day",
    Article9ElementName.DEFORESTATION_FREE_EVIDENCE.value: "5-14 days",
    Article9ElementName.LEGAL_PRODUCTION_EVIDENCE.value: "5-14 days",
    Article9ElementName.SUPPLY_CHAIN_INFORMATION.value: "7-21 days",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class CompletenessValidationEngine:
    """Engine for validating completeness of EUDR Article 9 information elements.

    Computes weighted completeness scores, classifies information sufficiency
    as INSUFFICIENT / PARTIAL / COMPLETE, and generates detailed gap reports
    with per-element remediation actions for missing or partial data.

    Supports simplified due diligence mode (Article 13) with relaxed
    thresholds for products sourced from EC-benchmarked low-risk countries.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> engine = CompletenessValidationEngine()
        >>> elements = {
        ...     "product_description": Article9ElementStatus(
        ...         element_name="product_description",
        ...         status=ElementStatus.COMPLETE,
        ...         confidence=Decimal("0.95"),
        ...     ),
        ... }
        >>> report = engine.validate_completeness("OP-001", EUDRCommodity.COFFEE, elements)
        >>> assert report.completeness_classification in CompletenessClassification
    """

    def __init__(self, config: Optional[InformationGatheringConfig] = None) -> None:
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._validation_history: List[CompletenessReport] = []
        logger.info(
            "CompletenessValidationEngine initialized "
            "(insufficient_threshold=%s, partial_threshold=%s, "
            "simplified_dd=%s, elements=%d)",
            self._config.insufficient_threshold,
            self._config.partial_threshold,
            self._config.simplified_dd_enabled,
            len(self._config.element_weights),
        )

    def validate_completeness(
        self,
        operation_id: str,
        commodity: EUDRCommodity,
        elements: Dict[str, Article9ElementStatus],
        is_simplified_dd: bool = False,
    ) -> CompletenessReport:
        """Validate completeness of Article 9 information elements.

        Computes a weighted completeness score from individual element
        scores, classifies the overall completeness, and generates a
        gap report with remediation actions for deficient elements.

        Args:
            operation_id: Unique operation identifier.
            commodity: EUDR regulated commodity.
            elements: Dict mapping element name to Article9ElementStatus.
            is_simplified_dd: If True, apply relaxed thresholds for
                simplified due diligence (Article 13).

        Returns:
            CompletenessReport with score, classification, and gap report.
        """
        start_time = time.monotonic()

        # Apply simplified DD only if enabled in config
        use_simplified = is_simplified_dd and self._config.simplified_dd_enabled

        # Compute weighted completeness score
        completeness_score = self._compute_weighted_score(elements)

        # Classify completeness
        classification = self.classify_completeness(completeness_score, use_simplified)

        # Generate gap report
        gap_report = self.generate_gap_report(elements)

        # Build element status list
        element_list: List[Article9ElementStatus] = []
        for elem_name in Article9ElementName:
            if elem_name.value in elements:
                element_list.append(elements[elem_name.value])
            else:
                # Element not provided at all -> MISSING
                element_list.append(
                    Article9ElementStatus(
                        element_name=elem_name.value,
                        status=ElementStatus.MISSING,
                        source="",
                        value_summary="Not provided",
                        confidence=Decimal("0"),
                        last_updated=_utcnow(),
                    )
                )

        # Provenance hash
        provenance_hash = _compute_hash({
            "operation_id": operation_id,
            "commodity": commodity.value,
            "completeness_score": str(completeness_score),
            "classification": classification.value,
            "total_gaps": gap_report.total_gaps,
        })

        # Build report
        report = CompletenessReport(
            operation_id=operation_id,
            commodity=commodity,
            elements=element_list,
            completeness_score=completeness_score,
            completeness_classification=classification,
            gap_report=gap_report,
            is_simplified_dd=use_simplified,
            validated_at=_utcnow(),
            provenance_hash=provenance_hash,
        )

        # Track in history
        self._validation_history.append(report)

        # Provenance chain entry
        self._provenance.create_entry(
            step="completeness_validation",
            source="article_9_elements",
            input_hash=_compute_hash({
                "operation_id": operation_id,
                "elements": {k: v.status.value for k, v in elements.items()},
            }),
            output_hash=provenance_hash,
        )

        # Metrics
        record_completeness_validation(classification.value)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Completeness validation for %s (%s): score=%s%%, "
            "classification=%s, gaps=%d, simplified_dd=%s (%.0fms)",
            operation_id,
            commodity.value,
            completeness_score,
            classification.value,
            gap_report.total_gaps,
            use_simplified,
            elapsed * 1000,
        )
        return report

    def compute_element_score(self, element: Article9ElementStatus) -> Decimal:
        """Compute the contribution score for a single Article 9 element.

        Score is based on element status multiplied by confidence:
            - COMPLETE: 1.0 * confidence
            - PARTIAL:  0.5 * confidence
            - MISSING:  0.0

        Args:
            element: Article 9 element status.

        Returns:
            Element score as Decimal in range [0.0, 1.0].
        """
        if element.status == ElementStatus.COMPLETE:
            base = Decimal("1.0")
        elif element.status == ElementStatus.PARTIAL:
            base = Decimal("0.5")
        else:
            return Decimal("0")

        confidence = element.confidence if element.confidence > Decimal("0") else Decimal("1.0")
        score = (base * confidence).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return min(score, Decimal("1.0"))

    def _compute_weighted_score(
        self,
        elements: Dict[str, Article9ElementStatus],
    ) -> Decimal:
        """Compute weighted completeness score across all Article 9 elements.

        Each element's score is multiplied by its configured weight and
        summed. Missing elements (not in the dict) contribute 0.

        Args:
            elements: Dict mapping element name to status.

        Returns:
            Weighted completeness score as Decimal percentage (0-100).
        """
        weights = self._config.element_weights
        total_weighted = Decimal("0")
        total_weight = Decimal("0")

        for elem_name in Article9ElementName:
            weight = weights.get(elem_name.value, Decimal("0.10"))
            total_weight += weight

            if elem_name.value in elements:
                elem_score = self.compute_element_score(elements[elem_name.value])
                total_weighted += weight * elem_score

        if total_weight == Decimal("0"):
            return Decimal("0")

        # Normalize to percentage
        score = (total_weighted / total_weight * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return min(score, Decimal("100.00"))

    def classify_completeness(
        self,
        score: Decimal,
        is_simplified_dd: bool = False,
    ) -> CompletenessClassification:
        """Classify completeness score into INSUFFICIENT / PARTIAL / COMPLETE.

        Standard thresholds (from config):
            - score < insufficient_threshold (60): INSUFFICIENT
            - score < partial_threshold (90): PARTIAL
            - score >= partial_threshold: COMPLETE

        Simplified DD thresholds (Article 13):
            - score < 50: INSUFFICIENT
            - score < 80: PARTIAL
            - score >= 80: COMPLETE

        Args:
            score: Completeness percentage (0-100).
            is_simplified_dd: If True, apply relaxed thresholds.

        Returns:
            CompletenessClassification enum value.
        """
        if is_simplified_dd:
            insufficient_threshold = _SIMPLIFIED_DD_INSUFFICIENT_THRESHOLD
            partial_threshold = _SIMPLIFIED_DD_PARTIAL_THRESHOLD
        else:
            insufficient_threshold = self._config.insufficient_threshold
            partial_threshold = self._config.partial_threshold

        if score < insufficient_threshold:
            return CompletenessClassification.INSUFFICIENT
        elif score < partial_threshold:
            return CompletenessClassification.PARTIAL
        else:
            return CompletenessClassification.COMPLETE

    def generate_gap_report(
        self,
        elements: Dict[str, Article9ElementStatus],
    ) -> GapReport:
        """Generate a detailed gap report for missing or partial elements.

        Scans all 10 Article 9 elements and creates a GapReportItem for
        each that is MISSING or PARTIAL, including severity classification,
        remediation action, and estimated effort.

        Args:
            elements: Dict mapping element name to status.

        Returns:
            GapReport with itemized gaps and severity counts.
        """
        items: List[GapReportItem] = []

        for elem_name in Article9ElementName:
            element = elements.get(elem_name.value)

            if element is None:
                # Element not provided -> treat as missing
                items.append(self._create_gap_item(
                    elem_name.value,
                    gap_type="missing",
                    status=ElementStatus.MISSING,
                ))
            elif element.status == ElementStatus.MISSING:
                items.append(self._create_gap_item(
                    elem_name.value,
                    gap_type="missing",
                    status=ElementStatus.MISSING,
                ))
            elif element.status == ElementStatus.PARTIAL:
                items.append(self._create_gap_item(
                    elem_name.value,
                    gap_type="partial",
                    status=ElementStatus.PARTIAL,
                ))
            # COMPLETE elements have no gap

        # Count severity levels
        critical = sum(1 for item in items if item.severity == "critical")
        high = sum(1 for item in items if item.severity == "high")
        medium = sum(1 for item in items if item.severity == "medium")
        low = sum(1 for item in items if item.severity == "low")

        report = GapReport(
            total_gaps=len(items),
            critical_gaps=critical,
            high_gaps=high,
            medium_gaps=medium,
            low_gaps=low,
            items=items,
            generated_at=_utcnow(),
        )

        if items:
            logger.warning(
                "Gap report: %d gaps (critical=%d, high=%d, medium=%d, low=%d)",
                len(items),
                critical,
                high,
                medium,
                low,
            )
        return report

    def _create_gap_item(
        self,
        element_name: str,
        gap_type: str,
        status: ElementStatus,
    ) -> GapReportItem:
        """Create a single gap report item with remediation details.

        Args:
            element_name: Article 9 element name.
            gap_type: Gap type string (missing or partial).
            status: Current element status.

        Returns:
            Populated GapReportItem.
        """
        remediation = self.get_remediation_action(element_name, status)
        severity = _ELEMENT_SEVERITY.get(element_name, "medium")
        effort = _ELEMENT_EFFORT.get(element_name, "unknown")

        # Partial elements have lower severity than missing
        if status == ElementStatus.PARTIAL and severity == "critical":
            severity = "high"
        elif status == ElementStatus.PARTIAL and severity == "high":
            severity = "medium"

        return GapReportItem(
            element_name=element_name,
            gap_type=gap_type,
            severity=severity,
            remediation_action=remediation,
            estimated_effort=effort,
        )

    def get_remediation_action(
        self,
        element_name: str,
        status: ElementStatus,
    ) -> str:
        """Return a specific remediation action for an element and status.

        Provides actionable guidance tailored to whether the element is
        completely missing or only partially populated.

        Args:
            element_name: Article 9 element name.
            status: Current element status (MISSING or PARTIAL).

        Returns:
            Human-readable remediation action string.
        """
        if status == ElementStatus.MISSING:
            return _REMEDIATION_ACTIONS_MISSING.get(
                element_name,
                f"Collect {element_name} data from available sources",
            )
        elif status == ElementStatus.PARTIAL:
            return _REMEDIATION_ACTIONS_PARTIAL.get(
                element_name,
                f"Complete partial {element_name} data with additional sources",
            )
        return ""

    def get_validation_stats(self) -> Dict[str, Any]:
        """Return completeness validation engine statistics.

        Returns:
            Dict with total_validations, classification_breakdown,
            average_score, and simplified_dd_count keys.
        """
        classification_counts: Dict[str, int] = {}
        simplified_count = 0
        total_score = Decimal("0")

        for report in self._validation_history:
            cls_key = report.completeness_classification.value
            classification_counts[cls_key] = classification_counts.get(cls_key, 0) + 1
            total_score += report.completeness_score
            if report.is_simplified_dd:
                simplified_count += 1

        total = len(self._validation_history)
        avg_score = (
            (total_score / Decimal(str(total))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if total > 0
            else Decimal("0")
        )

        return {
            "total_validations": total,
            "classification_breakdown": classification_counts,
            "average_score": float(avg_score),
            "simplified_dd_count": simplified_count,
        }

    def clear_history(self) -> None:
        """Clear validation history (for testing)."""
        self._validation_history.clear()
        logger.info("Completeness validation history cleared")
