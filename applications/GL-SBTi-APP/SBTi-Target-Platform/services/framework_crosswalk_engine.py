"""
Framework Crosswalk Engine -- Cross-Framework Alignment Mapping

Implements cross-framework alignment between SBTi requirements and
external regulatory/disclosure frameworks: CDP Climate Change, TCFD,
CSRD/ESRS E1, GHG Protocol, ISO 14064, SB 253, NZBA, and GFANZ.

Generates detailed alignment mappings showing which SBTi data and
criteria satisfy requirements in other frameworks, identifies gaps,
and produces multi-framework compliance reports.

All logic is deterministic (zero-hallucination).

Reference:
    - SBTi Corporate Net-Zero Standard v1.2 (2023)
    - CDP Climate Change Questionnaire 2024
    - TCFD Recommendations (2017), ISSB IFRS S2 (2023)
    - CSRD / ESRS E1 Climate Change (2023)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - ISO 14064-1:2018
    - California SB 253 Climate Corporate Data Accountability Act
    - Net-Zero Banking Alliance (NZBA) Commitment

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = FrameworkCrosswalkEngine(SBTiAppConfig())
    >>> mapping = engine.generate_mapping("org-1", "cdp")
    >>> print(mapping.coverage_pct)
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    CROSS_FRAMEWORK_ALIGNMENT,
    FRAMEWORK_MAPPING_REFS,
    FrameworkType,
    SBTiAppConfig,
)
from .models import (
    AlignmentItem,
    FrameworkMapping,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AlignmentDetail(BaseModel):
    """Detailed alignment result for a single requirement pair."""

    sbti_reference: str = Field(...)
    framework_reference: str = Field(...)
    status: str = Field(
        default="aligned",
        description="fully_aligned, partially_aligned, gap, not_applicable",
    )
    description: str = Field(default="")
    data_available: bool = Field(default=False)
    recommendation: Optional[str] = Field(None)


class FrameworkAlignmentResult(BaseModel):
    """Complete alignment result for a single framework."""

    org_id: str = Field(...)
    framework: str = Field(...)
    framework_name: str = Field(default="")
    total_requirements: int = Field(default=0)
    fully_aligned: int = Field(default=0)
    partially_aligned: int = Field(default=0)
    gaps: int = Field(default=0)
    not_applicable: int = Field(default=0)
    coverage_pct: float = Field(default=0.0)
    alignment_items: List[AlignmentDetail] = Field(default_factory=list)
    gap_items: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class MultiFrameworkResult(BaseModel):
    """Alignment results across multiple frameworks."""

    org_id: str = Field(...)
    frameworks_assessed: int = Field(default=0)
    overall_coverage_pct: float = Field(default=0.0)
    framework_results: List[Dict[str, Any]] = Field(default_factory=list)
    common_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    priority_actions: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class DataReuseSummary(BaseModel):
    """Summary of SBTi data reusable across frameworks."""

    org_id: str = Field(...)
    sbti_data_elements: List[Dict[str, Any]] = Field(default_factory=list)
    frameworks_served: Dict[str, List[str]] = Field(default_factory=dict)
    reuse_efficiency_pct: float = Field(default=0.0)
    message: str = Field(default="")


class FrameworkComparisonResult(BaseModel):
    """Side-by-side comparison of framework requirements."""

    requirement_area: str = Field(...)
    frameworks: Dict[str, str] = Field(default_factory=dict)
    sbti_status: str = Field(default="")
    notes: str = Field(default="")


# ---------------------------------------------------------------------------
# FrameworkCrosswalkEngine
# ---------------------------------------------------------------------------

class FrameworkCrosswalkEngine:
    """
    Cross-framework alignment mapping engine.

    Maps SBTi requirements and data to external framework requirements
    (CDP, TCFD, CSRD, GHG Protocol, ISO 14064, SB 253, NZBA, GFANZ).
    Identifies full alignment, partial alignment, and gaps. Generates
    multi-framework compliance reports and data reuse assessments.

    Attributes:
        config: Application configuration.
        _org_data: In-memory organization data availability keyed by org_id.
        _mappings: Computed mappings keyed by (org_id, framework).

    Example:
        >>> engine = FrameworkCrosswalkEngine(SBTiAppConfig())
        >>> mapping = engine.generate_mapping("org-1", "cdp")
    """

    # Supported frameworks
    SUPPORTED_FRAMEWORKS: List[str] = [
        "cdp", "tcfd", "csrd", "ghg_protocol", "iso14064", "sb253", "nzba", "gfanz",
    ]

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the FrameworkCrosswalkEngine."""
        self.config = config or SBTiAppConfig()
        self._org_data: Dict[str, Dict[str, bool]] = {}
        self._mappings: Dict[str, FrameworkMapping] = {}
        logger.info(
            "FrameworkCrosswalkEngine initialized with %d frameworks",
            len(self.SUPPORTED_FRAMEWORKS),
        )

    # ------------------------------------------------------------------
    # Data Registration
    # ------------------------------------------------------------------

    def register_org_data_availability(
        self, org_id: str, data_flags: Dict[str, bool],
    ) -> None:
        """
        Register data availability flags for an organization.

        Args:
            org_id: Organization identifier.
            data_flags: Dict with keys indicating data availability, e.g.:
                - has_scope1_inventory: bool
                - has_scope2_inventory: bool
                - has_scope3_screening: bool
                - has_near_term_target: bool
                - has_long_term_target: bool
                - has_net_zero_target: bool
                - has_transition_plan: bool
                - has_verification: bool
                - has_recalculation_policy: bool
                - has_progress_data: bool
        """
        self._org_data[org_id] = data_flags
        logger.info(
            "Registered data availability for org %s: %d flags",
            org_id, len(data_flags),
        )

    # ------------------------------------------------------------------
    # Single Framework Mapping
    # ------------------------------------------------------------------

    def generate_mapping(
        self, org_id: str, framework: str,
    ) -> FrameworkAlignmentResult:
        """
        Generate alignment mapping for a specific framework.

        Maps SBTi requirements and data against the framework's
        requirements, determining full alignment, partial alignment,
        or gaps for each requirement pair.

        Args:
            org_id: Organization identifier.
            framework: Framework key (cdp, tcfd, csrd, etc.).

        Returns:
            FrameworkAlignmentResult with detailed alignment items.

        Raises:
            ValueError: If framework is not supported.
        """
        start = datetime.utcnow()

        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework: {framework}. "
                f"Valid: {self.SUPPORTED_FRAMEWORKS}"
            )

        alignment_data = CROSS_FRAMEWORK_ALIGNMENT.get(framework)
        if alignment_data is None:
            return FrameworkAlignmentResult(
                org_id=org_id,
                framework=framework,
                message=f"No alignment data available for {framework}.",
            )

        framework_name = alignment_data.get("framework_name", framework)
        items = alignment_data.get("alignment_items", [])
        org_flags = self._org_data.get(org_id, {})

        alignment_items: List[AlignmentDetail] = []
        fully_aligned = 0
        partially_aligned = 0
        gaps = 0
        not_applicable = 0
        gap_items: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        for item in items:
            sbti_ref = item.get("sbti_ref", "")
            fw_ref = item.get("framework_ref", "")
            raw_status = item.get("status", "fully_aligned")

            # Check data availability to determine actual status
            data_available = self._check_data_for_requirement(
                org_flags, sbti_ref,
            )

            if raw_status == "fully_aligned" and data_available:
                status = "fully_aligned"
                fully_aligned += 1
            elif raw_status == "fully_aligned" and not data_available:
                status = "gap"
                gaps += 1
                gap_items.append({
                    "sbti_ref": sbti_ref,
                    "framework_ref": fw_ref,
                    "reason": "SBTi data not yet available",
                })
                recommendations.append(
                    f"Complete {sbti_ref} to satisfy {framework} requirement {fw_ref}."
                )
            elif raw_status == "partially_aligned":
                status = "partially_aligned"
                partially_aligned += 1
                if not data_available:
                    recommendations.append(
                        f"Partially aligned for {fw_ref}. "
                        f"Complete {sbti_ref} for full alignment."
                    )
            else:
                status = raw_status
                if status == "not_applicable":
                    not_applicable += 1
                else:
                    gaps += 1

            alignment_items.append(AlignmentDetail(
                sbti_reference=sbti_ref,
                framework_reference=fw_ref,
                status=status,
                description=f"SBTi '{sbti_ref}' maps to {framework} '{fw_ref}'",
                data_available=data_available,
                recommendation=(
                    f"Complete {sbti_ref}" if status == "gap" else None
                ),
            ))

        total = len(alignment_items)
        applicable = total - not_applicable
        coverage = (
            ((fully_aligned + partially_aligned * 0.5) / applicable * 100.0)
            if applicable > 0 else 0.0
        )

        # Store domain model mapping
        domain_items = [
            AlignmentItem(
                sbti_reference=ai.sbti_reference,
                framework_reference=ai.framework_reference,
                status=ai.status,
                description=ai.description,
            )
            for ai in alignment_items
        ]
        domain_mapping = FrameworkMapping(
            tenant_id="default",
            org_id=org_id,
            framework=framework,
            framework_name=framework_name,
            alignment_items=domain_items,
        )
        self._mappings[f"{org_id}:{framework}"] = domain_mapping

        provenance = _sha256(
            f"crosswalk:{org_id}:{framework}:{coverage}"
        )

        result = FrameworkAlignmentResult(
            org_id=org_id,
            framework=framework,
            framework_name=framework_name,
            total_requirements=total,
            fully_aligned=fully_aligned,
            partially_aligned=partially_aligned,
            gaps=gaps,
            not_applicable=not_applicable,
            coverage_pct=round(coverage, 2),
            alignment_items=alignment_items,
            gap_items=gap_items,
            recommendations=recommendations,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Framework mapping for org %s -> %s: coverage=%.1f%% "
            "(%d aligned, %d partial, %d gaps) in %.1f ms",
            org_id, framework, coverage, fully_aligned, partially_aligned,
            gaps, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Multi-Framework Assessment
    # ------------------------------------------------------------------

    def generate_multi_framework_assessment(
        self,
        org_id: str,
        frameworks: Optional[List[str]] = None,
    ) -> MultiFrameworkResult:
        """
        Generate alignment assessment across multiple frameworks.

        Evaluates the organization against all specified frameworks,
        identifies common gaps, and generates priority actions.

        Args:
            org_id: Organization identifier.
            frameworks: List of framework keys to assess (default: all).

        Returns:
            MultiFrameworkResult with aggregated analysis.
        """
        start = datetime.utcnow()

        if frameworks is None:
            frameworks = self.SUPPORTED_FRAMEWORKS

        framework_results: List[Dict[str, Any]] = []
        all_gaps: Dict[str, int] = {}
        total_coverage = 0.0

        for fw in frameworks:
            try:
                result = self.generate_mapping(org_id, fw)
                framework_results.append({
                    "framework": fw,
                    "framework_name": result.framework_name,
                    "coverage_pct": result.coverage_pct,
                    "fully_aligned": result.fully_aligned,
                    "gaps": result.gaps,
                    "total": result.total_requirements,
                })
                total_coverage += result.coverage_pct

                # Track common gaps
                for gap in result.gap_items:
                    ref = gap.get("sbti_ref", "unknown")
                    all_gaps[ref] = all_gaps.get(ref, 0) + 1
            except ValueError:
                logger.warning("Skipping unsupported framework: %s", fw)

        n = len(framework_results)
        overall_coverage = total_coverage / n if n > 0 else 0.0

        # Common gaps (appearing in multiple frameworks)
        common_gaps = [
            {"sbti_ref": ref, "frameworks_affected": count}
            for ref, count in sorted(all_gaps.items(), key=lambda x: -x[1])
            if count > 1
        ]

        # Priority actions based on gap frequency
        priority_actions: List[Dict[str, Any]] = []
        for i, gap in enumerate(common_gaps[:5], 1):
            priority_actions.append({
                "priority": i,
                "action": f"Complete '{gap['sbti_ref']}'",
                "impact": f"Closes gap in {gap['frameworks_affected']} frameworks",
            })

        provenance = _sha256(
            f"multi_crosswalk:{org_id}:{overall_coverage}:{n}"
        )

        result = MultiFrameworkResult(
            org_id=org_id,
            frameworks_assessed=n,
            overall_coverage_pct=round(overall_coverage, 2),
            framework_results=framework_results,
            common_gaps=common_gaps,
            priority_actions=priority_actions,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Multi-framework assessment for org %s: %d frameworks, "
            "overall=%.1f%%, %d common gaps in %.1f ms",
            org_id, n, overall_coverage, len(common_gaps), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Data Reuse Analysis
    # ------------------------------------------------------------------

    def analyze_data_reuse(self, org_id: str) -> DataReuseSummary:
        """
        Analyze SBTi data reusability across frameworks.

        Identifies which SBTi data elements can be reused for reporting
        to other frameworks, reducing redundant data collection.

        Args:
            org_id: Organization identifier.

        Returns:
            DataReuseSummary with reuse mapping and efficiency.
        """
        # Define SBTi data elements and which frameworks they serve
        data_elements = [
            {
                "element": "Scope 1 Emissions Inventory",
                "sbti_ref": "Emissions inventory",
                "frameworks": ["cdp", "tcfd", "csrd", "ghg_protocol", "iso14064", "sb253"],
            },
            {
                "element": "Scope 2 Emissions (Location & Market)",
                "sbti_ref": "Emissions inventory",
                "frameworks": ["cdp", "tcfd", "csrd", "ghg_protocol", "iso14064", "sb253"],
            },
            {
                "element": "Scope 3 Category Breakdown",
                "sbti_ref": "Scope 3 screening (C4)",
                "frameworks": ["cdp", "csrd", "ghg_protocol", "sb253"],
            },
            {
                "element": "Near-Term Target Details",
                "sbti_ref": "Near-term target (C3)",
                "frameworks": ["cdp", "tcfd", "csrd"],
            },
            {
                "element": "Long-Term / Net-Zero Target",
                "sbti_ref": "Near-term/long-term targets",
                "frameworks": ["cdp", "tcfd", "csrd", "nzba", "gfanz"],
            },
            {
                "element": "Target Progress Data",
                "sbti_ref": "Progress tracking",
                "frameworks": ["cdp", "csrd"],
            },
            {
                "element": "Transition Plan",
                "sbti_ref": "Transition plan / pathway",
                "frameworks": ["tcfd", "csrd", "gfanz"],
            },
            {
                "element": "Verification / Assurance",
                "sbti_ref": "Verification (optional for SBTi)",
                "frameworks": ["iso14064", "sb253"],
            },
            {
                "element": "Base Year Recalculation Policy",
                "sbti_ref": "Base year recalculation (C8)",
                "frameworks": ["ghg_protocol"],
            },
        ]

        org_flags = self._org_data.get(org_id, {})

        frameworks_served: Dict[str, List[str]] = {}
        available_elements = []

        for element in data_elements:
            sbti_ref = element["sbti_ref"]
            available = self._check_data_for_requirement(org_flags, sbti_ref)
            element["available"] = available

            if available:
                available_elements.append(element)
                for fw in element["frameworks"]:
                    frameworks_served.setdefault(fw, []).append(element["element"])

        total_elements = len(data_elements)
        available_count = len(available_elements)
        reuse_efficiency = (available_count / total_elements * 100.0) if total_elements > 0 else 0.0

        return DataReuseSummary(
            org_id=org_id,
            sbti_data_elements=data_elements,
            frameworks_served=frameworks_served,
            reuse_efficiency_pct=round(reuse_efficiency, 2),
            message=(
                f"{available_count} of {total_elements} SBTi data elements available. "
                f"Reuse efficiency: {reuse_efficiency:.0f}%. "
                f"Serving {len(frameworks_served)} frameworks."
            ),
        )

    # ------------------------------------------------------------------
    # Framework Comparison
    # ------------------------------------------------------------------

    def compare_frameworks(
        self,
        framework_a: str,
        framework_b: str,
    ) -> List[FrameworkComparisonResult]:
        """
        Generate a side-by-side comparison of two framework requirements.

        Args:
            framework_a: First framework key.
            framework_b: Second framework key.

        Returns:
            List of FrameworkComparisonResult for each requirement area.
        """
        data_a = CROSS_FRAMEWORK_ALIGNMENT.get(framework_a, {})
        data_b = CROSS_FRAMEWORK_ALIGNMENT.get(framework_b, {})

        items_a = {
            item.get("sbti_ref", ""): item
            for item in data_a.get("alignment_items", [])
        }
        items_b = {
            item.get("sbti_ref", ""): item
            for item in data_b.get("alignment_items", [])
        }

        all_refs = set(list(items_a.keys()) + list(items_b.keys()))
        comparisons: List[FrameworkComparisonResult] = []

        for ref in sorted(all_refs):
            a_ref = items_a.get(ref, {}).get("framework_ref", "N/A")
            b_ref = items_b.get(ref, {}).get("framework_ref", "N/A")
            a_status = items_a.get(ref, {}).get("status", "not_covered")
            b_status = items_b.get(ref, {}).get("status", "not_covered")

            comparisons.append(FrameworkComparisonResult(
                requirement_area=ref,
                frameworks={
                    framework_a: a_ref,
                    framework_b: b_ref,
                },
                sbti_status=f"{framework_a}: {a_status}, {framework_b}: {b_status}",
                notes=(
                    "Covered in both frameworks"
                    if a_ref != "N/A" and b_ref != "N/A"
                    else f"Only in {framework_a}" if b_ref == "N/A"
                    else f"Only in {framework_b}"
                ),
            ))

        return comparisons

    # ------------------------------------------------------------------
    # Retrieve Stored Mapping
    # ------------------------------------------------------------------

    def get_mapping(
        self, org_id: str, framework: str,
    ) -> Optional[FrameworkMapping]:
        """
        Retrieve a stored framework mapping.

        Args:
            org_id: Organization identifier.
            framework: Framework key.

        Returns:
            FrameworkMapping domain model or None.
        """
        return self._mappings.get(f"{org_id}:{framework}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_data_for_requirement(
        self, org_flags: Dict[str, bool], sbti_ref: str,
    ) -> bool:
        """
        Check if organization has data to satisfy an SBTi requirement.

        Maps SBTi references to organization data availability flags.

        Args:
            org_flags: Organization data availability flags.
            sbti_ref: SBTi reference string.

        Returns:
            True if data is available.
        """
        if not org_flags:
            return False

        ref_lower = sbti_ref.lower()

        # Map SBTi references to data flags
        mapping = {
            "emissions inventory": "has_scope1_inventory",
            "scope 1": "has_scope1_inventory",
            "scope 2": "has_scope2_inventory",
            "scope 3 screening": "has_scope3_screening",
            "scope 3": "has_scope3_screening",
            "near-term": "has_near_term_target",
            "long-term": "has_long_term_target",
            "net-zero": "has_net_zero_target",
            "transition plan": "has_transition_plan",
            "verification": "has_verification",
            "recalculation": "has_recalculation_policy",
            "progress": "has_progress_data",
            "sbti validation": "has_near_term_target",
            "scenario": "has_transition_plan",
            "portfolio": "has_near_term_target",
            "financed": "has_scope3_screening",
        }

        for keyword, flag_key in mapping.items():
            if keyword in ref_lower:
                return org_flags.get(flag_key, False)

        return False
