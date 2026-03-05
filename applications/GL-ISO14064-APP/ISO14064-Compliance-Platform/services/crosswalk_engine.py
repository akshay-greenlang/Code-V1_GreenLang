"""
Crosswalk Engine -- ISO 14064-1:2018 to GHG Protocol Scope Mapping

Maps between the 6 ISO 14064-1 categories and the GHG Protocol
Scope 1/2/3 model for dual-compliance reporting.

Mapping:
  - Category 1 (Direct)       -> Scope 1
  - Category 2 (Energy)       -> Scope 2
  - Category 3 (Transport)    -> Scope 3 (Cat 4, 6, 7, 9)
  - Category 4 (Products In)  -> Scope 3 (Cat 1, 2, 3, 5, 8)
  - Category 5 (Products Out) -> Scope 3 (Cat 10, 11, 12, 13, 14)
  - Category 6 (Other)        -> Scope 3 (Cat 15)

Key features:
  - Bidirectional crosswalk mapping (ISO -> GHG Protocol and reverse)
  - Gap analysis between frameworks
  - Dual-standard compliance check
  - Reconciliation report with difference explanation
  - Human-readable comparison table generation

Reference: ISO 14064-1:2018, GHG Protocol Corporate Standard (2004, revised 2015).

Example:
    >>> engine = CrosswalkEngine(config)
    >>> result = engine.generate_crosswalk("inv-001", category_results)
    >>> result.reconciliation_difference
    Decimal('0')
    >>> gaps = engine.gap_analysis("inv-001")
    >>> compliance = engine.dual_standard_compliance_check("inv-001")
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ISO14064AppConfig,
    ISO_CATEGORY_NAMES,
    ISOCategory,
    SignificanceLevel,
)
from .models import (
    CategoryResult,
    CrossWalkMapping,
    CrossWalkResult,
    SignificanceAssessment,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISO 14064-1 Category -> GHG Protocol Scope
# ---------------------------------------------------------------------------

_ISO_TO_SCOPE: Dict[ISOCategory, str] = {
    ISOCategory.CATEGORY_1_DIRECT: "scope_1",
    ISOCategory.CATEGORY_2_ENERGY: "scope_2",
    ISOCategory.CATEGORY_3_TRANSPORT: "scope_3",
    ISOCategory.CATEGORY_4_PRODUCTS_USED: "scope_3",
    ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: "scope_3",
    ISOCategory.CATEGORY_6_OTHER: "scope_3",
}

# Detailed GHG Protocol Scope 3 category mapping
_ISO_TO_GHG_CATEGORIES: Dict[ISOCategory, List[str]] = {
    ISOCategory.CATEGORY_1_DIRECT: [],
    ISOCategory.CATEGORY_2_ENERGY: [],
    ISOCategory.CATEGORY_3_TRANSPORT: [
        "Cat 4 - Upstream Transportation",
        "Cat 6 - Business Travel",
        "Cat 7 - Employee Commuting",
        "Cat 9 - Downstream Transportation",
    ],
    ISOCategory.CATEGORY_4_PRODUCTS_USED: [
        "Cat 1 - Purchased Goods & Services",
        "Cat 2 - Capital Goods",
        "Cat 3 - Fuel & Energy Activities",
        "Cat 5 - Waste Generated",
        "Cat 8 - Upstream Leased Assets",
    ],
    ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: [
        "Cat 10 - Processing of Sold Products",
        "Cat 11 - Use of Sold Products",
        "Cat 12 - End-of-Life Treatment",
        "Cat 13 - Downstream Leased Assets",
        "Cat 14 - Franchises",
    ],
    ISOCategory.CATEGORY_6_OTHER: [
        "Cat 15 - Investments",
    ],
}

# Reverse mapping: GHG Protocol Scope 3 categories -> ISO 14064-1 category
_GHG_CAT_TO_ISO: Dict[str, ISOCategory] = {
    "Cat 1": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "Cat 2": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "Cat 3": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "Cat 4": ISOCategory.CATEGORY_3_TRANSPORT,
    "Cat 5": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "Cat 6": ISOCategory.CATEGORY_3_TRANSPORT,
    "Cat 7": ISOCategory.CATEGORY_3_TRANSPORT,
    "Cat 8": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "Cat 9": ISOCategory.CATEGORY_3_TRANSPORT,
    "Cat 10": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "Cat 11": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "Cat 12": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "Cat 13": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "Cat 14": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "Cat 15": ISOCategory.CATEGORY_6_OTHER,
}

# GHG Protocol mandatory vs optional requirements
_GHG_PROTOCOL_REQUIREMENTS: Dict[str, str] = {
    "scope_1": "mandatory",
    "scope_2": "mandatory",
    "scope_3": "recommended",
}

# Framework difference notes for gap analysis
_FRAMEWORK_DIFFERENCES: List[Dict[str, str]] = [
    {
        "area": "Removals",
        "iso_14064": "Removals quantified and reported within Category 1",
        "ghg_protocol": "Removals reported separately outside scopes",
        "impact": "Net emissions may differ depending on treatment of removals",
    },
    {
        "area": "Biogenic CO2",
        "iso_14064": "Reported separately (mandatory)",
        "ghg_protocol": "Reported separately (recommended in Scope 3 Standard)",
        "impact": "Both require separate reporting but with different emphasis",
    },
    {
        "area": "Scope 2 Dual Reporting",
        "iso_14064": "Single Category 2 (combined energy indirect)",
        "ghg_protocol": "Dual reporting required (location + market-based)",
        "impact": "GHG Protocol requires two figures for Scope 2",
    },
    {
        "area": "Indirect Categories",
        "iso_14064": "6 categories (activity-based grouping)",
        "ghg_protocol": "15 Scope 3 categories (value chain-based)",
        "impact": "ISO categories aggregate multiple GHG Protocol categories",
    },
    {
        "area": "Significance Assessment",
        "iso_14064": "Required for indirect categories (Clause 5.2.2)",
        "ghg_protocol": "Relevance test for Scope 3 categories",
        "impact": "Both allow exclusion with justification but use different criteria",
    },
    {
        "area": "Base Year",
        "iso_14064": "Required with recalculation policy (Clause 5.3)",
        "ghg_protocol": "Required with recalculation policy",
        "impact": "Both require base year; recalculation triggers may differ",
    },
]


class CrosswalkEngine:
    """
    Generates crosswalk between ISO 14064-1 and GHG Protocol frameworks.

    Supports bidirectional mapping, gap analysis, dual-standard
    compliance checking, and reconciliation reporting.

    Attributes:
        config: Platform configuration.
        _crosswalk_results: Cache of crosswalk results by inventory_id.
    """

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
    ) -> None:
        """
        Initialize CrosswalkEngine.

        Args:
            config: Platform configuration.
        """
        self.config = config or ISO14064AppConfig()
        self._crosswalk_results: Dict[str, CrossWalkResult] = {}
        logger.info("CrosswalkEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_crosswalk(
        self,
        inventory_id: str,
        org_id: str,
        reporting_year: int,
        category_results: Dict[str, CategoryResult],
    ) -> CrossWalkResult:
        """
        Generate a crosswalk mapping from ISO 14064-1 to GHG Protocol.

        Args:
            inventory_id: Inventory ID.
            org_id: Organization ID.
            reporting_year: Reporting year.
            category_results: Aggregated category results keyed by ISOCategory value.

        Returns:
            CrossWalkResult with bidirectional mapping and reconciliation.
        """
        start = _now()
        mappings: List[CrossWalkMapping] = []
        iso_totals: Dict[str, Decimal] = {}
        scope_totals: Dict[str, Decimal] = {
            "scope_1": Decimal("0"),
            "scope_2": Decimal("0"),
            "scope_3": Decimal("0"),
        }

        for cat in ISOCategory:
            result = category_results.get(cat.value)
            cat_tco2e = result.total_tco2e if result else Decimal("0")
            iso_totals[cat.value] = cat_tco2e

            scope = _ISO_TO_SCOPE[cat]
            scope_totals[scope] += cat_tco2e

            ghg_categories = _ISO_TO_GHG_CATEGORIES.get(cat, [])

            mapping = CrossWalkMapping(
                iso_category=cat,
                ghg_protocol_scope=scope,
                ghg_protocol_categories=ghg_categories,
                detailed_mapping=self._build_detailed_mapping(cat, result),
                notes=self._get_mapping_notes(cat),
            )
            mappings.append(mapping)

        iso_total = sum(iso_totals.values())
        ghg_total = sum(scope_totals.values())
        difference = iso_total - ghg_total

        recon_notes = None
        if difference != Decimal("0"):
            recon_notes = (
                f"Reconciliation difference of {difference} tCO2e detected. "
                "This may be due to differences in removal treatment between frameworks."
            )

        result = CrossWalkResult(
            org_id=org_id,
            reporting_year=reporting_year,
            mappings=mappings,
            iso_totals=iso_totals,
            ghg_protocol_totals=scope_totals,
            reconciliation_difference=difference,
            reconciliation_notes=recon_notes,
        )

        self._crosswalk_results[inventory_id] = result

        logger.info(
            "Crosswalk generated for %s: ISO=%.2f, GHG=%.2f, diff=%.2f",
            inventory_id,
            iso_total,
            ghg_total,
            difference,
        )
        return result

    def get_crosswalk(self, inventory_id: str) -> Optional[CrossWalkResult]:
        """Get cached crosswalk result."""
        return self._crosswalk_results.get(inventory_id)

    def get_scope_breakdown(self, inventory_id: str) -> Dict[str, Decimal]:
        """Get GHG Protocol scope breakdown from cached crosswalk."""
        result = self._crosswalk_results.get(inventory_id)
        if not result:
            return {}
        return dict(result.ghg_protocol_totals)

    # ------------------------------------------------------------------
    # Gap Analysis
    # ------------------------------------------------------------------

    def gap_analysis(
        self,
        inventory_id: str,
        significance_assessments: Optional[List[SignificanceAssessment]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a gap analysis between ISO 14064-1 and GHG Protocol.

        Identifies categories covered under one framework but not the other,
        and highlights structural differences.

        Args:
            inventory_id: Inventory ID (must have cached crosswalk).
            significance_assessments: Optional significance assessments to
                identify excluded categories.

        Returns:
            Dict with gap analysis results.
        """
        result = self._crosswalk_results.get(inventory_id)
        if not result:
            return {"error": "No crosswalk generated for this inventory. Run generate_crosswalk first."}

        # Identify which ISO categories have data
        populated_iso: List[str] = [
            cat for cat, val in result.iso_totals.items()
            if val > 0
        ]

        # Identify which scopes have data
        populated_scopes: List[str] = [
            scope for scope, val in result.ghg_protocol_totals.items()
            if val > 0
        ]

        # ISO coverage assessment
        iso_gaps: List[Dict[str, str]] = []
        for cat in ISOCategory:
            if cat.value not in populated_iso:
                is_indirect = cat != ISOCategory.CATEGORY_1_DIRECT
                excluded = False
                exclusion_reason = ""
                if significance_assessments:
                    for a in significance_assessments:
                        if a.iso_category == cat and a.result == SignificanceLevel.NOT_SIGNIFICANT:
                            excluded = True
                            exclusion_reason = a.justification
                            break

                iso_gaps.append({
                    "category": cat.value,
                    "category_name": ISO_CATEGORY_NAMES.get(cat, cat.value),
                    "status": "excluded_justified" if excluded else "missing",
                    "reason": exclusion_reason if excluded else "No data provided",
                    "is_indirect": is_indirect,
                })

        # GHG Protocol coverage
        ghg_gaps: List[Dict[str, str]] = []
        for scope, requirement in _GHG_PROTOCOL_REQUIREMENTS.items():
            if scope not in populated_scopes:
                ghg_gaps.append({
                    "scope": scope,
                    "requirement": requirement,
                    "status": "missing",
                })

        return {
            "inventory_id": inventory_id,
            "iso_14064_coverage": {
                "total_categories": len(ISOCategory),
                "populated": len(populated_iso),
                "gaps": iso_gaps,
            },
            "ghg_protocol_coverage": {
                "total_scopes": len(_GHG_PROTOCOL_REQUIREMENTS),
                "populated": len(populated_scopes),
                "gaps": ghg_gaps,
            },
            "framework_differences": _FRAMEWORK_DIFFERENCES,
        }

    # ------------------------------------------------------------------
    # Dual-Standard Compliance Check
    # ------------------------------------------------------------------

    def dual_standard_compliance_check(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Check whether the inventory meets requirements of both standards.

        Evaluates compliance against ISO 14064-1 and GHG Protocol
        independently and identifies any dual-reporting gaps.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with compliance assessment for both frameworks.
        """
        result = self._crosswalk_results.get(inventory_id)
        if not result:
            return {"error": "No crosswalk generated for this inventory."}

        iso_checks = self._check_iso_compliance(result)
        ghg_checks = self._check_ghg_protocol_compliance(result)

        iso_pass_count = sum(1 for c in iso_checks if c["met"])
        ghg_pass_count = sum(1 for c in ghg_checks if c["met"])

        return {
            "inventory_id": inventory_id,
            "iso_14064_compliance": {
                "checks": iso_checks,
                "passed": iso_pass_count,
                "total": len(iso_checks),
                "compliant": iso_pass_count == len(iso_checks),
            },
            "ghg_protocol_compliance": {
                "checks": ghg_checks,
                "passed": ghg_pass_count,
                "total": len(ghg_checks),
                "compliant": ghg_pass_count == len(ghg_checks),
            },
            "dual_compliant": (
                iso_pass_count == len(iso_checks)
                and ghg_pass_count == len(ghg_checks)
            ),
        }

    # ------------------------------------------------------------------
    # Reconciliation Report
    # ------------------------------------------------------------------

    def generate_reconciliation_report(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a detailed reconciliation report between frameworks.

        Explains any differences in totals and provides a detailed
        mapping table suitable for auditor review.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with reconciliation details.
        """
        result = self._crosswalk_results.get(inventory_id)
        if not result:
            return {"error": "No crosswalk generated for this inventory."}

        iso_total = sum(result.iso_totals.values())
        ghg_total = sum(result.ghg_protocol_totals.values())
        difference = result.reconciliation_difference

        # Build detailed mapping table
        mapping_table = self.generate_comparison_table(inventory_id)

        # Build scope-level summary
        scope_summary: List[Dict[str, str]] = []
        for scope, total in result.ghg_protocol_totals.items():
            contributing_cats = [
                cat.value for cat in ISOCategory
                if _ISO_TO_SCOPE.get(cat) == scope
                and result.iso_totals.get(cat.value, Decimal("0")) > 0
            ]
            scope_summary.append({
                "scope": scope,
                "total_tco2e": str(total),
                "contributing_iso_categories": contributing_cats,
            })

        return {
            "inventory_id": inventory_id,
            "org_id": result.org_id,
            "reporting_year": result.reporting_year,
            "iso_14064_total_tco2e": str(iso_total),
            "ghg_protocol_total_tco2e": str(ghg_total),
            "reconciliation_difference_tco2e": str(difference),
            "reconciliation_notes": result.reconciliation_notes or "Totals reconcile exactly.",
            "scope_summary": scope_summary,
            "mapping_table": mapping_table,
            "generated_at": _now().isoformat(),
            "provenance_hash": _sha256(
                f"{inventory_id}:{iso_total}:{ghg_total}:{difference}"
            ),
        }

    # ------------------------------------------------------------------
    # Comparison Table
    # ------------------------------------------------------------------

    def generate_comparison_table(self, inventory_id: str) -> List[Dict[str, str]]:
        """Generate a human-readable comparison table."""
        result = self._crosswalk_results.get(inventory_id)
        if not result:
            return []

        rows: List[Dict[str, str]] = []
        for m in result.mappings:
            iso_name = ISO_CATEGORY_NAMES.get(m.iso_category, m.iso_category.value)
            tco2e = result.iso_totals.get(m.iso_category.value, Decimal("0"))
            ghg_cats_str = ", ".join(m.ghg_protocol_categories) if m.ghg_protocol_categories else "-"

            rows.append({
                "iso_category": m.iso_category.value,
                "iso_category_name": iso_name,
                "ghg_scope": m.ghg_protocol_scope.replace("_", " ").title(),
                "ghg_categories": ghg_cats_str,
                "tco2e": str(tco2e),
                "notes": m.notes or "",
            })
        return rows

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mapping_notes(cat: ISOCategory) -> str:
        """Get mapping notes for a category."""
        notes_map = {
            ISOCategory.CATEGORY_1_DIRECT: (
                "Includes removals (net). GHG Protocol Scope 1 does not "
                "include removals separately."
            ),
            ISOCategory.CATEGORY_2_ENERGY: (
                "Maps to Scope 2 (location-based and market-based combined). "
                "GHG Protocol requires dual Scope 2 reporting."
            ),
            ISOCategory.CATEGORY_3_TRANSPORT: (
                "Maps to Scope 3 transportation categories (4, 6, 7, 9)."
            ),
            ISOCategory.CATEGORY_4_PRODUCTS_USED: (
                "Maps to Scope 3 upstream value chain categories (1, 2, 3, 5, 8)."
            ),
            ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: (
                "Maps to Scope 3 downstream value chain categories (10-14)."
            ),
            ISOCategory.CATEGORY_6_OTHER: (
                "Maps to Scope 3 Cat 15 - Investments."
            ),
        }
        return notes_map.get(cat, "")

    @staticmethod
    def _build_detailed_mapping(
        cat: ISOCategory,
        cat_result: Optional[CategoryResult],
    ) -> Dict[str, str]:
        """Build detailed source-level mapping for a category."""
        if cat_result is None:
            return {}

        details: Dict[str, str] = {}
        for source in cat_result.sources:
            scope = _ISO_TO_SCOPE.get(cat, "unknown")
            ghg_cats = _ISO_TO_GHG_CATEGORIES.get(cat, [])
            details[source.name] = (
                f"{scope} -> {', '.join(ghg_cats) if ghg_cats else 'N/A'}"
            )
        return details

    @staticmethod
    def _check_iso_compliance(result: CrossWalkResult) -> List[Dict[str, Any]]:
        """Check ISO 14064-1 compliance requirements."""
        checks: List[Dict[str, Any]] = []

        # Cat 1 is mandatory
        cat1_total = result.iso_totals.get(
            ISOCategory.CATEGORY_1_DIRECT.value, Decimal("0"),
        )
        checks.append({
            "requirement": "Category 1 (Direct) quantified",
            "clause": "5.2.2",
            "met": cat1_total > 0,
        })

        # Cat 2 is mandatory
        cat2_total = result.iso_totals.get(
            ISOCategory.CATEGORY_2_ENERGY.value, Decimal("0"),
        )
        checks.append({
            "requirement": "Category 2 (Energy Indirect) quantified",
            "clause": "5.2.2",
            "met": cat2_total > 0,
        })

        # At least one indirect category assessed
        indirect_populated = any(
            result.iso_totals.get(cat.value, Decimal("0")) > 0
            for cat in ISOCategory
            if cat not in (
                ISOCategory.CATEGORY_1_DIRECT,
                ISOCategory.CATEGORY_2_ENERGY,
            )
        )
        checks.append({
            "requirement": "At least one indirect category (3-6) assessed",
            "clause": "5.2.2",
            "met": indirect_populated,
        })

        return checks

    @staticmethod
    def _check_ghg_protocol_compliance(result: CrossWalkResult) -> List[Dict[str, Any]]:
        """Check GHG Protocol compliance requirements."""
        checks: List[Dict[str, Any]] = []

        scope1 = result.ghg_protocol_totals.get("scope_1", Decimal("0"))
        checks.append({
            "requirement": "Scope 1 quantified",
            "standard": "GHG Protocol Corporate Standard",
            "met": scope1 > 0,
        })

        scope2 = result.ghg_protocol_totals.get("scope_2", Decimal("0"))
        checks.append({
            "requirement": "Scope 2 quantified",
            "standard": "GHG Protocol Corporate Standard",
            "met": scope2 > 0,
        })

        scope3 = result.ghg_protocol_totals.get("scope_3", Decimal("0"))
        checks.append({
            "requirement": "Scope 3 assessed (recommended)",
            "standard": "GHG Protocol Scope 3 Standard",
            "met": scope3 > 0,
        })

        return checks
