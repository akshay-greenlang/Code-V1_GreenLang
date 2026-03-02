"""
Completeness Checker -- GHG Protocol Mandatory and Optional Disclosures

Verifies that a GHG inventory meets all mandatory disclosure requirements
of the GHG Protocol Corporate Standard, assesses Scope 3 category
materiality, identifies data gaps, and computes an overall data quality
score.

GHG Protocol mandatory disclosures include:
  - Organizational and operational boundary descriptions
  - Scope 1 and Scope 2 emissions
  - Base year information
  - Methodology and emission factor details
  - Biogenic CO2 reporting

Example:
    >>> checker = CompletenessChecker(config)
    >>> result = checker.check_completeness(inventory)
    >>> print(f"Completeness: {result.overall_pct}%")
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    DataQualityTier,
    GHGAppConfig,
    Scope,
    Scope3Category,
)
from .models import (
    CompletenessResult,
    DataGap,
    Disclosure,
    GHGInventory,
    _now,
)

logger = logging.getLogger(__name__)


class CompletenessChecker:
    """
    Checks GHG Protocol mandatory and optional disclosures.

    Implements a structured assessment of inventory completeness
    covering required disclosures, data quality, Scope 3 materiality,
    and gap identification.
    """

    # ------------------------------------------------------------------
    # Mandatory Disclosures (GHG Protocol Corporate Standard)
    # ------------------------------------------------------------------

    MANDATORY_DISCLOSURES: List[Dict[str, Any]] = [
        {"id": "MD-01", "name": "Description of the company", "category": "general"},
        {"id": "MD-02", "name": "Consolidation approach for emissions", "category": "boundary"},
        {"id": "MD-03", "name": "Organizational boundary description", "category": "boundary"},
        {"id": "MD-04", "name": "Operational boundary (scopes included)", "category": "boundary"},
        {"id": "MD-05", "name": "Reporting period covered", "category": "general"},
        {"id": "MD-06", "name": "Scope 1 emissions data in tCO2e", "category": "emissions"},
        {"id": "MD-07", "name": "Scope 2 emissions (location-based) in tCO2e", "category": "emissions"},
        {"id": "MD-08", "name": "Scope 2 emissions (market-based) in tCO2e", "category": "emissions"},
        {"id": "MD-09", "name": "Year-over-year comparison data", "category": "tracking"},
        {"id": "MD-10", "name": "Base year emissions and recalculation policy", "category": "tracking"},
        {"id": "MD-11", "name": "Appropriate context for emissions data", "category": "context"},
        {"id": "MD-12", "name": "Emissions disaggregated by GHG gas", "category": "emissions"},
        {"id": "MD-13", "name": "Biogenic CO2 reported separately", "category": "emissions"},
        {"id": "MD-14", "name": "Methodology description and emission factors used", "category": "methodology"},
        {"id": "MD-15", "name": "Scope 2 contractual instruments description", "category": "scope2"},
    ]

    OPTIONAL_DISCLOSURES: List[Dict[str, Any]] = [
        {"id": "OD-01", "name": "Scope 3 emissions by category", "category": "scope3"},
        {"id": "OD-02", "name": "Scope 3 screening results", "category": "scope3"},
        {"id": "OD-03", "name": "Intensity metrics", "category": "metrics"},
        {"id": "OD-04", "name": "Emission reduction targets", "category": "targets"},
        {"id": "OD-05", "name": "Uncertainty assessment", "category": "quality"},
        {"id": "OD-06", "name": "Verification statement", "category": "assurance"},
        {"id": "OD-07", "name": "Emissions by country/region", "category": "geographic"},
        {"id": "OD-08", "name": "Emissions by business unit", "category": "organizational"},
        {"id": "OD-09", "name": "Emission factor sources", "category": "methodology"},
        {"id": "OD-10", "name": "Data quality assessment", "category": "quality"},
    ]

    # Scope 3 categories considered mandatory for most reporters
    SCOPE3_TYPICALLY_MATERIAL: List[str] = [
        Scope3Category.CAT1_PURCHASED_GOODS.value,
        Scope3Category.CAT3_FUEL_ENERGY.value,
        Scope3Category.CAT4_UPSTREAM_TRANSPORT.value,
        Scope3Category.CAT5_WASTE_GENERATED.value,
        Scope3Category.CAT6_BUSINESS_TRAVEL.value,
        Scope3Category.CAT7_EMPLOYEE_COMMUTING.value,
    ]

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
    ) -> None:
        """
        Initialize CompletenessChecker.

        Args:
            config: Application configuration.
        """
        self.config = config or GHGAppConfig()
        logger.info("CompletenessChecker initialized")

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def check_completeness(
        self,
        inventory: GHGInventory,
    ) -> CompletenessResult:
        """
        Run full completeness assessment on an inventory.

        Args:
            inventory: GHG inventory to assess.

        Returns:
            CompletenessResult with disclosure status, gaps, and scores.
        """
        logger.info("Checking completeness for inventory %s", inventory.id)

        mandatory = self.check_mandatory_disclosures(inventory)
        optional = self._check_optional_disclosures(inventory)
        scope3_materiality = self.check_scope3_materiality(inventory)
        gaps = self.identify_gaps(inventory)
        dq_score = self.get_data_quality_score(inventory)
        exclusion_assessment = self.get_exclusion_assessment(inventory)

        # Overall completeness = pct of mandatory disclosures present
        total_mandatory = len(mandatory)
        present_mandatory = sum(1 for d in mandatory if d.present)
        overall_pct = Decimal("0")
        if total_mandatory > 0:
            overall_pct = (
                Decimal(str(present_mandatory)) / Decimal(str(total_mandatory)) * 100
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        result = CompletenessResult(
            inventory_id=inventory.id,
            overall_pct=overall_pct,
            mandatory_disclosures=mandatory,
            optional_disclosures=optional,
            scope3_materiality=scope3_materiality,
            gaps=gaps,
            data_quality_score=dq_score,
            exclusion_assessment=exclusion_assessment,
        )

        logger.info(
            "Completeness check complete for inventory %s: %.1f%% (%d/%d mandatory), "
            "%d gaps, DQ score=%.1f",
            inventory.id,
            overall_pct,
            present_mandatory,
            total_mandatory,
            len(gaps),
            dq_score,
        )
        return result

    def check_mandatory_disclosures(
        self,
        inventory: GHGInventory,
    ) -> List[Disclosure]:
        """
        Check each mandatory disclosure against the inventory.

        Args:
            inventory: GHG inventory to assess.

        Returns:
            List of Disclosure objects with present/absent status.
        """
        disclosures: List[Disclosure] = []

        for md in self.MANDATORY_DISCLOSURES:
            present, evidence = self._evaluate_mandatory(md["id"], inventory)
            disclosures.append(
                Disclosure(
                    id=md["id"],
                    name=md["name"],
                    category=md["category"],
                    required=True,
                    present=present,
                    evidence=evidence,
                )
            )

        return disclosures

    def check_scope3_materiality(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, bool]:
        """
        Assess materiality of each Scope 3 category.

        A category is considered material if:
          1. It is in the typically-material list, OR
          2. It represents >= 1% of total Scope 3 emissions

        Args:
            inventory: GHG inventory.

        Returns:
            Dict of category name to materiality boolean.
        """
        materiality: Dict[str, bool] = {}
        scope3 = inventory.scope3

        if scope3 is None or scope3.total_tco2e == 0:
            for cat in Scope3Category:
                materiality[cat.value] = cat.value in self.SCOPE3_TYPICALLY_MATERIAL
            return materiality

        for cat in Scope3Category:
            cat_value = scope3.by_category.get(cat.value, Decimal("0"))
            pct_of_total = Decimal("0")
            if scope3.total_tco2e > 0:
                pct_of_total = cat_value / scope3.total_tco2e * 100

            is_material = (
                cat.value in self.SCOPE3_TYPICALLY_MATERIAL
                or pct_of_total >= Decimal("1.0")
            )
            materiality[cat.value] = is_material

        return materiality

    def identify_gaps(
        self,
        inventory: GHGInventory,
    ) -> List[DataGap]:
        """
        Identify data gaps in the inventory.

        Args:
            inventory: GHG inventory.

        Returns:
            List of DataGap objects.
        """
        gaps: List[DataGap] = []

        # Scope 1 gaps
        gaps.extend(self._check_scope1_gaps(inventory))

        # Scope 2 gaps
        gaps.extend(self._check_scope2_gaps(inventory))

        # Scope 3 gaps
        gaps.extend(self._check_scope3_gaps(inventory))

        # Boundary gaps
        gaps.extend(self._check_boundary_gaps(inventory))

        # Quality/methodology gaps
        gaps.extend(self._check_methodology_gaps(inventory))

        return gaps

    def get_data_quality_score(
        self,
        inventory: GHGInventory,
    ) -> Decimal:
        """
        Calculate an overall data quality score (0-100).

        Scoring methodology:
          - Tier 3 data = 100 points
          - Tier 2 data = 70 points
          - Tier 1 data = 30 points
          - Weighted by scope contribution to total emissions.

        Args:
            inventory: GHG inventory.

        Returns:
            Data quality score (0-100).
        """
        if inventory.grand_total_tco2e == 0:
            return Decimal("0")

        tier_scores = {
            DataQualityTier.TIER_3: Decimal("100"),
            DataQualityTier.TIER_2: Decimal("70"),
            DataQualityTier.TIER_1: Decimal("30"),
        }

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        scope_data_list = [
            inventory.scope1,
            inventory.scope2_location,
            inventory.scope2_market,
            inventory.scope3,
        ]

        for scope_data in scope_data_list:
            if scope_data is None or scope_data.total_tco2e == 0:
                continue

            weight = scope_data.total_tco2e / inventory.grand_total_tco2e
            tier_score = tier_scores.get(scope_data.data_quality_tier, Decimal("30"))
            weighted_sum += weight * tier_score
            total_weight += weight

        if total_weight == 0:
            return Decimal("0")

        score = (weighted_sum / total_weight).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        return min(score, Decimal("100"))

    def get_exclusion_assessment(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """
        Assess the materiality of exclusions.

        Args:
            inventory: GHG inventory.

        Returns:
            Dict with exclusion summary and risk assessment.
        """
        if inventory.boundary is None:
            return {
                "has_exclusions": False,
                "total_exclusion_pct": "0",
                "risk_level": "none",
            }

        exclusions = inventory.boundary.exclusions
        if not exclusions:
            return {
                "has_exclusions": False,
                "total_exclusion_pct": "0",
                "risk_level": "none",
            }

        total_pct = sum(e.magnitude_pct for e in exclusions)

        risk_level = "low"
        if total_pct > Decimal("5"):
            risk_level = "medium"
        if total_pct > Decimal("10"):
            risk_level = "high"
        if total_pct > Decimal("20"):
            risk_level = "critical"

        return {
            "has_exclusions": True,
            "exclusion_count": len(exclusions),
            "total_exclusion_pct": str(total_pct),
            "risk_level": risk_level,
            "exclusions": [
                {
                    "scope": e.scope.value,
                    "category": e.category,
                    "reason": e.reason,
                    "magnitude_pct": str(e.magnitude_pct),
                }
                for e in exclusions
            ],
            "recommendation": (
                "Exclusions exceed 5% of total emissions. "
                "Consider including these sources or providing additional justification."
                if total_pct > Decimal("5")
                else "Exclusions are within acceptable limits."
            ),
        }

    # ------------------------------------------------------------------
    # Mandatory Disclosure Evaluation
    # ------------------------------------------------------------------

    def _evaluate_mandatory(
        self,
        disclosure_id: str,
        inventory: GHGInventory,
    ) -> tuple:
        """
        Evaluate a single mandatory disclosure.

        Returns:
            Tuple of (present: bool, evidence: Optional[str]).
        """
        evaluators = {
            "MD-01": self._check_md01_company_description,
            "MD-02": self._check_md02_consolidation,
            "MD-03": self._check_md03_org_boundary,
            "MD-04": self._check_md04_operational_boundary,
            "MD-05": self._check_md05_reporting_period,
            "MD-06": self._check_md06_scope1,
            "MD-07": self._check_md07_scope2_location,
            "MD-08": self._check_md08_scope2_market,
            "MD-09": self._check_md09_yoy,
            "MD-10": self._check_md10_base_year,
            "MD-11": self._check_md11_context,
            "MD-12": self._check_md12_gas_breakdown,
            "MD-13": self._check_md13_biogenic,
            "MD-14": self._check_md14_methodology,
            "MD-15": self._check_md15_scope2_instruments,
        }

        evaluator = evaluators.get(disclosure_id)
        if evaluator is None:
            return False, None
        return evaluator(inventory)

    def _check_md01_company_description(self, inv: GHGInventory) -> tuple:
        return (inv.org_id is not None and len(inv.org_id) > 0, "Organization ID present")

    def _check_md02_consolidation(self, inv: GHGInventory) -> tuple:
        if inv.boundary and inv.boundary.consolidation_approach:
            return True, f"Approach: {inv.boundary.consolidation_approach.value}"
        return False, None

    def _check_md03_org_boundary(self, inv: GHGInventory) -> tuple:
        if inv.boundary and len(inv.boundary.entity_ids) > 0:
            return True, f"{len(inv.boundary.entity_ids)} entities in boundary"
        return False, None

    def _check_md04_operational_boundary(self, inv: GHGInventory) -> tuple:
        if inv.boundary and len(inv.boundary.scopes) > 0:
            return True, f"Scopes: {[s.value for s in inv.boundary.scopes]}"
        return False, None

    def _check_md05_reporting_period(self, inv: GHGInventory) -> tuple:
        if inv.year and inv.year > 0:
            return True, f"Year: {inv.year}"
        return False, None

    def _check_md06_scope1(self, inv: GHGInventory) -> tuple:
        if inv.scope1 and inv.scope1.total_tco2e >= 0:
            return True, f"Scope 1: {inv.scope1.total_tco2e} tCO2e"
        return False, None

    def _check_md07_scope2_location(self, inv: GHGInventory) -> tuple:
        if inv.scope2_location and inv.scope2_location.total_tco2e >= 0:
            return True, f"Scope 2 (location): {inv.scope2_location.total_tco2e} tCO2e"
        return False, None

    def _check_md08_scope2_market(self, inv: GHGInventory) -> tuple:
        if inv.scope2_market and inv.scope2_market.total_tco2e >= 0:
            return True, f"Scope 2 (market): {inv.scope2_market.total_tco2e} tCO2e"
        return False, None

    def _check_md09_yoy(self, inv: GHGInventory) -> tuple:
        # YoY requires at least base year or prior year data
        if inv.boundary and inv.boundary.base_year:
            return True, f"Base year: {inv.boundary.base_year}"
        return False, None

    def _check_md10_base_year(self, inv: GHGInventory) -> tuple:
        if inv.boundary and inv.boundary.base_year:
            return True, f"Base year: {inv.boundary.base_year}"
        return False, None

    def _check_md11_context(self, inv: GHGInventory) -> tuple:
        # Context = intensity metrics or organizational description
        if inv.intensity_metrics and len(inv.intensity_metrics) > 0:
            return True, f"{len(inv.intensity_metrics)} intensity metrics"
        return False, None

    def _check_md12_gas_breakdown(self, inv: GHGInventory) -> tuple:
        if inv.scope1 and inv.scope1.by_gas and len(inv.scope1.by_gas) > 0:
            return True, f"{len(inv.scope1.by_gas)} gases reported"
        return False, None

    def _check_md13_biogenic(self, inv: GHGInventory) -> tuple:
        # Biogenic CO2 must be separately reported (even if zero)
        has_biogenic = False
        for scope_data in [inv.scope1, inv.scope2_location, inv.scope3]:
            if scope_data and scope_data.biogenic_co2 is not None:
                has_biogenic = True
                break
        if has_biogenic:
            total_biogenic = sum(
                s.biogenic_co2
                for s in [inv.scope1, inv.scope2_location, inv.scope3]
                if s is not None
            )
            return True, f"Biogenic CO2: {total_biogenic} tCO2"
        return False, None

    def _check_md14_methodology(self, inv: GHGInventory) -> tuple:
        # Check if any scope has methodology notes
        for scope_data in [inv.scope1, inv.scope2_location, inv.scope2_market, inv.scope3]:
            if scope_data and scope_data.methodology_notes:
                return True, "Methodology notes present"
        return False, None

    def _check_md15_scope2_instruments(self, inv: GHGInventory) -> tuple:
        if inv.scope2_market and inv.scope2_market.methodology_notes:
            return True, "Scope 2 market-based methodology described"
        return False, None

    # ------------------------------------------------------------------
    # Optional Disclosure Evaluation
    # ------------------------------------------------------------------

    def _check_optional_disclosures(
        self,
        inventory: GHGInventory,
    ) -> List[Disclosure]:
        """Check optional disclosures."""
        optionals: List[Disclosure] = []

        checks = {
            "OD-01": (
                inventory.scope3 is not None
                and inventory.scope3.total_tco2e > 0
                and len(inventory.scope3.by_category) > 0
            ),
            "OD-02": (
                inventory.scope3 is not None and len(inventory.scope3.by_category) > 0
            ),
            "OD-03": len(inventory.intensity_metrics) > 0,
            "OD-04": False,  # Requires target tracker integration
            "OD-05": inventory.uncertainty is not None,
            "OD-06": inventory.verification is not None,
            "OD-07": any(
                len(s.by_entity) > 0
                for s in [inventory.scope1, inventory.scope2_location, inventory.scope3]
                if s is not None
            ),
            "OD-08": any(
                len(s.by_entity) > 0
                for s in [inventory.scope1, inventory.scope2_location, inventory.scope3]
                if s is not None
            ),
            "OD-09": any(
                s.methodology_notes is not None
                for s in [inventory.scope1, inventory.scope2_location, inventory.scope2_market, inventory.scope3]
                if s is not None
            ),
            "OD-10": inventory.data_quality_score > 0,
        }

        for od in self.OPTIONAL_DISCLOSURES:
            present = checks.get(od["id"], False)
            optionals.append(
                Disclosure(
                    id=od["id"],
                    name=od["name"],
                    category=od["category"],
                    required=False,
                    present=present,
                    evidence=None,
                )
            )

        return optionals

    # ------------------------------------------------------------------
    # Gap Identification
    # ------------------------------------------------------------------

    def _check_scope1_gaps(self, inv: GHGInventory) -> List[DataGap]:
        """Identify Scope 1 data gaps."""
        gaps: List[DataGap] = []

        if inv.scope1 is None:
            gaps.append(DataGap(
                scope=Scope.SCOPE_1,
                description="Scope 1 emissions data is completely missing",
                severity="critical",
                recommendation="Calculate Scope 1 emissions from all direct sources",
            ))
            return gaps

        if inv.scope1.total_tco2e == 0:
            gaps.append(DataGap(
                scope=Scope.SCOPE_1,
                description="Scope 1 emissions are zero -- verify no direct emissions exist",
                severity="high",
                recommendation="Confirm zero emissions are accurate; most organizations have some Scope 1",
            ))

        if not inv.scope1.by_category:
            gaps.append(DataGap(
                scope=Scope.SCOPE_1,
                category="all",
                description="Scope 1 category breakdown missing",
                severity="medium",
                recommendation="Break down Scope 1 by source category (combustion, process, fugitive, etc.)",
            ))

        return gaps

    def _check_scope2_gaps(self, inv: GHGInventory) -> List[DataGap]:
        """Identify Scope 2 data gaps."""
        gaps: List[DataGap] = []

        if inv.scope2_location is None:
            gaps.append(DataGap(
                scope=Scope.SCOPE_2_LOCATION,
                description="Scope 2 location-based emissions missing (mandatory)",
                severity="critical",
                recommendation="Calculate location-based Scope 2 using grid-average emission factors",
            ))

        if inv.scope2_market is None:
            gaps.append(DataGap(
                scope=Scope.SCOPE_2_MARKET,
                description="Scope 2 market-based emissions missing (mandatory per GHG Protocol 2015)",
                severity="critical",
                recommendation="Calculate market-based Scope 2 using contractual instruments hierarchy",
            ))

        # Check dual reporting
        if (
            inv.scope2_location
            and inv.scope2_market
            and inv.scope2_location.total_tco2e > 0
            and inv.scope2_market.total_tco2e == 0
        ):
            gaps.append(DataGap(
                scope=Scope.SCOPE_2_MARKET,
                description="Market-based Scope 2 is zero while location-based is non-zero",
                severity="medium",
                recommendation="Verify market-based calculation includes residual mix factor",
            ))

        return gaps

    def _check_scope3_gaps(self, inv: GHGInventory) -> List[DataGap]:
        """Identify Scope 3 data gaps for material categories."""
        gaps: List[DataGap] = []

        if inv.scope3 is None:
            gaps.append(DataGap(
                scope=Scope.SCOPE_3,
                description="Scope 3 emissions not reported",
                severity="medium",
                recommendation="Screen all 15 Scope 3 categories for materiality",
            ))
            return gaps

        for cat_name in self.SCOPE3_TYPICALLY_MATERIAL:
            cat_value = inv.scope3.by_category.get(cat_name, Decimal("0"))
            if cat_value == 0:
                gaps.append(DataGap(
                    scope=Scope.SCOPE_3,
                    category=cat_name,
                    description=f"Typically material Scope 3 category '{cat_name}' has zero or missing data",
                    severity="medium",
                    recommendation=f"Assess {cat_name} using spend-based or activity-based methods",
                    estimated_magnitude_pct=Decimal("5.0"),
                ))

        return gaps

    def _check_boundary_gaps(self, inv: GHGInventory) -> List[DataGap]:
        """Check for boundary-related gaps."""
        gaps: List[DataGap] = []

        if inv.boundary is None:
            gaps.append(DataGap(
                scope=Scope.SCOPE_1,
                description="Inventory boundary not defined",
                severity="critical",
                recommendation="Set organizational and operational boundary before calculating emissions",
            ))

        return gaps

    def _check_methodology_gaps(self, inv: GHGInventory) -> List[DataGap]:
        """Check for methodology documentation gaps."""
        gaps: List[DataGap] = []

        missing_notes = []
        for scope_name, scope_data in [
            ("Scope 1", inv.scope1),
            ("Scope 2 (location)", inv.scope2_location),
            ("Scope 2 (market)", inv.scope2_market),
            ("Scope 3", inv.scope3),
        ]:
            if scope_data and not scope_data.methodology_notes:
                missing_notes.append(scope_name)

        if missing_notes:
            gaps.append(DataGap(
                scope=Scope.SCOPE_1,
                description=f"Methodology notes missing for: {', '.join(missing_notes)}",
                severity="low",
                recommendation="Document calculation methods and emission factor sources for each scope",
            ))

        return gaps
