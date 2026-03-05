"""
Target Configuration Engine -- SBTi Target Lifecycle Management

Implements full target lifecycle CRUD operations, validation of target
windows and coverage, annual reduction rate calculations, Scope 3
requirement checks, target summary generation, and SBTi submission
form generation.

The engine manages near-term, long-term, and net-zero targets with
in-memory stores for organizations and targets.  All numeric calculations
are deterministic (zero-hallucination).

Reference:
    - SBTi Criteria and Recommendations v5.1 (April 2023)
    - SBTi Corporate Net-Zero Standard v1.2 (October 2023)
    - GHG Protocol Corporate Standard

Example:
    >>> engine = TargetConfigurationEngine(config)
    >>> target = engine.create_target("org-1", request)
    >>> summary = engine.get_target_summary("org-1")
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ACA_ANNUAL_RATES,
    BASE_YEAR_MINIMUM,
    FLAG_TRIGGER_THRESHOLD,
    SCOPE1_2_COVERAGE_THRESHOLD,
    SCOPE3_NEAR_TERM_COVERAGE,
    SCOPE3_TRIGGER_THRESHOLD,
    SBTiAppConfig,
    SBTiSector,
    TargetMethod,
    TargetScope,
    TargetStatus,
    TargetType,
    ValidationStatus,
)
from .models import (
    CreateTargetRequest,
    EmissionsInventory,
    Organization,
    Pathway,
    PathwayMilestone,
    SubmissionForm,
    Target,
    UpdateTargetRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coverage Validation Result
# ---------------------------------------------------------------------------

class CoverageValidation:
    """Result of coverage validation for a target."""

    def __init__(
        self,
        scope: str,
        coverage_pct: float,
        threshold_pct: float,
        is_sufficient: bool,
        message: str,
    ) -> None:
        self.scope = scope
        self.coverage_pct = coverage_pct
        self.threshold_pct = threshold_pct
        self.is_sufficient = is_sufficient
        self.message = message


class Scope3Requirement:
    """Result of Scope 3 requirement assessment."""

    def __init__(
        self,
        scope3_pct: float,
        threshold_pct: float,
        is_required: bool,
        scope3_total: Decimal,
        total_emissions: Decimal,
        message: str,
    ) -> None:
        self.scope3_pct = scope3_pct
        self.threshold_pct = threshold_pct
        self.is_required = is_required
        self.scope3_total = scope3_total
        self.total_emissions = total_emissions
        self.message = message


class TargetSummary:
    """Summary of all targets for an organization."""

    def __init__(
        self,
        org_id: str,
        total_targets: int,
        near_term_targets: int,
        long_term_targets: int,
        net_zero_targets: int,
        validated_targets: int,
        scope1_2_covered: bool,
        scope3_covered: bool,
        flag_covered: bool,
        targets: List[Dict[str, Any]],
    ) -> None:
        self.org_id = org_id
        self.total_targets = total_targets
        self.near_term_targets = near_term_targets
        self.long_term_targets = long_term_targets
        self.net_zero_targets = net_zero_targets
        self.validated_targets = validated_targets
        self.scope1_2_covered = scope1_2_covered
        self.scope3_covered = scope3_covered
        self.flag_covered = flag_covered
        self.targets = targets


class TargetConfigurationEngine:
    """
    SBTi Target Configuration and Lifecycle Management Engine.

    Manages the full lifecycle of science-based targets: creation,
    retrieval, update, deletion, status transitions, validation of
    target windows and coverage, and submission form generation.

    Attributes:
        config: Application configuration.
        _targets: In-memory store of targets keyed by target ID.
        _organizations: In-memory store of organizations keyed by org ID.
        _inventories: In-memory store of inventories keyed by org ID then year.
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """
        Initialize TargetConfigurationEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or SBTiAppConfig()
        self._targets: Dict[str, Target] = {}
        self._organizations: Dict[str, Organization] = {}
        self._inventories: Dict[str, Dict[int, EmissionsInventory]] = {}
        logger.info("TargetConfigurationEngine initialized")

    # ------------------------------------------------------------------
    # Organization Management
    # ------------------------------------------------------------------

    def register_organization(self, org: Organization) -> Organization:
        """
        Register an organization in the engine's store.

        Args:
            org: Organization model instance.

        Returns:
            The registered Organization.
        """
        self._organizations[org.id] = org
        logger.info("Registered organization %s (%s)", org.id, org.name)
        return org

    def register_inventory(self, inventory: EmissionsInventory) -> EmissionsInventory:
        """
        Register an emissions inventory for an organization.

        Args:
            inventory: EmissionsInventory model instance.

        Returns:
            The registered EmissionsInventory.
        """
        if inventory.org_id not in self._inventories:
            self._inventories[inventory.org_id] = {}
        self._inventories[inventory.org_id][inventory.year] = inventory
        logger.info(
            "Registered inventory for org %s year %d", inventory.org_id, inventory.year,
        )
        return inventory

    # ------------------------------------------------------------------
    # Target CRUD
    # ------------------------------------------------------------------

    def create_target(self, org_id: str, request: CreateTargetRequest) -> Target:
        """
        Create a new SBTi target.

        Calculates derived fields (annual reduction rate, target-year
        emissions) and stores the target in the in-memory store.

        Args:
            org_id: Organization ID.
            request: Target creation request.

        Returns:
            Created Target with computed fields.

        Raises:
            ValueError: If organization is not registered or request is invalid.
        """
        start = datetime.utcnow()

        if org_id not in self._organizations:
            logger.warning("Organization %s not found, creating target anyway", org_id)

        # Compute base year emissions from inventory if available
        base_emissions = Decimal("0")
        org_inventories = self._inventories.get(org_id, {})
        base_inv = org_inventories.get(request.base_year)
        if base_inv is not None:
            base_emissions = self._get_scope_emissions(base_inv, request.scope)

        # Compute target year emissions
        target_emissions = base_emissions * (Decimal("1") - request.reduction_pct / Decimal("100"))

        # Compute annual linear reduction rate
        years = request.target_year - request.base_year
        annual_rate = Decimal("0")
        if years > 0:
            annual_rate = (request.reduction_pct / Decimal(str(years))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )

        # Generate target name if not provided
        name = request.name or self._generate_target_name(request)

        target = Target(
            tenant_id="default",
            org_id=org_id,
            name=name,
            target_type=request.target_type,
            scope=request.scope,
            method=request.method,
            pathway_alignment=request.pathway_alignment,
            base_year=request.base_year,
            target_year=request.target_year,
            base_year_emissions_tco2e=base_emissions,
            target_year_emissions_tco2e=target_emissions.quantize(Decimal("0.01")),
            reduction_pct=request.reduction_pct,
            annual_linear_reduction_rate=annual_rate,
            coverage_pct=request.coverage_pct,
            is_intensity_target=request.is_intensity_target,
            intensity_metric=request.intensity_metric,
            base_intensity_value=request.base_intensity_value,
            target_intensity_value=request.target_intensity_value,
            is_flag_target=request.is_flag_target,
            flag_commodity=request.flag_commodity,
            deforestation_commitment=request.deforestation_commitment,
            status=TargetStatus.TARGETS_SET,
        )

        self._targets[target.id] = target

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Created target %s for org %s: type=%s scope=%s reduction=%.1f%% in %.1f ms",
            target.id, org_id, request.target_type.value, request.scope.value,
            float(request.reduction_pct), elapsed_ms,
        )
        return target

    def get_target(self, target_id: str) -> Target:
        """
        Retrieve a target by its ID.

        Args:
            target_id: Target ID.

        Returns:
            Target instance.

        Raises:
            ValueError: If target not found.
        """
        target = self._targets.get(target_id)
        if target is None:
            raise ValueError(f"Target {target_id} not found")
        return target

    def list_targets(
        self,
        org_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Target]:
        """
        List all targets for an organization with optional filters.

        Args:
            org_id: Organization ID.
            filters: Optional filters dict with keys: target_type, scope,
                     status, method.

        Returns:
            List of matching Target instances.
        """
        filters = filters or {}
        results: List[Target] = []

        for target in self._targets.values():
            if target.org_id != org_id:
                continue

            # Apply filters
            if "target_type" in filters and target.target_type != filters["target_type"]:
                continue
            if "scope" in filters and target.scope != filters["scope"]:
                continue
            if "status" in filters and target.status != filters["status"]:
                continue
            if "method" in filters and target.method != filters["method"]:
                continue

            results.append(target)

        results.sort(key=lambda t: (t.target_type.value, t.created_at))
        logger.info("Listed %d targets for org %s", len(results), org_id)
        return results

    def update_target(self, target_id: str, request: UpdateTargetRequest) -> Target:
        """
        Update an existing target.

        Args:
            target_id: Target ID.
            request: Fields to update.

        Returns:
            Updated Target.

        Raises:
            ValueError: If target not found.
        """
        target = self.get_target(target_id)
        data = target.model_dump()

        # Apply provided updates
        update_fields = request.model_dump(exclude_none=True)
        data.update(update_fields)
        data["updated_at"] = _now()

        # Recompute annual rate if reduction changed
        if "reduction_pct" in update_fields:
            years = data["target_year"] - data["base_year"]
            if years > 0:
                rate = Decimal(str(update_fields["reduction_pct"])) / Decimal(str(years))
                data["annual_linear_reduction_rate"] = rate.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )

        # Recompute target emissions
        if "reduction_pct" in update_fields and data["base_year_emissions_tco2e"] > 0:
            new_reduction = Decimal(str(update_fields["reduction_pct"]))
            data["target_year_emissions_tco2e"] = (
                data["base_year_emissions_tco2e"] * (Decimal("1") - new_reduction / Decimal("100"))
            ).quantize(Decimal("0.01"))

        # Recompute provenance hash
        payload = (
            f"{data['org_id']}:{data['target_type']}:{data['scope']}:"
            f"{data['base_year']}:{data['target_year']}:{data['reduction_pct']}"
        )
        data["provenance_hash"] = _sha256(payload)

        updated = Target(**data)
        self._targets[target_id] = updated

        logger.info("Updated target %s: fields=%s", target_id, list(update_fields.keys()))
        return updated

    def delete_target(self, target_id: str) -> bool:
        """
        Delete a target from the store.

        Args:
            target_id: Target ID.

        Returns:
            True if deleted, False if not found.
        """
        if target_id in self._targets:
            del self._targets[target_id]
            logger.info("Deleted target %s", target_id)
            return True
        logger.warning("Target %s not found for deletion", target_id)
        return False

    # ------------------------------------------------------------------
    # Status Management
    # ------------------------------------------------------------------

    def update_target_status(self, target_id: str, new_status: TargetStatus) -> Target:
        """
        Transition a target to a new lifecycle status.

        Validates that the transition is allowed based on current status.

        Args:
            target_id: Target ID.
            new_status: New status to transition to.

        Returns:
            Updated Target.

        Raises:
            ValueError: If transition is not allowed.
        """
        target = self.get_target(target_id)
        current = target.status

        allowed = self._get_allowed_transitions(current)
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from {current.value} to {new_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        data = target.model_dump()
        data["status"] = new_status
        data["updated_at"] = _now()

        # Set lifecycle dates
        if new_status == TargetStatus.SUBMITTED:
            data["submission_date"] = date.today()
        elif new_status == TargetStatus.VALIDATED:
            data["validation_date"] = date.today()
            # Expiry is 5 years from validation
            from datetime import timedelta
            data["expiry_date"] = date.today() + timedelta(days=5 * 365)
            data["next_review_date"] = date.today() + timedelta(days=5 * 365)

        updated = Target(**data)
        self._targets[target_id] = updated

        logger.info(
            "Target %s status: %s -> %s", target_id, current.value, new_status.value,
        )
        return updated

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------

    def validate_target_window(self, base_year: int, target_year: int) -> bool:
        """
        Validate that the target timeframe meets SBTi requirements.

        Near-term targets must be 5-10 years from submission.
        Base year must be 2015 or more recent.

        Args:
            base_year: Base year.
            target_year: Target year.

        Returns:
            True if valid, False otherwise.
        """
        if base_year < self.config.base_year_minimum:
            logger.warning(
                "Base year %d is before minimum %d", base_year, self.config.base_year_minimum,
            )
            return False

        years = target_year - base_year
        if years < self.config.target_min_years:
            logger.warning("Timeframe %d years is below minimum %d", years, self.config.target_min_years)
            return False

        if years > self.config.target_max_years:
            logger.warning("Timeframe %d years exceeds maximum %d", years, self.config.target_max_years)
            return False

        if target_year <= base_year:
            logger.warning("Target year %d must be after base year %d", target_year, base_year)
            return False

        logger.info("Target window %d-%d (%d years) is valid", base_year, target_year, years)
        return True

    def validate_coverage(self, target: Target) -> CoverageValidation:
        """
        Validate emission scope coverage meets SBTi thresholds.

        Scope 1+2: >= 95% coverage required.
        Scope 3 near-term: >= 67% coverage required.
        Scope 3 long-term: >= 90% coverage required.

        Args:
            target: Target to validate.

        Returns:
            CoverageValidation with pass/fail and details.
        """
        coverage = float(target.coverage_pct)

        if target.scope in (TargetScope.SCOPE_1, TargetScope.SCOPE_2, TargetScope.SCOPE_1_2):
            threshold = SCOPE1_2_COVERAGE_THRESHOLD * 100
            is_sufficient = coverage >= threshold
            return CoverageValidation(
                scope=target.scope.value,
                coverage_pct=coverage,
                threshold_pct=threshold,
                is_sufficient=is_sufficient,
                message=(
                    f"Scope 1+2 coverage {coverage:.1f}% "
                    f"{'meets' if is_sufficient else 'below'} "
                    f"threshold {threshold:.0f}%"
                ),
            )

        elif target.scope == TargetScope.SCOPE_3:
            if target.target_type == TargetType.NEAR_TERM:
                threshold = SCOPE3_NEAR_TERM_COVERAGE * 100
            else:
                threshold = self.config.scope3_long_term_coverage_min * 100
            is_sufficient = coverage >= threshold
            return CoverageValidation(
                scope="scope_3",
                coverage_pct=coverage,
                threshold_pct=threshold,
                is_sufficient=is_sufficient,
                message=(
                    f"Scope 3 coverage {coverage:.1f}% "
                    f"{'meets' if is_sufficient else 'below'} "
                    f"threshold {threshold:.0f}%"
                ),
            )

        # Combined scope
        threshold = SCOPE1_2_COVERAGE_THRESHOLD * 100
        is_sufficient = coverage >= threshold
        return CoverageValidation(
            scope=target.scope.value,
            coverage_pct=coverage,
            threshold_pct=threshold,
            is_sufficient=is_sufficient,
            message=f"Combined coverage {coverage:.1f}% vs threshold {threshold:.0f}%",
        )

    def calculate_annual_rate(
        self,
        base_emissions: float,
        target_emissions: float,
        years: int,
    ) -> float:
        """
        Calculate the annual linear reduction rate.

        Formula: rate = (base - target) / (base * years)
        Expressed as a percentage per year.

        Args:
            base_emissions: Base year emissions (tCO2e).
            target_emissions: Target year emissions (tCO2e).
            years: Number of years between base and target.

        Returns:
            Annual reduction rate as a percentage (e.g. 4.2 for 4.2%/yr).

        Raises:
            ValueError: If base_emissions is zero or years is zero.
        """
        if base_emissions <= 0:
            raise ValueError("Base emissions must be positive")
        if years <= 0:
            raise ValueError("Years must be positive")

        total_reduction = base_emissions - target_emissions
        rate = (total_reduction / base_emissions) / years * 100

        logger.info(
            "Annual rate: base=%.1f target=%.1f years=%d rate=%.2f%%/yr",
            base_emissions, target_emissions, years, rate,
        )
        return round(rate, 2)

    # ------------------------------------------------------------------
    # Target Summary and Scope 3 Checks
    # ------------------------------------------------------------------

    def get_target_summary(self, org_id: str) -> TargetSummary:
        """
        Generate a summary of all targets for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            TargetSummary with counts, coverage flags, and target list.
        """
        targets = self.list_targets(org_id)

        near_term = [t for t in targets if t.target_type == TargetType.NEAR_TERM]
        long_term = [t for t in targets if t.target_type == TargetType.LONG_TERM]
        net_zero = [t for t in targets if t.target_type == TargetType.NET_ZERO]
        validated = [t for t in targets if t.status == TargetStatus.VALIDATED]

        # Check scope coverage
        s12_covered = any(
            t.scope in (TargetScope.SCOPE_1_2, TargetScope.SCOPE_1_2_3)
            for t in near_term
        )
        s3_covered = any(
            t.scope in (TargetScope.SCOPE_3, TargetScope.SCOPE_1_2_3)
            for t in targets
        )
        flag_covered = any(t.is_flag_target for t in targets)

        target_dicts = []
        for t in targets:
            target_dicts.append({
                "id": t.id,
                "name": t.name,
                "type": t.target_type.value,
                "scope": t.scope.value,
                "method": t.method.value,
                "base_year": t.base_year,
                "target_year": t.target_year,
                "reduction_pct": str(t.reduction_pct),
                "annual_rate": str(t.annual_linear_reduction_rate),
                "status": t.status.value,
            })

        summary = TargetSummary(
            org_id=org_id,
            total_targets=len(targets),
            near_term_targets=len(near_term),
            long_term_targets=len(long_term),
            net_zero_targets=len(net_zero),
            validated_targets=len(validated),
            scope1_2_covered=s12_covered,
            scope3_covered=s3_covered,
            flag_covered=flag_covered,
            targets=target_dicts,
        )

        logger.info(
            "Target summary for org %s: %d total, %d validated",
            org_id, summary.total_targets, summary.validated_targets,
        )
        return summary

    def check_scope3_requirement(self, org_id: str) -> Scope3Requirement:
        """
        Assess whether an organization must set a Scope 3 target.

        Per SBTi criterion C8: Scope 3 target required if S3 >= 40%
        of total S1+S2+S3 emissions.

        Args:
            org_id: Organization ID.

        Returns:
            Scope3Requirement assessment result.
        """
        inventories = self._inventories.get(org_id, {})
        if not inventories:
            return Scope3Requirement(
                scope3_pct=0.0,
                threshold_pct=SCOPE3_TRIGGER_THRESHOLD * 100,
                is_required=False,
                scope3_total=Decimal("0"),
                total_emissions=Decimal("0"),
                message="No emissions inventory found for organization",
            )

        # Use most recent inventory
        latest_year = max(inventories.keys())
        inv = inventories[latest_year]

        total = float(inv.total_s1_s2_s3_tco2e)
        s3 = float(inv.scope3_total_tco2e)

        if total <= 0:
            return Scope3Requirement(
                scope3_pct=0.0,
                threshold_pct=SCOPE3_TRIGGER_THRESHOLD * 100,
                is_required=False,
                scope3_total=inv.scope3_total_tco2e,
                total_emissions=inv.total_s1_s2_s3_tco2e,
                message="Total emissions are zero, cannot assess Scope 3 trigger",
            )

        s3_pct = (s3 / total) * 100
        is_required = s3_pct >= (SCOPE3_TRIGGER_THRESHOLD * 100)

        result = Scope3Requirement(
            scope3_pct=round(s3_pct, 2),
            threshold_pct=SCOPE3_TRIGGER_THRESHOLD * 100,
            is_required=is_required,
            scope3_total=inv.scope3_total_tco2e,
            total_emissions=inv.total_s1_s2_s3_tco2e,
            message=(
                f"Scope 3 is {s3_pct:.1f}% of total emissions. "
                f"{'Target IS required' if is_required else 'Target is NOT required'} "
                f"(threshold: {SCOPE3_TRIGGER_THRESHOLD * 100:.0f}%)."
            ),
        )

        logger.info(
            "Scope 3 check for org %s: %.1f%% -> %s",
            org_id, s3_pct, "REQUIRED" if is_required else "NOT REQUIRED",
        )
        return result

    # ------------------------------------------------------------------
    # Submission Form Generation
    # ------------------------------------------------------------------

    def generate_submission_data(self, target_id: str) -> SubmissionForm:
        """
        Generate SBTi submission form data for a target.

        Assembles all required information from the organization,
        inventory, and target data into a structured submission form.

        Args:
            target_id: Target ID.

        Returns:
            SubmissionForm populated with all required fields.

        Raises:
            ValueError: If target or organization data is incomplete.
        """
        start = datetime.utcnow()
        target = self.get_target(target_id)
        org = self._organizations.get(target.org_id)

        if org is None:
            raise ValueError(f"Organization {target.org_id} not found")

        inventories = self._inventories.get(target.org_id, {})
        base_inv = inventories.get(target.base_year)

        # Determine Scope 3 requirement
        scope3_req = self.check_scope3_requirement(target.org_id)

        # Determine FLAG requirement
        flag_required = False
        if base_inv and float(base_inv.flag_as_pct_of_total) >= (FLAG_TRIGGER_THRESHOLD * 100):
            flag_required = True

        form = SubmissionForm(
            tenant_id="default",
            org_id=target.org_id,
            target_ids=[target_id],
            company_name=org.name,
            sector=org.sector,
            country=org.country,
            contact_name=org.contact_person or "Not specified",
            contact_email=org.contact_email or "Not specified",
            base_year=target.base_year,
            base_year_scope1_tco2e=base_inv.scope1_tco2e if base_inv else Decimal("0"),
            base_year_scope2_tco2e=base_inv.scope2_market_tco2e if base_inv else Decimal("0"),
            base_year_scope3_tco2e=base_inv.scope3_total_tco2e if base_inv else Decimal("0"),
            scope3_screening_complete=True if base_inv and base_inv.scope3_categories else False,
            scope3_target_required=scope3_req.is_required,
            flag_target_required=flag_required,
            near_term_target_description=(
                f"Reduce {target.scope.value} emissions by {target.reduction_pct}% "
                f"from {target.base_year} to {target.target_year} "
                f"using {target.method.value} method."
            ),
            net_zero_commitment=target.target_type == TargetType.NET_ZERO,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Generated submission form for target %s in %.1f ms", target_id, elapsed_ms,
        )
        return form

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_scope_emissions(
        self,
        inventory: EmissionsInventory,
        scope: TargetScope,
    ) -> Decimal:
        """Extract total emissions for a given scope from an inventory."""
        if scope == TargetScope.SCOPE_1:
            return inventory.scope1_tco2e
        elif scope == TargetScope.SCOPE_2:
            return inventory.scope2_market_tco2e
        elif scope == TargetScope.SCOPE_1_2:
            return inventory.total_s1_s2_tco2e
        elif scope == TargetScope.SCOPE_3:
            return inventory.scope3_total_tco2e
        elif scope == TargetScope.SCOPE_1_2_3:
            return inventory.total_s1_s2_s3_tco2e
        return Decimal("0")

    @staticmethod
    def _generate_target_name(request: CreateTargetRequest) -> str:
        """Generate a descriptive target name from request parameters."""
        type_label = request.target_type.value.replace("_", " ").title()
        scope_label = request.scope.value.replace("_", "+").upper()
        method_label = request.method.value.upper()[:3]
        alignment = request.pathway_alignment.value.replace("_", " ").title()

        return f"{type_label} {scope_label} {method_label} {alignment}"

    @staticmethod
    def _get_allowed_transitions(current: TargetStatus) -> List[TargetStatus]:
        """Return allowed status transitions from the current status."""
        transitions: Dict[TargetStatus, List[TargetStatus]] = {
            TargetStatus.COMMITTED: [TargetStatus.TARGETS_SET],
            TargetStatus.TARGETS_SET: [TargetStatus.SUBMITTED, TargetStatus.REMOVED],
            TargetStatus.SUBMITTED: [
                TargetStatus.VALIDATED, TargetStatus.REMOVED,
            ],
            TargetStatus.VALIDATED: [
                TargetStatus.REVALIDATION_REQUIRED, TargetStatus.EXPIRED,
                TargetStatus.REMOVED,
            ],
            TargetStatus.REVALIDATION_REQUIRED: [
                TargetStatus.SUBMITTED, TargetStatus.EXPIRED, TargetStatus.REMOVED,
            ],
            TargetStatus.EXPIRED: [TargetStatus.REMOVED],
            TargetStatus.REMOVED: [],
        }
        return transitions.get(current, [])
