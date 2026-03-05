"""
FLAG Engine -- Forest, Land and Agriculture Target Assessment

Implements the SBTi FLAG Guidance v1.1 for companies with significant
land-sector emissions.  Covers FLAG trigger assessment (>=20% of total),
commodity-level pathway calculations, sector-level pathway at 3.03%/yr,
deforestation commitment verification, removal/sequestration validation,
long-term FLAG target computation (>=72% by 2050), and commodity
benchmark comparison.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Forest, Land and Agriculture Guidance v1.1 (2024)
    - SBTi Criteria C15 (FLAG target requirement)
    - SBTi Corporate Net-Zero Standard v1.2, FLAG annex

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = FLAGEngine(SBTiAppConfig())
    >>> trigger = engine.assess_flag_trigger("org-1")
    >>> print(trigger.flag_target_required)
    True
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    FLAG_COMMODITY_PATHWAYS,
    FLAG_TRIGGER_THRESHOLD,
    FLAGCommodity,
    SBTiAppConfig,
    SBTiSector,
)
from .models import (
    CommodityData,
    EmissionsInventory,
    FLAGAssessment,
    Organization,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class FLAGTriggerResult(BaseModel):
    """Result of FLAG trigger assessment."""

    org_id: str = Field(...)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    flag_pct_of_total: float = Field(default=0.0)
    threshold_pct: float = Field(default=20.0)
    flag_target_required: bool = Field(default=False)
    flag_sectors_identified: List[str] = Field(default_factory=list)
    message: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class CommodityPathwayResult(BaseModel):
    """Calculated pathway for a specific FLAG commodity."""

    commodity: str = Field(...)
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    base_intensity: float = Field(default=0.0)
    target_intensity_2030: float = Field(default=0.0)
    target_intensity_2050: float = Field(default=0.0)
    annual_reduction_rate: float = Field(default=3.03)
    milestones: List[Dict[str, float]] = Field(default_factory=list)
    intensity_unit: str = Field(default="tCO2e/t_commodity")
    deforestation_commitment_required: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class SectorPathwayResult(BaseModel):
    """Calculated FLAG sector-level pathway at 3.03%/yr."""

    base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    annual_rate_pct: float = Field(default=3.03)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    milestones: List[Dict[str, float]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DeforestationStatus(BaseModel):
    """Deforestation commitment verification status."""

    org_id: str = Field(...)
    commitment_declared: bool = Field(default=False)
    commitment_date: Optional[date] = Field(None)
    target_date: date = Field(default_factory=lambda: date(2025, 12, 31))
    is_compliant: bool = Field(default=False)
    commodities_requiring_commitment: List[str] = Field(default_factory=list)
    message: str = Field(default="")


class RemovalValidation(BaseModel):
    """Validation of carbon removal/sequestration claims."""

    org_id: str = Field(...)
    claimed_removals_tco2e: float = Field(default=0.0, ge=0.0)
    validated_removals_tco2e: float = Field(default=0.0, ge=0.0)
    removal_types: List[Dict[str, Any]] = Field(default_factory=list)
    counted_toward_target: bool = Field(default=False)
    message: str = Field(default="")


class FLAGLongTermResult(BaseModel):
    """Long-term FLAG target assessment (by 2050)."""

    org_id: str = Field(...)
    base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=72.0)
    target_year: int = Field(default=2050)
    projected_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    meets_minimum: bool = Field(default=False)
    net_zero_compatible: bool = Field(default=False)
    message: str = Field(default="")


class EmissionsSeparation(BaseModel):
    """Separation of FLAG vs non-FLAG emissions."""

    org_id: str = Field(...)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    flag_tco2e: float = Field(default=0.0, ge=0.0)
    non_flag_tco2e: float = Field(default=0.0, ge=0.0)
    flag_pct: float = Field(default=0.0)
    flag_by_source: Dict[str, float] = Field(default_factory=dict)


class CommodityBenchmark(BaseModel):
    """Benchmark comparison for a FLAG commodity."""

    commodity: str = Field(...)
    company_intensity: float = Field(default=0.0)
    benchmark_2020: float = Field(default=0.0)
    benchmark_2030: float = Field(default=0.0)
    benchmark_2050: float = Field(default=0.0)
    gap_to_2030: float = Field(default=0.0)
    gap_to_2050: float = Field(default=0.0)
    intensity_unit: str = Field(default="tCO2e/t_commodity")
    on_track_2030: bool = Field(default=False)


class FLAGReport(BaseModel):
    """Comprehensive FLAG assessment report."""

    org_id: str = Field(...)
    trigger_result: Dict[str, Any] = Field(default_factory=dict)
    commodity_pathways: List[Dict[str, Any]] = Field(default_factory=list)
    sector_pathway: Optional[Dict[str, Any]] = Field(None)
    deforestation_status: Dict[str, Any] = Field(default_factory=dict)
    long_term_assessment: Dict[str, Any] = Field(default_factory=dict)
    emission_separation: Dict[str, Any] = Field(default_factory=dict)
    benchmarks: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# FLAGEngine
# ---------------------------------------------------------------------------

class FLAGEngine:
    """
    SBTi FLAG (Forest, Land and Agriculture) Assessment Engine.

    Determines FLAG target applicability, calculates commodity and sector
    pathways, verifies deforestation commitments, validates removal claims,
    and generates FLAG compliance reports.

    Attributes:
        config: Application configuration.
        _inventories: In-memory inventory store keyed by org_id.
        _organizations: In-memory organization store keyed by org_id.
        _assessments: In-memory FLAG assessment store keyed by org_id.
        _commodity_data: Commodity-level data keyed by org_id.

    Example:
        >>> engine = FLAGEngine(SBTiAppConfig())
        >>> trigger = engine.assess_flag_trigger("org-1")
    """

    FLAG_SECTOR_RATE: float = 0.0303  # 3.03% annual reduction
    FLAG_LONG_TERM_REDUCTION_PCT: float = 72.0  # Minimum by 2050
    FLAG_NET_ZERO_REDUCTION_PCT: float = 90.0  # Net-zero threshold
    DEFORESTATION_TARGET_YEAR: int = 2025

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the FLAG Engine."""
        self.config = config or SBTiAppConfig()
        self._inventories: Dict[str, EmissionsInventory] = {}
        self._organizations: Dict[str, Organization] = {}
        self._assessments: Dict[str, FLAGAssessment] = {}
        self._commodity_data: Dict[str, List[CommodityData]] = {}
        logger.info("FLAGEngine initialized")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_inventory(self, inventory: EmissionsInventory) -> None:
        """Register an emissions inventory."""
        self._inventories[inventory.org_id] = inventory

    def register_organization(self, org: Organization) -> None:
        """Register an organization."""
        self._organizations[org.id] = org

    def register_commodity_data(
        self, org_id: str, data: List[CommodityData],
    ) -> None:
        """Register commodity-level data for an organization."""
        self._commodity_data[org_id] = data

    # ------------------------------------------------------------------
    # FLAG Trigger Assessment
    # ------------------------------------------------------------------

    def assess_flag_trigger(self, org_id: str) -> FLAGTriggerResult:
        """
        Assess whether FLAG target is required per SBTi criterion C15.

        FLAG target is required when FLAG emissions >= 20% of total
        Scope 1+2+3 emissions.

        Args:
            org_id: Organization identifier.

        Returns:
            FLAGTriggerResult with determination.
        """
        inventory = self._inventories.get(org_id)
        if inventory is None:
            return FLAGTriggerResult(
                org_id=org_id,
                message="No inventory registered for this organization.",
            )

        flag = float(inventory.flag_tco2e)
        total = float(inventory.total_s1_s2_s3_tco2e)

        if total <= 0:
            return FLAGTriggerResult(
                org_id=org_id,
                flag_emissions_tco2e=flag,
                total_emissions_tco2e=total,
                message="Total emissions are zero; cannot assess FLAG trigger.",
            )

        flag_pct = (flag / total) * 100.0
        threshold = float(FLAG_TRIGGER_THRESHOLD)
        required = flag_pct >= threshold

        # Identify FLAG-relevant sectors
        sectors: List[str] = []
        org = self._organizations.get(org_id)
        if org and org.sector in (SBTiSector.AGRICULTURE, SBTiSector.PULP_PAPER):
            sectors.append(org.sector.value)
        if flag > 0:
            sectors.append("land_use")

        result = FLAGTriggerResult(
            org_id=org_id,
            flag_emissions_tco2e=round(flag, 2),
            total_emissions_tco2e=round(total, 2),
            flag_pct_of_total=round(flag_pct, 2),
            threshold_pct=threshold,
            flag_target_required=required,
            flag_sectors_identified=sectors,
            message=(
                f"FLAG emissions are {flag_pct:.1f}% of total. "
                f"{'FLAG target IS required.' if required else 'FLAG target is NOT required.'}"
            ),
        )

        logger.info(
            "FLAG trigger for org %s: %.1f%% -> %s",
            org_id, flag_pct, "REQUIRED" if required else "NOT REQUIRED",
        )
        return result

    # ------------------------------------------------------------------
    # FLAG Sector Classification
    # ------------------------------------------------------------------

    def classify_flag_sector(self, org_id: str) -> Dict[str, Any]:
        """
        Classify the organization's FLAG sector based on industry codes.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with sector classification and FLAG relevance.
        """
        org = self._organizations.get(org_id)
        if org is None:
            return {"org_id": org_id, "classified": False, "message": "Organization not found."}

        flag_sectors = {
            SBTiSector.AGRICULTURE: {"flag_relevant": True, "flag_type": "primary_producer"},
            SBTiSector.PULP_PAPER: {"flag_relevant": True, "flag_type": "forestry_products"},
        }

        classification = flag_sectors.get(org.sector, {"flag_relevant": False, "flag_type": "non_flag"})

        return {
            "org_id": org_id,
            "classified": True,
            "sector": org.sector.value,
            "is_flag_sector": classification["flag_relevant"],
            "flag_type": classification["flag_type"],
            "is_flag_relevant": org.is_flag_relevant,
            "isic_code": org.isic_code,
            "nace_code": org.nace_code,
        }

    # ------------------------------------------------------------------
    # Commodity Pathway Calculation
    # ------------------------------------------------------------------

    def calculate_commodity_pathway(
        self,
        commodity: str,
        base_intensity: float,
        base_year: int = 2020,
        target_year: int = 2030,
        production_volume: float = 1.0,
    ) -> CommodityPathwayResult:
        """
        Calculate a commodity-specific FLAG intensity pathway.

        Uses SBTi FLAG benchmark intensities for each of the 11 commodity
        categories to establish a convergence pathway.

        Args:
            commodity: FLAG commodity name (e.g. "cattle", "soy").
            base_intensity: Company base year intensity.
            base_year: Base year (default 2020).
            target_year: Target year (default 2030).
            production_volume: Annual production volume in tonnes.

        Returns:
            CommodityPathwayResult with annual milestones.

        Raises:
            ValueError: If commodity is not recognized.
        """
        start = datetime.utcnow()

        benchmarks = FLAG_COMMODITY_PATHWAYS.get(commodity)
        if benchmarks is None:
            raise ValueError(
                f"Unknown FLAG commodity: {commodity}. "
                f"Valid: {list(FLAG_COMMODITY_PATHWAYS.keys())}"
            )

        target_2030 = float(benchmarks["target_intensity_2030"])
        target_2050 = float(benchmarks["target_intensity_2050"])
        rate = float(benchmarks["annual_reduction_rate"])
        unit = benchmarks["intensity_unit"]
        defor_required = benchmarks["deforestation_commitment_required"]

        # Determine target intensity for the chosen target year
        if target_year <= 2030:
            target_intensity = target_2030
        elif target_year >= 2050:
            target_intensity = target_2050
        else:
            # Linear interpolation between 2030 and 2050
            progress = (target_year - 2030) / (2050 - 2030)
            target_intensity = target_2030 + (target_2050 - target_2030) * progress

        # Generate annual milestones
        total_years = target_year - base_year
        milestones: List[Dict[str, float]] = []

        for year in range(base_year, target_year + 1):
            elapsed = year - base_year
            if total_years > 0:
                progress = elapsed / total_years
            else:
                progress = 1.0
            expected = base_intensity + (target_intensity - base_intensity) * progress
            milestones.append({
                "year": year,
                "intensity": round(expected, 4),
                "emissions_tco2e": round(expected * production_volume, 2),
            })

        provenance = _sha256(
            f"flag_commodity:{commodity}:{base_intensity}:{base_year}:{target_year}"
        )

        result = CommodityPathwayResult(
            commodity=commodity,
            base_year=base_year,
            target_year=target_year,
            base_intensity=base_intensity,
            target_intensity_2030=target_2030,
            target_intensity_2050=target_2050,
            annual_reduction_rate=rate,
            milestones=milestones,
            intensity_unit=unit,
            deforestation_commitment_required=defor_required,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FLAG commodity pathway: %s base=%.4f target=%.4f in %.1f ms",
            commodity, base_intensity, target_intensity, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Sector Pathway (3.03%/yr)
    # ------------------------------------------------------------------

    def calculate_sector_pathway(
        self,
        base_emissions: float,
        base_year: int = 2020,
        target_year: int = 2030,
    ) -> SectorPathwayResult:
        """
        Calculate a FLAG sector-level absolute reduction pathway.

        Uses the uniform SBTi FLAG sector rate of 3.03% per year.

        Args:
            base_emissions: Base year FLAG emissions in tCO2e.
            base_year: Base year.
            target_year: Target year.

        Returns:
            SectorPathwayResult with annual milestones.
        """
        start = datetime.utcnow()
        rate = self.FLAG_SECTOR_RATE

        milestones: List[Dict[str, float]] = []
        for year in range(base_year, target_year + 1):
            elapsed = year - base_year
            expected = base_emissions * ((1 - rate) ** elapsed)
            reduction_pct = (1 - expected / base_emissions) * 100 if base_emissions > 0 else 0
            milestones.append({
                "year": year,
                "emissions_tco2e": round(expected, 2),
                "cumulative_reduction_pct": round(reduction_pct, 2),
            })

        target_emissions = base_emissions * ((1 - rate) ** (target_year - base_year))
        provenance = _sha256(f"flag_sector:{base_emissions}:{base_year}:{target_year}")

        result = SectorPathwayResult(
            base_emissions_tco2e=round(base_emissions, 2),
            base_year=base_year,
            target_year=target_year,
            annual_rate_pct=round(rate * 100, 2),
            target_emissions_tco2e=round(target_emissions, 2),
            milestones=milestones,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FLAG sector pathway: base=%.0f rate=%.2f%%/yr target=%.0f in %.1f ms",
            base_emissions, rate * 100, target_emissions, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Deforestation Commitment
    # ------------------------------------------------------------------

    def track_deforestation_commitment(self, org_id: str) -> DeforestationStatus:
        """
        Verify deforestation commitment status.

        SBTi FLAG Guidance requires companies sourcing deforestation-risk
        commodities to commit to zero deforestation by 2025.

        Args:
            org_id: Organization identifier.

        Returns:
            DeforestationStatus with compliance assessment.
        """
        org = self._organizations.get(org_id)
        commodities = self._commodity_data.get(org_id, [])

        # Determine which commodities require deforestation commitment
        requiring_commitment: List[str] = []
        for cd in commodities:
            commodity_key = cd.commodity.lower().replace(" ", "_")
            benchmarks = FLAG_COMMODITY_PATHWAYS.get(commodity_key, {})
            if benchmarks.get("deforestation_commitment_required", False):
                requiring_commitment.append(cd.commodity)

        # Check organization's commitment status
        commitment_declared = False
        commitment_date = None
        if org is not None:
            commitment_declared = org.is_flag_relevant
            commitment_date = org.commitment_date

        target_date = date(self.DEFORESTATION_TARGET_YEAR, 12, 31)
        is_compliant = commitment_declared or len(requiring_commitment) == 0

        result = DeforestationStatus(
            org_id=org_id,
            commitment_declared=commitment_declared,
            commitment_date=commitment_date,
            target_date=target_date,
            is_compliant=is_compliant,
            commodities_requiring_commitment=requiring_commitment,
            message=(
                "Deforestation commitment verified."
                if is_compliant
                else f"Deforestation commitment required for: {', '.join(requiring_commitment)}"
            ),
        )

        logger.info(
            "Deforestation check for org %s: compliant=%s, commodities=%d",
            org_id, is_compliant, len(requiring_commitment),
        )
        return result

    # ------------------------------------------------------------------
    # Removal Validation
    # ------------------------------------------------------------------

    def validate_flag_removals(
        self,
        org_id: str,
        removal_claims: List[Dict[str, Any]],
    ) -> RemovalValidation:
        """
        Validate carbon removal/sequestration claims for FLAG targets.

        Per SBTi FLAG Guidance, biological removals on company-owned or
        managed land may be counted separately but not toward the
        abatement target. Only high-quality removals are recognized.

        Args:
            org_id: Organization identifier.
            removal_claims: List of dicts with keys: type, amount_tco2e,
                           verification_status, permanence_years.

        Returns:
            RemovalValidation with validated amounts.
        """
        total_claimed = 0.0
        total_validated = 0.0
        removal_types: List[Dict[str, Any]] = []

        for claim in removal_claims:
            amount = claim.get("amount_tco2e", 0.0)
            rtype = claim.get("type", "unknown")
            verified = claim.get("verification_status", "unverified")
            permanence = claim.get("permanence_years", 0)

            total_claimed += amount

            # Validate: must be verified and have minimum permanence
            is_valid = verified in ("limited", "reasonable") and permanence >= 20
            validated_amount = amount if is_valid else 0.0
            total_validated += validated_amount

            removal_types.append({
                "type": rtype,
                "claimed_tco2e": round(amount, 2),
                "validated_tco2e": round(validated_amount, 2),
                "verification_status": verified,
                "permanence_years": permanence,
                "accepted": is_valid,
            })

        result = RemovalValidation(
            org_id=org_id,
            claimed_removals_tco2e=round(total_claimed, 2),
            validated_removals_tco2e=round(total_validated, 2),
            removal_types=removal_types,
            counted_toward_target=False,
            message=(
                f"Validated {total_validated:.0f} tCO2e of {total_claimed:.0f} tCO2e claimed. "
                f"Removals are reported separately and NOT counted toward abatement target."
            ),
        )

        logger.info(
            "Removal validation for org %s: claimed=%.0f validated=%.0f",
            org_id, total_claimed, total_validated,
        )
        return result

    # ------------------------------------------------------------------
    # Long-Term FLAG Target
    # ------------------------------------------------------------------

    def calculate_flag_long_term(
        self,
        org_id: str,
        base_emissions: float,
        target_year: int = 2050,
    ) -> FLAGLongTermResult:
        """
        Calculate long-term FLAG target (minimum 72% reduction by 2050).

        SBTi FLAG Guidance requires a minimum 72% absolute reduction in
        FLAG emissions by 2050 for 1.5C alignment. Net-zero requires
        at least 90% reduction.

        Args:
            org_id: Organization identifier.
            base_emissions: Base year FLAG emissions in tCO2e.
            target_year: Target year (default 2050).

        Returns:
            FLAGLongTermResult with projected emissions and compliance.
        """
        years = target_year - 2020  # Assume 2020 base
        rate = self.FLAG_SECTOR_RATE

        projected = base_emissions * ((1 - rate) ** years)
        reduction_pct = ((base_emissions - projected) / base_emissions * 100) if base_emissions > 0 else 0

        meets_minimum = reduction_pct >= self.FLAG_LONG_TERM_REDUCTION_PCT
        net_zero_compatible = reduction_pct >= self.FLAG_NET_ZERO_REDUCTION_PCT

        result = FLAGLongTermResult(
            org_id=org_id,
            base_emissions_tco2e=round(base_emissions, 2),
            target_reduction_pct=round(reduction_pct, 2),
            target_year=target_year,
            projected_emissions_tco2e=round(projected, 2),
            meets_minimum=meets_minimum,
            net_zero_compatible=net_zero_compatible,
            message=(
                f"Projected reduction of {reduction_pct:.1f}% by {target_year}. "
                f"{'Meets' if meets_minimum else 'Does NOT meet'} minimum 72% threshold. "
                f"{'Compatible' if net_zero_compatible else 'Not compatible'} with net-zero."
            ),
        )

        logger.info(
            "FLAG long-term for org %s: %.1f%% reduction, meets_min=%s, nz=%s",
            org_id, reduction_pct, meets_minimum, net_zero_compatible,
        )
        return result

    # ------------------------------------------------------------------
    # Emissions Separation
    # ------------------------------------------------------------------

    def separate_flag_emissions(self, org_id: str) -> EmissionsSeparation:
        """
        Separate FLAG emissions from non-FLAG emissions.

        SBTi requires FLAG emissions to be tracked and targeted separately
        from energy/industrial emissions.

        Args:
            org_id: Organization identifier.

        Returns:
            EmissionsSeparation with FLAG and non-FLAG breakdown.
        """
        inventory = self._inventories.get(org_id)
        if inventory is None:
            return EmissionsSeparation(org_id=org_id)

        flag = float(inventory.flag_tco2e)
        total = float(inventory.total_s1_s2_s3_tco2e)
        non_flag = total - flag

        flag_pct = (flag / total * 100) if total > 0 else 0

        # Break down FLAG sources
        flag_by_source: Dict[str, float] = {}
        if flag > 0:
            # Approximate breakdown from common FLAG sources
            flag_by_source["land_use_change"] = round(flag * 0.40, 2)
            flag_by_source["agriculture"] = round(flag * 0.35, 2)
            flag_by_source["forestry"] = round(flag * 0.15, 2)
            flag_by_source["other_land"] = round(flag * 0.10, 2)

        result = EmissionsSeparation(
            org_id=org_id,
            total_tco2e=round(total, 2),
            flag_tco2e=round(flag, 2),
            non_flag_tco2e=round(non_flag, 2),
            flag_pct=round(flag_pct, 2),
            flag_by_source=flag_by_source,
        )

        logger.info(
            "Emission separation for org %s: FLAG=%.0f (%.1f%%), non-FLAG=%.0f",
            org_id, flag, flag_pct, non_flag,
        )
        return result

    # ------------------------------------------------------------------
    # Commodity Benchmarks
    # ------------------------------------------------------------------

    def get_commodity_benchmarks(
        self,
        commodity: str,
        company_intensity: float,
    ) -> CommodityBenchmark:
        """
        Compare company intensity against FLAG commodity benchmarks.

        Args:
            commodity: FLAG commodity name.
            company_intensity: Company's current intensity value.

        Returns:
            CommodityBenchmark with gap analysis.

        Raises:
            ValueError: If commodity is not recognized.
        """
        benchmarks = FLAG_COMMODITY_PATHWAYS.get(commodity)
        if benchmarks is None:
            raise ValueError(f"Unknown FLAG commodity: {commodity}")

        b_2020 = float(benchmarks["base_intensity_2020"])
        b_2030 = float(benchmarks["target_intensity_2030"])
        b_2050 = float(benchmarks["target_intensity_2050"])
        unit = benchmarks["intensity_unit"]

        gap_2030 = company_intensity - b_2030
        gap_2050 = company_intensity - b_2050
        on_track = company_intensity <= b_2030

        return CommodityBenchmark(
            commodity=commodity,
            company_intensity=round(company_intensity, 4),
            benchmark_2020=b_2020,
            benchmark_2030=b_2030,
            benchmark_2050=b_2050,
            gap_to_2030=round(gap_2030, 4),
            gap_to_2050=round(gap_2050, 4),
            intensity_unit=unit,
            on_track_2030=on_track,
        )

    # ------------------------------------------------------------------
    # FLAG Assessment (Full)
    # ------------------------------------------------------------------

    def run_flag_assessment(self, org_id: str) -> FLAGAssessment:
        """
        Run a comprehensive FLAG assessment for an organization.

        Combines trigger assessment, commodity pathway analysis, and
        deforestation commitment verification into a single assessment.

        Args:
            org_id: Organization identifier.

        Returns:
            FLAGAssessment domain model.
        """
        start = datetime.utcnow()
        inventory = self._inventories.get(org_id)
        if inventory is None:
            raise ValueError(f"No inventory registered for org {org_id}")

        trigger = self.assess_flag_trigger(org_id)
        defor = self.track_deforestation_commitment(org_id)
        commodities = self._commodity_data.get(org_id, [])

        assessment = FLAGAssessment(
            tenant_id="default",
            org_id=org_id,
            inventory_id=inventory.id,
            flag_emissions_tco2e=Decimal(str(trigger.flag_emissions_tco2e)),
            total_emissions_tco2e=Decimal(str(trigger.total_emissions_tco2e)),
            flag_pct_of_total=Decimal(str(trigger.flag_pct_of_total)),
            flag_target_required=trigger.flag_target_required,
            commodity_data=commodities,
            sector_pathway_rate=Decimal(str(self.FLAG_SECTOR_RATE * 100)),
            deforestation_commitment=defor.commitment_declared,
            deforestation_commitment_date=defor.commitment_date,
            land_use_change_included=True,
        )

        self._assessments[org_id] = assessment

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FLAG assessment for org %s: required=%s, commodities=%d in %.1f ms",
            org_id, trigger.flag_target_required, len(commodities), elapsed_ms,
        )
        return assessment

    # ------------------------------------------------------------------
    # FLAG Report
    # ------------------------------------------------------------------

    def generate_flag_report(self, org_id: str) -> FLAGReport:
        """
        Generate a comprehensive FLAG assessment report.

        Args:
            org_id: Organization identifier.

        Returns:
            FLAGReport with all FLAG assessment results.
        """
        start = datetime.utcnow()

        trigger = self.assess_flag_trigger(org_id)
        defor = self.track_deforestation_commitment(org_id)
        separation = self.separate_flag_emissions(org_id)

        # Commodity pathways
        commodity_pathways: List[Dict[str, Any]] = []
        for cd in self._commodity_data.get(org_id, []):
            try:
                pathway = self.calculate_commodity_pathway(
                    commodity=cd.commodity.lower().replace(" ", "_"),
                    base_intensity=float(cd.base_intensity),
                )
                commodity_pathways.append(pathway.model_dump())
            except ValueError:
                logger.warning("Skipping unknown commodity: %s", cd.commodity)

        # Sector pathway
        sector_pathway = None
        if trigger.flag_emissions_tco2e > 0:
            sp = self.calculate_sector_pathway(trigger.flag_emissions_tco2e)
            sector_pathway = sp.model_dump()

        # Long-term assessment
        long_term = self.calculate_flag_long_term(org_id, trigger.flag_emissions_tco2e)

        # Benchmarks
        benchmarks: List[Dict[str, Any]] = []
        for cd in self._commodity_data.get(org_id, []):
            try:
                bm = self.get_commodity_benchmarks(
                    commodity=cd.commodity.lower().replace(" ", "_"),
                    company_intensity=float(cd.base_intensity),
                )
                benchmarks.append(bm.model_dump())
            except ValueError:
                pass

        provenance = _sha256(
            f"flag_report:{org_id}:{trigger.flag_pct_of_total}:{_now().isoformat()}"
        )

        report = FLAGReport(
            org_id=org_id,
            trigger_result=trigger.model_dump(),
            commodity_pathways=commodity_pathways,
            sector_pathway=sector_pathway,
            deforestation_status=defor.model_dump(),
            long_term_assessment=long_term.model_dump(),
            emission_separation=separation.model_dump(),
            benchmarks=benchmarks,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FLAG report for org %s: trigger=%s, commodities=%d in %.1f ms",
            org_id, trigger.flag_target_required, len(commodity_pathways), elapsed_ms,
        )
        return report
