"""
Removals Tracker -- ISO 14064-1:2018 Clause 6.2 GHG Removals

Implements GHG removal tracking as specified in ISO 14064-1:2018:
  - Clause 5.2.3: Reporting of removals alongside emissions
  - Clause 6.2: Quantification of GHG removals
  - Clause 9: Biogenic CO2 separate reporting

Manages seven removal types:
  - Forestry (afforestation, reforestation, improved forest management)
  - Soil carbon (regenerative agriculture, no-till)
  - CCS (carbon capture and storage)
  - Direct air capture (DACCS, DAC+S)
  - BECCS (bioenergy with carbon capture and storage)
  - Wetland restoration (mangroves, peatlands)
  - Ocean-based (alkalinity enhancement, seaweed farming)

Permanence assessment:
  - Permanent: geological storage, mineralization (>1000yr, discount=0%)
  - Long-term: >100yr forestry with monitoring (discount=10%)
  - Medium-term: 25-100yr soil carbon (discount=30%)
  - Short-term: <25yr temporary offsets (discount=60%)
  - Reversible: could be released at any time (discount=90%)

Uses in-memory storage for v1.0.

Example:
    >>> tracker = RemovalsTracker(config)
    >>> source = tracker.add_removal_source(
    ...     inventory_id="inv-1",
    ...     removal_type="forestry",
    ...     source_name="Amazonia reforestation project",
    ...     gross_removals_tco2e=Decimal("5000"),
    ...     permanence_level="long_term",
    ... )
    >>> net = tracker.calculate_net_emissions("inv-1", Decimal("50000"))
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    DataQualityTier,
    ISO14064AppConfig,
    PermanenceLevel,
    RemovalType,
    VerificationStage,
)
from .models import (
    RemovalSource,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permanence Discount Factors
# ---------------------------------------------------------------------------

PERMANENCE_DISCOUNT_FACTORS: Dict[PermanenceLevel, Decimal] = {
    PermanenceLevel.PERMANENT: Decimal("1.00"),    # No discount (geological, mineral)
    PermanenceLevel.LONG_TERM: Decimal("0.90"),    # 10% discount (>100yr forestry)
    PermanenceLevel.MEDIUM_TERM: Decimal("0.70"),  # 30% discount (25-100yr soil)
    PermanenceLevel.SHORT_TERM: Decimal("0.40"),   # 60% discount (<25yr offsets)
    PermanenceLevel.REVERSIBLE: Decimal("0.10"),   # 90% discount (may reverse)
}


# ---------------------------------------------------------------------------
# Permanence Classification Guidance
# ---------------------------------------------------------------------------

PERMANENCE_GUIDANCE: Dict[RemovalType, PermanenceLevel] = {
    RemovalType.CCS: PermanenceLevel.PERMANENT,
    RemovalType.DIRECT_AIR_CAPTURE: PermanenceLevel.PERMANENT,
    RemovalType.BECCS: PermanenceLevel.PERMANENT,
    RemovalType.FORESTRY: PermanenceLevel.LONG_TERM,
    RemovalType.WETLAND_RESTORATION: PermanenceLevel.LONG_TERM,
    RemovalType.SOIL_CARBON: PermanenceLevel.MEDIUM_TERM,
    RemovalType.OCEAN_BASED: PermanenceLevel.MEDIUM_TERM,
    RemovalType.OTHER: PermanenceLevel.SHORT_TERM,
}


class RemovalsTracker:
    """
    Tracks GHG removals per ISO 14064-1:2018 Clause 6.2.

    Responsibilities:
      - CRUD for removal sources
      - Permanence assessment and discount-factor application
      - Permanence-adjusted crediting
      - Net emissions calculation (gross_emissions - verified_removals)
      - Biogenic CO2 separate tracking
      - Removal verification status lifecycle
      - Querying by inventory, type, facility, and verification status

    All data is stored in-memory (dictionaries) for v1.0.
    """

    def __init__(self, config: Optional[ISO14064AppConfig] = None) -> None:
        """
        Initialize RemovalsTracker.

        Args:
            config: Application configuration.  Defaults are used if None.
        """
        self.config = config or ISO14064AppConfig()
        self._removal_sources: Dict[str, RemovalSource] = {}
        logger.info("RemovalsTracker initialized")

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def add_removal_source(
        self,
        inventory_id: str,
        removal_type: str,
        source_name: str,
        gross_removals_tco2e: Decimal,
        permanence_level: Optional[str] = None,
        facility_id: Optional[str] = None,
        biogenic_co2_removals: Decimal = Decimal("0"),
        biogenic_co2_emissions: Decimal = Decimal("0"),
        monitoring_plan: Optional[str] = None,
        data_quality_tier: str = "tier_2",
    ) -> RemovalSource:
        """
        Add a new GHG removal source to the tracker.

        Automatically determines the permanence level from the removal type
        if not explicitly provided, and applies the corresponding discount
        factor to compute credited (permanence-adjusted) removals.

        Args:
            inventory_id: Parent inventory ID.
            removal_type: Type of removal (forestry, soil_carbon, ccs, etc.).
            source_name: Human-readable removal source description.
            gross_removals_tco2e: Gross removals before permanence discount.
            permanence_level: Override permanence classification.
            facility_id: Optional facility/entity ID.
            biogenic_co2_removals: Biogenic CO2 removed from atmosphere (tCO2).
            biogenic_co2_emissions: Biogenic CO2 emitted from biomass (tCO2).
            monitoring_plan: Reference to the monitoring plan document.
            data_quality_tier: Data quality tier (tier_1 to tier_4).

        Returns:
            Newly created RemovalSource with credited removals computed.

        Raises:
            ValueError: If gross removals is negative or invalid type.
        """
        start = datetime.now(timezone.utc)

        if gross_removals_tco2e < 0:
            raise ValueError(
                f"Gross removals cannot be negative: {gross_removals_tco2e}"
            )

        rtype = RemovalType(removal_type)

        # Determine permanence level
        if permanence_level is not None:
            perm = PermanenceLevel(permanence_level)
        else:
            perm = PERMANENCE_GUIDANCE.get(rtype, PermanenceLevel.SHORT_TERM)

        # Apply permanence discount factor
        discount_factor = PERMANENCE_DISCOUNT_FACTORS[perm]
        credited = (gross_removals_tco2e * discount_factor).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        source = RemovalSource(
            inventory_id=inventory_id,
            facility_id=facility_id,
            removal_type=rtype,
            source_name=source_name.strip(),
            gross_removals_tco2e=gross_removals_tco2e,
            permanence_level=perm,
            permanence_discount_factor=discount_factor,
            credited_removals_tco2e=credited,
            biogenic_co2_removals=biogenic_co2_removals,
            biogenic_co2_emissions=biogenic_co2_emissions,
            monitoring_plan=monitoring_plan,
            data_quality_tier=DataQualityTier(data_quality_tier),
        )
        self._removal_sources[source.id] = source

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Added removal source '%s': type=%s, gross=%.4f, permanence=%s, "
            "discount=%.2f, credited=%.4f tCO2e in %.1f ms",
            source_name, rtype.value, gross_removals_tco2e, perm.value,
            discount_factor, credited, elapsed_ms,
        )
        return source

    def get_removal_source(self, source_id: str) -> Optional[RemovalSource]:
        """Retrieve a removal source by ID."""
        return self._removal_sources.get(source_id)

    def update_removal_source(
        self,
        source_id: str,
        source_name: Optional[str] = None,
        gross_removals_tco2e: Optional[Decimal] = None,
        permanence_level: Optional[str] = None,
        biogenic_co2_removals: Optional[Decimal] = None,
        biogenic_co2_emissions: Optional[Decimal] = None,
        monitoring_plan: Optional[str] = None,
        data_quality_tier: Optional[str] = None,
    ) -> RemovalSource:
        """
        Update an existing removal source.

        Recalculates credited removals if gross removals or permanence level
        are changed.

        Args:
            source_id: Removal source ID.
            source_name: New source name.
            gross_removals_tco2e: New gross removals.
            permanence_level: New permanence classification.
            biogenic_co2_removals: New biogenic CO2 removals.
            biogenic_co2_emissions: New biogenic CO2 emissions.
            monitoring_plan: New monitoring plan reference.
            data_quality_tier: New data quality tier.

        Returns:
            Updated RemovalSource.

        Raises:
            ValueError: If source not found.
        """
        source = self._get_source_or_raise(source_id)

        if source_name is not None:
            source.source_name = source_name.strip()
        if biogenic_co2_removals is not None:
            source.biogenic_co2_removals = biogenic_co2_removals
        if biogenic_co2_emissions is not None:
            source.biogenic_co2_emissions = biogenic_co2_emissions
        if monitoring_plan is not None:
            source.monitoring_plan = monitoring_plan
        if data_quality_tier is not None:
            source.data_quality_tier = DataQualityTier(data_quality_tier)

        # Recalculate credited removals if inputs changed
        recalculate = False
        if gross_removals_tco2e is not None:
            if gross_removals_tco2e < 0:
                raise ValueError(
                    f"Gross removals cannot be negative: {gross_removals_tco2e}"
                )
            source.gross_removals_tco2e = gross_removals_tco2e
            recalculate = True
        if permanence_level is not None:
            perm = PermanenceLevel(permanence_level)
            source.permanence_level = perm
            source.permanence_discount_factor = PERMANENCE_DISCOUNT_FACTORS[perm]
            recalculate = True

        if recalculate:
            source.credited_removals_tco2e = (
                source.gross_removals_tco2e * source.permanence_discount_factor
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        source.updated_at = _now()
        source.provenance_hash = _sha256(
            f"{source.inventory_id}:{source.removal_type}:"
            f"{source.gross_removals_tco2e}:{source.permanence_level}"
        )

        logger.info(
            "Updated removal source '%s' (id=%s): credited=%.4f tCO2e",
            source.source_name, source_id, source.credited_removals_tco2e,
        )
        return source

    def delete_removal_source(self, source_id: str) -> bool:
        """
        Delete a removal source.

        Args:
            source_id: Removal source ID.

        Returns:
            True if deleted successfully.

        Raises:
            ValueError: If source not found.
        """
        source = self._get_source_or_raise(source_id)
        del self._removal_sources[source_id]
        logger.info(
            "Deleted removal source '%s' (id=%s)",
            source.source_name, source_id,
        )
        return True

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def get_removals_by_inventory(
        self,
        inventory_id: str,
    ) -> List[RemovalSource]:
        """
        Get all removal sources for a given inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            List of RemovalSource objects sorted by credited amount descending.
        """
        sources = [
            s for s in self._removal_sources.values()
            if s.inventory_id == inventory_id
        ]
        return sorted(
            sources,
            key=lambda s: s.credited_removals_tco2e,
            reverse=True,
        )

    def get_removals_by_type(
        self,
        inventory_id: str,
        removal_type: str,
    ) -> List[RemovalSource]:
        """
        Get removal sources filtered by removal type.

        Args:
            inventory_id: Inventory ID.
            removal_type: Removal type (forestry, ccs, etc.).

        Returns:
            List of matching RemovalSource objects.
        """
        rtype = RemovalType(removal_type)
        return [
            s for s in self._removal_sources.values()
            if s.inventory_id == inventory_id and s.removal_type == rtype
        ]

    def get_removals_by_facility(
        self,
        inventory_id: str,
        facility_id: str,
    ) -> List[RemovalSource]:
        """
        Get removal sources filtered by facility/entity.

        Args:
            inventory_id: Inventory ID.
            facility_id: Facility/entity ID.

        Returns:
            List of matching RemovalSource objects.
        """
        return [
            s for s in self._removal_sources.values()
            if s.inventory_id == inventory_id and s.facility_id == facility_id
        ]

    def get_verified_removals(
        self,
        inventory_id: str,
    ) -> List[RemovalSource]:
        """
        Get only verified removal sources for an inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            List of verified RemovalSource objects.
        """
        return [
            s for s in self._removal_sources.values()
            if s.inventory_id == inventory_id
            and s.verification_status == VerificationStage.VERIFIED
        ]

    # ------------------------------------------------------------------
    # Verification Status Management
    # ------------------------------------------------------------------

    def update_verification_status(
        self,
        source_id: str,
        new_status: str,
    ) -> RemovalSource:
        """
        Update the verification status of a removal source.

        Status transitions:
          draft -> internal_review -> approved -> external_verification -> verified
          Any stage may also transition to draft (rejection / rework).

        Args:
            source_id: Removal source ID.
            new_status: New verification status.

        Returns:
            Updated RemovalSource.

        Raises:
            ValueError: If source not found or invalid status transition.
        """
        source = self._get_source_or_raise(source_id)
        target_status = VerificationStage(new_status)

        # Define valid transitions
        valid_transitions: Dict[VerificationStage, List[VerificationStage]] = {
            VerificationStage.DRAFT: [VerificationStage.INTERNAL_REVIEW],
            VerificationStage.INTERNAL_REVIEW: [
                VerificationStage.APPROVED,
                VerificationStage.DRAFT,
            ],
            VerificationStage.APPROVED: [
                VerificationStage.EXTERNAL_VERIFICATION,
                VerificationStage.DRAFT,
            ],
            VerificationStage.EXTERNAL_VERIFICATION: [
                VerificationStage.VERIFIED,
                VerificationStage.DRAFT,
            ],
            VerificationStage.VERIFIED: [VerificationStage.DRAFT],
        }

        allowed = valid_transitions.get(source.verification_status, [])
        if target_status not in allowed:
            raise ValueError(
                f"Invalid status transition: {source.verification_status.value} "
                f"-> {target_status.value}. Allowed: "
                f"{[s.value for s in allowed]}"
            )

        old_status = source.verification_status
        source.verification_status = target_status
        source.updated_at = _now()

        logger.info(
            "Verification status updated for '%s' (id=%s): %s -> %s",
            source.source_name, source_id,
            old_status.value, target_status.value,
        )
        return source

    # ------------------------------------------------------------------
    # Permanence Assessment
    # ------------------------------------------------------------------

    def assess_permanence(
        self,
        removal_type: str,
        storage_duration_years: Optional[int] = None,
        has_monitoring: bool = False,
        has_buffer_pool: bool = False,
    ) -> Dict[str, Any]:
        """
        Assess the permanence level for a removal type.

        Uses the removal type's default permanence level as a starting
        point, then adjusts based on storage duration, monitoring plan,
        and buffer pool availability.

        Args:
            removal_type: Type of removal activity.
            storage_duration_years: Expected storage duration in years.
            has_monitoring: Whether a monitoring plan is in place.
            has_buffer_pool: Whether a buffer pool/insurance exists.

        Returns:
            Dict with permanence assessment result.
        """
        rtype = RemovalType(removal_type)
        base_level = PERMANENCE_GUIDANCE.get(rtype, PermanenceLevel.SHORT_TERM)

        # Adjust based on storage duration if provided
        if storage_duration_years is not None:
            if storage_duration_years > 1000:
                assessed_level = PermanenceLevel.PERMANENT
            elif storage_duration_years > 100:
                assessed_level = PermanenceLevel.LONG_TERM
            elif storage_duration_years > 25:
                assessed_level = PermanenceLevel.MEDIUM_TERM
            elif storage_duration_years > 0:
                assessed_level = PermanenceLevel.SHORT_TERM
            else:
                assessed_level = PermanenceLevel.REVERSIBLE
        else:
            assessed_level = base_level

        discount_factor = PERMANENCE_DISCOUNT_FACTORS[assessed_level]

        # Monitoring and buffer pool can improve confidence but do not
        # change the permanence class directly
        confidence_bonus = Decimal("0")
        if has_monitoring:
            confidence_bonus += Decimal("0.05")
        if has_buffer_pool:
            confidence_bonus += Decimal("0.05")

        # Effective factor capped at 1.0
        effective_factor = min(
            discount_factor + confidence_bonus, Decimal("1.00")
        )

        result = {
            "removal_type": rtype.value,
            "base_permanence": base_level.value,
            "assessed_permanence": assessed_level.value,
            "storage_duration_years": storage_duration_years,
            "has_monitoring": has_monitoring,
            "has_buffer_pool": has_buffer_pool,
            "base_discount_factor": str(discount_factor),
            "confidence_bonus": str(confidence_bonus),
            "effective_discount_factor": str(effective_factor),
            "description": self._permanence_description(assessed_level),
        }

        logger.info(
            "Permanence assessment: type=%s, level=%s, factor=%.2f (effective=%.2f)",
            rtype.value, assessed_level.value, discount_factor, effective_factor,
        )
        return result

    # ------------------------------------------------------------------
    # Net Emissions Calculation
    # ------------------------------------------------------------------

    def calculate_net_emissions(
        self,
        inventory_id: str,
        gross_emissions_tco2e: Decimal,
        verified_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate net emissions: gross_emissions - verified_removals.

        Per ISO 14064-1, net emissions account for both anthropogenic
        emissions and verified removals.  The result may be negative
        if removals exceed emissions (net-negative).

        Args:
            inventory_id: Inventory ID.
            gross_emissions_tco2e: Total gross emissions (tCO2e).
            verified_only: If True, only count verified removals.

        Returns:
            Dict with:
              - gross_emissions_tco2e: Input gross emissions.
              - total_gross_removals: Sum of gross removals.
              - total_credited_removals: Sum of credited (adjusted) removals.
              - net_emissions_tco2e: gross - credited.
              - is_net_negative: Whether net < 0.
              - by_type: Breakdown by removal type.
              - by_permanence: Breakdown by permanence level.
        """
        start = datetime.now(timezone.utc)

        if verified_only:
            sources = self.get_verified_removals(inventory_id)
        else:
            sources = self.get_removals_by_inventory(inventory_id)

        total_gross = Decimal("0")
        total_credited = Decimal("0")
        by_type: Dict[str, Decimal] = {}
        by_permanence: Dict[str, Decimal] = {}

        for src in sources:
            total_gross += src.gross_removals_tco2e
            total_credited += src.credited_removals_tco2e

            type_key = src.removal_type.value
            by_type[type_key] = (
                by_type.get(type_key, Decimal("0"))
                + src.credited_removals_tco2e
            )

            perm_key = src.permanence_level.value
            by_permanence[perm_key] = (
                by_permanence.get(perm_key, Decimal("0"))
                + src.credited_removals_tco2e
            )

        net = gross_emissions_tco2e - total_credited
        is_net_negative = net < 0

        provenance = _sha256(
            f"net:{inventory_id}:{gross_emissions_tco2e}:"
            f"{total_credited}:{net}"
        )

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Net emissions for inventory %s: gross=%.4f - removals=%.4f = "
            "net=%.4f tCO2e (verified_only=%s) in %.1f ms",
            inventory_id, gross_emissions_tco2e, total_credited,
            net, verified_only, elapsed_ms,
        )

        return {
            "inventory_id": inventory_id,
            "gross_emissions_tco2e": gross_emissions_tco2e,
            "total_gross_removals": total_gross,
            "total_credited_removals": total_credited,
            "net_emissions_tco2e": net,
            "is_net_negative": is_net_negative,
            "removal_source_count": len(sources),
            "verified_only": verified_only,
            "by_type": by_type,
            "by_permanence": by_permanence,
            "provenance_hash": provenance,
        }

    # ------------------------------------------------------------------
    # Biogenic CO2 Tracking
    # ------------------------------------------------------------------

    def calculate_biogenic_balance(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate the biogenic CO2 balance for an inventory.

        Per ISO 14064-1 and the GHG Protocol, biogenic CO2 is reported
        separately from the main inventory.  This method computes:
          - Total biogenic CO2 removals (sequestration)
          - Total biogenic CO2 emissions (biomass combustion)
          - Net biogenic CO2 balance

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with biogenic CO2 balance details.
        """
        sources = self.get_removals_by_inventory(inventory_id)

        total_bio_removals = Decimal("0")
        total_bio_emissions = Decimal("0")
        by_type: Dict[str, Dict[str, Decimal]] = {}

        for src in sources:
            total_bio_removals += src.biogenic_co2_removals
            total_bio_emissions += src.biogenic_co2_emissions

            type_key = src.removal_type.value
            if type_key not in by_type:
                by_type[type_key] = {
                    "removals": Decimal("0"),
                    "emissions": Decimal("0"),
                }
            by_type[type_key]["removals"] += src.biogenic_co2_removals
            by_type[type_key]["emissions"] += src.biogenic_co2_emissions

        net_biogenic = total_bio_removals - total_bio_emissions

        # Convert by_type Decimals to strings for serialization
        by_type_str: Dict[str, Dict[str, str]] = {}
        for type_key, values in by_type.items():
            by_type_str[type_key] = {
                "removals": str(values["removals"]),
                "emissions": str(values["emissions"]),
                "net": str(values["removals"] - values["emissions"]),
            }

        logger.info(
            "Biogenic CO2 balance for inventory %s: removals=%.4f, "
            "emissions=%.4f, net=%.4f tCO2",
            inventory_id, total_bio_removals, total_bio_emissions, net_biogenic,
        )

        return {
            "inventory_id": inventory_id,
            "total_biogenic_removals_tco2": total_bio_removals,
            "total_biogenic_emissions_tco2": total_bio_emissions,
            "net_biogenic_tco2": net_biogenic,
            "is_net_sequestration": net_biogenic > 0,
            "by_type": by_type_str,
            "source_count": len(sources),
        }

    # ------------------------------------------------------------------
    # Summary / Aggregation
    # ------------------------------------------------------------------

    def get_removals_summary(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all removals for an inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with summary statistics, breakdowns, and status counts.
        """
        sources = self.get_removals_by_inventory(inventory_id)

        if not sources:
            return {
                "inventory_id": inventory_id,
                "source_count": 0,
                "total_gross_removals": Decimal("0"),
                "total_credited_removals": Decimal("0"),
                "by_type": {},
                "by_permanence": {},
                "by_verification_status": {},
                "biogenic_balance": {},
            }

        total_gross = Decimal("0")
        total_credited = Decimal("0")
        by_type: Dict[str, Dict[str, Any]] = {}
        by_permanence: Dict[str, Decimal] = {}
        by_status: Dict[str, int] = {}

        for src in sources:
            total_gross += src.gross_removals_tco2e
            total_credited += src.credited_removals_tco2e

            # By type
            type_key = src.removal_type.value
            if type_key not in by_type:
                by_type[type_key] = {
                    "count": 0,
                    "gross": Decimal("0"),
                    "credited": Decimal("0"),
                }
            by_type[type_key]["count"] += 1
            by_type[type_key]["gross"] += src.gross_removals_tco2e
            by_type[type_key]["credited"] += src.credited_removals_tco2e

            # By permanence
            perm_key = src.permanence_level.value
            by_permanence[perm_key] = (
                by_permanence.get(perm_key, Decimal("0"))
                + src.credited_removals_tco2e
            )

            # By verification status
            status_key = src.verification_status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

        # Compute discount impact
        discount_impact = total_gross - total_credited

        # Serialize by_type for output
        by_type_out: Dict[str, Dict[str, Any]] = {}
        for type_key, data in by_type.items():
            by_type_out[type_key] = {
                "count": data["count"],
                "gross_tco2e": str(data["gross"]),
                "credited_tco2e": str(data["credited"]),
                "discount_impact_tco2e": str(data["gross"] - data["credited"]),
            }

        biogenic = self.calculate_biogenic_balance(inventory_id)

        logger.info(
            "Removals summary for inventory %s: %d sources, gross=%.4f, "
            "credited=%.4f, discount_impact=%.4f tCO2e",
            inventory_id, len(sources), total_gross,
            total_credited, discount_impact,
        )

        return {
            "inventory_id": inventory_id,
            "source_count": len(sources),
            "total_gross_removals": total_gross,
            "total_credited_removals": total_credited,
            "permanence_discount_impact": discount_impact,
            "by_type": by_type_out,
            "by_permanence": {k: str(v) for k, v in by_permanence.items()},
            "by_verification_status": by_status,
            "biogenic_balance": biogenic,
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_source_or_raise(self, source_id: str) -> RemovalSource:
        """Retrieve a removal source or raise ValueError."""
        source = self._removal_sources.get(source_id)
        if source is None:
            raise ValueError(f"Removal source not found: {source_id}")
        return source

    def _permanence_description(self, level: PermanenceLevel) -> str:
        """Get a human-readable description of a permanence level."""
        descriptions: Dict[PermanenceLevel, str] = {
            PermanenceLevel.PERMANENT: (
                "Permanent storage (>1000 years). Examples: geological CCS, "
                "mineral carbonation. No reversal risk discount applied."
            ),
            PermanenceLevel.LONG_TERM: (
                "Long-term storage (100-1000 years). Examples: managed "
                "forests with monitoring plans, deep ocean storage. "
                "10% discount for reversal risk."
            ),
            PermanenceLevel.MEDIUM_TERM: (
                "Medium-term storage (25-100 years). Examples: soil carbon "
                "sequestration, wetland restoration. 30% discount for "
                "reversal risk."
            ),
            PermanenceLevel.SHORT_TERM: (
                "Short-term storage (<25 years). Examples: temporary "
                "carbon offsets, short-rotation crops. 60% discount for "
                "reversal risk."
            ),
            PermanenceLevel.REVERSIBLE: (
                "Reversible storage (could be released at any time). "
                "Examples: unmanaged forests without monitoring. "
                "90% discount for reversal risk."
            ),
        }
        return descriptions.get(level, "Unknown permanence level")
