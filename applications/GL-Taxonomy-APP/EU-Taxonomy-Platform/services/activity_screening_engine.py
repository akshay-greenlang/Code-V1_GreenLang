"""
Activity Screening Engine -- NACE Code Mapping, Eligibility, Sector Classification

Implements the first step of the EU Taxonomy alignment pipeline: determining
whether an economic activity is *taxonomy-eligible* by mapping NACE codes to
the activity catalogue in the Climate Delegated Act (EU 2021/2139) and the
Environmental Delegated Act (EU 2023/2486).

Key capabilities:
  - NACE code lookup and activity matching
  - Eligibility screening per environmental objective
  - Batch screening for multiple NACE codes
  - Sector-level activity breakdown
  - De minimis filtering (activities below 10% threshold)
  - Activity catalogue browsing and text search
  - Eligibility summary statistics

All calculations are deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852, Articles 3 and 5-8
    - Climate Delegated Act (EU) 2021/2139 Annexes I-II
    - Environmental Delegated Act (EU) 2023/2486 Annexes I-IV
    - NACE Rev. 2 Statistical Classification (Eurostat)

Example:
    >>> engine = ActivityScreeningEngine(config)
    >>> match = engine.screen_activity("D35.11", "Solar PV", "climate_mitigation")
    >>> match.is_eligible
    True
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    ActivityType,
    AlignmentStatus,
    EnvironmentalObjective,
    Sector,
    TaxonomyAppConfig,
    ENVIRONMENTAL_OBJECTIVES,
    TAXONOMY_ACTIVITIES,
)
from .models import (
    EconomicActivity,
    Organization,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ActivityMatch(BaseModel):
    """Result of mapping a NACE code to a taxonomy activity."""

    nace_code: str = Field(..., description="Input NACE code")
    matched: bool = Field(default=False, description="Whether a taxonomy match was found")
    activity_code: Optional[str] = Field(None, description="Matched taxonomy activity code")
    activity_name: Optional[str] = Field(None, description="Matched activity name")
    objective: Optional[str] = Field(None, description="Environmental objective")
    activity_type: Optional[str] = Field(None, description="enabling/transitional/own_performance")
    tsc_summary: Optional[str] = Field(None, description="TSC summary text")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Match confidence")
    is_eligible: bool = Field(default=False, description="Whether activity is taxonomy-eligible")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EligibilityResult(BaseModel):
    """Full eligibility screening result for an organization."""

    org_id: str = Field(...)
    total_activities: int = Field(default=0)
    eligible_count: int = Field(default=0)
    not_eligible_count: int = Field(default=0)
    eligibility_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    eligible_turnover_eur: Decimal = Field(default=Decimal("0"))
    eligible_capex_eur: Decimal = Field(default=Decimal("0"))
    eligible_opex_eur: Decimal = Field(default=Decimal("0"))
    activity_matches: List[ActivityMatch] = Field(default_factory=list)
    screened_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class SectorBreakdown(BaseModel):
    """Activities grouped by NACE sector."""

    org_id: str = Field(...)
    sectors: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_activities: int = Field(default=0)
    sector_count: int = Field(default=0)
    dominant_sector: Optional[str] = Field(None)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BatchScreeningResult(BaseModel):
    """Result of batch NACE code screening."""

    org_id: str = Field(...)
    total_codes: int = Field(default=0)
    matched_count: int = Field(default=0)
    unmatched_count: int = Field(default=0)
    matches: List[ActivityMatch] = Field(default_factory=list)
    unmatched_codes: List[str] = Field(default_factory=list)
    screening_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class NACELookupResult(BaseModel):
    """NACE code details and taxonomy mapping."""

    nace_code: str = Field(...)
    found: bool = Field(default=False)
    section: Optional[str] = Field(None, description="NACE section letter")
    description: Optional[str] = Field(None, description="NACE code description")
    taxonomy_activities: List[Dict[str, Any]] = Field(default_factory=list)
    activity_count: int = Field(default=0)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# NACE Section Descriptions (for sector breakdown)
# ---------------------------------------------------------------------------

_NACE_SECTIONS: Dict[str, str] = {
    "A": "Agriculture, Forestry and Fishing",
    "B": "Mining and Quarrying",
    "C": "Manufacturing",
    "D": "Electricity, Gas, Steam and Air Conditioning Supply",
    "E": "Water Supply; Sewerage, Waste Management and Remediation",
    "F": "Construction",
    "G": "Wholesale and Retail Trade",
    "H": "Transportation and Storage",
    "I": "Accommodation and Food Service Activities",
    "J": "Information and Communication",
    "K": "Financial and Insurance Activities",
    "L": "Real Estate Activities",
    "M": "Professional, Scientific and Technical Activities",
    "N": "Administrative and Support Service Activities",
}


# ---------------------------------------------------------------------------
# NACE Code to Description (extended subset for lookups)
# ---------------------------------------------------------------------------

_NACE_DESCRIPTIONS: Dict[str, str] = {
    "A02.10": "Silviculture and other forestry activities",
    "C20.11": "Manufacture of industrial gases",
    "C23.51": "Manufacture of cement",
    "C24.10": "Manufacture of basic iron and steel and of ferro-alloys",
    "C24.42": "Aluminium production",
    "C27.20": "Manufacture of batteries and accumulators",
    "C29.10": "Manufacture of motor vehicles",
    "D35.11": "Production of electricity",
    "D35.12": "Transmission of electricity",
    "D35.13": "Distribution of electricity",
    "D35.21": "Manufacture of gas",
    "D35.22": "Distribution of gaseous fuels through mains",
    "D35.30": "Steam and air conditioning supply",
    "E36.00": "Water collection, treatment and supply",
    "E37.00": "Sewerage",
    "E38.11": "Collection of non-hazardous waste",
    "E38.21": "Treatment and disposal of non-hazardous waste",
    "E38.32": "Recovery of sorted materials",
    "F41.10": "Development of building projects",
    "F41.20": "Construction of residential and non-residential buildings",
    "F42.11": "Construction of roads and motorways",
    "F42.12": "Construction of railways and underground railways",
    "F42.13": "Construction of bridges and tunnels",
    "F42.22": "Construction of utility projects for electricity and telecommunications",
    "F42.91": "Construction of water projects",
    "F42.99": "Construction of other civil engineering projects n.e.c.",
    "F43.00": "Specialised construction activities",
    "F43.21": "Electrical installation",
    "F43.22": "Plumbing, heat and air-conditioning installation",
    "F43.29": "Other construction installation",
    "H49.10": "Passenger rail transport, interurban",
    "H49.20": "Freight rail transport",
    "H49.31": "Urban and suburban passenger land transport",
    "H49.32": "Taxi operation",
    "H49.39": "Other passenger land transport n.e.c.",
    "H49.41": "Freight transport by road",
    "H49.42": "Removal services",
    "H50.10": "Sea and coastal passenger water transport",
    "H50.20": "Sea and coastal freight water transport",
    "H50.30": "Inland passenger water transport",
    "H50.40": "Inland freight water transport",
    "H52.22": "Service activities incidental to water transportation",
    "J61.00": "Telecommunications",
    "J62.00": "Computer programming, consultancy and related activities",
    "J63.11": "Data processing, hosting and related activities",
    "L68.10": "Buying and selling of own real estate",
    "L68.20": "Renting and operating of own or leased real estate",
    "M71.11": "Architectural activities",
    "M71.12": "Engineering activities and related technical consultancy",
    "M71.20": "Technical testing and analysis",
    "M72.10": "R&D on natural sciences and engineering",
    "M72.19": "Other R&D on natural sciences and engineering",
    "N77.11": "Renting and leasing of cars and light motor vehicles",
    "N77.21": "Renting and leasing of recreational and sports goods",
}


# ---------------------------------------------------------------------------
# Build reverse index: NACE code -> list of (activity_code, activity_data)
# ---------------------------------------------------------------------------

def _build_nace_index() -> Dict[str, List[Dict[str, Any]]]:
    """Build a reverse index from NACE code to taxonomy activities."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for act_code, act_data in TAXONOMY_ACTIVITIES.items():
        nace_codes = act_data.get("nace_codes", [])
        objectives = act_data.get("objectives", [])
        first_objective = objectives[0] if objectives else ""
        for nc in nace_codes:
            entry = {
                "activity_code": act_code,
                "name": act_data.get("name", ""),
                "objective": first_objective,
                "type": act_data.get("activity_type", "own_performance"),
                "tsc_summary": act_data.get("sc_criteria_ref", ""),
            }
            index.setdefault(nc, []).append(entry)
    return index


_NACE_TO_ACTIVITIES: Dict[str, List[Dict[str, Any]]] = _build_nace_index()


# ---------------------------------------------------------------------------
# ActivityScreeningEngine
# ---------------------------------------------------------------------------

class ActivityScreeningEngine:
    """
    Activity Screening Engine for EU Taxonomy eligibility assessment.

    Implements NACE code mapping, eligibility screening, sector classification,
    and de minimis filtering.  Uses the Climate Delegated Act and Environmental
    Delegated Act activity catalogues to determine whether an economic activity
    is taxonomy-eligible.

    Attributes:
        config: Application configuration.
        _organizations: In-memory store keyed by org_id.
        _activities: In-memory store keyed by org_id -> list of activities.
        _screening_results: Cache of screening results keyed by org_id.

    Example:
        >>> engine = ActivityScreeningEngine(config)
        >>> engine.register_organization("org-1", "Acme Corp", "non_financial", "D", "DE")
        >>> match = engine.screen_activity("D35.11", "Solar PV electricity", "climate_mitigation")
        >>> match.is_eligible
        True
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """
        Initialize ActivityScreeningEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or TaxonomyAppConfig()
        self._organizations: Dict[str, Dict[str, Any]] = {}
        self._activities: Dict[str, List[EconomicActivity]] = {}
        self._screening_results: Dict[str, EligibilityResult] = {}
        logger.info("ActivityScreeningEngine initialized")

    # ------------------------------------------------------------------
    # Organization Management
    # ------------------------------------------------------------------

    def register_organization(
        self,
        org_id: str,
        name: str,
        entity_type: str = "non_financial",
        sector: str = "C",
        country: str = "EU",
    ) -> Dict[str, Any]:
        """
        Register an organization for screening.

        Args:
            org_id: Unique organization identifier.
            name: Legal entity name.
            entity_type: Entity classification (non_financial, credit_institution, etc.).
            sector: Primary NACE sector letter (A-M).
            country: ISO country code.

        Returns:
            Organization registration record.
        """
        record = {
            "org_id": org_id,
            "name": name,
            "entity_type": entity_type,
            "sector": sector,
            "country": country,
            "registered_at": _now().isoformat(),
        }
        self._organizations[org_id] = record
        self._activities.setdefault(org_id, [])
        logger.info("Registered organization %s (%s) sector=%s", org_id, name, sector)
        return record

    # ------------------------------------------------------------------
    # Core Screening Methods
    # ------------------------------------------------------------------

    def screen_activity(
        self,
        nace_code: str,
        description: str = "",
        objective: str = "climate_mitigation",
    ) -> ActivityMatch:
        """
        Screen a single NACE code against the taxonomy activity catalogue.

        Maps the NACE code to taxonomy activities and determines eligibility.
        If multiple activities match, the first match for the requested
        objective is returned.

        Args:
            nace_code: NACE Rev.2 code (e.g. 'D35.11').
            description: Activity description for context (used in provenance).
            objective: Environmental objective to screen against.

        Returns:
            ActivityMatch with eligibility determination.

        Example:
            >>> match = engine.screen_activity("D35.11", "Solar PV", "climate_mitigation")
            >>> match.is_eligible
            True
        """
        start = datetime.utcnow()
        normalised = nace_code.strip().upper()

        # Look up in reverse index
        matching_activities = _NACE_TO_ACTIVITIES.get(normalised, [])

        if not matching_activities:
            provenance = _sha256(f"screen:{normalised}:{objective}:no_match")
            logger.debug("No taxonomy match for NACE %s", normalised)
            return ActivityMatch(
                nace_code=normalised,
                matched=False,
                is_eligible=False,
                confidence=0.0,
                provenance_hash=provenance,
            )

        # Find best match for requested objective
        best_match: Optional[Dict[str, Any]] = None
        for act in matching_activities:
            if act["objective"] == objective:
                best_match = act
                break

        # Fallback to first match regardless of objective
        if best_match is None:
            best_match = matching_activities[0]

        is_objective_match = best_match["objective"] == objective
        confidence = 1.0 if is_objective_match else 0.7

        provenance = _sha256(
            f"screen:{normalised}:{objective}:{best_match['activity_code']}"
        )

        match = ActivityMatch(
            nace_code=normalised,
            matched=True,
            activity_code=best_match["activity_code"],
            activity_name=best_match["name"],
            objective=best_match["objective"],
            activity_type=best_match["type"],
            tsc_summary=best_match.get("tsc_summary", ""),
            confidence=confidence,
            is_eligible=True,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Screened NACE %s -> activity %s (eligible=%s) in %.1f ms",
            normalised, best_match["activity_code"], True, elapsed_ms,
        )
        return match

    def screen_eligibility(
        self,
        org_id: str,
        activities: List[EconomicActivity],
    ) -> EligibilityResult:
        """
        Perform full eligibility screening for an organization's activities.

        Screens each activity's NACE code and computes aggregate eligibility
        metrics including eligible turnover, CapEx, and OpEx.

        Args:
            org_id: Organization identifier (must be registered).
            activities: List of economic activities to screen.

        Returns:
            EligibilityResult with per-activity matches and aggregates.

        Raises:
            ValueError: If organization is not registered.
        """
        if org_id not in self._organizations:
            raise ValueError(f"Organization {org_id} not registered")

        start = datetime.utcnow()
        matches: List[ActivityMatch] = []
        eligible_count = 0
        eligible_turnover = Decimal("0")
        eligible_capex = Decimal("0")
        eligible_opex = Decimal("0")

        for activity in activities:
            nace = activity.nace_code or ""
            objective = activity.objective or "climate_mitigation"
            match = self.screen_activity(nace, activity.name, objective)
            matches.append(match)

            if match.is_eligible:
                eligible_count += 1
                eligible_turnover += activity.turnover_eur
                eligible_capex += activity.capex_eur
                eligible_opex += activity.opex_eur

            # Store activity for later use
            self._activities.setdefault(org_id, []).append(activity)

        total = len(activities)
        ratio = eligible_count / total if total > 0 else 0.0

        provenance = _sha256(
            f"eligibility:{org_id}:{total}:{eligible_count}:{float(eligible_turnover)}"
        )

        result = EligibilityResult(
            org_id=org_id,
            total_activities=total,
            eligible_count=eligible_count,
            not_eligible_count=total - eligible_count,
            eligibility_ratio=round(ratio, 4),
            eligible_turnover_eur=eligible_turnover,
            eligible_capex_eur=eligible_capex,
            eligible_opex_eur=eligible_opex,
            activity_matches=matches,
            provenance_hash=provenance,
        )

        self._screening_results[org_id] = result

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Eligibility screening for org %s: %d/%d eligible (%.1f%%) in %.1f ms",
            org_id, eligible_count, total, ratio * 100, elapsed_ms,
        )
        return result

    def batch_screen(
        self,
        org_id: str,
        nace_codes: List[str],
    ) -> BatchScreeningResult:
        """
        Batch-screen multiple NACE codes for eligibility.

        Efficient bulk screening that processes a list of NACE codes and
        returns matched and unmatched results.

        Args:
            org_id: Organization identifier.
            nace_codes: List of NACE codes to screen.

        Returns:
            BatchScreeningResult with matches and unmatched codes.
        """
        start = datetime.utcnow()
        matches: List[ActivityMatch] = []
        unmatched: List[str] = []

        for code in nace_codes:
            match = self.screen_activity(code)
            matches.append(match)
            if not match.matched:
                unmatched.append(code.strip().upper())

        matched_count = len(matches) - len(unmatched)
        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        provenance = _sha256(
            f"batch_screen:{org_id}:{len(nace_codes)}:{matched_count}"
        )

        result = BatchScreeningResult(
            org_id=org_id,
            total_codes=len(nace_codes),
            matched_count=matched_count,
            unmatched_count=len(unmatched),
            matches=matches,
            unmatched_codes=unmatched,
            screening_time_ms=round(elapsed_ms, 2),
            provenance_hash=provenance,
        )

        logger.info(
            "Batch screening for org %s: %d/%d matched in %.1f ms",
            org_id, matched_count, len(nace_codes), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Sector Analysis
    # ------------------------------------------------------------------

    def get_sector_breakdown(self, org_id: str) -> SectorBreakdown:
        """
        Get activity breakdown by NACE sector for an organization.

        Groups registered activities by their NACE section letter and
        computes per-sector counts and financial totals.

        Args:
            org_id: Organization identifier.

        Returns:
            SectorBreakdown with per-sector data.
        """
        activities = self._activities.get(org_id, [])
        sectors: Dict[str, Dict[str, Any]] = {}

        for act in activities:
            nace = act.nace_code or ""
            section = nace[0].upper() if nace else "UNKNOWN"
            section_name = _NACE_SECTIONS.get(section, "Unknown Section")

            if section not in sectors:
                sectors[section] = {
                    "name": section_name,
                    "activity_count": 0,
                    "turnover_eur": Decimal("0"),
                    "capex_eur": Decimal("0"),
                    "opex_eur": Decimal("0"),
                    "activities": [],
                }

            sectors[section]["activity_count"] += 1
            sectors[section]["turnover_eur"] += act.turnover_eur
            sectors[section]["capex_eur"] += act.capex_eur
            sectors[section]["opex_eur"] += act.opex_eur
            sectors[section]["activities"].append(act.activity_code)

        # Determine dominant sector by activity count
        dominant = None
        max_count = 0
        for sec, data in sectors.items():
            if data["activity_count"] > max_count:
                max_count = data["activity_count"]
                dominant = sec

        # Convert Decimals to float for serialization in nested dicts
        serializable_sectors: Dict[str, Dict[str, Any]] = {}
        for sec, data in sectors.items():
            serializable_sectors[sec] = {
                "name": data["name"],
                "activity_count": data["activity_count"],
                "turnover_eur": float(data["turnover_eur"]),
                "capex_eur": float(data["capex_eur"]),
                "opex_eur": float(data["opex_eur"]),
                "activities": data["activities"],
            }

        provenance = _sha256(f"sector_breakdown:{org_id}:{len(activities)}")

        return SectorBreakdown(
            org_id=org_id,
            sectors=serializable_sectors,
            total_activities=len(activities),
            sector_count=len(sectors),
            dominant_sector=dominant,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # NACE Lookup
    # ------------------------------------------------------------------

    def lookup_nace(self, nace_code: str) -> NACELookupResult:
        """
        Look up detailed information for a NACE code.

        Returns the NACE description, section, and all taxonomy activities
        associated with the code.

        Args:
            nace_code: NACE Rev.2 code (e.g. 'D35.11').

        Returns:
            NACELookupResult with code details and taxonomy mappings.
        """
        normalised = nace_code.strip().upper()
        section = normalised[0] if normalised else ""
        description = _NACE_DESCRIPTIONS.get(normalised)
        matched_activities = _NACE_TO_ACTIVITIES.get(normalised, [])

        found = description is not None or len(matched_activities) > 0
        provenance = _sha256(f"nace_lookup:{normalised}:{found}")

        return NACELookupResult(
            nace_code=normalised,
            found=found,
            section=section if found else None,
            description=description,
            taxonomy_activities=matched_activities,
            activity_count=len(matched_activities),
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # De Minimis Filtering
    # ------------------------------------------------------------------

    def apply_de_minimis(
        self,
        org_id: str,
        threshold: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """
        Filter activities below de minimis threshold.

        Activities whose turnover represents less than the threshold
        fraction of total organizational turnover are flagged as de minimis.

        Args:
            org_id: Organization identifier.
            threshold: De minimis fraction (default 0.10 = 10%).

        Returns:
            List of activity records with de_minimis flag and turnover share.
        """
        activities = self._activities.get(org_id, [])
        if not activities:
            logger.warning("No activities registered for org %s", org_id)
            return []

        total_turnover = sum(
            float(a.turnover_eur) for a in activities
        )

        results: List[Dict[str, Any]] = []
        for act in activities:
            act_turnover = float(act.turnover_eur)
            share = act_turnover / total_turnover if total_turnover > 0 else 0.0
            is_de_minimis = share < threshold

            results.append({
                "activity_code": act.activity_code,
                "nace_code": act.nace_code or "",
                "name": act.name,
                "turnover_eur": act_turnover,
                "turnover_share": round(share, 6),
                "threshold": threshold,
                "is_de_minimis": is_de_minimis,
            })

        above_count = sum(1 for r in results if not r["is_de_minimis"])
        below_count = sum(1 for r in results if r["is_de_minimis"])
        logger.info(
            "De minimis filter for org %s: %d above threshold, %d below (%.0f%%)",
            org_id, above_count, below_count, threshold * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Activity Catalogue
    # ------------------------------------------------------------------

    def get_activity_catalog(
        self,
        sector: Optional[str] = None,
        objective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Browse the taxonomy activity catalogue with optional filters.

        Returns taxonomy activities filtered by NACE sector and/or
        environmental objective.

        Args:
            sector: NACE section letter filter (e.g. 'D' for Energy).
            objective: Environmental objective filter (e.g. 'climate_mitigation').

        Returns:
            List of activity catalogue entries matching filters.
        """
        results: List[Dict[str, Any]] = []

        for act_code, act_data in TAXONOMY_ACTIVITIES.items():
            # Sector filter: match first letter of NACE code
            if sector is not None:
                nace_codes = act_data.get("nace_codes", [])
                sector_match = any(nc.startswith(sector.upper()) for nc in nace_codes)
                if not sector_match:
                    continue

            # Objective filter
            act_objectives = act_data.get("objectives", [])
            if objective is not None and objective not in act_objectives:
                continue

            first_obj = act_objectives[0] if act_objectives else ""
            results.append({
                "activity_code": act_code,
                "name": act_data.get("name", ""),
                "nace": ", ".join(act_data.get("nace_codes", [])),
                "objective": first_obj,
                "type": act_data.get("activity_type", "own_performance"),
                "tsc_summary": act_data.get("sc_criteria_ref", ""),
            })

        logger.debug(
            "Activity catalogue query: sector=%s, objective=%s -> %d results",
            sector, objective, len(results),
        )
        return results

    def search_activities(self, query: str) -> List[Dict[str, Any]]:
        """
        Text search in activity names and descriptions.

        Performs case-insensitive substring matching against activity names
        and TSC summaries in the taxonomy catalogue.

        Args:
            query: Search query string.

        Returns:
            List of matching activity catalogue entries.
        """
        query_lower = query.strip().lower()
        if not query_lower:
            return []

        results: List[Dict[str, Any]] = []
        for act_code, act_data in TAXONOMY_ACTIVITIES.items():
            name = act_data.get("name", "").lower()
            sc_ref = act_data.get("sc_criteria_ref", "").lower()

            if query_lower in name or query_lower in sc_ref:
                act_objectives = act_data.get("objectives", [])
                first_obj = act_objectives[0] if act_objectives else ""
                results.append({
                    "activity_code": act_code,
                    "name": act_data.get("name", ""),
                    "nace": ", ".join(act_data.get("nace_codes", [])),
                    "objective": first_obj,
                    "type": act_data.get("activity_type", "own_performance"),
                    "tsc_summary": act_data.get("sc_criteria_ref", ""),
                })

        logger.debug("Activity search '%s' -> %d results", query, len(results))
        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_eligibility_summary(self, org_id: str) -> Dict[str, Any]:
        """
        Get eligibility summary statistics for an organization.

        Returns aggregate eligibility metrics from the most recent
        screening result.

        Args:
            org_id: Organization identifier.

        Returns:
            Dictionary with eligibility summary statistics.
        """
        cached = self._screening_results.get(org_id)
        activities = self._activities.get(org_id, [])

        if cached is not None:
            # Compute objective-level breakdown from matches
            obj_breakdown: Dict[str, int] = {}
            type_breakdown: Dict[str, int] = {}
            for m in cached.activity_matches:
                if m.is_eligible and m.objective:
                    obj_breakdown[m.objective] = obj_breakdown.get(m.objective, 0) + 1
                if m.is_eligible and m.activity_type:
                    type_breakdown[m.activity_type] = type_breakdown.get(m.activity_type, 0) + 1

            return {
                "org_id": org_id,
                "total_activities": cached.total_activities,
                "eligible_count": cached.eligible_count,
                "not_eligible_count": cached.not_eligible_count,
                "eligibility_ratio": cached.eligibility_ratio,
                "eligible_turnover_eur": float(cached.eligible_turnover_eur),
                "eligible_capex_eur": float(cached.eligible_capex_eur),
                "eligible_opex_eur": float(cached.eligible_opex_eur),
                "objective_breakdown": obj_breakdown,
                "type_breakdown": type_breakdown,
                "screened_at": cached.screened_at.isoformat(),
                "provenance_hash": cached.provenance_hash,
            }

        # No screening result yet -- return defaults
        return {
            "org_id": org_id,
            "total_activities": len(activities),
            "eligible_count": 0,
            "not_eligible_count": 0,
            "eligibility_ratio": 0.0,
            "eligible_turnover_eur": 0.0,
            "eligible_capex_eur": 0.0,
            "eligible_opex_eur": 0.0,
            "objective_breakdown": {},
            "type_breakdown": {},
            "screened_at": None,
            "provenance_hash": "",
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_section_for_nace(self, nace_code: str) -> str:
        """Extract the NACE section letter from a NACE code."""
        if not nace_code:
            return "UNKNOWN"
        return nace_code[0].upper()

    def _normalise_nace(self, nace_code: str) -> str:
        """Normalise a NACE code to uppercase with standard formatting."""
        return nace_code.strip().upper()
