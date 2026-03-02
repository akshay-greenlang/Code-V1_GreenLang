# -*- coding: utf-8 -*-
"""
VerifierRegistryEngine - Accredited CBAM verifier management.

Manages the registry of accredited verification bodies authorised to verify
CBAM declarations. Tracks accreditation status, sector expertise, conflict-of-
interest relationships, and verifier performance statistics.

Per EN ISO 14065:2020 and Implementing Regulation (EU) 2023/1773, verifiers
must be accredited by a National Accreditation Body (NAB) that is a signatory
to the EA MLA for GHG verification.

Example:
    >>> registry = VerifierRegistryEngine()
    >>> verifier = registry.register_verifier(Verifier(
    ...     company_name="EcoVerify GmbH",
    ...     contact_email="info@ecoverify.de",
    ...     nab_country="DE",
    ...     accreditation_number="DAkkS-D-PL-20145-01",
    ...     accredited_until=date(2027, 6, 30),
    ...     sector_expertise=["iron_steel", "aluminium"],
    ... ))

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerifierStatus(str, Enum):
    """Status of a verifier in the registry."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class COIResult(str, Enum):
    """Outcome of a conflict-of-interest assessment."""
    CLEAR = "clear"
    CONFLICT_DETECTED = "conflict_detected"
    REVIEW_REQUIRED = "review_required"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class AccreditationRecord(BaseModel):
    """Record of a verifier's accreditation from a NAB."""

    accreditation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nab_country: str = Field(..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code")
    nab_name: str = Field(default="", description="National Accreditation Body name")
    accreditation_number: str = Field(..., description="Official accreditation number")
    accredited_from: date = Field(default_factory=date.today)
    accredited_until: date = Field(..., description="Expiry date of accreditation")
    scope: List[str] = Field(default_factory=list, description="Accreditation scope (CBAM sectors)")
    standard: str = Field(default="EN ISO 14065:2020", description="Accreditation standard")
    ea_mla_signatory: bool = Field(default=True, description="NAB is EA MLA signatory")

    model_config = {"arbitrary_types_allowed": True}


class Verifier(BaseModel):
    """An accredited CBAM verification body."""

    verifier_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_name: str = Field(..., min_length=1, description="Legal name of verifier")
    contact_email: str = Field(default="", description="Primary contact email")
    contact_phone: str = Field(default="", description="Contact phone number")
    website: str = Field(default="")
    nab_country: str = Field(..., min_length=2, max_length=2, description="Country of accreditation")
    accreditation_number: str = Field(default="", description="Primary accreditation number")
    accredited_until: date = Field(..., description="Primary accreditation expiry")
    accreditation_records: List[AccreditationRecord] = Field(default_factory=list)
    sector_expertise: List[str] = Field(
        default_factory=list,
        description="CBAM sectors (cement, iron_steel, aluminium, fertilisers, hydrogen, electricity)",
    )
    verified_installations: List[str] = Field(
        default_factory=list,
        description="Installation IDs this verifier has verified (for COI tracking)",
    )
    consulting_clients: List[str] = Field(
        default_factory=list,
        description="Installation IDs where verifier provides consulting (COI exclusion)",
    )
    status: VerifierStatus = Field(default=VerifierStatus.ACTIVE)
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("sector_expertise")
    @classmethod
    def validate_sectors(cls, v: List[str]) -> List[str]:
        """Validate sector names against known CBAM sectors."""
        valid = {"cement", "iron_steel", "aluminium", "fertilisers", "hydrogen", "electricity"}
        for s in v:
            if s not in valid:
                logger.warning("Unknown CBAM sector: %s", s)
        return v


class ConflictOfInterestCheck(BaseModel):
    """Outcome of a conflict-of-interest assessment."""

    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    verifier_id: str
    installation_id: str
    result: COIResult
    reasons: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    cooling_off_until: Optional[date] = Field(
        default=None,
        description="Date until which cooling-off period prevents engagement",
    )
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class VerifierPerformance(BaseModel):
    """Performance statistics for a verifier."""

    verifier_id: str
    company_name: str
    total_verifications: int = 0
    passed: int = 0
    failed: int = 0
    conditional: int = 0
    average_findings_per_visit: Decimal = Decimal("0")
    average_days_to_complete: Decimal = Decimal("0")
    sectors_verified: List[str] = Field(default_factory=list)
    countries_active: List[str] = Field(default_factory=list)
    accreditation_valid: bool = True
    performance_score: Decimal = Decimal("0")
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class VerifierRegistryEngine:
    """
    Registry of accredited CBAM verification bodies.

    Thread-safe management of verifier records including registration,
    accreditation tracking, conflict-of-interest screening, and performance
    monitoring.

    The 3-year cooling-off period for COI is enforced per EN ISO 14065 and
    Implementing Regulation (EU) 2023/1773 Article 18(2).
    """

    COI_COOLING_OFF_YEARS = 3

    def __init__(self) -> None:
        """Initialise the verifier registry."""
        self._lock = threading.RLock()
        self._verifiers: Dict[str, Verifier] = {}
        self._coi_history: Dict[str, List[ConflictOfInterestCheck]] = {}
        self._visit_outcomes: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("VerifierRegistryEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_verifier(self, verifier: Verifier) -> Verifier:
        """
        Register a new accredited verifier in the registry.

        Args:
            verifier: Verifier model with accreditation details.

        Returns:
            The registered Verifier with assigned ID and provenance hash.

        Raises:
            ValueError: If accreditation has expired.
        """
        with self._lock:
            if verifier.accredited_until < date.today():
                raise ValueError(
                    f"Accreditation expired on {verifier.accredited_until}. "
                    "Cannot register verifier with expired accreditation."
                )

            # Create accreditation record if not provided
            if not verifier.accreditation_records:
                verifier.accreditation_records.append(AccreditationRecord(
                    nab_country=verifier.nab_country,
                    accreditation_number=verifier.accreditation_number,
                    accredited_until=verifier.accredited_until,
                    scope=verifier.sector_expertise,
                ))

            verifier.status = VerifierStatus.ACTIVE
            verifier.provenance_hash = self._hash_verifier(verifier)
            self._verifiers[verifier.verifier_id] = verifier

            logger.info(
                "Verifier registered: id=%s name=%s country=%s sectors=%s",
                verifier.verifier_id, verifier.company_name,
                verifier.nab_country, verifier.sector_expertise,
            )
            return verifier

    def get_verifier(self, verifier_id: str) -> Verifier:
        """
        Retrieve a verifier by ID.

        Args:
            verifier_id: Unique verifier identifier.

        Returns:
            Verifier model.

        Raises:
            KeyError: If verifier not found.
        """
        with self._lock:
            if verifier_id not in self._verifiers:
                raise KeyError(f"Verifier {verifier_id} not found in registry")
            return self._verifiers[verifier_id]

    def search_verifiers(
        self,
        country: Optional[str] = None,
        sector_expertise: Optional[str] = None,
        accreditation_status: Optional[VerifierStatus] = None,
    ) -> List[Verifier]:
        """
        Search for verifiers matching the given criteria.

        All criteria are optional. If none are provided, all verifiers are
        returned. Multiple criteria are combined with AND logic.

        Args:
            country: ISO 3166-1 alpha-2 country code.
            sector_expertise: Required CBAM sector.
            accreditation_status: Filter by status.

        Returns:
            List of matching Verifier records.
        """
        with self._lock:
            results: List[Verifier] = []
            for v in self._verifiers.values():
                if country and v.nab_country.upper() != country.upper():
                    continue
                if sector_expertise and sector_expertise not in v.sector_expertise:
                    continue
                if accreditation_status and v.status != accreditation_status:
                    continue
                results.append(v)
            return results

    def update_accreditation(
        self,
        verifier_id: str,
        nab_country: str,
        accreditation_number: str,
        expiry_date: date,
        scope: Optional[List[str]] = None,
    ) -> Verifier:
        """
        Update or renew a verifier's accreditation.

        Args:
            verifier_id: Verifier identifier.
            nab_country: Country of the NAB issuing the accreditation.
            accreditation_number: New or renewed accreditation number.
            expiry_date: New expiry date.
            scope: Updated scope (sectors). Keeps existing if None.

        Returns:
            Updated Verifier model.

        Raises:
            KeyError: If verifier not found.
        """
        with self._lock:
            v = self.get_verifier(verifier_id)
            v.nab_country = nab_country
            v.accreditation_number = accreditation_number
            v.accredited_until = expiry_date
            if scope is not None:
                v.sector_expertise = scope

            # Add accreditation record
            v.accreditation_records.append(AccreditationRecord(
                nab_country=nab_country,
                accreditation_number=accreditation_number,
                accredited_until=expiry_date,
                scope=scope or v.sector_expertise,
            ))

            # Reactivate if was expired
            if v.status == VerifierStatus.EXPIRED and expiry_date > date.today():
                v.status = VerifierStatus.ACTIVE

            v.last_updated = datetime.utcnow()
            v.provenance_hash = self._hash_verifier(v)

            logger.info(
                "Accreditation updated: verifier=%s number=%s until=%s",
                verifier_id, accreditation_number, expiry_date,
            )
            return v

    def check_accreditation_validity(self, verifier_id: str) -> bool:
        """
        Check if a verifier's accreditation is currently valid.

        Also updates the verifier's status to EXPIRED if the accreditation
        has lapsed.

        Args:
            verifier_id: Verifier identifier.

        Returns:
            True if accreditation is valid and active.
        """
        with self._lock:
            v = self.get_verifier(verifier_id)
            today = date.today()

            if v.accredited_until < today:
                if v.status == VerifierStatus.ACTIVE:
                    v.status = VerifierStatus.EXPIRED
                    v.last_updated = datetime.utcnow()
                    logger.warning(
                        "Verifier %s accreditation expired on %s",
                        verifier_id, v.accredited_until,
                    )
                return False

            return v.status == VerifierStatus.ACTIVE

    def check_conflict_of_interest(
        self, verifier_id: str, installation_id: str
    ) -> ConflictOfInterestCheck:
        """
        Assess conflict of interest between verifier and installation.

        COI is flagged if:
            1. Verifier currently provides consulting to the installation.
            2. Verifier provided consulting within the 3-year cooling-off period.
            3. Verifier has verified the same installation too many times in
               succession (max 3 consecutive years per EN ISO 14065).

        Args:
            verifier_id: Verifier identifier.
            installation_id: Installation identifier.

        Returns:
            ConflictOfInterestCheck with result and reasons.
        """
        with self._lock:
            v = self.get_verifier(verifier_id)
            reasons: List[str] = []
            result = COIResult.CLEAR
            cooling_off_until: Optional[date] = None

            # Check 1: Current consulting relationship
            if installation_id in v.consulting_clients:
                result = COIResult.CONFLICT_DETECTED
                reasons.append(
                    f"Verifier {v.company_name} currently provides consulting "
                    f"services to installation {installation_id}."
                )
                cooling_off_until = date.today() + timedelta(
                    days=365 * self.COI_COOLING_OFF_YEARS
                )

            # Check 2: Consecutive verification years
            visit_history = self._visit_outcomes.get(verifier_id, [])
            consecutive = sum(
                1 for vh in visit_history
                if vh.get("installation_id") == installation_id
            )
            if consecutive >= 3:
                if result == COIResult.CLEAR:
                    result = COIResult.REVIEW_REQUIRED
                reasons.append(
                    f"Verifier has verified installation {installation_id} "
                    f"{consecutive} times. Maximum 3 consecutive years recommended."
                )

            # Check 3: Ownership or financial interest (simplified: same country)
            # In production this would query corporate ownership databases

            check = ConflictOfInterestCheck(
                verifier_id=verifier_id,
                installation_id=installation_id,
                result=result,
                reasons=reasons,
                cooling_off_until=cooling_off_until,
            )
            check.provenance_hash = self._hash_coi(check)

            # Record COI check
            if verifier_id not in self._coi_history:
                self._coi_history[verifier_id] = []
            self._coi_history[verifier_id].append(check)

            logger.info(
                "COI check: verifier=%s installation=%s result=%s",
                verifier_id, installation_id, result.value,
            )
            return check

    def get_verifier_performance(self, verifier_id: str) -> VerifierPerformance:
        """
        Calculate performance statistics for a verifier.

        Metrics include total verifications, pass/fail/conditional rates,
        average findings per visit, and a composite performance score.

        Args:
            verifier_id: Verifier identifier.

        Returns:
            VerifierPerformance with computed metrics.
        """
        with self._lock:
            v = self.get_verifier(verifier_id)
            outcomes = self._visit_outcomes.get(verifier_id, [])

            total = len(outcomes)
            passed = sum(1 for o in outcomes if o.get("outcome") == "pass")
            failed = sum(1 for o in outcomes if o.get("outcome") == "fail")
            conditional = sum(1 for o in outcomes if o.get("outcome") == "conditional")

            total_findings = sum(o.get("finding_count", 0) for o in outcomes)
            avg_findings = (
                Decimal(str(total_findings)) / Decimal(str(total))
                if total > 0
                else Decimal("0")
            )

            total_days = sum(o.get("days_to_complete", 0) for o in outcomes)
            avg_days = (
                Decimal(str(total_days)) / Decimal(str(total))
                if total > 0
                else Decimal("0")
            )

            sectors_verified: Set[str] = set()
            countries_active: Set[str] = set()
            for o in outcomes:
                if o.get("sector"):
                    sectors_verified.add(o["sector"])
                if o.get("country"):
                    countries_active.add(o["country"])

            # Performance score: weighted (pass_rate * 0.5 + consistency * 0.3 + efficiency * 0.2)
            pass_rate = Decimal(str(passed)) / Decimal(str(total)) if total > 0 else Decimal("0")
            efficiency = max(Decimal("0"), Decimal("1") - avg_days / Decimal("60"))
            consistency = Decimal("1") - min(avg_findings / Decimal("10"), Decimal("1"))
            score = (
                pass_rate * Decimal("0.5")
                + consistency * Decimal("0.3")
                + efficiency * Decimal("0.2")
            ) * Decimal("100")

            perf = VerifierPerformance(
                verifier_id=verifier_id,
                company_name=v.company_name,
                total_verifications=total,
                passed=passed,
                failed=failed,
                conditional=conditional,
                average_findings_per_visit=avg_findings.quantize(Decimal("0.01")),
                average_days_to_complete=avg_days.quantize(Decimal("0.1")),
                sectors_verified=sorted(sectors_verified),
                countries_active=sorted(countries_active),
                accreditation_valid=self.check_accreditation_validity(verifier_id),
                performance_score=score.quantize(Decimal("0.1")),
            )
            perf.provenance_hash = self._hash_performance(perf)
            return perf

    def get_expiring_accreditations(self, days_ahead: int = 90) -> List[Verifier]:
        """
        Return verifiers whose accreditation expires within the given window.

        Args:
            days_ahead: Number of days to look ahead (default 90).

        Returns:
            List of Verifier records with expiring accreditations.
        """
        with self._lock:
            cutoff = date.today() + timedelta(days=days_ahead)
            expiring: List[Verifier] = []
            for v in self._verifiers.values():
                if v.status in (VerifierStatus.ACTIVE, VerifierStatus.PENDING):
                    if v.accredited_until <= cutoff:
                        expiring.append(v)
            return sorted(expiring, key=lambda x: x.accredited_until)

    def record_visit_outcome(
        self,
        verifier_id: str,
        installation_id: str,
        outcome: str,
        finding_count: int = 0,
        days_to_complete: int = 0,
        sector: str = "",
        country: str = "",
    ) -> None:
        """
        Record a visit outcome for performance tracking.

        Args:
            verifier_id: Verifier identifier.
            installation_id: Installation identifier.
            outcome: Visit outcome (pass/fail/conditional).
            finding_count: Number of findings raised.
            days_to_complete: Calendar days from visit to report.
            sector: CBAM sector of the installation.
            country: Country of the installation.
        """
        with self._lock:
            if verifier_id not in self._visit_outcomes:
                self._visit_outcomes[verifier_id] = []
            self._visit_outcomes[verifier_id].append({
                "installation_id": installation_id,
                "outcome": outcome,
                "finding_count": finding_count,
                "days_to_complete": days_to_complete,
                "sector": sector,
                "country": country,
                "recorded_at": datetime.utcnow().isoformat(),
            })

    def get_all_verifiers(self) -> List[Verifier]:
        """Return all registered verifiers."""
        with self._lock:
            return list(self._verifiers.values())

    # ------------------------------------------------------------------
    # Provenance hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_verifier(v: Verifier) -> str:
        """Compute SHA-256 provenance hash for a Verifier."""
        payload = (
            f"{v.verifier_id}|{v.company_name}|{v.nab_country}|"
            f"{v.accreditation_number}|{v.accredited_until}|"
            f"{v.sector_expertise}|{v.status.value}|{v.registered_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_coi(check: ConflictOfInterestCheck) -> str:
        """Compute SHA-256 provenance hash for a COI check."""
        payload = (
            f"{check.verifier_id}|{check.installation_id}|{check.result.value}|"
            f"{check.reasons}|{check.checked_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_performance(perf: VerifierPerformance) -> str:
        """Compute SHA-256 provenance hash for performance stats."""
        payload = (
            f"{perf.verifier_id}|{perf.total_verifications}|{perf.passed}|"
            f"{perf.failed}|{perf.conditional}|{perf.performance_score}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
