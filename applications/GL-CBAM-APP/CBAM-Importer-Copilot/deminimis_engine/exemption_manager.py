# -*- coding: utf-8 -*-
"""
ExemptionManagerEngine - De minimis exemption lifecycle management.

Handles exemption determination, certificate issuance, mid-year revocation,
retroactive reporting requirements, and SME simplified compliance paths.

Per the Omnibus Simplification Package (Oct 2025), importers below the
50-tonne annual threshold are exempt from full CBAM reporting. This engine
manages the complete lifecycle of that exemption status, including the
transition from exempt to subject-to-CBAM when the threshold is breached.

Example:
    >>> monitor = ThresholdMonitorEngine.get_instance()
    >>> mgr = ExemptionManagerEngine(monitor)
    >>> result = mgr.determine_exemption("IMP-001", 2026)
    >>> if result.status == "exempt":
    ...     cert = mgr.issue_exemption_certificate("IMP-001", 2026)

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import logging
import threading
import uuid
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from deminimis_engine.threshold_monitor import (
    DE_MINIMIS_THRESHOLD_MT,
    ThresholdMonitorEngine,
    ThresholdStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExemptionStatus(str, Enum):
    """Possible exemption states for an importer-year."""

    EXEMPT = "exempt"
    APPROACHING = "approaching"
    SUBJECT_TO_CBAM = "subject_to_cbam"
    REVOKED = "revoked"


class CertificateStatus(str, Enum):
    """Status of an exemption certificate."""

    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ExemptionResult(BaseModel):
    """Outcome of an exemption determination."""

    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    importer_id: str
    year: int
    status: ExemptionStatus
    cumulative_mt: Decimal = Field(default=Decimal("0"))
    threshold_mt: Decimal = Field(default=DE_MINIMIS_THRESHOLD_MT)
    percentage: Decimal = Field(default=Decimal("0"))
    loss_date: Optional[date] = Field(
        default=None,
        description="Date when exemption was lost (breach date)",
    )
    retroactive_reporting_required: bool = Field(default=False)
    retroactive_quarters: List[str] = Field(
        default_factory=list,
        description="Quarters requiring retroactive reports (e.g. ['Q1','Q2'])",
    )
    recommended_actions: List[str] = Field(default_factory=list)
    determined_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class ExemptionCertificate(BaseModel):
    """Digital certificate attesting de minimis exemption."""

    certificate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    importer_id: str
    year: int
    status: CertificateStatus = CertificateStatus.ACTIVE
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: date = Field(default_factory=lambda: date(date.today().year, 1, 1))
    valid_until: date = Field(default_factory=lambda: date(date.today().year, 12, 31))
    cumulative_mt_at_issuance: Decimal = Field(default=Decimal("0"))
    threshold_mt: Decimal = Field(default=DE_MINIMIS_THRESHOLD_MT)
    conditions: List[str] = Field(default_factory=list)
    revoked_at: Optional[datetime] = Field(default=None)
    revocation_reason: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class SMESimplifiedPath(BaseModel):
    """Simplified compliance pathway for small and medium enterprises."""

    importer_id: str
    is_sme: bool = Field(default=False)
    annual_import_volume_mt: Decimal = Field(default=Decimal("0"))
    simplified_reporting_eligible: bool = Field(default=False)
    reduced_frequency: str = Field(default="quarterly", description="Reporting frequency")
    default_values_allowed: bool = Field(default=False)
    estimated_compliance_cost_eur: Decimal = Field(default=Decimal("0"))
    recommended_tools: List[str] = Field(default_factory=list)
    guidance_notes: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class ExemptionManagerEngine:
    """
    Manages the de minimis exemption lifecycle for CBAM importers.

    Collaborates with ThresholdMonitorEngine for real-time cumulative data,
    then applies business rules for exemption determination, certificate
    management, and retroactive reporting obligations.

    Args:
        monitor: ThresholdMonitorEngine instance (singleton recommended).
    """

    def __init__(self, monitor: Optional[ThresholdMonitorEngine] = None) -> None:
        """Initialise with a threshold monitor instance."""
        self._monitor = monitor or ThresholdMonitorEngine.get_instance()
        self._lock = threading.RLock()
        self._certificates: Dict[str, Dict[int, ExemptionCertificate]] = {}
        self._history: Dict[str, List[ExemptionResult]] = {}
        logger.info("ExemptionManagerEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def determine_exemption(self, importer_id: str, year: int) -> ExemptionResult:
        """
        Auto-determine exemption status based on current threshold data.

        Decision logic:
            - cumulative < 80% threshold  -> EXEMPT
            - cumulative 80-99%           -> APPROACHING
            - cumulative >= 100%          -> SUBJECT_TO_CBAM

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year (>= 2026).

        Returns:
            ExemptionResult capturing the determination and any recommended actions.
        """
        with self._lock:
            status = self._monitor.check_threshold(importer_id, year)
            result = self._evaluate_status(status)
            self._record_history(result)
            logger.info(
                "Exemption determined: importer=%s year=%d status=%s (%s MT / %s%%)",
                importer_id, year, result.status.value,
                result.cumulative_mt, result.percentage,
            )
            return result

    def issue_exemption_certificate(
        self, importer_id: str, year: int
    ) -> ExemptionCertificate:
        """
        Issue a digital exemption certificate for the importer-year.

        The certificate is only issued if the importer is currently exempt.
        If the importer is not exempt, a ValueError is raised.

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            ExemptionCertificate with validity period and conditions.

        Raises:
            ValueError: If the importer is not currently exempt.
        """
        with self._lock:
            status = self._monitor.check_threshold(importer_id, year)
            if not status.exempt:
                raise ValueError(
                    f"Cannot issue exemption certificate: importer {importer_id} "
                    f"has {status.cumulative_mt} MT imports ({status.percentage}% of threshold)"
                )

            cert = ExemptionCertificate(
                importer_id=importer_id,
                year=year,
                status=CertificateStatus.ACTIVE,
                valid_from=date(year, 1, 1),
                valid_until=date(year, 12, 31),
                cumulative_mt_at_issuance=status.cumulative_mt,
                threshold_mt=DE_MINIMIS_THRESHOLD_MT,
                conditions=[
                    "Certificate valid only while total eligible CBAM imports remain below 50 MT.",
                    "Electricity (CN 2716) and hydrogen (CN 2804) imports are excluded from threshold.",
                    "Importer must notify authorities within 30 days if threshold is breached.",
                    "Certificate is automatically revoked upon threshold breach.",
                ],
            )
            cert.provenance_hash = self._hash_certificate(cert)

            if importer_id not in self._certificates:
                self._certificates[importer_id] = {}
            self._certificates[importer_id][year] = cert

            logger.info(
                "Exemption certificate issued: id=%s importer=%s year=%d",
                cert.certificate_id, importer_id, year,
            )
            return cert

    def revoke_exemption(
        self, importer_id: str, year: int, reason: str
    ) -> ExemptionResult:
        """
        Revoke an exemption due to mid-year threshold breach or other cause.

        Revocation triggers:
            - Automatic when 50 MT threshold is crossed
            - Manual by competent authority
            - Fraud detection or data correction

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.
            reason: Human-readable revocation reason.

        Returns:
            ExemptionResult with REVOKED status and retroactive obligations.
        """
        with self._lock:
            status = self._monitor.check_threshold(importer_id, year)
            today = date.today()

            # Revoke any active certificate
            cert = self._certificates.get(importer_id, {}).get(year)
            if cert and cert.status == CertificateStatus.ACTIVE:
                cert.status = CertificateStatus.REVOKED
                cert.revoked_at = datetime.utcnow()
                cert.revocation_reason = reason
                logger.warning(
                    "Certificate revoked: id=%s importer=%s reason=%s",
                    cert.certificate_id, importer_id, reason,
                )

            # Determine retroactive quarters
            retro_quarters = self._determine_retroactive_quarters(year, today)

            result = ExemptionResult(
                importer_id=importer_id,
                year=year,
                status=ExemptionStatus.REVOKED,
                cumulative_mt=status.cumulative_mt,
                percentage=status.percentage,
                loss_date=today,
                retroactive_reporting_required=len(retro_quarters) > 0,
                retroactive_quarters=retro_quarters,
                recommended_actions=[
                    "Submit retroactive CBAM declarations for all quarters since January 1.",
                    f"Prepare emission reports for quarters: {', '.join(retro_quarters)}.",
                    "Contact your CBAM authorized declarant within 30 days.",
                    "Begin purchasing CBAM certificates for applicable emissions.",
                    "Review all prior imports for correct emission factor assignment.",
                ],
            )
            result.provenance_hash = self._hash_result(result)
            self._record_history(result)

            logger.warning(
                "Exemption revoked: importer=%s year=%d reason=%s retro_quarters=%s",
                importer_id, year, reason, retro_quarters,
            )
            return result

    def get_exemption_history(self, importer_id: str) -> List[ExemptionResult]:
        """
        Return the full exemption determination history for an importer.

        Args:
            importer_id: EORI or internal identifier.

        Returns:
            List of ExemptionResult ordered by determination time (oldest first).
        """
        with self._lock:
            return list(self._history.get(importer_id, []))

    def handle_retroactive_reporting(
        self, importer_id: str, year: int
    ) -> Dict[str, Any]:
        """
        Generate retroactive reporting requirements when exemption is lost mid-year.

        This method produces a structured set of obligations including:
            - Quarters that need reports
            - Import records to include per quarter
            - Deadlines for submission
            - Estimated CBAM certificate requirements

        Args:
            importer_id: EORI or internal identifier.
            year: Calendar year.

        Returns:
            Dictionary with retroactive reporting plan.
        """
        with self._lock:
            status = self._monitor.check_threshold(importer_id, year)
            records = self._monitor.get_all_records(importer_id, year)
            today = date.today()
            retro_quarters = self._determine_retroactive_quarters(year, today)

            # Group records by quarter
            quarterly: Dict[str, List[Dict[str, Any]]] = {q: [] for q in retro_quarters}
            for r in records:
                q = self._date_to_quarter(r.recorded_at.date(), year)
                if q in quarterly:
                    quarterly[q].append({
                        "record_id": r.record_id,
                        "cn_code": r.cn_code,
                        "quantity_mt": float(r.quantity_mt),
                        "sector": r.sector,
                        "eligible": r.eligible_for_threshold,
                        "recorded_at": r.recorded_at.isoformat(),
                    })

            # Deadlines: 1 month after end of each quarter
            quarter_deadlines = {
                "Q1": date(year, 4, 30),
                "Q2": date(year, 7, 31),
                "Q3": date(year, 10, 31),
                "Q4": date(year + 1, 1, 31),
            }

            # Estimated certificates needed (simplified: total emissions * EU ETS price gap)
            total_eligible_mt = sum(
                r.quantity_mt for r in records if r.eligible_for_threshold
            )

            plan = {
                "importer_id": importer_id,
                "year": year,
                "retroactive_quarters": retro_quarters,
                "quarterly_imports": {
                    q: {
                        "records": recs,
                        "total_mt": sum(rec["quantity_mt"] for rec in recs),
                        "deadline": quarter_deadlines.get(q, date(year + 1, 1, 31)).isoformat(),
                        "overdue": today > quarter_deadlines.get(q, date(year + 1, 1, 31)),
                    }
                    for q, recs in quarterly.items()
                },
                "total_eligible_mt": float(total_eligible_mt),
                "status": "subject_to_cbam",
                "guidance": [
                    "File retroactive CBAM quarterly reports for each listed quarter.",
                    "Use actual emission data where available; default values with markup otherwise.",
                    "Overdue reports may incur penalties per Article 26 of the CBAM Regulation.",
                    "Engage an accredited CBAM verifier for emission data verification.",
                ],
                "generated_at": datetime.utcnow().isoformat(),
            }

            provenance_str = f"{importer_id}|{year}|{retro_quarters}|{total_eligible_mt}"
            plan["provenance_hash"] = hashlib.sha256(
                provenance_str.encode("utf-8")
            ).hexdigest()

            return plan

    def get_sme_simplified_path(self, importer_id: str) -> SMESimplifiedPath:
        """
        Determine if an importer qualifies for SME simplified compliance.

        SME criteria (Omnibus Simplification):
            - Annual import volume < 50 MT across all CBAM sectors
            - Fewer than 250 employees (if data available)
            - Turnover < EUR 50M (if data available)

        For importers close to the threshold, the engine recommends
        additional monitoring tools.

        Args:
            importer_id: EORI or internal identifier.

        Returns:
            SMESimplifiedPath with eligibility and recommended tools.
        """
        with self._lock:
            current_year = date.today().year
            status = self._monitor.check_threshold(importer_id, current_year)

            # SME determination based on import volume
            volume = status.cumulative_mt
            is_sme_volume = volume < Decimal("25")  # Below half the threshold
            is_below_threshold = volume < DE_MINIMIS_THRESHOLD_MT

            simplified = is_below_threshold
            reduced_freq = "annual" if volume < Decimal("10") else "quarterly"
            default_allowed = volume < Decimal("25")

            # Estimated compliance cost (heuristic)
            if volume < Decimal("10"):
                cost = Decimal("500")
            elif volume < Decimal("25"):
                cost = Decimal("2000")
            elif volume < Decimal("50"):
                cost = Decimal("5000")
            else:
                cost = Decimal("15000")

            tools = ["GL-CBAM Importer Copilot (free tier)"]
            guidance = []

            if is_sme_volume:
                tools.append("Simplified CBAM Declaration Template")
                guidance.append(
                    "Your import volume qualifies for simplified reporting. "
                    "Use EU Commission default values for emission calculations."
                )
            if is_below_threshold and not is_sme_volume:
                tools.append("GL-CBAM Threshold Monitor")
                tools.append("CBAM Verifier Matching Service")
                guidance.append(
                    "Your imports are below 50 MT but approaching the threshold. "
                    "Consider engaging a verifier proactively."
                )
            if not is_below_threshold:
                tools.extend([
                    "GL-CBAM Full Compliance Suite",
                    "Accredited Verifier Engagement",
                    "CBAM Certificate Purchase Module",
                ])
                guidance.append(
                    "Your imports exceed the de minimis threshold. "
                    "Full CBAM compliance is required."
                )

            path = SMESimplifiedPath(
                importer_id=importer_id,
                is_sme=is_sme_volume,
                annual_import_volume_mt=volume,
                simplified_reporting_eligible=simplified,
                reduced_frequency=reduced_freq,
                default_values_allowed=default_allowed,
                estimated_compliance_cost_eur=cost,
                recommended_tools=tools,
                guidance_notes=guidance,
            )
            path.provenance_hash = self._hash_sme_path(path)
            return path

    def get_certificate(
        self, importer_id: str, year: int
    ) -> Optional[ExemptionCertificate]:
        """Retrieve the exemption certificate for an importer-year, if any."""
        with self._lock:
            return self._certificates.get(importer_id, {}).get(year)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_status(self, status: ThresholdStatus) -> ExemptionResult:
        """Map a ThresholdStatus to an ExemptionResult with business rules."""
        pct = status.percentage
        actions: List[str] = []

        if pct < Decimal("80"):
            ex_status = ExemptionStatus.EXEMPT
            actions.append("Continue monitoring imports. No CBAM reporting required.")
            if pct > Decimal("50"):
                actions.append(
                    "Consider proactive engagement with a CBAM verifier as "
                    "you are past 50% of the threshold."
                )
        elif pct < Decimal("100"):
            ex_status = ExemptionStatus.APPROACHING
            actions.extend([
                f"WARNING: {pct}% of de minimis threshold consumed.",
                "Review remaining planned imports for the year.",
                "Prepare for potential transition to full CBAM compliance.",
                "Identify an accredited CBAM verifier.",
                "Begin collecting installation-level emission data from suppliers.",
            ])
        else:
            ex_status = ExemptionStatus.SUBJECT_TO_CBAM
            retro_quarters = self._determine_retroactive_quarters(
                status.year, date.today()
            )
            actions.extend([
                "De minimis threshold EXCEEDED. Full CBAM compliance required.",
                f"Submit retroactive reports for: {', '.join(retro_quarters)}.",
                "Register as CBAM authorized declarant if not already.",
                "Purchase CBAM certificates to cover embedded emissions.",
                "Engage accredited verifier for annual verification.",
            ])

        result = ExemptionResult(
            importer_id=status.importer_id,
            year=status.year,
            status=ex_status,
            cumulative_mt=status.cumulative_mt,
            percentage=status.percentage,
            loss_date=date.today() if ex_status == ExemptionStatus.SUBJECT_TO_CBAM else None,
            retroactive_reporting_required=ex_status == ExemptionStatus.SUBJECT_TO_CBAM,
            retroactive_quarters=(
                self._determine_retroactive_quarters(status.year, date.today())
                if ex_status == ExemptionStatus.SUBJECT_TO_CBAM
                else []
            ),
            recommended_actions=actions,
        )
        result.provenance_hash = self._hash_result(result)
        return result

    def _record_history(self, result: ExemptionResult) -> None:
        """Append an ExemptionResult to the importer's history (lock held)."""
        if result.importer_id not in self._history:
            self._history[result.importer_id] = []
        self._history[result.importer_id].append(result)

    @staticmethod
    def _determine_retroactive_quarters(year: int, reference: date) -> List[str]:
        """
        Determine which quarters require retroactive reporting.

        All quarters from Q1 up to and including the quarter of the reference
        date must be reported retroactively.

        Args:
            year: CBAM reporting year.
            reference: Reference date (typically today).

        Returns:
            List of quarter labels, e.g. ["Q1", "Q2"].
        """
        if reference.year < year:
            return []
        if reference.year > year:
            return ["Q1", "Q2", "Q3", "Q4"]
        month = reference.month
        quarters: List[str] = []
        if month >= 1:
            quarters.append("Q1")
        if month >= 4:
            quarters.append("Q2")
        if month >= 7:
            quarters.append("Q3")
        if month >= 10:
            quarters.append("Q4")
        return quarters

    @staticmethod
    def _date_to_quarter(d: date, year: int) -> str:
        """Convert a date to its quarter label within the given year."""
        if d.year != year:
            return "Q0"  # Out of range
        month = d.month
        if month <= 3:
            return "Q1"
        elif month <= 6:
            return "Q2"
        elif month <= 9:
            return "Q3"
        else:
            return "Q4"

    # ------------------------------------------------------------------
    # Provenance hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_result(result: ExemptionResult) -> str:
        """Compute SHA-256 provenance hash for an ExemptionResult."""
        payload = (
            f"{result.importer_id}|{result.year}|{result.status.value}|"
            f"{result.cumulative_mt}|{result.percentage}|"
            f"{result.loss_date}|{result.retroactive_reporting_required}|"
            f"{result.determined_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_certificate(cert: ExemptionCertificate) -> str:
        """Compute SHA-256 provenance hash for an ExemptionCertificate."""
        payload = (
            f"{cert.certificate_id}|{cert.importer_id}|{cert.year}|"
            f"{cert.status.value}|{cert.valid_from.isoformat()}|"
            f"{cert.valid_until.isoformat()}|{cert.cumulative_mt_at_issuance}|"
            f"{cert.issued_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_sme_path(path: SMESimplifiedPath) -> str:
        """Compute SHA-256 provenance hash for an SMESimplifiedPath."""
        payload = (
            f"{path.importer_id}|{path.is_sme}|{path.annual_import_volume_mt}|"
            f"{path.simplified_reporting_eligible}|{path.estimated_compliance_cost_eur}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
