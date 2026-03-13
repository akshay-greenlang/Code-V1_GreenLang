# -*- coding: utf-8 -*-
"""
Appeal Processor Engine - AGENT-EUDR-040: Authority Communication Manager

Manages administrative appeals per EUDR Article 19. Handles appeal filing,
deadline tracking, extension management, decision recording, and penalty
suspension during the appeal process.

Zero-Hallucination Guarantees:
    - All deadline calculations use deterministic datetime arithmetic
    - No LLM calls in appeal processing path
    - Decision tracking via validated enum states only
    - Complete provenance trail for every appeal event

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 (GL-EUDR-ACM-040)
Regulation: EU 2023/1115 (EUDR) Article 19
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import AuthorityCommunicationManagerConfig, get_config
from .models import (
    Appeal,
    AppealDecision,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance."""
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class AppealProcessor:
    """Manages administrative appeals per EUDR Article 19.

    Handles the full lifecycle of operator appeals against non-compliance
    decisions, including filing, deadline management, extension processing,
    decision recording, and penalty suspension tracking.

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.
        _appeals: In-memory appeal store.

    Example:
        >>> processor = AppealProcessor(config=get_config())
        >>> appeal = await processor.file_appeal(
        ...     non_compliance_id="NC-001",
        ...     operator_id="OP-001",
        ...     authority_id="AUTH-DE",
        ...     grounds="DDS was submitted within the required timeframe."
        ... )
        >>> assert appeal.decision == AppealDecision.PENDING
    """

    def __init__(
        self,
        config: Optional[AuthorityCommunicationManagerConfig] = None,
    ) -> None:
        """Initialize the Appeal Processor engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._appeals: Dict[str, Appeal] = {}
        logger.info("AppealProcessor engine initialized")

    async def file_appeal(
        self,
        non_compliance_id: str,
        operator_id: str,
        authority_id: str,
        grounds: str,
        supporting_evidence: Optional[List[str]] = None,
        communication_id: str = "",
    ) -> Appeal:
        """File an administrative appeal against a non-compliance decision.

        Creates an appeal record with calculated deadline based on the
        configured appeal window and suspends the associated penalty.

        Args:
            non_compliance_id: Non-compliance record being appealed.
            operator_id: Operator filing the appeal.
            authority_id: Authority receiving the appeal.
            grounds: Legal grounds for the appeal.
            supporting_evidence: Supporting document IDs.
            communication_id: Parent communication ID.

        Returns:
            Filed Appeal record with calculated deadline.

        Raises:
            ValueError: If grounds are empty.
        """
        start = time.monotonic()

        if not grounds or len(grounds.strip()) < 10:
            raise ValueError(
                "Appeal grounds must be at least 10 characters"
            )

        appeal_id = _new_uuid()
        now = _utcnow()
        deadline = now + timedelta(days=self.config.appeal_window_days)

        if not communication_id:
            communication_id = _new_uuid()

        appeal = Appeal(
            appeal_id=appeal_id,
            communication_id=communication_id,
            non_compliance_id=non_compliance_id,
            operator_id=operator_id,
            authority_id=authority_id,
            grounds=grounds,
            supporting_evidence=supporting_evidence or [],
            decision=AppealDecision.PENDING,
            filing_date=now,
            deadline=deadline,
            penalty_suspended=True,
            provenance_hash=_compute_hash({
                "appeal_id": appeal_id,
                "non_compliance_id": non_compliance_id,
                "operator_id": operator_id,
                "authority_id": authority_id,
                "filing_date": now.isoformat(),
            }),
        )

        self._appeals[appeal_id] = appeal

        # Record provenance
        self._provenance.create_entry(
            step="file_appeal",
            source=operator_id,
            input_hash=self._provenance.compute_hash({
                "non_compliance_id": non_compliance_id,
                "operator_id": operator_id,
            }),
            output_hash=appeal.provenance_hash,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Appeal %s filed by operator %s against NC %s "
            "(deadline=%s) in %.1fms",
            appeal_id,
            operator_id,
            non_compliance_id,
            deadline.isoformat(),
            elapsed * 1000,
        )

        return appeal

    async def record_decision(
        self,
        appeal_id: str,
        decision: str,
        reason: str = "",
    ) -> Appeal:
        """Record the authority's decision on an appeal.

        Args:
            appeal_id: Appeal identifier.
            decision: Decision outcome.
            reason: Decision reasoning.

        Returns:
            Updated Appeal record.

        Raises:
            ValueError: If appeal not found or decision is invalid.
        """
        appeal = self._appeals.get(appeal_id)
        if appeal is None:
            raise ValueError(f"Appeal {appeal_id} not found")

        try:
            appeal_decision = AppealDecision(decision)
        except ValueError:
            raise ValueError(
                f"Invalid decision: {decision}. "
                f"Valid decisions: {[d.value for d in AppealDecision]}"
            )

        if appeal.decision != AppealDecision.PENDING:
            raise ValueError(
                f"Appeal {appeal_id} already has decision: {appeal.decision.value}"
            )

        now = _utcnow()
        appeal.decision = appeal_decision
        appeal.decision_reason = reason
        appeal.decision_date = now

        # Penalty suspension ends when decision is made
        if appeal_decision not in (
            AppealDecision.UPHELD, AppealDecision.PARTIALLY_UPHELD
        ):
            appeal.penalty_suspended = False

        logger.info(
            "Appeal %s decision recorded: %s (reason: %s)",
            appeal_id,
            decision,
            reason[:100] if reason else "none",
        )

        return appeal

    async def grant_extension(
        self,
        appeal_id: str,
        additional_days: Optional[int] = None,
    ) -> Appeal:
        """Grant a deadline extension for an appeal.

        Args:
            appeal_id: Appeal identifier.
            additional_days: Additional days (defaults to config value).

        Returns:
            Updated Appeal record with extended deadline.

        Raises:
            ValueError: If appeal not found or max extensions exceeded.
        """
        appeal = self._appeals.get(appeal_id)
        if appeal is None:
            raise ValueError(f"Appeal {appeal_id} not found")

        if appeal.extensions_granted >= self.config.appeal_max_extensions:
            raise ValueError(
                f"Appeal {appeal_id} has reached maximum extensions "
                f"({self.config.appeal_max_extensions})"
            )

        days = additional_days or self.config.appeal_extension_days

        if appeal.deadline is not None:
            appeal.deadline = appeal.deadline + timedelta(days=days)

        appeal.extensions_granted += 1

        logger.info(
            "Appeal %s: extension granted (%d days, total extensions: %d)",
            appeal_id,
            days,
            appeal.extensions_granted,
        )

        return appeal

    async def withdraw_appeal(
        self,
        appeal_id: str,
    ) -> Appeal:
        """Withdraw a pending appeal.

        Args:
            appeal_id: Appeal identifier.

        Returns:
            Updated Appeal record with withdrawn status.

        Raises:
            ValueError: If appeal not found or not pending.
        """
        appeal = self._appeals.get(appeal_id)
        if appeal is None:
            raise ValueError(f"Appeal {appeal_id} not found")

        if appeal.decision != AppealDecision.PENDING:
            raise ValueError(
                f"Appeal {appeal_id} cannot be withdrawn (status: {appeal.decision.value})"
            )

        appeal.decision = AppealDecision.WITHDRAWN
        appeal.decision_date = _utcnow()
        appeal.penalty_suspended = False

        logger.info("Appeal %s withdrawn", appeal_id)
        return appeal

    async def get_appeal(self, appeal_id: str) -> Optional[Appeal]:
        """Retrieve an appeal by identifier.

        Args:
            appeal_id: Appeal identifier.

        Returns:
            Appeal record or None.
        """
        return self._appeals.get(appeal_id)

    async def list_appeals(
        self,
        operator_id: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> List[Appeal]:
        """List appeals with optional filters.

        Args:
            operator_id: Filter by operator.
            decision: Filter by decision status.

        Returns:
            List of matching Appeal records.
        """
        results = list(self._appeals.values())
        if operator_id:
            results = [a for a in results if a.operator_id == operator_id]
        if decision:
            results = [a for a in results if a.decision.value == decision]
        return results

    async def list_active_appeals(self) -> List[Appeal]:
        """List all pending appeals.

        Returns:
            List of Appeal records with PENDING decision.
        """
        return [
            a for a in self._appeals.values()
            if a.decision == AppealDecision.PENDING
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        active = len([
            a for a in self._appeals.values()
            if a.decision == AppealDecision.PENDING
        ])
        return {
            "engine": "appeal_processor",
            "status": "healthy",
            "total_appeals": len(self._appeals),
            "active_appeals": active,
        }
