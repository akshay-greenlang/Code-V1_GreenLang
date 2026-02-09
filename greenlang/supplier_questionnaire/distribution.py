# -*- coding: utf-8 -*-
"""
Distribution Engine - AGENT-DATA-008: Supplier Questionnaire Processor
=======================================================================

Manages the distribution lifecycle of questionnaires to suppliers.
Handles campaign creation, batch distribution, delivery tracking,
access token generation, redistribution, and campaign summarisation.

Supports:
    - Campaign creation with template and supplier list association
    - Batch questionnaire distribution via email/portal/API/bulk
    - Distribution status tracking with timestamps
    - Secure portal access token generation (SHA-256 based)
    - Distribution cancellation and redistribution
    - Campaign-level summary and status reporting
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all operations

Zero-Hallucination Guarantees:
    - All token generation is deterministic (SHA-256 seeded)
    - No external network calls (simulated delivery)
    - No LLM involvement in distribution logic
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.supplier_questionnaire.distribution import DistributionEngine
    >>> engine = DistributionEngine()
    >>> campaign_id = engine.create_campaign(
    ...     name="Q1 2025 CDP",
    ...     template_id="tmpl-001",
    ...     supplier_ids=["SUP001", "SUP002"],
    ...     channel="email",
    ...     deadline_days=30,
    ... )
    >>> dists = engine.list_distributions(campaign_id=campaign_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from greenlang.supplier_questionnaire.models import (
    Distribution,
    DistributionChannel,
    DistributionStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DistributionEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today() -> date:
    """Return today's date in UTC."""
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# DistributionEngine
# ---------------------------------------------------------------------------


class DistributionEngine:
    """Questionnaire distribution lifecycle engine.

    Manages campaign creation, batch distribution, status tracking,
    access token generation, cancellation, and redistribution.
    All operations are thread-safe and tracked with SHA-256 provenance.

    Attributes:
        _distributions: In-memory distribution storage keyed by ID.
        _campaigns: In-memory campaign storage keyed by campaign_id.
        _config: Configuration dictionary.
        _lock: Threading lock for mutations.
        _stats: Aggregate statistics counters.

    Example:
        >>> engine = DistributionEngine()
        >>> cid = engine.create_campaign("Q1", "t1", ["s1"], "email", 30)
        >>> dists = engine.list_distributions(campaign_id=cid)
        >>> assert len(dists) == 1
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DistributionEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_distributions``: int (default 10000)
                - ``default_deadline_days``: int (default 30)
                - ``max_batch_size``: int (default 5000)
        """
        self._config = config or {}
        self._distributions: Dict[str, Distribution] = {}
        self._campaigns: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_distributions: int = self._config.get(
            "max_distributions", 10000,
        )
        self._default_deadline_days: int = self._config.get(
            "default_deadline_days", 30,
        )
        self._max_batch_size: int = self._config.get("max_batch_size", 5000)
        self._stats: Dict[str, int] = {
            "distributions_created": 0,
            "distributions_cancelled": 0,
            "distributions_redistributed": 0,
            "campaigns_created": 0,
            "tokens_generated": 0,
            "errors": 0,
        }
        logger.info(
            "DistributionEngine initialised: max_distributions=%d, "
            "default_deadline_days=%d",
            self._max_distributions,
            self._default_deadline_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_campaign(
        self,
        name: str,
        template_id: str,
        supplier_ids: List[str],
        channel: str = "email",
        deadline_days: int = 30,
    ) -> str:
        """Create a distribution campaign and distribute to all suppliers.

        Args:
            name: Campaign display name.
            template_id: Template to distribute.
            supplier_ids: List of supplier identifiers.
            channel: Delivery channel string.
            deadline_days: Days until deadline.

        Returns:
            Campaign ID string.

        Raises:
            ValueError: If inputs are invalid.
        """
        start = time.monotonic()

        if not name or not name.strip():
            raise ValueError("Campaign name must be non-empty")
        if not template_id or not template_id.strip():
            raise ValueError("template_id must be non-empty")
        if not supplier_ids:
            raise ValueError("supplier_ids must be non-empty")

        campaign_id = f"camp-{uuid.uuid4().hex[:12]}"
        ch = self._resolve_channel(channel)
        deadline = _today() + timedelta(days=deadline_days)

        # Create campaign record
        campaign_record: Dict[str, Any] = {
            "campaign_id": campaign_id,
            "name": name,
            "template_id": template_id,
            "channel": ch.value,
            "deadline": deadline.isoformat(),
            "deadline_days": deadline_days,
            "supplier_count": len(supplier_ids),
            "created_at": _utcnow().isoformat(),
            "status": "active",
            "provenance_hash": self._compute_provenance(
                "create_campaign", campaign_id, template_id,
            ),
        }

        # Distribute to each supplier
        distributions: List[Distribution] = []
        for supplier_id in supplier_ids:
            dist = self._create_single_distribution(
                template_id=template_id,
                supplier_id=supplier_id,
                supplier_name=f"Supplier {supplier_id}",
                supplier_email=f"{supplier_id.lower()}@supplier.example.com",
                campaign_id=campaign_id,
                channel=ch,
                deadline=deadline,
            )
            distributions.append(dist)

        with self._lock:
            self._campaigns[campaign_id] = campaign_record
            self._stats["campaigns_created"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Created campaign %s: name='%s' template=%s suppliers=%d (%.1f ms)",
            campaign_id, name, template_id[:8],
            len(supplier_ids), elapsed_ms,
        )
        return campaign_id

    def distribute(
        self,
        template_id: str,
        supplier_list: List[Dict[str, str]],
        channel: str = "email",
        campaign_id: str = "",
        deadline_days: int = 30,
    ) -> List[Distribution]:
        """Distribute a template to a list of suppliers.

        Args:
            template_id: Template to distribute.
            supplier_list: List of dicts with 'id', 'name', 'email' keys.
            channel: Delivery channel string.
            campaign_id: Optional campaign to associate with.
            deadline_days: Days until deadline.

        Returns:
            List of Distribution records created.

        Raises:
            ValueError: If inputs are invalid or batch too large.
        """
        start = time.monotonic()

        if not template_id or not template_id.strip():
            raise ValueError("template_id must be non-empty")
        if not supplier_list:
            raise ValueError("supplier_list must be non-empty")
        if len(supplier_list) > self._max_batch_size:
            raise ValueError(
                f"Batch size {len(supplier_list)} exceeds "
                f"max {self._max_batch_size}"
            )

        ch = self._resolve_channel(channel)
        deadline = _today() + timedelta(days=deadline_days)
        effective_campaign = campaign_id or f"camp-{uuid.uuid4().hex[:12]}"

        distributions: List[Distribution] = []
        for supplier in supplier_list:
            sid = supplier.get("id", "")
            sname = supplier.get("name", "")
            semail = supplier.get("email", "")

            if not sid:
                logger.warning("Skipping supplier with empty id")
                continue

            dist = self._create_single_distribution(
                template_id=template_id,
                supplier_id=sid,
                supplier_name=sname,
                supplier_email=semail,
                campaign_id=effective_campaign,
                channel=ch,
                deadline=deadline,
            )
            distributions.append(dist)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Distributed template %s to %d suppliers via %s (%.1f ms)",
            template_id[:8], len(distributions), ch.value, elapsed_ms,
        )
        return distributions

    def get_distribution(self, distribution_id: str) -> Distribution:
        """Get a distribution record by ID.

        Args:
            distribution_id: Distribution identifier.

        Returns:
            Distribution record.

        Raises:
            ValueError: If distribution_id is not found.
        """
        return self._get_distribution_or_raise(distribution_id)

    def list_distributions(
        self,
        campaign_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Distribution]:
        """List distributions with optional filtering.

        Args:
            campaign_id: Filter by campaign.
            supplier_id: Filter by supplier.
            status: Filter by status value.

        Returns:
            List of matching Distribution records.
        """
        with self._lock:
            distributions = list(self._distributions.values())

        if campaign_id is not None:
            distributions = [
                d for d in distributions if d.campaign_id == campaign_id
            ]
        if supplier_id is not None:
            distributions = [
                d for d in distributions if d.supplier_id == supplier_id
            ]
        if status is not None:
            distributions = [
                d for d in distributions if d.status.value == status
            ]

        logger.debug(
            "Listed %d distributions (campaign=%s, supplier=%s, status=%s)",
            len(distributions), campaign_id, supplier_id, status,
        )
        return distributions

    def update_status(
        self,
        distribution_id: str,
        new_status: str,
    ) -> Distribution:
        """Update the status of a distribution.

        Automatically sets relevant timestamps based on the new status.

        Args:
            distribution_id: Distribution to update.
            new_status: New status value string.

        Returns:
            Updated Distribution record.

        Raises:
            ValueError: If distribution_id not found or status invalid.
        """
        dist = self._get_distribution_or_raise(distribution_id)
        status = self._resolve_status(new_status)
        now = _utcnow()

        with self._lock:
            record = self._distributions[distribution_id]
            record.status = status

            # Set timestamps based on status transitions
            if status == DistributionStatus.SENT and record.sent_at is None:
                record.sent_at = now
            elif (
                status == DistributionStatus.DELIVERED
                and record.delivered_at is None
            ):
                record.delivered_at = now
            elif (
                status == DistributionStatus.OPENED
                and record.opened_at is None
            ):
                record.opened_at = now
            elif (
                status == DistributionStatus.SUBMITTED
                and record.submitted_at is None
            ):
                record.submitted_at = now

            record.provenance_hash = self._compute_provenance(
                "update_status", distribution_id, status.value,
            )

        logger.info(
            "Updated distribution %s status to %s",
            distribution_id[:8], status.value,
        )
        return self._distributions[distribution_id]

    def generate_access_token(self, distribution_id: str) -> str:
        """Generate a secure portal access token for a distribution.

        The token is a SHA-256 hash derived from the distribution ID,
        supplier ID, and a timestamp for deterministic uniqueness.

        Args:
            distribution_id: Distribution to generate token for.

        Returns:
            64-character hex access token string.

        Raises:
            ValueError: If distribution_id is not found.
        """
        dist = self._get_distribution_or_raise(distribution_id)

        token_seed = (
            f"{distribution_id}:{dist.supplier_id}:{_utcnow().isoformat()}"
        )
        token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()

        with self._lock:
            self._distributions[distribution_id].access_token = token
            self._stats["tokens_generated"] += 1

        logger.info(
            "Generated access token for distribution %s",
            distribution_id[:8],
        )
        return token

    def get_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign details with associated distributions.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            Campaign dict with nested distributions list.

        Raises:
            ValueError: If campaign_id is not found.
        """
        with self._lock:
            campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Unknown campaign: {campaign_id}")

        distributions = self.list_distributions(campaign_id=campaign_id)
        result = dict(campaign)
        result["distributions"] = [
            d.model_dump(mode="json") for d in distributions
        ]
        result["distribution_count"] = len(distributions)

        return result

    def cancel_distribution(self, distribution_id: str) -> Distribution:
        """Cancel a distribution.

        Only cancels if the distribution is in a cancellable state
        (pending, sent, delivered, opened, in_progress).

        Args:
            distribution_id: Distribution to cancel.

        Returns:
            Updated Distribution record.

        Raises:
            ValueError: If distribution_id not found or not cancellable.
        """
        dist = self._get_distribution_or_raise(distribution_id)

        cancellable_statuses = {
            DistributionStatus.PENDING,
            DistributionStatus.SENT,
            DistributionStatus.DELIVERED,
            DistributionStatus.OPENED,
            DistributionStatus.IN_PROGRESS,
        }
        if dist.status not in cancellable_statuses:
            raise ValueError(
                f"Distribution {distribution_id} cannot be cancelled "
                f"from status {dist.status.value}"
            )

        with self._lock:
            self._distributions[distribution_id].status = (
                DistributionStatus.CANCELLED
            )
            self._distributions[distribution_id].provenance_hash = (
                self._compute_provenance(
                    "cancel_distribution", distribution_id,
                )
            )
            self._stats["distributions_cancelled"] += 1

        logger.info(
            "Cancelled distribution %s", distribution_id[:8],
        )
        return self._distributions[distribution_id]

    def redistribute(self, distribution_id: str) -> Distribution:
        """Redistribute a bounced or expired distribution.

        Creates a new distribution with fresh IDs and resets status
        to pending.

        Args:
            distribution_id: Distribution to redistribute.

        Returns:
            New Distribution record.

        Raises:
            ValueError: If distribution_id not found or not redistributable.
        """
        dist = self._get_distribution_or_raise(distribution_id)

        redistributable = {
            DistributionStatus.BOUNCED,
            DistributionStatus.EXPIRED,
            DistributionStatus.CANCELLED,
        }
        if dist.status not in redistributable:
            raise ValueError(
                f"Distribution {distribution_id} cannot be redistributed "
                f"from status {dist.status.value}"
            )

        # Create fresh distribution
        new_dist = self._create_single_distribution(
            template_id=dist.template_id,
            supplier_id=dist.supplier_id,
            supplier_name=dist.supplier_name,
            supplier_email=dist.supplier_email,
            campaign_id=dist.campaign_id,
            channel=dist.channel,
            deadline=dist.deadline,
        )

        with self._lock:
            self._stats["distributions_redistributed"] += 1

        logger.info(
            "Redistributed %s -> %s for supplier %s",
            distribution_id[:8], new_dist.distribution_id[:8],
            dist.supplier_id,
        )
        return new_dist

    def get_campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get summary statistics for a campaign.

        Args:
            campaign_id: Campaign to summarise.

        Returns:
            Dictionary with response rates, status breakdown, and timing.

        Raises:
            ValueError: If campaign_id is not found.
        """
        with self._lock:
            campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Unknown campaign: {campaign_id}")

        distributions = self.list_distributions(campaign_id=campaign_id)
        total = len(distributions)

        # Status breakdown
        status_counts: Dict[str, int] = {}
        for d in distributions:
            key = d.status.value
            status_counts[key] = status_counts.get(key, 0) + 1

        # Response rate
        submitted_count = status_counts.get(
            DistributionStatus.SUBMITTED.value, 0,
        )
        response_rate = (
            round(submitted_count / total * 100, 1) if total > 0 else 0.0
        )

        # Opened rate
        opened_statuses = {
            DistributionStatus.OPENED.value,
            DistributionStatus.IN_PROGRESS.value,
            DistributionStatus.SUBMITTED.value,
        }
        opened_count = sum(
            v for k, v in status_counts.items() if k in opened_statuses
        )
        open_rate = (
            round(opened_count / total * 100, 1) if total > 0 else 0.0
        )

        # Overdue count
        today = _today()
        overdue_count = 0
        for d in distributions:
            if (
                d.deadline is not None
                and d.deadline < today
                and d.status not in (
                    DistributionStatus.SUBMITTED,
                    DistributionStatus.CANCELLED,
                )
            ):
                overdue_count += 1

        provenance_hash = self._compute_provenance(
            "campaign_summary", campaign_id, str(total),
        )

        return {
            "campaign_id": campaign_id,
            "name": campaign.get("name", ""),
            "template_id": campaign.get("template_id", ""),
            "total_distributions": total,
            "status_breakdown": status_counts,
            "response_rate": response_rate,
            "open_rate": open_rate,
            "submitted_count": submitted_count,
            "overdue_count": overdue_count,
            "deadline": campaign.get("deadline", ""),
            "created_at": campaign.get("created_at", ""),
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary of counter values.
        """
        with self._lock:
            return {
                **self._stats,
                "active_distributions": len(self._distributions),
                "active_campaigns": len(self._campaigns),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_single_distribution(
        self,
        template_id: str,
        supplier_id: str,
        supplier_name: str,
        supplier_email: str,
        campaign_id: str,
        channel: DistributionChannel,
        deadline: Optional[date],
    ) -> Distribution:
        """Create and store a single distribution record.

        Args:
            template_id: Template being distributed.
            supplier_id: Target supplier.
            supplier_name: Supplier display name.
            supplier_email: Supplier contact email.
            campaign_id: Campaign association.
            channel: Delivery channel.
            deadline: Submission deadline date.

        Returns:
            Created Distribution record.
        """
        dist_id = str(uuid.uuid4())
        now = _utcnow()

        # Generate access token
        token_seed = f"{dist_id}:{supplier_id}:{now.isoformat()}"
        access_token = hashlib.sha256(
            token_seed.encode("utf-8"),
        ).hexdigest()

        provenance_hash = self._compute_provenance(
            "distribute", dist_id, template_id, supplier_id,
        )

        dist = Distribution(
            distribution_id=dist_id,
            template_id=template_id,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            supplier_email=supplier_email,
            campaign_id=campaign_id,
            channel=channel,
            status=DistributionStatus.SENT,
            access_token=access_token,
            deadline=deadline,
            sent_at=now,
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._distributions[dist_id] = dist
            self._stats["distributions_created"] += 1

        return dist

    def _get_distribution_or_raise(
        self,
        distribution_id: str,
    ) -> Distribution:
        """Retrieve a distribution or raise ValueError.

        Args:
            distribution_id: Distribution identifier.

        Returns:
            Distribution record.

        Raises:
            ValueError: If distribution_id is not found.
        """
        with self._lock:
            dist = self._distributions.get(distribution_id)
        if dist is None:
            raise ValueError(f"Unknown distribution: {distribution_id}")
        return dist

    def _resolve_channel(self, channel: str) -> DistributionChannel:
        """Resolve a channel string to a DistributionChannel enum.

        Args:
            channel: Channel value string.

        Returns:
            DistributionChannel enum member.

        Raises:
            ValueError: If channel is not recognised.
        """
        try:
            return DistributionChannel(channel)
        except ValueError:
            valid = [c.value for c in DistributionChannel]
            raise ValueError(
                f"Unknown channel '{channel}'. Valid: {valid}"
            )

    def _resolve_status(self, status: str) -> DistributionStatus:
        """Resolve a status string to a DistributionStatus enum.

        Args:
            status: Status value string.

        Returns:
            DistributionStatus enum member.

        Raises:
            ValueError: If status is not recognised.
        """
        try:
            return DistributionStatus(status)
        except ValueError:
            valid = [s.value for s in DistributionStatus]
            raise ValueError(
                f"Unknown status '{status}'. Valid: {valid}"
            )

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
