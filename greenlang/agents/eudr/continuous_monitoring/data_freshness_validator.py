# -*- coding: utf-8 -*-
"""
Data Freshness Validator Engine - AGENT-EUDR-033

Ensures data currency across the supply chain by validating entity
data age, identifying stale entities, scheduling refresh operations,
and generating freshness reports.

Zero-Hallucination:
    - All age calculations use pure datetime arithmetic
    - Freshness percentages are Decimal computations
    - Refresh scheduling uses deterministic priority ordering

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ContinuousMonitoringConfig, get_config
from .models import (
    AGENT_ID,
    DataFreshnessRecord,
    FreshnessStatus,
    StaleEntity,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class DataFreshnessValidator:
    """Data freshness validation and refresh scheduling engine.

    Validates data age across supply chain entities, classifies
    freshness status, schedules refresh operations, and generates
    freshness reports for compliance monitoring.

    Example:
        >>> validator = DataFreshnessValidator()
        >>> record = await validator.validate_data_age(
        ...     operator_id="OP-001",
        ...     entities=[{"entity_id": "S-001", "last_updated": "2026-03-01T00:00:00Z"}],
        ... )
        >>> assert record.freshness_percentage >= 0
    """

    def __init__(
        self, config: Optional[ContinuousMonitoringConfig] = None,
    ) -> None:
        """Initialize DataFreshnessValidator engine."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._records: Dict[str, DataFreshnessRecord] = {}
        logger.info("DataFreshnessValidator engine initialized")

    async def validate_data_age(
        self,
        operator_id: str,
        entities: List[Dict[str, Any]],
    ) -> DataFreshnessRecord:
        """Validate data freshness across entities.

        Args:
            operator_id: Operator identifier.
            entities: List of entity data with last_updated timestamps.

        Returns:
            DataFreshnessRecord with validation results.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        freshness_id = str(uuid.uuid4())

        stale_entities: List[StaleEntity] = []
        fresh_count = 0
        stale_warning = 0
        stale_critical = 0

        for entity_data in entities:
            entity_id = entity_data.get("entity_id", "")
            entity_type = entity_data.get("entity_type", "supplier")
            last_updated_str = entity_data.get("last_updated")

            if not last_updated_str:
                stale_entities.append(StaleEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    freshness_status=FreshnessStatus.UNKNOWN,
                    recommended_action="Establish data collection for this entity",
                ))
                stale_critical += 1
                continue

            try:
                last_updated = self._parse_datetime(last_updated_str)
            except (ValueError, TypeError):
                stale_entities.append(StaleEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    freshness_status=FreshnessStatus.UNKNOWN,
                    recommended_action="Fix timestamp format for this entity",
                ))
                stale_critical += 1
                continue

            age_hours = Decimal(str(
                round((now - last_updated).total_seconds() / 3600.0, 2)
            ))

            freshness = self._classify_freshness(age_hours)

            if freshness == FreshnessStatus.FRESH:
                fresh_count += 1
            elif freshness == FreshnessStatus.STALE_WARNING:
                stale_warning += 1
                stale_entities.append(StaleEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    last_updated=last_updated,
                    age_hours=age_hours,
                    freshness_status=freshness,
                    recommended_action="Schedule data refresh within 24 hours",
                ))
            else:
                stale_critical += 1
                stale_entities.append(StaleEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    last_updated=last_updated,
                    age_hours=age_hours,
                    freshness_status=freshness,
                    recommended_action="Immediate data refresh required",
                ))

        total = len(entities)
        freshness_pct = (
            (Decimal(str(fresh_count)) / Decimal(str(total))) * Decimal("100")
            if total > 0 else Decimal("0")
        ).quantize(Decimal("0.01"))

        target = self.config.data_freshness_target_percent
        meets_target = freshness_pct >= target

        # Identify entities needing refresh
        stale_for_refresh = await self.identify_stale_entities(stale_entities)

        # Schedule refreshes
        schedule = await self.schedule_refreshes(stale_for_refresh)

        record = DataFreshnessRecord(
            freshness_id=freshness_id,
            operator_id=operator_id,
            entities_checked=total,
            fresh_count=fresh_count,
            stale_warning_count=stale_warning,
            stale_critical_count=stale_critical,
            freshness_percentage=freshness_pct,
            meets_target=meets_target,
            stale_entities=stale_for_refresh,
            refresh_scheduled=len(schedule),
            refresh_schedule=schedule,
            checked_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "freshness_id": freshness_id,
            "operator_id": operator_id,
            "entities_checked": total,
            "freshness_percentage": str(freshness_pct),
            "created_at": now.isoformat(),
        })

        self._provenance.record(
            entity_type="data_freshness",
            action="validate",
            entity_id=freshness_id,
            actor=AGENT_ID,
            metadata={
                "operator_id": operator_id,
                "total": total,
                "fresh": fresh_count,
                "stale_warning": stale_warning,
                "stale_critical": stale_critical,
            },
        )

        self._records[freshness_id] = record
        elapsed = time.monotonic() - start_time
        logger.info(
            "Freshness check %s: %d entities, %.1f%% fresh, target=%s%% (%.3fs)",
            freshness_id, total, float(freshness_pct), target, elapsed,
        )
        return record

    async def identify_stale_entities(
        self,
        stale_entities: List[StaleEntity],
    ) -> List[StaleEntity]:
        """Identify and prioritize stale entities for refresh.

        Sorts stale entities by criticality (critical first, then by age).

        Args:
            stale_entities: List of stale entities to prioritize.

        Returns:
            Sorted list of stale entities.
        """
        # Sort: critical first, then by age descending
        priority_order = {
            FreshnessStatus.STALE_CRITICAL: 0,
            FreshnessStatus.UNKNOWN: 1,
            FreshnessStatus.STALE_WARNING: 2,
            FreshnessStatus.FRESH: 3,
        }

        sorted_entities = sorted(
            stale_entities,
            key=lambda e: (
                priority_order.get(e.freshness_status, 99),
                -float(e.age_hours),
            ),
        )
        return sorted_entities

    async def schedule_refreshes(
        self,
        stale_entities: List[StaleEntity],
    ) -> List[Dict[str, Any]]:
        """Schedule data refresh operations for stale entities.

        Batches entities based on configured batch size and assigns
        priority based on staleness level.

        Args:
            stale_entities: Prioritized list of stale entities.

        Returns:
            List of refresh schedule entries.
        """
        schedule: List[Dict[str, Any]] = []
        batch_size = self.config.data_refresh_batch_size

        for i in range(0, len(stale_entities), batch_size):
            batch = stale_entities[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            # Priority: batch 1 is highest priority
            priority = "critical" if batch_num == 1 else "high" if batch_num <= 3 else "medium"

            schedule.append({
                "batch_number": batch_num,
                "entity_count": len(batch),
                "entity_ids": [e.entity_id for e in batch],
                "priority": priority,
                "status": "scheduled",
            })

        return schedule

    async def generate_freshness_reports(
        self,
        operator_id: str,
    ) -> Dict[str, Any]:
        """Generate a freshness summary report for an operator.

        Args:
            operator_id: Operator identifier.

        Returns:
            Freshness report dictionary.
        """
        records = [r for r in self._records.values() if r.operator_id == operator_id]

        if not records:
            return {
                "operator_id": operator_id,
                "total_checks": 0,
                "average_freshness": "0",
                "meets_target_count": 0,
                "latest_check": None,
            }

        # Sort by checked_at descending
        records.sort(key=lambda r: r.checked_at, reverse=True)
        latest = records[0]

        avg_freshness = sum(r.freshness_percentage for r in records) / Decimal(str(len(records)))
        meets_target = sum(1 for r in records if r.meets_target)

        return {
            "operator_id": operator_id,
            "total_checks": len(records),
            "average_freshness": str(avg_freshness.quantize(Decimal("0.01"))),
            "meets_target_count": meets_target,
            "meets_target_rate": str(
                (Decimal(str(meets_target)) / Decimal(str(len(records))) * Decimal("100"))
                .quantize(Decimal("0.01"))
            ),
            "latest_check": {
                "freshness_id": latest.freshness_id,
                "freshness_percentage": str(latest.freshness_percentage),
                "entities_checked": latest.entities_checked,
                "checked_at": latest.checked_at.isoformat(),
            },
        }

    def _classify_freshness(self, age_hours: Decimal) -> FreshnessStatus:
        """Classify data freshness by age in hours."""
        warning = Decimal(str(self.config.data_stale_warning_hours))
        critical = Decimal(str(self.config.data_stale_critical_hours))

        if age_hours >= critical:
            return FreshnessStatus.STALE_CRITICAL
        elif age_hours >= warning:
            return FreshnessStatus.STALE_WARNING
        return FreshnessStatus.FRESH

    @staticmethod
    def _parse_datetime(val: Any) -> datetime:
        """Parse a value to timezone-aware datetime."""
        if isinstance(val, datetime):
            if val.tzinfo is None:
                return val.replace(tzinfo=timezone.utc)
            return val
        date_str = str(val).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    async def get_record(self, freshness_id: str) -> Optional[DataFreshnessRecord]:
        """Retrieve a freshness record by ID."""
        return self._records.get(freshness_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
    ) -> List[DataFreshnessRecord]:
        """List freshness records with optional filters."""
        results = list(self._records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        return results

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "DataFreshnessValidator",
            "status": "healthy",
            "record_count": len(self._records),
        }
