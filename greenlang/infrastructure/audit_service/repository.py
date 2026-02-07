# -*- coding: utf-8 -*-
"""
Audit Event Repository - SEC-005

Database repository for audit events, using async PostgreSQL with
TimescaleDB hypertables and continuous aggregates.

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from greenlang.infrastructure.audit_service.models import (
    AggregationBucket,
    AuditEvent,
    AuditEventSummary,
    EventCategory,
    EventOutcome,
    EventStatistics,
    HotspotEntry,
    SearchAggregation,
    SearchQuery,
    SeverityLevel,
    TimeRange,
)

logger = logging.getLogger(__name__)


class AuditEventRepository:
    """Repository for audit event persistence and querying."""

    def __init__(self, pool: Any):
        """Initialize repository with database connection pool.

        Args:
            pool: Async database connection pool (psycopg_pool.AsyncConnectionPool)
        """
        self._pool = pool

    async def get_event(self, event_id: UUID) -> Optional[AuditEvent]:
        """Retrieve a single audit event by ID.

        Args:
            event_id: Event UUID.

        Returns:
            AuditEvent if found, None otherwise.
        """
        query = """
            SELECT
                id, event_id, trace_id, span_id, performed_at,
                category::text, severity::text, event_type,
                schema_name, table_name, record_id, operation,
                old_data, new_data, changed_fields, change_summary,
                organization_id, user_id, user_email, user_role,
                service_account, impersonated_by,
                ip_address::text, user_agent, request_id, session_id,
                resource_type, resource_path, action, outcome,
                error_message, error_code, metadata, tags,
                data_classification, retention_days, gdpr_relevant
            FROM audit.audit_log
            WHERE id = $1
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (event_id,))
                row = await cur.fetchone()
                if row:
                    return self._row_to_event(row)
        return None

    async def list_events(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        severity: Optional[SeverityLevel] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        result: Optional[EventOutcome] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Tuple[List[AuditEventSummary], Optional[str]]:
        """List audit events with filters and cursor-based pagination.

        Args:
            since: Start of time range.
            until: End of time range.
            event_types: Filter by event types.
            severity: Filter by severity level.
            tenant_id: Filter by tenant/organization.
            user_id: Filter by user.
            resource_type: Filter by resource type.
            result: Filter by outcome.
            limit: Maximum events to return.
            cursor: Pagination cursor.

        Returns:
            Tuple of (events, next_cursor).
        """
        conditions = []
        params: List[Any] = []
        param_idx = 1

        # Parse cursor (timestamp:id format)
        cursor_time: Optional[datetime] = None
        cursor_id: Optional[UUID] = None
        if cursor:
            parts = cursor.split(":")
            if len(parts) == 2:
                cursor_time = datetime.fromisoformat(parts[0])
                cursor_id = UUID(parts[1])

        if since:
            conditions.append(f"performed_at >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if until:
            conditions.append(f"performed_at < ${param_idx}")
            params.append(until)
            param_idx += 1

        if event_types:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(event_types)))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
            param_idx += len(event_types)

        if severity:
            conditions.append(f"severity = ${param_idx}::audit.severity_level")
            params.append(severity.value)
            param_idx += 1

        if tenant_id:
            conditions.append(f"organization_id = ${param_idx}")
            params.append(UUID(tenant_id))
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if resource_type:
            conditions.append(f"resource_type = ${param_idx}")
            params.append(resource_type)
            param_idx += 1

        if result:
            conditions.append(f"outcome = ${param_idx}")
            params.append(result.value)
            param_idx += 1

        if cursor_time and cursor_id:
            conditions.append(
                f"(performed_at, id) < (${param_idx}, ${param_idx + 1})"
            )
            params.extend([cursor_time, cursor_id])
            param_idx += 2

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        query = f"""
            SELECT
                id, performed_at, category::text, severity::text,
                event_type, operation, user_email, resource_type, outcome
            FROM audit.audit_log
            {where_clause}
            ORDER BY performed_at DESC, id DESC
            LIMIT ${param_idx}
        """
        params.append(limit + 1)  # Fetch one extra to determine if there's more

        events: List[AuditEventSummary] = []
        next_cursor: Optional[str] = None

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

                for i, row in enumerate(rows):
                    if i >= limit:
                        # There are more results
                        last_event = events[-1]
                        next_cursor = (
                            f"{last_event.performed_at.isoformat()}:{last_event.id}"
                        )
                        break

                    events.append(
                        AuditEventSummary(
                            id=row[0],
                            performed_at=row[1],
                            category=EventCategory(row[2]),
                            severity=SeverityLevel(row[3]),
                            event_type=row[4],
                            operation=row[5],
                            user_email=row[6],
                            resource_type=row[7],
                            outcome=EventOutcome(row[8]),
                        )
                    )

        return events, next_cursor

    async def search(
        self,
        query: SearchQuery,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[AuditEvent], int, List[SearchAggregation]]:
        """Advanced search with aggregations.

        Args:
            query: Structured search query.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            Tuple of (events, total_count, aggregations).
        """
        conditions = []
        params: List[Any] = []
        param_idx = 1

        # Parse LogQL-like query string
        if query.query:
            parsed_conditions = self._parse_query_string(query.query, params)
            if parsed_conditions:
                conditions.append(parsed_conditions)
                param_idx = len(params) + 1

        # Time range
        if query.time_range:
            conditions.append(f"performed_at >= ${param_idx}")
            params.append(query.time_range.start)
            param_idx += 1
            conditions.append(f"performed_at < ${param_idx}")
            params.append(query.time_range.end)
            param_idx += 1

        # Category filter
        if query.categories:
            placeholders = ", ".join(
                f"${param_idx + i}::audit.event_category"
                for i in range(len(query.categories))
            )
            conditions.append(f"category IN ({placeholders})")
            params.extend(c.value for c in query.categories)
            param_idx += len(query.categories)

        # Severity filter
        if query.severities:
            placeholders = ", ".join(
                f"${param_idx + i}::audit.severity_level"
                for i in range(len(query.severities))
            )
            conditions.append(f"severity IN ({placeholders})")
            params.extend(s.value for s in query.severities)
            param_idx += len(query.severities)

        # Event type filter
        if query.event_types:
            placeholders = ", ".join(
                f"${param_idx + i}" for i in range(len(query.event_types))
            )
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(query.event_types)
            param_idx += len(query.event_types)

        # User filter
        if query.user_ids:
            placeholders = ", ".join(
                f"${param_idx + i}" for i in range(len(query.user_ids))
            )
            conditions.append(f"user_id IN ({placeholders})")
            params.extend(query.user_ids)
            param_idx += len(query.user_ids)

        # Organization filter
        if query.organization_ids:
            placeholders = ", ".join(
                f"${param_idx + i}" for i in range(len(query.organization_ids))
            )
            conditions.append(f"organization_id IN ({placeholders})")
            params.extend(query.organization_ids)
            param_idx += len(query.organization_ids)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Count query
        count_query = f"SELECT COUNT(*) FROM audit.audit_log {where_clause}"

        # Main query
        select_query = f"""
            SELECT
                id, event_id, trace_id, span_id, performed_at,
                category::text, severity::text, event_type,
                schema_name, table_name, record_id, operation,
                old_data, new_data, changed_fields, change_summary,
                organization_id, user_id, user_email, user_role,
                service_account, impersonated_by,
                ip_address::text, user_agent, request_id, session_id,
                resource_type, resource_path, action, outcome,
                error_message, error_code, metadata, tags,
                data_classification, retention_days, gdpr_relevant
            FROM audit.audit_log
            {where_clause}
            ORDER BY performed_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        events: List[AuditEvent] = []
        total_count = 0
        aggregations: List[SearchAggregation] = []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # Get count
                await cur.execute(count_query, params[:-2] if params else [])
                count_row = await cur.fetchone()
                total_count = count_row[0] if count_row else 0

                # Get events
                await cur.execute(select_query, params)
                rows = await cur.fetchall()
                for row in rows:
                    events.append(self._row_to_event(row))

                # Get aggregations
                if query.aggregations:
                    for agg_field in query.aggregations:
                        agg = await self._get_aggregation(
                            cur, agg_field, where_clause, params[:-2]
                        )
                        if agg:
                            aggregations.append(agg)

        return events, total_count, aggregations

    async def get_statistics(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        organization_id: Optional[UUID] = None,
    ) -> EventStatistics:
        """Get event statistics for a time range.

        Args:
            since: Start of time range.
            until: End of time range.
            organization_id: Filter by organization.

        Returns:
            EventStatistics with counts and breakdowns.
        """
        if not since:
            since = datetime.now(timezone.utc) - timedelta(days=7)
        if not until:
            until = datetime.now(timezone.utc)

        conditions = ["performed_at >= $1", "performed_at < $2"]
        params: List[Any] = [since, until]
        param_idx = 3

        if organization_id:
            conditions.append(f"organization_id = ${param_idx}")
            params.append(organization_id)

        where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                COUNT(*) as total_events,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT resource_type) as unique_resources
            FROM audit.audit_log
            {where_clause}
        """

        category_query = f"""
            SELECT category::text, COUNT(*)
            FROM audit.audit_log
            {where_clause}
            GROUP BY category
        """

        severity_query = f"""
            SELECT severity::text, COUNT(*)
            FROM audit.audit_log
            {where_clause}
            GROUP BY severity
        """

        outcome_query = f"""
            SELECT outcome, COUNT(*)
            FROM audit.audit_log
            {where_clause}
            GROUP BY outcome
        """

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # Main stats
                await cur.execute(query, params)
                row = await cur.fetchone()
                total_events = row[0] if row else 0
                unique_users = row[1] if row else 0
                unique_resources = row[2] if row else 0

                # By category
                await cur.execute(category_query, params)
                events_by_category = {r[0]: r[1] for r in await cur.fetchall()}

                # By severity
                await cur.execute(severity_query, params)
                events_by_severity = {r[0]: r[1] for r in await cur.fetchall()}

                # By outcome
                await cur.execute(outcome_query, params)
                events_by_outcome = {r[0]: r[1] for r in await cur.fetchall()}

        return EventStatistics(
            total_events=total_events,
            events_by_category=events_by_category,
            events_by_severity=events_by_severity,
            events_by_outcome=events_by_outcome,
            unique_users=unique_users,
            unique_resources=unique_resources,
            time_range=TimeRange(start=since, end=until),
        )

    async def get_hotspots(
        self,
        hotspot_type: str,
        since: Optional[datetime] = None,
        limit: int = 10,
        organization_id: Optional[UUID] = None,
    ) -> List[HotspotEntry]:
        """Get top users, resources, or IPs by event count.

        Args:
            hotspot_type: Type of hotspot (users, resources, ips).
            since: Start of time range.
            limit: Maximum entries to return.
            organization_id: Filter by organization.

        Returns:
            List of hotspot entries.
        """
        if not since:
            since = datetime.now(timezone.utc) - timedelta(days=1)

        field_map = {
            "users": ("user_id", "user_email"),
            "resources": ("resource_type", "resource_path"),
            "ips": ("ip_address::text", None),
        }

        if hotspot_type not in field_map:
            return []

        id_field, label_field = field_map[hotspot_type]
        label_select = f", {label_field}" if label_field else ", NULL"

        conditions = ["performed_at >= $1"]
        params: List[Any] = [since]
        param_idx = 2

        if organization_id:
            conditions.append(f"organization_id = ${param_idx}")
            params.append(organization_id)
            param_idx += 1

        where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                {id_field} as identifier{label_select} as label,
                COUNT(*) as event_count,
                MAX(performed_at) as last_seen,
                COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
                COUNT(*) FILTER (WHERE severity = 'warning') as warning_count
            FROM audit.audit_log
            {where_clause}
            AND {id_field} IS NOT NULL
            GROUP BY {id_field}{', ' + label_field if label_field else ''}
            ORDER BY event_count DESC
            LIMIT ${param_idx}
        """
        params.append(limit)

        hotspots: List[HotspotEntry] = []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

                for row in rows:
                    hotspots.append(
                        HotspotEntry(
                            identifier=str(row[0]) if row[0] else "unknown",
                            label=row[1] if row[1] else None,
                            event_count=row[2],
                            last_seen=row[3],
                            severity_breakdown={
                                "error": row[4] or 0,
                                "warning": row[5] or 0,
                            },
                        )
                    )

        return hotspots

    async def get_timeline(
        self,
        entity_type: str,
        entity_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get activity timeline for a user or resource.

        Args:
            entity_type: Type of entity (user, resource).
            entity_id: Entity identifier.
            since: Start of time range.
            until: End of time range.
            limit: Maximum entries.

        Returns:
            List of timeline entries.
        """
        if not since:
            since = datetime.now(timezone.utc) - timedelta(days=7)
        if not until:
            until = datetime.now(timezone.utc)

        conditions = ["performed_at >= $1", "performed_at < $2"]
        params: List[Any] = [since, until]
        param_idx = 3

        if entity_type == "user":
            conditions.append(f"user_id = ${param_idx}")
            params.append(UUID(entity_id))
        elif entity_type == "resource":
            conditions.append(f"resource_type = ${param_idx}")
            params.append(entity_id)
        else:
            return []

        params.append(limit)

        where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                performed_at, event_type, category::text, severity::text,
                change_summary, resource_path, outcome
            FROM audit.audit_log
            {where_clause}
            ORDER BY performed_at DESC
            LIMIT ${param_idx + 1}
        """

        timeline: List[Dict[str, Any]] = []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

                for row in rows:
                    timeline.append({
                        "timestamp": row[0].isoformat(),
                        "event_type": row[1],
                        "category": row[2],
                        "severity": row[3],
                        "description": row[4] or row[1],
                        "resource_path": row[5],
                        "outcome": row[6],
                    })

        return timeline

    def _parse_query_string(
        self, query_str: str, params: List[Any]
    ) -> Optional[str]:
        """Parse LogQL-like query string into SQL conditions.

        Supports: field:value, field:"quoted value", AND, OR

        Args:
            query_str: Query string to parse.
            params: Parameter list to append values to.

        Returns:
            SQL condition string or None.
        """
        # Simple parser for field:value patterns
        # Example: event_type:login AND severity:error OR user_email:admin@example.com
        tokens = re.findall(r'(\w+):(?:"([^"]+)"|(\S+))|(\bAND\b|\bOR\b)', query_str, re.IGNORECASE)

        if not tokens:
            return None

        conditions = []
        connector = "AND"
        param_idx = len(params) + 1

        for match in tokens:
            field, quoted_val, unquoted_val, logical = match

            if logical:
                connector = logical.upper()
                continue

            value = quoted_val or unquoted_val
            if not field or not value:
                continue

            # Map query fields to database columns
            field_map = {
                "event_type": "event_type",
                "category": "category::text",
                "severity": "severity::text",
                "user_email": "user_email",
                "user_id": "user_id::text",
                "resource_type": "resource_type",
                "outcome": "outcome",
                "ip_address": "ip_address::text",
                "action": "action",
                "operation": "operation",
            }

            db_field = field_map.get(field.lower())
            if not db_field:
                continue

            if conditions:
                conditions.append(connector)

            # Support wildcards
            if "*" in value:
                conditions.append(f"{db_field} LIKE ${param_idx}")
                params.append(value.replace("*", "%"))
            else:
                conditions.append(f"{db_field} = ${param_idx}")
                params.append(value)

            param_idx += 1
            connector = "AND"

        return " ".join(conditions) if conditions else None

    async def _get_aggregation(
        self,
        cursor: Any,
        field: str,
        where_clause: str,
        params: List[Any],
    ) -> Optional[SearchAggregation]:
        """Get aggregation for a single field.

        Args:
            cursor: Database cursor.
            field: Field to aggregate.
            where_clause: SQL WHERE clause.
            params: Query parameters.

        Returns:
            SearchAggregation or None.
        """
        field_map = {
            "category": "category::text",
            "severity": "severity::text",
            "event_type": "event_type",
            "outcome": "outcome",
            "user_id": "user_id::text",
            "resource_type": "resource_type",
        }

        db_field = field_map.get(field.lower())
        if not db_field:
            return None

        query = f"""
            SELECT {db_field}, COUNT(*) as cnt
            FROM audit.audit_log
            {where_clause}
            AND {db_field} IS NOT NULL
            GROUP BY {db_field}
            ORDER BY cnt DESC
            LIMIT 20
        """

        await cursor.execute(query, params)
        rows = await cursor.fetchall()

        total = sum(r[1] for r in rows)
        buckets = [
            AggregationBucket(
                key=str(r[0]),
                count=r[1],
                percentage=round(r[1] / total * 100, 2) if total > 0 else 0,
            )
            for r in rows
        ]

        return SearchAggregation(field=field, buckets=buckets, total=total)

    def _row_to_event(self, row: tuple) -> AuditEvent:
        """Convert database row to AuditEvent model.

        Args:
            row: Database row tuple.

        Returns:
            AuditEvent instance.
        """
        return AuditEvent(
            id=row[0],
            event_id=row[1],
            trace_id=row[2],
            span_id=row[3],
            performed_at=row[4],
            category=EventCategory(row[5]),
            severity=SeverityLevel(row[6]),
            event_type=row[7],
            schema_name=row[8],
            table_name=row[9],
            record_id=row[10],
            operation=row[11],
            old_data=row[12],
            new_data=row[13],
            changed_fields=row[14],
            change_summary=row[15],
            organization_id=row[16],
            user_id=row[17],
            user_email=row[18],
            user_role=row[19],
            service_account=row[20],
            impersonated_by=row[21],
            ip_address=row[22],
            user_agent=row[23],
            request_id=row[24],
            session_id=row[25],
            resource_type=row[26],
            resource_path=row[27],
            action=row[28],
            outcome=EventOutcome(row[29]) if row[29] else EventOutcome.SUCCESS,
            error_message=row[30],
            error_code=row[31],
            metadata=row[32] or {},
            tags=row[33],
            data_classification=row[34],
            retention_days=row[35],
            gdpr_relevant=row[36] or False,
        )


# Singleton instance
_repository: Optional[AuditEventRepository] = None


def get_audit_repository() -> AuditEventRepository:
    """Get the global AuditEventRepository instance.

    Returns:
        The singleton repository instance.

    Raises:
        RuntimeError: If repository not initialized.
    """
    global _repository
    if _repository is None:
        raise RuntimeError(
            "AuditEventRepository not initialized. "
            "Call init_audit_repository(pool) first."
        )
    return _repository


async def init_audit_repository(pool: Any) -> AuditEventRepository:
    """Initialize the global AuditEventRepository.

    Args:
        pool: Async database connection pool.

    Returns:
        The initialized repository.
    """
    global _repository
    _repository = AuditEventRepository(pool)
    return _repository
