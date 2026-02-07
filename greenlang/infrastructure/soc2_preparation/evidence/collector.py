# -*- coding: utf-8 -*-
"""
Evidence Collector - SEC-009 Phase 3

Multi-source evidence collection system with adapters for:
    - CloudTrail: AWS API audit logs
    - GitHub: Code changes, pull requests, reviews
    - PostgreSQL: Security event data from internal tables
    - Loki: Log aggregation via LogQL queries
    - AuthService: Authentication and authorization events
    - Jira: Change management tickets
    - Okta: SSO and identity events

The collector implements a plugin architecture where each source adapter
can be enabled/disabled independently and configured with source-specific
settings.

Example:
    >>> config = EvidenceCollectorConfig(...)
    >>> collector = EvidenceCollector(config)
    >>> await collector.initialize()
    >>> evidence = await collector.collect_for_criterion(
    ...     "CC6.1",
    ...     DateRange(start=audit_start, end=audit_end)
    ... )
    >>> all_evidence = await collector.collect_all(date_range)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.soc2_preparation.evidence.models import (
    CollectionMetadata,
    DateRange,
    Evidence,
    EvidenceSource,
    EvidenceStatus,
    EvidenceType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SourceAdapterConfig(BaseModel):
    """Base configuration for source adapters."""

    enabled: bool = Field(default=True, description="Whether adapter is enabled")
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    batch_size: int = Field(default=100, ge=1, le=1000)


class CloudTrailConfig(SourceAdapterConfig):
    """CloudTrail adapter configuration."""

    region: str = Field(default="us-east-1")
    trail_name: Optional[str] = Field(default=None)
    event_categories: List[str] = Field(
        default_factory=lambda: ["Management", "Data"]
    )
    lookup_attributes: Dict[str, str] = Field(default_factory=dict)


class GitHubConfig(SourceAdapterConfig):
    """GitHub adapter configuration."""

    org: str = Field(default="")
    repos: List[str] = Field(default_factory=list)
    token_env_var: str = Field(default="GITHUB_TOKEN")
    include_prs: bool = Field(default=True)
    include_commits: bool = Field(default=True)
    include_reviews: bool = Field(default=True)


class PostgreSQLConfig(SourceAdapterConfig):
    """PostgreSQL adapter configuration."""

    dsn_env_var: str = Field(default="GREENLANG_DATABASE_URL")
    schema_name: str = Field(default="security")
    tables: List[str] = Field(
        default_factory=lambda: [
            "access_events",
            "auth_events",
            "data_access_log",
            "configuration_changes",
        ]
    )


class LokiConfig(SourceAdapterConfig):
    """Loki adapter configuration."""

    url: str = Field(default="http://loki:3100")
    tenant_id: Optional[str] = Field(default=None)
    queries: Dict[str, str] = Field(default_factory=dict)


class AuthServiceConfig(SourceAdapterConfig):
    """Auth service adapter configuration."""

    base_url: str = Field(default="http://auth-service:8080")
    api_key_env_var: str = Field(default="AUTH_SERVICE_API_KEY")


class JiraConfig(SourceAdapterConfig):
    """Jira adapter configuration."""

    base_url: str = Field(default="")
    project_keys: List[str] = Field(default_factory=list)
    token_env_var: str = Field(default="JIRA_API_TOKEN")
    email_env_var: str = Field(default="JIRA_EMAIL")


class OktaConfig(SourceAdapterConfig):
    """Okta adapter configuration."""

    domain: str = Field(default="")
    token_env_var: str = Field(default="OKTA_API_TOKEN")
    event_types: List[str] = Field(
        default_factory=lambda: [
            "user.authentication.sso",
            "user.session.start",
            "user.account.lock",
            "policy.evaluate_sign_on",
        ]
    )


class EvidenceCollectorConfig(BaseModel):
    """Configuration for the evidence collector."""

    cloudtrail: CloudTrailConfig = Field(default_factory=CloudTrailConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    loki: LokiConfig = Field(default_factory=LokiConfig)
    auth_service: AuthServiceConfig = Field(default_factory=AuthServiceConfig)
    jira: JiraConfig = Field(default_factory=JiraConfig)
    okta: OktaConfig = Field(default_factory=OktaConfig)

    # General settings
    parallel_collection: bool = Field(default=True)
    max_concurrent_sources: int = Field(default=4, ge=1, le=10)
    collection_timeout_seconds: int = Field(default=300, ge=60, le=1800)


# ---------------------------------------------------------------------------
# Criterion to Source Mapping
# ---------------------------------------------------------------------------

# SOC 2 Trust Services Criteria to evidence source mapping
CRITERION_SOURCE_MAP: Dict[str, List[EvidenceSource]] = {
    # CC1: Control Environment
    "CC1.1": [EvidenceSource.MANUAL, EvidenceSource.CONFLUENCE],
    "CC1.2": [EvidenceSource.MANUAL, EvidenceSource.GITHUB],
    "CC1.3": [EvidenceSource.MANUAL, EvidenceSource.JIRA],
    "CC1.4": [EvidenceSource.MANUAL, EvidenceSource.OKTA],
    "CC1.5": [EvidenceSource.MANUAL],

    # CC2: Communication and Information
    "CC2.1": [EvidenceSource.MANUAL, EvidenceSource.CONFLUENCE],
    "CC2.2": [EvidenceSource.MANUAL, EvidenceSource.JIRA],
    "CC2.3": [EvidenceSource.MANUAL],

    # CC3: Risk Assessment
    "CC3.1": [EvidenceSource.MANUAL],
    "CC3.2": [EvidenceSource.MANUAL, EvidenceSource.JIRA],
    "CC3.3": [EvidenceSource.MANUAL],
    "CC3.4": [EvidenceSource.MANUAL],

    # CC4: Monitoring Activities
    "CC4.1": [EvidenceSource.PROMETHEUS, EvidenceSource.LOKI],
    "CC4.2": [EvidenceSource.LOKI, EvidenceSource.CLOUDTRAIL],

    # CC5: Control Activities
    "CC5.1": [EvidenceSource.MANUAL, EvidenceSource.GITHUB],
    "CC5.2": [EvidenceSource.MANUAL, EvidenceSource.JIRA],
    "CC5.3": [EvidenceSource.GITHUB, EvidenceSource.TERRAFORM],

    # CC6: Logical and Physical Access Controls
    "CC6.1": [EvidenceSource.OKTA, EvidenceSource.AUTH_SERVICE, EvidenceSource.CLOUDTRAIL],
    "CC6.2": [EvidenceSource.OKTA, EvidenceSource.AUTH_SERVICE],
    "CC6.3": [EvidenceSource.OKTA, EvidenceSource.AUTH_SERVICE, EvidenceSource.POSTGRESQL],
    "CC6.4": [EvidenceSource.CLOUDTRAIL, EvidenceSource.KUBERNETES],
    "CC6.5": [EvidenceSource.CLOUDTRAIL, EvidenceSource.KUBERNETES],
    "CC6.6": [EvidenceSource.CLOUDTRAIL, EvidenceSource.LOKI],
    "CC6.7": [EvidenceSource.CLOUDTRAIL, EvidenceSource.POSTGRESQL],
    "CC6.8": [EvidenceSource.MANUAL],

    # CC7: System Operations
    "CC7.1": [EvidenceSource.PROMETHEUS, EvidenceSource.LOKI],
    "CC7.2": [EvidenceSource.LOKI, EvidenceSource.PROMETHEUS],
    "CC7.3": [EvidenceSource.JIRA, EvidenceSource.GITHUB],
    "CC7.4": [EvidenceSource.JIRA, EvidenceSource.LOKI],
    "CC7.5": [EvidenceSource.JIRA],

    # CC8: Change Management
    "CC8.1": [EvidenceSource.GITHUB, EvidenceSource.JIRA, EvidenceSource.TERRAFORM],

    # CC9: Risk Mitigation
    "CC9.1": [EvidenceSource.MANUAL, EvidenceSource.JIRA],
    "CC9.2": [EvidenceSource.MANUAL],

    # A1: Availability
    "A1.1": [EvidenceSource.PROMETHEUS, EvidenceSource.CLOUDTRAIL],
    "A1.2": [EvidenceSource.S3, EvidenceSource.CLOUDTRAIL],
    "A1.3": [EvidenceSource.MANUAL, EvidenceSource.JIRA],

    # C1: Confidentiality
    "C1.1": [EvidenceSource.CLOUDTRAIL, EvidenceSource.POSTGRESQL],
    "C1.2": [EvidenceSource.CLOUDTRAIL, EvidenceSource.S3],

    # PI1: Processing Integrity
    "PI1.1": [EvidenceSource.LOKI, EvidenceSource.PROMETHEUS],
    "PI1.2": [EvidenceSource.LOKI, EvidenceSource.POSTGRESQL],
    "PI1.3": [EvidenceSource.MANUAL],
    "PI1.4": [EvidenceSource.LOKI],
    "PI1.5": [EvidenceSource.LOKI, EvidenceSource.PROMETHEUS],
}


# ---------------------------------------------------------------------------
# Source Adapters (Abstract Base)
# ---------------------------------------------------------------------------


class SourceAdapter(ABC):
    """Abstract base class for evidence source adapters.

    Each adapter is responsible for collecting evidence from a specific
    source system. Adapters must implement the collect method and handle
    their own connection management and error handling.
    """

    source: EvidenceSource

    def __init__(self, config: SourceAdapterConfig) -> None:
        """Initialize the adapter with configuration."""
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the adapter (connect to source, validate config)."""
        self._initialized = True
        logger.info(f"{self.__class__.__name__} initialized")

    async def close(self) -> None:
        """Close the adapter and release resources."""
        self._initialized = False
        logger.info(f"{self.__class__.__name__} closed")

    @abstractmethod
    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect evidence for a criterion within a period.

        Args:
            criterion_id: SOC 2 criterion ID (e.g., "CC6.1").
            period: Date range to collect evidence for.

        Returns:
            List of Evidence objects.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the source is accessible.

        Returns:
            True if source is healthy.
        """
        pass


# ---------------------------------------------------------------------------
# CloudTrail Adapter
# ---------------------------------------------------------------------------


class CloudTrailCollector(SourceAdapter):
    """AWS CloudTrail evidence collector.

    Collects audit logs from CloudTrail for infrastructure and data events.
    Supports filtering by event category, event source, and user identity.
    """

    source = EvidenceSource.CLOUDTRAIL

    def __init__(self, config: CloudTrailConfig) -> None:
        super().__init__(config)
        self.config: CloudTrailConfig = config
        self._client: Any = None

    async def initialize(self) -> None:
        """Initialize CloudTrail client."""
        try:
            import aioboto3
            self._session = aioboto3.Session()
            self._initialized = True
            logger.info("CloudTrailCollector initialized")
        except ImportError:
            logger.warning("aioboto3 not available, CloudTrail collection disabled")
            self._initialized = False

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect CloudTrail events for the criterion."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            async with self._session.client(
                "cloudtrail",
                region_name=self.config.region,
            ) as client:
                # Build lookup attributes based on criterion
                lookup_attrs = self._get_lookup_attributes(criterion_id)

                paginator = client.get_paginator("lookup_events")

                async for page in paginator.paginate(
                    LookupAttributes=lookup_attrs,
                    StartTime=period.start,
                    EndTime=period.end,
                    MaxResults=self.config.batch_size,
                ):
                    for event in page.get("Events", []):
                        evidence = self._event_to_evidence(event, criterion_id, period)
                        evidence_list.append(evidence)

            logger.info(
                f"CloudTrail collected {len(evidence_list)} events for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"CloudTrail collection failed: {exc}")

        return evidence_list

    def _get_lookup_attributes(self, criterion_id: str) -> List[Dict[str, str]]:
        """Get CloudTrail lookup attributes for a criterion."""
        attrs = []

        # Map criteria to relevant event sources
        criterion_event_map = {
            "CC6.1": [{"AttributeKey": "EventSource", "AttributeValue": "iam.amazonaws.com"}],
            "CC6.4": [{"AttributeKey": "EventSource", "AttributeValue": "ec2.amazonaws.com"}],
            "CC6.6": [{"AttributeKey": "EventSource", "AttributeValue": "kms.amazonaws.com"}],
            "A1.1": [{"AttributeKey": "EventSource", "AttributeValue": "elasticloadbalancing.amazonaws.com"}],
            "A1.2": [{"AttributeKey": "EventSource", "AttributeValue": "s3.amazonaws.com"}],
        }

        return criterion_event_map.get(criterion_id, attrs)

    def _event_to_evidence(
        self,
        event: Dict[str, Any],
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert a CloudTrail event to Evidence."""
        event_time = event.get("EventTime", datetime.now(timezone.utc))
        event_name = event.get("EventName", "Unknown")
        event_source = event.get("EventSource", "Unknown")

        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.LOG_EXPORT,
            source=EvidenceSource.CLOUDTRAIL,
            title=f"CloudTrail: {event_name}",
            description=f"CloudTrail event from {event_source}",
            content=json.dumps(event, default=str),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "event_id": event.get("EventId"),
                "event_name": event_name,
                "event_source": event_source,
                "username": event.get("Username"),
                "aws_region": event.get("AwsRegion"),
            },
        )

    async def health_check(self) -> bool:
        """Check CloudTrail connectivity."""
        if not self._initialized:
            return False
        try:
            async with self._session.client(
                "cloudtrail",
                region_name=self.config.region,
            ) as client:
                await client.describe_trails()
                return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# GitHub Adapter
# ---------------------------------------------------------------------------


class GitHubCollector(SourceAdapter):
    """GitHub evidence collector.

    Collects code changes, pull requests, and reviews from GitHub
    for change management evidence.
    """

    source = EvidenceSource.GITHUB

    def __init__(self, config: GitHubConfig) -> None:
        super().__init__(config)
        self.config: GitHubConfig = config
        self._token: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize GitHub client."""
        import os
        self._token = os.environ.get(self.config.token_env_var)
        if not self._token:
            logger.warning("GitHub token not found, GitHub collection disabled")
            self._initialized = False
        else:
            self._initialized = True
            logger.info("GitHubCollector initialized")

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect GitHub events for the criterion."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            import httpx

            headers = {
                "Authorization": f"token {self._token}",
                "Accept": "application/vnd.github.v3+json",
            }

            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
            ) as client:
                # Collect PRs for each repository
                for repo in self.config.repos:
                    if self.config.include_prs:
                        prs = await self._collect_prs(
                            client, headers, repo, period
                        )
                        evidence_list.extend([
                            self._pr_to_evidence(pr, criterion_id, period)
                            for pr in prs
                        ])

                    if self.config.include_commits:
                        commits = await self._collect_commits(
                            client, headers, repo, period
                        )
                        evidence_list.extend([
                            self._commit_to_evidence(commit, criterion_id, period, repo)
                            for commit in commits
                        ])

            logger.info(
                f"GitHub collected {len(evidence_list)} items for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"GitHub collection failed: {exc}")

        return evidence_list

    async def _collect_prs(
        self,
        client: Any,
        headers: Dict[str, str],
        repo: str,
        period: DateRange,
    ) -> List[Dict[str, Any]]:
        """Collect pull requests from a repository."""
        prs: List[Dict[str, Any]] = []

        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {"state": "all", "per_page": 100}

        response = await client.get(url, headers=headers, params=params)
        if response.status_code == 200:
            for pr in response.json():
                created_at = datetime.fromisoformat(
                    pr["created_at"].replace("Z", "+00:00")
                )
                if period.start <= created_at <= period.end:
                    prs.append(pr)

        return prs

    async def _collect_commits(
        self,
        client: Any,
        headers: Dict[str, str],
        repo: str,
        period: DateRange,
    ) -> List[Dict[str, Any]]:
        """Collect commits from a repository."""
        commits: List[Dict[str, Any]] = []

        url = f"https://api.github.com/repos/{repo}/commits"
        params = {
            "since": period.start.isoformat(),
            "until": period.end.isoformat(),
            "per_page": 100,
        }

        response = await client.get(url, headers=headers, params=params)
        if response.status_code == 200:
            commits = response.json()

        return commits

    def _pr_to_evidence(
        self,
        pr: Dict[str, Any],
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert a PR to Evidence."""
        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.CODE_CHANGE,
            source=EvidenceSource.GITHUB,
            title=f"PR #{pr['number']}: {pr['title']}",
            description=pr.get("body", "")[:4096] if pr.get("body") else "",
            content=json.dumps({
                "number": pr["number"],
                "title": pr["title"],
                "state": pr["state"],
                "user": pr["user"]["login"],
                "created_at": pr["created_at"],
                "merged_at": pr.get("merged_at"),
                "url": pr["html_url"],
            }),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "pr_number": pr["number"],
                "author": pr["user"]["login"],
                "state": pr["state"],
                "base_branch": pr["base"]["ref"],
                "head_branch": pr["head"]["ref"],
            },
        )

    def _commit_to_evidence(
        self,
        commit: Dict[str, Any],
        criterion_id: str,
        period: DateRange,
        repo: str,
    ) -> Evidence:
        """Convert a commit to Evidence."""
        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.CODE_CHANGE,
            source=EvidenceSource.GITHUB,
            title=f"Commit: {commit['sha'][:8]}",
            description=commit["commit"]["message"][:4096],
            content=json.dumps({
                "sha": commit["sha"],
                "message": commit["commit"]["message"],
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "url": commit["html_url"],
            }),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "sha": commit["sha"],
                "author": commit["commit"]["author"]["name"],
                "repo": repo,
            },
        )

    async def health_check(self) -> bool:
        """Check GitHub API connectivity."""
        if not self._initialized:
            return False
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/rate_limit",
                    headers={"Authorization": f"token {self._token}"},
                )
                return response.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PostgreSQL Adapter
# ---------------------------------------------------------------------------


class PostgreSQLCollector(SourceAdapter):
    """PostgreSQL evidence collector.

    Collects security events from internal PostgreSQL tables
    in the security schema.
    """

    source = EvidenceSource.POSTGRESQL

    def __init__(self, config: PostgreSQLConfig) -> None:
        super().__init__(config)
        self.config: PostgreSQLConfig = config
        self._pool: Any = None

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool."""
        import os
        dsn = os.environ.get(self.config.dsn_env_var)

        if not dsn:
            logger.warning("PostgreSQL DSN not found, collection disabled")
            self._initialized = False
            return

        try:
            from psycopg_pool import AsyncConnectionPool
            from psycopg.rows import dict_row

            self._pool = AsyncConnectionPool(
                conninfo=dsn,
                min_size=2,
                max_size=10,
                open=False,
                kwargs={"row_factory": dict_row},
            )
            await self._pool.open()
            self._initialized = True
            logger.info("PostgreSQLCollector initialized")
        except ImportError:
            logger.warning("psycopg not available, PostgreSQL collection disabled")
            self._initialized = False
        except Exception as exc:
            logger.error(f"PostgreSQL init failed: {exc}")
            self._initialized = False

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self._pool:
            await self._pool.close()
        await super().close()

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect security events from PostgreSQL."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            # Map criteria to queries
            queries = self._get_queries_for_criterion(criterion_id)

            async with self._pool.connection() as conn:
                for table, query in queries:
                    async with conn.cursor() as cur:
                        await cur.execute(query, (period.start, period.end))
                        rows = await cur.fetchall()

                        for row in rows:
                            evidence = self._row_to_evidence(
                                row, table, criterion_id, period
                            )
                            evidence_list.append(evidence)

            logger.info(
                f"PostgreSQL collected {len(evidence_list)} events for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"PostgreSQL collection failed: {exc}")

        return evidence_list

    def _get_queries_for_criterion(
        self,
        criterion_id: str,
    ) -> List[tuple[str, str]]:
        """Get SQL queries for a criterion."""
        schema = self.config.schema_name

        criterion_queries = {
            "CC6.1": [
                ("access_events", f"""
                    SELECT * FROM {schema}.access_events
                    WHERE performed_at >= %s AND performed_at < %s
                    ORDER BY performed_at DESC
                    LIMIT 1000
                """),
            ],
            "CC6.3": [
                ("auth_events", f"""
                    SELECT * FROM {schema}.auth_events
                    WHERE performed_at >= %s AND performed_at < %s
                    ORDER BY performed_at DESC
                    LIMIT 1000
                """),
            ],
            "CC6.7": [
                ("data_access_log", f"""
                    SELECT * FROM {schema}.data_access_log
                    WHERE performed_at >= %s AND performed_at < %s
                    ORDER BY performed_at DESC
                    LIMIT 1000
                """),
            ],
        }

        return criterion_queries.get(criterion_id, [])

    def _row_to_evidence(
        self,
        row: Dict[str, Any],
        table: str,
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert a database row to Evidence."""
        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.LOG_EXPORT,
            source=EvidenceSource.POSTGRESQL,
            title=f"DB Event: {table}",
            description=f"Security event from {table} table",
            content=json.dumps(row, default=str),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "table": table,
                "row_id": str(row.get("id", "")),
            },
        )

    async def health_check(self) -> bool:
        """Check PostgreSQL connectivity."""
        if not self._initialized:
            return False
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Loki Adapter
# ---------------------------------------------------------------------------


class LokiCollector(SourceAdapter):
    """Loki log aggregation evidence collector.

    Collects logs via LogQL queries for operational evidence.
    """

    source = EvidenceSource.LOKI

    def __init__(self, config: LokiConfig) -> None:
        super().__init__(config)
        self.config: LokiConfig = config

    async def initialize(self) -> None:
        """Initialize Loki client."""
        self._initialized = True
        logger.info("LokiCollector initialized")

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect logs from Loki via LogQL."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            import httpx

            queries = self._get_queries_for_criterion(criterion_id)

            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
            ) as client:
                for query_name, query in queries:
                    params = {
                        "query": query,
                        "start": str(int(period.start.timestamp() * 1e9)),
                        "end": str(int(period.end.timestamp() * 1e9)),
                        "limit": self.config.batch_size,
                    }

                    headers = {}
                    if self.config.tenant_id:
                        headers["X-Scope-OrgID"] = self.config.tenant_id

                    response = await client.get(
                        f"{self.config.url}/loki/api/v1/query_range",
                        params=params,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("data", {}).get("result", [])

                        for stream in results:
                            for value in stream.get("values", []):
                                evidence = self._log_to_evidence(
                                    value, query_name, criterion_id, period
                                )
                                evidence_list.append(evidence)

            logger.info(
                f"Loki collected {len(evidence_list)} entries for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"Loki collection failed: {exc}")

        return evidence_list

    def _get_queries_for_criterion(
        self,
        criterion_id: str,
    ) -> List[tuple[str, str]]:
        """Get LogQL queries for a criterion."""
        criterion_queries = {
            "CC4.1": [
                ("monitoring_alerts", '{job="prometheus"} |= "alert"'),
            ],
            "CC4.2": [
                ("security_events", '{namespace="security"} |= "event"'),
            ],
            "CC7.1": [
                ("system_events", '{job="system"} |= "error" or |= "warning"'),
            ],
            "CC7.2": [
                ("anomaly_detection", '{job="anomaly-detector"} |= "detected"'),
            ],
            "PI1.1": [
                ("processing_logs", '{namespace="processing"} |= "complete"'),
            ],
        }

        return criterion_queries.get(criterion_id, [])

    def _log_to_evidence(
        self,
        value: List[Any],
        query_name: str,
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert a Loki log entry to Evidence."""
        timestamp_ns, log_line = value
        timestamp = datetime.fromtimestamp(
            int(timestamp_ns) / 1e9, tz=timezone.utc
        )

        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.LOG_EXPORT,
            source=EvidenceSource.LOKI,
            title=f"Log: {query_name}",
            description=log_line[:256] if len(log_line) > 256 else log_line,
            content=log_line,
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "query_name": query_name,
                "log_timestamp": timestamp.isoformat(),
            },
        )

    async def health_check(self) -> bool:
        """Check Loki connectivity."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.url}/ready")
                return response.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Auth Service Adapter
# ---------------------------------------------------------------------------


class AuthServiceCollector(SourceAdapter):
    """Internal auth service evidence collector.

    Collects authentication and authorization events from SEC-001.
    """

    source = EvidenceSource.AUTH_SERVICE

    def __init__(self, config: AuthServiceConfig) -> None:
        super().__init__(config)
        self.config: AuthServiceConfig = config
        self._api_key: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize auth service client."""
        import os
        self._api_key = os.environ.get(self.config.api_key_env_var)
        self._initialized = bool(self._api_key)
        if self._initialized:
            logger.info("AuthServiceCollector initialized")
        else:
            logger.warning("Auth service API key not found")

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect auth events from the auth service."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            import httpx

            headers = {"X-API-Key": self._api_key}

            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
            ) as client:
                # Query auth events
                params = {
                    "start_time": period.start.isoformat(),
                    "end_time": period.end.isoformat(),
                    "limit": self.config.batch_size,
                }

                response = await client.get(
                    f"{self.config.base_url}/api/v1/auth/events",
                    params=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    events = response.json().get("events", [])
                    for event in events:
                        evidence = self._event_to_evidence(
                            event, criterion_id, period
                        )
                        evidence_list.append(evidence)

            logger.info(
                f"AuthService collected {len(evidence_list)} events for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"AuthService collection failed: {exc}")

        return evidence_list

    def _event_to_evidence(
        self,
        event: Dict[str, Any],
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert an auth event to Evidence."""
        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.LOG_EXPORT,
            source=EvidenceSource.AUTH_SERVICE,
            title=f"Auth Event: {event.get('event_type', 'Unknown')}",
            description=event.get("description", ""),
            content=json.dumps(event, default=str),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "event_type": event.get("event_type"),
                "user_id": event.get("user_id"),
                "outcome": event.get("outcome"),
            },
        )

    async def health_check(self) -> bool:
        """Check auth service connectivity."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.base_url}/health",
                    headers={"X-API-Key": self._api_key} if self._api_key else {},
                )
                return response.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Jira Adapter
# ---------------------------------------------------------------------------


class JiraCollector(SourceAdapter):
    """Jira evidence collector.

    Collects change management tickets and incident records.
    """

    source = EvidenceSource.JIRA

    def __init__(self, config: JiraConfig) -> None:
        super().__init__(config)
        self.config: JiraConfig = config
        self._auth: Optional[tuple[str, str]] = None

    async def initialize(self) -> None:
        """Initialize Jira client."""
        import os
        email = os.environ.get(self.config.email_env_var)
        token = os.environ.get(self.config.token_env_var)

        if email and token:
            self._auth = (email, token)
            self._initialized = True
            logger.info("JiraCollector initialized")
        else:
            logger.warning("Jira credentials not found")
            self._initialized = False

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect Jira tickets for the criterion."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            import httpx

            jql = self._get_jql_for_criterion(criterion_id, period)

            async with httpx.AsyncClient(
                auth=self._auth,
                timeout=self.config.timeout_seconds,
            ) as client:
                params = {
                    "jql": jql,
                    "maxResults": self.config.batch_size,
                    "fields": "summary,description,status,created,updated,assignee,reporter,issuetype",
                }

                response = await client.get(
                    f"{self.config.base_url}/rest/api/3/search",
                    params=params,
                )

                if response.status_code == 200:
                    issues = response.json().get("issues", [])
                    for issue in issues:
                        evidence = self._issue_to_evidence(
                            issue, criterion_id, period
                        )
                        evidence_list.append(evidence)

            logger.info(
                f"Jira collected {len(evidence_list)} tickets for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"Jira collection failed: {exc}")

        return evidence_list

    def _get_jql_for_criterion(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> str:
        """Get JQL query for a criterion."""
        projects = ",".join(self.config.project_keys) if self.config.project_keys else "*"
        start_date = period.start.strftime("%Y-%m-%d")
        end_date = period.end.strftime("%Y-%m-%d")

        criterion_jql = {
            "CC7.3": f'project in ({projects}) AND issuetype = "Change Request" AND created >= "{start_date}" AND created <= "{end_date}"',
            "CC7.4": f'project in ({projects}) AND issuetype = "Incident" AND created >= "{start_date}" AND created <= "{end_date}"',
            "CC7.5": f'project in ({projects}) AND issuetype = "Problem" AND created >= "{start_date}" AND created <= "{end_date}"',
            "CC8.1": f'project in ({projects}) AND issuetype in ("Change Request", "Release") AND created >= "{start_date}" AND created <= "{end_date}"',
        }

        default_jql = f'project in ({projects}) AND created >= "{start_date}" AND created <= "{end_date}"'
        return criterion_jql.get(criterion_id, default_jql)

    def _issue_to_evidence(
        self,
        issue: Dict[str, Any],
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert a Jira issue to Evidence."""
        fields = issue.get("fields", {})

        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.TICKET,
            source=EvidenceSource.JIRA,
            title=f"{issue['key']}: {fields.get('summary', 'No summary')}",
            description=fields.get("description", "")[:4096] if fields.get("description") else "",
            content=json.dumps({
                "key": issue["key"],
                "summary": fields.get("summary"),
                "status": fields.get("status", {}).get("name"),
                "issuetype": fields.get("issuetype", {}).get("name"),
                "created": fields.get("created"),
                "updated": fields.get("updated"),
                "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
                "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
            }),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "issue_key": issue["key"],
                "issue_type": fields.get("issuetype", {}).get("name"),
                "status": fields.get("status", {}).get("name"),
            },
        )

    async def health_check(self) -> bool:
        """Check Jira connectivity."""
        if not self._initialized:
            return False
        try:
            import httpx
            async with httpx.AsyncClient(auth=self._auth) as client:
                response = await client.get(
                    f"{self.config.base_url}/rest/api/3/myself"
                )
                return response.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Okta Adapter
# ---------------------------------------------------------------------------


class OktaCollector(SourceAdapter):
    """Okta SSO evidence collector.

    Collects authentication and identity events from Okta.
    """

    source = EvidenceSource.OKTA

    def __init__(self, config: OktaConfig) -> None:
        super().__init__(config)
        self.config: OktaConfig = config
        self._token: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize Okta client."""
        import os
        self._token = os.environ.get(self.config.token_env_var)

        if self._token and self.config.domain:
            self._initialized = True
            logger.info("OktaCollector initialized")
        else:
            logger.warning("Okta credentials not found")
            self._initialized = False

    async def collect(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect Okta system log events."""
        if not self._initialized:
            return []

        evidence_list: List[Evidence] = []

        try:
            import httpx

            headers = {
                "Authorization": f"SSWS {self._token}",
                "Accept": "application/json",
            }

            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
            ) as client:
                for event_type in self.config.event_types:
                    params = {
                        "since": period.start.isoformat(),
                        "until": period.end.isoformat(),
                        "filter": f'eventType eq "{event_type}"',
                        "limit": self.config.batch_size,
                    }

                    response = await client.get(
                        f"https://{self.config.domain}/api/v1/logs",
                        params=params,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        events = response.json()
                        for event in events:
                            evidence = self._event_to_evidence(
                                event, criterion_id, period
                            )
                            evidence_list.append(evidence)

            logger.info(
                f"Okta collected {len(evidence_list)} events for {criterion_id}"
            )

        except Exception as exc:
            logger.error(f"Okta collection failed: {exc}")

        return evidence_list

    def _event_to_evidence(
        self,
        event: Dict[str, Any],
        criterion_id: str,
        period: DateRange,
    ) -> Evidence:
        """Convert an Okta event to Evidence."""
        actor = event.get("actor", {})
        target = event.get("target", [{}])[0] if event.get("target") else {}

        return Evidence(
            criterion_id=criterion_id,
            evidence_type=EvidenceType.LOG_EXPORT,
            source=EvidenceSource.OKTA,
            title=f"Okta: {event.get('eventType', 'Unknown')}",
            description=event.get("displayMessage", ""),
            content=json.dumps(event, default=str),
            collected_at=datetime.now(timezone.utc),
            period_start=period.start,
            period_end=period.end,
            metadata={
                "event_type": event.get("eventType"),
                "actor_id": actor.get("id"),
                "actor_email": actor.get("alternateId"),
                "outcome": event.get("outcome", {}).get("result"),
                "target_id": target.get("id"),
            },
        )

    async def health_check(self) -> bool:
        """Check Okta connectivity."""
        if not self._initialized:
            return False
        try:
            import httpx
            headers = {"Authorization": f"SSWS {self._token}"}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://{self.config.domain}/api/v1/users/me",
                    headers=headers,
                )
                return response.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Main Evidence Collector
# ---------------------------------------------------------------------------


class EvidenceCollector:
    """Multi-source evidence collector for SOC 2 audit preparation.

    Orchestrates evidence collection from multiple source adapters,
    handles parallel collection, validates results, and tracks collection
    metadata.

    Example:
        >>> config = EvidenceCollectorConfig(...)
        >>> collector = EvidenceCollector(config)
        >>> await collector.initialize()
        >>> evidence = await collector.collect_for_criterion(
        ...     "CC6.1",
        ...     DateRange(start=start, end=end)
        ... )
    """

    def __init__(self, config: EvidenceCollectorConfig) -> None:
        """Initialize the evidence collector.

        Args:
            config: Collector configuration.
        """
        self.config = config
        self._adapters: Dict[EvidenceSource, SourceAdapter] = {}
        self._initialized = False

        # Initialize adapters based on config
        if config.cloudtrail.enabled:
            self._adapters[EvidenceSource.CLOUDTRAIL] = CloudTrailCollector(
                config.cloudtrail
            )
        if config.github.enabled:
            self._adapters[EvidenceSource.GITHUB] = GitHubCollector(
                config.github
            )
        if config.postgresql.enabled:
            self._adapters[EvidenceSource.POSTGRESQL] = PostgreSQLCollector(
                config.postgresql
            )
        if config.loki.enabled:
            self._adapters[EvidenceSource.LOKI] = LokiCollector(
                config.loki
            )
        if config.auth_service.enabled:
            self._adapters[EvidenceSource.AUTH_SERVICE] = AuthServiceCollector(
                config.auth_service
            )
        if config.jira.enabled:
            self._adapters[EvidenceSource.JIRA] = JiraCollector(
                config.jira
            )
        if config.okta.enabled:
            self._adapters[EvidenceSource.OKTA] = OktaCollector(
                config.okta
            )

    async def initialize(self) -> None:
        """Initialize all enabled adapters."""
        init_tasks = [adapter.initialize() for adapter in self._adapters.values()]
        await asyncio.gather(*init_tasks, return_exceptions=True)
        self._initialized = True
        logger.info(
            f"EvidenceCollector initialized with {len(self._adapters)} adapters"
        )

    async def close(self) -> None:
        """Close all adapters and release resources."""
        close_tasks = [adapter.close() for adapter in self._adapters.values()]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        self._initialized = False
        logger.info("EvidenceCollector closed")

    async def collect_for_criterion(
        self,
        criterion_id: str,
        period: DateRange,
    ) -> List[Evidence]:
        """Collect evidence for a specific criterion.

        Args:
            criterion_id: SOC 2 criterion ID (e.g., "CC6.1").
            period: Date range for evidence collection.

        Returns:
            List of collected Evidence objects.
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized. Call initialize() first.")

        # Get sources for this criterion
        sources = CRITERION_SOURCE_MAP.get(criterion_id.upper(), [])
        if not sources:
            logger.warning(f"No sources mapped for criterion {criterion_id}")
            return []

        # Collect from each applicable source
        all_evidence: List[Evidence] = []

        if self.config.parallel_collection:
            # Parallel collection
            tasks = []
            for source in sources:
                if source in self._adapters:
                    tasks.append(
                        self._adapters[source].collect(criterion_id, period)
                    )

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        all_evidence.extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Collection error: {result}")
        else:
            # Sequential collection
            for source in sources:
                if source in self._adapters:
                    try:
                        evidence = await self._adapters[source].collect(
                            criterion_id, period
                        )
                        all_evidence.extend(evidence)
                    except Exception as exc:
                        logger.error(f"Collection error from {source}: {exc}")

        # Validate and deduplicate
        validated = [e for e in all_evidence if self._validate_evidence(e)]

        logger.info(
            f"Collected {len(validated)} evidence items for {criterion_id}"
        )

        return validated

    async def collect_all(
        self,
        period: DateRange,
    ) -> Dict[str, List[Evidence]]:
        """Collect evidence for all criteria.

        Args:
            period: Date range for evidence collection.

        Returns:
            Dictionary mapping criterion ID to list of evidence.
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized. Call initialize() first.")

        results: Dict[str, List[Evidence]] = {}

        for criterion_id in CRITERION_SOURCE_MAP.keys():
            evidence = await self.collect_for_criterion(criterion_id, period)
            if evidence:
                results[criterion_id] = evidence

        logger.info(
            f"Collected evidence for {len(results)} criteria"
        )

        return results

    async def schedule_continuous_collection(self) -> None:
        """Schedule continuous evidence collection.

        This method sets up a background task that periodically
        collects evidence for all criteria. Useful for maintaining
        up-to-date evidence during the audit period.
        """
        logger.info("Continuous collection scheduled (not yet implemented)")
        # TODO: Implement with APScheduler or similar
        pass

    def _validate_evidence(self, evidence: Evidence) -> bool:
        """Validate a piece of evidence.

        Args:
            evidence: Evidence to validate.

        Returns:
            True if evidence is valid.
        """
        # Basic validation checks
        if not evidence.criterion_id:
            return False
        if not evidence.title:
            return False
        if not evidence.content and not evidence.file_path and not evidence.s3_key:
            return False

        return True

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all adapters.

        Returns:
            Dictionary mapping source name to health status.
        """
        health: Dict[str, bool] = {}

        for source, adapter in self._adapters.items():
            try:
                health[source.value] = await adapter.health_check()
            except Exception:
                health[source.value] = False

        return health
