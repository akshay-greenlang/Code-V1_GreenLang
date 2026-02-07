# -*- coding: utf-8 -*-
"""
Grafana HTTP API Client
========================

Async HTTP client for the Grafana API with retry logic, structured logging,
and comprehensive error handling. Uses httpx for async HTTP and tenacity for
retry with exponential backoff.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from greenlang.monitoring.grafana.models import (
    AlertRule,
    ContactPoint,
    Dashboard,
    DashboardSearchResult,
    DataSource,
    Folder,
    FolderPermission,
    HealthStatus,
    NotificationPolicy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class GrafanaError(Exception):
    """Base exception for all Grafana API errors.

    Attributes:
        status_code: HTTP status code from the API response.
        message: Human-readable error description.
        response_body: Raw response body for debugging.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        response_body: str = "",
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"[{status_code}] {message}")


class GrafanaNotFoundError(GrafanaError):
    """Raised when a requested resource does not exist (HTTP 404)."""


class GrafanaConflictError(GrafanaError):
    """Raised when a resource already exists or version conflict (HTTP 409/412)."""


class GrafanaAuthError(GrafanaError):
    """Raised when authentication or authorisation fails (HTTP 401/403)."""


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

_RETRY_POLICY = retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GrafanaClient:
    """Async HTTP client for the Grafana REST API.

    Provides typed methods for dashboard, folder, data-source, alerting,
    annotation, and health-check operations. All HTTP requests are retried
    up to 3 times with exponential backoff for transient transport errors.

    Usage::

        async with GrafanaClient(base_url="https://grafana.greenlang.io",
                                  api_key="glsa_...") as client:
            health = await client.get_health()
            dashboards = await client.search_dashboards(query="overview")

    Args:
        base_url: Grafana server URL (no trailing slash).
        api_key: Grafana Service Account token or API key.
        timeout: Request timeout in seconds.
        verify_ssl: Whether to verify TLS certificates.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._verify_ssl = verify_ssl
        self._client: Optional[httpx.AsyncClient] = None

    # -- Context manager -----------------------------------------------------

    async def __aenter__(self) -> GrafanaClient:
        """Open the underlying httpx async client."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
            verify=self._verify_ssl,
        )
        logger.info("GrafanaClient connected to %s", self._base_url)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the underlying httpx async client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("GrafanaClient disconnected")

    # -- Internal helpers ----------------------------------------------------

    def _ensure_connected(self) -> httpx.AsyncClient:
        """Return the active client or raise if not connected."""
        if self._client is None:
            raise RuntimeError(
                "GrafanaClient is not connected. Use 'async with GrafanaClient(...) as client:'"
            )
        return self._client

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP error codes to typed exceptions."""
        code = response.status_code
        if 200 <= code < 300:
            return

        body = response.text
        if code == 401 or code == 403:
            raise GrafanaAuthError(
                f"Authentication/authorisation failed: {body}",
                status_code=code,
                response_body=body,
            )
        if code == 404:
            raise GrafanaNotFoundError(
                f"Resource not found: {body}",
                status_code=code,
                response_body=body,
            )
        if code in (409, 412):
            raise GrafanaConflictError(
                f"Resource conflict: {body}",
                status_code=code,
                response_body=body,
            )
        raise GrafanaError(
            f"Grafana API error: {body}",
            status_code=code,
            response_body=body,
        )

    # -- Health --------------------------------------------------------------

    @_RETRY_POLICY
    async def get_health(self) -> HealthStatus:
        """Check Grafana server health.

        Returns:
            HealthStatus with commit, database status, and version.

        Raises:
            GrafanaError: If the health endpoint returns a non-2xx status.
        """
        client = self._ensure_connected()
        response = await client.get("/api/health")
        self._raise_for_status(response)
        return HealthStatus.model_validate(response.json())

    # -- Dashboards ----------------------------------------------------------

    @_RETRY_POLICY
    async def get_dashboard(self, uid: str) -> Dashboard:
        """Fetch a dashboard by UID.

        Args:
            uid: Dashboard unique identifier.

        Returns:
            Dashboard model populated from the API response.

        Raises:
            GrafanaNotFoundError: If the dashboard does not exist.
        """
        client = self._ensure_connected()
        response = await client.get(f"/api/dashboards/uid/{uid}")
        self._raise_for_status(response)
        data = response.json()
        return Dashboard.model_validate(data.get("dashboard", data))

    @_RETRY_POLICY
    async def create_dashboard(
        self,
        dashboard: Dashboard,
        folder_uid: str = "",
        overwrite: bool = False,
        message: str = "",
    ) -> dict[str, Any]:
        """Create or update a dashboard.

        Args:
            dashboard: Dashboard model to persist.
            folder_uid: Target folder UID.
            overwrite: Whether to overwrite an existing dashboard with the same UID.
            message: Commit message for the dashboard version.

        Returns:
            API response dict with id, uid, url, status, version, slug.
        """
        client = self._ensure_connected()
        payload: dict[str, Any] = {
            "dashboard": dashboard.model_dump(exclude_none=True),
            "overwrite": overwrite,
        }
        if folder_uid:
            payload["folderUid"] = folder_uid
        if message:
            payload["message"] = message

        response = await client.post("/api/dashboards/db", json=payload)
        self._raise_for_status(response)
        result = response.json()
        logger.info(
            "Dashboard saved: uid=%s title=%s version=%s",
            result.get("uid"),
            dashboard.title,
            result.get("version"),
        )
        return result

    @_RETRY_POLICY
    async def update_dashboard(
        self,
        dashboard: Dashboard,
        folder_uid: str = "",
        message: str = "",
    ) -> dict[str, Any]:
        """Update an existing dashboard (overwrite=True).

        Args:
            dashboard: Dashboard model with updated content.
            folder_uid: Target folder UID.
            message: Commit message for the version.

        Returns:
            API response dict with id, uid, url, status, version, slug.
        """
        return await self.create_dashboard(
            dashboard=dashboard,
            folder_uid=folder_uid,
            overwrite=True,
            message=message,
        )

    @_RETRY_POLICY
    async def delete_dashboard(self, uid: str) -> None:
        """Delete a dashboard by UID.

        Args:
            uid: Dashboard UID to delete.

        Raises:
            GrafanaNotFoundError: If the dashboard does not exist.
        """
        client = self._ensure_connected()
        response = await client.delete(f"/api/dashboards/uid/{uid}")
        self._raise_for_status(response)
        logger.info("Dashboard deleted: uid=%s", uid)

    @_RETRY_POLICY
    async def search_dashboards(
        self,
        query: str = "",
        folder_ids: Optional[list[int]] = None,
        tag: Optional[list[str]] = None,
        type_: str = "dash-db",
        limit: int = 100,
    ) -> list[DashboardSearchResult]:
        """Search dashboards and folders.

        Args:
            query: Search query string.
            folder_ids: Restrict results to these folder IDs.
            tag: Filter by tags.
            type_: Result type filter: 'dash-db' or 'dash-folder'.
            limit: Maximum number of results.

        Returns:
            List of search results.
        """
        client = self._ensure_connected()
        params: dict[str, Any] = {"type": type_, "limit": limit}
        if query:
            params["query"] = query
        if folder_ids:
            params["folderIds"] = ",".join(str(fid) for fid in folder_ids)
        if tag:
            params["tag"] = tag

        response = await client.get("/api/search", params=params)
        self._raise_for_status(response)
        return [DashboardSearchResult.model_validate(item) for item in response.json()]

    # -- Folders -------------------------------------------------------------

    @_RETRY_POLICY
    async def get_folders(self) -> list[Folder]:
        """List all folders.

        Returns:
            List of Folder models.
        """
        client = self._ensure_connected()
        response = await client.get("/api/folders")
        self._raise_for_status(response)
        return [Folder.model_validate(item) for item in response.json()]

    @_RETRY_POLICY
    async def get_folder(self, uid: str) -> Folder:
        """Get a folder by UID.

        Args:
            uid: Folder UID.

        Returns:
            Folder model.

        Raises:
            GrafanaNotFoundError: If the folder does not exist.
        """
        client = self._ensure_connected()
        response = await client.get(f"/api/folders/{uid}")
        self._raise_for_status(response)
        return Folder.model_validate(response.json())

    @_RETRY_POLICY
    async def create_folder(self, title: str, uid: str = "", parent_uid: str = "") -> Folder:
        """Create a new folder.

        Args:
            title: Folder display name.
            uid: Optional UID (auto-generated if empty).
            parent_uid: Parent folder UID for nested folders.

        Returns:
            Created Folder model.
        """
        client = self._ensure_connected()
        payload: dict[str, str] = {"title": title}
        if uid:
            payload["uid"] = uid
        if parent_uid:
            payload["parentUid"] = parent_uid

        response = await client.post("/api/folders", json=payload)
        self._raise_for_status(response)
        folder = Folder.model_validate(response.json())
        logger.info("Folder created: uid=%s title=%s", folder.uid, folder.title)
        return folder

    @_RETRY_POLICY
    async def update_folder(
        self, uid: str, title: str, version: int, overwrite: bool = False
    ) -> Folder:
        """Update an existing folder.

        Args:
            uid: Folder UID.
            title: New folder title.
            version: Current folder version for optimistic concurrency.
            overwrite: Force overwrite regardless of version.

        Returns:
            Updated Folder model.
        """
        client = self._ensure_connected()
        payload: dict[str, Any] = {
            "title": title,
            "version": version,
            "overwrite": overwrite,
        }
        response = await client.put(f"/api/folders/{uid}", json=payload)
        self._raise_for_status(response)
        return Folder.model_validate(response.json())

    @_RETRY_POLICY
    async def delete_folder(self, uid: str) -> None:
        """Delete a folder and all its dashboards.

        Args:
            uid: Folder UID.
        """
        client = self._ensure_connected()
        response = await client.delete(f"/api/folders/{uid}")
        self._raise_for_status(response)
        logger.info("Folder deleted: uid=%s", uid)

    @_RETRY_POLICY
    async def get_folder_permissions(self, uid: str) -> list[FolderPermission]:
        """Get folder permissions.

        Args:
            uid: Folder UID.

        Returns:
            List of FolderPermission models.
        """
        client = self._ensure_connected()
        response = await client.get(f"/api/folders/{uid}/permissions")
        self._raise_for_status(response)
        return [FolderPermission.model_validate(item) for item in response.json()]

    @_RETRY_POLICY
    async def set_folder_permissions(
        self, uid: str, permissions: list[FolderPermission]
    ) -> None:
        """Set folder permissions (replaces all existing permissions).

        Args:
            uid: Folder UID.
            permissions: Complete list of permissions to set.
        """
        client = self._ensure_connected()
        payload = {"items": [p.model_dump() for p in permissions]}
        response = await client.post(f"/api/folders/{uid}/permissions", json=payload)
        self._raise_for_status(response)
        logger.info("Folder permissions updated: uid=%s count=%d", uid, len(permissions))

    # -- Data sources --------------------------------------------------------

    @_RETRY_POLICY
    async def get_datasources(self) -> list[DataSource]:
        """List all data sources.

        Returns:
            List of DataSource models.
        """
        client = self._ensure_connected()
        response = await client.get("/api/datasources")
        self._raise_for_status(response)
        return [DataSource.model_validate(item) for item in response.json()]

    @_RETRY_POLICY
    async def get_datasource(self, uid: str) -> DataSource:
        """Get a data source by UID.

        Args:
            uid: Data source UID.

        Returns:
            DataSource model.
        """
        client = self._ensure_connected()
        response = await client.get(f"/api/datasources/uid/{uid}")
        self._raise_for_status(response)
        return DataSource.model_validate(response.json())

    @_RETRY_POLICY
    async def create_datasource(self, datasource: DataSource) -> DataSource:
        """Create a new data source.

        Args:
            datasource: DataSource model to create.

        Returns:
            Created DataSource model with server-assigned ID.
        """
        client = self._ensure_connected()
        response = await client.post(
            "/api/datasources",
            json=datasource.model_dump(exclude_none=True),
        )
        self._raise_for_status(response)
        result = response.json()
        logger.info(
            "DataSource created: uid=%s name=%s type=%s",
            result.get("datasource", {}).get("uid", datasource.uid),
            datasource.name,
            datasource.type,
        )
        return DataSource.model_validate(result.get("datasource", result))

    @_RETRY_POLICY
    async def update_datasource(self, datasource: DataSource) -> DataSource:
        """Update an existing data source.

        Args:
            datasource: DataSource model with updated fields. The id field must be set.

        Returns:
            Updated DataSource model.
        """
        client = self._ensure_connected()
        if datasource.id is None:
            raise ValueError("DataSource.id must be set for updates")
        response = await client.put(
            f"/api/datasources/{datasource.id}",
            json=datasource.model_dump(exclude_none=True),
        )
        self._raise_for_status(response)
        result = response.json()
        return DataSource.model_validate(result.get("datasource", result))

    @_RETRY_POLICY
    async def delete_datasource(self, uid: str) -> None:
        """Delete a data source by UID.

        Args:
            uid: Data source UID.
        """
        client = self._ensure_connected()
        response = await client.delete(f"/api/datasources/uid/{uid}")
        self._raise_for_status(response)
        logger.info("DataSource deleted: uid=%s", uid)

    @_RETRY_POLICY
    async def test_datasource(self, uid: str) -> dict[str, Any]:
        """Health-check a data source.

        Args:
            uid: Data source UID.

        Returns:
            Dict with status and message.
        """
        client = self._ensure_connected()
        # Grafana 11+ uses /api/datasources/uid/{uid}/health
        response = await client.get(f"/api/datasources/uid/{uid}/health")
        self._raise_for_status(response)
        return response.json()

    # -- Alert rules (Unified Alerting) --------------------------------------

    @_RETRY_POLICY
    async def get_alert_rules(self) -> list[AlertRule]:
        """List all alert rules.

        Returns:
            List of AlertRule models.
        """
        client = self._ensure_connected()
        response = await client.get("/api/v1/provisioning/alert-rules")
        self._raise_for_status(response)
        return [AlertRule.model_validate(item) for item in response.json()]

    @_RETRY_POLICY
    async def get_alert_rule(self, uid: str) -> AlertRule:
        """Get an alert rule by UID.

        Args:
            uid: Alert rule UID.

        Returns:
            AlertRule model.
        """
        client = self._ensure_connected()
        response = await client.get(f"/api/v1/provisioning/alert-rules/{uid}")
        self._raise_for_status(response)
        return AlertRule.model_validate(response.json())

    @_RETRY_POLICY
    async def create_alert_rule(self, rule: AlertRule) -> AlertRule:
        """Create a new alert rule.

        Args:
            rule: AlertRule model to create.

        Returns:
            Created AlertRule model.
        """
        client = self._ensure_connected()
        response = await client.post(
            "/api/v1/provisioning/alert-rules",
            json=rule.model_dump(by_alias=True, exclude_none=True),
        )
        self._raise_for_status(response)
        result = AlertRule.model_validate(response.json())
        logger.info("AlertRule created: uid=%s title=%s", result.uid, result.title)
        return result

    @_RETRY_POLICY
    async def update_alert_rule(self, uid: str, rule: AlertRule) -> AlertRule:
        """Update an existing alert rule.

        Args:
            uid: Alert rule UID to update.
            rule: AlertRule model with updated fields.

        Returns:
            Updated AlertRule model.
        """
        client = self._ensure_connected()
        response = await client.put(
            f"/api/v1/provisioning/alert-rules/{uid}",
            json=rule.model_dump(by_alias=True, exclude_none=True),
        )
        self._raise_for_status(response)
        return AlertRule.model_validate(response.json())

    @_RETRY_POLICY
    async def delete_alert_rule(self, uid: str) -> None:
        """Delete an alert rule by UID.

        Args:
            uid: Alert rule UID.
        """
        client = self._ensure_connected()
        response = await client.delete(f"/api/v1/provisioning/alert-rules/{uid}")
        self._raise_for_status(response)
        logger.info("AlertRule deleted: uid=%s", uid)

    # -- Contact points ------------------------------------------------------

    @_RETRY_POLICY
    async def get_contact_points(self) -> list[ContactPoint]:
        """List all contact points.

        Returns:
            List of ContactPoint models.
        """
        client = self._ensure_connected()
        response = await client.get("/api/v1/provisioning/contact-points")
        self._raise_for_status(response)
        return [ContactPoint.model_validate(item) for item in response.json()]

    @_RETRY_POLICY
    async def create_contact_point(self, cp: ContactPoint) -> ContactPoint:
        """Create a new contact point.

        Args:
            cp: ContactPoint model to create.

        Returns:
            Created ContactPoint model.
        """
        client = self._ensure_connected()
        response = await client.post(
            "/api/v1/provisioning/contact-points",
            json=cp.model_dump(exclude_none=True),
        )
        self._raise_for_status(response)
        result = ContactPoint.model_validate(response.json())
        logger.info("ContactPoint created: name=%s type=%s", result.name, result.type)
        return result

    @_RETRY_POLICY
    async def update_contact_point(self, uid: str, cp: ContactPoint) -> None:
        """Update a contact point.

        Args:
            uid: Contact point UID.
            cp: ContactPoint model with updated fields.
        """
        client = self._ensure_connected()
        response = await client.put(
            f"/api/v1/provisioning/contact-points/{uid}",
            json=cp.model_dump(exclude_none=True),
        )
        self._raise_for_status(response)

    @_RETRY_POLICY
    async def delete_contact_point(self, uid: str) -> None:
        """Delete a contact point.

        Args:
            uid: Contact point UID.
        """
        client = self._ensure_connected()
        response = await client.delete(f"/api/v1/provisioning/contact-points/{uid}")
        self._raise_for_status(response)
        logger.info("ContactPoint deleted: uid=%s", uid)

    # -- Notification policies -----------------------------------------------

    @_RETRY_POLICY
    async def get_notification_policy(self) -> NotificationPolicy:
        """Get the notification policy tree.

        Returns:
            Root NotificationPolicy node.
        """
        client = self._ensure_connected()
        response = await client.get("/api/v1/provisioning/policies")
        self._raise_for_status(response)
        return NotificationPolicy.model_validate(response.json())

    @_RETRY_POLICY
    async def set_notification_policy(self, policy: NotificationPolicy) -> None:
        """Set the entire notification policy tree (replaces existing).

        Args:
            policy: Root NotificationPolicy node.
        """
        client = self._ensure_connected()
        response = await client.put(
            "/api/v1/provisioning/policies",
            json=policy.model_dump(by_alias=True, exclude_none=True),
        )
        self._raise_for_status(response)
        logger.info("Notification policy updated")

    # -- Mute timings --------------------------------------------------------

    @_RETRY_POLICY
    async def get_mute_timings(self) -> list[dict[str, Any]]:
        """List all mute timings.

        Returns:
            List of mute timing dicts.
        """
        client = self._ensure_connected()
        response = await client.get("/api/v1/provisioning/mute-timings")
        self._raise_for_status(response)
        return response.json()

    @_RETRY_POLICY
    async def create_mute_timing(self, name: str, time_intervals: list[dict[str, Any]]) -> dict[str, Any]:
        """Create a mute timing.

        Args:
            name: Mute timing name.
            time_intervals: List of time interval definitions.

        Returns:
            Created mute timing dict.
        """
        client = self._ensure_connected()
        payload = {"name": name, "time_intervals": time_intervals}
        response = await client.post("/api/v1/provisioning/mute-timings", json=payload)
        self._raise_for_status(response)
        logger.info("Mute timing created: name=%s", name)
        return response.json()

    # -- Annotations ---------------------------------------------------------

    @_RETRY_POLICY
    async def create_annotation(
        self,
        text: str,
        tags: Optional[list[str]] = None,
        dashboard_uid: str = "",
        panel_id: int = 0,
        time_from: int = 0,
        time_to: int = 0,
    ) -> dict[str, Any]:
        """Create a dashboard annotation.

        Args:
            text: Annotation text / description.
            tags: Tags for filtering.
            dashboard_uid: Dashboard UID to attach to (empty for global).
            panel_id: Panel ID to attach to (0 for dashboard-level).
            time_from: Epoch milliseconds for start (0 = server now).
            time_to: Epoch milliseconds for end (0 = same as time_from).

        Returns:
            Created annotation dict with id and message.
        """
        client = self._ensure_connected()
        payload: dict[str, Any] = {"text": text}
        if tags:
            payload["tags"] = tags
        if dashboard_uid:
            payload["dashboardUID"] = dashboard_uid
        if panel_id:
            payload["panelId"] = panel_id
        if time_from:
            payload["time"] = time_from
        if time_to:
            payload["timeEnd"] = time_to

        response = await client.post("/api/annotations", json=payload)
        self._raise_for_status(response)
        result = response.json()
        logger.info("Annotation created: id=%s", result.get("id"))
        return result

    @_RETRY_POLICY
    async def get_annotations(
        self,
        dashboard_uid: str = "",
        panel_id: int = 0,
        tags: Optional[list[str]] = None,
        from_: int = 0,
        to: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query annotations.

        Args:
            dashboard_uid: Filter by dashboard UID.
            panel_id: Filter by panel ID.
            tags: Filter by tags.
            from_: Epoch milliseconds start.
            to: Epoch milliseconds end.
            limit: Maximum number of results.

        Returns:
            List of annotation dicts.
        """
        client = self._ensure_connected()
        params: dict[str, Any] = {"limit": limit}
        if dashboard_uid:
            params["dashboardUID"] = dashboard_uid
        if panel_id:
            params["panelId"] = panel_id
        if tags:
            params["tags"] = tags
        if from_:
            params["from"] = from_
        if to:
            params["to"] = to

        response = await client.get("/api/annotations", params=params)
        self._raise_for_status(response)
        return response.json()
