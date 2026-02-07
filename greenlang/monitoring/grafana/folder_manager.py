# -*- coding: utf-8 -*-
"""
Grafana Folder Manager
=======================

Manages the GreenLang folder hierarchy in Grafana, including folder creation,
nested folder support (Grafana 11+), permission assignment, and dashboard
organisation.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.monitoring.grafana.client import (
    GrafanaClient,
    GrafanaConflictError,
    GrafanaNotFoundError,
)
from greenlang.monitoring.grafana.models import (
    DashboardSearchResult,
    Folder,
    FolderPermission,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GreenLang standard folder hierarchy (from PRD Section 2.3)
# ---------------------------------------------------------------------------

GREENLANG_FOLDER_HIERARCHY: dict[str, dict[str, Any]] = {
    "gl-executive": {
        "title": "00-Executive",
        "description": "Executive dashboards, business KPIs, and platform overview",
    },
    "gl-infrastructure": {
        "title": "01-Infrastructure",
        "description": "Kubernetes, Kong, CI/CD, feature flags, agent factory dashboards",
    },
    "gl-data-stores": {
        "title": "02-Data-Stores",
        "description": "PostgreSQL, Redis, S3, pgvector, TimescaleDB dashboards",
    },
    "gl-observability": {
        "title": "03-Observability",
        "description": "Prometheus, Thanos, Loki, Alertmanager, Grafana health dashboards",
    },
    "gl-security": {
        "title": "04-Security",
        "description": "Auth, RBAC, encryption, TLS, audit, secrets, scanning, SOC2 dashboards",
    },
    "gl-applications": {
        "title": "05-Applications",
        "description": "GreenLang agents, API performance, DR status dashboards",
    },
    "gl-alerts": {
        "title": "06-Alerts",
        "description": "Active alerts summary and alert analytics dashboards",
    },
}


class FolderManager:
    """Manage Grafana folders and their permissions.

    Provides methods to create and synchronise the GreenLang folder hierarchy,
    set role-based and team-based permissions, and query folder contents.

    Usage::

        async with GrafanaClient(base_url, api_key) as client:
            manager = FolderManager(client)
            await manager.create_folder_hierarchy()
            await manager.set_role_permissions("gl-executive", "Viewer", permission=1)
    """

    def __init__(self, client: GrafanaClient) -> None:
        """Initialise the folder manager.

        Args:
            client: Connected GrafanaClient instance.
        """
        self._client = client

    # -- Hierarchy management ------------------------------------------------

    async def create_folder_hierarchy(
        self,
        parent_uid: str = "",
        hierarchy: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, Folder]:
        """Create the GreenLang folder hierarchy in Grafana.

        Creates folders that do not already exist and skips those that do.
        Returns a mapping of folder UID to Folder model for all folders
        in the hierarchy.

        Args:
            parent_uid: Optional parent folder UID for nested folders.
            hierarchy: Custom folder hierarchy dict. Defaults to the
                       standard GREENLANG_FOLDER_HIERARCHY.

        Returns:
            Dict mapping folder UID to created/existing Folder models.
        """
        if hierarchy is None:
            hierarchy = GREENLANG_FOLDER_HIERARCHY

        created: dict[str, Folder] = {}
        for uid, config in hierarchy.items():
            title = config["title"]
            try:
                folder = await self._client.get_folder(uid)
                logger.info("Folder already exists: uid=%s title=%s", uid, title)
                created[uid] = folder
            except GrafanaNotFoundError:
                folder = await self._client.create_folder(
                    title=title,
                    uid=uid,
                    parent_uid=parent_uid,
                )
                created[uid] = folder
                logger.info("Folder created: uid=%s title=%s", uid, title)

        logger.info(
            "Folder hierarchy sync complete: %d folders", len(created)
        )
        return created

    async def sync_folders(
        self,
        hierarchy: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, Folder]:
        """Synchronise folder hierarchy: create missing, update titles if changed.

        This is idempotent -- calling it multiple times produces the same result.

        Args:
            hierarchy: Custom folder hierarchy dict. Defaults to the
                       standard GREENLANG_FOLDER_HIERARCHY.

        Returns:
            Dict mapping folder UID to Folder models.
        """
        if hierarchy is None:
            hierarchy = GREENLANG_FOLDER_HIERARCHY

        synced: dict[str, Folder] = {}
        for uid, config in hierarchy.items():
            title = config["title"]
            try:
                existing = await self._client.get_folder(uid)
                if existing.title != title:
                    updated = await self._client.update_folder(
                        uid=uid,
                        title=title,
                        version=existing.version,
                    )
                    synced[uid] = updated
                    logger.info("Folder title updated: uid=%s '%s' -> '%s'", uid, existing.title, title)
                else:
                    synced[uid] = existing
            except GrafanaNotFoundError:
                folder = await self._client.create_folder(title=title, uid=uid)
                synced[uid] = folder
                logger.info("Folder created during sync: uid=%s title=%s", uid, title)

        return synced

    # -- Permissions ---------------------------------------------------------

    async def set_team_permissions(
        self,
        folder_uid: str,
        team_id: int,
        permission: int = 1,
    ) -> None:
        """Grant a team permission on a folder.

        Merges the new permission with existing permissions rather than
        replacing them.

        Args:
            folder_uid: Folder UID.
            team_id: Grafana team ID.
            permission: Permission level (1=View, 2=Edit, 4=Admin).
        """
        existing = await self._client.get_folder_permissions(folder_uid)
        # Remove any existing entry for this team
        filtered = [p for p in existing if p.teamId != team_id]
        filtered.append(FolderPermission(teamId=team_id, permission=permission))
        await self._client.set_folder_permissions(folder_uid, filtered)
        logger.info(
            "Team permission set: folder=%s team=%d permission=%d",
            folder_uid, team_id, permission,
        )

    async def set_role_permissions(
        self,
        folder_uid: str,
        role: str,
        permission: int = 1,
    ) -> None:
        """Grant an org role permission on a folder.

        Merges the new permission with existing permissions rather than
        replacing them.

        Args:
            folder_uid: Folder UID.
            role: Grafana org role ('Viewer', 'Editor', 'Admin').
            permission: Permission level (1=View, 2=Edit, 4=Admin).
        """
        existing = await self._client.get_folder_permissions(folder_uid)
        # Remove any existing entry for this role
        filtered = [p for p in existing if p.role != role]
        filtered.append(FolderPermission(role=role, permission=permission))
        await self._client.set_folder_permissions(folder_uid, filtered)
        logger.info(
            "Role permission set: folder=%s role=%s permission=%d",
            folder_uid, role, permission,
        )

    async def set_default_permissions(self) -> None:
        """Apply default GreenLang permission scheme to all standard folders.

        - Viewer role gets View (1) on all folders.
        - Editor role gets Edit (2) on all folders.
        - Admin role gets Admin (4) on all folders.
        """
        role_permissions = [
            ("Viewer", 1),
            ("Editor", 2),
            ("Admin", 4),
        ]
        for folder_uid in GREENLANG_FOLDER_HIERARCHY:
            permissions = [
                FolderPermission(role=role, permission=perm)
                for role, perm in role_permissions
            ]
            await self._client.set_folder_permissions(folder_uid, permissions)
            logger.info("Default permissions applied: folder=%s", folder_uid)

    # -- Querying ------------------------------------------------------------

    async def get_folder_dashboards(
        self,
        folder_uid: str,
    ) -> list[DashboardSearchResult]:
        """List all dashboards in a folder.

        Args:
            folder_uid: Folder UID.

        Returns:
            List of DashboardSearchResult for dashboards in the folder.
        """
        folder = await self._client.get_folder(folder_uid)
        if folder.id is None:
            return []
        return await self._client.search_dashboards(folder_ids=[folder.id])

    async def move_dashboard(
        self,
        dashboard_uid: str,
        target_folder_uid: str,
        message: str = "Moved via GreenLang SDK",
    ) -> dict[str, Any]:
        """Move a dashboard to a different folder.

        Args:
            dashboard_uid: Dashboard UID to move.
            target_folder_uid: Target folder UID.
            message: Version commit message.

        Returns:
            API response dict from the dashboard save.
        """
        dashboard = await self._client.get_dashboard(dashboard_uid)
        return await self._client.create_dashboard(
            dashboard=dashboard,
            folder_uid=target_folder_uid,
            overwrite=True,
            message=message,
        )

    async def list_folders(self) -> list[Folder]:
        """List all Grafana folders.

        Returns:
            List of Folder models.
        """
        return await self._client.get_folders()
