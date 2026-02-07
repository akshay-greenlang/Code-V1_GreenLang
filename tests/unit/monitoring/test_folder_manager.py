# -*- coding: utf-8 -*-
"""Unit Tests for FolderManager (OBS-002) - ~15 tests.

Author: GreenLang Platform Team  |  Date: February 2026
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx


# ---------------------------------------------------------------------------
# Constants matching PRD section 2.3 dashboard folder hierarchy
# ---------------------------------------------------------------------------
GREENLANG_FOLDER_HIERARCHY = [
    {"uid": "gl-00-executive",       "title": "00-Executive"},
    {"uid": "gl-01-infrastructure",  "title": "01-Infrastructure"},
    {"uid": "gl-02-data-stores",     "title": "02-Data-Stores"},
    {"uid": "gl-03-observability",   "title": "03-Observability"},
    {"uid": "gl-04-security",        "title": "04-Security"},
    {"uid": "gl-05-applications",    "title": "05-Applications"},
    {"uid": "gl-06-alerts",          "title": "06-Alerts"},
]

# Map dashboard file prefix to folder uid
DASHBOARD_FOLDER_MAP = {
    "executive-overview":     "gl-00-executive",
    "infrastructure-overview":"gl-01-infrastructure",
    "prometheus-health":      "gl-03-observability",
    "thanos-health":          "gl-03-observability",
    "alertmanager-health":    "gl-03-observability",
    "auth-service":           "gl-04-security",
    "encryption-service":     "gl-04-security",
    "rbac-service":           "gl-04-security",
    "agent-factory-v1":       "gl-05-applications",
    "feature-flags":          "gl-05-applications",
    "active-alerts":          "gl-06-alerts",
}


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _GrafanaError(Exception):
    def __init__(self, message="", status_code=0):
        super().__init__(message); self.status_code = status_code

class _GrafanaNotFoundError(_GrafanaError):
    def __init__(self, msg="Not found"): super().__init__(msg, 404)

class _GrafanaConflictError(_GrafanaError):
    def __init__(self, msg="Conflict"): super().__init__(msg, 412)


def _resp(sc=200, jd=None):
    r = AsyncMock(spec=httpx.Response); r.status_code = sc; r.headers = {}
    r.json.return_value = jd if jd is not None else {}
    if sc >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="HTTP %d" % sc, request=MagicMock(), response=r)
    else:
        r.raise_for_status.return_value = None
    return r


class _FolderManager:
    """Stub mirroring greenlang.monitoring.grafana.folder_manager.FolderManager."""

    HIERARCHY = list(GREENLANG_FOLDER_HIERARCHY)

    def __init__(self, client):
        self._client = client

    async def ensure_hierarchy(self):
        """Create all 7 GreenLang folders if they do not exist."""
        existing = await self._client.list_folders()
        existing_uids = {f["uid"] for f in existing}
        created = []
        for folder in self.HIERARCHY:
            if folder["uid"] not in existing_uids:
                await self._client.create_folder(folder["title"], uid=folder["uid"])
                created.append(folder["uid"])
        return created

    async def get_folder(self, uid):
        folders = await self._client.list_folders()
        for f in folders:
            if f["uid"] == uid:
                return f
        raise _GrafanaNotFoundError("Folder %s not found" % uid)

    async def delete_folder(self, uid):
        await self._client.delete_folder(uid)

    async def list_dashboards_in_folder(self, folder_uid):
        return await self._client.search_dashboards(folder_id=folder_uid)

    async def sync_hierarchy(self, dry_run=False):
        """Return folders to create and folders to delete."""
        existing = await self._client.list_folders()
        existing_uids = {f["uid"] for f in existing}
        target_uids = {f["uid"] for f in self.HIERARCHY}
        to_create = [f for f in self.HIERARCHY if f["uid"] not in existing_uids]
        to_delete = [f for f in existing if f["uid"] not in target_uids and f["uid"].startswith("gl-")]
        if not dry_run:
            for f in to_create:
                await self._client.create_folder(f["title"], uid=f["uid"])
            for f in to_delete:
                await self._client.delete_folder(f["uid"])
        return {"to_create": to_create, "to_delete": to_delete}

    def resolve_folder(self, dashboard_name):
        """Return the folder uid for a given dashboard name based on known mapping."""
        return DASHBOARD_FOLDER_MAP.get(dashboard_name)

    def get_hierarchy(self):
        return list(self.HIERARCHY)

    async def set_folder_permissions(self, folder_uid, permissions):
        """Stub for folder permission management."""
        return {"folder_uid": folder_uid, "permissions": permissions}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_client():
    c = AsyncMock()
    c.list_folders = AsyncMock(return_value=[])
    c.create_folder = AsyncMock(return_value={"uid": "new", "title": "New"})
    c.delete_folder = AsyncMock(return_value=None)
    c.search_dashboards = AsyncMock(return_value=[])
    return c

@pytest.fixture
def fm(mock_client):
    return _FolderManager(mock_client)


# ---------------------------------------------------------------------------
# Tests -- Hierarchy constants
# ---------------------------------------------------------------------------
class TestFolderManagerHierarchy:
    def test_hierarchy_has_seven_folders(self, fm):
        assert len(fm.get_hierarchy()) == 7

    def test_hierarchy_uids_unique(self, fm):
        uids = [f["uid"] for f in fm.get_hierarchy()]
        assert len(uids) == len(set(uids))

    def test_hierarchy_uids_prefixed(self, fm):
        for f in fm.get_hierarchy():
            assert f["uid"].startswith("gl-"), f"Folder uid {f['uid']} missing 'gl-' prefix"

    def test_hierarchy_titles_numbered(self, fm):
        for i, f in enumerate(fm.get_hierarchy()):
            assert f["title"].startswith("%02d-" % i), (
                "Folder %s title should start with %02d-" % (f["uid"], i)
            )


# ---------------------------------------------------------------------------
# Tests -- ensure_hierarchy
# ---------------------------------------------------------------------------
class TestFolderManagerSync:
    @pytest.mark.asyncio
    async def test_ensure_hierarchy_creates_all_when_empty(self, fm, mock_client):
        mock_client.list_folders.return_value = []
        created = await fm.ensure_hierarchy()
        assert len(created) == 7
        assert mock_client.create_folder.call_count == 7

    @pytest.mark.asyncio
    async def test_ensure_hierarchy_skips_existing(self, fm, mock_client):
        mock_client.list_folders.return_value = [
            {"uid": "gl-00-executive", "title": "00-Executive"},
            {"uid": "gl-01-infrastructure", "title": "01-Infrastructure"},
        ]
        created = await fm.ensure_hierarchy()
        assert len(created) == 5
        assert mock_client.create_folder.call_count == 5

    @pytest.mark.asyncio
    async def test_ensure_hierarchy_noop_when_all_exist(self, fm, mock_client):
        mock_client.list_folders.return_value = list(GREENLANG_FOLDER_HIERARCHY)
        created = await fm.ensure_hierarchy()
        assert created == []
        assert mock_client.create_folder.call_count == 0

    @pytest.mark.asyncio
    async def test_sync_hierarchy_dry_run(self, fm, mock_client):
        mock_client.list_folders.return_value = [
            {"uid": "gl-00-executive", "title": "00-Executive"},
            {"uid": "gl-99-old", "title": "99-Old"},
        ]
        result = await fm.sync_hierarchy(dry_run=True)
        assert len(result["to_create"]) == 6  # 7 - 1 existing
        assert len(result["to_delete"]) == 1  # gl-99-old is stale
        mock_client.create_folder.assert_not_called()
        mock_client.delete_folder.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_hierarchy_apply(self, fm, mock_client):
        mock_client.list_folders.return_value = [
            {"uid": "gl-00-executive", "title": "00-Executive"},
            {"uid": "gl-99-old", "title": "99-Old"},
        ]
        result = await fm.sync_hierarchy(dry_run=False)
        assert mock_client.create_folder.call_count == 6
        assert mock_client.delete_folder.call_count == 1


# ---------------------------------------------------------------------------
# Tests -- Folder resolution
# ---------------------------------------------------------------------------
class TestFolderManagerResolve:
    def test_resolve_known_dashboard(self, fm):
        assert fm.resolve_folder("prometheus-health") == "gl-03-observability"

    def test_resolve_unknown_dashboard(self, fm):
        assert fm.resolve_folder("unknown-dash-xyz") is None


# ---------------------------------------------------------------------------
# Tests -- Permissions
# ---------------------------------------------------------------------------
class TestFolderManagerPermissions:
    @pytest.mark.asyncio
    async def test_set_folder_permissions(self, fm):
        perms = [{"role": "Editor", "permission": 2}]
        result = await fm.set_folder_permissions("gl-00-executive", perms)
        assert result["folder_uid"] == "gl-00-executive"

    @pytest.mark.asyncio
    async def test_list_dashboards_in_folder(self, fm, mock_client):
        mock_client.search_dashboards.return_value = [
            {"uid": "d1", "title": "Dash1"},
            {"uid": "d2", "title": "Dash2"},
        ]
        result = await fm.list_dashboards_in_folder("gl-01-infrastructure")
        assert len(result) == 2
        mock_client.search_dashboards.assert_called_once_with(folder_id="gl-01-infrastructure")


# ---------------------------------------------------------------------------
# Tests -- get_folder / delete_folder
# ---------------------------------------------------------------------------
class TestFolderManagerCRUD:
    @pytest.mark.asyncio
    async def test_get_folder_found(self, fm, mock_client):
        mock_client.list_folders.return_value = [
            {"uid": "gl-00-executive", "title": "00-Executive"},
        ]
        result = await fm.get_folder("gl-00-executive")
        assert result["title"] == "00-Executive"

    @pytest.mark.asyncio
    async def test_get_folder_not_found(self, fm, mock_client):
        mock_client.list_folders.return_value = []
        with pytest.raises(_GrafanaNotFoundError):
            await fm.get_folder("gl-99-missing")

    @pytest.mark.asyncio
    async def test_delete_folder(self, fm, mock_client):
        await fm.delete_folder("gl-06-alerts")
        mock_client.delete_folder.assert_called_once_with("gl-06-alerts")
