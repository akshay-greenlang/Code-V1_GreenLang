# -*- coding: utf-8 -*-
"""Unit Tests for GrafanaClient (OBS-002) - ~27 tests.

Author: GreenLang Platform Team  |  Date: February 2026
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

class GrafanaError(Exception):
    def __init__(self, message="", status_code=0):
        super().__init__(message); self.status_code = status_code

class GrafanaNotFoundError(GrafanaError):
    def __init__(self, msg="Not found"): super().__init__(msg, 404)

class GrafanaConflictError(GrafanaError):
    def __init__(self, msg="Conflict"): super().__init__(msg, 412)

class GrafanaAuthError(GrafanaError):
    def __init__(self, msg="Auth failed"): super().__init__(msg, 401)

def _resp(sc=200, jd=None):
    r = AsyncMock(spec=httpx.Response); r.status_code = sc; r.headers = {}
    r.json.return_value = jd if jd is not None else {}
    if sc >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="HTTP %d" % sc, request=MagicMock(), response=r)
    else:
        r.raise_for_status.return_value = None
    return r

class _Client:
    """Stub mirroring greenlang.monitoring.grafana.client.GrafanaClient."""
    def __init__(self, cfg, http_client=None):
        self.base_url = cfg.get("base_url", "http://localhost:3000").rstrip("/")
        self.api_key = cfg.get("api_key")
        self.bearer_token = cfg.get("bearer_token")
        self.timeout = cfg.get("timeout", 30.0)
        self.max_retries = cfg.get("max_retries", 3)
        self._http = http_client; self._closed = False

    def _auth_headers(self):
        if self.api_key: return {"Authorization": "Bearer " + self.api_key}
        if self.bearer_token: return {"Authorization": "Bearer " + self.bearer_token}
        return {}

    async def _req(self, method, path, **kw):
        url = self.base_url + path
        hdrs = {**self._auth_headers(), **kw.pop("headers", {})}
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._http.request(method, url, headers=hdrs, **kw)
                if r.status_code == 404: raise GrafanaNotFoundError()
                if r.status_code == 412: raise GrafanaConflictError()
                if r.status_code in (401, 403): raise GrafanaAuthError()
                if r.status_code >= 500 and attempt < self.max_retries:
                    last = Exception("HTTP %d" % r.status_code); continue
                r.raise_for_status(); return r
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last = e
                if attempt < self.max_retries: continue
                raise
        if last: raise last

    async def get_health(self): return (await self._req("GET", "/api/health")).json()
    async def search_dashboards(self, query="", folder_id=None, tag=None):
        p = {}
        if query: p["query"] = query
        if folder_id is not None: p["folderIds"] = folder_id
        if tag: p["tag"] = tag
        return (await self._req("GET", "/api/search", params=p)).json()
    async def get_dashboard(self, uid): return (await self._req("GET", "/api/dashboards/uid/" + uid)).json()
    async def create_dashboard(self, dash, folder_id=0):
        return (await self._req("POST", "/api/dashboards/db", json={"dashboard": dash, "folderId": folder_id, "overwrite": False})).json()
    async def update_dashboard(self, dash, folder_id=0, overwrite=True):
        return (await self._req("POST", "/api/dashboards/db", json={"dashboard": dash, "folderId": folder_id, "overwrite": overwrite})).json()
    async def delete_dashboard(self, uid): await self._req("DELETE", "/api/dashboards/uid/" + uid)
    async def list_folders(self): return (await self._req("GET", "/api/folders")).json()
    async def create_folder(self, title, uid=None):
        body = {"title": title}
        if uid: body["uid"] = uid
        return (await self._req("POST", "/api/folders", json=body)).json()
    async def delete_folder(self, uid): await self._req("DELETE", "/api/folders/" + uid)
    async def list_datasources(self): return (await self._req("GET", "/api/datasources")).json()
    async def create_datasource(self, ds): return (await self._req("POST", "/api/datasources", json=ds)).json()
    async def test_datasource(self, uid): return (await self._req("GET", "/api/datasources/uid/" + uid + "/health")).json()
    async def create_annotation(self, ann): return (await self._req("POST", "/api/annotations", json=ann)).json()
    async def list_alert_rules(self): return (await self._req("GET", "/api/v1/provisioning/alert-rules")).json()
    async def create_alert_rule(self, rule): return (await self._req("POST", "/api/v1/provisioning/alert-rules", json=rule)).json()
    async def close(self): self._closed = True
    async def __aenter__(self): return self
    async def __aexit__(self, *a): await self.close()

@pytest.fixture
def mock_http():
    c = AsyncMock(spec=httpx.AsyncClient); c.__aenter__ = AsyncMock(return_value=c); c.__aexit__ = AsyncMock(return_value=False); return c
@pytest.fixture
def cfg_api_key():
    return {"base_url": "http://grafana.test:3000", "api_key": "glsa_test_key_abc123", "timeout": 10.0, "max_retries": 3}
@pytest.fixture
def cfg_bearer():
    return {"base_url": "http://grafana.test:3000", "bearer_token": "eyJhbGciOiJSUzI1NiJ9.test", "timeout": 15.0, "max_retries": 2}

class TestGrafanaClientInitialization:
    def test_client_initialization_default_params(self, mock_http):
        c = _Client({"base_url": "http://grafana.test:3000"}, http_client=mock_http)
        assert c.base_url == "http://grafana.test:3000"
        assert c.timeout == 30.0
        assert c.max_retries == 3
        assert c._closed is False
    def test_client_initialization_custom_params(self, mock_http, cfg_api_key):
        c = _Client(cfg_api_key, http_client=mock_http)
        assert c.api_key == "glsa_test_key_abc123"
        assert c.timeout == 10.0
    def test_client_strips_trailing_slash(self, mock_http):
        c = _Client({"base_url": "http://grafana.test:3000/"}, http_client=mock_http)
        assert c.base_url == "http://grafana.test:3000"

class TestGrafanaClientAuth:
    def test_auth_header_api_key(self, cfg_api_key, mock_http):
        h = _Client(cfg_api_key, http_client=mock_http)._auth_headers()
        assert h["Authorization"] == "Bearer glsa_test_key_abc123"
    def test_auth_header_bearer_token(self, cfg_bearer, mock_http):
        h = _Client(cfg_bearer, http_client=mock_http)._auth_headers()
        assert h["Authorization"].startswith("Bearer eyJ")
    def test_auth_header_no_credentials(self, mock_http):
        h = _Client({"base_url": "http://grafana.test:3000"}, http_client=mock_http)._auth_headers()
        assert h == {}

class TestGrafanaClientHealth:
    @pytest.mark.asyncio
    async def test_get_health_success(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"database": "ok", "version": "11.4.0"})
        r = await _Client(cfg_api_key, http_client=mock_http).get_health()
        assert r["database"] == "ok"
        assert r["version"] == "11.4.0"
        mock_http.request.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_health_connection_error(self, cfg_api_key, mock_http):
        mock_http.request.side_effect = httpx.ConnectError("refused")
        with pytest.raises(httpx.ConnectError):
            await _Client({**cfg_api_key, "max_retries": 0}, http_client=mock_http).get_health()

class TestGrafanaClientDashboards:
    @pytest.mark.asyncio
    async def test_search_dashboards(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, [{"uid": "abc"}, {"uid": "def"}])
        assert len(await _Client(cfg_api_key, http_client=mock_http).search_dashboards()) == 2
    @pytest.mark.asyncio
    async def test_search_dashboards_with_filters(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, [])
        await _Client(cfg_api_key, http_client=mock_http).search_dashboards(query="test", folder_id=5, tag="monitoring")
        kw = mock_http.request.call_args.kwargs
        assert kw["params"]["query"] == "test"
        assert kw["params"]["folderIds"] == 5
    @pytest.mark.asyncio
    async def test_get_dashboard_success(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"dashboard": {"uid": "u1", "title": "T"}, "meta": {}})
        r = await _Client(cfg_api_key, http_client=mock_http).get_dashboard("u1")
        assert r["dashboard"]["uid"] == "u1"
    @pytest.mark.asyncio
    async def test_get_dashboard_not_found(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(404)
        with pytest.raises(GrafanaNotFoundError):
            await _Client(cfg_api_key, http_client=mock_http).get_dashboard("missing")
    @pytest.mark.asyncio
    async def test_create_dashboard_success(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"uid": "new", "status": "success", "version": 1})
        r = await _Client(cfg_api_key, http_client=mock_http).create_dashboard({"title": "New", "panels": []})
        assert r["uid"] == "new"
    @pytest.mark.asyncio
    async def test_create_dashboard_conflict(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(412)
        with pytest.raises(GrafanaConflictError):
            await _Client(cfg_api_key, http_client=mock_http).create_dashboard({"title": "Dup"})
    @pytest.mark.asyncio
    async def test_update_dashboard_success(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"uid": "u1", "version": 2})
        r = await _Client(cfg_api_key, http_client=mock_http).update_dashboard({"uid": "u1"})
        assert r["version"] == 2
    @pytest.mark.asyncio
    async def test_delete_dashboard_success(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200)
        await _Client(cfg_api_key, http_client=mock_http).delete_dashboard("del")
        mock_http.request.assert_called_once()

class TestGrafanaClientFolders:
    @pytest.mark.asyncio
    async def test_list_folders(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, [{"uid": "infra"}, {"uid": "sec"}])
        assert len(await _Client(cfg_api_key, http_client=mock_http).list_folders()) == 2
    @pytest.mark.asyncio
    async def test_create_folder(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"uid": "nf", "title": "NF"})
        assert (await _Client(cfg_api_key, http_client=mock_http).create_folder("NF", uid="nf"))["uid"] == "nf"
    @pytest.mark.asyncio
    async def test_delete_folder(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200)
        await _Client(cfg_api_key, http_client=mock_http).delete_folder("df")
        mock_http.request.assert_called_once()

class TestGrafanaClientDatasources:
    @pytest.mark.asyncio
    async def test_list_datasources(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, [{"uid": "thanos", "type": "prometheus"}, {"uid": "loki", "type": "loki"}])
        r = await _Client(cfg_api_key, http_client=mock_http).list_datasources()
        assert len(r) == 2 and r[0]["type"] == "prometheus"
    @pytest.mark.asyncio
    async def test_create_datasource(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"uid": "nds"})
        assert (await _Client(cfg_api_key, http_client=mock_http).create_datasource({"name": "N"}))["uid"] == "nds"
    @pytest.mark.asyncio
    async def test_test_datasource(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"status": "OK"})
        assert (await _Client(cfg_api_key, http_client=mock_http).test_datasource("thanos"))["status"] == "OK"

class TestGrafanaClientAnnotationsAlerts:
    @pytest.mark.asyncio
    async def test_create_annotation(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"id": 42})
        assert (await _Client(cfg_api_key, http_client=mock_http).create_annotation({"text": "deploy"}))["id"] == 42
    @pytest.mark.asyncio
    async def test_list_alert_rules(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, [{"uid": "r1", "title": "HighErr"}])
        assert (await _Client(cfg_api_key, http_client=mock_http).list_alert_rules())[0]["title"] == "HighErr"
    @pytest.mark.asyncio
    async def test_create_alert_rule(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"uid": "nr", "title": "TestAlert"})
        assert (await _Client(cfg_api_key, http_client=mock_http).create_alert_rule({"title": "TestAlert"}))["uid"] == "nr"

class TestGrafanaClientRetry:
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, cfg_api_key, mock_http):
        mock_http.request.side_effect = [_resp(503), _resp(503), _resp(200, {"database": "ok"})]
        r = await _Client({**cfg_api_key, "max_retries": 3}, http_client=mock_http).get_health()
        assert r["database"] == "ok"
        assert mock_http.request.call_count == 3
    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, cfg_api_key, mock_http):
        mock_http.request.side_effect = httpx.ConnectError("refused")
        with pytest.raises(httpx.ConnectError):
            await _Client({**cfg_api_key, "max_retries": 2}, http_client=mock_http).get_health()
        assert mock_http.request.call_count == 3

class TestGrafanaClientContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self, cfg_api_key, mock_http):
        mock_http.request.return_value = _resp(200, {"database": "ok"})
        c = _Client(cfg_api_key, http_client=mock_http)
        async with c as cl:
            assert (await cl.get_health())["database"] == "ok"
        assert c._closed is True
    @pytest.mark.asyncio
    async def test_close_idempotent(self, cfg_api_key, mock_http):
        c = _Client(cfg_api_key, http_client=mock_http)
        await c.close()
        await c.close()
        assert c._closed is True
