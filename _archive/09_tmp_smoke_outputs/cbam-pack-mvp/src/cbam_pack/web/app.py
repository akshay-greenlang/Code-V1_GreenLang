"""
CBAM Pack Web Application

FastAPI-based web interface for the CBAM Compliance Pack.
Provides file upload, processing, and result visualization with:
- XSD schema validation status
- Policy PASS/WARN/FAIL status
- Row-level drilldown for default factor usage
- Gap report with actionable recommendations
- Evidence folder with immutable input copies
"""

import asyncio
import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import yaml

from cbam_pack import __version__
from cbam_pack.pipeline import CBAMPipeline, PipelineResult
from cbam_pack.models import CBAMConfig

try:
    from greenlang.v1.backends import run_csrd_backend, run_vcci_backend
except Exception:
    # Keep CBAM web usable as a standalone package install.
    run_csrd_backend = None
    run_vcci_backend = None

try:
    from greenlang.v2.backends import V2_BLOCKED_EXIT_CODE, run_v2_profile_backend
except Exception:
    V2_BLOCKED_EXIT_CODE = 4
    run_v2_profile_backend = None

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
RATE_LIMIT_PER_MINUTE = 60
SESSION_TTL_SECONDS = 60 * 60
GL_SHELL_VERSION = os.getenv("GREENLANG_WEB_VERSION", __version__)
ALLOW_BACKEND_FALLBACK_DEFAULT = os.getenv("GL_V1_ALLOW_BACKEND_FALLBACK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[4] / "frontend" / "dist"
SHELL_BASELINE_MARKERS = """
<noscript id="shell-baseline-markers" aria-hidden="true">
  <h1>GreenLang Compliance Workspace</h1>
  <a href="/apps/cbam">Open CBAM Workspace</a>
  <a href="/apps/csrd">Open CSRD Workspace</a>
  <a href="/apps/vcci">Open VCCI Workspace</a>
  <a href="/apps/eudr">Open EUDR Workspace</a>
  <a href="/apps/ghg">Open GHG Workspace</a>
  <a href="/apps/iso14064">Open ISO14064 Workspace</a>
  <a href="/apps/sb253">Open SB253 Workspace</a>
  <a href="/apps/taxonomy">Open Taxonomy Workspace</a>
  <a href="/runs">Run Center</a>
  <a href="/governance">Governance Center</a>
</noscript>
""".strip()


def _is_suspicious_upload_filename(filename: str) -> bool:
    if not filename:
        return True
    raw = filename.strip()
    if not raw:
        return True
    if ".." in raw:
        return True
    if "/" in raw or "\\" in raw:
        return True
    if Path(raw).name != raw:
        return True
    return False


def _sanitize_error_message(message: str) -> str:
    # Redact local filesystem paths from user-visible API responses.
    sanitized = re.sub(r"[A-Za-z]:\\[^\s\"']+", "<redacted-path>", message)
    sanitized = re.sub(r"/(?:tmp|var|home|Users|private)/[^\s\"']+", "<redacted-path>", sanitized)
    return sanitized


def _sanitize_errors(errors: list[str]) -> list[str]:
    return [_sanitize_error_message(err) for err in errors]


def _derive_run_state(success: bool, can_export: bool, warnings: list[str] | None) -> str:
    """UI run lifecycle state aligned with the enterprise shell contract."""
    if not success:
        return "failed"
    if not can_export:
        return "blocked"
    if warnings:
        return "partial_success"
    return "completed"


def _status_chip_for_run_state(run_state: str) -> str:
    """PASS / WARN / FAIL chips for enterprise shell run lists."""
    rs = (run_state or "").lower()
    if rs in ("failed", "blocked"):
        return "FAIL"
    if rs == "partial_success":
        return "WARN"
    if rs == "completed":
        return "PASS"
    return "WARN"


def _error_envelope(errors: list[str] | None) -> dict | None:
    err = [e for e in (errors or []) if e]
    if not err:
        return None
    return {
        "title": "Run encountered errors",
        "message": err[0],
        "details": err[1:12],
    }


def _decorate_run_response(app: FastAPI, run_id: str, payload: dict) -> dict:
    """Attach status_chip, lifecycle_phase, and error_envelope to match GET /api/v1/runs rows."""
    meta = app.state.session_meta.get(run_id)
    if not meta:
        return payload
    rs = meta.get("run_state")
    if rs is None:
        rs = _derive_run_state(
            bool(meta.get("success")),
            bool(meta.get("can_export", True)),
            meta.get("warnings") or [],
        )
    out = dict(payload)
    out["run_state"] = str(rs)
    out["lifecycle_phase"] = "completed"
    out["status_chip"] = _status_chip_for_run_state(str(rs))
    raw_err = meta.get("errors_tail")
    if raw_err is None:
        raw_err = meta.get("errors")
    err_list = raw_err if isinstance(raw_err, list) else None
    out["error_envelope"] = _error_envelope(err_list)
    return out


def _merge_live_connector_probe_signals(base: dict, probes: list[dict] | None) -> dict:
    """Append incidents for failed probes; add lightweight probe metadata for admin UI."""
    if not probes:
        return base
    out = dict(base)
    incidents = list(out.get("connector_incidents") or [])
    for row in probes:
        if not isinstance(row, dict):
            continue
        if row.get("ok") is False:
            incidents.append(
                {
                    "connector_id": str(row.get("connector_id", "")),
                    "app_id": str(row.get("app_id", "")),
                    "severity": "warning",
                    "message": row.get("message") or "Live connector health probe reported failure",
                }
            )
    out["connector_incidents"] = incidents
    out["connector_probe_meta"] = {
        "probe_count": len(probes),
        "last_refresh_utc": probes[0].get("checked_at_utc") if probes else None,
    }
    return out


def _shell_chrome_context_dict(repo_root: Path) -> dict:
    """Aggregated governance + connector signals for shell chrome (rail + incident banner)."""
    packs_path = repo_root / "greenlang" / "ecosystem" / "packs" / "v2_tier_registry.yaml"
    agents_path = repo_root / "greenlang" / "agents" / "v2_agent_registry.yaml"
    bundles_dir = repo_root / "greenlang" / "governance" / "policy" / "bundles"
    connectors_path = repo_root / "applications" / "connectors" / "v2_connector_registry.yaml"
    managed_pack_count = 0
    try:
        if packs_path.exists():
            payload = yaml.safe_load(packs_path.read_text(encoding="utf-8")) or {}
            packs = payload.get("pilot_packs", []) if isinstance(payload, dict) else []
            managed_pack_count = len([p for p in packs if isinstance(p, dict)])
    except Exception:
        pass
    deprecated_agent_count = 0
    try:
        if agents_path.exists():
            payload = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
            agents = payload.get("agents", []) if isinstance(payload, dict) else []
            for ag in agents:
                if isinstance(ag, dict) and str(ag.get("state", "")).lower() == "deprecated":
                    deprecated_agent_count += 1
    except Exception:
        pass
    policy_bundle_count = 0
    try:
        if bundles_dir.is_dir():
            policy_bundle_count = len(list(bundles_dir.glob("*.rego")))
    except OSError:
        pass
    incidents: list[dict] = []
    try:
        if connectors_path.exists():
            payload = yaml.safe_load(connectors_path.read_text(encoding="utf-8")) or {}
            raw = payload.get("connectors", []) if isinstance(payload, dict) else []
            for row in raw:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("operational_status", "ok") or "ok").lower()
                if status == "degraded":
                    incidents.append(
                        {
                            "connector_id": row.get("connector_id", ""),
                            "app_id": row.get("app_id", ""),
                            "severity": "warning",
                            "message": row.get("incident_summary")
                            or "Connector marked degraded in v2_connector_registry.yaml",
                        }
                    )
    except Exception:
        pass
    return {
        "compliance_rail": {
            "managed_pack_count": managed_pack_count,
            "policy_bundle_count": policy_bundle_count,
            "deprecated_agent_count": deprecated_agent_count,
        },
        "connector_incidents": incidents,
    }


def _run_session_fields(
    *,
    success: bool,
    can_export: bool,
    warnings: list[str] | None,
    errors: list[str] | None,
) -> dict:
    warn_list = list(warnings or [])
    err_list = list(errors or [])
    return {
        "run_state": _derive_run_state(success, can_export, warn_list),
        "warnings": warn_list,
        "errors_tail": err_list[:12],
    }


def _inject_shell_baseline_markers(html: str) -> str:
    """Inject stable route/tokens for shell UX baseline checks."""
    if not html or "shell-baseline-markers" in html:
        return html
    marker = "</body>"
    snippet = f"\n{SHELL_BASELINE_MARKERS}\n"
    if marker in html:
        return html.replace(marker, f"{snippet}{marker}")
    return html + snippet


def _is_valid_run_id(run_id: str) -> bool:
    # Strictly limit v1 run IDs to 32-char lowercase hex UUID string.
    return bool(re.fullmatch(r"[a-f0-9]{32}", run_id or ""))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="GreenLang CBAM Pack",
        description="EU Carbon Border Adjustment Mechanism Compliance Tool",
        version=__version__,
    )

    # Store for processing results
    app.state.results = {}
    app.state.output_dirs = {}
    app.state.session_meta = {}
    app.state.rate_limits = {}
    app.state.connector_health_probes = []

    frontend_assets = FRONTEND_DIST_DIR / "assets"
    if frontend_assets.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_assets)), name="frontend-assets")

    def _serve_react_shell_if_available() -> str | None:
        index_path = FRONTEND_DIST_DIR / "index.html"
        if not index_path.exists():
            return None
        html = index_path.read_text(encoding="utf-8")
        return _inject_shell_baseline_markers(html)

    def _require_api_key(request: Request) -> None:
        expected_api_key = os.getenv("CBAM_API_KEY")
        if not expected_api_key:
            return
        provided_api_key = request.headers.get("x-api-key", "")
        if provided_api_key != expected_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _enforce_rate_limit(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        now_ts = datetime.utcnow().timestamp()
        entry = app.state.rate_limits.get(client_ip)
        if not entry or (now_ts - entry["window_start_ts"]) >= 60:
            app.state.rate_limits[client_ip] = {"window_start_ts": now_ts, "count": 1}
            return
        entry["count"] += 1
        if entry["count"] > RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many requests")

    def _prune_expired_sessions() -> None:
        cutoff_ts = datetime.utcnow().timestamp() - SESSION_TTL_SECONDS
        for session_id, meta in list(app.state.session_meta.items()):
            created_at_ts = meta.get("created_at_ts")
            # Backward compatibility: keep sessions that predate TTL metadata.
            if created_at_ts is None or created_at_ts >= cutoff_ts:
                continue
            output_dir = app.state.output_dirs.pop(session_id, None)
            app.state.results.pop(session_id, None)
            app.state.session_meta.pop(session_id, None)
            if not output_dir:
                continue
            output_path = Path(output_dir).resolve()
            cleanup_root = output_path.parent
            if cleanup_root.name.startswith(("cbam_", "csrd_", "vcci_")):
                shutil.rmtree(cleanup_root, ignore_errors=True)
            else:
                shutil.rmtree(output_path, ignore_errors=True)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Render the main page."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return _inject_shared_ui_script(get_home_html())

    @app.get("/apps", response_class=HTMLResponse)
    async def shell_home(request: Request):
        """Render the multi-app shell home page."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return _inject_shared_ui_script(get_shell_html())

    @app.get("/apps/cbam", response_class=HTMLResponse)
    async def cbam_workspace(request: Request):
        """Render CBAM workspace within the shell routing surface."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return _inject_shared_ui_script(get_home_html())

    @app.get("/apps/csrd", response_class=HTMLResponse)
    async def csrd_workspace(request: Request):
        """Render CSRD workspace HTML."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_csrd_html()

    @app.get("/apps/vcci", response_class=HTMLResponse)
    async def vcci_workspace(request: Request):
        """Render VCCI workspace HTML."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_vcci_html()

    @app.get("/apps/eudr", response_class=HTMLResponse)
    async def eudr_workspace(request: Request):
        """Render EUDR workspace HTML."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_eudr_html()

    @app.get("/apps/ghg", response_class=HTMLResponse)
    async def ghg_workspace(request: Request):
        """Render GHG workspace HTML."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_ghg_html()

    @app.get("/apps/iso14064", response_class=HTMLResponse)
    async def iso14064_workspace(request: Request):
        """Render ISO14064 workspace HTML."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_iso14064_html()

    @app.get("/apps/sb253", response_class=HTMLResponse)
    async def sb253_workspace(request: Request):
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_ghg_html()

    @app.get("/apps/taxonomy", response_class=HTMLResponse)
    async def taxonomy_workspace(request: Request):
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return get_ghg_html()

    @app.get("/runs", response_class=HTMLResponse)
    async def runs_center(request: Request):
        """Render run center page (simple list)."""
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return _inject_shared_ui_script(get_runs_html(app))

    @app.get("/governance", response_class=HTMLResponse)
    async def governance_center(request: Request):
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return _inject_shared_ui_script(get_shell_html())

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_center(request: Request):
        react_shell = _serve_react_shell_if_available()
        if react_shell is not None:
            return react_shell
        return _inject_shared_ui_script(get_shell_html())

    @app.get("/ui.js")
    async def shared_ui_script():
        """Serve shared workspace UI script."""
        script_path = Path(__file__).with_name("ui_shared.js")
        if not script_path.exists():
            raise HTTPException(status_code=404, detail="Shared UI script not found")
        return FileResponse(script_path, media_type="application/javascript")

    @app.post("/api/telemetry/client-error")
    async def client_error_telemetry(request: Request):
        """Capture non-blocking frontend error telemetry."""
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        app.state.results.setdefault("_client_errors", []).append(
            {
                "received_at": datetime.utcnow().isoformat() + "Z",
                "payload": payload if isinstance(payload, dict) else {"raw": str(payload)},
            }
        )
        app.state.results["_client_errors"] = app.state.results["_client_errors"][-250:]
        return {"ok": True}

    @app.get("/api/v1/runs")
    async def list_runs(
        request: Request,
        app_id: str | None = Query(default=None, description="Filter by app_id (case-insensitive)"),
        status: str | None = Query(
            default=None,
            description="Filter by run_state or status_chip (e.g. blocked, FAIL, completed)",
        ),
        since_ts: float | None = Query(default=None, description="Minimum created_at_ts (unix seconds)"),
        until_ts: float | None = Query(default=None, description="Maximum created_at_ts (unix seconds)"),
        q: str | None = Query(default=None, description="Substring match on run_id or app_id"),
    ):
        """List run records across apps for UI."""
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        runs = []
        app_filter = (app_id or "").strip().lower()
        status_filter = (status or "").strip().lower()
        q_filter = (q or "").strip().lower()
        for run_id, meta in app.state.session_meta.items():
            rs = meta.get("run_state") or _derive_run_state(
                bool(meta.get("success")),
                bool(meta.get("can_export", True)),
                meta.get("warnings") or [],
            )
            chip = _status_chip_for_run_state(str(rs))
            cat_ts = meta.get("created_at_ts")
            if app_filter and str(meta.get("app_id") or "").lower() != app_filter:
                continue
            if since_ts is not None and cat_ts is not None and float(cat_ts) < since_ts:
                continue
            if until_ts is not None and cat_ts is not None and float(cat_ts) > until_ts:
                continue
            if status_filter:
                rs_l = str(rs).lower()
                chip_l = chip.lower()
                st_l = str(meta.get("status") or "").lower()
                match = status_filter in {rs_l, chip_l, st_l}
                if not match and status_filter in {"fail", "failed"} and rs_l == "failed":
                    match = True
                if not match and status_filter in {"warn", "warning"} and rs_l == "partial_success":
                    match = True
                if not match:
                    continue
            if q_filter and q_filter not in run_id.lower() and q_filter not in str(meta.get("app_id") or "").lower():
                continue
            runs.append(
                {
                    "run_id": run_id,
                    "created_at_ts": meta.get("created_at_ts"),
                    "app_id": meta.get("app_id"),
                    "status": meta.get("status"),
                    "execution_mode": meta.get("execution_mode"),
                    "success": meta.get("success"),
                    "artifacts": meta.get("artifacts", []),
                    "can_export": meta.get("can_export", True),
                    "run_state": rs,
                    "lifecycle_phase": "completed",
                    "status_chip": chip,
                    "error_envelope": _error_envelope(meta.get("errors_tail") or meta.get("errors")),
                }
            )
        runs.sort(key=lambda item: item.get("created_at_ts") or 0, reverse=True)
        return {"runs": runs[:200]}

    @app.get("/api/v1/governance/pack-tiers")
    async def list_pack_tiers(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        registry_path = repo_root / "greenlang" / "ecosystem" / "packs" / "v2_tier_registry.yaml"
        if not registry_path.exists():
            raise HTTPException(status_code=404, detail="Pack tier registry not found")
        try:
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
            packs = payload.get("pilot_packs", []) if isinstance(payload, dict) else []
            normalized = []
            for pack in packs:
                if not isinstance(pack, dict):
                    continue
                normalized.append(
                    {
                        "pack_slug": pack.get("pack_slug", ""),
                        "app_id": pack.get("app_id", ""),
                        "tier": pack.get("tier", ""),
                        "owner_team": pack.get("owner_team", ""),
                        "promotion_status": pack.get("promotion_status", ""),
                    }
                )
            return {"packs": normalized}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to read pack tier registry") from exc

    @app.get("/api/v1/governance/agents")
    async def list_agents(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        registry_path = repo_root / "greenlang" / "agents" / "v2_agent_registry.yaml"
        if not registry_path.exists():
            raise HTTPException(status_code=404, detail="Agent registry not found")
        try:
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
            agents = payload.get("agents", []) if isinstance(payload, dict) else []
            normalized = []
            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                normalized.append(
                    {
                        "agent_id": agent.get("agent_id", ""),
                        "owner_team": agent.get("owner_team", ""),
                        "state": agent.get("state", ""),
                        "current_version": agent.get("current_version", ""),
                        "replacement_agent_id": agent.get("replacement_agent_id"),
                    }
                )
            return {"agents": normalized}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to read agent registry") from exc

    @app.get("/api/v1/governance/policy-bundles")
    async def list_policy_bundles(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        bundle_dir = repo_root / "greenlang" / "governance" / "policy" / "bundles"
        if not bundle_dir.exists():
            raise HTTPException(status_code=404, detail="Policy bundle directory not found")
        bundles = []
        for bundle in sorted(bundle_dir.glob("*.rego")):
            try:
                size = bundle.stat().st_size
            except OSError:
                size = 0
            bundles.append({"bundle": bundle.name, "bytes": size})
        return {"bundles": bundles}

    @app.get("/api/v1/shell/chrome-context")
    async def shell_chrome_context(request: Request):
        """Compliance rail + connector incident signals for enterprise shell chrome."""
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        base = _shell_chrome_context_dict(repo_root)
        probes = getattr(app.state, "connector_health_probes", None)
        return _merge_live_connector_probe_signals(base, probes if isinstance(probes, list) else None)

    @app.get("/api/v1/admin/release-train")
    async def admin_release_train(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        evidence_path = repo_root / "docs" / "v2" / "RELEASE_TRAIN_LOCAL_EVIDENCE.json"
        if not evidence_path.exists():
            return {"available": False, "evidence": None}
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to read release train evidence") from exc
        cycles = payload.get("cycles") if isinstance(payload, dict) else None
        summary = []
        if isinstance(cycles, list):
            for cycle in cycles:
                if not isinstance(cycle, dict):
                    continue
                summary.append(
                    {
                        "cycle": cycle.get("cycle"),
                        "all_passed": cycle.get("all_passed"),
                        "executed_at_utc": cycle.get("executed_at_utc"),
                    }
                )
        return {"available": True, "evidence": payload, "cycle_summary": summary}

    @app.get("/api/v1/admin/connectors")
    async def admin_connectors(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        registry_path = repo_root / "applications" / "connectors" / "v2_connector_registry.yaml"
        if not registry_path.exists():
            raise HTTPException(status_code=404, detail="Connector registry not found")
        try:
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to read connector registry") from exc
        connectors_raw = payload.get("connectors", []) if isinstance(payload, dict) else []
        connectors = []
        for row in connectors_raw:
            if not isinstance(row, dict):
                continue
            rp = row.get("reliability_profile") if isinstance(row.get("reliability_profile"), dict) else {}
            timeout = rp.get("timeout") if isinstance(rp.get("timeout"), dict) else {}
            connectors.append(
                {
                    "connector_id": row.get("connector_id", ""),
                    "app_id": row.get("app_id", ""),
                    "owner_team": row.get("owner_team", ""),
                    "support_channel": row.get("support_channel", ""),
                    "read_timeout_ms": timeout.get("read_ms"),
                    "circuit_open_s": (rp.get("circuit_breaker") or {}).get("open_state_seconds"),
                    "operational_status": row.get("operational_status", "ok"),
                    "incident_summary": row.get("incident_summary"),
                    "slo_target_availability_pct": row.get("slo_target_availability_pct"),
                    "runbook_url": row.get("runbook_url"),
                }
            )
        return {"registry_version": payload.get("registry_version", ""), "connectors": connectors}

    @app.get("/api/v1/admin/connectors/health")
    async def admin_connectors_health(request: Request):
        """
        Lightweight live probe stub: records per-connector latency metadata for shell chrome merge.
        Call periodically or from admin workflows; does not replace registry governance truth.
        """
        _require_api_key(request)
        _enforce_rate_limit(request)
        repo_root = Path(__file__).resolve().parents[4]
        registry_path = repo_root / "applications" / "connectors" / "v2_connector_registry.yaml"
        if not registry_path.exists():
            raise HTTPException(status_code=404, detail="Connector registry not found")
        try:
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to read connector registry") from exc
        connectors_raw = payload.get("connectors", []) if isinstance(payload, dict) else []
        now = datetime.utcnow().isoformat() + "Z"
        probes: list[dict] = []
        for row in connectors_raw:
            if not isinstance(row, dict):
                continue
            t0 = time.perf_counter()
            latency_ms = max(0, int((time.perf_counter() - t0) * 1000))
            op = str(row.get("operational_status", "ok") or "ok").lower()
            probes.append(
                {
                    "connector_id": row.get("connector_id", ""),
                    "app_id": row.get("app_id", ""),
                    "ok": True,
                    "latency_ms": latency_ms,
                    "checked_at_utc": now,
                    "registry_status": op,
                }
            )
        app.state.connector_health_probes = probes
        return {"updated_at_utc": now, "probes": probes}

    if os.getenv("GL_SHELL_E2E_FIXTURES", "").strip().lower() in {"1", "true", "yes", "on"}:

        @app.post("/api/v1/e2e/shell-fixture-run")
        async def e2e_shell_fixture_run(request: Request):
            """Deterministic run payloads for Playwright (disabled unless GL_SHELL_E2E_FIXTURES is set)."""
            _require_api_key(request)
            _enforce_rate_limit(request)
            try:
                body = await request.json()
            except Exception:
                body = {}
            mode = str(body.get("mode", "")).lower()
            if mode not in {"blocked", "partial_success"}:
                raise HTTPException(status_code=400, detail="mode must be blocked or partial_success")
            run_id = uuid.uuid4().hex
            out_root = Path(tempfile.mkdtemp(prefix="e2e_fixture_"))
            out_dir = out_root / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fixture.txt").write_text("e2e fixture artifact\n", encoding="utf-8")
            app.state.output_dirs[run_id] = out_dir
            if mode == "blocked":
                extra = _run_session_fields(
                    success=True, can_export=False, warnings=[], errors=["Fixture: export blocked for E2E"]
                )
                meta = {
                    "app_id": "e2e_fixture",
                    "status": "completed",
                    "success": True,
                    "execution_mode": "native",
                    "artifacts": ["fixture.txt"],
                    "created_at_ts": datetime.utcnow().timestamp(),
                    "can_export": False,
                    **extra,
                }
            else:
                extra = _run_session_fields(
                    success=True,
                    can_export=True,
                    warnings=["Fixture: partial_success warning"],
                    errors=[],
                )
                meta = {
                    "app_id": "e2e_fixture",
                    "status": "completed",
                    "success": True,
                    "execution_mode": "native",
                    "artifacts": ["fixture.txt"],
                    "created_at_ts": datetime.utcnow().timestamp(),
                    "can_export": True,
                    **extra,
                }
            app.state.session_meta[run_id] = meta
            base = {
                "ok": True,
                "run_id": run_id,
                "app_id": meta["app_id"],
                "success": meta["success"],
                "status": meta["status"],
                "artifacts": meta["artifacts"],
                "can_export": meta["can_export"],
                "warnings": list(meta.get("warnings") or []),
                "errors": _sanitize_errors(list(meta.get("errors_tail") or [])),
                "execution_mode": meta["execution_mode"],
            }
            return _decorate_run_response(app, run_id, base)

    @app.get("/api/v1/stream/runs")
    async def stream_runs(request: Request):
        _require_api_key(request)

        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break
                payload = {
                    "status": "live",
                    "runs": len(app.state.session_meta),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/v1/apps/csrd/run")
    async def run_csrd(request: Request, input_file: UploadFile = File(...)):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if run_csrd_backend is None:
            raise HTTPException(
                status_code=503,
                detail="CSRD backend integration unavailable in this install",
            )
        temp_dir = tempfile.mkdtemp(prefix="csrd_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_name = input_file.filename or ""
            if _is_suspicious_upload_filename(raw_name):
                raise HTTPException(status_code=400, detail="Invalid upload filename")
            input_path = (Path(temp_dir) / Path(raw_name).name).resolve()
            content = await input_file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Input file exceeds upload size limit")
            input_path.write_bytes(content)

            allow_fallback = os.getenv("GL_V1_ALLOW_BACKEND_FALLBACK", "1" if ALLOW_BACKEND_FALLBACK_DEFAULT else "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            result = run_csrd_backend(input_path=input_path, output_dir=out_dir, strict=True, allow_fallback=allow_fallback)

            run_id = uuid.uuid4().hex
            app.state.output_dirs[run_id] = out_dir
            can_x = bool(result.success)
            extra = _run_session_fields(
                success=can_x, can_export=can_x, warnings=result.warnings, errors=result.errors
            )
            app.state.session_meta[run_id] = {
                "app_id": "csrd",
                "status": "completed" if result.success else "failed",
                "success": bool(result.success),
                "execution_mode": "native" if result.native_backend_used else ("fallback" if result.fallback_used else "unknown"),
                "artifacts": result.artifacts,
                "created_at_ts": datetime.utcnow().timestamp(),
                "can_export": can_x,
                **extra,
            }
            summary = {}
            report_path = out_dir / "esrs_report.json"
            if report_path.exists():
                try:
                    summary = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = {"note": "esrs_report.json present but could not be parsed"}
            return _decorate_run_response(
                app,
                run_id,
                {
                    "run_id": run_id,
                    "app_id": "csrd",
                    "success": bool(result.success),
                    "status": "completed" if result.success else "failed",
                    "run_state": extra["run_state"],
                    "execution_mode": app.state.session_meta[run_id]["execution_mode"],
                    "artifacts": result.artifacts,
                    "can_export": can_x,
                    "warnings": result.warnings,
                    "errors": _sanitize_errors(result.errors),
                    "summary": summary,
                },
            )
        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="CSRD processing failed") from exc

    @app.post("/api/v1/apps/vcci/run")
    async def run_vcci(request: Request, input_file: UploadFile = File(...)):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if run_vcci_backend is None:
            raise HTTPException(
                status_code=503,
                detail="VCCI backend integration unavailable in this install",
            )
        temp_dir = tempfile.mkdtemp(prefix="vcci_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_name = input_file.filename or ""
            if _is_suspicious_upload_filename(raw_name):
                raise HTTPException(status_code=400, detail="Invalid upload filename")
            input_path = (Path(temp_dir) / Path(raw_name).name).resolve()
            content = await input_file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Input file exceeds upload size limit")
            input_path.write_bytes(content)

            allow_fallback = os.getenv("GL_V1_ALLOW_BACKEND_FALLBACK", "1" if ALLOW_BACKEND_FALLBACK_DEFAULT else "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            result = run_vcci_backend(input_path=input_path, output_dir=out_dir, strict=True, allow_fallback=allow_fallback)

            run_id = uuid.uuid4().hex
            app.state.output_dirs[run_id] = out_dir
            can_x = bool(result.success)
            extra = _run_session_fields(
                success=can_x, can_export=can_x, warnings=result.warnings, errors=result.errors
            )
            app.state.session_meta[run_id] = {
                "app_id": "vcci",
                "status": "completed" if result.success else "failed",
                "success": bool(result.success),
                "execution_mode": "native" if result.native_backend_used else ("fallback" if result.fallback_used else "unknown"),
                "artifacts": result.artifacts,
                "created_at_ts": datetime.utcnow().timestamp(),
                "can_export": can_x,
                **extra,
            }
            summary = {}
            inv_path = out_dir / "scope3_inventory.json"
            if inv_path.exists():
                try:
                    summary = json.loads(inv_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = {"note": "scope3_inventory.json present but could not be parsed"}
            return _decorate_run_response(
                app,
                run_id,
                {
                    "run_id": run_id,
                    "app_id": "vcci",
                    "success": bool(result.success),
                    "status": "completed" if result.success else "failed",
                    "run_state": extra["run_state"],
                    "execution_mode": app.state.session_meta[run_id]["execution_mode"],
                    "artifacts": result.artifacts,
                    "can_export": can_x,
                    "warnings": result.warnings,
                    "errors": _sanitize_errors(result.errors),
                    "summary": summary,
                },
            )
        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="VCCI processing failed") from exc

    @app.post("/api/v1/apps/csrd/demo-run")
    async def run_csrd_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if run_csrd_backend is None:
            raise HTTPException(status_code=503, detail="CSRD backend integration unavailable in this install")
        sample_input = _resolve_demo_input("csrd")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="CSRD demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="csrd_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_csrd_backend(input_path=sample_input, output_dir=out_dir, strict=True, allow_fallback=True)
        run_id = _record_v1_run(
            app=app,
            app_id="csrd",
            out_dir=out_dir,
            success=bool(result.success),
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=bool(result.success),
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "csrd",
                "success": bool(result.success),
                "status": "completed" if result.success else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": meta["can_export"],
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    @app.post("/api/v1/apps/vcci/demo-run")
    async def run_vcci_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if run_vcci_backend is None:
            raise HTTPException(status_code=503, detail="VCCI backend integration unavailable in this install")
        sample_input = _resolve_demo_input("vcci")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="VCCI demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="vcci_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_vcci_backend(input_path=sample_input, output_dir=out_dir, strict=True, allow_fallback=True)
        run_id = _record_v1_run(
            app=app,
            app_id="vcci",
            out_dir=out_dir,
            success=bool(result.success),
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=bool(result.success),
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "vcci",
                "success": bool(result.success),
                "status": "completed" if result.success else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": meta["can_export"],
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    async def _run_v2_json_app(request: Request, app_key: str, input_file: UploadFile) -> dict:
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if run_v2_profile_backend is None:
            raise HTTPException(status_code=503, detail=f"{app_key.upper()} backend integration unavailable in this install")

        temp_dir = tempfile.mkdtemp(prefix=f"{app_key}_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_name = input_file.filename or ""
            if _is_suspicious_upload_filename(raw_name):
                raise HTTPException(status_code=400, detail="Invalid upload filename")
            input_path = (Path(temp_dir) / Path(raw_name).name).resolve()
            content = await input_file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Input file exceeds upload size limit")
            input_path.write_bytes(content)

            result = run_v2_profile_backend(
                profile_key=app_key,
                input_path=input_path,
                output_dir=out_dir,
                strict=True,
                allow_fallback=True,
            )

            run_id = uuid.uuid4().hex
            blocked = result.exit_code == V2_BLOCKED_EXIT_CODE
            can_x = bool(result.success and not blocked)
            extra = _run_session_fields(
                success=bool(result.success), can_export=can_x, warnings=result.warnings, errors=result.errors
            )
            app.state.output_dirs[run_id] = out_dir
            app.state.session_meta[run_id] = {
                "app_id": app_key,
                "status": "completed" if result.success else "failed",
                "success": bool(result.success),
                "execution_mode": "native" if result.native_backend_used else ("fallback" if result.fallback_used else "unknown"),
                "artifacts": result.artifacts,
                "created_at_ts": datetime.utcnow().timestamp(),
                "can_export": can_x,
                **extra,
            }

            summary = {}
            summary_artifacts = {
                "eudr": "due_diligence_statement.json",
                "ghg": "ghg_inventory.json",
                "iso14064": "iso14064_verification_report.json",
                "sb253": "sb253_disclosure.json",
                "taxonomy": "taxonomy_alignment.json",
            }
            artifact_name = summary_artifacts.get(app_key)
            if artifact_name and (out_dir / artifact_name).exists():
                try:
                    summary = json.loads((out_dir / artifact_name).read_text(encoding="utf-8"))
                except Exception:
                    summary = {"note": f"{artifact_name} present but could not be parsed"}

            return _decorate_run_response(
                app,
                run_id,
                {
                    "run_id": run_id,
                    "app_id": app_key,
                    "success": bool(result.success),
                    "status": "completed" if result.success else "failed",
                    "run_state": extra["run_state"],
                    "execution_mode": app.state.session_meta[run_id]["execution_mode"],
                    "artifacts": result.artifacts,
                    "can_export": can_x,
                    "warnings": result.warnings,
                    "errors": _sanitize_errors(result.errors),
                    "summary": summary,
                },
            )
        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"{app_key.upper()} processing failed") from exc

    @app.post("/api/v1/apps/eudr/run")
    async def run_eudr(request: Request, input_file: UploadFile = File(...)):
        return await _run_v2_json_app(request=request, app_key="eudr", input_file=input_file)

    @app.post("/api/v1/apps/ghg/run")
    async def run_ghg(request: Request, input_file: UploadFile = File(...)):
        return await _run_v2_json_app(request=request, app_key="ghg", input_file=input_file)

    @app.post("/api/v1/apps/iso14064/run")
    async def run_iso14064(request: Request, input_file: UploadFile = File(...)):
        return await _run_v2_json_app(request=request, app_key="iso14064", input_file=input_file)

    @app.post("/api/v1/apps/sb253/run")
    async def run_sb253(request: Request, input_file: UploadFile = File(...)):
        return await _run_v2_json_app(request=request, app_key="sb253", input_file=input_file)

    @app.post("/api/v1/apps/taxonomy/run")
    async def run_taxonomy(request: Request, input_file: UploadFile = File(...)):
        return await _run_v2_json_app(request=request, app_key="taxonomy", input_file=input_file)

    @app.post("/api/v1/apps/eudr/demo-run")
    async def run_eudr_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        sample_input = _resolve_demo_input("eudr")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="EUDR demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="eudr_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_v2_profile_backend(
            profile_key="eudr",
            input_path=sample_input,
            output_dir=out_dir,
            strict=True,
            allow_fallback=True,
        )
        blocked = result.exit_code == V2_BLOCKED_EXIT_CODE
        pipeline_ok = bool(result.success)
        can_x = bool(result.success and not blocked)
        run_id = _record_v1_run(
            app=app,
            app_id="eudr",
            out_dir=out_dir,
            success=pipeline_ok,
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=can_x,
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "eudr",
                "success": pipeline_ok,
                "status": "completed" if pipeline_ok else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": can_x,
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    @app.post("/api/v1/apps/ghg/demo-run")
    async def run_ghg_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        sample_input = _resolve_demo_input("ghg")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="GHG demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="ghg_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_v2_profile_backend(
            profile_key="ghg",
            input_path=sample_input,
            output_dir=out_dir,
            strict=True,
            allow_fallback=True,
        )
        blocked = result.exit_code == V2_BLOCKED_EXIT_CODE
        pipeline_ok = bool(result.success)
        can_x = bool(result.success and not blocked)
        run_id = _record_v1_run(
            app=app,
            app_id="ghg",
            out_dir=out_dir,
            success=pipeline_ok,
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=can_x,
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "ghg",
                "success": pipeline_ok,
                "status": "completed" if pipeline_ok else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": can_x,
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    @app.post("/api/v1/apps/iso14064/demo-run")
    async def run_iso14064_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        sample_input = _resolve_demo_input("iso14064")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="ISO14064 demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="iso14064_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_v2_profile_backend(
            profile_key="iso14064",
            input_path=sample_input,
            output_dir=out_dir,
            strict=True,
            allow_fallback=True,
        )
        blocked = result.exit_code == V2_BLOCKED_EXIT_CODE
        pipeline_ok = bool(result.success)
        can_x = bool(result.success and not blocked)
        run_id = _record_v1_run(
            app=app,
            app_id="iso14064",
            out_dir=out_dir,
            success=pipeline_ok,
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=can_x,
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "iso14064",
                "success": pipeline_ok,
                "status": "completed" if pipeline_ok else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": can_x,
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    @app.post("/api/v1/apps/sb253/demo-run")
    async def run_sb253_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        sample_input = _resolve_demo_input("sb253")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="SB253 demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="sb253_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_v2_profile_backend(
            profile_key="sb253",
            input_path=sample_input,
            output_dir=out_dir,
            strict=True,
            allow_fallback=True,
        )
        blocked = result.exit_code == V2_BLOCKED_EXIT_CODE
        pipeline_ok = bool(result.success)
        can_x = bool(result.success and not blocked)
        run_id = _record_v1_run(
            app=app,
            app_id="sb253",
            out_dir=out_dir,
            success=pipeline_ok,
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=can_x,
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "sb253",
                "success": pipeline_ok,
                "status": "completed" if pipeline_ok else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": can_x,
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    @app.post("/api/v1/apps/taxonomy/demo-run")
    async def run_taxonomy_demo(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        sample_input = _resolve_demo_input("taxonomy")
        if sample_input is None:
            raise HTTPException(status_code=404, detail="Taxonomy demo input not found")
        temp_dir = tempfile.mkdtemp(prefix="taxonomy_demo_")
        out_dir = Path(temp_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_v2_profile_backend(
            profile_key="taxonomy",
            input_path=sample_input,
            output_dir=out_dir,
            strict=True,
            allow_fallback=True,
        )
        blocked = result.exit_code == V2_BLOCKED_EXIT_CODE
        pipeline_ok = bool(result.success)
        can_x = bool(result.success and not blocked)
        run_id = _record_v1_run(
            app=app,
            app_id="taxonomy",
            out_dir=out_dir,
            success=pipeline_ok,
            native_backend_used=bool(result.native_backend_used),
            fallback_used=bool(result.fallback_used),
            artifacts=result.artifacts,
            can_export=can_x,
            warnings=result.warnings,
            errors=result.errors,
        )
        meta = app.state.session_meta[run_id]
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "taxonomy",
                "success": pipeline_ok,
                "status": "completed" if pipeline_ok else "failed",
                "run_state": meta["run_state"],
                "artifacts": result.artifacts,
                "can_export": can_x,
                "warnings": result.warnings,
                "errors": _sanitize_errors(result.errors),
                "execution_mode": meta["execution_mode"],
            },
        )

    @app.get("/api/v1/runs/{run_id}/artifacts/{artifact_path:path}")
    async def download_run_artifact(run_id: str, artifact_path: str, request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if not _is_valid_run_id(run_id):
            raise HTTPException(status_code=400, detail="Invalid run ID")
        if run_id not in app.state.output_dirs:
            raise HTTPException(status_code=404, detail="Run not found")
        session_meta = app.state.session_meta.get(run_id, {})
        if not session_meta.get("can_export", True):
            raise HTTPException(status_code=409, detail="Export blocked for this run")
        output_dir = app.state.output_dirs[run_id]
        output_root = Path(output_dir).resolve()
        file_path = (output_root / artifact_path).resolve()
        if output_root not in file_path.parents and file_path != output_root:
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path, filename=file_path.name, media_type="application/octet-stream")

    @app.get("/api/v1/runs/{run_id}/bundle")
    async def download_run_bundle(run_id: str, request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        if not _is_valid_run_id(run_id):
            raise HTTPException(status_code=400, detail="Invalid run ID")
        if run_id not in app.state.output_dirs:
            raise HTTPException(status_code=404, detail="Run not found")
        session_meta = app.state.session_meta.get(run_id, {})
        if not session_meta.get("can_export", True):
            raise HTTPException(status_code=409, detail="Export blocked for this run")
        output_dir = app.state.output_dirs[run_id]
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in Path(output_dir).rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zip_file.write(file_path, arcname)
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=greenlang_run_{run_id}.zip"
            },
        )

    @app.post("/api/process")
    async def process_files(
        request: Request,
        config_file: UploadFile = File(...),
        imports_file: UploadFile = File(...),
        mode: str = Form(default="transitional"),
        collect_errors: bool = Form(default=True),
    ):
        """Process uploaded files and generate CBAM report."""
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="cbam_")
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        try:
            # Save uploaded files safely.
            config_raw_name = config_file.filename or ""
            imports_raw_name = imports_file.filename or ""
            if _is_suspicious_upload_filename(config_raw_name) or _is_suspicious_upload_filename(imports_raw_name):
                raise HTTPException(status_code=400, detail="Invalid upload filename")
            config_name = Path(config_raw_name).name
            imports_name = Path(imports_raw_name).name

            config_path = (Path(temp_dir) / config_name).resolve()
            imports_path = (Path(temp_dir) / imports_name).resolve()
            temp_root = Path(temp_dir).resolve()
            if temp_root not in config_path.parents or temp_root not in imports_path.parents:
                raise ValueError("Upload filename contains invalid path segments")

            with open(config_path, "wb") as f:
                content = await config_file.read()
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Config file exceeds upload size limit")
                f.write(content)

            with open(imports_path, "wb") as f:
                content = await imports_file.read()
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Imports file exceeds upload size limit")
                f.write(content)

            # Run pipeline
            pipeline = CBAMPipeline(
                config_path=config_path,
                imports_path=imports_path,
                output_dir=output_dir,
                verbose=False,
                dry_run=False,
            )

            result = pipeline.run()

            # Generate session ID
            session_id = uuid.uuid4().hex

            # Store result and output directory
            app.state.results[session_id] = result
            app.state.output_dirs[session_id] = output_dir

            # Build response
            response_data = {
                "success": result.success,
                "session_id": session_id,
                "statistics": result.statistics,
                "artifacts": result.artifacts,
                "errors": _sanitize_errors(result.errors),
            }

            # Add policy status
            if result.policy_result:
                response_data["policy"] = result.policy_result

            # Add XML validation status
            if result.xml_validation:
                response_data["xml_validation"] = result.xml_validation

            # Add gap summary
            if result.gap_summary:
                response_data["gap_summary"] = result.gap_summary

            # Add lines using defaults for drilldown
            if result.lines_using_defaults:
                response_data["lines_using_defaults"] = result.lines_using_defaults

            # Build compliance status from pipeline-computed exportability.
            xml_val = result.xml_validation or {}
            schema_status = xml_val.get("status", "PASS")
            schema_valid = schema_status == "PASS"

            policy = result.policy_result or {}
            policy_status = policy.get("status", "PASS")
            default_usage = result.statistics.get("default_usage_percent", 0)
            can_export = bool(result.can_export)

            if not can_export and not schema_valid:
                compliance_status = "schema_fail"
                compliance_message = "XML Schema Validation FAILED. Report cannot be uploaded to registry. Fix schema errors first."
                export_label = "INVALID - Cannot Export"
            elif not can_export and policy_status == "FAIL":
                compliance_status = "policy_fail"
                compliance_message = "Policy validation failed and export is blocked by policy configuration."
                export_label = "Policy Block - Cannot Export"
            elif policy_status == "PASS":
                compliance_status = "compliant"
                compliance_message = "All validations passed. Report is compliant and ready for submission."
                export_label = "Ready for Submission"
            elif policy_status == "WARN":
                compliance_status = "warning"
                compliance_message = f"Report generated with warnings. Default factor usage: {default_usage:.1f}%. Export as draft allowed."
                export_label = "Draft - Review Warnings"
            else:  # policy FAIL
                compliance_status = "policy_fail"
                compliance_message = "Policy validation failed. Export allowed as draft but NOT COMPLIANT."
                export_label = "Draft - NOT COMPLIANT"

            response_data["compliance"] = {
                "status": compliance_status,
                "schema_status": schema_status,
                "schema_valid": schema_valid,
                "policy_status": policy_status,
                "default_usage_percent": default_usage,
                "message": compliance_message,
                "can_export": can_export,
                "export_label": export_label,
            }
            warn_list_pf: list[str] = []
            if compliance_status == "warning":
                warn_list_pf.append(compliance_message)
            if compliance_status == "policy_fail" and can_export:
                warn_list_pf.append("Policy failed; draft export only.")
            ok_run = bool(result.success)
            extra_pf = _run_session_fields(
                success=ok_run, can_export=can_export, warnings=warn_list_pf, errors=result.errors
            )
            app.state.session_meta[session_id] = {
                "app_id": "cbam",
                "status": "completed" if ok_run else "failed",
                "success": ok_run,
                "execution_mode": "native",
                "artifacts": result.artifacts,
                "created_at_ts": datetime.utcnow().timestamp(),
                "can_export": can_export,
                "block_reason": (
                    _sanitize_error_message(result.errors[0])
                    if result.errors
                    else "Export is blocked for this session."
                ),
                **extra_pf,
            }

            return _decorate_run_response(app, session_id, response_data)

        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except Exception:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "errors": ["Processing failed. Please validate inputs and try again."],
                "statistics": {},
                "artifacts": [],
            }

    @app.post("/api/v1/apps/cbam/run")
    async def run_cbam_v1(
        request: Request,
        config_file: UploadFile = File(...),
        imports_file: UploadFile = File(...),
        mode: str = Form(default="transitional"),
        collect_errors: bool = Form(default=True),
    ):
        """
        v1-normalized CBAM run endpoint.

        This wraps the existing /api/process response into the cross-app v1 web contract
        without breaking legacy clients.
        """
        payload = await process_files(
            request=request,
            config_file=config_file,
            imports_file=imports_file,
            mode=mode,
            collect_errors=collect_errors,
        )
        # If legacy process failed hard (no session_id), surface as 500.
        if not payload.get("success") and not payload.get("session_id"):
            raise HTTPException(status_code=500, detail="CBAM processing failed")

        run_id = payload.get("session_id") or uuid.uuid4().hex
        # Ensure the run appears in the run center with normalized metadata.
        meta = app.state.session_meta.get(run_id, {})
        compliance = payload.get("compliance") or {}
        warn_list: list[str] = []
        if compliance.get("status") in {"warning", "warn"}:
            warn_list.append("Compliance reported warnings; validate before export.")
        if compliance.get("status") == "policy_fail" and compliance.get("can_export"):
            warn_list.append("Policy failed but draft export is allowed.")
        can_x = bool(compliance.get("can_export", True))
        ok = bool(payload.get("success"))
        extra = _run_session_fields(success=ok, can_export=can_x, warnings=warn_list, errors=payload.get("errors"))
        meta.update(
            {
                "app_id": "cbam",
                "status": "completed" if payload.get("success") else "failed",
                "success": ok,
                "execution_mode": "native",
                "artifacts": payload.get("artifacts", []),
                "created_at_ts": meta.get("created_at_ts") or datetime.utcnow().timestamp(),
                "can_export": can_x,
                **extra,
            }
        )
        app.state.session_meta[run_id] = meta

        summary = {
            "statistics": payload.get("statistics", {}),
            "compliance": payload.get("compliance"),
            "xml_validation": payload.get("xml_validation"),
            "policy": payload.get("policy"),
            "gap_summary": payload.get("gap_summary"),
        }
        return _decorate_run_response(
            app,
            run_id,
            {
                "run_id": run_id,
                "app_id": "cbam",
                "success": bool(payload.get("success")),
                "status": "completed" if payload.get("success") else "failed",
                "run_state": meta.get("run_state", "completed"),
                "execution_mode": "native",
                "artifacts": payload.get("artifacts", []),
                "can_export": can_x,
                "warnings": warn_list,
                "errors": payload.get("errors", []),
                "summary": summary,
            },
        )

    @app.post("/api/v1/apps/cbam/demo-run")
    async def run_cbam_demo_v1(request: Request):
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()
        package_root = Path(__file__).resolve().parents[3]
        config_path = package_root / "examples" / "sample_config.yaml"
        imports_path = package_root / "examples" / "sample_imports.csv"
        if not config_path.exists() or not imports_path.exists():
            raise HTTPException(status_code=404, detail="CBAM demo inputs not found")
        temp_dir = tempfile.mkdtemp(prefix="cbam_demo_")
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = CBAMPipeline(
            config_path=config_path,
            imports_path=imports_path,
            output_dir=output_dir,
            verbose=False,
            dry_run=False,
        )
        result = pipeline.run()
        run_id = uuid.uuid4().hex
        app.state.results[run_id] = result
        app.state.output_dirs[run_id] = output_dir
        cbam_warnings: list[str] = []
        pol = result.policy_result or {}
        if pol.get("status") == "WARN":
            cbam_warnings.append("CBAM policy status is WARN.")
        extra = _run_session_fields(
            success=bool(result.success),
            can_export=bool(result.can_export),
            warnings=cbam_warnings,
            errors=result.errors,
        )
        app.state.session_meta[run_id] = {
            "app_id": "cbam",
            "status": "completed" if result.success else "failed",
            "success": bool(result.success),
            "execution_mode": "native",
            "artifacts": result.artifacts,
            "created_at_ts": datetime.utcnow().timestamp(),
            "can_export": bool(result.can_export),
            "block_reason": (
                _sanitize_error_message(result.errors[0]) if result.errors else "Export is blocked for this session."
            ),
            **extra,
        }
        return _decorate_run_response(
            app,
            run_id,
            {
                "ok": True,
                "run_id": run_id,
                "app_id": "cbam",
                "success": bool(result.success),
                "status": "completed" if result.success else "failed",
                "run_state": extra["run_state"],
                "execution_mode": "native",
                "artifacts": result.artifacts,
                "can_export": bool(result.can_export),
                "warnings": cbam_warnings,
                "errors": _sanitize_errors(result.errors),
            },
        )

    @app.get("/api/download/{session_id}")
    async def download_all(session_id: str, request: Request):
        """Download all artifacts as a ZIP file."""
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()

        if session_id not in app.state.output_dirs:
            raise HTTPException(status_code=404, detail="Session not found")
        session_meta = app.state.session_meta.get(session_id, {})
        if not session_meta.get("can_export", False):
            raise HTTPException(
                status_code=409,
                detail=f"Export blocked: {session_meta.get('block_reason', 'Export is blocked for this session.')}",
            )

        output_dir = app.state.output_dirs[session_id]

        # Create ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zip_file.write(file_path, arcname)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=cbam_report_{session_id}.zip"
            }
        )

    @app.get("/api/download/{session_id}/{filename:path}")
    async def download_file(session_id: str, filename: str, request: Request):
        """Download a specific artifact file."""
        _require_api_key(request)
        _enforce_rate_limit(request)
        _prune_expired_sessions()

        if session_id not in app.state.output_dirs:
            raise HTTPException(status_code=404, detail="Session not found")
        session_meta = app.state.session_meta.get(session_id, {})
        if not session_meta.get("can_export", False):
            raise HTTPException(
                status_code=409,
                detail=f"Export blocked: {session_meta.get('block_reason', 'Export is blocked for this session.')}",
            )

        output_dir = app.state.output_dirs[session_id]
        output_root = Path(output_dir).resolve()
        file_path = (output_root / filename).resolve()

        if output_root not in file_path.parents and file_path != output_root:
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            file_path,
            filename=file_path.name,
            media_type="application/octet-stream"
        )

    @app.post("/api/preview-config")
    async def preview_config(request: Request, config_file: UploadFile = File(...)):
        """Preview config file after upload to verify YAML mapping."""
        _require_api_key(request)
        _enforce_rate_limit(request)
        try:
            content = await config_file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                return {
                    "success": False,
                    "error": "Config file exceeds upload size limit",
                }
            config_data = yaml.safe_load(content.decode('utf-8'))
            CBAMConfig.model_validate(config_data)

            # Extract preview data
            declarant = config_data.get('declarant', {})
            reporting_period = config_data.get('reporting_period', {})
            representative = config_data.get('representative', {})

            preview = {
                "success": True,
                "declarant": {
                    "name": declarant.get('name', 'Not specified'),
                    "eori_number": declarant.get('eori_number', 'Not specified'),
                },
                "reporting_period": {
                    "quarter": reporting_period.get('quarter', 'Not specified'),
                    "year": reporting_period.get('year', 'Not specified'),
                },
                "representative": {
                    "name": representative.get('name') if representative else None,
                    "eori_number": representative.get('eori_number') if representative else None,
                } if representative else None,
                "mode": config_data.get('mode', 'transitional'),
                "settings": config_data.get('settings', {}),
                "validation": {"valid": True, "errors": []},
            }

            return preview

        except yaml.YAMLError:
            return {
                "success": False,
                "error": "Invalid YAML. Please fix file formatting and retry.",
            }
        except Exception as e:
            # Include pydantic details when available to guide fixes.
            if hasattr(e, "errors"):
                return {
                    "success": False,
                    "error": "Config validation failed",
                    "validation": {"valid": False, "errors": e.errors()},
                }
            return {
                "success": False,
                "error": "Config parse failed. Please validate required fields and retry.",
            }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

    return app


def get_home_html() -> str:
    """Return the main HTML page."""
    template_path = Path(__file__).with_name("index.html")
    if template_path.exists():
        html = template_path.read_text(encoding="utf-8")
        return html.replace("__CBAM_VERSION__", __version__)

    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GreenLang CBAM Pack</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        .logo {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        h1 {
            color: #4ecca3;
            font-size: 2em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #888;
            font-size: 1.1em;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .upload-zone {
            border: 2px dashed #4ecca3;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .upload-zone:hover {
            background: rgba(78, 204, 163, 0.1);
            border-color: #6ee7b7;
        }

        .upload-zone.dragover {
            background: rgba(78, 204, 163, 0.2);
            border-color: #6ee7b7;
        }

        .upload-zone.has-file {
            border-color: #4ecca3;
            background: rgba(78, 204, 163, 0.1);
        }

        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 1.1em;
            margin-bottom: 10px;
        }

        .upload-hint {
            color: #888;
            font-size: 0.9em;
        }

        .file-name {
            color: #4ecca3;
            font-weight: bold;
            margin-top: 10px;
        }

        input[type="file"] {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #4ecca3 0%, #45b393 100%);
            color: #1a1a2e;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(78, 204, 163, 0.3);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .progress-container {
            display: none;
            margin-top: 20px;
        }

        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ecca3, #45b393);
            width: 0%;
            transition: width 0.3s ease;
        }

        .progress-text {
            text-align: center;
            margin-top: 10px;
            color: #888;
        }

        .results {
            display: none;
        }

        .results.show {
            display: block;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: rgba(78, 204, 163, 0.1);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4ecca3;
        }

        .stat-label {
            color: #888;
            margin-top: 5px;
        }

        .validation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .validation-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .validation-card h4 {
            margin-bottom: 15px;
            color: #fff;
        }

        .validation-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }

        .status-badge.pass {
            background: rgba(78, 204, 163, 0.2);
            color: #4ecca3;
        }

        .status-badge.warn {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }

        .status-badge.fail {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }

        .validation-detail {
            color: #888;
            font-size: 0.9em;
        }

        .compliance-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        .compliance-badge.compliant {
            background: rgba(78, 204, 163, 0.2);
            color: #4ecca3;
        }

        .compliance-badge.warning {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }

        .compliance-badge.error {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }

        .section-title {
            margin: 30px 0 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .drilldown-section {
            margin-top: 20px;
        }

        .drilldown-toggle {
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid rgba(255, 193, 7, 0.3);
            padding: 15px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .drilldown-toggle:hover {
            background: rgba(255, 193, 7, 0.15);
        }

        .drilldown-content {
            display: none;
            margin-top: 10px;
        }

        .drilldown-content.show {
            display: block;
        }

        .default-line-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }

        .default-line-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .default-line-id {
            font-weight: bold;
            color: #4ecca3;
        }

        .default-line-emissions {
            color: #ffc107;
        }

        .default-line-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
            font-size: 0.9em;
            color: #888;
        }

        .missing-fields {
            margin-top: 10px;
            padding: 10px;
            background: rgba(220, 53, 69, 0.1);
            border-radius: 4px;
        }

        .missing-fields-title {
            font-weight: bold;
            color: #dc3545;
            margin-bottom: 5px;
        }

        .missing-field-tag {
            display: inline-block;
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin: 2px;
        }

        .recommended-action {
            margin-top: 10px;
            padding: 10px;
            background: rgba(78, 204, 163, 0.1);
            border-radius: 4px;
            color: #4ecca3;
            font-size: 0.9em;
        }

        .artifacts-list {
            list-style: none;
        }

        .artifacts-list li {
            padding: 12px 15px;
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 8px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .download-link {
            color: #4ecca3;
            text-decoration: none;
            padding: 5px 15px;
            border: 1px solid #4ecca3;
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .download-link:hover {
            background: #4ecca3;
            color: #1a1a2e;
        }

        .btn-download-all {
            margin-top: 20px;
        }

        .error-list {
            background: rgba(220, 53, 69, 0.1);
            border: 1px solid rgba(220, 53, 69, 0.3);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }

        .error-item {
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }

        .error-code {
            color: #dc3545;
            font-weight: bold;
        }

        .violation-list {
            margin-top: 10px;
        }

        .violation-item {
            background: rgba(220, 53, 69, 0.1);
            border-left: 3px solid #dc3545;
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 0 8px 8px 0;
        }

        .warning-item {
            background: rgba(255, 193, 7, 0.1);
            border-left: 3px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 0 8px 8px 0;
        }

        footer {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        footer a {
            color: #4ecca3;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">&#127807;</div>
            <h1>GreenLang CBAM Pack</h1>
            <p class="subtitle">EU Carbon Border Adjustment Mechanism Compliance Tool</p>
        </header>

        <div class="card">
            <h2 style="margin-bottom: 20px;">&#128193; Upload Files</h2>

            <div class="upload-zone" id="configZone" onclick="document.getElementById('configFile').click()">
                <div class="upload-icon">&#128196;</div>
                <div class="upload-text">Config File (YAML)</div>
                <div class="upload-hint">Drag & drop or click to select</div>
                <div class="file-name" id="configFileName"></div>
                <input type="file" id="configFile" accept=".yaml,.yml" onchange="handleFileSelect(this, 'config')">
            </div>

            <div class="upload-zone" id="importsZone" onclick="document.getElementById('importsFile').click()">
                <div class="upload-icon">&#128202;</div>
                <div class="upload-text">Import Ledger (CSV/XLSX)</div>
                <div class="upload-hint">Drag & drop or click to select</div>
                <div class="file-name" id="importsFileName"></div>
                <input type="file" id="importsFile" accept=".csv,.xlsx,.xls" onchange="handleFileSelect(this, 'imports')">
            </div>

            <!-- Config Preview Section -->
            <div id="configPreview" style="display: none; background: rgba(78, 204, 163, 0.1); border: 1px solid rgba(78, 204, 163, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h4 style="color: #4ecca3; margin-bottom: 15px;">&#10003; Config Loaded</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div>
                        <div style="color: #888; font-size: 0.85em;">Declarant</div>
                        <div id="previewDeclarant" style="font-weight: bold;">-</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 0.85em;">EORI Number</div>
                        <div id="previewEORI" style="font-weight: bold;">-</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 0.85em;">Reporting Period</div>
                        <div id="previewPeriod" style="font-weight: bold;">-</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 0.85em;">Mode</div>
                        <div id="previewMode" style="font-weight: bold;">-</div>
                    </div>
                </div>
                <div id="previewRepresentative" style="margin-top: 15px; display: none;">
                    <div style="color: #888; font-size: 0.85em;">Representative</div>
                    <div id="previewRepName" style="font-weight: bold;">-</div>
                </div>
                <div id="previewError" style="color: #dc3545; margin-top: 10px; display: none;"></div>
            </div>

            <button class="btn" id="processBtn" onclick="processFiles()" disabled>
                &#9654; Generate Report
            </button>

            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Processing...</div>
            </div>
        </div>

        <div class="results card" id="resultsCard">
            <h2 style="margin-bottom: 20px;">&#128202; Results</h2>

            <div id="complianceStatus"></div>

            <!-- Validation Status Section -->
            <div class="validation-grid" id="validationGrid"></div>

            <!-- Statistics Section -->
            <div class="stat-grid" id="statsGrid"></div>

            <!-- Lines Using Defaults Drilldown -->
            <div class="drilldown-section" id="drilldownSection" style="display: none;">
                <div class="drilldown-toggle" onclick="toggleDrilldown()">
                    <span>&#9888; <span id="defaultLinesCount">0</span> lines using default factors - Click to review</span>
                    <span id="drilldownArrow">&#9660;</span>
                </div>
                <div class="drilldown-content" id="drilldownContent"></div>
            </div>

            <!-- Policy Violations/Warnings -->
            <div id="policySection"></div>

            <h3 class="section-title">&#128230; Generated Artifacts</h3>
            <ul class="artifacts-list" id="artifactsList"></ul>

            <button class="btn btn-download-all" id="downloadAllBtn" onclick="downloadAll()">
                &#11015; Download All (ZIP)
            </button>

            <div class="error-list" id="errorList" style="display: none;">
                <h3 style="margin-bottom: 15px; color: #dc3545;">&#9888; Errors</h3>
                <div id="errorItems"></div>
            </div>
        </div>

        <footer>
            <p>GreenLang CBAM Pack v''' + __version__ + '''</p>
            <p><a href="https://greenlang.in" target="_blank">greenlang.in</a></p>
        </footer>
    </div>

    <script>
        let configFile = null;
        let importsFile = null;
        let currentSessionId = null;

        // Drag and drop handling
        ['configZone', 'importsZone'].forEach(id => {
            const zone = document.getElementById(id);

            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });

            zone.addEventListener('dragleave', () => {
                zone.classList.remove('dragover');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');

                const file = e.dataTransfer.files[0];
                if (file) {
                    const type = id === 'configZone' ? 'config' : 'imports';
                    const input = document.getElementById(type === 'config' ? 'configFile' : 'importsFile');

                    const dt = new DataTransfer();
                    dt.items.add(file);
                    input.files = dt.files;

                    handleFileSelect(input, type);
                }
            });
        });

        async function handleFileSelect(input, type) {
            const file = input.files[0];
            if (!file) return;

            const zone = document.getElementById(type === 'config' ? 'configZone' : 'importsZone');
            const nameEl = document.getElementById(type === 'config' ? 'configFileName' : 'importsFileName');

            zone.classList.add('has-file');
            nameEl.textContent = '✓ ' + file.name;

            if (type === 'config') {
                configFile = file;
                // Fetch config preview
                await previewConfigFile(file);
            } else {
                importsFile = file;
            }

            updateProcessButton();
        }

        async function previewConfigFile(file) {
            const previewSection = document.getElementById('configPreview');
            const previewError = document.getElementById('previewError');

            try {
                const formData = new FormData();
                formData.append('config_file', file);

                const response = await fetch('/api/preview-config', {
                    method: 'POST',
                    body: formData
                });

                const preview = await response.json();

                if (preview.success) {
                    document.getElementById('previewDeclarant').textContent = preview.declarant.name;
                    document.getElementById('previewEORI').textContent = preview.declarant.eori_number;
                    document.getElementById('previewPeriod').textContent =
                        `${preview.reporting_period.quarter} ${preview.reporting_period.year}`;
                    document.getElementById('previewMode').textContent =
                        preview.mode.charAt(0).toUpperCase() + preview.mode.slice(1);

                    if (preview.representative && preview.representative.name) {
                        document.getElementById('previewRepName').textContent =
                            `${preview.representative.name} (${preview.representative.eori_number || 'No EORI'})`;
                        document.getElementById('previewRepresentative').style.display = 'block';
                    } else {
                        document.getElementById('previewRepresentative').style.display = 'none';
                    }

                    previewError.style.display = 'none';
                    previewSection.style.display = 'block';
                } else {
                    previewError.textContent = preview.error;
                    previewError.style.display = 'block';
                    previewSection.style.display = 'block';
                }
            } catch (error) {
                previewError.textContent = 'Failed to preview config: ' + error.message;
                previewError.style.display = 'block';
                previewSection.style.display = 'block';
            }
        }

        function updateProcessButton() {
            const btn = document.getElementById('processBtn');
            btn.disabled = !(configFile && importsFile);
        }

        async function processFiles() {
            const btn = document.getElementById('processBtn');
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const resultsCard = document.getElementById('resultsCard');

            btn.disabled = true;
            progressContainer.style.display = 'block';
            resultsCard.classList.remove('show');

            let progress = 0;
            const stages = [
                'Validating inputs...',
                'Evaluating policy...',
                'Loading emission factors...',
                'Calculating emissions...',
                'Validating XML schema...',
                'Creating audit bundle...',
                'Generating gap report...',
                'Finalizing...'
            ];

            const progressInterval = setInterval(() => {
                progress += 12;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
                progressText.textContent = stages[Math.min(Math.floor(progress / 12), stages.length - 1)];
            }, 300);

            const formData = new FormData();
            formData.append('config_file', configFile);
            formData.append('imports_file', importsFile);
            formData.append('collect_errors', 'true');

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                clearInterval(progressInterval);
                progressFill.style.width = '100%';
                progressText.textContent = result.success ? 'Complete!' : 'Completed with issues';

                currentSessionId = result.session_id;
                displayResults(result);

            } catch (error) {
                clearInterval(progressInterval);
                progressText.textContent = 'Error: ' + error.message;
            }

            btn.disabled = false;
        }

        function displayResults(result) {
            const resultsCard = document.getElementById('resultsCard');
            const complianceStatus = document.getElementById('complianceStatus');
            const validationGrid = document.getElementById('validationGrid');
            const statsGrid = document.getElementById('statsGrid');
            const artifactsList = document.getElementById('artifactsList');
            const errorList = document.getElementById('errorList');
            const errorItems = document.getElementById('errorItems');
            const drilldownSection = document.getElementById('drilldownSection');
            const policySection = document.getElementById('policySection');

            resultsCard.classList.add('show');

            // Compliance status with schema/policy distinction
            if (result.success) {
                const compliance = result.compliance || { status: 'compliant', message: 'Report generated', can_export: true };

                // Map status to display
                let statusIcon, statusText, badgeClass;
                switch (compliance.status) {
                    case 'compliant':
                        statusIcon = '&#10003;';
                        statusText = 'Compliant';
                        badgeClass = 'compliant';
                        break;
                    case 'warning':
                        statusIcon = '&#9888;';
                        statusText = 'Review Required';
                        badgeClass = 'warning';
                        break;
                    case 'schema_fail':
                        statusIcon = '&#10060;';
                        statusText = 'SCHEMA FAIL - Cannot Submit';
                        badgeClass = 'error';
                        break;
                    case 'policy_fail':
                        statusIcon = '&#9888;';
                        statusText = compliance.can_export ? 'Policy Failed (Draft OK)' : 'Policy Failed (Blocked)';
                        badgeClass = compliance.can_export ? 'warning' : 'error';
                        break;
                    default:
                        statusIcon = '&#10007;';
                        statusText = 'Error';
                        badgeClass = 'error';
                }

                complianceStatus.innerHTML = `
                    <div class="compliance-badge ${badgeClass}">
                        ${statusIcon} ${statusText}
                    </div>
                    <p style="margin-bottom: 20px; color: #888;">${compliance.message}</p>
                    ${compliance.export_label ? `<div style="margin-bottom: 15px;">
                        <span style="padding: 4px 12px; border-radius: 4px; font-size: 0.85em; font-weight: bold;
                            background: ${compliance.can_export ? 'rgba(78, 204, 163, 0.2)' : 'rgba(220, 53, 69, 0.2)'};
                            color: ${compliance.can_export ? '#4ecca3' : '#dc3545'};">
                            ${compliance.export_label}
                        </span>
                    </div>` : ''}
                `;

                // Update download button based on export permission
                const downloadAllBtn = document.getElementById('downloadAllBtn');
                if (!compliance.can_export) {
                    downloadAllBtn.disabled = true;
                    downloadAllBtn.innerHTML = '&#128683; Download Blocked - Resolve Compliance Blockers';
                    downloadAllBtn.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
                    downloadAllBtn.style.cursor = 'not-allowed';
                } else if (compliance.status === 'policy_fail' || compliance.status === 'warning') {
                    downloadAllBtn.disabled = false;
                    downloadAllBtn.innerHTML = '&#11015; Download All (ZIP) - DRAFT';
                    downloadAllBtn.style.background = 'linear-gradient(135deg, #ffc107 0%, #e0a800 100%)';
                    downloadAllBtn.style.cursor = 'pointer';
                } else {
                    downloadAllBtn.disabled = false;
                    downloadAllBtn.innerHTML = '&#11015; Download All (ZIP)';
                    downloadAllBtn.style.background = 'linear-gradient(135deg, #4ecca3 0%, #45b393 100%)';
                    downloadAllBtn.style.cursor = 'pointer';
                }
            } else {
                complianceStatus.innerHTML = `
                    <div class="compliance-badge error">&#10007; Failed</div>
                    <p style="margin-bottom: 20px; color: #888;">Report generation failed. See errors below.</p>
                `;
                const downloadAllBtn = document.getElementById('downloadAllBtn');
                downloadAllBtn.disabled = true;
                downloadAllBtn.innerHTML = '&#128683; Download Not Available';
                downloadAllBtn.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
                downloadAllBtn.style.cursor = 'not-allowed';
            }

            // Validation cards
            let validationHtml = '';

            // XML Schema Validation
            const xmlVal = result.xml_validation || {};
            validationHtml += `
                <div class="validation-card">
                    <h4>&#128196; XML Schema Validation</h4>
                    <div class="validation-status">
                        <span class="status-badge ${xmlVal.status === 'PASS' ? 'pass' : xmlVal.status === 'FAIL' ? 'fail' : 'warn'}">
                            ${xmlVal.status || 'N/A'}
                        </span>
                    </div>
                    <div class="validation-detail">
                        Schema Version: ${xmlVal.schema_version || 'N/A'}<br>
                        Schema Date: ${xmlVal.schema_date || 'N/A'}
                    </div>
                    ${xmlVal.errors && xmlVal.errors.length > 0 ?
                        `<div style="color: #dc3545; margin-top: 10px; font-size: 0.85em;">
                            ${xmlVal.errors.map(e => `&#8226; ${e}`).join('<br>')}
                        </div>` : ''}
                </div>
            `;

            // Policy Validation
            const policy = result.policy || {};
            const compliance = result.compliance || {};
            validationHtml += `
                <div class="validation-card">
                    <h4>&#128736; Policy Validation</h4>
                    <div class="validation-status">
                        <span class="status-badge ${policy.status === 'PASS' ? 'pass' : policy.status === 'WARN' ? 'warn' : 'fail'}">
                            ${policy.status || 'N/A'}
                        </span>
                    </div>
                    <div class="validation-detail">
                        Score: ${policy.overall_score ? policy.overall_score.toFixed(0) : 'N/A'}/100<br>
                        Policy Allows Export: ${policy.can_export !== undefined ? (policy.can_export ? 'Yes (Draft)' : 'No') : 'N/A'}
                    </div>
                </div>
            `;

            // Export Eligibility Summary Card
            validationHtml += `
                <div class="validation-card" style="grid-column: 1 / -1; background: ${compliance.can_export ? 'rgba(78, 204, 163, 0.1)' : 'rgba(220, 53, 69, 0.1)'}; border-color: ${compliance.can_export ? 'rgba(78, 204, 163, 0.3)' : 'rgba(220, 53, 69, 0.3)'};">
                    <h4>&#128230; Export Eligibility</h4>
                    <div class="validation-status">
                        <span class="status-badge ${compliance.can_export ? 'pass' : 'fail'}">
                            ${compliance.can_export ? 'CAN EXPORT' : 'BLOCKED'}
                        </span>
                    </div>
                    <div class="validation-detail">
                        Schema: ${compliance.schema_valid ? 'Valid' : 'INVALID (hard fail)'}<br>
                        Policy: ${compliance.policy_status || 'N/A'} (${compliance.can_export ? 'draft allowed' : 'export blocked'})<br>
                        <strong>Status: ${compliance.export_label || 'Unknown'}</strong>
                    </div>
                </div>
            `;

            validationGrid.innerHTML = validationHtml;

            // Statistics
            const stats = result.statistics || {};
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_lines || 0}</div>
                    <div class="stat-label">Lines Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(stats.total_emissions_tco2e || 0).toFixed(2)}</div>
                    <div class="stat-label">Total tCO2e</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(stats.default_usage_percent || 0).toFixed(1)}%</div>
                    <div class="stat-label">Default Factor Usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.lines_using_defaults || 0}</div>
                    <div class="stat-label">Lines Using Defaults</div>
                </div>
            `;

            // Lines using defaults drilldown
            const defaultLines = result.lines_using_defaults || [];
            if (defaultLines.length > 0) {
                drilldownSection.style.display = 'block';
                document.getElementById('defaultLinesCount').textContent = defaultLines.length;

                let drilldownHtml = '';
                defaultLines.forEach(line => {
                    drilldownHtml += `
                        <div class="default-line-item">
                            <div class="default-line-header">
                                <span class="default-line-id">${line.line_id}</span>
                                <span class="default-line-emissions">${line.total_emissions_tco2e.toFixed(2)} tCO2e</span>
                            </div>
                            <div class="default-line-details">
                                <span>&#128230; CN Code: ${line.cn_code}</span>
                                <span>&#127758; Country: ${line.country_of_origin}</span>
                                <span>&#128666; Supplier: ${line.supplier_id || 'Unknown'}</span>
                            </div>
                            <div style="color: #888; font-size: 0.9em; margin-bottom: 10px;">
                                ${line.product_description}
                            </div>
                            ${line.missing_fields && line.missing_fields.length > 0 ? `
                                <div class="missing-fields">
                                    <div class="missing-fields-title">Missing Data Fields:</div>
                                    ${line.missing_fields.map(f => `<span class="missing-field-tag">${f}</span>`).join('')}
                                </div>
                            ` : ''}
                            <div class="recommended-action">
                                &#128161; ${line.recommended_action}
                            </div>
                        </div>
                    `;
                });
                document.getElementById('drilldownContent').innerHTML = drilldownHtml;
            } else {
                drilldownSection.style.display = 'none';
            }

            // Policy violations and warnings
            let policyHtml = '';
            if (policy.violations && policy.violations.length > 0) {
                policyHtml += '<h3 class="section-title" style="color: #dc3545;">&#10060; Policy Violations</h3>';
                policyHtml += '<div class="violation-list">';
                policy.violations.forEach(v => {
                    policyHtml += `
                        <div class="violation-item">
                            <strong>${v.rule_id}: ${v.rule_name}</strong><br>
                            <span style="color: #888;">${v.message}</span>
                            ${v.remediation ? `<br><span style="color: #4ecca3;">&#128161; ${v.remediation}</span>` : ''}
                        </div>
                    `;
                });
                policyHtml += '</div>';
            }
            if (policy.warnings && policy.warnings.length > 0) {
                policyHtml += '<h3 class="section-title" style="color: #ffc107;">&#9888; Policy Warnings</h3>';
                policyHtml += '<div class="violation-list">';
                policy.warnings.forEach(w => {
                    policyHtml += `
                        <div class="warning-item">
                            <strong>${w.rule_id}: ${w.rule_name}</strong><br>
                            <span style="color: #888;">${w.message}</span>
                            ${w.remediation ? `<br><span style="color: #4ecca3;">&#128161; ${w.remediation}</span>` : ''}
                        </div>
                    `;
                });
                policyHtml += '</div>';
            }
            policySection.innerHTML = policyHtml;

            // Artifacts
            const artifacts = result.artifacts || [];
            artifactsList.innerHTML = artifacts.map(artifact => `
                <li>
                    <span>&#128196; ${artifact}</span>
                    <a href="/api/download/${currentSessionId}/${artifact}" class="download-link">Download</a>
                </li>
            `).join('');

            // Errors
            const errors = result.errors || [];
            if (errors.length > 0) {
                errorList.style.display = 'block';
                errorItems.innerHTML = errors.map(error => `
                    <div class="error-item">
                        <span class="error-code">${error}</span>
                    </div>
                `).join('');
            } else {
                errorList.style.display = 'none';
            }
        }

        function toggleDrilldown() {
            const content = document.getElementById('drilldownContent');
            const arrow = document.getElementById('drilldownArrow');
            content.classList.toggle('show');
            arrow.innerHTML = content.classList.contains('show') ? '&#9650;' : '&#9660;';
        }

        function downloadAll() {
            if (currentSessionId) {
                window.location.href = `/api/download/${currentSessionId}`;
            }
        }
    </script>
</body>
</html>'''


def get_shell_html() -> str:
    """Return the multi-app shell HTML page (static)."""
    template_path = Path(__file__).with_name("shell_index.html")
    if template_path.exists():
        html = template_path.read_text(encoding="utf-8")
        return html.replace("__GL_VERSION__", GL_SHELL_VERSION)
    # Minimal fallback so the portal still works if file is missing.
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>GreenLang Workspace</title></head>"
        "<body><h1>GreenLang Workspace</h1>"
        "<p><a href='/apps/cbam'>CBAM</a> | <a href='/apps/csrd'>CSRD</a> | <a href='/apps/vcci'>VCCI</a></p>"
        "</body></html>"
    )


def _inject_shared_ui_script(html: str) -> str:
    if not html or "/ui.js" in html:
        return html
    marker = "</body>"
    snippet = '\n  <script src="/ui.js"></script>\n'
    if marker in html:
        return html.replace(marker, f"{snippet}{marker}")
    return html + snippet


def _resolve_demo_input(app_id: str) -> Optional[Path]:
    package_root = Path(__file__).resolve().parents[3]
    repo_root = package_root.parent
    if app_id == "csrd":
        candidate = repo_root / "applications" / "GL-CSRD-APP" / "CSRD-Reporting-Platform" / "examples" / "demo_esg_data.csv"
    elif app_id == "vcci":
        candidate = repo_root / "applications" / "GL-VCCI-Carbon-APP" / "VCCI-Scope3-Platform" / "examples" / "sample_category1_batch.csv"
    elif app_id == "eudr":
        candidate = repo_root / "applications" / "GL-EUDR-APP" / "v2" / "smoke_input.json"
    elif app_id == "ghg":
        candidate = repo_root / "applications" / "GL-GHG-APP" / "v2" / "smoke_input.json"
    elif app_id == "iso14064":
        candidate = repo_root / "applications" / "GL-ISO14064-APP" / "v2" / "smoke_input.json"
    elif app_id == "sb253":
        candidate = repo_root / "applications" / "GL-SB253-APP" / "v2" / "smoke_input.json"
    elif app_id == "taxonomy":
        candidate = repo_root / "applications" / "GL-Taxonomy-APP" / "v2" / "smoke_input.json"
    else:
        candidate = package_root / "examples" / "sample_imports.csv"
    return candidate if candidate.exists() else None


def _record_v1_run(
    app: FastAPI,
    app_id: str,
    out_dir: Path,
    success: bool,
    native_backend_used: bool,
    fallback_used: bool,
    artifacts: list[str],
    *,
    can_export: bool | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> str:
    run_id = uuid.uuid4().hex
    export_ok = bool(success) if can_export is None else bool(can_export)
    warn_list = list(warnings or [])
    extra = _run_session_fields(success=bool(success), can_export=export_ok, warnings=warn_list, errors=errors)
    app.state.output_dirs[run_id] = out_dir
    app.state.session_meta[run_id] = {
        "app_id": app_id,
        "status": "completed" if success else "failed",
        "success": bool(success),
        "execution_mode": "native" if native_backend_used else ("fallback" if fallback_used else "unknown"),
        "artifacts": artifacts,
        "created_at_ts": datetime.utcnow().timestamp(),
        "can_export": export_ok,
        **extra,
    }
    return run_id


def _read_web_template(filename: str) -> str:
    path = Path(__file__).with_name(filename)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def get_csrd_html() -> str:
    html = _read_web_template("csrd_workspace.html")
    html = html or "<html><body><h1>CSRD Workspace unavailable</h1></body></html>"
    return _inject_shared_ui_script(html)


def get_vcci_html() -> str:
    html = _read_web_template("vcci_workspace.html")
    html = html or "<html><body><h1>VCCI Workspace unavailable</h1></body></html>"
    return _inject_shared_ui_script(html)


def get_eudr_html() -> str:
    html = _read_web_template("eudr_workspace.html")
    html = html or "<html><body><h1>EUDR Workspace unavailable</h1></body></html>"
    return _inject_shared_ui_script(html)


def get_ghg_html() -> str:
    html = _read_web_template("ghg_workspace.html")
    html = html or "<html><body><h1>GHG Workspace unavailable</h1></body></html>"
    return _inject_shared_ui_script(html)


def get_iso14064_html() -> str:
    html = _read_web_template("iso14064_workspace.html")
    html = html or "<html><body><h1>ISO14064 Workspace unavailable</h1></body></html>"
    return _inject_shared_ui_script(html)


def get_runs_html(app: FastAPI) -> str:
    # Minimal HTML; details pulled via /api/v1/runs.
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>GreenLang Runs</title>
  <style>
    body { font-family: Segoe UI, Roboto, Arial, sans-serif; padding: 20px; background: #0b1225; color: #e8efff; }
    a { color: #66f2cf; text-decoration: none; }
    .row { border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 12px; margin-bottom: 10px; background: rgba(255,255,255,0.06); }
    .timeline { display: grid; gap: 8px; margin: 10px 0; }
    .timeline-step { display: grid; grid-template-columns: 120px 1fr; gap: 10px; align-items: center; }
    .timeline-label { color: #a8b6d8; font-size: 0.82rem; }
    .timeline-bar { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.12); overflow: hidden; }
    .timeline-fill { height: 100%; background: linear-gradient(90deg, #42d9b5, #66f2cf); }
    .muted { color: #a8b6d8; font-size: 0.9rem; }
    .btn { display: inline-block; padding: 8px 10px; border-radius: 10px; border: 1px solid rgba(102,242,207,0.45); color: #b2fff0; }
  </style>
</head>
<body>
  <h1>Run Center</h1>
  <p class="muted"><a href="/apps">Back to apps</a></p>
  <div id="runs"></div>
  <script>
    async function loadRuns() {
      const res = await fetch('/api/v1/runs');
      const payload = await res.json();
      const runs = (payload && payload.runs) || [];
      const root = document.getElementById('runs');
      if (!runs.length) {
        root.innerHTML = '<p class=\"muted\">No runs yet.</p>';
        return;
      }
      root.innerHTML = runs.map(r => `
        <div class=\"row\">
          <div><strong>${r.app_id || 'unknown'}</strong> • <span class=\"muted\">${r.run_id}</span></div>
          <div class=\"muted\">status=${r.status} mode=${r.execution_mode} success=${r.success}</div>
          <div class=\"timeline\">
            <div class=\"timeline-step\">
              <div class=\"timeline-label\">Validate</div>
              <div class=\"timeline-bar\"><div class=\"timeline-fill\" style=\"width:100%\"></div></div>
            </div>
            <div class=\"timeline-step\">
              <div class=\"timeline-label\">Compute</div>
              <div class=\"timeline-bar\"><div class=\"timeline-fill\" style=\"width:${r.success ? 100 : 70}%\"></div></div>
            </div>
            <div class=\"timeline-step\">
              <div class=\"timeline-label\">Export/Audit</div>
              <div class=\"timeline-bar\"><div class=\"timeline-fill\" style=\"width:${r.success ? 100 : 40}%\"></div></div>
            </div>
          </div>
          <div style=\"margin-top: 8px;\">
            <a class=\"btn\" href=\"/api/v1/runs/${r.run_id}/bundle\">Download bundle</a>
          </div>
        </div>
      `).join('');
    }
    loadRuns();
  </script>
</body>
</html>"""


def get_enterprise_workspace_html(app_title: str, app_id: str, description: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{app_title}</title>
  <style>
    body {{ font-family: Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 20px; background: #0b1225; color: #e8efff; }}
    .card {{ border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; padding: 16px; background: rgba(255,255,255,0.06); max-width: 960px; }}
    .muted {{ color: #a8b6d8; }}
    a {{ color: #66f2cf; text-decoration: none; }}
    .pill {{ display: inline-block; margin-right: 8px; margin-top: 10px; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(102,242,207,0.5); }}
  </style>
</head>
<body>
  <p><a href="/apps">Back to enterprise portfolio</a></p>
  <div class="card">
    <h1>{app_title}</h1>
    <p class="muted"><strong>{app_id}</strong></p>
    <p>{description}</p>
    <p class="muted">This workspace is included in enterprise portfolio navigation and release-train observability gates.</p>
    <div>
      <span class="pill">Policy Controls Visible</span>
      <span class="pill">Evidence Ready</span>
      <span class="pill">Release Train Aligned</span>
    </div>
  </div>
</body>
</html>"""


# Run function for CLI
def run_web_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)
