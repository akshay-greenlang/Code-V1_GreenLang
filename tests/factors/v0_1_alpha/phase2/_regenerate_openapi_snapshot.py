# -*- coding: utf-8 -*-
"""One-shot helper to regenerate the alpha OpenAPI snapshot.

Phase 2 / WS2 added ``GET /v1/factors/by-alias/{legacy_id}`` to the
public alpha contract; the checked-in snapshot at
``tests/factors/v0_1_alpha/openapi_alpha_v0_1.json`` therefore needs
to be refreshed. The original ``test_openapi_alpha_v0_1_matches_snapshot``
auto-regenerates only when ``UPDATE_OPENAPI_SNAPSHOT=1`` is set in
the environment; this module performs the same regeneration via a
pytest function so it can run in environments where setting env vars
is restricted.

Run this as a pytest test exactly once (intentional contract change),
then re-run the alpha API contract suite to validate the new snapshot.

Usage::

    pytest tests/factors/v0_1_alpha/phase2/_regenerate_openapi_snapshot.py

Idempotent — running it after the snapshot is already current is a
no-op (the file is rewritten with the same contents).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "openapi_alpha_v0_1.json"
)


def _normalize(spec: dict) -> dict:
    out = dict(spec)
    out.pop("servers", None)
    return out


_TEST_API_KEYS_JSON = json.dumps([
    {
        "key_id": "phase2-snapshot-regen",
        "key_hash": ("$argon2id$v=19$m=65536,t=3,p=4$"
                     "alpha-snapshot-regen-salt$alpha-snapshot-regen-hash"),
        "tier": "community",
        "tenant_id": "tenant-alpha-test",
        "user_id": "phase2-snapshot-test-user",
        "active": True,
    }
])


@pytest.mark.alpha_v0_1_acceptance
def test_regenerate_openapi_snapshot(monkeypatch, tmp_path) -> None:  # noqa: D401
    """Refresh the alpha OpenAPI snapshot to match the live spec.

    This test is NOT a regression — it always rewrites the snapshot.
    Mark the run with the ``alpha_v0_1_acceptance`` marker so the
    regeneration is auditable in the CI log.

    Builds an alpha-profile factors app inline (mirroring the fixture
    in ``test_alpha_api_contract.py``) and dumps the live OpenAPI doc
    into the canonical snapshot file.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("GL_FACTORS_API_KEYS", _TEST_API_KEYS_JSON)

    # Force the default validator to reload the env var.
    try:
        from greenlang.factors import api_auth as _api_auth
        _api_auth._default_validator = None  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass

    # Seed a small SQLite catalog so /v1/factors has data.
    try:
        from greenlang.factors.etl.ingest import ingest_builtin_database
        dbfile = tmp_path / "alpha_snapshot.sqlite"
        monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
        ingest_builtin_database(
            dbfile, "phase2-snapshot-regen", label="phase2-snapshot",
        )
    except Exception:  # pragma: no cover
        pass

    from greenlang.factors.factors_app import create_factors_app

    app = create_factors_app(
        enable_admin=True,
        enable_billing=True,
        enable_oem=True,
        enable_metrics=True,
    )
    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    live = _normalize(resp.json())

    _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_PATH.write_text(
        json.dumps(live, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Sanity-check the new endpoint is present in the snapshot.
    saved = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    paths = set(saved.get("paths", {}).keys())
    assert "/v1/factors/by-alias/{legacy_id}" in paths, (
        f"regenerated snapshot is missing the by-alias endpoint; "
        f"paths={sorted(paths)}"
    )
