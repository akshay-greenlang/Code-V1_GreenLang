# -*- coding: utf-8 -*-
"""
Tests for the Wave 5 release-notes auto-publish hook.

Covers :func:`greenlang.factors.watch.release_orchestrator.publish_release_notes`:

    1. Template renders with a realistic edition diff (factor-count delta,
       new-source list, P@1 / R@3 gold-eval delta).
    2. Prepending to ``docs/developer-portal/changelog.md`` preserves the
       page preamble and prior entries.
    3. Per-edition file is created at
       ``docs/developer-portal/releases/<slug>-<vintage>.md``.
    4. Structured log event ``factors.edition.release.published`` is
       emitted with the canonical payload shape the webhook system
       consumes.

All filesystem writes are scoped to a ``tmp_path`` via the ``docs_root``
parameter — the real ``docs/developer-portal/changelog.md`` is never
touched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from greenlang.factors.watch.release_orchestrator import publish_release_notes


# ---------------------------------------------------------------------------
# Shared repo double
# ---------------------------------------------------------------------------


def _make_repo(
    *,
    current_manifest: Dict[str, Any],
    previous_manifest: Dict[str, Any] = None,
    diff: Dict[str, List[str]] = None,
) -> MagicMock:
    """Build a ``FactorCatalogRepository`` double wired for the release
    notes diff path.

    ``diff`` mirrors the dict returned by
    :func:`FactorCatalogService.compare_editions`.
    """
    repo = MagicMock()
    previous_manifest = previous_manifest or {}
    diff = diff or {"added_factor_ids": [], "removed_factor_ids": [], "changed_factor_ids": []}

    def _manifest_for(edition_id: str):
        if edition_id == current_manifest.get("edition_id"):
            return dict(current_manifest)
        if previous_manifest and edition_id == previous_manifest.get("edition_id"):
            return dict(previous_manifest)
        return {}

    repo.get_manifest_dict.side_effect = _manifest_for
    repo.resolve_edition.side_effect = lambda e: e or current_manifest.get("edition_id")
    repo._compare_diff = diff  # stashed for the service patch below
    return repo


@pytest.fixture()
def repo_with_diff(monkeypatch):
    """Repo + patched FactorCatalogService so compare_editions is deterministic."""
    current = {
        "edition_id": "2026.05.0",
        "total_factors": 1275,
        "per_source_hashes": {
            "epa_egrid_2024": "abc",
            "desnz_2024": "def",
            "ipcc_ar6": "ghi",
            "ipcc_landuse_2019": "jkl",  # NEW
        },
        "deprecations": ["EF:legacy_coal_us_2019"],
        "schema_changes": ["Added `boundary` column"],
    }
    previous = {
        "edition_id": "2026.04.0",
        "total_factors": 1210,
        "per_source_hashes": {
            "epa_egrid_2024": "abc",
            "desnz_2024": "def",
            "ipcc_ar6": "ghi",
        },
    }
    diff = {
        "added_factor_ids": [f"EF:added_{i}" for i in range(75)],
        "removed_factor_ids": ["EF:legacy_coal_us_2019"],
        "changed_factor_ids": [f"EF:changed_{i}" for i in range(10)],
    }

    repo = _make_repo(current_manifest=current, previous_manifest=previous, diff=diff)

    # Patch compare_editions to return our canned diff regardless of
    # how the orchestrator instantiates the service.
    fake_svc = MagicMock()
    fake_svc.compare_editions.return_value = diff
    monkeypatch.setattr(
        "greenlang.factors.service.FactorCatalogService",
        MagicMock(return_value=fake_svc),
    )
    return repo


# ---------------------------------------------------------------------------
# (1) Template renders with a realistic edition diff
# ---------------------------------------------------------------------------


def test_template_renders_realistic_diff(repo_with_diff, tmp_path: Path):
    """The rendered markdown includes the factor-count delta, the new
    source, the gold-eval deltas, and the canonical verification footer.
    """
    gold_eval = {
        "p_at_1": 0.94,
        "previous_p_at_1": 0.91,
        "p_at_1_delta": 0.03,
        "r_at_3": 0.98,
        "previous_r_at_3": 0.97,
        "r_at_3_delta": 0.01,
    }

    outcome = publish_release_notes(
        repo_with_diff,
        edition_id="2026.05.0",
        previous_edition_id="2026.04.0",
        docs_root=tmp_path,
        gold_eval=gold_eval,
    )

    per_edition = Path(outcome["per_edition_path"])
    rendered = per_edition.read_text(encoding="utf-8")

    # Factor-count delta surfaces (1275 - 1210 = +65).
    assert "1275" in rendered
    assert "1210" in rendered
    # New sources list shows the delta source.
    assert "ipcc_landuse_2019" in rendered
    # Deprecated factor shows up.
    assert "EF:legacy_coal_us_2019" in rendered
    # Schema changes surface.
    assert "boundary" in rendered
    # Gold-eval deltas render.
    assert "0.94" in rendered
    assert "0.91" in rendered
    assert "0.98" in rendered
    # Canonical verification footer is always emitted.
    assert "Ed25519" in rendered or "ed25519" in rendered.lower()


# ---------------------------------------------------------------------------
# (2) Prepend preserves prior entries + page preamble
# ---------------------------------------------------------------------------


def test_changelog_prepend_preserves_prior_entries(repo_with_diff, tmp_path: Path):
    """Writing two releases in sequence keeps the first under the second,
    and keeps the page preamble intact across both writes.
    """
    # First release.
    publish_release_notes(
        repo_with_diff,
        edition_id="2026.05.0",
        previous_edition_id="2026.04.0",
        docs_root=tmp_path,
    )

    changelog_path = tmp_path / "changelog.md"
    assert changelog_path.exists()
    first_text = changelog_path.read_text(encoding="utf-8")
    assert "2026.05.0" in first_text
    # The auto-generated page preamble must be present on first write.
    assert "Public changelog" in first_text

    # Second release — flip the "current" manifest so we have something
    # new to prepend. (Edit in place on our test double.)
    new_current = {
        "edition_id": "2026.06.0",
        "total_factors": 1330,
        "per_source_hashes": {
            "epa_egrid_2024": "abc",
            "desnz_2024": "def",
            "ipcc_ar6": "ghi",
            "ipcc_landuse_2019": "jkl",
            "pcaf_v2": "mno",  # NEW
        },
    }
    repo_with_diff.get_manifest_dict.side_effect = lambda eid: {
        "2026.06.0": new_current,
        "2026.05.0": {
            "edition_id": "2026.05.0",
            "total_factors": 1275,
            "per_source_hashes": {
                "epa_egrid_2024": "abc",
                "desnz_2024": "def",
                "ipcc_ar6": "ghi",
                "ipcc_landuse_2019": "jkl",
            },
        },
    }.get(eid, {})

    publish_release_notes(
        repo_with_diff,
        edition_id="2026.06.0",
        previous_edition_id="2026.05.0",
        docs_root=tmp_path,
    )

    final_text = changelog_path.read_text(encoding="utf-8")

    # Both entries must be present.
    assert "2026.05.0" in final_text
    assert "2026.06.0" in final_text
    # Page preamble survives the second prepend.
    assert "Public changelog" in final_text
    # Newest entry (2026.06.0) appears BEFORE the older 2026.05.0 entry.
    assert final_text.index("2026.06.0") < final_text.index("2026.05.0")


# ---------------------------------------------------------------------------
# (3) Per-edition file created at docs_root/releases/<slug>-<vintage>.md
# ---------------------------------------------------------------------------


def test_per_edition_file_path(repo_with_diff, tmp_path: Path):
    outcome = publish_release_notes(
        repo_with_diff,
        edition_id="2026.05.0",
        previous_edition_id="2026.04.0",
        docs_root=tmp_path,
    )

    per_edition = Path(outcome["per_edition_path"])
    assert per_edition.exists()
    # Lives under docs_root/releases/.
    assert per_edition.parent == tmp_path / "releases"
    # Filename is "<slug>-<vintage>.md". slug := 2026.05.0 (already safe),
    # vintage := 2026.05. The renderer collapses consecutive dots into a
    # single "." separator.
    assert per_edition.name == "2026.05.0-2026.05.md"
    # Content is non-empty + contains the edition id.
    body = per_edition.read_text(encoding="utf-8")
    assert "2026.05.0" in body
    assert len(body) > 200  # sanity: rendered notes are substantial


# ---------------------------------------------------------------------------
# (4) Structured log event factors.edition.release.published
# ---------------------------------------------------------------------------


def test_structured_log_event_payload(
    repo_with_diff,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """The hook emits a structured ``factors.edition.release.published`` event.

    We capture the ``extra["structured_event"]`` attachment directly and
    also assert the message body is a parseable JSON payload so the
    webhook fan-out can consume either path.
    """
    gold_eval = {
        "p_at_1": 0.92,
        "previous_p_at_1": 0.90,
        "p_at_1_delta": 0.02,
        "r_at_3": 0.97,
        "previous_r_at_3": 0.96,
        "r_at_3_delta": 0.01,
    }

    # Target the logger the orchestrator module actually writes to.
    logger_name = "greenlang.factors.watch.release_orchestrator"
    caplog.set_level(logging.INFO, logger=logger_name)

    outcome = publish_release_notes(
        repo_with_diff,
        edition_id="2026.05.0",
        previous_edition_id="2026.04.0",
        docs_root=tmp_path,
        gold_eval=gold_eval,
    )

    event_records = [
        r for r in caplog.records
        if r.name == logger_name
        and "factors.edition.release.published" in r.getMessage()
    ]
    assert event_records, "no release.published log event captured"
    rec = event_records[-1]

    # Path 1: structured_event attached via `extra`.
    structured = getattr(rec, "structured_event", None)
    assert structured is not None, "structured_event missing from log record"
    assert structured["event"] == "factors.edition.release.published"
    assert structured["edition_id"] == "2026.05.0"
    assert structured["previous_edition_id"] == "2026.04.0"
    assert structured["factor_count"] == 1275
    assert structured["factor_count_delta"] == 65
    assert structured["added_count"] == 75
    assert structured["removed_count"] == 1
    assert structured["changed_count"] == 10
    assert "ipcc_landuse_2019" in structured["new_sources"]
    assert "gold_eval" in structured
    assert structured["gold_eval"]["p_at_1"] == 0.92

    # Path 2: message body carries the same payload as parseable JSON.
    msg = rec.getMessage()
    # Message shape: "factors.edition.release.published <json>"
    json_part = msg.split(" ", 1)[1]
    parsed = json.loads(json_part)
    assert parsed["event"] == "factors.edition.release.published"
    assert parsed["edition_id"] == "2026.05.0"

    # Outcome surface the caller can persist.
    assert outcome["event"]["edition_id"] == "2026.05.0"
    assert outcome["event"]["factor_count_delta"] == 65
