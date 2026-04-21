# -*- coding: utf-8 -*-
"""Tests for the edition-rollback CLI + cross-edition changelog + the
deprecated ``quality.review_queue`` compatibility shim."""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from greenlang.factors.watch.cross_edition_changelog import (
    CrossEditionChangelog,
    FactorHistory,
    build_cross_edition_changelog,
    collect_release_notes,
    summarize_factor_history,
)
from greenlang.factors.watch.regulatory_events import (
    RegulatoryEventKind,
    RegulatoryEventStore,
)
from greenlang.factors.watch.rollback_cli import (
    RollbackCandidate,
    RollbackPreview,
    RollbackReceipt,
    execute_rollback_with_receipt,
    list_rollback_candidates,
    preview_rollback,
)


# ---------------------------------------------------------------------------
# In-memory repo stub — only implements the surface the CLI touches.
# ---------------------------------------------------------------------------


class _StubRepo:
    """Intentionally minimal: dict-based editions so tests control state."""

    def __init__(
        self,
        editions: Dict[str, Dict[str, Any]],
        factor_summaries: Dict[str, List[Dict[str, str]]],
        changelog: Optional[Dict[str, List[str]]] = None,
        default_edition: Optional[str] = None,
    ) -> None:
        self._editions = editions
        self._factor_summaries = factor_summaries
        self._changelog = changelog or {}
        self._default = default_edition
        self.rollback_calls: List[Dict[str, Any]] = []

    # ---- contract used by rollback_cli / cross_edition_changelog -------

    def list_editions(self) -> List[Dict[str, Any]]:
        return [{"edition_id": eid, **meta} for eid, meta in self._editions.items()]

    def resolve_edition(self, edition_id: Optional[str]) -> str:
        if edition_id is None:
            if not self._default:
                raise ValueError("no default")
            return self._default
        if edition_id not in self._editions:
            raise ValueError(f"unknown edition {edition_id}")
        return edition_id

    def list_factor_summaries(self, edition_id: str) -> List[Dict[str, str]]:
        if edition_id not in self._editions:
            return []
        return list(self._factor_summaries.get(edition_id, []))

    def get_default_edition_id(self) -> Optional[str]:
        return self._default

    def get_manifest_dict(self, edition_id: str) -> Dict[str, Any]:
        return {}

    def get_changelog(self, edition_id: str) -> List[str]:
        return list(self._changelog.get(edition_id, []))

    # Support draft_changelog in cross-edition aggregator.
    def get_factor(self, edition_id: str, factor_id: str) -> Any:
        return None


# ---------------------------------------------------------------------------
# list_rollback_candidates
# ---------------------------------------------------------------------------


class TestListRollbackCandidates:
    def _repo(self) -> _StubRepo:
        return _StubRepo(
            editions={
                "2026.02.0": {"status": "rollback_demoted", "published_at": "2026-02-01"},
                "2026.03.0": {"status": "stable", "published_at": "2026-03-01"},
                "2026.04.0": {"status": "stable", "published_at": "2026-04-01"},
            },
            factor_summaries={
                "2026.02.0": [{"factor_id": "a", "content_hash": "h1", "factor_status": "certified"}],
                "2026.03.0": [{"factor_id": "a", "content_hash": "h2", "factor_status": "certified"}],
                "2026.04.0": [
                    {"factor_id": "a", "content_hash": "h3", "factor_status": "certified"},
                    {"factor_id": "b", "content_hash": "hb", "factor_status": "certified"},
                ],
            },
            default_edition="2026.04.0",
        )

    def test_candidates_sorted_newest_first(self) -> None:
        candidates = list_rollback_candidates(self._repo())
        # All three returned; only status "deprecated" is filtered by default.
        # rollback_demoted stays visible so operators can revert past rollbacks.
        ids = [c.edition_id for c in candidates]
        assert ids == ["2026.04.0", "2026.03.0", "2026.02.0"]

    def test_deprecated_status_is_excluded_by_default(self) -> None:
        repo = _StubRepo(
            editions={
                "2026.01.0": {"status": "deprecated", "published_at": "2026-01-01"},
                "2026.04.0": {"status": "stable", "published_at": "2026-04-01"},
            },
            factor_summaries={
                "2026.01.0": [{"factor_id": "a", "content_hash": "h1", "factor_status": "certified"}],
                "2026.04.0": [{"factor_id": "a", "content_hash": "h4", "factor_status": "certified"}],
            },
            default_edition="2026.04.0",
        )
        default_view = list_rollback_candidates(repo)
        assert {c.edition_id for c in default_view} == {"2026.04.0"}
        with_deprecated = list_rollback_candidates(repo, include_deprecated=True)
        assert {c.edition_id for c in with_deprecated} == {"2026.01.0", "2026.04.0"}

    def test_demoted_editions_remain_visible(self) -> None:
        """Past rollback victims should always be listed so operators
        can flip back if the rollback turns out to be wrong."""
        candidates = list_rollback_candidates(self._repo())
        ids = {c.edition_id for c in candidates}
        assert "2026.02.0" in ids

    def test_marks_current_default(self) -> None:
        candidates = list_rollback_candidates(self._repo())
        default = next(c for c in candidates if c.is_current_default)
        assert default.edition_id == "2026.04.0"

    def test_factor_count_populated(self) -> None:
        candidates = list_rollback_candidates(self._repo())
        count_by_id = {c.edition_id: c.factor_count for c in candidates}
        assert count_by_id["2026.04.0"] == 2
        assert count_by_id["2026.03.0"] == 1


# ---------------------------------------------------------------------------
# preview_rollback
# ---------------------------------------------------------------------------


class TestPreviewRollback:
    def _repo(self) -> _StubRepo:
        return _StubRepo(
            editions={
                "2026.03.0": {"status": "stable", "published_at": "2026-03-01"},
                "2026.04.0": {"status": "stable", "published_at": "2026-04-01"},
            },
            factor_summaries={
                "2026.03.0": [
                    {"factor_id": "a", "content_hash": "h2", "factor_status": "certified"},
                    {"factor_id": "c", "content_hash": "hc_old", "factor_status": "certified"},
                ],
                "2026.04.0": [
                    {"factor_id": "a", "content_hash": "h3", "factor_status": "certified"},
                    {"factor_id": "b", "content_hash": "hb", "factor_status": "certified"},
                ],
            },
            default_edition="2026.04.0",
        )

    def test_safe_diff_classification(self) -> None:
        preview = preview_rollback(self._repo(), "2026.03.0")
        assert preview.safe is True
        # 'c' exists only in target → "added" on rollback
        assert preview.added_factor_ids == ["c"]
        # 'b' exists only in current → "removed" on rollback
        assert preview.removed_factor_ids == ["b"]
        # 'a' differs → changed
        assert preview.changed_factor_ids == ["a"]
        assert preview.target_factor_count == 2

    def test_blocks_on_missing_edition(self) -> None:
        preview = preview_rollback(self._repo(), "not_a_real_edition")
        assert preview.safe is False
        assert any("does not exist" in b or "unknown" in b for b in preview.blockers)

    def test_blocks_when_target_is_already_default(self) -> None:
        preview = preview_rollback(self._repo(), "2026.04.0")
        assert preview.safe is False
        assert any("already the default" in b for b in preview.blockers)

    def test_summary_shape(self) -> None:
        preview = preview_rollback(self._repo(), "2026.03.0")
        summary = preview.summary()
        assert summary["added"] == 1
        assert summary["removed"] == 1
        assert summary["changed"] == 1


# ---------------------------------------------------------------------------
# execute_rollback_with_receipt
# ---------------------------------------------------------------------------


class TestExecuteRollbackWithReceipt:
    def _repo(self) -> _StubRepo:
        return _StubRepo(
            editions={
                "2026.03.0": {"status": "stable", "published_at": "2026-03-01"},
                "2026.04.0": {"status": "stable", "published_at": "2026-04-01"},
            },
            factor_summaries={
                "2026.03.0": [
                    {"factor_id": "a", "content_hash": "h2", "factor_status": "certified"},
                ],
                "2026.04.0": [
                    {"factor_id": "a", "content_hash": "h3", "factor_status": "certified"},
                ],
            },
            default_edition="2026.04.0",
        )

    def test_unsafe_preview_produces_failed_receipt(self, tmp_path: Path) -> None:
        receipt = execute_rollback_with_receipt(
            self._repo(),
            target_edition_id="no_such_edition",
            operator="alice",
            reason="test",
            receipt_dir=tmp_path,
        )
        assert receipt.success is False
        assert receipt.result.get("error") == "preview_unsafe"
        # Receipt JSON is persisted.
        assert (tmp_path / f"{receipt.receipt_id}.json").exists()

    def test_unsupported_repo_returns_failure_without_raising(
        self, tmp_path: Path
    ) -> None:
        """Our stub repo is neither SQLite nor in-memory; the rollback
        primitive treats it as unsupported and returns ``success=False``
        with a helpful reason.  The receipt captures that gracefully."""
        event_store = RegulatoryEventStore(tmp_path / "events.db")
        receipt = execute_rollback_with_receipt(
            self._repo(),
            target_edition_id="2026.03.0",
            operator="alice",
            reason="quarterly regression",
            event_store=event_store,
            receipt_dir=tmp_path,
        )
        assert receipt.preview["safe"] is True
        assert receipt.success is False
        assert "not implemented" in (receipt.result.get("reason") or "").lower()
        # No breaking-change event should fire on an unsuccessful rollback.
        assert event_store.count(source_id="greenlang_factors") == 0

    def test_preview_block_does_not_persist_event(self, tmp_path: Path) -> None:
        event_store = RegulatoryEventStore(tmp_path / "events.db")
        execute_rollback_with_receipt(
            self._repo(),
            target_edition_id="2026.04.0",  # already default → blocked
            operator="alice",
            reason="",
            event_store=event_store,
            receipt_dir=tmp_path,
        )
        assert event_store.count(source_id="greenlang_factors") == 0


# ---------------------------------------------------------------------------
# cross-edition changelog
# ---------------------------------------------------------------------------


class TestCrossEditionChangelog:
    def _repo(self) -> _StubRepo:
        return _StubRepo(
            editions={
                "2026.02.0": {"status": "stable"},
                "2026.03.0": {"status": "stable"},
                "2026.04.0": {"status": "stable"},
            },
            factor_summaries={
                "2026.02.0": [{"factor_id": "a", "content_hash": "h1", "factor_status": "certified"}],
                "2026.03.0": [
                    {"factor_id": "a", "content_hash": "h2", "factor_status": "certified"},
                    {"factor_id": "b", "content_hash": "hb", "factor_status": "certified"},
                ],
                "2026.04.0": [
                    {"factor_id": "a", "content_hash": "h3", "factor_status": "certified"},
                ],
            },
            changelog={
                "2026.03.0": ["Added factor b.", "Updated factor a."],
                "2026.04.0": ["Removed factor b."],
            },
            default_edition="2026.04.0",
        )

    def test_empty_when_fewer_than_two_editions(self) -> None:
        report = build_cross_edition_changelog(self._repo(), ["2026.04.0"])
        assert report.editions == ["2026.04.0"]
        assert report.sections == []

    def test_pairs_consecutive_editions(self) -> None:
        report = build_cross_edition_changelog(
            self._repo(), ["2026.02.0", "2026.03.0", "2026.04.0"]
        )
        assert len(report.sections) == 2
        assert report.sections[0].left_edition_id == "2026.02.0"
        assert report.sections[0].right_edition_id == "2026.03.0"
        assert report.sections[1].left_edition_id == "2026.03.0"
        assert report.sections[1].right_edition_id == "2026.04.0"

    def test_to_text_includes_editions_header(self) -> None:
        report = build_cross_edition_changelog(
            self._repo(), ["2026.02.0", "2026.03.0"]
        )
        text = report.to_text()
        assert "cross-edition changelog: 2026.02.0 -> 2026.03.0" in text

    def test_release_notes_collector(self) -> None:
        notes = collect_release_notes(
            self._repo(), ["2026.02.0", "2026.03.0", "2026.04.0"]
        )
        # 2026.02.0 has no changelog; 03 and 04 do.
        edition_ids = [e for e, _ in notes]
        assert "2026.03.0" in edition_ids
        assert "2026.04.0" in edition_ids


class TestFactorHistory:
    def _repo(self) -> _StubRepo:
        return _StubRepo(
            editions={
                "2026.02.0": {"status": "stable"},
                "2026.03.0": {"status": "stable"},
                "2026.04.0": {"status": "stable"},
            },
            factor_summaries={
                "2026.02.0": [{"factor_id": "a", "content_hash": "h1", "factor_status": "certified"}],
                "2026.03.0": [
                    {"factor_id": "a", "content_hash": "h2", "factor_status": "certified"},
                    {"factor_id": "b", "content_hash": "hb", "factor_status": "certified"},
                ],
                "2026.04.0": [
                    {"factor_id": "a", "content_hash": "h3", "factor_status": "deprecated"},
                ],
            },
        )

    def test_history_tracks_every_appearance(self) -> None:
        history = summarize_factor_history(
            self._repo(), "a",
            ["2026.02.0", "2026.03.0", "2026.04.0"],
        )
        assert history.factor_id == "a"
        assert history.total_editions == 3
        assert history.unique_content_hashes == 3
        assert history.first_seen_in == "2026.02.0"
        assert history.last_seen_in == "2026.04.0"
        # Last point reflects deprecation status.
        assert history.points[-1].factor_status == "deprecated"

    def test_history_handles_missing_factor(self) -> None:
        history = summarize_factor_history(
            self._repo(), "not_a_factor",
            ["2026.02.0", "2026.03.0", "2026.04.0"],
        )
        assert history.total_editions == 0
        assert history.first_seen_in is None
        assert history.last_seen_in is None


# ---------------------------------------------------------------------------
# Deprecated review_queue compatibility shim
# ---------------------------------------------------------------------------


class TestReviewQueueDeprecation:
    def test_emits_deprecation_warning(self) -> None:
        from greenlang.factors.quality.review_queue import enqueue_review

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE qa_reviews ("
            "review_id TEXT, edition_id TEXT, factor_id TEXT, "
            "status TEXT, payload_json TEXT)"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rid = enqueue_review(
                conn,
                edition_id="e1",
                factor_id="f1",
                status="pending",
                payload={"k": 1},
            )
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "review_workflow" in str(w.message)
            for w in caught
        )
        # And the shim still writes a row.
        row = conn.execute(
            "SELECT review_id, factor_id FROM qa_reviews WHERE review_id = ?",
            (rid,),
        ).fetchone()
        assert row is not None
        assert row[1] == "f1"
