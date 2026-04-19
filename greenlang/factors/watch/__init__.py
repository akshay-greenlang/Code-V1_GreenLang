# -*- coding: utf-8 -*-
"""Update engine: source watch, changelog draft, rollback helpers (U1-U6), scheduler (F050), release (F053)."""

from greenlang.factors.watch.source_watch import dry_run_registry_urls
from greenlang.factors.watch.change_classification import (
    ChangeClassification,
    ChangeType,
    classify_change,
)
from greenlang.factors.watch.changelog_draft import draft_changelog, draft_changelog_lines
from greenlang.factors.watch.rollback_edition import (
    RollbackResult,
    resolve_edition_with_rollback_override,
    rollback_to_edition,
)
from greenlang.factors.watch.doc_diff import (
    DocDiff,
    FieldDiff,
    diff_text_versions,
    fingerprint_factor,
    fingerprint_text,
    generate_doc_diff,
    summarize_changes,
)
from greenlang.factors.watch.scheduler import run_watch, watch_summary, WatchResult
from greenlang.factors.watch.release_orchestrator import prepare_release, publish_release, ReleaseReport
from greenlang.factors.watch.change_detector import detect_source_change, ChangeReport, ChangedFactor

__all__ = [
    # source_watch (U1)
    "dry_run_registry_urls",
    # change_classification (U3)
    "ChangeClassification",
    "ChangeType",
    "classify_change",
    # changelog_draft (U5)
    "draft_changelog",
    "draft_changelog_lines",
    # rollback_edition (U6)
    "RollbackResult",
    "resolve_edition_with_rollback_override",
    "rollback_to_edition",
    # doc_diff (U2)
    "DocDiff",
    "FieldDiff",
    "diff_text_versions",
    "fingerprint_factor",
    "fingerprint_text",
    "generate_doc_diff",
    "summarize_changes",
    # scheduler (F050)
    "run_watch",
    "watch_summary",
    "WatchResult",
    # release_orchestrator (F053)
    "prepare_release",
    "publish_release",
    "ReleaseReport",
    # change_detector (F051)
    "detect_source_change",
    "ChangeReport",
    "ChangedFactor",
]
