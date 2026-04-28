#!/usr/bin/env python3
"""
check_schema_migration_notes.py - GreenLang Factors schema-evolution CI gate.

This is the CI guardrail mandated by Phase 2 brief Section 2.6 (Schema Evolution
Policy, WS9). It diffs every changed `config/schemas/factor_record_*.schema.json`
file between a base ref and a head ref, runs a conservative compatibility
classifier, and asserts that the project CHANGELOG carries a matching entry of
sufficient severity for each version segment touched.

Authority: CTO Phase 2 brief 2026-04-27.
Sister artefacts (owned by WS9-A agent, NOT modified here):
  - docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md
  - docs/factors/schema/MIGRATION_NOTE_TEMPLATE.md
  - docs/factors/schema/CHANGELOG.md
  - greenlang/factors/schemas/_version_registry.py

Exit codes:
    0 -> all changed schemas have an adequate CHANGELOG entry.
    1 -> one or more changed schemas have NO CHANGELOG entry for their version.
    2 -> a CHANGELOG entry exists but its classification is too weak.
    3 -> usage / unrecoverable error (git failure, malformed JSON, etc.).

Classifier rule order (first match wins, conservative bias toward `breaking`):
    a. Any required field added                          -> breaking
    b. Any field removed                                 -> removed
    c. Any type narrowed (e.g., string -> integer)       -> breaking
    d. Any enum value removed                            -> breaking
    e. Any pattern tightened (string-source mismatch)    -> breaking
    f. Any field's `deprecated: true` flag added         -> deprecated
    g. Any optional field added                          -> additive
    h. Any enum value added                              -> additive
    i. Description-only change                           -> additive
    j. Default (when in doubt)                           -> breaking

Usage:
    python scripts/ci/check_schema_migration_notes.py \
        --base-ref origin/master \
        --head-ref HEAD \
        --changelog docs/factors/schema/CHANGELOG.md \
        --schema-glob "config/schemas/factor_record_*.schema.json"

Hard constraints:
    * stdlib + jsonschema only (no deepdiff, no jsondiff).
    * subprocess git calls always pass cwd=; never rely on global cwd.
    * Idempotent — never modifies files.
"""

from __future__ import annotations

import argparse
import enum
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Constants & regex
# ---------------------------------------------------------------------------

# CTO-mandated CHANGELOG anchor (ASCII only).
# Matches: "## v0.1 - 2026-04-26 - additive"
CHANGELOG_ENTRY_RE = re.compile(
    r"^## v(\d+)\.(\d+) - (\d{4}-\d{2}-\d{2}) - (additive|breaking|deprecated|removed)\s*$"
)

# Extract a "v0_1" style version tag from a schema filename, e.g.
# "factor_record_v0_1.schema.json" -> "v0_1" -> normalised "v0.1".
VERSION_FROM_FILENAME_RE = re.compile(r"factor_record_(v\d+_\d+)\.schema\.json$")

# Optional whitelist of "known superset" pattern transitions. If a tightened
# pattern appears in this allowlist, it's downgraded to additive. Empty by
# default; can be extended as needed.
KNOWN_PATTERN_SUPERSET_ALLOWLIST: set[tuple[str, str]] = set()

# ANSI colour helpers — degrade gracefully when stdout is not a TTY.
_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    """Wrap *s* in an ANSI colour escape if the terminal supports it."""
    if not _USE_COLOUR:
        return s
    return f"\033[{code}m{s}\033[0m"


def _green(s: str) -> str:
    return _c("32", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _red(s: str) -> str:
    return _c("31", s)


def _bold(s: str) -> str:
    return _c("1", s)


# ---------------------------------------------------------------------------
# Severity model
# ---------------------------------------------------------------------------

class Severity(enum.IntEnum):
    """Severity ordering: additive < deprecated < breaking <= removed."""

    ADDITIVE = 1
    DEPRECATED = 2
    BREAKING = 3
    REMOVED = 4

    @classmethod
    def from_label(cls, label: str) -> "Severity":
        return {
            "additive": cls.ADDITIVE,
            "deprecated": cls.DEPRECATED,
            "breaking": cls.BREAKING,
            "removed": cls.REMOVED,
        }[label.lower().strip()]

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class ClassificationResult:
    """Outcome of running the classifier against one schema diff."""

    severity: Severity
    rule: str  # one of: 'a','b','c','d','e','f','g','h','i','j'
    detail: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "severity": self.severity.label,
            "rule": self.rule,
            "detail": self.detail,
        }


@dataclass
class SchemaDiff:
    """The classifier's decision for a single changed schema path."""

    path: str
    version_tag: str  # normalised, e.g. "v0.1"
    classification: ClassificationResult


@dataclass
class ChangelogEntry:
    """A single parsed CHANGELOG anchor entry."""

    version_tag: str  # "v0.1"
    date: str
    severity: Severity
    line_index: int  # 0-based line in the source CHANGELOG


@dataclass
class GateOutcome:
    """Per-schema verdict produced by the gate."""

    diff: SchemaDiff
    matching_entry: Optional[ChangelogEntry]
    verdict: str  # "ok" | "missing_entry" | "classification_mismatch"
    message: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "path": self.diff.path,
            "version_tag": self.diff.version_tag,
            "classification": self.diff.classification.to_jsonable(),
            "matching_entry": (
                {
                    "version_tag": self.matching_entry.version_tag,
                    "date": self.matching_entry.date,
                    "severity": self.matching_entry.severity.label,
                    "line_index": self.matching_entry.line_index,
                }
                if self.matching_entry is not None
                else None
            ),
            "verdict": self.verdict,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

class GitError(RuntimeError):
    """Raised when a git subprocess fails or returns unparseable output."""


def _run_git(args: list[str], *, cwd: Path) -> str:
    """
    Run `git <args>` with an explicit cwd and return stdout (text).

    Never relies on the process-global cwd. Raises GitError on non-zero exit.
    """
    cmd = ["git"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitError(f"git executable not found: {exc}") from exc

    if proc.returncode != 0:
        raise GitError(
            f"git failed (exit={proc.returncode}): {' '.join(shlex.quote(a) for a in cmd)}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    return proc.stdout


def list_changed_files(*, base_ref: str, head_ref: str, cwd: Path) -> list[str]:
    """Return the set of files changed between *base_ref* and *head_ref*."""
    out = _run_git(
        ["diff", "--name-only", f"{base_ref}..{head_ref}"],
        cwd=cwd,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def show_file_at_ref(*, ref: str, path: str, cwd: Path) -> Optional[str]:
    """Return the *path* contents at *ref*, or None if the file did not exist."""
    cmd = ["git", "show", f"{ref}:{path}"]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # File didn't exist at base-ref or path is out of tree -> treat as
        # newly-added file.
        return None
    return proc.stdout


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

def _walk_properties(
    schema: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """
    Walk *schema* recursively and return a flat map of dotted-path -> property
    schema for every node under a `properties` block.

    This is intentionally simple: no $ref resolution, no oneOf/anyOf merging.
    The classifier is conservative — when in doubt the default verdict is
    `breaking`, so missed nesting tends to under-report fields, which is then
    caught by the structural fallback (rule j).
    """
    out: dict[str, dict[str, Any]] = {}

    props = schema.get("properties")
    if isinstance(props, dict):
        for name, child in props.items():
            dotted = f"{prefix}.{name}" if prefix else name
            if isinstance(child, dict):
                out[dotted] = child
                # Recurse into object-typed children (with their own properties).
                out.update(_walk_properties(child, prefix=dotted))

    # Items of an array — descend if it has its own properties.
    items = schema.get("items")
    if isinstance(items, dict):
        out.update(_walk_properties(items, prefix=prefix + "[]"))

    return out


def _walk_required(
    schema: dict[str, Any],
    *,
    prefix: str = "",
) -> set[str]:
    """Return dotted paths to every field listed under a `required` block."""
    out: set[str] = set()

    req = schema.get("required")
    if isinstance(req, list):
        for name in req:
            if isinstance(name, str):
                out.add(f"{prefix}.{name}" if prefix else name)

    props = schema.get("properties")
    if isinstance(props, dict):
        for name, child in props.items():
            if isinstance(child, dict):
                dotted = f"{prefix}.{name}" if prefix else name
                out.update(_walk_required(child, prefix=dotted))

    items = schema.get("items")
    if isinstance(items, dict):
        out.update(_walk_required(items, prefix=prefix + "[]"))

    return out


def _diff_required_fields(
    old: dict[str, Any],
    new: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """
    Return (added_required, removed_required) dotted-path sets.

    Public helper named by the spec; do not rename without updating tests.
    """
    old_req = _walk_required(old)
    new_req = _walk_required(new)
    return (new_req - old_req, old_req - new_req)


def _normalise_type(t: Any) -> set[str]:
    """Normalise a JSON-schema `type` value into a set of type strings."""
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    return set()


def _is_type_narrowed(old_prop: dict[str, Any], new_prop: dict[str, Any]) -> bool:
    """A type is *narrowed* if its allowed-type set strictly shrank."""
    old_t = _normalise_type(old_prop.get("type"))
    new_t = _normalise_type(new_prop.get("type"))
    if not old_t or not new_t:
        return False
    if new_t == old_t:
        return False
    return new_t.issubset(old_t) and new_t != old_t


def _enum_values(prop: dict[str, Any]) -> Optional[list[Any]]:
    e = prop.get("enum")
    if isinstance(e, list):
        return e
    return None


def _is_pattern_tightened(
    old_prop: dict[str, Any],
    new_prop: dict[str, Any],
) -> bool:
    """A pattern is *tightened* iff source string changed AND not allowlisted."""
    op = old_prop.get("pattern")
    np_ = new_prop.get("pattern")
    if not isinstance(op, str) or not isinstance(np_, str):
        return False
    if op == np_:
        return False
    if (op, np_) in KNOWN_PATTERN_SUPERSET_ALLOWLIST:
        return False
    return True


def _is_description_only(
    old_prop: dict[str, Any],
    new_prop: dict[str, Any],
) -> bool:
    """True iff *only* the `description` field differs between two prop nodes."""
    old_keys = set(old_prop.keys()) - {"description"}
    new_keys = set(new_prop.keys()) - {"description"}
    if old_keys != new_keys:
        return False
    for k in old_keys:
        if old_prop.get(k) != new_prop.get(k):
            return False
    return old_prop.get("description") != new_prop.get("description")


def _shallow_keys_changed(
    old: dict[str, Any],
    new: dict[str, Any],
) -> set[str]:
    """Return the set of top-level keys that differ between two schemas."""
    keys = set(old.keys()) | set(new.keys())
    return {k for k in keys if old.get(k) != new.get(k)}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(old: dict[str, Any], new: dict[str, Any]) -> ClassificationResult:
    """
    Classify the diff between *old* and *new* schema dicts.

    Rules apply in the documented order; the first one that fires wins.
    Conservative bias: rule (j) defaults to breaking on unrecognised changes.
    """
    old_props = _walk_properties(old)
    new_props = _walk_properties(new)

    added_paths = sorted(new_props.keys() - old_props.keys())
    removed_paths = sorted(old_props.keys() - new_props.keys())
    common_paths = sorted(old_props.keys() & new_props.keys())

    added_req, removed_req = _diff_required_fields(old, new)

    # ------------------------------------------------------------------ rule a
    # Any required field added -> breaking. We treat both "newly added & required"
    # and "field already existed but became required" as breaking.
    if added_req:
        return ClassificationResult(
            severity=Severity.BREAKING,
            rule="a",
            detail=f"required field(s) added: {sorted(added_req)}",
        )

    # ------------------------------------------------------------------ rule b
    # Any field removed -> removed.
    if removed_paths:
        return ClassificationResult(
            severity=Severity.REMOVED,
            rule="b",
            detail=f"field(s) removed: {removed_paths}",
        )

    # ------------------------------------------------------------------ rule c
    # Any type narrowed (e.g., string -> integer) -> breaking.
    for p in common_paths:
        if _is_type_narrowed(old_props[p], new_props[p]):
            return ClassificationResult(
                severity=Severity.BREAKING,
                rule="c",
                detail=(
                    f"type narrowed at '{p}': "
                    f"{old_props[p].get('type')!r} -> {new_props[p].get('type')!r}"
                ),
            )

    # ------------------------------------------------------------------ rule d
    # Any enum value removed -> breaking.
    for p in common_paths:
        old_enum = _enum_values(old_props[p])
        new_enum = _enum_values(new_props[p])
        if old_enum is not None and new_enum is not None:
            removed_enum = set(map(_jsonable_freeze, old_enum)) - set(
                map(_jsonable_freeze, new_enum)
            )
            if removed_enum:
                return ClassificationResult(
                    severity=Severity.BREAKING,
                    rule="d",
                    detail=f"enum value(s) removed at '{p}': {sorted(map(str, removed_enum))}",
                )

    # ------------------------------------------------------------------ rule e
    # Any pattern tightened -> breaking (unless on superset allowlist).
    for p in common_paths:
        if _is_pattern_tightened(old_props[p], new_props[p]):
            return ClassificationResult(
                severity=Severity.BREAKING,
                rule="e",
                detail=(
                    f"pattern tightened at '{p}': "
                    f"{old_props[p].get('pattern')!r} -> {new_props[p].get('pattern')!r}"
                ),
            )

    # ------------------------------------------------------------------ rule f
    # Any field's `deprecated: true` flag added -> deprecated.
    for p in common_paths:
        old_dep = bool(old_props[p].get("deprecated"))
        new_dep = bool(new_props[p].get("deprecated"))
        if not old_dep and new_dep:
            return ClassificationResult(
                severity=Severity.DEPRECATED,
                rule="f",
                detail=f"deprecated flag added at '{p}'",
            )

    # ------------------------------------------------------------------ rule g
    # Any optional field added -> additive.
    if added_paths:
        # All added must be optional (not in required) — rule (a) handles required.
        return ClassificationResult(
            severity=Severity.ADDITIVE,
            rule="g",
            detail=f"optional field(s) added: {added_paths}",
        )

    # ------------------------------------------------------------------ rule h
    # Any enum value added -> additive.
    for p in common_paths:
        old_enum = _enum_values(old_props[p])
        new_enum = _enum_values(new_props[p])
        if old_enum is not None and new_enum is not None:
            added_enum = set(map(_jsonable_freeze, new_enum)) - set(
                map(_jsonable_freeze, old_enum)
            )
            if added_enum:
                return ClassificationResult(
                    severity=Severity.ADDITIVE,
                    rule="h",
                    detail=f"enum value(s) added at '{p}': {sorted(map(str, added_enum))}",
                )

    # ------------------------------------------------------------------ rule i
    # Description-only change -> additive.
    desc_only_paths = [
        p
        for p in common_paths
        if _is_description_only(old_props[p], new_props[p])
    ]
    if desc_only_paths and not _shallow_structural_change(old, new):
        # Only call it description-only if no other top-level keys changed.
        return ClassificationResult(
            severity=Severity.ADDITIVE,
            rule="i",
            detail=f"description-only change at: {desc_only_paths}",
        )

    # ------------------------------------------------------------------ rule j
    # Default — when in doubt, breaking.
    diff_keys = _shallow_keys_changed(old, new)
    return ClassificationResult(
        severity=Severity.BREAKING,
        rule="j",
        detail=(
            "structural change not matched by rules (a)-(i); "
            f"top-level keys differing: {sorted(diff_keys)}"
        ),
    )


def _jsonable_freeze(v: Any) -> Any:
    """Make a JSON value hashable enough for set membership tests."""
    if isinstance(v, list):
        return tuple(_jsonable_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _jsonable_freeze(val)) for k, val in v.items()))
    return v


def _shallow_structural_change(
    old: dict[str, Any],
    new: dict[str, Any],
) -> bool:
    """
    Detect structural top-level changes that go beyond `properties` /
    `description`. Used as a guard for rule (i): if the schema gained or lost a
    top-level keyword (e.g. `oneOf`, `if`, `additionalProperties`), refuse to
    call the change description-only.
    """
    keys = (set(old.keys()) | set(new.keys())) - {"description", "properties", "required"}
    for k in keys:
        if old.get(k) != new.get(k):
            return True
    return False


# ---------------------------------------------------------------------------
# Filename -> version-tag helpers
# ---------------------------------------------------------------------------

def version_tag_from_path(path: str) -> Optional[str]:
    """
    Extract the canonical version tag from a schema filename.

    `config/schemas/factor_record_v0_1.schema.json` -> "v0.1"
    Returns None if the filename does not match.
    """
    m = VERSION_FROM_FILENAME_RE.search(path)
    if not m:
        return None
    raw = m.group(1)  # e.g., "v0_1"
    # Convert v0_1 -> v0.1
    return raw.replace("_", ".", 1)


# ---------------------------------------------------------------------------
# CHANGELOG parser
# ---------------------------------------------------------------------------

def parse_changelog(text: str) -> list[ChangelogEntry]:
    """
    Parse the CHANGELOG and return entries top-down (file order = newest first).

    Only lines exactly matching the CTO-mandated regex are considered entries.
    Non-matching content (prose, sub-bullets, blank lines) is ignored.
    """
    entries: list[ChangelogEntry] = []
    for idx, line in enumerate(text.splitlines()):
        m = CHANGELOG_ENTRY_RE.match(line)
        if not m:
            continue
        major, minor, date, label = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            sev = Severity.from_label(label)
        except KeyError:
            continue
        entries.append(
            ChangelogEntry(
                version_tag=f"v{major}.{minor}",
                date=date,
                severity=sev,
                line_index=idx,
            )
        )
    return entries


def topmost_entry_for_version(
    entries: list[ChangelogEntry],
    *,
    version_tag: str,
) -> Optional[ChangelogEntry]:
    """Return the first (= topmost = newest) CHANGELOG entry for a version."""
    for e in entries:
        if e.version_tag == version_tag:
            return e
    return None


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    """Resolved configuration for a single gate run."""

    base_ref: str
    head_ref: str
    changelog_path: Path
    schema_glob: str
    repo_root: Path
    json_output: bool

    def matches_schema_glob(self, path: str) -> bool:
        """Return True if *path* matches our schema glob."""
        return fnmatch.fnmatch(path, self.schema_glob)


def collect_changed_schemas(cfg: GateConfig) -> list[str]:
    """Return the list of changed files matching our schema glob."""
    all_changed = list_changed_files(
        base_ref=cfg.base_ref,
        head_ref=cfg.head_ref,
        cwd=cfg.repo_root,
    )
    return [p for p in all_changed if cfg.matches_schema_glob(p)]


def load_schema_pair(
    cfg: GateConfig,
    path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load (old, new) schema dicts for a changed path.

    `old` is the version at base_ref; if the file did not exist at base_ref
    (newly added schema) we treat it as `{}`. `new` is the working-tree copy.
    """
    old_text = show_file_at_ref(ref=cfg.base_ref, path=path, cwd=cfg.repo_root)
    if old_text is None:
        old: dict[str, Any] = {}
    else:
        try:
            old = json.loads(old_text)
        except json.JSONDecodeError as exc:
            raise GitError(f"old schema at {path}@{cfg.base_ref} is not valid JSON: {exc}") from exc

    new_path = cfg.repo_root / path
    if not new_path.exists():
        # File was deleted at HEAD.
        new: dict[str, Any] = {}
    else:
        try:
            new = json.loads(new_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GitError(f"new schema at {path} is not valid JSON: {exc}") from exc

    return old, new


def run_gate(cfg: GateConfig) -> tuple[int, list[GateOutcome]]:
    """
    Run the gate and return (exit_code, per-schema outcomes).

    Exit codes:
        0 -> all changed schemas have an adequate CHANGELOG entry.
        1 -> one or more schemas have NO CHANGELOG entry for their version.
        2 -> CHANGELOG entry exists but classification is too weak.
    """
    changed = collect_changed_schemas(cfg)
    if not changed:
        # Nothing schema-relevant in this diff -> always pass.
        return 0, []

    # CHANGELOG is required input; absence is a hard failure (treat as missing).
    if not cfg.changelog_path.is_file():
        outcomes = [
            GateOutcome(
                diff=SchemaDiff(
                    path=p,
                    version_tag=version_tag_from_path(p) or "?",
                    classification=ClassificationResult(
                        severity=Severity.BREAKING,
                        rule="j",
                        detail="CHANGELOG.md not present at expected path",
                    ),
                ),
                matching_entry=None,
                verdict="missing_entry",
                message=(
                    f"CHANGELOG.md not found at {cfg.changelog_path}; "
                    f"every schema change requires an entry."
                ),
            )
            for p in changed
        ]
        return 1, outcomes

    changelog_text = cfg.changelog_path.read_text(encoding="utf-8")
    entries = parse_changelog(changelog_text)

    outcomes: list[GateOutcome] = []
    worst_exit = 0

    for path in changed:
        version_tag = version_tag_from_path(path)
        if version_tag is None:
            # Filename didn't match the expected pattern — defensive: still
            # treat as missing (the CI gate must not silently let it pass).
            outcomes.append(
                GateOutcome(
                    diff=SchemaDiff(
                        path=path,
                        version_tag="?",
                        classification=ClassificationResult(
                            severity=Severity.BREAKING,
                            rule="j",
                            detail="filename does not match factor_record_v<M>_<N>.schema.json",
                        ),
                    ),
                    matching_entry=None,
                    verdict="missing_entry",
                    message=(
                        f"changed schema {path!r} does not match the "
                        f"factor_record_v<MAJOR>_<MINOR>.schema.json naming "
                        f"convention; cannot resolve a CHANGELOG version."
                    ),
                )
            )
            worst_exit = max(worst_exit, 1)
            continue

        old, new = load_schema_pair(cfg, path)
        classification = classify(old, new)
        diff = SchemaDiff(path=path, version_tag=version_tag, classification=classification)

        entry = topmost_entry_for_version(entries, version_tag=version_tag)
        if entry is None:
            outcomes.append(
                GateOutcome(
                    diff=diff,
                    matching_entry=None,
                    verdict="missing_entry",
                    message=(
                        f"missing CHANGELOG entry for {version_tag} at "
                        f"{cfg.changelog_path} (schema: {path}, "
                        f"required severity >= {classification.severity.label})."
                    ),
                )
            )
            worst_exit = max(worst_exit, 1)
            continue

        if entry.severity < classification.severity:
            outcomes.append(
                GateOutcome(
                    diff=diff,
                    matching_entry=entry,
                    verdict="classification_mismatch",
                    message=(
                        f"classification mismatch for {path}: classifier says "
                        f"{classification.severity.label} (rule {classification.rule}), "
                        f"but topmost CHANGELOG entry for {version_tag} is "
                        f"{entry.severity.label}."
                    ),
                )
            )
            worst_exit = max(worst_exit, 2)
            continue

        outcomes.append(
            GateOutcome(
                diff=diff,
                matching_entry=entry,
                verdict="ok",
                message=(
                    f"OK: {path} classified {classification.severity.label} "
                    f"(rule {classification.rule}); CHANGELOG entry "
                    f"{entry.severity.label} dated {entry.date} satisfies it."
                ),
            )
        )

    return worst_exit, outcomes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def emit_human_report(outcomes: list[GateOutcome], exit_code: int) -> None:
    """Pretty-print the per-schema outcomes to stdout."""
    if not outcomes:
        print(_green("[schema-evolution-check] no factor_record_*.schema.json files changed; nothing to do."))
        return

    print(_bold("[schema-evolution-check] schema migration-note gate"))
    print("=" * 72)

    for o in outcomes:
        if o.verdict == "ok":
            head = _green(f"PASS  {o.diff.path}")
        elif o.verdict == "classification_mismatch":
            head = _yellow(f"FAIL  {o.diff.path}  (classification mismatch)")
        else:
            head = _red(f"FAIL  {o.diff.path}  (missing CHANGELOG entry)")
        print(head)
        print(f"   classifier: {o.diff.classification.severity.label} (rule {o.diff.classification.rule})")
        print(f"   detail    : {o.diff.classification.detail}")
        if o.matching_entry is not None:
            print(
                f"   changelog : {o.matching_entry.version_tag} - "
                f"{o.matching_entry.date} - {o.matching_entry.severity.label} "
                f"(line {o.matching_entry.line_index + 1})"
            )
        else:
            print("   changelog : <none>")
        print(f"   message   : {o.message}")
        print()

    if exit_code == 0:
        print(_green("[schema-evolution-check] all changed schemas have adequate CHANGELOG entries."))
    elif exit_code == 1:
        print(_red("[schema-evolution-check] FAIL: missing CHANGELOG entry."))
    elif exit_code == 2:
        print(_yellow("[schema-evolution-check] FAIL: CHANGELOG classification weaker than detected severity."))
    else:
        print(_red(f"[schema-evolution-check] FAIL: unrecoverable error (exit={exit_code})."))


def emit_machine_report(outcomes: list[GateOutcome], exit_code: int) -> None:
    """Print one line per outcome to stderr in a machine-friendly format."""
    for o in outcomes:
        print(
            f"schema-evolution-check verdict={o.verdict} "
            f"path={o.diff.path} version={o.diff.version_tag} "
            f"classifier={o.diff.classification.severity.label} "
            f"rule={o.diff.classification.rule}",
            file=sys.stderr,
        )
        # Print the human-readable message on stderr too — the test fixture
        # asserts on stderr substring matching.
        print(f"  -> {o.message}", file=sys.stderr)
    print(f"schema-evolution-check exit={exit_code}", file=sys.stderr)


def emit_json_report(outcomes: list[GateOutcome], exit_code: int) -> None:
    """Print a single JSON document on stdout."""
    payload = {
        "exit_code": exit_code,
        "outcomes": [o.to_jsonable() for o in outcomes],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_schema_migration_notes",
        description=(
            "GreenLang Factors schema-evolution CI gate. "
            "Asserts every changed factor_record_*.schema.json has an "
            "adequate CHANGELOG entry."
        ),
    )
    p.add_argument("--base-ref", default="origin/master", help="git base ref (default: origin/master)")
    p.add_argument("--head-ref", default="HEAD", help="git head ref (default: HEAD)")
    p.add_argument(
        "--changelog",
        default="docs/factors/schema/CHANGELOG.md",
        help="path to CHANGELOG.md (default: docs/factors/schema/CHANGELOG.md)",
    )
    p.add_argument(
        "--schema-glob",
        default="config/schemas/factor_record_*.schema.json",
        help="glob for changed schema files (default: config/schemas/factor_record_*.schema.json)",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="path to git repo root; defaults to the script's git toplevel",
    )
    p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="emit JSON to stdout instead of human-readable",
    )
    return p


def _resolve_repo_root(arg: Optional[str]) -> Path:
    """Resolve --repo-root, falling back to `git rev-parse --show-toplevel`."""
    if arg:
        return Path(arg).resolve()
    # Use this script's directory and walk up via git.
    here = Path(__file__).resolve().parent
    try:
        out = _run_git(["rev-parse", "--show-toplevel"], cwd=here)
        return Path(out.strip()).resolve()
    except GitError:
        # Last-ditch fallback: cwd.
        return Path.cwd().resolve()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)

    try:
        repo_root = _resolve_repo_root(args.repo_root)
        cfg = GateConfig(
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            changelog_path=(repo_root / args.changelog).resolve(),
            schema_glob=args.schema_glob,
            repo_root=repo_root,
            json_output=args.json_output,
        )

        exit_code, outcomes = run_gate(cfg)

    except GitError as exc:
        print(f"[schema-evolution-check] git error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover - defensive last-ditch
        print(f"[schema-evolution-check] unexpected error: {exc}", file=sys.stderr)
        return 3

    if cfg.json_output:
        emit_json_report(outcomes, exit_code)
    else:
        emit_human_report(outcomes, exit_code)
    emit_machine_report(outcomes, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
