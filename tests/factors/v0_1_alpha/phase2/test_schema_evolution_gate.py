"""
Self-tests for scripts/ci/check_schema_migration_notes.py
(GreenLang Factors schema-evolution CI gate, Phase 2 §2.6 / WS9-B).

These tests spin up a hermetic temporary git repo and exercise the gate
end-to-end against synthetic schema diffs. No global git config is mutated
(every git invocation pins user.email / user.name with `git -c ...`), and
no network calls are made.

Test scenarios (per CTO brief):
    A - additive optional field + matching `additive` CHANGELOG -> exit 0
    B - new required field + missing CHANGELOG entry           -> exit 1
    C - field rename + weaker `additive` CHANGELOG entry       -> exit 2
    D - regex pattern tightened + `breaking` CHANGELOG entry   -> exit 0
    E - description-only change + `additive` CHANGELOG entry   -> exit 0
    F - unknown structural change ("when in doubt"):
          F1 - paired with `breaking` CHANGELOG entry          -> exit 0
          F2 - no CHANGELOG entry                              -> exit 1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pytest


# ---------------------------------------------------------------------------
# Locate the script under test (absolute path, never relies on cwd).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "check_schema_migration_notes.py"
SCHEMA_RELPATH = "config/schemas/factor_record_v0_1.schema.json"
CHANGELOG_RELPATH = "docs/factors/schema/CHANGELOG.md"


# ---------------------------------------------------------------------------
# Synthetic minimal schema (NOT the real frozen one — we never touch that).
# ---------------------------------------------------------------------------

INITIAL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.test/factor_record_v0_1.schema.json",
    "title": "Synthetic factor record (test fixture v0.1)",
    "type": "object",
    "additionalProperties": False,
    "required": ["urn", "name", "value"],
    "properties": {
        "urn": {
            "type": "string",
            "pattern": "^[a-z]+$",
            "description": "Canonical id (initial).",
        },
        "name": {
            "type": "string",
            "description": "Display name (initial).",
        },
        "value": {
            "type": "number",
            "description": "Numeric value (initial).",
        },
        "category": {
            "type": "string",
            "enum": ["a", "b", "c"],
            "description": "Category enum (initial).",
        },
    },
}


INITIAL_CHANGELOG = (
    "# GreenLang Factors Schema CHANGELOG (test fixture)\n"
    "\n"
    "Anchor format (CTO-mandated, ASCII only):\n"
    "    [hash][hash] vMAJOR.MINOR - YYYY-MM-DD - additive|breaking|deprecated|removed\n"
    "\n"
    "<!-- entries are prepended above this line -->\n"
)


# ---------------------------------------------------------------------------
# Hermetic git-repo fixture
# ---------------------------------------------------------------------------

@dataclass
class RepoCtx:
    """Helpers for manipulating the temp git repo from inside a test."""

    root: Path
    schema_path: Path
    changelog_path: Path

    # ----- git plumbing ----------------------------------------------------
    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command in the repo with hermetic identity."""
        full = [
            "git",
            "-c", "user.email=ci@test.local",
            "-c", "user.name=CI Bot",
            "-c", "commit.gpgsign=false",
            *args,
        ]
        return subprocess.run(
            full,
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )

    def commit_all(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-m", message)

    def head_sha(self) -> str:
        return self.git("rev-parse", "HEAD").stdout.strip()

    def base_sha(self) -> str:
        # The base is always the repo's first commit in our tests.
        return self.git("rev-list", "--max-parents=0", "HEAD").stdout.strip()

    # ----- file mutation ---------------------------------------------------
    def write_schema(self, schema: dict[str, Any]) -> None:
        self.schema_path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def read_schema(self) -> dict[str, Any]:
        return json.loads(self.schema_path.read_text(encoding="utf-8"))

    def append_changelog(self, body: str) -> None:
        """
        Prepend a new entry so it lands at column 0 above any existing
        entries. The header section (anything above the first line that is
        exactly ``## v<digit>`` at column 0) is preserved.
        """
        existing = self.changelog_path.read_text(encoding="utf-8")
        lines = existing.splitlines()

        # Find the index of the first column-0 entry anchor (matches the
        # CTO regex shape, not just any `## v`).
        import re as _re

        anchor = _re.compile(
            r"^## v\d+\.\d+ - \d{4}-\d{2}-\d{2} - "
            r"(additive|breaking|deprecated|removed)\s*$"
        )
        insert_at = len(lines)  # default: append to the end
        for i, line in enumerate(lines):
            if anchor.match(line):
                insert_at = i
                break

        body_lines = body.rstrip("\n").splitlines()
        # Ensure a blank line separates the new entry from what follows.
        new_block = body_lines + [""]
        new_lines = lines[:insert_at] + new_block + lines[insert_at:]
        new_text = "\n".join(new_lines)
        if not new_text.endswith("\n"):
            new_text += "\n"
        self.changelog_path.write_text(new_text, encoding="utf-8")

    def replace_changelog(self, body: str) -> None:
        self.changelog_path.write_text(body, encoding="utf-8")

    # ----- gate runner -----------------------------------------------------
    def run_gate(
        self,
        *,
        base_ref: Optional[str] = None,
        head_ref: str = "HEAD",
        json_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Invoke the gate against this repo. Returns CompletedProcess."""
        if base_ref is None:
            base_ref = self.base_sha()

        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-ref", base_ref,
            "--head-ref", head_ref,
            "--changelog", CHANGELOG_RELPATH,
            "--schema-glob", "config/schemas/factor_record_*.schema.json",
            "--repo-root", str(self.root),
        ]
        if json_output:
            cmd.append("--json")

        env = os.environ.copy()
        env["NO_COLOR"] = "1"  # keep the report parseable in tests
        return subprocess.run(
            cmd,
            cwd=str(self.root),
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[RepoCtx]:
    """Spawn a tmp git repo seeded with a synthetic schema + CHANGELOG."""
    root = tmp_path / "repo"
    root.mkdir()

    # Layout
    (root / "config" / "schemas").mkdir(parents=True)
    (root / "docs" / "factors" / "schema").mkdir(parents=True)

    schema_path = root / SCHEMA_RELPATH
    changelog_path = root / CHANGELOG_RELPATH

    ctx = RepoCtx(root=root, schema_path=schema_path, changelog_path=changelog_path)

    # `git init` on the repo (no global config writes).
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )

    ctx.write_schema(INITIAL_SCHEMA)
    changelog_path.write_text(INITIAL_CHANGELOG, encoding="utf-8")
    ctx.commit_all("seed: initial schema + CHANGELOG")

    yield ctx


# ---------------------------------------------------------------------------
# Sanity check: script is importable / has a main()
# ---------------------------------------------------------------------------

def test_script_is_present_and_executable() -> None:
    """The CI script must exist on disk before any scenario runs."""
    assert SCRIPT_PATH.is_file(), f"missing CI gate script at {SCRIPT_PATH}"


# ---------------------------------------------------------------------------
# Scenario A - additive optional field (exit 0)
# ---------------------------------------------------------------------------

def test_scenario_a_additive_optional_field(repo: RepoCtx) -> None:
    """Add an optional field; CHANGELOG carries an `additive` entry."""
    schema = repo.read_schema()
    schema["properties"]["new_optional"] = {
        "type": "string",
        "description": "Newly-added optional field.",
    }
    repo.write_schema(schema)
    repo.append_changelog(
        "## v0.1 - 2026-04-26 - additive\n"
        "Add optional `new_optional` string field.\n"
    )
    repo.commit_all("schema: add optional field new_optional")

    result = repo.run_gate()

    assert result.returncode == 0, (
        f"Scenario A expected exit 0, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The classifier must have selected rule (g).
    assert "rule=g" in result.stderr or "rule g" in result.stdout, (
        f"Expected rule (g) match for additive optional field.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Scenario B - new required field, no CHANGELOG entry (exit 1)
# ---------------------------------------------------------------------------

def test_scenario_b_breaking_no_entry(repo: RepoCtx) -> None:
    """Add a required field; do not touch CHANGELOG."""
    schema = repo.read_schema()
    schema["properties"]["new_required"] = {
        "type": "string",
        "description": "Newly-added required field.",
    }
    schema["required"] = list(schema["required"]) + ["new_required"]
    repo.write_schema(schema)
    repo.commit_all("schema: add required field new_required")

    result = repo.run_gate()

    assert result.returncode == 1, (
        f"Scenario B expected exit 1, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "missing CHANGELOG entry" in result.stderr, (
        f"Expected stderr to mention 'missing CHANGELOG entry'.\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Scenario C - field rename = breaking, but CHANGELOG says additive (exit 2)
# ---------------------------------------------------------------------------

def test_scenario_c_classification_mismatch(repo: RepoCtx) -> None:
    """Rename a field (= breaking by rule a or b) but flag it `additive`."""
    schema = repo.read_schema()
    # Rename "category" -> "kind" (remove + add). Required list updated to keep
    # rename "clean" but rule (b) still fires because a property was removed.
    cat_def = schema["properties"].pop("category")
    schema["properties"]["kind"] = cat_def
    repo.write_schema(schema)
    repo.append_changelog(
        "## v0.1 - 2026-04-26 - additive\n"
        "Renamed `category` -> `kind` (mislabelled).\n"
    )
    repo.commit_all("schema: rename category -> kind")

    result = repo.run_gate()

    assert result.returncode == 2, (
        f"Scenario C expected exit 2, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "classification mismatch" in result.stderr, (
        f"Expected stderr to mention 'classification mismatch'.\n"
        f"stderr:\n{result.stderr}"
    )
    assert SCHEMA_RELPATH in result.stderr, (
        f"Expected stderr to mention the changed schema path "
        f"{SCHEMA_RELPATH!r}.\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Scenario D - pattern tightened, properly labelled `breaking` (exit 0)
# ---------------------------------------------------------------------------

def test_scenario_d_pattern_tightened_breaking(repo: RepoCtx) -> None:
    """Tighten a regex pattern and flag the change as `breaking`."""
    schema = repo.read_schema()
    schema["properties"]["urn"]["pattern"] = "^[a-z]{1,5}$"
    repo.write_schema(schema)
    repo.append_changelog(
        "## v0.1 - 2026-04-26 - breaking\n"
        "Tighten urn pattern from ^[a-z]+$ to ^[a-z]{1,5}$.\n"
    )
    repo.commit_all("schema: tighten urn pattern")

    result = repo.run_gate()

    assert result.returncode == 0, (
        f"Scenario D expected exit 0, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The classifier must have selected rule (e).
    assert "rule=e" in result.stderr or "rule e" in result.stdout, (
        f"Expected rule (e) match for tightened pattern.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Scenario E - description-only change, labelled `additive` (exit 0)
# ---------------------------------------------------------------------------

def test_scenario_e_description_only(repo: RepoCtx) -> None:
    """Change only a description string and label it `additive`."""
    schema = repo.read_schema()
    schema["properties"]["name"]["description"] = "Display name (clarified wording)."
    repo.write_schema(schema)
    repo.append_changelog(
        "## v0.1 - 2026-04-26 - additive\n"
        "Doc-only: clarify `name` description.\n"
    )
    repo.commit_all("schema: clarify name description")

    result = repo.run_gate()

    assert result.returncode == 0, (
        f"Scenario E expected exit 0, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The classifier must have selected rule (i).
    assert "rule=i" in result.stderr or "rule i" in result.stdout, (
        f"Expected rule (i) match for description-only change.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Scenario F - "when in doubt" structural change defaults to breaking
# ---------------------------------------------------------------------------

def _apply_unknown_structural_change(repo: RepoCtx) -> None:
    """
    Add an unknown top-level keyword the classifier cannot match against any
    of rules (a)-(i). `propertyNames` is a valid JSON-Schema 2020-12 keyword
    that none of our explicit rules examine, so it falls through to rule (j).
    """
    schema = repo.read_schema()
    schema["propertyNames"] = {"pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"}
    repo.write_schema(schema)


def test_scenario_f1_when_in_doubt_with_breaking_entry(repo: RepoCtx) -> None:
    """Unknown structural change + `breaking` CHANGELOG -> exit 0."""
    _apply_unknown_structural_change(repo)
    repo.append_changelog(
        "## v0.1 - 2026-04-26 - breaking\n"
        "Add propertyNames pattern restricting permitted property identifiers.\n"
    )
    repo.commit_all("schema: add propertyNames pattern")

    result = repo.run_gate()

    assert result.returncode == 0, (
        f"Scenario F1 expected exit 0, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The classifier must have selected rule (j).
    assert "rule=j" in result.stderr or "rule j" in result.stdout, (
        f"Expected rule (j) match for unknown structural change.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_scenario_f2_when_in_doubt_without_entry(repo: RepoCtx) -> None:
    """Unknown structural change + no CHANGELOG entry -> exit 1."""
    _apply_unknown_structural_change(repo)
    repo.commit_all("schema: add propertyNames pattern (undocumented)")

    result = repo.run_gate()

    assert result.returncode == 1, (
        f"Scenario F2 expected exit 1, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "missing CHANGELOG entry" in result.stderr, (
        f"Expected stderr to mention 'missing CHANGELOG entry'.\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Bonus: --json output is well-formed and machine-parseable.
# ---------------------------------------------------------------------------

def test_json_output_is_machine_parseable(repo: RepoCtx) -> None:
    """The --json flag emits a single parseable JSON document on stdout."""
    schema = repo.read_schema()
    schema["properties"]["new_optional"] = {"type": "string", "description": "x"}
    repo.write_schema(schema)
    repo.append_changelog("## v0.1 - 2026-04-26 - additive\nAdd optional field.\n")
    repo.commit_all("schema: add optional")

    result = repo.run_gate(json_output=True)
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 0
    assert isinstance(payload["outcomes"], list)
    assert len(payload["outcomes"]) == 1
    assert payload["outcomes"][0]["verdict"] == "ok"
    assert payload["outcomes"][0]["classification"]["rule"] == "g"
