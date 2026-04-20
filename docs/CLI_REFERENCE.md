# GreenLang CLI Reference

**Version:** 0.3.0
**Last audit:** 2026-04-20
**Entry point:** `gl` (also aliased as `greenlang`) → `greenlang.cli.main:main`

---

## Overview

GreenLang exposes a single top-level `gl` command backed by Typer. Subcommands come from two places:

1. **Inline commands in `greenlang/cli/main.py`** (direct `@app.command()`).
2. **Sub-apps registered via `_safe_add_typer(...)`** (each `cmd_*.py` can expose its own Typer app mounted under a namespace).

Audit found **20 `cmd_*.py` files**, but only **3 are wired** into the current `gl` entry point. The remainder are orphaned Typer/Click apps that are not reachable via `gl`. Resolving this is part of Phase 1 (agent sprawl) and Phase 4 (hosted API) cleanup.

---

## 1. Active commands (reachable via `gl`)

### 1.1 Top-level (defined inline in `main.py`)

| Command | Purpose | Key args / flags |
|---|---|---|
| `gl version` | Show version + homepage | — |
| `gl doctor` | Environment diagnostics; Windows PATH setup | `--setup-path`, `--revert-path`, `--list-backups`, `--verbose` |
| `gl run <pipeline> [input] [cbam_imports] [output_dir]` | Run a pipeline file or the CBAM MVP flow | `--audit` (record in audit ledger), `--dry-run` (CBAM validation only) |
| `gl policy <action> [target]` | **Stub** — only echoes "Policy check passed" | `action` ∈ {check, list, add} |
| `gl verify <artifact>` | Stub verify of artifact provenance/signature | `--sig/-s <signature>` |

> **Note:** `gl policy` and `gl verify` inline in `main.py` are placeholder stubs. Richer implementations exist in `cmd_policy.py` and `cmd_verify.py` but are **not wired** (see §2).

### 1.2 `gl factors` (defined inline in `main.py`)

FY27 Factors catalog subcommands.

| Command | Purpose | Key args |
|---|---|---|
| `gl factors inventory` | Write factor source coverage matrix to JSON | `--out <path>` |
| `gl factors manifest` | Build EditionManifest JSON from built-in factor DB | `--edition-id`, `--out`, `--status`, `--message` |
| `gl factors ingest-builtin` | Load built-in factors into SQLite catalog | `--sqlite`, `--edition-id`, `--label` |
| `gl factors ingest-paths <paths...>` | Normalize JSON files and ingest into SQLite | `--sqlite`, `--edition-id`, `--label`, `--status` |

### 1.3 `gl pack` (from `cmd_pack.py`, registered)

| Command | Purpose |
|---|---|
| `gl pack create` | Create a new pack |
| `gl pack validate` | Validate a pack manifest |
| `gl pack publish` | Publish a pack to the registry |
| `gl pack add` | Add a pack dependency |
| `gl pack remove` | Remove a pack dependency |
| `gl pack info` | Show pack metadata |
| `gl pack list` | List installed packs |
| `gl pack search` | Search the pack registry |
| `gl pack index` | Rebuild local pack index |

### 1.4 `gl v1` (from `cmd_v1.py`, registered — **legacy**)

Legacy v1 platformization commands. Candidate for archival in **Phase 6.2**.

| Command | Purpose |
|---|---|
| `gl v1 status` | v1 runtime status |
| `gl v1 validate-contracts` | v1 contract validation |
| `gl v1 check-policy` | v1 policy checks |
| `gl v1 gate` | v1 release gate |
| `gl v1 full-backend-checks` | Full backend validation |
| `gl v1 run-profile` | Run a v1 profile |
| `gl v1 smoke` | v1 smoke tests |

### 1.5 `gl v2` (from `cmd_v2.py`, registered — **legacy**)

Legacy v2 scale/productization commands. Candidate for archival in **Phase 6.2**.

| Command | Purpose |
|---|---|
| `gl v2 status` | v2 runtime status |
| `gl v2 validate-contracts` | v2 contract validation |
| `gl v2 runtime-checks` | v2 runtime validation |
| `gl v2 docs-check` | v2 docs validation |
| `gl v2 agent-checks` | v2 agent validation |
| `gl v2 connector-checks` | v2 connector validation |
| `gl v2 gate` | v2 release gate |

---

## 2. Orphaned command modules (NOT reachable via `gl` today)

These modules define their own Typer or Click apps but are not mounted into `main.py`. They are reachable only by running `python -m greenlang.cli.<module>` or by separate entry points. **Action:** register, consolidate, or archive each in Phase 1-2.

| Module | Pattern | Commands it would expose | Recommended namespace |
|---|---|---|---|
| `cmd_agent.py` | Typer | register, list, (6 more) | `gl agent` |
| `cmd_init.py` | Typer | init | `gl init` |
| `cmd_init_agent.py` | Typer | init-agent | `gl agent init` (merge with above) |
| `cmd_generate.py` | Typer | generate | `gl generate` |
| `cmd_decarbonization.py` | Typer | 2+ commands | `gl decarb` |
| `cmd_doctor.py` | Typer | fix | merge into inline `gl doctor` |
| `cmd_demo.py` | Typer | run offline demo | `gl demo` |
| `cmd_rbac.py` | Typer | 9 commands | `gl rbac` |
| `cmd_pack_new.py` | Typer | list, info, add, remove, validate | **duplicate of `cmd_pack`** — archive |
| `cmd_policy.py` | Typer | check, run, list, add, show, validate | **replace inline stub** — register as `gl policy` |
| `cmd_registry.py` | Typer | publish, list, info, certify | `gl registry` |
| `cmd_sbom.py` | Typer | generate, verify, list, diff | `gl sbom` |
| `cmd_validate.py` | Typer | 2 commands | `gl validate` |
| `cmd_verify.py` | Typer | sbom, provenance | **replace inline stub** — register as `gl verify` |
| `cmd_run.py` | Typer | list, info | merge into inline `gl run` as subcommands |
| `cmd_capabilities.py` | **Click** (not Typer) | lint + more | convert to Typer, expose as `gl capabilities` |
| `cmd_schema.py` | **Click** (not Typer) | schema mgmt | convert to Typer, expose as `gl schema` |
| `rag_commands.py` | Typer | 5 commands | `gl rag` |
| `generate.py` | Typer | create, validate, info | **duplicate of `cmd_generate.py`** — dedupe |
| `main_new.py` | Typer | parallel main.py with 6 commands | **refactor candidate** — reconcile with `main.py` or archive |

---

## 3. Agent Factory CLI (separate subtree)

`greenlang/cli/agent_factory/` defines a distinct Typer app tree (`cli_main.py`) with commands: `new`, `validate`, `test`, `certify`, `deploy`, `list`, `diff`, `info`. Supporting modules: `certify_command.py`, `create_command.py`, `deploy_command.py`, `template_command.py`, `test_command.py`, `validate_command.py`, `productivity_helpers.py`.

**Status:** not mounted into `gl` today. Recommend exposing as `gl agent factory <subcmd>` or `gl factory <subcmd>` once **Phase 1.2** declares the canonical agent base class.

---

## 4. Missing commands (FY27 targets)

These commands are named in the FY27 Reality Analysis but do not exist yet. Each corresponds to a to-do-list item.

| Command | Source task | Purpose |
|---|---|---|
| `gl ledger record / verify / export` | Phase 2.1 | Climate Ledger writes + chain verification |
| `gl evidence bundle --case <id>` | Phase 2.2 | Evidence Vault bundle export (signed zip) |
| `gl evidence list / export` | Phase 2.2 | Vault management |
| `gl entity register / tree / query` | Phase 2.3 | Entity Graph CRUD |
| `gl policy applies-to / evaluate` | Phase 2.4 | Policy Graph applicability API |
| `gl connect test / extract / list` | Phase 2.5 | Connect enterprise integrations |
| `gl scope compute` | Phase 3.2 | Unified Scope Engine entry point |
| `gl comply run <request.json>` | Phase 3.1 | Comply orchestrator |

---

## 5. Consolidation recommendations

**Short term (Phase 0–1):**

1. Delete or archive duplicate modules: `cmd_pack_new.py` (dup of `cmd_pack`), `generate.py` (dup of `cmd_generate`), `main_new.py` (parallel main).
2. Replace inline stubs in `main.py` (`policy`, `verify`) by registering the richer `cmd_policy.py` and `cmd_verify.py` Typer apps.
3. Convert `cmd_capabilities.py` and `cmd_schema.py` from Click to Typer for consistency with the rest of the CLI.
4. Mount the 15+ orphaned Typer apps into `main.py` via additional `_safe_add_typer` calls, using the namespaces suggested in §2.

**Medium term (Phase 2–3):**

5. Add new command groups for L2/L3 products: `gl ledger`, `gl evidence`, `gl entity`, `gl connect`, `gl scope`, `gl comply`.
6. Standardize help text and argument patterns so every subcommand documents inputs, outputs, and provenance implications in a consistent format.

**Long term (Phase 6):**

7. Archive `cmd_v1.py`, `cmd_v2.py`, and their backends (`greenlang/v1/`, `greenlang/v2/`) to `_archive/08_legacy_v1_v2_runtime/` after grep-confirmed zero callers, and remove `gl v1` / `gl v2` subcommands.

---

## 6. How to extend the CLI

1. Create `greenlang/cli/cmd_<yourtopic>.py` with a Typer app:
   ```python
   import typer
   app = typer.Typer(help="Short description")

   @app.command("do-thing")
   def do_thing(arg: str):
       ...
   ```
2. Register it in `greenlang/cli/main.py`:
   ```python
   _safe_add_typer("cmd_<yourtopic>", "<topic>", "Help text shown in `gl --help`")
   ```
3. Add tests in `tests/cli/test_cmd_<yourtopic>.py`.

---

## 7. Summary counts

| Category | Count |
|---|---|
| Commands reachable via `gl` (top-level + factors + pack + v1 + v2) | ~30 |
| Orphaned `cmd_*.py` modules | 17 |
| Missing commands required for FY27 | 12 |
| Legacy commands slated for archival | 2 groups (`gl v1`, `gl v2`) |
| Canonical CLI entry point | `gl` (alias: `greenlang`) |

> Gaps between this reference and reality indicate drift. Re-run `grep -n "@app.command\|app = typer.Typer" greenlang/cli/cmd_*.py greenlang/cli/main.py` and update this file when adding or removing commands.
