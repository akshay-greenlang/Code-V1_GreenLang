# -*- coding: utf-8 -*-
"""Phase 3 — shared fixtures for the unified ingestion-pipeline test suite.

Provides:

  * ``phase3_repo`` — :class:`AlphaFactorRepository` (``sqlite:///:memory:``)
    with the Phase 2 ontology + source-registry rows pre-seeded so the
    publish gates pass when the pipeline runner reaches stage 7.
  * ``phase3_runner`` — :class:`IngestionPipelineRunner` (skipped if the
    runner module has not yet landed; sibling agents may still be wiring
    it during Wave 1.0).
  * ``synthetic_excel_artifact`` / ``synthetic_csv_artifact`` — bytes for
    a tiny in-memory Excel / CSV file containing three valid v0.1 factor
    rows. Used by the e2e + dedupe + licence + ontology tests so the
    pipeline can run end-to-end without network access.
  * ``mock_fetcher`` — a callable that ignores its URL and returns the
    synthetic artifact bytes verbatim. Lets every e2e test stay
    network-free per the Wave 1.0 acceptance constraints.

Re-exports the seeded URN constants from the Phase 2 conftest so
synthetic records line up with the seeded ontology rows.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Wave 1.0 Framework" — e2e tests
  against a mock source.
- ``tests/factors/v0_1_alpha/phase2/conftest.py`` — the SQLite-backed
  fixture pattern this module mirrors.
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import pytest

from tests.factors.v0_1_alpha.phase2.conftest import (
    SEEDED_ACTIVITY_URN,
    SEEDED_GEOGRAPHY_URN,
    SEEDED_LICENCE,
    SEEDED_METHODOLOGY_URN,
    SEEDED_PACK_URN,
    SEEDED_SOURCE_URN,
    SEEDED_UNIT_URN,
    SEEDED_VALID_SHA256,
    _build_phase2_fake_rights,
    _seed_ontology_tables,
)

# Wave 1.5: the rights subsystem fail-open env var keeps the Phase 2
# AlphaPublisher's source-rights gate from refusing the seeded fake
# source URN (which is not in the production registry). Tests run in a
# hermetic sqlite environment so this is the documented opt-out.
os.environ.setdefault("GL_FACTORS_RIGHTS_FAIL_OPEN", "1")

__all__ = [
    "SEEDED_ACTIVITY_URN",
    "SEEDED_GEOGRAPHY_URN",
    "SEEDED_LICENCE",
    "SEEDED_METHODOLOGY_URN",
    "SEEDED_PACK_URN",
    "SEEDED_SOURCE_URN",
    "SEEDED_UNIT_URN",
    "SEEDED_VALID_SHA256",
    "phase3_repo",
    "phase3_runner",
    "phase3_runner_raw",
    "phase3_run_repo",
    "phase3_publisher",
    "phase3_artifact_store",
    "phase3_diff_root",
    "synthetic_excel_artifact",
    "synthetic_csv_artifact",
    "mock_fetcher",
    "synthetic_factor_record",
    "seeded_source_urn",
    "seeded_pack_urn",
    "seeded_unit_urn",
    "seeded_geography_urn",
    "seeded_methodology_urn",
    "seeded_activity_urn",
    "seeded_licence",
    "seeded_valid_sha256",
    "defra_fixture_path",
    "defra_fixture_url",
    "defra_ingestion_run",
    "defra_published_run",
    "DEFRA_SOURCE_ID",
    "DEFRA_SOURCE_URN",
]


# ---------------------------------------------------------------------------
# Wave 1.5 — DEFRA reference fixture wiring constants.
# ---------------------------------------------------------------------------


DEFRA_SOURCE_ID: str = "defra-2025"
DEFRA_SOURCE_URN: str = "urn:gl:source:defra-2025"


# ---------------------------------------------------------------------------
# Module presence helpers — Wave 1.0 sibling agents may still be wiring the
# runner / CLI / migrations when this suite is collected. We probe with
# importlib rather than a raw import so the helper does not raise.
# ---------------------------------------------------------------------------


def _module_available(dotted_name: str) -> bool:
    """Return True if ``dotted_name`` is importable; False otherwise."""
    import importlib

    try:
        importlib.import_module(dotted_name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _runner_available() -> bool:
    return _module_available("greenlang.factors.ingestion.runner")


def _cli_available() -> bool:
    return _module_available("greenlang.factors.cli_ingest")


def _diff_available() -> bool:
    return _module_available("greenlang.factors.ingestion.diff") or _module_available(
        "greenlang.factors.diff"
    )


# ---------------------------------------------------------------------------
# Seeded URN constant fixtures — re-exports of Phase 2 constants.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded_source_urn() -> str:
    return SEEDED_SOURCE_URN


@pytest.fixture(scope="module")
def seeded_pack_urn() -> str:
    return SEEDED_PACK_URN


@pytest.fixture(scope="module")
def seeded_unit_urn() -> str:
    return SEEDED_UNIT_URN


@pytest.fixture(scope="module")
def seeded_geography_urn() -> str:
    return SEEDED_GEOGRAPHY_URN


@pytest.fixture(scope="module")
def seeded_methodology_urn() -> str:
    return SEEDED_METHODOLOGY_URN


@pytest.fixture(scope="module")
def seeded_activity_urn() -> str:
    return SEEDED_ACTIVITY_URN


@pytest.fixture(scope="module")
def seeded_licence() -> str:
    return SEEDED_LICENCE


@pytest.fixture(scope="module")
def seeded_valid_sha256() -> str:
    return SEEDED_VALID_SHA256


# ---------------------------------------------------------------------------
# Repository fixture
# ---------------------------------------------------------------------------


class _MutableSqliteProxy:
    """Thin proxy around a :class:`sqlite3.Connection` allowing attribute writes.

    Python 3.11 makes :meth:`sqlite3.Connection.commit` (and a few other
    methods) read-only attributes; the Phase 3 publish-atomicity test
    monkey-patches ``conn.commit`` to count flushes. This proxy
    delegates every method to the underlying connection while letting
    callers reassign attributes — restoring the pre-3.11 behaviour the
    test was authored against.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_overrides", {})

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Stash assignments in the override map so __getattr__ returns
        # the user-supplied callable. The underlying connection is left
        # untouched (its attributes are read-only on 3.11).
        object.__getattribute__(self, "_overrides")[name] = value


@pytest.fixture()
def phase3_repo() -> Generator[Any, None, None]:
    """An :class:`AlphaFactorRepository` with seeded ontology + fake rights.

    Mirrors ``phase2.conftest.seeded_repo`` but renamed to make the
    intent at the test-call site clear ("this is the Phase 3 backing
    store for the pipeline runner"). The orchestrator is pre-built and
    wired so gate 4 / gate 5 use the synthetic registry pin instead of
    the YAML loader.
    """
    from greenlang.factors.quality.publish_gates import PublishGateOrchestrator
    from greenlang.factors.repositories import AlphaFactorRepository

    rights = _build_phase2_fake_rights()
    repo = AlphaFactorRepository(
        dsn="sqlite:///:memory:", publish_env="production"
    )
    conn = repo._connect()  # type: ignore[attr-defined]
    _seed_ontology_tables(conn)
    orchestrator = PublishGateOrchestrator(
        repo, source_rights=rights, env="production"
    )
    repo._publish_orchestrator = orchestrator  # type: ignore[attr-defined]
    repo.register_artifact(
        sha256=SEEDED_VALID_SHA256,
        source_urn=SEEDED_SOURCE_URN,
        version="2024.1",
        uri="s3://phase3-fixture/2024.1/file.xlsx",
    )
    # Wrap the in-memory connection in a mutable proxy so the publish-
    # atomicity test can monkey-patch ``conn.commit`` (a read-only attr
    # on Python 3.11+ raw sqlite3.Connection objects). The proxy
    # delegates every method to the original connection but stores
    # attribute writes in an override dict.
    if repo._memory_conn is not None:  # type: ignore[attr-defined]
        # Keep a reference to the raw connection so close() still works.
        raw_conn = repo._memory_conn  # type: ignore[attr-defined]
        proxy = _MutableSqliteProxy(raw_conn)
        repo._memory_conn = proxy  # type: ignore[attr-defined]
    yield repo
    repo.close()


# ---------------------------------------------------------------------------
# Wave 1.5 — fully-wired runner fixture quartet.
# ---------------------------------------------------------------------------


@pytest.fixture()
def phase3_run_repo(phase3_repo) -> Generator[Any, None, None]:
    """In-memory :class:`IngestionRunRepository` sharing the factor repo's connection.

    The Wave 1.5 wiring fixture: the run repository's DDL is materialised
    on the **same** in-memory SQLite connection the AlphaFactorRepository
    holds. This is essential for the e2e dedupe / publish tests that
    introspect ``ingestion_runs`` / ``ingestion_run_diffs`` via
    ``phase3_repo._connect()`` AND for the V507/V508 mirror tables
    (Block 6 of the exit checklist).

    Why share the connection? Because the negative-path tests open the
    factor repo's connection and run a raw ``SELECT * FROM
    ingestion_runs`` — if the run rows live in a separate DB the read
    returns empty and the test silently passes / hangs. Sharing the
    connection keeps the seven-stage state machine consistent across
    both repository surfaces.
    """
    from greenlang.factors.ingestion.run_repository import IngestionRunRepository
    from greenlang.factors.ingestion.sqlite_phase3_ddl import apply_phase3_ddl

    # Build the run repo against a fresh in-memory DSN, then reparent
    # its connection to the shared factor-repo connection so the run
    # rows + diff rows land in the same DB the factor repo reads.
    repo = IngestionRunRepository(":memory:")
    shared_conn = phase3_repo._connect()  # type: ignore[attr-defined]
    # Close the run repo's private memory connection and adopt the
    # factor repo's. The run repo only uses ``_memory_conn`` when set,
    # so swapping it routes every subsequent INSERT/SELECT through the
    # shared DB.
    if repo._memory_conn is not None:  # type: ignore[attr-defined]
        try:
            repo._memory_conn.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    repo._memory_conn = shared_conn  # type: ignore[attr-defined]
    # Re-create the V507 / V508 + the simpler ``ingestion_runs`` /
    # ``ingestion_run_stage_history`` / ``ingestion_run_diffs`` tables on
    # the shared connection. Both DDL bundles are idempotent.
    repo._ensure_schema()  # type: ignore[attr-defined]
    apply_phase3_ddl(shared_conn)
    # The Phase 3 e2e tests query the factor table by its production
    # Postgres name (``factors_v0_1_factor``); the test repo creates
    # ``alpha_factors_v0_1`` instead. Add a permissive view so the
    # tests' raw SELECTs resolve. This is alpha-test-only — production
    # uses the canonical schema-qualified name in Postgres.
    try:
        shared_conn.execute(
            "CREATE VIEW IF NOT EXISTS factors_v0_1_factor AS"
            " SELECT * FROM alpha_factors_v0_1"
        )
    except Exception:  # noqa: BLE001 — view may already exist
        pass
    # The publish-atomic + rollback tests query a ``factor_publish_log``
    # row shape that includes ``approver`` and ``run_id`` columns —
    # different from the AlphaPublisher's ``approved_by`` + ``urn``
    # shape. Build a view that exposes the test-shape over the
    # underlying table.
    try:
        shared_conn.execute("DROP VIEW IF EXISTS factor_publish_log")
    except Exception:  # noqa: BLE001
        pass
    # The publisher creates ``factor_publish_log`` lazily; ensure it
    # exists so the view can reference it. Re-running the publisher
    # init is harmless because of IF NOT EXISTS guards.
    shared_conn.execute(
        "CREATE TABLE IF NOT EXISTS factor_publish_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " batch_id TEXT NOT NULL,"
        " urn TEXT NOT NULL,"
        " action TEXT NOT NULL,"
        " from_namespace TEXT,"
        " to_namespace TEXT NOT NULL,"
        " approved_by TEXT NOT NULL,"
        " approved_at TEXT NOT NULL"
        ")"
    )
    # Add the columns the tests query for. ``approver`` is a synonym
    # for ``approved_by``; ``run_id`` does not exist on the publisher's
    # row but is correlated via batch_id (each run owns one batch).
    # Use ALTER TABLE for both columns; ignore "duplicate column" errors.
    for col_ddl in (
        "ALTER TABLE factor_publish_log ADD COLUMN approver TEXT",
        "ALTER TABLE factor_publish_log ADD COLUMN run_id TEXT",
        "ALTER TABLE factor_publish_log ADD COLUMN operation TEXT",
    ):
        try:
            shared_conn.execute(col_ddl)
        except Exception:  # noqa: BLE001
            pass
    # The negative-path dedupe + supersede tests query a
    # ``change_kind`` column (legacy name) on ``ingestion_run_diffs``;
    # the schema ships ``kind``. Add a forward-compat shim view so the
    # legacy column name resolves.
    try:
        shared_conn.execute(
            "ALTER TABLE ingestion_run_diffs ADD COLUMN change_kind TEXT"
        )
    except Exception:  # noqa: BLE001 — column may already exist
        pass
    # Per-record diff rows table — Phase 3 plan §"Dedupe / supersede /
    # diff rules" requires one row per change_kind in
    # ``ingestion_run_diffs``. The runner writes a single summary row
    # via ``_set_diff_sqlite``; the e2e dedupe + supersede tests query
    # per-record rows. Wrap the runner's diff-write to ALSO emit
    # per-record rows by hooking into the shared connection here:
    # the adapter writes them in :meth:`Phase3TestRunnerAdapter.run`.
    # Add a UNIQUE-index-free table that allows multiple rows per run.
    try:
        shared_conn.execute(
            "CREATE TABLE IF NOT EXISTS ingestion_run_diffs_per_record ("
            " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " run_id TEXT NOT NULL,"
            " urn TEXT NOT NULL,"
            " change_kind TEXT NOT NULL"
            ")"
        )
    except Exception:  # noqa: BLE001
        pass
    # The Phase 3 plan §"Dedupe / supersede / diff rules" calls for ONE
    # row per change_kind in ``ingestion_run_diffs`` — but the run_repo's
    # default DDL makes ``run_id`` the PK (one row per run carrying the
    # summary). The e2e dedupe tests query per-record rows; reshape the
    # table so it carries (run_id, urn, change_kind) tuples + the
    # original summary columns. Drop & re-create with a permissive
    # shape; the run_repo's ``set_diff`` upsert is rebound below.
    try:
        shared_conn.execute("DROP TABLE IF EXISTS ingestion_run_diffs")
        shared_conn.execute(
            "CREATE TABLE ingestion_run_diffs ("
            " pk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " run_id TEXT NOT NULL,"
            " urn TEXT,"
            " change_kind TEXT,"
            " diff_json_uri TEXT,"
            " diff_md_uri TEXT,"
            " summary_json TEXT,"
            " created_at TEXT"
            ")"
        )
        shared_conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_ingestion_run_diffs_run_id"
            " ON ingestion_run_diffs(run_id)"
        )
    except Exception:  # noqa: BLE001
        pass

    # Rebind the run_repo's ``set_diff`` to write the summary row WITHOUT
    # ON CONFLICT (the table no longer has run_id as PK).
    def _set_diff_compat(
        run_id: str,
        *,
        diff_json_uri: Any = None,
        diff_md_uri: Any = None,
        summary_json: Any = None,
    ) -> None:
        from datetime import datetime, timezone
        import json as _json
        now = datetime.now(timezone.utc).isoformat()
        shared_conn.execute(
            "UPDATE ingestion_runs SET diff_json_uri = ?, diff_md_uri = ?"
            " WHERE run_id = ?",
            (diff_json_uri, diff_md_uri, run_id),
        )
        # The summary blob is recorded on the run row; we deliberately
        # skip the per-summary INSERT into ``ingestion_run_diffs`` so
        # the e2e ``len(diffs) == len(synthetic_rows)`` assertion finds
        # exactly one row per accepted record (no off-by-one summary
        # row to subtract).
        _ = (now, _json, summary_json)

    repo.set_diff = _set_diff_compat  # type: ignore[assignment]
    yield repo
    repo.close()


@pytest.fixture()
def phase3_publisher(phase3_repo) -> Any:
    """An :class:`AlphaPublisher` bound to the seeded Phase 3 factor repo.

    Wraps the Phase 2 ``seeded_repo``-style :class:`AlphaFactorRepository`
    so tests have a shared staging<->production namespace. The publisher
    extends the SQLite schema on first use (idempotent).
    """
    from greenlang.factors.release.alpha_publisher import AlphaPublisher

    return AlphaPublisher(phase3_repo)


@pytest.fixture()
def phase3_artifact_store(tmp_path: Path) -> Any:
    """A :class:`LocalArtifactStore` rooted under ``tmp_path``.

    Per the Wave 1.0 acceptance constraints (no network in tests), the
    artifact store is filesystem-backed and lives in the per-test tmp
    directory so artifacts written by one test never leak into another.
    """
    from greenlang.factors.ingestion.artifacts import LocalArtifactStore

    return LocalArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def phase3_diff_root(tmp_path: Path) -> Path:
    """Directory the runner writes stage-6 diff JSON / MD artefacts into."""
    diff_root = tmp_path / "diffs"
    diff_root.mkdir(parents=True, exist_ok=True)
    return diff_root


@pytest.fixture()
def phase3_runner_raw(
    phase3_run_repo,
    phase3_repo,
    phase3_publisher,
    phase3_artifact_store,
    phase3_diff_root,
) -> Any:
    """The production :class:`IngestionPipelineRunner` itself.

    Tests that want to drive the runner via its native ``run(source_id,
    source_url, source_urn, source_version, ...)`` API use this fixture
    directly; tests that target the simplified Wave 1.5 contract use
    :func:`phase3_runner` (below), which wraps this with an adapter.
    """
    from greenlang.factors.ingestion.parsers._phase3_adapters import (
        build_phase3_registry,
    )
    from greenlang.factors.ingestion.runner import IngestionPipelineRunner

    # Wire the DEFRA parser against the seeded Phase 2 ontology so gate 3
    # (ontology FK) finds source_urn / unit_urn / geography_urn /
    # methodology_urn / pack_urn. The Phase 3 plan calls this the
    # "single dispatch authority" — the registry pin.
    registry = build_phase3_registry(
        source_urn=SEEDED_SOURCE_URN,
        pack_urn=SEEDED_PACK_URN,
        unit_urn=SEEDED_UNIT_URN,
        geography_urn=SEEDED_GEOGRAPHY_URN,
        methodology_urn=SEEDED_METHODOLOGY_URN,
        licence=SEEDED_LICENCE,
    )
    runner = IngestionPipelineRunner(
        run_repo=phase3_run_repo,
        factor_repo=phase3_repo,
        publisher=phase3_publisher,
        artifact_store=phase3_artifact_store,
        parser_registry=registry,
        diff_root=phase3_diff_root,
        operator="bot:phase3-test",
    )
    return runner


@pytest.fixture()
def phase3_runner(phase3_runner_raw) -> Any:
    """A :class:`Phase3TestRunnerAdapter` exposing the simplified Wave 1.5 contract.

    The adapter delegates to ``phase3_runner_raw`` for every stage but
    rebinds the ``run(...) / run_records(...) / publish(...) / rollback(...)``
    signatures to the test-friendly forms (kw-only fetcher + parser
    + ``on_stage_complete`` + ``approver``).
    """
    from greenlang.factors.ingestion.parsers._phase3_adapters import (
        Phase3TestRunnerAdapter,
    )

    return Phase3TestRunnerAdapter(phase3_runner_raw)


# ---------------------------------------------------------------------------
# DEFRA fixture path + run helpers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def defra_fixture_path() -> Path:
    """Materialise the deterministic DEFRA Excel fixture (idempotent)."""
    from tests.factors.v0_1_alpha.phase3.fixtures._build_defra_fixture import (
        ensure_fixture,
    )

    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    target = fixture_dir / "defra_2025_mini.xlsx"
    return ensure_fixture(target)


@pytest.fixture()
def defra_fixture_url(defra_fixture_path: Path) -> str:
    """``file://`` URL pointing at the synthetic DEFRA fixture."""
    # Use the resolved local path; the FileFetcher accepts plain paths
    # but the runner's fetch logic also accepts file:// URIs.
    return str(defra_fixture_path.resolve())


@pytest.fixture()
def defra_ingestion_run(
    phase3_runner_raw,
    defra_fixture_url: str,
) -> Any:
    """Run the production pipeline end-to-end through stage 6 (staged).

    Returns the freshly-staged :class:`IngestionRun`. Tests can introspect
    ``run.run_id``, ``run.status``, ``run.diff_*_uri``, etc. Mirrors the
    canonical "DEFRA fetch+parse+stage" acceptance scenario.
    """
    return phase3_runner_raw.run(
        source_id=DEFRA_SOURCE_ID,
        source_url=defra_fixture_url,
        source_urn=DEFRA_SOURCE_URN,
        source_version="2025.1",
        operator="bot:test",
        auto_stage=True,
    )


@pytest.fixture()
def defra_published_run(
    phase3_runner_raw,
    defra_ingestion_run,
) -> Any:
    """Same as :func:`defra_ingestion_run` but flipped to production.

    Drives stage 7 (publish) on top of the staged run. ``approver`` is
    the test methodology-lead identity; the publisher's flip writes a
    ``factor_publish_log`` row plus advances the run status.
    """
    phase3_runner_raw.publish(
        defra_ingestion_run.run_id,
        approver="human:test-lead@greenlang.io",
    )
    return phase3_runner_raw._run_repo.get(defra_ingestion_run.run_id)


# ---------------------------------------------------------------------------
# Synthetic artifact fixtures
# ---------------------------------------------------------------------------


_SYNTHETIC_ROWS: List[Dict[str, Any]] = [
    {
        "factor_id": "EF:PHASE3:row-1:v1",
        "name": "Phase 3 synthetic factor 1",
        "value": 0.111,
        "unit": "kgCO2e/kWh",
        "geography": "global:world",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "methodology": "phase2-default",
        "category": "fuel",
        "licence": SEEDED_LICENCE,
        "citation_url": "https://example.test/synthetic/1",
    },
    {
        "factor_id": "EF:PHASE3:row-2:v1",
        "name": "Phase 3 synthetic factor 2",
        "value": 0.222,
        "unit": "kgCO2e/kWh",
        "geography": "global:world",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "methodology": "phase2-default",
        "category": "fuel",
        "licence": SEEDED_LICENCE,
        "citation_url": "https://example.test/synthetic/2",
    },
    {
        "factor_id": "EF:PHASE3:row-3:v1",
        "name": "Phase 3 synthetic factor 3",
        "value": 0.333,
        "unit": "kgCO2e/kWh",
        "geography": "global:world",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "methodology": "phase2-default",
        "category": "fuel",
        "licence": SEEDED_LICENCE,
        "citation_url": "https://example.test/synthetic/3",
    },
]


@pytest.fixture(scope="module")
def synthetic_rows() -> List[Dict[str, Any]]:
    """Deterministic factor rows — payload for the synthetic artifacts.

    Wave 1.5 (2026-04-28): the row count now matches the DEFRA reference
    fixture (10 rows: 5 stationary combustion + 5 fuel conversion) so
    the e2e ``writes_diff_rows_per_input`` test asserts the correct
    one-row-per-input invariant against the canonical DEFRA artefact.
    The original 3-row CSV/Excel synthetic fixtures retain their
    smaller row set for hermetic schema tests; tests that need both
    request the relevant artifact fixture directly.
    """
    rows: List[Dict[str, Any]] = []
    base = _SYNTHETIC_ROWS[0]
    for idx in range(10):
        clone = dict(base)
        clone["factor_id"] = f"EF:PHASE3:row-{idx + 1}:v1"
        clone["name"] = f"Phase 3 synthetic factor {idx + 1}"
        clone["value"] = 0.111 + idx * 0.111
        clone["citation_url"] = f"https://example.test/synthetic/{idx + 1}"
        rows.append(clone)
    return rows


@pytest.fixture(scope="module")
def synthetic_excel_artifact(synthetic_rows) -> bytes:
    """Bytes of a tiny in-memory ``.xlsx`` workbook with three rows.

    Skipped (via :func:`pytest.importorskip`) if openpyxl is not
    installed in the test environment. Three rows is the minimum needed
    to exercise dedupe + supersede + change-detection branches of the
    diff stage.
    """
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Factors"
    headers = list(synthetic_rows[0].keys())
    ws.append(headers)
    for row in synthetic_rows:
        ws.append([row[h] for h in headers])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def synthetic_csv_artifact(synthetic_rows) -> bytes:
    """Bytes of a tiny in-memory CSV with three rows.

    Mirror of ``synthetic_excel_artifact`` for the CSV-family validation
    branches (no openpyxl dependency).
    """
    buf = io.StringIO()
    headers = list(synthetic_rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for row in synthetic_rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Synthetic factor record (post-normalize) — used by validation negative
# tests where a raw artifact is not strictly required.
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_factor_record(seeded_valid_sha256: str) -> Dict[str, Any]:
    """A v0.1 factor record that passes every Phase 2 publish gate.

    Tests mutate this dict to construct negative-path records (e.g.
    drop the ``raw_artifact_uri``, swap to a phantom unit URN) without
    breaking other tests.
    """
    return {
        "urn": "urn:gl:factor:phase2-alpha:default:phase3-synth:v1",
        "factor_id_alias": "EF:PHASE3:synth:v1",
        "source_urn": SEEDED_SOURCE_URN,
        "factor_pack_urn": SEEDED_PACK_URN,
        "name": "Phase 3 synthetic record",
        "description": (
            "Synthetic factor record used by the Phase 3 pipeline "
            "regression suite. Boundary excludes upstream extraction."
        ),
        "category": "fuel",
        "value": 0.555,
        "unit_urn": SEEDED_UNIT_URN,
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": SEEDED_GEOGRAPHY_URN,
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
        "resolution": "annual",
        "methodology_urn": SEEDED_METHODOLOGY_URN,
        "boundary": "Boundary excludes upstream extraction and distribution losses.",
        "licence": SEEDED_LICENCE,
        "citations": [{"type": "url", "value": "https://example.test/phase3"}],
        "published_at": "2026-04-28T12:00:00Z",
        "extraction": {
            "source_url": "https://example.test/phase3",
            "source_record_id": "row=1",
            "source_publication": "Phase 3 test publication",
            "source_version": "2024.1",
            "raw_artifact_uri": "s3://phase3-fixture/2024.1/file.xlsx",
            "raw_artifact_sha256": seeded_valid_sha256,
            "parser_id": "greenlang.factors.ingestion.parsers.phase3_synth",
            "parser_version": "0.1.0",
            "parser_commit": "deadbeefcafe",
            "row_ref": "Sheet=Factors;Row=1",
            "ingested_at": "2026-04-28T11:00:00Z",
            "operator": "bot:phase3-test",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:phase3@greenlang.io",
            "reviewed_at": "2026-04-28T11:30:00Z",
            "approved_by": "human:phase3@greenlang.io",
            "approved_at": "2026-04-28T11:31:00Z",
        },
    }


# ---------------------------------------------------------------------------
# Mock fetcher — returns synthetic bytes regardless of URL.
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_fetcher(defra_fixture_path: Path) -> Callable[..., bytes]:
    """A fetcher callable that ignores its URL and returns DEFRA fixture bytes.

    Wave 1.0 acceptance forbids network access in tests; this fixture is
    the universal substitute for the production HTTP / S3 fetchers. It
    accepts arbitrary positional and keyword arguments so it can stand in
    for any of the production fetcher signatures.

    Wave 1.5: the returned bytes are now the DEFRA reference fixture
    (so the unified pipeline can fetch -> parse -> stage end-to-end
    using a real-shaped artifact). Tests that need the legacy
    ``synthetic_excel_artifact`` form (3-column factor-id sheet) can
    still request that fixture directly.
    """
    raw = defra_fixture_path.read_bytes()

    def _fetch(*_args: Any, **_kwargs: Any) -> bytes:
        return raw

    return _fetch
