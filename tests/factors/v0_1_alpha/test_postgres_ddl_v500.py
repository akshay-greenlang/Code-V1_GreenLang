"""Static + optional integration tests for V500 (factors_v0_1 canonical DDL).

Wave B / TaskCreate #2 / WS1-T2.

This test does NOT require a running Postgres. It loads the migration SQL
file and asserts that every table, constraint, index, trigger, and check
expected by the alpha contract is textually present. The test exists to
prevent silent regressions of the canonical schema mirror of
`factor_record_v0_1.schema.json` (FROZEN 2026-04-25).

An additional `requires_postgres` test creates a temporary database and
applies the DDL via psycopg if Postgres is available; it is skipped
otherwise so default CI does not need a database container.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import List

import pytest

try:
    import sqlparse  # type: ignore[import-untyped]

    _HAS_SQLPARSE = True
except ImportError:  # pragma: no cover - sqlparse is in dev deps but be defensive
    _HAS_SQLPARSE = False


REPO_ROOT = Path(__file__).resolve().parents[3]
DDL_PATH = REPO_ROOT / "deployment" / "database" / "migrations" / "sql" / "V500__factors_v0_1_canonical.sql"
DDL_DOWN_PATH = REPO_ROOT / "deployment" / "database" / "migrations" / "sql" / "V500__factors_v0_1_canonical_DOWN.sql"

EXPECTED_TABLES = [
    "source",
    "methodology",
    "geography",
    "unit",
    "factor_pack",
    "factor",
    "factor_publish_log",
]

# Named CHECK / FK constraints that must appear by name in the DDL text.
EXPECTED_NAMED_CONSTRAINTS = [
    "source_urn_pattern",
    "methodology_urn_pattern",
    "geography_urn_pattern",
    "unit_urn_pattern",
    "factor_pack_urn_pattern",
    "factor_urn_pattern",
    "factor_extraction_required_fields",
    "factor_review_required_fields",
    "factor_review_approved_requires_approver",
    "factor_review_rejected_requires_reason",
]

EXPECTED_INDEXES = [
    "source_alpha_v0_1_idx",
    "factor_pack_source_idx",
    "factor_source_idx",
    "factor_pack_idx",
    "factor_geo_vintage_idx",
    "factor_category_idx",
    "factor_published_at_idx",
    "factor_active_idx",
    "factor_fts_idx",
    "factor_tags_idx",
    "factor_alias_idx",
    "factor_publish_log_factor_idx",
    "factor_publish_log_edition_idx",
]

EXPECTED_CATEGORY_VALUES = [
    "scope1",
    "scope2_location_based",
    "scope2_market_based",
    "grid_intensity",
    "fuel",
    "refrigerant",
    "fugitive",
    "process",
    "cbam_default",
]

EXPECTED_TRUST_TIERS = ["tier_1", "tier_2", "tier_3"]
EXPECTED_REVIEW_STATUSES = ["pending", "approved", "rejected"]
EXPECTED_RESOLUTIONS = ["annual", "monthly", "hourly", "point-in-time"]
EXPECTED_GEO_TYPES = [
    "global",
    "country",
    "subregion",
    "state_or_province",
    "grid_zone",
    "bidding_zone",
    "balancing_authority",
]
EXPECTED_LICENSE_CLASSES = [
    "public_us_government",
    "uk_open_government",
    "public_eu",
    "cc_by",
    "cc_by_sa",
    "registry_terms",
    "commercial_connector",
    "tenant_private",
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ddl_text() -> str:
    assert DDL_PATH.exists(), f"V500 DDL not found at {DDL_PATH}"
    return DDL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ddl_down_text() -> str:
    assert DDL_DOWN_PATH.exists(), f"V500 DOWN DDL not found at {DDL_DOWN_PATH}"
    return DDL_DOWN_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parsed_statements(ddl_text: str) -> List[str]:
    """Split the DDL into top-level statements via sqlparse for sanity-check."""
    if not _HAS_SQLPARSE:
        pytest.skip("sqlparse not installed")
    statements = [s.strip() for s in sqlparse.split(ddl_text) if s.strip()]
    return statements


# --------------------------------------------------------------------------- #
# Static tests
# --------------------------------------------------------------------------- #


def test_ddl_file_exists() -> None:
    assert DDL_PATH.exists(), f"Missing migration: {DDL_PATH}"
    assert DDL_DOWN_PATH.exists(), f"Missing DOWN migration: {DDL_DOWN_PATH}"


def test_ddl_header_references_authority(ddl_text: str) -> None:
    """Header must cite CTO doc §6.1, schema $id, and freeze date per task spec."""
    assert "§6.1" in ddl_text, "DDL header missing CTO doc §6.1 reference"
    assert "factor_record_v0_1.schema.json" in ddl_text, (
        "DDL header missing schema $id reference"
    )
    assert "2026-04-25" in ddl_text, "DDL header missing freeze date"


def test_ddl_creates_target_schema(ddl_text: str) -> None:
    assert re.search(
        r"CREATE\s+SCHEMA\s+IF\s+NOT\s+EXISTS\s+factors_v0_1",
        ddl_text,
        re.IGNORECASE,
    ), "factors_v0_1 schema creation missing"
    assert re.search(
        r"SET\s+search_path\s+TO\s+factors_v0_1\s*,\s*public",
        ddl_text,
        re.IGNORECASE,
    ), "search_path not set to factors_v0_1, public"


@pytest.mark.parametrize("table_name", EXPECTED_TABLES)
def test_each_table_is_created(ddl_text: str, table_name: str) -> None:
    pattern = rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table_name)}\b"
    assert re.search(pattern, ddl_text, re.IGNORECASE), (
        f"Missing CREATE TABLE for `{table_name}`"
    )


@pytest.mark.parametrize("constraint_name", EXPECTED_NAMED_CONSTRAINTS)
def test_each_named_constraint_present(ddl_text: str, constraint_name: str) -> None:
    pattern = rf"CONSTRAINT\s+{re.escape(constraint_name)}\b"
    assert re.search(pattern, ddl_text), (
        f"Missing named constraint `{constraint_name}`"
    )


@pytest.mark.parametrize("index_name", EXPECTED_INDEXES)
def test_each_index_is_created(ddl_text: str, index_name: str) -> None:
    pattern = rf"CREATE\s+INDEX\s+{re.escape(index_name)}\b"
    assert re.search(pattern, ddl_text, re.IGNORECASE), (
        f"Missing CREATE INDEX `{index_name}`"
    )


def test_factor_value_is_strictly_positive(ddl_text: str) -> None:
    assert re.search(r"value\s+NUMERIC\(30,12\)\s+NOT\s+NULL\s+CHECK\s*\(\s*value\s*>\s*0\s*\)",
                     ddl_text, re.IGNORECASE), "factor.value must enforce > 0"


def test_factor_gwp_basis_locked_to_ar6(ddl_text: str) -> None:
    assert re.search(r"gwp_basis\s+TEXT\s+NOT\s+NULL\s+CHECK\s*\(\s*gwp_basis\s*=\s*'ar6'\s*\)",
                     ddl_text, re.IGNORECASE), (
        "gwp_basis must be locked to literal 'ar6' in alpha"
    )


def test_factor_gwp_horizon_enum(ddl_text: str) -> None:
    assert re.search(r"gwp_horizon\s+INTEGER\s+NOT\s+NULL\s+CHECK\s*\(\s*gwp_horizon\s+IN\s*\(\s*20\s*,\s*100\s*,\s*500\s*\)\s*\)",
                     ddl_text, re.IGNORECASE), "gwp_horizon must allow only (20,100,500)"


def test_factor_citations_min_one(ddl_text: str) -> None:
    assert re.search(r"jsonb_array_length\(citations\)\s*>=\s*1", ddl_text, re.IGNORECASE), (
        "citations must enforce minItems=1"
    )


def test_factor_extraction_required_fields_listed(ddl_text: str) -> None:
    """Every required extraction sub-field must be checked via JSONB ? operator."""
    required = [
        "source_url",
        "source_record_id",
        "source_publication",
        "source_version",
        "raw_artifact_uri",
        "raw_artifact_sha256",
        "parser_id",
        "parser_version",
        "parser_commit",
        "row_ref",
        "ingested_at",
        "operator",
    ]
    for fld in required:
        assert re.search(rf"extraction\s*\?\s*'{fld}'", ddl_text), (
            f"factor_extraction_required_fields missing `{fld}` ? check"
        )
    # And the SHA-256 regex
    assert "^[a-f0-9]{64}$" in ddl_text, "raw_artifact_sha256 SHA-256 regex missing"


def test_factor_review_required_fields_listed(ddl_text: str) -> None:
    required = ["review_status", "reviewer", "reviewed_at"]
    for fld in required:
        assert re.search(rf"review\s*\?\s*'{fld}'", ddl_text), (
            f"factor_review_required_fields missing `{fld}` ? check"
        )


def test_factor_review_status_values(ddl_text: str) -> None:
    for value in EXPECTED_REVIEW_STATUSES:
        assert f"'{value}'" in ddl_text, f"review_status enum missing `{value}`"


@pytest.mark.parametrize("category", EXPECTED_CATEGORY_VALUES)
def test_category_enum_values(ddl_text: str, category: str) -> None:
    assert f"'{category}'" in ddl_text, f"category enum missing `{category}`"


@pytest.mark.parametrize("tier", EXPECTED_TRUST_TIERS)
def test_trust_tier_values(ddl_text: str, tier: str) -> None:
    assert f"'{tier}'" in ddl_text, f"trust_tier enum missing `{tier}`"


@pytest.mark.parametrize("resolution", EXPECTED_RESOLUTIONS)
def test_resolution_enum_values(ddl_text: str, resolution: str) -> None:
    assert f"'{resolution}'" in ddl_text, f"resolution enum missing `{resolution}`"


@pytest.mark.parametrize("geo_type", EXPECTED_GEO_TYPES)
def test_geography_type_values(ddl_text: str, geo_type: str) -> None:
    assert f"'{geo_type}'" in ddl_text, f"geography.type enum missing `{geo_type}`"


@pytest.mark.parametrize("license_class", EXPECTED_LICENSE_CLASSES)
def test_license_class_values(ddl_text: str, license_class: str) -> None:
    assert f"'{license_class}'" in ddl_text, (
        f"license_class enum missing `{license_class}`"
    )


def test_immutability_function_defined(ddl_text: str) -> None:
    assert re.search(
        r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+factor_immutable_trigger\s*\(\s*\)\s*RETURNS\s+TRIGGER",
        ddl_text,
        re.IGNORECASE,
    ), "factor_immutable_trigger function missing"
    # It must guard the five immutable fields
    for guarded in ["urn", "value", "published_at", "gwp_basis", "unit_urn"]:
        assert re.search(
            rf"NEW\.{guarded}\s*!=\s*OLD\.{guarded}",
            ddl_text,
        ), f"immutability trigger does not guard `{guarded}`"


def test_immutability_trigger_attached(ddl_text: str) -> None:
    assert re.search(
        r"CREATE\s+TRIGGER\s+factor_no_mutate_after_publish\s+BEFORE\s+UPDATE\s+ON\s+factor",
        ddl_text,
        re.IGNORECASE,
    ), "factor_no_mutate_after_publish trigger not attached BEFORE UPDATE on factor"
    assert re.search(
        r"FOR\s+EACH\s+ROW\s+EXECUTE\s+FUNCTION\s+factor_immutable_trigger",
        ddl_text,
        re.IGNORECASE,
    ), "trigger does not call factor_immutable_trigger per row"


def test_factor_publish_log_is_appendable_audit(ddl_text: str) -> None:
    """factor_publish_log must have factor_urn, edition_id, published_by, JSONB metadata."""
    cols = ["factor_urn", "edition_id", "published_at", "published_by", "metadata"]
    # Find the CREATE TABLE block for factor_publish_log specifically
    m = re.search(
        r"CREATE\s+TABLE\s+factor_publish_log\s*\((?P<body>[\s\S]+?)\)\s*;",
        ddl_text,
        re.IGNORECASE,
    )
    assert m, "factor_publish_log table block not found"
    body = m.group("body")
    for col in cols:
        assert re.search(rf"\b{col}\b", body), f"factor_publish_log missing `{col}` column"


def test_foreign_keys_into_registries(ddl_text: str) -> None:
    """factor must FK into source, factor_pack, unit, geography, methodology."""
    expected_fks = {
        "source_urn": "source",
        "factor_pack_urn": "factor_pack",
        "unit_urn": "unit",
        "geography_urn": "geography",
        "methodology_urn": "methodology",
    }
    m = re.search(r"CREATE\s+TABLE\s+factor\s*\(", ddl_text, re.IGNORECASE)
    assert m, "factor table not found"
    factor_block = ddl_text[m.start():]
    # Cut at the first standalone closing paren followed by semicolon
    end = factor_block.find(");")
    assert end > 0
    factor_block = factor_block[: end + 2]
    for col, ref_table in expected_fks.items():
        assert re.search(
            rf"{col}\s+TEXT\s+NOT\s+NULL\s+REFERENCES\s+{ref_table}\(urn\)",
            factor_block,
            re.IGNORECASE,
        ), f"factor.{col} must REFERENCES {ref_table}(urn)"


def test_self_reference_supersedes(ddl_text: str) -> None:
    assert re.search(
        r"supersedes_urn\s+TEXT\s+REFERENCES\s+factor\(urn\)",
        ddl_text,
        re.IGNORECASE,
    ), "factor.supersedes_urn must self-reference factor(urn)"


def test_geography_self_reference_parent(ddl_text: str) -> None:
    assert re.search(
        r"parent_urn\s+TEXT\s+REFERENCES\s+geography\(urn\)",
        ddl_text,
        re.IGNORECASE,
    ), "geography.parent_urn must self-reference geography(urn)"


def test_active_index_partial_predicate(ddl_text: str) -> None:
    assert re.search(
        r"CREATE\s+INDEX\s+factor_active_idx[\s\S]+?WHERE\s+review->>'review_status'\s*=\s*'approved'",
        ddl_text,
        re.IGNORECASE,
    ), "factor_active_idx must be partial WHERE review_status='approved'"


def test_fts_gin_index_uses_english_tsvector(ddl_text: str) -> None:
    assert re.search(
        r"factor_fts_idx[\s\S]+?GIN\s*\(\s*to_tsvector\(\s*'english'",
        ddl_text,
        re.IGNORECASE,
    ), "factor_fts_idx must be GIN over to_tsvector('english', ...)"


def test_tags_gin_index_present(ddl_text: str) -> None:
    assert re.search(
        r"factor_tags_idx[\s\S]+?USING\s+GIN\s*\(\s*tags\s*\)",
        ddl_text,
        re.IGNORECASE,
    ), "factor_tags_idx must be GIN over tags"


def test_sqlparse_can_parse_ddl(parsed_statements: List[str]) -> None:
    """Sanity: sqlparse must successfully split the DDL into statements."""
    assert parsed_statements, "sqlparse returned zero statements"
    # Loose lower bound: ~7 tables + 13 indexes + 1 function + 1 trigger + 1 schema
    # + SET + dollar-quoted body etc. We expect at least 20 top-level statements.
    assert len(parsed_statements) >= 20, (
        f"Expected >=20 statements, got {len(parsed_statements)}: "
        "DDL may be malformed or truncated"
    )


# --------------------------------------------------------------------------- #
# DOWN migration tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "drop_target",
    [
        "DROP TABLE IF EXISTS factors_v0_1.factor_publish_log",
        "DROP TRIGGER IF EXISTS factor_no_mutate_after_publish",
        "DROP FUNCTION IF EXISTS factors_v0_1.factor_immutable_trigger",
        "DROP TABLE IF EXISTS factors_v0_1.factor",
        "DROP TABLE IF EXISTS factors_v0_1.factor_pack",
        "DROP TABLE IF EXISTS factors_v0_1.unit",
        "DROP TABLE IF EXISTS factors_v0_1.geography",
        "DROP TABLE IF EXISTS factors_v0_1.methodology",
        "DROP TABLE IF EXISTS factors_v0_1.source",
        "DROP SCHEMA IF EXISTS factors_v0_1 CASCADE",
    ],
)
def test_down_migration_drops_each_object(ddl_down_text: str, drop_target: str) -> None:
    assert drop_target in ddl_down_text, (
        f"DOWN migration missing statement: {drop_target}"
    )


def test_down_drops_factor_before_registries(ddl_down_text: str) -> None:
    """Order matters: factor_publish_log -> trigger/function -> factor -> factor_pack -> registries -> schema.

    Use full statement-terminator match (`;`) so `factor` does not also match
    `factor_publish_log` or `factor_pack` (prefix-collision).
    """
    indices = {
        "log": ddl_down_text.index("DROP TABLE IF EXISTS factors_v0_1.factor_publish_log;"),
        "trigger": ddl_down_text.index("DROP TRIGGER IF EXISTS factor_no_mutate_after_publish"),
        "func": ddl_down_text.index("DROP FUNCTION IF EXISTS factors_v0_1.factor_immutable_trigger"),
        "factor": ddl_down_text.index("DROP TABLE IF EXISTS factors_v0_1.factor;"),
        "pack": ddl_down_text.index("DROP TABLE IF EXISTS factors_v0_1.factor_pack;"),
        "source": ddl_down_text.index("DROP TABLE IF EXISTS factors_v0_1.source;"),
        "schema": ddl_down_text.index("DROP SCHEMA IF EXISTS factors_v0_1 CASCADE"),
    }
    assert indices["log"] < indices["trigger"] < indices["func"] < indices["factor"]
    assert indices["factor"] < indices["pack"] < indices["source"] < indices["schema"]


# --------------------------------------------------------------------------- #
# Optional integration test (skipped unless Postgres is reachable)
# --------------------------------------------------------------------------- #


def _postgres_dsn_or_skip() -> str:
    """Return a DSN if Postgres is reachable; otherwise skip the test."""
    dsn = os.environ.get("FACTORS_TEST_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("No Postgres DSN set (FACTORS_TEST_POSTGRES_DSN / DATABASE_URL)")
    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("psycopg not installed")
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:  # noqa: F841
            pass
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"Postgres not reachable: {exc}")
    return dsn


@pytest.mark.requires_postgres
def test_ddl_applies_against_real_postgres(ddl_text: str, ddl_down_text: str) -> None:
    """Apply the V500 DDL against an isolated temp DB to prove it loads cleanly."""
    dsn = _postgres_dsn_or_skip()
    import psycopg  # type: ignore[import-untyped]

    temp_db = f"factors_v0_1_test_{uuid.uuid4().hex[:8]}"

    # Bootstrap connection: create temp DB.
    with psycopg.connect(dsn, autocommit=True) as boot:
        with boot.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{temp_db}"')

    try:
        # Connect to the temp DB and apply the DDL.
        temp_dsn = re.sub(r"/[^/?]+(\?|$)", f"/{temp_db}\\1", dsn, count=1)
        with psycopg.connect(temp_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl_text)

                # Sanity: factor table exists in factors_v0_1.
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema='factors_v0_1' AND table_name='factor'"
                )
                row = cur.fetchone()
                assert row is not None and row[0] == 1

                # Trigger present
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.triggers "
                    "WHERE event_object_schema='factors_v0_1' "
                    "AND trigger_name='factor_no_mutate_after_publish'"
                )
                row = cur.fetchone()
                assert row is not None and row[0] >= 1

                # Apply DOWN: should drop everything cleanly.
                cur.execute(ddl_down_text)

                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.schemata "
                    "WHERE schema_name='factors_v0_1'"
                )
                row = cur.fetchone()
                assert row is not None and row[0] == 0
    finally:
        # Tear down temp DB regardless of test outcome.
        with psycopg.connect(dsn, autocommit=True) as boot:
            with boot.cursor() as cur:
                cur.execute(
                    f'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
                    f"WHERE datname='{temp_db}'"
                )
                cur.execute(f'DROP DATABASE IF EXISTS "{temp_db}"')
