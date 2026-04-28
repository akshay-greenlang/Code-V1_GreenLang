# -*- coding: utf-8 -*-
"""Phase 2 / WS7-T1 — AlphaFactorRepository Phase 2 surface tests.

Covers the new methods added by WS7:

  * find_by_methodology(methodology_urn) -> List[record]
  * find_by_alias(legacy_id) -> Optional[record]
  * register_alias(urn, legacy_id, kind='EF') -> int
  * register_artifact(sha256, source_urn, version, uri, parser_meta) -> int
  * link_provenance(factor_urn, artifact_pk, row_ref, edge_type) -> int
  * record_changelog_event(...) -> int

All tests run against the in-memory SQLite backend (the repo carries
SQLite parity DDL for every Phase 2 table — alpha_factor_aliases_v0_1,
alpha_source_artifacts_v0_1, alpha_provenance_edges_v0_1,
alpha_changelog_events_v0_1).
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from greenlang.factors.repositories import (
    AlphaFactorRepository,
    FactorURNAlreadyExistsError,
)

from tests.factors.v0_1_alpha._e2e_helpers import good_ipcc_ar6_factor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo() -> AlphaFactorRepository:
    # legacy mode — Phase 1 provenance gate only; Phase 2 orchestrator covered by tests/factors/v0_1_alpha/phase2/test_publish_pipeline_e2e.py
    r = AlphaFactorRepository(dsn="sqlite:///:memory:", publish_env="legacy")
    yield r
    r.close()


def _record(*, leaf: str, methodology: str | None = None) -> Dict[str, Any]:
    """Mint a v0.1 factor whose URN is unique by ``leaf``."""
    rec = copy.deepcopy(good_ipcc_ar6_factor())
    rec["urn"] = (
        "urn:gl:factor:ipcc-ar6:stationary-combustion:" + leaf + ":v1"
    )
    rec["factor_id_alias"] = "EF:IPCC:stationary-combustion:" + leaf + ":v1"
    if methodology is not None:
        rec["methodology_urn"] = methodology
    return rec


_VALID_SHA256_A = "a" * 64
_VALID_SHA256_B = "b" * 64


# ---------------------------------------------------------------------------
# find_by_methodology
# ---------------------------------------------------------------------------


def test_find_by_methodology_returns_matching_records(
    repo: AlphaFactorRepository,
) -> None:
    repo.publish(
        _record(leaf="natural-gas-residential",
                methodology="urn:gl:methodology:ipcc-tier-1-stationary-combustion")
    )
    repo.publish(
        _record(leaf="natural-gas-industrial",
                methodology="urn:gl:methodology:ipcc-tier-1-stationary-combustion")
    )
    repo.publish(
        _record(leaf="diesel-mobile",
                methodology="urn:gl:methodology:ghgp-scope-1-mobile")
    )

    results = repo.find_by_methodology(
        "urn:gl:methodology:ipcc-tier-1-stationary-combustion"
    )
    assert len(results) == 2
    urns = {r["urn"] for r in results}
    assert urns == {
        "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1",
        "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-industrial:v1",
    }


def test_find_by_methodology_no_match_returns_empty(
    repo: AlphaFactorRepository,
) -> None:
    repo.publish(_record(leaf="natural-gas-residential"))
    assert repo.find_by_methodology("urn:gl:methodology:non-existent") == []


def test_find_by_methodology_rejects_empty_input(
    repo: AlphaFactorRepository,
) -> None:
    assert repo.find_by_methodology("") == []
    assert repo.find_by_methodology(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# register_alias + find_by_alias
# ---------------------------------------------------------------------------


def test_register_alias_then_find_by_alias_round_trip(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)

    pk = repo.register_alias(rec["urn"], "EF:legacy:natural-gas-residential:v1")
    assert isinstance(pk, int) and pk > 0

    found = repo.find_by_alias("EF:legacy:natural-gas-residential:v1")
    assert found is not None
    assert found["urn"] == rec["urn"]
    # Record must be byte-equal to the original (no coercion in the join path).
    assert found == rec


def test_register_alias_default_kind_is_ef(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    pk = repo.register_alias(rec["urn"], "EF:any-legacy")
    assert pk > 0


def test_register_alias_rejects_bad_kind(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    with pytest.raises(ValueError, match="kind must be"):
        repo.register_alias(rec["urn"], "EF:legacy", kind="invalid")


def test_register_alias_rejects_empty_urn_or_legacy(
    repo: AlphaFactorRepository,
) -> None:
    with pytest.raises(ValueError, match="urn must be"):
        repo.register_alias("", "EF:legacy")
    with pytest.raises(ValueError, match="legacy_id must be"):
        repo.register_alias("urn:gl:factor:x:y:z:v1", "")


def test_register_alias_duplicate_legacy_id_raises(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    repo.register_alias(rec["urn"], "EF:duplicate")
    with pytest.raises(FactorURNAlreadyExistsError):
        repo.register_alias(rec["urn"], "EF:duplicate")


def test_find_by_alias_misses_return_none(
    repo: AlphaFactorRepository,
) -> None:
    assert repo.find_by_alias("EF:never-registered") is None
    assert repo.find_by_alias("") is None


# ---------------------------------------------------------------------------
# register_artifact
# ---------------------------------------------------------------------------


def test_register_artifact_round_trip(
    repo: AlphaFactorRepository,
) -> None:
    pk = repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="AR6-WG3-Annex-III",
        uri="s3://greenlang-factors-raw/ipcc/ar6/wg3-annex-iii.pdf",
        parser_meta={
            "parser_id": "greenlang.factors.ingestion.parsers.ipcc_defaults",
            "parser_version": "0.1.0",
            "parser_commit": "deadbeefcafe",
            "content_type": "application/pdf",
            "size_bytes": 1024,
            "metadata": {"pages": 42},
        },
    )
    assert isinstance(pk, int) and pk > 0


def test_register_artifact_rejects_bad_sha256(
    repo: AlphaFactorRepository,
) -> None:
    bad = ("", "not-hex", "g" * 64, "A" * 64, "a" * 63, "a" * 65)
    for sha in bad:
        with pytest.raises(ValueError, match="sha256 must be"):
            repo.register_artifact(
                sha256=sha,
                source_urn="urn:gl:source:ipcc-ar6",
                version="v1",
                uri="s3://x",
            )


def test_register_artifact_rejects_empty_args(
    repo: AlphaFactorRepository,
) -> None:
    with pytest.raises(ValueError, match="source_urn"):
        repo.register_artifact(
            sha256=_VALID_SHA256_A, source_urn="", version="v1", uri="s3://x"
        )
    with pytest.raises(ValueError, match="version"):
        repo.register_artifact(
            sha256=_VALID_SHA256_A,
            source_urn="urn:gl:source:ipcc-ar6",
            version="",
            uri="s3://x",
        )
    with pytest.raises(ValueError, match="uri"):
        repo.register_artifact(
            sha256=_VALID_SHA256_A,
            source_urn="urn:gl:source:ipcc-ar6",
            version="v1",
            uri="",
        )


def test_register_artifact_duplicate_sha256_raises(
    repo: AlphaFactorRepository,
) -> None:
    repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="v1",
        uri="s3://x",
    )
    with pytest.raises(FactorURNAlreadyExistsError):
        repo.register_artifact(
            sha256=_VALID_SHA256_A,
            source_urn="urn:gl:source:ipcc-ar6",
            version="v1",
            uri="s3://x",
        )


# ---------------------------------------------------------------------------
# link_provenance
# ---------------------------------------------------------------------------


def test_link_provenance_round_trip(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    artifact_pk = repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="AR6-WG3",
        uri="s3://x",
    )
    edge_pk = repo.link_provenance(
        factor_urn=rec["urn"],
        artifact_pk=artifact_pk,
        row_ref="Sheet=Annex2;Row=42",
    )
    assert isinstance(edge_pk, int) and edge_pk > 0


def test_link_provenance_default_edge_type_is_extraction(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    artifact_pk = repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="v1",
        uri="s3://x",
    )
    pk = repo.link_provenance(rec["urn"], artifact_pk, "row=1")
    assert pk > 0


def test_link_provenance_rejects_invalid_edge_type(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    artifact_pk = repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="v1",
        uri="s3://x",
    )
    with pytest.raises(ValueError, match="edge_type"):
        repo.link_provenance(rec["urn"], artifact_pk, "row=1", edge_type="bogus")


def test_link_provenance_accepts_all_valid_edge_types(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    artifact_pk = repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="v1",
        uri="s3://x",
    )
    for et in ("extraction", "derivation", "correction", "supersedes"):
        # row_ref differs so the UNIQUE composite isn't violated.
        pk = repo.link_provenance(
            rec["urn"], artifact_pk, "row=" + et, edge_type=et
        )
        assert pk > 0


def test_link_provenance_rejects_bad_artifact_pk(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    with pytest.raises(ValueError, match="artifact_pk"):
        repo.link_provenance(rec["urn"], 0, "row=1")
    with pytest.raises(ValueError, match="artifact_pk"):
        repo.link_provenance(rec["urn"], -1, "row=1")


def test_link_provenance_duplicate_edge_raises(
    repo: AlphaFactorRepository,
) -> None:
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)
    artifact_pk = repo.register_artifact(
        sha256=_VALID_SHA256_A,
        source_urn="urn:gl:source:ipcc-ar6",
        version="v1",
        uri="s3://x",
    )
    repo.link_provenance(rec["urn"], artifact_pk, "row=1")
    with pytest.raises(FactorURNAlreadyExistsError):
        repo.link_provenance(rec["urn"], artifact_pk, "row=1")


# ---------------------------------------------------------------------------
# record_changelog_event
# ---------------------------------------------------------------------------


def test_record_changelog_event_round_trip(
    repo: AlphaFactorRepository,
) -> None:
    pk = repo.record_changelog_event(
        event_type="factor_publish",
        subject_urn="urn:gl:factor:ipcc-ar6:x:y:v1",
        change_class="additive",
        actor="bot:publisher",
        metadata={"source": "ipcc-ar6"},
    )
    assert isinstance(pk, int) and pk > 0


def test_record_changelog_event_rejects_invalid_event_type(
    repo: AlphaFactorRepository,
) -> None:
    with pytest.raises(ValueError, match="event_type"):
        repo.record_changelog_event(
            event_type="bogus",
            subject_urn=None,
            change_class=None,
            actor="bot:test",
        )


def test_record_changelog_event_rejects_invalid_change_class(
    repo: AlphaFactorRepository,
) -> None:
    with pytest.raises(ValueError, match="change_class"):
        repo.record_changelog_event(
            event_type="schema_change",
            subject_urn=None,
            change_class="invalid",
            actor="bot:test",
        )


def test_record_changelog_event_rejects_empty_actor(
    repo: AlphaFactorRepository,
) -> None:
    with pytest.raises(ValueError, match="actor"):
        repo.record_changelog_event(
            event_type="schema_change",
            subject_urn=None,
            change_class=None,
            actor="",
        )


def test_record_changelog_event_accepts_all_event_types(
    repo: AlphaFactorRepository,
) -> None:
    types = (
        "schema_change", "factor_publish", "factor_supersede",
        "factor_deprecate", "source_add", "source_deprecate",
        "pack_release", "migration_apply",
    )
    for et in types:
        pk = repo.record_changelog_event(
            event_type=et,
            subject_urn="urn:gl:factor:x:y:z:v1",
            change_class=None,
            actor="bot:test",
        )
        assert pk > 0


def test_record_changelog_event_optional_kwargs_round_trip(
    repo: AlphaFactorRepository,
) -> None:
    pk = repo.record_changelog_event(
        event_type="schema_change",
        subject_urn=None,
        change_class="breaking",
        actor="bot:schema-bot",
        schema_version="0.2.0",
        migration_note_uri="https://docs.greenlang.io/factors/schema/v0_2",
    )
    assert pk > 0


# ---------------------------------------------------------------------------
# End-to-end Phase 2 round-trip — publish + register_artifact +
# link_provenance + find_by_alias all in the same in-memory repo.
# ---------------------------------------------------------------------------


def test_full_phase2_round_trip(repo: AlphaFactorRepository) -> None:
    """The CTO Phase 2 brief's "must work end-to-end" smoke flow."""
    rec = _record(leaf="natural-gas-residential")
    repo.publish(rec)

    # 1. Register the raw artifact the factor was extracted from.
    artifact_pk = repo.register_artifact(
        sha256=rec["extraction"]["raw_artifact_sha256"],
        source_urn="urn:gl:source:ipcc-ar6",
        version="AR6-WG3-Annex-III",
        uri=rec["extraction"]["raw_artifact_uri"],
        parser_meta={
            "parser_id": rec["extraction"]["parser_id"],
            "parser_version": rec["extraction"]["parser_version"],
            "parser_commit": rec["extraction"]["parser_commit"],
        },
    )

    # 2. Link factor -> artifact at the row level.
    edge_pk = repo.link_provenance(
        factor_urn=rec["urn"],
        artifact_pk=artifact_pk,
        row_ref=rec["extraction"]["row_ref"],
    )

    # 3. Backfill the legacy EF: alias.
    alias_pk = repo.register_alias(rec["urn"], rec["factor_id_alias"])

    # 4. Record the publish event in the audit log.
    event_pk = repo.record_changelog_event(
        event_type="factor_publish",
        subject_urn=rec["urn"],
        change_class="additive",
        actor=rec["extraction"]["operator"],
        metadata={"artifact_pk": artifact_pk, "edge_pk": edge_pk, "alias_pk": alias_pk},
    )

    # All four pks are positive ints.
    for pk in (artifact_pk, edge_pk, alias_pk, event_pk):
        assert isinstance(pk, int) and pk > 0

    # Round-trip the alias path back to the same record.
    via_alias = repo.find_by_alias(rec["factor_id_alias"])
    assert via_alias == rec

    # Round-trip the methodology query.
    via_methodology = repo.find_by_methodology(rec["methodology_urn"])
    assert len(via_methodology) == 1
    assert via_methodology[0] == rec
