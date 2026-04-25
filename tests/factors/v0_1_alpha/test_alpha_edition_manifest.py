# -*- coding: utf-8 -*-
"""
Wave E / TaskCreate #24 / WS9-T2 — tests for ``alpha_edition_manifest``.

Exercises:
  * build_manifest produces a manifest with 691 factor entries (post-backfill).
  * every factor's record_sha256 is reproducible (run twice -> same hash).
  * manifest_sha256 changes when ANY factor's record_sha256 changes.
  * write/read round-trip preserves all fields.
  * verify_manifest returns True on a known-good signature; False on tampered
    manifest bytes.
  * canonical-JSON is whitespace-stable across platforms (Linux vs Windows
    line endings).
  * per-source counts in the manifest match the catalog_seed_v0_1 input
    counts.
  * bad approver email format raises.
  * placeholder signature file is recognized as "unsigned" by verify_manifest.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict

import pytest

from greenlang.factors.release.alpha_edition_manifest import (
    API_RELEASE_PROFILE,
    BUILDER_ID,
    DEFAULT_METHODOLOGY_LEAD,
    PLACEHOLDER_SUFFIX,
    SCHEMA_ID,
    SDK_VERSION,
    AlphaEditionManifest,
    FactorManifestEntry,
    SourceManifestEntry,
    build_manifest,
    canonical_json_bytes,
    render_release_notes,
    verify_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

EXPECTED_TOTAL_FACTORS = 691

# Per-source ground truth (source_id -> record count). Mirrors what's in
# greenlang/factors/data/catalog_seed_v0_1/<source>/v1.json after the Wave D
# #6 backfill.
EXPECTED_PER_SOURCE: Dict[str, int] = {
    "cbam_default_values": 60,
    "desnz_ghg_conversion": 195,
    "egrid": 79,
    "epa_hub": 84,
    "india_cea_co2_baseline": 38,
    "ipcc_2006_nggi": 235,
}


@pytest.fixture()
def fixed_timestamp() -> str:
    """Deterministic timestamp so two builds produce the same manifest_sha256."""
    return "2026-04-25T12:00:00Z"


@pytest.fixture()
def manifest(fixed_timestamp) -> AlphaEditionManifest:
    """Default manifest used across most tests."""
    return build_manifest(
        edition_id="factors-v0.1.0-alpha-2026-04-25",
        methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
        methodology_lead_approved_at=fixed_timestamp,
        build_timestamp=fixed_timestamp,
        git_commit="0000000000000000000000000000000000000000",
    )


def _make_pem_keypair():
    """Generate an ephemeral Ed25519 PEM keypair for signing tests."""
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


# ---------------------------------------------------------------------------
# build_manifest — content tests
# ---------------------------------------------------------------------------


def test_build_manifest_total_factor_count(manifest):
    """Wave D #6 produced 691 v0.1-shape records. The manifest must mirror that."""
    assert len(manifest.factors) == EXPECTED_TOTAL_FACTORS


def test_build_manifest_six_sources(manifest):
    assert len(manifest.sources) == 6
    source_ids = {s.source_id for s in manifest.sources}
    assert source_ids == set(EXPECTED_PER_SOURCE)


def test_build_manifest_per_source_counts_match_seeds(manifest):
    by_id = {s.source_id: s.factor_count for s in manifest.sources}
    assert by_id == EXPECTED_PER_SOURCE


def test_build_manifest_sum_of_source_counts_equals_factor_count(manifest):
    assert sum(s.factor_count for s in manifest.sources) == len(manifest.factors)


def test_build_manifest_static_metadata_fields(manifest):
    assert manifest.schema_id == SCHEMA_ID
    assert manifest.sdk_version == SDK_VERSION
    assert manifest.api_release_profile == API_RELEASE_PROFILE
    assert manifest.builder == BUILDER_ID
    # The schema sha256 is a 64-char lowercase hex digest.
    assert len(manifest.schema_sha256) == 64
    assert all(c in "0123456789abcdef" for c in manifest.schema_sha256)
    assert len(manifest.factor_record_v0_1_freeze_sha256) == 64


def test_build_manifest_edition_id_default_format(monkeypatch):
    """Without --edition-id, build_manifest derives one from today's date."""
    m = build_manifest(
        methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
        methodology_lead_approved_at="2026-04-25T00:00:00Z",
        build_timestamp="2026-04-25T00:00:00Z",
        git_commit="abc",
    )
    assert m.edition_id.startswith("factors-v0.1.0-alpha-")
    # The trailing date portion must be 10 chars: YYYY-MM-DD.
    assert len(m.edition_id.split("alpha-")[-1]) == 10


def test_build_manifest_factor_pack_urns_present(manifest):
    """CTO §6.3: every factor belongs to exactly one pack."""
    for f in manifest.factors:
        assert f.factor_pack_urn.startswith("urn:gl:pack:")
    for s in manifest.sources:
        assert len(s.factor_pack_urns) >= 1
        for urn in s.factor_pack_urns:
            assert urn.startswith("urn:gl:pack:")


def test_build_manifest_parser_commits_present(manifest):
    """Every source contributes a parser_commit; collected on the manifest."""
    assert len(manifest.parser_commits) == 6
    for source_id, commit in manifest.parser_commits.items():
        assert source_id in EXPECTED_PER_SOURCE
        assert isinstance(commit, str) and commit  # non-empty


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_record_sha256_is_reproducible(fixed_timestamp):
    """Two builds with identical inputs produce identical record hashes."""
    m1 = build_manifest(
        edition_id="factors-v0.1.0-alpha-2026-04-25",
        methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
        methodology_lead_approved_at=fixed_timestamp,
        build_timestamp=fixed_timestamp,
        git_commit="x",
    )
    m2 = build_manifest(
        edition_id="factors-v0.1.0-alpha-2026-04-25",
        methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
        methodology_lead_approved_at=fixed_timestamp,
        build_timestamp=fixed_timestamp,
        git_commit="x",
    )
    h1 = {f.urn: f.record_sha256 for f in m1.factors}
    h2 = {f.urn: f.record_sha256 for f in m2.factors}
    assert h1 == h2


def test_manifest_sha256_is_reproducible(fixed_timestamp):
    m1 = build_manifest(
        edition_id="factors-v0.1.0-alpha-2026-04-25",
        methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
        methodology_lead_approved_at=fixed_timestamp,
        build_timestamp=fixed_timestamp,
        git_commit="x",
    )
    m2 = build_manifest(
        edition_id="factors-v0.1.0-alpha-2026-04-25",
        methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
        methodology_lead_approved_at=fixed_timestamp,
        build_timestamp=fixed_timestamp,
        git_commit="x",
    )
    assert m1.manifest_sha256 == m2.manifest_sha256
    # And it must equal a fresh recomputation:
    assert m1.compute_manifest_sha256() == m1.manifest_sha256


def test_manifest_sha256_changes_when_a_factor_hash_changes(manifest):
    """Tampering with ANY record_sha256 must invalidate the rollup hash."""
    original_hash = manifest.manifest_sha256

    # Perturb one factor and recompute.
    f0 = manifest.factors[0]
    manifest.factors[0] = FactorManifestEntry(
        urn=f0.urn,
        factor_pack_urn=f0.factor_pack_urn,
        source_urn=f0.source_urn,
        record_sha256="0" * 64,  # clearly different from any real sha256
    )
    new_hash = manifest.compute_manifest_sha256()
    assert new_hash != original_hash


def test_manifest_sha256_changes_when_a_source_count_changes(manifest):
    """Same: changing a source's factor_count must invalidate the hash."""
    original_hash = manifest.manifest_sha256
    s0 = manifest.sources[0]
    manifest.sources[0] = SourceManifestEntry(
        source_urn=s0.source_urn,
        source_id=s0.source_id,
        source_version=s0.source_version,
        publication_url=s0.publication_url,
        licence=s0.licence,
        factor_count=s0.factor_count + 1,  # lie about the count
        factor_pack_urns=list(s0.factor_pack_urns),
        parser_module=s0.parser_module,
        parser_version=s0.parser_version,
        parser_commit=s0.parser_commit,
    )
    assert manifest.compute_manifest_sha256() != original_hash


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


def test_canonical_json_is_sorted_and_compact():
    body = canonical_json_bytes({"b": 1, "a": 2})
    # No whitespace.
    assert b" " not in body
    # Sorted keys.
    assert body == b'{"a":2,"b":1}'


def test_canonical_json_is_platform_stable(tmp_path):
    """Writing canonical JSON in 'wb' must produce identical bytes regardless
    of the platform's default line endings — i.e. no embedded CR/LF.
    """
    payload = {"factors": [{"urn": "x", "v": 1}, {"urn": "y", "v": 2}]}
    body = canonical_json_bytes(payload)
    assert b"\r" not in body
    assert b"\n" not in body  # no trailing newline either

    # Round-trip through a file in binary mode.
    p = tmp_path / "c.json"
    p.write_bytes(body)
    assert p.read_bytes() == body


def test_canonical_json_unicode_preserved():
    """ensure_ascii=False — non-ASCII bytes must round-trip cleanly."""
    body = canonical_json_bytes({"name": "natural gas — UK"})
    assert "—".encode("utf-8") in body


# ---------------------------------------------------------------------------
# Approver validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "methodology-lead@greenlang.io",  # no actor prefix
        "human:not-an-email",
        "human:",
        "user:methodology-lead@greenlang.io",  # wrong actor type
        "",
        None,
    ],
)
def test_bad_approver_format_raises(bad):
    with pytest.raises((ValueError, TypeError)):
        build_manifest(
            edition_id="factors-v0.1.0-alpha-2026-04-25",
            methodology_lead_approver=bad,  # type: ignore[arg-type]
            methodology_lead_approved_at="2026-04-25T12:00:00Z",
            build_timestamp="2026-04-25T12:00:00Z",
            git_commit="x",
        )


def test_good_approver_formats_accepted():
    for good in (
        "human:methodology-lead@greenlang.io",
        "bot:provenance-gate@greenlang.io",
        "human:a.b+c@example.co.uk",
    ):
        m = build_manifest(
            edition_id="factors-v0.1.0-alpha-2026-04-25",
            methodology_lead_approver=good,
            methodology_lead_approved_at="2026-04-25T12:00:00Z",
            build_timestamp="2026-04-25T12:00:00Z",
            git_commit="x",
        )
        assert m.methodology_lead_approver == good


def test_bad_edition_id_raises():
    with pytest.raises(ValueError):
        build_manifest(
            edition_id="factors-v0.2.0-alpha-2026-04-25",  # wrong major
            methodology_lead_approver=DEFAULT_METHODOLOGY_LEAD,
            methodology_lead_approved_at="2026-04-25T12:00:00Z",
            build_timestamp="2026-04-25T12:00:00Z",
            git_commit="x",
        )


# ---------------------------------------------------------------------------
# Round-trip: dict / from_dict / write_manifest / read
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip(manifest):
    d = manifest.to_dict()
    assert isinstance(d, dict)
    rebuilt = AlphaEditionManifest.from_dict(d)
    # The dataclasses are equal field-for-field.
    assert rebuilt.edition_id == manifest.edition_id
    assert rebuilt.manifest_sha256 == manifest.manifest_sha256
    assert len(rebuilt.factors) == len(manifest.factors)
    assert len(rebuilt.sources) == len(manifest.sources)


def test_write_manifest_creates_files_and_round_trips(tmp_path, manifest):
    out = tmp_path / "edition"
    manifest_path, sig_path = write_manifest(manifest, out)
    assert manifest_path.is_file()
    # No env key in this test -> placeholder.
    assert sig_path.name.endswith(PLACEHOLDER_SUFFIX)

    # MANIFEST_HASH.txt + RELEASE_NOTES.md also created.
    assert (out / "MANIFEST_HASH.txt").is_file()
    assert (out / "RELEASE_NOTES.md").is_file()

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["edition_id"] == manifest.edition_id
    assert on_disk["manifest_sha256"] == manifest.manifest_sha256
    assert len(on_disk["factors"]) == EXPECTED_TOTAL_FACTORS

    # MANIFEST_HASH.txt is "sha256:<hex>"
    h = (out / "MANIFEST_HASH.txt").read_text(encoding="utf-8").strip()
    assert h == f"sha256:{manifest.manifest_sha256}"


def test_release_notes_contain_summary_and_per_source_counts(manifest):
    notes = render_release_notes(manifest)
    assert manifest.edition_id in notes
    assert manifest.manifest_sha256 in notes
    assert "Per-source counts" in notes
    # Every source appears in the table.
    for s in manifest.sources:
        assert s.source_id in notes
    assert "Approved by" in notes
    assert "verify_manifest" in notes


# ---------------------------------------------------------------------------
# Signing / verification
# ---------------------------------------------------------------------------


def test_verify_manifest_real_signature_verifies(tmp_path, manifest):
    priv_pem, pub_pem = _make_pem_keypair()
    out = tmp_path / "signed"
    manifest_path, sig_path = write_manifest(
        manifest, out, private_key_pem=priv_pem.decode("utf-8")
    )
    # Real signature: file does NOT end in placeholder suffix.
    assert not sig_path.name.endswith(PLACEHOLDER_SUFFIX)
    assert sig_path.name == "manifest.json.sig"
    assert verify_manifest(manifest_path, sig_path, pub_pem) is True


def test_verify_manifest_tampered_manifest_fails(tmp_path, manifest):
    priv_pem, pub_pem = _make_pem_keypair()
    out = tmp_path / "tampered"
    manifest_path, sig_path = write_manifest(
        manifest, out, private_key_pem=priv_pem.decode("utf-8")
    )
    assert verify_manifest(manifest_path, sig_path, pub_pem) is True
    # Mutate one byte in the manifest.
    raw = manifest_path.read_bytes()
    manifest_path.write_bytes(raw.replace(b"factors-v0.1.0-alpha", b"factors-v0.1.1-alpha"))
    assert verify_manifest(manifest_path, sig_path, pub_pem) is False


def test_verify_manifest_placeholder_is_unsigned(tmp_path, manifest):
    out = tmp_path / "unsigned"
    # No PEM -> placeholder.
    manifest_path, sig_path = write_manifest(manifest, out, private_key_pem=None)
    assert sig_path.name.endswith(PLACEHOLDER_SUFFIX)
    # Public key is irrelevant here — placeholders are always unsigned.
    pub_pem = _make_pem_keypair()[1]
    assert verify_manifest(manifest_path, sig_path, pub_pem) is False


def test_verify_manifest_missing_signature_returns_false(tmp_path, manifest):
    out = tmp_path / "nosig"
    manifest_path, sig_path = write_manifest(manifest, out)
    # Delete BOTH the placeholder and try to verify against a sibling that
    # doesn't exist.
    sig_path.unlink()
    pub_pem = _make_pem_keypair()[1]
    assert verify_manifest(manifest_path, out / "manifest.json.sig", pub_pem) is False


def test_verify_manifest_garbage_signature_returns_false(tmp_path, manifest):
    priv_pem, pub_pem = _make_pem_keypair()
    out = tmp_path / "garbage"
    manifest_path, sig_path = write_manifest(
        manifest, out, private_key_pem=priv_pem.decode("utf-8")
    )
    # Replace signature with random bytes (still valid base64).
    sig_path.write_bytes(base64.b64encode(b"\x00" * 64))
    assert verify_manifest(manifest_path, sig_path, pub_pem) is False


def test_verify_manifest_wrong_pubkey_returns_false(tmp_path, manifest):
    priv_pem, _ = _make_pem_keypair()
    _, other_pub = _make_pem_keypair()  # different key
    out = tmp_path / "wrongkey"
    manifest_path, sig_path = write_manifest(
        manifest, out, private_key_pem=priv_pem.decode("utf-8")
    )
    assert verify_manifest(manifest_path, sig_path, other_pub) is False


# ---------------------------------------------------------------------------
# Sanity: order independence
# ---------------------------------------------------------------------------


def test_factors_are_sorted_by_urn_for_determinism(manifest):
    urns = [f.urn for f in manifest.factors]
    assert urns == sorted(urns)


def test_sources_are_sorted_by_source_id_for_determinism(manifest):
    ids = [s.source_id for s in manifest.sources]
    assert ids == sorted(ids)
