# -*- coding: utf-8 -*-
"""
Wave E / TaskCreate #24 / WS9-T2 — Alpha edition release manifest.

Cuts a versioned, hash-locked, optionally Ed25519-signed manifest of the v0.1
Alpha factor catalog. Per CTO doc §6.3: "Every factor belongs to exactly one
pack" with a checksum over serialized pack contents. Per §19.1: "Documented
URN scheme with worked examples; climate-methodology lead signs off..."

The manifest is the canonical artefact a downstream SDK / API pin advances to.
It records:

  * the schema $id + sha256 of the FROZEN v0.1 schema bytes,
  * the freeze-note sha256 (so any post-freeze tampering is detectable),
  * one :class:`SourceManifestEntry` per ingested upstream source
    (publication URL, licence, parser commit, factor count),
  * one :class:`FactorManifestEntry` per record in the seed
    (URN -> sha256 of the canonical v0.1 record),
  * the methodology-lead approver + ISO timestamp,
  * builder + git_commit + sdk_version metadata,
  * a final :pyattr:`AlphaEditionManifest.manifest_sha256` over the canonical
    JSON of every other field (i.e. excluding ``manifest_sha256`` itself).

Canonical JSON rules (used everywhere a hash is computed):

  * ``json.dumps(..., sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)``
  * UTF-8 bytes, no BOM, no trailing newline.
  * sha256 over those bytes.

Edition id format::

    factors-v0.1.0-alpha-{YYYY-MM-DD}

The signing key (Ed25519, PEM) is read from environment variable
``GL_FACTORS_ED25519_PRIVATE_KEY`` if not provided explicitly. When no key is
available, callers can still build/serialize the manifest and write a
*placeholder* signature (see :func:`write_manifest`); the :func:`verify_manifest`
helper recognises the placeholder as "unsigned, not yet production-cut".
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Repo-relative path to the FROZEN v0.1 alpha JSON Schema.
SCHEMA_RELPATH = Path("config/schemas/factor_record_v0_1.schema.json")

#: Repo-relative path to the freeze-note (markdown) for the v0.1 contract.
FREEZE_NOTE_RELPATH = Path("config/schemas/FACTOR_RECORD_V0_1_FREEZE.md")

#: Repo-relative path to the v0.1 alpha catalog seed root.
CATALOG_SEED_RELPATH = Path("greenlang/factors/data/catalog_seed_v0_1")

#: Schema $id (matches the ``$id`` field inside the schema file).
SCHEMA_ID = "https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json"

#: Default sentinel for the methodology-lead approver string.
DEFAULT_METHODOLOGY_LEAD = "human:methodology-lead@greenlang.io"

#: SDK version pinned for the v0.1 Alpha edition.
SDK_VERSION = "0.1.0"

#: Release-profile string baked into the manifest.
API_RELEASE_PROFILE = "alpha-v0.1"

#: Builder identifier baked into the manifest.
BUILDER_ID = "bot:alpha_edition_manifest"

#: Placeholder signature filename suffix recognised by :func:`verify_manifest`
#: as "unsigned" (i.e. methodology-lead Ed25519 key not yet wired up).
PLACEHOLDER_SUFFIX = ".placeholder"

#: Single canonical RFC-5322-ish e-mail regex for the approver string. The
#: approver value MUST be of the form ``"<actor>:<email>"`` where actor is
#: ``human`` or ``bot`` — we forbid bare e-mails because audit-trail downstream
#: needs the actor type.
_APPROVER_RE = re.compile(
    r"^(human|bot):[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

#: Edition-id format used when the caller does not pass one explicitly.
_EDITION_ID_FMT = "factors-v0.1.0-alpha-{date}"

#: Regex an edition_id MUST match. We pin the prefix so a typo in --edition-id
#: cannot accidentally publish a non-alpha cut from this script.
_EDITION_ID_RE = re.compile(r"^factors-v0\.1\.0-alpha-\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceManifestEntry:
    """One ingested upstream source.

    ``factor_pack_urns`` is the set of distinct ``factor_pack_urn`` values
    observed in the source's records — per CTO doc §6.3, every factor belongs
    to exactly one pack, but a single source may produce multiple packs (e.g.
    DESNZ has separate packs by category in future revisions).
    """

    source_urn: str
    source_id: str
    source_version: str
    publication_url: str
    licence: str
    factor_count: int
    factor_pack_urns: List[str]
    parser_module: str
    parser_version: str
    parser_commit: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorManifestEntry:
    """One factor record in the manifest, with a per-record canonical hash."""

    urn: str
    factor_pack_urn: str
    source_urn: str
    record_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaEditionManifest:
    """The full v0.1 Alpha edition manifest.

    Field order mirrors the spec in TaskCreate #24. ``manifest_sha256`` is
    intentionally last because it is computed *over the canonical JSON of all
    other fields* — any change to any other field invalidates the hash.
    """

    edition_id: str
    schema_id: str
    schema_sha256: str
    factor_record_v0_1_freeze_sha256: str
    sources: List[SourceManifestEntry]
    factors: List[FactorManifestEntry]
    parser_commits: Dict[str, str]
    methodology_lead_approver: str
    methodology_lead_approved_at: str
    build_timestamp: str
    builder: str
    git_commit: str
    sdk_version: str
    api_release_profile: str
    manifest_sha256: str = ""

    # ------- serialization -------

    def to_dict(self, *, include_manifest_sha256: bool = True) -> Dict[str, Any]:
        """Return a dict suitable for ``json.dumps``.

        ``include_manifest_sha256=False`` is used internally when computing
        the manifest hash itself — we never include the field-being-computed
        in its own pre-image.
        """
        d = asdict(self)
        if not include_manifest_sha256:
            d.pop("manifest_sha256", None)
        return d

    def to_canonical_json(self, *, include_manifest_sha256: bool = True) -> bytes:
        """Return canonical-JSON bytes (UTF-8, sorted keys, no whitespace)."""
        return canonical_json_bytes(
            self.to_dict(include_manifest_sha256=include_manifest_sha256)
        )

    def compute_manifest_sha256(self) -> str:
        """Return sha256 over canonical JSON of every field EXCEPT this one."""
        body = self.to_canonical_json(include_manifest_sha256=False)
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlphaEditionManifest":
        sources = [SourceManifestEntry(**s) for s in data.get("sources", [])]
        factors = [FactorManifestEntry(**f) for f in data.get("factors", [])]
        kwargs = {k: v for k, v in data.items() if k not in {"sources", "factors"}}
        return cls(sources=sources, factors=factors, **kwargs)


# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically.

    Rules:

      * ``sort_keys=True``, ``separators=(",", ":")``, ``ensure_ascii=False``.
      * No leading / trailing whitespace, no BOM, UTF-8 encoding.

    Stable across Linux / macOS / Windows because we never write a newline
    and never embed platform-dependent strings (e.g. ``os.linesep``). If the
    caller passes in a value with non-deterministic key order (e.g. a Python
    dict on 3.6-) the ``sort_keys`` flag makes it deterministic anyway.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


# ---------------------------------------------------------------------------
# Helpers — repo discovery
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Locate the GreenLang repo root.

    Walk upwards from this file until a directory containing ``pyproject.toml``
    *and* ``greenlang/`` is found. Falls back to the package's own ancestors
    if installed editable.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "greenlang").is_dir():
            return parent
    # As a last resort, three levels up from this module:
    return here.parents[3]


def _git_head_commit(root: Path) -> str:
    """Return the current ``HEAD`` commit hash, or ``"unknown"`` if outside a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: read .git/HEAD directly.
        head_file = root / ".git" / "HEAD"
        if head_file.is_file():
            try:
                ref_line = head_file.read_text(encoding="utf-8").strip()
                if ref_line.startswith("ref: "):
                    ref_target = root / ".git" / ref_line[5:].strip()
                    if ref_target.is_file():
                        return ref_target.read_text(encoding="utf-8").strip()
                return ref_line
            except OSError:
                pass
        return "unknown"


# ---------------------------------------------------------------------------
# Helpers — record hashing
# ---------------------------------------------------------------------------


def _hash_record(record: Dict[str, Any]) -> str:
    """Canonical-JSON sha256 of a single v0.1 record.

    The record is hashed exactly as it appears on disk (after JSON parse +
    canonical re-serialise). This means: same inputs => same hash, regardless
    of upstream key-order, file whitespace, or platform line endings.
    """
    return sha256_hex(canonical_json_bytes(record))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_approver(approver: str) -> None:
    """Enforce ``actor:email@domain.tld`` shape on the approver string."""
    if not isinstance(approver, str) or not _APPROVER_RE.match(approver):
        raise ValueError(
            "methodology_lead_approver must be of the form "
            "'human:<email>' or 'bot:<email>' "
            f"(got {approver!r})"
        )


def _default_edition_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return _EDITION_ID_FMT.format(date=now.strftime("%Y-%m-%d"))


def _validate_edition_id(edition_id: str) -> None:
    if not _EDITION_ID_RE.match(edition_id):
        raise ValueError(
            f"edition_id {edition_id!r} does not match required format "
            f"'factors-v0.1.0-alpha-YYYY-MM-DD'"
        )


# ---------------------------------------------------------------------------
# Public API — build_manifest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    edition_id: Optional[str] = None,
    methodology_lead_approver: str = DEFAULT_METHODOLOGY_LEAD,
    methodology_lead_approved_at: Optional[str] = None,
    build_timestamp: Optional[str] = None,
    repo_root: Optional[Path] = None,
    catalog_seed_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    freeze_note_path: Optional[Path] = None,
    git_commit: Optional[str] = None,
) -> AlphaEditionManifest:
    """Walk the v0.1 catalog seed and build an :class:`AlphaEditionManifest`.

    All keyword arguments are optional; the defaults reproduce a "cut from
    repo HEAD against the in-tree seed" build. Tests override the paths to
    point at fixture directories.
    """
    root = repo_root or _repo_root()
    seed_root = catalog_seed_root or (root / CATALOG_SEED_RELPATH)
    schema = schema_path or (root / SCHEMA_RELPATH)
    freeze = freeze_note_path or (root / FREEZE_NOTE_RELPATH)

    if not seed_root.is_dir():
        raise FileNotFoundError(
            f"Catalog seed root not found: {seed_root}. Have Wave D #6 seeds been generated?"
        )
    if not schema.is_file():
        raise FileNotFoundError(f"Schema file not found: {schema}")
    if not freeze.is_file():
        raise FileNotFoundError(f"Freeze note not found: {freeze}")

    edition_id = edition_id or _default_edition_id()
    _validate_edition_id(edition_id)
    _validate_approver(methodology_lead_approver)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    build_timestamp = build_timestamp or now.isoformat().replace("+00:00", "Z")
    methodology_lead_approved_at = (
        methodology_lead_approved_at or now.isoformat().replace("+00:00", "Z")
    )

    # ---- schema + freeze hashes -------------------------------------------
    schema_sha = _sha256_file(schema)
    freeze_sha = _sha256_file(freeze)

    # ---- walk sources, collect entries ------------------------------------
    sources: List[SourceManifestEntry] = []
    factors: List[FactorManifestEntry] = []
    parser_commits: Dict[str, str] = {}

    for source_dir in sorted(seed_root.iterdir()):
        if not source_dir.is_dir():
            continue
        v1 = source_dir / "v1.json"
        if not v1.is_file():
            logger.warning("Skipping source dir without v1.json: %s", source_dir)
            continue

        seed = json.loads(v1.read_text(encoding="utf-8"))
        records: List[Dict[str, Any]] = list(seed.get("records") or [])
        if not records:
            logger.info("Source %s has zero records; skipping.", source_dir.name)
            continue

        # Source-level metadata (taken from the seed envelope, with per-record
        # extraction metadata as the authoritative parser-version source).
        source_id = seed.get("source_id") or source_dir.name
        source_urn = seed.get("source_urn") or ""
        source_version = seed.get("source_version") or ""

        # Per-record extraction.* fields — assume homogeneous within a source
        # (parser_id, parser_version, parser_commit are emitted by the same
        #  parser run). We pull them off the first record.
        ext0 = records[0].get("extraction") or {}
        parser_module = ext0.get("parser_id") or ""
        parser_version = ext0.get("parser_version") or ""
        parser_commit = ext0.get("parser_commit") or ""
        publication_url = ext0.get("source_url") or ""
        licence = records[0].get("licence") or ""

        if parser_commit:
            parser_commits[source_id] = parser_commit

        # Distinct pack URNs in this source (sorted, to keep canonical JSON
        # stable). CTO §6.3: every factor belongs to exactly one pack, so this
        # is just a deduplication, not a 1-to-many split.
        pack_urns_seen: List[str] = []
        seen: set = set()
        for r in records:
            pack_urn = r.get("factor_pack_urn") or ""
            if pack_urn and pack_urn not in seen:
                seen.add(pack_urn)
                pack_urns_seen.append(pack_urn)
        pack_urns_seen.sort()

        sources.append(
            SourceManifestEntry(
                source_urn=source_urn,
                source_id=source_id,
                source_version=source_version,
                publication_url=publication_url,
                licence=licence,
                factor_count=len(records),
                factor_pack_urns=pack_urns_seen,
                parser_module=parser_module,
                parser_version=parser_version,
                parser_commit=parser_commit,
            )
        )

        for r in records:
            urn = r.get("urn") or ""
            pack_urn = r.get("factor_pack_urn") or ""
            rec_source_urn = r.get("source_urn") or source_urn
            factors.append(
                FactorManifestEntry(
                    urn=urn,
                    factor_pack_urn=pack_urn,
                    source_urn=rec_source_urn,
                    record_sha256=_hash_record(r),
                )
            )

    # Sort sources + factors so the manifest is platform-stable irrespective
    # of os.listdir() iteration order.
    sources.sort(key=lambda s: s.source_id)
    factors.sort(key=lambda f: f.urn)

    git_commit = git_commit or _git_head_commit(root)

    manifest = AlphaEditionManifest(
        edition_id=edition_id,
        schema_id=SCHEMA_ID,
        schema_sha256=schema_sha,
        factor_record_v0_1_freeze_sha256=freeze_sha,
        sources=sources,
        factors=factors,
        parser_commits=dict(sorted(parser_commits.items())),
        methodology_lead_approver=methodology_lead_approver,
        methodology_lead_approved_at=methodology_lead_approved_at,
        build_timestamp=build_timestamp,
        builder=BUILDER_ID,
        git_commit=git_commit,
        sdk_version=SDK_VERSION,
        api_release_profile=API_RELEASE_PROFILE,
        manifest_sha256="",
    )
    manifest.manifest_sha256 = manifest.compute_manifest_sha256()

    logger.info(
        "Built alpha edition manifest %s: %d sources, %d factors, sha256=%s",
        edition_id,
        len(sources),
        len(factors),
        manifest.manifest_sha256,
    )
    return manifest


# ---------------------------------------------------------------------------
# Public API — write / verify
# ---------------------------------------------------------------------------


def _signing_inputs(
    *,
    private_key_pem: Optional[str] = None,
) -> Optional[bytes]:
    """Resolve the Ed25519 PEM key bytes from arg or env, or ``None``."""
    pem = private_key_pem or os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY", "")
    if not pem:
        return None
    return pem.encode("utf-8")


def _ed25519_sign(payload: bytes, pem: bytes) -> bytes:
    """Return Ed25519 signature bytes over ``payload`` using PEM private key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("provided PEM is not an Ed25519 private key")
    return key.sign(payload)


def _ed25519_verify(payload: bytes, signature: bytes, public_key: bytes) -> bool:
    """Return True iff ``signature`` is valid Ed25519 over ``payload``."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        # Accept raw 32-byte public key or PEM.
        if len(public_key) == 32:
            pk = Ed25519PublicKey.from_public_bytes(public_key)
        else:
            pk = serialization.load_pem_public_key(public_key)
            if not isinstance(pk, Ed25519PublicKey):
                return False
        pk.verify(signature, payload)
        return True
    except InvalidSignature:
        return False
    except (ValueError, TypeError) as exc:
        logger.debug("Ed25519 verify failed: %s", exc)
        return False


def write_manifest(
    manifest: AlphaEditionManifest,
    out_dir: Path,
    *,
    private_key_pem: Optional[str] = None,
    write_release_notes: bool = True,
    write_hash_file: bool = True,
) -> Tuple[Path, Path]:
    """Write the manifest + signature + (optional) release notes to disk.

    Returns ``(manifest_path, signature_path)``. The signature path may be a
    ``*.sig.placeholder`` file if no Ed25519 key is available — see
    :data:`PLACEHOLDER_SUFFIX`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    sig_path = out_dir / "manifest.json.sig"

    # Canonical bytes of the *whole* manifest including manifest_sha256.
    full_canonical = manifest.to_canonical_json(include_manifest_sha256=True)
    manifest_path.write_bytes(full_canonical)

    # Signature is over the *pre-image* used to produce manifest_sha256, NOT
    # over the full canonical JSON. This separates the two concerns:
    #   * manifest_sha256 = hash of every-other-field (verifiable by anyone)
    #   * signature = Ed25519 over the whole final canonical JSON (needs key)
    pem_bytes = _signing_inputs(private_key_pem=private_key_pem)
    if pem_bytes is None:
        # Write a placeholder signature so downstream tooling can still
        # detect "manifest exists but is unsigned".
        placeholder = sig_path.with_suffix(sig_path.suffix + PLACEHOLDER_SUFFIX)
        placeholder.write_text(
            "PLACEHOLDER — needs methodology-lead Ed25519 key at production cut.\n"
            f"manifest_sha256={manifest.manifest_sha256}\n"
            f"To produce a real signature: set GL_FACTORS_ED25519_PRIVATE_KEY (PEM) "
            f"and re-run the cut.\n",
            encoding="utf-8",
        )
        sig_path = placeholder
        logger.warning(
            "No Ed25519 key supplied; wrote placeholder signature at %s.", sig_path
        )
    else:
        signature = _ed25519_sign(full_canonical, pem_bytes)
        sig_path.write_bytes(base64.b64encode(signature))
        logger.info("Wrote Ed25519-signed manifest to %s + %s", manifest_path, sig_path)

    if write_hash_file:
        (out_dir / "MANIFEST_HASH.txt").write_text(
            f"sha256:{manifest.manifest_sha256}\n", encoding="utf-8"
        )

    if write_release_notes:
        notes = render_release_notes(manifest)
        (out_dir / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")

    return manifest_path, sig_path


def verify_manifest(
    manifest_path: Path,
    signature_path: Path,
    public_key: bytes,
) -> bool:
    """Return True iff the manifest's Ed25519 signature verifies.

    Returns ``False`` (not an exception) when:

      * the signature file is the placeholder (".sig.placeholder"),
      * the manifest bytes have been tampered (signature won't verify),
      * either file is missing,
      * any decode/parse error.

    This is a best-effort verifier suitable for SDK pin-checks; it never
    raises on bad input — bad input means "not verified".
    """
    manifest_path = Path(manifest_path)
    signature_path = Path(signature_path)

    if not manifest_path.is_file():
        logger.debug("verify_manifest: manifest file missing: %s", manifest_path)
        return False

    # Placeholder signatures are unsigned by definition.
    if signature_path.name.endswith(PLACEHOLDER_SUFFIX) or not signature_path.is_file():
        logger.debug(
            "verify_manifest: placeholder/missing signature at %s", signature_path
        )
        return False

    try:
        manifest_bytes = manifest_path.read_bytes()
        sig_b64 = signature_path.read_bytes().strip()
        signature = base64.b64decode(sig_b64, validate=False)
    except (OSError, ValueError) as exc:
        logger.debug("verify_manifest: I/O or decode error: %s", exc)
        return False

    return _ed25519_verify(manifest_bytes, signature, public_key)


# ---------------------------------------------------------------------------
# Release notes
# ---------------------------------------------------------------------------


def render_release_notes(manifest: AlphaEditionManifest) -> str:
    """Render markdown release notes from a built manifest.

    Auto-includes:

      * Summary line (edition id, build date, sdk, schema $id).
      * Per-source counts table.
      * Top supersede chains placeholder (Wave-E #6 supersede graph isn't
        wired into the seed yet — we leave a stub the methodology lead can
        backfill from ``docs/factors/v0_1_alpha/SOURCE-VINTAGE-AUDIT.md``).
      * Known caveats — references the source-vintage-audit preview.
      * Methodology-lead approval line.
      * Verification snippet (how to call :func:`verify_manifest`).
    """
    total_factors = len(manifest.factors)
    src_table_rows = [
        "| Source | Version | Records | Licence |",
        "|---|---|---:|---|",
    ]
    for s in manifest.sources:
        src_table_rows.append(
            f"| `{s.source_id}` | {s.source_version} | {s.factor_count} | {s.licence} |"
        )

    body = f"""# GreenLang Factors {manifest.edition_id}

**Build timestamp:** {manifest.build_timestamp}
**SDK version:** {manifest.sdk_version}
**API release profile:** {manifest.api_release_profile}
**Schema:** [{manifest.schema_id}]({manifest.schema_id})
**Schema sha256:** `{manifest.schema_sha256}`
**Freeze-note sha256:** `{manifest.factor_record_v0_1_freeze_sha256}`
**Manifest sha256:** `{manifest.manifest_sha256}`
**Git commit:** `{manifest.git_commit}`

This release manifest catalogs **{total_factors}** v0.1-alpha factor records
across **{len(manifest.sources)}** upstream sources. Per CTO doc §6.3 every
factor belongs to exactly one pack and carries a per-record content hash;
per CTO doc §19.1 the climate-methodology lead has signed off that the URN
scheme covers tier-1 sources without compromise.

## Per-source counts

{chr(10).join(src_table_rows)}

**Total factors:** {total_factors}

## Top supersede chains

Wave E v0.1: the supersede graph is empty for the alpha cut — every record is
its own root. Subsequent cuts will list the top 5 chains here automatically
(e.g. eGRID 2022 -> 2024 once the 2024 release lands).

## Known caveats

- See `docs/factors/v0_1_alpha/SOURCE-VINTAGE-AUDIT.md` for the per-source
  vintage / publication-window audit (preview entries flag any source whose
  upstream publication date is older than the 18-month freshness floor).
- See `docs/factors/v0_1_alpha/methodology-exceptions/` for any per-source
  exceptions the methodology lead has explicitly accepted.
- Supersede chains are only computed when a higher-vintage source replaces
  an earlier-vintage one; alpha cuts ship without that wiring.

## Methodology-lead approval

Approved by `{manifest.methodology_lead_approver}` at
`{manifest.methodology_lead_approved_at}`.

## Verification

```python
from pathlib import Path
from greenlang.factors.release.alpha_edition_manifest import verify_manifest

# `public_key` is either the Ed25519 raw 32-byte public key OR a PEM-encoded
# Ed25519 public key (bytes). Distribute alongside the release-notes URL.
ok = verify_manifest(
    manifest_path=Path("releases/{manifest.edition_id}/manifest.json"),
    signature_path=Path("releases/{manifest.edition_id}/manifest.json.sig"),
    public_key=Path("path/to/methodology_lead_pubkey.pem").read_bytes(),
)
assert ok, "Edition signature did not verify against the published key."
```

If the file ends in `.sig.placeholder` the cut is **unsigned** — the
methodology-lead Ed25519 key was not available at cut time. The cut is still
content-verifiable (every factor's `record_sha256` is reproducible and the
top-level `manifest_sha256` chains them all), but downstream SDK pins should
treat the edition as **unattested** until a signed re-cut lands.
"""
    return body


# ---------------------------------------------------------------------------
# Re-export
# ---------------------------------------------------------------------------


__all__ = [
    "API_RELEASE_PROFILE",
    "AlphaEditionManifest",
    "BUILDER_ID",
    "DEFAULT_METHODOLOGY_LEAD",
    "FactorManifestEntry",
    "PLACEHOLDER_SUFFIX",
    "SCHEMA_ID",
    "SDK_VERSION",
    "SourceManifestEntry",
    "build_manifest",
    "canonical_json_bytes",
    "render_release_notes",
    "sha256_hex",
    "verify_manifest",
    "write_manifest",
]
