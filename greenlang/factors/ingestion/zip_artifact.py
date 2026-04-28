# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — zip artifact handler for the EcoSpold2 / MRIO family.

Why this module exists
----------------------
The ecoinvent (EcoSpold2) and EXIOBASE MRIO sources both ship as
**single-file zip bundles** containing thousands of inner XML / CSV
files. The Phase 3 ingestion runner stores the raw zip as one
:class:`StoredArtifact` (so the SHA-256 pin survives through publish),
but the parser layer needs the *expanded* tree. This module provides a
single, audited expansion utility that:

  * Refuses to expand zip-bombs (max uncompressed bytes + max member
    count + per-member ratio limit).
  * Pins the original bundle's SHA-256 + URI on the returned dataclass
    so the parser can stamp ``extraction.raw_artifact_uri`` /
    ``extraction.raw_artifact_sha256`` on every emitted record (gate 6
    provenance completeness).
  * Returns a deterministic, sorted ``member_paths`` list so iteration
    order is reproducible across hosts and Python versions.

The expansion is defensive: it never trusts the zip's stated
``file_size`` field and instead computes the running uncompressed total
from the actually-extracted bytes. This catches the classic "zip
declares 1 KB, member is actually 10 GB" attack.

Determinism contract
--------------------
* Member ordering: ``sorted(zip.namelist())`` — alphabetical, stable
  across hosts.
* Date stamping: callers pin ``ZipInfo.date_time = (2026, 4, 28, 0, 0, 0)``
  in fixture builders; this module does not depend on date_time.
* Output path: extracted under ``dest_dir / member_path`` preserving the
  zip's directory structure. Directory entries (trailing slash) are
  created but contribute zero bytes.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Block 3: ecospold_mrio family
  contract" (Wave 2.5).
- ``greenlang.factors.ingestion.exceptions.ArtifactStoreError`` —
  raised on zip-bomb defence trip and on store-side checksum failures.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from greenlang.factors.ingestion.exceptions import ArtifactStoreError

logger = logging.getLogger(__name__)

__all__ = [
    "ZippedArtifact",
    "extract_zip_artifact",
    "DEFAULT_MAX_UNCOMPRESSED_BYTES",
    "DEFAULT_MAX_MEMBERS",
    "DEFAULT_MAX_COMPRESSION_RATIO",
]


#: Default uncompressed-byte budget per zip artifact. EXIOBASE v3.8.2 is
#: ~250 MB compressed and ~1.5 GB uncompressed; ecoinvent 3.10 cutoff is
#: ~50 MB compressed and ~600 MB uncompressed. 500 MB is generous for
#: synthetic test fixtures (which fit in a few KB) and conservative for
#: the production payloads (callers can override).
DEFAULT_MAX_UNCOMPRESSED_BYTES: int = 500 * 1024 * 1024  # 500 MB

#: Hard cap on member count. ecoinvent ships ~25k .spold files per
#: system model; EXIOBASE has ~50 core CSV files. 100k is a generous
#: safety margin without allowing a "100M tiny files" exhaustion attack.
DEFAULT_MAX_MEMBERS: int = 100_000

#: Per-member compression ratio (uncompressed / compressed). Anything
#: above ~1024:1 is the canonical zip-bomb signature (a billion-laughs
#: equivalent for binary). Real text data (XML / CSV) compresses ~5-10:1.
DEFAULT_MAX_COMPRESSION_RATIO: int = 1024


@dataclass
class ZippedArtifact:
    """Result of a successful, bomb-checked zip expansion.

    Attributes
    ----------
    bundle_uri:
        The ``file://`` (or future ``s3://``) URI the runner stored the
        original zip at. Pinned on every emitted factor record's
        ``extraction.raw_artifact_uri`` so gate 6 (provenance) finds
        the source-of-truth pin.
    bundle_sha256:
        The SHA-256 of the original zip (computed inside this function;
        callers MUST verify it matches the runner-recorded
        :class:`StoredArtifact.sha256` before proceeding to parse).
    member_paths:
        Sorted list of relative member paths (zip-internal names). Empty
        directory entries are excluded — only file members appear here.
    members_extracted_to:
        Filesystem root the expansion wrote under. Each member is at
        ``members_extracted_to / member_path`` after expansion.
    bytes_uncompressed:
        Running uncompressed total across all extracted members. Useful
        for ingestion observability (recorded in the run's stage receipt).
    """

    bundle_uri: str
    bundle_sha256: str
    member_paths: List[str] = field(default_factory=list)
    members_extracted_to: Optional[Path] = None
    bytes_uncompressed: int = 0


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------


def extract_zip_artifact(
    raw_bytes: bytes,
    dest_dir: Path,
    *,
    bundle_uri: str = "memory://zip",
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_compression_ratio: int = DEFAULT_MAX_COMPRESSION_RATIO,
) -> ZippedArtifact:
    """Expand a zip artifact under ``dest_dir`` with zip-bomb guards.

    Args:
        raw_bytes: Full raw zip bytes (already stored on the artifact
            store at ``bundle_uri``).
        dest_dir: Directory the members will be expanded under. Created
            if absent.
        bundle_uri: The artifact-store URI of the original zip. Embedded
            on the returned :class:`ZippedArtifact` so callers can stamp
            ``extraction.raw_artifact_uri`` on every emitted record.
        max_uncompressed_bytes: Hard cap on the running total of
            uncompressed bytes. Defaults to 500 MB. A trip raises
            :class:`ArtifactStoreError` BEFORE the offending member is
            written to disk.
        max_members: Hard cap on member count (defaults to 100k).
        max_compression_ratio: Per-member compression-ratio limit.
            Anything above the limit raises :class:`ArtifactStoreError`.

    Returns:
        A populated :class:`ZippedArtifact`. ``member_paths`` is sorted.

    Raises:
        ArtifactStoreError: On any of:
            * Truncated / malformed zip (caught from
              :class:`zipfile.BadZipFile`).
            * Member count exceeds ``max_members``.
            * Per-member compression ratio exceeds
              ``max_compression_ratio``.
            * Cumulative uncompressed bytes exceed
              ``max_uncompressed_bytes``.
            * Path-traversal attempt (``..`` segments or absolute paths
              in member names).
    """
    if not raw_bytes:
        raise ArtifactStoreError("zip artifact is empty", bytes_size=0)

    bundle_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Open the zip from in-memory bytes via BytesIO. zipfile accepts a
    # file-like object directly so we avoid an intermediate temp file.
    import io  # noqa: PLC0415 — local import keeps cold-start clean.

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes), mode="r")
    except zipfile.BadZipFile as exc:
        raise ArtifactStoreError(
            "malformed zip artifact: %s" % exc,
            bytes_size=len(raw_bytes),
        ) from exc

    artifact = ZippedArtifact(
        bundle_uri=bundle_uri,
        bundle_sha256=bundle_sha256,
        members_extracted_to=dest_dir,
    )

    try:
        infolist = zf.infolist()
    except Exception as exc:  # noqa: BLE001
        zf.close()
        raise ArtifactStoreError(
            "could not enumerate zip members: %s" % exc,
            bytes_size=len(raw_bytes),
        ) from exc

    # --- guard 1: member count --------------------------------------------
    if len(infolist) > max_members:
        zf.close()
        raise ArtifactStoreError(
            "zip member count %d exceeds limit %d (suspected bomb)"
            % (len(infolist), max_members),
            bytes_size=len(raw_bytes),
        )

    running_uncompressed = 0
    extracted_members: List[str] = []

    try:
        # Sort by ZipInfo.filename for deterministic iteration. ``zf.read``
        # accepts either a name or a ZipInfo; we stick with the name form.
        for info in sorted(infolist, key=lambda i: i.filename):
            name = info.filename
            # Skip empty directory entries (trailing slash).
            if name.endswith("/"):
                continue

            # --- guard 2: path traversal ----------------------------------
            normalized = _safe_member_path(name)
            if normalized is None:
                raise ArtifactStoreError(
                    "zip member %r escapes destination (traversal attempt)"
                    % name,
                    bytes_size=len(raw_bytes),
                )

            # --- guard 3: per-member compression ratio --------------------
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > max_compression_ratio:
                    raise ArtifactStoreError(
                        "zip member %r compression ratio %.1f > limit %d"
                        " (suspected bomb)"
                        % (name, ratio, max_compression_ratio),
                        bytes_size=len(raw_bytes),
                    )

            # --- guard 4: extracted-byte budget ---------------------------
            # Compute *projected* total before writing. We enforce on the
            # actually-read bytes too, so a lying header is caught.
            projected = running_uncompressed + info.file_size
            if projected > max_uncompressed_bytes:
                raise ArtifactStoreError(
                    "zip uncompressed total would exceed limit "
                    "(member %r pushes %d > %d)"
                    % (name, projected, max_uncompressed_bytes),
                    bytes_size=len(raw_bytes),
                )

            # Read + write. We intentionally read the full member into
            # memory rather than streaming because the per-member size
            # is already capped above. Streaming would catch a lying
            # header — but so does the post-write sanity check below.
            data = zf.read(name)
            actual_size = len(data)
            running_uncompressed += actual_size
            if running_uncompressed > max_uncompressed_bytes:
                raise ArtifactStoreError(
                    "zip uncompressed total exceeded limit during "
                    "extraction of %r (%d > %d)"
                    % (name, running_uncompressed, max_uncompressed_bytes),
                    bytes_size=len(raw_bytes),
                )

            target = dest_dir / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            extracted_members.append(normalized)

    finally:
        zf.close()

    artifact.member_paths = sorted(extracted_members)
    artifact.bytes_uncompressed = running_uncompressed
    logger.info(
        "extracted zip bundle sha=%s members=%d bytes=%d -> %s",
        bundle_sha256,
        len(artifact.member_paths),
        running_uncompressed,
        dest_dir,
    )
    return artifact


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_member_path(name: str) -> Optional[str]:
    """Return a normalised relative path, or None if the member escapes.

    Rejects:
      * Absolute paths (``/etc/passwd``, ``C:\\Windows\\...``).
      * Paths containing ``..`` segments.
      * Empty paths after normalisation.
    """
    if not name:
        return None
    # Reject Windows-style absolute drives and POSIX absolutes.
    if name.startswith("/") or name.startswith("\\"):
        return None
    if len(name) >= 2 and name[1] == ":":
        return None
    # Normalise separators.
    pure = name.replace("\\", "/")
    parts = [p for p in pure.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    if not parts:
        return None
    return "/".join(parts)
