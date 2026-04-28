"""Schema version registry for greenlang.factors.

Single source of truth for which schema $ids exist, when they took
effect, when their lock windows close, and which versions overlap.

Authoritative for dates; the policy doc references this registry.

The companion policy document
``docs/factors/schema/SCHEMA_EVOLUTION_POLICY.md`` is authoritative for
*rules* (what counts as additive/breaking, who must sign off, what the
lock and overlap windows mean). This module is authoritative for
*dates* (when each version's lock and overlap windows close). Both
artefacts MUST agree on the lock-month and overlap-month numbers
mandated by the CTO Phase 2 brief:

- v1.0 lock = 24 months from GA effective date.
- v1<->v2 overlap = 12 months.
- v2<->v3 overlap = 18 months.

CI gates and runtime code MUST read dates from this registry; do not
parse them out of the policy doc or the CHANGELOG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaVersion:
    """A registered factor-record schema version.

    Fields
    ------
    schema_id:
        The canonical JSON Schema ``$id`` URL. Permanent for the lifetime
        of any record produced against this version.
    version:
        Short version string, e.g. ``"0.1"`` or ``"1.0"``. The leading
        ``v`` is omitted; callers that need the policy-style ``vX.Y``
        format should prepend ``v`` themselves.
    status:
        One of ``frozen`` | ``draft`` | ``deprecated`` | ``removed``.
        ``frozen`` means GA-eligible and producible. ``draft`` means
        in-development; no production records may be produced. ``deprecated``
        means readable but not producible. ``removed`` means archival only.
    effective_date:
        ISO ``YYYY-MM-DD`` calendar date the version took effect. ``None``
        for drafts.
    supersedes:
        Tuple of ``schema_id`` URLs that this version supersedes. Empty
        tuple for the first version.
    lock_months:
        Number of calendar months from ``effective_date`` during which no
        breaking change is permitted. ``0`` for ``v0.x`` (alpha, no lock).
        ``24`` for ``v1.0`` per CTO mandate.
    overlap_with_next_months:
        Number of calendar months after the *next* major's effective date
        during which this version remains validatable. ``12`` for v1<->v2,
        ``18`` for v2<->v3 per CTO mandate. ``0`` for ``v0.x`` (no GA
        contract).
    changelog_uri:
        Repo-relative path + anchor pointing to the CHANGELOG entry that
        introduced this version. Used by audit tooling and the schema-
        discovery API.
    """

    schema_id: str
    version: str  # "0.1", "1.0", etc.
    status: str  # 'frozen' | 'draft' | 'deprecated' | 'removed'
    effective_date: Optional[str]  # ISO date or None for draft
    supersedes: Tuple[str, ...] = field(default_factory=tuple)
    lock_months: int = 0  # 0 for v0.x, 24 for v1.0, etc.
    overlap_with_next_months: int = 0
    changelog_uri: str = ""

    @property
    def lock_until(self) -> Optional[str]:
        """ISO date when the lock window closes; ``None`` if no lock applies.

        Computed as ``effective_date + lock_months`` using calendar-month
        arithmetic. Returns ``None`` for drafts (no effective date) and
        for versions with ``lock_months == 0`` (alpha, no lock).
        """
        if not self.effective_date or self.lock_months == 0:
            return None
        return _add_months_iso(self.effective_date, self.lock_months)

    @property
    def overlap_until(self) -> Optional[str]:
        """ISO date when the overlap window closes; ``None`` if no overlap applies.

        For a version that overlaps with its successor, callers conventionally
        compute overlap from the *successor's* effective date plus this
        version's ``overlap_with_next_months``. The dataclass-local helper
        below applies the offset to *this* version's own effective date for
        symmetry with ``lock_until``; production callers needing the
        successor-relative date should use
        :func:`overlap_until_for_successor`.
        """
        if not self.effective_date or self.overlap_with_next_months == 0:
            return None
        return _add_months_iso(self.effective_date, self.overlap_with_next_months)


# ---------------------------------------------------------------------------
# Date arithmetic helper
# ---------------------------------------------------------------------------


def _add_months_iso(iso_date: str, months: int) -> str:
    """Add ``months`` calendar months to ``iso_date`` (YYYY-MM-DD).

    Calendar-month arithmetic: keeps the day-of-month constant where
    possible. The CTO's lock/overlap windows are expressed in whole
    calendar months, not days, so this is the correct semantic.
    """
    y, m, d = (int(part) for part in iso_date.split("-"))
    total = m + months
    ny = y + (total - 1) // 12
    nm = ((total - 1) % 12) + 1
    return "%04d-%02d-%02d" % (ny, nm, d)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


REGISTRY: Dict[str, SchemaVersion] = {
    "v0_1": SchemaVersion(
        schema_id="https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json",
        version="0.1",
        status="frozen",
        effective_date="2026-04-25",
        supersedes=(),
        lock_months=0,  # v0.x has no lock
        overlap_with_next_months=0,
        changelog_uri="docs/factors/schema/CHANGELOG.md#v01---2026-04-25---additive",
    ),
    # v1.0 entry added when promoted from draft to frozen.
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_version(version_key: str) -> SchemaVersion:
    """Return the :class:`SchemaVersion` registered under ``version_key``.

    ``version_key`` is the registry dict key, e.g. ``"v0_1"`` or
    ``"v1_0"``. Raises :class:`KeyError` if the key is not registered.
    """
    if version_key not in REGISTRY:
        logger.error("Unknown schema version key: %s", version_key)
        raise KeyError(version_key)
    return REGISTRY[version_key]


def latest_frozen() -> SchemaVersion:
    """Return the highest-version ``frozen`` :class:`SchemaVersion`.

    "Highest" is determined by string-sorting the ``version`` field after
    splitting on ``.`` and converting each segment to int. Raises
    :class:`LookupError` if no frozen version is registered.
    """
    frozen_versions = [v for v in REGISTRY.values() if v.status == "frozen"]
    if not frozen_versions:
        logger.error("No frozen schema versions registered")
        raise LookupError("no frozen schema versions registered")
    frozen_versions.sort(key=_version_sort_key, reverse=True)
    return frozen_versions[0]


def is_locked(version_key: str, today: Optional[str] = None) -> bool:
    """Return ``True`` if the version is inside its lock window on ``today``.

    ``today`` is an ISO ``YYYY-MM-DD`` string; defaults to the system date.
    A version with ``lock_months == 0`` (e.g. v0.x alpha) is never locked
    and returns ``False``. A draft version (no ``effective_date``) is
    never locked and returns ``False``.
    """
    version = get_version(version_key)
    lock_until = version.lock_until
    if lock_until is None:
        return False
    today_iso = today if today is not None else date.today().isoformat()
    return today_iso < lock_until


def is_in_overlap_window(version_key: str, today: Optional[str] = None) -> bool:
    """Return ``True`` if the version is inside its overlap window on ``today``.

    The overlap window is the period after this version has been
    superseded but before its readability obligation expires. A version
    is in overlap iff:

    - There exists a successor in the registry (i.e. a version whose
      ``supersedes`` tuple contains this version's ``schema_id``).
    - The successor has an ``effective_date``.
    - Today is on or after that successor effective date AND strictly
      before ``successor.effective_date + this_version.overlap_with_next_months``.

    A version with ``overlap_with_next_months == 0`` (e.g. v0.x alpha) is
    never in an overlap window and returns ``False``.
    """
    version = get_version(version_key)
    if version.overlap_with_next_months == 0:
        return False
    successor = _find_successor(version)
    if successor is None or successor.effective_date is None:
        return False
    today_iso = today if today is not None else date.today().isoformat()
    overlap_end = _add_months_iso(
        successor.effective_date, version.overlap_with_next_months
    )
    return successor.effective_date <= today_iso < overlap_end


def all_active() -> List[SchemaVersion]:
    """Return all versions that are currently ``frozen`` or in overlap.

    A version is "active" iff it is currently producible (``frozen`` and
    not yet superseded) or readable under the overlap-window protocol
    (``deprecated`` but inside its overlap window). ``draft`` and
    ``removed`` versions are excluded.
    """
    today_iso = date.today().isoformat()
    out: List[SchemaVersion] = []
    for key, version in REGISTRY.items():
        if version.status == "frozen":
            out.append(version)
            continue
        if version.status == "deprecated" and is_in_overlap_window(key, today_iso):
            out.append(version)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _version_sort_key(v: SchemaVersion) -> Tuple[int, ...]:
    """Sort key for ``SchemaVersion``: tuple of ints from ``version`` field."""
    return tuple(int(part) for part in v.version.split("."))


def overlap_until_for_successor(
    version_key: str, successor_key: str
) -> Optional[str]:
    """Compute the overlap-end date for ``version_key`` given its successor.

    Returns ``successor.effective_date + version.overlap_with_next_months``
    as an ISO date, or ``None`` if either party lacks the necessary fields.
    """
    version = get_version(version_key)
    successor = get_version(successor_key)
    if successor.effective_date is None or version.overlap_with_next_months == 0:
        return None
    return _add_months_iso(
        successor.effective_date, version.overlap_with_next_months
    )


def _find_successor(version: SchemaVersion) -> Optional[SchemaVersion]:
    """Return the registry entry whose ``supersedes`` includes this version's ``schema_id``."""
    for candidate in REGISTRY.values():
        if version.schema_id in candidate.supersedes:
            return candidate
    return None
