# -*- coding: utf-8 -*-
"""Industry classification mappings (ISIC, NACE, NAICS, ANZSIC, JSIC).

This module loads the YAML taxonomy files under
``greenlang/factors/data/taxonomies/`` and exposes:

- :class:`IndustryCodeSystem` enum for the five supported systems.
- :class:`IndustryCode` record (system, code, label, parent_code, level).
- :class:`CodeCrosswalk` record (from_system, from_code, to_system, to_code,
  match_quality).
- Lookup helpers (:func:`lookup_industry_code`) with case-insensitive labels.
- Crosswalk helpers (:func:`crosswalk_code`) that translate a code between
  systems and are bidirectionally consistent.
- Sector-default emission-factor accessor for spend-based Scope 3.

ISIC Rev.4 (UN), NACE Rev.2 (EU), and NAICS 2022 (US Census) are in the
public domain. ANZSIC 2006 (Australian Bureau of Statistics / Stats NZ)
and JSIC Rev.14 (Japan Ministry of Internal Affairs) are copyright their
respective agencies but freely redistributable with attribution.
"""
from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from greenlang.factors.mapping.base import (
    MappingConfidence,
    MappingError,
    MappingResult,
    normalize_text,
)

logger = logging.getLogger(__name__)


try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IndustryCodeSystem(str, Enum):
    """Supported industrial classification systems."""

    ISIC = "isic"
    NACE = "nace"
    NAICS = "naics"
    ANZSIC = "anzsic"
    JSIC = "jsic"

    @classmethod
    def from_str(cls, name: str) -> "IndustryCodeSystem":
        if not name:
            raise MappingError("Empty industry code system name")
        needle = name.strip().lower()
        for member in cls:
            if member.value == needle:
                return member
        raise MappingError(
            "Unknown industry code system %r; expected one of %s"
            % (name, [m.value for m in cls])
        )


class CodeLevel(str, Enum):
    """Depth within a code system's hierarchy."""

    SECTION = "section"       # 1-letter (ISIC/NACE A-U), 2-digit NAICS sector
    DIVISION = "division"     # 2-digit
    GROUP = "group"           # 3-digit
    CLASS = "class"           # 4-digit (ISIC/NACE) or 5-digit (NAICS)
    NATIONAL = "national"     # 6-digit NAICS national industry


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class IndustryCode:
    """A single code entry in a classification system."""

    __slots__ = (
        "system", "code", "label", "parent_code", "level", "ef_default", "extras",
    )

    def __init__(
        self,
        system: IndustryCodeSystem,
        code: str,
        label: str,
        parent_code: Optional[str] = None,
        level: CodeLevel = CodeLevel.CLASS,
        ef_default: Optional[float] = None,
        extras: Optional[Dict[str, Any]] = None,
    ):
        self.system = system
        self.code = code
        self.label = label
        self.parent_code = parent_code
        self.level = level
        self.ef_default = float(ef_default) if ef_default is not None else None
        self.extras = extras or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system.value,
            "code": self.code,
            "label": self.label,
            "parent_code": self.parent_code,
            "level": self.level.value,
            "ef_default": self.ef_default,
            "extras": dict(self.extras),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"IndustryCode({self.system.value}:{self.code}, {self.label!r})"


class CodeCrosswalk:
    """One row of a code-system crosswalk (possibly 1-to-many)."""

    __slots__ = ("from_system", "from_code", "to_system", "to_code", "match_quality")

    def __init__(
        self,
        from_system: IndustryCodeSystem,
        from_code: str,
        to_system: IndustryCodeSystem,
        to_code: Any,  # str | list[str]
        match_quality: str,
    ):
        self.from_system = from_system
        self.from_code = from_code
        self.to_system = to_system
        self.to_code = to_code
        self.match_quality = match_quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_system": self.from_system.value,
            "from_code": self.from_code,
            "to_system": self.to_system.value,
            "to_code": self.to_code,
            "match_quality": self.match_quality,
        }


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------


_TAXONOMY_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "taxonomies"
)


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise MappingError("PyYAML is required for industry_codes loading")
    if not path.exists():
        raise MappingError("Taxonomy file not found: %s" % path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def load_codes(system: IndustryCodeSystem) -> Dict[str, IndustryCode]:
    """Load codes for one classification system.

    For ISIC / NACE / NAICS, reads the dedicated YAML file. For ANZSIC /
    JSIC the data is served from an embedded micro-taxonomy (see
    :data:`_ANZSIC_MIN` and :data:`_JSIC_MIN`) since redistribution of the
    full classifications would require upstream licensing we do not yet
    hold.
    """
    if system == IndustryCodeSystem.ISIC:
        return _load_isic()
    if system == IndustryCodeSystem.NACE:
        return _load_nace()
    if system == IndustryCodeSystem.NAICS:
        return _load_naics()
    if system == IndustryCodeSystem.ANZSIC:
        return _load_min(system, _ANZSIC_MIN)
    if system == IndustryCodeSystem.JSIC:
        return _load_min(system, _JSIC_MIN)
    raise MappingError("Unsupported industry code system: %s" % system)  # pragma: no cover


def _load_isic() -> Dict[str, IndustryCode]:
    payload = _load_yaml(_TAXONOMY_DIR / "isic_rev4.yaml")
    out: Dict[str, IndustryCode] = {}
    for letter, label in (payload.get("sections") or {}).items():
        out[str(letter)] = IndustryCode(
            system=IndustryCodeSystem.ISIC,
            code=str(letter),
            label=str(label),
            parent_code=None,
            level=CodeLevel.SECTION,
        )
    for code, body in (payload.get("divisions") or {}).items():
        out[str(code)] = IndustryCode(
            system=IndustryCodeSystem.ISIC,
            code=str(code),
            label=str(body.get("label", code)),
            parent_code=str(body.get("section")) if body.get("section") else None,
            level=CodeLevel.DIVISION,
        )
    for code, body in (payload.get("classes") or {}).items():
        out[str(code)] = IndustryCode(
            system=IndustryCodeSystem.ISIC,
            code=str(code),
            label=str(body.get("label", code)),
            parent_code=str(body.get("division")) if body.get("division") else None,
            level=CodeLevel.CLASS,
            ef_default=body.get("ef_default"),
        )
    logger.debug("Loaded %d ISIC entries", len(out))
    return out


def _load_nace() -> Dict[str, IndustryCode]:
    payload = _load_yaml(_TAXONOMY_DIR / "nace_rev2.yaml")
    out: Dict[str, IndustryCode] = {}
    for letter, label in (payload.get("sections") or {}).items():
        out[str(letter)] = IndustryCode(
            system=IndustryCodeSystem.NACE,
            code=str(letter),
            label=str(label),
            parent_code=None,
            level=CodeLevel.SECTION,
        )
    for code, body in (payload.get("divisions") or {}).items():
        out[str(code)] = IndustryCode(
            system=IndustryCodeSystem.NACE,
            code=str(code),
            label=str(body.get("label", code)),
            parent_code=str(body.get("section")) if body.get("section") else None,
            level=CodeLevel.DIVISION,
        )
    for code, body in (payload.get("classes") or {}).items():
        out[str(code)] = IndustryCode(
            system=IndustryCodeSystem.NACE,
            code=str(code),
            label=str(body.get("label", code)),
            parent_code=str(body.get("division")) if body.get("division") else None,
            level=CodeLevel.CLASS,
            ef_default=body.get("ef_default"),
        )
    logger.debug("Loaded %d NACE entries", len(out))
    return out


def _load_naics() -> Dict[str, IndustryCode]:
    payload = _load_yaml(_TAXONOMY_DIR / "naics_2022.yaml")
    out: Dict[str, IndustryCode] = {}
    for sec, label in (payload.get("sectors") or {}).items():
        out[str(sec)] = IndustryCode(
            system=IndustryCodeSystem.NAICS,
            code=str(sec),
            label=str(label),
            parent_code=None,
            level=CodeLevel.SECTION,
        )
    for sub, label in (payload.get("subsectors") or {}).items():
        parent = str(sub)[:2]
        out[str(sub)] = IndustryCode(
            system=IndustryCodeSystem.NAICS,
            code=str(sub),
            label=str(label),
            parent_code=parent,
            level=CodeLevel.GROUP,
        )
    for code, body in (payload.get("classes") or {}).items():
        code_s = str(code)
        parent = str(body.get("subsector") or code_s[:3])
        out[code_s] = IndustryCode(
            system=IndustryCodeSystem.NAICS,
            code=code_s,
            label=str(body.get("label", code_s)),
            parent_code=parent,
            level=CodeLevel.NATIONAL,
            ef_default=body.get("ef_default"),
        )
    logger.debug("Loaded %d NAICS entries", len(out))
    return out


# ANZSIC 2006 and JSIC Rev.14 - embedded minimal taxonomies covering the
# top-level Divisions and a representative set of classes. Full
# classifications can be dropped in later via additional YAML files.
_ANZSIC_MIN: Dict[str, Dict[str, Any]] = {
    # Divisions (A-S)
    "A": {"label": "Agriculture, Forestry and Fishing", "level": "section"},
    "B": {"label": "Mining", "level": "section"},
    "C": {"label": "Manufacturing", "level": "section"},
    "D": {"label": "Electricity, Gas, Water and Waste Services", "level": "section"},
    "E": {"label": "Construction", "level": "section"},
    "F": {"label": "Wholesale Trade", "level": "section"},
    "G": {"label": "Retail Trade", "level": "section"},
    "H": {"label": "Accommodation and Food Services", "level": "section"},
    "I": {"label": "Transport, Postal and Warehousing", "level": "section"},
    "J": {"label": "Information Media and Telecommunications", "level": "section"},
    "K": {"label": "Financial and Insurance Services", "level": "section"},
    "L": {"label": "Rental, Hiring and Real Estate Services", "level": "section"},
    "M": {"label": "Professional, Scientific and Technical Services", "level": "section"},
    "N": {"label": "Administrative and Support Services", "level": "section"},
    "O": {"label": "Public Administration and Safety", "level": "section"},
    "P": {"label": "Education and Training", "level": "section"},
    "Q": {"label": "Health Care and Social Assistance", "level": "section"},
    "R": {"label": "Arts and Recreation Services", "level": "section"},
    "S": {"label": "Other Services", "level": "section"},
    # A sample of subdivisions / classes
    "06": {"label": "Coal Mining", "level": "division", "parent": "B"},
    "07": {"label": "Oil and Gas Extraction", "level": "division", "parent": "B"},
    "08": {"label": "Metal Ore Mining", "level": "division", "parent": "B"},
    "11": {"label": "Food Product Manufacturing", "level": "division", "parent": "C"},
    "13": {"label": "Textile, Leather, Clothing and Footwear Manufacturing", "level": "division", "parent": "C"},
    "18": {"label": "Basic Chemical and Chemical Product Manufacturing", "level": "division", "parent": "C"},
    "20": {"label": "Primary Metal and Metal Product Manufacturing", "level": "division", "parent": "C"},
    "26": {"label": "Electricity Supply", "level": "division", "parent": "D"},
    "46": {"label": "Road Transport", "level": "division", "parent": "I"},
    "47": {"label": "Rail Transport", "level": "division", "parent": "I"},
    "62": {"label": "Finance", "level": "division", "parent": "K"},
    # A few 4-digit classes (illustrative)
    "0601": {"label": "Coal Mining", "level": "class", "parent": "06", "ef_default": 0.85},
    "1111": {"label": "Meat Processing", "level": "class", "parent": "11", "ef_default": 0.45},
    "1831": {"label": "Fertiliser Manufacturing", "level": "class", "parent": "18", "ef_default": 1.55},
    "2011": {"label": "Iron Smelting and Steel Manufacturing", "level": "class", "parent": "20", "ef_default": 1.85},
    "2021": {"label": "Alumina Production", "level": "class", "parent": "20", "ef_default": 1.60},
    "2022": {"label": "Aluminium Smelting", "level": "class", "parent": "20", "ef_default": 2.15},
    "2611": {"label": "Fossil Fuel Electricity Generation", "level": "class", "parent": "26", "ef_default": 1.05},
    "2612": {"label": "Hydro-Electricity Generation", "level": "class", "parent": "26", "ef_default": 0.04},
    "2619": {"label": "Other Electricity Generation", "level": "class", "parent": "26", "ef_default": 0.30},
    "4610": {"label": "Road Freight Transport", "level": "class", "parent": "46", "ef_default": 0.43},
    "4720": {"label": "Rail Freight Transport", "level": "class", "parent": "47", "ef_default": 0.18},
    "6221": {"label": "Banking", "level": "class", "parent": "62", "ef_default": 0.05},
    "6310": {"label": "Life Insurance", "level": "class", "parent": "62", "ef_default": 0.04},
}

_JSIC_MIN: Dict[str, Dict[str, Any]] = {
    # Divisions (A-T) - Japan Standard Industrial Classification Rev.14
    "A": {"label": "Agriculture and Forestry", "level": "section"},
    "B": {"label": "Fisheries", "level": "section"},
    "C": {"label": "Mining and Quarrying of Stone and Gravel", "level": "section"},
    "D": {"label": "Construction", "level": "section"},
    "E": {"label": "Manufacturing", "level": "section"},
    "F": {"label": "Electricity, Gas, Heat Supply and Water", "level": "section"},
    "G": {"label": "Information and Communications", "level": "section"},
    "H": {"label": "Transport and Postal Activities", "level": "section"},
    "I": {"label": "Wholesale and Retail Trade", "level": "section"},
    "J": {"label": "Finance and Insurance", "level": "section"},
    "K": {"label": "Real Estate and Goods Rental and Leasing", "level": "section"},
    "L": {"label": "Scientific Research, Professional and Technical Services", "level": "section"},
    "M": {"label": "Accommodations, Eating and Drinking Services", "level": "section"},
    "N": {"label": "Living-Related and Personal Services and Amusement", "level": "section"},
    "O": {"label": "Education, Learning Support", "level": "section"},
    "P": {"label": "Medical, Health Care and Welfare", "level": "section"},
    "Q": {"label": "Compound Services", "level": "section"},
    "R": {"label": "Services (not elsewhere classified)", "level": "section"},
    "S": {"label": "Government (except elsewhere classified)", "level": "section"},
    "T": {"label": "Industries Unable to Classify", "level": "section"},
    # Divisions (2-digit major groups - illustrative subset)
    "05": {"label": "Mining and Quarrying of Stone and Gravel", "level": "division", "parent": "C"},
    "09": {"label": "Manufacture of Food", "level": "division", "parent": "E"},
    "14": {"label": "Manufacture of Pulp, Paper and Paper Products", "level": "division", "parent": "E"},
    "16": {"label": "Manufacture of Chemical and Allied Products", "level": "division", "parent": "E"},
    "22": {"label": "Manufacture of Iron and Steel", "level": "division", "parent": "E"},
    "23": {"label": "Manufacture of Non-ferrous Metals and Products", "level": "division", "parent": "E"},
    "33": {"label": "Electricity", "level": "division", "parent": "F"},
    "44": {"label": "Road Passenger Transport", "level": "division", "parent": "H"},
    "45": {"label": "Road Freight Transport", "level": "division", "parent": "H"},
    "62": {"label": "Banking", "level": "division", "parent": "J"},
    # A handful of 4-digit classes
    "2211": {"label": "Pig iron and crude steel", "level": "class", "parent": "22", "ef_default": 1.85},
    "2321": {"label": "Aluminium smelting", "level": "class", "parent": "23", "ef_default": 2.15},
    "3311": {"label": "Electricity generation", "level": "class", "parent": "33", "ef_default": 0.72},
    "4511": {"label": "Freight road transport", "level": "class", "parent": "45", "ef_default": 0.42},
    "6211": {"label": "Banking", "level": "class", "parent": "62", "ef_default": 0.05},
}


def _load_min(
    system: IndustryCodeSystem, data: Dict[str, Dict[str, Any]]
) -> Dict[str, IndustryCode]:
    out: Dict[str, IndustryCode] = {}
    for code, body in data.items():
        try:
            level = CodeLevel(str(body.get("level", "class")))
        except ValueError:
            level = CodeLevel.CLASS
        out[code] = IndustryCode(
            system=system,
            code=code,
            label=str(body.get("label", code)),
            parent_code=body.get("parent"),
            level=level,
            ef_default=body.get("ef_default"),
        )
    return out


# ---------------------------------------------------------------------------
# Crosswalks
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_raw_crosswalk(
    from_system: IndustryCodeSystem, to_system: IndustryCodeSystem
) -> Dict[str, Any]:
    # Only ISIC-NACE and ISIC-NAICS have dedicated YAML files. Other pairs
    # are inferred by routing via ISIC.
    if from_system == IndustryCodeSystem.ISIC and to_system == IndustryCodeSystem.NACE:
        return _load_yaml(_TAXONOMY_DIR / "isic_to_nace_crosswalk.yaml")
    if from_system == IndustryCodeSystem.ISIC and to_system == IndustryCodeSystem.NAICS:
        return _load_yaml(_TAXONOMY_DIR / "isic_to_naics_crosswalk.yaml")
    raise MappingError(
        "No direct crosswalk file %s -> %s; use crosswalk_code() which "
        "routes via ISIC" % (from_system.value, to_system.value)
    )


def _invert_crosswalk(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """Given an ISIC->X raw crosswalk, return a flat inverse: X code -> [ISIC codes]."""
    reverse: Dict[str, List[str]] = {}
    for bucket in ("divisions", "classes"):
        for isic_code, body in (payload.get(bucket) or {}).items():
            to_codes = body.get("to")
            if isinstance(to_codes, list):
                for tc in to_codes:
                    reverse.setdefault(str(tc), []).append(str(isic_code))
            elif to_codes is not None:
                reverse.setdefault(str(to_codes), []).append(str(isic_code))
    return reverse


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def lookup_industry_code(
    system: str, code: str
) -> MappingResult:
    """Exact code lookup in a classification system.

    Raises ``MappingError`` only for invalid ``system`` names. A missing
    ``code`` returns a low-confidence result with ``canonical=None``.
    """
    sys_enum = IndustryCodeSystem.from_str(system)
    codes = load_codes(sys_enum)
    needle = str(code).strip().upper().replace(".", "").replace("-", "")
    if needle in codes:
        record = codes[needle]
        return MappingResult(
            canonical=record.to_dict(),
            confidence=1.0,
            band=MappingConfidence.EXACT,
            rationale="Exact %s lookup: %s -> %s"
            % (sys_enum.value, needle, record.label),
            matched_pattern=needle,
            raw_input="%s:%s" % (system, code),
        )
    # ISIC / NACE-style lookup with section-letter prefix: e.g. 'C2410' ->
    # strip the single leading letter and retry.
    if (
        sys_enum in (IndustryCodeSystem.ISIC, IndustryCodeSystem.NACE)
        and len(needle) >= 2
        and needle[0].isalpha()
        and needle[1:].isdigit()
        and needle[1:] in codes
    ):
        record = codes[needle[1:]]
        return MappingResult(
            canonical=record.to_dict(),
            confidence=1.0,
            band=MappingConfidence.EXACT,
            rationale="Exact %s lookup: %s -> %s (section-prefix stripped)"
            % (sys_enum.value, needle, record.label),
            matched_pattern=needle[1:],
            raw_input="%s:%s" % (system, code),
        )
    return MappingResult(
        canonical=None,
        confidence=0.0,
        band=MappingConfidence.UNKNOWN,
        rationale="%s code %r not found" % (sys_enum.value, code),
        raw_input="%s:%s" % (system, code),
    )


def lookup_industry_label(system: str, label: str) -> MappingResult:
    """Case-insensitive label lookup within a classification system."""
    sys_enum = IndustryCodeSystem.from_str(system)
    codes = load_codes(sys_enum)
    needle = normalize_text(label)
    if not needle:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="Empty label",
            raw_input=label,
        )
    best: Optional[IndustryCode] = None
    best_score = 0.0
    exact: Optional[IndustryCode] = None
    for record in codes.values():
        candidate = normalize_text(record.label)
        if candidate == needle:
            exact = record
            break
        cand_tokens = set(candidate.split())
        need_tokens = set(needle.split())
        if not cand_tokens or not need_tokens:
            continue
        overlap = cand_tokens & need_tokens
        if not overlap:
            continue
        score = len(overlap) / max(len(cand_tokens), len(need_tokens))
        if score > best_score:
            best_score = score
            best = record
    if exact is not None:
        return MappingResult(
            canonical=exact.to_dict(),
            confidence=1.0,
            band=MappingConfidence.EXACT,
            rationale="Exact label match: %s" % exact.label,
            matched_pattern=exact.label,
            raw_input=label,
        )
    if best is not None and best_score >= 0.3:
        return MappingResult(
            canonical=best.to_dict(),
            confidence=best_score,
            band=MappingConfidence.from_score(best_score),
            rationale="Token-overlap label match %.2f: %s" % (best_score, best.label),
            matched_pattern=best.label,
            raw_input=label,
        )
    return MappingResult(
        canonical=None,
        confidence=0.0,
        band=MappingConfidence.UNKNOWN,
        rationale="No label match for %r in %s" % (label, sys_enum.value),
        raw_input=label,
    )


def children_of(system: str, parent_code: str) -> List[IndustryCode]:
    """Return all direct children of ``parent_code`` within a system."""
    sys_enum = IndustryCodeSystem.from_str(system)
    codes = load_codes(sys_enum)
    parent = str(parent_code).strip().upper()
    return [r for r in codes.values() if r.parent_code == parent]


def parent_of(system: str, code: str) -> Optional[IndustryCode]:
    """Return the direct parent of ``code`` if present."""
    sys_enum = IndustryCodeSystem.from_str(system)
    codes = load_codes(sys_enum)
    record = codes.get(str(code).strip().upper())
    if record is None or record.parent_code is None:
        return None
    return codes.get(record.parent_code)


def get_sector_default_ef(system: str, code: str) -> Optional[float]:
    """Return the spend-based default emission factor for a sector code."""
    result = lookup_industry_code(system, code)
    if result.canonical is None:
        return None
    return result.canonical.get("ef_default")


# ---------------------------------------------------------------------------
# Crosswalk API
# ---------------------------------------------------------------------------


def crosswalk_code(
    from_system: str, code: str, to_system: str
) -> MappingResult:
    """Translate a code from one system to another.

    Supported direct pairs: ISIC<->NACE, ISIC<->NAICS (both directions).
    Other pairs (e.g. NACE->NAICS) are routed via ISIC. ANZSIC / JSIC do
    not yet have crosswalk data so those calls return UNKNOWN.
    """
    src = IndustryCodeSystem.from_str(from_system)
    dst = IndustryCodeSystem.from_str(to_system)
    if src == dst:
        return lookup_industry_code(from_system, code)

    needle = str(code).strip().upper().replace(".", "").replace("-", "")
    # Strip leading ISIC/NACE section letter if present (e.g. 'C2410' -> '2410').
    if (
        src in (IndustryCodeSystem.ISIC, IndustryCodeSystem.NACE)
        and len(needle) >= 2
        and needle[0].isalpha()
        and needle[1:].isdigit()
    ):
        needle = needle[1:]

    # Direct ISIC <-> NACE / NAICS
    if src == IndustryCodeSystem.ISIC and dst in (
        IndustryCodeSystem.NACE, IndustryCodeSystem.NAICS,
    ):
        payload = _load_raw_crosswalk(src, dst)
        for bucket in ("classes", "divisions"):
            entry = (payload.get(bucket) or {}).get(needle)
            if entry is not None:
                return _build_crosswalk_result(src, needle, dst, entry)
        return _not_found(from_system, code, to_system)

    if dst == IndustryCodeSystem.ISIC and src in (
        IndustryCodeSystem.NACE, IndustryCodeSystem.NAICS,
    ):
        payload = _load_raw_crosswalk(dst, src)
        inverse = _invert_crosswalk(payload)
        isic_candidates = inverse.get(needle)
        if isic_candidates:
            to_code = isic_candidates[0] if len(isic_candidates) == 1 else isic_candidates
            return _build_crosswalk_result(
                src, needle, dst, {"to": to_code, "match_quality": "inverse"}
            )
        return _not_found(from_system, code, to_system)

    # NACE -> NAICS or NAICS -> NACE: route via ISIC
    if {src, dst} == {IndustryCodeSystem.NACE, IndustryCodeSystem.NAICS}:
        # src -> ISIC
        intermediate = crosswalk_code(src.value, code, IndustryCodeSystem.ISIC.value)
        if intermediate.canonical is None:
            return _not_found(from_system, code, to_system)
        isic_target = intermediate.canonical["to_code"]
        if isinstance(isic_target, list):
            isic_target = isic_target[0]
        # ISIC -> dst
        final = crosswalk_code(
            IndustryCodeSystem.ISIC.value, isic_target, dst.value
        )
        if final.canonical is None:
            return _not_found(from_system, code, to_system)
        final.rationale = (
            "Routed %s:%s -> ISIC:%s -> %s:%s"
            % (src.value, needle, isic_target, dst.value, final.canonical["to_code"])
        )
        final.confidence = min(final.confidence, 0.80)
        final.band = MappingConfidence.from_score(final.confidence)
        return final

    # ANZSIC / JSIC crosswalks are not provided yet.
    return MappingResult(
        canonical=None,
        confidence=0.0,
        band=MappingConfidence.UNKNOWN,
        rationale="No crosswalk data for %s -> %s"
        % (src.value, dst.value),
        raw_input="%s:%s" % (from_system, code),
    )


def _build_crosswalk_result(
    src: IndustryCodeSystem,
    from_code: str,
    dst: IndustryCodeSystem,
    entry: Dict[str, Any],
) -> MappingResult:
    to_code = entry.get("to")
    match_quality = str(entry.get("match_quality", "primary"))
    quality_confidence = {
        "identical": 1.0,
        "one_to_one": 0.98,
        "primary": 0.85,
        "split": 0.75,
        "merge": 0.75,
        "partial": 0.65,
        "inverse": 0.78,
    }.get(match_quality, 0.70)
    return MappingResult(
        canonical={
            "from_system": src.value,
            "from_code": from_code,
            "to_system": dst.value,
            "to_code": to_code,
            "match_quality": match_quality,
        },
        confidence=quality_confidence,
        band=MappingConfidence.from_score(quality_confidence),
        rationale="%s:%s -> %s:%s (%s)"
        % (src.value, from_code, dst.value, to_code, match_quality),
        matched_pattern=from_code,
        raw_input="%s:%s" % (src.value, from_code),
    )


def _not_found(from_system: str, code: str, to_system: str) -> MappingResult:
    return MappingResult(
        canonical=None,
        confidence=0.0,
        band=MappingConfidence.UNKNOWN,
        rationale="No crosswalk entry for %s:%s -> %s"
        % (from_system, code, to_system),
        raw_input="%s:%s" % (from_system, code),
    )


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def list_systems() -> List[str]:
    return [s.value for s in IndustryCodeSystem]


def count_codes(system: str) -> int:
    sys_enum = IndustryCodeSystem.from_str(system)
    return len(load_codes(sys_enum))


__all__ = [
    "CodeCrosswalk",
    "CodeLevel",
    "IndustryCode",
    "IndustryCodeSystem",
    "children_of",
    "count_codes",
    "crosswalk_code",
    "get_sector_default_ef",
    "list_systems",
    "load_codes",
    "lookup_industry_code",
    "lookup_industry_label",
    "parent_of",
]
