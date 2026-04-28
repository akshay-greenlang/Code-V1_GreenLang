# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.0 — eGRID mini fixture builder.

Builds a deterministic .xlsx workbook for the ``egrid`` source.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

_FROZEN_STAMP = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)

#: eGRID workbook headers (subregion-shape).
EGRID_HEADERS: Tuple[str, ...] = (
    "subregion",
    "unit",
    "co2_factor",
    "ch4_factor",
    "n2o_factor",
    "notes",
)

SUBREGION_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("AKGD", "kgco2e/kwh", 0.47624, 0.00004, 0.00001, "ASCC Alaska Grid"),
    ("CAMX", "kgco2e/kwh", 0.20598, 0.00002, 0.00001, "WECC California"),
    ("ERCT", "kgco2e/kwh", 0.36791, 0.00003, 0.00001, "ERCOT All"),
    ("RFCM", "kgco2e/kwh", 0.43882, 0.00005, 0.00001, "RFC Michigan"),
    ("MROW", "kgco2e/kwh", 0.40117, 0.00004, 0.00001, "MRO West"),
)

STATE_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("CA", "kgco2e/kwh", 0.20109, 0.00002, 0.00001, "California state-average"),
    ("TX", "kgco2e/kwh", 0.36544, 0.00003, 0.00001, "Texas state-average"),
    ("WA", "kgco2e/kwh", 0.07932, 0.00001, 0.00001, "Washington state-average"),
)


def build_workbook_bytes() -> bytes:
    """Return deterministic .xlsx bytes for the eGRID mini fixture."""
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.Workbook()
    wb.properties.creator = "GreenLang Factors / Phase 3 fixture"
    wb.properties.created = _FROZEN_STAMP
    wb.properties.modified = _FROZEN_STAMP
    wb.properties.lastModifiedBy = "phase3-fixture"

    default = wb.active
    default.title = "Subregion Factors"
    default.append(list(EGRID_HEADERS))
    for row in SUBREGION_ROWS:
        default.append(list(row))

    sheet2 = wb.create_sheet("State Factors")
    sheet2.append(list(EGRID_HEADERS))
    for row in STATE_ROWS:
        sheet2.append(list(row))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def ensure_fixture(path: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_workbook_bytes())
    return p


__all__ = [
    "EGRID_HEADERS",
    "SUBREGION_ROWS",
    "STATE_ROWS",
    "build_workbook_bytes",
    "ensure_fixture",
]


if __name__ == "__main__":  # pragma: no cover
    fixture_path = Path(__file__).resolve().parent / "egrid_2024_mini.xlsx"
    ensure_fixture(fixture_path)
    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes)")
