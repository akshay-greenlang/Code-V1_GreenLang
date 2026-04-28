# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.0 — IEA Emissions Factors mini fixture builder.

Builds a deterministic .xlsx workbook for the ``iea_emission_factors``
source — country-level grid intensities.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

_FROZEN_STAMP = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)

#: IEA workbook headers (country-grid shape).
IEA_HEADERS: Tuple[str, ...] = (
    "country",
    "unit",
    "co2_factor",
    "vintage",
    "methodology",
    "notes",
)

COUNTRY_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("FRA", "kgco2e/kwh", 0.056, "2024.1", "iea_t1", "France grid (low-carbon nuclear)"),
    ("DEU", "kgco2e/kwh", 0.349, "2024.1", "iea_t1", "Germany grid"),
    ("CHN", "kgco2e/kwh", 0.586, "2024.1", "iea_t1", "China grid"),
    ("USA", "kgco2e/kwh", 0.367, "2024.1", "iea_t1", "United States grid"),
    ("BRA", "kgco2e/kwh", 0.119, "2024.1", "iea_t1", "Brazil grid (high hydro share)"),
)


def build_workbook_bytes() -> bytes:
    """Return deterministic .xlsx bytes for the IEA mini fixture."""
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.Workbook()
    wb.properties.creator = "GreenLang Factors / Phase 3 fixture"
    wb.properties.created = _FROZEN_STAMP
    wb.properties.modified = _FROZEN_STAMP
    wb.properties.lastModifiedBy = "phase3-fixture"

    default = wb.active
    default.title = "Country Grid Factors"
    default.append(list(IEA_HEADERS))
    for row in COUNTRY_ROWS:
        default.append(list(row))

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
    "IEA_HEADERS",
    "COUNTRY_ROWS",
    "build_workbook_bytes",
    "ensure_fixture",
]


if __name__ == "__main__":  # pragma: no cover
    fixture_path = Path(__file__).resolve().parent / "iea_2024_mini.xlsx"
    ensure_fixture(fixture_path)
    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes)")
