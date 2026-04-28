# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.0 — India CEA CO2 Baseline mini fixture builder."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

_FROZEN_STAMP = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)

#: India CEA workbook headers (regional grid shape).
CEA_HEADERS: Tuple[str, ...] = (
    "grid",
    "unit",
    "co2_factor",
    "financial_year",
    "publication_version",
    "notes",
)

GRID_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("All_India", "kgco2e/kwh", 0.727, "2023-24", "v20.0", "All-India composite grid"),
    ("NEWNE", "kgco2e/kwh", 0.764, "2023-24", "v20.0", "Northern + Eastern + Western + NER"),
    ("S", "kgco2e/kwh", 0.583, "2023-24", "v20.0", "Southern regional grid"),
    ("NER", "kgco2e/kwh", 0.291, "2023-24", "v20.0", "North-Eastern regional grid (standalone)"),
)


def build_workbook_bytes() -> bytes:
    """Return deterministic .xlsx bytes for the CEA mini fixture."""
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.Workbook()
    wb.properties.creator = "GreenLang Factors / Phase 3 fixture"
    wb.properties.created = _FROZEN_STAMP
    wb.properties.modified = _FROZEN_STAMP
    wb.properties.lastModifiedBy = "phase3-fixture"

    default = wb.active
    default.title = "Grid Emission Factors"
    default.append(list(CEA_HEADERS))
    for row in GRID_ROWS:
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
    "CEA_HEADERS",
    "GRID_ROWS",
    "build_workbook_bytes",
    "ensure_fixture",
]


if __name__ == "__main__":  # pragma: no cover
    fixture_path = Path(__file__).resolve().parent / "cea_2024_mini.xlsx"
    ensure_fixture(fixture_path)
    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes)")
