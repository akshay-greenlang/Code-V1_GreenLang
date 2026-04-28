# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.0 — India BEE PAT mini fixture builder.

Builds a deterministic .xlsx workbook for the ``india_bee_pat`` source.
The shape is sectoral specific energy consumption (SEC) baselines.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

_FROZEN_STAMP = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)

#: India BEE PAT workbook headers (sectoral baseline shape).
BEE_HEADERS: Tuple[str, ...] = (
    "sector",
    "unit",
    "sec_baseline",
    "pat_cycle",
    "vintage",
    "notes",
)

SECTOR_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("cement", "kgco2e/tonne", 0.795, "PAT-II", "2024.1", "Sectoral SEC baseline — cement"),
    ("iron_and_steel", "kgco2e/tonne", 1.910, "PAT-II", "2024.1", "Sectoral SEC baseline — integrated steel"),
    ("aluminium", "kgco2e/tonne", 14.250, "PAT-II", "2024.1", "Sectoral SEC baseline — primary aluminium"),
    ("fertilizer", "kgco2e/tonne", 0.622, "PAT-II", "2024.1", "Sectoral SEC baseline — urea"),
    ("pulp_and_paper", "kgco2e/tonne", 1.035, "PAT-II", "2024.1", "Sectoral SEC baseline — pulp & paper"),
)


def build_workbook_bytes() -> bytes:
    """Return deterministic .xlsx bytes for the BEE mini fixture."""
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.Workbook()
    wb.properties.creator = "GreenLang Factors / Phase 3 fixture"
    wb.properties.created = _FROZEN_STAMP
    wb.properties.modified = _FROZEN_STAMP
    wb.properties.lastModifiedBy = "phase3-fixture"

    default = wb.active
    default.title = "PAT Sectoral Baselines"
    default.append(list(BEE_HEADERS))
    for row in SECTOR_ROWS:
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
    "BEE_HEADERS",
    "SECTOR_ROWS",
    "build_workbook_bytes",
    "ensure_fixture",
]


if __name__ == "__main__":  # pragma: no cover
    fixture_path = Path(__file__).resolve().parent / "bee_2024_mini.xlsx"
    ensure_fixture(fixture_path)
    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes)")
