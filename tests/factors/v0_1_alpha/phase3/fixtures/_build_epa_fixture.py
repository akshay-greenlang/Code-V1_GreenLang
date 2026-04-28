# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.0 — EPA GHG Emission Factors Hub mini fixture builder.

Builds a deterministic, byte-stable .xlsx workbook for the
``epa_hub`` source. Mirrors :mod:`_build_defra_fixture` (frozen
``wb.properties.created/modified``, single ``BytesIO`` save).

Tabs
----
- ``Stationary Combustion`` — primary EPA Hub section (5 rows)
- ``Mobile Combustion`` — secondary EPA Hub section (3 rows)

Headers match the EPA-shaped column layout the
:class:`Phase3EPAExcelParser` validates against.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

#: Pinned creation/modified stamp so the embedded core.xml is stable.
_FROZEN_STAMP = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)

#: Required header columns the synthetic EPA workbook ships on every tab.
EPA_HEADERS: Tuple[str, ...] = (
    "fuel_type",
    "unit",
    "co2_factor",
    "ch4_factor",
    "n2o_factor",
    "notes",
)

STATIONARY_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("natural_gas", "mmbtu", 53.06, 0.001, 0.0001, "EPA Hub 2024 default"),
    ("distillate_fuel_oil_no_2", "mmbtu", 73.96, 0.003, 0.0006, "EPA Hub 2024 default"),
    ("propane", "mmbtu", 62.87, 0.003, 0.0006, "EPA Hub 2024 default"),
    ("bituminous_coal", "mmbtu", 93.40, 0.011, 0.0016, "EPA Hub 2024 default"),
    ("wood_waste", "mmbtu", 93.80, 0.032, 0.0042, "Biogenic component reported separately"),
)

MOBILE_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("motor_gasoline", "gallons", 8.78, 0.00038, 0.00022, "On-road passenger vehicles, EPA 2024"),
    ("diesel_fuel", "gallons", 10.21, 0.00029, 0.00026, "On-road heavy-duty diesel, EPA 2024"),
    ("jet_fuel", "gallons", 9.75, 0.00027, 0.00031, "Commercial aviation, EPA 2024"),
)


def build_workbook_bytes() -> bytes:
    """Return deterministic .xlsx bytes for the EPA mini fixture (no I/O)."""
    import openpyxl  # noqa: PLC0415 — heavyweight import deferred.

    wb = openpyxl.Workbook()
    wb.properties.creator = "GreenLang Factors / Phase 3 fixture"
    wb.properties.created = _FROZEN_STAMP
    wb.properties.modified = _FROZEN_STAMP
    wb.properties.lastModifiedBy = "phase3-fixture"

    default = wb.active
    default.title = "Stationary Combustion"
    default.append(list(EPA_HEADERS))
    for row in STATIONARY_ROWS:
        default.append(list(row))

    sheet2 = wb.create_sheet("Mobile Combustion")
    sheet2.append(list(EPA_HEADERS))
    for row in MOBILE_ROWS:
        sheet2.append(list(row))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def ensure_fixture(path: Path) -> Path:
    """Materialise the EPA fixture at ``path`` if missing. Idempotent."""
    p = Path(path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_workbook_bytes())
    return p


__all__ = [
    "EPA_HEADERS",
    "STATIONARY_ROWS",
    "MOBILE_ROWS",
    "build_workbook_bytes",
    "ensure_fixture",
]


if __name__ == "__main__":  # pragma: no cover — manual regen helper
    fixture_path = Path(__file__).resolve().parent / "epa_2024_mini.xlsx"
    ensure_fixture(fixture_path)
    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes)")
