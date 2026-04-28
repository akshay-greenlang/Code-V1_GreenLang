# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — synthetic EXIOBASE v3.8.2 mini fixture builder.

Builds a deterministic mini EXIOBASE zip with:

  * ``meta/sectors.csv`` — 5 sectors (energy, transport, agriculture,
    manufacturing, services).
  * ``meta/regions.csv`` — 3 regions (US, DE, WA — one country + one
    EU country + one ROW).
  * ``meta/extensions.csv`` — 2 environmental extensions (CO2, CH4).
  * ``F.csv`` — 5 sectors × 3 regions × 2 extensions = 30 rows.

Every row is reproducible across hosts because:

  * Member ordering is sorted alphabetically.
  * ``ZipInfo.date_time`` pinned to ``(2026, 4, 28, 0, 0, 0)``.
  * CSV values are deterministic functions of the (sector, region,
    extension) tuple.
"""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

__all__ = [
    "EXIOBASE_SECTORS",
    "EXIOBASE_REGIONS",
    "EXIOBASE_EXTENSIONS",
    "EXIOBASE_FIXTURE_FILENAME",
    "build_zip_bytes",
    "ensure_fixture",
]


_DETERMINISTIC_MTIME: Tuple[int, int, int, int, int, int] = (2026, 4, 28, 0, 0, 0)

EXIOBASE_SECTORS: List[Dict[str, str]] = [
    {"code": "ENERGY", "name": "Electricity, gas, steam"},
    {"code": "TRANSPORT", "name": "Transport services"},
    {"code": "AGRICULTURE", "name": "Agriculture and forestry"},
    {"code": "MANUFACTURING", "name": "Manufacturing"},
    {"code": "SERVICES", "name": "Services"},
]

EXIOBASE_REGIONS: List[Dict[str, str]] = [
    {"code": "US", "name": "United States"},
    {"code": "DE", "name": "Germany"},
    {"code": "WA", "name": "Rest of World — Asia Pacific"},
]

EXIOBASE_EXTENSIONS: List[Dict[str, str]] = [
    {"code": "CO2", "name": "Carbon dioxide", "unit": "kg/EUR"},
    {"code": "CH4", "name": "Methane", "unit": "kg/EUR"},
]

EXIOBASE_FIXTURE_FILENAME: str = "exiobase_v3.8.2_mini.zip"


def _f_value(sector: str, region: str, extension: str) -> float:
    """Deterministic factor value: hash-derived but stable."""
    h = hashlib.sha256(
        ("%s|%s|%s" % (sector, region, extension)).encode("utf-8"),
    ).hexdigest()
    # Convert the first 6 hex chars to an int and scale into ~[0, 5).
    return round((int(h[:6], 16) % 5000) / 1000.0, 3)


def _csv_bytes(headers: List[str], rows: List[List[str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")


def _members() -> List[Tuple[str, bytes]]:
    sectors_csv = _csv_bytes(
        ["code", "name"],
        [[s["code"], s["name"]] for s in EXIOBASE_SECTORS],
    )
    regions_csv = _csv_bytes(
        ["code", "name"],
        [[r["code"], r["name"]] for r in EXIOBASE_REGIONS],
    )
    ext_csv = _csv_bytes(
        ["code", "name", "unit"],
        [[e["code"], e["name"], e["unit"]] for e in EXIOBASE_EXTENSIONS],
    )

    f_rows: List[List[str]] = []
    for sector in EXIOBASE_SECTORS:
        for region in EXIOBASE_REGIONS:
            for ext in EXIOBASE_EXTENSIONS:
                value = _f_value(sector["code"], region["code"], ext["code"])
                f_rows.append([sector["code"], region["code"], ext["code"], "%.3f" % value])
    f_csv = _csv_bytes(["sector", "region", "extension", "value"], f_rows)

    return [
        ("meta/extensions.csv", ext_csv),
        ("meta/regions.csv", regions_csv),
        ("meta/sectors.csv", sectors_csv),
        ("F.csv", f_csv),
    ]


def build_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member_path, payload in sorted(_members(), key=lambda m: m[0]):
            info = zipfile.ZipInfo(filename=member_path, date_time=_DETERMINISTIC_MTIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)
    return buf.getvalue()


def ensure_fixture(target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = build_zip_bytes()
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha:
            return target
    target.write_bytes(expected_bytes)
    return target


if __name__ == "__main__":  # pragma: no cover
    here = Path(__file__).resolve().parent
    out = ensure_fixture(here / EXIOBASE_FIXTURE_FILENAME)
    print("wrote", out, "sha=", hashlib.sha256(out.read_bytes()).hexdigest())
