# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — synthetic ecoinvent 3.10 mini fixture builder.

Builds a deterministic mini ecoinvent zip containing 3 ``.spold`` XML
files — one per system model (cutoff, apos, consequential), each with
exactly one activity + one reference-product exchange. The output is
*byte-stable* across runs and platforms because:

  * ``ZipInfo.date_time`` is pinned to ``(2026, 4, 28, 0, 0, 0)``.
  * ``ZIP_DEFLATED`` is used with ``ZipFile.writestr`` (which records
    the modification time, file mode, and CRC deterministically given
    fixed input bytes).
  * Member iteration order in :meth:`build_zip` is ``sorted(...)``.

The synthetic XML is intentionally minimal: it carries the EcoSpold2
namespace + the elements the parser actually inspects
(:class:`Phase3EcoSpoldParser` reads ``activityDescription/activity``,
``activityDescription/geography/shortname``,
``activityDescription/timePeriod[@startDate]``, and the first
``flowData/intermediateExchange[outputGroup=0]``). Anything else is
omitted to keep the fixture small.

Determinism: re-running this builder on the same machine produces a
zip with the same SHA-256.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

__all__ = [
    "ECOINVENT_SYSTEM_MODELS",
    "ECOINVENT_FIXTURE_FILENAME",
    "build_zip_bytes",
    "ensure_fixture",
]


#: Pinned mtime stamp for every ZipInfo entry. Wave 2.5 directive:
#: deterministic zip fixtures, ``ZipInfo.date_time = (2026,4,28,0,0,0)``.
_DETERMINISTIC_MTIME: Tuple[int, int, int, int, int, int] = (2026, 4, 28, 0, 0, 0)

#: Ordered tuple of (system_model, activity_id, activity_name, value,
#: product_name, geography_short, vintage_start, vintage_end).
ECOINVENT_SYSTEM_MODELS: List[Dict[str, str]] = [
    {
        "system_model": "apos",
        "activity_id": "11111111-1111-1111-1111-111111111111",
        "activity_name": "electricity production, hard coal",
        "value": "0.95",
        "product_name": "electricity, high voltage",
        "geography_short": "GLO",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
    },
    {
        "system_model": "consequential",
        "activity_id": "22222222-2222-2222-2222-222222222222",
        "activity_name": "diesel, burned in industrial machinery",
        "value": "3.21",
        "product_name": "diesel, burned in industrial machinery",
        "geography_short": "RoW",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
    },
    {
        "system_model": "cutoff",
        "activity_id": "33333333-3333-3333-3333-333333333333",
        "activity_name": "transport, freight, lorry 16-32 metric ton, EURO6",
        "value": "0.18",
        "product_name": "transport, freight, lorry",
        "geography_short": "GLO",
        "vintage_start": "2024-01-01",
        "vintage_end": "2024-12-31",
    },
]

#: Filename of the deterministic mini fixture.
ECOINVENT_FIXTURE_FILENAME: str = "ecoinvent_3.10_mini.zip"


# ---------------------------------------------------------------------------
# XML synthesis
# ---------------------------------------------------------------------------


_SPOLD_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<ecoSpold xmlns="http://www.EcoInvent.org/EcoSpold02">
  <activityDataset>
    <activityDescription>
      <activity id="{activity_id}">
        <activityName>{activity_name}</activityName>
      </activity>
      <geography>
        <shortname>{geography_short}</shortname>
      </geography>
      <timePeriod startDate="{vintage_start}" endDate="{vintage_end}"/>
    </activityDescription>
    <flowData>
      <intermediateExchange id="{exchange_id}" amount="{value}">
        <name>{product_name}</name>
        <outputGroup>0</outputGroup>
      </intermediateExchange>
    </flowData>
  </activityDataset>
</ecoSpold>
"""


def _spold_xml_for(model: Dict[str, str]) -> str:
    return _SPOLD_TEMPLATE.format(
        activity_id=model["activity_id"],
        activity_name=model["activity_name"],
        geography_short=model["geography_short"],
        vintage_start=model["vintage_start"],
        vintage_end=model["vintage_end"],
        exchange_id="exchange-" + model["activity_id"][:8],
        value=model["value"],
        product_name=model["product_name"],
    )


# ---------------------------------------------------------------------------
# Deterministic zip writer
# ---------------------------------------------------------------------------


def build_zip_bytes() -> bytes:
    """Return deterministic zip bytes for the mini ecoinvent fixture."""
    buf = io.BytesIO()
    # ``ZipFile.writestr`` honours a passed ZipInfo and ignores wall-clock
    # time when the ZipInfo carries an explicit ``date_time`` tuple.
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Pin determinism by sorting + explicit ZipInfo per member.
        for model in sorted(ECOINVENT_SYSTEM_MODELS, key=lambda m: m["system_model"]):
            sys_model = model["system_model"]
            member_path = "%s/dataset_%s.spold" % (sys_model, model["activity_id"][:8])
            xml = _spold_xml_for(model)
            info = zipfile.ZipInfo(filename=member_path, date_time=_DETERMINISTIC_MTIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16  # Pin file permissions.
            zf.writestr(info, xml.encode("utf-8"))
    return buf.getvalue()


def ensure_fixture(target: Path) -> Path:
    """Materialise the mini ecoinvent fixture at ``target`` if absent.

    Idempotent: if the file already exists with the correct SHA-256 the
    function is a no-op.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = build_zip_bytes()
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()
    if target.exists():
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha == expected_sha:
            return target
    target.write_bytes(expected_bytes)
    return target


if __name__ == "__main__":  # pragma: no cover
    here = Path(__file__).resolve().parent
    out = ensure_fixture(here / ECOINVENT_FIXTURE_FILENAME)
    print("wrote", out, "sha=", hashlib.sha256(out.read_bytes()).hexdigest())
