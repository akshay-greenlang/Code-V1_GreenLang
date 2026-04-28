# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.0 — CSV/JSON/XML family parsers end-to-end test.

Parametrized over the three CSV/JSON/XML sources shipped in Wave 2.0
(EDGAR, ENTSO-E, Climate TRACE). Each parametrisation exercises the
same acceptance assertions Wave 1.5 ran against DEFRA Excel:

  1. Direct parser invocation: ``parse_bytes(ctx, raw)`` returns one
     v0.1 factor record per fixture row, each carrying every required
     provenance pin (``raw_artifact_uri``, ``raw_artifact_sha256``,
     ``parser_id``, ``parser_version``, ``row_ref``, ``citations``,
     ``licence``).
  2. Snapshot match: the parsed output is byte-stable against the
     committed golden file under
     ``tests/factors/v0_1_alpha/phase3/parser_snapshots/``.
  3. Registry dispatch: ``build_phase3_registry().get(source_id)`` finds
     the parser and the parser_version matches the source_registry pin.
  4. Negative paths: missing required field, malformed timestamp,
     out-of-range value, unknown geography code each raise
     :class:`ParserDispatchError` with the offending row/line.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Wave 2.0 Parser migrations"
- ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 4 ("CSV/JSON/XML
  family validation"), Block 6 ("snapshot tests").
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.factors.ingestion.exceptions import ParserDispatchError
from greenlang.factors.ingestion.parser_harness import ParserContext
from greenlang.factors.ingestion.parsers._phase3_adapters import (
    PHASE3_CLIMATE_TRACE_PARSER_VERSION,
    PHASE3_CLIMATE_TRACE_SOURCE_ID,
    PHASE3_EDGAR_PARSER_VERSION,
    PHASE3_EDGAR_SOURCE_ID,
    PHASE3_ENTSOE_PARSER_VERSION,
    PHASE3_ENTSOE_SOURCE_ID,
    Phase3ClimateTRACECsvParser,
    Phase3EDGARCsvParser,
    Phase3ENTSOEXmlParser,
    build_phase3_registry,
)
from tests.factors.v0_1_alpha.phase3.parser_snapshots._helper import (
    compare_to_snapshot,
    regenerate_if_env,
)


_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Parametrisation: (source_id, parser_cls, parser_version, fixture_filename,
# snapshot_id, expected_row_count, content_type)
# ---------------------------------------------------------------------------


_FAMILY_PARAMS: List[Tuple[str, Any, str, str, str, int, str]] = [
    (
        PHASE3_EDGAR_SOURCE_ID,
        Phase3EDGARCsvParser,
        PHASE3_EDGAR_PARSER_VERSION,
        "edgar_v8_mini.csv",
        "edgar_v8",
        10,
        "text/csv",
    ),
    (
        PHASE3_ENTSOE_SOURCE_ID,
        Phase3ENTSOEXmlParser,
        PHASE3_ENTSOE_PARSER_VERSION,
        "entsoe_2024_mini.xml",
        "entsoe_2024",
        10,
        "application/xml",
    ),
    (
        PHASE3_CLIMATE_TRACE_SOURCE_ID,
        Phase3ClimateTRACECsvParser,
        PHASE3_CLIMATE_TRACE_PARSER_VERSION,
        "climate_trace_2024_mini.csv",
        "climate_trace_2024",
        10,
        "text/csv",
    ),
]


_PARAM_IDS: List[str] = [p[4] for p in _FAMILY_PARAMS]


def _make_ctx(source_id: str, content_type: str, filename: str) -> ParserContext:
    """Build a :class:`ParserContext` carrying the family-detection hints."""
    ctx = ParserContext(
        artifact_id=filename,
        source_id=source_id,
        parser_id="phase3-test",
    )
    ctx.fetch_metadata = {  # type: ignore[attr-defined]
        "content_type": content_type,
        "filename": filename,
    }
    return ctx


def _strip_volatile(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop wall-clock fields the snapshot framework is allowed to ignore.

    Mirrors the DEFRA snapshot test: the parser emits ``ingested_at``,
    ``published_at``, ``reviewed_at``, and ``approved_at`` at parse time;
    the goldens never encode wall-clock values.
    """
    for row in rows:
        row.pop("published_at", None)
        ext = row.get("extraction") or {}
        if isinstance(ext, dict):
            ext.pop("ingested_at", None)
        review = row.get("review") or {}
        if isinstance(review, dict):
            for vol_key in ("reviewed_at", "approved_at"):
                review.pop(vol_key, None)
    return rows


# ---------------------------------------------------------------------------
# 1. Happy-path parse: each source emits the expected row count + every
#    required provenance pin.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_id", "parser_cls", "parser_version", "fixture_filename",
     "snapshot_id", "expected_row_count", "content_type"),
    _FAMILY_PARAMS,
    ids=_PARAM_IDS,
)
def test_csv_json_xml_parse_bytes_emits_full_provenance(
    source_id: str,
    parser_cls: Any,
    parser_version: str,
    fixture_filename: str,
    snapshot_id: str,
    expected_row_count: int,
    content_type: str,
) -> None:
    """``parse_bytes`` returns 10 rows with full provenance pins.

    Every row must carry ``urn``, ``source_urn``, ``factor_pack_urn``,
    ``unit_urn``, ``geography_urn``, ``methodology_urn``, ``licence``,
    ``citations`` (non-empty list), and a fully-populated ``extraction``
    block. Drift on any of these is a Phase 2 gate-6 (provenance
    completeness) failure.
    """
    fixture_path = _FIXTURES_DIR / fixture_filename
    raw = fixture_path.read_bytes()
    parser = parser_cls()
    ctx = _make_ctx(source_id, content_type, fixture_filename)
    rows = parser.parse_bytes(
        ctx, raw,
        artifact_uri=f"file://{fixture_filename}",
        artifact_sha256="0" * 64,
    )
    assert len(rows) == expected_row_count, (
        f"{snapshot_id}: expected {expected_row_count} rows, got {len(rows)}"
    )
    required_top_keys = {
        "urn", "source_urn", "factor_pack_urn", "unit_urn",
        "geography_urn", "methodology_urn", "licence", "citations",
        "extraction", "review", "value", "vintage_start", "vintage_end",
    }
    required_extraction_keys = {
        "raw_artifact_uri", "raw_artifact_sha256", "parser_id",
        "parser_version", "row_ref",
    }
    for idx, row in enumerate(rows):
        missing_top = required_top_keys - row.keys()
        assert not missing_top, (
            f"{snapshot_id} row {idx}: missing top-level keys {missing_top}"
        )
        assert isinstance(row["citations"], list) and row["citations"], (
            f"{snapshot_id} row {idx}: citations must be a non-empty list"
        )
        ext = row["extraction"]
        assert isinstance(ext, dict)
        missing_ext = required_extraction_keys - ext.keys()
        assert not missing_ext, (
            f"{snapshot_id} row {idx}: missing extraction keys {missing_ext}"
        )
        assert ext["parser_version"] == parser_version


# ---------------------------------------------------------------------------
# 2. Snapshot match — byte-stable golden comparison.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_id", "parser_cls", "parser_version", "fixture_filename",
     "snapshot_id", "expected_row_count", "content_type"),
    _FAMILY_PARAMS,
    ids=_PARAM_IDS,
)
def test_csv_json_xml_parser_snapshot_matches_golden(
    source_id: str,
    parser_cls: Any,
    parser_version: str,
    fixture_filename: str,
    snapshot_id: str,
    expected_row_count: int,
    content_type: str,
) -> None:
    """Each source's parsed output matches its committed golden snapshot.

    Set ``UPDATE_PARSER_SNAPSHOT=1`` to regenerate the goldens. The
    snapshot helper drops volatile fields (timestamps, hashes when
    explicitly frozen) before comparison.
    """
    fixture_path = _FIXTURES_DIR / fixture_filename
    raw = fixture_path.read_bytes()
    parser = parser_cls()
    ctx = _make_ctx(source_id, content_type, fixture_filename)
    rows = parser.parse_bytes(
        ctx, raw,
        artifact_uri=f"file://{fixture_filename}",
        artifact_sha256="0" * 64,
    )
    rows = _strip_volatile(rows)
    regenerate_if_env(snapshot_id, parser_version, rows)
    compare_to_snapshot(snapshot_id, parser_version, rows)


# ---------------------------------------------------------------------------
# 3. Registry dispatch: build_phase3_registry surfaces all three sources.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_id", "parser_cls", "parser_version", "fixture_filename",
     "snapshot_id", "expected_row_count", "content_type"),
    _FAMILY_PARAMS,
    ids=_PARAM_IDS,
)
def test_csv_json_xml_registry_dispatch(
    source_id: str,
    parser_cls: Any,
    parser_version: str,
    fixture_filename: str,
    snapshot_id: str,
    expected_row_count: int,
    content_type: str,
) -> None:
    """``build_phase3_registry().get(source_id)`` returns the right parser."""
    registry = build_phase3_registry()
    found = registry.get(source_id)
    assert found is not None, f"{source_id}: not registered"
    assert isinstance(found, parser_cls), (
        f"{source_id}: registered parser is {type(found).__name__}, "
        f"expected {parser_cls.__name__}"
    )
    assert found.parser_version == parser_version


# ---------------------------------------------------------------------------
# 4. Negative paths
# ---------------------------------------------------------------------------


def test_edgar_negative_missing_required_column() -> None:
    """A CSV missing the ``unit`` column raises :class:`ParserDispatchError`."""
    raw = (
        b"country_iso3,sector,year,pollutant,value\n"
        b"USA,energy,2023,co2,5078.0\n"
    )
    parser = Phase3EDGARCsvParser()
    ctx = _make_ctx(PHASE3_EDGAR_SOURCE_ID, "text/csv", "missing.csv")
    with pytest.raises(ParserDispatchError, match="missing required columns"):
        parser.parse_bytes(ctx, raw)


def test_edgar_negative_unknown_geography() -> None:
    """A row with ``country_iso3=XYZ`` fails with the row number in the message."""
    raw = (
        b"country_iso3,sector,year,pollutant,value,unit\n"
        b"XYZ,energy,2023,co2,5078.0,Mt\n"
    )
    parser = Phase3EDGARCsvParser()
    ctx = _make_ctx(PHASE3_EDGAR_SOURCE_ID, "text/csv", "bad-geo.csv")
    with pytest.raises(ParserDispatchError, match=r"row=2"):
        parser.parse_bytes(ctx, raw)


def test_edgar_negative_zero_value() -> None:
    """A zero-valued emission factor is rejected (Phase 3 spec)."""
    raw = (
        b"country_iso3,sector,year,pollutant,value,unit\n"
        b"USA,energy,2023,co2,0,Mt\n"
    )
    parser = Phase3EDGARCsvParser()
    ctx = _make_ctx(PHASE3_EDGAR_SOURCE_ID, "text/csv", "zero.csv")
    with pytest.raises(ParserDispatchError, match="zero-valued"):
        parser.parse_bytes(ctx, raw)


def test_edgar_negative_negative_value() -> None:
    """Negative emission factors are forbidden (no sequestration in EDGAR)."""
    raw = (
        b"country_iso3,sector,year,pollutant,value,unit\n"
        b"USA,energy,2023,co2,-100,Mt\n"
    )
    parser = Phase3EDGARCsvParser()
    ctx = _make_ctx(PHASE3_EDGAR_SOURCE_ID, "text/csv", "neg.csv")
    with pytest.raises(ParserDispatchError, match="negative"):
        parser.parse_bytes(ctx, raw)


def test_climate_trace_negative_malformed_timestamp() -> None:
    """A malformed ``start_time`` raises with the offending row index."""
    raw = (
        b"asset_id,country_iso3,sector,subsector,gas,emissions_quantity,"
        b"emissions_unit,start_time,end_time\n"
        b"ct-bad-01,USA,power,electricity-generation,co2e,1234.0,tCO2e,"
        b"not-a-date,2024-12-31T23:59:59+00:00\n"
    )
    parser = Phase3ClimateTRACECsvParser()
    ctx = _make_ctx(
        PHASE3_CLIMATE_TRACE_SOURCE_ID, "text/csv", "bad-ts.csv",
    )
    with pytest.raises(ParserDispatchError, match="ISO-8601"):
        parser.parse_bytes(ctx, raw)


def test_climate_trace_sequestration_carve_out_allows_negative() -> None:
    """Forestry sequestration rows MAY be negative (carve-out path)."""
    raw = (
        b"asset_id,country_iso3,sector,subsector,gas,emissions_quantity,"
        b"emissions_unit,start_time,end_time\n"
        b"ct-seq-01,BRA,forestry-and-land-use,reforestation,co2e,"
        b"-12345.0,tCO2e,"
        b"2024-01-01T00:00:00+00:00,2024-12-31T23:59:59+00:00\n"
    )
    parser = Phase3ClimateTRACECsvParser()
    ctx = _make_ctx(
        PHASE3_CLIMATE_TRACE_SOURCE_ID, "text/csv", "seq.csv",
    )
    rows = parser.parse_bytes(ctx, raw)
    assert len(rows) == 1
    assert rows[0]["value"] == -12345.0


def test_entsoe_negative_unknown_area_code() -> None:
    """An ``in_Domain.mRID`` outside the known set raises."""
    raw = b"""<?xml version="1.0"?>
<root>
  <TimeSeries>
    <mRID>TS-XX-0001</mRID>
    <businessType>A04</businessType>
    <in_Domain.mRID>UNKNOWN_AREA_999</in_Domain.mRID>
    <quantity>1234.5</quantity>
    <unit>MWh</unit>
    <timestamp>2024-04-28T00:00:00+00:00</timestamp>
  </TimeSeries>
</root>"""
    parser = Phase3ENTSOEXmlParser()
    ctx = _make_ctx(
        PHASE3_ENTSOE_SOURCE_ID, "application/xml", "bad-area.xml",
    )
    with pytest.raises(ParserDispatchError, match="not a known area"):
        parser.parse_bytes(ctx, raw)


def test_entsoe_negative_format_family_mismatch() -> None:
    """Calling the XML adapter with CSV bytes raises a dispatch error."""
    raw = b"country_iso3,sector,year,pollutant,value,unit\nUSA,energy,2023,co2,5078,Mt\n"
    parser = Phase3ENTSOEXmlParser()
    ctx = _make_ctx(
        PHASE3_ENTSOE_SOURCE_ID, "text/csv", "wrong-family.csv",
    )
    with pytest.raises(ParserDispatchError, match="XML-family"):
        parser.parse_bytes(ctx, raw)


def test_entsoe_negative_missing_required_field() -> None:
    """A ``<TimeSeries>`` missing the ``unit`` field raises with row pointer."""
    raw = b"""<?xml version="1.0"?>
<root>
  <TimeSeries>
    <mRID>TS-DE-NOMIT</mRID>
    <businessType>A04</businessType>
    <in_Domain.mRID>10YDE-VE-------2</in_Domain.mRID>
    <quantity>1234.5</quantity>
    <timestamp>2024-04-28T00:00:00+00:00</timestamp>
  </TimeSeries>
</root>"""
    parser = Phase3ENTSOEXmlParser()
    ctx = _make_ctx(
        PHASE3_ENTSOE_SOURCE_ID, "application/xml", "no-unit.xml",
    )
    with pytest.raises(ParserDispatchError, match="missing required fields"):
        parser.parse_bytes(ctx, raw)
