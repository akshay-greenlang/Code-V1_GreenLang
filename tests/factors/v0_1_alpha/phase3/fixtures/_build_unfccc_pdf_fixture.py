# -*- coding: utf-8 -*-
"""Build the deterministic synthetic UNFCCC BUR mini PDF fixture.

Phase 3 Wave 2.5 calls for a byte-identical, network-free, UNFCCC
BUR-shaped PDF that the unified ingestion pipeline can fetch -> parse
-> normalize -> validate -> dedupe -> stage -> publish end-to-end.

Determinism contract
--------------------
* The fixture is built with :mod:`reportlab` using a fixed pageCompression
  setting and an explicit ``invariant=1`` flag. The reportlab Canvas is
  given an explicit, frozen author/title/subject/keywords payload via
  ``setAuthor`` / ``setTitle`` / ``setSubject`` / ``setKeywords`` so the
  embedded PDF info dict carries no wall-clock value.
* The reportlab PDF info dict normally embeds the *current* time as the
  ``/CreationDate`` and ``/ModDate`` keys; we override these by patching
  the canvas's ``_doc.info`` block AFTER ``setTitle`` and BEFORE
  ``save``.
* If reportlab is not available in the local environment, the fixture
  builder falls back to a hand-crafted minimal PDF byte sequence (a
  single uncompressed text-only page with the table laid out as plain
  text). The fall-back PDF is intentionally *not* table-extractable by
  pdfplumber's table heuristic — tests that need the full pdfplumber
  table-extraction path skip themselves when reportlab is absent.

Fixture content
---------------
* 1 page.
* 1 simple table with a header row + 5 data rows. The columns mirror a
  trimmed UNFCCC BUR national-inventory-report excerpt:

      Fuel | Unit  | EF   | Notes
      -----|-------|------|------------------------
      ng   | TJ    | 56.1 | Stationary combustion
      die  | TJ    | 74.1 | Mobile combustion
      lpg  | TJ    | 63.1 | Stationary combustion
      coal | TJ    | 94.6 | Stationary combustion
      bio  | TJ    | 0.0  | Biogenic, reported separately

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Block 3 -- PDF/OCR family"
- ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 3.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

#: Deterministic stamp written into the PDF's info dict so the embedded
#: ``/CreationDate`` and ``/ModDate`` keys do NOT carry a wall-clock value.
_FROZEN_STAMP = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)


#: Header columns the synthetic UNFCCC BUR mini fixture carries.
UNFCCC_BUR_HEADERS: Tuple[str, ...] = ("Fuel", "Unit", "EF", "Notes")

#: Five data rows. Values are intentionally small + plain-ASCII so the
#: PDF text stream is byte-stable across reportlab font caches.
UNFCCC_BUR_ROWS: Tuple[Tuple[Any, ...], ...] = (
    ("natural_gas", "TJ", 56.1, "Stationary combustion"),
    ("diesel", "TJ", 74.1, "Mobile combustion"),
    ("lpg", "TJ", 63.1, "Stationary combustion"),
    ("coal_industrial", "TJ", 94.6, "Stationary combustion"),
    ("biomass_wood_pellets", "TJ", 0.0, "Biogenic, reported separately"),
)


def _build_with_reportlab() -> bytes:
    """Build the fixture via reportlab's Canvas + Table primitives."""
    from reportlab.lib import colors  # noqa: PLC0415
    from reportlab.lib.pagesizes import letter  # noqa: PLC0415
    from reportlab.pdfgen import canvas  # noqa: PLC0415
    from reportlab.platypus import Table, TableStyle  # noqa: PLC0415

    buf = io.BytesIO()
    # ``invariant=1`` strips the wall-clock CreationDate. ``pageCompression=0``
    # keeps the content stream uncompressed so the bytes are inspection-
    # friendly + deterministic across reportlab versions.
    c = canvas.Canvas(
        buf,
        pagesize=letter,
        invariant=1,
        pageCompression=0,
    )
    c.setAuthor("GreenLang Factors / Phase 3 Wave 2.5 fixture")
    c.setTitle("UNFCCC BUR mini fixture")
    c.setSubject("Phase 3 Wave 2.5 PDF/OCR test fixture")
    c.setKeywords("greenlang factors phase3 unfccc bur fixture")

    # Override the info dict's date fields with the frozen stamp so the
    # PDF byte stream is byte-stable across runs / clocks.
    try:
        info = c._doc.info  # noqa: SLF001 — reportlab internal
        # reportlab's Date objects accept datetime instances. Some
        # versions key the dates by attribute name; try both.
        info.invariant = True
        if hasattr(info, "_creationDate"):
            info._creationDate = _FROZEN_STAMP  # noqa: SLF001
        if hasattr(info, "_modDate"):
            info._modDate = _FROZEN_STAMP  # noqa: SLF001
    except Exception:  # noqa: BLE001 — invariant=1 already does most of the work
        pass

    # Page header.
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "UNFCCC BUR — National Inventory excerpt")
    c.setFont("Helvetica", 10)
    c.drawString(
        72, 700,
        "Synthetic Phase 3 Wave 2.5 fixture (deterministic).",
    )

    # Build the data table.
    data: List[List[Any]] = [list(UNFCCC_BUR_HEADERS)]
    for row in UNFCCC_BUR_ROWS:
        data.append(list(row))
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    table.wrapOn(c, 460, 200)
    table.drawOn(c, 72, 580)

    c.showPage()
    c.save()
    return buf.getvalue()


def _build_minimal_fallback() -> bytes:
    """Hand-crafted minimal PDF byte stream — fallback when reportlab is absent.

    The output is a syntactically valid 1-page PDF with the header +
    table laid out as plain text (no PDF table primitives). pdfplumber
    can extract the text but its table heuristic does not detect the
    layout as a table — tests that need the full table-extraction
    path skip themselves when reportlab is absent (the parser's
    ``injected_tables`` kwarg is the canonical bypass).
    """
    # Build the body content stream first so we know its byte length.
    text_lines: List[str] = ["UNFCCC BUR -- National Inventory excerpt", ""]
    text_lines.append(" | ".join(UNFCCC_BUR_HEADERS))
    text_lines.append("-" * 60)
    for row in UNFCCC_BUR_ROWS:
        text_lines.append(" | ".join(str(v) for v in row))
    body_lines = "\n".join(text_lines)

    # Construct an uncompressed PDF page content stream. ``Tj`` shows
    # text; ``T*`` advances to the next line in current text object.
    stream_parts: List[str] = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
    for line in body_lines.split("\n"):
        # Escape PDF string special characters.
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_parts.append("(%s) Tj T*" % safe)
    stream_parts.append("ET")
    content_stream = "\n".join(stream_parts).encode("ascii")

    # Build the indirect objects.
    objects: List[bytes] = []
    # Object 1: catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2: pages tree
    objects.append(
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    )
    # Object 3: page
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    # Object 4: content stream
    obj4 = (
        b"4 0 obj\n<< /Length %d >>\nstream\n" % len(content_stream)
        + content_stream
        + b"\nendstream\nendobj\n"
    )
    objects.append(obj4)
    # Object 5: font
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )
    # Object 6: info dict (frozen)
    objects.append(
        b"6 0 obj\n"
        b"<< /Title (UNFCCC BUR mini fixture) "
        b"/Author (GreenLang Factors / Phase 3 Wave 2.5 fixture) "
        b"/Subject (Phase 3 Wave 2.5 PDF/OCR test fixture) "
        b"/Producer (GreenLang fallback PDF builder) "
        b"/CreationDate (D:20260428000000Z) "
        b"/ModDate (D:20260428000000Z) >>\n"
        b"endobj\n"
    )

    # Assemble the PDF — header + objects + xref + trailer.
    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets: List[int] = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj
    xref_offset = len(pdf)
    n_objects = len(objects) + 1  # +1 for the conventional object 0
    pdf += b"xref\n0 %d\n" % n_objects
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += (
        b"trailer\n<< /Size %d /Root 1 0 R /Info 6 0 R >>\n"
        b"startxref\n%d\n%%%%EOF\n"
    ) % (n_objects, xref_offset)
    return pdf


def build_pdf_bytes() -> bytes:
    """Return the deterministic PDF bytes (no I/O).

    Uses reportlab when available; falls back to a hand-crafted minimal
    PDF byte stream otherwise. Both code paths produce *deterministic*
    byte output — the choice between them is locked at fixture-build
    time by whether ``reportlab`` is importable.
    """
    try:
        import reportlab  # noqa: F401, PLC0415
    except ImportError:
        return _build_minimal_fallback()
    return _build_with_reportlab()


def ensure_fixture(path: Path) -> Path:
    """Materialise the UNFCCC BUR mini PDF fixture at ``path`` if missing.

    Idempotent: if the file already exists, returns immediately. If the
    file is absent, creates parent directories and writes the
    deterministic bytes.

    Returns:
        ``path`` (for ergonomic chaining at fixture-collection time).
    """
    p = Path(path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_pdf_bytes())
    return p


def all_rows() -> List[Tuple[str, Tuple[str, ...], Tuple[Tuple[Any, ...], ...]]]:
    """Return ``[(table_label, headers, rows)]`` for assertions.

    Phase 3 e2e tests use this to count expected rows without re-parsing
    the PDF (which would tie the test to pdfplumber).
    """
    return [
        ("UNFCCC BUR — National Inventory", UNFCCC_BUR_HEADERS, UNFCCC_BUR_ROWS),
    ]


__all__ = [
    "UNFCCC_BUR_HEADERS",
    "UNFCCC_BUR_ROWS",
    "build_pdf_bytes",
    "ensure_fixture",
    "all_rows",
]


if __name__ == "__main__":  # pragma: no cover — manual regen helper
    fixture_path = Path(__file__).resolve().parent / "unfccc_bur_mini.pdf"
    ensure_fixture(fixture_path)
    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes)")
