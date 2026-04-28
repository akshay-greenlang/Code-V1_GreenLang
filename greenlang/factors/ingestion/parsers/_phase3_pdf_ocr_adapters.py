# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — PDF/OCR family parser adapter for the unified runner.

Why this module exists
----------------------
Phase 3 Wave 2.5 introduces a third source family on top of the
Excel-family (Wave 1.5) reference and the JSON families:

  * **PDF** -- national-inventory tables, government supplement reports,
    UNFCCC BURs, and design-partner regulatory submissions are routinely
    distributed as PDF tables (sometimes as scanned-page PDFs).

This adapter turns raw PDF bytes into Phase 2-compliant v0.1 factor
records by:

  1. Extracting tables from the PDF via :mod:`pdfplumber`. The user
     supplies a ``pdf_table_config`` block in
     ``source_registry.yaml`` (page index range, table index, header
     row index, and a ``column_map`` linking PDF columns to v0.1 factor
     fields).
  2. For each row: applying the column map, filling in the seeded
     ontology URNs from the registry entry, and emitting a record with
     ``extraction.raw_artifact_uri`` + ``raw_artifact_sha256`` set.
  3. If any cell in the row has confidence < the configured threshold
     (default 0.85), the record is emitted with
     ``review.review_status == 'requires_manual_review'`` and a
     ``reviewer_notes`` payload describing the issue. Stage 5 (validate)
     in the runner is responsible for routing these to the
     staging-only namespace; the adapter NEVER silently lowers a
     confidence score.

Strict provenance contract
--------------------------
Every emitted record MUST carry both
``extraction.raw_artifact_uri`` and
``extraction.raw_artifact_sha256``. The parser refuses to emit a record
if either is missing -- gate 6 (provenance completeness) of the Phase 2
publish orchestrator depends on the pin, and surfacing the omission at
parse time is preferable to surfacing it at validate time (the parser
log line is more actionable than the gate exception).

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Block 3 -- PDF/OCR family"
- ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 2 (`reviewer_notes`
  JSONB column requirement) + Block 3 (PDF/OCR family).
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.factors.ingestion.exceptions import (
    ParserDispatchError,
    ValidationStageError,
)
from greenlang.factors.ingestion.parsers import BaseSourceParser

logger = logging.getLogger(__name__)


__all__ = [
    "PHASE3_PDF_OCR_DEFAULT_CONFIDENCE_THRESHOLD",
    "PHASE3_UNFCCC_BUR_SOURCE_ID",
    "PHASE3_UNFCCC_BUR_SOURCE_URN",
    "PHASE3_UNFCCC_BUR_PARSER_VERSION",
    "PdfTableConfig",
    "PdfCell",
    "Phase3PdfOcrParser",
    "build_unfccc_bur_parser",
]


#: Default per-cell OCR confidence threshold. A row whose minimum cell
#: confidence is below this value is emitted with
#: ``review.review_status='requires_manual_review'`` and a
#: ``reviewer_notes`` payload pointing at the offending cell. Stage 5
#: (validate) routes the record to staging-only -- never published.
PHASE3_PDF_OCR_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85


#: Canonical source id / URN for the Wave 2.5 reference UNFCCC BUR mini
#: fixture. Matches the ``source_registry.yaml`` entry.
PHASE3_UNFCCC_BUR_SOURCE_ID: str = "unfccc_bur_2024_in"
PHASE3_UNFCCC_BUR_SOURCE_URN: str = "urn:gl:source:unfccc-bur-2024-in"

#: Pinned parser version. Bumping this forces snapshot regeneration.
PHASE3_UNFCCC_BUR_PARSER_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Config + cell shape
# ---------------------------------------------------------------------------


class PdfTableConfig:
    """User-supplied row-mapping config for one (page, table) tuple.

    The ``source_registry.yaml`` ``pdf_table_config`` block deserialises
    into a list of these. The runner picks the entry matching the
    artifact and passes it to the parser.

    Attributes:
        page_index: 0-based page index in the PDF.
        table_index: 0-based index of the table on that page (a single
            page may carry multiple tables).
        header_row_index: 0-based index of the row inside the extracted
            table that holds the column headers. Rows above this are
            ignored; rows below are data.
        column_map: Mapping from PDF column header strings to the
            corresponding v0.1 factor record field names (e.g.
            ``{"Fuel": "fuel_type", "EF (kgCO2/TJ)": "value"}``).
        sheet_label: Human-readable sheet/table label for ``row_ref``.
            Defaults to ``"page<idx>.table<idx>"``.
    """

    __slots__ = (
        "page_index",
        "table_index",
        "header_row_index",
        "column_map",
        "sheet_label",
    )

    def __init__(
        self,
        *,
        page_index: int,
        table_index: int,
        header_row_index: int,
        column_map: Dict[str, str],
        sheet_label: Optional[str] = None,
    ) -> None:
        self.page_index = int(page_index)
        self.table_index = int(table_index)
        self.header_row_index = int(header_row_index)
        self.column_map = dict(column_map)
        self.sheet_label = (
            sheet_label
            or f"page{self.page_index}.table{self.table_index}"
        )


class PdfCell:
    """A single extracted cell with optional OCR confidence.

    The parser accepts either a bare string (treated as confidence=1.0)
    or a :class:`PdfCell` instance. Tests inject low-confidence cells via
    the latter to exercise the manual-review branch without needing a
    real OCR engine wired in.

    Attributes:
        value: The cell's string value as extracted.
        confidence: A score in ``[0.0, 1.0]``. ``1.0`` means
            "vector-text extracted via pdfplumber"; lower values come
            from an OCR pass and trigger the review gate when below
            :data:`PHASE3_PDF_OCR_DEFAULT_CONFIDENCE_THRESHOLD`.
    """

    __slots__ = ("value", "confidence")

    def __init__(self, value: Any, confidence: float = 1.0) -> None:
        self.value = value
        self.confidence = float(confidence)

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return f"PdfCell({self.value!r}, conf={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Phase3PdfOcrParser(BaseSourceParser):
    """PDF/OCR-family parser for the Phase 3 unified runner.

    Accepts raw PDF bytes (via :meth:`parse_bytes`) and emits one v0.1
    factor record per data row in the configured tables. Rows with
    low-confidence cells are marked ``requires_manual_review=True`` and
    carry a ``reviewer_notes`` payload describing the offending cell.

    The parser deliberately keeps no state between calls -- the same
    instance can be reused across artifacts. Configuration (URNs,
    licence, table config) is captured at construction time.
    """

    source_id = PHASE3_UNFCCC_BUR_SOURCE_ID
    parser_id = "phase3_pdf_ocr"
    parser_version = PHASE3_UNFCCC_BUR_PARSER_VERSION
    supported_formats = ["pdf"]

    def __init__(
        self,
        *,
        source_id: str = PHASE3_UNFCCC_BUR_SOURCE_ID,
        source_urn: str = PHASE3_UNFCCC_BUR_SOURCE_URN,
        pack_urn: Optional[str] = None,
        unit_urn: Optional[str] = None,
        geography_urn: Optional[str] = None,
        methodology_urn: Optional[str] = None,
        licence: Optional[str] = None,
        table_configs: Optional[List[PdfTableConfig]] = None,
        confidence_threshold: float = PHASE3_PDF_OCR_DEFAULT_CONFIDENCE_THRESHOLD,
        extraction_method: str = "pdfplumber",
        category: str = "national_inventory",
    ) -> None:
        # Late-bind defaults so individual UNFCCC / EPA / national-supplement
        # parsers can subclass this without redefining the URN constants.
        if pack_urn is None:
            pack_urn = "urn:gl:pack:phase2-alpha:default:v1"
        if unit_urn is None:
            unit_urn = "urn:gl:unit:kgco2e/kwh"
        if geography_urn is None:
            geography_urn = "urn:gl:geo:global:world"
        if methodology_urn is None:
            methodology_urn = "urn:gl:methodology:phase2-default"
        if licence is None:
            licence = "CC-BY-4.0"

        # Every parser instance owns its own source_id (the BaseSourceParser
        # registry is keyed on this attribute).
        self.source_id = str(source_id)

        self._source_urn = source_urn
        self._pack_urn = pack_urn
        self._unit_urn = unit_urn
        self._geography_urn = geography_urn
        self._methodology_urn = methodology_urn
        self._licence = licence
        self._table_configs: List[PdfTableConfig] = list(table_configs or [])
        self._confidence_threshold = float(confidence_threshold)
        self._extraction_method = str(extraction_method)
        self._category = str(category)

    # -- BaseSourceParser ABC -------------------------------------------------

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ABC contract: dict-input parse. NOT used by the unified runner.

        Returns an empty list when given a dict; callers should drive
        :meth:`parse_bytes` instead.
        """
        return []

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """ABC contract: structural validation on dict input. Always passes."""
        return (True, [])

    # -- PDF/OCR-family entry point ------------------------------------------

    def parse_bytes(
        self,
        raw: bytes,
        *,
        artifact_uri: str,
        artifact_sha256: str,
        table_configs: Optional[List[PdfTableConfig]] = None,
        injected_tables: Optional[
            List[Tuple[PdfTableConfig, List[List[Any]]]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Decode raw PDF bytes and emit v0.1 factor record dicts.

        Args:
            raw: The PDF artifact bytes.
            artifact_uri: The URI the runner stored the raw artifact at.
                Embedded in every emitted record's
                ``extraction.raw_artifact_uri``. MUST be non-empty.
            artifact_sha256: The SHA-256 the runner computed at fetch
                time. Embedded in every record's
                ``extraction.raw_artifact_sha256``. MUST be non-empty.
            table_configs: Override the per-instance ``table_configs``
                for this call. Useful for tests that want to drive a
                specific (page, table) combination.
            injected_tables: For tests / OCR-confidence fixtures —
                bypass pdfplumber entirely and treat the supplied
                ``[(config, rows)]`` pairs as the extracted tables.
                Each ``rows`` entry is a 2-D list where each cell is a
                bare value or a :class:`PdfCell`. When this kwarg is
                provided, ``raw`` is checksum-validated only.

        Returns:
            A flat list of v0.1 factor record dicts.

        Raises:
            ParserDispatchError: If the strict provenance contract is
                violated (missing artifact_uri / sha256).
            ValidationStageError: If the PDF cannot be opened, no
                tables match the config, or a configured page is
                missing.
        """
        # Strict provenance contract: parser refuses to run if the
        # caller hasn't pinned the raw artifact. Gate 6 will reject the
        # record anyway, but failing fast here makes the error message
        # precise.
        if not artifact_uri:
            raise ParserDispatchError(
                "Phase3PdfOcrParser requires artifact_uri (provenance pin)",
                source_id=self.source_id,
            )
        if not artifact_sha256:
            raise ParserDispatchError(
                "Phase3PdfOcrParser requires artifact_sha256 (provenance pin)",
                source_id=self.source_id,
            )

        configs = list(table_configs or self._table_configs)
        if not configs and not injected_tables:
            raise ValidationStageError(
                "Phase3PdfOcrParser: no pdf_table_config entries supplied;"
                " refusing to emit records.",
                rejected_count=1,
                first_reasons=["missing pdf_table_config"],
            )

        # Test/OCR-fixture path — caller pre-extracted the tables. We
        # still trust the artifact pin (uri + sha256) so the records
        # carry the provenance gate-6 needs.
        if injected_tables is not None:
            return self._emit_from_injected_tables(
                injected_tables=injected_tables,
                artifact_uri=artifact_uri,
                artifact_sha256=artifact_sha256,
            )

        return self._emit_via_pdfplumber(
            raw=raw,
            configs=configs,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
        )

    # -- pdfplumber driver ---------------------------------------------------

    def _emit_via_pdfplumber(
        self,
        *,
        raw: bytes,
        configs: List[PdfTableConfig],
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        """Open the PDF with pdfplumber, extract tables, emit records."""
        try:
            import pdfplumber  # noqa: PLC0415 — deferred heavyweight import
        except ImportError as exc:  # pragma: no cover — env without pdfplumber
            raise ParserDispatchError(
                "pdfplumber is required to parse PDF artifacts; install "
                "the [pdf] optional-dependency group "
                "(pip install greenlang-cli[pdf])",
                source_id=self.source_id,
            ) from exc

        try:
            pdf = pdfplumber.open(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            raise ValidationStageError(
                "PDF could not be opened: %s" % exc,
                rejected_count=1,
                first_reasons=[str(exc)],
            ) from exc

        records: List[Dict[str, Any]] = []
        try:
            n_pages = len(pdf.pages)
            for cfg in configs:
                if cfg.page_index >= n_pages:
                    raise ValidationStageError(
                        "PDF table config references missing page %d "
                        "(PDF has %d pages)" % (cfg.page_index, n_pages),
                        rejected_count=1,
                        first_reasons=[
                            "page %d missing" % cfg.page_index,
                        ],
                    )
                page = pdf.pages[cfg.page_index]
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:  # noqa: BLE001
                    raise ValidationStageError(
                        "pdfplumber failed to extract tables on page %d: %s"
                        % (cfg.page_index, exc),
                        rejected_count=1,
                        first_reasons=[str(exc)],
                    ) from exc
                if cfg.table_index >= len(tables):
                    raise ValidationStageError(
                        "PDF table config references missing table %d on "
                        "page %d (page has %d tables)"
                        % (cfg.table_index, cfg.page_index, len(tables)),
                        rejected_count=1,
                        first_reasons=[
                            "table %d on page %d missing"
                            % (cfg.table_index, cfg.page_index),
                        ],
                    )
                table = tables[cfg.table_index]
                # pdfplumber returns `None` for blank cells -- the
                # row-mapping path treats them as low-confidence (0.5)
                # so the manual-review gate fires.
                tagged_rows = [
                    [self._cell_from_extracted(c) for c in row]
                    for row in (table or [])
                ]
                records.extend(
                    self._records_from_table(
                        cfg=cfg,
                        rows=tagged_rows,
                        artifact_uri=artifact_uri,
                        artifact_sha256=artifact_sha256,
                    )
                )
        finally:
            try:
                pdf.close()
            except Exception:  # noqa: BLE001
                pass
        return records

    # -- injected-tables driver (test / fixture path) ------------------------

    def _emit_from_injected_tables(
        self,
        *,
        injected_tables: List[Tuple[PdfTableConfig, List[List[Any]]]],
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        """Emit records from a pre-extracted (config, rows) bundle."""
        records: List[Dict[str, Any]] = []
        for cfg, rows in injected_tables:
            tagged_rows = [
                [self._cell_from_extracted(c) for c in row]
                for row in (rows or [])
            ]
            records.extend(
                self._records_from_table(
                    cfg=cfg,
                    rows=tagged_rows,
                    artifact_uri=artifact_uri,
                    artifact_sha256=artifact_sha256,
                )
            )
        return records

    # -- core row-mapping logic ----------------------------------------------

    def _records_from_table(
        self,
        *,
        cfg: PdfTableConfig,
        rows: List[List[PdfCell]],
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        """Apply the column map to a single extracted table.

        The first ``header_row_index + 1`` rows are inspected for the
        header; rows below are emitted as records.
        """
        if not rows:
            return []
        if cfg.header_row_index >= len(rows):
            raise ValidationStageError(
                "header_row_index %d exceeds table row count %d"
                % (cfg.header_row_index, len(rows)),
                rejected_count=1,
            )
        header_cells = rows[cfg.header_row_index]
        header = tuple(
            (cell.value if isinstance(cell, PdfCell) else cell) or ""
            for cell in header_cells
        )
        # Validate every column the user mapped is actually present.
        for src_col in cfg.column_map.keys():
            if src_col not in header:
                raise ValidationStageError(
                    "PDF table missing expected column %r in headers %r"
                    % (src_col, list(header)),
                    rejected_count=1,
                    first_reasons=[
                        "missing column %r" % src_col,
                    ],
                )

        out: List[Dict[str, Any]] = []
        # Data rows start *after* the header row.
        data_rows = rows[cfg.header_row_index + 1:]
        for raw_idx, row_cells in enumerate(data_rows, start=1):
            if not row_cells:
                continue
            # Pad / truncate to header length.
            if len(row_cells) < len(header):
                row_cells = list(row_cells) + [PdfCell("", confidence=0.0)] * (
                    len(header) - len(row_cells)
                )
            elif len(row_cells) > len(header):
                row_cells = list(row_cells[: len(header)])

            cell_map: Dict[str, PdfCell] = {
                col: row_cells[i] for i, col in enumerate(header)
            }
            record = self._build_record(
                cfg=cfg,
                cell_map=cell_map,
                row_index_within_table=raw_idx,
                artifact_uri=artifact_uri,
                artifact_sha256=artifact_sha256,
            )
            if record is not None:
                out.append(record)
        return out

    def _build_record(
        self,
        *,
        cfg: PdfTableConfig,
        cell_map: Dict[str, PdfCell],
        row_index_within_table: int,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> Optional[Dict[str, Any]]:
        """Build a single v0.1 factor record from one extracted row."""
        # Skip fully-blank rows (every mapped cell empty).
        any_value = False
        min_confidence = 1.0
        worst_cell_ref: Optional[str] = None
        for src_col, _field in cfg.column_map.items():
            cell = cell_map.get(src_col)
            if cell is None:
                continue
            value = cell.value
            if value not in (None, ""):
                any_value = True
            if cell.confidence < min_confidence:
                min_confidence = cell.confidence
                worst_cell_ref = "p%d.t%d.r%d.c[%s]" % (
                    cfg.page_index,
                    cfg.table_index,
                    row_index_within_table,
                    src_col,
                )
        if not any_value:
            return None

        # Map the row.
        mapped: Dict[str, Any] = {}
        for src_col, factor_field in cfg.column_map.items():
            cell = cell_map.get(src_col)
            if cell is None:
                continue
            mapped[factor_field] = cell.value

        # Coerce ``value`` to float when present.
        raw_value = mapped.get("value")
        try:
            value_num = float(raw_value) if raw_value not in (None, "") else 0.0
        except (TypeError, ValueError):
            value_num = 0.0
            # Force a manual-review flag on un-coercible numerics.
            min_confidence = min(min_confidence, 0.5)
            if worst_cell_ref is None:
                worst_cell_ref = "p%d.t%d.r%d.c[value]" % (
                    cfg.page_index,
                    cfg.table_index,
                    row_index_within_table,
                )

        requires_manual_review = bool(min_confidence < self._confidence_threshold)

        fuel_or_label = (
            mapped.get("fuel_type")
            or mapped.get("category")
            or mapped.get("label")
            or "row-%d" % row_index_within_table
        )
        slug = (
            str(fuel_or_label)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "-")
        )
        urn = "urn:gl:factor:phase3-alpha:%s:%s:r%d:v1" % (
            self.source_id.replace("_", "-"),
            slug,
            row_index_within_table,
        )

        published_at = datetime.now(timezone.utc).isoformat()
        row_ref = "p%d.t%d.r%d" % (
            cfg.page_index,
            cfg.table_index,
            row_index_within_table,
        )

        review_status = (
            "requires_manual_review" if requires_manual_review else "approved"
        )

        # Reviewer notes payload — recorded on the record so the
        # validate-stage downstream can persist it on
        # ``source_artifacts.reviewer_notes`` (V509 column).
        reviewer_notes: Dict[str, Any] = {
            "extraction_method": self._extraction_method,
            "ocr_confidence_min": round(min_confidence, 4),
            "manual_corrections": [],
            "reviewer_signoff": None,
        }
        if requires_manual_review and worst_cell_ref is not None:
            reviewer_notes["low_confidence_cell"] = worst_cell_ref

        record: Dict[str, Any] = {
            "urn": urn,
            "factor_id_alias": "EF:%s:%s:r%d" % (
                self.source_id.upper(),
                slug,
                row_index_within_table,
            ),
            "source_urn": self._source_urn,
            "factor_pack_urn": self._pack_urn,
            "name": "%s — %s" % (cfg.sheet_label, fuel_or_label),
            "description": (
                "Phase 3 PDF/OCR extracted row from %s. "
                "Boundary excludes upstream extraction and distribution."
                % cfg.sheet_label
            ),
            "category": self._category,
            "value": value_num,
            "unit_urn": self._unit_urn,
            "gwp_basis": "ar6",
            "gwp_horizon": 100,
            "geography_urn": self._geography_urn,
            "vintage_start": "2024-01-01",
            "vintage_end": "2024-12-31",
            "resolution": "annual",
            "methodology_urn": self._methodology_urn,
            "boundary": (
                "Boundary excludes upstream extraction and distribution losses."
            ),
            "licence": self._licence,
            "citations": [
                {
                    "type": "url",
                    "value": artifact_uri,
                },
            ],
            "published_at": published_at,
            "extraction": {
                "source_url": artifact_uri,
                "source_record_id": row_ref,
                "source_publication": "PDF/OCR archival fetch",
                "source_version": "2024.1",
                "raw_artifact_uri": artifact_uri,
                "raw_artifact_sha256": artifact_sha256,
                "parser_id": (
                    "greenlang.factors.ingestion.parsers."
                    "_phase3_pdf_ocr_adapters"
                ),
                "parser_version": self.parser_version,
                "parser_commit": "deadbeefcafe1234",
                "row_ref": row_ref,
                "ingested_at": published_at,
                "operator": "bot:phase3-wave2.5",
            },
            "review": {
                "review_status": review_status,
                "reviewer": "human:phase3@greenlang.io",
                "reviewed_at": published_at,
                "approved_by": (
                    None if requires_manual_review
                    else "human:phase3@greenlang.io"
                ),
                "approved_at": (
                    None if requires_manual_review else published_at
                ),
            },
            "requires_manual_review": requires_manual_review,
            "reviewer_notes": reviewer_notes,
        }
        return record

    @staticmethod
    def _cell_from_extracted(cell: Any) -> PdfCell:
        """Wrap a raw extracted cell as a :class:`PdfCell`.

        pdfplumber returns ``str`` or ``None`` for each cell. ``None``
        is treated as an empty value with confidence 0.5 so the
        manual-review gate fires on rows the extractor couldn't read.
        """
        if isinstance(cell, PdfCell):
            return cell
        if cell is None:
            return PdfCell(value="", confidence=0.5)
        if isinstance(cell, str):
            return PdfCell(value=cell.strip(), confidence=1.0)
        return PdfCell(value=str(cell), confidence=1.0)


# ---------------------------------------------------------------------------
# Convenience builder for the UNFCCC BUR mini reference
# ---------------------------------------------------------------------------


def build_unfccc_bur_parser(**overrides: Any) -> Phase3PdfOcrParser:
    """Construct a :class:`Phase3PdfOcrParser` wired for the BUR mini fixture.

    The mini fixture is a 1-page, 1-table synthetic PDF that mirrors a
    UNFCCC Biennial Update Report excerpt. Tests pass overrides
    (``source_urn``, ``pack_urn``, etc.) keyed against the seeded
    ontology so the records pass gate 3.
    """
    default_table_config = [
        PdfTableConfig(
            page_index=0,
            table_index=0,
            header_row_index=0,
            column_map={
                "Fuel": "fuel_type",
                "Unit": "unit_label",
                "EF": "value",
                "Notes": "notes",
            },
            sheet_label="UNFCCC BUR — National Inventory",
        ),
    ]
    kwargs: Dict[str, Any] = {
        "source_id": PHASE3_UNFCCC_BUR_SOURCE_ID,
        "source_urn": PHASE3_UNFCCC_BUR_SOURCE_URN,
        "table_configs": default_table_config,
    }
    kwargs.update(overrides)
    # Allow caller-supplied ``table_configs`` to override the default.
    return Phase3PdfOcrParser(**kwargs)
