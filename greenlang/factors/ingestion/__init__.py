# -*- coding: utf-8 -*-
"""Raw artifact store, fetchers, parser harness (D1–D4) + Phase 3 runner.

Phase 3 (2026-04-28) adds the unified ingestion-pipeline composition layer:

- ``pipeline``       — RunStatus / Stage enums + IngestionRun / StageResult
                       dataclasses + transition matrix.
- ``runner``         — IngestionPipelineRunner that composes existing
                       fetchers / parsers / normalizer / dedupe /
                       AlphaPublisher into the strict 7-stage contract.
- ``diff``           — RunDiff dataclass + deterministic JSON/MD export
                       used as the methodology-lead approval artefact.
- ``run_repository`` — IngestionRunRepository (SQLite + Postgres, mirrors
                       the alpha repo dual-backend pattern).
- ``exceptions``     — Typed errors raised by the runner.

Lazy-import via the explicit module references is preferred for the new
modules so importing this package does not pull the publish-gate stack
unless a caller actually needs the runner.
"""

from greenlang.factors.ingestion.artifacts import ArtifactStore, LocalArtifactStore
from greenlang.factors.ingestion.fetchers import BaseFetcher, HttpFetcher, FileFetcher
from greenlang.factors.ingestion.parser_harness import ParserContext, ParserResult, run_parser

__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "BaseFetcher",
    "HttpFetcher",
    "FileFetcher",
    "ParserContext",
    "ParserResult",
    "run_parser",
]
