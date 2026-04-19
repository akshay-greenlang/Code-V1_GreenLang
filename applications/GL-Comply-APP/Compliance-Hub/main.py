# -*- coding: utf-8 -*-
"""GL-Comply-APP FastAPI entrypoint."""

from __future__ import annotations

import logging

try:
    from fastapi import FastAPI
except ImportError as e:
    raise RuntimeError("FastAPI not installed; install greenlang[server]") from e

try:
    __version__ = __import__("__init__", fromlist=["__version__"]).__version__
except Exception:
    __version__ = "0.1.0"
from api.v1 import build_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="GL-Comply-APP — Unified Compliance Hub",
        version=__version__,
        description=(
            "Single API surface for CSRD, CBAM, EUDR, GHG, ISO14064, "
            "SB253, SBTi, Taxonomy, TCFD, CDP compliance workflows."
        ),
    )
    app.include_router(build_router(), prefix="/api/v1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
