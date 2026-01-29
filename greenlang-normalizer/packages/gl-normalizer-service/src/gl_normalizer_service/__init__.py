"""
GL Normalizer Service - FastAPI REST API for GreenLang Normalizer

This package provides a production-grade REST API for the GreenLang Normalizer,
implementing GL-FOUND-X-003 specifications for climate data normalization.

Features:
    - Single value normalization (POST /v1/normalize)
    - Batch normalization up to 10K items (POST /v1/normalize/batch)
    - Async job processing for 100K+ items (POST /v1/jobs)
    - Vocabulary management (GET /v1/vocabularies)
    - JWT/API Key authentication
    - Rate limiting and audit trails
    - OpenAPI documentation

Example:
    >>> from gl_normalizer_service import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn gl_normalizer_service.main:app

Version: 1.0.0
API Revision: 2026-01-30
"""

__version__ = "1.0.0"
__api_revision__ = "2026-01-30"

from gl_normalizer_service.main import create_app

__all__ = ["create_app", "__version__", "__api_revision__"]
