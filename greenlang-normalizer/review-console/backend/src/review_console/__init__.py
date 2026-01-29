"""
Review Console Backend for GreenLang Entity Resolution (GL-FOUND-X-003).

This package provides the FastAPI backend for the human review queue,
enabling domain experts to review and resolve low-confidence entity
matches from the GreenLang normalizer pipeline.

Key Components:
    - API routes for queue management, resolution, and statistics
    - PostgreSQL-backed persistence with SQLAlchemy
    - JWT-based authentication and RBAC
    - Rate limiting and audit logging

Example:
    >>> from review_console.main import app
    >>> # Run with: uvicorn review_console.main:app --reload
"""

__version__ = "0.1.0"
__author__ = "GreenLang Team"
__email__ = "engineering@greenlang.io"

__all__ = ["__version__"]
