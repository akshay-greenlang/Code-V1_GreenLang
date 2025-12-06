"""
GreenLang API Documentation

OpenAPI 3.1 specification and documentation customization:
- Custom Redoc theme
- Code samples (Python, JavaScript, cURL)
- Authentication guide
- Rate limiting documentation
"""

from app.docs.openapi_config import (
    get_openapi_spec,
    configure_openapi,
    OpenAPIConfig,
)

__all__ = [
    "get_openapi_spec",
    "configure_openapi",
    "OpenAPIConfig",
]
