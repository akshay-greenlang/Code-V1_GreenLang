"""
GreenLang Agent Factory - End-to-End Test Suite

This package contains end-to-end tests that validate the complete system
behavior across all services, including:

- Agent lifecycle management (creation, execution, versioning)
- Registry workflows (publish, pull, search)
- Authentication and authorization flows
- Multi-tenant isolation
- Certification workflows
- Performance under realistic conditions

Run with: tox -e e2e
Or: pytest tests/e2e/ -v -m e2e

Environment Variables:
    E2E_TESTS: Set to "1" to enable E2E tests
    E2E_BASE_URL: Base URL for API (default: http://localhost:8000)
    E2E_REGISTRY_URL: Registry service URL (default: http://localhost:8002)
    E2E_TIMEOUT: Request timeout in seconds (default: 30)

Prerequisites:
    - Docker Compose services running: docker-compose up -d
    - Playwright installed: playwright install chromium
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"
