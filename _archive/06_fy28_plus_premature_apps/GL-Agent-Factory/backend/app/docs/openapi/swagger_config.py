"""
Swagger UI Configuration for GL-Agent-Factory

This module provides Swagger UI customization and configuration for the API documentation.

Usage:
    from app.docs.openapi.swagger_config import configure_swagger_ui, SWAGGER_UI_CONFIG
"""
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from starlette.requests import Request


# Swagger UI configuration options
SWAGGER_UI_CONFIG: Dict[str, Any] = {
    # Display settings
    "deepLinking": True,
    "displayOperationId": True,
    "defaultModelsExpandDepth": 2,
    "defaultModelExpandDepth": 2,
    "defaultModelRendering": "example",
    "displayRequestDuration": True,
    "docExpansion": "list",
    "filter": True,
    "showExtensions": True,
    "showCommonExtensions": True,

    # Try it out settings
    "tryItOutEnabled": True,
    "supportedSubmitMethods": ["get", "post", "put", "delete", "patch"],

    # Syntax highlighting
    "syntaxHighlight.activate": True,
    "syntaxHighlight.theme": "monokai",

    # Layout
    "layout": "StandaloneLayout",

    # Validation
    "validatorUrl": None,

    # Persist authorization
    "persistAuthorization": True,

    # Request snippets
    "requestSnippetsEnabled": True,
    "requestSnippets": {
        "generators": {
            "curl_bash": {
                "title": "cURL (bash)",
                "syntax": "bash"
            },
            "curl_powershell": {
                "title": "cURL (PowerShell)",
                "syntax": "powershell"
            },
            "curl_cmd": {
                "title": "cURL (CMD)",
                "syntax": "bash"
            }
        },
        "defaultExpanded": True,
        "languagesMask": ["curl_bash", "curl_powershell"]
    }
}


# Custom CSS for Swagger UI
SWAGGER_UI_CUSTOM_CSS = """
/* GreenLang Theme for Swagger UI */

.swagger-ui {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Header customization */
.swagger-ui .topbar {
    background-color: #1a472a;
    padding: 10px 0;
}

.swagger-ui .topbar .download-url-wrapper .select-label span {
    color: #fff;
}

.swagger-ui .topbar a {
    color: #fff;
}

/* Info section */
.swagger-ui .info .title {
    color: #1a472a;
    font-weight: 700;
}

.swagger-ui .info .title small {
    background: #4caf50;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
}

/* Operation tags */
.swagger-ui .opblock-tag {
    color: #1a472a;
    font-size: 18px;
    border-bottom: 2px solid #e8f5e9;
}

/* HTTP method colors */
.swagger-ui .opblock.opblock-get {
    background: rgba(97, 175, 254, 0.1);
    border-color: #61affe;
}

.swagger-ui .opblock.opblock-get .opblock-summary-method {
    background: #61affe;
}

.swagger-ui .opblock.opblock-post {
    background: rgba(73, 204, 144, 0.1);
    border-color: #49cc90;
}

.swagger-ui .opblock.opblock-post .opblock-summary-method {
    background: #49cc90;
}

.swagger-ui .opblock.opblock-put {
    background: rgba(252, 161, 48, 0.1);
    border-color: #fca130;
}

.swagger-ui .opblock.opblock-put .opblock-summary-method {
    background: #fca130;
}

.swagger-ui .opblock.opblock-delete {
    background: rgba(249, 62, 62, 0.1);
    border-color: #f93e3e;
}

.swagger-ui .opblock.opblock-delete .opblock-summary-method {
    background: #f93e3e;
}

/* Authorize button */
.swagger-ui .btn.authorize {
    background-color: #4caf50;
    border-color: #4caf50;
    color: #fff;
}

.swagger-ui .btn.authorize svg {
    fill: #fff;
}

/* Execute button */
.swagger-ui .btn.execute {
    background-color: #1a472a;
    border-color: #1a472a;
}

/* Models section */
.swagger-ui section.models {
    border: 1px solid #e8f5e9;
    border-radius: 8px;
}

.swagger-ui section.models h4 {
    color: #1a472a;
}

/* Response section */
.swagger-ui .responses-inner h4 {
    color: #1a472a;
}

/* Code samples */
.swagger-ui .highlight-code {
    border-radius: 4px;
}

/* Schema table */
.swagger-ui table.model tbody tr td {
    padding: 8px 0;
}

/* Loading bar */
.swagger-ui .loading-container {
    background: #e8f5e9;
}

.swagger-ui .loading-container .loading {
    background-color: #4caf50;
}
"""


# Custom JavaScript for additional functionality
SWAGGER_UI_CUSTOM_JS = """
// GreenLang Custom Swagger UI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add custom logo if element exists
    const logo = document.querySelector('.swagger-ui .topbar-wrapper a');
    if (logo) {
        logo.innerHTML = '<span style="font-size: 24px; font-weight: bold; color: #fff;">🌿 GreenLang API</span>';
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to execute
        if (e.ctrlKey && e.key === 'Enter') {
            const executeBtn = document.querySelector('.try-out__btn');
            if (executeBtn) {
                executeBtn.click();
            }
        }
    });
});
"""


def get_openapi_spec_path() -> Path:
    """Get the path to the OpenAPI specification file."""
    return Path(__file__).parent / "openapi.json"


def load_openapi_spec() -> Dict[str, Any]:
    """Load the OpenAPI specification from file."""
    spec_path = get_openapi_spec_path()
    if spec_path.exists():
        with open(spec_path, "r") as f:
            return json.load(f)
    return {}


def configure_swagger_ui(app: FastAPI) -> None:
    """
    Configure Swagger UI with custom settings and styling.

    Args:
        app: FastAPI application instance
    """

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
        """Serve custom Swagger UI."""
        openapi_url = app.openapi_url or "/openapi.json"

        return get_swagger_ui_html(
            openapi_url=openapi_url,
            title=f"{app.title} - API Documentation",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url="https://greenlang.io/favicon.ico",
        )

    @app.get("/docs/custom", include_in_schema=False)
    async def custom_swagger_ui_with_theme(request: Request) -> HTMLResponse:
        """Serve Swagger UI with custom GreenLang theme."""
        openapi_url = app.openapi_url or "/openapi.json"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{app.title} - API Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <link rel="icon" type="image/png" href="https://greenlang.io/favicon.ico">
    <style>
        {SWAGGER_UI_CUSTOM_CSS}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: "{openapi_url}",
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                {_config_to_js(SWAGGER_UI_CONFIG)}
            }});
            window.ui = ui;
        }};

        {SWAGGER_UI_CUSTOM_JS}
    </script>
</body>
</html>
"""
        return HTMLResponse(content=html)

    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi_json():
        """Serve OpenAPI specification as JSON."""
        spec = load_openapi_spec()
        if not spec:
            # Fall back to FastAPI's generated spec
            return app.openapi()
        return spec


def _config_to_js(config: Dict[str, Any]) -> str:
    """Convert Python config dict to JavaScript object properties."""
    lines = []
    for key, value in config.items():
        if isinstance(value, bool):
            js_value = "true" if value else "false"
        elif isinstance(value, str):
            js_value = f'"{value}"'
        elif value is None:
            js_value = "null"
        elif isinstance(value, dict):
            js_value = json.dumps(value)
        elif isinstance(value, list):
            js_value = json.dumps(value)
        else:
            js_value = str(value)

        lines.append(f'                {key}: {js_value}')

    return ",\n".join(lines)


# Export default config
__all__ = [
    "SWAGGER_UI_CONFIG",
    "SWAGGER_UI_CUSTOM_CSS",
    "configure_swagger_ui",
    "load_openapi_spec",
    "get_openapi_spec_path",
]
