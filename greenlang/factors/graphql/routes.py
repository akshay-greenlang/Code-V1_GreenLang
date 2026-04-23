# -*- coding: utf-8 -*-
"""FastAPI router mounting the GraphQL schema at /v1/graphql.

The router sits behind the same middleware stack as the REST routes
(AuthMetering, RateLimit, LicensingGuard, SignedReceipts). We pass the
raw :class:`starlette.requests.Request` into the GraphQL execution
context so resolvers can honour tier enforcement and licensing class
gates picked up by middleware.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

graphql_router = APIRouter(prefix="/v1", tags=["factors-graphql"])


def _dev_mode() -> bool:
    env = (os.getenv("APP_ENV") or os.getenv("GL_ENV") or "dev").lower()
    return env not in {"staging", "production", "prod"}


def _get_schema():
    from .schema import _HAS_STRAWBERRY, build_schema

    if not _HAS_STRAWBERRY:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "graphql_unavailable",
                "message": (
                    "GraphQL requires 'strawberry-graphql'. "
                    "Install it with `pip install strawberry-graphql[fastapi]`."
                ),
            },
        )
    return build_schema()


def _check_tier(request: Request) -> None:
    """GraphQL requires an authed caller at least at community tier.

    This is a thin dependency so unauthed requests to /v1/graphql
    return 401 rather than leaking schema introspection. It does not
    replace the licensing-guard middleware.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        # Mirror REST behaviour: the auth middleware assigns a
        # community-anon user in dev. Real unauthed callers in prod
        # will have request.state.user unset — reject them.
        if not _dev_mode():
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "auth_required",
                    "message": "/v1/graphql requires a valid API key or JWT.",
                },
            )


@graphql_router.post("/graphql")
async def graphql_execute(request: Request) -> Dict[str, Any]:
    """Execute a GraphQL query.

    Request body::

        {
          "query": "{ methodPacks { packId } }",
          "variables": {...},
          "operationName": null
        }
    """
    _check_tier(request)
    schema = _get_schema()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    query = (body or {}).get("query")
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Missing 'query' field")
    variables = body.get("variables") or {}
    operation_name = body.get("operationName")

    context = {"request": request}
    try:
        result = await schema.execute(
            query,
            variable_values=variables,
            context_value=context,
            operation_name=operation_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("GraphQL execution error")
        return {
            "data": None,
            "errors": [{"message": f"execution_error: {exc}"}],
        }

    response: Dict[str, Any] = {"data": None}
    if result.data is not None:
        try:
            # strawberry's execute returns python dataclass instances;
            # coerce to JSON via the schema's built-in serializer.
            response["data"] = json.loads(json.dumps(result.data, default=_serialize_default))
        except Exception:
            response["data"] = result.data
    if getattr(result, "errors", None):
        response["errors"] = [
            {"message": str(getattr(e, "message", e))} for e in result.errors
        ]
    # Fold any GraphQL error into a 400 so the rate-limit header path
    # and signed-receipt middleware treat this as a client error.
    if response.get("errors") and response.get("data") is None:
        # Do not raise — GraphQL semantics want a 200 with errors[].
        pass
    return response


def _serialize_default(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


@graphql_router.get("/graphql")
def graphql_ui(request: Request) -> Any:
    """GraphiQL UI (dev-only)."""
    _check_tier(request)
    if not _dev_mode():
        raise HTTPException(
            status_code=404,
            detail="GraphiQL UI is disabled in staging/production.",
        )
    from fastapi.responses import HTMLResponse
    html = """<!doctype html><html><head>
<title>GreenLang Factors GraphiQL</title>
<link rel="stylesheet" href="https://unpkg.com/graphiql/graphiql.min.css" />
</head><body style="margin:0;height:100vh">
<div id="graphiql" style="height:100vh"></div>
<script crossorigin src="https://unpkg.com/react/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom/umd/react-dom.production.min.js"></script>
<script crossorigin src="https://unpkg.com/graphiql/graphiql.min.js"></script>
<script>
const fetcher = GraphiQL.createFetcher({ url: '/v1/graphql' });
ReactDOM.render(React.createElement(GraphiQL, { fetcher }), document.getElementById('graphiql'));
</script></body></html>"""
    return HTMLResponse(html)


@graphql_router.get("/graphql/schema")
def graphql_schema_sdl(request: Request) -> Dict[str, str]:
    """Expose the SDL for codegen tooling."""
    _check_tier(request)
    schema = _get_schema()
    try:
        sdl = str(schema)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"schema dump failed: {exc}")
    return {"sdl": sdl}


__all__ = ["graphql_router"]
