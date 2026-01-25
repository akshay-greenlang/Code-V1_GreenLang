# -*- coding: utf-8 -*-
"""
GraphQL Server Setup
FastAPI-based GraphQL server with Strawberry
"""

from __future__ import annotations
from typing import Optional, Callable, Any
import logging
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager
from greenlang.api.graphql.context import GraphQLContext
from greenlang.api.graphql.resolvers import Query, Mutation
from greenlang.api.graphql.subscriptions import Subscription
from greenlang.api.graphql.complexity import ComplexityValidator, ComplexityConfig
from greenlang.api.graphql.playground import get_playground_html

logger = logging.getLogger(__name__)


# ==============================================================================
# Context Getter
# ==============================================================================

async def get_context(
    request: Request = None,
    websocket: WebSocket = None,
    orchestrator: Orchestrator = None,
    auth_manager: AuthManager = None,
    rbac_manager: RBACManager = None,
) -> GraphQLContext:
    """
    Get GraphQL context from request

    Args:
        request: HTTP request (for queries/mutations)
        websocket: WebSocket (for subscriptions)
        orchestrator: Orchestrator instance
        auth_manager: AuthManager instance
        rbac_manager: RBACManager instance

    Returns:
        GraphQLContext

    Raises:
        PermissionError: If authentication fails
    """
    # Get request object (HTTP or WebSocket)
    req = request or websocket

    if not req:
        raise ValueError("No request or websocket provided")

    # Create context from request
    context = await GraphQLContext.from_request(
        req,
        orchestrator=orchestrator,
        auth_manager=auth_manager,
        rbac_manager=rbac_manager,
    )

    return context


# ==============================================================================
# Validation Extension
# ==============================================================================

class ComplexityValidationExtension:
    """
    Strawberry extension for query complexity validation
    """

    def __init__(self, validator: ComplexityValidator):
        self.validator = validator

    async def on_request_start(self, execution_context):
        """Validate query before execution"""
        try:
            # Get query document and schema
            document = execution_context.query
            schema = execution_context.schema
            variables = execution_context.variables

            # Validate complexity
            result = self.validator.validate(document, schema._strawberry_definition, variables)

            # Log validation result
            logger.info(
                f"Query complexity: {result['complexity']}, "
                f"depth: {result['depth']}"
            )

        except ValueError as e:
            logger.warning(f"Query validation failed: {e}")
            # Re-raise to prevent execution
            raise


# ==============================================================================
# Error Formatter
# ==============================================================================

def format_error(error: Exception, debug: bool = False) -> dict:
    """
    Format GraphQL errors

    Args:
        error: Exception that occurred
        debug: Include debug information

    Returns:
        Formatted error dict
    """
    formatted = {
        "message": str(error),
        "extensions": {
            "code": error.__class__.__name__,
        },
    }

    # Add debug info if enabled
    if debug:
        import traceback

        formatted["extensions"]["exception"] = {
            "stacktrace": traceback.format_exception(
                type(error), error, error.__traceback__
            ),
        }

    # Handle permission errors
    if isinstance(error, PermissionError):
        formatted["extensions"]["code"] = "FORBIDDEN"

    # Handle value errors
    elif isinstance(error, ValueError):
        formatted["extensions"]["code"] = "BAD_REQUEST"

    return formatted


# ==============================================================================
# GraphQL Application Factory
# ==============================================================================

def create_graphql_app(
    orchestrator: Orchestrator,
    auth_manager: AuthManager,
    rbac_manager: RBACManager,
    enable_playground: bool = True,
    enable_introspection: bool = True,
    enable_subscriptions: bool = True,
    complexity_config: Optional[ComplexityConfig] = None,
    cors_origins: Optional[list] = None,
    debug: bool = False,
) -> FastAPI:
    """
    Create FastAPI GraphQL application

    Args:
        orchestrator: Orchestrator instance
        auth_manager: AuthManager instance
        rbac_manager: RBACManager instance
        enable_playground: Enable GraphQL Playground UI
        enable_introspection: Enable schema introspection
        enable_subscriptions: Enable WebSocket subscriptions
        complexity_config: Query complexity configuration
        cors_origins: Allowed CORS origins
        debug: Enable debug mode

    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="GreenLang GraphQL API",
        description="Comprehensive GraphQL API for GreenLang",
        version="1.0.0",
        debug=debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Create GraphQL schema
    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription if enable_subscriptions else None,
    )

    # Create complexity validator
    validator = ComplexityValidator(complexity_config)

    # Create context getter with services
    async def context_getter(
        request: Request = None,
        websocket: WebSocket = None,
    ) -> GraphQLContext:
        return await get_context(
            request=request,
            websocket=websocket,
            orchestrator=orchestrator,
            auth_manager=auth_manager,
            rbac_manager=rbac_manager,
        )

    # Create GraphQL router
    graphql_router = GraphQLRouter(
        schema,
        context_getter=context_getter,
        graphiql=False,  # We use custom playground
        allow_queries_via_get=True,
    )

    # Add GraphQL endpoint
    app.include_router(graphql_router, prefix="/graphql")

    # Add playground endpoint
    if enable_playground:

        @app.get("/", response_class=HTMLResponse)
        async def playground():
            """GraphQL Playground UI"""
            return get_playground_html(
                endpoint="/graphql",
                subscription_endpoint="/graphql" if enable_subscriptions else None,
            )

        @app.get("/playground", response_class=HTMLResponse)
        async def playground_alt():
            """GraphQL Playground UI (alternative route)"""
            return get_playground_html(
                endpoint="/graphql",
                subscription_endpoint="/graphql" if enable_subscriptions else None,
            )

    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "greenlang-graphql-api",
            "version": "1.0.0",
        }

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus-style metrics endpoint"""
        from greenlang.api.graphql.subscriptions import subscription_manager

        subscribers = subscription_manager.get_subscriber_counts()

        return {
            "agents_total": len(orchestrator.list_agents()),
            "workflows_total": len(orchestrator.list_workflows()),
            "executions_total": len(orchestrator.get_execution_history()),
            "subscriptions_active": sum(subscribers.values()),
            "subscriptions_by_type": subscribers,
        }

    # Add startup event
    @app.on_event("startup")
    async def startup():
        """Startup event handler"""
        logger.info("Starting GreenLang GraphQL API...")
        logger.info(f"Playground enabled: {enable_playground}")
        logger.info(f"Subscriptions enabled: {enable_subscriptions}")
        logger.info(f"Introspection enabled: {enable_introspection}")
        logger.info(f"Debug mode: {debug}")

    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown():
        """Shutdown event handler"""
        logger.info("Shutting down GreenLang GraphQL API...")

    # Add error handlers
    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        """Handle permission errors"""
        return JSONResponse(
            status_code=403,
            content={
                "error": "Forbidden",
                "message": str(exc),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle validation errors"""
        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "message": str(exc),
            },
        )

    logger.info("GreenLang GraphQL API created successfully")

    return app


# ==============================================================================
# Development Server
# ==============================================================================

def run_dev_server(
    orchestrator: Orchestrator,
    auth_manager: AuthManager,
    rbac_manager: RBACManager,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
):
    """
    Run development server

    Args:
        orchestrator: Orchestrator instance
        auth_manager: AuthManager instance
        rbac_manager: RBACManager instance
        host: Server host
        port: Server port
        reload: Enable auto-reload
    """
    import uvicorn

    app = create_graphql_app(
        orchestrator=orchestrator,
        auth_manager=auth_manager,
        rbac_manager=rbac_manager,
        enable_playground=True,
        enable_introspection=True,
        enable_subscriptions=True,
        debug=True,
    )

    logger.info(f"Starting development server on {host}:{port}")
    logger.info(f"GraphQL endpoint: http://{host}:{port}/graphql")
    logger.info(f"Playground: http://{host}:{port}/playground")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ==============================================================================
# Production Server
# ==============================================================================

def create_production_app(
    orchestrator: Orchestrator,
    auth_manager: AuthManager,
    rbac_manager: RBACManager,
    complexity_config: Optional[ComplexityConfig] = None,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create production GraphQL application

    Args:
        orchestrator: Orchestrator instance
        auth_manager: AuthManager instance
        rbac_manager: RBACManager instance
        complexity_config: Query complexity configuration
        cors_origins: Allowed CORS origins

    Returns:
        Production-ready FastAPI application
    """
    return create_graphql_app(
        orchestrator=orchestrator,
        auth_manager=auth_manager,
        rbac_manager=rbac_manager,
        enable_playground=False,  # Disable in production
        enable_introspection=False,  # Disable in production
        enable_subscriptions=True,
        complexity_config=complexity_config,
        cors_origins=cors_origins or [],
        debug=False,
    )
