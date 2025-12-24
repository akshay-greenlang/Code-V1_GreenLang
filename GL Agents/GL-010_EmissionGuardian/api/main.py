# -*- coding: utf-8 -*-
"""GL-010 EmissionsGuardian - REST API Application"""

import os
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .routes_cems import router as cems_router
from .routes_compliance import router as compliance_router
from .routes_rata import router as rata_router
from .routes_fugitive import router as fugitive_router
from .routes_trading import router as trading_router
from .routes_reports import router as reports_router
from .schemas import HealthCheckResponse, ReadinessCheckResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "GL-010 EmissionsGuardian API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Production REST API for emissions compliance monitoring."
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
    yield
    logger.info("Shutting down EmissionsGuardian API")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000", "https://*.greenlang.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.greenlang.io", "localhost", "127.0.0.1"])

rate_limit_store: dict = {}
RATE_LIMIT_RPM = 1000


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Callable):
    client_id = request.client.host if request.client else "unknown"
    current_minute = int(time.time() / 60)
    key = f"{client_id}:{current_minute}"
    count = rate_limit_store.get(key, 0)
    if count >= RATE_LIMIT_RPM:
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    rate_limit_store[key] = count + 1
    return await call_next(request)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Callable):
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(time.time() - start_time)
    return response


app.include_router(cems_router, prefix="/api/v1")
app.include_router(compliance_router, prefix="/api/v1")
app.include_router(rata_router, prefix="/api/v1")
app.include_router(fugitive_router, prefix="/api/v1")
app.include_router(trading_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    return HealthCheckResponse(status="healthy", version=APP_VERSION, 
                                components={"api": "healthy"}, uptime_seconds=time.time() - START_TIME)


@app.get("/ready", response_model=ReadinessCheckResponse, tags=["System"])
async def readiness_check():
    return ReadinessCheckResponse(ready=True, checks={"api": True}, message="All systems operational")


@app.get("/", tags=["System"])
async def root():
    return {"name": APP_TITLE, "version": APP_VERSION, "status": "operational", "docs": "/api/docs"}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(title=APP_TITLE, version=APP_VERSION, description=APP_DESCRIPTION, routes=app.routes)
    openapi_schema["components"]["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}}
    openapi_schema["security"] = [{"bearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "message": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": "An internal error occurred"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
