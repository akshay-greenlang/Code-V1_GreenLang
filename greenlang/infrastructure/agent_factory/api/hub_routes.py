# -*- coding: utf-8 -*-
"""
Hub Routes - Agent Hub package registry endpoints.

Router prefix: /api/v1/factory/hub

Endpoints:
    GET    /packages                      - Search packages.
    GET    /packages/{key}                - Get package details.
    GET    /packages/{key}/versions       - List package versions.
    POST   /packages                      - Publish a package (multipart).
    DELETE /packages/{key}/{version}      - Unpublish a version.
    GET    /packages/{key}/{version}/download - Download a package.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/factory/hub", tags=["Agent Hub"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PackageVersion(BaseModel):
    """Metadata for a single package version."""

    version: str
    published_at: str
    checksum: str
    size_bytes: int
    downloads: int = 0
    tag: Optional[str] = None


class PackageSummary(BaseModel):
    """Summary of a package in search results."""

    package_key: str
    latest_version: str
    description: str
    agent_type: str
    tags: List[str] = Field(default_factory=list)
    total_downloads: int = 0
    published_at: str


class PackageDetail(BaseModel):
    """Full package details."""

    package_key: str
    latest_version: str
    description: str
    agent_type: str
    author: str
    license: str
    tags: List[str] = Field(default_factory=list)
    total_downloads: int = 0
    versions: List[PackageVersion]
    created_at: str
    updated_at: str


class PackageSearchResponse(BaseModel):
    """Search results."""

    packages: List[PackageSummary]
    total: int
    page: int
    page_size: int


class VersionListResponse(BaseModel):
    """Version list for a package."""

    package_key: str
    versions: List[PackageVersion]
    total: int


class PublishResponse(BaseModel):
    """Result of publishing a package."""

    package_key: str
    version: str
    checksum: str
    published_at: str
    download_url: str


class UnpublishResponse(BaseModel):
    """Result of unpublishing a package version."""

    package_key: str
    version: str
    status: str
    removed_at: str


# ---------------------------------------------------------------------------
# In-memory store (replaced by DB + S3 in production)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_package_store: Dict[str, Dict[str, Any]] = {}


def _seed_hub_data() -> None:
    """Populate demo packages if store is empty."""
    if _package_store:
        return

    now = _now_iso()
    demo_packages = [
        {
            "package_key": "carbon-calc",
            "latest_version": "2.1.0",
            "description": "Scope 1-3 carbon emissions calculator agent.",
            "agent_type": "deterministic",
            "author": "GreenLang Platform Team",
            "license": "Apache-2.0",
            "tags": ["carbon", "emissions", "scope1", "scope2", "scope3"],
            "total_downloads": 5420,
            "created_at": "2025-06-01T00:00:00Z",
            "updated_at": now,
            "versions": [
                {"version": "2.1.0", "published_at": now, "checksum": "sha256:abc1", "size_bytes": 45_000, "downloads": 3200, "tag": "latest"},
                {"version": "2.0.0", "published_at": "2025-11-01T00:00:00Z", "checksum": "sha256:abc0", "size_bytes": 42_000, "downloads": 2220, "tag": None},
            ],
        },
        {
            "package_key": "eudr-compliance",
            "latest_version": "1.3.0",
            "description": "EU Deforestation Regulation compliance checker.",
            "agent_type": "reasoning",
            "author": "GreenLang Platform Team",
            "license": "Apache-2.0",
            "tags": ["eudr", "deforestation", "compliance", "eu"],
            "total_downloads": 1890,
            "created_at": "2025-08-15T00:00:00Z",
            "updated_at": now,
            "versions": [
                {"version": "1.3.0", "published_at": now, "checksum": "sha256:def1", "size_bytes": 62_000, "downloads": 890, "tag": "latest"},
                {"version": "1.2.0", "published_at": "2025-12-01T00:00:00Z", "checksum": "sha256:def0", "size_bytes": 58_000, "downloads": 1000, "tag": None},
            ],
        },
        {
            "package_key": "csrd-disclosure",
            "latest_version": "0.9.1",
            "description": "CSRD double-materiality disclosure agent.",
            "agent_type": "insight",
            "author": "GreenLang Platform Team",
            "license": "Apache-2.0",
            "tags": ["csrd", "disclosure", "materiality", "esrs"],
            "total_downloads": 745,
            "created_at": "2025-10-01T00:00:00Z",
            "updated_at": now,
            "versions": [
                {"version": "0.9.1", "published_at": now, "checksum": "sha256:ghi1", "size_bytes": 55_000, "downloads": 745, "tag": "beta"},
            ],
        },
    ]

    for pkg in demo_packages:
        _package_store[pkg["package_key"]] = pkg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/packages", response_model=PackageSearchResponse)
async def search_packages(
    query: Optional[str] = Query(None, description="Search query string."),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by."),
    agent_type: Optional[str] = Query(None, description="Filter by agent type."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> PackageSearchResponse:
    """Search the Agent Hub for packages."""
    _seed_hub_data()

    results = list(_package_store.values())

    if query:
        q = query.lower()
        results = [
            p for p in results
            if q in p["package_key"].lower() or q in p["description"].lower()
        ]

    if tags:
        tag_set = {t.strip().lower() for t in tags.split(",")}
        results = [
            p for p in results
            if tag_set.intersection(t.lower() for t in p.get("tags", []))
        ]

    if agent_type:
        results = [p for p in results if p["agent_type"] == agent_type]

    total = len(results)
    start = (page - 1) * page_size
    page_items = results[start : start + page_size]

    return PackageSearchResponse(
        packages=[
            PackageSummary(
                package_key=p["package_key"],
                latest_version=p["latest_version"],
                description=p["description"],
                agent_type=p["agent_type"],
                tags=p.get("tags", []),
                total_downloads=p.get("total_downloads", 0),
                published_at=p.get("updated_at", ""),
            )
            for p in page_items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/packages/{key}", response_model=PackageDetail)
async def get_package(key: str) -> PackageDetail:
    """Get full details for a package."""
    _seed_hub_data()

    pkg = _package_store.get(key)
    if pkg is None:
        raise HTTPException(status_code=404, detail=f"Package '{key}' not found.")

    return PackageDetail(
        package_key=pkg["package_key"],
        latest_version=pkg["latest_version"],
        description=pkg["description"],
        agent_type=pkg["agent_type"],
        author=pkg.get("author", ""),
        license=pkg.get("license", ""),
        tags=pkg.get("tags", []),
        total_downloads=pkg.get("total_downloads", 0),
        versions=[PackageVersion(**v) for v in pkg["versions"]],
        created_at=pkg.get("created_at", ""),
        updated_at=pkg.get("updated_at", ""),
    )


@router.get("/packages/{key}/versions", response_model=VersionListResponse)
async def list_versions(key: str) -> VersionListResponse:
    """List all versions for a package."""
    _seed_hub_data()

    pkg = _package_store.get(key)
    if pkg is None:
        raise HTTPException(status_code=404, detail=f"Package '{key}' not found.")

    versions = [PackageVersion(**v) for v in pkg["versions"]]
    return VersionListResponse(
        package_key=key,
        versions=versions,
        total=len(versions),
    )


@router.post("/packages", response_model=PublishResponse, status_code=201)
async def publish_package(
    package: UploadFile = File(..., description="The .glpack archive."),
    version: str = Form(..., description="Semantic version."),
    checksum: str = Form("", description="SHA-256 checksum for validation."),
    tag: Optional[str] = Form(None, description="Version tag (e.g. latest, beta)."),
) -> PublishResponse:
    """Publish a new package or version to the Agent Hub."""
    _seed_hub_data()

    # Derive package key from filename
    filename = package.filename or "unknown.glpack"
    package_key = filename.replace(".glpack", "").rsplit("-", 1)[0]
    now = _now_iso()

    # Read file content for checksum validation
    content = await package.read()
    size_bytes = len(content)

    import hashlib
    computed_checksum = hashlib.sha256(content).hexdigest()
    if checksum and checksum != computed_checksum:
        raise HTTPException(
            status_code=400,
            detail=f"Checksum mismatch: expected {checksum}, got {computed_checksum}",
        )

    # Upsert package
    if package_key not in _package_store:
        _package_store[package_key] = {
            "package_key": package_key,
            "latest_version": version,
            "description": "",
            "agent_type": "deterministic",
            "author": "",
            "license": "",
            "tags": [],
            "total_downloads": 0,
            "created_at": now,
            "updated_at": now,
            "versions": [],
        }

    pkg = _package_store[package_key]
    pkg["latest_version"] = version
    pkg["updated_at"] = now
    pkg["versions"].append({
        "version": version,
        "published_at": now,
        "checksum": f"sha256:{computed_checksum[:12]}",
        "size_bytes": size_bytes,
        "downloads": 0,
        "tag": tag,
    })

    download_url = f"/api/v1/factory/hub/packages/{package_key}/{version}/download"
    logger.info("Package published: %s v%s (%d bytes)", package_key, version, size_bytes)

    return PublishResponse(
        package_key=package_key,
        version=version,
        checksum=computed_checksum,
        published_at=now,
        download_url=download_url,
    )


@router.delete("/packages/{key}/{version}", response_model=UnpublishResponse)
async def unpublish_package(key: str, version: str) -> UnpublishResponse:
    """Remove a specific package version from the Hub."""
    _seed_hub_data()

    pkg = _package_store.get(key)
    if pkg is None:
        raise HTTPException(status_code=404, detail=f"Package '{key}' not found.")

    versions = pkg["versions"]
    original_count = len(versions)
    pkg["versions"] = [v for v in versions if v["version"] != version]

    if len(pkg["versions"]) == original_count:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found for '{key}'.")

    # Update latest_version if we removed it
    if pkg["latest_version"] == version and pkg["versions"]:
        pkg["latest_version"] = pkg["versions"][-1]["version"]

    now = _now_iso()
    pkg["updated_at"] = now

    logger.info("Package unpublished: %s v%s", key, version)

    return UnpublishResponse(
        package_key=key,
        version=version,
        status="removed",
        removed_at=now,
    )


@router.get("/packages/{key}/{version}/download")
async def download_package(key: str, version: str) -> StreamingResponse:
    """Download a package archive.

    In production this returns a pre-signed S3 URL or streams the
    archive from object storage.  This stub returns a placeholder.
    """
    _seed_hub_data()

    pkg = _package_store.get(key)
    if pkg is None:
        raise HTTPException(status_code=404, detail=f"Package '{key}' not found.")

    version_exists = any(v["version"] == version for v in pkg["versions"])
    if not version_exists:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    # Increment download counter
    for v in pkg["versions"]:
        if v["version"] == version:
            v["downloads"] = v.get("downloads", 0) + 1
    pkg["total_downloads"] = sum(v.get("downloads", 0) for v in pkg["versions"])

    # In production: stream from S3. Stub returns a placeholder.
    async def _placeholder_stream():
        yield b"# Placeholder .glpack content\n"
        yield f"# {key} v{version}\n".encode()

    return StreamingResponse(
        _placeholder_stream(),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename={key}-{version}.glpack",
        },
    )
