# GreenLang API Versioning

## Overview

GreenLang uses **URI path versioning** with a `v1` prefix for all API
endpoints.  A `v2` generation of application backends exists for compliance
workloads (CBAM, CSRD, VCCI, EUDR, GHG, ISO 14064, SB 253, Taxonomy) and
introduces stricter audit controls and a native execution model.

---

## Versioning Strategy

### URI Path Versioning

All API endpoints include the version in the URL path:

```
/api/v1/{resource}
```

**Why path versioning:**

1. Explicit and visible in every request.
2. Easy to route at the load balancer and API gateway (Kong).
3. Simple to test with cURL, browser, or any HTTP client.
4. Compatible with OpenAPI spec generation.

### Version Prefix Convention

| Version | Status | Base Path | Description |
|---------|--------|-----------|-------------|
| `v1` | **Production** | `/api/v1/` | Current stable API. All CRUD, execution, and management endpoints. |
| `v2` | **Production** | Internal backend dispatch | Application execution backends with native pipelines and audit bundles. v2 endpoints are exposed through the v1 API surface via `/api/v1/apps/{app_id}/run`. |

---

## v1 API Surface

The v1 API is the primary interface for all external consumers.  All v2
application backends are invoked through v1 endpoints.

### Endpoint Structure

```
/api/v1/apps/{app_id}/run          POST   Run an application pipeline
/api/v1/apps/{app_id}/demo-run     POST   Run with demo data
/api/v1/runs                       GET    List pipeline runs
/api/v1/runs/{run_id}/bundle       GET    Download run artifact bundle
/api/v1/runs/{run_id}/artifacts/{path}  GET    Download a specific artifact
/api/v1/auth/token                 POST   Obtain JWT token
/api/v1/auth/refresh               POST   Refresh JWT token
/api/v1/auth/api-keys              POST   Create API key
/api/v1/agents                     GET    List registered agents
/api/v1/agents/{agent_id}          GET    Get agent details
/api/v1/admin/release-train        GET    Release train metadata
/api/v1/admin/connectors           GET    Connector registry status
/api/v1/admin/connectors/health    GET    Connector health checks
/api/v1/shell/chrome-context       GET    Shell UI context for frontend
```

### Supported Application IDs

The following `app_id` values are accepted by `/api/v1/apps/{app_id}/run`:

| App ID | Application | v2 Backend |
|--------|------------|------------|
| `cbam` | CBAM Carbon Border Adjustment | Yes |
| `csrd` | CSRD Sustainability Reporting | Yes |
| `vcci` | VCCI Scope 3 Carbon Accounting | Yes |
| `eudr` | EUDR Deforestation Due Diligence | Yes |
| `ghg` | GHG Protocol Accounting | Yes |
| `iso14064` | ISO 14064 Verification | Yes |
| `sb253` | California SB 253 Reporting | Yes |
| `taxonomy` | EU Taxonomy Alignment | Yes |

---

## v2 Application Backends

The v2 backend system provides **native pipeline execution** with enhanced
audit capabilities.  V2 backends are not exposed as separate `/api/v2/`
endpoints; instead, they are dispatched internally when a v1 run endpoint
is called.

### Architecture

```
Client
  |
  v
/api/v1/apps/cbam/run  (FastAPI endpoint)
  |
  v
v2 Backend Dispatcher (greenlang/v2/backends.py)
  |
  v
Native Pipeline Execution
  |
  v
Audit Bundle Generation
  |
  v
/api/v1/runs/{run_id}/bundle  (artifact download)
```

### V2 App Profiles

Each application has a registered profile in `greenlang/v2/profiles.py`:

```python
V2_APP_PROFILES = {
    "cbam": V2AppProfile(
        app_id="GL-CBAM-APP",
        key="cbam",
        v2_dir=Path("applications/GL-CBAM-APP/v2"),
        command_template="gl run cbam <config.yaml> <imports.csv> <output_dir>",
    ),
    "csrd": V2AppProfile(
        app_id="GL-CSRD-APP",
        key="csrd",
        v2_dir=Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/v2"),
        command_template="gl run csrd <input.csv|json> <output_dir>",
    ),
    # ... (8 total app profiles)
}
```

### V2 Backend Execution

The `run_v2_profile_backend()` function in `greenlang/v2/backends.py`:

1. Resolves the app profile from `V2_APP_PROFILES`.
2. Executes the native pipeline subprocess.
3. Generates an audit bundle with:
   - `audit/run_manifest.json` -- pipeline metadata, status, warnings.
   - `audit/checksums.json` -- SHA-256 hashes of all output artifacts.
4. Returns a `BackendRunResult` with output directory and status.

### V2 Blocked Exit Code

If a v2 backend returns exit code `4`, it indicates the pipeline was
**blocked** (e.g., missing required configuration or compliance gate failure).
The API returns HTTP 422 with diagnostic details.

---

## V1-to-V2 Fallback

The `GL_V1_ALLOW_BACKEND_FALLBACK` environment variable controls whether
the system falls back to v1 backends when v2 execution is unavailable:

| Value | Behavior |
|-------|----------|
| `0` (default) | v2 backend only; error if unavailable |
| `1` | Fall back to v1 backend if v2 fails |

In production, fallback should be disabled (`0`) to ensure all runs go through
the v2 audit pipeline.

---

## API Compatibility Guarantees

### Within a Major Version (v1)

The following are guaranteed within the v1 API:

1. **No breaking changes** to existing endpoint URLs, request formats, or
   response schemas.
2. **Additive changes only** -- new optional fields, new endpoints, new query
   parameters.
3. **Deprecation notices** are provided in response headers and documentation
   at least 90 days before removal.

### Changes That Are NOT Breaking

| Change | Example |
|--------|---------|
| Adding a new endpoint | `GET /api/v1/reports` |
| Adding an optional query parameter | `?include_metadata=true` |
| Adding a new field to a response object | `"data_quality_score": 98.5` |
| Adding a new enum value | New `lifecycle_state` value |
| Adding a new error code | New `GL_CALC_*` error |
| Relaxing a validation constraint | Accepting a wider date range |

### Changes That ARE Breaking

| Change | Mitigation |
|--------|------------|
| Removing an endpoint | Deprecation notice, new version |
| Renaming a field | Deprecation notice, dual-output period |
| Changing a field type | New version |
| Tightening a validation constraint | New version |
| Removing an enum value | Deprecation notice |

---

## Deprecation Policy

When an endpoint, field, or behavior is scheduled for removal:

1. A `Deprecation` header is added to responses:
   ```
   Deprecation: true
   Sunset: Sat, 04 Jul 2026 00:00:00 GMT
   ```

2. The documentation is updated with a deprecation notice.

3. After the sunset date, the deprecated feature returns HTTP 410 Gone.

---

## Request Headers

### Version Negotiation (Informational)

While the primary version is in the URL path, clients can include an
informational version header:

```http
X-GreenLang-API-Version: 2026-04-01
```

This is not currently used for routing but is logged for analytics and
compatibility tracking.

### Required Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes (except excluded paths) | `Bearer <jwt>` or omit if using API key |
| `X-API-Key` | Alternative to Authorization | API key for programmatic access |
| `Content-Type` | Yes (for POST/PUT/PATCH) | `application/json` or `multipart/form-data` |
| `Accept` | Recommended | `application/json` (default) |
| `X-Tenant-ID` | Fallback for tenant context | Tenant ID if not in JWT |

---

## Migration Guide: Integrating with V2 Backends

If you currently use v1 application backends and want to take advantage of
v2 features (audit bundles, SHA-256 checksums, native pipelines):

### Step 1: Verify V2 Availability

Check that v2 backends are registered for your application:

```bash
curl -s "https://api.greenlang.io/api/v1/admin/release-train" \
  -H "Authorization: Bearer $TOKEN" | jq '.v2_apps'
```

### Step 2: No Endpoint Changes Required

V2 backends are invoked through the same `/api/v1/apps/{app_id}/run` endpoint.
No URL changes are needed.

### Step 3: Access Audit Bundles

After a successful run, download the audit bundle:

```bash
# Run the pipeline (same endpoint as before)
RUN_ID=$(curl -s -X POST "https://api.greenlang.io/api/v1/apps/cbam/run" \
  -H "Authorization: Bearer $TOKEN" \
  -F "input_file=@imports.csv" | jq -r '.run_id')

# Download the audit bundle (new v2 artifact)
curl -s "https://api.greenlang.io/api/v1/runs/$RUN_ID/bundle" \
  -H "Authorization: Bearer $TOKEN" -o bundle.zip

# Inspect audit artifacts
unzip -l bundle.zip | grep audit/
# audit/run_manifest.json
# audit/checksums.json
```

### Step 4: Verify Checksums

The audit bundle includes SHA-256 checksums for all output artifacts:

```python
import json
import hashlib
from pathlib import Path

# Load checksums from audit bundle
checksums = json.loads(Path("audit/checksums.json").read_text())

# Verify each artifact
for artifact, expected_hash in checksums.items():
    actual_hash = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"Checksum mismatch for {artifact}"
    print(f"OK: {artifact}")
```

---

## Source Files

| File | Purpose |
|------|---------|
| `greenlang/v2/profiles.py` | V2 app profile registry (8 apps: CBAM, CSRD, VCCI, EUDR, GHG, ISO14064, SB253, Taxonomy) |
| `greenlang/v2/backends.py` | V2 backend execution dispatcher, audit bundle generation |
| `greenlang/v2/conformance.py` | V2 conformance checking and validation |
| `greenlang/v1/backends.py` | V1 legacy backend execution (CBAM, CSRD, VCCI) |
| `cbam-pack-mvp/src/cbam_pack/web/app.py` | FastAPI application with `/api/v1/` route definitions |
