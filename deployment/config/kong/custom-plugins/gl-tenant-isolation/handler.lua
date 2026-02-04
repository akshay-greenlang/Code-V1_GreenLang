-- =============================================================================
-- GreenLang Climate OS - Tenant Isolation Plugin (handler.lua)
-- =============================================================================
-- PRD: INFRA-006 / SEC-003 Multi-Tenant Data Isolation
--
-- This custom Kong plugin enforces tenant-level isolation at the API gateway
-- layer. It extracts the tenant identifier from authenticated JWT claims and
-- propagates it as a request header to upstream services. This ensures that
-- every authenticated request carries an authoritative tenant context that
-- backend services can trust without re-parsing the JWT.
--
-- Execution phases:
--   access()        - Extract tenant_id from JWT, inject upstream header,
--                     optionally reject requests missing tenant context.
--   header_filter() - Optionally expose tenant_id in the response for
--                     client-side correlation and debugging.
--   log()           - Attach tenant context to the log serializer for
--                     centralized audit trail enrichment.
--
-- Priority: 900 (runs after JWT authentication at priority 1005, but before
-- most other plugins so upstream headers are set early).
--
-- Version: 1.0.0
-- =============================================================================

local kong = kong

local TenantIsolation = {
  PRIORITY = 900,
  VERSION = "1.0.0",
}

-- ---------------------------------------------------------------------------
-- access() - Runs on every proxied request during the access phase.
-- ---------------------------------------------------------------------------
-- Reads the tenant_id claim from the JWT payload (set by the jwt plugin in
-- kong.ctx.shared.authenticated_jwt_claims). If the claim is present, the
-- plugin injects it as X-Tenant-ID into the upstream request headers and
-- stores it in the shared plugin context for downstream phases.
--
-- When conf.require_tenant is true, requests without a valid tenant_id claim
-- are rejected with HTTP 403. This mode should be enabled on all
-- authenticated routes in production to prevent data leakage between tenants.
-- ---------------------------------------------------------------------------
function TenantIsolation:access(conf)
  -- Retrieve the claim name from plugin configuration (default: "tenant_id")
  local claim_name = conf.tenant_claim or "tenant_id"

  -- The jwt plugin stores decoded claims in shared context
  local jwt_claims = kong.ctx.shared.authenticated_jwt_claims

  if jwt_claims and jwt_claims[claim_name] then
    local tenant_id = jwt_claims[claim_name]

    -- Validate tenant_id is a non-empty string to prevent header injection
    if type(tenant_id) ~= "string" or tenant_id == "" then
      kong.log.warn(
        "[gl-tenant-isolation] Invalid tenant_id type or empty value; ",
        "claim_name=", claim_name,
        " type=", type(tenant_id)
      )

      if conf.require_tenant then
        return kong.response.exit(403, {
          message = "Tenant identification required",
          error = "invalid_tenant_id"
        })
      end

      return
    end

    -- Sanitize: strip whitespace and limit length to prevent abuse
    tenant_id = tenant_id:match("^%s*(.-)%s*$")
    if #tenant_id > 128 then
      kong.log.warn(
        "[gl-tenant-isolation] Tenant ID exceeds maximum length (128); ",
        "length=", #tenant_id
      )
      return kong.response.exit(400, {
        message = "Invalid tenant identifier",
        error = "tenant_id_too_long"
      })
    end

    -- Inject the authoritative tenant header into the upstream request.
    -- This header MUST be trusted by backend services; Kong strips any
    -- client-supplied X-Tenant-ID via the request-transformer plugin.
    kong.service.request.set_header("X-Tenant-ID", tenant_id)

    -- Store in shared context for use in header_filter and log phases
    kong.ctx.shared.tenant_id = tenant_id

    kong.log.debug(
      "[gl-tenant-isolation] Tenant isolation applied; ",
      "tenant_id=", tenant_id
    )

  elseif conf.require_tenant then
    -- No JWT claims or missing tenant_id claim on a route that requires it
    kong.log.warn(
      "[gl-tenant-isolation] Missing tenant_id claim; ",
      "claim_name=", claim_name,
      " route=", kong.request.get_path()
    )

    return kong.response.exit(403, {
      message = "Tenant identification required",
      error = "missing_tenant_id"
    })
  end
end

-- ---------------------------------------------------------------------------
-- header_filter() - Runs during response header construction.
-- ---------------------------------------------------------------------------
-- Optionally adds the X-Tenant-ID header to the response so that API
-- consumers can correlate responses with their tenant context. This is
-- controlled by conf.expose_header (default: true).
-- ---------------------------------------------------------------------------
function TenantIsolation:header_filter(conf)
  local tenant_id = kong.ctx.shared.tenant_id

  if tenant_id and conf.expose_header then
    kong.response.set_header("X-Tenant-ID", tenant_id)
  end
end

-- ---------------------------------------------------------------------------
-- log() - Runs during the log phase after the response has been sent.
-- ---------------------------------------------------------------------------
-- Attaches the tenant_id to the log serializer so that all log entries
-- (shipped via http-log, file-log, or any other logging plugin) include
-- tenant context for audit and compliance purposes.
-- ---------------------------------------------------------------------------
function TenantIsolation:log(conf)
  local tenant_id = kong.ctx.shared.tenant_id

  if tenant_id then
    -- Attach to the Nginx log serializer for structured logging
    local log_data = kong.log.serialize()
    if log_data then
      log_data.tenant_id = tenant_id
    end
  end
end

return TenantIsolation
