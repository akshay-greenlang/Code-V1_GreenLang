-- =============================================================================
-- GreenLang Climate OS - Tenant Isolation Plugin (schema.lua)
-- =============================================================================
-- PRD: INFRA-006 / SEC-003 Multi-Tenant Data Isolation
--
-- Defines the configuration schema for the gl-tenant-isolation plugin.
-- Kong validates plugin configuration against this schema before activating
-- the plugin on any service, route, or globally.
--
-- Configuration fields:
--
--   require_tenant (boolean, default: false)
--     When true, requests without a valid tenant_id JWT claim are rejected
--     with HTTP 403. Enable this on all authenticated routes in production
--     to enforce strict tenant isolation.
--
--   expose_header (boolean, default: true)
--     When true, the X-Tenant-ID header is included in the response so
--     that API consumers can verify tenant context. Disable in environments
--     where tenant IDs should not be exposed to end clients.
--
--   tenant_claim (string, default: "tenant_id")
--     The JWT claim name from which the tenant identifier is extracted.
--     Override this if your identity provider uses a different claim name
--     (e.g., "org_id", "account_id").
--
-- Version: 1.0.0
-- =============================================================================

local typedefs = require "kong.db.schema.typedefs"

return {
  name = "gl-tenant-isolation",
  fields = {
    {
      -- Restrict to HTTP/HTTPS protocols only; gRPC tenant isolation
      -- is handled separately in the gRPC service mesh layer.
      protocols = typedefs.protocols_http,
    },
    {
      -- This plugin operates on the request context, not on a specific
      -- consumer. Tenant identity is derived from the JWT, not from the
      -- Kong consumer entity.
      consumer = typedefs.no_consumer,
    },
    {
      config = {
        type = "record",
        fields = {
          {
            -- When true, requests missing the tenant_id claim are rejected
            -- with HTTP 403. Set to true for all production authenticated
            -- routes. Set to false for internal/service-to-service routes
            -- where tenant context is optional.
            require_tenant = {
              type = "boolean",
              default = false,
              description = "Reject requests that lack a tenant_id claim in the JWT.",
            },
          },
          {
            -- Controls whether the X-Tenant-ID header appears in the
            -- HTTP response. Useful for debugging and client-side correlation.
            -- Disable if tenant identifiers are considered sensitive.
            expose_header = {
              type = "boolean",
              default = true,
              description = "Include X-Tenant-ID in the response headers.",
            },
          },
          {
            -- The name of the JWT claim containing the tenant identifier.
            -- Must match the claim name issued by your identity provider.
            -- Common alternatives: "org_id", "account_id", "tid".
            tenant_claim = {
              type = "string",
              default = "tenant_id",
              description = "JWT claim name for the tenant identifier.",
            },
          },
        },
      },
    },
  },
}
