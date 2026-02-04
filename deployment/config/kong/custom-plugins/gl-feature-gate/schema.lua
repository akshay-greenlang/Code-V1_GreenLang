-- =============================================================================
-- GreenLang Climate OS - Feature Gate Plugin (schema.lua)
-- =============================================================================
-- PRD: INFRA-008 Feature Flags System
--
-- Defines the configuration schema for the gl-feature-gate plugin. Kong
-- validates plugin configuration against this schema before activating the
-- plugin on any service, route, or globally.
--
-- Configuration fields:
--
--   flag_key (string, required)
--     The feature flag key to evaluate. This maps to the Redis key pattern
--     ff:{environment}:flag:{flag_key}. Use descriptive names such as
--     "enable_new_calculation_engine" or "route:api-agents".
--
--   redis_host (string, default: "redis-master.greenlang.svc.cluster.local")
--     Hostname of the Redis instance storing feature flag state. In
--     production, this points to the ElastiCache Redis cluster.
--
--   redis_port (integer, default: 6379)
--     Redis port number.
--
--   redis_database (integer, default: 3)
--     Redis database number. Database 3 is reserved for Kong plugins
--     (rate limiting, feature flags) to avoid collision with application
--     caches on database 0-2.
--
--   redis_timeout (integer, default: 500)
--     Connection and command timeout in milliseconds. Set low to avoid
--     blocking the request pipeline if Redis is unresponsive.
--
--   redis_password (string, optional)
--     Redis AUTH password. Leave empty if Redis does not require
--     authentication (e.g., within the K8s network with NetworkPolicy).
--
--   environment (string, default: "prod")
--     Environment name used as a namespace prefix in Redis keys. Allows
--     the same Redis cluster to serve multiple environments.
--
--   default_on_error (boolean, default: true)
--     Fail-open behavior: when Redis is unavailable or the flag key is
--     missing, this value determines whether traffic is allowed (true)
--     or blocked (false). Production default is true (fail-open) to
--     prevent Redis outages from causing cascading service disruptions.
--
--   cache_ttl (integer, default: 30)
--     Seconds to cache the flag evaluation result in the nginx shared
--     dictionary. Reduces Redis round-trips on high-traffic routes.
--     Set to 0 to disable caching (every request hits Redis).
--
--   check_tenant_override (boolean, default: false)
--     When true, after evaluating the global flag, the plugin also
--     checks for a tenant-specific override at the Redis key
--     ff:{environment}:override:{flag_key}:{tenant_id}. The tenant ID
--     is read from X-Tenant-ID header (set by gl-tenant-isolation).
--
--   response_status (integer, default: 503)
--     HTTP status code returned when a feature flag is disabled.
--     503 (Service Unavailable) signals a temporary condition with
--     the Retry-After header.
--
--   retry_after (integer, default: 300)
--     Value for the Retry-After response header in seconds. Tells
--     clients when to retry the request. Default 5 minutes.
--
-- Version: 1.0.0
-- =============================================================================

local typedefs = require "kong.db.schema.typedefs"

return {
  name = "gl-feature-gate",
  fields = {
    {
      -- Restrict to HTTP/HTTPS protocols; feature gating applies only to
      -- synchronous request/response flows. gRPC and WebSocket feature
      -- gating is handled at the application layer.
      protocols = typedefs.protocols_http,
    },
    {
      -- This plugin operates at the route level, not per-consumer.
      -- Feature flags apply to all consumers accessing a route equally
      -- (tenant overrides are handled separately via X-Tenant-ID).
      consumer = typedefs.no_consumer,
    },
    {
      config = {
        type = "record",
        fields = {
          {
            -- The feature flag key to evaluate against Redis. This is the
            -- only required field. All other fields have sensible defaults
            -- for the GreenLang production environment.
            flag_key = {
              type = "string",
              required = true,
              description = "Feature flag key to evaluate (e.g., 'enable_new_calculation_engine').",
            },
          },
          {
            -- Redis host defaults to the GreenLang in-cluster Redis master.
            -- Override for external Redis or non-standard service names.
            redis_host = {
              type = "string",
              default = "redis-master.greenlang.svc.cluster.local",
              description = "Redis host for feature flag storage.",
            },
          },
          {
            redis_port = {
              type = "integer",
              default = 6379,
              between = { 1, 65535 },
              description = "Redis port number.",
            },
          },
          {
            -- Database 3 is reserved for Kong plugins across the GreenLang
            -- platform. Databases 0-2 are used by the application layer.
            redis_database = {
              type = "integer",
              default = 3,
              between = { 0, 15 },
              description = "Redis database number (0-15). Database 3 is reserved for Kong.",
            },
          },
          {
            -- Low timeout to prevent request pipeline blocking. Redis
            -- operations should complete in <5ms on a healthy cluster;
            -- 500ms allows for transient network jitter.
            redis_timeout = {
              type = "integer",
              default = 500,
              between = { 100, 10000 },
              description = "Redis connection and command timeout in milliseconds.",
            },
          },
          {
            -- Optional Redis AUTH password. Injected via Kong environment
            -- variable or Kubernetes secret in production.
            redis_password = {
              type = "string",
              default = nil,
              description = "Redis AUTH password (optional).",
            },
          },
          {
            -- Environment prefix for Redis key namespacing. Allows a single
            -- Redis cluster to serve dev, staging, and prod flags.
            environment = {
              type = "string",
              default = "prod",
              one_of = { "dev", "staging", "prod" },
              description = "Environment name for Redis key prefix.",
            },
          },
          {
            -- Fail-open (true) or fail-closed (false) when Redis is down.
            -- Production default is fail-open to prevent Redis outages
            -- from cascading into full service unavailability.
            default_on_error = {
              type = "boolean",
              default = true,
              description = "Allow traffic when Redis is unavailable (fail-open).",
            },
          },
          {
            -- Cache TTL in seconds. The shared dict cache avoids a Redis
            -- round-trip on every request. 30 seconds balances freshness
            -- with performance; flag changes take up to 30s to propagate.
            cache_ttl = {
              type = "integer",
              default = 30,
              between = { 0, 3600 },
              description = "Seconds to cache flag evaluation in shared dict (0 = no cache).",
            },
          },
          {
            -- When enabled, checks for tenant-specific flag overrides.
            -- Requires the gl-tenant-isolation plugin to be active on
            -- the same route (sets X-Tenant-ID header at priority 900).
            check_tenant_override = {
              type = "boolean",
              default = false,
              description = "Check tenant-specific flag overrides via X-Tenant-ID.",
            },
          },
          {
            -- HTTP status code for blocked requests. 503 is the standard
            -- choice for temporary unavailability with a Retry-After hint.
            response_status = {
              type = "integer",
              default = 503,
              one_of = { 403, 404, 501, 503 },
              description = "HTTP status code when feature is disabled.",
            },
          },
          {
            -- Retry-After header value. 300 seconds (5 minutes) gives
            -- operators time to flip the flag while signaling clients
            -- to back off without aggressive retries.
            retry_after = {
              type = "integer",
              default = 300,
              between = { 0, 86400 },
              description = "Retry-After header value in seconds.",
            },
          },
        },
      },
    },
  },
}
