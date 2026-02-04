-- =============================================================================
-- GreenLang Climate OS - Feature Gate Plugin (handler.lua)
-- =============================================================================
-- PRD: INFRA-008 Feature Flags System
--
-- This custom Kong plugin gates routes based on feature flag state stored in
-- Redis. It evaluates a flag key for the current route and either allows the
-- request to proceed or returns a 503 "Feature temporarily unavailable"
-- response. This enables zero-downtime feature rollouts, kill switches, and
-- tenant-specific feature overrides at the API gateway layer.
--
-- Execution phases:
--   access()        - Read feature flag from Redis (with shared dict cache),
--                     evaluate flag status, optionally check tenant overrides,
--                     and either allow or reject the request.
--   header_filter() - Add X-Feature-Flag-Status response header for
--                     observability and client-side correlation.
--   log()           - Attach flag evaluation metadata to the log serializer
--                     for audit trail enrichment.
--
-- Redis key patterns:
--   ff:{environment}:flag:{flag_key}               - Global flag state
--   ff:{environment}:override:{flag_key}:{tenant}  - Tenant-specific override
--
-- Redis value format (JSON):
--   {"status": "active"|"inactive", "default_value": true|false, ...}
--
-- Priority: 850 (after gl-tenant-isolation at 900 so X-Tenant-ID is available,
-- before request-transformer and other lower-priority plugins).
--
-- Version: 1.0.0
-- =============================================================================

local kong = kong
local cjson = require "cjson.safe"
local redis = require "resty.redis"

local fmt = string.format
local ngx_now = ngx.now
local ngx_log = ngx.log
local ngx_DEBUG = ngx.DEBUG

-- ---------------------------------------------------------------------------
-- Plugin definition
-- ---------------------------------------------------------------------------
local FeatureGate = {
  PRIORITY = 850,
  VERSION = "1.0.0",
}

-- ---------------------------------------------------------------------------
-- Internal: shared dictionary cache key builder
-- ---------------------------------------------------------------------------
-- Builds a namespaced cache key for the kong.ctx.shared local cache to avoid
-- collisions with other plugins that also use shared context.
-- ---------------------------------------------------------------------------
local function build_cache_key(environment, flag_key, tenant_id)
  if tenant_id then
    return fmt("ff:%s:override:%s:%s", environment, flag_key, tenant_id)
  end
  return fmt("ff:%s:flag:%s", environment, flag_key)
end

-- ---------------------------------------------------------------------------
-- Internal: read cached flag value from shared dict
-- ---------------------------------------------------------------------------
-- Returns the cached value and a boolean indicating whether the cache entry
-- is still valid (not expired). We store both the value and the expiry
-- timestamp in kong.ctx.shared to avoid Redis round-trips on every request.
-- ---------------------------------------------------------------------------
local function get_cached_flag(cache_key, cache_ttl)
  local shared = ngx.shared.kong_db_cache
  if not shared then
    return nil, false
  end

  local cached = shared:get(cache_key)
  if cached then
    return cached, true
  end

  return nil, false
end

-- ---------------------------------------------------------------------------
-- Internal: store flag value in shared dict cache
-- ---------------------------------------------------------------------------
local function set_cached_flag(cache_key, value, cache_ttl)
  local shared = ngx.shared.kong_db_cache
  if not shared then
    return
  end

  -- Store with TTL; if the shared dict is full, eviction is handled by nginx
  local ok, err = shared:set(cache_key, value, cache_ttl)
  if not ok then
    kong.log.warn(
      "[gl-feature-gate] Failed to cache flag value; ",
      "key=", cache_key,
      " err=", err
    )
  end
end

-- ---------------------------------------------------------------------------
-- Internal: connect to Redis and fetch a flag value
-- ---------------------------------------------------------------------------
-- Opens a Redis connection with the configured timeout, selects the correct
-- database, and fetches the flag value by key. Uses socket keepalive to
-- reuse connections across requests (connection pooling via cosocket).
--
-- Returns:
--   flag_data (table|nil) - Decoded JSON flag data, or nil on error
--   err (string|nil)      - Error message if Redis is unavailable
-- ---------------------------------------------------------------------------
local function fetch_flag_from_redis(conf, redis_key)
  local red = redis:new()
  red:set_timeout(conf.redis_timeout)

  -- Connect to Redis
  local ok, err = red:connect(conf.redis_host, conf.redis_port)
  if not ok then
    kong.log.err(
      "[gl-feature-gate] Redis connection failed; ",
      "host=", conf.redis_host,
      ":", conf.redis_port,
      " err=", err
    )
    return nil, err
  end

  -- Authenticate if password is configured
  if conf.redis_password and conf.redis_password ~= "" then
    local auth_ok, auth_err = red:auth(conf.redis_password)
    if not auth_ok then
      kong.log.err(
        "[gl-feature-gate] Redis auth failed; err=", auth_err
      )
      red:close()
      return nil, auth_err
    end
  end

  -- Select the correct database (default 3 for Kong feature flags)
  if conf.redis_database > 0 then
    local select_ok, select_err = red:select(conf.redis_database)
    if not select_ok then
      kong.log.err(
        "[gl-feature-gate] Redis SELECT failed; db=", conf.redis_database,
        " err=", select_err
      )
      red:close()
      return nil, select_err
    end
  end

  -- Fetch the flag value
  local value, get_err = red:get(redis_key)
  if not value then
    kong.log.err(
      "[gl-feature-gate] Redis GET failed; key=", redis_key,
      " err=", get_err
    )
    red:close()
    return nil, get_err
  end

  -- Return connection to the pool (keepalive 60s, pool size 100)
  local keepalive_ok, keepalive_err = red:set_keepalive(60000, 100)
  if not keepalive_ok then
    kong.log.warn(
      "[gl-feature-gate] Redis keepalive failed; err=", keepalive_err
    )
  end

  -- Redis returns ngx.null for missing keys
  if value == ngx.null then
    return nil, nil
  end

  -- Decode JSON flag data
  local flag_data, decode_err = cjson.decode(value)
  if not flag_data then
    kong.log.err(
      "[gl-feature-gate] Failed to decode flag JSON; key=", redis_key,
      " err=", decode_err
    )
    return nil, decode_err
  end

  return flag_data, nil
end

-- ---------------------------------------------------------------------------
-- Internal: evaluate whether a flag is enabled
-- ---------------------------------------------------------------------------
-- A flag is considered enabled when:
--   1. status == "active" AND default_value is truthy (not false/nil)
--   2. If the flag key is missing from Redis, the plugin falls through to
--      the default_on_error behavior.
--
-- Returns: boolean (true = allow request, false = block request)
-- ---------------------------------------------------------------------------
local function is_flag_enabled(flag_data)
  if not flag_data then
    return false
  end

  if flag_data.status ~= "active" then
    return false
  end

  -- Explicit false check; treat nil/missing default_value as enabled when
  -- the status is "active" (the status field is the primary gate).
  if flag_data.default_value == false then
    return false
  end

  return true
end

-- ---------------------------------------------------------------------------
-- access() - Runs on every proxied request during the access phase.
-- ---------------------------------------------------------------------------
-- Evaluates the feature flag for the current route. Uses a three-layer
-- resolution strategy:
--   1. Check shared dict cache (avoids Redis on hot paths)
--   2. If cache miss, fetch from Redis
--   3. If Redis is unavailable, use conf.default_on_error
--
-- When conf.check_tenant_override is true, also checks for a tenant-specific
-- override flag keyed by X-Tenant-ID (set by the gl-tenant-isolation plugin
-- which runs at priority 900, before this plugin at 850).
-- ---------------------------------------------------------------------------
function FeatureGate:access(conf)
  local start_time = ngx_now()
  local flag_key = conf.flag_key
  local environment = conf.environment

  -- Build the primary Redis key
  local redis_key = build_cache_key(environment, flag_key, nil)
  local cache_key = "gl_ff:" .. redis_key

  -- Step 1: Check shared dict cache
  local cached_value, cache_hit = get_cached_flag(cache_key, conf.cache_ttl)
  local flag_enabled = nil
  local resolution_source = "cache"

  if cache_hit then
    -- Cached value is stored as "1" (enabled) or "0" (disabled)
    flag_enabled = (cached_value == "1")
  else
    -- Step 2: Fetch from Redis
    resolution_source = "redis"
    local flag_data, redis_err = fetch_flag_from_redis(conf, redis_key)

    if redis_err then
      -- Step 3: Redis unavailable - use default_on_error
      resolution_source = "default"
      flag_enabled = conf.default_on_error

      kong.log.warn(
        "[gl-feature-gate] Redis unavailable, using default_on_error=",
        tostring(conf.default_on_error),
        "; flag_key=", flag_key,
        " err=", redis_err
      )
    elseif flag_data then
      flag_enabled = is_flag_enabled(flag_data)
      -- Cache the result
      set_cached_flag(cache_key, flag_enabled and "1" or "0", conf.cache_ttl)
    else
      -- Flag key not found in Redis (nil data, no error)
      resolution_source = "missing"
      flag_enabled = conf.default_on_error

      kong.log.info(
        "[gl-feature-gate] Flag key not found in Redis; ",
        "flag_key=", flag_key,
        " redis_key=", redis_key,
        " default_on_error=", tostring(conf.default_on_error)
      )
      -- Cache the miss to avoid repeated Redis lookups
      set_cached_flag(
        cache_key, conf.default_on_error and "1" or "0", conf.cache_ttl
      )
    end
  end

  -- Step 4: Check tenant-specific override if enabled and flag is disabled
  local tenant_id = nil
  if conf.check_tenant_override then
    tenant_id = kong.ctx.shared.tenant_id
        or kong.request.get_header("X-Tenant-ID")

    if tenant_id and tenant_id ~= "" then
      local override_redis_key = build_cache_key(
        environment, flag_key, tenant_id
      )
      local override_cache_key = "gl_ff:" .. override_redis_key

      -- Check override cache first
      local override_cached, override_cache_hit = get_cached_flag(
        override_cache_key, conf.cache_ttl
      )

      if override_cache_hit then
        flag_enabled = (override_cached == "1")
        resolution_source = "tenant_cache"
      else
        -- Fetch tenant override from Redis
        local override_data, override_err = fetch_flag_from_redis(
          conf, override_redis_key
        )

        if override_data then
          flag_enabled = is_flag_enabled(override_data)
          resolution_source = "tenant_override"
          set_cached_flag(
            override_cache_key,
            flag_enabled and "1" or "0",
            conf.cache_ttl
          )
        elseif not override_err then
          -- No tenant override exists; cache the miss
          set_cached_flag(
            override_cache_key,
            flag_enabled and "1" or "0",
            conf.cache_ttl
          )
        end
        -- If Redis error on override lookup, keep the global flag result
      end
    end
  end

  -- Store evaluation result in shared context for header_filter and log phases
  local evaluation_time_ms = (ngx_now() - start_time) * 1000
  kong.ctx.shared.feature_flag_key = flag_key
  kong.ctx.shared.feature_flag_enabled = flag_enabled
  kong.ctx.shared.feature_flag_source = resolution_source
  kong.ctx.shared.feature_flag_eval_ms = evaluation_time_ms

  kong.log.debug(
    "[gl-feature-gate] Flag evaluated; ",
    "flag_key=", flag_key,
    " enabled=", tostring(flag_enabled),
    " source=", resolution_source,
    " tenant=", tostring(tenant_id),
    " eval_ms=", fmt("%.2f", evaluation_time_ms)
  )

  -- Block the request if the feature flag is disabled
  if not flag_enabled then
    kong.log.info(
      "[gl-feature-gate] Request blocked by feature gate; ",
      "flag_key=", flag_key,
      " path=", kong.request.get_path(),
      " source=", resolution_source
    )

    return kong.response.exit(conf.response_status, {
      message = "Feature temporarily unavailable",
      flag_key = flag_key,
      retry_after = conf.retry_after,
    }, {
      ["Retry-After"] = tostring(conf.retry_after),
      ["Content-Type"] = "application/json",
      ["X-Feature-Flag-Status"] = "disabled",
    })
  end
end

-- ---------------------------------------------------------------------------
-- header_filter() - Runs during response header construction.
-- ---------------------------------------------------------------------------
-- Adds the X-Feature-Flag-Status header to the response for observability.
-- Downstream services and monitoring systems can use this header to track
-- which requests were gated and the evaluation source.
-- ---------------------------------------------------------------------------
function FeatureGate:header_filter(conf)
  local flag_enabled = kong.ctx.shared.feature_flag_enabled

  if flag_enabled ~= nil then
    local status = flag_enabled and "enabled" or "disabled"
    kong.response.set_header("X-Feature-Flag-Status", status)
  end
end

-- ---------------------------------------------------------------------------
-- log() - Runs during the log phase after the response has been sent.
-- ---------------------------------------------------------------------------
-- Attaches feature flag evaluation metadata to the log serializer so that
-- all log entries include flag context for audit and debugging purposes.
-- ---------------------------------------------------------------------------
function FeatureGate:log(conf)
  local flag_key = kong.ctx.shared.feature_flag_key

  if flag_key then
    local log_data = kong.log.serialize()
    if log_data then
      log_data.feature_flag = {
        key = flag_key,
        enabled = kong.ctx.shared.feature_flag_enabled,
        source = kong.ctx.shared.feature_flag_source,
        evaluation_time_ms = kong.ctx.shared.feature_flag_eval_ms,
      }
    end
  end
end

return FeatureGate
