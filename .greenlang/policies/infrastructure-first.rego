# GreenLang Infrastructure-First Policy
# ======================================
#
# This OPA policy enforces infrastructure-first principles at runtime.
# It ensures all API calls, LLM operations, and cache operations go through
# GreenLang infrastructure.

package greenlang.infrastructure_first

import future.keywords.if
import future.keywords.in

# Default deny
default allow = false

# =============================================================================
# API Authentication Requirements
# =============================================================================

# Allow API calls only with valid GreenLang auth token
allow if {
    input.type == "api_call"
    has_greenlang_token
    token_is_valid
}

has_greenlang_token if {
    input.headers["Authorization"]
    startswith(input.headers["Authorization"], "Bearer gl_")
}

token_is_valid if {
    # Token format: gl_<env>_<random>
    token := trim_prefix(input.headers["Authorization"], "Bearer ")
    parts := split(token, "_")
    count(parts) >= 3
    parts[0] == "gl"
}

# =============================================================================
# LLM Call Requirements
# =============================================================================

# Allow LLM calls only through ChatSession
allow if {
    input.type == "llm_call"
    uses_chat_session
    has_budget_check
}

uses_chat_session if {
    input.metadata.caller == "greenlang.intelligence.ChatSession"
}

uses_chat_session if {
    input.metadata.caller == "greenlang.intelligence.runtime.session.ChatSession"
}

has_budget_check if {
    input.metadata.budget_checked == true
}

# Deny direct LLM provider calls
deny[msg] if {
    input.type == "llm_call"
    not uses_chat_session
    msg := sprintf(
        "Direct LLM call blocked. Use greenlang.intelligence.ChatSession. Caller: %s",
        [input.metadata.caller]
    )
}

# =============================================================================
# Cache Operation Requirements
# =============================================================================

# Allow cache operations only through CacheManager
allow if {
    input.type == "cache_operation"
    uses_cache_manager
}

uses_cache_manager if {
    input.metadata.caller == "greenlang.cache.CacheManager"
}

uses_cache_manager if {
    input.metadata.caller == "greenlang.intelligence.semantic_cache.SemanticCache"
}

# Deny direct Redis/cache calls
deny[msg] if {
    input.type == "cache_operation"
    not uses_cache_manager
    msg := sprintf(
        "Direct cache access blocked. Use greenlang.cache.CacheManager. Caller: %s",
        [input.metadata.caller]
    )
}

# =============================================================================
# Database Operation Requirements
# =============================================================================

# Allow database operations only through GreenLang connectors
allow if {
    input.type == "db_operation"
    uses_greenlang_db
}

uses_greenlang_db if {
    startswith(input.metadata.caller, "greenlang.db")
}

uses_greenlang_db if {
    startswith(input.metadata.caller, "greenlang.database")
}

# Warn on direct database access
warn[msg] if {
    input.type == "db_operation"
    not uses_greenlang_db
    msg := sprintf(
        "Direct database access detected. Consider using greenlang.db. Caller: %s",
        [input.metadata.caller]
    )
}

# =============================================================================
# Agent Execution Requirements
# =============================================================================

# Allow agent execution only for proper Agent subclasses
allow if {
    input.type == "agent_execution"
    is_greenlang_agent
}

is_greenlang_agent if {
    input.metadata.base_class == "greenlang.sdk.base.Agent"
}

is_greenlang_agent if {
    # Check if agent metadata includes proper inheritance
    some base in input.metadata.base_classes
    base == "greenlang.sdk.base.Agent"
}

deny[msg] if {
    input.type == "agent_execution"
    not is_greenlang_agent
    msg := sprintf(
        "Agent does not inherit from greenlang.sdk.base.Agent. Agent: %s",
        [input.metadata.agent_name]
    )
}

# =============================================================================
# ADR Override
# =============================================================================

# Allow if ADR exists and is approved
allow if {
    input.override.adr_exists
    input.override.adr_approved
    input.override.adr_id != ""
}

# Log ADR usage
log_adr_usage[entry] if {
    input.override.adr_exists
    entry := {
        "timestamp": time.now_ns(),
        "adr_id": input.override.adr_id,
        "operation": input.type,
        "user": input.user,
        "reason": input.override.reason
    }
}

# =============================================================================
# Audit Trail
# =============================================================================

# Log all denials
audit_log[entry] if {
    count(deny) > 0
    entry := {
        "timestamp": time.now_ns(),
        "type": "denial",
        "operation": input.type,
        "caller": input.metadata.caller,
        "user": input.user,
        "violations": deny
    }
}

# Log all warnings
audit_log[entry] if {
    count(warn) > 0
    entry := {
        "timestamp": time.now_ns(),
        "type": "warning",
        "operation": input.type,
        "caller": input.metadata.caller,
        "user": input.user,
        "warnings": warn
    }
}

# =============================================================================
# Test Cases
# =============================================================================

# Test: Valid LLM call through ChatSession
test_valid_llm_call if {
    allow with input as {
        "type": "llm_call",
        "metadata": {
            "caller": "greenlang.intelligence.ChatSession",
            "budget_checked": true
        }
    }
}

# Test: Invalid direct LLM call
test_invalid_llm_call if {
    not allow with input as {
        "type": "llm_call",
        "metadata": {
            "caller": "openai.ChatCompletion",
            "budget_checked": true
        }
    }
}

# Test: Valid cache operation
test_valid_cache if {
    allow with input as {
        "type": "cache_operation",
        "metadata": {
            "caller": "greenlang.cache.CacheManager"
        }
    }
}

# Test: Invalid direct cache access
test_invalid_cache if {
    not allow with input as {
        "type": "cache_operation",
        "metadata": {
            "caller": "redis.Redis"
        }
    }
}

# Test: Valid agent execution
test_valid_agent if {
    allow with input as {
        "type": "agent_execution",
        "metadata": {
            "agent_name": "MyAgent",
            "base_class": "greenlang.sdk.base.Agent"
        }
    }
}

# Test: Invalid custom agent
test_invalid_agent if {
    not allow with input as {
        "type": "agent_execution",
        "metadata": {
            "agent_name": "CustomAgent",
            "base_class": "object"
        }
    }
}

# Test: ADR override
test_adr_override if {
    allow with input as {
        "type": "llm_call",
        "metadata": {
            "caller": "custom.llm.Client"
        },
        "override": {
            "adr_exists": true,
            "adr_approved": true,
            "adr_id": "ADR-2024-001",
            "reason": "Performance optimization"
        },
        "user": "admin@greenlang.io"
    }
}
