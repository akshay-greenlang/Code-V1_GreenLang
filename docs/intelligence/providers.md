# GreenLang Intelligence Providers

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** October 1, 2025

---

## Overview

GreenLang Intelligence provides a unified interface for interacting with multiple LLM providers (OpenAI, Anthropic, etc.) with built-in cost control, JSON validation, and tool calling support.

### Supported Providers

| Provider | Models | Function Calling | JSON Mode | Max Context |
|----------|--------|------------------|-----------|-------------|
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 | ✅ Yes | ✅ Native | 128K tokens |
| **Anthropic** | Claude-3 (Opus, Sonnet, Haiku), Claude-2 | ✅ Yes (Claude-3) | ⚠️ Emulated | 200K tokens |

---

## Quick Start

### 1. Select Provider and Model

```python
from greenlang.intelligence.providers.openai import OpenAIProvider
from greenlang.intelligence.providers.anthropic import AnthropicProvider
from greenlang.intelligence.providers.base import LLMProviderConfig

# OpenAI Provider
config = LLMProviderConfig(
    model="gpt-4o-mini",  # Cost-effective, fast
    api_key_env="OPENAI_API_KEY",
    timeout_s=30.0,
    max_retries=3
)
provider = OpenAIProvider(config)

# Anthropic Provider
config = LLMProviderConfig(
    model="claude-3-sonnet-20240229",  # Balanced cost/quality
    api_key_env="ANTHROPIC_API_KEY",
    timeout_s=60.0,
    max_retries=3
)
provider = AnthropicProvider(config)
```

### 2. Simple Chat Completion

```python
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.runtime.budget import Budget

# Create messages
messages = [
    ChatMessage(role=Role.system, content="You are a climate analyst"),
    ChatMessage(role=Role.user, content="What is 2+2?")
]

# Set budget cap
budget = Budget(max_usd=0.10)

# Call provider
response = await provider.chat(messages=messages, budget=budget)

print(f"Response: {response.text}")
print(f"Cost: ${response.usage.cost_usd:.4f}")
print(f"Tokens: {response.usage.total_tokens}")
```

---

## Model Selection Guide

### By Use Case

**Simple Calculations / Quick Queries:**
- `gpt-4o-mini` ($0.15/1M input, $0.60/1M output) - Best value
- `claude-3-haiku-20240307` ($0.25/1M input, $1.25/1M output) - Fast

**Complex Analysis / Reasoning:**
- `gpt-4-turbo` ($10/1M input, $30/1M output) - Strong reasoning
- `claude-3-sonnet-20240229` ($3/1M input, $15/1M output) - Best value for complex tasks

**Maximum Capability:**
- `gpt-4` ($30/1M input, $60/1M output) - Highest quality OpenAI
- `claude-3-opus-20240229` ($15/1M input, $75/1M output) - Highest quality Anthropic

### By Latency Requirement

**Real-time (<1s):**
- `gpt-4o-mini`, `claude-3-haiku`

**Interactive (1-5s):**
- `gpt-4-turbo`, `gpt-4o`, `claude-3-sonnet`

**Batch (>5s acceptable):**
- `gpt-4`, `claude-3-opus`

---

## JSON Strict Mode & Repair Flow

### Overview

GreenLang enforces strict JSON output from LLMs using schema validation and automatic repair prompts.

**CTO Specification:**
- Maximum 3 repair retries
- Hard failure after >3 attempts
- Cost metered on EVERY attempt

### Usage

```python
# Define JSON schema
json_schema = {
    "type": "object",
    "properties": {
        "emissions": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string", "enum": ["kg_CO2e"]},
                "source": {"type": "string"}
            },
            "required": ["value", "unit", "source"]
        },
        "recommendation": {"type": "string"}
    },
    "required": ["emissions", "recommendation"]
}

# Call with JSON schema
response = await provider.chat(
    messages=messages,
    json_schema=json_schema,
    budget=budget
)

# Response.text is guaranteed to be valid JSON matching schema
data = json.loads(response.text)
print(data["emissions"]["value"])  # Always exists, always a number
```

### Repair Flow

**Attempt 1:** Native JSON mode (OpenAI) or system prompt (Anthropic)
- If valid → return immediately (cost calls = 1)
- If invalid → generate repair prompt

**Attempt 2-3:** Retry with repair instructions
- Parse error → repair prompt includes error details
- Schema mismatch → repair prompt shows required format
- If valid → return (cost calls = 2 or 3)

**Attempt 4:** Final attempt
- If still invalid → raise `GLJsonParseError`
- Cost metered on all 4 attempts

### Example: Handling JSON Parse Errors

```python
from greenlang.intelligence.runtime.json_validator import GLJsonParseError

try:
    response = await provider.chat(
        messages=messages,
        json_schema=schema,
        budget=budget,
        metadata={"request_id": "req_123"}
    )
except GLJsonParseError as e:
    print(f"JSON parsing failed after {e.attempts} attempts")
    print(f"Request ID: {e.request_id}")
    print(f"Last error: {e.last_error}")
    print(f"History: {e.history}")
    # Cost was still metered on all attempts
```

---

## Budgeting and Cost Telemetry

### Budget Enforcement

**Pre-call budget check:**
```python
budget = Budget(max_usd=0.50)

# Estimate cost before call
estimated_tokens = 1000
estimated_cost = provider._calculate_cost(estimated_tokens, 500)

# Check if would exceed budget
budget.check(add_usd=estimated_cost, add_tokens=estimated_tokens)

# Make call
response = await provider.chat(messages=messages, budget=budget)

# Cost automatically added to budget after call
print(f"Spent: ${budget.spent_usd:.4f} / ${budget.max_usd:.2f}")
print(f"Remaining: ${budget.remaining_usd:.4f}")
```

### Token Cap

```python
# Limit both cost and tokens
budget = Budget(
    max_usd=1.00,  # Dollar cap
    max_tokens=8000  # Token cap (for context limits)
)

try:
    response = await provider.chat(messages=messages, budget=budget)
except BudgetExceeded as e:
    print(f"Exceeded: ${e.spent_usd:.4f} / ${e.max_usd:.2f}")
    print(f"Tokens: {e.spent_tokens} / {e.max_tokens}")
```

### Cost Breakdown

```python
# Get detailed usage
usage = response.usage

print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
print(f"Cost: ${usage.cost_usd:.6f}")

# Cost calculation
# gpt-4o-mini: $0.15/1M input, $0.60/1M output
# Cost = (prompt_tokens / 1M * $0.15) + (completion_tokens / 1M * $0.60)
```

---

## Error Types and Retry Strategies

### Error Taxonomy

| Error Class | HTTP Code | Retry? | Backoff |
|-------------|-----------|--------|---------|
| `ProviderAuthError` | 401, 403 | ❌ No | N/A |
| `ProviderRateLimit` | 429 | ✅ Yes | Exponential + retry_after |
| `ProviderTimeout` | 408, 504 | ✅ Yes | Exponential |
| `ProviderServerError` | 500, 502, 503 | ✅ Yes | Exponential |
| `ProviderBadRequest` | 400, 422 | ❌ No | N/A |
| `ProviderContentFilter` | 400 | ❌ No | N/A |

### Retry Logic

```python
from greenlang.intelligence.providers.errors import (
    ProviderRateLimit,
    ProviderTimeout,
    ProviderServerError
)

max_retries = 3
base_delay = 1.0

for attempt in range(max_retries + 1):
    try:
        response = await provider.chat(messages=messages, budget=budget)
        break  # Success
    except ProviderRateLimit as e:
        if attempt >= max_retries:
            raise
        delay = e.retry_after or (base_delay * (2 ** attempt))
        await asyncio.sleep(delay)
    except (ProviderTimeout, ProviderServerError) as e:
        if attempt >= max_retries:
            raise
        delay = base_delay * (2 ** attempt)  # Exponential backoff
        await asyncio.sleep(delay)
    except ProviderAuthError:
        # Never retry auth errors
        raise
```

### Exponential Backoff

**Attempt 1:** 1s delay
**Attempt 2:** 2s delay
**Attempt 3:** 4s delay
**Attempt 4:** Raise error

---

## Function Calling / Tool Use

### Defining Tools

```python
from greenlang.intelligence.schemas.tools import ToolDef

tools = [
    ToolDef(
        name="calculate_emissions",
        description="Calculate CO2e emissions from fuel combustion",
        parameters={
            "type": "object",
            "properties": {
                "fuel_type": {
                    "type": "string",
                    "enum": ["diesel", "gasoline", "natural_gas"],
                    "description": "Type of fuel"
                },
                "amount": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Fuel amount in gallons"
                }
            },
            "required": ["fuel_type", "amount"]
        }
    )
]
```

### Calling with Tools

```python
response = await provider.chat(
    messages=messages,
    tools=tools,
    tool_choice="auto",  # "auto", "none", "required", or tool name
    budget=budget
)

# Check if LLM wants to call a tool
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Arguments: {tool_call['arguments']}")

        # Execute tool
        result = execute_tool(tool_call['name'], tool_call['arguments'])

        # Send result back to LLM
        messages.append(ChatMessage(
            role=Role.tool,
            content=json.dumps(result),
            name=tool_call['name'],
            tool_call_id=tool_call['id']
        ))

        # Continue conversation
        response = await provider.chat(messages=messages, tools=tools, budget=budget)
```

### Neutral Tool Call Format

All providers return tool calls in consistent format:

```python
[
    {
        "id": "call_abc123",          # Provider-specific ID
        "name": "calculate_emissions", # Tool name
        "arguments": {                 # Parsed JSON arguments
            "fuel_type": "diesel",
            "amount": 100
        }
    }
]
```

**Arguments are NEVER modified** by the provider layer. Validation happens in the runtime/tool layer.

---

## Advanced Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export GL_OPENAI_API_KEY=sk-...  # Optional: GreenLang-specific

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export GL_ANTHROPIC_API_KEY=sk-ant-...  # Optional

# Custom base URLs (for proxies/testing)
export GL_OPENAI_BASE_URL=https://custom-proxy.com/v1
export GL_ANTHROPIC_BASE_URL=https://custom-proxy.com
```

### Timeout Configuration

```python
config = LLMProviderConfig(
    model="gpt-4o-mini",
    api_key_env="OPENAI_API_KEY",
    timeout_s=60.0,  # 60 second timeout
    max_retries=3    # Retry up to 3 times
)
```

**Timeout behavior:**
- `timeout_s` applies to each individual API call
- Retries extend total time: max_time = timeout_s * (max_retries + 1)
- Timeout errors are retryable (exponential backoff)

### Temperature and Sampling

```python
response = await provider.chat(
    messages=messages,
    temperature=0.0,  # Deterministic (0.0-2.0)
    top_p=1.0,        # Nucleus sampling (0.0-1.0)
    seed=42,          # Reproducibility (OpenAI only)
    budget=budget
)
```

**Temperature guide:**
- `0.0`: Fully deterministic (recommended for production)
- `0.7`: Balanced creativity/consistency
- `1.5-2.0`: Maximum creativity (use with caution)

### Metadata and Request IDs

```python
response = await provider.chat(
    messages=messages,
    budget=budget,
    metadata={
        "request_id": "req_abc123",
        "user_id": "user_456",
        "session_id": "sess_789"
    }
)

# Access provider info
print(f"Provider: {response.provider_info.provider}")
print(f"Model: {response.provider_info.model}")
print(f"Request ID: {response.provider_info.request_id}")
```

---

## Security Best Practices

### API Key Management

✅ **DO:**
- Load API keys from environment variables
- Use different keys for dev/staging/prod
- Rotate keys regularly
- Use read-only keys when possible

❌ **DON'T:**
- Hardcode API keys in code
- Commit API keys to version control
- Log raw API keys
- Share API keys across environments

### Logging and Monitoring

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Provider logs include:
# - Model and provider name
# - Token counts and costs
# - Finish reasons
# - Retry attempts
# - Request IDs (for debugging)

# API keys are NEVER logged (always REDACTED)
# Raw responses are NOT logged (security)
```

### Rate Limiting

```python
# Implement application-level rate limiting
from time import sleep

max_requests_per_minute = 60
request_count = 0
start_time = time.time()

while request_count < max_requests_per_minute:
    response = await provider.chat(messages=messages, budget=budget)
    request_count += 1

    # Sleep if approaching rate limit
    elapsed = time.time() - start_time
    if elapsed < 60:
        sleep((60 - elapsed) / (max_requests_per_minute - request_count))
```

---

## Performance Optimization

### Batch Processing

```python
# Process multiple requests in parallel
import asyncio

async def process_batch(messages_list, provider, budget):
    tasks = [
        provider.chat(messages=msgs, budget=Budget(max_usd=0.10))
        for msgs in messages_list
    ]
    results = await asyncio.gather(*tasks)
    return results

# Execute
messages_batch = [messages1, messages2, messages3]
results = await process_batch(messages_batch, provider, budget)
```

### Cost Optimization

1. **Use cheaper models for simple tasks:**
   ```python
   if task_complexity == "simple":
       config.model = "gpt-4o-mini"  # $0.15/1M vs $10/1M
   else:
       config.model = "gpt-4-turbo"
   ```

2. **Cache results:**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   async def cached_chat(messages_hash):
       return await provider.chat(messages=messages, budget=budget)
   ```

3. **Limit output tokens:**
   ```python
   # For OpenAI (via parameters)
   response = await provider.chat(
       messages=messages,
       max_tokens=100,  # Limit output
       budget=budget
   )
   ```

---

## Troubleshooting

### Common Issues

**Issue: `ValueError: API key not found`**
```bash
# Solution: Set environment variable
export OPENAI_API_KEY=sk-...
```

**Issue: `BudgetExceeded` raised unexpectedly**
```python
# Solution: Increase budget or use cheaper model
budget = Budget(max_usd=1.00)  # Increase from 0.10
```

**Issue: `GLJsonParseError` after >3 retries**
```python
# Solution: Simplify schema or improve prompt
json_schema = {
    "type": "object",
    "properties": {
        "result": {"type": "number"}  # Simpler schema
    }
}
```

**Issue: Slow responses**
```python
# Solution: Use faster model or increase timeout
config.model = "gpt-4o-mini"  # Faster
config.timeout_s = 120.0  # Longer timeout
```

---

## Examples

See `examples/intelligence/complete_demo.py` for comprehensive examples demonstrating:
- Tool calling
- JSON schema validation
- Budget enforcement
- Cost tracking
- Error handling

---

## API Reference

### LLMProviderConfig

```python
class LLMProviderConfig(BaseModel):
    model: str                  # Model name
    api_key_env: str           # Env var for API key
    timeout_s: float = 60.0    # Request timeout
    max_retries: int = 3       # Max retry attempts
```

### Budget

```python
class Budget(BaseModel):
    max_usd: float             # Dollar cap
    max_tokens: int | None     # Token cap (optional)
    spent_usd: float = 0.0     # Current spend
    spent_tokens: int = 0      # Current tokens
```

### ChatResponse

```python
class ChatResponse(BaseModel):
    text: str | None                     # Text response
    tool_calls: list[dict] | None        # Tool calls
    usage: Usage                          # Token usage and cost
    finish_reason: FinishReason          # Why generation stopped
    provider_info: ProviderInfo          # Provider metadata
```

---

## Support

- **Documentation:** [docs.greenlang.com](https://docs.greenlang.com)
- **Issues:** [github.com/greenlang/issues](https://github.com/greenlang/greenlang/issues)
- **Examples:** `examples/intelligence/`

---

**Last Updated:** October 1, 2025
**Version:** 1.0.0
**Status:** Production Ready
