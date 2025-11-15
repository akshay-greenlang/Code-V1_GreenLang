# GreenLang LLM Integration Infrastructure

Production-ready multi-provider LLM infrastructure with intelligent routing, automatic failover, circuit breaker protection, cost tracking, and budget management.

## Features

### 1. Multi-Provider Support
- **Anthropic Claude** (claude-3-opus, claude-3-sonnet, claude-3-haiku)
- **OpenAI GPT** (gpt-4-turbo, gpt-4, gpt-3.5-turbo)
- **Custom Providers** (extend `BaseLLMProvider`)

### 2. Intelligent Routing (LLMRouter)
- **Routing Strategies**:
  - `PRIORITY`: Use providers in priority order with fallback
  - `LEAST_COST`: Route to cheapest provider for request
  - `LEAST_LATENCY`: Route to fastest provider
  - `ROUND_ROBIN`: Distribute load evenly
  - `RANDOM`: Random selection from healthy providers

- **Automatic Failover**: Tries primary provider, falls back to secondary on failures
- **Health Monitoring**: Background health checks every 30 seconds (configurable)
- **Circuit Breaker**: Prevents cascading failures (opens after 5 failures, recovers after 60s)
- **Provider Metrics**: Track requests, costs, latency per provider

### 3. Cost Tracking (CostTracker)
- **Multi-Dimensional Tracking**:
  - By Provider (anthropic, openai, etc.)
  - By Tenant (customer/organization)
  - By Agent (specific agent instances)
  - By Model (claude-3-opus, gpt-4, etc.)

- **Budget Management**:
  - Set monthly budget limits per tenant
  - Configurable alert thresholds (80%, 90%, 100%)
  - Automatic budget reset (monthly)
  - Budget forecasting and projections

- **Reporting & Export**:
  - Cost summaries with breakdowns
  - Time-series data (hourly, daily, monthly)
  - Export to CSV/JSON
  - Real-time cost monitoring

### 4. Circuit Breaker Protection
- **States**: CLOSED → OPEN → HALF_OPEN → CLOSED
- **Automatic Recovery**: Tests recovery after timeout
- **Configurable Thresholds**: Failures before opening (default: 5)
- **Recovery Timeout**: Seconds before attempting recovery (default: 60)

### 5. Provider Features
- **Retry Logic**: Exponential backoff (1s, 2s, 4s, 8s)
- **Rate Limiting**: Automatic handling of rate limit errors
- **Token Counting**: Accurate token usage tracking
- **Cost Calculation**: Real-time cost tracking
- **Health Checks**: Periodic provider health monitoring
- **Streaming Support**: Server-sent events for real-time responses

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Request                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LLMRouter (selects provider)              │
│  Strategies: PRIORITY | LEAST_COST | LEAST_LATENCY |        │
│             ROUND_ROBIN | RANDOM                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ CircuitBreaker  │  │ CircuitBreaker  │  │ CircuitBreaker  │
│   (Provider 1)  │  │   (Provider 2)  │  │   (Provider 3)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         ↓                    ↓                    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Anthropic     │  │     OpenAI      │  │     Custom      │
│  (Claude 3)     │  │    (GPT-4)      │  │   Provider      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         ↓                    ↓                    ↓
         └────────────────────┼────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               CostTracker (tracks all usage)                 │
│  Dimensions: Provider, Tenant, Agent, Model, Time            │
│  Budgets: Monthly limits with alerts                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- `anthropic>=0.18.0`
- `openai>=1.12.0`
- `tiktoken>=0.6.0`
- `pydantic>=2.5.0`
- `aiohttp>=3.9.0`
- `httpx>=0.26.0`

## Quick Start

### 1. Basic Usage with Single Provider

```python
import asyncio
from llm.providers import AnthropicProvider
from llm.providers.base_provider import GenerationRequest

async def main():
    # Initialize provider
    provider = AnthropicProvider(
        model_id="claude-3-sonnet-20240229",
        api_key="your-api-key"
    )

    # Create request
    request = GenerationRequest(
        prompt="Explain carbon accounting in 2 sentences.",
        temperature=0.7,
        max_tokens=100
    )

    # Generate
    response = await provider.generate(request)
    print(f"Response: {response.text}")
    print(f"Cost: ${response.usage.total_cost_usd:.4f}")

    await provider.close()

asyncio.run(main())
```

### 2. Multi-Provider with Automatic Failover

```python
import asyncio
from llm import LLMRouter, RoutingStrategy
from llm.providers import AnthropicProvider, OpenAIProvider
from llm.providers.base_provider import GenerationRequest

async def main():
    # Initialize router with least-cost strategy
    router = LLMRouter(
        strategy=RoutingStrategy.LEAST_COST,
        enable_circuit_breaker=True
    )

    # Register providers (priority 1 = highest)
    anthropic = AnthropicProvider(model_id="claude-3-sonnet-20240229")
    router.register_provider("anthropic", anthropic, priority=1)

    openai = OpenAIProvider(model_id="gpt-3.5-turbo")
    router.register_provider("openai", openai, priority=2)

    # Start health monitoring
    await router.start_health_monitoring()

    # Generate with automatic provider selection
    request = GenerationRequest(
        prompt="Analyze ESG data...",
        max_tokens=500
    )

    response = await router.generate(request)
    print(f"Provider used: {response.provider}")
    print(f"Cost: ${response.usage.total_cost_usd:.4f}")

    # Get metrics
    metrics = router.get_metrics()
    print(f"Total cost: ${metrics['global']['total_cost_usd']:.4f}")
    print(f"Failovers: {metrics['global']['failover_count']}")

    await router.close()

asyncio.run(main())
```

### 3. Cost Tracking with Budget Alerts

```python
from llm import CostTracker
from llm.providers.base_provider import TokenUsage

# Initialize tracker
tracker = CostTracker()

# Set budget with alerts at 80%, 90%, 100%
tracker.set_budget(
    tenant_id="acme-corp",
    monthly_limit_usd=1000.0,
    alert_thresholds=[0.8, 0.9, 1.0]
)

# Register alert callback
def on_alert(tenant_id, threshold, current_cost):
    print(f"ALERT: {threshold*100:.0f}% budget used (${current_cost:.2f})")

tracker.register_alert_callback(on_alert)

# Track usage
usage = TokenUsage(
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
    input_cost_usd=0.003,
    output_cost_usd=0.0075,
    total_cost_usd=0.0105
)

tracker.track_usage(
    provider="anthropic",
    tenant_id="acme-corp",
    agent_id="esg-agent",
    model_id="claude-3-sonnet",
    usage=usage
)

# Check budget
status = tracker.check_budget("acme-corp")
print(f"Budget used: {status.percentage_used:.1f}%")
print(f"Remaining: ${status.remaining_usd:.2f}")

# Export report
tracker.export_csv("costs.csv")
tracker.export_json("costs.json", include_summary=True)
```

### 4. Complete Integration (Router + Cost Tracker)

```python
import asyncio
from llm import LLMRouter, CostTracker, RoutingStrategy
from llm.providers import AnthropicProvider, OpenAIProvider
from llm.providers.base_provider import GenerationRequest

async def main():
    # Initialize components
    router = LLMRouter(strategy=RoutingStrategy.LEAST_COST)
    tracker = CostTracker()

    # Setup providers
    router.register_provider("anthropic", AnthropicProvider(...), priority=1)
    router.register_provider("openai", OpenAIProvider(...), priority=2)

    # Setup budget
    tracker.set_budget("tenant-123", monthly_limit_usd=500.0)

    # Start monitoring
    await router.start_health_monitoring()

    # Generate
    request = GenerationRequest(prompt="Analyze data...", max_tokens=500)
    response = await router.generate(request)

    # Track costs
    tracker.track_usage(
        provider=response.provider,
        tenant_id="tenant-123",
        agent_id="my-agent",
        model_id=response.model_id,
        usage=response.usage
    )

    # Get reports
    budget_status = tracker.check_budget("tenant-123")
    print(f"Budget: {budget_status.percentage_used:.1f}% used")

    router_metrics = router.get_metrics()
    print(f"Requests: {router_metrics['global']['total_requests']}")

    await router.close()

asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Provider API Keys
export ANTHROPIC_API_KEY=your-anthropic-key
export OPENAI_API_KEY=your-openai-key

# Optional: Logging
export LOG_LEVEL=INFO
```

### Router Configuration

```python
router = LLMRouter(
    strategy=RoutingStrategy.LEAST_COST,      # Routing strategy
    health_check_interval=30.0,                # Health check frequency (seconds)
    enable_circuit_breaker=True,               # Enable circuit breaker protection
    circuit_breaker_threshold=5,               # Failures before opening
    circuit_breaker_timeout=60.0,              # Recovery timeout (seconds)
    max_retries=3                              # Max retries across providers
)
```

### Cost Tracker Configuration

```python
tracker = CostTracker(
    auto_reset_budgets=True  # Auto-reset monthly budgets
)

tracker.set_budget(
    tenant_id="customer-123",
    monthly_limit_usd=1000.0,
    alert_thresholds=[0.8, 0.9, 1.0]  # Alert at 80%, 90%, 100%
)
```

### Provider Configuration

```python
# Anthropic
provider = AnthropicProvider(
    model_id="claude-3-opus-20240229",
    api_key="your-key",
    max_retries=4,
    base_delay=1.0,
    max_delay=8.0,
    timeout=60.0
)

# OpenAI
provider = OpenAIProvider(
    model_id="gpt-4-turbo-preview",
    api_key="your-key",
    max_retries=4,
    base_delay=1.0,
    max_delay=8.0,
    timeout=60.0,
    max_connections=100
)
```

## Model Pricing (as of 2024)

### Anthropic Claude 3

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| claude-3-opus | $0.015 | $0.075 |
| claude-3-sonnet | $0.003 | $0.015 |
| claude-3-haiku | $0.00025 | $0.00125 |

### OpenAI GPT

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| gpt-4-turbo | $0.01 | $0.03 |
| gpt-4 | $0.03 | $0.06 |
| gpt-3.5-turbo | $0.0005 | $0.0015 |

## Routing Strategies

### 1. PRIORITY
Uses providers in priority order (1 = highest). Falls back to next provider on failure.

**Use case**: Preferred provider with fallbacks

```python
router.register_provider("anthropic", provider1, priority=1)  # Try first
router.register_provider("openai", provider2, priority=2)     # Fallback
```

### 2. LEAST_COST
Routes to the cheapest provider for the request.

**Use case**: Cost optimization

```python
router = LLMRouter(strategy=RoutingStrategy.LEAST_COST)
# Automatically selects cheapest model
```

### 3. LEAST_LATENCY
Routes to the fastest provider based on historical latency.

**Use case**: Low-latency applications

```python
router = LLMRouter(strategy=RoutingStrategy.LEAST_LATENCY)
# Tracks avg latency per provider
```

### 4. ROUND_ROBIN
Distributes requests evenly across all healthy providers.

**Use case**: Load balancing

```python
router = LLMRouter(strategy=RoutingStrategy.ROUND_ROBIN)
# Request 1 → Provider A, Request 2 → Provider B, etc.
```

### 5. RANDOM
Randomly selects from healthy providers.

**Use case**: Simple distribution

```python
router = LLMRouter(strategy=RoutingStrategy.RANDOM)
```

## Cost Tracking & Reporting

### Budget Management

```python
# Set budget
tracker.set_budget("tenant-123", monthly_limit_usd=1000.0)

# Check status
status = tracker.check_budget("tenant-123")
print(f"Used: {status.percentage_used:.1f}%")
print(f"Remaining: ${status.remaining_usd:.2f}")
print(f"Projected: ${status.projected_monthly_cost:.2f}")

# Alert triggered?
if status.alert_triggered:
    print(f"Alert at {status.alert_level * 100:.0f}%!")
```

### Cost Summaries

```python
# Get summary for time period
summary = tracker.get_summary(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    tenant_id="tenant-123"  # Optional filter
)

print(f"Total cost: ${summary.total_cost_usd:.2f}")
print(f"Total tokens: {summary.total_tokens:,}")
print(f"Total requests: {summary.total_requests}")
print(f"By provider: {summary.breakdown_by_provider}")
print(f"By agent: {summary.breakdown_by_agent}")
```

### Time Series Data

```python
# Get daily cost time series
time_series = tracker.get_time_series(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    granularity="daily",  # hourly | daily | monthly
    tenant_id="tenant-123"
)

for timestamp, cost in time_series:
    print(f"{timestamp.date()}: ${cost:.2f}")
```

### Export Reports

```python
# Export to CSV
tracker.export_csv(
    "cost_report.csv",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

# Export to JSON (with summary)
tracker.export_json(
    "cost_report.json",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    include_summary=True
)
```

## Monitoring & Metrics

### Router Metrics

```python
metrics = router.get_metrics()

# Global metrics
print(f"Total requests: {metrics['global']['total_requests']}")
print(f"Successful: {metrics['global']['successful_requests']}")
print(f"Failed: {metrics['global']['failed_requests']}")
print(f"Failovers: {metrics['global']['failover_count']}")
print(f"Total cost: ${metrics['global']['total_cost_usd']:.2f}")

# Per-provider metrics
for name, pmetrics in metrics['providers'].items():
    print(f"\n{name}:")
    print(f"  Requests: {pmetrics['total_requests']}")
    print(f"  Success rate: {pmetrics['success_rate']:.1f}%")
    print(f"  Avg latency: {pmetrics['avg_latency_ms']:.0f}ms")
    print(f"  Total cost: ${pmetrics['total_cost_usd']:.4f}")
    print(f"  Circuit breaker: {pmetrics['circuit_breaker_state']}")
```

### Health Checks

```python
# Manual health check
health_results = await router.health_check_all()

for provider_name, health in health_results.items():
    print(f"{provider_name}: {'healthy' if health.is_healthy else 'unhealthy'}")
    if health.latency_ms:
        print(f"  Latency: {health.latency_ms:.0f}ms")
    if health.last_error:
        print(f"  Error: {health.last_error}")

# Automatic background monitoring
await router.start_health_monitoring()  # Checks every 30s
# ... do work ...
await router.stop_health_monitoring()
```

## Error Handling

```python
from llm.providers.base_provider import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ServiceUnavailableError
)
from llm.circuit_breaker import CircuitBreakerOpenError

try:
    response = await router.generate(request)
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after}s")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except CircuitBreakerOpenError as e:
    print(f"Circuit breaker open. Retry after {e.retry_after}s")
except ProviderError as e:
    print(f"Provider error: {e} (retryable: {e.retryable})")
```

## Best Practices

### 1. Use Least-Cost Strategy for Production
```python
router = LLMRouter(strategy=RoutingStrategy.LEAST_COST)
```

### 2. Enable Circuit Breakers
```python
router = LLMRouter(enable_circuit_breaker=True)
```

### 3. Set Budget Limits
```python
tracker.set_budget("tenant-123", monthly_limit_usd=1000.0)
```

### 4. Register Alert Callbacks
```python
tracker.register_alert_callback(send_slack_alert)
```

### 5. Monitor Health
```python
await router.start_health_monitoring()
```

### 6. Export Regular Reports
```python
# Daily export
tracker.export_csv(f"costs_{date.today()}.csv")
```

### 7. Use Cheaper Models When Possible
```python
# Claude Sonnet is 5x cheaper than Opus
provider = AnthropicProvider(model_id="claude-3-sonnet-20240229")
```

### 8. Track Costs per Tenant
```python
tracker.track_usage(
    provider=response.provider,
    tenant_id=tenant_id,  # Always include tenant_id
    agent_id=agent_id,
    model_id=response.model_id,
    usage=response.usage
)
```

## Testing

Run the comprehensive example:

```bash
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key
python example_integrated.py
```

## Files Structure

```
llm/
├── __init__.py                 # Package exports
├── README.md                   # This file
├── circuit_breaker.py          # Circuit breaker implementation
├── cost_tracker.py             # Cost tracking and budgets
├── llm_router.py               # Multi-provider routing
├── rate_limiter.py             # Rate limiting (existing)
├── example_integrated.py       # Complete integration example
└── providers/
    ├── __init__.py
    ├── base_provider.py        # Abstract base class
    ├── anthropic_provider.py   # Anthropic Claude integration
    └── openai_provider.py      # OpenAI GPT integration
```

## License

Copyright 2025 GreenLang. All rights reserved.
