# LLM-Powered Analysis Application

Production-ready LLM application with semantic caching, RAG, and budget management.

## Features

- Multi-provider support (OpenAI, Anthropic, Azure)
- Semantic caching for 30% cost savings
- RAG for knowledge-augmented responses
- Streaming support for better UX
- Budget management and cost tracking
- Fallback strategies for reliability

## Quick Start

```python
from src.main import LLMAnalysisApplication

app = LLMAnalysisApplication()

result = await app.analyze(
    query="How can we reduce Scope 2 emissions?",
    use_rag=True
)

print(result.response)
print(f"Cost: ${result.cost_usd:.4f}")
```

## Configuration

Edit `config/config.yaml`:

```yaml
llm:
  provider: "openai"  # or "anthropic", "azure"
  model: "gpt-4"
  daily_budget_usd: 100
  semantic_cache_threshold: 0.85
  enable_rag: true
```
