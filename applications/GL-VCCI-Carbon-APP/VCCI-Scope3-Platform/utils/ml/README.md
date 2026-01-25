# GL-VCCI Spend Classification ML System

**Version:** 1.0.0
**Phase:** 5 - Spend Classification ML
**Status:** COMPLETE

## Overview

Production-ready Machine Learning system for classifying procurement spend into 15 GHG Protocol Scope 3 categories. Combines LLM-based classification (OpenAI GPT-3.5/GPT-4, Anthropic Claude) with rule-based fallback for robust, cost-effective classification.

### Target Performance
- **Classification Accuracy:** ≥90%
- **Confidence Threshold:** 0.85
- **Human Review Threshold:** 0.5
- **Cache Hit Rate:** >70%
- **Average Latency:** <2s per classification

## Architecture

```
utils/ml/
├── __init__.py              (352 lines) - Module exports and convenience functions
├── config.py                (504 lines) - Pydantic configuration models
├── exceptions.py            (567 lines) - Custom exception hierarchy
├── llm_client.py            (630 lines) - LLM API client with caching
├── rules_engine.py          (548 lines) - Rule-based classifier
├── spend_classification.py  (583 lines) - Main classification logic
├── training_data.py         (578 lines) - Training data management
└── evaluation.py            (559 lines) - Evaluation framework

Total: 4,321 lines of production code
```

## Features

### 1. LLM-Based Classification
- **Multi-Provider Support:** OpenAI (GPT-3.5, GPT-4), Anthropic (Claude 3)
- **Prompt Engineering:** Optimized prompts for Scope 3 classification
- **Confidence Scoring:** Probabilistic confidence scores (0.0-1.0)
- **Structured Output:** JSON-formatted responses with reasoning
- **Retry Logic:** Exponential backoff (1s, 2s, 4s, 8s)
- **Cost Tracking:** Token usage and cost monitoring

### 2. Rule-Based Fallback
- **Keyword Matching:** Category-specific keyword dictionaries
- **Regex Patterns:** Advanced pattern matching
- **Fuzzy Matching:** Levenshtein distance-based matching (threshold: 0.8)
- **Multi-Evidence Aggregation:** Combines evidence from multiple rules
- **Confidence Scoring:** Rule-based confidence calculation

### 3. Caching & Performance
- **Redis Caching:** 30-day TTL for classifications
- **Batch Processing:** Concurrent classification (configurable batch size)
- **Token Caching:** 55-minute TTL for LLM tokens
- **Cost Optimization:** Cache hit rates >70%

### 4. Evaluation & Monitoring
- **Comprehensive Metrics:** Accuracy, Precision, Recall, F1-score
- **Confusion Matrix:** 15x15 category confusion matrix
- **Error Analysis:** Common misclassifications, category errors
- **Performance Comparison:** LLM vs rules performance
- **Export Capabilities:** JSON, CSV export

## Quick Start

### Installation

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

Required packages:
- `anthropic>=0.18.0` - Claude API
- `openai>=1.10.0` - OpenAI API
- `redis>=5.0.0` - Redis caching
- `pydantic>=2.5.0` - Configuration models
- `fuzzywuzzy>=0.18.0` - Fuzzy matching
- `scikit-learn>=1.3.0` - Evaluation metrics

### Configuration

Set environment variables in `.env`:
```bash
# LLM Provider
LLM_PROVIDER=openai  # or "anthropic"
LLM_MODEL_NAME=gpt-3.5-turbo  # or "claude-3-haiku-20240307"
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Redis
REDIS_URL=redis://localhost:6379/0

# Classification
ML_CONFIDENCE_THRESHOLD=0.85
```

### Basic Usage

```python
from utils.ml import SpendClassifier, MLConfig

# Initialize classifier
config = MLConfig()
classifier = SpendClassifier(config)

# Classify single spend
result = await classifier.classify(
    description="Office furniture purchase from IKEA",
    amount=5000.0,
    supplier="IKEA"
)

print(f"Category: {result.category_name}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Method: {result.method}")
print(f"Needs Review: {result.needs_human_review}")
print(f"Reasoning: {result.reasoning}")

# Close connections
await classifier.close()
```

### Batch Classification

```python
from utils.ml import classify_spend_batch

# Classify multiple spends
descriptions = [
    "Flight to customer meeting in NYC",
    "Monthly electricity bill payment",
    "Freight shipping to warehouse",
    "Laptop purchase for software engineer",
    "Office waste disposal service"
]

results = await classify_spend_batch(descriptions)

for desc, result in zip(descriptions, results):
    print(f"{desc[:40]:40s} → {result.category_name} ({result.confidence:.2%})")
```

### Advanced Configuration

```python
from utils.ml import MLConfig, LLMConfig, ClassificationConfig

config = MLConfig(
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=500,
        max_retries=3
    ),
    classification=ClassificationConfig(
        confidence_threshold=0.85,
        use_llm_primary=True,
        use_rules_fallback=True,
        require_human_review_threshold=0.5
    )
)

classifier = SpendClassifier(config)
```

## Scope 3 Categories

The system classifies spend into 15 GHG Protocol Scope 3 categories:

| Category | Name | Example Keywords |
|----------|------|------------------|
| 1 | Purchased Goods & Services | material, component, supplies, packaging |
| 2 | Capital Goods | equipment, machinery, building, furniture |
| 3 | Fuel and Energy Related | electricity, power, fuel, gas, energy |
| 4 | Upstream Transportation | freight, shipping, logistics, delivery |
| 5 | Waste Generated | waste, disposal, recycling, landfill |
| 6 | Business Travel | flight, hotel, taxi, rental car |
| 7 | Employee Commuting | commute, parking, public transit |
| 8 | Upstream Leased Assets | lease, rental, leased equipment |
| 9 | Downstream Transportation | customer delivery, outbound logistics |
| 10 | Processing of Sold Products | product processing, downstream processing |
| 11 | Use of Sold Products | product use, customer use |
| 12 | End-of-Life Treatment | product disposal, end of life, recycling |
| 13 | Downstream Leased Assets | lease to customer, downstream lease |
| 14 | Franchises | franchise, franchisee, royalty |
| 15 | Investments | investment, portfolio, equity, stock |

## Training & Evaluation

### Load Training Data

```python
from utils.ml import TrainingDataLoader

loader = TrainingDataLoader()

# Load from CSV
dataset = loader.load_csv("data/training/spend_labels.csv")

# Load from Excel
dataset = loader.load_excel("data/training/spend_labels.xlsx", sheet_name="training_data")

# Validate dataset
validation_report = loader.validate_dataset(dataset, min_samples=100)
print(f"Valid: {validation_report['valid']}")
```

### Split Dataset

```python
# Stratified train/val/test split
train, val, test = loader.split_dataset(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True
)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

### Augment Dataset

```python
# Augment imbalanced categories
augmented = loader.augment_dataset(
    dataset,
    min_samples_per_category=100
)
```

### Evaluate Model

```python
from utils.ml import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate on test set
results = await evaluator.evaluate_classifier(classifier, test)

# Print summary
evaluator.print_summary(results)

# Export results
evaluator.export_results(results, "evaluation_results.json")
```

### Error Analysis

```python
# Analyze errors
y_true = [ex.category for ex in test.examples]
y_pred = [result.category for result in results]

error_analysis = evaluator.analyze_errors(y_true, y_pred, predictions)

print(f"Total Errors: {error_analysis.total_errors}")
print(f"Error Rate: {error_analysis.error_rate:.2%}")
print(f"Common Errors: {error_analysis.common_errors[:5]}")
```

## Classification Strategy

### Hybrid Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    Spend Description                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   LLM Classification    │
         │   (OpenAI/Anthropic)    │
         └─────────────┬───────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Confidence ≥    │
              │     0.85?      │
              └────┬───────┬───┘
                   │ YES   │ NO
                   │       │
                   ▼       ▼
            ┌──────────┐  ┌────────────────┐
            │  Accept  │  │ Rules Fallback │
            │   LLM    │  │  Classification │
            └────┬─────┘  └────────┬────────┘
                 │                 │
                 │                 ▼
                 │        ┌────────────────┐
                 │        │ Choose Better  │
                 │        │   Confidence   │
                 │        └────────┬────────┘
                 │                 │
                 └─────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Confidence <   │
                  │     0.5?       │
                  └────┬───────┬───┘
                       │ YES   │ NO
                       │       │
                       ▼       ▼
              ┌──────────────┐  ┌──────────┐
              │ Flag for     │  │  Accept  │
              │ Human Review │  │  Result  │
              └──────────────┘  └──────────┘
```

### Confidence Thresholds

- **High Confidence (≥0.85):** Accept LLM classification
- **Medium Confidence (0.5-0.85):** Use rules fallback or flag for review
- **Low Confidence (<0.5):** Flag for human review

## Cost Tracking

```python
# Get cost summary
classifier = SpendClassifier(config)

# ... perform classifications ...

cost_summary = classifier.llm_client.get_cost_summary()
print(f"Total Requests: {cost_summary['total_requests']}")
print(f"Total Tokens: {cost_summary['total_tokens']:,}")
print(f"Total Cost: ${cost_summary['total_cost_usd']:.4f}")
print(f"Avg Cost/Request: ${cost_summary['avg_cost_per_request']:.6f}")
```

## Performance Metrics

```python
# Get classification metrics
metrics = classifier.get_metrics()

print(f"Total Classifications: {metrics['total_classifications']}")
print(f"LLM Classifications: {metrics['llm_classifications']} ({metrics['llm_percentage']:.1f}%)")
print(f"Rules Classifications: {metrics['rules_classifications']} ({metrics['rules_percentage']:.1f}%)")
print(f"Human Review Rate: {metrics['review_rate']:.1f}%")
```

## Integration Points

### With ERP Connectors (Phase 4)
```python
from connectors.erp import SAPConnector
from utils.ml import SpendClassifier

# Get spend data from SAP
sap = SAPConnector(config)
spend_items = await sap.get_procurement_data(start_date="2024-01-01")

# Classify spend
classifier = SpendClassifier(ml_config)
for item in spend_items:
    result = await classifier.classify(
        description=item.description,
        amount=item.amount,
        supplier=item.supplier
    )
    # Store classification result
    item.scope3_category = result.category
    item.classification_confidence = result.confidence
```

### With Entity MDM (Phase 2)
```python
from entity_mdm import EntityResolver
from utils.ml import SpendClassifier

# Resolve supplier entity
resolver = EntityResolver(config)
supplier_entity = await resolver.resolve_supplier(supplier_name)

# Classify with enriched context
result = await classifier.classify(
    description=description,
    supplier=supplier_entity.canonical_name,
    additional_context={
        "supplier_lei": supplier_entity.lei,
        "supplier_industry": supplier_entity.industry
    }
)
```

## SOC 2 Compliance

### Audit Logging
All classification events are logged with:
- Input description (sanitized)
- Classification result
- Confidence score
- Method used (LLM/rules/hybrid)
- Timestamp
- User/session ID (if available)

### Data Privacy
- Sensitive data sanitization
- Encrypted API communications (HTTPS)
- Redis data encryption at rest (if configured)
- API key protection (SecretStr)

## Troubleshooting

### Common Issues

**Issue:** LLM rate limiting
```python
# Solution: Increase retry delay
config.llm.retry_delay_seconds = 2.0
config.llm.max_retries = 5
```

**Issue:** Low cache hit rate
```python
# Solution: Increase cache TTL
config.cache.ttl_seconds = 2592000  # 30 days
```

**Issue:** High cost
```python
# Solution: Use rules-first approach
config.classification.use_llm_primary = False
config.classification.use_rules_fallback = True
```

**Issue:** Low accuracy
```python
# Solution: Lower confidence threshold or use LLM
config.classification.confidence_threshold = 0.75
config.classification.use_llm_primary = True
```

## Testing

```python
import pytest
from utils.ml import SpendClassifier, MLConfig

@pytest.mark.asyncio
async def test_spend_classification():
    config = MLConfig()
    classifier = SpendClassifier(config)

    result = await classifier.classify("Office furniture purchase")

    assert result.category == "category_1_purchased_goods_services"
    assert result.confidence >= 0.5
    assert not result.needs_human_review

    await classifier.close()
```

## Future Enhancements

1. **Fine-tuning:** Fine-tune LLMs on labeled procurement data
2. **Active Learning:** Use human review feedback to improve model
3. **Multi-language:** Support non-English descriptions
4. **Industry-specific:** Industry-specific classification rules
5. **Real-time Learning:** Update rules based on classification patterns

## Support

For issues or questions:
- **Documentation:** See inline docstrings and code comments
- **Examples:** See `examples/ml_classification.py`
- **Tests:** See `tests/utils/ml/`

## License

Copyright © 2024 GreenLang. All rights reserved.

---

**Phase 5 Implementation Complete**
**Status:** Production Ready
**Total Code:** 4,321 lines
**Target Accuracy:** ≥90%
