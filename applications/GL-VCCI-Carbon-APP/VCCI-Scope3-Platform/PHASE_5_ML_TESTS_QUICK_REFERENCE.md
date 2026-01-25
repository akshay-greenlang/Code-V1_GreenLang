# Phase 5 ML Tests - Quick Reference Guide

## Test File Structure

```
GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/
├── entity_mdm/ml/tests/
│   ├── __init__.py
│   ├── conftest.py                    (487 lines - fixtures)
│   ├── test_config.py                 (270 lines - 12 tests)
│   ├── test_embeddings.py             (337 lines - 15 tests)
│   ├── test_vector_store.py           (424 lines - 18 tests)
│   ├── test_matching_model.py         (355 lines - 20 tests)
│   ├── test_resolver.py               (469 lines - 25 tests)
│   ├── test_evaluation.py             (314 lines - 15 tests)
│   └── test_training.py               (346 lines - 15 tests)
│
└── utils/ml/tests/
    ├── __init__.py
    ├── conftest.py                    (366 lines - fixtures)
    ├── test_config.py                 (160 lines - 10 tests)
    ├── test_spend_classification.py   (364 lines - 30 tests)
    ├── test_llm_client.py             (313 lines - 15 tests)
    ├── test_rules_engine.py           (374 lines - 20 tests)
    └── test_evaluation.py             (383 lines - 15 tests)
```

## Quick Stats

- **Total Files:** 14 (2 conftest + 12 test files)
- **Total Tests:** 191 (109 Entity Resolution + 82 Spend Classification)
- **Total Lines:** 5,093 lines
- **Coverage Target:** 90%+

## Running Tests

### All Tests
```bash
# Run all ML tests
pytest entity_mdm/ml/tests/ utils/ml/tests/ -v

# With coverage
pytest entity_mdm/ml/tests/ utils/ml/tests/ -v --cov --cov-report=html
```

### Entity Resolution Tests
```bash
# All entity resolution tests
pytest entity_mdm/ml/tests/ -v

# Specific test file
pytest entity_mdm/ml/tests/test_resolver.py -v

# Specific test
pytest entity_mdm/ml/tests/test_resolver.py::TestStage1CandidateGeneration::test_candidate_generation_returns_similar_entities -v
```

### Spend Classification Tests
```bash
# All spend classification tests
pytest utils/ml/tests/ -v

# Specific test file
pytest utils/ml/tests/test_spend_classification.py -v

# Specific test
pytest utils/ml/tests/test_spend_classification.py::TestLLMClassification::test_classify_business_travel -v
```

## Key Test Fixtures (conftest.py)

### Entity Resolution Fixtures
- `mock_weaviate_client` - Mock Weaviate vector database
- `mock_sentence_transformer` - Mock embedding model
- `mock_cross_encoder` - Mock BERT re-ranking model
- `sample_suppliers` - 100+ supplier variations
- `sample_supplier_pairs` - Labeled training pairs
- `entity_mdm_config` - Configuration fixture

### Spend Classification Fixtures
- `mock_llm_client` - Mock LLM (OpenAI/Anthropic)
- `mock_rule_engine` - Mock rules engine
- `scope3_categories` - All 15 Scope 3 categories
- `sample_procurement_descriptions` - 200+ procurement examples
- `spend_classification_config` - Configuration fixture

## Test Coverage Map

### Entity Resolution ML (109 tests)

| Module | Tests | Key Coverage |
|--------|-------|--------------|
| Configuration | 12 | Env vars, validation, defaults |
| Embeddings | 15 | Generation, caching, normalization |
| Vector Store | 18 | CRUD, search, filtering |
| Matching Model | 20 | Training, inference, persistence |
| Resolver | 25 | Two-stage, thresholds, integration |
| Evaluation | 15 | Metrics, confusion matrix, ROC |
| Training | 15 | Pipeline, checkpointing, hyperparams |

### Spend Classification ML (82 tests)

| Module | Tests | Key Coverage |
|--------|-------|--------------|
| Configuration | 10 | LLM settings, thresholds |
| Classification | 30 | LLM, rules, routing, batch |
| LLM Client | 15 | API calls, caching, retry |
| Rules Engine | 20 | 15 categories, scoring |
| Evaluation | 15 | Accuracy, per-category, comparison |

## Common Test Patterns

### Testing with Mocks
```python
def test_example(mock_weaviate_client, mock_sentence_transformer):
    """Test with mocked dependencies."""
    service = MyService(mock_weaviate_client, mock_sentence_transformer)
    result = service.some_method()
    assert result is not None
```

### Testing Configuration
```python
def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError, match="must be between"):
        Config({"threshold": 1.5})
```

### Testing Batch Operations
```python
def test_batch_processing(service):
    """Test batch processing."""
    items = ["item1", "item2", "item3"]
    results = service.process_batch(items)
    assert len(results) == 3
```

### Testing Edge Cases
```python
def test_empty_input(service):
    """Test empty input handling."""
    with pytest.raises(ValueError, match="cannot be empty"):
        service.process("")
```

## Debugging Failed Tests

### Run with verbose output
```bash
pytest entity_mdm/ml/tests/test_resolver.py -v -s
```

### Run specific failed test
```bash
pytest entity_mdm/ml/tests/test_resolver.py::TestStage1CandidateGeneration::test_candidate_generation_returns_similar_entities -v
```

### Show local variables on failure
```bash
pytest entity_mdm/ml/tests/ -l
```

### Stop on first failure
```bash
pytest entity_mdm/ml/tests/ -x
```

## Integration with Production Code

### Expected Module Structure

**Entity Resolution:**
```
entity_mdm/ml/
├── __init__.py
├── config.py              → EntityMDMConfig
├── embeddings.py          → EmbeddingService
├── vector_store.py        → VectorStore
├── matching_model.py      → MatchingModel
├── resolver.py            → EntityResolver
├── evaluation.py          → EntityMatchingEvaluator
└── training.py            → TrainingPipeline
```

**Spend Classification:**
```
utils/ml/
├── __init__.py
├── config.py              → SpendClassificationConfig
├── llm_client.py          → LLMClient
├── rules_engine.py        → RulesEngine
├── spend_classification.py → SpendClassificationService
└── evaluation.py          → SpendClassificationEvaluator
```

## Coverage Report Generation

```bash
# Generate HTML coverage report
pytest entity_mdm/ml/tests/ utils/ml/tests/ --cov=entity_mdm/ml --cov=utils/ml --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: ML Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-test.txt
      - run: pytest entity_mdm/ml/tests/ utils/ml/tests/ --cov --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Required Dependencies

```txt
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
numpy==1.24.3
```

## Key Metrics to Monitor

1. **Test Coverage:** Target 90%+
2. **Test Execution Time:** All tests should complete < 5 minutes
3. **Flaky Tests:** Zero tolerance for non-deterministic tests
4. **Test Failures:** Immediate investigation for CI failures

## Support & Documentation

- **Full Delivery Report:** `PHASE_5_ML_TEST_SUITE_DELIVERY.md`
- **Test Files:** `entity_mdm/ml/tests/` and `utils/ml/tests/`
- **Fixtures:** Review `conftest.py` in each test directory
- **Coverage:** Run with `--cov-report=html` flag

---

**Status:** ✅ All 14 test files delivered and ready for execution
**Last Updated:** November 6, 2025
