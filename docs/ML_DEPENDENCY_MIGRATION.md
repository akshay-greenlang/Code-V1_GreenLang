# ML Dependency Migration Guide

## Overview

**Version:** 0.3.0+
**Impact:** Breaking change for ML/AI users
**Benefit:** 250MB reduction in installation size

Starting in version 0.3.0, GreenLang has made ML and vector database dependencies **optional extras** to reduce installation bloat. This change saves 250MB for users who don't need ML capabilities.

## What Changed?

### Before (v0.2.x and earlier)
```bash
pip install greenlang-cli
# Installed everything including 250MB of ML dependencies
```

### After (v0.3.0+)
```bash
# Minimal install (no ML)
pip install greenlang-cli

# With ML capabilities
pip install greenlang-cli[ml]
```

## Who is Affected?

You're affected if you use any of these features:

1. **Entity Resolution ML Pipeline**
   - `greenlang.services.entity_mdm.ml.embeddings.EmbeddingPipeline`
   - `greenlang.services.entity_mdm.ml.vector_store.VectorStore`
   - `greenlang.services.entity_mdm.ml.matching_model`

2. **Semantic Caching**
   - `greenlang.intelligence.semantic_cache`

3. **RAG Systems**
   - Any vector database integrations
   - Document embedding pipelines

4. **Custom ML Models**
   - Any code importing `torch`, `transformers`, `sentence-transformers`
   - Any code using vector databases (Weaviate, ChromaDB, Pinecone, Qdrant, FAISS)

## Migration Steps

### Step 1: Identify Your Usage

Check if you're using ML features:

```bash
# Run the ML feature detection utility
python -m greenlang.utils.ml_imports
```

Output example:
```
GreenLang ML Feature Availability
================================================================================
torch                     ✗ Not installed
sentence_transformers     ✗ Not installed
transformers             ✗ Not installed
weaviate                 ✗ Not installed
chromadb                 ✗ Not installed
pinecone                 ✗ Not installed
qdrant                   ✗ Not installed
faiss                    ✗ Not installed

================================================================================

To install ML features:
  pip install greenlang-cli[ml]

To install vector databases:
  pip install greenlang-cli[vector-db]

To install all AI features:
  pip install greenlang-cli[ai-full]
```

### Step 2: Choose Your Installation Profile

Based on your needs, choose one of these installation profiles:

#### Profile A: No ML Features Needed
**Use cases:**
- Calculation engines only
- Regulatory compliance reporting
- Data intake and validation
- Basic LLM integration (OpenAI, Anthropic APIs)

**Action:** No changes needed, continue using minimal install
```bash
pip install greenlang-cli
```

#### Profile B: ML Features Required
**Use cases:**
- Entity resolution with embeddings
- Semantic similarity matching
- Custom ML models with PyTorch
- Transformer-based NLP

**Action:** Install ML extras
```bash
pip install greenlang-cli[ml]
```

This adds:
- PyTorch (~100MB)
- sentence-transformers (~50MB)
- transformers (~30MB)
- scikit-learn (~20MB)

#### Profile C: Vector Databases Required
**Use cases:**
- Vector similarity search
- Semantic caching
- Knowledge graph embeddings
- RAG systems

**Action:** Install vector-db extras
```bash
pip install greenlang-cli[vector-db]
```

This adds:
- Weaviate client
- ChromaDB
- Pinecone client
- Qdrant client
- FAISS (~30MB)

#### Profile D: Full AI Capabilities
**Use cases:**
- Complete entity resolution pipeline
- Advanced RAG systems
- Semantic search + embeddings
- Production ML deployments

**Action:** Install ai-full extras
```bash
pip install greenlang-cli[ai-full]
```

This combines `[llm,ml,vector-db]` for complete AI functionality.

### Step 3: Update Your Installation

#### For Direct Installations
```bash
# Uninstall old version
pip uninstall greenlang-cli

# Reinstall with appropriate extras
pip install greenlang-cli[ai-full]
```

#### For requirements.txt
```diff
# Before
greenlang-cli>=0.2.0

# After (choose based on your profile)
greenlang-cli[ai-full]>=0.3.0
```

#### For pyproject.toml
```diff
# Before
dependencies = [
    "greenlang-cli>=0.2.0",
]

# After (choose based on your profile)
dependencies = [
    "greenlang-cli[ai-full]>=0.3.0",
]
```

#### For Dockerfile
```diff
# Before
FROM python:3.11-slim
RUN pip install greenlang-cli

# After (choose based on your profile)
FROM python:3.11-slim
RUN pip install greenlang-cli[ai-full]
```

### Step 4: Update Application Code

No code changes are required. The runtime checks will provide clear error messages if dependencies are missing.

Example error message:
```
================================================================================
Missing Optional Dependency: torch
================================================================================

Entity Resolution Embedding Pipeline requires the 'torch' package.

To install it, run:
  pip install greenlang-cli[ml]

Or install all AI capabilities:
  pip install greenlang-cli[ai-full]
================================================================================
```

### Step 5: Update CI/CD Pipelines

#### GitHub Actions
```yaml
# Before
- name: Install dependencies
  run: pip install greenlang-cli

# After
- name: Install dependencies
  run: pip install greenlang-cli[ai-full]
```

#### GitLab CI
```yaml
# Before
install:
  script:
    - pip install greenlang-cli

# After
install:
  script:
    - pip install greenlang-cli[ai-full]
```

#### Docker Compose
```yaml
# Before
services:
  app:
    build: .
    environment:
      - PIP_INSTALL_ARGS=

# After
services:
  app:
    build: .
    environment:
      - PIP_INSTALL_ARGS=[ai-full]
```

## Testing Your Migration

### 1. Test ML Features
```python
from greenlang.utils.ml_imports import get_available_ml_features

features = get_available_ml_features()
print(f"PyTorch available: {features['torch']['available']}")
print(f"Sentence Transformers available: {features['sentence_transformers']['available']}")
```

### 2. Test Entity Resolution
```python
try:
    from greenlang.services.entity_mdm.ml.embeddings import EmbeddingPipeline
    pipeline = EmbeddingPipeline()
    print("✓ Entity resolution ML available")
except ImportError as e:
    print(f"✗ ML dependencies missing: {e}")
```

### 3. Test Vector Databases
```python
try:
    from greenlang.services.entity_mdm.ml.vector_store import VectorStore
    store = VectorStore()
    print("✓ Vector store available")
except ImportError as e:
    print(f"✗ Vector DB dependencies missing: {e}")
```

## Troubleshooting

### Issue: Import errors after upgrade
**Symptom:**
```python
ImportError: Missing Optional Dependency: torch
```

**Solution:**
```bash
pip install greenlang-cli[ml]
```

### Issue: Vector database connection fails
**Symptom:**
```python
ImportError: Missing Optional Dependency: weaviate-client
```

**Solution:**
```bash
pip install greenlang-cli[vector-db]
```

### Issue: Large Docker images
**Problem:** Docker image still 1GB+ after migration

**Solution:** Use multi-stage builds
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
RUN pip install --user greenlang-cli[ai-full]

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
CMD ["gl"]
```

### Issue: CI/CD builds failing
**Problem:** Tests fail with missing dependencies

**Solution:** Update CI configuration
```yaml
# Add explicit ML installation
- run: pip install greenlang-cli[ai-full]
# Or install only what's needed for tests
- run: pip install greenlang-cli[test,ml]
```

## Rollback Instructions

If you need to rollback to the old behavior:

```bash
# Install version 0.2.x
pip install greenlang-cli==0.2.999

# Or pin to last known good version
pip install greenlang-cli==0.2.15
```

Note: This is not recommended as v0.2.x will not receive security updates.

## Performance Impact

### Installation Time
- **Minimal install:** 30 seconds (vs 3-5 minutes with ML)
- **With ML extras:** 3-5 minutes (same as before)

### Disk Space
- **Minimal install:** ~50MB (vs 300MB with ML)
- **With ML extras:** ~300MB (same as before)

### Runtime Performance
- **No impact:** Runtime performance is identical
- **Lazy loading:** ML modules only loaded when used

## Available Extras Reference

| Extra | Size | Dependencies | Use Case |
|-------|------|--------------|----------|
| `cli` | 5MB | click, rich | CLI enhancements |
| `analytics` | 30MB | pandas, numpy | Data analysis |
| `data` | 50MB | pandas, numpy, openpyxl, sqlalchemy | Data processing |
| `llm` | 20MB | openai, langchain, anthropic | LLM integration |
| `ml` | 200MB | torch, transformers, sentence-transformers | ML models |
| `vector-db` | 50MB | weaviate, chromadb, pinecone, qdrant, faiss | Vector search |
| `server` | 30MB | fastapi, uvicorn, celery, redis | Web services |
| `security` | 10MB | cryptography, PyJWT | Auth and encryption |
| `test` | 20MB | pytest, coverage | Testing framework |
| `dev` | 30MB | mypy, ruff, black, pre-commit | Development tools |
| `doc` | 15MB | mkdocs, mkdocs-material | Documentation |
| `ai-full` | 250MB | Combines llm, ml, vector-db | Complete AI stack |
| `full` | 100MB | All except ai-full | All features (no ML) |
| `all` | 350MB | Everything | Maximum functionality |

## FAQ

### Q: Why make ML dependencies optional?
**A:** The majority of GreenLang users (70%+) use only calculation engines and regulatory frameworks. Forcing 250MB of ML dependencies on everyone was wasteful.

### Q: Will this break my existing code?
**A:** No. Existing code continues to work. You'll just need to install the appropriate extras.

### Q: What if I forget to install ML extras?
**A:** You'll get a clear error message with installation instructions the moment you try to use ML features.

### Q: Can I install multiple extras?
**A:** Yes! Combine them: `pip install greenlang-cli[ml,vector-db,server]`

### Q: What about legacy requirements.txt?
**A:** The old requirements.txt still exists for backwards compatibility, but ML dependencies are commented out with migration instructions.

### Q: How do I check what's installed?
**A:** Run `python -m greenlang.utils.ml_imports` to see all ML feature availability.

## Support

If you encounter issues during migration:

1. Check the [Installation Guide](installation.md)
2. Run `gl doctor` to diagnose issues
3. Check [GitHub Issues](https://github.com/greenlang/greenlang/issues)
4. Join [Discord](https://discord.gg/greenlang) for help

## Timeline

- **v0.3.0 (Current):** ML dependencies made optional
- **v0.4.0 (Planned):** Further dependency optimization
- **v0.5.0 (Planned):** Plugin system for ML backends

## Related Documentation

- [Installation Guide](installation.md)
- [ML Features Documentation](ml/README.md)
- [Entity Resolution Guide](entity-resolution.md)
- [Vector Database Integration](vector-databases.md)
