# Demo Corpus for INTL-104 RAG v1

This directory contains a demonstration corpus for testing and documenting the GreenLang RAG (Retrieval-Augmented Generation) system.

## Contents

### Documents

1. **ghg_protocol_summary.md** (Markdown)
   - Summary of GHG Protocol Corporate Standard
   - Topics: Scope 1/2/3 emissions, accounting principles, emission factors
   - Length: ~5,800 tokens across 12 chunks
   - Purpose: Demonstrates technical climate documentation ingestion

2. **climate_finance_mechanisms.md** (Markdown)
   - Overview of climate finance for developing nations
   - Topics: Green Climate Fund, Adaptation Fund, blended finance
   - Length: ~8,400 tokens across 18 chunks
   - Purpose: Demonstrates long-form policy document ingestion

3. **carbon_offset_standards.pdf** (PDF)
   - Guide to carbon offset standards and best practices
   - Topics: VCS, Gold Standard, project types, quality criteria
   - Length: 6 pages, ~6,700 tokens across 15 chunks
   - Purpose: Demonstrates PDF ingestion and extraction

### Metadata

4. **MANIFEST.json**
   - Ingestion manifest with document metadata
   - Schema version: 1.0.0
   - Contains: doc IDs, hashes, chunk counts, tags, validation info
   - Purpose: Demonstrates audit trail and reproducibility

### Utilities

5. **create_sample_pdf.py**
   - Script to generate the sample PDF document
   - Uses reportlab library
   - Can be regenerated with: `python create_sample_pdf.py`

## Usage

### Ingestion

Ingest the demo corpus using the GreenLang CLI:

```bash
# Start Weaviate
gl rag up

# Ingest markdown documents
gl rag ingest demo_corpus_w1 artifacts/W1/ghg_protocol_summary.md --title "GHG Protocol Corporate Standard Summary"
gl rag ingest demo_corpus_w1 artifacts/W1/climate_finance_mechanisms.md --title "Climate Finance Mechanisms"

# Ingest PDF document
gl rag ingest demo_corpus_w1 artifacts/W1/carbon_offset_standards.pdf --title "Carbon Offset Standards"
```

### Query

Query the ingested documents:

```bash
# Query about GHG scopes
gl rag query "What are Scope 1 emissions?" --collections demo_corpus_w1 --top-k 3

# Query about climate finance
gl rag query "How does the Green Climate Fund work?" --collections demo_corpus_w1 --top-k 5

# Query about carbon offsets
gl rag query "What are the quality criteria for carbon offsets?" --collections demo_corpus_w1 --top-k 3
```

### Programmatic Usage

Using Python API:

```python
from greenlang.intelligence.rag import ingest, query, RAGConfig

# Configure RAG
config = RAGConfig(
    mode="live",
    embedding_provider="minilm",
    vector_store_provider="weaviate",
    retrieval_method="mmr",
)

# Ingest documents
result = ingest.ingest_path(
    path="artifacts/W1/ghg_protocol_summary.md",
    collection="demo_corpus_w1",
    config=config,
)
print(f"Ingested {result['num_chunks']} chunks")

# Query
results = query.query(
    q="What are Scope 1 emissions?",
    top_k=3,
    collections=["demo_corpus_w1"],
    config=config,
)

for result in results:
    print(f"[{result.relevance_score:.2f}] {result.citation.formatted}")
    print(f"Text: {result.text[:200]}...")
    print()
```

## Testing

This corpus is used in DoD-required tests:

1. **MMR Diversity Test** (`tests/rag/test_dod_requirements.py`)
   - Verifies MMR retrieval produces diverse results
   - Uses synthetic near-duplicate corpus

2. **Ingest Round-trip Test** (`tests/rag/test_dod_requirements.py`)
   - Ingests demo documents and verifies hash integrity
   - Ensures chunk IDs are stable UUID v5 values

3. **Network Isolation Test** (`tests/rag/test_dod_requirements.py`)
   - Verifies replay mode blocks network calls
   - Uses cached results only

## Corpus Statistics

- **Total documents**: 3
- **Total chunks**: 45
- **Total tokens**: ~21,000
- **Document types**: 2 markdown, 1 PDF
- **Languages**: English
- **Date range**: 2015-03-24 to 2025-10-03

## Topics Covered

### GHG Accounting
- Scope 1, 2, and 3 classifications
- Emission factors (natural gas, diesel, electricity)
- Organizational boundaries (equity share vs. control)
- Verification and reporting requirements

### Climate Finance
- Green Climate Fund mechanisms
- Adaptation Fund and Direct Access
- Climate Investment Funds (CTF, SCF)
- Blended finance approaches
- Accessing climate finance step-by-step

### Carbon Offsets
- Renewable energy projects
- Forestry and land use projects
- Verification standards (VCS, Gold Standard)
- Quality criteria (additionality, permanence, verification)
- Best practices for corporate buyers

## Compliance

This demo corpus supports INTL-104 DoD requirements:

- ✅ Demonstrates PDF ingestion with table extraction
- ✅ Demonstrates markdown ingestion with section hierarchy
- ✅ Provides MANIFEST.json with audit trail
- ✅ Includes diverse document types and topics
- ✅ Suitable for testing MMR retrieval diversity
- ✅ Enables hash verification and round-trip testing

## Notes

- All content is for educational and testing purposes
- Documents are derived from public climate standards and resources
- Suitable for demonstrating RAG capabilities to stakeholders
- Can be expanded with additional climate documents as needed

## References

- GHG Protocol: https://ghgprotocol.org
- Green Climate Fund: https://www.greenclimate.fund
- Verified Carbon Standard: https://verra.org/programs/verified-carbon-standard

---

**Created**: 2025-10-03
**Version**: 1.0
**Status**: Production-ready for INTL-104 demos
