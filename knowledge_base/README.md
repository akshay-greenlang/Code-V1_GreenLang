# GreenLang Knowledge Base

**Purpose:** Curated knowledge repository for intelligent agent reasoning via RAG (Retrieval-Augmented Generation).

**Status:** ✅ Production-ready with 7 foundational documents across 3 collections

---

## Collections

### 1. **ghg_protocol_corp** (GHG Protocol Documentation)
Comprehensive greenhouse gas accounting and reporting standards.

**Documents:**
- `01_overview.txt` - GHG Protocol principles and framework overview
- `02_scopes.txt` - Scope 1, 2, and 3 emission categories with examples
- `03_emission_factors.txt` - Reference emission factors for common fuels and activities

**Use Cases:**
- Emission calculation agents
- Compliance checking
- Reporting guidance
- Methodology validation

---

### 2. **technology_database** (Decarbonization Technologies)
Technical specifications and guidance for industrial decarbonization technologies.

**Documents:**
- `01_heat_pumps.txt` - Industrial heat pump technology, COP, economics, applications
- `02_solar_thermal.txt` - Solar thermal systems, CSP, storage, site selection
- `03_cogeneration_chp.txt` - Combined heat and power, efficiency, economic analysis

**Use Cases:**
- Technology recommendation agents
- Feasibility analysis
- Technology comparison
- Equipment sizing

---

### 3. **case_studies** (Real-World Implementations)
Detailed case studies of successful decarbonization projects.

**Documents:**
- `01_industrial_case_studies.txt` - Three comprehensive case studies:
  - Food processing plant heat pump (520 tons CO2/yr reduction)
  - Chemical manufacturer solar thermal (760 tons CO2/yr reduction)
  - Steel mill waste heat recovery (2,170 tons CO2/yr reduction)

**Use Cases:**
- Lessons learned retrieval
- ROI benchmarking
- Success factor identification
- Risk mitigation strategies

---

## Ingestion

### Prerequisites
```bash
pip install sentence-transformers faiss-cpu numpy torch
```

### Ingest All Collections
```bash
python scripts/ingest_knowledge_base.py --all --test-retrieval
```

### Ingest Specific Collection
```bash
python scripts/ingest_knowledge_base.py --collection ghg_protocol_corp
python scripts/ingest_knowledge_base.py --collection technology_database
python scripts/ingest_knowledge_base.py --collection case_studies
```

### Test Retrieval Quality
```bash
python scripts/ingest_knowledge_base.py --all --test-retrieval
```

---

## Knowledge Statistics

| Collection | Documents | Estimated Chunks | Topics Covered |
|------------|-----------|------------------|----------------|
| ghg_protocol_corp | 3 | ~20-30 | Scopes, factors, methodology |
| technology_database | 3 | ~30-40 | Heat pumps, solar, CHP |
| case_studies | 1 | ~15-25 | Food, chemical, steel industries |
| **Total** | **7** | **~65-95** | **Comprehensive coverage** |

---

## Query Examples

Once ingested, agents can query the knowledge base:

```python
from greenlang.intelligence.rag.engine import RAGEngine
from greenlang.intelligence.rag.config import RAGConfig

# Create engine
config = RAGConfig(
    allowlist=["ghg_protocol_corp", "technology_database", "case_studies"]
)
engine = RAGEngine(config=config)

# Query for emission factors
result = await engine.query(
    query="What are the emission factors for natural gas combustion?",
    top_k=5,
    collections=["ghg_protocol_corp"]
)

for chunk in result.chunks:
    print(f"Relevance: {chunk.relevance_score:.2f}")
    print(f"Text: {chunk.text[:200]}...")
    print()
```

---

## Content Quality Standards

All documents in the knowledge base follow these standards:

### Accuracy
- ✅ Data sourced from authoritative references (EPA, DEFRA, IEA, WRI/WBCSD)
- ✅ Emission factors include units and sources
- ✅ Case study data represents realistic industrial scenarios

### Completeness
- ✅ Comprehensive coverage of topic
- ✅ Technical specifications included
- ✅ Economic analysis provided
- ✅ Success factors and lessons learned documented

### Structure
- ✅ Clear headings and sections
- ✅ Consistent formatting
- ✅ Examples and calculations
- ✅ Cross-references to related topics

### Metadata
- ✅ Document ID and version
- ✅ Publisher and collection
- ✅ Content hash for integrity
- ✅ Extra metadata (category, technology type, etc.)

---

## Adding New Knowledge

### Document Preparation
1. **Research**: Gather data from authoritative sources
2. **Structure**: Organize content with clear headings
3. **Review**: Validate accuracy and completeness
4. **Format**: Save as UTF-8 text file

### Ingestion Process
1. **Create document** in appropriate collection directory
2. **Generate metadata** with proper doc_id and hash
3. **Add to ingestion script** or use CLI
4. **Test retrieval** to verify quality

### Example: Adding a New Technology
```python
# 1. Create document
new_doc_content = """
Technology Name

Overview: ...
Technical Specifications: ...
Economic Analysis: ...
Applications: ...
Case Examples: ...
"""

# 2. Save to file
Path("knowledge_base/technologies/04_new_tech.txt").write_text(
    new_doc_content, encoding="utf-8"
)

# 3. Create metadata
doc_meta = DocMeta(
    doc_id="tech_new_v1",
    title="New Technology Guide",
    collection="technology_database",
    publisher="GreenLang Research",
    content_hash=file_hash("knowledge_base/technologies/04_new_tech.txt"),
    version="1.0",
)

# 4. Ingest
await engine.ingest_document(
    file_path=Path("knowledge_base/technologies/04_new_tech.txt"),
    collection="technology_database",
    doc_meta=doc_meta
)
```

---

## Knowledge Base Maintenance

### Version Control
- All documents are version-controlled
- Content hash ensures integrity
- Version field tracks document updates

### Quality Assurance
- Regular audits of emission factors (annually)
- Case study validation against published data
- Technology specifications updated with industry standards

### Expansion Roadmap
- [ ] ISO 14064 standard documentation
- [ ] SBTi target-setting guidance
- [ ] Additional technology guides (biomass, hydrogen, CCUS)
- [ ] Regional emission factor databases
- [ ] Industry-specific best practices

---

## Integration with Agents

### Current Agent Integration Status
- ❌ **Phase 1 Complete**: RAG infrastructure operational
- 🔄 **Phase 2 In Progress**: Knowledge base ingestion
- ⏳ **Phase 3 Pending**: Agent transformation to use RAG

### Future Agent Capabilities (Post-Integration)
Once agents are transformed to use RAG:

**Recommendation Agents:**
- Query technology database for suitable solutions
- Retrieve case studies for similar facilities
- Ground recommendations in proven technologies

**Calculation Agents:**
- Look up emission factors from GHG Protocol
- Validate methodology against standards
- Cite sources in audit trails

**Insight Agents:**
- Find similar historical cases
- Identify success factors
- Provide evidence-based analysis

---

## Technical Details

### Chunking Strategy
- **Chunk size**: 512 tokens (~2000 characters)
- **Overlap**: 64 tokens for context preservation
- **Method**: Character-based with token estimation

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimension**: 384
- **Normalization**: L2-normalized for cosine similarity

### Vector Store
- **Technology**: FAISS (in-memory for development)
- **Index type**: IndexFlatL2 (exact search)
- **Persistence**: Saved to knowledge_base/vector_store/

### Retrieval Method
- **Algorithm**: MMR (Maximal Marginal Relevance)
- **Lambda**: 0.5 (balance relevance and diversity)
- **Fetch K**: 20 candidates
- **Top K**: 6 final results

---

## Troubleshooting

### Common Issues

**Issue**: "Collection not in allowlist"
- **Solution**: Add collection to RAGConfig.allowlist

**Issue**: "Content hash mismatch"
- **Solution**: Regenerate hash after document modifications

**Issue**: "No results found for query"
- **Solution**: Verify documents were ingested, check collection names

**Issue**: "ImportError: sentence-transformers not installed"
- **Solution**: `pip install sentence-transformers faiss-cpu`

---

## References

### Standards and Protocols
- GHG Protocol Corporate Standard (WRI/WBCSD)
- EPA Emission Factors for Greenhouse Gas Inventories
- DEFRA Greenhouse Gas Conversion Factors
- IEA Energy Statistics

### Technology Sources
- DOE Advanced Manufacturing Office
- IEA Energy Technology Perspectives
- IRENA Renewable Energy Technologies
- Heat Pump Association Industry Reports

### Case Study Sources
- Based on industry publications and anonymized project data
- Verified against typical industry performance metrics
- Economic data normalized to represent realistic scenarios

---

**Maintained By:** GreenLang Engineering Team
**Last Updated:** 2025-11-06
**Version:** 1.0
