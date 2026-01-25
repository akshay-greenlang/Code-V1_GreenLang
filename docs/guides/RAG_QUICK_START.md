# GreenLang RAG System - Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
# Install all dependencies including RAG requirements
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 2. Optional: Install Neo4j for Knowledge Graph

```bash
# Using Docker (recommended)
docker run -d \
  --name greenlang-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/greenlang2024 \
  neo4j:5.0

# Or download from: https://neo4j.com/download/
```

---

## Basic Usage

### Example 1: Simple RAG Pipeline

```python
from GreenLang_2030.agent_foundation.rag import RAGSystem, create_vector_store

# Initialize RAG system
vector_store = create_vector_store("faiss", dimension=768)
rag = RAGSystem(
    vector_store=vector_store,
    confidence_threshold=0.8,  # GreenLang requirement: 80%+
    use_reranker=True,
    enable_caching=True  # 66% cost reduction
)

# Ingest documents
documents = [
    "The CSRD (Corporate Sustainability Reporting Directive) requires companies to report Scope 1, 2, and 3 emissions.",
    "Scope 1 emissions are direct emissions from owned or controlled sources.",
    "Scope 2 emissions are indirect emissions from purchased electricity, heat, or steam.",
    "Scope 3 emissions are all other indirect emissions in the value chain.",
]

num_chunks = rag.ingest_documents(documents)
print(f"Ingested {num_chunks} chunks")

# Retrieve with confidence scoring
result = rag.retrieve("What are Scope 3 emissions?", top_k=3)

print(f"Confidence: {result.confidence:.2%}")
for i, (doc, score) in enumerate(zip(result.documents, result.scores), 1):
    print(f"\n[{i}] Score: {score:.2f}")
    print(f"Content: {doc.content[:200]}...")

# Generate response with LLM (if integrated)
response = rag.generate_response(
    query="What does CSRD require for emissions reporting?",
    top_k=5,
    use_hybrid=True
)

print(f"\nAnswer: {response['answer']}")
print(f"Confidence: {response['confidence']:.2%}")
print(f"Sources used: {response['metrics']['num_sources']}")
```

### Example 2: Process PDF Documents

```python
from GreenLang_2030.agent_foundation.rag import (
    DocumentProcessor,
    ChunkingStrategy,
    EmbeddingGenerator,
    create_vector_store
)

# Initialize components
processor = DocumentProcessor(
    strategy=ChunkingStrategy.SLIDING_WINDOW,
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = EmbeddingGenerator(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_embeddings=True
)

vector_store = create_vector_store("faiss", dimension=768)

# Process PDF document
chunks = processor.process_file(
    "path/to/CSRD_regulation.pdf",
    custom_metadata={
        "source": "EU Regulation",
        "year": 2023,
        "category": "regulatory"
    }
)

print(f"Created {len(chunks)} chunks from PDF")

# Generate embeddings
chunk_embeddings = embeddings.embed_documents(chunks)

# Store in vector database
ids = vector_store.add_documents(chunks, chunk_embeddings)
print(f"Stored {len(ids)} chunks in vector database")

# Query
query_embedding = embeddings.embed_query("CSRD disclosure requirements")
docs, scores = vector_store.similarity_search(query_embedding, top_k=5)

for doc, score in zip(docs, scores):
    print(f"\nScore: {score:.2f}")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.content[:150]}...")
```

### Example 3: Hybrid Search with Multiple Strategies

```python
from GreenLang_2030.agent_foundation.rag import (
    SemanticSearch,
    KeywordSearch,
    HybridSearch,
    MMRRetrieval,
    RerankedRetrieval,
    ContextAssembler
)

# Prepare documents (already embedded)
documents = [...]  # Your Document objects
embeddings_gen = EmbeddingGenerator()
vector_store = create_vector_store("faiss", dimension=768)

# Setup retrieval strategies
semantic = SemanticSearch(vector_store, embeddings_gen)
keyword = KeywordSearch(documents, algorithm="bm25", k1=1.2, b=0.75)

# Hybrid search with Reciprocal Rank Fusion
hybrid = HybridSearch(
    semantic_strategy=semantic,
    keyword_strategy=keyword,
    alpha=0.7,  # 70% semantic, 30% keyword
    fusion_method="rrf"
)

# Add Maximum Marginal Relevance for diversity
mmr = MMRRetrieval(
    base_strategy=hybrid,
    embedding_generator=embeddings_gen,
    lambda_param=0.7  # 70% relevance, 30% diversity
)

# Add cross-encoder reranking
reranked = RerankedRetrieval(
    base_strategy=mmr,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Retrieve
result = reranked.retrieve(
    query="carbon emissions reporting requirements",
    top_k=10
)

# Assemble context for LLM
assembler = ContextAssembler(
    max_context_length=8000,
    include_confidence=True,
    source_format="numbered"
)

context = assembler.assemble_for_llm(
    documents=result.documents,
    scores=result.scores,
    query="What are the carbon emissions reporting requirements?"
)

print(context)
```

### Example 4: Knowledge Graph Integration

```python
from GreenLang_2030.agent_foundation.rag import (
    KnowledgeGraphStore,
    GraphRetrieval
)

# Initialize knowledge graph (requires Neo4j running)
kg = KnowledgeGraphStore(
    neo4j_config={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "greenlang2024",
        "database": "neo4j"
    }
)

# Ingest document with automatic entity/relationship extraction
text = """
Company XYZ is a manufacturing company that emits 50,000 tonnes CO2e annually.
They comply with the CSRD regulation and report their Scope 1, Scope 2, and
Scope 3 emissions according to the GHG Protocol. Their main emissions come from
electricity consumption and transportation.
"""

stats = kg.add_document(
    text=text,
    source="XYZ Annual Report 2023",
    metadata={"company": "XYZ", "year": 2023}
)

print(f"Entities extracted: {stats['entities_extracted']}")
print(f"Relationships created: {stats['relationships_created']}")

# Search for entities
entities = kg.search_entities(query="XYZ", entity_type="ORGANIZATION", limit=10)
for entity in entities:
    print(f"\nEntity: {entity.name} ({entity.type})")
    print(f"Confidence: {entity.confidence:.2%}")
    print(f"Properties: {entity.properties}")

# Get entity context with relationships
context = kg.get_entity_context(entity_id=entities[0].id, max_depth=2)
print(f"\nRelated entities: {len(context['related_entities'])}")
for related in context['related_entities']:
    print(f"  - {related.name} ({related.type})")

# Graph-enhanced retrieval
graph_retrieval = GraphRetrieval(
    knowledge_graph=kg,
    vector_store=vector_store,
    embedding_generator=embeddings_gen
)

results = graph_retrieval.retrieve(
    query="Companies complying with CSRD",
    top_k=10,
    use_graph=True,
    use_vector=True,
    graph_depth=2
)

print(f"\nGraph results: {len(results['graph_results'])}")
print(f"Vector results: {len(results['vector_results'])}")
print(f"Combined: {len(results['combined_results'])}")
```

### Example 5: Safe LLM Integration

```python
from GreenLang_2030.agent_foundation.rag import RAGSystem

# Initialize RAG
rag = RAGSystem(
    vector_store=create_vector_store("faiss", dimension=768),
    confidence_threshold=0.8,  # Enforce 80%+ confidence
    use_reranker=True
)

# Ingest knowledge base
rag.ingest_documents([
    "GHG Protocol Scope 3 Category 1: Purchased Goods and Services includes emissions from production of products purchased.",
    "GHG Protocol Scope 3 Category 3: Fuel and Energy Related Activities covers upstream emissions from fuel and energy.",
    "Emissions should be calculated using activity data multiplied by emission factors.",
])

# Retrieve with safety guarantees
def safe_llm_query(user_query: str):
    """Example of safe LLM integration"""

    # Retrieve relevant context
    result = rag.retrieve(user_query, top_k=5, use_hybrid=True)

    # Check confidence threshold (GreenLang requirement)
    if result.confidence < 0.8:
        return {
            "answer": "I don't have high enough confidence to answer this question accurately.",
            "confidence": result.confidence,
            "reason": "Confidence below 80% threshold"
        }

    # Generate augmented prompt with safety instructions
    augmented_prompt = rag.augment_prompt(
        query=user_query,
        context_documents=result.documents,
        max_context_length=8000
    )

    # Call your LLM (example with OpenAI)
    # response = openai_client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": augmented_prompt}]
    # )
    # answer = response.choices[0].message.content

    # For this example, just return the prompt
    return {
        "prompt": augmented_prompt,
        "confidence": result.confidence,
        "sources": [doc.content[:100] for doc in result.documents[:3]],
        "provenance_hash": result.metadata.get("provenance_hash")
    }

# Use the safe query
response = safe_llm_query("How do I calculate Scope 3 emissions?")
print(f"Confidence: {response['confidence']:.2%}")
print(f"\nPrompt:\n{response['prompt'][:500]}...")
```

---

## Monitoring & Metrics

```python
# Get system metrics
metrics = rag.get_metrics()

print(f"Total queries: {metrics['total_queries']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
print(f"Average confidence: {metrics['avg_confidence']:.2%}")
print(f"Average retrieval time: {metrics['avg_retrieval_time']:.3f}s")
print(f"Cache size: {metrics['cache_size']}")

# Clear cache if needed
if metrics['cache_size'] > 10000:
    rag.embedding_generator.clear_cache()
```

---

## Configuration Best Practices

### For Development
```python
rag = RAGSystem(
    vector_store=create_vector_store("chroma", collection_name="dev"),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Faster
    chunking_strategy=ChunkingStrategy.FIXED_SIZE,
    chunk_size=512,
    use_reranker=False,  # Faster development
    confidence_threshold=0.7,  # Lower for testing
    enable_caching=True
)
```

### For Production
```python
rag = RAGSystem(
    vector_store=create_vector_store(
        "faiss",
        dimension=768,
        index_path="./data/faiss_index.bin",
        metadata_path="./data/faiss_metadata.json"
    ),
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # Best quality
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=1000,
    chunk_overlap=200,
    use_reranker=True,  # Best relevance
    confidence_threshold=0.8,  # GreenLang requirement
    enable_caching=True,  # 66% cost reduction
)
```

### For Scale (Pinecone)
```python
vector_store = create_vector_store(
    "pinecone",
    api_key="your-pinecone-api-key",
    environment="us-west1-gcp",
    index_name="greenlang-production",
    dimension=768
)

rag = RAGSystem(
    vector_store=vector_store,
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    use_reranker=True,
    confidence_threshold=0.8,
    enable_caching=True
)
```

---

## Troubleshooting

### Issue: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Issue: FAISS index error
```python
# Rebuild index
vector_store.save()  # Save current state
vector_store = create_vector_store("faiss", dimension=768)
```

### Issue: Low confidence scores
```python
# Check retrieval strategy
result = rag.retrieve(query, top_k=10)
print(f"Strategy: {result.strategy}")
print(f"Confidence: {result.confidence:.2%}")

# Try hybrid search
result = rag.retrieve(query, top_k=10, use_hybrid=True)

# Try with reranking
rag.use_reranker = True
result = rag.retrieve(query, top_k=10, use_hybrid=True)
```

### Issue: Neo4j connection failed
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs greenlang-neo4j

# Restart
docker restart greenlang-neo4j
```

---

## Performance Tuning

### 1. Batch Processing
```python
# Ingest many documents at once
documents = load_all_documents()  # Your loader
chunks_count = rag.ingest_documents(documents)
```

### 2. GPU Acceleration
```python
# Use GPU for embeddings
embeddings = EmbeddingGenerator(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda",  # or "mps" for Apple Silicon
    cache_embeddings=True
)

# Use FAISS GPU
vector_store = create_vector_store(
    "faiss",
    dimension=768,
    use_gpu=True  # Requires faiss-gpu
)
```

### 3. Optimize Chunk Size
```python
# Smaller chunks = more precise but slower
processor = DocumentProcessor(
    strategy=ChunkingStrategy.SLIDING_WINDOW,
    chunk_size=500,
    chunk_overlap=100
)

# Larger chunks = faster but less precise
processor = DocumentProcessor(
    strategy=ChunkingStrategy.SLIDING_WINDOW,
    chunk_size=2000,
    chunk_overlap=400
)
```

---

## Next Steps

1. **Integrate with Agent Intelligence**: See `RAG_SYSTEM_IMPLEMENTATION_SUMMARY.md` for integration patterns
2. **Add Custom Embeddings**: Implement OpenAI or Cohere for better quality
3. **Deploy Knowledge Graph**: Set up Neo4j for relationship-based retrieval
4. **Monitor in Production**: Set up Prometheus metrics collection
5. **Fine-tune Models**: Train custom embeddings on your domain data

---

## Support & Documentation

- **Full Documentation**: `RAG_SYSTEM_IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `Agent_Foundation_Architecture.md` (lines 216-352)
- **Vector Stores**: `GreenLang_2030/agent_foundation/rag/vector_stores/EXAMPLES.md`
- **Tests**: `GreenLang_2030/agent_foundation/rag/test_rag_system.py`

---

**Last Updated:** 2025-11-15
**Version:** 1.0
