# Intelligence Layer Guide

## LLM Integration and RAG Systems

Guide to implementing intelligent agents with LLM orchestration and retrieval-augmented generation.

---

## LLM Integration

### Multi-Provider Setup

```python
from greenlang import LLMClient

# Initialize with multiple providers
llm = LLMClient(
    providers={
        'primary': {'provider': 'openai', 'model': 'gpt-4'},
        'fallback': {'provider': 'anthropic', 'model': 'claude-3-opus'},
        'local': {'provider': 'ollama', 'model': 'llama2'}
    },
    strategy='cost_optimized'  # or 'quality_optimized', 'latency_optimized'
)

# Use with automatic fallback
response = await llm.generate(
    prompt="Analyze carbon emissions",
    max_tokens=2000,
    temperature=0.7
)
```

### Prompt Engineering

```python
class PromptManager:
    """Structured prompt management."""

    templates = {
        'analysis': """
You are an expert carbon emissions analyst.

Context:
{context}

Task:
Analyze the following data and provide:
1. Total emissions calculation
2. Breakdown by scope
3. Key insights
4. Recommendations

Data:
{data}

Response format: JSON
""",
        'reasoning': """
Think step by step to solve this problem:

Problem: {problem}

Steps:
1. Understand the problem
2. Break it down
3. Solve each part
4. Combine results

Solution:
"""
    }

    async def render(self, template: str, **kwargs) -> str:
        """Render template with variables."""
        return self.templates[template].format(**kwargs)
```

### Token Optimization

```python
class TokenOptimizer:
    """Optimize token usage."""

    async def compress_context(self, context: str, max_tokens: int) -> str:
        """Compress context to fit token limit."""
        # Summarize if too long
        if self.count_tokens(context) > max_tokens:
            return await self.summarize(context, target_tokens=max_tokens)
        return context

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
```

---

## RAG System Implementation

### Document Processing

```python
from greenlang.rag import RAGSystem, DocumentProcessor

# Initialize RAG
rag = RAGSystem(
    vector_store='faiss',
    embedding_model='all-mpnet-base-v2',
    chunk_size=1000,
    chunk_overlap=200
)

# Process documents
processor = DocumentProcessor()

# Add documents
await rag.add_document(
    "sustainability_report.pdf",
    metadata={'year': 2024, 'type': 'report'}
)

# Query
results = await rag.query(
    "What are our Scope 3 emissions?",
    top_k=5,
    filters={'year': 2024}
)

print(f"Answer: {results.answer}")
print(f"Sources: {results.sources}")
```

### Semantic Search

```python
class SemanticSearch:
    """Semantic search implementation."""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search by semantic similarity."""
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # Search vector store
        results = await self.vector_store.search(
            vector=query_embedding,
            top_k=top_k * 2  # Get more for filtering
        )

        # Filter by threshold
        filtered = [
            r for r in results
            if r.score >= threshold
        ]

        # Rerank if needed
        if self.reranker:
            filtered = await self.reranker.rerank(query, filtered)

        return filtered[:top_k]
```

---

## Context Management

### Hierarchical Context

```python
class ContextManager:
    """Manage hierarchical context."""

    async def build_context(
        self,
        query: str,
        max_tokens: int = 8000
    ) -> str:
        """Build hierarchical context."""
        context_parts = []
        remaining_tokens = max_tokens

        # 1. Immediate context (highest priority)
        immediate = await self.get_immediate_context(query)
        context_parts.append(immediate)
        remaining_tokens -= self.count_tokens(immediate)

        # 2. Recent context
        if remaining_tokens > 0:
            recent = await self.get_recent_context(
                query,
                max_tokens=remaining_tokens // 2
            )
            context_parts.append(recent)
            remaining_tokens -= self.count_tokens(recent)

        # 3. Relevant knowledge
        if remaining_tokens > 0:
            knowledge = await self.get_relevant_knowledge(
                query,
                max_tokens=remaining_tokens
            )
            context_parts.append(knowledge)

        return "\n\n".join(context_parts)
```

---

## Embeddings

### Vector Generation

```python
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Generate embeddings."""

    def __init__(self):
        self.model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2'
        )

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings in batches."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())

        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Generate single embedding."""
        return self.model.encode([text])[0].tolist()
```

---

## Best Practices

### LLM Best Practices

1. **Use system prompts**: Define agent behavior
2. **Implement caching**: Cache repeated queries
3. **Handle rate limits**: Exponential backoff
4. **Monitor costs**: Track token usage
5. **Validate outputs**: Parse and verify responses

### RAG Best Practices

1. **Chunk documents properly**: 1000-2000 tokens
2. **Use overlap**: 200 token overlap
3. **Filter by metadata**: Improve relevance
4. **Rerank results**: Cross-encoder reranking
5. **Cite sources**: Maintain provenance

---

**Last Updated**: November 2024
**Version**: 1.0.0
