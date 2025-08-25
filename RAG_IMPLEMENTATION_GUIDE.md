# GreenLang RAG Implementation Guide

## Overview
GreenLang now includes a Retrieval-Augmented Generation (RAG) system that provides evidence-based answers to climate intelligence queries. The system combines vector search with Large Language Models to deliver accurate, contextual responses backed by authoritative sources.

## Architecture

### Components
1. **Document Ingestion Pipeline** (`scripts/build_vector_store.py`)
   - Loads documents from various formats (PDF, TXT, MD, JSON, CSV)
   - Splits documents into chunks for efficient retrieval
   - Creates embeddings using sentence transformers
   - Stores vectors in FAISS or ChromaDB

2. **RAG-Enhanced Assistant** (`greenlang/cli/assistant_rag.py`)
   - Searches knowledge base for relevant documents
   - Combines retrieved context with LLM responses
   - Provides source citations for transparency
   - Falls back gracefully when components unavailable

3. **Knowledge Base** (`knowledge_base/`)
   - `documents/`: Source documents for ingestion
   - `vector_store/`: Persisted vector embeddings
   - Includes climate science, emission factors, methodologies

## Setup Instructions

### 1. Install Dependencies
```bash
# Install RAG dependencies
pip install langchain langchain-community faiss-cpu sentence-transformers pypdf chromadb

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Set OpenAI API Key (Optional but Recommended)
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-api-key-here
```

### 3. Build Vector Store
```bash
# Build vector store with default climate documents
python scripts/build_vector_store.py --add-defaults

# Or specify custom paths
python scripts/build_vector_store.py \
  --docs-path knowledge_base/documents \
  --vector-path knowledge_base/vector_store \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Use ChromaDB instead of FAISS
python scripts/build_vector_store.py --use-chroma --add-defaults

# Test search functionality
python scripts/build_vector_store.py --test-search "carbon emissions"
```

### 4. Add Your Documents
Place documents in `knowledge_base/documents/`:
- **PDF**: Research papers, reports, standards
- **TXT/MD**: Documentation, guides, notes
- **JSON**: Structured data, emission factors
- **CSV**: Datasets, benchmarks

Then rebuild the vector store:
```bash
python scripts/build_vector_store.py
```

## Usage

### CLI Integration
The RAG system is automatically integrated with the GreenLang CLI:

```bash
# Ask general questions - uses RAG for context
greenlang ask "What are the emission factors for renewable energy?"

# The assistant will search the knowledge base and provide:
# - Evidence-based answer
# - Source citations
# - Specific emission factors
```

### Python API
```python
from greenlang.cli.assistant_rag import RAGAssistant

# Initialize with RAG
assistant = RAGAssistant(
    api_key="your-openai-key",  # Optional
    vector_store_path="knowledge_base/vector_store",
    use_chroma=False  # Use FAISS by default
)

# Process query with RAG
result = assistant.process_query(
    "What is the carbon intensity of solar energy?",
    use_rag=True
)

print(result["response"])
print(f"Sources: {result.get('sources', [])}")
print(f"Method: {result.get('method', 'RAG')}")

# Search knowledge base directly
docs = assistant.search_knowledge_base("building emissions", k=5)
for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...")
```

### Fallback Behavior
The system gracefully handles missing components:

1. **With OpenAI + RAG**: Full RAG-enhanced responses with LLM
2. **With RAG only**: Context retrieval without LLM synthesis
3. **With OpenAI only**: LLM responses without document context
4. **Neither**: Static knowledge base responses

## Testing

### Test RAG Implementation
```bash
# Run comprehensive RAG tests
python scripts/test_rag.py

# Test with GreenLang CLI
greenlang ask "What are Scope 1, 2, and 3 emissions?"
greenlang ask "Compare emission factors for coal vs natural gas"
greenlang ask "How do I reduce building carbon emissions?"
```

### Verify Components
```python
from greenlang.cli.assistant_rag import RAGAssistant

assistant = RAGAssistant()
print(f"LLM Available: {assistant.llm_available}")
print(f"RAG Available: {assistant.rag_available}")
```

## Knowledge Base Management

### Adding Domain-Specific Documents
1. **Climate Reports**: IPCC reports, national inventories
2. **Standards**: ISO 14064, GHG Protocol, SBTi
3. **Regional Data**: Grid emission factors by country
4. **Industry Benchmarks**: Sector-specific intensities
5. **Best Practices**: Decarbonization strategies

### Document Format Guidelines
- **Structure**: Use clear headings and sections
- **Data**: Include specific numbers and factors
- **Context**: Provide methodology explanations
- **Citations**: Include original sources

### Updating Vector Store
```bash
# Add new documents to knowledge_base/documents/
cp new_report.pdf knowledge_base/documents/

# Rebuild vector store
python scripts/build_vector_store.py

# Verify new content is searchable
python scripts/build_vector_store.py --test-search "new report topic"
```

## Configuration Options

### Embedding Models
```python
# Lightweight, fast (default)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# More accurate, slower
embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Multilingual support
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Chunk Settings
```python
# Adjust for your documents
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks
```

### Vector Store Choice
- **FAISS**: Fast, efficient, local storage (default)
- **ChromaDB**: Persistent, metadata filtering, cloud-ready

## Example Queries

### Emission Factors
```
"What is the emission factor for electricity in India?"
"Compare emission factors for different transportation fuels"
"Regional grid emission factors for renewable energy"
```

### Methodologies
```
"Explain the GHG Protocol scopes"
"How to calculate Scope 3 emissions"
"Life cycle assessment methodology for buildings"
```

### Best Practices
```
"Strategies for reducing manufacturing emissions"
"Building decarbonization roadmap"
"Carbon offset best practices"
```

### Calculations
```
"Calculate emissions for 5000 kWh solar energy"
"Carbon footprint of 1000 gallons diesel"
"Building emissions for 50,000 sqft office"
```

## Troubleshooting

### Vector Store Not Found
```bash
# Error: No existing vector store found
# Solution: Build the vector store first
python scripts/build_vector_store.py --add-defaults
```

### Import Errors
```bash
# Error: RAG components not available
# Solution: Install dependencies
pip install langchain langchain-community faiss-cpu sentence-transformers
```

### OpenAI API Issues
```bash
# Error: OpenAI API key not found
# Solution: Set in .env file
echo "OPENAI_API_KEY=your-key" > .env
```

### Memory Issues
```python
# For large document sets, use smaller batches
builder = GreenLangVectorStore(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=50  # Less overlap
)
```

## Performance Optimization

### Caching
- Vector store is cached locally
- Embeddings are reused for similar queries
- LLM responses can be cached (optional)

### Parallel Processing
- Document loading uses multiprocessing
- Batch embedding generation
- Concurrent similarity searches

### Resource Management
- Lazy loading of vector store
- Efficient memory usage with generators
- Automatic cleanup of temporary files

## Future Enhancements

### Planned Features
1. **Multi-modal RAG**: Support for images, charts, tables
2. **Dynamic Updates**: Real-time document ingestion
3. **Query Routing**: Intelligent query classification
4. **Hybrid Search**: Combine vector and keyword search
5. **Fine-tuning**: Domain-specific model adaptation
6. **Streaming**: Real-time response generation
7. **Multi-language**: Support for multiple languages
8. **Web Interface**: Browser-based knowledge exploration

### Integration Roadmap
- Connect to external databases
- API endpoints for RAG queries
- Webhook support for document updates
- Cloud vector store options
- Enterprise authentication

## API Reference

### RAGAssistant Class
```python
class RAGAssistant:
    def __init__(
        api_key: Optional[str] = None,
        vector_store_path: str = "knowledge_base/vector_store",
        use_chroma: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    def process_query(
        query: str,
        verbose: bool = False,
        use_rag: bool = True
    ) -> Dict[str, Any]
    
    def search_knowledge_base(
        query: str,
        k: int = 5
    ) -> List[Document]
    
    def get_rag_response(
        query: str,
        k: int = 5
    ) -> Tuple[str, List[str]]
```

### GreenLangVectorStore Class
```python
class GreenLangVectorStore:
    def __init__(
        docs_path: str,
        vector_store_path: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        use_chroma: bool
    )
    
    def build() -> None
    def load_documents() -> List[Document]
    def split_documents(documents: List[Document]) -> List[Document]
    def create_vector_store(documents: List[Document]) -> None
    def load_vector_store() -> Any
    def add_documents(new_documents: List[Document]) -> None
    def search(query: str, k: int) -> List[Document]
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example queries
3. Run test scripts
4. Consult API reference

## Conclusion

The RAG implementation transforms GreenLang's AI Assistant from a simple query processor to an evidence-based climate intelligence system. By combining retrieval with generation, the system provides:

- **Accuracy**: Grounded in authoritative sources
- **Transparency**: Clear source citations
- **Flexibility**: Works with or without LLM
- **Extensibility**: Easy to add new knowledge
- **Reliability**: Graceful fallback behavior

Start building your climate knowledge base today!